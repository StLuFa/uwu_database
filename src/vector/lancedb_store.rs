//! LanceDB 后端：嵌入式列存向量数据库。
//!
//! 表结构（每个 collection 独立）：
//! - `id`         : Utf8
//! - `vector`     : FixedSizeList<Float32, dim>
//! - `metadata`   : Utf8（JSON 字符串，避免动态 schema 复杂度）
//!
//! Upsert 语义：通过 `merge_insert` 以 `id` 为主键合并，等效于 ON CONFLICT DO UPDATE。
//! Filter 下推：metadata 过滤在 Rust 侧执行（LanceDB OSS 版暂不支持 JSON 字段下推）。

use super::*;
use crate::error::DbError;
use arrow_array::{
    Array, FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray,
    builder::{FixedSizeListBuilder, Float32Builder},
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, DistanceType};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub struct LanceDbVectorStore {
    conn: Connection,
    /// collection -> distance（search 时正确计算 score）
    distances: Arc<RwLock<HashMap<String, Distance>>>,
}

impl LanceDbVectorStore {
    pub async fn open(uri: &str) -> Result<Self> {
        let conn = lancedb::connect(uri).execute().await
            .map_err(|e| DbError::Other(e.to_string()))?;
        Ok(Self { conn, distances: Default::default() })
    }

    fn distance_of(&self, collection: &str) -> Distance {
        self.distances.read().get(collection).copied().unwrap_or(Distance::Cosine)
    }

    /// 在指定集合的 `vector` 列上创建 IVF_PQ 索引。
    /// `num_partitions` 推荐 `sqrt(N)`，`num_sub_vectors` 必须能整除 dim。
    pub async fn create_ivf_pq_index(
        &self,
        collection: &str,
        distance: Distance,
        num_partitions: u32,
        num_sub_vectors: u32,
    ) -> Result<()> {
        let table = self.conn.open_table(collection).execute().await
            .map_err(|e| DbError::Other(e.to_string()))?;
        let builder = IvfPqIndexBuilder::default()
            .distance_type(to_distance_type(distance))
            .num_partitions(num_partitions)
            .num_sub_vectors(num_sub_vectors);
        table.create_index(&["vector"], Index::IvfPq(builder))
            .execute().await
            .map_err(|e| DbError::Other(e.to_string()))?;
        Ok(())
    }
}

fn schema_for(dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
        Field::new("metadata", DataType::Utf8, false),
    ]))
}

fn to_distance_type(d: Distance) -> DistanceType {
    match d {
        Distance::Cosine => DistanceType::Cosine,
        Distance::L2 => DistanceType::L2,
        Distance::Dot => DistanceType::Dot,
    }
}

fn dist_to_score(d: Distance, dist: f32) -> f32 {
    match d {
        Distance::Cosine => 1.0 - dist,  // cosine distance ∈ [0,2], score ∈ [-1,1]
        Distance::L2 => -dist,           // 越小越好 -> 取负
        Distance::Dot => -dist,          // LanceDB Dot 返回 -dot
    }
}

fn build_batch(records: &[Record], dim: usize) -> Result<RecordBatch> {
    let schema = schema_for(dim);

    let ids: Vec<&str> = records.iter().map(|r| r.id.as_str()).collect();
    let id_arr = StringArray::from(ids);

    let mut vec_builder = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    for r in records {
        if r.vector.len() != dim {
            return Err(DbError::Other(format!(
                "dim mismatch: expected {dim}, got {}", r.vector.len())));
        }
        for v in &r.vector {
            vec_builder.values().append_value(*v);
        }
        vec_builder.append(true);
    }
    let vec_arr: FixedSizeListArray = vec_builder.finish();

    let metas: Vec<String> = records.iter()
        .map(|r| serde_json::to_string(&r.metadata).unwrap_or_else(|_| "{}".into()))
        .collect();
    let meta_arr = StringArray::from(metas);

    RecordBatch::try_new(
        schema,
        vec![Arc::new(id_arr), Arc::new(vec_arr), Arc::new(meta_arr)],
    ).map_err(|e| DbError::Other(e.to_string()))
}

#[async_trait]
impl VectorStore for LanceDbVectorStore {
    async fn ensure_collection(&self, spec: CollectionSpec<'_>) -> Result<()> {
        self.distances.write().insert(spec.name.to_string(), spec.distance);

        let names = self.conn.table_names().execute().await
            .map_err(|e| DbError::Other(e.to_string()))?;
        if names.iter().any(|n| n == spec.name) {
            return Ok(());
        }
        let schema = schema_for(spec.dim);
        let empty = RecordBatch::new_empty(schema.clone());
        let iter = RecordBatchIterator::new(vec![Ok(empty)], schema);
        self.conn.create_table(spec.name, iter).execute().await
            .map_err(|e| DbError::Other(e.to_string()))?;
        Ok(())
    }

    async fn drop_collection(&self, name: &str) -> Result<()> {
        self.distances.write().remove(name);
        self.conn.drop_table(name).await
            .map_err(|e| DbError::Other(e.to_string()))?;
        Ok(())
    }

    /// 真正的 upsert：以 `id` 列为 key 通过 `merge_insert` 合并写入。
    /// 与旧的 `add`（累积重复行）不同，此实现保证幂等。
    async fn upsert(&self, collection: &str, records: &[Record]) -> Result<()> {
        if records.is_empty() { return Ok(()); }
        let dim = records[0].vector.len();

        let table = self.conn.open_table(collection).execute().await
            .map_err(|e| DbError::Other(e.to_string()))?;

        let batch = build_batch(records, dim)?;
        let schema = batch.schema();
        let iter = RecordBatchIterator::new(vec![Ok(batch)], schema);

        // merge_insert: 匹配 id 则更新，否则插入
        table
            .merge_insert(&["id"])
            .when_matched_update_all(None)
            .when_not_matched_insert_all()
            .execute(Box::new(iter))
            .await
            .map_err(|e| DbError::Other(e.to_string()))?;
        Ok(())
    }

    async fn delete(&self, collection: &str, ids: &[String]) -> Result<()> {
        if ids.is_empty() { return Ok(()); }
        let table = self.conn.open_table(collection).execute().await
            .map_err(|e| DbError::Other(e.to_string()))?;
        let quoted: Vec<String> = ids.iter()
            .map(|s| format!("'{}'", s.replace('\'', "''")))
            .collect();
        let predicate = format!("id IN ({})", quoted.join(","));
        table.delete(&predicate).await
            .map_err(|e| DbError::Other(e.to_string()))?;
        Ok(())
    }

    async fn search(&self, collection: &str, query: Query<'_>) -> Result<Vec<Match>> {
        let dist = self.distance_of(collection);

        let table = self.conn.open_table(collection).execute().await
            .map_err(|e| DbError::Other(e.to_string()))?;

        // 请求比 top_k 多一些，以应对客户端过滤后数量不足
        let fetch_k = if query.filter.map_or(false, |f| !f.is_empty()) {
            query.top_k * 4
        } else {
            query.top_k
        };

        let stream = table
            .vector_search(query.vector.to_vec())
            .map_err(|e| DbError::Other(e.to_string()))?
            .distance_type(to_distance_type(dist))
            .limit(fetch_k)
            .execute()
            .await
            .map_err(|e| DbError::Other(e.to_string()))?;

        let batches: Vec<RecordBatch> = stream.try_collect().await
            .map_err(|e| DbError::Other(e.to_string()))?;

        let mut out = Vec::new();
        'outer: for batch in batches {
            let id_col = batch.column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| DbError::Other("missing id column".into()))?;
            let meta_col = batch.column_by_name("metadata")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| DbError::Other("missing metadata column".into()))?;
            let dist_col = batch.column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>())
                .ok_or_else(|| DbError::Other("missing _distance column".into()))?;

            for i in 0..batch.num_rows() {
                if out.len() >= query.top_k { break 'outer; }

                let id = id_col.value(i).to_string();
                let raw_dist = dist_col.value(i);
                let meta_str = meta_col.value(i);
                let metadata: HashMap<String, serde_json::Value> =
                    serde_json::from_str(meta_str).unwrap_or_default();

                // 客户端 metadata 过滤
                if let Some(f) = query.filter {
                    if !f.is_empty() && !f.iter().all(|(k, v)| metadata.get(k) == Some(v)) {
                        continue;
                    }
                }

                out.push(Match {
                    id,
                    score: dist_to_score(dist, raw_dist),
                    metadata,
                });
            }
        }
        Ok(out)
    }

    fn backend_name(&self) -> &'static str { "lancedb" }
}
