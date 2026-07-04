//! Qdrant Edge 嵌入式向量存储后端。
//!
//! Qdrant Edge 是进程内的嵌入库（类似 SQLite），无需独立服务、无需网络连接。
//!
//! ## 实现要点
//!
//! - 每个 collection → 独立子目录 + `EdgeShard` 实例
//! - String ID ↔ u64 自增计数器映射（保证往返一致性）
//! - metadata 缓存在 ShardEntry 层，search 时原样返回

use super::*;
use crate::error::DbError;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use qdrant_edge::{
    EdgeShard, EdgeConfigBuilder, EdgeVectorParams,
    UpdateOperation, PointOperations, PointInsertOperations,
    QueryRequest, ScoringQuery, QueryEnum,
    PointId, PointStruct, Vectors, Vector,
    VectorInternal,
    Filter, Condition, FieldCondition, Match as EdgeMatch, MatchValue, ValueVariants,
    JsonPath,
    WithPayloadInterface, WithVector,
    NamedQuery,
    DEFAULT_VECTOR_NAME,
};

type Metadata = HashMap<String, serde_json::Value>;

struct ShardEntry {
    shard: EdgeShard,
    dim: usize,
    distance: Distance,
    next_id: u64,
    /// String ID → u64 映射
    id_map: HashMap<String, u64>,
    /// u64 → String ID 反向映射
    rev_map: HashMap<u64, String>,
    /// u64 → metadata 缓存（edge segment types 无法在 search 结果中直接返回 serde_json::Value）
    meta_cache: HashMap<u64, Metadata>,
}

pub struct QdrantEdgeVectorStore {
    base_dir: PathBuf,
    shards: std::sync::Mutex<HashMap<String, Arc<std::sync::Mutex<ShardEntry>>>>,
}

impl QdrantEdgeVectorStore {
    pub fn new(data_dir: impl Into<PathBuf>) -> Result<Self> {
        let base_dir = data_dir.into();
        std::fs::create_dir_all(&base_dir)
            .map_err(|e| DbError::Other(format!("qdrant-edge dir: {e}")))?;
        Ok(Self { base_dir, shards: Default::default() })
    }

    fn coll_dir(&self, name: &str) -> PathBuf { self.base_dir.join(name) }

    /// 仅查找已有 shard，不自动创建。
    fn get(&self, name: &str) -> Option<Arc<std::sync::Mutex<ShardEntry>>> {
        self.shards.lock().unwrap().get(name).cloned()
    }

    /// 查找或创建 shard（用于 ensure_collection）。
    fn get_or_create(
        &self, name: &str, dim: usize, distance: Distance,
    ) -> Result<Arc<std::sync::Mutex<ShardEntry>>> {
        if let Some(s) = self.get(name) {
            let entry = s.lock().unwrap();
            if entry.dim != dim || entry.distance != distance {
                return Err(DbError::Other(format!(
                    "collection `{name}` exists with dim={} distance={:?}, \
                     requested dim={dim} distance={distance:?}",
                    entry.dim, entry.distance,
                )));
            }
            return Ok(s.clone());
        }

        let mut w = self.shards.lock().unwrap();
        if let Some(s) = w.get(name) {
            return Ok(s.clone());
        }

        let dir = self.coll_dir(name);
        std::fs::create_dir_all(&dir)
            .map_err(|e| DbError::Other(format!("qdrant-edge mkdir: {e}")))?;

        let edge_dist = to_edge_distance(distance);
        let config = EdgeConfigBuilder::new()
            .vector(DEFAULT_VECTOR_NAME, EdgeVectorParams {
                size: dim, distance: edge_dist,
                on_disk: Some(false), multivector_config: None,
                datatype: None, quantization_config: None, hnsw_config: None,
            })
            .build();

        let shard = if dir.join("segments").exists() {
            EdgeShard::load(&dir, Some(config))
        } else {
            EdgeShard::new(&dir, config)
        }
        .map_err(|e| DbError::Other(format!("qdrant-edge {name}: {e}")))?;

        let entry = Arc::new(std::sync::Mutex::new(ShardEntry {
            shard, dim, distance, next_id: 1,
            id_map: HashMap::new(), rev_map: HashMap::new(),
            meta_cache: HashMap::new(),
        }));
        w.insert(name.to_string(), entry.clone());
        Ok(entry)
    }

    fn remove_shard(&self, name: &str) {
        self.shards.lock().unwrap().remove(name);
        let _ = std::fs::remove_dir_all(self.coll_dir(name));
    }
}

// ── 距离转换 ────────────────────────────────────────────

fn to_edge_distance(d: Distance) -> qdrant_edge::Distance {
    match d {
        Distance::Cosine => qdrant_edge::Distance::Cosine,
        Distance::L2 => qdrant_edge::Distance::Euclid,
        Distance::Dot => qdrant_edge::Distance::Dot,
    }
}

// ── ID 映射 + metadata ──────────────────────────────────

fn assign_id(entry: &mut ShardEntry, str_id: &str, metadata: &Metadata) -> u64 {
    if let Some(&n) = entry.id_map.get(str_id) {
        // 更新 metadata 缓存（upsert 覆盖）
        entry.meta_cache.insert(n, metadata.clone());
        return n;
    }
    let n = entry.next_id;
    entry.next_id += 1;
    entry.id_map.insert(str_id.to_string(), n);
    entry.rev_map.insert(n, str_id.to_string());
    entry.meta_cache.insert(n, metadata.clone());
    n
}

fn lookup_id(entry: &ShardEntry, num_id: u64) -> String {
    entry.rev_map.get(&num_id).cloned().unwrap_or_else(|| num_id.to_string())
}

fn lookup_meta(entry: &ShardEntry, num_id: u64) -> Metadata {
    entry.meta_cache.get(&num_id).cloned().unwrap_or_default()
}

// ── 构造 PointStruct ───────────────────────────────────

fn record_to_point(entry: &mut ShardEntry, r: &Record) -> PointStruct {
    let num_id = assign_id(entry, &r.id, &r.metadata);
    let payload: serde_json::Value = if r.metadata.is_empty() {
        serde_json::Value::Object(Default::default())
    } else {
        serde_json::Value::Object(r.metadata.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    };
    let vectors = Vectors::new_named(vec![(DEFAULT_VECTOR_NAME, Vector::new_dense(r.vector.clone()))]);
    PointStruct::new(num_id, vectors, payload)
}

// ── Filter ────────────────────────────────────────────

fn build_filter(f: &HashMap<String, serde_json::Value>) -> Option<Filter> {
    let conditions: Vec<Condition> = f.iter().filter_map(|(k, v)| {
        let mv = match v {
            serde_json::Value::String(s) => MatchValue { value: ValueVariants::String(s.clone()) },
            serde_json::Value::Bool(b) => MatchValue { value: ValueVariants::Bool(*b) },
            serde_json::Value::Number(n) if n.is_i64() =>
                MatchValue { value: ValueVariants::Integer(n.as_i64().unwrap()) },
            _ => return None,
        };
        Some(Condition::Field(FieldCondition {
            key: k.as_str().parse().expect("valid json path"),
            r#match: Some(EdgeMatch::Value(mv)),
            range: None, geo_bounding_box: None, geo_radius: None,
            values_count: None, geo_polygon: None,
            is_empty: None, is_null: None,
        }))
    }).collect();

    if conditions.is_empty() { None }
    else {
        Some(Filter {
            should: None, must: Some(conditions), must_not: None, min_should: None,
        })
    }
}

// ===========================================================================
// VectorStore
// ===========================================================================

#[async_trait]
impl VectorStore for QdrantEdgeVectorStore {
    async fn ensure_collection(&self, spec: CollectionSpec<'_>) -> Result<()> {
        self.get_or_create(spec.name, spec.dim, spec.distance)?;
        Ok(())
    }

    async fn drop_collection(&self, name: &str) -> Result<()> {
        self.remove_shard(name);
        Ok(())
    }

    async fn upsert(&self, collection: &str, records: &[Record]) -> Result<()> {
        if records.is_empty() { return Ok(()); }

        let dim = records[0].vector.len();

        // 已有 collection → 只验证维度；不存在 → 以默认 Cosine 创建
        let s: Arc<std::sync::Mutex<ShardEntry>> = match self.get(collection) {
            Some(s) => {
                let entry = s.lock().unwrap();
                if entry.dim != dim {
                    return Err(DbError::Other(format!(
                        "dim mismatch: `{collection}` has dim={}, got {dim}", entry.dim,
                    )));
                }
                drop(entry);
                s
            }
            None => self.get_or_create(collection, dim, Distance::Cosine)?,
        };

        let mut entry = s.lock().unwrap();
        let points: Vec<_> = records.iter()
            .map(|r| record_to_point(&mut entry, r).into())
            .collect();

        entry.shard.update(
            UpdateOperation::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperations::PointsList(points)),
            ),
        ).map_err(|e| DbError::Other(format!("qdrant-edge upsert: {e}")))?;

        entry.shard.flush();
        Ok(())
    }

    async fn delete(&self, collection: &str, ids: &[String]) -> Result<()> {
        if ids.is_empty() { return Ok(()); }

        let Some(s) = self.get(collection) else { return Ok(()) };
        let mut entry = s.lock().unwrap();

        let point_ids: Vec<PointId> = ids.iter()
            .filter_map(|id| entry.id_map.get(id).copied().map(PointId::NumId))
            .collect();
        if point_ids.is_empty() { return Ok(()); }

        entry.shard.update(
            UpdateOperation::PointOperation(
                PointOperations::DeletePoints { ids: point_ids },
            ),
        ).map_err(|e| DbError::Other(format!("qdrant-edge delete: {e}")))?;

        // 清理映射
        for id in ids {
            if let Some(&n) = entry.id_map.get(id) {
                entry.rev_map.remove(&n);
                entry.id_map.remove(id);
                entry.meta_cache.remove(&n);
            }
        }
        Ok(())
    }

    async fn search(&self, collection: &str, query: Query<'_>) -> Result<Vec<Match>> {
        let Some(s) = self.get(collection) else {
            return Err(DbError::Other(format!("collection `{collection}` not found")));
        };
        let entry = s.lock().unwrap();

        let filter = query.filter.and_then(build_filter);
        let qvec = VectorInternal::from(query.vector.to_vec());

        let req = QueryRequest {
            prefetches: vec![],
            query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery {
                query: qvec,
                using: Some(DEFAULT_VECTOR_NAME.to_string()),
            }))),
            filter, score_threshold: None,
            limit: query.top_k, offset: 0, params: None,
            with_payload: WithPayloadInterface::Bool(false), // 不需要 segment payload
            with_vector: WithVector::Bool(false),
        };

        let results = entry.shard.query(req)
            .map_err(|e| DbError::Other(format!("qdrant-edge query: {e}")))?;

        let dist_fn: fn(f32) -> f32 = match entry.distance {
            Distance::Cosine => |d| 1.0 - d,
            Distance::L2 => |d| -d,
            Distance::Dot => |d| d,
        };

        let matches: Vec<Match> = results.into_iter().map(|sp| {
            let num_id = match sp.id {
                PointId::NumId(n) => n,
                _ => 0,
            };
            Match {
                id: lookup_id(&entry, num_id),
                score: dist_fn(sp.score),
                metadata: lookup_meta(&entry, num_id),
            }
        }).collect();

        Ok(matches)
    }

    fn backend_name(&self) -> &'static str { "qdrant-edge" }
}

// ===========================================================================
// 测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils;

    fn store() -> QdrantEdgeVectorStore {
        let dir = std::env::temp_dir()
            .join("uwu_edge_test")
            .join(test_utils::unique_prefix());
        QdrantEdgeVectorStore::new(&dir).unwrap()
    }

    fn rec(id: &str, v: Vec<f32>) -> Record {
        Record { id: id.into(), vector: v, metadata: Default::default() }
    }

    async fn setup(s: &QdrantEdgeVectorStore, name: &str, dim: usize) {
        let _ = s.drop_collection(name).await;
        s.ensure_collection(CollectionSpec { name, dim, distance: Distance::Cosine }).await.unwrap();
    }

    #[tokio::test]
    async fn test_ensure_and_drop() {
        let s = store();
        s.ensure_collection(CollectionSpec { name: "c1", dim: 3, distance: Distance::Cosine }).await.unwrap();
        // 幂等
        s.ensure_collection(CollectionSpec { name: "c1", dim: 3, distance: Distance::Cosine }).await.unwrap();
        s.drop_collection("c1").await.unwrap();
    }

    #[tokio::test]
    async fn test_dimension_mismatch_rejected() {
        let s = store();
        s.ensure_collection(CollectionSpec { name: "dm", dim: 3, distance: Distance::Cosine }).await.unwrap();
        let r = s.ensure_collection(CollectionSpec { name: "dm", dim: 5, distance: Distance::Cosine }).await;
        assert!(r.is_err(), "dimension mismatch should be rejected");
    }

    #[tokio::test]
    async fn test_upsert_and_search() {
        let s = store();
        setup(&s, "cs", 3).await;
        s.upsert("cs", &[
            Record { id: "a".into(), vector: vec![1.0, 0.0, 0.0], metadata: Default::default() },
            Record { id: "b".into(), vector: vec![0.0, 1.0, 0.0], metadata: Default::default() },
            Record { id: "c".into(), vector: vec![1.0, 1.0, 0.0], metadata: Default::default() },
        ]).await.unwrap();

        let hits = s.search("cs", Query {
            vector: &[1.0, 0.0, 0.0], top_k: 2, filter: None,
        }).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "a");
        s.drop_collection("cs").await.unwrap();
    }

    #[tokio::test]
    async fn test_metadata_roundtrip() {
        let s = store();
        setup(&s, "meta", 2).await;

        let mut meta = HashMap::new();
        meta.insert("v".into(), serde_json::json!(42));
        meta.insert("label".into(), serde_json::json!("hello"));
        s.upsert("meta", &[Record {
            id: "m1".into(), vector: vec![1.0, 0.0], metadata: meta,
        }]).await.unwrap();

        let hits = s.search("meta", Query {
            vector: &[1.0, 0.0], top_k: 1, filter: None,
        }).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].metadata.get("v").and_then(|v| v.as_i64()), Some(42));
        assert_eq!(hits[0].metadata.get("label").and_then(|v| v.as_str()), Some("hello"));

        s.drop_collection("meta").await.unwrap();
    }

    #[tokio::test]
    async fn test_upsert_updates_metadata() {
        let s = store();
        setup(&s, "updm", 2).await;

        let mut m1 = HashMap::new();
        m1.insert("v".into(), serde_json::json!(1));
        s.upsert("updm", &[Record { id: "r1".into(), vector: vec![1.0, 0.0], metadata: m1 }]).await.unwrap();

        let mut m2 = HashMap::new();
        m2.insert("v".into(), serde_json::json!(99));
        s.upsert("updm", &[Record { id: "r1".into(), vector: vec![0.0, 1.0], metadata: m2 }]).await.unwrap();

        let hits = s.search("updm", Query {
            vector: &[0.0, 1.0], top_k: 1, filter: None,
        }).await.unwrap();
        assert_eq!(hits[0].metadata.get("v").and_then(|v| v.as_i64()), Some(99));

        s.drop_collection("updm").await.unwrap();
    }

    #[tokio::test]
    async fn test_search_with_filter() {
        let s = store();
        setup(&s, "filt", 2).await;

        let mut ma = HashMap::new();
        ma.insert("tag".into(), serde_json::json!("keep"));
        let mut mb = HashMap::new();
        mb.insert("tag".into(), serde_json::json!("skip"));

        s.upsert("filt", &[
            Record { id: "keep".into(), vector: vec![1.0, 0.0], metadata: ma },
            Record { id: "skip".into(), vector: vec![0.99, 0.01], metadata: mb },
        ]).await.unwrap();

        let mut filter = HashMap::new();
        filter.insert("tag".into(), serde_json::json!("keep"));
        let hits = s.search("filt", Query {
            vector: &[1.0, 0.0], top_k: 5, filter: Some(&filter),
        }).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "keep");
        s.drop_collection("filt").await.unwrap();
    }

    #[tokio::test]
    async fn test_delete_records() {
        let s = store();
        setup(&s, "del", 2).await;
        s.upsert("del", &[rec("a", vec![1.0, 0.0]), rec("b", vec![0.0, 1.0])]).await.unwrap();
        s.delete("del", &["a".into()]).await.unwrap();

        let hits = s.search("del", Query {
            vector: &[1.0, 0.0], top_k: 5, filter: None,
        }).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "b");
        s.drop_collection("del").await.unwrap();
    }

    #[tokio::test]
    async fn test_empty_ops_noop() {
        let s = store();
        setup(&s, "empty", 2).await;
        s.upsert("empty", &[]).await.unwrap();
        s.delete("empty", &[]).await.unwrap();
    }

    #[tokio::test]
    async fn test_l2_distance() {
        let s = store();
        let _ = s.drop_collection("l2").await;
        s.ensure_collection(CollectionSpec { name: "l2", dim: 2, distance: Distance::L2 }).await.unwrap();
        s.upsert("l2", &[rec("near", vec![0.0, 0.0]), rec("far", vec![10.0, 10.0])]).await.unwrap();

        let hits = s.search("l2", Query {
            vector: &[0.0, 0.0], top_k: 2, filter: None,
        }).await.unwrap();
        assert!(hits[0].score > hits[1].score, "L2: nearer should score higher");
        s.drop_collection("l2").await.unwrap();
    }

    #[test]
    fn test_backend_name() {
        assert_eq!(store().backend_name(), "qdrant-edge");
    }

    #[tokio::test]
    async fn test_search_nonexistent_collection_error() {
        let s = store();
        assert!(s.search("ghost", Query { vector: &[1.0], top_k: 1, filter: None }).await.is_err());
    }

    #[tokio::test]
    async fn test_id_roundtrip() {
        let s = store();
        setup(&s, "idrt", 2).await;
        s.upsert("idrt", &[rec("my-custom-id-123", vec![1.0, 0.0])]).await.unwrap();

        let hits = s.search("idrt", Query {
            vector: &[1.0, 0.0], top_k: 1, filter: None,
        }).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "my-custom-id-123");

        s.delete("idrt", &["my-custom-id-123".into()]).await.unwrap();
        let after = s.search("idrt", Query {
            vector: &[1.0, 0.0], top_k: 1, filter: None,
        }).await.unwrap();
        assert!(after.is_empty());
        s.drop_collection("idrt").await.unwrap();
    }
}