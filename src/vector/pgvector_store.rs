//! pgvector 后端：复用现有 PostgreSQL Pool。
//!
//! 表结构（由 [`PgVectorStore`] 自动建立）：
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS <name> (
//!   id        TEXT PRIMARY KEY,
//!   embedding vector(<dim>) NOT NULL,
//!   metadata  JSONB NOT NULL DEFAULT '{}'::jsonb
//! );
//! ```
//!
//! 性能优化点：
//! - `upsert` 使用单条多值 `INSERT ... VALUES (...), (...) ...`，按 batch 切分（每批 1000 条），
//!   避免 round-trip。
//! - 支持按集合配置 `Distance`（cosine / l2 / dot），search 时用对应运算符。
//! - 提供 [`PgVectorStore::create_hnsw_index`] / [`create_ivfflat_index`] 显式建索引。

use super::*;
use crate::error::DbError;
use crate::sql::DbPool;
use parking_lot::RwLock;
use sqlx::Row;
use std::collections::HashMap;
use std::sync::Arc;

pub struct PgVectorStore {
    pool: Arc<DbPool>,
    /// collection -> distance（建表时记录，search 自动选择运算符）
    distances: Arc<RwLock<HashMap<String, Distance>>>,
}

impl PgVectorStore {
    pub fn new(pool: Arc<DbPool>) -> Result<Self> {
        pool.as_postgres()?;
        Ok(Self {
            pool,
            distances: Default::default(),
        })
    }

    fn pg(&self) -> &sqlx::PgPool {
        self.pool.as_postgres().expect("postgres pool")
    }

    fn distance_of(&self, collection: &str) -> Distance {
        self.distances
            .read()
            .get(collection)
            .copied()
            .unwrap_or(Distance::Cosine)
    }

    /// 创建 HNSW 索引（pgvector 0.5+）。
    pub async fn create_hnsw_index(
        &self,
        collection: &str,
        m: u32,
        ef_construction: u32,
    ) -> Result<()> {
        validate_ident(collection)?;
        let op = index_op(self.distance_of(collection));
        let sql = format!(
            "CREATE INDEX IF NOT EXISTS {collection}_hnsw \
             ON {collection} USING hnsw (embedding {op}) \
             WITH (m = {m}, ef_construction = {ef_construction})"
        );
        sqlx::query(&sql).execute(self.pg()).await?;
        Ok(())
    }

    /// 创建 IVFFlat 索引（pgvector 0.4+）。
    pub async fn create_ivfflat_index(&self, collection: &str, lists: u32) -> Result<()> {
        validate_ident(collection)?;
        let op = index_op(self.distance_of(collection));
        let sql = format!(
            "CREATE INDEX IF NOT EXISTS {collection}_ivf \
             ON {collection} USING ivfflat (embedding {op}) WITH (lists = {lists})"
        );
        sqlx::query(&sql).execute(self.pg()).await?;
        Ok(())
    }
}

fn op_str(d: Distance) -> &'static str {
    match d {
        Distance::Cosine => "<=>",
        Distance::L2 => "<->",
        Distance::Dot => "<#>",
    }
}

fn index_op(d: Distance) -> &'static str {
    match d {
        Distance::Cosine => "vector_cosine_ops",
        Distance::L2 => "vector_l2_ops",
        Distance::Dot => "vector_ip_ops",
    }
}

fn vector_literal(v: &[f32]) -> String {
    let mut s = String::with_capacity(v.len() * 8 + 2);
    s.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&x.to_string());
    }
    s.push(']');
    s
}

const UPSERT_BATCH: usize = 1000;

#[async_trait]
impl VectorStore for PgVectorStore {
    async fn ensure_collection(&self, spec: CollectionSpec<'_>) -> Result<()> {
        validate_ident(spec.name)?;
        let sql = format!(
            "CREATE TABLE IF NOT EXISTS {name} (\
                id TEXT PRIMARY KEY, \
                embedding vector({dim}) NOT NULL, \
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb)",
            name = spec.name,
            dim = spec.dim,
        );
        sqlx::query(&sql).execute(self.pg()).await?;
        self.distances
            .write()
            .insert(spec.name.to_string(), spec.distance);
        Ok(())
    }

    async fn drop_collection(&self, name: &str) -> Result<()> {
        validate_ident(name)?;
        let sql = format!("DROP TABLE IF EXISTS {name}");
        sqlx::query(&sql).execute(self.pg()).await?;
        self.distances.write().remove(name);
        Ok(())
    }

    /// 批量 upsert：按 1000 条/批拼成多值 INSERT，单事务提交。
    async fn upsert(&self, collection: &str, records: &[Record]) -> Result<()> {
        validate_ident(collection)?;
        if records.is_empty() {
            return Ok(());
        }

        let mut tx = self.pg().begin().await?;
        for chunk in records.chunks(UPSERT_BATCH) {
            // 构造 $1,$2,$3 / $4,$5,$6 / ...
            let mut placeholders = String::new();
            for i in 0..chunk.len() {
                if i > 0 {
                    placeholders.push(',');
                }
                let base = i * 3;
                placeholders.push_str(&format!(
                    "(${},${}::vector,${}::jsonb)",
                    base + 1,
                    base + 2,
                    base + 3
                ));
            }
            let sql = format!(
                "INSERT INTO {collection} (id, embedding, metadata) VALUES {placeholders} \
                 ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata"
            );
            let mut q = sqlx::query(&sql);
            for r in chunk {
                let meta = serde_json::Value::Object(
                    r.metadata
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                );
                q = q
                    .bind(r.id.clone())
                    .bind(vector_literal(&r.vector))
                    .bind(meta);
            }
            q.execute(&mut *tx).await?;
        }
        tx.commit().await?;
        Ok(())
    }

    async fn delete(&self, collection: &str, ids: &[String]) -> Result<()> {
        validate_ident(collection)?;
        if ids.is_empty() {
            return Ok(());
        }
        let sql = format!("DELETE FROM {collection} WHERE id = ANY($1)");
        sqlx::query(&sql).bind(ids).execute(self.pg()).await?;
        Ok(())
    }

    async fn search(&self, collection: &str, query: Query<'_>) -> Result<Vec<Match>> {
        validate_ident(collection)?;
        let dist = self.distance_of(collection);
        let op = op_str(dist);
        let (filter_sql, filter_value) = match query.filter {
            Some(f) if !f.is_empty() => (
                "WHERE metadata @> $2".to_string(),
                Some(serde_json::Value::Object(f.clone().into_iter().collect())),
            ),
            _ => (String::new(), None),
        };
        let sql = format!(
            "SELECT id, metadata, embedding {op} $1::vector AS distance \
             FROM {collection} {filter_sql} \
             ORDER BY embedding {op} $1::vector ASC \
             LIMIT {limit}",
            limit = query.top_k as i64,
        );
        let mut q = sqlx::query(&sql).bind(vector_literal(query.vector));
        if let Some(v) = filter_value {
            q = q.bind(v);
        }
        let rows = q.fetch_all(self.pg()).await?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let id: String = row.try_get("id")?;
            let meta: serde_json::Value = row.try_get("metadata").unwrap_or(serde_json::json!({}));
            let dist_val: f64 = row.try_get("distance")?;
            // 距离 -> 相似度分数
            let score = match dist {
                Distance::Cosine => 1.0 - dist_val as f32, // cosine distance ∈ [0,2]，score ∈ [-1,1]
                Distance::L2 => -(dist_val as f32),        // 越小越相似 -> 取负
                Distance::Dot => -(dist_val as f32),       // pgvector <#> 返回 -inner_product
            };
            let metadata = match meta {
                serde_json::Value::Object(m) => m.into_iter().collect(),
                _ => Default::default(),
            };
            out.push(Match {
                id,
                score,
                metadata,
            });
        }
        Ok(out)
    }

    fn backend_name(&self) -> &'static str {
        "pgvector"
    }
}

fn validate_ident(s: &str) -> Result<()> {
    if s.is_empty() || s.len() > 63 {
        return Err(DbError::Other(format!("invalid identifier `{s}`")));
    }
    if !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(DbError::Other(format!("invalid identifier `{s}`")));
    }
    Ok(())
}

// ===========================================================================
// PG 集成测试（需要 pgvector 扩展）
// ===========================================================================

#[cfg(test)]
mod pg_tests {
    use super::*;
    use crate::test_utils;

    fn require_pg() -> String {
        test_utils::pg_url().expect("SKIP: DATABASE_URL not set")
    }

    async fn setup_vector_store() -> (PgVectorStore, String) {
        let _url = require_pg();
        let pool = test_utils::pg_pool().await.unwrap();
        let arc_pool = std::sync::Arc::new(pool);

        // 创建 pgvector 扩展（需超级用户权限）
        let pg = arc_pool.as_postgres().unwrap();
        let _ = sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(pg)
            .await;

        let store = PgVectorStore::new(arc_pool).unwrap();
        let prefix = test_utils::unique_prefix();
        (store, prefix)
    }

    // ── 集合管理 ───────────────────────────────────────

    #[tokio::test]
    async fn test_ensure_and_drop_collection() {
        let (store, prefix) = setup_vector_store().await;
        let coll = format!("{prefix}_coll");

        let spec = CollectionSpec {
            name: &coll,
            dim: 3,
            distance: Distance::Cosine,
        };
        store.ensure_collection(spec).await.unwrap();

        // 确保 idempotent
        store.ensure_collection(spec).await.unwrap();

        // 验证表存在
        let pg = store.pg();
        let (exists,): (bool,) = sqlx::query_as(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
        )
        .bind(&coll)
        .fetch_one(pg)
        .await
        .unwrap();
        assert!(exists, "collection table should exist");

        // 删除
        store.drop_collection(&coll).await.unwrap();

        let (exists,): (bool,) = sqlx::query_as(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
        )
        .bind(&coll)
        .fetch_one(pg)
        .await
        .unwrap();
        assert!(!exists, "collection table should be dropped");
    }

    #[tokio::test]
    async fn test_drop_nonexistent_collection() {
        let (store, _prefix) = setup_vector_store().await;
        // 不应报错
        store
            .drop_collection("nonexistent_collection_12345")
            .await
            .unwrap();
    }

    // ── Upsert & Search ────────────────────────────────

    #[tokio::test]
    async fn test_upsert_and_search_cosine() {
        let (store, prefix) = setup_vector_store().await;
        let coll = format!("{prefix}_vs");

        store
            .ensure_collection(CollectionSpec {
                name: &coll,
                dim: 3,
                distance: Distance::Cosine,
            })
            .await
            .unwrap();

        // 写入 3 条记录
        let records = vec![
            Record {
                id: "a".into(),
                vector: vec![1.0, 0.0, 0.0],
                metadata: [("label".into(), serde_json::json!("x"))].into(),
            },
            Record {
                id: "b".into(),
                vector: vec![0.0, 1.0, 0.0],
                metadata: [("label".into(), serde_json::json!("y"))].into(),
            },
            Record {
                id: "c".into(),
                vector: vec![1.0, 1.0, 0.0],
                metadata: [("label".into(), serde_json::json!("x"))].into(),
            },
        ];
        store.upsert(&coll, &records).await.unwrap();

        // 查询最接近 [1.0, 0.0, 0.0] 的向量
        let results = store
            .search(
                &coll,
                Query {
                    vector: &[1.0, 0.0, 0.0],
                    top_k: 2,
                    filter: None,
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a", "a should be closest to [1,0,0]");
        assert!(
            results[0].score > results[1].score,
            "results should be sorted by score desc"
        );

        store.drop_collection(&coll).await.unwrap();
    }

    #[tokio::test]
    async fn test_upsert_updates_existing_record() {
        let (store, prefix) = setup_vector_store().await;
        let coll = format!("{prefix}_upd");

        store
            .ensure_collection(CollectionSpec {
                name: &coll,
                dim: 2,
                distance: Distance::Cosine,
            })
            .await
            .unwrap();

        // 首次插入
        store
            .upsert(
                &coll,
                &[Record {
                    id: "r1".into(),
                    vector: vec![1.0, 0.0],
                    metadata: Default::default(),
                }],
            )
            .await
            .unwrap();

        // 覆盖更新
        store
            .upsert(
                &coll,
                &[Record {
                    id: "r1".into(),
                    vector: vec![0.0, 1.0],
                    metadata: [("v".into(), serde_json::json!(2))].into(),
                }],
            )
            .await
            .unwrap();

        let results = store
            .search(
                &coll,
                Query {
                    vector: &[0.0, 1.0],
                    top_k: 1,
                    filter: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "r1");
        assert_eq!(
            results[0].metadata.get("v").and_then(|v| v.as_i64()),
            Some(2)
        );

        store.drop_collection(&coll).await.unwrap();
    }

    #[tokio::test]
    async fn test_search_with_metadata_filter() {
        let (store, prefix) = setup_vector_store().await;
        let coll = format!("{prefix}_filt");

        store
            .ensure_collection(CollectionSpec {
                name: &coll,
                dim: 2,
                distance: Distance::Cosine,
            })
            .await
            .unwrap();

        store
            .upsert(
                &coll,
                &[
                    Record {
                        id: "keep".into(),
                        vector: vec![1.0, 0.0],
                        metadata: [("active".into(), serde_json::json!(true))].into(),
                    },
                    Record {
                        id: "skip".into(),
                        vector: vec![0.99, 0.01],
                        metadata: [("active".into(), serde_json::json!(false))].into(),
                    },
                ],
            )
            .await
            .unwrap();

        let mut filter = std::collections::HashMap::new();
        filter.insert("active".into(), serde_json::json!(true));

        let results = store
            .search(
                &coll,
                Query {
                    vector: &[1.0, 0.0],
                    top_k: 5,
                    filter: Some(&filter),
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1, "only active=true should match");
        assert_eq!(results[0].id, "keep");

        store.drop_collection(&coll).await.unwrap();
    }

    #[tokio::test]
    async fn test_search_l2_distance() {
        let (store, prefix) = setup_vector_store().await;
        let coll = format!("{prefix}_l2");

        store
            .ensure_collection(CollectionSpec {
                name: &coll,
                dim: 2,
                distance: Distance::L2,
            })
            .await
            .unwrap();

        store
            .upsert(
                &coll,
                &[
                    Record {
                        id: "near".into(),
                        vector: vec![0.0, 0.0],
                        metadata: Default::default(),
                    },
                    Record {
                        id: "far".into(),
                        vector: vec![10.0, 10.0],
                        metadata: Default::default(),
                    },
                ],
            )
            .await
            .unwrap();

        let results = store
            .search(
                &coll,
                Query {
                    vector: &[0.0, 0.0],
                    top_k: 2,
                    filter: None,
                },
            )
            .await
            .unwrap();

        assert_eq!(results[0].id, "near");
        assert!(
            results[0].score > results[1].score,
            "L2: nearer should have higher similarity score"
        );

        store.drop_collection(&coll).await.unwrap();
    }

    #[tokio::test]
    async fn test_search_dot_product() {
        let (store, prefix) = setup_vector_store().await;
        let coll = format!("{prefix}_dot");

        store
            .ensure_collection(CollectionSpec {
                name: &coll,
                dim: 2,
                distance: Distance::Dot,
            })
            .await
            .unwrap();

        store
            .upsert(
                &coll,
                &[
                    Record {
                        id: "aligned".into(),
                        vector: vec![2.0, 0.0],
                        metadata: Default::default(),
                    },
                    Record {
                        id: "opposite".into(),
                        vector: vec![-2.0, 0.0],
                        metadata: Default::default(),
                    },
                ],
            )
            .await
            .unwrap();

        let results = store
            .search(
                &coll,
                Query {
                    vector: &[1.0, 0.0],
                    top_k: 2,
                    filter: None,
                },
            )
            .await
            .unwrap();

        assert_eq!(results[0].id, "aligned");
        store.drop_collection(&coll).await.unwrap();
    }

    // ── Delete ─────────────────────────────────────────

    #[tokio::test]
    async fn test_delete_vectors() {
        let (store, prefix) = setup_vector_store().await;
        let coll = format!("{prefix}_del");

        store
            .ensure_collection(CollectionSpec {
                name: &coll,
                dim: 2,
                distance: Distance::Cosine,
            })
            .await
            .unwrap();

        store
            .upsert(
                &coll,
                &[
                    Record {
                        id: "d1".into(),
                        vector: vec![1.0, 0.0],
                        metadata: Default::default(),
                    },
                    Record {
                        id: "d2".into(),
                        vector: vec![0.0, 1.0],
                        metadata: Default::default(),
                    },
                ],
            )
            .await
            .unwrap();

        // 删除一条
        store.delete(&coll, &["d1".into()]).await.unwrap();

        let results = store
            .search(
                &coll,
                Query {
                    vector: &[1.0, 0.0],
                    top_k: 5,
                    filter: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "d2");

        store.drop_collection(&coll).await.unwrap();
    }

    #[tokio::test]
    async fn test_delete_empty_ids_noop() {
        let (store, prefix) = setup_vector_store().await;
        let coll = format!("{prefix}_edel");

        store
            .ensure_collection(CollectionSpec {
                name: &coll,
                dim: 2,
                distance: Distance::Cosine,
            })
            .await
            .unwrap();

        // 空列表不应报错
        store.delete(&coll, &[]).await.unwrap();

        store.drop_collection(&coll).await.unwrap();
    }

    #[tokio::test]
    async fn test_upsert_empty_records_noop() {
        let (store, prefix) = setup_vector_store().await;
        let coll = format!("{prefix}_eup");

        store
            .ensure_collection(CollectionSpec {
                name: &coll,
                dim: 2,
                distance: Distance::Cosine,
            })
            .await
            .unwrap();

        store.upsert(&coll, &[]).await.unwrap();

        store.drop_collection(&coll).await.unwrap();
    }

    // ── 索引 ───────────────────────────────────────────

    #[tokio::test]
    async fn test_create_hnsw_index() {
        let (store, prefix) = setup_vector_store().await;
        let coll = format!("{prefix}_hnsw");

        store
            .ensure_collection(CollectionSpec {
                name: &coll,
                dim: 3,
                distance: Distance::Cosine,
            })
            .await
            .unwrap();

        // 插入一些数据再建索引
        store
            .upsert(
                &coll,
                &[Record {
                    id: "i1".into(),
                    vector: vec![1.0, 0.0, 0.0],
                    metadata: Default::default(),
                }],
            )
            .await
            .unwrap();

        store.create_hnsw_index(&coll, 16, 200).await.unwrap();

        // 建完索引后仍可正常查询
        let results = store
            .search(
                &coll,
                Query {
                    vector: &[1.0, 0.0, 0.0],
                    top_k: 1,
                    filter: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);

        store.drop_collection(&coll).await.unwrap();
    }

    #[tokio::test]
    async fn test_create_ivfflat_index() {
        let (store, prefix) = setup_vector_store().await;
        let coll = format!("{prefix}_ivf");

        store
            .ensure_collection(CollectionSpec {
                name: &coll,
                dim: 4,
                distance: Distance::L2,
            })
            .await
            .unwrap();

        store
            .upsert(
                &coll,
                &[
                    Record {
                        id: "j1".into(),
                        vector: vec![0.0, 0.0, 0.0, 0.0],
                        metadata: Default::default(),
                    },
                    Record {
                        id: "j2".into(),
                        vector: vec![1.0, 1.0, 1.0, 1.0],
                        metadata: Default::default(),
                    },
                ],
            )
            .await
            .unwrap();

        store.create_ivfflat_index(&coll, 1).await.unwrap();

        let results = store
            .search(
                &coll,
                Query {
                    vector: &[0.0, 0.0, 0.0, 0.0],
                    top_k: 1,
                    filter: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);

        store.drop_collection(&coll).await.unwrap();
    }

    // ── 标识符验证 ──────────────────────────────────────

    #[tokio::test]
    async fn test_validate_ident_rejects_invalid() {
        let (store, _prefix) = setup_vector_store().await;

        // 空名称
        let r = store
            .ensure_collection(CollectionSpec {
                name: "",
                dim: 2,
                distance: Distance::Cosine,
            })
            .await;
        assert!(r.is_err(), "empty name should be rejected");

        // 特殊字符
        let r = store
            .ensure_collection(CollectionSpec {
                name: "bad;drop--table",
                dim: 2,
                distance: Distance::Cosine,
            })
            .await;
        assert!(r.is_err(), "special chars should be rejected");
    }

    // ── 后端名 ──────────────────────────────────────────

    #[test]
    fn test_backend_name() {
        let _url = require_pg();
        // 需要通过 async 上下文创建，但这个测试只验证静态返回值
        // backend_name 在构造后即可调用
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (store, _prefix) = setup_vector_store().await;
            assert_eq!(store.backend_name(), "pgvector");
        });
    }
}
