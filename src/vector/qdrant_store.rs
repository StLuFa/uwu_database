//! Qdrant 后端，使用官方 `qdrant-client` (gRPC)。

use super::*;
use crate::error::DbError;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance as QDistance, Filter,
    PointId, PointStruct, PointsIdsList, SearchPointsBuilder, UpsertPointsBuilder, Value,
    VectorParamsBuilder, value::Kind,
};
use qdrant_client::{Payload, Qdrant};

pub struct QdrantVectorStore {
    client: Qdrant,
}

impl QdrantVectorStore {
    pub fn new(url: &str, api_key: Option<String>) -> Result<Self> {
        let mut b = Qdrant::from_url(url);
        if let Some(k) = api_key {
            b = b.api_key(k);
        }
        let client = b.build().map_err(|e| DbError::Other(e.to_string()))?;
        Ok(Self { client })
    }

    pub fn from_client(client: Qdrant) -> Self {
        Self { client }
    }
}

fn to_qdistance(d: Distance) -> QDistance {
    match d {
        Distance::Cosine => QDistance::Cosine,
        Distance::L2 => QDistance::Euclid,
        Distance::Dot => QDistance::Dot,
    }
}

fn to_payload(meta: &std::collections::HashMap<String, serde_json::Value>) -> Payload {
    let obj: serde_json::Map<String, serde_json::Value> =
        meta.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    Payload::try_from(serde_json::Value::Object(obj)).unwrap_or_default()
}

fn json_value_from_qdrant(v: Value) -> serde_json::Value {
    match v.kind {
        Some(Kind::NullValue(_)) | None => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(b),
        Some(Kind::IntegerValue(i)) => serde_json::Value::from(i),
        Some(Kind::DoubleValue(d)) => serde_json::Number::from_f64(d)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s),
        Some(Kind::ListValue(l)) => {
            serde_json::Value::Array(l.values.into_iter().map(json_value_from_qdrant).collect())
        }
        Some(Kind::StructValue(s)) => serde_json::Value::Object(
            s.fields
                .into_iter()
                .map(|(k, v)| (k, json_value_from_qdrant(v)))
                .collect(),
        ),
    }
}

fn from_payload(
    p: std::collections::HashMap<String, Value>,
) -> std::collections::HashMap<String, serde_json::Value> {
    p.into_iter()
        .map(|(k, v)| (k, json_value_from_qdrant(v)))
        .collect()
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn ensure_collection(&self, spec: CollectionSpec<'_>) -> Result<()> {
        if let Ok(true) = self.client.collection_exists(spec.name).await {
            return Ok(());
        }
        self.client
            .create_collection(CreateCollectionBuilder::new(spec.name).vectors_config(
                VectorParamsBuilder::new(spec.dim as u64, to_qdistance(spec.distance)),
            ))
            .await
            .map_err(|e| DbError::Other(e.to_string()))?;
        Ok(())
    }

    async fn drop_collection(&self, name: &str) -> Result<()> {
        self.client
            .delete_collection(name)
            .await
            .map_err(|e| DbError::Other(e.to_string()))?;
        Ok(())
    }

    async fn upsert(&self, collection: &str, records: &[Record]) -> Result<()> {
        let points: Vec<PointStruct> = records
            .iter()
            .map(|r| PointStruct::new(r.id.clone(), r.vector.clone(), to_payload(&r.metadata)))
            .collect();
        self.client
            .upsert_points(UpsertPointsBuilder::new(collection, points))
            .await
            .map_err(|e| DbError::Other(e.to_string()))?;
        Ok(())
    }

    async fn delete(&self, collection: &str, ids: &[String]) -> Result<()> {
        let point_ids: Vec<PointId> = ids.iter().cloned().map(PointId::from).collect();
        let list = PointsIdsList { ids: point_ids };
        self.client
            .delete_points(DeletePointsBuilder::new(collection).points(list))
            .await
            .map_err(|e| DbError::Other(e.to_string()))?;
        Ok(())
    }

    async fn search(&self, collection: &str, query: Query<'_>) -> Result<Vec<Match>> {
        let mut req =
            SearchPointsBuilder::new(collection, query.vector.to_vec(), query.top_k as u64)
                .with_payload(true);
        if let Some(f) = query.filter {
            if !f.is_empty() {
                let conds: Vec<Condition> = f
                    .iter()
                    .filter_map(|(k, v)| match v {
                        serde_json::Value::String(s) => {
                            Some(Condition::matches(k.as_str(), s.clone()))
                        }
                        serde_json::Value::Bool(b) => Some(Condition::matches(k.as_str(), *b)),
                        serde_json::Value::Number(n) => {
                            n.as_i64().map(|i| Condition::matches(k.as_str(), i))
                        }
                        _ => None,
                    })
                    .collect();
                if !conds.is_empty() {
                    req = req.filter(Filter::must(conds));
                }
            }
        }
        let resp = self
            .client
            .search_points(req)
            .await
            .map_err(|e| DbError::Other(e.to_string()))?;

        let out = resp
            .result
            .into_iter()
            .map(|p| {
                let id =
                    p.id.map(|pid| match pid.point_id_options {
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u)) => u,
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => {
                            n.to_string()
                        }
                        None => String::new(),
                    })
                    .unwrap_or_default();
                Match {
                    id,
                    score: p.score,
                    metadata: from_payload(p.payload),
                }
            })
            .collect();
        Ok(out)
    }

    fn backend_name(&self) -> &'static str {
        "qdrant"
    }
}

#[allow(dead_code)]
fn _ensure_selector_compiles() {
    let _: Option<PointsIdsList> = None;
}

// ===========================================================================
// Qdrant 集成测试
// ===========================================================================

#[cfg(all(test, feature = "vector-qdrant"))]
mod qdrant_tests {
    use super::*;
    use crate::test_utils;

    fn require_qdrant() -> String {
        test_utils::qdrant_url().expect("SKIP: QDRANT_URL not set")
    }

    fn store() -> QdrantVectorStore {
        let url = require_qdrant();
        QdrantVectorStore::new(&url, None).expect("qdrant connect")
    }

    fn record(id: &str, vector: Vec<f32>) -> Record {
        Record {
            id: id.into(),
            vector,
            metadata: Default::default(),
        }
    }

    async fn setup_coll(store: &QdrantVectorStore, name: &str, dim: usize) {
        // 确保干净起点
        let _ = store.drop_collection(name).await;
        store
            .ensure_collection(CollectionSpec {
                name,
                dim,
                distance: Distance::Cosine,
            })
            .await
            .unwrap();
    }

    // ── 集合管理 ───────────────────────────────────────

    #[tokio::test]
    async fn test_ensure_and_drop_collection() {
        let s = store();
        let coll = test_utils::unique_prefix();

        s.ensure_collection(CollectionSpec {
            name: &coll,
            dim: 3,
            distance: Distance::Cosine,
        })
        .await
        .unwrap();
        // 幂等
        s.ensure_collection(CollectionSpec {
            name: &coll,
            dim: 3,
            distance: Distance::Cosine,
        })
        .await
        .unwrap();

        s.drop_collection(&coll).await.unwrap();
    }

    // ── Upsert & Search ────────────────────────────────

    #[tokio::test]
    async fn test_upsert_and_search() {
        let s = store();
        let coll = test_utils::unique_prefix();
        setup_coll(&s, &coll, 3).await;

        s.upsert(
            &coll,
            &[
                Record {
                    id: "a".into(),
                    vector: vec![1.0, 0.0, 0.0],
                    metadata: Default::default(),
                },
                Record {
                    id: "b".into(),
                    vector: vec![0.0, 1.0, 0.0],
                    metadata: Default::default(),
                },
                Record {
                    id: "c".into(),
                    vector: vec![1.0, 1.0, 0.0],
                    metadata: Default::default(),
                },
            ],
        )
        .await
        .unwrap();

        let hits = s
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

        assert_eq!(hits.len(), 2);
        assert!(
            hits[0].score > hits[1].score,
            "qdrant: results sorted by score"
        );

        s.drop_collection(&coll).await.unwrap();
    }

    #[tokio::test]
    async fn test_upsert_overwrites() {
        let s = store();
        let coll = test_utils::unique_prefix();
        setup_coll(&s, &coll, 2).await;

        s.upsert(&coll, &[record("r1", vec![1.0, 0.0])])
            .await
            .unwrap();

        let mut meta = std::collections::HashMap::new();
        meta.insert("v".into(), serde_json::json!(99));
        s.upsert(
            &coll,
            &[Record {
                id: "r1".into(),
                vector: vec![0.0, 1.0],
                metadata: meta.clone(),
            }],
        )
        .await
        .unwrap();

        let hits = s
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
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "r1");
        assert_eq!(hits[0].metadata.get("v").and_then(|v| v.as_i64()), Some(99));

        s.drop_collection(&coll).await.unwrap();
    }

    // ── 过滤搜索 ──────────────────────────────────────

    #[tokio::test]
    async fn test_search_with_string_filter() {
        let s = store();
        let coll = test_utils::unique_prefix();
        setup_coll(&s, &coll, 2).await;

        let mut meta_a = std::collections::HashMap::new();
        meta_a.insert("tag".into(), serde_json::json!("keep"));
        let mut meta_b = std::collections::HashMap::new();
        meta_b.insert("tag".into(), serde_json::json!("skip"));

        s.upsert(
            &coll,
            &[
                Record {
                    id: "keep".into(),
                    vector: vec![1.0, 0.0],
                    metadata: meta_a,
                },
                Record {
                    id: "skip".into(),
                    vector: vec![0.99, 0.01],
                    metadata: meta_b,
                },
            ],
        )
        .await
        .unwrap();

        let mut filter = std::collections::HashMap::new();
        filter.insert("tag".into(), serde_json::json!("keep"));

        let hits = s
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
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "keep");

        s.drop_collection(&coll).await.unwrap();
    }

    #[tokio::test]
    async fn test_search_with_bool_filter() {
        let s = store();
        let coll = test_utils::unique_prefix();
        setup_coll(&s, &coll, 2).await;

        let mut meta = std::collections::HashMap::new();
        meta.insert("active".into(), serde_json::json!(true));
        s.upsert(
            &coll,
            &[
                Record {
                    id: "yes".into(),
                    vector: vec![1.0, 0.0],
                    metadata: meta,
                },
                Record {
                    id: "no".into(),
                    vector: vec![0.99, 0.01],
                    metadata: Default::default(),
                },
            ],
        )
        .await
        .unwrap();

        let mut filter = std::collections::HashMap::new();
        filter.insert("active".into(), serde_json::json!(true));
        let hits = s
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
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "yes");

        s.drop_collection(&coll).await.unwrap();
    }

    // ── 删除 ───────────────────────────────────────────

    #[tokio::test]
    async fn test_delete_records() {
        let s = store();
        let coll = test_utils::unique_prefix();
        setup_coll(&s, &coll, 2).await;

        s.upsert(
            &coll,
            &[record("a", vec![1.0, 0.0]), record("b", vec![0.0, 1.0])],
        )
        .await
        .unwrap();
        s.delete(&coll, &["a".into()]).await.unwrap();

        let hits = s
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
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "b");

        s.drop_collection(&coll).await.unwrap();
    }

    #[tokio::test]
    async fn test_delete_empty_noop() {
        let s = store();
        let coll = test_utils::unique_prefix();
        setup_coll(&s, &coll, 2).await;
        s.delete(&coll, &[]).await.unwrap();
        s.drop_collection(&coll).await.unwrap();
    }

    // ── 空操作 ─────────────────────────────────────────

    #[tokio::test]
    async fn test_upsert_empty_noop() {
        let s = store();
        let coll = test_utils::unique_prefix();
        setup_coll(&s, &coll, 2).await;
        s.upsert(&coll, &[]).await.unwrap();
        s.drop_collection(&coll).await.unwrap();
    }

    // ── 后端名 ─────────────────────────────────────────

    #[test]
    fn test_backend_name() {
        let s = store();
        assert_eq!(s.backend_name(), "qdrant");
    }

    // ── L2 和 Dot 距离 ────────────────────────────────

    #[tokio::test]
    async fn test_search_l2_distance() {
        let s = store();
        let coll = test_utils::unique_prefix();
        let _ = s.drop_collection(&coll).await;
        s.ensure_collection(CollectionSpec {
            name: &coll,
            dim: 2,
            distance: Distance::L2,
        })
        .await
        .unwrap();

        s.upsert(
            &coll,
            &[
                record("near", vec![0.0, 0.0]),
                record("far", vec![10.0, 10.0]),
            ],
        )
        .await
        .unwrap();

        let hits = s
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
        assert!(hits[0].score > hits[1].score);
        s.drop_collection(&coll).await.unwrap();
    }
}
