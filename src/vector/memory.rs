//! 内存向量存储：brute-force 扫描，仅供开发/测试或小数据量使用。

use super::*;
use crate::error::DbError;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

struct Collection {
    dim: usize,
    distance: Distance,
    records: HashMap<String, Record>,
}

#[derive(Default)]
pub struct MemoryVectorStore {
    inner: Arc<RwLock<HashMap<String, Collection>>>,
}

impl MemoryVectorStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl VectorStore for MemoryVectorStore {
    async fn ensure_collection(&self, spec: CollectionSpec<'_>) -> Result<()> {
        let mut g = self.inner.write();
        g.entry(spec.name.to_string())
            .or_insert_with(|| Collection {
                dim: spec.dim,
                distance: spec.distance,
                records: HashMap::new(),
            });
        Ok(())
    }

    async fn drop_collection(&self, name: &str) -> Result<()> {
        self.inner.write().remove(name);
        Ok(())
    }

    async fn upsert(&self, collection: &str, records: &[Record]) -> Result<()> {
        let mut g = self.inner.write();
        let c = g
            .get_mut(collection)
            .ok_or_else(|| DbError::Other(format!("collection `{collection}` not found")))?;
        for r in records {
            if r.vector.len() != c.dim {
                return Err(DbError::Other(format!(
                    "dim mismatch: expected {}, got {}",
                    c.dim,
                    r.vector.len()
                )));
            }
            c.records.insert(r.id.clone(), r.clone());
        }
        Ok(())
    }

    async fn delete(&self, collection: &str, ids: &[String]) -> Result<()> {
        let mut g = self.inner.write();
        if let Some(c) = g.get_mut(collection) {
            for id in ids {
                c.records.remove(id);
            }
        }
        Ok(())
    }

    async fn search(&self, collection: &str, query: Query<'_>) -> Result<Vec<Match>> {
        let g = self.inner.read();
        let c = g
            .get(collection)
            .ok_or_else(|| DbError::Other(format!("collection `{collection}` not found")))?;
        if query.vector.len() != c.dim {
            return Err(DbError::Other("query dim mismatch".into()));
        }

        let distance = c.distance;
        let qvec = query.vector;
        let filter = query.filter;
        let top_k = query.top_k.max(1);

        // 1. 过滤 + 计分；启用 vector-parallel 时用 rayon 并行
        let scored: Vec<(f32, &Record)> = {
            let iter = c.records.values().filter(|r| match filter {
                None => true,
                Some(f) => f.iter().all(|(k, v)| r.metadata.get(k) == Some(v)),
            });

            #[cfg(feature = "vector-parallel")]
            {
                use rayon::prelude::*;
                let v: Vec<&Record> = iter.collect();
                v.into_par_iter()
                    .map(|r| (score(distance, qvec, &r.vector), r))
                    .collect()
            }
            #[cfg(not(feature = "vector-parallel"))]
            {
                iter.map(|r| (score(distance, qvec, &r.vector), r))
                    .collect()
            }
        };

        // 2. top-k 用 BinaryHeap 求最大 k 个，O(n log k) 而非 O(n log n)
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        // 维护一个最小堆（堆顶是当前 top-k 中最小分），超过 k 就 pop
        let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::with_capacity(top_k + 1);
        for (sc, r) in scored {
            heap.push(Reverse(HeapEntry {
                score: sc,
                id: r.id.clone(),
                record: r,
            }));
            if heap.len() > top_k {
                heap.pop();
            }
        }

        // 3. 输出按分数降序
        let mut hits: Vec<Match> = heap
            .into_iter()
            .map(|Reverse(e)| Match {
                id: e.id,
                score: e.score,
                metadata: e.record.metadata.clone(),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(hits)
    }

    fn backend_name(&self) -> &'static str {
        "memory"
    }
}

struct HeapEntry<'a> {
    score: f32,
    id: String,
    record: &'a Record,
}

impl<'a> PartialEq for HeapEntry<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl<'a> Eq for HeapEntry<'a> {}
impl<'a> PartialOrd for HeapEntry<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<'a> Ord for HeapEntry<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn score(d: Distance, a: &[f32], b: &[f32]) -> f32 {
    match d {
        Distance::Cosine => cosine_similarity(a, b),
        Distance::Dot => a.iter().zip(b).map(|(x, y)| x * y).sum(),
        // 转成相似度：距离越小越好 -> 取负
        Distance::L2 => {
            let s: f32 = a
                .iter()
                .zip(b)
                .map(|(x, y)| {
                    let d = x - y;
                    d * d
                })
                .sum();
            -s.sqrt()
        }
    }
}

// ===========================================================================
// 测试（无需外部依赖，始终可运行）
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> MemoryVectorStore {
        MemoryVectorStore::new()
    }

    async fn setup(store: &MemoryVectorStore, name: &str, dim: usize, distance: Distance) {
        store
            .ensure_collection(CollectionSpec {
                name,
                dim,
                distance,
            })
            .await
            .unwrap();
    }

    // ── 集合管理 ───────────────────────────────────────

    #[tokio::test]
    async fn test_ensure_collection() {
        let s = store();
        s.ensure_collection(CollectionSpec {
            name: "coll",
            dim: 3,
            distance: Distance::Cosine,
        })
        .await
        .unwrap();
        // 幂等
        s.ensure_collection(CollectionSpec {
            name: "coll",
            dim: 3,
            distance: Distance::Cosine,
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_drop_collection() {
        let s = store();
        setup(&s, "dropped", 2, Distance::L2).await;
        s.drop_collection("dropped").await.unwrap();
        // 删除后 upsert 应该报错
        assert!(
            s.upsert("dropped", &[record("x", vec![0.0, 0.0])])
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_drop_nonexistent_noop() {
        let s = store();
        s.drop_collection("ghost").await.unwrap();
    }

    // ── Upsert & Search ────────────────────────────────

    fn record(id: &str, vector: Vec<f32>) -> Record {
        Record {
            id: id.into(),
            vector,
            metadata: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_upsert_and_search_cosine() {
        let s = store();
        let coll = "cs";
        setup(&s, coll, 3, Distance::Cosine).await;

        s.upsert(
            coll,
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
                coll,
                Query {
                    vector: &[1.0, 0.0, 0.0],
                    top_k: 2,
                    filter: None,
                },
            )
            .await
            .unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "a", "cosine: closest to [1,0,0] should be a");
        assert!(hits[0].score > hits[1].score);
    }

    #[tokio::test]
    async fn test_search_l2_distance() {
        let s = store();
        let coll = "l2";
        setup(&s, coll, 2, Distance::L2).await;

        s.upsert(
            coll,
            &[
                record("near", vec![0.0, 0.0]),
                record("far", vec![10.0, 10.0]),
            ],
        )
        .await
        .unwrap();

        let hits = s
            .search(
                coll,
                Query {
                    vector: &[0.0, 0.0],
                    top_k: 2,
                    filter: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(hits[0].id, "near");
        assert!(
            hits[0].score > hits[1].score,
            "L2: near should score higher"
        );
    }

    #[tokio::test]
    async fn test_search_dot_product() {
        let s = store();
        let coll = "dot";
        setup(&s, coll, 2, Distance::Dot).await;

        s.upsert(
            coll,
            &[
                record("aligned", vec![2.0, 0.0]),
                record("opposite", vec![-2.0, 0.0]),
            ],
        )
        .await
        .unwrap();

        let hits = s
            .search(
                coll,
                Query {
                    vector: &[1.0, 0.0],
                    top_k: 2,
                    filter: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(hits[0].id, "aligned");
    }

    // ── Metadata 过滤 ──────────────────────────────────

    #[tokio::test]
    async fn test_search_with_filter() {
        let s = store();
        let coll = "filt";
        setup(&s, coll, 2, Distance::Cosine).await;

        let mut meta = std::collections::HashMap::new();
        meta.insert("tag".into(), serde_json::json!("keep"));
        s.upsert(
            coll,
            &[
                Record {
                    id: "keep".into(),
                    vector: vec![1.0, 0.0],
                    metadata: meta.clone(),
                },
                record("skip", vec![0.99, 0.01]),
            ],
        )
        .await
        .unwrap();

        let mut filter = std::collections::HashMap::new();
        filter.insert("tag".into(), serde_json::json!("keep"));

        let hits = s
            .search(
                coll,
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
    }

    // ── Delete ─────────────────────────────────────────

    #[tokio::test]
    async fn test_delete_records() {
        let s = store();
        let coll = "del";
        setup(&s, coll, 2, Distance::Cosine).await;

        s.upsert(
            coll,
            &[record("a", vec![1.0, 0.0]), record("b", vec![0.0, 1.0])],
        )
        .await
        .unwrap();
        s.delete(coll, &["a".into()]).await.unwrap();

        let hits = s
            .search(
                coll,
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
    }

    #[tokio::test]
    async fn test_delete_empty_noop() {
        let s = store();
        setup(&s, "edel", 2, Distance::Cosine).await;
        s.delete("edel", &[]).await.unwrap();
    }

    #[tokio::test]
    async fn test_upsert_empty_noop() {
        let s = store();
        setup(&s, "eup", 2, Distance::Cosine).await;
        s.upsert("eup", &[]).await.unwrap();
    }

    // ── 维度校验 ──────────────────────────────────────

    #[tokio::test]
    async fn test_upsert_dimension_mismatch() {
        let s = store();
        setup(&s, "dim", 3, Distance::Cosine).await;
        let r = s.upsert("dim", &[record("x", vec![1.0, 2.0])]).await;
        assert!(r.is_err(), "dimension mismatch should error");
    }

    #[tokio::test]
    async fn test_search_dimension_mismatch() {
        let s = store();
        setup(&s, "dim2", 3, Distance::Cosine).await;
        let r = s
            .search(
                "dim2",
                Query {
                    vector: &[1.0, 2.0],
                    top_k: 1,
                    filter: None,
                },
            )
            .await;
        assert!(r.is_err(), "dimension mismatch should error");
    }

    // ── Upsert 覆盖 ────────────────────────────────────

    #[tokio::test]
    async fn test_upsert_overwrites() {
        let s = store();
        let coll = "over";
        setup(&s, coll, 2, Distance::Cosine).await;

        s.upsert(coll, &[record("r1", vec![1.0, 0.0])])
            .await
            .unwrap();

        let mut meta = std::collections::HashMap::new();
        meta.insert("v".into(), serde_json::json!(2));
        s.upsert(
            coll,
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
                coll,
                Query {
                    vector: &[0.0, 1.0],
                    top_k: 1,
                    filter: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(hits[0].id, "r1");
        assert_eq!(hits[0].metadata.get("v").and_then(|v| v.as_i64()), Some(2));
    }

    // ── 后端名 ──────────────────────────────────────────

    #[test]
    fn test_backend_name() {
        assert_eq!(store().backend_name(), "memory");
    }

    // ── Top-K = 1 边界 ─────────────────────────────────

    #[tokio::test]
    async fn test_search_top_k_one() {
        let s = store();
        let coll = "top1";
        setup(&s, coll, 2, Distance::Cosine).await;
        s.upsert(coll, &[record("only", vec![1.0, 0.0])])
            .await
            .unwrap();

        let hits = s
            .search(
                coll,
                Query {
                    vector: &[1.0, 0.0],
                    top_k: 1,
                    filter: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
    }
}
