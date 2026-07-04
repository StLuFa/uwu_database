//! 向量数据库统一抽象。
//!
//! 通过 [`VectorStore`] trait 屏蔽不同后端，支持的实现由 Cargo features 决定：
//! - `vector-memory`：纯内存 brute-force（开发/测试）
//! - `vector-pgvector`：复用现有 PG Pool + pgvector 扩展
//! - `vector-qdrant`：Qdrant gRPC 客户端
//! - `vector-lancedb`：LanceDB 嵌入式列存
//!
//! 所有实现共享同一份 [`Record`] / [`Query`] / [`Match`] 数据模型，
//! 业务层只依赖 [`VectorStore`]。

use crate::error::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "vector-memory")]
pub mod memory;

#[cfg(feature = "vector-pgvector")]
pub mod pgvector_store;

#[cfg(feature = "vector-qdrant")]
pub mod qdrant_store;

#[cfg(feature = "vector-qdrant-edge")]
pub mod qdrant_edge_store;

#[cfg(feature = "vector-lancedb")]
pub mod lancedb_store;

/// 距离度量。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Distance {
    #[default]
    Cosine,
    L2,
    Dot,
}

/// 向量记录。`metadata` 用于过滤与回显。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// 检索请求。
#[derive(Debug, Clone)]
pub struct Query<'a> {
    pub vector: &'a [f32],
    pub top_k: usize,
    /// 简单 metadata 等值过滤（各后端按能力实现，过弱时可整体忽略并在上层过滤）。
    pub filter: Option<&'a HashMap<String, serde_json::Value>>,
}

/// 命中结果。`score` 为相似度（越大越相似），各后端内部已对距离做过归一。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Match {
    pub id: String,
    pub score: f32,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// 集合（表/collection）规格。
#[derive(Debug, Clone)]
pub struct CollectionSpec<'a> {
    pub name: &'a str,
    pub dim: usize,
    pub distance: Distance,
}

/// 向量存储统一接口。所有方法以集合名定位，租户隔离由调用方在集合命名上体现
/// （例如 `format!("{}{}", ctx.cache_prefix(), "docs")`）。
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// 创建集合（已存在则跳过）。
    async fn ensure_collection(&self, spec: CollectionSpec<'_>) -> Result<()>;

    /// 删除整个集合。
    async fn drop_collection(&self, name: &str) -> Result<()>;

    /// 批量写入 / 覆盖。
    async fn upsert(&self, collection: &str, records: &[Record]) -> Result<()>;

    /// 按 id 删除。
    async fn delete(&self, collection: &str, ids: &[String]) -> Result<()>;

    /// 相似度检索。
    async fn search(&self, collection: &str, query: Query<'_>) -> Result<Vec<Match>>;

    /// 后端名（诊断用）。
    fn backend_name(&self) -> &'static str;
}

/// 工具：cosine 相似度。
///
/// 当启用 `vector-simd` feature 时使用 `wide` 的 8-lane f32 SIMD 加速。
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(feature = "vector-simd")]
    {
        return cosine_simd(a, b);
    }
    #[cfg(not(feature = "vector-simd"))]
    {
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        let denom = (na.sqrt() * nb.sqrt()).max(f32::EPSILON);
        dot / denom
    }
}

#[cfg(feature = "vector-simd")]
fn cosine_simd(a: &[f32], b: &[f32]) -> f32 {
    use wide::f32x8;
    let n = a.len().min(b.len()); // Safety: never read past the shorter vector
    let chunks = n / 8;
    let mut dot = f32x8::ZERO;
    let mut na = f32x8::ZERO;
    let mut nb = f32x8::ZERO;
    for i in 0..chunks {
        let off = i * 8;
        let va = f32x8::from(<[f32; 8]>::try_from(&a[off..off + 8]).unwrap());
        let vb = f32x8::from(<[f32; 8]>::try_from(&b[off..off + 8]).unwrap());
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }
    let dot_arr: [f32; 8] = dot.into();
    let na_arr: [f32; 8] = na.into();
    let nb_arr: [f32; 8] = nb.into();
    let mut sd = dot_arr.iter().sum::<f32>();
    let mut sa = na_arr.iter().sum::<f32>();
    let mut sb = nb_arr.iter().sum::<f32>();
    for i in (chunks * 8)..n {
        sd += a[i] * b[i];
        sa += a[i] * a[i];
        sb += b[i] * b[i];
    }
    let denom = (sa.sqrt() * sb.sqrt()).max(f32::EPSILON);
    sd / denom
}
