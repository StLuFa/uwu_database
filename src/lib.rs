//! uwu_database
//!
//! 统一数据访问层，封装 SQL（PostgreSQL / MySQL / SQLite）与缓存（Memory / Redis），
//! 内置多租户上下文与社区版/企业版功能开关。
//!
//! 通过 Cargo features 在编译期裁剪后端：
//! - `postgres` / `mysql` / `sqlite`：启用对应 SQL 驱动
//! - `cache-memory` / `cache-redis`：启用对应缓存实现
//! - `vector-memory` / `vector-pgvector` / `vector-qdrant` / `vector-lancedb`：向量后端
//! - `community` / `enterprise`：版本开关（企业版包含 `multi-tenant` 与 `audit-log`）

pub mod cache;
pub mod config;
pub mod error;
pub mod features;
pub mod migrate;
pub mod repo;
pub mod sql;
pub mod tenant;
pub mod vector;

#[cfg(test)]
pub mod test_utils;

pub use cache::Cache;
pub use config::{
    AppConfig, CacheBackend, CacheConfig, DbConfig, DeployMode, Edition, RuntimeConfig, SqlBackend,
    VectorBackend, VectorConfig,
};
pub use error::{DbError, Result};
pub use features::{FeatureKey, Features};
pub use migrate::{Migration, MigrationRecord, Migrator, SqlMigration};
pub use repo::{Page, PagedResult, Repository};
pub use sql::DbPool;
pub use tenant::{TenantCtx, TenantId};
pub use vector::{CollectionSpec, Distance, Match, Query, Record, VectorStore};

use std::sync::Arc;

/// 数据库门面对象，业务层注入此结构即可。
#[derive(Clone)]
pub struct Database {
    pub pool: DbPool,
    pub cache: Arc<dyn Cache>,
    pub features: Arc<Features>,
    /// 可选向量存储；未配置时为 None。
    pub vector: Option<Arc<dyn VectorStore>>,
}

impl Database {
    /// 按运行时配置构建（不含向量后端）。
    pub async fn connect(cfg: &RuntimeConfig) -> Result<Self> {
        let pool = sql::build_pool(&cfg.database).await?;
        let cache = cache::build_cache(&cfg.cache).await?;
        let features = Arc::new(Features::from_config(cfg));
        Ok(Self {
            pool,
            cache,
            features,
            vector: None,
        })
    }

    /// 按运行时配置构建，同时初始化向量后端。
    pub async fn connect_with_vector(cfg: &RuntimeConfig) -> Result<Self> {
        let mut db = Self::connect(cfg).await?;
        let vs = build_vector_store(&cfg.vector, &db.pool).await?;
        db.vector = Some(vs);
        Ok(db)
    }

    /// 获取向量存储；未配置时返回错误。
    pub fn vector_store(&self) -> Result<&Arc<dyn VectorStore>> {
        self.vector
            .as_ref()
            .ok_or_else(|| DbError::Config("vector store not configured".into()))
    }

    /// 关闭底层资源。
    pub async fn close(&self) {
        self.pool.close().await;
    }

    /// 获取一个全新的 [`migrate::Migrator`] 实例，用于注册迁移。
    ///
    /// # 示例
    /// ```rust,ignore
    /// let db = Database::connect(&cfg).await?;
    /// let m = db.migrator()
    ///     .add(SqlMigration::new(1, "init", "CREATE TABLE t (id INT);", None));
    /// m.up(&db.pool).await?;
    /// ```
    pub fn migrator(&self) -> migrate::Migrator {
        migrate::Migrator::new()
    }

    /// 应用所有待执行迁移（使用已注册的 Migrator）。
    ///
    /// 如果只需要简单场景，可以直接用 `Migrator::load_dir()` 发现磁盘上的 SQL 文件。
    pub async fn migrate_up(&self, m: &migrate::Migrator) -> Result<()> {
        m.up(&self.pool).await
    }

    /// 查看迁移状态。
    pub async fn migrate_status(
        &self,
        m: &migrate::Migrator,
    ) -> Result<Vec<migrate::MigrationRecord>> {
        m.status(&self.pool).await
    }
}

/// 按 VectorConfig 构建向量存储实例。
#[allow(unused_variables)]
pub async fn build_vector_store(
    cfg: &config::VectorConfig,
    pool: &DbPool,
) -> Result<Arc<dyn VectorStore>> {
    match cfg.backend {
        VectorBackend::Memory => {
            #[cfg(feature = "vector-memory")]
            {
                Ok(Arc::new(vector::memory::MemoryVectorStore::new()))
            }
            #[cfg(not(feature = "vector-memory"))]
            {
                Err(DbError::Unsupported(
                    "vector-memory feature disabled".into(),
                ))
            }
        }
        VectorBackend::Pgvector => {
            #[cfg(feature = "vector-pgvector")]
            {
                let store = vector::pgvector_store::PgVectorStore::new(Arc::new(pool.clone()))?;
                Ok(Arc::new(store))
            }
            #[cfg(not(feature = "vector-pgvector"))]
            {
                Err(DbError::Unsupported(
                    "vector-pgvector feature disabled".into(),
                ))
            }
        }
        VectorBackend::Qdrant => {
            #[cfg(feature = "vector-qdrant")]
            {
                let url = cfg
                    .url
                    .as_deref()
                    .ok_or_else(|| DbError::Config("qdrant url is required".into()))?;
                let store = vector::qdrant_store::QdrantVectorStore::new(url, cfg.api_key.clone())?;
                Ok(Arc::new(store))
            }
            #[cfg(not(feature = "vector-qdrant"))]
            {
                Err(DbError::Unsupported(
                    "vector-qdrant feature disabled".into(),
                ))
            }
        }
        VectorBackend::QdrantEdge => {
            #[cfg(feature = "vector-qdrant-edge")]
            {
                let dir = cfg.url.as_deref().unwrap_or("./uwu_edge_data");
                let store = vector::qdrant_edge_store::QdrantEdgeVectorStore::new(dir)?;
                Ok(Arc::new(store))
            }
            #[cfg(not(feature = "vector-qdrant-edge"))]
            {
                Err(DbError::Unsupported(
                    "vector-qdrant-edge feature disabled".into(),
                ))
            }
        }
        VectorBackend::LanceDb => {
            #[cfg(feature = "vector-lancedb")]
            {
                let uri = cfg
                    .url
                    .as_deref()
                    .ok_or_else(|| DbError::Config("lancedb uri is required".into()))?;
                let store = vector::lancedb_store::LanceDbVectorStore::open(uri).await?;
                Ok(Arc::new(store))
            }
            #[cfg(not(feature = "vector-lancedb"))]
            {
                Err(DbError::Unsupported(
                    "vector-lancedb feature disabled".into(),
                ))
            }
        }
    }
}
