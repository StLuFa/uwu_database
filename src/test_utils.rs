//! 多后端集成测试工具函数。
//!
//! - PG:    `DATABASE_URL` 环境变量
//! - MySQL: `MYSQL_DATABASE_URL` 环境变量
//! - SQLite: 内存模式 `:memory:`，无需外部依赖
//! - Qdrant: `QDRANT_URL` 环境变量
//!
//! 每个测试模块使用独立的前缀隔离，避免相互干扰。
//! 无对应环境变量时 PG/MySQL/Qdrant 测试会 panic-skip。

use crate::config::{CacheBackend, CacheConfig, DbConfig, DeployConfig, RuntimeConfig, SqlBackend, VectorBackend, VectorConfig};
use crate::sql::{self, DbPool};
use std::sync::atomic::{AtomicU32, Ordering};
use uuid::Uuid;

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// 生成唯一的测试前缀（基于 UUID），用于隔离测试表名。
pub fn unique_prefix() -> String {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("t{:x}_{}", n, &Uuid::new_v4().to_string()[..8])
}

/// 创建一个包含唯一前缀的表名。
pub fn tn(prefix: &str, name: &str) -> String {
    format!("{prefix}_{name}")
}

// ===========================================================================
// PostgreSQL
// ===========================================================================

pub fn pg_url() -> Option<String> {
    std::env::var("DATABASE_URL").ok()
}

pub fn has_pg() -> bool {
    pg_url().is_some()
}

pub fn pg_config() -> Option<RuntimeConfig> {
    let url = pg_url()?;
    Some(RuntimeConfig {
        deploy: DeployConfig::default(),
        database: DbConfig {
            backend: SqlBackend::Postgres,
            url,
            max_connections: 2,
            min_connections: 0,
            acquire_timeout_secs: 5,
            idle_timeout_secs: 60,
            max_lifetime_secs: 300,
            test_before_acquire: false,
            statement_cache_capacity: 100,
            application_name: Some("uwu_db_test".into()),
        },
        cache: CacheConfig { backend: CacheBackend::None, capacity: 0, url: None },
        vector: VectorConfig { backend: VectorBackend::Memory, url: None, api_key: None },
    })
}

pub async fn pg_pool() -> Option<DbPool> {
    let cfg = pg_config()?;
    sql::build_pool(&cfg.database).await.ok()
}

// ===========================================================================
// MySQL
// ===========================================================================

pub fn mysql_url() -> Option<String> {
    std::env::var("MYSQL_DATABASE_URL").ok()
}

pub fn has_mysql() -> bool {
    mysql_url().is_some()
}

pub fn mysql_config() -> Option<RuntimeConfig> {
    let url = mysql_url()?;
    Some(RuntimeConfig {
        deploy: DeployConfig::default(),
        database: DbConfig {
            backend: SqlBackend::MySql,
            url,
            max_connections: 2,
            min_connections: 0,
            acquire_timeout_secs: 5,
            idle_timeout_secs: 60,
            max_lifetime_secs: 300,
            test_before_acquire: false,
            statement_cache_capacity: 100,
            application_name: Some("uwu_db_test".into()),
        },
        cache: CacheConfig { backend: CacheBackend::None, capacity: 0, url: None },
        vector: VectorConfig { backend: VectorBackend::Memory, url: None, api_key: None },
    })
}

pub async fn mysql_pool() -> Option<DbPool> {
    let cfg = mysql_config()?;
    sql::build_pool(&cfg.database).await.ok()
}

// ===========================================================================
// SQLite（内存模式，无需外部依赖）
// ===========================================================================

pub fn sqlite_config() -> RuntimeConfig {
    RuntimeConfig {
        deploy: DeployConfig::default(),
        database: DbConfig {
            backend: SqlBackend::Sqlite,
            url: "sqlite::memory:".to_string(),
            max_connections: 1,
            min_connections: 0,
            acquire_timeout_secs: 5,
            idle_timeout_secs: 60,
            max_lifetime_secs: 300,
            test_before_acquire: false,
            statement_cache_capacity: 100,
            application_name: Some("uwu_db_test".into()),
        },
        cache: CacheConfig { backend: CacheBackend::None, capacity: 0, url: None },
        vector: VectorConfig { backend: VectorBackend::Memory, url: None, api_key: None },
    }
}

pub async fn sqlite_pool() -> DbPool {
    let cfg = sqlite_config();
    sql::build_pool(&cfg.database).await.expect("sqlite memory pool")
}

// ===========================================================================
// Qdrant
// ===========================================================================

pub fn qdrant_url() -> Option<String> {
    std::env::var("QDRANT_URL").ok()
}

pub fn has_qdrant() -> bool {
    qdrant_url().is_some()
}
