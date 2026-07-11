use serde::{Deserialize, Serialize};
use std::time::Duration;

/// 顶层运行时配置。一般由调用方读取 `config.toml` / 环境变量后构造。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub deploy: DeployConfig,
    pub database: DbConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub vector: VectorConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeployConfig {
    #[serde(default)]
    pub mode: DeployMode,
    #[serde(default)]
    pub edition: Edition,
    /// 企业版离线许可证字符串（可选）。
    #[serde(default)]
    pub license: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeployMode {
    #[default]
    SelfHosted,
    Cloud,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Edition {
    #[default]
    Community,
    Enterprise,
}

/// 数据库配置。`backend` 必须与编译时启用的 feature 一致。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbConfig {
    pub backend: SqlBackend,
    pub url: String,
    #[serde(default = "default_max_conn")]
    pub max_connections: u32,
    #[serde(default = "default_min_conn")]
    pub min_connections: u32,
    #[serde(default = "default_acquire_secs")]
    pub acquire_timeout_secs: u64,
    #[serde(default = "default_idle_secs")]
    pub idle_timeout_secs: u64,
    /// 单条连接最大存活，避免长连接挂在云端 LB 后被静默断开。
    #[serde(default = "default_max_lifetime_secs")]
    pub max_lifetime_secs: u64,
    /// 取连接前 ping 校验（默认关闭，开启会牺牲一次 round-trip）。
    #[serde(default)]
    pub test_before_acquire: bool,
    /// prepared statement 缓存容量。0 表示禁用。
    #[serde(default = "default_stmt_cache")]
    pub statement_cache_capacity: usize,
    /// 应用名（PG `application_name`），便于慢查询定位。
    #[serde(default)]
    pub application_name: Option<String>,
}

fn default_max_conn() -> u32 {
    10
}
fn default_min_conn() -> u32 {
    0
}
fn default_acquire_secs() -> u64 {
    5
}
fn default_idle_secs() -> u64 {
    600
}
fn default_max_lifetime_secs() -> u64 {
    1800
}
fn default_stmt_cache() -> usize {
    100
}

impl DbConfig {
    pub fn acquire_timeout(&self) -> Duration {
        Duration::from_secs(self.acquire_timeout_secs)
    }
    pub fn idle_timeout(&self) -> Duration {
        Duration::from_secs(self.idle_timeout_secs)
    }
    pub fn max_lifetime(&self) -> Duration {
        Duration::from_secs(self.max_lifetime_secs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SqlBackend {
    Postgres,
    MySql,
    Sqlite,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheConfig {
    #[serde(default)]
    pub backend: CacheBackend,
    /// Redis URL，仅 `backend = redis` 时使用。
    #[serde(default)]
    pub url: Option<String>,
    /// 内存缓存容量，仅 `backend = memory` 时使用。
    #[serde(default = "default_capacity")]
    pub capacity: usize,
}

fn default_capacity() -> usize {
    10_000
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheBackend {
    #[default]
    Memory,
    Redis,
    None,
}

/// 向量后端配置。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorConfig {
    #[serde(default)]
    pub backend: VectorBackend,
    /// Qdrant / LanceDB 连接 URL；内存 / pgvector 模式可留空。
    #[serde(default)]
    pub url: Option<String>,
    /// Qdrant API Key（可选）。
    #[serde(default)]
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorBackend {
    /// 纯内存 brute-force，仅供开发 / 测试。
    #[default]
    Memory,
    /// 复用 PG Pool + pgvector 扩展。
    Pgvector,
    /// Qdrant gRPC。
    Qdrant,
    /// Qdrant Edge 嵌入式库（进程内，类似 SQLite）。
    QdrantEdge,
    /// LanceDB 嵌入式列存（uri 为本地路径或 S3 地址）。
    LanceDb,
}

/// 顶层应用配置（业务层可在此基础上扩展）。
pub type AppConfig = RuntimeConfig;
