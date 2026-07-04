use thiserror::Error;

pub type Result<T> = std::result::Result<T, DbError>;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("config error: {0}")]
    Config(String),

    #[error("sql error: {0}")]
    Sql(#[from] sqlx::Error),

    #[error("cache error: {0}")]
    Cache(String),

    #[error("vector error: {0}")]
    Vector(String),

    #[error("migration error: {0}")]
    Migrate(String),

    #[error("feature `{0}` is not licensed in current edition")]
    FeatureLocked(&'static str),

    #[error("tenant context required but missing")]
    TenantRequired,

    #[error("unsupported backend: {0}")]
    Unsupported(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("other: {0}")]
    Other(String),
}

impl DbError {
    pub fn cache<E: std::fmt::Display>(e: E) -> Self {
        Self::Cache(e.to_string())
    }

    pub fn vector<E: std::fmt::Display>(e: E) -> Self {
        Self::Vector(e.to_string())
    }
}
