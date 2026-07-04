use crate::config::{CacheBackend, CacheConfig};
use crate::error::{DbError, Result};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

/// 统一缓存接口。`get` 返回 None 表示未命中。
#[async_trait]
pub trait Cache: Send + Sync {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    async fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()>;
    async fn del(&self, key: &str) -> Result<()>;
    /// 批量删除。默认实现逐条删除；后端有原生批删时可覆盖以提升效率。
    async fn del_many(&self, keys: &[String]) -> Result<()> {
        for k in keys {
            self.del(k).await?;
        }
        Ok(())
    }
    fn backend(&self) -> CacheBackend;
}

/// 不做任何缓存（用于关闭缓存的部署形态）。
pub struct NoopCache;

#[async_trait]
impl Cache for NoopCache {
    async fn get(&self, _: &str) -> Result<Option<Vec<u8>>> { Ok(None) }
    async fn set(&self, _: &str, _: &[u8], _: Option<Duration>) -> Result<()> { Ok(()) }
    async fn del(&self, _: &str) -> Result<()> { Ok(()) }
    fn backend(&self) -> CacheBackend { CacheBackend::None }
}

#[cfg(feature = "cache-memory")]
mod memory {
    use super::*;
    use moka::future::Cache as MokaCache;
    use std::time::Instant;

    /// 带过期时间的缓存条目。
    #[derive(Clone)]
    struct Entry {
        data: Vec<u8>,
        /// 绝对过期时间。None 表示不过期。
        expires_at: Option<Instant>,
    }

    impl Entry {
        fn is_expired(&self) -> bool {
            self.expires_at.map_or(false, |t| Instant::now() >= t)
        }
    }

    /// 基于 moka 的高性能内存缓存：分段并发、TinyLFU 淘汰、per-entry TTL（通过时间戳实现）。
    ///
    /// moka 0.12 不支持 `insert_with_ttl` per-entry API，故将过期时间戳编码进 Entry，
    /// get 时懒惰检查是否已过期。moka 全局 TTL 设为 24h 作为兜底上限。
    pub struct MemoryCache {
        inner: MokaCache<String, Entry>,
    }

    impl MemoryCache {
        pub fn new(capacity: usize) -> Self {
            let inner = MokaCache::builder()
                .max_capacity(capacity as u64)
                // 24h 兜底：无 per-entry TTL 的条目最终也会被淘汰
                .time_to_live(Duration::from_secs(86400))
                .build();
            Self { inner }
        }
    }

    #[async_trait]
    impl Cache for MemoryCache {
        async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
            match self.inner.get(key).await {
                Some(e) if e.is_expired() => {
                    // 懒惰删除
                    self.inner.invalidate(key).await;
                    Ok(None)
                }
                Some(e) => Ok(Some(e.data)),
                None => Ok(None),
            }
        }

        async fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()> {
            let expires_at = ttl.map(|d| Instant::now() + d);
            self.inner.insert(key.to_string(), Entry {
                data: value.to_vec(),
                expires_at,
            }).await;
            Ok(())
        }

        async fn del(&self, key: &str) -> Result<()> {
            self.inner.invalidate(key).await;
            Ok(())
        }

        async fn del_many(&self, keys: &[String]) -> Result<()> {
            for k in keys {
                self.inner.invalidate(k.as_str()).await;
            }
            Ok(())
        }

        fn backend(&self) -> CacheBackend { CacheBackend::Memory }
    }
}

#[cfg(feature = "cache-redis")]
mod redis_impl {
    use super::*;
    use deadpool_redis::{Config as RedisConfig, Pool, Runtime};
    use redis::AsyncCommands;

    pub struct RedisCache { pool: Pool }

    impl RedisCache {
        pub fn new(url: &str) -> Result<Self> {
            let cfg = RedisConfig::from_url(url);
            let pool = cfg.create_pool(Some(Runtime::Tokio1))
                .map_err(DbError::cache)?;
            Ok(Self { pool })
        }
    }

    #[async_trait]
    impl Cache for RedisCache {
        async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
            let mut conn = self.pool.get().await.map_err(DbError::cache)?;
            let v: Option<Vec<u8>> = conn.get(key).await.map_err(DbError::cache)?;
            Ok(v)
        }

        async fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()> {
            let mut conn = self.pool.get().await.map_err(DbError::cache)?;
            match ttl {
                Some(d) => {
                    let secs = d.as_secs().max(1);
                    let _: () = conn.set_ex(key, value, secs)
                        .await.map_err(DbError::cache)?;
                }
                None => {
                    let _: () = conn.set(key, value).await.map_err(DbError::cache)?;
                }
            }
            Ok(())
        }

        async fn del(&self, key: &str) -> Result<()> {
            let mut conn = self.pool.get().await.map_err(DbError::cache)?;
            let _: () = conn.del(key).await.map_err(DbError::cache)?;
            Ok(())
        }

        async fn del_many(&self, keys: &[String]) -> Result<()> {
            if keys.is_empty() { return Ok(()); }
            let mut conn = self.pool.get().await.map_err(DbError::cache)?;
            let _: () = conn.del(keys).await.map_err(DbError::cache)?;
            Ok(())
        }

        fn backend(&self) -> CacheBackend { CacheBackend::Redis }
    }
}

/// 按配置构建缓存实现。
pub async fn build_cache(cfg: &CacheConfig) -> Result<Arc<dyn Cache>> {
    match cfg.backend {
        CacheBackend::None => Ok(Arc::new(NoopCache)),
        CacheBackend::Memory => {
            #[cfg(feature = "cache-memory")]
            { Ok(Arc::new(memory::MemoryCache::new(cfg.capacity))) }
            #[cfg(not(feature = "cache-memory"))]
            { Err(DbError::Unsupported("cache-memory feature disabled".into())) }
        }
        CacheBackend::Redis => {
            #[cfg(feature = "cache-redis")]
            {
                let url = cfg.url.as_deref().ok_or_else(||
                    DbError::Config("redis url is required".into()))?;
                Ok(Arc::new(redis_impl::RedisCache::new(url)?))
            }
            #[cfg(not(feature = "cache-redis"))]
            { Err(DbError::Unsupported("cache-redis feature disabled".into())) }
        }
    }
}
