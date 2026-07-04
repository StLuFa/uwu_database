use crate::error::Result;
use crate::tenant::TenantCtx;
use crate::Database;
use async_trait::async_trait;

/// 分页参数。
#[derive(Debug, Clone)]
pub struct Page {
    /// 偏移量（0-based）。
    pub offset: u64,
    /// 每页条数。
    pub limit: u64,
}

impl Page {
    pub fn new(offset: u64, limit: u64) -> Self { Self { offset, limit } }
    pub fn first(limit: u64) -> Self { Self { offset: 0, limit } }
}

impl Default for Page {
    fn default() -> Self { Self { offset: 0, limit: 20 } }
}

/// 分页结果。
#[derive(Debug, Clone)]
pub struct PagedResult<T> {
    pub items: Vec<T>,
    /// 满足条件的总行数（不含分页），由 Repository 实现提供。
    pub total: u64,
    pub page: Page,
}

impl<T> PagedResult<T> {
    pub fn new(items: Vec<T>, total: u64, page: Page) -> Self {
        Self { items, total, page }
    }

    pub fn has_next(&self) -> bool {
        self.page.offset + (self.items.len() as u64) < self.total
    }
}

/// 业务 Repository 的统一基 trait。
#[async_trait]
pub trait Repository: Send + Sync {
    type Entity: Send + Sync;
    type Id: Send + Sync;

    async fn find_by_id(&self, ctx: &TenantCtx, id: &Self::Id) -> Result<Option<Self::Entity>>;
    async fn insert(&self, ctx: &TenantCtx, entity: &Self::Entity) -> Result<Self::Id>;
    async fn update(&self, ctx: &TenantCtx, entity: &Self::Entity) -> Result<()>;
    async fn delete(&self, ctx: &TenantCtx, id: &Self::Id) -> Result<()>;

    /// 分页列举。默认实现返回空，各后端可覆盖以提供高效实现。
    async fn list(&self, ctx: &TenantCtx, page: Page) -> Result<PagedResult<Self::Entity>> {
        let _ = (ctx, page);
        Ok(PagedResult::new(vec![], 0, Page::default()))
    }

    /// 批量插入。默认实现逐条插入；后端可覆盖以获得更好性能。
    async fn insert_many(&self, ctx: &TenantCtx, entities: &[Self::Entity]) -> Result<Vec<Self::Id>> {
        let mut ids = Vec::with_capacity(entities.len());
        for e in entities {
            ids.push(self.insert(ctx, e).await?);
        }
        Ok(ids)
    }

    /// 批量删除。默认实现逐条删除；后端可覆盖。
    async fn delete_many(&self, ctx: &TenantCtx, ids: &[Self::Id]) -> Result<()> {
        for id in ids {
            self.delete(ctx, id).await?;
        }
        Ok(())
    }
}

/// Cache-Aside 读取：先查缓存，未命中回源 loader 并回填。
///
/// 启用 `single-flight` feature 时，相同 key 的并发请求会合并为一次回源，避免缓存击穿。
pub async fn cache_aside<T, F, Fut>(
    db: &Database,
    ctx: &TenantCtx,
    key: &str,
    ttl: Option<std::time::Duration>,
    loader: F,
) -> Result<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Clone + Send + 'static,
    F: FnOnce() -> Fut + Send,
    Fut: std::future::Future<Output = Result<T>> + Send,
{
    let full_key = format!("{}{}", ctx.cache_prefix(), key);

    if let Some(bytes) = db.cache.get(&full_key).await? {
        if let Ok(v) = serde_json::from_slice::<T>(&bytes) {
            return Ok(v);
        }
        // 缓存内容损坏，删除后重新加载
        let _ = db.cache.del(&full_key).await;
    }

    #[cfg(feature = "single-flight")]
    {
        let value = single_flight::run(&full_key, loader).await?;
        if let Ok(bytes) = serde_json::to_vec(&value) {
            let _ = db.cache.set(&full_key, &bytes, ttl).await;
        }
        Ok(value)
    }

    #[cfg(not(feature = "single-flight"))]
    {
        let value = loader().await?;
        if let Ok(bytes) = serde_json::to_vec(&value) {
            let _ = db.cache.set(&full_key, &bytes, ttl).await;
        }
        Ok(value)
    }
}

/// 使缓存 key 失效（带租户前缀）。
pub async fn cache_invalidate(db: &Database, ctx: &TenantCtx, key: &str) -> Result<()> {
    let full_key = format!("{}{}", ctx.cache_prefix(), key);
    db.cache.del(&full_key).await
}

/// 批量使缓存失效。
pub async fn cache_invalidate_many(db: &Database, ctx: &TenantCtx, keys: &[&str]) -> Result<()> {
    let full_keys: Vec<String> = keys.iter()
        .map(|k| format!("{}{}", ctx.cache_prefix(), k))
        .collect();
    db.cache.del_many(&full_keys).await
}

#[cfg(feature = "single-flight")]
mod single_flight {
    use crate::error::{DbError, Result};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    use tokio::sync::broadcast;

    type Sender = broadcast::Sender<std::result::Result<Vec<u8>, String>>;

    struct Registry {
        inflight: Mutex<HashMap<String, Sender>>,
    }

    fn registry() -> &'static Registry {
        static R: OnceLock<Arc<Registry>> = OnceLock::new();
        R.get_or_init(|| Arc::new(Registry { inflight: Mutex::new(HashMap::new()) }))
    }

    /// 同 key 并发只发一次 loader；其余等待结果广播。
    pub async fn run<T, F, Fut>(key: &str, loader: F) -> Result<T>
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Send,
        F: FnOnce() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T>> + Send,
    {
        let reg = registry();

        // 尝试成为 leader；若已有 inflight 则订阅
        let (is_leader, mut rx) = {
            let mut g = reg.inflight.lock().unwrap();
            if let Some(tx) = g.get(key) {
                (false, tx.subscribe())
            } else {
                let (tx, rx) = broadcast::channel(1);
                g.insert(key.to_string(), tx);
                (true, rx)
            }
        };

        if is_leader {
            let result = loader().await;
            // 发布
            let payload = match &result {
                Ok(v) => serde_json::to_vec(v).map_err(|e| e.to_string()),
                Err(e) => Err(e.to_string()),
            };
            {
                let mut g = reg.inflight.lock().unwrap();
                if let Some(tx) = g.remove(key) {
                    let _ = tx.send(payload);
                }
            }
            result
        } else {
            match rx.recv().await {
                Ok(Ok(bytes)) => serde_json::from_slice::<T>(&bytes)
                    .map_err(|e| DbError::Other(format!("single-flight decode: {e}"))),
                Ok(Err(e)) => Err(DbError::Other(format!("leader failed: {e}"))),
                Err(e) => Err(DbError::Other(format!("single-flight recv: {e}"))),
            }
        }
    }
}
