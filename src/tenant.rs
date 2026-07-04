use serde::{Deserialize, Serialize};
use std::fmt;

/// 租户上下文，贯穿请求生命周期。
///
/// 个人版部署可只用 [`TenantCtx::personal`]。企业版部署在认证中间件中
/// 解析用户/组织后填入。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantCtx {
    pub tenant_id: TenantId,
    pub user_id: Option<String>,
    /// 可选：每租户独立 schema（PostgreSQL）或独立库。
    pub schema: Option<String>,
}

impl TenantCtx {
    pub fn personal() -> Self {
        Self {
            tenant_id: TenantId::Personal,
            user_id: None,
            schema: None,
        }
    }

    pub fn enterprise(tenant_id: impl Into<String>, user_id: Option<String>) -> Self {
        Self {
            tenant_id: TenantId::Enterprise(tenant_id.into()),
            user_id,
            schema: None,
        }
    }

    pub fn with_schema(mut self, schema: impl Into<String>) -> Self {
        self.schema = Some(schema.into());
        self
    }

    /// 用作缓存 key 前缀，避免跨租户串数据。
    pub fn cache_prefix(&self) -> String {
        match &self.tenant_id {
            TenantId::Personal => "p:".to_string(),
            TenantId::Enterprise(id) => format!("e:{id}:"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum TenantId {
    Personal,
    Enterprise(String),
}

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TenantId::Personal => f.write_str("personal"),
            TenantId::Enterprise(id) => write!(f, "enterprise:{id}"),
        }
    }
}
