use crate::config::{Edition, RuntimeConfig};
use crate::error::{DbError, Result};

/// 受许可证 / Edition 控制的功能键。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureKey {
    MultiTenant,
    AuditLog,
    Sso,
    Cluster,
}

impl FeatureKey {
    pub const fn name(self) -> &'static str {
        match self {
            FeatureKey::MultiTenant => "multi_tenant",
            FeatureKey::AuditLog => "audit_log",
            FeatureKey::Sso => "sso",
            FeatureKey::Cluster => "cluster",
        }
    }
}

/// 运行时功能开关集合。
#[derive(Debug, Clone)]
pub struct Features {
    edition: Edition,
}

impl Features {
    pub fn from_config(cfg: &RuntimeConfig) -> Self {
        Self {
            edition: cfg.deploy.edition,
        }
    }

    pub fn edition(&self) -> Edition {
        self.edition
    }

    pub fn is_enterprise(&self) -> bool {
        self.edition == Edition::Enterprise
    }

    /// 判断是否启用某功能。同时受编译期 feature 与运行时 edition 影响。
    pub fn enabled(&self, key: FeatureKey) -> bool {
        match key {
            FeatureKey::MultiTenant => cfg!(feature = "multi-tenant") && self.is_enterprise(),
            FeatureKey::AuditLog => cfg!(feature = "audit-log") && self.is_enterprise(),
            FeatureKey::Sso | FeatureKey::Cluster => self.is_enterprise(),
        }
    }

    /// 关键路径强校验：未启用即报错。
    pub fn require(&self, key: FeatureKey) -> Result<()> {
        if self.enabled(key) {
            Ok(())
        } else {
            Err(DbError::FeatureLocked(key.name()))
        }
    }
}
