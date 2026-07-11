//! 迁移模块：Migrator 主逻辑与 embedded_sql! 宏。
//!
//! # 快速开始
//!
//! ```rust,ignore
//! use uwu_database::migrate::{Migrator, SqlMigration};
//!
//! let migrator = Migrator::new()
//!     .add(SqlMigration::new(1, "create_users",
//!         "CREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT NOT NULL)",
//!         Some("DROP TABLE users"),
//!     ));
//!
//! // 应用所有待迁移
//! migrator.up(&db.pool).await?;
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use async_trait::async_trait;
use tracing::{info, warn};

use crate::error::Result;
use crate::sql::DbPool;

mod support;
use support::now_rfc3339;

// ── 公开宏 ─────────────────────────────────────────────

/// 便捷宏：从字面量创建 [`SqlMigration`]。
///
/// ```rust,ignore
/// let m = embedded_sql!(1, "init", "CREATE TABLE t (id INT);", None);
/// ```
#[macro_export]
macro_rules! embedded_sql {
    ($version:expr, $name:expr, $up:expr, $down:expr) => {
        $crate::migrate::SqlMigration::new($version, $name, $up, $down)
    };
    ($version:expr, $name:expr, $up:expr) => {
        $crate::migrate::SqlMigration::new($version, $name, $up, ::core::option::Option::None)
    };
}

pub use embedded_sql;

// ── 版本表 DDL ───────────────────────────────────────

const DEFAULT_VERSION_TABLE: &str = "_uwu_schema_version";

fn create_version_table_sql(table: &str) -> String {
    format!(
        r#"CREATE TABLE IF NOT EXISTS "{table}" (
            version     BIGINT PRIMARY KEY,
            name        TEXT    NOT NULL,
            applied_at  TEXT    NOT NULL,
            checksum    TEXT
        )"#
    )
}

// ── Migration trait ─────────────────────────────────────

/// 单个迁移的定义。
#[async_trait]
pub trait Migration: Send + Sync + 'static {
    fn version(&self) -> i64;
    fn name(&self) -> &str;
    async fn up(&self, pool: &DbPool) -> Result<()>;
    fn down_sql(&self) -> Option<&str>;
}

// ── SqlMigration ───────────────────────────────────────

/// 基于 SQL 语句的迁移。
///
/// **注意—事务语义：** PostgreSQL 和 MySQL 的 DDL 支持事务回滚；
/// SQLite 的 DDL 在事务中执行时行为不同（某些语句会隐式提交）。
/// 如需跨语句原子性，请在 SQL 文件中显式书写 `BEGIN;` / `COMMIT;`。
pub struct SqlMigration {
    pub version: i64,
    pub name: String,
    pub up_sql: String,
    pub down_sql: Option<String>,
}

impl SqlMigration {
    pub fn new(
        version: i64,
        name: impl Into<String>,
        up_sql: impl Into<String>,
        down_sql: Option<impl Into<String>>,
    ) -> Self {
        Self {
            version,
            name: name.into(),
            up_sql: up_sql.into(),
            down_sql: down_sql.map(Into::into),
        }
    }

    pub fn from_files(
        version: i64,
        name: impl Into<String>,
        up_path: impl AsRef<Path>,
        down_path: Option<impl AsRef<Path>>,
    ) -> std::io::Result<Self> {
        let up_sql = std::fs::read_to_string(up_path)
            .map_err(|e| std::io::Error::new(e.kind(), format!("read up.sql: {e}")))?;
        let down_sql = match down_path {
            Some(p) => Some(
                std::fs::read_to_string(p)
                    .map_err(|e| std::io::Error::new(e.kind(), format!("read down.sql: {e}")))?,
            ),
            None => None,
        };
        Ok(Self::new(version, name, up_sql, down_sql))
    }
}

#[async_trait]
impl Migration for SqlMigration {
    fn version(&self) -> i64 {
        self.version
    }
    fn name(&self) -> &str {
        &self.name
    }
    async fn up(&self, pool: &DbPool) -> Result<()> {
        pool.exec(&self.up_sql).await
    }
    fn down_sql(&self) -> Option<&str> {
        self.down_sql.as_deref()
    }
}

// ── MigrationRecord ─────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MigrationRecord {
    pub version: i64,
    pub name: String,
    pub applied_at: Option<String>,
    pub pending: bool,
}

// ── Migrator ───────────────────────────────────────────

/// 迁移管理器。
pub struct Migrator {
    migrations: BTreeMap<i64, Box<dyn Migration>>,
    version_table: String,
}

impl Migrator {
    pub fn new() -> Self {
        Self {
            migrations: BTreeMap::new(),
            version_table: DEFAULT_VERSION_TABLE.to_string(),
        }
    }

    pub fn with_version_table(mut self, table: impl Into<String>) -> Self {
        self.version_table = table.into();
        self
    }

    pub fn add(mut self, m: impl Migration + 'static) -> Self {
        let v = m.version();
        self.migrations.insert(v, Box::new(m));
        self
    }

    /// 从目录加载 SQL 文件迁移。
    ///
    /// 文件命名：`<version>_<name>.up.sql` / `<version>_<name>.down.sql`
    pub fn load_dir(path: impl AsRef<Path>) -> Result<Self> {
        let dir = path.as_ref();
        if !dir.is_dir() {
            return Err(crate::error::DbError::Migrate(format!(
                "migrations directory not found: {}",
                dir.display()
            )));
        }

        let mut migrator = Self::new();
        let mut seen: BTreeMap<i64, String> = BTreeMap::new();

        let entries = std::fs::read_dir(dir)
            .map_err(|e| crate::error::DbError::Migrate(format!("read migrations dir: {e}")))?;
        for entry in entries {
            let entry = entry
                .map_err(|e| crate::error::DbError::Migrate(format!("read dir entry: {e}")))?;
            let fname = entry.file_name().to_string_lossy().to_string();
            if let Some((version, name)) = parse_filename(&fname) {
                seen.insert(version, name);
            }
        }

        for (version, name) in seen {
            let up_path = dir.join(format!("{version:04}_{name}.up.sql"));
            let down_path = dir.join(format!("{version:04}_{name}.down.sql"));
            let down_sql =
                if down_path.exists() {
                    Some(std::fs::read_to_string(&down_path).map_err(|e| {
                        crate::error::DbError::Migrate(format!("read down.sql: {e}"))
                    })?)
                } else {
                    None
                };
            let up_sql = std::fs::read_to_string(&up_path)
                .map_err(|e| crate::error::DbError::Migrate(format!("read up.sql: {e}")))?;
            migrator = migrator.add(SqlMigration {
                version,
                name,
                up_sql,
                down_sql,
            });
        }

        Ok(migrator)
    }

    async fn ensure_version_table(&self, pool: &DbPool) -> Result<()> {
        let sql = create_version_table_sql(&self.version_table);
        pool.exec(&sql).await?;
        Ok(())
    }

    async fn get_applied(&self, pool: &DbPool) -> Result<Vec<(i64, String, String)>> {
        pool.fetch_version_records(&self.version_table).await
    }

    // ── 公共 API ───────────────────────────────────────

    pub async fn status(&self, pool: &DbPool) -> Result<Vec<MigrationRecord>> {
        self.ensure_version_table(pool).await?;
        let applied = self.get_applied(pool).await?;
        let applied_set: std::collections::HashSet<i64> =
            applied.iter().map(|(v, _, _)| *v).collect();

        Ok(self
            .migrations
            .iter()
            .map(|(v, m)| {
                let applied_info = applied.iter().find(|(av, _, _)| av == v);
                MigrationRecord {
                    version: *v,
                    name: m.name().to_string(),
                    applied_at: applied_info.map(|(_, _, t)| t.clone()),
                    pending: !applied_set.contains(v),
                }
            })
            .collect())
    }

    pub async fn up(&self, pool: &DbPool) -> Result<()> {
        self.up_to(pool, None).await
    }

    pub async fn up_to(&self, pool: &DbPool, target: Option<i64>) -> Result<()> {
        self.ensure_version_table(pool).await?;
        let applied = self.get_applied(pool).await?;
        let applied_set: std::collections::HashSet<i64> =
            applied.iter().map(|(v, _, _)| *v).collect();

        let target = target.unwrap_or(i64::MAX);

        for (v, m) in &self.migrations {
            if *v > target {
                break;
            }
            if applied_set.contains(v) {
                continue;
            }
            info!(version = v, name = m.name(), "applying migration");
            m.up(pool).await?;
            let now = now_rfc3339();
            pool.insert_version_record(&self.version_table, *v, m.name(), &now, "")
                .await?;
            info!(version = v, name = m.name(), "migration applied");
        }
        Ok(())
    }

    pub async fn down(&self, pool: &DbPool, target: i64) -> Result<()> {
        self.ensure_version_table(pool).await?;
        let applied = self.get_applied(pool).await?;

        let mut to_rollback: Vec<i64> = applied
            .iter()
            .map(|(v, _, _)| *v)
            .filter(|v| *v > target)
            .collect();
        to_rollback.sort_unstable_by(|a, b| b.cmp(a));

        for v in to_rollback {
            let Some(m) = self.migrations.get(&v) else {
                warn!(version = v, "migration definition not found, skipping");
                continue;
            };
            let Some(down_sql) = m.down_sql() else {
                return Err(crate::error::DbError::Migrate(format!(
                    "migration {v} ({}) has no down script",
                    m.name()
                )));
            };
            info!(version = v, name = m.name(), "rolling back migration");
            pool.exec(down_sql).await?;
            pool.delete_version_record(&self.version_table, v).await?;
            info!(version = v, name = m.name(), "migration rolled back");
        }
        Ok(())
    }
}

impl Default for Migrator {
    fn default() -> Self {
        Self::new()
    }
}

// ── 文件名解析 ─────────────────────────────────────────

fn parse_filename(fname: &str) -> Option<(i64, String)> {
    if let Some(stem) = fname.strip_suffix(".sql") {
        let (prefix, ext) = stem.rsplit_once('.')?;
        if ext != "up" {
            return None;
        }
        let mut parts = prefix.splitn(2, '_');
        let ver_str = parts.next()?;
        let name = parts.next()?;
        let version = ver_str.parse::<i64>().ok()?;
        Some((version, name.to_string()))
    } else {
        None
    }
}

// ── 测试 ───────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_filename_up() {
        assert_eq!(
            parse_filename("0001_init_users.up.sql"),
            Some((1, "init_users".to_string()))
        );
        assert_eq!(parse_filename("0010_add_email.down.sql"), None);
        assert_eq!(parse_filename("readme.md"), None);
    }
}

// ===========================================================================
// PG 集成测试
// ===========================================================================

#[cfg(test)]
mod pg_tests {
    use super::*;
    use crate::test_utils;

    fn require_pg() -> String {
        test_utils::pg_url().expect("SKIP: DATABASE_URL not set")
    }

    /// 为每个测试创建独立 Migrator 和唯一表名前缀，避免跨测试冲突。
    async fn setup() -> (DbPool, String) {
        let _url = require_pg();
        let pool = test_utils::pg_pool().await.unwrap();
        let prefix = test_utils::unique_prefix();
        (pool, prefix)
    }

    // ── 版本表 ─────────────────────────────────────────

    #[tokio::test]
    async fn test_ensure_version_table_creates_table() {
        let (pool, prefix) = setup().await;
        let vt = format!("{prefix}_vtab");

        let migrator = Migrator::new().with_version_table(&vt);
        migrator.ensure_version_table(&pool).await.unwrap();

        // 验证表存在
        let pg = pool.as_postgres().unwrap();
        let exists: (bool,) = sqlx::query_as(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
        )
        .bind(&vt)
        .fetch_one(pg)
        .await
        .unwrap();
        assert!(exists.0, "version table should exist");

        // 清理
        let _ = pool.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        pool.close().await;
    }

    // ── 应用迁移 ───────────────────────────────────────

    #[tokio::test]
    async fn test_apply_single_migration() {
        let (pool, prefix) = setup().await;
        let table = format!("{prefix}_users");
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new()
            .with_version_table(&vt)
            .add(SqlMigration::new(
                1,
                "create_users",
                format!("CREATE TABLE {table} (id SERIAL PRIMARY KEY, name TEXT)"),
                Some(format!("DROP TABLE IF EXISTS {table}")),
            ));

        migrator.up(&pool).await.unwrap();

        // 验证：表存在
        let pg = pool.as_postgres().unwrap();
        let exists: (bool,) = sqlx::query_as(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
        )
        .bind(&table)
        .fetch_one(pg)
        .await
        .unwrap();
        assert!(exists.0, "users table should exist");

        // 验证：可直接插入
        pool.exec(&format!("INSERT INTO {table} (name) VALUES ('bob')"))
            .await
            .unwrap();

        // 验证：版本记录存在
        let records = pool.fetch_version_records(&vt).await.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0, 1);
        assert_eq!(records[0].1, "create_users");

        // 清理
        let _ = pool.exec(&format!("DROP TABLE IF EXISTS {table}")).await;
        let _ = pool.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        pool.close().await;
    }

    #[tokio::test]
    async fn test_apply_multiple_migrations_in_order() {
        let (pool, prefix) = setup().await;
        let t1 = format!("{prefix}_t1");
        let t2 = format!("{prefix}_t2");
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new()
            .with_version_table(&vt)
            .add(SqlMigration::new(
                1,
                "create_t1",
                format!("CREATE TABLE {t1} (a INT)"),
                Some(format!("DROP TABLE IF EXISTS {t1}")),
            ))
            .add(SqlMigration::new(
                2,
                "create_t2",
                format!("CREATE TABLE {t2} (b INT)"),
                Some(format!("DROP TABLE IF EXISTS {t2}")),
            ));

        migrator.up(&pool).await.unwrap();

        // 两张表都存在
        let pg = pool.as_postgres().unwrap();
        for t in &[&t1, &t2] {
            let (exists,): (bool,) = sqlx::query_as(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
            )
            .bind(t)
            .fetch_one(pg)
            .await
            .unwrap();
            assert!(exists, "{t} should exist");
        }

        let records = pool.fetch_version_records(&vt).await.unwrap();
        assert_eq!(records.len(), 2);

        // 清理
        for t in &[&t1, &t2] {
            let _ = pool.exec(&format!("DROP TABLE IF EXISTS {t}")).await;
        }
        let _ = pool.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        pool.close().await;
    }

    #[tokio::test]
    async fn test_migration_idempotent() {
        let (pool, prefix) = setup().await;
        let table = format!("{prefix}_idem");
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new()
            .with_version_table(&vt)
            .add(SqlMigration::new(
                1,
                "init",
                format!("CREATE TABLE IF NOT EXISTS {table} (x INT)"),
                Some(format!("DROP TABLE IF EXISTS {table}")),
            ));

        // 第一次
        migrator.up(&pool).await.unwrap();
        // 第二次：应该跳过已应用的
        migrator.up(&pool).await.unwrap();

        let records = pool.fetch_version_records(&vt).await.unwrap();
        assert_eq!(records.len(), 1, "should not duplicate migration records");

        let _ = pool.exec(&format!("DROP TABLE IF EXISTS {table}")).await;
        let _ = pool.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        pool.close().await;
    }

    #[tokio::test]
    async fn test_migration_up_to_partial() {
        let (pool, prefix) = setup().await;
        let t1 = format!("{prefix}_p1");
        let t2 = format!("{prefix}_p2");
        let t3 = format!("{prefix}_p3");
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new()
            .with_version_table(&vt)
            .add(SqlMigration::new(
                1,
                "m1",
                format!("CREATE TABLE {t1} (a INT)"),
                Some(format!("DROP TABLE IF EXISTS {t1}")),
            ))
            .add(SqlMigration::new(
                2,
                "m2",
                format!("CREATE TABLE {t2} (b INT)"),
                Some(format!("DROP TABLE IF EXISTS {t2}")),
            ))
            .add(SqlMigration::new(
                3,
                "m3",
                format!("CREATE TABLE {t3} (c INT)"),
                Some(format!("DROP TABLE IF EXISTS {t3}")),
            ));

        // 只应用到版本 2
        migrator.up_to(&pool, Some(2)).await.unwrap();

        let records = pool.fetch_version_records(&vt).await.unwrap();
        assert_eq!(records.len(), 2);
        let versions: Vec<i64> = records.iter().map(|r| r.0).collect();
        assert_eq!(versions, vec![1, 2]);

        // t3 不应存在
        let pg = pool.as_postgres().unwrap();
        let (t3_exists,): (bool,) = sqlx::query_as(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
        )
        .bind(&t3)
        .fetch_one(pg)
        .await
        .unwrap();
        assert!(!t3_exists, "t3 should not be created");

        // 清理
        for t in &[&t1, &t2, &t3] {
            let _ = pool.exec(&format!("DROP TABLE IF EXISTS {t}")).await;
        }
        let _ = pool.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        pool.close().await;
    }

    // ── 状态查询 ───────────────────────────────────────

    #[tokio::test]
    async fn test_migration_status() {
        let (pool, prefix) = setup().await;
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new()
            .with_version_table(&vt)
            .add(SqlMigration::new(1, "m1", "SELECT 1", None::<&str>))
            .add(SqlMigration::new(2, "m2", "SELECT 2", None::<&str>));

        // 应用前：全部 pending
        let status_before = migrator.status(&pool).await.unwrap();
        assert_eq!(status_before.len(), 2);
        assert!(status_before.iter().all(|r| r.pending));

        // 应用后：全部 applied
        migrator.up(&pool).await.unwrap();
        let status_after = migrator.status(&pool).await.unwrap();
        assert!(status_after.iter().all(|r| !r.pending));
        assert!(status_after.iter().all(|r| r.applied_at.is_some()));

        let _ = pool.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        pool.close().await;
    }

    // ── 回滚 ───────────────────────────────────────────

    #[tokio::test]
    async fn test_migration_rollback() {
        let (pool, prefix) = setup().await;
        let table = format!("{prefix}_rb");
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new()
            .with_version_table(&vt)
            .add(SqlMigration::new(
                1,
                "create_rb",
                format!("CREATE TABLE {table} (x INT)"),
                Some(format!("DROP TABLE IF EXISTS {table}")),
            ));

        // up
        migrator.up(&pool).await.unwrap();

        // down to 0（回滚版本 1）
        migrator.down(&pool, 0).await.unwrap();

        // 验证：表被删除
        let pg = pool.as_postgres().unwrap();
        let (exists,): (bool,) = sqlx::query_as(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
        )
        .bind(&table)
        .fetch_one(pg)
        .await
        .unwrap();
        assert!(!exists, "table should be dropped after rollback");

        // 验证：版本记录被删除
        let records = pool.fetch_version_records(&vt).await.unwrap();
        assert!(records.is_empty(), "version records should be removed");

        let _ = pool.exec(&format!("DROP TABLE IF EXISTS {table}")).await;
        let _ = pool.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        pool.close().await;
    }

    #[tokio::test]
    async fn test_down_missing_script_returns_error() {
        let (pool, prefix) = setup().await;
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new()
            .with_version_table(&vt)
            .add(SqlMigration::new(1, "no_down", "SELECT 1", None::<&str>));

        migrator.up(&pool).await.unwrap();

        // down 应报错（无 down_sql）
        let result = migrator.down(&pool, 0).await;
        assert!(result.is_err(), "down without down_sql should error");

        let _ = pool.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        pool.close().await;
    }

    #[tokio::test]
    async fn test_no_migrations_registered() {
        let (pool, prefix) = setup().await;
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new().with_version_table(&vt);
        // 空 migrator 不报错
        migrator.up(&pool).await.unwrap();
        let status = migrator.status(&pool).await.unwrap();
        assert!(status.is_empty());

        let _ = pool.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        pool.close().await;
    }
}

// ===========================================================================
// SQLite 集成测试（内存模式，无需外部依赖）
// ===========================================================================

#[cfg(all(test, feature = "sqlite"))]
mod sqlite_tests {
    use super::*;
    use crate::test_utils;

    async fn pool() -> DbPool {
        test_utils::sqlite_pool().await
    }

    #[tokio::test]
    async fn test_apply_single_migration() {
        let p = pool().await;
        let prefix = test_utils::unique_prefix();
        let table = format!("{prefix}_users");
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new()
            .with_version_table(&vt)
            .add(SqlMigration::new(
                1,
                "create_users",
                format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT)"),
                Some(format!("DROP TABLE IF EXISTS {table}")),
            ));

        migrator.up(&p).await.unwrap();

        // 可插入数据
        p.exec(&format!("INSERT INTO {table} (name) VALUES ('test')"))
            .await
            .unwrap();

        // 版本记录存在
        let records = p.fetch_version_records(&vt).await.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].1, "create_users");

        let _ = p.exec(&format!("DROP TABLE IF EXISTS {table}")).await;
        let _ = p.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        p.close().await;
    }

    #[tokio::test]
    async fn test_migration_idempotent() {
        let p = pool().await;
        let prefix = test_utils::unique_prefix();
        let table = format!("{prefix}_i");
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new()
            .with_version_table(&vt)
            .add(SqlMigration::new(
                1,
                "init",
                format!("CREATE TABLE IF NOT EXISTS {table} (x INTEGER)"),
                Some(format!("DROP TABLE IF EXISTS {table}")),
            ));

        migrator.up(&p).await.unwrap();
        migrator.up(&p).await.unwrap(); // 幂等

        let records = p.fetch_version_records(&vt).await.unwrap();
        assert_eq!(records.len(), 1);

        let _ = p.exec(&format!("DROP TABLE IF EXISTS {table}")).await;
        let _ = p.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        p.close().await;
    }

    #[tokio::test]
    async fn test_migration_rollback() {
        let p = pool().await;
        let prefix = test_utils::unique_prefix();
        let table = format!("{prefix}_rb");
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new()
            .with_version_table(&vt)
            .add(SqlMigration::new(
                1,
                "m1",
                format!("CREATE TABLE {table} (x INTEGER)"),
                Some(format!("DROP TABLE IF EXISTS {table}")),
            ));

        migrator.up(&p).await.unwrap();
        migrator.down(&p, 0).await.unwrap();

        // 回滚后版本表无记录
        assert!(p.fetch_version_records(&vt).await.unwrap().is_empty());

        let _ = p.exec(&format!("DROP TABLE IF EXISTS {table}")).await;
        let _ = p.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        p.close().await;
    }

    #[tokio::test]
    async fn test_migration_status() {
        let p = pool().await;
        let prefix = test_utils::unique_prefix();
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new()
            .with_version_table(&vt)
            .add(SqlMigration::new(1, "m1", "SELECT 1", None::<&str>));

        let before = migrator.status(&p).await.unwrap();
        assert!(before[0].pending);

        migrator.up(&p).await.unwrap();
        let after = migrator.status(&p).await.unwrap();
        assert!(!after[0].pending);

        let _ = p.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        p.close().await;
    }
}

// ===========================================================================
// MySQL 集成测试
// ===========================================================================

#[cfg(all(test, feature = "mysql"))]
mod mysql_tests {
    use super::*;
    use crate::test_utils;

    fn require_mysql() -> String {
        test_utils::mysql_url().expect("SKIP: MYSQL_DATABASE_URL not set")
    }

    async fn setup() -> (DbPool, String) {
        let _url = require_mysql();
        let pool = test_utils::mysql_pool().await.unwrap();
        let prefix = test_utils::unique_prefix();
        (pool, prefix)
    }

    #[tokio::test]
    async fn test_apply_single_migration() {
        let (p, prefix) = setup().await;
        let table = format!("{prefix}_users");
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new()
            .with_version_table(&vt)
            .add(SqlMigration::new(
                1,
                "create_users",
                format!(
                    "CREATE TABLE {table} (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100))"
                ),
                Some(format!("DROP TABLE IF EXISTS {table}")),
            ));

        migrator.up(&p).await.unwrap();

        p.exec(&format!("INSERT INTO {table} (name) VALUES ('mysql_test')"))
            .await
            .unwrap();

        let records = p.fetch_version_records(&vt).await.unwrap();
        assert_eq!(records.len(), 1);

        let _ = p.exec(&format!("DROP TABLE IF EXISTS {table}")).await;
        let _ = p.exec(&format!("DROP TABLE IF EXISTS `{vt}`")).await;
        p.close().await;
    }

    #[tokio::test]
    async fn test_migration_rollback() {
        let (p, prefix) = setup().await;
        let table = format!("{prefix}_rb");
        let vt = format!("{prefix}_v");

        let migrator = Migrator::new()
            .with_version_table(&vt)
            .add(SqlMigration::new(
                1,
                "m1",
                format!("CREATE TABLE {table} (x INT)"),
                Some(format!("DROP TABLE IF EXISTS {table}")),
            ));

        migrator.up(&p).await.unwrap();
        migrator.down(&p, 0).await.unwrap();

        assert!(p.fetch_version_records(&vt).await.unwrap().is_empty());

        let _ = p.exec(&format!("DROP TABLE IF EXISTS {table}")).await;
        let _ = p.exec(&format!("DROP TABLE IF EXISTS `{vt}`")).await;
        p.close().await;
    }
}
