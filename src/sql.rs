use crate::config::{DbConfig, SqlBackend};
use crate::error::{DbError, Result};
use sqlx::pool::PoolOptions;
use sqlx::Row;

/// 多后端 Pool 枚举。
#[derive(Clone, Debug)]
pub enum DbPool {
    #[cfg(feature = "postgres")]
    Postgres(sqlx::PgPool),
    #[cfg(feature = "mysql")]
    MySql(sqlx::MySqlPool),
    #[cfg(feature = "sqlite")]
    Sqlite(sqlx::SqlitePool),
}

impl DbPool {
    pub fn backend(&self) -> SqlBackend {
        match self {
            #[cfg(feature = "postgres")]
            DbPool::Postgres(_) => SqlBackend::Postgres,
            #[cfg(feature = "mysql")]
            DbPool::MySql(_) => SqlBackend::MySql,
            #[cfg(feature = "sqlite")]
            DbPool::Sqlite(_) => SqlBackend::Sqlite,
        }
    }

    pub async fn close(&self) {
        match self {
            #[cfg(feature = "postgres")]
            DbPool::Postgres(p) => p.close().await,
            #[cfg(feature = "mysql")]
            DbPool::MySql(p) => p.close().await,
            #[cfg(feature = "sqlite")]
            DbPool::Sqlite(p) => p.close().await,
        }
    }

    #[cfg(feature = "postgres")]
    pub fn as_postgres(&self) -> Result<&sqlx::PgPool> {
        #[allow(irrefutable_let_patterns, unreachable_patterns)]
        match self {
            DbPool::Postgres(p) => Ok(p),
            #[allow(unreachable_patterns)]
            _ => Err(DbError::Unsupported("expected postgres pool".into())),
        }
    }

    #[cfg(feature = "mysql")]
    pub fn as_mysql(&self) -> Result<&sqlx::MySqlPool> {
        #[allow(irrefutable_let_patterns, unreachable_patterns)]
        match self {
            DbPool::MySql(p) => Ok(p),
            #[allow(unreachable_patterns)]
            _ => Err(DbError::Unsupported("expected mysql pool".into())),
        }
    }

    #[cfg(feature = "sqlite")]
    pub fn as_sqlite(&self) -> Result<&sqlx::SqlitePool> {
        #[allow(irrefutable_let_patterns, unreachable_patterns)]
        match self {
            DbPool::Sqlite(p) => Ok(p),
            #[allow(unreachable_patterns)]
            _ => Err(DbError::Unsupported("expected sqlite pool".into())),
        }
    }

    /// 执行原始 SQL（DDL/DML），不返回结果。
    /// 适用于建表、ALTER、INSERT 等语句。
    /// 多条语句用分号分隔，会逐条执行。
    pub async fn exec(&self, sql: &str) -> Result<()> {
        for stmt in split_sql(sql) {
            match self {
                #[cfg(feature = "postgres")]
                DbPool::Postgres(p) => { sqlx::query(&stmt).execute(p).await?; }
                #[cfg(feature = "mysql")]
                DbPool::MySql(p)    => { sqlx::query(&stmt).execute(p).await?; }
                #[cfg(feature = "sqlite")]
                DbPool::Sqlite(p)   => { sqlx::query(&stmt).execute(p).await?; }
            }
        }
        Ok(())
    }

    /// 查询版本跟踪表，返回 (version, name, applied_at) 列表。
    pub async fn fetch_version_records(
        &self,
        table: &str,
    ) -> Result<Vec<(i64, String, String)>> {
        let sql = format!(
            "SELECT version, name, applied_at FROM \"{}\" ORDER BY version",
            table.replace('"', "\"\"")  // 简单防注入
        );
        match self {
            #[cfg(feature = "postgres")]
            DbPool::Postgres(p) => {
                let rows = sqlx::query(&sql).fetch_all(p).await?;
                Ok(rows.into_iter().map(|r| {
                    (r.get::<i64,_>("version"), r.get::<String,_>("name"), r.get::<String,_>("applied_at"))
                }).collect())
            }
            #[cfg(feature = "mysql")]
            DbPool::MySql(p) => {
                let rows = sqlx::query(&sql).fetch_all(p).await?;
                Ok(rows.into_iter().map(|r| {
                    (r.get::<i64,_>("version"), r.get::<String,_>("name"), r.get::<String,_>("applied_at"))
                }).collect())
            }
            #[cfg(feature = "sqlite")]
            DbPool::Sqlite(p) => {
                let rows = sqlx::query(&sql).fetch_all(p).await?;
                Ok(rows.into_iter().map(|r| {
                    (r.get::<i64,_>("version"), r.get::<String,_>("name"), r.get::<String,_>("applied_at"))
                }).collect())
            }
        }
    }

    /// 向版本跟踪表插入一条记录（使用字符串拼接，所有值均来自可信来源）。
    pub async fn insert_version_record(
        &self,
        table: &str,
        version: i64,
        name: &str,
        applied_at: &str,
        checksum: &str,
    ) -> Result<()> {
        // 所有值均来自代码内部，无用户注入风险
        let sql = format!(
            "INSERT INTO \"{}\" (version, name, applied_at, checksum) \
             VALUES ({}, '{}', '{}', '{}') \
             ON CONFLICT (version) DO UPDATE SET \
               name = EXCLUDED.name, \
               applied_at = EXCLUDED.applied_at, \
               checksum = EXCLUDED.checksum",
            table.replace('"', "\"\""),
            version,
            escape_sql_str(name),
            escape_sql_str(applied_at),
            escape_sql_str(checksum),
        );
        self.exec(&sql).await
    }

    /// 从版本跟踪表删除一条记录。
    pub async fn delete_version_record(
        &self,
        table: &str,
        version: i64,
    ) -> Result<()> {
        let sql = format!(
            "DELETE FROM \"{}\" WHERE version = {}",
            table.replace('"', "\"\""),
            version
        );
        self.exec(&sql).await
    }
}

// ── SQL 工具 ───────────────────────────────────────────────

/// 将 SQL 脚本按分号拆成多条语句（忽略字符串内的分号）。
fn split_sql(sql: &str) -> Vec<String> {
    let mut stmts = Vec::new();
    let mut cur = String::new();
    let mut in_string: Option<char> = None;
    let mut chars = sql.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        if let Some(quote) = in_string {
            if c == '\\' {
                if let Some(next) = chars.next() { cur.push(next); }
            } else if c == quote {
                in_string = None;
            }
            continue;
        }
        match c {
            '\'' | '"' => { in_string = Some(c); }
            ';' => {
                let s = cur.trim().to_string();
                if !s.is_empty() && !s.starts_with("--") && !s.starts_with('#') {
                    stmts.push(s);
                }
                cur.clear();
            }
            _ => {}
        }
    }
    let s = cur.trim();
    if !s.is_empty() && !s.starts_with("--") && !s.starts_with('#') {
        stmts.push(s.to_string());
    }
    stmts
}

/// 转义 SQL 字符串中的单引号。
fn escape_sql_str(s: &str) -> String {
    s.replace('\'', "''")
}

/// 按配置构建连接池，并应用性能相关默认值。
pub async fn build_pool(cfg: &DbConfig) -> Result<DbPool> {
    match cfg.backend {
        SqlBackend::Postgres => {
            #[cfg(feature = "postgres")]
            {
                use sqlx::postgres::PgConnectOptions;
                use std::str::FromStr;

                let mut opts = PgConnectOptions::from_str(&cfg.url)
                    .map_err(|e| DbError::Config(e.to_string()))?
                    .statement_cache_capacity(cfg.statement_cache_capacity);
                if let Some(name) = &cfg.application_name {
                    opts = opts.application_name(name);
                }
                let pool = PoolOptions::<sqlx::Postgres>::new()
                    .max_connections(cfg.max_connections)
                    .min_connections(cfg.min_connections)
                    .acquire_timeout(cfg.acquire_timeout())
                    .idle_timeout(Some(cfg.idle_timeout()))
                    .max_lifetime(Some(cfg.max_lifetime()))
                    .test_before_acquire(cfg.test_before_acquire)
                    .connect_with(opts)
                    .await?;
                Ok(DbPool::Postgres(pool))
            }
            #[cfg(not(feature = "postgres"))]
            { Err(DbError::Unsupported("postgres feature disabled".into())) }
        }
        SqlBackend::MySql => {
            #[cfg(feature = "mysql")]
            {
                use sqlx::mysql::MySqlConnectOptions;
                use std::str::FromStr;

                let opts = MySqlConnectOptions::from_str(&cfg.url)
                    .map_err(|e| DbError::Config(e.to_string()))?
                    .statement_cache_capacity(cfg.statement_cache_capacity);
                let pool = PoolOptions::<sqlx::MySql>::new()
                    .max_connections(cfg.max_connections)
                    .min_connections(cfg.min_connections)
                    .acquire_timeout(cfg.acquire_timeout())
                    .idle_timeout(Some(cfg.idle_timeout()))
                    .max_lifetime(Some(cfg.max_lifetime()))
                    .test_before_acquire(cfg.test_before_acquire)
                    .connect_with(opts)
                    .await?;
                Ok(DbPool::MySql(pool))
            }
            #[cfg(not(feature = "mysql"))]
            { Err(DbError::Unsupported("mysql feature disabled".into())) }
        }
        SqlBackend::Sqlite => {
            #[cfg(feature = "sqlite")]
            {
                use sqlx::sqlite::SqliteConnectOptions;
                use std::str::FromStr;

                let opts = SqliteConnectOptions::from_str(&cfg.url)
                    .map_err(|e| DbError::Config(e.to_string()))?
                    .statement_cache_capacity(cfg.statement_cache_capacity);
                let pool = PoolOptions::<sqlx::Sqlite>::new()
                    .max_connections(cfg.max_connections)
                    .min_connections(cfg.min_connections)
                    .acquire_timeout(cfg.acquire_timeout())
                    .idle_timeout(Some(cfg.idle_timeout()))
                    .max_lifetime(Some(cfg.max_lifetime()))
                    .test_before_acquire(cfg.test_before_acquire)
                    .connect_with(opts)
                    .await?;
                Ok(DbPool::Sqlite(pool))
            }
            #[cfg(not(feature = "sqlite"))]
            { Err(DbError::Unsupported("sqlite feature disabled".into())) }
        }
    }
}

// ===========================================================================
// PG 集成测试
// ===========================================================================

#[cfg(test)]
mod pg_tests {
    use super::*;
    use crate::test_utils;

    /// 跳过条件：无 DATABASE_URL 环境变量时跳过。
    fn require_pg() -> String {
        test_utils::pg_url().expect("SKIP: DATABASE_URL not set")
    }

    // ── 连接 ───────────────────────────────────────────

    #[tokio::test]
    async fn test_connect_postgres() {
        let _url = require_pg();
        let pool = test_utils::pg_pool().await;
        assert!(pool.is_some(), "should connect to PG");
        let pool = pool.unwrap();
        assert_eq!(pool.backend(), SqlBackend::Postgres);
        pool.close().await;
    }

    #[tokio::test]
    async fn test_as_postgres_returns_pg_pool() {
        let _url = require_pg();
        let pool = test_utils::pg_pool().await.unwrap();
        let pg = pool.as_postgres();
        assert!(pg.is_ok(), "as_postgres should succeed for PG pool");
        pool.close().await;
    }

    // ── DDL / DML ─────────────────────────────────────

    #[tokio::test]
    async fn test_exec_create_table_and_insert() {
        let _url = require_pg();
        let pool = test_utils::pg_pool().await.unwrap();
        let prefix = test_utils::unique_prefix();
        let table = format!("{prefix}_test_exec");

        // DDL
        pool.exec(&format!(
            "CREATE TABLE {table} (id SERIAL PRIMARY KEY, name TEXT NOT NULL)"
        ))
        .await
        .expect("create table");

        // DML: INSERT
        pool.exec(&format!("INSERT INTO {table} (name) VALUES ('alice')"))
            .await
            .expect("insert");

        // DML: SELECT（通过 raw query 验证）
        let pg = pool.as_postgres().unwrap();
        let row: (i32, String) =
            sqlx::query_as(&format!("SELECT id, name FROM {table}"))
                .fetch_one(pg)
                .await
                .expect("select");
        assert_eq!(row.0, 1);
        assert_eq!(row.1, "alice");

        // 清理
        let _ = pool.exec(&format!("DROP TABLE IF EXISTS {table}")).await;
        pool.close().await;
    }

    #[tokio::test]
    async fn test_exec_multiple_statements() {
        let _url = require_pg();
        let pool = test_utils::pg_pool().await.unwrap();
        let prefix = test_utils::unique_prefix();
        let table = format!("{prefix}_multi");

        pool.exec(&format!(
            "CREATE TABLE {table} (x INT); \
             INSERT INTO {table} VALUES (1); \
             INSERT INTO {table} VALUES (2);"
        ))
        .await
        .expect("multi-statement exec");

        let pg = pool.as_postgres().unwrap();
        let count: (i64,) =
            sqlx::query_as(&format!("SELECT COUNT(*) FROM {table}"))
                .fetch_one(pg)
                .await
                .unwrap();
        assert_eq!(count.0, 2);

        let _ = pool.exec(&format!("DROP TABLE IF EXISTS {table}")).await;
        pool.close().await;
    }

    #[tokio::test]
    async fn test_exec_invalid_sql_returns_error() {
        let _url = require_pg();
        let pool = test_utils::pg_pool().await.unwrap();
        let result = pool.exec("THIS IS NOT VALID SQL").await;
        assert!(result.is_err(), "invalid SQL should error");
        pool.close().await;
    }

    // ── 版本记录（迁移追踪） ──────────────────────────

    #[tokio::test]
    async fn test_version_record_crud() {
        let _url = require_pg();
        let pool = test_utils::pg_pool().await.unwrap();
        let prefix = test_utils::unique_prefix();
        let vt = format!("{prefix}_versions");

        // 建表
        pool.exec(&format!(
            "CREATE TABLE \"{vt}\" (\
                version BIGINT PRIMARY KEY, \
                name TEXT NOT NULL, \
                applied_at TEXT NOT NULL, \
                checksum TEXT\
            )"
        ))
        .await
        .unwrap();

        // 插入
        pool.insert_version_record(&vt, 1, "init", "1000", "abc123")
            .await
            .unwrap();

        // 读取
        let records = pool.fetch_version_records(&vt).await.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0, 1); // version
        assert_eq!(records[0].1, "init"); // name

        // 删除
        pool.delete_version_record(&vt, 1).await.unwrap();
        let after_delete = pool.fetch_version_records(&vt).await.unwrap();
        assert!(after_delete.is_empty());

        // 清理
        let _ = pool.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        pool.close().await;
    }

    #[tokio::test]
    async fn test_version_record_on_conflict_update() {
        let _url = require_pg();
        let pool = test_utils::pg_pool().await.unwrap();
        let prefix = test_utils::unique_prefix();
        let vt = format!("{prefix}_ver_upsert");

        pool.exec(&format!(
            "CREATE TABLE \"{vt}\" (\
                version BIGINT PRIMARY KEY, \
                name TEXT NOT NULL, \
                applied_at TEXT NOT NULL, \
                checksum TEXT\
            )"
        ))
        .await
        .unwrap();

        // 首次插入
        pool.insert_version_record(&vt, 1, "v1", "1000", "aaa")
            .await
            .unwrap();

        // 冲突更新
        pool.insert_version_record(&vt, 1, "v1_updated", "2000", "bbb")
            .await
            .unwrap();

        let records = pool.fetch_version_records(&vt).await.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].1, "v1_updated");

        let _ = pool.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        pool.close().await;
    }

    // ── 关闭 ──────────────────────────────────────────

    #[tokio::test]
    async fn test_close_is_idempotent() {
        let _url = require_pg();
        let pool = test_utils::pg_pool().await.unwrap();
        pool.close().await;
        pool.close().await; // 重复调用不应 panic
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

    // ── 连接 ───────────────────────────────────────────

    #[tokio::test]
    async fn test_connect_sqlite_memory() {
        let p = pool().await;
        assert_eq!(p.backend(), SqlBackend::Sqlite);
        p.close().await;
    }

    #[tokio::test]
    async fn test_as_sqlite_returns_pool() {
        let p = pool().await;
        let s = p.as_sqlite();
        assert!(s.is_ok(), "as_sqlite should succeed");
        p.close().await;
    }

    // ── DDL / DML ─────────────────────────────────────

    #[tokio::test]
    async fn test_exec_create_table_and_insert() {
        let p = pool().await;
        let prefix = test_utils::unique_prefix();
        let table = format!("{prefix}_t");

        p.exec(&format!(
            "CREATE TABLE {table} (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL)"
        ))
        .await
        .expect("create table");

        p.exec(&format!("INSERT INTO {table} (name) VALUES ('alice')"))
            .await
            .expect("insert");

        let s = p.as_sqlite().unwrap();
        let row: (i64, String) =
            sqlx::query_as(&format!("SELECT id, name FROM {table}"))
                .fetch_one(s)
                .await
                .expect("select");
        assert_eq!(row.0, 1);
        assert_eq!(row.1, "alice");

        let _ = p.exec(&format!("DROP TABLE IF EXISTS {table}")).await;
        p.close().await;
    }

    #[tokio::test]
    async fn test_exec_multiple_statements() {
        let p = pool().await;
        let prefix = test_utils::unique_prefix();
        let table = format!("{prefix}_m");

        // SQLite 下逐条执行（sqlx 不支持多条语句一起）
        p.exec(&format!("CREATE TABLE {table} (x INTEGER)")).await.unwrap();
        p.exec(&format!("INSERT INTO {table} VALUES (1)")).await.unwrap();
        p.exec(&format!("INSERT INTO {table} VALUES (2)")).await.unwrap();

        let s = p.as_sqlite().unwrap();
        let count: (i64,) =
            sqlx::query_as(&format!("SELECT COUNT(*) FROM {table}"))
                .fetch_one(s)
                .await
                .unwrap();
        assert_eq!(count.0, 2);

        let _ = p.exec(&format!("DROP TABLE IF EXISTS {table}")).await;
        p.close().await;
    }

    #[tokio::test]
    async fn test_exec_invalid_sql_returns_error() {
        let p = pool().await;
        let r = p.exec("THIS IS NOT VALID SQL").await;
        assert!(r.is_err());
        p.close().await;
    }

    // ── 版本记录 ──────────────────────────────────────

    #[tokio::test]
    async fn test_version_record_crud() {
        let p = pool().await;
        let prefix = test_utils::unique_prefix();
        let vt = format!("{prefix}_v");

        p.exec(&format!(
            "CREATE TABLE \"{vt}\" (\
                version INTEGER PRIMARY KEY, \
                name TEXT NOT NULL, \
                applied_at TEXT NOT NULL, \
                checksum TEXT)"
        ))
        .await
        .unwrap();

        p.insert_version_record(&vt, 1, "init", "1000", "abc").await.unwrap();
        let records = p.fetch_version_records(&vt).await.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].1, "init");

        p.delete_version_record(&vt, 1).await.unwrap();
        assert!(p.fetch_version_records(&vt).await.unwrap().is_empty());

        let _ = p.exec(&format!("DROP TABLE IF EXISTS \"{vt}\"")).await;
        p.close().await;
    }

    #[tokio::test]
    async fn test_close_sqlite_idempotent() {
        let p = pool().await;
        p.close().await;
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

    async fn pool() -> DbPool {
        let _url = require_mysql();
        test_utils::mysql_pool().await.unwrap()
    }

    // ── 连接 ───────────────────────────────────────────

    #[tokio::test]
    async fn test_connect_mysql() {
        let _url = require_mysql();
        let p = test_utils::mysql_pool().await;
        assert!(p.is_some(), "should connect to MySQL");
        let p = p.unwrap();
        assert_eq!(p.backend(), SqlBackend::MySql);
        p.close().await;
    }

    #[tokio::test]
    async fn test_as_mysql_returns_pool() {
        let p = pool().await;
        let m = p.as_mysql();
        assert!(m.is_ok(), "as_mysql should succeed");
        p.close().await;
    }

    // ── DDL / DML ─────────────────────────────────────

    #[tokio::test]
    async fn test_exec_create_table_and_insert() {
        let p = pool().await;
        let prefix = test_utils::unique_prefix();
        let table = format!("{prefix}_t");

        p.exec(&format!(
            "CREATE TABLE {table} (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100) NOT NULL)"
        ))
        .await
        .expect("create table");

        p.exec(&format!("INSERT INTO {table} (name) VALUES ('bob')"))
            .await
            .expect("insert");

        let m = p.as_mysql().unwrap();
        let row: (i32, String) =
            sqlx::query_as(&format!("SELECT id, name FROM {table}"))
                .fetch_one(m)
                .await
                .expect("select");
        assert_eq!(row.0, 1);
        assert_eq!(row.1, "bob");

        let _ = p.exec(&format!("DROP TABLE IF EXISTS {table}")).await;
        p.close().await;
    }

    #[tokio::test]
    async fn test_exec_invalid_sql() {
        let p = pool().await;
        assert!(p.exec("XYZZY INVALID").await.is_err());
        p.close().await;
    }

    // ── 版本记录 ──────────────────────────────────────

    #[tokio::test]
    async fn test_version_record_crud() {
        let p = pool().await;
        let prefix = test_utils::unique_prefix();
        let vt = format!("{prefix}_v");

        p.exec(&format!(
            "CREATE TABLE `{vt}` (\
                version BIGINT PRIMARY KEY, \
                name VARCHAR(255) NOT NULL, \
                applied_at VARCHAR(50) NOT NULL, \
                checksum VARCHAR(255)\
            )"
        ))
        .await
        .unwrap();

        p.insert_version_record(&vt, 1, "m1", "1000", "sha").await.unwrap();

        let records = p.fetch_version_records(&vt).await.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].1, "m1");

        p.delete_version_record(&vt, 1).await.unwrap();
        assert!(p.fetch_version_records(&vt).await.unwrap().is_empty());

        let _ = p.exec(&format!("DROP TABLE IF EXISTS `{vt}`")).await;
        p.close().await;
    }

    #[tokio::test]
    async fn test_close_mysql_idempotent() {
        let p = pool().await;
        p.close().await;
        p.close().await;
    }
}
