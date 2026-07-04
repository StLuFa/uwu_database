# uwu_database

统一数据访问层，封装 SQL 与缓存，并提供多租户与社区版/企业版功能开关。

## 架构

```
Database
├── DbPool         (PG / MySQL / SQLite，由 sqlx 驱动)
├── Cache          (Memory / Redis / Noop)
├── Features       (运行时 + 编译期开关)
└── VectorStore?   (可选，按 VectorConfig 初始化)

VectorStore        (统一向量库接口)
├── memory         (brute-force，开发/测试)
├── pgvector       (复用 PG Pool)
├── qdrant         (gRPC, qdrant-client)
└── lancedb        (嵌入式列存，upsert 通过 merge_insert 保证幂等)

TenantCtx          (个人/企业租户上下文)
Repository<T>      (业务 Repository 基 trait，含分页 Page / PagedResult)
cache_aside(...)   (Cache-Aside 读路径)
cache_invalidate(...)  (带租户前缀的缓存失效)
```

## Cargo features

SQL 后端：
- `postgres`（默认）
- `mysql`
- `sqlite`

缓存后端：
- `cache-memory`（默认）
- `cache-redis`

向量后端：
- `vector-memory` — 内存 brute-force
- `vector-pgvector` — 复用 PG，需数据库 `CREATE EXTENSION vector;`
- `vector-qdrant` — Qdrant gRPC
- `vector-lancedb` — LanceDB 嵌入式（编译时需 `protoc`）

版本：
- `community`（默认）
- `enterprise` = `multi-tenant` + `audit-log`

构建示例：

```bash
# 社区版 PG + 内存缓存
cargo build -p uwu_database

# 企业版 PG + Redis
cargo build -p uwu_database --no-default-features \
  --features "postgres cache-redis enterprise"

# 自托管 SQLite + 无缓存（嵌入式形态）
cargo build -p uwu_database --no-default-features \
  --features "sqlite cache-memory community"
```

## 使用

### 基本连接

```rust
use uwu_database::{Database, RuntimeConfig, TenantCtx};

let cfg: RuntimeConfig = toml::from_str(include_str!("config.toml"))?;
let db = Database::connect(&cfg).await?;

let ctx = TenantCtx::personal();

// 直接使用底层 sqlx Pool
let pg = db.pool.as_postgres()?;
let row = sqlx::query!("SELECT 1 as v").fetch_one(pg).await?;

// Cache-Aside
let v: String = uwu_database::repo::cache_aside(
    &db, &ctx, "user:1", Some(std::time::Duration::from_secs(60)),
    || async { Ok("alice".to_string()) },
).await?;

// 缓存失效
uwu_database::repo::cache_invalidate(&db, &ctx, "user:1").await?;

// 企业功能校验
db.features.require(uwu_database::FeatureKey::AuditLog)?;
```

### 数据库版本迁移（Schema Migrations）

内置版本管理系统，支持自动发现磁盘 SQL 文件、事务化执行与回滚。

#### 快速开始

```rust
use uwu_database::migrate::{Migrator, SqlMigration};

let db = Database::connect(&cfg).await?;

// 手动注册迁移
let m = Migrator::new()
    .add(SqlMigration::new(
        1, "create_users",
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
        Some("DROP TABLE users"),
    ))
    .add(SqlMigration::new(
        2, "add_email",
        "ALTER TABLE users ADD COLUMN email TEXT",
        Some("ALTER TABLE users DROP COLUMN email"),
    ));

// 应用所有待迁移
m.up(&db.pool).await?;

// 查看状态
let status = m.status(&db.pool).await?;
for r in &status {
    println!("v{} {} applied={:?}", r.version, r.name, r.applied_at);
}
```

#### 从目录自动发现

将迁移 SQL 文件放在 `migrations/` 目录下，命名格式：
`<版本>_<名称>.up.sql` / `<版本>_<名称>.down.sql`

```text
migrations/
  0001_create_users.up.sql
  0001_create_users.down.sql
  0002_add_email.up.sql
```

```rust
let m = Migrator::load_dir("migrations")?;
m.up(&db.pool).await?;
```

#### 嵌入宏（编译时打包 SQL）

```rust
use uwu_database::embedded_sql;

let m = Migrator::new()
    .add(embedded_sql!(
        1, "init",
        include_str!("migrations/0001_init.up.sql"),
        Some(include_str!("migrations/0001_init.down.sql"))
    ));
```

#### API 一览

| 方法 | 说明 |
|---|---|
| `Migrator::new()` | 创建空迁移管理器 |
| `.add(m)` | 注册一个迁移 |
| `.load_dir(path)?` | 从目录自动发现迁移文件 |
| `.up(pool).await?` | 应用所有待执行迁移 |
| `.up_to(pool, target).await?` | 应用到目标版本 |
| `.down(pool, target).await?` | 回滚到目标版本（不含）|
| `.status(pool).await?` | 查看迁移状态 |
| `Database::migrator()` | 获取新 Migrator 实例 |
| `Database::migrate_up(&m).await?` | 应用迁移（便捷方法）|

版本跟踪表：`_uwu_schema_version`（自动创建，含 version / name / applied_at / checksum 字段）。

### 含向量后端

```rust
use uwu_database::{Database, RuntimeConfig};

let cfg: RuntimeConfig = toml::from_str(include_str!("config.toml"))?;
// connect_with_vector 按 cfg.vector 初始化向量后端
let db = Database::connect_with_vector(&cfg).await?;

let vs = db.vector_store()?;
```

### 分页 Repository

```rust
use uwu_database::repo::{Page, PagedResult, Repository};

// 实现 Repository trait 后：
let page = Page::first(20);
let result: PagedResult<User> = user_repo.list(&ctx, page).await?;
println!("total: {}, has_next: {}", result.total, result.has_next());
```

## 向量库使用

```rust
use uwu_database::vector::{
    CollectionSpec, Distance, Query, Record, VectorStore,
    memory::MemoryVectorStore,
};

let store = MemoryVectorStore::new();
store.ensure_collection(CollectionSpec {
    name: "docs", dim: 384, distance: Distance::Cosine
}).await?;

store.upsert("docs", &[Record {
    id: "doc-1".into(),
    vector: vec![0.1; 384],
    metadata: [("kind".into(), serde_json::json!("page"))].into_iter().collect(),
}]).await?;

let hits = store.search("docs", Query {
    vector: &vec![0.1; 384],
    top_k: 5,
    filter: None,
}).await?;
```

切换到 pgvector / Qdrant / LanceDB 只需替换构造，业务代码不变。

## 配置示例（TOML）

```toml
[deploy]
mode = "self_hosted"
edition = "community"

[database]
backend = "postgres"
url = "postgres://user:pass@localhost/db"
max_connections = 20
max_lifetime_secs = 1800
application_name = "my_app"

[cache]
backend = "memory"
capacity = 50000

[vector]
backend = "memory"
# backend = "pgvector"  # 复用 PG Pool，需 CREATE EXTENSION vector;
# backend = "qdrant"
# url = "http://localhost:6334"
# backend = "lancedb"
# url = "./data/lancedb"
```

## 性能优化项（features）

- `vector-simd` — `cosine_similarity` 用 `wide::f32x8` 8 路 SIMD
- `vector-parallel` — `MemoryVectorStore::search` 用 rayon 并行打分
- `single-flight` — `cache_aside` 同 key 并发合并为单次回源（防缓存击穿）

内置硬性优化（无需开关）：

- `MemoryCache` 基于 **moka**：分段并发 + TinyLFU 淘汰，per-entry TTL 通过时间戳懒惰检查实现
- `MemoryVectorStore::search`：`BinaryHeap` 求 top-k，复杂度 `O(n log k)`
- `PgVectorStore::upsert`：每批 1000 条多值 `INSERT ... ON CONFLICT`，单事务
- `PgVectorStore` 支持按集合配置 `Distance`，并提供 `create_hnsw_index` / `create_ivfflat_index`
- `LanceDbVectorStore::upsert`：使用 `merge_insert` 保证真正的幂等 upsert 语义
- `LanceDbVectorStore` 按集合记录 `Distance` 并在 search 时正确转换 score
- `DbConfig` 暴露 `max_lifetime_secs` / `test_before_acquire` / `statement_cache_capacity` / `application_name`

### 调优建议

- **PG 连接池**：云端部署务必设 `max_lifetime_secs <= 1800`，避开 LB 静默断开
- **prepared statement**：`statement_cache_capacity` 默认 100，重复 SQL 多的服务可调到 500+
- **向量库**：>10 万行务必建 HNSW（pgvector）或 IVF_PQ（LanceDB），裸表查询会随数据量线性退化
- **Redis 缓存**：高 QPS 热点 key 启用 `single-flight`，避免缓存重建期击穿后端

## 集成

- 被 agent-memory 消费（可选，`feature = "database"` 启用 VectorStore 后端）

## 测试

```bash
cargo test -p uwu_database
```

当前测试：1 个。

## 部署形态

| 形态 | features | 数据库 | 缓存 | 多租户 |
|---|---|---|---|---|
| 个人版 / 嵌入式 | `community sqlite cache-memory` | SQLite | 内存 | 否 |
| 自托管标准版 | `community postgres cache-memory` | PG | 内存 | 否 |
| 自托管企业版 | `enterprise postgres cache-redis` | PG | Redis | 是 |
| SaaS | `enterprise postgres cache-redis` | PG (托管) | Redis | 是 |
