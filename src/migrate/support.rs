//! 迁移模块的内部工具函数。

/// 返回当前时间的可读字符串表示。
/// 使用 UNIX 时间戳（秒），简单且跨平台无额外依赖。
pub fn now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}", secs)
}
