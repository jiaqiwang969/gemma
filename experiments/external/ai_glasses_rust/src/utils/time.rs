//! 时间工具

use std::time::{SystemTime, UNIX_EPOCH};

/// 获取当前 Unix 时间戳 (秒)
pub fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// 格式化时间戳为 MM:SS
pub fn format_mmss(seconds: f64) -> String {
    let mm = (seconds / 60.0) as u32;
    let ss = (seconds % 60.0) as u32;
    format!("{:02}:{:02}", mm, ss)
}

/// 格式化时间戳为 MM:SS.ms
pub fn format_mmss_ms(seconds: f64) -> String {
    let mm = (seconds / 60.0) as u32;
    let ss = (seconds % 60.0) as u32;
    let ms = ((seconds % 1.0) * 1000.0) as u32;
    format!("{:02}:{:02}.{:03}", mm, ss, ms)
}

/// 解析 MM:SS 格式
pub fn parse_mmss(s: &str) -> Option<f64> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return None;
    }

    let mm: f64 = parts[0].parse().ok()?;
    let ss: f64 = parts[1].parse().ok()?;

    Some(mm * 60.0 + ss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_mmss() {
        assert_eq!(format_mmss(0.0), "00:00");
        assert_eq!(format_mmss(65.5), "01:05");
        assert_eq!(format_mmss(3661.0), "61:01");
    }

    #[test]
    fn test_parse_mmss() {
        assert_eq!(parse_mmss("00:00"), Some(0.0));
        assert_eq!(parse_mmss("01:30"), Some(90.0));
        assert_eq!(parse_mmss("invalid"), None);
    }
}
