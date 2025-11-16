//! CLI Utility Functions
//!
//! Common utility functions for CLI operations, formatting, and user interaction.

use std::io::Write;
use std::path::Path;
use std::time::Duration;

/// Format duration in human-readable format
///
/// # Examples
/// - 0.001s -> "1.0ms"
/// - 0.5s -> "500ms"
/// - 1.5s -> "1.50s"
/// - 65s -> "1m 5s"
/// - 3661s -> "1h 1m 1s"
pub fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let millis = duration.subsec_millis();

    if total_secs == 0 {
        if millis == 0 {
            let micros = duration.as_micros();
            if micros < 1000 {
                format!("{}μs", micros)
            } else {
                format!("{:.1}ms", micros as f64 / 1000.0)
            }
        } else {
            format!("{}ms", millis)
        }
    } else if total_secs < 60 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if total_secs < 3600 {
        let mins = total_secs / 60;
        let secs = total_secs % 60;
        format!("{}m {}s", mins, secs)
    } else {
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        let secs = total_secs % 60;
        format!("{}h {}m {}s", hours, mins, secs)
    }
}

/// Format byte size in human-readable format
///
/// # Examples
/// - 512 -> "512 B"
/// - 1024 -> "1.0 KB"
/// - 1536 -> "1.5 KB"
/// - 1048576 -> "1.0 MB"
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];

    if bytes < 1024 {
        return format!("{} B", bytes);
    }

    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    if size >= 100.0 {
        format!("{:.0} {}", size, UNITS[unit_idx])
    } else if size >= 10.0 {
        format!("{:.1} {}", size, UNITS[unit_idx])
    } else {
        format!("{:.2} {}", size, UNITS[unit_idx])
    }
}

/// Format number with thousands separator
///
/// # Examples
/// - 1000 -> "1,000"
/// - 1234567 -> "1,234,567"
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();

    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(*c);
    }

    result
}

/// Truncate string to max length with ellipsis
pub fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        "...".to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Get file size in bytes
pub fn file_size(path: &Path) -> std::io::Result<u64> {
    let metadata = std::fs::metadata(path)?;
    Ok(metadata.len())
}

/// Calculate percentage
pub fn percentage(part: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        (part as f64 / total as f64) * 100.0
    }
}

/// Format percentage with appropriate precision
pub fn format_percentage(part: u64, total: u64) -> String {
    let pct = percentage(part, total);

    if pct >= 10.0 {
        format!("{:.1}%", pct)
    } else if pct >= 1.0 {
        format!("{:.2}%", pct)
    } else if pct > 0.0 {
        format!("{:.3}%", pct)
    } else {
        "0.0%".to_string()
    }
}

/// Print a progress bar
pub fn print_progress_bar(current: u64, total: u64, width: usize) {
    let pct = percentage(current, total);
    let filled = ((pct / 100.0) * width as f64) as usize;
    let empty = width.saturating_sub(filled);

    print!("\r[");
    for _ in 0..filled {
        print!("=");
    }
    if filled < width {
        print!(">");
    }
    for _ in 0..empty {
        print!(" ");
    }
    print!(
        "] {:.1}% ({}/{})",
        pct,
        format_number(current),
        format_number(total)
    );

    std::io::stdout().flush().unwrap_or(());
}

/// Clear current line in terminal
pub fn clear_line() {
    print!("\r\x1B[K");
    std::io::stdout().flush().unwrap_or(());
}

/// Check if output is to a terminal (supports colors)
pub fn is_terminal() -> bool {
    atty::is(atty::Stream::Stdout)
}

/// Get terminal width, default to 80 if not available
pub fn terminal_width() -> usize {
    terminal_size::terminal_size()
        .map(|(w, _)| w.0 as usize)
        .unwrap_or(80)
}

/// Wrap text to terminal width
pub fn wrap_text(text: &str, width: usize) -> Vec<String> {
    textwrap::wrap(text, width)
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

/// Format table header
pub fn format_table_header(columns: &[&str]) -> String {
    let mut result = String::new();
    result.push('┌');
    for (i, col) in columns.iter().enumerate() {
        let width = col.len() + 2;
        result.push_str(&"─".repeat(width));
        if i < columns.len() - 1 {
            result.push('┬');
        }
    }
    result.push_str("┐\n│");

    for (i, col) in columns.iter().enumerate() {
        result.push(' ');
        result.push_str(col);
        result.push(' ');
        if i < columns.len() - 1 {
            result.push('│');
        }
    }
    result.push_str("│\n├");

    for (i, col) in columns.iter().enumerate() {
        let width = col.len() + 2;
        result.push_str(&"─".repeat(width));
        if i < columns.len() - 1 {
            result.push('┼');
        }
    }
    result.push('┤');

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_millis(1)), "1ms");
        assert_eq!(format_duration(Duration::from_millis(500)), "500ms");
        assert_eq!(format_duration(Duration::from_secs(1)), "1.00s");
        assert_eq!(format_duration(Duration::from_secs(65)), "1m 5s");
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h 1m 1s");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1234567), "1,234,567");
        assert_eq!(format_number(999), "999");
    }

    #[test]
    fn test_truncate_str() {
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello world", 8), "hello...");
        assert_eq!(truncate_str("hi", 2), "hi");
    }

    #[test]
    fn test_percentage() {
        assert_eq!(percentage(50, 100), 50.0);
        assert_eq!(percentage(1, 4), 25.0);
        assert_eq!(percentage(0, 100), 0.0);
        assert_eq!(percentage(10, 0), 0.0); // Division by zero protection
    }

    #[test]
    fn test_format_percentage() {
        assert_eq!(format_percentage(50, 100), "50.0%");
        assert_eq!(format_percentage(1, 100), "1.00%");
        assert_eq!(format_percentage(1, 1000), "0.100%");
    }
}
