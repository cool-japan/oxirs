//! UTF-8 validation and repair tool
//!
//! Validates whether a string or file is valid UTF-8, and optionally replaces
//! invalid byte sequences with the Unicode replacement character U+FFFD.

use super::ToolResult;
use std::io::Read;

/// Summary of a UTF-8 validation pass.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// `true` when every byte sequence is valid UTF-8.
    pub is_valid: bool,
    /// Number of invalid bytes encountered (zero when `is_valid` is `true`).
    pub invalid_byte_count: usize,
}

/// Read raw bytes from a file path.
fn read_bytes_from_file(path: &str) -> ToolResult<Vec<u8>> {
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("cannot open file {path:?}: {e}"))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|e| format!("cannot read file {path:?}: {e}"))?;
    Ok(buf)
}

/// Validate raw bytes as UTF-8.
///
/// Returns a `ValidationResult` describing whether the bytes are fully valid
/// and how many invalid bytes were found.
pub fn validate_bytes(bytes: &[u8]) -> ValidationResult {
    match std::str::from_utf8(bytes) {
        Ok(_) => ValidationResult {
            is_valid: true,
            invalid_byte_count: 0,
        },
        Err(_) => {
            // Count invalid bytes by scanning the lossy replacement.
            // `from_utf8_lossy` replaces each maximal invalid sequence with one
            // U+FFFD, so we count replacement characters in the lossy output
            // against the original byte count to get an approximation.
            let lossy = String::from_utf8_lossy(bytes);
            let replacement_count = lossy.chars().filter(|&c| c == '\u{FFFD}').count();
            // Each invalid run of bytes in the original maps to one replacement
            // character — this is the minimum possible invalid-byte count.
            ValidationResult {
                is_valid: false,
                invalid_byte_count: replacement_count,
            }
        }
    }
}

/// Replace invalid UTF-8 byte sequences with U+FFFD.
pub fn fix_bytes(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Run the UTF-8 utility tool.
///
/// # Parameters
/// - `_input`:    A raw string value **or** a file-system path (see `_file`).
/// - `_file`:     When `true`, treat `_input` as a file path and read its bytes.
/// - `_validate`: Check whether the input is valid UTF-8 and print the result.
/// - `_fix`:      Replace invalid byte sequences with U+FFFD and print the
///   repaired string.
///
/// When neither `_validate` nor `_fix` is set, the tool prints the input/file
/// content as-is (using lossless conversion so invalid bytes appear as U+FFFD).
pub async fn run(_input: String, _file: bool, _validate: bool, _fix: bool) -> ToolResult {
    let bytes: Vec<u8> = if _file {
        read_bytes_from_file(&_input)?
    } else {
        _input.into_bytes()
    };

    if _validate {
        let result = validate_bytes(&bytes);
        if result.is_valid {
            println!("Valid UTF-8");
        } else {
            println!(
                "Invalid UTF-8: {} invalid byte sequence(s) found",
                result.invalid_byte_count
            );
        }
    }

    if _fix {
        let fixed = fix_bytes(&bytes);
        println!("{fixed}");
    }

    // Default output when no specific mode is requested.
    if !_validate && !_fix {
        let output = String::from_utf8_lossy(&bytes);
        println!("{output}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_validate_valid_utf8() {
        let result = validate_bytes("hello, world!".as_bytes());
        assert!(result.is_valid);
        assert_eq!(result.invalid_byte_count, 0);
    }

    #[test]
    fn test_validate_invalid_utf8() {
        // 0xFF is never valid in UTF-8.
        let bytes: Vec<u8> = vec![b'h', b'i', 0xFF];
        let result = validate_bytes(&bytes);
        assert!(!result.is_valid);
        assert!(result.invalid_byte_count > 0);
    }

    #[test]
    fn test_fix_bytes_valid_input() {
        let fixed = fix_bytes("good".as_bytes());
        assert_eq!(fixed, "good");
    }

    #[test]
    fn test_fix_bytes_replaces_invalid() {
        let bytes: Vec<u8> = vec![b'a', 0xFF, b'b'];
        let fixed = fix_bytes(&bytes);
        assert!(fixed.contains('\u{FFFD}'));
        assert!(fixed.starts_with('a'));
        assert!(fixed.ends_with('b'));
    }

    #[tokio::test]
    async fn test_run_validate_valid() {
        run("hello".to_string(), false, true, false)
            .await
            .expect("should succeed");
    }

    #[tokio::test]
    async fn test_run_fix_mode() {
        run("good string".to_string(), false, false, true)
            .await
            .expect("should succeed");
    }

    #[tokio::test]
    async fn test_run_default_mode() {
        run("plain".to_string(), false, false, false)
            .await
            .expect("should succeed");
    }

    #[tokio::test]
    async fn test_run_from_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxirs_utf8_test.txt");
        {
            let mut f = std::fs::File::create(&path).expect("create temp file");
            f.write_all(b"file content").expect("write");
        }
        let path_str = path.to_string_lossy().into_owned();
        run(path_str, true, true, false)
            .await
            .expect("should read file successfully");
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_run_missing_file_returns_error() {
        let missing = std::env::temp_dir()
            .join(format!("oxirs_utf8_nonexistent_{}.txt", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let result = run(missing, true, true, false).await;
        assert!(result.is_err());
    }
}
