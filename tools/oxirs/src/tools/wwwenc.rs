//! URL percent-encoding tool
//!
//! Encodes a string using percent-encoding (RFC 3986) or
//! `application/x-www-form-urlencoded` encoding.

use super::ToolResult;

/// Encoding mode selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncodingMode {
    /// Standard percent-encoding per RFC 3986.
    Percent,
    /// `application/x-www-form-urlencoded` (spaces become `+`).
    Form,
}

impl EncodingMode {
    fn from_str(s: &str) -> EncodingMode {
        match s.to_ascii_lowercase().trim() {
            "form" | "form-encoded" | "application/x-www-form-urlencoded" => EncodingMode::Form,
            _ => EncodingMode::Percent,
        }
    }
}

/// Percent-encode a byte using uppercase hex digits.
fn percent_encode_byte(byte: u8) -> String {
    format!("%{:02X}", byte)
}

/// Characters that are unreserved in RFC 3986 and must not be encoded.
fn is_unreserved(c: u8) -> bool {
    matches!(c, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~')
}

/// Encode `input` using standard percent-encoding (RFC 3986).
/// Every byte that is not an unreserved character is encoded.
fn percent_encode(input: &str) -> String {
    let mut output = String::with_capacity(input.len() * 3);
    for &byte in input.as_bytes() {
        if is_unreserved(byte) {
            output.push(byte as char);
        } else {
            output.push_str(&percent_encode_byte(byte));
        }
    }
    output
}

/// Encode `input` using `application/x-www-form-urlencoded` rules.
/// Spaces become `+`; all other non-unreserved characters are percent-encoded.
fn form_encode(input: &str) -> String {
    let mut output = String::with_capacity(input.len() * 3);
    for &byte in input.as_bytes() {
        if byte == b' ' {
            output.push('+');
        } else if is_unreserved(byte) {
            output.push(byte as char);
        } else {
            output.push_str(&percent_encode_byte(byte));
        }
    }
    output
}

/// Run the URL encoding tool.
///
/// # Parameters
/// - `_input`: The string to encode.
/// - `_encoding`: Encoding mode — `"form"` for `application/x-www-form-urlencoded`,
///   anything else selects standard percent-encoding (RFC 3986).
pub async fn run(_input: String, _encoding: String) -> ToolResult {
    let mode = EncodingMode::from_str(&_encoding);

    let encoded = match mode {
        EncodingMode::Form => form_encode(&_input),
        EncodingMode::Percent => percent_encode(&_input),
    };

    println!("{encoded}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percent_encode_simple() {
        assert_eq!(percent_encode("hello world"), "hello%20world");
    }

    #[test]
    fn test_percent_encode_unreserved_passthrough() {
        let input = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.~";
        assert_eq!(percent_encode(input), input);
    }

    #[test]
    fn test_percent_encode_special_chars() {
        assert_eq!(percent_encode("a=b&c=d"), "a%3Db%26c%3Dd");
    }

    #[test]
    fn test_form_encode_space_becomes_plus() {
        assert_eq!(form_encode("hello world"), "hello+world");
    }

    #[test]
    fn test_form_encode_special_chars() {
        assert_eq!(form_encode("a=b&c=d"), "a%3Db%26c%3Dd");
    }

    #[test]
    fn test_encoding_mode_from_str() {
        assert_eq!(EncodingMode::from_str("form"), EncodingMode::Form);
        assert_eq!(
            EncodingMode::from_str("application/x-www-form-urlencoded"),
            EncodingMode::Form
        );
        assert_eq!(EncodingMode::from_str("percent"), EncodingMode::Percent);
        assert_eq!(EncodingMode::from_str(""), EncodingMode::Percent);
    }

    #[tokio::test]
    async fn test_run_percent() {
        run("hello world".to_string(), "percent".to_string())
            .await
            .expect("run should succeed");
    }

    #[tokio::test]
    async fn test_run_form() {
        run("key=val ue".to_string(), "form".to_string())
            .await
            .expect("run should succeed");
    }
}
