//! URL percent-decoding tool
//!
//! Decodes a percent-encoded string (RFC 3986) or an
//! `application/x-www-form-urlencoded` string.

use super::ToolResult;

/// A typed decode error so callers can distinguish error kinds without panics.
#[derive(Debug)]
pub enum DecodeError {
    /// The input contained a `%` not followed by two valid hex digits.
    InvalidPercentSequence { position: usize, fragment: String },
    /// The decoded bytes were not valid UTF-8.
    InvalidUtf8(std::string::FromUtf8Error),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::InvalidPercentSequence { position, fragment } => write!(
                f,
                "invalid percent sequence at position {position}: {fragment:?}"
            ),
            DecodeError::InvalidUtf8(err) => write!(f, "decoded bytes are not valid UTF-8: {err}"),
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DecodeError::InvalidUtf8(err) => Some(err),
            DecodeError::InvalidPercentSequence { .. } => None,
        }
    }
}

/// Parse a single hex nibble. Returns `None` for non-hex characters.
fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// Decode a percent-encoded string (RFC 3986).
///
/// Returns a `DecodeError` if the input contains an invalid `%` sequence
/// or if the resulting bytes are not valid UTF-8.
pub fn percent_decode(input: &str) -> Result<String, DecodeError> {
    let bytes = input.as_bytes();
    let mut decoded: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] == b'%' {
            // Need exactly two hex digits following the '%'.
            if i + 2 >= bytes.len() {
                return Err(DecodeError::InvalidPercentSequence {
                    position: i,
                    fragment: input[i..].to_string(),
                });
            }
            let hi =
                hex_nibble(bytes[i + 1]).ok_or_else(|| DecodeError::InvalidPercentSequence {
                    position: i,
                    fragment: input[i..i + 3].to_string(),
                })?;
            let lo =
                hex_nibble(bytes[i + 2]).ok_or_else(|| DecodeError::InvalidPercentSequence {
                    position: i,
                    fragment: input[i..i + 3].to_string(),
                })?;
            decoded.push((hi << 4) | lo);
            i += 3;
        } else {
            decoded.push(bytes[i]);
            i += 1;
        }
    }

    String::from_utf8(decoded).map_err(DecodeError::InvalidUtf8)
}

/// Decode an `application/x-www-form-urlencoded` string.
///
/// `+` characters are converted to spaces before percent-decoding.
pub fn form_decode(input: &str) -> Result<String, DecodeError> {
    let replaced = input.replace('+', " ");
    percent_decode(&replaced)
}

/// Run the URL decoding tool.
///
/// # Parameters
/// - `_input`: The encoded string to decode.
/// - `_decoding`: Decoding mode — `"form"` for `application/x-www-form-urlencoded`,
///   anything else selects standard percent-decoding (RFC 3986).
pub async fn run(_input: String, _decoding: String) -> ToolResult {
    let is_form = matches!(
        _decoding.to_ascii_lowercase().trim(),
        "form" | "form-encoded" | "application/x-www-form-urlencoded"
    );

    let decoded = if is_form {
        form_decode(&_input).map_err(|e| format!("form-decode error: {e}"))?
    } else {
        percent_decode(&_input).map_err(|e| format!("percent-decode error: {e}"))?
    };

    println!("{decoded}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percent_decode_simple() {
        assert_eq!(percent_decode("hello%20world").unwrap(), "hello world");
    }

    #[test]
    fn test_percent_decode_passthrough() {
        assert_eq!(percent_decode("hello").unwrap(), "hello");
    }

    #[test]
    fn test_percent_decode_uppercase_hex() {
        assert_eq!(percent_decode("a%3Db%26c%3Dd").unwrap(), "a=b&c=d");
    }

    #[test]
    fn test_percent_decode_lowercase_hex() {
        assert_eq!(percent_decode("a%3db%26c%3dd").unwrap(), "a=b&c=d");
    }

    #[test]
    fn test_percent_decode_invalid_sequence() {
        let result = percent_decode("bad%GG");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DecodeError::InvalidPercentSequence { .. }));
    }

    #[test]
    fn test_percent_decode_truncated_sequence() {
        let result = percent_decode("truncated%2");
        assert!(result.is_err());
    }

    #[test]
    fn test_form_decode_plus_to_space() {
        assert_eq!(form_decode("hello+world").unwrap(), "hello world");
    }

    #[test]
    fn test_form_decode_combined() {
        assert_eq!(form_decode("key+val%3Dvalue").unwrap(), "key val=value");
    }

    #[tokio::test]
    async fn test_run_percent() {
        run("hello%20world".to_string(), "percent".to_string())
            .await
            .expect("run should succeed");
    }

    #[tokio::test]
    async fn test_run_form() {
        run("hello+world".to_string(), "form".to_string())
            .await
            .expect("run should succeed");
    }

    #[tokio::test]
    async fn test_run_invalid_returns_error() {
        let result = run("bad%XX".to_string(), "percent".to_string()).await;
        assert!(result.is_err());
    }
}
