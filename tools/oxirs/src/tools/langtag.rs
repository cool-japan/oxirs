//! BCP-47 language tag validation and normalization tool
//!
//! Uses the `oxilangtag` crate (RFC 5646 compliant) to validate and normalize
//! IETF BCP-47 language tags such as `en`, `zh-cmn-Hans-CN`, or `en-US`.

use super::ToolResult;
use oxilangtag::LanguageTag;

/// Validate a BCP-47 language tag and report whether it is well-formed.
///
/// Returns `Ok(true)` when valid, `Ok(false)` when the tag is not well-formed
/// (the parse error is printed rather than propagated so the tool can continue).
pub fn validate_tag(tag: &str) -> bool {
    LanguageTag::parse(tag).is_ok()
}

/// Parse and normalize a BCP-47 language tag into its canonical form.
///
/// Canonical form follows RFC 5646 case conventions:
/// - language subtag → lowercase (`en`)
/// - script subtag   → title case (`Hans`)
/// - region subtag   → uppercase (`CN`)
///
/// Returns the normalized tag string on success.
pub fn normalize_tag(tag: &str) -> Result<String, String> {
    LanguageTag::parse_and_normalize(tag)
        .map(|t| t.into_inner())
        .map_err(|e| format!("invalid BCP-47 language tag: {e}"))
}

/// Run the language tag tool.
///
/// # Parameters
/// - `_tag`:       The BCP-47 language tag string to process.
/// - `_validate`:  When `true`, check if the tag is well-formed and print the result.
/// - `_normalize`: When `true`, output the canonical BCP-47 form of the tag.
///
/// When neither flag is set the raw input tag is echoed unchanged.
pub async fn run(_tag: String, _validate: bool, _normalize: bool) -> ToolResult {
    if _validate {
        if validate_tag(&_tag) {
            println!("Valid BCP-47 language tag: {_tag}");
        } else {
            println!("Invalid BCP-47 language tag: {_tag}");
            return Err(format!("invalid BCP-47 language tag: {_tag}").into());
        }
    }

    if _normalize {
        let normalized = normalize_tag(&_tag)?;
        println!("Normalized: {normalized}");
    }

    if !_validate && !_normalize {
        println!("{_tag}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_simple_valid() {
        assert!(validate_tag("en"));
        assert!(validate_tag("fr"));
        assert!(validate_tag("zh"));
    }

    #[test]
    fn test_validate_complex_valid() {
        assert!(validate_tag("en-US"));
        assert!(validate_tag("zh-cmn-Hans-CN"));
        assert!(validate_tag("de-CH"));
    }

    #[test]
    fn test_validate_invalid() {
        // Numeric-only tags or obviously broken tags should fail.
        assert!(!validate_tag("123"));
        assert!(!validate_tag(""));
    }

    #[test]
    fn test_normalize_lowercase_input() {
        // "en-us" should become "en-US" (region uppercased).
        let result = normalize_tag("en-us").expect("should normalize");
        assert_eq!(result, "en-US");
    }

    #[test]
    fn test_normalize_script_title_case() {
        // "zh-hans-cn" → "zh-Hans-CN"
        let result = normalize_tag("zh-hans-cn").expect("should normalize");
        assert_eq!(result, "zh-Hans-CN");
    }

    #[test]
    fn test_normalize_already_canonical() {
        let result = normalize_tag("en-US").expect("should normalize");
        assert_eq!(result, "en-US");
    }

    #[test]
    fn test_normalize_invalid_returns_error() {
        assert!(normalize_tag("123").is_err());
    }

    #[tokio::test]
    async fn test_run_validate_valid() {
        run("en-US".to_string(), true, false)
            .await
            .expect("should succeed for valid tag");
    }

    #[tokio::test]
    async fn test_run_validate_invalid_returns_error() {
        let result = run("123".to_string(), true, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_normalize() {
        run("en-us".to_string(), false, true)
            .await
            .expect("should normalize successfully");
    }

    #[tokio::test]
    async fn test_run_both_flags() {
        run("zh-hans-cn".to_string(), true, true)
            .await
            .expect("should succeed with both flags");
    }

    #[tokio::test]
    async fn test_run_default_echo() {
        run("de-CH".to_string(), false, false)
            .await
            .expect("should echo tag");
    }
}
