//! RFC 3986 percent-encoding utilities.
//!
//! Pure-std, in-house replacement for the external `urlencoding` crate with
//! 1:1 API semantics so call sites translate directly:
//!
//! - [`percent_encode`] mirrors `urlencoding::encode`: every byte outside the
//!   RFC 3986 *unreserved* set (`A-Z a-z 0-9 - _ . ~`) is encoded as `%XX`
//!   with uppercase hexadecimal digits. Spaces become `%20` (never `+`).
//! - [`percent_decode`] mirrors `urlencoding::decode`: `%XX` sequences are
//!   decoded case-insensitively, malformed or truncated `%` runs are passed
//!   through unchanged, and an error is returned only when the decoded bytes
//!   are not valid UTF-8.
//! - [`percent_encode_strict`] mirrors
//!   `percent_encoding::NON_ALPHANUMERIC`: every byte that is not an ASCII
//!   alphanumeric is encoded, including the unreserved punctuation
//!   `- _ . ~` (so `_` becomes `%5F`). Used where form-style strict
//!   encoding is the established wire contract (e.g. OAuth2 query
//!   parameters).
//!
//! All functions return [`Cow::Borrowed`] when the input needs no changes,
//! avoiding allocation on the fast path.
//!
//! # Examples
//!
//! ```
//! use oxirs_core::encoding::{percent_encode, percent_decode};
//!
//! # fn main() -> Result<(), std::string::FromUtf8Error> {
//! let encoded = percent_encode("SELECT * WHERE { ?s ?p ?o }");
//! assert_eq!(
//!     encoded,
//!     "SELECT%20%2A%20WHERE%20%7B%20%3Fs%20%3Fp%20%3Fo%20%7D"
//! );
//! assert_eq!(percent_decode(&encoded)?, "SELECT * WHERE { ?s ?p ?o }");
//! # Ok(())
//! # }
//! ```

use std::borrow::Cow;
use std::string::FromUtf8Error;

/// Uppercase hexadecimal digits used by [`percent_encode`].
const HEX_UPPER: &[u8; 16] = b"0123456789ABCDEF";

/// Returns `true` if the byte belongs to the RFC 3986 *unreserved* set:
/// `ALPHA / DIGIT / "-" / "." / "_" / "~"`.
#[inline]
const fn is_unreserved(byte: u8) -> bool {
    matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~')
}

/// Returns `true` if the byte is an ASCII alphanumeric character
/// (`ALPHA / DIGIT`), the set kept literal by [`percent_encode_strict`].
#[inline]
const fn is_ascii_alphanumeric_byte(byte: u8) -> bool {
    matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9')
}

/// Converts an ASCII hexadecimal digit (case-insensitive) to its value.
#[inline]
const fn from_hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

/// Shared percent-encoding loop: bytes for which `keep` returns `true` are
/// copied verbatim, every other byte becomes `%XX` with uppercase hex.
///
/// `keep` must only accept ASCII bytes so that the unchanged prefix is a
/// valid char boundary. Returns [`Cow::Borrowed`] when no byte requires
/// encoding.
fn encode_with(input: &str, keep: fn(u8) -> bool) -> Cow<'_, str> {
    let bytes = input.as_bytes();

    // Fast path: count the leading run of kept bytes.
    let unchanged_prefix = bytes.iter().take_while(|&&b| keep(b)).count();
    if unchanged_prefix == bytes.len() {
        return Cow::Borrowed(input);
    }

    let mut encoded = String::with_capacity(bytes.len() + 2 * (bytes.len() - unchanged_prefix));
    // All bytes in the prefix are ASCII, so this index is a char boundary.
    encoded.push_str(&input[..unchanged_prefix]);

    for &byte in &bytes[unchanged_prefix..] {
        if keep(byte) {
            encoded.push(byte as char);
        } else {
            encoded.push('%');
            encoded.push(HEX_UPPER[(byte >> 4) as usize] as char);
            encoded.push(HEX_UPPER[(byte & 0x0F) as usize] as char);
        }
    }

    Cow::Owned(encoded)
}

/// Percent-encodes a string per RFC 3986.
///
/// Every byte that is **not** in the unreserved set
/// (`A-Z a-z 0-9 - _ . ~`) is replaced by `%XX` where `XX` is the byte value
/// in uppercase hexadecimal. Multi-byte UTF-8 characters are encoded byte by
/// byte. Returns [`Cow::Borrowed`] when no byte requires encoding.
///
/// This is a drop-in replacement for `urlencoding::encode`.
///
/// # Examples
///
/// ```
/// use std::borrow::Cow;
/// use oxirs_core::encoding::percent_encode;
///
/// // Spaces become %20 (never `+`), reserved characters are escaped.
/// assert_eq!(percent_encode("a b/c?d#e"), "a%20b%2Fc%3Fd%23e");
///
/// // Multi-byte UTF-8 is encoded byte by byte with uppercase hex.
/// assert_eq!(percent_encode("日本"), "%E6%97%A5%E6%9C%AC");
///
/// // Unreserved input is returned without allocation.
/// assert!(matches!(percent_encode("AZaz09-_.~"), Cow::Borrowed(_)));
/// ```
pub fn percent_encode(input: &str) -> Cow<'_, str> {
    encode_with(input, is_unreserved)
}

/// Percent-encodes a string, escaping every byte that is not an ASCII
/// alphanumeric character (`A-Z a-z 0-9`).
///
/// This is stricter than [`percent_encode`]: the RFC 3986 unreserved
/// punctuation `- _ . ~` is escaped as well (`_` becomes `%5F`), matching
/// the byte set of `percent_encoding::NON_ALPHANUMERIC`. Use it where the
/// established wire contract expects form-style strict encoding, such as
/// OAuth2/OIDC authorization-URL query parameters. Returns
/// [`Cow::Borrowed`] when no byte requires encoding.
///
/// # Examples
///
/// ```
/// use std::borrow::Cow;
/// use oxirs_core::encoding::percent_encode_strict;
///
/// // Unreserved punctuation is escaped, unlike `percent_encode`.
/// assert_eq!(percent_encode_strict("test_client_id"), "test%5Fclient%5Fid");
/// assert_eq!(percent_encode_strict("a-b.c~d"), "a%2Db%2Ec%7Ed");
///
/// // Spaces become %20 (never `+`).
/// assert_eq!(percent_encode_strict("openid profile"), "openid%20profile");
///
/// // Purely alphanumeric input is returned without allocation.
/// assert!(matches!(percent_encode_strict("AZaz09"), Cow::Borrowed(_)));
/// ```
pub fn percent_encode_strict(input: &str) -> Cow<'_, str> {
    encode_with(input, is_ascii_alphanumeric_byte)
}

/// Decodes percent-encoded bytes, passing malformed `%` runs through
/// unchanged.
///
/// Returns [`Cow::Borrowed`] when the input contains no `%` byte at all
/// (mirroring `urlencoding::decode_binary`).
fn percent_decode_bytes(data: &[u8]) -> Cow<'_, [u8]> {
    // Fast path: no '%' means nothing can change.
    let unchanged_prefix = data.iter().take_while(|&&b| b != b'%').count();
    if unchanged_prefix == data.len() {
        return Cow::Borrowed(data);
    }

    let mut decoded = Vec::with_capacity(data.len());
    decoded.extend_from_slice(&data[..unchanged_prefix]);

    let mut index = unchanged_prefix;
    while index < data.len() {
        let byte = data[index];
        if byte != b'%' {
            decoded.push(byte);
            index += 1;
            continue;
        }

        match (data.get(index + 1).copied(), data.get(index + 2).copied()) {
            (Some(first), Some(second)) => match from_hex_digit(first) {
                Some(high) => match from_hex_digit(second) {
                    Some(low) => {
                        decoded.push((high << 4) | low);
                        index += 3;
                    }
                    None => {
                        // "%X?" where ? is not hex: emit '%' and the first
                        // digit; rescan from the second byte (it may be '%').
                        decoded.push(b'%');
                        decoded.push(first);
                        index += 2;
                    }
                },
                None => {
                    // "%?" where ? is not hex: emit '%' alone and rescan
                    // from the next byte (it may itself start a sequence).
                    decoded.push(b'%');
                    index += 1;
                }
            },
            _ => {
                // Truncated "%" or "%X" at the end: pass through verbatim.
                decoded.extend_from_slice(&data[index..]);
                break;
            }
        }
    }

    Cow::Owned(decoded)
}

/// Decodes a percent-encoded string per RFC 3986.
///
/// `%XX` sequences are decoded with case-insensitive hexadecimal digits.
/// Malformed or truncated sequences (`%`, `%2`, `%ZZ`, ...) are passed
/// through unchanged instead of producing an error; an error is returned
/// only when the decoded byte sequence is not valid UTF-8.
///
/// Returns [`Cow::Borrowed`] when the input contains no `%` character.
///
/// This is a drop-in replacement for `urlencoding::decode`.
///
/// # Errors
///
/// Returns [`FromUtf8Error`] when the decoded bytes are not valid UTF-8
/// (for example `"%FF"`).
///
/// # Examples
///
/// ```
/// use oxirs_core::encoding::percent_decode;
///
/// # fn main() -> Result<(), std::string::FromUtf8Error> {
/// assert_eq!(percent_decode("hello%20world")?, "hello world");
/// // Hex digits are case-insensitive.
/// assert_eq!(percent_decode("%e6%97%A5")?, "日");
/// // Malformed sequences pass through unchanged.
/// assert_eq!(percent_decode("100%")?, "100%");
/// // Invalid UTF-8 in the decoded output is the only error case.
/// assert!(percent_decode("%FF").is_err());
/// # Ok(())
/// # }
/// ```
pub fn percent_decode(input: &str) -> Result<Cow<'_, str>, FromUtf8Error> {
    match percent_decode_bytes(input.as_bytes()) {
        Cow::Borrowed(_) => Ok(Cow::Borrowed(input)),
        Cow::Owned(bytes) => Ok(Cow::Owned(String::from_utf8(bytes)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_empty_string_is_borrowed() {
        let encoded = percent_encode("");
        assert_eq!(encoded, "");
        assert!(matches!(encoded, Cow::Borrowed(_)));
    }

    #[test]
    fn encode_unreserved_is_borrowed_fast_path() {
        let input = "ABCXYZabcxyz0189-_.~";
        let encoded = percent_encode(input);
        assert_eq!(encoded, input);
        assert!(matches!(encoded, Cow::Borrowed(_)));
    }

    #[test]
    fn encode_space_is_percent20_not_plus() {
        assert_eq!(percent_encode("hello world"), "hello%20world");
        assert!(!percent_encode("hello world").contains('+'));
    }

    #[test]
    fn encode_reserved_characters() {
        assert_eq!(percent_encode("/"), "%2F");
        assert_eq!(percent_encode("?"), "%3F");
        assert_eq!(percent_encode("#"), "%23");
        assert_eq!(percent_encode("&"), "%26");
        assert_eq!(percent_encode("="), "%3D");
        assert_eq!(
            percent_encode("http://example.org/a?b=c&d=e#f"),
            "http%3A%2F%2Fexample.org%2Fa%3Fb%3Dc%26d%3De%23f"
        );
    }

    #[test]
    fn encode_percent_literal() {
        assert_eq!(percent_encode("100%"), "100%25");
        assert_eq!(percent_encode("%"), "%25");
    }

    #[test]
    fn encode_multibyte_utf8_uppercase_hex() {
        // "日本語" is 9 UTF-8 bytes; every one must be encoded uppercase.
        assert_eq!(percent_encode("日本語"), "%E6%97%A5%E6%9C%AC%E8%AA%9E");
        assert_eq!(percent_encode("ä"), "%C3%A4");
        assert_eq!(percent_encode("🦀"), "%F0%9F%A6%80");
    }

    #[test]
    fn encode_mixed_prefix_keeps_unreserved_run() {
        let encoded = percent_encode("abc def");
        assert_eq!(encoded, "abc%20def");
        assert!(matches!(encoded, Cow::Owned(_)));
    }

    #[test]
    fn strict_encode_alphanumeric_is_borrowed_fast_path() {
        let input = "ABCXYZabcxyz0189";
        let encoded = percent_encode_strict(input);
        assert_eq!(encoded, input);
        assert!(matches!(encoded, Cow::Borrowed(_)));
    }

    #[test]
    fn strict_encode_escapes_unreserved_punctuation() {
        // The four RFC 3986 unreserved punctuation bytes are escaped too.
        assert_eq!(percent_encode_strict("-"), "%2D");
        assert_eq!(percent_encode_strict("_"), "%5F");
        assert_eq!(percent_encode_strict("."), "%2E");
        assert_eq!(percent_encode_strict("~"), "%7E");
        assert_eq!(
            percent_encode_strict("test_client_id"),
            "test%5Fclient%5Fid"
        );
    }

    #[test]
    fn strict_encode_space_and_reserved_match_rfc_variant() {
        assert_eq!(percent_encode_strict("hello world"), "hello%20world");
        assert_eq!(
            percent_encode_strict("a/b?c#d&e=f"),
            "a%2Fb%3Fc%23d%26e%3Df"
        );
        assert_eq!(percent_encode_strict("100%"), "100%25");
    }

    #[test]
    fn strict_encode_multibyte_utf8_uppercase_hex() {
        assert_eq!(percent_encode_strict("日本"), "%E6%97%A5%E6%9C%AC");
        assert_eq!(percent_encode_strict("ä"), "%C3%A4");
    }

    #[test]
    fn strict_encode_round_trips_through_percent_decode() {
        let samples = [
            "",
            "plain",
            "test_client_id",
            "openid profile email",
            "code-challenge_~.value",
            "日本語のテキスト",
        ];
        for sample in samples {
            let encoded = percent_encode_strict(sample);
            let decoded = percent_decode(&encoded).expect("strict round trip decodes");
            assert_eq!(decoded, sample, "strict round trip failed for {sample:?}");
        }
    }

    #[test]
    fn strict_encoding_is_superset_of_rfc_encoding() {
        // Every byte escaped by `percent_encode` is escaped by the strict
        // variant; the strict variant only adds the unreserved punctuation.
        let input = "AZaz09-_.~ /?#&=%";
        let rfc = percent_encode(input);
        let strict = percent_encode_strict(input);
        assert_eq!(rfc, "AZaz09-_.~%20%2F%3F%23%26%3D%25");
        assert_eq!(strict, "AZaz09%2D%5F%2E%7E%20%2F%3F%23%26%3D%25");
    }

    #[test]
    fn decode_empty_string_is_borrowed() {
        let decoded = percent_decode("").expect("empty input decodes");
        assert_eq!(decoded, "");
        assert!(matches!(decoded, Cow::Borrowed(_)));
    }

    #[test]
    fn decode_without_percent_is_borrowed_fast_path() {
        let decoded = percent_decode("plain text!").expect("plain input decodes");
        assert_eq!(decoded, "plain text!");
        assert!(matches!(decoded, Cow::Borrowed(_)));
    }

    #[test]
    fn decode_basic_sequences() {
        assert_eq!(
            percent_decode("hello%20world").expect("valid input"),
            "hello world"
        );
        assert_eq!(
            percent_decode("%2F%3F%23%26%3D").expect("valid input"),
            "/?#&="
        );
        assert_eq!(percent_decode("%25").expect("valid input"), "%");
    }

    #[test]
    fn decode_hex_is_case_insensitive() {
        assert_eq!(percent_decode("%2f").expect("valid input"), "/");
        assert_eq!(percent_decode("%c3%a4").expect("valid input"), "ä");
        assert_eq!(percent_decode("%C3%A4").expect("valid input"), "ä");
        assert_eq!(percent_decode("%e6%97%A5").expect("valid input"), "日");
    }

    #[test]
    fn decode_malformed_sequences_pass_through() {
        // Lone '%' at the end.
        assert_eq!(percent_decode("%").expect("malformed passes through"), "%");
        assert_eq!(
            percent_decode("100%").expect("malformed passes through"),
            "100%"
        );
        // Truncated "%X" at the end.
        assert_eq!(
            percent_decode("%2").expect("malformed passes through"),
            "%2"
        );
        // Non-hex digits.
        assert_eq!(
            percent_decode("%ZZ").expect("malformed passes through"),
            "%ZZ"
        );
        // First digit valid, second invalid, then a valid sequence resumes.
        assert_eq!(
            percent_decode("%2%41").expect("malformed passes through"),
            "%2A"
        );
        // '%' immediately followed by a valid sequence.
        assert_eq!(
            percent_decode("%%41").expect("malformed passes through"),
            "%A"
        );
    }

    #[test]
    fn decode_invalid_utf8_errors() {
        assert!(percent_decode("%FF").is_err());
        // Lone continuation byte.
        assert!(percent_decode("%80").is_err());
        // Truncated multi-byte sequence (first byte of "ä" only).
        assert!(percent_decode("%C3").is_err());
    }

    #[test]
    fn round_trip_ascii_and_unicode() {
        let samples = [
            "",
            "plain",
            "hello world",
            "a/b?c#d&e=f",
            "100% sure",
            "日本語のテキスト",
            "emoji 🦀 crab",
            "tab\tnewline\nquote\"",
            "AZaz09-_.~",
        ];
        for sample in samples {
            let encoded = percent_encode(sample);
            let decoded = percent_decode(&encoded).expect("round trip decodes");
            assert_eq!(decoded, sample, "round trip failed for {sample:?}");
        }
    }

    #[test]
    fn encode_matches_sparql_encode_for_uri_semantics() {
        // SPARQL ENCODE_FOR_URI("Los Angeles") = "Los%20Angeles"
        assert_eq!(percent_encode("Los Angeles"), "Los%20Angeles");
    }
}
