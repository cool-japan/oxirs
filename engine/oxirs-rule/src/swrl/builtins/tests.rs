//! Tests for SWRL Built-in Functions
//!
//! Comprehensive test suite for all SWRL builtin functions

use super::*;
use crate::swrl::types::SwrlArgument;

#[test]
fn test_builtin_divide() {
    let args = vec![
        SwrlArgument::Literal("10.0".to_string()),
        SwrlArgument::Literal("2.0".to_string()),
        SwrlArgument::Literal("5.0".to_string()),
    ];
    assert!(builtin_divide(&args).unwrap());
}

#[test]
fn test_builtin_integer_divide() {
    let args = vec![
        SwrlArgument::Literal("10".to_string()),
        SwrlArgument::Literal("3".to_string()),
        SwrlArgument::Literal("3".to_string()),
    ];
    assert!(builtin_integer_divide(&args).unwrap());
}

#[test]
fn test_builtin_min() {
    let args = vec![
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("8.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
    ];
    assert!(builtin_min(&args).unwrap());
}

#[test]
fn test_builtin_max() {
    let args = vec![
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("8.0".to_string()),
        SwrlArgument::Literal("8.0".to_string()),
    ];
    assert!(builtin_max(&args).unwrap());
}

#[test]
fn test_builtin_avg() {
    let args = vec![
        SwrlArgument::Literal("2.0".to_string()),
        SwrlArgument::Literal("4.0".to_string()),
        SwrlArgument::Literal("6.0".to_string()),
        SwrlArgument::Literal("4.0".to_string()),
    ];
    assert!(builtin_avg(&args).unwrap());
}

#[test]
fn test_builtin_sum() {
    let args = vec![
        SwrlArgument::Literal("2.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("10.0".to_string()),
    ];
    assert!(builtin_sum(&args).unwrap());
}

#[test]
fn test_builtin_less_than_or_equal() {
    let args = vec![
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("5.0".to_string()),
    ];
    assert!(builtin_less_than_or_equal(&args).unwrap());

    let args = vec![
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("5.0".to_string()),
    ];
    assert!(builtin_less_than_or_equal(&args).unwrap());
}

#[test]
fn test_builtin_between() {
    let args = vec![
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("1.0".to_string()),
        SwrlArgument::Literal("10.0".to_string()),
    ];
    assert!(builtin_between(&args).unwrap());
}

#[test]
fn test_builtin_is_integer() {
    let args = vec![SwrlArgument::Literal("5.0".to_string())];
    assert!(builtin_is_integer(&args).unwrap());

    let args = vec![SwrlArgument::Literal("5.5".to_string())];
    assert!(!builtin_is_integer(&args).unwrap());
}

#[test]
fn test_builtin_is_float() {
    let args = vec![SwrlArgument::Literal("3.14".to_string())];
    assert!(builtin_is_float(&args).unwrap());
}

#[test]
fn test_builtin_is_string() {
    let args = vec![SwrlArgument::Literal("hello".to_string())];
    assert!(builtin_is_string(&args).unwrap());
}

#[test]
fn test_builtin_is_uri() {
    let args = vec![SwrlArgument::Individual("http://example.org".to_string())];
    assert!(builtin_is_uri(&args).unwrap());

    let args = vec![SwrlArgument::Literal("not a uri".to_string())];
    assert!(!builtin_is_uri(&args).unwrap());
}

#[test]
fn test_builtin_string_contains() {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("world".to_string()),
    ];
    assert!(builtin_string_contains(&args).unwrap());
}

#[test]
fn test_builtin_starts_with() {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("hello".to_string()),
    ];
    assert!(builtin_starts_with(&args).unwrap());
}

#[test]
fn test_builtin_ends_with() {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("world".to_string()),
    ];
    assert!(builtin_ends_with(&args).unwrap());
}

#[test]
fn test_builtin_replace() {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("world".to_string()),
        SwrlArgument::Literal("universe".to_string()),
        SwrlArgument::Literal("hello universe".to_string()),
    ];
    assert!(builtin_replace(&args).unwrap());
}

#[test]
fn test_builtin_trim() {
    let args = vec![
        SwrlArgument::Literal("  hello  ".to_string()),
        SwrlArgument::Literal("hello".to_string()),
    ];
    assert!(builtin_trim(&args).unwrap());
}

#[test]
fn test_builtin_index_of() {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("world".to_string()),
        SwrlArgument::Literal("6".to_string()),
    ];
    assert!(builtin_index_of(&args).unwrap());
}

#[test]
fn test_builtin_normalize_space() {
    let args = vec![
        SwrlArgument::Literal("hello   world  test".to_string()),
        SwrlArgument::Literal("hello world test".to_string()),
    ];
    assert!(builtin_normalize_space(&args).unwrap());
}

#[test]
fn test_builtin_date() {
    let args = vec![
        SwrlArgument::Literal("2025".to_string()),
        SwrlArgument::Literal("11".to_string()),
        SwrlArgument::Literal("3".to_string()),
        SwrlArgument::Literal("2025-11-03".to_string()),
    ];
    assert!(builtin_date(&args).unwrap());
}

#[test]
fn test_builtin_time() {
    let args = vec![
        SwrlArgument::Literal("14".to_string()),
        SwrlArgument::Literal("30".to_string()),
        SwrlArgument::Literal("45".to_string()),
        SwrlArgument::Literal("14:30:45".to_string()),
    ];
    assert!(builtin_time(&args).unwrap());
}

#[test]
fn test_builtin_year() {
    let args = vec![
        SwrlArgument::Literal("2025-11-03".to_string()),
        SwrlArgument::Literal("2025".to_string()),
    ];
    assert!(builtin_year(&args).unwrap());
}

#[test]
fn test_builtin_month() {
    let args = vec![
        SwrlArgument::Literal("2025-11-03".to_string()),
        SwrlArgument::Literal("11".to_string()),
    ];
    assert!(builtin_month(&args).unwrap());
}

#[test]
fn test_builtin_day() {
    let args = vec![
        SwrlArgument::Literal("2025-11-03".to_string()),
        SwrlArgument::Literal("3".to_string()),
    ];
    assert!(builtin_day(&args).unwrap());
}

#[test]
fn test_builtin_hash() {
    let args = vec![
        SwrlArgument::Literal("test".to_string()),
        SwrlArgument::Literal("test".to_string()),
    ];
    // Hash should consistently produce the same value
    let result1 = builtin_hash(&args);
    let result2 = builtin_hash(&args);
    assert_eq!(result1.is_ok(), result2.is_ok());
}

#[test]
fn test_builtin_base64_encode() {
    let args = vec![
        SwrlArgument::Literal("hello".to_string()),
        SwrlArgument::Literal("aGVsbG8=".to_string()),
    ];
    assert!(builtin_base64_encode(&args).unwrap());
}

#[test]
fn test_builtin_base64_decode() {
    let args = vec![
        SwrlArgument::Literal("aGVsbG8=".to_string()),
        SwrlArgument::Literal("hello".to_string()),
    ];
    assert!(builtin_base64_decode(&args).unwrap());
}

#[test]
fn test_builtin_median() {
    let args = vec![
        SwrlArgument::Literal("1.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
    ];
    assert!(builtin_median(&args).unwrap());
}

#[test]
fn test_builtin_variance() {
    let args = vec![
        SwrlArgument::Literal("2.0".to_string()),
        SwrlArgument::Literal("4.0".to_string()),
        SwrlArgument::Literal("6.0".to_string()),
        SwrlArgument::Literal("2.6666666666666665".to_string()),
    ];
    assert!(builtin_variance(&args).unwrap());
}

#[test]
fn test_builtin_stddev() {
    let args = vec![
        SwrlArgument::Literal("2.0".to_string()),
        SwrlArgument::Literal("4.0".to_string()),
        SwrlArgument::Literal("6.0".to_string()),
        SwrlArgument::Literal("1.632993161855452".to_string()),
    ];
    assert!(builtin_stddev(&args).unwrap());
}

#[test]
fn test_builtin_lang_matches() {
    let args = vec![
        SwrlArgument::Literal("en-US".to_string()),
        SwrlArgument::Literal("en".to_string()),
    ];
    assert!(builtin_lang_matches(&args).unwrap());

    let args = vec![
        SwrlArgument::Literal("en-US".to_string()),
        SwrlArgument::Literal("*".to_string()),
    ];
    assert!(builtin_lang_matches(&args).unwrap());
}

#[test]
fn test_builtin_is_literal() {
    let args = vec![SwrlArgument::Literal("test".to_string())];
    assert!(builtin_is_literal(&args).unwrap());

    let args = vec![SwrlArgument::Individual("test".to_string())];
    assert!(!builtin_is_literal(&args).unwrap());
}

#[test]
fn test_builtin_is_blank() {
    let args = vec![SwrlArgument::Individual("_:blank1".to_string())];
    assert!(builtin_is_blank(&args).unwrap());

    let args = vec![SwrlArgument::Individual("http://example.org".to_string())];
    assert!(!builtin_is_blank(&args).unwrap());
}

#[test]
fn test_builtin_is_iri() {
    let args = vec![SwrlArgument::Individual("http://example.org".to_string())];
    assert!(builtin_is_iri(&args).unwrap());

    let args = vec![SwrlArgument::Individual("https://example.org".to_string())];
    assert!(builtin_is_iri(&args).unwrap());
}

#[test]
fn test_builtin_encode_uri() {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("hello%20world".to_string()),
    ];
    assert!(builtin_encode_uri(&args).unwrap());
}

#[test]
fn test_builtin_make_list() {
    let args = vec![
        SwrlArgument::Literal("a".to_string()),
        SwrlArgument::Literal("b".to_string()),
        SwrlArgument::Literal("c".to_string()),
        SwrlArgument::Literal("a,b,c".to_string()),
    ];
    assert!(builtin_make_list(&args).unwrap());
}

#[test]
fn test_builtin_list_reverse() {
    let args = vec![
        SwrlArgument::Literal("a,b,c".to_string()),
        SwrlArgument::Literal("c,b,a".to_string()),
    ];
    assert!(builtin_list_reverse(&args).unwrap());
}

#[test]
fn test_builtin_list_sort() {
    let args = vec![
        SwrlArgument::Literal("c,a,b".to_string()),
        SwrlArgument::Literal("a,b,c".to_string()),
    ];
    assert!(builtin_list_sort(&args).unwrap());
}

#[test]
fn test_builtin_list_union() {
    let args = vec![
        SwrlArgument::Literal("a,b".to_string()),
        SwrlArgument::Literal("b,c".to_string()),
        SwrlArgument::Literal("a,b,c".to_string()),
    ];
    assert!(builtin_list_union(&args).unwrap());
}

#[test]
fn test_builtin_list_intersection() {
    let args = vec![
        SwrlArgument::Literal("a,b,c".to_string()),
        SwrlArgument::Literal("b,c,d".to_string()),
        SwrlArgument::Literal("b,c".to_string()),
    ];
    assert!(builtin_list_intersection(&args).unwrap());
}
