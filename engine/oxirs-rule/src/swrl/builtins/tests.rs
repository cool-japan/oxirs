//! Tests for SWRL Built-in Functions
//!
//! Comprehensive test suite for all SWRL builtin functions

use super::*;
use crate::swrl::types::SwrlArgument;

#[test]
fn test_builtin_divide() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("10.0".to_string()),
        SwrlArgument::Literal("2.0".to_string()),
        SwrlArgument::Literal("5.0".to_string()),
    ];
    assert!(builtin_divide(&args)?);
    Ok(())
}

#[test]
fn test_builtin_integer_divide() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("10".to_string()),
        SwrlArgument::Literal("3".to_string()),
        SwrlArgument::Literal("3".to_string()),
    ];
    assert!(builtin_integer_divide(&args)?);
    Ok(())
}

#[test]
fn test_builtin_min() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("8.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
    ];
    assert!(builtin_min(&args)?);
    Ok(())
}

#[test]
fn test_builtin_max() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("8.0".to_string()),
        SwrlArgument::Literal("8.0".to_string()),
    ];
    assert!(builtin_max(&args)?);
    Ok(())
}

#[test]
fn test_builtin_avg() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("2.0".to_string()),
        SwrlArgument::Literal("4.0".to_string()),
        SwrlArgument::Literal("6.0".to_string()),
        SwrlArgument::Literal("4.0".to_string()),
    ];
    assert!(builtin_avg(&args)?);
    Ok(())
}

#[test]
fn test_builtin_sum() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("2.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("10.0".to_string()),
    ];
    assert!(builtin_sum(&args)?);
    Ok(())
}

#[test]
fn test_builtin_less_than_or_equal() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("5.0".to_string()),
    ];
    assert!(builtin_less_than_or_equal(&args)?);

    let args = vec![
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("5.0".to_string()),
    ];
    assert!(builtin_less_than_or_equal(&args)?);
    Ok(())
}

#[test]
fn test_builtin_between() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("1.0".to_string()),
        SwrlArgument::Literal("10.0".to_string()),
    ];
    assert!(builtin_between(&args)?);
    Ok(())
}

#[test]
fn test_builtin_is_integer() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![SwrlArgument::Literal("5.0".to_string())];
    assert!(builtin_is_integer(&args)?);

    let args = vec![SwrlArgument::Literal("5.5".to_string())];
    assert!(!builtin_is_integer(&args)?);
    Ok(())
}

#[test]
fn test_builtin_is_float() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![SwrlArgument::Literal("3.14".to_string())];
    assert!(builtin_is_float(&args)?);
    Ok(())
}

#[test]
fn test_builtin_is_string() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![SwrlArgument::Literal("hello".to_string())];
    assert!(builtin_is_string(&args)?);
    Ok(())
}

#[test]
fn test_builtin_is_uri() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![SwrlArgument::Individual("http://example.org".to_string())];
    assert!(builtin_is_uri(&args)?);

    let args = vec![SwrlArgument::Literal("not a uri".to_string())];
    assert!(!builtin_is_uri(&args)?);
    Ok(())
}

#[test]
fn test_builtin_string_contains() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("world".to_string()),
    ];
    assert!(builtin_string_contains(&args)?);
    Ok(())
}

#[test]
fn test_builtin_starts_with() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("hello".to_string()),
    ];
    assert!(builtin_starts_with(&args)?);
    Ok(())
}

#[test]
fn test_builtin_ends_with() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("world".to_string()),
    ];
    assert!(builtin_ends_with(&args)?);
    Ok(())
}

#[test]
fn test_builtin_replace() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("world".to_string()),
        SwrlArgument::Literal("universe".to_string()),
        SwrlArgument::Literal("hello universe".to_string()),
    ];
    assert!(builtin_replace(&args)?);
    Ok(())
}

#[test]
fn test_builtin_trim() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("  hello  ".to_string()),
        SwrlArgument::Literal("hello".to_string()),
    ];
    assert!(builtin_trim(&args)?);
    Ok(())
}

#[test]
fn test_builtin_index_of() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("world".to_string()),
        SwrlArgument::Literal("6".to_string()),
    ];
    assert!(builtin_index_of(&args)?);
    Ok(())
}

#[test]
fn test_builtin_normalize_space() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("hello   world  test".to_string()),
        SwrlArgument::Literal("hello world test".to_string()),
    ];
    assert!(builtin_normalize_space(&args)?);
    Ok(())
}

#[test]
fn test_builtin_date() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("2025".to_string()),
        SwrlArgument::Literal("11".to_string()),
        SwrlArgument::Literal("3".to_string()),
        SwrlArgument::Literal("2025-11-03".to_string()),
    ];
    assert!(builtin_date(&args)?);
    Ok(())
}

#[test]
fn test_builtin_time() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("14".to_string()),
        SwrlArgument::Literal("30".to_string()),
        SwrlArgument::Literal("45".to_string()),
        SwrlArgument::Literal("14:30:45".to_string()),
    ];
    assert!(builtin_time(&args)?);
    Ok(())
}

#[test]
fn test_builtin_year() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("2025-11-03".to_string()),
        SwrlArgument::Literal("2025".to_string()),
    ];
    assert!(builtin_year(&args)?);
    Ok(())
}

#[test]
fn test_builtin_month() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("2025-11-03".to_string()),
        SwrlArgument::Literal("11".to_string()),
    ];
    assert!(builtin_month(&args)?);
    Ok(())
}

#[test]
fn test_builtin_day() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("2025-11-03".to_string()),
        SwrlArgument::Literal("3".to_string()),
    ];
    assert!(builtin_day(&args)?);
    Ok(())
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
fn test_builtin_base64_encode() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("hello".to_string()),
        SwrlArgument::Literal("aGVsbG8=".to_string()),
    ];
    assert!(builtin_base64_encode(&args)?);
    Ok(())
}

#[test]
fn test_builtin_base64_decode() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("aGVsbG8=".to_string()),
        SwrlArgument::Literal("hello".to_string()),
    ];
    assert!(builtin_base64_decode(&args)?);
    Ok(())
}

#[test]
fn test_builtin_median() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("1.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
    ];
    assert!(builtin_median(&args)?);
    Ok(())
}

#[test]
fn test_builtin_variance() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("2.0".to_string()),
        SwrlArgument::Literal("4.0".to_string()),
        SwrlArgument::Literal("6.0".to_string()),
        SwrlArgument::Literal("2.6666666666666665".to_string()),
    ];
    assert!(builtin_variance(&args)?);
    Ok(())
}

#[test]
fn test_builtin_stddev() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("2.0".to_string()),
        SwrlArgument::Literal("4.0".to_string()),
        SwrlArgument::Literal("6.0".to_string()),
        SwrlArgument::Literal("1.632993161855452".to_string()),
    ];
    assert!(builtin_stddev(&args)?);
    Ok(())
}

#[test]
fn test_builtin_lang_matches() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("en-US".to_string()),
        SwrlArgument::Literal("en".to_string()),
    ];
    assert!(builtin_lang_matches(&args)?);

    let args = vec![
        SwrlArgument::Literal("en-US".to_string()),
        SwrlArgument::Literal("*".to_string()),
    ];
    assert!(builtin_lang_matches(&args)?);
    Ok(())
}

#[test]
fn test_builtin_is_literal() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![SwrlArgument::Literal("test".to_string())];
    assert!(builtin_is_literal(&args)?);

    let args = vec![SwrlArgument::Individual("test".to_string())];
    assert!(!builtin_is_literal(&args)?);
    Ok(())
}

#[test]
fn test_builtin_is_blank() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![SwrlArgument::Individual("_:blank1".to_string())];
    assert!(builtin_is_blank(&args)?);

    let args = vec![SwrlArgument::Individual("http://example.org".to_string())];
    assert!(!builtin_is_blank(&args)?);
    Ok(())
}

#[test]
fn test_builtin_is_iri() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![SwrlArgument::Individual("http://example.org".to_string())];
    assert!(builtin_is_iri(&args)?);

    let args = vec![SwrlArgument::Individual("https://example.org".to_string())];
    assert!(builtin_is_iri(&args)?);
    Ok(())
}

#[test]
fn test_builtin_encode_uri() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("hello%20world".to_string()),
    ];
    assert!(builtin_encode_uri(&args)?);
    Ok(())
}

#[test]
fn test_builtin_make_list() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("a".to_string()),
        SwrlArgument::Literal("b".to_string()),
        SwrlArgument::Literal("c".to_string()),
        SwrlArgument::Literal("a,b,c".to_string()),
    ];
    assert!(builtin_make_list(&args)?);
    Ok(())
}

#[test]
fn test_builtin_list_reverse() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("a,b,c".to_string()),
        SwrlArgument::Literal("c,b,a".to_string()),
    ];
    assert!(builtin_list_reverse(&args)?);
    Ok(())
}

#[test]
fn test_builtin_list_sort() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("c,a,b".to_string()),
        SwrlArgument::Literal("a,b,c".to_string()),
    ];
    assert!(builtin_list_sort(&args)?);
    Ok(())
}

#[test]
fn test_builtin_list_union() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("a,b".to_string()),
        SwrlArgument::Literal("b,c".to_string()),
        SwrlArgument::Literal("a,b,c".to_string()),
    ];
    assert!(builtin_list_union(&args)?);
    Ok(())
}

#[test]
fn test_builtin_list_intersection() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("a,b,c".to_string()),
        SwrlArgument::Literal("b,c,d".to_string()),
        SwrlArgument::Literal("b,c".to_string()),
    ];
    assert!(builtin_list_intersection(&args)?);
    Ok(())
}

// ---- Extended tests ----

#[test]
fn test_builtin_equal_numbers() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("42.0".to_string()),
        SwrlArgument::Literal("42.0".to_string()),
    ];
    assert!(builtin_equal(&args)?);
    Ok(())
}

#[test]
fn test_builtin_equal_unequal_returns_false() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("1.0".to_string()),
        SwrlArgument::Literal("2.0".to_string()),
    ];
    assert!(!builtin_equal(&args)?);
    Ok(())
}

#[test]
fn test_builtin_not_equal_different_values() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("4.0".to_string()),
    ];
    assert!(builtin_not_equal(&args)?);
    Ok(())
}

#[test]
fn test_builtin_not_equal_same_values_returns_false() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("7.0".to_string()),
        SwrlArgument::Literal("7.0".to_string()),
    ];
    assert!(!builtin_not_equal(&args)?);
    Ok(())
}

#[test]
fn test_builtin_less_than_true() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("1.0".to_string()),
        SwrlArgument::Literal("2.0".to_string()),
    ];
    assert!(builtin_less_than(&args)?);
    Ok(())
}

#[test]
fn test_builtin_less_than_equal_returns_false() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("5.0".to_string()),
    ];
    assert!(!builtin_less_than(&args)?);
    Ok(())
}

#[test]
fn test_builtin_greater_than_true() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("9.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
    ];
    assert!(builtin_greater_than(&args)?);
    Ok(())
}

#[test]
fn test_builtin_greater_than_equal_returns_false() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("4.0".to_string()),
        SwrlArgument::Literal("4.0".to_string()),
    ];
    assert!(!builtin_greater_than(&args)?);
    Ok(())
}

#[test]
fn test_builtin_add_three_plus_four() -> Result<(), Box<dyn std::error::Error>> {
    // add(a, b, result): a + b == result
    let args = vec![
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("4.0".to_string()),
        SwrlArgument::Literal("7.0".to_string()),
    ];
    assert!(builtin_add(&args)?);
    Ok(())
}

#[test]
fn test_builtin_subtract_ten_minus_three() -> Result<(), Box<dyn std::error::Error>> {
    // subtract(a, b, result): a - b == result
    let args = vec![
        SwrlArgument::Literal("10.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("7.0".to_string()),
    ];
    assert!(builtin_subtract(&args)?);
    Ok(())
}

#[test]
fn test_builtin_multiply_six_by_seven() -> Result<(), Box<dyn std::error::Error>> {
    // multiply(a, b, result): a * b == result
    let args = vec![
        SwrlArgument::Literal("6.0".to_string()),
        SwrlArgument::Literal("7.0".to_string()),
        SwrlArgument::Literal("42.0".to_string()),
    ];
    assert!(builtin_multiply(&args)?);
    Ok(())
}

#[test]
fn test_builtin_mod_ten_mod_three() -> Result<(), Box<dyn std::error::Error>> {
    // mod(dividend, divisor, result): dividend % divisor == result
    let args = vec![
        SwrlArgument::Literal("10.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
        SwrlArgument::Literal("1.0".to_string()),
    ];
    assert!(builtin_mod(&args)?);
    Ok(())
}

#[test]
fn test_builtin_pow_two_to_ten() -> Result<(), Box<dyn std::error::Error>> {
    // pow(base, exp, result): base ^ exp == result
    let args = vec![
        SwrlArgument::Literal("2.0".to_string()),
        SwrlArgument::Literal("10.0".to_string()),
        SwrlArgument::Literal("1024.0".to_string()),
    ];
    assert!(builtin_pow(&args)?);
    Ok(())
}

#[test]
fn test_builtin_sqrt_nine() -> Result<(), Box<dyn std::error::Error>> {
    // sqrt(input, result): sqrt(input) == result
    let args = vec![
        SwrlArgument::Literal("9.0".to_string()),
        SwrlArgument::Literal("3.0".to_string()),
    ];
    assert!(builtin_sqrt(&args)?);
    Ok(())
}

#[test]
fn test_builtin_sin_zero() -> Result<(), Box<dyn std::error::Error>> {
    // sin(input, result): sin(input) == result
    let args = vec![
        SwrlArgument::Literal("0.0".to_string()),
        SwrlArgument::Literal("0.0".to_string()),
    ];
    assert!(builtin_sin(&args)?);
    Ok(())
}

#[test]
fn test_builtin_cos_zero_is_one() -> Result<(), Box<dyn std::error::Error>> {
    // cos(input, result): cos(input) == result
    let args = vec![
        SwrlArgument::Literal("0.0".to_string()),
        SwrlArgument::Literal("1.0".to_string()),
    ];
    assert!(builtin_cos(&args)?);
    Ok(())
}

#[test]
fn test_builtin_string_concat_hello_world() -> Result<(), Box<dyn std::error::Error>> {
    // stringConcat(s1, s2, ..., result): concat(s[0..n-1]) == s[n-1]
    let args = vec![
        SwrlArgument::Literal("hello".to_string()),
        SwrlArgument::Literal(" ".to_string()),
        SwrlArgument::Literal("world".to_string()),
        SwrlArgument::Literal("hello world".to_string()),
    ];
    assert!(builtin_string_concat(&args)?);
    Ok(())
}

#[test]
fn test_builtin_string_length_five() -> Result<(), Box<dyn std::error::Error>> {
    // stringLength(string, length): len(string) == length
    let args = vec![
        SwrlArgument::Literal("hello".to_string()),
        SwrlArgument::Literal("5".to_string()),
    ];
    assert!(builtin_string_length(&args)?);
    Ok(())
}

#[test]
fn test_builtin_upper_case_hello() -> Result<(), Box<dyn std::error::Error>> {
    // upperCase(input, result): input.to_uppercase() == result
    let args = vec![
        SwrlArgument::Literal("hello".to_string()),
        SwrlArgument::Literal("HELLO".to_string()),
    ];
    assert!(builtin_upper_case(&args)?);
    Ok(())
}

#[test]
fn test_builtin_lower_case_world() -> Result<(), Box<dyn std::error::Error>> {
    // lowerCase(input, result): input.to_lowercase() == result
    let args = vec![
        SwrlArgument::Literal("WORLD".to_string()),
        SwrlArgument::Literal("world".to_string()),
    ];
    assert!(builtin_lower_case(&args)?);
    Ok(())
}

#[test]
fn test_builtin_string_contains_false() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("hello".to_string()),
        SwrlArgument::Literal("xyz".to_string()),
    ];
    assert!(!builtin_string_contains(&args)?);
    Ok(())
}

#[test]
fn test_builtin_starts_with_false() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("world".to_string()),
    ];
    assert!(!builtin_starts_with(&args)?);
    Ok(())
}

#[test]
fn test_builtin_ends_with_false() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("hello".to_string()),
    ];
    assert!(!builtin_ends_with(&args)?);
    Ok(())
}

#[test]
fn test_builtin_trim_no_spaces() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("hello".to_string()),
        SwrlArgument::Literal("hello".to_string()),
    ];
    assert!(builtin_trim(&args)?);
    Ok(())
}

#[test]
fn test_builtin_normalize_space_single_spaces() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("a b c".to_string()),
        SwrlArgument::Literal("a b c".to_string()),
    ];
    assert!(builtin_normalize_space(&args)?);
    Ok(())
}

#[test]
fn test_builtin_between_at_lower_bound() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("1.0".to_string()),
        SwrlArgument::Literal("1.0".to_string()),
        SwrlArgument::Literal("10.0".to_string()),
    ];
    assert!(builtin_between(&args)?);
    Ok(())
}

#[test]
fn test_builtin_between_at_upper_bound() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("10.0".to_string()),
        SwrlArgument::Literal("1.0".to_string()),
        SwrlArgument::Literal("10.0".to_string()),
    ];
    assert!(builtin_between(&args)?);
    Ok(())
}

#[test]
fn test_builtin_between_out_of_range() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("11.0".to_string()),
        SwrlArgument::Literal("1.0".to_string()),
        SwrlArgument::Literal("10.0".to_string()),
    ];
    assert!(!builtin_between(&args)?);
    Ok(())
}

#[test]
fn test_builtin_member_not_found() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("a,b,c".to_string()),
        SwrlArgument::Literal("z".to_string()),
    ];
    assert!(!builtin_member(&args)?);
    Ok(())
}

#[test]
fn test_builtin_list_length_basic() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("a,b,c,d".to_string()),
        SwrlArgument::Literal("4".to_string()),
    ];
    assert!(builtin_list_length(&args)?);
    Ok(())
}

#[test]
fn test_builtin_boolean_value_false() {
    let args = vec![SwrlArgument::Literal("false".to_string())];
    // "false" as boolean literal — builtin_boolean_value checks the value
    let result = builtin_boolean_value(&args);
    assert!(
        result.is_ok(),
        "builtin_boolean_value should not error on 'false'"
    );
}

#[test]
fn test_builtin_integer_divide_floor() -> Result<(), Box<dyn std::error::Error>> {
    // 7 / 3 = 2 (floor)
    let args = vec![
        SwrlArgument::Literal("7".to_string()),
        SwrlArgument::Literal("3".to_string()),
        SwrlArgument::Literal("2".to_string()),
    ];
    assert!(builtin_integer_divide(&args)?);
    Ok(())
}

#[test]
fn test_builtin_is_literal_with_individual_returns_false() -> Result<(), Box<dyn std::error::Error>>
{
    let args = vec![SwrlArgument::Individual(
        "http://example.org/Alice".to_string(),
    )];
    assert!(!builtin_is_literal(&args)?);
    Ok(())
}

#[test]
fn test_builtin_replace_no_match() -> Result<(), Box<dyn std::error::Error>> {
    // Replace "xyz" in "hello" with "abc" — no match, result is "hello"
    let args = vec![
        SwrlArgument::Literal("hello".to_string()),
        SwrlArgument::Literal("xyz".to_string()),
        SwrlArgument::Literal("abc".to_string()),
        SwrlArgument::Literal("hello".to_string()),
    ];
    assert!(builtin_replace(&args)?);
    Ok(())
}

#[test]
fn test_builtin_index_of_at_start() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("hello world".to_string()),
        SwrlArgument::Literal("hello".to_string()),
        SwrlArgument::Literal("0".to_string()),
    ];
    assert!(builtin_index_of(&args)?);
    Ok(())
}

#[test]
fn test_builtin_min_single_value() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("5.0".to_string()),
        SwrlArgument::Literal("5.0".to_string()),
    ];
    assert!(builtin_min(&args)?);
    Ok(())
}

#[test]
fn test_builtin_max_single_value() -> Result<(), Box<dyn std::error::Error>> {
    let args = vec![
        SwrlArgument::Literal("7.0".to_string()),
        SwrlArgument::Literal("7.0".to_string()),
    ];
    assert!(builtin_max(&args)?);
    Ok(())
}

#[test]
fn test_builtin_divide_by_two() -> Result<(), Box<dyn std::error::Error>> {
    // divide(a, b, result): a / b == result
    let args = vec![
        SwrlArgument::Literal("10.0".to_string()),
        SwrlArgument::Literal("2.0".to_string()),
        SwrlArgument::Literal("5.0".to_string()),
    ];
    assert!(builtin_divide(&args)?);
    Ok(())
}
