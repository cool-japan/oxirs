//! SWRL String Built-in Functions
//!
//! This module implements string operations for SWRL rules including:
//! - Concatenation and length: string_concat, string_length
//! - Pattern matching: string_matches, string_matches_regex
//! - Substring operations: substring, string_contains, starts_with, ends_with
//! - Case conversion: upper_case, lower_case
//! - Manipulation: replace, trim, normalize_space
//! - Searching: index_of, last_index_of
//! - Splitting: split
//! - Conversion: str

use anyhow::Result;
use regex::{Regex, RegexBuilder};

use super::super::types::SwrlArgument;
use super::utils::*;

pub fn builtin_string_concat(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 3 {
        return Err(anyhow::anyhow!(
            "stringConcat requires at least 3 arguments"
        ));
    }

    let mut concat_result = String::new();
    for arg in &args[0..args.len() - 1] {
        concat_result.push_str(&extract_string_value(arg)?);
    }

    let expected = extract_string_value(&args[args.len() - 1])?;
    Ok(concat_result == expected)
}

pub(crate) fn builtin_string_length(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("stringLength requires exactly 2 arguments"));
    }

    let string_val = extract_string_value(&args[0])?;
    let length_val = extract_numeric_value(&args[1])? as usize;

    Ok(string_val.len() == length_val)
}

pub(crate) fn builtin_string_matches(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 || args.len() > 3 {
        return Err(anyhow::anyhow!("stringMatches requires 2 or 3 arguments"));
    }

    let input = extract_string_value(&args[0])?;
    let pattern = extract_string_value(&args[1])?;

    // Simple pattern matching (could be enhanced with full regex support)
    if args.len() == 3 {
        let _flags = extract_string_value(&args[2])?;
        // Ignore flags in simple implementation
    }

    // Basic wildcard matching (* and ?)
    let regex_pattern = pattern.replace("*", ".*").replace("?", ".");

    match Regex::new(&regex_pattern) {
        Ok(re) => Ok(re.is_match(&input)),
        Err(_) => Ok(false),
    }
}

pub(crate) fn builtin_substring(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 3 || args.len() > 4 {
        return Err(anyhow::anyhow!("substring requires 3 or 4 arguments"));
    }

    let input = extract_string_value(&args[0])?;
    let start = extract_numeric_value(&args[1])? as usize;
    let result = extract_string_value(&args[args.len() - 1])?;

    let extracted = if args.len() == 4 {
        let length = extract_numeric_value(&args[2])? as usize;
        if start > input.len() {
            String::new()
        } else {
            let end = std::cmp::min(start + length, input.len());
            input.chars().skip(start).take(end - start).collect()
        }
    } else if start > input.len() {
        String::new()
    } else {
        input.chars().skip(start).collect()
    };

    Ok(extracted == result)
}

pub fn builtin_upper_case(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("upperCase requires exactly 2 arguments"));
    }

    let input = extract_string_value(&args[0])?;
    let result = extract_string_value(&args[1])?;

    Ok(input.to_uppercase() == result)
}

pub(crate) fn builtin_lower_case(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("lowerCase requires exactly 2 arguments"));
    }

    let input = extract_string_value(&args[0])?;
    let result = extract_string_value(&args[1])?;

    Ok(input.to_lowercase() == result)
}

pub(crate) fn builtin_string_matches_regex(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 || args.len() > 3 {
        return Err(anyhow::anyhow!(
            "stringMatchesRegex requires 2 or 3 arguments"
        ));
    }

    let input = extract_string_value(&args[0])?;
    let pattern = extract_string_value(&args[1])?;

    // Full regex support with optional flags
    let regex_builder = if args.len() == 3 {
        let flags = extract_string_value(&args[2])?;
        let mut builder = RegexBuilder::new(&pattern);

        for flag in flags.chars() {
            match flag {
                'i' => {
                    builder.case_insensitive(true);
                }
                'm' => {
                    builder.multi_line(true);
                }
                's' => {
                    builder.dot_matches_new_line(true);
                }
                'x' => {
                    builder.ignore_whitespace(true);
                }
                _ => return Err(anyhow::anyhow!("Unknown regex flag: {}", flag)),
            }
        }
        builder
    } else {
        RegexBuilder::new(&pattern)
    };

    match regex_builder.build() {
        Ok(re) => Ok(re.is_match(&input)),
        Err(e) => Err(anyhow::anyhow!("Invalid regex pattern: {}", e)),
    }
}

pub(crate) fn builtin_string_contains(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("contains requires exactly 2 arguments"));
    }

    let haystack = extract_string_value(&args[0])?;
    let needle = extract_string_value(&args[1])?;

    Ok(haystack.contains(&needle))
}

pub(crate) fn builtin_starts_with(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("startsWith requires exactly 2 arguments"));
    }

    let string = extract_string_value(&args[0])?;
    let prefix = extract_string_value(&args[1])?;

    Ok(string.starts_with(&prefix))
}

pub(crate) fn builtin_ends_with(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("endsWith requires exactly 2 arguments"));
    }

    let string = extract_string_value(&args[0])?;
    let suffix = extract_string_value(&args[1])?;

    Ok(string.ends_with(&suffix))
}

pub(crate) fn builtin_replace(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 4 {
        return Err(anyhow::anyhow!(
            "replace requires exactly 4 arguments: input, search, replacement, result"
        ));
    }

    let input = extract_string_value(&args[0])?;
    let search = extract_string_value(&args[1])?;
    let replacement = extract_string_value(&args[2])?;
    let expected = extract_string_value(&args[3])?;

    let result = input.replace(&search, &replacement);
    Ok(result == expected)
}

pub(crate) fn builtin_trim(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("trim requires exactly 2 arguments"));
    }

    let input = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    Ok(input.trim() == expected)
}

pub(crate) fn builtin_split(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "split requires exactly 3 arguments: input, delimiter, result"
        ));
    }

    let input = extract_string_value(&args[0])?;
    let delimiter = extract_string_value(&args[1])?;
    let expected = extract_string_value(&args[2])?;

    let parts: Vec<&str> = input.split(&delimiter as &str).collect();
    let result = parts.join(",");

    Ok(result == expected)
}

pub(crate) fn builtin_index_of(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "indexOf requires exactly 3 arguments: string, substring, result"
        ));
    }

    let string = extract_string_value(&args[0])?;
    let substring = extract_string_value(&args[1])?;
    let expected_index = extract_numeric_value(&args[2])? as i64;

    let actual_index = string
        .find(&substring as &str)
        .map(|i| i as i64)
        .unwrap_or(-1);

    Ok(actual_index == expected_index)
}

pub(crate) fn builtin_last_index_of(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "lastIndexOf requires exactly 3 arguments: string, substring, result"
        ));
    }

    let string = extract_string_value(&args[0])?;
    let substring = extract_string_value(&args[1])?;
    let expected_index = extract_numeric_value(&args[2])? as i64;

    let actual_index = string
        .rfind(&substring as &str)
        .map(|i| i as i64)
        .unwrap_or(-1);

    Ok(actual_index == expected_index)
}

pub(crate) fn builtin_normalize_space(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "normalizeSpace requires exactly 2 arguments"
        ));
    }

    let input = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Normalize whitespace: trim and replace multiple spaces with single space
    let normalized = input.split_whitespace().collect::<Vec<&str>>().join(" ");

    Ok(normalized == expected)
}

pub(crate) fn builtin_str(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "str requires exactly 2 arguments: node, result"
        ));
    }

    let node = extract_string_value(&args[0])?;
    let result = extract_string_value(&args[1])?;

    Ok(node == result)
}
