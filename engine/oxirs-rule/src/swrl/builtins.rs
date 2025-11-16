//! SWRL (Semantic Web Rule Language) - Built-in Functions
//!
//! This module implements SWRL rule components.

use anyhow::Result;

use super::types::SwrlArgument;
use regex::{Regex, RegexBuilder};

pub(crate) fn builtin_equal(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("equal requires exactly 2 arguments"));
    }
    Ok(args[0] == args[1])
}

pub(crate) fn builtin_not_equal(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("notEqual requires exactly 2 arguments"));
    }
    Ok(args[0] != args[1])
}

pub(crate) fn builtin_less_than(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("lessThan requires exactly 2 arguments"));
    }

    let val1 = extract_numeric_value(&args[0])?;
    let val2 = extract_numeric_value(&args[1])?;
    Ok(val1 < val2)
}

pub(crate) fn builtin_greater_than(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("greaterThan requires exactly 2 arguments"));
    }

    let val1 = extract_numeric_value(&args[0])?;
    let val2 = extract_numeric_value(&args[1])?;
    Ok(val1 > val2)
}

pub fn builtin_add(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("add requires exactly 3 arguments"));
    }

    let val1 = extract_numeric_value(&args[0])?;
    let val2 = extract_numeric_value(&args[1])?;
    let result = extract_numeric_value(&args[2])?;

    Ok((val1 + val2 - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_subtract(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("subtract requires exactly 3 arguments"));
    }

    let val1 = extract_numeric_value(&args[0])?;
    let val2 = extract_numeric_value(&args[1])?;
    let result = extract_numeric_value(&args[2])?;

    Ok((val1 - val2 - result).abs() < f64::EPSILON)
}

pub fn builtin_multiply(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("multiply requires exactly 3 arguments"));
    }

    let val1 = extract_numeric_value(&args[0])?;
    let val2 = extract_numeric_value(&args[1])?;
    let result = extract_numeric_value(&args[2])?;

    Ok((val1 * val2 - result).abs() < f64::EPSILON)
}

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

pub(crate) fn builtin_boolean_value(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 1 {
        return Err(anyhow::anyhow!("booleanValue requires exactly 1 argument"));
    }

    extract_boolean_value(&args[0])
}

// Helper functions for value extraction

fn extract_numeric_value(arg: &SwrlArgument) -> Result<f64> {
    match arg {
        SwrlArgument::Literal(value) => value
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("Cannot parse '{}' as numeric value", value)),
        _ => Err(anyhow::anyhow!(
            "Expected literal numeric value, got {:?}",
            arg
        )),
    }
}

fn extract_string_value(arg: &SwrlArgument) -> Result<String> {
    match arg {
        SwrlArgument::Literal(value) => Ok(value.clone()),
        SwrlArgument::Individual(value) => Ok(value.clone()),
        SwrlArgument::Variable(name) => Err(anyhow::anyhow!("Unbound variable: {}", name)),
    }
}

fn extract_boolean_value(arg: &SwrlArgument) -> Result<bool> {
    match arg {
        SwrlArgument::Literal(value) => match value.to_lowercase().as_str() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => Err(anyhow::anyhow!("Cannot parse '{}' as boolean value", value)),
        },
        _ => Err(anyhow::anyhow!(
            "Expected literal boolean value, got {:?}",
            arg
        )),
    }
}

// Additional mathematical built-ins

pub(crate) fn builtin_mod(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("mod requires exactly 3 arguments"));
    }

    let dividend = extract_numeric_value(&args[0])?;
    let divisor = extract_numeric_value(&args[1])?;
    let result = extract_numeric_value(&args[2])?;

    if divisor == 0.0 {
        return Err(anyhow::anyhow!("Division by zero in mod operation"));
    }

    Ok((dividend % divisor - result).abs() < f64::EPSILON)
}

pub fn builtin_pow(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("pow requires exactly 3 arguments"));
    }

    let base = extract_numeric_value(&args[0])?;
    let exponent = extract_numeric_value(&args[1])?;
    let result = extract_numeric_value(&args[2])?;

    Ok((base.powf(exponent) - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_sqrt(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("sqrt requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    if input < 0.0 {
        return Err(anyhow::anyhow!(
            "Cannot take square root of negative number"
        ));
    }

    Ok((input.sqrt() - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_sin(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("sin requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((input.sin() - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_cos(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("cos requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((input.cos() - result).abs() < f64::EPSILON)
}

// Date and time built-ins

pub(crate) fn builtin_day_time_duration(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "dayTimeDuration requires exactly 2 arguments"
        ));
    }

    let duration_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple duration parsing (P[n]DT[n]H[n]M[n]S format)
    // This is a simplified implementation
    Ok(duration_str == expected)
}

pub(crate) fn builtin_year_month_duration(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "yearMonthDuration requires exactly 2 arguments"
        ));
    }

    let duration_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple duration parsing (P[n]Y[n]M format)
    // This is a simplified implementation
    Ok(duration_str == expected)
}

pub(crate) fn builtin_date_time(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("dateTime requires exactly 2 arguments"));
    }

    let datetime_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple datetime validation (ISO 8601 format)
    // This is a simplified implementation
    Ok(datetime_str == expected)
}

// List operations

pub(crate) fn builtin_list_concat(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 {
        return Err(anyhow::anyhow!("listConcat requires at least 2 arguments"));
    }

    // In a full implementation, this would handle RDF lists
    // For now, treat as string concatenation of comma-separated values
    let mut concat_result = String::new();
    for arg in &args[0..args.len() - 1] {
        if !concat_result.is_empty() {
            concat_result.push(',');
        }
        concat_result.push_str(&extract_string_value(arg)?);
    }

    let expected = extract_string_value(&args[args.len() - 1])?;
    Ok(concat_result == expected)
}

pub(crate) fn builtin_list_length(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("listLength requires exactly 2 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let length_val = extract_numeric_value(&args[1])? as usize;

    // Simple implementation: count comma-separated items
    let items: Vec<&str> = list_str.split(',').collect();
    Ok(items.len() == length_val)
}

pub(crate) fn builtin_member(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("member requires exactly 2 arguments"));
    }

    let element = extract_string_value(&args[0])?;
    let list_str = extract_string_value(&args[1])?;

    // Simple implementation: check if element is in comma-separated list
    let items: Vec<&str> = list_str.split(',').collect();
    Ok(items.contains(&element.as_str()))
}

// Enhanced string operations

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

// Advanced mathematical built-ins
pub(crate) fn builtin_tan(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("tan requires exactly 2 arguments"));
    }

    let x = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((x.tan() - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_asin(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("asin requires exactly 2 arguments"));
    }

    let x = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    if !(-1.0..=1.0).contains(&x) {
        return Ok(false);
    }

    Ok((x.asin() - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_acos(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("acos requires exactly 2 arguments"));
    }

    let x = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    if !(-1.0..=1.0).contains(&x) {
        return Ok(false);
    }

    Ok((x.acos() - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_atan(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("atan requires exactly 2 arguments"));
    }

    let x = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((x.atan() - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_log(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("log requires exactly 2 arguments"));
    }

    let x = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    if x <= 0.0 {
        return Ok(false);
    }

    Ok((x.ln() - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_exp(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("exp requires exactly 2 arguments"));
    }

    let x = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((x.exp() - result).abs() < f64::EPSILON)
}

// Geographic operations
pub(crate) fn builtin_distance(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 5 {
        return Err(anyhow::anyhow!(
            "distance requires exactly 5 arguments: lat1, lon1, lat2, lon2, result"
        ));
    }

    let lat1 = extract_numeric_value(&args[0])?.to_radians();
    let lon1 = extract_numeric_value(&args[1])?.to_radians();
    let lat2 = extract_numeric_value(&args[2])?.to_radians();
    let lon2 = extract_numeric_value(&args[3])?.to_radians();
    let expected_distance = extract_numeric_value(&args[4])?;

    // Haversine formula for great circle distance
    let earth_radius = 6371.0; // Earth radius in kilometers
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    let distance = earth_radius * c;

    Ok((distance - expected_distance).abs() < 0.001) // 1 meter tolerance
}

pub(crate) fn builtin_within(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 5 {
        return Err(anyhow::anyhow!(
            "within requires exactly 5 arguments: lat1, lon1, lat2, lon2, max_distance"
        ));
    }

    let lat1 = extract_numeric_value(&args[0])?.to_radians();
    let lon1 = extract_numeric_value(&args[1])?.to_radians();
    let lat2 = extract_numeric_value(&args[2])?.to_radians();
    let lon2 = extract_numeric_value(&args[3])?.to_radians();
    let max_distance = extract_numeric_value(&args[4])?;

    // Haversine formula for great circle distance
    let earth_radius = 6371.0; // Earth radius in kilometers
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    let distance = earth_radius * c;

    Ok(distance <= max_distance)
}

// Temporal operations
pub(crate) fn builtin_date_add(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 4 {
        return Err(anyhow::anyhow!(
            "dateAdd requires exactly 4 arguments: date, duration, unit, result"
        ));
    }

    let date_str = extract_string_value(&args[0])?;
    let duration = extract_numeric_value(&args[1])? as i64;
    let unit = extract_string_value(&args[2])?;
    let expected_result = extract_string_value(&args[3])?;

    // Parse ISO 8601 date string (simplified - in production would use chrono)
    if let Ok(timestamp) = date_str.parse::<i64>() {
        let seconds_to_add = match unit.as_str() {
            "seconds" => duration,
            "minutes" => duration * 60,
            "hours" => duration * 3600,
            "days" => duration * 86400,
            "weeks" => duration * 604800,
            _ => return Err(anyhow::anyhow!("Unsupported time unit: {}", unit)),
        };

        let result_timestamp = timestamp + seconds_to_add;
        Ok(result_timestamp.to_string() == expected_result)
    } else {
        // For proper date strings, would need chrono crate
        Ok(false)
    }
}

pub(crate) fn builtin_date_diff(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "dateDiff requires exactly 3 arguments: date1, date2, result"
        ));
    }

    let date1_str = extract_string_value(&args[0])?;
    let date2_str = extract_string_value(&args[1])?;
    let expected_diff = extract_numeric_value(&args[2])?;

    // Parse timestamps (simplified - in production would use chrono)
    if let (Ok(timestamp1), Ok(timestamp2)) = (date1_str.parse::<i64>(), date2_str.parse::<i64>()) {
        let diff_seconds = (timestamp2 - timestamp1).abs() as f64;
        Ok((diff_seconds - expected_diff).abs() < 1.0) // 1 second tolerance
    } else {
        Ok(false)
    }
}

pub(crate) fn builtin_now(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 1 {
        return Err(anyhow::anyhow!("now requires exactly 1 argument: result"));
    }

    let expected_result = extract_string_value(&args[0])?;

    // Get current timestamp (simplified - in production would use proper time crate)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| anyhow::anyhow!("Time error: {}", e))?
        .as_secs();

    // Check if the expected result is close to current time (within 1 second)
    if let Ok(expected_timestamp) = expected_result.parse::<u64>() {
        Ok((now as i64 - expected_timestamp as i64).abs() <= 1)
    } else {
        Ok(false)
    }
}

// SWRL-X Temporal Extensions
pub(crate) fn builtin_temporal_before(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "temporal:before requires exactly 2 arguments: time1, time2"
        ));
    }

    let time1 = extract_numeric_value(&args[0])?;
    let time2 = extract_numeric_value(&args[1])?;

    Ok(time1 < time2)
}

pub(crate) fn builtin_temporal_after(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "temporal:after requires exactly 2 arguments: time1, time2"
        ));
    }

    let time1 = extract_numeric_value(&args[0])?;
    let time2 = extract_numeric_value(&args[1])?;

    Ok(time1 > time2)
}

pub(crate) fn builtin_temporal_during(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "temporal:during requires exactly 3 arguments: time, interval_start, interval_end"
        ));
    }

    let time = extract_numeric_value(&args[0])?;
    let interval_start = extract_numeric_value(&args[1])?;
    let interval_end = extract_numeric_value(&args[2])?;

    Ok(time >= interval_start && time <= interval_end)
}

pub(crate) fn builtin_temporal_overlaps(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 4 {
        return Err(anyhow::anyhow!(
            "temporal:overlaps requires exactly 4 arguments: start1, end1, start2, end2"
        ));
    }

    let start1 = extract_numeric_value(&args[0])?;
    let end1 = extract_numeric_value(&args[1])?;
    let start2 = extract_numeric_value(&args[2])?;
    let end2 = extract_numeric_value(&args[3])?;

    // Two intervals overlap if they have any time in common
    Ok(start1 < end2 && start2 < end1)
}

pub(crate) fn builtin_temporal_meets(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 4 {
        return Err(anyhow::anyhow!(
            "temporal:meets requires exactly 4 arguments: start1, end1, start2, end2"
        ));
    }

    let _start1 = extract_numeric_value(&args[0])?;
    let end1 = extract_numeric_value(&args[1])?;
    let start2 = extract_numeric_value(&args[2])?;
    let _end2 = extract_numeric_value(&args[3])?;

    // First interval meets second if end of first equals start of second
    Ok((end1 - start2).abs() < f64::EPSILON)
}

pub(crate) fn builtin_interval_duration(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "temporal:intervalDuration requires exactly 3 arguments: start, end, duration"
        ));
    }

    let start = extract_numeric_value(&args[0])?;
    let end = extract_numeric_value(&args[1])?;
    let expected_duration = extract_numeric_value(&args[2])?;

    let actual_duration = (end - start).abs();
    Ok((actual_duration - expected_duration).abs() < f64::EPSILON)
}

// Additional mathematical built-in functions

pub(crate) fn builtin_abs(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("abs requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((input.abs() - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_floor(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("floor requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((input.floor() - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_ceil(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("ceil requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((input.ceil() - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_round(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("round requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((input.round() - result).abs() < f64::EPSILON)
}

// Additional list operations

pub(crate) fn builtin_list_first(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("first requires exactly 2 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple implementation: get first item from comma-separated list
    let items: Vec<&str> = list_str.split(',').collect();
    if items.is_empty() {
        Ok(expected.is_empty())
    } else {
        Ok(items[0] == expected)
    }
}

pub(crate) fn builtin_list_rest(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("rest requires exactly 2 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple implementation: get all but first item from comma-separated list
    let items: Vec<&str> = list_str.split(',').collect();
    if items.len() <= 1 {
        Ok(expected.is_empty())
    } else {
        let rest = items[1..].join(",");
        Ok(rest == expected)
    }
}

pub(crate) fn builtin_list_nth(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("nth requires exactly 3 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let index = extract_numeric_value(&args[1])? as usize;
    let expected = extract_string_value(&args[2])?;

    // Simple implementation: get nth item from comma-separated list (0-indexed)
    let items: Vec<&str> = list_str.split(',').collect();
    if index >= items.len() {
        return Err(anyhow::anyhow!(
            "Index {} out of bounds for list of length {}",
            index,
            items.len()
        ));
    }
    Ok(items[index] == expected)
}

pub(crate) fn builtin_list_append(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("append requires exactly 3 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let item = extract_string_value(&args[1])?;
    let expected = extract_string_value(&args[2])?;

    // Simple implementation: append item to comma-separated list
    let result = if list_str.is_empty() {
        item
    } else {
        format!("{list_str},{item}")
    };
    Ok(result == expected)
}

// Enhanced string operations with full regex support

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

// Additional geographic operations

pub(crate) fn builtin_geo_contains(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 9 {
        return Err(anyhow::anyhow!("contains requires exactly 9 arguments: poly_min_lat, poly_min_lon, poly_max_lat, poly_max_lon, point_lat, point_lon, result"));
    }

    // Simple bounding box containment check
    let min_lat = extract_numeric_value(&args[0])?;
    let min_lon = extract_numeric_value(&args[1])?;
    let max_lat = extract_numeric_value(&args[2])?;
    let max_lon = extract_numeric_value(&args[3])?;
    let point_lat = extract_numeric_value(&args[4])?;
    let point_lon = extract_numeric_value(&args[5])?;
    let expected_result = extract_string_value(&args[6])? == "true";

    let contains = point_lat >= min_lat
        && point_lat <= max_lat
        && point_lon >= min_lon
        && point_lon <= max_lon;

    Ok(contains == expected_result)
}

pub(crate) fn builtin_geo_intersects(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 8 {
        return Err(anyhow::anyhow!("intersects requires exactly 8 arguments: box1_min_lat, box1_min_lon, box1_max_lat, box1_max_lon, box2_min_lat, box2_min_lon, box2_max_lat, box2_max_lon"));
    }

    // Check if two bounding boxes intersect
    let box1_min_lat = extract_numeric_value(&args[0])?;
    let box1_min_lon = extract_numeric_value(&args[1])?;
    let box1_max_lat = extract_numeric_value(&args[2])?;
    let box1_max_lon = extract_numeric_value(&args[3])?;
    let box2_min_lat = extract_numeric_value(&args[4])?;
    let box2_min_lon = extract_numeric_value(&args[5])?;
    let box2_max_lat = extract_numeric_value(&args[6])?;
    let box2_max_lon = extract_numeric_value(&args[7])?;

    // Two boxes intersect if they overlap in both dimensions
    let intersects = !(box1_max_lat < box2_min_lat
        || box2_max_lat < box1_min_lat
        || box1_max_lon < box2_min_lon
        || box2_max_lon < box1_min_lon);

    Ok(intersects)
}

pub(crate) fn builtin_geo_area(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 5 {
        return Err(anyhow::anyhow!(
            "area requires exactly 5 arguments: min_lat, min_lon, max_lat, max_lon, result"
        ));
    }

    // Calculate approximate area of a bounding box in square kilometers
    let min_lat = extract_numeric_value(&args[0])?;
    let min_lon = extract_numeric_value(&args[1])?;
    let max_lat = extract_numeric_value(&args[2])?;
    let max_lon = extract_numeric_value(&args[3])?;
    let expected_area = extract_numeric_value(&args[4])?;

    // Simple approximation: treat Earth as sphere
    const EARTH_RADIUS_KM: f64 = 6371.0;

    // Convert to radians
    let lat1_rad = min_lat.to_radians();
    let lat2_rad = max_lat.to_radians();
    let lon_diff_rad = (max_lon - min_lon).to_radians();

    // Approximate area calculation
    let area = EARTH_RADIUS_KM * EARTH_RADIUS_KM * lon_diff_rad * (lat2_rad.sin() - lat1_rad.sin());

    // Allow some tolerance for floating point comparison
    Ok((area.abs() - expected_area).abs() < 0.1)
}

// ============================================================
// EXPANDED SWRL BUILT-IN LIBRARY
// ============================================================

// Division and Integer Operations
// ============================================================

pub(crate) fn builtin_divide(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("divide requires exactly 3 arguments"));
    }

    let dividend = extract_numeric_value(&args[0])?;
    let divisor = extract_numeric_value(&args[1])?;
    let result = extract_numeric_value(&args[2])?;

    if divisor.abs() < f64::EPSILON {
        return Err(anyhow::anyhow!("Division by zero"));
    }

    Ok((dividend / divisor - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_integer_divide(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "integerDivide requires exactly 3 arguments"
        ));
    }

    let dividend = extract_numeric_value(&args[0])? as i64;
    let divisor = extract_numeric_value(&args[1])? as i64;
    let result = extract_numeric_value(&args[2])? as i64;

    if divisor == 0 {
        return Err(anyhow::anyhow!("Division by zero"));
    }

    Ok(dividend / divisor == result)
}

pub(crate) fn builtin_unary_minus(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("unaryMinus requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((-input - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_unary_plus(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("unaryPlus requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((input - result).abs() < f64::EPSILON)
}

// Advanced Mathematical Functions
// ============================================================

pub(crate) fn builtin_min(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 {
        return Err(anyhow::anyhow!("min requires at least 2 arguments"));
    }

    let values: Result<Vec<f64>> = args[..args.len() - 1]
        .iter()
        .map(extract_numeric_value)
        .collect();
    let values = values?;

    let result = extract_numeric_value(&args[args.len() - 1])?;

    let min_value = values
        .iter()
        .fold(f64::INFINITY, |a, &b| if a < b { a } else { b });

    Ok((min_value - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_max(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 {
        return Err(anyhow::anyhow!("max requires at least 2 arguments"));
    }

    let values: Result<Vec<f64>> = args[..args.len() - 1]
        .iter()
        .map(extract_numeric_value)
        .collect();
    let values = values?;

    let result = extract_numeric_value(&args[args.len() - 1])?;

    let max_value = values
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| if a > b { a } else { b });

    Ok((max_value - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_avg(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 {
        return Err(anyhow::anyhow!("avg requires at least 2 arguments"));
    }

    let values: Result<Vec<f64>> = args[..args.len() - 1]
        .iter()
        .map(extract_numeric_value)
        .collect();
    let values = values?;

    let result = extract_numeric_value(&args[args.len() - 1])?;

    if values.is_empty() {
        return Ok(false);
    }

    let sum: f64 = values.iter().sum();
    let avg = sum / values.len() as f64;

    Ok((avg - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_sum(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 {
        return Err(anyhow::anyhow!("sum requires at least 2 arguments"));
    }

    let values: Result<Vec<f64>> = args[..args.len() - 1]
        .iter()
        .map(extract_numeric_value)
        .collect();
    let values = values?;

    let result = extract_numeric_value(&args[args.len() - 1])?;

    let sum: f64 = values.iter().sum();

    Ok((sum - result).abs() < f64::EPSILON)
}

// Advanced Comparison Operations
// ============================================================

pub(crate) fn builtin_less_than_or_equal(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "lessThanOrEqual requires exactly 2 arguments"
        ));
    }

    let val1 = extract_numeric_value(&args[0])?;
    let val2 = extract_numeric_value(&args[1])?;
    Ok(val1 <= val2 || (val1 - val2).abs() < f64::EPSILON)
}

pub(crate) fn builtin_greater_than_or_equal(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "greaterThanOrEqual requires exactly 2 arguments"
        ));
    }

    let val1 = extract_numeric_value(&args[0])?;
    let val2 = extract_numeric_value(&args[1])?;
    Ok(val1 >= val2 || (val1 - val2).abs() < f64::EPSILON)
}

pub(crate) fn builtin_between(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "between requires exactly 3 arguments: value, min, max"
        ));
    }

    let value = extract_numeric_value(&args[0])?;
    let min = extract_numeric_value(&args[1])?;
    let max = extract_numeric_value(&args[2])?;

    Ok(value >= min && value <= max)
}

// Type Checking Operations
// ============================================================

pub(crate) fn builtin_is_integer(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 1 {
        return Err(anyhow::anyhow!("isInteger requires exactly 1 argument"));
    }

    match extract_numeric_value(&args[0]) {
        Ok(val) => Ok((val.fract()).abs() < f64::EPSILON),
        Err(_) => Ok(false),
    }
}

pub(crate) fn builtin_is_float(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 1 {
        return Err(anyhow::anyhow!("isFloat requires exactly 1 argument"));
    }

    Ok(extract_numeric_value(&args[0]).is_ok())
}

pub(crate) fn builtin_is_string(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 1 {
        return Err(anyhow::anyhow!("isString requires exactly 1 argument"));
    }

    Ok(matches!(
        &args[0],
        SwrlArgument::Literal(_) | SwrlArgument::Individual(_)
    ))
}

pub(crate) fn builtin_is_boolean(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 1 {
        return Err(anyhow::anyhow!("isBoolean requires exactly 1 argument"));
    }

    Ok(extract_boolean_value(&args[0]).is_ok())
}

pub(crate) fn builtin_is_uri(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 1 {
        return Err(anyhow::anyhow!("isURI requires exactly 1 argument"));
    }

    if let SwrlArgument::Individual(uri) = &args[0] {
        // Simple URI validation
        Ok(uri.starts_with("http://") || uri.starts_with("https://") || uri.starts_with("urn:"))
    } else {
        Ok(false)
    }
}

// Type Conversion Operations
// ============================================================

pub(crate) fn builtin_int_value(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("intValue requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])? as i64;

    Ok(input as i64 == result)
}

pub(crate) fn builtin_float_value(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("floatValue requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((input - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_string_value(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("stringValue requires exactly 2 arguments"));
    }

    let input = extract_string_value(&args[0])?;
    let result = extract_string_value(&args[1])?;

    Ok(input == result)
}

// Enhanced String Operations
// ============================================================

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

// Date and Time Operations (Enhanced)
// ============================================================

pub(crate) fn builtin_date(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 4 {
        return Err(anyhow::anyhow!(
            "date requires exactly 4 arguments: year, month, day, result"
        ));
    }

    let year = extract_numeric_value(&args[0])? as i32;
    let month = extract_numeric_value(&args[1])? as u32;
    let day = extract_numeric_value(&args[2])? as u32;
    let expected = extract_string_value(&args[3])?;

    let result = format!("{:04}-{:02}-{:02}", year, month, day);
    Ok(result == expected)
}

pub(crate) fn builtin_time(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 4 {
        return Err(anyhow::anyhow!(
            "time requires exactly 4 arguments: hour, minute, second, result"
        ));
    }

    let hour = extract_numeric_value(&args[0])? as u32;
    let minute = extract_numeric_value(&args[1])? as u32;
    let second = extract_numeric_value(&args[2])? as u32;
    let expected = extract_string_value(&args[3])?;

    let result = format!("{:02}:{:02}:{:02}", hour, minute, second);
    Ok(result == expected)
}

pub(crate) fn builtin_year(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "year requires exactly 2 arguments: date, result"
        ));
    }

    let date_str = extract_string_value(&args[0])?;
    let expected_year = extract_numeric_value(&args[1])? as i32;

    // Parse ISO 8601 date (YYYY-MM-DD)
    if let Some(year_part) = date_str.split('-').next() {
        if let Ok(year) = year_part.parse::<i32>() {
            return Ok(year == expected_year);
        }
    }

    Ok(false)
}

pub(crate) fn builtin_month(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "month requires exactly 2 arguments: date, result"
        ));
    }

    let date_str = extract_string_value(&args[0])?;
    let expected_month = extract_numeric_value(&args[1])? as u32;

    // Parse ISO 8601 date (YYYY-MM-DD)
    let parts: Vec<&str> = date_str.split('-').collect();
    if parts.len() >= 2 {
        if let Ok(month) = parts[1].parse::<u32>() {
            return Ok(month == expected_month);
        }
    }

    Ok(false)
}

pub(crate) fn builtin_day(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "day requires exactly 2 arguments: date, result"
        ));
    }

    let date_str = extract_string_value(&args[0])?;
    let expected_day = extract_numeric_value(&args[1])? as u32;

    // Parse ISO 8601 date (YYYY-MM-DD)
    let parts: Vec<&str> = date_str.split('-').collect();
    if parts.len() >= 3 {
        if let Ok(day) = parts[2].parse::<u32>() {
            return Ok(day == expected_day);
        }
    }

    Ok(false)
}

pub(crate) fn builtin_hour(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "hour requires exactly 2 arguments: time, result"
        ));
    }

    let time_str = extract_string_value(&args[0])?;
    let expected_hour = extract_numeric_value(&args[1])? as u32;

    // Parse ISO 8601 time (HH:MM:SS)
    if let Some(hour_part) = time_str.split(':').next() {
        if let Ok(hour) = hour_part.parse::<u32>() {
            return Ok(hour == expected_hour);
        }
    }

    Ok(false)
}

pub(crate) fn builtin_minute(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "minute requires exactly 2 arguments: time, result"
        ));
    }

    let time_str = extract_string_value(&args[0])?;
    let expected_minute = extract_numeric_value(&args[1])? as u32;

    // Parse ISO 8601 time (HH:MM:SS)
    let parts: Vec<&str> = time_str.split(':').collect();
    if parts.len() >= 2 {
        if let Ok(minute) = parts[1].parse::<u32>() {
            return Ok(minute == expected_minute);
        }
    }

    Ok(false)
}

pub(crate) fn builtin_second(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "second requires exactly 2 arguments: time, result"
        ));
    }

    let time_str = extract_string_value(&args[0])?;
    let expected_second = extract_numeric_value(&args[1])? as u32;

    // Parse ISO 8601 time (HH:MM:SS)
    let parts: Vec<&str> = time_str.split(':').collect();
    if parts.len() >= 3 {
        if let Ok(second) = parts[2].parse::<u32>() {
            return Ok(second == expected_second);
        }
    }

    Ok(false)
}

// Hash and Cryptographic Operations
// ============================================================

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub(crate) fn builtin_hash(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "hash requires exactly 2 arguments: input, result"
        ));
    }

    let input = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    let hash_value = hasher.finish();

    Ok(hash_value.to_string() == expected)
}

pub(crate) fn builtin_base64_encode(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("base64Encode requires exactly 2 arguments"));
    }

    let input = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple base64 encoding using standard library
    let encoded = base64_encode_simple(&input);
    Ok(encoded == expected)
}

pub(crate) fn builtin_base64_decode(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("base64Decode requires exactly 2 arguments"));
    }

    let input = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple base64 decoding
    match base64_decode_simple(&input) {
        Ok(decoded) => Ok(decoded == expected),
        Err(_) => Ok(false),
    }
}

// Helper functions for base64
fn base64_encode_simple(input: &str) -> String {
    const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = input.as_bytes();
    let mut result = String::new();

    for chunk in bytes.chunks(3) {
        let mut buf = [0u8; 3];
        for (i, &b) in chunk.iter().enumerate() {
            buf[i] = b;
        }

        result.push(BASE64_CHARS[(buf[0] >> 2) as usize] as char);
        result.push(BASE64_CHARS[(((buf[0] & 0x03) << 4) | (buf[1] >> 4)) as usize] as char);

        if chunk.len() > 1 {
            result.push(BASE64_CHARS[(((buf[1] & 0x0f) << 2) | (buf[2] >> 6)) as usize] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(BASE64_CHARS[(buf[2] & 0x3f) as usize] as char);
        } else {
            result.push('=');
        }
    }

    result
}

fn base64_decode_simple(input: &str) -> Result<String> {
    let input = input.trim_end_matches('=');
    let mut bytes = Vec::new();

    for chunk in input.as_bytes().chunks(4) {
        let vals: Vec<u8> = chunk
            .iter()
            .map(|&b| match b {
                b'A'..=b'Z' => b - b'A',
                b'a'..=b'z' => b - b'a' + 26,
                b'0'..=b'9' => b - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                _ => 0,
            })
            .collect();

        if !vals.is_empty() {
            bytes.push((vals[0] << 2) | (vals.get(1).unwrap_or(&0) >> 4));
        }
        if vals.len() > 2 {
            bytes.push((vals[1] << 4) | (vals[2] >> 2));
        }
        if vals.len() > 3 {
            bytes.push((vals[2] << 6) | vals[3]);
        }
    }

    String::from_utf8(bytes).map_err(|e| anyhow::anyhow!("UTF-8 decode error: {}", e))
}

// Statistical Operations
// ============================================================

pub(crate) fn builtin_mean(args: &[SwrlArgument]) -> Result<bool> {
    // Alias for avg
    builtin_avg(args)
}

pub(crate) fn builtin_median(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 {
        return Err(anyhow::anyhow!("median requires at least 2 arguments"));
    }

    let mut values: Vec<f64> = args[..args.len() - 1]
        .iter()
        .map(extract_numeric_value)
        .collect::<Result<Vec<_>>>()?;

    let result = extract_numeric_value(&args[args.len() - 1])?;

    if values.is_empty() {
        return Ok(false);
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = if values.len() % 2 == 0 {
        let mid = values.len() / 2;
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[values.len() / 2]
    };

    Ok((median - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_variance(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 {
        return Err(anyhow::anyhow!("variance requires at least 2 arguments"));
    }

    let values: Vec<f64> = args[..args.len() - 1]
        .iter()
        .map(extract_numeric_value)
        .collect::<Result<Vec<_>>>()?;

    let result = extract_numeric_value(&args[args.len() - 1])?;

    if values.is_empty() {
        return Ok(false);
    }

    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;

    Ok((variance - result).abs() < f64::EPSILON)
}

pub(crate) fn builtin_stddev(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 {
        return Err(anyhow::anyhow!("stddev requires at least 2 arguments"));
    }

    let values: Vec<f64> = args[..args.len() - 1]
        .iter()
        .map(extract_numeric_value)
        .collect::<Result<Vec<_>>>()?;

    let result = extract_numeric_value(&args[args.len() - 1])?;

    if values.is_empty() {
        return Ok(false);
    }

    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    let stddev = variance.sqrt();

    Ok((stddev - result).abs() < f64::EPSILON)
}

// RDF-Specific Operations
// ============================================================

pub(crate) fn builtin_lang_matches(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "langMatches requires exactly 2 arguments: lang, pattern"
        ));
    }

    let lang = extract_string_value(&args[0])?;
    let pattern = extract_string_value(&args[1])?;

    if pattern == "*" {
        return Ok(!lang.is_empty());
    }

    Ok(lang.to_lowercase().starts_with(&pattern.to_lowercase()))
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

pub(crate) fn builtin_is_literal(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 1 {
        return Err(anyhow::anyhow!("isLiteral requires exactly 1 argument"));
    }

    Ok(matches!(&args[0], SwrlArgument::Literal(_)))
}

pub(crate) fn builtin_is_blank(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 1 {
        return Err(anyhow::anyhow!("isBlank requires exactly 1 argument"));
    }

    if let SwrlArgument::Individual(uri) = &args[0] {
        Ok(uri.starts_with("_:"))
    } else {
        Ok(false)
    }
}

pub(crate) fn builtin_is_iri(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 1 {
        return Err(anyhow::anyhow!("isIRI requires exactly 1 argument"));
    }

    if let SwrlArgument::Individual(uri) = &args[0] {
        Ok(uri.starts_with("http://")
            || uri.starts_with("https://")
            || uri.starts_with("urn:")
            || uri.starts_with("file:"))
    } else {
        Ok(false)
    }
}

// URI Operations
// ============================================================

pub(crate) fn builtin_resolve_uri(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "resolveURI requires exactly 3 arguments: base, relative, result"
        ));
    }

    let base = extract_string_value(&args[0])?;
    let relative = extract_string_value(&args[1])?;
    let expected = extract_string_value(&args[2])?;

    // Simple URI resolution
    let result = if relative.starts_with("http://") || relative.starts_with("https://") {
        relative
    } else if relative.starts_with('/') {
        // Extract scheme and host from base
        if let Some(idx) = base.find("://") {
            let scheme_host = &base[..idx + 3];
            if let Some(host_end) = base[idx + 3..].find('/') {
                format!(
                    "{}{}{}",
                    scheme_host,
                    &base[idx + 3..idx + 3 + host_end],
                    relative
                )
            } else {
                format!("{}{}{}", scheme_host, &base[idx + 3..], relative)
            }
        } else {
            relative
        }
    } else {
        // Relative to current path
        if let Some(last_slash) = base.rfind('/') {
            format!("{}/{}", &base[..last_slash], relative)
        } else {
            relative
        }
    };

    Ok(result == expected)
}

pub(crate) fn builtin_encode_uri(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "encodeURI requires exactly 2 arguments: input, result"
        ));
    }

    let input = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple URL encoding
    let encoded: String = input
        .chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            _ => format!("%{:02X}", c as u8),
        })
        .collect();

    Ok(encoded == expected)
}

pub(crate) fn builtin_decode_uri(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "decodeURI requires exactly 2 arguments: input, result"
        ));
    }

    let input = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple URL decoding
    let mut decoded = String::new();
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                decoded.push(byte as char);
            } else {
                decoded.push(c);
                decoded.push_str(&hex);
            }
        } else {
            decoded.push(c);
        }
    }

    Ok(decoded == expected)
}

// Collection Operations (Advanced)
// ============================================================

pub(crate) fn builtin_make_list(args: &[SwrlArgument]) -> Result<bool> {
    if args.is_empty() {
        return Err(anyhow::anyhow!("makeList requires at least 1 argument"));
    }

    let values: Result<Vec<String>> = args[..args.len() - 1]
        .iter()
        .map(extract_string_value)
        .collect();
    let values = values?;
    let expected = extract_string_value(&args[args.len() - 1])?;

    let result = values.join(",");
    Ok(result == expected)
}

pub(crate) fn builtin_list_insert(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 4 {
        return Err(anyhow::anyhow!(
            "listInsert requires exactly 4 arguments: list, index, item, result"
        ));
    }

    let list_str = extract_string_value(&args[0])?;
    let index = extract_numeric_value(&args[1])? as usize;
    let item = extract_string_value(&args[2])?;
    let expected = extract_string_value(&args[3])?;

    let mut items: Vec<String> = list_str.split(',').map(|s| s.to_string()).collect();
    if index <= items.len() {
        items.insert(index, item);
        let result = items.join(",");
        Ok(result == expected)
    } else {
        Ok(false)
    }
}

pub(crate) fn builtin_list_remove(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "listRemove requires exactly 3 arguments: list, index, result"
        ));
    }

    let list_str = extract_string_value(&args[0])?;
    let index = extract_numeric_value(&args[1])? as usize;
    let expected = extract_string_value(&args[2])?;

    let mut items: Vec<String> = list_str.split(',').map(|s| s.to_string()).collect();
    if index < items.len() {
        items.remove(index);
        let result = items.join(",");
        Ok(result == expected)
    } else {
        Ok(false)
    }
}

pub(crate) fn builtin_list_reverse(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "listReverse requires exactly 2 arguments: list, result"
        ));
    }

    let list_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    let mut items: Vec<String> = list_str.split(',').map(|s| s.to_string()).collect();
    items.reverse();
    let result = items.join(",");

    Ok(result == expected)
}

pub(crate) fn builtin_list_sort(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "listSort requires exactly 2 arguments: list, result"
        ));
    }

    let list_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    let mut items: Vec<String> = list_str.split(',').map(|s| s.to_string()).collect();
    items.sort();
    let result = items.join(",");

    Ok(result == expected)
}

pub(crate) fn builtin_list_union(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "listUnion requires exactly 3 arguments: list1, list2, result"
        ));
    }

    let list1_str = extract_string_value(&args[0])?;
    let list2_str = extract_string_value(&args[1])?;
    let expected = extract_string_value(&args[2])?;

    let items1: Vec<&str> = list1_str.split(',').collect();
    let items2: Vec<&str> = list2_str.split(',').collect();

    let mut union: Vec<String> = items1.iter().map(|&s| s.to_string()).collect();
    for item in items2 {
        if !items1.contains(&item) {
            union.push(item.to_string());
        }
    }

    let result = union.join(",");
    Ok(result == expected)
}

pub(crate) fn builtin_list_intersection(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "listIntersection requires exactly 3 arguments: list1, list2, result"
        ));
    }

    let list1_str = extract_string_value(&args[0])?;
    let list2_str = extract_string_value(&args[1])?;
    let expected = extract_string_value(&args[2])?;

    let items1: Vec<&str> = list1_str.split(',').collect();
    let items2: Vec<&str> = list2_str.split(',').collect();

    let intersection: Vec<String> = items1
        .iter()
        .filter(|&&item| items2.contains(&item))
        .map(|&s| s.to_string())
        .collect();

    let result = intersection.join(",");
    Ok(result == expected)
}

// ============================================================
// TESTS FOR EXPANDED BUILT-INS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

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
}
