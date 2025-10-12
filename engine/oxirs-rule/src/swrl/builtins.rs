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
