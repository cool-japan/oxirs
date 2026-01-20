//! SWRL Date/Time Built-in Functions
//!
//! This module implements date and time operations for SWRL rules including:
//! - Date/time construction: date, time, date_time
//! - Duration operations: day_time_duration, year_month_duration, interval_duration
//! - Arithmetic: date_add, date_diff, now
//! - Component extraction: year, month, day, hour, minute, second
//! - Temporal relations: temporal_before, temporal_after, temporal_during, temporal_overlaps, temporal_meets

use anyhow::Result;

use super::super::types::SwrlArgument;
use super::utils::*;

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
