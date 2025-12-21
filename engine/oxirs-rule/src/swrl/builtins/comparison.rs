//! SWRL Comparison Built-in Functions
//!
//! This module implements comparison operations for SWRL rules including:
//! - equal, not_equal
//! - less_than, greater_than
//! - less_than_or_equal, greater_than_or_equal
//! - between

use anyhow::Result;

use super::super::types::SwrlArgument;
use super::utils::*;

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
