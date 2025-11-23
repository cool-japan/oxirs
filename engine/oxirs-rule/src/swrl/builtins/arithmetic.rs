//! SWRL Arithmetic Built-in Functions
//!
//! This module implements mathematical operations for SWRL rules including:
//! - Basic arithmetic: add, subtract, multiply, divide, integer_divide, mod
//! - Unary operations: unary_minus, unary_plus, abs, sqrt, pow
//! - Rounding: floor, ceil, round
//! - Trigonometric: sin, cos, tan, asin, acos, atan
//! - Logarithmic: log, exp
//! - Aggregate: min, max, avg, sum, mean, median
//! - Statistical: variance, stddev

use anyhow::Result;

use super::super::types::SwrlArgument;
use super::utils::*;

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
