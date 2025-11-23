//! SWRL Type Checking Built-in Functions
//!
//! This module implements type checking and conversion operations for SWRL rules including:
//! - Type checks: is_integer, is_float, is_string, is_boolean, is_uri, is_literal, is_blank, is_iri
//! - Type conversions: int_value, float_value, string_value

use anyhow::Result;

use super::super::types::SwrlArgument;
use super::utils::*;

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
