//! SWRL Built-in Utility Functions
//!
//! Helper functions for extracting and converting SWRL argument values.

use super::super::types::SwrlArgument;
use anyhow::Result;

/// Extract numeric value from SWRL argument
pub(crate) fn extract_numeric_value(arg: &SwrlArgument) -> Result<f64> {
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

/// Extract string value from SWRL argument
pub(crate) fn extract_string_value(arg: &SwrlArgument) -> Result<String> {
    match arg {
        SwrlArgument::Literal(value) => Ok(value.clone()),
        SwrlArgument::Individual(value) => Ok(value.clone()),
        SwrlArgument::Variable(name) => Err(anyhow::anyhow!("Unbound variable: {}", name)),
    }
}

/// Extract boolean value from SWRL argument
pub(crate) fn extract_boolean_value(arg: &SwrlArgument) -> Result<bool> {
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
