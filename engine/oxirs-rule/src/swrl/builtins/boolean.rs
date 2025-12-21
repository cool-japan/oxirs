//! SWRL Boolean Built-in Functions
//!
//! This module implements boolean operations for SWRL rules.

use anyhow::Result;

use super::super::types::SwrlArgument;
use super::utils::*;

pub(crate) fn builtin_boolean_value(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 1 {
        return Err(anyhow::anyhow!("booleanValue requires exactly 1 argument"));
    }

    extract_boolean_value(&args[0])
}
