//! SWRL Language Tag Built-in Functions
//!
//! This module implements language tag matching operations for SWRL rules.

use anyhow::Result;

use super::super::types::SwrlArgument;
use super::utils::*;

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
