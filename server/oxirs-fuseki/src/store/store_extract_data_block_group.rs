//! # Store - extract_data_block_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Extract data block from SPARQL update (between { and })
    pub(super) fn extract_data_block(&self, sparql: &str, operation: &str) -> FusekiResult<String> {
        let operation_upper = operation.to_uppercase();
        let sparql_upper = sparql.to_uppercase();
        let operation_pos = sparql_upper.find(&operation_upper).ok_or_else(|| {
            FusekiError::update_execution(format!("Operation '{operation}' not found"))
        })?;
        let remaining = &sparql[operation_pos + operation.len()..];
        let open_brace_pos = remaining.find('{').ok_or_else(|| {
            FusekiError::update_execution("Opening brace '{' not found".to_string())
        })?;
        let mut brace_count = 0;
        let mut close_brace_pos = None;
        // Track the matching '}' by BYTE offset (via `char_indices`), not by the
        // `chars().enumerate()` char index used previously. `open_brace_pos` is a
        // byte offset (`find`), so mixing it with a char-index end bound made the
        // final slice `&remaining[open_brace_pos + 1..close_brace_pos]` land in
        // the middle of a multibyte code point (Japanese, emoji, …) inside the
        // data block and panic ("byte index N is not a char boundary"). Iterating
        // `char_indices` keeps both bounds byte-consistent and valid.
        for (byte_idx, ch) in remaining
            .char_indices()
            .skip_while(|&(b, _)| b < open_brace_pos)
        {
            match ch {
                '{' => brace_count += 1,
                '}' => {
                    brace_count -= 1;
                    if brace_count == 0 {
                        close_brace_pos = Some(byte_idx);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close_brace_pos = close_brace_pos.ok_or_else(|| {
            FusekiError::update_execution("Matching closing brace '}' not found".to_string())
        })?;
        // `open_brace_pos` indexes the ASCII '{' (1 byte); `close_brace_pos` is the
        // byte offset of the matching ASCII '}'. Both are valid char boundaries.
        let data_block = &remaining[open_brace_pos + 1..close_brace_pos];
        Ok(data_block.trim().to_string())
    }
}
