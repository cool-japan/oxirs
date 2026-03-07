//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::RifDialect;

/// RIF Compact Syntax Parser
#[derive(Debug)]
pub struct RifParser {
    /// Target dialect
    pub(super) dialect: RifDialect,
    /// Collected prefixes
    pub(super) prefixes: HashMap<String, String>,
    /// Current position in input
    pub(super) pos: usize,
    /// Input text
    pub(super) input: String,
}
