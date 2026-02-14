//! # ReteNetwork - accessors Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Enable or disable debug mode
    pub fn set_debug_mode(&mut self, debug: bool) {
        self.debug_mode = debug;
    }
}
