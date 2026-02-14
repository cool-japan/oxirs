//! # ReteNetwork - clear_group Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::Token;

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Clear all facts and reset the network
    pub fn clear(&mut self) {
        self.alpha_memory.clear();
        self.beta_memory.clear();
        self.enhanced_beta_nodes.clear();
        for tokens in self.token_memory.values_mut() {
            tokens.clear();
        }
        self.token_memory.insert(self.root_id, vec![Token::new()]);
    }
}
