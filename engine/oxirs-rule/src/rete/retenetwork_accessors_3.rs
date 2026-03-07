//! # ReteNetwork - accessors Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::rete_enhanced::MemoryStrategy;

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Set memory strategy for all beta nodes
    pub fn set_memory_strategy(&mut self, strategy: MemoryStrategy) {
        self.memory_strategy = strategy;
        for node in self.enhanced_beta_nodes.values_mut() {
            node.memory.set_memory_strategy(strategy);
        }
    }
}
