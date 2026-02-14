//! # ReteNetwork - accessors Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::rete_enhanced::ConflictResolution;

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Set conflict resolution strategy
    pub fn set_conflict_resolution(&mut self, strategy: ConflictResolution) {
        self.conflict_resolution = strategy;
        for node in self.enhanced_beta_nodes.values_mut() {
            node.conflict_resolution = strategy;
        }
    }
}
