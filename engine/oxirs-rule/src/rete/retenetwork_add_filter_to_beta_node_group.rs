//! # ReteNetwork - add_filter_to_beta_node_group Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use tracing::debug;

use super::functions::NodeId;

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Add a filter condition to an existing beta join node
    pub(super) fn add_filter_to_beta_node(
        &mut self,
        beta_id: NodeId,
        condition: crate::rete_enhanced::JoinCondition,
    ) -> Result<()> {
        if let Some(enhanced_node) = self.enhanced_beta_nodes.get_mut(&beta_id) {
            if self.debug_mode {
                debug!(
                    "Adding filter condition {:?} to beta node {}",
                    condition, beta_id
                );
            }
            enhanced_node.conditions.push(condition);
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Beta node {} not found for adding filter",
                beta_id
            ))
        }
    }
}
