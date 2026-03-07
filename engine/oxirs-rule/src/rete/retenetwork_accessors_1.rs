//! # ReteNetwork - accessors Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::RuleAtom;
use anyhow::Result;

use super::functions::NodeId;
use super::types::ReteNode;

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Get the pattern associated with a node
    pub(super) fn get_node_pattern(&self, node_id: NodeId) -> Result<Option<RuleAtom>> {
        match self.nodes.get(&node_id) {
            Some(ReteNode::Alpha { pattern, .. }) => Ok(Some(pattern.clone())),
            Some(ReteNode::Beta { left_parent, .. }) => self.get_node_pattern(*left_parent),
            _ => Ok(None),
        }
    }
}
