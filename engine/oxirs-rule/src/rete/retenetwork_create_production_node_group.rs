//! # ReteNetwork - create_production_node_group Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::RuleAtom;
use anyhow::Result;
use tracing::debug;

use super::functions::NodeId;
use super::types::ReteNode;

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Create a production node
    pub(super) fn create_production_node(
        &mut self,
        rule_name: &str,
        rule_head: &[RuleAtom],
        parent: NodeId,
    ) -> Result<NodeId> {
        let node_id = self.create_node(ReteNode::Production {
            rule_name: rule_name.to_string(),
            rule_head: rule_head.to_vec(),
            parent,
        });
        self.add_child(parent, node_id)?;
        if self.debug_mode {
            debug!(
                "Created production node {} for rule '{}'",
                node_id, rule_name
            );
        }
        Ok(node_id)
    }
    /// Add a child to a node
    pub(super) fn add_child(&mut self, parent_id: NodeId, child_id: NodeId) -> Result<()> {
        match self.nodes.get_mut(&parent_id) {
            Some(ReteNode::Alpha { children, .. }) => {
                children.push(child_id);
            }
            Some(ReteNode::Beta { children, .. }) => {
                children.push(child_id);
            }
            _ => {
                return Err(anyhow::anyhow!("Cannot add child to node type"));
            }
        }
        Ok(())
    }
}
