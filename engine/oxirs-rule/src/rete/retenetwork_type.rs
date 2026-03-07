//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::rete_enhanced::{BetaJoinNode, ConflictResolution, MemoryStrategy};
use crate::RuleAtom;
use std::collections::{HashMap, HashSet};

use super::functions::NodeId;
use super::types::{ReteNode, Token};

/// RETE network implementation
#[derive(Debug)]
pub struct ReteNetwork {
    /// Network nodes indexed by ID
    pub(super) nodes: HashMap<NodeId, ReteNode>,
    /// Next available node ID
    pub(super) next_node_id: NodeId,
    /// Token memory for each node
    pub(super) token_memory: HashMap<NodeId, Vec<Token>>,
    /// Alpha memory for alpha nodes
    pub(super) alpha_memory: HashMap<NodeId, HashSet<RuleAtom>>,
    /// Beta memory for beta nodes (left and right)
    pub(super) beta_memory: HashMap<NodeId, (Vec<Token>, Vec<Token>)>,
    /// Enhanced beta join nodes
    pub(super) enhanced_beta_nodes: HashMap<NodeId, BetaJoinNode>,
    /// Root node ID
    pub(super) root_id: NodeId,
    /// Pattern to alpha node mapping for efficiency
    pub(super) pattern_index: HashMap<String, Vec<NodeId>>,
    /// Debug mode
    pub(super) debug_mode: bool,
    /// Memory management strategy
    pub(super) memory_strategy: MemoryStrategy,
    /// Conflict resolution strategy
    pub(super) conflict_resolution: ConflictResolution,
}
