//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::forward::Substitution;
use crate::RuleAtom;
use std::collections::HashMap;

use super::functions::NodeId;

/// Token representing partial matches flowing through the network
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// Current variable bindings
    pub bindings: Substitution,
    /// Tags for tracking token origin
    pub tags: Vec<String>,
    /// Facts that contributed to this token
    pub facts: Vec<RuleAtom>,
}
impl Token {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            tags: Vec::new(),
            facts: Vec::new(),
        }
    }
    pub fn with_fact(fact: RuleAtom) -> Self {
        Self {
            bindings: HashMap::new(),
            tags: Vec::new(),
            facts: vec![fact],
        }
    }
}
/// Types of RETE nodes
#[derive(Debug, Clone)]
pub enum ReteNode {
    /// Root node - entry point for all facts
    Root,
    /// Alpha node - tests individual fact patterns
    Alpha {
        pattern: RuleAtom,
        children: Vec<NodeId>,
    },
    /// Beta node - joins two streams of tokens
    Beta {
        left_parent: NodeId,
        right_parent: NodeId,
        join_condition: JoinCondition,
        children: Vec<NodeId>,
    },
    /// Production node - executes rule actions
    Production {
        rule_name: String,
        rule_head: Vec<RuleAtom>,
        parent: NodeId,
    },
}
/// RETE network statistics
#[derive(Debug, Clone)]
pub struct ReteStats {
    pub total_nodes: usize,
    pub alpha_nodes: usize,
    pub beta_nodes: usize,
    pub production_nodes: usize,
    pub total_tokens: usize,
}
/// Enhanced RETE statistics with beta join performance metrics
#[derive(Debug, Clone)]
pub struct EnhancedReteStats {
    pub basic: ReteStats,
    pub total_beta_joins: usize,
    pub successful_beta_joins: usize,
    pub join_success_rate: f64,
    pub memory_evictions: usize,
    pub peak_memory_usage: usize,
    pub enhanced_nodes: usize,
}
/// Join condition for beta nodes
#[derive(Debug, Clone, Default)]
pub struct JoinCondition {
    /// Variable constraints between left and right tokens
    pub constraints: Vec<(String, String)>,
    /// Additional filters
    pub filters: Vec<String>,
}
