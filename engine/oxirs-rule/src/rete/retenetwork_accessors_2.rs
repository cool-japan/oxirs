//! # ReteNetwork - accessors Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{EnhancedReteStats, ReteNode, ReteStats};

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Get network statistics
    pub fn get_stats(&self) -> ReteStats {
        let mut alpha_count = 0;
        let mut beta_count = 0;
        let mut production_count = 0;
        let mut total_tokens = 0;
        for node in self.nodes.values() {
            match node {
                ReteNode::Alpha { .. } => alpha_count += 1,
                ReteNode::Beta { .. } => beta_count += 1,
                ReteNode::Production { .. } => production_count += 1,
                _ => {}
            }
        }
        for tokens in self.token_memory.values() {
            total_tokens += tokens.len();
        }
        ReteStats {
            total_nodes: self.nodes.len(),
            alpha_nodes: alpha_count,
            beta_nodes: beta_count,
            production_nodes: production_count,
            total_tokens,
        }
    }
    /// Get enhanced network statistics including beta join performance
    pub fn get_enhanced_stats(&self) -> EnhancedReteStats {
        let basic_stats = self.get_stats();
        let mut total_joins = 0;
        let mut successful_joins = 0;
        let mut total_evictions = 0;
        let mut peak_memory = 0;
        for enhanced_node in self.enhanced_beta_nodes.values() {
            let stats = enhanced_node.get_stats();
            total_joins += stats.total_joins;
            successful_joins += stats.successful_joins;
            total_evictions += stats.evictions;
            peak_memory = peak_memory.max(stats.peak_size);
        }
        EnhancedReteStats {
            basic: basic_stats,
            total_beta_joins: total_joins,
            successful_beta_joins: successful_joins,
            join_success_rate: if total_joins > 0 {
                successful_joins as f64 / total_joins as f64
            } else {
                0.0
            },
            memory_evictions: total_evictions,
            peak_memory_usage: peak_memory,
            enhanced_nodes: self.enhanced_beta_nodes.len(),
        }
    }
}
