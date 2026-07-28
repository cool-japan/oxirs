//! # ReteNetwork - clear_group Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::Token;

use super::retenetwork_type::ReteNetwork;

impl ReteNetwork {
    /// Clear all facts and reset the network for reuse.
    ///
    /// This clears the runtime *fact* state (alpha/beta token memories) but
    /// deliberately preserves the compiled rule graph — both `nodes` and the
    /// `enhanced_beta_nodes` map. Dropping the enhanced beta nodes (as an earlier
    /// version did) left stale `Beta` node ids in `nodes` with no enhanced entry,
    /// forcing the next join onto the fallback path whose `apply_filter` cannot
    /// evaluate compiled NotEqual/GreaterThan/LessThan/type/domain-range
    /// conditions — silently producing over-broad, filter-ignoring joins after a
    /// clear()-then-reuse cycle. We instead reset each enhanced node's token
    /// memory in place, keeping its compiled conditions intact.
    pub fn clear(&mut self) {
        // Preserve alpha node keys, drop only their stored facts.
        for facts in self.alpha_memory.values_mut() {
            facts.clear();
        }
        // Preserve beta memory keys, drop only their stored tokens.
        for (left, right) in self.beta_memory.values_mut() {
            left.clear();
            right.clear();
        }
        // Preserve compiled enhanced beta nodes, reset only their token memory.
        for enhanced in self.enhanced_beta_nodes.values_mut() {
            enhanced.clear_memory();
        }
        for tokens in self.token_memory.values_mut() {
            tokens.clear();
        }
        self.token_memory.insert(self.root_id, vec![Token::new()]);
    }
}
