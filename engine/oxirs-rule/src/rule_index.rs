//! # Rule Indexing for High-Performance Rule Matching
//!
//! This module provides indexed rule lookup for efficient rule matching in large rule sets.
//! Instead of sequential scanning through all rules, this module enables O(1) lookup
//! based on predicate and first-argument patterns.
//!
//! ## Features
//!
//! - **Predicate Indexing**: Index rules by their body predicate patterns
//! - **First-Argument Indexing**: Additional indexing by first argument for common patterns
//! - **Hash-Based Lookup**: O(1) average case retrieval
//! - **Index Statistics**: Track hit rates and performance metrics
//! - **Automatic Maintenance**: Self-updating indices on rule addition/removal
//!
//! ## Performance Impact
//!
//! - **Without indexing**: O(n) rule scan per fact (n = number of rules)
//! - **With indexing**: O(1) average lookup + O(m) matching rules (m << n typically)
//! - **Expected speedup**: 10-100x for large rule sets (100+ rules)
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::rule_index::{RuleIndex, IndexConfig};
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! let config = IndexConfig::default()
//!     .with_predicate_indexing(true)
//!     .with_first_arg_indexing(true);
//!
//! let mut index = RuleIndex::new(config);
//!
//! // Add rules to index
//! let rule = Rule {
//!     name: "ancestor".to_string(),
//!     body: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("parent".to_string()),
//!         object: Term::Variable("Y".to_string()),
//!     }],
//!     head: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("ancestor".to_string()),
//!         object: Term::Variable("Y".to_string()),
//!     }],
//! };
//!
//! index.add_rule(rule);
//!
//! // Fast lookup by predicate
//! let matching_rules = index.find_rules_by_predicate("parent");
//! ```

// Re-export all public types from the sibling modules
pub use crate::rule_index_store::{wildcard_matches, PriorityIndex, RuleIndex, RuleIndexBuilder};
pub use crate::rule_index_types::{
    ArgType, CombinedKey, DependencyEdge, FirstArgKey, IndexConfig, IndexStatistics,
    IndexStatisticsSnapshot, PredicateKey, PrioritizedRule, RuleDependencyGraph, RuleId,
};
