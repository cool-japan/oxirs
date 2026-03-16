//! # OxiRS Rule Engine
//!
//! [![Version](https://img.shields.io/badge/version-0.2.2-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-rule/badge.svg)](https://docs.rs/oxirs-rule)
//!
//! **Status**: Production Release (v0.2.2)
//! **Stability**: Public APIs are stable. Production-ready with comprehensive testing.
//!
//! Forward/backward rule engine for RDFS, OWL, and SWRL reasoning with RETE optimization.
//! Provides Jena-compatible rule-based inference for knowledge graphs.
//!
//! ## Features
//!
//! - **Forward Chaining** - Data-driven rule inference
//! - **Backward Chaining** - Goal-driven query answering
//! - **RETE Algorithm** - Efficient pattern matching
//! - **RDFS/OWL Support** - Standard ontology reasoning
//! - **SWRL Integration** - Semantic Web Rule Language
//! - **Rule Composition** - Complex multi-step reasoning
//!
//! ## Quick Start
//!
//! ```rust
//! use oxirs_rule::{RuleEngine, Rule, RuleAtom, Term};
//!
//! let mut engine = RuleEngine::new();
//!
//! // Define rule: parent(X,Y) → ancestor(X,Y)
//! engine.add_rule(Rule {
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
//! });
//!
//! // Add facts and infer
//! let facts = vec![RuleAtom::Triple {
//!     subject: Term::Constant("john".to_string()),
//!     predicate: Term::Constant("parent".to_string()),
//!     object: Term::Constant("mary".to_string()),
//! }];
//!
//! let results = engine.forward_chain(&facts)?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## See Also
//!
//! - [`oxirs-core`](https://docs.rs/oxirs-core) - RDF data model
//! - [`oxirs-arq`](https://docs.rs/oxirs-arq) - SPARQL query engine

use anyhow::Result;

pub mod active_learning;
pub mod adaptive_strategies;
pub mod advanced_integration_example;
pub mod asp;
pub mod backward;
pub mod benchmark_suite;
pub mod cache;
pub mod chr;
pub mod composition;
pub mod comprehensive_tutorial;
pub mod conflict;
pub mod coverage;
pub mod datalog_engine;
pub mod debug;
pub mod dempster_shafer;
pub mod description_logic;
pub mod distributed;
pub mod entailment;
pub mod explainable_generation;
pub mod explanation;
pub mod forward;
pub mod fuzzy;
pub mod getting_started;
pub mod gpu_matching;
pub mod hermit_reasoner;
pub mod incremental;
pub mod integration;
pub mod integration_benchmarks;
pub mod language;
pub mod lazy_materialization;
pub mod lockfree;
pub mod materialization;
pub mod migration;
pub mod negation;
pub mod optimization;
pub mod owl;
pub mod owl_dl;
pub mod owl_el;
pub mod owl_profiles;
pub mod owl_ql;
pub mod owl_rl;
pub use owl_ql::{Owl2QLTBox, QueryAtom, QueryRewriter, RewrittenQuery};
pub mod n3logic;
pub mod parallel;
pub mod pellet_classifier;
pub mod performance;
pub mod possibilistic;
pub mod probabilistic;
pub mod probabilistic_rdf;
pub mod problog;
pub mod production_utils;
pub mod profiler;
pub mod quantum_optimizer;
pub mod rdf_integration;
pub mod rdf_processing_simple;
pub mod rdfs;
pub mod rete;
pub mod rete_enhanced;
pub mod rif;
pub mod rule_compression;
pub mod rule_index;
pub mod rule_learning;
pub mod rule_refinement;
pub mod shacl_integration;
pub mod simd_ops;
pub mod skos;
pub mod sparql_integration;
pub mod statistical_relational;
pub mod swrl;
pub mod tabling;
pub mod temporal;
pub mod test_generator;
pub mod transaction;
pub mod transfer_learning;
pub mod uncertainty_propagation;

/// Rule representation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Rule {
    pub name: String,
    pub body: Vec<RuleAtom>,
    pub head: Vec<RuleAtom>,
}

/// Rule atom (triple pattern or builtin)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum RuleAtom {
    Triple {
        subject: Term,
        predicate: Term,
        object: Term,
    },
    Builtin {
        name: String,
        args: Vec<Term>,
    },
    NotEqual {
        left: Term,
        right: Term,
    },
    GreaterThan {
        left: Term,
        right: Term,
    },
    LessThan {
        left: Term,
        right: Term,
    },
}

/// Rule term
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Term {
    Variable(String),
    Constant(String),
    Literal(String),
    Function { name: String, args: Vec<Term> },
}

/// Integrated rule engine combining all reasoning modes
#[derive(Debug)]
pub struct RuleEngine {
    rules: Vec<Rule>,
    forward_chainer: forward::ForwardChainer,
    backward_chainer: backward::BackwardChainer,
    rete_network: rete::ReteNetwork,
    cache: Option<crate::cache::RuleCache>,
}

impl RuleEngine {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            forward_chainer: forward::ForwardChainer::new(),
            backward_chainer: backward::BackwardChainer::new(),
            rete_network: rete::ReteNetwork::new(),
            cache: Some(crate::cache::RuleCache::new()),
        }
    }

    /// Add a rule to the engine
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule.clone());
        self.forward_chainer.add_rule(rule.clone());
        self.backward_chainer.add_rule(rule.clone());
        let _ = self.rete_network.add_rule(&rule);
    }

    /// Add multiple rules
    pub fn add_rules(&mut self, rules: Vec<Rule>) {
        for rule in rules {
            self.add_rule(rule);
        }
    }

    /// Add facts to the knowledge base
    pub fn add_facts(&mut self, facts: Vec<RuleAtom>) {
        self.forward_chainer.add_facts(facts.clone());
        self.backward_chainer.add_facts(facts);
    }

    /// Perform forward chaining inference
    pub fn forward_chain(&mut self, facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        self.forward_chainer.add_facts(facts.to_vec());
        self.forward_chainer.infer()
    }

    /// Perform backward chaining to prove a goal
    pub fn backward_chain(&mut self, goal: &RuleAtom) -> Result<bool> {
        self.backward_chainer.prove(goal)
    }

    /// Perform RETE-based forward chaining
    pub fn rete_forward_chain(&mut self, facts: Vec<RuleAtom>) -> Result<Vec<RuleAtom>> {
        self.rete_network.forward_chain(facts)
    }

    /// Get all current facts
    pub fn get_facts(&self) -> Vec<RuleAtom> {
        self.forward_chainer.get_facts()
    }

    /// Clear all facts and caches
    pub fn clear(&mut self) {
        self.forward_chainer.clear_facts();
        self.backward_chainer.clear_facts();
        self.rete_network.clear();
        self.clear_cache();
    }

    /// Set maximum proof depth for backward chaining
    ///
    /// This limits the recursion depth to prevent stack overflow.
    /// Lower values (e.g., 20-30) are safer for large datasets.
    /// Default is 100.
    pub fn set_backward_chain_max_depth(&mut self, max_depth: usize) {
        self.backward_chainer = backward::BackwardChainer::with_config(max_depth, false);
        // Re-add existing rules to the new backward chainer
        for rule in &self.rules {
            self.backward_chainer.add_rule(rule.clone());
        }
    }

    /// Add a single fact to the knowledge base
    pub fn add_fact(&mut self, fact: RuleAtom) {
        self.add_facts(vec![fact]);
    }

    /// Set cache for the rule engine
    ///
    /// Enables or disables caching by setting the cache instance.
    /// Pass `Some(cache)` to enable caching with a specific cache configuration,
    /// or `None` to disable caching.
    pub fn set_cache(&mut self, cache: Option<crate::cache::RuleCache>) {
        self.cache = cache;
        tracing::debug!(
            "Rule engine cache {}",
            if self.cache.is_some() {
                "enabled"
            } else {
                "disabled"
            }
        );
    }

    /// Get a reference to the cache
    ///
    /// Returns a reference to the cache if enabled.
    /// The cache is wrapped in Option to allow disabling caching.
    pub fn get_cache(&self) -> Option<&crate::cache::RuleCache> {
        self.cache.as_ref()
    }

    /// Get mutable reference to the cache
    ///
    /// Returns a mutable reference to the cache if enabled.
    /// Useful for clearing cache, updating statistics, etc.
    pub fn get_cache_mut(&mut self) -> Option<&mut crate::cache::RuleCache> {
        self.cache.as_mut()
    }

    /// Enable caching with default settings
    ///
    /// Creates a new cache with default configuration and enables it.
    pub fn enable_cache(&mut self) {
        self.cache = Some(crate::cache::RuleCache::new());
        tracing::info!("Rule engine cache enabled with default settings");
    }

    /// Disable caching
    ///
    /// Removes the cache and disables caching functionality.
    pub fn disable_cache(&mut self) {
        self.cache = None;
        tracing::info!("Rule engine cache disabled");
    }

    /// Check if caching is enabled
    pub fn is_cache_enabled(&self) -> bool {
        self.cache.is_some()
    }

    /// Clear the cache if enabled
    ///
    /// Clears all cached rule results, derivations, unifications, and patterns.
    pub fn clear_cache(&mut self) {
        if let Some(cache) = &self.cache {
            cache.clear_all();
            tracing::debug!("Rule engine cache cleared");
        }
    }

    /// Get cache statistics if caching is enabled
    ///
    /// Returns combined statistics for all cache types (rule results, derivations, etc.)
    pub fn get_cache_statistics(&self) -> Option<crate::cache::CachingStatistics> {
        self.cache.as_ref().map(|cache| cache.get_statistics())
    }

    /// Warm up the cache with current rules and common facts
    ///
    /// Pre-populates the cache with patterns from the current rule set
    /// and provided common facts to improve initial query performance.
    pub fn warm_cache(&self, common_facts: &[RuleAtom]) {
        if let Some(cache) = &self.cache {
            cache.warm_cache(&self.rules, common_facts);
            tracing::info!(
                "Cache warmed up with {} rules and {} facts",
                self.rules.len(),
                common_facts.len()
            );
        }
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

// RDFS entailment regime (v1.1.0 round 5)
pub mod rdfs_entailment;

// Semi-naive forward chaining rule engine (v1.1.0 round 6)
pub mod forward_chainer;

// N3/Notation3 rule syntax parser (v1.1.0 round 7)
pub mod rule_parser;

// Simplified Rete algorithm network for forward-chain rule matching (v1.1.0 round 8)
pub mod rete_network;

// Conflict detection and resolution for rule-based reasoning (v1.1.0 round 9)
pub mod conflict_resolver;

// Goal-directed backward-chaining reasoner (v1.1.0 round 10)
pub mod backward_chainer;

// Rule dependency graph for optimization (v1.1.0 round 11)
pub mod rule_graph;

// Rule compilation to bytecode-like IR (v1.1.0 round 12)
pub mod rule_compiler;

// Truth Maintenance System for belief revision (v1.1.0 round 13)
pub mod truth_maintenance;

// Rule execution tracing and debugging (v1.1.0 round 12)
pub mod rule_tracer;

// Rule serialization/deserialization — N3 and JSON formats (v1.1.0 round 13)
pub mod rule_serializer;

// Rule syntax and semantic validation (v1.1.0 round 14)
pub mod rule_validator;

// Forward-chaining rule execution engine (v1.1.0 round 15)
pub mod rule_executor;

// Rule execution statistics and profiling (v1.1.0 round 16)
pub mod rule_statistics;

#[cfg(test)]
mod comprehensive_tests;
#[cfg(test)]
mod comprehensive_tests_extended;
