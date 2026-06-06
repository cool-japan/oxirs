//! Probabilistic Datalog (ProbLog) — types, structs, and enums
//!
//! This module contains all data type definitions for the ProbLog engine:
//! structs, enums, probabilistic logic types, clause types, and evidence types.

use crate::{Rule, RuleAtom};
use anyhow::{anyhow, Result};

/// Probabilistic fact with associated probability
#[derive(Debug, Clone)]
pub struct ProbabilisticFact {
    /// Probability in [0, 1]
    pub probability: f64,
    /// The fact itself
    pub fact: RuleAtom,
}

impl ProbabilisticFact {
    pub fn new(probability: f64, fact: RuleAtom) -> Result<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(anyhow!(
                "Probability must be in [0, 1], got {}",
                probability
            ));
        }
        Ok(Self { probability, fact })
    }
}

/// Probabilistic rule with optional probability
#[derive(Debug, Clone)]
pub struct ProbabilisticRule {
    /// Optional probability (if None, probability is 1.0)
    pub probability: Option<f64>,
    /// The rule itself
    pub rule: Rule,
}

impl ProbabilisticRule {
    pub fn deterministic(rule: Rule) -> Self {
        Self {
            probability: None,
            rule,
        }
    }

    pub fn probabilistic(probability: f64, rule: Rule) -> Result<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(anyhow!(
                "Probability must be in [0, 1], got {}",
                probability
            ));
        }
        Ok(Self {
            probability: Some(probability),
            rule,
        })
    }
}

/// Derivation tree tracking provenance
#[derive(Debug, Clone)]
pub struct DerivationTree {
    /// The derived fact
    pub fact: RuleAtom,
    /// Probability of this derivation
    pub probability: f64,
    /// Facts this was derived from
    pub premises: Vec<DerivationTree>,
}

impl DerivationTree {
    pub fn leaf(fact: RuleAtom, probability: f64) -> Self {
        Self {
            fact,
            probability,
            premises: Vec::new(),
        }
    }

    pub fn node(fact: RuleAtom, probability: f64, premises: Vec<DerivationTree>) -> Self {
        Self {
            fact,
            probability,
            premises,
        }
    }
}

/// Evaluation strategy for recursive queries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationStrategy {
    /// Top-down backward chaining (with cycle detection)
    TopDown,
    /// Bottom-up forward chaining with fixpoint iteration
    BottomUp,
    /// Automatic selection based on query characteristics
    Auto,
}

/// Statistics for ProbLog engine
#[derive(Debug, Clone, Default)]
pub struct ProbLogStats {
    pub queries: usize,
    pub inferences: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub fixpoint_iterations: usize,
    pub materialized_facts_count: usize,
}
