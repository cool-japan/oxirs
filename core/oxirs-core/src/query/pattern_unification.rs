//! Pattern type unification for OxiRS query processing
//!
//! This module provides utilities to unify different pattern representations
//! used across the algebra and model systems, resolving type conflicts and
//! enabling seamless interoperability.

use crate::model::*;
use crate::query::algebra::{AlgebraTriplePattern, TermPattern as AlgebraTermPattern};
use crate::OxirsError;
use std::collections::HashSet;

/// Unified pattern representation that can handle both algebra and model patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnifiedTriplePattern {
    /// Subject pattern
    pub subject: UnifiedTermPattern,
    /// Predicate pattern  
    pub predicate: UnifiedTermPattern,
    /// Object pattern
    pub object: UnifiedTermPattern,
}

/// Unified term pattern that works with both systems
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UnifiedTermPattern {
    /// Named node (IRI)
    NamedNode(NamedNode),
    /// Blank node
    BlankNode(BlankNode),
    /// Literal value
    Literal(Literal),
    /// Variable
    Variable(Variable),
    /// Wildcard (matches anything)
    Wildcard,
}

impl UnifiedTriplePattern {
    /// Create a new unified triple pattern
    pub fn new(
        subject: UnifiedTermPattern,
        predicate: UnifiedTermPattern,
        object: UnifiedTermPattern,
    ) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Convert to algebra TriplePattern
    pub fn to_algebra_pattern(&self) -> Result<AlgebraTriplePattern, OxirsError> {
        let subject = self.subject.to_algebra_term_pattern()?;
        let predicate = self.predicate.to_algebra_term_pattern()?;
        let object = self.object.to_algebra_term_pattern()?;

        Ok(AlgebraTriplePattern::new(subject, predicate, object))
    }

    /// Convert to model TriplePattern
    pub fn to_model_pattern(&self) -> TriplePattern {
        let subject = self.subject.to_model_subject_pattern();
        let predicate = self.predicate.to_model_predicate_pattern();
        let object = self.object.to_model_object_pattern();

        TriplePattern::new(subject, predicate, object)
    }

    /// Create from algebra TriplePattern
    pub fn from_algebra_pattern(pattern: &AlgebraTriplePattern) -> Self {
        Self {
            subject: UnifiedTermPattern::from_algebra_term(&pattern.subject),
            predicate: UnifiedTermPattern::from_algebra_term(&pattern.predicate),
            object: UnifiedTermPattern::from_algebra_term(&pattern.object),
        }
    }

    /// Create from model TriplePattern
    pub fn from_model_pattern(pattern: &TriplePattern) -> Self {
        Self {
            subject: pattern
                .subject()
                .map(UnifiedTermPattern::from_model_subject)
                .unwrap_or(UnifiedTermPattern::Wildcard),
            predicate: pattern
                .predicate()
                .map(UnifiedTermPattern::from_model_predicate)
                .unwrap_or(UnifiedTermPattern::Wildcard),
            object: pattern
                .object()
                .map(UnifiedTermPattern::from_model_object)
                .unwrap_or(UnifiedTermPattern::Wildcard),
        }
    }

    /// Extract all variables from this pattern
    pub fn extract_variables(&self) -> HashSet<Variable> {
        let mut vars = HashSet::new();

        if let UnifiedTermPattern::Variable(v) = &self.subject {
            vars.insert(v.clone());
        }
        if let UnifiedTermPattern::Variable(v) = &self.predicate {
            vars.insert(v.clone());
        }
        if let UnifiedTermPattern::Variable(v) = &self.object {
            vars.insert(v.clone());
        }

        vars
    }

    /// Check if this pattern matches a concrete triple
    pub fn matches(&self, triple: &Triple) -> bool {
        self.subject.matches_subject(triple.subject())
            && self.predicate.matches_predicate(triple.predicate())
            && self.object.matches_object(triple.object())
    }

    /// Get pattern selectivity estimate (0.0 = most selective, 1.0 = least selective)
    pub fn selectivity_estimate(&self) -> f64 {
        let subject_selectivity = self.subject.selectivity_factor();
        let predicate_selectivity = self.predicate.selectivity_factor();
        let object_selectivity = self.object.selectivity_factor();

        // Combined selectivity using independence assumption
        subject_selectivity * predicate_selectivity * object_selectivity
    }
}

impl UnifiedTermPattern {
    /// Convert to algebra TermPattern
    pub fn to_algebra_term_pattern(&self) -> Result<AlgebraTermPattern, OxirsError> {
        match self {
            UnifiedTermPattern::NamedNode(nn) => Ok(AlgebraTermPattern::NamedNode(nn.clone())),
            UnifiedTermPattern::BlankNode(bn) => Ok(AlgebraTermPattern::BlankNode(bn.clone())),
            UnifiedTermPattern::Literal(lit) => Ok(AlgebraTermPattern::Literal(lit.clone())),
            UnifiedTermPattern::Variable(var) => Ok(AlgebraTermPattern::Variable(var.clone())),
            UnifiedTermPattern::Wildcard => Err(OxirsError::Query(
                "Wildcard patterns cannot be converted to algebra representation".to_string(),
            )),
        }
    }

    /// Convert to model SubjectPattern
    pub fn to_model_subject_pattern(&self) -> Option<SubjectPattern> {
        match self {
            UnifiedTermPattern::NamedNode(nn) => Some(SubjectPattern::NamedNode(nn.clone())),
            UnifiedTermPattern::BlankNode(bn) => Some(SubjectPattern::BlankNode(bn.clone())),
            UnifiedTermPattern::Variable(var) => Some(SubjectPattern::Variable(var.clone())),
            UnifiedTermPattern::Literal(_) | UnifiedTermPattern::Wildcard => None,
        }
    }

    /// Convert to model PredicatePattern
    pub fn to_model_predicate_pattern(&self) -> Option<PredicatePattern> {
        match self {
            UnifiedTermPattern::NamedNode(nn) => Some(PredicatePattern::NamedNode(nn.clone())),
            UnifiedTermPattern::Variable(var) => Some(PredicatePattern::Variable(var.clone())),
            UnifiedTermPattern::BlankNode(_)
            | UnifiedTermPattern::Literal(_)
            | UnifiedTermPattern::Wildcard => None,
        }
    }

    /// Convert to model ObjectPattern
    pub fn to_model_object_pattern(&self) -> Option<ObjectPattern> {
        match self {
            UnifiedTermPattern::NamedNode(nn) => Some(ObjectPattern::NamedNode(nn.clone())),
            UnifiedTermPattern::BlankNode(bn) => Some(ObjectPattern::BlankNode(bn.clone())),
            UnifiedTermPattern::Literal(lit) => Some(ObjectPattern::Literal(lit.clone())),
            UnifiedTermPattern::Variable(var) => Some(ObjectPattern::Variable(var.clone())),
            UnifiedTermPattern::Wildcard => None,
        }
    }

    /// Create from algebra TermPattern
    pub fn from_algebra_term(term: &AlgebraTermPattern) -> Self {
        match term {
            AlgebraTermPattern::NamedNode(nn) => UnifiedTermPattern::NamedNode(nn.clone()),
            AlgebraTermPattern::BlankNode(bn) => UnifiedTermPattern::BlankNode(bn.clone()),
            AlgebraTermPattern::Literal(lit) => UnifiedTermPattern::Literal(lit.clone()),
            AlgebraTermPattern::Variable(var) => UnifiedTermPattern::Variable(var.clone()),
        }
    }

    /// Create from model SubjectPattern
    pub fn from_model_subject(subject: &SubjectPattern) -> Self {
        match subject {
            SubjectPattern::NamedNode(nn) => UnifiedTermPattern::NamedNode(nn.clone()),
            SubjectPattern::BlankNode(bn) => UnifiedTermPattern::BlankNode(bn.clone()),
            SubjectPattern::Variable(var) => UnifiedTermPattern::Variable(var.clone()),
        }
    }

    /// Create from model PredicatePattern
    pub fn from_model_predicate(predicate: &PredicatePattern) -> Self {
        match predicate {
            PredicatePattern::NamedNode(nn) => UnifiedTermPattern::NamedNode(nn.clone()),
            PredicatePattern::Variable(var) => UnifiedTermPattern::Variable(var.clone()),
        }
    }

    /// Create from model ObjectPattern
    pub fn from_model_object(object: &ObjectPattern) -> Self {
        match object {
            ObjectPattern::NamedNode(nn) => UnifiedTermPattern::NamedNode(nn.clone()),
            ObjectPattern::BlankNode(bn) => UnifiedTermPattern::BlankNode(bn.clone()),
            ObjectPattern::Literal(lit) => UnifiedTermPattern::Literal(lit.clone()),
            ObjectPattern::Variable(var) => UnifiedTermPattern::Variable(var.clone()),
        }
    }

    /// Check if this pattern matches a subject
    pub fn matches_subject(&self, subject: &Subject) -> bool {
        match (self, subject) {
            (UnifiedTermPattern::NamedNode(pn), Subject::NamedNode(sn)) => pn == sn,
            (UnifiedTermPattern::BlankNode(pb), Subject::BlankNode(sb)) => pb == sb,
            (UnifiedTermPattern::Variable(_), _) | (UnifiedTermPattern::Wildcard, _) => true,
            _ => false,
        }
    }

    /// Check if this pattern matches a predicate
    pub fn matches_predicate(&self, predicate: &Predicate) -> bool {
        match (self, predicate) {
            (UnifiedTermPattern::NamedNode(pn), Predicate::NamedNode(sn)) => pn == sn,
            (UnifiedTermPattern::Variable(_), _) | (UnifiedTermPattern::Wildcard, _) => true,
            _ => false,
        }
    }

    /// Check if this pattern matches an object
    pub fn matches_object(&self, object: &Object) -> bool {
        match (self, object) {
            (UnifiedTermPattern::NamedNode(pn), Object::NamedNode(on)) => pn == on,
            (UnifiedTermPattern::BlankNode(pb), Object::BlankNode(ob)) => pb == ob,
            (UnifiedTermPattern::Literal(pl), Object::Literal(ol)) => pl == ol,
            (UnifiedTermPattern::Variable(_), _) | (UnifiedTermPattern::Wildcard, _) => true,
            _ => false,
        }
    }

    /// Get selectivity factor for cost estimation
    pub fn selectivity_factor(&self) -> f64 {
        match self {
            UnifiedTermPattern::NamedNode(_) => 0.001, // Very selective
            UnifiedTermPattern::BlankNode(_) => 0.01,  // Selective
            UnifiedTermPattern::Literal(_) => 0.001,   // Very selective
            UnifiedTermPattern::Variable(_) => 1.0,    // Not selective
            UnifiedTermPattern::Wildcard => 1.0,       // Not selective
        }
    }
}

/// Pattern conversion utilities
pub struct PatternConverter;

impl PatternConverter {
    /// Convert a vector of algebra patterns to model patterns
    pub fn algebra_to_model_patterns(patterns: &[AlgebraTriplePattern]) -> Vec<TriplePattern> {
        patterns
            .iter()
            .map(|p| UnifiedTriplePattern::from_algebra_pattern(p).to_model_pattern())
            .collect()
    }

    /// Convert a vector of model patterns to algebra patterns
    pub fn model_to_algebra_patterns(
        patterns: &[TriplePattern],
    ) -> Result<Vec<AlgebraTriplePattern>, OxirsError> {
        patterns
            .iter()
            .map(|p| UnifiedTriplePattern::from_model_pattern(p).to_algebra_pattern())
            .collect()
    }

    /// Extract all variables from a set of algebra patterns
    pub fn extract_variables_from_algebra(patterns: &[AlgebraTriplePattern]) -> HashSet<Variable> {
        patterns
            .iter()
            .flat_map(|p| UnifiedTriplePattern::from_algebra_pattern(p).extract_variables())
            .collect()
    }

    /// Extract all variables from a set of model patterns
    pub fn extract_variables_from_model(patterns: &[TriplePattern]) -> HashSet<Variable> {
        patterns
            .iter()
            .flat_map(|p| UnifiedTriplePattern::from_model_pattern(p).extract_variables())
            .collect()
    }

    /// Estimate combined selectivity for a set of patterns
    pub fn estimate_pattern_selectivity(patterns: &[UnifiedTriplePattern]) -> f64 {
        if patterns.is_empty() {
            return 1.0;
        }

        patterns
            .iter()
            .map(|p| p.selectivity_estimate())
            .fold(1.0, |acc, s| acc * s)
    }
}

/// Query optimization utilities using unified patterns
pub struct PatternOptimizer;

impl PatternOptimizer {
    /// Reorder patterns for optimal execution based on selectivity
    pub fn optimize_pattern_order(patterns: &[UnifiedTriplePattern]) -> Vec<UnifiedTriplePattern> {
        let mut sorted_patterns = patterns.to_vec();

        // Sort by selectivity (most selective first)
        sorted_patterns.sort_by(|a, b| {
            a.selectivity_estimate()
                .partial_cmp(&b.selectivity_estimate())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        sorted_patterns
    }

    /// Find optimal join order for patterns
    pub fn optimize_join_order(patterns: &[UnifiedTriplePattern]) -> Vec<usize> {
        if patterns.is_empty() {
            return Vec::new();
        }

        // Simple greedy algorithm: start with most selective pattern
        let mut remaining: Vec<usize> = (0..patterns.len()).collect();
        let mut order = Vec::new();

        // Find most selective pattern as starting point
        if let Some(min_idx) = remaining
            .iter()
            .min_by(|&&a, &&b| {
                patterns[a]
                    .selectivity_estimate()
                    .partial_cmp(&patterns[b].selectivity_estimate())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
        {
            order.push(min_idx);
            remaining.retain(|&x| x != min_idx);
        }

        // Greedily add patterns that share most variables with already selected patterns
        while !remaining.is_empty() {
            let selected_vars: HashSet<Variable> = order
                .iter()
                .flat_map(|&i| patterns[i].extract_variables())
                .collect();

            if let Some(best_idx) = remaining
                .iter()
                .max_by_key(|&&i| {
                    let pattern_vars = patterns[i].extract_variables();
                    pattern_vars.intersection(&selected_vars).count()
                })
                .copied()
            {
                order.push(best_idx);
                remaining.retain(|&x| x != best_idx);
            } else {
                // Fallback: add remaining patterns in selectivity order
                order.extend(remaining);
                break;
            }
        }

        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_pattern_conversion() {
        // Create an algebra pattern
        let algebra_pattern = AlgebraTriplePattern::new(
            AlgebraTermPattern::Variable(Variable::new("s").unwrap()),
            AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/pred").unwrap()),
            AlgebraTermPattern::Literal(Literal::new("test")),
        );

        // Convert to unified pattern
        let unified = UnifiedTriplePattern::from_algebra_pattern(&algebra_pattern);

        // Convert back to algebra pattern
        let converted_back = unified.to_algebra_pattern().unwrap();

        assert_eq!(algebra_pattern, converted_back);
    }

    #[test]
    fn test_pattern_selectivity() {
        let patterns = vec![
            UnifiedTriplePattern::new(
                UnifiedTermPattern::Variable(Variable::new("s").unwrap()),
                UnifiedTermPattern::Variable(Variable::new("p").unwrap()),
                UnifiedTermPattern::Variable(Variable::new("o").unwrap()),
            ),
            UnifiedTriplePattern::new(
                UnifiedTermPattern::NamedNode(NamedNode::new("http://example.org/s").unwrap()),
                UnifiedTermPattern::NamedNode(NamedNode::new("http://example.org/p").unwrap()),
                UnifiedTermPattern::Variable(Variable::new("o").unwrap()),
            ),
        ];

        // Second pattern should be more selective
        assert!(patterns[1].selectivity_estimate() < patterns[0].selectivity_estimate());
    }

    #[test]
    fn test_pattern_optimization() {
        let patterns = vec![
            UnifiedTriplePattern::new(
                UnifiedTermPattern::Variable(Variable::new("s").unwrap()),
                UnifiedTermPattern::Variable(Variable::new("p").unwrap()),
                UnifiedTermPattern::Variable(Variable::new("o").unwrap()),
            ),
            UnifiedTriplePattern::new(
                UnifiedTermPattern::NamedNode(NamedNode::new("http://example.org/s").unwrap()),
                UnifiedTermPattern::NamedNode(NamedNode::new("http://example.org/p").unwrap()),
                UnifiedTermPattern::Variable(Variable::new("o").unwrap()),
            ),
        ];

        let optimized = PatternOptimizer::optimize_pattern_order(&patterns);

        // More selective pattern should come first
        assert_eq!(optimized[0], patterns[1]);
        assert_eq!(optimized[1], patterns[0]);
    }
}
