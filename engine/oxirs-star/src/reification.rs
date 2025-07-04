//! Reification utilities for converting between RDF-star and standard RDF.
//!
//! This module provides functionality to convert quoted triples to/from
//! standard RDF reification using rdf:Statement, rdf:subject, rdf:predicate, rdf:object.

use std::collections::HashMap;

use lru::LruCache;
use oxirs_core::model::{NamedNode as CoreNamedNode, Triple as CoreTriple};
use tracing::{debug, span, Level};

use crate::model::{NamedNode, StarGraph, StarTerm, StarTriple};
use crate::{StarError, StarResult};

/// Standard RDF vocabulary for reification
pub mod vocab {
    pub const RDF_STATEMENT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement";
    pub const RDF_SUBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject";
    pub const RDF_PREDICATE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate";
    pub const RDF_OBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object";
    pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
}

/// Reification strategy for handling quoted triples
#[derive(Debug, Clone, PartialEq)]
pub enum ReificationStrategy {
    /// Standard RDF reification using rdf:Statement
    StandardReification,
    /// Use unique IRIs for each quoted triple
    UniqueIris,
    /// Use blank nodes for quoted triples
    BlankNodes,
}

/// Reification context for managing identifiers and mappings
#[derive(Debug)]
pub struct ReificationContext {
    /// Strategy to use for reification
    strategy: ReificationStrategy,
    /// Counter for generating unique identifiers
    counter: usize,
    /// Base IRI for generating statement IRIs
    base_iri: String,
    /// Mapping from quoted triples to their reification identifiers
    triple_to_id: HashMap<String, String>,
    /// Mapping from reification identifiers to quoted triples
    id_to_triple: HashMap<String, StarTriple>,
}

impl ReificationContext {
    /// Create a new reification context
    pub fn new(strategy: ReificationStrategy, base_iri: Option<String>) -> Self {
        Self {
            strategy,
            counter: 0,
            base_iri: base_iri.unwrap_or_else(|| "http://example.org/statement/".to_string()),
            triple_to_id: HashMap::new(),
            id_to_triple: HashMap::new(),
        }
    }

    /// Generate a unique identifier for a quoted triple
    fn generate_id(&mut self, triple: &StarTriple) -> String {
        let triple_key = format!("{}|{}|{}", triple.subject, triple.predicate, triple.object);

        if let Some(existing_id) = self.triple_to_id.get(&triple_key) {
            return existing_id.clone();
        }

        let id = match self.strategy {
            ReificationStrategy::StandardReification | ReificationStrategy::UniqueIris => {
                self.counter += 1;
                format!("{}{}", self.base_iri, self.counter)
            }
            ReificationStrategy::BlankNodes => {
                self.counter += 1;
                format!("_:stmt{}", self.counter)
            }
        };

        self.triple_to_id.insert(triple_key, id.clone());
        self.id_to_triple.insert(id.clone(), triple.clone());
        id
    }

    /// Get the identifier for a quoted triple if it exists
    pub fn get_id(&self, triple: &StarTriple) -> Option<&String> {
        let triple_key = format!("{}|{}|{}", triple.subject, triple.predicate, triple.object);
        self.triple_to_id.get(&triple_key)
    }

    /// Get the quoted triple for an identifier if it exists
    pub fn get_triple(&self, id: &str) -> Option<&StarTriple> {
        self.id_to_triple.get(id)
    }
}

/// RDF-star to standard RDF reification converter
pub struct Reificator {
    context: ReificationContext,
}

impl Reificator {
    /// Create a new reificator with the specified strategy
    pub fn new(strategy: ReificationStrategy, base_iri: Option<String>) -> Self {
        Self {
            context: ReificationContext::new(strategy, base_iri),
        }
    }

    /// Convert an RDF-star graph to standard RDF using reification
    pub fn reify_graph(&mut self, star_graph: &StarGraph) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "reify_graph");
        let _enter = span.enter();

        let mut reified_graph = StarGraph::new();

        for triple in star_graph.triples() {
            let reified_triples = self.reify_triple(triple)?;
            for reified_triple in reified_triples {
                reified_graph.insert(reified_triple)?;
            }
        }

        debug!(
            "Reified {} triples to {} standard RDF triples",
            star_graph.len(),
            reified_graph.len()
        );
        Ok(reified_graph)
    }

    /// Convert a single RDF-star triple to standard RDF triples
    pub fn reify_triple(&mut self, triple: &StarTriple) -> StarResult<Vec<StarTriple>> {
        let mut result = Vec::new();

        // Process subject
        let subject = self.reify_term(&triple.subject, &mut result)?;

        // Process predicate (should not contain quoted triples in valid RDF-star)
        let predicate = self.reify_term(&triple.predicate, &mut result)?;

        // Process object
        let object = self.reify_term(&triple.object, &mut result)?;

        // Create the main triple with reified terms
        let main_triple = StarTriple::new(subject, predicate, object);
        result.push(main_triple);

        Ok(result)
    }

    /// Reify a single term, generating additional triples if needed
    fn reify_term(
        &mut self,
        term: &StarTerm,
        additional_triples: &mut Vec<StarTriple>,
    ) -> StarResult<StarTerm> {
        match term {
            StarTerm::QuotedTriple(quoted_triple) => {
                // Generate identifier for the quoted triple
                let stmt_id = self.context.generate_id(quoted_triple);

                // Create reification triples
                let reification_triples =
                    self.create_reification_triples(&stmt_id, quoted_triple)?;
                additional_triples.extend(reification_triples);

                // Return the statement identifier as the term
                match self.context.strategy {
                    ReificationStrategy::StandardReification | ReificationStrategy::UniqueIris => {
                        Ok(StarTerm::iri(&stmt_id)?)
                    }
                    ReificationStrategy::BlankNodes => {
                        let blank_id = &stmt_id[2..]; // Remove "_:" prefix
                        Ok(StarTerm::blank_node(blank_id)?)
                    }
                }
            }
            _ => Ok(term.clone()),
        }
    }

    /// Create the standard reification triples for a quoted triple
    fn create_reification_triples(
        &mut self,
        stmt_id: &str,
        triple: &StarTriple,
    ) -> StarResult<Vec<StarTriple>> {
        let mut triples = Vec::new();

        // Create statement identifier term
        let stmt_term = match self.context.strategy {
            ReificationStrategy::StandardReification | ReificationStrategy::UniqueIris => {
                StarTerm::iri(stmt_id)?
            }
            ReificationStrategy::BlankNodes => {
                let blank_id = &stmt_id[2..]; // Remove "_:" prefix
                StarTerm::blank_node(blank_id)?
            }
        };

        // stmt_id rdf:type rdf:Statement
        if matches!(
            self.context.strategy,
            ReificationStrategy::StandardReification
        ) {
            triples.push(StarTriple::new(
                stmt_term.clone(),
                StarTerm::iri(vocab::RDF_TYPE)?,
                StarTerm::iri(vocab::RDF_STATEMENT)?,
            ));
        }

        // Recursively reify subject, predicate, object
        let mut subject_additional = Vec::new();
        let reified_subject = self.reify_term(&triple.subject, &mut subject_additional)?;
        triples.extend(subject_additional);

        let mut predicate_additional = Vec::new();
        let reified_predicate = self.reify_term(&triple.predicate, &mut predicate_additional)?;
        triples.extend(predicate_additional);

        let mut object_additional = Vec::new();
        let reified_object = self.reify_term(&triple.object, &mut object_additional)?;
        triples.extend(object_additional);

        // stmt_id rdf:subject subject
        triples.push(StarTriple::new(
            stmt_term.clone(),
            StarTerm::iri(vocab::RDF_SUBJECT)?,
            reified_subject,
        ));

        // stmt_id rdf:predicate predicate
        triples.push(StarTriple::new(
            stmt_term.clone(),
            StarTerm::iri(vocab::RDF_PREDICATE)?,
            reified_predicate,
        ));

        // stmt_id rdf:object object
        triples.push(StarTriple::new(
            stmt_term,
            StarTerm::iri(vocab::RDF_OBJECT)?,
            reified_object,
        ));

        Ok(triples)
    }

    /// Convert standard RDF reification back to RDF-star (dereification)
    pub fn dereify_graph(&mut self, reified_graph: &StarGraph) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "dereify_graph");
        let _enter = span.enter();

        let mut star_graph = StarGraph::new();
        let mut processed_statements = std::collections::HashSet::new();

        // Find all rdf:Statement instances
        for triple in reified_graph.triples() {
            if let (StarTerm::NamedNode(predicate), StarTerm::NamedNode(object)) =
                (&triple.predicate, &triple.object)
            {
                if predicate.iri == vocab::RDF_TYPE && object.iri == vocab::RDF_STATEMENT {
                    if let StarTerm::NamedNode(stmt_node) = &triple.subject {
                        if !processed_statements.contains(&stmt_node.iri) {
                            if let Some(star_triple) =
                                self.reconstruct_quoted_triple(reified_graph, &stmt_node.iri)?
                            {
                                star_graph.insert(star_triple)?;
                                processed_statements.insert(stmt_node.iri.clone());
                            }
                        }
                    }
                }
            }
        }

        // Process remaining triples that don't involve reified statements
        for triple in reified_graph.triples() {
            if !self.involves_reified_statement(triple, &processed_statements) {
                star_graph.insert(triple.clone())?;
            }
        }

        debug!(
            "Dereified {} reified triples back to {} RDF-star triples",
            reified_graph.len(),
            star_graph.len()
        );
        Ok(star_graph)
    }

    /// Reconstruct a quoted triple from its reification
    fn reconstruct_quoted_triple(
        &self,
        graph: &StarGraph,
        stmt_iri: &str,
    ) -> StarResult<Option<StarTriple>> {
        let mut subject = None;
        let mut predicate = None;
        let mut object = None;

        let stmt_term = StarTerm::iri(stmt_iri)?;

        // Find the reification triples
        for triple in graph.triples() {
            if triple.subject == stmt_term {
                if let StarTerm::NamedNode(pred_node) = &triple.predicate {
                    match pred_node.iri.as_str() {
                        vocab::RDF_SUBJECT => subject = Some(triple.object.clone()),
                        vocab::RDF_PREDICATE => predicate = Some(triple.object.clone()),
                        vocab::RDF_OBJECT => object = Some(triple.object.clone()),
                        _ => {}
                    }
                }
            }
        }

        if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
            Ok(Some(StarTriple::new(s, p, o)))
        } else {
            Ok(None)
        }
    }

    /// Check if a triple involves a reified statement
    fn involves_reified_statement(
        &self,
        triple: &StarTriple,
        processed_statements: &std::collections::HashSet<String>,
    ) -> bool {
        if let StarTerm::NamedNode(subj_node) = &triple.subject {
            if processed_statements.contains(&subj_node.iri) {
                return true;
            }
        }
        if let StarTerm::NamedNode(obj_node) = &triple.object {
            if processed_statements.contains(&obj_node.iri) {
                return true;
            }
        }
        false
    }
}

/// Advanced reification strategies with hybrid approaches
#[derive(Debug, Clone, PartialEq)]
pub enum AdvancedReificationStrategy {
    /// Standard reification strategies
    Standard(ReificationStrategy),
    /// Hybrid approach: use different strategies based on context
    Hybrid {
        /// Strategy for simple quoted triples
        simple_strategy: ReificationStrategy,
        /// Strategy for nested quoted triples
        nested_strategy: ReificationStrategy,
        /// Strategy for quoted triples in specific predicates
        predicate_strategies: HashMap<String, ReificationStrategy>,
    },
    /// Conditional reification based on triple characteristics
    Conditional {
        /// Default strategy
        default_strategy: ReificationStrategy,
        /// Rules for selecting strategy
        rules: Vec<ReificationRule>,
    },
    /// Memory-optimized reification with caching
    Optimized {
        /// Base strategy
        base_strategy: ReificationStrategy,
        /// Enable aggressive caching
        aggressive_caching: bool,
        /// Maximum cache size
        max_cache_size: usize,
    },
}

/// Rule for conditional reification strategy selection
#[derive(Debug, Clone, PartialEq)]
pub struct ReificationRule {
    /// Condition to match
    pub condition: ReificationCondition,
    /// Strategy to use when condition matches
    pub strategy: ReificationStrategy,
    /// Priority (higher number = higher priority)
    pub priority: u32,
}

/// Conditions for reification strategy selection
#[derive(Debug, Clone, PartialEq)]
pub enum ReificationCondition {
    /// Match by predicate IRI
    PredicateIri(String),
    /// Match by subject type
    SubjectType(TermType),
    /// Match by object type
    ObjectType(TermType),
    /// Match by nesting depth
    NestingDepth(usize),
    /// Match by graph size
    GraphSize(usize),
    /// Custom condition function name
    Custom(String),
}

/// RDF term types for condition matching
#[derive(Debug, Clone, PartialEq)]
pub enum TermType {
    NamedNode,
    BlankNode,
    Literal,
    QuotedTriple,
    Variable,
}

/// Enhanced reificator with advanced strategies
pub struct AdvancedReificator {
    strategy: AdvancedReificationStrategy,
    contexts: HashMap<String, ReificationContext>,
    cache: lru::LruCache<String, Vec<StarTriple>>,
    statistics: ReificationStatistics,
}

/// Statistics for reification operations
#[derive(Debug, Clone, Default)]
pub struct ReificationStatistics {
    /// Total triples processed
    pub total_triples: usize,
    /// Total quoted triples found
    pub quoted_triples: usize,
    /// Total reification triples generated
    pub reification_triples: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average processing time per triple (microseconds)
    pub avg_processing_time: f64,
    /// Strategy usage counts
    pub strategy_usage: HashMap<String, usize>,
}

impl AdvancedReificator {
    /// Create a new advanced reificator
    pub fn new(strategy: AdvancedReificationStrategy) -> Self {
        Self {
            strategy,
            contexts: HashMap::new(),
            cache: lru::LruCache::new(std::num::NonZeroUsize::new(1000).unwrap()),
            statistics: ReificationStatistics::default(),
        }
    }

    /// Process a graph with advanced reification strategies
    pub fn reify_graph_advanced(&mut self, star_graph: &StarGraph) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "reify_graph_advanced");
        let _enter = span.enter();

        let start_time = std::time::Instant::now();
        let mut reified_graph = StarGraph::new();

        for triple in star_graph.triples() {
            self.statistics.total_triples += 1;

            let strategy = self.select_strategy_for_triple(triple)?;
            let strategy_name = format!("{:?}", strategy);
            *self
                .statistics
                .strategy_usage
                .entry(strategy_name)
                .or_insert(0) += 1;

            let reified_triples = self.reify_triple_with_strategy(triple, &strategy)?;
            for reified_triple in reified_triples {
                reified_graph.insert(reified_triple)?;
                self.statistics.reification_triples += 1;
            }
        }

        let processing_time = start_time.elapsed();
        self.statistics.avg_processing_time =
            processing_time.as_micros() as f64 / self.statistics.total_triples as f64;

        debug!(
            "Advanced reification completed: {} triples -> {} triples in {:?}",
            star_graph.len(),
            reified_graph.len(),
            processing_time
        );

        Ok(reified_graph)
    }

    /// Select the appropriate reification strategy for a triple
    fn select_strategy_for_triple(&self, triple: &StarTriple) -> StarResult<ReificationStrategy> {
        match &self.strategy {
            AdvancedReificationStrategy::Standard(strategy) => Ok(strategy.clone()),
            AdvancedReificationStrategy::Hybrid {
                simple_strategy,
                nested_strategy,
                predicate_strategies,
            } => {
                // Check for predicate-specific strategies
                if let StarTerm::NamedNode(pred_node) = &triple.predicate {
                    if let Some(strategy) = predicate_strategies.get(&pred_node.iri) {
                        return Ok(strategy.clone());
                    }
                }

                // Check for nesting
                if self.has_nested_quoted_triples(triple) {
                    Ok(nested_strategy.clone())
                } else {
                    Ok(simple_strategy.clone())
                }
            }
            AdvancedReificationStrategy::Conditional {
                default_strategy,
                rules,
            } => {
                // Evaluate rules by priority
                let mut applicable_rules: Vec<_> = rules
                    .iter()
                    .filter(|rule| self.evaluate_condition(&rule.condition, triple))
                    .collect();
                applicable_rules.sort_by_key(|rule| std::cmp::Reverse(rule.priority));

                if let Some(rule) = applicable_rules.first() {
                    Ok(rule.strategy.clone())
                } else {
                    Ok(default_strategy.clone())
                }
            }
            AdvancedReificationStrategy::Optimized { base_strategy, .. } => {
                Ok(base_strategy.clone())
            }
        }
    }

    /// Check if a triple contains nested quoted triples
    fn has_nested_quoted_triples(&self, triple: &StarTriple) -> bool {
        self.term_has_quoted_triples(&triple.subject)
            || self.term_has_quoted_triples(&triple.predicate)
            || self.term_has_quoted_triples(&triple.object)
    }

    /// Check if a term contains quoted triples
    fn term_has_quoted_triples(&self, term: &StarTerm) -> bool {
        match term {
            StarTerm::QuotedTriple(_inner_triple) => {
                // The term itself is a quoted triple
                true
            }
            _ => false,
        }
    }

    /// Evaluate a reification condition against a triple
    fn evaluate_condition(&self, condition: &ReificationCondition, triple: &StarTriple) -> bool {
        match condition {
            ReificationCondition::PredicateIri(iri) => {
                if let StarTerm::NamedNode(pred_node) = &triple.predicate {
                    pred_node.iri == *iri
                } else {
                    false
                }
            }
            ReificationCondition::SubjectType(term_type) => {
                self.matches_term_type(&triple.subject, term_type)
            }
            ReificationCondition::ObjectType(term_type) => {
                self.matches_term_type(&triple.object, term_type)
            }
            ReificationCondition::NestingDepth(max_depth) => {
                self.calculate_nesting_depth(triple) <= *max_depth
            }
            ReificationCondition::GraphSize(_) => {
                // Would need graph context to evaluate properly
                true
            }
            ReificationCondition::Custom(_) => {
                // Would need custom evaluation logic
                false
            }
        }
    }

    /// Check if a term matches a term type
    fn matches_term_type(&self, term: &StarTerm, term_type: &TermType) -> bool {
        match (term, term_type) {
            (StarTerm::NamedNode(_), TermType::NamedNode) => true,
            (StarTerm::BlankNode(_), TermType::BlankNode) => true,
            (StarTerm::Literal(_), TermType::Literal) => true,
            (StarTerm::QuotedTriple(_), TermType::QuotedTriple) => true,
            (StarTerm::Variable(_), TermType::Variable) => true,
            _ => false,
        }
    }

    /// Calculate the nesting depth of quoted triples in a triple
    fn calculate_nesting_depth(&self, triple: &StarTriple) -> usize {
        let subject_depth = self.term_nesting_depth(&triple.subject);
        let predicate_depth = self.term_nesting_depth(&triple.predicate);
        let object_depth = self.term_nesting_depth(&triple.object);

        subject_depth.max(predicate_depth).max(object_depth)
    }

    /// Calculate the nesting depth of a term
    fn term_nesting_depth(&self, term: &StarTerm) -> usize {
        match term {
            StarTerm::QuotedTriple(inner_triple) => 1 + self.calculate_nesting_depth(inner_triple),
            _ => 0,
        }
    }

    /// Reify a triple using a specific strategy
    fn reify_triple_with_strategy(
        &mut self,
        triple: &StarTriple,
        strategy: &ReificationStrategy,
    ) -> StarResult<Vec<StarTriple>> {
        // Get or create context for this strategy
        let context_key = format!("{:?}", strategy);
        if !self.contexts.contains_key(&context_key) {
            self.contexts.insert(
                context_key.clone(),
                ReificationContext::new(strategy.clone(), None),
            );
        }

        // Create a temporary reificator for this strategy
        let context = self.contexts.get_mut(&context_key).unwrap();
        let mut temp_reificator = Reificator {
            context: ReificationContext::new(strategy.clone(), None),
        };

        // Copy the state from our context
        temp_reificator.context.counter = context.counter;
        temp_reificator.context.triple_to_id = context.triple_to_id.clone();
        temp_reificator.context.id_to_triple = context.id_to_triple.clone();

        let result = temp_reificator.reify_triple(triple);

        // Copy back the state
        context.counter = temp_reificator.context.counter;
        context.triple_to_id = temp_reificator.context.triple_to_id;
        context.id_to_triple = temp_reificator.context.id_to_triple;

        result
    }

    /// Get reification statistics
    pub fn get_statistics(&self) -> &ReificationStatistics {
        &self.statistics
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = ReificationStatistics::default();
    }

    /// Export reification mapping for external use
    pub fn export_mappings(&self) -> HashMap<String, HashMap<String, String>> {
        let mut mappings = HashMap::new();

        for (strategy_key, context) in &self.contexts {
            mappings.insert(strategy_key.clone(), context.triple_to_id.clone());
        }

        mappings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_reification() {
        let mut reificator = Reificator::new(ReificationStrategy::StandardReification, None);

        // Create a simple RDF-star triple
        let quoted_triple = StarTriple::new(
            StarTerm::iri("http://example.org/subject").unwrap(),
            StarTerm::iri("http://example.org/predicate").unwrap(),
            StarTerm::iri("http://example.org/object").unwrap(),
        );

        let triple_with_quoted = StarTriple::new(
            StarTerm::QuotedTriple(Box::new(quoted_triple)),
            StarTerm::iri("http://example.org/hasMetadata").unwrap(),
            StarTerm::literal("metadata").unwrap(),
        );

        let reified_triples = reificator.reify_triple(&triple_with_quoted).unwrap();

        // Should generate multiple triples for reification
        assert!(reified_triples.len() > 1);
    }

    #[test]
    fn test_dereification() {
        let mut reificator = Reificator::new(ReificationStrategy::StandardReification, None);

        // Create a test graph with RDF-star
        let mut star_graph = StarGraph::new();
        let quoted_triple = StarTriple::new(
            StarTerm::iri("http://example.org/subject").unwrap(),
            StarTerm::iri("http://example.org/predicate").unwrap(),
            StarTerm::iri("http://example.org/object").unwrap(),
        );

        let triple_with_quoted = StarTriple::new(
            StarTerm::QuotedTriple(Box::new(quoted_triple)),
            StarTerm::iri("http://example.org/hasMetadata").unwrap(),
            StarTerm::literal("metadata").unwrap(),
        );

        star_graph.insert(triple_with_quoted).unwrap();

        // Reify and then dereify
        let reified_graph = reificator.reify_graph(&star_graph).unwrap();
        let dereified_graph = reificator.dereify_graph(&reified_graph).unwrap();

        // Should round-trip successfully
        assert_eq!(star_graph.len(), dereified_graph.len());
    }

    #[test]
    fn test_advanced_reification_strategies() {
        let hybrid_strategy = AdvancedReificationStrategy::Hybrid {
            simple_strategy: ReificationStrategy::StandardReification,
            nested_strategy: ReificationStrategy::BlankNodes,
            predicate_strategies: HashMap::new(),
        };

        let mut advanced_reificator = AdvancedReificator::new(hybrid_strategy);

        // Test with a simple graph
        let mut star_graph = StarGraph::new();
        let quoted_triple = StarTriple::new(
            StarTerm::iri("http://example.org/subject").unwrap(),
            StarTerm::iri("http://example.org/predicate").unwrap(),
            StarTerm::iri("http://example.org/object").unwrap(),
        );

        let triple_with_quoted = StarTriple::new(
            StarTerm::QuotedTriple(Box::new(quoted_triple)),
            StarTerm::iri("http://example.org/hasMetadata").unwrap(),
            StarTerm::literal("metadata").unwrap(),
        );

        star_graph.insert(triple_with_quoted).unwrap();

        let reified_graph = advanced_reificator
            .reify_graph_advanced(&star_graph)
            .unwrap();
        assert!(reified_graph.len() > 0);

        let stats = advanced_reificator.get_statistics();
        assert!(stats.total_triples > 0);
    }

    #[test]
    fn test_conditional_reification() {
        let rules = vec![ReificationRule {
            condition: ReificationCondition::PredicateIri("http://example.org/special".to_string()),
            strategy: ReificationStrategy::BlankNodes,
            priority: 10,
        }];

        let conditional_strategy = AdvancedReificationStrategy::Conditional {
            default_strategy: ReificationStrategy::StandardReification,
            rules,
        };

        let mut advanced_reificator = AdvancedReificator::new(conditional_strategy);

        // Create test data that matches the condition
        let mut star_graph = StarGraph::new();
        let quoted_triple = StarTriple::new(
            StarTerm::iri("http://example.org/subject").unwrap(),
            StarTerm::iri("http://example.org/special").unwrap(), // Matches condition
            StarTerm::iri("http://example.org/object").unwrap(),
        );

        let triple_with_quoted = StarTriple::new(
            StarTerm::QuotedTriple(Box::new(quoted_triple)),
            StarTerm::iri("http://example.org/hasMetadata").unwrap(),
            StarTerm::literal("metadata").unwrap(),
        );

        star_graph.insert(triple_with_quoted).unwrap();

        let reified_graph = advanced_reificator
            .reify_graph_advanced(&star_graph)
            .unwrap();
        assert!(reified_graph.len() > 0);

        let stats = advanced_reificator.get_statistics();
        assert!(stats.strategy_usage.len() > 0);
    }
}

/// Utility functions for reification
pub mod utils {
    use super::*;

    /// Check if a graph contains reification patterns
    pub fn has_reifications(graph: &StarGraph) -> bool {
        for triple in graph.triples() {
            if let StarTerm::NamedNode(node) = &triple.predicate {
                if matches!(
                    node.iri.as_str(),
                    vocab::RDF_SUBJECT | vocab::RDF_PREDICATE | vocab::RDF_OBJECT
                ) {
                    return true;
                }
            }
        }
        false
    }

    /// Count the number of reification statements in a graph
    pub fn count_reifications(graph: &StarGraph) -> usize {
        let mut statements = std::collections::HashSet::new();

        for triple in graph.triples() {
            if let StarTerm::NamedNode(pred_node) = &triple.predicate {
                if matches!(
                    pred_node.iri.as_str(),
                    vocab::RDF_SUBJECT | vocab::RDF_PREDICATE | vocab::RDF_OBJECT
                ) {
                    if let StarTerm::NamedNode(subj_node) = &triple.subject {
                        statements.insert(&subj_node.iri);
                    } else if let StarTerm::BlankNode(subj_node) = &triple.subject {
                        statements.insert(&subj_node.id);
                    }
                }
            }
        }

        statements.len()
    }

    /// Validate that reification patterns are complete
    pub fn validate_reifications(graph: &StarGraph) -> StarResult<()> {
        let mut statements = HashMap::new();

        for triple in graph.triples() {
            if let StarTerm::NamedNode(pred_node) = &triple.predicate {
                match pred_node.iri.as_str() {
                    vocab::RDF_SUBJECT => {
                        if let Some(stmt_id) = extract_statement_id(&triple.subject) {
                            statements.entry(stmt_id).or_insert([false, false, false])[0] = true;
                        }
                    }
                    vocab::RDF_PREDICATE => {
                        if let Some(stmt_id) = extract_statement_id(&triple.subject) {
                            statements.entry(stmt_id).or_insert([false, false, false])[1] = true;
                        }
                    }
                    vocab::RDF_OBJECT => {
                        if let Some(stmt_id) = extract_statement_id(&triple.subject) {
                            statements.entry(stmt_id).or_insert([false, false, false])[2] = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Check for incomplete reifications
        for (stmt_id, completeness) in statements {
            if !completeness.iter().all(|&x| x) {
                return Err(StarError::reification_error(format!(
                    "Incomplete reification for statement {}",
                    stmt_id
                )));
            }
        }

        Ok(())
    }

    fn extract_statement_id(term: &StarTerm) -> Option<String> {
        match term {
            StarTerm::NamedNode(node) => Some(node.iri.clone()),
            StarTerm::BlankNode(node) => Some(format!("_:{}", node.id)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;

    #[test]
    fn test_basic_reification() {
        let mut reificator = Reificator::new(
            ReificationStrategy::StandardReification,
            Some("http://example.org/stmt/".to_string()),
        );

        // Create a quoted triple
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );

        let mut star_graph = StarGraph::new();
        star_graph.insert(outer).unwrap();

        let reified = reificator.reify_graph(&star_graph).unwrap();

        // Should have multiple triples for reification
        assert!(reified.len() > 1);

        // Should contain rdf:type rdf:Statement triple
        let has_type_triple = reified.triples().iter().any(|t| {
            if let (StarTerm::NamedNode(p), StarTerm::NamedNode(o)) = (&t.predicate, &t.object) {
                p.iri == vocab::RDF_TYPE && o.iri == vocab::RDF_STATEMENT
            } else {
                false
            }
        });
        assert!(has_type_triple);
    }

    #[test]
    fn test_dereification() {
        // Create a reified graph manually
        let mut reified_graph = StarGraph::new();

        let stmt_iri = "http://example.org/stmt/1";

        // stmt rdf:type rdf:Statement
        reified_graph
            .insert(StarTriple::new(
                StarTerm::iri(stmt_iri).unwrap(),
                StarTerm::iri(vocab::RDF_TYPE).unwrap(),
                StarTerm::iri(vocab::RDF_STATEMENT).unwrap(),
            ))
            .unwrap();

        // stmt rdf:subject alice
        reified_graph
            .insert(StarTriple::new(
                StarTerm::iri(stmt_iri).unwrap(),
                StarTerm::iri(vocab::RDF_SUBJECT).unwrap(),
                StarTerm::iri("http://example.org/alice").unwrap(),
            ))
            .unwrap();

        // stmt rdf:predicate age
        reified_graph
            .insert(StarTriple::new(
                StarTerm::iri(stmt_iri).unwrap(),
                StarTerm::iri(vocab::RDF_PREDICATE).unwrap(),
                StarTerm::iri("http://example.org/age").unwrap(),
            ))
            .unwrap();

        // stmt rdf:object "25"
        reified_graph
            .insert(StarTriple::new(
                StarTerm::iri(stmt_iri).unwrap(),
                StarTerm::iri(vocab::RDF_OBJECT).unwrap(),
                StarTerm::literal("25").unwrap(),
            ))
            .unwrap();

        // stmt certainty "0.9"
        reified_graph
            .insert(StarTriple::new(
                StarTerm::iri(stmt_iri).unwrap(),
                StarTerm::iri("http://example.org/certainty").unwrap(),
                StarTerm::literal("0.9").unwrap(),
            ))
            .unwrap();

        let mut dereificator = Reificator::new(
            ReificationStrategy::StandardReification,
            Some("http://example.org/stmt/".to_string()),
        );

        let star_graph = dereificator.dereify_graph(&reified_graph).unwrap();

        // Should have one triple with quoted triple as subject
        assert_eq!(star_graph.len(), 1);

        let triple = &star_graph.triples()[0];
        assert!(triple.subject.is_quoted_triple());
    }

    #[test]
    fn test_reification_roundtrip() {
        // Original RDF-star graph
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );

        let mut original_graph = StarGraph::new();
        original_graph.insert(outer).unwrap();

        // Reify
        let mut reificator = Reificator::new(
            ReificationStrategy::StandardReification,
            Some("http://example.org/stmt/".to_string()),
        );
        let reified_graph = reificator.reify_graph(&original_graph).unwrap();

        // Dereify
        let mut dereificator = Reificator::new(
            ReificationStrategy::StandardReification,
            Some("http://example.org/stmt/".to_string()),
        );
        let recovered_graph = dereificator.dereify_graph(&reified_graph).unwrap();

        // Should have the same structure (though possibly different identifiers)
        assert_eq!(recovered_graph.len(), original_graph.len());

        let recovered_triple = &recovered_graph.triples()[0];
        assert!(recovered_triple.subject.is_quoted_triple());
    }

    #[test]
    fn test_utils() {
        let mut graph = StarGraph::new();

        // Add some reification triples
        graph
            .insert(StarTriple::new(
                StarTerm::iri("http://example.org/stmt1").unwrap(),
                StarTerm::iri(vocab::RDF_SUBJECT).unwrap(),
                StarTerm::iri("http://example.org/alice").unwrap(),
            ))
            .unwrap();

        assert!(utils::has_reifications(&graph));
        assert_eq!(utils::count_reifications(&graph), 1);

        // Incomplete reification should fail validation
        assert!(utils::validate_reifications(&graph).is_err());
    }
}
