use std::collections::HashMap;

use crate::model::StarTriple;

/// Reification strategy for handling quoted triples
#[derive(Debug, Clone, PartialEq)]
pub enum ReificationStrategy {
    /// Standard RDF reification using rdf:Statement
    StandardReification,
    /// Use unique IRIs for each quoted triple
    UniqueIris,
    /// Use blank nodes for quoted triples
    BlankNodes,
    /// Singleton properties - each statement gets a unique property
    SingletonProperties,
}

/// Reification context for managing identifiers and mappings
#[derive(Debug)]
pub struct ReificationContext {
    pub strategy: ReificationStrategy,
    pub counter: usize,
    pub base_iri: String,
    pub triple_to_id: HashMap<String, String>,
    pub id_to_triple: HashMap<String, StarTriple>,
}

impl ReificationContext {
    pub fn new(strategy: ReificationStrategy, base_iri: Option<String>) -> Self {
        Self {
            strategy,
            counter: 0,
            base_iri: base_iri.unwrap_or_else(|| "http://example.org/statement/".to_string()),
            triple_to_id: HashMap::new(),
            id_to_triple: HashMap::new(),
        }
    }

    pub fn generate_id(&mut self, triple: &StarTriple) -> String {
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
            ReificationStrategy::SingletonProperties => {
                self.counter += 1;
                format!("{}property/{}", self.base_iri, self.counter)
            }
        };

        self.triple_to_id.insert(triple_key, id.clone());
        self.id_to_triple.insert(id.clone(), triple.clone());
        id
    }

    pub fn get_id(&self, triple: &StarTriple) -> Option<&String> {
        let triple_key = format!("{}|{}|{}", triple.subject, triple.predicate, triple.object);
        self.triple_to_id.get(&triple_key)
    }

    pub fn get_triple(&self, id: &str) -> Option<&StarTriple> {
        self.id_to_triple.get(id)
    }
}

/// Advanced reification strategies with hybrid approaches
#[derive(Debug, Clone, PartialEq)]
pub enum AdvancedReificationStrategy {
    Standard(ReificationStrategy),
    Hybrid {
        simple_strategy: ReificationStrategy,
        nested_strategy: ReificationStrategy,
        predicate_strategies: HashMap<String, ReificationStrategy>,
    },
    Conditional {
        default_strategy: ReificationStrategy,
        rules: Vec<ReificationRule>,
    },
    Optimized {
        base_strategy: ReificationStrategy,
        aggressive_caching: bool,
        max_cache_size: usize,
    },
}

/// Rule for conditional reification strategy selection
#[derive(Debug, Clone, PartialEq)]
pub struct ReificationRule {
    pub condition: ReificationCondition,
    pub strategy: ReificationStrategy,
    pub priority: u32,
}

/// Conditions for reification strategy selection
#[derive(Debug, Clone, PartialEq)]
pub enum ReificationCondition {
    PredicateIri(String),
    SubjectType(TermType),
    ObjectType(TermType),
    NestingDepth(usize),
    GraphSize(usize),
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

/// Statistics for reification operations
#[derive(Debug, Clone, Default)]
pub struct ReificationStatistics {
    pub total_triples: usize,
    pub quoted_triples: usize,
    pub reification_triples: usize,
    pub cache_hit_rate: f64,
    pub avg_processing_time: f64,
    pub strategy_usage: HashMap<String, usize>,
}

/// A triple used as an embedded (quoted) subject or object of another triple.
pub type EmbeddedTriple = StarTriple;

/// Controls which encoding strategy is used when converting RDF-star quoted
/// triples into standard RDF graphs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum AnnotationStyle {
    /// Standard RDF reification: `rdf:Statement` + `rdf:subject /predicate /object`.
    #[default]
    Reification,
    /// Singleton properties: each statement gets a unique property IRI.
    Singleton,
    /// N-ary relation pattern: introduce an intermediate node linked via
    /// domain-specific predicates.
    NaryRelation,
}
