//! Reification utilities for converting between RDF-star and standard RDF.
//!
//! This module provides functionality to convert quoted triples to/from
//! standard RDF reification using rdf:Statement, rdf:subject, rdf:predicate, rdf:object.

use std::collections::HashMap;

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
}

/// Standard RDF to RDF-star conversion (de-reification)
pub struct Dereificator {
    context: ReificationContext,
}

impl Dereificator {
    /// Create a new dereificator
    pub fn new(strategy: ReificationStrategy, base_iri: Option<String>) -> Self {
        Self {
            context: ReificationContext::new(strategy, base_iri),
        }
    }

    /// Convert standard RDF with reification back to RDF-star
    pub fn dereify_graph(&mut self, reified_graph: &StarGraph) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "dereify_graph");
        let _enter = span.enter();

        // First pass: identify reification patterns
        let mut statements = HashMap::new();
        let mut non_reification_triples = Vec::new();

        self.identify_reifications(reified_graph, &mut statements, &mut non_reification_triples)?;

        // Second pass: reconstruct quoted triples
        let mut star_graph = StarGraph::new();

        // Add non-reification triples, substituting statement identifiers with quoted triples
        for triple in non_reification_triples {
            let star_triple = self.substitute_quoted_triples(triple, &statements)?;
            star_graph.insert(star_triple)?;
        }

        debug!(
            "Dereified {} standard RDF triples to {} RDF-star triples",
            reified_graph.len(),
            star_graph.len()
        );
        Ok(star_graph)
    }

    /// Identify reification patterns in the graph
    fn identify_reifications(
        &mut self,
        graph: &StarGraph,
        statements: &mut HashMap<String, (StarTerm, StarTerm, StarTerm)>,
        non_reification_triples: &mut Vec<StarTriple>,
    ) -> StarResult<()> {
        let mut potential_statements = HashMap::new();

        for triple in graph.triples() {
            // Check if this is a reification triple
            if let Some(stmt_id) = self.extract_statement_id(&triple.subject) {
                if let Some(reification_property) = self.get_reification_property(&triple.predicate)
                {
                    match reification_property.as_str() {
                        "subject" => {
                            potential_statements
                                .entry(stmt_id)
                                .or_insert((None, None, None))
                                .0 = Some(triple.object.clone());
                        }
                        "predicate" => {
                            potential_statements
                                .entry(stmt_id)
                                .or_insert((None, None, None))
                                .1 = Some(triple.object.clone());
                        }
                        "object" => {
                            potential_statements
                                .entry(stmt_id)
                                .or_insert((None, None, None))
                                .2 = Some(triple.object.clone());
                        }
                        "type" => {
                            // Skip rdf:type rdf:Statement triples - they're reification metadata
                            // We don't include them in the dereified graph
                        }
                        _ => {
                            // Not a reification property, add to non-reification triples
                            non_reification_triples.push(triple.clone());
                        }
                    }
                } else {
                    // Not a reification property, add to non-reification triples
                    non_reification_triples.push(triple.clone());
                }
            } else {
                // Not a statement identifier, add to non-reification triples
                non_reification_triples.push(triple.clone());
            }
        }

        // Convert complete reifications to statements
        for (stmt_id, (subject, predicate, object)) in potential_statements {
            if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
                statements.insert(stmt_id, (s, p, o));
            }
        }

        Ok(())
    }

    /// Extract statement identifier if this term represents a reified statement
    fn extract_statement_id(&self, term: &StarTerm) -> Option<String> {
        match term {
            StarTerm::NamedNode(node) => {
                if node.iri.starts_with(&self.context.base_iri) {
                    Some(node.iri.clone())
                } else {
                    None
                }
            }
            StarTerm::BlankNode(node) => {
                if matches!(self.context.strategy, ReificationStrategy::BlankNodes) {
                    Some(format!("_:{}", node.id))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if this is a reification property and return its type
    fn get_reification_property(&self, term: &StarTerm) -> Option<String> {
        if let StarTerm::NamedNode(node) = term {
            match node.iri.as_str() {
                vocab::RDF_SUBJECT => Some("subject".to_string()),
                vocab::RDF_PREDICATE => Some("predicate".to_string()),
                vocab::RDF_OBJECT => Some("object".to_string()),
                vocab::RDF_TYPE => Some("type".to_string()),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Substitute statement identifiers with quoted triples in a triple
    fn substitute_quoted_triples(
        &self,
        triple: StarTriple,
        statements: &HashMap<String, (StarTerm, StarTerm, StarTerm)>,
    ) -> StarResult<StarTriple> {
        let subject = self.substitute_term(triple.subject, statements)?;
        let predicate = self.substitute_term(triple.predicate, statements)?;
        let object = self.substitute_term(triple.object, statements)?;

        Ok(StarTriple::new(subject, predicate, object))
    }

    /// Substitute a term if it's a statement identifier
    fn substitute_term(
        &self,
        term: StarTerm,
        statements: &HashMap<String, (StarTerm, StarTerm, StarTerm)>,
    ) -> StarResult<StarTerm> {
        if let Some(stmt_id) = self.extract_statement_id(&term) {
            if let Some((subject, predicate, object)) = statements.get(&stmt_id) {
                // Recursively substitute in the components
                let sub_subject = self.substitute_term(subject.clone(), statements)?;
                let sub_predicate = self.substitute_term(predicate.clone(), statements)?;
                let sub_object = self.substitute_term(object.clone(), statements)?;

                let quoted_triple = StarTriple::new(sub_subject, sub_predicate, sub_object);
                Ok(StarTerm::quoted_triple(quoted_triple))
            } else {
                Ok(term)
            }
        } else {
            Ok(term)
        }
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
                return Err(StarError::ReificationError(format!(
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
mod tests {
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

        let mut dereificator = Dereificator::new(
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
        let mut dereificator = Dereificator::new(
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
