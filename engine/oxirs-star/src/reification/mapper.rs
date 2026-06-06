use std::collections::HashMap;

use crate::model::{StarGraph, StarTerm, StarTriple};
use crate::{StarError, StarResult};

use super::types::{AnnotationStyle, EmbeddedTriple};
use super::vocab;

/// High-level bidirectional bridge between RDF-star quoted triples and
/// standard-RDF reification / annotation styles.
///
/// # Usage
///
/// ```
/// use oxirs_star::reification::{ReificationBridge, AnnotationStyle};
/// use oxirs_star::model::{StarTerm, StarTriple};
///
/// let bridge = ReificationBridge::new(AnnotationStyle::Reification);
///
/// let inner = StarTriple::new(
///     StarTerm::iri("http://example.org/alice").unwrap(),
///     StarTerm::iri("http://example.org/age").unwrap(),
///     StarTerm::literal("30").unwrap(),
/// );
///
/// let reified = bridge.star_to_reification(&inner);
/// assert!(!reified.is_empty());
/// ```
pub struct ReificationBridge {
    pub style: AnnotationStyle,
    base_iri: String,
    counter: std::sync::atomic::AtomicUsize,
}

impl ReificationBridge {
    pub fn new(style: AnnotationStyle) -> Self {
        Self {
            style,
            base_iri: "http://reification.example/stmt/".to_string(),
            counter: std::sync::atomic::AtomicUsize::new(1),
        }
    }

    pub fn with_base_iri(style: AnnotationStyle, base_iri: impl Into<String>) -> Self {
        Self {
            style,
            base_iri: base_iri.into(),
            counter: std::sync::atomic::AtomicUsize::new(1),
        }
    }

    pub fn style(&self) -> &AnnotationStyle {
        &self.style
    }

    pub fn star_to_reification(&self, triple: &EmbeddedTriple) -> Vec<StarTriple> {
        match self.style {
            AnnotationStyle::Reification => self.to_standard_reification(triple),
            AnnotationStyle::Singleton => self.to_singleton(triple),
            AnnotationStyle::NaryRelation => self.to_nary_relation(triple),
        }
    }

    fn next_id(&self) -> String {
        let n = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("{}{}", self.base_iri, n)
    }

    fn to_standard_reification(&self, triple: &EmbeddedTriple) -> Vec<StarTriple> {
        let stmt_iri = self.next_id();
        let mut triples = Vec::with_capacity(4);

        let stmt_term = match StarTerm::iri(&stmt_iri) {
            Ok(t) => t,
            Err(_) => return triples,
        };
        let rdf_type = match StarTerm::iri(vocab::RDF_TYPE) {
            Ok(t) => t,
            Err(_) => return triples,
        };
        let rdf_statement = match StarTerm::iri(vocab::RDF_STATEMENT) {
            Ok(t) => t,
            Err(_) => return triples,
        };
        let rdf_subject = match StarTerm::iri(vocab::RDF_SUBJECT) {
            Ok(t) => t,
            Err(_) => return triples,
        };
        let rdf_predicate = match StarTerm::iri(vocab::RDF_PREDICATE) {
            Ok(t) => t,
            Err(_) => return triples,
        };
        let rdf_object = match StarTerm::iri(vocab::RDF_OBJECT) {
            Ok(t) => t,
            Err(_) => return triples,
        };

        triples.push(StarTriple::new(stmt_term.clone(), rdf_type, rdf_statement));
        triples.push(StarTriple::new(
            stmt_term.clone(),
            rdf_subject,
            triple.subject.clone(),
        ));
        triples.push(StarTriple::new(
            stmt_term.clone(),
            rdf_predicate,
            triple.predicate.clone(),
        ));
        triples.push(StarTriple::new(
            stmt_term,
            rdf_object,
            triple.object.clone(),
        ));
        triples
    }

    fn to_singleton(&self, triple: &EmbeddedTriple) -> Vec<StarTriple> {
        let prop_iri = self.next_id();
        let mut triples = Vec::with_capacity(2);

        let prop_term = match StarTerm::iri(&prop_iri) {
            Ok(t) => t,
            Err(_) => return triples,
        };
        let singleton_of =
            match StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#singletonPropertyOf") {
                Ok(t) => t,
                Err(_) => return triples,
            };

        triples.push(StarTriple::new(
            triple.subject.clone(),
            prop_term.clone(),
            triple.object.clone(),
        ));
        triples.push(StarTriple::new(
            prop_term,
            singleton_of,
            triple.predicate.clone(),
        ));
        triples
    }

    fn to_nary_relation(&self, triple: &EmbeddedTriple) -> Vec<StarTriple> {
        let node_iri = self.next_id();
        let mut triples = Vec::with_capacity(3);

        let node_term = match StarTerm::iri(&node_iri) {
            Ok(t) => t,
            Err(_) => return triples,
        };

        let nary_subject = match StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject")
        {
            Ok(t) => t,
            Err(_) => return triples,
        };
        let nary_predicate =
            match StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate") {
                Ok(t) => t,
                Err(_) => return triples,
            };
        let nary_object = match StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#object") {
            Ok(t) => t,
            Err(_) => return triples,
        };

        triples.push(StarTriple::new(
            node_term.clone(),
            nary_subject,
            triple.subject.clone(),
        ));
        triples.push(StarTriple::new(
            node_term.clone(),
            nary_predicate,
            triple.predicate.clone(),
        ));
        triples.push(StarTriple::new(
            node_term,
            nary_object,
            triple.object.clone(),
        ));
        triples
    }

    pub fn reification_to_star(&self, stmts: &[StarTriple]) -> Option<EmbeddedTriple> {
        let mut subject = None;
        let mut predicate = None;
        let mut object = None;

        for triple in stmts {
            if let StarTerm::NamedNode(pred_node) = &triple.predicate {
                match pred_node.iri.as_str() {
                    s if s == vocab::RDF_SUBJECT => subject = Some(triple.object.clone()),
                    s if s == vocab::RDF_PREDICATE => predicate = Some(triple.object.clone()),
                    s if s == vocab::RDF_OBJECT => object = Some(triple.object.clone()),
                    _ => {}
                }
            }
        }

        if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
            Some(StarTriple::new(s, p, o))
        } else {
            None
        }
    }

    pub fn convert_graph_to_reification(&self, graph: &[StarTriple]) -> Vec<StarTriple> {
        let mut result = Vec::new();

        for triple in graph {
            let has_quoted = matches!(&triple.subject, StarTerm::QuotedTriple(_))
                || matches!(&triple.object, StarTerm::QuotedTriple(_));

            if has_quoted {
                if let StarTerm::QuotedTriple(inner) = &triple.subject {
                    result.extend(self.star_to_reification(inner));
                }
                if let StarTerm::QuotedTriple(inner) = &triple.object {
                    result.extend(self.star_to_reification(inner));
                }
            } else {
                result.push(triple.clone());
            }
        }

        result
    }

    pub fn convert_reification_to_star(&self, graph: &[StarTriple]) -> Vec<EmbeddedTriple> {
        use std::collections::HashSet;

        let mut stmt_nodes: HashSet<String> = HashSet::new();
        for triple in graph {
            if let (StarTerm::NamedNode(pred), StarTerm::NamedNode(obj)) =
                (&triple.predicate, &triple.object)
            {
                if pred.iri == vocab::RDF_TYPE && obj.iri == vocab::RDF_STATEMENT {
                    if let StarTerm::NamedNode(subj) = &triple.subject {
                        stmt_nodes.insert(subj.iri.clone());
                    }
                }
            }
        }

        let mut clusters: HashMap<String, Vec<StarTriple>> = HashMap::new();
        for triple in graph {
            if let StarTerm::NamedNode(subj) = &triple.subject {
                if stmt_nodes.contains(&subj.iri) {
                    clusters
                        .entry(subj.iri.clone())
                        .or_default()
                        .push(triple.clone());
                }
            }
        }

        let mut embedded_triples = Vec::new();
        for cluster_triples in clusters.values() {
            if let Some(et) = self.reification_to_star(cluster_triples) {
                embedded_triples.push(et);
            }
        }
        embedded_triples
    }
}

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

    for (stmt_id, completeness) in statements {
        if !completeness.iter().all(|&x| x) {
            return Err(StarError::reification_error(format!(
                "Incomplete reification for statement {stmt_id}"
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
