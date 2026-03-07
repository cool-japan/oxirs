// RDF 1.1 reification ↔ RDF-star bidirectional conversion
// Added in v1.1.0 Round 7

/// Standard RDF vocabulary IRIs for reification.
pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
pub const RDF_STATEMENT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement";
pub const RDF_SUBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject";
pub const RDF_PREDICATE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate";
pub const RDF_OBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object";

/// An RDF term (IRI, literal, blank node, or RDF-star quoted triple).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RdfTerm {
    Iri(String),
    Literal {
        value: String,
        datatype: String,
        lang: Option<String>,
    },
    BlankNode(String),
    QuotedTriple(Box<RdfTriple>),
}

impl RdfTerm {
    pub fn iri(s: impl Into<String>) -> Self {
        RdfTerm::Iri(s.into())
    }

    pub fn blank(s: impl Into<String>) -> Self {
        RdfTerm::BlankNode(s.into())
    }

    pub fn literal(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        RdfTerm::Literal {
            value: value.into(),
            datatype: datatype.into(),
            lang: None,
        }
    }

    pub fn literal_lang(value: impl Into<String>, lang: impl Into<String>) -> Self {
        RdfTerm::Literal {
            value: value.into(),
            datatype: "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".to_string(),
            lang: Some(lang.into()),
        }
    }
}

/// An RDF triple.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RdfTriple {
    pub subject: RdfTerm,
    pub predicate: RdfTerm,
    pub object: RdfTerm,
}

impl RdfTriple {
    pub fn new(subject: RdfTerm, predicate: RdfTerm, object: RdfTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }
}

/// A reified RDF statement: a statement node plus the 4 reification triples.
#[derive(Debug, Clone)]
pub struct ReifiedStatement {
    pub node: String,
    pub triples: Vec<RdfTriple>,
}

/// Errors that can occur during reification/dereification.
#[derive(Debug)]
pub enum ReifierError {
    MissingSubject,
    MissingPredicate,
    MissingObject,
    MissingType,
    AmbiguousReification,
}

impl std::fmt::Display for ReifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReifierError::MissingSubject => write!(f, "Reification missing rdf:subject"),
            ReifierError::MissingPredicate => write!(f, "Reification missing rdf:predicate"),
            ReifierError::MissingObject => write!(f, "Reification missing rdf:object"),
            ReifierError::MissingType => write!(f, "Reification missing rdf:type rdf:Statement"),
            ReifierError::AmbiguousReification => write!(
                f,
                "Ambiguous reification: multiple values for same property"
            ),
        }
    }
}

impl std::error::Error for ReifierError {}

/// Bidirectional converter between RDF 1.1 reification and RDF-star quoted triples.
pub struct TripleReifier {
    blank_node_counter: u64,
}

impl TripleReifier {
    pub fn new() -> Self {
        Self {
            blank_node_counter: 0,
        }
    }

    /// Generate a fresh blank node ID.
    fn fresh_blank_node(&mut self) -> String {
        let id = format!("_:rs{}", self.blank_node_counter);
        self.blank_node_counter += 1;
        id
    }

    /// Convert a triple to RDF 1.1 reification form (4 triples).
    ///
    /// Given triple (S, P, O), produces:
    /// - _:rs rdf:type rdf:Statement
    /// - _:rs rdf:subject S
    /// - _:rs rdf:predicate P
    /// - _:rs rdf:object O
    pub fn reify(&mut self, triple: &RdfTriple) -> ReifiedStatement {
        let node = self.fresh_blank_node();
        let stmt_node = RdfTerm::blank(&node);

        let triples = vec![
            RdfTriple::new(
                stmt_node.clone(),
                RdfTerm::iri(RDF_TYPE),
                RdfTerm::iri(RDF_STATEMENT),
            ),
            RdfTriple::new(
                stmt_node.clone(),
                RdfTerm::iri(RDF_SUBJECT),
                triple.subject.clone(),
            ),
            RdfTriple::new(
                stmt_node.clone(),
                RdfTerm::iri(RDF_PREDICATE),
                triple.predicate.clone(),
            ),
            RdfTriple::new(stmt_node, RdfTerm::iri(RDF_OBJECT), triple.object.clone()),
        ];
        ReifiedStatement { node, triples }
    }

    /// Reconstruct a triple from RDF 1.1 reification triples for the given node.
    pub fn dereify(node: &str, triples: &[RdfTriple]) -> Result<RdfTriple, ReifierError> {
        let node_term = RdfTerm::blank(node);
        let mut has_type = false;
        let mut subject: Option<RdfTerm> = None;
        let mut predicate: Option<RdfTerm> = None;
        let mut object: Option<RdfTerm> = None;

        for triple in triples {
            if triple.subject != node_term {
                continue;
            }
            match &triple.predicate {
                RdfTerm::Iri(p) if p.as_str() == RDF_TYPE => {
                    if triple.object == RdfTerm::iri(RDF_STATEMENT) {
                        has_type = true;
                    }
                }
                RdfTerm::Iri(p) if p.as_str() == RDF_SUBJECT => {
                    if subject.is_some() {
                        return Err(ReifierError::AmbiguousReification);
                    }
                    subject = Some(triple.object.clone());
                }
                RdfTerm::Iri(p) if p.as_str() == RDF_PREDICATE => {
                    if predicate.is_some() {
                        return Err(ReifierError::AmbiguousReification);
                    }
                    predicate = Some(triple.object.clone());
                }
                RdfTerm::Iri(p) if p.as_str() == RDF_OBJECT => {
                    if object.is_some() {
                        return Err(ReifierError::AmbiguousReification);
                    }
                    object = Some(triple.object.clone());
                }
                _ => {}
            }
        }

        if !has_type {
            return Err(ReifierError::MissingType);
        }
        let s = subject.ok_or(ReifierError::MissingSubject)?;
        let p = predicate.ok_or(ReifierError::MissingPredicate)?;
        let o = object.ok_or(ReifierError::MissingObject)?;
        Ok(RdfTriple::new(s, p, o))
    }

    /// Check if a set of triples contains a complete reification for the given node.
    pub fn is_complete_reification(node: &str, triples: &[RdfTriple]) -> bool {
        let node_term = RdfTerm::blank(node);
        let mut has_type = false;
        let mut has_subject = false;
        let mut has_predicate = false;
        let mut has_object = false;

        for triple in triples {
            if triple.subject != node_term {
                continue;
            }
            match &triple.predicate {
                RdfTerm::Iri(p) if p.as_str() == RDF_TYPE => {
                    if triple.object == RdfTerm::iri(RDF_STATEMENT) {
                        has_type = true;
                    }
                }
                RdfTerm::Iri(p) if p.as_str() == RDF_SUBJECT => has_subject = true,
                RdfTerm::Iri(p) if p.as_str() == RDF_PREDICATE => has_predicate = true,
                RdfTerm::Iri(p) if p.as_str() == RDF_OBJECT => has_object = true,
                _ => {}
            }
        }
        has_type && has_subject && has_predicate && has_object
    }

    /// Find all reification statement nodes in a set of triples.
    /// A node is a reification node if it has `rdf:type rdf:Statement`.
    pub fn find_reification_nodes(triples: &[RdfTriple]) -> Vec<String> {
        let mut nodes = Vec::new();
        for triple in triples {
            if triple.predicate == RdfTerm::iri(RDF_TYPE)
                && triple.object == RdfTerm::iri(RDF_STATEMENT)
            {
                match &triple.subject {
                    RdfTerm::BlankNode(id) => {
                        let formatted = format!("_:{id}");
                        if !nodes.contains(&formatted) {
                            nodes.push(formatted);
                        }
                    }
                    RdfTerm::Iri(iri) => {
                        if !nodes.contains(iri) {
                            nodes.push(iri.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
        nodes
    }

    /// Reify a triple and attach annotation triples to the statement node.
    ///
    /// Annotations are extra (node, predicate, object) triples attached to
    /// the reification node.
    pub fn reify_with_annotations(
        &mut self,
        triple: &RdfTriple,
        annotations: &[(String, RdfTerm)],
    ) -> Vec<RdfTriple> {
        let stmt = self.reify(triple);
        let stmt_node = RdfTerm::blank(&stmt.node);
        let mut all_triples = stmt.triples;
        for (pred_iri, obj) in annotations {
            all_triples.push(RdfTriple::new(
                stmt_node.clone(),
                RdfTerm::iri(pred_iri),
                obj.clone(),
            ));
        }
        all_triples
    }

    /// Reify multiple triples, returning one ReifiedStatement per input triple.
    pub fn reify_batch(&mut self, triples: &[RdfTriple]) -> Vec<ReifiedStatement> {
        triples.iter().map(|t| self.reify(t)).collect()
    }
}

impl Default for TripleReifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_triple() -> RdfTriple {
        RdfTriple::new(
            RdfTerm::iri("http://example.org/s"),
            RdfTerm::iri("http://example.org/p"),
            RdfTerm::iri("http://example.org/o"),
        )
    }

    // ---- reify ----

    #[test]
    fn test_reify_produces_4_triples() {
        let mut r = TripleReifier::new();
        let triple = make_simple_triple();
        let stmt = r.reify(&triple);
        assert_eq!(stmt.triples.len(), 4);
    }

    #[test]
    fn test_reify_contains_type_statement() {
        let mut r = TripleReifier::new();
        let triple = make_simple_triple();
        let stmt = r.reify(&triple);
        let has_type = stmt.triples.iter().any(|t| {
            t.predicate == RdfTerm::iri(RDF_TYPE) && t.object == RdfTerm::iri(RDF_STATEMENT)
        });
        assert!(has_type);
    }

    #[test]
    fn test_reify_contains_subject() {
        let mut r = TripleReifier::new();
        let triple = make_simple_triple();
        let stmt = r.reify(&triple);
        let has_subj = stmt.triples.iter().any(|t| {
            t.predicate == RdfTerm::iri(RDF_SUBJECT)
                && t.object == RdfTerm::iri("http://example.org/s")
        });
        assert!(has_subj);
    }

    #[test]
    fn test_reify_contains_predicate() {
        let mut r = TripleReifier::new();
        let triple = make_simple_triple();
        let stmt = r.reify(&triple);
        let has_pred = stmt.triples.iter().any(|t| {
            t.predicate == RdfTerm::iri(RDF_PREDICATE)
                && t.object == RdfTerm::iri("http://example.org/p")
        });
        assert!(has_pred);
    }

    #[test]
    fn test_reify_contains_object() {
        let mut r = TripleReifier::new();
        let triple = make_simple_triple();
        let stmt = r.reify(&triple);
        let has_obj = stmt.triples.iter().any(|t| {
            t.predicate == RdfTerm::iri(RDF_OBJECT)
                && t.object == RdfTerm::iri("http://example.org/o")
        });
        assert!(has_obj);
    }

    #[test]
    fn test_reify_node_format() {
        let mut r = TripleReifier::new();
        let triple = make_simple_triple();
        let stmt = r.reify(&triple);
        assert!(stmt.node.starts_with("_:rs"), "Node: {}", stmt.node);
    }

    // ---- fresh_blank_node increments ----

    #[test]
    fn test_fresh_blank_node_increments() {
        let mut r = TripleReifier::new();
        let n1 = r.fresh_blank_node();
        let n2 = r.fresh_blank_node();
        let n3 = r.fresh_blank_node();
        assert_ne!(n1, n2);
        assert_ne!(n2, n3);
        assert!(n1.contains("0"), "n1={n1}");
        assert!(n2.contains("1"), "n2={n2}");
        assert!(n3.contains("2"), "n3={n3}");
    }

    // ---- dereify ----

    #[test]
    fn test_dereify_round_trip() {
        let mut r = TripleReifier::new();
        let original = make_simple_triple();
        let stmt = r.reify(&original);
        let recovered = TripleReifier::dereify(&stmt.node, &stmt.triples).unwrap();
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_dereify_missing_type_error() {
        let triples = vec![RdfTriple::new(
            RdfTerm::blank("_:rs0"),
            RdfTerm::iri(RDF_SUBJECT),
            RdfTerm::iri("http://s"),
        )];
        let result = TripleReifier::dereify("_:rs0", &triples);
        assert!(matches!(result, Err(ReifierError::MissingType)));
    }

    #[test]
    fn test_dereify_missing_subject_error() {
        let node = "_:rs0";
        let node_term = RdfTerm::blank(node);
        let triples = vec![
            RdfTriple::new(
                node_term.clone(),
                RdfTerm::iri(RDF_TYPE),
                RdfTerm::iri(RDF_STATEMENT),
            ),
            RdfTriple::new(
                node_term.clone(),
                RdfTerm::iri(RDF_PREDICATE),
                RdfTerm::iri("http://p"),
            ),
            RdfTriple::new(
                node_term,
                RdfTerm::iri(RDF_OBJECT),
                RdfTerm::iri("http://o"),
            ),
        ];
        let result = TripleReifier::dereify(node, &triples);
        assert!(matches!(result, Err(ReifierError::MissingSubject)));
    }

    #[test]
    fn test_dereify_missing_predicate_error() {
        let node = "_:rs0";
        let node_term = RdfTerm::blank(node);
        let triples = vec![
            RdfTriple::new(
                node_term.clone(),
                RdfTerm::iri(RDF_TYPE),
                RdfTerm::iri(RDF_STATEMENT),
            ),
            RdfTriple::new(
                node_term.clone(),
                RdfTerm::iri(RDF_SUBJECT),
                RdfTerm::iri("http://s"),
            ),
            RdfTriple::new(
                node_term,
                RdfTerm::iri(RDF_OBJECT),
                RdfTerm::iri("http://o"),
            ),
        ];
        let result = TripleReifier::dereify(node, &triples);
        assert!(matches!(result, Err(ReifierError::MissingPredicate)));
    }

    #[test]
    fn test_dereify_missing_object_error() {
        let node = "_:rs0";
        let node_term = RdfTerm::blank(node);
        let triples = vec![
            RdfTriple::new(
                node_term.clone(),
                RdfTerm::iri(RDF_TYPE),
                RdfTerm::iri(RDF_STATEMENT),
            ),
            RdfTriple::new(
                node_term.clone(),
                RdfTerm::iri(RDF_SUBJECT),
                RdfTerm::iri("http://s"),
            ),
            RdfTriple::new(
                node_term,
                RdfTerm::iri(RDF_PREDICATE),
                RdfTerm::iri("http://p"),
            ),
        ];
        let result = TripleReifier::dereify(node, &triples);
        assert!(matches!(result, Err(ReifierError::MissingObject)));
    }

    // ---- is_complete_reification ----

    #[test]
    fn test_is_complete_reification_true() {
        let mut r = TripleReifier::new();
        let triple = make_simple_triple();
        let stmt = r.reify(&triple);
        assert!(TripleReifier::is_complete_reification(
            &stmt.node,
            &stmt.triples
        ));
    }

    #[test]
    fn test_is_complete_reification_false_missing_object() {
        let node = "_:rs0";
        let node_term = RdfTerm::blank(node);
        let triples = vec![
            RdfTriple::new(
                node_term.clone(),
                RdfTerm::iri(RDF_TYPE),
                RdfTerm::iri(RDF_STATEMENT),
            ),
            RdfTriple::new(
                node_term.clone(),
                RdfTerm::iri(RDF_SUBJECT),
                RdfTerm::iri("http://s"),
            ),
            RdfTriple::new(
                node_term,
                RdfTerm::iri(RDF_PREDICATE),
                RdfTerm::iri("http://p"),
            ),
        ];
        assert!(!TripleReifier::is_complete_reification(node, &triples));
    }

    #[test]
    fn test_is_complete_reification_false_empty() {
        assert!(!TripleReifier::is_complete_reification("_:rs0", &[]));
    }

    #[test]
    fn test_is_complete_reification_false_wrong_type() {
        let node = "_:rs0";
        let node_term = RdfTerm::blank(node);
        // type is wrong
        let triples = vec![
            RdfTriple::new(
                node_term.clone(),
                RdfTerm::iri(RDF_TYPE),
                RdfTerm::iri("http://other#Thing"),
            ),
            RdfTriple::new(
                node_term.clone(),
                RdfTerm::iri(RDF_SUBJECT),
                RdfTerm::iri("http://s"),
            ),
            RdfTriple::new(
                node_term.clone(),
                RdfTerm::iri(RDF_PREDICATE),
                RdfTerm::iri("http://p"),
            ),
            RdfTriple::new(
                node_term,
                RdfTerm::iri(RDF_OBJECT),
                RdfTerm::iri("http://o"),
            ),
        ];
        assert!(!TripleReifier::is_complete_reification(node, &triples));
    }

    // ---- find_reification_nodes ----

    #[test]
    fn test_find_reification_nodes_none() {
        let triples = vec![make_simple_triple()];
        assert!(TripleReifier::find_reification_nodes(&triples).is_empty());
    }

    #[test]
    fn test_find_reification_nodes_one() {
        let mut r = TripleReifier::new();
        let stmt = r.reify(&make_simple_triple());
        let nodes = TripleReifier::find_reification_nodes(&stmt.triples);
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn test_find_reification_nodes_multiple() {
        let mut r = TripleReifier::new();
        let stmt1 = r.reify(&make_simple_triple());
        let stmt2 = r.reify(&RdfTriple::new(
            RdfTerm::iri("http://a"),
            RdfTerm::iri("http://b"),
            RdfTerm::iri("http://c"),
        ));
        let mut all_triples = stmt1.triples.clone();
        all_triples.extend(stmt2.triples.clone());
        let nodes = TripleReifier::find_reification_nodes(&all_triples);
        assert_eq!(nodes.len(), 2);
    }

    // ---- reify_with_annotations ----

    #[test]
    fn test_reify_with_annotations_adds_extra_triples() {
        let mut r = TripleReifier::new();
        let triple = make_simple_triple();
        let annotations = vec![(
            "http://example.org/certainty".to_string(),
            RdfTerm::literal("0.9", "http://www.w3.org/2001/XMLSchema#decimal"),
        )];
        let triples = r.reify_with_annotations(&triple, &annotations);
        assert_eq!(triples.len(), 5); // 4 reification + 1 annotation
    }

    #[test]
    fn test_reify_with_annotations_no_annotations() {
        let mut r = TripleReifier::new();
        let triple = make_simple_triple();
        let triples = r.reify_with_annotations(&triple, &[]);
        assert_eq!(triples.len(), 4);
    }

    #[test]
    fn test_reify_with_annotations_multiple_annotations() {
        let mut r = TripleReifier::new();
        let triple = make_simple_triple();
        let annotations = vec![
            ("http://a".to_string(), RdfTerm::iri("http://v1")),
            ("http://b".to_string(), RdfTerm::iri("http://v2")),
            ("http://c".to_string(), RdfTerm::iri("http://v3")),
        ];
        let triples = r.reify_with_annotations(&triple, &annotations);
        assert_eq!(triples.len(), 7); // 4 + 3
    }

    // ---- reify_batch ----

    #[test]
    fn test_reify_batch_empty() {
        let mut r = TripleReifier::new();
        let stmts = r.reify_batch(&[]);
        assert!(stmts.is_empty());
    }

    #[test]
    fn test_reify_batch_multiple() {
        let mut r = TripleReifier::new();
        let triples = vec![
            make_simple_triple(),
            RdfTriple::new(
                RdfTerm::iri("http://a"),
                RdfTerm::iri("http://b"),
                RdfTerm::iri("http://c"),
            ),
            RdfTriple::new(
                RdfTerm::iri("http://x"),
                RdfTerm::iri("http://y"),
                RdfTerm::iri("http://z"),
            ),
        ];
        let stmts = r.reify_batch(&triples);
        assert_eq!(stmts.len(), 3);
        // Each should have a unique node
        let nodes: std::collections::HashSet<_> = stmts.iter().map(|s| &s.node).collect();
        assert_eq!(nodes.len(), 3);
    }

    // ---- nested quoted triples ----

    #[test]
    fn test_reify_nested_quoted_triple() {
        let mut r = TripleReifier::new();
        let inner = make_simple_triple();
        let outer = RdfTriple::new(
            RdfTerm::QuotedTriple(Box::new(inner.clone())),
            RdfTerm::iri("http://ex/p"),
            RdfTerm::iri("http://ex/o"),
        );
        let stmt = r.reify(&outer);
        assert_eq!(stmt.triples.len(), 4);
        // The rdf:subject triple's object should be the quoted triple
        let subj_triple = stmt
            .triples
            .iter()
            .find(|t| t.predicate == RdfTerm::iri(RDF_SUBJECT))
            .unwrap();
        assert!(matches!(&subj_triple.object, RdfTerm::QuotedTriple(_)));
    }

    #[test]
    fn test_reify_with_literal_object() {
        let mut r = TripleReifier::new();
        let triple = RdfTriple::new(
            RdfTerm::iri("http://ex/s"),
            RdfTerm::iri("http://ex/name"),
            RdfTerm::literal("Alice", "http://www.w3.org/2001/XMLSchema#string"),
        );
        let stmt = r.reify(&triple);
        assert_eq!(stmt.triples.len(), 4);
        let obj_triple = stmt
            .triples
            .iter()
            .find(|t| t.predicate == RdfTerm::iri(RDF_OBJECT))
            .unwrap();
        assert!(matches!(&obj_triple.object, RdfTerm::Literal { .. }));
    }

    // ---- ReifierError display ----

    #[test]
    fn test_reifier_error_display() {
        let errs = [
            ReifierError::MissingSubject,
            ReifierError::MissingPredicate,
            ReifierError::MissingObject,
            ReifierError::MissingType,
            ReifierError::AmbiguousReification,
        ];
        for err in &errs {
            let s = format!("{err}");
            assert!(!s.is_empty());
        }
    }

    // ---- RdfTerm helpers ----

    #[test]
    fn test_rdf_term_iri() {
        let t = RdfTerm::iri("http://example.org/");
        assert_eq!(t, RdfTerm::Iri("http://example.org/".to_string()));
    }

    #[test]
    fn test_rdf_term_blank() {
        let t = RdfTerm::blank("b0");
        assert_eq!(t, RdfTerm::BlankNode("b0".to_string()));
    }

    #[test]
    fn test_rdf_term_literal_lang() {
        let t = RdfTerm::literal_lang("hello", "en");
        match t {
            RdfTerm::Literal { value, lang, .. } => {
                assert_eq!(value, "hello");
                assert_eq!(lang, Some("en".to_string()));
            }
            _ => panic!("expected literal"),
        }
    }

    // ---- default ----

    #[test]
    fn test_default() {
        let r = TripleReifier::default();
        assert_eq!(r.blank_node_counter, 0);
    }
}
