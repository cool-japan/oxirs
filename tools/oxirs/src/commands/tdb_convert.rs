//! Bridges between `oxirs-core` RDF terms (used by the in-RAM `RdfStore`
//! backend and the RDF parsers/serializers) and the on-disk `oxirs-tdb`
//! dictionary term representation (used by the `tdb2` backend), plus TDB2
//! dataset detection.
//!
//! This lets `oxirs import`/`oxirs query`/`oxirs export` share one parser
//! and one serializer pipeline across both storage backends: only the term
//! representation at the store boundary differs.

use oxirs_core::model::{
    BlankNode, GraphName as CoreGraphName, Literal, NamedNode, Object, Predicate, Subject,
};
use oxirs_tdb::dictionary::Term as TdbTerm;
use std::path::Path;

/// True when `dataset_dir` looks like an on-disk `oxirs-tdb` dataset, as
/// opposed to the in-memory `RdfStore`'s N-Quads-log format.
///
/// `oxirs-tdb` always keeps its single data file at `<dir>/data.tdb` (see
/// `oxirs_tdb::TdbConfig::new`), while `RdfStore` persists to `<dir>/data.nq`
/// — the two formats are mutually exclusive on disk, so this is a reliable
/// discriminator without requiring an explicit CLI flag on every command.
pub fn is_tdb2_dataset(dataset_dir: &Path) -> bool {
    dataset_dir.join("data.tdb").is_file()
}

// ─── oxirs-core -> oxirs-tdb (used by `import`) ───────────────────────────

/// Convert an `oxirs-core` [`Subject`] into a TDB dictionary [`TdbTerm`].
pub fn subject_to_tdb_term(subject: &Subject) -> Result<TdbTerm, String> {
    match subject {
        Subject::NamedNode(n) => Ok(TdbTerm::Iri(n.as_str().to_string())),
        Subject::BlankNode(b) => Ok(TdbTerm::BlankNode(b.as_str().to_string())),
        Subject::Variable(v) => Err(format!(
            "cannot import a variable as an RDF subject: ?{}",
            v.name()
        )),
        Subject::QuotedTriple(_) => {
            Err("RDF-star quoted-triple subjects are not supported by the tdb2 backend".to_string())
        }
    }
}

/// Convert an `oxirs-core` [`Predicate`] into a TDB dictionary [`TdbTerm`].
pub fn predicate_to_tdb_term(predicate: &Predicate) -> Result<TdbTerm, String> {
    match predicate {
        Predicate::NamedNode(n) => Ok(TdbTerm::Iri(n.as_str().to_string())),
        Predicate::Variable(v) => Err(format!(
            "cannot import a variable as an RDF predicate: ?{}",
            v.name()
        )),
    }
}

/// Convert an `oxirs-core` [`Object`] into a TDB dictionary [`TdbTerm`].
pub fn object_to_tdb_term(object: &Object) -> Result<TdbTerm, String> {
    match object {
        Object::NamedNode(n) => Ok(TdbTerm::Iri(n.as_str().to_string())),
        Object::BlankNode(b) => Ok(TdbTerm::BlankNode(b.as_str().to_string())),
        Object::Literal(lit) => {
            if let Some(lang) = lit.language() {
                Ok(TdbTerm::Literal {
                    value: lit.value().to_string(),
                    language: Some(lang.to_string()),
                    datatype: None,
                })
            } else {
                Ok(TdbTerm::Literal {
                    value: lit.value().to_string(),
                    language: None,
                    datatype: Some(lit.datatype().as_str().to_string()),
                })
            }
        }
        Object::Variable(v) => Err(format!(
            "cannot import a variable as an RDF object: ?{}",
            v.name()
        )),
        Object::QuotedTriple(_) => {
            Err("RDF-star quoted-triple objects are not supported by the tdb2 backend".to_string())
        }
    }
}

/// Convert an `oxirs-core` [`CoreGraphName`] into an optional TDB graph term
/// (`None` means the default graph, routed to the triple indexes).
pub fn graph_name_to_tdb_term(graph: &CoreGraphName) -> Result<Option<TdbTerm>, String> {
    match graph {
        CoreGraphName::DefaultGraph => Ok(None),
        CoreGraphName::NamedNode(n) => Ok(Some(TdbTerm::Iri(n.as_str().to_string()))),
        CoreGraphName::BlankNode(b) => Ok(Some(TdbTerm::BlankNode(b.as_str().to_string()))),
        CoreGraphName::Variable(v) => Err(format!(
            "cannot import a variable as an RDF graph name: ?{}",
            v.name()
        )),
    }
}

// ─── oxirs-tdb -> oxirs-core (used by `export`/`query`) ───────────────────

/// Convert a TDB dictionary [`TdbTerm`] into an `oxirs-core` [`Subject`].
pub fn tdb_term_to_subject(term: &TdbTerm) -> Result<Subject, String> {
    match term {
        TdbTerm::Iri(iri) => NamedNode::new(iri)
            .map(Subject::NamedNode)
            .map_err(|e| e.to_string()),
        TdbTerm::BlankNode(id) => BlankNode::new(id)
            .map(Subject::BlankNode)
            .map_err(|e| e.to_string()),
        TdbTerm::Literal { .. } => Err("a literal cannot be used as an RDF subject".to_string()),
    }
}

/// Convert a TDB dictionary [`TdbTerm`] into an `oxirs-core` [`NamedNode`]
/// (used for predicates and named graphs, which must be IRIs).
pub fn tdb_term_to_named_node(term: &TdbTerm) -> Result<NamedNode, String> {
    match term {
        TdbTerm::Iri(iri) => NamedNode::new(iri).map_err(|e| e.to_string()),
        _ => Err("only an IRI can be used as an RDF predicate or named graph".to_string()),
    }
}

/// Convert a TDB dictionary [`TdbTerm`] into an `oxirs-core` [`Object`].
pub fn tdb_term_to_object(term: &TdbTerm) -> Result<Object, String> {
    match term {
        TdbTerm::Iri(iri) => NamedNode::new(iri)
            .map(Object::NamedNode)
            .map_err(|e| e.to_string()),
        TdbTerm::BlankNode(id) => BlankNode::new(id)
            .map(Object::BlankNode)
            .map_err(|e| e.to_string()),
        TdbTerm::Literal {
            value,
            language,
            datatype,
        } => {
            if let Some(lang) = language {
                Literal::new_language_tagged_literal(value, lang)
                    .map(Object::Literal)
                    .map_err(|e| e.to_string())
            } else if let Some(dt) = datatype {
                let dt_node = NamedNode::new(dt).map_err(|e| e.to_string())?;
                Ok(Object::Literal(Literal::new_typed_literal(value, dt_node)))
            } else {
                Ok(Object::Literal(Literal::new_simple_literal(value)))
            }
        }
    }
}

/// Convert a TDB quad-scan [`oxirs_tdb::GraphName`] into an `oxirs-core`
/// [`CoreGraphName`].
pub fn tdb_graph_to_core_graph_name(graph: &oxirs_tdb::GraphName) -> Result<CoreGraphName, String> {
    match graph {
        oxirs_tdb::GraphName::DefaultGraph => Ok(CoreGraphName::DefaultGraph),
        oxirs_tdb::GraphName::Named(term) => match term {
            TdbTerm::Iri(iri) => NamedNode::new(iri)
                .map(CoreGraphName::NamedNode)
                .map_err(|e| e.to_string()),
            TdbTerm::BlankNode(id) => BlankNode::new(id)
                .map(CoreGraphName::BlankNode)
                .map_err(|e| e.to_string()),
            TdbTerm::Literal { .. } => {
                Err("a literal cannot be used as an RDF graph name".to_string())
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_tdb2_dataset_detects_data_tdb_file() {
        let unique = std::process::id() as u64 * 1_000_003 + line!() as u64;
        let dir = std::env::temp_dir().join(format!("oxirs-tdb-detect-test-{unique}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");

        assert!(!is_tdb2_dataset(&dir), "empty dir must not look like tdb2");

        std::fs::write(dir.join("data.tdb"), b"").expect("write marker file");
        assert!(
            is_tdb2_dataset(&dir),
            "dir containing data.tdb must be detected as tdb2"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_is_tdb2_dataset_ignores_memory_backend_marker() {
        let unique = std::process::id() as u64 * 1_000_003 + line!() as u64;
        let dir = std::env::temp_dir().join(format!("oxirs-tdb-detect-mem-test-{unique}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");

        std::fs::write(dir.join("data.nq"), b"").expect("write marker file");
        assert!(
            !is_tdb2_dataset(&dir),
            "a memory-backend data.nq file must not be detected as tdb2"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_round_trip_iri_subject_and_object() {
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/s").unwrap());
        let tdb_subject = subject_to_tdb_term(&subject).expect("convert subject");
        assert_eq!(
            tdb_subject,
            TdbTerm::Iri("http://example.org/s".to_string())
        );
        let back = tdb_term_to_subject(&tdb_subject).expect("convert back");
        assert_eq!(back, subject);
    }

    #[test]
    fn test_round_trip_blank_node() {
        let subject = Subject::BlankNode(BlankNode::new("b1").unwrap());
        let tdb_subject = subject_to_tdb_term(&subject).expect("convert subject");
        assert_eq!(tdb_subject, TdbTerm::BlankNode("b1".to_string()));
        let back = tdb_term_to_subject(&tdb_subject).expect("convert back");
        assert_eq!(back, subject);
    }

    #[test]
    fn test_round_trip_language_tagged_literal() {
        let lit = Literal::new_language_tagged_literal("hello", "en").unwrap();
        let object = Object::Literal(lit);
        let tdb_object = object_to_tdb_term(&object).expect("convert object");
        assert_eq!(
            tdb_object,
            TdbTerm::Literal {
                value: "hello".to_string(),
                language: Some("en".to_string()),
                datatype: None,
            }
        );
        let back = tdb_term_to_object(&tdb_object).expect("convert back");
        assert_eq!(back, object);
    }

    #[test]
    fn test_round_trip_typed_literal() {
        let dt = NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap();
        let object = Object::Literal(Literal::new_typed_literal("42", dt));
        let tdb_object = object_to_tdb_term(&object).expect("convert object");
        assert!(
            matches!(tdb_object, TdbTerm::Literal { ref datatype, .. } if datatype.as_deref() == Some("http://www.w3.org/2001/XMLSchema#integer"))
        );
        let back = tdb_term_to_object(&tdb_object).expect("convert back");
        assert_eq!(back, object);
    }

    #[test]
    fn test_variable_subject_is_rejected() {
        use oxirs_core::model::Variable;
        let subject = Subject::Variable(Variable::new("x").unwrap());
        assert!(subject_to_tdb_term(&subject).is_err());
    }

    #[test]
    fn test_graph_name_round_trip_default_and_named() {
        assert_eq!(
            graph_name_to_tdb_term(&CoreGraphName::DefaultGraph).unwrap(),
            None
        );

        let named = CoreGraphName::NamedNode(NamedNode::new("http://example.org/g").unwrap());
        let tdb_graph = graph_name_to_tdb_term(&named).unwrap();
        assert_eq!(
            tdb_graph,
            Some(TdbTerm::Iri("http://example.org/g".to_string()))
        );

        let scan_graph = oxirs_tdb::GraphName::Named(tdb_graph.unwrap());
        let back = tdb_graph_to_core_graph_name(&scan_graph).unwrap();
        assert_eq!(back, named);

        let default_scan_graph = oxirs_tdb::GraphName::DefaultGraph;
        assert_eq!(
            tdb_graph_to_core_graph_name(&default_scan_graph).unwrap(),
            CoreGraphName::DefaultGraph
        );
    }
}
