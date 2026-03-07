/// RDF-star annotation graph for metadata on triples.
///
/// Enables arbitrary annotations (key-value pairs) to be attached to
/// base triples, as described by the W3C RDF-star specification.
use std::collections::HashMap;

/// A plain RDF triple (subject, predicate, object as string IRIs or literals).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BaseTriple {
    pub s: String,
    pub p: String,
    pub o: String,
}

impl BaseTriple {
    /// Convenience constructor.
    pub fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            s: s.into(),
            p: p.into(),
            o: o.into(),
        }
    }

    fn key(&self) -> (String, String, String) {
        (self.s.clone(), self.p.clone(), self.o.clone())
    }
}

/// A typed annotation value.
#[derive(Debug, Clone, PartialEq)]
pub enum AnnotationValue {
    /// An IRI reference.
    Iri(String),
    /// A plain string literal.
    Literal(String),
    /// An integer value.
    Integer(i64),
    /// A floating-point value.
    Float(f64),
    /// A boolean value.
    Boolean(bool),
}

/// A single annotation: a property IRI and its value.
#[derive(Debug, Clone)]
pub struct Annotation {
    /// Property IRI (e.g. `ex:confidence`, `prov:generatedAtTime`).
    pub property: String,
    /// The annotation value.
    pub value: AnnotationValue,
}

/// A triple together with all its annotations.
#[derive(Debug, Clone)]
pub struct AnnotatedTriple {
    /// The annotated base triple.
    pub triple: BaseTriple,
    /// All annotations attached to this triple.
    pub annotations: Vec<Annotation>,
}

/// Errors from annotation graph operations.
#[derive(Debug)]
pub enum AnnotationError {
    /// The target triple does not exist in the graph.
    TripleNotFound,
    /// An annotation with this property already exists on the triple.
    DuplicateProperty(String),
}

impl std::fmt::Display for AnnotationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TripleNotFound => write!(f, "Triple not found in annotation graph"),
            Self::DuplicateProperty(p) => write!(f, "Duplicate annotation property: {p}"),
        }
    }
}

impl std::error::Error for AnnotationError {}

/// An in-memory RDF-star annotation graph.
#[derive(Debug, Default)]
pub struct AnnotationGraph {
    /// Ordered list of base triples (insertion order).
    triples: Vec<BaseTriple>,
    /// Annotation index keyed by (s, p, o).
    annotations: HashMap<(String, String, String), Vec<Annotation>>,
}

impl AnnotationGraph {
    /// Create a new empty annotation graph.
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
            annotations: HashMap::new(),
        }
    }

    /// Add a base triple. Returns `true` if the triple is new, `false` if it already existed.
    pub fn add_triple(&mut self, s: &str, p: &str, o: &str) -> bool {
        let triple = BaseTriple::new(s, p, o);
        if self.triples.contains(&triple) {
            return false;
        }
        self.annotations.insert(triple.key(), Vec::new());
        self.triples.push(triple);
        true
    }

    /// Add an annotation to an existing triple.
    ///
    /// Fails with `TripleNotFound` if the triple does not exist.
    /// Fails with `DuplicateProperty` if a annotation with the same property is already present.
    pub fn annotate(
        &mut self,
        triple: &BaseTriple,
        annotation: Annotation,
    ) -> Result<(), AnnotationError> {
        let key = triple.key();
        let list = self
            .annotations
            .get_mut(&key)
            .ok_or(AnnotationError::TripleNotFound)?;
        if list.iter().any(|a| a.property == annotation.property) {
            return Err(AnnotationError::DuplicateProperty(
                annotation.property.clone(),
            ));
        }
        list.push(annotation);
        Ok(())
    }

    /// Return all annotations on a triple.
    pub fn get_annotations(&self, triple: &BaseTriple) -> Vec<&Annotation> {
        self.annotations
            .get(&triple.key())
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Return the annotation for a specific property, if any.
    pub fn get_annotation(&self, triple: &BaseTriple, property: &str) -> Option<&Annotation> {
        self.annotations
            .get(&triple.key())
            .and_then(|list| list.iter().find(|a| a.property == property))
    }

    /// Remove an annotation by property. Returns `true` if removed.
    pub fn remove_annotation(&mut self, triple: &BaseTriple, property: &str) -> bool {
        if let Some(list) = self.annotations.get_mut(&triple.key()) {
            let before = list.len();
            list.retain(|a| a.property != property);
            list.len() < before
        } else {
            false
        }
    }

    /// Find all triples that have an annotation matching the given property and value.
    pub fn find_by_annotation(&self, property: &str, value: &AnnotationValue) -> Vec<&BaseTriple> {
        self.triples
            .iter()
            .filter(|t| {
                self.annotations
                    .get(&t.key())
                    .map(|anns| {
                        anns.iter()
                            .any(|a| a.property == property && &a.value == value)
                    })
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Return all triples together with their annotations.
    pub fn annotated_triples(&self) -> Vec<AnnotatedTriple> {
        self.triples
            .iter()
            .map(|t| {
                let annotations = self.annotations.get(&t.key()).cloned().unwrap_or_default();
                AnnotatedTriple {
                    triple: t.clone(),
                    annotations,
                }
            })
            .collect()
    }

    /// Return the total number of base triples.
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    /// Return the total number of annotations across all triples.
    pub fn annotation_count(&self) -> usize {
        self.annotations.values().map(|v| v.len()).sum()
    }

    /// Return triples that have no annotations.
    pub fn unannotated_triples(&self) -> Vec<&BaseTriple> {
        self.triples
            .iter()
            .filter(|t| {
                self.annotations
                    .get(&t.key())
                    .map(|v| v.is_empty())
                    .unwrap_or(true)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triple() -> BaseTriple {
        BaseTriple::new("http://alice", "http://knows", "http://bob")
    }

    fn iri_annotation(property: &str, iri: &str) -> Annotation {
        Annotation {
            property: property.to_string(),
            value: AnnotationValue::Iri(iri.to_string()),
        }
    }

    fn lit_annotation(property: &str, lit: &str) -> Annotation {
        Annotation {
            property: property.to_string(),
            value: AnnotationValue::Literal(lit.to_string()),
        }
    }

    // --- add_triple ---

    #[test]
    fn test_add_triple_new() {
        let mut ag = AnnotationGraph::new();
        assert!(ag.add_triple("http://s", "http://p", "http://o"));
    }

    #[test]
    fn test_add_triple_duplicate() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        assert!(!ag.add_triple("http://s", "http://p", "http://o"));
    }

    #[test]
    fn test_triple_count_after_adds() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://a", "http://b", "http://c");
        ag.add_triple("http://d", "http://e", "http://f");
        assert_eq!(ag.triple_count(), 2);
    }

    #[test]
    fn test_triple_count_empty() {
        let ag = AnnotationGraph::new();
        assert_eq!(ag.triple_count(), 0);
    }

    // --- annotate ---

    #[test]
    fn test_annotate_success() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        let t = BaseTriple::new("http://s", "http://p", "http://o");
        let result = ag.annotate(&t, iri_annotation("ex:source", "http://source1"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_annotate_triple_not_found() {
        let mut ag = AnnotationGraph::new();
        let t = make_triple();
        let err = ag.annotate(&t, iri_annotation("ex:prop", "http://val"));
        assert!(matches!(err, Err(AnnotationError::TripleNotFound)));
    }

    #[test]
    fn test_annotate_duplicate_property() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        let t = BaseTriple::new("http://s", "http://p", "http://o");
        ag.annotate(&t, iri_annotation("ex:source", "http://source1"))
            .unwrap();
        let err = ag.annotate(&t, iri_annotation("ex:source", "http://source2"));
        assert!(matches!(err, Err(AnnotationError::DuplicateProperty(_))));
    }

    // --- get_annotations ---

    #[test]
    fn test_get_annotations_empty() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        let t = BaseTriple::new("http://s", "http://p", "http://o");
        assert!(ag.get_annotations(&t).is_empty());
    }

    #[test]
    fn test_get_annotations_returns_all() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        let t = BaseTriple::new("http://s", "http://p", "http://o");
        ag.annotate(&t, iri_annotation("ex:a", "http://v1"))
            .unwrap();
        ag.annotate(&t, lit_annotation("ex:b", "hello")).unwrap();
        assert_eq!(ag.get_annotations(&t).len(), 2);
    }

    // --- get_annotation ---

    #[test]
    fn test_get_annotation_specific_property() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        let t = BaseTriple::new("http://s", "http://p", "http://o");
        ag.annotate(&t, lit_annotation("ex:label", "test")).unwrap();
        let ann = ag.get_annotation(&t, "ex:label").unwrap();
        assert_eq!(ann.property, "ex:label");
    }

    #[test]
    fn test_get_annotation_missing_property() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        let t = BaseTriple::new("http://s", "http://p", "http://o");
        assert!(ag.get_annotation(&t, "ex:missing").is_none());
    }

    // --- remove_annotation ---

    #[test]
    fn test_remove_annotation_success() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        let t = BaseTriple::new("http://s", "http://p", "http://o");
        ag.annotate(&t, iri_annotation("ex:src", "http://source"))
            .unwrap();
        assert!(ag.remove_annotation(&t, "ex:src"));
        assert!(ag.get_annotations(&t).is_empty());
    }

    #[test]
    fn test_remove_annotation_absent() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        let t = BaseTriple::new("http://s", "http://p", "http://o");
        assert!(!ag.remove_annotation(&t, "ex:missing"));
    }

    // --- find_by_annotation ---

    #[test]
    fn test_find_by_annotation_iri() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s1", "http://p", "http://o");
        ag.add_triple("http://s2", "http://p", "http://o");
        let t1 = BaseTriple::new("http://s1", "http://p", "http://o");
        ag.annotate(
            &t1,
            Annotation {
                property: "ex:type".to_string(),
                value: AnnotationValue::Iri("ex:Fact".to_string()),
            },
        )
        .unwrap();
        let found = ag.find_by_annotation("ex:type", &AnnotationValue::Iri("ex:Fact".to_string()));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].s, "http://s1");
    }

    #[test]
    fn test_find_by_annotation_no_match() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        let found = ag.find_by_annotation("ex:x", &AnnotationValue::Boolean(true));
        assert!(found.is_empty());
    }

    // --- annotated_triples ---

    #[test]
    fn test_annotated_triples_all_included() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://a", "http://b", "http://c");
        ag.add_triple("http://d", "http://e", "http://f");
        assert_eq!(ag.annotated_triples().len(), 2);
    }

    #[test]
    fn test_annotated_triples_carries_annotations() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        let t = BaseTriple::new("http://s", "http://p", "http://o");
        ag.annotate(&t, lit_annotation("ex:label", "val")).unwrap();
        let at = ag.annotated_triples();
        assert_eq!(at[0].annotations.len(), 1);
    }

    // --- annotation_count ---

    #[test]
    fn test_annotation_count_zero() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        assert_eq!(ag.annotation_count(), 0);
    }

    #[test]
    fn test_annotation_count_multiple() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        let t = BaseTriple::new("http://s", "http://p", "http://o");
        ag.annotate(&t, iri_annotation("ex:a", "http://v1"))
            .unwrap();
        ag.annotate(&t, lit_annotation("ex:b", "hello")).unwrap();
        assert_eq!(ag.annotation_count(), 2);
    }

    // --- unannotated_triples ---

    #[test]
    fn test_unannotated_triples_all() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://a", "http://b", "http://c");
        ag.add_triple("http://d", "http://e", "http://f");
        assert_eq!(ag.unannotated_triples().len(), 2);
    }

    #[test]
    fn test_unannotated_triples_partial() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s1", "http://p", "http://o");
        ag.add_triple("http://s2", "http://p", "http://o");
        let t1 = BaseTriple::new("http://s1", "http://p", "http://o");
        ag.annotate(&t1, lit_annotation("ex:x", "y")).unwrap();
        assert_eq!(ag.unannotated_triples().len(), 1);
    }

    // --- AnnotationValue variants ---

    #[test]
    fn test_annotation_value_iri() {
        let v = AnnotationValue::Iri("http://example.org".to_string());
        assert!(matches!(v, AnnotationValue::Iri(_)));
    }

    #[test]
    fn test_annotation_value_literal() {
        let v = AnnotationValue::Literal("hello".to_string());
        assert!(matches!(v, AnnotationValue::Literal(_)));
    }

    #[test]
    fn test_annotation_value_integer() {
        let v = AnnotationValue::Integer(42);
        assert!(matches!(v, AnnotationValue::Integer(42)));
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_annotation_value_float() {
        let v = AnnotationValue::Float(3.14);
        if let AnnotationValue::Float(f) = v {
            assert!((f - 3.14).abs() < 1e-10);
        } else {
            panic!("Expected Float");
        }
    }

    #[test]
    fn test_annotation_value_boolean() {
        let v = AnnotationValue::Boolean(true);
        assert!(matches!(v, AnnotationValue::Boolean(true)));
    }

    // --- find_by_annotation with integer ---

    #[test]
    fn test_find_by_integer_annotation() {
        let mut ag = AnnotationGraph::new();
        ag.add_triple("http://s", "http://p", "http://o");
        let t = BaseTriple::new("http://s", "http://p", "http://o");
        ag.annotate(
            &t,
            Annotation {
                property: "ex:count".to_string(),
                value: AnnotationValue::Integer(7),
            },
        )
        .unwrap();
        let found = ag.find_by_annotation("ex:count", &AnnotationValue::Integer(7));
        assert_eq!(found.len(), 1);
    }

    // --- error display ---

    #[test]
    fn test_triple_not_found_display() {
        let e = AnnotationError::TripleNotFound;
        assert!(format!("{e}").contains("not found"));
    }

    #[test]
    fn test_duplicate_property_display() {
        let e = AnnotationError::DuplicateProperty("ex:src".to_string());
        assert!(format!("{e}").contains("ex:src"));
    }
}
