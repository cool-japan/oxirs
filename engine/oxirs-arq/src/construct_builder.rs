/// SPARQL CONSTRUCT query builder.
///
/// Implements template triple instantiation from SPARQL CONSTRUCT queries,
/// including blank node generation, variable binding, deduplication, and
/// CONSTRUCT WHERE shorthand support.
use std::collections::{HashMap, HashSet};

use thiserror::Error;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur during CONSTRUCT processing.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConstructError {
    /// A variable used in the template was not found in the solution.
    #[error("unbound variable in template: ?{name}")]
    UnboundVariable { name: String },

    /// A triple term (subject, predicate, or object) was empty.
    #[error("empty term in template triple at position {position}")]
    EmptyTerm { position: &'static str },

    /// Predicate position contained a blank node (SPARQL prohibits this).
    #[error("blank node in predicate position: {node}")]
    BlankNodeInPredicate { node: String },
}

// ── Core data structures ──────────────────────────────────────────────────────

/// A single RDF term (IRI, literal, blank node, or variable).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RdfTerm {
    /// An IRI reference (angle-bracket form or prefixed name resolved to full IRI).
    Iri(String),
    /// A plain or typed literal.
    Literal {
        value: String,
        datatype: Option<String>,
        lang_tag: Option<String>,
    },
    /// A blank node with a locally-scoped label.
    BlankNode(String),
    /// A SPARQL variable (without the `?` sigil).
    Variable(String),
}

impl RdfTerm {
    /// Returns `true` if this term is a variable.
    pub fn is_variable(&self) -> bool {
        matches!(self, RdfTerm::Variable(_))
    }

    /// Returns `true` if this term is a blank node.
    pub fn is_blank_node(&self) -> bool {
        matches!(self, RdfTerm::BlankNode(_))
    }

    /// Returns the variable name, if this is a variable.
    pub fn variable_name(&self) -> Option<&str> {
        match self {
            RdfTerm::Variable(n) => Some(n.as_str()),
            _ => None,
        }
    }
}

/// A triple pattern in the CONSTRUCT template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateTriple {
    pub subject: RdfTerm,
    pub predicate: RdfTerm,
    pub object: RdfTerm,
}

impl TemplateTriple {
    /// Construct a new template triple.
    pub fn new(subject: RdfTerm, predicate: RdfTerm, object: RdfTerm) -> Self {
        TemplateTriple {
            subject,
            predicate,
            object,
        }
    }
}

/// A concrete (ground) RDF triple produced after binding all variables.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroundTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// A single solution row from the WHERE clause evaluation.
/// Maps variable name → bound value (string serialisation of the term).
#[derive(Debug, Clone, Default)]
pub struct SolutionRow {
    bindings: HashMap<String, String>,
}

impl SolutionRow {
    /// Create an empty solution row.
    pub fn new() -> Self {
        SolutionRow {
            bindings: HashMap::new(),
        }
    }

    /// Bind a variable to a value.
    pub fn bind(&mut self, var: impl Into<String>, value: impl Into<String>) {
        self.bindings.insert(var.into(), value.into());
    }

    /// Look up a variable binding.
    pub fn get(&self, var: &str) -> Option<&str> {
        self.bindings.get(var).map(String::as_str)
    }

    /// Returns `true` if the variable is bound.
    pub fn is_bound(&self, var: &str) -> bool {
        self.bindings.contains_key(var)
    }

    /// Returns all variable names bound in this row.
    pub fn bound_vars(&self) -> impl Iterator<Item = &str> {
        self.bindings.keys().map(String::as_str)
    }
}

/// Statistics about the constructed graph.
#[derive(Debug, Clone, Default)]
pub struct ConstructStats {
    /// Number of solution rows processed.
    pub rows_processed: usize,
    /// Number of raw triples generated before deduplication.
    pub raw_triple_count: usize,
    /// Number of triples skipped because a variable was unbound.
    pub skipped_unbound: usize,
    /// Number of duplicate triples eliminated.
    pub duplicates_eliminated: usize,
    /// Number of fresh blank nodes generated.
    pub blank_nodes_generated: usize,
}

// ── Blank-node allocator ──────────────────────────────────────────────────────

/// Generates unique blank node identifiers.
pub struct BlankNodeAllocator {
    counter: u64,
}

impl BlankNodeAllocator {
    fn new() -> Self {
        BlankNodeAllocator { counter: 0 }
    }

    /// Produce a fresh blank node label scoped to a given solution row.
    ///
    /// The `template_label` is the label used in the CONSTRUCT template; a
    /// distinct identifier is generated for each (row_index, template_label)
    /// pair so that blank nodes are kept separate across solution rows.
    fn fresh(&mut self, row_index: usize, template_label: &str) -> String {
        self.counter += 1;
        format!("_:b{}_{}_r{}", self.counter, template_label, row_index)
    }
}

// ── ConstructBuilder ──────────────────────────────────────────────────────────

/// Builds a CONSTRUCT result graph from a template and a set of solution rows.
pub struct ConstructBuilder {
    template: Vec<TemplateTriple>,
    skip_on_unbound: bool,
}

impl ConstructBuilder {
    /// Create a builder with the given CONSTRUCT template.
    pub fn new(template: Vec<TemplateTriple>) -> Self {
        ConstructBuilder {
            template,
            skip_on_unbound: true,
        }
    }

    /// Create a builder using CONSTRUCT WHERE shorthand.
    ///
    /// In the shorthand form the template is identical to the WHERE basic
    /// graph pattern, expressed as a list of triple patterns.
    pub fn from_where_shorthand(pattern: Vec<TemplateTriple>) -> Self {
        Self::new(pattern)
    }

    /// If `true` (the default), template triples that contain an unbound
    /// variable are silently skipped.  If `false`, an error is returned.
    pub fn skip_unbound(mut self, skip: bool) -> Self {
        self.skip_on_unbound = skip;
        self
    }

    /// Instantiate the CONSTRUCT template for a single solution row.
    ///
    /// Each blank-node label in the template is mapped to a fresh identifier
    /// scoped to `row_index` so that blank nodes from different rows are never
    /// merged.
    pub fn instantiate_row(
        &self,
        row: &SolutionRow,
        row_index: usize,
        alloc: &mut BlankNodeAllocator,
        stats: &mut ConstructStats,
    ) -> Result<Vec<GroundTriple>, ConstructError> {
        // Per-row blank-node mapping (template label → fresh label).
        let mut bnode_map: HashMap<String, String> = HashMap::new();
        let mut triples = Vec::new();

        for tpl in &self.template {
            let s =
                self.resolve_term(&tpl.subject, row, row_index, alloc, &mut bnode_map, stats)?;
            let p =
                self.resolve_term(&tpl.predicate, row, row_index, alloc, &mut bnode_map, stats)?;
            let o = self.resolve_term(&tpl.object, row, row_index, alloc, &mut bnode_map, stats)?;

            match (s, p, o) {
                (Some(s_val), Some(p_val), Some(o_val)) => {
                    // Validate: predicate must not be a blank node.
                    if p_val.starts_with("_:") {
                        return Err(ConstructError::BlankNodeInPredicate { node: p_val });
                    }
                    triples.push(GroundTriple {
                        subject: s_val,
                        predicate: p_val,
                        object: o_val,
                    });
                }
                _ => {
                    // One or more terms resolved to None (unbound variable, skip-mode).
                    stats.skipped_unbound += 1;
                }
            }
        }
        Ok(triples)
    }

    /// Resolve a single `RdfTerm` from the template into a ground string, or
    /// `None` if the term is an unbound variable and `skip_on_unbound` is set.
    fn resolve_term(
        &self,
        term: &RdfTerm,
        row: &SolutionRow,
        row_index: usize,
        alloc: &mut BlankNodeAllocator,
        bnode_map: &mut HashMap<String, String>,
        stats: &mut ConstructStats,
    ) -> Result<Option<String>, ConstructError> {
        match term {
            RdfTerm::Iri(iri) => Ok(Some(format!("<{}>", iri))),
            RdfTerm::Literal {
                value,
                datatype,
                lang_tag,
            } => {
                let serialised = if let Some(dt) = datatype {
                    format!("\"{}\"^^<{}>", value, dt)
                } else if let Some(lang) = lang_tag {
                    format!("\"{}\"@{}", value, lang)
                } else {
                    format!("\"{}\"", value)
                };
                Ok(Some(serialised))
            }
            RdfTerm::BlankNode(label) => {
                let fresh = bnode_map.entry(label.clone()).or_insert_with(|| {
                    stats.blank_nodes_generated += 1;
                    alloc.fresh(row_index, label)
                });
                Ok(Some(fresh.clone()))
            }
            RdfTerm::Variable(name) => {
                if let Some(val) = row.get(name) {
                    Ok(Some(val.to_owned()))
                } else if self.skip_on_unbound {
                    Ok(None)
                } else {
                    Err(ConstructError::UnboundVariable { name: name.clone() })
                }
            }
        }
    }

    /// Build the full constructed graph from all solution rows.
    ///
    /// Returns deduplicated ground triples and accompanying statistics.
    pub fn build(
        &self,
        solutions: &[SolutionRow],
    ) -> Result<(Vec<GroundTriple>, ConstructStats), ConstructError> {
        let mut stats = ConstructStats::default();
        let mut alloc = BlankNodeAllocator::new();
        let mut seen: HashSet<GroundTriple> = HashSet::new();
        let mut result: Vec<GroundTriple> = Vec::new();

        for (row_index, row) in solutions.iter().enumerate() {
            stats.rows_processed += 1;
            let row_triples = self.instantiate_row(row, row_index, &mut alloc, &mut stats)?;
            stats.raw_triple_count += row_triples.len();

            for triple in row_triples {
                if seen.insert(triple.clone()) {
                    result.push(triple);
                } else {
                    stats.duplicates_eliminated += 1;
                }
            }
        }

        Ok((result, stats))
    }

    /// Check whether a variable is present (bound) across *all* solution rows.
    pub fn variable_present_in_all(var: &str, solutions: &[SolutionRow]) -> bool {
        solutions.iter().all(|r| r.is_bound(var))
    }

    /// Check whether a variable is present in *any* solution row.
    pub fn variable_present_in_any(var: &str, solutions: &[SolutionRow]) -> bool {
        solutions.iter().any(|r| r.is_bound(var))
    }

    /// Return the template triples.
    pub fn template(&self) -> &[TemplateTriple] {
        &self.template
    }
}

// ── Convenience constructors for `RdfTerm` ────────────────────────────────────

impl RdfTerm {
    /// Create an IRI term.
    pub fn iri(iri: impl Into<String>) -> Self {
        RdfTerm::Iri(iri.into())
    }

    /// Create a plain string literal.
    pub fn string_literal(value: impl Into<String>) -> Self {
        RdfTerm::Literal {
            value: value.into(),
            datatype: None,
            lang_tag: None,
        }
    }

    /// Create a typed literal.
    pub fn typed_literal(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        RdfTerm::Literal {
            value: value.into(),
            datatype: Some(datatype.into()),
            lang_tag: None,
        }
    }

    /// Create a language-tagged literal.
    pub fn lang_literal(value: impl Into<String>, lang: impl Into<String>) -> Self {
        RdfTerm::Literal {
            value: value.into(),
            datatype: None,
            lang_tag: Some(lang.into()),
        }
    }

    /// Create a boolean typed literal.
    pub fn boolean(v: bool) -> Self {
        RdfTerm::typed_literal(v.to_string(), "http://www.w3.org/2001/XMLSchema#boolean")
    }

    /// Create an integer typed literal.
    pub fn integer(v: i64) -> Self {
        RdfTerm::typed_literal(v.to_string(), "http://www.w3.org/2001/XMLSchema#integer")
    }

    /// Create a floating-point typed literal.
    pub fn double(v: f64) -> Self {
        RdfTerm::typed_literal(v.to_string(), "http://www.w3.org/2001/XMLSchema#double")
    }

    /// Create a variable term.
    pub fn var(name: impl Into<String>) -> Self {
        RdfTerm::Variable(name.into())
    }

    /// Create a blank node term.
    pub fn blank(label: impl Into<String>) -> Self {
        RdfTerm::BlankNode(label.into())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(pairs: &[(&str, &str)]) -> SolutionRow {
        let mut row = SolutionRow::new();
        for (k, v) in pairs {
            row.bind(*k, *v);
        }
        row
    }

    // ── Basic instantiation ───────────────────────────────────────────────────

    #[test]
    fn test_single_iri_triple() {
        let template = vec![TemplateTriple::new(
            RdfTerm::iri("http://example.org/s"),
            RdfTerm::iri("http://example.org/p"),
            RdfTerm::iri("http://example.org/o"),
        )];
        let builder = ConstructBuilder::new(template);
        let row = SolutionRow::new();
        let (triples, stats) = builder.build(&[row]).expect("build");
        assert_eq!(triples.len(), 1);
        assert_eq!(stats.rows_processed, 1);
        assert_eq!(stats.raw_triple_count, 1);
    }

    #[test]
    fn test_variable_binding() {
        let template = vec![TemplateTriple::new(
            RdfTerm::var("s"),
            RdfTerm::iri("http://example.org/type"),
            RdfTerm::var("t"),
        )];
        let builder = ConstructBuilder::new(template);
        let row = make_row(&[
            ("s", "<http://example.org/Alice>"),
            ("t", "<http://example.org/Person>"),
        ]);
        let (triples, _stats) = builder.build(&[row]).expect("build");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, "<http://example.org/Alice>");
        assert_eq!(triples[0].object, "<http://example.org/Person>");
    }

    #[test]
    fn test_unbound_variable_skipped_by_default() {
        let template = vec![TemplateTriple::new(
            RdfTerm::var("s"),
            RdfTerm::iri("http://example.org/p"),
            RdfTerm::var("o"),
        )];
        let builder = ConstructBuilder::new(template);
        // ?o is not bound
        let row = make_row(&[("s", "<http://example.org/Alice>")]);
        let (triples, stats) = builder.build(&[row]).expect("build");
        assert_eq!(triples.len(), 0);
        assert_eq!(stats.skipped_unbound, 1);
    }

    #[test]
    fn test_unbound_variable_error_mode() {
        let template = vec![TemplateTriple::new(
            RdfTerm::var("s"),
            RdfTerm::iri("http://example.org/p"),
            RdfTerm::var("o"),
        )];
        let builder = ConstructBuilder::new(template).skip_unbound(false);
        let row = make_row(&[("s", "<http://example.org/Alice>")]);
        let result = builder.build(&[row]);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ConstructError::UnboundVariable { name } if name == "o")
        );
    }

    // ── Blank node generation ─────────────────────────────────────────────────

    #[test]
    fn test_blank_node_unique_per_row() {
        let template = vec![TemplateTriple::new(
            RdfTerm::blank("b"),
            RdfTerm::iri("http://example.org/value"),
            RdfTerm::var("v"),
        )];
        let builder = ConstructBuilder::new(template);
        let rows = vec![make_row(&[("v", "\"1\"")]), make_row(&[("v", "\"2\"")])];
        let (triples, stats) = builder.build(&rows).expect("build");
        assert_eq!(triples.len(), 2);
        // The two blank nodes must be distinct.
        assert_ne!(triples[0].subject, triples[1].subject);
        assert_eq!(stats.blank_nodes_generated, 2);
    }

    #[test]
    fn test_blank_node_shared_within_row() {
        // Two triples in the same template both reference blank node `b`.
        // Within the same row they should resolve to the same label.
        let template = vec![
            TemplateTriple::new(
                RdfTerm::blank("b"),
                RdfTerm::iri("http://example.org/type"),
                RdfTerm::iri("http://example.org/Thing"),
            ),
            TemplateTriple::new(
                RdfTerm::blank("b"),
                RdfTerm::iri("http://example.org/name"),
                RdfTerm::string_literal("test"),
            ),
        ];
        let builder = ConstructBuilder::new(template);
        let row = SolutionRow::new();
        let (triples, stats) = builder.build(&[row]).expect("build");
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0].subject, triples[1].subject);
        // Only one blank node was allocated (shared within the row).
        assert_eq!(stats.blank_nodes_generated, 1);
    }

    // ── Deduplication ─────────────────────────────────────────────────────────

    #[test]
    fn test_duplicate_triple_elimination() {
        let template = vec![TemplateTriple::new(
            RdfTerm::iri("http://example.org/s"),
            RdfTerm::iri("http://example.org/p"),
            RdfTerm::iri("http://example.org/o"),
        )];
        let builder = ConstructBuilder::new(template);
        // Same IRI triple produced by three solution rows → only one in output.
        let rows = vec![SolutionRow::new(), SolutionRow::new(), SolutionRow::new()];
        let (triples, stats) = builder.build(&rows).expect("build");
        assert_eq!(triples.len(), 1);
        assert_eq!(stats.duplicates_eliminated, 2);
        assert_eq!(stats.raw_triple_count, 3);
    }

    #[test]
    fn test_no_duplicates_when_all_distinct() {
        let template = vec![TemplateTriple::new(
            RdfTerm::var("s"),
            RdfTerm::iri("http://example.org/p"),
            RdfTerm::iri("http://example.org/o"),
        )];
        let builder = ConstructBuilder::new(template);
        let rows = vec![
            make_row(&[("s", "<http://example.org/A>")]),
            make_row(&[("s", "<http://example.org/B>")]),
        ];
        let (triples, stats) = builder.build(&rows).expect("build");
        assert_eq!(triples.len(), 2);
        assert_eq!(stats.duplicates_eliminated, 0);
    }

    // ── CONSTRUCT WHERE shorthand ──────────────────────────────────────────────

    #[test]
    fn test_construct_where_shorthand() {
        let pattern = vec![TemplateTriple::new(
            RdfTerm::var("x"),
            RdfTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            RdfTerm::var("t"),
        )];
        let builder = ConstructBuilder::from_where_shorthand(pattern.clone());
        assert_eq!(builder.template().len(), pattern.len());
    }

    // ── Literal binding ───────────────────────────────────────────────────────

    #[test]
    fn test_string_literal_binding() {
        let template = vec![TemplateTriple::new(
            RdfTerm::iri("http://example.org/s"),
            RdfTerm::iri("http://example.org/name"),
            RdfTerm::string_literal("Alice"),
        )];
        let builder = ConstructBuilder::new(template);
        let (triples, _stats) = builder.build(&[SolutionRow::new()]).expect("build");
        assert_eq!(triples[0].object, "\"Alice\"");
    }

    #[test]
    fn test_integer_literal_binding() {
        let template = vec![TemplateTriple::new(
            RdfTerm::iri("http://example.org/s"),
            RdfTerm::iri("http://example.org/age"),
            RdfTerm::integer(42),
        )];
        let builder = ConstructBuilder::new(template);
        let (triples, _stats) = builder.build(&[SolutionRow::new()]).expect("build");
        assert!(triples[0].object.contains("42"));
        assert!(triples[0].object.contains("integer"));
    }

    #[test]
    fn test_boolean_literal_binding() {
        let term = RdfTerm::boolean(true);
        if let RdfTerm::Literal {
            value, datatype, ..
        } = &term
        {
            assert_eq!(value, "true");
            assert!(datatype.as_deref().unwrap_or("").contains("boolean"));
        } else {
            panic!("expected Literal");
        }
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_double_literal_binding() {
        let term = RdfTerm::double(3.14);
        if let RdfTerm::Literal {
            value, datatype, ..
        } = &term
        {
            assert!(value.contains("3.14"));
            assert!(datatype.as_deref().unwrap_or("").contains("double"));
        } else {
            panic!("expected Literal");
        }
    }

    #[test]
    fn test_lang_tagged_literal() {
        let template = vec![TemplateTriple::new(
            RdfTerm::iri("http://example.org/s"),
            RdfTerm::iri("http://example.org/label"),
            RdfTerm::lang_literal("Hallo", "de"),
        )];
        let builder = ConstructBuilder::new(template);
        let (triples, _) = builder.build(&[SolutionRow::new()]).expect("build");
        assert_eq!(triples[0].object, "\"Hallo\"@de");
    }

    // ── Variable presence checking ────────────────────────────────────────────

    #[test]
    fn test_variable_present_in_all() {
        let rows = vec![
            make_row(&[("x", "1"), ("y", "2")]),
            make_row(&[("x", "3"), ("y", "4")]),
        ];
        assert!(ConstructBuilder::variable_present_in_all("x", &rows));
        assert!(ConstructBuilder::variable_present_in_all("y", &rows));
        assert!(!ConstructBuilder::variable_present_in_all("z", &rows));
    }

    #[test]
    fn test_variable_present_in_any() {
        let rows = vec![make_row(&[("x", "1")]), make_row(&[("y", "2")])];
        assert!(ConstructBuilder::variable_present_in_any("x", &rows));
        assert!(ConstructBuilder::variable_present_in_any("y", &rows));
        assert!(!ConstructBuilder::variable_present_in_any("z", &rows));
    }

    #[test]
    fn test_variable_present_partial_binding() {
        let rows = vec![
            make_row(&[("x", "1"), ("y", "2")]),
            make_row(&[("x", "3")]), // y missing
        ];
        // y is not present in all rows
        assert!(!ConstructBuilder::variable_present_in_all("y", &rows));
        // y is present in at least one row
        assert!(ConstructBuilder::variable_present_in_any("y", &rows));
    }

    // ── Graph statistics ──────────────────────────────────────────────────────

    #[test]
    fn test_construct_stats_populated() {
        let template = vec![
            TemplateTriple::new(
                RdfTerm::var("s"),
                RdfTerm::iri("http://example.org/p"),
                RdfTerm::var("o"),
            ),
            TemplateTriple::new(
                RdfTerm::var("s"),
                RdfTerm::iri("http://example.org/q"),
                RdfTerm::var("missing"),
            ),
        ];
        let builder = ConstructBuilder::new(template);
        let rows = vec![
            make_row(&[
                ("s", "<http://example.org/A>"),
                ("o", "<http://example.org/B>"),
            ]),
            make_row(&[
                ("s", "<http://example.org/A>"),
                ("o", "<http://example.org/B>"),
            ]),
        ];
        let (triples, stats) = builder.build(&rows).expect("build");
        assert_eq!(stats.rows_processed, 2);
        // One triple per row (second triple skipped due to ?missing unbound).
        assert_eq!(stats.raw_triple_count, 2);
        assert_eq!(stats.skipped_unbound, 2);
        // Both rows produce the same triple → one eliminated.
        assert_eq!(triples.len(), 1);
        assert_eq!(stats.duplicates_eliminated, 1);
    }

    #[test]
    fn test_empty_solution_set() {
        let template = vec![TemplateTriple::new(
            RdfTerm::iri("http://example.org/s"),
            RdfTerm::iri("http://example.org/p"),
            RdfTerm::iri("http://example.org/o"),
        )];
        let builder = ConstructBuilder::new(template);
        let (triples, stats) = builder.build(&[]).expect("build");
        assert_eq!(triples.len(), 0);
        assert_eq!(stats.rows_processed, 0);
    }

    #[test]
    fn test_blank_node_in_predicate_rejected() {
        let template = vec![TemplateTriple::new(
            RdfTerm::iri("http://example.org/s"),
            RdfTerm::blank("b"),
            RdfTerm::iri("http://example.org/o"),
        )];
        let builder = ConstructBuilder::new(template);
        let result = builder.build(&[SolutionRow::new()]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConstructError::BlankNodeInPredicate { .. }
        ));
    }

    #[test]
    fn test_multiple_variables_multiple_rows() {
        let template = vec![
            TemplateTriple::new(
                RdfTerm::var("person"),
                RdfTerm::iri("http://schema.org/name"),
                RdfTerm::var("name"),
            ),
            TemplateTriple::new(
                RdfTerm::var("person"),
                RdfTerm::iri("http://schema.org/age"),
                RdfTerm::var("age"),
            ),
        ];
        let builder = ConstructBuilder::new(template);
        let rows = vec![
            make_row(&[
                ("person", "<http://example.org/Alice>"),
                ("name", "\"Alice\""),
                ("age", "\"30\"^^<http://www.w3.org/2001/XMLSchema#integer>"),
            ]),
            make_row(&[
                ("person", "<http://example.org/Bob>"),
                ("name", "\"Bob\""),
                ("age", "\"25\"^^<http://www.w3.org/2001/XMLSchema#integer>"),
            ]),
        ];
        let (triples, stats) = builder.build(&rows).expect("build");
        assert_eq!(triples.len(), 4);
        assert_eq!(stats.rows_processed, 2);
        assert_eq!(stats.duplicates_eliminated, 0);
    }

    #[test]
    fn test_typed_literal_serialisation() {
        let term = RdfTerm::typed_literal("2024-01-01", "http://www.w3.org/2001/XMLSchema#date");
        let template = vec![TemplateTriple::new(
            RdfTerm::iri("http://example.org/s"),
            RdfTerm::iri("http://example.org/date"),
            term,
        )];
        let builder = ConstructBuilder::new(template);
        let (triples, _) = builder.build(&[SolutionRow::new()]).expect("build");
        assert_eq!(
            triples[0].object,
            "\"2024-01-01\"^^<http://www.w3.org/2001/XMLSchema#date>"
        );
    }

    #[test]
    fn test_solution_row_is_bound() {
        let mut row = SolutionRow::new();
        row.bind("x", "value");
        assert!(row.is_bound("x"));
        assert!(!row.is_bound("y"));
    }

    #[test]
    fn test_solution_row_bound_vars_iteration() {
        let row = make_row(&[("a", "1"), ("b", "2"), ("c", "3")]);
        let vars: Vec<&str> = row.bound_vars().collect();
        assert_eq!(vars.len(), 3);
    }

    #[test]
    fn test_large_result_set_deduplication() {
        // One template triple, 100 rows all binding same values → 1 unique triple.
        let template = vec![TemplateTriple::new(
            RdfTerm::iri("http://example.org/s"),
            RdfTerm::iri("http://example.org/p"),
            RdfTerm::iri("http://example.org/o"),
        )];
        let builder = ConstructBuilder::new(template);
        let rows: Vec<SolutionRow> = (0..100).map(|_| SolutionRow::new()).collect();
        let (triples, stats) = builder.build(&rows).expect("build");
        assert_eq!(triples.len(), 1);
        assert_eq!(stats.duplicates_eliminated, 99);
    }

    // ── RdfTerm helpers ───────────────────────────────────────────────────────

    #[test]
    fn test_rdf_term_is_variable() {
        assert!(RdfTerm::var("x").is_variable());
        assert!(!RdfTerm::iri("http://example.org/x").is_variable());
    }

    #[test]
    fn test_rdf_term_is_blank_node() {
        assert!(RdfTerm::blank("b0").is_blank_node());
        assert!(!RdfTerm::var("x").is_blank_node());
    }

    #[test]
    fn test_rdf_term_variable_name() {
        assert_eq!(RdfTerm::var("foo").variable_name(), Some("foo"));
        assert_eq!(RdfTerm::iri("http://example.org/").variable_name(), None);
    }

    #[test]
    fn test_iri_term_serialisation() {
        let template = vec![TemplateTriple::new(
            RdfTerm::iri("http://example.org/s"),
            RdfTerm::iri("http://example.org/p"),
            RdfTerm::iri("http://example.org/o"),
        )];
        let builder = ConstructBuilder::new(template);
        let (triples, _) = builder.build(&[SolutionRow::new()]).expect("build");
        // IRI terms should be wrapped in angle brackets.
        assert!(triples[0].subject.starts_with('<'));
        assert!(triples[0].subject.ends_with('>'));
    }

    // ── GroundTriple equality ─────────────────────────────────────────────────

    #[test]
    fn test_ground_triple_equality() {
        let t1 = GroundTriple {
            subject: "s".into(),
            predicate: "p".into(),
            object: "o".into(),
        };
        let t2 = GroundTriple {
            subject: "s".into(),
            predicate: "p".into(),
            object: "o".into(),
        };
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_ground_triple_inequality() {
        let t1 = GroundTriple {
            subject: "s1".into(),
            predicate: "p".into(),
            object: "o".into(),
        };
        let t2 = GroundTriple {
            subject: "s2".into(),
            predicate: "p".into(),
            object: "o".into(),
        };
        assert_ne!(t1, t2);
    }

    // ── BlankNodeAllocator uniqueness ─────────────────────────────────────────

    #[test]
    fn test_blank_node_allocator_monotonic() {
        let mut alloc = BlankNodeAllocator::new();
        let a = alloc.fresh(0, "b");
        let b = alloc.fresh(0, "b");
        // Even with the same row and label, fresh() must generate distinct IDs.
        assert_ne!(a, b);
    }

    #[test]
    fn test_blank_node_allocator_different_rows() {
        let mut alloc = BlankNodeAllocator::new();
        let r0 = alloc.fresh(0, "same");
        let r1 = alloc.fresh(1, "same");
        assert_ne!(r0, r1);
    }

    // ── TemplateTriple ────────────────────────────────────────────────────────

    #[test]
    fn test_template_triple_fields() {
        let t = TemplateTriple::new(
            RdfTerm::iri("http://s"),
            RdfTerm::iri("http://p"),
            RdfTerm::iri("http://o"),
        );
        assert!(matches!(t.subject, RdfTerm::Iri(_)));
        assert!(matches!(t.predicate, RdfTerm::Iri(_)));
        assert!(matches!(t.object, RdfTerm::Iri(_)));
    }

    // ── Stats accumulation ────────────────────────────────────────────────────

    #[test]
    fn test_stats_blank_nodes_counted_across_rows() {
        let template = vec![TemplateTriple::new(
            RdfTerm::blank("b"),
            RdfTerm::iri("http://p"),
            RdfTerm::iri("http://o"),
        )];
        let builder = ConstructBuilder::new(template);
        let rows = vec![SolutionRow::new(), SolutionRow::new(), SolutionRow::new()];
        let (_, stats) = builder.build(&rows).expect("build");
        // One blank node per row.
        assert_eq!(stats.blank_nodes_generated, 3);
    }

    #[test]
    fn test_template_clone() {
        let t = TemplateTriple::new(RdfTerm::var("x"), RdfTerm::iri("p"), RdfTerm::var("y"));
        let t2 = t.clone();
        assert_eq!(t, t2);
    }

    // ── construct_builder builder method ──────────────────────────────────────

    #[test]
    fn test_template_accessor() {
        let template = vec![
            TemplateTriple::new(RdfTerm::var("a"), RdfTerm::iri("b"), RdfTerm::var("c")),
            TemplateTriple::new(RdfTerm::iri("x"), RdfTerm::iri("y"), RdfTerm::iri("z")),
        ];
        let builder = ConstructBuilder::new(template);
        assert_eq!(builder.template().len(), 2);
    }

    #[test]
    fn test_skip_unbound_chained() {
        let builder = ConstructBuilder::new(vec![]).skip_unbound(false);
        // Just verify the builder doesn't panic; an empty template with
        // skip_on_unbound=false on an empty solution set should succeed.
        let (triples, _) = builder.build(&[]).expect("build empty");
        assert!(triples.is_empty());
    }
}
