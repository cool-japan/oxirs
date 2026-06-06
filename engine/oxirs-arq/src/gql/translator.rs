//! GQL → SPARQL string translator.
//!
//! Translation rules (with default prefix `http://example.org/`):
//!
//! | GQL construct                     | SPARQL output                                      |
//! |-----------------------------------|----------------------------------------------------|
//! | `(x:Person)`                      | `?x a <http://example.org/Person> .`               |
//! | `(x) -[e:knows]-> (y)`            | `?x <http://example.org/knows> ?y .`               |
//! | `(x) <-[e:knows]- (y)`            | `?y <http://example.org/knows> ?x .`               |
//! | `{name: "Alice"}`                 | `?x <http://example.org/name> "Alice" .`           |
//! | `RETURN x, y`                     | `SELECT ?x ?y WHERE { … }`                         |
//! | `WHERE x.age = 30`                | `?x <http://example.org/age> ?x_age .`             |
//! |                                   | `FILTER(?x_age = 30)`                              |

use super::{
    ast::{
        EdgeDirection, EdgePattern, GqlLiteral, GqlPredicate, GqlQuery, NodePattern, PathSegment,
    },
    parser::parse_gql,
    GqlTranslateError,
};

// ─────────────────────────────────────────────────────────────────────────────
// Translator
// ─────────────────────────────────────────────────────────────────────────────

/// Translates GQL MATCH queries into SPARQL SELECT queries.
///
/// # Examples
///
/// ```
/// use oxirs_arq::GqlToSparqlTranslator;
///
/// let t = GqlToSparqlTranslator::new();
/// let sparql = t.translate("MATCH (x:Person) RETURN x").unwrap();
/// assert!(sparql.contains("SELECT ?x"));
/// assert!(sparql.contains("?x a <http://example.org/Person>"));
/// ```
#[derive(Debug, Clone)]
pub struct GqlToSparqlTranslator {
    /// Base IRI prefix appended to labels and property names.
    pub base_prefix: String,
}

impl Default for GqlToSparqlTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl GqlToSparqlTranslator {
    /// Create a translator with the default base prefix `http://example.org/`.
    pub fn new() -> Self {
        Self {
            base_prefix: "http://example.org/".to_string(),
        }
    }

    /// Create a translator with a custom base prefix.
    ///
    /// # Example
    ///
    /// ```
    /// use oxirs_arq::GqlToSparqlTranslator;
    ///
    /// let t = GqlToSparqlTranslator::with_prefix("http://kg.example.com/");
    /// let sparql = t.translate("MATCH (x:Thing) RETURN x").unwrap();
    /// assert!(sparql.contains("http://kg.example.com/Thing"));
    /// ```
    pub fn with_prefix(prefix: &str) -> Self {
        Self {
            base_prefix: prefix.to_string(),
        }
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// Parse `gql` and translate it to a SPARQL SELECT query string.
    pub fn translate(&self, gql: &str) -> Result<String, GqlTranslateError> {
        let query = parse_gql(gql)?;
        if query.return_vars.is_empty() {
            return Err(GqlTranslateError::EmptyReturn);
        }
        Ok(self.translate_ast(&query))
    }

    // ── AST → SPARQL ────────────────────────────────────────────────────────

    /// Translate a fully parsed [`GqlQuery`] into a SPARQL string.
    fn translate_ast(&self, query: &GqlQuery) -> String {
        // Build the SELECT projection.
        let select_vars: String = query
            .return_vars
            .iter()
            .map(|v| format!("?{v}"))
            .collect::<Vec<_>>()
            .join(" ");

        // Collect SPARQL WHERE body triples.
        let mut triples: Vec<String> = Vec::new();
        // Optional FILTER expressions appended after all triples.
        let mut filters: Vec<String> = Vec::new();

        // Walk segments pairwise: Node, (Edge, Node)*
        let segments = &query.match_pattern;
        let mut idx = 0;

        // Resolve the first node's SPARQL variable name.
        let first_var = match segments.first() {
            Some(PathSegment::Node(n)) => {
                let v = self.node_var_name(n, idx);
                for t in self.node_to_sparql(n, &v) {
                    triples.push(t);
                }
                idx += 1;
                v
            }
            _ => {
                // Malformed — produce an empty query (checked before this
                // call, so this is a defensive branch).
                String::new()
            }
        };

        let mut prev_var = first_var;

        // Process alternating Edge / Node pairs.
        while idx + 1 < segments.len() {
            if let (PathSegment::Edge(edge), PathSegment::Node(node)) =
                (&segments[idx], &segments[idx + 1])
            {
                let next_var = self.node_var_name(node, idx + 1);
                let edge_triple = self.edge_to_sparql(&prev_var, edge, &next_var);
                triples.push(edge_triple);
                for t in self.node_to_sparql(node, &next_var) {
                    triples.push(t);
                }
                prev_var = next_var;
                idx += 2;
            } else {
                // Unexpected segment order — stop.
                break;
            }
        }

        // WHERE predicate → auxiliary triple + FILTER.
        if let Some(pred) = &query.where_pred {
            let (aux_triple, filter) = self.predicate_to_sparql(pred);
            triples.push(aux_triple);
            filters.push(filter);
        }

        // Assemble the query.
        let mut body_lines: Vec<String> = triples.iter().map(|t| format!("  {t}")).collect();
        for f in &filters {
            body_lines.push(format!("  {f}"));
        }
        let body = body_lines.join("\n");

        format!("SELECT {select_vars} WHERE {{\n{body}\n}}")
    }

    // ── Node helpers ────────────────────────────────────────────────────────

    /// Return the SPARQL variable name for a node.
    ///
    /// If the node has an explicit variable binding, use it; otherwise
    /// generate `_node{segment_index}`.
    fn node_var_name(&self, node: &NodePattern, segment_idx: usize) -> String {
        node.var
            .clone()
            .unwrap_or_else(|| format!("_node{segment_idx}"))
    }

    /// Produce SPARQL triples for a node pattern.
    ///
    /// - `(x:Person)` → `?x a <…Person> .`
    /// - `{name: "Alice"}` → `?x <…name> "Alice" .`
    pub fn node_to_sparql(&self, node: &NodePattern, var: &str) -> Vec<String> {
        let mut triples = Vec::new();

        if let Some(label) = &node.label {
            let iri = self.iri(label);
            triples.push(format!("?{var} a {iri} ."));
        }

        for (prop, lit) in &node.props {
            let pred_iri = self.iri(prop);
            let lit_str = self.literal_to_sparql(lit);
            triples.push(format!("?{var} {pred_iri} {lit_str} ."));
        }

        triples
    }

    // ── Edge helpers ────────────────────────────────────────────────────────

    /// Produce a single SPARQL triple for a directed edge pattern.
    ///
    /// If the edge has no label, a blank predicate variable is used.
    pub fn edge_to_sparql(&self, prev_var: &str, edge: &EdgePattern, next_var: &str) -> String {
        let pred = if let Some(label) = &edge.label {
            self.iri(label)
        } else {
            // Anonymous predicate: use the edge variable if present, otherwise
            // generate a fresh blank-node-like variable.
            let ev = edge
                .var
                .as_deref()
                .map(|v| format!("?{v}"))
                .unwrap_or_else(|| "?_p".to_string());
            ev
        };

        match edge.direction {
            EdgeDirection::Forward => {
                format!("?{prev_var} {pred} ?{next_var} .")
            }
            EdgeDirection::Backward => {
                // Source is the textually "right" node; reverse the triple.
                format!("?{next_var} {pred} ?{prev_var} .")
            }
        }
    }

    // ── WHERE predicate ─────────────────────────────────────────────────────

    /// Translate a WHERE predicate into an auxiliary triple and a FILTER.
    ///
    /// `x.age = 30` → `(?x <…age> ?x_age .)` + `FILTER(?x_age = 30)`
    fn predicate_to_sparql(&self, pred: &GqlPredicate) -> (String, String) {
        let prop_iri = self.iri(&pred.prop);
        // Auxiliary variable: `var_prop` (e.g. `x_age`).
        let aux_var = format!("{}_{}", pred.var, pred.prop);
        let lit_str = self.literal_to_sparql(&pred.value);

        let triple = format!("?{} {} ?{} .", pred.var, prop_iri, aux_var);
        let filter = format!("FILTER(?{aux_var} = {lit_str})");
        (triple, filter)
    }

    // ── Literal serialisation ────────────────────────────────────────────────

    /// Serialise a [`GqlLiteral`] to its SPARQL representation.
    pub fn literal_to_sparql(&self, lit: &GqlLiteral) -> String {
        match lit {
            GqlLiteral::Str(s) => {
                // Escape inner double-quotes.
                let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
                format!("\"{escaped}\"")
            }
            GqlLiteral::Int(n) => n.to_string(),
            GqlLiteral::Float(f) => {
                // Use Debug to avoid loss of trailing zeros, then clean up.
                format!("{f:?}")
            }
            GqlLiteral::Bool(b) => {
                // Typed boolean per XSD.
                format!("\"{}\"^^xsd:boolean", if *b { "true" } else { "false" })
            }
        }
    }

    // ── IRI helper ───────────────────────────────────────────────────────────

    /// Expand a local name using the configured base prefix.
    fn iri(&self, local: &str) -> String {
        format!("<{}{}>", self.base_prefix, local)
    }
}
