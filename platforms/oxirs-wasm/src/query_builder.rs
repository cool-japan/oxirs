//! # SPARQL Query Builder
//!
//! A programmatic, fluent API for constructing SPARQL 1.1 SELECT queries.
//! Supports prefixes, DISTINCT, WHERE clause triple patterns, FILTER, OPTIONAL,
//! ORDER BY, GROUP BY, HAVING, LIMIT, OFFSET, and the ASK / COUNT query forms.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_wasm::query_builder::{QueryBuilder, SparqlTerm, OrderDirection};
//!
//! let query = QueryBuilder::new()
//!     .prefix("foaf", "http://xmlns.com/foaf/0.1/")
//!     .select(&["name", "age"])
//!     .where_triple(
//!         SparqlTerm::var("person"),
//!         SparqlTerm::iri("foaf:name"),
//!         SparqlTerm::var("name"),
//!     )
//!     .filter("?age > 18")
//!     .order_by("name", OrderDirection::Asc)
//!     .limit(10)
//!     .build();
//!
//! assert!(query.contains("SELECT"));
//! assert!(query.contains("FILTER"));
//! assert!(query.contains("LIMIT 10"));
//! ```

/// A SPARQL term: IRI, Literal, Variable, or Blank Node
#[derive(Debug, Clone, PartialEq)]
pub enum SparqlTerm {
    /// An absolute or prefixed IRI: `<http://...>` or `prefix:local`
    Iri(String),
    /// A literal value with optional datatype or language tag
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    /// A SPARQL variable: `?name`
    Var(String),
    /// A blank node: `_:label`
    BNode(String),
}

impl SparqlTerm {
    /// Create an IRI term.  Absolute IRIs (starting with `http`) are wrapped in `<>`;
    /// prefixed names are passed through as-is.
    pub fn iri(iri: impl Into<String>) -> Self {
        SparqlTerm::Iri(iri.into())
    }

    /// Create a variable term (the `?` prefix is added automatically if not present)
    pub fn var(name: impl Into<String>) -> Self {
        SparqlTerm::Var(name.into())
    }

    /// Create a plain string literal: `"value"`
    pub fn literal(value: impl Into<String>) -> Self {
        SparqlTerm::Literal {
            value: value.into(),
            datatype: None,
            lang: None,
        }
    }

    /// Create a language-tagged literal: `"value"@lang`
    pub fn lang_literal(value: impl Into<String>, lang: impl Into<String>) -> Self {
        SparqlTerm::Literal {
            value: value.into(),
            datatype: None,
            lang: Some(lang.into()),
        }
    }

    /// Create a typed literal: `"value"^^<datatype>`
    pub fn typed_literal(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        SparqlTerm::Literal {
            value: value.into(),
            datatype: Some(datatype.into()),
            lang: None,
        }
    }

    /// Serialise the term to a SPARQL string fragment
    pub fn to_sparql_string(&self) -> String {
        match self {
            SparqlTerm::Iri(iri) => {
                // If it looks like an absolute IRI wrap it in angle brackets
                if iri.starts_with("http://")
                    || iri.starts_with("https://")
                    || iri.starts_with("urn:")
                    || iri.starts_with("ftp://")
                {
                    format!("<{}>", iri)
                } else {
                    // Prefixed name — pass through
                    iri.clone()
                }
            }
            SparqlTerm::Literal { value, datatype, lang } => {
                let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
                let base = format!("\"{}\"", escaped);
                if let Some(lang_tag) = lang {
                    format!("{}@{}", base, lang_tag)
                } else if let Some(dt) = datatype {
                    if dt.starts_with("http://") || dt.starts_with("https://") {
                        format!("{}^^<{}>", base, dt)
                    } else {
                        format!("{}^^{}", base, dt)
                    }
                } else {
                    base
                }
            }
            SparqlTerm::Var(name) => {
                if name.starts_with('?') {
                    name.clone()
                } else {
                    format!("?{}", name)
                }
            }
            SparqlTerm::BNode(label) => format!("_:{}", label),
        }
    }
}

/// Sort direction for ORDER BY clauses
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderDirection {
    Asc,
    Desc,
}

/// A triple pattern in the WHERE clause
#[derive(Debug, Clone)]
pub struct TriplePattern {
    pub s: SparqlTerm,
    pub p: SparqlTerm,
    pub o: SparqlTerm,
}

impl TriplePattern {
    fn to_sparql_string(&self) -> String {
        format!(
            "  {} {} {} .",
            self.s.to_sparql_string(),
            self.p.to_sparql_string(),
            self.o.to_sparql_string()
        )
    }
}

/// Fluent SPARQL SELECT query builder
#[derive(Debug, Clone)]
pub struct QueryBuilder {
    select_vars: Vec<String>,
    distinct: bool,
    where_patterns: Vec<TriplePattern>,
    filters: Vec<String>,
    optional_patterns: Vec<Vec<TriplePattern>>,
    order_by: Vec<(String, OrderDirection)>,
    group_by: Vec<String>,
    having: Vec<String>,
    limit: Option<usize>,
    offset: Option<usize>,
    prefixes: Vec<(String, String)>,
}

impl QueryBuilder {
    /// Create a new, empty query builder
    pub fn new() -> Self {
        QueryBuilder {
            select_vars: Vec::new(),
            distinct: false,
            where_patterns: Vec::new(),
            filters: Vec::new(),
            optional_patterns: Vec::new(),
            order_by: Vec::new(),
            group_by: Vec::new(),
            having: Vec::new(),
            limit: None,
            offset: None,
            prefixes: Vec::new(),
        }
    }

    /// Declare a namespace prefix
    pub fn prefix(mut self, prefix: &str, iri: &str) -> Self {
        self.prefixes.push((prefix.to_string(), iri.to_string()));
        self
    }

    /// Specify the variables to select (generates `SELECT ?var1 ?var2 ...`)
    pub fn select(mut self, vars: &[&str]) -> Self {
        self.select_vars = vars.iter().map(|v| v.to_string()).collect();
        self
    }

    /// Equivalent to `SELECT *`
    pub fn select_star(mut self) -> Self {
        self.select_vars.clear();
        self
    }

    /// Add the DISTINCT keyword to the SELECT clause
    pub fn distinct(mut self) -> Self {
        self.distinct = true;
        self
    }

    /// Add a triple pattern to the WHERE clause
    pub fn where_triple(mut self, s: SparqlTerm, p: SparqlTerm, o: SparqlTerm) -> Self {
        self.where_patterns.push(TriplePattern { s, p, o });
        self
    }

    /// Add a FILTER expression (the `FILTER(...)` wrapper is added automatically)
    pub fn filter(mut self, expr: impl Into<String>) -> Self {
        self.filters.push(expr.into());
        self
    }

    /// Add an OPTIONAL { ... } block
    pub fn optional(mut self, patterns: Vec<TriplePattern>) -> Self {
        if !patterns.is_empty() {
            self.optional_patterns.push(patterns);
        }
        self
    }

    /// Add an ORDER BY clause
    pub fn order_by(mut self, var: &str, dir: OrderDirection) -> Self {
        self.order_by.push((var.to_string(), dir));
        self
    }

    /// Add a GROUP BY variable
    pub fn group_by(mut self, var: &str) -> Self {
        self.group_by.push(var.to_string());
        self
    }

    /// Add a HAVING expression
    pub fn having(mut self, expr: impl Into<String>) -> Self {
        self.having.push(expr.into());
        self
    }

    /// Set the LIMIT
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Set the OFFSET
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }

    // ===== Rendering helpers =====

    fn render_prefixes(&self) -> String {
        self.prefixes
            .iter()
            .map(|(p, iri)| format!("PREFIX {}: <{}>", p, iri))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn render_select_clause(&self, projection: &str) -> String {
        let distinct_kw = if self.distinct { "DISTINCT " } else { "" };
        format!("SELECT {}{}", distinct_kw, projection)
    }

    fn render_where_body(&self) -> String {
        let mut lines: Vec<String> = Vec::new();

        for tp in &self.where_patterns {
            lines.push(tp.to_sparql_string());
        }

        for filter in &self.filters {
            lines.push(format!("  FILTER({})", filter));
        }

        for opt_group in &self.optional_patterns {
            let inner: Vec<String> = opt_group.iter().map(|tp| tp.to_sparql_string()).collect();
            lines.push(format!("  OPTIONAL {{\n{}\n  }}", inner.join("\n")));
        }

        lines.join("\n")
    }

    fn render_tail(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        if !self.group_by.is_empty() {
            let vars: Vec<String> = self
                .group_by
                .iter()
                .map(|v| {
                    if v.starts_with('?') {
                        v.clone()
                    } else {
                        format!("?{}", v)
                    }
                })
                .collect();
            parts.push(format!("GROUP BY {}", vars.join(" ")));
        }

        if !self.having.is_empty() {
            let exprs: Vec<String> = self
                .having
                .iter()
                .map(|e| format!("HAVING({})", e))
                .collect();
            parts.push(exprs.join(" "));
        }

        if !self.order_by.is_empty() {
            let clauses: Vec<String> = self
                .order_by
                .iter()
                .map(|(var, dir)| {
                    let v = if var.starts_with('?') {
                        var.clone()
                    } else {
                        format!("?{}", var)
                    };
                    match dir {
                        OrderDirection::Asc => format!("ASC({})", v),
                        OrderDirection::Desc => format!("DESC({})", v),
                    }
                })
                .collect();
            parts.push(format!("ORDER BY {}", clauses.join(" ")));
        }

        if let Some(limit) = self.limit {
            parts.push(format!("LIMIT {}", limit));
        }

        if let Some(offset) = self.offset {
            parts.push(format!("OFFSET {}", offset));
        }

        parts.join("\n")
    }

    fn var_list_projection(&self) -> String {
        if self.select_vars.is_empty() {
            return "*".to_string();
        }
        self.select_vars
            .iter()
            .map(|v| {
                if v.starts_with('?') {
                    v.clone()
                } else {
                    format!("?{}", v)
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Build a SPARQL SELECT query string
    pub fn build(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        let prefix_block = self.render_prefixes();
        if !prefix_block.is_empty() {
            parts.push(prefix_block);
        }

        parts.push(self.render_select_clause(&self.var_list_projection()));
        parts.push(format!("WHERE {{\n{}\n}}", self.render_where_body()));

        let tail = self.render_tail();
        if !tail.is_empty() {
            parts.push(tail);
        }

        parts.join("\n")
    }

    /// Build a SPARQL ASK query
    pub fn build_ask(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        let prefix_block = self.render_prefixes();
        if !prefix_block.is_empty() {
            parts.push(prefix_block);
        }

        parts.push("ASK".to_string());
        parts.push(format!("WHERE {{\n{}\n}}", self.render_where_body()));
        parts.join("\n")
    }

    /// Build a `SELECT (COUNT(?var) AS ?count)` query
    pub fn build_count(&self, var: &str) -> String {
        let var_name = if var.starts_with('?') {
            var.to_string()
        } else {
            format!("?{}", var)
        };

        let mut parts: Vec<String> = Vec::new();

        let prefix_block = self.render_prefixes();
        if !prefix_block.is_empty() {
            parts.push(prefix_block);
        }

        let projection = format!("(COUNT({}) AS ?count)", var_name);
        parts.push(self.render_select_clause(&projection));
        parts.push(format!("WHERE {{\n{}\n}}", self.render_where_body()));

        let tail = self.render_tail();
        if !tail.is_empty() {
            parts.push(tail);
        }

        parts.join("\n")
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== SparqlTerm to_sparql_string =====

    #[test]
    fn test_term_iri_absolute_wrapped() {
        let t = SparqlTerm::iri("http://example.org/foo");
        assert_eq!(t.to_sparql_string(), "<http://example.org/foo>");
    }

    #[test]
    fn test_term_iri_prefixed_not_wrapped() {
        let t = SparqlTerm::iri("foaf:name");
        assert_eq!(t.to_sparql_string(), "foaf:name");
    }

    #[test]
    fn test_term_iri_urn_wrapped() {
        let t = SparqlTerm::iri("urn:example:thing");
        assert_eq!(t.to_sparql_string(), "<urn:example:thing>");
    }

    #[test]
    fn test_term_var_adds_question_mark() {
        let t = SparqlTerm::var("name");
        assert_eq!(t.to_sparql_string(), "?name");
    }

    #[test]
    fn test_term_var_already_has_question_mark() {
        let t = SparqlTerm::var("?name");
        assert_eq!(t.to_sparql_string(), "?name");
    }

    #[test]
    fn test_term_literal_plain() {
        let t = SparqlTerm::literal("hello");
        assert_eq!(t.to_sparql_string(), "\"hello\"");
    }

    #[test]
    fn test_term_literal_lang_tagged() {
        let t = SparqlTerm::lang_literal("bonjour", "fr");
        assert_eq!(t.to_sparql_string(), "\"bonjour\"@fr");
    }

    #[test]
    fn test_term_literal_typed_absolute() {
        let t = SparqlTerm::typed_literal("42", "http://www.w3.org/2001/XMLSchema#integer");
        assert!(t.to_sparql_string().contains("\"42\""));
        assert!(t.to_sparql_string().contains("^^<http://www.w3.org/2001/XMLSchema#integer>"));
    }

    #[test]
    fn test_term_literal_typed_prefixed() {
        let t = SparqlTerm::typed_literal("42", "xsd:integer");
        assert_eq!(t.to_sparql_string(), "\"42\"^^xsd:integer");
    }

    #[test]
    fn test_term_bnode() {
        let t = SparqlTerm::BNode("b0".to_string());
        assert_eq!(t.to_sparql_string(), "_:b0");
    }

    // ===== SELECT * =====

    #[test]
    fn test_build_select_star() {
        let q = QueryBuilder::new()
            .select_star()
            .where_triple(
                SparqlTerm::var("s"),
                SparqlTerm::var("p"),
                SparqlTerm::var("o"),
            )
            .build();
        assert!(q.contains("SELECT *"), "Expected SELECT *");
    }

    // ===== SELECT vars =====

    #[test]
    fn test_build_select_vars() {
        let q = QueryBuilder::new()
            .select(&["name", "age"])
            .where_triple(
                SparqlTerm::var("x"),
                SparqlTerm::iri("foaf:name"),
                SparqlTerm::var("name"),
            )
            .build();
        assert!(q.contains("SELECT ?name ?age") || q.contains("?name") && q.contains("?age"));
        assert!(q.contains("SELECT"));
    }

    // ===== DISTINCT =====

    #[test]
    fn test_build_distinct() {
        let q = QueryBuilder::new()
            .select(&["x"])
            .distinct()
            .where_triple(SparqlTerm::var("x"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .build();
        assert!(q.contains("DISTINCT"));
    }

    // ===== WHERE triples =====

    #[test]
    fn test_build_where_triple_appears() {
        let q = QueryBuilder::new()
            .where_triple(
                SparqlTerm::var("person"),
                SparqlTerm::iri("http://xmlns.com/foaf/0.1/name"),
                SparqlTerm::var("name"),
            )
            .build();
        assert!(q.contains("WHERE"));
        assert!(q.contains("?person"));
        assert!(q.contains("?name"));
    }

    #[test]
    fn test_build_multiple_where_triples() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .where_triple(SparqlTerm::var("s"), SparqlTerm::iri("rdf:type"), SparqlTerm::var("type"))
            .build();
        assert!(q.contains("rdf:type"));
    }

    // ===== FILTER =====

    #[test]
    fn test_build_filter() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("x"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .filter("?x > 0")
            .build();
        assert!(q.contains("FILTER"), "Expected FILTER keyword");
        assert!(q.contains("?x > 0"));
    }

    #[test]
    fn test_build_multiple_filters() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("x"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .filter("?x > 0")
            .filter("?x < 100")
            .build();
        assert!(q.contains("?x > 0"));
        assert!(q.contains("?x < 100"));
    }

    // ===== OPTIONAL =====

    #[test]
    fn test_build_optional() {
        let opt = vec![TriplePattern {
            s: SparqlTerm::var("s"),
            p: SparqlTerm::iri("foaf:email"),
            o: SparqlTerm::var("email"),
        }];
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::iri("foaf:name"), SparqlTerm::var("name"))
            .optional(opt)
            .build();
        assert!(q.contains("OPTIONAL"));
        assert!(q.contains("foaf:email"));
    }

    // ===== ORDER BY =====

    #[test]
    fn test_build_order_by_asc() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .order_by("name", OrderDirection::Asc)
            .build();
        assert!(q.contains("ORDER BY"));
        assert!(q.contains("ASC(?name)"));
    }

    #[test]
    fn test_build_order_by_desc() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .order_by("score", OrderDirection::Desc)
            .build();
        assert!(q.contains("DESC(?score)"));
    }

    #[test]
    fn test_build_order_by_multiple() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .order_by("a", OrderDirection::Asc)
            .order_by("b", OrderDirection::Desc)
            .build();
        assert!(q.contains("ASC(?a)"));
        assert!(q.contains("DESC(?b)"));
    }

    // ===== GROUP BY + HAVING =====

    #[test]
    fn test_build_group_by() {
        let q = QueryBuilder::new()
            .select(&["type", "count"])
            .where_triple(SparqlTerm::var("s"), SparqlTerm::iri("rdf:type"), SparqlTerm::var("type"))
            .group_by("type")
            .build();
        assert!(q.contains("GROUP BY ?type"));
    }

    #[test]
    fn test_build_having() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .group_by("s")
            .having("COUNT(?o) > 5")
            .build();
        assert!(q.contains("HAVING(COUNT(?o) > 5)"));
    }

    // ===== LIMIT + OFFSET =====

    #[test]
    fn test_build_limit() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .limit(25)
            .build();
        assert!(q.contains("LIMIT 25"));
    }

    #[test]
    fn test_build_offset() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .offset(100)
            .build();
        assert!(q.contains("OFFSET 100"));
    }

    #[test]
    fn test_build_limit_and_offset() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .limit(10)
            .offset(20)
            .build();
        assert!(q.contains("LIMIT 10"));
        assert!(q.contains("OFFSET 20"));
    }

    // ===== PREFIX declarations =====

    #[test]
    fn test_build_prefix_appears() {
        let q = QueryBuilder::new()
            .prefix("foaf", "http://xmlns.com/foaf/0.1/")
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .build();
        assert!(q.contains("PREFIX foaf: <http://xmlns.com/foaf/0.1/>"));
    }

    #[test]
    fn test_build_multiple_prefixes() {
        let q = QueryBuilder::new()
            .prefix("foaf", "http://xmlns.com/foaf/0.1/")
            .prefix("schema", "http://schema.org/")
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .build();
        assert!(q.contains("PREFIX foaf:"));
        assert!(q.contains("PREFIX schema:"));
    }

    // ===== build_ask =====

    #[test]
    fn test_build_ask_contains_ask() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::iri("rdf:type"), SparqlTerm::var("t"))
            .build_ask();
        assert!(q.contains("ASK"));
        assert!(!q.contains("SELECT"));
    }

    #[test]
    fn test_build_ask_contains_where() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .build_ask();
        assert!(q.contains("WHERE"));
    }

    #[test]
    fn test_build_ask_with_prefix() {
        let q = QueryBuilder::new()
            .prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
            .where_triple(SparqlTerm::var("s"), SparqlTerm::iri("rdf:type"), SparqlTerm::var("t"))
            .build_ask();
        assert!(q.contains("PREFIX rdf:"));
        assert!(q.contains("ASK"));
    }

    // ===== build_count =====

    #[test]
    fn test_build_count_contains_count() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .build_count("s");
        assert!(q.contains("COUNT(?s)"));
        assert!(q.contains("AS ?count"));
    }

    #[test]
    fn test_build_count_contains_select() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .build_count("s");
        assert!(q.contains("SELECT"));
    }

    #[test]
    fn test_build_count_var_with_question_mark() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .build_count("?s");
        // Should not double the ?
        assert!(q.contains("COUNT(?s)"));
        assert!(!q.contains("COUNT(??s)"));
    }

    // ===== Chained builder =====

    #[test]
    fn test_chained_builder() {
        let q = QueryBuilder::new()
            .prefix("foaf", "http://xmlns.com/foaf/0.1/")
            .select(&["name"])
            .where_triple(
                SparqlTerm::var("person"),
                SparqlTerm::iri("foaf:name"),
                SparqlTerm::var("name"),
            )
            .filter("LANG(?name) = 'en'")
            .order_by("name", OrderDirection::Asc)
            .limit(50)
            .offset(0)
            .build();

        assert!(q.contains("PREFIX foaf:"));
        assert!(q.contains("SELECT"));
        assert!(q.contains("WHERE"));
        assert!(q.contains("FILTER"));
        assert!(q.contains("ORDER BY"));
        assert!(q.contains("LIMIT 50"));
    }

    #[test]
    fn test_empty_builder_produces_select_star() {
        let q = QueryBuilder::new().build();
        assert!(q.contains("SELECT *"));
    }

    #[test]
    fn test_default_builder_is_equivalent_to_new() {
        let q = QueryBuilder::default().build();
        assert!(q.contains("SELECT *"));
    }

    // ===== Additional coverage =====

    #[test]
    fn test_term_literal_escapes_quotes() {
        let t = SparqlTerm::literal("say \"hello\"");
        let s = t.to_sparql_string();
        assert!(s.contains("\\\"hello\\\""));
    }

    #[test]
    fn test_build_no_limit_when_not_set() {
        let q = QueryBuilder::new().build();
        assert!(!q.contains("LIMIT"));
    }

    #[test]
    fn test_build_no_offset_when_not_set() {
        let q = QueryBuilder::new().build();
        assert!(!q.contains("OFFSET"));
    }

    #[test]
    fn test_build_where_keyword_present() {
        let q = QueryBuilder::new()
            .where_triple(SparqlTerm::var("s"), SparqlTerm::var("p"), SparqlTerm::var("o"))
            .build();
        assert!(q.contains("WHERE {"));
    }
}
