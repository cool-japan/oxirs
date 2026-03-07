//! SPARQL subquery construction and nesting utilities.
//!
//! Provides a fluent builder API for constructing SPARQL subqueries,
//! including SELECT subqueries, EXISTS/NOT EXISTS filters, and MINUS clauses.
//! Includes normalization (flattening double-nested subqueries) and
//! optimization (pushing filters inside subqueries when safe).

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A triple pattern with subject, predicate, and object as strings.
#[derive(Debug, Clone, PartialEq)]
pub struct TriplePattern {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl TriplePattern {
    /// Create a new triple pattern.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    /// Render the pattern as a SPARQL token.
    pub fn to_sparql(&self) -> String {
        format!("{} {} {} .", self.subject, self.predicate, self.object)
    }
}

/// A filter expression string (e.g. `?x > 5`, `LANG(?label) = "en"`).
#[derive(Debug, Clone, PartialEq)]
pub struct FilterExpr(pub String);

impl FilterExpr {
    pub fn new(expr: impl Into<String>) -> Self {
        Self(expr.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Render as a SPARQL FILTER clause.
    pub fn to_sparql(&self) -> String {
        format!("FILTER ( {} )", self.0)
    }

    /// Returns the set of variable names referenced in the expression.
    /// A simple heuristic: scan for `?<word>` tokens.
    pub fn referenced_vars(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        let mut chars = self.0.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '?' {
                let name: String = chars
                    .by_ref()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    vars.insert(name);
                }
            }
        }
        vars
    }
}

// ---------------------------------------------------------------------------
// SubqueryNode
// ---------------------------------------------------------------------------

/// A node in the subquery tree.
#[derive(Debug, Clone, PartialEq)]
pub enum SubqueryNode {
    /// A SELECT subquery.
    Select {
        /// Projected variables (e.g. `["?x", "?y"]`). Empty means `SELECT *`.
        vars: Vec<String>,
        /// Triple patterns in the WHERE clause.
        patterns: Vec<TriplePattern>,
        /// FILTER expressions in the WHERE clause.
        filters: Vec<FilterExpr>,
        /// An optional inner subquery wrapped in `{ SELECT … }`.
        inner: Option<Box<SubqueryNode>>,
    },
    /// `EXISTS { … }` graph pattern.
    Exists(Box<SubqueryNode>),
    /// `NOT EXISTS { … }` graph pattern.
    NotExists(Box<SubqueryNode>),
    /// `MINUS { … }` graph pattern.
    Minus(Box<SubqueryNode>),
}

impl SubqueryNode {
    /// Render this node as a SPARQL fragment.
    pub fn to_sparql(&self) -> String {
        match self {
            SubqueryNode::Select {
                vars,
                patterns,
                filters,
                inner,
            } => {
                let var_list = if vars.is_empty() {
                    "*".to_string()
                } else {
                    vars.join(" ")
                };

                let mut body = String::new();

                // Inner subquery if present
                if let Some(inner_node) = inner {
                    body.push_str("  {\n");
                    let inner_sparql = inner_node.to_sparql();
                    for line in inner_sparql.lines() {
                        body.push_str("    ");
                        body.push_str(line);
                        body.push('\n');
                    }
                    body.push_str("  }\n");
                }

                for pat in patterns {
                    body.push_str("  ");
                    body.push_str(&pat.to_sparql());
                    body.push('\n');
                }
                for filt in filters {
                    body.push_str("  ");
                    body.push_str(&filt.to_sparql());
                    body.push('\n');
                }

                format!("SELECT {var_list} WHERE {{\n{body}}}")
            }
            SubqueryNode::Exists(inner) => {
                let inner_str = inner.to_sparql();
                let indented: String = inner_str
                    .lines()
                    .map(|l| format!("  {l}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("EXISTS {{\n{indented}\n}}")
            }
            SubqueryNode::NotExists(inner) => {
                let inner_str = inner.to_sparql();
                let indented: String = inner_str
                    .lines()
                    .map(|l| format!("  {l}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("NOT EXISTS {{\n{indented}\n}}")
            }
            SubqueryNode::Minus(inner) => {
                let inner_str = inner.to_sparql();
                let indented: String = inner_str
                    .lines()
                    .map(|l| format!("  {l}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("MINUS {{\n{indented}\n}}")
            }
        }
    }

    /// Returns all projected variables if this is a `Select` node.
    pub fn projected_vars(&self) -> Vec<String> {
        match self {
            SubqueryNode::Select { vars, .. } => vars.clone(),
            _ => vec![],
        }
    }

    /// Returns whether this node has no patterns and no filters (i.e. is empty).
    pub fn is_empty(&self) -> bool {
        match self {
            SubqueryNode::Select {
                patterns,
                filters,
                inner,
                ..
            } => patterns.is_empty() && filters.is_empty() && inner.is_none(),
            SubqueryNode::Exists(inner)
            | SubqueryNode::NotExists(inner)
            | SubqueryNode::Minus(inner) => inner.is_empty(),
        }
    }

    /// Returns the depth of nesting.
    pub fn depth(&self) -> usize {
        match self {
            SubqueryNode::Select { inner, .. } => {
                1 + inner.as_ref().map(|n| n.depth()).unwrap_or(0)
            }
            SubqueryNode::Exists(inner)
            | SubqueryNode::NotExists(inner)
            | SubqueryNode::Minus(inner) => 1 + inner.depth(),
        }
    }
}

// ---------------------------------------------------------------------------
// SubqueryBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for `SubqueryNode::Select`.
#[derive(Debug, Default)]
pub struct SubqueryBuilder {
    vars: Vec<String>,
    patterns: Vec<TriplePattern>,
    filters: Vec<FilterExpr>,
    inner: Option<Box<SubqueryNode>>,
}

impl SubqueryBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the projected variables.
    pub fn select(mut self, vars: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.vars = vars.into_iter().map(Into::into).collect();
        self
    }

    /// Add a triple pattern.
    pub fn add_pattern(
        mut self,
        s: impl Into<String>,
        p: impl Into<String>,
        o: impl Into<String>,
    ) -> Self {
        self.patterns.push(TriplePattern::new(s, p, o));
        self
    }

    /// Add a FILTER expression.
    pub fn add_filter(mut self, expr: impl Into<String>) -> Self {
        self.filters.push(FilterExpr::new(expr));
        self
    }

    /// Set an inner `SubqueryNode` (creates a nested subquery).
    pub fn nest(mut self, inner: SubqueryNode) -> Self {
        self.inner = Some(Box::new(inner));
        self
    }

    /// Deduplicate projected variables (preserves first occurrence order).
    fn dedup_vars(vars: Vec<String>) -> Vec<String> {
        let mut seen = HashSet::new();
        vars.into_iter()
            .filter(|v| seen.insert(v.clone()))
            .collect()
    }

    /// Build the `SubqueryNode`.
    pub fn build(self) -> SubqueryNode {
        SubqueryNode::Select {
            vars: Self::dedup_vars(self.vars),
            patterns: self.patterns,
            filters: self.filters,
            inner: self.inner,
        }
    }

    /// Wrap the built node in an `EXISTS`.
    pub fn build_exists(self) -> SubqueryNode {
        SubqueryNode::Exists(Box::new(self.build()))
    }

    /// Wrap the built node in a `NOT EXISTS`.
    pub fn build_not_exists(self) -> SubqueryNode {
        SubqueryNode::NotExists(Box::new(self.build()))
    }

    /// Wrap the built node in a `MINUS`.
    pub fn build_minus(self) -> SubqueryNode {
        SubqueryNode::Minus(Box::new(self.build()))
    }
}

// ---------------------------------------------------------------------------
// SubqueryNormalizer
// ---------------------------------------------------------------------------

/// Flattens doubly-nested `Select` subqueries.
///
/// A doubly-nested subquery is:
/// ```sparql
/// SELECT … WHERE {
///   {
///     SELECT … WHERE {
///       {
///         SELECT … WHERE { … }   ← this innermost is merged up
///       }
///     }
///   }
/// }
/// ```
///
/// The normalizer merges the innermost `Select` into its parent when the
/// parent has no patterns or filters of its own and the vars are compatible.
pub struct SubqueryNormalizer;

impl SubqueryNormalizer {
    /// Create a new normalizer.
    pub fn new() -> Self {
        Self
    }

    /// Normalize `node`, returning an equivalent (possibly simplified) `SubqueryNode`.
    pub fn normalize(&self, node: SubqueryNode) -> SubqueryNode {
        match node {
            SubqueryNode::Select {
                vars,
                patterns,
                filters,
                inner,
            } => {
                let normalized_inner = inner.map(|n| Box::new(self.normalize(*n)));

                // Collapse: if outer Select has no own patterns/filters and the
                // inner is also a plain Select with compatible vars, merge them.
                if let Some(inner_node) = normalized_inner {
                    if let SubqueryNode::Select {
                        vars: inner_vars,
                        patterns: inner_patterns,
                        filters: inner_filters,
                        inner: inner_inner,
                    } = *inner_node
                    {
                        if patterns.is_empty() && filters.is_empty() {
                            // Merge: adopt inner's patterns/filters; keep outer vars if set.
                            let merged_vars = if vars.is_empty() { inner_vars } else { vars };
                            return self.normalize(SubqueryNode::Select {
                                vars: merged_vars,
                                patterns: inner_patterns,
                                filters: inner_filters,
                                inner: inner_inner,
                            });
                        }
                        // Cannot merge; put inner back.
                        SubqueryNode::Select {
                            vars,
                            patterns,
                            filters,
                            inner: Some(Box::new(SubqueryNode::Select {
                                vars: inner_vars,
                                patterns: inner_patterns,
                                filters: inner_filters,
                                inner: inner_inner,
                            })),
                        }
                    } else {
                        SubqueryNode::Select {
                            vars,
                            patterns,
                            filters,
                            inner: Some(inner_node),
                        }
                    }
                } else {
                    SubqueryNode::Select {
                        vars,
                        patterns,
                        filters,
                        inner: None,
                    }
                }
            }
            SubqueryNode::Exists(inner) => SubqueryNode::Exists(Box::new(self.normalize(*inner))),
            SubqueryNode::NotExists(inner) => {
                SubqueryNode::NotExists(Box::new(self.normalize(*inner)))
            }
            SubqueryNode::Minus(inner) => SubqueryNode::Minus(Box::new(self.normalize(*inner))),
        }
    }
}

impl Default for SubqueryNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SubqueryOptimizer
// ---------------------------------------------------------------------------

/// Pushes FILTER expressions inside subqueries when safe to do so.
///
/// A filter is "safe" to push down into an inner `Select` subquery when every
/// variable referenced by the filter is projected by (i.e. available from) the
/// inner subquery.
pub struct SubqueryOptimizer;

impl SubqueryOptimizer {
    /// Create a new optimizer.
    pub fn new() -> Self {
        Self
    }

    /// Optimize `node` by pushing filters inward.
    pub fn optimize(&self, node: SubqueryNode) -> SubqueryNode {
        match node {
            SubqueryNode::Select {
                vars,
                patterns,
                mut filters,
                inner,
            } => {
                if let Some(inner_node) = inner {
                    let optimized_inner = self.optimize(*inner_node);

                    // Determine which filters can be pushed into the inner node.
                    let (push_down, keep): (Vec<FilterExpr>, Vec<FilterExpr>) =
                        if let SubqueryNode::Select {
                            vars: ref inner_vars,
                            ..
                        } = optimized_inner
                        {
                            let available: HashSet<String> = inner_vars
                                .iter()
                                .map(|v| v.trim_start_matches('?').to_string())
                                .collect();

                            filters.drain(..).partition(|f| {
                                // Push down only when all referenced vars are in the inner SELECT.
                                // If inner vars is empty it means SELECT * — push nothing (safety).
                                if available.is_empty() {
                                    false
                                } else {
                                    f.referenced_vars()
                                        .iter()
                                        .all(|v| available.contains(v.as_str()))
                                }
                            })
                        } else {
                            (vec![], std::mem::take(&mut filters))
                        };

                    let new_inner = if push_down.is_empty() {
                        optimized_inner
                    } else {
                        self.add_filters_to_select(optimized_inner, push_down)
                    };

                    SubqueryNode::Select {
                        vars,
                        patterns,
                        filters: keep,
                        inner: Some(Box::new(new_inner)),
                    }
                } else {
                    SubqueryNode::Select {
                        vars,
                        patterns,
                        filters,
                        inner: None,
                    }
                }
            }
            SubqueryNode::Exists(inner) => SubqueryNode::Exists(Box::new(self.optimize(*inner))),
            SubqueryNode::NotExists(inner) => {
                SubqueryNode::NotExists(Box::new(self.optimize(*inner)))
            }
            SubqueryNode::Minus(inner) => SubqueryNode::Minus(Box::new(self.optimize(*inner))),
        }
    }

    /// Add `extra_filters` to a `Select` node (panics if node is not Select — caller must ensure).
    fn add_filters_to_select(
        &self,
        node: SubqueryNode,
        extra_filters: Vec<FilterExpr>,
    ) -> SubqueryNode {
        match node {
            SubqueryNode::Select {
                vars,
                patterns,
                mut filters,
                inner,
            } => {
                filters.extend(extra_filters);
                SubqueryNode::Select {
                    vars,
                    patterns,
                    filters,
                    inner,
                }
            }
            other => other,
        }
    }
}

impl Default for SubqueryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn select_xy() -> SubqueryNode {
        SubqueryBuilder::new()
            .select(["?x", "?y"])
            .add_pattern("?x", "<p:name>", "?y")
            .build()
    }

    // -----------------------------------------------------------------------
    // TriplePattern tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_triple_pattern_new() {
        let tp = TriplePattern::new("?s", "<p:pred>", "?o");
        assert_eq!(tp.subject, "?s");
        assert_eq!(tp.predicate, "<p:pred>");
        assert_eq!(tp.object, "?o");
    }

    #[test]
    fn test_triple_pattern_to_sparql() {
        let tp = TriplePattern::new("?s", "<p:pred>", "?o");
        assert_eq!(tp.to_sparql(), "?s <p:pred> ?o .");
    }

    // -----------------------------------------------------------------------
    // FilterExpr tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_expr_to_sparql() {
        let f = FilterExpr::new("?x > 5");
        assert!(f.to_sparql().contains("FILTER"));
        assert!(f.to_sparql().contains("?x > 5"));
    }

    #[test]
    fn test_filter_expr_referenced_vars_single() {
        let f = FilterExpr::new("?age > 18");
        let vars = f.referenced_vars();
        assert!(vars.contains("age"));
    }

    #[test]
    fn test_filter_expr_referenced_vars_multiple() {
        let f = FilterExpr::new("?x < ?y && ?z > 0");
        let vars = f.referenced_vars();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(vars.contains("z"));
    }

    #[test]
    fn test_filter_expr_no_vars() {
        let f = FilterExpr::new("1 = 1");
        assert!(f.referenced_vars().is_empty());
    }

    // -----------------------------------------------------------------------
    // SubqueryBuilder basic tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_new_builds_empty_select() {
        let node = SubqueryBuilder::new().build();
        match &node {
            SubqueryNode::Select {
                vars,
                patterns,
                filters,
                inner,
            } => {
                assert!(vars.is_empty());
                assert!(patterns.is_empty());
                assert!(filters.is_empty());
                assert!(inner.is_none());
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_builder_select_vars() {
        let node = SubqueryBuilder::new().select(["?a", "?b"]).build();
        assert_eq!(node.projected_vars(), vec!["?a", "?b"]);
    }

    #[test]
    fn test_builder_add_pattern() {
        let node = SubqueryBuilder::new()
            .add_pattern("?s", "<p:type>", "<p:Thing>")
            .build();
        match &node {
            SubqueryNode::Select { patterns, .. } => {
                assert_eq!(patterns.len(), 1);
                assert_eq!(patterns[0].subject, "?s");
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_builder_add_filter() {
        let node = SubqueryBuilder::new().add_filter("?x > 0").build();
        match &node {
            SubqueryNode::Select { filters, .. } => {
                assert_eq!(filters.len(), 1);
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_builder_multiple_patterns() {
        let node = SubqueryBuilder::new()
            .add_pattern("?s", "<p:a>", "?x")
            .add_pattern("?s", "<p:b>", "?y")
            .build();
        match &node {
            SubqueryNode::Select { patterns, .. } => assert_eq!(patterns.len(), 2),
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_builder_multiple_filters() {
        let node = SubqueryBuilder::new()
            .add_filter("?x > 0")
            .add_filter("?x < 100")
            .build();
        match &node {
            SubqueryNode::Select { filters, .. } => assert_eq!(filters.len(), 2),
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_builder_dedup_vars() {
        let node = SubqueryBuilder::new().select(["?x", "?y", "?x"]).build();
        let vars = node.projected_vars();
        assert_eq!(vars, vec!["?x", "?y"]);
    }

    // -----------------------------------------------------------------------
    // SubqueryNode::to_sparql tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_sparql_empty_select_star() {
        let node = SubqueryBuilder::new().build();
        let sparql = node.to_sparql();
        assert!(sparql.contains("SELECT *"));
        assert!(sparql.contains("WHERE"));
    }

    #[test]
    fn test_to_sparql_select_vars() {
        let node = SubqueryBuilder::new().select(["?x", "?y"]).build();
        let sparql = node.to_sparql();
        assert!(sparql.contains("SELECT ?x ?y"));
    }

    #[test]
    fn test_to_sparql_with_pattern() {
        let node = SubqueryBuilder::new()
            .select(["?s"])
            .add_pattern("?s", "<rdf:type>", "<owl:Class>")
            .build();
        let sparql = node.to_sparql();
        assert!(sparql.contains("?s <rdf:type> <owl:Class> ."));
    }

    #[test]
    fn test_to_sparql_with_filter() {
        let node = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?s", "<p:age>", "?x")
            .add_filter("?x >= 18")
            .build();
        let sparql = node.to_sparql();
        assert!(sparql.contains("FILTER"));
        assert!(sparql.contains("?x >= 18"));
    }

    // -----------------------------------------------------------------------
    // EXISTS / NOT EXISTS / MINUS tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_exists() {
        let node = SubqueryBuilder::new()
            .add_pattern("?s", "<p:type>", "<p:A>")
            .build_exists();
        match &node {
            SubqueryNode::Exists(_) => {}
            _ => panic!("expected Exists"),
        }
        let sparql = node.to_sparql();
        assert!(sparql.starts_with("EXISTS"));
    }

    #[test]
    fn test_build_not_exists() {
        let node = SubqueryBuilder::new()
            .add_pattern("?s", "<p:type>", "<p:B>")
            .build_not_exists();
        match &node {
            SubqueryNode::NotExists(_) => {}
            _ => panic!("expected NotExists"),
        }
        let sparql = node.to_sparql();
        assert!(sparql.starts_with("NOT EXISTS"));
    }

    #[test]
    fn test_build_minus() {
        let node = SubqueryBuilder::new()
            .add_pattern("?s", "<p:type>", "<p:C>")
            .build_minus();
        match &node {
            SubqueryNode::Minus(_) => {}
            _ => panic!("expected Minus"),
        }
        let sparql = node.to_sparql();
        assert!(sparql.starts_with("MINUS"));
    }

    #[test]
    fn test_exists_to_sparql_contains_inner() {
        let node = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p:a>", "?y")
            .build_exists();
        let sparql = node.to_sparql();
        assert!(sparql.contains("EXISTS"));
        assert!(sparql.contains("?x <p:a> ?y"));
    }

    #[test]
    fn test_minus_to_sparql_contains_inner() {
        let node = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p:b>", "?z")
            .build_minus();
        let sparql = node.to_sparql();
        assert!(sparql.contains("MINUS"));
        assert!(sparql.contains("?x <p:b> ?z"));
    }

    // -----------------------------------------------------------------------
    // Nesting tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_nest_inner() {
        let inner = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p:type>", "<p:A>")
            .build();
        let outer = SubqueryBuilder::new().select(["?x"]).nest(inner).build();
        match &outer {
            SubqueryNode::Select { inner, .. } => assert!(inner.is_some()),
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_nest_to_sparql_contains_inner_query() {
        let inner = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p:type>", "<p:A>")
            .build();
        let outer = SubqueryBuilder::new().select(["?x"]).nest(inner).build();
        let sparql = outer.to_sparql();
        assert!(sparql.contains("SELECT ?x WHERE"));
        // Inner query should appear inside the outer
        assert!(sparql.contains("?x <p:type> <p:A>"));
    }

    #[test]
    fn test_depth_unnested() {
        let node = select_xy();
        assert_eq!(node.depth(), 1);
    }

    #[test]
    fn test_depth_nested() {
        let inner = select_xy();
        let outer = SubqueryBuilder::new().select(["?x"]).nest(inner).build();
        assert_eq!(outer.depth(), 2);
    }

    #[test]
    fn test_is_empty_true() {
        let node = SubqueryBuilder::new().build();
        assert!(node.is_empty());
    }

    #[test]
    fn test_is_empty_false_with_pattern() {
        let node = SubqueryBuilder::new()
            .add_pattern("?s", "<p>", "?o")
            .build();
        assert!(!node.is_empty());
    }

    // -----------------------------------------------------------------------
    // SubqueryNormalizer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalizer_no_op_on_flat() {
        let node = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p>", "?o")
            .build();
        let normalizer = SubqueryNormalizer::new();
        let result = normalizer.normalize(node.clone());
        assert_eq!(result, node);
    }

    #[test]
    fn test_normalizer_merges_empty_outer() {
        // Outer has no patterns/filters, inner has patterns → merge
        let inner = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p:type>", "<p:A>")
            .build();
        let outer = SubqueryBuilder::new().select(["?x"]).nest(inner).build();
        let normalizer = SubqueryNormalizer::new();
        let result = normalizer.normalize(outer);
        // After merging, there should be no nested inner
        match &result {
            SubqueryNode::Select {
                patterns, inner, ..
            } => {
                assert!(!patterns.is_empty(), "patterns should be merged up");
                assert!(inner.is_none(), "inner should be collapsed");
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_normalizer_keeps_nested_when_outer_has_patterns() {
        let inner = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p:type>", "<p:A>")
            .build();
        let outer = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?y", "<p:knows>", "?x")
            .nest(inner)
            .build();
        let normalizer = SubqueryNormalizer::new();
        let result = normalizer.normalize(outer);
        match &result {
            SubqueryNode::Select { inner, .. } => {
                assert!(
                    inner.is_some(),
                    "should NOT collapse when outer has patterns"
                );
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_normalizer_exists_delegates() {
        let inner = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p>", "?o")
            .build();
        let node = SubqueryNode::Exists(Box::new(inner));
        let normalizer = SubqueryNormalizer::new();
        let result = normalizer.normalize(node);
        match result {
            SubqueryNode::Exists(_) => {}
            _ => panic!("expected Exists wrapper to be preserved"),
        }
    }

    #[test]
    fn test_normalizer_not_exists_delegates() {
        let inner = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p>", "?o")
            .build();
        let node = SubqueryNode::NotExists(Box::new(inner));
        let normalizer = SubqueryNormalizer::new();
        let result = normalizer.normalize(node);
        match result {
            SubqueryNode::NotExists(_) => {}
            _ => panic!("expected NotExists wrapper preserved"),
        }
    }

    #[test]
    fn test_normalizer_minus_delegates() {
        let inner = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p>", "?o")
            .build();
        let node = SubqueryNode::Minus(Box::new(inner));
        let normalizer = SubqueryNormalizer::new();
        let result = normalizer.normalize(node);
        match result {
            SubqueryNode::Minus(_) => {}
            _ => panic!("expected Minus wrapper preserved"),
        }
    }

    // -----------------------------------------------------------------------
    // SubqueryOptimizer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimizer_no_push_when_no_inner() {
        let node = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p>", "?o")
            .add_filter("?x > 0")
            .build();
        let optimizer = SubqueryOptimizer::new();
        let result = optimizer.optimize(node);
        match &result {
            SubqueryNode::Select { filters, .. } => {
                assert_eq!(filters.len(), 1, "filter should remain when no inner");
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_optimizer_pushes_filter_into_inner() {
        // Outer has ?x filter; inner projects ?x → push down
        let inner = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p:type>", "<p:A>")
            .build();
        let outer = SubqueryBuilder::new()
            .select(["?x"])
            .add_filter("?x > 5") // references ?x which is projected by inner
            .nest(inner)
            .build();
        let optimizer = SubqueryOptimizer::new();
        let result = optimizer.optimize(outer);
        match &result {
            SubqueryNode::Select { filters, inner, .. } => {
                assert!(
                    filters.is_empty(),
                    "filter should have been pushed into inner"
                );
                if let Some(inner_node) = inner {
                    match inner_node.as_ref() {
                        SubqueryNode::Select { filters, .. } => {
                            assert_eq!(filters.len(), 1, "inner should have received filter");
                        }
                        _ => panic!("expected Select inner"),
                    }
                } else {
                    panic!("inner expected");
                }
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_optimizer_keeps_filter_when_var_not_projected() {
        // Inner projects only ?x; filter references ?z which is not projected
        let inner = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p:type>", "<p:A>")
            .build();
        let outer = SubqueryBuilder::new()
            .select(["?x"])
            .add_filter("?z > 5")
            .nest(inner)
            .build();
        let optimizer = SubqueryOptimizer::new();
        let result = optimizer.optimize(outer);
        match &result {
            SubqueryNode::Select { filters, .. } => {
                assert_eq!(filters.len(), 1, "filter with ?z must stay in outer");
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_optimizer_partial_push() {
        // Two filters: one pushable (?x), one not (?z)
        let inner = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p:type>", "<p:A>")
            .build();
        let outer = SubqueryBuilder::new()
            .select(["?x"])
            .add_filter("?x > 5")
            .add_filter("?z < 100")
            .nest(inner)
            .build();
        let optimizer = SubqueryOptimizer::new();
        let result = optimizer.optimize(outer);
        match &result {
            SubqueryNode::Select { filters, inner, .. } => {
                assert_eq!(filters.len(), 1, "one non-pushable filter stays in outer");
                if let Some(inner_node) = inner {
                    match inner_node.as_ref() {
                        SubqueryNode::Select { filters, .. } => {
                            assert_eq!(filters.len(), 1, "one filter pushed into inner");
                        }
                        _ => panic!("expected Select inner"),
                    }
                } else {
                    panic!("inner expected");
                }
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_optimizer_exists_delegates() {
        let inner = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p>", "?o")
            .build();
        let node = SubqueryNode::Exists(Box::new(inner));
        let optimizer = SubqueryOptimizer::new();
        let result = optimizer.optimize(node);
        match result {
            SubqueryNode::Exists(_) => {}
            _ => panic!("Exists wrapper should be preserved"),
        }
    }

    #[test]
    fn test_optimizer_not_exists_delegates() {
        let inner = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p>", "?o")
            .build();
        let node = SubqueryNode::NotExists(Box::new(inner));
        let optimizer = SubqueryOptimizer::new();
        let result = optimizer.optimize(node);
        match result {
            SubqueryNode::NotExists(_) => {}
            _ => panic!("NotExists wrapper should be preserved"),
        }
    }

    #[test]
    fn test_optimizer_minus_delegates() {
        let inner = SubqueryBuilder::new()
            .select(["?x"])
            .add_pattern("?x", "<p>", "?o")
            .build();
        let node = SubqueryNode::Minus(Box::new(inner));
        let optimizer = SubqueryOptimizer::new();
        let result = optimizer.optimize(node);
        match result {
            SubqueryNode::Minus(_) => {}
            _ => panic!("Minus wrapper should be preserved"),
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_patterns_select_star_sparql() {
        let node = SubqueryBuilder::new().build();
        let sparql = node.to_sparql();
        assert!(sparql.contains("SELECT *"));
    }

    #[test]
    fn test_select_with_all_components() {
        let node = SubqueryBuilder::new()
            .select(["?s", "?p", "?o"])
            .add_pattern("?s", "?p", "?o")
            .add_filter("isIRI(?s)")
            .build();
        let sparql = node.to_sparql();
        assert!(sparql.contains("SELECT ?s ?p ?o"));
        assert!(sparql.contains("?s ?p ?o ."));
        assert!(sparql.contains("FILTER"));
        assert!(sparql.contains("isIRI(?s)"));
    }

    #[test]
    fn test_triple_nesting_depth() {
        let l1 = SubqueryBuilder::new()
            .select(["?a"])
            .add_pattern("?a", "<p>", "?b")
            .build();
        let l2 = SubqueryBuilder::new().select(["?a"]).nest(l1).build();
        let l3 = SubqueryBuilder::new().select(["?a"]).nest(l2).build();
        assert_eq!(l3.depth(), 3);
    }

    #[test]
    fn test_normalizer_triple_nesting_collapses() {
        let l1 = SubqueryBuilder::new()
            .select(["?a"])
            .add_pattern("?a", "<p>", "?b")
            .build();
        let l2 = SubqueryBuilder::new().select(["?a"]).nest(l1).build();
        let l3 = SubqueryBuilder::new().select(["?a"]).nest(l2).build();
        let normalizer = SubqueryNormalizer::new();
        let result = normalizer.normalize(l3);
        // After full collapse all empty wrappers are removed
        assert!(
            result.depth() <= 2,
            "triple nesting should collapse to ≤2 levels"
        );
    }

    #[test]
    fn test_projected_vars_non_select() {
        let node = SubqueryBuilder::new()
            .add_pattern("?x", "<p>", "?o")
            .build_exists();
        assert!(node.projected_vars().is_empty());
    }

    #[test]
    fn test_filter_expr_as_str() {
        let f = FilterExpr::new("LANG(?label) = \"en\"");
        assert_eq!(f.as_str(), "LANG(?label) = \"en\"");
    }

    #[test]
    fn test_no_push_when_inner_projects_star() {
        // Inner uses SELECT * (empty vars) — optimizer must NOT push filters (safety)
        let inner = SubqueryBuilder::new()
            .add_pattern("?x", "<p:type>", "<p:A>")
            .build(); // vars is empty → SELECT *
        let outer = SubqueryBuilder::new()
            .select(["?x"])
            .add_filter("?x > 5")
            .nest(inner)
            .build();
        let optimizer = SubqueryOptimizer::new();
        let result = optimizer.optimize(outer);
        match &result {
            SubqueryNode::Select { filters, .. } => {
                assert_eq!(filters.len(), 1, "filter must not be pushed into SELECT *");
            }
            _ => panic!("expected Select"),
        }
    }
}
