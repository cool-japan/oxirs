//! SPARQL DESCRIBE query builder.
//!
//! Provides a fluent builder API for constructing SPARQL DESCRIBE queries.
//! Supports IRI targets, variable targets, WHERE patterns, FROM clauses, and LIMIT.

/// Target type for a DESCRIBE query.
///
/// A DESCRIBE query can target explicit IRIs, variables bound in a WHERE clause,
/// or a combination of both.
#[derive(Debug, Clone, PartialEq)]
pub enum DescribeTarget {
    /// A concrete IRI to describe.
    Iri(String),
    /// A variable whose bindings will be described.
    Variable(String),
}

/// Builder for SPARQL DESCRIBE queries.
///
/// Constructs valid SPARQL 1.1 DESCRIBE query strings with optional FROM,
/// WHERE, and LIMIT clauses.
///
/// # Examples
///
/// ```rust
/// use oxirs_core::sparql::describe_builder::DescribeBuilder;
///
/// let query = DescribeBuilder::new()
///     .add_iri("http://example.org/subject")
///     .with_limit(10)
///     .build();
///
/// assert!(query.contains("DESCRIBE"));
/// assert!(query.contains("<http://example.org/subject>"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct DescribeBuilder {
    targets: Vec<DescribeTarget>,
    from_graphs: Vec<String>,
    where_pattern: Option<String>,
    limit: Option<usize>,
}

impl DescribeBuilder {
    /// Create a new empty DESCRIBE builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a concrete IRI as a DESCRIBE target.
    pub fn add_iri(mut self, iri: &str) -> Self {
        self.targets.push(DescribeTarget::Iri(iri.to_string()));
        self
    }

    /// Add a variable as a DESCRIBE target.
    ///
    /// The variable name should be given without the leading `?`.
    pub fn add_variable(mut self, var: &str) -> Self {
        self.targets.push(DescribeTarget::Variable(var.to_string()));
        self
    }

    /// Set the WHERE graph pattern.
    pub fn with_where(mut self, pattern: &str) -> Self {
        self.where_pattern = Some(pattern.to_string());
        self
    }

    /// Add a FROM clause (named default graph).
    pub fn with_from(mut self, graph: &str) -> Self {
        self.from_graphs.push(graph.to_string());
        self
    }

    /// Set a LIMIT on the number of described resources.
    pub fn with_limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Build the SPARQL DESCRIBE query string.
    ///
    /// Returns a complete SPARQL query string. If no targets are given,
    /// produces `DESCRIBE *` (wildcard form).
    pub fn build(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        // DESCRIBE keyword and targets
        if self.targets.is_empty() {
            parts.push("DESCRIBE *".to_string());
        } else {
            let target_strs: Vec<String> = self
                .targets
                .iter()
                .map(|t| match t {
                    DescribeTarget::Iri(iri) => format!("<{}>", iri),
                    DescribeTarget::Variable(var) => format!("?{}", var),
                })
                .collect();
            parts.push(format!("DESCRIBE {}", target_strs.join(" ")));
        }

        // FROM clauses
        for graph in &self.from_graphs {
            parts.push(format!("FROM <{}>", graph));
        }

        // WHERE clause
        if let Some(pattern) = &self.where_pattern {
            parts.push(format!("WHERE {{\n  {}\n}}", pattern));
        }

        // LIMIT clause
        if let Some(limit) = self.limit {
            parts.push(format!("LIMIT {}", limit));
        }

        parts.join("\n")
    }

    /// Return the list of targets configured in this builder.
    pub fn targets(&self) -> &[DescribeTarget] {
        &self.targets
    }

    /// Return the WHERE pattern if one has been set.
    pub fn where_pattern(&self) -> Option<&str> {
        self.where_pattern.as_deref()
    }

    /// Return the FROM graphs configured in this builder.
    pub fn from_graphs(&self) -> &[String] {
        &self.from_graphs
    }

    /// Return the LIMIT value if set.
    pub fn limit(&self) -> Option<usize> {
        self.limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Basic construction ---

    #[test]
    fn test_new_is_default() {
        let b = DescribeBuilder::new();
        assert!(b.targets().is_empty());
        assert!(b.where_pattern().is_none());
        assert!(b.from_graphs().is_empty());
        assert!(b.limit().is_none());
    }

    #[test]
    fn test_empty_target_produces_wildcard() {
        let q = DescribeBuilder::new().build();
        assert!(q.starts_with("DESCRIBE *"));
    }

    #[test]
    fn test_single_iri() {
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .build();
        assert!(q.contains("DESCRIBE"));
        assert!(q.contains("<http://example.org/a>"));
        assert!(!q.contains('*'));
    }

    #[test]
    fn test_multiple_iris() {
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .add_iri("http://example.org/b")
            .build();
        assert!(q.contains("<http://example.org/a>"));
        assert!(q.contains("<http://example.org/b>"));
    }

    #[test]
    fn test_single_variable() {
        let q = DescribeBuilder::new().add_variable("x").build();
        assert!(q.contains("?x"));
        assert!(!q.contains('*'));
    }

    #[test]
    fn test_multiple_variables() {
        let q = DescribeBuilder::new()
            .add_variable("x")
            .add_variable("y")
            .build();
        assert!(q.contains("?x"));
        assert!(q.contains("?y"));
    }

    #[test]
    fn test_iri_and_variable_mixed() {
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .add_variable("x")
            .build();
        assert!(q.contains("<http://example.org/a>"));
        assert!(q.contains("?x"));
    }

    // --- WHERE clause ---

    #[test]
    fn test_with_where_present() {
        let q = DescribeBuilder::new()
            .add_variable("x")
            .with_where("?x a <http://example.org/Type>")
            .build();
        assert!(q.contains("WHERE"));
        assert!(q.contains("?x a <http://example.org/Type>"));
    }

    #[test]
    fn test_without_where_absent() {
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .build();
        assert!(!q.contains("WHERE"));
    }

    #[test]
    fn test_where_overwrites_previous() {
        let b = DescribeBuilder::new()
            .add_variable("x")
            .with_where("?x a <http://example.org/Old>")
            .with_where("?x a <http://example.org/New>");
        let q = b.build();
        assert!(q.contains("<http://example.org/New>"));
        assert!(!q.contains("<http://example.org/Old>"));
    }

    #[test]
    fn test_where_with_multiple_patterns() {
        let pattern = "?x <http://schema.org/name> ?name . ?x <http://schema.org/age> ?age";
        let q = DescribeBuilder::new()
            .add_variable("x")
            .with_where(pattern)
            .build();
        assert!(q.contains("<http://schema.org/name>"));
        assert!(q.contains("<http://schema.org/age>"));
    }

    // --- FROM clause ---

    #[test]
    fn test_single_from() {
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .with_from("http://example.org/graph1")
            .build();
        assert!(q.contains("FROM <http://example.org/graph1>"));
    }

    #[test]
    fn test_multiple_from() {
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .with_from("http://example.org/graph1")
            .with_from("http://example.org/graph2")
            .build();
        assert!(q.contains("FROM <http://example.org/graph1>"));
        assert!(q.contains("FROM <http://example.org/graph2>"));
    }

    #[test]
    fn test_no_from_by_default() {
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .build();
        assert!(!q.contains("FROM"));
    }

    // --- LIMIT clause ---

    #[test]
    fn test_with_limit() {
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .with_limit(50)
            .build();
        assert!(q.contains("LIMIT 50"));
    }

    #[test]
    fn test_limit_zero() {
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .with_limit(0)
            .build();
        assert!(q.contains("LIMIT 0"));
    }

    #[test]
    fn test_limit_overwrite() {
        let b = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .with_limit(10)
            .with_limit(99);
        assert_eq!(b.limit(), Some(99));
        let q = b.build();
        assert!(q.contains("LIMIT 99"));
        assert!(!q.contains("LIMIT 10"));
    }

    #[test]
    fn test_no_limit_by_default() {
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .build();
        assert!(!q.contains("LIMIT"));
    }

    // --- Combinations ---

    #[test]
    fn test_full_query_iri_from_where_limit() {
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/subject")
            .with_from("http://example.org/graph")
            .with_where("?x a <http://example.org/Type>")
            .with_limit(100)
            .build();
        assert!(q.contains("DESCRIBE <http://example.org/subject>"));
        assert!(q.contains("FROM <http://example.org/graph>"));
        assert!(q.contains("WHERE"));
        assert!(q.contains("LIMIT 100"));
    }

    #[test]
    fn test_full_query_variable_from_where_limit() {
        let q = DescribeBuilder::new()
            .add_variable("resource")
            .with_from("http://example.org/graph")
            .with_where("?resource a <http://example.org/Type>")
            .with_limit(20)
            .build();
        assert!(q.contains("DESCRIBE ?resource"));
        assert!(q.contains("FROM <http://example.org/graph>"));
        assert!(q.contains("WHERE"));
        assert!(q.contains("LIMIT 20"));
    }

    #[test]
    fn test_variable_with_where_no_from_no_limit() {
        let q = DescribeBuilder::new()
            .add_variable("s")
            .with_where("?s <http://purl.org/dc/terms/title> ?t")
            .build();
        assert!(q.contains("?s"));
        assert!(q.contains("WHERE"));
        assert!(!q.contains("FROM"));
        assert!(!q.contains("LIMIT"));
    }

    #[test]
    fn test_wildcard_with_where() {
        let q = DescribeBuilder::new()
            .with_where("?x a <http://example.org/T>")
            .build();
        assert!(q.contains("DESCRIBE *"));
        assert!(q.contains("WHERE"));
    }

    #[test]
    fn test_wildcard_with_limit() {
        let q = DescribeBuilder::new().with_limit(5).build();
        assert!(q.contains("DESCRIBE *"));
        assert!(q.contains("LIMIT 5"));
    }

    // --- Accessor methods ---

    #[test]
    fn test_targets_accessor() {
        let b = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .add_variable("x");
        let targets = b.targets();
        assert_eq!(targets.len(), 2);
        assert_eq!(
            targets[0],
            DescribeTarget::Iri("http://example.org/a".to_string())
        );
        assert_eq!(targets[1], DescribeTarget::Variable("x".to_string()));
    }

    #[test]
    fn test_where_pattern_accessor_none() {
        let b = DescribeBuilder::new();
        assert!(b.where_pattern().is_none());
    }

    #[test]
    fn test_where_pattern_accessor_some() {
        let b = DescribeBuilder::new().with_where("?x a <http://example.org/T>");
        assert_eq!(b.where_pattern(), Some("?x a <http://example.org/T>"));
    }

    #[test]
    fn test_from_graphs_accessor() {
        let b = DescribeBuilder::new()
            .with_from("http://example.org/g1")
            .with_from("http://example.org/g2");
        assert_eq!(b.from_graphs().len(), 2);
        assert_eq!(b.from_graphs()[0], "http://example.org/g1");
    }

    #[test]
    fn test_limit_accessor_none() {
        let b = DescribeBuilder::new();
        assert!(b.limit().is_none());
    }

    #[test]
    fn test_limit_accessor_some() {
        let b = DescribeBuilder::new().with_limit(42);
        assert_eq!(b.limit(), Some(42));
    }

    // --- Clone and Debug ---

    #[test]
    fn test_clone_independence() {
        let b1 = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .with_limit(10);
        let b2 = b1.clone().add_iri("http://example.org/b");
        assert_eq!(b1.targets().len(), 1);
        assert_eq!(b2.targets().len(), 2);
    }

    #[test]
    fn test_debug_format() {
        let b = DescribeBuilder::new().add_iri("http://example.org/a");
        let debug = format!("{:?}", b);
        assert!(debug.contains("DescribeBuilder"));
    }

    // --- Query structure ---

    #[test]
    fn test_query_starts_with_describe() {
        let q = DescribeBuilder::new().build();
        assert!(q.starts_with("DESCRIBE"));
    }

    #[test]
    fn test_from_appears_before_where() {
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .with_from("http://example.org/g")
            .with_where("?x a <http://example.org/T>")
            .build();
        let from_pos = q.find("FROM").unwrap_or(usize::MAX);
        let where_pos = q.find("WHERE").unwrap_or(usize::MAX);
        assert!(from_pos < where_pos);
    }

    #[test]
    fn test_limit_appears_after_where() {
        let q = DescribeBuilder::new()
            .add_variable("x")
            .with_where("?x a <http://example.org/T>")
            .with_limit(5)
            .build();
        let where_pos = q.find("WHERE").unwrap_or(usize::MAX);
        let limit_pos = q.find("LIMIT").unwrap_or(usize::MAX);
        assert!(where_pos < limit_pos);
    }

    #[test]
    fn test_iri_targets_wrapped_in_angle_brackets() {
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/resource")
            .build();
        assert!(q.contains("<http://example.org/resource>"));
    }

    #[test]
    fn test_variable_targets_prefixed_with_question_mark() {
        let q = DescribeBuilder::new().add_variable("myVar").build();
        assert!(q.contains("?myVar"));
    }

    #[test]
    fn test_multiple_from_clauses_all_present() {
        let graphs = vec!["http://g1.org/", "http://g2.org/", "http://g3.org/"];
        let mut b = DescribeBuilder::new().add_iri("http://example.org/r");
        for g in &graphs {
            b = b.with_from(g);
        }
        let q = b.build();
        for g in &graphs {
            assert!(q.contains(g));
        }
    }

    #[test]
    fn test_describe_target_iri_equality() {
        let t1 = DescribeTarget::Iri("http://example.org/a".to_string());
        let t2 = DescribeTarget::Iri("http://example.org/a".to_string());
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_describe_target_variable_equality() {
        let t1 = DescribeTarget::Variable("x".to_string());
        let t2 = DescribeTarget::Variable("x".to_string());
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_describe_target_iri_ne_variable() {
        let t1 = DescribeTarget::Iri("x".to_string());
        let t2 = DescribeTarget::Variable("x".to_string());
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_target_ordering_preserved() {
        let b = DescribeBuilder::new()
            .add_iri("http://example.org/first")
            .add_variable("second")
            .add_iri("http://example.org/third");
        let targets = b.targets();
        match &targets[0] {
            DescribeTarget::Iri(s) => assert_eq!(s, "http://example.org/first"),
            _ => panic!("Expected IRI at index 0"),
        }
        match &targets[1] {
            DescribeTarget::Variable(s) => assert_eq!(s, "second"),
            _ => panic!("Expected Variable at index 1"),
        }
    }

    #[test]
    fn test_chaining_is_fluent() {
        // Verify method chaining doesn't require intermediate bindings
        let q = DescribeBuilder::new()
            .add_iri("http://example.org/a")
            .add_variable("x")
            .with_from("http://example.org/g")
            .with_where("?x <http://schema.org/name> ?n")
            .with_limit(10)
            .build();
        assert!(q.contains("DESCRIBE"));
        assert!(q.contains("FROM"));
        assert!(q.contains("WHERE"));
        assert!(q.contains("LIMIT 10"));
    }
}
