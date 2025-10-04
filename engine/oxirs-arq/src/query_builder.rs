//! Query Builder API for SPARQL
//!
//! This module provides a fluent API for programmatically constructing SPARQL queries.
//! Inspired by Apache Jena's QueryBuilder framework.
//!
//! # Example
//!
//! ```rust
//! use oxirs_arq::query_builder::SelectBuilder;
//!
//! let query = SelectBuilder::new()
//!     .add_var("?person")
//!     .add_var("?name")
//!     .add_where("?person", "a", "<http://xmlns.com/foaf/0.1/Person>")
//!     .add_where("?person", "<http://xmlns.com/foaf/0.1/name>", "?name")
//!     .add_filter("LANG(?name) = \"en\"")
//!     .add_order_by("?name", true)
//!     .limit(10)
//!     .build();
//! ```

use anyhow::{bail, Result};
use std::collections::HashMap;

/// Fluent builder for SELECT queries
#[derive(Debug, Clone)]
pub struct SelectBuilder {
    distinct: bool,
    reduced: bool,
    variables: Vec<String>,
    where_patterns: Vec<WherePattern>,
    filters: Vec<String>,
    optionals: Vec<OptionalPattern>,
    unions: Vec<Vec<WherePattern>>,
    group_by: Vec<String>,
    having: Vec<String>,
    order_by: Vec<OrderSpec>,
    limit: Option<usize>,
    offset: Option<usize>,
    values: Vec<ValuesClause>,
    binds: Vec<BindClause>,
    prefixes: HashMap<String, String>,
}

impl SelectBuilder {
    /// Create a new SELECT query builder
    pub fn new() -> Self {
        Self {
            distinct: false,
            reduced: false,
            variables: Vec::new(),
            where_patterns: Vec::new(),
            filters: Vec::new(),
            optionals: Vec::new(),
            unions: Vec::new(),
            group_by: Vec::new(),
            having: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
            values: Vec::new(),
            binds: Vec::new(),
            prefixes: HashMap::new(),
        }
    }

    /// Add DISTINCT modifier
    pub fn distinct(mut self) -> Self {
        self.distinct = true;
        self
    }

    /// Add REDUCED modifier
    pub fn reduced(mut self) -> Self {
        self.reduced = true;
        self
    }

    /// Add a projection variable
    pub fn add_var(mut self, var: &str) -> Self {
        self.variables.push(var.to_string());
        self
    }

    /// Add multiple projection variables
    pub fn add_vars(mut self, vars: &[&str]) -> Self {
        for var in vars {
            self.variables.push(var.to_string());
        }
        self
    }

    /// Select all variables (SELECT *)
    pub fn select_all(mut self) -> Self {
        self.variables.clear();
        self.variables.push("*".to_string());
        self
    }

    /// Add a triple pattern to WHERE clause
    pub fn add_where(mut self, subject: &str, predicate: &str, object: &str) -> Self {
        self.where_patterns.push(WherePattern::Triple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        });
        self
    }

    /// Add a property path pattern
    pub fn add_path(mut self, subject: &str, path: &str, object: &str) -> Self {
        self.where_patterns.push(WherePattern::PropertyPath {
            subject: subject.to_string(),
            path: path.to_string(),
            object: object.to_string(),
        });
        self
    }

    /// Add a FILTER clause
    pub fn add_filter(mut self, filter: &str) -> Self {
        self.filters.push(filter.to_string());
        self
    }

    /// Add an OPTIONAL pattern
    pub fn add_optional(mut self, patterns: Vec<(&str, &str, &str)>) -> Self {
        let optional_patterns: Vec<WherePattern> = patterns
            .into_iter()
            .map(|(s, p, o)| WherePattern::Triple {
                subject: s.to_string(),
                predicate: p.to_string(),
                object: o.to_string(),
            })
            .collect();
        self.optionals.push(OptionalPattern {
            patterns: optional_patterns,
        });
        self
    }

    /// Add a UNION pattern
    pub fn add_union(
        mut self,
        left: Vec<(&str, &str, &str)>,
        right: Vec<(&str, &str, &str)>,
    ) -> Self {
        let left_patterns: Vec<WherePattern> = left
            .into_iter()
            .map(|(s, p, o)| WherePattern::Triple {
                subject: s.to_string(),
                predicate: p.to_string(),
                object: o.to_string(),
            })
            .collect();
        let right_patterns: Vec<WherePattern> = right
            .into_iter()
            .map(|(s, p, o)| WherePattern::Triple {
                subject: s.to_string(),
                predicate: p.to_string(),
                object: o.to_string(),
            })
            .collect();
        self.unions.push(left_patterns);
        self.unions.push(right_patterns);
        self
    }

    /// Add GROUP BY variable
    pub fn add_group_by(mut self, var: &str) -> Self {
        self.group_by.push(var.to_string());
        self
    }

    /// Add HAVING clause
    pub fn add_having(mut self, condition: &str) -> Self {
        self.having.push(condition.to_string());
        self
    }

    /// Add ORDER BY clause
    pub fn add_order_by(mut self, var: &str, ascending: bool) -> Self {
        self.order_by.push(OrderSpec {
            variable: var.to_string(),
            ascending,
        });
        self
    }

    /// Set LIMIT
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set OFFSET
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Add VALUES clause
    pub fn add_values(mut self, var: &str, values: Vec<&str>) -> Self {
        self.values.push(ValuesClause {
            variable: var.to_string(),
            values: values.into_iter().map(|v| v.to_string()).collect(),
        });
        self
    }

    /// Add BIND clause
    pub fn add_bind(mut self, expression: &str, var: &str) -> Self {
        self.binds.push(BindClause {
            expression: expression.to_string(),
            variable: var.to_string(),
        });
        self
    }

    /// Add a PREFIX
    pub fn add_prefix(mut self, prefix: &str, uri: &str) -> Self {
        self.prefixes.insert(prefix.to_string(), uri.to_string());
        self
    }

    /// Validate the query structure
    pub fn validate(&self) -> Result<()> {
        if self.variables.is_empty() {
            bail!("SELECT query must have at least one projection variable");
        }

        if self.where_patterns.is_empty() && self.optionals.is_empty() && self.unions.is_empty() {
            bail!("Query must have at least one pattern (WHERE, OPTIONAL, or UNION)");
        }

        if self.distinct && self.reduced {
            bail!("Cannot use both DISTINCT and REDUCED");
        }

        if !self.having.is_empty() && self.group_by.is_empty() {
            bail!("HAVING clause requires GROUP BY");
        }

        Ok(())
    }

    /// Build the SPARQL query string
    pub fn build(self) -> Result<String> {
        self.validate()?;

        let mut query = String::new();

        // Prefixes
        for (prefix, uri) in &self.prefixes {
            query.push_str(&format!("PREFIX {prefix}: <{uri}>\n"));
        }
        if !self.prefixes.is_empty() {
            query.push('\n');
        }

        // SELECT clause
        query.push_str("SELECT ");
        if self.distinct {
            query.push_str("DISTINCT ");
        }
        if self.reduced {
            query.push_str("REDUCED ");
        }
        query.push_str(&self.variables.join(" "));
        query.push_str("\nWHERE {\n");

        // WHERE patterns
        for pattern in &self.where_patterns {
            query.push_str("  ");
            query.push_str(&pattern.to_sparql());
            query.push_str(" .\n");
        }

        // FILTER clauses
        for filter in &self.filters {
            query.push_str(&format!("  FILTER ({filter})\n"));
        }

        // OPTIONAL patterns
        for optional in &self.optionals {
            query.push_str("  OPTIONAL {\n");
            for pattern in &optional.patterns {
                query.push_str("    ");
                query.push_str(&pattern.to_sparql());
                query.push_str(" .\n");
            }
            query.push_str("  }\n");
        }

        // UNION patterns
        if !self.unions.is_empty() {
            query.push_str("  {\n");
            for (i, union_patterns) in self.unions.iter().enumerate() {
                if i > 0 {
                    query.push_str("  } UNION {\n");
                }
                for pattern in union_patterns {
                    query.push_str("    ");
                    query.push_str(&pattern.to_sparql());
                    query.push_str(" .\n");
                }
            }
            query.push_str("  }\n");
        }

        // BIND clauses
        for bind in &self.binds {
            query.push_str(&format!(
                "  BIND ({} AS {})\n",
                bind.expression, bind.variable
            ));
        }

        // VALUES clauses
        for values in &self.values {
            query.push_str(&format!(
                "  VALUES {} {{ {} }}\n",
                values.variable,
                values.values.join(" ")
            ));
        }

        query.push_str("}\n");

        // GROUP BY
        if !self.group_by.is_empty() {
            query.push_str(&format!("GROUP BY {}\n", self.group_by.join(" ")));
        }

        // HAVING
        if !self.having.is_empty() {
            query.push_str(&format!("HAVING ({})\n", self.having.join(" && ")));
        }

        // ORDER BY
        if !self.order_by.is_empty() {
            query.push_str("ORDER BY ");
            let order_specs: Vec<String> = self
                .order_by
                .iter()
                .map(|spec| {
                    if spec.ascending {
                        spec.variable.clone()
                    } else {
                        format!("DESC({})", spec.variable)
                    }
                })
                .collect();
            query.push_str(&order_specs.join(" "));
            query.push('\n');
        }

        // LIMIT
        if let Some(limit) = self.limit {
            query.push_str(&format!("LIMIT {limit}\n"));
        }

        // OFFSET
        if let Some(offset) = self.offset {
            query.push_str(&format!("OFFSET {offset}\n"));
        }

        Ok(query)
    }
}

impl Default for SelectBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// WHERE pattern types
#[derive(Debug, Clone)]
enum WherePattern {
    Triple {
        subject: String,
        predicate: String,
        object: String,
    },
    PropertyPath {
        subject: String,
        path: String,
        object: String,
    },
}

impl WherePattern {
    fn to_sparql(&self) -> String {
        match self {
            Self::Triple {
                subject,
                predicate,
                object,
            } => {
                format!("{subject} {predicate} {object}")
            }
            Self::PropertyPath {
                subject,
                path,
                object,
            } => {
                format!("{subject} {path} {object}")
            }
        }
    }
}

/// OPTIONAL pattern container
#[derive(Debug, Clone)]
struct OptionalPattern {
    patterns: Vec<WherePattern>,
}

/// ORDER BY specification
#[derive(Debug, Clone)]
struct OrderSpec {
    variable: String,
    ascending: bool,
}

/// VALUES clause
#[derive(Debug, Clone)]
struct ValuesClause {
    variable: String,
    values: Vec<String>,
}

/// BIND clause
#[derive(Debug, Clone)]
struct BindClause {
    expression: String,
    variable: String,
}

/// Fluent builder for CONSTRUCT queries
#[derive(Debug, Clone)]
pub struct ConstructBuilder {
    construct_templates: Vec<WherePattern>,
    where_patterns: Vec<WherePattern>,
    filters: Vec<String>,
    limit: Option<usize>,
    offset: Option<usize>,
    prefixes: HashMap<String, String>,
}

impl ConstructBuilder {
    /// Create a new CONSTRUCT query builder
    pub fn new() -> Self {
        Self {
            construct_templates: Vec::new(),
            where_patterns: Vec::new(),
            filters: Vec::new(),
            limit: None,
            offset: None,
            prefixes: HashMap::new(),
        }
    }

    /// Add a template triple to CONSTRUCT clause
    pub fn add_construct(mut self, subject: &str, predicate: &str, object: &str) -> Self {
        self.construct_templates.push(WherePattern::Triple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        });
        self
    }

    /// Add a triple pattern to WHERE clause
    pub fn add_where(mut self, subject: &str, predicate: &str, object: &str) -> Self {
        self.where_patterns.push(WherePattern::Triple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        });
        self
    }

    /// Add a FILTER clause
    pub fn add_filter(mut self, filter: &str) -> Self {
        self.filters.push(filter.to_string());
        self
    }

    /// Set LIMIT
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set OFFSET
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Add a PREFIX
    pub fn add_prefix(mut self, prefix: &str, uri: &str) -> Self {
        self.prefixes.insert(prefix.to_string(), uri.to_string());
        self
    }

    /// Build the SPARQL query string
    pub fn build(self) -> Result<String> {
        if self.construct_templates.is_empty() {
            bail!("CONSTRUCT query must have at least one template triple");
        }

        if self.where_patterns.is_empty() {
            bail!("CONSTRUCT query must have at least one WHERE pattern");
        }

        let mut query = String::new();

        // Prefixes
        for (prefix, uri) in &self.prefixes {
            query.push_str(&format!("PREFIX {prefix}: <{uri}>\n"));
        }
        if !self.prefixes.is_empty() {
            query.push('\n');
        }

        // CONSTRUCT clause
        query.push_str("CONSTRUCT {\n");
        for template in &self.construct_templates {
            query.push_str("  ");
            query.push_str(&template.to_sparql());
            query.push_str(" .\n");
        }
        query.push_str("}\nWHERE {\n");

        // WHERE patterns
        for pattern in &self.where_patterns {
            query.push_str("  ");
            query.push_str(&pattern.to_sparql());
            query.push_str(" .\n");
        }

        // FILTER clauses
        for filter in &self.filters {
            query.push_str(&format!("  FILTER ({filter})\n"));
        }

        query.push_str("}\n");

        // LIMIT
        if let Some(limit) = self.limit {
            query.push_str(&format!("LIMIT {limit}\n"));
        }

        // OFFSET
        if let Some(offset) = self.offset {
            query.push_str(&format!("OFFSET {offset}\n"));
        }

        Ok(query)
    }
}

impl Default for ConstructBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for ASK queries
#[derive(Debug, Clone)]
pub struct AskBuilder {
    where_patterns: Vec<WherePattern>,
    filters: Vec<String>,
    prefixes: HashMap<String, String>,
}

impl AskBuilder {
    /// Create a new ASK query builder
    pub fn new() -> Self {
        Self {
            where_patterns: Vec::new(),
            filters: Vec::new(),
            prefixes: HashMap::new(),
        }
    }

    /// Add a triple pattern to WHERE clause
    pub fn add_where(mut self, subject: &str, predicate: &str, object: &str) -> Self {
        self.where_patterns.push(WherePattern::Triple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        });
        self
    }

    /// Add a FILTER clause
    pub fn add_filter(mut self, filter: &str) -> Self {
        self.filters.push(filter.to_string());
        self
    }

    /// Add a PREFIX
    pub fn add_prefix(mut self, prefix: &str, uri: &str) -> Self {
        self.prefixes.insert(prefix.to_string(), uri.to_string());
        self
    }

    /// Build the SPARQL query string
    pub fn build(self) -> Result<String> {
        if self.where_patterns.is_empty() {
            bail!("ASK query must have at least one WHERE pattern");
        }

        let mut query = String::new();

        // Prefixes
        for (prefix, uri) in &self.prefixes {
            query.push_str(&format!("PREFIX {prefix}: <{uri}>\n"));
        }
        if !self.prefixes.is_empty() {
            query.push('\n');
        }

        // ASK clause
        query.push_str("ASK\nWHERE {\n");

        // WHERE patterns
        for pattern in &self.where_patterns {
            query.push_str("  ");
            query.push_str(&pattern.to_sparql());
            query.push_str(" .\n");
        }

        // FILTER clauses
        for filter in &self.filters {
            query.push_str(&format!("  FILTER ({filter})\n"));
        }

        query.push_str("}\n");

        Ok(query)
    }
}

impl Default for AskBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for DESCRIBE queries
#[derive(Debug, Clone)]
pub struct DescribeBuilder {
    resources: Vec<String>,
    where_patterns: Vec<WherePattern>,
    filters: Vec<String>,
    limit: Option<usize>,
    offset: Option<usize>,
    prefixes: HashMap<String, String>,
}

impl DescribeBuilder {
    /// Create a new DESCRIBE query builder
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            where_patterns: Vec::new(),
            filters: Vec::new(),
            limit: None,
            offset: None,
            prefixes: HashMap::new(),
        }
    }

    /// Add a resource to describe
    pub fn add_resource(mut self, resource: &str) -> Self {
        self.resources.push(resource.to_string());
        self
    }

    /// Add a triple pattern to WHERE clause
    pub fn add_where(mut self, subject: &str, predicate: &str, object: &str) -> Self {
        self.where_patterns.push(WherePattern::Triple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        });
        self
    }

    /// Add a FILTER clause
    pub fn add_filter(mut self, filter: &str) -> Self {
        self.filters.push(filter.to_string());
        self
    }

    /// Set LIMIT
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set OFFSET
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Add a PREFIX
    pub fn add_prefix(mut self, prefix: &str, uri: &str) -> Self {
        self.prefixes.insert(prefix.to_string(), uri.to_string());
        self
    }

    /// Build the SPARQL query string
    pub fn build(self) -> Result<String> {
        if self.resources.is_empty() {
            bail!("DESCRIBE query must have at least one resource");
        }

        let mut query = String::new();

        // Prefixes
        for (prefix, uri) in &self.prefixes {
            query.push_str(&format!("PREFIX {prefix}: <{uri}>\n"));
        }
        if !self.prefixes.is_empty() {
            query.push('\n');
        }

        // DESCRIBE clause
        query.push_str(&format!("DESCRIBE {}\n", self.resources.join(" ")));

        // WHERE clause (optional for DESCRIBE)
        if !self.where_patterns.is_empty() {
            query.push_str("WHERE {\n");

            for pattern in &self.where_patterns {
                query.push_str("  ");
                query.push_str(&pattern.to_sparql());
                query.push_str(" .\n");
            }

            // FILTER clauses
            for filter in &self.filters {
                query.push_str(&format!("  FILTER ({filter})\n"));
            }

            query.push_str("}\n");
        }

        // LIMIT
        if let Some(limit) = self.limit {
            query.push_str(&format!("LIMIT {limit}\n"));
        }

        // OFFSET
        if let Some(offset) = self.offset {
            query.push_str(&format!("OFFSET {offset}\n"));
        }

        Ok(query)
    }
}

impl Default for DescribeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select() {
        let query = SelectBuilder::new()
            .add_var("?s")
            .add_var("?p")
            .add_var("?o")
            .add_where("?s", "?p", "?o")
            .build()
            .unwrap();

        assert!(query.contains("SELECT ?s ?p ?o"));
        assert!(query.contains("WHERE"));
        assert!(query.contains("?s ?p ?o"));
    }

    #[test]
    fn test_select_distinct() {
        let query = SelectBuilder::new()
            .distinct()
            .add_var("?person")
            .add_where("?person", "a", "<http://xmlns.com/foaf/0.1/Person>")
            .build()
            .unwrap();

        assert!(query.contains("SELECT DISTINCT ?person"));
    }

    #[test]
    fn test_select_with_filter() {
        let query = SelectBuilder::new()
            .add_var("?name")
            .add_where("?person", "<http://xmlns.com/foaf/0.1/name>", "?name")
            .add_filter("LANG(?name) = \"en\"")
            .build()
            .unwrap();

        assert!(query.contains("FILTER (LANG(?name) = \"en\")"));
    }

    #[test]
    fn test_select_with_optional() {
        let query = SelectBuilder::new()
            .add_var("?person")
            .add_var("?email")
            .add_where("?person", "a", "<http://xmlns.com/foaf/0.1/Person>")
            .add_optional(vec![(
                "?person",
                "<http://xmlns.com/foaf/0.1/mbox>",
                "?email",
            )])
            .build()
            .unwrap();

        assert!(query.contains("OPTIONAL"));
        assert!(query.contains("?person <http://xmlns.com/foaf/0.1/mbox> ?email"));
    }

    #[test]
    fn test_select_with_order_limit() {
        let query = SelectBuilder::new()
            .add_var("?name")
            .add_where("?person", "<http://xmlns.com/foaf/0.1/name>", "?name")
            .add_order_by("?name", true)
            .limit(10)
            .build()
            .unwrap();

        assert!(query.contains("ORDER BY ?name"));
        assert!(query.contains("LIMIT 10"));
    }

    #[test]
    fn test_select_with_group_by() {
        let query = SelectBuilder::new()
            .add_var("?type")
            .add_var("(COUNT(?item) AS ?count)")
            .add_where("?item", "a", "?type")
            .add_group_by("?type")
            .build()
            .unwrap();

        assert!(query.contains("GROUP BY ?type"));
    }

    #[test]
    fn test_select_with_values() {
        let query = SelectBuilder::new()
            .add_var("?person")
            .add_var("?name")
            .add_where("?person", "<http://xmlns.com/foaf/0.1/name>", "?name")
            .add_values(
                "?person",
                vec![
                    "<http://example.org/person1>",
                    "<http://example.org/person2>",
                ],
            )
            .build()
            .unwrap();

        assert!(query.contains("VALUES ?person"));
        assert!(query.contains("<http://example.org/person1>"));
    }

    #[test]
    fn test_select_with_bind() {
        let query = SelectBuilder::new()
            .add_var("?person")
            .add_var("?label")
            .add_where("?person", "a", "<http://xmlns.com/foaf/0.1/Person>")
            .add_bind("\"Person\" AS ?label", "")
            .build()
            .unwrap();

        assert!(query.contains("BIND"));
    }

    #[test]
    fn test_select_with_prefix() {
        let query = SelectBuilder::new()
            .add_prefix("foaf", "http://xmlns.com/foaf/0.1/")
            .add_var("?person")
            .add_where("?person", "a", "foaf:Person")
            .build()
            .unwrap();

        assert!(query.contains("PREFIX foaf: <http://xmlns.com/foaf/0.1/>"));
    }

    #[test]
    fn test_select_validation_no_vars() {
        let result = SelectBuilder::new().add_where("?s", "?p", "?o").build();

        assert!(result.is_err());
    }

    #[test]
    fn test_select_validation_no_patterns() {
        let result = SelectBuilder::new().add_var("?s").build();

        assert!(result.is_err());
    }

    #[test]
    fn test_construct_builder() {
        let query = ConstructBuilder::new()
            .add_construct("?person", "a", "<http://xmlns.com/foaf/0.1/Person>")
            .add_where("?person", "<http://xmlns.com/foaf/0.1/name>", "?name")
            .build()
            .unwrap();

        assert!(query.contains("CONSTRUCT"));
        assert!(query.contains("?person a <http://xmlns.com/foaf/0.1/Person>"));
    }

    #[test]
    fn test_ask_builder() {
        let query = AskBuilder::new()
            .add_where("?person", "a", "<http://xmlns.com/foaf/0.1/Person>")
            .add_filter("?person = <http://example.org/alice>")
            .build()
            .unwrap();

        assert!(query.contains("ASK"));
        assert!(query.contains("FILTER"));
    }

    #[test]
    fn test_describe_builder() {
        let query = DescribeBuilder::new()
            .add_resource("<http://example.org/alice>")
            .build()
            .unwrap();

        assert!(query.contains("DESCRIBE <http://example.org/alice>"));
    }

    #[test]
    fn test_describe_with_where() {
        let query = DescribeBuilder::new()
            .add_resource("?person")
            .add_where("?person", "a", "<http://xmlns.com/foaf/0.1/Person>")
            .limit(5)
            .build()
            .unwrap();

        assert!(query.contains("DESCRIBE ?person"));
        assert!(query.contains("WHERE"));
        assert!(query.contains("LIMIT 5"));
    }

    #[test]
    fn test_select_star() {
        let query = SelectBuilder::new()
            .select_all()
            .add_where("?s", "?p", "?o")
            .build()
            .unwrap();

        assert!(query.contains("SELECT *"));
    }

    #[test]
    fn test_order_by_desc() {
        let query = SelectBuilder::new()
            .add_var("?name")
            .add_where("?person", "<http://xmlns.com/foaf/0.1/name>", "?name")
            .add_order_by("?name", false)
            .build()
            .unwrap();

        assert!(query.contains("ORDER BY DESC(?name)"));
    }

    #[test]
    fn test_offset() {
        let query = SelectBuilder::new()
            .add_var("?s")
            .add_where("?s", "?p", "?o")
            .limit(10)
            .offset(20)
            .build()
            .unwrap();

        assert!(query.contains("LIMIT 10"));
        assert!(query.contains("OFFSET 20"));
    }

    #[test]
    fn test_having_without_group_by_fails() {
        let result = SelectBuilder::new()
            .add_var("?count")
            .add_where("?s", "?p", "?o")
            .add_having("?count > 5")
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_union() {
        let query = SelectBuilder::new()
            .add_var("?person")
            .add_union(
                vec![("?person", "a", "<http://xmlns.com/foaf/0.1/Person>")],
                vec![("?person", "a", "<http://example.org/Agent>")],
            )
            .build()
            .unwrap();

        assert!(query.contains("UNION"));
    }

    #[test]
    fn test_property_path() {
        let query = SelectBuilder::new()
            .add_var("?person")
            .add_var("?ancestor")
            .add_path("?person", "<http://example.org/parent>+", "?ancestor")
            .build()
            .unwrap();

        assert!(query.contains("?person <http://example.org/parent>+ ?ancestor"));
    }
}
