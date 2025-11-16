//! Interactive SPARQL Query Builder
//!
//! Provides a fluent, type-safe API for constructing SPARQL queries programmatically.
//! Supports SELECT, ASK, CONSTRUCT, DESCRIBE queries with comprehensive pattern building.

use crate::algebra::{
    Aggregate, Algebra, Expression, GroupCondition, Iri, Literal, OrderCondition, Term,
    TriplePattern, Variable,
};
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Interactive query builder for SPARQL queries
#[derive(Debug, Clone)]
pub struct InteractiveQueryBuilder {
    query_type: Option<QueryType>,
    variables: Vec<Variable>,
    patterns: Vec<PatternBuilder>,
    filters: Vec<Expression>,
    optional_patterns: Vec<Vec<PatternBuilder>>,
    unions: Vec<Vec<PatternBuilder>>,
    bindings: Vec<(Variable, Expression)>,
    order_by: Vec<OrderCondition>,
    group_by: Vec<GroupCondition>,
    aggregates: Vec<(Variable, Aggregate)>,
    having: Option<Expression>,
    limit: Option<usize>,
    offset: Option<usize>,
    distinct: bool,
    reduced: bool,
    prefixes: HashMap<String, String>,
}

/// Query type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    Select,
    Ask,
    Construct,
    Describe,
}

/// Pattern builder for triple patterns
#[derive(Debug, Clone)]
pub struct PatternBuilder {
    subject: Option<Term>,
    predicate: Option<Term>,
    object: Option<Term>,
}

impl InteractiveQueryBuilder {
    /// Create a new query builder
    pub fn new() -> Self {
        Self {
            query_type: None,
            variables: Vec::new(),
            patterns: Vec::new(),
            filters: Vec::new(),
            optional_patterns: Vec::new(),
            unions: Vec::new(),
            bindings: Vec::new(),
            order_by: Vec::new(),
            group_by: Vec::new(),
            aggregates: Vec::new(),
            having: None,
            limit: None,
            offset: None,
            distinct: false,
            reduced: false,
            prefixes: HashMap::new(),
        }
    }

    /// Start building a SELECT query
    pub fn select(mut self, variables: Vec<&str>) -> Result<Self> {
        self.query_type = Some(QueryType::Select);
        self.variables = variables
            .into_iter()
            .map(Variable::new)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(self)
    }

    /// Start building a SELECT * query
    pub fn select_all(mut self) -> Self {
        self.query_type = Some(QueryType::Select);
        self.variables = Vec::new(); // Empty means SELECT *
        self
    }

    /// Start building an ASK query
    pub fn ask(mut self) -> Self {
        self.query_type = Some(QueryType::Ask);
        self
    }

    /// Start building a CONSTRUCT query
    pub fn construct(mut self, variables: Vec<&str>) -> Result<Self> {
        self.query_type = Some(QueryType::Construct);
        self.variables = variables
            .into_iter()
            .map(Variable::new)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(self)
    }

    /// Start building a DESCRIBE query
    pub fn describe(mut self, resources: Vec<&str>) -> Result<Self> {
        self.query_type = Some(QueryType::Describe);
        self.variables = resources
            .into_iter()
            .map(Variable::new)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(self)
    }

    /// Add a WHERE pattern
    pub fn r#where(mut self, pattern: PatternBuilder) -> Self {
        self.patterns.push(pattern);
        self
    }

    /// Add multiple WHERE patterns
    pub fn where_all(mut self, patterns: Vec<PatternBuilder>) -> Self {
        self.patterns.extend(patterns);
        self
    }

    /// Add a FILTER condition
    pub fn filter(mut self, condition: Expression) -> Self {
        self.filters.push(condition);
        self
    }

    /// Add an OPTIONAL pattern
    pub fn optional(mut self, patterns: Vec<PatternBuilder>) -> Self {
        self.optional_patterns.push(patterns);
        self
    }

    /// Add a UNION pattern
    pub fn union(mut self, patterns: Vec<PatternBuilder>) -> Self {
        self.unions.push(patterns);
        self
    }

    /// Add a BIND clause
    pub fn bind(mut self, var: &str, expr: Expression) -> Result<Self> {
        self.bindings.push((Variable::new(var)?, expr));
        Ok(self)
    }

    /// Add ORDER BY clause
    pub fn order_by(mut self, conditions: Vec<OrderCondition>) -> Self {
        self.order_by = conditions;
        self
    }

    /// Add GROUP BY clause
    pub fn group_by(mut self, conditions: Vec<GroupCondition>) -> Self {
        self.group_by = conditions;
        self
    }

    /// Add HAVING clause
    pub fn having(mut self, condition: Expression) -> Self {
        self.having = Some(condition);
        self
    }

    /// Add COUNT aggregate
    pub fn count(mut self, var: &str, expr: Option<Expression>, distinct: bool) -> Result<Self> {
        let variable = Variable::new(var)?;
        self.aggregates
            .push((variable, Aggregate::Count { distinct, expr }));
        Ok(self)
    }

    /// Add SUM aggregate
    pub fn sum(mut self, var: &str, expr: Expression, distinct: bool) -> Result<Self> {
        let variable = Variable::new(var)?;
        self.aggregates
            .push((variable, Aggregate::Sum { distinct, expr }));
        Ok(self)
    }

    /// Add MIN aggregate
    pub fn min(mut self, var: &str, expr: Expression, distinct: bool) -> Result<Self> {
        let variable = Variable::new(var)?;
        self.aggregates
            .push((variable, Aggregate::Min { distinct, expr }));
        Ok(self)
    }

    /// Add MAX aggregate
    pub fn max(mut self, var: &str, expr: Expression, distinct: bool) -> Result<Self> {
        let variable = Variable::new(var)?;
        self.aggregates
            .push((variable, Aggregate::Max { distinct, expr }));
        Ok(self)
    }

    /// Add AVG aggregate
    pub fn avg(mut self, var: &str, expr: Expression, distinct: bool) -> Result<Self> {
        let variable = Variable::new(var)?;
        self.aggregates
            .push((variable, Aggregate::Avg { distinct, expr }));
        Ok(self)
    }

    /// Add SAMPLE aggregate
    pub fn sample(mut self, var: &str, expr: Expression, distinct: bool) -> Result<Self> {
        let variable = Variable::new(var)?;
        self.aggregates
            .push((variable, Aggregate::Sample { distinct, expr }));
        Ok(self)
    }

    /// Add GROUP_CONCAT aggregate
    pub fn group_concat(
        mut self,
        var: &str,
        expr: Expression,
        distinct: bool,
        separator: Option<String>,
    ) -> Result<Self> {
        let variable = Variable::new(var)?;
        self.aggregates.push((
            variable,
            Aggregate::GroupConcat {
                distinct,
                expr,
                separator,
            },
        ));
        Ok(self)
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

    /// Set DISTINCT modifier
    pub fn distinct(mut self) -> Self {
        self.distinct = true;
        self
    }

    /// Set REDUCED modifier
    pub fn reduced(mut self) -> Self {
        self.reduced = true;
        self
    }

    /// Add a prefix declaration
    pub fn prefix(mut self, prefix: &str, uri: &str) -> Self {
        self.prefixes.insert(prefix.to_string(), uri.to_string());
        self
    }

    /// Build the SPARQL query string
    pub fn build_string(&self) -> Result<String> {
        let mut query = String::new();

        // Add prefixes
        for (prefix, uri) in &self.prefixes {
            query.push_str(&format!("PREFIX {}: <{}>\n", prefix, uri));
        }
        if !self.prefixes.is_empty() {
            query.push('\n');
        }

        // Add query type
        let query_type = self
            .query_type
            .ok_or_else(|| anyhow!("Query type not set"))?;

        match query_type {
            QueryType::Select => {
                query.push_str("SELECT ");
                if self.distinct {
                    query.push_str("DISTINCT ");
                }
                if self.reduced {
                    query.push_str("REDUCED ");
                }
                if self.variables.is_empty() {
                    query.push('*');
                } else {
                    for (i, var) in self.variables.iter().enumerate() {
                        if i > 0 {
                            query.push(' ');
                        }
                        query.push_str(&format!("{}", var));
                    }
                }
                query.push('\n');
            }
            QueryType::Ask => {
                query.push_str("ASK\n");
            }
            QueryType::Construct => {
                query.push_str("CONSTRUCT {\n");
                for pattern in &self.patterns {
                    query.push_str(&format!("  {}\n", pattern.to_string()?));
                }
                query.push_str("}\n");
            }
            QueryType::Describe => {
                query.push_str("DESCRIBE ");
                for (i, var) in self.variables.iter().enumerate() {
                    if i > 0 {
                        query.push(' ');
                    }
                    query.push_str(&format!("{}", var));
                }
                query.push('\n');
            }
        }

        // Add WHERE clause
        query.push_str("WHERE {\n");

        // Add patterns
        for pattern in &self.patterns {
            query.push_str(&format!("  {} .\n", pattern.to_string()?));
        }

        // Add FILTERs
        for filter in &self.filters {
            query.push_str(&format!("  FILTER ({:?})\n", filter));
        }

        // Add OPTIONALs
        for optional in &self.optional_patterns {
            query.push_str("  OPTIONAL {\n");
            for pattern in optional {
                query.push_str(&format!("    {} .\n", pattern.to_string()?));
            }
            query.push_str("  }\n");
        }

        // Add UNIONs
        if !self.unions.is_empty() {
            query.push_str("  {\n");
            for (i, union_patterns) in self.unions.iter().enumerate() {
                if i > 0 {
                    query.push_str("  } UNION {\n");
                }
                for pattern in union_patterns {
                    query.push_str(&format!("    {} .\n", pattern.to_string()?));
                }
            }
            query.push_str("  }\n");
        }

        // Add BINDs
        for (var, expr) in &self.bindings {
            query.push_str(&format!("  BIND ({:?} AS {})\n", expr, var));
        }

        query.push_str("}\n");

        // Add GROUP BY
        if !self.group_by.is_empty() {
            query.push_str("GROUP BY ");
            for (i, condition) in self.group_by.iter().enumerate() {
                if i > 0 {
                    query.push(' ');
                }
                query.push_str(&format!("({:?})", condition.expr));
                if let Some(alias) = &condition.alias {
                    query.push_str(&format!(" AS {}", alias));
                }
            }
            query.push('\n');
        }

        // Add HAVING
        if let Some(having) = &self.having {
            query.push_str(&format!("HAVING ({:?})\n", having));
        }

        // Add ORDER BY
        if !self.order_by.is_empty() {
            query.push_str("ORDER BY ");
            for (i, condition) in self.order_by.iter().enumerate() {
                if i > 0 {
                    query.push(' ');
                }
                if condition.ascending {
                    query.push_str(&format!("ASC({:?})", condition.expr));
                } else {
                    query.push_str(&format!("DESC({:?})", condition.expr));
                }
            }
            query.push('\n');
        }

        // Add LIMIT
        if let Some(limit) = self.limit {
            query.push_str(&format!("LIMIT {}\n", limit));
        }

        // Add OFFSET
        if let Some(offset) = self.offset {
            query.push_str(&format!("OFFSET {}\n", offset));
        }

        Ok(query)
    }

    /// Build the algebra representation
    pub fn build_algebra(&self) -> Result<Algebra> {
        let query_type = self
            .query_type
            .ok_or_else(|| anyhow!("Query type not set"))?;

        // Build basic graph pattern from patterns
        let mut algebra = if !self.patterns.is_empty() {
            let triple_patterns: Result<Vec<TriplePattern>> = self
                .patterns
                .iter()
                .map(|p| p.to_triple_pattern())
                .collect();
            Algebra::Bgp(triple_patterns?)
        } else {
            Algebra::Bgp(Vec::new())
        };

        // Add filters
        for filter in &self.filters {
            algebra = Algebra::Filter {
                pattern: Box::new(algebra),
                condition: filter.clone(),
            };
        }

        // Add optionals
        for optional in &self.optional_patterns {
            let optional_patterns: Result<Vec<TriplePattern>> =
                optional.iter().map(|p| p.to_triple_pattern()).collect();
            let optional_algebra = Algebra::Bgp(optional_patterns?);
            algebra = Algebra::LeftJoin {
                left: Box::new(algebra),
                right: Box::new(optional_algebra),
                filter: None,
            };
        }

        // Add unions
        if !self.unions.is_empty() {
            let mut union_algebra = None;
            for union_patterns in &self.unions {
                let patterns: Result<Vec<TriplePattern>> = union_patterns
                    .iter()
                    .map(|p| p.to_triple_pattern())
                    .collect();
                let union_bgp = Algebra::Bgp(patterns?);
                union_algebra = Some(match union_algebra {
                    None => union_bgp,
                    Some(left) => Algebra::Union {
                        left: Box::new(left),
                        right: Box::new(union_bgp),
                    },
                });
            }
            if let Some(union_alg) = union_algebra {
                algebra = Algebra::Join {
                    left: Box::new(algebra),
                    right: Box::new(union_alg),
                };
            }
        }

        // Add bindings
        for (var, expr) in &self.bindings {
            algebra = Algebra::Extend {
                pattern: Box::new(algebra),
                variable: var.clone(),
                expr: expr.clone(),
            };
        }

        // Add group by
        if !self.group_by.is_empty() || !self.aggregates.is_empty() {
            algebra = Algebra::Group {
                pattern: Box::new(algebra),
                variables: self.group_by.clone(),
                aggregates: self.aggregates.clone(),
            };
        }

        // Add having
        if let Some(having) = &self.having {
            algebra = Algebra::Having {
                pattern: Box::new(algebra),
                condition: having.clone(),
            };
        }

        // Add order by
        if !self.order_by.is_empty() {
            algebra = Algebra::OrderBy {
                pattern: Box::new(algebra),
                conditions: self.order_by.clone(),
            };
        }

        // Add projection for SELECT
        if query_type == QueryType::Select && !self.variables.is_empty() {
            algebra = Algebra::Project {
                pattern: Box::new(algebra),
                variables: self.variables.clone(),
            };
        }

        // Add distinct
        if self.distinct {
            algebra = Algebra::Distinct {
                pattern: Box::new(algebra),
            };
        }

        // Add reduced
        if self.reduced {
            algebra = Algebra::Reduced {
                pattern: Box::new(algebra),
            };
        }

        // Add slice
        if self.limit.is_some() || self.offset.is_some() {
            algebra = Algebra::Slice {
                pattern: Box::new(algebra),
                offset: self.offset,
                limit: self.limit,
            };
        }

        Ok(algebra)
    }
}

impl Default for InteractiveQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternBuilder {
    /// Create a new pattern builder
    pub fn new() -> Self {
        Self {
            subject: None,
            predicate: None,
            object: None,
        }
    }

    /// Set the subject
    pub fn subject(mut self, subject: Term) -> Self {
        self.subject = Some(subject);
        self
    }

    /// Set the subject as a variable
    pub fn subject_var(self, var: &str) -> Result<Self> {
        Ok(self.subject(Term::Variable(Variable::new(var)?)))
    }

    /// Set the subject as an IRI
    pub fn subject_iri(self, iri: &str) -> Result<Self> {
        Ok(self.subject(Term::Iri(Iri::new(iri)?)))
    }

    /// Set the predicate
    pub fn predicate(mut self, predicate: Term) -> Self {
        self.predicate = Some(predicate);
        self
    }

    /// Set the predicate as a variable
    pub fn predicate_var(self, var: &str) -> Result<Self> {
        Ok(self.predicate(Term::Variable(Variable::new(var)?)))
    }

    /// Set the predicate as an IRI
    pub fn predicate_iri(self, iri: &str) -> Result<Self> {
        Ok(self.predicate(Term::Iri(Iri::new(iri)?)))
    }

    /// Set the object
    pub fn object(mut self, object: Term) -> Self {
        self.object = Some(object);
        self
    }

    /// Set the object as a variable
    pub fn object_var(self, var: &str) -> Result<Self> {
        Ok(self.object(Term::Variable(Variable::new(var)?)))
    }

    /// Set the object as an IRI
    pub fn object_iri(self, iri: &str) -> Result<Self> {
        Ok(self.object(Term::Iri(Iri::new(iri)?)))
    }

    /// Set the object as a literal
    pub fn object_literal(self, value: &str) -> Self {
        self.object(Term::Literal(Literal {
            value: value.to_string(),
            language: None,
            datatype: None,
        }))
    }

    /// Convert to triple pattern
    pub fn to_triple_pattern(&self) -> Result<TriplePattern> {
        Ok(TriplePattern {
            subject: self
                .subject
                .clone()
                .ok_or_else(|| anyhow!("Subject not set"))?,
            predicate: self
                .predicate
                .clone()
                .ok_or_else(|| anyhow!("Predicate not set"))?,
            object: self
                .object
                .clone()
                .ok_or_else(|| anyhow!("Object not set"))?,
        })
    }

    /// Convert to string representation
    pub fn to_string(&self) -> Result<String> {
        let subject = self
            .subject
            .as_ref()
            .ok_or_else(|| anyhow!("Subject not set"))?;
        let predicate = self
            .predicate
            .as_ref()
            .ok_or_else(|| anyhow!("Predicate not set"))?;
        let object = self
            .object
            .as_ref()
            .ok_or_else(|| anyhow!("Object not set"))?;

        Ok(format!("{} {} {}", subject, predicate, object))
    }
}

impl Default for PatternBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper functions for building common patterns
pub mod helpers {
    use super::*;

    /// Create a triple pattern: ?s ?p ?o
    pub fn triple(s: &str, p: &str, o: &str) -> Result<PatternBuilder> {
        PatternBuilder::new()
            .subject_var(s)?
            .predicate_var(p)?
            .object_var(o)
    }

    /// Create a pattern: ?s <predicate> ?o
    pub fn triple_with_iri_predicate(s: &str, p: &str, o: &str) -> Result<PatternBuilder> {
        PatternBuilder::new()
            .subject_var(s)?
            .predicate_iri(p)?
            .object_var(o)
    }

    /// Create a pattern: <subject> <predicate> ?o
    pub fn triple_with_iri_subject_predicate(s: &str, p: &str, o: &str) -> Result<PatternBuilder> {
        PatternBuilder::new()
            .subject_iri(s)?
            .predicate_iri(p)?
            .object_var(o)
    }

    /// Create a pattern with literal object
    pub fn triple_with_literal(s: &str, p: &str, literal: &str) -> Result<PatternBuilder> {
        Ok(PatternBuilder::new()
            .subject_var(s)?
            .predicate_iri(p)?
            .object_literal(literal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select() -> Result<()> {
        let query = InteractiveQueryBuilder::new()
            .select(vec!["s", "p", "o"])?
            .r#where(
                PatternBuilder::new()
                    .subject_var("s")?
                    .predicate_var("p")?
                    .object_var("o")?,
            )
            .build_string()?;

        assert!(query.contains("SELECT ?s ?p ?o"));
        assert!(query.contains("WHERE"));
        Ok(())
    }

    #[test]
    fn test_select_with_filter() -> Result<()> {
        let builder = InteractiveQueryBuilder::new()
            .select(vec!["name"])?
            .r#where(
                PatternBuilder::new()
                    .subject_var("person")?
                    .predicate_iri("http://xmlns.com/foaf/0.1/name")?
                    .object_var("name")?,
            )
            .limit(10);

        let query = builder.build_string()?;
        assert!(query.contains("SELECT ?name"));
        assert!(query.contains("LIMIT 10"));
        Ok(())
    }

    #[test]
    fn test_ask_query() -> Result<()> {
        let query = InteractiveQueryBuilder::new()
            .ask()
            .r#where(
                PatternBuilder::new()
                    .subject_var("s")?
                    .predicate_var("p")?
                    .object_var("o")?,
            )
            .build_string()?;

        assert!(query.contains("ASK"));
        assert!(query.contains("WHERE"));
        Ok(())
    }

    #[test]
    fn test_distinct_modifier() -> Result<()> {
        let query = InteractiveQueryBuilder::new()
            .select(vec!["s"])?
            .distinct()
            .r#where(
                PatternBuilder::new()
                    .subject_var("s")?
                    .predicate_var("p")?
                    .object_var("o")?,
            )
            .build_string()?;

        assert!(query.contains("SELECT DISTINCT"));
        Ok(())
    }

    #[test]
    fn test_with_prefixes() -> Result<()> {
        let query = InteractiveQueryBuilder::new()
            .prefix("foaf", "http://xmlns.com/foaf/0.1/")
            .prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
            .select(vec!["name"])?
            .r#where(
                PatternBuilder::new()
                    .subject_var("person")?
                    .predicate_iri("http://xmlns.com/foaf/0.1/name")?
                    .object_var("name")?,
            )
            .build_string()?;

        assert!(query.contains("PREFIX foaf:"));
        assert!(query.contains("PREFIX rdf:"));
        Ok(())
    }

    #[test]
    fn test_algebra_generation() -> Result<()> {
        let algebra = InteractiveQueryBuilder::new()
            .select(vec!["s", "p", "o"])?
            .r#where(
                PatternBuilder::new()
                    .subject_var("s")?
                    .predicate_var("p")?
                    .object_var("o")?,
            )
            .limit(10)
            .build_algebra()?;

        // Should have Slice > Project > Bgp structure
        match algebra {
            Algebra::Slice { .. } => {}
            _ => panic!("Expected Slice algebra"),
        }

        Ok(())
    }

    #[test]
    fn test_helper_functions() -> Result<()> {
        let pattern = helpers::triple("s", "p", "o")?;
        let triple = pattern.to_triple_pattern()?;

        assert!(matches!(triple.subject, Term::Variable(_)));
        assert!(matches!(triple.predicate, Term::Variable(_)));
        assert!(matches!(triple.object, Term::Variable(_)));

        Ok(())
    }

    #[test]
    fn test_optional_pattern() -> Result<()> {
        let query = InteractiveQueryBuilder::new()
            .select(vec!["name", "email"])?
            .r#where(
                PatternBuilder::new()
                    .subject_var("person")?
                    .predicate_iri("http://xmlns.com/foaf/0.1/name")?
                    .object_var("name")?,
            )
            .optional(vec![PatternBuilder::new()
                .subject_var("person")?
                .predicate_iri("http://xmlns.com/foaf/0.1/mbox")?
                .object_var("email")?])
            .build_string()?;

        assert!(query.contains("OPTIONAL"));
        Ok(())
    }

    #[test]
    fn test_limit_offset() -> Result<()> {
        let query = InteractiveQueryBuilder::new()
            .select(vec!["s"])?
            .r#where(
                PatternBuilder::new()
                    .subject_var("s")?
                    .predicate_var("p")?
                    .object_var("o")?,
            )
            .limit(10)
            .offset(20)
            .build_string()?;

        assert!(query.contains("LIMIT 10"));
        assert!(query.contains("OFFSET 20"));
        Ok(())
    }

    #[test]
    fn test_count_aggregate() -> Result<()> {
        let algebra = InteractiveQueryBuilder::new()
            .select_all()
            .r#where(
                PatternBuilder::new()
                    .subject_var("s")?
                    .predicate_var("p")?
                    .object_var("o")?,
            )
            .count("count", None, false)?
            .build_algebra()?;

        // Check that aggregate is in the algebra
        match algebra {
            Algebra::Group { aggregates, .. } => {
                assert_eq!(aggregates.len(), 1);
                let (var, agg) = &aggregates[0];
                assert_eq!(var.name(), "count");
                matches!(agg, Aggregate::Count { .. });
            }
            _ => panic!("Expected Group algebra"),
        }

        Ok(())
    }

    #[test]
    fn test_sum_aggregate() -> Result<()> {
        let algebra = InteractiveQueryBuilder::new()
            .select_all()
            .r#where(
                PatternBuilder::new()
                    .subject_var("s")?
                    .predicate_var("p")?
                    .object_var("o")?,
            )
            .sum(
                "total",
                Expression::Variable(Variable::new_unchecked("value")),
                false,
            )?
            .build_algebra()?;

        match algebra {
            Algebra::Group { aggregates, .. } => {
                assert_eq!(aggregates.len(), 1);
                let (var, agg) = &aggregates[0];
                assert_eq!(var.name(), "total");
                matches!(agg, Aggregate::Sum { .. });
            }
            _ => panic!("Expected Group algebra"),
        }

        Ok(())
    }

    #[test]
    fn test_multiple_aggregates() -> Result<()> {
        let algebra = InteractiveQueryBuilder::new()
            .select_all()
            .r#where(
                PatternBuilder::new()
                    .subject_var("s")?
                    .predicate_var("p")?
                    .object_var("o")?,
            )
            .count("cnt", None, false)?
            .sum(
                "total",
                Expression::Variable(Variable::new_unchecked("value")),
                false,
            )?
            .avg(
                "avg",
                Expression::Variable(Variable::new_unchecked("value")),
                false,
            )?
            .build_algebra()?;

        match algebra {
            Algebra::Group { aggregates, .. } => {
                assert_eq!(aggregates.len(), 3);
                assert_eq!(aggregates[0].0.name(), "cnt");
                assert_eq!(aggregates[1].0.name(), "total");
                assert_eq!(aggregates[2].0.name(), "avg");
            }
            _ => panic!("Expected Group algebra"),
        }

        Ok(())
    }

    #[test]
    fn test_group_concat_aggregate() -> Result<()> {
        let algebra = InteractiveQueryBuilder::new()
            .select_all()
            .r#where(
                PatternBuilder::new()
                    .subject_var("s")?
                    .predicate_var("p")?
                    .object_var("o")?,
            )
            .group_concat(
                "concat",
                Expression::Variable(Variable::new_unchecked("name")),
                false,
                Some(", ".to_string()),
            )?
            .build_algebra()?;

        match algebra {
            Algebra::Group { aggregates, .. } => {
                assert_eq!(aggregates.len(), 1);
                match &aggregates[0].1 {
                    Aggregate::GroupConcat { separator, .. } => {
                        assert_eq!(separator.as_deref(), Some(", "));
                    }
                    _ => panic!("Expected GroupConcat aggregate"),
                }
            }
            _ => panic!("Expected Group algebra"),
        }

        Ok(())
    }
}
