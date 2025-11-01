//! Aggregation support for GraphQL queries
//!
//! This module provides comprehensive aggregation capabilities:
//! - COUNT, SUM, AVG, MIN, MAX aggregations
//! - GROUP BY with multiple fields
//! - HAVING clauses for filtered aggregations
//! - Nested aggregations
//! - DISTINCT support

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Aggregation function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregateFunction {
    /// Count of results
    Count,
    /// Sum of numeric values
    Sum,
    /// Average of numeric values
    Avg,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Group concatenation (string aggregation)
    GroupConcat,
    /// Sample value (arbitrary value from group)
    Sample,
}

impl fmt::Display for AggregateFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregateFunction::Count => write!(f, "COUNT"),
            AggregateFunction::Sum => write!(f, "SUM"),
            AggregateFunction::Avg => write!(f, "AVG"),
            AggregateFunction::Min => write!(f, "MIN"),
            AggregateFunction::Max => write!(f, "MAX"),
            AggregateFunction::GroupConcat => write!(f, "GROUP_CONCAT"),
            AggregateFunction::Sample => write!(f, "SAMPLE"),
        }
    }
}

/// Aggregation specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aggregation {
    /// Aggregation function
    pub function: AggregateFunction,
    /// Field to aggregate
    pub field: String,
    /// Alias for the result
    pub alias: String,
    /// Whether to use DISTINCT
    pub distinct: bool,
    /// Separator for GROUP_CONCAT (default: " ")
    pub separator: Option<String>,
}

impl Aggregation {
    pub fn new(function: AggregateFunction, field: String, alias: String) -> Self {
        Self {
            function,
            field,
            alias,
            distinct: false,
            separator: None,
        }
    }

    pub fn with_distinct(mut self) -> Self {
        self.distinct = true;
        self
    }

    pub fn with_separator(mut self, separator: String) -> Self {
        self.separator = Some(separator);
        self
    }

    /// Convert to SPARQL aggregation expression
    pub fn to_sparql_expression(&self) -> String {
        let distinct_prefix = if self.distinct { "DISTINCT " } else { "" };
        let var = format!("?{}", self.field);

        match self.function {
            AggregateFunction::Count => {
                if self.field == "*" {
                    format!("(COUNT({}*) AS ?{})", distinct_prefix, self.alias)
                } else {
                    format!("(COUNT({}{}) AS ?{})", distinct_prefix, var, self.alias)
                }
            }
            AggregateFunction::Sum => {
                format!("(SUM({}{}) AS ?{})", distinct_prefix, var, self.alias)
            }
            AggregateFunction::Avg => {
                format!("(AVG({}{}) AS ?{})", distinct_prefix, var, self.alias)
            }
            AggregateFunction::Min => {
                format!("(MIN({}) AS ?{})", var, self.alias)
            }
            AggregateFunction::Max => {
                format!("(MAX({}) AS ?{})", var, self.alias)
            }
            AggregateFunction::GroupConcat => {
                let separator = self.separator.as_deref().unwrap_or(" ");
                format!(
                    "(GROUP_CONCAT({}{}; SEPARATOR=\"{}\") AS ?{})",
                    distinct_prefix, var, separator, self.alias
                )
            }
            AggregateFunction::Sample => {
                format!("(SAMPLE({}) AS ?{})", var, self.alias)
            }
        }
    }
}

/// GROUP BY specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupBy {
    /// Fields to group by
    pub fields: Vec<String>,
}

impl GroupBy {
    pub fn new(fields: Vec<String>) -> Self {
        Self { fields }
    }

    /// Convert to SPARQL GROUP BY clause
    pub fn to_sparql_clause(&self) -> String {
        let vars: Vec<String> = self.fields.iter().map(|f| format!("?{}", f)).collect();
        format!("GROUP BY {}", vars.join(" "))
    }
}

/// HAVING clause condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HavingCondition {
    /// Aggregation to test
    pub aggregation: Aggregation,
    /// Comparison operator
    pub operator: String,
    /// Value to compare against
    pub value: serde_json::Value,
}

impl HavingCondition {
    pub fn new(aggregation: Aggregation, operator: String, value: serde_json::Value) -> Self {
        Self {
            aggregation,
            operator,
            value,
        }
    }

    /// Convert to SPARQL HAVING clause
    pub fn to_sparql_clause(&self) -> Result<String> {
        let agg_expr = self.aggregation.to_sparql_expression();
        // Extract the expression part without the alias
        let expr_part = if let Some(start) = agg_expr.find('(') {
            if let Some(end) = agg_expr.find(" AS ") {
                &agg_expr[start..end]
            } else {
                return Err(anyhow!("Invalid aggregation expression"));
            }
        } else {
            return Err(anyhow!("Invalid aggregation expression"));
        };

        let value_str = match &self.value {
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::String(s) => format!("\"{}\"", s),
            serde_json::Value::Bool(b) => b.to_string(),
            _ => return Err(anyhow!("Unsupported value type for HAVING")),
        };

        Ok(format!("{} {} {}", expr_part, self.operator, value_str))
    }
}

/// Complete aggregation query specification
#[derive(Debug, Clone)]
pub struct AggregationQuery {
    /// List of aggregations to perform
    pub aggregations: Vec<Aggregation>,
    /// GROUP BY specification
    pub group_by: Option<GroupBy>,
    /// HAVING conditions
    pub having: Vec<HavingCondition>,
    /// Additional non-aggregated fields to select
    pub select_fields: Vec<String>,
}

impl AggregationQuery {
    pub fn new() -> Self {
        Self {
            aggregations: Vec::new(),
            group_by: None,
            having: Vec::new(),
            select_fields: Vec::new(),
        }
    }

    pub fn with_aggregation(mut self, aggregation: Aggregation) -> Self {
        self.aggregations.push(aggregation);
        self
    }

    pub fn with_group_by(mut self, group_by: GroupBy) -> Self {
        self.group_by = Some(group_by);
        self
    }

    pub fn with_having(mut self, having: HavingCondition) -> Self {
        self.having.push(having);
        self
    }

    pub fn with_select_field(mut self, field: String) -> Self {
        self.select_fields.push(field);
        self
    }

    /// Generate SPARQL SELECT clause with aggregations
    pub fn to_sparql_select(&self) -> String {
        let mut select_parts = Vec::new();

        // Add non-aggregated fields
        for field in &self.select_fields {
            select_parts.push(format!("?{}", field));
        }

        // Add aggregations
        for agg in &self.aggregations {
            select_parts.push(agg.to_sparql_expression());
        }

        format!("SELECT {}", select_parts.join(" "))
    }

    /// Generate complete SPARQL aggregation query
    pub fn to_sparql_query(&self, where_clause: &str) -> Result<String> {
        let mut query = String::new();

        // SELECT clause
        query.push_str(&self.to_sparql_select());
        query.push('\n');

        // WHERE clause
        query.push_str("WHERE {\n");
        query.push_str(where_clause);
        query.push_str("\n}\n");

        // GROUP BY clause
        if let Some(group_by) = &self.group_by {
            query.push_str(&group_by.to_sparql_clause());
            query.push('\n');
        }

        // HAVING clause
        if !self.having.is_empty() {
            let having_conditions: Result<Vec<String>> =
                self.having.iter().map(|h| h.to_sparql_clause()).collect();
            let having_conditions = having_conditions?;
            query.push_str(&format!("HAVING ({})\n", having_conditions.join(" && ")));
        }

        Ok(query)
    }
}

impl Default for AggregationQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of an aggregation query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    /// Field values for grouped fields
    pub group_values: HashMap<String, serde_json::Value>,
    /// Aggregation results
    pub aggregates: HashMap<String, serde_json::Value>,
}

impl AggregationResult {
    pub fn new() -> Self {
        Self {
            group_values: HashMap::new(),
            aggregates: HashMap::new(),
        }
    }

    pub fn add_group_value(&mut self, field: String, value: serde_json::Value) {
        self.group_values.insert(field, value);
    }

    pub fn add_aggregate(&mut self, alias: String, value: serde_json::Value) {
        self.aggregates.insert(alias, value);
    }
}

impl Default for AggregationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for aggregation queries
pub struct AggregationQueryBuilder {
    query: AggregationQuery,
}

impl AggregationQueryBuilder {
    pub fn new() -> Self {
        Self {
            query: AggregationQuery::new(),
        }
    }

    /// Add a COUNT aggregation
    pub fn count(mut self, field: String, alias: String) -> Self {
        self.query =
            self.query
                .with_aggregation(Aggregation::new(AggregateFunction::Count, field, alias));
        self
    }

    /// Add a SUM aggregation
    pub fn sum(mut self, field: String, alias: String) -> Self {
        self.query =
            self.query
                .with_aggregation(Aggregation::new(AggregateFunction::Sum, field, alias));
        self
    }

    /// Add an AVG aggregation
    pub fn avg(mut self, field: String, alias: String) -> Self {
        self.query =
            self.query
                .with_aggregation(Aggregation::new(AggregateFunction::Avg, field, alias));
        self
    }

    /// Add a MIN aggregation
    pub fn min(mut self, field: String, alias: String) -> Self {
        self.query =
            self.query
                .with_aggregation(Aggregation::new(AggregateFunction::Min, field, alias));
        self
    }

    /// Add a MAX aggregation
    pub fn max(mut self, field: String, alias: String) -> Self {
        self.query =
            self.query
                .with_aggregation(Aggregation::new(AggregateFunction::Max, field, alias));
        self
    }

    /// Add GROUP BY fields
    pub fn group_by(mut self, fields: Vec<String>) -> Self {
        self.query = self.query.with_group_by(GroupBy::new(fields));
        self
    }

    /// Add HAVING condition
    pub fn having(
        mut self,
        aggregation: Aggregation,
        operator: String,
        value: serde_json::Value,
    ) -> Self {
        self.query = self
            .query
            .with_having(HavingCondition::new(aggregation, operator, value));
        self
    }

    /// Add a field to select (for grouping)
    pub fn select(mut self, field: String) -> Self {
        self.query = self.query.with_select_field(field);
        self
    }

    /// Build the aggregation query
    pub fn build(self) -> AggregationQuery {
        self.query
    }
}

impl Default for AggregationQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_function_display() {
        assert_eq!(AggregateFunction::Count.to_string(), "COUNT");
        assert_eq!(AggregateFunction::Sum.to_string(), "SUM");
        assert_eq!(AggregateFunction::Avg.to_string(), "AVG");
        assert_eq!(AggregateFunction::Min.to_string(), "MIN");
        assert_eq!(AggregateFunction::Max.to_string(), "MAX");
    }

    #[test]
    fn test_aggregation_count() {
        let agg = Aggregation::new(
            AggregateFunction::Count,
            "name".to_string(),
            "total".to_string(),
        );

        let sparql = agg.to_sparql_expression();
        assert!(sparql.contains("COUNT"));
        assert!(sparql.contains("?name"));
        assert!(sparql.contains("?total"));
    }

    #[test]
    fn test_aggregation_sum() {
        let agg = Aggregation::new(
            AggregateFunction::Sum,
            "price".to_string(),
            "total_price".to_string(),
        );

        let sparql = agg.to_sparql_expression();
        assert!(sparql.contains("SUM"));
        assert!(sparql.contains("?price"));
        assert!(sparql.contains("?total_price"));
    }

    #[test]
    fn test_aggregation_with_distinct() {
        let agg = Aggregation::new(
            AggregateFunction::Count,
            "category".to_string(),
            "unique_categories".to_string(),
        )
        .with_distinct();

        let sparql = agg.to_sparql_expression();
        assert!(sparql.contains("DISTINCT"));
    }

    #[test]
    fn test_group_by() {
        let group_by = GroupBy::new(vec!["category".to_string(), "brand".to_string()]);

        let sparql = group_by.to_sparql_clause();
        assert_eq!(sparql, "GROUP BY ?category ?brand");
    }

    #[test]
    fn test_aggregation_query_select() {
        let query = AggregationQuery::new()
            .with_select_field("category".to_string())
            .with_aggregation(Aggregation::new(
                AggregateFunction::Count,
                "id".to_string(),
                "count".to_string(),
            ))
            .with_aggregation(Aggregation::new(
                AggregateFunction::Sum,
                "price".to_string(),
                "total".to_string(),
            ));

        let select = query.to_sparql_select();
        assert!(select.contains("SELECT"));
        assert!(select.contains("?category"));
        assert!(select.contains("COUNT"));
        assert!(select.contains("SUM"));
    }

    #[test]
    fn test_aggregation_query_full() {
        let query = AggregationQuery::new()
            .with_select_field("category".to_string())
            .with_aggregation(Aggregation::new(
                AggregateFunction::Count,
                "id".to_string(),
                "count".to_string(),
            ))
            .with_group_by(GroupBy::new(vec!["category".to_string()]));

        let where_clause = "?s rdf:type ?category . ?s ex:id ?id .";
        let sparql = query.to_sparql_query(where_clause).unwrap();

        assert!(sparql.contains("SELECT"));
        assert!(sparql.contains("WHERE"));
        assert!(sparql.contains("GROUP BY"));
        assert!(sparql.contains("?category"));
    }

    #[test]
    fn test_aggregation_query_builder() {
        let query = AggregationQueryBuilder::new()
            .select("category".to_string())
            .count("id".to_string(), "total".to_string())
            .sum("price".to_string(), "total_price".to_string())
            .group_by(vec!["category".to_string()])
            .build();

        assert_eq!(query.select_fields.len(), 1);
        assert_eq!(query.aggregations.len(), 2);
        assert!(query.group_by.is_some());
    }

    #[test]
    fn test_aggregation_result() {
        let mut result = AggregationResult::new();
        result.add_group_value("category".to_string(), serde_json::json!("Books"));
        result.add_aggregate("count".to_string(), serde_json::json!(42));
        result.add_aggregate("total_price".to_string(), serde_json::json!(199.99));

        assert_eq!(result.group_values.len(), 1);
        assert_eq!(result.aggregates.len(), 2);
    }

    #[test]
    fn test_having_condition() {
        let agg = Aggregation::new(
            AggregateFunction::Count,
            "id".to_string(),
            "count".to_string(),
        );
        let having = HavingCondition::new(agg, ">".to_string(), serde_json::json!(10));

        let clause = having.to_sparql_clause().unwrap();
        assert!(clause.contains(">"));
        assert!(clause.contains("10"));
    }

    #[test]
    fn test_group_concat_with_separator() {
        let agg = Aggregation::new(
            AggregateFunction::GroupConcat,
            "name".to_string(),
            "names".to_string(),
        )
        .with_separator(", ".to_string());

        let sparql = agg.to_sparql_expression();
        assert!(sparql.contains("GROUP_CONCAT"));
        assert!(sparql.contains("SEPARATOR"));
        assert!(sparql.contains(", "));
    }

    #[test]
    fn test_sample_aggregation() {
        let agg = Aggregation::new(
            AggregateFunction::Sample,
            "value".to_string(),
            "sample_value".to_string(),
        );

        let sparql = agg.to_sparql_expression();
        assert!(sparql.contains("SAMPLE"));
        assert!(sparql.contains("?value"));
    }
}
