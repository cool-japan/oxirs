//! Advanced pagination and filtering for GraphQL queries
//!
//! This module provides comprehensive pagination and filtering capabilities:
//! - Cursor-based pagination (Relay spec)
//! - Offset-based pagination
//! - Field-level filtering with operators
//! - Sorting and ordering
//! - Search and full-text capabilities

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Pagination method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaginationMethod {
    /// Cursor-based (Relay-style)
    Cursor,
    /// Offset-based (traditional)
    Offset,
}

/// Pagination configuration
#[derive(Debug, Clone)]
pub struct PaginationConfig {
    pub method: PaginationMethod,
    pub default_page_size: usize,
    pub max_page_size: usize,
    pub enable_total_count: bool,
}

impl Default for PaginationConfig {
    fn default() -> Self {
        Self {
            method: PaginationMethod::Cursor,
            default_page_size: 20,
            max_page_size: 100,
            enable_total_count: true,
        }
    }
}

/// Cursor-based pagination parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPaginationParams {
    pub first: Option<usize>,
    pub after: Option<String>,
    pub last: Option<usize>,
    pub before: Option<String>,
}

impl Default for CursorPaginationParams {
    fn default() -> Self {
        Self {
            first: Some(20),
            after: None,
            last: None,
            before: None,
        }
    }
}

/// Offset-based pagination parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetPaginationParams {
    pub limit: usize,
    pub offset: usize,
}

impl Default for OffsetPaginationParams {
    fn default() -> Self {
        Self {
            limit: 20,
            offset: 0,
        }
    }
}

/// Page info for cursor-based pagination (Relay spec)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageInfo {
    pub has_next_page: bool,
    pub has_previous_page: bool,
    pub start_cursor: Option<String>,
    pub end_cursor: Option<String>,
}

/// Edge for cursor-based pagination (Relay spec)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge<T> {
    pub cursor: String,
    pub node: T,
}

/// Connection for cursor-based pagination (Relay spec)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection<T> {
    pub edges: Vec<Edge<T>>,
    pub page_info: PageInfo,
    pub total_count: Option<usize>,
}

/// Filter operator
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterOperator {
    /// Equals
    Eq,
    /// Not equals
    Ne,
    /// Greater than
    Gt,
    /// Greater than or equal
    Gte,
    /// Less than
    Lt,
    /// Less than or equal
    Lte,
    /// Contains (string)
    Contains,
    /// Starts with (string)
    StartsWith,
    /// Ends with (string)
    EndsWith,
    /// In list
    In,
    /// Not in list
    NotIn,
    /// Is null
    IsNull,
    /// Is not null
    IsNotNull,
    /// Regex match
    Regex,
}

impl fmt::Display for FilterOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterOperator::Eq => write!(f, "="),
            FilterOperator::Ne => write!(f, "!="),
            FilterOperator::Gt => write!(f, ">"),
            FilterOperator::Gte => write!(f, ">="),
            FilterOperator::Lt => write!(f, "<"),
            FilterOperator::Lte => write!(f, "<="),
            FilterOperator::Contains => write!(f, "CONTAINS"),
            FilterOperator::StartsWith => write!(f, "STARTS_WITH"),
            FilterOperator::EndsWith => write!(f, "ENDS_WITH"),
            FilterOperator::In => write!(f, "IN"),
            FilterOperator::NotIn => write!(f, "NOT IN"),
            FilterOperator::IsNull => write!(f, "IS NULL"),
            FilterOperator::IsNotNull => write!(f, "IS NOT NULL"),
            FilterOperator::Regex => write!(f, "REGEX"),
        }
    }
}

/// Filter value
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterValue {
    String(String),
    Int(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<FilterValue>),
    Null,
}

impl FilterValue {
    pub fn to_sparql_literal(&self) -> String {
        match self {
            FilterValue::String(s) => format!("\"{}\"", s.replace('"', "\\\"")),
            FilterValue::Int(i) => i.to_string(),
            FilterValue::Float(f) => f.to_string(),
            FilterValue::Boolean(b) => b.to_string(),
            FilterValue::Null => "NULL".to_string(),
            FilterValue::List(items) => {
                let values: Vec<String> = items.iter().map(|v| v.to_sparql_literal()).collect();
                format!("({})", values.join(", "))
            }
        }
    }
}

/// Filter condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    pub field: String,
    pub operator: FilterOperator,
    pub value: FilterValue,
}

impl FilterCondition {
    pub fn new(field: String, operator: FilterOperator, value: FilterValue) -> Self {
        Self {
            field,
            operator,
            value,
        }
    }

    /// Convert to SPARQL FILTER expression
    pub fn to_sparql_filter(&self, _var_prefix: &str) -> Result<String> {
        let var_name = format!("?{}", self.field);

        let filter = match &self.operator {
            FilterOperator::Eq => {
                format!("{} = {}", var_name, self.value.to_sparql_literal())
            }
            FilterOperator::Ne => {
                format!("{} != {}", var_name, self.value.to_sparql_literal())
            }
            FilterOperator::Gt => {
                format!("{} > {}", var_name, self.value.to_sparql_literal())
            }
            FilterOperator::Gte => {
                format!("{} >= {}", var_name, self.value.to_sparql_literal())
            }
            FilterOperator::Lt => {
                format!("{} < {}", var_name, self.value.to_sparql_literal())
            }
            FilterOperator::Lte => {
                format!("{} <= {}", var_name, self.value.to_sparql_literal())
            }
            FilterOperator::Contains => {
                if let FilterValue::String(s) = &self.value {
                    format!("CONTAINS(LCASE(STR({})), LCASE(\"{}\"))", var_name, s)
                } else {
                    return Err(anyhow!("CONTAINS operator requires string value"));
                }
            }
            FilterOperator::StartsWith => {
                if let FilterValue::String(s) = &self.value {
                    format!("STRSTARTS(LCASE(STR({})), LCASE(\"{}\"))", var_name, s)
                } else {
                    return Err(anyhow!("STARTS_WITH operator requires string value"));
                }
            }
            FilterOperator::EndsWith => {
                if let FilterValue::String(s) = &self.value {
                    format!("STRENDS(LCASE(STR({})), LCASE(\"{}\"))", var_name, s)
                } else {
                    return Err(anyhow!("ENDS_WITH operator requires string value"));
                }
            }
            FilterOperator::In => {
                if let FilterValue::List(items) = &self.value {
                    let values: Vec<String> = items.iter().map(|v| v.to_sparql_literal()).collect();
                    format!("{} IN ({})", var_name, values.join(", "))
                } else {
                    return Err(anyhow!("IN operator requires list value"));
                }
            }
            FilterOperator::NotIn => {
                if let FilterValue::List(items) = &self.value {
                    let values: Vec<String> = items.iter().map(|v| v.to_sparql_literal()).collect();
                    format!("NOT ({} IN ({}))", var_name, values.join(", "))
                } else {
                    return Err(anyhow!("NOT_IN operator requires list value"));
                }
            }
            FilterOperator::IsNull => format!("!BOUND({})", var_name),
            FilterOperator::IsNotNull => format!("BOUND({})", var_name),
            FilterOperator::Regex => {
                if let FilterValue::String(pattern) = &self.value {
                    format!("REGEX(STR({}), \"{}\")", var_name, pattern)
                } else {
                    return Err(anyhow!("REGEX operator requires string pattern"));
                }
            }
        };

        Ok(format!("FILTER ({})", filter))
    }
}

/// Logical combination of filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterCombinator {
    And(Vec<FilterGroup>),
    Or(Vec<FilterGroup>),
    Not(Box<FilterGroup>),
}

/// Filter group (conditions or nested combinators)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterGroup {
    Condition(FilterCondition),
    Combinator(FilterCombinator),
}

impl FilterGroup {
    pub fn to_sparql_filter(&self, var_prefix: &str) -> Result<String> {
        match self {
            FilterGroup::Condition(cond) => cond.to_sparql_filter(var_prefix),
            FilterGroup::Combinator(comb) => match comb {
                FilterCombinator::And(groups) => {
                    let filters: Result<Vec<String>> = groups
                        .iter()
                        .map(|g| g.to_sparql_filter(var_prefix))
                        .collect();
                    let filters = filters?;
                    Ok(format!(
                        "FILTER ({})",
                        filters
                            .iter()
                            .map(|f| f.strip_prefix("FILTER (").unwrap_or(f))
                            .map(|f| f.strip_suffix(")").unwrap_or(f))
                            .map(|f| format!("({})", f))
                            .collect::<Vec<_>>()
                            .join(" && ")
                    ))
                }
                FilterCombinator::Or(groups) => {
                    let filters: Result<Vec<String>> = groups
                        .iter()
                        .map(|g| g.to_sparql_filter(var_prefix))
                        .collect();
                    let filters = filters?;
                    Ok(format!(
                        "FILTER ({})",
                        filters
                            .iter()
                            .map(|f| f.strip_prefix("FILTER (").unwrap_or(f))
                            .map(|f| f.strip_suffix(")").unwrap_or(f))
                            .map(|f| format!("({})", f))
                            .collect::<Vec<_>>()
                            .join(" || ")
                    ))
                }
                FilterCombinator::Not(group) => {
                    let filter = group.to_sparql_filter(var_prefix)?;
                    let inner = filter
                        .strip_prefix("FILTER (")
                        .unwrap_or(&filter)
                        .strip_suffix(")")
                        .unwrap_or(&filter);
                    Ok(format!("FILTER (!({})", inner))
                }
            },
        }
    }
}

/// Sort direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortDirection {
    Asc,
    Desc,
}

impl fmt::Display for SortDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SortDirection::Asc => write!(f, "ASC"),
            SortDirection::Desc => write!(f, "DESC"),
        }
    }
}

/// Sort field specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortField {
    pub field: String,
    pub direction: SortDirection,
}

impl SortField {
    pub fn new(field: String, direction: SortDirection) -> Self {
        Self { field, direction }
    }

    pub fn to_sparql_order(&self) -> String {
        format!("{} (?{})", self.direction, self.field)
    }
}

/// Complete filter and pagination specification
#[derive(Debug, Clone)]
pub struct QueryFilter {
    pub filter: Option<FilterGroup>,
    pub sort: Vec<SortField>,
    pub pagination: PaginationParams,
}

#[derive(Debug, Clone)]
pub enum PaginationParams {
    Cursor(CursorPaginationParams),
    Offset(OffsetPaginationParams),
}

impl QueryFilter {
    pub fn new() -> Self {
        Self {
            filter: None,
            sort: Vec::new(),
            pagination: PaginationParams::Offset(OffsetPaginationParams::default()),
        }
    }

    pub fn with_filter(mut self, filter: FilterGroup) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_sort(mut self, sort: Vec<SortField>) -> Self {
        self.sort = sort;
        self
    }

    pub fn with_cursor_pagination(mut self, params: CursorPaginationParams) -> Self {
        self.pagination = PaginationParams::Cursor(params);
        self
    }

    pub fn with_offset_pagination(mut self, params: OffsetPaginationParams) -> Self {
        self.pagination = PaginationParams::Offset(params);
        self
    }

    /// Generate SPARQL modifiers (FILTER, ORDER BY, LIMIT, OFFSET)
    pub fn to_sparql_modifiers(&self) -> Result<String> {
        let mut modifiers = Vec::new();

        // Add FILTER clause
        if let Some(filter) = &self.filter {
            modifiers.push(filter.to_sparql_filter("")?);
        }

        // Add ORDER BY clause
        if !self.sort.is_empty() {
            let order_clauses: Vec<String> =
                self.sort.iter().map(|s| s.to_sparql_order()).collect();
            modifiers.push(format!("ORDER BY {}", order_clauses.join(" ")));
        }

        // Add LIMIT and OFFSET
        match &self.pagination {
            PaginationParams::Cursor(params) => {
                if let Some(first) = params.first {
                    modifiers.push(format!("LIMIT {}", first + 1)); // +1 to check for next page
                } else if let Some(last) = params.last {
                    modifiers.push(format!("LIMIT {}", last + 1));
                }
            }
            PaginationParams::Offset(params) => {
                modifiers.push(format!("LIMIT {}", params.limit));
                if params.offset > 0 {
                    modifiers.push(format!("OFFSET {}", params.offset));
                }
            }
        }

        Ok(modifiers.join("\n"))
    }
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_operator_display() {
        assert_eq!(FilterOperator::Eq.to_string(), "=");
        assert_eq!(FilterOperator::Contains.to_string(), "CONTAINS");
        assert_eq!(FilterOperator::Gte.to_string(), ">=");
    }

    #[test]
    fn test_filter_value_to_sparql() {
        assert_eq!(
            FilterValue::String("test".to_string()).to_sparql_literal(),
            "\"test\""
        );
        assert_eq!(FilterValue::Int(42).to_sparql_literal(), "42");
        assert_eq!(FilterValue::Float(3.5).to_sparql_literal(), "3.5");
        assert_eq!(FilterValue::Boolean(true).to_sparql_literal(), "true");
    }

    #[test]
    fn test_filter_condition_equals() {
        let condition = FilterCondition::new(
            "name".to_string(),
            FilterOperator::Eq,
            FilterValue::String("Alice".to_string()),
        );

        let filter = condition.to_sparql_filter("").unwrap();
        assert!(filter.contains("?name = \"Alice\""));
    }

    #[test]
    fn test_filter_condition_contains() {
        let condition = FilterCondition::new(
            "description".to_string(),
            FilterOperator::Contains,
            FilterValue::String("test".to_string()),
        );

        let filter = condition.to_sparql_filter("").unwrap();
        assert!(filter.contains("CONTAINS"));
        assert!(filter.contains("?description"));
    }

    #[test]
    fn test_filter_condition_in() {
        let condition = FilterCondition::new(
            "status".to_string(),
            FilterOperator::In,
            FilterValue::List(vec![
                FilterValue::String("active".to_string()),
                FilterValue::String("pending".to_string()),
            ]),
        );

        let filter = condition.to_sparql_filter("").unwrap();
        assert!(filter.contains("IN"));
        assert!(filter.contains("\"active\""));
        assert!(filter.contains("\"pending\""));
    }

    #[test]
    fn test_sort_field() {
        let sort = SortField::new("created_at".to_string(), SortDirection::Desc);
        let sparql = sort.to_sparql_order();

        assert_eq!(sparql, "DESC (?created_at)");
    }

    #[test]
    fn test_query_filter_with_limit() {
        let filter = QueryFilter::new().with_offset_pagination(OffsetPaginationParams {
            limit: 10,
            offset: 0,
        });

        let sparql = filter.to_sparql_modifiers().unwrap();
        assert!(sparql.contains("LIMIT 10"));
    }

    #[test]
    fn test_query_filter_with_offset() {
        let filter = QueryFilter::new().with_offset_pagination(OffsetPaginationParams {
            limit: 10,
            offset: 20,
        });

        let sparql = filter.to_sparql_modifiers().unwrap();
        assert!(sparql.contains("LIMIT 10"));
        assert!(sparql.contains("OFFSET 20"));
    }

    #[test]
    fn test_query_filter_with_sort() {
        let filter = QueryFilter::new().with_sort(vec![
            SortField::new("name".to_string(), SortDirection::Asc),
            SortField::new("created_at".to_string(), SortDirection::Desc),
        ]);

        let sparql = filter.to_sparql_modifiers().unwrap();
        assert!(sparql.contains("ORDER BY"));
        assert!(sparql.contains("ASC (?name)"));
        assert!(sparql.contains("DESC (?created_at)"));
    }

    #[test]
    fn test_pagination_config_defaults() {
        let config = PaginationConfig::default();

        assert_eq!(config.method, PaginationMethod::Cursor);
        assert_eq!(config.default_page_size, 20);
        assert_eq!(config.max_page_size, 100);
        assert!(config.enable_total_count);
    }

    #[test]
    fn test_cursor_pagination_params() {
        let params = CursorPaginationParams {
            first: Some(10),
            after: Some("cursor123".to_string()),
            last: None,
            before: None,
        };

        assert_eq!(params.first, Some(10));
        assert_eq!(params.after, Some("cursor123".to_string()));
    }

    #[test]
    fn test_page_info() {
        let page_info = PageInfo {
            has_next_page: true,
            has_previous_page: false,
            start_cursor: Some("start".to_string()),
            end_cursor: Some("end".to_string()),
        };

        assert!(page_info.has_next_page);
        assert!(!page_info.has_previous_page);
    }
}
