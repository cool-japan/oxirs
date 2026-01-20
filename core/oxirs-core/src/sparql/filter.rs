//! FILTER expression parsing and evaluation for SPARQL queries

use crate::error::OxirsError;
use crate::model::Term;
use crate::rdf_store::VariableBinding;
use crate::Result;

/// FILTER expression types
#[derive(Debug, Clone)]
pub enum FilterExpression {
    Comparison {
        left: FilterValue,
        op: ComparisonOp,
        right: FilterValue,
    },
    Bound {
        variable: String,
    },
}

/// Comparison operators for FILTER
#[derive(Debug, Clone)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessOrEqual,
    GreaterOrEqual,
}

/// Values in FILTER expressions
#[derive(Debug, Clone)]
pub enum FilterValue {
    Variable(String),
    Integer(i64),
    Float(f64),
    String(String),
    Uri(String),
}

/// Extract all FILTER expressions from a SPARQL query
pub fn extract_filter_expressions(sparql: &str) -> Result<Vec<FilterExpression>> {
    let mut filters = Vec::new();

    // Find all FILTER clauses
    let sparql_upper = sparql.to_uppercase();
    let mut search_pos = 0;

    while let Some(filter_pos) = sparql_upper[search_pos..].find("FILTER") {
        let abs_pos = search_pos + filter_pos;
        let after_filter = &sparql[abs_pos + 6..];

        // Find the opening parenthesis
        if let Some(paren_start) = after_filter.find('(') {
            // Find matching closing parenthesis
            let mut paren_count = 1;
            let mut paren_end = None;

            for (idx, ch) in after_filter[paren_start + 1..].char_indices() {
                if ch == '(' {
                    paren_count += 1;
                } else if ch == ')' {
                    paren_count -= 1;
                    if paren_count == 0 {
                        paren_end = Some(paren_start + 1 + idx);
                        break;
                    }
                }
            }

            if let Some(end) = paren_end {
                let filter_expr = after_filter[paren_start + 1..end].trim();
                filters.push(parse_filter_expression(filter_expr)?);
                search_pos = abs_pos + 6 + end;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    Ok(filters)
}

/// Parse a FILTER expression
fn parse_filter_expression(expr: &str) -> Result<FilterExpression> {
    let expr = expr.trim();

    // Try to parse comparison operators
    for (op_str, op) in &[
        ("<=", ComparisonOp::LessOrEqual),
        (">=", ComparisonOp::GreaterOrEqual),
        ("!=", ComparisonOp::NotEqual),
        ("<", ComparisonOp::LessThan),
        (">", ComparisonOp::GreaterThan),
        ("=", ComparisonOp::Equal),
    ] {
        if let Some(op_pos) = expr.find(op_str) {
            let left = expr[..op_pos].trim();
            let right = expr[op_pos + op_str.len()..].trim();

            return Ok(FilterExpression::Comparison {
                left: parse_filter_value(left)?,
                op: op.clone(),
                right: parse_filter_value(right)?,
            });
        }
    }

    // Try to parse function calls (e.g., BOUND(?var))
    if expr.to_uppercase().starts_with("BOUND(") {
        let var_part = expr[6..].trim_end_matches(')').trim();
        return Ok(FilterExpression::Bound {
            variable: var_part.to_string(),
        });
    }

    Err(OxirsError::Query(format!(
        "Unsupported FILTER expression: {}",
        expr
    )))
}

/// Parse a filter value (variable or literal)
fn parse_filter_value(value: &str) -> Result<FilterValue> {
    let value = value.trim();

    if value.starts_with('?') {
        Ok(FilterValue::Variable(value.to_string()))
    } else if value.starts_with('"') && value.ends_with('"') {
        // String literal
        Ok(FilterValue::String(value[1..value.len() - 1].to_string()))
    } else if let Ok(num) = value.parse::<i64>() {
        // Integer literal
        Ok(FilterValue::Integer(num))
    } else if let Ok(num) = value.parse::<f64>() {
        // Float literal
        Ok(FilterValue::Float(num))
    } else {
        // Try as URI
        if value.starts_with('<') && value.ends_with('>') {
            Ok(FilterValue::Uri(value[1..value.len() - 1].to_string()))
        } else {
            Ok(FilterValue::String(value.to_string()))
        }
    }
}

/// Evaluate FILTER expressions against a binding
pub fn evaluate_filters(binding: &VariableBinding, filters: &[FilterExpression]) -> bool {
    for filter in filters {
        if !evaluate_filter(binding, filter) {
            return false;
        }
    }
    true
}

/// Evaluate a single FILTER expression
fn evaluate_filter(binding: &VariableBinding, filter: &FilterExpression) -> bool {
    match filter {
        FilterExpression::Comparison { left, op, right } => {
            let left_val = resolve_filter_value(binding, left);
            let right_val = resolve_filter_value(binding, right);

            if let (Some(l), Some(r)) = (left_val, right_val) {
                compare_values(&l, op, &r)
            } else {
                false
            }
        }
        FilterExpression::Bound { variable } => binding.get(variable).is_some(),
    }
}

/// Resolve a filter value using the current binding
fn resolve_filter_value(binding: &VariableBinding, value: &FilterValue) -> Option<FilterValue> {
    match value {
        FilterValue::Variable(var) => {
            let var_name = var.strip_prefix('?').unwrap_or(var);
            if let Some(term) = binding.get(var_name) {
                // Convert Term to FilterValue
                match term {
                    Term::Literal(lit) => {
                        let val = lit.value();
                        // Try to parse as number
                        if let Ok(num) = val.parse::<i64>() {
                            Some(FilterValue::Integer(num))
                        } else if let Ok(num) = val.parse::<f64>() {
                            Some(FilterValue::Float(num))
                        } else {
                            Some(FilterValue::String(val.to_string()))
                        }
                    }
                    Term::NamedNode(node) => Some(FilterValue::Uri(node.as_str().to_string())),
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => Some(value.clone()),
    }
}

/// Compare two filter values
fn compare_values(left: &FilterValue, op: &ComparisonOp, right: &FilterValue) -> bool {
    match (left, right) {
        (FilterValue::Integer(l), FilterValue::Integer(r)) => match op {
            ComparisonOp::Equal => l == r,
            ComparisonOp::NotEqual => l != r,
            ComparisonOp::LessThan => l < r,
            ComparisonOp::GreaterThan => l > r,
            ComparisonOp::LessOrEqual => l <= r,
            ComparisonOp::GreaterOrEqual => l >= r,
        },
        (FilterValue::Float(l), FilterValue::Float(r)) => match op {
            ComparisonOp::Equal => (l - r).abs() < f64::EPSILON,
            ComparisonOp::NotEqual => (l - r).abs() >= f64::EPSILON,
            ComparisonOp::LessThan => l < r,
            ComparisonOp::GreaterThan => l > r,
            ComparisonOp::LessOrEqual => l <= r,
            ComparisonOp::GreaterOrEqual => l >= r,
        },
        (FilterValue::Integer(l), FilterValue::Float(r)) => {
            let l = *l as f64;
            match op {
                ComparisonOp::Equal => (l - r).abs() < f64::EPSILON,
                ComparisonOp::NotEqual => (l - r).abs() >= f64::EPSILON,
                ComparisonOp::LessThan => l < *r,
                ComparisonOp::GreaterThan => l > *r,
                ComparisonOp::LessOrEqual => l <= *r,
                ComparisonOp::GreaterOrEqual => l >= *r,
            }
        }
        (FilterValue::Float(l), FilterValue::Integer(r)) => {
            let r = *r as f64;
            match op {
                ComparisonOp::Equal => (l - r).abs() < f64::EPSILON,
                ComparisonOp::NotEqual => (l - r).abs() >= f64::EPSILON,
                ComparisonOp::LessThan => *l < r,
                ComparisonOp::GreaterThan => *l > r,
                ComparisonOp::LessOrEqual => *l <= r,
                ComparisonOp::GreaterOrEqual => *l >= r,
            }
        }
        (FilterValue::String(l), FilterValue::String(r)) => match op {
            ComparisonOp::Equal => l == r,
            ComparisonOp::NotEqual => l != r,
            ComparisonOp::LessThan => l < r,
            ComparisonOp::GreaterThan => l > r,
            ComparisonOp::LessOrEqual => l <= r,
            ComparisonOp::GreaterOrEqual => l >= r,
        },
        (FilterValue::Uri(l), FilterValue::Uri(r)) => match op {
            ComparisonOp::Equal => l == r,
            ComparisonOp::NotEqual => l != r,
            _ => false,
        },
        _ => false,
    }
}
