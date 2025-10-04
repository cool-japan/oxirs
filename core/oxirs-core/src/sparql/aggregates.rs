//! SPARQL aggregate functions: COUNT, SUM, AVG, MIN, MAX

use crate::error::OxirsError;
use crate::model::{Literal, Term};
use crate::rdf_store::VariableBinding;
use crate::sparql::modifiers::compare_terms;
use crate::Result;

/// Aggregate function type
#[derive(Debug, Clone)]
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// Aggregate expression in SELECT clause
#[derive(Debug, Clone)]
pub struct AggregateExpression {
    pub function: AggregateFunction,
    pub variable: Option<String>, // None for COUNT(*)
    pub alias: String,
}

/// Extract aggregate expressions from SELECT clause
pub fn extract_aggregates(sparql: &str) -> Result<Vec<AggregateExpression>> {
    let mut aggregates = Vec::new();

    if let Some(select_start) = sparql.to_uppercase().find("SELECT") {
        if let Some(where_start) = sparql.to_uppercase().find("WHERE") {
            let select_clause = &sparql[select_start + 6..where_start];

            // Look for aggregate patterns like (COUNT(?var) AS ?alias)
            let mut pos = 0;
            while pos < select_clause.len() {
                if let Some(paren_start) = select_clause[pos..].find('(') {
                    let abs_pos = pos + paren_start;

                    // Find matching closing paren
                    if let Some(paren_end) = find_matching_paren(&select_clause[abs_pos..]) {
                        let expr = &select_clause[abs_pos..abs_pos + paren_end + 1];

                        // Check for COUNT, SUM, AVG, MIN, MAX
                        let expr_upper = expr.to_uppercase();
                        let function = if expr_upper.starts_with("(COUNT") {
                            Some(AggregateFunction::Count)
                        } else if expr_upper.starts_with("(SUM") {
                            Some(AggregateFunction::Sum)
                        } else if expr_upper.starts_with("(AVG") {
                            Some(AggregateFunction::Avg)
                        } else if expr_upper.starts_with("(MIN") {
                            Some(AggregateFunction::Min)
                        } else if expr_upper.starts_with("(MAX") {
                            Some(AggregateFunction::Max)
                        } else {
                            None
                        };

                        if let Some(func) = function {
                            // Extract variable from inside parentheses
                            let inner = &expr[1..expr.len() - 1]; // Remove outer parens

                            // Find the function name end
                            let func_name_end = if let Some(inner_paren) = inner.find('(') {
                                inner_paren
                            } else {
                                continue;
                            };

                            // Check for AS keyword inside the aggregate expression
                            let after_func = &inner[func_name_end..];
                            let after_func_upper = after_func.to_uppercase();
                            let (var_part, alias_part) =
                                if let Some(as_pos) = after_func_upper.find(" AS ") {
                                    (&after_func[1..as_pos], &after_func[as_pos + 4..])
                                } else {
                                    (&after_func[1..], "")
                                };

                            let args_trimmed = var_part.trim_end_matches(')').trim();

                            // Extract variable (or * for COUNT(*))
                            let variable = if args_trimmed == "*" {
                                None
                            } else if let Some(var_name) = args_trimmed.strip_prefix('?') {
                                Some(var_name.to_string())
                            } else {
                                Some(args_trimmed.to_string())
                            };

                            // Extract alias
                            let mut alias = String::from("aggregate");
                            if !alias_part.is_empty() {
                                for token in alias_part.split_whitespace() {
                                    if let Some(var_name) = token.strip_prefix('?') {
                                        alias = var_name.trim_end_matches(')').to_string();
                                        break;
                                    }
                                }
                            }

                            aggregates.push(AggregateExpression {
                                function: func,
                                variable,
                                alias,
                            });
                        }

                        pos = abs_pos + paren_end + 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    Ok(aggregates)
}

/// Find matching closing parenthesis
pub fn find_matching_paren(text: &str) -> Option<usize> {
    let mut paren_count = 1;
    let chars: Vec<char> = text.chars().collect();

    for (i, &ch) in chars.iter().enumerate().skip(1) {
        if ch == '(' {
            paren_count += 1;
        } else if ch == ')' {
            paren_count -= 1;
            if paren_count == 0 {
                return Some(i);
            }
        }
    }

    None
}

/// Apply aggregate functions to results
pub fn apply_aggregates(
    results: Vec<VariableBinding>,
    aggregates: &[AggregateExpression],
) -> Result<(Vec<VariableBinding>, Vec<String>)> {
    if aggregates.is_empty() {
        return Err(OxirsError::Query("No aggregates to apply".to_string()));
    }

    // For now, only support aggregates without GROUP BY
    // This means we compute one aggregate value across all results
    let mut aggregate_binding = VariableBinding::new();
    let mut result_variables = Vec::new();

    for agg_expr in aggregates {
        let value = match agg_expr.function {
            AggregateFunction::Count => {
                // COUNT(*) counts all bindings
                // COUNT(?var) counts bindings where var is bound
                let count = if let Some(var) = &agg_expr.variable {
                    results.iter().filter(|b| b.get(var).is_some()).count()
                } else {
                    results.len()
                };
                Term::from(Literal::new(count.to_string()))
            }
            AggregateFunction::Sum => {
                if let Some(var) = &agg_expr.variable {
                    let sum: f64 = results
                        .iter()
                        .filter_map(|b| b.get(var))
                        .filter_map(|term| {
                            if let Term::Literal(lit) = term {
                                lit.value().parse::<f64>().ok()
                            } else {
                                None
                            }
                        })
                        .sum();
                    Term::from(Literal::new(sum.to_string()))
                } else {
                    Term::from(Literal::new("0"))
                }
            }
            AggregateFunction::Avg => {
                if let Some(var) = &agg_expr.variable {
                    let values: Vec<f64> = results
                        .iter()
                        .filter_map(|b| b.get(var))
                        .filter_map(|term| {
                            if let Term::Literal(lit) = term {
                                lit.value().parse::<f64>().ok()
                            } else {
                                None
                            }
                        })
                        .collect();

                    let avg = if !values.is_empty() {
                        values.iter().sum::<f64>() / values.len() as f64
                    } else {
                        0.0
                    };
                    Term::from(Literal::new(avg.to_string()))
                } else {
                    Term::from(Literal::new("0"))
                }
            }
            AggregateFunction::Min => {
                if let Some(var) = &agg_expr.variable {
                    let min_val = results
                        .iter()
                        .filter_map(|b| b.get(var))
                        .min_by(|a, b| compare_terms(a, b));

                    if let Some(val) = min_val {
                        val.clone()
                    } else {
                        Term::from(Literal::new(""))
                    }
                } else {
                    Term::from(Literal::new(""))
                }
            }
            AggregateFunction::Max => {
                if let Some(var) = &agg_expr.variable {
                    let max_val = results
                        .iter()
                        .filter_map(|b| b.get(var))
                        .max_by(|a, b| compare_terms(a, b));

                    if let Some(val) = max_val {
                        val.clone()
                    } else {
                        Term::from(Literal::new(""))
                    }
                } else {
                    Term::from(Literal::new(""))
                }
            }
        };

        aggregate_binding.bind(agg_expr.alias.clone(), value);
        result_variables.push(agg_expr.alias.clone());
    }

    Ok((vec![aggregate_binding], result_variables))
}
