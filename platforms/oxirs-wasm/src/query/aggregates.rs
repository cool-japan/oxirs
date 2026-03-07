//! SPARQL 1.1 GROUP BY, HAVING, and aggregate functions
//!
//! Supports:
//! - COUNT(*), COUNT(?x), COUNT(DISTINCT ?x)
//! - SUM(?x), AVG(?x), MIN(?x), MAX(?x)
//! - GROUP_CONCAT(?x; separator="sep")
//! - SAMPLE(?x)
//! - GROUP BY ?var1 ?var2 ...
//! - HAVING (condition)

use crate::error::{WasmError, WasmResult};
use crate::query::filter::{extract_literal_value, parse_filter_inner, FilterExpr};
use std::collections::HashMap;

pub(crate) type Binding = HashMap<String, String>;

// ---------------------------------------------------------------------------
// Aggregate function enum
// ---------------------------------------------------------------------------

/// A SPARQL aggregate function
#[derive(Debug, Clone)]
pub(crate) enum AggregateFunc {
    /// COUNT(*) or COUNT(?x) or COUNT(DISTINCT ?x)
    Count {
        distinct: bool,
        variable: Option<String>,
    },
    /// SUM(?x)
    Sum { variable: String },
    /// AVG(?x)
    Avg { variable: String },
    /// MIN(?x)
    Min { variable: String },
    /// MAX(?x)
    Max { variable: String },
    /// GROUP_CONCAT(?x; separator="sep")
    GroupConcat { variable: String, separator: String },
    /// SAMPLE(?x) — returns one arbitrary value from the group
    Sample { variable: String },
}

// ---------------------------------------------------------------------------
// GROUP BY clause
// ---------------------------------------------------------------------------

/// GROUP BY clause with optional HAVING
#[derive(Debug, Clone)]
pub(crate) struct GroupByClause {
    /// Variables to group by
    pub(crate) variables: Vec<String>,
    /// Optional HAVING filter
    pub(crate) having: Option<HavingClause>,
}

/// HAVING clause — filters groups after aggregation
#[derive(Debug, Clone)]
pub(crate) struct HavingClause {
    pub(crate) condition: FilterExpr,
}

// ---------------------------------------------------------------------------
// Aggregate projection item — a SELECT expression like (COUNT(?o) AS ?count)
// ---------------------------------------------------------------------------

/// A single aggregate expression bound to an output variable
#[derive(Debug, Clone)]
pub(crate) struct AggregateProjection {
    /// Output variable name (e.g. "count")
    pub(crate) alias: String,
    /// The aggregate function
    pub(crate) func: AggregateFunc,
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// Evaluates GROUP BY + aggregate functions over a set of solution bindings
pub(crate) struct AggregateEvaluator;

impl AggregateEvaluator {
    /// Group `rows` by `group_by.variables`, apply each aggregate projection,
    /// then apply the HAVING filter if present.
    ///
    /// Returns one output row per group.
    pub(crate) fn apply(
        rows: &[Binding],
        group_by: &GroupByClause,
        aggregates: &[AggregateProjection],
    ) -> WasmResult<Vec<Binding>> {
        // Group rows
        let groups = Self::group_rows(rows, &group_by.variables);

        // Build one output binding per group
        let mut output: Vec<Binding> = Vec::new();

        for (group_key_values, group_rows) in &groups {
            let mut out_binding: Binding = HashMap::new();

            // Copy group-by variable bindings into output
            for (i, var) in group_by.variables.iter().enumerate() {
                if let Some(val) = group_key_values.get(i) {
                    out_binding.insert(var.clone(), val.clone());
                }
            }

            // Evaluate each aggregate function
            for proj in aggregates {
                let value = Self::eval_aggregate(group_rows, &proj.func);
                out_binding.insert(proj.alias.clone(), value);
            }

            output.push(out_binding);
        }

        // Apply HAVING filter
        if let Some(having) = &group_by.having {
            output.retain(|b| having.condition.evaluate(b));
        }

        Ok(output)
    }

    /// Group rows by the specified variables, returning a map from key-tuple to group rows.
    /// Uses a Vec to preserve insertion order (important for determinism).
    fn group_rows(rows: &[Binding], variables: &[String]) -> Vec<(Vec<String>, Vec<Binding>)> {
        // We use a Vec of (key, rows) pairs to preserve order; a secondary index
        // maps key -> position in the vec for O(1) lookup.
        let mut groups: Vec<(Vec<String>, Vec<Binding>)> = Vec::new();
        let mut key_index: HashMap<Vec<String>, usize> = HashMap::new();

        for row in rows {
            // Build group key from the GROUP BY variables
            let key: Vec<String> = if variables.is_empty() {
                // No GROUP BY — all rows form a single group
                vec![]
            } else {
                variables
                    .iter()
                    .map(|v| row.get(v).cloned().unwrap_or_default())
                    .collect()
            };

            if let Some(&pos) = key_index.get(&key) {
                groups[pos].1.push(row.clone());
            } else {
                let pos = groups.len();
                key_index.insert(key.clone(), pos);
                groups.push((key, vec![row.clone()]));
            }
        }

        groups
    }

    /// Evaluate a single aggregate function over one group of rows
    fn eval_aggregate(group: &[Binding], func: &AggregateFunc) -> String {
        match func {
            AggregateFunc::Count { distinct, variable } => {
                let count = if let Some(var) = variable {
                    // COUNT(?x) — count non-null bindings
                    if *distinct {
                        let mut seen: std::collections::HashSet<String> =
                            std::collections::HashSet::new();
                        for row in group {
                            if let Some(val) = row.get(var) {
                                seen.insert(val.clone());
                            }
                        }
                        seen.len()
                    } else {
                        group.iter().filter(|r| r.contains_key(var)).count()
                    }
                } else {
                    // COUNT(*) — count all rows
                    if *distinct {
                        let mut seen: std::collections::HashSet<String> =
                            std::collections::HashSet::new();
                        for row in group {
                            let mut pairs: Vec<_> = row.iter().collect();
                            pairs.sort_by_key(|(k, _)| k.as_str());
                            seen.insert(format!("{:?}", pairs));
                        }
                        seen.len()
                    } else {
                        group.len()
                    }
                };
                count.to_string()
            }

            AggregateFunc::Sum { variable } => {
                let sum: f64 = group
                    .iter()
                    .filter_map(|r| r.get(variable))
                    .filter_map(|v| extract_literal_value(v).parse::<f64>().ok())
                    .sum();
                format_number(sum)
            }

            AggregateFunc::Avg { variable } => {
                let values: Vec<f64> = group
                    .iter()
                    .filter_map(|r| r.get(variable))
                    .filter_map(|v| extract_literal_value(v).parse::<f64>().ok())
                    .collect();
                if values.is_empty() {
                    "0".to_string()
                } else {
                    let avg = values.iter().sum::<f64>() / values.len() as f64;
                    format_number(avg)
                }
            }

            AggregateFunc::Min { variable } => {
                let mut string_min: Option<String> = None;
                let mut num_min: Option<f64> = None;
                let mut all_numeric = true;

                for row in group {
                    if let Some(val) = row.get(variable) {
                        let raw = extract_literal_value(val);
                        if let Ok(n) = raw.parse::<f64>() {
                            let cur = num_min.get_or_insert(n);
                            if n < *cur {
                                *cur = n;
                            }
                        } else {
                            all_numeric = false;
                            match &string_min {
                                None => string_min = Some(val.clone()),
                                Some(cur) => {
                                    if val < cur {
                                        string_min = Some(val.clone());
                                    }
                                }
                            }
                        }
                    }
                }

                if all_numeric {
                    num_min.map(format_number).unwrap_or_default()
                } else {
                    string_min.unwrap_or_default()
                }
            }

            AggregateFunc::Max { variable } => {
                let mut string_max: Option<String> = None;
                let mut num_max: Option<f64> = None;
                let mut all_numeric = true;

                for row in group {
                    if let Some(val) = row.get(variable) {
                        let raw = extract_literal_value(val);
                        if let Ok(n) = raw.parse::<f64>() {
                            let cur = num_max.get_or_insert(n);
                            if n > *cur {
                                *cur = n;
                            }
                        } else {
                            all_numeric = false;
                            match &string_max {
                                None => string_max = Some(val.clone()),
                                Some(cur) => {
                                    if val > cur {
                                        string_max = Some(val.clone());
                                    }
                                }
                            }
                        }
                    }
                }

                if all_numeric {
                    num_max.map(format_number).unwrap_or_default()
                } else {
                    string_max.unwrap_or_default()
                }
            }

            AggregateFunc::GroupConcat {
                variable,
                separator,
            } => {
                let parts: Vec<String> = group
                    .iter()
                    .filter_map(|r| r.get(variable))
                    .map(|v| extract_literal_value(v).to_string())
                    .collect();
                parts.join(separator)
            }

            AggregateFunc::Sample { variable } => {
                // Return the first non-null value
                group
                    .iter()
                    .find_map(|r| r.get(variable).cloned())
                    .unwrap_or_default()
            }
        }
    }
}

/// Format a floating-point number: prefer integer representation when exact
fn format_number(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{}", n)
    }
}

// ---------------------------------------------------------------------------
// Parser helpers — parse SELECT aggregate expressions and GROUP BY / HAVING
// ---------------------------------------------------------------------------

/// Parse a parenthesized aggregate expression like:
/// `(COUNT(?o) AS ?count)` or `(SUM(?val) AS ?total)`
///
/// Returns `Some((alias, func))` on success, `None` if not an aggregate.
pub(crate) fn parse_aggregate_expr(s: &str) -> Option<AggregateProjection> {
    // Must be wrapped in outer parens: (FUNC(...) AS ?alias)
    let s = s.trim();

    // Strip outer parentheses if present
    let inner = if s.starts_with('(') && s.ends_with(')') {
        &s[1..s.len() - 1]
    } else {
        return None;
    };

    // Find " AS " (case-insensitive)
    let upper = inner.to_uppercase();
    let as_pos = upper.find(" AS ")?;
    let func_part = inner[..as_pos].trim();
    let alias_part = inner[as_pos + 4..].trim();

    // Parse alias
    let alias = alias_part.trim_start_matches(['?', '$']).to_string();
    if alias.is_empty() {
        return None;
    }

    // Parse aggregate function
    let func = parse_aggregate_func(func_part)?;

    Some(AggregateProjection { alias, func })
}

/// Parse an aggregate function call like `COUNT(?x)`, `COUNT(DISTINCT ?x)`,
/// `SUM(?x)`, `AVG(?x)`, `MIN(?x)`, `MAX(?x)`,
/// `GROUP_CONCAT(?x; separator=", ")`, `SAMPLE(?x)`
fn parse_aggregate_func(s: &str) -> Option<AggregateFunc> {
    let s = s.trim();
    let upper = s.to_uppercase();

    // Find the opening paren
    let paren_pos = s.find('(')?;
    let func_name = s[..paren_pos].trim().to_uppercase();

    // Extract content inside parens (strip outermost)
    let args_raw = s[paren_pos + 1..].trim_end_matches(')').trim();

    match func_name.as_str() {
        "COUNT" => {
            if args_raw.trim() == "*" {
                Some(AggregateFunc::Count {
                    distinct: false,
                    variable: None,
                })
            } else {
                let args_upper = args_raw.to_uppercase();
                let distinct = args_upper.starts_with("DISTINCT");
                let var_part = if distinct {
                    args_raw[8..].trim()
                } else {
                    args_raw
                };
                let variable = if var_part.is_empty() || var_part == "*" {
                    None
                } else {
                    Some(var_part.trim_start_matches(['?', '$']).to_string())
                };
                Some(AggregateFunc::Count { distinct, variable })
            }
        }

        "SUM" => {
            let variable = args_raw.trim_start_matches(['?', '$']).to_string();
            Some(AggregateFunc::Sum { variable })
        }

        "AVG" => {
            let variable = args_raw.trim_start_matches(['?', '$']).to_string();
            Some(AggregateFunc::Avg { variable })
        }

        "MIN" => {
            let variable = args_raw.trim_start_matches(['?', '$']).to_string();
            Some(AggregateFunc::Min { variable })
        }

        "MAX" => {
            let variable = args_raw.trim_start_matches(['?', '$']).to_string();
            Some(AggregateFunc::Max { variable })
        }

        "GROUP_CONCAT" => {
            // GROUP_CONCAT(?var; separator="sep") or GROUP_CONCAT(?var)
            let separator = if let Some(sep_pos) = args_raw.find(';') {
                let sep_part = &args_raw[sep_pos + 1..];
                // Look for separator="..." or separator='...'
                extract_separator(sep_part).unwrap_or_else(|| " ".to_string())
            } else {
                " ".to_string()
            };
            let var_part = if let Some(semi) = args_raw.find(';') {
                args_raw[..semi].trim()
            } else {
                args_raw.trim()
            };
            let variable = var_part.trim_start_matches(['?', '$']).to_string();
            Some(AggregateFunc::GroupConcat {
                variable,
                separator,
            })
        }

        "SAMPLE" => {
            let variable = args_raw.trim_start_matches(['?', '$']).to_string();
            Some(AggregateFunc::Sample { variable })
        }

        _ => {
            let _ = upper;
            None
        }
    }
}

/// Extract separator value from `separator="..."` or `separator='...'`
fn extract_separator(s: &str) -> Option<String> {
    let s = s.trim();
    let upper = s.to_uppercase();
    let sep_kw = upper.find("SEPARATOR")?;
    let rest = &s[sep_kw + 9..]; // skip "SEPARATOR"
    let rest = rest.trim().trim_start_matches('=').trim();

    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        Some(stripped[..end].to_string())
    } else if let Some(stripped) = rest.strip_prefix('\'') {
        let end = stripped.find('\'')?;
        Some(stripped[..end].to_string())
    } else {
        None
    }
}

/// Parse a GROUP BY clause from the text after the WHERE body.
///
/// Recognizes:
/// ```text
/// GROUP BY ?var1 ?var2 ...
/// GROUP BY ?var HAVING (expr)
/// ```
pub(crate) fn parse_group_by(after_where: &str) -> Option<GroupByClause> {
    let upper = after_where.to_uppercase();
    let group_pos = upper.find("GROUP")?;
    let rest = &after_where[group_pos + 5..];
    let upper_rest = rest.to_uppercase();
    let by_pos = upper_rest.find("BY")?;
    let vars_str = &rest[by_pos + 2..];

    // Find end of GROUP BY: HAVING, ORDER BY, LIMIT, OFFSET, or end
    let upper_vars = vars_str.to_uppercase();
    let end = ["HAVING", "ORDER", "LIMIT", "OFFSET"]
        .iter()
        .filter_map(|kw| upper_vars.find(kw))
        .min()
        .unwrap_or(vars_str.len());

    let vars_segment = &vars_str[..end];
    let variables: Vec<String> = vars_segment
        .split_whitespace()
        .filter(|t| t.starts_with('?') || t.starts_with('$'))
        .map(|t| t.trim_start_matches(['?', '$']).to_string())
        .collect();

    // Parse HAVING clause if present
    let having = parse_having(vars_str);

    Some(GroupByClause { variables, having })
}

/// Parse HAVING clause from the text after GROUP BY variables.
fn parse_having(s: &str) -> Option<HavingClause> {
    let upper = s.to_uppercase();
    let having_pos = upper.find("HAVING")?;
    let rest = &s[having_pos + 6..].trim_start();

    // HAVING must be followed by (expr)
    if !rest.starts_with('(') {
        return None;
    }

    // Find the closing paren — handle nested parens
    let chars: Vec<char> = rest.chars().collect();
    let mut depth = 0usize;
    let mut end = 0usize;
    for (i, &c) in chars.iter().enumerate() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }

    let expr_str: String = chars[1..end].iter().collect();
    let condition = parse_filter_inner(&expr_str)?;
    Some(HavingClause { condition })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_binding(pairs: &[(&str, &str)]) -> Binding {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // ---- parse_aggregate_expr ----

    #[test]
    fn test_parse_count_star() {
        let proj = parse_aggregate_expr("(COUNT(*) AS ?count)").expect("parse");
        assert_eq!(proj.alias, "count");
        assert!(matches!(
            proj.func,
            AggregateFunc::Count {
                distinct: false,
                variable: None
            }
        ));
    }

    #[test]
    fn test_parse_count_var() {
        let proj = parse_aggregate_expr("(COUNT(?o) AS ?cnt)").expect("parse");
        assert_eq!(proj.alias, "cnt");
        if let AggregateFunc::Count {
            distinct: false,
            variable: Some(v),
        } = proj.func
        {
            assert_eq!(v, "o");
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_parse_count_distinct() {
        let proj = parse_aggregate_expr("(COUNT(DISTINCT ?x) AS ?ux)").expect("parse");
        assert_eq!(proj.alias, "ux");
        if let AggregateFunc::Count {
            distinct: true,
            variable: Some(v),
        } = proj.func
        {
            assert_eq!(v, "x");
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_parse_sum() {
        let proj = parse_aggregate_expr("(SUM(?val) AS ?total)").expect("parse");
        assert_eq!(proj.alias, "total");
        assert!(matches!(proj.func, AggregateFunc::Sum { .. }));
    }

    #[test]
    fn test_parse_avg() {
        let proj = parse_aggregate_expr("(AVG(?val) AS ?avg)").expect("parse");
        assert_eq!(proj.alias, "avg");
        assert!(matches!(proj.func, AggregateFunc::Avg { .. }));
    }

    #[test]
    fn test_parse_min() {
        let proj = parse_aggregate_expr("(MIN(?val) AS ?min)").expect("parse");
        assert_eq!(proj.alias, "min");
        assert!(matches!(proj.func, AggregateFunc::Min { .. }));
    }

    #[test]
    fn test_parse_max() {
        let proj = parse_aggregate_expr("(MAX(?val) AS ?max)").expect("parse");
        assert_eq!(proj.alias, "max");
        assert!(matches!(proj.func, AggregateFunc::Max { .. }));
    }

    #[test]
    fn test_parse_group_concat() {
        let proj = parse_aggregate_expr("(GROUP_CONCAT(?name; separator=\", \") AS ?names)")
            .expect("parse");
        assert_eq!(proj.alias, "names");
        if let AggregateFunc::GroupConcat {
            variable,
            separator,
        } = proj.func
        {
            assert_eq!(variable, "name");
            assert_eq!(separator, ", ");
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_parse_sample() {
        let proj = parse_aggregate_expr("(SAMPLE(?x) AS ?s)").expect("parse");
        assert_eq!(proj.alias, "s");
        assert!(matches!(proj.func, AggregateFunc::Sample { .. }));
    }

    // ---- AggregateEvaluator::apply ----

    #[test]
    fn test_count_star_no_group_by() {
        let rows = vec![
            make_binding(&[("s", "http://a"), ("o", "http://x")]),
            make_binding(&[("s", "http://b"), ("o", "http://y")]),
            make_binding(&[("s", "http://c"), ("o", "http://z")]),
        ];
        let group_by = GroupByClause {
            variables: vec![],
            having: None,
        };
        let aggregates = vec![AggregateProjection {
            alias: "count".to_string(),
            func: AggregateFunc::Count {
                distinct: false,
                variable: None,
            },
        }];
        let result = AggregateEvaluator::apply(&rows, &group_by, &aggregates).expect("eval");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("count").expect("count"), "3");
    }

    #[test]
    fn test_count_by_subject() {
        let rows = vec![
            make_binding(&[("s", "http://a"), ("o", "http://x")]),
            make_binding(&[("s", "http://a"), ("o", "http://y")]),
            make_binding(&[("s", "http://b"), ("o", "http://z")]),
        ];
        let group_by = GroupByClause {
            variables: vec!["s".to_string()],
            having: None,
        };
        let aggregates = vec![AggregateProjection {
            alias: "count".to_string(),
            func: AggregateFunc::Count {
                distinct: false,
                variable: None,
            },
        }];
        let mut result = AggregateEvaluator::apply(&rows, &group_by, &aggregates).expect("eval");
        result.sort_by_key(|r| r.get("s").cloned().unwrap_or_default());
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].get("count").expect("count"), "2"); // http://a
        assert_eq!(result[1].get("count").expect("count"), "1"); // http://b
    }

    #[test]
    fn test_sum_aggregate() {
        let rows = vec![
            make_binding(&[("type", "A"), ("val", "\"10\"")]),
            make_binding(&[("type", "A"), ("val", "\"20\"")]),
            make_binding(&[("type", "B"), ("val", "\"5\"")]),
        ];
        let group_by = GroupByClause {
            variables: vec!["type".to_string()],
            having: None,
        };
        let aggregates = vec![AggregateProjection {
            alias: "total".to_string(),
            func: AggregateFunc::Sum {
                variable: "val".to_string(),
            },
        }];
        let mut result = AggregateEvaluator::apply(&rows, &group_by, &aggregates).expect("eval");
        result.sort_by_key(|r| r.get("type").cloned().unwrap_or_default());
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].get("total").expect("total"), "30"); // type A: 10+20
        assert_eq!(result[1].get("total").expect("total"), "5"); // type B: 5
    }

    #[test]
    fn test_avg_aggregate() {
        let rows = vec![
            make_binding(&[("g", "X"), ("v", "\"10\"")]),
            make_binding(&[("g", "X"), ("v", "\"20\"")]),
            make_binding(&[("g", "X"), ("v", "\"30\"")]),
        ];
        let group_by = GroupByClause {
            variables: vec!["g".to_string()],
            having: None,
        };
        let aggregates = vec![AggregateProjection {
            alias: "avg".to_string(),
            func: AggregateFunc::Avg {
                variable: "v".to_string(),
            },
        }];
        let result = AggregateEvaluator::apply(&rows, &group_by, &aggregates).expect("eval");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("avg").expect("avg"), "20");
    }

    #[test]
    fn test_min_aggregate_numeric() {
        let rows = vec![
            make_binding(&[("v", "\"30\"")]),
            make_binding(&[("v", "\"10\"")]),
            make_binding(&[("v", "\"20\"")]),
        ];
        let group_by = GroupByClause {
            variables: vec![],
            having: None,
        };
        let aggregates = vec![AggregateProjection {
            alias: "m".to_string(),
            func: AggregateFunc::Min {
                variable: "v".to_string(),
            },
        }];
        let result = AggregateEvaluator::apply(&rows, &group_by, &aggregates).expect("eval");
        assert_eq!(result[0].get("m").expect("m"), "10");
    }

    #[test]
    fn test_max_aggregate_numeric() {
        let rows = vec![
            make_binding(&[("v", "\"30\"")]),
            make_binding(&[("v", "\"10\"")]),
            make_binding(&[("v", "\"20\"")]),
        ];
        let group_by = GroupByClause {
            variables: vec![],
            having: None,
        };
        let aggregates = vec![AggregateProjection {
            alias: "m".to_string(),
            func: AggregateFunc::Max {
                variable: "v".to_string(),
            },
        }];
        let result = AggregateEvaluator::apply(&rows, &group_by, &aggregates).expect("eval");
        assert_eq!(result[0].get("m").expect("m"), "30");
    }

    #[test]
    fn test_group_concat_aggregate() {
        let rows = vec![
            make_binding(&[("type", "A"), ("name", "\"Alice\"")]),
            make_binding(&[("type", "A"), ("name", "\"Bob\"")]),
        ];
        let group_by = GroupByClause {
            variables: vec!["type".to_string()],
            having: None,
        };
        let aggregates = vec![AggregateProjection {
            alias: "names".to_string(),
            func: AggregateFunc::GroupConcat {
                variable: "name".to_string(),
                separator: ", ".to_string(),
            },
        }];
        let result = AggregateEvaluator::apply(&rows, &group_by, &aggregates).expect("eval");
        assert_eq!(result.len(), 1);
        let names = result[0].get("names").expect("names");
        assert!(names.contains("Alice"));
        assert!(names.contains("Bob"));
        assert!(names.contains(", "));
    }

    #[test]
    fn test_sample_aggregate() {
        let rows = vec![
            make_binding(&[("g", "X"), ("v", "\"one\"")]),
            make_binding(&[("g", "X"), ("v", "\"two\"")]),
        ];
        let group_by = GroupByClause {
            variables: vec!["g".to_string()],
            having: None,
        };
        let aggregates = vec![AggregateProjection {
            alias: "sample".to_string(),
            func: AggregateFunc::Sample {
                variable: "v".to_string(),
            },
        }];
        let result = AggregateEvaluator::apply(&rows, &group_by, &aggregates).expect("eval");
        assert_eq!(result.len(), 1);
        // SAMPLE returns one of the values
        let val = result[0].get("sample").expect("sample");
        assert!(val == "\"one\"" || val == "\"two\"");
    }

    #[test]
    fn test_having_filter() {
        use crate::query::filter::{FilterExpr, FilterTerm};
        let rows = vec![
            make_binding(&[("s", "http://a"), ("o", "http://x")]),
            make_binding(&[("s", "http://a"), ("o", "http://y")]),
            make_binding(&[("s", "http://b"), ("o", "http://z")]),
        ];
        // HAVING (count > 1) — should keep only group http://a (count=2)
        let having_expr = FilterExpr::GreaterThan {
            lhs: Box::new(FilterTerm::Variable("count".to_string())),
            rhs: Box::new(FilterTerm::Literal("1".to_string())),
        };
        let group_by = GroupByClause {
            variables: vec!["s".to_string()],
            having: Some(HavingClause {
                condition: having_expr,
            }),
        };
        let aggregates = vec![AggregateProjection {
            alias: "count".to_string(),
            func: AggregateFunc::Count {
                distinct: false,
                variable: None,
            },
        }];
        let result = AggregateEvaluator::apply(&rows, &group_by, &aggregates).expect("eval");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("s").expect("s"), "http://a");
    }

    #[test]
    fn test_count_distinct_var() {
        let rows = vec![
            make_binding(&[("s", "http://a"), ("p", "http://p1")]),
            make_binding(&[("s", "http://a"), ("p", "http://p1")]), // duplicate
            make_binding(&[("s", "http://a"), ("p", "http://p2")]),
        ];
        let group_by = GroupByClause {
            variables: vec!["s".to_string()],
            having: None,
        };
        let aggregates = vec![AggregateProjection {
            alias: "ucount".to_string(),
            func: AggregateFunc::Count {
                distinct: true,
                variable: Some("p".to_string()),
            },
        }];
        let result = AggregateEvaluator::apply(&rows, &group_by, &aggregates).expect("eval");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("ucount").expect("ucount"), "2"); // p1, p2
    }

    #[test]
    fn test_parse_group_by_basic() {
        let after_where = " GROUP BY ?s ?p LIMIT 10";
        let group_by = parse_group_by(after_where).expect("parse");
        assert_eq!(group_by.variables, vec!["s", "p"]);
        assert!(group_by.having.is_none());
    }

    #[test]
    fn test_parse_group_by_with_having() {
        let after_where = " GROUP BY ?s HAVING (COUNT(*) > 1) LIMIT 10";
        let group_by = parse_group_by(after_where).expect("parse");
        assert_eq!(group_by.variables, vec!["s"]);
        assert!(group_by.having.is_some());
    }

    #[test]
    fn test_format_number_integer() {
        assert_eq!(format_number(42.0), "42");
        assert_eq!(format_number(0.0), "0");
        assert_eq!(format_number(-5.0), "-5");
    }

    #[test]
    fn test_format_number_float() {
        assert_eq!(
            format_number(std::f64::consts::PI),
            format!("{}", std::f64::consts::PI)
        );
    }

    #[test]
    fn test_extract_separator_double_quote() {
        let result = extract_separator("separator=\", \"");
        assert_eq!(result, Some(", ".to_string()));
    }

    #[test]
    fn test_extract_separator_single_quote() {
        let result = extract_separator("separator=', '");
        assert_eq!(result, Some(", ".to_string()));
    }

    #[test]
    fn test_min_aggregate_string() {
        let rows = vec![
            make_binding(&[("v", "\"banana\"")]),
            make_binding(&[("v", "\"apple\"")]),
            make_binding(&[("v", "\"cherry\"")]),
        ];
        let group_by = GroupByClause {
            variables: vec![],
            having: None,
        };
        let aggregates = vec![AggregateProjection {
            alias: "m".to_string(),
            func: AggregateFunc::Min {
                variable: "v".to_string(),
            },
        }];
        let result = AggregateEvaluator::apply(&rows, &group_by, &aggregates).expect("eval");
        assert_eq!(result[0].get("m").expect("m"), "\"apple\"");
    }

    #[test]
    fn test_max_aggregate_string() {
        let rows = vec![
            make_binding(&[("v", "\"banana\"")]),
            make_binding(&[("v", "\"apple\"")]),
            make_binding(&[("v", "\"cherry\"")]),
        ];
        let group_by = GroupByClause {
            variables: vec![],
            having: None,
        };
        let aggregates = vec![AggregateProjection {
            alias: "m".to_string(),
            func: AggregateFunc::Max {
                variable: "v".to_string(),
            },
        }];
        let result = AggregateEvaluator::apply(&rows, &group_by, &aggregates).expect("eval");
        assert_eq!(result[0].get("m").expect("m"), "\"cherry\"");
    }

    // -----------------------------------------------------------------------
    // End-to-end integration tests via execute_select
    // -----------------------------------------------------------------------

    fn make_person_store() -> crate::store::OxiRSStore {
        let mut store = crate::store::OxiRSStore::new();
        // Three people
        store.insert("http://alice", "http://type", "http://Person");
        store.insert("http://bob", "http://type", "http://Person");
        store.insert("http://carol", "http://type", "http://Person");
        // Ages
        store.insert("http://alice", "http://age", "\"30\"");
        store.insert("http://bob", "http://age", "\"25\"");
        store.insert("http://carol", "http://age", "\"35\"");
        // Names
        store.insert("http://alice", "http://name", "\"Alice\"");
        store.insert("http://bob", "http://name", "\"Bob\"");
        store.insert("http://carol", "http://name", "\"Carol\"");
        // Groups
        store.insert("http://alice", "http://group", "\"A\"");
        store.insert("http://bob", "http://group", "\"A\"");
        store.insert("http://carol", "http://group", "\"B\"");
        store
    }

    #[test]
    fn test_e2e_count_star_no_group_by() {
        let store = make_person_store();
        let sparql = "SELECT (COUNT(*) AS ?total) WHERE { ?s <http://type> <http://Person> }";
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("total").expect("total"), "3");
    }

    #[test]
    fn test_e2e_count_by_group() {
        let store = make_person_store();
        let sparql = "SELECT ?g (COUNT(?s) AS ?cnt) WHERE { ?s <http://group> ?g } GROUP BY ?g";
        let mut results = crate::query::execute_select(sparql, &store).expect("execute");
        results.sort_by_key(|r| r.get("g").cloned().unwrap_or_default());
        assert_eq!(results.len(), 2);
        // Group A: alice + bob = 2
        assert_eq!(results[0].get("cnt").expect("cnt"), "2");
        // Group B: carol = 1
        assert_eq!(results[1].get("cnt").expect("cnt"), "1");
    }

    #[test]
    fn test_e2e_sum_by_group() {
        let store = make_person_store();
        let sparql =
            "SELECT ?g (SUM(?age) AS ?total) WHERE { ?s <http://group> ?g . ?s <http://age> ?age } GROUP BY ?g";
        let mut results = crate::query::execute_select(sparql, &store).expect("execute");
        results.sort_by_key(|r| r.get("g").cloned().unwrap_or_default());
        assert_eq!(results.len(), 2);
        // Group A: 30 + 25 = 55
        assert_eq!(results[0].get("total").expect("total"), "55");
        // Group B: 35
        assert_eq!(results[1].get("total").expect("total"), "35");
    }

    #[test]
    fn test_e2e_avg_by_group() {
        let store = make_person_store();
        let sparql =
            "SELECT ?g (AVG(?age) AS ?avg) WHERE { ?s <http://group> ?g . ?s <http://age> ?age } GROUP BY ?g";
        let mut results = crate::query::execute_select(sparql, &store).expect("execute");
        results.sort_by_key(|r| r.get("g").cloned().unwrap_or_default());
        assert_eq!(results.len(), 2);
        // Group A: (30+25)/2 = 27.5
        let avg_a: f64 = results[0]
            .get("avg")
            .expect("avg")
            .parse()
            .expect("parse f64");
        assert!((avg_a - 27.5).abs() < 0.001, "expected 27.5, got {}", avg_a);
        // Group B: 35
        assert_eq!(results[1].get("avg").expect("avg"), "35");
    }

    #[test]
    fn test_e2e_min_age() {
        let store = make_person_store();
        let sparql = "SELECT (MIN(?age) AS ?min_age) WHERE { ?s <http://age> ?age }";
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("min_age").expect("min_age"), "25");
    }

    #[test]
    fn test_e2e_max_age() {
        let store = make_person_store();
        let sparql = "SELECT (MAX(?age) AS ?max_age) WHERE { ?s <http://age> ?age }";
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("max_age").expect("max_age"), "35");
    }

    #[test]
    fn test_e2e_group_concat_names_by_group() {
        let store = make_person_store();
        let sparql = r#"SELECT ?g (GROUP_CONCAT(?name; separator=", ") AS ?names) WHERE { ?s <http://group> ?g . ?s <http://name> ?name } GROUP BY ?g"#;
        let mut results = crate::query::execute_select(sparql, &store).expect("execute");
        results.sort_by_key(|r| r.get("g").cloned().unwrap_or_default());
        assert_eq!(results.len(), 2);
        // Group A: "Alice, Bob" or "Bob, Alice"
        let names_a = results[0].get("names").expect("names_a");
        assert!(names_a.contains("Alice"));
        assert!(names_a.contains("Bob"));
        // Group B: "Carol"
        let names_b = results[1].get("names").expect("names_b");
        assert!(names_b.contains("Carol"));
    }

    #[test]
    fn test_e2e_sample_returns_value() {
        let store = make_person_store();
        let sparql = "SELECT (SAMPLE(?name) AS ?sample_name) WHERE { ?s <http://name> ?name }";
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 1);
        let val = results[0].get("sample_name").expect("sample_name");
        // Should be one of the names
        assert!(
            val == "\"Alice\"" || val == "\"Bob\"" || val == "\"Carol\"",
            "unexpected sample: {}",
            val
        );
    }

    #[test]
    fn test_e2e_having_count_gt_1() {
        let store = make_person_store();
        // Only group A has count > 1; use the alias ?cnt in HAVING
        let sparql =
            "SELECT ?g (COUNT(?s) AS ?cnt) WHERE { ?s <http://group> ?g } GROUP BY ?g HAVING (?cnt > 1)";
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("g").expect("g"), "\"A\"");
    }

    #[test]
    fn test_e2e_count_distinct_predicates() {
        let mut store = crate::store::OxiRSStore::new();
        store.insert("http://alice", "http://p", "http://x");
        store.insert("http://alice", "http://p", "http://y"); // same predicate, different object
        store.insert("http://alice", "http://q", "http://z"); // different predicate
        let sparql = "SELECT (COUNT(DISTINCT ?p) AS ?num_preds) WHERE { <http://alice> ?p ?o }";
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("num_preds").expect("num_preds"), "2");
    }

    #[test]
    fn test_e2e_count_with_filter() {
        let store = make_person_store();
        // Count people in group A (filter applied before aggregation)
        let sparql =
            r#"SELECT (COUNT(*) AS ?cnt) WHERE { ?s <http://group> ?g . FILTER(?g = "\"A\"") }"#;
        // Simpler without escaped quotes:
        let sparql2 = "SELECT (COUNT(*) AS ?cnt) WHERE { ?s <http://group> \"A\" }";
        let results = crate::query::execute_select(sparql2, &store).expect("execute");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("cnt").expect("cnt"), "2");
    }

    #[test]
    fn test_e2e_aggregate_with_order_by() {
        let store = make_person_store();
        // Get groups ordered by count descending
        let sparql =
            "SELECT ?g (COUNT(?s) AS ?cnt) WHERE { ?s <http://group> ?g } GROUP BY ?g ORDER BY DESC(?cnt)";
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 2);
        // Group A (count=2) should come first
        let counts: Vec<&str> = results
            .iter()
            .filter_map(|r| r.get("cnt").map(|s| s.as_str()))
            .collect();
        assert_eq!(counts[0], "2");
        assert_eq!(counts[1], "1");
    }

    #[test]
    fn test_e2e_aggregate_with_limit() {
        let store = make_person_store();
        let sparql =
            "SELECT ?g (COUNT(?s) AS ?cnt) WHERE { ?s <http://group> ?g } GROUP BY ?g LIMIT 1";
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_e2e_multiple_aggregates_in_select() {
        let store = make_person_store();
        let sparql =
            "SELECT (COUNT(*) AS ?cnt) (SUM(?age) AS ?total) WHERE { ?s <http://age> ?age }";
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("cnt").expect("cnt"), "3");
        // 30 + 25 + 35 = 90
        assert_eq!(results[0].get("total").expect("total"), "90");
    }

    #[test]
    fn test_e2e_group_by_two_variables() {
        let mut store = crate::store::OxiRSStore::new();
        store.insert("http://a1", "http://cat", "\"X\"");
        store.insert("http://a1", "http://sub", "\"Y\"");
        store.insert("http://a2", "http://cat", "\"X\"");
        store.insert("http://a2", "http://sub", "\"Y\"");
        store.insert("http://b1", "http://cat", "\"X\"");
        store.insert("http://b1", "http://sub", "\"Z\"");
        let sparql = "SELECT ?cat ?sub (COUNT(?s) AS ?cnt) WHERE { ?s <http://cat> ?cat . ?s <http://sub> ?sub } GROUP BY ?cat ?sub";
        let mut results = crate::query::execute_select(sparql, &store).expect("execute");
        results.sort_by_key(|r| {
            format!(
                "{}-{}",
                r.get("cat").cloned().unwrap_or_default(),
                r.get("sub").cloned().unwrap_or_default()
            )
        });
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].get("cnt").expect("cnt"), "2"); // X-Y: a1, a2
        assert_eq!(results[1].get("cnt").expect("cnt"), "1"); // X-Z: b1
    }

    #[test]
    fn test_e2e_count_empty_group_returns_zero_rows() {
        let store = crate::store::OxiRSStore::new(); // empty store
        let sparql = "SELECT (COUNT(*) AS ?cnt) WHERE { ?s ?p ?o }";
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        // With no input rows, the aggregate produces one group with zero rows → count=0
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("cnt").expect("cnt"), "0");
    }

    #[test]
    fn test_e2e_sum_no_group_by_single_value() {
        let mut store = crate::store::OxiRSStore::new();
        store.insert("http://x", "http://val", "\"42\"");
        let sparql = "SELECT (SUM(?v) AS ?total) WHERE { ?s <http://val> ?v }";
        let results = crate::query::execute_select(sparql, &store).expect("execute");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("total").expect("total"), "42");
    }
}
