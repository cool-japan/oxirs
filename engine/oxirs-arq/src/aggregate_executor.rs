//! SPARQL 1.1 Aggregate Function Executor
//!
//! Implements GROUP BY, HAVING, and aggregate functions (COUNT, SUM, AVG, MIN, MAX,
//! SAMPLE, GROUP_CONCAT) as specified in the SPARQL 1.1 Query Language specification.

use std::collections::HashMap;
use std::fmt;

/// SPARQL 1.1 aggregate functions.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateFunc {
    /// COUNT(?var) or COUNT(DISTINCT ?var)
    Count { distinct: bool },
    /// SUM(?var)
    Sum,
    /// AVG(?var)
    Avg,
    /// MIN(?var)
    Min,
    /// MAX(?var)
    Max,
    /// SAMPLE(?var) — returns an arbitrary non-null value
    Sample,
    /// GROUP_CONCAT(?var; separator="...")
    GroupConcat { separator: String },
    /// COUNT(*) — counts all rows including those where ?var is unbound
    CountAll,
}

/// A value produced by an aggregate expression.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateValue {
    /// Integer aggregate result (e.g., COUNT)
    Integer(i64),
    /// Floating-point aggregate result (SUM/AVG of decimals)
    Float(f64),
    /// String aggregate result (SAMPLE, GROUP_CONCAT)
    Text(String),
    /// No value — group was empty or variable was unbound throughout
    Null,
}

impl fmt::Display for AggregateValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregateValue::Integer(n) => write!(f, "{n}"),
            AggregateValue::Float(v) => write!(f, "{v}"),
            AggregateValue::Text(s) => write!(f, "{s}"),
            AggregateValue::Null => write!(f, "NULL"),
        }
    }
}

impl PartialOrd for AggregateValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        match (self, other) {
            (AggregateValue::Null, AggregateValue::Null) => Some(Ordering::Equal),
            (AggregateValue::Null, _) => Some(Ordering::Less),
            (_, AggregateValue::Null) => Some(Ordering::Greater),
            (AggregateValue::Integer(a), AggregateValue::Integer(b)) => a.partial_cmp(b),
            (AggregateValue::Integer(a), AggregateValue::Float(b)) => (*a as f64).partial_cmp(b),
            (AggregateValue::Float(a), AggregateValue::Integer(b)) => a.partial_cmp(&(*b as f64)),
            (AggregateValue::Float(a), AggregateValue::Float(b)) => a.partial_cmp(b),
            (AggregateValue::Text(a), AggregateValue::Text(b)) => a.partial_cmp(b),
            // Integer / Float < Text
            (AggregateValue::Integer(_), AggregateValue::Text(_)) => Some(Ordering::Less),
            (AggregateValue::Float(_), AggregateValue::Text(_)) => Some(Ordering::Less),
            (AggregateValue::Text(_), AggregateValue::Integer(_)) => Some(Ordering::Greater),
            (AggregateValue::Text(_), AggregateValue::Float(_)) => Some(Ordering::Greater),
        }
    }
}

impl std::ops::Add for AggregateValue {
    type Output = AggregateValue;
    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AggregateValue::Null, _) | (_, AggregateValue::Null) => AggregateValue::Null,
            (AggregateValue::Integer(a), AggregateValue::Integer(b)) => {
                AggregateValue::Integer(a.saturating_add(b))
            }
            (AggregateValue::Integer(a), AggregateValue::Float(b)) => {
                AggregateValue::Float(a as f64 + b)
            }
            (AggregateValue::Float(a), AggregateValue::Integer(b)) => {
                AggregateValue::Float(a + b as f64)
            }
            (AggregateValue::Float(a), AggregateValue::Float(b)) => AggregateValue::Float(a + b),
            _ => AggregateValue::Null,
        }
    }
}

impl std::ops::Sub for AggregateValue {
    type Output = AggregateValue;
    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AggregateValue::Null, _) | (_, AggregateValue::Null) => AggregateValue::Null,
            (AggregateValue::Integer(a), AggregateValue::Integer(b)) => {
                AggregateValue::Integer(a.saturating_sub(b))
            }
            (AggregateValue::Integer(a), AggregateValue::Float(b)) => {
                AggregateValue::Float(a as f64 - b)
            }
            (AggregateValue::Float(a), AggregateValue::Integer(b)) => {
                AggregateValue::Float(a - b as f64)
            }
            (AggregateValue::Float(a), AggregateValue::Float(b)) => AggregateValue::Float(a - b),
            _ => AggregateValue::Null,
        }
    }
}

impl std::ops::Mul for AggregateValue {
    type Output = AggregateValue;
    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AggregateValue::Null, _) | (_, AggregateValue::Null) => AggregateValue::Null,
            (AggregateValue::Integer(a), AggregateValue::Integer(b)) => {
                AggregateValue::Integer(a.saturating_mul(b))
            }
            (AggregateValue::Integer(a), AggregateValue::Float(b)) => {
                AggregateValue::Float(a as f64 * b)
            }
            (AggregateValue::Float(a), AggregateValue::Integer(b)) => {
                AggregateValue::Float(a * b as f64)
            }
            (AggregateValue::Float(a), AggregateValue::Float(b)) => AggregateValue::Float(a * b),
            _ => AggregateValue::Null,
        }
    }
}

/// GROUP BY key: ordered list of (variable_name, value) pairs.
pub type GroupKey = Vec<(String, String)>;

/// The result of an aggregate over one group.
#[derive(Debug, Clone)]
pub struct AggregateResult {
    /// The values of the GROUP BY variables for this group.
    pub group_key: GroupKey,
    /// Map from output variable name to its aggregate value.
    pub bindings: HashMap<String, AggregateValue>,
}

/// Executes SPARQL 1.1 aggregate operations.
pub struct AggregateExecutor;

impl AggregateExecutor {
    /// Partition `rows` into groups based on the values of `group_vars`.
    ///
    /// Rows that do not bind a group variable get an empty string for that variable.
    /// If `group_vars` is empty, all rows form a single group with an empty key.
    pub fn group_by(
        rows: &[HashMap<String, String>],
        group_vars: &[String],
    ) -> HashMap<GroupKey, Vec<HashMap<String, String>>> {
        let mut groups: HashMap<GroupKey, Vec<HashMap<String, String>>> = HashMap::new();
        for row in rows {
            let key: GroupKey = group_vars
                .iter()
                .map(|v| {
                    let val = row.get(v).cloned().unwrap_or_default();
                    (v.clone(), val)
                })
                .collect();
            groups.entry(key).or_default().push(row.clone());
        }
        // Ensure at least one group exists even when rows is empty and group_vars is empty
        if rows.is_empty() && group_vars.is_empty() {
            groups.entry(vec![]).or_default();
        }
        groups
    }

    /// Apply an aggregate function over a single group.
    ///
    /// `var` is the name of the variable being aggregated.
    pub fn apply(
        func: &AggregateFunc,
        var: &str,
        group: &[HashMap<String, String>],
    ) -> AggregateValue {
        match func {
            AggregateFunc::CountAll => AggregateValue::Integer(group.len() as i64),

            AggregateFunc::Count { distinct } => {
                let values: Vec<&str> = group
                    .iter()
                    .filter_map(|row| row.get(var).map(|s| s.as_str()))
                    .collect();
                if *distinct {
                    let mut seen = std::collections::HashSet::new();
                    let count = values.into_iter().filter(|v| seen.insert(*v)).count();
                    AggregateValue::Integer(count as i64)
                } else {
                    AggregateValue::Integer(values.len() as i64)
                }
            }

            AggregateFunc::Sum => {
                let nums: Vec<f64> = group
                    .iter()
                    .filter_map(|row| row.get(var).and_then(|s| s.parse::<f64>().ok()))
                    .collect();
                if nums.is_empty() {
                    AggregateValue::Null
                } else {
                    let sum: f64 = nums.iter().sum();
                    // If all values are integers, return integer
                    if nums.iter().all(|n| n.fract() == 0.0) {
                        AggregateValue::Integer(sum as i64)
                    } else {
                        AggregateValue::Float(sum)
                    }
                }
            }

            AggregateFunc::Avg => {
                let nums: Vec<f64> = group
                    .iter()
                    .filter_map(|row| row.get(var).and_then(|s| s.parse::<f64>().ok()))
                    .collect();
                if nums.is_empty() {
                    AggregateValue::Null
                } else {
                    let avg = nums.iter().sum::<f64>() / nums.len() as f64;
                    AggregateValue::Float(avg)
                }
            }

            AggregateFunc::Min => {
                let mut min_val: Option<AggregateValue> = None;
                for row in group {
                    if let Some(s) = row.get(var) {
                        let v = if let Ok(n) = s.parse::<f64>() {
                            if n.fract() == 0.0 {
                                AggregateValue::Integer(n as i64)
                            } else {
                                AggregateValue::Float(n)
                            }
                        } else {
                            AggregateValue::Text(s.clone())
                        };
                        min_val = Some(match min_val {
                            None => v,
                            Some(cur) => {
                                if v < cur {
                                    v
                                } else {
                                    cur
                                }
                            }
                        });
                    }
                }
                min_val.unwrap_or(AggregateValue::Null)
            }

            AggregateFunc::Max => {
                let mut max_val: Option<AggregateValue> = None;
                for row in group {
                    if let Some(s) = row.get(var) {
                        let v = if let Ok(n) = s.parse::<f64>() {
                            if n.fract() == 0.0 {
                                AggregateValue::Integer(n as i64)
                            } else {
                                AggregateValue::Float(n)
                            }
                        } else {
                            AggregateValue::Text(s.clone())
                        };
                        max_val = Some(match max_val {
                            None => v,
                            Some(cur) => {
                                if v > cur {
                                    v
                                } else {
                                    cur
                                }
                            }
                        });
                    }
                }
                max_val.unwrap_or(AggregateValue::Null)
            }

            AggregateFunc::Sample => {
                for row in group {
                    if let Some(s) = row.get(var) {
                        return AggregateValue::Text(s.clone());
                    }
                }
                AggregateValue::Null
            }

            AggregateFunc::GroupConcat { separator } => {
                let parts: Vec<&str> = group
                    .iter()
                    .filter_map(|row| row.get(var).map(|s| s.as_str()))
                    .collect();
                if parts.is_empty() {
                    AggregateValue::Null
                } else {
                    AggregateValue::Text(parts.join(separator.as_str()))
                }
            }
        }
    }

    /// Execute aggregate functions over all groups, returning sorted results.
    ///
    /// `aggregates` is a list of `(input_var, func, output_var)` triples.
    pub fn execute(
        rows: &[HashMap<String, String>],
        group_vars: &[String],
        aggregates: &[(String, AggregateFunc, String)],
    ) -> Vec<AggregateResult> {
        let groups = Self::group_by(rows, group_vars);
        let mut results: Vec<AggregateResult> = groups
            .into_iter()
            .map(|(key, group_rows)| {
                let mut bindings = HashMap::new();
                for (input_var, func, output_var) in aggregates {
                    let value = Self::apply(func, input_var, &group_rows);
                    bindings.insert(output_var.clone(), value);
                }
                AggregateResult {
                    group_key: key,
                    bindings,
                }
            })
            .collect();

        // Sort by group key for determinism
        results.sort_by(|a, b| a.group_key.cmp(&b.group_key));
        results
    }

    /// Filter aggregate results using a HAVING-like condition.
    ///
    /// Compares the `AggregateValue` bound to `var` against `value` using `op`.
    /// Supported operators: `"="`, `"!="`, `"<"`, `">"`, `"<="`, `">="`.
    pub fn having_filter(
        results: &[AggregateResult],
        var: &str,
        op: &str,
        value: &str,
    ) -> Vec<AggregateResult> {
        results
            .iter()
            .filter(|r| {
                let bound = r.bindings.get(var);
                Self::compare_value(bound, op, value)
            })
            .cloned()
            .collect()
    }

    /// Compare an optional AggregateValue against a string threshold using the given operator.
    fn compare_value(bound: Option<&AggregateValue>, op: &str, threshold: &str) -> bool {
        let Some(av) = bound else {
            return false;
        };

        // Try numeric comparison first
        if let Ok(threshold_f) = threshold.parse::<f64>() {
            let av_f = match av {
                AggregateValue::Integer(n) => Some(*n as f64),
                AggregateValue::Float(f) => Some(*f),
                AggregateValue::Text(t) => t.parse::<f64>().ok(),
                AggregateValue::Null => None,
            };
            if let Some(av_f) = av_f {
                return match op {
                    "=" => (av_f - threshold_f).abs() < f64::EPSILON,
                    "!=" => (av_f - threshold_f).abs() >= f64::EPSILON,
                    "<" => av_f < threshold_f,
                    ">" => av_f > threshold_f,
                    "<=" => av_f <= threshold_f,
                    ">=" => av_f >= threshold_f,
                    _ => false,
                };
            }
        }

        // String comparison
        let av_s = match av {
            AggregateValue::Text(t) => t.as_str(),
            AggregateValue::Integer(n) => {
                // Avoid allocation — convert to string inline, only for string comparison
                return match op {
                    "=" => n.to_string() == threshold,
                    "!=" => n.to_string() != threshold,
                    _ => false,
                };
            }
            AggregateValue::Float(f) => {
                return match op {
                    "=" => f.to_string() == threshold,
                    "!=" => f.to_string() != threshold,
                    _ => false,
                };
            }
            AggregateValue::Null => return false,
        };

        match op {
            "=" => av_s == threshold,
            "!=" => av_s != threshold,
            "<" => av_s < threshold,
            ">" => av_s > threshold,
            "<=" => av_s <= threshold,
            ">=" => av_s >= threshold,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // ------ AggregateValue Display ------

    #[test]
    fn test_display_integer() {
        assert_eq!(AggregateValue::Integer(42).to_string(), "42");
    }

    #[test]
    fn test_display_float() {
        assert_eq!(AggregateValue::Float(2.71).to_string(), "2.71");
    }

    #[test]
    fn test_display_text() {
        assert_eq!(
            AggregateValue::Text("hello".to_string()).to_string(),
            "hello"
        );
    }

    #[test]
    fn test_display_null() {
        assert_eq!(AggregateValue::Null.to_string(), "NULL");
    }

    // ------ Ordering ------

    #[test]
    fn test_ordering_null_less_than_integer() {
        assert!(AggregateValue::Null < AggregateValue::Integer(0));
    }

    #[test]
    fn test_ordering_integer_less_than_float() {
        assert!(AggregateValue::Integer(1) < AggregateValue::Float(1.5));
    }

    #[test]
    fn test_ordering_float_less_than_text() {
        assert!(AggregateValue::Float(99.9) < AggregateValue::Text("a".to_string()));
    }

    #[test]
    fn test_ordering_integers() {
        assert!(AggregateValue::Integer(1) < AggregateValue::Integer(2));
        assert!(AggregateValue::Integer(2) > AggregateValue::Integer(1));
    }

    // ------ Arithmetic ------

    #[test]
    fn test_add_integers() {
        let r = AggregateValue::Integer(3) + AggregateValue::Integer(4);
        assert_eq!(r, AggregateValue::Integer(7));
    }

    #[test]
    fn test_add_float_and_integer() {
        let r = AggregateValue::Float(1.5) + AggregateValue::Integer(2);
        assert_eq!(r, AggregateValue::Float(3.5));
    }

    #[test]
    fn test_add_null_propagates() {
        let r = AggregateValue::Null + AggregateValue::Integer(5);
        assert_eq!(r, AggregateValue::Null);
    }

    #[test]
    fn test_sub_integers() {
        let r = AggregateValue::Integer(10) - AggregateValue::Integer(3);
        assert_eq!(r, AggregateValue::Integer(7));
    }

    #[test]
    fn test_mul_integers() {
        let r = AggregateValue::Integer(4) * AggregateValue::Integer(5);
        assert_eq!(r, AggregateValue::Integer(20));
    }

    // ------ group_by ------

    #[test]
    fn test_group_by_empty_rows() {
        let groups = AggregateExecutor::group_by(&[], &["x".to_string()]);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_group_by_no_group_vars_single_group() {
        let rows = vec![row(&[("x", "1")]), row(&[("x", "2")])];
        let groups = AggregateExecutor::group_by(&rows, &[]);
        assert_eq!(groups.len(), 1);
        let group = groups.get(&vec![]).expect("single empty-key group");
        assert_eq!(group.len(), 2);
    }

    #[test]
    fn test_group_by_single_var() {
        let rows = vec![
            row(&[("type", "a"), ("val", "1")]),
            row(&[("type", "b"), ("val", "2")]),
            row(&[("type", "a"), ("val", "3")]),
        ];
        let groups = AggregateExecutor::group_by(&rows, &["type".to_string()]);
        assert_eq!(groups.len(), 2);
        let a_key = vec![("type".to_string(), "a".to_string())];
        assert_eq!(groups[&a_key].len(), 2);
    }

    #[test]
    fn test_group_by_multiple_vars() {
        let rows = vec![
            row(&[("a", "1"), ("b", "x")]),
            row(&[("a", "1"), ("b", "y")]),
            row(&[("a", "2"), ("b", "x")]),
        ];
        let groups = AggregateExecutor::group_by(&rows, &["a".to_string(), "b".to_string()]);
        assert_eq!(groups.len(), 3);
    }

    // ------ apply ------

    #[test]
    fn test_count_basic() {
        let group = vec![row(&[("x", "a")]), row(&[("x", "b")]), row(&[])];
        let r = AggregateExecutor::apply(&AggregateFunc::Count { distinct: false }, "x", &group);
        assert_eq!(r, AggregateValue::Integer(2));
    }

    #[test]
    fn test_count_distinct() {
        let group = vec![row(&[("x", "a")]), row(&[("x", "a")]), row(&[("x", "b")])];
        let r = AggregateExecutor::apply(&AggregateFunc::Count { distinct: true }, "x", &group);
        assert_eq!(r, AggregateValue::Integer(2));
    }

    #[test]
    fn test_count_all() {
        let group = vec![row(&[("x", "a")]), row(&[])];
        let r = AggregateExecutor::apply(&AggregateFunc::CountAll, "x", &group);
        assert_eq!(r, AggregateValue::Integer(2));
    }

    #[test]
    fn test_sum_integers() {
        let group = vec![row(&[("n", "10")]), row(&[("n", "20")]), row(&[("n", "5")])];
        let r = AggregateExecutor::apply(&AggregateFunc::Sum, "n", &group);
        assert_eq!(r, AggregateValue::Integer(35));
    }

    #[test]
    fn test_sum_floats() {
        let group = vec![row(&[("n", "1.5")]), row(&[("n", "2.5")])];
        let r = AggregateExecutor::apply(&AggregateFunc::Sum, "n", &group);
        assert_eq!(r, AggregateValue::Float(4.0));
    }

    #[test]
    fn test_sum_empty() {
        let group: Vec<HashMap<String, String>> = vec![];
        let r = AggregateExecutor::apply(&AggregateFunc::Sum, "n", &group);
        assert_eq!(r, AggregateValue::Null);
    }

    #[test]
    fn test_avg_basic() {
        let group = vec![row(&[("n", "10")]), row(&[("n", "20")])];
        let r = AggregateExecutor::apply(&AggregateFunc::Avg, "n", &group);
        assert_eq!(r, AggregateValue::Float(15.0));
    }

    #[test]
    fn test_avg_empty() {
        let group: Vec<HashMap<String, String>> = vec![];
        let r = AggregateExecutor::apply(&AggregateFunc::Avg, "n", &group);
        assert_eq!(r, AggregateValue::Null);
    }

    #[test]
    fn test_min_numeric() {
        let group = vec![row(&[("n", "5")]), row(&[("n", "2")]), row(&[("n", "8")])];
        let r = AggregateExecutor::apply(&AggregateFunc::Min, "n", &group);
        assert_eq!(r, AggregateValue::Integer(2));
    }

    #[test]
    fn test_max_numeric() {
        let group = vec![row(&[("n", "5")]), row(&[("n", "2")]), row(&[("n", "8")])];
        let r = AggregateExecutor::apply(&AggregateFunc::Max, "n", &group);
        assert_eq!(r, AggregateValue::Integer(8));
    }

    #[test]
    fn test_min_empty() {
        let group: Vec<HashMap<String, String>> = vec![];
        let r = AggregateExecutor::apply(&AggregateFunc::Min, "n", &group);
        assert_eq!(r, AggregateValue::Null);
    }

    #[test]
    fn test_min_text() {
        let group = vec![row(&[("s", "banana")]), row(&[("s", "apple")])];
        let r = AggregateExecutor::apply(&AggregateFunc::Min, "s", &group);
        assert_eq!(r, AggregateValue::Text("apple".to_string()));
    }

    #[test]
    fn test_max_text() {
        let group = vec![row(&[("s", "banana")]), row(&[("s", "apple")])];
        let r = AggregateExecutor::apply(&AggregateFunc::Max, "s", &group);
        assert_eq!(r, AggregateValue::Text("banana".to_string()));
    }

    #[test]
    fn test_sample_returns_first_non_null() {
        let group = vec![row(&[]), row(&[("x", "second")]), row(&[("x", "third")])];
        let r = AggregateExecutor::apply(&AggregateFunc::Sample, "x", &group);
        assert_eq!(r, AggregateValue::Text("second".to_string()));
    }

    #[test]
    fn test_sample_empty() {
        let group: Vec<HashMap<String, String>> = vec![];
        let r = AggregateExecutor::apply(&AggregateFunc::Sample, "x", &group);
        assert_eq!(r, AggregateValue::Null);
    }

    #[test]
    fn test_group_concat_default_separator() {
        let group = vec![row(&[("x", "a")]), row(&[("x", "b")]), row(&[("x", "c")])];
        let r = AggregateExecutor::apply(
            &AggregateFunc::GroupConcat {
                separator: " ".to_string(),
            },
            "x",
            &group,
        );
        assert_eq!(r, AggregateValue::Text("a b c".to_string()));
    }

    #[test]
    fn test_group_concat_custom_separator() {
        let group = vec![row(&[("x", "a")]), row(&[("x", "b")])];
        let r = AggregateExecutor::apply(
            &AggregateFunc::GroupConcat {
                separator: ",".to_string(),
            },
            "x",
            &group,
        );
        assert_eq!(r, AggregateValue::Text("a,b".to_string()));
    }

    #[test]
    fn test_group_concat_empty() {
        let group: Vec<HashMap<String, String>> = vec![];
        let r = AggregateExecutor::apply(
            &AggregateFunc::GroupConcat {
                separator: ",".to_string(),
            },
            "x",
            &group,
        );
        assert_eq!(r, AggregateValue::Null);
    }

    // ------ execute ------

    #[test]
    fn test_execute_grouped_count() {
        let rows = vec![
            row(&[("type", "a"), ("val", "1")]),
            row(&[("type", "a"), ("val", "2")]),
            row(&[("type", "b"), ("val", "3")]),
        ];
        let aggs = vec![(
            "val".to_string(),
            AggregateFunc::Count { distinct: false },
            "cnt".to_string(),
        )];
        let results = AggregateExecutor::execute(&rows, &["type".to_string()], &aggs);
        assert_eq!(results.len(), 2);
        // sorted: a before b
        assert_eq!(
            results[0].bindings.get("cnt"),
            Some(&AggregateValue::Integer(2))
        );
        assert_eq!(
            results[1].bindings.get("cnt"),
            Some(&AggregateValue::Integer(1))
        );
    }

    #[test]
    fn test_execute_no_group_vars() {
        let rows = vec![row(&[("n", "10")]), row(&[("n", "20")])];
        let aggs = vec![("n".to_string(), AggregateFunc::Sum, "total".to_string())];
        let results = AggregateExecutor::execute(&rows, &[], &aggs);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].bindings.get("total"),
            Some(&AggregateValue::Integer(30))
        );
    }

    #[test]
    fn test_execute_multiple_aggregates() {
        let rows = vec![
            row(&[("g", "x"), ("n", "10")]),
            row(&[("g", "x"), ("n", "20")]),
        ];
        let aggs = vec![
            ("n".to_string(), AggregateFunc::Min, "mn".to_string()),
            ("n".to_string(), AggregateFunc::Max, "mx".to_string()),
            ("n".to_string(), AggregateFunc::Avg, "av".to_string()),
        ];
        let results = AggregateExecutor::execute(&rows, &["g".to_string()], &aggs);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].bindings.get("mn"),
            Some(&AggregateValue::Integer(10))
        );
        assert_eq!(
            results[0].bindings.get("mx"),
            Some(&AggregateValue::Integer(20))
        );
        assert_eq!(
            results[0].bindings.get("av"),
            Some(&AggregateValue::Float(15.0))
        );
    }

    #[test]
    fn test_execute_sorted_deterministic() {
        let rows = vec![
            row(&[("g", "c"), ("n", "1")]),
            row(&[("g", "a"), ("n", "2")]),
            row(&[("g", "b"), ("n", "3")]),
        ];
        let aggs = vec![("n".to_string(), AggregateFunc::Sum, "s".to_string())];
        let results = AggregateExecutor::execute(&rows, &["g".to_string()], &aggs);
        assert_eq!(results[0].group_key[0].1, "a");
        assert_eq!(results[1].group_key[0].1, "b");
        assert_eq!(results[2].group_key[0].1, "c");
    }

    // ------ having_filter ------

    #[test]
    fn test_having_eq() {
        let rows = vec![row(&[("n", "1")]), row(&[("n", "2")]), row(&[("n", "2")])];
        let aggs = vec![(
            "n".to_string(),
            AggregateFunc::Count { distinct: false },
            "cnt".to_string(),
        )];
        let results = AggregateExecutor::execute(&rows, &["n".to_string()], &aggs);
        let filtered = AggregateExecutor::having_filter(&results, "cnt", "=", "2");
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_having_neq() {
        let rows = vec![row(&[("n", "1")]), row(&[("n", "2")]), row(&[("n", "2")])];
        let aggs = vec![(
            "n".to_string(),
            AggregateFunc::Count { distinct: false },
            "cnt".to_string(),
        )];
        let results = AggregateExecutor::execute(&rows, &["n".to_string()], &aggs);
        let filtered = AggregateExecutor::having_filter(&results, "cnt", "!=", "1");
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_having_gt() {
        let rows = vec![
            row(&[("g", "a"), ("n", "10")]),
            row(&[("g", "a"), ("n", "20")]),
            row(&[("g", "b"), ("n", "5")]),
        ];
        let aggs = vec![("n".to_string(), AggregateFunc::Sum, "s".to_string())];
        let results = AggregateExecutor::execute(&rows, &["g".to_string()], &aggs);
        let filtered = AggregateExecutor::having_filter(&results, "s", ">", "10");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].group_key[0].1, "a");
    }

    #[test]
    fn test_having_lt() {
        let rows = vec![
            row(&[("g", "a"), ("n", "10")]),
            row(&[("g", "b"), ("n", "5")]),
        ];
        let aggs = vec![("n".to_string(), AggregateFunc::Sum, "s".to_string())];
        let results = AggregateExecutor::execute(&rows, &["g".to_string()], &aggs);
        let filtered = AggregateExecutor::having_filter(&results, "s", "<", "8");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].group_key[0].1, "b");
    }

    #[test]
    fn test_having_lte() {
        let rows = vec![
            row(&[("g", "a"), ("n", "10")]),
            row(&[("g", "b"), ("n", "5")]),
        ];
        let aggs = vec![("n".to_string(), AggregateFunc::Sum, "s".to_string())];
        let results = AggregateExecutor::execute(&rows, &["g".to_string()], &aggs);
        let filtered = AggregateExecutor::having_filter(&results, "s", "<=", "10");
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_having_gte() {
        let rows = vec![
            row(&[("g", "a"), ("n", "10")]),
            row(&[("g", "b"), ("n", "5")]),
        ];
        let aggs = vec![("n".to_string(), AggregateFunc::Sum, "s".to_string())];
        let results = AggregateExecutor::execute(&rows, &["g".to_string()], &aggs);
        let filtered = AggregateExecutor::having_filter(&results, "s", ">=", "10");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].group_key[0].1, "a");
    }

    #[test]
    fn test_having_no_match() {
        let rows = vec![row(&[("n", "5")])];
        let aggs = vec![("n".to_string(), AggregateFunc::Sum, "s".to_string())];
        let results = AggregateExecutor::execute(&rows, &[], &aggs);
        let filtered = AggregateExecutor::having_filter(&results, "s", ">", "100");
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_execute_empty_rows_no_group_vars() {
        let rows: Vec<HashMap<String, String>> = vec![];
        let aggs = vec![("n".to_string(), AggregateFunc::Sum, "s".to_string())];
        let results = AggregateExecutor::execute(&rows, &[], &aggs);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].bindings.get("s"), Some(&AggregateValue::Null));
    }

    #[test]
    fn test_count_distinct_all_unique() {
        let group = vec![row(&[("x", "a")]), row(&[("x", "b")]), row(&[("x", "c")])];
        let r = AggregateExecutor::apply(&AggregateFunc::Count { distinct: true }, "x", &group);
        assert_eq!(r, AggregateValue::Integer(3));
    }

    #[test]
    fn test_group_concat_skips_nulls() {
        let group = vec![row(&[("x", "a")]), row(&[]), row(&[("x", "b")])];
        let r = AggregateExecutor::apply(
            &AggregateFunc::GroupConcat {
                separator: "-".to_string(),
            },
            "x",
            &group,
        );
        assert_eq!(r, AggregateValue::Text("a-b".to_string()));
    }
}
