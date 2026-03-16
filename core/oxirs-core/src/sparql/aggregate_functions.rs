//! SPARQL 1.1 built-in aggregate functions.
//!
//! Provides COUNT, SUM, AVG, MIN, MAX, SAMPLE, GROUP_CONCAT with full
//! group-by support for SPARQL 1.1 aggregate queries.

use std::collections::HashMap;

/// SPARQL 1.1 aggregate operation.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateOp {
    /// COUNT(?var) or COUNT(*)
    Count,
    /// SUM(?var)
    Sum,
    /// AVG(?var)
    Avg,
    /// MIN(?var)
    Min,
    /// MAX(?var)
    Max,
    /// SAMPLE(?var) — returns an arbitrary value from the group
    Sample,
    /// GROUP_CONCAT(?var ; separator="...")
    GroupConcat {
        /// Separator string; default is a single space.
        separator: String,
    },
}

/// An RDF term as used within aggregate computation.
#[derive(Debug, Clone, PartialEq)]
pub enum RdfTerm {
    /// An IRI.
    Iri(String),
    /// A typed or plain literal.
    Literal {
        /// Lexical value.
        value: String,
        /// Datatype IRI (e.g. `xsd:integer`).
        datatype: String,
    },
    /// A blank node.
    Blank(String),
    /// SPARQL unbound / error value.
    Null,
}

impl RdfTerm {
    /// Convenience constructor for a plain-string literal.
    pub fn string_literal(value: impl Into<String>) -> Self {
        RdfTerm::Literal {
            value: value.into(),
            datatype: "http://www.w3.org/2001/XMLSchema#string".to_string(),
        }
    }

    /// Convenience constructor for an integer literal.
    pub fn integer_literal(n: i64) -> Self {
        RdfTerm::Literal {
            value: n.to_string(),
            datatype: "http://www.w3.org/2001/XMLSchema#integer".to_string(),
        }
    }

    /// Convenience constructor for a double literal.
    pub fn double_literal(n: f64) -> Self {
        RdfTerm::Literal {
            value: n.to_string(),
            datatype: "http://www.w3.org/2001/XMLSchema#double".to_string(),
        }
    }

    /// Return the lexical string representation.
    pub fn lexical(&self) -> &str {
        match self {
            RdfTerm::Iri(s) | RdfTerm::Blank(s) => s.as_str(),
            RdfTerm::Literal { value, .. } => value.as_str(),
            RdfTerm::Null => "",
        }
    }
}

impl std::fmt::Display for RdfTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdfTerm::Iri(s) => write!(f, "<{s}>"),
            RdfTerm::Literal { value, datatype } => {
                write!(f, "\"{value}\"^^<{datatype}>")
            }
            RdfTerm::Blank(s) => write!(f, "_:{s}"),
            RdfTerm::Null => write!(f, "NULL"),
        }
    }
}

/// The result of one aggregate computation for a single group.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateValue {
    /// The key values identifying this group (one per GROUP BY variable).
    pub group_key: Vec<String>,
    /// The aggregate result term.
    pub result: RdfTerm,
}

/// Parse a numeric value out of an RDF term, if possible.
fn numeric_value(term: &RdfTerm) -> Option<f64> {
    match term {
        RdfTerm::Literal { value, datatype } => {
            // Accept xsd:integer, xsd:decimal, xsd:float, xsd:double, and bare digits
            let is_numeric = datatype.contains("integer")
                || datatype.contains("decimal")
                || datatype.contains("float")
                || datatype.contains("double")
                || datatype.contains("nonNegativeInteger")
                || datatype.contains("long")
                || datatype.contains("int")
                || datatype.contains("short")
                || datatype.contains("byte");
            if is_numeric {
                value.trim().parse::<f64>().ok()
            } else {
                // Try to parse anyway for bare numeric strings
                value.trim().parse::<f64>().ok()
            }
        }
        _ => None,
    }
}

/// Stateful accumulator for one group's aggregate computation.
#[derive(Debug)]
pub struct AggregateAccumulator {
    op: AggregateOp,
    values: Vec<f64>,
    strings: Vec<String>,
    count: usize,
    first_term: Option<RdfTerm>,
}

impl AggregateAccumulator {
    /// Create a new accumulator for the given aggregate operation.
    pub fn new(op: AggregateOp) -> Self {
        Self {
            op,
            values: Vec::new(),
            strings: Vec::new(),
            count: 0,
            first_term: None,
        }
    }

    /// Add one RDF term to the accumulator.
    pub fn push(&mut self, val: &RdfTerm) {
        if matches!(val, RdfTerm::Null) {
            // SPARQL spec: NULL values are ignored in aggregates (except COUNT(*))
            return;
        }
        self.count += 1;
        if self.first_term.is_none() {
            self.first_term = Some(val.clone());
        }
        // Collect numeric value for numeric aggregates
        if let Some(n) = numeric_value(val) {
            self.values.push(n);
        }
        // Always collect string representation for MIN/MAX/GROUP_CONCAT
        self.strings.push(val.lexical().to_string());
    }

    /// Finish the accumulation and return the aggregate result.
    pub fn finish(&self) -> RdfTerm {
        match &self.op {
            AggregateOp::Count => RdfTerm::integer_literal(self.count as i64),
            AggregateOp::Sum => {
                let sum: f64 = self.values.iter().sum();
                if self.values.is_empty() {
                    RdfTerm::integer_literal(0)
                } else {
                    RdfTerm::double_literal(sum)
                }
            }
            AggregateOp::Avg => {
                if self.values.is_empty() {
                    RdfTerm::Null
                } else {
                    let avg = self.values.iter().sum::<f64>() / self.values.len() as f64;
                    RdfTerm::double_literal(avg)
                }
            }
            AggregateOp::Min => {
                if self.values.is_empty() {
                    // String MIN
                    self.strings
                        .iter()
                        .min()
                        .map(|s| RdfTerm::string_literal(s))
                        .unwrap_or(RdfTerm::Null)
                } else {
                    let min = self.values.iter().cloned().fold(f64::INFINITY, f64::min);
                    RdfTerm::double_literal(min)
                }
            }
            AggregateOp::Max => {
                if self.values.is_empty() {
                    self.strings
                        .iter()
                        .max()
                        .map(|s| RdfTerm::string_literal(s))
                        .unwrap_or(RdfTerm::Null)
                } else {
                    let max = self
                        .values
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    RdfTerm::double_literal(max)
                }
            }
            AggregateOp::Sample => {
                self.first_term.clone().unwrap_or(RdfTerm::Null)
            }
            AggregateOp::GroupConcat { separator } => {
                RdfTerm::string_literal(self.strings.join(separator.as_str()))
            }
        }
    }
}

/// Stateless helper functions for SPARQL 1.1 aggregate operations.
pub struct AggregateExecutor;

impl AggregateExecutor {
    /// COUNT — number of non-null values.
    pub fn count(values: &[RdfTerm]) -> usize {
        values.iter().filter(|v| !matches!(v, RdfTerm::Null)).count()
    }

    /// SUM — sum of numeric values; returns None if no numeric values exist.
    pub fn sum(values: &[RdfTerm]) -> Option<f64> {
        let nums: Vec<f64> = values.iter().filter_map(numeric_value).collect();
        if nums.is_empty() {
            None
        } else {
            Some(nums.iter().sum())
        }
    }

    /// AVG — arithmetic mean; returns None if no numeric values.
    pub fn avg(values: &[RdfTerm]) -> Option<f64> {
        let nums: Vec<f64> = values.iter().filter_map(numeric_value).collect();
        if nums.is_empty() {
            None
        } else {
            Some(nums.iter().sum::<f64>() / nums.len() as f64)
        }
    }

    /// MIN over numeric values.
    pub fn min_numeric(values: &[RdfTerm]) -> Option<f64> {
        values
            .iter()
            .filter_map(numeric_value)
            .reduce(f64::min)
    }

    /// MAX over numeric values.
    pub fn max_numeric(values: &[RdfTerm]) -> Option<f64> {
        values
            .iter()
            .filter_map(numeric_value)
            .reduce(f64::max)
    }

    /// MIN over string representations.
    pub fn min_string(values: &[RdfTerm]) -> Option<String> {
        values
            .iter()
            .filter(|v| !matches!(v, RdfTerm::Null))
            .map(|v| v.lexical().to_string())
            .min()
    }

    /// MAX over string representations.
    pub fn max_string(values: &[RdfTerm]) -> Option<String> {
        values
            .iter()
            .filter(|v| !matches!(v, RdfTerm::Null))
            .map(|v| v.lexical().to_string())
            .max()
    }

    /// SAMPLE — return the first non-null value (deterministic for testing).
    pub fn sample(values: &[RdfTerm]) -> Option<&RdfTerm> {
        values.iter().find(|v| !matches!(v, RdfTerm::Null))
    }

    /// GROUP_CONCAT — concatenate lexical values with the given separator.
    pub fn group_concat(values: &[RdfTerm], separator: &str) -> String {
        values
            .iter()
            .filter(|v| !matches!(v, RdfTerm::Null))
            .map(|v| v.lexical())
            .collect::<Vec<_>>()
            .join(separator)
    }

    /// GROUP BY — partition rows into groups based on the specified variable names.
    ///
    /// Each row is a `Vec<(variable_name, term)>`.  The group key is
    /// the ordered tuple of `group_vars` values for that row.
    pub fn group_by(
        rows: &[Vec<(String, RdfTerm)>],
        group_vars: &[&str],
    ) -> HashMap<Vec<String>, Vec<Vec<(String, RdfTerm)>>> {
        let mut groups: HashMap<Vec<String>, Vec<Vec<(String, RdfTerm)>>> = HashMap::new();

        for row in rows {
            let key: Vec<String> = group_vars
                .iter()
                .map(|var| {
                    row.iter()
                        .find(|(name, _)| name == var)
                        .map(|(_, term)| term.lexical().to_string())
                        .unwrap_or_default()
                })
                .collect();
            groups.entry(key).or_default().push(row.clone());
        }

        groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helper constructors ---
    fn int(n: i64) -> RdfTerm {
        RdfTerm::integer_literal(n)
    }
    fn dbl(n: f64) -> RdfTerm {
        RdfTerm::double_literal(n)
    }
    fn str_lit(s: &str) -> RdfTerm {
        RdfTerm::string_literal(s)
    }
    fn iri(s: &str) -> RdfTerm {
        RdfTerm::Iri(s.to_string())
    }

    // --- AggregateExecutor::count ---
    #[test]
    fn test_count_empty() {
        assert_eq!(AggregateExecutor::count(&[]), 0);
    }

    #[test]
    fn test_count_all_null() {
        let v = vec![RdfTerm::Null, RdfTerm::Null];
        assert_eq!(AggregateExecutor::count(&v), 0);
    }

    #[test]
    fn test_count_mixed() {
        let v = vec![int(1), RdfTerm::Null, int(3), RdfTerm::Null, int(5)];
        assert_eq!(AggregateExecutor::count(&v), 3);
    }

    #[test]
    fn test_count_iris() {
        let v = vec![iri("http://a"), iri("http://b")];
        assert_eq!(AggregateExecutor::count(&v), 2);
    }

    // --- AggregateExecutor::sum ---
    #[test]
    fn test_sum_empty() {
        assert!(AggregateExecutor::sum(&[]).is_none());
    }

    #[test]
    fn test_sum_integers() {
        let v = vec![int(1), int(2), int(3)];
        assert_eq!(AggregateExecutor::sum(&v), Some(6.0));
    }

    #[test]
    fn test_sum_doubles() {
        let v = vec![dbl(1.5), dbl(2.5)];
        assert_eq!(AggregateExecutor::sum(&v), Some(4.0));
    }

    #[test]
    fn test_sum_with_nulls() {
        let v = vec![int(10), RdfTerm::Null, int(20)];
        assert_eq!(AggregateExecutor::sum(&v), Some(30.0));
    }

    #[test]
    fn test_sum_non_numeric() {
        let v = vec![str_lit("hello"), str_lit("world")];
        assert!(AggregateExecutor::sum(&v).is_none());
    }

    // --- AggregateExecutor::avg ---
    #[test]
    fn test_avg_empty() {
        assert!(AggregateExecutor::avg(&[]).is_none());
    }

    #[test]
    fn test_avg_basic() {
        let v = vec![int(2), int(4), int(6)];
        assert_eq!(AggregateExecutor::avg(&v), Some(4.0));
    }

    #[test]
    fn test_avg_doubles() {
        let v = vec![dbl(1.0), dbl(3.0)];
        assert_eq!(AggregateExecutor::avg(&v), Some(2.0));
    }

    // --- AggregateExecutor::min / max numeric ---
    #[test]
    fn test_min_numeric_empty() {
        assert!(AggregateExecutor::min_numeric(&[]).is_none());
    }

    #[test]
    fn test_min_numeric_basic() {
        let v = vec![int(5), int(2), int(8)];
        assert_eq!(AggregateExecutor::min_numeric(&v), Some(2.0));
    }

    #[test]
    fn test_max_numeric_basic() {
        let v = vec![int(5), int(2), int(8)];
        assert_eq!(AggregateExecutor::max_numeric(&v), Some(8.0));
    }

    #[test]
    fn test_min_max_numeric_single() {
        let v = vec![int(42)];
        assert_eq!(AggregateExecutor::min_numeric(&v), Some(42.0));
        assert_eq!(AggregateExecutor::max_numeric(&v), Some(42.0));
    }

    // --- AggregateExecutor::min_string / max_string ---
    #[test]
    fn test_min_string_empty() {
        assert!(AggregateExecutor::min_string(&[]).is_none());
    }

    #[test]
    fn test_min_string_basic() {
        let v = vec![str_lit("banana"), str_lit("apple"), str_lit("cherry")];
        assert_eq!(AggregateExecutor::min_string(&v), Some("apple".to_string()));
    }

    #[test]
    fn test_max_string_basic() {
        let v = vec![str_lit("banana"), str_lit("apple"), str_lit("cherry")];
        assert_eq!(AggregateExecutor::max_string(&v), Some("cherry".to_string()));
    }

    #[test]
    fn test_min_string_with_null() {
        let v = vec![str_lit("b"), RdfTerm::Null, str_lit("a")];
        assert_eq!(AggregateExecutor::min_string(&v), Some("a".to_string()));
    }

    // --- AggregateExecutor::sample ---
    #[test]
    fn test_sample_empty() {
        assert!(AggregateExecutor::sample(&[]).is_none());
    }

    #[test]
    fn test_sample_all_null() {
        let v = vec![RdfTerm::Null, RdfTerm::Null];
        assert!(AggregateExecutor::sample(&v).is_none());
    }

    #[test]
    fn test_sample_returns_first_non_null() {
        let v = vec![RdfTerm::Null, int(7), int(8)];
        assert_eq!(AggregateExecutor::sample(&v), Some(&int(7)));
    }

    // --- AggregateExecutor::group_concat ---
    #[test]
    fn test_group_concat_empty() {
        assert_eq!(AggregateExecutor::group_concat(&[], ","), "");
    }

    #[test]
    fn test_group_concat_basic() {
        let v = vec![str_lit("a"), str_lit("b"), str_lit("c")];
        assert_eq!(AggregateExecutor::group_concat(&v, ","), "a,b,c");
    }

    #[test]
    fn test_group_concat_space_separator() {
        let v = vec![str_lit("hello"), str_lit("world")];
        assert_eq!(AggregateExecutor::group_concat(&v, " "), "hello world");
    }

    #[test]
    fn test_group_concat_skips_null() {
        let v = vec![str_lit("a"), RdfTerm::Null, str_lit("b")];
        assert_eq!(AggregateExecutor::group_concat(&v, "-"), "a-b");
    }

    // --- AggregateExecutor::group_by ---
    fn make_row(pairs: &[(&str, RdfTerm)]) -> Vec<(String, RdfTerm)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn test_group_by_empty() {
        let groups = AggregateExecutor::group_by(&[], &["x"]);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_group_by_single_var() {
        let rows = vec![
            make_row(&[("type", str_lit("A")), ("val", int(1))]),
            make_row(&[("type", str_lit("B")), ("val", int(2))]),
            make_row(&[("type", str_lit("A")), ("val", int(3))]),
        ];
        let groups = AggregateExecutor::group_by(&rows, &["type"]);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[&vec!["A".to_string()]].len(), 2);
        assert_eq!(groups[&vec!["B".to_string()]].len(), 1);
    }

    #[test]
    fn test_group_by_multi_var() {
        let rows = vec![
            make_row(&[("a", str_lit("x")), ("b", str_lit("1")), ("v", int(10))]),
            make_row(&[("a", str_lit("x")), ("b", str_lit("2")), ("v", int(20))]),
            make_row(&[("a", str_lit("y")), ("b", str_lit("1")), ("v", int(30))]),
            make_row(&[("a", str_lit("x")), ("b", str_lit("1")), ("v", int(40))]),
        ];
        let groups = AggregateExecutor::group_by(&rows, &["a", "b"]);
        assert_eq!(groups.len(), 3);
        let key = vec!["x".to_string(), "1".to_string()];
        assert_eq!(groups[&key].len(), 2);
    }

    // --- AggregateAccumulator ---
    #[test]
    fn test_accumulator_count() {
        let mut acc = AggregateAccumulator::new(AggregateOp::Count);
        acc.push(&int(1));
        acc.push(&RdfTerm::Null);
        acc.push(&int(3));
        let result = acc.finish();
        assert_eq!(result, int(2));
    }

    #[test]
    fn test_accumulator_sum_empty() {
        let acc = AggregateAccumulator::new(AggregateOp::Sum);
        assert_eq!(acc.finish(), int(0));
    }

    #[test]
    fn test_accumulator_sum_basic() {
        let mut acc = AggregateAccumulator::new(AggregateOp::Sum);
        acc.push(&int(10));
        acc.push(&int(20));
        let result = acc.finish();
        if let RdfTerm::Literal { value, .. } = result {
            let v: f64 = value.parse().expect("numeric value");
            assert!((v - 30.0).abs() < 1e-9);
        } else {
            panic!("Expected a literal");
        }
    }

    #[test]
    fn test_accumulator_avg_basic() {
        let mut acc = AggregateAccumulator::new(AggregateOp::Avg);
        acc.push(&dbl(4.0));
        acc.push(&dbl(8.0));
        let result = acc.finish();
        if let RdfTerm::Literal { value, .. } = result {
            let v: f64 = value.parse().expect("numeric value");
            assert!((v - 6.0).abs() < 1e-9);
        } else {
            panic!("Expected a literal");
        }
    }

    #[test]
    fn test_accumulator_avg_empty() {
        let acc = AggregateAccumulator::new(AggregateOp::Avg);
        assert_eq!(acc.finish(), RdfTerm::Null);
    }

    #[test]
    fn test_accumulator_min_numeric() {
        let mut acc = AggregateAccumulator::new(AggregateOp::Min);
        acc.push(&int(5));
        acc.push(&int(2));
        acc.push(&int(9));
        let result = acc.finish();
        if let RdfTerm::Literal { value, .. } = result {
            let v: f64 = value.parse().expect("numeric value");
            assert!((v - 2.0).abs() < 1e-9);
        } else {
            panic!("Expected a literal");
        }
    }

    #[test]
    fn test_accumulator_max_numeric() {
        let mut acc = AggregateAccumulator::new(AggregateOp::Max);
        acc.push(&int(5));
        acc.push(&int(2));
        acc.push(&int(9));
        let result = acc.finish();
        if let RdfTerm::Literal { value, .. } = result {
            let v: f64 = value.parse().expect("numeric value");
            assert!((v - 9.0).abs() < 1e-9);
        } else {
            panic!("Expected a literal");
        }
    }

    #[test]
    fn test_accumulator_min_string() {
        let mut acc = AggregateAccumulator::new(AggregateOp::Min);
        acc.push(&str_lit("b"));
        acc.push(&str_lit("a"));
        acc.push(&str_lit("c"));
        let result = acc.finish();
        assert_eq!(result, str_lit("a"));
    }

    #[test]
    fn test_accumulator_max_string() {
        let mut acc = AggregateAccumulator::new(AggregateOp::Max);
        acc.push(&str_lit("b"));
        acc.push(&str_lit("a"));
        acc.push(&str_lit("c"));
        let result = acc.finish();
        assert_eq!(result, str_lit("c"));
    }

    #[test]
    fn test_accumulator_sample() {
        let mut acc = AggregateAccumulator::new(AggregateOp::Sample);
        acc.push(&str_lit("first"));
        acc.push(&str_lit("second"));
        assert_eq!(acc.finish(), str_lit("first"));
    }

    #[test]
    fn test_accumulator_sample_empty() {
        let acc = AggregateAccumulator::new(AggregateOp::Sample);
        assert_eq!(acc.finish(), RdfTerm::Null);
    }

    #[test]
    fn test_accumulator_group_concat_default_separator() {
        let mut acc =
            AggregateAccumulator::new(AggregateOp::GroupConcat { separator: " ".to_string() });
        acc.push(&str_lit("hello"));
        acc.push(&str_lit("world"));
        assert_eq!(acc.finish(), str_lit("hello world"));
    }

    #[test]
    fn test_accumulator_group_concat_custom_separator() {
        let mut acc =
            AggregateAccumulator::new(AggregateOp::GroupConcat { separator: "|".to_string() });
        acc.push(&str_lit("a"));
        acc.push(&str_lit("b"));
        acc.push(&str_lit("c"));
        assert_eq!(acc.finish(), str_lit("a|b|c"));
    }

    #[test]
    fn test_accumulator_group_concat_empty() {
        let acc =
            AggregateAccumulator::new(AggregateOp::GroupConcat { separator: ",".to_string() });
        assert_eq!(acc.finish(), str_lit(""));
    }

    // --- RdfTerm display ---
    #[test]
    fn test_rdf_term_display_iri() {
        let t = iri("http://example.org/foo");
        assert_eq!(t.to_string(), "<http://example.org/foo>");
    }

    #[test]
    fn test_rdf_term_display_literal() {
        let t = str_lit("hello");
        assert!(t.to_string().contains("hello"));
    }

    #[test]
    fn test_rdf_term_display_blank() {
        let t = RdfTerm::Blank("b1".to_string());
        assert_eq!(t.to_string(), "_:b1");
    }

    #[test]
    fn test_rdf_term_display_null() {
        assert_eq!(RdfTerm::Null.to_string(), "NULL");
    }

    // --- numeric_value ---
    #[test]
    fn test_numeric_value_integer() {
        assert_eq!(numeric_value(&int(42)), Some(42.0));
    }

    #[test]
    fn test_numeric_value_double() {
        assert!((numeric_value(&dbl(2.71)).expect("numeric value should be extractable") - 2.71).abs() < 1e-9);
    }

    #[test]
    fn test_numeric_value_non_numeric() {
        assert!(numeric_value(&str_lit("hello")).is_none());
    }

    #[test]
    fn test_numeric_value_null() {
        assert!(numeric_value(&RdfTerm::Null).is_none());
    }

    #[test]
    fn test_numeric_value_iri() {
        assert!(numeric_value(&iri("http://example.org")).is_none());
    }

    // --- Integration: GROUP BY + aggregate pipeline ---
    #[test]
    fn test_group_by_then_sum() {
        let rows = vec![
            make_row(&[("category", str_lit("fruit")), ("price", int(2))]),
            make_row(&[("category", str_lit("veg")), ("price", int(1))]),
            make_row(&[("category", str_lit("fruit")), ("price", int(3))]),
        ];
        let groups = AggregateExecutor::group_by(&rows, &["category"]);

        let fruit_rows = &groups[&vec!["fruit".to_string()]];
        let prices: Vec<RdfTerm> = fruit_rows
            .iter()
            .flat_map(|row| row.iter().filter(|(k, _)| k == "price").map(|(_, v)| v.clone()))
            .collect();
        let total = AggregateExecutor::sum(&prices);
        assert_eq!(total, Some(5.0));
    }

    #[test]
    fn test_group_by_then_count() {
        let rows = vec![
            make_row(&[("g", str_lit("A")), ("v", int(1))]),
            make_row(&[("g", str_lit("A")), ("v", int(2))]),
            make_row(&[("g", str_lit("B")), ("v", int(3))]),
        ];
        let groups = AggregateExecutor::group_by(&rows, &["g"]);
        let a_rows = &groups[&vec!["A".to_string()]];
        assert_eq!(AggregateExecutor::count(&[int(1), int(2)]), 2);
        assert_eq!(a_rows.len(), 2);
    }

    #[test]
    fn test_group_by_empty_groups() {
        let groups = AggregateExecutor::group_by(&[], &["g"]);
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn test_avg_single_value() {
        let v = vec![int(10)];
        assert_eq!(AggregateExecutor::avg(&v), Some(10.0));
    }

    #[test]
    fn test_sum_negative_values() {
        let v = vec![int(-5), int(3), int(-2)];
        assert_eq!(AggregateExecutor::sum(&v), Some(-4.0));
    }

    #[test]
    fn test_min_max_symmetric() {
        let v: Vec<RdfTerm> = (1..=10).map(int).collect();
        assert_eq!(AggregateExecutor::min_numeric(&v), Some(1.0));
        assert_eq!(AggregateExecutor::max_numeric(&v), Some(10.0));
    }
}
