/// SPARQL 1.1 GROUP BY clause evaluation.
///
/// Implements hash-based grouping of solution bindings by one or more key
/// expressions, HAVING filter application, and aggregate computation
/// (COUNT, SUM, AVG, MIN, MAX, GROUP_CONCAT, SAMPLE).
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single RDF term value that can appear in a binding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BindingValue {
    /// An IRI reference.
    Iri(String),
    /// A plain or language-tagged literal.
    Literal(String),
    /// A typed literal (value, datatype IRI).
    TypedLiteral(String, String),
    /// A blank node.
    BlankNode(String),
    /// The value is unbound (not present in the solution).
    Unbound,
}

impl BindingValue {
    /// Return the string representation of this value.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Iri(s) | Self::Literal(s) | Self::BlankNode(s) => s,
            Self::TypedLiteral(v, _) => v,
            Self::Unbound => "",
        }
    }

    /// Attempt to interpret the value as an `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Literal(s) | Self::TypedLiteral(s, _) => s.parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Return `true` when the value is `Unbound`.
    pub fn is_unbound(&self) -> bool {
        matches!(self, Self::Unbound)
    }
}

/// A solution row — mapping from variable name to binding value.
pub type SolutionRow = HashMap<String, BindingValue>;

/// An expression that can be evaluated against a solution row.
#[derive(Debug, Clone)]
pub enum GroupExpression {
    /// A bare variable reference (?x).
    Variable(String),
    /// String concatenation of sub-expressions.
    Concat(Vec<GroupExpression>),
    /// String length of a sub-expression.
    StrLen(Box<GroupExpression>),
    /// Upper-case of a sub-expression.
    UCase(Box<GroupExpression>),
    /// Lower-case of a sub-expression.
    LCase(Box<GroupExpression>),
    /// A constant literal value.
    Constant(BindingValue),
    /// Coalesce: first non-unbound sub-expression.
    Coalesce(Vec<GroupExpression>),
    /// Conditional: IF(cond, then, else).
    If {
        /// Condition (evaluated as boolean: non-empty string / non-zero number = true).
        condition: Box<GroupExpression>,
        /// Value when condition is true.
        then_expr: Box<GroupExpression>,
        /// Value when condition is false.
        else_expr: Box<GroupExpression>,
    },
}

/// Evaluate a `GroupExpression` against a single solution row.
pub fn eval_expression(expr: &GroupExpression, row: &SolutionRow) -> BindingValue {
    match expr {
        GroupExpression::Variable(name) => row.get(name).cloned().unwrap_or(BindingValue::Unbound),
        GroupExpression::Concat(parts) => {
            let mut buf = String::new();
            for p in parts {
                let v = eval_expression(p, row);
                buf.push_str(v.as_str());
            }
            BindingValue::Literal(buf)
        }
        GroupExpression::StrLen(inner) => {
            let v = eval_expression(inner, row);
            BindingValue::TypedLiteral(
                v.as_str().len().to_string(),
                "http://www.w3.org/2001/XMLSchema#integer".to_string(),
            )
        }
        GroupExpression::UCase(inner) => {
            let v = eval_expression(inner, row);
            BindingValue::Literal(v.as_str().to_uppercase())
        }
        GroupExpression::LCase(inner) => {
            let v = eval_expression(inner, row);
            BindingValue::Literal(v.as_str().to_lowercase())
        }
        GroupExpression::Constant(val) => val.clone(),
        GroupExpression::Coalesce(exprs) => {
            for e in exprs {
                let v = eval_expression(e, row);
                if !v.is_unbound() {
                    return v;
                }
            }
            BindingValue::Unbound
        }
        GroupExpression::If {
            condition,
            then_expr,
            else_expr,
        } => {
            let cond_val = eval_expression(condition, row);
            let truthy = match &cond_val {
                BindingValue::Unbound => false,
                BindingValue::Literal(s) | BindingValue::TypedLiteral(s, _) => {
                    !s.is_empty() && s != "0" && s != "false"
                }
                _ => true,
            };
            if truthy {
                eval_expression(then_expr, row)
            } else {
                eval_expression(else_expr, row)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Group key
// ---------------------------------------------------------------------------

/// A composite group key (one element per GROUP BY expression).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroupKey(pub Vec<BindingValue>);

impl GroupKey {
    /// Build a key by evaluating a list of expressions against a row.
    pub fn from_row(expressions: &[GroupExpression], row: &SolutionRow) -> Self {
        let values = expressions
            .iter()
            .map(|e| eval_expression(e, row))
            .collect();
        Self(values)
    }
}

// ---------------------------------------------------------------------------
// Aggregate functions
// ---------------------------------------------------------------------------

/// Supported aggregate operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregateOp {
    /// COUNT(*) or COUNT(?var).
    Count,
    /// SUM(?var).
    Sum,
    /// AVG(?var).
    Avg,
    /// MIN(?var).
    Min,
    /// MAX(?var).
    Max,
    /// GROUP_CONCAT(?var; separator=sep).
    GroupConcat(String),
    /// SAMPLE(?var) — pick an arbitrary value.
    Sample,
}

/// Specification of a single aggregate computation within a GROUP BY.
#[derive(Debug, Clone)]
pub struct AggregateSpec {
    /// The aggregate operation.
    pub op: AggregateOp,
    /// Variable (or expression) to aggregate over.
    pub input_expr: GroupExpression,
    /// If true, only distinct input values are considered.
    pub distinct: bool,
    /// Alias variable to store the result.
    pub alias: String,
}

/// Accumulator that incrementally collects values for a single aggregate.
#[derive(Debug)]
struct Accumulator {
    op: AggregateOp,
    distinct: bool,
    values: Vec<BindingValue>,
    seen: std::collections::HashSet<BindingValue>,
}

impl Accumulator {
    fn new(op: AggregateOp, distinct: bool) -> Self {
        Self {
            op,
            distinct,
            values: Vec::new(),
            seen: std::collections::HashSet::new(),
        }
    }

    fn add(&mut self, value: BindingValue) {
        if self.distinct {
            if self.seen.contains(&value) {
                return;
            }
            self.seen.insert(value.clone());
        }
        self.values.push(value);
    }

    fn finalize(&self) -> BindingValue {
        match &self.op {
            AggregateOp::Count => {
                let cnt = self.values.iter().filter(|v| !v.is_unbound()).count();
                BindingValue::TypedLiteral(
                    cnt.to_string(),
                    "http://www.w3.org/2001/XMLSchema#integer".to_string(),
                )
            }
            AggregateOp::Sum => {
                let total: f64 = self.values.iter().filter_map(|v| v.as_f64()).sum();
                BindingValue::TypedLiteral(
                    format_f64(total),
                    "http://www.w3.org/2001/XMLSchema#decimal".to_string(),
                )
            }
            AggregateOp::Avg => {
                let nums: Vec<f64> = self.values.iter().filter_map(|v| v.as_f64()).collect();
                if nums.is_empty() {
                    return BindingValue::TypedLiteral(
                        "0".to_string(),
                        "http://www.w3.org/2001/XMLSchema#decimal".to_string(),
                    );
                }
                let avg = nums.iter().sum::<f64>() / nums.len() as f64;
                BindingValue::TypedLiteral(
                    format_f64(avg),
                    "http://www.w3.org/2001/XMLSchema#decimal".to_string(),
                )
            }
            AggregateOp::Min => {
                let min_val = self
                    .values
                    .iter()
                    .filter(|v| !v.is_unbound())
                    .min_by(|a, b| a.as_str().cmp(b.as_str()));
                min_val.cloned().unwrap_or(BindingValue::Unbound)
            }
            AggregateOp::Max => {
                let max_val = self
                    .values
                    .iter()
                    .filter(|v| !v.is_unbound())
                    .max_by(|a, b| a.as_str().cmp(b.as_str()));
                max_val.cloned().unwrap_or(BindingValue::Unbound)
            }
            AggregateOp::GroupConcat(sep) => {
                let parts: Vec<&str> = self
                    .values
                    .iter()
                    .filter(|v| !v.is_unbound())
                    .map(|v| v.as_str())
                    .collect();
                BindingValue::Literal(parts.join(sep))
            }
            AggregateOp::Sample => self
                .values
                .first()
                .cloned()
                .unwrap_or(BindingValue::Unbound),
        }
    }
}

/// Format an f64 avoiding unnecessary trailing zeros.
fn format_f64(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{:.1}", v)
    } else {
        format!("{v}")
    }
}

// ---------------------------------------------------------------------------
// HAVING filter
// ---------------------------------------------------------------------------

/// A HAVING condition applied after aggregation.
#[derive(Debug, Clone)]
pub enum HavingCondition {
    /// aggregate_alias > numeric threshold.
    GreaterThan(String, f64),
    /// aggregate_alias < numeric threshold.
    LessThan(String, f64),
    /// aggregate_alias = numeric threshold.
    Equal(String, f64),
    /// aggregate_alias != numeric threshold.
    NotEqual(String, f64),
    /// aggregate_alias >= numeric threshold.
    GreaterOrEqual(String, f64),
    /// aggregate_alias <= numeric threshold.
    LessOrEqual(String, f64),
    /// Logical AND of conditions.
    And(Vec<HavingCondition>),
    /// Logical OR of conditions.
    Or(Vec<HavingCondition>),
    /// Bound(variable) — true when the variable has a non-Unbound value.
    Bound(String),
}

/// Evaluate a `HavingCondition` against one aggregated result row.
pub fn eval_having(cond: &HavingCondition, row: &SolutionRow) -> bool {
    match cond {
        HavingCondition::GreaterThan(alias, threshold) => {
            row_f64(row, alias).is_some_and(|v| v > *threshold)
        }
        HavingCondition::LessThan(alias, threshold) => {
            row_f64(row, alias).is_some_and(|v| v < *threshold)
        }
        HavingCondition::Equal(alias, threshold) => {
            row_f64(row, alias).is_some_and(|v| (v - threshold).abs() < f64::EPSILON)
        }
        HavingCondition::NotEqual(alias, threshold) => {
            row_f64(row, alias).is_some_and(|v| (v - threshold).abs() >= f64::EPSILON)
        }
        HavingCondition::GreaterOrEqual(alias, threshold) => {
            row_f64(row, alias).is_some_and(|v| v >= *threshold - f64::EPSILON)
        }
        HavingCondition::LessOrEqual(alias, threshold) => {
            row_f64(row, alias).is_some_and(|v| v <= *threshold + f64::EPSILON)
        }
        HavingCondition::And(conds) => conds.iter().all(|c| eval_having(c, row)),
        HavingCondition::Or(conds) => conds.iter().any(|c| eval_having(c, row)),
        HavingCondition::Bound(var) => row.get(var).is_some_and(|v| !v.is_unbound()),
    }
}

fn row_f64(row: &SolutionRow, alias: &str) -> Option<f64> {
    row.get(alias).and_then(|v| v.as_f64())
}

// ---------------------------------------------------------------------------
// GroupByEvaluator
// ---------------------------------------------------------------------------

/// Error type for GROUP BY evaluation.
#[derive(Debug)]
pub enum GroupByError {
    /// An expression referenced a variable that does not appear in the solution.
    UnboundVariable(String),
    /// A numeric aggregate could not be computed (e.g. no numeric values).
    NumericError(String),
    /// Configuration error.
    Config(String),
}

impl std::fmt::Display for GroupByError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnboundVariable(v) => write!(f, "Unbound variable: {v}"),
            Self::NumericError(msg) => write!(f, "Numeric error: {msg}"),
            Self::Config(msg) => write!(f, "Configuration error: {msg}"),
        }
    }
}

impl std::error::Error for GroupByError {}

/// Result of a GROUP BY evaluation.
#[derive(Debug, Clone)]
pub struct GroupByResult {
    /// Each row is a solution containing the group keys plus aggregate values.
    pub rows: Vec<SolutionRow>,
    /// Number of groups before HAVING was applied.
    pub groups_before_having: usize,
    /// Number of groups after HAVING was applied.
    pub groups_after_having: usize,
    /// Total number of input rows processed.
    pub input_row_count: usize,
}

/// Configuration for the GROUP BY evaluator.
#[derive(Debug, Clone)]
pub struct GroupByConfig {
    /// If true, treat completely empty input as a single implicit empty group.
    pub implicit_empty_group: bool,
}

impl Default for GroupByConfig {
    fn default() -> Self {
        Self {
            implicit_empty_group: true,
        }
    }
}

/// Evaluates SPARQL GROUP BY / HAVING / aggregation.
#[derive(Debug)]
pub struct GroupByEvaluator {
    /// GROUP BY key expressions.
    group_keys: Vec<GroupExpression>,
    /// Aggregate specifications.
    aggregates: Vec<AggregateSpec>,
    /// Optional HAVING filter.
    having: Option<HavingCondition>,
    /// Configuration.
    config: GroupByConfig,
}

impl GroupByEvaluator {
    /// Create a new evaluator with the given group key expressions.
    pub fn new(group_keys: Vec<GroupExpression>) -> Self {
        Self {
            group_keys,
            aggregates: Vec::new(),
            having: None,
            config: GroupByConfig::default(),
        }
    }

    /// Create a new evaluator with configuration.
    pub fn with_config(group_keys: Vec<GroupExpression>, config: GroupByConfig) -> Self {
        Self {
            group_keys,
            aggregates: Vec::new(),
            having: None,
            config,
        }
    }

    /// Add an aggregate specification.
    pub fn add_aggregate(&mut self, spec: AggregateSpec) {
        self.aggregates.push(spec);
    }

    /// Set the HAVING filter.
    pub fn set_having(&mut self, condition: HavingCondition) {
        self.having = Some(condition);
    }

    /// Return the number of group key expressions.
    pub fn key_count(&self) -> usize {
        self.group_keys.len()
    }

    /// Return the number of aggregate specifications.
    pub fn aggregate_count(&self) -> usize {
        self.aggregates.len()
    }

    /// Evaluate GROUP BY over a set of input solution rows.
    pub fn evaluate(
        &self,
        rows: &[SolutionRow],
    ) -> std::result::Result<GroupByResult, GroupByError> {
        let input_row_count = rows.len();

        // --- 1. Hash-based grouping ---
        let mut groups: HashMap<GroupKey, Vec<&SolutionRow>> = HashMap::new();

        for row in rows {
            let key = GroupKey::from_row(&self.group_keys, row);
            groups.entry(key).or_default().push(row);
        }

        // Handle empty input: produce a single group when configured
        if groups.is_empty() && self.config.implicit_empty_group && !self.aggregates.is_empty() {
            let empty_key = GroupKey(vec![BindingValue::Unbound; self.group_keys.len()]);
            groups.insert(empty_key, Vec::new());
        }

        let groups_before_having = groups.len();

        // --- 2. Compute aggregates per group ---
        let mut result_rows: Vec<SolutionRow> = Vec::with_capacity(groups_before_having);

        for (key, members) in &groups {
            let mut result_row = SolutionRow::new();

            // Populate group key variables
            for (i, expr) in self.group_keys.iter().enumerate() {
                if let GroupExpression::Variable(var_name) = expr {
                    if let Some(val) = key.0.get(i) {
                        result_row.insert(var_name.clone(), val.clone());
                    }
                }
            }

            // Compute aggregates
            for spec in &self.aggregates {
                let mut acc = Accumulator::new(spec.op.clone(), spec.distinct);
                for member in members {
                    let val = eval_expression(&spec.input_expr, member);
                    acc.add(val);
                }
                result_row.insert(spec.alias.clone(), acc.finalize());
            }

            result_rows.push(result_row);
        }

        // --- 3. Apply HAVING ---
        if let Some(having_cond) = &self.having {
            result_rows.retain(|row| eval_having(having_cond, row));
        }

        let groups_after_having = result_rows.len();

        Ok(GroupByResult {
            rows: result_rows,
            groups_before_having,
            groups_after_having,
            input_row_count,
        })
    }

    /// Convenience: group only (no aggregates, no HAVING).
    pub fn group_only(
        &self,
        rows: &[SolutionRow],
    ) -> std::result::Result<HashMap<GroupKey, Vec<SolutionRow>>, GroupByError> {
        let mut groups: HashMap<GroupKey, Vec<SolutionRow>> = HashMap::new();
        for row in rows {
            let key = GroupKey::from_row(&self.group_keys, row);
            groups.entry(key).or_default().push(row.clone());
        }
        Ok(groups)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(s: &str) -> BindingValue {
        BindingValue::Literal(s.to_string())
    }

    fn typed(v: &str, dt: &str) -> BindingValue {
        BindingValue::TypedLiteral(v.to_string(), dt.to_string())
    }

    fn iri(s: &str) -> BindingValue {
        BindingValue::Iri(s.to_string())
    }

    fn num_lit(n: f64) -> BindingValue {
        BindingValue::TypedLiteral(
            n.to_string(),
            "http://www.w3.org/2001/XMLSchema#decimal".to_string(),
        )
    }

    fn row(pairs: &[(&str, BindingValue)]) -> SolutionRow {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    // -- BindingValue tests --

    #[test]
    fn test_binding_value_as_str() {
        assert_eq!(iri("http://ex.org/a").as_str(), "http://ex.org/a");
        assert_eq!(lit("hello").as_str(), "hello");
        assert_eq!(BindingValue::Unbound.as_str(), "");
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_binding_value_as_f64() {
        assert_eq!(lit("42").as_f64(), Some(42.0));
        assert_eq!(lit("3.14").as_f64(), Some(3.14));
        assert!(lit("abc").as_f64().is_none());
        assert!(BindingValue::Unbound.as_f64().is_none());
        assert!(iri("http://ex.org").as_f64().is_none());
    }

    #[test]
    fn test_binding_value_is_unbound() {
        assert!(BindingValue::Unbound.is_unbound());
        assert!(!lit("x").is_unbound());
    }

    #[test]
    fn test_binding_value_typed_literal_as_f64() {
        let v = typed("99.5", "http://www.w3.org/2001/XMLSchema#decimal");
        assert_eq!(v.as_f64(), Some(99.5));
    }

    #[test]
    fn test_binding_value_blank_node() {
        let bn = BindingValue::BlankNode("_:b0".to_string());
        assert_eq!(bn.as_str(), "_:b0");
        assert!(!bn.is_unbound());
    }

    // -- Expression evaluation tests --

    #[test]
    fn test_eval_variable() {
        let r = row(&[("x", lit("hello"))]);
        let val = eval_expression(&GroupExpression::Variable("x".to_string()), &r);
        assert_eq!(val, lit("hello"));
    }

    #[test]
    fn test_eval_variable_unbound() {
        let r = SolutionRow::new();
        let val = eval_expression(&GroupExpression::Variable("x".to_string()), &r);
        assert_eq!(val, BindingValue::Unbound);
    }

    #[test]
    fn test_eval_concat() {
        let r = row(&[("a", lit("foo")), ("b", lit("bar"))]);
        let expr = GroupExpression::Concat(vec![
            GroupExpression::Variable("a".to_string()),
            GroupExpression::Variable("b".to_string()),
        ]);
        let val = eval_expression(&expr, &r);
        assert_eq!(val, lit("foobar"));
    }

    #[test]
    fn test_eval_strlen() {
        let r = row(&[("s", lit("hello"))]);
        let expr = GroupExpression::StrLen(Box::new(GroupExpression::Variable("s".to_string())));
        let val = eval_expression(&expr, &r);
        assert_eq!(val, typed("5", "http://www.w3.org/2001/XMLSchema#integer"));
    }

    #[test]
    fn test_eval_ucase() {
        let r = row(&[("s", lit("hello"))]);
        let expr = GroupExpression::UCase(Box::new(GroupExpression::Variable("s".to_string())));
        assert_eq!(eval_expression(&expr, &r), lit("HELLO"));
    }

    #[test]
    fn test_eval_lcase() {
        let r = row(&[("s", lit("HELLO"))]);
        let expr = GroupExpression::LCase(Box::new(GroupExpression::Variable("s".to_string())));
        assert_eq!(eval_expression(&expr, &r), lit("hello"));
    }

    #[test]
    fn test_eval_constant() {
        let r = SolutionRow::new();
        let expr = GroupExpression::Constant(lit("fixed"));
        assert_eq!(eval_expression(&expr, &r), lit("fixed"));
    }

    #[test]
    fn test_eval_coalesce() {
        let r = row(&[("b", lit("fallback"))]);
        let expr = GroupExpression::Coalesce(vec![
            GroupExpression::Variable("a".to_string()),
            GroupExpression::Variable("b".to_string()),
        ]);
        assert_eq!(eval_expression(&expr, &r), lit("fallback"));
    }

    #[test]
    fn test_eval_coalesce_all_unbound() {
        let r = SolutionRow::new();
        let expr = GroupExpression::Coalesce(vec![
            GroupExpression::Variable("a".to_string()),
            GroupExpression::Variable("b".to_string()),
        ]);
        assert_eq!(eval_expression(&expr, &r), BindingValue::Unbound);
    }

    #[test]
    fn test_eval_if_true() {
        let r = row(&[("flag", lit("1")), ("a", lit("yes")), ("b", lit("no"))]);
        let expr = GroupExpression::If {
            condition: Box::new(GroupExpression::Variable("flag".to_string())),
            then_expr: Box::new(GroupExpression::Variable("a".to_string())),
            else_expr: Box::new(GroupExpression::Variable("b".to_string())),
        };
        assert_eq!(eval_expression(&expr, &r), lit("yes"));
    }

    #[test]
    fn test_eval_if_false() {
        let r = row(&[("flag", lit("0")), ("a", lit("yes")), ("b", lit("no"))]);
        let expr = GroupExpression::If {
            condition: Box::new(GroupExpression::Variable("flag".to_string())),
            then_expr: Box::new(GroupExpression::Variable("a".to_string())),
            else_expr: Box::new(GroupExpression::Variable("b".to_string())),
        };
        assert_eq!(eval_expression(&expr, &r), lit("no"));
    }

    #[test]
    fn test_eval_if_unbound_condition() {
        let r = row(&[("a", lit("yes")), ("b", lit("no"))]);
        let expr = GroupExpression::If {
            condition: Box::new(GroupExpression::Variable("flag".to_string())),
            then_expr: Box::new(GroupExpression::Variable("a".to_string())),
            else_expr: Box::new(GroupExpression::Variable("b".to_string())),
        };
        assert_eq!(eval_expression(&expr, &r), lit("no"));
    }

    // -- GroupKey tests --

    #[test]
    fn test_group_key_single_variable() {
        let r = row(&[("city", lit("Tokyo"))]);
        let exprs = vec![GroupExpression::Variable("city".to_string())];
        let key = GroupKey::from_row(&exprs, &r);
        assert_eq!(key.0.len(), 1);
        assert_eq!(key.0[0], lit("Tokyo"));
    }

    #[test]
    fn test_group_key_multi_variable() {
        let r = row(&[("city", lit("Tokyo")), ("country", lit("JP"))]);
        let exprs = vec![
            GroupExpression::Variable("city".to_string()),
            GroupExpression::Variable("country".to_string()),
        ];
        let key = GroupKey::from_row(&exprs, &r);
        assert_eq!(key.0.len(), 2);
    }

    #[test]
    fn test_group_key_equality() {
        let r1 = row(&[("x", lit("A"))]);
        let r2 = row(&[("x", lit("A"))]);
        let exprs = vec![GroupExpression::Variable("x".to_string())];
        assert_eq!(
            GroupKey::from_row(&exprs, &r1),
            GroupKey::from_row(&exprs, &r2)
        );
    }

    #[test]
    fn test_group_key_inequality() {
        let r1 = row(&[("x", lit("A"))]);
        let r2 = row(&[("x", lit("B"))]);
        let exprs = vec![GroupExpression::Variable("x".to_string())];
        assert_ne!(
            GroupKey::from_row(&exprs, &r1),
            GroupKey::from_row(&exprs, &r2)
        );
    }

    // -- Aggregate tests --

    #[test]
    fn test_aggregate_count() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", lit("1"))]),
            row(&[("g", lit("A")), ("v", lit("2"))]),
            row(&[("g", lit("A")), ("v", lit("3"))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows.len(), 1);
        let cnt = &result.rows[0]["cnt"];
        assert_eq!(cnt.as_str(), "3");
    }

    #[test]
    fn test_aggregate_count_distinct() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", lit("x"))]),
            row(&[("g", lit("A")), ("v", lit("x"))]),
            row(&[("g", lit("A")), ("v", lit("y"))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: true,
            alias: "cnt".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows[0]["cnt"].as_str(), "2");
    }

    #[test]
    fn test_aggregate_sum() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", num_lit(10.0))]),
            row(&[("g", lit("A")), ("v", num_lit(20.0))]),
            row(&[("g", lit("A")), ("v", num_lit(30.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Sum,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "total".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        let total = result.rows[0]["total"].as_f64().expect("should be numeric");
        assert!((total - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_aggregate_avg() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", num_lit(10.0))]),
            row(&[("g", lit("A")), ("v", num_lit(20.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Avg,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "avg_v".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        let avg = result.rows[0]["avg_v"].as_f64().expect("should be numeric");
        assert!((avg - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_aggregate_min() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", lit("banana"))]),
            row(&[("g", lit("A")), ("v", lit("apple"))]),
            row(&[("g", lit("A")), ("v", lit("cherry"))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Min,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "min_v".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows[0]["min_v"], lit("apple"));
    }

    #[test]
    fn test_aggregate_max() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", lit("banana"))]),
            row(&[("g", lit("A")), ("v", lit("apple"))]),
            row(&[("g", lit("A")), ("v", lit("cherry"))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Max,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "max_v".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows[0]["max_v"], lit("cherry"));
    }

    #[test]
    fn test_aggregate_group_concat() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", lit("x"))]),
            row(&[("g", lit("A")), ("v", lit("y"))]),
            row(&[("g", lit("A")), ("v", lit("z"))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::GroupConcat(",".to_string()),
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "concat_v".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        let concat_val = result.rows[0]["concat_v"].as_str();
        // Order within group is insertion order
        assert!(concat_val.contains("x"));
        assert!(concat_val.contains("y"));
        assert!(concat_val.contains("z"));
    }

    #[test]
    fn test_aggregate_sample() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", lit("first"))]),
            row(&[("g", lit("A")), ("v", lit("second"))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Sample,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "sample_v".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        let sample_val = &result.rows[0]["sample_v"];
        assert!(!sample_val.is_unbound());
    }

    // -- Multi-group tests --

    #[test]
    fn test_multiple_groups() {
        let rows = vec![
            row(&[("dept", lit("eng")), ("salary", num_lit(100.0))]),
            row(&[("dept", lit("eng")), ("salary", num_lit(200.0))]),
            row(&[("dept", lit("sales")), ("salary", num_lit(150.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("dept".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("salary".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.groups_before_having, 2);
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_multi_key_group_by() {
        let rows = vec![
            row(&[
                ("dept", lit("eng")),
                ("loc", lit("NYC")),
                ("v", num_lit(1.0)),
            ]),
            row(&[
                ("dept", lit("eng")),
                ("loc", lit("NYC")),
                ("v", num_lit(2.0)),
            ]),
            row(&[
                ("dept", lit("eng")),
                ("loc", lit("SF")),
                ("v", num_lit(3.0)),
            ]),
        ];
        let eval = GroupByEvaluator::new(vec![
            GroupExpression::Variable("dept".to_string()),
            GroupExpression::Variable("loc".to_string()),
        ]);
        let groups = eval.group_only(&rows).expect("group_only should succeed");
        assert_eq!(groups.len(), 2);
    }

    // -- HAVING tests --

    #[test]
    fn test_having_greater_than() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", num_lit(1.0))]),
            row(&[("g", lit("A")), ("v", num_lit(2.0))]),
            row(&[("g", lit("B")), ("v", num_lit(3.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        eval.set_having(HavingCondition::GreaterThan("cnt".to_string(), 1.0));
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.groups_before_having, 2);
        assert_eq!(result.groups_after_having, 1);
        assert_eq!(result.rows[0]["g"], lit("A"));
    }

    #[test]
    fn test_having_less_than() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", num_lit(1.0))]),
            row(&[("g", lit("A")), ("v", num_lit(2.0))]),
            row(&[("g", lit("B")), ("v", num_lit(3.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        eval.set_having(HavingCondition::LessThan("cnt".to_string(), 2.0));
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.groups_after_having, 1);
        assert_eq!(result.rows[0]["g"], lit("B"));
    }

    #[test]
    fn test_having_equal() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", num_lit(1.0))]),
            row(&[("g", lit("B")), ("v", num_lit(2.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        eval.set_having(HavingCondition::Equal("cnt".to_string(), 1.0));
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_having_and() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", num_lit(10.0))]),
            row(&[("g", lit("A")), ("v", num_lit(20.0))]),
            row(&[("g", lit("B")), ("v", num_lit(5.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Sum,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "total".to_string(),
        });
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        eval.set_having(HavingCondition::And(vec![
            HavingCondition::GreaterThan("total".to_string(), 10.0),
            HavingCondition::GreaterThan("cnt".to_string(), 1.0),
        ]));
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_having_or() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", num_lit(100.0))]),
            row(&[("g", lit("B")), ("v", num_lit(1.0))]),
            row(&[("g", lit("B")), ("v", num_lit(2.0))]),
            row(&[("g", lit("C")), ("v", num_lit(5.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Sum,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "total".to_string(),
        });
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        eval.set_having(HavingCondition::Or(vec![
            HavingCondition::GreaterThan("total".to_string(), 50.0),
            HavingCondition::GreaterThan("cnt".to_string(), 1.0),
        ]));
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_having_bound() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", lit("x"))]),
            row(&[("g", lit("B"))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Sample,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "sample_v".to_string(),
        });
        eval.set_having(HavingCondition::Bound("sample_v".to_string()));
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert!(result.rows.len() <= 2);
    }

    // -- Empty input tests --

    #[test]
    fn test_empty_input_implicit_group() {
        let rows: Vec<SolutionRow> = vec![];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["cnt"].as_str(), "0");
    }

    #[test]
    fn test_empty_input_no_implicit_group() {
        let rows: Vec<SolutionRow> = vec![];
        let config = GroupByConfig {
            implicit_empty_group: false,
        };
        let mut eval =
            GroupByEvaluator::with_config(vec![GroupExpression::Variable("g".to_string())], config);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows.len(), 0);
    }

    // -- Null/unbound handling --

    #[test]
    fn test_unbound_in_group_key() {
        let rows = vec![row(&[("g", lit("A"))]), row(&[]), row(&[("g", lit("A"))])];
        let eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        let groups = eval.group_only(&rows).expect("group_only should succeed");
        assert_eq!(groups.len(), 2); // A and Unbound
    }

    #[test]
    fn test_unbound_values_in_count() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", lit("x"))]),
            row(&[("g", lit("A"))]),
            row(&[("g", lit("A")), ("v", lit("y"))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        // COUNT should exclude unbound
        assert_eq!(result.rows[0]["cnt"].as_str(), "2");
    }

    #[test]
    fn test_unbound_values_in_sum() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", num_lit(10.0))]),
            row(&[("g", lit("A"))]),
            row(&[("g", lit("A")), ("v", num_lit(20.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Sum,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "total".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        let total = result.rows[0]["total"].as_f64().expect("numeric");
        assert!((total - 30.0).abs() < 0.001);
    }

    // -- GROUP BY with expression keys --

    #[test]
    fn test_group_by_expression_strlen() {
        let rows = vec![
            row(&[("name", lit("ab"))]),
            row(&[("name", lit("cd"))]),
            row(&[("name", lit("efg"))]),
        ];
        let eval = GroupByEvaluator::new(vec![GroupExpression::StrLen(Box::new(
            GroupExpression::Variable("name".to_string()),
        ))]);
        let groups = eval.group_only(&rows).expect("group_only should succeed");
        assert_eq!(groups.len(), 2); // length 2 and length 3
    }

    #[test]
    fn test_group_by_expression_ucase() {
        let rows = vec![
            row(&[("name", lit("alice"))]),
            row(&[("name", lit("Alice"))]),
            row(&[("name", lit("bob"))]),
        ];
        let eval = GroupByEvaluator::new(vec![GroupExpression::UCase(Box::new(
            GroupExpression::Variable("name".to_string()),
        ))]);
        let groups = eval.group_only(&rows).expect("group_only should succeed");
        assert_eq!(groups.len(), 2); // ALICE and BOB
    }

    // -- Evaluator metadata --

    #[test]
    fn test_key_count() {
        let eval = GroupByEvaluator::new(vec![
            GroupExpression::Variable("a".to_string()),
            GroupExpression::Variable("b".to_string()),
        ]);
        assert_eq!(eval.key_count(), 2);
    }

    #[test]
    fn test_aggregate_count_method() {
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        assert_eq!(eval.aggregate_count(), 0);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        assert_eq!(eval.aggregate_count(), 1);
    }

    #[test]
    fn test_result_metadata() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", num_lit(1.0))]),
            row(&[("g", lit("A")), ("v", num_lit(2.0))]),
            row(&[("g", lit("B")), ("v", num_lit(3.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.input_row_count, 3);
        assert_eq!(result.groups_before_having, 2);
        assert_eq!(result.groups_after_having, 2);
    }

    // -- having_not_equal / greater_or_equal / less_or_equal --

    #[test]
    fn test_having_not_equal() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", num_lit(1.0))]),
            row(&[("g", lit("B")), ("v", num_lit(2.0))]),
            row(&[("g", lit("B")), ("v", num_lit(3.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        eval.set_having(HavingCondition::NotEqual("cnt".to_string(), 1.0));
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_having_greater_or_equal() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", num_lit(1.0))]),
            row(&[("g", lit("B")), ("v", num_lit(2.0))]),
            row(&[("g", lit("B")), ("v", num_lit(3.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        eval.set_having(HavingCondition::GreaterOrEqual("cnt".to_string(), 2.0));
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_having_less_or_equal() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", num_lit(1.0))]),
            row(&[("g", lit("B")), ("v", num_lit(2.0))]),
            row(&[("g", lit("B")), ("v", num_lit(3.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        eval.set_having(HavingCondition::LessOrEqual("cnt".to_string(), 1.0));
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows.len(), 1);
    }

    // -- Multiple aggregates at once --

    #[test]
    fn test_multiple_aggregates() {
        let rows = vec![
            row(&[("g", lit("A")), ("v", num_lit(10.0))]),
            row(&[("g", lit("A")), ("v", num_lit(20.0))]),
            row(&[("g", lit("A")), ("v", num_lit(30.0))]),
        ];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Count,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "cnt".to_string(),
        });
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Sum,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "total".to_string(),
        });
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Avg,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "avg_v".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["cnt"].as_str(), "3");
        let total = result.rows[0]["total"].as_f64().expect("numeric");
        assert!((total - 60.0).abs() < 0.001);
        let avg = result.rows[0]["avg_v"].as_f64().expect("numeric");
        assert!((avg - 20.0).abs() < 0.001);
    }

    // -- group_only --

    #[test]
    fn test_group_only() {
        let rows = vec![
            row(&[("dept", lit("A")), ("name", lit("Alice"))]),
            row(&[("dept", lit("A")), ("name", lit("Bob"))]),
            row(&[("dept", lit("B")), ("name", lit("Charlie"))]),
        ];
        let eval = GroupByEvaluator::new(vec![GroupExpression::Variable("dept".to_string())]);
        let groups = eval.group_only(&rows).expect("should work");
        assert_eq!(groups.len(), 2);
        let key_a = GroupKey(vec![lit("A")]);
        assert_eq!(groups.get(&key_a).map(|v| v.len()), Some(2));
    }

    // -- Error display --

    #[test]
    fn test_error_display() {
        let e1 = GroupByError::UnboundVariable("x".to_string());
        assert!(e1.to_string().contains("x"));
        let e2 = GroupByError::NumericError("overflow".to_string());
        assert!(e2.to_string().contains("overflow"));
        let e3 = GroupByError::Config("bad".to_string());
        assert!(e3.to_string().contains("bad"));
    }

    // -- Aggregate on empty group members --

    #[test]
    fn test_avg_empty_numeric_values() {
        let rows = vec![row(&[("g", lit("A")), ("v", lit("not_a_number"))])];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Avg,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "avg_v".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        // No numeric values => avg should be 0
        let avg = result.rows[0]["avg_v"].as_f64().expect("numeric");
        assert!((avg - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_min_max_on_unbound() {
        let rows = vec![row(&[("g", lit("A"))])];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Min,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "min_v".to_string(),
        });
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Max,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "max_v".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert!(result.rows[0]["min_v"].is_unbound());
        assert!(result.rows[0]["max_v"].is_unbound());
    }

    #[test]
    fn test_group_concat_empty() {
        let rows = vec![row(&[("g", lit("A"))])];
        let mut eval = GroupByEvaluator::new(vec![GroupExpression::Variable("g".to_string())]);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::GroupConcat(", ".to_string()),
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "gc".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        // No non-unbound values => empty string
        assert_eq!(result.rows[0]["gc"], lit(""));
    }

    #[test]
    fn test_sample_on_empty() {
        let rows: Vec<SolutionRow> = vec![];
        let config = GroupByConfig {
            implicit_empty_group: true,
        };
        let mut eval = GroupByEvaluator::with_config(vec![], config);
        eval.add_aggregate(AggregateSpec {
            op: AggregateOp::Sample,
            input_expr: GroupExpression::Variable("v".to_string()),
            distinct: false,
            alias: "s".to_string(),
        });
        let result = eval.evaluate(&rows).expect("evaluate should succeed");
        assert_eq!(result.rows.len(), 1);
        assert!(result.rows[0]["s"].is_unbound());
    }

    // -- format_f64 --

    #[test]
    fn test_format_f64_integer() {
        assert_eq!(format_f64(42.0), "42.0");
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_format_f64_decimal() {
        assert_eq!(format_f64(3.14), "3.14");
    }
}
