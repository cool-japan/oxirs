/// SPARQL 1.1 OPTIONAL clause evaluator.
///
/// Implements left outer join semantics for the OPTIONAL keyword, keeping all
/// left-side bindings and extending them with matching right-side bindings when
/// available.  Supports nested OPTIONAL, OPTIONAL with FILTER, BIND
/// expressions, and hash-based join for large result sets.
use std::collections::{HashMap, HashSet};

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur during OPTIONAL clause evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptionalError {
    /// A variable referenced in a filter or bind is invalid.
    InvalidVariable(String),
    /// The nesting depth of OPTIONAL clauses exceeds the configured limit.
    NestingDepthExceeded {
        /// The maximum allowed nesting depth.
        max_depth: usize,
    },
    /// A BIND expression references a variable that is already bound.
    BindConflict {
        /// The conflicting variable name.
        variable: String,
    },
    /// Generic evaluation failure.
    EvaluationError(String),
}

impl std::fmt::Display for OptionalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidVariable(v) => write!(f, "invalid variable: {v}"),
            Self::NestingDepthExceeded { max_depth } => {
                write!(f, "OPTIONAL nesting depth exceeded (max {max_depth})")
            }
            Self::BindConflict { variable } => {
                write!(f, "BIND conflict: variable ?{variable} already bound")
            }
            Self::EvaluationError(msg) => write!(f, "evaluation error: {msg}"),
        }
    }
}

impl std::error::Error for OptionalError {}

// ── Solution mapping ──────────────────────────────────────────────────────────

/// A single SPARQL solution: a mapping from variable names to values.
///
/// Variables are stored without the leading `?` sigil.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolutionMapping {
    bindings: HashMap<String, String>,
}

impl SolutionMapping {
    /// Create an empty solution mapping.
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Create a mapping from an iterator of `(variable, value)` pairs.
    pub fn from_pairs(iter: impl IntoIterator<Item = (String, String)>) -> Self {
        Self {
            bindings: iter.into_iter().collect(),
        }
    }

    /// Bind a variable to a value.
    pub fn bind(&mut self, var: impl Into<String>, val: impl Into<String>) {
        self.bindings.insert(var.into(), val.into());
    }

    /// Look up the value bound to a variable.
    pub fn get(&self, var: &str) -> Option<&str> {
        self.bindings.get(var).map(|s| s.as_str())
    }

    /// Returns `true` if the variable is bound.
    pub fn is_bound(&self, var: &str) -> bool {
        self.bindings.contains_key(var)
    }

    /// Returns the set of bound variable names.
    pub fn variables(&self) -> HashSet<&str> {
        self.bindings.keys().map(|k| k.as_str()).collect()
    }

    /// Returns the number of bound variables.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Returns `true` if no variables are bound.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Check whether two mappings are *compatible*: they agree on every
    /// shared variable.
    pub fn is_compatible_with(&self, other: &SolutionMapping) -> bool {
        for (k, v) in &self.bindings {
            if let Some(other_v) = other.bindings.get(k) {
                if v != other_v {
                    return false;
                }
            }
        }
        true
    }

    /// Merge two compatible mappings.  Returns `None` if they are incompatible.
    pub fn merge(&self, other: &SolutionMapping) -> Option<SolutionMapping> {
        if !self.is_compatible_with(other) {
            return None;
        }
        let mut merged = self.clone();
        for (k, v) in &other.bindings {
            merged
                .bindings
                .entry(k.clone())
                .or_insert_with(|| v.clone());
        }
        Some(merged)
    }

    /// Return an immutable reference to the inner map.
    pub fn inner(&self) -> &HashMap<String, String> {
        &self.bindings
    }
}

impl Default for SolutionMapping {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<(String, String)> for SolutionMapping {
    fn from_iter<I: IntoIterator<Item = (String, String)>>(iter: I) -> Self {
        Self {
            bindings: iter.into_iter().collect(),
        }
    }
}

// ── Filter expression ─────────────────────────────────────────────────────────

/// A simple filter expression that can be evaluated against a `SolutionMapping`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterExpr {
    /// BOUND(?var) — true when the variable is bound.
    Bound(String),
    /// !BOUND(?var) — true when the variable is *not* bound.
    NotBound(String),
    /// ?var = "value" — equality check.
    Equals { var: String, value: String },
    /// ?var != "value" — inequality check.
    NotEquals { var: String, value: String },
    /// Logical AND.
    And(Box<FilterExpr>, Box<FilterExpr>),
    /// Logical OR.
    Or(Box<FilterExpr>, Box<FilterExpr>),
    /// Logical NOT.
    Not(Box<FilterExpr>),
    /// ?var > "value" (lexicographic comparison).
    GreaterThan { var: String, value: String },
    /// ?var < "value" (lexicographic comparison).
    LessThan { var: String, value: String },
    /// Always true.
    True,
    /// Always false.
    False,
}

impl FilterExpr {
    /// Evaluate the filter against a solution mapping.
    pub fn evaluate(&self, mapping: &SolutionMapping) -> bool {
        match self {
            Self::Bound(var) => mapping.is_bound(var),
            Self::NotBound(var) => !mapping.is_bound(var),
            Self::Equals { var, value } => mapping.get(var).is_some_and(|v| v == value),
            Self::NotEquals { var, value } => mapping.get(var).map_or(true, |v| v != value),
            Self::And(a, b) => a.evaluate(mapping) && b.evaluate(mapping),
            Self::Or(a, b) => a.evaluate(mapping) || b.evaluate(mapping),
            Self::Not(inner) => !inner.evaluate(mapping),
            Self::GreaterThan { var, value } => {
                mapping.get(var).is_some_and(|v| v > value.as_str())
            }
            Self::LessThan { var, value } => mapping.get(var).is_some_and(|v| v < value.as_str()),
            Self::True => true,
            Self::False => false,
        }
    }

    /// Collect all variables referenced by this expression.
    pub fn referenced_variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut HashSet<String>) {
        match self {
            Self::Bound(v) | Self::NotBound(v) => {
                vars.insert(v.clone());
            }
            Self::Equals { var, .. }
            | Self::NotEquals { var, .. }
            | Self::GreaterThan { var, .. }
            | Self::LessThan { var, .. } => {
                vars.insert(var.clone());
            }
            Self::And(a, b) | Self::Or(a, b) => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }
            Self::Not(inner) => inner.collect_vars(vars),
            Self::True | Self::False => {}
        }
    }
}

// ── BIND expression ───────────────────────────────────────────────────────────

/// A BIND expression: `BIND(expr AS ?var)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindExpr {
    /// The target variable name (without `?`).
    pub variable: String,
    /// The expression producing the value.
    pub expression: BindValue,
}

/// The value side of a BIND expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindValue {
    /// A constant value.
    Constant(String),
    /// Copy the value from another variable.
    Variable(String),
    /// Concatenation of two expressions.
    Concat(Box<BindValue>, Box<BindValue>),
}

impl BindValue {
    /// Evaluate the bind value in the context of a solution mapping.
    pub fn evaluate(&self, mapping: &SolutionMapping) -> Option<String> {
        match self {
            Self::Constant(c) => Some(c.clone()),
            Self::Variable(var) => mapping.get(var).map(|s| s.to_string()),
            Self::Concat(a, b) => {
                let a_val = a.evaluate(mapping)?;
                let b_val = b.evaluate(mapping)?;
                Some(format!("{a_val}{b_val}"))
            }
        }
    }
}

// ── Triple pattern (right-side data) ──────────────────────────────────────────

/// A triple pattern used in the optional clause body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionalTriplePattern {
    /// Subject (variable or constant).
    pub subject: String,
    /// Predicate (variable or constant).
    pub predicate: String,
    /// Object (variable or constant).
    pub object: String,
}

impl OptionalTriplePattern {
    /// Create a new triple pattern.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    /// Returns `true` if a term is a variable (starts with `?`).
    pub fn is_variable(term: &str) -> bool {
        term.starts_with('?')
    }

    /// Collect all variables appearing in this pattern.
    pub fn variables(&self) -> Vec<&str> {
        let mut vars = Vec::new();
        if Self::is_variable(&self.subject) {
            vars.push(self.subject.as_str());
        }
        if Self::is_variable(&self.predicate) {
            vars.push(self.predicate.as_str());
        }
        if Self::is_variable(&self.object) {
            vars.push(self.object.as_str());
        }
        vars
    }

    /// Try to match this pattern against a concrete triple `(s, p, o)` given
    /// the current bindings, returning an extended mapping on success.
    pub fn match_triple(
        &self,
        s: &str,
        p: &str,
        o: &str,
        current: &SolutionMapping,
    ) -> Option<SolutionMapping> {
        let mut extended = current.clone();
        if !self.match_term(&self.subject, s, &mut extended) {
            return None;
        }
        if !self.match_term(&self.predicate, p, &mut extended) {
            return None;
        }
        if !self.match_term(&self.object, o, &mut extended) {
            return None;
        }
        Some(extended)
    }

    fn match_term(&self, pattern: &str, value: &str, mapping: &mut SolutionMapping) -> bool {
        if Self::is_variable(pattern) {
            let var_name = &pattern[1..];
            if let Some(bound) = mapping.get(var_name) {
                bound == value
            } else {
                mapping.bind(var_name.to_string(), value.to_string());
                true
            }
        } else {
            pattern == value
        }
    }
}

// ── Optional clause ───────────────────────────────────────────────────────────

/// Represents one `OPTIONAL { ... }` clause in a SPARQL query.
#[derive(Debug, Clone)]
pub struct OptionalClause {
    /// Triple patterns in the optional body.
    pub patterns: Vec<OptionalTriplePattern>,
    /// Filter expression applied within the optional scope.
    pub filter: Option<FilterExpr>,
    /// BIND expressions inside the optional scope.
    pub bind_exprs: Vec<BindExpr>,
    /// Nested OPTIONAL clauses.
    pub nested: Vec<OptionalClause>,
}

impl OptionalClause {
    /// Create an optional clause with the given patterns.
    pub fn new(patterns: Vec<OptionalTriplePattern>) -> Self {
        Self {
            patterns,
            filter: None,
            bind_exprs: Vec::new(),
            nested: Vec::new(),
        }
    }

    /// Set a filter for this optional scope.
    pub fn with_filter(mut self, filter: FilterExpr) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Add a BIND expression.
    pub fn with_bind(mut self, bind: BindExpr) -> Self {
        self.bind_exprs.push(bind);
        self
    }

    /// Add a nested OPTIONAL clause.
    pub fn with_nested(mut self, nested: OptionalClause) -> Self {
        self.nested.push(nested);
        self
    }
}

// ── Evaluator configuration ──────────────────────────────────────────────────

/// Configuration for the OPTIONAL evaluator.
#[derive(Debug, Clone)]
pub struct OptionalConfig {
    /// Maximum allowed nesting depth for OPTIONAL clauses.
    pub max_nesting_depth: usize,
    /// When `true`, use a hash-based join strategy for large result sets.
    pub use_hash_join: bool,
    /// Threshold number of right-side solutions above which hash join is used.
    pub hash_join_threshold: usize,
}

impl Default for OptionalConfig {
    fn default() -> Self {
        Self {
            max_nesting_depth: 16,
            use_hash_join: true,
            hash_join_threshold: 64,
        }
    }
}

// ── Evaluation statistics ────────────────────────────────────────────────────

/// Statistics gathered during OPTIONAL evaluation.
#[derive(Debug, Clone, Default)]
pub struct OptionalStats {
    /// Number of left-side solutions processed.
    pub left_solutions: usize,
    /// Number of right-side solutions considered.
    pub right_solutions: usize,
    /// Number of successful joins (left matched at least one right).
    pub joined_count: usize,
    /// Number of left solutions that had no right-side match (pass-through).
    pub unmatched_count: usize,
    /// Number of nested OPTIONAL evaluations performed.
    pub nested_evaluations: usize,
    /// Number of solutions eliminated by filters.
    pub filtered_count: usize,
}

// ── Evaluator ────────────────────────────────────────────────────────────────

/// SPARQL 1.1 OPTIONAL clause evaluator implementing left outer join.
pub struct OptionalEvaluator {
    config: OptionalConfig,
}

impl OptionalEvaluator {
    /// Create a new evaluator with default configuration.
    pub fn new() -> Self {
        Self {
            config: OptionalConfig::default(),
        }
    }

    /// Create a new evaluator with the given configuration.
    pub fn with_config(config: OptionalConfig) -> Self {
        Self { config }
    }

    /// Evaluate an OPTIONAL clause against left-side solutions and a set of
    /// data triples, returning the resulting solution sequence and statistics.
    ///
    /// This implements the left outer join: for each left solution, attempt to
    /// match the optional patterns against the data.  If matching succeeds,
    /// extend the solution; otherwise, keep the original left solution.
    pub fn evaluate(
        &self,
        left: &[SolutionMapping],
        clause: &OptionalClause,
        data: &[(String, String, String)],
    ) -> Result<(Vec<SolutionMapping>, OptionalStats), OptionalError> {
        self.evaluate_at_depth(left, clause, data, 0)
    }

    fn evaluate_at_depth(
        &self,
        left: &[SolutionMapping],
        clause: &OptionalClause,
        data: &[(String, String, String)],
        depth: usize,
    ) -> Result<(Vec<SolutionMapping>, OptionalStats), OptionalError> {
        if depth > self.config.max_nesting_depth {
            return Err(OptionalError::NestingDepthExceeded {
                max_depth: self.config.max_nesting_depth,
            });
        }

        // Compute right-side solutions by matching patterns against data.
        let right = self.match_patterns(&clause.patterns, data);

        let mut stats = OptionalStats {
            left_solutions: left.len(),
            right_solutions: right.len(),
            ..OptionalStats::default()
        };

        // Choose join strategy based on configuration and data size.
        let use_hash = self.config.use_hash_join && right.len() >= self.config.hash_join_threshold;

        let mut result = Vec::new();

        if use_hash {
            // Determine shared variables for hash key.
            let shared_vars = self.shared_variables(left, &right);
            let hash_index = self.build_hash_index(&right, &shared_vars);

            for left_sol in left {
                let key = self.hash_key(left_sol, &shared_vars);
                let mut matched = false;

                if let Some(candidates) = hash_index.get(&key) {
                    for right_sol in candidates {
                        if let Some(merged) = left_sol.merge(right_sol) {
                            let merged = self.apply_binds(&merged, &clause.bind_exprs)?;
                            if self.passes_filter(&merged, &clause.filter) {
                                result.push(merged);
                                matched = true;
                            } else {
                                stats.filtered_count += 1;
                            }
                        }
                    }
                }

                if !matched {
                    result.push(left_sol.clone());
                    stats.unmatched_count += 1;
                } else {
                    stats.joined_count += 1;
                }
            }
        } else {
            // Nested-loop join.
            for left_sol in left {
                let mut matched = false;

                for right_sol in &right {
                    if let Some(merged) = left_sol.merge(right_sol) {
                        let merged = self.apply_binds(&merged, &clause.bind_exprs)?;
                        if self.passes_filter(&merged, &clause.filter) {
                            result.push(merged);
                            matched = true;
                        } else {
                            stats.filtered_count += 1;
                        }
                    }
                }

                if !matched {
                    result.push(left_sol.clone());
                    stats.unmatched_count += 1;
                } else {
                    stats.joined_count += 1;
                }
            }
        }

        // Process nested OPTIONAL clauses.
        for nested in &clause.nested {
            stats.nested_evaluations += 1;
            let (nested_result, nested_stats) =
                self.evaluate_at_depth(&result, nested, data, depth + 1)?;
            result = nested_result;
            stats.joined_count += nested_stats.joined_count;
            stats.unmatched_count += nested_stats.unmatched_count;
            stats.filtered_count += nested_stats.filtered_count;
            stats.nested_evaluations += nested_stats.nested_evaluations;
        }

        Ok((result, stats))
    }

    /// Match a set of triple patterns against data, producing all matching
    /// solution mappings.
    fn match_patterns(
        &self,
        patterns: &[OptionalTriplePattern],
        data: &[(String, String, String)],
    ) -> Vec<SolutionMapping> {
        if patterns.is_empty() {
            return vec![SolutionMapping::new()];
        }

        let mut solutions = vec![SolutionMapping::new()];

        for pattern in patterns {
            let mut next_solutions = Vec::new();
            for sol in &solutions {
                for (s, p, o) in data {
                    if let Some(extended) = pattern.match_triple(s, p, o, sol) {
                        next_solutions.push(extended);
                    }
                }
            }
            solutions = next_solutions;
        }

        solutions
    }

    /// Compute the set of variable names (without `?`) that appear bound in
    /// both left and right solution sequences.
    fn shared_variables(&self, left: &[SolutionMapping], right: &[SolutionMapping]) -> Vec<String> {
        let left_vars: HashSet<String> = left
            .iter()
            .flat_map(|s| s.bindings.keys().cloned())
            .collect();
        let right_vars: HashSet<String> = right
            .iter()
            .flat_map(|s| s.bindings.keys().cloned())
            .collect();
        left_vars.intersection(&right_vars).cloned().collect()
    }

    /// Build a hash index over right-side solutions keyed by the shared
    /// variables.
    fn build_hash_index(
        &self,
        right: &[SolutionMapping],
        shared_vars: &[String],
    ) -> HashMap<Vec<Option<String>>, Vec<SolutionMapping>> {
        let mut index: HashMap<Vec<Option<String>>, Vec<SolutionMapping>> = HashMap::new();
        for sol in right {
            let key = self.hash_key(sol, shared_vars);
            index.entry(key).or_default().push(sol.clone());
        }
        index
    }

    /// Produce a key from a solution mapping over the given shared variables.
    fn hash_key(&self, sol: &SolutionMapping, shared_vars: &[String]) -> Vec<Option<String>> {
        shared_vars
            .iter()
            .map(|v| sol.get(v).map(|s| s.to_string()))
            .collect()
    }

    /// Apply BIND expressions to a solution, returning a new extended mapping.
    fn apply_binds(
        &self,
        mapping: &SolutionMapping,
        binds: &[BindExpr],
    ) -> Result<SolutionMapping, OptionalError> {
        let mut result = mapping.clone();
        for bind in binds {
            if result.is_bound(&bind.variable) {
                return Err(OptionalError::BindConflict {
                    variable: bind.variable.clone(),
                });
            }
            if let Some(val) = bind.expression.evaluate(&result) {
                result.bind(bind.variable.clone(), val);
            }
        }
        Ok(result)
    }

    /// Check whether a mapping passes the optional-scope filter.
    fn passes_filter(&self, mapping: &SolutionMapping, filter: &Option<FilterExpr>) -> bool {
        match filter {
            Some(expr) => expr.evaluate(mapping),
            None => true,
        }
    }

    /// Convenience: evaluate multiple OPTIONAL clauses in sequence (left-to-right
    /// composition).
    pub fn evaluate_sequence(
        &self,
        initial: &[SolutionMapping],
        clauses: &[OptionalClause],
        data: &[(String, String, String)],
    ) -> Result<(Vec<SolutionMapping>, Vec<OptionalStats>), OptionalError> {
        let mut current = initial.to_vec();
        let mut all_stats = Vec::new();

        for clause in clauses {
            let (next, stats) = self.evaluate(&current, clause, data)?;
            current = next;
            all_stats.push(stats);
        }

        Ok((current, all_stats))
    }
}

impl Default for OptionalEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helper ──────────────────────────────────────────────────────────────

    fn mapping(pairs: &[(&str, &str)]) -> SolutionMapping {
        SolutionMapping::from_iter(pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())))
    }

    fn data() -> Vec<(String, String, String)> {
        vec![
            ("alice".into(), "name".into(), "Alice".into()),
            ("alice".into(), "age".into(), "30".into()),
            ("alice".into(), "email".into(), "alice@example.com".into()),
            ("bob".into(), "name".into(), "Bob".into()),
            ("bob".into(), "age".into(), "25".into()),
            ("charlie".into(), "name".into(), "Charlie".into()),
        ]
    }

    // ── SolutionMapping tests ───────────────────────────────────────────────

    #[test]
    fn test_solution_mapping_new() {
        let m = SolutionMapping::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn test_solution_mapping_bind_and_get() {
        let mut m = SolutionMapping::new();
        m.bind("x", "1");
        assert_eq!(m.get("x"), Some("1"));
        assert_eq!(m.get("y"), None);
        assert!(m.is_bound("x"));
        assert!(!m.is_bound("y"));
    }

    #[test]
    fn test_solution_mapping_from_iter() {
        let m = mapping(&[("x", "1"), ("y", "2")]);
        assert_eq!(m.len(), 2);
        assert_eq!(m.get("x"), Some("1"));
        assert_eq!(m.get("y"), Some("2"));
    }

    #[test]
    fn test_solution_mapping_variables() {
        let m = mapping(&[("a", "1"), ("b", "2"), ("c", "3")]);
        let vars = m.variables();
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
        assert!(vars.contains("c"));
        assert_eq!(vars.len(), 3);
    }

    #[test]
    fn test_solution_mapping_compatible_same_values() {
        let a = mapping(&[("x", "1"), ("y", "2")]);
        let b = mapping(&[("x", "1"), ("z", "3")]);
        assert!(a.is_compatible_with(&b));
    }

    #[test]
    fn test_solution_mapping_incompatible() {
        let a = mapping(&[("x", "1")]);
        let b = mapping(&[("x", "99")]);
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn test_solution_mapping_merge_compatible() {
        let a = mapping(&[("x", "1")]);
        let b = mapping(&[("y", "2")]);
        let merged = a.merge(&b);
        assert!(merged.is_some());
        let m = merged.expect("merge should succeed");
        assert_eq!(m.get("x"), Some("1"));
        assert_eq!(m.get("y"), Some("2"));
    }

    #[test]
    fn test_solution_mapping_merge_incompatible() {
        let a = mapping(&[("x", "1")]);
        let b = mapping(&[("x", "2")]);
        assert!(a.merge(&b).is_none());
    }

    #[test]
    fn test_solution_mapping_merge_overlapping_same() {
        let a = mapping(&[("x", "1"), ("y", "2")]);
        let b = mapping(&[("x", "1"), ("z", "3")]);
        let merged = a.merge(&b);
        assert!(merged.is_some());
        let m = merged.expect("merge should succeed");
        assert_eq!(m.get("x"), Some("1"));
        assert_eq!(m.get("y"), Some("2"));
        assert_eq!(m.get("z"), Some("3"));
    }

    #[test]
    fn test_solution_mapping_default() {
        let m = SolutionMapping::default();
        assert!(m.is_empty());
    }

    // ── FilterExpr tests ────────────────────────────────────────────────────

    #[test]
    fn test_filter_bound() {
        let m = mapping(&[("x", "1")]);
        assert!(FilterExpr::Bound("x".into()).evaluate(&m));
        assert!(!FilterExpr::Bound("y".into()).evaluate(&m));
    }

    #[test]
    fn test_filter_not_bound() {
        let m = mapping(&[("x", "1")]);
        assert!(!FilterExpr::NotBound("x".into()).evaluate(&m));
        assert!(FilterExpr::NotBound("y".into()).evaluate(&m));
    }

    #[test]
    fn test_filter_equals() {
        let m = mapping(&[("x", "hello")]);
        let eq = FilterExpr::Equals {
            var: "x".into(),
            value: "hello".into(),
        };
        assert!(eq.evaluate(&m));
        let ne = FilterExpr::Equals {
            var: "x".into(),
            value: "world".into(),
        };
        assert!(!ne.evaluate(&m));
    }

    #[test]
    fn test_filter_not_equals() {
        let m = mapping(&[("x", "hello")]);
        let ne = FilterExpr::NotEquals {
            var: "x".into(),
            value: "world".into(),
        };
        assert!(ne.evaluate(&m));
    }

    #[test]
    fn test_filter_and() {
        let m = mapping(&[("x", "1"), ("y", "2")]);
        let f = FilterExpr::And(
            Box::new(FilterExpr::Bound("x".into())),
            Box::new(FilterExpr::Bound("y".into())),
        );
        assert!(f.evaluate(&m));
    }

    #[test]
    fn test_filter_or() {
        let m = mapping(&[("x", "1")]);
        let f = FilterExpr::Or(
            Box::new(FilterExpr::Bound("x".into())),
            Box::new(FilterExpr::Bound("z".into())),
        );
        assert!(f.evaluate(&m));
    }

    #[test]
    fn test_filter_not() {
        let m = mapping(&[("x", "1")]);
        let f = FilterExpr::Not(Box::new(FilterExpr::Bound("z".into())));
        assert!(f.evaluate(&m));
    }

    #[test]
    fn test_filter_greater_than() {
        let m = mapping(&[("x", "b")]);
        let f = FilterExpr::GreaterThan {
            var: "x".into(),
            value: "a".into(),
        };
        assert!(f.evaluate(&m));
    }

    #[test]
    fn test_filter_less_than() {
        let m = mapping(&[("x", "a")]);
        let f = FilterExpr::LessThan {
            var: "x".into(),
            value: "b".into(),
        };
        assert!(f.evaluate(&m));
    }

    #[test]
    fn test_filter_true_false() {
        let m = SolutionMapping::new();
        assert!(FilterExpr::True.evaluate(&m));
        assert!(!FilterExpr::False.evaluate(&m));
    }

    #[test]
    fn test_filter_referenced_variables() {
        let f = FilterExpr::And(
            Box::new(FilterExpr::Bound("x".into())),
            Box::new(FilterExpr::Equals {
                var: "y".into(),
                value: "v".into(),
            }),
        );
        let vars = f.referenced_variables();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert_eq!(vars.len(), 2);
    }

    // ── BindValue / BindExpr tests ──────────────────────────────────────────

    #[test]
    fn test_bind_value_constant() {
        let m = SolutionMapping::new();
        let bv = BindValue::Constant("hello".into());
        assert_eq!(bv.evaluate(&m), Some("hello".into()));
    }

    #[test]
    fn test_bind_value_variable() {
        let m = mapping(&[("x", "world")]);
        let bv = BindValue::Variable("x".into());
        assert_eq!(bv.evaluate(&m), Some("world".into()));
    }

    #[test]
    fn test_bind_value_variable_unbound() {
        let m = SolutionMapping::new();
        let bv = BindValue::Variable("x".into());
        assert_eq!(bv.evaluate(&m), None);
    }

    #[test]
    fn test_bind_value_concat() {
        let m = mapping(&[("first", "John"), ("last", "Doe")]);
        let bv = BindValue::Concat(
            Box::new(BindValue::Variable("first".into())),
            Box::new(BindValue::Concat(
                Box::new(BindValue::Constant(" ".into())),
                Box::new(BindValue::Variable("last".into())),
            )),
        );
        assert_eq!(bv.evaluate(&m), Some("John Doe".into()));
    }

    // ── OptionalTriplePattern tests ─────────────────────────────────────────

    #[test]
    fn test_triple_pattern_is_variable() {
        assert!(OptionalTriplePattern::is_variable("?x"));
        assert!(!OptionalTriplePattern::is_variable("alice"));
    }

    #[test]
    fn test_triple_pattern_variables() {
        let p = OptionalTriplePattern::new("?s", "?p", "?o");
        let vars = p.variables();
        assert_eq!(vars.len(), 3);
    }

    #[test]
    fn test_triple_pattern_match_all_vars() {
        let p = OptionalTriplePattern::new("?s", "?p", "?o");
        let m = SolutionMapping::new();
        let result = p.match_triple("alice", "name", "Alice", &m);
        assert!(result.is_some());
        let r = result.expect("should match");
        assert_eq!(r.get("s"), Some("alice"));
        assert_eq!(r.get("p"), Some("name"));
        assert_eq!(r.get("o"), Some("Alice"));
    }

    #[test]
    fn test_triple_pattern_match_with_constant() {
        let p = OptionalTriplePattern::new("?s", "name", "?o");
        let m = SolutionMapping::new();
        let result = p.match_triple("alice", "name", "Alice", &m);
        assert!(result.is_some());
        let fail = p.match_triple("alice", "age", "30", &m);
        assert!(fail.is_none());
    }

    #[test]
    fn test_triple_pattern_match_existing_binding() {
        let p = OptionalTriplePattern::new("?s", "name", "?o");
        let m = mapping(&[("s", "alice")]);
        let result = p.match_triple("alice", "name", "Alice", &m);
        assert!(result.is_some());
        let fail = p.match_triple("bob", "name", "Bob", &m);
        assert!(fail.is_none());
    }

    // ── OptionalClause builder tests ────────────────────────────────────────

    #[test]
    fn test_optional_clause_new() {
        let patterns = vec![OptionalTriplePattern::new("?s", "email", "?e")];
        let clause = OptionalClause::new(patterns);
        assert_eq!(clause.patterns.len(), 1);
        assert!(clause.filter.is_none());
        assert!(clause.bind_exprs.is_empty());
        assert!(clause.nested.is_empty());
    }

    #[test]
    fn test_optional_clause_with_filter() {
        let clause = OptionalClause::new(vec![]).with_filter(FilterExpr::Bound("x".into()));
        assert!(clause.filter.is_some());
    }

    #[test]
    fn test_optional_clause_with_bind() {
        let bind = BindExpr {
            variable: "full".into(),
            expression: BindValue::Constant("test".into()),
        };
        let clause = OptionalClause::new(vec![]).with_bind(bind);
        assert_eq!(clause.bind_exprs.len(), 1);
    }

    #[test]
    fn test_optional_clause_with_nested() {
        let inner = OptionalClause::new(vec![]);
        let clause = OptionalClause::new(vec![]).with_nested(inner);
        assert_eq!(clause.nested.len(), 1);
    }

    // ── OptionalEvaluator: basic left outer join ────────────────────────────

    #[test]
    fn test_basic_left_outer_join() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice")]), mapping(&[("s", "charlie")])];
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "email", "?e")]);
        let d = data();
        let (result, stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");

        // alice has email, charlie does not
        assert_eq!(result.len(), 2);
        // alice should have email binding
        let alice_sol = result.iter().find(|m| m.get("s") == Some("alice"));
        assert!(alice_sol.is_some());
        assert_eq!(
            alice_sol.expect("alice exists").get("e"),
            Some("alice@example.com")
        );
        // charlie should not have email binding
        let charlie_sol = result.iter().find(|m| m.get("s") == Some("charlie"));
        assert!(charlie_sol.is_some());
        assert!(charlie_sol.expect("charlie exists").get("e").is_none());
        assert_eq!(stats.joined_count, 1);
        assert_eq!(stats.unmatched_count, 1);
    }

    #[test]
    fn test_optional_all_match() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice")]), mapping(&[("s", "bob")])];
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "name", "?n")]);
        let d = data();
        let (result, stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        assert_eq!(result.len(), 2);
        assert_eq!(stats.joined_count, 2);
        assert_eq!(stats.unmatched_count, 0);
    }

    #[test]
    fn test_optional_none_match() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice")]), mapping(&[("s", "bob")])];
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "phone", "?p")]);
        let d = data();
        let (result, stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        // All should pass through unmatched
        assert_eq!(result.len(), 2);
        assert_eq!(stats.unmatched_count, 2);
        assert_eq!(stats.joined_count, 0);
    }

    #[test]
    fn test_optional_empty_left() {
        let eval = OptionalEvaluator::new();
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "name", "?n")]);
        let d = data();
        let (result, _stats) = eval
            .evaluate(&[], &clause, &d)
            .expect("evaluation should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_optional_empty_patterns() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("x", "1")])];
        let clause = OptionalClause::new(vec![]);
        let d = data();
        let (result, _stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        // Empty pattern matches anything; left solution is extended with empty.
        assert!(!result.is_empty());
    }

    // ── OPTIONAL with FILTER ────────────────────────────────────────────────

    #[test]
    fn test_optional_with_filter_passes() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice")])];
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "age", "?a")])
            .with_filter(FilterExpr::Equals {
                var: "a".into(),
                value: "30".into(),
            });
        let d = data();
        let (result, stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("a"), Some("30"));
        assert_eq!(stats.filtered_count, 0);
    }

    #[test]
    fn test_optional_with_filter_rejects() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice")])];
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "age", "?a")])
            .with_filter(FilterExpr::Equals {
                var: "a".into(),
                value: "99".into(), // alice's age is 30, not 99
            });
        let d = data();
        let (result, stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        // Filter rejects the match, so alice gets pass-through without age binding.
        assert_eq!(result.len(), 1);
        assert!(result[0].get("a").is_none());
        assert_eq!(stats.filtered_count, 1);
        assert_eq!(stats.unmatched_count, 1);
    }

    // ── OPTIONAL with BIND ──────────────────────────────────────────────────

    #[test]
    fn test_optional_with_bind() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice")])];
        let bind = BindExpr {
            variable: "label".into(),
            expression: BindValue::Concat(
                Box::new(BindValue::Variable("n".into())),
                Box::new(BindValue::Constant("!".into())),
            ),
        };
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "name", "?n")])
            .with_bind(bind);
        let d = data();
        let (result, _stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("label"), Some("Alice!"));
    }

    #[test]
    fn test_optional_bind_conflict() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice"), ("n", "existing")])];
        let bind = BindExpr {
            variable: "n".into(), // conflict: already bound
            expression: BindValue::Constant("other".into()),
        };
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "email", "?e")])
            .with_bind(bind);
        let d = data();
        let result = eval.evaluate(&left, &clause, &d);
        assert!(result.is_err());
        match result {
            Err(OptionalError::BindConflict { variable }) => {
                assert_eq!(variable, "n");
            }
            other => panic!("expected BindConflict, got {other:?}"),
        }
    }

    // ── Nested OPTIONAL ─────────────────────────────────────────────────────

    #[test]
    fn test_nested_optional() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice")]), mapping(&[("s", "charlie")])];
        let inner = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "email", "?e")]);
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "name", "?n")])
            .with_nested(inner);
        let d = data();
        let (result, stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        // alice: name=Alice, email=alice@example.com
        // charlie: name=Charlie, email=<unbound>
        assert_eq!(result.len(), 2);
        let alice = result
            .iter()
            .find(|m| m.get("s") == Some("alice"))
            .expect("alice exists");
        assert_eq!(alice.get("n"), Some("Alice"));
        assert_eq!(alice.get("e"), Some("alice@example.com"));
        let charlie = result
            .iter()
            .find(|m| m.get("s") == Some("charlie"))
            .expect("charlie exists");
        assert_eq!(charlie.get("n"), Some("Charlie"));
        assert!(charlie.get("e").is_none());
        assert!(stats.nested_evaluations > 0);
    }

    #[test]
    fn test_deeply_nested_optional() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice")])];
        let inner2 = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "email", "?e")]);
        let inner1 = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "age", "?a")])
            .with_nested(inner2);
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "name", "?n")])
            .with_nested(inner1);
        let d = data();
        let (result, _stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        assert_eq!(result.len(), 1);
        let sol = &result[0];
        assert_eq!(sol.get("n"), Some("Alice"));
        assert_eq!(sol.get("a"), Some("30"));
        assert_eq!(sol.get("e"), Some("alice@example.com"));
    }

    // ── Nesting depth limit ─────────────────────────────────────────────────

    #[test]
    fn test_nesting_depth_exceeded() {
        // max_nesting_depth=0 means no nested OPTIONAL is allowed.
        // The top-level evaluate starts at depth=0; the nested clause will
        // attempt depth=1 which exceeds 0.
        let config = OptionalConfig {
            max_nesting_depth: 0,
            ..OptionalConfig::default()
        };
        let eval = OptionalEvaluator::with_config(config);
        let left = vec![mapping(&[("s", "alice")])];
        let inner = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "email", "?e")]);
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "name", "?n")])
            .with_nested(inner);
        let d = data();
        let result = eval.evaluate(&left, &clause, &d);
        assert!(result.is_err());
        match result {
            Err(OptionalError::NestingDepthExceeded { max_depth }) => {
                assert_eq!(max_depth, 0);
            }
            other => panic!("expected NestingDepthExceeded, got {other:?}"),
        }
    }

    // ── Multi-variable optional binding ─────────────────────────────────────

    #[test]
    fn test_multi_variable_optional() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice")])];
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "?p", "?o")]);
        let d = data();
        let (result, _stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        // alice has 3 triples: name, age, email
        assert_eq!(result.len(), 3);
    }

    // ── Hash-based join ─────────────────────────────────────────────────────

    #[test]
    fn test_hash_join_threshold() {
        let config = OptionalConfig {
            use_hash_join: true,
            hash_join_threshold: 1, // force hash join even for small sets
            ..OptionalConfig::default()
        };
        let eval = OptionalEvaluator::with_config(config);
        let left = vec![mapping(&[("s", "alice")]), mapping(&[("s", "bob")])];
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "name", "?n")]);
        let d = data();
        let (result, _stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_hash_join_disabled() {
        let config = OptionalConfig {
            use_hash_join: false,
            ..OptionalConfig::default()
        };
        let eval = OptionalEvaluator::with_config(config);
        let left = vec![mapping(&[("s", "alice")])];
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "name", "?n")]);
        let d = data();
        let (result, _stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("n"), Some("Alice"));
    }

    // ── evaluate_sequence ──────────────────────────────────────────────────

    #[test]
    fn test_evaluate_sequence() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice")])];
        let clauses = vec![
            OptionalClause::new(vec![OptionalTriplePattern::new("?s", "name", "?n")]),
            OptionalClause::new(vec![OptionalTriplePattern::new("?s", "email", "?e")]),
        ];
        let d = data();
        let (result, stats_vec) = eval
            .evaluate_sequence(&left, &clauses, &d)
            .expect("evaluation should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("n"), Some("Alice"));
        assert_eq!(result[0].get("e"), Some("alice@example.com"));
        assert_eq!(stats_vec.len(), 2);
    }

    #[test]
    fn test_evaluate_sequence_empty_clauses() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("x", "1")])];
        let d = data();
        let (result, stats_vec) = eval
            .evaluate_sequence(&left, &[], &d)
            .expect("evaluation should succeed");
        assert_eq!(result.len(), 1);
        assert!(stats_vec.is_empty());
    }

    // ── Bound/unbound variable propagation ──────────────────────────────────

    #[test]
    fn test_bound_unbound_propagation() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice"), ("x", "extra")])];
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "name", "?n")]);
        let d = data();
        let (result, _stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        // Existing bindings should be preserved
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("x"), Some("extra"));
        assert_eq!(result[0].get("n"), Some("Alice"));
    }

    // ── Empty optional result handling ──────────────────────────────────────

    #[test]
    fn test_empty_data() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice")])];
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "name", "?n")]);
        let empty_data: Vec<(String, String, String)> = vec![];
        let (result, stats) = eval
            .evaluate(&left, &clause, &empty_data)
            .expect("evaluation should succeed");
        // No data means no match; left passes through.
        assert_eq!(result.len(), 1);
        assert!(result[0].get("n").is_none());
        assert_eq!(stats.unmatched_count, 1);
    }

    // ── Config tests ────────────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let config = OptionalConfig::default();
        assert_eq!(config.max_nesting_depth, 16);
        assert!(config.use_hash_join);
        assert_eq!(config.hash_join_threshold, 64);
    }

    // ── Error display ───────────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let err = OptionalError::InvalidVariable("foo".into());
        assert!(err.to_string().contains("foo"));

        let err2 = OptionalError::NestingDepthExceeded { max_depth: 5 };
        assert!(err2.to_string().contains("5"));

        let err3 = OptionalError::BindConflict {
            variable: "x".into(),
        };
        assert!(err3.to_string().contains("x"));

        let err4 = OptionalError::EvaluationError("oops".into());
        assert!(err4.to_string().contains("oops"));
    }

    // ── Compatible binding merge ─────────────────────────────────────────────

    #[test]
    fn test_compatible_binding_merge_join_on_shared() {
        let eval = OptionalEvaluator::new();
        // Two left solutions with different "s" values
        let left = vec![
            mapping(&[("s", "alice"), ("g", "group1")]),
            mapping(&[("s", "bob"), ("g", "group2")]),
        ];
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "age", "?a")]);
        let d = data();
        let (result, _stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        assert_eq!(result.len(), 2);
        let alice_sol = result
            .iter()
            .find(|m| m.get("s") == Some("alice"))
            .expect("alice exists");
        assert_eq!(alice_sol.get("a"), Some("30"));
        assert_eq!(alice_sol.get("g"), Some("group1"));
        let bob_sol = result
            .iter()
            .find(|m| m.get("s") == Some("bob"))
            .expect("bob exists");
        assert_eq!(bob_sol.get("a"), Some("25"));
        assert_eq!(bob_sol.get("g"), Some("group2"));
    }

    #[test]
    fn test_optional_multiple_matches_per_left() {
        let eval = OptionalEvaluator::new();
        // alice has name AND age AND email — matching ?s ?p ?o gives 3 results
        let left = vec![mapping(&[("s", "alice")])];
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "?p", "?val")]);
        let d = data();
        let (result, stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        assert_eq!(result.len(), 3);
        assert_eq!(stats.joined_count, 1);
    }

    // ── Filter with BOUND check ─────────────────────────────────────────────

    #[test]
    fn test_optional_filter_bound_check() {
        let eval = OptionalEvaluator::new();
        let left = vec![mapping(&[("s", "alice")])];
        let clause = OptionalClause::new(vec![OptionalTriplePattern::new("?s", "name", "?n")])
            .with_filter(FilterExpr::Bound("n".into()));
        let d = data();
        let (result, _stats) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("n"), Some("Alice"));
    }

    // ── Statistics ──────────────────────────────────────────────────────────

    #[test]
    fn test_stats_default() {
        let stats = OptionalStats::default();
        assert_eq!(stats.left_solutions, 0);
        assert_eq!(stats.right_solutions, 0);
        assert_eq!(stats.joined_count, 0);
        assert_eq!(stats.unmatched_count, 0);
        assert_eq!(stats.nested_evaluations, 0);
        assert_eq!(stats.filtered_count, 0);
    }

    #[test]
    fn test_evaluator_default() {
        let eval = OptionalEvaluator::default();
        let left = vec![mapping(&[("s", "alice")])];
        let clause = OptionalClause::new(vec![]);
        let d = data();
        let (result, _) = eval
            .evaluate(&left, &clause, &d)
            .expect("evaluation should succeed");
        assert!(!result.is_empty());
    }
}
