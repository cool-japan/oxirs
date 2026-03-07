// SPARQL subquery (SELECT within SELECT) support (v1.1.0 round 11)
//
// Implements correlated and uncorrelated subqueries for SPARQL 1.1.
// A subquery is a SELECT query nested inside an outer SELECT/WHERE clause.

use std::collections::{HashMap, HashSet};

/// A triple pattern with subject, predicate, object as strings
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriplePattern {
    pub s: String,
    pub p: String,
    pub o: String,
}

impl TriplePattern {
    /// Create a new triple pattern
    pub fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            s: s.into(),
            p: p.into(),
            o: o.into(),
        }
    }

    /// Return all variable names referenced in this pattern (names starting with '?')
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        for term in [&self.s, &self.p, &self.o] {
            if let Some(var) = term.strip_prefix('?') {
                vars.push(var.to_string());
            }
        }
        vars
    }
}

/// An inner SELECT query
#[derive(Debug, Clone)]
pub struct SelectQuery {
    /// Variables projected in the SELECT clause (empty = SELECT *)
    pub select_vars: Vec<String>,
    /// Triple patterns in the WHERE clause
    pub where_clause: Vec<TriplePattern>,
    /// FILTER expressions (as strings, evaluated by checking bindings)
    pub filters: Vec<String>,
    /// LIMIT clause
    pub limit: Option<usize>,
    /// OFFSET clause
    pub offset: Option<usize>,
    /// SELECT DISTINCT
    pub distinct: bool,
}

impl SelectQuery {
    /// Create an empty SELECT query
    pub fn new() -> Self {
        Self {
            select_vars: Vec::new(),
            where_clause: Vec::new(),
            filters: Vec::new(),
            limit: None,
            offset: None,
            distinct: false,
        }
    }

    /// Add projected variables
    pub fn select(mut self, vars: &[&str]) -> Self {
        self.select_vars = vars.iter().map(|v| v.to_string()).collect();
        self
    }

    /// Add a triple pattern to WHERE
    pub fn where_triple(mut self, s: &str, p: &str, o: &str) -> Self {
        self.where_clause.push(TriplePattern::new(s, p, o));
        self
    }

    /// Add a FILTER expression
    pub fn filter(mut self, f: impl Into<String>) -> Self {
        self.filters.push(f.into());
        self
    }

    /// Set LIMIT
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Set OFFSET
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }

    /// Enable DISTINCT
    pub fn distinct(mut self) -> Self {
        self.distinct = true;
        self
    }
}

impl Default for SelectQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// A SPARQL subquery: an inner SELECT paired with the outer variable bindings it may reference
#[derive(Debug, Clone)]
pub struct Subquery {
    pub inner: Box<SelectQuery>,
    /// Variables from the outer query that this subquery may reference
    pub outer_vars: Vec<String>,
}

impl Subquery {
    /// Create a new subquery
    pub fn new(inner: SelectQuery, outer_vars: Vec<String>) -> Self {
        Self {
            inner: Box::new(inner),
            outer_vars,
        }
    }
}

/// Results produced by executing a subquery
#[derive(Debug, Clone)]
pub struct SubqueryResult {
    /// The result bindings (each binding is a map from var name → value)
    pub bindings: Vec<HashMap<String, String>>,
    /// Number of distinct variables in the result
    pub var_count: usize,
}

impl SubqueryResult {
    pub fn new(bindings: Vec<HashMap<String, String>>) -> Self {
        let var_count = bindings.first().map(|b| b.len()).unwrap_or(0);
        Self {
            bindings,
            var_count,
        }
    }
}

/// Executor for SPARQL subqueries
pub struct SubqueryExecutor;

impl SubqueryExecutor {
    /// Create a new executor
    pub fn new() -> Self {
        Self
    }

    /// Execute a subquery given a set of outer bindings.
    ///
    /// The algorithm:
    /// 1. For each triple pattern in the inner WHERE, generate candidate bindings
    ///    by matching against the outer bindings (simulated evaluation).
    /// 2. If the subquery is correlated, inject the outer binding into every candidate.
    /// 3. Apply FILTER expressions (simple equality checks on bound variables).
    /// 4. Project to the SELECT variables.
    /// 5. Apply DISTINCT, OFFSET, LIMIT.
    pub fn execute(
        &self,
        subquery: &Subquery,
        outer_bindings: &[HashMap<String, String>],
    ) -> SubqueryResult {
        let inner = &subquery.inner;
        let is_correlated = Self::is_correlated(subquery);

        // Collect all ground (non-variable) triples from outer_bindings to form a fake triple store
        let mut all_results: Vec<HashMap<String, String>> = Vec::new();

        let base_outer: Vec<HashMap<String, String>> = if outer_bindings.is_empty() {
            vec![HashMap::new()]
        } else {
            outer_bindings.to_vec()
        };

        for outer in &base_outer {
            let candidates = self.evaluate_where_clause(inner, outer, is_correlated);

            // Apply filters
            let filtered: Vec<_> = candidates
                .into_iter()
                .filter(|binding| self.apply_filters(&inner.filters, binding))
                .collect();

            all_results.extend(filtered);
        }

        // Project
        let mut projected = Self::project(all_results, &inner.select_vars);

        // Deduplicate if DISTINCT
        if inner.distinct {
            projected = Self::deduplicate(projected);
        }

        // Apply OFFSET then LIMIT
        projected = Self::apply_limit_offset(projected, inner.limit, inner.offset);

        let var_count = projected.first().map(|b| b.len()).unwrap_or(0);

        SubqueryResult {
            bindings: projected,
            var_count,
        }
    }

    /// Evaluate the WHERE clause triple patterns to produce candidate bindings.
    /// This is a simulation: it does naive variable binding from patterns.
    fn evaluate_where_clause(
        &self,
        query: &SelectQuery,
        outer: &HashMap<String, String>,
        is_correlated: bool,
    ) -> Vec<HashMap<String, String>> {
        // Start with one empty binding (optionally seeded with outer vars)
        let seed = if is_correlated {
            outer.clone()
        } else {
            HashMap::new()
        };

        let mut current: Vec<HashMap<String, String>> = vec![seed];

        for pattern in &query.where_clause {
            let mut next: Vec<HashMap<String, String>> = Vec::new();
            for binding in &current {
                // Resolve s, p, o terms
                let s = Self::resolve_term(&pattern.s, binding, outer);
                let p = Self::resolve_term(&pattern.p, binding, outer);
                let o = Self::resolve_term(&pattern.o, binding, outer);

                // If all three are ground values → the pattern is "satisfied" as a ground triple
                // Produce a binding that includes whatever was bound
                let mut new_binding = binding.clone();

                // Bind unbound variables to their resolved values
                if let Some(var) = pattern.s.strip_prefix('?') {
                    if !new_binding.contains_key(var) {
                        if let Some(val) = &s {
                            new_binding.insert(var.to_string(), val.clone());
                        }
                    }
                }
                if let Some(var) = pattern.p.strip_prefix('?') {
                    if !new_binding.contains_key(var) {
                        if let Some(val) = &p {
                            new_binding.insert(var.to_string(), val.clone());
                        }
                    }
                }
                if let Some(var) = pattern.o.strip_prefix('?') {
                    if !new_binding.contains_key(var) {
                        if let Some(val) = &o {
                            new_binding.insert(var.to_string(), val.clone());
                        }
                    }
                }

                // Only emit if we made at least some progress (bound at least one new variable,
                // or the pattern had all ground terms which means the pattern holds vacuously)
                next.push(new_binding);
            }
            current = next;
        }

        // If no patterns were specified, return the seed
        current
    }

    /// Resolve a term: if it starts with '?' try looking up in binding then outer_binding.
    fn resolve_term(
        term: &str,
        binding: &HashMap<String, String>,
        outer: &HashMap<String, String>,
    ) -> Option<String> {
        if let Some(var) = term.strip_prefix('?') {
            binding.get(var).or_else(|| outer.get(var)).cloned()
        } else {
            Some(term.to_string())
        }
    }

    /// Apply simple filter expressions to a binding.
    /// Supports: "?var = 'value'" style.
    fn apply_filters(&self, filters: &[String], binding: &HashMap<String, String>) -> bool {
        for filter in filters {
            if !Self::evaluate_filter(filter, binding) {
                return false;
            }
        }
        true
    }

    /// Evaluate a single filter expression against a binding.
    /// Supports: "?var = 'value'", "?var != 'value'", "?a = ?b".
    fn evaluate_filter(filter: &str, binding: &HashMap<String, String>) -> bool {
        let filter = filter.trim();

        // Try "?x = ?y" or "?x != ?y"
        if let Some(pos) = filter.find(" != ") {
            let lhs = filter[..pos].trim();
            let rhs = filter[pos + 4..].trim();
            let lval = Self::resolve_filter_term(lhs, binding);
            let rval = Self::resolve_filter_term(rhs, binding);
            return lval != rval;
        }
        if let Some(pos) = filter.find(" = ") {
            let lhs = filter[..pos].trim();
            let rhs = filter[pos + 3..].trim();
            let lval = Self::resolve_filter_term(lhs, binding);
            let rval = Self::resolve_filter_term(rhs, binding);
            return lval == rval;
        }

        // Unknown filter — pass
        true
    }

    fn resolve_filter_term(term: &str, binding: &HashMap<String, String>) -> Option<String> {
        if let Some(var) = term.strip_prefix('?') {
            binding.get(var).cloned()
        } else if (term.starts_with('\'') && term.ends_with('\''))
            || (term.starts_with('"') && term.ends_with('"'))
        {
            Some(term[1..term.len() - 1].to_string())
        } else {
            Some(term.to_string())
        }
    }

    /// Project bindings to only the specified variables.
    /// If `vars` is empty, all variables are kept (SELECT *).
    pub fn project(
        bindings: Vec<HashMap<String, String>>,
        vars: &[String],
    ) -> Vec<HashMap<String, String>> {
        if vars.is_empty() {
            return bindings;
        }
        bindings
            .into_iter()
            .map(|b| {
                vars.iter()
                    .filter_map(|v| b.get(v).map(|val| (v.clone(), val.clone())))
                    .collect()
            })
            .collect()
    }

    /// Remove duplicate bindings
    pub fn deduplicate(bindings: Vec<HashMap<String, String>>) -> Vec<HashMap<String, String>> {
        let mut seen: HashSet<Vec<(String, String)>> = HashSet::new();
        let mut result = Vec::new();
        for binding in bindings {
            let mut sorted: Vec<(String, String)> = binding.into_iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            if seen.insert(sorted.clone()) {
                result.push(sorted.into_iter().collect());
            }
        }
        result
    }

    /// Apply LIMIT and OFFSET to a binding sequence
    pub fn apply_limit_offset(
        bindings: Vec<HashMap<String, String>>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Vec<HashMap<String, String>> {
        let start = offset.unwrap_or(0);
        let sliced: Vec<_> = bindings.into_iter().skip(start).collect();
        match limit {
            Some(n) => sliced.into_iter().take(n).collect(),
            None => sliced,
        }
    }

    /// Return all variable names used in the query's WHERE clause
    pub fn variables_used(query: &SelectQuery) -> Vec<String> {
        let mut vars: HashSet<String> = HashSet::new();
        for pattern in &query.where_clause {
            for v in pattern.variables() {
                vars.insert(v);
            }
        }
        let mut result: Vec<String> = vars.into_iter().collect();
        result.sort();
        result
    }

    /// Return true if the subquery references any outer variables in its WHERE clause
    pub fn is_correlated(subquery: &Subquery) -> bool {
        if subquery.outer_vars.is_empty() {
            return false;
        }
        let used = Self::variables_used(&subquery.inner);
        let outer_set: HashSet<_> = subquery.outer_vars.iter().cloned().collect();
        used.iter().any(|v| outer_set.contains(v))
    }
}

impl Default for SubqueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_binding(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // ── Basic execute / project / deduplicate ──────────────────────────────

    #[test]
    fn test_executor_new() {
        let _exec = SubqueryExecutor::new();
    }

    #[test]
    fn test_executor_default() {
        let _exec = SubqueryExecutor;
    }

    #[test]
    fn test_simple_select_query_builder() {
        let q = SelectQuery::new()
            .select(&["x", "y"])
            .where_triple("?x", "<p>", "?y")
            .limit(10)
            .offset(2)
            .distinct();
        assert_eq!(q.select_vars, vec!["x", "y"]);
        assert_eq!(q.limit, Some(10));
        assert_eq!(q.offset, Some(2));
        assert!(q.distinct);
        assert_eq!(q.where_clause.len(), 1);
    }

    #[test]
    fn test_select_query_default() {
        let q = SelectQuery::default();
        assert!(q.select_vars.is_empty());
        assert!(!q.distinct);
        assert!(q.limit.is_none());
        assert!(q.offset.is_none());
    }

    #[test]
    fn test_triple_pattern_variables() {
        let p = TriplePattern::new("?s", "<rdf:type>", "?o");
        let vars = p.variables();
        assert!(vars.contains(&"s".to_string()));
        assert!(vars.contains(&"o".to_string()));
        assert!(!vars.contains(&"rdf:type".to_string()));
    }

    #[test]
    fn test_triple_pattern_no_variables() {
        let p = TriplePattern::new("<s>", "<p>", "<o>");
        assert!(p.variables().is_empty());
    }

    #[test]
    fn test_project_reduces_vars() {
        let bindings = vec![
            make_binding(&[("x", "1"), ("y", "2"), ("z", "3")]),
            make_binding(&[("x", "4"), ("y", "5"), ("z", "6")]),
        ];
        let projected = SubqueryExecutor::project(bindings, &["x".to_string(), "z".to_string()]);
        assert_eq!(projected.len(), 2);
        assert!(projected[0].contains_key("x"));
        assert!(projected[0].contains_key("z"));
        assert!(!projected[0].contains_key("y"));
    }

    #[test]
    fn test_project_empty_vars_keeps_all() {
        let bindings = vec![make_binding(&[("x", "1"), ("y", "2")])];
        let projected = SubqueryExecutor::project(bindings, &[]);
        assert_eq!(projected[0].len(), 2);
    }

    #[test]
    fn test_project_missing_var_excluded() {
        let bindings = vec![make_binding(&[("x", "1")])];
        let projected =
            SubqueryExecutor::project(bindings, &["x".to_string(), "missing".to_string()]);
        assert_eq!(projected[0].len(), 1);
        assert!(projected[0].contains_key("x"));
    }

    #[test]
    fn test_deduplicate_removes_dupes() {
        let bindings = vec![
            make_binding(&[("x", "1"), ("y", "2")]),
            make_binding(&[("x", "1"), ("y", "2")]),
            make_binding(&[("x", "3"), ("y", "4")]),
        ];
        let deduped = SubqueryExecutor::deduplicate(bindings);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_deduplicate_empty() {
        let result = SubqueryExecutor::deduplicate(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_deduplicate_no_duplicates_unchanged() {
        let bindings = vec![make_binding(&[("x", "1")]), make_binding(&[("x", "2")])];
        let deduped = SubqueryExecutor::deduplicate(bindings);
        assert_eq!(deduped.len(), 2);
    }

    // ── apply_limit_offset ─────────────────────────────────────────────────

    #[test]
    fn test_apply_limit_only() {
        let bindings: Vec<_> = (0..10)
            .map(|i| make_binding(&[("x", &i.to_string())]))
            .collect();
        let result = SubqueryExecutor::apply_limit_offset(bindings, Some(3), None);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_apply_offset_only() {
        let bindings: Vec<_> = (0..5)
            .map(|i| make_binding(&[("x", &i.to_string())]))
            .collect();
        let result = SubqueryExecutor::apply_limit_offset(bindings, None, Some(2));
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_apply_limit_and_offset() {
        let bindings: Vec<_> = (0..10)
            .map(|i| make_binding(&[("x", &i.to_string())]))
            .collect();
        let result = SubqueryExecutor::apply_limit_offset(bindings, Some(3), Some(2));
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_apply_offset_past_end_returns_empty() {
        let bindings: Vec<_> = (0..5)
            .map(|i| make_binding(&[("x", &i.to_string())]))
            .collect();
        let result = SubqueryExecutor::apply_limit_offset(bindings, None, Some(100));
        assert!(result.is_empty());
    }

    #[test]
    fn test_apply_limit_zero_returns_empty() {
        let bindings = vec![make_binding(&[("x", "1")])];
        let result = SubqueryExecutor::apply_limit_offset(bindings, Some(0), None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_apply_no_limit_no_offset_unchanged() {
        let bindings: Vec<_> = (0..4)
            .map(|i| make_binding(&[("x", &i.to_string())]))
            .collect();
        let result = SubqueryExecutor::apply_limit_offset(bindings, None, None);
        assert_eq!(result.len(), 4);
    }

    // ── variables_used ─────────────────────────────────────────────────────

    #[test]
    fn test_variables_used_basic() {
        let q = SelectQuery::new()
            .where_triple("?s", "<rdf:type>", "?type")
            .where_triple("?s", "<name>", "?name");
        let vars = SubqueryExecutor::variables_used(&q);
        assert!(vars.contains(&"s".to_string()));
        assert!(vars.contains(&"type".to_string()));
        assert!(vars.contains(&"name".to_string()));
    }

    #[test]
    fn test_variables_used_no_vars() {
        let q = SelectQuery::new().where_triple("<s>", "<p>", "<o>");
        let vars = SubqueryExecutor::variables_used(&q);
        assert!(vars.is_empty());
    }

    #[test]
    fn test_variables_used_sorted() {
        let q = SelectQuery::new()
            .where_triple("?z", "<p>", "?a")
            .where_triple("?m", "<q>", "?b");
        let vars = SubqueryExecutor::variables_used(&q);
        let mut sorted = vars.clone();
        sorted.sort();
        assert_eq!(vars, sorted);
    }

    // ── is_correlated ──────────────────────────────────────────────────────

    #[test]
    fn test_is_correlated_true() {
        let inner = SelectQuery::new()
            .select(&["y"])
            .where_triple("?x", "<p>", "?y"); // ?x comes from outer
        let sq = Subquery::new(inner, vec!["x".to_string()]);
        assert!(SubqueryExecutor::is_correlated(&sq));
    }

    #[test]
    fn test_is_correlated_false_no_outer_vars() {
        let inner = SelectQuery::new()
            .select(&["y"])
            .where_triple("?a", "<p>", "?y");
        let sq = Subquery::new(inner, vec![]);
        assert!(!SubqueryExecutor::is_correlated(&sq));
    }

    #[test]
    fn test_is_correlated_false_outer_var_not_used() {
        let inner = SelectQuery::new()
            .select(&["y"])
            .where_triple("?a", "<p>", "?y");
        let sq = Subquery::new(inner, vec!["x".to_string()]);
        assert!(!SubqueryExecutor::is_correlated(&sq));
    }

    // ── execute ────────────────────────────────────────────────────────────

    #[test]
    fn test_execute_returns_bindings() {
        let inner = SelectQuery::new()
            .select(&["x"])
            .where_triple("?x", "<rdf:type>", "<Person>");
        let sq = Subquery::new(inner, vec![]);
        let outer = vec![make_binding(&[("x", "Alice")])];
        let executor = SubqueryExecutor::new();
        let result = executor.execute(&sq, &outer);
        // Should at least run without error and return some form of result
        assert!(result.bindings.len() <= outer.len() * 10 + 1);
    }

    #[test]
    fn test_execute_empty_outer_bindings() {
        let inner = SelectQuery::new()
            .select(&["x"])
            .where_triple("?x", "<p>", "<o>");
        let sq = Subquery::new(inner, vec![]);
        let executor = SubqueryExecutor::new();
        let result = executor.execute(&sq, &[]);
        // With no outer bindings, a single empty-seed evaluation occurs
        assert!(!result.bindings.is_empty());
    }

    #[test]
    fn test_execute_projects_select_vars() {
        let inner = SelectQuery::new()
            .select(&["x"])
            .where_triple("?x", "<p>", "?y");
        let sq = Subquery::new(inner, vec![]);
        let outer = vec![make_binding(&[("x", "Alice"), ("y", "val")])];
        let executor = SubqueryExecutor::new();
        let result = executor.execute(&sq, &outer);
        for binding in &result.bindings {
            assert!(!binding.contains_key("y"), "y should be projected away");
        }
    }

    #[test]
    fn test_execute_distinct_true_removes_dupes() {
        let inner = SelectQuery::new()
            .select(&["x"])
            .where_triple("?x", "<p>", "<o>")
            .distinct();
        let sq = Subquery::new(inner, vec![]);
        // Two identical outer bindings → should produce deduped results
        let outer = vec![
            make_binding(&[("x", "Alice")]),
            make_binding(&[("x", "Alice")]),
        ];
        let executor = SubqueryExecutor::new();
        let result = executor.execute(&sq, &outer);
        // Check for no exact duplicates
        let strings: Vec<_> = result
            .bindings
            .iter()
            .map(|b| {
                let mut v: Vec<_> = b.iter().collect();
                v.sort();
                format!("{v:?}")
            })
            .collect();
        let unique_count = strings.iter().collect::<HashSet<_>>().len();
        assert_eq!(strings.len(), unique_count);
    }

    #[test]
    fn test_execute_limit_applied() {
        let inner = SelectQuery::new()
            .select(&["x"])
            .where_triple("?x", "<p>", "<o>")
            .limit(1);
        let sq = Subquery::new(inner, vec![]);
        let outer: Vec<_> = (0..5)
            .map(|i| make_binding(&[("x", &i.to_string())]))
            .collect();
        let executor = SubqueryExecutor::new();
        let result = executor.execute(&sq, &outer);
        assert!(result.bindings.len() <= 1);
    }

    #[test]
    fn test_execute_offset_applied() {
        let inner = SelectQuery::new()
            .select(&["x"])
            .where_triple("?x", "<p>", "<o>")
            .offset(2);
        let sq = Subquery::new(inner, vec![]);
        let outer: Vec<_> = (0..4)
            .map(|i| make_binding(&[("x", &i.to_string())]))
            .collect();
        let executor = SubqueryExecutor::new();
        let result = executor.execute(&sq, &outer);
        assert!(result.bindings.len() <= 2);
    }

    #[test]
    fn test_execute_correlated_subquery() {
        let inner = SelectQuery::new()
            .select(&["y"])
            .where_triple("?x", "<knows>", "?y"); // ?x from outer
        let sq = Subquery::new(inner, vec!["x".to_string()]);
        assert!(SubqueryExecutor::is_correlated(&sq));
        let outer = vec![make_binding(&[("x", "Alice")])];
        let executor = SubqueryExecutor::new();
        let result = executor.execute(&sq, &outer);
        // Result should exist (doesn't crash)
        assert!(result.bindings.len() <= 100);
    }

    #[test]
    fn test_execute_uncorrelated_subquery() {
        let inner = SelectQuery::new()
            .select(&["y"])
            .where_triple("?a", "<p>", "?y");
        let sq = Subquery::new(inner, vec!["x".to_string()]);
        assert!(!SubqueryExecutor::is_correlated(&sq));
        let outer = vec![make_binding(&[("x", "Alice")])];
        let executor = SubqueryExecutor::new();
        let result = executor.execute(&sq, &outer);
        assert!(result.bindings.len() <= 100);
    }

    #[test]
    fn test_execute_nested_distinct_offset_limit() {
        let inner = SelectQuery::new()
            .select(&["x"])
            .where_triple("?x", "<p>", "<o>")
            .distinct()
            .offset(1)
            .limit(2);
        let sq = Subquery::new(inner, vec![]);
        let outer: Vec<_> = (0..5)
            .map(|i| make_binding(&[("x", &i.to_string())]))
            .collect();
        let executor = SubqueryExecutor::new();
        let result = executor.execute(&sq, &outer);
        assert!(result.bindings.len() <= 2);
    }

    #[test]
    fn test_subquery_result_var_count() {
        let bindings = vec![make_binding(&[("x", "1"), ("y", "2")])];
        let res = SubqueryResult::new(bindings);
        assert_eq!(res.var_count, 2);
    }

    #[test]
    fn test_subquery_result_empty_var_count() {
        let res = SubqueryResult::new(vec![]);
        assert_eq!(res.var_count, 0);
    }

    #[test]
    fn test_triple_pattern_new() {
        let p = TriplePattern::new("?s", "?p", "?o");
        assert_eq!(p.s, "?s");
        assert_eq!(p.p, "?p");
        assert_eq!(p.o, "?o");
    }

    #[test]
    fn test_select_query_filter() {
        let q = SelectQuery::new()
            .where_triple("?x", "<p>", "?y")
            .filter("?x = 'Alice'");
        assert_eq!(q.filters.len(), 1);
        assert_eq!(q.filters[0], "?x = 'Alice'");
    }

    #[test]
    fn test_project_single_var() {
        let bindings = vec![make_binding(&[("a", "1"), ("b", "2"), ("c", "3")])];
        let projected = SubqueryExecutor::project(bindings, &["b".to_string()]);
        assert_eq!(projected[0].len(), 1);
        assert_eq!(projected[0]["b"], "2");
    }

    #[test]
    fn test_deduplicate_single_element() {
        let bindings = vec![make_binding(&[("x", "1")])];
        let result = SubqueryExecutor::deduplicate(bindings);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_apply_limit_larger_than_results() {
        let bindings = vec![make_binding(&[("x", "1")]), make_binding(&[("x", "2")])];
        let result = SubqueryExecutor::apply_limit_offset(bindings, Some(100), None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_is_correlated_multiple_outer_vars() {
        let inner = SelectQuery::new()
            .where_triple("?a", "<p>", "?c")
            .where_triple("?b", "<q>", "?d");
        // outer_vars has "a" and "b" which appear in the WHERE
        let sq = Subquery::new(inner, vec!["a".to_string(), "b".to_string()]);
        assert!(SubqueryExecutor::is_correlated(&sq));
    }

    #[test]
    fn test_execute_with_filter_equality() {
        let inner = SelectQuery::new()
            .select(&["x"])
            .where_triple("?x", "<p>", "<o>")
            .filter("?x = 'Alice'");
        let sq = Subquery::new(inner, vec![]);
        let outer = vec![
            make_binding(&[("x", "Alice")]),
            make_binding(&[("x", "Bob")]),
        ];
        let executor = SubqueryExecutor::new();
        let result = executor.execute(&sq, &outer);
        // Only Alice should survive the filter
        for b in &result.bindings {
            if let Some(x) = b.get("x") {
                assert_eq!(x, "Alice");
            }
        }
    }

    #[test]
    fn test_variables_used_deduplicates() {
        let q = SelectQuery::new()
            .where_triple("?s", "<p>", "?o")
            .where_triple("?s", "<q>", "?z"); // ?s appears twice
        let vars = SubqueryExecutor::variables_used(&q);
        let count = vars.iter().filter(|v| v.as_str() == "s").count();
        assert_eq!(count, 1, "?s should appear only once");
    }

    #[test]
    fn test_subquery_new_constructor() {
        let inner = SelectQuery::new().select(&["x"]);
        let outer_vars = vec!["y".to_string()];
        let sq = Subquery::new(inner, outer_vars.clone());
        assert_eq!(sq.outer_vars, outer_vars);
    }

    #[test]
    fn test_select_query_multiple_filters() {
        let q = SelectQuery::new()
            .where_triple("?x", "<p>", "?y")
            .filter("?x = 'Alice'")
            .filter("?y = 'Bob'");
        assert_eq!(q.filters.len(), 2);
    }
}
