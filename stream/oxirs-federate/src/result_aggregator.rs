//! # Federated Query Result Aggregator
//!
//! Merges result sets from multiple SPARQL endpoints using configurable
//! strategies: union, intersection, difference, ordered, and weighted.

use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// Result set returned by a single federated endpoint.
#[derive(Clone, Debug, Default)]
pub struct EndpointResult {
    /// URL of the endpoint that produced this result.
    pub endpoint_url: String,
    /// Rows of variable bindings (variable name → value).
    pub rows: Vec<HashMap<String, String>>,
    /// Time taken by the endpoint to respond.
    pub duration_ms: u64,
    /// Error message if the endpoint failed.
    pub error: Option<String>,
    /// Names of the variables present in the results.
    pub variable_names: Vec<String>,
}

impl EndpointResult {
    /// Create a successful result.
    pub fn new(
        endpoint_url: impl Into<String>,
        rows: Vec<HashMap<String, String>>,
        duration_ms: u64,
        variable_names: Vec<String>,
    ) -> Self {
        Self {
            endpoint_url: endpoint_url.into(),
            rows,
            duration_ms,
            error: None,
            variable_names,
        }
    }

    /// Create a failed result.
    pub fn failed(
        endpoint_url: impl Into<String>,
        error: impl Into<String>,
        duration_ms: u64,
    ) -> Self {
        Self {
            endpoint_url: endpoint_url.into(),
            rows: vec![],
            duration_ms,
            error: Some(error.into()),
            variable_names: vec![],
        }
    }
}

/// Strategy used to combine results from multiple endpoints.
#[derive(Clone, Debug)]
pub enum MergeStrategy {
    /// Concatenate all rows, removing duplicates.
    Union,
    /// Rows present in every endpoint's result set.
    Intersection,
    /// Rows from the first endpoint not present in any other.
    Difference,
    /// Union of all rows, sorted ascending by `primary` variable value.
    Ordered { primary: String },
    /// Interleave rows from endpoints proportionally by weight (round-robin weighted).
    Weighted { weights: HashMap<String, f64> },
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Stable serialisation of a row for deduplication purposes.
fn row_key(row: &HashMap<String, String>) -> String {
    let mut pairs: Vec<(&String, &String)> = row.iter().collect();
    pairs.sort_by_key(|(k, _)| *k);
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(";")
}

// ─────────────────────────────────────────────────────────────────────────────
// ResultAggregator
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregates and manipulates result sets from federated SPARQL endpoints.
#[derive(Debug, Default)]
pub struct ResultAggregator;

impl ResultAggregator {
    /// Create a new aggregator.
    pub fn new() -> Self {
        Self
    }

    /// Merge results according to the chosen strategy.
    pub fn merge(
        &self,
        results: &[EndpointResult],
        strategy: &MergeStrategy,
    ) -> Vec<HashMap<String, String>> {
        match strategy {
            MergeStrategy::Union => {
                let all: Vec<HashMap<String, String>> =
                    results.iter().flat_map(|r| r.rows.clone()).collect();
                self.deduplicate(all)
            }

            MergeStrategy::Intersection => {
                if results.is_empty() {
                    return vec![];
                }
                // Start with first endpoint's rows, keep only those present in all others.
                let mut candidate_keys: HashSet<String> =
                    results[0].rows.iter().map(row_key).collect();
                for result in results.iter().skip(1) {
                    let keys: HashSet<String> = result.rows.iter().map(row_key).collect();
                    candidate_keys = candidate_keys.intersection(&keys).cloned().collect();
                }
                results[0]
                    .rows
                    .iter()
                    .filter(|r| candidate_keys.contains(&row_key(r)))
                    .cloned()
                    .collect()
            }

            MergeStrategy::Difference => {
                if results.is_empty() {
                    return vec![];
                }
                // Rows in first endpoint not present in any other.
                let other_keys: HashSet<String> = results
                    .iter()
                    .skip(1)
                    .flat_map(|r| r.rows.iter().map(row_key))
                    .collect();
                results[0]
                    .rows
                    .iter()
                    .filter(|r| !other_keys.contains(&row_key(r)))
                    .cloned()
                    .collect()
            }

            MergeStrategy::Ordered { primary } => {
                let mut all: Vec<HashMap<String, String>> =
                    results.iter().flat_map(|r| r.rows.clone()).collect();
                all = self.deduplicate(all);
                all.sort_by(|a, b| {
                    let av = a.get(primary.as_str()).map(String::as_str).unwrap_or("");
                    let bv = b.get(primary.as_str()).map(String::as_str).unwrap_or("");
                    av.cmp(bv)
                });
                all
            }

            MergeStrategy::Weighted { weights } => {
                // Weighted round-robin: each endpoint gets a normalised share of slots.
                // We compute integer quota proportional to weight.
                if results.is_empty() {
                    return vec![];
                }
                let total_weight: f64 = results
                    .iter()
                    .map(|r| weights.get(&r.endpoint_url).copied().unwrap_or(1.0))
                    .sum();
                if total_weight == 0.0 {
                    return self.merge(results, &MergeStrategy::Union);
                }

                // Build per-endpoint iterators with a fractional counter.
                let mut iterators: Vec<(std::slice::Iter<'_, HashMap<String, String>>, f64)> =
                    results
                        .iter()
                        .map(|r| {
                            let w =
                                weights.get(&r.endpoint_url).copied().unwrap_or(1.0) / total_weight;
                            (r.rows.iter(), w)
                        })
                        .collect();

                let max_rows = results.iter().map(|r| r.rows.len()).max().unwrap_or(0);
                let mut out = Vec::new();
                let mut seen = HashSet::new();

                // Simple weighted interleaving: emit ceil(weight * max_rows) rows from each.
                for (iter, w) in &mut iterators {
                    let quota = (*w * max_rows as f64).ceil() as usize;
                    for row in iter.by_ref().take(quota) {
                        let k = row_key(row);
                        if seen.insert(k) {
                            out.push(row.clone());
                        }
                    }
                }
                // Drain any remaining rows from endpoints that were under-represented.
                for (iter, _) in iterators.iter_mut() {
                    for row in iter.by_ref() {
                        let k = row_key(row);
                        if seen.insert(k) {
                            out.push(row.clone());
                        }
                    }
                }
                out
            }
        }
    }

    /// Variables present in *all* endpoint results.
    pub fn common_variables(&self, results: &[EndpointResult]) -> Vec<String> {
        if results.is_empty() {
            return vec![];
        }
        let mut common: HashSet<String> = results[0].variable_names.iter().cloned().collect();
        for r in results.iter().skip(1) {
            let set: HashSet<String> = r.variable_names.iter().cloned().collect();
            common = common.intersection(&set).cloned().collect();
        }
        let mut v: Vec<String> = common.into_iter().collect();
        v.sort();
        v
    }

    /// Union of all variable names across all endpoints, sorted.
    pub fn all_variables(&self, results: &[EndpointResult]) -> Vec<String> {
        let mut all: HashSet<String> = HashSet::new();
        for r in results {
            all.extend(r.variable_names.iter().cloned());
        }
        let mut v: Vec<String> = all.into_iter().collect();
        v.sort();
        v
    }

    /// Project rows to keep only the given variable columns.
    pub fn project(
        &self,
        rows: &[HashMap<String, String>],
        vars: &[String],
    ) -> Vec<HashMap<String, String>> {
        rows.iter()
            .map(|row| {
                vars.iter()
                    .filter_map(|v| row.get(v).map(|val| (v.clone(), val.clone())))
                    .collect()
            })
            .collect()
    }

    /// Remove duplicate rows (preserving first occurrence order).
    pub fn deduplicate(&self, rows: Vec<HashMap<String, String>>) -> Vec<HashMap<String, String>> {
        let mut seen = HashSet::new();
        rows.into_iter()
            .filter(|r| seen.insert(row_key(r)))
            .collect()
    }

    /// Compute aggregation statistics for a set of endpoint results.
    pub fn stats(&self, results: &[EndpointResult]) -> AggregationStats {
        let endpoint_count = results.len();
        let error_count = results.iter().filter(|r| r.error.is_some()).count();
        let total_rows: usize = results.iter().map(|r| r.rows.len()).sum();

        let all_rows: Vec<HashMap<String, String>> =
            results.iter().flat_map(|r| r.rows.clone()).collect();
        let unique_rows = self.deduplicate(all_rows).len();

        let avg_duration_ms = if endpoint_count == 0 {
            0.0
        } else {
            results.iter().map(|r| r.duration_ms as f64).sum::<f64>() / endpoint_count as f64
        };

        let slowest_endpoint = results
            .iter()
            .max_by_key(|r| r.duration_ms)
            .map(|r| r.endpoint_url.clone());

        AggregationStats {
            total_rows,
            unique_rows,
            endpoint_count,
            error_count,
            avg_duration_ms,
            slowest_endpoint,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AggregationStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics produced by aggregating multiple endpoint results.
#[derive(Clone, Debug)]
pub struct AggregationStats {
    /// Total number of rows across all endpoints.
    pub total_rows: usize,
    /// Number of distinct rows across all endpoints.
    pub unique_rows: usize,
    /// Number of endpoints queried.
    pub endpoint_count: usize,
    /// Number of endpoints that returned an error.
    pub error_count: usize,
    /// Average response time in milliseconds.
    pub avg_duration_ms: f64,
    /// URL of the endpoint with the highest response time.
    pub slowest_endpoint: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn ep(
        url: &str,
        rows: Vec<HashMap<String, String>>,
        duration_ms: u64,
        vars: &[&str],
    ) -> EndpointResult {
        EndpointResult::new(
            url,
            rows,
            duration_ms,
            vars.iter().map(|v| v.to_string()).collect(),
        )
    }

    fn agg() -> ResultAggregator {
        ResultAggregator::new()
    }

    // ── EndpointResult constructors ──────────────────────────────────────────

    #[test]
    fn test_endpoint_result_new() {
        let r = EndpointResult::new("http://ep1", vec![], 100, vec!["x".into()]);
        assert_eq!(r.endpoint_url, "http://ep1");
        assert!(r.error.is_none());
    }

    #[test]
    fn test_endpoint_result_failed() {
        let r = EndpointResult::failed("http://ep1", "timeout", 500);
        assert!(r.error.is_some());
        assert_eq!(r.rows.len(), 0);
    }

    // ── Union ────────────────────────────────────────────────────────────────

    #[test]
    fn test_union_concatenates_and_deduplicates() {
        let r1 = row(&[("x", "a")]);
        let r2 = row(&[("x", "b")]);
        let dup = row(&[("x", "a")]); // duplicate
        let results = vec![
            ep("http://ep1", vec![r1.clone()], 10, &["x"]),
            ep("http://ep2", vec![r2.clone(), dup], 20, &["x"]),
        ];
        let merged = agg().merge(&results, &MergeStrategy::Union);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_union_single_endpoint() {
        let results = vec![ep("http://ep1", vec![row(&[("x", "v")])], 10, &["x"])];
        let merged = agg().merge(&results, &MergeStrategy::Union);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_union_empty_results() {
        let merged = agg().merge(&[], &MergeStrategy::Union);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_union_all_duplicates() {
        let r = row(&[("x", "same")]);
        let results = vec![
            ep("http://ep1", vec![r.clone()], 10, &["x"]),
            ep("http://ep2", vec![r.clone()], 10, &["x"]),
        ];
        let merged = agg().merge(&results, &MergeStrategy::Union);
        assert_eq!(merged.len(), 1);
    }

    // ── Intersection ─────────────────────────────────────────────────────────

    #[test]
    fn test_intersection_common_rows() {
        let common = row(&[("x", "shared")]);
        let only_ep1 = row(&[("x", "ep1_only")]);
        let results = vec![
            ep("http://ep1", vec![common.clone(), only_ep1], 10, &["x"]),
            ep("http://ep2", vec![common.clone()], 20, &["x"]),
        ];
        let merged = agg().merge(&results, &MergeStrategy::Intersection);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].get("x").map(String::as_str), Some("shared"));
    }

    #[test]
    fn test_intersection_no_common_rows() {
        let results = vec![
            ep("http://ep1", vec![row(&[("x", "a")])], 10, &["x"]),
            ep("http://ep2", vec![row(&[("x", "b")])], 20, &["x"]),
        ];
        let merged = agg().merge(&results, &MergeStrategy::Intersection);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_intersection_empty_endpoints() {
        let merged = agg().merge(&[], &MergeStrategy::Intersection);
        assert!(merged.is_empty());
    }

    // ── Difference ───────────────────────────────────────────────────────────

    #[test]
    fn test_difference_rows_only_in_first() {
        let r_unique = row(&[("x", "unique")]);
        let r_shared = row(&[("x", "shared")]);
        let results = vec![
            ep(
                "http://ep1",
                vec![r_unique.clone(), r_shared.clone()],
                10,
                &["x"],
            ),
            ep("http://ep2", vec![r_shared.clone()], 10, &["x"]),
        ];
        let merged = agg().merge(&results, &MergeStrategy::Difference);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].get("x").map(String::as_str), Some("unique"));
    }

    #[test]
    fn test_difference_all_shared_returns_empty() {
        let r = row(&[("x", "v")]);
        let results = vec![
            ep("http://ep1", vec![r.clone()], 10, &["x"]),
            ep("http://ep2", vec![r.clone()], 10, &["x"]),
        ];
        let merged = agg().merge(&results, &MergeStrategy::Difference);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_difference_single_endpoint_returns_all() {
        let results = vec![ep("http://ep1", vec![row(&[("x", "v")])], 10, &["x"])];
        let merged = agg().merge(&results, &MergeStrategy::Difference);
        assert_eq!(merged.len(), 1);
    }

    // ── Ordered ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ordered_sorts_by_primary() {
        let results = vec![
            ep(
                "http://ep1",
                vec![row(&[("name", "charlie")]), row(&[("name", "alice")])],
                10,
                &["name"],
            ),
            ep("http://ep2", vec![row(&[("name", "bob")])], 10, &["name"]),
        ];
        let merged = agg().merge(
            &results,
            &MergeStrategy::Ordered {
                primary: "name".into(),
            },
        );
        assert_eq!(merged.len(), 3);
        let names: Vec<&str> = merged.iter().map(|r| r["name"].as_str()).collect();
        assert_eq!(names, vec!["alice", "bob", "charlie"]);
    }

    #[test]
    fn test_ordered_deduplicates() {
        let r = row(&[("x", "same")]);
        let results = vec![
            ep("http://ep1", vec![r.clone()], 10, &["x"]),
            ep("http://ep2", vec![r.clone()], 10, &["x"]),
        ];
        let merged = agg().merge(
            &results,
            &MergeStrategy::Ordered {
                primary: "x".into(),
            },
        );
        assert_eq!(merged.len(), 1);
    }

    // ── Weighted ─────────────────────────────────────────────────────────────

    #[test]
    fn test_weighted_includes_rows_from_all_endpoints() {
        let mut weights = HashMap::new();
        weights.insert("http://ep1".to_string(), 1.0);
        weights.insert("http://ep2".to_string(), 1.0);
        let results = vec![
            ep(
                "http://ep1",
                vec![row(&[("x", "a")]), row(&[("x", "b")])],
                10,
                &["x"],
            ),
            ep(
                "http://ep2",
                vec![row(&[("x", "c")]), row(&[("x", "d")])],
                10,
                &["x"],
            ),
        ];
        let merged = agg().merge(&results, &MergeStrategy::Weighted { weights });
        assert_eq!(merged.len(), 4);
    }

    #[test]
    fn test_weighted_empty_results() {
        let merged = agg().merge(
            &[],
            &MergeStrategy::Weighted {
                weights: HashMap::new(),
            },
        );
        assert!(merged.is_empty());
    }

    // ── common_variables ─────────────────────────────────────────────────────

    #[test]
    fn test_common_variables_intersection() {
        let r1 = ep("http://ep1", vec![], 0, &["x", "y"]);
        let r2 = ep("http://ep2", vec![], 0, &["y", "z"]);
        let common = agg().common_variables(&[r1, r2]);
        assert_eq!(common, vec!["y"]);
    }

    #[test]
    fn test_common_variables_empty_endpoints() {
        let common = agg().common_variables(&[]);
        assert!(common.is_empty());
    }

    #[test]
    fn test_common_variables_single_endpoint() {
        let r = ep("http://ep1", vec![], 0, &["a", "b"]);
        let common = agg().common_variables(&[r]);
        assert_eq!(common, vec!["a", "b"]);
    }

    // ── all_variables ────────────────────────────────────────────────────────

    #[test]
    fn test_all_variables_union() {
        let r1 = ep("http://ep1", vec![], 0, &["x", "y"]);
        let r2 = ep("http://ep2", vec![], 0, &["y", "z"]);
        let all = agg().all_variables(&[r1, r2]);
        assert_eq!(all, vec!["x", "y", "z"]);
    }

    #[test]
    fn test_all_variables_empty() {
        let all = agg().all_variables(&[]);
        assert!(all.is_empty());
    }

    // ── project ──────────────────────────────────────────────────────────────

    #[test]
    fn test_project_keeps_only_selected_vars() {
        let rows = vec![row(&[("x", "1"), ("y", "2"), ("z", "3")])];
        let projected = agg().project(&rows, &["x".into(), "z".into()]);
        assert_eq!(projected.len(), 1);
        assert!(projected[0].contains_key("x"));
        assert!(projected[0].contains_key("z"));
        assert!(!projected[0].contains_key("y"));
    }

    #[test]
    fn test_project_missing_var_skipped() {
        let rows = vec![row(&[("x", "1")])];
        let projected = agg().project(&rows, &["x".into(), "missing".into()]);
        assert_eq!(projected[0].len(), 1);
    }

    // ── deduplicate ──────────────────────────────────────────────────────────

    #[test]
    fn test_deduplicate_removes_exact_duplicates() {
        let r = row(&[("x", "v")]);
        let rows = vec![r.clone(), r.clone(), r.clone()];
        let deduped = agg().deduplicate(rows);
        assert_eq!(deduped.len(), 1);
    }

    #[test]
    fn test_deduplicate_preserves_unique_rows() {
        let rows = vec![row(&[("x", "a")]), row(&[("x", "b")])];
        let deduped = agg().deduplicate(rows);
        assert_eq!(deduped.len(), 2);
    }

    // ── stats ────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_total_rows() {
        let results = vec![
            ep(
                "http://ep1",
                vec![row(&[("x", "a")]), row(&[("x", "b")])],
                100,
                &["x"],
            ),
            ep("http://ep2", vec![row(&[("x", "c")])], 200, &["x"]),
        ];
        let s = agg().stats(&results);
        assert_eq!(s.total_rows, 3);
    }

    #[test]
    fn test_stats_unique_rows() {
        let r = row(&[("x", "same")]);
        let results = vec![
            ep("http://ep1", vec![r.clone()], 10, &["x"]),
            ep("http://ep2", vec![r.clone()], 10, &["x"]),
        ];
        let s = agg().stats(&results);
        assert_eq!(s.unique_rows, 1);
    }

    #[test]
    fn test_stats_error_count() {
        let results = vec![
            ep("http://ep1", vec![], 10, &[]),
            EndpointResult::failed("http://ep2", "timeout", 500),
        ];
        let s = agg().stats(&results);
        assert_eq!(s.error_count, 1);
    }

    #[test]
    fn test_stats_slowest_endpoint() {
        let results = vec![
            ep("http://ep1", vec![], 100, &[]),
            ep("http://ep2", vec![], 500, &[]),
        ];
        let s = agg().stats(&results);
        assert_eq!(s.slowest_endpoint.as_deref(), Some("http://ep2"));
    }

    #[test]
    fn test_stats_avg_duration() {
        let results = vec![
            ep("http://ep1", vec![], 100, &[]),
            ep("http://ep2", vec![], 300, &[]),
        ];
        let s = agg().stats(&results);
        assert!((s.avg_duration_ms - 200.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_empty_results() {
        let s = agg().stats(&[]);
        assert_eq!(s.total_rows, 0);
        assert_eq!(s.endpoint_count, 0);
        assert!(s.slowest_endpoint.is_none());
    }
}
