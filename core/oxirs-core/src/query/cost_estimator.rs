/// SPARQL query cost estimator for join ordering.
///
/// Provides cardinality estimation and cost-based join reordering
/// to produce efficient SPARQL query execution plans.
use std::collections::HashMap;

/// Cost estimate for a SPARQL query pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct CostEstimate {
    /// Estimated number of result tuples (cardinality).
    pub cardinality: usize,
    /// Estimated CPU cost (arbitrary units).
    pub cpu_cost: f64,
    /// Estimated I/O cost (arbitrary units).
    pub io_cost: f64,
    /// Total combined cost.
    pub total_cost: f64,
}

impl CostEstimate {
    /// Create a new cost estimate with computed total.
    pub fn new(cardinality: usize, cpu_cost: f64, io_cost: f64) -> Self {
        let total_cost = cpu_cost + io_cost;
        Self {
            cardinality,
            cpu_cost,
            io_cost,
            total_cost,
        }
    }

    /// Zero-cost estimate (e.g., for empty patterns).
    pub fn zero() -> Self {
        Self::new(0, 0.0, 0.0)
    }

    /// Unit cost estimate with cardinality 1.
    pub fn unit() -> Self {
        Self::new(1, 1.0, 1.0)
    }
}

/// Type of SPARQL pattern for cost estimation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternType {
    /// Basic triple pattern.
    TriplePattern,
    /// Inner join of two patterns.
    Join,
    /// Left outer join (OPTIONAL).
    LeftJoin,
    /// Union of two patterns.
    Union,
    /// Filter applied to a pattern.
    Filter,
    /// Graph pattern (named graph).
    Graph,
    /// Optional pattern (alias for LeftJoin).
    Optional,
}

/// SPARQL query cost estimator using predicate statistics.
///
/// Uses collected predicate frequency statistics to estimate
/// cardinality and cost of SPARQL query patterns.
pub struct CostEstimator {
    /// Predicate → triple count mapping.
    predicate_stats: HashMap<String, usize>,
    /// Total number of triples in the dataset.
    total_triples: usize,
}

impl CostEstimator {
    /// Create a new cost estimator with empty statistics.
    pub fn new() -> Self {
        Self {
            predicate_stats: HashMap::new(),
            total_triples: 0,
        }
    }

    /// Set the total number of triples in the dataset.
    pub fn set_total_triples(&mut self, n: usize) {
        self.total_triples = n;
    }

    /// Record how many triples use a given predicate.
    pub fn record_predicate_count(&mut self, predicate: &str, count: usize) {
        self.predicate_stats.insert(predicate.to_string(), count);
    }

    /// Estimate cost of a basic triple pattern.
    ///
    /// Cardinality is estimated from predicate statistics when the
    /// predicate is bound, otherwise from total triples adjusted by
    /// how many terms are bound (subject/object).
    pub fn estimate_triple_pattern(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> CostEstimate {
        if self.total_triples == 0 {
            return CostEstimate::zero();
        }

        // Estimate cardinality based on bound terms.
        let cardinality = self.estimate_triple_cardinality(subject, predicate, object);
        let io_cost = cardinality as f64 * 0.1;
        let cpu_cost = cardinality as f64 * 0.05;
        CostEstimate::new(cardinality, cpu_cost, io_cost)
    }

    fn estimate_triple_cardinality(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> usize {
        let total = self.total_triples;
        let bound_count =
            subject.is_some() as u32 + predicate.is_some() as u32 + object.is_some() as u32;

        match bound_count {
            3 => {
                // All bound → typically 0 or 1 result.
                1
            }
            2 => {
                if let Some(pred) = predicate {
                    // Predicate + one more bound: divide predicate count by estimated distinct values.
                    let pred_count = *self.predicate_stats.get(pred).unwrap_or(&(total / 10));
                    // Assume ~10 distinct subjects or objects per predicate.
                    (pred_count / 10).max(1)
                } else {
                    // Subject + object bound, no predicate: very selective.
                    (total / 100).max(1)
                }
            }
            1 => {
                if let Some(pred) = predicate {
                    // Only predicate bound: return all triples with that predicate.
                    *self.predicate_stats.get(pred).unwrap_or(&(total / 10))
                } else if subject.is_some() {
                    // Only subject bound: assume ~10 predicates per subject.
                    (total / 10).max(1)
                } else {
                    // Only object bound: assume moderate selectivity.
                    (total / 5).max(1)
                }
            }
            _ => {
                // Nothing bound: full scan.
                total
            }
        }
    }

    /// Estimate cost of joining two patterns.
    ///
    /// Uses a nested-loop join model: left × right with a selectivity factor.
    pub fn estimate_join(&self, left: &CostEstimate, right: &CostEstimate) -> CostEstimate {
        // Assume 10% join selectivity.
        let selectivity = 0.1_f64;
        let cardinality =
            ((left.cardinality as f64 * right.cardinality as f64 * selectivity) as usize).max(1);
        let cpu_cost = left.cpu_cost + right.cpu_cost + cardinality as f64 * 0.1;
        let io_cost = left.io_cost + right.io_cost;
        CostEstimate::new(cardinality, cpu_cost, io_cost)
    }

    /// Estimate cost of a UNION of two patterns.
    ///
    /// Cardinality is the sum of both sides.
    pub fn estimate_union(&self, left: &CostEstimate, right: &CostEstimate) -> CostEstimate {
        let cardinality = left.cardinality + right.cardinality;
        let cpu_cost = left.cpu_cost + right.cpu_cost + cardinality as f64 * 0.02;
        let io_cost = left.io_cost + right.io_cost;
        CostEstimate::new(cardinality, cpu_cost, io_cost)
    }

    /// Estimate cost of applying a filter with given selectivity.
    ///
    /// `selectivity` is in `[0.0, 1.0]` — fraction of tuples that pass the filter.
    pub fn estimate_filter(&self, inner: &CostEstimate, selectivity: f64) -> CostEstimate {
        let selectivity = selectivity.clamp(0.0, 1.0);
        let cardinality = (inner.cardinality as f64 * selectivity) as usize;
        // Filter adds CPU cost for evaluation.
        let cpu_cost = inner.cpu_cost + inner.cardinality as f64 * 0.01;
        let io_cost = inner.io_cost;
        CostEstimate::new(cardinality, cpu_cost, io_cost)
    }

    /// Compute the selectivity of a predicate (fraction of total triples).
    ///
    /// Returns a value in `(0.0, 1.0]`.
    pub fn selectivity_predicate(&self, predicate: &str) -> f64 {
        if self.total_triples == 0 {
            return 1.0;
        }
        let count = *self
            .predicate_stats
            .get(predicate)
            .unwrap_or(&self.total_triples);
        (count as f64 / self.total_triples as f64).clamp(1e-9, 1.0)
    }

    /// Order a list of patterns by ascending total cost.
    ///
    /// Returns the patterns sorted from cheapest to most expensive,
    /// which is the optimal order for a left-deep join tree.
    pub fn order_joins(&self, mut patterns: Vec<CostEstimate>) -> Vec<CostEstimate> {
        patterns.sort_by(|a, b| {
            a.total_cost
                .partial_cmp(&b.total_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        patterns
    }

    /// Return the number of predicates with recorded statistics.
    pub fn predicate_count(&self) -> usize {
        self.predicate_stats.len()
    }

    /// Return total triples setting.
    pub fn total_triples(&self) -> usize {
        self.total_triples
    }
}

impl Default for CostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_estimator() -> CostEstimator {
        let mut est = CostEstimator::new();
        est.set_total_triples(10_000);
        est.record_predicate_count("rdf:type", 3_000);
        est.record_predicate_count("foaf:name", 500);
        est.record_predicate_count("owl:sameAs", 100);
        est
    }

    // --- CostEstimate tests ---

    #[test]
    fn test_cost_estimate_new() {
        let ce = CostEstimate::new(100, 5.0, 3.0);
        assert_eq!(ce.cardinality, 100);
        assert!((ce.cpu_cost - 5.0).abs() < 1e-9);
        assert!((ce.io_cost - 3.0).abs() < 1e-9);
        assert!((ce.total_cost - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_cost_estimate_zero() {
        let ce = CostEstimate::zero();
        assert_eq!(ce.cardinality, 0);
        assert_eq!(ce.total_cost, 0.0);
    }

    #[test]
    fn test_cost_estimate_unit() {
        let ce = CostEstimate::unit();
        assert_eq!(ce.cardinality, 1);
        assert!(ce.total_cost > 0.0);
    }

    #[test]
    fn test_cost_estimate_clone() {
        let ce = CostEstimate::new(42, 1.0, 2.0);
        let ce2 = ce.clone();
        assert_eq!(ce, ce2);
    }

    #[test]
    fn test_cost_estimate_total_is_sum() {
        let ce = CostEstimate::new(50, 7.5, 2.5);
        assert!((ce.total_cost - 10.0).abs() < 1e-9);
    }

    // --- CostEstimator construction ---

    #[test]
    fn test_new_estimator_defaults() {
        let est = CostEstimator::new();
        assert_eq!(est.total_triples(), 0);
        assert_eq!(est.predicate_count(), 0);
    }

    #[test]
    fn test_default_is_same_as_new() {
        let est = CostEstimator::default();
        assert_eq!(est.total_triples(), 0);
    }

    #[test]
    fn test_set_total_triples() {
        let mut est = CostEstimator::new();
        est.set_total_triples(50_000);
        assert_eq!(est.total_triples(), 50_000);
    }

    #[test]
    fn test_record_predicate_count() {
        let mut est = CostEstimator::new();
        est.record_predicate_count("ex:p", 200);
        assert_eq!(est.predicate_count(), 1);
    }

    #[test]
    fn test_record_multiple_predicates() {
        let mut est = CostEstimator::new();
        est.record_predicate_count("ex:p1", 100);
        est.record_predicate_count("ex:p2", 200);
        est.record_predicate_count("ex:p3", 300);
        assert_eq!(est.predicate_count(), 3);
    }

    #[test]
    fn test_record_overwrite_predicate() {
        let mut est = CostEstimator::new();
        est.record_predicate_count("ex:p", 100);
        est.record_predicate_count("ex:p", 500);
        // Count should be overwritten.
        assert_eq!(est.predicate_count(), 1);
    }

    // --- estimate_triple_pattern ---

    #[test]
    fn test_estimate_full_scan() {
        let est = make_estimator();
        let ce = est.estimate_triple_pattern(None, None, None);
        // Full scan = total_triples.
        assert_eq!(ce.cardinality, 10_000);
    }

    #[test]
    fn test_estimate_predicate_only() {
        let est = make_estimator();
        let ce = est.estimate_triple_pattern(None, Some("rdf:type"), None);
        assert_eq!(ce.cardinality, 3_000);
    }

    #[test]
    fn test_estimate_rare_predicate() {
        let est = make_estimator();
        let ce = est.estimate_triple_pattern(None, Some("owl:sameAs"), None);
        assert_eq!(ce.cardinality, 100);
    }

    #[test]
    fn test_estimate_unknown_predicate_fallback() {
        let est = make_estimator();
        let ce = est.estimate_triple_pattern(None, Some("ex:unknown"), None);
        // Falls back to total/10.
        assert_eq!(ce.cardinality, 1_000);
    }

    #[test]
    fn test_estimate_subject_only() {
        let est = make_estimator();
        let ce = est.estimate_triple_pattern(Some("ex:s"), None, None);
        assert!(ce.cardinality > 0);
        assert!(ce.cardinality <= 10_000);
    }

    #[test]
    fn test_estimate_object_only() {
        let est = make_estimator();
        let ce = est.estimate_triple_pattern(None, None, Some("ex:o"));
        assert!(ce.cardinality > 0);
        assert!(ce.cardinality <= 10_000);
    }

    #[test]
    fn test_estimate_subject_predicate() {
        let est = make_estimator();
        let ce = est.estimate_triple_pattern(Some("ex:s"), Some("rdf:type"), None);
        // Predicate known, two terms bound → predicate_count / 10.
        assert_eq!(ce.cardinality, 300);
    }

    #[test]
    fn test_estimate_all_bound() {
        let est = make_estimator();
        let ce = est.estimate_triple_pattern(Some("ex:s"), Some("ex:p"), Some("ex:o"));
        assert_eq!(ce.cardinality, 1);
    }

    #[test]
    fn test_estimate_subject_object_no_predicate() {
        let est = make_estimator();
        let ce = est.estimate_triple_pattern(Some("ex:s"), None, Some("ex:o"));
        // total/100 = 100.
        assert_eq!(ce.cardinality, 100);
    }

    #[test]
    fn test_estimate_zero_total_triples() {
        let est = CostEstimator::new(); // total = 0
        let ce = est.estimate_triple_pattern(None, None, None);
        assert_eq!(ce.cardinality, 0);
        assert_eq!(ce.total_cost, 0.0);
    }

    #[test]
    fn test_estimate_costs_are_positive() {
        let est = make_estimator();
        let ce = est.estimate_triple_pattern(None, Some("rdf:type"), None);
        assert!(ce.cpu_cost >= 0.0);
        assert!(ce.io_cost >= 0.0);
        assert!(ce.total_cost > 0.0);
    }

    // --- estimate_join ---

    #[test]
    fn test_estimate_join_basic() {
        let est = make_estimator();
        let left = CostEstimate::new(100, 5.0, 2.0);
        let right = CostEstimate::new(200, 8.0, 3.0);
        let joined = est.estimate_join(&left, &right);
        // 100 * 200 * 0.1 = 2000.
        assert_eq!(joined.cardinality, 2000);
        assert!(joined.cpu_cost > left.cpu_cost + right.cpu_cost);
        assert!(joined.io_cost >= left.io_cost + right.io_cost);
    }

    #[test]
    fn test_estimate_join_zero_left() {
        let est = make_estimator();
        let left = CostEstimate::zero();
        let right = CostEstimate::new(100, 5.0, 2.0);
        let joined = est.estimate_join(&left, &right);
        // 0 * 100 * 0.1 = 0, clamped to 1.
        assert_eq!(joined.cardinality, 1);
    }

    #[test]
    fn test_estimate_join_commutativity_cost() {
        let est = make_estimator();
        let a = CostEstimate::new(50, 3.0, 1.0);
        let b = CostEstimate::new(200, 7.0, 2.0);
        let ab = est.estimate_join(&a, &b);
        let ba = est.estimate_join(&b, &a);
        // Cardinality should be same regardless of order.
        assert_eq!(ab.cardinality, ba.cardinality);
    }

    // --- estimate_union ---

    #[test]
    fn test_estimate_union_basic() {
        let est = make_estimator();
        let left = CostEstimate::new(100, 5.0, 2.0);
        let right = CostEstimate::new(200, 8.0, 3.0);
        let union = est.estimate_union(&left, &right);
        assert_eq!(union.cardinality, 300);
        assert!(union.cpu_cost >= left.cpu_cost + right.cpu_cost);
    }

    #[test]
    fn test_estimate_union_zero_sides() {
        let est = make_estimator();
        let zero = CostEstimate::zero();
        let union = est.estimate_union(&zero, &zero);
        assert_eq!(union.cardinality, 0);
    }

    #[test]
    fn test_estimate_union_asymmetric() {
        let est = make_estimator();
        let small = CostEstimate::new(10, 1.0, 0.5);
        let large = CostEstimate::new(1000, 50.0, 20.0);
        let union = est.estimate_union(&small, &large);
        assert_eq!(union.cardinality, 1010);
    }

    // --- estimate_filter ---

    #[test]
    fn test_estimate_filter_half_selectivity() {
        let est = make_estimator();
        let inner = CostEstimate::new(1000, 10.0, 5.0);
        let filtered = est.estimate_filter(&inner, 0.5);
        assert_eq!(filtered.cardinality, 500);
    }

    #[test]
    fn test_estimate_filter_zero_selectivity() {
        let est = make_estimator();
        let inner = CostEstimate::new(1000, 10.0, 5.0);
        let filtered = est.estimate_filter(&inner, 0.0);
        assert_eq!(filtered.cardinality, 0);
    }

    #[test]
    fn test_estimate_filter_full_selectivity() {
        let est = make_estimator();
        let inner = CostEstimate::new(1000, 10.0, 5.0);
        let filtered = est.estimate_filter(&inner, 1.0);
        assert_eq!(filtered.cardinality, 1000);
    }

    #[test]
    fn test_estimate_filter_clamps_above_one() {
        let est = make_estimator();
        let inner = CostEstimate::new(100, 5.0, 2.0);
        let filtered = est.estimate_filter(&inner, 1.5);
        assert_eq!(filtered.cardinality, 100);
    }

    #[test]
    fn test_estimate_filter_clamps_below_zero() {
        let est = make_estimator();
        let inner = CostEstimate::new(100, 5.0, 2.0);
        let filtered = est.estimate_filter(&inner, -0.5);
        assert_eq!(filtered.cardinality, 0);
    }

    #[test]
    fn test_estimate_filter_adds_cpu_cost() {
        let est = make_estimator();
        let inner = CostEstimate::new(1000, 10.0, 5.0);
        let filtered = est.estimate_filter(&inner, 0.5);
        // cpu_cost should include evaluation overhead.
        assert!(filtered.cpu_cost > 10.0);
    }

    // --- selectivity_predicate ---

    #[test]
    fn test_selectivity_known_predicate() {
        let est = make_estimator();
        let sel = est.selectivity_predicate("rdf:type");
        // 3000 / 10000 = 0.3.
        assert!((sel - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_selectivity_rare_predicate() {
        let est = make_estimator();
        let sel = est.selectivity_predicate("owl:sameAs");
        // 100 / 10000 = 0.01.
        assert!((sel - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_selectivity_unknown_predicate() {
        let est = make_estimator();
        let sel = est.selectivity_predicate("ex:unknown");
        // Falls back to total/total = 1.0.
        assert!((sel - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_selectivity_zero_total() {
        let est = CostEstimator::new();
        let sel = est.selectivity_predicate("ex:p");
        assert!((sel - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_selectivity_in_range() {
        let est = make_estimator();
        let sel = est.selectivity_predicate("foaf:name");
        assert!(sel > 0.0);
        assert!(sel <= 1.0);
    }

    // --- order_joins ---

    #[test]
    fn test_order_joins_empty() {
        let est = make_estimator();
        let ordered = est.order_joins(vec![]);
        assert!(ordered.is_empty());
    }

    #[test]
    fn test_order_joins_single() {
        let est = make_estimator();
        let ce = CostEstimate::new(100, 5.0, 3.0);
        let ordered = est.order_joins(vec![ce.clone()]);
        assert_eq!(ordered.len(), 1);
        assert_eq!(ordered[0], ce);
    }

    #[test]
    fn test_order_joins_ascending() {
        let est = make_estimator();
        let patterns = vec![
            CostEstimate::new(1000, 50.0, 30.0), // total = 80
            CostEstimate::new(100, 5.0, 3.0),    // total = 8
            CostEstimate::new(500, 20.0, 15.0),  // total = 35
        ];
        let ordered = est.order_joins(patterns);
        assert!(ordered[0].total_cost <= ordered[1].total_cost);
        assert!(ordered[1].total_cost <= ordered[2].total_cost);
    }

    #[test]
    fn test_order_joins_already_ordered() {
        let est = make_estimator();
        let patterns = vec![
            CostEstimate::new(10, 1.0, 0.5),
            CostEstimate::new(100, 5.0, 3.0),
            CostEstimate::new(1000, 50.0, 30.0),
        ];
        let ordered = est.order_joins(patterns);
        assert!(ordered[0].total_cost <= ordered[1].total_cost);
        assert!(ordered[1].total_cost <= ordered[2].total_cost);
    }

    #[test]
    fn test_order_joins_equal_costs() {
        let est = make_estimator();
        let ce = CostEstimate::new(100, 5.0, 3.0);
        let patterns = vec![ce.clone(), ce.clone(), ce.clone()];
        let ordered = est.order_joins(patterns);
        assert_eq!(ordered.len(), 3);
    }

    #[test]
    fn test_order_joins_preserves_all_estimates() {
        let est = make_estimator();
        let patterns = vec![
            CostEstimate::new(50, 3.0, 1.0),
            CostEstimate::new(200, 10.0, 5.0),
        ];
        let ordered = est.order_joins(patterns);
        assert_eq!(ordered.len(), 2);
        // The cheaper one should be first.
        assert!(ordered[0].total_cost <= ordered[1].total_cost);
    }

    // --- Integration tests ---

    #[test]
    fn test_full_pipeline_select_with_filter() {
        let est = make_estimator();
        let tp = est.estimate_triple_pattern(None, Some("rdf:type"), None);
        let filtered = est.estimate_filter(&tp, 0.1);
        assert!(filtered.cardinality < tp.cardinality);
    }

    #[test]
    fn test_full_pipeline_join_order() {
        let est = make_estimator();
        let tp1 = est.estimate_triple_pattern(None, Some("rdf:type"), None);
        let tp2 = est.estimate_triple_pattern(None, Some("foaf:name"), None);
        let tp3 = est.estimate_triple_pattern(None, Some("owl:sameAs"), None);
        let ordered = est.order_joins(vec![tp1, tp2, tp3]);
        assert!(ordered[0].total_cost <= ordered[1].total_cost);
        assert!(ordered[1].total_cost <= ordered[2].total_cost);
    }

    #[test]
    fn test_chain_join_then_filter() {
        let est = make_estimator();
        let a = CostEstimate::new(100, 5.0, 2.0);
        let b = CostEstimate::new(50, 3.0, 1.0);
        let joined = est.estimate_join(&a, &b);
        let filtered = est.estimate_filter(&joined, 0.2);
        assert!(filtered.cardinality <= joined.cardinality);
    }

    #[test]
    fn test_union_then_filter() {
        let est = make_estimator();
        let a = CostEstimate::new(100, 5.0, 2.0);
        let b = CostEstimate::new(100, 5.0, 2.0);
        let union = est.estimate_union(&a, &b);
        let filtered = est.estimate_filter(&union, 0.5);
        assert_eq!(filtered.cardinality, 100);
    }

    #[test]
    fn test_pattern_type_variants() {
        // Ensure all variants are constructible.
        let _variants = [
            PatternType::TriplePattern,
            PatternType::Join,
            PatternType::LeftJoin,
            PatternType::Union,
            PatternType::Filter,
            PatternType::Graph,
            PatternType::Optional,
        ];
    }
}
