//! SHACL constraint ranking by violation severity and impact.
//!
//! Ranks constraints according to configurable criteria such as violation count,
//! affected node count, severity level, fix-ability, and composite weighted scores.

/// Severity level of a SHACL constraint violation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SeverityLevel {
    Info,
    Warning,
    Error,
    Critical,
}

impl SeverityLevel {
    /// Numeric score used in ranking calculations.
    pub fn score(&self) -> f64 {
        match self {
            SeverityLevel::Info => 1.0,
            SeverityLevel::Warning => 2.0,
            SeverityLevel::Error => 3.0,
            SeverityLevel::Critical => 4.0,
        }
    }
}

/// Description of a single SHACL constraint and its violation statistics.
#[derive(Debug, Clone)]
pub struct ConstraintInfo {
    /// Unique constraint identifier (e.g. `sh:MinCountConstraint`)
    pub id: String,
    /// Human-readable constraint type name
    pub type_name: String,
    /// Total number of violations recorded
    pub violation_count: usize,
    /// Number of distinct graph nodes affected
    pub affected_nodes: usize,
    /// Severity level of the constraint
    pub severity: SeverityLevel,
    /// Whether the violation can be auto-fixed
    pub auto_fixable: bool,
}

/// Criteria used to rank constraints.
#[derive(Debug, Clone)]
pub enum RankingCriteria {
    /// Rank by severity level (Critical first)
    BySeverity,
    /// Rank by total violation count (highest first)
    ByViolationCount,
    /// Rank by number of affected nodes (highest first)
    ByAffectedNodes,
    /// Rank auto-fixable constraints first, then by violation count
    ByFixability,
    /// Weighted combination of severity, violation count, and affected nodes
    Composite {
        severity_weight: f64,
        count_weight: f64,
        node_weight: f64,
    },
}

/// A constraint together with its computed rank and score.
#[derive(Debug, Clone)]
pub struct RankedConstraint {
    pub constraint: ConstraintInfo,
    /// Computed ranking score (higher = more important)
    pub score: f64,
    /// 1-based rank position (1 = highest priority)
    pub rank: usize,
}

/// Engine for ranking SHACL constraints.
pub struct ConstraintRanker {
    constraints: Vec<ConstraintInfo>,
}

impl ConstraintRanker {
    /// Create a new empty ranker.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Add a constraint to the ranker.
    pub fn add_constraint(&mut self, c: ConstraintInfo) {
        self.constraints.push(c);
    }

    /// Return the total number of managed constraints.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Rank all constraints according to `criteria`, assigning 1-based rank positions.
    pub fn rank(&self, criteria: &RankingCriteria) -> Vec<RankedConstraint> {
        if self.constraints.is_empty() {
            return vec![];
        }

        // Compute (index, score) pairs
        let mut scored: Vec<(usize, f64)> = self
            .constraints
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.compute_score(c, criteria)))
            .collect();

        // Sort descending by score; stable secondary sort by id for determinism
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.constraints[a.0].id.cmp(&self.constraints[b.0].id))
        });

        scored
            .into_iter()
            .enumerate()
            .map(|(rank_idx, (orig_idx, score))| RankedConstraint {
                constraint: self.constraints[orig_idx].clone(),
                score,
                rank: rank_idx + 1,
            })
            .collect()
    }

    /// Return the top `n` ranked constraints.
    pub fn top_n(&self, n: usize, criteria: &RankingCriteria) -> Vec<RankedConstraint> {
        self.rank(criteria).into_iter().take(n).collect()
    }

    /// Return references to all auto-fixable constraints (unordered).
    pub fn fixable_constraints(&self) -> Vec<&ConstraintInfo> {
        self.constraints.iter().filter(|c| c.auto_fixable).collect()
    }

    /// Return references to all constraints with the given severity.
    pub fn by_severity(&self, severity: SeverityLevel) -> Vec<&ConstraintInfo> {
        self.constraints
            .iter()
            .filter(|c| c.severity == severity)
            .collect()
    }

    // --- private helpers ---

    /// Compute a comparable score for `constraint` under `criteria`.
    fn compute_score(&self, constraint: &ConstraintInfo, criteria: &RankingCriteria) -> f64 {
        match criteria {
            RankingCriteria::BySeverity => constraint.severity.score(),

            RankingCriteria::ByViolationCount => constraint.violation_count as f64,

            RankingCriteria::ByAffectedNodes => constraint.affected_nodes as f64,

            RankingCriteria::ByFixability => {
                // Auto-fixable gets a large bonus; tie-break by violation count
                let fix_bonus = if constraint.auto_fixable {
                    1_000_000.0
                } else {
                    0.0
                };
                fix_bonus + constraint.violation_count as f64
            }

            RankingCriteria::Composite {
                severity_weight,
                count_weight,
                node_weight,
            } => {
                let sev_score = constraint.severity.score();

                let (max_count, max_nodes) = self.max_counts();
                let norm_count = if max_count > 0 {
                    constraint.violation_count as f64 / max_count as f64
                } else {
                    0.0
                };
                let norm_nodes = if max_nodes > 0 {
                    constraint.affected_nodes as f64 / max_nodes as f64
                } else {
                    0.0
                };

                severity_weight * sev_score + count_weight * norm_count + node_weight * norm_nodes
            }
        }
    }

    /// Return (max_violation_count, max_affected_nodes) across all constraints.
    fn max_counts(&self) -> (usize, usize) {
        let max_count = self
            .constraints
            .iter()
            .map(|c| c.violation_count)
            .max()
            .unwrap_or(0);
        let max_nodes = self
            .constraints
            .iter()
            .map(|c| c.affected_nodes)
            .max()
            .unwrap_or(0);
        (max_count, max_nodes)
    }
}

impl Default for ConstraintRanker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_constraint(
        id: &str,
        violations: usize,
        nodes: usize,
        severity: SeverityLevel,
        fixable: bool,
    ) -> ConstraintInfo {
        ConstraintInfo {
            id: id.to_string(),
            type_name: format!("{id}Type"),
            violation_count: violations,
            affected_nodes: nodes,
            severity,
            auto_fixable: fixable,
        }
    }

    fn sample_ranker() -> ConstraintRanker {
        let mut r = ConstraintRanker::new();
        r.add_constraint(make_constraint("c1", 10, 5, SeverityLevel::Warning, true));
        r.add_constraint(make_constraint("c2", 50, 20, SeverityLevel::Error, false));
        r.add_constraint(make_constraint("c3", 2, 1, SeverityLevel::Info, true));
        r.add_constraint(make_constraint(
            "c4",
            100,
            40,
            SeverityLevel::Critical,
            false,
        ));
        r
    }

    // --- Basic structure tests ---

    #[test]
    fn test_new_ranker_empty() {
        let r = ConstraintRanker::new();
        assert_eq!(r.constraint_count(), 0);
    }

    #[test]
    fn test_add_constraint() {
        let mut r = ConstraintRanker::new();
        r.add_constraint(make_constraint("x", 1, 1, SeverityLevel::Info, false));
        assert_eq!(r.constraint_count(), 1);
    }

    #[test]
    fn test_constraint_count_multiple() {
        let r = sample_ranker();
        assert_eq!(r.constraint_count(), 4);
    }

    // --- BySeverity tests ---

    #[test]
    fn test_rank_by_severity_order() {
        let r = sample_ranker();
        let ranked = r.rank(&RankingCriteria::BySeverity);
        assert_eq!(ranked[0].constraint.id, "c4"); // Critical
        assert_eq!(ranked[1].constraint.id, "c2"); // Error
        assert_eq!(ranked[2].constraint.id, "c1"); // Warning
        assert_eq!(ranked[3].constraint.id, "c3"); // Info
    }

    #[test]
    fn test_rank_by_severity_scores() {
        let r = sample_ranker();
        let ranked = r.rank(&RankingCriteria::BySeverity);
        assert_eq!(ranked[0].score, 4.0);
        assert_eq!(ranked[1].score, 3.0);
        assert_eq!(ranked[2].score, 2.0);
        assert_eq!(ranked[3].score, 1.0);
    }

    #[test]
    fn test_rank_by_severity_rank_numbers() {
        let r = sample_ranker();
        let ranked = r.rank(&RankingCriteria::BySeverity);
        for (i, rc) in ranked.iter().enumerate() {
            assert_eq!(rc.rank, i + 1);
        }
    }

    // --- ByViolationCount tests ---

    #[test]
    fn test_rank_by_violation_count_order() {
        let r = sample_ranker();
        let ranked = r.rank(&RankingCriteria::ByViolationCount);
        assert_eq!(ranked[0].constraint.id, "c4"); // 100
        assert_eq!(ranked[1].constraint.id, "c2"); // 50
        assert_eq!(ranked[2].constraint.id, "c1"); // 10
        assert_eq!(ranked[3].constraint.id, "c3"); // 2
    }

    #[test]
    fn test_rank_by_violation_count_scores() {
        let r = sample_ranker();
        let ranked = r.rank(&RankingCriteria::ByViolationCount);
        assert_eq!(ranked[0].score, 100.0);
        assert_eq!(ranked[1].score, 50.0);
    }

    // --- ByAffectedNodes tests ---

    #[test]
    fn test_rank_by_affected_nodes_order() {
        let r = sample_ranker();
        let ranked = r.rank(&RankingCriteria::ByAffectedNodes);
        assert_eq!(ranked[0].constraint.id, "c4"); // 40 nodes
        assert_eq!(ranked[1].constraint.id, "c2"); // 20 nodes
    }

    #[test]
    fn test_rank_by_affected_nodes_scores() {
        let r = sample_ranker();
        let ranked = r.rank(&RankingCriteria::ByAffectedNodes);
        assert_eq!(ranked[0].score, 40.0);
        assert_eq!(ranked[1].score, 20.0);
    }

    // --- ByFixability tests ---

    #[test]
    fn test_rank_by_fixability_fixable_first() {
        let r = sample_ranker();
        let ranked = r.rank(&RankingCriteria::ByFixability);
        // c1 and c3 are fixable; they should come before c2 and c4
        let fixable_first_two: Vec<bool> = ranked
            .iter()
            .take(2)
            .map(|rc| rc.constraint.auto_fixable)
            .collect();
        assert!(fixable_first_two.iter().all(|&f| f));
    }

    #[test]
    fn test_rank_by_fixability_among_fixable_by_count() {
        let r = sample_ranker();
        let ranked = r.rank(&RankingCriteria::ByFixability);
        // Among fixable: c1 has 10 violations, c3 has 2 → c1 before c3
        let fixable_ids: Vec<&str> = ranked
            .iter()
            .filter(|rc| rc.constraint.auto_fixable)
            .map(|rc| rc.constraint.id.as_str())
            .collect();
        assert_eq!(fixable_ids[0], "c1");
        assert_eq!(fixable_ids[1], "c3");
    }

    // --- top_n tests ---

    #[test]
    fn test_top_n_returns_n() {
        let r = sample_ranker();
        let top2 = r.top_n(2, &RankingCriteria::BySeverity);
        assert_eq!(top2.len(), 2);
    }

    #[test]
    fn test_top_n_correct_order() {
        let r = sample_ranker();
        let top2 = r.top_n(2, &RankingCriteria::BySeverity);
        assert_eq!(top2[0].rank, 1);
        assert_eq!(top2[1].rank, 2);
    }

    #[test]
    fn test_top_n_exceeds_size() {
        let r = sample_ranker();
        let top10 = r.top_n(10, &RankingCriteria::ByViolationCount);
        assert_eq!(top10.len(), 4); // Only 4 constraints exist
    }

    #[test]
    fn test_top_n_zero() {
        let r = sample_ranker();
        let top0 = r.top_n(0, &RankingCriteria::BySeverity);
        assert!(top0.is_empty());
    }

    // --- fixable_constraints tests ---

    #[test]
    fn test_fixable_constraints_count() {
        let r = sample_ranker();
        let fixable = r.fixable_constraints();
        assert_eq!(fixable.len(), 2);
    }

    #[test]
    fn test_fixable_constraints_all_fixable() {
        let r = sample_ranker();
        for c in r.fixable_constraints() {
            assert!(c.auto_fixable);
        }
    }

    #[test]
    fn test_fixable_constraints_empty() {
        let mut r = ConstraintRanker::new();
        r.add_constraint(make_constraint("x", 5, 2, SeverityLevel::Error, false));
        assert!(r.fixable_constraints().is_empty());
    }

    // --- by_severity filter tests ---

    #[test]
    fn test_by_severity_filter_critical() {
        let r = sample_ranker();
        let crits = r.by_severity(SeverityLevel::Critical);
        assert_eq!(crits.len(), 1);
        assert_eq!(crits[0].id, "c4");
    }

    #[test]
    fn test_by_severity_filter_info() {
        let r = sample_ranker();
        let infos = r.by_severity(SeverityLevel::Info);
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].id, "c3");
    }

    #[test]
    fn test_by_severity_filter_absent() {
        let r = sample_ranker();
        // No constraints with Warning severity named differently; let's add one
        let warnings = r.by_severity(SeverityLevel::Warning);
        assert_eq!(warnings.len(), 1);
    }

    // --- Composite scoring tests ---

    #[test]
    fn test_composite_score_positive() {
        let r = sample_ranker();
        let criteria = RankingCriteria::Composite {
            severity_weight: 1.0,
            count_weight: 1.0,
            node_weight: 1.0,
        };
        let ranked = r.rank(&criteria);
        for rc in &ranked {
            assert!(rc.score >= 0.0);
        }
    }

    #[test]
    fn test_composite_score_severity_dominated() {
        let r = sample_ranker();
        // With huge severity weight, critical should be #1
        let criteria = RankingCriteria::Composite {
            severity_weight: 1000.0,
            count_weight: 0.0,
            node_weight: 0.0,
        };
        let ranked = r.rank(&criteria);
        assert_eq!(ranked[0].constraint.id, "c4");
    }

    #[test]
    fn test_composite_score_count_dominated() {
        let r = sample_ranker();
        // With huge count weight, c4 (100 violations) should be #1
        let criteria = RankingCriteria::Composite {
            severity_weight: 0.0,
            count_weight: 1000.0,
            node_weight: 0.0,
        };
        let ranked = r.rank(&criteria);
        assert_eq!(ranked[0].constraint.id, "c4");
    }

    // --- Empty / single constraint edge cases ---

    #[test]
    fn test_rank_empty_ranker() {
        let r = ConstraintRanker::new();
        let ranked = r.rank(&RankingCriteria::BySeverity);
        assert!(ranked.is_empty());
    }

    #[test]
    fn test_rank_single_constraint() {
        let mut r = ConstraintRanker::new();
        r.add_constraint(make_constraint("only", 7, 3, SeverityLevel::Error, true));
        let ranked = r.rank(&RankingCriteria::BySeverity);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].rank, 1);
        assert_eq!(ranked[0].constraint.id, "only");
    }

    #[test]
    fn test_default_impl() {
        let r = ConstraintRanker::default();
        assert_eq!(r.constraint_count(), 0);
    }
}
