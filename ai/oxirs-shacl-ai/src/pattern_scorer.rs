//! SHACL shape pattern scoring and ranking.
//!
//! Provides scoring and ranking of SHACL shape patterns extracted from
//! RDF data, enabling selection of the most useful shape constraints.

/// A SHACL shape pattern extracted from data.
#[derive(Debug, Clone)]
pub struct ShapePattern {
    /// Unique identifier for this pattern.
    pub pattern_id: String,
    /// Optional target class this pattern applies to.
    pub target_class: Option<String>,
    /// List of property paths included in this pattern.
    pub property_paths: Vec<String>,
    /// Number of nodes matching this pattern.
    pub support: usize,
    /// Fraction of target class nodes matching (in \[0.0, 1.0\]).
    pub confidence: f64,
}

/// Scoring criteria for ranking patterns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoringCriteria {
    /// Rank by support (frequency).
    BySupport,
    /// Rank by confidence.
    ByConfidence,
    /// Rank by F1 = 2 * support * confidence / (support + confidence).
    ByF1,
    /// Rank by (support as f64 * confidence) combined score.
    ByCombined,
}

/// Scorer that ranks SHACL shape patterns.
pub struct PatternScorer;

impl Default for PatternScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternScorer {
    /// Create a new `PatternScorer`.
    pub fn new() -> Self {
        PatternScorer
    }

    /// Compute the score for a single pattern under the given criteria.
    pub fn score(&self, pattern: &ShapePattern, criteria: ScoringCriteria) -> f64 {
        match criteria {
            ScoringCriteria::BySupport => pattern.support as f64,
            ScoringCriteria::ByConfidence => pattern.confidence,
            ScoringCriteria::ByF1 => {
                let s = pattern.support as f64;
                let c = pattern.confidence;
                let denom = s + c;
                if denom < 1e-12 {
                    0.0
                } else {
                    2.0 * s * c / denom
                }
            }
            ScoringCriteria::ByCombined => pattern.support as f64 * pattern.confidence,
        }
    }

    /// Return indices of patterns sorted in descending order of score.
    pub fn rank(&self, patterns: &[ShapePattern], criteria: ScoringCriteria) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..patterns.len()).collect();
        indices.sort_by(|&a, &b| {
            let score_a = self.score(&patterns[a], criteria);
            let score_b = self.score(&patterns[b], criteria);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Return up to `k` top-scoring patterns in descending order.
    pub fn top_k<'a>(
        &self,
        patterns: &'a [ShapePattern],
        k: usize,
        criteria: ScoringCriteria,
    ) -> Vec<&'a ShapePattern> {
        let ranked = self.rank(patterns, criteria);
        ranked.into_iter().take(k).map(|i| &patterns[i]).collect()
    }

    /// Filter patterns whose confidence is at least `min_confidence`.
    pub fn filter_by_confidence<'a>(
        &self,
        patterns: &'a [ShapePattern],
        min_confidence: f64,
    ) -> Vec<&'a ShapePattern> {
        patterns
            .iter()
            .filter(|p| p.confidence >= min_confidence)
            .collect()
    }

    /// Filter patterns whose support is at least `min_support`.
    pub fn filter_by_support<'a>(
        &self,
        patterns: &'a [ShapePattern],
        min_support: usize,
    ) -> Vec<&'a ShapePattern> {
        patterns
            .iter()
            .filter(|p| p.support >= min_support)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pattern(id: &str, support: usize, confidence: f64) -> ShapePattern {
        ShapePattern {
            pattern_id: id.to_string(),
            target_class: Some("http://example.org/Class".to_string()),
            property_paths: vec!["ex:name".to_string(), "ex:age".to_string()],
            support,
            confidence,
        }
    }

    fn scorer() -> PatternScorer {
        PatternScorer::new()
    }

    // --- score: BySupport ---

    #[test]
    fn test_score_by_support() {
        let p = make_pattern("p1", 42, 0.8);
        let s = scorer();
        assert!((s.score(&p, ScoringCriteria::BySupport) - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_score_by_support_zero() {
        let p = make_pattern("p1", 0, 0.5);
        let s = scorer();
        assert!((s.score(&p, ScoringCriteria::BySupport)).abs() < 1e-9);
    }

    // --- score: ByConfidence ---

    #[test]
    fn test_score_by_confidence() {
        let p = make_pattern("p1", 10, 0.75);
        let s = scorer();
        assert!((s.score(&p, ScoringCriteria::ByConfidence) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_score_by_confidence_full() {
        let p = make_pattern("p1", 10, 1.0);
        let s = scorer();
        assert!((s.score(&p, ScoringCriteria::ByConfidence) - 1.0).abs() < 1e-9);
    }

    // --- score: ByF1 ---

    #[test]
    fn test_score_by_f1_basic() {
        let p = make_pattern("p1", 4, 0.5);
        let s = scorer();
        // F1 = 2 * 4 * 0.5 / (4 + 0.5) = 4.0 / 4.5
        let expected = 2.0 * 4.0 * 0.5 / (4.0 + 0.5);
        assert!((s.score(&p, ScoringCriteria::ByF1) - expected).abs() < 1e-9);
    }

    #[test]
    fn test_score_by_f1_zero_support() {
        let p = make_pattern("p1", 0, 0.0);
        let s = scorer();
        assert!((s.score(&p, ScoringCriteria::ByF1)).abs() < 1e-9);
    }

    #[test]
    fn test_score_by_f1_high_support() {
        let p = make_pattern("p1", 1000, 0.9);
        let s = scorer();
        let score = s.score(&p, ScoringCriteria::ByF1);
        // F1 = 2 * 1000 * 0.9 / 1000.9 ≈ 1.798
        assert!(score > 1.0 && score < 2.0, "F1 score unexpected: {score}");
    }

    // --- score: ByCombined ---

    #[test]
    fn test_score_by_combined() {
        let p = make_pattern("p1", 10, 0.6);
        let s = scorer();
        assert!((s.score(&p, ScoringCriteria::ByCombined) - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_score_by_combined_zero() {
        let p = make_pattern("p1", 0, 0.9);
        let s = scorer();
        assert!((s.score(&p, ScoringCriteria::ByCombined)).abs() < 1e-9);
    }

    // --- rank ---

    #[test]
    fn test_rank_by_support_descending() {
        let patterns = vec![
            make_pattern("a", 5, 0.9),
            make_pattern("b", 20, 0.1),
            make_pattern("c", 10, 0.5),
        ];
        let s = scorer();
        let ranked = s.rank(&patterns, ScoringCriteria::BySupport);
        assert_eq!(ranked[0], 1); // support=20
        assert_eq!(ranked[1], 2); // support=10
        assert_eq!(ranked[2], 0); // support=5
    }

    #[test]
    fn test_rank_by_confidence_descending() {
        let patterns = vec![
            make_pattern("a", 5, 0.3),
            make_pattern("b", 20, 0.9),
            make_pattern("c", 10, 0.6),
        ];
        let s = scorer();
        let ranked = s.rank(&patterns, ScoringCriteria::ByConfidence);
        assert_eq!(ranked[0], 1); // confidence=0.9
        assert_eq!(ranked[1], 2); // confidence=0.6
        assert_eq!(ranked[2], 0); // confidence=0.3
    }

    #[test]
    fn test_rank_by_f1() {
        let patterns = vec![make_pattern("a", 1, 0.99), make_pattern("b", 100, 0.5)];
        let s = scorer();
        let ranked = s.rank(&patterns, ScoringCriteria::ByF1);
        // b has higher F1 due to high support
        assert_eq!(ranked[0], 1);
    }

    #[test]
    fn test_rank_by_combined() {
        let patterns = vec![
            make_pattern("a", 10, 0.5), // combined=5.0
            make_pattern("b", 3, 0.9),  // combined=2.7
            make_pattern("c", 8, 0.8),  // combined=6.4
        ];
        let s = scorer();
        let ranked = s.rank(&patterns, ScoringCriteria::ByCombined);
        assert_eq!(ranked[0], 2); // 6.4
        assert_eq!(ranked[1], 0); // 5.0
        assert_eq!(ranked[2], 1); // 2.7
    }

    #[test]
    fn test_rank_empty_patterns() {
        let s = scorer();
        let ranked = s.rank(&[], ScoringCriteria::BySupport);
        assert!(ranked.is_empty());
    }

    #[test]
    fn test_rank_single_pattern() {
        let patterns = vec![make_pattern("only", 42, 0.5)];
        let s = scorer();
        let ranked = s.rank(&patterns, ScoringCriteria::BySupport);
        assert_eq!(ranked, vec![0]);
    }

    // --- top_k ---

    #[test]
    fn test_top_k_returns_k_patterns() {
        let patterns: Vec<ShapePattern> = (0..10)
            .map(|i| make_pattern(&format!("p{i}"), i * 10, 0.5))
            .collect();
        let s = scorer();
        let top = s.top_k(&patterns, 3, ScoringCriteria::BySupport);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_top_k_k_greater_than_len() {
        let patterns = vec![make_pattern("a", 5, 0.9), make_pattern("b", 10, 0.5)];
        let s = scorer();
        let top = s.top_k(&patterns, 100, ScoringCriteria::BySupport);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_top_k_zero() {
        let patterns = vec![make_pattern("a", 5, 0.9)];
        let s = scorer();
        let top = s.top_k(&patterns, 0, ScoringCriteria::BySupport);
        assert!(top.is_empty());
    }

    #[test]
    fn test_top_k_descending_order() {
        let patterns = vec![
            make_pattern("low", 1, 0.1),
            make_pattern("high", 100, 0.9),
            make_pattern("mid", 50, 0.5),
        ];
        let s = scorer();
        let top = s.top_k(&patterns, 2, ScoringCriteria::BySupport);
        assert_eq!(top[0].pattern_id, "high");
        assert_eq!(top[1].pattern_id, "mid");
    }

    #[test]
    fn test_top_k_empty_patterns() {
        let s = scorer();
        let top = s.top_k(&[], 5, ScoringCriteria::BySupport);
        assert!(top.is_empty());
    }

    // --- filter_by_confidence ---

    #[test]
    fn test_filter_by_confidence_basic() {
        let patterns = vec![
            make_pattern("a", 5, 0.3),
            make_pattern("b", 5, 0.7),
            make_pattern("c", 5, 0.5),
        ];
        let s = scorer();
        let filtered = s.filter_by_confidence(&patterns, 0.5);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|p| p.confidence >= 0.5));
    }

    #[test]
    fn test_filter_by_confidence_all_pass() {
        let patterns: Vec<ShapePattern> = (0..5)
            .map(|i| make_pattern(&format!("p{i}"), i, 1.0))
            .collect();
        let s = scorer();
        let filtered = s.filter_by_confidence(&patterns, 0.0);
        assert_eq!(filtered.len(), 5);
    }

    #[test]
    fn test_filter_by_confidence_none_pass() {
        let patterns: Vec<ShapePattern> = (0..5)
            .map(|i| make_pattern(&format!("p{i}"), i, 0.1))
            .collect();
        let s = scorer();
        let filtered = s.filter_by_confidence(&patterns, 0.9);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_by_confidence_empty() {
        let s = scorer();
        let filtered = s.filter_by_confidence(&[], 0.5);
        assert!(filtered.is_empty());
    }

    // --- filter_by_support ---

    #[test]
    fn test_filter_by_support_basic() {
        let patterns = vec![
            make_pattern("a", 2, 0.5),
            make_pattern("b", 10, 0.5),
            make_pattern("c", 5, 0.5),
        ];
        let s = scorer();
        let filtered = s.filter_by_support(&patterns, 5);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|p| p.support >= 5));
    }

    #[test]
    fn test_filter_by_support_zero_min() {
        let patterns: Vec<ShapePattern> = (0..4)
            .map(|i| make_pattern(&format!("p{i}"), i, 0.5))
            .collect();
        let s = scorer();
        let filtered = s.filter_by_support(&patterns, 0);
        assert_eq!(filtered.len(), 4);
    }

    #[test]
    fn test_filter_by_support_none_pass() {
        let patterns = vec![make_pattern("a", 1, 0.9), make_pattern("b", 2, 0.9)];
        let s = scorer();
        let filtered = s.filter_by_support(&patterns, 100);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_by_support_empty() {
        let s = scorer();
        let filtered = s.filter_by_support(&[], 1);
        assert!(filtered.is_empty());
    }

    // --- Misc / Default ---

    #[test]
    fn test_default() {
        let _ = PatternScorer;
    }

    #[test]
    fn test_shape_pattern_debug() {
        let p = make_pattern("p1", 5, 0.9);
        let s = format!("{p:?}");
        assert!(s.contains("p1"));
    }

    #[test]
    fn test_scoring_criteria_eq() {
        assert_eq!(ScoringCriteria::BySupport, ScoringCriteria::BySupport);
        assert_ne!(ScoringCriteria::BySupport, ScoringCriteria::ByConfidence);
    }

    #[test]
    fn test_pattern_without_target_class() {
        let p = ShapePattern {
            pattern_id: "p_no_class".to_string(),
            target_class: None,
            property_paths: vec!["ex:prop".to_string()],
            support: 10,
            confidence: 0.8,
        };
        let s = scorer();
        let score = s.score(&p, ScoringCriteria::ByCombined);
        assert!((score - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_rank_returns_all_indices() {
        let patterns: Vec<ShapePattern> = (0..5)
            .map(|i| make_pattern(&format!("p{i}"), i * 3, 0.5))
            .collect();
        let s = scorer();
        let ranked = s.rank(&patterns, ScoringCriteria::BySupport);
        let mut sorted_ranked = ranked.clone();
        sorted_ranked.sort();
        assert_eq!(sorted_ranked, vec![0, 1, 2, 3, 4]);
    }
}
