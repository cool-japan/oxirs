//! ML-based SHACL shape recommendations from data statistics.
//!
//! Analyses per-property statistics gathered from RDF data and produces
//! SHACL constraint recommendations with confidence scores and explanations.

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Statistical profile of a property's usage across a sample of RDF subjects.
#[derive(Debug, Clone)]
pub struct DataStats {
    /// The property IRI being profiled.
    pub property: String,
    /// Number of subjects sampled.
    pub sample_count: usize,
    /// Fraction of subjects that have a `null` / missing value (0.0–1.0).
    pub null_rate: f64,
    /// Fraction of non-null values that are distinct (0.0–1.0).
    pub distinct_rate: f64,
    /// Fraction of values that parse as numeric (0.0–1.0).
    pub numeric_rate: f64,
    /// Fraction of values that are IRIs (0.0–1.0).
    pub iri_rate: f64,
    /// Fraction of values that are RDF literals (0.0–1.0).
    pub literal_rate: f64,
    /// Average character/token length of values.
    pub avg_length: f64,
}

/// A SHACL shape recommendation for a single property.
#[derive(Debug, Clone)]
pub struct ShapeRecommendation {
    /// The property this recommendation targets.
    pub property: String,
    /// Constraint names that are recommended (e.g. `"sh:minCount 1"`).
    pub recommended_constraints: Vec<String>,
    /// Aggregate confidence score in `[0.0, 1.0]`.
    pub confidence: f64,
    /// Human-readable explanation of the recommendation rationale.
    pub explanation: String,
}

/// Configuration for the [`ShapeRecommender`].
#[derive(Debug, Clone)]
pub struct RecommenderConfig {
    /// Minimum number of samples required before making any recommendation.
    pub min_samples: usize,
    /// Minimum confidence required to include a constraint in the output.
    pub confidence_threshold: f64,
}

impl Default for RecommenderConfig {
    fn default() -> Self {
        Self {
            min_samples: 10,
            confidence_threshold: 0.5,
        }
    }
}

/// Produces SHACL shape recommendations from property statistics.
pub struct ShapeRecommender {
    config: RecommenderConfig,
}

impl ShapeRecommender {
    /// Create a new `ShapeRecommender` with the supplied configuration.
    pub fn new(config: RecommenderConfig) -> Self {
        Self { config }
    }

    /// Produce a recommendation for a single property's statistics.
    pub fn recommend(&self, stats: &DataStats) -> ShapeRecommendation {
        if stats.sample_count < self.config.min_samples {
            return ShapeRecommendation {
                property: stats.property.clone(),
                recommended_constraints: Vec::new(),
                confidence: 0.0,
                explanation: format!(
                    "Insufficient samples ({} < {})",
                    stats.sample_count, self.config.min_samples
                ),
            };
        }

        let mut constraints = Vec::new();
        let mut explanations = Vec::new();
        let mut total_conf = 0.0;
        let mut conf_count = 0usize;

        // sh:minCount 1
        if let Some(min) = Self::infer_min_count(stats) {
            let conf = Self::confidence_score(stats, "sh:minCount");
            if conf >= self.config.confidence_threshold {
                constraints.push(format!("sh:minCount {min}"));
                explanations.push(format!(
                    "null_rate={:.2} → minCount {}",
                    stats.null_rate, min
                ));
                total_conf += conf;
                conf_count += 1;
            }
        }

        // sh:maxCount 1
        if let Some(max) = Self::infer_max_count(stats) {
            let conf = Self::confidence_score(stats, "sh:maxCount");
            if conf >= self.config.confidence_threshold {
                constraints.push(format!("sh:maxCount {max}"));
                explanations.push(format!(
                    "distinct_rate={:.2} → maxCount {}",
                    stats.distinct_rate, max
                ));
                total_conf += conf;
                conf_count += 1;
            }
        }

        // sh:nodeKind
        let node_kind = Self::infer_node_kind(stats);
        {
            let conf = Self::confidence_score(stats, "sh:nodeKind");
            if conf >= self.config.confidence_threshold {
                constraints.push(format!("sh:nodeKind sh:{node_kind}"));
                explanations.push(format!("iri_rate={:.2} → nodeKind", stats.iri_rate));
                total_conf += conf;
                conf_count += 1;
            }
        }

        // sh:datatype
        if let Some(dt) = Self::infer_datatype(stats) {
            let conf = Self::confidence_score(stats, "sh:datatype");
            if conf >= self.config.confidence_threshold {
                constraints.push(format!("sh:datatype xsd:{dt}"));
                explanations.push(format!("numeric_rate={:.2} → datatype", stats.numeric_rate));
                total_conf += conf;
                conf_count += 1;
            }
        }

        let avg_confidence = if conf_count > 0 {
            total_conf / conf_count as f64
        } else {
            0.0
        };

        ShapeRecommendation {
            property: stats.property.clone(),
            recommended_constraints: constraints,
            confidence: avg_confidence.clamp(0.0, 1.0),
            explanation: if explanations.is_empty() {
                "No constraints exceed the confidence threshold.".to_string()
            } else {
                explanations.join("; ")
            },
        }
    }

    /// Produce recommendations for a slice of property statistics.
    pub fn recommend_all(&self, all_stats: &[DataStats]) -> Vec<ShapeRecommendation> {
        all_stats.iter().map(|s| self.recommend(s)).collect()
    }

    /// Infer the predominant `sh:nodeKind` for the property.
    ///
    /// Returns `"IRI"` when `iri_rate >= 0.8`, `"Literal"` when
    /// `literal_rate >= 0.8`, and `"Both"` otherwise.
    pub fn infer_node_kind(stats: &DataStats) -> &'static str {
        if stats.iri_rate >= 0.8 {
            "IRI"
        } else if stats.literal_rate >= 0.8 {
            "Literal"
        } else {
            "Both"
        }
    }

    /// Infer `sh:minCount` from the null rate.
    ///
    /// Returns `Some(1)` when fewer than 5 % of subjects are missing a value,
    /// `None` otherwise.
    pub fn infer_min_count(stats: &DataStats) -> Option<usize> {
        if stats.null_rate < 0.05 {
            Some(1)
        } else {
            None
        }
    }

    /// Infer `sh:maxCount` from the distinct rate.
    ///
    /// Returns `Some(1)` when more than 95 % of values are distinct (each
    /// subject has at most one value), `None` otherwise.
    pub fn infer_max_count(stats: &DataStats) -> Option<usize> {
        if stats.distinct_rate > 0.95 {
            Some(1)
        } else {
            None
        }
    }

    /// Infer the XSD datatype from the statistics.
    ///
    /// Returns `Some("decimal")` when `numeric_rate >= 0.9` and the
    /// `avg_length` suggests fractional values, `Some("integer")` for pure
    /// integers, and `None` otherwise.
    pub fn infer_datatype(stats: &DataStats) -> Option<&'static str> {
        if stats.numeric_rate >= 0.9 {
            if stats.avg_length > 3.0 {
                Some("decimal")
            } else {
                Some("integer")
            }
        } else {
            None
        }
    }

    /// Compute a confidence score in `[0.0, 1.0]` for a specific constraint
    /// on the given statistics.
    pub fn confidence_score(stats: &DataStats, constraint: &str) -> f64 {
        let sample_factor = (stats.sample_count as f64 / 100.0).min(1.0);

        let signal = match constraint {
            "sh:minCount" => {
                // High confidence when almost no values are missing
                1.0 - stats.null_rate
            }
            "sh:maxCount" => {
                // High confidence when values are highly distinct
                stats.distinct_rate
            }
            "sh:nodeKind" => {
                // High confidence when the kind is unambiguous
                (stats.iri_rate - stats.literal_rate)
                    .abs()
                    .max((stats.literal_rate - stats.iri_rate).abs())
            }
            "sh:datatype" => {
                // High confidence when nearly all values are numeric
                stats.numeric_rate
            }
            _ => 0.5,
        };

        (signal * 0.7 + sample_factor * 0.3).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn stats(
        property: &str,
        sample_count: usize,
        null_rate: f64,
        distinct_rate: f64,
        numeric_rate: f64,
        iri_rate: f64,
        literal_rate: f64,
        avg_length: f64,
    ) -> DataStats {
        DataStats {
            property: property.to_string(),
            sample_count,
            null_rate,
            distinct_rate,
            numeric_rate,
            iri_rate,
            literal_rate,
            avg_length,
        }
    }

    fn default_cfg() -> RecommenderConfig {
        RecommenderConfig {
            min_samples: 10,
            confidence_threshold: 0.3,
        }
    }

    // --- RecommenderConfig ---

    #[test]
    fn test_default_config() {
        let cfg = RecommenderConfig::default();
        assert_eq!(cfg.min_samples, 10);
        assert!((cfg.confidence_threshold - 0.5).abs() < 1e-9);
    }

    // --- infer_min_count ---

    #[test]
    fn test_infer_min_count_low_null_rate() {
        let s = stats("p", 100, 0.02, 0.5, 0.0, 0.0, 1.0, 5.0);
        assert_eq!(ShapeRecommender::infer_min_count(&s), Some(1));
    }

    #[test]
    fn test_infer_min_count_high_null_rate() {
        let s = stats("p", 100, 0.20, 0.5, 0.0, 0.0, 1.0, 5.0);
        assert_eq!(ShapeRecommender::infer_min_count(&s), None);
    }

    #[test]
    fn test_infer_min_count_exactly_at_threshold() {
        // null_rate == 0.05 → NOT less than 0.05 → None
        let s = stats("p", 100, 0.05, 0.5, 0.0, 0.0, 1.0, 5.0);
        assert_eq!(ShapeRecommender::infer_min_count(&s), None);
    }

    #[test]
    fn test_infer_min_count_zero_null_rate() {
        let s = stats("p", 100, 0.0, 0.5, 0.0, 0.0, 1.0, 5.0);
        assert_eq!(ShapeRecommender::infer_min_count(&s), Some(1));
    }

    // --- infer_max_count ---

    #[test]
    fn test_infer_max_count_high_distinct_rate() {
        let s = stats("p", 100, 0.0, 0.99, 0.0, 0.0, 1.0, 5.0);
        assert_eq!(ShapeRecommender::infer_max_count(&s), Some(1));
    }

    #[test]
    fn test_infer_max_count_low_distinct_rate() {
        let s = stats("p", 100, 0.0, 0.50, 0.0, 0.0, 1.0, 5.0);
        assert_eq!(ShapeRecommender::infer_max_count(&s), None);
    }

    #[test]
    fn test_infer_max_count_exactly_at_threshold() {
        // distinct_rate == 0.95 → NOT > 0.95 → None
        let s = stats("p", 100, 0.0, 0.95, 0.0, 0.0, 1.0, 5.0);
        assert_eq!(ShapeRecommender::infer_max_count(&s), None);
    }

    // --- infer_node_kind ---

    #[test]
    fn test_infer_node_kind_iri_dominant() {
        let s = stats("p", 100, 0.0, 0.5, 0.0, 0.9, 0.1, 5.0);
        assert_eq!(ShapeRecommender::infer_node_kind(&s), "IRI");
    }

    #[test]
    fn test_infer_node_kind_literal_dominant() {
        let s = stats("p", 100, 0.0, 0.5, 0.5, 0.1, 0.9, 5.0);
        assert_eq!(ShapeRecommender::infer_node_kind(&s), "Literal");
    }

    #[test]
    fn test_infer_node_kind_mixed() {
        let s = stats("p", 100, 0.0, 0.5, 0.3, 0.5, 0.5, 5.0);
        assert_eq!(ShapeRecommender::infer_node_kind(&s), "Both");
    }

    #[test]
    fn test_infer_node_kind_exactly_at_iri_threshold() {
        let s = stats("p", 100, 0.0, 0.5, 0.0, 0.8, 0.2, 5.0);
        assert_eq!(ShapeRecommender::infer_node_kind(&s), "IRI");
    }

    // --- infer_datatype ---

    #[test]
    fn test_infer_datatype_high_numeric_long_avg() {
        let s = stats("p", 100, 0.0, 0.5, 0.95, 0.0, 1.0, 6.0);
        assert_eq!(ShapeRecommender::infer_datatype(&s), Some("decimal"));
    }

    #[test]
    fn test_infer_datatype_high_numeric_short_avg() {
        let s = stats("p", 100, 0.0, 0.5, 0.95, 0.0, 1.0, 2.0);
        assert_eq!(ShapeRecommender::infer_datatype(&s), Some("integer"));
    }

    #[test]
    fn test_infer_datatype_low_numeric_rate() {
        let s = stats("p", 100, 0.0, 0.5, 0.3, 0.0, 1.0, 5.0);
        assert_eq!(ShapeRecommender::infer_datatype(&s), None);
    }

    #[test]
    fn test_infer_datatype_exactly_at_threshold() {
        // numeric_rate = 0.9 → Some(…)
        let s = stats("p", 100, 0.0, 0.5, 0.9, 0.0, 1.0, 5.0);
        assert!(ShapeRecommender::infer_datatype(&s).is_some());
    }

    // --- confidence_score ---

    #[test]
    fn test_confidence_score_min_count_high_when_no_nulls() {
        let s = stats("p", 200, 0.0, 0.5, 0.0, 0.0, 1.0, 5.0);
        let c = ShapeRecommender::confidence_score(&s, "sh:minCount");
        assert!(c > 0.7);
    }

    #[test]
    fn test_confidence_score_max_count_high_when_distinct() {
        let s = stats("p", 200, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0);
        let c = ShapeRecommender::confidence_score(&s, "sh:maxCount");
        assert!(c > 0.7);
    }

    #[test]
    fn test_confidence_score_clamped_to_one() {
        let s = stats("p", 10000, 0.0, 1.0, 1.0, 1.0, 0.0, 10.0);
        let c = ShapeRecommender::confidence_score(&s, "sh:maxCount");
        assert!(c <= 1.0);
    }

    #[test]
    fn test_confidence_score_clamped_to_zero() {
        let s = stats("p", 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let c = ShapeRecommender::confidence_score(&s, "sh:minCount");
        assert!(c >= 0.0);
    }

    #[test]
    fn test_confidence_score_unknown_constraint_returns_mid() {
        let s = stats("p", 50, 0.1, 0.5, 0.5, 0.5, 0.5, 5.0);
        let c = ShapeRecommender::confidence_score(&s, "sh:unknown");
        // Should not panic and should be in [0, 1]
        assert!((0.0..=1.0).contains(&c));
    }

    // --- recommend ---

    #[test]
    fn test_recommend_insufficient_samples() {
        let rec = ShapeRecommender::new(RecommenderConfig {
            min_samples: 50,
            confidence_threshold: 0.5,
        });
        let s = stats("p", 5, 0.0, 1.0, 1.0, 0.0, 1.0, 5.0);
        let r = rec.recommend(&s);
        assert!(r.recommended_constraints.is_empty());
        assert_eq!(r.confidence, 0.0);
    }

    #[test]
    fn test_recommend_contains_min_count_when_no_nulls() {
        let rec = ShapeRecommender::new(default_cfg());
        let s = stats(
            "http://schema.org/email",
            100,
            0.0,
            0.99,
            0.0,
            0.0,
            1.0,
            20.0,
        );
        let r = rec.recommend(&s);
        let has_min = r
            .recommended_constraints
            .iter()
            .any(|c| c.contains("minCount"));
        assert!(has_min, "expected sh:minCount constraint");
    }

    #[test]
    fn test_recommend_contains_max_count_when_highly_distinct() {
        let rec = ShapeRecommender::new(default_cfg());
        let s = stats("http://schema.org/id", 100, 0.0, 0.99, 0.0, 0.0, 1.0, 5.0);
        let r = rec.recommend(&s);
        let has_max = r
            .recommended_constraints
            .iter()
            .any(|c| c.contains("maxCount"));
        assert!(has_max, "expected sh:maxCount constraint");
    }

    #[test]
    fn test_recommend_property_name_preserved() {
        let rec = ShapeRecommender::new(default_cfg());
        let s = stats("http://ex.org/age", 100, 0.0, 0.5, 0.0, 0.0, 1.0, 3.0);
        let r = rec.recommend(&s);
        assert_eq!(r.property, "http://ex.org/age");
    }

    #[test]
    fn test_recommend_confidence_in_range() {
        let rec = ShapeRecommender::new(default_cfg());
        let s = stats("p", 100, 0.01, 0.98, 0.95, 0.0, 1.0, 4.0);
        let r = rec.recommend(&s);
        assert!((0.0..=1.0).contains(&r.confidence));
    }

    #[test]
    fn test_recommend_explanation_non_empty_when_constraints_found() {
        let rec = ShapeRecommender::new(default_cfg());
        let s = stats("p", 100, 0.01, 0.98, 0.0, 0.0, 1.0, 5.0);
        let r = rec.recommend(&s);
        if !r.recommended_constraints.is_empty() {
            assert!(!r.explanation.is_empty());
        }
    }

    // --- recommend_all ---

    #[test]
    fn test_recommend_all_returns_one_per_stat() {
        let rec = ShapeRecommender::new(default_cfg());
        let all = vec![
            stats("p1", 100, 0.01, 0.5, 0.0, 0.0, 1.0, 5.0),
            stats("p2", 100, 0.20, 0.5, 0.0, 0.9, 0.1, 5.0),
            stats("p3", 100, 0.01, 0.99, 0.9, 0.0, 1.0, 3.0),
        ];
        let results = rec.recommend_all(&all);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_recommend_all_empty_input() {
        let rec = ShapeRecommender::new(default_cfg());
        let results = rec.recommend_all(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_recommend_all_properties_match() {
        let rec = ShapeRecommender::new(default_cfg());
        let all = vec![
            stats("alpha", 50, 0.0, 0.5, 0.0, 0.0, 1.0, 5.0),
            stats("beta", 50, 0.5, 0.5, 0.0, 0.0, 1.0, 5.0),
        ];
        let results = rec.recommend_all(&all);
        assert_eq!(results[0].property, "alpha");
        assert_eq!(results[1].property, "beta");
    }

    // --- node kind variations ---

    #[test]
    fn test_iri_only_stats_produces_iri_node_kind() {
        let rec = ShapeRecommender::new(default_cfg());
        let s = stats("p", 100, 0.0, 0.5, 0.0, 1.0, 0.0, 30.0);
        let r = rec.recommend(&s);
        let has_iri = r.recommended_constraints.iter().any(|c| c.contains("IRI"));
        assert!(has_iri, "expected sh:nodeKind sh:IRI");
    }

    #[test]
    fn test_literal_only_stats_produces_literal_node_kind() {
        let rec = ShapeRecommender::new(default_cfg());
        let s = stats("p", 100, 0.0, 0.5, 0.0, 0.0, 1.0, 10.0);
        let r = rec.recommend(&s);
        let has_lit = r
            .recommended_constraints
            .iter()
            .any(|c| c.contains("Literal"));
        assert!(has_lit, "expected sh:nodeKind sh:Literal");
    }

    // --- confidence threshold filtering ---

    #[test]
    fn test_high_threshold_filters_all_constraints() {
        let rec = ShapeRecommender::new(RecommenderConfig {
            min_samples: 5,
            confidence_threshold: 0.99,
        });
        let s = stats("p", 10, 0.5, 0.5, 0.5, 0.5, 0.5, 5.0);
        let r = rec.recommend(&s);
        assert!(r.recommended_constraints.is_empty());
    }

    #[test]
    fn test_zero_threshold_allows_all_constraints() {
        let rec = ShapeRecommender::new(RecommenderConfig {
            min_samples: 5,
            confidence_threshold: 0.0,
        });
        // Strong signals for all constraints
        let s = stats("p", 100, 0.0, 0.99, 0.95, 0.9, 0.1, 5.0);
        let r = rec.recommend(&s);
        assert!(!r.recommended_constraints.is_empty());
    }

    // --- edge cases ---

    #[test]
    fn test_all_rates_zero() {
        let rec = ShapeRecommender::new(default_cfg());
        let s = stats("p", 100, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let r = rec.recommend(&s);
        // Should not panic; confidence must be in [0, 1]
        assert!((0.0..=1.0).contains(&r.confidence));
    }

    #[test]
    fn test_all_rates_one() {
        let rec = ShapeRecommender::new(default_cfg());
        let s = stats("p", 500, 0.0, 1.0, 1.0, 1.0, 1.0, 10.0);
        let r = rec.recommend(&s);
        assert!((0.0..=1.0).contains(&r.confidence));
    }

    #[test]
    fn test_large_sample_count_raises_confidence() {
        let rec = ShapeRecommender::new(default_cfg());
        let s_small = stats("p", 10, 0.0, 0.99, 0.0, 0.0, 1.0, 5.0);
        let s_large = stats("p", 10_000, 0.0, 0.99, 0.0, 0.0, 1.0, 5.0);
        let r_small = rec.recommend(&s_small);
        let r_large = rec.recommend(&s_large);
        // Larger sample should give equal or higher confidence
        assert!(r_large.confidence >= r_small.confidence - 1e-6);
    }

    // --- additional coverage ---

    #[test]
    fn test_infer_node_kind_below_iri_threshold() {
        // iri_rate = 0.79 < 0.8 → not IRI
        let s = stats("p", 100, 0.0, 0.5, 0.0, 0.79, 0.21, 5.0);
        let kind = ShapeRecommender::infer_node_kind(&s);
        assert_ne!(kind, "IRI");
    }

    #[test]
    fn test_infer_node_kind_exactly_at_literal_threshold() {
        let s = stats("p", 100, 0.0, 0.5, 0.0, 0.1, 0.8, 5.0);
        assert_eq!(ShapeRecommender::infer_node_kind(&s), "Literal");
    }

    #[test]
    fn test_infer_datatype_exactly_at_numeric_threshold() {
        // numeric_rate == 0.9 → Some(…)
        let s = stats("p", 100, 0.0, 0.5, 0.9, 0.0, 1.0, 3.0);
        assert!(ShapeRecommender::infer_datatype(&s).is_some());
    }

    #[test]
    fn test_infer_datatype_avg_length_boundary() {
        // avg_length == 3.0 → not > 3.0 → integer
        let s = stats("p", 100, 0.0, 0.5, 0.95, 0.0, 1.0, 3.0);
        assert_eq!(ShapeRecommender::infer_datatype(&s), Some("integer"));
    }

    #[test]
    fn test_confidence_score_node_kind_symmetric_rates() {
        // iri_rate == literal_rate → signal = 0 → low confidence
        let s = stats("p", 10, 0.0, 0.5, 0.0, 0.5, 0.5, 5.0);
        let c = ShapeRecommender::confidence_score(&s, "sh:nodeKind");
        assert!(c >= 0.0);
    }

    #[test]
    fn test_recommend_datatype_constraint_integer() {
        let rec = ShapeRecommender::new(RecommenderConfig {
            min_samples: 5,
            confidence_threshold: 0.0,
        });
        let s = stats("p", 100, 0.0, 0.5, 0.95, 0.0, 1.0, 2.0);
        let r = rec.recommend(&s);
        let has_integer = r
            .recommended_constraints
            .iter()
            .any(|c| c.contains("integer"));
        assert!(has_integer, "expected xsd:integer datatype constraint");
    }

    #[test]
    fn test_recommend_datatype_constraint_decimal() {
        let rec = ShapeRecommender::new(RecommenderConfig {
            min_samples: 5,
            confidence_threshold: 0.0,
        });
        let s = stats("p", 100, 0.0, 0.5, 0.95, 0.0, 1.0, 8.0);
        let r = rec.recommend(&s);
        let has_decimal = r
            .recommended_constraints
            .iter()
            .any(|c| c.contains("decimal"));
        assert!(has_decimal, "expected xsd:decimal datatype constraint");
    }

    #[test]
    fn test_recommend_explanation_contains_null_rate() {
        let rec = ShapeRecommender::new(default_cfg());
        let s = stats("p", 100, 0.01, 0.5, 0.0, 0.0, 1.0, 5.0);
        let r = rec.recommend(&s);
        // If minCount was recommended, explanation should mention null_rate
        if r.recommended_constraints
            .iter()
            .any(|c| c.contains("minCount"))
        {
            assert!(r.explanation.contains("null_rate"));
        }
    }

    #[test]
    fn test_recommend_all_single_stat() {
        let rec = ShapeRecommender::new(default_cfg());
        let all = vec![stats("solo", 100, 0.0, 0.5, 0.0, 0.0, 1.0, 5.0)];
        let results = rec.recommend_all(&all);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].property, "solo");
    }
}
