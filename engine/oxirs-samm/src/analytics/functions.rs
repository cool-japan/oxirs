//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::analytics::{
    AnomalyType, BenchmarkLevel, ComplexityLevel, CorrelationDirection, CorrelationStrength,
    ModelAnalytics, Severity,
};
use crate::metamodel::{Aspect, Characteristic, CharacteristicKind, ModelElement, Property};
use crate::query::ModelQuery;
use crate::utils;
use scirs2_core::ndarray_ext::stats::{mean, variance};
use scirs2_core::ndarray_ext::{Array1, ArrayView1};
use scirs2_stats::{
    coef_variation, iqr, kurtosis, mean_abs_deviation, median, median_abs_deviation, skew, std,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
/// Generate human-readable interpretation of correlation
fn generate_correlation_interpretation(feat1: &str, feat2: &str, coef: f64) -> String {
    let direction = if coef > 0.0 { "increases" } else { "decreases" };
    let strength = if coef.abs() > 0.7 {
        "strongly"
    } else if coef.abs() > 0.5 {
        "moderately"
    } else {
        "weakly"
    };
    format!(
        "As {} increases, {} {} {}",
        feat1, feat2, strength, direction
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Characteristic, CharacteristicKind, Property};
    fn create_test_aspect() -> Aspect {
        let mut aspect = Aspect::new("urn:samm:org.test:1.0.0#TestAspect".to_string());
        aspect
            .metadata
            .add_preferred_name("en".to_string(), "Test Aspect".to_string());
        aspect
            .metadata
            .add_description("en".to_string(), "A test aspect for analytics".to_string());
        for i in 0..5 {
            let mut prop = Property::new(format!("urn:samm:org.test:1.0.0#property{}", i));
            let mut char = Characteristic::new(
                format!("urn:samm:org.test:1.0.0#char{}", i),
                CharacteristicKind::Trait,
            );
            char.data_type = Some("xsd:string".to_string());
            prop.characteristic = Some(char);
            prop.optional = i % 2 == 0;
            aspect.add_property(prop);
        }
        aspect
    }
    #[test]
    fn test_basic_analytics() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        assert!(analytics.quality_score > 0.0);
        assert!(analytics.quality_score <= 100.0);
        assert!(analytics.best_practices.total_checks > 0);
    }
    #[test]
    fn test_complexity_assessment() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        assert!(analytics.complexity_assessment.structural >= 0.0);
        assert!(analytics.complexity_assessment.cognitive >= 0.0);
        assert!(matches!(
            analytics.complexity_assessment.overall_level,
            ComplexityLevel::Low | ComplexityLevel::Medium
        ));
    }
    #[test]
    fn test_best_practices() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        assert!(analytics.best_practices.compliance_percentage > 50.0);
        assert!(analytics.best_practices.passed_checks > 0);
    }
    #[test]
    fn test_distributions() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        assert!(analytics.distributions.optionality_ratio > 0.0);
        assert!(analytics.distributions.optionality_ratio <= 1.0);
        assert!(!analytics.distributions.type_distribution.is_empty());
    }
    #[test]
    fn test_dependency_metrics() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        assert!(analytics.dependency_metrics.coupling_factor >= 0.0);
        assert!(analytics.dependency_metrics.coupling_factor <= 1.0);
        assert!(analytics.dependency_metrics.cohesion_score >= 0.0);
        assert!(analytics.dependency_metrics.cohesion_score <= 1.0);
    }
    #[test]
    fn test_anomaly_detection() {
        let aspect = Aspect::new("urn:samm:org.test:1.0.0#TestAspect".to_string());
        let analytics = ModelAnalytics::analyze(&aspect);
        assert!(!analytics.anomalies.is_empty());
        let has_missing_doc_anomaly = analytics
            .anomalies
            .iter()
            .any(|a| matches!(a.anomaly_type, AnomalyType::MissingDocumentation));
        assert!(has_missing_doc_anomaly);
    }
    #[test]
    fn test_recommendations() {
        let aspect = Aspect::new("urn:samm:org.test:1.0.0#TestAspect".to_string());
        let analytics = ModelAnalytics::analyze(&aspect);
        assert!(!analytics.recommendations.is_empty());
    }
    #[test]
    fn test_benchmark_comparison() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        assert!(analytics.benchmark.property_count_percentile >= 0.0);
        assert!(analytics.benchmark.property_count_percentile <= 100.0);
        assert!(matches!(
            analytics.benchmark.comparison,
            BenchmarkLevel::BelowAverage
                | BenchmarkLevel::Average
                | BenchmarkLevel::AboveAverage
                | BenchmarkLevel::Excellent
        ));
    }
    #[test]
    fn test_quality_score_calculation() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        assert!(analytics.quality_score > 60.0);
    }
    #[test]
    fn test_html_report_generation() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let html = analytics.generate_html_report();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Quality Score"));
        assert!(html.contains(&format!("{:.1}", analytics.quality_score)));
    }
    #[test]
    fn test_high_property_count_anomaly() {
        let mut aspect = Aspect::new("urn:samm:org.test:1.0.0#LargeAspect".to_string());
        for i in 0..60 {
            let prop = Property::new(format!("urn:samm:org.test:1.0.0#prop{}", i));
            aspect.add_property(prop);
        }
        let analytics = ModelAnalytics::analyze(&aspect);
        let has_high_count = analytics
            .anomalies
            .iter()
            .any(|a| matches!(a.anomaly_type, AnomalyType::HighPropertyCount));
        assert!(has_high_count);
    }
    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", Severity::Info), "INFO");
        assert_eq!(format!("{}", Severity::Warning), "WARNING");
        assert_eq!(format!("{}", Severity::Error), "ERROR");
        assert_eq!(format!("{}", Severity::Critical), "CRITICAL");
    }
    #[test]
    fn test_complexity_levels() {
        assert!(matches!(ComplexityLevel::Low, ComplexityLevel::Low));
        assert!(matches!(ComplexityLevel::Medium, ComplexityLevel::Medium));
        assert!(matches!(ComplexityLevel::High, ComplexityLevel::High));
        assert!(matches!(
            ComplexityLevel::VeryHigh,
            ComplexityLevel::VeryHigh
        ));
    }
    #[test]
    fn test_statistical_metrics() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let stats = analytics.compute_statistical_metrics();
        assert!(stats.mean >= 0.0);
        assert!(stats.median >= 0.0);
        assert!(stats.std_dev >= 0.0);
        assert!(stats.variance >= 0.0);
        assert!(stats.mean_abs_deviation >= 0.0);
        assert!(stats.median_abs_deviation >= 0.0);
        assert!(stats.interquartile_range >= 0.0);
        assert!(stats.coefficient_variation >= 0.0);
    }
    #[test]
    fn test_statistical_anomaly_detection() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let anomalies = analytics.detect_statistical_anomalies();
        for anomaly in &anomalies {
            assert!(!anomaly.metric_name.is_empty());
            assert!(!anomaly.description.is_empty());
            assert!(anomaly.deviation_score >= 0.0);
        }
    }
    #[test]
    fn test_quality_test() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let test = analytics.statistical_quality_test();
        assert!(test.confidence_level >= 0.0 && test.confidence_level <= 1.0);
        assert!(!test.details.is_empty());
        assert!(test.cv_check || test.skewness_check || test.score_check || !test.passes_threshold);
    }
    #[test]
    fn test_statistical_metrics_values() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let stats = analytics.compute_statistical_metrics();
        assert!(stats.mean > 0.0);
        assert!(stats.mean.is_finite());
        assert!(stats.median >= 0.0 && stats.median.is_finite());
        assert!(stats.std_dev >= 0.0 && stats.std_dev.is_finite());
        assert!(stats.variance >= 0.0 && stats.variance.is_finite());
        assert!(stats.coefficient_variation.is_finite());
        assert!(stats.coefficient_variation >= 0.0);
    }
    #[test]
    fn test_high_variability_anomaly() {
        let mut aspect = Aspect::new("urn:samm:org.test:1.0.0#VariableAspect".to_string());
        for i in 0..100 {
            let prop = Property::new(format!("urn:samm:org.test:1.0.0#prop{}", i));
            aspect.add_property(prop);
        }
        let analytics = ModelAnalytics::analyze(&aspect);
        let stats = analytics.compute_statistical_metrics();
        assert!(stats.coefficient_variation > 0.0);
    }
    #[test]
    fn test_property_correlations() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let correlations = analytics.compute_property_correlations();
        assert_eq!(correlations.feature_names.len(), 5);
        assert_eq!(correlations.correlation_matrix.len(), 5);
        for row in &correlations.correlation_matrix {
            assert_eq!(row.len(), 5);
        }
        for i in 0..5 {
            assert_eq!(correlations.correlation_matrix[i][i], 1.0);
        }
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(
                    correlations.correlation_matrix[i][j],
                    correlations.correlation_matrix[j][i]
                );
            }
        }
        assert_eq!(correlations.method, "Pearson");
    }
    #[test]
    fn test_correlation_insights() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let correlations = analytics.compute_property_correlations();
        for insight in &correlations.insights {
            assert!(!insight.feature1.is_empty());
            assert!(!insight.feature2.is_empty());
            assert!(insight.coefficient >= -1.0 && insight.coefficient <= 1.0);
            assert!(!insight.interpretation.is_empty());
        }
    }
    #[test]
    fn test_correlation_strength_classification() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let correlations = analytics.compute_property_correlations();
        for insight in &correlations.insights {
            let abs_coef = insight.coefficient.abs();
            match insight.strength {
                CorrelationStrength::Weak => {
                    assert!(abs_coef > 0.3 && abs_coef <= 0.5);
                }
                CorrelationStrength::Moderate => {
                    assert!(abs_coef > 0.5 && abs_coef <= 0.7);
                }
                CorrelationStrength::Strong => {
                    assert!(abs_coef > 0.7);
                }
            }
        }
    }
    #[test]
    fn test_correlation_direction() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let correlations = analytics.compute_property_correlations();
        for insight in &correlations.insights {
            if insight.coefficient > 0.0 {
                assert_eq!(insight.direction, CorrelationDirection::Positive);
            } else {
                assert_eq!(insight.direction, CorrelationDirection::Negative);
            }
        }
    }
    #[test]
    fn test_correlation_matrix_values() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let correlations = analytics.compute_property_correlations();
        for row in &correlations.correlation_matrix {
            for &value in row {
                assert!((-1.0..=1.0).contains(&value));
                assert!(value.is_finite());
            }
        }
    }
    #[test]
    fn test_spearman_correlations() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let spearman = analytics.compute_spearman_correlations();
        assert_eq!(spearman.feature_names.len(), 5);
        assert_eq!(spearman.correlation_matrix.len(), 5);
        for row in &spearman.correlation_matrix {
            assert_eq!(row.len(), 5);
        }
        for i in 0..5 {
            assert_eq!(spearman.correlation_matrix[i][i], 1.0);
        }
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(
                    spearman.correlation_matrix[i][j],
                    spearman.correlation_matrix[j][i]
                );
            }
        }
        assert_eq!(spearman.method, "Spearman");
        for row in &spearman.correlation_matrix {
            for &value in row {
                assert!((-1.0..=1.0).contains(&value));
                assert!(value.is_finite());
            }
        }
    }
    #[test]
    fn test_kendall_correlations() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let kendall = analytics.compute_kendall_correlations();
        assert_eq!(kendall.feature_names.len(), 5);
        assert_eq!(kendall.correlation_matrix.len(), 5);
        for row in &kendall.correlation_matrix {
            assert_eq!(row.len(), 5);
        }
        for i in 0..5 {
            assert_eq!(kendall.correlation_matrix[i][i], 1.0);
        }
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(
                    kendall.correlation_matrix[i][j],
                    kendall.correlation_matrix[j][i]
                );
            }
        }
        assert_eq!(kendall.method, "Kendall");
        for row in &kendall.correlation_matrix {
            for &value in row {
                assert!((-1.0..=1.0).contains(&value));
                assert!(value.is_finite());
            }
        }
    }
    #[test]
    fn test_correlation_methods_comparison() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let pearson = analytics.compute_property_correlations();
        let spearman = analytics.compute_spearman_correlations();
        let kendall = analytics.compute_kendall_correlations();
        assert_eq!(pearson.feature_names, spearman.feature_names);
        assert_eq!(pearson.feature_names, kendall.feature_names);
        assert_eq!(
            pearson.correlation_matrix.len(),
            spearman.correlation_matrix.len()
        );
        assert_eq!(
            pearson.correlation_matrix.len(),
            kendall.correlation_matrix.len()
        );
        assert_eq!(pearson.method, "Pearson");
        assert_eq!(spearman.method, "Spearman");
        assert_eq!(kendall.method, "Kendall");
        for i in 0..5 {
            assert_eq!(pearson.correlation_matrix[i][i], 1.0);
            assert_eq!(spearman.correlation_matrix[i][i], 1.0);
            assert_eq!(kendall.correlation_matrix[i][i], 1.0);
        }
    }
    #[test]
    fn test_spearman_insights() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let spearman = analytics.compute_spearman_correlations();
        for insight in &spearman.insights {
            assert!(!insight.feature1.is_empty());
            assert!(!insight.feature2.is_empty());
            assert!(insight.coefficient >= -1.0 && insight.coefficient <= 1.0);
            assert!(!insight.interpretation.is_empty());
            let abs_coef = insight.coefficient.abs();
            match insight.strength {
                CorrelationStrength::Weak => {
                    assert!(abs_coef > 0.3 && abs_coef <= 0.5);
                }
                CorrelationStrength::Moderate => {
                    assert!(abs_coef > 0.5 && abs_coef <= 0.7);
                }
                CorrelationStrength::Strong => {
                    assert!(abs_coef > 0.7);
                }
            }
        }
    }
    #[test]
    fn test_kendall_insights() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let kendall = analytics.compute_kendall_correlations();
        for insight in &kendall.insights {
            assert!(!insight.feature1.is_empty());
            assert!(!insight.feature2.is_empty());
            assert!(insight.coefficient >= -1.0 && insight.coefficient <= 1.0);
            assert!(!insight.interpretation.is_empty());
            if insight.coefficient > 0.0 {
                assert_eq!(insight.direction, CorrelationDirection::Positive);
            } else {
                assert_eq!(insight.direction, CorrelationDirection::Negative);
            }
        }
    }
}
