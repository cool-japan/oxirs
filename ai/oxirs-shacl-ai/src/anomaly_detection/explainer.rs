//! Explainability module for anomaly detection

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::Anomaly;

/// Explanation report for an anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationReport {
    /// Anomaly ID
    pub anomaly_id: String,
    /// Primary reason
    pub primary_reason: String,
    /// Contributing factors ranked by importance
    pub contributing_factors: Vec<ContributingFactor>,
    /// Rule violations
    pub rule_violations: Vec<RuleViolation>,
    /// Similar normal examples
    pub similar_normal_examples: Vec<NormalExample>,
    /// Counterfactual explanations
    pub counterfactuals: Vec<Counterfactual>,
    /// Feature importance
    pub feature_importance: HashMap<String, f64>,
    /// Confidence in explanation
    pub explanation_confidence: f64,
}

/// Contributing factor to an anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributingFactor {
    /// Factor name
    pub name: String,
    /// Description
    pub description: String,
    /// Importance score (0-1)
    pub importance: f64,
    /// Evidence
    pub evidence: Vec<String>,
}

/// Rule violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleViolation {
    /// Rule name
    pub rule_name: String,
    /// Rule description
    pub rule_description: String,
    /// Severity (0-1)
    pub severity: f64,
    /// Violated constraint
    pub violated_constraint: String,
}

/// Normal example for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalExample {
    /// Example ID
    pub id: String,
    /// Values
    pub values: HashMap<String, f64>,
    /// Similarity to anomaly
    pub similarity: f64,
}

/// Counterfactual explanation (what would make it normal)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counterfactual {
    /// Feature to change
    pub feature: String,
    /// Original value
    pub original_value: f64,
    /// Suggested value
    pub suggested_value: f64,
    /// Probability of becoming normal
    pub probability_normal: f64,
}

/// Anomaly explainer
pub struct AnomalyExplainer {
    min_importance_threshold: f64,
}

impl AnomalyExplainer {
    pub fn new() -> Self {
        Self {
            min_importance_threshold: 0.1,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.min_importance_threshold = threshold;
        self
    }

    /// Explain an anomaly
    pub fn explain(&self, anomaly: &Anomaly) -> ExplanationReport {
        // Extract primary reason from anomaly description
        let primary_reason = self.extract_primary_reason(anomaly);

        // Analyze contributing factors from score factors
        let contributing_factors = self.analyze_factors(&anomaly.score.factors);

        // Identify rule violations from context
        let rule_violations = self.identify_rule_violations(anomaly);

        // Generate counterfactuals
        let counterfactuals = self.generate_counterfactuals(anomaly);

        // Calculate feature importance
        let feature_importance = self.calculate_feature_importance(&anomaly.score.factors);

        ExplanationReport {
            anomaly_id: anomaly.id.clone(),
            primary_reason,
            contributing_factors,
            rule_violations,
            similar_normal_examples: Vec::new(), // Would need access to training data
            counterfactuals,
            feature_importance,
            explanation_confidence: anomaly.score.confidence * 0.9, // Slightly lower than detection confidence
        }
    }

    fn extract_primary_reason(&self, anomaly: &Anomaly) -> String {
        // Extract from description or context
        if let Some(reason) = anomaly.context.get("primary_reason") {
            reason.clone()
        } else {
            anomaly.description.clone()
        }
    }

    fn analyze_factors(&self, factors: &HashMap<String, f64>) -> Vec<ContributingFactor> {
        let mut contributing_factors: Vec<_> = factors
            .iter()
            .filter(|(_, &importance)| importance >= self.min_importance_threshold)
            .map(|(name, &importance)| ContributingFactor {
                name: name.clone(),
                description: self.describe_factor(name),
                importance,
                evidence: vec![format!("Score: {:.3}", importance)],
            })
            .collect();

        // Sort by importance (descending)
        contributing_factors.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        contributing_factors
    }

    fn describe_factor(&self, factor_name: &str) -> String {
        match factor_name {
            "z_score" => "Statistical deviation from mean".to_string(),
            "iqr_deviation" => "Outside interquartile range".to_string(),
            "isolation_score" => "Easily isolated from other points".to_string(),
            "lof_score" => "Local density significantly different".to_string(),
            "modified_z_score" => "Deviation using robust statistics".to_string(),
            "min_distance" => "Distance to nearest normal sample".to_string(),
            "ensemble_vote" => "Agreement among multiple detectors".to_string(),
            _ => format!("Factor: {}", factor_name),
        }
    }

    fn identify_rule_violations(&self, anomaly: &Anomaly) -> Vec<RuleViolation> {
        let mut violations = Vec::new();

        // Check context for violations
        if let Some(threshold) = anomaly.context.get("threshold") {
            if let Some(value) = anomaly.context.get("value") {
                violations.push(RuleViolation {
                    rule_name: "Threshold Violation".to_string(),
                    rule_description: format!("Value must be below threshold {}", threshold),
                    severity: anomaly.score.score,
                    violated_constraint: format!("value <= {}", threshold),
                });
            }
        }

        if let Some(lower) = anomaly.context.get("lower_bound") {
            if let Some(upper) = anomaly.context.get("upper_bound") {
                violations.push(RuleViolation {
                    rule_name: "Range Violation".to_string(),
                    rule_description: format!("Value must be within range [{}, {}]", lower, upper),
                    severity: anomaly.score.score,
                    violated_constraint: format!("{} <= value <= {}", lower, upper),
                });
            }
        }

        violations
    }

    fn generate_counterfactuals(&self, anomaly: &Anomaly) -> Vec<Counterfactual> {
        let mut counterfactuals = Vec::new();

        // Generate counterfactuals based on context
        if let Some(value_str) = anomaly.context.get("value") {
            if let Ok(value) = value_str.parse::<f64>() {
                // Suggest moving towards mean or median
                if let Some(mean_str) = anomaly.context.get("mean") {
                    if let Ok(mean) = mean_str.parse::<f64>() {
                        counterfactuals.push(Counterfactual {
                            feature: "value".to_string(),
                            original_value: value,
                            suggested_value: mean,
                            probability_normal: 0.9,
                        });
                    }
                }

                // Suggest moving within bounds
                if let (Some(lower_str), Some(upper_str)) = (
                    anomaly.context.get("lower_bound"),
                    anomaly.context.get("upper_bound"),
                ) {
                    if let (Ok(lower), Ok(upper)) =
                        (lower_str.parse::<f64>(), upper_str.parse::<f64>())
                    {
                        let suggested = if value < lower {
                            lower + (upper - lower) * 0.1
                        } else {
                            upper - (upper - lower) * 0.1
                        };

                        counterfactuals.push(Counterfactual {
                            feature: "value".to_string(),
                            original_value: value,
                            suggested_value: suggested,
                            probability_normal: 0.95,
                        });
                    }
                }
            }
        }

        counterfactuals
    }

    fn calculate_feature_importance(&self, factors: &HashMap<String, f64>) -> HashMap<String, f64> {
        if factors.is_empty() {
            return HashMap::new();
        }

        let total: f64 = factors.values().sum();
        if total == 0.0 {
            return factors.clone();
        }

        factors
            .iter()
            .map(|(k, &v)| (k.clone(), v / total))
            .collect()
    }
}

impl Default for AnomalyExplainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anomaly_detection::types::{AnomalyScore, AnomalyType};

    #[test]
    fn test_explainer_creation() {
        let explainer = AnomalyExplainer::new();
        assert_eq!(explainer.min_importance_threshold, 0.1);
    }

    #[test]
    fn test_explain_anomaly() {
        let explainer = AnomalyExplainer::new();

        let anomaly_score = AnomalyScore::new(0.8, 0.9, 0.5)
            .with_factor("z_score".to_string(), 3.5)
            .with_factor("iqr_deviation".to_string(), 2.0);

        let anomaly = Anomaly {
            id: "test_anomaly".to_string(),
            anomaly_type: AnomalyType::Outlier,
            score: anomaly_score,
            description: "Test anomaly description".to_string(),
            affected_entities: vec!["entity_1".to_string()],
            timestamp: chrono::Utc::now(),
            context: HashMap::from([
                ("value".to_string(), "100.0".to_string()),
                ("mean".to_string(), "50.0".to_string()),
            ]),
            recommendations: vec![],
        };

        let explanation = explainer.explain(&anomaly);

        assert_eq!(explanation.anomaly_id, "test_anomaly");
        assert!(!explanation.contributing_factors.is_empty());
        assert!(!explanation.feature_importance.is_empty());
        assert!(!explanation.counterfactuals.is_empty());
    }

    #[test]
    fn test_factor_description() {
        let explainer = AnomalyExplainer::new();

        assert_eq!(
            explainer.describe_factor("z_score"),
            "Statistical deviation from mean"
        );
        assert_eq!(
            explainer.describe_factor("iqr_deviation"),
            "Outside interquartile range"
        );
    }

    #[test]
    fn test_feature_importance_calculation() {
        let explainer = AnomalyExplainer::new();

        let factors = HashMap::from([("factor1".to_string(), 2.0), ("factor2".to_string(), 3.0)]);

        let importance = explainer.calculate_feature_importance(&factors);

        assert_eq!(importance.get("factor1"), Some(&0.4));
        assert_eq!(importance.get("factor2"), Some(&0.6));
    }
}
