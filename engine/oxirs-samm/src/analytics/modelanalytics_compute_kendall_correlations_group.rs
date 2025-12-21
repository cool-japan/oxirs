//! # ModelAnalytics - compute_kendall_correlations_group Methods
//!
//! This module contains method implementations for `ModelAnalytics`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::modelanalytics_type::ModelAnalytics;
use crate::analytics::{
    CorrelationDirection, CorrelationInsight, CorrelationStrength, PropertyCorrelationMatrix,
};
use scirs2_core::ndarray_ext::Array1;
use std::collections::{HashMap, HashSet};

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

impl ModelAnalytics {
    /// Compute Kendall tau correlations between model properties
    ///
    /// Kendall tau is a non-parametric measure of ordinal association.
    /// It measures the similarity of orderings when ranked by each variable.
    /// More robust to errors and discrepancies in data than other methods.
    ///
    /// # Returns
    ///
    /// A correlation matrix with Kendall tau correlation coefficients and insights
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_samm::analytics::ModelAnalytics;
    /// use oxirs_samm::metamodel::Aspect;
    ///
    /// # fn example(aspect: &Aspect) -> Result<(), Box<dyn std::error::Error>> {
    /// let analytics = ModelAnalytics::from_aspect(aspect)?;
    /// let kendall = analytics.compute_kendall_correlations();
    ///
    /// println!("Kendall tau correlation matrix computed");
    /// println!("Features analyzed: {:?}", kendall.feature_names);
    /// for insight in &kendall.insights {
    ///     println!("  {} <-> {}: {:.3}", insight.feature1, insight.feature2, insight.coefficient);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn compute_kendall_correlations(&self) -> PropertyCorrelationMatrix {
        use scirs2_stats::{CorrelationBuilder, CorrelationMethod};
        let features = [
            (
                "property_count",
                self.distributions.property_distribution.mean,
            ),
            (
                "structural_complexity",
                self.complexity_assessment.structural,
            ),
            ("cognitive_complexity", self.complexity_assessment.cognitive),
            ("coupling", self.complexity_assessment.coupling * 100.0),
            ("quality_score", self.quality_score),
        ];
        let n = features.len();
        let mut matrix = vec![vec![0.0; n]; n];
        let mut insights = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[i][j] = 1.0;
                    continue;
                }
                if i > j {
                    matrix[i][j] = matrix[j][i];
                    continue;
                }
                let x = Array1::from_vec(vec![features[i].1]);
                let y = Array1::from_vec(vec![features[j].1]);
                let corr_result = CorrelationBuilder::new()
                    .method(CorrelationMethod::Kendall)
                    .compute(x.view(), y.view());
                let coefficient: f64 = match corr_result {
                    Ok(result) => result.value.correlation,
                    Err(_) => 0.0,
                };
                matrix[i][j] = coefficient;
                let abs_coef = coefficient.abs();
                if abs_coef > 0.3 && i != j {
                    let strength = if abs_coef > 0.7 {
                        CorrelationStrength::Strong
                    } else if abs_coef > 0.5 {
                        CorrelationStrength::Moderate
                    } else {
                        CorrelationStrength::Weak
                    };
                    let direction = if coefficient > 0.0 {
                        CorrelationDirection::Positive
                    } else {
                        CorrelationDirection::Negative
                    };
                    insights.push(CorrelationInsight {
                        feature1: features[i].0.to_string(),
                        feature2: features[j].0.to_string(),
                        coefficient,
                        strength,
                        direction,
                        interpretation: generate_correlation_interpretation(
                            features[i].0,
                            features[j].0,
                            coefficient,
                        ),
                    });
                }
            }
        }
        PropertyCorrelationMatrix {
            feature_names: features.iter().map(|(name, _)| name.to_string()).collect(),
            correlation_matrix: matrix,
            insights,
            method: "Kendall".to_string(),
        }
    }
}
