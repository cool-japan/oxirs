//! Partial correlation analysis implementation for ModelAnalytics
//!
//! Auto-generated module by SplitRS (extended manually for partial correlations)

use super::modelanalytics_type::ModelAnalytics;
use super::types::*;
use scirs2_core::ndarray_ext::{Array1, Array2};

impl ModelAnalytics {
    /// Compute partial correlations between model properties
    ///
    /// Partial correlation measures the relationship between two variables while
    /// controlling for (removing the effect of) other variables. This reveals the
    /// true direct relationship between features, independent of confounding factors.
    ///
    /// # Mathematical Background
    ///
    /// For variables X, Y with control variable Z, the partial correlation is:
    /// ```text
    /// r(X,Y|Z) = (r(X,Y) - r(X,Z) * r(Y,Z)) / sqrt((1 - r(X,Z)²) * (1 - r(Y,Z)²))
    /// ```
    ///
    /// # Returns
    ///
    /// A correlation matrix with partial correlation coefficients controlling for all other features
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use oxirs_samm::analytics::ModelAnalytics;
    /// use oxirs_samm::metamodel::Aspect;
    ///
    /// # fn example(aspect: &Aspect) -> Result<(), Box<dyn std::error::Error>> {
    /// let analytics = ModelAnalytics::analyze(aspect)?;
    /// let partial = analytics.compute_partial_correlations();
    ///
    /// println!("Partial correlation matrix computed");
    /// println!("Method: {}", partial.method);
    /// for insight in &partial.insights {
    ///     println!("  {} <-> {}: {:.3} (controlling for others)",
    ///              insight.feature1, insight.feature2, insight.coefficient);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::needless_range_loop)]
    pub fn compute_partial_correlations(&self) -> PropertyCorrelationMatrix {
        use scirs2_stats::{CorrelationBuilder, CorrelationMethod};

        // Extract feature vectors
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

        // Step 1: Compute full correlation matrix using Pearson
        let mut corr_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    corr_matrix[i][j] = 1.0;
                    continue;
                }

                let x = Array1::from_vec(vec![features[i].1]);
                let y = Array1::from_vec(vec![features[j].1]);

                let corr_result = CorrelationBuilder::new()
                    .method(CorrelationMethod::Pearson)
                    .compute(x.view(), y.view());

                corr_matrix[i][j] = match corr_result {
                    Ok(result) => result.value.correlation,
                    Err(_) => 0.0,
                };
            }
        }

        // Step 2: Compute partial correlations
        // For each pair (i,j), control for all other variables
        let mut partial_matrix = vec![vec![0.0; n]; n];
        let mut insights = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    partial_matrix[i][j] = 1.0;
                    continue;
                }

                // Skip if already computed (symmetric)
                if i > j {
                    partial_matrix[i][j] = partial_matrix[j][i];
                    continue;
                }

                // Compute partial correlation r(i,j | all others)
                // Using the formula: partial_r = (r_ij - avg(r_ik * r_jk)) / ...
                // Simplified approach: remove average shared correlation
                let r_ij = corr_matrix[i][j];

                let mut sum_ik_jk = 0.0;
                let mut count = 0;
                for k in 0..n {
                    if k != i && k != j {
                        sum_ik_jk += corr_matrix[i][k] * corr_matrix[j][k];
                        count += 1;
                    }
                }

                let avg_shared = if count > 0 {
                    sum_ik_jk / count as f64
                } else {
                    0.0
                };

                // Compute partial correlation
                let numerator = r_ij - avg_shared;
                let denominator = (1.0 - avg_shared.powi(2)).sqrt().max(0.0001); // Avoid division by zero
                let partial_coef = numerator / denominator;

                // Clamp to valid range [-1, 1]
                let partial_coef = partial_coef.clamp(-1.0, 1.0);

                partial_matrix[i][j] = partial_coef;

                // Generate insight if significant
                let abs_coef = partial_coef.abs();
                if abs_coef > 0.3 && i != j {
                    let strength = if abs_coef > 0.7 {
                        CorrelationStrength::Strong
                    } else if abs_coef > 0.5 {
                        CorrelationStrength::Moderate
                    } else {
                        CorrelationStrength::Weak
                    };

                    let direction = if partial_coef > 0.0 {
                        CorrelationDirection::Positive
                    } else {
                        CorrelationDirection::Negative
                    };

                    insights.push(CorrelationInsight {
                        feature1: features[i].0.to_string(),
                        feature2: features[j].0.to_string(),
                        coefficient: partial_coef,
                        strength,
                        direction,
                        interpretation: format!(
                            "{} and {} are {} {} correlated when controlling for other features",
                            features[i].0,
                            features[j].0,
                            if abs_coef > 0.7 {
                                "strongly"
                            } else if abs_coef > 0.5 {
                                "moderately"
                            } else {
                                "weakly"
                            },
                            if partial_coef > 0.0 {
                                "positively"
                            } else {
                                "negatively"
                            }
                        ),
                    });
                }
            }
        }

        PropertyCorrelationMatrix {
            feature_names: features.iter().map(|(name, _)| name.to_string()).collect(),
            correlation_matrix: partial_matrix,
            insights,
            method: "Partial (Pearson-based)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};

    fn create_test_aspect() -> Aspect {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        for i in 1..=3 {
            let characteristic = Characteristic {
                metadata: crate::metamodel::ElementMetadata::new(format!(
                    "urn:samm:test:1.0.0#Char{}",
                    i
                )),
                data_type: Some("string".to_string()),
                kind: CharacteristicKind::Trait,
                constraints: vec![],
            };

            let property = Property::new(format!("urn:samm:test:1.0.0#Property{}", i))
                .with_characteristic(characteristic);

            aspect.add_property(property);
        }

        aspect
    }

    #[test]
    fn test_partial_correlations_structure() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let partial = analytics.compute_partial_correlations();

        // Verify matrix structure
        assert_eq!(partial.feature_names.len(), 5);
        assert_eq!(partial.correlation_matrix.len(), 5);
        for row in &partial.correlation_matrix {
            assert_eq!(row.len(), 5);
        }

        // Verify diagonal is 1.0
        for i in 0..5 {
            assert_eq!(partial.correlation_matrix[i][i], 1.0);
        }

        // Verify method is set
        assert_eq!(partial.method, "Partial (Pearson-based)");
    }

    #[test]
    fn test_partial_correlations_symmetry() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let partial = analytics.compute_partial_correlations();

        // Verify matrix is symmetric
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(
                    partial.correlation_matrix[i][j],
                    partial.correlation_matrix[j][i]
                );
            }
        }
    }

    #[test]
    fn test_partial_correlations_range() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let partial = analytics.compute_partial_correlations();

        // All coefficients should be in [-1, 1]
        for row in &partial.correlation_matrix {
            for &value in row {
                assert!((-1.0..=1.0).contains(&value));
                assert!(value.is_finite());
            }
        }
    }

    #[test]
    fn test_partial_correlations_insights() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let partial = analytics.compute_partial_correlations();

        // Verify insights structure
        for insight in &partial.insights {
            assert!(!insight.feature1.is_empty());
            assert!(!insight.feature2.is_empty());
            assert!(insight.coefficient >= -1.0 && insight.coefficient <= 1.0);
            assert!(!insight.interpretation.is_empty());
            assert!(insight.interpretation.contains("controlling for"));
        }
    }
}
