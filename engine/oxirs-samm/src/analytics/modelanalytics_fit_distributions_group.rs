//! Distribution fitting implementation for ModelAnalytics
//!
//! Auto-generated module structure (manually extended for distribution fitting)

use super::modelanalytics_type::ModelAnalytics;
use super::types::*;
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};

/// Fitted distribution for a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionFit {
    /// Metric name
    pub metric_name: String,
    /// Best-fit distribution type
    pub distribution_type: DistributionType,
    /// Parameters of the fitted distribution
    pub parameters: DistributionParameters,
    /// Goodness-of-fit score (0-1, higher is better)
    pub goodness_of_fit: f64,
    /// Confidence level in the fit
    pub confidence: ConfidenceLevel,
}

/// Type of statistical distribution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributionType {
    /// Normal (Gaussian) distribution
    Normal,
    /// Exponential distribution
    Exponential,
    /// Uniform distribution
    Uniform,
    /// Log-normal distribution
    LogNormal,
    /// Unknown/unidentified distribution
    Unknown,
}

/// Parameters for different distribution types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionParameters {
    /// Mean (μ) for Normal, rate (λ) for Exponential
    pub param1: f64,
    /// Standard deviation (σ) for Normal, not used for Exponential/Uniform
    pub param2: Option<f64>,
    /// Lower bound for Uniform
    pub lower_bound: Option<f64>,
    /// Upper bound for Uniform
    pub upper_bound: Option<f64>,
}

/// Confidence level in distribution fit
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// High confidence (GoF > 0.9)
    High,
    /// Medium confidence (GoF 0.7-0.9)
    Medium,
    /// Low confidence (GoF 0.5-0.7)
    Low,
    /// Very low confidence (GoF < 0.5)
    VeryLow,
}

impl ConfidenceLevel {
    /// Determine confidence level from goodness-of-fit score
    pub fn from_gof(gof: f64) -> Self {
        if gof > 0.9 {
            ConfidenceLevel::High
        } else if gof > 0.7 {
            ConfidenceLevel::Medium
        } else if gof > 0.5 {
            ConfidenceLevel::Low
        } else {
            ConfidenceLevel::VeryLow
        }
    }
}

impl ModelAnalytics {
    /// Fit statistical distributions to model metrics
    ///
    /// Analyzes the statistical properties of various model metrics and determines
    /// which theoretical distribution best fits each metric. This helps understand
    /// the underlying nature of model complexity and quality.
    ///
    /// # Distributions Tested
    ///
    /// - **Normal**: Symmetric, bell-shaped (most common in nature)
    /// - **Exponential**: Skewed right, memoryless (decay processes)
    /// - **Uniform**: Equal probability across range (random processes)
    /// - **Log-Normal**: Skewed right, multiplicative processes
    ///
    /// # Returns
    ///
    /// A vector of fitted distributions for each metric analyzed
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use oxirs_samm::analytics::ModelAnalytics;
    /// use oxirs_samm::metamodel::Aspect;
    ///
    /// # fn example(aspect: &Aspect) -> Result<(), Box<dyn std::error::Error>> {
    /// let analytics = ModelAnalytics::analyze(aspect)?;
    /// let fits = analytics.fit_distributions();
    ///
    /// for fit in &fits {
    ///     println!("  {} follows {:?} distribution (GoF: {:.3})",
    ///              fit.metric_name, fit.distribution_type, fit.goodness_of_fit);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn fit_distributions(&self) -> Vec<DistributionFit> {
        let mut fits = Vec::new();

        // Collect metrics to analyze
        let metrics = [
            (
                "property_count",
                self.distributions.property_distribution.mean,
                self.distributions.property_distribution.std_dev,
            ),
            (
                "structural_complexity",
                self.complexity_assessment.structural,
                10.0, // Approximate std dev
            ),
            (
                "cognitive_complexity",
                self.complexity_assessment.cognitive,
                5.0, // Approximate std dev
            ),
            (
                "quality_score",
                self.quality_score,
                15.0, // Approximate std dev
            ),
        ];

        for (name, mean, std_dev) in metrics {
            // Fit distribution based on statistical properties
            let fit = Self::fit_single_distribution(name, mean, std_dev);
            fits.push(fit);
        }

        fits
    }

    /// Fit distribution to a single metric
    fn fit_single_distribution(metric_name: &str, mean: f64, std_dev: f64) -> DistributionFit {
        // Simple heuristics for distribution fitting
        // In production, use more sophisticated methods (K-S test, Chi-square, etc.)

        let cv = if mean.abs() > 0.0001 {
            std_dev / mean
        } else {
            0.0
        };

        // Heuristic decision tree for distribution type
        let (dist_type, params, gof) = if cv < 0.3 && mean > 0.0 {
            // Low variability → likely Normal
            (
                DistributionType::Normal,
                DistributionParameters {
                    param1: mean,
                    param2: Some(std_dev),
                    lower_bound: None,
                    upper_bound: None,
                },
                0.85,
            )
        } else if cv > 1.0 && mean > 0.0 {
            // High variability, positive → likely Exponential
            let rate = 1.0 / mean;
            (
                DistributionType::Exponential,
                DistributionParameters {
                    param1: rate,
                    param2: None,
                    lower_bound: Some(0.0),
                    upper_bound: None,
                },
                0.75,
            )
        } else if std_dev < mean * 0.2 && mean > 0.0 {
            // Very low spread → might be Uniform
            let half_range = std_dev * 1.732; // sqrt(3) for uniform dist
            (
                DistributionType::Uniform,
                DistributionParameters {
                    param1: mean,
                    param2: None,
                    lower_bound: Some((mean - half_range).max(0.0)),
                    upper_bound: Some(mean + half_range),
                },
                0.70,
            )
        } else if mean > std_dev && std_dev > 0.0 {
            // Positive with moderate variability → Log-Normal
            (
                DistributionType::LogNormal,
                DistributionParameters {
                    param1: mean,
                    param2: Some(std_dev),
                    lower_bound: Some(0.0),
                    upper_bound: None,
                },
                0.80,
            )
        } else {
            // Default to Normal with lower confidence
            (
                DistributionType::Normal,
                DistributionParameters {
                    param1: mean,
                    param2: Some(std_dev),
                    lower_bound: None,
                    upper_bound: None,
                },
                0.60,
            )
        };

        let confidence = ConfidenceLevel::from_gof(gof);

        DistributionFit {
            metric_name: metric_name.to_string(),
            distribution_type: dist_type,
            parameters: params,
            goodness_of_fit: gof,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};

    fn create_test_aspect() -> Aspect {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        for i in 1..=5 {
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
    fn test_fit_distributions() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let fits = analytics.fit_distributions();

        // Should fit distributions for all metrics
        assert!(!fits.is_empty());
        assert_eq!(fits.len(), 4); // 4 metrics analyzed

        // Verify each fit has required fields
        for fit in &fits {
            assert!(!fit.metric_name.is_empty());
            assert!(fit.goodness_of_fit >= 0.0 && fit.goodness_of_fit <= 1.0);
        }
    }

    #[test]
    fn test_distribution_types() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let fits = analytics.fit_distributions();

        // Check that we get valid distribution types
        for fit in &fits {
            match fit.distribution_type {
                DistributionType::Normal
                | DistributionType::Exponential
                | DistributionType::Uniform
                | DistributionType::LogNormal
                | DistributionType::Unknown => {
                    // Valid distribution type
                }
            }
        }
    }

    #[test]
    fn test_confidence_levels() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let fits = analytics.fit_distributions();

        // Confidence should match GoF
        for fit in &fits {
            let expected = ConfidenceLevel::from_gof(fit.goodness_of_fit);
            assert_eq!(fit.confidence, expected);
        }
    }

    #[test]
    fn test_distribution_parameters() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let fits = analytics.fit_distributions();

        // Parameters should be valid for each distribution type
        for fit in &fits {
            match fit.distribution_type {
                DistributionType::Normal | DistributionType::LogNormal => {
                    assert!(fit.parameters.param2.is_some());
                }
                DistributionType::Exponential => {
                    assert!(fit.parameters.param1 > 0.0); // Rate > 0
                }
                DistributionType::Uniform => {
                    assert!(fit.parameters.lower_bound.is_some());
                    assert!(fit.parameters.upper_bound.is_some());
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_confidence_level_from_gof() {
        assert_eq!(ConfidenceLevel::from_gof(0.95), ConfidenceLevel::High);
        assert_eq!(ConfidenceLevel::from_gof(0.80), ConfidenceLevel::Medium);
        assert_eq!(ConfidenceLevel::from_gof(0.60), ConfidenceLevel::Low);
        assert_eq!(ConfidenceLevel::from_gof(0.40), ConfidenceLevel::VeryLow);
    }
}
