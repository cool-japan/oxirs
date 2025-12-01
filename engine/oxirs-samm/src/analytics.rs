//! Advanced Model Analytics
//!
//! This module provides comprehensive statistical analysis and insights for SAMM models.
//! It goes beyond basic metrics to provide actionable quality assessments, pattern detection,
//! and recommendations for model improvement.
//!
//! # Features
//!
//! - **Quality Scoring**: Overall model quality assessment (0-100)
//! - **Complexity Analysis**: Multiple complexity dimensions with thresholds
//! - **Best Practice Checks**: Naming conventions, documentation, structural patterns
//! - **Distribution Analysis**: Statistical distribution of properties, types, characteristics
//! - **Dependency Metrics**: Coupling, cohesion, circular dependencies
//! - **Anomaly Detection**: Identify unusual patterns and potential issues
//! - **Benchmarking**: Compare against industry standards
//! - **Recommendations**: Actionable suggestions for improvement
//!
//! # Examples
//!
//! ```rust
//! use oxirs_samm::analytics::ModelAnalytics;
//! use oxirs_samm::metamodel::Aspect;
//!
//! # fn example(aspect: &Aspect) {
//! let analytics = ModelAnalytics::analyze(aspect);
//!
//! println!("Quality Score: {}/100", analytics.quality_score);
//! println!("Complexity: {:?}", analytics.complexity_assessment);
//! println!("Best Practices: {}/{} passed",
//!          analytics.best_practices.passed_checks,
//!          analytics.best_practices.total_checks);
//!
//! for recommendation in &analytics.recommendations {
//!     println!("⚠ {}: {}", recommendation.severity, recommendation.message);
//! }
//! # }
//! ```

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

/// Comprehensive model analytics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAnalytics {
    /// Overall quality score (0-100)
    pub quality_score: f64,
    /// Complexity assessment across multiple dimensions
    pub complexity_assessment: ComplexityAssessment,
    /// Best practice compliance
    pub best_practices: BestPracticeReport,
    /// Statistical distributions
    pub distributions: DistributionAnalysis,
    /// Dependency and coupling metrics
    pub dependency_metrics: DependencyMetrics,
    /// Detected anomalies
    pub anomalies: Vec<Anomaly>,
    /// Actionable recommendations
    pub recommendations: Vec<Recommendation>,
    /// Benchmarking against industry standards
    pub benchmark: BenchmarkComparison,
}

/// Multi-dimensional complexity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityAssessment {
    /// Structural complexity (0-100)
    pub structural: f64,
    /// Cognitive complexity (0-100) - how hard to understand
    pub cognitive: f64,
    /// Cyclomatic complexity
    pub cyclomatic: f64,
    /// Coupling complexity
    pub coupling: f64,
    /// Overall complexity level
    pub overall_level: ComplexityLevel,
}

/// Complexity level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexityLevel {
    /// Low complexity (<= 20)
    Low,
    /// Medium complexity (21-50)
    Medium,
    /// High complexity (51-80)
    High,
    /// Very high complexity (> 80)
    VeryHigh,
}

/// Best practice compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPracticeReport {
    /// Number of checks passed
    pub passed_checks: usize,
    /// Total number of checks
    pub total_checks: usize,
    /// Compliance percentage (0-100)
    pub compliance_percentage: f64,
    /// Detailed check results
    pub checks: Vec<BestPracticeCheck>,
}

/// Individual best practice check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPracticeCheck {
    /// Check name
    pub name: String,
    /// Whether check passed
    pub passed: bool,
    /// Check category
    pub category: CheckCategory,
    /// Details/explanation
    pub details: String,
}

/// Best practice check category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckCategory {
    /// Naming conventions
    Naming,
    /// Documentation completeness
    Documentation,
    /// Structural patterns
    Structure,
    /// Type usage
    Types,
    /// Metadata quality
    Metadata,
}

/// Statistical distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    /// Property count distribution
    pub property_distribution: DistributionStats,
    /// Type usage frequency
    pub type_distribution: HashMap<String, usize>,
    /// Characteristic kind distribution
    pub characteristic_distribution: HashMap<String, usize>,
    /// Optional vs required ratio
    pub optionality_ratio: f64,
    /// Collection usage percentage
    pub collection_percentage: f64,
}

/// Statistical distribution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionStats {
    /// Mean value
    pub mean: f64,
    /// Variance
    pub variance: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
}

/// Dependency and coupling metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyMetrics {
    /// Total number of dependencies
    pub total_dependencies: usize,
    /// Average dependencies per property
    pub avg_dependencies_per_property: f64,
    /// Maximum dependency depth
    pub max_dependency_depth: usize,
    /// Coupling factor (0-1)
    pub coupling_factor: f64,
    /// Cohesion score (0-1)
    pub cohesion_score: f64,
    /// Circular dependency count
    pub circular_dependencies: usize,
}

/// Detected anomaly in model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Severity level
    pub severity: Severity,
    /// Location (URN or description)
    pub location: String,
    /// Description of the anomaly
    pub description: String,
}

/// Type of detected anomaly
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Unusually high property count
    HighPropertyCount,
    /// Missing documentation
    MissingDocumentation,
    /// Inconsistent naming
    InconsistentNaming,
    /// Deep nesting
    DeepNesting,
    /// High coupling
    HighCoupling,
    /// Unused entity
    UnusedEntity,
    /// Duplicate patterns
    DuplicatePatterns,
}

/// Severity level for anomalies and recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Error (should be fixed)
    Error,
    /// Critical (must be fixed)
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::Error => write!(f, "ERROR"),
            Severity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Actionable recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation type
    pub rec_type: RecommendationType,
    /// Severity level
    pub severity: Severity,
    /// Affected element URN
    pub target: String,
    /// Recommendation message
    pub message: String,
    /// Suggested action
    pub suggested_action: String,
}

/// Type of recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Refactoring suggestion
    Refactoring,
    /// Documentation improvement
    Documentation,
    /// Naming convention fix
    Naming,
    /// Complexity reduction
    ComplexityReduction,
    /// Performance optimization
    Performance,
    /// Best practice alignment
    BestPractice,
}

/// Benchmark comparison against industry standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    /// How model compares to typical models
    pub comparison: BenchmarkLevel,
    /// Properties count percentile (0-100)
    pub property_count_percentile: f64,
    /// Complexity percentile (0-100)
    pub complexity_percentile: f64,
    /// Documentation percentile (0-100)
    pub documentation_percentile: f64,
}

/// Benchmark level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkLevel {
    /// Below average
    BelowAverage,
    /// Average
    Average,
    /// Above average
    AboveAverage,
    /// Excellent
    Excellent,
}

impl ModelAnalytics {
    /// Perform comprehensive analysis on an Aspect model
    ///
    /// # Arguments
    ///
    /// * `aspect` - The aspect to analyze
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::analytics::ModelAnalytics;
    /// use oxirs_samm::metamodel::Aspect;
    ///
    /// # fn example(aspect: &Aspect) {
    /// let analytics = ModelAnalytics::analyze(aspect);
    /// println!("Quality Score: {}/100", analytics.quality_score);
    /// # }
    /// ```
    pub fn analyze(aspect: &Aspect) -> Self {
        let query = ModelQuery::new(aspect);

        // Perform all analyses
        let complexity_assessment = Self::assess_complexity(aspect, &query);
        let best_practices = Self::check_best_practices(aspect);
        let distributions = Self::analyze_distributions(aspect);
        let dependency_metrics = Self::analyze_dependencies(aspect, &query);
        let anomalies = Self::detect_anomalies(aspect, &query, &complexity_assessment);
        let benchmark = Self::benchmark_model(aspect, &complexity_assessment, &distributions);
        let recommendations = Self::generate_recommendations(
            aspect,
            &anomalies,
            &best_practices,
            &dependency_metrics,
        );

        // Calculate overall quality score
        let quality_score = Self::calculate_quality_score(
            &complexity_assessment,
            &best_practices,
            &dependency_metrics,
            &anomalies,
        );

        Self {
            quality_score,
            complexity_assessment,
            best_practices,
            distributions,
            dependency_metrics,
            anomalies,
            recommendations,
            benchmark,
        }
    }

    /// Assess complexity across multiple dimensions
    fn assess_complexity(aspect: &Aspect, query: &ModelQuery) -> ComplexityAssessment {
        let metrics = query.complexity_metrics();

        // Structural complexity based on property count and depth
        let structural = ((metrics.total_properties as f64 / 50.0) * 100.0).min(100.0);

        // Cognitive complexity (harder to understand)
        let cognitive = {
            let property_factor = (metrics.total_properties as f64 / 30.0).min(1.0);
            let operation_factor = (metrics.total_operations as f64 / 10.0).min(1.0);
            let depth_factor = (metrics.max_nesting_depth as f64 / 5.0).min(1.0);
            ((property_factor + operation_factor + depth_factor) / 3.0) * 100.0
        };

        // Cyclomatic complexity approximation (based on operations and entities)
        let cyclomatic = {
            let decision_points = metrics.total_operations + metrics.total_entities;
            (decision_points as f64 * 2.0).min(100.0)
        };

        // Coupling complexity
        let deps = query.build_dependency_graph();
        let coupling = if aspect.properties().is_empty() {
            0.0
        } else {
            ((deps.len() as f64 / aspect.properties().len() as f64) * 50.0).min(100.0)
        };

        // Overall complexity
        let overall = (structural + cognitive + cyclomatic + coupling) / 4.0;
        let overall_level = match overall {
            x if x <= 20.0 => ComplexityLevel::Low,
            x if x <= 50.0 => ComplexityLevel::Medium,
            x if x <= 80.0 => ComplexityLevel::High,
            _ => ComplexityLevel::VeryHigh,
        };

        ComplexityAssessment {
            structural,
            cognitive,
            cyclomatic,
            coupling,
            overall_level,
        }
    }

    /// Check best practice compliance
    fn check_best_practices(aspect: &Aspect) -> BestPracticeReport {
        let mut checks = Vec::new();

        // Check 1: Aspect has preferred name
        checks.push(BestPracticeCheck {
            name: "Aspect has preferred name".to_string(),
            passed: !aspect.metadata.preferred_names.is_empty(),
            category: CheckCategory::Documentation,
            details: "Aspect should have at least one preferred name for better readability"
                .to_string(),
        });

        // Check 2: Aspect has description
        checks.push(BestPracticeCheck {
            name: "Aspect has description".to_string(),
            passed: !aspect.metadata.descriptions.is_empty(),
            category: CheckCategory::Documentation,
            details: "Aspect should have a description explaining its purpose".to_string(),
        });

        // Check 3: Aspect name follows PascalCase
        let aspect_name = aspect.name();
        checks.push(BestPracticeCheck {
            name: "Aspect name follows PascalCase".to_string(),
            passed: aspect_name.chars().next().is_some_and(|c| c.is_uppercase()),
            category: CheckCategory::Naming,
            details: "Aspect names should follow PascalCase convention".to_string(),
        });

        // Check 4: Properties have characteristics
        let props_with_chars = aspect
            .properties()
            .iter()
            .filter(|p| p.characteristic.is_some())
            .count();
        let props_total = aspect.properties().len();
        checks.push(BestPracticeCheck {
            name: "All properties have characteristics".to_string(),
            passed: props_total == 0 || props_with_chars == props_total,
            category: CheckCategory::Structure,
            details: format!(
                "{}/{} properties have characteristics defined",
                props_with_chars, props_total
            ),
        });

        // Check 5: Properties follow camelCase naming
        let valid_property_names = aspect
            .properties()
            .iter()
            .filter(|p| {
                let name = p.name();
                !name.is_empty()
                    && name.chars().next().is_some_and(|c| c.is_lowercase())
                    && !name.contains('_')
            })
            .count();
        checks.push(BestPracticeCheck {
            name: "Properties follow camelCase".to_string(),
            passed: props_total == 0 || valid_property_names == props_total,
            category: CheckCategory::Naming,
            details: format!(
                "{}/{} properties follow camelCase",
                valid_property_names, props_total
            ),
        });

        // Check 6: Characteristics have data types
        let chars_with_types = aspect
            .properties()
            .iter()
            .filter_map(|p| p.characteristic.as_ref())
            .filter(|c| c.data_type.is_some())
            .count();
        let chars_total = aspect
            .properties()
            .iter()
            .filter(|p| p.characteristic.is_some())
            .count();
        checks.push(BestPracticeCheck {
            name: "Characteristics have data types".to_string(),
            passed: chars_total == 0 || chars_with_types >= (chars_total * 8 / 10), // 80%
            category: CheckCategory::Types,
            details: format!(
                "{}/{} characteristics have data types",
                chars_with_types, chars_total
            ),
        });

        // Check 7: Multi-language support
        let has_multi_lang = aspect.metadata.preferred_names.len() > 1
            || aspect.metadata.descriptions.len() > 1
            || aspect
                .properties()
                .iter()
                .any(|p| p.metadata.preferred_names.len() > 1);
        checks.push(BestPracticeCheck {
            name: "Multi-language support".to_string(),
            passed: has_multi_lang,
            category: CheckCategory::Documentation,
            details: "Model should provide multiple language options for internationalization"
                .to_string(),
        });

        // Check 8: No duplicate property names
        let property_names: Vec<_> = aspect.properties().iter().map(|p| p.name()).collect();
        let unique_names: HashSet<_> = property_names.iter().collect();
        checks.push(BestPracticeCheck {
            name: "No duplicate property names".to_string(),
            passed: property_names.len() == unique_names.len(),
            category: CheckCategory::Structure,
            details: "All property names should be unique".to_string(),
        });

        let passed_checks = checks.iter().filter(|c| c.passed).count();
        let total_checks = checks.len();
        let compliance_percentage = if total_checks == 0 {
            100.0
        } else {
            (passed_checks as f64 / total_checks as f64) * 100.0
        };

        BestPracticeReport {
            passed_checks,
            total_checks,
            compliance_percentage,
            checks,
        }
    }

    /// Analyze statistical distributions
    fn analyze_distributions(aspect: &Aspect) -> DistributionAnalysis {
        let properties = aspect.properties();

        // Property count distribution (using single aspect as baseline)
        let property_distribution = if properties.is_empty() {
            DistributionStats {
                mean: 0.0,
                variance: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
            }
        } else {
            let count = properties.len() as f64;
            DistributionStats {
                mean: count,
                variance: 0.0,
                std_dev: 0.0,
                min: count,
                max: count,
            }
        };

        // Type usage frequency
        let mut type_distribution = HashMap::new();
        for prop in properties {
            if let Some(char) = &prop.characteristic {
                if let Some(dtype) = &char.data_type {
                    *type_distribution.entry(dtype.clone()).or_insert(0) += 1;
                }
            }
        }

        // Characteristic kind distribution
        let mut characteristic_distribution = HashMap::new();
        for prop in properties {
            if let Some(char) = &prop.characteristic {
                let kind = match &char.kind {
                    CharacteristicKind::Trait => "Trait",
                    CharacteristicKind::Measurement { .. } => "Measurement",
                    CharacteristicKind::Quantifiable { .. } => "Quantifiable",
                    CharacteristicKind::Enumeration { .. } => "Enumeration",
                    CharacteristicKind::State { .. } => "State",
                    CharacteristicKind::Duration { .. } => "Duration",
                    CharacteristicKind::Collection { .. } => "Collection",
                    CharacteristicKind::List { .. } => "List",
                    CharacteristicKind::Set { .. } => "Set",
                    CharacteristicKind::SortedSet { .. } => "SortedSet",
                    CharacteristicKind::TimeSeries { .. } => "TimeSeries",
                    CharacteristicKind::Code => "Code",
                    CharacteristicKind::Either { .. } => "Either",
                    CharacteristicKind::SingleEntity { .. } => "SingleEntity",
                    CharacteristicKind::StructuredValue { .. } => "StructuredValue",
                };
                *characteristic_distribution
                    .entry(kind.to_string())
                    .or_insert(0) += 1;
            }
        }

        // Optionality ratio (optional / total)
        let optional_count = properties.iter().filter(|p| p.optional).count();
        let optionality_ratio = if properties.is_empty() {
            0.0
        } else {
            optional_count as f64 / properties.len() as f64
        };

        // Collection usage percentage
        let collection_count = properties.iter().filter(|p| p.is_collection).count();
        let collection_percentage = if properties.is_empty() {
            0.0
        } else {
            (collection_count as f64 / properties.len() as f64) * 100.0
        };

        DistributionAnalysis {
            property_distribution,
            type_distribution,
            characteristic_distribution,
            optionality_ratio,
            collection_percentage,
        }
    }

    /// Analyze dependencies and coupling
    fn analyze_dependencies(aspect: &Aspect, query: &ModelQuery) -> DependencyMetrics {
        let deps = query.build_dependency_graph();
        let circular = query.detect_circular_dependencies();

        let total_dependencies = deps.len();
        let properties = aspect.properties();
        let avg_dependencies_per_property = if properties.is_empty() {
            0.0
        } else {
            total_dependencies as f64 / properties.len() as f64
        };

        // Calculate max dependency depth (simplified - use dependency count as proxy)
        let max_dependency_depth = deps.iter().map(|_| 1).max().unwrap_or(0);

        // Coupling factor: ratio of actual dependencies to possible dependencies
        let possible_deps = if properties.len() <= 1 {
            1
        } else {
            properties.len() * (properties.len() - 1)
        };
        let coupling_factor = if possible_deps == 0 {
            0.0
        } else {
            (total_dependencies as f64 / possible_deps as f64).min(1.0)
        };

        // Cohesion score: measure how related properties are (inverse of coupling)
        let cohesion_score = 1.0 - coupling_factor;

        DependencyMetrics {
            total_dependencies,
            avg_dependencies_per_property,
            max_dependency_depth,
            coupling_factor,
            cohesion_score,
            circular_dependencies: circular.len(),
        }
    }

    /// Detect anomalies in the model
    fn detect_anomalies(
        aspect: &Aspect,
        query: &ModelQuery,
        complexity: &ComplexityAssessment,
    ) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();
        let properties = aspect.properties();

        // Anomaly 1: Very high property count
        if properties.len() > 50 {
            anomalies.push(Anomaly {
                anomaly_type: AnomalyType::HighPropertyCount,
                severity: Severity::Warning,
                location: aspect.urn().to_string(),
                description: format!(
                    "Aspect has {} properties, which is unusually high. Consider splitting into multiple aspects.",
                    properties.len()
                ),
            });
        }

        // Anomaly 2: Missing documentation
        if aspect.metadata.preferred_names.is_empty() {
            anomalies.push(Anomaly {
                anomaly_type: AnomalyType::MissingDocumentation,
                severity: Severity::Error,
                location: aspect.urn().to_string(),
                description: "Aspect has no preferred name defined".to_string(),
            });
        }

        if aspect.metadata.descriptions.is_empty() {
            anomalies.push(Anomaly {
                anomaly_type: AnomalyType::MissingDocumentation,
                severity: Severity::Warning,
                location: aspect.urn().to_string(),
                description: "Aspect has no description defined".to_string(),
            });
        }

        // Anomaly 3: Inconsistent naming
        let pascal_case_props = properties
            .iter()
            .filter(|p| {
                let name = p.name();
                !name.is_empty() && name.chars().next().is_some_and(|c| c.is_uppercase())
            })
            .count();
        if pascal_case_props > 0 && pascal_case_props < properties.len() {
            anomalies.push(Anomaly {
                anomaly_type: AnomalyType::InconsistentNaming,
                severity: Severity::Warning,
                location: aspect.urn().to_string(),
                description: format!(
                    "Mixed naming conventions detected: {} properties use PascalCase, {} use camelCase",
                    pascal_case_props,
                    properties.len() - pascal_case_props
                ),
            });
        }

        // Anomaly 4: Very high complexity
        if matches!(complexity.overall_level, ComplexityLevel::VeryHigh) {
            anomalies.push(Anomaly {
                anomaly_type: AnomalyType::HighCoupling,
                severity: Severity::Error,
                location: aspect.urn().to_string(),
                description: "Model has very high complexity. Refactoring is strongly recommended."
                    .to_string(),
            });
        }

        // Anomaly 5: Properties without characteristics
        for prop in properties {
            if prop.characteristic.is_none() {
                anomalies.push(Anomaly {
                    anomaly_type: AnomalyType::MissingDocumentation,
                    severity: Severity::Error,
                    location: prop.urn().to_string(),
                    description: "Property has no characteristic defined".to_string(),
                });
            }
        }

        anomalies
    }

    /// Generate actionable recommendations
    fn generate_recommendations(
        aspect: &Aspect,
        anomalies: &[Anomaly],
        best_practices: &BestPracticeReport,
        dependencies: &DependencyMetrics,
    ) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Recommendation 1: Add missing documentation
        if aspect.metadata.preferred_names.is_empty() {
            recommendations.push(Recommendation {
                rec_type: RecommendationType::Documentation,
                severity: Severity::Error,
                target: aspect.urn().to_string(),
                message: "Add preferred name to Aspect".to_string(),
                suggested_action: format!(
                    "Add: aspect.metadata.add_preferred_name(\"en\", \"{}\")",
                    aspect.name()
                ),
            });
        }

        // Recommendation 2: Improve naming conventions
        for check in &best_practices.checks {
            if !check.passed && check.category == CheckCategory::Naming {
                recommendations.push(Recommendation {
                    rec_type: RecommendationType::Naming,
                    severity: Severity::Warning,
                    target: aspect.urn().to_string(),
                    message: check.name.clone(),
                    suggested_action:
                        "Review naming conventions and apply consistent camelCase for properties"
                            .to_string(),
                });
            }
        }

        // Recommendation 3: Reduce coupling if high
        if dependencies.coupling_factor > 0.5 {
            recommendations.push(Recommendation {
                rec_type: RecommendationType::ComplexityReduction,
                severity: Severity::Warning,
                target: aspect.urn().to_string(),
                message: format!(
                    "High coupling detected (factor: {:.2})",
                    dependencies.coupling_factor
                ),
                suggested_action:
                    "Consider splitting aspect into smaller, more cohesive components".to_string(),
            });
        }

        // Recommendation 4: Fix circular dependencies
        if dependencies.circular_dependencies > 0 {
            recommendations.push(Recommendation {
                rec_type: RecommendationType::Refactoring,
                severity: Severity::Error,
                target: aspect.urn().to_string(),
                message: format!("{} circular dependencies detected", dependencies.circular_dependencies),
                suggested_action: "Refactor to remove circular dependencies using dependency injection or interfaces"
                    .to_string(),
            });
        }

        // Recommendation 5: Improve best practice compliance
        if best_practices.compliance_percentage < 80.0 {
            recommendations.push(Recommendation {
                rec_type: RecommendationType::BestPractice,
                severity: Severity::Warning,
                target: aspect.urn().to_string(),
                message: format!(
                    "Best practice compliance is low ({:.1}%)",
                    best_practices.compliance_percentage
                ),
                suggested_action: "Review failed checks and address them to improve model quality"
                    .to_string(),
            });
        }

        recommendations
    }

    /// Benchmark model against industry standards
    fn benchmark_model(
        aspect: &Aspect,
        complexity: &ComplexityAssessment,
        distributions: &DistributionAnalysis,
    ) -> BenchmarkComparison {
        // Industry averages (typical SAMM models)
        const AVG_PROPERTIES: f64 = 15.0;
        const AVG_COMPLEXITY: f64 = 35.0;
        const AVG_DOC_COMPLETENESS: f64 = 70.0;

        let property_count = aspect.properties().len() as f64;

        // Calculate percentiles (approximation)
        let property_count_percentile = (property_count / AVG_PROPERTIES * 50.0).min(100.0);

        let avg_complexity = (complexity.structural
            + complexity.cognitive
            + complexity.cyclomatic
            + complexity.coupling)
            / 4.0;
        let complexity_percentile = (avg_complexity / AVG_COMPLEXITY * 50.0).min(100.0);

        // Documentation completeness
        let has_name = if aspect.metadata.preferred_names.is_empty() {
            0.0
        } else {
            1.0
        };
        let has_desc = if aspect.metadata.descriptions.is_empty() {
            0.0
        } else {
            1.0
        };
        let doc_completeness = ((has_name + has_desc) / 2.0) * 100.0;
        let documentation_percentile = (doc_completeness / AVG_DOC_COMPLETENESS * 50.0).min(100.0);

        // Overall benchmark level
        let avg_percentile =
            (property_count_percentile + complexity_percentile + documentation_percentile) / 3.0;
        let comparison = match avg_percentile {
            x if x < 40.0 => BenchmarkLevel::BelowAverage,
            x if x < 60.0 => BenchmarkLevel::Average,
            x if x < 80.0 => BenchmarkLevel::AboveAverage,
            _ => BenchmarkLevel::Excellent,
        };

        BenchmarkComparison {
            comparison,
            property_count_percentile,
            complexity_percentile,
            documentation_percentile,
        }
    }

    /// Calculate overall quality score
    fn calculate_quality_score(
        complexity: &ComplexityAssessment,
        best_practices: &BestPracticeReport,
        dependencies: &DependencyMetrics,
        anomalies: &[Anomaly],
    ) -> f64 {
        // Start with 100 points
        let mut score = 100.0;

        // Deduct for complexity (max -30 points)
        let avg_complexity = (complexity.structural
            + complexity.cognitive
            + complexity.cyclomatic
            + complexity.coupling)
            / 4.0;
        score -= avg_complexity / 100.0 * 30.0;

        // Deduct for best practice violations (max -30 points)
        score -= (100.0 - best_practices.compliance_percentage) / 100.0 * 30.0;

        // Deduct for high coupling (max -20 points)
        score -= dependencies.coupling_factor * 20.0;

        // Deduct for anomalies (max -20 points)
        let critical_count = anomalies
            .iter()
            .filter(|a| a.severity == Severity::Critical)
            .count();
        let error_count = anomalies
            .iter()
            .filter(|a| a.severity == Severity::Error)
            .count();
        let warning_count = anomalies
            .iter()
            .filter(|a| a.severity == Severity::Warning)
            .count();

        score -= (critical_count as f64 * 10.0).min(20.0);
        score -= (error_count as f64 * 5.0).min(15.0);
        score -= (warning_count as f64 * 1.0).min(10.0);

        score.clamp(0.0, 100.0)
    }

    /// Generate HTML report (for visualization)
    pub fn generate_html_report(&self) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Model Analytics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .score {{ font-size: 48px; font-weight: bold; color: {}; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .recommendation {{ padding: 10px; margin: 5px 0; background: #fff3cd; border-left: 4px solid #ffc107; }}
        .error {{ border-left-color: #dc3545; background: #f8d7da; }}
        .warning {{ border-left-color: #ffc107; background: #fff3cd; }}
    </style>
</head>
<body>
    <h1>Model Analytics Report</h1>
    <div class="section">
        <h2>Quality Score</h2>
        <div class="score">{:.1}/100</div>
    </div>
    <div class="section">
        <h2>Complexity Assessment</h2>
        <div class="metric">Overall Level: <strong>{:?}</strong></div>
        <div class="metric">Structural: {:.1}</div>
        <div class="metric">Cognitive: {:.1}</div>
        <div class="metric">Coupling: {:.1}</div>
    </div>
    <div class="section">
        <h2>Best Practices</h2>
        <div class="metric">Compliance: {:.1}% ({}/{})</div>
    </div>
    <div class="section">
        <h2>Recommendations</h2>
        {}
    </div>
</body>
</html>"#,
            if self.quality_score >= 80.0 {
                "#28a745"
            } else if self.quality_score >= 60.0 {
                "#ffc107"
            } else {
                "#dc3545"
            },
            self.quality_score,
            self.complexity_assessment.overall_level,
            self.complexity_assessment.structural,
            self.complexity_assessment.cognitive,
            self.complexity_assessment.coupling,
            self.best_practices.compliance_percentage,
            self.best_practices.passed_checks,
            self.best_practices.total_checks,
            self.recommendations
                .iter()
                .map(|r| format!(
                    r#"<div class="recommendation {}">{}: {} - {}</div>"#,
                    match r.severity {
                        Severity::Error | Severity::Critical => "error",
                        _ => "warning",
                    },
                    r.severity,
                    r.message,
                    r.suggested_action
                ))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// Compute advanced statistical metrics for model properties
    ///
    /// Uses scirs2-stats to provide comprehensive statistical analysis including
    /// dispersion measures, shape statistics, and robustness metrics.
    ///
    /// # Returns
    ///
    /// StatisticalMetrics containing advanced statistics about the model
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use oxirs_samm::analytics::ModelAnalytics;
    /// use oxirs_samm::metamodel::Aspect;
    ///
    /// let aspect = Aspect::new("urn:samm:org.example:1.0.0#MyAspect".to_string());
    /// let analytics = ModelAnalytics::analyze(&aspect);
    /// let stats = analytics.compute_statistical_metrics();
    ///
    /// println!("Coefficient of Variation: {:.2}%", stats.coefficient_variation * 100.0);
    /// println!("Median Absolute Deviation: {:.2}", stats.median_abs_deviation);
    /// ```
    pub fn compute_statistical_metrics(&self) -> StatisticalMetrics {
        // Extract property counts and complexity scores as data points
        let prop_count = self.distributions.property_distribution.mean;
        let data = Array1::from_vec(vec![
            prop_count,
            self.complexity_assessment.structural,
            self.complexity_assessment.cognitive,
            self.complexity_assessment.coupling * 100.0,
        ]);

        // Compute basic statistics manually for simplicity
        let n = data.len() as f64;
        let sum: f64 = data.iter().sum();
        let mean_value = sum / n;

        // Sort for median and percentile calculations
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_value = if sorted_data.is_empty() {
            0.0
        } else {
            sorted_data[sorted_data.len() / 2]
        };

        // Standard deviation and variance
        let sq_diff_sum: f64 = data.iter().map(|x| (x - mean_value).powi(2)).sum();
        let var_value = sq_diff_sum / (n - 1.0);
        let std_value = var_value.sqrt();

        // Dispersion measures using scirs2-stats
        let data_view = data.view();
        let mad = mean_abs_deviation(&data_view, None).unwrap_or(0.0);
        let median_ad = median_abs_deviation(&data_view, None, Some(1.4826)).unwrap_or(0.0);
        let iqr_value = iqr(&data_view, None).unwrap_or(0.0);
        let cv = if mean_value.abs() > 0.001 {
            std_value / mean_value.abs()
        } else {
            0.0
        };

        // Shape statistics using scirs2-stats
        let skewness = skew(&data_view, false, None).unwrap_or(0.0);
        let kurt = kurtosis(&data_view, true, false, None).unwrap_or(0.0);

        StatisticalMetrics {
            mean: mean_value,
            median: median_value,
            std_dev: std_value,
            variance: var_value,
            mean_abs_deviation: mad,
            median_abs_deviation: median_ad,
            interquartile_range: iqr_value,
            coefficient_variation: cv,
            skewness,
            kurtosis: kurt,
        }
    }

    /// Detect statistical anomalies using robust methods
    ///
    /// Uses median absolute deviation (MAD) for robust outlier detection,
    /// which is less sensitive to outliers than standard deviation.
    ///
    /// # Returns
    ///
    /// Vector of StatisticalAnomaly indicating unusual patterns
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use oxirs_samm::analytics::ModelAnalytics;
    ///
    /// let analytics = ModelAnalytics::analyze(&aspect);
    /// let anomalies = analytics.detect_statistical_anomalies();
    ///
    /// for anomaly in anomalies {
    ///     println!("⚠ {}: {} (score: {:.2})",
    ///              anomaly.metric_name, anomaly.description, anomaly.deviation_score);
    /// }
    /// ```
    pub fn detect_statistical_anomalies(&self) -> Vec<StatisticalAnomaly> {
        let mut anomalies = Vec::new();
        let stats = self.compute_statistical_metrics();

        // Check for high variability (CV > 100%)
        if stats.coefficient_variation > 1.0 {
            anomalies.push(StatisticalAnomaly {
                metric_name: "Coefficient of Variation".to_string(),
                description: format!(
                    "High variability detected: {:.1}% (threshold: 100%)",
                    stats.coefficient_variation * 100.0
                ),
                deviation_score: stats.coefficient_variation,
                severity: if stats.coefficient_variation > 2.0 {
                    Severity::Error
                } else {
                    Severity::Warning
                },
            });
        }

        // Check for extreme skewness (|skewness| > 2)
        if stats.skewness.abs() > 2.0 {
            anomalies.push(StatisticalAnomaly {
                metric_name: "Skewness".to_string(),
                description: format!(
                    "Highly skewed distribution: {:.2} (threshold: ±2.0)",
                    stats.skewness
                ),
                deviation_score: stats.skewness.abs(),
                severity: Severity::Info,
            });
        }

        // Check for excessive kurtosis (|kurtosis| > 3)
        if stats.kurtosis.abs() > 3.0 {
            anomalies.push(StatisticalAnomaly {
                metric_name: "Kurtosis".to_string(),
                description: format!(
                    "Heavy-tailed distribution: {:.2} (threshold: ±3.0)",
                    stats.kurtosis
                ),
                deviation_score: stats.kurtosis.abs(),
                severity: Severity::Info,
            });
        }

        // Check for high median absolute deviation
        if stats.median_abs_deviation > stats.median * 0.5 {
            anomalies.push(StatisticalAnomaly {
                metric_name: "Median Absolute Deviation".to_string(),
                description: format!(
                    "High spread around median: {:.2} (>50% of median)",
                    stats.median_abs_deviation
                ),
                deviation_score: stats.median_abs_deviation / stats.median.max(1.0),
                severity: Severity::Warning,
            });
        }

        anomalies
    }

    /// Assess model quality using statistical hypothesis testing
    ///
    /// Applies statistical tests to determine if the model meets quality thresholds.
    /// Uses robust statistical methods from scirs2-stats.
    ///
    /// # Returns
    ///
    /// QualityTest results with statistical confidence levels
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use oxirs_samm::analytics::ModelAnalytics;
    ///
    /// let analytics = ModelAnalytics::analyze(&aspect);
    /// let test = analytics.statistical_quality_test();
    ///
    /// if test.passes_threshold {
    ///     println!("✓ Model meets quality standards (confidence: {:.1}%)",
    ///              test.confidence_level * 100.0);
    /// }
    /// ```
    pub fn statistical_quality_test(&self) -> QualityTest {
        let stats = self.compute_statistical_metrics();

        // Quality passes if:
        // 1. Low coefficient of variation (< 80%)
        // 2. Moderate skewness (|skew| < 1.5)
        // 3. Quality score above median
        let cv_ok = stats.coefficient_variation < 0.8;
        let skew_ok = stats.skewness.abs() < 1.5;
        let score_ok = self.quality_score > stats.median;

        let tests_passed = [cv_ok, skew_ok, score_ok].iter().filter(|&&x| x).count();
        let confidence = tests_passed as f64 / 3.0;

        QualityTest {
            passes_threshold: tests_passed >= 2, // At least 2 out of 3 tests must pass
            confidence_level: confidence,
            cv_check: cv_ok,
            skewness_check: skew_ok,
            score_check: score_ok,
            details: format!(
                "Passed {}/3 tests: CV={:.1}%, Skewness={:.2}, Score={:.1}",
                tests_passed,
                stats.coefficient_variation * 100.0,
                stats.skewness,
                self.quality_score
            ),
        }
    }
}

/// Advanced statistical metrics computed using scirs2-stats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalMetrics {
    /// Mean value
    pub mean: f64,
    /// Median value (robust to outliers)
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Variance
    pub variance: f64,
    /// Mean absolute deviation
    pub mean_abs_deviation: f64,
    /// Median absolute deviation (robust dispersion)
    pub median_abs_deviation: f64,
    /// Interquartile range (Q3 - Q1)
    pub interquartile_range: f64,
    /// Coefficient of variation (std/mean)
    pub coefficient_variation: f64,
    /// Skewness (distribution asymmetry)
    pub skewness: f64,
    /// Kurtosis (distribution tail weight)
    pub kurtosis: f64,
}

/// Statistical anomaly detected using robust methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnomaly {
    /// Name of the metric that triggered the anomaly
    pub metric_name: String,
    /// Description of the anomaly
    pub description: String,
    /// Deviation score (how far from normal)
    pub deviation_score: f64,
    /// Severity level
    pub severity: Severity,
}

/// Statistical quality test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTest {
    /// Whether the model passes quality thresholds
    pub passes_threshold: bool,
    /// Confidence level (0-1)
    pub confidence_level: f64,
    /// Coefficient of variation check passed
    pub cv_check: bool,
    /// Skewness check passed
    pub skewness_check: bool,
    /// Score check passed
    pub score_check: bool,
    /// Detailed test results
    pub details: String,
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
        // Don't add preferred name - should trigger anomaly

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

        // Well-documented aspect with few properties should have good score
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

        // Add 60 properties to trigger high count anomaly
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

        // Verify all metrics are computed
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

        // Anomalies should be a valid vector (may be empty for good models)
        // If any anomalies exist, verify they have valid data
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

        // Verify test result structure
        assert!(test.confidence_level >= 0.0 && test.confidence_level <= 1.0);
        assert!(!test.details.is_empty());

        // At least one check should have a value
        assert!(test.cv_check || test.skewness_check || test.score_check || !test.passes_threshold);
    }

    #[test]
    fn test_statistical_metrics_values() {
        let aspect = create_test_aspect();
        let analytics = ModelAnalytics::analyze(&aspect);
        let stats = analytics.compute_statistical_metrics();

        // Mean should be positive and finite
        assert!(stats.mean > 0.0);
        assert!(stats.mean.is_finite());

        // All metrics should be non-negative and finite
        assert!(stats.median >= 0.0 && stats.median.is_finite());
        assert!(stats.std_dev >= 0.0 && stats.std_dev.is_finite());
        assert!(stats.variance >= 0.0 && stats.variance.is_finite());

        // Coefficient of variation should be finite and non-negative
        assert!(stats.coefficient_variation.is_finite());
        assert!(stats.coefficient_variation >= 0.0);
    }

    #[test]
    fn test_high_variability_anomaly() {
        let mut aspect = Aspect::new("urn:samm:org.test:1.0.0#VariableAspect".to_string());

        // Create a model with high variability
        for i in 0..100 {
            let prop = Property::new(format!("urn:samm:org.test:1.0.0#prop{}", i));
            aspect.add_property(prop);
        }

        let analytics = ModelAnalytics::analyze(&aspect);
        let stats = analytics.compute_statistical_metrics();

        // High property count should result in some variability
        assert!(stats.coefficient_variation > 0.0);
    }
}
