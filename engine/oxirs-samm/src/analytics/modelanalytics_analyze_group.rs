//! # ModelAnalytics - analyze_group Methods
//!
//! This module contains method implementations for `ModelAnalytics`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::modelanalytics_type::ModelAnalytics;
use crate::analytics::{
    Anomaly, AnomalyType, BenchmarkComparison, BenchmarkLevel, BestPracticeCheck,
    BestPracticeReport, CheckCategory, ComplexityAssessment, ComplexityLevel, DependencyMetrics,
    DistributionAnalysis, DistributionStats, DistributionType, Recommendation, RecommendationType,
    Severity,
};
use crate::metamodel::{Aspect, CharacteristicKind, ModelElement};
use crate::query::ModelQuery;
use std::collections::{HashMap, HashSet};

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
        let structural = ((metrics.total_properties as f64 / 50.0) * 100.0).min(100.0);
        let cognitive = {
            let property_factor = (metrics.total_properties as f64 / 30.0).min(1.0);
            let operation_factor = (metrics.total_operations as f64 / 10.0).min(1.0);
            let depth_factor = (metrics.max_nesting_depth as f64 / 5.0).min(1.0);
            ((property_factor + operation_factor + depth_factor) / 3.0) * 100.0
        };
        let cyclomatic = {
            let decision_points = metrics.total_operations + metrics.total_entities;
            (decision_points as f64 * 2.0).min(100.0)
        };
        let deps = query.build_dependency_graph();
        let coupling = if aspect.properties().is_empty() {
            0.0
        } else {
            ((deps.len() as f64 / aspect.properties().len() as f64) * 50.0).min(100.0)
        };
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
        checks.push(BestPracticeCheck {
            name: "Aspect has preferred name".to_string(),
            passed: !aspect.metadata.preferred_names.is_empty(),
            category: CheckCategory::Documentation,
            details: "Aspect should have at least one preferred name for better readability"
                .to_string(),
        });
        checks.push(BestPracticeCheck {
            name: "Aspect has description".to_string(),
            passed: !aspect.metadata.descriptions.is_empty(),
            category: CheckCategory::Documentation,
            details: "Aspect should have a description explaining its purpose".to_string(),
        });
        let aspect_name = aspect.name();
        checks.push(BestPracticeCheck {
            name: "Aspect name follows PascalCase".to_string(),
            passed: aspect_name.chars().next().is_some_and(|c| c.is_uppercase()),
            category: CheckCategory::Naming,
            details: "Aspect names should follow PascalCase convention".to_string(),
        });
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
            passed: chars_total == 0 || chars_with_types >= (chars_total * 8 / 10),
            category: CheckCategory::Types,
            details: format!(
                "{}/{} characteristics have data types",
                chars_with_types, chars_total
            ),
        });
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
        let mut type_distribution = HashMap::new();
        for prop in properties {
            if let Some(char) = &prop.characteristic {
                if let Some(dtype) = &char.data_type {
                    *type_distribution.entry(dtype.clone()).or_insert(0) += 1;
                }
            }
        }
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
        let optional_count = properties.iter().filter(|p| p.optional).count();
        let optionality_ratio = if properties.is_empty() {
            0.0
        } else {
            optional_count as f64 / properties.len() as f64
        };
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
        let max_dependency_depth = deps.iter().map(|_| 1).max().unwrap_or(0);
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
        if properties.len() > 50 {
            anomalies
                .push(Anomaly {
                    anomaly_type: AnomalyType::HighPropertyCount,
                    severity: Severity::Warning,
                    location: aspect.urn().to_string(),
                    description: format!(
                        "Aspect has {} properties, which is unusually high. Consider splitting into multiple aspects.",
                        properties.len()
                    ),
                });
        }
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
        let pascal_case_props = properties
            .iter()
            .filter(|p| {
                let name = p.name();
                !name.is_empty() && name.chars().next().is_some_and(|c| c.is_uppercase())
            })
            .count();
        if pascal_case_props > 0 && pascal_case_props < properties.len() {
            anomalies
                .push(Anomaly {
                    anomaly_type: AnomalyType::InconsistentNaming,
                    severity: Severity::Warning,
                    location: aspect.urn().to_string(),
                    description: format!(
                        "Mixed naming conventions detected: {} properties use PascalCase, {} use camelCase",
                        pascal_case_props, properties.len() - pascal_case_props
                    ),
                });
        }
        if matches!(complexity.overall_level, ComplexityLevel::VeryHigh) {
            anomalies.push(Anomaly {
                anomaly_type: AnomalyType::HighCoupling,
                severity: Severity::Error,
                location: aspect.urn().to_string(),
                description: "Model has very high complexity. Refactoring is strongly recommended."
                    .to_string(),
            });
        }
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
        if dependencies.circular_dependencies > 0 {
            recommendations
                .push(Recommendation {
                    rec_type: RecommendationType::Refactoring,
                    severity: Severity::Error,
                    target: aspect.urn().to_string(),
                    message: format!(
                        "{} circular dependencies detected", dependencies
                        .circular_dependencies
                    ),
                    suggested_action: "Refactor to remove circular dependencies using dependency injection or interfaces"
                        .to_string(),
                });
        }
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
        const AVG_PROPERTIES: f64 = 15.0;
        const AVG_COMPLEXITY: f64 = 35.0;
        const AVG_DOC_COMPLETENESS: f64 = 70.0;
        let property_count = aspect.properties().len() as f64;
        let property_count_percentile = (property_count / AVG_PROPERTIES * 50.0).min(100.0);
        let avg_complexity = (complexity.structural
            + complexity.cognitive
            + complexity.cyclomatic
            + complexity.coupling)
            / 4.0;
        let complexity_percentile = (avg_complexity / AVG_COMPLEXITY * 50.0).min(100.0);
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
        let mut score: f64 = 100.0;
        let avg_complexity = (complexity.structural
            + complexity.cognitive
            + complexity.cyclomatic
            + complexity.coupling)
            / 4.0;
        score -= avg_complexity / 100.0 * 30.0;
        score -= (100.0 - best_practices.compliance_percentage) / 100.0 * 30.0;
        score -= dependencies.coupling_factor * 20.0;
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
}
