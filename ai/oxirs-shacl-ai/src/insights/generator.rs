//! Insight generation engine

use super::collection::InsightCollection;
use super::config::InsightGenerationConfig;
use super::types::*;
use crate::{
    analytics::{InsightSeverity, TrendDirection},
    Result,
};
use oxirs_shacl::ShapeId;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Main insight generation engine
#[derive(Debug)]
pub struct InsightGenerator {
    config: InsightGenerationConfig,
    /// Historical data for trend analysis
    historical_data: HashMap<String, Vec<(SystemTime, f64)>>,
}

impl InsightGenerator {
    /// Create a new insight generator with default configuration
    pub fn new() -> Self {
        Self::with_config(InsightGenerationConfig::default())
    }

    /// Create a new insight generator with custom configuration
    pub fn with_config(config: InsightGenerationConfig) -> Self {
        Self {
            config,
            historical_data: HashMap::new(),
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &InsightGenerationConfig {
        &self.config
    }

    /// Generate insights from validation data
    pub fn generate_insights(&self, data: &ValidationData) -> Result<InsightCollection> {
        let mut collection = InsightCollection::new();

        // Generate validation insights
        let validation_insights = self.generate_validation_insights(data)?;
        for insight in validation_insights {
            collection.add_validation_insight(insight);
        }

        Ok(collection)
    }

    /// Generate validation insights
    pub fn generate_validation_insights(
        &self,
        data: &ValidationData,
    ) -> Result<Vec<ValidationInsight>> {
        let mut insights = Vec::new();

        // Calculate success rate
        let success_rate = if data.total_validations > 0 {
            data.successful_validations as f64 / data.total_validations as f64
        } else {
            0.0
        };

        // Generate low success rate insight
        if success_rate < 0.6 {
            insights.push(ValidationInsight {
                insight_type: ValidationInsightType::LowSuccessRate,
                title: format!("Low Validation Success Rate Detected: {:.1}%", success_rate * 100.0),
                description: format!(
                    "The validation success rate is {:.1}%, which is below the recommended threshold of 60%. \
                    Out of {} total validations, only {} were successful. This indicates systematic \
                    issues with data quality or shape definitions.",
                    success_rate * 100.0, data.total_validations, data.successful_validations
                ),
                severity: if success_rate < 0.3 {
                    InsightSeverity::Critical
                } else if success_rate < 0.4 {
                    InsightSeverity::High
                } else {
                    InsightSeverity::Medium
                },
                confidence: 0.95,
                affected_shapes: data.shapes_validated.clone(),
                recommendations: vec![
                    "Review and refine shape definitions to match actual data patterns".to_string(),
                    "Implement data quality improvement processes".to_string(),
                    "Consider relaxing overly restrictive constraints".to_string(),
                    format!("Focus on shapes with highest violation rates: {:?}", 
                           self.get_top_violation_shapes(&data.violations_by_shape, 3)),
                ],
                supporting_data: HashMap::from([
                    ("success_rate".to_string(), format!("{:.3}", success_rate)),
                    ("total_validations".to_string(), data.total_validations.to_string()),
                    ("failed_validations".to_string(), data.failed_validations.to_string()),
                ]),
            });
        }

        // Generate performance degradation insight
        if data.avg_validation_time > Duration::from_millis(1000) {
            insights.push(ValidationInsight {
                insight_type: ValidationInsightType::PerformanceDegradation,
                title: "Validation Performance Degradation".to_string(),
                description: format!(
                    "Average validation time is {:.2}ms, which exceeds the recommended threshold. \
                    This may indicate shape complexity issues or inefficient constraint ordering.",
                    data.avg_validation_time.as_millis()
                ),
                severity: if data.avg_validation_time > Duration::from_millis(5000) {
                    InsightSeverity::High
                } else {
                    InsightSeverity::Medium
                },
                confidence: 0.85,
                affected_shapes: data.shapes_validated.clone(),
                recommendations: vec![
                    "Optimize constraint ordering for better performance".to_string(),
                    "Consider simplifying complex shapes".to_string(),
                    "Implement caching for frequently validated patterns".to_string(),
                    "Review target selection efficiency".to_string(),
                ],
                supporting_data: HashMap::from([
                    (
                        "avg_validation_time_ms".to_string(),
                        data.avg_validation_time.as_millis().to_string(),
                    ),
                    (
                        "shapes_count".to_string(),
                        data.shapes_validated.len().to_string(),
                    ),
                ]),
            });
        }

        // Generate violation pattern insight
        if let Some((most_common_error, count)) =
            data.error_types.iter().max_by_key(|(_, &count)| count)
        {
            if *count as f64 / data.total_validations as f64 > 0.1 {
                insights.push(ValidationInsight {
                    insight_type: ValidationInsightType::ViolationPattern,
                    title: format!("Recurring Violation Pattern: {}", most_common_error),
                    description: format!(
                        "The error type '{}' occurs in {:.1}% of validations ({} out of {}). \
                        This suggests a systematic issue that could be addressed with targeted improvements.",
                        most_common_error,
                        (*count as f64 / data.total_validations as f64) * 100.0,
                        count,
                        data.total_validations
                    ),
                    severity: InsightSeverity::Medium,
                    confidence: 0.90,
                    affected_shapes: self.get_shapes_with_error_type(&data.violations_by_shape, most_common_error),
                    recommendations: vec![
                        format!("Investigate root cause of '{}' violations", most_common_error),
                        "Consider data preprocessing to prevent this error type".to_string(),
                        "Review shape definitions related to this constraint".to_string(),
                        "Implement targeted data quality checks".to_string(),
                    ],
                    supporting_data: HashMap::from([
                        ("error_type".to_string(), most_common_error.clone()),
                        ("occurrence_count".to_string(), count.to_string()),
                        ("occurrence_percentage".to_string(), 
                         format!("{:.1}", (*count as f64 / data.total_validations as f64) * 100.0)),
                    ]),
                });
            }
        }

        // Generate trend-based insights if historical data is available
        if data.historical_success_rates.len() >= 3 {
            let trend = self.calculate_trend(&data.historical_success_rates)?;
            if trend == TrendDirection::Decreasing {
                insights.push(ValidationInsight {
                    insight_type: ValidationInsightType::QualityIssue,
                    title: "Decreasing Validation Success Trend".to_string(),
                    description:
                        "Validation success rates are trending downward over time, indicating \
                                 deteriorating data quality or increasing shape complexity."
                            .to_string(),
                    severity: InsightSeverity::High,
                    confidence: 0.80,
                    affected_shapes: data.shapes_validated.clone(),
                    recommendations: vec![
                        "Monitor data sources for quality degradation".to_string(),
                        "Review recent changes to shape definitions".to_string(),
                        "Implement automated data quality monitoring".to_string(),
                        "Consider implementing data validation pipelines".to_string(),
                    ],
                    supporting_data: HashMap::from([
                        ("trend_direction".to_string(), "decreasing".to_string()),
                        (
                            "data_points".to_string(),
                            data.historical_success_rates.len().to_string(),
                        ),
                    ]),
                });
            }
        }

        Ok(insights)
    }

    /// Generate quality insights
    pub fn generate_quality_insights(&self, data: &QualityData) -> Result<Vec<QualityInsight>> {
        let mut insights = Vec::new();

        // Overall quality score insight
        if data.overall_score < 0.7 {
            let trend = if data.quality_trend.len() >= 3 {
                self.calculate_trend(&data.quality_trend)?
            } else {
                TrendDirection::Stable
            };

            insights.push(QualityInsight {
                insight_type: QualityInsightType::TrendAnalysis,
                title: format!("Overall Data Quality Below Threshold: {:.1}%", data.overall_score * 100.0),
                description: format!(
                    "Overall data quality score is {:.1}%, which is below the recommended threshold of 70%. \
                    Key areas of concern include completeness ({:.1}%), consistency ({:.1}%), and accuracy ({:.1}%).",
                    data.overall_score * 100.0,
                    data.completeness_percentage,
                    data.consistency_score * 100.0,
                    data.accuracy_score * 100.0
                ),
                severity: if data.overall_score < 0.5 {
                    InsightSeverity::Critical
                } else {
                    InsightSeverity::High
                },
                confidence: 0.90,
                quality_dimension: "overall".to_string(),
                current_score: data.overall_score,
                trend_direction: trend,
                recommendations: vec![
                    "Focus on improving data completeness through validation rules".to_string(),
                    "Implement consistency checks across data sources".to_string(),
                    "Establish data quality monitoring dashboards".to_string(),
                    "Create automated data quality remediation processes".to_string(),
                ],
                supporting_data: HashMap::from([
                    ("overall_score".to_string(), format!("{:.3}", data.overall_score)),
                    ("completeness_percentage".to_string(), format!("{:.1}", data.completeness_percentage)),
                    ("consistency_score".to_string(), format!("{:.3}", data.consistency_score)),
                    ("accuracy_score".to_string(), format!("{:.3}", data.accuracy_score)),
                ]),
            });
        }

        // Completeness insight
        if data.completeness_percentage < 80.0 {
            insights.push(QualityInsight {
                insight_type: QualityInsightType::Completeness,
                title: format!(
                    "Data Completeness Issue: {:.1}%",
                    data.completeness_percentage
                ),
                description: format!(
                    "Data completeness is {:.1}%, indicating significant missing information. \
                    This can lead to validation failures and reduce data reliability.",
                    data.completeness_percentage
                ),
                severity: if data.completeness_percentage < 60.0 {
                    InsightSeverity::High
                } else {
                    InsightSeverity::Medium
                },
                confidence: 0.85,
                quality_dimension: "completeness".to_string(),
                current_score: data.completeness_percentage / 100.0,
                trend_direction: TrendDirection::Stable,
                recommendations: vec![
                    "Implement mandatory field validation".to_string(),
                    "Add data completeness checks to ingestion pipelines".to_string(),
                    "Identify and address sources of missing data".to_string(),
                ],
                supporting_data: HashMap::from([(
                    "completeness_percentage".to_string(),
                    format!("{:.1}", data.completeness_percentage),
                )]),
            });
        }

        // Anomaly detection insight
        if data.anomalies_count > 10 {
            insights.push(QualityInsight {
                insight_type: QualityInsightType::AnomalyDetection,
                title: format!(
                    "High Number of Data Anomalies Detected: {}",
                    data.anomalies_count
                ),
                description: format!(
                    "Detected {} data anomalies, which is above the expected threshold. \
                    This suggests potential data quality issues or unusual patterns in the data.",
                    data.anomalies_count
                ),
                severity: if data.anomalies_count > 50 {
                    InsightSeverity::High
                } else {
                    InsightSeverity::Medium
                },
                confidence: 0.80,
                quality_dimension: "anomaly_detection".to_string(),
                current_score: 1.0 - (data.anomalies_count as f64 / 1000.0).min(1.0),
                trend_direction: TrendDirection::Stable,
                recommendations: vec![
                    "Investigate root causes of detected anomalies".to_string(),
                    "Implement automated anomaly detection alerts".to_string(),
                    "Review data collection and processing procedures".to_string(),
                    "Consider implementing data outlier filtering".to_string(),
                ],
                supporting_data: HashMap::from([(
                    "anomalies_count".to_string(),
                    data.anomalies_count.to_string(),
                )]),
            });
        }

        Ok(insights)
    }

    /// Generate performance insights
    pub fn generate_performance_insights(
        &self,
        data: &PerformanceData,
    ) -> Result<Vec<PerformanceInsight>> {
        let mut insights = Vec::new();

        // Response time insight
        if data.avg_response_time > 1000.0 {
            let trend = if data.performance_trend.len() >= 3 {
                self.calculate_trend(&data.performance_trend)?
            } else {
                TrendDirection::Stable
            };

            insights.push(PerformanceInsight {
                insight_type: PerformanceInsightType::LatencyIssue,
                title: format!("High Average Response Time: {:.0}ms", data.avg_response_time),
                description: format!(
                    "Average response time is {:.0}ms, which exceeds the recommended threshold of 1000ms. \
                    This may impact user experience and system throughput.",
                    data.avg_response_time
                ),
                severity: if data.avg_response_time > 5000.0 {
                    InsightSeverity::Critical
                } else if data.avg_response_time > 2000.0 {
                    InsightSeverity::High
                } else {
                    InsightSeverity::Medium
                },
                confidence: 0.90,
                metric_name: "avg_response_time".to_string(),
                current_value: data.avg_response_time,
                trend_direction: trend,
                recommendations: vec![
                    "Optimize database queries and indexing".to_string(),
                    "Implement caching for frequently accessed data".to_string(),
                    "Consider horizontal scaling or load balancing".to_string(),
                    "Profile and optimize bottleneck operations".to_string(),
                ],
                supporting_data: HashMap::from([
                    ("avg_response_time_ms".to_string(), format!("{:.0}", data.avg_response_time)),
                    ("throughput".to_string(), format!("{:.2}", data.throughput)),
                ]),
            });
        }

        // Memory usage insight
        if data.memory_usage > 1024.0 {
            insights.push(PerformanceInsight {
                insight_type: PerformanceInsightType::MemoryIssue,
                title: format!("High Memory Usage: {:.0}MB", data.memory_usage),
                description: format!(
                    "Memory usage is {:.0}MB, which may indicate memory leaks or inefficient memory management. \
                    This could lead to performance degradation or system instability.",
                    data.memory_usage
                ),
                severity: if data.memory_usage > 4096.0 {
                    InsightSeverity::High
                } else {
                    InsightSeverity::Medium
                },
                confidence: 0.85,
                metric_name: "memory_usage".to_string(),
                current_value: data.memory_usage,
                trend_direction: TrendDirection::Stable,
                recommendations: vec![
                    "Implement memory profiling and leak detection".to_string(),
                    "Optimize data structures and algorithms".to_string(),
                    "Consider implementing memory pooling".to_string(),
                    "Review garbage collection settings".to_string(),
                ],
                supporting_data: HashMap::from([
                    ("memory_usage_mb".to_string(), format!("{:.0}", data.memory_usage)),
                    ("cpu_usage".to_string(), format!("{:.1}%", data.cpu_usage)),
                ]),
            });
        }

        // Cache performance insight
        if data.cache_hit_rate < 80.0 {
            insights.push(PerformanceInsight {
                insight_type: PerformanceInsightType::CacheOptimization,
                title: format!("Low Cache Hit Rate: {:.1}%", data.cache_hit_rate),
                description: format!(
                    "Cache hit rate is {:.1}%, which is below the recommended threshold of 80%. \
                    This indicates poor cache efficiency and potential for performance optimization.",
                    data.cache_hit_rate
                ),
                severity: if data.cache_hit_rate < 50.0 {
                    InsightSeverity::High
                } else {
                    InsightSeverity::Medium
                },
                confidence: 0.80,
                metric_name: "cache_hit_rate".to_string(),
                current_value: data.cache_hit_rate,
                trend_direction: TrendDirection::Stable,
                recommendations: vec![
                    "Review and optimize cache eviction policies".to_string(),
                    "Increase cache size if memory allows".to_string(),
                    "Implement cache warming strategies".to_string(),
                    "Analyze cache access patterns for optimization".to_string(),
                ],
                supporting_data: HashMap::from([
                    ("cache_hit_rate".to_string(), format!("{:.1}%", data.cache_hit_rate)),
                ]),
            });
        }

        // Bottleneck detection insight
        if !data.bottlenecks.is_empty() {
            insights.push(PerformanceInsight {
                insight_type: PerformanceInsightType::BottleneckDetected,
                title: format!(
                    "Performance Bottlenecks Detected: {}",
                    data.bottlenecks.len()
                ),
                description: format!(
                    "Detected {} performance bottlenecks: {}. \
                    These are limiting system performance and should be addressed.",
                    data.bottlenecks.len(),
                    data.bottlenecks.join(", ")
                ),
                severity: if data.bottlenecks.len() > 3 {
                    InsightSeverity::High
                } else {
                    InsightSeverity::Medium
                },
                confidence: 0.85,
                metric_name: "bottlenecks_count".to_string(),
                current_value: data.bottlenecks.len() as f64,
                trend_direction: TrendDirection::Stable,
                recommendations: vec![
                    "Prioritize bottleneck resolution based on impact".to_string(),
                    "Implement parallel processing where possible".to_string(),
                    "Optimize critical path operations".to_string(),
                    "Consider architectural improvements".to_string(),
                ],
                supporting_data: HashMap::from([
                    (
                        "bottlenecks_count".to_string(),
                        data.bottlenecks.len().to_string(),
                    ),
                    ("bottlenecks".to_string(), data.bottlenecks.join(", ")),
                ]),
            });
        }

        Ok(insights)
    }

    /// Generate shape insights
    pub fn generate_shape_insights(&self, data: &ShapeData) -> Result<Vec<ShapeInsight>> {
        let mut insights = Vec::new();

        // Shape complexity insight
        if data.complexity_score > 0.8 {
            insights.push(ShapeInsight {
                insight_type: ShapeInsightType::OverlyComplex,
                title: format!("Shape Complexity Too High: {:.1}", data.complexity_score),
                description: format!(
                    "Shape '{}' has a complexity score of {:.1}, which may lead to performance issues \
                    and maintenance difficulties. The shape contains {} constraints.",
                    data.shape_id,data.complexity_score, data.constraint_count
                ),
                severity: if data.complexity_score > 0.9 {
                    InsightSeverity::High
                } else {
                    InsightSeverity::Medium
                },
                confidence: 0.85,
                shape_id: data.shape_id.clone(),
                effectiveness_score: data.success_rate,
                complexity_metrics: ShapeComplexityMetrics {
                    overall_complexity: if data.complexity_score > 0.9 {
                        ComplexityLevel::VeryHigh
                    } else if data.complexity_score > 0.7 {
                        ComplexityLevel::High
                    } else if data.complexity_score > 0.5 {
                        ComplexityLevel::Medium
                    } else {
                        ComplexityLevel::Low
                    },
                    constraint_count: data.constraint_count,
                    max_constraint_depth: (data.constraint_count as f64 * 0.3) as usize,
                    target_count: 1,
                    path_complexity: data.complexity_score,
                    has_cycles: false,
                },
                recommendations: vec![
                    "Consider breaking down complex shapes into smaller, composable shapes".to_string(),
                    "Review and eliminate redundant constraints".to_string(),
                    "Optimize constraint ordering for better performance".to_string(),
                    "Use shape inheritance to reduce duplication".to_string(),
                ],
                supporting_data: HashMap::from([
                    ("complexity_score".to_string(), format!("{:.3}", data.complexity_score)),
                    ("constraint_count".to_string(), data.constraint_count.to_string()),
                    ("success_rate".to_string(), format!("{:.3}", data.success_rate)),
                ]),
            });
        }

        // Low success rate insight
        if data.success_rate < 0.6 {
            insights.push(ShapeInsight {
                insight_type: ShapeInsightType::IneffectiveConstraints,
                title: format!("Low Shape Success Rate: {:.1}%", data.success_rate * 100.0),
                description: format!(
                    "Shape '{}' has a success rate of {:.1}%, indicating that constraints may be \
                    too restrictive or misaligned with actual data patterns. Common violations include: {}",
                    data.shape_id,
                    data.success_rate * 100.0,
                    data.common_violations.join(", ")
                ),
                severity: if data.success_rate < 0.3 {
                    InsightSeverity::Critical
                } else {
                    InsightSeverity::High
                },
                confidence: 0.90,
                shape_id: data.shape_id.clone(),
                effectiveness_score: data.success_rate,
                complexity_metrics: ShapeComplexityMetrics {
                    overall_complexity: ComplexityLevel::Medium,
                    constraint_count: data.constraint_count,
                    max_constraint_depth: (data.constraint_count as f64 * 0.3) as usize,
                    target_count: 1,
                    path_complexity: data.complexity_score,
                    has_cycles: false,
                },
                recommendations: vec![
                    "Review constraint definitions against actual data patterns".to_string(),
                    "Consider relaxing overly restrictive constraints".to_string(),
                    format!("Focus on addressing common violations: {}", data.common_violations.join(", ")),
                    "Implement data quality improvement processes".to_string(),
                ],
                supporting_data: HashMap::from([
                    ("success_rate".to_string(), format!("{:.3}", data.success_rate)),
                    ("common_violations".to_string(), data.common_violations.join(", ")),
                    ("usage_frequency".to_string(), data.usage_frequency.to_string()),
                ]),
            });
        }

        // Target effectiveness insight
        if data.target_effectiveness < 0.7 {
            insights.push(ShapeInsight {
                insight_type: ShapeInsightType::PoorTargetSelection,
                title: format!(
                    "Poor Target Selection Effectiveness: {:.1}%",
                    data.target_effectiveness * 100.0
                ),
                description: format!(
                    "Shape '{}' has a target effectiveness score of {:.1}%, suggesting that the \
                    target selection may not be optimal for the intended validation scope.",
                    data.shape_id,
                    data.target_effectiveness * 100.0
                ),
                severity: InsightSeverity::Medium,
                confidence: 0.75,
                shape_id: data.shape_id.clone(),
                effectiveness_score: data.target_effectiveness,
                complexity_metrics: ShapeComplexityMetrics {
                    overall_complexity: ComplexityLevel::Low,
                    constraint_count: data.constraint_count,
                    max_constraint_depth: (data.constraint_count as f64 * 0.3) as usize,
                    target_count: 1,
                    path_complexity: data.complexity_score,
                    has_cycles: false,
                },
                recommendations: vec![
                    "Review target selection criteria and scope".to_string(),
                    "Consider using more specific target selectors".to_string(),
                    "Evaluate whether targets are too broad or too narrow".to_string(),
                    "Test target selection with representative data samples".to_string(),
                ],
                supporting_data: HashMap::from([
                    (
                        "target_effectiveness".to_string(),
                        format!("{:.3}", data.target_effectiveness),
                    ),
                    (
                        "usage_frequency".to_string(),
                        data.usage_frequency.to_string(),
                    ),
                ]),
            });
        }

        // Reusability opportunity insight
        if data.usage_frequency > 100 && !data.related_shapes.is_empty() {
            insights.push(ShapeInsight {
                insight_type: ShapeInsightType::ReusabilityOpportunity,
                title: "Shape Reusability Opportunity Detected".to_string(),
                description: format!(
                    "Shape '{}' is frequently used ({} times) and has {} related shapes. \
                    Consider creating reusable components or inheritance hierarchies.",
                    data.shape_id,
                    data.usage_frequency,
                    data.related_shapes.len()
                ),
                severity: InsightSeverity::Low,
                confidence: 0.70,
                shape_id: data.shape_id.clone(),
                effectiveness_score: data.success_rate,
                complexity_metrics: ShapeComplexityMetrics {
                    overall_complexity: ComplexityLevel::Medium,
                    constraint_count: data.constraint_count,
                    max_constraint_depth: (data.constraint_count as f64 * 0.3) as usize,
                    target_count: 1,
                    path_complexity: data.complexity_score,
                    has_cycles: false,
                },
                recommendations: vec![
                    "Extract common constraints into reusable shape components".to_string(),
                    "Consider implementing shape inheritance patterns".to_string(),
                    "Create a shape library for commonly used patterns".to_string(),
                    "Document shape reusability guidelines".to_string(),
                ],
                supporting_data: HashMap::from([
                    (
                        "usage_frequency".to_string(),
                        data.usage_frequency.to_string(),
                    ),
                    (
                        "related_shapes_count".to_string(),
                        data.related_shapes.len().to_string(),
                    ),
                    (
                        "related_shapes".to_string(),
                        format!("{:?}", data.related_shapes),
                    ),
                ]),
            });
        }

        Ok(insights)
    }

    /// Generate data insights
    pub fn generate_data_insights(&self, data: &DataAnalysisData) -> Result<Vec<DataInsight>> {
        let mut insights = Vec::new();

        // Data completeness insight
        let completeness_score = self.calculate_data_completeness_score(data);
        if completeness_score < 0.8 {
            insights.push(DataInsight {
                insight_type: DataInsightType::Completeness,
                title: format!(
                    "Data Completeness Issue: {:.1}%",
                    completeness_score * 100.0
                ),
                description: format!(
                    "Data completeness is {:.1}%, indicating missing or incomplete information. \
                    This affects {} triples across {} unique subjects.",
                    completeness_score * 100.0,
                    data.total_triples,
                    data.unique_subjects
                ),
                severity: if completeness_score < 0.5 {
                    InsightSeverity::High
                } else {
                    InsightSeverity::Medium
                },
                confidence: 0.85,
                data_characteristics: vec![
                    format!("Total triples: {}", data.total_triples),
                    format!("Unique subjects: {}", data.unique_subjects),
                    format!("Completeness score: {:.3}", completeness_score),
                ],
                data_improvements: vec![
                    "Implement data validation at ingestion time".to_string(),
                    "Add completeness checks to data pipelines".to_string(),
                    "Identify and fill critical data gaps".to_string(),
                ],
                supporting_data: HashMap::from([
                    (
                        "completeness_score".to_string(),
                        format!("{:.3}", completeness_score),
                    ),
                    ("total_triples".to_string(), data.total_triples.to_string()),
                ]),
            });
        }

        // Data freshness insight
        if data.freshness_score < 0.7 {
            insights.push(DataInsight {
                insight_type: DataInsightType::Freshness,
                title: "Data Freshness Concern".to_string(),
                description: format!(
                    "Data freshness score is {:.1}%, indicating stale or outdated information. \
                    This may affect the accuracy of validation results.",
                    data.freshness_score * 100.0
                ),
                severity: InsightSeverity::Medium,
                confidence: 0.75,
                data_characteristics: vec![
                    format!("Freshness score: {:.3}", data.freshness_score),
                    format!("Growth rate: {:.2}%", data.growth_rate * 100.0),
                ],
                data_improvements: vec![
                    "Implement automated data refresh processes".to_string(),
                    "Set up data staleness monitoring".to_string(),
                    "Establish data update schedules".to_string(),
                ],
                supporting_data: HashMap::from([
                    (
                        "freshness_score".to_string(),
                        format!("{:.3}", data.freshness_score),
                    ),
                    (
                        "growth_rate".to_string(),
                        format!("{:.3}", data.growth_rate),
                    ),
                ]),
            });
        }

        Ok(insights)
    }

    // Helper methods for insight generation

    /// Get top violation shapes by violation count
    fn get_top_violation_shapes(
        &self,
        violations_by_shape: &HashMap<ShapeId, Vec<String>>,
        limit: usize,
    ) -> Vec<ShapeId> {
        let mut shape_counts: Vec<(ShapeId, usize)> = violations_by_shape
            .iter()
            .map(|(shape, violations)| (shape.clone(), violations.len()))
            .collect();

        shape_counts.sort_by(|a, b| b.1.cmp(&a.1));
        shape_counts
            .into_iter()
            .take(limit)
            .map(|(shape, _)| shape)
            .collect()
    }

    /// Get shapes that have a specific error type
    fn get_shapes_with_error_type(
        &self,
        violations_by_shape: &HashMap<ShapeId, Vec<String>>,
        error_type: &str,
    ) -> Vec<ShapeId> {
        violations_by_shape
            .iter()
            .filter(|(_, violations)| violations.iter().any(|v| v.contains(error_type)))
            .map(|(shape, _)| shape.clone())
            .collect()
    }

    /// Calculate trend direction from time series data
    fn calculate_trend(&self, data: &[(SystemTime, f64)]) -> Result<TrendDirection> {
        if data.len() < 2 {
            return Ok(TrendDirection::Stable);
        }

        // Simple linear regression slope calculation
        let n = data.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, (_, value)) in data.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += value;
            sum_xy += x * value;
            sum_xx += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

        if slope > 0.01 {
            Ok(TrendDirection::Increasing)
        } else if slope < -0.01 {
            Ok(TrendDirection::Decreasing)
        } else {
            Ok(TrendDirection::Stable)
        }
    }

    /// Calculate data completeness score based on expected vs actual data
    fn calculate_data_completeness_score(&self, data: &DataAnalysisData) -> f64 {
        // Simple heuristic: ratio of unique predicates to expected predicates
        // This could be enhanced with domain-specific knowledge
        let expected_predicates = 50.0; // Baseline expectation
        let actual_ratio = data.unique_predicates as f64 / expected_predicates;
        actual_ratio.min(1.0)
    }

    /// Calculate performance status based on metrics
    fn calculate_performance_status(&self, metrics: &HashMap<String, f64>) -> PerformanceStatus {
        let avg_score = metrics.values().sum::<f64>() / metrics.len() as f64;

        match avg_score {
            score if score > 0.9 => PerformanceStatus::Excellent,
            score if score > 0.8 => PerformanceStatus::Good,
            score if score > 0.6 => PerformanceStatus::Stable,
            score if score > 0.4 => PerformanceStatus::Degrading,
            _ => PerformanceStatus::Critical,
        }
    }

    /// Add historical data point for trend analysis
    pub fn add_historical_data_point(&mut self, metric_name: String, value: f64) {
        let now = SystemTime::now();
        self.historical_data
            .entry(metric_name)
            .or_insert_with(Vec::new)
            .push((now, value));
    }

    /// Get trend for a specific metric
    pub fn get_metric_trend(&self, metric_name: &str) -> Result<TrendDirection> {
        if let Some(data) = self.historical_data.get(metric_name) {
            self.calculate_trend(data)
        } else {
            Ok(TrendDirection::Stable)
        }
    }
}

impl Default for InsightGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// Data types for insight generation
#[derive(Debug, Clone)]
pub struct ValidationData {
    /// Total validation runs performed
    pub total_validations: usize,
    /// Successful validations
    pub successful_validations: usize,
    /// Failed validations
    pub failed_validations: usize,
    /// Validation errors by type
    pub error_types: HashMap<String, usize>,
    /// Average validation time
    pub avg_validation_time: Duration,
    /// Shapes involved in validation
    pub shapes_validated: Vec<ShapeId>,
    /// Constraint violations by shape
    pub violations_by_shape: HashMap<ShapeId, Vec<String>>,
    /// Historical success rates
    pub historical_success_rates: Vec<(SystemTime, f64)>,
}

#[derive(Debug, Clone)]
pub struct QualityData {
    /// Overall quality score (0.0 - 1.0)
    pub overall_score: f64,
    /// Quality scores by dimension
    pub dimension_scores: HashMap<String, f64>,
    /// Data completeness percentage
    pub completeness_percentage: f64,
    /// Data consistency score
    pub consistency_score: f64,
    /// Data accuracy score
    pub accuracy_score: f64,
    /// Detected anomalies count
    pub anomalies_count: usize,
    /// Quality trend over time
    pub quality_trend: Vec<(SystemTime, f64)>,
    /// Issues by category
    pub issues_by_category: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
pub struct PerformanceData {
    /// Average response time in milliseconds
    pub avg_response_time: f64,
    /// Memory usage in MB
    pub memory_usage: f64,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Cache hit rate percentage
    pub cache_hit_rate: f64,
    /// Error rate percentage
    pub error_rate: f64,
    /// Performance trend over time
    pub performance_trend: Vec<(SystemTime, f64)>,
    /// Detected bottlenecks
    pub bottlenecks: Vec<String>,
    /// Resource utilization metrics
    pub resource_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct ShapeData {
    /// Shape identifier
    pub shape_id: ShapeId,
    /// Number of constraints in shape
    pub constraint_count: usize,
    /// Shape complexity score
    pub complexity_score: f64,
    /// Success rate for this shape
    pub success_rate: f64,
    /// Average validation time for this shape
    pub avg_validation_time: Duration,
    /// Common constraint violations
    pub common_violations: Vec<String>,
    /// Target effectiveness score
    pub target_effectiveness: f64,
    /// Shape usage frequency
    pub usage_frequency: usize,
    /// Related shapes
    pub related_shapes: Vec<ShapeId>,
}

#[derive(Debug, Clone)]
pub struct DataAnalysisData {
    /// Total number of triples analyzed
    pub total_triples: usize,
    /// Unique subjects count
    pub unique_subjects: usize,
    /// Unique predicates count
    pub unique_predicates: usize,
    /// Unique objects count
    pub unique_objects: usize,
    /// Data growth rate
    pub growth_rate: f64,
    /// Data freshness score
    pub freshness_score: f64,
    /// Data volume trend
    pub volume_trend: Vec<(SystemTime, usize)>,
    /// Property distribution
    pub property_distribution: HashMap<String, usize>,
    /// Class distribution
    pub class_distribution: HashMap<String, usize>,
    /// Data quality indicators
    pub quality_indicators: HashMap<String, f64>,
}
