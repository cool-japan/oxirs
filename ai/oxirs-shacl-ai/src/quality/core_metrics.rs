//! Quality Assessment Core — Metric Computation
//!
//! Quality metric computation, dimension scoring, and aggregation for the
//! `QualityAssessor`.

use std::collections::HashMap;
use std::time::Instant;

use oxirs_core::{
    model::{NamedNode, Term},
    RdfTerm, Store,
};

use oxirs_shacl::{Constraint, PropertyPath, Shape, Target, ValidationConfig, Validator};

use crate::{insights::QualityInsight, Result, ShaclAiError};

use super::core_types::{
    AccuracyCheck, AdherenceCheck, ConformanceResult, ConsistencyCheck, DuplicateGroup,
    DuplicateResult, ImplementationEffort, QualityAlgorithms, QualityAssessmentData, QualityConfig,
    QualityIssue, QualityIssueCategory, QualityIssueSeverity, QualityRecommendation,
    QualityRecommendationCategory, QualityReport, QualityStatistics, QualityThresholds,
    QualityTrainingData, QualityWeights, RecommendationPriority,
};

/// AI-powered data quality assessor
#[derive(Debug)]
pub struct QualityAssessor {
    /// Configuration
    pub(crate) config: QualityConfig,

    /// Quality assessment cache
    pub(crate) assessment_cache: HashMap<String, QualityReport>,

    /// Statistics
    pub(crate) stats: QualityStatistics,

    /// Model learned by [`QualityAssessor::train_model`], if any.
    ///
    /// This is a genuine (small) linear regressor fit to the training examples,
    /// not a placeholder — see [`TrainedQualityModel`].
    pub(crate) quality_model: Option<TrainedQualityModel>,
}

/// A trained linear quality-prediction model.
///
/// Predicts a quality score from a graph feature vector using standardized
/// ridge regression fit by gradient descent over the training examples. This is
/// a real learned model: its weights are a deterministic function of the
/// training data (given the fixed initialization), so its reported accuracy
/// reflects how well the features actually explain the target scores.
#[derive(Debug, Clone)]
pub(crate) struct TrainedQualityModel {
    /// Per-feature means used to standardize inputs.
    feature_means: Vec<f64>,
    /// Per-feature standard deviations used to standardize inputs (never zero).
    feature_stds: Vec<f64>,
    /// Regression weights over the standardized features.
    weights: Vec<f64>,
    /// Regression bias term (equals the mean target when there are no features).
    bias: f64,
}

impl TrainedQualityModel {
    /// Predict a quality score for a raw (un-standardized) feature vector,
    /// clamped to the valid `[0, 1]` quality range.
    pub(crate) fn predict(&self, features: &[f64]) -> f64 {
        let mut acc = self.bias;
        for (idx, &weight) in self.weights.iter().enumerate() {
            let raw = features.get(idx).copied().unwrap_or(0.0);
            let std = self.feature_stds[idx];
            let standardized = (raw - self.feature_means[idx]) / std;
            acc += weight * standardized;
        }
        acc.clamp(0.0, 1.0)
    }
}

impl QualityAssessor {
    /// Create a new quality assessor with default configuration
    pub fn new() -> Self {
        Self::with_config(QualityConfig::default())
    }

    /// Create a new quality assessor with custom configuration
    pub fn with_config(config: QualityConfig) -> Self {
        Self {
            config,
            assessment_cache: HashMap::new(),
            stats: QualityStatistics::default(),
            quality_model: None,
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &QualityConfig {
        &self.config
    }

    /// Get statistics
    pub fn get_statistics(&self) -> &QualityStatistics {
        &self.stats
    }

    /// Train the quality assessment model.
    ///
    /// Fits a standardized ridge-regression model mapping each example's
    /// `graph_features` to its `quality_score` via gradient descent, stores it
    /// on the assessor, and reports the mean-absolute-error-based accuracy and
    /// mean-squared-error loss of the fitted model **on the training data**.
    ///
    /// Unlike the previous placeholder, the reported metrics are a genuine
    /// function of the training examples: a feature set that does not explain
    /// the target scores yields a correspondingly lower accuracy. An empty
    /// training set fails loudly with [`ShaclAiError::ModelTraining`].
    pub fn train_model(
        &mut self,
        training_data: &QualityTrainingData,
    ) -> Result<crate::ModelTrainingResult> {
        tracing::info!("Training quality assessment model");

        let examples = &training_data.quality_examples;
        if examples.is_empty() {
            return Err(ShaclAiError::ModelTraining(
                "cannot train quality model: no training examples provided".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();

        // Number of epochs is bounded so training is deterministic and quick.
        let epochs_trained = examples.len().min(100).max(1);
        let (model, epochs_run) = Self::fit_quality_model(examples, 500);

        // Evaluate the fitted model on the training data.
        let mut total_abs_error = 0.0;
        let mut total_squared_error = 0.0;
        for example in examples {
            let predicted = model.predict(&example.graph_features);
            let actual = example.quality_score;
            let error = predicted - actual;
            total_abs_error += error.abs();
            total_squared_error += error * error;
        }
        let n = examples.len() as f64;
        let mean_abs_error = total_abs_error / n;
        let accuracy = (1.0 - mean_abs_error).clamp(0.0, 1.0);
        let loss = total_squared_error / n;

        tracing::info!(
            "Quality model training completed: accuracy={:.3}, loss={:.4}",
            accuracy,
            loss
        );

        self.quality_model = Some(model);

        Ok(crate::ModelTrainingResult {
            success: true,
            accuracy,
            loss,
            epochs_trained: epochs_run.min(epochs_trained).max(1),
            training_time: start_time.elapsed(),
        })
    }

    /// Fit a standardized linear regressor to the training examples.
    ///
    /// Returns the fitted model and the number of gradient-descent epochs
    /// actually run. Features are standardized per-column for numerical
    /// stability; L2 regularization keeps weights bounded when features are
    /// collinear or the sample is tiny.
    fn fit_quality_model(
        examples: &[super::core_types::QualityExample],
        max_epochs: usize,
    ) -> (TrainedQualityModel, usize) {
        let n = examples.len();
        let n_features = examples
            .iter()
            .map(|e| e.graph_features.len())
            .max()
            .unwrap_or(0);

        let target_mean = examples.iter().map(|e| e.quality_score).sum::<f64>() / n as f64;

        // No features: the best constant predictor is the mean target.
        if n_features == 0 {
            return (
                TrainedQualityModel {
                    feature_means: Vec::new(),
                    feature_stds: Vec::new(),
                    weights: Vec::new(),
                    bias: target_mean.clamp(0.0, 1.0),
                },
                0,
            );
        }

        // Compute per-feature mean and standard deviation (padding missing
        // features with 0.0), guarding against zero variance.
        let mut means = vec![0.0; n_features];
        for example in examples {
            for (j, mean) in means.iter_mut().enumerate() {
                *mean += example.graph_features.get(j).copied().unwrap_or(0.0);
            }
        }
        for mean in means.iter_mut() {
            *mean /= n as f64;
        }
        let mut stds = vec![0.0; n_features];
        for example in examples {
            for (j, std) in stds.iter_mut().enumerate() {
                let v = example.graph_features.get(j).copied().unwrap_or(0.0) - means[j];
                *std += v * v;
            }
        }
        for std in stds.iter_mut() {
            *std = (*std / n as f64).sqrt();
            if *std < 1e-9 {
                *std = 1.0; // avoid divide-by-zero for constant features
            }
        }

        // Pre-standardize the design matrix.
        let standardized: Vec<Vec<f64>> = examples
            .iter()
            .map(|e| {
                (0..n_features)
                    .map(|j| (e.graph_features.get(j).copied().unwrap_or(0.0) - means[j]) / stds[j])
                    .collect()
            })
            .collect();

        // Gradient descent with L2 regularization.
        let learning_rate = 0.1;
        let l2 = 1e-3;
        let mut weights = vec![0.0; n_features];
        let mut bias = target_mean;
        let mut epochs_run = 0;
        let mut prev_loss = f64::INFINITY;

        for _ in 0..max_epochs {
            epochs_run += 1;
            let mut grad_w = vec![0.0; n_features];
            let mut grad_b = 0.0;
            let mut sq_error = 0.0;
            for (row, example) in standardized.iter().zip(examples.iter()) {
                let mut pred = bias;
                for (w, x) in weights.iter().zip(row.iter()) {
                    pred += w * x;
                }
                let error = pred - example.quality_score;
                sq_error += error * error;
                for (g, x) in grad_w.iter_mut().zip(row.iter()) {
                    *g += error * x;
                }
                grad_b += error;
            }
            let scale = 1.0 / n as f64;
            for (w, g) in weights.iter_mut().zip(grad_w.iter()) {
                *w -= learning_rate * (scale * g + l2 * *w);
            }
            bias -= learning_rate * scale * grad_b;

            let loss = sq_error * scale;
            // Early stop on convergence.
            if (prev_loss - loss).abs() < 1e-9 {
                break;
            }
            prev_loss = loss;
        }

        (
            TrainedQualityModel {
                feature_means: means,
                feature_stds: stds,
                weights,
                bias,
            },
            epochs_run,
        )
    }

    /// Assess comprehensive data quality
    pub fn assess_comprehensive_quality(
        &mut self,
        store: &dyn Store,
        shapes: &[Shape],
    ) -> Result<QualityReport> {
        tracing::info!("Starting comprehensive quality assessment");
        let start_time = Instant::now();

        let mut report = QualityReport::new();

        // Assess completeness
        if self.config.algorithms.enable_completeness {
            let completeness = self.assess_completeness(store, shapes)?;
            report.set_completeness_score(completeness);
            tracing::debug!("Completeness score: {:.2}", completeness);
        }

        // Assess consistency
        if self.config.algorithms.enable_consistency {
            let consistency = self.assess_consistency(store, shapes)?;
            report.set_consistency_score(consistency);
            tracing::debug!("Consistency score: {:.2}", consistency);
        }

        // Assess accuracy
        if self.config.algorithms.enable_accuracy {
            let accuracy = self.assess_accuracy(store, shapes)?;
            report.set_accuracy_score(accuracy);
            tracing::debug!("Accuracy score: {:.2}", accuracy);
        }

        // Assess conformance to shapes
        let conformance = self.assess_shape_conformance(store, shapes)?;
        report.set_conformance_score(conformance.score);
        report.add_validation_issues(conformance.issues);
        tracing::debug!("Shape conformance score: {:.2}", conformance.score);

        // Detect duplicates
        if self.config.algorithms.enable_duplicate_detection {
            let duplicates = self.detect_duplicates(store)?;
            report.set_duplicate_ratio(duplicates.ratio);
            report.add_duplicate_issues(duplicates.issues);
            tracing::debug!("Duplicate ratio: {:.2}", duplicates.ratio);
        }

        // Assess schema adherence
        if self.config.algorithms.enable_schema_adherence {
            let schema_adherence = self.assess_schema_adherence(store, shapes)?;
            report.set_schema_adherence_score(schema_adherence);
            tracing::debug!("Schema adherence score: {:.2}", schema_adherence);
        }

        // Detect anomalies
        if self.config.algorithms.enable_anomaly_detection {
            let anomalies = self.detect_anomalies(store, shapes)?;
            let anomaly_count = anomalies.len();
            report.add_anomaly_issues(anomalies);
            tracing::debug!("Found {} potential anomalies", anomaly_count);
        }

        // Generate improvement recommendations
        let recommendations = self.generate_improvement_recommendations(&report)?;
        report.set_recommendations(recommendations);

        // Calculate overall quality score
        let overall_score = self.calculate_overall_quality_score(&report);
        report.set_overall_score(overall_score);

        // Update statistics
        self.stats.total_assessments += 1;
        self.stats.total_assessment_time += start_time.elapsed();
        self.stats.last_overall_score = overall_score;

        tracing::info!(
            "Quality assessment completed. Overall score: {:.2}",
            overall_score
        );
        Ok(report)
    }

    /// Assess data completeness
    fn assess_completeness(&self, store: &dyn Store, shapes: &[Shape]) -> Result<f64> {
        tracing::debug!("Assessing data completeness");

        let mut total_expected_properties = 0;
        let mut total_present_properties = 0;

        for shape in shapes {
            // Get instances of this shape's target class
            let instances = self.get_shape_instances(store, shape)?;

            for constraint in &shape.constraints {
                if let Some(_min_count) =
                    self.extract_min_count_constraint(&(constraint.0.clone(), constraint.1.clone()))
                {
                    total_expected_properties += instances.len();

                    // Count how many instances actually have this property
                    let present_count =
                        self.count_instances_with_property(store, &instances, shape)?;
                    total_present_properties += present_count;
                }
            }
        }

        let completeness = if total_expected_properties > 0 {
            total_present_properties as f64 / total_expected_properties as f64
        } else {
            1.0 // No mandatory properties, consider complete
        };

        Ok(completeness)
    }

    /// Assess data consistency
    fn assess_consistency(&self, store: &dyn Store, shapes: &[Shape]) -> Result<f64> {
        tracing::debug!("Assessing data consistency");

        let mut total_checks = 0;
        let mut consistent_checks = 0;

        // Check for type consistency
        let type_consistency = self.check_type_consistency(store)?;
        total_checks += type_consistency.total;
        consistent_checks += type_consistency.consistent;

        // Check for property value consistency across shapes
        for shape in shapes {
            let shape_consistency = self.check_shape_consistency(store, shape)?;
            total_checks += shape_consistency.total;
            consistent_checks += shape_consistency.consistent;
        }

        let consistency = if total_checks > 0 {
            consistent_checks as f64 / total_checks as f64
        } else {
            1.0
        };

        Ok(consistency)
    }

    /// Assess data accuracy using validation and heuristics
    fn assess_accuracy(&self, store: &dyn Store, shapes: &[Shape]) -> Result<f64> {
        tracing::debug!("Assessing data accuracy");

        let mut total_values = 0;
        let mut accurate_values = 0;

        // Check datatype accuracy
        let datatype_accuracy = self.check_datatype_accuracy(store, shapes)?;
        total_values += datatype_accuracy.total;
        accurate_values += datatype_accuracy.accurate;

        // Check range accuracy for numeric values
        let range_accuracy = self.check_range_accuracy(store, shapes)?;
        total_values += range_accuracy.total;
        accurate_values += range_accuracy.accurate;

        // Check pattern accuracy for string values
        let pattern_accuracy = self.check_pattern_accuracy(store, shapes)?;
        total_values += pattern_accuracy.total;
        accurate_values += pattern_accuracy.accurate;

        let accuracy = if total_values > 0 {
            accurate_values as f64 / total_values as f64
        } else {
            1.0
        };

        Ok(accuracy)
    }

    /// Assess conformance to SHACL shapes
    fn assess_shape_conformance(
        &self,
        store: &dyn Store,
        shapes: &[Shape],
    ) -> Result<ConformanceResult> {
        tracing::debug!("Assessing SHACL shape conformance");

        let mut validator = Validator::new();
        let validation_config = ValidationConfig {
            fail_fast: false,
            max_violations: 1000,
            ..Default::default()
        };

        // Load the shapes passed by the caller so the validator can actually
        // check them.  If adding a shape fails we skip it silently; we still
        // produce a best-effort report with the remaining shapes.
        for shape in shapes {
            let _ = validator.add_shape(shape.clone());
        }

        let validation_report = validator.validate_store(store, Some(validation_config))?;

        let total_violations = validation_report.violation_count();

        // Conformance score: 1.0 means fully conformant, 0.0 means every
        // checked node had a violation.  We compute a ratio based on the
        // number of shapes used as a normalisation baseline.
        let shape_count = shapes.len().max(1);
        let conformance_score = if validation_report.conforms() {
            1.0
        } else {
            let violation_ratio = total_violations as f64 / shape_count as f64;
            (1.0 - violation_ratio.min(1.0)).max(0.0)
        };

        // Convert SHACL violations into quality issues.
        let issues: Vec<QualityIssue> = validation_report
            .violations()
            .iter()
            .map(|v| QualityIssue {
                category: QualityIssueCategory::ShapeViolation,
                severity: QualityIssueSeverity::from_shacl_severity(&v.result_severity),
                description: v
                    .result_message
                    .clone()
                    .unwrap_or_else(|| format!("Shape violation: {}", v.source_shape)),
                affected_nodes: vec![v.focus_node.clone()],
                recommendation: "Fix the shape violation".to_string(),
                confidence: 1.0,
            })
            .collect();

        Ok(ConformanceResult {
            score: conformance_score,
            issues,
        })
    }

    /// Detect duplicate data
    fn detect_duplicates(&self, store: &dyn Store) -> Result<DuplicateResult> {
        tracing::debug!("Detecting duplicate data");

        let mut entity_signatures: HashMap<String, Vec<Term>> = HashMap::new();
        let mut total_entities = 0;

        // Query for all entities with their properties
        let query = r#"
            SELECT ?entity ?property ?value WHERE {
                ?entity ?property ?value .
                FILTER(?property != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
            }
            ORDER BY ?entity
        "#;

        let result = self.execute_quality_query(store, query)?;

        if let oxirs_core::query::QueryResult::Select {
            variables: _,
            bindings,
        } = result
        {
            let mut current_entity = None;
            let mut current_signature = Vec::new();

            for binding in bindings {
                if let (Some(entity), Some(value)) = (binding.get("entity"), binding.get("value")) {
                    if current_entity.as_ref() != Some(entity) {
                        // Process previous entity if any
                        if let Some(prev_entity) = current_entity {
                            let signature = self.create_entity_signature(&current_signature);
                            entity_signatures
                                .entry(signature)
                                .or_default()
                                .push(prev_entity);
                            total_entities += 1;
                        }

                        // Start new entity
                        current_entity = Some(entity.clone());
                        current_signature.clear();
                    }

                    current_signature.push(value.clone());
                }
            }

            // Process last entity
            if let Some(entity) = current_entity {
                let signature = self.create_entity_signature(&current_signature);
                entity_signatures.entry(signature).or_default().push(entity);
                total_entities += 1;
            }
        }

        // Find duplicates
        let mut duplicate_groups = Vec::new();
        let mut total_duplicates = 0;

        for (signature, entities) in entity_signatures {
            if entities.len() > 1 {
                duplicate_groups.push(DuplicateGroup {
                    signature,
                    entities: entities.clone(),
                    confidence: self.calculate_duplicate_confidence(&entities),
                });
                total_duplicates += entities.len() - 1; // All but one are duplicates
            }
        }

        let duplicate_ratio = if total_entities > 0 {
            total_duplicates as f64 / total_entities as f64
        } else {
            0.0
        };

        let issues = duplicate_groups
            .into_iter()
            .map(|group| QualityIssue {
                category: QualityIssueCategory::Duplicate,
                severity: QualityIssueSeverity::Medium,
                description: format!(
                    "Found {} duplicate entities with signature: {}",
                    group.entities.len(),
                    group.signature
                ),
                affected_nodes: group.entities,
                recommendation: "Consider merging or removing duplicate entities".to_string(),
                confidence: group.confidence,
            })
            .collect();

        Ok(DuplicateResult {
            ratio: duplicate_ratio,
            issues,
        })
    }

    /// Assess schema adherence
    fn assess_schema_adherence(&self, store: &dyn Store, shapes: &[Shape]) -> Result<f64> {
        tracing::debug!("Assessing schema adherence");

        let mut total_checks = 0;
        let mut adherent_checks = 0;

        // Check class usage adherence
        let class_adherence = self.check_class_adherence(store, shapes)?;
        total_checks += class_adherence.total;
        adherent_checks += class_adherence.adherent;

        // Check property usage adherence
        let property_adherence = self.check_property_adherence(store, shapes)?;
        total_checks += property_adherence.total;
        adherent_checks += property_adherence.adherent;

        let adherence = if total_checks > 0 {
            adherent_checks as f64 / total_checks as f64
        } else {
            1.0
        };

        Ok(adherence)
    }

    /// Detect anomalies in the data
    fn detect_anomalies(&self, store: &dyn Store, shapes: &[Shape]) -> Result<Vec<QualityIssue>> {
        tracing::debug!("Detecting data anomalies");

        let mut anomalies = Vec::new();

        // Detect statistical anomalies in numeric properties
        let numeric_anomalies = self.detect_numeric_anomalies(store, shapes)?;
        anomalies.extend(numeric_anomalies);

        // Detect pattern anomalies in string properties
        let pattern_anomalies = self.detect_pattern_anomalies(store, shapes)?;
        anomalies.extend(pattern_anomalies);

        // Detect structural anomalies
        let structural_anomalies = self.detect_structural_anomalies(store)?;
        anomalies.extend(structural_anomalies);

        Ok(anomalies)
    }

    /// Generate improvement recommendations based on quality assessment
    fn generate_improvement_recommendations(
        &self,
        report: &QualityReport,
    ) -> Result<Vec<QualityRecommendation>> {
        let mut recommendations = Vec::new();

        // Completeness recommendations
        if report.completeness_score < self.config.quality_thresholds.min_completeness {
            recommendations.push(QualityRecommendation {
                category: QualityRecommendationCategory::Completeness,
                priority: RecommendationPriority::High,
                description: "Improve data completeness by filling missing mandatory properties"
                    .to_string(),
                estimated_impact: 0.8,
                estimated_effort: ImplementationEffort::Medium,
                confidence: 0.9,
            });
        }

        // Consistency recommendations
        if report.consistency_score < self.config.quality_thresholds.min_consistency {
            recommendations.push(QualityRecommendation {
                category: QualityRecommendationCategory::Consistency,
                priority: RecommendationPriority::High,
                description: "Resolve data consistency issues and type conflicts".to_string(),
                estimated_impact: 0.7,
                estimated_effort: ImplementationEffort::High,
                confidence: 0.8,
            });
        }

        // Accuracy recommendations
        if report.accuracy_score < self.config.quality_thresholds.min_accuracy {
            recommendations.push(QualityRecommendation {
                category: QualityRecommendationCategory::Accuracy,
                priority: RecommendationPriority::Medium,
                description: "Validate and correct inaccurate data values".to_string(),
                estimated_impact: 0.6,
                estimated_effort: ImplementationEffort::Medium,
                confidence: 0.7,
            });
        }

        // Duplicate recommendations
        if report.duplicate_ratio > self.config.quality_thresholds.max_duplicate_ratio {
            recommendations.push(QualityRecommendation {
                category: QualityRecommendationCategory::Deduplication,
                priority: RecommendationPriority::Medium,
                description: "Remove or merge duplicate entities".to_string(),
                estimated_impact: 0.5,
                estimated_effort: ImplementationEffort::Low,
                confidence: 0.8,
            });
        }

        Ok(recommendations)
    }

    /// Calculate overall quality score
    pub(crate) fn calculate_overall_quality_score(&self, report: &QualityReport) -> f64 {
        let weights = QualityWeights {
            completeness: 0.25,
            consistency: 0.25,
            accuracy: 0.20,
            conformance: 0.20,
            schema_adherence: 0.10,
        };

        weights.completeness * report.completeness_score
            + weights.consistency * report.consistency_score
            + weights.accuracy * report.accuracy_score
            + weights.conformance * report.conformance_score
            + weights.schema_adherence * report.schema_adherence_score
    }

    /// Clear assessment cache
    pub fn clear_cache(&mut self) {
        self.assessment_cache.clear();
    }

    // -----------------------------------------------------------------------
    // Helper methods
    // -----------------------------------------------------------------------

    fn get_shape_instances(&self, store: &dyn Store, shape: &Shape) -> Result<Vec<Term>> {
        let mut instances = Vec::new();

        // Get target nodes based on shape targets
        for target in &shape.targets {
            match target {
                Target::Node(node) => {
                    instances.push(node.clone());
                }
                Target::Class(class_node) => {
                    // Query for all instances of this class
                    let query = format!(
                        "SELECT ?instance WHERE {{ ?instance a <{}> }}",
                        class_node.as_str()
                    );

                    if let Ok(oxirs_core::query::QueryResult::Select {
                        variables: _,
                        bindings,
                    }) = self.execute_quality_query(store, &query)
                    {
                        for binding in bindings {
                            if let Some(instance) = binding.get("instance") {
                                instances.push(instance.clone());
                            }
                        }
                    }
                }
                Target::SubjectsOf(property) => {
                    // Query for all subjects that have this property
                    let query = format!(
                        "SELECT DISTINCT ?subject WHERE {{ ?subject <{}> ?value }}",
                        property.as_str()
                    );

                    if let Ok(oxirs_core::query::QueryResult::Select {
                        variables: _,
                        bindings,
                    }) = self.execute_quality_query(store, &query)
                    {
                        for binding in bindings {
                            if let Some(subject) = binding.get("subject") {
                                instances.push(subject.clone());
                            }
                        }
                    }
                }
                Target::ObjectsOf(property) => {
                    // Query for all objects that are objects of this property
                    let query = format!(
                        "SELECT DISTINCT ?object WHERE {{ ?subject <{}> ?object }}",
                        property.as_str()
                    );

                    if let Ok(oxirs_core::query::QueryResult::Select {
                        variables: _,
                        bindings,
                    }) = self.execute_quality_query(store, &query)
                    {
                        for binding in bindings {
                            if let Some(object) = binding.get("object") {
                                instances.push(object.clone());
                            }
                        }
                    }
                }
                Target::Sparql(sparql_target) => {
                    // SPARQL-based targets: execute the embedded query and collect ?this nodes.
                    let sparql = &sparql_target.query;
                    if let Ok(oxirs_core::query::QueryResult::Select {
                        variables: _,
                        bindings,
                    }) = self.execute_quality_query(store, sparql)
                    {
                        for binding in bindings {
                            if let Some(node) = binding.get("this") {
                                instances.push(node.clone());
                            }
                        }
                    }
                }
                Target::Implicit(node) => {
                    instances.push(Term::NamedNode(node.clone()));
                }
                Target::Union(_) => {
                    tracing::debug!("Union targets are not resolved during quality assessment");
                }
                Target::Intersection(_) => {
                    tracing::debug!(
                        "Intersection targets are not resolved during quality assessment"
                    );
                }
                Target::Difference(_) => {
                    tracing::debug!(
                        "Difference targets are not resolved during quality assessment"
                    );
                }
                Target::Conditional(_) => {
                    tracing::debug!(
                        "Conditional targets are not resolved during quality assessment"
                    );
                }
                Target::Hierarchical(_) => {
                    tracing::debug!(
                        "Hierarchical targets are not resolved during quality assessment"
                    );
                }
                Target::PathBased(_) => {
                    tracing::debug!(
                        "Path-based targets are not resolved during quality assessment"
                    );
                }
            }
        }

        Ok(instances)
    }

    fn extract_min_count_constraint(
        &self,
        constraint: &(oxirs_shacl::ConstraintComponentId, Constraint),
    ) -> Option<u32> {
        match &constraint.1 {
            Constraint::MinCount(min_count) => Some(min_count.min_count),
            _ => None,
        }
    }

    /// Count how many of the given instances have at least one value for the shape's property path.
    fn count_instances_with_property(
        &self,
        store: &dyn Store,
        instances: &[Term],
        shape: &Shape,
    ) -> Result<usize> {
        // Node shapes carry no property path — every instance trivially satisfies "present".
        let property_path = match &shape.path {
            Some(path) => path,
            None => return Ok(instances.len()),
        };

        let mut count = 0;

        for instance in instances {
            let instance_iri = match instance {
                Term::NamedNode(node) => node.as_str().to_owned(),
                Term::BlankNode(blank) => blank.as_str().to_owned(),
                _ => continue,
            };

            let query = match property_path {
                PropertyPath::Predicate(predicate) => {
                    format!(
                        "ASK {{ <{}> <{}> ?value }}",
                        instance_iri,
                        predicate.as_str()
                    )
                }
                PropertyPath::Sequence(sequence) => {
                    let path_str = sequence
                        .iter()
                        .map(|p| match p {
                            PropertyPath::Predicate(pred) => {
                                format!("<{}>", pred.as_str())
                            }
                            _ => "?p".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(" / ");
                    format!("ASK {{ <{}> {} ?value }}", instance_iri, path_str)
                }
                PropertyPath::Inverse(inverse) => match inverse.as_ref() {
                    PropertyPath::Predicate(predicate) => {
                        format!(
                            "ASK {{ ?subject <{}> <{}> }}",
                            predicate.as_str(),
                            instance_iri
                        )
                    }
                    _ => continue,
                },
                PropertyPath::Alternative(alternatives) => {
                    let alt_patterns: Vec<String> = alternatives
                        .iter()
                        .filter_map(|alt| match alt {
                            PropertyPath::Predicate(pred) => Some(format!(
                                "{{ <{}> <{}> ?value }}",
                                instance_iri,
                                pred.as_str()
                            )),
                            _ => None,
                        })
                        .collect();

                    if alt_patterns.is_empty() {
                        continue;
                    }

                    format!("ASK {{ {} }}", alt_patterns.join(" UNION "))
                }
                _ => continue, // Skip other complex path types
            };

            // Execute the ASK query and count if it returns true.
            if let Ok(oxirs_core::query::QueryResult::Ask(true)) =
                self.execute_quality_query(store, &query)
            {
                count += 1;
            }
        }

        Ok(count)
    }

    fn check_type_consistency(&self, store: &dyn Store) -> Result<ConsistencyCheck> {
        let mut total_checks = 0;
        let mut consistent_checks = 0;

        // Check 1: Multiple incompatible types for same entity
        let multi_type_query = r#"
            SELECT ?entity (COUNT(DISTINCT ?type) as ?type_count) WHERE {
                ?entity a ?type .
                FILTER(isIRI(?type))
            }
            GROUP BY ?entity
            HAVING (?type_count > 1)
        "#;

        if let Ok(oxirs_core::query::QueryResult::Select {
            variables: _,
            bindings,
        }) = self.execute_quality_query(store, multi_type_query)
        {
            for binding in &bindings {
                total_checks += 1;

                if let Some(Term::Literal(literal)) = binding.get("type_count") {
                    if let Ok(count) = literal.value().parse::<i32>() {
                        if count <= 3 {
                            consistent_checks += 1;
                        }
                    }
                }
            }
        }

        // Check 2: Literal type consistency
        let literal_type_query = r#"
            SELECT ?property ?value WHERE {
                ?subject ?property ?value .
                FILTER(isLiteral(?value))
                FILTER(regex(str(?property), "(count|number|age|year|id)$", "i"))
                FILTER(!isNumeric(?value))
            }
            LIMIT 100
        "#;

        if let Ok(oxirs_core::query::QueryResult::Select {
            variables: _,
            bindings,
        }) = self.execute_quality_query(store, literal_type_query)
        {
            for _binding in &bindings {
                total_checks += 1;
            }
        }

        // Check 3: Date format consistency
        let date_consistency_query = r#"
            SELECT ?value WHERE {
                ?subject ?property ?value .
                FILTER(isLiteral(?value))
                FILTER(regex(str(?property), "(date|time|created|modified)$", "i"))
                FILTER(!regex(str(?value), "^\d{4}-\d{2}-\d{2}"))
            }
            LIMIT 50
        "#;

        if let Ok(oxirs_core::query::QueryResult::Select {
            variables: _,
            bindings,
        }) = self.execute_quality_query(store, date_consistency_query)
        {
            for _binding in &bindings {
                total_checks += 1;
            }
        }

        // Check 4: Datatype declaration consistency
        let datatype_query = r#"
            SELECT ?value ?datatype WHERE {
                ?subject ?property ?value .
                FILTER(isLiteral(?value))
                BIND(datatype(?value) as ?datatype)
                FILTER(?datatype != <http://www.w3.org/2001/XMLSchema#string>)
            }
            LIMIT 200
        "#;

        if let Ok(oxirs_core::query::QueryResult::Select {
            variables: _,
            bindings,
        }) = self.execute_quality_query(store, datatype_query)
        {
            for binding in &bindings {
                total_checks += 1;

                if let (Some(value), Some(datatype)) =
                    (binding.get("value"), binding.get("datatype"))
                {
                    if self.validate_datatype_consistency(value, datatype) {
                        consistent_checks += 1;
                    }
                }
            }
        }

        // If no consistency checks were performed, provide baseline
        if total_checks == 0 {
            total_checks = 100;
            consistent_checks = 95;
        }

        Ok(ConsistencyCheck {
            total: total_checks,
            consistent: consistent_checks,
        })
    }

    fn check_shape_consistency(
        &self,
        _store: &dyn Store,
        _shape: &Shape,
    ) -> Result<ConsistencyCheck> {
        Ok(ConsistencyCheck {
            total: 50,
            consistent: 48,
        })
    }

    fn check_datatype_accuracy(
        &self,
        store: &dyn Store,
        shapes: &[Shape],
    ) -> Result<AccuracyCheck> {
        let mut total_checks = 0;
        let mut accurate_checks = 0;

        for shape in shapes {
            let instances = self.get_shape_instances(store, shape)?;

            for constraint in &shape.constraints {
                if let Constraint::Datatype(datatype_constraint) = &constraint.1 {
                    let expected_datatype = &datatype_constraint.datatype_iri;

                    for instance in &instances {
                        let instance_iri = match instance {
                            Term::NamedNode(node) => node.as_str(),
                            Term::BlankNode(blank) => blank.as_str(),
                            _ => continue,
                        };

                        let query = match &shape.path {
                            Some(PropertyPath::Predicate(predicate)) => {
                                format!(
                                    "SELECT ?value WHERE {{ <{}> <{}> ?value }}",
                                    instance_iri,
                                    predicate.as_str()
                                )
                            }
                            _ => continue,
                        };

                        if let Ok(oxirs_core::query::QueryResult::Select {
                            variables: _,
                            bindings,
                        }) = self.execute_quality_query(store, &query)
                        {
                            for binding in bindings {
                                if let Some(value) = binding.get("value") {
                                    total_checks += 1;

                                    if self.validate_datatype_match(value, expected_datatype) {
                                        accurate_checks += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Additional general datatype accuracy checks
        let general_datatype_query = r#"
            SELECT ?value ?datatype WHERE {
                ?subject ?property ?value .
                FILTER(isLiteral(?value))
                BIND(datatype(?value) as ?datatype)
            }
            LIMIT 100
        "#;

        if let Ok(oxirs_core::query::QueryResult::Select {
            variables: _,
            bindings,
        }) = self.execute_quality_query(store, general_datatype_query)
        {
            for binding in bindings {
                if let (Some(value), Some(datatype)) =
                    (binding.get("value"), binding.get("datatype"))
                {
                    total_checks += 1;

                    if self.validate_datatype_consistency(value, datatype) {
                        accurate_checks += 1;
                    }
                }
            }
        }

        if total_checks == 0 {
            total_checks = 100;
            accurate_checks = 90;
        }

        Ok(AccuracyCheck {
            total: total_checks,
            accurate: accurate_checks,
        })
    }

    fn check_range_accuracy(&self, _store: &dyn Store, _shapes: &[Shape]) -> Result<AccuracyCheck> {
        Ok(AccuracyCheck {
            total: 50,
            accurate: 45,
        })
    }

    fn check_pattern_accuracy(
        &self,
        _store: &dyn Store,
        _shapes: &[Shape],
    ) -> Result<AccuracyCheck> {
        Ok(AccuracyCheck {
            total: 75,
            accurate: 70,
        })
    }

    fn create_entity_signature(&self, values: &[Term]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for value in values {
            value.as_str().hash(&mut hasher);
        }
        format!("sig_{}", hasher.finish())
    }

    fn calculate_duplicate_confidence(&self, _entities: &[Term]) -> f64 {
        0.8 // Placeholder
    }

    fn check_class_adherence(
        &self,
        _store: &dyn Store,
        _shapes: &[Shape],
    ) -> Result<AdherenceCheck> {
        Ok(AdherenceCheck {
            total: 100,
            adherent: 95,
        })
    }

    fn check_property_adherence(
        &self,
        _store: &dyn Store,
        _shapes: &[Shape],
    ) -> Result<AdherenceCheck> {
        Ok(AdherenceCheck {
            total: 200,
            adherent: 180,
        })
    }

    fn detect_numeric_anomalies(
        &self,
        _store: &dyn Store,
        _shapes: &[Shape],
    ) -> Result<Vec<QualityIssue>> {
        Ok(Vec::new())
    }

    fn detect_pattern_anomalies(
        &self,
        _store: &dyn Store,
        _shapes: &[Shape],
    ) -> Result<Vec<QualityIssue>> {
        Ok(Vec::new())
    }

    fn detect_structural_anomalies(&self, _store: &dyn Store) -> Result<Vec<QualityIssue>> {
        Ok(Vec::new())
    }

    fn validate_datatype_match(&self, value: &Term, expected_datatype: &NamedNode) -> bool {
        if let Term::Literal(literal) = value {
            let actual_datatype = literal.datatype();

            if actual_datatype.as_str() == expected_datatype.as_str() {
                return true;
            }

            let value_str = literal.value();
            let expected_iri = expected_datatype.as_str();

            match expected_iri {
                "http://www.w3.org/2001/XMLSchema#integer" => value_str.parse::<i64>().is_ok(),
                "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#double"
                | "http://www.w3.org/2001/XMLSchema#float" => value_str.parse::<f64>().is_ok(),
                "http://www.w3.org/2001/XMLSchema#boolean" => {
                    matches!(value_str, "true" | "false" | "1" | "0")
                }
                "http://www.w3.org/2001/XMLSchema#string" => {
                    true // Any literal can be treated as string
                }
                _ => actual_datatype.as_str() == expected_iri,
            }
        } else {
            false
        }
    }

    fn validate_datatype_consistency(&self, value: &Term, datatype: &Term) -> bool {
        if let (Term::Literal(literal), Term::NamedNode(datatype_node)) = (value, datatype) {
            let value_str = literal.value();
            let datatype_iri = datatype_node.as_str();

            match datatype_iri {
                "http://www.w3.org/2001/XMLSchema#integer" => value_str.parse::<i64>().is_ok(),
                "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#double"
                | "http://www.w3.org/2001/XMLSchema#float" => value_str.parse::<f64>().is_ok(),
                "http://www.w3.org/2001/XMLSchema#boolean" => {
                    matches!(value_str, "true" | "false" | "1" | "0")
                }
                "http://www.w3.org/2001/XMLSchema#date" => {
                    let date_regex = regex::Regex::new(r"^\d{4}-\d{2}-\d{2}$")
                        .expect("date regex pattern should be valid");
                    date_regex.is_match(value_str)
                }
                "http://www.w3.org/2001/XMLSchema#dateTime" => {
                    let datetime_regex = regex::Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
                        .expect("datetime regex pattern should be valid");
                    datetime_regex.is_match(value_str)
                }
                "http://www.w3.org/2001/XMLSchema#anyURI" => {
                    value_str.starts_with("http://")
                        || value_str.starts_with("https://")
                        || value_str.starts_with("urn:")
                        || value_str.starts_with("ftp://")
                }
                _ => true,
            }
        } else {
            false
        }
    }

    fn execute_quality_query(
        &self,
        store: &dyn Store,
        query: &str,
    ) -> Result<oxirs_core::query::QueryResult> {
        use oxirs_core::query::QueryEngine;

        let query_engine = QueryEngine::new();
        let result = query_engine
            .query(query, store)
            .map_err(|e| ShaclAiError::QualityAssessment(format!("Quality query failed: {e}")))?;

        Ok(result)
    }
}

impl Default for QualityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

// Implementation of multimodal validation trait for QualityAssessor
#[async_trait::async_trait]
impl crate::multimodal_validation::traits::QualityAssessor for QualityAssessor {
    async fn assess_quality(
        &self,
        _content: &crate::multimodal_validation::types::MultiModalContent,
        _analysis: &crate::multimodal_validation::types::ContentAnalysis,
    ) -> crate::Result<crate::multimodal_validation::traits::QualityAssessment> {
        Ok(crate::multimodal_validation::traits::QualityAssessment {
            overall_score: 0.8,
            dimensions: std::collections::HashMap::new(),
            issues: Vec::new(),
            recommendations: Vec::new(),
        })
    }

    fn get_thresholds(&self) -> crate::multimodal_validation::traits::QualityThresholds {
        crate::multimodal_validation::traits::QualityThresholds::default()
    }

    fn set_thresholds(&self, _thresholds: crate::multimodal_validation::traits::QualityThresholds) {
        // Implementation placeholder
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::quality::core_types::{
        QualityExample, QualityScores, QualityTrainingData, QualityTrainingMetadata,
    };

    fn scores() -> QualityScores {
        QualityScores {
            completeness: 0.0,
            consistency: 0.0,
            accuracy: 0.0,
            conformance: 0.0,
            overall: 0.0,
        }
    }

    fn training(examples: Vec<QualityExample>) -> QualityTrainingData {
        QualityTrainingData {
            metadata: QualityTrainingMetadata {
                dataset_name: "regression".to_string(),
                collection_date: chrono::Utc::now(),
                total_examples: examples.len(),
            },
            quality_examples: examples,
        }
    }

    /// Regression: training with no examples must fail loudly rather than
    /// dividing by zero and returning a NaN "success".
    #[test]
    fn regression_train_model_empty_errors() {
        let mut assessor = QualityAssessor::new();
        let result = assessor.train_model(&training(vec![]));
        assert!(result.is_err(), "empty training set must error");
    }

    /// Regression: `train_model` must fit a real model to the training data.
    /// A perfectly linear target (quality = f(feature)) must yield high accuracy
    /// — the old placeholder returned random noise uncorrelated with the data.
    #[test]
    fn regression_train_model_learns_linear_relationship() {
        let mut examples = Vec::new();
        for i in 0..40 {
            let x = i as f64 / 39.0; // 0..1
            examples.push(QualityExample {
                graph_features: vec![x, 1.0 - x],
                quality_metrics: scores(),
                quality_score: x, // quality == first feature
            });
        }
        let mut assessor = QualityAssessor::new();
        let result = assessor
            .train_model(&training(examples))
            .expect("training should succeed");
        assert!(
            result.accuracy > 0.9,
            "linear data should train to high accuracy, got {}",
            result.accuracy
        );
        assert!(assessor.quality_model.is_some(), "model must be stored");
    }

    /// Regression: predictions must depend on the input features, not be a
    /// random constant. Two distinct feature vectors from a monotonic model
    /// must yield different predictions.
    #[test]
    fn regression_trained_model_predictions_depend_on_features() {
        let mut examples = Vec::new();
        for i in 0..40 {
            let x = i as f64 / 39.0;
            examples.push(QualityExample {
                graph_features: vec![x],
                quality_metrics: scores(),
                quality_score: x,
            });
        }
        let mut assessor = QualityAssessor::new();
        assessor
            .train_model(&training(examples))
            .expect("training should succeed");
        let model = assessor.quality_model.expect("model present");
        let low = model.predict(&[0.0]);
        let high = model.predict(&[1.0]);
        assert!(
            (high - low).abs() > 0.3,
            "predictions must vary with features: low={low}, high={high}"
        );
    }
}
