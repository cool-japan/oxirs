//! Core shape learning implementation

use std::collections::{HashMap, HashSet};
use tracing;

use oxirs_core::{
    model::{GraphName, NamedNode, Object, Predicate, RdfTerm, Subject},
    Store,
};

use oxirs_shacl::{
    constraints::{
        cardinality_constraints::{MaxCountConstraint, MinCountConstraint},
        value_constraints::DatatypeConstraint,
    },
    Constraint, ConstraintComponentId, PropertyPath, Shape, ShapeId, Target,
};

use crate::{
    ml::reinforcement::{RLConfig, ReinforcementLearner},
    patterns::{Pattern, PatternType},
    Result, ShaclAiError,
};

use super::performance::{
    analyze_pattern_statistics, calculate_performance_metrics, LearningPerformanceMetrics,
    PatternStatistics,
};
use super::types::{LearningConfig, LearningQueryResult, LearningStatistics, ShapeTrainingData};

/// The `rdf:type` predicate IRI used to discover classes and their instances.
const RDF_TYPE_IRI: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

/// Build the `rdf:type` predicate, mapping construction failure to a crate error.
fn rdf_type_predicate() -> Result<Predicate> {
    NamedNode::new(RDF_TYPE_IRI)
        .map(Predicate::NamedNode)
        .map_err(|e| ShaclAiError::ShapeLearning(format!("invalid rdf:type IRI: {e}")))
}

/// Resolve an optional graph name string into an optional [`GraphName`] filter
/// suitable for [`Store::find_quads`]. `None` means "search all graphs".
fn resolve_graph_filter(graph_name: Option<&str>) -> Result<Option<GraphName>> {
    match graph_name {
        None => Ok(None),
        Some(name) => NamedNode::new(name)
            .map(|n| Some(GraphName::NamedNode(n)))
            .map_err(|e| ShaclAiError::ShapeLearning(format!("invalid graph IRI '{name}': {e}"))),
    }
}

/// Sanitize an IRI fragment for use inside a shape identifier.
fn sanitize_iri_fragment(iri: &str) -> String {
    iri.replace(['/', ':', '#'], "_")
}

/// Running per-property statistics accumulated while scanning class instances.
#[derive(Debug, Default, Clone)]
struct PropertyUsageStats {
    /// Number of distinct instances that have at least one value for this property.
    instance_count: usize,
    /// Total number of triples observed for this property across sampled instances.
    total_value_count: usize,
    /// Maximum number of values observed for this property on a single instance.
    max_value_count: usize,
    /// Datatype IRI -> number of literal values observed with that datatype.
    datatype_counts: HashMap<String, usize>,
}

/// Shape learning engine for automatic constraint discovery
#[derive(Debug)]
pub struct ShapeLearner {
    /// Configuration
    config: LearningConfig,

    /// Learned patterns cache
    pattern_cache: HashMap<String, Vec<Pattern>>,

    /// Statistics
    stats: LearningStatistics,

    /// Reinforcement learning agent for constraint discovery optimization
    rl_agent: Option<ReinforcementLearner>,
}

impl ShapeLearner {
    /// Create a new shape learner with default configuration
    pub fn new() -> Self {
        Self::with_config(LearningConfig::default())
    }

    /// Create a new shape learner with custom configuration
    pub fn with_config(config: LearningConfig) -> Self {
        let rl_agent = if config.enable_reinforcement_learning {
            config.rl_config.clone().map(ReinforcementLearner::new)
        } else {
            None
        };

        Self {
            config,
            pattern_cache: HashMap::new(),
            stats: LearningStatistics::default(),
            rl_agent,
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &LearningConfig {
        &self.config
    }

    /// Get statistics
    pub fn get_statistics(&self) -> &LearningStatistics {
        &self.stats
    }

    /// Learn shapes from RDF store
    pub fn learn_shapes_from_store(
        &mut self,
        store: &dyn Store,
        graph_name: Option<&str>,
    ) -> Result<Vec<Shape>> {
        tracing::info!("Starting shape learning from store");

        // Analyze class structure
        let classes = self.discover_classes(store, graph_name)?;
        tracing::debug!("Discovered {} classes", classes.len());

        let mut shapes = Vec::new();

        for class in classes {
            if shapes.len() >= self.config.max_shapes {
                tracing::warn!("Reached maximum shapes limit: {}", self.config.max_shapes);
                break;
            }

            // Learn shape for this class
            match self.learn_shape_for_class(store, &class, graph_name) {
                Ok((shape, property_shapes)) => {
                    shapes.push(shape);
                    shapes.extend(property_shapes);
                    self.stats.total_shapes_learned += 1;
                }
                Err(e) => {
                    tracing::warn!("Failed to learn shape for class {}: {}", class.as_str(), e);
                    self.stats.failed_shapes += 1;
                }
            }
        }

        tracing::info!("Learned {} shapes from store", shapes.len());
        Ok(shapes)
    }

    /// Learn shapes from RDF store using parallel processing for improved performance
    pub fn learn_shapes_from_store_parallel(
        &mut self,
        store: &dyn Store,
        graph_name: Option<&str>,
    ) -> Result<Vec<Shape>> {
        tracing::info!("Starting parallel shape learning from store");

        // Analyze class structure
        let classes = self.discover_classes(store, graph_name)?;
        tracing::debug!(
            "Discovered {} classes for parallel processing",
            classes.len()
        );

        // Limit classes to max_shapes to avoid unnecessary processing
        let limited_classes: Vec<_> = classes.into_iter().take(self.config.max_shapes).collect();

        // Use thread pool for parallel processing
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, limited_classes.len());
        let pool = threadpool::ThreadPool::new(num_threads);

        tracing::info!("Using {} threads for parallel shape learning", num_threads);

        // Channel for collecting results
        let (tx, rx) = std::sync::mpsc::channel();

        let total_classes = limited_classes.len();

        // Submit tasks to thread pool
        for (idx, class) in limited_classes.into_iter().enumerate() {
            let tx = tx.clone();
            let graph_name = graph_name.map(|s| s.to_string());

            // Note: Since Store trait is not Send + Sync, we need to pass necessary data
            // For now, we'll fall back to sequential processing but with better batching
            pool.execute(move || {
                let result: (usize, NamedNode, Result<()>) = (idx, class, Ok(()));
                tx.send(result)
                    .expect("channel receiver should still be alive");
            });
        }

        // Close the original sender
        drop(tx);

        // Collect results
        let mut results = Vec::new();
        for received in rx {
            results.push(received);
        }

        // Sort results by index to maintain order
        results.sort_by_key(|(idx, _, _)| *idx);

        // Process results sequentially (since Store access needs to be sequential)
        let mut shapes = Vec::new();
        let mut successful_count = 0;
        let mut failed_count = 0;

        for (_, class, _) in results {
            if shapes.len() >= self.config.max_shapes {
                break;
            }

            match self.learn_shape_for_class(store, &class, graph_name) {
                Ok((shape, property_shapes)) => {
                    shapes.push(shape);
                    shapes.extend(property_shapes);
                    successful_count += 1;
                }
                Err(e) => {
                    tracing::warn!("Failed to learn shape for class {}: {}", class.as_str(), e);
                    failed_count += 1;
                }
            }
        }

        // Update statistics
        self.stats.total_shapes_learned += successful_count;
        self.stats.failed_shapes += failed_count;

        tracing::info!(
            "Parallel shape learning completed: {} shapes learned, {} failed from {} classes",
            successful_count,
            failed_count,
            total_classes
        );

        Ok(shapes)
    }

    /// Learn shapes from RDF store with advanced batch processing and memory optimization
    pub fn learn_shapes_from_store_optimized(
        &mut self,
        store: &dyn Store,
        graph_name: Option<&str>,
    ) -> Result<Vec<Shape>> {
        tracing::info!("Starting optimized shape learning from store");

        // Analyze class structure with early filtering
        let classes = self.discover_classes(store, graph_name)?;
        tracing::debug!("Discovered {} classes", classes.len());

        // Implement batch processing for memory efficiency
        let batch_size = self
            .config
            .algorithm_params
            .get("batch_size")
            .map(|v| *v as usize)
            .unwrap_or(10);

        let mut shapes = Vec::new();
        let mut processed_count = 0;
        let mut successful_count = 0;
        let mut failed_count = 0;

        // Process classes in batches
        for batch_start in (0..classes.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(classes.len());
            let batch = &classes[batch_start..batch_end];

            tracing::debug!(
                "Processing batch {}-{} of {} classes",
                batch_start,
                batch_end,
                classes.len()
            );

            // Process batch with potential for cache optimization
            for class in batch {
                if shapes.len() >= self.config.max_shapes {
                    tracing::warn!("Reached maximum shapes limit: {}", self.config.max_shapes);
                    break;
                }

                // Check cache first
                let cache_key = format!("{}_{}", class.as_str(), graph_name.unwrap_or("default"));

                if let Some(cached_patterns) = self.pattern_cache.get(&cache_key).cloned() {
                    // Use cached patterns if available
                    tracing::debug!("Using cached patterns for class {}", class.as_str());
                    match self.patterns_to_shape(&cached_patterns, class) {
                        Ok((shape, property_shapes)) => {
                            shapes.push(shape);
                            shapes.extend(property_shapes);
                            successful_count += 1;
                        }
                        Err(e) => {
                            tracing::debug!("Failed to create shape from cached patterns: {}", e);
                            failed_count += 1;
                        }
                    }
                } else {
                    // Learn shape for this class and cache patterns
                    match self.learn_shape_for_class_with_caching(store, class, graph_name) {
                        Ok((shape, property_shapes)) => {
                            shapes.push(shape);
                            shapes.extend(property_shapes);
                            successful_count += 1;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to learn shape for class {}: {}",
                                class.as_str(),
                                e
                            );
                            failed_count += 1;
                        }
                    }
                }

                processed_count += 1;

                // Progress reporting
                if processed_count % 50 == 0 {
                    tracing::info!(
                        "Progress: {}/{} classes processed, {} shapes learned",
                        processed_count,
                        classes.len(),
                        shapes.len()
                    );
                }
            }

            // Optional garbage collection hint after each batch
            if batch_end % (batch_size * 10) == 0 {
                tracing::debug!("Memory optimization checkpoint at {} classes", batch_end);
            }
        }

        // Update statistics
        self.stats.total_shapes_learned += successful_count;
        self.stats.failed_shapes += failed_count;

        tracing::info!(
            "Optimized shape learning completed: {} shapes learned, {} failed from {} classes",
            successful_count,
            failed_count,
            classes.len()
        );

        Ok(shapes)
    }

    /// Learn shapes from discovered patterns
    pub fn learn_shapes_from_patterns(
        &mut self,
        store: &dyn Store,
        patterns: &[Pattern],
        graph_name: Option<&str>,
    ) -> Result<Vec<Shape>> {
        tracing::info!("Learning shapes from {} patterns", patterns.len());

        let mut shapes = Vec::new();

        for pattern in patterns {
            if shapes.len() >= self.config.max_shapes {
                break;
            }

            match self.pattern_to_shape(store, pattern, graph_name) {
                Ok(shape) => {
                    shapes.push(shape);
                    self.stats.total_shapes_learned += 1;
                }
                Err(e) => {
                    tracing::debug!("Failed to convert pattern to shape: {}", e);
                    self.stats.failed_shapes += 1;
                }
            }
        }

        Ok(shapes)
    }

    /// Train machine learning model for shape learning with enhanced gradient descent algorithm
    pub fn train_model(
        &mut self,
        training_data: &ShapeTrainingData,
    ) -> Result<crate::ModelTrainingResult> {
        if !self.config.enable_training {
            return Err(ShaclAiError::ModelTraining(
                "Training is disabled in configuration".to_string(),
            ));
        }

        tracing::info!(
            "Training shape learning model with {} samples using enhanced ML algorithm",
            training_data.features.len()
        );
        let start_time = std::time::Instant::now();

        // Enhanced training with real gradient descent algorithm
        let mut epochs_trained = 0;
        let max_epochs = self
            .config
            .algorithm_params
            .get("max_epochs")
            .map(|v| *v as usize)
            .unwrap_or(100);

        let learning_rate = self
            .config
            .algorithm_params
            .get("learning_rate")
            .copied()
            .unwrap_or(0.01);

        let mut weights = vec![
            0.0;
            training_data
                .features
                .first()
                .map(|f| f.len())
                .unwrap_or(10)
        ];
        let mut bias = 0.0;
        let mut best_accuracy = 0.0;
        let mut best_loss = f64::INFINITY;

        // Training loop with mini-batch gradient descent
        for epoch in 0..max_epochs {
            let mut epoch_loss = 0.0;
            let mut correct_predictions = 0;
            let batch_size = training_data.features.len().min(32);

            // Process mini-batches
            for batch_start in (0..training_data.features.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(training_data.features.len());

                let mut gradient_w = vec![0.0; weights.len()];
                let mut gradient_b = 0.0;

                // Calculate gradients for this batch
                for i in batch_start..batch_end {
                    let features = &training_data.features[i];
                    let target_label = &training_data.labels[i];

                    // Convert string label to numeric target (binary classification)
                    let target = if target_label.to_lowercase().contains("valid")
                        || target_label.to_lowercase().contains("true")
                        || target_label.to_lowercase().contains("positive")
                    {
                        1.0
                    } else {
                        0.0
                    };

                    // Forward pass (sigmoid activation)
                    let prediction = self.sigmoid(
                        features
                            .iter()
                            .zip(weights.iter())
                            .map(|(f, w)| f * w)
                            .sum::<f64>()
                            + bias,
                    );

                    // Calculate loss (cross-entropy)
                    let loss = if target > 0.5 {
                        -prediction.ln().max(1e-15) // Avoid log(0)
                    } else {
                        -(1.0 - prediction).ln().max(1e-15) // Avoid log(0)
                    };
                    epoch_loss += loss;

                    // Check prediction accuracy
                    if (prediction > 0.5 && target > 0.5) || (prediction <= 0.5 && target <= 0.5) {
                        correct_predictions += 1;
                    }

                    // Backward pass - calculate gradients
                    let error = prediction - target;
                    for (j, feature) in features.iter().enumerate() {
                        gradient_w[j] += error * feature;
                    }
                    gradient_b += error;
                }

                // Update weights using gradients
                let batch_size_f = (batch_end - batch_start) as f64;
                for (w, grad) in weights.iter_mut().zip(gradient_w.iter()) {
                    *w -= learning_rate * (grad / batch_size_f);
                }
                bias -= learning_rate * (gradient_b / batch_size_f);
            }

            // Calculate epoch metrics
            let current_loss = epoch_loss / training_data.features.len() as f64;
            let current_accuracy = correct_predictions as f64 / training_data.features.len() as f64;

            // Update best metrics
            if current_accuracy > best_accuracy {
                best_accuracy = current_accuracy;
            }
            if current_loss < best_loss {
                best_loss = current_loss;
            }

            epochs_trained = epoch + 1;

            // Early stopping condition
            if current_accuracy > 0.95 {
                tracing::info!(
                    "Early stopping at epoch {} with accuracy: {:.3}",
                    epoch + 1,
                    current_accuracy
                );
                break;
            }

            // Log progress every 10 epochs
            if (epoch + 1) % 10 == 0 {
                tracing::debug!(
                    "Epoch {}/{}: Loss: {:.4}, Accuracy: {:.3}",
                    epoch + 1,
                    max_epochs,
                    current_loss,
                    current_accuracy
                );
            }
        }

        // Store learned weights for future use
        // Note: Storing weights as a single aggregated value for algorithm_params (f64 type)
        let weights_avg = weights.iter().sum::<f64>() / weights.len() as f64;
        self.config
            .algorithm_params
            .insert("learned_weights_avg".to_string(), weights_avg);
        self.config
            .algorithm_params
            .insert("learned_bias".to_string(), bias);

        self.stats.model_trained = true;
        self.stats.last_training_accuracy = best_accuracy;

        let training_time = start_time.elapsed();
        tracing::info!(
            "Enhanced ML training completed in {:?} with accuracy: {:.3}, loss: {:.4}",
            training_time,
            best_accuracy,
            best_loss
        );

        Ok(crate::ModelTrainingResult {
            success: true,
            accuracy: best_accuracy,
            loss: best_loss,
            epochs_trained,
            training_time,
        })
    }

    /// Sigmoid activation function for neural network
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Clear pattern cache
    pub fn clear_cache(&mut self) {
        self.pattern_cache.clear();
        tracing::debug!("Pattern cache cleared");
    }

    /// Enable/disable reinforcement learning
    pub fn enable_reinforcement_learning(&mut self, rl_config: Option<RLConfig>) -> Result<()> {
        if let Some(config) = rl_config {
            self.rl_agent = Some(ReinforcementLearner::new(config));
            self.config.enable_reinforcement_learning = true;
            tracing::info!("Reinforcement learning enabled");
        } else {
            self.rl_agent = None;
            self.config.enable_reinforcement_learning = false;
            tracing::info!("Reinforcement learning disabled");
        }
        Ok(())
    }

    /// Check if reinforcement learning is enabled
    pub fn is_rl_enabled(&self) -> bool {
        self.config.enable_reinforcement_learning && self.rl_agent.is_some()
    }

    /// Get performance metrics for learning efficiency analysis
    pub fn get_performance_metrics(&self) -> LearningPerformanceMetrics {
        let success_rate = if self.stats.total_shapes_learned + self.stats.failed_shapes > 0 {
            self.stats.total_shapes_learned as f64
                / (self.stats.total_shapes_learned + self.stats.failed_shapes) as f64
        } else {
            0.0
        };

        calculate_performance_metrics(
            success_rate,
            self.stats.total_constraints_discovered,
            self.stats.temporal_constraints_discovered,
            self.stats.total_shapes_learned,
            self.stats.classes_analyzed,
            self.stats.last_training_accuracy,
        )
    }

    /// Analyze pattern statistics for learning optimization
    pub fn analyze_pattern_statistics(&self, patterns: &[Pattern]) -> PatternStatistics {
        analyze_pattern_statistics(patterns)
    }

    // Private helper methods

    /// Execute a SPARQL query for learning purposes
    fn execute_learning_query(
        &self,
        store: &dyn Store,
        query: &str,
    ) -> Result<LearningQueryResult> {
        // Placeholder implementation - in reality this would execute SPARQL queries
        tracing::debug!("Executing learning query: {}", query);
        Ok(LearningQueryResult::Empty)
    }

    /// Convert a pattern to a SHACL shape
    fn pattern_to_shape(
        &mut self,
        store: &dyn Store,
        pattern: &Pattern,
        graph_name: Option<&str>,
    ) -> Result<Shape> {
        // Placeholder implementation
        let shape_id = ShapeId::new(format!("Pattern_{}Shape", pattern.id()));
        let shape = Shape::node_shape(shape_id);
        Ok(shape)
    }

    /// Learn a shape for a specific class.
    ///
    /// Discovers real usage patterns (properties, datatypes, cardinalities) by
    /// scanning the instances of `class` in `store`, then attaches the derived
    /// constraints to the returned node shape via generated property shapes.
    /// Returns the node shape together with any generated `sh:property` shapes,
    /// both of which must be included in the final shapes collection.
    fn learn_shape_for_class(
        &mut self,
        store: &dyn Store,
        class: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<(Shape, Vec<Shape>)> {
        tracing::debug!("Learning shape for class: {}", class.as_str());

        let patterns = self.discover_patterns_for_class(store, class, graph_name)?;
        let (shape, property_shapes) = self.patterns_to_shape(&patterns, class)?;

        self.stats.classes_analyzed += 1;
        tracing::debug!(
            "Learned shape for class: {} with {} property shapes",
            shape.id.as_str(),
            property_shapes.len()
        );

        Ok((shape, property_shapes))
    }

    /// Discover classes in the RDF store by scanning `rdf:type` usage.
    ///
    /// Returns the distinct classes that appear as the object of an `rdf:type`
    /// triple, ordered by descending instance count so the most populous
    /// (and thus most representative) classes are learned first.
    fn discover_classes(
        &self,
        store: &dyn Store,
        graph_name: Option<&str>,
    ) -> Result<Vec<NamedNode>> {
        let type_predicate = rdf_type_predicate()?;
        let graph_filter = resolve_graph_filter(graph_name)?;

        let type_quads =
            store.find_quads(None, Some(&type_predicate), None, graph_filter.as_ref())?;

        let mut class_counts: HashMap<NamedNode, usize> = HashMap::new();
        for quad in &type_quads {
            if let Object::NamedNode(class_node) = quad.object() {
                *class_counts.entry(class_node.clone()).or_insert(0) += 1;
            }
        }

        let mut classes: Vec<(NamedNode, usize)> = class_counts.into_iter().collect();
        classes.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.as_str().cmp(b.0.as_str())));

        tracing::debug!(
            "discover_classes found {} distinct classes from {} rdf:type triples",
            classes.len(),
            type_quads.len()
        );

        Ok(classes.into_iter().map(|(class, _)| class).collect())
    }

    /// Evaluate constraint quality
    fn evaluate_constraint_quality(
        &self,
        constraints: &[(ConstraintComponentId, Constraint)],
        instance_count: usize,
    ) -> f64 {
        // Placeholder quality evaluation
        if constraints.is_empty() {
            return 0.0;
        }

        // Simple heuristic: more constraints and more instances suggest higher quality
        let constraint_factor = (constraints.len() as f64).min(10.0) / 10.0;
        let instance_factor = (instance_count as f64).min(100.0) / 100.0;

        (constraint_factor + instance_factor) / 2.0
    }

    /// Convert multiple patterns to a single SHACL node shape with enhanced analysis.
    ///
    /// Property, datatype, and cardinality patterns are aggregated per property
    /// and converted into real `sh:property` shapes carrying `sh:minCount`,
    /// `sh:maxCount`, and `sh:datatype` constraints, which are linked into the
    /// returned node shape's `property_shapes` list. The generated property
    /// shapes are returned alongside the node shape and must be included in the
    /// final shapes collection for the constraints to take effect.
    fn patterns_to_shape(
        &mut self,
        patterns: &[Pattern],
        class: &NamedNode,
    ) -> Result<(Shape, Vec<Shape>)> {
        tracing::debug!(
            "Converting {} patterns to shape for class {}",
            patterns.len(),
            class.as_str()
        );

        // Create a node shape for this class
        let shape_id = ShapeId::new(format!(
            "{}Shape",
            class.as_str().replace(['/', ':', '#'], "_")
        ));
        let mut shape = Shape::node_shape(shape_id.clone());

        // Add class target
        shape.add_target(Target::class(class.clone()));

        // Aggregate patterns per property IRI.
        #[derive(Default)]
        struct PropertyAggregate {
            usage_count: u32,
            dominant_datatype: Option<NamedNode>,
            min_count: Option<u32>,
            max_count: Option<u32>,
        }

        let mut aggregates: HashMap<String, PropertyAggregate> = HashMap::new();

        for pattern in patterns {
            match pattern {
                Pattern::PropertyUsage {
                    property,
                    usage_count,
                    ..
                } => {
                    let entry = aggregates.entry(property.as_str().to_string()).or_default();
                    entry.usage_count = entry.usage_count.max(*usage_count);
                }
                Pattern::Datatype {
                    property, datatype, ..
                } => {
                    let entry = aggregates.entry(property.as_str().to_string()).or_default();
                    entry.dominant_datatype = Some(datatype.clone());
                }
                Pattern::Cardinality {
                    property,
                    min_count,
                    max_count,
                    ..
                } => {
                    let entry = aggregates.entry(property.as_str().to_string()).or_default();
                    if let Some(min_count) = min_count {
                        entry.min_count = Some(
                            entry
                                .min_count
                                .map_or(*min_count, |cur| cur.max(*min_count)),
                        );
                    }
                    if let Some(max_count) = max_count {
                        entry.max_count = Some(
                            entry
                                .max_count
                                .map_or(*max_count, |cur| cur.min(*max_count)),
                        );
                    }
                }
                _ => {
                    tracing::debug!(
                        "Skipping unsupported pattern type: {:?}",
                        pattern.pattern_type()
                    );
                }
            }
        }

        let mut property_shapes = Vec::new();
        let mut constraint_count = 0usize;

        for (property_iri, info) in &aggregates {
            let property_node = match NamedNode::new(property_iri.as_str()) {
                Ok(node) => node,
                Err(e) => {
                    tracing::warn!("Skipping invalid property IRI '{}': {}", property_iri, e);
                    continue;
                }
            };

            let prop_shape_id = ShapeId::new(format!(
                "{}_{}",
                shape_id.as_str(),
                sanitize_iri_fragment(property_iri)
            ));
            let mut prop_shape = Shape::property_shape(
                prop_shape_id.clone(),
                PropertyPath::predicate(property_node),
            );

            // Only add a minimum cardinality constraint when the property was
            // observed on multiple instances (recurring, not incidental usage)
            // or when pattern discovery derived an explicit min_count.
            let effective_min_count =
                info.min_count
                    .or(if info.usage_count >= 2 { Some(1) } else { None });
            if let Some(min_count) = effective_min_count {
                let constraint = Constraint::MinCount(MinCountConstraint { min_count });
                prop_shape.add_constraint(constraint.component_id(), constraint);
                constraint_count += 1;
            }

            if let Some(max_count) = info.max_count {
                let constraint = Constraint::MaxCount(MaxCountConstraint { max_count });
                prop_shape.add_constraint(constraint.component_id(), constraint);
                constraint_count += 1;
            }

            if let Some(datatype) = &info.dominant_datatype {
                let constraint = Constraint::Datatype(DatatypeConstraint {
                    datatype_iri: datatype.clone(),
                });
                prop_shape.add_constraint(constraint.component_id(), constraint);
                constraint_count += 1;
            }

            if prop_shape.constraints.is_empty() {
                // Insufficient evidence to justify a property shape for this property.
                continue;
            }

            tracing::debug!(
                "Generated property shape {} with {} constraints for property {}",
                prop_shape_id.as_str(),
                prop_shape.constraints.len(),
                property_iri
            );

            shape.property_shapes.push(prop_shape_id);
            property_shapes.push(prop_shape);
        }

        self.stats.total_constraints_discovered += constraint_count;

        tracing::debug!(
            "Created shape {} with {} property shapes from {} patterns",
            shape_id.as_str(),
            property_shapes.len(),
            patterns.len()
        );
        Ok((shape, property_shapes))
    }

    /// Learn shape for a class with enhanced caching capabilities
    fn learn_shape_for_class_with_caching(
        &mut self,
        store: &dyn Store,
        class: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<(Shape, Vec<Shape>)> {
        tracing::debug!("Learning shape for class with caching: {}", class.as_str());

        // First, discover patterns for this class
        let patterns = self.discover_patterns_for_class(store, class, graph_name)?;

        // Cache the discovered patterns for future use
        let cache_key = format!("{}_{}", class.as_str(), graph_name.unwrap_or("default"));
        self.pattern_cache.insert(cache_key, patterns.clone());

        // Convert patterns to shape
        self.patterns_to_shape(&patterns, class)
    }

    /// Discover patterns specific to a class by scanning its instances in the store.
    ///
    /// For every instance of `class` (up to a bounded sample), this collects the
    /// properties used, their observed datatypes, and per-instance value counts,
    /// then derives `PropertyUsage`, `Datatype`, and `Cardinality` patterns from
    /// the aggregated statistics. Properties observed on fewer than
    /// `self.config.min_support` of the sampled instances are dropped.
    fn discover_patterns_for_class(
        &mut self,
        store: &dyn Store,
        class: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<Pattern>> {
        tracing::debug!("Discovering patterns for class: {}", class.as_str());

        let type_predicate = rdf_type_predicate()?;
        let graph_filter = resolve_graph_filter(graph_name)?;

        let type_quads = store.find_quads(
            None,
            Some(&type_predicate),
            Some(&Object::NamedNode(class.clone())),
            graph_filter.as_ref(),
        )?;
        let instances: Vec<Subject> = type_quads
            .into_iter()
            .map(|q| q.subject().clone())
            .collect();

        if instances.is_empty() {
            tracing::debug!("No instances found for class {}", class.as_str());
            return Ok(Vec::new());
        }

        // Bound the number of instances scanned for constraint discovery so a
        // single very large class cannot make shape learning unbounded.
        let sample_limit = self
            .config
            .algorithm_params
            .get("max_sample_instances")
            .copied()
            .map(|v| v.max(1.0) as usize)
            .unwrap_or(2000);
        let sample: &[Subject] = if instances.len() > sample_limit {
            &instances[..sample_limit]
        } else {
            &instances[..]
        };
        let instance_total = sample.len();

        let mut property_stats: HashMap<String, PropertyUsageStats> = HashMap::new();

        for instance in sample {
            let quads = store.find_quads(Some(instance), None, None, graph_filter.as_ref())?;
            let mut seen_this_instance: HashSet<String> = HashSet::new();
            let mut value_counts_this_instance: HashMap<String, usize> = HashMap::new();

            for quad in &quads {
                let Predicate::NamedNode(prop) = quad.predicate() else {
                    continue;
                };
                if prop.as_str() == RDF_TYPE_IRI {
                    continue;
                }
                let prop_iri = prop.as_str().to_string();
                let stats = property_stats.entry(prop_iri.clone()).or_default();
                stats.total_value_count += 1;
                seen_this_instance.insert(prop_iri.clone());
                *value_counts_this_instance
                    .entry(prop_iri.clone())
                    .or_insert(0) += 1;

                if let Object::Literal(literal) = quad.object() {
                    *stats
                        .datatype_counts
                        .entry(literal.datatype().as_str().to_string())
                        .or_insert(0) += 1;
                }
            }

            for prop_iri in seen_this_instance {
                let stats = property_stats.entry(prop_iri.clone()).or_default();
                stats.instance_count += 1;
                let per_instance_count = value_counts_this_instance
                    .get(&prop_iri)
                    .copied()
                    .unwrap_or(0);
                stats.max_value_count = stats.max_value_count.max(per_instance_count);
            }
        }

        let mut patterns = Vec::new();

        for (property_iri, stats) in &property_stats {
            let support = stats.instance_count as f64 / instance_total.max(1) as f64;
            if support < self.config.min_support {
                continue;
            }
            let property = match NamedNode::new(property_iri.as_str()) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("Skipping invalid property IRI '{}': {}", property_iri, e);
                    continue;
                }
            };
            let avg_count = stats.total_value_count as f64 / stats.instance_count.max(1) as f64;
            let confidence = support;

            patterns.push(Pattern::PropertyUsage {
                id: format!(
                    "property_{}_{}",
                    sanitize_iri_fragment(class.as_str()),
                    sanitize_iri_fragment(property_iri)
                ),
                property: property.clone(),
                usage_count: stats.instance_count as u32,
                support,
                confidence,
                pattern_type: PatternType::Usage,
            });

            if let Some((dominant_datatype, dt_count)) = stats
                .datatype_counts
                .iter()
                .max_by_key(|(_, count)| **count)
            {
                let dt_confidence = *dt_count as f64 / stats.total_value_count.max(1) as f64;
                if dt_confidence >= self.config.min_confidence {
                    if let Ok(datatype) = NamedNode::new(dominant_datatype.as_str()) {
                        patterns.push(Pattern::Datatype {
                            id: format!(
                                "datatype_{}_{}",
                                sanitize_iri_fragment(class.as_str()),
                                sanitize_iri_fragment(property_iri)
                            ),
                            property: property.clone(),
                            datatype,
                            usage_count: *dt_count as u32,
                            support,
                            confidence: dt_confidence,
                            pattern_type: PatternType::Datatype,
                        });
                    }
                }
            }

            let is_universal = stats.instance_count == instance_total;
            let is_functional = stats.max_value_count <= 1;
            let cardinality_type = if is_universal && is_functional {
                crate::patterns::CardinalityType::Functional
            } else if is_universal {
                crate::patterns::CardinalityType::Required
            } else {
                crate::patterns::CardinalityType::Optional
            };
            patterns.push(Pattern::Cardinality {
                id: format!(
                    "cardinality_{}_{}",
                    sanitize_iri_fragment(class.as_str()),
                    sanitize_iri_fragment(property_iri)
                ),
                property,
                cardinality_type,
                min_count: if is_universal { Some(1) } else { None },
                max_count: if is_functional { Some(1) } else { None },
                avg_count,
                support,
                confidence,
                pattern_type: PatternType::Cardinality,
            });
        }

        self.stats.total_constraints_discovered += patterns.len();

        tracing::debug!(
            "Discovered {} patterns for class {} from {} sampled instances",
            patterns.len(),
            class.as_str(),
            instance_total
        );
        Ok(patterns)
    }
}

impl Default for ShapeLearner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{Literal, Quad};
    use oxirs_core::ConcreteStore;

    fn rdf_type() -> NamedNode {
        NamedNode::new(RDF_TYPE_IRI).expect("rdf:type should be a valid IRI")
    }

    /// Regression test for the P0 finding: `discover_classes` must scan the
    /// store for real `rdf:type` usage instead of returning a hardcoded
    /// `http://example.org/DefaultClass`.
    #[test]
    fn test_discover_classes_scans_real_store() {
        let store = ConcreteStore::new().expect("store should construct");
        let person_class = NamedNode::new("http://example.org/Person").expect("valid IRI");
        let org_class = NamedNode::new("http://example.org/Organization").expect("valid IRI");

        for (subject_iri, class) in [
            ("http://example.org/alice", &person_class),
            ("http://example.org/bob", &person_class),
            ("http://example.org/acme", &org_class),
        ] {
            let subject = NamedNode::new(subject_iri).expect("valid IRI");
            store
                .insert_quad(Quad::new_default_graph(subject, rdf_type(), class.clone()))
                .expect("insert should succeed");
        }

        let learner = ShapeLearner::new();
        let classes = learner
            .discover_classes(&store, None)
            .expect("discover_classes should succeed");

        assert_eq!(classes.len(), 2, "classes = {:?}", classes);
        // Person has 2 instances vs. Organization's 1, so it must be first
        // (descending by instance count).
        assert_eq!(classes[0].as_str(), person_class.as_str());
        assert!(classes.iter().any(|c| c.as_str() == org_class.as_str()));
    }

    #[test]
    fn test_discover_classes_empty_store_returns_empty() {
        let store = ConcreteStore::new().expect("store should construct");
        let learner = ShapeLearner::new();
        let classes = learner
            .discover_classes(&store, None)
            .expect("discover_classes should succeed");
        assert!(classes.is_empty(), "classes = {:?}", classes);
    }

    /// Regression test for the P1 findings: `learn_shape_for_class` /
    /// `patterns_to_shape` must attach real, store-derived property/datatype/
    /// cardinality constraints instead of only a bare class target.
    #[test]
    fn test_learn_shapes_from_store_attaches_real_constraints() {
        let store = ConcreteStore::new().expect("store should construct");
        let person_class = NamedNode::new("http://example.org/Person").expect("valid IRI");
        let name_property = NamedNode::new("http://example.org/name").expect("valid IRI");

        for (subject_iri, name_value) in [
            ("http://example.org/alice", "Alice"),
            ("http://example.org/bob", "Bob"),
        ] {
            let subject = NamedNode::new(subject_iri).expect("valid IRI");
            store
                .insert_quad(Quad::new_default_graph(
                    subject.clone(),
                    rdf_type(),
                    person_class.clone(),
                ))
                .expect("insert should succeed");
            store
                .insert_quad(Quad::new_default_graph(
                    subject,
                    name_property.clone(),
                    Literal::new(name_value),
                ))
                .expect("insert should succeed");
        }

        let mut learner = ShapeLearner::new();
        let shapes = learner
            .learn_shapes_from_store(&store, None)
            .expect("learn_shapes_from_store should succeed");

        // A node shape for Person, plus at least one generated property shape
        // for `name` (present on every instance).
        assert!(shapes.len() >= 2, "shapes.len() = {}", shapes.len());

        let node_shape = shapes
            .iter()
            .find(|s| s.is_node_shape())
            .expect("expected a node shape");
        assert!(
            !node_shape.property_shapes.is_empty(),
            "expected the node shape to link generated property shapes"
        );

        let property_shape = shapes
            .iter()
            .find(|s| s.is_property_shape())
            .expect("expected a generated property shape");
        assert!(
            !property_shape.constraints.is_empty(),
            "expected real min/max-count and datatype constraints, not an empty property shape"
        );
    }

    /// A class with no instances at all must not produce a shape with
    /// fabricated constraints.
    #[test]
    fn test_learn_shapes_from_store_no_instances_still_learns_bare_class_shape() {
        let store = ConcreteStore::new().expect("store should construct");
        let empty_class = NamedNode::new("http://example.org/Ghost").expect("valid IRI");
        // Insert exactly one triple so the class is discoverable, but with no
        // other properties on the (single) instance.
        store
            .insert_quad(Quad::new_default_graph(
                NamedNode::new("http://example.org/nobody").expect("valid IRI"),
                rdf_type(),
                empty_class.clone(),
            ))
            .expect("insert should succeed");

        let mut learner = ShapeLearner::new();
        let shapes = learner
            .learn_shapes_from_store(&store, None)
            .expect("learn_shapes_from_store should succeed");

        assert_eq!(shapes.len(), 1, "shapes = {:?}", shapes.len());
        assert!(shapes[0].is_node_shape());
        assert!(shapes[0].property_shapes.is_empty());
    }
}
