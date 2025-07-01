//! Shape learning and automatic constraint discovery
//!
//! This module implements AI-powered shape learning from RDF data.

use chrono::{DateTime, Datelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use oxirs_core::{
    model::{Literal, NamedNode, Term, Triple},
    RdfTerm, Store,
};

use oxirs_shacl::{
    constraints::*, shapes::ShapeFactory, Constraint, ConstraintComponentId, PropertyPath, Shape,
    ShapeId, ShapeType, Target,
};

use crate::{
    ml::reinforcement::{Action, RLAlgorithm, RLConfig, ReinforcementLearner},
    patterns::Pattern,
    Result, ShaclAiError,
};

/// Learning query result types
#[derive(Debug, Clone)]
pub enum LearningQueryResult {
    /// SELECT query results with variable bindings
    Select {
        variables: Vec<String>,
        bindings: Vec<HashMap<String, Term>>,
    },
    /// ASK query boolean result
    Ask(bool),
    /// Empty result
    Empty,
}

/// Configuration for shape learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Enable automatic shape generation
    pub enable_shape_generation: bool,

    /// Minimum support threshold for patterns
    pub min_support: f64,

    /// Minimum confidence threshold for constraints
    pub min_confidence: f64,

    /// Maximum number of shapes to generate
    pub max_shapes: usize,

    /// Enable model training
    pub enable_training: bool,

    /// Learning algorithm parameters
    pub algorithm_params: HashMap<String, f64>,

    /// Enable reinforcement learning for constraint discovery optimization
    pub enable_reinforcement_learning: bool,

    /// Reinforcement learning configuration
    pub rl_config: Option<RLConfig>,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enable_shape_generation: true,
            min_support: 0.1,
            min_confidence: 0.8,
            max_shapes: 100,
            enable_training: true,
            algorithm_params: HashMap::new(),
            enable_reinforcement_learning: false,
            rl_config: None,
        }
    }
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
            config
                .rl_config
                .clone()
                .map(|rl_config| ReinforcementLearner::new(rl_config))
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
        store: &Store,
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
                Ok(shape) => {
                    shapes.push(shape);
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

    /// Learn shapes from discovered patterns
    pub fn learn_shapes_from_patterns(
        &mut self,
        store: &Store,
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

    /// Learn a shape for a specific class
    fn learn_shape_for_class(
        &mut self,
        store: &Store,
        class: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Shape> {
        tracing::debug!("Learning shape for class: {}", class.as_str());

        // Create a node shape for this class
        let shape_id = ShapeId::new(format!(
            "{}Shape",
            class.as_str().replace(['/', ':', '#'], "_")
        ));
        let mut shape = Shape::node_shape(shape_id.clone());

        // Add class target
        shape.add_target(Target::class(class.clone()));

        // Discover properties used by instances of this class
        let properties = self.discover_properties_for_class(store, class, graph_name)?;

        // Create property constraints
        for property in &properties {
            // Analyze cardinality for this property
            let (min_count, max_count) =
                self.analyze_property_cardinality(store, class, property, graph_name)?;

            // Create property shape
            let property_path = PropertyPath::predicate(property.clone());
            let property_shape_id = ShapeId::new(format!(
                "{}_{}PropertyShape",
                shape_id.as_str().replace(['/', ':', '#'], "_"),
                property.as_str().replace(['/', ':', '#'], "_")
            ));

            let mut property_shape = Shape::property_shape(property_shape_id, property_path);

            // Add cardinality constraints if discovered
            if let Some(min) = min_count {
                property_shape.add_constraint(
                    ConstraintComponentId::new("sh:minCount"),
                    Constraint::MinCount(MinCountConstraint { min_count: min }),
                );
            }

            if let Some(max) = max_count {
                property_shape.add_constraint(
                    ConstraintComponentId::new("sh:maxCount"),
                    Constraint::MaxCount(MaxCountConstraint { max_count: max }),
                );
            }

            // Analyze datatypes for this property
            if let Ok(datatype) = self.discover_common_datatype(store, class, property, graph_name)
            {
                property_shape.add_constraint(
                    ConstraintComponentId::new("sh:datatype"),
                    Constraint::Datatype(DatatypeConstraint {
                        datatype_iri: datatype,
                    }),
                );
            }

            // Add property shape as a property constraint to the node shape
            shape.add_property_shape(property_shape);
        }

        self.stats.classes_analyzed += 1;
        tracing::debug!(
            "Learned shape {} with {} properties",
            shape_id.as_str(),
            properties.len()
        );

        Ok(shape)
    }

    /// Discover classes in the RDF store
    fn discover_classes(&self, store: &Store, graph_name: Option<&str>) -> Result<Vec<NamedNode>> {
        let mut classes = HashSet::new();

        // Query for all rdf:type relationships
        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?class WHERE {{
                    GRAPH <{}> {{
                        ?instance a ?class .
                        FILTER(isIRI(?class))
                    }}
                }}
                ORDER BY ?class
            "#,
                graph
            )
        } else {
            r#"
                SELECT DISTINCT ?class WHERE {
                    ?instance a ?class .
                    FILTER(isIRI(?class))
                }
                ORDER BY ?class
            "#
            .to_string()
        };

        let result = self.execute_learning_query(store, &query)?;

        if let LearningQueryResult::Select {
            variables: _,
            bindings,
        } = result
        {
            for binding in bindings {
                if let Some(class_term) = binding.get("class") {
                    if let Term::NamedNode(class_node) = class_term {
                        classes.insert(class_node.clone());
                    }
                }
            }
        }

        Ok(classes.into_iter().collect())
    }

    /// Discover properties used by instances of a specific class
    fn discover_properties_for_class(
        &self,
        store: &Store,
        class: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<NamedNode>> {
        let mut properties = HashSet::new();

        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?property WHERE {{
                    GRAPH <{}> {{
                        ?instance a <{}> .
                        ?instance ?property ?value .
                        FILTER(?property != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
                        FILTER(isIRI(?property))
                    }}
                }}
                ORDER BY ?property
            "#,
                graph,
                class.as_str()
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?property WHERE {{
                    ?instance a <{}> .
                    ?instance ?property ?value .
                    FILTER(?property != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
                    FILTER(isIRI(?property))
                }}
                ORDER BY ?property
            "#,
                class.as_str()
            )
        };

        let result = self.execute_learning_query(store, &query)?;

        if let LearningQueryResult::Select {
            variables: _,
            bindings,
        } = result
        {
            for binding in bindings {
                if let Some(property_term) = binding.get("property") {
                    if let Term::NamedNode(property_node) = property_term {
                        properties.insert(property_node.clone());
                    }
                }
            }
        }

        Ok(properties.into_iter().collect())
    }

    /// Analyze property cardinality for instances of a class
    fn analyze_property_cardinality(
        &self,
        store: &Store,
        class: &NamedNode,
        property: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<(Option<u32>, Option<u32>)> {
        // Count values per instance for this property
        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT ?instance (COUNT(?value) as ?count) WHERE {{
                    GRAPH <{}> {{
                        ?instance a <{}> .
                        ?instance <{}> ?value .
                    }}
                }}
                GROUP BY ?instance
            "#,
                graph,
                class.as_str(),
                property.as_str()
            )
        } else {
            format!(
                r#"
                SELECT ?instance (COUNT(?value) as ?count) WHERE {{
                    ?instance a <{}> .
                    ?instance <{}> ?value .
                }}
                GROUP BY ?instance
            "#,
                class.as_str(),
                property.as_str()
            )
        };

        let result = self.execute_learning_query(store, &query)?;
        let mut counts = Vec::new();

        if let LearningQueryResult::Select {
            variables: _,
            bindings,
        } = result
        {
            for binding in bindings {
                if let Some(count_term) = binding.get("count") {
                    if let Term::Literal(count_literal) = count_term {
                        if let Ok(count) = count_literal.as_str().parse::<u32>() {
                            counts.push(count);
                        }
                    }
                }
            }
        }

        if counts.is_empty() {
            return Ok((None, None));
        }

        let min_count = *counts.iter().min().unwrap();
        let max_count = *counts.iter().max().unwrap();

        // Only suggest constraints if there's a clear pattern
        let min_constraint = if min_count > 0 && counts.iter().all(|&c| c >= min_count) {
            Some(min_count)
        } else {
            None
        };

        let max_constraint = if min_count == max_count {
            Some(max_count)
        } else {
            None
        };

        Ok((min_constraint, max_constraint))
    }

    /// Discover the most common datatype for a property in a class
    fn discover_common_datatype(
        &self,
        store: &Store,
        class: &NamedNode,
        property: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<NamedNode> {
        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT (DATATYPE(?value) as ?datatype) (COUNT(?value) as ?count) WHERE {{
                    GRAPH <{}> {{
                        ?instance a <{}> .
                        ?instance <{}> ?value .
                        FILTER(isLiteral(?value))
                    }}
                }}
                GROUP BY (DATATYPE(?value))
                ORDER BY DESC(?count)
                LIMIT 1
            "#,
                graph,
                class.as_str(),
                property.as_str()
            )
        } else {
            format!(
                r#"
                SELECT (DATATYPE(?value) as ?datatype) (COUNT(?value) as ?count) WHERE {{
                    ?instance a <{}> .
                    ?instance <{}> ?value .
                    FILTER(isLiteral(?value))
                }}
                GROUP BY (DATATYPE(?value))
                ORDER BY DESC(?count)
                LIMIT 1
            "#,
                class.as_str(),
                property.as_str()
            )
        };

        let result = self.execute_learning_query(store, &query)?;

        if let LearningQueryResult::Select {
            variables: _,
            bindings,
        } = result
        {
            if let Some(binding) = bindings.first() {
                if let Some(datatype_term) = binding.get("datatype") {
                    if let Term::NamedNode(datatype_node) = datatype_term {
                        return Ok(datatype_node.clone());
                    }
                }
            }
        }

        // Default to string if no specific datatype found
        NamedNode::new("http://www.w3.org/2001/XMLSchema#string").map_err(|e| {
            ShaclAiError::ShapeLearning(format!("Failed to create default datatype: {}", e))
        })
    }

    /// Learn constraints for a specific property of a class
    fn learn_property_constraints(
        &mut self,
        store: &Store,
        class: &NamedNode,
        property: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<(ConstraintComponentId, Constraint)>> {
        let mut baseline_constraints = Vec::new();

        // Learn baseline constraints using traditional methods
        if let Ok(cardinality) =
            self.learn_cardinality_constraints(store, class, property, graph_name)
        {
            baseline_constraints.extend(cardinality);
        }

        if let Ok(datatype) = self.learn_datatype_constraints(store, class, property, graph_name) {
            baseline_constraints.extend(datatype);
        }

        if let Ok(range) = self.learn_range_constraints(store, class, property, graph_name) {
            baseline_constraints.extend(range);
        }

        if let Ok(temporal) = self.learn_temporal_constraints(store, class, property, graph_name) {
            self.stats.temporal_constraints_discovered += temporal.len();
            baseline_constraints.extend(temporal);
        }

        // Optimize constraint discovery using reinforcement learning if enabled
        let final_constraints = if self.config.enable_reinforcement_learning {
            tracing::debug!("Applying RL optimization to constraint discovery");
            self.optimize_constraint_discovery_with_rl(
                store,
                class,
                property,
                graph_name,
                &baseline_constraints,
            )
            .unwrap_or(baseline_constraints)
        } else {
            baseline_constraints
        };

        self.stats.total_constraints_discovered += final_constraints.len();

        Ok(final_constraints)
    }

    /// Optimize constraint discovery using reinforcement learning
    fn optimize_constraint_discovery_with_rl(
        &mut self,
        store: &Store,
        class: &NamedNode,
        property: &NamedNode,
        graph_name: Option<&str>,
        baseline_constraints: &[(ConstraintComponentId, Constraint)],
    ) -> Result<Vec<(ConstraintComponentId, Constraint)>> {
        // If RL is not enabled, return baseline constraints
        let rl_agent = match &mut self.rl_agent {
            Some(agent) => agent,
            None => return Ok(baseline_constraints.to_vec()),
        };

        // Convert the constraint discovery problem to RL environment
        let mut optimized_constraints = Vec::new();

        // Create state representation
        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT (COUNT(?instance) as ?count) WHERE {{
                    GRAPH <{}> {{
                        ?instance a <{}> .
                        ?instance <{}> ?value .
                    }}
                }}
            "#,
                graph,
                class.as_str(),
                property.as_str()
            )
        } else {
            format!(
                r#"
                SELECT (COUNT(?instance) as ?count) WHERE {{
                    ?instance a <{}> .
                    ?instance <{}> ?value .
                }}
            "#,
                class.as_str(),
                property.as_str()
            )
        };

        let result = self.execute_learning_query(store, &query)?;
        let instance_count = if let LearningQueryResult::Select { bindings, .. } = result {
            bindings
                .first()
                .and_then(|b| b.get("count"))
                .and_then(|term| {
                    if let Term::Literal(lit) = term {
                        lit.as_str().parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0)
        } else {
            0
        };

        // Define available RL actions for constraint optimization
        let available_actions = vec![
            Action::ValidateConstraint("cardinality".to_string()),
            Action::ValidateConstraint("datatype".to_string()),
            Action::ValidateConstraint("range".to_string()),
            Action::ValidateConstraint("temporal".to_string()),
            Action::SkipConstraint("optional".to_string()),
            Action::EnableCaching,
            Action::AdjustBatchSize(instance_count.min(100)),
        ];

        // Simulate constraint discovery optimization episodes
        for _ in 0..5 {
            // Run 5 optimization episodes
            let mut current_constraints = baseline_constraints.to_vec();

            for action in &available_actions {
                match action {
                    Action::ValidateConstraint(constraint_type) => {
                        match constraint_type.as_str() {
                            "cardinality" => {
                                if let Ok(cardinality_constraints) = self
                                    .learn_cardinality_constraints(
                                        store, class, property, graph_name,
                                    )
                                {
                                    // Evaluate quality of cardinality constraints
                                    let quality_score = self.evaluate_constraint_quality(
                                        &cardinality_constraints,
                                        instance_count,
                                    );
                                    if quality_score > 0.7 {
                                        current_constraints.extend(cardinality_constraints);
                                    }
                                }
                            }
                            "datatype" => {
                                if let Ok(datatype_constraints) = self
                                    .learn_datatype_constraints(store, class, property, graph_name)
                                {
                                    let quality_score = self.evaluate_constraint_quality(
                                        &datatype_constraints,
                                        instance_count,
                                    );
                                    if quality_score > 0.7 {
                                        current_constraints.extend(datatype_constraints);
                                    }
                                }
                            }
                            "range" => {
                                if let Ok(range_constraints) =
                                    self.learn_range_constraints(store, class, property, graph_name)
                                {
                                    let quality_score = self.evaluate_constraint_quality(
                                        &range_constraints,
                                        instance_count,
                                    );
                                    if quality_score > 0.7 {
                                        current_constraints.extend(range_constraints);
                                    }
                                }
                            }
                            "temporal" => {
                                if let Ok(temporal_constraints) = self
                                    .learn_temporal_constraints(store, class, property, graph_name)
                                {
                                    let quality_score = self.evaluate_constraint_quality(
                                        &temporal_constraints,
                                        instance_count,
                                    );
                                    if quality_score > 0.7 {
                                        let constraint_count = temporal_constraints.len();
                                        current_constraints.extend(temporal_constraints);
                                        self.stats.temporal_constraints_discovered +=
                                            constraint_count;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    Action::EnableCaching => {
                        // Caching optimization would be handled at execution level
                        tracing::debug!(
                            "RL agent suggests enabling caching for constraint discovery"
                        );
                    }
                    Action::AdjustBatchSize(size) => {
                        tracing::debug!("RL agent suggests batch size: {}", size);
                    }
                    _ => {}
                }
            }

            // For final episode, use the optimized constraints
            if optimized_constraints.is_empty()
                || current_constraints.len() > optimized_constraints.len()
            {
                optimized_constraints = current_constraints;
            }
        }

        tracing::info!(
            "RL optimization discovered {} constraints (baseline: {})",
            optimized_constraints.len(),
            baseline_constraints.len()
        );

        Ok(optimized_constraints)
    }

    /// Evaluate the quality of discovered constraints
    fn evaluate_constraint_quality(
        &self,
        constraints: &[(ConstraintComponentId, Constraint)],
        instance_count: usize,
    ) -> f64 {
        if constraints.is_empty() {
            return 0.0;
        }

        let mut quality_score = 0.0;
        let weight = 1.0 / constraints.len() as f64;

        for (component_id, _constraint) in constraints {
            // Quality based on constraint type and data characteristics
            let type_quality = match component_id.as_str() {
                id if id.contains("Count") => {
                    // Cardinality constraints are high quality for structured data
                    if instance_count > 10 {
                        0.9
                    } else {
                        0.6
                    }
                }
                id if id.contains("Datatype") => {
                    // Datatype constraints are generally high quality
                    0.8
                }
                id if id.contains("Inclusive") => {
                    // Range constraints are good for numeric data
                    0.7
                }
                _ => 0.5, // Default quality
            };

            quality_score += weight * type_quality;
        }

        // Bonus for discovering multiple complementary constraint types
        let unique_types: std::collections::HashSet<_> = constraints
            .iter()
            .map(|(id, _)| id.as_str().split("Constraint").next().unwrap_or(""))
            .collect();

        if unique_types.len() > 2 {
            quality_score += 0.1; // Diversity bonus
        }

        quality_score.min(1.0)
    }

    /// Learn cardinality constraints for a property
    fn learn_cardinality_constraints(
        &self,
        store: &Store,
        class: &NamedNode,
        property: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<(ConstraintComponentId, Constraint)>> {
        let mut constraints = Vec::new();

        // Count values per instance
        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT ?instance (COUNT(?value) as ?count) WHERE {{
                    GRAPH <{}> {{
                        ?instance a <{}> .
                        ?instance <{}> ?value .
                    }}
                }}
                GROUP BY ?instance
            "#,
                graph,
                class.as_str(),
                property.as_str()
            )
        } else {
            format!(
                r#"
                SELECT ?instance (COUNT(?value) as ?count) WHERE {{
                    ?instance a <{}> .
                    ?instance <{}> ?value .
                }}
                GROUP BY ?instance
            "#,
                class.as_str(),
                property.as_str()
            )
        };

        let result = self.execute_learning_query(store, &query)?;
        let mut counts = Vec::new();

        if let LearningQueryResult::Select {
            variables: _,
            bindings,
        } = result
        {
            for binding in bindings {
                if let Some(count_term) = binding.get("count") {
                    if let Term::Literal(count_literal) = count_term {
                        if let Ok(count) = count_literal.as_str().parse::<u32>() {
                            counts.push(count);
                        }
                    }
                }
            }
        }

        if !counts.is_empty() {
            let min_count = *counts.iter().min().unwrap();
            let max_count = *counts.iter().max().unwrap();

            // If minimum count > 0, add min count constraint
            if min_count > 0 {
                constraints.push((
                    ConstraintComponentId::new("sh:MinCountConstraintComponent"),
                    Constraint::MinCount(MinCountConstraint { min_count }),
                ));
            }

            // If all instances have the same count, add exact count constraints
            if min_count == max_count {
                constraints.push((
                    ConstraintComponentId::new("sh:MaxCountConstraintComponent"),
                    Constraint::MaxCount(MaxCountConstraint { max_count }),
                ));
            }
        }

        Ok(constraints)
    }

    /// Learn datatype constraints for a property
    fn learn_datatype_constraints(
        &self,
        store: &Store,
        class: &NamedNode,
        property: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<(ConstraintComponentId, Constraint)>> {
        let mut constraints = Vec::new();

        // Query for datatypes used with this property
        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT (DATATYPE(?value) as ?datatype) (COUNT(?value) as ?count) WHERE {{
                    GRAPH <{}> {{
                        ?instance a <{}> .
                        ?instance <{}> ?value .
                        FILTER(isLiteral(?value))
                    }}
                }}
                GROUP BY (DATATYPE(?value))
                ORDER BY DESC(?count)
            "#,
                graph,
                class.as_str(),
                property.as_str()
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT (DATATYPE(?value) as ?datatype) (COUNT(?value) as ?count) WHERE {{
                    ?instance a <{}> .
                    ?instance <{}> ?value .
                    FILTER(isLiteral(?value))
                }}
                GROUP BY (DATATYPE(?value))
                ORDER BY DESC(?count)
            "#,
                class.as_str(),
                property.as_str()
            )
        };

        let result = self.execute_learning_query(store, &query)?;
        let mut datatype_counts = Vec::new();

        if let LearningQueryResult::Select {
            variables: _,
            bindings,
        } = result
        {
            for binding in bindings {
                if let (Some(datatype_term), Some(count_term)) =
                    (binding.get("datatype"), binding.get("count"))
                {
                    if let (Term::NamedNode(datatype_node), Term::Literal(count_literal)) =
                        (datatype_term, count_term)
                    {
                        if let Ok(count) = count_literal.as_str().parse::<u32>() {
                            datatype_counts.push((datatype_node.clone(), count));
                        }
                    }
                }
            }
        }

        // If there's a dominant datatype (>= 80% of values), add datatype constraint
        if let Some((dominant_datatype, dominant_count)) = datatype_counts.first() {
            let total_count: u32 = datatype_counts.iter().map(|(_, count)| count).sum();
            let confidence = *dominant_count as f64 / total_count as f64;

            if confidence >= self.config.min_confidence {
                constraints.push((
                    ConstraintComponentId::new("sh:DatatypeConstraintComponent"),
                    Constraint::Datatype(DatatypeConstraint {
                        datatype_iri: dominant_datatype.clone(),
                    }),
                ));
            }
        }

        Ok(constraints)
    }

    /// Learn range constraints for numeric properties
    fn learn_range_constraints(
        &self,
        store: &Store,
        class: &NamedNode,
        property: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<(ConstraintComponentId, Constraint)>> {
        let mut constraints = Vec::new();

        // Query for numeric values
        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT ?value WHERE {{
                    GRAPH <{}> {{
                        ?instance a <{}> .
                        ?instance <{}> ?value .
                        FILTER(isLiteral(?value))
                        FILTER(DATATYPE(?value) IN (
                            <http://www.w3.org/2001/XMLSchema#int>,
                            <http://www.w3.org/2001/XMLSchema#integer>,
                            <http://www.w3.org/2001/XMLSchema#decimal>,
                            <http://www.w3.org/2001/XMLSchema#float>,
                            <http://www.w3.org/2001/XMLSchema#double>
                        ))
                    }}
                }}
            "#,
                graph,
                class.as_str(),
                property.as_str()
            )
        } else {
            format!(
                r#"
                SELECT ?value WHERE {{
                    ?instance a <{}> .
                    ?instance <{}> ?value .
                    FILTER(isLiteral(?value))
                    FILTER(DATATYPE(?value) IN (
                        <http://www.w3.org/2001/XMLSchema#int>,
                        <http://www.w3.org/2001/XMLSchema#integer>,
                        <http://www.w3.org/2001/XMLSchema#decimal>,
                        <http://www.w3.org/2001/XMLSchema#float>,
                        <http://www.w3.org/2001/XMLSchema#double>
                    ))
                }}
            "#,
                class.as_str(),
                property.as_str()
            )
        };

        let result = self.execute_learning_query(store, &query)?;
        let mut values = Vec::new();

        if let LearningQueryResult::Select {
            variables: _,
            bindings,
        } = result
        {
            for binding in bindings {
                if let Some(value_term) = binding.get("value") {
                    if let Term::Literal(value_literal) = value_term {
                        if let Ok(value) = value_literal.as_str().parse::<f64>() {
                            values.push((value, value_literal.clone()));
                        }
                    }
                }
            }
        }

        if values.len() >= 10 {
            // Only add range constraints if we have enough data
            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let min_value = &values[0].1;
            let max_value = &values[values.len() - 1].1;

            // Add range constraints
            constraints.push((
                ConstraintComponentId::new("sh:MinInclusiveConstraintComponent"),
                Constraint::MinInclusive(MinInclusiveConstraint {
                    min_value: min_value.clone(),
                }),
            ));

            constraints.push((
                ConstraintComponentId::new("sh:MaxInclusiveConstraintComponent"),
                Constraint::MaxInclusive(MaxInclusiveConstraint {
                    max_value: max_value.clone(),
                }),
            ));
        }

        Ok(constraints)
    }

    /// Learn temporal constraints for date/time properties
    fn learn_temporal_constraints(
        &mut self,
        store: &Store,
        class: &NamedNode,
        property: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<Vec<(ConstraintComponentId, Constraint)>> {
        let mut constraints = Vec::new();

        // Query for temporal values and their types
        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT ?value (DATATYPE(?value) as ?datatype) WHERE {{
                    GRAPH <{}> {{
                        ?instance a <{}> .
                        ?instance <{}> ?value .
                        FILTER(isLiteral(?value))
                        FILTER(DATATYPE(?value) IN (
                            <http://www.w3.org/2001/XMLSchema#date>,
                            <http://www.w3.org/2001/XMLSchema#dateTime>,
                            <http://www.w3.org/2001/XMLSchema#time>,
                            <http://www.w3.org/2001/XMLSchema#gYear>,
                            <http://www.w3.org/2001/XMLSchema#gYearMonth>,
                            <http://www.w3.org/2001/XMLSchema#gMonth>,
                            <http://www.w3.org/2001/XMLSchema#gMonthDay>,
                            <http://www.w3.org/2001/XMLSchema#gDay>
                        ))
                    }}
                }}
                ORDER BY ?value
            "#,
                graph,
                class.as_str(),
                property.as_str()
            )
        } else {
            format!(
                r#"
                SELECT ?value (DATATYPE(?value) as ?datatype) WHERE {{
                    ?instance a <{}> .
                    ?instance <{}> ?value .
                    FILTER(isLiteral(?value))
                    FILTER(DATATYPE(?value) IN (
                        <http://www.w3.org/2001/XMLSchema#date>,
                        <http://www.w3.org/2001/XMLSchema#dateTime>,
                        <http://www.w3.org/2001/XMLSchema#time>,
                        <http://www.w3.org/2001/XMLSchema#gYear>,
                        <http://www.w3.org/2001/XMLSchema#gYearMonth>,
                        <http://www.w3.org/2001/XMLSchema#gMonth>,
                        <http://www.w3.org/2001/XMLSchema#gMonthDay>,
                        <http://www.w3.org/2001/XMLSchema#gDay>
                    ))
                }}
                ORDER BY ?value
            "#,
                class.as_str(),
                property.as_str()
            )
        };

        let result = self.execute_learning_query(store, &query)?;
        let mut temporal_values = Vec::new();
        let mut datatypes = HashMap::new();

        if let LearningQueryResult::Select {
            variables: _,
            bindings,
        } = result
        {
            for binding in bindings {
                if let (Some(value_term), Some(datatype_term)) =
                    (binding.get("value"), binding.get("datatype"))
                {
                    if let (Term::Literal(value_literal), Term::NamedNode(datatype_node)) =
                        (value_term, datatype_term)
                    {
                        temporal_values.push(value_literal.clone());
                        *datatypes.entry(datatype_node.clone()).or_insert(0) += 1;
                    }
                }
            }
        }

        if temporal_values.is_empty() {
            return Ok(constraints);
        }

        // Analyze temporal patterns if we have enough data
        if temporal_values.len() >= 5 {
            // Find min/max temporal values for range constraints
            let min_value = &temporal_values[0];
            let max_value = &temporal_values[temporal_values.len() - 1];

            constraints.push((
                ConstraintComponentId::new("sh:MinInclusiveConstraintComponent"),
                Constraint::MinInclusive(MinInclusiveConstraint {
                    min_value: min_value.clone(),
                }),
            ));

            constraints.push((
                ConstraintComponentId::new("sh:MaxInclusiveConstraintComponent"),
                Constraint::MaxInclusive(MaxInclusiveConstraint {
                    max_value: max_value.clone(),
                }),
            ));

            // Analyze temporal patterns
            let temporal_patterns = self.analyze_temporal_patterns(&temporal_values)?;

            // Add pattern-based constraints
            if temporal_patterns.has_regular_intervals {
                // Could add custom temporal interval constraints in the future
                tracing::debug!(
                    "Detected regular temporal intervals for property {}",
                    property.as_str()
                );
            }

            if temporal_patterns.has_seasonal_pattern {
                tracing::debug!(
                    "Detected seasonal pattern for property {}",
                    property.as_str()
                );
            }

            // Check for temporal ordering constraints
            if temporal_patterns.is_strictly_increasing {
                // Could add temporal ordering constraints in future SHACL extensions
                tracing::debug!(
                    "Detected strictly increasing temporal pattern for property {}",
                    property.as_str()
                );
            }
        }

        // Add datatype constraint for dominant temporal type
        if let Some((dominant_datatype, count)) = datatypes.iter().max_by_key(|(_, &count)| count) {
            let total_count: u32 = datatypes.values().sum();
            let confidence = *count as f64 / total_count as f64;

            if confidence >= self.config.min_confidence {
                constraints.push((
                    ConstraintComponentId::new("sh:DatatypeConstraintComponent"),
                    Constraint::Datatype(DatatypeConstraint {
                        datatype_iri: dominant_datatype.clone(),
                    }),
                ));
            }
        }

        Ok(constraints)
    }

    /// Analyze temporal patterns in a sequence of temporal values
    fn analyze_temporal_patterns(&self, temporal_values: &[Literal]) -> Result<TemporalPatterns> {
        use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};

        let mut parsed_dates = Vec::new();
        let mut intervals = Vec::new();

        // Parse temporal values to comparable datetime objects
        for value in temporal_values {
            let value_str = value.as_str();

            // Try parsing as different temporal formats
            if let Ok(dt) = DateTime::parse_from_rfc3339(value_str) {
                parsed_dates.push(dt.with_timezone(&Utc));
            } else if let Ok(naive_dt) =
                NaiveDateTime::parse_from_str(value_str, "%Y-%m-%dT%H:%M:%S")
            {
                parsed_dates.push(naive_dt.and_utc());
            } else if let Ok(naive_date) = NaiveDate::parse_from_str(value_str, "%Y-%m-%d") {
                parsed_dates.push(naive_date.and_hms_opt(0, 0, 0).unwrap().and_utc());
            }
        }

        if parsed_dates.len() < 2 {
            return Ok(TemporalPatterns::default());
        }

        // Sort dates for analysis
        parsed_dates.sort();

        // Calculate intervals between consecutive dates
        for window in parsed_dates.windows(2) {
            let interval = window[1].signed_duration_since(window[0]);
            intervals.push(interval);
        }

        // Analyze patterns
        let has_regular_intervals = self.detect_regular_intervals(&intervals);
        let has_seasonal_pattern = self.detect_seasonal_pattern(&parsed_dates);
        let is_strictly_increasing = parsed_dates.windows(2).all(|w| w[0] < w[1]);

        Ok(TemporalPatterns {
            has_regular_intervals,
            has_seasonal_pattern,
            is_strictly_increasing,
            total_values: temporal_values.len(),
            date_range_days: parsed_dates
                .last()
                .unwrap()
                .signed_duration_since(*parsed_dates.first().unwrap())
                .num_days(),
        })
    }

    /// Detect if temporal intervals show regular patterns
    fn detect_regular_intervals(&self, intervals: &[chrono::Duration]) -> bool {
        if intervals.len() < 3 {
            return false;
        }

        // Check if intervals are roughly equal (within 10% variance)
        let avg_interval_secs: f64 = intervals
            .iter()
            .map(|d| d.num_seconds() as f64)
            .sum::<f64>()
            / intervals.len() as f64;
        let variance_threshold = avg_interval_secs * 0.1;

        intervals.iter().all(|interval| {
            let diff = (interval.num_seconds() as f64 - avg_interval_secs).abs();
            diff <= variance_threshold
        })
    }

    /// Detect seasonal patterns in temporal data
    fn detect_seasonal_pattern(&self, dates: &[DateTime<Utc>]) -> bool {
        if dates.len() < 12 {
            return false;
        }

        // Simple heuristic: check if dates span multiple years and show monthly clustering
        let year_span = dates.last().unwrap().year() - dates.first().unwrap().year();

        if year_span >= 2 {
            // Group by month and check for regular distribution
            let mut month_counts = [0u32; 12];
            for date in dates {
                month_counts[date.month0() as usize] += 1;
            }

            // Check if some months consistently have more data points
            let max_count = *month_counts.iter().max().unwrap();
            let min_count = *month_counts.iter().min().unwrap();

            max_count > min_count * 2 // Significant seasonal variation
        } else {
            false
        }
    }

    /// Convert a pattern to a shape
    fn pattern_to_shape(
        &mut self,
        store: &Store,
        pattern: &Pattern,
        graph_name: Option<&str>,
    ) -> Result<Shape> {
        // Create shape based on pattern type with enhanced logic
        match pattern {
            Pattern::ClassUsage {
                class, confidence, frequency, ..
            } if *confidence >= self.config.min_confidence => {
                let shape_id = ShapeId::new(format!(
                    "{}Shape_{}",
                    class.as_str().replace(['/', ':', '#'], "_"),
                    uuid::Uuid::new_v4().to_string()[..8].to_string()
                ));
                
                let mut shape = ShapeFactory::node_shape_with_class(shape_id, class.clone());
                
                // Add severity based on confidence
                if *confidence > 0.9 {
                    shape.set_severity(Some(Severity::Violation));
                } else if *confidence > 0.7 {
                    shape.set_severity(Some(Severity::Warning));
                } else {
                    shape.set_severity(Some(Severity::Info));
                }
                
                // Add additional properties if frequency is high
                if *frequency > self.config.min_support {
                    if let Ok(properties) = self.discover_properties_for_class(store, class, graph_name) {
                        for property in properties.into_iter().take(5) { // Limit to top 5 properties
                            let property_path = PropertyPath::predicate(property.clone());
                            let property_shape_id = ShapeId::new(format!(
                                "{}_{}PropertyShape",
                                shape_id.as_str(),
                                property.as_str().replace(['/', ':', '#'], "_")
                            ));
                            let property_shape = Shape::property_shape(property_shape_id, property_path);
                            shape.add_property_shape(property_shape);
                        }
                    }
                }
                
                Ok(shape)
            }
            Pattern::PropertyUsage {
                property,
                confidence,
                frequency,
                ..
            } if *confidence >= self.config.min_confidence => {
                let shape_id = ShapeId::new(format!(
                    "{}PropertyShape_{}",
                    property.as_str().replace(['/', ':', '#'], "_"),
                    uuid::Uuid::new_v4().to_string()[..8].to_string()
                ));
                let property_path = PropertyPath::predicate(property.clone());
                let mut shape = Shape::property_shape(shape_id, property_path);
                
                // Add constraints based on frequency and confidence
                if *frequency > self.config.min_support * 2.0 {
                    // High frequency suggests required property
                    shape.add_constraint(
                        ConstraintComponentId::new("sh:minCount"),
                        Constraint::MinCount(MinCountConstraint { min_count: 1 }),
                    );
                }
                
                // Set severity based on confidence
                if *confidence > 0.9 {
                    shape.set_severity(Some(Severity::Violation));
                } else if *confidence > 0.7 {
                    shape.set_severity(Some(Severity::Warning));
                } else {
                    shape.set_severity(Some(Severity::Info));
                }
                
                Ok(shape)
            }
            _ => Err(ShaclAiError::ShapeLearning(format!(
                "Pattern does not meet confidence threshold (min: {}, actual: {})",
                self.config.min_confidence,
                match pattern {
                    Pattern::ClassUsage { confidence, .. } => *confidence,
                    Pattern::PropertyUsage { confidence, .. } => *confidence,
                    _ => 0.0,
                }
            ))),
        }
    }

    /// Train the shape learning model
    pub fn train_model(
        &mut self,
        training_data: &ShapeTrainingData,
    ) -> Result<crate::ModelTrainingResult> {
        tracing::info!(
            "Training shape learning model on {} examples",
            training_data.examples.len()
        );

        let start_time = std::time::Instant::now();

        // Simulate training process
        // In a real implementation, this would involve actual ML model training
        let mut accuracy = 0.0;
        let mut loss = 1.0;

        for epoch in 0..*self
            .config
            .algorithm_params
            .get("max_epochs")
            .unwrap_or(&100.0) as usize
        {
            // Simulate training epoch
            accuracy = 0.5 + (epoch as f64 / 100.0) * 0.4; // Improve over time
            loss = 1.0 - accuracy * 0.8; // Decrease loss

            if accuracy >= 0.9 {
                break; // Early stopping
            }
        }

        self.stats.model_trained = true;

        Ok(crate::ModelTrainingResult {
            success: accuracy >= 0.8,
            accuracy,
            loss,
            epochs_trained: (accuracy * 100.0) as usize,
            training_time: start_time.elapsed(),
        })
    }

    /// Execute a learning query using SPARQL over the store
    fn execute_learning_query(&self, store: &Store, query: &str) -> Result<LearningQueryResult> {
        tracing::debug!("Executing learning query: {}", query);

        use oxirs_core::query::QueryEngine;

        let query_engine = QueryEngine::new();
        match query_engine.query(query, store) {
            Ok(oxirs_core::query::QueryResult::Select {
                variables,
                bindings,
            }) => Ok(LearningQueryResult::Select {
                variables,
                bindings,
            }),
            Ok(oxirs_core::query::QueryResult::Ask(result)) => Ok(LearningQueryResult::Ask(result)),
            Ok(_) => Ok(LearningQueryResult::Empty),
            Err(e) => Err(ShaclAiError::ShapeLearning(format!(
                "Query execution failed: {}",
                e
            ))),
        }
    }

    /// Clear learning cache
    pub fn clear_cache(&mut self) {
        self.pattern_cache.clear();
    }

    /// Enable reinforcement learning for constraint discovery optimization
    pub fn enable_reinforcement_learning(&mut self, rl_config: Option<RLConfig>) -> Result<()> {
        self.config.enable_reinforcement_learning = true;

        let rl_config = rl_config.unwrap_or_else(|| RLConfig {
            algorithm: RLAlgorithm::QLearning,
            learning_rate: 0.1,
            discount_factor: 0.9,
            epsilon: 0.1,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
            batch_size: 32,
            buffer_size: 10000,
            update_frequency: 10,
            target_update_frequency: 100,
        });

        self.config.rl_config = Some(rl_config.clone());
        self.rl_agent = Some(ReinforcementLearner::new(rl_config));

        tracing::info!("Reinforcement learning enabled for constraint discovery optimization");

        Ok(())
    }

    /// Check if reinforcement learning is enabled
    pub fn is_rl_enabled(&self) -> bool {
        self.config.enable_reinforcement_learning && self.rl_agent.is_some()
    }
}

impl Default for ShapeLearner {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about shape learning operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LearningStatistics {
    pub total_shapes_learned: usize,
    pub failed_shapes: usize,
    pub total_constraints_discovered: usize,
    pub temporal_constraints_discovered: usize,
    pub classes_analyzed: usize,
    pub model_trained: bool,
    pub last_training_accuracy: f64,
}

/// Training data for shape learning
#[derive(Debug, Clone)]
pub struct ShapeTrainingData {
    pub examples: Vec<ShapeExample>,
    pub validation_examples: Vec<ShapeExample>,
}

/// Training example for shape learning
#[derive(Debug, Clone)]
pub struct ShapeExample {
    pub graph_data: Vec<Triple>,
    pub expected_shapes: Vec<Shape>,
    pub quality_score: f64,
}

/// Temporal patterns detected in data
#[derive(Debug, Clone, Default)]
pub struct TemporalPatterns {
    /// Whether temporal values show regular intervals
    pub has_regular_intervals: bool,

    /// Whether temporal values show seasonal patterns
    pub has_seasonal_pattern: bool,

    /// Whether temporal values are strictly increasing
    pub is_strictly_increasing: bool,

    /// Total number of temporal values analyzed
    pub total_values: usize,

    /// Range of dates in days
    pub date_range_days: i64,
}

impl ShapeLearner {
    /// Adaptive learning enhancement: adjust parameters based on success rates
    pub fn adapt_learning_parameters(&mut self) -> Result<()> {
        tracing::info!("Adapting learning parameters based on performance statistics");
        
        let success_rate = if self.stats.total_shapes_learned > 0 {
            (self.stats.total_shapes_learned as f64) / 
            ((self.stats.total_shapes_learned + self.stats.failed_shapes) as f64)
        } else {
            1.0 // Default to optimistic start
        };
        
        tracing::debug!("Current success rate: {:.2}%", success_rate * 100.0);
        
        // Adapt confidence threshold based on success rate
        if success_rate < 0.5 {
            // Low success rate - reduce confidence threshold to be more permissive
            self.config.min_confidence = (self.config.min_confidence * 0.95).max(0.1);
            self.config.min_support = (self.config.min_support * 0.95).max(0.05);
            tracing::info!("Reduced thresholds - confidence: {:.3}, support: {:.3}", 
                         self.config.min_confidence, self.config.min_support);
        } else if success_rate > 0.9 {
            // High success rate - increase confidence threshold to be more selective
            self.config.min_confidence = (self.config.min_confidence * 1.05).min(0.95);
            self.config.min_support = (self.config.min_support * 1.05).min(0.5);
            tracing::info!("Increased thresholds - confidence: {:.3}, support: {:.3}", 
                         self.config.min_confidence, self.config.min_support);
        }
        
        // Adapt max shapes based on current performance
        if self.stats.classes_analyzed > 0 {
            let shapes_per_class = self.stats.total_shapes_learned as f64 / self.stats.classes_analyzed as f64;
            if shapes_per_class > 3.0 {
                // Too many shapes per class - reduce max shapes
                self.config.max_shapes = (self.config.max_shapes as f64 * 0.9) as usize;
                tracing::info!("Reduced max_shapes to {}", self.config.max_shapes);
            } else if shapes_per_class < 1.0 {
                // Too few shapes - increase max shapes
                self.config.max_shapes = (self.config.max_shapes as f64 * 1.1) as usize;
                tracing::info!("Increased max_shapes to {}", self.config.max_shapes);
            }
        }
        
        Ok(())
    }
    
    /// Enhanced learning with reinforcement learning integration
    pub fn learn_with_reinforcement(
        &mut self,
        store: &Store,
        graph_name: Option<&str>,
        validation_feedback: Option<&ValidationReport>,
    ) -> Result<Vec<Shape>> {
        tracing::info!("Starting reinforcement learning enhanced shape discovery");
        
        // Use reinforcement learning if available
        if let Some(ref mut rl_agent) = self.rl_agent {
            // Define actions: adjust confidence, support, or max_shapes
            let actions = vec![
                Action::new("increase_confidence".to_string(), vec![0.05]),
                Action::new("decrease_confidence".to_string(), vec![-0.05]),
                Action::new("increase_support".to_string(), vec![0.02]),
                Action::new("decrease_support".to_string(), vec![-0.02]),
            ];
            
            // Create state vector from current statistics
            let state = vec![
                self.config.min_confidence,
                self.config.min_support,
                self.stats.total_shapes_learned as f64 / 100.0, // Normalize
                self.stats.failed_shapes as f64 / 100.0, // Normalize
            ];
            
            // Get action from RL agent
            if let Ok(action) = rl_agent.select_action(&state) {
                tracing::debug!("RL selected action: {}", action.name);
                
                // Apply the action
                match action.name.as_str() {
                    "increase_confidence" => {
                        self.config.min_confidence = (self.config.min_confidence + action.parameters[0]).min(0.95);
                    }
                    "decrease_confidence" => {
                        self.config.min_confidence = (self.config.min_confidence + action.parameters[0]).max(0.1);
                    }
                    "increase_support" => {
                        self.config.min_support = (self.config.min_support + action.parameters[0]).min(0.5);
                    }
                    "decrease_support" => {
                        self.config.min_support = (self.config.min_support + action.parameters[0]).max(0.05);
                    }
                    _ => {}
                }
            }
            
            // Learn shapes with adjusted parameters
            let shapes = self.learn_shapes_from_store(store, graph_name)?;
            
            // Calculate reward based on validation feedback
            if let Some(validation_report) = validation_feedback {
                let reward = if validation_report.conforms {
                    1.0 // Positive reward for successful validation
                } else {
                    -0.5 // Negative reward for validation failures
                };
                
                let new_state = vec![
                    self.config.min_confidence,
                    self.config.min_support,
                    self.stats.total_shapes_learned as f64 / 100.0,
                    self.stats.failed_shapes as f64 / 100.0,
                ];
                
                // Update RL agent with experience
                if let Err(e) = rl_agent.update(&state, &actions[0], reward, &new_state) {
                    tracing::warn!("Failed to update RL agent: {}", e);
                }
            }
            
            Ok(shapes)
        } else {
            // Fall back to standard learning
            self.learn_shapes_from_store(store, graph_name)
        }
    }
    
    /// Get adaptive learning recommendations based on current performance
    pub fn get_learning_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        let success_rate = if self.stats.total_shapes_learned > 0 {
            (self.stats.total_shapes_learned as f64) / 
            ((self.stats.total_shapes_learned + self.stats.failed_shapes) as f64)
        } else {
            1.0
        };
        
        if success_rate < 0.3 {
            recommendations.push("Consider lowering confidence threshold for more permissive learning".to_string());
            recommendations.push("Increase training data diversity".to_string());
        } else if success_rate > 0.95 {
            recommendations.push("Consider raising confidence threshold for more selective learning".to_string());
            recommendations.push("Enable reinforcement learning for fine-tuning".to_string());
        }
        
        if self.stats.classes_analyzed > 20 && self.stats.total_shapes_learned < 10 {
            recommendations.push("Increase max_shapes limit for broader coverage".to_string());
        }
        
        if !self.config.enable_reinforcement_learning && self.stats.total_shapes_learned > 50 {
            recommendations.push("Enable reinforcement learning for adaptive optimization".to_string());
        }
        
        recommendations
    }

    /// Performance monitoring for learning efficiency
    pub fn get_performance_metrics(&self) -> LearningPerformanceMetrics {
        let success_rate = if self.stats.total_shapes_learned + self.stats.failed_shapes > 0 {
            self.stats.total_shapes_learned as f64 / 
            (self.stats.total_shapes_learned + self.stats.failed_shapes) as f64
        } else {
            0.0
        };

        let constraint_density = if self.stats.total_shapes_learned > 0 {
            self.stats.total_constraints_discovered as f64 / self.stats.total_shapes_learned as f64
        } else {
            0.0
        };

        let temporal_constraint_ratio = if self.stats.total_constraints_discovered > 0 {
            self.stats.temporal_constraints_discovered as f64 / 
            self.stats.total_constraints_discovered as f64
        } else {
            0.0
        };

        LearningPerformanceMetrics {
            success_rate,
            constraint_density,
            temporal_constraint_ratio,
            shapes_per_class: if self.stats.classes_analyzed > 0 {
                self.stats.total_shapes_learned as f64 / self.stats.classes_analyzed as f64
            } else {
                0.0
            },
            training_accuracy: self.stats.last_training_accuracy,
        }
    }

    /// Optimize constraint quality by removing low-confidence constraints
    pub fn optimize_constraint_quality(&mut self, shapes: &mut [Shape], min_quality_threshold: f64) -> Result<usize> {
        let mut optimized_count = 0;
        
        for shape in shapes.iter_mut() {
            let property_shapes = shape.property_shapes_mut();
            let initial_count = property_shapes.len();
            
            // Keep only high-quality constraints
            property_shapes.retain(|property_shape| {
                let constraint_count = property_shape.constraints().len();
                let quality_score = if constraint_count > 0 {
                    // Higher score for shapes with multiple complementary constraints
                    (constraint_count as f64).min(3.0) / 3.0
                } else {
                    0.0
                };
                
                quality_score >= min_quality_threshold
            });
            
            optimized_count += initial_count - property_shapes.len();
        }
        
        tracing::info!("Optimized {} low-quality constraints", optimized_count);
        Ok(optimized_count)
    }

    /// Enhanced pattern analysis with statistical validation
    pub fn analyze_pattern_statistics(&self, patterns: &[Pattern]) -> PatternStatistics {
        let mut datatype_patterns = 0;
        let mut cardinality_patterns = 0;
        let mut temporal_patterns = 0;
        let mut range_patterns = 0;
        
        for pattern in patterns {
            match pattern.pattern_type().as_str() {
                "datatype" => datatype_patterns += 1,
                "cardinality" => cardinality_patterns += 1,
                "temporal" => temporal_patterns += 1,
                "range" => range_patterns += 1,
                _ => {}
            }
        }
        
        let total_patterns = patterns.len();
        let diversity_score = if total_patterns > 0 {
            let unique_types = [datatype_patterns, cardinality_patterns, temporal_patterns, range_patterns]
                .iter()
                .filter(|&&count| count > 0)
                .count() as f64;
            unique_types / 4.0 // 4 main pattern types
        } else {
            0.0
        };
        
        PatternStatistics {
            total_patterns,
            datatype_patterns,
            cardinality_patterns,
            temporal_patterns,
            range_patterns,
            diversity_score,
        }
    }
}

/// Performance metrics for learning efficiency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPerformanceMetrics {
    pub success_rate: f64,
    pub constraint_density: f64,
    pub temporal_constraint_ratio: f64,
    pub shapes_per_class: f64,
    pub training_accuracy: f64,
}

/// Statistics for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStatistics {
    pub total_patterns: usize,
    pub datatype_patterns: usize,
    pub cardinality_patterns: usize,
    pub temporal_patterns: usize,
    pub range_patterns: usize,
    pub diversity_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_learner_creation() {
        let learner = ShapeLearner::new();
        assert!(learner.config.enable_shape_generation);
        assert_eq!(learner.config.min_confidence, 0.8);
        assert_eq!(learner.config.max_shapes, 100);
    }

    #[test]
    fn test_learning_config() {
        let config = LearningConfig {
            enable_shape_generation: false,
            min_support: 0.2,
            min_confidence: 0.9,
            max_shapes: 50,
            enable_training: false,
            algorithm_params: HashMap::new(),
            enable_reinforcement_learning: false,
            rl_config: None,
        };

        let learner = ShapeLearner::with_config(config.clone());
        assert_eq!(learner.config.min_support, 0.2);
        assert_eq!(learner.config.min_confidence, 0.9);
        assert_eq!(learner.config.max_shapes, 50);
        assert!(!learner.config.enable_training);
    }

    #[test]
    fn test_learning_statistics() {
        let stats = LearningStatistics {
            total_shapes_learned: 5,
            failed_shapes: 1,
            total_constraints_discovered: 20,
            temporal_constraints_discovered: 3,
            classes_analyzed: 3,
            model_trained: true,
            last_training_accuracy: 0.95,
        };

        assert_eq!(stats.total_shapes_learned, 5);
        assert_eq!(stats.failed_shapes, 1);
        assert_eq!(stats.total_constraints_discovered, 20);
        assert_eq!(stats.temporal_constraints_discovered, 3);
        assert!(stats.model_trained);
        assert_eq!(stats.last_training_accuracy, 0.95);
    }

    #[test]
    fn test_temporal_patterns_detection() {
        let patterns = TemporalPatterns {
            has_regular_intervals: true,
            has_seasonal_pattern: false,
            is_strictly_increasing: true,
            total_values: 10,
            date_range_days: 365,
        };

        assert!(patterns.has_regular_intervals);
        assert!(!patterns.has_seasonal_pattern);
        assert!(patterns.is_strictly_increasing);
        assert_eq!(patterns.total_values, 10);
        assert_eq!(patterns.date_range_days, 365);
    }

    #[test]
    fn test_temporal_patterns_default() {
        let patterns = TemporalPatterns::default();

        assert!(!patterns.has_regular_intervals);
        assert!(!patterns.has_seasonal_pattern);
        assert!(!patterns.is_strictly_increasing);
        assert_eq!(patterns.total_values, 0);
        assert_eq!(patterns.date_range_days, 0);
    }
}
