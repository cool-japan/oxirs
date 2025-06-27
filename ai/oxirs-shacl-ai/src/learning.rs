//! Shape learning and automatic constraint discovery
//!
//! This module implements AI-powered shape learning from RDF data.

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

use crate::{patterns::Pattern, Result, ShaclAiError};

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
}

impl ShapeLearner {
    /// Create a new shape learner with default configuration
    pub fn new() -> Self {
        Self::with_config(LearningConfig::default())
    }

    /// Create a new shape learner with custom configuration
    pub fn with_config(config: LearningConfig) -> Self {
        Self {
            config,
            pattern_cache: HashMap::new(),
            stats: LearningStatistics::default(),
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
            // For now, just add a simplified constraint
            // TODO: Implement proper property shape inclusion
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
        let mut constraints = Vec::new();

        // Learn cardinality constraints
        if let Ok(cardinality) =
            self.learn_cardinality_constraints(store, class, property, graph_name)
        {
            constraints.extend(cardinality);
        }

        // Learn datatype constraints
        if let Ok(datatype) = self.learn_datatype_constraints(store, class, property, graph_name) {
            constraints.extend(datatype);
        }

        // Learn range constraints
        if let Ok(range) = self.learn_range_constraints(store, class, property, graph_name) {
            constraints.extend(range);
        }

        Ok(constraints)
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

    /// Convert a pattern to a shape
    fn pattern_to_shape(
        &mut self,
        _store: &Store,
        pattern: &Pattern,
        _graph_name: Option<&str>,
    ) -> Result<Shape> {
        // Create shape based on pattern type
        match pattern {
            Pattern::ClassUsage {
                class, confidence, ..
            } if *confidence >= self.config.min_confidence => {
                let shape_id = ShapeId::new(format!("{}Shape", class.as_str()));
                Ok(ShapeFactory::node_shape_with_class(shape_id, class.clone()))
            }
            Pattern::PropertyUsage {
                property,
                confidence,
                ..
            } if *confidence >= self.config.min_confidence => {
                let shape_id = ShapeId::new(format!("{}PropertyShape", property.as_str()));
                let property_path = PropertyPath::predicate(property.clone());
                Ok(Shape::property_shape(shape_id, property_path))
            }
            _ => Err(ShaclAiError::ShapeLearning(
                "Pattern does not meet confidence threshold".to_string(),
            )),
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

    /// Get learning statistics
    pub fn get_statistics(&self) -> &LearningStatistics {
        &self.stats
    }

    /// Clear learning cache
    pub fn clear_cache(&mut self) {
        self.pattern_cache.clear();
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
            classes_analyzed: 3,
            model_trained: true,
            last_training_accuracy: 0.95,
        };

        assert_eq!(stats.total_shapes_learned, 5);
        assert_eq!(stats.failed_shapes, 1);
        assert_eq!(stats.total_constraints_discovered, 20);
        assert!(stats.model_trained);
        assert_eq!(stats.last_training_accuracy, 0.95);
    }
}
