//! Graph exploration module for dynamic knowledge graph navigation
//!
//! This module provides advanced graph exploration capabilities including:
//! - Path discovery between entities
//! - Entity expansion with relationship following
//! - Schema-aware processing and constraint handling
//! - Intelligent graph traversal with scoring

use anyhow::{anyhow, Result};
use oxirs_core::{model::NamedNode, Store};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};

/// Configuration for graph exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationConfig {
    /// Maximum depth for path discovery
    pub max_depth: usize,
    /// Maximum number of paths to explore
    pub max_paths: usize,
    /// Maximum number of neighbors to expand per entity
    pub max_neighbors: usize,
    /// Minimum relevance score for path inclusion
    pub min_relevance_score: f32,
    /// Enable schema-aware filtering
    pub schema_aware: bool,
    /// Preferred relationship types for exploration
    pub preferred_relationships: Vec<String>,
    /// Blacklisted relationship types to avoid
    pub blacklisted_relationships: Vec<String>,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            max_paths: 100,
            max_neighbors: 20,
            min_relevance_score: 0.1,
            schema_aware: true,
            preferred_relationships: vec![
                "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                "http://xmlns.com/foaf/0.1/knows".to_string(),
                "http://purl.org/dc/elements/1.1/creator".to_string(),
            ],
            blacklisted_relationships: vec!["http://www.w3.org/2002/07/owl#sameAs".to_string()],
        }
    }
}

/// A path through the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    /// Entities in the path
    pub entities: Vec<String>,
    /// Relationships connecting the entities
    pub relationships: Vec<String>,
    /// Path length (number of hops)
    pub length: usize,
    /// Relevance score for this path
    pub relevance_score: f32,
    /// Explanation of why this path is relevant
    pub explanation: String,
    /// Metadata about the path
    pub metadata: HashMap<String, String>,
}

/// Information about an expanded entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandedEntity {
    /// The entity URI
    pub entity: String,
    /// Direct neighbors
    pub neighbors: Vec<EntityNeighbor>,
    /// Entity types
    pub types: Vec<String>,
    /// Properties and their values
    pub properties: HashMap<String, Vec<String>>,
    /// Relevance score
    pub relevance_score: f32,
    /// Schema information
    pub schema_info: Option<SchemaInfo>,
}

/// Information about a neighboring entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNeighbor {
    /// Neighbor entity URI
    pub entity: String,
    /// Relationship connecting to this neighbor
    pub relationship: String,
    /// Direction of relationship (outgoing/incoming)
    pub direction: RelationshipDirection,
    /// Strength/weight of this relationship
    pub strength: f32,
    /// Labels or human-readable names
    pub labels: Vec<String>,
}

/// Direction of a relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipDirection {
    Outgoing,
    Incoming,
    Bidirectional,
}

/// Schema information for an entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaInfo {
    /// Classes this entity belongs to
    pub classes: Vec<String>,
    /// Domain restrictions
    pub domain_restrictions: Vec<String>,
    /// Range restrictions  
    pub range_restrictions: Vec<String>,
    /// Cardinality constraints
    pub cardinality_constraints: HashMap<String, (Option<u32>, Option<u32>)>,
    /// Functional properties
    pub functional_properties: Vec<String>,
    /// Equivalent classes
    pub equivalent_classes: Vec<String>,
    /// Disjoint classes
    pub disjoint_classes: Vec<String>,
    /// SHACL shapes
    pub shacl_shapes: Vec<ShaclShape>,
}

/// SHACL shape definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaclShape {
    pub shape_id: String,
    pub target_class: Option<String>,
    pub property_shapes: Vec<PropertyShape>,
    pub constraints: Vec<ShapeConstraint>,
}

/// SHACL property shape
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyShape {
    pub path: String,
    pub datatype: Option<String>,
    pub min_count: Option<u32>,
    pub max_count: Option<u32>,
    pub node_kind: Option<String>,
    pub pattern: Option<String>,
}

/// SHACL constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeConstraint {
    pub constraint_type: String,
    pub value: String,
    pub message: Option<String>,
}

/// Query guidance suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryGuidance {
    pub suggestion_type: GuidanceType,
    pub title: String,
    pub description: String,
    pub sparql_template: String,
    pub confidence: f32,
    pub schema_rationale: String,
}

/// Types of query guidance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GuidanceType {
    ValidPropertyPath,
    TypeConstraint,
    CardinalityAwareness,
    BestPractice,
    ConsistencyCheck,
    SchemaRecommendation,
}

/// Graph exploration engine
pub struct GraphExplorer {
    store: Arc<dyn Store>,
    config: ExplorationConfig,
    schema_cache: HashMap<String, SchemaInfo>,
}

impl GraphExplorer {
    /// Create a new graph explorer
    pub fn new(store: Arc<dyn Store>, config: ExplorationConfig) -> Self {
        Self {
            store,
            config,
            schema_cache: HashMap::new(),
        }
    }

    /// Discover paths between two entities
    pub async fn discover_paths(
        &self,
        start_entity: &str,
        end_entity: &str,
    ) -> Result<Vec<GraphPath>> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Initialize BFS with start entity
        queue.push_back(PathState {
            current_entity: start_entity.to_string(),
            path_entities: vec![start_entity.to_string()],
            path_relationships: Vec::new(),
            depth: 0,
            score: 1.0,
        });

        while let Some(state) = queue.pop_front() {
            if state.depth >= self.config.max_depth {
                continue;
            }

            let state_key = format!("{}:{}", state.current_entity, state.depth);
            if visited.contains(&state_key) {
                continue;
            }
            visited.insert(state_key);

            // Check if we've reached the target
            if state.current_entity == end_entity && state.depth > 0 {
                paths.push(GraphPath {
                    entities: state.path_entities.clone(),
                    relationships: state.path_relationships.clone(),
                    length: state.depth,
                    relevance_score: state.score,
                    explanation: self.generate_path_explanation(&state)?,
                    metadata: HashMap::new(),
                });

                if paths.len() >= self.config.max_paths {
                    break;
                }
                continue;
            }

            // Expand current entity
            let neighbors = self.get_entity_neighbors(&state.current_entity).await?;

            for neighbor in neighbors {
                if state.path_entities.contains(&neighbor.entity) {
                    continue; // Avoid cycles
                }

                if self.is_relationship_blacklisted(&neighbor.relationship) {
                    continue;
                }

                let new_score = state.score * self.calculate_relationship_weight(&neighbor);

                if new_score < self.config.min_relevance_score {
                    continue;
                }

                let mut new_path_entities = state.path_entities.clone();
                new_path_entities.push(neighbor.entity.clone());

                let mut new_path_relationships = state.path_relationships.clone();
                new_path_relationships.push(neighbor.relationship.clone());

                queue.push_back(PathState {
                    current_entity: neighbor.entity,
                    path_entities: new_path_entities,
                    path_relationships: new_path_relationships,
                    depth: state.depth + 1,
                    score: new_score,
                });
            }
        }

        // Sort paths by relevance score
        paths.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());

        Ok(paths)
    }

    /// Expand an entity to get its neighborhood
    pub async fn expand_entity(&self, entity: &str) -> Result<ExpandedEntity> {
        let neighbors = self.get_entity_neighbors(entity).await?;
        let types = self.get_entity_types(entity).await?;
        let properties = self.get_entity_properties(entity).await?;
        let schema_info = if self.config.schema_aware {
            Some(self.get_schema_info(entity).await?)
        } else {
            None
        };

        let relevance_score = self
            .calculate_entity_relevance(entity, &neighbors, &types)
            .await?;

        Ok(ExpandedEntity {
            entity: entity.to_string(),
            neighbors,
            types,
            properties,
            relevance_score,
            schema_info,
        })
    }

    /// Get schema-aware suggestions for query expansion
    pub async fn get_schema_suggestions(&self, entity: &str) -> Result<Vec<String>> {
        let mut suggestions = Vec::new();

        if let Ok(schema_info) = self.get_schema_info(entity).await {
            // Suggest related classes
            for class in &schema_info.classes {
                suggestions.push(format!("Find all instances of {class}"));
                suggestions.push(format!("Find subclasses of {class}"));
            }

            // Suggest functional properties
            for prop in &schema_info.functional_properties {
                suggestions.push(format!("Find the {prop} of {entity}"));
            }

            // Suggest based on domain/range restrictions
            for domain in &schema_info.domain_restrictions {
                suggestions.push(format!("Find entities in domain {domain}"));
            }
        }

        Ok(suggestions)
    }

    /// Find the shortest path between entities
    pub async fn find_shortest_path(
        &self,
        start_entity: &str,
        end_entity: &str,
    ) -> Result<Option<GraphPath>> {
        let paths = self.discover_paths(start_entity, end_entity).await?;

        // Find the shortest path (minimum length)
        let shortest = paths.iter().min_by_key(|path| path.length);

        Ok(shortest.cloned())
    }

    /// Find entities within a certain distance
    pub async fn find_entities_within_distance(
        &self,
        center_entity: &str,
        max_distance: usize,
    ) -> Result<Vec<String>> {
        let mut entities = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((center_entity.to_string(), 0));

        while let Some((current_entity, distance)) = queue.pop_front() {
            if distance > max_distance {
                continue;
            }

            if visited.contains(&current_entity) {
                continue;
            }
            visited.insert(current_entity.clone());

            entities.insert(current_entity.clone());

            if distance < max_distance {
                let neighbors = self.get_entity_neighbors(&current_entity).await?;
                for neighbor in neighbors {
                    if !visited.contains(&neighbor.entity) {
                        queue.push_back((neighbor.entity, distance + 1));
                    }
                }
            }
        }

        Ok(entities.into_iter().collect())
    }

    /// Find multiple alternative paths between entities
    pub async fn find_multiple_paths(
        &self,
        start_entity: &str,
        end_entity: &str,
        max_paths: usize,
    ) -> Result<Vec<GraphPath>> {
        let mut all_paths = Vec::new();
        let mut visited_path_signatures = HashSet::new();

        // Use a priority queue to explore paths by relevance score
        let mut queue = std::collections::BinaryHeap::new();
        queue.push(PathStateOrdered {
            state: PathState {
                current_entity: start_entity.to_string(),
                path_entities: vec![start_entity.to_string()],
                path_relationships: Vec::new(),
                depth: 0,
                score: 1.0,
            },
        });

        while let Some(PathStateOrdered { state }) = queue.pop() {
            if state.depth >= self.config.max_depth {
                continue;
            }

            // Create a signature for path deduplication
            let path_signature = state.path_entities.join("->");
            if visited_path_signatures.contains(&path_signature) {
                continue;
            }

            if state.current_entity == end_entity && state.depth > 0 {
                visited_path_signatures.insert(path_signature);

                all_paths.push(GraphPath {
                    entities: state.path_entities.clone(),
                    relationships: state.path_relationships.clone(),
                    length: state.depth,
                    relevance_score: state.score,
                    explanation: self.generate_path_explanation(&state)?,
                    metadata: HashMap::new(),
                });

                if all_paths.len() >= max_paths {
                    break;
                }
                continue;
            }

            let neighbors = self.get_entity_neighbors(&state.current_entity).await?;

            for neighbor in neighbors {
                if state.path_entities.contains(&neighbor.entity) {
                    continue;
                }

                if self.is_relationship_blacklisted(&neighbor.relationship) {
                    continue;
                }

                let relationship_weight = self.calculate_relationship_weight(&neighbor);
                let new_score = state.score * relationship_weight;

                if new_score < self.config.min_relevance_score {
                    continue;
                }

                let mut new_path_entities = state.path_entities.clone();
                new_path_entities.push(neighbor.entity.clone());

                let mut new_path_relationships = state.path_relationships.clone();
                new_path_relationships.push(neighbor.relationship.clone());

                queue.push(PathStateOrdered {
                    state: PathState {
                        current_entity: neighbor.entity,
                        path_entities: new_path_entities,
                        path_relationships: new_path_relationships,
                        depth: state.depth + 1,
                        score: new_score,
                    },
                });
            }
        }

        Ok(all_paths)
    }

    /// Calculate relationship strength between entities
    pub async fn calculate_relationship_strength(
        &self,
        entity1: &str,
        entity2: &str,
        relationship: &str,
    ) -> Result<f32> {
        // Calculate strength based on various factors
        let mut strength = 0.5; // Base strength

        // Check if it's a functional property (1-to-1 relationship)
        if self.is_functional_property(relationship).await? {
            strength += 0.3;
        }

        // Check frequency of this relationship type
        let frequency = self.get_relationship_frequency(relationship).await?;
        strength += (1.0 / (frequency as f32 + 1.0)) * 0.2;

        // Check if entities share common types
        let entity1_types = self.get_entity_types(entity1).await?;
        let entity2_types = self.get_entity_types(entity2).await?;
        let common_types_count = entity1_types
            .iter()
            .collect::<HashSet<_>>()
            .intersection(&entity2_types.iter().collect())
            .count();

        if common_types_count > 0 {
            strength += 0.1;
        }

        Ok(strength.min(1.0))
    }

    /// Rank paths by multiple criteria
    pub async fn rank_paths(&self, paths: &mut [GraphPath]) -> Result<()> {
        for path in paths.iter_mut() {
            let mut ranking_score = 0.0;

            // Factor 1: Shorter paths are better
            ranking_score += 1.0 / (path.length as f32 + 1.0) * 0.3;

            // Factor 2: Higher relevance scores are better
            ranking_score += path.relevance_score * 0.4;

            // Factor 3: Paths with preferred relationships are better
            let preferred_relationship_count = path
                .relationships
                .iter()
                .filter(|rel| self.config.preferred_relationships.contains(rel))
                .count();
            ranking_score +=
                (preferred_relationship_count as f32 / path.relationships.len() as f32) * 0.2;

            // Factor 4: Paths through hub entities (high connectivity) are better
            ranking_score += self.calculate_hub_connectivity_bonus(path).await? * 0.1;

            path.relevance_score = ranking_score;
        }

        // Sort by ranking score
        paths.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());

        Ok(())
    }

    /// Generate interactive exploration suggestions
    pub async fn generate_exploration_suggestions(
        &self,
        current_entity: &str,
    ) -> Result<Vec<ExplorationSuggestion>> {
        let mut suggestions = Vec::new();

        // Get neighbors for local exploration
        let neighbors = self.get_entity_neighbors(current_entity).await?;
        let types = self.get_entity_types(current_entity).await?;

        // Suggest exploring neighbors
        for neighbor in neighbors.iter().take(5) {
            suggestions.push(ExplorationSuggestion {
                suggestion_type: SuggestionType::ExploreNeighbor,
                title: format!("Explore {}", self.simplify_uri(&neighbor.entity)),
                description: format!(
                    "Follow {} relationship",
                    self.simplify_uri(&neighbor.relationship)
                ),
                action: ExplorationAction::NavigateToEntity(neighbor.entity.clone()),
                confidence: neighbor.strength,
            });
        }

        // Suggest type-based exploration
        for entity_type in types.iter().take(3) {
            suggestions.push(ExplorationSuggestion {
                suggestion_type: SuggestionType::ExploreType,
                title: format!("Find similar {}", self.simplify_uri(entity_type)),
                description: format!("Find other instances of {}", self.simplify_uri(entity_type)),
                action: ExplorationAction::FindSimilarEntities(entity_type.clone()),
                confidence: 0.8,
            });
        }

        // Suggest path discovery
        suggestions.push(ExplorationSuggestion {
            suggestion_type: SuggestionType::DiscoverPaths,
            title: "Discover connection paths".to_string(),
            description: "Find how this entity connects to others".to_string(),
            action: ExplorationAction::DiscoverConnections(current_entity.to_string()),
            confidence: 0.7,
        });

        // Sort by confidence
        suggestions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(suggestions)
    }

    /// Discover related entities using similarity clustering
    pub async fn discover_related_entities(
        &self,
        entity: &str,
        similarity_threshold: f32,
    ) -> Result<Vec<RelatedEntity>> {
        let mut related_entities = Vec::new();

        let entity_types = self.get_entity_types(entity).await?;
        let entity_properties = self.get_entity_properties(entity).await?;
        let _entity_neighbors = self.get_entity_neighbors(entity).await?;

        // Find entities with similar types
        for entity_type in &entity_types {
            let similar_by_type = self.find_entities_by_type(entity_type).await?;
            for similar_entity in similar_by_type {
                if similar_entity != entity {
                    let similarity = self
                        .calculate_entity_similarity(entity, &similar_entity)
                        .await?;
                    if similarity >= similarity_threshold {
                        related_entities.push(RelatedEntity {
                            entity: similar_entity,
                            similarity_score: similarity,
                            relationship_type: "type_similarity".to_string(),
                            explanation: format!("Similar {} type", self.simplify_uri(entity_type)),
                        });
                    }
                }
            }
        }

        // Find entities with similar properties
        for property in entity_properties.keys() {
            let similar_by_property = self.find_entities_with_property(property).await?;
            for similar_entity in similar_by_property {
                if similar_entity != entity {
                    let similarity = self
                        .calculate_property_similarity(entity, &similar_entity, property)
                        .await?;
                    if similarity >= similarity_threshold {
                        related_entities.push(RelatedEntity {
                            entity: similar_entity,
                            similarity_score: similarity,
                            relationship_type: "property_similarity".to_string(),
                            explanation: format!("Shares {} property", self.simplify_uri(property)),
                        });
                    }
                }
            }
        }

        // Remove duplicates and sort by similarity
        related_entities
            .sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
        related_entities.dedup_by(|a, b| a.entity == b.entity);

        Ok(related_entities.into_iter().take(20).collect())
    }

    /// Generate schema-based query guidance
    pub async fn generate_query_guidance(
        &self,
        context_entities: &[String],
        intent: &str,
    ) -> Result<Vec<QueryGuidance>> {
        let mut guidance = Vec::new();

        // Analyze intent and context
        for entity in context_entities {
            let schema_info = self.get_schema_info(entity).await?;

            // Generate valid property path suggestions
            guidance.extend(
                self.generate_property_path_guidance(entity, &schema_info)
                    .await?,
            );

            // Generate type constraint guidance
            guidance.extend(
                self.generate_type_constraint_guidance(entity, &schema_info)
                    .await?,
            );

            // Generate cardinality awareness guidance
            guidance.extend(
                self.generate_cardinality_guidance(entity, &schema_info)
                    .await?,
            );

            // Generate best practice guidance
            guidance.extend(
                self.generate_best_practice_guidance(entity, &schema_info, intent)
                    .await?,
            );

            // Generate consistency check guidance
            guidance.extend(
                self.generate_consistency_guidance(entity, &schema_info)
                    .await?,
            );
        }

        // Sort by confidence and deduplicate
        guidance.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        guidance.dedup_by(|a, b| a.sparql_template == b.sparql_template);

        Ok(guidance.into_iter().take(10).collect())
    }

    /// Validate query against schema constraints
    pub async fn validate_query_against_schema(
        &self,
        query: &str,
    ) -> Result<SchemaValidationResult> {
        let mut validation_result = SchemaValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            suggestions: Vec::new(),
        };

        // Parse query to extract entities and properties (simplified)
        let entities = self.extract_entities_from_query(query).await?;
        let properties = self.extract_properties_from_query(query).await?;

        // Validate entities against known classes
        for entity in &entities {
            if !self.is_valid_entity(entity).await? {
                validation_result
                    .errors
                    .push(format!("Unknown entity: {entity}"));
                validation_result.is_valid = false;
            }
        }

        // Validate properties against domain/range constraints
        for property in &properties {
            let domain_range_valid = self
                .validate_property_domain_range(property, &entities)
                .await?;
            if !domain_range_valid {
                validation_result.warnings.push(format!(
                    "Property {property} may have domain/range mismatch"
                ));
            }
        }

        // Check cardinality constraints
        for entity in &entities {
            let cardinality_issues = self
                .check_cardinality_constraints(entity, &properties)
                .await?;
            validation_result.warnings.extend(cardinality_issues);
        }

        Ok(validation_result)
    }

    /// Generate class hierarchy analysis
    pub async fn analyze_class_hierarchy(&self, class: &str) -> Result<ClassHierarchyAnalysis> {
        let superclasses = self.get_superclasses(class).await?;
        let subclasses = self.get_subclasses(class).await?;
        let equivalent_classes = self.get_equivalent_classes(class).await?;
        let disjoint_classes = self.get_disjoint_classes(class).await?;

        Ok(ClassHierarchyAnalysis {
            class: class.to_string(),
            superclasses,
            subclasses,
            equivalent_classes,
            disjoint_classes,
            depth_from_root: self.calculate_class_depth(class).await?,
            instance_count: self.count_class_instances(class).await?,
        })
    }

    /// Check consistency of entity against SHACL shapes
    pub async fn validate_entity_against_shapes(
        &self,
        entity: &str,
    ) -> Result<ShapeValidationResult> {
        let mut validation_result = ShapeValidationResult {
            is_valid: true,
            violations: Vec::new(),
            satisfied_shapes: Vec::new(),
        };

        let entity_types = self.get_entity_types(entity).await?;
        let entity_properties = self.get_entity_properties(entity).await?;

        // Find applicable SHACL shapes
        for entity_type in &entity_types {
            let shapes = self.get_shapes_for_class(entity_type).await?;

            for shape in shapes {
                let shape_validation = self
                    .validate_against_single_shape(entity, &shape, &entity_properties)
                    .await?;

                if shape_validation.is_valid {
                    validation_result.satisfied_shapes.push(shape.shape_id);
                } else {
                    validation_result.is_valid = false;
                    validation_result
                        .violations
                        .extend(shape_validation.violations);
                }
            }
        }

        Ok(validation_result)
    }

    // Private helper methods for schema-aware processing

    async fn generate_property_path_guidance(
        &self,
        _entity: &str,
        schema_info: &SchemaInfo,
    ) -> Result<Vec<QueryGuidance>> {
        let mut guidance = Vec::new();

        for class in &schema_info.classes {
            guidance.push(QueryGuidance {
                suggestion_type: GuidanceType::ValidPropertyPath,
                title: format!("Valid properties for {}", self.simplify_uri(class)),
                description: "Properties that can be used with this class".to_string(),
                sparql_template: format!(
                    "SELECT ?property WHERE {{ ?instance rdf:type <{class}> . ?instance ?property ?value }}"
                ),
                confidence: 0.9,
                schema_rationale: format!("Based on class definition for {class}"),
            });
        }

        Ok(guidance)
    }

    async fn generate_type_constraint_guidance(
        &self,
        _entity: &str,
        schema_info: &SchemaInfo,
    ) -> Result<Vec<QueryGuidance>> {
        let mut guidance = Vec::new();

        for class in &schema_info.classes {
            guidance.push(QueryGuidance {
                suggestion_type: GuidanceType::TypeConstraint,
                title: format!("Filter by type {}", self.simplify_uri(class)),
                description: "Add type constraint to improve query precision".to_string(),
                sparql_template: format!("?entity rdf:type <{class}> ."),
                confidence: 0.8,
                schema_rationale: format!("Entity belongs to class {class}"),
            });
        }

        Ok(guidance)
    }

    async fn generate_cardinality_guidance(
        &self,
        _entity: &str,
        schema_info: &SchemaInfo,
    ) -> Result<Vec<QueryGuidance>> {
        let mut guidance = Vec::new();

        for (property, (_min_card, max_card)) in &schema_info.cardinality_constraints {
            if let Some(max) = max_card {
                if *max == 1 {
                    guidance.push(QueryGuidance {
                        suggestion_type: GuidanceType::CardinalityAwareness,
                        title: format!("Single-valued property {}", self.simplify_uri(property)),
                        description: "This property has maximum cardinality 1".to_string(),
                        sparql_template: format!("?entity <{property}> ?value"),
                        confidence: 0.85,
                        schema_rationale: "Based on cardinality constraint".to_string(),
                    });
                }
            }
        }

        Ok(guidance)
    }

    async fn generate_best_practice_guidance(
        &self,
        _entity: &str,
        schema_info: &SchemaInfo,
        intent: &str,
    ) -> Result<Vec<QueryGuidance>> {
        let mut guidance = Vec::new();

        if intent.to_lowercase().contains("find") || intent.to_lowercase().contains("get") {
            guidance.push(QueryGuidance {
                suggestion_type: GuidanceType::BestPractice,
                title: "Use LIMIT for better performance".to_string(),
                description: "Add LIMIT clause to prevent large result sets".to_string(),
                sparql_template: "LIMIT 100".to_string(),
                confidence: 0.7,
                schema_rationale: "General best practice for queries".to_string(),
            });
        }

        if !schema_info.functional_properties.is_empty() {
            guidance.push(QueryGuidance {
                suggestion_type: GuidanceType::BestPractice,
                title: "Use functional properties for efficiency".to_string(),
                description: "Functional properties are more efficient for lookups".to_string(),
                sparql_template: format!(
                    "?entity <{}> ?value",
                    schema_info.functional_properties[0]
                ),
                confidence: 0.75,
                schema_rationale: "Functional properties have unique values".to_string(),
            });
        }

        Ok(guidance)
    }

    async fn generate_consistency_guidance(
        &self,
        _entity: &str,
        schema_info: &SchemaInfo,
    ) -> Result<Vec<QueryGuidance>> {
        let mut guidance = Vec::new();

        for disjoint_class in &schema_info.disjoint_classes {
            guidance.push(QueryGuidance {
                suggestion_type: GuidanceType::ConsistencyCheck,
                title: format!("Check disjoint class {}", self.simplify_uri(disjoint_class)),
                description: "Ensure entity doesn't belong to disjoint classes".to_string(),
                sparql_template: format!(
                    "FILTER NOT EXISTS {{ ?entity rdf:type <{disjoint_class}> }}"
                ),
                confidence: 0.8,
                schema_rationale: format!("Class disjointness constraint with {disjoint_class}"),
            });
        }

        Ok(guidance)
    }

    async fn extract_entities_from_query(&self, query: &str) -> Result<Vec<String>> {
        // Simple regex-based extraction of URIs from SPARQL query
        // In practice, this would use a proper SPARQL parser
        let uri_pattern = regex::Regex::new(r"<([^>]+)>").unwrap();
        let entities: Vec<String> = uri_pattern
            .captures_iter(query)
            .map(|cap| cap[1].to_string())
            .collect();
        Ok(entities)
    }

    async fn extract_properties_from_query(&self, query: &str) -> Result<Vec<String>> {
        // Extract property URIs from SPARQL query
        let properties = self.extract_entities_from_query(query).await?;
        // Filter to only include properties (this is simplified)
        Ok(properties
            .into_iter()
            .filter(|uri| uri.contains("property") || uri.contains("#"))
            .collect())
    }

    async fn is_valid_entity(&self, entity: &str) -> Result<bool> {
        // Check if entity exists in the knowledge graph
        let types = self.get_entity_types(entity).await?;
        Ok(!types.is_empty())
    }

    async fn validate_property_domain_range(
        &self,
        _property: &str,
        _entities: &[String],
    ) -> Result<bool> {
        // Check if property usage matches domain/range constraints
        // This would query the schema for property constraints
        Ok(true) // Simplified implementation
    }

    async fn check_cardinality_constraints(
        &self,
        entity: &str,
        properties: &[String],
    ) -> Result<Vec<String>> {
        let mut issues = Vec::new();
        let schema_info = self.get_schema_info(entity).await?;

        for property in properties {
            if let Some((min_card, max_card)) = schema_info.cardinality_constraints.get(property) {
                let actual_count = self.count_property_values(entity, property).await?;

                if let Some(min) = min_card {
                    if actual_count < *min {
                        issues.push(format!(
                            "Property {property} has {actual_count} values, minimum required: {min}"
                        ));
                    }
                }

                if let Some(max) = max_card {
                    if actual_count > *max {
                        issues.push(format!(
                            "Property {property} has {actual_count} values, maximum allowed: {max}"
                        ));
                    }
                }
            }
        }

        Ok(issues)
    }

    async fn get_superclasses(&self, _class: &str) -> Result<Vec<String>> {
        // This would execute: SELECT ?super WHERE { <class> rdfs:subClassOf ?super }
        Ok(vec!["http://www.w3.org/2002/07/owl#Thing".to_string()])
    }

    async fn get_subclasses(&self, _class: &str) -> Result<Vec<String>> {
        // This would execute: SELECT ?sub WHERE { ?sub rdfs:subClassOf <class> }
        Ok(vec![])
    }

    async fn get_equivalent_classes(&self, _class: &str) -> Result<Vec<String>> {
        // This would execute: SELECT ?equiv WHERE { <class> owl:equivalentClass ?equiv }
        Ok(vec![])
    }

    async fn get_disjoint_classes(&self, _class: &str) -> Result<Vec<String>> {
        // This would execute: SELECT ?disjoint WHERE { <class> owl:disjointWith ?disjoint }
        Ok(vec![])
    }

    async fn calculate_class_depth(&self, class: &str) -> Result<u32> {
        // Calculate depth from owl:Thing
        let mut depth = 0;
        let mut current_class = class.to_string();

        while current_class != "http://www.w3.org/2002/07/owl#Thing" {
            let superclasses = self.get_superclasses(&current_class).await?;
            if superclasses.is_empty() {
                break;
            }
            current_class = superclasses[0].clone();
            depth += 1;
            if depth > 20 {
                // Prevent infinite loops
                break;
            }
        }

        Ok(depth)
    }

    async fn count_class_instances(&self, _class: &str) -> Result<u32> {
        // This would execute: SELECT (COUNT(?instance) as ?count) WHERE { ?instance rdf:type <class> }
        Ok(42) // Mock value
    }

    async fn get_shapes_for_class(&self, class: &str) -> Result<Vec<ShaclShape>> {
        // This would query SHACL shapes that target this class
        Ok(vec![ShaclShape {
            shape_id: format!("{class}Shape"),
            target_class: Some(class.to_string()),
            property_shapes: vec![],
            constraints: vec![],
        }])
    }

    async fn validate_against_single_shape(
        &self,
        _entity: &str,
        shape: &ShaclShape,
        properties: &HashMap<String, Vec<String>>,
    ) -> Result<ShapeValidationResult> {
        let mut result = ShapeValidationResult {
            is_valid: true,
            violations: vec![],
            satisfied_shapes: vec![],
        };

        // Validate property shapes
        for prop_shape in &shape.property_shapes {
            if let Some(values) = properties.get(&prop_shape.path) {
                // Check min count
                if let Some(min_count) = prop_shape.min_count {
                    if values.len() < min_count as usize {
                        result.is_valid = false;
                        result.violations.push(ShapeViolation {
                            shape_id: shape.shape_id.clone(),
                            property_path: prop_shape.path.clone(),
                            violation_type: "MinCountConstraint".to_string(),
                            message: format!(
                                "Property {} has {} values, minimum required: {}",
                                prop_shape.path,
                                values.len(),
                                min_count
                            ),
                            severity: ViolationSeverity::Violation,
                        });
                    }
                }

                // Check max count
                if let Some(max_count) = prop_shape.max_count {
                    if values.len() > max_count as usize {
                        result.is_valid = false;
                        result.violations.push(ShapeViolation {
                            shape_id: shape.shape_id.clone(),
                            property_path: prop_shape.path.clone(),
                            violation_type: "MaxCountConstraint".to_string(),
                            message: format!(
                                "Property {} has {} values, maximum allowed: {}",
                                prop_shape.path,
                                values.len(),
                                max_count
                            ),
                            severity: ViolationSeverity::Violation,
                        });
                    }
                }
            }
        }

        Ok(result)
    }

    async fn count_property_values(&self, _entity: &str, _property: &str) -> Result<u32> {
        // This would execute: SELECT (COUNT(?value) as ?count) WHERE { <entity> <property> ?value }
        Ok(1) // Mock value
    }

    async fn is_functional_property(&self, property: &str) -> Result<bool> {
        // Check if property is declared as functional in the ontology
        // This would query the schema/ontology for functional property declarations
        let functional_properties = [
            "http://xmlns.com/foaf/0.1/name",
            "http://purl.org/dc/elements/1.1/title",
        ];
        Ok(functional_properties.contains(&property))
    }

    async fn get_relationship_frequency(&self, relationship: &str) -> Result<u32> {
        // Count how many times this relationship appears in the graph
        // This would execute: SELECT (COUNT(*) as ?count) WHERE { ?s <relationship> ?o }
        // For now, returning a mock value
        match relationship {
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" => Ok(1000),
            "http://www.w3.org/2000/01/rdf-schema#subClassOf" => Ok(100),
            _ => Ok(50),
        }
    }

    async fn calculate_hub_connectivity_bonus(&self, path: &GraphPath) -> Result<f32> {
        let mut total_bonus = 0.0;

        for entity in &path.entities {
            let neighbors = self.get_entity_neighbors(entity).await?;
            let connectivity = neighbors.len() as f32;

            // Entities with more connections get higher bonus
            if connectivity > 10.0 {
                total_bonus += 0.1;
            } else if connectivity > 5.0 {
                total_bonus += 0.05;
            }
        }

        Ok(total_bonus / path.entities.len() as f32)
    }

    async fn find_entities_by_type(&self, entity_type: &str) -> Result<Vec<String>> {
        // This would execute: SELECT ?entity WHERE { ?entity rdf:type <entity_type> }
        // For now, returning mock entities
        Ok(vec![
            format!(
                "http://example.org/entity1_of_{}",
                self.simplify_uri(entity_type)
            ),
            format!(
                "http://example.org/entity2_of_{}",
                self.simplify_uri(entity_type)
            ),
            format!(
                "http://example.org/entity3_of_{}",
                self.simplify_uri(entity_type)
            ),
        ])
    }

    async fn find_entities_with_property(&self, property: &str) -> Result<Vec<String>> {
        // This would execute: SELECT DISTINCT ?entity WHERE { ?entity <property> ?value }
        // For now, returning mock entities
        Ok(vec![
            format!(
                "http://example.org/entity_with_{}",
                self.simplify_uri(property)
            ),
            format!(
                "http://example.org/another_entity_with_{}",
                self.simplify_uri(property)
            ),
        ])
    }

    async fn calculate_entity_similarity(&self, entity1: &str, entity2: &str) -> Result<f32> {
        // Calculate similarity based on shared properties, types, and relationships
        let mut similarity = 0.0;

        // Compare types
        let types1 = self.get_entity_types(entity1).await?;
        let types2 = self.get_entity_types(entity2).await?;
        let common_types_count = types1
            .iter()
            .collect::<HashSet<_>>()
            .intersection(&types2.iter().collect())
            .count();

        if !types1.is_empty() && !types2.is_empty() {
            similarity +=
                (common_types_count as f32) / (types1.len().max(types2.len()) as f32) * 0.4;
        }

        // Compare properties
        let props1 = self.get_entity_properties(entity1).await?;
        let props2 = self.get_entity_properties(entity2).await?;
        let common_props_count = props1
            .keys()
            .collect::<HashSet<_>>()
            .intersection(&props2.keys().collect())
            .count();

        if !props1.is_empty() && !props2.is_empty() {
            similarity +=
                (common_props_count as f32) / (props1.len().max(props2.len()) as f32) * 0.4;
        }

        // Compare neighbors (structural similarity)
        let neighbors1 = self.get_entity_neighbors(entity1).await?;
        let neighbors2 = self.get_entity_neighbors(entity2).await?;
        let common_neighbors_count = neighbors1
            .iter()
            .map(|n| &n.entity)
            .collect::<HashSet<_>>()
            .intersection(&neighbors2.iter().map(|n| &n.entity).collect())
            .count();

        if !neighbors1.is_empty() && !neighbors2.is_empty() {
            similarity += (common_neighbors_count as f32)
                / (neighbors1.len().max(neighbors2.len()) as f32)
                * 0.2;
        }

        Ok(similarity.min(1.0))
    }

    async fn calculate_property_similarity(
        &self,
        entity1: &str,
        entity2: &str,
        property: &str,
    ) -> Result<f32> {
        let props1 = self.get_entity_properties(entity1).await?;
        let props2 = self.get_entity_properties(entity2).await?;

        if let (Some(values1), Some(values2)) = (props1.get(property), props2.get(property)) {
            // Calculate overlap between property values
            let common_values_count = values1
                .iter()
                .collect::<HashSet<_>>()
                .intersection(&values2.iter().collect())
                .count();

            let total_values = values1.len().max(values2.len());
            if total_values > 0 {
                Ok(common_values_count as f32 / total_values as f32)
            } else {
                Ok(0.0)
            }
        } else {
            Ok(0.0)
        }
    }

    async fn get_entity_neighbors(&self, entity: &str) -> Result<Vec<EntityNeighbor>> {
        let mut neighbors = Vec::new();

        // Get outgoing relationships
        // This would use SPARQL queries like: SELECT ?p ?o WHERE { <entity> ?p ?o }
        // For now, implementing a basic version that would be replaced with actual SPARQL

        let _entity_node =
            NamedNode::new(entity).map_err(|e| anyhow!("Invalid entity URI: {}", e))?;

        // Note: This is a simplified implementation
        // In practice, this would execute SPARQL queries against the store

        neighbors.push(EntityNeighbor {
            entity: "http://example.org/related_entity".to_string(),
            relationship: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            direction: RelationshipDirection::Outgoing,
            strength: 1.0,
            labels: vec!["Related Entity".to_string()],
        });

        // Limit to max_neighbors
        neighbors.truncate(self.config.max_neighbors);

        Ok(neighbors)
    }

    async fn get_entity_types(&self, _entity: &str) -> Result<Vec<String>> {
        // This would execute: SELECT ?type WHERE { <entity> rdf:type ?type }
        Ok(vec!["http://www.w3.org/2002/07/owl#Thing".to_string()])
    }

    async fn get_entity_properties(&self, _entity: &str) -> Result<HashMap<String, Vec<String>>> {
        // This would execute: SELECT ?p ?o WHERE { <entity> ?p ?o }
        let mut properties = HashMap::new();
        properties.insert(
            "http://www.w3.org/2000/01/rdf-schema#label".to_string(),
            vec!["Example Entity".to_string()],
        );
        Ok(properties)
    }

    async fn get_schema_info(&self, entity: &str) -> Result<SchemaInfo> {
        if let Some(cached) = self.schema_cache.get(entity) {
            return Ok(cached.clone());
        }

        // This would analyze the schema for the entity
        let schema_info = SchemaInfo {
            classes: vec!["http://www.w3.org/2002/07/owl#Thing".to_string()],
            domain_restrictions: Vec::new(),
            range_restrictions: Vec::new(),
            cardinality_constraints: HashMap::new(),
            functional_properties: Vec::new(),
            equivalent_classes: Vec::new(),
            disjoint_classes: Vec::new(),
            shacl_shapes: Vec::new(),
        };

        Ok(schema_info)
    }

    async fn calculate_entity_relevance(
        &self,
        _entity: &str,
        neighbors: &[EntityNeighbor],
        _types: &[String],
    ) -> Result<f32> {
        // Calculate relevance based on connectivity, types, etc.
        let base_score = 0.5;
        let neighbor_bonus = neighbors.len() as f32 * 0.1;
        Ok((base_score + neighbor_bonus).min(1.0))
    }

    fn calculate_relationship_weight(&self, neighbor: &EntityNeighbor) -> f32 {
        // Prefer certain relationship types
        if self
            .config
            .preferred_relationships
            .contains(&neighbor.relationship)
        {
            1.0
        } else {
            0.7
        }
    }

    fn is_relationship_blacklisted(&self, relationship: &str) -> bool {
        self.config
            .blacklisted_relationships
            .contains(&relationship.to_string())
    }

    fn generate_path_explanation(&self, state: &PathState) -> Result<String> {
        if state.path_entities.len() < 2 {
            return Ok("Single entity path".to_string());
        }

        let mut explanation = format!(
            "Path from {} to {} via",
            state.path_entities.first().unwrap(),
            state.path_entities.last().unwrap()
        );

        for (i, relationship) in state.path_relationships.iter().enumerate() {
            if i < state.path_entities.len() - 1 {
                explanation.push_str(&format!(" {} ->", self.simplify_uri(relationship)));
            }
        }

        Ok(explanation)
    }

    fn simplify_uri(&self, uri: &str) -> String {
        // Extract the local name from URI
        if let Some(pos) = uri.rfind(['#', '/']) {
            uri[pos + 1..].to_string()
        } else {
            uri.to_string()
        }
    }
}

/// Internal state for path discovery
#[derive(Debug, Clone)]
struct PathState {
    current_entity: String,
    path_entities: Vec<String>,
    path_relationships: Vec<String>,
    depth: usize,
    score: f32,
}

/// Ordered wrapper for PathState to use in priority queue
#[derive(Debug, Clone)]
struct PathStateOrdered {
    state: PathState,
}

impl PartialEq for PathStateOrdered {
    fn eq(&self, other: &Self) -> bool {
        self.state.score == other.state.score
    }
}

impl Eq for PathStateOrdered {}

impl PartialOrd for PathStateOrdered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PathStateOrdered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.state
            .score
            .partial_cmp(&other.state.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Exploration suggestion for interactive navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationSuggestion {
    pub suggestion_type: SuggestionType,
    pub title: String,
    pub description: String,
    pub action: ExplorationAction,
    pub confidence: f32,
}

/// Types of exploration suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionType {
    ExploreNeighbor,
    ExploreType,
    DiscoverPaths,
    FindSimilar,
    SchemaAnalysis,
}

/// Actions that can be taken from exploration suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplorationAction {
    NavigateToEntity(String),
    FindSimilarEntities(String),
    DiscoverConnections(String),
    AnalyzeSchema(String),
    ExecuteQuery(String),
}

/// Related entity found through similarity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedEntity {
    pub entity: String,
    pub similarity_score: f32,
    pub relationship_type: String,
    pub explanation: String,
}

/// Schema validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Class hierarchy analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassHierarchyAnalysis {
    pub class: String,
    pub superclasses: Vec<String>,
    pub subclasses: Vec<String>,
    pub equivalent_classes: Vec<String>,
    pub disjoint_classes: Vec<String>,
    pub depth_from_root: u32,
    pub instance_count: u32,
}

/// SHACL shape validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeValidationResult {
    pub is_valid: bool,
    pub violations: Vec<ShapeViolation>,
    pub satisfied_shapes: Vec<String>,
}

/// SHACL shape violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeViolation {
    pub shape_id: String,
    pub property_path: String,
    pub violation_type: String,
    pub message: String,
    pub severity: ViolationSeverity,
}

/// Severity levels for SHACL violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Info,
    Warning,
    Violation,
}

/// Graph exploration results aggregator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationResults {
    pub paths: Vec<GraphPath>,
    pub expanded_entities: Vec<ExpandedEntity>,
    pub schema_suggestions: Vec<String>,
    pub exploration_metadata: HashMap<String, String>,
}

impl ExplorationResults {
    pub fn new() -> Self {
        Self {
            paths: Vec::new(),
            expanded_entities: Vec::new(),
            schema_suggestions: Vec::new(),
            exploration_metadata: HashMap::new(),
        }
    }

    /// Add exploration metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.exploration_metadata.insert(key, value);
    }

    /// Get a summary of the exploration results
    pub fn get_summary(&self) -> String {
        format!(
            "Exploration Results: {} paths found, {} entities expanded, {} schema suggestions",
            self.paths.len(),
            self.expanded_entities.len(),
            self.schema_suggestions.len()
        )
    }

    /// Convert to JSON for API responses
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| anyhow!("JSON serialization failed: {}", e))
    }
}

impl Default for ExplorationResults {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_exploration_config() {
        let config = ExplorationConfig::default();
        assert_eq!(config.max_depth, 5);
        assert_eq!(config.max_paths, 100);
        assert!(config.schema_aware);
    }

    #[tokio::test]
    async fn test_graph_path_creation() {
        let path = GraphPath {
            entities: vec!["entity1".to_string(), "entity2".to_string()],
            relationships: vec!["relationship1".to_string()],
            length: 1,
            relevance_score: 0.8,
            explanation: "Test path".to_string(),
            metadata: HashMap::new(),
        };

        assert_eq!(path.length, 1);
        assert_eq!(path.relevance_score, 0.8);
    }

    #[tokio::test]
    async fn test_exploration_results() {
        let mut results = ExplorationResults::new();
        results.add_metadata("test_key".to_string(), "test_value".to_string());

        let summary = results.get_summary();
        assert!(summary.contains("0 paths found"));

        assert!(results.to_json().is_ok());
    }
}
