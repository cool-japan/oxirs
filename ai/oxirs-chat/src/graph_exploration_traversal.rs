//! BFS/DFS graph traversal, path finding, subgraph extraction
use super::graph_exploration_types::{
    EntityNeighbor, ExplorationAction, ExplorationConfig, ExplorationSuggestion, GraphPath,
    PathState, PathStateOrdered, RelatedEntity, SchemaInfo, SuggestionType,
};
use anyhow::{anyhow, Result};
use oxirs_core::{model::NamedNode, Store};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};

/// Graph exploration engine
pub struct GraphExplorer {
    pub(crate) store: Arc<dyn Store>,
    pub(crate) config: ExplorationConfig,
    pub(crate) schema_cache: HashMap<String, SchemaInfo>,
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

    /// Discover paths between two entities using BFS
    pub async fn discover_paths(
        &self,
        start_entity: &str,
        end_entity: &str,
    ) -> Result<Vec<GraphPath>> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

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

            let neighbors = self.get_entity_neighbors(&state.current_entity).await?;

            for neighbor in neighbors {
                if state.path_entities.contains(&neighbor.entity) {
                    continue;
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

        paths.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(paths)
    }

    /// Find the shortest path between entities
    pub async fn find_shortest_path(
        &self,
        start_entity: &str,
        end_entity: &str,
    ) -> Result<Option<GraphPath>> {
        let paths = self.discover_paths(start_entity, end_entity).await?;
        Ok(paths.iter().min_by_key(|path| path.length).cloned())
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

        queue.push_back((center_entity.to_string(), 0usize));

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

    /// Find multiple alternative paths using a priority queue
    pub async fn find_multiple_paths(
        &self,
        start_entity: &str,
        end_entity: &str,
        max_paths: usize,
    ) -> Result<Vec<GraphPath>> {
        let mut all_paths = Vec::new();
        let mut visited_path_signatures = HashSet::new();
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

                let new_score = state.score * self.calculate_relationship_weight(&neighbor);
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

    /// Generate interactive exploration suggestions
    pub async fn generate_exploration_suggestions(
        &self,
        current_entity: &str,
    ) -> Result<Vec<ExplorationSuggestion>> {
        let mut suggestions = Vec::new();
        let neighbors = self.get_entity_neighbors(current_entity).await?;
        let types = self.get_entity_types(current_entity).await?;

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

        for entity_type in types.iter().take(3) {
            suggestions.push(ExplorationSuggestion {
                suggestion_type: SuggestionType::ExploreType,
                title: format!("Find similar {}", self.simplify_uri(entity_type)),
                description: format!("Find other instances of {}", self.simplify_uri(entity_type)),
                action: ExplorationAction::FindSimilarEntities(entity_type.clone()),
                confidence: 0.8,
            });
        }

        suggestions.push(ExplorationSuggestion {
            suggestion_type: SuggestionType::DiscoverPaths,
            title: "Discover connection paths".to_string(),
            description: "Find how this entity connects to others".to_string(),
            action: ExplorationAction::DiscoverConnections(current_entity.to_string()),
            confidence: 0.7,
        });

        suggestions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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

        related_entities.sort_by(|a, b| {
            b.similarity_score
                .partial_cmp(&a.similarity_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        related_entities.dedup_by(|a, b| a.entity == b.entity);

        Ok(related_entities.into_iter().take(20).collect())
    }

    // ---- Internal helpers ----

    pub(crate) async fn get_entity_neighbors(&self, entity: &str) -> Result<Vec<EntityNeighbor>> {
        let mut neighbors = Vec::new();

        let _entity_node =
            NamedNode::new(entity).map_err(|e| anyhow!("Invalid entity URI: {}", e))?;

        neighbors.push(EntityNeighbor {
            entity: "http://example.org/related_entity".to_string(),
            relationship: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            direction: super::graph_exploration_types::RelationshipDirection::Outgoing,
            strength: 1.0,
            labels: vec!["Related Entity".to_string()],
        });

        neighbors.truncate(self.config.max_neighbors);
        Ok(neighbors)
    }

    pub(crate) async fn get_entity_types(&self, _entity: &str) -> Result<Vec<String>> {
        Ok(vec!["http://www.w3.org/2002/07/owl#Thing".to_string()])
    }

    pub(crate) async fn get_entity_properties(
        &self,
        _entity: &str,
    ) -> Result<HashMap<String, Vec<String>>> {
        let mut properties = HashMap::new();
        properties.insert(
            "http://www.w3.org/2000/01/rdf-schema#label".to_string(),
            vec!["Example Entity".to_string()],
        );
        Ok(properties)
    }

    pub(crate) async fn get_schema_info(&self, entity: &str) -> Result<SchemaInfo> {
        if let Some(cached) = self.schema_cache.get(entity) {
            return Ok(cached.clone());
        }
        Ok(SchemaInfo {
            classes: vec!["http://www.w3.org/2002/07/owl#Thing".to_string()],
            domain_restrictions: Vec::new(),
            range_restrictions: Vec::new(),
            cardinality_constraints: HashMap::new(),
            functional_properties: Vec::new(),
            equivalent_classes: Vec::new(),
            disjoint_classes: Vec::new(),
            shacl_shapes: Vec::new(),
        })
    }

    pub(crate) fn calculate_relationship_weight(&self, neighbor: &EntityNeighbor) -> f32 {
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

    pub(crate) fn is_relationship_blacklisted(&self, relationship: &str) -> bool {
        self.config
            .blacklisted_relationships
            .contains(&relationship.to_string())
    }

    pub(crate) fn generate_path_explanation(&self, state: &PathState) -> Result<String> {
        if state.path_entities.len() < 2 {
            return Ok("Single entity path".to_string());
        }

        let mut explanation = format!(
            "Path from {} to {} via",
            state
                .path_entities
                .first()
                .expect("collection validated to be non-empty"),
            state
                .path_entities
                .last()
                .expect("collection validated to be non-empty")
        );

        for (i, relationship) in state.path_relationships.iter().enumerate() {
            if i < state.path_entities.len() - 1 {
                explanation.push_str(&format!(" {} ->", self.simplify_uri(relationship)));
            }
        }

        Ok(explanation)
    }

    pub(crate) fn simplify_uri(&self, uri: &str) -> String {
        if let Some(pos) = uri.rfind(['#', '/']) {
            uri[pos + 1..].to_string()
        } else {
            uri.to_string()
        }
    }

    async fn find_entities_by_type(&self, entity_type: &str) -> Result<Vec<String>> {
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
        let mut similarity = 0.0;

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
}
