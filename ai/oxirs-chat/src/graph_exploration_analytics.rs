//! Analytics on explored subgraphs: centrality, density, connectivity metrics
use super::graph_exploration_traversal::GraphExplorer;
use super::graph_exploration_types::{
    ClassHierarchyAnalysis, GraphPath, GuidanceType, QueryGuidance, SchemaInfo,
    SchemaValidationResult, ShaclShape, ShapeValidationResult, ShapeViolation, ViolationSeverity,
};
use anyhow::Result;
use std::collections::HashMap;

impl GraphExplorer {
    /// Calculate relationship strength between two entities
    pub async fn calculate_relationship_strength(
        &self,
        entity1: &str,
        entity2: &str,
        relationship: &str,
    ) -> Result<f32> {
        let mut strength = 0.5;

        if self.is_functional_property(relationship).await? {
            strength += 0.3;
        }

        let frequency = self.get_relationship_frequency(relationship).await?;
        strength += (1.0 / (frequency as f32 + 1.0)) * 0.2;

        let entity1_types = self.get_entity_types(entity1).await?;
        let entity2_types = self.get_entity_types(entity2).await?;
        let common_types_count = entity1_types
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .intersection(&entity2_types.iter().collect())
            .count();

        if common_types_count > 0 {
            strength += 0.1;
        }

        Ok(strength.min(1.0))
    }

    /// Rank paths by multiple criteria (relevance, length, preferred rels, hub connectivity)
    pub async fn rank_paths(&self, paths: &mut [GraphPath]) -> Result<()> {
        for path in paths.iter_mut() {
            let mut ranking_score = 0.0;

            ranking_score += 1.0 / (path.length as f32 + 1.0) * 0.3;
            ranking_score += path.relevance_score * 0.4;

            let preferred_relationship_count = path
                .relationships
                .iter()
                .filter(|rel| self.config.preferred_relationships.contains(rel))
                .count();
            ranking_score +=
                (preferred_relationship_count as f32 / path.relationships.len() as f32) * 0.2;

            ranking_score += self.calculate_hub_connectivity_bonus(path).await? * 0.1;

            path.relevance_score = ranking_score;
        }

        paths.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(())
    }

    /// Generate schema-based query guidance
    pub async fn generate_query_guidance(
        &self,
        context_entities: &[String],
        intent: &str,
    ) -> Result<Vec<QueryGuidance>> {
        let mut guidance = Vec::new();

        for entity in context_entities {
            let schema_info = self.get_schema_info(entity).await?;

            guidance.extend(
                self.generate_property_path_guidance(entity, &schema_info)
                    .await?,
            );
            guidance.extend(
                self.generate_type_constraint_guidance(entity, &schema_info)
                    .await?,
            );
            guidance.extend(
                self.generate_cardinality_guidance(entity, &schema_info)
                    .await?,
            );
            guidance.extend(
                self.generate_best_practice_guidance(entity, &schema_info, intent)
                    .await?,
            );
            guidance.extend(
                self.generate_consistency_guidance(entity, &schema_info)
                    .await?,
            );
        }

        guidance.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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

        let entities = self.extract_entities_from_query(query).await?;
        let properties = self.extract_properties_from_query(query).await?;

        for entity in &entities {
            if !self.is_valid_entity(entity).await? {
                validation_result
                    .errors
                    .push(format!("Unknown entity: {entity}"));
                validation_result.is_valid = false;
            }
        }

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

    /// Validate entity against SHACL shapes
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

    // ---- Private analytics helpers ----

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
        let uri_pattern = regex::Regex::new(r"<([^>]+)>").expect("regex pattern should be valid");
        let entities: Vec<String> = uri_pattern
            .captures_iter(query)
            .map(|cap| cap[1].to_string())
            .collect();
        Ok(entities)
    }

    async fn extract_properties_from_query(&self, query: &str) -> Result<Vec<String>> {
        let properties = self.extract_entities_from_query(query).await?;
        Ok(properties
            .into_iter()
            .filter(|uri| uri.contains("property") || uri.contains('#'))
            .collect())
    }

    async fn is_valid_entity(&self, entity: &str) -> Result<bool> {
        let types = self.get_entity_types(entity).await?;
        Ok(!types.is_empty())
    }

    async fn validate_property_domain_range(
        &self,
        _property: &str,
        _entities: &[String],
    ) -> Result<bool> {
        Ok(true)
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
        Ok(vec!["http://www.w3.org/2002/07/owl#Thing".to_string()])
    }

    async fn get_subclasses(&self, _class: &str) -> Result<Vec<String>> {
        Ok(vec![])
    }

    async fn get_equivalent_classes(&self, _class: &str) -> Result<Vec<String>> {
        Ok(vec![])
    }

    async fn get_disjoint_classes(&self, _class: &str) -> Result<Vec<String>> {
        Ok(vec![])
    }

    async fn calculate_class_depth(&self, class: &str) -> Result<u32> {
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
                break;
            }
        }
        Ok(depth)
    }

    async fn count_class_instances(&self, _class: &str) -> Result<u32> {
        Ok(42)
    }

    async fn get_shapes_for_class(&self, class: &str) -> Result<Vec<ShaclShape>> {
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

        for prop_shape in &shape.property_shapes {
            if let Some(values) = properties.get(&prop_shape.path) {
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
        Ok(1)
    }

    async fn is_functional_property(&self, property: &str) -> Result<bool> {
        let functional_properties = [
            "http://xmlns.com/foaf/0.1/name",
            "http://purl.org/dc/elements/1.1/title",
        ];
        Ok(functional_properties.contains(&property))
    }

    async fn get_relationship_frequency(&self, relationship: &str) -> Result<u32> {
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
            if connectivity > 10.0 {
                total_bonus += 0.1;
            } else if connectivity > 5.0 {
                total_bonus += 0.05;
            }
        }
        Ok(total_bonus / path.entities.len() as f32)
    }

    /// Get schema-aware suggestions for query expansion
    pub async fn get_schema_suggestions(&self, entity: &str) -> Result<Vec<String>> {
        let mut suggestions = Vec::new();
        if let Ok(schema_info) = self.get_schema_info(entity).await {
            for class in &schema_info.classes {
                suggestions.push(format!("Find all instances of {class}"));
                suggestions.push(format!("Find subclasses of {class}"));
            }
            for prop in &schema_info.functional_properties {
                suggestions.push(format!("Find the {prop} of {entity}"));
            }
            for domain in &schema_info.domain_restrictions {
                suggestions.push(format!("Find entities in domain {domain}"));
            }
        }
        Ok(suggestions)
    }

    /// Expand an entity to get its full neighborhood
    pub async fn expand_entity(
        &self,
        entity: &str,
    ) -> Result<super::graph_exploration_types::ExpandedEntity> {
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

        Ok(super::graph_exploration_types::ExpandedEntity {
            entity: entity.to_string(),
            neighbors,
            types,
            properties,
            relevance_score,
            schema_info,
        })
    }

    async fn calculate_entity_relevance(
        &self,
        _entity: &str,
        neighbors: &[super::graph_exploration_types::EntityNeighbor],
        _types: &[String],
    ) -> Result<f32> {
        let base_score = 0.5;
        let neighbor_bonus = neighbors.len() as f32 * 0.1;
        Ok((base_score + neighbor_bonus).min(1.0))
    }
}
