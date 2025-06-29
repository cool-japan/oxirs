//! Shape Inheritance and Composition Module
//!
//! Provides support for SHACL shape inheritance, composition, deactivation,
//! and advanced shape management features.

use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};

use oxirs_core::model::{NamedNode, Term};

use crate::{constraints::Constraint, ConstraintComponentId, Result, ShaclError, Shape, ShapeId};

/// Shape inheritance relationship types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InheritanceType {
    /// Direct inheritance via rdfs:subClassOf
    SubClass,
    /// Shape composition via sh:and
    AndComposition,
    /// Shape union via sh:or  
    OrComposition,
    /// Shape exclusion via sh:not
    NotComposition,
    /// Explicit shape reference via sh:node
    NodeReference,
}

/// Conflict resolution strategy for inheritance
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictResolutionStrategy {
    /// Base shape takes precedence (default)
    BaseShapePrecedence,
    /// Higher priority shape takes precedence
    PriorityBased,
    /// Latest defined constraint takes precedence
    LastWins,
    /// Merge constraints when compatible
    MergeCompatible,
    /// Fail on any conflicts
    FailOnConflict,
}

/// Shape inheritance relationship
#[derive(Debug, Clone)]
pub struct InheritanceRelation {
    /// Type of inheritance relationship
    pub inheritance_type: InheritanceType,
    /// Source shape (inheritor)
    pub source_shape: ShapeId,
    /// Target shape (inherited from)
    pub target_shape: ShapeId,
    /// Priority for ordering (lower number = higher priority)
    pub priority: i32,
}

/// Shape metadata for organization and documentation
#[derive(Debug, Clone)]
pub struct ShapeMetadata {
    /// Human-readable label
    pub label: Option<String>,
    /// Description or comment
    pub comment: Option<String>,
    /// Group classification
    pub group: Option<String>,
    /// Priority for validation ordering
    pub priority: i32,
    /// Whether the shape is deactivated
    pub deactivated: bool,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Version information
    pub version: Option<String>,
    /// Author information
    pub author: Option<String>,
    /// Creation date
    pub created: Option<String>,
    /// Last modification date
    pub modified: Option<String>,
}

impl Default for ShapeMetadata {
    fn default() -> Self {
        Self {
            label: None,
            comment: None,
            group: None,
            priority: 0,
            deactivated: false,
            tags: Vec::new(),
            version: None,
            author: None,
            created: None,
            modified: None,
        }
    }
}

/// Shape inheritance manager
#[derive(Debug)]
pub struct ShapeInheritanceManager {
    /// All inheritance relationships
    inheritance_relations: Vec<InheritanceRelation>,
    /// Shape metadata cache
    metadata_cache: HashMap<ShapeId, ShapeMetadata>,
    /// Computed inheritance hierarchy
    inheritance_hierarchy: HashMap<ShapeId, Vec<ShapeId>>,
    /// Resolved constraint cache
    resolved_constraints_cache: HashMap<ShapeId, IndexMap<ConstraintComponentId, Constraint>>,
    /// Conflict resolution strategy
    conflict_resolution_strategy: ConflictResolutionStrategy,
}

impl ShapeInheritanceManager {
    /// Create a new shape inheritance manager
    pub fn new() -> Self {
        Self {
            inheritance_relations: Vec::new(),
            metadata_cache: HashMap::new(),
            inheritance_hierarchy: HashMap::new(),
            resolved_constraints_cache: HashMap::new(),
            conflict_resolution_strategy: ConflictResolutionStrategy::BaseShapePrecedence,
        }
    }

    /// Create a new manager with specific conflict resolution strategy
    pub fn with_conflict_resolution(strategy: ConflictResolutionStrategy) -> Self {
        Self {
            inheritance_relations: Vec::new(),
            metadata_cache: HashMap::new(),
            inheritance_hierarchy: HashMap::new(),
            resolved_constraints_cache: HashMap::new(),
            conflict_resolution_strategy: strategy,
        }
    }

    /// Set the conflict resolution strategy
    pub fn set_conflict_resolution_strategy(&mut self, strategy: ConflictResolutionStrategy) {
        self.conflict_resolution_strategy = strategy;
        self.invalidate_caches(); // Clear cache since resolution might change
    }

    /// Get the current conflict resolution strategy
    pub fn get_conflict_resolution_strategy(&self) -> &ConflictResolutionStrategy {
        &self.conflict_resolution_strategy
    }

    /// Add an inheritance relationship
    pub fn add_inheritance_relation(&mut self, relation: InheritanceRelation) {
        self.inheritance_relations.push(relation);
        self.invalidate_caches();
    }

    /// Remove an inheritance relationship
    pub fn remove_inheritance_relation(&mut self, source: &ShapeId, target: &ShapeId) {
        self.inheritance_relations
            .retain(|r| !(r.source_shape == *source && r.target_shape == *target));
        self.invalidate_caches();
    }

    /// Set metadata for a shape
    pub fn set_shape_metadata(&mut self, shape_id: ShapeId, metadata: ShapeMetadata) {
        self.metadata_cache.insert(shape_id, metadata);
    }

    /// Get metadata for a shape
    pub fn get_shape_metadata(&self, shape_id: &ShapeId) -> Option<&ShapeMetadata> {
        self.metadata_cache.get(shape_id)
    }

    /// Check if a shape is deactivated
    pub fn is_shape_deactivated(&self, shape_id: &ShapeId) -> bool {
        self.metadata_cache
            .get(shape_id)
            .map(|m| m.deactivated)
            .unwrap_or(false)
    }

    /// Deactivate a shape
    pub fn deactivate_shape(&mut self, shape_id: &ShapeId) {
        let metadata = self
            .metadata_cache
            .entry(shape_id.clone())
            .or_insert_with(ShapeMetadata::default);
        metadata.deactivated = true;
    }

    /// Activate a shape
    pub fn activate_shape(&mut self, shape_id: &ShapeId) {
        let metadata = self
            .metadata_cache
            .entry(shape_id.clone())
            .or_insert_with(ShapeMetadata::default);
        metadata.deactivated = false;
    }

    /// Get all parent shapes for a given shape
    pub fn get_parent_shapes(&self, shape_id: &ShapeId) -> Vec<ShapeId> {
        self.inheritance_relations
            .iter()
            .filter(|r| r.source_shape == *shape_id)
            .map(|r| r.target_shape.clone())
            .collect()
    }

    /// Get all child shapes for a given shape
    pub fn get_child_shapes(&self, shape_id: &ShapeId) -> Vec<ShapeId> {
        self.inheritance_relations
            .iter()
            .filter(|r| r.target_shape == *shape_id)
            .map(|r| r.source_shape.clone())
            .collect()
    }

    /// Compute the complete inheritance hierarchy for a shape
    pub fn compute_inheritance_hierarchy(&mut self, shape_id: &ShapeId) -> Vec<ShapeId> {
        if let Some(cached) = self.inheritance_hierarchy.get(shape_id) {
            return cached.clone();
        }

        let mut hierarchy = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = vec![shape_id.clone()];

        while let Some(current_shape) = queue.pop() {
            if visited.contains(&current_shape) {
                continue; // Avoid cycles
            }
            visited.insert(current_shape.clone());

            // Add parents to hierarchy
            let parents = self.get_parent_shapes(&current_shape);
            for parent in parents {
                if !hierarchy.contains(&parent) && parent != *shape_id {
                    hierarchy.push(parent.clone());
                    queue.push(parent);
                }
            }
        }

        // Sort by priority (if available)
        hierarchy.sort_by(|a, b| {
            let priority_a = self.metadata_cache.get(a).map(|m| m.priority).unwrap_or(0);
            let priority_b = self.metadata_cache.get(b).map(|m| m.priority).unwrap_or(0);
            priority_a.cmp(&priority_b)
        });

        self.inheritance_hierarchy
            .insert(shape_id.clone(), hierarchy.clone());
        hierarchy
    }

    /// Resolve all constraints for a shape including inherited constraints
    pub fn resolve_shape_constraints(
        &mut self,
        shape_id: &ShapeId,
        base_shape: &Shape,
        all_shapes: &IndexMap<ShapeId, Shape>,
    ) -> Result<IndexMap<ConstraintComponentId, Constraint>> {
        if let Some(cached) = self.resolved_constraints_cache.get(shape_id) {
            return Ok(cached.clone());
        }

        let resolved_constraints = match self.conflict_resolution_strategy {
            ConflictResolutionStrategy::BaseShapePrecedence => {
                self.resolve_with_base_precedence(shape_id, base_shape, all_shapes)?
            }
            ConflictResolutionStrategy::PriorityBased => {
                self.resolve_with_priority(shape_id, base_shape, all_shapes)?
            }
            ConflictResolutionStrategy::LastWins => {
                self.resolve_with_last_wins(shape_id, base_shape, all_shapes)?
            }
            ConflictResolutionStrategy::MergeCompatible => {
                self.resolve_with_merge(shape_id, base_shape, all_shapes)?
            }
            ConflictResolutionStrategy::FailOnConflict => {
                self.resolve_with_conflict_detection(shape_id, base_shape, all_shapes)?
            }
        };

        self.resolved_constraints_cache
            .insert(shape_id.clone(), resolved_constraints.clone());
        Ok(resolved_constraints)
    }

    /// Resolve constraints with base shape precedence
    fn resolve_with_base_precedence(
        &mut self,
        shape_id: &ShapeId,
        base_shape: &Shape,
        all_shapes: &IndexMap<ShapeId, Shape>,
    ) -> Result<IndexMap<ConstraintComponentId, Constraint>> {
        let mut resolved_constraints = IndexMap::new();

        // Start with base shape constraints
        for (constraint_id, constraint) in &base_shape.constraints {
            resolved_constraints.insert(constraint_id.clone(), constraint.clone());
        }

        // Add inherited constraints (base shape constraints take precedence)
        let hierarchy = self.compute_inheritance_hierarchy(shape_id);
        for parent_shape_id in hierarchy {
            if let Some(parent_shape) = all_shapes.get(&parent_shape_id) {
                if self.is_shape_deactivated(&parent_shape_id) {
                    continue;
                }

                for (constraint_id, constraint) in &parent_shape.constraints {
                    if !resolved_constraints.contains_key(constraint_id) {
                        resolved_constraints.insert(constraint_id.clone(), constraint.clone());
                    }
                }
            }
        }

        Ok(resolved_constraints)
    }

    /// Resolve constraints with priority-based resolution
    fn resolve_with_priority(
        &mut self,
        shape_id: &ShapeId,
        base_shape: &Shape,
        all_shapes: &IndexMap<ShapeId, Shape>,
    ) -> Result<IndexMap<ConstraintComponentId, Constraint>> {
        let mut constraints_by_priority: Vec<(
            i32,
            ShapeId,
            &IndexMap<ConstraintComponentId, Constraint>,
        )> = Vec::new();

        // Add base shape
        let base_priority = base_shape.effective_priority();
        constraints_by_priority.push((base_priority, shape_id.clone(), &base_shape.constraints));

        // Add inherited shapes
        let hierarchy = self.compute_inheritance_hierarchy(shape_id);
        for parent_shape_id in hierarchy {
            if let Some(parent_shape) = all_shapes.get(&parent_shape_id) {
                if self.is_shape_deactivated(&parent_shape_id) {
                    continue;
                }
                let priority = parent_shape.effective_priority();
                constraints_by_priority.push((
                    priority,
                    parent_shape_id,
                    &parent_shape.constraints,
                ));
            }
        }

        // Sort by priority (highest first)
        constraints_by_priority.sort_by(|a, b| b.0.cmp(&a.0));

        let mut resolved_constraints = IndexMap::new();
        for (_, _, constraints) in constraints_by_priority {
            for (constraint_id, constraint) in constraints {
                if !resolved_constraints.contains_key(constraint_id) {
                    resolved_constraints.insert(constraint_id.clone(), constraint.clone());
                }
            }
        }

        Ok(resolved_constraints)
    }

    /// Resolve constraints with last-wins strategy
    fn resolve_with_last_wins(
        &mut self,
        shape_id: &ShapeId,
        base_shape: &Shape,
        all_shapes: &IndexMap<ShapeId, Shape>,
    ) -> Result<IndexMap<ConstraintComponentId, Constraint>> {
        let mut resolved_constraints = IndexMap::new();

        // Add inherited constraints first
        let hierarchy = self.compute_inheritance_hierarchy(shape_id);
        for parent_shape_id in hierarchy {
            if let Some(parent_shape) = all_shapes.get(&parent_shape_id) {
                if self.is_shape_deactivated(&parent_shape_id) {
                    continue;
                }

                for (constraint_id, constraint) in &parent_shape.constraints {
                    resolved_constraints.insert(constraint_id.clone(), constraint.clone());
                }
            }
        }

        // Add base shape constraints last (they win)
        for (constraint_id, constraint) in &base_shape.constraints {
            resolved_constraints.insert(constraint_id.clone(), constraint.clone());
        }

        Ok(resolved_constraints)
    }

    /// Resolve constraints with merging compatible constraints
    fn resolve_with_merge(
        &mut self,
        shape_id: &ShapeId,
        base_shape: &Shape,
        all_shapes: &IndexMap<ShapeId, Shape>,
    ) -> Result<IndexMap<ConstraintComponentId, Constraint>> {
        let mut resolved_constraints = IndexMap::new();
        let mut constraint_sources: HashMap<ConstraintComponentId, Vec<(ShapeId, Constraint)>> =
            HashMap::new();

        // Collect all constraints from all shapes
        constraint_sources.extend(
            base_shape
                .constraints
                .iter()
                .map(|(id, constraint)| (id.clone(), vec![(shape_id.clone(), constraint.clone())])),
        );

        let hierarchy = self.compute_inheritance_hierarchy(shape_id);
        for parent_shape_id in hierarchy {
            if let Some(parent_shape) = all_shapes.get(&parent_shape_id) {
                if self.is_shape_deactivated(&parent_shape_id) {
                    continue;
                }

                for (constraint_id, constraint) in &parent_shape.constraints {
                    constraint_sources
                        .entry(constraint_id.clone())
                        .or_insert_with(Vec::new)
                        .push((parent_shape_id.clone(), constraint.clone()));
                }
            }
        }

        // Merge compatible constraints
        for (constraint_id, sources) in constraint_sources {
            if sources.len() == 1 {
                resolved_constraints.insert(constraint_id, sources[0].1.clone());
            } else {
                // Try to merge constraints
                let merged = self.try_merge_constraints(&sources)?;
                resolved_constraints.insert(constraint_id, merged);
            }
        }

        Ok(resolved_constraints)
    }

    /// Resolve constraints with conflict detection (fails on conflicts)
    fn resolve_with_conflict_detection(
        &mut self,
        shape_id: &ShapeId,
        base_shape: &Shape,
        all_shapes: &IndexMap<ShapeId, Shape>,
    ) -> Result<IndexMap<ConstraintComponentId, Constraint>> {
        let mut constraint_sources: HashMap<ConstraintComponentId, Vec<ShapeId>> = HashMap::new();

        // Track constraint sources
        for constraint_id in base_shape.constraints.keys() {
            constraint_sources.insert(constraint_id.clone(), vec![shape_id.clone()]);
        }

        let hierarchy = self.compute_inheritance_hierarchy(shape_id);
        for parent_shape_id in hierarchy {
            if let Some(parent_shape) = all_shapes.get(&parent_shape_id) {
                if self.is_shape_deactivated(&parent_shape_id) {
                    continue;
                }

                for constraint_id in parent_shape.constraints.keys() {
                    constraint_sources
                        .entry(constraint_id.clone())
                        .or_insert_with(Vec::new)
                        .push(parent_shape_id.clone());
                }
            }
        }

        // Check for conflicts
        for (constraint_id, sources) in &constraint_sources {
            if sources.len() > 1 {
                return Err(ShaclError::ShapeParsing(format!(
                    "Constraint conflict detected for '{}' in shapes: {:?}",
                    constraint_id, sources
                )));
            }
        }

        // No conflicts, use base precedence
        self.resolve_with_base_precedence(shape_id, base_shape, all_shapes)
    }

    /// Try to merge compatible constraints
    fn try_merge_constraints(&self, sources: &[(ShapeId, Constraint)]) -> Result<Constraint> {
        if sources.is_empty() {
            return Err(ShaclError::ShapeParsing(
                "No constraints to merge".to_string(),
            ));
        }

        // For now, just take the first constraint
        // In a more sophisticated implementation, this would check constraint compatibility
        // and merge numeric ranges, combine string patterns, etc.
        Ok(sources[0].1.clone())
    }

    /// Check for inheritance cycles
    pub fn check_for_cycles(&self) -> Result<()> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for relation in &self.inheritance_relations {
            if !visited.contains(&relation.source_shape) {
                if self.has_cycle_util(&relation.source_shape, &mut visited, &mut rec_stack)? {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Inheritance cycle detected involving shape: {}",
                        relation.source_shape
                    )));
                }
            }
        }

        Ok(())
    }

    fn has_cycle_util(
        &self,
        shape_id: &ShapeId,
        visited: &mut HashSet<ShapeId>,
        rec_stack: &mut HashSet<ShapeId>,
    ) -> Result<bool> {
        visited.insert(shape_id.clone());
        rec_stack.insert(shape_id.clone());

        // Check all children
        for child in self.get_child_shapes(shape_id) {
            if !visited.contains(&child) {
                if self.has_cycle_util(&child, visited, rec_stack)? {
                    return Ok(true);
                }
            } else if rec_stack.contains(&child) {
                return Ok(true);
            }
        }

        rec_stack.remove(shape_id);
        Ok(false)
    }

    /// Get shapes ordered by validation priority
    pub fn get_shapes_by_priority(&self, shapes: &IndexMap<ShapeId, Shape>) -> Vec<ShapeId> {
        let mut shape_priorities: Vec<(ShapeId, i32)> = shapes
            .keys()
            .map(|shape_id| {
                let priority = self
                    .metadata_cache
                    .get(shape_id)
                    .map(|m| m.priority)
                    .unwrap_or(0);
                (shape_id.clone(), priority)
            })
            .collect();

        shape_priorities.sort_by(|a, b| a.1.cmp(&b.1));
        shape_priorities
            .into_iter()
            .map(|(shape_id, _)| shape_id)
            .collect()
    }

    /// Get active shapes (non-deactivated)
    pub fn get_active_shapes(&self, shapes: &IndexMap<ShapeId, Shape>) -> Vec<ShapeId> {
        shapes
            .keys()
            .filter(|shape_id| !self.is_shape_deactivated(shape_id))
            .cloned()
            .collect()
    }

    /// Get shapes by group
    pub fn get_shapes_by_group(&self, group: &str) -> Vec<ShapeId> {
        self.metadata_cache
            .iter()
            .filter(|(_, metadata)| metadata.group.as_ref().map(|g| g == group).unwrap_or(false))
            .map(|(shape_id, _)| shape_id.clone())
            .collect()
    }

    /// Get shapes by tag
    pub fn get_shapes_by_tag(&self, tag: &str) -> Vec<ShapeId> {
        self.metadata_cache
            .iter()
            .filter(|(_, metadata)| metadata.tags.contains(&tag.to_string()))
            .map(|(shape_id, _)| shape_id.clone())
            .collect()
    }

    /// Compute shape composition for AND, OR, NOT constraints
    pub fn compute_shape_composition(
        &self,
        composition_type: InheritanceType,
        shape_ids: &[ShapeId],
        all_shapes: &IndexMap<ShapeId, Shape>,
    ) -> Result<IndexMap<ConstraintComponentId, Constraint>> {
        let mut composed_constraints = IndexMap::new();

        match composition_type {
            InheritanceType::AndComposition => {
                // For AND composition, include all constraints from all shapes
                for shape_id in shape_ids {
                    if let Some(shape) = all_shapes.get(shape_id) {
                        if self.is_shape_deactivated(shape_id) {
                            continue;
                        }
                        for (constraint_id, constraint) in &shape.constraints {
                            composed_constraints.insert(
                                ConstraintComponentId::new(format!(
                                    "{}_{}",
                                    shape_id, constraint_id
                                )),
                                constraint.clone(),
                            );
                        }
                    }
                }
            }
            InheritanceType::OrComposition => {
                // For OR composition, this would require more complex logic
                // For now, we'll just include constraints from the first active shape
                for shape_id in shape_ids {
                    if let Some(shape) = all_shapes.get(shape_id) {
                        if !self.is_shape_deactivated(shape_id) {
                            for (constraint_id, constraint) in &shape.constraints {
                                composed_constraints
                                    .insert(constraint_id.clone(), constraint.clone());
                            }
                            break; // Take first active shape for OR
                        }
                    }
                }
            }
            InheritanceType::NotComposition => {
                // NOT composition would require special handling in validation
                // For now, we'll mark these constraints specially
                for shape_id in shape_ids {
                    if let Some(shape) = all_shapes.get(shape_id) {
                        if !self.is_shape_deactivated(shape_id) {
                            for (constraint_id, constraint) in &shape.constraints {
                                composed_constraints.insert(
                                    ConstraintComponentId::new(format!("NOT_{}", constraint_id)),
                                    constraint.clone(),
                                );
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(ShaclError::ShapeParsing(format!(
                    "Unsupported composition type: {:?}",
                    composition_type
                )));
            }
        }

        Ok(composed_constraints)
    }

    /// Clear all caches
    fn invalidate_caches(&mut self) {
        self.inheritance_hierarchy.clear();
        self.resolved_constraints_cache.clear();
    }

    /// Get inheritance statistics
    pub fn get_inheritance_stats(&self) -> InheritanceStats {
        let total_relations = self.inheritance_relations.len();
        let active_shapes = self
            .metadata_cache
            .values()
            .filter(|m| !m.deactivated)
            .count();
        let deactivated_shapes = self
            .metadata_cache
            .values()
            .filter(|m| m.deactivated)
            .count();

        let mut inheritance_types = HashMap::new();
        for relation in &self.inheritance_relations {
            *inheritance_types
                .entry(relation.inheritance_type.clone())
                .or_insert(0) += 1;
        }

        InheritanceStats {
            total_relations,
            active_shapes,
            deactivated_shapes,
            inheritance_types,
        }
    }

    /// Export inheritance relationships
    pub fn export_inheritance_relations(&self) -> Vec<InheritanceRelation> {
        self.inheritance_relations.clone()
    }

    /// Import inheritance relationships
    pub fn import_inheritance_relations(&mut self, relations: Vec<InheritanceRelation>) {
        self.inheritance_relations = relations;
        self.invalidate_caches();
    }
}

impl Default for ShapeInheritanceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about shape inheritance
#[derive(Debug, Clone)]
pub struct InheritanceStats {
    pub total_relations: usize,
    pub active_shapes: usize,
    pub deactivated_shapes: usize,
    pub inheritance_types: HashMap<InheritanceType, usize>,
}

/// Utilities for parsing shape inheritance from RDF
pub struct InheritanceParser;

impl InheritanceParser {
    /// Parse shape metadata from RDF terms
    pub fn parse_shape_metadata(shape_terms: &[(Term, Term)]) -> ShapeMetadata {
        let mut metadata = ShapeMetadata::default();

        for (predicate, object) in shape_terms {
            if let Term::NamedNode(pred_node) = predicate {
                match pred_node.as_str() {
                    "http://www.w3.org/2000/01/rdf-schema#label" => {
                        if let Term::Literal(lit) = object {
                            metadata.label = Some(lit.value().to_string());
                        }
                    }
                    "http://www.w3.org/2000/01/rdf-schema#comment" => {
                        if let Term::Literal(lit) = object {
                            metadata.comment = Some(lit.value().to_string());
                        }
                    }
                    "http://www.w3.org/ns/shacl#deactivated" => {
                        if let Term::Literal(lit) = object {
                            metadata.deactivated = lit.value() == "true";
                        }
                    }
                    "http://www.w3.org/ns/shacl#group" => {
                        if let Term::NamedNode(group_node) = object {
                            metadata.group = Some(group_node.as_str().to_string());
                        }
                    }
                    "http://www.w3.org/ns/shacl#order" => {
                        if let Term::Literal(lit) = object {
                            if let Ok(priority) = lit.value().parse::<i32>() {
                                metadata.priority = priority;
                            }
                        }
                    }
                    _ => {} // Ignore unknown properties
                }
            }
        }

        metadata
    }

    /// Parse inheritance relationships from RDF terms
    pub fn parse_inheritance_relations(
        shape_id: &ShapeId,
        shape_terms: &[(Term, Term)],
    ) -> Vec<InheritanceRelation> {
        let mut relations = Vec::new();

        for (predicate, object) in shape_terms {
            if let Term::NamedNode(pred_node) = predicate {
                let inheritance_type = match pred_node.as_str() {
                    "http://www.w3.org/2000/01/rdf-schema#subClassOf" => {
                        Some(InheritanceType::SubClass)
                    }
                    "http://www.w3.org/ns/shacl#and" => Some(InheritanceType::AndComposition),
                    "http://www.w3.org/ns/shacl#or" => Some(InheritanceType::OrComposition),
                    "http://www.w3.org/ns/shacl#not" => Some(InheritanceType::NotComposition),
                    "http://www.w3.org/ns/shacl#node" => Some(InheritanceType::NodeReference),
                    _ => None,
                };

                if let (Some(inh_type), Term::NamedNode(target_node)) = (inheritance_type, object) {
                    relations.push(InheritanceRelation {
                        inheritance_type: inh_type,
                        source_shape: shape_id.clone(),
                        target_shape: ShapeId::new(target_node.as_str()),
                        priority: 0,
                    });
                }
            }
        }

        relations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_inheritance_manager_creation() {
        let manager = ShapeInheritanceManager::new();
        assert_eq!(manager.inheritance_relations.len(), 0);
        assert_eq!(manager.metadata_cache.len(), 0);
    }

    #[test]
    fn test_shape_deactivation() {
        let mut manager = ShapeInheritanceManager::new();
        let shape_id = ShapeId::new("http://example.org/shape1");

        assert!(!manager.is_shape_deactivated(&shape_id));

        manager.deactivate_shape(&shape_id);
        assert!(manager.is_shape_deactivated(&shape_id));

        manager.activate_shape(&shape_id);
        assert!(!manager.is_shape_deactivated(&shape_id));
    }

    #[test]
    fn test_inheritance_relation_management() {
        let mut manager = ShapeInheritanceManager::new();

        let relation = InheritanceRelation {
            inheritance_type: InheritanceType::SubClass,
            source_shape: ShapeId::new("http://example.org/child"),
            target_shape: ShapeId::new("http://example.org/parent"),
            priority: 0,
        };

        manager.add_inheritance_relation(relation);
        assert_eq!(manager.inheritance_relations.len(), 1);

        let parents = manager.get_parent_shapes(&ShapeId::new("http://example.org/child"));
        assert_eq!(parents.len(), 1);
        assert_eq!(parents[0], ShapeId::new("http://example.org/parent"));

        let children = manager.get_child_shapes(&ShapeId::new("http://example.org/parent"));
        assert_eq!(children.len(), 1);
        assert_eq!(children[0], ShapeId::new("http://example.org/child"));
    }

    #[test]
    fn test_shape_metadata() {
        let mut manager = ShapeInheritanceManager::new();
        let shape_id = ShapeId::new("http://example.org/shape1");

        let metadata = ShapeMetadata {
            label: Some("Test Shape".to_string()),
            comment: Some("A test shape for validation".to_string()),
            group: Some("validation".to_string()),
            priority: 10,
            deactivated: false,
            tags: vec!["test".to_string(), "validation".to_string()],
            version: Some("1.0".to_string()),
            author: Some("Test Author".to_string()),
            created: Some("2023-01-01".to_string()),
            modified: Some("2023-01-02".to_string()),
        };

        manager.set_shape_metadata(shape_id.clone(), metadata.clone());

        let retrieved = manager.get_shape_metadata(&shape_id).unwrap();
        assert_eq!(retrieved.label, metadata.label);
        assert_eq!(retrieved.priority, metadata.priority);
        assert_eq!(retrieved.tags, metadata.tags);
    }

    #[test]
    fn test_metadata_parsing() {
        use oxirs_core::model::{Literal, NamedNode};

        let terms = vec![
            (
                Term::NamedNode(
                    NamedNode::new("http://www.w3.org/2000/01/rdf-schema#label").unwrap(),
                ),
                Term::Literal(Literal::new_simple_literal("Test Shape")),
            ),
            (
                Term::NamedNode(NamedNode::new("http://www.w3.org/ns/shacl#deactivated").unwrap()),
                Term::Literal(Literal::new_simple_literal("true")),
            ),
        ];

        let metadata = InheritanceParser::parse_shape_metadata(&terms);
        assert_eq!(metadata.label, Some("Test Shape".to_string()));
        assert!(metadata.deactivated);
    }

    #[test]
    fn test_inheritance_stats() {
        let mut manager = ShapeInheritanceManager::new();

        // Add some relations and metadata
        manager.add_inheritance_relation(InheritanceRelation {
            inheritance_type: InheritanceType::SubClass,
            source_shape: ShapeId::new("child1"),
            target_shape: ShapeId::new("parent1"),
            priority: 0,
        });

        manager.add_inheritance_relation(InheritanceRelation {
            inheritance_type: InheritanceType::AndComposition,
            source_shape: ShapeId::new("child2"),
            target_shape: ShapeId::new("parent2"),
            priority: 0,
        });

        manager.set_shape_metadata(ShapeId::new("shape1"), ShapeMetadata::default());
        manager.deactivate_shape(&ShapeId::new("shape2"));

        let stats = manager.get_inheritance_stats();
        assert_eq!(stats.total_relations, 2);
        assert_eq!(stats.active_shapes, 1);
        assert_eq!(stats.deactivated_shapes, 1);
    }

    #[test]
    fn test_conflict_resolution_strategies() {
        let mut manager = ShapeInheritanceManager::new();

        // Test setting different strategies
        assert_eq!(
            manager.get_conflict_resolution_strategy(),
            &ConflictResolutionStrategy::BaseShapePrecedence
        );

        manager.set_conflict_resolution_strategy(ConflictResolutionStrategy::PriorityBased);
        assert_eq!(
            manager.get_conflict_resolution_strategy(),
            &ConflictResolutionStrategy::PriorityBased
        );

        // Test creating manager with specific strategy
        let priority_manager =
            ShapeInheritanceManager::with_conflict_resolution(ConflictResolutionStrategy::LastWins);
        assert_eq!(
            priority_manager.get_conflict_resolution_strategy(),
            &ConflictResolutionStrategy::LastWins
        );
    }

    #[test]
    fn test_advanced_constraint_resolution() {
        use crate::{constraints::*, ConstraintComponentId, Shape, ShapeType};

        let mut manager = ShapeInheritanceManager::new();
        let mut all_shapes = IndexMap::new();

        // Create base shape with priority 10
        let mut base_shape = Shape::new(ShapeId::new("base"), ShapeType::NodeShape);
        base_shape.priority = Some(10);
        base_shape.add_constraint(
            ConstraintComponentId::new("minCount"),
            Constraint::MinCount(MinCountConstraint { min_count: 1 }),
        );

        // Create parent shape with priority 20 (higher)
        let mut parent_shape = Shape::new(ShapeId::new("parent"), ShapeType::NodeShape);
        parent_shape.priority = Some(20);
        parent_shape.add_constraint(
            ConstraintComponentId::new("minCount"),
            Constraint::MinCount(MinCountConstraint { min_count: 2 }),
        );
        parent_shape.add_constraint(
            ConstraintComponentId::new("maxCount"),
            Constraint::MaxCount(MaxCountConstraint { max_count: 10 }),
        );

        all_shapes.insert(ShapeId::new("base"), base_shape.clone());
        all_shapes.insert(ShapeId::new("parent"), parent_shape);

        // Add inheritance relation
        manager.add_inheritance_relation(InheritanceRelation {
            inheritance_type: InheritanceType::SubClass,
            source_shape: ShapeId::new("base"),
            target_shape: ShapeId::new("parent"),
            priority: 0,
        });

        // Test base precedence (default)
        let resolved = manager
            .resolve_shape_constraints(&ShapeId::new("base"), &base_shape, &all_shapes)
            .unwrap();
        assert_eq!(resolved.len(), 2);
        // Base shape minCount should win
        if let Some(Constraint::MinCount(count)) =
            resolved.get(&ConstraintComponentId::new("minCount"))
        {
            assert_eq!(count.min_count, 1);
        } else {
            panic!("Expected MinCount constraint");
        }

        // Test priority-based resolution
        manager.set_conflict_resolution_strategy(ConflictResolutionStrategy::PriorityBased);
        let resolved = manager
            .resolve_shape_constraints(&ShapeId::new("base"), &base_shape, &all_shapes)
            .unwrap();
        assert_eq!(resolved.len(), 2);
        // Parent shape minCount should win (higher priority)
        if let Some(Constraint::MinCount(count)) =
            resolved.get(&ConstraintComponentId::new("minCount"))
        {
            assert_eq!(count.min_count, 2);
        } else {
            panic!("Expected MinCount constraint");
        }
    }
}
