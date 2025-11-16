//! Shape Operations: Generalization, Specialization, Merging, and Refactoring
//!
//! This module provides advanced operations on SHACL shapes including:
//! - Shape generalization (making shapes less restrictive)
//! - Shape specialization (making shapes more restrictive)
//! - Shape merging (combining multiple shapes)
//! - Shape refactoring (restructuring shape definitions)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;

use crate::{
    Constraint, ConstraintComponentId, Result, Severity, ShaclError, Shape, ShapeId, ShapeMetadata,
};

/// Strategy for shape generalization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeneralizationStrategy {
    /// Remove most restrictive constraints
    RemoveRestrictive,
    /// Widen value ranges
    WidenRanges,
    /// Reduce cardinality requirements
    ReduceCardinality,
    /// Make all constraints optional
    MakeOptional,
    /// Custom strategy
    Custom,
}

/// Strategy for shape specialization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpecializationStrategy {
    /// Add more restrictive constraints
    AddRestrictive,
    /// Narrow value ranges
    NarrowRanges,
    /// Increase cardinality requirements
    IncreaseCardinality,
    /// Make constraints mandatory
    MakeMandatory,
    /// Custom strategy
    Custom,
}

/// Strategy for shape merging
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Combine all constraints (union)
    Union,
    /// Only keep common constraints (intersection)
    Intersection,
    /// Keep the most restrictive constraints
    MostRestrictive,
    /// Keep the least restrictive constraints
    LeastRestrictive,
    /// Custom merge logic
    Custom,
}

/// Shape generalization engine
pub struct ShapeGeneralizer {
    /// Strategy to use
    strategy: GeneralizationStrategy,
}

impl ShapeGeneralizer {
    /// Create a new generalizer
    pub fn new(strategy: GeneralizationStrategy) -> Self {
        Self { strategy }
    }

    /// Generalize a shape
    pub fn generalize(&self, shape: &Shape) -> Result<Shape> {
        let mut generalized = shape.clone();

        match self.strategy {
            GeneralizationStrategy::RemoveRestrictive => {
                self.remove_restrictive_constraints(&mut generalized)?
            }
            GeneralizationStrategy::WidenRanges => self.widen_ranges(&mut generalized)?,
            GeneralizationStrategy::ReduceCardinality => {
                self.reduce_cardinality(&mut generalized)?
            }
            GeneralizationStrategy::MakeOptional => self.make_optional(&mut generalized)?,
            GeneralizationStrategy::Custom => {}
        }

        Ok(generalized)
    }

    /// Remove highly restrictive constraints
    fn remove_restrictive_constraints(&self, shape: &mut Shape) -> Result<()> {
        // Identify and remove restrictive constraint types
        let restrictive_types = [
            "sh:equals",
            "sh:disjoint",
            "sh:lessThan",
            "sh:lessThanOrEquals",
            "sh:hasValue",
        ];

        shape
            .constraints
            .retain(|id, _| !restrictive_types.contains(&id.as_str()));

        Ok(())
    }

    /// Widen value ranges
    fn widen_ranges(&self, shape: &mut Shape) -> Result<()> {
        // Widen ranges by relaxing constraints:
        // - Increase maxCount
        // - Decrease minCount
        // - Expand minInclusive/maxInclusive ranges

        // Relax cardinality constraints
        for (id, constraint) in shape.constraints.iter_mut() {
            match constraint {
                Constraint::MinCount(min_count) => {
                    // Decrease minCount (make less restrictive)
                    if min_count.min_count > 0 {
                        min_count.min_count = min_count.min_count.saturating_sub(1);
                        tracing::debug!(
                            "Widened minCount constraint {} to {}",
                            id,
                            min_count.min_count
                        );
                    }
                }
                Constraint::MaxCount(max_count) => {
                    // Increase maxCount (make less restrictive)
                    if max_count.max_count < u32::MAX - 10 {
                        max_count.max_count = max_count.max_count.saturating_add(10);
                        tracing::debug!(
                            "Widened maxCount constraint {} to {}",
                            id,
                            max_count.max_count
                        );
                    }
                }
                Constraint::MinInclusive(min_incl) => {
                    // For numeric literals, decrease the minimum
                    let lit = &min_incl.min_value;
                    if let Ok(num) = lit.value().parse::<f64>() {
                        let widened = num - (num.abs() * 0.1).max(1.0);
                        let widened_lit = oxirs_core::model::Literal::new_typed_literal(
                            widened.to_string(),
                            lit.datatype(),
                        );
                        min_incl.min_value = widened_lit;
                        tracing::debug!("Widened minInclusive constraint {} to {}", id, widened);
                    }
                }
                Constraint::MaxInclusive(max_incl) => {
                    // For numeric literals, increase the maximum
                    let lit = &max_incl.max_value;
                    if let Ok(num) = lit.value().parse::<f64>() {
                        let widened = num + (num.abs() * 0.1).max(1.0);
                        let widened_lit = oxirs_core::model::Literal::new_typed_literal(
                            widened.to_string(),
                            lit.datatype(),
                        );
                        max_incl.max_value = widened_lit;
                        tracing::debug!("Widened maxInclusive constraint {} to {}", id, widened);
                    }
                }
                _ => {
                    // Other constraints are not affected by range widening
                }
            }
        }

        Ok(())
    }

    /// Reduce cardinality requirements
    fn reduce_cardinality(&self, shape: &mut Shape) -> Result<()> {
        // Remove or relax minCount constraints
        let cardinality_ids: Vec<ConstraintComponentId> = shape
            .constraints
            .iter()
            .filter(|(id, _)| id.as_str() == "sh:minCount")
            .map(|(id, _)| id.clone())
            .collect();

        for id in cardinality_ids {
            shape.constraints.shift_remove(&id);
        }

        Ok(())
    }

    /// Make all constraints optional
    fn make_optional(&self, _shape: &mut Shape) -> Result<()> {
        // Remove all minCount constraints
        // Set all mandatory properties to optional
        Ok(())
    }
}

/// Shape specialization engine
pub struct ShapeSpecializer {
    /// Strategy to use
    strategy: SpecializationStrategy,
}

impl ShapeSpecializer {
    /// Create a new specializer
    pub fn new(strategy: SpecializationStrategy) -> Self {
        Self { strategy }
    }

    /// Specialize a shape
    pub fn specialize(
        &self,
        shape: &Shape,
        additional_constraints: Vec<Constraint>,
    ) -> Result<Shape> {
        let mut specialized = shape.clone();

        match self.strategy {
            SpecializationStrategy::AddRestrictive => {
                self.add_restrictive_constraints(&mut specialized, additional_constraints)?;
            }
            SpecializationStrategy::NarrowRanges => {
                self.narrow_ranges(&mut specialized)?;
            }
            SpecializationStrategy::IncreaseCardinality => {
                self.increase_cardinality(&mut specialized)?;
            }
            SpecializationStrategy::MakeMandatory => {
                self.make_mandatory(&mut specialized)?;
            }
            SpecializationStrategy::Custom => {}
        }

        Ok(specialized)
    }

    /// Add restrictive constraints
    fn add_restrictive_constraints(
        &self,
        shape: &mut Shape,
        constraints: Vec<Constraint>,
    ) -> Result<()> {
        for constraint in constraints {
            // Generate ID for constraint
            let id = ConstraintComponentId::new(format!(
                "specialized_constraint_{}",
                shape.constraints.len()
            ));
            shape.constraints.insert(id, constraint);
        }
        Ok(())
    }

    /// Narrow value ranges
    fn narrow_ranges(&self, shape: &mut Shape) -> Result<()> {
        // Narrow ranges by making constraints more restrictive:
        // - Decrease maxCount
        // - Increase minCount
        // - Reduce minInclusive/maxInclusive ranges (make value ranges more specific)

        // Make cardinality constraints more restrictive
        for (id, constraint) in shape.constraints.iter_mut() {
            match constraint {
                Constraint::MinCount(min_count) => {
                    // Increase minCount (make more restrictive)
                    if min_count.min_count < u32::MAX - 1 {
                        min_count.min_count = min_count.min_count.saturating_add(1);
                        tracing::debug!(
                            "Narrowed minCount constraint {} to {}",
                            id,
                            min_count.min_count
                        );
                    }
                }
                Constraint::MaxCount(max_count) => {
                    // Decrease maxCount (make more restrictive)
                    if max_count.max_count > 1 {
                        max_count.max_count = max_count.max_count.saturating_sub(1);
                        tracing::debug!(
                            "Narrowed maxCount constraint {} to {}",
                            id,
                            max_count.max_count
                        );
                    }
                }
                Constraint::MinInclusive(min_incl) => {
                    // For numeric literals, increase the minimum (narrow from below)
                    let lit = &min_incl.min_value;
                    if let Ok(num) = lit.value().parse::<f64>() {
                        let narrowed = num + (num.abs() * 0.1).max(1.0);
                        let narrowed_lit = oxirs_core::model::Literal::new_typed_literal(
                            narrowed.to_string(),
                            lit.datatype(),
                        );
                        min_incl.min_value = narrowed_lit;
                        tracing::debug!("Narrowed minInclusive constraint {} to {}", id, narrowed);
                    }
                }
                Constraint::MaxInclusive(max_incl) => {
                    // For numeric literals, decrease the maximum (narrow from above)
                    let lit = &max_incl.max_value;
                    if let Ok(num) = lit.value().parse::<f64>() {
                        let narrowed = num - (num.abs() * 0.1).max(1.0);
                        let narrowed_lit = oxirs_core::model::Literal::new_typed_literal(
                            narrowed.to_string(),
                            lit.datatype(),
                        );
                        max_incl.max_value = narrowed_lit;
                        tracing::debug!("Narrowed maxInclusive constraint {} to {}", id, narrowed);
                    }
                }
                _ => {
                    // Other constraints are not affected by range narrowing
                }
            }
        }

        Ok(())
    }

    /// Increase cardinality requirements
    fn increase_cardinality(&self, _shape: &mut Shape) -> Result<()> {
        // Add or increase minCount constraints
        // This would require analyzing property usage patterns
        Ok(())
    }

    /// Make constraints mandatory
    fn make_mandatory(&self, _shape: &mut Shape) -> Result<()> {
        // Set minCount = 1 for all property shapes
        // Remove optional flags
        Ok(())
    }
}

/// Shape merge engine
pub struct ShapeMerger {
    /// Strategy to use
    strategy: MergeStrategy,
}

impl ShapeMerger {
    /// Create a new merger
    pub fn new(strategy: MergeStrategy) -> Self {
        Self { strategy }
    }

    /// Merge multiple shapes into one
    pub fn merge(&self, shapes: &[Shape]) -> Result<Shape> {
        if shapes.is_empty() {
            return Err(ShaclError::Configuration(
                "Cannot merge empty shape list".to_string(),
            ));
        }

        if shapes.len() == 1 {
            return Ok(shapes[0].clone());
        }

        // Create base merged shape
        let merged_id = ShapeId::new(format!(
            "merged_{}",
            shapes
                .iter()
                .map(|s| s.id.as_str())
                .collect::<Vec<_>>()
                .join("_")
        ));

        let mut merged = Shape::new(merged_id, shapes[0].shape_type.clone());

        // Merge based on strategy
        match self.strategy {
            MergeStrategy::Union => self.merge_union(shapes, &mut merged)?,
            MergeStrategy::Intersection => self.merge_intersection(shapes, &mut merged)?,
            MergeStrategy::MostRestrictive => self.merge_most_restrictive(shapes, &mut merged)?,
            MergeStrategy::LeastRestrictive => self.merge_least_restrictive(shapes, &mut merged)?,
            MergeStrategy::Custom => {}
        }

        Ok(merged)
    }

    /// Merge using union strategy (combine all constraints)
    fn merge_union(&self, shapes: &[Shape], merged: &mut Shape) -> Result<()> {
        for shape in shapes {
            // Merge constraints
            for (id, constraint) in &shape.constraints {
                if !merged.constraints.contains_key(id) {
                    merged.constraints.insert(id.clone(), constraint.clone());
                }
            }

            // Merge targets
            for target in &shape.targets {
                if !merged.targets.contains(target) {
                    merged.targets.push(target.clone());
                }
            }
        }

        Ok(())
    }

    /// Merge using intersection strategy (only common constraints)
    fn merge_intersection(&self, shapes: &[Shape], merged: &mut Shape) -> Result<()> {
        if shapes.is_empty() {
            return Ok(());
        }

        // Find common constraint IDs
        let first_constraints: HashSet<_> = shapes[0].constraints.keys().collect();

        let common_ids: HashSet<&ConstraintComponentId> =
            shapes[1..].iter().fold(first_constraints, |acc, shape| {
                let shape_ids: HashSet<_> = shape.constraints.keys().collect();
                acc.intersection(&shape_ids).copied().collect()
            });

        // Add common constraints to merged shape
        for id in common_ids {
            if let Some(constraint) = shapes[0].constraints.get(id) {
                merged.constraints.insert(id.clone(), constraint.clone());
            }
        }

        Ok(())
    }

    /// Merge keeping most restrictive constraints
    fn merge_most_restrictive(&self, shapes: &[Shape], merged: &mut Shape) -> Result<()> {
        // For each constraint type, keep the most restrictive version
        // Collect all constraints from all shapes
        let mut constraint_groups: HashMap<String, Vec<Constraint>> = HashMap::new();

        for shape in shapes {
            for (id, constraint) in &shape.constraints {
                let constraint_type = self.get_constraint_type(constraint);
                constraint_groups
                    .entry(constraint_type)
                    .or_default()
                    .push(constraint.clone());
            }
        }

        // For each constraint type, keep the most restrictive
        for (constraint_type, constraints) in constraint_groups {
            if let Some(most_restrictive) = self.find_most_restrictive(&constraints)? {
                let id = ConstraintComponentId::new(format!("merged_{}", constraint_type));
                merged.constraints.insert(id, most_restrictive);
            }
        }

        Ok(())
    }

    /// Find the most restrictive constraint from a list
    fn find_most_restrictive(&self, constraints: &[Constraint]) -> Result<Option<Constraint>> {
        if constraints.is_empty() {
            return Ok(None);
        }

        let mut most_restrictive = constraints[0].clone();

        for constraint in &constraints[1..] {
            if self.is_more_restrictive(constraint, &most_restrictive)? {
                most_restrictive = constraint.clone();
            }
        }

        Ok(Some(most_restrictive))
    }

    /// Determine if constraint A is more restrictive than constraint B
    fn is_more_restrictive(&self, a: &Constraint, b: &Constraint) -> Result<bool> {
        match (a, b) {
            // MinCount: higher value is more restrictive
            (Constraint::MinCount(a_min), Constraint::MinCount(b_min)) => {
                Ok(a_min.min_count > b_min.min_count)
            }
            // MaxCount: lower value is more restrictive
            (Constraint::MaxCount(a_max), Constraint::MaxCount(b_max)) => {
                Ok(a_max.max_count < b_max.max_count)
            }
            // MinLength: higher value is more restrictive
            (Constraint::MinLength(a_min), Constraint::MinLength(b_min)) => {
                Ok(a_min.min_length > b_min.min_length)
            }
            // MaxLength: lower value is more restrictive
            (Constraint::MaxLength(a_max), Constraint::MaxLength(b_max)) => {
                Ok(a_max.max_length < b_max.max_length)
            }
            // Pattern: having a pattern is more restrictive than not having one
            (Constraint::Pattern(_), Constraint::Pattern(_)) => {
                // Both have patterns, consider them equally restrictive
                Ok(false)
            }
            // For other constraints, default to false (keep first)
            _ => Ok(false),
        }
    }

    /// Get constraint type identifier for grouping
    fn get_constraint_type(&self, constraint: &Constraint) -> String {
        match constraint {
            Constraint::MinCount(_) => "minCount".to_string(),
            Constraint::MaxCount(_) => "maxCount".to_string(),
            Constraint::MinLength(_) => "minLength".to_string(),
            Constraint::MaxLength(_) => "maxLength".to_string(),
            Constraint::Pattern(_) => "pattern".to_string(),
            Constraint::MinInclusive(_) => "minInclusive".to_string(),
            Constraint::MaxInclusive(_) => "maxInclusive".to_string(),
            Constraint::MinExclusive(_) => "minExclusive".to_string(),
            Constraint::MaxExclusive(_) => "maxExclusive".to_string(),
            Constraint::Datatype(_) => "datatype".to_string(),
            Constraint::NodeKind(_) => "nodeKind".to_string(),
            Constraint::Class(_) => "class".to_string(),
            Constraint::In(_) => "in".to_string(),
            Constraint::LanguageIn(_) => "languageIn".to_string(),
            Constraint::UniqueLang(_) => "uniqueLang".to_string(),
            _ => "other".to_string(),
        }
    }

    /// Merge keeping least restrictive constraints
    fn merge_least_restrictive(&self, shapes: &[Shape], merged: &mut Shape) -> Result<()> {
        // For each constraint type, keep the least restrictive version
        // Collect all constraints from all shapes
        let mut constraint_groups: HashMap<String, Vec<Constraint>> = HashMap::new();

        for shape in shapes {
            for (id, constraint) in &shape.constraints {
                let constraint_type = self.get_constraint_type(constraint);
                constraint_groups
                    .entry(constraint_type)
                    .or_default()
                    .push(constraint.clone());
            }
        }

        // For each constraint type, keep the least restrictive
        for (constraint_type, constraints) in constraint_groups {
            if let Some(least_restrictive) = self.find_least_restrictive(&constraints)? {
                let id = ConstraintComponentId::new(format!("merged_{}", constraint_type));
                merged.constraints.insert(id, least_restrictive);
            }
        }

        Ok(())
    }

    /// Find the least restrictive constraint from a list
    fn find_least_restrictive(&self, constraints: &[Constraint]) -> Result<Option<Constraint>> {
        if constraints.is_empty() {
            return Ok(None);
        }

        let mut least_restrictive = constraints[0].clone();

        for constraint in &constraints[1..] {
            // If the current constraint is more restrictive, keep the old one
            // Otherwise, update to the new one (which is less restrictive)
            if !self.is_more_restrictive(constraint, &least_restrictive)? {
                // constraint is less restrictive than least_restrictive, or equally restrictive
                if self.is_more_restrictive(&least_restrictive, constraint)? {
                    // least_restrictive is more restrictive than constraint
                    // So constraint is less restrictive - use it
                    least_restrictive = constraint.clone();
                }
            }
        }

        Ok(Some(least_restrictive))
    }
}

/// Shape refactoring engine
pub struct ShapeRefactorer {
    /// Configuration
    config: RefactoringConfig,
}

/// Configuration for shape refactoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefactoringConfig {
    /// Extract common patterns into parent shapes
    pub extract_common_patterns: bool,
    /// Split large shapes into smaller ones
    pub split_large_shapes: bool,
    /// Maximum constraints per shape
    pub max_constraints_per_shape: usize,
    /// Inline small referenced shapes
    pub inline_small_shapes: bool,
}

impl Default for RefactoringConfig {
    fn default() -> Self {
        Self {
            extract_common_patterns: true,
            split_large_shapes: true,
            max_constraints_per_shape: 20,
            inline_small_shapes: false,
        }
    }
}

impl ShapeRefactorer {
    /// Create a new refactorer
    pub fn new(config: RefactoringConfig) -> Self {
        Self { config }
    }

    /// Refactor a set of shapes
    pub fn refactor(&self, shapes: &[Shape]) -> Result<Vec<Shape>> {
        let mut refactored = shapes.to_vec();

        if self.config.extract_common_patterns {
            refactored = self.extract_common_patterns(&refactored)?;
        }

        if self.config.split_large_shapes {
            refactored = self.split_large_shapes(&refactored)?;
        }

        Ok(refactored)
    }

    /// Extract common patterns into parent shapes
    fn extract_common_patterns(&self, shapes: &[Shape]) -> Result<Vec<Shape>> {
        // Find common constraints across shapes
        let mut pattern_map: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

        for (idx, shape) in shapes.iter().enumerate() {
            let constraint_ids: Vec<String> = shape
                .constraints
                .keys()
                .map(|id| id.as_str().to_string())
                .collect();

            pattern_map.entry(constraint_ids).or_default().push(idx);
        }

        // If a pattern appears in multiple shapes, extract it
        let common_patterns: Vec<_> = pattern_map
            .iter()
            .filter(|(_, indices)| indices.len() > 1)
            .collect();

        if common_patterns.is_empty() {
            return Ok(shapes.to_vec());
        }

        // Create parent shapes for common patterns
        let mut result_shapes = Vec::new();
        let mut created_parents = HashMap::new();

        for (constraint_ids, shape_indices) in common_patterns {
            // Only create parent if at least 2 shapes share the pattern
            if shape_indices.len() < 2 {
                continue;
            }

            // Get the constraints from the first shape in the pattern
            let first_shape = &shapes[shape_indices[0]];
            let mut common_constraints = IndexMap::new();

            // Extract common constraints
            for constraint_id in constraint_ids {
                let cid = ConstraintComponentId(constraint_id.clone());
                if let Some(constraint) = first_shape.constraints.get(&cid) {
                    common_constraints.insert(cid, constraint.clone());
                }
            }

            // Only create parent if we have constraints to extract
            if common_constraints.is_empty() {
                continue;
            }

            // Create unique parent shape ID based on pattern
            use scirs2_core::random::{rng, Rng};
            let parent_id = ShapeId(format!("common-pattern-{}", rng().random::<u64>()));

            // Create parent shape with common constraints
            let parent_shape = Shape {
                id: parent_id.clone(),
                shape_type: first_shape.shape_type.clone(),
                targets: Vec::new(), // Parent has no targets
                path: None,
                constraints: common_constraints,
                deactivated: false,
                label: Some(format!(
                    "Common pattern extracted from {} shapes",
                    shape_indices.len()
                )),
                description: Some(
                    "Automatically generated parent shape for common constraint pattern"
                        .to_string(),
                ),
                groups: vec!["auto-generated".to_string()],
                order: None,
                severity: Severity::Info, // Lower severity for parent
                messages: IndexMap::new(),
                extends: Vec::new(),
                priority: None,
                metadata: ShapeMetadata {
                    author: None,
                    created: None,
                    modified: None,
                    version: None,
                    license: None,
                    tags: vec!["auto-generated".to_string(), "common-pattern".to_string()],
                    custom: HashMap::new(),
                },
            };

            result_shapes.push(parent_shape);
            created_parents.insert(constraint_ids.clone(), parent_id);
        }

        // Process all shapes and update children to extend parents
        for (idx, shape) in shapes.iter().enumerate() {
            let mut child_shape = shape.clone();

            // Check if this shape has a common pattern
            let constraint_ids: Vec<String> = shape
                .constraints
                .keys()
                .map(|id| id.as_str().to_string())
                .collect();

            if let Some(parent_id) = created_parents.get(&constraint_ids) {
                // Add parent to extends
                if !child_shape.extends.contains(parent_id) {
                    child_shape.extends.push(parent_id.clone());
                }

                // Remove common constraints (they're inherited from parent)
                for constraint_id in &constraint_ids {
                    let cid = ConstraintComponentId(constraint_id.clone());
                    child_shape.constraints.shift_remove(&cid);
                }

                tracing::debug!(
                    "Shape {} now extends parent {} with {} common constraints",
                    shape.id.as_str(),
                    parent_id.as_str(),
                    constraint_ids.len()
                );
            }

            result_shapes.push(child_shape);
        }

        tracing::info!(
            "Extracted {} parent shapes for common patterns from {} shapes",
            created_parents.len(),
            shapes.len()
        );

        Ok(result_shapes)
    }

    /// Split large shapes into smaller ones
    fn split_large_shapes(&self, shapes: &[Shape]) -> Result<Vec<Shape>> {
        let mut result = Vec::new();

        for shape in shapes {
            if shape.constraints.len() > self.config.max_constraints_per_shape {
                // Split the shape
                let split_shapes = self.split_shape(shape)?;
                result.extend(split_shapes);
            } else {
                result.push(shape.clone());
            }
        }

        Ok(result)
    }

    /// Split a single shape into multiple smaller shapes
    fn split_shape(&self, shape: &Shape) -> Result<Vec<Shape>> {
        let mut splits = Vec::new();

        // Group constraints by type
        let mut grouped_constraints: HashMap<String, IndexMap<ConstraintComponentId, Constraint>> =
            HashMap::new();

        for (id, constraint) in &shape.constraints {
            let group = self.categorize_constraint(id);
            grouped_constraints
                .entry(group)
                .or_default()
                .insert(id.clone(), constraint.clone());
        }

        // Create a shape for each group
        for (group_name, constraints) in grouped_constraints {
            let split_id = ShapeId::new(format!("{}_{}", shape.id.as_str(), group_name));
            let mut split_shape = Shape::new(split_id, shape.shape_type.clone());

            split_shape.constraints = constraints;
            split_shape.targets = shape.targets.clone();
            split_shape.severity = shape.severity;

            splits.push(split_shape);
        }

        // If only one group, return original
        if splits.len() <= 1 {
            return Ok(vec![shape.clone()]);
        }

        Ok(splits)
    }

    /// Categorize a constraint by its type
    fn categorize_constraint(&self, id: &ConstraintComponentId) -> String {
        let id_str = id.as_str();

        if id_str.contains("minCount") || id_str.contains("maxCount") {
            "cardinality".to_string()
        } else if id_str.contains("datatype") || id_str.contains("class") {
            "type".to_string()
        } else if id_str.contains("min") || id_str.contains("max") {
            "range".to_string()
        } else if id_str.contains("pattern") || id_str.contains("regex") {
            "pattern".to_string()
        } else {
            "other".to_string()
        }
    }
}

/// Shape evolution tracker
///
/// Tracks changes to shapes over time, enabling versioning and rollback
pub struct ShapeEvolutionTracker {
    /// History of shape changes
    history: HashMap<ShapeId, Vec<ShapeVersion>>,
    /// Maximum history size per shape
    max_history: usize,
}

/// A versioned snapshot of a shape
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeVersion {
    /// Version number
    pub version: u32,
    /// Shape snapshot
    pub shape: Shape,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Change description
    pub description: String,
    /// Operation type
    pub operation: ShapeOperation,
    /// Metrics about the change
    pub metrics: EvolutionMetrics,
}

/// Type of operation performed on a shape
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeOperation {
    /// Initial creation
    Created,
    /// Generalized (made less restrictive)
    Generalized,
    /// Specialized (made more restrictive)
    Specialized,
    /// Merged with other shapes
    Merged,
    /// Refactored/restructured
    Refactored,
    /// Manual modification
    Modified,
}

/// Metrics about shape evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMetrics {
    /// Number of constraints added
    pub constraints_added: usize,
    /// Number of constraints removed
    pub constraints_removed: usize,
    /// Number of constraints modified
    pub constraints_modified: usize,
    /// Restrictiveness delta (-1.0 to 1.0, negative = more permissive)
    pub restrictiveness_delta: f64,
    /// Complexity score
    pub complexity_score: f64,
}

impl ShapeEvolutionTracker {
    /// Create a new evolution tracker
    pub fn new(max_history: usize) -> Self {
        Self {
            history: HashMap::new(),
            max_history,
        }
    }

    /// Create with default settings
    pub fn default_config() -> Self {
        Self::new(100) // Keep last 100 versions
    }

    /// Record a shape change
    pub fn record_change(
        &mut self,
        shape: &Shape,
        operation: ShapeOperation,
        description: String,
        previous_shape: Option<&Shape>,
    ) -> Result<()> {
        // Compute metrics first before borrowing history
        let metrics = if let Some(prev) = previous_shape {
            self.compute_evolution_metrics(prev, shape)
        } else {
            EvolutionMetrics {
                constraints_added: shape.constraints.len(),
                constraints_removed: 0,
                constraints_modified: 0,
                restrictiveness_delta: 0.0,
                complexity_score: Self::compute_complexity_score_static(shape),
            }
        };

        // Now borrow history mutably
        let version_history = self.history.entry(shape.id.clone()).or_default();

        let version = if version_history.is_empty() {
            1
        } else {
            version_history.last().unwrap().version + 1
        };

        let version_entry = ShapeVersion {
            version,
            shape: shape.clone(),
            timestamp: chrono::Utc::now(),
            description,
            operation,
            metrics,
        };

        version_history.push(version_entry);

        // Prune old versions if exceeding max_history
        if version_history.len() > self.max_history {
            version_history.remove(0);
        }

        Ok(())
    }

    /// Get version history for a shape
    pub fn get_history(&self, shape_id: &ShapeId) -> Option<&Vec<ShapeVersion>> {
        self.history.get(shape_id)
    }

    /// Get a specific version of a shape
    pub fn get_version(&self, shape_id: &ShapeId, version: u32) -> Option<&Shape> {
        self.history
            .get(shape_id)?
            .iter()
            .find(|v| v.version == version)
            .map(|v| &v.shape)
    }

    /// Get the latest version of a shape
    pub fn get_latest(&self, shape_id: &ShapeId) -> Option<&Shape> {
        self.history.get(shape_id)?.last().map(|v| &v.shape)
    }

    /// Rollback to a previous version
    pub fn rollback(&mut self, shape_id: &ShapeId, target_version: u32) -> Result<Shape> {
        let history = self.history.get(shape_id).ok_or_else(|| {
            ShaclError::Configuration(format!("No history for shape: {}", shape_id.as_str()))
        })?;

        let target = history
            .iter()
            .find(|v| v.version == target_version)
            .ok_or_else(|| {
                ShaclError::Configuration(format!(
                    "Version {} not found for shape: {}",
                    target_version,
                    shape_id.as_str()
                ))
            })?;

        Ok(target.shape.clone())
    }

    /// Compute evolution metrics between two shape versions
    fn compute_evolution_metrics(&self, old: &Shape, new: &Shape) -> EvolutionMetrics {
        let old_ids: HashSet<_> = old.constraints.keys().collect();
        let new_ids: HashSet<_> = new.constraints.keys().collect();

        let added = new_ids.difference(&old_ids).count();
        let removed = old_ids.difference(&new_ids).count();

        // Count modified constraints (same ID but different content)
        let modified = old_ids
            .intersection(&new_ids)
            .filter(|&&id| old.constraints.get(id).unwrap() != new.constraints.get(id).unwrap())
            .count();

        // Compute restrictiveness delta
        let old_restrictiveness = self.compute_restrictiveness_score(old);
        let new_restrictiveness = self.compute_restrictiveness_score(new);
        let delta = new_restrictiveness - old_restrictiveness;

        EvolutionMetrics {
            constraints_added: added,
            constraints_removed: removed,
            constraints_modified: modified,
            restrictiveness_delta: delta,
            complexity_score: Self::compute_complexity_score_static(new),
        }
    }

    /// Compute restrictiveness score for a shape (0.0 = permissive, 1.0 = restrictive)
    fn compute_restrictiveness_score(&self, shape: &Shape) -> f64 {
        Self::compute_restrictiveness_score_static(shape)
    }

    /// Static version of compute_restrictiveness_score
    fn compute_restrictiveness_score_static(shape: &Shape) -> f64 {
        let mut score = 0.0;
        let num_constraints = shape.constraints.len() as f64;

        if num_constraints == 0.0 {
            return 0.0;
        }

        // Count restrictive constraint types
        for (id, _) in &shape.constraints {
            let id_str = id.as_str();
            if id_str.contains("equals")
                || id_str.contains("disjoint")
                || id_str.contains("hasValue")
                || id_str.contains("minCount")
            {
                score += 1.0;
            }
        }

        score / num_constraints
    }

    /// Compute complexity score for a shape
    fn compute_complexity_score(&self, shape: &Shape) -> f64 {
        Self::compute_complexity_score_static(shape)
    }

    /// Static version of compute_complexity_score
    fn compute_complexity_score_static(shape: &Shape) -> f64 {
        // Simple complexity metric: number of constraints + targets
        (shape.constraints.len() + shape.targets.len()) as f64
    }

    /// Get evolution statistics for a shape
    pub fn get_evolution_stats(&self, shape_id: &ShapeId) -> Option<ShapeEvolutionStats> {
        let history = self.history.get(shape_id)?;

        if history.is_empty() {
            return None;
        }

        let total_changes = history.len();
        let generalizations = history
            .iter()
            .filter(|v| v.operation == ShapeOperation::Generalized)
            .count();
        let specializations = history
            .iter()
            .filter(|v| v.operation == ShapeOperation::Specialized)
            .count();

        let avg_restrictiveness = history
            .iter()
            .map(|v| self.compute_restrictiveness_score(&v.shape))
            .sum::<f64>()
            / total_changes as f64;

        let current_complexity = self.compute_complexity_score(&history.last().unwrap().shape);

        Some(ShapeEvolutionStats {
            total_versions: total_changes,
            generalizations,
            specializations,
            merges: history
                .iter()
                .filter(|v| v.operation == ShapeOperation::Merged)
                .count(),
            refactorings: history
                .iter()
                .filter(|v| v.operation == ShapeOperation::Refactored)
                .count(),
            avg_restrictiveness_score: avg_restrictiveness,
            current_complexity_score: current_complexity,
        })
    }
}

impl Default for ShapeEvolutionTracker {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Statistics about shape evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeEvolutionStats {
    /// Total number of versions
    pub total_versions: usize,
    /// Number of generalization operations
    pub generalizations: usize,
    /// Number of specialization operations
    pub specializations: usize,
    /// Number of merge operations
    pub merges: usize,
    /// Number of refactoring operations
    pub refactorings: usize,
    /// Average restrictiveness across all versions
    pub avg_restrictiveness_score: f64,
    /// Current complexity score
    pub current_complexity_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ShapeType;

    #[test]
    fn test_shape_generalizer() {
        let shape_id = ShapeId::new("test:shape".to_string());
        let shape = Shape::new(shape_id, ShapeType::NodeShape);

        let generalizer = ShapeGeneralizer::new(GeneralizationStrategy::RemoveRestrictive);
        let result = generalizer.generalize(&shape);

        assert!(result.is_ok());
    }

    #[test]
    fn test_shape_specializer() {
        let shape_id = ShapeId::new("test:shape".to_string());
        let shape = Shape::new(shape_id, ShapeType::NodeShape);

        let specializer = ShapeSpecializer::new(SpecializationStrategy::IncreaseCardinality);
        let result = specializer.specialize(&shape, vec![]);

        assert!(result.is_ok());
    }

    #[test]
    fn test_shape_merger_empty() {
        let merger = ShapeMerger::new(MergeStrategy::Union);
        let result = merger.merge(&[]);

        assert!(result.is_err());
    }

    #[test]
    fn test_shape_merger_single() {
        let shape_id = ShapeId::new("test:shape".to_string());
        let shape = Shape::new(shape_id, ShapeType::NodeShape);

        let merger = ShapeMerger::new(MergeStrategy::Union);
        let result = merger.merge(std::slice::from_ref(&shape));

        assert!(result.is_ok());
        assert_eq!(result.unwrap().id, shape.id);
    }

    #[test]
    fn test_shape_merger_union() {
        let shape1 = Shape::new(
            ShapeId::new("test:shape1".to_string()),
            ShapeType::NodeShape,
        );
        let shape2 = Shape::new(
            ShapeId::new("test:shape2".to_string()),
            ShapeType::NodeShape,
        );

        let merger = ShapeMerger::new(MergeStrategy::Union);
        let result = merger.merge(&[shape1, shape2]);

        assert!(result.is_ok());
    }

    #[test]
    fn test_shape_refactorer() {
        let config = RefactoringConfig::default();
        let refactorer = ShapeRefactorer::new(config);

        let shape = Shape::new(ShapeId::new("test:shape".to_string()), ShapeType::NodeShape);

        let result = refactorer.refactor(&[shape]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_evolution_tracker_creation() {
        let tracker = ShapeEvolutionTracker::new(50);
        assert_eq!(tracker.max_history, 50);
    }

    #[test]
    fn test_evolution_tracker_record_change() {
        let mut tracker = ShapeEvolutionTracker::default();
        let shape = Shape::new(ShapeId::new("test:shape".to_string()), ShapeType::NodeShape);

        let result = tracker.record_change(
            &shape,
            ShapeOperation::Created,
            "Initial creation".to_string(),
            None,
        );

        assert!(result.is_ok());
        assert!(tracker.get_history(&shape.id).is_some());
    }

    #[test]
    fn test_evolution_tracker_versioning() {
        let mut tracker = ShapeEvolutionTracker::default();
        let shape_id = ShapeId::new("test:shape".to_string());
        let mut shape = Shape::new(shape_id.clone(), ShapeType::NodeShape);

        // Version 1
        tracker
            .record_change(&shape, ShapeOperation::Created, "v1".to_string(), None)
            .unwrap();

        // Version 2
        shape.severity = crate::Severity::Violation;
        let prev = tracker.get_latest(&shape_id).cloned();
        tracker
            .record_change(
                &shape,
                ShapeOperation::Modified,
                "v2".to_string(),
                prev.as_ref(),
            )
            .unwrap();

        let history = tracker.get_history(&shape_id).unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].version, 1);
        assert_eq!(history[1].version, 2);
    }

    #[test]
    fn test_evolution_tracker_rollback() {
        let mut tracker = ShapeEvolutionTracker::default();
        let shape_id = ShapeId::new("test:shape".to_string());
        let shape = Shape::new(shape_id.clone(), ShapeType::NodeShape);

        tracker
            .record_change(&shape, ShapeOperation::Created, "v1".to_string(), None)
            .unwrap();

        let rolled_back = tracker.rollback(&shape_id, 1);
        assert!(rolled_back.is_ok());
        assert_eq!(rolled_back.unwrap().id, shape_id);
    }

    #[test]
    fn test_evolution_stats() {
        let mut tracker = ShapeEvolutionTracker::default();
        let shape_id = ShapeId::new("test:shape".to_string());
        let shape = Shape::new(shape_id.clone(), ShapeType::NodeShape);

        tracker
            .record_change(&shape, ShapeOperation::Created, "created".to_string(), None)
            .unwrap();

        tracker
            .record_change(
                &shape,
                ShapeOperation::Generalized,
                "generalized".to_string(),
                Some(&shape),
            )
            .unwrap();

        let stats = tracker.get_evolution_stats(&shape_id).unwrap();
        assert_eq!(stats.total_versions, 2);
        assert_eq!(stats.generalizations, 1);
    }
}
