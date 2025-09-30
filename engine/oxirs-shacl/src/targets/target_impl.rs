//! Target implementation for SHACL target selection
//!
//! This module contains the implementation of the Target enum and its methods.

use crate::targets::types::*;
use oxirs_core::model::{NamedNode, Term};

impl Target {
    /// Create a class target
    pub fn class(class_iri: NamedNode) -> Self {
        Target::Class(class_iri)
    }

    /// Create a node target
    pub fn node(node: Term) -> Self {
        Target::Node(node)
    }

    /// Create an objects-of target
    pub fn objects_of(property: NamedNode) -> Self {
        Target::ObjectsOf(property)
    }

    /// Create a subjects-of target
    pub fn subjects_of(property: NamedNode) -> Self {
        Target::SubjectsOf(property)
    }

    /// Create a SPARQL target
    pub fn sparql(query: String, prefixes: Option<String>) -> Self {
        Target::Sparql(SparqlTarget { query, prefixes })
    }

    /// Create an implicit target
    pub fn implicit(class_iri: NamedNode) -> Self {
        Target::Implicit(class_iri)
    }

    /// Create a union target combining multiple targets
    pub fn union(targets: Vec<Target>) -> Self {
        Target::Union(TargetUnion {
            targets,
            optimization_hints: None,
        })
    }

    /// Create a union target with optimization hints
    pub fn union_with_hints(targets: Vec<Target>, hints: UnionOptimizationHints) -> Self {
        Target::Union(TargetUnion {
            targets,
            optimization_hints: Some(hints),
        })
    }

    /// Create an intersection target requiring all targets to match
    pub fn intersection(targets: Vec<Target>) -> Self {
        Target::Intersection(TargetIntersection {
            targets,
            optimization_hints: None,
        })
    }

    /// Create an intersection target with optimization hints
    pub fn intersection_with_hints(
        targets: Vec<Target>,
        hints: IntersectionOptimizationHints,
    ) -> Self {
        Target::Intersection(TargetIntersection {
            targets,
            optimization_hints: Some(hints),
        })
    }

    /// Create a difference target (primary minus exclusion)
    pub fn difference(primary: Target, exclusion: Target) -> Self {
        Target::Difference(TargetDifference {
            primary_target: Box::new(primary),
            exclusion_target: Box::new(exclusion),
        })
    }

    /// Create a conditional target with a condition
    pub fn conditional(base: Target, condition: TargetCondition) -> Self {
        Target::Conditional(ConditionalTarget {
            base_target: Box::new(base),
            condition,
            context: None,
        })
    }

    /// Create a conditional target with context
    pub fn conditional_with_context(
        base: Target,
        condition: TargetCondition,
        context: TargetContext,
    ) -> Self {
        Target::Conditional(ConditionalTarget {
            base_target: Box::new(base),
            condition,
            context: Some(context),
        })
    }

    /// Create a hierarchical target following relationships
    pub fn hierarchical(
        root: Target,
        relationship: HierarchicalRelationship,
        max_depth: i32,
    ) -> Self {
        Target::Hierarchical(HierarchicalTarget {
            root_target: Box::new(root),
            relationship,
            max_depth,
            include_intermediate: false,
        })
    }

    /// Create a hierarchical target with intermediate nodes included
    pub fn hierarchical_with_intermediate(
        root: Target,
        relationship: HierarchicalRelationship,
        max_depth: i32,
    ) -> Self {
        Target::Hierarchical(HierarchicalTarget {
            root_target: Box::new(root),
            relationship,
            max_depth,
            include_intermediate: true,
        })
    }

    /// Create a path-based target following property paths
    pub fn path_based(
        start: Target,
        path: crate::paths::PropertyPath,
        direction: PathDirection,
    ) -> Self {
        Target::PathBased(PathBasedTarget {
            start_target: Box::new(start),
            path,
            direction,
            filters: Vec::new(),
        })
    }

    /// Create a path-based target with filters
    pub fn path_based_with_filters(
        start: Target,
        path: crate::paths::PropertyPath,
        direction: PathDirection,
        filters: Vec<PathFilter>,
    ) -> Self {
        Target::PathBased(PathBasedTarget {
            start_target: Box::new(start),
            path,
            direction,
            filters,
        })
    }

    /// Check if this target requires complex evaluation
    pub fn is_complex(&self) -> bool {
        matches!(
            self,
            Target::Union(_)
                | Target::Intersection(_)
                | Target::Difference(_)
                | Target::Conditional(_)
                | Target::Hierarchical(_)
                | Target::PathBased(_)
        )
    }

    /// Get the estimated complexity of this target (0-10 scale)
    pub fn complexity_score(&self) -> u8 {
        match self {
            Target::Node(_) => 1,
            Target::Class(_) | Target::Implicit(_) => 2,
            Target::ObjectsOf(_) | Target::SubjectsOf(_) => 3,
            Target::Sparql(_) => 4,
            Target::Union(u) => 5 + (u.targets.len() as u8).min(3),
            Target::Intersection(i) => 6 + (i.targets.len() as u8).min(3),
            Target::Difference(_) => 7,
            Target::Conditional(_) => 8,
            Target::Hierarchical(_) => 9,
            Target::PathBased(_) => 10,
        }
    }

    /// Get all nested targets within complex targets
    pub fn nested_targets(&self) -> Vec<&Target> {
        match self {
            Target::Union(u) => u.targets.iter().collect(),
            Target::Intersection(i) => i.targets.iter().collect(),
            Target::Difference(d) => vec![&d.primary_target, &d.exclusion_target],
            Target::Conditional(c) => vec![&c.base_target],
            Target::Hierarchical(h) => vec![&h.root_target],
            Target::PathBased(p) => vec![&p.start_target],
            _ => vec![],
        }
    }
}
