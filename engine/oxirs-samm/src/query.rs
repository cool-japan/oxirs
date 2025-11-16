//! Model Query Utilities
//!
//! This module provides powerful utilities for querying and analyzing SAMM Aspect Models.
//! It enables sophisticated model introspection, dependency analysis, and element discovery.
//!
//! # Features
//!
//! - **Element Discovery**: Find properties, characteristics, entities, operations by criteria
//! - **Dependency Analysis**: Discover element dependencies and circular references
//! - **Type Queries**: Find all elements of specific types or with specific characteristics
//! - **Naming Queries**: Search by name patterns, URN prefixes, or namespaces
//! - **Statistical Queries**: Get counts, distributions, and complexity metrics
//!
//! # Examples
//!
//! ```rust
//! use oxirs_samm::query::ModelQuery;
//! use oxirs_samm::metamodel::Aspect;
//!
//! # fn example(aspect: &Aspect) {
//! let query = ModelQuery::new(aspect);
//!
//! // Find all properties with Collection characteristics
//! let collections = query.find_properties_with_collection_characteristic();
//!
//! // Find all referenced entities
//! let entities = query.find_all_referenced_entities();
//!
//! // Get model complexity metrics
//! let metrics = query.complexity_metrics();
//! println!("Total properties: {}", metrics.total_properties);
//! println!("Max nesting depth: {}", metrics.max_nesting_depth);
//! # }
//! ```

use crate::metamodel::{Aspect, CharacteristicKind, Entity, ModelElement, Operation, Property};
use std::collections::{HashMap, HashSet, VecDeque};

/// Query builder for SAMM Aspect Models
///
/// Provides a fluent API for querying and analyzing SAMM models.
pub struct ModelQuery<'a> {
    aspect: &'a Aspect,
}

/// Complexity metrics for a SAMM model
///
/// Provides quantitative measures of model complexity.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexityMetrics {
    /// Total number of properties
    pub total_properties: usize,
    /// Total number of entities
    pub total_entities: usize,
    /// Total number of operations
    pub total_operations: usize,
    /// Maximum nesting depth of entities
    pub max_nesting_depth: usize,
    /// Number of optional properties
    pub optional_properties: usize,
    /// Number of collection properties
    pub collection_properties: usize,
    /// Number of circular references detected
    pub circular_references: usize,
}

/// Dependency information for a model element
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dependency {
    /// URN of the source element
    pub from: String,
    /// URN of the target element
    pub to: String,
    /// Type of dependency (e.g., "property", "characteristic", "entity")
    pub dependency_type: String,
}

impl<'a> ModelQuery<'a> {
    /// Creates a new query builder for the given aspect
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::query::ModelQuery;
    /// # use oxirs_samm::metamodel::Aspect;
    /// # fn example(aspect: &Aspect) {
    /// let query = ModelQuery::new(aspect);
    /// # }
    /// ```
    pub fn new(aspect: &'a Aspect) -> Self {
        Self { aspect }
    }

    /// Finds all properties with Collection, List, Set, or SortedSet characteristics
    ///
    /// # Returns
    ///
    /// Vector of properties that have collection-type characteristics
    pub fn find_properties_with_collection_characteristic(&self) -> Vec<&Property> {
        self.aspect
            .properties()
            .iter()
            .filter(|prop| {
                if let Some(ref characteristic) = prop.characteristic {
                    matches!(
                        characteristic.kind(),
                        CharacteristicKind::Collection { .. }
                            | CharacteristicKind::List { .. }
                            | CharacteristicKind::Set { .. }
                            | CharacteristicKind::SortedSet { .. }
                    )
                } else {
                    false
                }
            })
            .collect()
    }

    /// Finds all optional properties
    ///
    /// # Returns
    ///
    /// Vector of properties marked as optional
    pub fn find_optional_properties(&self) -> Vec<&Property> {
        self.aspect
            .properties()
            .iter()
            .filter(|prop| prop.optional)
            .collect()
    }

    /// Finds all required (non-optional) properties
    ///
    /// # Returns
    ///
    /// Vector of properties that are required
    pub fn find_required_properties(&self) -> Vec<&Property> {
        self.aspect
            .properties()
            .iter()
            .filter(|prop| !prop.optional)
            .collect()
    }

    /// Finds properties by URN namespace
    ///
    /// # Arguments
    ///
    /// * `namespace` - The URN namespace to match (e.g., "urn:samm:org.example:1.0.0")
    ///
    /// # Returns
    ///
    /// Vector of properties in the specified namespace
    pub fn find_properties_in_namespace(&self, namespace: &str) -> Vec<&Property> {
        self.aspect
            .properties()
            .iter()
            .filter(|prop| {
                // Extract namespace from URN (before the #)
                if let Some(prop_ns) = prop.urn().rsplit_once('#').map(|(ns, _)| ns) {
                    if let Some(target_ns) = namespace.rsplit_once('#').map(|(ns, _)| ns) {
                        prop_ns == target_ns
                    } else {
                        prop_ns == namespace
                    }
                } else {
                    false
                }
            })
            .collect()
    }

    /// Finds all properties with specific characteristic type
    ///
    /// # Arguments
    ///
    /// * `predicate` - Function to test characteristic kind
    ///
    /// # Returns
    ///
    /// Vector of properties matching the predicate
    pub fn find_properties_by_characteristic<F>(&self, predicate: F) -> Vec<&Property>
    where
        F: Fn(&CharacteristicKind) -> bool,
    {
        self.aspect
            .properties()
            .iter()
            .filter(|prop| {
                if let Some(ref characteristic) = prop.characteristic {
                    predicate(characteristic.kind())
                } else {
                    false
                }
            })
            .collect()
    }

    /// Finds all entities referenced directly or indirectly by the aspect
    ///
    /// This performs a breadth-first search through all properties and their characteristics
    /// to discover all entity references.
    ///
    /// # Returns
    ///
    /// Set of unique entity URNs referenced by the model
    pub fn find_all_referenced_entities(&self) -> HashSet<String> {
        let mut entities = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start with all properties
        for property in self.aspect.properties() {
            if !visited.contains(property.urn()) {
                queue.push_back(property);
                visited.insert(property.urn().to_string());
            }
        }

        // BFS through properties and characteristics
        while let Some(property) = queue.pop_front() {
            if let Some(ref characteristic) = property.characteristic {
                // Check for entity references in characteristic
                match characteristic.kind() {
                    CharacteristicKind::SingleEntity { .. } => {
                        if let Some(ref data_type) = characteristic.data_type {
                            entities.insert(data_type.clone());
                        }
                    }
                    CharacteristicKind::Collection {
                        element_characteristic,
                        ..
                    }
                    | CharacteristicKind::List {
                        element_characteristic,
                        ..
                    }
                    | CharacteristicKind::Set {
                        element_characteristic,
                        ..
                    }
                    | CharacteristicKind::SortedSet {
                        element_characteristic,
                        ..
                    } => {
                        if let Some(elem_char) = element_characteristic {
                            if let Some(ref data_type) = elem_char.data_type {
                                entities.insert(data_type.clone());
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        entities
    }

    /// Builds a dependency graph of all model elements
    ///
    /// # Returns
    ///
    /// Vector of dependencies showing relationships between elements
    pub fn build_dependency_graph(&self) -> Vec<Dependency> {
        let mut dependencies = Vec::new();

        // Property -> Characteristic dependencies
        for property in self.aspect.properties() {
            if let Some(ref characteristic) = property.characteristic {
                dependencies.push(Dependency {
                    from: property.urn().to_string(),
                    to: characteristic.urn().to_string(),
                    dependency_type: "characteristic".to_string(),
                });

                // Characteristic -> DataType/Entity dependencies
                if let Some(ref data_type) = characteristic.data_type {
                    dependencies.push(Dependency {
                        from: characteristic.urn().to_string(),
                        to: data_type.clone(),
                        dependency_type: "datatype".to_string(),
                    });
                }

                // Handle nested characteristics (e.g., Collection element characteristic)
                match characteristic.kind() {
                    CharacteristicKind::Collection {
                        element_characteristic,
                        ..
                    }
                    | CharacteristicKind::List {
                        element_characteristic,
                        ..
                    }
                    | CharacteristicKind::Set {
                        element_characteristic,
                        ..
                    }
                    | CharacteristicKind::SortedSet {
                        element_characteristic,
                        ..
                    } => {
                        if let Some(elem_char) = element_characteristic {
                            dependencies.push(Dependency {
                                from: characteristic.urn().to_string(),
                                to: elem_char.urn().to_string(),
                                dependency_type: "element_characteristic".to_string(),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }

        // Operation dependencies
        for operation in self.aspect.operations() {
            // Input property dependencies
            for input in operation.input() {
                dependencies.push(Dependency {
                    from: operation.urn().to_string(),
                    to: input.urn().to_string(),
                    dependency_type: "input".to_string(),
                });
            }

            // Output property dependencies
            if let Some(output) = operation.output() {
                dependencies.push(Dependency {
                    from: operation.urn().to_string(),
                    to: output.urn().to_string(),
                    dependency_type: "output".to_string(),
                });
            }
        }

        dependencies
    }

    /// Detects circular dependencies in the model
    ///
    /// Uses depth-first search to detect cycles in the dependency graph.
    ///
    /// # Returns
    ///
    /// Vector of URN chains representing circular dependencies
    pub fn detect_circular_dependencies(&self) -> Vec<Vec<String>> {
        let dependencies = self.build_dependency_graph();
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();

        // Build adjacency list
        for dep in &dependencies {
            graph
                .entry(dep.from.clone())
                .or_default()
                .push(dep.to.clone());
        }

        let mut cycles = Vec::new();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut path = Vec::new();

        for node in graph.keys() {
            if !visited.contains(node) {
                Self::dfs_detect_cycle(
                    node,
                    &graph,
                    &mut visited,
                    &mut rec_stack,
                    &mut path,
                    &mut cycles,
                );
            }
        }

        cycles
    }

    /// Helper function for DFS cycle detection
    fn dfs_detect_cycle(
        node: &str,
        graph: &HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
        path: &mut Vec<String>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());
        path.push(node.to_string());

        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    Self::dfs_detect_cycle(neighbor, graph, visited, rec_stack, path, cycles);
                } else if rec_stack.contains(neighbor) {
                    // Found a cycle
                    if let Some(cycle_start) = path.iter().position(|n| n == neighbor) {
                        cycles.push(path[cycle_start..].to_vec());
                    }
                }
            }
        }

        path.pop();
        rec_stack.remove(node);
    }

    /// Calculates complexity metrics for the model
    ///
    /// # Returns
    ///
    /// Complexity metrics including property counts, nesting depth, and reference counts
    pub fn complexity_metrics(&self) -> ComplexityMetrics {
        let total_properties = self.aspect.properties().len();
        let total_operations = self.aspect.operations().len();
        let optional_properties = self.find_optional_properties().len();
        let collection_properties = self.find_properties_with_collection_characteristic().len();
        let total_entities = self.find_all_referenced_entities().len();
        let circular_references = self.detect_circular_dependencies().len();

        // Calculate max nesting depth
        let max_nesting_depth = self.calculate_max_nesting_depth();

        ComplexityMetrics {
            total_properties,
            total_entities,
            total_operations,
            max_nesting_depth,
            optional_properties,
            collection_properties,
            circular_references,
        }
    }

    /// Calculates the maximum nesting depth of entity references
    fn calculate_max_nesting_depth(&self) -> usize {
        let mut max_depth = 1; // Aspect itself is depth 1

        for property in self.aspect.properties() {
            if let Some(ref characteristic) = property.characteristic {
                let depth =
                    Self::calculate_characteristic_depth(characteristic, &mut HashSet::new());
                max_depth = max_depth.max(depth);
            }
        }

        max_depth
    }

    /// Helper to calculate depth of a characteristic (handles nested collections)
    fn calculate_characteristic_depth(
        characteristic: &crate::metamodel::Characteristic,
        visited: &mut HashSet<String>,
    ) -> usize {
        if visited.contains(characteristic.urn()) {
            return 0; // Avoid infinite recursion on circular refs
        }
        visited.insert(characteristic.urn().to_string());

        match characteristic.kind() {
            CharacteristicKind::Collection {
                element_characteristic,
                ..
            }
            | CharacteristicKind::List {
                element_characteristic,
                ..
            }
            | CharacteristicKind::Set {
                element_characteristic,
                ..
            }
            | CharacteristicKind::SortedSet {
                element_characteristic,
                ..
            } => {
                if let Some(elem_char) = element_characteristic {
                    1 + Self::calculate_characteristic_depth(elem_char, visited)
                } else {
                    1
                }
            }
            CharacteristicKind::SingleEntity { .. } => 2,
            _ => 1,
        }
    }

    /// Finds properties by name pattern (case-insensitive)
    ///
    /// # Arguments
    ///
    /// * `pattern` - Pattern to match against property names
    ///
    /// # Returns
    ///
    /// Vector of properties with names containing the pattern
    pub fn find_properties_by_name_pattern(&self, pattern: &str) -> Vec<&Property> {
        let pattern_lower = pattern.to_lowercase();
        self.aspect
            .properties()
            .iter()
            .filter(|prop| prop.name().to_lowercase().contains(&pattern_lower))
            .collect()
    }

    /// Groups properties by their characteristic type
    ///
    /// # Returns
    ///
    /// HashMap mapping characteristic type names to vectors of properties
    pub fn group_properties_by_characteristic_type(&self) -> HashMap<String, Vec<&Property>> {
        let mut groups: HashMap<String, Vec<&Property>> = HashMap::new();

        for property in self.aspect.properties() {
            let type_name = if let Some(ref characteristic) = property.characteristic {
                match characteristic.kind() {
                    CharacteristicKind::Trait => "Trait".to_string(),
                    CharacteristicKind::Quantifiable { .. } => "Quantifiable".to_string(),
                    CharacteristicKind::Measurement { .. } => "Measurement".to_string(),
                    CharacteristicKind::Enumeration { .. } => "Enumeration".to_string(),
                    CharacteristicKind::State { .. } => "State".to_string(),
                    CharacteristicKind::Duration { .. } => "Duration".to_string(),
                    CharacteristicKind::Collection { .. } => "Collection".to_string(),
                    CharacteristicKind::List { .. } => "List".to_string(),
                    CharacteristicKind::Set { .. } => "Set".to_string(),
                    CharacteristicKind::SortedSet { .. } => "SortedSet".to_string(),
                    CharacteristicKind::TimeSeries { .. } => "TimeSeries".to_string(),
                    CharacteristicKind::Code => "Code".to_string(),
                    CharacteristicKind::Either { .. } => "Either".to_string(),
                    CharacteristicKind::SingleEntity { .. } => "SingleEntity".to_string(),
                    CharacteristicKind::StructuredValue { .. } => "StructuredValue".to_string(),
                }
            } else {
                "NoCharacteristic".to_string()
            };

            groups.entry(type_name).or_default().push(property);
        }

        groups
    }

    /// Gets the aspect reference
    pub fn aspect(&self) -> &Aspect {
        self.aspect
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Characteristic, CharacteristicKind};

    #[test]
    fn test_find_optional_properties() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let prop1 = Property::new("urn:samm:test:1.0.0#required".to_string());

        let mut prop2 = Property::new("urn:samm:test:1.0.0#optional".to_string());
        prop2.optional = true;

        aspect.add_property(prop1);
        aspect.add_property(prop2);

        let query = ModelQuery::new(&aspect);
        let optional = query.find_optional_properties();

        assert_eq!(optional.len(), 1);
        assert_eq!(optional[0].name(), "optional");
    }

    #[test]
    fn test_find_required_properties() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let prop1 = Property::new("urn:samm:test:1.0.0#required".to_string());

        let mut prop2 = Property::new("urn:samm:test:1.0.0#optional".to_string());
        prop2.optional = true;

        aspect.add_property(prop1);
        aspect.add_property(prop2);

        let query = ModelQuery::new(&aspect);
        let required = query.find_required_properties();

        assert_eq!(required.len(), 1);
        assert_eq!(required[0].name(), "required");
    }

    #[test]
    fn test_find_collection_properties() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let mut prop1 = Property::new("urn:samm:test:1.0.0#list".to_string());
        let char1 = Characteristic::new(
            "urn:samm:test:1.0.0#ListChar".to_string(),
            CharacteristicKind::List {
                element_characteristic: None,
            },
        );
        prop1.characteristic = Some(char1);

        let mut prop2 = Property::new("urn:samm:test:1.0.0#simple".to_string());
        let char2 = Characteristic::new(
            "urn:samm:test:1.0.0#TraitChar".to_string(),
            CharacteristicKind::Trait,
        );
        prop2.characteristic = Some(char2);

        aspect.add_property(prop1);
        aspect.add_property(prop2);

        let query = ModelQuery::new(&aspect);
        let collections = query.find_properties_with_collection_characteristic();

        assert_eq!(collections.len(), 1);
        assert_eq!(collections[0].name(), "list");
    }

    #[test]
    fn test_complexity_metrics() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let prop1 = Property::new("urn:samm:test:1.0.0#prop1".to_string());

        let mut prop2 = Property::new("urn:samm:test:1.0.0#prop2".to_string());
        prop2.optional = true;

        aspect.add_property(prop1);
        aspect.add_property(prop2);

        let query = ModelQuery::new(&aspect);
        let metrics = query.complexity_metrics();

        assert_eq!(metrics.total_properties, 2);
        assert_eq!(metrics.optional_properties, 1);
        assert_eq!(metrics.total_operations, 0);
    }

    #[test]
    fn test_find_properties_by_name_pattern() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        aspect.add_property(Property::new("urn:samm:test:1.0.0#speedLimit".to_string()));
        aspect.add_property(Property::new(
            "urn:samm:test:1.0.0#currentSpeed".to_string(),
        ));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#temperature".to_string()));

        let query = ModelQuery::new(&aspect);
        let results = query.find_properties_by_name_pattern("speed");

        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|p| p.name() == "speedLimit"));
        assert!(results.iter().any(|p| p.name() == "currentSpeed"));
    }

    #[test]
    fn test_group_properties_by_characteristic_type() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let mut prop1 = Property::new("urn:samm:test:1.0.0#list1".to_string());
        prop1.characteristic = Some(Characteristic::new(
            "urn:samm:test:1.0.0#ListChar1".to_string(),
            CharacteristicKind::List {
                element_characteristic: None,
            },
        ));

        let mut prop2 = Property::new("urn:samm:test:1.0.0#list2".to_string());
        prop2.characteristic = Some(Characteristic::new(
            "urn:samm:test:1.0.0#ListChar2".to_string(),
            CharacteristicKind::List {
                element_characteristic: None,
            },
        ));

        let mut prop3 = Property::new("urn:samm:test:1.0.0#trait1".to_string());
        prop3.characteristic = Some(Characteristic::new(
            "urn:samm:test:1.0.0#TraitChar".to_string(),
            CharacteristicKind::Trait,
        ));

        aspect.add_property(prop1);
        aspect.add_property(prop2);
        aspect.add_property(prop3);

        let query = ModelQuery::new(&aspect);
        let groups = query.group_properties_by_characteristic_type();

        assert_eq!(groups.get("List").unwrap().len(), 2);
        assert_eq!(groups.get("Trait").unwrap().len(), 1);
    }

    #[test]
    fn test_build_dependency_graph() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let mut prop1 = Property::new("urn:samm:test:1.0.0#prop1".to_string());
        let char1 = Characteristic::new(
            "urn:samm:test:1.0.0#Char1".to_string(),
            CharacteristicKind::Trait,
        );
        prop1.characteristic = Some(char1);

        aspect.add_property(prop1);

        let query = ModelQuery::new(&aspect);
        let deps = query.build_dependency_graph();

        assert!(!deps.is_empty());
        assert!(deps.iter().any(|d| d.from.contains("prop1")));
        assert!(deps.iter().any(|d| d.to.contains("Char1")));
    }

    #[test]
    fn test_detect_circular_dependencies_none() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let mut prop1 = Property::new("urn:samm:test:1.0.0#prop1".to_string());
        let char1 = Characteristic::new(
            "urn:samm:test:1.0.0#Char1".to_string(),
            CharacteristicKind::Trait,
        );
        prop1.characteristic = Some(char1);

        aspect.add_property(prop1);

        let query = ModelQuery::new(&aspect);
        let cycles = query.detect_circular_dependencies();

        assert_eq!(cycles.len(), 0);
    }

    #[test]
    fn test_find_properties_in_namespace() {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());

        aspect.add_property(Property::new(
            "urn:samm:org.example:1.0.0#prop1".to_string(),
        ));
        aspect.add_property(Property::new("urn:samm:org.other:1.0.0#prop2".to_string()));

        let query = ModelQuery::new(&aspect);
        let results = query.find_properties_in_namespace("urn:samm:org.example:1.0.0");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name(), "prop1");
    }
}
