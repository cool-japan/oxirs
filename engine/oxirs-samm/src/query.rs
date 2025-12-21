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

    /// Find properties by fuzzy name matching
    ///
    /// Uses Levenshtein distance to find properties whose names are similar to the query.
    /// Useful for finding properties when you don't know the exact name.
    ///
    /// # Arguments
    ///
    /// * `query` - The property name to search for (partial or misspelled)
    /// * `max_distance` - Maximum Levenshtein distance (lower = stricter matching)
    ///
    /// # Returns
    ///
    /// Vector of (property, distance) tuples, sorted by distance (best matches first)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::query::ModelQuery;
    /// # use oxirs_samm::metamodel::{Aspect, Property};
    /// # fn example(aspect: &Aspect) {
    /// let query = ModelQuery::new(aspect);
    ///
    /// // Find properties with names similar to "temperture" (misspelled)
    /// let results = query.fuzzy_find_properties("temperture", 3);
    /// for (property, distance) in results {
    ///     println!("Found: {} (distance: {})", property.name(), distance);
    /// }
    /// # }
    /// ```
    pub fn fuzzy_find_properties(
        &self,
        query: &str,
        max_distance: usize,
    ) -> Vec<(&Property, usize)> {
        let mut results: Vec<(&Property, usize)> = self
            .aspect
            .properties()
            .iter()
            .map(|prop| {
                let distance = levenshtein_distance(query, &prop.name());
                (prop, distance)
            })
            .filter(|(_, distance)| *distance <= max_distance)
            .collect();

        // Sort by distance (best matches first)
        results.sort_by_key(|(_, distance)| *distance);
        results
    }

    /// Find operations by fuzzy name matching
    ///
    /// Uses Levenshtein distance to find operations whose names are similar to the query.
    ///
    /// # Arguments
    ///
    /// * `query` - The operation name to search for
    /// * `max_distance` - Maximum Levenshtein distance
    ///
    /// # Returns
    ///
    /// Vector of (operation, distance) tuples, sorted by distance
    pub fn fuzzy_find_operations(
        &self,
        query: &str,
        max_distance: usize,
    ) -> Vec<(&Operation, usize)> {
        let mut results: Vec<(&Operation, usize)> = self
            .aspect
            .operations()
            .iter()
            .map(|op| {
                let distance = levenshtein_distance(query, &op.name());
                (op, distance)
            })
            .filter(|(_, distance)| *distance <= max_distance)
            .collect();

        results.sort_by_key(|(_, distance)| *distance);
        results
    }

    /// Find all model elements (properties, operations) by fuzzy search
    ///
    /// Searches across all element names in the aspect.
    ///
    /// # Arguments
    ///
    /// * `query` - The element name to search for
    /// * `max_distance` - Maximum Levenshtein distance
    ///
    /// # Returns
    ///
    /// Vector of (element name, URN, distance) tuples, sorted by distance
    pub fn fuzzy_find_any_element(
        &self,
        query: &str,
        max_distance: usize,
    ) -> Vec<(String, String, usize)> {
        let mut results = Vec::new();

        // Search properties
        for prop in self.aspect.properties() {
            let distance = levenshtein_distance(query, &prop.name());
            if distance <= max_distance {
                results.push((prop.name().to_string(), prop.urn().to_string(), distance));
            }
        }

        // Search operations
        for op in self.aspect.operations() {
            let distance = levenshtein_distance(query, &op.name());
            if distance <= max_distance {
                results.push((op.name().to_string(), op.urn().to_string(), distance));
            }
        }

        // Sort by distance
        results.sort_by_key(|(_, _, distance)| *distance);
        results
    }

    /// Find properties with similar names (auto-suggest)
    ///
    /// Provides auto-complete style suggestions based on prefix matching
    /// combined with fuzzy matching as fallback.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The partial property name to match
    /// * `limit` - Maximum number of suggestions to return
    ///
    /// # Returns
    ///
    /// Vector of suggested property names
    pub fn suggest_properties(&self, prefix: &str, limit: usize) -> Vec<String> {
        let prefix_lower = prefix.to_lowercase();
        let mut suggestions = Vec::new();

        // First, collect exact prefix matches
        let mut prefix_matches: Vec<_> = self
            .aspect
            .properties()
            .iter()
            .filter(|prop| prop.name().to_lowercase().starts_with(&prefix_lower))
            .map(|prop| (prop.name().to_string(), 0))
            .collect();

        // Then, collect fuzzy matches (not already in prefix matches)
        let prefix_match_names: HashSet<_> = prefix_matches
            .iter()
            .map(|(name, _)| name.clone())
            .collect();

        let mut fuzzy_matches: Vec<_> = self
            .aspect
            .properties()
            .iter()
            .filter(|prop| !prefix_match_names.contains(&prop.name()))
            .map(|prop| {
                let distance = levenshtein_distance(prefix, &prop.name());
                (prop.name().to_string(), distance)
            })
            .filter(|(_, distance)| *distance <= 3)
            .collect();

        // Combine results
        suggestions.append(&mut prefix_matches);
        suggestions.append(&mut fuzzy_matches);
        suggestions.sort_by_key(|(_, distance)| *distance);
        suggestions.truncate(limit);

        suggestions.into_iter().map(|(name, _)| name).collect()
    }
}

/// Calculate Levenshtein distance between two strings
///
/// The Levenshtein distance is the minimum number of single-character edits
/// (insertions, deletions, or substitutions) required to change one string into another.
///
/// # Arguments
///
/// * `a` - First string
/// * `b` - Second string
///
/// # Returns
///
/// The Levenshtein distance as a usize
#[allow(clippy::needless_range_loop)]
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_len = a.chars().count();
    let b_len = b.chars().count();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let mut matrix = vec![vec![0; b_len + 1]; a_len + 1];

    // Initialize first row and column
    for i in 0..=a_len {
        matrix[i][0] = i;
    }
    for j in 0..=b_len {
        matrix[0][j] = j;
    }

    // Compute distances
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    for (i, a_char) in a_chars.iter().enumerate() {
        for (j, b_char) in b_chars.iter().enumerate() {
            let cost = if a_char == b_char { 0 } else { 1 };

            matrix[i + 1][j + 1] = *[
                matrix[i][j + 1] + 1, // deletion
                matrix[i + 1][j] + 1, // insertion
                matrix[i][j] + cost,  // substitution
            ]
            .iter()
            .min()
            .unwrap();
        }
    }

    matrix[a_len][b_len]
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

    // Fuzzy Search Tests

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("", "test"), 4);
        assert_eq!(levenshtein_distance("test", ""), 4);
        assert_eq!(levenshtein_distance("abc", "def"), 3);
    }

    #[test]
    fn test_fuzzy_find_properties_exact_match() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#temperature".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#humidity".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#pressure".to_string()));

        let query = ModelQuery::new(&aspect);
        let results = query.fuzzy_find_properties("temperature", 0);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.name(), "temperature");
        assert_eq!(results[0].1, 0); // exact match
    }

    #[test]
    fn test_fuzzy_find_properties_typo() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#temperature".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#humidity".to_string()));

        let query = ModelQuery::new(&aspect);
        // "temperture" is missing an 'a' - distance of 1
        let results = query.fuzzy_find_properties("temperture", 2);

        assert!(!results.is_empty());
        assert!(results.iter().any(|(prop, _)| prop.name() == "temperature"));
    }

    #[test]
    fn test_fuzzy_find_properties_sorted_by_distance() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#temp".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#temperature".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#tempValue".to_string()));

        let query = ModelQuery::new(&aspect);
        let results = query.fuzzy_find_properties("temp", 5);

        // Should be sorted by distance
        assert!(!results.is_empty());
        assert_eq!(results[0].0.name(), "temp"); // exact match first
        assert_eq!(results[0].1, 0);
    }

    #[test]
    fn test_fuzzy_find_operations() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_operation(Operation::new(
            "urn:samm:test:1.0.0#startEngine".to_string(),
        ));
        aspect.add_operation(Operation::new("urn:samm:test:1.0.0#stopEngine".to_string()));

        let query = ModelQuery::new(&aspect);
        let results = query.fuzzy_find_operations("startEngin", 2); // missing 'e'

        assert!(!results.is_empty());
        assert!(results.iter().any(|(op, _)| op.name() == "startEngine"));
    }

    #[test]
    fn test_fuzzy_find_any_element() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#temperature".to_string()));
        aspect.add_operation(Operation::new("urn:samm:test:1.0.0#measure".to_string()));

        let query = ModelQuery::new(&aspect);
        // "temp" vs "temperature" has distance of 7 (need to add "erature")
        let results = query.fuzzy_find_any_element("temp", 8);

        assert!(!results.is_empty());
        assert!(results.iter().any(|(name, _, _)| name == "temperature"));
    }

    #[test]
    fn test_suggest_properties_prefix_match() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#temperature".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#tempValue".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#humidity".to_string()));

        let query = ModelQuery::new(&aspect);
        let suggestions = query.suggest_properties("temp", 5);

        assert_eq!(suggestions.len(), 2);
        assert!(suggestions.contains(&"temperature".to_string()));
        assert!(suggestions.contains(&"tempValue".to_string()));
    }

    #[test]
    fn test_suggest_properties_limit() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop1".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop2".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop3".to_string()));

        let query = ModelQuery::new(&aspect);
        let suggestions = query.suggest_properties("prop", 2);

        assert_eq!(suggestions.len(), 2);
    }

    #[test]
    fn test_suggest_properties_fuzzy_fallback() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#temperature".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#humidity".to_string()));

        let query = ModelQuery::new(&aspect);
        // "temper" doesn't match "humidity", but fuzzy match should find "temperature"
        let suggestions = query.suggest_properties("temper", 5);

        assert!(!suggestions.is_empty());
        assert!(suggestions.contains(&"temperature".to_string()));
    }
}
