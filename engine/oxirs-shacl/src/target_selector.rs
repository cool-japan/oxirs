//! SHACL target declaration evaluation.
//!
//! Implements the five SHACL target types (sh:targetClass, sh:targetSubjectsOf,
//! sh:targetObjectsOf, sh:targetNode, and implicit class targets) and resolves
//! them against an in-memory RDF graph.

use std::collections::HashMap;

// Well-known predicate constants
const RDF_TYPE: &str = "rdf:type";
const RDFS_SUBCLASS_OF: &str = "rdfs:subClassOf";

/// The five SHACL target declaration types.
#[derive(Debug, Clone, PartialEq)]
pub enum TargetType {
    /// `sh:targetClass` — all instances (direct and via subclass) of the given class.
    Class(String),
    /// `sh:targetSubjectsOf` — all subjects that have the given predicate.
    SubjectsOf(String),
    /// `sh:targetObjectsOf` — all IRI objects of the given predicate.
    ObjectsOf(String),
    /// `sh:targetNode` — a single specific node.
    Node(String),
    /// Implicit class target — the shape is itself a class; targets its instances
    /// plus the class node itself.
    ImplicitClass(String),
}

/// A minimal in-memory RDF graph: subject → [(predicate, object)].
#[derive(Debug, Default, Clone)]
pub struct RdfGraph {
    data: HashMap<String, Vec<(String, String)>>,
}

impl RdfGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple to the graph.
    pub fn add_triple(&mut self, s: &str, p: &str, o: &str) {
        self.data
            .entry(s.to_string())
            .or_default()
            .push((p.to_string(), o.to_string()));
    }

    /// Return all subjects in the graph (deduplicated, unsorted).
    pub fn subjects(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    /// Return all objects of predicate `p` for subject `s`.
    pub fn objects_of(&self, s: &str, p: &str) -> Vec<String> {
        self.data
            .get(s)
            .map(|pairs| {
                pairs
                    .iter()
                    .filter(|(pred, _)| pred == p)
                    .map(|(_, obj)| obj.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Return all subjects that have at least one triple with predicate `p`.
    pub fn subjects_with_predicate(&self, p: &str) -> Vec<String> {
        self.data
            .iter()
            .filter(|(_, pairs)| pairs.iter().any(|(pred, _)| pred == p))
            .map(|(s, _)| s.clone())
            .collect()
    }

    /// Return all objects (of any subject) with predicate `p`.
    pub fn objects_with_predicate(&self, p: &str) -> Vec<String> {
        let mut result = Vec::new();
        for pairs in self.data.values() {
            for (pred, obj) in pairs {
                if pred == p {
                    result.push(obj.clone());
                }
            }
        }
        result
    }

    /// Return all `rdf:type` values (classes) for subject `s`.
    pub fn types_of(&self, subject: &str) -> Vec<String> {
        self.objects_of(subject, RDF_TYPE)
    }

    /// Check whether a node (IRI or blank node) exists as a subject in the graph.
    pub fn contains_subject(&self, node: &str) -> bool {
        self.data.contains_key(node)
    }

    /// Return true if `node` appears as any object in the graph.
    pub fn contains_as_object(&self, node: &str) -> bool {
        self.data
            .values()
            .any(|pairs| pairs.iter().any(|(_, o)| o == node))
    }

    /// Return true if `node` appears anywhere in the graph (subject or object).
    pub fn node_exists(&self, node: &str) -> bool {
        self.contains_subject(node) || self.contains_as_object(node)
    }
}

/// Evaluates SHACL target declarations against an RDF graph.
pub struct TargetSelector;

impl TargetSelector {
    /// Compute the rdfs:subClassOf* closure — all classes that are (transitively)
    /// subclasses of `class` within the graph.
    ///
    /// Returns the set of classes that have `class` as a (possibly indirect) superclass,
    /// not including `class` itself.
    pub fn subclass_closure(class: &str, graph: &RdfGraph) -> Vec<String> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        Self::subclass_closure_inner(class, graph, &mut result, &mut visited);
        result.sort();
        result.dedup();
        result
    }

    fn subclass_closure_inner(
        class: &str,
        graph: &RdfGraph,
        result: &mut Vec<String>,
        visited: &mut std::collections::HashSet<String>,
    ) {
        // Find all subjects S such that S rdfs:subClassOf class
        for (subject, pairs) in &graph.data {
            for (pred, obj) in pairs {
                if pred == RDFS_SUBCLASS_OF && obj == class && !visited.contains(subject) {
                    visited.insert(subject.clone());
                    result.push(subject.clone());
                    Self::subclass_closure_inner(subject, graph, result, visited);
                }
            }
        }
    }

    /// Select target nodes for a single target declaration.
    pub fn select_targets(target: &TargetType, graph: &RdfGraph) -> Vec<String> {
        match target {
            TargetType::Class(class) => {
                // All direct instances of class, plus all instances of subclasses
                let mut classes = vec![class.clone()];
                classes.extend(Self::subclass_closure(class, graph));

                let mut targets = Vec::new();
                for c in &classes {
                    for subject in graph.subjects() {
                        let types = graph.types_of(&subject);
                        if types.contains(c) {
                            targets.push(subject);
                        }
                    }
                }
                targets.sort();
                targets.dedup();
                targets
            }

            TargetType::SubjectsOf(predicate) => {
                let mut targets = graph.subjects_with_predicate(predicate);
                targets.sort();
                targets.dedup();
                targets
            }

            TargetType::ObjectsOf(predicate) => {
                // Only IRI objects — filter out blank nodes (starting with "_:")
                let mut targets: Vec<String> = graph
                    .objects_with_predicate(predicate)
                    .into_iter()
                    .filter(|o| !o.starts_with("_:"))
                    .collect();
                targets.sort();
                targets.dedup();
                targets
            }

            TargetType::Node(node) => {
                if graph.node_exists(node) {
                    vec![node.clone()]
                } else {
                    vec![]
                }
            }

            TargetType::ImplicitClass(class) => {
                // Instances of the class plus the class node itself
                let mut targets = Self::select_targets(&TargetType::Class(class.clone()), graph);
                // Add the class IRI itself
                if !targets.contains(class) {
                    targets.push(class.clone());
                }
                targets.sort();
                targets.dedup();
                targets
            }
        }
    }

    /// Select targets for multiple target declarations, return sorted deduplicated union.
    pub fn select_all(targets: &[TargetType], graph: &RdfGraph) -> Vec<String> {
        let mut result = Vec::new();
        for target in targets {
            result.extend(Self::select_targets(target, graph));
        }
        result.sort();
        result.dedup();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> RdfGraph {
        let mut g = RdfGraph::new();
        g.add_triple("Alice", RDF_TYPE, "Person");
        g.add_triple("Bob", RDF_TYPE, "Person");
        g.add_triple("Carol", RDF_TYPE, "Employee");
        g.add_triple("Employee", RDFS_SUBCLASS_OF, "Person");
        g.add_triple("Alice", "knows", "Bob");
        g.add_triple("Bob", "knows", "Carol");
        g
    }

    // ------ RdfGraph ------

    #[test]
    fn test_graph_add_triple() {
        let mut g = RdfGraph::new();
        g.add_triple("s", "p", "o");
        assert!(g.contains_subject("s"));
    }

    #[test]
    fn test_graph_objects_of() {
        let g = make_graph();
        let types = g.objects_of("Alice", RDF_TYPE);
        assert!(types.contains(&"Person".to_string()));
    }

    #[test]
    fn test_graph_subjects_with_predicate() {
        let g = make_graph();
        let subs = g.subjects_with_predicate("knows");
        assert!(subs.contains(&"Alice".to_string()));
        assert!(subs.contains(&"Bob".to_string()));
    }

    #[test]
    fn test_graph_objects_with_predicate() {
        let g = make_graph();
        let objs = g.objects_with_predicate("knows");
        assert!(objs.contains(&"Bob".to_string()));
        assert!(objs.contains(&"Carol".to_string()));
    }

    #[test]
    fn test_graph_types_of() {
        let g = make_graph();
        let t = g.types_of("Alice");
        assert_eq!(t, vec!["Person".to_string()]);
    }

    #[test]
    fn test_graph_contains_as_object() {
        let g = make_graph();
        assert!(g.contains_as_object("Bob"));
        assert!(!g.contains_as_object("NonExistent"));
    }

    #[test]
    fn test_graph_node_exists_subject() {
        let g = make_graph();
        assert!(g.node_exists("Alice"));
    }

    #[test]
    fn test_graph_node_exists_object_only() {
        let mut g = RdfGraph::new();
        g.add_triple("a", "p", "b");
        // "b" is only an object, not a subject
        assert!(g.node_exists("b"));
    }

    #[test]
    fn test_graph_node_not_exists() {
        let g = make_graph();
        assert!(!g.node_exists("Zzz_not_in_graph"));
    }

    // ------ subclass_closure ------

    #[test]
    fn test_subclass_closure_direct() {
        let g = make_graph();
        let closure = TargetSelector::subclass_closure("Person", &g);
        assert!(closure.contains(&"Employee".to_string()));
    }

    #[test]
    fn test_subclass_closure_transitive() {
        let mut g = RdfGraph::new();
        g.add_triple("C", RDFS_SUBCLASS_OF, "B");
        g.add_triple("B", RDFS_SUBCLASS_OF, "A");
        let closure = TargetSelector::subclass_closure("A", &g);
        assert!(closure.contains(&"B".to_string()));
        assert!(closure.contains(&"C".to_string()));
    }

    #[test]
    fn test_subclass_closure_empty() {
        let g = make_graph();
        let closure = TargetSelector::subclass_closure("Employee", &g);
        assert!(closure.is_empty());
    }

    #[test]
    fn test_subclass_closure_no_cycle() {
        // Ensure no infinite loop even with a cycle in data
        let mut g = RdfGraph::new();
        g.add_triple("A", RDFS_SUBCLASS_OF, "B");
        g.add_triple("B", RDFS_SUBCLASS_OF, "A");
        // Should terminate
        let _ = TargetSelector::subclass_closure("A", &g);
    }

    // ------ select_targets: Class ------

    #[test]
    fn test_target_class_direct_instances() {
        let g = make_graph();
        let targets = TargetSelector::select_targets(&TargetType::Class("Person".to_string()), &g);
        // Alice and Bob are directly typed as Person; Carol is Employee (subclass of Person)
        assert!(targets.contains(&"Alice".to_string()));
        assert!(targets.contains(&"Bob".to_string()));
        assert!(targets.contains(&"Carol".to_string()));
    }

    #[test]
    fn test_target_class_no_instances() {
        let g = make_graph();
        let targets = TargetSelector::select_targets(&TargetType::Class("Vehicle".to_string()), &g);
        assert!(targets.is_empty());
    }

    #[test]
    fn test_target_class_subclass_instances() {
        let g = make_graph();
        let targets = TargetSelector::select_targets(&TargetType::Class("Person".to_string()), &g);
        assert!(targets.contains(&"Carol".to_string()));
    }

    // ------ select_targets: SubjectsOf ------

    #[test]
    fn test_target_subjects_of() {
        let g = make_graph();
        let targets =
            TargetSelector::select_targets(&TargetType::SubjectsOf("knows".to_string()), &g);
        assert!(targets.contains(&"Alice".to_string()));
        assert!(targets.contains(&"Bob".to_string()));
        assert!(!targets.contains(&"Carol".to_string()));
    }

    #[test]
    fn test_target_subjects_of_nonexistent_predicate() {
        let g = make_graph();
        let targets =
            TargetSelector::select_targets(&TargetType::SubjectsOf("xxx".to_string()), &g);
        assert!(targets.is_empty());
    }

    // ------ select_targets: ObjectsOf ------

    #[test]
    fn test_target_objects_of_iris() {
        let g = make_graph();
        let targets =
            TargetSelector::select_targets(&TargetType::ObjectsOf("knows".to_string()), &g);
        assert!(targets.contains(&"Bob".to_string()));
        assert!(targets.contains(&"Carol".to_string()));
    }

    #[test]
    fn test_target_objects_of_filters_blank_nodes() {
        let mut g = RdfGraph::new();
        g.add_triple("s", "p", "_:b0");
        g.add_triple("s", "p", "IRI");
        let targets = TargetSelector::select_targets(&TargetType::ObjectsOf("p".to_string()), &g);
        assert!(!targets.contains(&"_:b0".to_string()));
        assert!(targets.contains(&"IRI".to_string()));
    }

    // ------ select_targets: Node ------

    #[test]
    fn test_target_node_exists() {
        let g = make_graph();
        let targets = TargetSelector::select_targets(&TargetType::Node("Alice".to_string()), &g);
        assert_eq!(targets, vec!["Alice".to_string()]);
    }

    #[test]
    fn test_target_node_not_exists() {
        let g = make_graph();
        let targets = TargetSelector::select_targets(&TargetType::Node("Nobody".to_string()), &g);
        assert!(targets.is_empty());
    }

    // ------ select_targets: ImplicitClass ------

    #[test]
    fn test_target_implicit_class_includes_class_itself() {
        let g = make_graph();
        let targets =
            TargetSelector::select_targets(&TargetType::ImplicitClass("Person".to_string()), &g);
        assert!(targets.contains(&"Person".to_string()));
        assert!(targets.contains(&"Alice".to_string()));
        assert!(targets.contains(&"Bob".to_string()));
    }

    // ------ select_all ------

    #[test]
    fn test_select_all_union_dedup() {
        let g = make_graph();
        let targets_list = vec![
            TargetType::SubjectsOf("knows".to_string()),
            TargetType::ObjectsOf("knows".to_string()),
        ];
        let all = TargetSelector::select_all(&targets_list, &g);
        // Alice, Bob, Carol — deduplicated and sorted
        assert!(all.contains(&"Alice".to_string()));
        assert!(all.contains(&"Bob".to_string()));
        assert!(all.contains(&"Carol".to_string()));
        // No duplicates
        let mut dedup = all.clone();
        dedup.dedup();
        assert_eq!(all, dedup);
    }

    #[test]
    fn test_select_all_empty_targets() {
        let g = make_graph();
        let all = TargetSelector::select_all(&[], &g);
        assert!(all.is_empty());
    }

    #[test]
    fn test_select_all_sorted() {
        let g = make_graph();
        let targets_list = vec![TargetType::Class("Person".to_string())];
        let all = TargetSelector::select_all(&targets_list, &g);
        let mut sorted = all.clone();
        sorted.sort();
        assert_eq!(all, sorted);
    }

    #[test]
    fn test_objects_of_dedup() {
        let mut g = RdfGraph::new();
        g.add_triple("a", "p", "X");
        g.add_triple("b", "p", "X");
        let targets = TargetSelector::select_targets(&TargetType::ObjectsOf("p".to_string()), &g);
        assert_eq!(targets, vec!["X".to_string()]);
    }
}
