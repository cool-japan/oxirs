//! SAMM aspect hierarchical property chain traversal.
//!
//! This module models a directed acyclic hierarchy of SAMM aspects, each
//! carrying properties. A "property chain" is a sequence of steps through
//! the hierarchy: each step is either an aspect id (navigate to sub-aspect)
//! or a property name (terminal step that resolves to a type).

use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────
// Property
// ─────────────────────────────────────────────────

/// A single property inside a SAMM aspect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Property {
    /// Property name (identifier)
    pub name: String,
    /// XSD datatype URI or shorthand (e.g. "xsd:string")
    pub datatype: String,
    /// Whether this property may be absent in an instance
    pub optional: bool,
    /// Optional human-readable description
    pub description: Option<String>,
}

impl Property {
    /// Construct a required property.
    pub fn new(name: impl Into<String>, datatype: impl Into<String>) -> Self {
        Property {
            name: name.into(),
            datatype: datatype.into(),
            optional: false,
            description: None,
        }
    }

    /// Mark the property as optional.
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Add a description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

// ─────────────────────────────────────────────────
// Aspect
// ─────────────────────────────────────────────────

/// A SAMM aspect: a named container of properties and sub-aspect references.
#[derive(Debug, Clone)]
pub struct Aspect {
    /// Unique identifier for this aspect
    pub id: String,
    /// Human-readable display name
    pub name: String,
    /// Properties directly owned by this aspect
    pub properties: Vec<Property>,
    /// IDs of direct child aspects.
    pub sub_aspects: Vec<String>,
}

impl Aspect {
    /// Construct a new aspect with no properties or sub-aspects.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Aspect {
            id: id.into(),
            name: name.into(),
            properties: Vec::new(),
            sub_aspects: Vec::new(),
        }
    }

    /// Add a property.
    pub fn with_property(mut self, prop: Property) -> Self {
        self.properties.push(prop);
        self
    }

    /// Add a sub-aspect id reference.
    pub fn with_sub_aspect(mut self, child_id: impl Into<String>) -> Self {
        self.sub_aspects.push(child_id.into());
        self
    }
}

// ─────────────────────────────────────────────────
// PropertyChain
// ─────────────────────────────────────────────────

/// A sequence of step names used to navigate through the aspect hierarchy
/// and finally resolve to a property.
#[derive(Debug, Clone)]
pub struct PropertyChain {
    /// Ordered list of step identifiers (aspect ids or property names)
    pub steps: Vec<String>,
}

impl PropertyChain {
    /// Construct a chain from a list of step names.
    pub fn new(steps: Vec<impl Into<String>>) -> Self {
        PropertyChain {
            steps: steps.into_iter().map(Into::into).collect(),
        }
    }

    /// Single-step chain.
    pub fn single(step: impl Into<String>) -> Self {
        PropertyChain {
            steps: vec![step.into()],
        }
    }
}

// ─────────────────────────────────────────────────
// ChainResult
// ─────────────────────────────────────────────────

/// The result of resolving a property chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainResult {
    /// Full path of ids/names traversed.
    pub path: Vec<String>,
    /// Resolved datatype of the terminal property.
    pub value_type: String,
    /// Whether any property in the chain is optional.
    pub optional_in_chain: bool,
}

// ─────────────────────────────────────────────────
// ChainError
// ─────────────────────────────────────────────────

/// Error encountered while resolving a property chain.
#[derive(Debug, PartialEq, Eq)]
pub enum ChainError {
    /// No aspect with the given identifier exists in the registry
    AspectNotFound(String),
    /// The named property was not found in the specified aspect
    PropertyNotFound(String),
    /// A cycle was detected at the given aspect id during traversal
    CycleDetected(String),
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainError::AspectNotFound(id) => write!(f, "Aspect not found: {id}"),
            ChainError::PropertyNotFound(name) => write!(f, "Property not found: {name}"),
            ChainError::CycleDetected(id) => write!(f, "Cycle detected at: {id}"),
        }
    }
}

// ─────────────────────────────────────────────────
// AspectChain
// ─────────────────────────────────────────────────

/// Registry of aspects, supporting hierarchical property chain resolution.
#[derive(Debug, Default)]
pub struct AspectChain {
    aspects: HashMap<String, Aspect>,
}

impl AspectChain {
    /// Create an empty registry.
    pub fn new() -> Self {
        AspectChain {
            aspects: HashMap::new(),
        }
    }

    /// Register an aspect.
    pub fn add_aspect(&mut self, aspect: Aspect) {
        self.aspects.insert(aspect.id.clone(), aspect);
    }

    /// Look up an aspect by id.
    pub fn get_aspect(&self, id: &str) -> Option<&Aspect> {
        self.aspects.get(id)
    }

    /// Number of registered aspects.
    pub fn aspect_count(&self) -> usize {
        self.aspects.len()
    }

    /// Resolve a property chain starting from `root_id`.
    ///
    /// Each step in the chain is tried as a property name on the current aspect
    /// first; if it is the last step, it must be a property. If not the last step,
    /// it may be a sub-aspect id or a property leading to another aspect.
    ///
    /// This implementation: navigates by sub-aspect id when step matches a sub-aspect,
    /// and resolves the final step as a property name on the current aspect.
    pub fn resolve_chain(
        &self,
        root_id: &str,
        chain: &PropertyChain,
    ) -> Result<ChainResult, ChainError> {
        if chain.steps.is_empty() {
            return Err(ChainError::PropertyNotFound("<empty chain>".to_string()));
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut path: Vec<String> = Vec::new();
        let mut optional_in_chain = false;

        let mut current_id = root_id.to_string();
        visited.insert(current_id.clone());
        path.push(current_id.clone());

        let steps = &chain.steps;
        let last_idx = steps.len() - 1;

        for (i, step) in steps.iter().enumerate() {
            let aspect = self
                .aspects
                .get(&current_id)
                .ok_or_else(|| ChainError::AspectNotFound(current_id.clone()))?;

            if i == last_idx {
                // Final step: must resolve to a property
                let prop = aspect
                    .properties
                    .iter()
                    .find(|p| p.name == *step)
                    .ok_or_else(|| ChainError::PropertyNotFound(step.clone()))?;
                if prop.optional {
                    optional_in_chain = true;
                }
                path.push(step.clone());
                return Ok(ChainResult {
                    path,
                    value_type: prop.datatype.clone(),
                    optional_in_chain,
                });
            } else {
                // Intermediate step: navigate to sub-aspect
                let child_id = aspect
                    .sub_aspects
                    .iter()
                    .find(|id| id.as_str() == step)
                    .cloned()
                    .ok_or_else(|| ChainError::AspectNotFound(step.clone()))?;

                if visited.contains(&child_id) {
                    return Err(ChainError::CycleDetected(child_id));
                }
                visited.insert(child_id.clone());
                path.push(child_id.clone());
                current_id = child_id;
            }
        }

        Err(ChainError::PropertyNotFound("<no steps>".to_string()))
    }

    /// Return all (aspect_id, property) pairs reachable from `aspect_id`,
    /// including sub-aspects recursively.  Cycle-safe.
    pub fn all_properties<'a>(&'a self, aspect_id: &'a str) -> Vec<(&'a str, &'a Property)> {
        let mut result = Vec::new();
        let mut visited: HashSet<&'a str> = HashSet::new();
        self.collect_properties(aspect_id, &mut visited, &mut result);
        result
    }

    fn collect_properties<'a>(
        &'a self,
        aspect_id: &'a str,
        visited: &mut HashSet<&'a str>,
        result: &mut Vec<(&'a str, &'a Property)>,
    ) {
        if visited.contains(aspect_id) {
            return;
        }
        visited.insert(aspect_id);

        if let Some(aspect) = self.aspects.get(aspect_id) {
            for prop in &aspect.properties {
                result.push((aspect_id, prop));
            }
            for child_id in &aspect.sub_aspects {
                self.collect_properties(child_id, visited, result);
            }
        }
    }

    /// Compute the maximum depth of the hierarchy rooted at `aspect_id`.
    /// Depth 0 means a leaf (no sub-aspects), depth N means N levels deep.
    pub fn depth(&self, aspect_id: &str) -> usize {
        let mut visited: HashSet<&str> = HashSet::new();
        self.compute_depth(aspect_id, &mut visited)
    }

    fn compute_depth<'a>(&'a self, aspect_id: &'a str, visited: &mut HashSet<&'a str>) -> usize {
        if visited.contains(aspect_id) {
            return 0; // Cycle guard
        }
        visited.insert(aspect_id);

        match self.aspects.get(aspect_id) {
            None => 0,
            Some(aspect) if aspect.sub_aspects.is_empty() => 0,
            Some(aspect) => {
                let max_child = aspect
                    .sub_aspects
                    .iter()
                    .map(|id| self.compute_depth(id, visited))
                    .max()
                    .unwrap_or(0);
                1 + max_child
            }
        }
    }

    /// Flatten all properties from `aspect_id` and all its descendants
    /// into a single vec (depth-first, cycle-safe).
    pub fn flatten(&self, aspect_id: &str) -> Vec<Property> {
        let mut result = Vec::new();
        let mut visited: HashSet<&str> = HashSet::new();
        self.collect_flat(aspect_id, &mut visited, &mut result);
        result
    }

    fn collect_flat<'a>(
        &'a self,
        aspect_id: &'a str,
        visited: &mut HashSet<&'a str>,
        result: &mut Vec<Property>,
    ) {
        if visited.contains(aspect_id) {
            return;
        }
        visited.insert(aspect_id);

        if let Some(aspect) = self.aspects.get(aspect_id) {
            for prop in &aspect.properties {
                result.push(prop.clone());
            }
            for child_id in &aspect.sub_aspects {
                self.collect_flat(child_id, visited, result);
            }
        }
    }

    /// Detect whether there is a cycle reachable from `aspect_id`.
    pub fn has_cycle(&self, aspect_id: &str) -> bool {
        let mut on_stack: HashSet<&str> = HashSet::new();
        self.detect_cycle(aspect_id, &mut on_stack)
    }

    fn detect_cycle<'a>(&'a self, aspect_id: &'a str, on_stack: &mut HashSet<&'a str>) -> bool {
        if on_stack.contains(aspect_id) {
            return true;
        }
        on_stack.insert(aspect_id);
        if let Some(aspect) = self.aspects.get(aspect_id) {
            for child_id in &aspect.sub_aspects {
                if self.detect_cycle(child_id, on_stack) {
                    on_stack.remove(aspect_id);
                    return true;
                }
            }
        }
        on_stack.remove(aspect_id);
        false
    }
}

// ─────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain(steps: &[&str]) -> PropertyChain {
        PropertyChain::new(steps.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    // ── Property tests ─────────────────────────────────────────

    #[test]
    fn test_property_new() {
        let p = Property::new("name", "xsd:string");
        assert_eq!(p.name, "name");
        assert_eq!(p.datatype, "xsd:string");
        assert!(!p.optional);
        assert!(p.description.is_none());
    }

    #[test]
    fn test_property_optional() {
        let p = Property::new("age", "xsd:integer").optional();
        assert!(p.optional);
    }

    #[test]
    fn test_property_with_description() {
        let p = Property::new("label", "xsd:string").with_description("A label");
        assert_eq!(p.description, Some("A label".to_string()));
    }

    // ── Aspect tests ───────────────────────────────────────────

    #[test]
    fn test_aspect_new() {
        let a = Aspect::new("a1", "MyAspect");
        assert_eq!(a.id, "a1");
        assert_eq!(a.name, "MyAspect");
        assert!(a.properties.is_empty());
        assert!(a.sub_aspects.is_empty());
    }

    #[test]
    fn test_aspect_with_property() {
        let a = Aspect::new("a1", "A").with_property(Property::new("p", "string"));
        assert_eq!(a.properties.len(), 1);
    }

    #[test]
    fn test_aspect_with_sub_aspect() {
        let a = Aspect::new("a1", "A").with_sub_aspect("a2");
        assert_eq!(a.sub_aspects, vec!["a2"]);
    }

    // ── AspectChain basics ─────────────────────────────────────

    #[test]
    fn test_aspect_chain_empty() {
        let ac = AspectChain::new();
        assert_eq!(ac.aspect_count(), 0);
    }

    #[test]
    fn test_add_aspect_increases_count() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("a1", "A"));
        assert_eq!(ac.aspect_count(), 1);
    }

    #[test]
    fn test_get_aspect_found() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("a1", "A"));
        assert!(ac.get_aspect("a1").is_some());
    }

    #[test]
    fn test_get_aspect_not_found() {
        let ac = AspectChain::new();
        assert!(ac.get_aspect("missing").is_none());
    }

    #[test]
    fn test_aspect_chain_default() {
        let ac = AspectChain::default();
        assert_eq!(ac.aspect_count(), 0);
    }

    // ── Single-step chain resolution ──────────────────────────

    #[test]
    fn test_resolve_chain_single_step() {
        let mut ac = AspectChain::new();
        ac.add_aspect(
            Aspect::new("root", "Root").with_property(Property::new("name", "xsd:string")),
        );
        let chain = make_chain(&["name"]);
        let result = ac.resolve_chain("root", &chain).unwrap();
        assert_eq!(result.value_type, "xsd:string");
        assert!(!result.optional_in_chain);
    }

    #[test]
    fn test_resolve_chain_single_optional() {
        let mut ac = AspectChain::new();
        ac.add_aspect(
            Aspect::new("root", "Root")
                .with_property(Property::new("nickname", "xsd:string").optional()),
        );
        let chain = make_chain(&["nickname"]);
        let result = ac.resolve_chain("root", &chain).unwrap();
        assert!(result.optional_in_chain);
    }

    #[test]
    fn test_resolve_chain_property_not_found() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("root", "Root"));
        let chain = make_chain(&["missing"]);
        let err = ac.resolve_chain("root", &chain).unwrap_err();
        assert_eq!(err, ChainError::PropertyNotFound("missing".to_string()));
    }

    #[test]
    fn test_resolve_chain_aspect_not_found() {
        let ac = AspectChain::new();
        let chain = make_chain(&["p"]);
        let err = ac.resolve_chain("nonexistent", &chain).unwrap_err();
        assert_eq!(err, ChainError::AspectNotFound("nonexistent".to_string()));
    }

    // ── Multi-step chain resolution ────────────────────────────

    #[test]
    fn test_resolve_chain_two_steps() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("root", "Root").with_sub_aspect("child"));
        ac.add_aspect(
            Aspect::new("child", "Child").with_property(Property::new("value", "xsd:integer")),
        );
        let chain = make_chain(&["child", "value"]);
        let result = ac.resolve_chain("root", &chain).unwrap();
        assert_eq!(result.value_type, "xsd:integer");
    }

    #[test]
    fn test_resolve_chain_three_steps() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("root", "Root").with_sub_aspect("mid"));
        ac.add_aspect(Aspect::new("mid", "Mid").with_sub_aspect("leaf"));
        ac.add_aspect(
            Aspect::new("leaf", "Leaf").with_property(Property::new("data", "xsd:float")),
        );
        let chain = make_chain(&["mid", "leaf", "data"]);
        let result = ac.resolve_chain("root", &chain).unwrap();
        assert_eq!(result.value_type, "xsd:float");
        assert_eq!(result.path.len(), 4); // root, mid, leaf, data
    }

    #[test]
    fn test_resolve_chain_intermediate_not_a_sub_aspect() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("root", "Root").with_sub_aspect("child"));
        ac.add_aspect(
            Aspect::new("child", "Child").with_property(Property::new("val", "xsd:string")),
        );
        // "wrongchild" is not a sub-aspect of root
        let chain = make_chain(&["wrongchild", "val"]);
        let err = ac.resolve_chain("root", &chain).unwrap_err();
        assert!(matches!(err, ChainError::AspectNotFound(_)));
    }

    // ── Cycle detection ────────────────────────────────────────

    #[test]
    fn test_has_cycle_no_cycle() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("a", "A").with_sub_aspect("b"));
        ac.add_aspect(Aspect::new("b", "B"));
        assert!(!ac.has_cycle("a"));
    }

    #[test]
    fn test_has_cycle_direct_cycle() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("a", "A").with_sub_aspect("b"));
        ac.add_aspect(Aspect::new("b", "B").with_sub_aspect("a")); // cycle: a→b→a
        assert!(ac.has_cycle("a"));
    }

    #[test]
    fn test_has_cycle_self_loop() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("a", "A").with_sub_aspect("a")); // self-loop
        assert!(ac.has_cycle("a"));
    }

    #[test]
    fn test_resolve_chain_returns_cycle_error() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("root", "Root").with_sub_aspect("child"));
        ac.add_aspect(Aspect::new("child", "Child").with_sub_aspect("root")); // cycle
        let chain = make_chain(&["child", "root", "p"]);
        let err = ac.resolve_chain("root", &chain).unwrap_err();
        assert!(matches!(err, ChainError::CycleDetected(_)));
    }

    // ── all_properties ────────────────────────────────────────

    #[test]
    fn test_all_properties_leaf() {
        let mut ac = AspectChain::new();
        ac.add_aspect(
            Aspect::new("a", "A")
                .with_property(Property::new("p1", "t1"))
                .with_property(Property::new("p2", "t2")),
        );
        let props = ac.all_properties("a");
        assert_eq!(props.len(), 2);
    }

    #[test]
    fn test_all_properties_recursive() {
        let mut ac = AspectChain::new();
        ac.add_aspect(
            Aspect::new("root", "Root")
                .with_property(Property::new("rp", "string"))
                .with_sub_aspect("child"),
        );
        ac.add_aspect(Aspect::new("child", "Child").with_property(Property::new("cp", "int")));
        let props = ac.all_properties("root");
        assert_eq!(props.len(), 2);
        let names: Vec<_> = props.iter().map(|(_, p)| p.name.as_str()).collect();
        assert!(names.contains(&"rp"));
        assert!(names.contains(&"cp"));
    }

    #[test]
    fn test_all_properties_cycle_safe() {
        let mut ac = AspectChain::new();
        ac.add_aspect(
            Aspect::new("a", "A")
                .with_property(Property::new("pa", "string"))
                .with_sub_aspect("b"),
        );
        ac.add_aspect(
            Aspect::new("b", "B")
                .with_property(Property::new("pb", "int"))
                .with_sub_aspect("a"), // cycle
        );
        // Should not infinite-loop
        let props = ac.all_properties("a");
        assert!(!props.is_empty());
    }

    // ── depth ─────────────────────────────────────────────────

    #[test]
    fn test_depth_leaf() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("leaf", "Leaf"));
        assert_eq!(ac.depth("leaf"), 0);
    }

    #[test]
    fn test_depth_one_level() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("root", "Root").with_sub_aspect("leaf"));
        ac.add_aspect(Aspect::new("leaf", "Leaf"));
        assert_eq!(ac.depth("root"), 1);
    }

    #[test]
    fn test_depth_two_levels() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("root", "Root").with_sub_aspect("mid"));
        ac.add_aspect(Aspect::new("mid", "Mid").with_sub_aspect("leaf"));
        ac.add_aspect(Aspect::new("leaf", "Leaf"));
        assert_eq!(ac.depth("root"), 2);
    }

    #[test]
    fn test_depth_missing_aspect_zero() {
        let ac = AspectChain::new();
        assert_eq!(ac.depth("nonexistent"), 0);
    }

    // ── flatten ───────────────────────────────────────────────

    #[test]
    fn test_flatten_single_aspect() {
        let mut ac = AspectChain::new();
        ac.add_aspect(
            Aspect::new("a", "A")
                .with_property(Property::new("p1", "t1"))
                .with_property(Property::new("p2", "t2")),
        );
        let flat = ac.flatten("a");
        assert_eq!(flat.len(), 2);
    }

    #[test]
    fn test_flatten_recursive() {
        let mut ac = AspectChain::new();
        ac.add_aspect(
            Aspect::new("root", "Root")
                .with_property(Property::new("rp", "t"))
                .with_sub_aspect("child"),
        );
        ac.add_aspect(
            Aspect::new("child", "Child")
                .with_property(Property::new("cp1", "t"))
                .with_property(Property::new("cp2", "t")),
        );
        let flat = ac.flatten("root");
        assert_eq!(flat.len(), 3);
    }

    #[test]
    fn test_flatten_empty_hierarchy() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("a", "A"));
        let flat = ac.flatten("a");
        assert!(flat.is_empty());
    }

    // ── ChainError Display ────────────────────────────────────

    #[test]
    fn test_chain_error_display_aspect_not_found() {
        let err = ChainError::AspectNotFound("myId".to_string());
        assert!(err.to_string().contains("myId"));
    }

    #[test]
    fn test_chain_error_display_property_not_found() {
        let err = ChainError::PropertyNotFound("myProp".to_string());
        assert!(err.to_string().contains("myProp"));
    }

    #[test]
    fn test_chain_error_display_cycle() {
        let err = ChainError::CycleDetected("nodeX".to_string());
        assert!(err.to_string().contains("nodeX"));
    }

    // ── Path contents ─────────────────────────────────────────

    #[test]
    fn test_resolved_path_contains_root() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("root", "Root").with_property(Property::new("p", "string")));
        let result = ac.resolve_chain("root", &make_chain(&["p"])).unwrap();
        assert!(result.path.contains(&"root".to_string()));
    }

    #[test]
    fn test_resolved_path_two_step() {
        let mut ac = AspectChain::new();
        ac.add_aspect(Aspect::new("root", "Root").with_sub_aspect("child"));
        ac.add_aspect(Aspect::new("child", "Child").with_property(Property::new("x", "int")));
        let result = ac
            .resolve_chain("root", &make_chain(&["child", "x"]))
            .unwrap();
        assert_eq!(result.path, vec!["root", "child", "x"]);
    }
}
