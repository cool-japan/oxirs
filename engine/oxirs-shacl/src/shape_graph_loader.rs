/// Loading and indexing of SHACL shape graphs.
///
/// Provides in-memory representation and index structures for SHACL shapes,
/// supporting efficient lookup by class target, type, and property path.
use std::collections::HashMap;

/// Discriminator for shape kinds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeType {
    /// An `sh:NodeShape`.
    NodeShape,
    /// An `sh:PropertyShape`.
    PropertyShape,
}

/// A single SHACL constraint attached to a shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeConstraint {
    /// Constraint keyword (e.g. "sh:minCount", "sh:datatype").
    pub constraint_type: String,
    /// Constraint value as string.
    pub value: String,
}

/// A node in the SHACL shape graph.
#[derive(Debug, Clone)]
pub struct ShapeNode {
    /// IRI identifier of the shape.
    pub id: String,
    /// Whether this is a node or property shape.
    pub shape_type: ShapeType,
    /// `sh:targetClass` IRI (NodeShape only).
    pub target_class: Option<String>,
    /// `sh:targetNode` IRI (NodeShape only).
    pub target_node: Option<String>,
    /// Constraints attached directly to this shape.
    pub constraints: Vec<ShapeConstraint>,
    /// IRIs of child property shapes (for NodeShape).
    pub property_shapes: Vec<String>,
    /// `sh:path` (PropertyShape only).
    pub path: Option<String>,
    /// Whether the shape is `sh:deactivated true`.
    pub deactivated: bool,
}

/// An in-memory SHACL shape graph with fast lookup indices.
#[derive(Debug, Default)]
pub struct ShapeGraph {
    shapes: HashMap<String, ShapeNode>,
}

/// Result of loading shapes into the graph.
#[derive(Debug, Clone)]
pub struct LoadResult {
    /// Number of shapes successfully loaded.
    pub shapes_loaded: usize,
    /// Non-fatal warnings (e.g. deprecated features).
    pub warnings: Vec<String>,
}

impl ShapeGraph {
    /// Create an empty shape graph.
    pub fn new() -> Self {
        Self {
            shapes: HashMap::new(),
        }
    }

    /// Add a shape to the graph.
    ///
    /// Returns `Err(ShapeGraphError::DuplicateId)` if a shape with the same ID already exists.
    pub fn add_shape(&mut self, shape: ShapeNode) -> Result<(), ShapeGraphError> {
        if self.shapes.contains_key(&shape.id) {
            return Err(ShapeGraphError::DuplicateId(shape.id.clone()));
        }
        self.shapes.insert(shape.id.clone(), shape);
        Ok(())
    }

    /// Look up a shape by IRI.
    pub fn get_shape(&self, id: &str) -> Option<&ShapeNode> {
        self.shapes.get(id)
    }

    /// Return all `NodeShape`s.
    pub fn node_shapes(&self) -> Vec<&ShapeNode> {
        self.shapes
            .values()
            .filter(|s| s.shape_type == ShapeType::NodeShape)
            .collect()
    }

    /// Return all `PropertyShape`s.
    pub fn property_shapes(&self) -> Vec<&ShapeNode> {
        self.shapes
            .values()
            .filter(|s| s.shape_type == ShapeType::PropertyShape)
            .collect()
    }

    /// Return all node shapes whose `sh:targetClass` matches `class_iri`.
    pub fn shapes_for_class(&self, class_iri: &str) -> Vec<&ShapeNode> {
        self.shapes
            .values()
            .filter(|s| {
                s.shape_type == ShapeType::NodeShape && s.target_class.as_deref() == Some(class_iri)
            })
            .collect()
    }

    /// Resolve the property shapes referenced by a node shape.
    pub fn resolve_property_shapes(&self, node_shape_id: &str) -> Vec<&ShapeNode> {
        match self.shapes.get(node_shape_id) {
            None => vec![],
            Some(node_shape) => node_shape
                .property_shapes
                .iter()
                .filter_map(|ps_id| self.shapes.get(ps_id))
                .collect(),
        }
    }

    /// Return the total number of shapes.
    pub fn shape_count(&self) -> usize {
        self.shapes.len()
    }

    /// Validate internal consistency of the graph and return a list of warnings/errors.
    pub fn validate_graph(&self) -> Vec<String> {
        let mut issues = Vec::new();
        for shape in self.shapes.values() {
            // Warn if a property shape has no sh:path
            if shape.shape_type == ShapeType::PropertyShape && shape.path.is_none() {
                issues.push(format!(
                    "PropertyShape <{}> has no sh:path defined.",
                    shape.id
                ));
            }
            // Warn if a node shape references a property shape that does not exist
            for ps_id in &shape.property_shapes {
                if !self.shapes.contains_key(ps_id) {
                    issues.push(format!(
                        "NodeShape <{}> references missing PropertyShape <{}>.",
                        shape.id, ps_id
                    ));
                }
            }
            // Warn about deactivated shapes
            if shape.deactivated {
                issues.push(format!("Shape <{}> is deactivated.", shape.id));
            }
        }
        issues
    }

    /// Remove a shape by IRI. Returns true if it was present.
    pub fn remove_shape(&mut self, id: &str) -> bool {
        self.shapes.remove(id).is_some()
    }

    /// Return all shapes where `sh:deactivated` is true.
    pub fn deactivated_shapes(&self) -> Vec<&ShapeNode> {
        self.shapes.values().filter(|s| s.deactivated).collect()
    }
}

/// Errors from shape graph operations.
#[derive(Debug)]
pub enum ShapeGraphError {
    /// A shape with this IRI already exists in the graph.
    DuplicateId(String),
    /// The shape definition is structurally invalid.
    InvalidShape(String),
}

impl std::fmt::Display for ShapeGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateId(id) => write!(f, "Duplicate shape id: {id}"),
            Self::InvalidShape(msg) => write!(f, "Invalid shape: {msg}"),
        }
    }
}

impl std::error::Error for ShapeGraphError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn node_shape(id: &str, class: Option<&str>) -> ShapeNode {
        ShapeNode {
            id: id.to_string(),
            shape_type: ShapeType::NodeShape,
            target_class: class.map(|s| s.to_string()),
            target_node: None,
            constraints: vec![],
            property_shapes: vec![],
            path: None,
            deactivated: false,
        }
    }

    fn property_shape(id: &str, path: Option<&str>) -> ShapeNode {
        ShapeNode {
            id: id.to_string(),
            shape_type: ShapeType::PropertyShape,
            target_class: None,
            target_node: None,
            constraints: vec![],
            property_shapes: vec![],
            path: path.map(|s| s.to_string()),
            deactivated: false,
        }
    }

    // --- add/get ---

    #[test]
    fn test_add_and_get_shape() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(node_shape("http://s1", Some("http://Person")))
            .unwrap();
        assert!(sg.get_shape("http://s1").is_some());
    }

    #[test]
    fn test_get_missing_shape() {
        let sg = ShapeGraph::new();
        assert!(sg.get_shape("http://missing").is_none());
    }

    #[test]
    fn test_duplicate_id_error() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(node_shape("http://s1", None)).unwrap();
        let err = sg.add_shape(node_shape("http://s1", None));
        assert!(matches!(err, Err(ShapeGraphError::DuplicateId(_))));
    }

    #[test]
    fn test_shape_count_empty() {
        let sg = ShapeGraph::new();
        assert_eq!(sg.shape_count(), 0);
    }

    #[test]
    fn test_shape_count_after_adds() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(node_shape("http://s1", None)).unwrap();
        sg.add_shape(property_shape("http://ps1", Some("http://name")))
            .unwrap();
        assert_eq!(sg.shape_count(), 2);
    }

    // --- node_shapes / property_shapes ---

    #[test]
    fn test_node_shapes_filter() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(node_shape("http://ns1", None)).unwrap();
        sg.add_shape(property_shape("http://ps1", Some("http://p")))
            .unwrap();
        let ns = sg.node_shapes();
        assert_eq!(ns.len(), 1);
        assert_eq!(ns[0].id, "http://ns1");
    }

    #[test]
    fn test_property_shapes_filter() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(node_shape("http://ns1", None)).unwrap();
        sg.add_shape(property_shape("http://ps1", Some("http://p")))
            .unwrap();
        sg.add_shape(property_shape("http://ps2", Some("http://q")))
            .unwrap();
        let ps = sg.property_shapes();
        assert_eq!(ps.len(), 2);
    }

    // --- shapes_for_class ---

    #[test]
    fn test_shapes_for_class_match() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(node_shape("http://s1", Some("http://Person")))
            .unwrap();
        sg.add_shape(node_shape("http://s2", Some("http://Organization")))
            .unwrap();
        let result = sg.shapes_for_class("http://Person");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "http://s1");
    }

    #[test]
    fn test_shapes_for_class_no_match() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(node_shape("http://s1", Some("http://Person")))
            .unwrap();
        let result = sg.shapes_for_class("http://Animal");
        assert!(result.is_empty());
    }

    #[test]
    fn test_shapes_for_class_multiple() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(node_shape("http://s1", Some("http://Person")))
            .unwrap();
        sg.add_shape(node_shape("http://s2", Some("http://Person")))
            .unwrap();
        let result = sg.shapes_for_class("http://Person");
        assert_eq!(result.len(), 2);
    }

    // --- resolve_property_shapes ---

    #[test]
    fn test_resolve_property_shapes() {
        let mut sg = ShapeGraph::new();
        let mut ns = node_shape("http://ns1", None);
        ns.property_shapes = vec!["http://ps1".to_string()];
        sg.add_shape(ns).unwrap();
        sg.add_shape(property_shape("http://ps1", Some("http://name")))
            .unwrap();
        let resolved = sg.resolve_property_shapes("http://ns1");
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].id, "http://ps1");
    }

    #[test]
    fn test_resolve_property_shapes_missing_ref() {
        let mut sg = ShapeGraph::new();
        let mut ns = node_shape("http://ns1", None);
        ns.property_shapes = vec!["http://missing_ps".to_string()];
        sg.add_shape(ns).unwrap();
        let resolved = sg.resolve_property_shapes("http://ns1");
        assert!(resolved.is_empty());
    }

    #[test]
    fn test_resolve_property_shapes_unknown_node() {
        let sg = ShapeGraph::new();
        let resolved = sg.resolve_property_shapes("http://unknown");
        assert!(resolved.is_empty());
    }

    // --- validate_graph ---

    #[test]
    fn test_validate_no_warnings() {
        let mut sg = ShapeGraph::new();
        let mut ns = node_shape("http://ns1", None);
        ns.property_shapes = vec!["http://ps1".to_string()];
        sg.add_shape(ns).unwrap();
        sg.add_shape(property_shape("http://ps1", Some("http://name")))
            .unwrap();
        let warnings: Vec<_> = sg
            .validate_graph()
            .into_iter()
            .filter(|w| w.contains("missing"))
            .collect();
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_validate_missing_property_shape() {
        let mut sg = ShapeGraph::new();
        let mut ns = node_shape("http://ns1", None);
        ns.property_shapes = vec!["http://missing_ps".to_string()];
        sg.add_shape(ns).unwrap();
        let warnings = sg.validate_graph();
        assert!(warnings.iter().any(|w| w.contains("missing")));
    }

    #[test]
    fn test_validate_property_shape_no_path() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(property_shape("http://ps1", None)).unwrap();
        let warnings = sg.validate_graph();
        assert!(warnings.iter().any(|w| w.contains("no sh:path")));
    }

    #[test]
    fn test_validate_deactivated_shape_warning() {
        let mut sg = ShapeGraph::new();
        let mut s = node_shape("http://s1", None);
        s.deactivated = true;
        sg.add_shape(s).unwrap();
        let warnings = sg.validate_graph();
        assert!(warnings.iter().any(|w| w.contains("deactivated")));
    }

    // --- remove_shape ---

    #[test]
    fn test_remove_shape_present() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(node_shape("http://s1", None)).unwrap();
        assert!(sg.remove_shape("http://s1"));
        assert_eq!(sg.shape_count(), 0);
    }

    #[test]
    fn test_remove_shape_absent() {
        let mut sg = ShapeGraph::new();
        assert!(!sg.remove_shape("http://missing"));
    }

    // --- deactivated_shapes ---

    #[test]
    fn test_deactivated_shapes_none() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(node_shape("http://s1", None)).unwrap();
        assert!(sg.deactivated_shapes().is_empty());
    }

    #[test]
    fn test_deactivated_shapes_some() {
        let mut sg = ShapeGraph::new();
        let mut s = node_shape("http://s1", None);
        s.deactivated = true;
        sg.add_shape(s).unwrap();
        sg.add_shape(node_shape("http://s2", None)).unwrap();
        assert_eq!(sg.deactivated_shapes().len(), 1);
    }

    // --- constraints stored correctly ---

    #[test]
    fn test_constraints_stored() {
        let mut sg = ShapeGraph::new();
        let mut s = node_shape("http://s1", None);
        s.constraints = vec![ShapeConstraint {
            constraint_type: "sh:minCount".to_string(),
            value: "1".to_string(),
        }];
        sg.add_shape(s).unwrap();
        let stored = sg.get_shape("http://s1").unwrap();
        assert_eq!(stored.constraints.len(), 1);
        assert_eq!(stored.constraints[0].constraint_type, "sh:minCount");
        assert_eq!(stored.constraints[0].value, "1");
    }

    #[test]
    fn test_multiple_constraints() {
        let mut sg = ShapeGraph::new();
        let mut s = property_shape("http://ps1", Some("http://name"));
        s.constraints = vec![
            ShapeConstraint {
                constraint_type: "sh:minCount".to_string(),
                value: "1".to_string(),
            },
            ShapeConstraint {
                constraint_type: "sh:datatype".to_string(),
                value: "xsd:string".to_string(),
            },
        ];
        sg.add_shape(s).unwrap();
        let stored = sg.get_shape("http://ps1").unwrap();
        assert_eq!(stored.constraints.len(), 2);
    }

    // --- error display ---

    #[test]
    fn test_duplicate_id_error_display() {
        let err = ShapeGraphError::DuplicateId("http://s1".to_string());
        assert!(format!("{err}").contains("http://s1"));
    }

    #[test]
    fn test_invalid_shape_error_display() {
        let err = ShapeGraphError::InvalidShape("bad".to_string());
        assert!(format!("{err}").contains("bad"));
    }

    // --- shape_type stored ---

    #[test]
    fn test_shape_type_node() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(node_shape("http://ns1", None)).unwrap();
        assert_eq!(
            sg.get_shape("http://ns1").unwrap().shape_type,
            ShapeType::NodeShape
        );
    }

    #[test]
    fn test_shape_type_property() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(property_shape("http://ps1", Some("http://p")))
            .unwrap();
        assert_eq!(
            sg.get_shape("http://ps1").unwrap().shape_type,
            ShapeType::PropertyShape
        );
    }

    #[test]
    fn test_target_node_stored() {
        let mut sg = ShapeGraph::new();
        let mut s = node_shape("http://ns1", None);
        s.target_node = Some("http://alice".to_string());
        sg.add_shape(s).unwrap();
        assert_eq!(
            sg.get_shape("http://ns1").unwrap().target_node.as_deref(),
            Some("http://alice")
        );
    }

    #[test]
    fn test_path_stored_in_property_shape() {
        let mut sg = ShapeGraph::new();
        sg.add_shape(property_shape("http://ps1", Some("http://schema.org/name")))
            .unwrap();
        assert_eq!(
            sg.get_shape("http://ps1").unwrap().path.as_deref(),
            Some("http://schema.org/name")
        );
    }

    #[test]
    fn test_shape_graph_default() {
        let sg = ShapeGraph::default();
        assert_eq!(sg.shape_count(), 0);
    }
}
