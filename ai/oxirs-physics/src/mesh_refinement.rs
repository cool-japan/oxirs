//! Adaptive mesh refinement for FEM (Finite Element Method)
//!
//! This module implements 2D triangular mesh refinement using the
//! 4-triangle subdivision (midpoint refinement) approach. Elements that
//! exceed quality thresholds or have large error estimates are subdivided
//! into 4 congruent sub-triangles by inserting midpoints on each edge.
//!
//! # Example
//!
//! ```rust
//! use oxirs_physics::mesh_refinement::{Mesh2D, RefinementCriteria};
//! use std::collections::HashMap;
//!
//! let mut mesh = Mesh2D::new();
//! let n0 = mesh.add_node(0.0, 0.0);
//! let n1 = mesh.add_node(1.0, 0.0);
//! let n2 = mesh.add_node(0.5, 1.0);
//! let _e0 = mesh.add_element(n0, n1, n2);
//!
//! let criteria = RefinementCriteria {
//!     max_edge_length: 0.6,
//!     max_aspect_ratio: 10.0,
//!     min_angle_deg: 5.0,
//!     error_threshold: 0.1,
//! };
//! let error_map: HashMap<u32, f64> = HashMap::new();
//! let stats = mesh.refine_mesh(&criteria, &error_map);
//! assert!(stats.elements_refined > 0);
//! ```

use std::collections::HashMap;

/// Maximum refinement level to prevent infinite subdivision
pub const MAX_REFINEMENT_LEVEL: u8 = 8;

/// A 2D mesh node with coordinates
#[derive(Debug, Clone, PartialEq)]
pub struct MeshNode2D {
    pub id: u32,
    pub x: f64,
    pub y: f64,
}

impl MeshNode2D {
    /// Create a new mesh node
    pub fn new(id: u32, x: f64, y: f64) -> Self {
        Self { id, x, y }
    }
}

/// A triangular mesh element referencing three node IDs
#[derive(Debug, Clone, PartialEq)]
pub struct MeshElement {
    pub id: u32,
    /// Node IDs forming the triangle (counter-clockwise ordering preferred)
    pub nodes: [u32; 3],
    /// Current subdivision level (0 = original)
    pub refinement_level: u8,
}

impl MeshElement {
    /// Create a new triangular element
    pub fn new(id: u32, n0: u32, n1: u32, n2: u32) -> Self {
        Self {
            id,
            nodes: [n0, n1, n2],
            refinement_level: 0,
        }
    }

    /// Create a refined child element with incremented level
    pub fn child(id: u32, n0: u32, n1: u32, n2: u32, parent_level: u8) -> Self {
        Self {
            id,
            nodes: [n0, n1, n2],
            refinement_level: parent_level.saturating_add(1),
        }
    }
}

/// Criteria controlling when elements should be refined
#[derive(Debug, Clone)]
pub struct RefinementCriteria {
    /// Maximum allowed edge length before refinement is triggered
    pub max_edge_length: f64,
    /// Maximum allowed aspect ratio (longest/shortest edge)
    pub max_aspect_ratio: f64,
    /// Minimum allowed interior angle in degrees
    pub min_angle_deg: f64,
    /// Error threshold — elements with error > this value are refined
    pub error_threshold: f64,
}

impl RefinementCriteria {
    /// Default criteria suitable for general FEM meshes
    pub fn default_criteria() -> Self {
        Self {
            max_edge_length: 1.0,
            max_aspect_ratio: 5.0,
            min_angle_deg: 10.0,
            error_threshold: 0.05,
        }
    }
}

/// Statistics from a single refinement pass
#[derive(Debug, Clone, Default)]
pub struct RefinementStats {
    /// Number of elements that were subdivided
    pub elements_refined: u32,
    /// Number of new nodes inserted during refinement
    pub nodes_added: u32,
    /// Maximum refinement level reached in the mesh after this pass
    pub max_level: u8,
}

/// 2D triangular mesh supporting adaptive refinement
#[derive(Debug, Clone)]
pub struct Mesh2D {
    /// All nodes indexed by their ID
    pub nodes: HashMap<u32, MeshNode2D>,
    /// All elements indexed by their ID
    pub elements: HashMap<u32, MeshElement>,
    /// Next available node ID
    pub next_node_id: u32,
    /// Next available element ID
    pub next_elem_id: u32,
}

impl Mesh2D {
    /// Create an empty mesh
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            elements: HashMap::new(),
            next_node_id: 0,
            next_elem_id: 0,
        }
    }

    /// Add a node at (x, y) and return its assigned ID
    pub fn add_node(&mut self, x: f64, y: f64) -> u32 {
        let id = self.next_node_id;
        self.nodes.insert(id, MeshNode2D::new(id, x, y));
        self.next_node_id += 1;
        id
    }

    /// Add a triangular element referencing three existing node IDs
    /// Returns the new element ID
    pub fn add_element(&mut self, n0: u32, n1: u32, n2: u32) -> u32 {
        let id = self.next_elem_id;
        self.elements.insert(id, MeshElement::new(id, n0, n1, n2));
        self.next_elem_id += 1;
        id
    }

    /// Compute the signed area of a triangle element
    /// Returns 0.0 if any referenced node is missing
    pub fn element_area(&self, elem: &MeshElement) -> f64 {
        let (n0, n1, n2) = match (
            self.nodes.get(&elem.nodes[0]),
            self.nodes.get(&elem.nodes[1]),
            self.nodes.get(&elem.nodes[2]),
        ) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => return 0.0,
        };
        let area = 0.5 * ((n1.x - n0.x) * (n2.y - n0.y) - (n2.x - n0.x) * (n1.y - n0.y));
        area.abs()
    }

    /// Compute the three edge lengths [|e01|, |e12|, |e20|]
    /// Returns `[0,0,0]` if any node is missing
    pub fn element_edge_lengths(&self, elem: &MeshElement) -> [f64; 3] {
        let (n0, n1, n2) = match (
            self.nodes.get(&elem.nodes[0]),
            self.nodes.get(&elem.nodes[1]),
            self.nodes.get(&elem.nodes[2]),
        ) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => return [0.0, 0.0, 0.0],
        };
        let e01 = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();
        let e12 = ((n2.x - n1.x).powi(2) + (n2.y - n1.y).powi(2)).sqrt();
        let e20 = ((n0.x - n2.x).powi(2) + (n0.y - n2.y).powi(2)).sqrt();
        [e01, e12, e20]
    }

    /// Compute the aspect ratio as longest_edge / shortest_edge
    /// Returns f64::INFINITY for degenerate elements
    pub fn element_aspect_ratio(&self, elem: &MeshElement) -> f64 {
        let edges = self.element_edge_lengths(elem);
        let min_e = edges.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_e = edges.iter().cloned().fold(0.0_f64, f64::max);
        if min_e <= 0.0 {
            f64::INFINITY
        } else {
            max_e / min_e
        }
    }

    /// Compute the minimum interior angle in degrees
    /// Uses the law of cosines; returns 0.0 for degenerate elements
    pub fn element_min_angle_deg(&self, elem: &MeshElement) -> f64 {
        let (n0, n1, n2) = match (
            self.nodes.get(&elem.nodes[0]),
            self.nodes.get(&elem.nodes[1]),
            self.nodes.get(&elem.nodes[2]),
        ) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => return 0.0,
        };

        let edges = self.element_edge_lengths(elem);
        let a = edges[0]; // opposite n2
        let b = edges[1]; // opposite n0
        let c = edges[2]; // opposite n1

        // Avoid division by zero for degenerate triangles
        if a <= 0.0 || b <= 0.0 || c <= 0.0 {
            return 0.0;
        }

        // Angle at each vertex using law of cosines
        let cos_n0 = ((b * b + c * c - a * a) / (2.0 * b * c)).clamp(-1.0, 1.0);
        let cos_n1 = ((a * a + c * c - b * b) / (2.0 * a * c)).clamp(-1.0, 1.0);
        let cos_n2 = ((a * a + b * b - c * c) / (2.0 * a * b)).clamp(-1.0, 1.0);

        // Suppress unused variable warning — n0/n1/n2 only used for existence check above
        let _ = (n0, n1, n2);

        let angle_n0 = cos_n0.acos().to_degrees();
        let angle_n1 = cos_n1.acos().to_degrees();
        let angle_n2 = cos_n2.acos().to_degrees();

        angle_n0.min(angle_n1).min(angle_n2)
    }

    /// Compute the midpoint coordinates between two nodes
    /// Returns (0,0) if either node is missing
    pub fn midpoint(&self, n1: u32, n2: u32) -> (f64, f64) {
        match (self.nodes.get(&n1), self.nodes.get(&n2)) {
            (Some(a), Some(b)) => ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0),
            _ => (0.0, 0.0),
        }
    }

    /// Refine a single element by bisecting all three edges.
    ///
    /// Inserts midpoint nodes on each edge and creates 4 congruent sub-triangles:
    /// ```text
    ///        n2
    ///       /  \
    ///      m20--m12
    ///     / \  / \
    ///   n0--m01--n1
    /// ```
    /// Returns the IDs of the 4 new child elements.
    /// Returns an empty array if the element is not found or already at MAX_REFINEMENT_LEVEL.
    pub fn refine_element(&mut self, elem_id: u32) -> [u32; 4] {
        // Clone what we need to avoid simultaneous borrow
        let elem = match self.elements.get(&elem_id).cloned() {
            Some(e) => e,
            None => return [0; 4],
        };

        if elem.refinement_level >= MAX_REFINEMENT_LEVEL {
            return [0; 4];
        }

        let [n0, n1, n2] = elem.nodes;

        // Insert midpoint nodes
        let (mx01, my01) = self.midpoint(n0, n1);
        let (mx12, my12) = self.midpoint(n1, n2);
        let (mx20, my20) = self.midpoint(n2, n0);

        let m01 = self.add_node(mx01, my01);
        let m12 = self.add_node(mx12, my12);
        let m20 = self.add_node(mx20, my20);

        let lvl = elem.refinement_level;

        // Create 4 child elements
        let c0_id = self.next_elem_id;
        self.elements
            .insert(c0_id, MeshElement::child(c0_id, n0, m01, m20, lvl));
        self.next_elem_id += 1;

        let c1_id = self.next_elem_id;
        self.elements
            .insert(c1_id, MeshElement::child(c1_id, m01, n1, m12, lvl));
        self.next_elem_id += 1;

        let c2_id = self.next_elem_id;
        self.elements
            .insert(c2_id, MeshElement::child(c2_id, m20, m12, n2, lvl));
        self.next_elem_id += 1;

        let c3_id = self.next_elem_id;
        self.elements
            .insert(c3_id, MeshElement::child(c3_id, m01, m12, m20, lvl));
        self.next_elem_id += 1;

        // Remove the parent element (replaced by 4 children)
        self.elements.remove(&elem_id);

        [c0_id, c1_id, c2_id, c3_id]
    }

    /// Determine whether an element should be refined given criteria and an error map
    pub fn should_refine(
        &self,
        elem: &MeshElement,
        criteria: &RefinementCriteria,
        error_map: &HashMap<u32, f64>,
    ) -> bool {
        if elem.refinement_level >= MAX_REFINEMENT_LEVEL {
            return false;
        }

        // Check user-supplied error estimate
        if let Some(&err) = error_map.get(&elem.id) {
            if err > criteria.error_threshold {
                return true;
            }
        }

        // Check geometric quality metrics
        let edges = self.element_edge_lengths(elem);
        let max_edge = edges.iter().cloned().fold(0.0_f64, f64::max);
        if max_edge > criteria.max_edge_length {
            return true;
        }

        let aspect = self.element_aspect_ratio(elem);
        if aspect > criteria.max_aspect_ratio {
            return true;
        }

        let min_angle = self.element_min_angle_deg(elem);
        if min_angle < criteria.min_angle_deg && min_angle > 0.0 {
            return true;
        }

        false
    }

    /// Perform one global refinement pass: refine all elements meeting the criteria.
    ///
    /// Returns statistics about this refinement pass.
    pub fn refine_mesh(
        &mut self,
        criteria: &RefinementCriteria,
        error_map: &HashMap<u32, f64>,
    ) -> RefinementStats {
        // Collect IDs of elements to refine (snapshot to avoid mutation during iteration)
        let to_refine: Vec<u32> = self
            .elements
            .values()
            .filter(|e| self.should_refine(e, criteria, error_map))
            .map(|e| e.id)
            .collect();

        let elements_refined = to_refine.len() as u32;
        let nodes_before = self.nodes.len() as u32;

        for id in to_refine {
            self.refine_element(id);
        }

        let nodes_added = self.nodes.len() as u32 - nodes_before;
        let max_level = self
            .elements
            .values()
            .map(|e| e.refinement_level)
            .max()
            .unwrap_or(0);

        RefinementStats {
            elements_refined,
            nodes_added,
            max_level,
        }
    }

    /// Return the current number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the current number of elements
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Compute the total mesh area (sum of all element areas)
    pub fn total_area(&self) -> f64 {
        self.elements.values().map(|e| self.element_area(e)).sum()
    }
}

impl Default for Mesh2D {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple right triangle with vertices at (0,0), (1,0), (0,1)
    fn unit_triangle() -> (Mesh2D, u32, u32, u32, u32) {
        let mut mesh = Mesh2D::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        let n2 = mesh.add_node(0.0, 1.0);
        let e0 = mesh.add_element(n0, n1, n2);
        (mesh, n0, n1, n2, e0)
    }

    /// Build an equilateral triangle with side length 1
    fn equilateral() -> (Mesh2D, u32) {
        let mut mesh = Mesh2D::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        let n2 = mesh.add_node(0.5, 3_f64.sqrt() / 2.0);
        let e0 = mesh.add_element(n0, n1, n2);
        (mesh, e0)
    }

    // --- Node management ---

    #[test]
    fn test_add_node_returns_sequential_ids() {
        let mut mesh = Mesh2D::new();
        assert_eq!(mesh.add_node(0.0, 0.0), 0);
        assert_eq!(mesh.add_node(1.0, 0.0), 1);
        assert_eq!(mesh.add_node(0.5, 1.0), 2);
    }

    #[test]
    fn test_node_count_matches_insertions() {
        let mut mesh = Mesh2D::new();
        for i in 0..10 {
            mesh.add_node(i as f64, 0.0);
        }
        assert_eq!(mesh.node_count(), 10);
    }

    #[test]
    fn test_node_coordinates_stored_correctly() {
        let mut mesh = Mesh2D::new();
        let id = mesh.add_node(1.23, 2.72);
        let node = mesh.nodes.get(&id).expect("node should exist");
        assert!((node.x - 1.23).abs() < 1e-10);
        assert!((node.y - 2.72).abs() < 1e-10);
    }

    // --- Element management ---

    #[test]
    fn test_add_element_returns_sequential_ids() {
        let (mut mesh, _n0, n1, n2, _e0) = unit_triangle();
        let n3 = mesh.add_node(1.0, 1.0);
        let e1 = mesh.add_element(n1, n3, n2);
        assert_eq!(e1, 1);
    }

    #[test]
    fn test_element_count_matches_insertions() {
        let mut mesh = Mesh2D::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        let n2 = mesh.add_node(0.5, 1.0);
        let n3 = mesh.add_node(1.5, 1.0);
        mesh.add_element(n0, n1, n2);
        mesh.add_element(n1, n3, n2);
        assert_eq!(mesh.element_count(), 2);
    }

    #[test]
    fn test_element_nodes_stored_correctly() {
        let (mesh, n0, n1, n2, e0) = unit_triangle();
        let elem = mesh.elements.get(&e0).expect("element should exist");
        assert_eq!(elem.nodes, [n0, n1, n2]);
    }

    #[test]
    fn test_element_initial_level_is_zero() {
        let (mesh, _, _, _, e0) = unit_triangle();
        let elem = mesh.elements.get(&e0).expect("element should exist");
        assert_eq!(elem.refinement_level, 0);
    }

    // --- Area calculations ---

    #[test]
    fn test_unit_triangle_area() {
        let (mesh, _, _, _, e0) = unit_triangle();
        let elem = mesh.elements.get(&e0).expect("element should exist");
        let area = mesh.element_area(elem);
        assert!((area - 0.5).abs() < 1e-10, "Expected 0.5, got {}", area);
    }

    #[test]
    fn test_equilateral_area() {
        let (mesh, e0) = equilateral();
        let elem = mesh.elements.get(&e0).expect("element should exist");
        let expected = 3_f64.sqrt() / 4.0;
        let area = mesh.element_area(elem);
        assert!(
            (area - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            area
        );
    }

    #[test]
    fn test_total_area_single_element() {
        let (mesh, _, _, _, _) = unit_triangle();
        assert!((mesh.total_area() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_total_area_two_elements() {
        let mut mesh = Mesh2D::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        let n2 = mesh.add_node(1.0, 1.0);
        let n3 = mesh.add_node(0.0, 1.0);
        mesh.add_element(n0, n1, n2);
        mesh.add_element(n0, n2, n3);
        assert!((mesh.total_area() - 1.0).abs() < 1e-10);
    }

    // --- Edge lengths ---

    #[test]
    fn test_unit_triangle_edge_lengths() {
        let (mesh, _, _, _, e0) = unit_triangle();
        let elem = mesh.elements.get(&e0).expect("element should exist");
        let edges = mesh.element_edge_lengths(elem);
        // Edges: (0,0)-(1,0)=1, (1,0)-(0,1)=sqrt(2), (0,1)-(0,0)=1
        assert!((edges[0] - 1.0).abs() < 1e-10);
        assert!((edges[1] - 2_f64.sqrt()).abs() < 1e-10);
        assert!((edges[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equilateral_edge_lengths_equal() {
        let (mesh, e0) = equilateral();
        let elem = mesh.elements.get(&e0).expect("element should exist");
        let edges = mesh.element_edge_lengths(elem);
        assert!((edges[0] - edges[1]).abs() < 1e-10);
        assert!((edges[1] - edges[2]).abs() < 1e-10);
    }

    // --- Aspect ratio ---

    #[test]
    fn test_equilateral_aspect_ratio_is_one() {
        let (mesh, e0) = equilateral();
        let elem = mesh.elements.get(&e0).expect("element should exist");
        let ar = mesh.element_aspect_ratio(elem);
        assert!((ar - 1.0).abs() < 1e-10, "Expected 1.0, got {}", ar);
    }

    #[test]
    fn test_right_triangle_aspect_ratio_gt_one() {
        let (mesh, _, _, _, e0) = unit_triangle();
        let elem = mesh.elements.get(&e0).expect("element should exist");
        let ar = mesh.element_aspect_ratio(elem);
        assert!(ar > 1.0, "Expected AR > 1, got {}", ar);
    }

    // --- Angle calculations ---

    #[test]
    fn test_equilateral_min_angle_60_degrees() {
        let (mesh, e0) = equilateral();
        let elem = mesh.elements.get(&e0).expect("element should exist");
        let min_angle = mesh.element_min_angle_deg(elem);
        assert!(
            (min_angle - 60.0).abs() < 1e-6,
            "Expected 60°, got {}",
            min_angle
        );
    }

    #[test]
    fn test_right_triangle_min_angle_45_degrees() {
        let (mesh, _, _, _, e0) = unit_triangle();
        let elem = mesh.elements.get(&e0).expect("element should exist");
        let min_angle = mesh.element_min_angle_deg(elem);
        assert!(
            (min_angle - 45.0).abs() < 1e-6,
            "Expected 45°, got {}",
            min_angle
        );
    }

    // --- Midpoint ---

    #[test]
    fn test_midpoint_basic() {
        let mut mesh = Mesh2D::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(2.0, 4.0);
        let (mx, my) = mesh.midpoint(n0, n1);
        assert!((mx - 1.0).abs() < 1e-10);
        assert!((my - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_midpoint_missing_node_returns_origin() {
        let mesh = Mesh2D::new();
        let (mx, my) = mesh.midpoint(99, 100);
        assert_eq!(mx, 0.0);
        assert_eq!(my, 0.0);
    }

    // --- Single element refinement ---

    #[test]
    fn test_refine_element_produces_four_children() {
        let (mut mesh, _, _, _, e0) = unit_triangle();
        let children = mesh.refine_element(e0);
        // All 4 child IDs should be non-zero or valid IDs
        assert_ne!(children, [0; 4]);
        // Parent element should be removed
        assert!(!mesh.elements.contains_key(&e0));
        // Mesh should now have 4 elements
        assert_eq!(mesh.element_count(), 4);
    }

    #[test]
    fn test_refine_element_adds_three_midpoint_nodes() {
        let (mut mesh, _, _, _, e0) = unit_triangle();
        let initial_nodes = mesh.node_count();
        mesh.refine_element(e0);
        assert_eq!(mesh.node_count(), initial_nodes + 3);
    }

    #[test]
    fn test_refine_element_child_levels_incremented() {
        let (mut mesh, _, _, _, e0) = unit_triangle();
        let children = mesh.refine_element(e0);
        for &child_id in &children {
            if child_id != 0 {
                let child = mesh.elements.get(&child_id).expect("child should exist");
                assert_eq!(child.refinement_level, 1);
            }
        }
    }

    #[test]
    fn test_refine_element_preserves_total_area() {
        let (mut mesh, _, _, _, e0) = unit_triangle();
        let area_before = mesh.total_area();
        mesh.refine_element(e0);
        let area_after = mesh.total_area();
        assert!(
            (area_before - area_after).abs() < 1e-10,
            "Area should be preserved: {} vs {}",
            area_before,
            area_after
        );
    }

    #[test]
    fn test_refine_missing_element_returns_empty() {
        let mut mesh = Mesh2D::new();
        let result = mesh.refine_element(999);
        assert_eq!(result, [0; 4]);
    }

    #[test]
    fn test_refine_at_max_level_returns_empty() {
        let mut mesh = Mesh2D::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        let n2 = mesh.add_node(0.5, 1.0);
        let e_id = mesh.next_elem_id;
        mesh.elements.insert(
            e_id,
            MeshElement {
                id: e_id,
                nodes: [n0, n1, n2],
                refinement_level: MAX_REFINEMENT_LEVEL,
            },
        );
        mesh.next_elem_id += 1;
        let result = mesh.refine_element(e_id);
        assert_eq!(result, [0; 4]);
    }

    // --- should_refine ---

    #[test]
    fn test_should_refine_by_edge_length() {
        let (mesh, _, _, _, e0) = unit_triangle();
        let elem = mesh.elements.get(&e0).expect("element should exist");
        let criteria = RefinementCriteria {
            max_edge_length: 0.5, // all edges > 0.5
            max_aspect_ratio: 100.0,
            min_angle_deg: 0.0,
            error_threshold: f64::INFINITY,
        };
        let empty_map = HashMap::new();
        assert!(mesh.should_refine(elem, &criteria, &empty_map));
    }

    #[test]
    fn test_should_not_refine_below_thresholds() {
        let (mesh, _, _, _, e0) = unit_triangle();
        let elem = mesh.elements.get(&e0).expect("element should exist");
        let criteria = RefinementCriteria {
            max_edge_length: 10.0,
            max_aspect_ratio: 100.0,
            min_angle_deg: 0.0,
            error_threshold: f64::INFINITY,
        };
        let empty_map = HashMap::new();
        assert!(!mesh.should_refine(elem, &criteria, &empty_map));
    }

    #[test]
    fn test_should_refine_by_error_map() {
        let (mesh, _, _, _, e0) = unit_triangle();
        let elem = mesh.elements.get(&e0).expect("element should exist");
        let criteria = RefinementCriteria {
            max_edge_length: 100.0,
            max_aspect_ratio: 100.0,
            min_angle_deg: 0.0,
            error_threshold: 0.01,
        };
        let mut error_map = HashMap::new();
        error_map.insert(e0, 0.5); // exceeds threshold
        assert!(mesh.should_refine(elem, &criteria, &error_map));
    }

    #[test]
    fn test_should_refine_by_aspect_ratio() {
        // n0=(0,0), n1=(100,0), n2=(0.1, 0.0001)
        // e01=100, e12≈99.9, e20≈0.1  → AR≈100/0.1=1000 >> 2.0
        let mut mesh = Mesh2D::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(100.0, 0.0);
        let n2 = mesh.add_node(0.1, 0.0001);
        let e0 = mesh.add_element(n0, n1, n2);
        let elem = mesh.elements.get(&e0).expect("element should exist");
        let criteria = RefinementCriteria {
            max_edge_length: 200.0,
            max_aspect_ratio: 2.0,
            min_angle_deg: 0.0,
            error_threshold: f64::INFINITY,
        };
        let empty_map = HashMap::new();
        assert!(
            mesh.should_refine(elem, &criteria, &empty_map),
            "AR of very thin triangle should trigger refinement"
        );
    }

    #[test]
    fn test_should_not_refine_at_max_level() {
        let mut mesh = Mesh2D::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(0.001, 0.0);
        let n2 = mesh.add_node(0.0005, 0.001);
        let e_id = mesh.next_elem_id;
        mesh.elements.insert(
            e_id,
            MeshElement {
                id: e_id,
                nodes: [n0, n1, n2],
                refinement_level: MAX_REFINEMENT_LEVEL,
            },
        );
        mesh.next_elem_id += 1;
        let elem = mesh.elements.get(&e_id).expect("element should exist");
        let criteria = RefinementCriteria {
            max_edge_length: 0.00001,
            max_aspect_ratio: 1.0,
            min_angle_deg: 89.0,
            error_threshold: 0.0,
        };
        let mut error_map = HashMap::new();
        error_map.insert(e_id, 999.0);
        assert!(!mesh.should_refine(elem, &criteria, &empty_map_helper()));
    }

    // helper to avoid borrowing issue
    fn empty_map_helper() -> HashMap<u32, f64> {
        HashMap::new()
    }

    // --- Adaptive global refinement ---

    #[test]
    fn test_refine_mesh_refines_eligible_elements() {
        let (mut mesh, _, _, _, _) = unit_triangle();
        let criteria = RefinementCriteria {
            max_edge_length: 0.5,
            max_aspect_ratio: 100.0,
            min_angle_deg: 0.0,
            error_threshold: f64::INFINITY,
        };
        let empty_map = HashMap::new();
        let stats = mesh.refine_mesh(&criteria, &empty_map);
        assert_eq!(stats.elements_refined, 1);
        assert_eq!(stats.nodes_added, 3);
        assert_eq!(stats.max_level, 1);
        assert_eq!(mesh.element_count(), 4);
    }

    #[test]
    fn test_refine_mesh_stats_max_level() {
        let (mut mesh, _, _, _, _) = unit_triangle();
        let criteria = RefinementCriteria {
            max_edge_length: 0.01,
            max_aspect_ratio: 100.0,
            min_angle_deg: 0.0,
            error_threshold: f64::INFINITY,
        };
        let empty_map = HashMap::new();
        // Two passes
        let s1 = mesh.refine_mesh(&criteria, &empty_map);
        let s2 = mesh.refine_mesh(&criteria, &empty_map);
        assert!(s1.max_level <= s2.max_level);
    }

    #[test]
    fn test_refine_mesh_preserves_total_area() {
        let (mut mesh, _, _, _, _) = unit_triangle();
        let area_before = mesh.total_area();
        let criteria = RefinementCriteria {
            max_edge_length: 0.5,
            max_aspect_ratio: 100.0,
            min_angle_deg: 0.0,
            error_threshold: f64::INFINITY,
        };
        let empty_map = HashMap::new();
        mesh.refine_mesh(&criteria, &empty_map);
        let area_after = mesh.total_area();
        assert!(
            (area_before - area_after).abs() < 1e-9,
            "Area changed: {} -> {}",
            area_before,
            area_after
        );
    }

    #[test]
    fn test_refine_mesh_error_driven() {
        let mut mesh = Mesh2D::new();
        // 4-element unit square mesh
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        let n2 = mesh.add_node(1.0, 1.0);
        let n3 = mesh.add_node(0.0, 1.0);
        let e0 = mesh.add_element(n0, n1, n2);
        let e1 = mesh.add_element(n0, n2, n3);

        let criteria = RefinementCriteria {
            max_edge_length: 100.0,
            max_aspect_ratio: 100.0,
            min_angle_deg: 0.0,
            error_threshold: 0.1,
        };
        let mut error_map = HashMap::new();
        error_map.insert(e0, 0.5); // only e0 has high error
        error_map.insert(e1, 0.01);

        let stats = mesh.refine_mesh(&criteria, &error_map);
        assert_eq!(stats.elements_refined, 1);
        assert_eq!(mesh.element_count(), 5); // 1 original + 4 children
    }

    #[test]
    fn test_refine_mesh_no_eligible_elements() {
        let (mut mesh, _, _, _, _) = unit_triangle();
        let criteria = RefinementCriteria {
            max_edge_length: 100.0,
            max_aspect_ratio: 100.0,
            min_angle_deg: 0.0,
            error_threshold: f64::INFINITY,
        };
        let empty_map = HashMap::new();
        let stats = mesh.refine_mesh(&criteria, &empty_map);
        assert_eq!(stats.elements_refined, 0);
        assert_eq!(stats.nodes_added, 0);
        assert_eq!(mesh.element_count(), 1);
    }

    #[test]
    fn test_double_refinement_grows_correctly() {
        let (mut mesh, _, _, _, _) = unit_triangle();
        let criteria = RefinementCriteria {
            max_edge_length: 0.5,
            max_aspect_ratio: 100.0,
            min_angle_deg: 0.0,
            error_threshold: f64::INFINITY,
        };
        let empty_map = HashMap::new();
        mesh.refine_mesh(&criteria, &empty_map);
        let area_after_1 = mesh.total_area();
        mesh.refine_mesh(&criteria, &empty_map);
        let area_after_2 = mesh.total_area();
        assert!((area_after_1 - area_after_2).abs() < 1e-9);
        assert!(mesh.element_count() > 4);
    }

    #[test]
    fn test_refinement_criteria_default() {
        let c = RefinementCriteria::default_criteria();
        assert!(c.max_edge_length > 0.0);
        assert!(c.max_aspect_ratio > 1.0);
        assert!(c.min_angle_deg > 0.0);
        assert!(c.error_threshold > 0.0);
    }

    #[test]
    fn test_mesh_default_is_empty() {
        let mesh = Mesh2D::default();
        assert_eq!(mesh.node_count(), 0);
        assert_eq!(mesh.element_count(), 0);
        assert_eq!(mesh.total_area(), 0.0);
    }

    #[test]
    fn test_multiple_sequential_refinements_bounded_by_max_level() {
        let (mut mesh, _, _, _, _) = unit_triangle();
        let criteria = RefinementCriteria {
            max_edge_length: 0.0001, // extremely small — will always trigger until max level
            max_aspect_ratio: 1.0001,
            min_angle_deg: 89.0,
            error_threshold: 0.0,
        };
        let empty_map = HashMap::new();
        for _ in 0..20 {
            mesh.refine_mesh(&criteria, &empty_map);
        }
        let max_level = mesh
            .elements
            .values()
            .map(|e| e.refinement_level)
            .max()
            .unwrap_or(0);
        assert!(max_level <= MAX_REFINEMENT_LEVEL);
    }
}
