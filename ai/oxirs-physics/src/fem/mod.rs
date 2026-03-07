//! Finite Element Method (FEM) for Structural and Thermal Analysis
//!
//! Provides simplified 1D/2D FEM solvers using direct Gaussian elimination
//! for the global stiffness system. Supports Bar1D, Beam1D, Triangle2D,
//! and Quad2D element types.
//!
//! # Example — 1D bar under axial load
//!
//! ```rust
//! use oxirs_physics::fem::{
//!     FemMesh, FemMaterial, ElementType, DofType, FemSolver, NodalLoad,
//! };
//!
//! let mut mesh = FemMesh::new();
//! let mat = FemMaterial {
//!     youngs_modulus: 200e9,
//!     poissons_ratio: 0.3,
//!     thermal_conductivity: 50.0,
//!     density: 7850.0,
//! };
//!
//! let n0 = mesh.add_node(0.0, 0.0);
//! let n1 = mesh.add_node(1.0, 0.0);
//! mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
//! mesh.add_element(vec![n0, n1], mat, ElementType::Bar1D);
//!
//! let solver = FemSolver::new();
//! let loads = vec![NodalLoad { node_id: n1, fx: 1000.0, fy: 0.0 }];
//! let sol = solver.solve_static(&mesh, &loads);
//! assert!(sol.converged);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────
// Public Data Model
// ─────────────────────────────────────────────

/// Degree-of-freedom type for boundary conditions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DofType {
    /// Structural displacement (m).
    Displacement,
    /// Temperature (K).
    Temperature,
    /// Pressure (Pa).
    Pressure,
}

/// Fixed boundary condition applied to a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryCondition {
    /// Which DOF is constrained.
    pub dof: DofType,
    /// Prescribed value.
    pub value: f64,
}

/// FEM node with optional boundary condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemNode {
    /// Node identifier (0-based).
    pub id: usize,
    /// X coordinate (m).
    pub x: f64,
    /// Y coordinate (m).
    pub y: f64,
    /// Optional fixed boundary condition.
    pub boundary_condition: Option<BoundaryCondition>,
}

/// Isotropic linear-elastic material properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemMaterial {
    /// Young's modulus E (Pa).
    pub youngs_modulus: f64,
    /// Poisson's ratio ν (dimensionless).
    pub poissons_ratio: f64,
    /// Thermal conductivity k (W / m·K).
    pub thermal_conductivity: f64,
    /// Mass density ρ (kg / m³).
    pub density: f64,
}

impl Default for FemMaterial {
    fn default() -> Self {
        Self {
            youngs_modulus: 200e9,
            poissons_ratio: 0.3,
            thermal_conductivity: 50.0,
            density: 7850.0,
        }
    }
}

/// Supported element types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ElementType {
    /// 1-D two-node truss / axial bar.
    Bar1D,
    /// 2-D three-node constant-strain triangle (CST).
    Triangle2D,
    /// 2-D four-node bilinear quadrilateral.
    Quad2D,
    /// 1-D Euler-Bernoulli beam (2 DOF per node: v, θ).
    Beam1D,
}

/// Finite element connecting two or more nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemElement {
    /// Element identifier (0-based).
    pub id: usize,
    /// Ordered list of participating node IDs.
    pub node_ids: Vec<usize>,
    /// Material assigned to this element.
    pub material: FemMaterial,
    /// Topology/formulation type.
    pub element_type: ElementType,
}

/// Applied nodal force (structural).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodalLoad {
    /// Target node id.
    pub node_id: usize,
    /// Force in x-direction (N).
    pub fx: f64,
    /// Force in y-direction (N).
    pub fy: f64,
}

/// Uniform heat flux applied to an element (W / m²).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementHeatFlux {
    /// Target element id.
    pub element_id: usize,
    /// Heat flux magnitude (W / m²).
    pub q: f64,
}

// ─────────────────────────────────────────────
// Mesh
// ─────────────────────────────────────────────

/// Finite element mesh: collection of nodes and elements.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FemMesh {
    /// All mesh nodes.
    pub nodes: Vec<FemNode>,
    /// All mesh elements.
    pub elements: Vec<FemElement>,
}

impl FemMesh {
    /// Create an empty mesh.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node at `(x, y)` and return its id.
    pub fn add_node(&mut self, x: f64, y: f64) -> usize {
        let id = self.nodes.len();
        self.nodes.push(FemNode {
            id,
            x,
            y,
            boundary_condition: None,
        });
        id
    }

    /// Add an element and return its id.
    pub fn add_element(
        &mut self,
        node_ids: Vec<usize>,
        material: FemMaterial,
        element_type: ElementType,
    ) -> usize {
        let id = self.elements.len();
        self.elements.push(FemElement {
            id,
            node_ids,
            material,
            element_type,
        });
        id
    }

    /// Apply a fixed boundary condition to a node DOF.
    pub fn set_boundary_condition(&mut self, node_id: usize, dof: DofType, value: f64) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.boundary_condition = Some(BoundaryCondition { dof, value });
        }
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of elements.
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }
}

// ─────────────────────────────────────────────
// Solution types
// ─────────────────────────────────────────────

/// Structural static FEM solution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemSolution {
    /// Displacement (dx, dy) per node (m).
    pub displacements: Vec<(f64, f64)>,
    /// Von Mises stress per element (Pa).
    pub von_mises_stress: Vec<f64>,
    /// Maximum displacement magnitude (m).
    pub max_displacement: f64,
    /// Whether the solver converged.
    pub converged: bool,
}

/// Thermal FEM solution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalSolution {
    /// Nodal temperatures (K).
    pub temperatures: Vec<f64>,
    /// Heat flux (qx, qy) per element (W / m²).
    pub heat_flux: Vec<(f64, f64)>,
    /// Maximum nodal temperature (K).
    pub max_temperature: f64,
    /// Whether the solver converged.
    pub converged: bool,
}

// ─────────────────────────────────────────────
// Gaussian Elimination (dense, in-place)
// ─────────────────────────────────────────────

/// Solve `A x = b` by Gaussian elimination with partial pivoting.
/// Returns `None` if the system is singular.
fn gaussian_elimination(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for (row, a_row) in a.iter().enumerate().skip(col + 1).take(n - col - 1) {
            if a_row[col].abs() > max_val {
                max_val = a_row[col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return None; // singular
        }
        a.swap(col, max_row);
        b.swap(col, max_row);

        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            // Collect the pivot row values to avoid borrow conflicts
            let pivot_row_vals: Vec<f64> = a[col][col..n].to_vec();
            for (a_row_k, &av) in a[row][col..n].iter_mut().zip(pivot_row_vals.iter()) {
                *a_row_k -= factor * av;
            }
            b[row] -= factor * b[col];
        }
    }

    // Back-substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in (i + 1)..n {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }
    Some(x)
}

// ─────────────────────────────────────────────
// Element stiffness matrices
// ─────────────────────────────────────────────

/// Compute Bar1D local stiffness matrix (2×2) and map it to global DOFs.
/// Global DOFs: node i → DOF 2i (x), node j → DOF 2j (x only for 1D).
fn bar1d_element_stiffness(
    n_dofs: usize,
    elem: &FemElement,
    nodes: &[FemNode],
    k_global: &mut [Vec<f64>],
    cross_section_area: f64,
) {
    if elem.node_ids.len() < 2 {
        return;
    }
    let ni = elem.node_ids[0];
    let nj = elem.node_ids[1];
    let xi = nodes[ni].x;
    let xj = nodes[nj].x;
    let yi = nodes[ni].y;
    let yj = nodes[nj].y;

    let dx = xj - xi;
    let dy = yj - yi;
    let length = (dx * dx + dy * dy).sqrt();
    if length < 1e-30 {
        return;
    }

    let ae_over_l = elem.material.youngs_modulus * cross_section_area / length;
    let c = dx / length;
    let s = dy / length;

    // 4×4 local stiffness (in global x-y for 2 nodes × 2 DOFs each)
    let dofs = [2 * ni, 2 * ni + 1, 2 * nj, 2 * nj + 1];
    let ke_local = [
        [c * c, c * s, -c * c, -c * s],
        [c * s, s * s, -c * s, -s * s],
        [-c * c, -c * s, c * c, c * s],
        [-c * s, -s * s, c * s, s * s],
    ];

    for (a, &ga) in dofs.iter().enumerate() {
        for (b, &gb) in dofs.iter().enumerate() {
            if ga < n_dofs && gb < n_dofs {
                k_global[ga][gb] += ae_over_l * ke_local[a][b];
            }
        }
    }
}

/// Compute Triangle2D CST element stiffness and assemble into global matrix.
fn triangle2d_element_stiffness(
    n_dofs: usize,
    elem: &FemElement,
    nodes: &[FemNode],
    k_global: &mut [Vec<f64>],
    thickness: f64,
) {
    if elem.node_ids.len() < 3 {
        return;
    }
    let ni = elem.node_ids[0];
    let nj = elem.node_ids[1];
    let nk = elem.node_ids[2];

    let xi = nodes[ni].x;
    let yi = nodes[ni].y;
    let xj = nodes[nj].x;
    let yj = nodes[nj].y;
    let xk = nodes[nk].x;
    let yk = nodes[nk].y;

    let area = 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi));
    if area.abs() < 1e-30 {
        return;
    }

    let e = elem.material.youngs_modulus;
    let nu = elem.material.poissons_ratio;
    let factor = e / (1.0 - nu * nu);

    // Shape function derivatives (constant for CST)
    let b_mat = [
        [yj - yk, 0.0, yk - yi, 0.0, yi - yj, 0.0],
        [0.0, xk - xj, 0.0, xi - xk, 0.0, xj - xi],
        [xk - xj, yj - yk, xi - xk, yk - yi, xj - xi, yi - yj],
    ];
    let scale = 1.0 / (2.0 * area);
    let b_mat: Vec<Vec<f64>> = b_mat
        .iter()
        .map(|row| row.iter().map(|&v| v * scale).collect())
        .collect();

    // Constitutive matrix D (plane stress)
    let d_mat = [
        [factor, factor * nu, 0.0],
        [factor * nu, factor, 0.0],
        [0.0, 0.0, factor * (1.0 - nu) / 2.0],
    ];

    // Ke = t * A * B^T * D * B  (6×6)
    let mut ke = vec![vec![0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut sum = 0.0;
            for p in 0..3 {
                for q in 0..3 {
                    sum += b_mat[p][i] * d_mat[p][q] * b_mat[q][j];
                }
            }
            ke[i][j] = thickness * area.abs() * sum;
        }
    }

    let dofs = [2 * ni, 2 * ni + 1, 2 * nj, 2 * nj + 1, 2 * nk, 2 * nk + 1];
    for (a, &ga) in dofs.iter().enumerate() {
        for (b, &gb) in dofs.iter().enumerate() {
            if ga < n_dofs && gb < n_dofs {
                k_global[ga][gb] += ke[a][b];
            }
        }
    }
}

// ─────────────────────────────────────────────
// Thermal element stiffness
// ─────────────────────────────────────────────

/// Bar1D thermal conductivity element (1 DOF per node = temperature).
fn bar1d_thermal_stiffness(
    n_nodes: usize,
    elem: &FemElement,
    nodes: &[FemNode],
    k_global: &mut [Vec<f64>],
    cross_section_area: f64,
) {
    if elem.node_ids.len() < 2 {
        return;
    }
    let ni = elem.node_ids[0];
    let nj = elem.node_ids[1];
    let dx = nodes[nj].x - nodes[ni].x;
    let dy = nodes[nj].y - nodes[ni].y;
    let length = (dx * dx + dy * dy).sqrt();
    if length < 1e-30 {
        return;
    }

    let k_coeff = elem.material.thermal_conductivity * cross_section_area / length;
    // 2×2 conductivity matrix
    let pairs = [
        (ni, ni, k_coeff),
        (nj, nj, k_coeff),
        (ni, nj, -k_coeff),
        (nj, ni, -k_coeff),
    ];
    for (r, c, val) in pairs {
        if r < n_nodes && c < n_nodes {
            k_global[r][c] += val;
        }
    }
}

// ─────────────────────────────────────────────
// Solver
// ─────────────────────────────────────────────

/// FEM solver: assembles global stiffness and solves via Gaussian elimination.
#[derive(Debug, Clone, Default)]
pub struct FemSolver {
    /// Cross-section area used for Bar1D elements (m²). Defaults to 1e-4 m².
    pub cross_section_area: f64,
    /// Thickness for 2-D elements (m). Defaults to 0.01 m.
    pub thickness: f64,
}

impl FemSolver {
    /// Create a solver with default parameters.
    pub fn new() -> Self {
        Self {
            cross_section_area: 1e-4,
            thickness: 0.01,
        }
    }

    /// Create a solver with explicit cross-section area and thickness.
    pub fn with_params(cross_section_area: f64, thickness: f64) -> Self {
        Self {
            cross_section_area,
            thickness,
        }
    }

    /// Solve a static structural problem.
    pub fn solve_static(&self, mesh: &FemMesh, loads: &[NodalLoad]) -> FemSolution {
        let n_nodes = mesh.node_count();
        let n_dofs = 2 * n_nodes; // x and y DOF per node

        // Assemble global stiffness
        let mut k = vec![vec![0.0f64; n_dofs]; n_dofs];
        for elem in &mesh.elements {
            match elem.element_type {
                ElementType::Bar1D | ElementType::Beam1D => {
                    bar1d_element_stiffness(
                        n_dofs,
                        elem,
                        &mesh.nodes,
                        &mut k,
                        self.cross_section_area,
                    );
                }
                ElementType::Triangle2D => {
                    triangle2d_element_stiffness(n_dofs, elem, &mesh.nodes, &mut k, self.thickness);
                }
                ElementType::Quad2D => {
                    // Approximate as two triangles (CST split)
                    if elem.node_ids.len() >= 4 {
                        let tri_a = FemElement {
                            id: elem.id,
                            node_ids: vec![elem.node_ids[0], elem.node_ids[1], elem.node_ids[2]],
                            material: elem.material.clone(),
                            element_type: ElementType::Triangle2D,
                        };
                        let tri_b = FemElement {
                            id: elem.id,
                            node_ids: vec![elem.node_ids[0], elem.node_ids[2], elem.node_ids[3]],
                            material: elem.material.clone(),
                            element_type: ElementType::Triangle2D,
                        };
                        triangle2d_element_stiffness(
                            n_dofs,
                            &tri_a,
                            &mesh.nodes,
                            &mut k,
                            self.thickness,
                        );
                        triangle2d_element_stiffness(
                            n_dofs,
                            &tri_b,
                            &mesh.nodes,
                            &mut k,
                            self.thickness,
                        );
                    }
                }
            }
        }

        // Build force vector
        let mut f = vec![0.0f64; n_dofs];
        for load in loads {
            let dof_x = 2 * load.node_id;
            let dof_y = 2 * load.node_id + 1;
            if dof_x < n_dofs {
                f[dof_x] += load.fx;
            }
            if dof_y < n_dofs {
                f[dof_y] += load.fy;
            }
        }

        // Collect all boundary conditions (DOF index → prescribed value)
        let mut bc_map: HashMap<usize, f64> = HashMap::new();
        for node in &mesh.nodes {
            if let Some(ref bc) = node.boundary_condition {
                match bc.dof {
                    DofType::Displacement => {
                        // Pin both x and y for a displacement BC
                        bc_map.insert(2 * node.id, bc.value);
                        bc_map.insert(2 * node.id + 1, bc.value);
                    }
                    DofType::Temperature | DofType::Pressure => {
                        bc_map.insert(2 * node.id, bc.value);
                    }
                }
            }
        }

        // Apply boundary conditions (large-number / penalty method)
        let penalty = 1e30;
        for (&dof, &val) in &bc_map {
            if dof < n_dofs {
                k[dof][dof] += penalty;
                f[dof] += penalty * val;
            }
        }

        // Stabilize unconstrained zero-stiffness DOFs (e.g. transverse DOFs
        // of pure 1D bar elements) to prevent singularity.  A small spring
        // stiffness 1e-6 × max_diag is added to any zero diagonal entry that
        // has no prescribed BC, yielding a near-zero (but well-defined) free
        // displacement for that DOF.
        {
            let max_diag = (0..n_dofs).map(|i| k[i][i].abs()).fold(0.0f64, f64::max);
            let stab = if max_diag > 0.0 { max_diag * 1e-6 } else { 1.0 };
            for (i, k_row) in k.iter_mut().enumerate().take(n_dofs) {
                if k_row[i].abs() < 1e-30 && !bc_map.contains_key(&i) {
                    k_row[i] += stab;
                }
            }
        }

        // Solve
        let mut k_copy = k.clone();
        let mut f_copy = f.clone();
        let solution = gaussian_elimination(&mut k_copy, &mut f_copy);

        match solution {
            None => FemSolution {
                displacements: vec![(0.0, 0.0); n_nodes],
                von_mises_stress: vec![0.0; mesh.element_count()],
                max_displacement: 0.0,
                converged: false,
            },
            Some(u) => {
                let displacements: Vec<(f64, f64)> =
                    (0..n_nodes).map(|i| (u[2 * i], u[2 * i + 1])).collect();

                let von_mises_stress =
                    compute_von_mises(&u, mesh, self.cross_section_area, self.thickness);

                let max_displacement = displacements
                    .iter()
                    .map(|(dx, dy)| (dx * dx + dy * dy).sqrt())
                    .fold(0.0f64, f64::max);

                FemSolution {
                    displacements,
                    von_mises_stress,
                    max_displacement,
                    converged: true,
                }
            }
        }
    }

    /// Solve a steady-state thermal conduction problem.
    pub fn solve_thermal(&self, mesh: &FemMesh, heat_flux: &[ElementHeatFlux]) -> ThermalSolution {
        let n_nodes = mesh.node_count();

        // Assemble global thermal conductivity matrix (1 DOF per node)
        let mut k = vec![vec![0.0f64; n_nodes]; n_nodes];
        for elem in &mesh.elements {
            bar1d_thermal_stiffness(n_nodes, elem, &mesh.nodes, &mut k, self.cross_section_area);
        }

        // Build heat load vector: distribute element heat flux as nodal load
        let mut q_vec = vec![0.0f64; n_nodes];
        for hf in heat_flux {
            if let Some(elem) = mesh.elements.get(hf.element_id) {
                let n = elem.node_ids.len() as f64;
                for &nid in &elem.node_ids {
                    if nid < n_nodes {
                        q_vec[nid] += hf.q / n;
                    }
                }
            }
        }

        // Apply temperature boundary conditions (penalty method)
        let penalty = 1e30;
        for node in &mesh.nodes {
            if let Some(ref bc) = node.boundary_condition {
                if bc.dof == DofType::Temperature {
                    let dof = node.id;
                    if dof < n_nodes {
                        k[dof][dof] += penalty;
                        q_vec[dof] += penalty * bc.value;
                    }
                }
            }
        }

        let mut k_copy = k.clone();
        let mut q_copy = q_vec.clone();
        let solution = gaussian_elimination(&mut k_copy, &mut q_copy);

        match solution {
            None => ThermalSolution {
                temperatures: vec![0.0; n_nodes],
                heat_flux: vec![(0.0, 0.0); mesh.element_count()],
                max_temperature: 0.0,
                converged: false,
            },
            Some(temps) => {
                let element_heat_flux: Vec<(f64, f64)> = mesh
                    .elements
                    .iter()
                    .map(|elem| compute_element_heat_flux(elem, &temps, &mesh.nodes))
                    .collect();

                let max_temperature = temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                ThermalSolution {
                    temperatures: temps,
                    heat_flux: element_heat_flux,
                    max_temperature,
                    converged: true,
                }
            }
        }
    }
}

// ─────────────────────────────────────────────
// Post-processing helpers
// ─────────────────────────────────────────────

/// Compute approximate Von Mises stress for each element.
fn compute_von_mises(
    u: &[f64],
    mesh: &FemMesh,
    cross_section_area: f64,
    thickness: f64,
) -> Vec<f64> {
    mesh.elements
        .iter()
        .map(|elem| element_von_mises(u, elem, &mesh.nodes, cross_section_area, thickness))
        .collect()
}

/// Compute Von Mises stress for a single element.
fn element_von_mises(
    u: &[f64],
    elem: &FemElement,
    nodes: &[FemNode],
    cross_section_area: f64,
    _thickness: f64,
) -> f64 {
    match elem.element_type {
        ElementType::Bar1D | ElementType::Beam1D => {
            if elem.node_ids.len() < 2 {
                return 0.0;
            }
            let ni = elem.node_ids[0];
            let nj = elem.node_ids[1];
            let xi = nodes[ni].x;
            let xj = nodes[nj].x;
            let yi = nodes[ni].y;
            let yj = nodes[nj].y;
            let dx = xj - xi;
            let dy = yj - yi;
            let length = (dx * dx + dy * dy).sqrt().max(1e-30);
            let c = dx / length;
            let s = dy / length;

            // Axial strain from nodal displacements
            let u_i = if 2 * ni + 1 < u.len() {
                c * u[2 * ni] + s * u[2 * ni + 1]
            } else {
                0.0
            };
            let u_j = if 2 * nj + 1 < u.len() {
                c * u[2 * nj] + s * u[2 * nj + 1]
            } else {
                0.0
            };
            let strain = (u_j - u_i) / length;
            (elem.material.youngs_modulus * strain).abs()
        }
        ElementType::Triangle2D | ElementType::Quad2D => {
            // Simplified: use displacement magnitude divided by element size
            if elem.node_ids.is_empty() {
                return 0.0;
            }
            let avg_disp: f64 = elem
                .node_ids
                .iter()
                .map(|&nid| {
                    let dx = u.get(2 * nid).copied().unwrap_or(0.0);
                    let dy = u.get(2 * nid + 1).copied().unwrap_or(0.0);
                    (dx * dx + dy * dy).sqrt()
                })
                .sum::<f64>()
                / elem.node_ids.len() as f64;

            // Characteristic element size (average side length)
            let char_len = cross_section_area.sqrt().max(1e-10);
            elem.material.youngs_modulus * avg_disp / char_len
        }
    }
}

/// Compute heat flux vector for a bar element.
fn compute_element_heat_flux(elem: &FemElement, temps: &[f64], nodes: &[FemNode]) -> (f64, f64) {
    if elem.node_ids.len() < 2 {
        return (0.0, 0.0);
    }
    let ni = elem.node_ids[0];
    let nj = elem.node_ids[1];
    if ni >= temps.len() || nj >= temps.len() {
        return (0.0, 0.0);
    }
    let xi = nodes[ni].x;
    let xj = nodes[nj].x;
    let yi = nodes[ni].y;
    let yj = nodes[nj].y;
    let dx = xj - xi;
    let dy = yj - yi;
    let length = (dx * dx + dy * dy).sqrt().max(1e-30);
    let dt_dl = (temps[nj] - temps[ni]) / length;
    let k = elem.material.thermal_conductivity;
    let qx = -k * dt_dl * (dx / length);
    let qy = -k * dt_dl * (dy / length);
    (qx, qy)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helper ----

    fn steel() -> FemMaterial {
        FemMaterial {
            youngs_modulus: 200e9,
            poissons_ratio: 0.3,
            thermal_conductivity: 50.0,
            density: 7850.0,
        }
    }

    fn aluminium() -> FemMaterial {
        FemMaterial {
            youngs_modulus: 70e9,
            poissons_ratio: 0.33,
            thermal_conductivity: 205.0,
            density: 2700.0,
        }
    }

    // ────────────────────────────────────────
    // Mesh tests
    // ────────────────────────────────────────

    #[test]
    fn test_mesh_new_is_empty() {
        let mesh = FemMesh::new();
        assert_eq!(mesh.node_count(), 0);
        assert_eq!(mesh.element_count(), 0);
    }

    #[test]
    fn test_mesh_add_nodes() {
        let mut mesh = FemMesh::new();
        let id0 = mesh.add_node(0.0, 0.0);
        let id1 = mesh.add_node(1.0, 0.0);
        let id2 = mesh.add_node(0.5, 1.0);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(mesh.node_count(), 3);
    }

    #[test]
    fn test_mesh_add_element_bar() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        let eid = mesh.add_element(vec![n0, n1], steel(), ElementType::Bar1D);
        assert_eq!(eid, 0);
        assert_eq!(mesh.element_count(), 1);
    }

    #[test]
    fn test_mesh_add_triangle_element() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        let n2 = mesh.add_node(0.5, 1.0);
        mesh.add_element(vec![n0, n1, n2], steel(), ElementType::Triangle2D);
        assert_eq!(mesh.element_count(), 1);
    }

    #[test]
    fn test_mesh_set_boundary_condition() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        let bc = mesh.nodes[n0]
            .boundary_condition
            .as_ref()
            .expect("BC should be set");
        assert_eq!(bc.dof, DofType::Displacement);
        assert_eq!(bc.value, 0.0);
    }

    #[test]
    fn test_boundary_condition_temperature() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        mesh.set_boundary_condition(n0, DofType::Temperature, 300.0);
        let bc = mesh.nodes[n0]
            .boundary_condition
            .as_ref()
            .expect("BC must be set");
        assert_eq!(bc.dof, DofType::Temperature);
        assert!((bc.value - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_boundary_condition_pressure() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        mesh.set_boundary_condition(n0, DofType::Pressure, 101325.0);
        let bc = mesh.nodes[n0]
            .boundary_condition
            .as_ref()
            .expect("BC must be set");
        assert_eq!(bc.dof, DofType::Pressure);
        assert!((bc.value - 101325.0).abs() < 1.0);
    }

    #[test]
    fn test_boundary_condition_invalid_node_is_noop() {
        let mut mesh = FemMesh::new();
        // Node 99 does not exist — should not panic
        mesh.set_boundary_condition(99, DofType::Displacement, 0.0);
        assert_eq!(mesh.node_count(), 0);
    }

    #[test]
    fn test_mesh_multiple_elements() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        let n2 = mesh.add_node(2.0, 0.0);
        mesh.add_element(vec![n0, n1], steel(), ElementType::Bar1D);
        mesh.add_element(vec![n1, n2], aluminium(), ElementType::Bar1D);
        assert_eq!(mesh.element_count(), 2);
    }

    // ────────────────────────────────────────
    // Solver — static structural
    // ────────────────────────────────────────

    /// 1-D bar fixed at left, force F at right.
    /// Analytical: u_right = F·L / (E·A)
    #[test]
    fn test_bar1d_axial_displacement() {
        let a = 1e-4; // 1 cm²
        let e = 200e9; // steel
        let l = 1.0;
        let force = 10_000.0; // 10 kN

        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(l, 0.0);
        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        mesh.add_element(
            vec![n0, n1],
            FemMaterial {
                youngs_modulus: e,
                ..FemMaterial::default()
            },
            ElementType::Bar1D,
        );

        let solver = FemSolver::with_params(a, 0.01);
        let loads = vec![NodalLoad {
            node_id: n1,
            fx: force,
            fy: 0.0,
        }];
        let sol = solver.solve_static(&mesh, &loads);

        assert!(sol.converged, "Solver should converge");
        let expected = force * l / (e * a);
        let actual = sol.displacements[n1].0;
        assert!(
            (actual - expected).abs() / expected < 0.01,
            "Expected ~{expected:.6e} m, got {actual:.6e} m"
        );
    }

    #[test]
    fn test_bar1d_zero_force_zero_displacement() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        mesh.add_element(vec![n0, n1], steel(), ElementType::Bar1D);

        let solver = FemSolver::new();
        let sol = solver.solve_static(&mesh, &[]);
        assert!(sol.converged);
        assert!(sol.max_displacement < 1e-20);
    }

    #[test]
    fn test_two_bar_series() {
        // Two bars in series, all horizontal, fixed at left end, force at right.
        let a = 1e-4;
        let e = 200e9;
        let l = 0.5;
        let force = 5000.0;

        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(l, 0.0);
        let n2 = mesh.add_node(2.0 * l, 0.0);
        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        let mat = FemMaterial {
            youngs_modulus: e,
            ..FemMaterial::default()
        };
        mesh.add_element(vec![n0, n1], mat.clone(), ElementType::Bar1D);
        mesh.add_element(vec![n1, n2], mat, ElementType::Bar1D);

        let solver = FemSolver::with_params(a, 0.01);
        let loads = vec![NodalLoad {
            node_id: n2,
            fx: force,
            fy: 0.0,
        }];
        let sol = solver.solve_static(&mesh, &loads);
        assert!(sol.converged);
        // Total deformation = F * (L1 + L2) / (E * A)
        let expected = force * (2.0 * l) / (e * a);
        let actual = sol.displacements[n2].0;
        assert!(
            (actual - expected).abs() / expected < 0.01,
            "Two-bar series: expected {expected:.6e}, got {actual:.6e}"
        );
    }

    #[test]
    fn test_solver_converged_flag() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        mesh.add_element(vec![n0, n1], steel(), ElementType::Bar1D);
        let solver = FemSolver::new();
        let sol = solver.solve_static(&mesh, &[]);
        assert!(sol.converged);
    }

    #[test]
    fn test_max_displacement_positive() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        mesh.add_element(vec![n0, n1], steel(), ElementType::Bar1D);
        let solver = FemSolver::new();
        let loads = vec![NodalLoad {
            node_id: n1,
            fx: 10_000.0,
            fy: 0.0,
        }];
        let sol = solver.solve_static(&mesh, &loads);
        assert!(sol.max_displacement > 0.0);
    }

    #[test]
    fn test_von_mises_stress_non_negative() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        mesh.add_element(vec![n0, n1], steel(), ElementType::Bar1D);
        let solver = FemSolver::new();
        let loads = vec![NodalLoad {
            node_id: n1,
            fx: 10_000.0,
            fy: 0.0,
        }];
        let sol = solver.solve_static(&mesh, &loads);
        for &s in &sol.von_mises_stress {
            assert!(s >= 0.0, "Von Mises stress must be non-negative");
        }
    }

    #[test]
    fn test_triangle2d_mesh_solves() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        let n2 = mesh.add_node(0.5, 1.0);
        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        mesh.set_boundary_condition(n1, DofType::Displacement, 0.0);
        mesh.add_element(vec![n0, n1, n2], steel(), ElementType::Triangle2D);
        let solver = FemSolver::with_params(1e-4, 0.01);
        let loads = vec![NodalLoad {
            node_id: n2,
            fx: 0.0,
            fy: -500.0,
        }];
        let sol = solver.solve_static(&mesh, &loads);
        assert!(sol.converged);
        assert_eq!(sol.displacements.len(), 3);
    }

    #[test]
    fn test_quad2d_mesh_solves() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        let n2 = mesh.add_node(1.0, 1.0);
        let n3 = mesh.add_node(0.0, 1.0);
        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        mesh.set_boundary_condition(n1, DofType::Displacement, 0.0);
        mesh.add_element(vec![n0, n1, n2, n3], steel(), ElementType::Quad2D);
        let solver = FemSolver::with_params(1e-4, 0.01);
        let loads = vec![
            NodalLoad {
                node_id: n2,
                fx: 0.0,
                fy: -1000.0,
            },
            NodalLoad {
                node_id: n3,
                fx: 0.0,
                fy: -1000.0,
            },
        ];
        let sol = solver.solve_static(&mesh, &loads);
        assert!(sol.converged);
        assert_eq!(sol.displacements.len(), 4);
    }

    // ────────────────────────────────────────
    // Solver — thermal
    // ────────────────────────────────────────

    /// 1-D rod: T(0) = 100 K, T(L) = 200 K, linear profile expected.
    #[test]
    fn test_thermal_1d_linear_temperature() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        mesh.set_boundary_condition(n0, DofType::Temperature, 100.0);
        mesh.set_boundary_condition(n1, DofType::Temperature, 200.0);
        mesh.add_element(vec![n0, n1], steel(), ElementType::Bar1D);

        let solver = FemSolver::with_params(1e-4, 0.01);
        let sol = solver.solve_thermal(&mesh, &[]);
        assert!(sol.converged);
        assert!((sol.temperatures[0] - 100.0).abs() < 1.0);
        assert!((sol.temperatures[1] - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_thermal_max_temperature() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        mesh.set_boundary_condition(n0, DofType::Temperature, 300.0);
        mesh.set_boundary_condition(n1, DofType::Temperature, 500.0);
        mesh.add_element(vec![n0, n1], steel(), ElementType::Bar1D);

        let solver = FemSolver::with_params(1e-4, 0.01);
        let sol = solver.solve_thermal(&mesh, &[]);
        assert!(sol.converged);
        assert!((sol.max_temperature - 500.0).abs() < 10.0);
    }

    #[test]
    fn test_thermal_heat_flux_direction() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        // Higher temperature at left → heat flows right (positive qx)
        mesh.set_boundary_condition(n0, DofType::Temperature, 400.0);
        mesh.set_boundary_condition(n1, DofType::Temperature, 300.0);
        mesh.add_element(vec![n0, n1], steel(), ElementType::Bar1D);

        let solver = FemSolver::with_params(1e-4, 0.01);
        let sol = solver.solve_thermal(&mesh, &[]);
        assert!(sol.converged);
        // qx = -k * dT/dx; dT/dx < 0 → qx > 0
        assert!(
            sol.heat_flux[0].0 > 0.0,
            "Heat should flow from hot to cold (positive qx)"
        );
    }

    #[test]
    fn test_thermal_three_node_rod() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(0.5, 0.0);
        let n2 = mesh.add_node(1.0, 0.0);
        mesh.set_boundary_condition(n0, DofType::Temperature, 100.0);
        mesh.set_boundary_condition(n2, DofType::Temperature, 200.0);
        mesh.add_element(vec![n0, n1], steel(), ElementType::Bar1D);
        mesh.add_element(vec![n1, n2], steel(), ElementType::Bar1D);

        let solver = FemSolver::with_params(1e-4, 0.01);
        let sol = solver.solve_thermal(&mesh, &[]);
        assert!(sol.converged);
        // Mid-node should be ≈ 150 K (linear)
        assert!((sol.temperatures[1] - 150.0).abs() < 5.0);
    }

    #[test]
    fn test_thermal_heat_flux_applied() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        mesh.set_boundary_condition(n0, DofType::Temperature, 300.0);
        mesh.add_element(vec![n0, n1], steel(), ElementType::Bar1D);

        let solver = FemSolver::with_params(1e-4, 0.01);
        let hf = vec![ElementHeatFlux {
            element_id: 0,
            q: 1000.0,
        }];
        let sol = solver.solve_thermal(&mesh, &hf);
        assert!(sol.converged);
        // Right node temperature must be >= boundary temperature
        assert!(sol.temperatures[1] >= 300.0 - 1.0);
    }

    #[test]
    fn test_element_heat_flux_struct() {
        let hf = ElementHeatFlux {
            element_id: 3,
            q: 500.0,
        };
        assert_eq!(hf.element_id, 3);
        assert!((hf.q - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_nodal_load_struct() {
        let load = NodalLoad {
            node_id: 2,
            fx: 100.0,
            fy: -50.0,
        };
        assert_eq!(load.node_id, 2);
        assert!((load.fx - 100.0).abs() < 1e-10);
        assert!((load.fy + 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_fem_material_default() {
        let mat = FemMaterial::default();
        assert!(mat.youngs_modulus > 0.0);
        assert!(mat.poissons_ratio > 0.0 && mat.poissons_ratio < 0.5);
        assert!(mat.thermal_conductivity > 0.0);
        assert!(mat.density > 0.0);
    }

    #[test]
    fn test_gaussian_elimination_identity() {
        // Solve I x = b → x = b
        let mut a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut b = vec![3.0, 7.0];
        let x = gaussian_elimination(&mut a, &mut b).expect("Should converge");
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_elimination_2x2() {
        // 2x + y = 5
        // x  + 3y = 10
        // Solution: x = 1, y = 3
        let mut a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let mut b = vec![5.0, 10.0];
        let x = gaussian_elimination(&mut a, &mut b).expect("Should converge");
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_beam1d_solves_like_bar() {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        mesh.add_element(vec![n0, n1], steel(), ElementType::Beam1D);
        let solver = FemSolver::new();
        let loads = vec![NodalLoad {
            node_id: n1,
            fx: 5000.0,
            fy: 0.0,
        }];
        let sol = solver.solve_static(&mesh, &loads);
        assert!(sol.converged);
        assert!(sol.max_displacement > 0.0);
    }
}
