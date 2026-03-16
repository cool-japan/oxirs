//! Thermal finite-element analysis.
//!
//! Implements triangular element heat-conduction FEM with Dirichlet, Neumann,
//! and Robin (convection) boundary conditions. Uses Gaussian elimination to
//! solve K·T = F without any external linear-algebra dependencies.

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// A node in the thermal mesh.
#[derive(Debug, Clone)]
pub struct ThermalNode {
    /// Node index.
    pub id: usize,
    /// X coordinate (meters).
    pub x: f64,
    /// Y coordinate (meters).
    pub y: f64,
    /// Initial or prescribed temperature (Kelvin or °C).
    pub temperature: f64,
    /// `true` when this node has a Dirichlet (prescribed temperature) BC.
    pub is_boundary: bool,
}

impl ThermalNode {
    /// Create an interior node.
    pub fn interior(id: usize, x: f64, y: f64) -> Self {
        Self {
            id,
            x,
            y,
            temperature: 0.0,
            is_boundary: false,
        }
    }

    /// Create a boundary node with a prescribed temperature.
    pub fn boundary(id: usize, x: f64, y: f64, temperature: f64) -> Self {
        Self {
            id,
            x,
            y,
            temperature,
            is_boundary: true,
        }
    }
}

/// Triangular thermal element (CST – Constant Strain Triangle).
#[derive(Debug, Clone)]
pub struct ThermalElement {
    /// Node indices (counter-clockwise ordering preferred).
    pub nodes: [usize; 3],
    /// Isotropic thermal conductivity (W/m·K).
    pub conductivity: f64,
    /// Element thickness (m). For 2-D plane problems.
    pub thickness: f64,
}

/// Thermal boundary conditions.
#[derive(Debug, Clone)]
pub enum ThermalBc {
    /// Prescribed temperature at a node (Dirichlet).
    DirichletTemp { node_id: usize, temp: f64 },
    /// Prescribed heat flux at a node (Neumann, positive = into domain).
    HeatFlux {
        node_id: usize,
        /// Heat flux (W/m²).
        flux: f64,
    },
    /// Convection BC at a node (Robin).
    Convection {
        node_id: usize,
        /// Convection coefficient (W/m²·K).
        h: f64,
        /// Ambient temperature (same unit as nodal temperatures).
        t_inf: f64,
    },
}

/// The thermal mesh: nodes + elements.
#[derive(Debug, Clone)]
pub struct ThermalMesh {
    pub nodes: Vec<ThermalNode>,
    pub elements: Vec<ThermalElement>,
}

impl ThermalMesh {
    /// Create a mesh from node and element lists.
    pub fn new(nodes: Vec<ThermalNode>, elements: Vec<ThermalElement>) -> Self {
        Self { nodes, elements }
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
}

/// Solver error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverError {
    /// The global stiffness matrix is singular.
    SingularMatrix,
    /// A mesh integrity issue was detected.
    InvalidMesh(String),
    /// An element referenced a node index that does not exist.
    InvalidNodeIndex(usize),
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::SingularMatrix => write!(f, "global conductivity matrix is singular"),
            SolverError::InvalidMesh(m) => write!(f, "invalid mesh: {m}"),
            SolverError::InvalidNodeIndex(i) => write!(f, "invalid node index: {i}"),
        }
    }
}

impl std::error::Error for SolverError {}

/// Thermal analysis result.
#[derive(Debug, Clone)]
pub struct ThermalResult {
    /// Solved nodal temperatures.
    pub temperatures: Vec<f64>,
    /// Maximum nodal temperature.
    pub max_temp: f64,
    /// Minimum nodal temperature.
    pub min_temp: f64,
    /// Heat flux (qx, qy) per element.
    pub heat_fluxes: Vec<(f64, f64)>,
}

// ──────────────────────────────────────────────────────────────────────────────
// ThermalSolver
// ──────────────────────────────────────────────────────────────────────────────

/// Thermal FEM solver.
pub struct ThermalSolver;

impl ThermalSolver {
    /// Create a new solver instance.
    pub fn new() -> Self {
        Self
    }

    /// Assemble the global conductivity matrix K (n×n dense).
    pub fn assemble_conductivity_matrix(
        &self,
        mesh: &ThermalMesh,
    ) -> Result<Vec<Vec<f64>>, SolverError> {
        let n = mesh.n_nodes();
        let mut k = vec![vec![0.0; n]; n];

        for elem in &mesh.elements {
            let ke = self.element_conductivity_matrix(mesh, elem)?;
            for (local_i, &global_i) in elem.nodes.iter().enumerate() {
                for (local_j, &global_j) in elem.nodes.iter().enumerate() {
                    k[global_i][global_j] += ke[local_i][local_j];
                }
            }
        }
        Ok(k)
    }

    /// Assemble the global load vector F (length n).
    pub fn assemble_load_vector(&self, mesh: &ThermalMesh, bcs: &[ThermalBc]) -> Vec<f64> {
        let n = mesh.n_nodes();
        let mut f = vec![0.0; n];

        for bc in bcs {
            match bc {
                ThermalBc::HeatFlux { node_id, flux } => {
                    if *node_id < n {
                        f[*node_id] += flux;
                    }
                }
                ThermalBc::Convection { node_id, h, t_inf } => {
                    if *node_id < n {
                        f[*node_id] += h * t_inf;
                    }
                }
                ThermalBc::DirichletTemp { .. } => {
                    // Applied via penalty method in solve()
                }
            }
        }
        f
    }

    /// Solve K·T = F for nodal temperatures.
    ///
    /// Dirichlet BCs are enforced with the large-number (penalty) method.
    pub fn solve(&self, mesh: &ThermalMesh, bcs: &[ThermalBc]) -> Result<Vec<f64>, SolverError> {
        if mesh.nodes.is_empty() {
            return Err(SolverError::InvalidMesh("empty mesh".to_string()));
        }

        let n = mesh.n_nodes();
        let mut k = self.assemble_conductivity_matrix(mesh)?;
        let mut f = self.assemble_load_vector(mesh, bcs);

        // Apply Dirichlet BCs via penalty method (large number α)
        let alpha = self.penalty_factor(&k);

        // Also apply prescribed temperatures from node.is_boundary
        for node in &mesh.nodes {
            if node.is_boundary {
                k[node.id][node.id] += alpha;
                f[node.id] += alpha * node.temperature;
            }
        }

        // Dirichlet BCs from BC list override node.temperature
        for bc in bcs {
            if let ThermalBc::DirichletTemp { node_id, temp } = bc {
                if *node_id < n {
                    k[*node_id][*node_id] += alpha;
                    f[*node_id] += alpha * temp;
                }
            }
        }

        // Convection BCs modify the diagonal of K
        for bc in bcs {
            if let ThermalBc::Convection { node_id, h, .. } = bc {
                if *node_id < n {
                    k[*node_id][*node_id] += h;
                }
            }
        }

        gaussian_elimination(&mut k, &mut f)
    }

    /// Compute the element heat flux (qx, qy) from solved temperatures.
    ///
    /// For a CST element: q = −k · B · T_e
    pub fn heat_flux_at_element(
        &self,
        mesh: &ThermalMesh,
        temps: &[f64],
        elem_id: usize,
    ) -> (f64, f64) {
        let elem = &mesh.elements[elem_id];
        let [i, j, m] = elem.nodes;

        let (xi, yi) = (mesh.nodes[i].x, mesh.nodes[i].y);
        let (xj, yj) = (mesh.nodes[j].x, mesh.nodes[j].y);
        let (xm, ym) = (mesh.nodes[m].x, mesh.nodes[m].y);

        // B matrix rows (1 / 2A)
        let two_area = (xj - xi) * (ym - yi) - (xm - xi) * (yj - yi);
        if two_area.abs() < 1e-15 {
            return (0.0, 0.0);
        }

        // dN/dx and dN/dy for the three shape functions
        let b_x = [(yj - ym), (ym - yi), (yi - yj)];
        let b_y = [(xm - xj), (xi - xm), (xj - xi)];

        let ti = temps[i];
        let tj = temps[j];
        let tm = temps[m];
        let t_e = [ti, tj, tm];

        let grad_t_x: f64 = (b_x[0] * t_e[0] + b_x[1] * t_e[1] + b_x[2] * t_e[2]) / two_area;
        let grad_t_y: f64 = (b_y[0] * t_e[0] + b_y[1] * t_e[1] + b_y[2] * t_e[2]) / two_area;

        // q = −k ∇T
        let qx = -elem.conductivity * grad_t_x;
        let qy = -elem.conductivity * grad_t_y;
        (qx, qy)
    }

    /// Solve and return a full `ThermalResult`.
    pub fn analyze(
        &self,
        mesh: &ThermalMesh,
        bcs: &[ThermalBc],
    ) -> Result<ThermalResult, SolverError> {
        let temperatures = self.solve(mesh, bcs)?;
        let max_temp = temperatures
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_temp = temperatures.iter().cloned().fold(f64::INFINITY, f64::min);
        let heat_fluxes: Vec<(f64, f64)> = (0..mesh.elements.len())
            .map(|i| self.heat_flux_at_element(mesh, &temperatures, i))
            .collect();
        Ok(ThermalResult {
            temperatures,
            max_temp,
            min_temp,
            heat_fluxes,
        })
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// 3×3 element conductivity matrix for a CST element.
    fn element_conductivity_matrix(
        &self,
        mesh: &ThermalMesh,
        elem: &ThermalElement,
    ) -> Result<[[f64; 3]; 3], SolverError> {
        let [i, j, m] = elem.nodes;
        let n = mesh.n_nodes();
        for &idx in &[i, j, m] {
            if idx >= n {
                return Err(SolverError::InvalidNodeIndex(idx));
            }
        }

        let (xi, yi) = (mesh.nodes[i].x, mesh.nodes[i].y);
        let (xj, yj) = (mesh.nodes[j].x, mesh.nodes[j].y);
        let (xm, ym) = (mesh.nodes[m].x, mesh.nodes[m].y);

        let two_area = (xj - xi) * (ym - yi) - (xm - xi) * (yj - yi);
        if two_area.abs() < 1e-15 {
            // Degenerate (zero-area) element: contribute nothing
            return Ok([[0.0; 3]; 3]);
        }
        let area = two_area.abs() / 2.0;

        // Shape-function gradient components (constant over CST element)
        let b_x = [(yj - ym), (ym - yi), (yi - yj)];
        let b_y = [(xm - xj), (xi - xm), (xj - xi)];

        let k_factor = elem.conductivity * elem.thickness * area / (two_area * two_area);

        let mut ke = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                ke[r][c] = k_factor * (b_x[r] * b_x[c] + b_y[r] * b_y[c]);
            }
        }
        Ok(ke)
    }

    /// Choose a penalty factor α ≈ 10⁶ × max(|K_{ii}|).
    fn penalty_factor(&self, k: &[Vec<f64>]) -> f64 {
        let max_diag = k
            .iter()
            .enumerate()
            .map(|(i, row)| row[i].abs())
            .fold(0.0_f64, f64::max);
        if max_diag < 1e-15 {
            1e6_f64
        } else {
            max_diag * 1e6
        }
    }
}

impl Default for ThermalSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Gaussian elimination
// ──────────────────────────────────────────────────────────────────────────────

/// Solve A·x = b via partial-pivot Gaussian elimination.
/// Modifies `a` and `b` in place; returns the solution vector.
fn gaussian_elimination(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<Vec<f64>, SolverError> {
    let n = b.len();

    for col in 0..n {
        // Partial pivot
        let (pivot_row, _) = (col..n)
            .map(|r| (r, a[r][col].abs()))
            .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((col, 0.0));

        if a[pivot_row][col].abs() < 1e-15 {
            return Err(SolverError::SingularMatrix);
        }

        a.swap(col, pivot_row);
        b.swap(col, pivot_row);

        let pivot = a[col][col];
        for row in col + 1..n {
            let factor = a[row][col] / pivot;
            b[row] -= factor * b[col];
            // row > col always here; split to satisfy borrow checker
            let (upper, lower) = a.split_at_mut(row);
            for (rv, cv) in lower[0][col..n].iter_mut().zip(upper[col][col..n].iter()) {
                *rv -= factor * cv;
            }
        }
    }

    // Back-substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Ok(x)
}

// ──────────────────────────────────────────────────────────────────────────────
// Test meshes helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Build a simple two-triangle unit-square mesh.
///
/// Nodes: (0,0), (1,0), (1,1), (0,1)
/// Elements: [(0,1,2), (0,2,3)]
#[cfg(test)]
fn square_mesh(k: f64) -> ThermalMesh {
    let nodes = vec![
        ThermalNode::interior(0, 0.0, 0.0),
        ThermalNode::interior(1, 1.0, 0.0),
        ThermalNode::interior(2, 1.0, 1.0),
        ThermalNode::interior(3, 0.0, 1.0),
    ];
    let elements = vec![
        ThermalElement {
            nodes: [0, 1, 2],
            conductivity: k,
            thickness: 1.0,
        },
        ThermalElement {
            nodes: [0, 2, 3],
            conductivity: k,
            thickness: 1.0,
        },
    ];
    ThermalMesh::new(nodes, elements)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ThermalNode ───────────────────────────────────────────────────────────

    #[test]
    fn test_node_interior() {
        let n = ThermalNode::interior(0, 1.0, 2.0);
        assert_eq!(n.id, 0);
        assert!(!n.is_boundary);
        assert_eq!(n.temperature, 0.0);
    }

    #[test]
    fn test_node_boundary() {
        let n = ThermalNode::boundary(1, 0.0, 0.0, 300.0);
        assert!(n.is_boundary);
        assert_eq!(n.temperature, 300.0);
    }

    // ── ThermalMesh ───────────────────────────────────────────────────────────

    #[test]
    fn test_mesh_n_nodes() {
        let mesh = square_mesh(10.0);
        assert_eq!(mesh.n_nodes(), 4);
    }

    #[test]
    fn test_mesh_n_elements() {
        let mesh = square_mesh(10.0);
        assert_eq!(mesh.elements.len(), 2);
    }

    // ── SolverError ───────────────────────────────────────────────────────────

    #[test]
    fn test_error_display_singular() {
        let e = SolverError::SingularMatrix;
        assert!(format!("{e}").contains("singular"));
    }

    #[test]
    fn test_error_display_invalid_mesh() {
        let e = SolverError::InvalidMesh("empty".to_string());
        assert!(format!("{e}").contains("empty"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(SolverError::SingularMatrix);
        assert!(!e.to_string().is_empty());
    }

    // ── assemble_conductivity_matrix ──────────────────────────────────────────

    #[test]
    fn test_conductivity_matrix_size() {
        let mesh = square_mesh(10.0);
        let solver = ThermalSolver::new();
        let k = solver
            .assemble_conductivity_matrix(&mesh)
            .expect("should succeed");
        assert_eq!(k.len(), 4);
        assert_eq!(k[0].len(), 4);
    }

    #[test]
    fn test_conductivity_matrix_symmetric() {
        let mesh = square_mesh(5.0);
        let solver = ThermalSolver::new();
        let k = solver
            .assemble_conductivity_matrix(&mesh)
            .expect("should succeed");
        for (i, row_i) in k.iter().enumerate() {
            for (j, &kij) in row_i.iter().enumerate() {
                let kji = k[j][i];
                assert!(
                    (kij - kji).abs() < 1e-12,
                    "K[{i}][{j}] = {kij} ≠ K[{j}][{i}] = {kji}",
                );
            }
        }
    }

    #[test]
    fn test_conductivity_matrix_positive_diagonal() {
        let mesh = square_mesh(1.0);
        let solver = ThermalSolver::new();
        let k = solver
            .assemble_conductivity_matrix(&mesh)
            .expect("should succeed");
        for (i, row_i) in k.iter().enumerate() {
            assert!(row_i[i] >= 0.0, "K[{i}][{i}] should be non-negative");
        }
    }

    #[test]
    fn test_conductivity_matrix_row_sum_near_zero() {
        // For a pure conductivity matrix (no BCs) the row sums should be 0.
        let mesh = square_mesh(1.0);
        let solver = ThermalSolver::new();
        let k = solver
            .assemble_conductivity_matrix(&mesh)
            .expect("should succeed");
        for (i, row) in k.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-10, "Row {i} sum = {sum} should be ~0");
        }
    }

    // ── assemble_load_vector ──────────────────────────────────────────────────

    #[test]
    fn test_load_vector_zeros_without_bcs() {
        let mesh = square_mesh(1.0);
        let solver = ThermalSolver::new();
        let f = solver.assemble_load_vector(&mesh, &[]);
        assert!(f.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_load_vector_heat_flux() {
        let mesh = square_mesh(1.0);
        let solver = ThermalSolver::new();
        let bcs = vec![ThermalBc::HeatFlux {
            node_id: 0,
            flux: 100.0,
        }];
        let f = solver.assemble_load_vector(&mesh, &bcs);
        assert!((f[0] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_load_vector_convection() {
        let mesh = square_mesh(1.0);
        let solver = ThermalSolver::new();
        let bcs = vec![ThermalBc::Convection {
            node_id: 1,
            h: 20.0,
            t_inf: 300.0,
        }];
        let f = solver.assemble_load_vector(&mesh, &bcs);
        assert!((f[1] - 20.0 * 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_load_vector_dirichlet_not_in_f() {
        let mesh = square_mesh(1.0);
        let solver = ThermalSolver::new();
        let bcs = vec![ThermalBc::DirichletTemp {
            node_id: 0,
            temp: 500.0,
        }];
        let f = solver.assemble_load_vector(&mesh, &bcs);
        // Dirichlet is handled in solve(), not assemble_load_vector
        assert_eq!(f[0], 0.0);
    }

    // ── solve ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_solve_uniform_temperature() {
        // If all boundary nodes are at 100°C the interior should also be 100°C.
        let nodes = vec![
            ThermalNode::boundary(0, 0.0, 0.0, 100.0),
            ThermalNode::boundary(1, 1.0, 0.0, 100.0),
            ThermalNode::interior(2, 0.5, 0.5),
            ThermalNode::boundary(3, 0.0, 1.0, 100.0),
        ];
        let elements = vec![
            ThermalElement {
                nodes: [0, 1, 2],
                conductivity: 1.0,
                thickness: 1.0,
            },
            ThermalElement {
                nodes: [0, 2, 3],
                conductivity: 1.0,
                thickness: 1.0,
            },
        ];
        let mesh = ThermalMesh::new(nodes, vec![]);
        let mesh = ThermalMesh::new(mesh.nodes, elements);
        let solver = ThermalSolver::new();
        let bcs = vec![
            ThermalBc::DirichletTemp {
                node_id: 0,
                temp: 100.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 1,
                temp: 100.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 3,
                temp: 100.0,
            },
        ];
        let temps = solver.solve(&mesh, &bcs).expect("should succeed");
        // Interior node 2 should be ≈ 100°C
        assert!(
            (temps[2] - 100.0).abs() < 1.0,
            "Interior temp = {}",
            temps[2]
        );
    }

    #[test]
    fn test_solve_returns_n_temps() {
        let mesh = square_mesh(1.0);
        let solver = ThermalSolver::new();
        let bcs = vec![
            ThermalBc::DirichletTemp {
                node_id: 0,
                temp: 0.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 1,
                temp: 0.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 2,
                temp: 100.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 3,
                temp: 100.0,
            },
        ];
        let temps = solver.solve(&mesh, &bcs).expect("should succeed");
        assert_eq!(temps.len(), 4);
    }

    #[test]
    fn test_solve_dirichlet_nodes_match_prescribed() {
        let mesh = square_mesh(10.0);
        let solver = ThermalSolver::new();
        let bcs = vec![
            ThermalBc::DirichletTemp {
                node_id: 0,
                temp: 0.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 1,
                temp: 0.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 2,
                temp: 200.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 3,
                temp: 200.0,
            },
        ];
        let temps = solver.solve(&mesh, &bcs).expect("should succeed");
        // Penalty method → very close to prescribed values
        assert!(temps[0].abs() < 1.0, "node 0 ≈ 0: {}", temps[0]);
        assert!(temps[1].abs() < 1.0, "node 1 ≈ 0: {}", temps[1]);
        assert!((temps[2] - 200.0).abs() < 1.0, "node 2 ≈ 200: {}", temps[2]);
        assert!((temps[3] - 200.0).abs() < 1.0, "node 3 ≈ 200: {}", temps[3]);
    }

    #[test]
    fn test_solve_empty_mesh_error() {
        let mesh = ThermalMesh::new(vec![], vec![]);
        let solver = ThermalSolver::new();
        assert!(matches!(
            solver.solve(&mesh, &[]),
            Err(SolverError::InvalidMesh(_))
        ));
    }

    #[test]
    fn test_solve_gradient_monotone() {
        // Linear gradient: nodes 0,1 at 0°C and nodes 2,3 at 100°C.
        let mesh = square_mesh(1.0);
        let solver = ThermalSolver::new();
        let bcs = vec![
            ThermalBc::DirichletTemp {
                node_id: 0,
                temp: 0.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 1,
                temp: 0.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 2,
                temp: 100.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 3,
                temp: 100.0,
            },
        ];
        let temps = solver.solve(&mesh, &bcs).expect("should succeed");
        // Nodes 2,3 are hotter than 0,1
        assert!(temps[2] > temps[0]);
        assert!(temps[3] > temps[1]);
    }

    // ── heat_flux_at_element ──────────────────────────────────────────────────

    #[test]
    fn test_heat_flux_uniform_temperature_is_zero() {
        let mesh = square_mesh(1.0);
        let solver = ThermalSolver::new();
        let temps = vec![100.0; 4];
        let (qx, qy) = solver.heat_flux_at_element(&mesh, &temps, 0);
        assert!(qx.abs() < 1e-10, "qx should be ~0, got {qx}");
        assert!(qy.abs() < 1e-10, "qy should be ~0, got {qy}");
    }

    #[test]
    fn test_heat_flux_direction() {
        // Temperature increases in the y direction → expect nonzero qy.
        let mesh = square_mesh(1.0);
        let solver = ThermalSolver::new();
        let bcs = vec![
            ThermalBc::DirichletTemp {
                node_id: 0,
                temp: 0.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 1,
                temp: 0.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 2,
                temp: 100.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 3,
                temp: 100.0,
            },
        ];
        let temps = solver.solve(&mesh, &bcs).expect("should succeed");
        let (_, qy) = solver.heat_flux_at_element(&mesh, &temps, 0);
        // qy should be negative (heat flows from high T to low T, i.e. downward)
        assert!(qy < 0.0, "qy = {qy} should be < 0");
    }

    #[test]
    fn test_heat_flux_all_elements() {
        let mesh = square_mesh(1.0);
        let solver = ThermalSolver::new();
        let temps = vec![0.0, 0.0, 100.0, 100.0];
        for elem_id in 0..mesh.elements.len() {
            let (qx, qy) = solver.heat_flux_at_element(&mesh, &temps, elem_id);
            assert!(qx.is_finite());
            assert!(qy.is_finite());
        }
    }

    // ── analyze ───────────────────────────────────────────────────────────────

    #[test]
    fn test_analyze_result_fields() {
        let mesh = square_mesh(1.0);
        let solver = ThermalSolver::new();
        let bcs = vec![
            ThermalBc::DirichletTemp {
                node_id: 0,
                temp: 20.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 1,
                temp: 20.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 2,
                temp: 80.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 3,
                temp: 80.0,
            },
        ];
        let result = solver.analyze(&mesh, &bcs).expect("should succeed");
        assert_eq!(result.temperatures.len(), 4);
        assert!(result.max_temp >= result.min_temp);
        assert_eq!(result.heat_fluxes.len(), 2);
    }

    #[test]
    fn test_analyze_max_min_temp() {
        let mesh = square_mesh(1.0);
        let solver = ThermalSolver::new();
        let bcs = vec![
            ThermalBc::DirichletTemp {
                node_id: 0,
                temp: 0.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 1,
                temp: 0.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 2,
                temp: 100.0,
            },
            ThermalBc::DirichletTemp {
                node_id: 3,
                temp: 100.0,
            },
        ];
        let result = solver.analyze(&mesh, &bcs).expect("should succeed");
        assert!(result.max_temp >= 90.0);
        assert!(result.min_temp <= 10.0);
    }

    // ── Gaussian elimination ───────────────────────────────────────────────────

    #[test]
    fn test_gaussian_elimination_2x2() {
        // 2x - y = 3 ; x + y = 3 → x=2, y=1
        let mut a = vec![vec![2.0_f64, -1.0], vec![1.0, 1.0]];
        let mut b = vec![3.0_f64, 3.0];
        let x = gaussian_elimination(&mut a, &mut b).expect("should succeed");
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_elimination_3x3() {
        // x + y + z = 6 ; x - y + z = 2 ; 2x + y - z = 1 → x=1, y=2, z=3
        let mut a = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, -1.0, 1.0],
            vec![2.0, 1.0, -1.0],
        ];
        let mut b = vec![6.0, 2.0, 1.0];
        let x = gaussian_elimination(&mut a, &mut b).expect("should succeed");
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] - 2.0).abs() < 1e-9);
        assert!((x[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_solver_default() {
        let _ = ThermalSolver::new();
    }

    #[test]
    fn test_boundary_node_via_is_boundary_flag() {
        let nodes = vec![
            ThermalNode::boundary(0, 0.0, 0.0, 50.0),
            ThermalNode::interior(1, 1.0, 0.0),
            ThermalNode::boundary(2, 0.5, 1.0, 50.0),
        ];
        let elements = vec![ThermalElement {
            nodes: [0, 1, 2],
            conductivity: 1.0,
            thickness: 1.0,
        }];
        let mesh = ThermalMesh::new(nodes, elements);
        let solver = ThermalSolver::new();
        let temps = solver.solve(&mesh, &[]).expect("should succeed");
        // Node 1 should converge close to 50°C
        assert!(
            (temps[1] - 50.0).abs() < 5.0,
            "Interior node temp = {}",
            temps[1]
        );
    }
}
