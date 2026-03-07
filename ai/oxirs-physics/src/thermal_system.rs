//! Thermal analysis for physical systems: heat conduction, convection, and radiation.
//!
//! Provides lumped-capacity models, Fourier / Newton / Stefan-Boltzmann calculations,
//! Biot and Fourier numbers, 1-D steady-state temperature profiles, and a
//! network-level heat-balance solver.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Material
// ─────────────────────────────────────────────────────────────────────────────

/// Thermal material properties.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalMaterial {
    /// Material name (e.g. "copper", "aluminium").
    pub name: String,
    /// Thermal conductivity k (W/(m·K)).
    pub thermal_conductivity: f64,
    /// Density ρ (kg/m³).
    pub density: f64,
    /// Specific heat capacity cp (J/(kg·K)).
    pub specific_heat: f64,
}

impl ThermalMaterial {
    /// Create a new material.
    pub fn new(
        name: impl Into<String>,
        thermal_conductivity: f64,
        density: f64,
        specific_heat: f64,
    ) -> Self {
        Self {
            name: name.into(),
            thermal_conductivity,
            density,
            specific_heat,
        }
    }

    /// Thermal diffusivity α = k / (ρ · cp)  [m²/s].
    pub fn thermal_diffusivity(&self) -> f64 {
        if self.density * self.specific_heat == 0.0 {
            return 0.0;
        }
        self.thermal_conductivity / (self.density * self.specific_heat)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Network model
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the thermal network.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalNetNode {
    /// Unique node identifier.
    pub id: String,
    /// Current temperature (K or °C, consistent within the model).
    pub temperature: f64,
    /// Internal heat generation (W). Positive = source.
    pub heat_source: f64,
}

impl ThermalNetNode {
    /// Create a new node.
    pub fn new(id: impl Into<String>, temperature: f64, heat_source: f64) -> Self {
        Self {
            id: id.into(),
            temperature,
            heat_source,
        }
    }
}

/// A thermal conductance link between two nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalEdge {
    /// Source node id.
    pub from: String,
    /// Destination node id.
    pub to: String,
    /// Thermal conductance G (W/K). Heat flow = G × ΔT.
    pub conductance: f64,
}

impl ThermalEdge {
    /// Create a new edge.
    pub fn new(from: impl Into<String>, to: impl Into<String>, conductance: f64) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            conductance,
        }
    }
}

/// A lumped thermal network of nodes connected by conductance edges.
#[derive(Debug, Clone, Default)]
pub struct ThermalSystem {
    /// Nodes indexed by their id.
    pub nodes: HashMap<String, ThermalNetNode>,
    /// Directed conductance edges (bidirectional heat flow).
    pub edges: Vec<ThermalEdge>,
    /// Simulation time (s). Updated by transient solvers.
    pub time: f64,
}

impl ThermalSystem {
    /// Create an empty thermal system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node. Returns `false` if a node with the same id already exists.
    pub fn add_node(&mut self, node: ThermalNetNode) -> bool {
        use std::collections::hash_map::Entry;
        match self.nodes.entry(node.id.clone()) {
            Entry::Vacant(e) => {
                e.insert(node);
                true
            }
            Entry::Occupied(_) => false,
        }
    }

    /// Add a conductance edge.
    pub fn add_edge(&mut self, edge: ThermalEdge) {
        self.edges.push(edge);
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Stefan-Boltzmann constant σ [W/(m²·K⁴)].
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

// ─────────────────────────────────────────────────────────────────────────────
// ThermalAnalysis
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless heat-transfer analysis utilities.
///
/// All functions are pure; no mutable state is kept.
pub struct ThermalAnalysis;

impl ThermalAnalysis {
    // ── Fourier's law of heat conduction ─────────────────────────────────────

    /// Heat flux via Fourier's law: q = k · A · ΔT / L  (W).
    ///
    /// # Arguments
    /// * `conductivity` – thermal conductivity k (W/(m·K))
    /// * `area` – cross-sectional area A (m²)
    /// * `temp_diff` – temperature difference ΔT (K or °C)
    /// * `thickness` – slab thickness L (m)
    pub fn fourier_heat_flux(conductivity: f64, area: f64, temp_diff: f64, thickness: f64) -> f64 {
        if thickness == 0.0 {
            return 0.0;
        }
        conductivity * area * temp_diff / thickness
    }

    // ── Newton's law of cooling ───────────────────────────────────────────────

    /// Convective heat transfer: Q = h · A · (T_surface − T_fluid)  (W).
    ///
    /// # Arguments
    /// * `h` – convective heat-transfer coefficient (W/(m²·K))
    /// * `area` – surface area (m²)
    /// * `t_surface` – surface temperature (K or °C)
    /// * `t_fluid` – fluid (ambient) temperature (K or °C)
    pub fn convective_heat_transfer(h: f64, area: f64, t_surface: f64, t_fluid: f64) -> f64 {
        h * area * (t_surface - t_fluid)
    }

    // ── Stefan-Boltzmann radiation ────────────────────────────────────────────

    /// Radiative heat transfer between two grey surfaces via Stefan-Boltzmann:
    /// Q = ε · σ · A · (T_hot⁴ − T_cold⁴)  (W).
    ///
    /// Temperatures must be in **Kelvin**.
    ///
    /// # Arguments
    /// * `emissivity` – surface emissivity ε (0–1)
    /// * `area` – radiating area A (m²)
    /// * `t_hot` – temperature of the hot surface (K)
    /// * `t_cold` – temperature of the cold surface / surroundings (K)
    pub fn radiative_heat_transfer(emissivity: f64, area: f64, t_hot: f64, t_cold: f64) -> f64 {
        emissivity * STEFAN_BOLTZMANN * area * (t_hot.powi(4) - t_cold.powi(4))
    }

    // ── Thermal resistance ────────────────────────────────────────────────────

    /// Thermal resistance of a slab: R = L / (k · A)  (K/W).
    ///
    /// # Arguments
    /// * `thickness` – L (m)
    /// * `conductivity` – k (W/(m·K))
    /// * `area` – A (m²)
    pub fn thermal_resistance(thickness: f64, conductivity: f64, area: f64) -> f64 {
        if conductivity * area == 0.0 {
            return f64::INFINITY;
        }
        thickness / (conductivity * area)
    }

    // ── Biot number ───────────────────────────────────────────────────────────

    /// Biot number: Bi = h · Lc / k  (dimensionless).
    ///
    /// Bi < 0.1 justifies the lumped-capacity assumption.
    ///
    /// # Arguments
    /// * `h` – convection coefficient (W/(m²·K))
    /// * `lc` – characteristic length Lc = V/A (m)
    /// * `k` – thermal conductivity (W/(m·K))
    pub fn biot_number(h: f64, lc: f64, k: f64) -> f64 {
        if k == 0.0 {
            return f64::INFINITY;
        }
        h * lc / k
    }

    // ── Fourier number ────────────────────────────────────────────────────────

    /// Fourier number: Fo = α · t / Lc²  (dimensionless).
    ///
    /// # Arguments
    /// * `alpha` – thermal diffusivity α = k/(ρ·cp)  (m²/s)
    /// * `t` – time (s)
    /// * `lc` – characteristic length (m)
    pub fn fourier_number(alpha: f64, t: f64, lc: f64) -> f64 {
        if lc == 0.0 {
            return f64::INFINITY;
        }
        alpha * t / (lc * lc)
    }

    // ── Lumped-capacity transient temperature ─────────────────────────────────

    /// Transient temperature of a lumped body (Bi < 0.1):
    ///
    /// θ*(t) = (T(t) − T∞) / (T₀ − T∞) = exp(−Bi · Fo)
    ///
    /// Returns T(t).
    ///
    /// # Arguments
    /// * `t0` – initial body temperature
    /// * `t_inf` – ambient temperature (same units as t0)
    /// * `biot` – Biot number
    /// * `fourier` – Fourier number
    pub fn transient_temperature(t0: f64, t_inf: f64, biot: f64, fourier: f64) -> f64 {
        let theta_star = (-biot * fourier).exp();
        t_inf + theta_star * (t0 - t_inf)
    }

    // ── 1-D steady-state conduction ───────────────────────────────────────────

    /// Linear temperature profile for 1-D steady-state conduction in a homogeneous slab.
    ///
    /// Returns `length + 1` nodal temperatures evenly spaced from `t_left` to `t_right`.
    ///
    /// # Arguments
    /// * `t_left` – temperature at the left boundary
    /// * `t_right` – temperature at the right boundary
    /// * `_conductivity` – conductivity (not needed for linear profile, kept for API symmetry)
    /// * `length` – number of **intervals** (so `length + 1` nodes are returned)
    pub fn steady_state_1d(
        t_left: f64,
        t_right: f64,
        _conductivity: f64,
        length: usize,
    ) -> Vec<f64> {
        if length == 0 {
            return vec![t_left];
        }
        let n = length + 1;
        (0..n)
            .map(|i| t_left + (t_right - t_left) * (i as f64 / length as f64))
            .collect()
    }

    // ── Network heat balance ──────────────────────────────────────────────────

    /// Compute the net heat flow (W) into each node of a thermal network.
    ///
    /// Net heat flow at node i =
    ///   Σ_{edges (i,j)} G_{ij} · (T_j − T_i)
    /// + Σ_{edges (j,i)} G_{ji} · (T_j − T_i)
    /// + Q_source,i
    ///
    /// Returns a `HashMap<node_id, net_heat_flow>`.
    pub fn heat_balance(system: &ThermalSystem) -> HashMap<String, f64> {
        let mut balance: HashMap<String, f64> = system
            .nodes
            .iter()
            .map(|(id, node)| (id.clone(), node.heat_source))
            .collect();

        for edge in &system.edges {
            if let (Some(from_node), Some(to_node)) =
                (system.nodes.get(&edge.from), system.nodes.get(&edge.to))
            {
                let delta_t = to_node.temperature - from_node.temperature;
                let q = edge.conductance * delta_t;
                // Heat flows from hot to cold through the conductance
                *balance.entry(edge.from.clone()).or_insert(0.0) += q;
                *balance.entry(edge.to.clone()).or_insert(0.0) -= q;
            }
        }

        balance
    }

    // ── Composite resistance ──────────────────────────────────────────────────

    /// Series combination of two thermal resistances R1 + R2  (K/W).
    pub fn series_resistance(r1: f64, r2: f64) -> f64 {
        r1 + r2
    }

    /// Parallel combination of two thermal resistances (K/W).
    pub fn parallel_resistance(r1: f64, r2: f64) -> f64 {
        if r1 + r2 == 0.0 {
            return 0.0;
        }
        r1 * r2 / (r1 + r2)
    }

    // ── Overall heat transfer coefficient ────────────────────────────────────

    /// Overall heat-transfer coefficient U from component resistances:
    /// U = 1 / (R_total · A)  (W/(m²·K)).
    ///
    /// R_total is the sum of all resistance values provided.
    pub fn overall_htc(resistances: &[f64], area: f64) -> f64 {
        let r_total: f64 = resistances.iter().sum();
        if r_total * area == 0.0 {
            return f64::INFINITY;
        }
        1.0 / (r_total * area)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ThermalMaterial ───────────────────────────────────────────────────────

    #[test]
    fn test_material_new() {
        let m = ThermalMaterial::new("copper", 401.0, 8960.0, 385.0);
        assert_eq!(m.name, "copper");
        assert_eq!(m.thermal_conductivity, 401.0);
        assert_eq!(m.density, 8960.0);
        assert_eq!(m.specific_heat, 385.0);
    }

    #[test]
    fn test_material_diffusivity_copper() {
        // α_copper ≈ 1.16e-4 m²/s
        let m = ThermalMaterial::new("copper", 401.0, 8960.0, 385.0);
        let alpha = m.thermal_diffusivity();
        assert!((alpha - 1.163e-4).abs() < 5e-7, "α = {alpha}");
    }

    #[test]
    fn test_material_diffusivity_zero_rho() {
        let m = ThermalMaterial::new("dummy", 1.0, 0.0, 1.0);
        assert_eq!(m.thermal_diffusivity(), 0.0);
    }

    #[test]
    fn test_material_clone() {
        let m = ThermalMaterial::new("steel", 50.0, 7850.0, 490.0);
        let m2 = m.clone();
        assert_eq!(m, m2);
    }

    // ── ThermalNetNode ────────────────────────────────────────────────────────

    #[test]
    fn test_node_new() {
        let n = ThermalNetNode::new("A", 300.0, 100.0);
        assert_eq!(n.id, "A");
        assert_eq!(n.temperature, 300.0);
        assert_eq!(n.heat_source, 100.0);
    }

    #[test]
    fn test_node_clone() {
        let n = ThermalNetNode::new("B", 250.0, 0.0);
        let n2 = n.clone();
        assert_eq!(n, n2);
    }

    // ── ThermalEdge ───────────────────────────────────────────────────────────

    #[test]
    fn test_edge_new() {
        let e = ThermalEdge::new("A", "B", 5.0);
        assert_eq!(e.from, "A");
        assert_eq!(e.to, "B");
        assert_eq!(e.conductance, 5.0);
    }

    #[test]
    fn test_edge_clone() {
        let e = ThermalEdge::new("X", "Y", 2.5);
        let e2 = e.clone();
        assert_eq!(e, e2);
    }

    // ── ThermalSystem ─────────────────────────────────────────────────────────

    #[test]
    fn test_system_empty() {
        let sys = ThermalSystem::new();
        assert_eq!(sys.node_count(), 0);
        assert!(sys.edges.is_empty());
    }

    #[test]
    fn test_system_add_node() {
        let mut sys = ThermalSystem::new();
        assert!(sys.add_node(ThermalNetNode::new("A", 300.0, 0.0)));
        assert_eq!(sys.node_count(), 1);
    }

    #[test]
    fn test_system_duplicate_node_rejected() {
        let mut sys = ThermalSystem::new();
        assert!(sys.add_node(ThermalNetNode::new("A", 300.0, 0.0)));
        assert!(!sys.add_node(ThermalNetNode::new("A", 400.0, 0.0)));
        assert_eq!(sys.node_count(), 1);
    }

    #[test]
    fn test_system_add_edge() {
        let mut sys = ThermalSystem::new();
        sys.add_edge(ThermalEdge::new("A", "B", 3.0));
        assert_eq!(sys.edges.len(), 1);
    }

    // ── fourier_heat_flux ─────────────────────────────────────────────────────

    #[test]
    fn test_fourier_heat_flux_basic() {
        // k=10, A=2, ΔT=50, L=0.1 → Q = 10·2·50/0.1 = 10000 W
        let q = ThermalAnalysis::fourier_heat_flux(10.0, 2.0, 50.0, 0.1);
        assert!((q - 10_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_fourier_heat_flux_zero_thickness() {
        let q = ThermalAnalysis::fourier_heat_flux(10.0, 1.0, 100.0, 0.0);
        assert_eq!(q, 0.0);
    }

    #[test]
    fn test_fourier_heat_flux_positive() {
        let q = ThermalAnalysis::fourier_heat_flux(50.0, 0.5, 30.0, 0.05);
        assert!(q > 0.0);
    }

    #[test]
    fn test_fourier_heat_flux_proportional_to_area() {
        let q1 = ThermalAnalysis::fourier_heat_flux(1.0, 1.0, 10.0, 1.0);
        let q2 = ThermalAnalysis::fourier_heat_flux(1.0, 2.0, 10.0, 1.0);
        assert!((q2 - 2.0 * q1).abs() < 1e-10);
    }

    // ── convective_heat_transfer ──────────────────────────────────────────────

    #[test]
    fn test_convective_heat_transfer_basic() {
        // h=25, A=1, Ts=80, Tf=20 → Q = 25·1·60 = 1500 W
        let q = ThermalAnalysis::convective_heat_transfer(25.0, 1.0, 80.0, 20.0);
        assert!((q - 1500.0).abs() < 1e-9);
    }

    #[test]
    fn test_convective_heat_transfer_zero_diff() {
        let q = ThermalAnalysis::convective_heat_transfer(10.0, 2.0, 50.0, 50.0);
        assert_eq!(q, 0.0);
    }

    #[test]
    fn test_convective_heat_transfer_negative_when_cooler() {
        // Surface cooler than fluid → Q negative
        let q = ThermalAnalysis::convective_heat_transfer(10.0, 1.0, 10.0, 30.0);
        assert!(q < 0.0);
    }

    #[test]
    fn test_convective_proportional_to_h() {
        let q1 = ThermalAnalysis::convective_heat_transfer(5.0, 1.0, 100.0, 20.0);
        let q2 = ThermalAnalysis::convective_heat_transfer(10.0, 1.0, 100.0, 20.0);
        assert!((q2 - 2.0 * q1).abs() < 1e-10);
    }

    // ── radiative_heat_transfer ───────────────────────────────────────────────

    #[test]
    fn test_radiative_basic() {
        // ε=1, A=1, T_hot=1000K, T_cold=300K
        let q = ThermalAnalysis::radiative_heat_transfer(1.0, 1.0, 1000.0, 300.0);
        // Expected ≈ 5.67e-8 * (1e12 - 8.1e9) ≈ 56_129 W
        assert!(q > 50_000.0 && q < 60_000.0, "Q = {q}");
    }

    #[test]
    fn test_radiative_zero_emissivity() {
        let q = ThermalAnalysis::radiative_heat_transfer(0.0, 1.0, 1000.0, 300.0);
        assert_eq!(q, 0.0);
    }

    #[test]
    fn test_radiative_equal_temperatures() {
        let q = ThermalAnalysis::radiative_heat_transfer(0.9, 2.0, 400.0, 400.0);
        assert!(q.abs() < 1e-6);
    }

    #[test]
    fn test_radiative_hot_exceeds_cold() {
        let q = ThermalAnalysis::radiative_heat_transfer(0.8, 1.0, 600.0, 300.0);
        assert!(q > 0.0);
    }

    #[test]
    fn test_radiative_stefan_boltzmann_constant() {
        // Single-surface verification: ε=1, A=1, T_hot=T, T_cold=0K
        // Q = σ · T⁴
        let t = 500.0_f64;
        let q = ThermalAnalysis::radiative_heat_transfer(1.0, 1.0, t, 0.0);
        let expected = STEFAN_BOLTZMANN * t.powi(4);
        assert!((q - expected).abs() < 1e-4);
    }

    // ── thermal_resistance ────────────────────────────────────────────────────

    #[test]
    fn test_thermal_resistance_basic() {
        // R = 0.1 / (10 * 2) = 0.005 K/W
        let r = ThermalAnalysis::thermal_resistance(0.1, 10.0, 2.0);
        assert!((r - 0.005).abs() < 1e-12);
    }

    #[test]
    fn test_thermal_resistance_zero_conductivity() {
        let r = ThermalAnalysis::thermal_resistance(0.1, 0.0, 1.0);
        assert!(r.is_infinite());
    }

    #[test]
    fn test_thermal_resistance_zero_area() {
        let r = ThermalAnalysis::thermal_resistance(0.1, 10.0, 0.0);
        assert!(r.is_infinite());
    }

    // ── biot_number ───────────────────────────────────────────────────────────

    #[test]
    fn test_biot_number_basic() {
        // h=10, Lc=0.01, k=100 → Bi = 0.001
        let bi = ThermalAnalysis::biot_number(10.0, 0.01, 100.0);
        assert!((bi - 0.001).abs() < 1e-12);
    }

    #[test]
    fn test_biot_number_lumped_condition() {
        // Bi < 0.1 → lumped capacity valid
        let bi = ThermalAnalysis::biot_number(5.0, 0.01, 200.0);
        assert!(bi < 0.1);
    }

    #[test]
    fn test_biot_number_zero_k() {
        let bi = ThermalAnalysis::biot_number(10.0, 0.01, 0.0);
        assert!(bi.is_infinite());
    }

    // ── fourier_number ────────────────────────────────────────────────────────

    #[test]
    fn test_fourier_number_basic() {
        // α=1e-5, t=100, Lc=0.1 → Fo = 1e-5 * 100 / 0.01 = 0.1
        let fo = ThermalAnalysis::fourier_number(1e-5, 100.0, 0.1);
        assert!((fo - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_fourier_number_zero_lc() {
        let fo = ThermalAnalysis::fourier_number(1e-5, 100.0, 0.0);
        assert!(fo.is_infinite());
    }

    #[test]
    fn test_fourier_number_scales_with_time() {
        let fo1 = ThermalAnalysis::fourier_number(1e-5, 10.0, 0.1);
        let fo2 = ThermalAnalysis::fourier_number(1e-5, 20.0, 0.1);
        assert!((fo2 - 2.0 * fo1).abs() < 1e-14);
    }

    // ── transient_temperature ─────────────────────────────────────────────────

    #[test]
    fn test_transient_temperature_at_zero_time() {
        // At t=0 → Fo=0 → exp(0) = 1 → T = T0
        let t = ThermalAnalysis::transient_temperature(200.0, 20.0, 0.05, 0.0);
        assert!((t - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_transient_temperature_approaches_ambient() {
        // Very large Fo → temperature → T_inf
        let t = ThermalAnalysis::transient_temperature(200.0, 20.0, 0.05, 1000.0);
        assert!((t - 20.0).abs() < 1e-3);
    }

    #[test]
    fn test_transient_temperature_monotone_cooling() {
        let t_inf = 20.0;
        let t0 = 200.0;
        let bi = 0.05;
        let fo_values = [0.0, 1.0, 5.0, 10.0];
        let temps: Vec<f64> = fo_values
            .iter()
            .map(|&fo| ThermalAnalysis::transient_temperature(t0, t_inf, bi, fo))
            .collect();
        for w in temps.windows(2) {
            assert!(
                w[0] >= w[1],
                "Temperature should decrease: {} >= {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_transient_temperature_no_cooling_when_equal() {
        // If T0 == T_inf → T(t) = T_inf always
        let t = ThermalAnalysis::transient_temperature(50.0, 50.0, 0.1, 10.0);
        assert!((t - 50.0).abs() < 1e-10);
    }

    // ── steady_state_1d ───────────────────────────────────────────────────────

    #[test]
    fn test_steady_state_1d_length_zero() {
        let profile = ThermalAnalysis::steady_state_1d(0.0, 100.0, 1.0, 0);
        assert_eq!(profile, vec![0.0]);
    }

    #[test]
    fn test_steady_state_1d_two_nodes() {
        let profile = ThermalAnalysis::steady_state_1d(0.0, 100.0, 50.0, 1);
        assert_eq!(profile.len(), 2);
        assert!((profile[0] - 0.0).abs() < 1e-10);
        assert!((profile[1] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_steady_state_1d_midpoint() {
        let profile = ThermalAnalysis::steady_state_1d(0.0, 100.0, 1.0, 4);
        // index 2 of 5 → 50.0
        assert!((profile[2] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_steady_state_1d_boundary_values() {
        let profile = ThermalAnalysis::steady_state_1d(20.0, 80.0, 1.0, 10);
        assert!((profile[0] - 20.0).abs() < 1e-10);
        assert!((profile[10] - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_steady_state_1d_monotone() {
        let profile = ThermalAnalysis::steady_state_1d(300.0, 100.0, 1.0, 5);
        for w in profile.windows(2) {
            assert!(w[0] >= w[1], "{} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_steady_state_1d_node_count() {
        let n = 8_usize;
        let profile = ThermalAnalysis::steady_state_1d(0.0, 1.0, 1.0, n);
        assert_eq!(profile.len(), n + 1);
    }

    // ── heat_balance ──────────────────────────────────────────────────────────

    #[test]
    fn test_heat_balance_empty_system() {
        let sys = ThermalSystem::new();
        let bal = ThermalAnalysis::heat_balance(&sys);
        assert!(bal.is_empty());
    }

    #[test]
    fn test_heat_balance_single_node_source() {
        let mut sys = ThermalSystem::new();
        sys.add_node(ThermalNetNode::new("A", 300.0, 500.0));
        let bal = ThermalAnalysis::heat_balance(&sys);
        assert!((bal["A"] - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_heat_balance_two_nodes_equal_temp() {
        let mut sys = ThermalSystem::new();
        sys.add_node(ThermalNetNode::new("A", 300.0, 0.0));
        sys.add_node(ThermalNetNode::new("B", 300.0, 0.0));
        sys.add_edge(ThermalEdge::new("A", "B", 10.0));
        let bal = ThermalAnalysis::heat_balance(&sys);
        assert!(bal["A"].abs() < 1e-9);
        assert!(bal["B"].abs() < 1e-9);
    }

    #[test]
    fn test_heat_balance_two_nodes_unequal_temp() {
        // A=400K, B=300K, G=5 W/K → Q_A→B = 5*(300-400)= -500 W (A loses heat)
        let mut sys = ThermalSystem::new();
        sys.add_node(ThermalNetNode::new("A", 400.0, 0.0));
        sys.add_node(ThermalNetNode::new("B", 300.0, 0.0));
        sys.add_edge(ThermalEdge::new("A", "B", 5.0));
        let bal = ThermalAnalysis::heat_balance(&sys);
        assert!(bal["A"] < 0.0, "A should lose heat");
        assert!(bal["B"] > 0.0, "B should gain heat");
    }

    #[test]
    fn test_heat_balance_conservation() {
        // Net heat of entire isolated system should sum to zero
        let mut sys = ThermalSystem::new();
        sys.add_node(ThermalNetNode::new("A", 500.0, 0.0));
        sys.add_node(ThermalNetNode::new("B", 300.0, 0.0));
        sys.add_edge(ThermalEdge::new("A", "B", 8.0));
        let bal = ThermalAnalysis::heat_balance(&sys);
        let total: f64 = bal.values().sum();
        assert!(total.abs() < 1e-9, "total = {total}");
    }

    // ── series / parallel resistance ──────────────────────────────────────────

    #[test]
    fn test_series_resistance() {
        let r = ThermalAnalysis::series_resistance(2.0, 3.0);
        assert!((r - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_parallel_resistance() {
        // 1/(1/2 + 1/3) = 6/5
        let r = ThermalAnalysis::parallel_resistance(2.0, 3.0);
        assert!((r - 1.2).abs() < 1e-12);
    }

    #[test]
    fn test_parallel_resistance_zero() {
        let r = ThermalAnalysis::parallel_resistance(0.0, 0.0);
        assert_eq!(r, 0.0);
    }

    // ── overall_htc ───────────────────────────────────────────────────────────

    #[test]
    fn test_overall_htc_basic() {
        // R = 0.01 + 0.02 = 0.03, A = 2 → U = 1/(0.03*2) ≈ 16.67
        let u = ThermalAnalysis::overall_htc(&[0.01, 0.02], 2.0);
        assert!((u - 1.0 / 0.06).abs() < 1e-9);
    }

    #[test]
    fn test_overall_htc_zero_area() {
        let u = ThermalAnalysis::overall_htc(&[0.01], 0.0);
        assert!(u.is_infinite());
    }
}
