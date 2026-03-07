//! Conservation Laws, Physical Bounds, and Buckingham Pi Dimensional Analysis
//!
//! Provides:
//! - [`ConservationLaw`] trait + standard implementations
//!   (energy, momentum, mass, entropy).
//! - [`PhysicalBoundsValidator`] for range checks.
//! - [`BuckinghamPiAnalyzer`] implementing the Buckingham π theorem.

pub mod checkers;

pub use checkers::{
    ConservationReport, ConservationSuite, ConservationViolationDetail, EnergyConservationChecker,
    MassConservationChecker, MomentumConservationChecker, ViolationSeverity,
};

use crate::error::{PhysicsError, PhysicsResult};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// PhysState – thin wrapper around quantity map
// ─────────────────────────────────────────────────────────────────────────────

/// A snapshot of physical quantities at a single instant.
#[derive(Debug, Clone, Default)]
pub struct PhysState {
    /// Quantity name → value (SI).
    pub quantities: HashMap<String, f64>,
}

impl PhysState {
    /// Create an empty state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert / update a quantity.
    pub fn set(&mut self, name: impl Into<String>, value: f64) {
        self.quantities.insert(name.into(), value);
    }

    /// Get a quantity, returning `None` if absent.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.quantities.get(name).copied()
    }

    /// Get a quantity or return an error.
    pub fn require(&self, name: &str) -> PhysicsResult<f64> {
        self.quantities
            .get(name)
            .copied()
            .ok_or_else(|| PhysicsError::ConstraintViolation(format!("missing quantity: {name}")))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConservationLaw trait
// ─────────────────────────────────────────────────────────────────────────────

/// A conservation law that can be checked against a pair of states.
pub trait ConservationLaw: Send + Sync {
    /// Name of the law (for reporting).
    fn name(&self) -> &str;

    /// Check the law given the state *before* (`initial`) and *after* (`final`)
    /// a process.
    fn check(&self, initial: &PhysState, final_state: &PhysState) -> PhysicsResult<()>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Energy Conservation
// ─────────────────────────────────────────────────────────────────────────────

/// Checks that total mechanical energy is conserved within `tolerance`.
pub struct EnergyConservation {
    pub tolerance: f64,
}

impl EnergyConservation {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    fn total_energy(state: &PhysState) -> f64 {
        if let Some(e) = state.get("total_energy") {
            return e;
        }
        state.get("kinetic_energy").unwrap_or(0.0) + state.get("potential_energy").unwrap_or(0.0)
    }
}

impl ConservationLaw for EnergyConservation {
    fn name(&self) -> &str {
        "Energy Conservation"
    }

    fn check(&self, initial: &PhysState, final_state: &PhysState) -> PhysicsResult<()> {
        let e_i = Self::total_energy(initial);
        let e_f = Self::total_energy(final_state);
        if (e_f - e_i).abs() > self.tolerance {
            return Err(PhysicsError::ConservationViolation {
                law: self.name().to_string(),
                expected: e_i,
                actual: e_f,
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Momentum Conservation
// ─────────────────────────────────────────────────────────────────────────────

/// Checks conservation of linear and angular momentum.
pub struct MomentumConservation {
    pub tolerance: f64,
}

impl MomentumConservation {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }
}

impl ConservationLaw for MomentumConservation {
    fn name(&self) -> &str {
        "Momentum Conservation"
    }

    fn check(&self, initial: &PhysState, final_state: &PhysState) -> PhysicsResult<()> {
        for component in &["momentum_x", "momentum_y", "momentum_z"] {
            let p_i = initial.get(component).unwrap_or(0.0);
            let p_f = final_state.get(component).unwrap_or(0.0);
            if (p_f - p_i).abs() > self.tolerance {
                return Err(PhysicsError::ConservationViolation {
                    law: format!("{} ({})", self.name(), component),
                    expected: p_i,
                    actual: p_f,
                });
            }
        }
        let l_i = initial.get("angular_momentum").unwrap_or(0.0);
        let l_f = final_state.get("angular_momentum").unwrap_or(0.0);
        if (l_f - l_i).abs() > self.tolerance {
            return Err(PhysicsError::ConservationViolation {
                law: format!("{} (angular)", self.name()),
                expected: l_i,
                actual: l_f,
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mass Conservation
// ─────────────────────────────────────────────────────────────────────────────

/// Checks that total mass does not change.
pub struct MassConservation {
    pub tolerance: f64,
}

impl MassConservation {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }
}

impl ConservationLaw for MassConservation {
    fn name(&self) -> &str {
        "Mass Conservation"
    }

    fn check(&self, initial: &PhysState, final_state: &PhysState) -> PhysicsResult<()> {
        let m_i = initial
            .get("total_mass")
            .or_else(|| initial.get("mass"))
            .unwrap_or(0.0);
        let m_f = final_state
            .get("total_mass")
            .or_else(|| final_state.get("mass"))
            .unwrap_or(0.0);
        if (m_f - m_i).abs() > self.tolerance {
            return Err(PhysicsError::ConservationViolation {
                law: self.name().to_string(),
                expected: m_i,
                actual: m_f,
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entropy Law (2nd Law of Thermodynamics)
// ─────────────────────────────────────────────────────────────────────────────

/// Checks that entropy is non-decreasing: ΔS ≥ −`tolerance`.
pub struct EntropyLaw {
    pub tolerance: f64,
}

impl EntropyLaw {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }
}

impl ConservationLaw for EntropyLaw {
    fn name(&self) -> &str {
        "Second Law of Thermodynamics (Entropy Non-Decrease)"
    }

    fn check(&self, initial: &PhysState, final_state: &PhysState) -> PhysicsResult<()> {
        let s_i = initial.get("entropy").unwrap_or(0.0);
        let s_f = final_state.get("entropy").unwrap_or(0.0);
        if s_f - s_i < -self.tolerance {
            return Err(PhysicsError::ConservationViolation {
                law: self.name().to_string(),
                expected: s_i,
                actual: s_f,
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PhysicalBoundsValidator
// ─────────────────────────────────────────────────────────────────────────────

/// Describes a physical bound on a named quantity.
#[derive(Debug, Clone)]
pub struct PhysicalBound {
    pub quantity: String,
    pub min: f64,
    pub max: f64,
    pub description: String,
}

/// A bound violation.
#[derive(Debug, Clone)]
pub struct BoundViolation {
    pub quantity: String,
    pub actual: f64,
    pub bound: PhysicalBound,
    /// "below_minimum" | "above_maximum"
    pub violation_kind: String,
}

/// Validates a set of physical quantities against registered bounds.
#[derive(Debug, Default)]
pub struct PhysicalBoundsValidator {
    bounds: Vec<PhysicalBound>,
}

impl PhysicalBoundsValidator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_bound(
        &mut self,
        quantity: impl Into<String>,
        min: f64,
        max: f64,
        description: impl Into<String>,
    ) {
        self.bounds.push(PhysicalBound {
            quantity: quantity.into(),
            min,
            max,
            description: description.into(),
        });
    }

    pub fn validate(&self, state: &HashMap<String, f64>) -> PhysicsResult<Vec<BoundViolation>> {
        let violations = self
            .bounds
            .iter()
            .filter_map(|bound| {
                state.get(&bound.quantity).and_then(|&actual| {
                    if actual < bound.min {
                        Some(BoundViolation {
                            quantity: bound.quantity.clone(),
                            actual,
                            bound: bound.clone(),
                            violation_kind: "below_minimum".to_string(),
                        })
                    } else if actual > bound.max {
                        Some(BoundViolation {
                            quantity: bound.quantity.clone(),
                            actual,
                            bound: bound.clone(),
                            violation_kind: "above_maximum".to_string(),
                        })
                    } else {
                        None
                    }
                })
            })
            .collect();
        Ok(violations)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Buckingham Pi Theorem
// ─────────────────────────────────────────────────────────────────────────────

/// SI base dimensions (exponents).  Order: [M, L, T, I, Θ, N, J].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dimensions {
    pub mass: i8,
    pub length: i8,
    pub time: i8,
    pub current: i8,
    pub temperature: i8,
    pub amount: i8,
    pub luminosity: i8,
}

impl Dimensions {
    pub const DIMENSIONLESS: Self = Self {
        mass: 0,
        length: 0,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };

    fn as_array(self) -> [i8; 7] {
        [
            self.mass,
            self.length,
            self.time,
            self.current,
            self.temperature,
            self.amount,
            self.luminosity,
        ]
    }

    fn is_zero(self) -> bool {
        self.as_array().iter().all(|&x| x == 0)
    }
}

/// A physical quantity used as input to the Buckingham π analyzer.
#[derive(Debug, Clone)]
pub struct PhysicalQuantity {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub dimensions: Dimensions,
}

impl PhysicalQuantity {
    pub fn new(
        name: impl Into<String>,
        value: f64,
        unit: impl Into<String>,
        dimensions: Dimensions,
    ) -> Self {
        Self {
            name: name.into(),
            value,
            unit: unit.into(),
            dimensions,
        }
    }
}

/// A dimensionless π group produced by the Buckingham π theorem.
#[derive(Debug, Clone)]
pub struct DimensionlessPi {
    pub label: String,
    pub quantity_indices: Vec<usize>,
    pub exponents: Vec<f64>,
    pub value: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear algebra helpers (clippy-clean, no raw index loops where avoidable)
// ─────────────────────────────────────────────────────────────────────────────

/// Row-reduce `matrix` (DIM rows × ncols columns) in-place and return the rank.
fn row_echelon_rank(matrix: &mut [Vec<f64>], ncols: usize) -> usize {
    let nrows = matrix.len();
    let mut rank = 0_usize;
    let mut pivot_col = 0_usize;
    let mut row = 0_usize;

    while row < nrows && pivot_col < ncols {
        // Find pivot in this column at or below `row`.
        let pivot_row = (row..nrows).find(|&r| matrix[r][pivot_col].abs() > 1e-12);
        if let Some(pr) = pivot_row {
            matrix.swap(row, pr);
            rank += 1;
            let pivot_val = matrix[row][pivot_col];
            // Normalise pivot row.
            matrix[row].iter_mut().for_each(|v| *v /= pivot_val);
            // Eliminate column entries in all other rows.
            for r in 0..nrows {
                if r != row {
                    let factor = matrix[r][pivot_col];
                    // borrow issue: collect target row first then update other row
                    let pivot_row_copy = matrix[row].clone();
                    matrix[r]
                        .iter_mut()
                        .zip(pivot_row_copy.iter())
                        .for_each(|(cell, &p)| *cell -= factor * p);
                }
            }
            row += 1;
        }
        pivot_col += 1;
    }
    rank
}

/// Return the matrix rank.
fn rank_of(matrix: &[Vec<f64>], ncols: usize) -> usize {
    let mut m = matrix.to_vec();
    row_echelon_rank(&mut m, ncols)
}

/// Return the pivot column indices after row reduction.
fn find_pivot_columns(matrix: &[Vec<f64>], ncols: usize, rank: usize) -> Vec<usize> {
    let nrows = matrix.len();
    let mut m = matrix.to_vec();
    let mut pivots = Vec::new();
    let mut pivot_col = 0_usize;
    let mut row = 0_usize;

    while row < nrows && pivot_col < ncols && pivots.len() < rank {
        let pivot_row = (row..nrows).find(|&r| m[r][pivot_col].abs() > 1e-12);
        if let Some(pr) = pivot_row {
            m.swap(row, pr);
            pivots.push(pivot_col);
            let pv = m[row][pivot_col];
            m[row].iter_mut().for_each(|v| *v /= pv);
            for r in 0..nrows {
                if r != row {
                    let f = m[r][pivot_col];
                    let pivot_copy = m[row].clone();
                    m[r].iter_mut()
                        .zip(pivot_copy.iter())
                        .for_each(|(cell, &p)| *cell -= f * p);
                }
            }
            row += 1;
        }
        pivot_col += 1;
    }
    pivots
}

fn build_pi_group(
    quantities: &[PhysicalQuantity],
    pivot_cols: &[usize],
    rem_idx: usize,
    dim_matrix: &[Vec<f64>],
    group_num: usize,
) -> DimensionlessPi {
    const DIM: usize = 7;
    let r = pivot_cols.len();

    // Build A (DIM × r) and b (DIM).
    let a: Vec<Vec<f64>> = (0..DIM)
        .map(|d| pivot_cols.iter().map(|&pc| dim_matrix[d][pc]).collect())
        .collect();
    let b: Vec<f64> = (0..DIM).map(|d| -dim_matrix[d][rem_idx]).collect();

    // Normal equations: (A^T A) x = A^T b.
    let mut ata: Vec<Vec<f64>> = vec![vec![0.0; r]; r];
    let mut atb: Vec<f64> = vec![0.0; r];
    for i in 0..r {
        for j in 0..r {
            ata[i][j] = (0..DIM).map(|d| a[d][i] * a[d][j]).sum();
        }
        atb[i] = (0..DIM).map(|d| a[d][i] * b[d]).sum();
    }

    let x = solve_linear_system(&ata, &atb);

    let mut indices = pivot_cols.to_vec();
    indices.push(rem_idx);
    let mut exponents = x;
    exponents.push(1.0);

    let value = indices
        .iter()
        .zip(exponents.iter())
        .map(|(&idx, &exp)| quantities[idx].value.powf(exp))
        .product::<f64>();

    DimensionlessPi {
        label: format!("π{group_num}"),
        quantity_indices: indices,
        exponents,
        value,
    }
}

/// Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return Vec::new();
    }

    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut rhs: Vec<f64> = b.to_vec();

    for col in 0..n {
        // Partial pivot.
        if let Some(max_row) =
            (col..n).max_by(|&i, &j| m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap())
        {
            m.swap(col, max_row);
            rhs.swap(col, max_row);
        }
        let pv = m[col][col];
        if pv.abs() < 1e-14 {
            continue;
        }
        m[col].iter_mut().for_each(|v| *v /= pv);
        rhs[col] /= pv;

        for row in (col + 1)..n {
            let f = m[row][col];
            let pivot_copy = m[col].clone();
            m[row]
                .iter_mut()
                .zip(pivot_copy.iter())
                .for_each(|(cell, &p)| *cell -= f * p);
            rhs[row] -= f * rhs[col];
        }
    }

    // Back substitution.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        x[i] = rhs[i] - ((i + 1)..n).map(|j| m[i][j] * x[j]).sum::<f64>();
    }
    x
}

fn is_dimensionless(quantities: &[PhysicalQuantity], pi: &DimensionlessPi) -> bool {
    let mut total = Dimensions::DIMENSIONLESS;
    for (&idx, &exp) in pi.quantity_indices.iter().zip(pi.exponents.iter()) {
        let d = quantities[idx].dimensions;
        let rounded = exp.round() as i8;
        total.mass += d.mass * rounded;
        total.length += d.length * rounded;
        total.time += d.time * rounded;
        total.current += d.current * rounded;
        total.temperature += d.temperature * rounded;
        total.amount += d.amount * rounded;
        total.luminosity += d.luminosity * rounded;
    }
    total.is_zero()
}

/// Analyzes a set of physical quantities and returns dimensionless π groups
/// using the Buckingham π theorem.
pub struct BuckinghamPiAnalyzer;

impl BuckinghamPiAnalyzer {
    pub fn analyze(quantities: &[PhysicalQuantity]) -> PhysicsResult<Vec<DimensionlessPi>> {
        if quantities.is_empty() {
            return Err(PhysicsError::ConstraintViolation(
                "no quantities provided for Buckingham Pi analysis".to_string(),
            ));
        }

        let n = quantities.len();
        const DIM: usize = 7;

        // Build dimensional matrix: DIM rows × n columns.
        let dim_matrix: Vec<Vec<f64>> = (0..DIM)
            .map(|i| {
                quantities
                    .iter()
                    .map(|q| q.dimensions.as_array()[i] as f64)
                    .collect()
            })
            .collect();

        let rank = rank_of(&dim_matrix, n);
        let num_pi = n.saturating_sub(rank);

        if num_pi == 0 {
            return Ok(Vec::new());
        }

        let pivot_cols = find_pivot_columns(&dim_matrix, n, rank);
        let remaining: Vec<usize> = (0..n).filter(|c| !pivot_cols.contains(c)).collect();

        let pis: Vec<DimensionlessPi> = remaining
            .iter()
            .enumerate()
            .map(|(k, &rem_idx)| {
                build_pi_group(quantities, &pivot_cols, rem_idx, &dim_matrix, k + 1)
            })
            .collect();

        for pi in &pis {
            if !is_dimensionless(quantities, pi) {
                return Err(PhysicsError::ConstraintViolation(
                    "Buckingham Pi group is not dimensionless (numerical error)".to_string(),
                ));
            }
        }

        Ok(pis)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with(pairs: &[(&str, f64)]) -> PhysState {
        let mut s = PhysState::new();
        for &(k, v) in pairs {
            s.set(k, v);
        }
        s
    }

    // ── Energy Conservation ───────────────────────────────────────────────────

    #[test]
    fn energy_conservation_ok() {
        let law = EnergyConservation::new(1e-6);
        let s0 = state_with(&[("kinetic_energy", 100.0), ("potential_energy", 50.0)]);
        let s1 = state_with(&[("kinetic_energy", 140.0), ("potential_energy", 10.0)]);
        assert!(law.check(&s0, &s1).is_ok());
    }

    #[test]
    fn energy_conservation_violated() {
        let law = EnergyConservation::new(1.0);
        let s0 = state_with(&[("total_energy", 100.0)]);
        let s1 = state_with(&[("total_energy", 200.0)]);
        assert!(law.check(&s0, &s1).is_err());
    }

    // ── Momentum Conservation ─────────────────────────────────────────────────

    #[test]
    fn momentum_conservation_ok() {
        let law = MomentumConservation::new(1e-6);
        let s0 = state_with(&[
            ("momentum_x", 10.0),
            ("momentum_y", 5.0),
            ("momentum_z", 0.0),
        ]);
        let s1 = s0.clone();
        assert!(law.check(&s0, &s1).is_ok());
    }

    #[test]
    fn momentum_conservation_violated() {
        let law = MomentumConservation::new(0.01);
        let s0 = state_with(&[("momentum_x", 10.0)]);
        let s1 = state_with(&[("momentum_x", 15.0)]);
        assert!(law.check(&s0, &s1).is_err());
    }

    // ── Mass Conservation ─────────────────────────────────────────────────────

    #[test]
    fn mass_conservation_ok() {
        let law = MassConservation::new(1e-9);
        let s0 = state_with(&[("total_mass", 1.0)]);
        let s1 = state_with(&[("total_mass", 1.0 + 1e-12)]);
        assert!(law.check(&s0, &s1).is_ok());
    }

    #[test]
    fn mass_conservation_violated() {
        let law = MassConservation::new(1e-6);
        let s0 = state_with(&[("mass", 2.0)]);
        let s1 = state_with(&[("mass", 3.0)]);
        assert!(law.check(&s0, &s1).is_err());
    }

    // ── Entropy Law ───────────────────────────────────────────────────────────

    #[test]
    fn entropy_non_decreasing_ok() {
        let law = EntropyLaw::new(1e-9);
        let s0 = state_with(&[("entropy", 100.0)]);
        let s1 = state_with(&[("entropy", 105.0)]);
        assert!(law.check(&s0, &s1).is_ok());
    }

    #[test]
    fn entropy_decrease_violates() {
        let law = EntropyLaw::new(1e-9);
        let s0 = state_with(&[("entropy", 100.0)]);
        let s1 = state_with(&[("entropy", 95.0)]);
        assert!(law.check(&s0, &s1).is_err());
    }

    // ── PhysicalBoundsValidator ───────────────────────────────────────────────

    #[test]
    fn bounds_no_violation() {
        let mut v = PhysicalBoundsValidator::new();
        v.add_bound("temperature", 0.0, 1000.0, "Temperature in K");
        v.add_bound("pressure", 0.0, 1e8, "Pressure in Pa");

        let state: HashMap<String, f64> = [
            ("temperature".to_string(), 300.0),
            ("pressure".to_string(), 1e5),
        ]
        .into_iter()
        .collect();

        let violations = v.validate(&state).unwrap();
        assert!(violations.is_empty());
    }

    #[test]
    fn bounds_below_minimum() {
        let mut v = PhysicalBoundsValidator::new();
        v.add_bound("temperature", 200.0, 1000.0, "Temperature must be >= 200 K");

        let state: HashMap<String, f64> = [("temperature".to_string(), 50.0)].into_iter().collect();

        let violations = v.validate(&state).unwrap();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].violation_kind, "below_minimum");
    }

    #[test]
    fn bounds_above_maximum() {
        let mut v = PhysicalBoundsValidator::new();
        v.add_bound("speed", 0.0, 300.0, "Speed <= 300 m/s");

        let state: HashMap<String, f64> = [("speed".to_string(), 400.0)].into_iter().collect();

        let violations = v.validate(&state).unwrap();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].violation_kind, "above_maximum");
    }

    // ── Buckingham Pi ─────────────────────────────────────────────────────────

    fn dim(mass: i8, length: i8, time: i8) -> Dimensions {
        Dimensions {
            mass,
            length,
            time,
            current: 0,
            temperature: 0,
            amount: 0,
            luminosity: 0,
        }
    }

    #[test]
    fn buckingham_pi_pendulum() {
        // n=3, rank=2 → 1 π group.
        let quantities = vec![
            PhysicalQuantity::new("T_period", 2.0, "s", dim(0, 0, 1)),
            PhysicalQuantity::new("L_length", 1.0, "m", dim(0, 1, 0)),
            PhysicalQuantity::new("g_accel", 9.81, "m/s^2", dim(0, 1, -2)),
        ];

        let pis = BuckinghamPiAnalyzer::analyze(&quantities).unwrap();
        assert_eq!(pis.len(), 1, "expected 1 dimensionless group for pendulum");
    }

    #[test]
    fn buckingham_pi_empty_input_is_error() {
        let result = BuckinghamPiAnalyzer::analyze(&[]);
        assert!(result.is_err());
    }

    // ── Additional PhysState tests ────────────────────────────────────────────

    #[test]
    fn test_physstate_set_and_get() {
        let mut s = PhysState::new();
        s.set("velocity_x", 3.0);
        s.set("velocity_y", 4.0);
        assert_eq!(s.get("velocity_x"), Some(3.0));
        assert_eq!(s.get("velocity_y"), Some(4.0));
        assert_eq!(s.get("velocity_z"), None);
    }

    #[test]
    fn test_physstate_require_missing_is_error() {
        let s = PhysState::new();
        let result = s.require("mass");
        assert!(result.is_err(), "require on missing key should error");
    }

    #[test]
    fn test_physstate_require_present_ok() {
        let mut s = PhysState::new();
        s.set("pressure", 101325.0);
        let v = s.require("pressure").unwrap();
        assert!((v - 101325.0).abs() < 1e-6);
    }

    #[test]
    fn test_physstate_overwrite() {
        let mut s = PhysState::new();
        s.set("temp", 300.0);
        s.set("temp", 350.0);
        assert_eq!(s.get("temp"), Some(350.0));
    }

    // ── EnergyConservation (trait impl) tests ─────────────────────────────────

    #[test]
    fn energy_law_pass_total_energy() {
        let law = EnergyConservation::new(1.0);
        let s0 = state_with(&[("total_energy", 500.0)]);
        let s1 = state_with(&[("total_energy", 500.5)]);
        assert!(law.check(&s0, &s1).is_ok());
    }

    #[test]
    fn energy_law_fail_large_change() {
        let law = EnergyConservation::new(0.1);
        let s0 = state_with(&[("total_energy", 1000.0)]);
        let s1 = state_with(&[("total_energy", 2000.0)]);
        assert!(law.check(&s0, &s1).is_err());
    }

    #[test]
    fn energy_law_name_correct() {
        let law = EnergyConservation::new(1.0);
        assert_eq!(law.name(), "Energy Conservation");
    }

    // ── MomentumConservation (trait impl) tests ───────────────────────────────

    #[test]
    fn momentum_law_pass() {
        let law = MomentumConservation::new(0.01);
        let s0 = state_with(&[
            ("momentum_x", 5.0),
            ("momentum_y", 0.0),
            ("momentum_z", 0.0),
        ]);
        let s1 = state_with(&[
            ("momentum_x", 5.0),
            ("momentum_y", 0.0),
            ("momentum_z", 0.0),
        ]);
        assert!(law.check(&s0, &s1).is_ok());
    }

    #[test]
    fn momentum_law_fail() {
        let law = MomentumConservation::new(0.001);
        let s0 = state_with(&[("momentum_x", 10.0)]);
        let s1 = state_with(&[("momentum_x", 15.0)]);
        assert!(law.check(&s0, &s1).is_err());
    }

    #[test]
    fn momentum_law_name_correct() {
        let law = MomentumConservation::new(0.1);
        assert_eq!(law.name(), "Momentum Conservation");
    }

    // ── MassConservation (trait impl) tests ──────────────────────────────────

    #[test]
    fn mass_law_pass() {
        let law = MassConservation::new(1e-6);
        let s0 = state_with(&[("mass", 5.0)]);
        let s1 = state_with(&[("mass", 5.0 + 1e-8)]);
        assert!(law.check(&s0, &s1).is_ok());
    }

    #[test]
    fn mass_law_fail() {
        let law = MassConservation::new(1e-6);
        let s0 = state_with(&[("mass", 1.0)]);
        let s1 = state_with(&[("mass", 3.0)]);
        assert!(law.check(&s0, &s1).is_err());
    }

    #[test]
    fn mass_law_name_correct() {
        let law = MassConservation::new(1e-6);
        assert_eq!(law.name(), "Mass Conservation");
    }

    // ── PhysicalBoundsValidator tests ─────────────────────────────────────────

    #[test]
    fn bounds_multiple_violations() {
        let mut v = PhysicalBoundsValidator::new();
        v.add_bound("temperature", 200.0, 1000.0, "K");
        v.add_bound("pressure", 0.0, 1e6, "Pa");
        v.add_bound("speed", 0.0, 300.0, "m/s");

        let state: HashMap<String, f64> = [
            ("temperature".to_string(), 50.0), // below min
            ("pressure".to_string(), 2e6),     // above max
            ("speed".to_string(), 100.0),      // ok
        ]
        .into_iter()
        .collect();

        let violations = v.validate(&state).unwrap();
        assert_eq!(violations.len(), 2, "expected 2 violations");
    }

    #[test]
    fn bounds_empty_state_no_violations() {
        let mut v = PhysicalBoundsValidator::new();
        v.add_bound("temperature", 0.0, 1000.0, "K");
        let state: HashMap<String, f64> = HashMap::new();
        let violations = v.validate(&state).unwrap();
        // Missing quantities are not violations (no value present)
        assert!(
            violations.is_empty(),
            "missing quantities should not trigger violations"
        );
    }

    // ── Buckingham Pi tests ───────────────────────────────────────────────────

    #[test]
    fn buckingham_pi_reynolds_number() {
        // Re = ρ*v*L/μ — 4 quantities, rank = 3 → 1 pi group
        fn dim(mass: i8, length: i8, time: i8) -> Dimensions {
            Dimensions {
                mass,
                length,
                time,
                current: 0,
                temperature: 0,
                amount: 0,
                luminosity: 0,
            }
        }
        let quantities = vec![
            PhysicalQuantity::new("rho", 1000.0, "kg/m3", dim(1, -3, 0)),
            PhysicalQuantity::new("v", 1.0, "m/s", dim(0, 1, -1)),
            PhysicalQuantity::new("L", 0.1, "m", dim(0, 1, 0)),
            PhysicalQuantity::new("mu", 1e-3, "Pa-s", dim(1, -1, -1)),
        ];
        let pis = BuckinghamPiAnalyzer::analyze(&quantities).unwrap();
        assert_eq!(pis.len(), 1, "Reynolds number → 1 pi group");
    }

    #[test]
    fn buckingham_pi_two_groups() {
        // 5 quantities, rank = 3 → 2 pi groups
        fn dim(mass: i8, length: i8, time: i8) -> Dimensions {
            Dimensions {
                mass,
                length,
                time,
                current: 0,
                temperature: 0,
                amount: 0,
                luminosity: 0,
            }
        }
        let quantities = vec![
            PhysicalQuantity::new("rho", 1000.0, "kg/m3", dim(1, -3, 0)),
            PhysicalQuantity::new("v", 1.0, "m/s", dim(0, 1, -1)),
            PhysicalQuantity::new("L", 0.1, "m", dim(0, 1, 0)),
            PhysicalQuantity::new("mu", 1e-3, "Pa-s", dim(1, -1, -1)),
            PhysicalQuantity::new("dp", 100.0, "Pa", dim(1, -1, -2)), // pressure gradient
        ];
        let pis = BuckinghamPiAnalyzer::analyze(&quantities).unwrap();
        assert_eq!(pis.len(), 2, "5 quantities - rank 3 = 2 pi groups");
    }
}
