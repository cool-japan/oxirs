//! Buckingham π theorem — dimensional analysis for similarity parameters.
//!
//! Given a set of physically meaningful variables, the theorem states that any
//! dimensionally consistent equation relating n variables built from k
//! independent base dimensions can be rewritten in terms of n − k
//! dimensionless groups (π groups).
//!
//! This module provides:
//! - [`BaseUnit`] — the seven SI base dimensions.
//! - [`PhysicalVar`] — a named variable with its dimensional exponents.
//! - [`PiGroup`] — a dimensionless combination of the input variables.
//! - [`BuckinghamPi`] — runs the algorithm via Gaussian elimination over ℚ.
//! - [`BuckinghamPiError`] — failure modes.
//!
//! # Example
//!
//! ```
//! use oxirs_physics::constraints::buckingham_pi::{BuckinghamPi, BaseUnit, PhysicalVar};
//!
//! let mut length = PhysicalVar::new("L");
//! length.set_dimension(BaseUnit::Length, 1);
//! let mut time = PhysicalVar::new("T");
//! time.set_dimension(BaseUnit::Time, 1);
//! let mut velocity = PhysicalVar::new("v");
//! velocity.set_dimension(BaseUnit::Length, 1);
//! velocity.set_dimension(BaseUnit::Time, -1);
//!
//! // v = L / T → the single π group is v·T/L (dimensionless).
//! let pi_groups = BuckinghamPi::analyze(&[length, time, velocity]).unwrap();
//! assert_eq!(pi_groups.len(), 1);
//! ```

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// BaseUnit
// ──────────────────────────────────────────────────────────────────────────────

/// The seven SI base dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseUnit {
    /// Length — metre (m).
    Length,
    /// Mass — kilogram (kg).
    Mass,
    /// Time — second (s).
    Time,
    /// Electric current — ampere (A).
    Current,
    /// Thermodynamic temperature — kelvin (K).
    Temperature,
    /// Amount of substance — mole (mol).
    Amount,
    /// Luminous intensity — candela (cd).
    LuminousIntensity,
}

impl BaseUnit {
    /// All seven base units in a fixed canonical order.
    pub const ALL: [BaseUnit; 7] = [
        BaseUnit::Length,
        BaseUnit::Mass,
        BaseUnit::Time,
        BaseUnit::Current,
        BaseUnit::Temperature,
        BaseUnit::Amount,
        BaseUnit::LuminousIntensity,
    ];
}

// ──────────────────────────────────────────────────────────────────────────────
// PhysicalVar
// ──────────────────────────────────────────────────────────────────────────────

/// A physical variable with its SI dimensional exponents.
///
/// Only dimensions with non-zero exponents need to be set.
#[derive(Debug, Clone)]
pub struct PhysicalVar {
    /// Human-readable name (symbol or description).
    pub name: String,
    /// Dimensional exponents: `{BaseUnit → exponent}`.
    pub dimensions: HashMap<BaseUnit, i32>,
}

impl PhysicalVar {
    /// Create a dimensionless variable (all exponents zero).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dimensions: HashMap::new(),
        }
    }

    /// Set the exponent for one SI base dimension.
    pub fn set_dimension(&mut self, unit: BaseUnit, exponent: i32) {
        if exponent == 0 {
            self.dimensions.remove(&unit);
        } else {
            self.dimensions.insert(unit, exponent);
        }
    }

    /// Return the exponent for a base unit (0 if absent).
    pub fn exponent(&self, unit: BaseUnit) -> i32 {
        self.dimensions.get(&unit).copied().unwrap_or(0)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PiGroup
// ──────────────────────────────────────────────────────────────────────────────

/// A dimensionless π group expressed as a product of powers of the input
/// variables: `π = V₀^a₀ · V₁^a₁ · … · Vₙ^aₙ`.
///
/// Exponents are rational numbers represented as `(numerator, denominator)` in
/// lowest terms.
#[derive(Debug, Clone)]
pub struct PiGroup {
    /// Name assigned to the group (e.g. `"π1"`, `"Re"`, …).
    pub name: String,
    /// Variable-name → rational exponent (as `(numerator, denominator)`).
    pub exponents: Vec<(String, (i64, i64))>,
}

impl PiGroup {
    /// Return the exponent for `var_name` as an `f64`, or 0.0 if absent.
    pub fn exponent_f64(&self, var_name: &str) -> f64 {
        self.exponents
            .iter()
            .find(|(n, _)| n == var_name)
            .map(|(_, (num, den))| *num as f64 / *den as f64)
            .unwrap_or(0.0)
    }

    /// Human-readable representation, e.g. `"π1 = v^1 * L^(-1) * T^1"`.
    pub fn display(&self) -> String {
        let terms: Vec<String> = self
            .exponents
            .iter()
            .filter(|(_, (num, _))| *num != 0)
            .map(|(name, (num, den))| {
                if *den == 1 {
                    format!("{name}^{num}")
                } else {
                    format!("{name}^({num}/{den})")
                }
            })
            .collect();
        if terms.is_empty() {
            format!("{} = 1 (trivially dimensionless)", self.name)
        } else {
            format!("{} = {}", self.name, terms.join(" · "))
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`BuckinghamPi::analyze`].
#[derive(Debug, Clone, PartialEq)]
pub enum BuckinghamPiError {
    /// Fewer than two variables supplied.
    TooFewVariables,
    /// All variables are dimensionless — no analysis needed.
    AllDimensionless,
    /// n ≤ k: no dimensionless groups exist (more dimensions than variables).
    NoPiGroupsPossible {
        /// Number of variables.
        n: usize,
        /// Rank of dimensional matrix (independent base dimensions).
        k: usize,
    },
}

impl std::fmt::Display for BuckinghamPiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuckinghamPiError::TooFewVariables => {
                write!(f, "at least 2 variables are required")
            }
            BuckinghamPiError::AllDimensionless => {
                write!(f, "all variables are already dimensionless")
            }
            BuckinghamPiError::NoPiGroupsPossible { n, k } => {
                write!(
                    f,
                    "no π groups possible: n={n} variables, k={k} independent dimensions (n ≤ k)"
                )
            }
        }
    }
}

impl std::error::Error for BuckinghamPiError {}

// ──────────────────────────────────────────────────────────────────────────────
// BuckinghamPi analyser
// ──────────────────────────────────────────────────────────────────────────────

/// Implements the Buckingham π theorem via Gaussian elimination over ℚ.
///
/// The algorithm:
/// 1. Build a `k × n` dimensional matrix `A` where rows are SI base dimensions
///    and columns are variables.
/// 2. Compute the rank `r` of `A` (= number of independent dimensions used).
/// 3. Produce `n − r` dimensionless π groups from the null space of `A`.
pub struct BuckinghamPi;

impl BuckinghamPi {
    /// Analyse a list of physical variables and return the π groups.
    ///
    /// # Errors
    ///
    /// Returns a [`BuckinghamPiError`] if:
    /// - fewer than 2 variables are given,
    /// - all variables are dimensionless, or
    /// - no π groups exist (n ≤ k).
    pub fn analyze(variables: &[PhysicalVar]) -> Result<Vec<PiGroup>, BuckinghamPiError> {
        let n = variables.len();
        if n < 2 {
            return Err(BuckinghamPiError::TooFewVariables);
        }

        // Determine which base dimensions are actually used.
        let active_dims: Vec<BaseUnit> = BaseUnit::ALL
            .iter()
            .copied()
            .filter(|&dim| variables.iter().any(|v| v.exponent(dim) != 0))
            .collect();

        if active_dims.is_empty() {
            return Err(BuckinghamPiError::AllDimensionless);
        }

        let m = active_dims.len(); // number of rows (dimensions)

        // Build dimensional matrix as rational numbers (num, den).
        // Row i = dimension active_dims[i], column j = variable j.
        // Stored row-major: matrix[i][j] = (numerator, denominator).
        let matrix: Vec<Vec<(i64, i64)>> = (0..m)
            .map(|i| {
                (0..n)
                    .map(|j| (variables[j].exponent(active_dims[i]) as i64, 1i64))
                    .collect()
            })
            .collect();

        // Augment with identity block for the column space tracking.
        // We need to track transformations to find the null space.
        // Strategy: work on the full m×(n+m) augmented system.
        // Column 0..n are the variable columns; columns n..n+m are the identity.
        let mut aug: Vec<Vec<(i64, i64)>> = (0..m)
            .map(|i| {
                let mut row: Vec<(i64, i64)> = matrix[i].clone();
                for k in 0..m {
                    row.push(if k == i { (1, 1) } else { (0, 1) });
                }
                row
            })
            .collect();

        // ── Gaussian elimination (row echelon form) ────────────────────────

        let mut pivot_cols: Vec<usize> = Vec::new(); // pivot column indices
        let mut row = 0_usize;

        for col in 0..n {
            // Find a non-zero entry in column `col` at or below `row`.
            let pivot = (row..m).find(|&r| aug[r][col].0 != 0);
            if let Some(p) = pivot {
                // Swap rows `row` and `p`.
                aug.swap(row, p);
                pivot_cols.push(col);

                // Scale the pivot row so that aug[row][col] = 1.
                let pivot_val = aug[row][col];
                for entry in &mut aug[row] {
                    *entry = rat_div(*entry, pivot_val);
                }

                // Eliminate column `col` in all other rows.
                // Clone the normalized pivot row to allow independent mutable access.
                let pivot_row = aug[row].clone();
                for (r, aug_row) in aug.iter_mut().enumerate() {
                    if r != row && aug_row[col].0 != 0 {
                        let factor = aug_row[col];
                        for (entry, &pv) in aug_row.iter_mut().zip(pivot_row.iter()) {
                            let sub = rat_mul(factor, pv);
                            *entry = rat_sub(*entry, sub);
                        }
                    }
                }
                row += 1;
            }
        }

        let rank = pivot_cols.len(); // = r
        let pi_count = n - rank;

        if pi_count == 0 {
            return Err(BuckinghamPiError::NoPiGroupsPossible { n, k: rank });
        }

        // ── Identify free (non-pivot) column indices ────────────────────────
        let pivot_set: std::collections::HashSet<usize> = pivot_cols.iter().copied().collect();
        let free_cols: Vec<usize> = (0..n).filter(|c| !pivot_set.contains(c)).collect();

        // ── Build null-space vectors (one per free variable) ────────────────
        //
        // For each free column f, set the exponent of variable f to 1 and the
        // exponents of the other free variables to 0.  The exponents of the
        // pivot variables are read from the reduced row-echelon form.
        let mut pi_groups: Vec<PiGroup> = Vec::with_capacity(pi_count);

        for (pi_idx, &free_col) in free_cols.iter().enumerate() {
            let mut exponents: Vec<(String, (i64, i64))> = Vec::with_capacity(n);

            // Pivot variable exponents from RREF.
            for (piv_row, &piv_col) in pivot_cols.iter().enumerate() {
                // Exponent = -A[piv_row][free_col] (negated because we moved
                // the free-variable term to the right-hand side).
                let coeff = rat_neg(aug[piv_row][free_col]);
                let red = rat_reduce(coeff);
                exponents.push((variables[piv_col].name.clone(), red));
            }

            // Free variable exponents: 1 for this free column, 0 for others.
            for &fc in &free_cols {
                let exp = if fc == free_col { (1, 1) } else { (0, 1) };
                exponents.push((variables[fc].name.clone(), exp));
            }

            // Remove zero entries for clarity.
            exponents.retain(|(_, (num, _))| *num != 0);

            pi_groups.push(PiGroup {
                name: format!("π{}", pi_idx + 1),
                exponents,
            });
        }

        let _ = matrix; // suppress warning — used above for construction
        Ok(pi_groups)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Rational arithmetic helpers
// ──────────────────────────────────────────────────────────────────────────────

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        a %= b;
        std::mem::swap(&mut a, &mut b);
    }
    a.max(1)
}

fn rat_reduce((num, den): (i64, i64)) -> (i64, i64) {
    if num == 0 {
        return (0, 1);
    }
    let sign = if (num < 0) ^ (den < 0) { -1 } else { 1 };
    let g = gcd(num.abs(), den.abs());
    (sign * num.abs() / g, den.abs() / g)
}

fn rat_neg((num, den): (i64, i64)) -> (i64, i64) {
    (-num, den)
}

fn rat_mul((an, ad): (i64, i64), (bn, bd): (i64, i64)) -> (i64, i64) {
    rat_reduce((an * bn, ad * bd))
}

fn rat_div((an, ad): (i64, i64), (bn, bd): (i64, i64)) -> (i64, i64) {
    rat_reduce((an * bd, ad * bn))
}

fn rat_sub((an, ad): (i64, i64), (bn, bd): (i64, i64)) -> (i64, i64) {
    rat_reduce((an * bd - bn * ad, ad * bd))
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str, dims: &[(BaseUnit, i32)]) -> PhysicalVar {
        let mut v = PhysicalVar::new(name);
        for &(unit, exp) in dims {
            v.set_dimension(unit, exp);
        }
        v
    }

    // ── pendulum ─────────────────────────────────────────────────────────────

    /// Simple pendulum: variables are L (length), g (acceleration), T (period).
    /// n = 3, k = 2 (Length, Time) → 1 π group.
    /// Expected: T·√(g/L) = dimensionless → π1 = T * g^(1/2) * L^(-1/2).
    #[test]
    fn test_pendulum_one_pi_group() {
        let length = var("L", &[(BaseUnit::Length, 1)]);
        let gravity = var("g", &[(BaseUnit::Length, 1), (BaseUnit::Time, -2)]);
        let period = var("T", &[(BaseUnit::Time, 1)]);

        let groups = BuckinghamPi::analyze(&[length, gravity, period]).unwrap();
        assert_eq!(groups.len(), 1, "Pendulum should produce exactly 1 π group");
    }

    // ── Reynolds number ───────────────────────────────────────────────────────

    /// Reynolds number: ρ·v·L / μ.
    /// Variables: ρ (density: M·L⁻³), v (velocity: L·T⁻¹), L (length: L),
    ///            μ (dynamic viscosity: M·L⁻¹·T⁻¹).
    /// n = 4, k = 3 (M, L, T) → 1 π group (Re).
    #[test]
    fn test_reynolds_number_one_pi_group() {
        let rho = var("rho", &[(BaseUnit::Mass, 1), (BaseUnit::Length, -3)]);
        let v = var("v", &[(BaseUnit::Length, 1), (BaseUnit::Time, -1)]);
        let l = var("L", &[(BaseUnit::Length, 1)]);
        let mu = var(
            "mu",
            &[
                (BaseUnit::Mass, 1),
                (BaseUnit::Length, -1),
                (BaseUnit::Time, -1),
            ],
        );

        let groups = BuckinghamPi::analyze(&[rho, v, l, mu]).unwrap();
        assert_eq!(groups.len(), 1, "Reynolds number: 1 π group expected");
    }

    // ── two pi groups ─────────────────────────────────────────────────────────

    /// Drag force: F (M·L·T⁻²), ρ (M·L⁻³), v (L·T⁻¹), L (L), μ (M·L⁻¹·T⁻¹).
    /// n = 5, k = 3 (M, L, T) → 2 π groups (drag coefficient and Re).
    #[test]
    fn test_drag_two_pi_groups() {
        let f = var(
            "F",
            &[
                (BaseUnit::Mass, 1),
                (BaseUnit::Length, 1),
                (BaseUnit::Time, -2),
            ],
        );
        let rho = var("rho", &[(BaseUnit::Mass, 1), (BaseUnit::Length, -3)]);
        let v = var("v", &[(BaseUnit::Length, 1), (BaseUnit::Time, -1)]);
        let l = var("L", &[(BaseUnit::Length, 1)]);
        let mu = var(
            "mu",
            &[
                (BaseUnit::Mass, 1),
                (BaseUnit::Length, -1),
                (BaseUnit::Time, -1),
            ],
        );

        let groups = BuckinghamPi::analyze(&[f, rho, v, l, mu]).unwrap();
        assert_eq!(groups.len(), 2, "Drag: 2 π groups expected");
    }

    // ── all dimensionless ─────────────────────────────────────────────────────

    #[test]
    fn test_all_dimensionless_error() {
        let a = PhysicalVar::new("a");
        let b = PhysicalVar::new("b");
        let result = BuckinghamPi::analyze(&[a, b]);
        assert!(matches!(result, Err(BuckinghamPiError::AllDimensionless)));
    }

    // ── too few variables ─────────────────────────────────────────────────────

    #[test]
    fn test_too_few_variables_error() {
        let v = var("L", &[(BaseUnit::Length, 1)]);
        let result = BuckinghamPi::analyze(&[v]);
        assert!(matches!(result, Err(BuckinghamPiError::TooFewVariables)));
    }

    // ── pi group is dimensionless ─────────────────────────────────────────────

    /// Verify that the π groups output are genuinely dimensionless by
    /// checking that the sum of dimensional contributions is zero for each
    /// base dimension.
    #[test]
    fn test_pi_groups_are_dimensionless() {
        // pendulum
        let length = var("L", &[(BaseUnit::Length, 1)]);
        let gravity = var("g", &[(BaseUnit::Length, 1), (BaseUnit::Time, -2)]);
        let period = var("T", &[(BaseUnit::Time, 1)]);
        let variables = [length, gravity, period];

        let groups = BuckinghamPi::analyze(&variables).unwrap();
        for group in &groups {
            for &dim in &BaseUnit::ALL {
                let dim_sum: f64 = group
                    .exponents
                    .iter()
                    .map(|(vname, (num, den))| {
                        let var = variables.iter().find(|v| v.name == *vname).unwrap();
                        var.exponent(dim) as f64 * (*num as f64 / *den as f64)
                    })
                    .sum();
                assert!(
                    dim_sum.abs() < 1e-10,
                    "π group '{}' is not dimensionless in {dim:?}: sum = {dim_sum}",
                    group.name
                );
            }
        }
    }

    // ── display ───────────────────────────────────────────────────────────────

    #[test]
    fn test_pi_group_display_non_empty() {
        let length = var("L", &[(BaseUnit::Length, 1)]);
        let gravity = var("g", &[(BaseUnit::Length, 1), (BaseUnit::Time, -2)]);
        let period = var("T", &[(BaseUnit::Time, 1)]);

        let groups = BuckinghamPi::analyze(&[length, gravity, period]).unwrap();
        let s = groups[0].display();
        assert!(s.contains("π1"), "Display should contain 'π1'");
    }

    // ── rational helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_rat_reduce() {
        assert_eq!(rat_reduce((6, 4)), (3, 2));
        assert_eq!(rat_reduce((-6, 4)), (-3, 2));
        assert_eq!(rat_reduce((0, 5)), (0, 1));
    }

    #[test]
    fn test_rat_mul() {
        assert_eq!(rat_mul((2, 3), (3, 4)), (1, 2));
    }

    #[test]
    fn test_rat_div() {
        assert_eq!(rat_div((1, 2), (1, 4)), (2, 1));
    }

    #[test]
    fn test_rat_sub() {
        assert_eq!(rat_sub((3, 4), (1, 4)), (1, 2));
    }
}
