//! Structural stress analysis module for OxiRS Physics.
//!
//! Implements von Mises, Tresca, and principal stress calculations for
//! 2D and 3D stress states. Used in digital twin structural simulations.

/// Full 3D stress state (Cauchy stress tensor components).
#[derive(Debug, Clone, PartialEq)]
pub struct StressState {
    /// Normal stress in X direction [Pa]
    pub sigma_x: f64,
    /// Normal stress in Y direction [Pa]
    pub sigma_y: f64,
    /// Normal stress in Z direction [Pa]
    pub sigma_z: f64,
    /// Shear stress in XY plane [Pa]
    pub tau_xy: f64,
    /// Shear stress in YZ plane [Pa]
    pub tau_yz: f64,
    /// Shear stress in XZ plane [Pa]
    pub tau_xz: f64,
}

impl StressState {
    /// Create a 2D plane-stress state (sigma_z = tau_yz = tau_xz = 0).
    pub fn new_2d(sigma_x: f64, sigma_y: f64, tau_xy: f64) -> Self {
        Self {
            sigma_x,
            sigma_y,
            sigma_z: 0.0,
            tau_xy,
            tau_yz: 0.0,
            tau_xz: 0.0,
        }
    }

    /// Create a uniaxial stress state (only sigma_x is non-zero).
    pub fn new_uniaxial(sigma: f64) -> Self {
        Self {
            sigma_x: sigma,
            sigma_y: 0.0,
            sigma_z: 0.0,
            tau_xy: 0.0,
            tau_yz: 0.0,
            tau_xz: 0.0,
        }
    }

    /// Create a biaxial stress state (sigma_x and sigma_y; no shear, no Z).
    pub fn new_biaxial(sigma_x: f64, sigma_y: f64) -> Self {
        Self {
            sigma_x,
            sigma_y,
            sigma_z: 0.0,
            tau_xy: 0.0,
            tau_yz: 0.0,
            tau_xz: 0.0,
        }
    }

    /// Create a hydrostatic (isotropic) stress state: sigma_x = sigma_y = sigma_z = p.
    pub fn new_hydrostatic(p: f64) -> Self {
        Self {
            sigma_x: p,
            sigma_y: p,
            sigma_z: p,
            tau_xy: 0.0,
            tau_yz: 0.0,
            tau_xz: 0.0,
        }
    }
}

/// Principal stress eigenvalues.
#[derive(Debug, Clone, PartialEq)]
pub struct PrincipalStresses {
    /// Maximum principal stress [Pa]
    pub sigma_1: f64,
    /// Intermediate principal stress [Pa]
    pub sigma_2: f64,
    /// Minimum principal stress [Pa]
    pub sigma_3: f64,
}

/// Complete result of a stress analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct StressAnalysisResult {
    /// Principal stress eigenvalues
    pub principal: PrincipalStresses,
    /// von Mises equivalent stress [Pa]
    pub von_mises: f64,
    /// Tresca (maximum shear stress) criterion value [Pa]
    pub tresca: f64,
    /// Hydrostatic (mean normal) stress [Pa]
    pub hydrostatic: f64,
    /// Magnitude of the deviatoric stress tensor [Pa]
    pub deviatoric_magnitude: f64,
    /// Safety factor = yield_strength / von_mises  (inf when von_mises == 0)
    pub safety_factor: f64,
}

/// Stateless stress analysis engine.
pub struct StressAnalyzer;

impl StressAnalyzer {
    /// Perform a full stress analysis given a stress state and yield strength.
    ///
    /// Returns a [`StressAnalysisResult`] containing all derived quantities.
    pub fn analyze(state: &StressState, yield_strength: f64) -> StressAnalysisResult {
        let vm = Self::von_mises(state);
        let tr = Self::tresca(state);
        let hs = Self::hydrostatic_stress(state);
        let principal = Self::principal_stresses_3d(state);

        // Deviatoric stress tensor magnitude (Frobenius norm of deviatoric part)
        let s_x = state.sigma_x - hs;
        let s_y = state.sigma_y - hs;
        let s_z = state.sigma_z - hs;
        let dev_mag = (s_x * s_x
            + s_y * s_y
            + s_z * s_z
            + 2.0 * state.tau_xy * state.tau_xy
            + 2.0 * state.tau_yz * state.tau_yz
            + 2.0 * state.tau_xz * state.tau_xz)
            .sqrt();

        let safety_factor = if vm == 0.0 {
            f64::INFINITY
        } else {
            yield_strength / vm
        };

        StressAnalysisResult {
            principal,
            von_mises: vm,
            tresca: tr,
            hydrostatic: hs,
            deviatoric_magnitude: dev_mag,
            safety_factor,
        }
    }

    /// Compute the von Mises equivalent stress.
    ///
    /// `sigma_vm = sqrt(0.5 * [(sx-sy)^2 + (sy-sz)^2 + (sz-sx)^2 + 6*(txy^2+tyz^2+txz^2)])`
    pub fn von_mises(state: &StressState) -> f64 {
        let d1 = state.sigma_x - state.sigma_y;
        let d2 = state.sigma_y - state.sigma_z;
        let d3 = state.sigma_z - state.sigma_x;
        let shear_term = 6.0
            * (state.tau_xy * state.tau_xy
                + state.tau_yz * state.tau_yz
                + state.tau_xz * state.tau_xz);
        (0.5 * (d1 * d1 + d2 * d2 + d3 * d3 + shear_term)).sqrt()
    }

    /// Compute the Tresca criterion value = sigma_1 - sigma_3 (max - min principal stress).
    pub fn tresca(state: &StressState) -> f64 {
        let p = Self::principal_stresses_3d(state);
        // sigma_1 >= sigma_2 >= sigma_3 guaranteed by ordering
        p.sigma_1 - p.sigma_3
    }

    /// Compute the 2D principal stresses using the standard Mohr circle formula.
    ///
    /// Returns `(sigma_max, sigma_min)`.
    pub fn principal_stresses_2d(sigma_x: f64, sigma_y: f64, tau_xy: f64) -> (f64, f64) {
        let avg = (sigma_x + sigma_y) / 2.0;
        let r = Self::mohr_circle_radius(sigma_x, sigma_y, tau_xy);
        (avg + r, avg - r)
    }

    /// Compute all three principal stresses for the full 3D stress tensor.
    ///
    /// Eigenvalues of the symmetric 3×3 stress tensor are found using the
    /// analytical trigonometric method (Smith 1961), which is numerically stable
    /// for all symmetric matrices including degenerate cases.
    pub fn principal_stresses_3d(state: &StressState) -> PrincipalStresses {
        // Stress invariants
        let i1 = state.sigma_x + state.sigma_y + state.sigma_z;

        let i2 = state.sigma_x * state.sigma_y
            + state.sigma_y * state.sigma_z
            + state.sigma_z * state.sigma_x
            - state.tau_xy * state.tau_xy
            - state.tau_yz * state.tau_yz
            - state.tau_xz * state.tau_xz;

        let i3 = state.sigma_x * state.sigma_y * state.sigma_z
            + 2.0 * state.tau_xy * state.tau_yz * state.tau_xz
            - state.sigma_x * state.tau_yz * state.tau_yz
            - state.sigma_y * state.tau_xz * state.tau_xz
            - state.sigma_z * state.tau_xy * state.tau_xy;

        // Shift to a traceless problem: lambda = mu + I1/3
        let shift = i1 / 3.0;

        // Coefficients of the depressed cubic: mu^3 + p*mu + q = 0
        // p = I2 - I1^2/3
        // q = -2*I1^3/27 + I1*I2/3 - I3
        let p_coeff = i2 - i1 * i1 / 3.0;   // <= 0 always (for real eigenvalues)
        let q_coeff = -2.0 * i1 * i1 * i1 / 27.0 + i1 * i2 / 3.0 - i3;

        // Use trigonometric solution — valid whenever the cubic has three real roots.
        // A symmetric real matrix always has three real eigenvalues.
        // To guard against floating-point negative under sqrt, clamp.
        let r_sq = (-p_coeff / 3.0).max(0.0);
        let r_val = 2.0 * r_sq.sqrt();

        let (s1, s2, s3) = if r_val < 1e-14 * (i1.abs() + 1.0) {
            // All three eigenvalues are equal (hydrostatic or near-hydrostatic)
            (shift, shift, shift)
        } else {
            // cos(3*theta) = 3*q / (2*p) * sqrt(-3/p) = 3*q_coeff * sqrt(-1/p_coeff) / 2
            // Equivalently: cos(3*theta) = q_coeff / (r_val^3 / 4) ... derive carefully:
            // r_val = 2*sqrt(-p/3)  =>  r_val^3/8 = -p^(3/2)/3^(3/2)
            // cos(3*theta) = (q_coeff * 3) / (p_coeff * r_val)  [standard form]
            let cos_arg = {
                let denom = p_coeff * r_val;
                if denom.abs() < 1e-30 {
                    0.0
                } else {
                    (3.0 * q_coeff / denom).clamp(-1.0, 1.0)
                }
            };
            let theta = cos_arg.acos() / 3.0;
            let two_pi_3 = 2.0 * std::f64::consts::PI / 3.0;

            let t0 = r_val * theta.cos() + shift;
            let t1 = r_val * (theta - two_pi_3).cos() + shift;
            let t2 = r_val * (theta - 2.0 * two_pi_3).cos() + shift;
            (t0, t1, t2)
        };

        // Sort: sigma_1 >= sigma_2 >= sigma_3
        let mut vals = [s1, s2, s3];
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        PrincipalStresses {
            sigma_1: vals[0],
            sigma_2: vals[1],
            sigma_3: vals[2],
        }
    }

    /// Compute the hydrostatic (mean normal) stress = (sigma_x + sigma_y + sigma_z) / 3.
    pub fn hydrostatic_stress(state: &StressState) -> f64 {
        (state.sigma_x + state.sigma_y + state.sigma_z) / 3.0
    }

    /// Determine if the material has yielded using the von Mises criterion.
    ///
    /// Returns `true` when `von_mises >= yield_strength`.
    pub fn is_yielded(state: &StressState, yield_strength: f64) -> bool {
        Self::von_mises(state) >= yield_strength
    }

    /// Compute the Mohr circle radius for a 2D stress state.
    ///
    /// `r = sqrt(((sigma_x - sigma_y)/2)^2 + tau_xy^2)`
    pub fn mohr_circle_radius(sigma_x: f64, sigma_y: f64, tau_xy: f64) -> f64 {
        let half_diff = (sigma_x - sigma_y) / 2.0;
        (half_diff * half_diff + tau_xy * tau_xy).sqrt()
    }

    /// Maximum shear stress in 2D = Mohr circle radius.
    pub fn max_shear_stress_2d(sigma_x: f64, sigma_y: f64, tau_xy: f64) -> f64 {
        Self::mohr_circle_radius(sigma_x, sigma_y, tau_xy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // ── StressState constructors ──────────────────────────────────────────────

    #[test]
    fn test_new_uniaxial_zero_components() {
        let s = StressState::new_uniaxial(100.0);
        assert_eq!(s.sigma_x, 100.0);
        assert_eq!(s.sigma_y, 0.0);
        assert_eq!(s.sigma_z, 0.0);
        assert_eq!(s.tau_xy, 0.0);
        assert_eq!(s.tau_yz, 0.0);
        assert_eq!(s.tau_xz, 0.0);
    }

    #[test]
    fn test_new_2d_zero_z_components() {
        let s = StressState::new_2d(50.0, 30.0, 10.0);
        assert_eq!(s.sigma_x, 50.0);
        assert_eq!(s.sigma_y, 30.0);
        assert_eq!(s.sigma_z, 0.0);
        assert_eq!(s.tau_xy, 10.0);
        assert_eq!(s.tau_yz, 0.0);
        assert_eq!(s.tau_xz, 0.0);
    }

    #[test]
    fn test_new_biaxial_zero_shear() {
        let s = StressState::new_biaxial(200.0, -50.0);
        assert_eq!(s.sigma_x, 200.0);
        assert_eq!(s.sigma_y, -50.0);
        assert_eq!(s.tau_xy, 0.0);
        assert_eq!(s.tau_xz, 0.0);
        assert_eq!(s.tau_yz, 0.0);
    }

    #[test]
    fn test_new_hydrostatic_equal_normals() {
        let s = StressState::new_hydrostatic(300.0);
        assert_eq!(s.sigma_x, 300.0);
        assert_eq!(s.sigma_y, 300.0);
        assert_eq!(s.sigma_z, 300.0);
        assert_eq!(s.tau_xy, 0.0);
    }

    // ── von Mises ─────────────────────────────────────────────────────────────

    #[test]
    fn test_von_mises_uniaxial_equals_sigma() {
        // For uniaxial: vm = |sigma_x|
        let s = StressState::new_uniaxial(250.0);
        let vm = StressAnalyzer::von_mises(&s);
        assert!((vm - 250.0).abs() < EPS, "vm={vm}");
    }

    #[test]
    fn test_von_mises_uniaxial_negative_equals_magnitude() {
        let s = StressState::new_uniaxial(-300.0);
        let vm = StressAnalyzer::von_mises(&s);
        assert!((vm - 300.0).abs() < EPS, "vm={vm}");
    }

    #[test]
    fn test_von_mises_hydrostatic_zero() {
        // Hydrostatic stress has zero deviatoric part → von Mises = 0
        let s = StressState::new_hydrostatic(500.0);
        let vm = StressAnalyzer::von_mises(&s);
        assert!(vm < EPS, "vm should be ~0 for hydrostatic, got {vm}");
    }

    #[test]
    fn test_von_mises_pure_shear() {
        // Pure shear: vm = sqrt(3) * tau
        let tau = 100.0;
        let s = StressState {
            sigma_x: 0.0,
            sigma_y: 0.0,
            sigma_z: 0.0,
            tau_xy: tau,
            tau_yz: 0.0,
            tau_xz: 0.0,
        };
        let vm = StressAnalyzer::von_mises(&s);
        let expected = (3.0_f64).sqrt() * tau;
        assert!((vm - expected).abs() < EPS * 100.0, "vm={vm}, expected={expected}");
    }

    #[test]
    fn test_von_mises_biaxial_equal() {
        // Equal biaxial (2D): sigma_x = sigma_y = S, sigma_z = 0
        // 0.5*[(S-S)^2 + (S-0)^2 + (0-S)^2] = 0.5*(0 + S^2 + S^2) = S^2 → vm = S
        let s = StressState::new_biaxial(100.0, 100.0);
        let vm = StressAnalyzer::von_mises(&s);
        assert!((vm - 100.0).abs() < EPS * 10.0, "vm should be 100.0, got {vm}");
    }

    #[test]
    fn test_von_mises_biaxial_opposite() {
        // sigma_x = S, sigma_y = -S → vm = sqrt(3) * S
        let sigma = 200.0;
        let s = StressState::new_biaxial(sigma, -sigma);
        let vm = StressAnalyzer::von_mises(&s);
        let expected = (3.0_f64).sqrt() * sigma;
        assert!((vm - expected).abs() < EPS * 1000.0, "vm={vm}, expected={expected}");
    }

    // ── Tresca ────────────────────────────────────────────────────────────────

    #[test]
    fn test_tresca_hydrostatic_zero() {
        // All principal stresses equal → Tresca = 0
        let s = StressState::new_hydrostatic(400.0);
        let tr = StressAnalyzer::tresca(&s);
        assert!(tr.abs() < 1e-6, "tresca should be 0, got {tr}");
    }

    #[test]
    fn test_tresca_uniaxial() {
        // sigma_1 = S, sigma_2 = sigma_3 = 0 → Tresca = S
        let s = StressState::new_uniaxial(500.0);
        let tr = StressAnalyzer::tresca(&s);
        assert!((tr - 500.0).abs() < 1e-6, "tresca={tr}");
    }

    #[test]
    fn test_tresca_nonnegative() {
        let s = StressState::new_biaxial(300.0, -200.0);
        let tr = StressAnalyzer::tresca(&s);
        assert!(tr >= 0.0, "Tresca must be non-negative, got {tr}");
    }

    // ── Hydrostatic ───────────────────────────────────────────────────────────

    #[test]
    fn test_hydrostatic_uniaxial() {
        let s = StressState::new_uniaxial(300.0);
        let hs = StressAnalyzer::hydrostatic_stress(&s);
        assert!((hs - 100.0).abs() < EPS, "hs={hs}");
    }

    #[test]
    fn test_hydrostatic_symmetric() {
        let s = StressState::new_hydrostatic(150.0);
        let hs = StressAnalyzer::hydrostatic_stress(&s);
        assert!((hs - 150.0).abs() < EPS, "hs={hs}");
    }

    #[test]
    fn test_hydrostatic_zero_for_pure_shear() {
        let s = StressState { sigma_x: 0.0, sigma_y: 0.0, sigma_z: 0.0,
            tau_xy: 50.0, tau_yz: 0.0, tau_xz: 0.0 };
        let hs = StressAnalyzer::hydrostatic_stress(&s);
        assert!(hs.abs() < EPS);
    }

    // ── Mohr circle radius ────────────────────────────────────────────────────

    #[test]
    fn test_mohr_radius_pure_normal() {
        // No shear: radius = |sigma_x - sigma_y| / 2
        let r = StressAnalyzer::mohr_circle_radius(100.0, 60.0, 0.0);
        assert!((r - 20.0).abs() < EPS);
    }

    #[test]
    fn test_mohr_radius_pure_shear() {
        // No normal difference: radius = |tau|
        let r = StressAnalyzer::mohr_circle_radius(50.0, 50.0, 30.0);
        assert!((r - 30.0).abs() < EPS);
    }

    #[test]
    fn test_mohr_radius_nonnegative() {
        let r = StressAnalyzer::mohr_circle_radius(-100.0, 100.0, 50.0);
        assert!(r >= 0.0);
    }

    #[test]
    fn test_mohr_radius_pythagorean() {
        // half_diff = 30, tau = 40 → radius = 50
        let r = StressAnalyzer::mohr_circle_radius(130.0, 70.0, 40.0);
        assert!((r - 50.0).abs() < EPS);
    }

    // ── 2D principal stresses ─────────────────────────────────────────────────

    #[test]
    fn test_principal_2d_no_shear() {
        // No shear: principals = sigma_x, sigma_y
        let (p1, p2) = StressAnalyzer::principal_stresses_2d(100.0, 40.0, 0.0);
        assert!((p1 - 100.0).abs() < EPS);
        assert!((p2 - 40.0).abs() < EPS);
    }

    #[test]
    fn test_principal_2d_ordering() {
        let (p1, p2) = StressAnalyzer::principal_stresses_2d(40.0, 100.0, 0.0);
        assert!(p1 >= p2);
    }

    #[test]
    fn test_principal_2d_symmetric() {
        // Equal normal stresses, pure shear
        let tau = 50.0;
        let (p1, p2) = StressAnalyzer::principal_stresses_2d(100.0, 100.0, tau);
        assert!((p1 - (100.0 + tau)).abs() < EPS);
        assert!((p2 - (100.0 - tau)).abs() < EPS);
    }

    #[test]
    fn test_principal_2d_known_example() {
        // sigma_x=80, sigma_y=40, tau=30
        // avg = (80+40)/2 = 60
        // half_diff = (80-40)/2 = 20
        // R = sqrt(20^2 + 30^2) = sqrt(400+900) = sqrt(1300) ≈ 36.056
        // p1 = 60 + R ≈ 96.056, p2 = 60 - R ≈ 23.944
        let (p1, p2) = StressAnalyzer::principal_stresses_2d(80.0, 40.0, 30.0);
        let r = (1300.0_f64).sqrt();
        assert!((p1 - (60.0 + r)).abs() < EPS * 100.0, "p1={p1}, expected={}", 60.0 + r);
        assert!((p2 - (60.0 - r)).abs() < EPS * 100.0, "p2={p2}, expected={}", 60.0 - r);
    }

    // ── 3D principal stresses ─────────────────────────────────────────────────

    #[test]
    fn test_principal_3d_diagonal_ordering() {
        // Diagonal (no shear): principals are sigma_x, sigma_y, sigma_z sorted
        let s = StressState {
            sigma_x: 300.0, sigma_y: 100.0, sigma_z: 200.0,
            tau_xy: 0.0, tau_yz: 0.0, tau_xz: 0.0,
        };
        let p = StressAnalyzer::principal_stresses_3d(&s);
        assert!((p.sigma_1 - 300.0).abs() < 1e-6, "sigma_1={}", p.sigma_1);
        assert!((p.sigma_2 - 200.0).abs() < 1e-6, "sigma_2={}", p.sigma_2);
        assert!((p.sigma_3 - 100.0).abs() < 1e-6, "sigma_3={}", p.sigma_3);
    }

    #[test]
    fn test_principal_3d_ordering_invariant() {
        let s = StressState::new_2d(80.0, 40.0, 30.0);
        let p = StressAnalyzer::principal_stresses_3d(&s);
        assert!(p.sigma_1 >= p.sigma_2 - EPS);
        assert!(p.sigma_2 >= p.sigma_3 - EPS);
    }

    #[test]
    fn test_principal_3d_hydrostatic_equal() {
        let s = StressState::new_hydrostatic(200.0);
        let p = StressAnalyzer::principal_stresses_3d(&s);
        assert!((p.sigma_1 - 200.0).abs() < 1e-6);
        assert!((p.sigma_2 - 200.0).abs() < 1e-6);
        assert!((p.sigma_3 - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_principal_3d_uniaxial() {
        let s = StressState::new_uniaxial(500.0);
        let p = StressAnalyzer::principal_stresses_3d(&s);
        assert!((p.sigma_1 - 500.0).abs() < 1e-6, "sigma_1={}", p.sigma_1);
        assert!(p.sigma_2.abs() < 1e-6);
        assert!(p.sigma_3.abs() < 1e-6);
    }

    // ── is_yielded ────────────────────────────────────────────────────────────

    #[test]
    fn test_is_yielded_below_threshold() {
        let s = StressState::new_uniaxial(100.0);
        assert!(!StressAnalyzer::is_yielded(&s, 200.0));
    }

    #[test]
    fn test_is_yielded_at_threshold() {
        let s = StressState::new_uniaxial(200.0);
        assert!(StressAnalyzer::is_yielded(&s, 200.0));
    }

    #[test]
    fn test_is_yielded_above_threshold() {
        let s = StressState::new_uniaxial(300.0);
        assert!(StressAnalyzer::is_yielded(&s, 250.0));
    }

    #[test]
    fn test_is_yielded_zero_stress_never_yields() {
        let s = StressState::new_uniaxial(0.0);
        // von Mises == 0 < any positive yield strength
        assert!(!StressAnalyzer::is_yielded(&s, 100.0));
    }

    // ── safety_factor ─────────────────────────────────────────────────────────

    #[test]
    fn test_safety_factor_uniaxial() {
        let s = StressState::new_uniaxial(100.0);
        let r = StressAnalyzer::analyze(&s, 400.0);
        assert!((r.safety_factor - 4.0).abs() < EPS * 10.0);
    }

    #[test]
    fn test_safety_factor_zero_stress_is_inf() {
        let s = StressState::new_uniaxial(0.0);
        let r = StressAnalyzer::analyze(&s, 250.0);
        assert!(r.safety_factor.is_infinite());
    }

    #[test]
    fn test_safety_factor_equals_one_at_yield() {
        let s = StressState::new_uniaxial(300.0);
        let r = StressAnalyzer::analyze(&s, 300.0);
        assert!((r.safety_factor - 1.0).abs() < EPS * 10.0);
    }

    // ── Full analyze() ────────────────────────────────────────────────────────

    #[test]
    fn test_analyze_fields_consistent() {
        let s = StressState::new_biaxial(100.0, 50.0);
        let r = StressAnalyzer::analyze(&s, 300.0);
        // von_mises should match direct call
        let vm_direct = StressAnalyzer::von_mises(&s);
        assert!((r.von_mises - vm_direct).abs() < EPS);
    }

    #[test]
    fn test_analyze_tresca_nonnegative() {
        let s = StressState::new_2d(200.0, -100.0, 75.0);
        let r = StressAnalyzer::analyze(&s, 500.0);
        assert!(r.tresca >= 0.0);
    }

    #[test]
    fn test_analyze_deviatoric_zero_for_hydrostatic() {
        let s = StressState::new_hydrostatic(123.0);
        let r = StressAnalyzer::analyze(&s, 500.0);
        assert!(r.deviatoric_magnitude < 1e-9, "dev_mag={}", r.deviatoric_magnitude);
    }

    #[test]
    fn test_analyze_hydrostatic_value() {
        let s = StressState {
            sigma_x: 60.0, sigma_y: 90.0, sigma_z: 120.0,
            tau_xy: 0.0, tau_yz: 0.0, tau_xz: 0.0,
        };
        let r = StressAnalyzer::analyze(&s, 500.0);
        // (60+90+120)/3 = 90
        assert!((r.hydrostatic - 90.0).abs() < EPS * 10.0);
    }

    // ── max_shear_stress_2d ───────────────────────────────────────────────────

    #[test]
    fn test_max_shear_2d_equals_radius() {
        let r = StressAnalyzer::max_shear_stress_2d(80.0, 40.0, 30.0);
        let radius = StressAnalyzer::mohr_circle_radius(80.0, 40.0, 30.0);
        assert!((r - radius).abs() < EPS);
    }

    #[test]
    fn test_max_shear_2d_uniaxial() {
        // tau_max = S/2
        let r = StressAnalyzer::max_shear_stress_2d(200.0, 0.0, 0.0);
        assert!((r - 100.0).abs() < EPS);
    }

    // ── Regression / integration ──────────────────────────────────────────────

    #[test]
    fn test_von_mises_biaxial_known() {
        // sigma_x=100, sigma_y=50, tau=0 → vm = sqrt(100^2 - 100*50 + 50^2) = sqrt(7500)
        let s = StressState::new_biaxial(100.0, 50.0);
        let vm = StressAnalyzer::von_mises(&s);
        // Manual: 0.5*((100-50)^2+(50-0)^2+(0-100)^2) = 0.5*(2500+2500+10000) = 7500
        let expected = 7500.0_f64.sqrt();
        assert!((vm - expected).abs() < 1e-6, "vm={vm}, expected={expected}");
    }

    #[test]
    fn test_safety_below_one_means_yielded() {
        let s = StressState::new_uniaxial(500.0);
        let r = StressAnalyzer::analyze(&s, 200.0);
        assert!(r.safety_factor < 1.0);
        assert!(StressAnalyzer::is_yielded(&s, 200.0));
    }

    #[test]
    fn test_principal_3d_sum_equals_trace() {
        let s = StressState {
            sigma_x: 100.0, sigma_y: 200.0, sigma_z: 300.0,
            tau_xy: 50.0, tau_yz: 30.0, tau_xz: 20.0,
        };
        let p = StressAnalyzer::principal_stresses_3d(&s);
        let trace = s.sigma_x + s.sigma_y + s.sigma_z;
        let principal_sum = p.sigma_1 + p.sigma_2 + p.sigma_3;
        // I1 invariant: sum of principals = trace
        assert!((principal_sum - trace).abs() < 1e-4, "sum={principal_sum}, trace={trace}");
    }

    #[test]
    fn test_analyze_full_3d() {
        let s = StressState {
            sigma_x: 150.0, sigma_y: 100.0, sigma_z: 50.0,
            tau_xy: 40.0, tau_yz: 20.0, tau_xz: 10.0,
        };
        let result = StressAnalyzer::analyze(&s, 400.0);
        assert!(result.von_mises > 0.0);
        assert!(result.tresca > 0.0);
        assert!(result.safety_factor > 0.0);
        assert!(result.principal.sigma_1 >= result.principal.sigma_2 - 1e-6);
        assert!(result.principal.sigma_2 >= result.principal.sigma_3 - 1e-6);
    }
}
