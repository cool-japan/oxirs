//! Optical physics.
//!
//! Implements Snell's law, Fresnel equations, thin-lens equation, geometric
//! ray tracing, diffraction grating analysis, Brewster angle, total internal
//! reflection, optical path length, prism dispersion, and numerical aperture.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to optics calculations.
#[derive(Debug, Clone, PartialEq)]
pub enum OpticsError {
    /// Total internal reflection occurs (no transmitted ray).
    TotalInternalReflection,
    /// A physical parameter is invalid (e.g. negative refractive index).
    InvalidParameter(String),
    /// No real image is formed.
    NoImage,
}

impl std::fmt::Display for OpticsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpticsError::TotalInternalReflection => write!(f, "total internal reflection"),
            OpticsError::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            OpticsError::NoImage => write!(f, "no real image formed"),
        }
    }
}

impl std::error::Error for OpticsError {}

pub type OpticsResult<T> = Result<T, OpticsError>;

// ---------------------------------------------------------------------------
// Surface description for ray tracing
// ---------------------------------------------------------------------------

/// A single optical surface (planar interface between two media).
#[derive(Debug, Clone)]
pub struct OpticalSurface {
    /// Refractive index on the incoming side.
    pub n1: f64,
    /// Refractive index on the outgoing side.
    pub n2: f64,
}

/// Result of tracing a ray through a sequence of surfaces.
#[derive(Debug, Clone)]
pub struct RayTraceResult {
    /// Angle of incidence at each surface (radians).
    pub incidence_angles: Vec<f64>,
    /// Angle of refraction at each surface (radians).
    pub refraction_angles: Vec<f64>,
    /// Final transmitted angle (radians).
    pub final_angle: f64,
}

// ---------------------------------------------------------------------------
// Fresnel result
// ---------------------------------------------------------------------------

/// Reflectance and transmittance from the Fresnel equations.
#[derive(Debug, Clone)]
pub struct FresnelResult {
    /// Reflectance for s-polarisation (TE).
    pub rs: f64,
    /// Reflectance for p-polarisation (TM).
    pub rp: f64,
    /// Transmittance for s-polarisation.
    pub ts: f64,
    /// Transmittance for p-polarisation.
    pub tp: f64,
}

// ---------------------------------------------------------------------------
// Thin lens result
// ---------------------------------------------------------------------------

/// Output of the thin lens equation.
#[derive(Debug, Clone)]
pub struct ThinLensResult {
    /// Image distance (positive = real image on far side).
    pub image_distance: f64,
    /// Lateral magnification (negative = inverted).
    pub magnification: f64,
}

// ---------------------------------------------------------------------------
// Diffraction grating result
// ---------------------------------------------------------------------------

/// Result of diffraction grating analysis.
#[derive(Debug, Clone)]
pub struct DiffractionResult {
    /// Maximum diffraction order observable.
    pub max_order: i32,
    /// Angles of maxima for each order `(order, angle_rad)`.
    pub maxima: Vec<(i32, f64)>,
    /// Resolving power `R = m * N`.
    pub resolving_power: f64,
}

// ---------------------------------------------------------------------------
// Core optics functions
// ---------------------------------------------------------------------------

/// Compute the refraction angle using Snell's law.
///
/// `n1 * sin(theta_i) = n2 * sin(theta_t)`
///
/// Returns the transmitted angle in radians.  Returns
/// `OpticsError::TotalInternalReflection` if `sin(theta_t) > 1`.
pub fn snells_law(n1: f64, n2: f64, theta_i: f64) -> OpticsResult<f64> {
    if n1 <= 0.0 || n2 <= 0.0 {
        return Err(OpticsError::InvalidParameter(
            "refractive indices must be positive".to_string(),
        ));
    }
    let sin_t = n1 * theta_i.sin() / n2;
    if sin_t.abs() > 1.0 {
        return Err(OpticsError::TotalInternalReflection);
    }
    Ok(sin_t.asin())
}

/// Compute Fresnel reflectance and transmittance at an interface.
///
/// Returns `FresnelResult` with s- and p-polarisation coefficients.
pub fn fresnel(n1: f64, n2: f64, theta_i: f64) -> OpticsResult<FresnelResult> {
    let theta_t = snells_law(n1, n2, theta_i)?;

    let cos_i = theta_i.cos();
    let cos_t = theta_t.cos();

    // Amplitude reflection coefficients
    let rs_num = n1 * cos_i - n2 * cos_t;
    let rs_den = n1 * cos_i + n2 * cos_t;
    let rp_num = n2 * cos_i - n1 * cos_t;
    let rp_den = n2 * cos_i + n1 * cos_t;

    let rs = if rs_den.abs() < 1e-15 {
        0.0
    } else {
        (rs_num / rs_den).powi(2)
    };
    let rp = if rp_den.abs() < 1e-15 {
        0.0
    } else {
        (rp_num / rp_den).powi(2)
    };

    Ok(FresnelResult {
        rs,
        rp,
        ts: 1.0 - rs,
        tp: 1.0 - rp,
    })
}

/// Thin lens equation: `1/f = 1/d_o + 1/d_i`.
///
/// `focal_length` and `object_distance` are signed per the sign convention
/// (positive for real objects/converging lenses).
pub fn thin_lens(focal_length: f64, object_distance: f64) -> OpticsResult<ThinLensResult> {
    if focal_length.abs() < 1e-15 {
        return Err(OpticsError::InvalidParameter(
            "focal length cannot be zero".to_string(),
        ));
    }
    let inv_di = 1.0 / focal_length - 1.0 / object_distance;
    if inv_di.abs() < 1e-15 {
        return Err(OpticsError::NoImage);
    }
    let image_distance = 1.0 / inv_di;
    let magnification = -image_distance / object_distance;

    Ok(ThinLensResult {
        image_distance,
        magnification,
    })
}

/// Trace a ray through a sequence of optical surfaces.
///
/// `initial_angle` is the angle of incidence on the first surface (radians).
pub fn trace_ray(surfaces: &[OpticalSurface], initial_angle: f64) -> OpticsResult<RayTraceResult> {
    let mut incidence_angles = Vec::with_capacity(surfaces.len());
    let mut refraction_angles = Vec::with_capacity(surfaces.len());

    let mut current_angle = initial_angle;

    for surface in surfaces {
        incidence_angles.push(current_angle);
        let refracted = snells_law(surface.n1, surface.n2, current_angle)?;
        refraction_angles.push(refracted);
        current_angle = refracted;
    }

    Ok(RayTraceResult {
        incidence_angles,
        refraction_angles,
        final_angle: current_angle,
    })
}

/// Analyse a diffraction grating.
///
/// * `d` – grating spacing in metres.
/// * `wavelength` – light wavelength in metres.
/// * `n_slits` – number of slits.
pub fn diffraction_grating(
    d: f64,
    wavelength: f64,
    n_slits: usize,
) -> OpticsResult<DiffractionResult> {
    if d <= 0.0 || wavelength <= 0.0 {
        return Err(OpticsError::InvalidParameter(
            "grating spacing and wavelength must be positive".to_string(),
        ));
    }

    let max_order = (d / wavelength).floor() as i32;
    let mut maxima = Vec::new();

    for m in -max_order..=max_order {
        let sin_theta = (m as f64) * wavelength / d;
        if sin_theta.abs() <= 1.0 {
            maxima.push((m, sin_theta.asin()));
        }
    }

    let resolving_power = max_order as f64 * n_slits as f64;

    Ok(DiffractionResult {
        max_order,
        maxima,
        resolving_power,
    })
}

/// Compute the Brewster angle (angle of incidence at which p-polarised
/// light is perfectly transmitted).
///
/// `theta_B = atan(n2 / n1)`
pub fn brewster_angle(n1: f64, n2: f64) -> OpticsResult<f64> {
    if n1 <= 0.0 || n2 <= 0.0 {
        return Err(OpticsError::InvalidParameter(
            "refractive indices must be positive".to_string(),
        ));
    }
    Ok((n2 / n1).atan())
}

/// Compute the critical angle for total internal reflection.
///
/// `theta_c = asin(n2 / n1)` where `n1 > n2`.
pub fn critical_angle(n1: f64, n2: f64) -> OpticsResult<f64> {
    if n1 <= 0.0 || n2 <= 0.0 {
        return Err(OpticsError::InvalidParameter(
            "refractive indices must be positive".to_string(),
        ));
    }
    if n1 <= n2 {
        return Err(OpticsError::InvalidParameter(
            "n1 must be greater than n2 for total internal reflection".to_string(),
        ));
    }
    let ratio = n2 / n1;
    Ok(ratio.asin())
}

/// Compute the optical path length through a medium.
///
/// `OPL = n * d`
pub fn optical_path_length(refractive_index: f64, physical_length: f64) -> OpticsResult<f64> {
    if refractive_index <= 0.0 {
        return Err(OpticsError::InvalidParameter(
            "refractive index must be positive".to_string(),
        ));
    }
    if physical_length < 0.0 {
        return Err(OpticsError::InvalidParameter(
            "physical length cannot be negative".to_string(),
        ));
    }
    Ok(refractive_index * physical_length)
}

/// Compute the minimum deviation angle of a prism.
///
/// At minimum deviation: `n = sin((A + D_min)/2) / sin(A/2)`
/// Solving: `D_min = 2 * asin(n * sin(A/2)) - A`
///
/// * `n` – refractive index of the prism.
/// * `apex_angle` – apex angle of the prism in radians.
pub fn prism_minimum_deviation(n: f64, apex_angle: f64) -> OpticsResult<f64> {
    if n <= 0.0 {
        return Err(OpticsError::InvalidParameter(
            "refractive index must be positive".to_string(),
        ));
    }
    if apex_angle <= 0.0 || apex_angle >= PI {
        return Err(OpticsError::InvalidParameter(
            "apex angle must be in (0, pi)".to_string(),
        ));
    }

    let sin_half = (apex_angle / 2.0).sin();
    let arg = n * sin_half;
    if arg.abs() > 1.0 {
        return Err(OpticsError::TotalInternalReflection);
    }
    let d_min = 2.0 * arg.asin() - apex_angle;
    Ok(d_min)
}

/// Compute the numerical aperture of a fiber or lens.
///
/// `NA = sqrt(n_core^2 - n_cladding^2)`
pub fn numerical_aperture(n_core: f64, n_cladding: f64) -> OpticsResult<f64> {
    if n_core <= 0.0 || n_cladding <= 0.0 {
        return Err(OpticsError::InvalidParameter(
            "refractive indices must be positive".to_string(),
        ));
    }
    if n_core < n_cladding {
        return Err(OpticsError::InvalidParameter(
            "core index must be >= cladding index".to_string(),
        ));
    }
    let na_sq = n_core * n_core - n_cladding * n_cladding;
    Ok(na_sq.sqrt())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DEG: f64 = PI / 180.0;

    // ── Snell's law ──────────────────────────────────────────────────────

    #[test]
    fn test_snells_law_normal_incidence() {
        let theta_t = snells_law(1.0, 1.5, 0.0).expect("should not fail");
        assert!(theta_t.abs() < 1e-10, "normal incidence → 0 refraction");
    }

    #[test]
    fn test_snells_law_air_to_glass() {
        let theta_t = snells_law(1.0, 1.5, 30.0 * DEG).expect("should work");
        // sin(30) = 0.5 → sin(theta_t) = 0.5/1.5 ≈ 0.333 → theta_t ≈ 19.47 deg
        let expected = (1.0_f64 / 3.0).asin();
        assert!((theta_t - expected).abs() < 1e-10);
    }

    #[test]
    fn test_snells_law_total_internal_reflection() {
        // Glass to air at steep angle
        let result = snells_law(1.5, 1.0, 50.0 * DEG);
        assert!(result.is_err());
    }

    #[test]
    fn test_snells_law_negative_n() {
        let result = snells_law(-1.0, 1.5, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_snells_law_symmetry() {
        // Reversibility: n1→n2 at theta gives theta_t, n2→n1 at theta_t gives theta
        let theta_i = 30.0 * DEG;
        let theta_t = snells_law(1.0, 1.5, theta_i).expect("ok");
        let theta_back = snells_law(1.5, 1.0, theta_t).expect("ok");
        assert!((theta_back - theta_i).abs() < 1e-10);
    }

    // ── Fresnel equations ────────────────────────────────────────────────

    #[test]
    fn test_fresnel_normal_incidence() {
        let f = fresnel(1.0, 1.5, 0.0).expect("ok");
        // At normal incidence Rs == Rp
        assert!((f.rs - f.rp).abs() < 1e-10);
    }

    #[test]
    fn test_fresnel_energy_conservation() {
        let f = fresnel(1.0, 1.5, 30.0 * DEG).expect("ok");
        assert!((f.rs + f.ts - 1.0).abs() < 1e-10);
        assert!((f.rp + f.tp - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fresnel_same_medium() {
        let f = fresnel(1.5, 1.5, 20.0 * DEG).expect("ok");
        assert!(f.rs < 1e-10);
        assert!(f.rp < 1e-10);
    }

    #[test]
    fn test_fresnel_total_internal_reflection() {
        let result = fresnel(1.5, 1.0, 50.0 * DEG);
        assert!(result.is_err());
    }

    // ── Thin lens equation ───────────────────────────────────────────────

    #[test]
    fn test_thin_lens_converging() {
        let r = thin_lens(10.0, 20.0).expect("ok");
        // 1/10 - 1/20 = 1/20 → d_i = 20
        assert!((r.image_distance - 20.0).abs() < 1e-10);
        // m = -20/20 = -1 (inverted, same size)
        assert!((r.magnification - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_thin_lens_object_at_2f() {
        let f = 5.0;
        let d_o = 2.0 * f;
        let r = thin_lens(f, d_o).expect("ok");
        assert!((r.image_distance - d_o).abs() < 1e-10);
    }

    #[test]
    fn test_thin_lens_zero_focal() {
        let result = thin_lens(0.0, 10.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_thin_lens_virtual_image() {
        // Object inside focal length → virtual image (negative d_i)
        let r = thin_lens(10.0, 5.0).expect("ok");
        assert!(r.image_distance < 0.0);
        assert!(r.magnification > 0.0); // upright
    }

    // ── Ray tracing ──────────────────────────────────────────────────────

    #[test]
    fn test_trace_ray_single_surface() {
        let surfaces = vec![OpticalSurface { n1: 1.0, n2: 1.5 }];
        let result = trace_ray(&surfaces, 30.0 * DEG).expect("ok");
        assert_eq!(result.incidence_angles.len(), 1);
        assert_eq!(result.refraction_angles.len(), 1);
    }

    #[test]
    fn test_trace_ray_two_surfaces_roundtrip() {
        // Air → glass → air: final angle should equal initial
        let surfaces = vec![
            OpticalSurface { n1: 1.0, n2: 1.5 },
            OpticalSurface { n1: 1.5, n2: 1.0 },
        ];
        let initial = 20.0 * DEG;
        let result = trace_ray(&surfaces, initial).expect("ok");
        assert!((result.final_angle - initial).abs() < 1e-10);
    }

    #[test]
    fn test_trace_ray_empty() {
        let result = trace_ray(&[], 0.0).expect("ok");
        assert!(result.incidence_angles.is_empty());
        assert_eq!(result.final_angle, 0.0);
    }

    // ── Diffraction grating ──────────────────────────────────────────────

    #[test]
    fn test_diffraction_grating_zeroth_order() {
        let result = diffraction_grating(1e-6, 500e-9, 100).expect("ok");
        // Zeroth order is always at 0
        assert!(result
            .maxima
            .iter()
            .any(|(m, a)| *m == 0 && a.abs() < 1e-10));
    }

    #[test]
    fn test_diffraction_grating_max_order() {
        let d = 2e-6;
        let wl = 500e-9;
        let result = diffraction_grating(d, wl, 50).expect("ok");
        // max_order = floor(d/wl) = floor(4.0) = 4
        assert_eq!(result.max_order, 4);
    }

    #[test]
    fn test_diffraction_grating_resolving_power() {
        let result = diffraction_grating(1e-6, 500e-9, 100).expect("ok");
        assert!(result.resolving_power > 0.0);
    }

    #[test]
    fn test_diffraction_grating_invalid() {
        assert!(diffraction_grating(0.0, 500e-9, 10).is_err());
        assert!(diffraction_grating(1e-6, -1.0, 10).is_err());
    }

    // ── Brewster angle ───────────────────────────────────────────────────

    #[test]
    fn test_brewster_angle_air_glass() {
        let theta_b = brewster_angle(1.0, 1.5).expect("ok");
        let expected = (1.5_f64).atan();
        assert!((theta_b - expected).abs() < 1e-10);
    }

    #[test]
    fn test_brewster_angle_complement_critical() {
        // For air-glass, Brewster + critical < 90 deg
        let theta_b = brewster_angle(1.5, 1.0).expect("ok");
        let theta_c = critical_angle(1.5, 1.0).expect("ok");
        assert!(theta_b + theta_c < PI / 2.0 + 0.1);
    }

    #[test]
    fn test_brewster_angle_invalid() {
        assert!(brewster_angle(-1.0, 1.5).is_err());
    }

    // ── Critical angle ───────────────────────────────────────────────────

    #[test]
    fn test_critical_angle_glass_air() {
        let theta_c = critical_angle(1.5, 1.0).expect("ok");
        let expected = (1.0_f64 / 1.5).asin();
        assert!((theta_c - expected).abs() < 1e-10);
    }

    #[test]
    fn test_critical_angle_n1_less_than_n2() {
        // No TIR possible when going from less dense to denser
        assert!(critical_angle(1.0, 1.5).is_err());
    }

    #[test]
    fn test_critical_angle_equal_indices() {
        assert!(critical_angle(1.5, 1.5).is_err());
    }

    // ── Optical path length ──────────────────────────────────────────────

    #[test]
    fn test_optical_path_length_vacuum() {
        let opl = optical_path_length(1.0, 10.0).expect("ok");
        assert!((opl - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_optical_path_length_glass() {
        let opl = optical_path_length(1.5, 10.0).expect("ok");
        assert!((opl - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_optical_path_length_zero_length() {
        let opl = optical_path_length(1.5, 0.0).expect("ok");
        assert!(opl.abs() < 1e-10);
    }

    #[test]
    fn test_optical_path_length_invalid_n() {
        assert!(optical_path_length(0.0, 10.0).is_err());
    }

    #[test]
    fn test_optical_path_length_negative_length() {
        assert!(optical_path_length(1.5, -1.0).is_err());
    }

    // ── Prism minimum deviation ──────────────────────────────────────────

    #[test]
    fn test_prism_min_deviation_equilateral() {
        // n=1.5, A=60deg → D_min ≈ 2*asin(1.5*sin(30)) - 60 deg
        let a = 60.0 * DEG;
        let d = prism_minimum_deviation(1.5, a).expect("ok");
        // For n=1.5, A=60: D_min ≈ 2*asin(0.75) - pi/3 ≈ 37.18 deg
        let expected = 2.0 * (1.5 * (a / 2.0).sin()).asin() - a;
        assert!((d - expected).abs() < 1e-10);
    }

    #[test]
    fn test_prism_min_deviation_positive() {
        let d = prism_minimum_deviation(1.5, 60.0 * DEG).expect("ok");
        assert!(d > 0.0);
    }

    #[test]
    fn test_prism_min_deviation_invalid_n() {
        assert!(prism_minimum_deviation(0.0, 60.0 * DEG).is_err());
    }

    #[test]
    fn test_prism_min_deviation_invalid_angle() {
        assert!(prism_minimum_deviation(1.5, 0.0).is_err());
        assert!(prism_minimum_deviation(1.5, PI).is_err());
    }

    // ── Numerical aperture ───────────────────────────────────────────────

    #[test]
    fn test_numerical_aperture_fiber() {
        let na = numerical_aperture(1.5, 1.45).expect("ok");
        let expected = ((1.5f64).powi(2) - (1.45f64).powi(2)).sqrt();
        assert!((na - expected).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_aperture_same_index() {
        let na = numerical_aperture(1.5, 1.5).expect("ok");
        assert!(na.abs() < 1e-10);
    }

    #[test]
    fn test_numerical_aperture_invalid() {
        assert!(numerical_aperture(1.0, 1.5).is_err()); // core < cladding
        assert!(numerical_aperture(-1.0, 0.5).is_err());
    }

    // ── OpticsError display ──────────────────────────────────────────────

    #[test]
    fn test_optics_error_display_tir() {
        let e = OpticsError::TotalInternalReflection;
        assert!(format!("{e}").contains("total internal reflection"));
    }

    #[test]
    fn test_optics_error_display_invalid() {
        let e = OpticsError::InvalidParameter("test".to_string());
        assert!(format!("{e}").contains("test"));
    }

    #[test]
    fn test_optics_error_display_no_image() {
        let e = OpticsError::NoImage;
        assert!(format!("{e}").contains("no real image"));
    }

    // ── Struct field access ──────────────────────────────────────────────

    #[test]
    fn test_ray_trace_result_fields() {
        let surfaces = vec![OpticalSurface { n1: 1.0, n2: 1.5 }];
        let r = trace_ray(&surfaces, 10.0 * DEG).expect("ok");
        assert_eq!(r.incidence_angles.len(), 1);
        assert_eq!(r.refraction_angles.len(), 1);
        assert!(r.final_angle < 10.0 * DEG); // bends toward normal
    }

    #[test]
    fn test_diffraction_result_maxima_symmetric() {
        let r = diffraction_grating(1e-6, 500e-9, 10).expect("ok");
        // For each positive order there should be a matching negative order
        for &(m, angle) in &r.maxima {
            if m > 0 {
                let neg = r.maxima.iter().find(|&&(mm, _)| mm == -m);
                assert!(neg.is_some());
                // Angles should be opposite in sign
                let neg_angle = neg.map(|&(_, a)| a).unwrap_or(0.0);
                assert!((angle + neg_angle).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_thin_lens_result_fields() {
        let r = thin_lens(10.0, 30.0).expect("ok");
        assert!(r.image_distance > 0.0);
        assert!(r.magnification < 0.0); // inverted real image
    }
}
