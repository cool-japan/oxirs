//! GPU-accelerated FEM stress assembly dispatcher.
//!
//! When the `gpu` feature is enabled, [`StressAssemblyDispatcher`] attempts
//! to dispatch element-wise stiffness and consistent-mass matrix assembly to
//! the `scirs2_core` GPU backend. When disabled, every method short-circuits
//! to [`GpuError::BackendUnavailable`] so that callers can drop in a CPU
//! fallback without `cfg` plumbing at the call site.
//!
//! The dispatcher is intentionally a thin scaffold: in this round we
//! reproduce CPU semantics on the "GPU side" so behaviour is bit-identical
//! between CPU and GPU paths. This matches the SAMM W3-S12 pattern where
//! the dispatcher is in place ahead of a full kernel implementation.

use super::{GpuError, GpuResult};

/// Element type recognised by the stress-assembly dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FemElementKind {
    /// 1D bar element — 2 nodes, 1 DoF/node.
    Bar1D,
    /// 1D Euler-Bernoulli beam — 2 nodes, 2 DoF/node.
    Beam1D,
    /// 2D constant-strain triangle — 3 nodes, 2 DoF/node.
    Triangle2D,
    /// 2D bilinear quadrilateral — 4 nodes, 2 DoF/node.
    Quad2D,
}

impl FemElementKind {
    /// Number of DoFs per element of this kind.
    #[inline]
    pub fn dofs(self) -> usize {
        match self {
            Self::Bar1D => 2,
            Self::Beam1D => 4,
            Self::Triangle2D => 6,
            Self::Quad2D => 8,
        }
    }
}

/// Description of a single FEM element used during GPU dispatch.
///
/// The dispatcher does not carry node geometry; it only carries the element
/// kind and a per-element scaling factor (typically the product of Young's
/// modulus and characteristic length used to scale the element stiffness).
#[derive(Debug, Clone, PartialEq)]
pub struct GpuElementDescriptor {
    /// Element kind.
    pub kind: FemElementKind,
    /// Scaling factor applied to the element stiffness (Pa·m).
    pub stiffness_scale: f64,
    /// Scaling factor applied to the consistent mass matrix (kg).
    pub mass_scale: f64,
}

impl Default for GpuElementDescriptor {
    fn default() -> Self {
        Self {
            kind: FemElementKind::Bar1D,
            stiffness_scale: 1.0,
            mass_scale: 1.0,
        }
    }
}

/// Output of a stress-assembly dispatch — one entry per element.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuElementContribution {
    /// Element index in the input batch.
    pub element_index: usize,
    /// Trace of the element stiffness matrix (for sanity checks).
    pub stiffness_trace: f64,
    /// Trace of the element mass matrix.
    pub mass_trace: f64,
    /// Number of DoFs contributed by this element.
    pub dofs: usize,
}

/// GPU-accelerated FEM stress / mass assembly dispatcher.
#[derive(Debug, Default)]
pub struct StressAssemblyDispatcher {
    /// Whether the underlying GPU backend is ready to accept work.
    backend_ready: bool,
}

impl StressAssemblyDispatcher {
    /// Create a new dispatcher.
    ///
    /// With the `gpu` feature enabled the backend is reported ready; without
    /// it the dispatcher always reports unavailable.
    pub fn new() -> Self {
        Self {
            backend_ready: super::backend_available(),
        }
    }

    /// Returns `true` when a usable GPU backend is available.
    pub fn is_available(&self) -> bool {
        self.backend_ready
    }

    /// Dispatch element stiffness assembly for `elements`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::BackendUnavailable`] when compiled without the
    /// `gpu` feature, or [`GpuError::InvalidInput`] when any element has a
    /// non-finite scaling factor.
    pub fn dispatch_stiffness_assembly(
        &self,
        elements: &[GpuElementDescriptor],
    ) -> GpuResult<Vec<GpuElementContribution>> {
        validate_elements(elements)?;
        #[cfg(feature = "gpu")]
        {
            if !self.backend_ready {
                return Err(GpuError::BackendUnavailable);
            }
            // Backend ready: produce the same per-element contributions the
            // CPU path would. The trace of an element stiffness matrix is
            // proportional to (stiffness_scale * dofs); we expose that scalar
            // so callers can validate against the CPU reference assembly.
            Ok(elements
                .iter()
                .enumerate()
                .map(|(idx, e)| GpuElementContribution {
                    element_index: idx,
                    stiffness_trace: e.stiffness_scale * e.kind.dofs() as f64,
                    mass_trace: e.mass_scale * e.kind.dofs() as f64,
                    dofs: e.kind.dofs(),
                })
                .collect())
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(GpuError::BackendUnavailable)
        }
    }

    /// Convenience wrapper around [`Self::dispatch_stiffness_assembly`] that
    /// returns just the per-element DoF count (used to size the global
    /// stiffness scatter-add).
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::BackendUnavailable`] when the backend is not ready.
    pub fn element_dof_layout(&self, elements: &[GpuElementDescriptor]) -> GpuResult<Vec<usize>> {
        let contribs = self.dispatch_stiffness_assembly(elements)?;
        Ok(contribs.into_iter().map(|c| c.dofs).collect())
    }

    /// Dispatch the consistent-mass matrix assembly for `elements`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::BackendUnavailable`] when compiled without the
    /// `gpu` feature.
    pub fn dispatch_mass_assembly(&self, elements: &[GpuElementDescriptor]) -> GpuResult<Vec<f64>> {
        let contribs = self.dispatch_stiffness_assembly(elements)?;
        Ok(contribs.into_iter().map(|c| c.mass_trace).collect())
    }
}

fn validate_elements(elements: &[GpuElementDescriptor]) -> GpuResult<()> {
    for (idx, e) in elements.iter().enumerate() {
        if !e.stiffness_scale.is_finite() {
            return Err(GpuError::InvalidInput(format!(
                "element {idx}: stiffness_scale is not finite"
            )));
        }
        if !e.mass_scale.is_finite() {
            return Err(GpuError::InvalidInput(format!(
                "element {idx}: mass_scale is not finite"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_elements() -> Vec<GpuElementDescriptor> {
        vec![
            GpuElementDescriptor {
                kind: FemElementKind::Bar1D,
                stiffness_scale: 1.0e6,
                mass_scale: 2.0,
            },
            GpuElementDescriptor {
                kind: FemElementKind::Triangle2D,
                stiffness_scale: 5.0e5,
                mass_scale: 1.0,
            },
            GpuElementDescriptor {
                kind: FemElementKind::Quad2D,
                stiffness_scale: 7.0e5,
                mass_scale: 0.5,
            },
        ]
    }

    #[test]
    fn dofs_match_element_kind() {
        assert_eq!(FemElementKind::Bar1D.dofs(), 2);
        assert_eq!(FemElementKind::Beam1D.dofs(), 4);
        assert_eq!(FemElementKind::Triangle2D.dofs(), 6);
        assert_eq!(FemElementKind::Quad2D.dofs(), 8);
    }

    #[test]
    fn dispatcher_availability_matches_feature() {
        let d = StressAssemblyDispatcher::new();
        #[cfg(feature = "gpu")]
        assert!(d.is_available());
        #[cfg(not(feature = "gpu"))]
        assert!(!d.is_available());
    }

    #[test]
    fn stiffness_assembly_no_feature_returns_unavailable() {
        let d = StressAssemblyDispatcher::new();
        let elements = sample_elements();
        let result = d.dispatch_stiffness_assembly(&elements);
        #[cfg(not(feature = "gpu"))]
        assert!(matches!(result, Err(GpuError::BackendUnavailable)));
        #[cfg(feature = "gpu")]
        {
            let contribs = result.expect("dispatch should succeed under gpu feature");
            assert_eq!(contribs.len(), elements.len());
            assert_eq!(contribs[0].dofs, 2);
            assert_eq!(contribs[1].dofs, 6);
            assert_eq!(contribs[2].dofs, 8);
        }
    }

    #[test]
    fn mass_assembly_no_feature_returns_unavailable() {
        let d = StressAssemblyDispatcher::new();
        let elements = sample_elements();
        let result = d.dispatch_mass_assembly(&elements);
        #[cfg(not(feature = "gpu"))]
        assert!(matches!(result, Err(GpuError::BackendUnavailable)));
        #[cfg(feature = "gpu")]
        {
            let traces = result.expect("dispatch should succeed under gpu feature");
            assert_eq!(traces.len(), elements.len());
            assert!((traces[0] - 4.0).abs() < 1e-12);
        }
    }

    #[test]
    fn invalid_input_caught_eagerly() {
        let d = StressAssemblyDispatcher::new();
        let bad = vec![GpuElementDescriptor {
            kind: FemElementKind::Bar1D,
            stiffness_scale: f64::NAN,
            mass_scale: 1.0,
        }];
        // Validation runs before the feature gate so this should always fail
        // with InvalidInput regardless of how the crate is compiled.
        let result = d.dispatch_stiffness_assembly(&bad);
        assert!(matches!(result, Err(GpuError::InvalidInput(_))));
    }

    #[test]
    fn element_dof_layout_matches_dofs() {
        let d = StressAssemblyDispatcher::new();
        let elements = sample_elements();
        let result = d.element_dof_layout(&elements);
        #[cfg(not(feature = "gpu"))]
        assert!(matches!(result, Err(GpuError::BackendUnavailable)));
        #[cfg(feature = "gpu")]
        {
            let dofs = result.expect("dispatch should succeed under gpu feature");
            assert_eq!(dofs, vec![2, 6, 8]);
        }
    }
}
