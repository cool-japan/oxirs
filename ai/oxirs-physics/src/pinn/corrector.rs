//! Online PINN corrector wrapper.
//!
//! [`PinnCorrector`] wraps a loaded [`ResidualModel`] and exposes the public
//! call surface used by the simulator. When the `pinn_correction` feature
//! is disabled every public method returns
//! [`PinnError::FeatureDisabled`] so callers can fall back to the existing
//! CPU pipeline without conditional compilation at the call site.

use super::residual_model::{PinnError, PinnResult, ResidualModel};

/// Online PINN correction wrapper.
#[derive(Debug, Clone)]
pub struct PinnCorrector {
    /// Loaded residual model. Wrapped in `Option` so the corrector can be
    /// instantiated even when the feature is disabled (the loader simply
    /// never populates it).
    model: Option<ResidualModel>,
}

impl Default for PinnCorrector {
    fn default() -> Self {
        Self::empty()
    }
}

impl PinnCorrector {
    /// Construct a corrector with no model attached. Calls to [`Self::apply`]
    /// will return [`PinnError::FeatureDisabled`] until a model is set.
    pub fn empty() -> Self {
        Self { model: None }
    }

    /// Construct a corrector backed by `model`. When the `pinn_correction`
    /// feature is disabled, the model is dropped and the corrector behaves
    /// identically to [`Self::empty`].
    pub fn new(model: ResidualModel) -> Self {
        #[cfg(feature = "pinn_correction")]
        {
            Self { model: Some(model) }
        }
        #[cfg(not(feature = "pinn_correction"))]
        {
            let _ = model;
            Self { model: None }
        }
    }

    /// Returns `true` when a model is loaded and the feature is enabled.
    pub fn is_active(&self) -> bool {
        cfg!(feature = "pinn_correction") && self.model.is_some()
    }

    /// Returns the parameter count when active, otherwise `None`.
    pub fn parameter_count(&self) -> Option<usize> {
        self.model.as_ref().map(|m| m.parameter_count())
    }

    /// Apply the residual correction to `solver_step` for the supplied
    /// `previous_state`. Returns `solver_step + scale * nn(previous_state)`.
    ///
    /// # Errors
    ///
    /// - [`PinnError::FeatureDisabled`] when the feature is off or no model
    ///   is loaded.
    /// - [`PinnError::InputShape`] when `previous_state` does not match the
    ///   model's expected input width.
    /// - [`PinnError::InvalidModel`] when `solver_step` and
    ///   `previous_state` have different shapes.
    pub fn apply(&self, previous_state: &[f64], solver_step: &[f64]) -> PinnResult<Vec<f64>> {
        if !cfg!(feature = "pinn_correction") {
            return Err(PinnError::FeatureDisabled);
        }
        if previous_state.len() != solver_step.len() {
            return Err(PinnError::InvalidModel(format!(
                "previous_state and solver_step have different lengths: {} vs {}",
                previous_state.len(),
                solver_step.len()
            )));
        }
        let model = match self.model.as_ref() {
            Some(m) => m,
            None => return Err(PinnError::FeatureDisabled),
        };
        let raw = model.forward_raw(previous_state)?;
        Ok(solver_step
            .iter()
            .zip(raw.iter())
            .map(|(s, r)| *s + model.residual_scale * *r)
            .collect())
    }

    /// Returns the underlying model (if any) for inspection.
    pub fn model(&self) -> Option<&ResidualModel> {
        self.model.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pinn::residual_model::{Activation, DenseLayer, ResidualModelConfig};

    fn identity_residual_model() -> ResidualModel {
        let cfg = ResidualModelConfig {
            input_dim: 2,
            output_dim: 2,
            hidden_widths: vec![2],
            hidden_activation: Activation::Relu,
            output_activation: Activation::Identity,
            description: "identity".to_string(),
        };
        // Compose a network whose forward_raw is the identity.
        let h = DenseLayer::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![0.0, 0.0],
            Activation::Identity,
        )
        .expect("layer ok");
        let o = DenseLayer::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![0.0, 0.0],
            Activation::Identity,
        )
        .expect("layer ok");
        ResidualModel::from_layers(cfg, vec![h, o]).expect("model ok")
    }

    #[test]
    fn empty_corrector_is_inactive() {
        let c = PinnCorrector::empty();
        assert!(!c.is_active());
        assert_eq!(c.parameter_count(), None);
    }

    #[test]
    fn apply_without_feature_returns_disabled() {
        let model = identity_residual_model();
        let c = PinnCorrector::new(model);
        let prev = vec![1.0, 2.0];
        let step = vec![3.0, 4.0];
        let result = c.apply(&prev, &step);
        #[cfg(feature = "pinn_correction")]
        {
            // Identity model: nn(prev) = prev, so corrected = step + prev.
            let v = result.expect("apply ok");
            assert_eq!(v, vec![3.0 + 1.0, 4.0 + 2.0]);
            assert!(c.is_active());
        }
        #[cfg(not(feature = "pinn_correction"))]
        {
            assert!(matches!(result, Err(PinnError::FeatureDisabled)));
            assert!(!c.is_active());
        }
    }

    #[test]
    fn shape_mismatch_between_previous_and_step() {
        let model = identity_residual_model();
        let c = PinnCorrector::new(model);
        let result = c.apply(&[1.0, 2.0], &[1.0, 2.0, 3.0]);
        #[cfg(feature = "pinn_correction")]
        assert!(matches!(result, Err(PinnError::InvalidModel(_))));
        #[cfg(not(feature = "pinn_correction"))]
        assert!(matches!(result, Err(PinnError::FeatureDisabled)));
    }
}
