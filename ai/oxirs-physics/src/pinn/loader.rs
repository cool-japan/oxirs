//! Loaders for serialised PINN residual models.
//!
//! The plan called for `.npz` via `scirs2_core` but `scirs2-core 0.4.2` does
//! not expose an NPZ writer/reader. Switching to `serde_json` is a
//! deliberate trade-off:
//!
//! - it keeps the dependency surface small (we already pull in `serde_json`),
//! - the model is tiny (≤ 10 K parameters), so the JSON overhead is
//!   negligible,
//! - it avoids hard-coupling to a binary format we cannot validate yet.
//!
//! The serialised structure is
//!
//! ```json
//! {
//!   "config": { ... ResidualModelConfig ... },
//!   "hidden_layers": [ {"weights": [[..]], "biases": [..], "activation": "Relu"}, ... ],
//!   "output_layer": { ... },
//!   "residual_scale": 1.0
//! }
//! ```
//!
//! which is exactly what `serde` derives for [`ResidualModel`].

use std::fs;
use std::path::Path;

use super::residual_model::{PinnError, PinnResult, ResidualModel};

/// Deserialise a [`ResidualModel`] from a JSON byte slice.
pub fn load_residual_model_from_json(bytes: &[u8]) -> PinnResult<ResidualModel> {
    let model: ResidualModel =
        serde_json::from_slice(bytes).map_err(|e| PinnError::Deserialise(e.to_string()))?;
    model.config.validate()?;
    if model.hidden_layers.len() != model.config.hidden_widths.len() {
        return Err(PinnError::InvalidModel(format!(
            "expected {} hidden layers but found {}",
            model.config.hidden_widths.len(),
            model.hidden_layers.len()
        )));
    }
    Ok(model)
}

/// Convenience wrapper that reads `path` from disk.
pub fn load_residual_model_from_path<P: AsRef<Path>>(path: P) -> PinnResult<ResidualModel> {
    let bytes = fs::read(path.as_ref()).map_err(|e| {
        PinnError::Io(format!(
            "failed to read PINN model from {}: {e}",
            path.as_ref().display()
        ))
    })?;
    load_residual_model_from_json(&bytes)
}

/// Serialise a model back to a JSON `Vec<u8>`. Used by the test suite to
/// round-trip a model through the loader.
pub fn dump_residual_model_to_json(model: &ResidualModel) -> PinnResult<Vec<u8>> {
    serde_json::to_vec_pretty(model).map_err(|e| PinnError::Deserialise(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pinn::residual_model::{Activation, DenseLayer, ResidualModel, ResidualModelConfig};
    use std::env::temp_dir;

    fn tiny_model() -> ResidualModel {
        let cfg = ResidualModelConfig {
            input_dim: 2,
            output_dim: 2,
            hidden_widths: vec![2],
            hidden_activation: Activation::Relu,
            output_activation: Activation::Identity,
            description: "tiny".to_string(),
        };
        let h = DenseLayer::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![0.0, 0.0],
            Activation::Relu,
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
    fn round_trip_through_json_bytes() {
        let model = tiny_model();
        let bytes = dump_residual_model_to_json(&model).expect("dump ok");
        let loaded = load_residual_model_from_json(&bytes).expect("load ok");
        assert_eq!(loaded, model);
    }

    #[test]
    fn round_trip_through_temp_file() {
        let model = tiny_model();
        let bytes = dump_residual_model_to_json(&model).expect("dump ok");
        let mut path = temp_dir();
        path.push(format!(
            "oxirs_physics_pinn_loader_{}.json",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("write ok");
        let loaded = load_residual_model_from_path(&path).expect("load ok");
        assert_eq!(loaded, model);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_malformed_json() {
        let result = load_residual_model_from_json(b"not json");
        assert!(matches!(result, Err(PinnError::Deserialise(_))));
    }

    #[test]
    fn rejects_layer_count_mismatch() {
        // build a valid model, then drop a hidden layer
        let mut model = tiny_model();
        model.hidden_layers.clear();
        let bytes = dump_residual_model_to_json(&model).expect("dump ok");
        let result = load_residual_model_from_json(&bytes);
        assert!(matches!(result, Err(PinnError::InvalidModel(_))));
    }

    #[test]
    fn missing_file_returns_io_error() {
        let mut path = temp_dir();
        path.push("oxirs_physics_pinn_loader_does_not_exist_xyz.json");
        let result = load_residual_model_from_path(&path);
        assert!(matches!(result, Err(PinnError::Io(_))));
    }
}
