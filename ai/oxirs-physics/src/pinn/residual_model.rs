//! Tiny ResNet-style residual network used as a PINN correction term.
//!
//! The model is implemented as a hand-written feed-forward network so that
//! enabling the `pinn_correction` feature does not pull in a heavy ML
//! runtime. Layers are stored row-major (`weights[row][col]`) and the
//! forward pass is plain Rust — no SIMD, no parallelism — because the
//! design budget is < 10 000 parameters.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Result type for PINN operations.
pub type PinnResult<T> = Result<T, PinnError>;

/// Errors produced by the PINN module.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PinnError {
    /// The `pinn_correction` feature is not compiled in.
    #[error("PINN correction feature is disabled; recompile with `--features pinn_correction`")]
    FeatureDisabled,

    /// The model file is malformed.
    #[error("invalid PINN model: {0}")]
    InvalidModel(String),

    /// The model could not be deserialised.
    #[error("PINN model deserialisation error: {0}")]
    Deserialise(String),

    /// The input vector has the wrong shape.
    #[error("PINN input mismatch: expected {expected}, got {actual}")]
    InputShape {
        /// Number of features the network expects.
        expected: usize,
        /// Number of features the caller passed.
        actual: usize,
    },

    /// Filesystem error while loading a serialised model.
    #[error("PINN model I/O error: {0}")]
    Io(String),
}

/// Activation function applied after each hidden layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    /// `f(x) = max(0, x)`
    Relu,
    /// `f(x) = tanh(x)`
    Tanh,
    /// Identity (linear) — used by the output layer.
    Identity,
}

impl Activation {
    fn apply(&self, x: f64) -> f64 {
        match self {
            Self::Relu => x.max(0.0),
            Self::Tanh => x.tanh(),
            Self::Identity => x,
        }
    }
}

/// Static configuration of a residual model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidualModelConfig {
    /// Number of input features.
    pub input_dim: usize,
    /// Number of output features (must equal `input_dim` so that the skip
    /// connection is shape-compatible).
    pub output_dim: usize,
    /// Hidden layer widths (3 or 4 entries are typical).
    pub hidden_widths: Vec<usize>,
    /// Activation applied after every hidden layer.
    pub hidden_activation: Activation,
    /// Activation applied after the final layer (typically `Identity`).
    pub output_activation: Activation,
    /// Free-form description used by `loader::load_residual_model_from_path`
    /// to print clearer error messages.
    #[serde(default)]
    pub description: String,
}

impl ResidualModelConfig {
    /// Returns the total parameter count for a model of this configuration.
    pub fn parameter_count(&self) -> usize {
        let mut count = 0;
        let mut prev = self.input_dim;
        for &w in &self.hidden_widths {
            count += prev * w + w; // weights + bias
            prev = w;
        }
        count += prev * self.output_dim + self.output_dim;
        count
    }

    /// Sanity-check the configuration before consuming it.
    pub fn validate(&self) -> PinnResult<()> {
        if self.input_dim == 0 || self.output_dim == 0 {
            return Err(PinnError::InvalidModel(
                "input_dim and output_dim must be > 0".to_string(),
            ));
        }
        if self.input_dim != self.output_dim {
            return Err(PinnError::InvalidModel(format!(
                "residual skip connection requires input_dim == output_dim; got {}/{}",
                self.input_dim, self.output_dim
            )));
        }
        if self.hidden_widths.is_empty() {
            return Err(PinnError::InvalidModel(
                "hidden_widths must contain at least one layer".to_string(),
            ));
        }
        if self.hidden_widths.contains(&0) {
            return Err(PinnError::InvalidModel(
                "hidden layer widths must all be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// One fully-connected layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseLayer {
    /// Row-major `[out_dim][in_dim]` weights.
    pub weights: Vec<Vec<f64>>,
    /// Bias vector with one entry per output unit.
    pub biases: Vec<f64>,
    /// Activation applied to the layer output.
    pub activation: Activation,
}

impl DenseLayer {
    /// Build a layer with explicit dimensions, validating the shapes.
    pub fn new(
        weights: Vec<Vec<f64>>,
        biases: Vec<f64>,
        activation: Activation,
    ) -> PinnResult<Self> {
        if weights.is_empty() {
            return Err(PinnError::InvalidModel(
                "DenseLayer.weights must be non-empty".to_string(),
            ));
        }
        let in_dim = weights[0].len();
        if in_dim == 0 {
            return Err(PinnError::InvalidModel(
                "DenseLayer.weights[0] must be non-empty".to_string(),
            ));
        }
        for (i, row) in weights.iter().enumerate() {
            if row.len() != in_dim {
                return Err(PinnError::InvalidModel(format!(
                    "DenseLayer.weights row {i} has length {}, expected {}",
                    row.len(),
                    in_dim
                )));
            }
        }
        if biases.len() != weights.len() {
            return Err(PinnError::InvalidModel(format!(
                "DenseLayer.biases has length {}, expected {}",
                biases.len(),
                weights.len()
            )));
        }
        Ok(Self {
            weights,
            biases,
            activation,
        })
    }

    /// Number of input units.
    #[inline]
    pub fn in_dim(&self) -> usize {
        self.weights[0].len()
    }

    /// Number of output units.
    #[inline]
    pub fn out_dim(&self) -> usize {
        self.weights.len()
    }

    /// Forward pass `output = activation(W x + b)`.
    pub fn forward(&self, input: &[f64]) -> PinnResult<Vec<f64>> {
        if input.len() != self.in_dim() {
            return Err(PinnError::InputShape {
                expected: self.in_dim(),
                actual: input.len(),
            });
        }
        let mut out = Vec::with_capacity(self.out_dim());
        for (row, b) in self.weights.iter().zip(self.biases.iter()) {
            let mut acc = *b;
            for (w, x) in row.iter().zip(input.iter()) {
                acc += *w * *x;
            }
            out.push(self.activation.apply(acc));
        }
        Ok(out)
    }
}

/// Residual feed-forward network with a skip connection from input to
/// output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidualModel {
    /// Configuration metadata.
    pub config: ResidualModelConfig,
    /// Hidden layers (the last one feeds into the output layer).
    pub hidden_layers: Vec<DenseLayer>,
    /// Output layer.
    pub output_layer: DenseLayer,
    /// Magnitude scaling applied to the residual *before* it is added to
    /// the input. Lets users dampen the correction without retraining.
    #[serde(default = "default_residual_scale")]
    pub residual_scale: f64,
}

fn default_residual_scale() -> f64 {
    1.0
}

impl ResidualModel {
    /// Build a residual model from a configuration and a list of layers.
    ///
    /// `layers` must contain `config.hidden_widths.len() + 1` entries — the
    /// trailing entry is treated as the output layer.
    pub fn from_layers(config: ResidualModelConfig, layers: Vec<DenseLayer>) -> PinnResult<Self> {
        config.validate()?;
        let expected_layers = config.hidden_widths.len() + 1;
        if layers.len() != expected_layers {
            return Err(PinnError::InvalidModel(format!(
                "residual model expects {expected_layers} layers, got {}",
                layers.len()
            )));
        }

        let mut prev = config.input_dim;
        for (idx, layer) in layers.iter().take(config.hidden_widths.len()).enumerate() {
            if layer.in_dim() != prev {
                return Err(PinnError::InvalidModel(format!(
                    "hidden layer {idx} expects in_dim {prev}, got {}",
                    layer.in_dim()
                )));
            }
            if layer.out_dim() != config.hidden_widths[idx] {
                return Err(PinnError::InvalidModel(format!(
                    "hidden layer {idx} expects out_dim {}, got {}",
                    config.hidden_widths[idx],
                    layer.out_dim()
                )));
            }
            prev = layer.out_dim();
        }
        let output = match layers.last() {
            Some(o) => o,
            None => {
                return Err(PinnError::InvalidModel(
                    "residual model has no output layer".to_string(),
                ))
            }
        };
        if output.in_dim() != prev {
            return Err(PinnError::InvalidModel(format!(
                "output layer expects in_dim {prev}, got {}",
                output.in_dim()
            )));
        }
        if output.out_dim() != config.output_dim {
            return Err(PinnError::InvalidModel(format!(
                "output layer expects out_dim {}, got {}",
                config.output_dim,
                output.out_dim()
            )));
        }

        let (output_layer, hidden) = match layers.split_last() {
            Some((last, rest)) => (last.clone(), rest.to_vec()),
            None => {
                return Err(PinnError::InvalidModel(
                    "residual model has no layers".to_string(),
                ))
            }
        };

        Ok(Self {
            config,
            hidden_layers: hidden,
            output_layer,
            residual_scale: 1.0,
        })
    }

    /// Forward pass: returns just the network output (no skip added).
    pub fn forward_raw(&self, input: &[f64]) -> PinnResult<Vec<f64>> {
        if input.len() != self.config.input_dim {
            return Err(PinnError::InputShape {
                expected: self.config.input_dim,
                actual: input.len(),
            });
        }
        let mut activations = input.to_vec();
        for layer in &self.hidden_layers {
            activations = layer.forward(&activations)?;
        }
        self.output_layer.forward(&activations)
    }

    /// Apply the residual to `input`, returning `input + scale * nn(input)`.
    pub fn apply_residual(&self, input: &[f64]) -> PinnResult<Vec<f64>> {
        let raw = self.forward_raw(input)?;
        Ok(input
            .iter()
            .zip(raw.iter())
            .map(|(x, r)| *x + self.residual_scale * *r)
            .collect())
    }

    /// Replace the residual scale (e.g. to dampen the correction online).
    pub fn with_residual_scale(mut self, scale: f64) -> Self {
        self.residual_scale = scale;
        self
    }

    /// Total trainable parameter count.
    pub fn parameter_count(&self) -> usize {
        let mut count = 0;
        for layer in &self.hidden_layers {
            count += layer.weights.iter().map(|r| r.len()).sum::<usize>();
            count += layer.biases.len();
        }
        count += self
            .output_layer
            .weights
            .iter()
            .map(|r| r.len())
            .sum::<usize>();
        count += self.output_layer.biases.len();
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_layer(dim: usize, activation: Activation) -> DenseLayer {
        let mut weights = vec![vec![0.0; dim]; dim];
        for (i, row) in weights.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        DenseLayer::new(weights, vec![0.0; dim], activation).expect("valid layer")
    }

    fn zero_layer(in_dim: usize, out_dim: usize, activation: Activation) -> DenseLayer {
        DenseLayer::new(
            vec![vec![0.0; in_dim]; out_dim],
            vec![0.0; out_dim],
            activation,
        )
        .expect("valid layer")
    }

    #[test]
    fn dense_layer_forward_shapes() {
        let layer = identity_layer(3, Activation::Identity);
        let out = layer.forward(&[1.0, -2.0, 3.0]).expect("forward ok");
        assert_eq!(out, vec![1.0, -2.0, 3.0]);
    }

    #[test]
    fn dense_layer_relu_clips_negatives() {
        let layer = identity_layer(3, Activation::Relu);
        let out = layer.forward(&[-1.0, 2.0, -3.0]).expect("forward ok");
        assert_eq!(out, vec![0.0, 2.0, 0.0]);
    }

    #[test]
    fn dense_layer_tanh_squash() {
        let layer = identity_layer(2, Activation::Tanh);
        let out = layer.forward(&[100.0, -100.0]).expect("forward ok");
        assert!((out[0] - 1.0).abs() < 1e-9);
        assert!((out[1] + 1.0).abs() < 1e-9);
    }

    #[test]
    fn dense_layer_input_shape_mismatch() {
        let layer = identity_layer(3, Activation::Identity);
        let result = layer.forward(&[1.0]);
        assert!(matches!(result, Err(PinnError::InputShape { .. })));
    }

    fn config_3_relu_id() -> ResidualModelConfig {
        ResidualModelConfig {
            input_dim: 3,
            output_dim: 3,
            hidden_widths: vec![3, 3],
            hidden_activation: Activation::Relu,
            output_activation: Activation::Identity,
            description: "test".to_string(),
        }
    }

    #[test]
    fn residual_model_zero_weights_is_identity_residual() {
        let cfg = config_3_relu_id();
        let layers = vec![
            zero_layer(3, 3, Activation::Relu),
            zero_layer(3, 3, Activation::Relu),
            zero_layer(3, 3, Activation::Identity),
        ];
        let model = ResidualModel::from_layers(cfg, layers).expect("model ok");
        // Skip connection: x + 0 = x
        let input = vec![0.5, -0.7, 1.2];
        let out = model.apply_residual(&input).expect("apply ok");
        assert!((out[0] - input[0]).abs() < 1e-12);
        assert!((out[1] - input[1]).abs() < 1e-12);
        assert!((out[2] - input[2]).abs() < 1e-12);
    }

    #[test]
    fn residual_scale_dampens_correction() {
        let cfg = ResidualModelConfig {
            input_dim: 1,
            output_dim: 1,
            hidden_widths: vec![1],
            hidden_activation: Activation::Identity,
            output_activation: Activation::Identity,
            description: "scale".to_string(),
        };
        let layers = vec![
            DenseLayer::new(vec![vec![1.0]], vec![0.0], Activation::Identity).expect("layer ok"),
            DenseLayer::new(vec![vec![1.0]], vec![0.0], Activation::Identity).expect("layer ok"),
        ];
        let mut model = ResidualModel::from_layers(cfg, layers).expect("model ok");
        // raw forward = x; corrected = x + 1.0 * x = 2x
        let out = model.apply_residual(&[3.0]).expect("apply ok");
        assert!((out[0] - 6.0).abs() < 1e-12);

        // dampen
        model.residual_scale = 0.0;
        let out = model.apply_residual(&[3.0]).expect("apply ok");
        assert!((out[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn validate_rejects_zero_dims() {
        let mut cfg = config_3_relu_id();
        cfg.input_dim = 0;
        assert!(matches!(cfg.validate(), Err(PinnError::InvalidModel(_))));
    }

    #[test]
    fn validate_rejects_input_output_mismatch() {
        let mut cfg = config_3_relu_id();
        cfg.output_dim = 4;
        assert!(matches!(cfg.validate(), Err(PinnError::InvalidModel(_))));
    }

    #[test]
    fn from_layers_rejects_wrong_count() {
        let cfg = config_3_relu_id();
        let layers = vec![identity_layer(3, Activation::Relu)]; // too few
        let result = ResidualModel::from_layers(cfg, layers);
        assert!(matches!(result, Err(PinnError::InvalidModel(_))));
    }

    #[test]
    fn parameter_count_matches_config() {
        let cfg = config_3_relu_id();
        let layers = vec![
            zero_layer(3, 3, Activation::Relu),
            zero_layer(3, 3, Activation::Relu),
            zero_layer(3, 3, Activation::Identity),
        ];
        let model = ResidualModel::from_layers(cfg.clone(), layers).expect("model ok");
        let expected = (3 * 3 + 3) + (3 * 3 + 3) + (3 * 3 + 3);
        assert_eq!(cfg.parameter_count(), expected);
        assert_eq!(model.parameter_count(), expected);
    }

    #[test]
    fn input_shape_mismatch_propagates() {
        let cfg = config_3_relu_id();
        let layers = vec![
            zero_layer(3, 3, Activation::Relu),
            zero_layer(3, 3, Activation::Relu),
            zero_layer(3, 3, Activation::Identity),
        ];
        let model = ResidualModel::from_layers(cfg, layers).expect("model ok");
        let result = model.apply_residual(&[1.0]);
        assert!(matches!(result, Err(PinnError::InputShape { .. })));
    }
}
