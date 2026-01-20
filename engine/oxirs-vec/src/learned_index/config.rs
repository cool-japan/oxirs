//! Configuration for learned indexes

use serde::{Deserialize, Serialize};

/// Model architecture for learned index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Simple linear model (fast, low accuracy)
    Linear,
    /// Two-layer neural network
    TwoLayer,
    /// Three-layer neural network
    ThreeLayer,
    /// Recursive Model Index (RMI) with multiple stages
    Rmi,
}

/// Configuration for learned index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedIndexConfig {
    /// Model architecture
    pub architecture: ModelArchitecture,

    /// Number of models in each RMI stage
    pub rmi_stages: Vec<usize>,

    /// Hidden layer sizes
    pub hidden_sizes: Vec<usize>,

    /// Error bound multiplier for search range
    pub error_bound_multiplier: f32,

    /// Minimum training examples required
    pub min_training_examples: usize,

    /// Enable hybrid mode (fallback to binary search)
    pub enable_hybrid: bool,

    /// Training configuration
    pub training: TrainingConfig,
}

impl LearnedIndexConfig {
    /// Create default configuration
    pub fn default_config() -> Self {
        Self {
            architecture: ModelArchitecture::TwoLayer,
            rmi_stages: vec![1, 10, 100],
            hidden_sizes: vec![64, 32],
            error_bound_multiplier: 2.0,
            min_training_examples: 1000,
            enable_hybrid: true,
            training: TrainingConfig::default_config(),
        }
    }

    /// Create configuration optimized for speed
    pub fn speed_optimized() -> Self {
        Self {
            architecture: ModelArchitecture::Linear,
            rmi_stages: vec![1, 5, 25],
            hidden_sizes: vec![32],
            error_bound_multiplier: 3.0,
            min_training_examples: 500,
            enable_hybrid: true,
            training: TrainingConfig::speed_optimized(),
        }
    }

    /// Create configuration optimized for accuracy
    pub fn accuracy_optimized() -> Self {
        Self {
            architecture: ModelArchitecture::ThreeLayer,
            rmi_stages: vec![1, 20, 200],
            hidden_sizes: vec![128, 64, 32],
            error_bound_multiplier: 1.5,
            min_training_examples: 5000,
            enable_hybrid: true,
            training: TrainingConfig::accuracy_optimized(),
        }
    }

    /// Create RMI configuration
    pub fn rmi_config() -> Self {
        Self {
            architecture: ModelArchitecture::Rmi,
            rmi_stages: vec![1, 100, 10000],
            hidden_sizes: vec![64, 32],
            error_bound_multiplier: 2.0,
            min_training_examples: 10000,
            enable_hybrid: true,
            training: TrainingConfig::default_config(),
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.rmi_stages.is_empty() {
            return Err("RMI stages cannot be empty".to_string());
        }

        if self.error_bound_multiplier < 1.0 {
            return Err("Error bound multiplier must be >= 1.0".to_string());
        }

        if self.min_training_examples < 10 {
            return Err("Minimum training examples must be >= 10".to_string());
        }

        self.training.validate()?;

        Ok(())
    }
}

impl Default for LearnedIndexConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,

    /// Number of training epochs
    pub num_epochs: usize,

    /// Batch size
    pub batch_size: usize,

    /// Early stopping patience (epochs without improvement)
    pub early_stopping_patience: usize,

    /// Loss function
    pub loss_function: LossFunction,

    /// Optimizer
    pub optimizer: Optimizer,

    /// Use data augmentation
    pub use_data_augmentation: bool,

    /// Validation split (0.0 to 1.0)
    pub validation_split: f32,
}

impl TrainingConfig {
    pub fn default_config() -> Self {
        Self {
            learning_rate: 0.001,
            num_epochs: 100,
            batch_size: 32,
            early_stopping_patience: 10,
            loss_function: LossFunction::MeanSquaredError,
            optimizer: Optimizer::Adam,
            use_data_augmentation: false,
            validation_split: 0.2,
        }
    }

    pub fn speed_optimized() -> Self {
        Self {
            learning_rate: 0.01,
            num_epochs: 20,
            batch_size: 128,
            early_stopping_patience: 5,
            loss_function: LossFunction::MeanAbsoluteError,
            optimizer: Optimizer::Sgd,
            use_data_augmentation: false,
            validation_split: 0.1,
        }
    }

    pub fn accuracy_optimized() -> Self {
        Self {
            learning_rate: 0.0001,
            num_epochs: 200,
            batch_size: 16,
            early_stopping_patience: 20,
            loss_function: LossFunction::Huber,
            optimizer: Optimizer::Adam,
            use_data_augmentation: true,
            validation_split: 0.3,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err("Learning rate must be in (0, 1]".to_string());
        }

        if self.num_epochs == 0 {
            return Err("Number of epochs must be > 0".to_string());
        }

        if self.batch_size == 0 {
            return Err("Batch size must be > 0".to_string());
        }

        if self.validation_split < 0.0 || self.validation_split >= 1.0 {
            return Err("Validation split must be in [0, 1)".to_string());
        }

        Ok(())
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Loss function for training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossFunction {
    /// Mean Squared Error
    MeanSquaredError,
    /// Mean Absolute Error
    MeanAbsoluteError,
    /// Huber loss (robust to outliers)
    Huber,
    /// Quantile loss
    Quantile,
}

/// Optimizer for training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Optimizer {
    /// Stochastic Gradient Descent
    Sgd,
    /// Adam optimizer
    Adam,
    /// RMSprop
    RmsProp,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LearnedIndexConfig::default_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.architecture, ModelArchitecture::TwoLayer);
    }

    #[test]
    fn test_speed_optimized() {
        let config = LearnedIndexConfig::speed_optimized();
        assert!(config.validate().is_ok());
        assert_eq!(config.architecture, ModelArchitecture::Linear);
    }

    #[test]
    fn test_accuracy_optimized() {
        let config = LearnedIndexConfig::accuracy_optimized();
        assert!(config.validate().is_ok());
        assert_eq!(config.architecture, ModelArchitecture::ThreeLayer);
    }

    #[test]
    fn test_training_config_validation() {
        let mut config = TrainingConfig::default_config();
        assert!(config.validate().is_ok());

        config.learning_rate = 0.0;
        assert!(config.validate().is_err());

        config.learning_rate = 0.001;
        config.validation_split = 1.0;
        assert!(config.validate().is_err());
    }
}
