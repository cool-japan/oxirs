//! Tests for neural pattern learning.

#[cfg(test)]
mod tests {
    use crate::neural_patterns::learning::NeuralPatternLearner;
    use crate::neural_patterns::types::NeuralPatternConfig;

    #[test]
    fn test_learner_creation() {
        let config = NeuralPatternConfig::default();
        let learner = NeuralPatternLearner::new(config);
        assert_eq!(learner.current_learning_rate, learner.config.learning_rate);
    }

    #[test]
    fn test_default_config() {
        let config = NeuralPatternConfig::default();
        assert!(config.embedding_dim > 0);
        assert!(config.attention_heads > 0);
        assert!(config.max_epochs > 0);
    }
}
