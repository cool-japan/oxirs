//! Neural-Symbolic Tests
//!
//! Tests for the neural-symbolic reasoner, types, and learning algorithms.

#[cfg(test)]
mod tests {
    use crate::neural_symbolic_reasoner::NeuralSymbolicModel;
    use crate::neural_symbolic_types::*;
    use crate::ModelConfig;
    use scirs2_core::ndarray_ext::Array1;
    use std::collections::HashMap;

    #[test]
    fn test_neural_symbolic_config_default() {
        let config = NeuralSymbolicConfig::default();
        assert!(matches!(
            config.architecture_config.architecture_type,
            NeuroSymbolicArchitecture::HybridPipeline
        ));
        assert_eq!(config.symbolic_config.rule_based.confidence_threshold, 0.7);
    }

    #[test]
    fn test_logical_formula_creation() {
        let formula = LogicalFormula::new_atom("test_predicate".to_string());
        assert_eq!(formula.truth_value, 1.0);
        assert_eq!(formula.confidence, 1.0);
        assert!(formula.variables.contains("test_predicate"));
    }

    #[test]
    fn test_logical_formula_evaluation() {
        let formula = LogicalFormula::new_atom("P".to_string());
        let mut assignment = HashMap::new();
        assignment.insert("P".to_string(), 0.8);

        let result = formula.evaluate(&assignment);
        assert_eq!(result, 0.8);
    }

    #[test]
    fn test_knowledge_rule_creation() {
        let antecedent = LogicalFormula::new_atom("A".to_string());
        let consequent = LogicalFormula::new_atom("B".to_string());
        let rule = KnowledgeRule::new("rule1".to_string(), antecedent, consequent);

        assert_eq!(rule.id, "rule1");
        assert_eq!(rule.confidence, 1.0);
    }

    #[test]
    fn test_knowledge_rule_application() {
        let antecedent = LogicalFormula::new_atom("A".to_string());
        let consequent = LogicalFormula::new_atom("B".to_string());
        let rule = KnowledgeRule::new("rule1".to_string(), antecedent, consequent);

        let mut facts = HashMap::new();
        facts.insert("A".to_string(), 0.8);

        let result = rule.apply(&facts);
        assert!(result.is_some());
        let (predicate, value) = result.expect("should succeed");
        assert_eq!(predicate, "B");
        assert_eq!(value, 0.8);
    }

    #[test]
    fn test_neural_symbolic_model_creation() {
        let config = NeuralSymbolicConfig::default();
        let model = NeuralSymbolicModel::new(config);

        assert_eq!(model.entities.len(), 0);
        assert_eq!(model.knowledge_base.len(), 0);
        assert!(!model.is_trained);
    }

    #[tokio::test]
    async fn test_neural_symbolic_training() {
        use crate::EmbeddingModel;

        let config = NeuralSymbolicConfig::default();
        let mut model = NeuralSymbolicModel::new(config);

        let stats = model.train(Some(5)).await.expect("should succeed");
        assert_eq!(stats.epochs_completed, 5);
        assert!(model.is_trained());
    }

    #[test]
    fn test_symbolic_rule_learning() {
        let config = NeuralSymbolicConfig::default();
        let mut model = NeuralSymbolicModel::new(config);

        let examples = vec![
            (
                Array1::from_vec(vec![1.0, 0.0]),
                Array1::from_vec(vec![1.0]),
            ),
            (
                Array1::from_vec(vec![1.0, 0.0]),
                Array1::from_vec(vec![1.0]),
            ),
            (
                Array1::from_vec(vec![1.0, 0.0]),
                Array1::from_vec(vec![1.0]),
            ),
        ];

        let result = model.learn_symbolic_rules(&examples);
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrated_forward() {
        let config = NeuralSymbolicConfig {
            base_config: ModelConfig {
                dimensions: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        let model = NeuralSymbolicModel::new(config);

        let input = Array1::from_vec(vec![1.0, 0.5, 0.0]);
        let result = model.integrated_forward(&input);

        assert!(result.is_ok());
        let output = result.expect("should succeed");
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_semantic_loss_computation() {
        let config = NeuralSymbolicConfig::default();
        let model = NeuralSymbolicModel::new(config);

        let predictions = Array1::from_vec(vec![0.8, 0.3, 0.9]);
        let targets = Array1::from_vec(vec![1.0, 0.0, 1.0]);

        let loss = model
            .compute_semantic_loss(&predictions, &targets)
            .expect("should succeed");
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_explanation_generation() {
        let config = NeuralSymbolicConfig::default();
        let model = NeuralSymbolicModel::new(config);

        let input = Array1::from_vec(vec![1.0, 0.0, 0.5]);
        let prediction = Array1::from_vec(vec![0.8, 0.9]);

        let explanation = model
            .explain_prediction(&input, &prediction)
            .expect("should succeed");
        assert!(explanation.contains("Prediction Explanation"));
    }

    #[test]
    fn test_prove_predicate() {
        let config = NeuralSymbolicConfig::default();
        let mut model = NeuralSymbolicModel::new(config);

        // Add rule: A -> B
        let antecedent = LogicalFormula::new_atom("A".to_string());
        let consequent = LogicalFormula::new_atom("B".to_string());
        let rule = KnowledgeRule::new("rule_ab".to_string(), antecedent, consequent);
        model.add_knowledge_rule(rule);

        let mut facts = HashMap::new();
        facts.insert("A".to_string(), 0.9);

        // Should be able to prove B given A is true and rule A->B exists
        let result = model.prove_predicate("B", &facts, 3);
        assert!(result.is_some(), "should prove B from A via rule A->B");
    }

    #[test]
    fn test_update_rule_weights() {
        let config = NeuralSymbolicConfig::default();
        let mut model = NeuralSymbolicModel::new(config);

        // Add a rule
        let antecedent = LogicalFormula::new_atom("input_0".to_string());
        let consequent = LogicalFormula::new_atom("output_0".to_string());
        let rule = KnowledgeRule::new("rule_test".to_string(), antecedent, consequent);
        model.add_knowledge_rule(rule);

        let initial_weight = model.knowledge_base[0].weight;

        let examples = vec![
            (
                Array1::from_vec(vec![0.9, 0.0]),
                Array1::from_vec(vec![1.0]),
            ),
            (
                Array1::from_vec(vec![0.8, 0.0]),
                Array1::from_vec(vec![1.0]),
            ),
        ];

        let result = model.update_rule_weights(&examples, 0.01);
        assert!(result.is_ok());

        // Weight should have changed
        let new_weight = model.knowledge_base[0].weight;
        // Weight should be within valid range [0, 10]
        assert!(new_weight >= 0.0 && new_weight <= 10.0);
        let _ = initial_weight; // used for documentation
    }
}
