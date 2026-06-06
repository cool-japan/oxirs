//! Neural-Symbolic Integration — tests

#[cfg(test)]
mod tests {
    use scirs2_core::ndarray_ext::Array1;
    use std::collections::HashMap;

    use crate::neural_symbolic_integration::{
        KnowledgeRule, LogicalFormula, NeuralSymbolicConfig, NeuralSymbolicModel,
        NeuroSymbolicArchitecture,
    };
    use crate::neural_symbolic_integration_loss::{
        compute_constraint_violation_loss, compute_mse_loss, compute_semantic_loss,
    };
    use crate::ModelConfig;

    #[test]
    fn test_neural_symbolic_config_default_from_companion() {
        let config = NeuralSymbolicConfig::default();
        assert!(matches!(
            config.architecture_config.architecture_type,
            NeuroSymbolicArchitecture::HybridPipeline
        ));
        assert_eq!(config.symbolic_config.rule_based.confidence_threshold, 0.7);
    }

    #[test]
    fn test_logical_formula_creation_companion() {
        let formula = LogicalFormula::new_atom("test_predicate".to_string());
        assert_eq!(formula.truth_value, 1.0);
        assert_eq!(formula.confidence, 1.0);
        assert!(formula.variables.contains("test_predicate"));
    }

    #[test]
    fn test_logical_formula_evaluation_companion() {
        let formula = LogicalFormula::new_atom("P".to_string());
        let mut assignment = HashMap::new();
        assignment.insert("P".to_string(), 0.8);

        let result = formula.evaluate(&assignment);
        assert_eq!(result, 0.8);
    }

    #[test]
    fn test_knowledge_rule_creation_companion() {
        let antecedent = LogicalFormula::new_atom("A".to_string());
        let consequent = LogicalFormula::new_atom("B".to_string());
        let rule = KnowledgeRule::new("rule1".to_string(), antecedent, consequent);

        assert_eq!(rule.id, "rule1");
        assert_eq!(rule.confidence, 1.0);
    }

    #[test]
    fn test_knowledge_rule_application_companion() {
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
    fn test_neural_symbolic_model_creation_companion() {
        let config = NeuralSymbolicConfig::default();
        let model = NeuralSymbolicModel::new(config);

        assert_eq!(model.entities.len(), 0);
        assert_eq!(model.knowledge_base.len(), 0);
        assert!(!model.is_trained);
    }

    #[tokio::test]
    async fn test_neural_symbolic_training_companion() {
        use crate::EmbeddingModel;

        let config = NeuralSymbolicConfig::default();
        let mut model = NeuralSymbolicModel::new(config);

        let stats = model.train(Some(5)).await.expect("should succeed");
        assert_eq!(stats.epochs_completed, 5);
        assert!(model.is_trained());
    }

    #[test]
    fn test_symbolic_rule_learning_companion() {
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
    fn test_integrated_forward_companion() {
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
    fn test_semantic_loss_computation_companion() {
        let config = NeuralSymbolicConfig::default();
        let model = NeuralSymbolicModel::new(config);

        let predictions = Array1::from_vec(vec![0.8, 0.3, 0.9]);
        let targets = Array1::from_vec(vec![1.0, 0.0, 1.0]);

        let loss = compute_semantic_loss(
            &predictions,
            &targets,
            &model.constraints,
            &model.knowledge_base,
        )
        .expect("should succeed");
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_explanation_generation_companion() {
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
    fn test_mse_loss_zero_companion() {
        let predictions = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let targets = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let loss = compute_mse_loss(&predictions, &targets);
        assert!(loss.abs() < 1e-6, "MSE should be 0 for identical vectors");
    }

    #[test]
    fn test_constraint_violation_loss_no_constraints_companion() {
        let predictions = Array1::from_vec(vec![0.5, 0.5]);
        let loss = compute_constraint_violation_loss(&predictions, &[]);
        assert_eq!(loss, 0.0, "no constraints gives zero loss");
    }
}
