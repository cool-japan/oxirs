//! Causal Representation Learning — Tests

#[cfg(test)]
mod tests {
    use crate::causal_representation_learning::CausalRepresentationModel;
    use crate::causal_representation_learning::{
        CausalGraph, CausalRepresentationConfig, CounterfactualQuery, Intervention,
        InterventionType, StructuralEquation,
    };
    use crate::EmbeddingModel;
    use scirs2_core::ndarray_ext::Array1;
    use std::collections::HashMap;

    #[test]
    fn test_causal_representation_config_default() {
        use crate::causal_representation_learning::CausalDiscoveryAlgorithm;
        let config = CausalRepresentationConfig::default();
        assert!(matches!(
            config.causal_discovery.algorithm,
            CausalDiscoveryAlgorithm::PC
        ));
        assert_eq!(config.causal_discovery.significance_threshold, 0.05);
    }

    #[test]
    fn test_causal_graph_creation() {
        let variables = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
        let mut graph = CausalGraph::new(variables);

        graph.add_edge(0, 1, 0.5);
        graph.add_edge(1, 2, 0.8);

        assert_eq!(graph.get_children(0), vec![1]);
        assert_eq!(graph.get_parents(1), vec![0]);
        assert!(graph.is_acyclic());
    }

    #[test]
    fn test_structural_equation_creation() {
        let equation = StructuralEquation::new("Y".to_string(), vec!["X".to_string()]);
        assert_eq!(equation.target, "Y");
        assert_eq!(equation.parents, vec!["X".to_string()]);
    }

    #[test]
    fn test_intervention_creation() {
        let intervention = Intervention::new(
            vec!["X".to_string()],
            Array1::from_vec(vec![1.0]),
            InterventionType::Do,
        );
        assert_eq!(intervention.targets, vec!["X".to_string()]);
        assert!(matches!(
            intervention.intervention_type,
            InterventionType::Do
        ));
    }

    #[test]
    fn test_causal_representation_model_creation() {
        let config = CausalRepresentationConfig::default();
        let model = CausalRepresentationModel::new(config);
        assert_eq!(model.entities.len(), 0);
        assert_eq!(model.causal_graph.variables.len(), 0);
        assert!(!model.is_trained);
    }

    #[tokio::test]
    async fn test_causal_training() {
        let config = CausalRepresentationConfig::default();
        let mut model = CausalRepresentationModel::new(config);

        let mut data1 = HashMap::new();
        data1.insert("X".to_string(), 1.0);
        data1.insert("Y".to_string(), 2.0);
        model.add_observational_data(data1);

        let stats = model.train(Some(5)).await.expect("should succeed");
        assert_eq!(stats.epochs_completed, 5);
        assert!(model.is_trained());
    }

    #[test]
    fn test_causal_discovery() {
        let config = CausalRepresentationConfig::default();
        let mut model = CausalRepresentationModel::new(config);

        let mut data = HashMap::new();
        data.insert("X".to_string(), 1.0);
        data.insert("Y".to_string(), 2.0);
        model.add_observational_data(data);

        let result = model.discover_causal_structure();
        assert!(result.is_ok());
    }

    #[test]
    fn test_counterfactual_query() {
        let config = CausalRepresentationConfig::default();
        let model = CausalRepresentationModel::new(config);

        let mut evidence = HashMap::new();
        evidence.insert("X".to_string(), 1.0);

        let intervention = Intervention::new(
            vec!["X".to_string()],
            Array1::from_vec(vec![2.0]),
            InterventionType::Do,
        );

        let query = CounterfactualQuery {
            factual_evidence: evidence,
            intervention,
            query_variables: vec!["Y".to_string()],
        };

        let result = model.answer_counterfactual(&query);
        assert!(result.is_ok());
    }
}
