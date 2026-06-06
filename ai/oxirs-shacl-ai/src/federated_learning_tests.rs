//! Tests for the federated learning module split.

#[cfg(test)]
mod tests {
    use super::super::federated_learning_privacy::AdvancedDifferentialPrivacy;
    use super::super::federated_learning_trainer::{
        ConsensusManager, FederatedLearningCoordinator,
    };
    use super::super::federated_learning_types::*;
    use uuid::Uuid;

    #[test]
    fn test_federated_node_creation() {
        let addr = ([127, 0, 0, 1], 8080).into();
        let node = FederatedNode::new(addr, PrivacyLevel::Statistical);

        assert_eq!(node.address, addr);
        assert_eq!(node.privacy_level, PrivacyLevel::Statistical);
        assert_eq!(node.reputation, 1.0);
        assert!(node.is_active());
    }

    #[test]
    fn test_privacy_levels() {
        let levels = [
            PrivacyLevel::Open,
            PrivacyLevel::Statistical,
            PrivacyLevel::DifferentialPrivacy { epsilon: 1.0 },
            PrivacyLevel::HomomorphicEncryption,
            PrivacyLevel::SecureMultiParty,
        ];

        assert_eq!(levels.len(), 5);
    }

    #[tokio::test]
    async fn test_federated_coordinator() {
        let addr = ([127, 0, 0, 1], 8080).into();
        let config = FederatedLearningConfig::default();
        let coordinator =
            FederatedLearningCoordinator::new(addr, PrivacyLevel::Statistical, config);

        let stats = coordinator
            .get_federation_stats()
            .await
            .expect("should succeed");
        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.active_nodes, 0);
    }

    #[test]
    fn test_computational_capacity() {
        let capacity = ComputationalCapacity::default();
        assert!(capacity.cpu_cores > 0);
        assert!(capacity.ram_mb > 0);
    }

    #[test]
    fn test_node_trust_score() {
        let addr = ([127, 0, 0, 1], 8080).into();
        let mut node = FederatedNode::new(addr, PrivacyLevel::Statistical);

        let initial_score = node.trust_score();
        assert!((initial_score - 0.6).abs() < 1e-6);

        node.contribution_score = 0.8;
        let updated_score = node.trust_score();
        assert!((updated_score - 0.92).abs() < 1e-6);
    }

    #[test]
    fn test_node_activity_tracking() {
        let addr = ([127, 0, 0, 1], 8080).into();
        let mut node = FederatedNode::new(addr, PrivacyLevel::Statistical);

        assert!(node.is_active());
        node.update_activity();
        assert!(node.is_active());
    }

    #[test]
    fn test_aggregation_strategies_variants() {
        let _fed_avg = AggregationStrategy::FederatedAveraging;
        let _weighted = AggregationStrategy::WeightedAveraging;
    }

    #[test]
    fn test_consensus_algorithm_variants() {
        let _pbft = ConsensusAlgorithm::PBFT;
        let _raft = ConsensusAlgorithm::RAFT;
        let _pos = ConsensusAlgorithm::PoS;
    }

    #[test]
    fn test_privacy_level_variants() {
        assert_eq!(PrivacyLevel::Open, PrivacyLevel::Open);
        assert_eq!(PrivacyLevel::Statistical, PrivacyLevel::Statistical);
        assert_eq!(
            PrivacyLevel::DifferentialPrivacy { epsilon: 1.0 },
            PrivacyLevel::DifferentialPrivacy { epsilon: 1.0 }
        );
    }

    #[test]
    fn test_security_level_variants() {
        let levels = [
            SecurityLevel::Low,
            SecurityLevel::Medium,
            SecurityLevel::High,
        ];
        assert_eq!(levels.len(), 3);
    }

    #[test]
    fn test_noise_mechanism_variants() {
        let mechanisms = [
            NoiseMechanism::Laplace,
            NoiseMechanism::Gaussian,
            NoiseMechanism::Exponential,
        ];
        assert_eq!(mechanisms.len(), 3);
    }

    #[tokio::test]
    async fn test_federated_learning_config_defaults() {
        let config = FederatedLearningConfig::default();
        assert_eq!(config.min_nodes_for_consensus, 3);
        assert!((config.byzantine_tolerance - 0.33).abs() < 1e-6);
        assert!(config.privacy_config.enable_differential_privacy);
    }

    #[test]
    fn test_privacy_config_defaults() {
        let config = PrivacyConfig::default();
        assert!(config.enable_differential_privacy);
        assert_eq!(config.epsilon, 1.0);
    }

    #[tokio::test]
    async fn test_honey_badger_bft_construction() {
        use super::super::federated_learning_trainer::HoneyBadgerBFT;
        let node_count = 4;
        let _hb_bft = HoneyBadgerBFT::new(Uuid::new_v4(), node_count);
    }

    #[test]
    fn test_consensus_manager_new() {
        let mgr = ConsensusManager::new();
        assert!(mgr.voting_history.is_empty());
    }

    #[test]
    fn test_global_model_default() {
        let model = GlobalModel::default();
        assert_eq!(model.version, 1);
        assert!(model.shapes.is_empty());
    }

    #[test]
    fn test_global_model_update_shapes() {
        let mut model = GlobalModel::default();
        let initial_version = model.version;
        model.update_shapes(Vec::new());
        assert_eq!(model.version, initial_version + 1);
    }

    #[test]
    fn test_differential_privacy_budget_exhaustion() {
        let mut dp = AdvancedDifferentialPrivacy::new(0.0001, 1e-5);
        // Budget is effectively zero — privatize_update should fail
        use super::super::federated_learning_types::{
            FederatedUpdate, ModelParameterDelta, TrainingMetadata,
        };
        use std::collections::HashMap;
        let mut update = FederatedUpdate {
            update_id: Uuid::new_v4(),
            node_id: Uuid::new_v4(),
            parameter_delta: ModelParameterDelta {
                deltas: HashMap::new(),
                gradients: HashMap::new(),
                learning_rate: 0.001,
                batch_size: 32,
                local_epochs: 5,
            },
            training_metadata: TrainingMetadata {
                sample_count: 10,
                loss: 0.1,
                accuracy: 0.9,
                training_time: 1.0,
                data_quality: 0.8,
            },
            privacy_proof: None,
            signature: vec![],
            timestamp: std::time::SystemTime::now(),
        };
        // Drain budget
        for _ in 0..100 {
            let _ = dp.privatize_update(&mut update);
        }
        // Should fail after budget is exhausted
        let result = dp.privatize_update(&mut update);
        assert!(result.is_err());
    }
}
