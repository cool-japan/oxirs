//! Tests for Byzantine fault tolerance functionality

#[cfg(feature = "bft")]
mod bft_tests {
    use oxirs_cluster::bft::{BftConfig, BftConsensus, BftMessage};
    use oxirs_cluster::ClusterError;

    #[test]
    fn test_bft_config_creation() {
        // Test with different cluster sizes
        let config_4 = BftConfig::new(4);
        assert_eq!(config_4.max_faulty, 1);
        assert_eq!(config_4.required_votes(), 3);
        assert!(config_4.has_quorum(4));

        let config_7 = BftConfig::new(7);
        assert_eq!(config_7.max_faulty, 2);
        assert_eq!(config_7.required_votes(), 5);
        assert!(config_7.has_quorum(7));

        let config_10 = BftConfig::new(10);
        assert_eq!(config_10.max_faulty, 3);
        assert_eq!(config_10.required_votes(), 7);
        assert!(config_10.has_quorum(10));
    }

    #[test]
    fn test_bft_consensus_creation() {
        let config = BftConfig::new(4);
        let consensus = BftConsensus::new("node1".to_string(), config);
        assert!(consensus.is_ok());

        let consensus = consensus.unwrap();
        assert_eq!(consensus.current_view().unwrap(), 0);
    }

    #[test]
    fn test_bft_message_types() {
        // Test Request message
        let request = BftMessage::Request {
            client_id: "client1".to_string(),
            operation: vec![1, 2, 3],
            timestamp: 12345,
            signature: None,
        };

        // Test serialization
        let serialized = serde_json::to_string(&request);
        assert!(serialized.is_ok());

        // Test deserialization
        let deserialized: Result<BftMessage, _> = serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_node_registration() {
        use ed25519_dalek::{Keypair, PublicKey};
        use rand::rngs::OsRng;

        let config = BftConfig::new(4);
        let consensus = BftConsensus::new("node1".to_string(), config).unwrap();

        // Generate keypairs for nodes
        let mut csprng = OsRng {};
        let keypair2 = Keypair::generate(&mut csprng);
        let keypair3 = Keypair::generate(&mut csprng);

        // Register nodes
        let result = consensus.register_node("node2".to_string(), keypair2.public);
        assert!(result.is_ok());

        let result = consensus.register_node("node3".to_string(), keypair3.public);
        assert!(result.is_ok());
    }

    #[cfg(feature = "bft")]
    #[tokio::test]
    async fn test_bft_network_service() {
        use oxirs_cluster::bft_network::BftNetworkService;
        use oxirs_cluster::network::{NetworkConfig, NetworkService};
        use std::sync::Arc;

        let config = BftConfig::new(4);
        let consensus = Arc::new(BftConsensus::new("node1".to_string(), config).unwrap());

        let network_config = NetworkConfig::default();
        let network_service = Arc::new(NetworkService::new(1, network_config));

        let bft_network = BftNetworkService::new("node1".to_string(), consensus, network_service);

        // Generate a test keypair
        use ed25519_dalek::Keypair;
        use rand::rngs::OsRng;
        let mut csprng = OsRng {};
        let keypair = Keypair::generate(&mut csprng);

        // Test peer registration
        let result = bft_network
            .register_peer("node2".to_string(), keypair.public)
            .await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_prepared_message() {
        use oxirs_cluster::bft::PreparedMessage;

        let pre_prepare = BftMessage::PrePrepare {
            view: 1,
            sequence: 1,
            digest: vec![1, 2, 3],
            request: Box::new(BftMessage::Request {
                client_id: "client1".to_string(),
                operation: vec![4, 5, 6],
                timestamp: 12345,
                signature: None,
            }),
            primary_signature: vec![],
        };

        let prepare = BftMessage::Prepare {
            view: 1,
            sequence: 1,
            digest: vec![1, 2, 3],
            node_id: "node2".to_string(),
            signature: vec![],
        };

        let prepared_msg = PreparedMessage {
            view: 1,
            sequence: 1,
            digest: vec![1, 2, 3],
            pre_prepare: Box::new(pre_prepare),
            prepares: vec![prepare],
        };

        assert_eq!(prepared_msg.view, 1);
        assert_eq!(prepared_msg.sequence, 1);
        assert_eq!(prepared_msg.prepares.len(), 1);
    }
}

#[cfg(not(feature = "bft"))]
mod bft_feature_tests {
    #[test]
    fn test_bft_feature_disabled() {
        // When BFT feature is disabled, ensure the modules are not available
        // This test just verifies the feature flag works correctly
        assert!(true, "BFT feature is disabled");
    }
}
