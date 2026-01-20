//! Byzantine Fault Tolerance (BFT) consensus for untrusted environments
//!
//! This module implements PBFT (Practical Byzantine Fault Tolerance) for RDF stores
//! operating in untrusted environments where nodes may act maliciously.
//!
//! The module is organized into focused sub-modules:
//! - `types`: Core types, configuration, and basic enums
//! - `messages`: BFT message types and protocol definitions
//! - `detection`: Byzantine behavior detection and security systems
//! - `node`: Main BFT node implementation and consensus logic
//! - `state_machine`: RDF state machine for executing operations

// Sub-modules
pub mod detection;
pub mod messages;
pub mod node;
pub mod state_machine;
pub mod types;

// Re-export main types for backward compatibility
pub use detection::{
    ByzantineDetector, CollusionDetector, EquivocationDetector, PartitionDetector, ReplayDetector,
    ResourceMonitor, TimingAnalysis,
};
pub use messages::BftMessage;
pub use node::{BftNode, ConsensusState, NodeStatus};
pub use state_machine::RdfStateMachine;
pub use types::{
    BftConfig, CheckpointProof, NodeId, NodeInfo, ObjectType, OperationResult, Phase,
    PreparedProof, RdfOperation, SequenceNumber, SerializableTriple, ThreatLevel, ViewNumber,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[tokio::test]
    async fn test_bft_consensus_basic() {
        // Create 4 nodes (tolerating 1 Byzantine failure)
        let config = BftConfig::default();

        let nodes = vec![
            NodeInfo {
                id: 0,
                address: "node0".to_string(),
                public_key: None,
            },
            NodeInfo {
                id: 1,
                address: "node1".to_string(),
                public_key: None,
            },
            NodeInfo {
                id: 2,
                address: "node2".to_string(),
                public_key: None,
            },
            NodeInfo {
                id: 3,
                address: "node3".to_string(),
                public_key: None,
            },
        ];

        let node0 = BftNode::new(config.clone(), 0, nodes.clone());

        // Test primary detection
        assert!(node0.is_primary());
        assert_eq!(node0.get_primary(0), 0);
        assert_eq!(node0.get_primary(1), 1);
        assert_eq!(node0.get_primary(2), 2);
        assert_eq!(node0.get_primary(3), 3);
        assert_eq!(node0.get_primary(4), 0); // Wraps around
    }

    #[test]
    fn test_message_digest() {
        let request = BftMessage::Request {
            client_id: "client1".to_string(),
            operation: RdfOperation::Insert(SerializableTriple {
                subject: "http://example.org/s".to_string(),
                predicate: "http://example.org/p".to_string(),
                object: "value".to_string(),
                object_type: ObjectType::Literal {
                    datatype: None,
                    language: None,
                },
            }),
            timestamp: SystemTime::now(),
            signature: None,
        };

        let digest1 = request.digest();
        let digest2 = request.digest();

        assert_eq!(digest1, digest2); // Same message produces same digest
    }

    #[test]
    fn test_state_machine() {
        let mut state_machine = RdfStateMachine::new();

        // Test insert
        let triple = SerializableTriple {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "value".to_string(),
            object_type: ObjectType::Literal {
                datatype: None,
                language: None,
            },
        };

        let result = state_machine
            .execute(RdfOperation::Insert(triple.clone()))
            .unwrap();
        assert!(matches!(result, OperationResult::Success));
        assert_eq!(state_machine.triple_count(), 1);

        // Test digest calculation
        let digest1 = state_machine.calculate_digest();
        let digest2 = state_machine.calculate_digest();
        assert_eq!(digest1, digest2); // Cached digest

        // Test remove
        let result = state_machine.execute(RdfOperation::Remove(triple)).unwrap();
        assert!(matches!(result, OperationResult::Success));
        assert_eq!(state_machine.triple_count(), 0);

        // Digest should change after operation
        let digest3 = state_machine.calculate_digest();
        assert_ne!(digest1, digest3);
    }

    #[test]
    fn test_checkpoint_proof() {
        use std::collections::HashMap;

        let mut proof = CheckpointProof {
            sequence: 100,
            state_digest: vec![1, 2, 3, 4],
            signatures: HashMap::new(),
        };

        // Add signatures
        proof.signatures.insert(0, vec![]);
        proof.signatures.insert(1, vec![]);
        proof.signatures.insert(2, vec![]);

        // With f=1, need 2f+1 = 3 signatures
        assert_eq!(proof.signatures.len(), 3);
    }

    #[test]
    fn test_byzantine_detector_creation() {
        let detector = ByzantineDetector::new(3);
        assert!(!detector.is_suspected(0));
        assert_eq!(detector.get_suspected_nodes().len(), 0);
    }

    #[test]
    fn test_threat_level_assessment() {
        let detector = ByzantineDetector::new(3);

        // New node should have low threat level
        let threat = detector.get_threat_assessment(0);
        assert_eq!(threat, ThreatLevel::Low);
    }

    #[test]
    fn test_message_types() {
        // Test message view extraction
        let prepare = BftMessage::Prepare {
            view: 5,
            sequence: 10,
            digest: vec![1, 2, 3],
            node_id: 1,
        };

        assert_eq!(prepare.view(), Some(5));
        assert_eq!(prepare.sequence(), Some(10));
        assert_eq!(prepare.node_id(), Some(1));
        assert!(prepare.is_consensus_message());
        assert!(!prepare.is_view_change_message());

        let view_change = BftMessage::ViewChange {
            new_view: 6,
            node_id: 2,
            last_sequence: 15,
            checkpoints: vec![],
            prepared_messages: vec![],
        };

        assert_eq!(view_change.view(), Some(6));
        assert_eq!(view_change.node_id(), Some(2));
        assert!(!view_change.is_consensus_message());
        assert!(view_change.is_view_change_message());
    }
}
