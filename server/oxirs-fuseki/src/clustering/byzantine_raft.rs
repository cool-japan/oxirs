//! Byzantine Fault Tolerant Raft (BFT-Raft) implementation
//!
//! This module enhances the standard Raft consensus algorithm with Byzantine fault tolerance
//! capabilities to handle malicious or arbitrarily faulty nodes. It implements cryptographic
//! signatures, message verification, and Byzantine behavior detection.
//!
//! Key BFT features:
//! - Cryptographic authentication of all messages
//! - Byzantine behavior detection and node blacklisting
//! - Enhanced quorum requirements (2f+1 for f Byzantine nodes)
//! - Message integrity verification and replay attack prevention
//! - Secure leader election with proof-of-work challenges
//! - Distributed key management for node authentication

use crate::clustering::raft::{
    AppendEntriesRequest, RequestVoteRequest,
    RpcMessage,
};
use crate::error::{FusekiError, FusekiResult};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ring::{
    digest::{Context, SHA256},
    hmac::{self, Key},
    rand::{self, SecureRandom},
    signature::{Ed25519KeyPair, KeyPair, UnparsedPublicKey, ED25519},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    time::{Duration, Instant},
};
use tracing::{info, warn};

/// Maximum number of Byzantine nodes the system can tolerate
const MAX_BYZANTINE_NODES: usize = 10;

/// Proof-of-work difficulty for leader election
const POW_DIFFICULTY: u32 = 4;

/// Message signature expiry time (5 minutes)
const MESSAGE_TTL: Duration = Duration::from_secs(300);

/// Byzantine node detection threshold
const BYZANTINE_THRESHOLD: u32 = 3;

/// BFT-enhanced RPC message with cryptographic authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BftMessage {
    /// Original Raft message
    pub inner: RpcMessage,
    /// Cryptographic signature
    pub signature: Vec<u8>,
    /// Sender's public key identifier
    pub sender_key_id: String,
    /// Message timestamp (for replay protection)
    pub timestamp: DateTime<Utc>,
    /// Nonce for uniqueness
    pub nonce: Vec<u8>,
    /// Proof-of-work (for leader election)
    pub proof_of_work: Option<ProofOfWork>,
}

/// Proof-of-work for Byzantine-resistant leader election
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofOfWork {
    /// Nonce that produces the required hash
    pub nonce: u64,
    /// Target hash with required number of leading zeros
    pub hash: Vec<u8>,
    /// Difficulty level achieved
    pub difficulty: u32,
    /// Computation time (for fairness verification)
    pub compute_time_ms: u64,
}

/// Node's cryptographic identity
#[derive(Debug)]
pub struct NodeIdentity {
    /// Node's unique identifier
    pub node_id: String,
    /// Ed25519 key pair for signing
    pub key_pair: Ed25519KeyPair,
    /// Public key bytes
    pub public_key: Vec<u8>,
    /// HMAC key for message authentication
    pub hmac_key: Key,
}

/// Byzantine behavior evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineEvidence {
    /// Node that exhibited Byzantine behavior
    pub node_id: String,
    /// Type of Byzantine behavior detected
    pub behavior_type: ByzantineBehavior,
    /// Evidence timestamp
    pub detected_at: DateTime<Utc>,
    /// Additional evidence data
    pub evidence_data: Vec<u8>,
    /// Witness nodes that can verify this evidence
    pub witnesses: Vec<String>,
}

/// Types of Byzantine behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ByzantineBehavior {
    /// Double voting in the same term
    DoubleVoting,
    /// Sending conflicting append entries
    ConflictingAppendEntries,
    /// Invalid message signature
    InvalidSignature,
    /// Replay attack detected
    ReplayAttack,
    /// Invalid proof-of-work
    InvalidProofOfWork,
    /// Term manipulation
    TermManipulation,
    /// Log inconsistency attack
    LogInconsistency,
}

/// BFT node state management
#[derive(Debug)]
pub struct BftNodeState {
    /// Node's cryptographic identity
    pub identity: NodeIdentity,
    /// Known public keys of other nodes
    pub known_public_keys: HashMap<String, Vec<u8>>,
    /// Byzantine nodes detection
    pub suspected_byzantine: HashMap<String, ByzantineEvidence>,
    /// Message deduplication (prevents replay attacks)
    pub seen_messages: HashMap<String, DateTime<Utc>>,
    /// Byzantine behavior evidence
    pub byzantine_evidence: Vec<ByzantineEvidence>,
    /// Blacklisted nodes
    pub blacklisted_nodes: HashSet<String>,
    /// Vote tracking for Byzantine detection
    pub vote_tracking: HashMap<u64, HashMap<String, RequestVoteRequest>>,
    /// Append entries tracking
    pub append_entries_tracking: HashMap<String, VecDeque<AppendEntriesRequest>>,
}

impl BftNodeState {
    /// Create new BFT node state
    pub fn new(node_id: String) -> FusekiResult<Self> {
        // Generate Ed25519 key pair for signing
        let rng = rand::SystemRandom::new();
        let key_bytes = Ed25519KeyPair::generate_pkcs8(&rng)
            .map_err(|e| FusekiError::internal(format!("Failed to generate key pair: {e:?}")))?;

        let key_pair = Ed25519KeyPair::from_pkcs8(key_bytes.as_ref())
            .map_err(|e| FusekiError::internal(format!("Failed to parse key pair: {e:?}")))?;

        let public_key = key_pair.public_key().as_ref().to_vec();

        // Generate HMAC key
        let mut hmac_key_bytes = [0u8; 32];
        rng.fill(&mut hmac_key_bytes)
            .map_err(|e| FusekiError::internal(format!("Failed to generate HMAC key: {e:?}")))?;
        let hmac_key = Key::new(hmac::HMAC_SHA256, &hmac_key_bytes);

        let identity = NodeIdentity {
            node_id: node_id.clone(),
            key_pair,
            public_key,
            hmac_key,
        };

        Ok(Self {
            identity,
            known_public_keys: HashMap::new(),
            suspected_byzantine: HashMap::new(),
            seen_messages: HashMap::new(),
            byzantine_evidence: Vec::new(),
            blacklisted_nodes: HashSet::new(),
            vote_tracking: HashMap::new(),
            append_entries_tracking: HashMap::new(),
        })
    }

    /// Sign a message with cryptographic signature
    pub fn sign_message(&self, message: &RpcMessage) -> FusekiResult<BftMessage> {
        let timestamp = Utc::now();

        // Generate nonce for uniqueness
        let rng = rand::SystemRandom::new();
        let mut nonce = [0u8; 16];
        rng.fill(&mut nonce)
            .map_err(|e| FusekiError::internal(format!("Failed to generate nonce: {e:?}")))?;

        // Serialize message for signing
        let message_bytes = bincode::serialize(message)
            .map_err(|e| FusekiError::internal(format!("Failed to serialize message: {e}")))?;

        // Create message hash including timestamp and nonce
        let mut context = Context::new(&SHA256);
        context.update(&message_bytes);
        context.update(&timestamp.timestamp().to_le_bytes());
        context.update(&nonce);
        let message_digest = context.finish();

        // Sign the hash
        let signature = self.identity.key_pair.sign(message_digest.as_ref());

        Ok(BftMessage {
            inner: message.clone(),
            signature: signature.as_ref().to_vec(),
            sender_key_id: self.identity.node_id.clone(),
            timestamp,
            nonce: nonce.to_vec(),
            proof_of_work: None,
        })
    }

    /// Verify message signature and detect Byzantine behavior
    pub fn verify_message(&mut self, bft_message: &BftMessage) -> FusekiResult<bool> {
        // Check if message is too old (replay attack protection)
        let age = Utc::now() - bft_message.timestamp;
        if age > chrono::Duration::from_std(MESSAGE_TTL).unwrap() {
            self.record_byzantine_behavior(
                &bft_message.sender_key_id,
                ByzantineBehavior::ReplayAttack,
                format!("Message too old: {} seconds", age.num_seconds()).into_bytes(),
            );
            return Ok(false);
        }

        // Check for message replay
        let message_id = self.compute_message_id(bft_message)?;
        if let Some(&seen_time) = self.seen_messages.get(&message_id) {
            self.record_byzantine_behavior(
                &bft_message.sender_key_id,
                ByzantineBehavior::ReplayAttack,
                format!("Duplicate message: {message_id}").into_bytes(),
            );
            return Ok(false);
        }

        // Get sender's public key
        let public_key_bytes = self
            .known_public_keys
            .get(&bft_message.sender_key_id)
            .ok_or_else(|| FusekiError::authentication("Unknown sender"))?;

        let public_key = UnparsedPublicKey::new(&ED25519, public_key_bytes);

        // Recreate message hash
        let message_bytes = bincode::serialize(&bft_message.inner)
            .map_err(|e| FusekiError::internal(format!("Failed to serialize message: {e}")))?;

        let mut context = Context::new(&SHA256);
        context.update(&message_bytes);
        context.update(&bft_message.timestamp.timestamp().to_le_bytes());
        context.update(&bft_message.nonce);
        let message_digest = context.finish();

        // Verify signature
        match public_key.verify(message_digest.as_ref(), &bft_message.signature) {
            Ok(()) => {
                // Record message as seen
                self.seen_messages.insert(message_id, bft_message.timestamp);

                // Check for Byzantine behavior patterns
                self.detect_byzantine_patterns(bft_message)?;

                Ok(true)
            }
            Err(_) => {
                self.record_byzantine_behavior(
                    &bft_message.sender_key_id,
                    ByzantineBehavior::InvalidSignature,
                    b"Signature verification failed".to_vec(),
                );
                Ok(false)
            }
        }
    }

    /// Detect Byzantine behavior patterns
    fn detect_byzantine_patterns(&mut self, bft_message: &BftMessage) -> FusekiResult<()> {
        match &bft_message.inner {
            RpcMessage::RequestVote(vote_req) => {
                self.check_double_voting(vote_req)?;
            }
            RpcMessage::AppendEntries(append_req) => {
                self.check_conflicting_append_entries(append_req)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Check for double voting (Byzantine behavior)
    fn check_double_voting(&mut self, vote_req: &RequestVoteRequest) -> FusekiResult<()> {
        // Check if this node has already voted for a different candidate in this term
        let previous_candidate_id = {
            let term_votes = self
                .vote_tracking
                .entry(vote_req.term)
                .or_default();

            // Use this node's ID as key to track its votes
            let node_id = &self.identity.node_id;

            if let Some(previous_vote) = term_votes.get(node_id) {
                if previous_vote.candidate_id != vote_req.candidate_id {
                    Some(previous_vote.candidate_id.clone())
                } else {
                    None
                }
            } else {
                term_votes.insert(node_id.clone(), vote_req.clone());
                None
            }
        };

        // Now record byzantine behavior if detected (after borrow is dropped)
        if let Some(prev_candidate) = previous_candidate_id {
            let node_id = self.identity.node_id.clone();
            self.record_byzantine_behavior(
                &node_id,
                ByzantineBehavior::DoubleVoting,
                format!(
                    "Double vote in term {}: {} vs {}",
                    vote_req.term, prev_candidate, vote_req.candidate_id
                )
                .into_bytes(),
            );
        }

        Ok(())
    }

    /// Check for conflicting append entries
    fn check_conflicting_append_entries(
        &mut self,
        append_req: &AppendEntriesRequest,
    ) -> FusekiResult<()> {
        // First, check for conflicts and collect information
        let conflict_detected = {
            let entries = self
                .append_entries_tracking
                .entry(append_req.leader_id.clone())
                .or_default();

            // Keep only recent entries
            while entries.len() > 100 {
                entries.pop_front();
            }

            // Check for conflicts
            let has_conflict = entries.iter().any(|previous_req| {
                previous_req.term == append_req.term
                    && previous_req.prev_log_index == append_req.prev_log_index
                    && previous_req.entries.len() != append_req.entries.len()
            });

            entries.push_back(append_req.clone());
            has_conflict
        };

        // Record byzantine behavior if conflict detected (after borrow is dropped)
        if conflict_detected {
            self.record_byzantine_behavior(
                &append_req.leader_id,
                ByzantineBehavior::ConflictingAppendEntries,
                format!(
                    "Conflicting append entries at term {} index {}",
                    append_req.term, append_req.prev_log_index
                )
                .into_bytes(),
            );
        }

        Ok(())
    }

    /// Record Byzantine behavior evidence
    fn record_byzantine_behavior(
        &mut self,
        node_id: &str,
        behavior: ByzantineBehavior,
        evidence: Vec<u8>,
    ) {
        let evidence = ByzantineEvidence {
            node_id: node_id.to_string(),
            behavior_type: behavior,
            detected_at: Utc::now(),
            evidence_data: evidence,
            witnesses: vec![self.identity.node_id.clone()],
        };

        info!("Byzantine behavior detected: {:?}", evidence);

        // Track repeated offenses
        if let Some(existing) = self.suspected_byzantine.get_mut(node_id) {
            existing.witnesses.push(self.identity.node_id.clone());
            existing.witnesses.dedup();
        } else {
            self.suspected_byzantine
                .insert(node_id.to_string(), evidence.clone());
        }

        self.byzantine_evidence.push(evidence);

        // Blacklist node if sufficient evidence
        let evidence_count = self
            .byzantine_evidence
            .iter()
            .filter(|e| e.node_id == node_id)
            .count() as u32;

        if evidence_count >= BYZANTINE_THRESHOLD {
            warn!("Blacklisting Byzantine node: {}", node_id);
            self.blacklisted_nodes.insert(node_id.to_string());
        }
    }

    /// Compute unique message identifier
    fn compute_message_id(&self, bft_message: &BftMessage) -> FusekiResult<String> {
        let mut context = Context::new(&SHA256);
        context.update(&bft_message.signature);
        context.update(bft_message.sender_key_id.as_bytes());
        context.update(&bft_message.timestamp.timestamp().to_le_bytes());
        context.update(&bft_message.nonce);

        let hash = context.finish();
        Ok(hex::encode(hash.as_ref()))
    }

    /// Add a known public key for a node
    pub fn add_public_key(&mut self, node_id: String, public_key: Vec<u8>) {
        self.known_public_keys.insert(node_id, public_key);
    }

    /// Check if a node is blacklisted
    pub fn is_blacklisted(&self, node_id: &str) -> bool {
        self.blacklisted_nodes.contains(node_id)
    }

    /// Get Byzantine evidence for a node
    pub fn get_byzantine_evidence(&self, node_id: &str) -> Option<&ByzantineEvidence> {
        self.suspected_byzantine.get(node_id)
    }

    /// Clean up old messages and evidence
    pub fn cleanup_old_data(&mut self) {
        let cutoff = Utc::now() - chrono::Duration::hours(1);

        // Clean up seen messages
        self.seen_messages
            .retain(|_, &mut timestamp| timestamp > cutoff);

        // Clean up old vote tracking
        self.vote_tracking.clear(); // Simple cleanup - in production, be more selective

        // Clean up old append entries
        for entries in self.append_entries_tracking.values_mut() {
            while entries.len() > 50 {
                entries.pop_front();
            }
        }
    }

    /// Generate proof-of-work for leader election
    pub fn generate_proof_of_work(&self, term: u64, candidate: &str) -> FusekiResult<ProofOfWork> {
        let start_time = Instant::now();
        let mut nonce = 0u64;

        loop {
            // Create hash input
            let mut context = Context::new(&SHA256);
            context.update(&term.to_le_bytes());
            context.update(candidate.as_bytes());
            context.update(&nonce.to_le_bytes());
            context.update(self.identity.node_id.as_bytes());

            let hash = context.finish();
            let hash_bytes = hash.as_ref();

            let difficulty = count_leading_zeros(hash_bytes);

            if difficulty >= POW_DIFFICULTY {
                let compute_time = start_time.elapsed().as_millis() as u64;
                return Ok(ProofOfWork {
                    nonce,
                    hash: hash_bytes.to_vec(),
                    difficulty,
                    compute_time_ms: compute_time,
                });
            }

            nonce += 1;

            // Prevent infinite loops in tests
            if nonce > 1_000_000 {
                return Err(FusekiError::internal(
                    "Proof of work computation took too long",
                ));
            }
        }
    }

    /// Verify proof-of-work
    pub fn verify_proof_of_work(&self, pow: &ProofOfWork, term: u64, candidate: &str) -> bool {
        // Recreate hash
        let mut context = Context::new(&SHA256);
        context.update(&term.to_le_bytes());
        context.update(candidate.as_bytes());
        context.update(&pow.nonce.to_le_bytes());
        context.update(self.identity.node_id.as_bytes());

        let hash = context.finish();
        let hash_bytes = hash.as_ref();

        // Verify hash matches
        if hash_bytes != pow.hash.as_slice() {
            return false;
        }

        // Verify difficulty
        let actual_difficulty = count_leading_zeros(hash_bytes);
        actual_difficulty >= POW_DIFFICULTY && actual_difficulty == pow.difficulty
    }
}

/// Count leading zeros in a hash (for proof-of-work)
fn count_leading_zeros(hash: &[u8]) -> u32 {
    let mut count = 0;
    for &byte in hash {
        if byte == 0 {
            count += 8;
        } else {
            count += byte.leading_zeros();
            break;
        }
    }
    count
}

/// BFT-enhanced Raft consensus trait
#[async_trait]
pub trait ByzantineFaultTolerantRaft {
    /// Enhanced quorum size calculation for Byzantine fault tolerance
    /// Requires 2f+1 nodes for f Byzantine nodes
    fn bft_quorum_size(&self, total_nodes: usize) -> usize {
        let max_byzantine = std::cmp::min(total_nodes / 3, MAX_BYZANTINE_NODES);
        2 * max_byzantine + 1
    }

    /// Verify that enough honest nodes participated in consensus
    fn verify_bft_quorum(&self, responses: usize, total_nodes: usize) -> bool {
        responses >= self.bft_quorum_size(total_nodes)
    }

    /// Send BFT-authenticated message
    async fn send_bft_message(&self, node_id: &str, message: BftMessage) -> FusekiResult<()>;

    /// Handle incoming BFT message with verification
    async fn handle_bft_message(&mut self, message: BftMessage) -> FusekiResult<()>;

    /// Broadcast Byzantine evidence to all nodes
    async fn broadcast_byzantine_evidence(&self, evidence: ByzantineEvidence) -> FusekiResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bft_node_creation() {
        let node = BftNodeState::new("test_node".to_string()).unwrap();
        assert_eq!(node.identity.node_id, "test_node");
        assert!(!node.identity.public_key.is_empty());
    }

    #[tokio::test]
    async fn test_message_signing_and_verification() {
        let mut node1 = BftNodeState::new("node1".to_string()).unwrap();
        let mut node2 = BftNodeState::new("node2".to_string()).unwrap();

        // Share public keys
        node2.add_public_key("node1".to_string(), node1.identity.public_key.clone());
        node1.add_public_key("node2".to_string(), node2.identity.public_key.clone());

        // Create and sign a message
        let message = RpcMessage::RequestVote(RequestVoteRequest {
            term: 1,
            candidate_id: "node1".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        });

        let signed_message = node1.sign_message(&message).unwrap();

        // Verify the message
        assert!(node2.verify_message(&signed_message).unwrap());
    }

    #[tokio::test]
    async fn test_byzantine_detection() {
        let mut node = BftNodeState::new("test_node".to_string()).unwrap();

        // Simulate double voting
        let vote1 = RequestVoteRequest {
            term: 1,
            candidate_id: "candidate1".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        };

        let vote2 = RequestVoteRequest {
            term: 1,
            candidate_id: "candidate2".to_string(), // Different candidate, same term
            last_log_index: 0,
            last_log_term: 0,
        };

        node.check_double_voting(&vote1).unwrap();
        node.check_double_voting(&vote2).unwrap();

        // Should detect Byzantine behavior
        assert!(!node.byzantine_evidence.is_empty());
    }

    #[tokio::test]
    async fn test_proof_of_work() {
        let node = BftNodeState::new("test_node".to_string()).unwrap();

        let pow = node.generate_proof_of_work(1, "candidate").unwrap();
        assert!(pow.difficulty >= POW_DIFFICULTY);
        assert!(node.verify_proof_of_work(&pow, 1, "candidate"));
    }

    #[test]
    fn test_leading_zeros_count() {
        assert_eq!(count_leading_zeros(&[0, 0, 0xFF]), 16); // Two zero bytes = 16 zeros
        assert_eq!(count_leading_zeros(&[0x80, 0xFF]), 0); // 0x80 = 10000000 = 0 leading zeros
        assert_eq!(count_leading_zeros(&[0x40, 0xFF]), 1); // 0x40 = 01000000 = 1 leading zero
        assert_eq!(count_leading_zeros(&[0x20, 0xFF]), 2); // 0x20 = 00100000 = 2 leading zeros
    }
}
