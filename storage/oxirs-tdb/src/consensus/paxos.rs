//! Paxos Consensus Algorithm Implementation
//!
//! This module implements the Paxos consensus protocol for achieving agreement
//! among distributed nodes in the presence of failures. Paxos is a fault-tolerant
//! consensus algorithm that guarantees safety (agreement) while making progress
//! when a majority of nodes are available.
//!
//! # Protocol Overview
//!
//! Paxos has three primary roles:
//! - **Proposer**: Proposes values to be agreed upon
//! - **Acceptor**: Votes on proposals
//! - **Learner**: Learns the chosen value
//!
//! ## Phase 1: Prepare Phase
//! 1. Proposer selects a proposal number N and sends PREPARE(N) to acceptors
//! 2. Acceptors respond with PROMISE(N) if N is highest they've seen
//! 3. Acceptors also return any previously accepted value
//!
//! ## Phase 2: Accept Phase
//! 1. If proposer receives PROMISE from majority, it sends ACCEPT(N, V)
//! 2. Acceptors accept the proposal if they haven't promised higher N
//! 3. Learners learn the value once majority accepts
//!
//! # Guarantees
//!
//! - **Safety**: Only one value is chosen
//! - **Liveness**: Progress when majority available (not guaranteed in all cases)
//! - **Fault Tolerance**: Tolerates up to f failures with 2f+1 nodes
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_tdb::consensus::paxos::{PaxosProposer, PaxosAcceptor, ProposalValue};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create proposer
//! let mut proposer = PaxosProposer::new("proposer-1".to_string());
//! proposer.add_acceptor("acceptor-1".to_string());
//! proposer.add_acceptor("acceptor-2".to_string());
//! proposer.add_acceptor("acceptor-3".to_string());
//!
//! // Propose a value
//! let value = ProposalValue::Data(vec![1, 2, 3, 4]);
//! let chosen = proposer.propose(value).await?;
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, TdbError};
use anyhow::Context;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

/// Proposal number for Paxos protocol
///
/// Format: (round_number, proposer_id)
/// Comparison: Higher round number wins, tie-broken by proposer_id
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ProposalNumber {
    /// Round number (monotonically increasing)
    pub round: u64,
    /// Proposer ID (for tie-breaking)
    pub proposer_id: u64,
}

impl ProposalNumber {
    /// Create a new proposal number
    pub fn new(round: u64, proposer_id: u64) -> Self {
        Self { round, proposer_id }
    }

    /// Increment round number
    pub fn next_round(&self) -> Self {
        Self {
            round: self.round + 1,
            proposer_id: self.proposer_id,
        }
    }
}

/// Value being proposed in Paxos
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalValue {
    /// No operation
    Noop,
    /// Raw data
    Data(Vec<u8>),
    /// Configuration change
    ConfigChange(String),
    /// Transaction decision
    TxnDecision {
        /// Transaction ID
        txn_id: String,
        /// Whether to commit (true) or abort (false)
        commit: bool,
    },
}

/// Proposal in Paxos protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    /// Proposal number
    pub number: ProposalNumber,
    /// Proposed value
    pub value: ProposalValue,
}

/// Promise response from acceptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Promise {
    /// Acceptor ID
    pub acceptor_id: String,
    /// Promised proposal number
    pub promised_number: ProposalNumber,
    /// Previously accepted proposal (if any)
    pub accepted_proposal: Option<Proposal>,
}

/// Accept response from acceptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptResponse {
    /// Acceptor ID
    pub acceptor_id: String,
    /// Accepted proposal number
    pub accepted_number: ProposalNumber,
    /// Whether the proposal was accepted
    pub accepted: bool,
}

/// Paxos Proposer
///
/// Proposes values and drives the consensus protocol.
pub struct PaxosProposer {
    /// Proposer ID
    id: String,
    /// Numeric proposer ID (for proposal numbers)
    numeric_id: u64,
    /// Current round number
    current_round: Arc<Mutex<u64>>,
    /// Acceptor IDs
    acceptors: Arc<Mutex<HashSet<String>>>,
    /// Promises received in current round
    promises: Arc<Mutex<HashMap<String, Promise>>>,
    /// Statistics
    stats: Arc<Mutex<PaxosProposerStats>>,
}

/// Paxos Proposer Statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PaxosProposerStats {
    /// Total proposals initiated
    pub total_proposals: u64,
    /// Successful consensus
    pub successful_consensus: u64,
    /// Failed proposals
    pub failed_proposals: u64,
    /// Average rounds to consensus
    pub avg_rounds_to_consensus: f64,
    /// Total rounds (for calculating average)
    total_rounds: u64,
}

impl PaxosProposer {
    /// Create a new Paxos proposer
    pub fn new(id: String) -> Self {
        // Generate numeric ID from string hash
        let numeric_id = Self::hash_id(&id);

        Self {
            id,
            numeric_id,
            current_round: Arc::new(Mutex::new(0)),
            acceptors: Arc::new(Mutex::new(HashSet::new())),
            promises: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(PaxosProposerStats::default())),
        }
    }

    /// Hash string ID to numeric ID
    fn hash_id(id: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        hasher.finish()
    }

    /// Add an acceptor
    pub fn add_acceptor(&mut self, acceptor_id: String) {
        self.acceptors.lock().insert(acceptor_id);
    }

    /// Get majority count
    fn majority_count(&self) -> usize {
        let total = self.acceptors.lock().len();
        (total / 2) + 1
    }

    /// Propose a value and achieve consensus
    ///
    /// # Returns
    ///
    /// - `Ok(value)` if consensus achieved
    /// - `Err(_)` if consensus failed
    pub async fn propose(&mut self, initial_value: ProposalValue) -> Result<ProposalValue> {
        {
            let mut stats = self.stats.lock();
            stats.total_proposals += 1;
        }

        let mut value = initial_value;
        let mut rounds = 0;
        const MAX_ROUNDS: u32 = 10;

        // Retry with increasing proposal numbers
        for _ in 0..MAX_ROUNDS {
            rounds += 1;

            // Phase 1: Prepare
            let proposal_number = self.next_proposal_number();
            let prepare_result = self.prepare_phase(proposal_number).await?;

            if !prepare_result.0 {
                // Didn't get majority, retry with higher proposal number
                continue;
            }

            // Check if we need to use a previously accepted value
            if let Some(prev_value) = prepare_result.1 {
                value = prev_value;
            }

            // Phase 2: Accept
            let accept_result = self.accept_phase(proposal_number, value.clone()).await?;

            if accept_result {
                // Consensus achieved!
                let mut stats = self.stats.lock();
                stats.successful_consensus += 1;
                stats.total_rounds += rounds;
                stats.avg_rounds_to_consensus =
                    stats.total_rounds as f64 / stats.successful_consensus as f64;
                return Ok(value);
            }

            // Accept phase failed, retry
        }

        // Failed to achieve consensus after MAX_ROUNDS
        let mut stats = self.stats.lock();
        stats.failed_proposals += 1;

        Err(TdbError::Other(format!(
            "Failed to achieve consensus after {} rounds",
            MAX_ROUNDS
        )))
    }

    /// Get next proposal number
    fn next_proposal_number(&self) -> ProposalNumber {
        let mut round = self.current_round.lock();
        *round += 1;
        ProposalNumber::new(*round, self.numeric_id)
    }

    /// Phase 1: Prepare phase
    ///
    /// # Returns
    ///
    /// - (true, Some(value)) if majority promised and previous value exists
    /// - (true, None) if majority promised with no previous value
    /// - (false, _) if no majority
    async fn prepare_phase(
        &self,
        proposal_number: ProposalNumber,
    ) -> Result<(bool, Option<ProposalValue>)> {
        self.promises.lock().clear();

        let acceptors = self.acceptors.lock().clone();
        let majority = self.majority_count();

        // Send PREPARE to all acceptors
        for acceptor_id in acceptors.iter() {
            let promise = self.send_prepare(acceptor_id, proposal_number).await?;
            self.promises.lock().insert(acceptor_id.clone(), promise);
        }

        // Check if we have majority
        let promises = self.promises.lock();
        if promises.len() < majority {
            return Ok((false, None));
        }

        // Find highest accepted proposal (if any)
        let highest_accepted = promises
            .values()
            .filter_map(|p| p.accepted_proposal.as_ref())
            .max_by_key(|p| p.number);

        let previous_value = highest_accepted.map(|p| p.value.clone());

        Ok((true, previous_value))
    }

    /// Send PREPARE message to acceptor (simulated)
    async fn send_prepare(
        &self,
        acceptor_id: &str,
        proposal_number: ProposalNumber,
    ) -> Result<Promise> {
        // TODO: Implement actual network communication
        tokio::time::sleep(Duration::from_millis(5)).await;

        Ok(Promise {
            acceptor_id: acceptor_id.to_string(),
            promised_number: proposal_number,
            accepted_proposal: None,
        })
    }

    /// Phase 2: Accept phase
    ///
    /// # Returns
    ///
    /// - `true` if majority accepted
    /// - `false` if no majority
    async fn accept_phase(
        &self,
        proposal_number: ProposalNumber,
        value: ProposalValue,
    ) -> Result<bool> {
        let acceptors = self.acceptors.lock().clone();
        let majority = self.majority_count();

        let mut accept_count = 0;

        // Send ACCEPT to all acceptors
        for acceptor_id in acceptors.iter() {
            let response = self
                .send_accept(acceptor_id, proposal_number, value.clone())
                .await?;

            if response.accepted {
                accept_count += 1;
            }
        }

        Ok(accept_count >= majority)
    }

    /// Send ACCEPT message to acceptor (simulated)
    async fn send_accept(
        &self,
        acceptor_id: &str,
        proposal_number: ProposalNumber,
        _value: ProposalValue,
    ) -> Result<AcceptResponse> {
        // TODO: Implement actual network communication
        tokio::time::sleep(Duration::from_millis(5)).await;

        Ok(AcceptResponse {
            acceptor_id: acceptor_id.to_string(),
            accepted_number: proposal_number,
            accepted: true,
        })
    }

    /// Get proposer ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get statistics
    pub fn stats(&self) -> PaxosProposerStats {
        self.stats.lock().clone()
    }

    /// Get acceptor count
    pub fn acceptor_count(&self) -> usize {
        self.acceptors.lock().len()
    }
}

/// Paxos Acceptor
///
/// Votes on proposals and maintains promises.
pub struct PaxosAcceptor {
    /// Acceptor ID
    id: String,
    /// Highest promised proposal number
    promised_number: Arc<RwLock<Option<ProposalNumber>>>,
    /// Accepted proposal
    accepted_proposal: Arc<RwLock<Option<Proposal>>>,
    /// Statistics
    stats: Arc<Mutex<PaxosAcceptorStats>>,
}

/// Paxos Acceptor Statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PaxosAcceptorStats {
    /// Total PREPARE messages received
    pub total_prepares: u64,
    /// Total ACCEPT messages received
    pub total_accepts: u64,
    /// Total promises sent
    pub total_promises: u64,
    /// Total accepts sent
    pub total_accepted: u64,
    /// Total rejections
    pub total_rejections: u64,
}

impl PaxosAcceptor {
    /// Create a new Paxos acceptor
    pub fn new(id: String) -> Self {
        Self {
            id,
            promised_number: Arc::new(RwLock::new(None)),
            accepted_proposal: Arc::new(RwLock::new(None)),
            stats: Arc::new(Mutex::new(PaxosAcceptorStats::default())),
        }
    }

    /// Handle PREPARE message from proposer
    ///
    /// # Returns
    ///
    /// - `Some(Promise)` if proposal number is acceptable
    /// - `None` if proposal number is too low
    pub async fn handle_prepare(&self, proposal_number: ProposalNumber) -> Result<Option<Promise>> {
        let mut stats = self.stats.lock();
        stats.total_prepares += 1;
        drop(stats);

        let mut promised = self.promised_number.write();

        // Check if proposal number is higher than promised
        if let Some(current_promised) = *promised {
            if proposal_number <= current_promised {
                // Reject - already promised to higher number
                let mut stats = self.stats.lock();
                stats.total_rejections += 1;
                return Ok(None);
            }
        }

        // Promise this proposal number
        *promised = Some(proposal_number);

        let mut stats = self.stats.lock();
        stats.total_promises += 1;
        drop(stats);

        // Return promise with any previously accepted proposal
        let accepted = self.accepted_proposal.read().clone();

        Ok(Some(Promise {
            acceptor_id: self.id.clone(),
            promised_number: proposal_number,
            accepted_proposal: accepted,
        }))
    }

    /// Handle ACCEPT message from proposer
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if proposal accepted
    /// - `Ok(false)` if proposal rejected
    pub async fn handle_accept(&self, proposal: Proposal) -> Result<bool> {
        let mut stats = self.stats.lock();
        stats.total_accepts += 1;
        drop(stats);

        let promised = self.promised_number.read();

        // Check if proposal number matches or exceeds promise
        if let Some(promised_num) = *promised {
            if proposal.number < promised_num {
                // Reject - promised to higher number
                let mut stats = self.stats.lock();
                stats.total_rejections += 1;
                return Ok(false);
            }
        }
        drop(promised);

        // Accept the proposal
        *self.accepted_proposal.write() = Some(proposal);

        let mut stats = self.stats.lock();
        stats.total_accepted += 1;

        Ok(true)
    }

    /// Get acceptor ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get currently accepted proposal
    pub fn accepted_proposal(&self) -> Option<Proposal> {
        self.accepted_proposal.read().clone()
    }

    /// Get statistics
    pub fn stats(&self) -> PaxosAcceptorStats {
        self.stats.lock().clone()
    }
}

/// Paxos Learner
///
/// Learns the chosen value once consensus is achieved.
pub struct PaxosLearner {
    /// Learner ID
    id: String,
    /// Accepted proposals from acceptors
    accepted_proposals: Arc<Mutex<HashMap<String, Proposal>>>,
    /// Learned value
    learned_value: Arc<RwLock<Option<ProposalValue>>>,
    /// Required acceptor count for learning
    required_acceptors: usize,
}

impl PaxosLearner {
    /// Create a new Paxos learner
    pub fn new(id: String, total_acceptors: usize) -> Self {
        let required_acceptors = (total_acceptors / 2) + 1;

        Self {
            id,
            accepted_proposals: Arc::new(Mutex::new(HashMap::new())),
            learned_value: Arc::new(RwLock::new(None)),
            required_acceptors,
        }
    }

    /// Receive accepted proposal from acceptor
    ///
    /// # Returns
    ///
    /// - `Some(value)` if consensus achieved
    /// - `None` if more acceptances needed
    pub async fn learn_from_acceptor(
        &self,
        acceptor_id: String,
        proposal: Proposal,
    ) -> Result<Option<ProposalValue>> {
        let mut accepted = self.accepted_proposals.lock();
        accepted.insert(acceptor_id, proposal.clone());

        // Check if we have majority for this proposal
        let count = accepted
            .values()
            .filter(|p| p.number == proposal.number && p.value == proposal.value)
            .count();

        if count >= self.required_acceptors {
            // Consensus achieved!
            *self.learned_value.write() = Some(proposal.value.clone());
            return Ok(Some(proposal.value));
        }

        Ok(None)
    }

    /// Get learner ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get learned value (if any)
    pub fn learned_value(&self) -> Option<ProposalValue> {
        self.learned_value.read().clone()
    }

    /// Check if value has been learned
    pub fn has_learned(&self) -> bool {
        self.learned_value.read().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proposal_number_ordering() {
        let p1 = ProposalNumber::new(1, 100);
        let p2 = ProposalNumber::new(2, 100);
        let p3 = ProposalNumber::new(2, 101);

        assert!(p1 < p2);
        assert!(p2 < p3);
        assert!(p1 < p3);
    }

    #[test]
    fn test_proposal_number_next_round() {
        let p1 = ProposalNumber::new(5, 100);
        let p2 = p1.next_round();

        assert_eq!(p2.round, 6);
        assert_eq!(p2.proposer_id, 100);
    }

    #[tokio::test]
    async fn test_paxos_proposer_creation() {
        let proposer = PaxosProposer::new("proposer-1".to_string());
        assert_eq!(proposer.id(), "proposer-1");
        assert_eq!(proposer.acceptor_count(), 0);
    }

    #[tokio::test]
    async fn test_paxos_proposer_add_acceptors() {
        let mut proposer = PaxosProposer::new("proposer-1".to_string());

        proposer.add_acceptor("acceptor-1".to_string());
        proposer.add_acceptor("acceptor-2".to_string());
        proposer.add_acceptor("acceptor-3".to_string());

        assert_eq!(proposer.acceptor_count(), 3);
        assert_eq!(proposer.majority_count(), 2);
    }

    #[tokio::test]
    async fn test_paxos_proposer_successful_consensus() {
        let mut proposer = PaxosProposer::new("proposer-1".to_string());

        proposer.add_acceptor("acceptor-1".to_string());
        proposer.add_acceptor("acceptor-2".to_string());
        proposer.add_acceptor("acceptor-3".to_string());

        let value = ProposalValue::Data(vec![1, 2, 3, 4]);
        let result = proposer.propose(value.clone()).await.unwrap();

        assert_eq!(result, value);

        let stats = proposer.stats();
        assert_eq!(stats.successful_consensus, 1);
        assert_eq!(stats.total_proposals, 1);
    }

    #[tokio::test]
    async fn test_paxos_acceptor_creation() {
        let acceptor = PaxosAcceptor::new("acceptor-1".to_string());
        assert_eq!(acceptor.id(), "acceptor-1");
        assert!(acceptor.accepted_proposal().is_none());
    }

    #[tokio::test]
    async fn test_paxos_acceptor_handle_prepare() {
        let acceptor = PaxosAcceptor::new("acceptor-1".to_string());

        let proposal_number = ProposalNumber::new(1, 100);
        let promise = acceptor
            .handle_prepare(proposal_number)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(promise.acceptor_id, "acceptor-1");
        assert_eq!(promise.promised_number, proposal_number);
        assert!(promise.accepted_proposal.is_none());

        let stats = acceptor.stats();
        assert_eq!(stats.total_prepares, 1);
        assert_eq!(stats.total_promises, 1);
    }

    #[tokio::test]
    async fn test_paxos_acceptor_reject_lower_prepare() {
        let acceptor = PaxosAcceptor::new("acceptor-1".to_string());

        // First promise
        let p1 = ProposalNumber::new(5, 100);
        acceptor.handle_prepare(p1).await.unwrap();

        // Try lower proposal number
        let p2 = ProposalNumber::new(3, 100);
        let result = acceptor.handle_prepare(p2).await.unwrap();

        assert!(result.is_none(), "Should reject lower proposal number");

        let stats = acceptor.stats();
        assert_eq!(stats.total_rejections, 1);
    }

    #[tokio::test]
    async fn test_paxos_acceptor_handle_accept() {
        let acceptor = PaxosAcceptor::new("acceptor-1".to_string());

        let proposal_number = ProposalNumber::new(1, 100);

        // First prepare
        acceptor.handle_prepare(proposal_number).await.unwrap();

        // Then accept
        let proposal = Proposal {
            number: proposal_number,
            value: ProposalValue::Data(vec![1, 2, 3]),
        };

        let accepted = acceptor.handle_accept(proposal.clone()).await.unwrap();
        assert!(accepted);

        let stats = acceptor.stats();
        assert_eq!(stats.total_accepts, 1);
        assert_eq!(stats.total_accepted, 1);

        assert_eq!(acceptor.accepted_proposal().unwrap().value, proposal.value);
    }

    #[tokio::test]
    async fn test_paxos_acceptor_reject_lower_accept() {
        let acceptor = PaxosAcceptor::new("acceptor-1".to_string());

        // Promise higher number
        let p1 = ProposalNumber::new(5, 100);
        acceptor.handle_prepare(p1).await.unwrap();

        // Try to accept lower number
        let p2 = ProposalNumber::new(3, 100);
        let proposal = Proposal {
            number: p2,
            value: ProposalValue::Data(vec![1, 2, 3]),
        };

        let accepted = acceptor.handle_accept(proposal).await.unwrap();
        assert!(!accepted, "Should reject lower proposal number");
    }

    #[tokio::test]
    async fn test_paxos_learner_creation() {
        let learner = PaxosLearner::new("learner-1".to_string(), 3);
        assert_eq!(learner.id(), "learner-1");
        assert!(!learner.has_learned());
        assert_eq!(learner.required_acceptors, 2);
    }

    #[tokio::test]
    async fn test_paxos_learner_learn_from_majority() {
        let learner = PaxosLearner::new("learner-1".to_string(), 3);

        let proposal_number = ProposalNumber::new(1, 100);
        let value = ProposalValue::Data(vec![1, 2, 3]);
        let proposal = Proposal {
            number: proposal_number,
            value: value.clone(),
        };

        // First acceptor
        let result = learner
            .learn_from_acceptor("acceptor-1".to_string(), proposal.clone())
            .await
            .unwrap();
        assert!(result.is_none(), "Should need more acceptors");

        // Second acceptor (reaches majority)
        let result = learner
            .learn_from_acceptor("acceptor-2".to_string(), proposal.clone())
            .await
            .unwrap();
        assert!(result.is_some(), "Should learn value with majority");
        assert_eq!(result.unwrap(), value);

        assert!(learner.has_learned());
        assert_eq!(learner.learned_value().unwrap(), value);
    }

    #[tokio::test]
    async fn test_proposal_value_types() {
        let noop = ProposalValue::Noop;
        let data = ProposalValue::Data(vec![1, 2, 3]);
        let config = ProposalValue::ConfigChange("new-config".to_string());
        let txn = ProposalValue::TxnDecision {
            txn_id: "txn-001".to_string(),
            commit: true,
        };

        assert_ne!(noop, data);
        assert_ne!(data, config);
        assert_ne!(config, txn);
    }

    #[tokio::test]
    async fn test_paxos_proposer_stats() {
        let mut proposer = PaxosProposer::new("proposer-1".to_string());

        proposer.add_acceptor("acceptor-1".to_string());
        proposer.add_acceptor("acceptor-2".to_string());
        proposer.add_acceptor("acceptor-3".to_string());

        // First proposal
        let value1 = ProposalValue::Data(vec![1, 2, 3]);
        proposer.propose(value1).await.unwrap();

        // Second proposal
        let value2 = ProposalValue::Data(vec![4, 5, 6]);
        proposer.propose(value2).await.unwrap();

        let stats = proposer.stats();
        assert_eq!(stats.total_proposals, 2);
        assert_eq!(stats.successful_consensus, 2);
        assert!(stats.avg_rounds_to_consensus > 0.0);
    }
}
