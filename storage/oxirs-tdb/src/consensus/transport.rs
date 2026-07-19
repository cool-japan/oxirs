//! Network Transport Abstraction for Distributed Consensus/Commit Protocols
//!
//! [`PaxosProposer`](crate::consensus::paxos::PaxosProposer),
//! [`TwoPhaseCoordinator`](crate::transaction::two_phase_commit::TwoPhaseCoordinator),
//! and [`ReplicationManager`](crate::distributed::replication::ReplicationManager)
//! all need to exchange messages with *other* processes over a real network to
//! provide the guarantees their names promise. This module defines the
//! [`NetworkTransport`] trait that abstracts that exchange.
//!
//! # Why this exists
//!
//! Earlier revisions of these protocols simulated every remote call: `send_prepare`,
//! `send_accept`, `request_prepare_vote`, `send_commit_message`, and
//! `send_changes_to_replica` all just `tokio::time::sleep`'d for a few
//! milliseconds and then returned a hardcoded success value, regardless of
//! whether any other node existed, was reachable, or agreed to anything. That
//! meant every "distributed" commit or consensus round always reported success
//! for a network that was never actually contacted — the exact kind of
//! fabricated-success behavior the project's honesty policy forbids.
//!
//! # The fix
//!
//! - Real network I/O (gRPC/HTTP/TCP to other `oxirs-tdb` nodes) is out of
//!   scope for this pass — it requires a wire protocol, service discovery, and
//!   a multi-node test harness that does not exist in this crate yet.
//! - Instead of continuing to fabricate acknowledgements, every production
//!   entry point that drives a distributed protocol now **requires** a
//!   caller-supplied `Arc<dyn NetworkTransport>`. Without one, construction
//!   fails loudly with [`TdbError::DistributedTransportNotConfigured`] rather
//!   than silently pretending consensus/replication succeeded.
//! - The only bundled implementation is [`LoopbackSimulationTransport`], which
//!   is explicitly named and documented as a single-node, in-process
//!   simulation for tests and local development. It always "acknowledges"
//!   because there is only one process — it must never be used to back a real
//!   multi-node deployment.
//!
//! Anyone wiring up a genuine multi-node cluster must provide their own
//! `NetworkTransport` implementation (e.g. backed by gRPC/HTTP) — the trait
//! boundary here is exactly where that plugs in.

use crate::consensus::paxos::{AcceptResponse, Promise, ProposalNumber, ProposalValue};
use crate::error::{Result, TdbError};
use crate::transaction::two_phase_commit::Vote;
use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Abstraction over the network calls a distributed commit/consensus protocol
/// needs to make to a remote node.
///
/// Implementations are expected to perform real network I/O (or, for
/// [`LoopbackSimulationTransport`], explicitly simulate it for a single
/// process). All methods are keyed by `node_id`, an opaque identifier that
/// the caller (Paxos/2PC/3PC/Replication) already tracks per-participant.
#[async_trait]
pub trait NetworkTransport: Send + Sync {
    /// Send a Paxos PREPARE(proposal_number) to `node_id` and return the raw
    /// bytes of its response (encoding is owned by the caller, e.g. the
    /// Paxos module encodes/decodes `Promise`).
    async fn send_prepare(&self, node_id: &str, payload: &[u8]) -> Result<Vec<u8>>;

    /// Send a Paxos ACCEPT(proposal_number, value) to `node_id` and return the
    /// raw bytes of its response.
    async fn send_accept(&self, node_id: &str, payload: &[u8]) -> Result<Vec<u8>>;

    /// Send a 2PC/3PC PREPARE vote request to `node_id`; returns the raw
    /// encoded vote.
    async fn request_prepare_vote(&self, node_id: &str, payload: &[u8]) -> Result<Vec<u8>>;

    /// Send a 2PC/3PC COMMIT decision to `node_id`.
    async fn send_commit(&self, node_id: &str, payload: &[u8]) -> Result<()>;

    /// Send a 2PC/3PC ABORT decision to `node_id`.
    async fn send_abort(&self, node_id: &str, payload: &[u8]) -> Result<()>;

    /// Ship a batch of replication changes to `node_id` and wait for its
    /// acknowledgement.
    async fn send_replication_changes(&self, node_id: &str, payload: &[u8]) -> Result<()>;

    /// Human-readable name of this transport, used in error messages and logs.
    fn name(&self) -> &str;
}

/// A single-process, in-memory simulation of [`NetworkTransport`].
///
/// This performs **no real network I/O**. It exists purely so that
/// unit/integration tests (and local, single-node development) can exercise
/// the Paxos / 2PC / 3PC / replication *protocol logic* without standing up a
/// real cluster. It always "acknowledges" every call after a short simulated
/// delay, which is only a safe approximation of reality when there is
/// genuinely just one process in play.
///
/// # Do not use this to back a real multi-node deployment
///
/// Because it never actually contacts another node, using it in production
/// with `CommitProtocol::Paxos/TwoPhase/ThreePhase` or `ReplicationManager`
/// across real, independent processes would silently recreate the exact
/// fabricated-consensus bug this module was introduced to eliminate. Supply a
/// real network-backed `NetworkTransport` (gRPC/HTTP/TCP) for any deployment
/// spanning more than one process.
pub struct LoopbackSimulationTransport {
    simulated_latency: std::time::Duration,
    calls_made: AtomicU64,
}

impl LoopbackSimulationTransport {
    /// Create a loopback transport with a default 5ms simulated latency.
    pub fn new() -> Self {
        Self {
            simulated_latency: std::time::Duration::from_millis(5),
            calls_made: AtomicU64::new(0),
        }
    }

    /// Create a loopback transport with a custom simulated latency (useful
    /// for making tests deterministic/fast).
    pub fn with_latency(latency: std::time::Duration) -> Self {
        Self {
            simulated_latency: latency,
            calls_made: AtomicU64::new(0),
        }
    }

    /// Wrap in an `Arc` for handing to Paxos/2PC/Replication constructors.
    pub fn arc() -> Arc<dyn NetworkTransport> {
        Arc::new(Self::new())
    }

    /// Number of simulated calls made so far (test observability).
    pub fn calls_made(&self) -> u64 {
        self.calls_made.load(Ordering::Relaxed)
    }

    async fn simulate(&self) {
        self.calls_made.fetch_add(1, Ordering::Relaxed);
        if !self.simulated_latency.is_zero() {
            tokio::time::sleep(self.simulated_latency).await;
        }
    }
}

impl Default for LoopbackSimulationTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl NetworkTransport for LoopbackSimulationTransport {
    async fn send_prepare(&self, node_id: &str, payload: &[u8]) -> Result<Vec<u8>> {
        self.simulate().await;

        // Decode the real ProposalNumber the proposer sent, and construct a
        // real Promise for it. There is no remote acceptor state to consult
        // (single process), so the loopback always promises — this is the
        // single-node approximation documented on the type.
        let (proposal_number, _): (ProposalNumber, usize) =
            oxicode::serde::decode_from_slice(payload, oxicode::config::standard())
                .map_err(|e| TdbError::Deserialization(e.to_string()))?;

        let promise = Promise {
            acceptor_id: node_id.to_string(),
            promised_number: proposal_number,
            accepted_proposal: None,
        };

        oxicode::serde::encode_to_vec(&promise, oxicode::config::standard())
            .map_err(|e| TdbError::Serialization(e.to_string()))
    }

    async fn send_accept(&self, node_id: &str, payload: &[u8]) -> Result<Vec<u8>> {
        self.simulate().await;

        let ((proposal_number, _value), _): ((ProposalNumber, ProposalValue), usize) =
            oxicode::serde::decode_from_slice(payload, oxicode::config::standard())
                .map_err(|e| TdbError::Deserialization(e.to_string()))?;

        let response = AcceptResponse {
            acceptor_id: node_id.to_string(),
            accepted_number: proposal_number,
            accepted: true,
        };

        oxicode::serde::encode_to_vec(&response, oxicode::config::standard())
            .map_err(|e| TdbError::Serialization(e.to_string()))
    }

    async fn request_prepare_vote(&self, _node_id: &str, _payload: &[u8]) -> Result<Vec<u8>> {
        self.simulate().await;

        // Single process: the local participant always votes Yes (there is
        // no independent remote node whose local state could disagree).
        oxicode::serde::encode_to_vec(&Vote::Yes, oxicode::config::standard())
            .map_err(|e| TdbError::Serialization(e.to_string()))
    }

    async fn send_commit(&self, _node_id: &str, _payload: &[u8]) -> Result<()> {
        self.simulate().await;
        Ok(())
    }

    async fn send_abort(&self, _node_id: &str, _payload: &[u8]) -> Result<()> {
        self.simulate().await;
        Ok(())
    }

    async fn send_replication_changes(&self, _node_id: &str, _payload: &[u8]) -> Result<()> {
        self.simulate().await;
        Ok(())
    }

    fn name(&self) -> &str {
        "LoopbackSimulationTransport (single-node simulation, NOT real networking)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn loopback_transport_acknowledges_and_counts_calls() {
        let transport = LoopbackSimulationTransport::with_latency(std::time::Duration::ZERO);

        let proposal_number = ProposalNumber::new(1, 42);
        let prepare_payload =
            oxicode::serde::encode_to_vec(&proposal_number, oxicode::config::standard()).unwrap();
        let response = transport
            .send_prepare("node-1", &prepare_payload)
            .await
            .unwrap();
        let (promise, _): (Promise, usize) =
            oxicode::serde::decode_from_slice(&response, oxicode::config::standard()).unwrap();
        assert_eq!(promise.acceptor_id, "node-1");
        assert_eq!(promise.promised_number, proposal_number);

        let accept_payload = oxicode::serde::encode_to_vec(
            &(proposal_number, ProposalValue::Noop),
            oxicode::config::standard(),
        )
        .unwrap();
        let response = transport
            .send_accept("node-1", &accept_payload)
            .await
            .unwrap();
        let (accept_response, _): (AcceptResponse, usize) =
            oxicode::serde::decode_from_slice(&response, oxicode::config::standard()).unwrap();
        assert!(accept_response.accepted);

        let vote_response = transport
            .request_prepare_vote("node-1", b"txn-001")
            .await
            .unwrap();
        let (vote, _): (Vote, usize) =
            oxicode::serde::decode_from_slice(&vote_response, oxicode::config::standard()).unwrap();
        assert_eq!(vote, Vote::Yes);

        transport.send_commit("node-1", b"txn-001").await.unwrap();
        transport.send_abort("node-1", b"txn-001").await.unwrap();
        transport
            .send_replication_changes("node-1", b"changes")
            .await
            .unwrap();

        assert_eq!(transport.calls_made(), 6);
        assert!(transport.name().contains("simulation"));
    }
}
