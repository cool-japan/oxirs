//! Consensus Algorithms for Distributed Coordination
//!
//! This module provides implementations of various consensus algorithms
//! used for achieving agreement among distributed nodes in oxirs-tdb.
//!
//! # Algorithms
//!
//! - **Paxos**: Classic consensus algorithm with proven safety guarantees
//! - **Raft**: (Future) Leader-based consensus with understandable design
//! - **Multi-Paxos**: (Future) Optimized Paxos for multiple decisions
//!
//! # Use Cases
//!
//! - Distributed transaction coordination
//! - Configuration management
//! - Leader election
//! - Replicated state machines

/// Paxos consensus algorithm
pub mod paxos;

pub use paxos::{
    PaxosAcceptor, PaxosAcceptorStats, PaxosLearner, PaxosProposer, PaxosProposerStats, Proposal,
    ProposalNumber, ProposalValue,
};
