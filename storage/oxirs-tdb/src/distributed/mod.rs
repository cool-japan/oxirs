//! Distributed Coordination and Replication
//!
//! This module provides comprehensive distributed system support for oxirs-tdb,
//! including transaction coordination, deadlock detection, and replication.
//!
//! # Components
//!
//! - **Transaction Coordinator**: Manages distributed transactions using 2PC, 3PC, and Paxos
//! - **Deadlock Detector**: Detects and resolves distributed deadlocks
//! - **Replication Manager**: Handles master-slave and master-master replication
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────┐
//! │   Distributed Coordination Layer           │
//! │                                            │
//! │  ┌──────────────┐  ┌──────────────────┐   │
//! │  │ Transaction  │  │    Deadlock      │   │
//! │  │ Coordinator  │  │    Detector      │   │
//! │  └──────────────┘  └──────────────────┘   │
//! │                                            │
//! │  ┌──────────────────────────────────────┐ │
//! │  │      Replication Manager             │ │
//! │  │  - Master-Slave                      │ │
//! │  │  - Master-Master                     │ │
//! │  │  - Async/Sync modes                  │ │
//! │  └──────────────────────────────────────┘ │
//! └────────────────────────────────────────────┘
//! ```

/// Transaction coordinator service
pub mod coordinator;

/// Distributed deadlock detection
pub mod deadlock;

/// Database replication (master-slave, master-master)
pub mod replication;

/// Saga pattern for long-running distributed transactions
pub mod saga;

/// Integration layer for distributed features
pub mod integration;

pub use coordinator::{
    CommitProtocol, CoordinatorConfig, CoordinatorStats, CoordinatorTxnState, ParticipantNode,
    TransactionCoordinator, TransactionMetadata,
};

pub use deadlock::{
    DeadlockCycle, DeadlockDetectorConfig, DeadlockStats, DistributedDeadlockDetector, NodeStatus,
    VictimSelectionStrategy, WaitEdge,
};

pub use replication::{
    ReplicaNode, ReplicaRole, ReplicationConfig, ReplicationManager, ReplicationMode,
    ReplicationStats, ReplicationStatus,
};

pub use saga::{
    SagaConfig, SagaOrchestrator, SagaStats, SagaStatus, SagaStep, SagaStrategy, StepStatus,
};

pub use integration::{
    DistributedConfig, DistributedStoreStats, DistributedTdbStore, DistributedTransaction,
    HealthStatus,
};
