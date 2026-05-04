//! Real-TCP-network E2E cluster harness — Phase C of 1000-node scaling.
//!
//! This module provides a lightweight multi-node TCP cluster substrate that
//! proves the gossip and replication primitives work over actual network
//! sockets rather than in-memory channels.  It is intentionally kept small
//! and focused: it is a *test harness*, not a production cluster.
//!
//! ## Architecture
//!
//! ```text
//!  ┌──────────────────────────────────────────┐
//!  │            TcpClusterNetwork              │
//!  │  ┌──────────┐  ┌──────────┐  ┌────────┐ │
//!  │  │  Node 0  │  │  Node 1  │  │  ...   │ │
//!  │  │127.0.0.1:│◄─┤127.0.0.1:│◄─┤        │ │
//!  │  │  :PORT_A │  │  :PORT_B │  │        │ │
//!  │  └────┬─────┘  └────┬─────┘  └───┬────┘ │
//!  │       │  TCP gossip  │            │      │
//!  │       └──────────────┴────────────┘      │
//!  └──────────────────────────────────────────┘
//! ```
//!
//! Each node binds on `127.0.0.1:0` (OS-assigned port), making tests
//! safe for parallel execution with no port conflicts.
//!
//! ## Sub-modules
//!
//! - [`codec`] — `ClusterMessage` enum and `MessageCodec` (length-prefixed framing)
//! - [`node`]  — `TcpClusterNode`, `TcpNodeConfig`, `GossipState`
//! - [`network`] — `TcpClusterNetwork`, `NetworkStats`

pub mod codec;
pub mod network;
pub mod node;

pub use codec::{ClusterMessage, MessageCodec};
pub use network::{NetworkStats, TcpClusterNetwork};
pub use node::{GossipState, TcpClusterNode, TcpNodeConfig, TcpNodeError};
