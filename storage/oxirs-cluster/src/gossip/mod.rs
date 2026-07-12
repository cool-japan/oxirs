//! Gossip protocol primitives for cluster membership dissemination.
//!
//! This module houses the foundational gossip-protocol building blocks used
//! across the OxiRS cluster layer.  Higher-level adaptive fanout control lives
//! in [`crate::gossip_scaling`]; the simpler static-policy enum lives here.
//!
//! # Sub-modules
//!
//! - [`gossip::fanout`] — `GossipFanout` enum: `Bounded`, `Sqrt`, `Unbounded` fanout
//!   policies with O(1) `resolve(N)` computation.

pub mod fanout;
