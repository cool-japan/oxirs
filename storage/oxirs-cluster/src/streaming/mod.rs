//! # Real-time streaming integration with the cluster Raft log (W3-S9)
//!
//! Wires an external streaming ingest pipeline directly into the cluster's
//! Raft log so streaming events become durable cluster state without going
//! through SPARQL UPDATE.
//!
//! ## Components
//!
//! * [`streaming::StreamSink`] — local trait describing a write-only sink that accepts a
//!   batch of cluster-shaped streaming events. The matching producer side
//!   (e.g. `oxirs-stream`) implements *its* end of the bridge by exposing a
//!   transform that calls into a `StreamSink`. We deliberately keep the trait
//!   local to avoid a circular dependency with `oxirs-stream` (W3-S11 will
//!   pull from this side, not the reverse).
//! * [`streaming::ClusterSink`] — the production [`streaming::StreamSink`] implementation that
//!   proposes each event batch through the [`crate::consensus::ConsensusManager`]
//!   as a sequence of [`crate::raft::RdfCommand`] entries. Read replicas serve
//!   queries against committed events via the existing `RdfApp` query path.
//! * [`streaming::BackpressureBridge`] — exposes a [`streaming::BackpressureSignal`] that upstream
//!   operators can poll. The sink updates the bridge based on the depth of the
//!   pending log queue and the configured high/low watermarks.
//! * [`streaming::EventMapper`] — pluggable transformer from
//!   [`crate::stream_integration::StreamMessage`] to a vector of
//!   [`crate::raft::RdfCommand`]. The default
//!   [`streaming::DefaultEventMapper`] turns inserts/deletes/truncates one-to-one.
//!
//! ## Why not extend `RdfCommand` directly?
//!
//! `RdfCommand` is the on-disk Raft log payload across many crates and tests.
//! Extending it is a larger ABI change than this slice owns. Mapping
//! streaming events down onto the existing closed enum keeps the slice
//! self-contained and reuses the existing state-machine apply path (so
//! committed entries are immediately queryable through
//! `ClusterNode::query_triples`).
//!
//! ## Example
//!
//! ```ignore
//! use oxirs_cluster::streaming::{
//!     ClusterSink, ClusterSinkConfig, DefaultEventMapper, StreamSink,
//! };
//! use oxirs_cluster::stream_integration::{StreamMessage, StreamTriple};
//! use std::sync::Arc;
//!
//! # async fn demo(consensus: std::sync::Arc<oxirs_cluster::consensus::ConsensusManager>) -> anyhow::Result<()> {
//! let mapper = Arc::new(DefaultEventMapper::default());
//! let sink = ClusterSink::new(consensus, mapper, ClusterSinkConfig::default());
//! let events = vec![StreamMessage::insert(
//!     "rdf-stream",
//!     1,
//!     vec![StreamTriple::new("http://s", "http://p", "\"o\"")],
//! )];
//! sink.write_batch(events).await?;
//! # Ok(()) }
//! ```

pub mod backpressure_bridge;
pub mod cluster_sink;
pub mod event_mapper;

pub use backpressure_bridge::{BackpressureBridge, BackpressureConfig, BackpressureSignal};
pub use cluster_sink::{
    ClusterSink, ClusterSinkConfig, ClusterSinkStats, SinkError, SinkResult, StreamSink,
};
pub use event_mapper::{DefaultEventMapper, EventMapper, MapperError};
