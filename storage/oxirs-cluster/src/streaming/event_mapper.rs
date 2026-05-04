//! Schema mapping from streaming events to Raft commands.
//!
//! The cluster sink converts each [`crate::stream_integration::StreamMessage`]
//! into a vector of [`crate::raft::RdfCommand`] entries through an
//! [`EventMapper`]. The default implementation
//! ([`DefaultEventMapper`]) is a one-to-one passthrough that emits an
//! `Insert` command per triple in an `Insert` message, a `Delete` command
//! per triple in a `Delete` message, and a single `Clear` for `Truncate`.
//!
//! Custom mappers can override this to apply schema reshaping (graph rewriting,
//! IRI canonicalisation, predicate filtering, …).
//!
//! ## Example
//!
//! ```ignore
//! use oxirs_cluster::streaming::{DefaultEventMapper, EventMapper};
//! use oxirs_cluster::stream_integration::{StreamMessage, StreamTriple};
//!
//! let mapper = DefaultEventMapper::default();
//! let msg = StreamMessage::insert(
//!     "rdf-stream",
//!     1,
//!     vec![StreamTriple::new("http://s", "http://p", "\"o\"")],
//! );
//! let cmds = mapper.map_message(&msg).expect("valid");
//! assert_eq!(cmds.len(), 1);
//! ```

use thiserror::Error;

use crate::raft::RdfCommand;
use crate::stream_integration::{MutationOp, StreamMessage};

/// Errors that can occur during schema mapping.
#[derive(Debug, Error, Clone)]
pub enum MapperError {
    /// A triple was rejected by the mapper (e.g. failed validation).
    #[error("rejected triple: {0}")]
    RejectedTriple(String),
    /// An unsupported event type was passed to the mapper.
    #[error("unsupported event type: {0}")]
    Unsupported(String),
}

/// Pluggable mapper from streaming events to Raft commands.
pub trait EventMapper: Send + Sync {
    /// Maps a single [`StreamMessage`] into zero or more [`RdfCommand`] entries.
    fn map_message(&self, msg: &StreamMessage) -> Result<Vec<RdfCommand>, MapperError>;

    /// Maps a batch of messages by concatenating per-message results.
    /// Default impl preserves cross-message ordering.
    fn map_batch(&self, msgs: &[StreamMessage]) -> Result<Vec<RdfCommand>, MapperError> {
        let mut out = Vec::new();
        for m in msgs {
            out.extend(self.map_message(m)?);
        }
        Ok(out)
    }
}

/// Default identity mapper: each triple emits a matching `Insert` / `Delete`,
/// each `Truncate` emits a single `Clear`.
///
/// Note: the cluster's `RdfCommand::Clear` is graph-agnostic. If a `Truncate`
/// targets a specific named graph, the default mapper still emits `Clear` —
/// the named-graph dimension is currently dropped because the underlying
/// state machine does not yet model named graphs. Callers that need
/// graph-scoped truncation should provide a custom mapper.
#[derive(Debug, Default, Clone)]
pub struct DefaultEventMapper {
    /// If `true`, triples whose subject/predicate/object is empty are
    /// rejected with [`MapperError::RejectedTriple`].
    /// If `false`, empty fields fall through unchanged.
    pub strict: bool,
}

impl DefaultEventMapper {
    /// Creates a mapper that rejects empty-field triples.
    pub fn strict() -> Self {
        Self { strict: true }
    }

    /// Creates a mapper that passes empty-field triples through.
    pub fn lenient() -> Self {
        Self { strict: false }
    }
}

impl EventMapper for DefaultEventMapper {
    fn map_message(&self, msg: &StreamMessage) -> Result<Vec<RdfCommand>, MapperError> {
        match &msg.op {
            MutationOp::Insert => {
                let mut out = Vec::with_capacity(msg.triples.len());
                for t in &msg.triples {
                    if self.strict
                        && (t.subject.is_empty() || t.predicate.is_empty() || t.object.is_empty())
                    {
                        return Err(MapperError::RejectedTriple(format!(
                            "empty field in <{},{},{}>",
                            t.subject, t.predicate, t.object
                        )));
                    }
                    out.push(RdfCommand::Insert {
                        subject: t.subject.clone(),
                        predicate: t.predicate.clone(),
                        object: t.object.clone(),
                    });
                }
                Ok(out)
            }
            MutationOp::Delete => {
                let mut out = Vec::with_capacity(msg.triples.len());
                for t in &msg.triples {
                    if self.strict
                        && (t.subject.is_empty() || t.predicate.is_empty() || t.object.is_empty())
                    {
                        return Err(MapperError::RejectedTriple(format!(
                            "empty field in <{},{},{}>",
                            t.subject, t.predicate, t.object
                        )));
                    }
                    out.push(RdfCommand::Delete {
                        subject: t.subject.clone(),
                        predicate: t.predicate.clone(),
                        object: t.object.clone(),
                    });
                }
                Ok(out)
            }
            MutationOp::Truncate { .. } => Ok(vec![RdfCommand::Clear]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream_integration::StreamTriple;

    fn triple(idx: usize) -> StreamTriple {
        StreamTriple::new(
            format!("http://s/{idx}"),
            "http://p/has",
            format!("\"v{idx}\""),
        )
    }

    #[test]
    fn maps_insert_message_one_to_one() {
        let mapper = DefaultEventMapper::default();
        let msg = StreamMessage::insert("rdf", 1, vec![triple(0), triple(1), triple(2)]);
        let cmds = mapper.map_message(&msg).expect("ok");
        assert_eq!(cmds.len(), 3);
        assert!(matches!(cmds[0], RdfCommand::Insert { .. }));
    }

    #[test]
    fn maps_delete_message_one_to_one() {
        let mapper = DefaultEventMapper::default();
        let msg = StreamMessage::delete("rdf", 1, vec![triple(0), triple(1)]);
        let cmds = mapper.map_message(&msg).expect("ok");
        assert_eq!(cmds.len(), 2);
        assert!(matches!(cmds[1], RdfCommand::Delete { .. }));
    }

    #[test]
    fn maps_truncate_to_single_clear() {
        let mapper = DefaultEventMapper::default();
        let msg = StreamMessage {
            stream_id: "rdf".into(),
            offset: 1,
            op: MutationOp::Truncate { graph: None },
            triples: vec![],
            timestamp_ms: 0,
        };
        let cmds = mapper.map_message(&msg).expect("ok");
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], RdfCommand::Clear));
    }

    #[test]
    fn strict_mapper_rejects_empty_triple() {
        let mapper = DefaultEventMapper::strict();
        let msg = StreamMessage::insert("rdf", 1, vec![StreamTriple::new("", "http://p", "\"o\"")]);
        assert!(matches!(
            mapper.map_message(&msg),
            Err(MapperError::RejectedTriple(_))
        ));
    }

    #[test]
    fn lenient_mapper_passes_empty_triple() {
        let mapper = DefaultEventMapper::lenient();
        let msg = StreamMessage::insert("rdf", 1, vec![StreamTriple::new("", "http://p", "\"o\"")]);
        let cmds = mapper.map_message(&msg).expect("ok");
        assert_eq!(cmds.len(), 1);
    }

    #[test]
    fn map_batch_concatenates() {
        let mapper = DefaultEventMapper::default();
        let m1 = StreamMessage::insert("rdf", 1, vec![triple(0)]);
        let m2 = StreamMessage::insert("rdf", 2, vec![triple(1), triple(2)]);
        let cmds = mapper.map_batch(&[m1, m2]).expect("ok");
        assert_eq!(cmds.len(), 3);
    }

    #[test]
    fn empty_batch_yields_no_commands() {
        let mapper = DefaultEventMapper::default();
        let cmds = mapper.map_batch(&[]).expect("ok");
        assert!(cmds.is_empty());
    }
}
