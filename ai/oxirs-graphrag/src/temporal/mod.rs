//! Temporal reasoning and time-aware retrieval for GraphRAG

pub mod knowledge_graph;
pub mod temporal_retrieval;

pub use knowledge_graph::{
    EntityHistory, TemporalGraphRag, TemporalKnowledgeGraph, TemporalTriple,
};
pub use temporal_retrieval::{
    annotate_timestamps, extract_timestamp_from_metadata, extract_timestamp_from_triple,
    parse_timestamp, DecayFn, TemporalRetrievalConfig, TemporalRetriever, TimeWindow,
    UnknownTimestampPolicy,
};
