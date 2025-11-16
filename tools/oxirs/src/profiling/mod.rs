//! Profiling utilities for SPARQL query performance analysis

pub mod flamegraph;

pub use flamegraph::{
    DiffSummary, DifferentialFlameGraph, ExecutionPhase, FlameGraphDirection, FlameGraphGenerator,
    FlameGraphOptions, PhaseStats, ProfilingSample,
};
