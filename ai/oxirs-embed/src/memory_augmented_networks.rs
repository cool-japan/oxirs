//! Memory-Augmented Networks for Advanced Knowledge Graph Embeddings
//!
//! This module implements state-of-the-art memory-augmented neural networks including:
//! - Differentiable Neural Computers (DNC) for external memory management
//! - Neural Turing Machines (NTM) for programmatic memory access
//! - Memory Networks for fact storage and retrieval
//! - Episodic Memory Networks for sequential knowledge storage
//! - Relational Memory Core for structured knowledge representation
//! - Sparse Access Memory (SAM) for efficient large-scale memory
//!
//! Implementation is split across sibling modules:
//! - `memory_nets_controller`: DNC, NTM, controller network, heads, addressing
//! - `memory_nets_ops`: MemoryNetworks, Episodic, Relational, Sparse, orchestrator
//! - `memory_nets_tests`: unit tests

pub use crate::memory_nets_controller::*;
pub use crate::memory_nets_ops::*;
