//! RAG (Retrieval-Augmented Generation) System for OxiRS Chat
//!
//! This module has been refactored into a modular structure for better maintainability.
//! The original 3573-line file has been broken down into focused sub-modules:
//!
//! - `quantum_rag`: Quantum-inspired retrieval optimization
//! - `consciousness`: Consciousness-aware processing with memory traces  
//! - `vector_search`: Vector-based semantic search and document management
//! - `embedding_providers`: Enhanced embedding models and multiple providers
//! - `graph_traversal`: Knowledge graph exploration and entity expansion
//! - `entity_extraction`: LLM-powered entity and relationship extraction
//! - `query_processing`: Query constraint processing and analysis utilities
//!
//! # Migration Notes
//!
//! The original large file has been preserved as `rag_old_large.rs` for reference.
//! All public APIs have been maintained for backward compatibility.

// Re-export everything from the rag submodule directory
pub use self::rag_impl::*;

// The actual implementation lives in the rag/ directory (mod.rs and submodules)
#[path = "rag/mod.rs"] 
mod rag_impl;