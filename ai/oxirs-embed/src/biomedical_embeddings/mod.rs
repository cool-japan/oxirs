//! Biomedical knowledge graph embeddings for scientific applications
//!
//! This module provides specialized embeddings for biomedical knowledge graphs
//! including gene-disease associations, drug-target interactions, pathways,
//! protein structures, and medical concept hierarchies.


pub mod embedding;
pub mod network_analysis;
pub mod text_model;
pub mod types;

pub use network_analysis::*;
pub use text_model::*;
pub use types::*;
