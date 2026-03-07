//! Extended storage modules for RDF-star data.
//!
//! - [`star_dict`] – compressed term dictionary for RDF-star, supporting
//!   quoted triples stored by component ID rather than redundant strings.
//! - [`hdt_star_v2`] – enhanced HDT-star compression with front-coding and
//!   XOR delta bitmap indices.
//! - [`rdf_star_index`] – compound B-tree key index for efficient nested
//!   triple pattern matching with prefix scans.

/// Compressed dictionary for RDF-star terms and quoted triples.
pub mod star_dict;

/// Enhanced HDT-star v2 storage with front-coding and XOR delta encoding.
pub mod hdt_star_v2;

/// Compound B-tree key index for RDF-star quoted triple storage.
pub mod rdf_star_index;
