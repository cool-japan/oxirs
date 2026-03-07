//! Dataset import/export format converters for OxiRS TDB.
//!
//! Provides conversions between:
//! - CSV ↔ RDF triples ([`csv_rdf`])
//! - JSON-LD → triples ([`JsonLdImporter`])
//! - N-Quads ↔ (quad) tuples ([`NQuadsImporter`], [`NQuadsExporter`])

pub mod csv_rdf;

pub use csv_rdf::{CsvRdfMapper, JsonLdImporter, NQuadsExporter, NQuadsImporter, RdfColumnType};
