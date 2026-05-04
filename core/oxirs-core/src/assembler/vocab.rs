//! Jena Assembler vocabulary constants.
//!
//! The `ja:` namespace provides RDF-based configuration for Jena-compatible
//! dataset assembly. This module contains every IRI constant needed to parse
//! `.ttl` assembler files that describe datasets, named graphs, and storage
//! backends.
//!
//! ## Reference
//! - Jena Assembler: <https://jena.apache.org/documentation/assembler/>
//! - TDB2: <https://jena.apache.org/documentation/tdb2/>

/// The Jena Assembler namespace IRI (`http://jena.hpl.hp.com/2005/11/Assembler#`).
pub const JA_NAMESPACE: &str = "http://jena.hpl.hp.com/2005/11/Assembler#";

/// The TDB2 (Jena TDB2 backend) namespace IRI (`http://jena.apache.org/2016/tdb#`).
pub const TDB2_NAMESPACE: &str = "http://jena.apache.org/2016/tdb#";

/// The RDFS namespace IRI (`http://www.w3.org/2000/01/rdf-schema#`).
pub const RDFS_NAMESPACE: &str = "http://www.w3.org/2000/01/rdf-schema#";

/// `rdf:type` predicate IRI.
pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

// ---------------------------------------------------------------------------
// ja: class IRIs
// ---------------------------------------------------------------------------

/// `ja:RDFDataset` — a generic RDF dataset assembled from named-graph descriptions.
pub const JA_RDF_DATASET: &str = "http://jena.hpl.hp.com/2005/11/Assembler#RDFDataset";

/// `ja:MemoryModel` — an in-memory RDF model (single graph).
pub const JA_MEMORY_MODEL: &str = "http://jena.hpl.hp.com/2005/11/Assembler#MemoryModel";

/// `ja:MemoryDataset` — an in-memory RDF dataset (multiple named graphs).
pub const JA_MEMORY_DATASET: &str = "http://jena.hpl.hp.com/2005/11/Assembler#MemoryDataset";

// ---------------------------------------------------------------------------
// ja: property IRIs
// ---------------------------------------------------------------------------

/// `ja:namedGraph` — links a dataset to a named-graph configuration blank node.
pub const JA_NAMED_GRAPH: &str = "http://jena.hpl.hp.com/2005/11/Assembler#namedGraph";

/// `ja:graphName` — the IRI of a named graph within a `ja:namedGraph` description.
pub const JA_GRAPH_NAME: &str = "http://jena.hpl.hp.com/2005/11/Assembler#graphName";

/// `ja:graph` — points from a named-graph description to its model resource.
pub const JA_GRAPH: &str = "http://jena.hpl.hp.com/2005/11/Assembler#graph";

/// `ja:defaultGraph` — points from a dataset to its default-graph model resource.
pub const JA_DEFAULT_GRAPH: &str = "http://jena.hpl.hp.com/2005/11/Assembler#defaultGraph";

/// `ja:content` — links a model to a content description (file or URL).
pub const JA_CONTENT: &str = "http://jena.hpl.hp.com/2005/11/Assembler#content";

/// `ja:contentURL` — a URL from which to load initial RDF content.
pub const JA_CONTENT_URL: &str = "http://jena.hpl.hp.com/2005/11/Assembler#contentURL";

// ---------------------------------------------------------------------------
// tdb2: class and property IRIs
// ---------------------------------------------------------------------------

/// `tdb2:DatasetTDB2` — a TDB2 disk-backed dataset.
pub const TDB2_DATASET: &str = "http://jena.apache.org/2016/tdb#DatasetTDB2";

/// `tdb2:location` — filesystem path for the TDB2 dataset storage directory.
pub const TDB2_LOCATION: &str = "http://jena.apache.org/2016/tdb#location";
