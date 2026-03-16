//! # API Stability Markers for OxiRS
//!
//! This module provides stability documentation and tracking for OxiRS public APIs,
//! following Rust API guidelines and SemVer compatibility promises.

pub mod compatibility;

/// API stability levels for OxiRS public interfaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StabilityLevel {
    /// Guaranteed backward compatible through the 1.x.y series.
    Stable,
    /// Likely stable but may change in minor releases.
    Beta,
    /// May change or be removed in any release.
    Experimental,
    /// Scheduled for removal in the next major version.
    Deprecated {
        /// The version in which this API was deprecated.
        since: &'static str,
        /// The recommended replacement API, if one exists.
        replacement: Option<&'static str>,
    },
}

impl StabilityLevel {
    /// Returns true if the API is considered production-ready (Stable or Beta).
    pub fn is_production_ready(&self) -> bool {
        matches!(
            self,
            StabilityLevel::Stable | StabilityLevel::Beta | StabilityLevel::Deprecated { .. }
        )
    }

    /// Returns a short descriptive label for the stability level.
    pub fn label(&self) -> &'static str {
        match self {
            StabilityLevel::Stable => "Stable",
            StabilityLevel::Beta => "Beta",
            StabilityLevel::Experimental => "Experimental",
            StabilityLevel::Deprecated { .. } => "Deprecated",
        }
    }

    /// Returns a bracket indicator for report formatting.
    pub fn indicator(&self) -> &'static str {
        match self {
            StabilityLevel::Stable => "[STABLE]",
            StabilityLevel::Beta => "[BETA]",
            StabilityLevel::Experimental => "[EXPERIMENTAL]",
            StabilityLevel::Deprecated { .. } => "[DEPRECATED]",
        }
    }
}

/// The broad functional category of an OxiRS API.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ApiCategory {
    QueryLanguage,
    DataModel,
    Reasoning,
    GraphQuery,
    Geospatial,
    Validation,
    Storage,
    Streaming,
    Industrial,
    ArtificialIntelligence,
    Security,
    RdfStar,
    Federation,
    Server,
    WebAssembly,
    TimeSeries,
    PhysicsSimulation,
    Quantum,
    Tooling,
}

impl ApiCategory {
    /// Returns a human-readable label for the category.
    pub fn label(&self) -> &'static str {
        match self {
            ApiCategory::QueryLanguage => "Query Language",
            ApiCategory::DataModel => "Data Model",
            ApiCategory::Reasoning => "Reasoning",
            ApiCategory::GraphQuery => "Graph Query",
            ApiCategory::Geospatial => "Geospatial",
            ApiCategory::Validation => "Validation",
            ApiCategory::Storage => "Storage",
            ApiCategory::Streaming => "Streaming",
            ApiCategory::Industrial => "Industrial",
            ApiCategory::ArtificialIntelligence => "Artificial Intelligence",
            ApiCategory::Security => "Security",
            ApiCategory::RdfStar => "RDF-star",
            ApiCategory::Federation => "Federation",
            ApiCategory::Server => "Server",
            ApiCategory::WebAssembly => "WebAssembly",
            ApiCategory::TimeSeries => "Time-Series",
            ApiCategory::PhysicsSimulation => "Physics Simulation",
            ApiCategory::Quantum => "Quantum",
            ApiCategory::Tooling => "Tooling",
        }
    }
}

/// A single API stability marker recording the stability commitment for one feature.
#[derive(Debug, Clone)]
pub struct ApiStabilityMarker {
    /// The feature name.
    pub feature: &'static str,
    /// The functional category this feature belongs to.
    pub category: ApiCategory,
    /// The stability guarantee for this feature.
    pub level: StabilityLevel,
    /// The version in which this API was introduced.
    pub since: &'static str,
    /// Human-readable description of the feature and its stability rationale.
    pub description: &'static str,
    /// Optional link to specification or documentation.
    pub spec_url: Option<&'static str>,
}

impl ApiStabilityMarker {
    /// Creates a Stable API marker.
    pub fn stable(
        feature: &'static str,
        category: ApiCategory,
        since: &'static str,
        description: &'static str,
    ) -> Self {
        Self {
            feature,
            category,
            level: StabilityLevel::Stable,
            since,
            description,
            spec_url: None,
        }
    }

    /// Creates a Beta API marker.
    pub fn beta(
        feature: &'static str,
        category: ApiCategory,
        since: &'static str,
        description: &'static str,
    ) -> Self {
        Self {
            feature,
            category,
            level: StabilityLevel::Beta,
            since,
            description,
            spec_url: None,
        }
    }

    /// Creates an Experimental API marker.
    pub fn experimental(
        feature: &'static str,
        category: ApiCategory,
        since: &'static str,
        description: &'static str,
    ) -> Self {
        Self {
            feature,
            category,
            level: StabilityLevel::Experimental,
            since,
            description,
            spec_url: None,
        }
    }

    /// Creates a Deprecated API marker.
    pub fn deprecated(
        feature: &'static str,
        category: ApiCategory,
        since: &'static str,
        deprecated_since: &'static str,
        replacement: Option<&'static str>,
        description: &'static str,
    ) -> Self {
        Self {
            feature,
            category,
            level: StabilityLevel::Deprecated {
                since: deprecated_since,
                replacement,
            },
            since,
            description,
            spec_url: None,
        }
    }

    /// Attaches a specification URL (builder pattern).
    pub fn with_spec(mut self, url: &'static str) -> Self {
        self.spec_url = Some(url);
        self
    }

    /// Returns true if this marker is at the Stable level.
    pub fn is_stable(&self) -> bool {
        matches!(self.level, StabilityLevel::Stable)
    }

    /// Returns true if this marker is at the Beta level.
    pub fn is_beta(&self) -> bool {
        matches!(self.level, StabilityLevel::Beta)
    }

    /// Returns true if this marker is at the Experimental level.
    pub fn is_experimental(&self) -> bool {
        matches!(self.level, StabilityLevel::Experimental)
    }

    /// Returns true if this marker is Deprecated.
    pub fn is_deprecated(&self) -> bool {
        matches!(self.level, StabilityLevel::Deprecated { .. })
    }
}

/// Summary statistics for a stability registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistrySummary {
    pub total: usize,
    pub stable_count: usize,
    pub beta_count: usize,
    pub experimental_count: usize,
    pub deprecated_count: usize,
}

impl RegistrySummary {
    /// Percentage of APIs that are Stable (0-100).
    pub fn stable_percentage(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.stable_count as f64 / self.total as f64) * 100.0
    }

    /// Percentage of APIs that are production-ready (Stable + Beta + Deprecated).
    pub fn production_ready_percentage(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        ((self.stable_count + self.beta_count + self.deprecated_count) as f64 / self.total as f64)
            * 100.0
    }
}

/// Registry of all OxiRS public API stability commitments.
pub struct StabilityRegistry {
    markers: Vec<ApiStabilityMarker>,
}

impl StabilityRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            markers: Vec::new(),
        }
    }

    /// Registers a new API stability marker.
    pub fn register(&mut self, marker: ApiStabilityMarker) {
        self.markers.push(marker);
    }

    /// Returns the complete OxiRS v1 stability registry.
    pub fn oxirs_v1_stability() -> Self {
        let mut r = Self::new();

        // SPARQL Query Language
        r.register(ApiStabilityMarker::stable("SPARQL 1.1 SELECT", ApiCategory::QueryLanguage, "0.1.0",
            "Full SPARQL 1.1 SELECT query support including projections, DISTINCT, ORDER BY, LIMIT, OFFSET, GROUP BY, HAVING, and aggregate functions.")
            .with_spec("https://www.w3.org/TR/sparql11-query/"));
        r.register(
            ApiStabilityMarker::stable(
                "SPARQL 1.1 ASK",
                ApiCategory::QueryLanguage,
                "0.1.0",
                "SPARQL 1.1 ASK queries returning boolean existence results.",
            )
            .with_spec("https://www.w3.org/TR/sparql11-query/"),
        );
        r.register(
            ApiStabilityMarker::stable(
                "SPARQL 1.1 CONSTRUCT",
                ApiCategory::QueryLanguage,
                "0.1.0",
                "SPARQL 1.1 CONSTRUCT queries producing new RDF graphs from existing data.",
            )
            .with_spec("https://www.w3.org/TR/sparql11-query/"),
        );
        r.register(
            ApiStabilityMarker::stable(
                "SPARQL 1.1 DESCRIBE",
                ApiCategory::QueryLanguage,
                "0.1.0",
                "SPARQL 1.1 DESCRIBE queries returning RDF descriptions of resources.",
            )
            .with_spec("https://www.w3.org/TR/sparql11-query/"),
        );
        r.register(ApiStabilityMarker::stable("SPARQL 1.1 Update", ApiCategory::QueryLanguage, "0.1.0",
            "Full SPARQL 1.1 Update support: INSERT DATA, DELETE DATA, INSERT/DELETE WHERE, LOAD, CLEAR, CREATE, DROP.")
            .with_spec("https://www.w3.org/TR/sparql11-update/"));
        r.register(ApiStabilityMarker::stable("SPARQL 1.1 Property Paths", ApiCategory::QueryLanguage, "0.1.0",
            "SPARQL 1.1 property path expressions including sequence, alternative, inverse, kleene star, and kleene plus."));
        r.register(ApiStabilityMarker::stable(
            "SPARQL 1.1 Aggregates",
            ApiCategory::QueryLanguage,
            "0.1.0",
            "SPARQL 1.1 aggregate functions: COUNT, SUM, MIN, MAX, AVG, GROUP_CONCAT, SAMPLE.",
        ));
        r.register(
            ApiStabilityMarker::stable(
                "SPARQL 1.1 Federated Query (SERVICE)",
                ApiCategory::QueryLanguage,
                "0.1.0",
                "SPARQL 1.1 SERVICE clause for federated queries across multiple SPARQL endpoints.",
            )
            .with_spec("https://www.w3.org/TR/sparql11-federated-query/"),
        );
        r.register(
            ApiStabilityMarker::stable(
                "SPARQL 1.2 RDF-star Expressions",
                ApiCategory::RdfStar,
                "0.1.0",
                "SPARQL 1.2 triple expressions and annotation syntax for quoted triple patterns.",
            )
            .with_spec("https://www.w3.org/TR/sparql12-query/"),
        );

        // RDF Data Model
        r.register(ApiStabilityMarker::stable("RDF 1.2 Data Model", ApiCategory::DataModel, "0.1.0",
            "Complete RDF 1.2 data model: IRIs, blank nodes, literals, language-tagged strings, datatyped literals, and quoted triples.")
            .with_spec("https://www.w3.org/TR/rdf12-concepts/"));
        r.register(
            ApiStabilityMarker::stable(
                "Turtle Serialization",
                ApiCategory::DataModel,
                "0.1.0",
                "Streaming Turtle 1.1 parser and writer with full namespace prefix support.",
            )
            .with_spec("https://www.w3.org/TR/turtle/"),
        );
        r.register(
            ApiStabilityMarker::stable(
                "N-Triples Serialization",
                ApiCategory::DataModel,
                "0.1.0",
                "N-Triples parser and writer for simple line-oriented RDF serialization.",
            )
            .with_spec("https://www.w3.org/TR/n-triples/"),
        );
        r.register(
            ApiStabilityMarker::stable(
                "N-Quads Serialization",
                ApiCategory::DataModel,
                "0.1.0",
                "N-Quads parser and writer for quad-based RDF dataset serialization.",
            )
            .with_spec("https://www.w3.org/TR/n-quads/"),
        );
        r.register(
            ApiStabilityMarker::stable(
                "TriG Serialization",
                ApiCategory::DataModel,
                "0.1.0",
                "TriG parser and writer for named graph serialization.",
            )
            .with_spec("https://www.w3.org/TR/trig/"),
        );
        r.register(
            ApiStabilityMarker::stable(
                "RDF/XML Serialization",
                ApiCategory::DataModel,
                "0.1.0",
                "RDF/XML parser and writer for legacy compatibility.",
            )
            .with_spec("https://www.w3.org/TR/rdf-syntax-grammar/"),
        );
        r.register(
            ApiStabilityMarker::stable(
                "JSON-LD 1.1 Serialization",
                ApiCategory::DataModel,
                "0.1.0",
                "JSON-LD 1.1 parser and writer with context expansion and compaction.",
            )
            .with_spec("https://www.w3.org/TR/json-ld11/"),
        );
        r.register(ApiStabilityMarker::stable(
            "Named Graphs",
            ApiCategory::DataModel,
            "0.1.0",
            "Full quad store support with named graph CRUD operations and graph management.",
        ));

        // Reasoning
        r.register(ApiStabilityMarker::stable("RDFS Inference", ApiCategory::Reasoning, "0.1.0",
            "RDFS entailment regime including subClassOf, subPropertyOf, domain, range, and transitive closure.")
            .with_spec("https://www.w3.org/TR/rdf-schema/"));
        r.register(ApiStabilityMarker::stable("OWL 2 RL Reasoning", ApiCategory::Reasoning, "0.1.0",
            "OWL 2 RL profile reasoning with forward-chaining rules for description logic reasoning over large datasets.")
            .with_spec("https://www.w3.org/TR/owl2-profiles/#OWL_2_RL"));
        r.register(ApiStabilityMarker::stable("OWL 2 EL Reasoning", ApiCategory::Reasoning, "0.1.0",
            "OWL 2 EL profile reasoning optimized for large biomedical and enterprise ontologies.")
            .with_spec("https://www.w3.org/TR/owl2-profiles/#OWL_2_EL"));
        r.register(ApiStabilityMarker::beta("OWL 2 QL Reasoning", ApiCategory::Reasoning, "0.2.0",
            "OWL 2 QL profile reasoning for query rewriting over relational sources. Conjunctive query rewriting is complete.")
            .with_spec("https://www.w3.org/TR/owl2-profiles/#OWL_2_QL"));
        r.register(
            ApiStabilityMarker::stable(
                "SWRL Rule Execution",
                ApiCategory::Reasoning,
                "0.2.0",
                "Semantic Web Rule Language (SWRL) rule parsing and forward-chaining execution.",
            )
            .with_spec("https://www.w3.org/Submission/SWRL/"),
        );

        // GraphQL
        r.register(ApiStabilityMarker::stable("GraphQL API", ApiCategory::GraphQuery, "0.1.0",
            "Full GraphQL query execution over RDF datasets with automatic schema generation from RDF shapes."));
        r.register(ApiStabilityMarker::stable("GraphQL Federation", ApiCategory::GraphQuery, "0.2.0",
            "GraphQL Federation v2 support for composing distributed subgraphs into a unified supergraph."));
        r.register(ApiStabilityMarker::beta(
            "GraphQL Subscriptions",
            ApiCategory::GraphQuery,
            "0.2.0",
            "GraphQL subscriptions over WebSocket with change tracking and subscription optimizer.",
        ));
        r.register(ApiStabilityMarker::beta("GraphQL Mutations", ApiCategory::GraphQuery, "0.2.0",
            "GraphQL mutations mapped to SPARQL UPDATE operations for write-through to the RDF store."));

        // GeoSPARQL
        r.register(ApiStabilityMarker::stable("GeoSPARQL 1.1", ApiCategory::Geospatial, "0.1.0",
            "GeoSPARQL 1.1 spatial query functions, geometry relations, and WKT/GeoJSON serialization.")
            .with_spec("https://www.ogc.org/standard/geosparql/"));
        r.register(ApiStabilityMarker::beta(
            "GeoSPARQL 3D Geometry",
            ApiCategory::Geospatial,
            "0.2.0",
            "3D geometry support (PolyhedralSurface, MultiSolid) for volumetric spatial queries.",
        ));
        r.register(ApiStabilityMarker::beta("GeoSPARQL Coordinate Reference Systems", ApiCategory::Geospatial, "0.2.0",
            "CRS transformation support for queries involving data in multiple spatial reference systems."));
        r.register(ApiStabilityMarker::beta(
            "GeoSPARQL R-Tree Index",
            ApiCategory::Geospatial,
            "0.2.0",
            "R-tree spatial index for accelerating GeoSPARQL geometric relation queries.",
        ));

        // Validation
        r.register(ApiStabilityMarker::stable("SHACL Validation", ApiCategory::Validation, "0.1.0",
            "Full SHACL 1.0 constraint validation including cardinality, value type, pattern, and SPARQL constraints.")
            .with_spec("https://www.w3.org/TR/shacl/"));
        r.register(ApiStabilityMarker::stable(
            "SHACL-SPARQL Constraints",
            ApiCategory::Validation,
            "0.1.0",
            "SHACL-SPARQL constraints allowing arbitrary SPARQL-based shape validation.",
        ));
        r.register(ApiStabilityMarker::stable("SAMM Aspect Model", ApiCategory::Validation, "0.1.0",
            "SAMM (Semantic Aspect Meta Model) support for industrial asset modeling and validation."));
        r.register(ApiStabilityMarker::beta("SHACL-AI Shape Learning", ApiCategory::Validation, "0.2.0",
            "AI-powered automatic SHACL shape inference from example data using pattern mining and constraint learning."));

        // Storage
        r.register(ApiStabilityMarker::stable("In-Memory RDF Store", ApiCategory::Storage, "0.1.0",
            "High-performance in-memory RDF store with multi-index support (SPO/POS/OSP) and ACID transactions."));
        r.register(ApiStabilityMarker::stable("TDB Persistent Storage", ApiCategory::Storage, "0.1.0",
            "TDB-compatible disk-based RDF storage with B-tree indices, WAL journal, and crash recovery."));
        r.register(ApiStabilityMarker::stable("RocksDB Backend", ApiCategory::Storage, "0.1.0",
            "RocksDB-backed persistent storage for high-throughput write workloads with SST compaction."));
        r.register(ApiStabilityMarker::beta("Distributed Cluster Storage", ApiCategory::Storage, "0.2.0",
            "Raft-based distributed storage cluster with automatic sharding, replication, and leader election."));
        r.register(ApiStabilityMarker::beta("Full-Text Search Index", ApiCategory::Storage, "0.2.0",
            "Tantivy-powered full-text search index for literal values with BM25 ranking and language filtering."));
        r.register(ApiStabilityMarker::beta("Temporal RDF Storage", ApiCategory::Storage, "0.2.0",
            "Versioned triple storage with time-travel queries, changelog, and bitemporal data support."));

        // Streaming / Federation
        r.register(ApiStabilityMarker::beta("RDF Streaming Processing", ApiCategory::Streaming, "0.1.0",
            "Real-time RDF stream processing with windowed aggregations and complex event patterns."));
        r.register(ApiStabilityMarker::beta("Kafka Integration", ApiCategory::Streaming, "0.1.0",
            "Apache Kafka consumer/producer integration for ingesting RDF triples from event streams."));
        r.register(ApiStabilityMarker::beta(
            "NATS Integration",
            ApiCategory::Streaming,
            "0.1.0",
            "NATS.io message broker integration for lightweight RDF event streaming.",
        ));
        r.register(ApiStabilityMarker::beta("Federated SPARQL Query", ApiCategory::Federation, "0.2.0",
            "Distributed SPARQL query execution across multiple OxiRS endpoints with cost-based join reordering."));
        r.register(ApiStabilityMarker::beta("Stream Fault Tolerance", ApiCategory::Streaming, "0.2.0",
            "Bulkhead isolation, supervisor trees, and checkpoint recovery for streaming pipeline resilience."));

        // Industrial
        r.register(ApiStabilityMarker::beta("Modbus TCP/RTU", ApiCategory::Industrial, "0.1.0",
            "Modbus TCP and RTU protocol client with RDF mapping for industrial sensor data ingestion."));
        r.register(ApiStabilityMarker::beta(
            "Modbus ASCII Protocol",
            ApiCategory::Industrial,
            "0.2.0",
            "Modbus ASCII framing with LRC checksum for legacy industrial device connectivity.",
        ));
        r.register(ApiStabilityMarker::beta(
            "Modbus TLS Client",
            ApiCategory::Industrial,
            "0.2.0",
            "Modbus Secure (TLS over IANA port 802) for encrypted industrial communications.",
        ));
        r.register(ApiStabilityMarker::beta("CANbus/J1939 Integration", ApiCategory::Industrial, "0.1.0",
            "CANbus and J1939 protocol support with automatic RDF triple generation from CAN frames."));
        r.register(ApiStabilityMarker::beta("UDS ISO 14229 Diagnostics", ApiCategory::Industrial, "0.2.0",
            "Unified Diagnostic Services (ISO 14229) protocol client for automotive ECU diagnostics."));
        r.register(ApiStabilityMarker::beta(
            "CANopen DS-301",
            ApiCategory::Industrial,
            "0.2.0",
            "CANopen DS-301 protocol support with NMT, SDO, PDO, and object dictionary management.",
        ));

        // AI
        r.register(ApiStabilityMarker::beta("Vector Similarity Search", ApiCategory::ArtificialIntelligence, "0.1.0",
            "HNSW-based approximate nearest neighbour search for semantic similarity queries over RDF entity embeddings."));
        r.register(ApiStabilityMarker::beta("Knowledge Graph Embeddings", ApiCategory::ArtificialIntelligence, "0.1.0",
            "TransE, DistMult, ComplEx, and RotatE knowledge graph embedding models for entity and relation representation."));
        r.register(ApiStabilityMarker::beta("GraphRAG Retrieval", ApiCategory::ArtificialIntelligence, "0.2.0",
            "Graph-Retrieval-Augmented Generation (GraphRAG) combining SPARQL traversal with LLM reasoning for QA."));
        r.register(ApiStabilityMarker::beta("Graph Neural Networks", ApiCategory::ArtificialIntelligence, "0.2.0",
            "GraphSAGE and Graph Attention Network (GAT) implementations for inductive knowledge graph representation learning."));
        r.register(ApiStabilityMarker::beta("Conversational AI (RAG Chat)", ApiCategory::ArtificialIntelligence, "0.1.0",
            "Multi-turn conversational AI with retrieval-augmented generation over RDF knowledge graphs."));
        r.register(ApiStabilityMarker::experimental("Physics Simulation", ApiCategory::PhysicsSimulation, "0.2.0",
            "RDF-backed physics simulation with digital twin support (DTDL v2), conservation laws, and predictive maintenance."));
        r.register(ApiStabilityMarker::experimental(
            "Digital Twin Framework",
            ApiCategory::PhysicsSimulation,
            "0.2.0",
            "DTDL v2 digital twin modeling with synchronisation reports and RDF integration.",
        ));
        r.register(ApiStabilityMarker::experimental("Predictive Maintenance", ApiCategory::PhysicsSimulation, "0.2.0",
            "ML-based predictive maintenance using anomaly detection and remaining useful life estimation."));

        // Time-Series
        r.register(ApiStabilityMarker::beta("Time-Series Database", ApiCategory::TimeSeries, "0.1.0",
            "High-performance time-series storage with columnar compression, range queries, and downsampling."));
        r.register(ApiStabilityMarker::beta("Anomaly Detection", ApiCategory::TimeSeries, "0.2.0",
            "Statistical anomaly detection using Z-score, IQR, EWMA, and Isolation Forest algorithms."));
        r.register(ApiStabilityMarker::beta(
            "Time-Series Forecasting",
            ApiCategory::TimeSeries,
            "0.2.0",
            "Holt-Winters exponential smoothing for time-series forecasting and trend analysis.",
        ));
        r.register(ApiStabilityMarker::beta("Prometheus Remote Write", ApiCategory::TimeSeries, "0.2.0",
            "Prometheus remote write protocol support for ingesting metrics into OxiRS time-series storage."));

        // Security
        r.register(ApiStabilityMarker::beta("DID (Decentralised Identifiers)", ApiCategory::Security, "0.2.0",
            "W3C Decentralised Identifier support (did:key, did:web, did:ion, did:pkh, did:ethr) with key management.")
            .with_spec("https://www.w3.org/TR/did-core/"));
        r.register(ApiStabilityMarker::beta("W3C Verifiable Credentials", ApiCategory::Security, "0.2.0",
            "W3C Verifiable Credentials Data Model 2.0 with JSON-LD proof signatures and verification.")
            .with_spec("https://www.w3.org/TR/vc-data-model-2.0/"));
        r.register(ApiStabilityMarker::experimental(
            "Zero-Knowledge Proofs",
            ApiCategory::Security,
            "0.2.0",
            "ZKP-based credential presentations for privacy-preserving identity verification.",
        ));
        r.register(ApiStabilityMarker::experimental(
            "DID Key Rotation",
            ApiCategory::Security,
            "0.2.0",
            "Automated cryptographic key rotation with DID document update propagation.",
        ));

        // Server
        r.register(ApiStabilityMarker::stable("SPARQL 1.1 HTTP Protocol", ApiCategory::Server, "0.1.0",
            "W3C SPARQL 1.1 HTTP protocol endpoint with GET and POST query support, content negotiation, and streaming results.")
            .with_spec("https://www.w3.org/TR/sparql11-protocol/"));
        r.register(ApiStabilityMarker::stable("Fuseki-Compatible REST API", ApiCategory::Server, "0.1.0",
            "Apache Jena Fuseki-compatible REST API for dataset management, upload, and administration."));
        r.register(ApiStabilityMarker::stable("GraphQL HTTP Server", ApiCategory::Server, "0.1.0",
            "GraphQL-over-HTTP server with multipart upload, introspection, and schema-first API design."));
        r.register(ApiStabilityMarker::beta(
            "Multi-tenancy",
            ApiCategory::Server,
            "0.2.0",
            "Dataset-level multi-tenancy with isolated namespaces and per-tenant resource quotas.",
        ));

        // WebAssembly
        r.register(ApiStabilityMarker::beta(
            "WASM RDF Store",
            ApiCategory::WebAssembly,
            "0.2.0",
            "Full OxiRS RDF store compiled to WebAssembly for in-browser and edge deployment.",
        ));
        r.register(ApiStabilityMarker::beta(
            "WASM SPARQL Engine",
            ApiCategory::WebAssembly,
            "0.2.0",
            "SPARQL query execution in WebAssembly with zero native dependencies.",
        ));
        r.register(ApiStabilityMarker::beta("WASM SPARQL UPDATE", ApiCategory::WebAssembly, "0.2.0",
            "SPARQL UPDATE operations available in the WASM build for client-side data modification."));

        // Quantum
        r.register(ApiStabilityMarker::experimental("Quantum Graph Optimization", ApiCategory::Quantum, "0.2.0",
            "Quantum-inspired optimization algorithms for SPARQL query planning over large knowledge graphs."));
        r.register(ApiStabilityMarker::experimental("Quantum SPARQL Sampling", ApiCategory::Quantum, "0.2.0",
            "Quantum amplitude estimation for approximate SPARQL aggregate queries on large datasets."));

        // Tooling
        r.register(ApiStabilityMarker::stable("OxiRS CLI Tool", ApiCategory::Tooling, "0.1.0",
            "Command-line interface for SPARQL queries, dataset management, benchmarking, and graph analytics."));
        r.register(ApiStabilityMarker::stable("SPARQL Query Profiler", ApiCategory::Tooling, "0.2.0",
            "Query execution profiler with operator-level timing, memory usage, and optimization suggestions."));
        r.register(ApiStabilityMarker::beta("Jena Parity Checker", ApiCategory::Tooling, "0.2.0",
            "Automated verification tool comparing OxiRS feature coverage against Apache Jena baseline."));
        r.register(ApiStabilityMarker::beta("API Stability Report", ApiCategory::Tooling, "0.2.0",
            "Automated API stability report generation tracking stability commitments across OxiRS releases."));

        r
    }

    /// Returns all registered API stability markers.
    pub fn all_apis(&self) -> &[ApiStabilityMarker] {
        &self.markers
    }

    /// Returns all APIs at the Stable level.
    pub fn stable_apis(&self) -> Vec<&ApiStabilityMarker> {
        self.markers.iter().filter(|m| m.is_stable()).collect()
    }

    /// Returns all APIs at the Beta level.
    pub fn beta_apis(&self) -> Vec<&ApiStabilityMarker> {
        self.markers.iter().filter(|m| m.is_beta()).collect()
    }

    /// Returns all APIs at the Experimental level.
    pub fn experimental_apis(&self) -> Vec<&ApiStabilityMarker> {
        self.markers
            .iter()
            .filter(|m| m.is_experimental())
            .collect()
    }

    /// Returns all Deprecated APIs.
    pub fn deprecated_apis(&self) -> Vec<&ApiStabilityMarker> {
        self.markers.iter().filter(|m| m.is_deprecated()).collect()
    }

    /// Returns all APIs in a given category.
    pub fn apis_by_category(&self, category: &ApiCategory) -> Vec<&ApiStabilityMarker> {
        self.markers
            .iter()
            .filter(|m| &m.category == category)
            .collect()
    }

    /// Returns all APIs that are production-ready (Stable, Beta, or Deprecated).
    pub fn production_ready_apis(&self) -> Vec<&ApiStabilityMarker> {
        self.markers
            .iter()
            .filter(|m| m.level.is_production_ready())
            .collect()
    }

    /// Computes summary statistics for this registry.
    pub fn summary(&self) -> RegistrySummary {
        RegistrySummary {
            total: self.markers.len(),
            stable_count: self.stable_apis().len(),
            beta_count: self.beta_apis().len(),
            experimental_count: self.experimental_apis().len(),
            deprecated_count: self.deprecated_apis().len(),
        }
    }

    /// Generates a comprehensive Markdown stability report.
    pub fn generate_report(&self) -> String {
        let summary = self.summary();
        let mut report = String::with_capacity(8192);

        report.push_str("# OxiRS API Stability Report\n\n");
        report.push_str("> Auto-generated from the OxiRS StabilityRegistry.\n\n");
        report.push_str("## Summary\n\n");
        report.push_str("| Metric | Value |\n|--------|-------|\n");
        report.push_str(&format!("| Total APIs tracked | {} |\n", summary.total));
        report.push_str(&format!(
            "| Stable | {} ({:.0}%) |\n",
            summary.stable_count,
            summary.stable_percentage()
        ));
        report.push_str(&format!("| Beta | {} |\n", summary.beta_count));
        report.push_str(&format!(
            "| Experimental | {} |\n",
            summary.experimental_count
        ));
        report.push_str(&format!("| Deprecated | {} |\n", summary.deprecated_count));
        report.push_str(&format!(
            "| Production-ready (Stable+Beta) | {} ({:.0}%) |\n\n",
            summary.stable_count + summary.beta_count,
            (summary.stable_count + summary.beta_count) as f64 / summary.total.max(1) as f64
                * 100.0
        ));

        let categories = [
            ApiCategory::QueryLanguage,
            ApiCategory::DataModel,
            ApiCategory::RdfStar,
            ApiCategory::Reasoning,
            ApiCategory::GraphQuery,
            ApiCategory::Geospatial,
            ApiCategory::Validation,
            ApiCategory::Storage,
            ApiCategory::Streaming,
            ApiCategory::Federation,
            ApiCategory::Industrial,
            ApiCategory::ArtificialIntelligence,
            ApiCategory::TimeSeries,
            ApiCategory::PhysicsSimulation,
            ApiCategory::Security,
            ApiCategory::Server,
            ApiCategory::WebAssembly,
            ApiCategory::Quantum,
            ApiCategory::Tooling,
        ];

        report.push_str("## API Details by Category\n\n");
        for category in &categories {
            let apis = self.apis_by_category(category);
            if apis.is_empty() {
                continue;
            }
            report.push_str(&format!("### {}\n\n", category.label()));
            report.push_str("| Feature | Status | Since | Description |\n");
            report.push_str("|---------|--------|-------|-------------|\n");
            for api in &apis {
                let desc = if api.description.len() > 80 {
                    format!("{}...", &api.description[..77])
                } else {
                    api.description.to_string()
                };
                report.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    api.feature,
                    api.level.indicator(),
                    api.since,
                    desc
                ));
            }
            report.push('\n');
        }

        report.push_str("## Stability Level Definitions\n\n");
        report.push_str("| Level | Guarantee |\n|-------|----------|\n");
        report.push_str("| [STABLE] | Backward compatible throughout 1.x.y |\n");
        report.push_str("| [BETA] | Stable but may change in minor releases |\n");
        report.push_str("| [EXPERIMENTAL] | May change or be removed in any release |\n");
        report.push_str("| [DEPRECATED] | Scheduled for removal in next major version |\n");

        report
    }

    /// Validates all markers for non-empty required fields.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for (i, marker) in self.markers.iter().enumerate() {
            if marker.feature.is_empty() {
                errors.push(format!("Marker #{i}: feature name is empty"));
            }
            if marker.description.is_empty() {
                errors.push(format!(
                    "Marker #{}: '{}' has empty description",
                    i, marker.feature
                ));
            }
            if marker.since.is_empty() {
                errors.push(format!(
                    "Marker #{}: '{}' has empty 'since' version",
                    i, marker.feature
                ));
            }
        }
        errors
    }
}

impl Default for StabilityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn registry() -> StabilityRegistry {
        StabilityRegistry::oxirs_v1_stability()
    }

    #[test]
    fn test_registry_is_non_empty() {
        assert!(!registry().all_apis().is_empty());
    }

    #[test]
    fn test_registry_has_minimum_api_count() {
        assert!(
            registry().all_apis().len() >= 40,
            "Registry should track at least 40 APIs"
        );
    }

    #[test]
    fn test_registry_validates_cleanly() {
        let errors = registry().validate();
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);
    }

    #[test]
    fn test_stable_apis_non_empty() {
        assert!(!registry().stable_apis().is_empty());
    }

    #[test]
    fn test_beta_apis_non_empty() {
        assert!(!registry().beta_apis().is_empty());
    }

    #[test]
    fn test_experimental_apis_non_empty() {
        assert!(!registry().experimental_apis().is_empty());
    }

    #[test]
    fn test_deprecated_apis_empty_by_default() {
        assert!(registry().deprecated_apis().is_empty());
    }

    #[test]
    fn test_stable_apis_are_all_stable() {
        for api in registry().stable_apis() {
            assert!(
                matches!(api.level, StabilityLevel::Stable),
                "{} should be Stable",
                api.feature
            );
        }
    }

    #[test]
    fn test_beta_apis_are_all_beta() {
        for api in registry().beta_apis() {
            assert!(
                matches!(api.level, StabilityLevel::Beta),
                "{} should be Beta",
                api.feature
            );
        }
    }

    #[test]
    fn test_experimental_apis_are_all_experimental() {
        for api in registry().experimental_apis() {
            assert!(
                matches!(api.level, StabilityLevel::Experimental),
                "{} should be Experimental",
                api.feature
            );
        }
    }

    #[test]
    fn test_counts_sum_to_total() {
        let reg = registry();
        let s = reg.summary();
        assert_eq!(
            s.stable_count + s.beta_count + s.experimental_count + s.deprecated_count,
            s.total
        );
    }

    #[test]
    fn test_query_language_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::QueryLanguage)
            .is_empty());
    }

    #[test]
    fn test_data_model_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::DataModel)
            .is_empty());
    }

    #[test]
    fn test_reasoning_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::Reasoning)
            .is_empty());
    }

    #[test]
    fn test_ai_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::ArtificialIntelligence)
            .is_empty());
    }

    #[test]
    fn test_security_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::Security)
            .is_empty());
    }

    #[test]
    fn test_tooling_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::Tooling)
            .is_empty());
    }

    #[test]
    fn test_geospatial_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::Geospatial)
            .is_empty());
    }

    #[test]
    fn test_storage_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::Storage)
            .is_empty());
    }

    #[test]
    fn test_streaming_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::Streaming)
            .is_empty());
    }

    #[test]
    fn test_industrial_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::Industrial)
            .is_empty());
    }

    #[test]
    fn test_server_apis_exist() {
        assert!(!registry().apis_by_category(&ApiCategory::Server).is_empty());
    }

    #[test]
    fn test_wasm_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::WebAssembly)
            .is_empty());
    }

    #[test]
    fn test_quantum_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::Quantum)
            .is_empty());
    }

    #[test]
    fn test_time_series_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::TimeSeries)
            .is_empty());
    }

    #[test]
    fn test_physics_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::PhysicsSimulation)
            .is_empty());
    }

    #[test]
    fn test_rdf_star_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::RdfStar)
            .is_empty());
    }

    #[test]
    fn test_federation_apis_exist() {
        assert!(!registry()
            .apis_by_category(&ApiCategory::Federation)
            .is_empty());
    }

    #[test]
    fn test_sparql_11_select_is_stable() {
        let reg = registry();
        let api = reg
            .all_apis()
            .iter()
            .find(|m| m.feature == "SPARQL 1.1 SELECT");
        assert!(api.is_some(), "SPARQL 1.1 SELECT should be registered");
        assert!(api.expect("API should be registered").is_stable());
    }

    #[test]
    fn test_rdf_data_model_is_stable() {
        let reg = registry();
        let api = reg
            .all_apis()
            .iter()
            .find(|m| m.feature == "RDF 1.2 Data Model");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").is_stable());
    }

    #[test]
    fn test_graphql_api_is_stable() {
        let reg = registry();
        let api = reg.all_apis().iter().find(|m| m.feature == "GraphQL API");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").is_stable());
    }

    #[test]
    fn test_geosparql_is_stable() {
        let reg = registry();
        let api = reg.all_apis().iter().find(|m| m.feature == "GeoSPARQL 1.1");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").is_stable());
    }

    #[test]
    fn test_shacl_is_stable() {
        let reg = registry();
        let api = reg
            .all_apis()
            .iter()
            .find(|m| m.feature == "SHACL Validation");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").is_stable());
    }

    #[test]
    fn test_time_series_db_is_beta() {
        let reg = registry();
        let api = reg
            .all_apis()
            .iter()
            .find(|m| m.feature == "Time-Series Database");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").is_beta());
    }

    #[test]
    fn test_graphrag_is_beta() {
        let reg = registry();
        let api = reg
            .all_apis()
            .iter()
            .find(|m| m.feature == "GraphRAG Retrieval");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").is_beta());
    }

    #[test]
    fn test_physics_simulation_is_experimental() {
        let reg = registry();
        let api = reg
            .all_apis()
            .iter()
            .find(|m| m.feature == "Physics Simulation");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").is_experimental());
    }

    #[test]
    fn test_quantum_is_experimental() {
        let reg = registry();
        let api = reg
            .all_apis()
            .iter()
            .find(|m| m.feature == "Quantum Graph Optimization");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").is_experimental());
    }

    #[test]
    fn test_stable_percentage_in_range() {
        let s = registry().summary();
        assert!(s.stable_percentage() > 0.0 && s.stable_percentage() <= 100.0);
    }

    #[test]
    fn test_production_ready_percentage_in_range() {
        let s = registry().summary();
        assert!(s.production_ready_percentage() > 0.0 && s.production_ready_percentage() <= 100.0);
    }

    #[test]
    fn test_production_ready_ge_stable_percentage() {
        let s = registry().summary();
        assert!(s.production_ready_percentage() >= s.stable_percentage());
    }

    #[test]
    fn test_generate_report_non_empty() {
        assert!(!registry().generate_report().is_empty());
    }

    #[test]
    fn test_report_contains_summary_header() {
        assert!(registry().generate_report().contains("## Summary"));
    }

    #[test]
    fn test_report_contains_stable_indicator() {
        assert!(registry().generate_report().contains("[STABLE]"));
    }

    #[test]
    fn test_report_contains_beta_indicator() {
        assert!(registry().generate_report().contains("[BETA]"));
    }

    #[test]
    fn test_report_contains_experimental_indicator() {
        assert!(registry().generate_report().contains("[EXPERIMENTAL]"));
    }

    #[test]
    fn test_report_contains_api_details_header() {
        assert!(registry()
            .generate_report()
            .contains("## API Details by Category"));
    }

    #[test]
    fn test_report_contains_sparql_select() {
        assert!(registry().generate_report().contains("SPARQL 1.1 SELECT"));
    }

    #[test]
    fn test_stable_is_production_ready() {
        assert!(StabilityLevel::Stable.is_production_ready());
    }

    #[test]
    fn test_beta_is_production_ready() {
        assert!(StabilityLevel::Beta.is_production_ready());
    }

    #[test]
    fn test_experimental_is_not_production_ready() {
        assert!(!StabilityLevel::Experimental.is_production_ready());
    }

    #[test]
    fn test_deprecated_is_production_ready() {
        let dep = StabilityLevel::Deprecated {
            since: "0.9.0",
            replacement: None,
        };
        assert!(dep.is_production_ready());
    }

    #[test]
    fn test_stability_labels() {
        assert_eq!(StabilityLevel::Stable.label(), "Stable");
        assert_eq!(StabilityLevel::Beta.label(), "Beta");
        assert_eq!(StabilityLevel::Experimental.label(), "Experimental");
        let dep = StabilityLevel::Deprecated {
            since: "0.9.0",
            replacement: None,
        };
        assert_eq!(dep.label(), "Deprecated");
    }

    #[test]
    fn test_stability_indicators() {
        assert_eq!(StabilityLevel::Stable.indicator(), "[STABLE]");
        assert_eq!(StabilityLevel::Beta.indicator(), "[BETA]");
        assert_eq!(StabilityLevel::Experimental.indicator(), "[EXPERIMENTAL]");
        let dep = StabilityLevel::Deprecated {
            since: "0.9.0",
            replacement: None,
        };
        assert_eq!(dep.indicator(), "[DEPRECATED]");
    }

    #[test]
    fn test_marker_stable_builder() {
        let m = ApiStabilityMarker::stable("Test API", ApiCategory::Tooling, "1.0.0", "A test API");
        assert!(m.is_stable());
        assert_eq!(m.feature, "Test API");
        assert!(m.spec_url.is_none());
    }

    #[test]
    fn test_marker_with_spec_url() {
        let m = ApiStabilityMarker::stable("Spec API", ApiCategory::QueryLanguage, "1.0.0", "desc")
            .with_spec("https://example.org/spec");
        assert_eq!(m.spec_url, Some("https://example.org/spec"));
    }

    #[test]
    fn test_marker_beta_builder() {
        let m = ApiStabilityMarker::beta("Beta Feature", ApiCategory::Streaming, "0.5.0", "desc");
        assert!(m.is_beta());
        assert!(!m.is_stable());
    }

    #[test]
    fn test_marker_experimental_builder() {
        let m =
            ApiStabilityMarker::experimental("Exp Feature", ApiCategory::Quantum, "0.9.0", "desc");
        assert!(m.is_experimental());
        assert!(!m.is_beta());
    }

    #[test]
    fn test_marker_deprecated_builder() {
        let m = ApiStabilityMarker::deprecated(
            "Old Feature",
            ApiCategory::Tooling,
            "0.1.0",
            "0.9.0",
            Some("New Feature"),
            "desc",
        );
        assert!(m.is_deprecated());
        if let StabilityLevel::Deprecated { since, replacement } = &m.level {
            assert_eq!(*since, "0.9.0");
            assert_eq!(*replacement, Some("New Feature"));
        }
    }

    #[test]
    fn test_feature_names_are_unique() {
        let reg = registry();
        let mut seen = HashSet::new();
        for api in reg.all_apis() {
            assert!(
                seen.insert(api.feature),
                "Duplicate feature name: '{}'",
                api.feature
            );
        }
    }

    #[test]
    fn test_production_ready_apis_non_empty() {
        assert!(!registry().production_ready_apis().is_empty());
    }

    #[test]
    fn test_production_ready_count_matches_summary() {
        let reg = registry();
        let s = reg.summary();
        let actual = reg.production_ready_apis().len();
        assert_eq!(actual, s.stable_count + s.beta_count + s.deprecated_count);
    }

    #[test]
    fn test_default_registry_is_empty() {
        let reg = StabilityRegistry::default();
        assert_eq!(reg.all_apis().len(), 0);
    }

    #[test]
    fn test_register_and_retrieve() {
        let mut reg = StabilityRegistry::new();
        reg.register(ApiStabilityMarker::stable(
            "Custom API",
            ApiCategory::Tooling,
            "1.0.0",
            "Custom",
        ));
        assert_eq!(reg.all_apis().len(), 1);
        assert_eq!(reg.stable_apis().len(), 1);
    }

    #[test]
    fn test_all_markers_have_valid_since_version() {
        for api in registry().all_apis() {
            assert!(
                api.since.starts_with('0') || api.since.starts_with('1'),
                "'{}' has invalid since version: '{}'",
                api.feature,
                api.since
            );
        }
    }

    #[test]
    fn test_stable_sparql_update_exists() {
        let reg = registry();
        let api = reg
            .all_apis()
            .iter()
            .find(|m| m.feature == "SPARQL 1.1 Update");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").is_stable());
    }

    #[test]
    fn test_owl2_rl_is_stable() {
        let reg = registry();
        let api = reg
            .all_apis()
            .iter()
            .find(|m| m.feature == "OWL 2 RL Reasoning");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").is_stable());
    }

    #[test]
    fn test_owl2_ql_is_beta() {
        let reg = registry();
        let api = reg
            .all_apis()
            .iter()
            .find(|m| m.feature == "OWL 2 QL Reasoning");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").is_beta());
    }

    #[test]
    fn test_category_label_non_empty() {
        let categories = [
            ApiCategory::QueryLanguage,
            ApiCategory::DataModel,
            ApiCategory::Reasoning,
            ApiCategory::GraphQuery,
            ApiCategory::Geospatial,
            ApiCategory::Validation,
            ApiCategory::Storage,
            ApiCategory::Streaming,
            ApiCategory::Industrial,
            ApiCategory::ArtificialIntelligence,
            ApiCategory::Security,
            ApiCategory::RdfStar,
            ApiCategory::Federation,
            ApiCategory::Server,
            ApiCategory::WebAssembly,
            ApiCategory::TimeSeries,
            ApiCategory::PhysicsSimulation,
            ApiCategory::Quantum,
            ApiCategory::Tooling,
        ];
        for cat in &categories {
            assert!(
                !cat.label().is_empty(),
                "Category label should not be empty"
            );
        }
    }

    #[test]
    fn test_zkp_is_experimental() {
        let reg = registry();
        let api = reg
            .all_apis()
            .iter()
            .find(|m| m.feature == "Zero-Knowledge Proofs");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").is_experimental());
    }

    #[test]
    fn test_did_is_beta() {
        let reg = registry();
        let api = reg
            .all_apis()
            .iter()
            .find(|m| m.feature == "DID (Decentralised Identifiers)");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").is_beta());
    }

    #[test]
    fn test_turtle_serialization_has_spec_url() {
        let reg = registry();
        let api = reg
            .all_apis()
            .iter()
            .find(|m| m.feature == "Turtle Serialization");
        assert!(api.is_some());
        assert!(api.expect("API should be registered").spec_url.is_some());
    }

    #[test]
    fn test_report_contains_stability_level_defs() {
        let report = registry().generate_report();
        assert!(report.contains("Stability Level Definitions"));
    }
}
