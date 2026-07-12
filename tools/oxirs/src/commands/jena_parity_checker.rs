//! The Jena parity checker engine.
//!
//! [`JenaParityChecker`](crate::commands::jena_parity_checker::JenaParityChecker) holds the full feature catalogue, the per-category
//! registration routines, the coverage analysis helpers and the Markdown report
//! generator.

use super::jena_parity_types::{FeatureStatus, ParityCategory, ParityFeature, ParitySummary};

/// The Jena parity checker: a comprehensive feature-by-feature comparison.
pub struct JenaParityChecker {
    features: Vec<ParityFeature>,
}

impl JenaParityChecker {
    /// Creates an empty parity checker.
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    /// Creates the full OxiRS vs. Apache Jena feature comparison.
    pub fn full_comparison() -> Self {
        let mut checker = Self::new();
        checker.register_query_language();
        checker.register_data_formats();
        checker.register_reasoning();
        checker.register_storage();
        checker.register_protocols();
        checker.register_security();
        checker.register_networking();
        checker.register_geospatial();
        checker.register_validation();
        checker.register_streaming();
        checker.register_ai();
        checker.register_tooling();
        checker.register_industrial();
        checker
    }

    fn register_query_language(&mut self) {
        self.add(ParityFeature::parity(
            "SPARQL 1.1 SELECT",
            ParityCategory::QueryLanguage,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.1 ASK",
            ParityCategory::QueryLanguage,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.1 CONSTRUCT",
            ParityCategory::QueryLanguage,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.1 DESCRIBE",
            ParityCategory::QueryLanguage,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.1 Update (INSERT/DELETE)",
            ParityCategory::QueryLanguage,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.1 Property Paths",
            ParityCategory::QueryLanguage,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.1 Aggregates (COUNT, SUM, AVG…)",
            ParityCategory::QueryLanguage,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.1 GROUP BY / HAVING",
            ParityCategory::QueryLanguage,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.1 OPTIONAL / FILTER",
            ParityCategory::QueryLanguage,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.1 BIND / VALUES",
            ParityCategory::QueryLanguage,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.1 Subqueries",
            ParityCategory::QueryLanguage,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.1 Federated Query (SERVICE)",
            ParityCategory::QueryLanguage,
            Some("OxiRS adds cost-based service selection on top of basic SERVICE"),
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.2 RDF-star (Quoted Triples)",
            ParityCategory::QueryLanguage,
            Some("Full SPARQL 1.2 draft compliance"),
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.2 Triple Annotations",
            ParityCategory::QueryLanguage,
            None,
        ));
        self.add(ParityFeature::beyond_jena(
            "SPARQL Query Compilation (JIT)",
            ParityCategory::QueryLanguage,
            Some("OxiRS compiles SPARQL to native code for repeated queries; Jena interprets"),
        ));
        self.add(ParityFeature::beyond_jena(
            "SPARQL Query Result Caching",
            ParityCategory::QueryLanguage,
            Some("LRU result cache with delta invalidation; not available in Jena"),
        ));
        self.add(ParityFeature::beyond_jena(
            "SPARQL Adaptive Query Optimizer",
            ParityCategory::QueryLanguage,
            Some("Runtime cardinality feedback loop for join reordering"),
        ));
    }

    fn register_data_formats(&mut self) {
        self.add(ParityFeature::parity(
            "Turtle 1.1",
            ParityCategory::DataFormats,
            None,
        ));
        self.add(ParityFeature::parity(
            "N-Triples",
            ParityCategory::DataFormats,
            None,
        ));
        self.add(ParityFeature::parity(
            "N-Quads",
            ParityCategory::DataFormats,
            None,
        ));
        self.add(ParityFeature::parity(
            "TriG",
            ParityCategory::DataFormats,
            None,
        ));
        self.add(ParityFeature::parity(
            "RDF/XML",
            ParityCategory::DataFormats,
            None,
        ));
        self.add(ParityFeature::parity(
            "JSON-LD 1.1",
            ParityCategory::DataFormats,
            None,
        ));
        self.add(ParityFeature::parity(
            "RDF-star / Turtle-star",
            ParityCategory::DataFormats,
            Some("Supported via the RDF 1.2 compliant parser"),
        ));
        self.add(ParityFeature::parity(
            "RDF Binary (RDF Thrift)",
            ParityCategory::DataFormats,
            Some("Full read/write with LEB128 varint, prefix compression, and EOF marker"),
        ));
        self.add(ParityFeature::beyond_jena(
            "RDF Streaming Parser (zero-copy)",
            ParityCategory::DataFormats,
            Some("OxiRS streaming parser allocates near-zero heap during parse"),
        ));
    }

    fn register_reasoning(&mut self) {
        self.add(ParityFeature::parity(
            "RDFS Inference",
            ParityCategory::Reasoning,
            None,
        ));
        self.add(ParityFeature::parity(
            "OWL 2 RL Reasoning",
            ParityCategory::Reasoning,
            None,
        ));
        self.add(ParityFeature::parity(
            "OWL 2 EL Reasoning",
            ParityCategory::Reasoning,
            None,
        ));
        self.add(ParityFeature::parity("OWL 2 QL Reasoning", ParityCategory::Reasoning,
            Some("UCQ rewriting complete: ConceptExpr, SubClassOfUnion, union query rewriting all implemented")));
        self.add(ParityFeature::parity("OWL 2 DL Reasoning", ParityCategory::Reasoning,
            Some("100% OWL 2 DL ABox reasoning: 29 rule groups — subclass/equiv/nominal/hasValue/allValuesFrom/someValuesFrom/chains/transitivity/symmetry/inverseOf/domain-range/sameAs/intersectionOf/functionalProp/inverseFunctional/hasKey/subObjectProp/subDataProp/equivProps/reflexive/irreflexive/hasSelf/negativeAssertion/maxCardinality/minCardinality/exactCardinality/unionOf/DataSomeValuesFrom/DataAllValuesFrom/AllDifferent/sameAsCongruence")));
        self.add(ParityFeature::parity(
            "SWRL Rules",
            ParityCategory::Reasoning,
            None,
        ));
        self.add(ParityFeature::parity(
            "RDF Entailment Regimes",
            ParityCategory::Reasoning,
            None,
        ));
        self.add(ParityFeature::beyond_jena(
            "Probabilistic RDF Reasoning",
            ParityCategory::Reasoning,
            Some("Markov Logic Network-style probabilistic entailment; no Jena equivalent"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Temporal RDF Reasoning",
            ParityCategory::Reasoning,
            Some("Bitemporal triple validity windows; Jena has no native temporal reasoning"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Neural-Symbolic Reasoning Fusion",
            ParityCategory::Reasoning,
            Some("LLM-guided rule selection combined with forward-chaining inference"),
        ));
    }

    fn register_storage(&mut self) {
        self.add(ParityFeature::parity(
            "In-Memory RDF Dataset",
            ParityCategory::Storage,
            None,
        ));
        self.add(ParityFeature::parity(
            "TDB2 Persistent Storage",
            ParityCategory::Storage,
            Some("OxiRS TDB is Jena-TDB2 API compatible"),
        ));
        self.add(ParityFeature::parity(
            "Transactional Isolation",
            ParityCategory::Storage,
            Some("ACID transactions with Jena-compatible isolation levels"),
        ));
        self.add(ParityFeature::parity(
            "Named Graph Management",
            ParityCategory::Storage,
            None,
        ));
        self.add(ParityFeature::beyond_jena(
            "RocksDB Backend",
            ParityCategory::Storage,
            Some("High-throughput LSM-tree storage; Jena uses custom B-tree"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Distributed Raft Cluster",
            ParityCategory::Storage,
            Some("Built-in Raft consensus for multi-node replication; Jena needs external cluster"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Vector Index (HNSW)",
            ParityCategory::Storage,
            Some(
                "Approximate nearest-neighbour index for embedding similarity; no Jena equivalent",
            ),
        ));
        self.add(ParityFeature::beyond_jena(
            "Full-Text Search (BM25)",
            ParityCategory::Storage,
            Some("Tantivy-powered BM25 full-text search; Jena uses Lucene separately"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Temporal Triple Versioning",
            ParityCategory::Storage,
            Some("Bitemporal versioned storage with time-travel queries"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Adaptive Compression (ZStd)",
            ParityCategory::Storage,
            Some("Per-index ZStd dictionary compression; Jena has no built-in compression"),
        ));
    }

    fn register_protocols(&mut self) {
        self.add(ParityFeature::parity(
            "SPARQL 1.1 HTTP Protocol",
            ParityCategory::Protocols,
            Some("Full W3C SPARQL 1.1 protocol compliance"),
        ));
        self.add(ParityFeature::parity(
            "SPARQL 1.1 Graph Store HTTP Protocol",
            ParityCategory::Protocols,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL Results CSV/TSV",
            ParityCategory::Protocols,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL Results JSON",
            ParityCategory::Protocols,
            None,
        ));
        self.add(ParityFeature::parity(
            "SPARQL Results XML",
            ParityCategory::Protocols,
            None,
        ));
        self.add(ParityFeature::parity(
            "Fuseki REST Management API",
            ParityCategory::Protocols,
            Some("Fuseki v2 admin REST API for dataset management"),
        ));
        self.add(ParityFeature::beyond_jena(
            "GraphQL API",
            ParityCategory::Protocols,
            Some("Full GraphQL query and mutation support; Jena has no GraphQL"),
        ));
        self.add(ParityFeature::beyond_jena(
            "GraphQL Federation v2",
            ParityCategory::Protocols,
            Some("Apollo Federation supergraph composition"),
        ));
        self.add(ParityFeature::beyond_jena(
            "GraphQL Subscriptions (WebSocket)",
            ParityCategory::Protocols,
            Some("Real-time GraphQL subscriptions with change-feed optimization"),
        ));
        self.add(ParityFeature::beyond_jena(
            "gRPC Streaming Query API",
            ParityCategory::Protocols,
            Some("High-performance binary streaming API for large result sets"),
        ));
    }

    fn register_security(&mut self) {
        self.add(ParityFeature::partial(
            "Basic Authentication",
            ParityCategory::Security,
            100,
            Some("HTTP Basic Auth over TLS"),
        ));
        self.add(ParityFeature::parity("OAuth 2.0 / OIDC", ParityCategory::Security,
            Some("Refresh token rotation with replay attack detection and cascade family revocation implemented")));
        self.add(ParityFeature::beyond_jena(
            "W3C DID (Decentralised Identifiers)",
            ParityCategory::Security,
            Some("did:key, did:web, did:ion, did:pkh, did:ethr — no Jena equivalent"),
        ));
        self.add(ParityFeature::beyond_jena(
            "W3C Verifiable Credentials 2.0",
            ParityCategory::Security,
            Some("JSON-LD data integrity proofs; no Jena equivalent"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Zero-Knowledge Proofs (ZKP)",
            ParityCategory::Security,
            Some("Privacy-preserving credential presentations"),
        ));
        self.add(ParityFeature::beyond_jena(
            "End-to-End Encrypted RDF Storage",
            ParityCategory::Security,
            Some("AES-256-GCM transparent encryption at rest"),
        ));
        self.add(ParityFeature::parity("Role-Based Access Control (RBAC)", ParityCategory::Security,
            Some("Graph-level ACL (GraphAcl, GraphAclStore, GraphAclPolicy) implemented alongside dataset-level RBAC")));
        self.add(ParityFeature::beyond_jena(
            "ReBAC (Relationship-Based Access Control)",
            ParityCategory::Security,
            Some("Zanzibar-style relationship-based permissions for RDF datasets"),
        ));
    }

    fn register_networking(&mut self) {
        self.add(ParityFeature::parity(
            "SPARQL Federated Query (SERVICE)",
            ParityCategory::Networking,
            None,
        ));
        self.add(ParityFeature::parity(
            "Remote SPARQL Endpoint Client",
            ParityCategory::Networking,
            None,
        ));
        self.add(ParityFeature::beyond_jena(
            "Distributed SPARQL with Cost-Based Routing",
            ParityCategory::Networking,
            Some("Query decomposition with source selection and join reordering across endpoints"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Cluster Auto-Sharding",
            ParityCategory::Networking,
            Some("Consistent-hash sharding of named graphs across nodes"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Gossip Protocol Scaling",
            ParityCategory::Networking,
            Some("Fanout-controlled gossip with ZStd compression for large clusters"),
        ));
        self.add(ParityFeature::beyond_jena(
            "SLA-Aware Query Scheduling",
            ParityCategory::Networking,
            Some("Per-tenant SLO targets with compliance snapshot monitoring"),
        ));
    }

    fn register_geospatial(&mut self) {
        self.add(ParityFeature::parity(
            "GeoSPARQL 1.1 Core",
            ParityCategory::Geospatial,
            None,
        ));
        self.add(ParityFeature::parity(
            "GeoSPARQL Topological Relations (DE-9IM)",
            ParityCategory::Geospatial,
            None,
        ));
        self.add(ParityFeature::parity(
            "WKT Geometry Serialization",
            ParityCategory::Geospatial,
            None,
        ));
        self.add(ParityFeature::parity(
            "GeoJSON Geometry Serialization",
            ParityCategory::Geospatial,
            None,
        ));
        self.add(ParityFeature::beyond_jena(
            "GeoSPARQL 3D Geometry (PolyhedralSurface)",
            ParityCategory::Geospatial,
            Some("Volumetric geometry support not in Jena's GeoSPARQL implementation"),
        ));
        self.add(ParityFeature::beyond_jena(
            "GeoSPARQL R-Tree Spatial Index",
            ParityCategory::Geospatial,
            Some("In-process R-tree avoids need for external PostGIS"),
        ));
        self.add(ParityFeature::beyond_jena(
            "CRS Transformation",
            ParityCategory::Geospatial,
            Some("On-the-fly coordinate system conversion during query execution"),
        ));
    }

    fn register_validation(&mut self) {
        self.add(ParityFeature::parity(
            "SHACL 1.0 Core Constraints",
            ParityCategory::Validation,
            None,
        ));
        self.add(ParityFeature::parity(
            "SHACL-SPARQL Constraints",
            ParityCategory::Validation,
            None,
        ));
        self.add(ParityFeature::parity(
            "SHACL Violation Reports",
            ParityCategory::Validation,
            None,
        ));
        self.add(ParityFeature::parity(
            "SAMM Aspect Model Validation",
            ParityCategory::Validation,
            None,
        ));
        self.add(ParityFeature::parity(
            "SHACL-AF (Advanced Features)",
            ParityCategory::Validation,
            Some("sh:SPARQLTarget, sh:SPARQLTargetType, and sh:SPARQLAskValidator all implemented"),
        ));
        self.add(ParityFeature::beyond_jena(
            "AI-Powered SHACL Shape Mining",
            ParityCategory::Validation,
            Some("Automatic shape inference from example data; no Jena equivalent"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Incremental SHACL Validation",
            ParityCategory::Validation,
            Some("Delta-based re-validation without full graph scan"),
        ));
    }

    fn register_streaming(&mut self) {
        self.add(ParityFeature::beyond_jena(
            "Apache Kafka Integration",
            ParityCategory::Streaming,
            Some("Jena has no built-in streaming; OxiRS provides native Kafka consumer/producer"),
        ));
        self.add(ParityFeature::beyond_jena(
            "NATS.io Integration",
            ParityCategory::Streaming,
            Some("Lightweight publish-subscribe for IoT RDF streams"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Windowed Stream Aggregation",
            ParityCategory::Streaming,
            Some("Tumbling, sliding, and session windows over RDF event streams"),
        ));
        self.add(ParityFeature::beyond_jena(
            "CRDT Distributed State",
            ParityCategory::Streaming,
            Some("Conflict-free replicated data types for eventually consistent stream state"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Stream Bulkhead Isolation",
            ParityCategory::Streaming,
            Some("Per-pipeline resource isolation to prevent cascade failures"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Checkpoint-Based Recovery",
            ParityCategory::Streaming,
            Some("Exactly-once delivery guarantees with periodic state snapshots"),
        ));
    }

    fn register_ai(&mut self) {
        self.add(ParityFeature::beyond_jena(
            "Knowledge Graph Embeddings (TransE/RotatE)",
            ParityCategory::Ai,
            Some("Entity/relation representation learning; no Jena equivalent"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Vector Similarity Search (HNSW)",
            ParityCategory::Ai,
            Some("Approximate nearest-neighbour for semantic search"),
        ));
        self.add(ParityFeature::beyond_jena(
            "GraphRAG Retrieval-Augmented Generation",
            ParityCategory::Ai,
            Some("SPARQL + LLM hybrid for natural language question answering over RDF"),
        ));
        self.add(ParityFeature::beyond_jena(
            "GraphSAGE Inductive Embeddings",
            ParityCategory::Ai,
            Some("Inductive representation learning for unseen nodes"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Graph Attention Networks (GAT)",
            ParityCategory::Ai,
            Some("Multi-head attention over knowledge graph neighbourhoods"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Conversational AI (RAG Chat)",
            ParityCategory::Ai,
            Some("Multi-turn chat over knowledge bases with context-aware retrieval"),
        ));
        self.add(ParityFeature::beyond_jena(
            "A/B Testing Framework for Embeddings",
            ParityCategory::Ai,
            Some("Statistical significance testing for embedding model comparisons"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Physics Digital Twin (DTDL v2)",
            ParityCategory::Ai,
            Some("RDF-backed physics simulation for industrial digital twins"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Predictive Maintenance ML",
            ParityCategory::Ai,
            Some("Anomaly detection and RUL estimation for industrial equipment"),
        ));
    }

    fn register_tooling(&mut self) {
        self.add(ParityFeature::parity(
            "CLI SPARQL Query Tool",
            ParityCategory::Tooling,
            None,
        ));
        self.add(ParityFeature::parity(
            "Dataset Import/Export (CLI)",
            ParityCategory::Tooling,
            None,
        ));
        self.add(ParityFeature::parity(
            "RDF Validation CLI",
            ParityCategory::Tooling,
            None,
        ));
        self.add(ParityFeature::beyond_jena(
            "Jena Parity Checker CLI",
            ParityCategory::Tooling,
            Some("Self-referential parity comparison tool"),
        ));
        self.add(ParityFeature::beyond_jena(
            "SPARQL Query Profiler",
            ParityCategory::Tooling,
            Some("Operator-level timing and memory attribution"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Adaptive Query Advisor",
            ParityCategory::Tooling,
            Some("ML-powered SPARQL rewriting suggestions based on query history"),
        ));
        self.add(ParityFeature::beyond_jena(
            "WASM Build Target",
            ParityCategory::Tooling,
            Some("Full OxiRS SPARQL engine compiled to WebAssembly"),
        ));
        self.add(ParityFeature::beyond_jena(
            "API Stability Report Generator",
            ParityCategory::Tooling,
            Some("Automated Markdown stability report for public API commitments"),
        ));
    }

    fn register_industrial(&mut self) {
        self.add(ParityFeature::beyond_jena(
            "Modbus TCP/RTU Client",
            ParityCategory::Industrial,
            Some("Industrial sensor data ingestion via Modbus; no Jena equivalent"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Modbus ASCII Protocol",
            ParityCategory::Industrial,
            Some("LRC-checksummed ASCII framing for legacy PLCs"),
        ));
        self.add(ParityFeature::beyond_jena(
            "Modbus TLS (Secure)",
            ParityCategory::Industrial,
            Some("Encrypted Modbus over TLS (IANA port 802)"),
        ));
        self.add(ParityFeature::beyond_jena(
            "CANbus / J1939 Integration",
            ParityCategory::Industrial,
            Some("Automotive and heavy-equipment bus data as RDF triples"),
        ));
        self.add(ParityFeature::beyond_jena(
            "UDS ISO 14229 Diagnostics Client",
            ParityCategory::Industrial,
            Some("ECU diagnostics protocol for automotive applications"),
        ));
        self.add(ParityFeature::beyond_jena(
            "CANopen DS-301 (NMT/SDO/PDO)",
            ParityCategory::Industrial,
            Some("Industrial automation bus with object dictionary management"),
        ));
        self.add(ParityFeature::beyond_jena(
            "SOSA/SSN RDF Sensor Mapping",
            ParityCategory::Industrial,
            Some("Automatic triple generation following W3C SOSA ontology"),
        ));
    }

    /// Adds a feature to the checker.
    fn add(&mut self, feature: ParityFeature) {
        self.features.push(feature);
    }

    /// Registers a custom feature.
    pub fn register(&mut self, feature: ParityFeature) {
        self.add(feature);
    }

    /// Returns all features.
    pub fn all_features(&self) -> &[ParityFeature] {
        &self.features
    }

    /// Returns features where OxiRS has full or better parity.
    pub fn at_parity(&self) -> Vec<&ParityFeature> {
        self.features.iter().filter(|f| f.is_at_parity()).collect()
    }

    /// Returns features that Jena has but OxiRS has not fully implemented.
    pub fn gaps(&self) -> Vec<&ParityFeature> {
        self.features
            .iter()
            .filter(|f| f.jena_support && !matches!(f.oxirs_support, FeatureStatus::Implemented))
            .collect()
    }

    /// Returns features OxiRS provides beyond the Jena baseline.
    pub fn beyond_jena_features(&self) -> Vec<&ParityFeature> {
        self.features
            .iter()
            .filter(|f| matches!(f.oxirs_support, FeatureStatus::BeyondJena))
            .collect()
    }

    /// Returns features in a specific category.
    pub fn by_category(&self, category: &ParityCategory) -> Vec<&ParityFeature> {
        self.features
            .iter()
            .filter(|f| &f.category == category)
            .collect()
    }

    /// Computes summary coverage statistics.
    pub fn summary(&self) -> ParitySummary {
        let implemented = self
            .features
            .iter()
            .filter(|f| matches!(f.oxirs_support, FeatureStatus::Implemented))
            .count();
        let partial = self
            .features
            .iter()
            .filter(|f| matches!(f.oxirs_support, FeatureStatus::PartiallyImplemented { .. }))
            .count();
        let not_implemented = self
            .features
            .iter()
            .filter(|f| matches!(f.oxirs_support, FeatureStatus::NotImplemented))
            .count();
        let beyond_jena = self
            .features
            .iter()
            .filter(|f| matches!(f.oxirs_support, FeatureStatus::BeyondJena))
            .count();
        let jena_features_total = self.features.iter().filter(|f| f.jena_support).count();

        ParitySummary {
            total_features: self.features.len(),
            implemented,
            partial,
            not_implemented,
            beyond_jena,
            jena_features_total,
        }
    }

    /// Computes the weighted coverage percentage over all Jena features.
    pub fn weighted_coverage_percentage(&self) -> f64 {
        let summary = self.summary();
        summary.weighted_coverage(&self.features)
    }

    /// Generates a full Markdown parity report.
    pub fn generate_report(&self) -> String {
        let summary = self.summary();
        let mut report = String::with_capacity(16384);

        report.push_str("# OxiRS vs. Apache Jena: Feature Parity Report\n\n");
        report.push_str(
            "> [OK] = Implemented | [~] = Partial | [X] = Not Implemented | [+] = Beyond Jena\n\n",
        );

        report.push_str("## Summary\n\n");
        report.push_str("| Metric | Count |\n|--------|-------|\n");
        report.push_str(&format!(
            "| Total features tracked | {} |\n",
            summary.total_features
        ));
        report.push_str(&format!(
            "| Jena features covered (full) | {} |\n",
            summary.implemented
        ));
        report.push_str(&format!(
            "| Jena features covered (partial) | {} |\n",
            summary.partial
        ));
        report.push_str(&format!(
            "| Jena features missing | {} |\n",
            summary.not_implemented
        ));
        report.push_str(&format!(
            "| Beyond-Jena extensions | {} |\n",
            summary.beyond_jena
        ));
        report.push_str(&format!(
            "| Jena parity (full) | {:.0}% |\n",
            summary.jena_parity_percentage()
        ));
        report.push_str(&format!(
            "| Weighted coverage | {:.0}% |\n\n",
            self.weighted_coverage_percentage()
        ));

        let categories = [
            ParityCategory::QueryLanguage,
            ParityCategory::DataFormats,
            ParityCategory::Reasoning,
            ParityCategory::Storage,
            ParityCategory::Protocols,
            ParityCategory::Security,
            ParityCategory::Networking,
            ParityCategory::Geospatial,
            ParityCategory::Validation,
            ParityCategory::Streaming,
            ParityCategory::Ai,
            ParityCategory::Tooling,
            ParityCategory::Industrial,
        ];

        report.push_str("## Feature Details by Category\n\n");
        for category in &categories {
            let features = self.by_category(category);
            if features.is_empty() {
                continue;
            }

            let cat_impl = features.iter().filter(|f| f.is_at_parity()).count();
            report.push_str(&format!(
                "### {} ({}/{} at parity)\n\n",
                category.label(),
                cat_impl,
                features.len()
            ));
            report.push_str("| Feature | Jena | OxiRS | Notes |\n");
            report.push_str("|---------|------|-------|-------|\n");
            for f in &features {
                let jena = if f.jena_support { "Yes" } else { "No" };
                let notes = f.notes.unwrap_or("");
                report.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    f.name,
                    jena,
                    f.oxirs_support.indicator(),
                    notes
                ));
            }
            report.push('\n');
        }

        report
    }
}

impl Default for JenaParityChecker {
    fn default() -> Self {
        Self::new()
    }
}
