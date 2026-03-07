//! Command implementations for the Oxirs CLI
//!
//! This module contains implementations for all the CLI commands available
//! in the Oxirs CLI toolkit, providing comprehensive functionality for
//! RDF data management, SPARQL operations, and performance monitoring.

use crate::cli::CliResult;

/// Result type for command execution
pub type CommandResult = CliResult<()>;

/// Dataset initialization commands
pub mod init;

/// Server management commands  
pub mod serve;

pub mod export;
/// Data import/export commands
pub mod import;

/// Batch operations for high-performance processing
pub mod batch;

/// Query and update commands
pub mod query;
pub mod update;

/// Performance monitoring and profiling commands
pub mod performance;

/// Benchmarking commands
pub mod benchmark;

/// Migration and compatibility commands
pub mod migrate;

/// Dataset generation commands
pub mod generate;

/// Advanced RDF graph analytics using scirs2-graph
pub mod graph_analytics;

/// Index management commands
pub mod index;

/// Configuration management commands
pub mod config;

/// Interactive REPL mode
pub mod interactive;

/// SAMM Aspect Model commands (Java ESMF SDK compatible)
pub mod aspect;

/// AAS (Asset Administration Shell) commands (Java ESMF SDK compatible)
pub mod aas;

/// Package management commands (Java ESMF SDK compatible)
pub mod package;

/// Query EXPLAIN/ANALYZE commands
pub mod explain;

/// Query optimization analyzer
pub mod query_optimizer;

/// Intelligent query advisor with best practices
pub mod query_advisor;

/// Advanced statistical performance analyzer (SciRS2-powered)
pub mod performance_analyzer;

/// Performance optimization utilities (SciRS2-powered)
pub mod performance_optimizer;

/// ML-powered query performance predictor (SciRS2-powered)
pub mod query_predictor;

/// Query similarity detection and analysis
pub mod query_similarity;

/// Query template management
pub mod templates;

/// Query result caching
pub mod cache;

/// Query history management
pub mod history;

/// CI/CD integration commands
pub mod cicd;

/// Alias management commands
pub mod alias;

/// RDF graph visualization export
pub mod visualize;

/// ReBAC relationship management
pub mod rebac;

/// ReBAC manager implementation
pub mod rebac_manager;

/// Command stubs and utilities
pub mod stubs;

// === Phase D: Industrial Connectivity CLI Commands (0.1.0) ===

/// Time-series database commands
pub mod tsdb;

/// Modbus protocol commands
pub mod modbus;

/// CANbus/J1939 protocol commands
pub mod canbus;

/// Jena feature parity verification tool
pub mod jena_parity;

/// SPARQL query profiler with operator-level timing
pub mod query_profiler;

/// Query result caching with LRU eviction
pub mod result_cache;

/// Stream processing CLI commands
pub mod stream;

/// Streaming SPARQL query results (NDJSON, CSV-stream, TSV-stream)
pub mod query_stream;

/// ML-based query optimization advisor
pub mod ml_advisor;

/// Query history store with analytics and CSV export
pub mod query_history;

/// RDF dataset profiler with statistical analysis and quality checks
pub mod data_profiler;

/// RDF schema inferencer (class/property discovery, domain/range, cardinality)
pub mod schema_inferencer;

/// SPARQL benchmark runner with statistical metrics and regression detection
pub mod benchmark_runner;

/// SPARQL query validator (syntax, structure, prefixes, variables)
pub mod query_validator;

/// Multi-format RDF data exporter (Turtle, N-Triples, N-Quads, JSON-LD, RDF/XML, TriG, CSV)
pub mod export_command;

/// Multi-format RDF data importer (Turtle, N-Triples, N-Quads, JSON-LD, RDF/XML, TriG, CSV)
pub mod import_command;

/// SHACL/RDF validation CLI command (simulated SHACL validation with text/JSON/Turtle output)
pub mod validate_command;

/// RDF format conversion CLI command (N-Triples, N-Quads, Turtle, TriG, JSON-LD, RDF/XML, CSV)
pub mod convert_command;

/// RDF graph diff/comparison command (added/removed/common triples + similarity score)
pub mod diff_command;

/// RDF dataset statistics command (triple counts, unique URIs, predicate analysis)
pub mod stats_command;

/// Real-time SPARQL endpoint monitoring (latency, uptime, P95, health checks)
pub mod monitor_command;

/// RDF/SPARQL linting command (empty prefix, undeclared prefix, duplicate triples, long literals)
pub mod lint_command;

/// RDF merge command (set-union, blank node renaming, conflict detection, provenance tracking)
pub mod merge_command;

/// SPARQL query command with multi-format output (Table, JSON, CSV, TSV) and validation
pub mod query_command;

/// RDF graph inspection command (triple count, predicates, subjects, namespaces, connectivity, object types)
pub mod inspect_command;

/// SPARQL query profiler CLI command (analyzes query complexity and estimates cost)
pub mod profile_command;

/// SPARQL endpoint benchmark CLI command (simulated load testing with statistical analysis)
pub mod benchmark_command;

/// Server lifecycle management command (config validation + dry-run start)
pub mod serve_command;
