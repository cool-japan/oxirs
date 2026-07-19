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

/// Term-conversion bridge between `oxirs-core` and `oxirs-tdb` term types,
/// plus tdb2-dataset auto-detection, shared by `import`/`query`/`export`.
pub mod tdb_convert;

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
pub mod interactive_session;
mod interactive_tests;
pub mod interactive_types;

/// SAMM Aspect Model commands (Java ESMF SDK compatible)
pub mod aspect;
pub mod aspect_analyzer;
pub mod aspect_analyzer_formats;
pub mod aspect_analyzer_runner;
mod aspect_analyzer_tests;
pub mod aspect_analyzer_types;

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

/// Core data types for the Jena parity verifier
pub mod jena_parity_types;

/// The Jena parity checker engine
pub mod jena_parity_checker;

#[cfg(test)]
mod jena_parity_tests;

/// SPARQL query profiler with operator-level timing
pub mod query_profiler;

/// Query result caching with LRU eviction
pub mod result_cache;

/// Stream processing CLI commands
pub mod stream;

/// Query history store with analytics and CSV export
pub mod query_history;

/// RDF dataset profiler with statistical analysis and quality checks.
///
/// Also the home of the consolidated `inspect` command (triple/subject/predicate
/// counts, namespaces, connectivity, object-type distribution, quality checks) —
/// the former `stats_command` and `inspect_command` modules were merged here.
pub mod data_profiler;

/// RDF schema inferencer (class/property discovery, domain/range, cardinality)
pub mod schema_inferencer;

/// SPARQL query validator (syntax, structure, prefixes, variables)
pub mod query_validator;

/// Real-time SPARQL endpoint monitoring (latency, uptime, P95, health checks)
pub mod monitor_command;

/// RDF/SPARQL linting command (empty prefix, undeclared prefix, duplicate triples, long literals)
pub mod lint_command;

/// RDF merge command (set-union, blank node renaming, conflict detection, provenance tracking)
pub mod merge_command;

/// Server lifecycle management command (config validation + dry-run start)
pub mod serve_command;
