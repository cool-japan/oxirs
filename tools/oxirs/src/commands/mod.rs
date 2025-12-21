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
