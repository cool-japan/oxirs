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

/// Query and update commands
pub mod query;
pub mod update;

/// Performance monitoring and profiling commands
pub mod performance;

/// Benchmarking commands
pub mod benchmark;

/// Migration and compatibility commands
pub mod migrate;

/// Configuration management commands
pub mod config;

/// Command stubs and utilities
pub mod stubs;
