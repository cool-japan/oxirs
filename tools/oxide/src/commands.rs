//! CLI command implementations

use std::path::PathBuf;

pub mod benchmark;
pub mod config;
pub mod export;
pub mod import;
pub mod init;
pub mod migrate;
pub mod query;
pub mod serve;
pub mod update;

/// Common command result type
pub type CommandResult = Result<(), Box<dyn std::error::Error>>;
