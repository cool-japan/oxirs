//! CLI command implementations

use std::path::PathBuf;

pub mod init;
pub mod serve;
pub mod import;
pub mod export;
pub mod query;
pub mod update;
pub mod benchmark;
pub mod migrate;
pub mod config;

/// Common command result type
pub type CommandResult = Result<(), Box<dyn std::error::Error>>;