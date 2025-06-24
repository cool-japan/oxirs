//! Remote SPARQL Update tool

use std::path::PathBuf;
use super::ToolResult;

pub async fn run(service: String, update: Option<String>, update_file: Option<PathBuf>, timeout: u64) -> ToolResult {
    println!("Remote SPARQL Update tool not yet implemented");
    Ok(())
}