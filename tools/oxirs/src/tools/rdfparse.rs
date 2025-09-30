//! RDF Parse tool

use super::ToolResult;
use std::path::PathBuf;

pub async fn run(_file: PathBuf, _format: Option<String>, _base: Option<String>) -> ToolResult {
    println!("RDF Parse tool not yet implemented");
    Ok(())
}
