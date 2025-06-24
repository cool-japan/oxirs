//! TDB Dump tool

use std::path::PathBuf;
use super::ToolResult;

pub async fn run(location: PathBuf, output: Option<PathBuf>, format: String, graph: Option<String>) -> ToolResult {
    println!("TDB Dump tool not yet implemented");
    Ok(())
}