//! Remote SPARQL tool

use super::ToolResult;
use std::path::PathBuf;

pub async fn run(
    service: String,
    query: Option<String>,
    query_file: Option<PathBuf>,
    results: String,
    timeout: u64,
) -> ToolResult {
    println!("Remote SPARQL tool not yet implemented");
    Ok(())
}
