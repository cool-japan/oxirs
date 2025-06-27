//! Remote SPARQL tool

use super::ToolResult;
use std::path::PathBuf;

pub async fn run(
    _service: String,
    _query: Option<String>,
    _query_file: Option<PathBuf>,
    _results: String,
    _timeout: u64,
) -> ToolResult {
    println!("Remote SPARQL tool not yet implemented");
    Ok(())
}
