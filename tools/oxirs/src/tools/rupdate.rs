//! Remote SPARQL Update tool

use super::ToolResult;
use std::path::PathBuf;

pub async fn run(
    _service: String,
    _update: Option<String>,
    _update_file: Option<PathBuf>,
    _timeout: u64,
) -> ToolResult {
    println!("Remote SPARQL Update tool not yet implemented");
    Ok(())
}
