//! Remote SPARQL Update tool

use super::ToolResult;
use std::path::PathBuf;

pub async fn run(
    service: String,
    update: Option<String>,
    update_file: Option<PathBuf>,
    timeout: u64,
) -> ToolResult {
    println!("Remote SPARQL Update tool not yet implemented");
    Ok(())
}
