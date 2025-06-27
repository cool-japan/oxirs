//! RDF Diff tool

use super::ToolResult;
use std::path::PathBuf;

pub async fn run(_first: PathBuf, _second: PathBuf, _format: String) -> ToolResult {
    println!("RDF Diff tool not yet implemented");
    Ok(())
}
