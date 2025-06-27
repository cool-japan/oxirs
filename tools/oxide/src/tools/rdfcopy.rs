//! RDF Copy tool - Copy RDF datasets with format conversion

use super::ToolResult;
use std::path::PathBuf;

/// Run rdfcopy command
pub async fn run(
    source: PathBuf,
    target: PathBuf,
    _source_format: Option<String>,
    _target_format: Option<String>,
) -> ToolResult {
    println!("RDF Copy tool not yet implemented");
    println!("Source: {}", source.display());
    println!("Target: {}", target.display());
    Ok(())
}
