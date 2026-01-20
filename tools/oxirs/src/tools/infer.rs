//! Inference tool

use super::ToolResult;
use std::path::PathBuf;

pub async fn run(
    _data: PathBuf,
    _ontology: Option<PathBuf>,
    _profile: String,
    _output: Option<PathBuf>,
    _format: String,
) -> ToolResult {
    println!("Inference tool not yet implemented");
    Ok(())
}
