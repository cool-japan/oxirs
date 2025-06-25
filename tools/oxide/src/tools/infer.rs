//! Inference tool

use super::ToolResult;
use std::path::PathBuf;

pub async fn run(
    data: PathBuf,
    ontology: Option<PathBuf>,
    profile: String,
    output: Option<PathBuf>,
    format: String,
) -> ToolResult {
    println!("Inference tool not yet implemented");
    Ok(())
}
