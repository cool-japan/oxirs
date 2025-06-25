//! TDB Dump tool

use super::ToolResult;
use std::path::PathBuf;

pub async fn run(
    location: PathBuf,
    output: Option<PathBuf>,
    format: String,
    graph: Option<String>,
) -> ToolResult {
    println!("TDB Dump tool not yet implemented");
    Ok(())
}
