//! TDB Dump tool

use super::ToolResult;
use std::path::PathBuf;

pub async fn run(
    _location: PathBuf,
    _output: Option<PathBuf>,
    _format: String,
    _graph: Option<String>,
) -> ToolResult {
    println!("TDB Dump tool not yet implemented");
    Ok(())
}
