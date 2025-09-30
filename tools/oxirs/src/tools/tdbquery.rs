//\! TDB Query tool

use super::ToolResult;
use std::path::PathBuf;

pub async fn run(_location: PathBuf, _query: String, _file: bool, _results: String) -> ToolResult {
    println!("TDB Query tool not yet implemented");
    Ok(())
}
