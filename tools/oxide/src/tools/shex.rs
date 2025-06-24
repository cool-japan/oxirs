//! ShEx validation tool

use std::path::PathBuf;
use super::ToolResult;

pub async fn run(_data: Option<PathBuf>, _dataset: Option<PathBuf>, _schema: PathBuf, _shape_map: Option<PathBuf>, _format: String) -> ToolResult {
    println!("ShEx validation tool not yet implemented");
    Ok(())
}
