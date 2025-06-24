//\! TDB Backup tool

use std::path::PathBuf;
use super::ToolResult;

pub async fn run(source: PathBuf, target: PathBuf, compress: bool, incremental: bool) -> ToolResult {
    println!("TDB Backup tool not yet implemented");
    Ok(())
}
