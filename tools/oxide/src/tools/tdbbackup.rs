//\! TDB Backup tool

use super::ToolResult;
use std::path::PathBuf;

pub async fn run(
    source: PathBuf,
    target: PathBuf,
    compress: bool,
    incremental: bool,
) -> ToolResult {
    println!("TDB Backup tool not yet implemented");
    Ok(())
}
