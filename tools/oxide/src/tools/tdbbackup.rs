//\! TDB Backup tool

use super::ToolResult;
use std::path::PathBuf;

pub async fn run(
    _source: PathBuf,
    _target: PathBuf,
    _compress: bool,
    _incremental: bool,
) -> ToolResult {
    println!("TDB Backup tool not yet implemented");
    Ok(())
}
