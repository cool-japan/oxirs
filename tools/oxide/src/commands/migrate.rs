//! Data migration command
use super::CommandResult;
use std::path::PathBuf;

pub async fn run(source: PathBuf, target: PathBuf, from: String, to: String) -> CommandResult {
    println!(
        "Migrating data from {source:?} ({from}) to {target:?} ({to})"
    );
    // TODO: Implement migration
    Ok(())
}
