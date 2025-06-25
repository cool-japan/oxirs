//! RDF Concatenation Tool
//!
//! Concatenates multiple RDF files and optionally converts format.

use super::{utils, ToolResult};
use std::path::PathBuf;

/// Run rdfcat command
pub async fn run(files: Vec<PathBuf>, format: String, output: Option<PathBuf>) -> ToolResult {
    println!("RDF Concatenation Tool");
    println!("Input files: {}", files.len());
    println!("Output format: {}", format);

    if !utils::is_supported_output_format(&format) {
        return Err(format!("Unsupported output format: {}", format).into());
    }

    // TODO: Implement RDF file concatenation
    println!("RDF concatenation not yet implemented");

    Ok(())
}
