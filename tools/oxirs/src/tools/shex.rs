//! ShEx validation tool

use super::ToolResult;
use crate::cli::error::CliError;
use std::path::PathBuf;

pub async fn run(
    _data: Option<PathBuf>,
    _dataset: Option<PathBuf>,
    _schema: PathBuf,
    _shape_map: Option<PathBuf>,
    _format: String,
) -> ToolResult {
    Err(Box::new(
        CliError::unimplemented(
            "ShEx validation is not yet implemented. \
            For shape constraint validation, use `oxirs shacl` which supports SHACL 1.1 \
            (compatible with many ShEx use cases).",
        )
        .with_context("oxirs shex")
        .with_suggestion("Run `oxirs shacl --help` to see available SHACL validation options")
        .with_suggestion(
            "SHACL 1.1 covers most common ShEx constraint patterns including \
            node shapes, property shapes, and cardinality constraints",
        )
        .with_code("E-SHEX-001"),
    ))
}
