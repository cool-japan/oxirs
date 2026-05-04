//! SPARQL Update parser tool
//!
//! Parses a SPARQL 1.1 Update string (or a file containing one) using
//! the `oxirs-arq` crate's `SparqlUpdateParser` and reports success or failure.
//! When `_print_ast` is enabled the parsed update structure is pretty-printed
//! using Rust's standard `Debug` formatter.

use super::ToolResult;
use oxirs_arq::{SparqlUpdate, SparqlUpdateParser};
use std::io::Read;

/// Read a SPARQL Update string from a file path.
fn read_update_from_file(path: &str) -> ToolResult<String> {
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("cannot open file {path:?}: {e}"))?;
    let mut content = String::new();
    file.read_to_string(&mut content)
        .map_err(|e| format!("cannot read file {path:?}: {e}"))?;
    Ok(content)
}

/// Pretty-print a single `SparqlUpdate` variant for human inspection.
fn describe_update(update: &SparqlUpdate) -> String {
    format!("{update:#?}")
}

/// Run the SPARQL Update parse tool.
///
/// # Parameters
/// - `_update`:    The SPARQL Update text **or** a file-system path (see `_file`).
/// - `_file`:      When `true`, read the update string from the path in `_update`.
/// - `_print_ast`: When `true`, pretty-print the parsed AST to stdout.
pub async fn run(_update: String, _file: bool, _print_ast: bool) -> ToolResult {
    let update_text = if _file {
        read_update_from_file(&_update)?
    } else {
        _update
    };

    match SparqlUpdateParser::parse(&update_text) {
        Ok(updates) => {
            let count = updates.len();
            println!(
                "Successfully parsed {} SPARQL Update operation{}.",
                count,
                if count == 1 { "" } else { "s" }
            );

            if _print_ast {
                for (idx, update) in updates.iter().enumerate() {
                    println!("\n--- Operation {} ---", idx + 1);
                    println!("{}", describe_update(update));
                }
            }
        }
        Err(parse_err) => {
            let msg = format!("SPARQL Update parse error: {parse_err}");
            println!("{msg}");
            return Err(msg.into());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const INSERT_DATA: &str =
        "INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }";
    const DELETE_DATA: &str =
        "DELETE DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }";

    #[test]
    fn test_describe_insert_data() {
        let updates = SparqlUpdateParser::parse(INSERT_DATA).expect("should parse");
        assert_eq!(updates.len(), 1);
        let desc = describe_update(&updates[0]);
        assert!(desc.contains("InsertData") || desc.contains("Insert"));
    }

    #[tokio::test]
    async fn test_run_insert_data_no_ast() {
        run(INSERT_DATA.to_string(), false, false)
            .await
            .expect("should parse INSERT DATA");
    }

    #[tokio::test]
    async fn test_run_delete_data_with_ast() {
        run(DELETE_DATA.to_string(), false, true)
            .await
            .expect("should parse DELETE DATA with AST");
    }

    #[tokio::test]
    async fn test_run_invalid_update_returns_error() {
        let result = run("NOT A VALID UPDATE".to_string(), false, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_from_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxirs_uparse_test.rq");
        {
            let mut f = std::fs::File::create(&path).expect("create temp file");
            f.write_all(INSERT_DATA.as_bytes()).expect("write");
        }
        let path_str = path.to_string_lossy().into_owned();
        run(path_str, true, true)
            .await
            .expect("should read and parse from file");
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_run_missing_file_returns_error() {
        let result = run(
            "/tmp/oxirs_uparse_nonexistent_xyz.rq".to_string(),
            true,
            false,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_multiple_operations() {
        let multi = format!("{INSERT_DATA} ; {DELETE_DATA}");
        run(multi, false, true)
            .await
            .expect("should parse multiple operations");
    }
}
