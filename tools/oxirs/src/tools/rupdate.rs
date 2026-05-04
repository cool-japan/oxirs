//! Remote SPARQL Update tool
//!
//! Send a SPARQL Update request via HTTP POST to a remote SPARQL endpoint
//! using the SPARQL 1.1 Protocol (application/sparql-update body).

use super::ToolResult;
use std::path::PathBuf;
use std::time::Duration;

/// Run rupdate command — POST a SPARQL Update to a remote endpoint.
pub async fn run(
    _service: String,
    _update: Option<String>,
    _update_file: Option<PathBuf>,
    _timeout: u64,
) -> ToolResult {
    let update_string = resolve_update_string(_update, _update_file)?;

    println!("Endpoint : {_service}");
    println!("Update   :");
    println!("{update_string}");
    println!();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(_timeout))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    let response = client
        .post(&_service)
        .header("Content-Type", "application/sparql-update")
        .header("Accept", "*/*")
        .body(update_string)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    let status = response.status();
    println!("HTTP {}", status.as_u16());

    let body = response.text().await.unwrap_or_else(|_| String::new());

    if !body.is_empty() {
        println!("{body}");
    }

    if !status.is_success() {
        return Err(format!("SPARQL Update failed: HTTP {}", status.as_u16()).into());
    }

    println!("Update OK");
    Ok(())
}

/// Resolve the update string from either an inline string or a file path.
fn resolve_update_string(
    update: Option<String>,
    update_file: Option<PathBuf>,
) -> ToolResult<String> {
    match (update, update_file) {
        (Some(u), None) => Ok(u),
        (None, Some(path)) => {
            if !path.exists() {
                return Err(format!("Update file not found: {}", path.display()).into());
            }
            std::fs::read_to_string(&path)
                .map_err(|e| format!("Cannot read update file {}: {e}", path.display()).into())
        }
        (Some(_), Some(_)) => Err("Specify either --update or --update-file, not both".into()),
        (None, None) => Err("No update provided. Use --update or --update-file".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_resolve_update_inline() {
        let s =
            resolve_update_string(Some("INSERT DATA {}".to_string()), None).expect("inline update");
        assert_eq!(s, "INSERT DATA {}");
    }

    #[test]
    fn test_resolve_update_file() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        write!(tmp, "DELETE DATA {{ <http://s> <http://p> <http://o> . }}").expect("write");
        let s = resolve_update_string(None, Some(tmp.path().to_path_buf())).expect("file update");
        assert!(s.contains("DELETE DATA"));
    }

    #[test]
    fn test_resolve_both_is_error() {
        let result = resolve_update_string(
            Some("INSERT DATA {}".to_string()),
            Some(PathBuf::from("/tmp/x.sparql")),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_neither_is_error() {
        let result = resolve_update_string(None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_missing_file() {
        let result = resolve_update_string(None, Some(PathBuf::from("/nonexistent/update.sparql")));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_with_mock_server() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/sparql"))
            .and(header("content-type", "application/sparql-update"))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        let url = format!("{}/sparql", server.uri());
        let result = run(url, Some("INSERT DATA {}".to_string()), None, 30).await;
        assert!(result.is_ok(), "HTTP 204 should be success: {result:?}");
    }

    #[tokio::test]
    async fn test_run_server_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/sparql"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&server)
            .await;

        let url = format!("{}/sparql", server.uri());
        let result = run(url, Some("INSERT DATA {}".to_string()), None, 30).await;
        assert!(result.is_err(), "HTTP 500 should fail");
    }
}
