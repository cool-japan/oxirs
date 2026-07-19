//! GraphQL HTTP server implementation

use crate::ast::Document;
use crate::execution::{ExecutionContext, FieldResolver, QueryExecutor};
use crate::rate_limiting::{RateLimitConfig, RateLimiter};
use crate::types::Schema;
use crate::validation::{QueryValidator, ValidationConfig};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tracing::{error, info, warn};

/// Default cap on the number of bytes the server will read for the request
/// headers (the request line plus all header fields, up to and including
/// the terminating blank line). This bounds memory use for clients that
/// never send a complete header block.
const DEFAULT_MAX_HEADER_SIZE: usize = 64 * 1024;

/// Default cap on the number of bytes accepted for a request body.
const DEFAULT_MAX_BODY_SIZE: usize = 10 * 1024 * 1024;

/// Default idle/read timeout applied to every socket read while parsing a
/// request. Guards against Slowloris-style connections that open a socket
/// and then trickle bytes (or send none at all).
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Default cap on the number of concurrent connections being served.
const DEFAULT_MAX_CONNECTIONS: usize = 1024;

/// GraphQL request payload
#[derive(Debug, Deserialize)]
struct GraphQLRequest {
    query: String,
    variables: Option<HashMap<String, JsonValue>>,
    operation_name: Option<String>,
}

/// GraphQL response payload
#[derive(Debug, Serialize)]
struct GraphQLResponse {
    data: Option<JsonValue>,
    errors: Option<Vec<GraphQLErrorResponse>>,
}

/// GraphQL error in response format
#[derive(Debug, Serialize)]
struct GraphQLErrorResponse {
    message: String,
    locations: Option<Vec<Location>>,
    path: Option<Vec<String>>,
    extensions: Option<HashMap<String, JsonValue>>,
}

/// Error location information
#[derive(Debug, Serialize)]
struct Location {
    line: u32,
    column: u32,
}

/// GraphQL HTTP server
pub struct Server {
    executor: QueryExecutor,
    enable_playground: bool,
    enable_introspection: bool,
    validator: Option<QueryValidator>,
    enable_validation: bool,
    max_header_size: usize,
    max_body_size: usize,
    read_timeout: Duration,
    max_connections: usize,
    rate_limiter: Option<Arc<RateLimiter>>,
}

impl Server {
    pub fn new(schema: Schema) -> Self {
        Self {
            executor: QueryExecutor::new(schema),
            enable_playground: true,
            enable_introspection: true,
            validator: None,
            enable_validation: false,
            max_header_size: DEFAULT_MAX_HEADER_SIZE,
            max_body_size: DEFAULT_MAX_BODY_SIZE,
            read_timeout: DEFAULT_READ_TIMEOUT,
            max_connections: DEFAULT_MAX_CONNECTIONS,
            rate_limiter: None,
        }
    }

    pub fn with_playground(mut self, enable: bool) -> Self {
        self.enable_playground = enable;
        self
    }

    pub fn with_introspection(mut self, enable: bool) -> Self {
        self.enable_introspection = enable;
        self
    }

    pub fn with_validation(mut self, config: ValidationConfig, schema: Schema) -> Self {
        self.validator = Some(QueryValidator::new(config, schema));
        self.enable_validation = true;
        self
    }

    pub fn with_validation_enabled(mut self, enable: bool) -> Self {
        self.enable_validation = enable;
        self
    }

    /// Set the maximum accepted request body size in bytes (`Content-Length`
    /// beyond this value is rejected with `413 Payload Too Large`).
    pub fn with_max_body_size(mut self, max_body_size: usize) -> Self {
        self.max_body_size = max_body_size;
        self
    }

    /// Set the maximum accepted request header size in bytes.
    pub fn with_max_header_size(mut self, max_header_size: usize) -> Self {
        self.max_header_size = max_header_size;
        self
    }

    /// Set the idle/read timeout applied to each socket read while parsing
    /// a request (covers both the header phase and the body phase).
    pub fn with_read_timeout(mut self, read_timeout: Duration) -> Self {
        self.read_timeout = read_timeout;
        self
    }

    /// Set the maximum number of connections served concurrently. Additional
    /// connections are rejected with `503 Service Unavailable` immediately.
    pub fn with_max_connections(mut self, max_connections: usize) -> Self {
        self.max_connections = max_connections;
        self
    }

    /// Enable per-client-IP rate limiting for incoming requests, using the
    /// full [`RateLimiter`] (token bucket / sliding window / fixed window /
    /// adaptive) implemented in [`crate::rate_limiting`]. Requests that
    /// exceed the configured limit are rejected with `429 Too Many
    /// Requests` before any GraphQL parsing/execution happens.
    pub fn with_rate_limiting(mut self, config: RateLimitConfig) -> Self {
        self.rate_limiter = Some(Arc::new(RateLimiter::new(config)));
        self
    }

    pub fn add_resolver(&mut self, type_name: String, resolver: Arc<dyn FieldResolver>) {
        self.executor.add_resolver(type_name, resolver);
    }

    /// Mark `type_name` as eagerly resolved by the executor -- see
    /// [`crate::execution::QueryExecutor::add_eager_type`].
    pub fn add_eager_object_type(&mut self, type_name: impl Into<String>) {
        self.executor.add_eager_type(type_name);
    }

    pub async fn start(self, addr: SocketAddr) -> Result<()> {
        info!("Starting GraphQL server on {}", addr);

        let listener = TcpListener::bind(addr).await?;
        info!("GraphQL server listening on {}", addr);

        let max_connections = self.max_connections;
        let server = Arc::new(self);
        let connection_semaphore = Arc::new(Semaphore::new(max_connections));

        loop {
            match listener.accept().await {
                Ok((stream, peer_addr)) => {
                    let server = Arc::clone(&server);
                    let semaphore = Arc::clone(&connection_semaphore);
                    tokio::spawn(async move {
                        let permit = match semaphore.try_acquire_owned() {
                            Ok(permit) => permit,
                            Err(_) => {
                                warn!(
                                    "Connection limit ({}) reached, rejecting connection from {}",
                                    max_connections, peer_addr
                                );
                                let mut stream = stream;
                                let response = Server::static_service_unavailable_response();
                                let _ = stream.write_all(response.as_bytes()).await;
                                let _ = stream.flush().await;
                                return;
                            }
                        };
                        if let Err(e) = server.handle_connection(stream, peer_addr).await {
                            error!("Connection error from {}: {}", peer_addr, e);
                        }
                        drop(permit);
                    });
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                }
            }
        }
    }

    /// A minimal, allocation-light `503` response used when the connection
    /// cap has been reached, before any per-connection state exists.
    fn static_service_unavailable_response() -> &'static str {
        "HTTP/1.1 503 Service Unavailable\r\n\
         Content-Type: text/plain\r\n\
         Content-Length: 24\r\n\
         Connection: close\r\n\
         \r\n\
         Too many connections\r\n"
    }

    /// Read a complete HTTP/1.x request from `stream`: the header block
    /// (terminated by an empty line) followed by a `Content-Length`-bound
    /// body, if any. Every individual socket read is bounded by
    /// `self.read_timeout` so a client that stalls mid-request cannot pin
    /// the connection open indefinitely (Slowloris protection). Returns
    /// `Ok(None)` when the response has already been written directly to
    /// the stream (e.g. a `4xx`/`413` early-rejection) and the caller
    /// should simply close the connection.
    async fn read_request(
        &self,
        stream: &mut tokio::net::TcpStream,
    ) -> Result<Option<(String, Vec<u8>)>> {
        let mut buf: Vec<u8> = Vec::with_capacity(4096);
        let mut chunk = [0u8; 4096];

        // 1. Read until we see the header terminator "\r\n\r\n", bounded by
        //    max_header_size to protect against unbounded header growth.
        let header_end = loop {
            if let Some(pos) = find_subsequence(&buf, b"\r\n\r\n") {
                break pos;
            }

            if buf.len() >= self.max_header_size {
                let response =
                    self.create_http_response(431, "Request Header Fields Too Large", "text/plain");
                stream.write_all(response.as_bytes()).await?;
                stream.flush().await?;
                return Ok(None);
            }

            let n = tokio::time::timeout(self.read_timeout, stream.read(&mut chunk))
                .await
                .map_err(|_| anyhow::anyhow!("timed out waiting for request headers"))??;

            if n == 0 {
                // Peer closed the connection before sending a full header block.
                return Ok(None);
            }

            buf.extend_from_slice(&chunk[..n]);
        };

        let header_bytes = &buf[..header_end];
        let headers_str = String::from_utf8_lossy(header_bytes).into_owned();
        let mut body = buf[header_end + 4..].to_vec();

        // 2. Determine how much body (if any) is expected via Content-Length.
        let content_length = parse_content_length(&headers_str);
        let method = headers_str
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().next())
            .unwrap_or("");

        if let Some(content_length) = content_length {
            if content_length > self.max_body_size {
                let response = self.create_http_response(413, "Payload Too Large", "text/plain");
                stream.write_all(response.as_bytes()).await?;
                stream.flush().await?;
                return Ok(None);
            }

            while body.len() < content_length {
                let n = tokio::time::timeout(self.read_timeout, stream.read(&mut chunk))
                    .await
                    .map_err(|_| anyhow::anyhow!("timed out waiting for request body"))??;

                if n == 0 {
                    // Peer closed before sending the full declared body;
                    // treat as a bad request rather than silently truncating.
                    let response = self.create_http_response(400, "Bad Request", "text/plain");
                    stream.write_all(response.as_bytes()).await?;
                    stream.flush().await?;
                    return Ok(None);
                }

                body.extend_from_slice(&chunk[..n]);

                if body.len() > self.max_body_size {
                    let response =
                        self.create_http_response(413, "Payload Too Large", "text/plain");
                    stream.write_all(response.as_bytes()).await?;
                    stream.flush().await?;
                    return Ok(None);
                }
            }
            body.truncate(content_length);
        } else if method.eq_ignore_ascii_case("POST") {
            // A POST with no Content-Length and no chunked-encoding support
            // in this minimal server cannot be read safely (we would not
            // know when the body ends), so fail loudly instead of silently
            // truncating or hanging.
            let response = self.create_http_response(411, "Length Required", "text/plain");
            stream.write_all(response.as_bytes()).await?;
            stream.flush().await?;
            return Ok(None);
        }

        Ok(Some((headers_str, body)))
    }

    async fn handle_connection(
        &self,
        mut stream: tokio::net::TcpStream,
        peer_addr: SocketAddr,
    ) -> Result<()> {
        let Some((headers_str, body)) = self.read_request(&mut stream).await? else {
            // A response (or nothing, on early EOF) was already written by
            // read_request; nothing further to do.
            return Ok(());
        };

        // Rate limit by client IP before any GraphQL parsing/execution.
        if let Some(ref rate_limiter) = self.rate_limiter {
            let client_id = peer_addr.ip().to_string();
            match rate_limiter.check_rate_limit(&client_id, None).await {
                Ok(result) if !result.allowed => {
                    warn!("Rate limit exceeded for {}", client_id);
                    let response = self.create_rate_limited_response(&result);
                    stream.write_all(response.as_bytes()).await?;
                    stream.flush().await?;
                    return Ok(());
                }
                Ok(_) => {}
                Err(e) => {
                    // Fail open on internal rate-limiter errors rather than
                    // dropping legitimate traffic because of a bookkeeping
                    // bug, but make the failure visible in logs.
                    error!("Rate limiter error for {}: {}", client_id, e);
                }
            }
        }

        // Reassemble a request string compatible with process_http_request's
        // existing parsing (request line + headers + blank line + body).
        let mut request = String::with_capacity(headers_str.len() + body.len() + 4);
        request.push_str(&headers_str);
        request.push_str("\r\n\r\n");
        request.push_str(&String::from_utf8_lossy(&body));

        let response = self.process_http_request(&request).await;

        stream.write_all(response.as_bytes()).await?;
        stream.flush().await?;

        Ok(())
    }

    /// Build a `429 Too Many Requests` response carrying the standard
    /// `Retry-After`/`X-RateLimit-*` headers from a rejected
    /// [`crate::rate_limiting::RateLimitResult`].
    fn create_rate_limited_response(
        &self,
        result: &crate::rate_limiting::RateLimitResult,
    ) -> String {
        let retry_after = result.retry_after.unwrap_or(result.reset_after);
        let body = r#"{"errors":[{"message":"Too many requests"}]}"#;
        format!(
            "HTTP/1.1 429 Too Many Requests\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Retry-After: {}\r\n\
             X-RateLimit-Limit: {}\r\n\
             X-RateLimit-Remaining: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            body.len(),
            retry_after,
            result.limit,
            result.remaining,
            body
        )
    }

    async fn process_http_request(&self, request: &str) -> String {
        // Parse HTTP request line
        let lines: Vec<&str> = request.lines().collect();
        if lines.is_empty() {
            return self.create_http_response(400, "Bad Request", "text/plain");
        }

        let request_line = lines[0];
        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 3 {
            return self.create_http_response(400, "Bad Request", "text/plain");
        }

        let method = parts[0];
        let path = parts[1];

        match (method, path) {
            ("GET", "/") if self.enable_playground => {
                self.create_http_response(200, &self.get_playground_html(), "text/html")
            }
            ("POST", "/graphql") => {
                // Find the JSON body
                if let Some(body_start) = request.find("\r\n\r\n") {
                    let body = &request[body_start + 4..];
                    match self.execute_graphql_from_json(body).await {
                        Ok(response) => {
                            self.create_http_response(200, &response, "application/json")
                        }
                        Err(_) => self.create_http_response(
                            500,
                            r#"{"errors":[{"message":"Internal server error"}]}"#,
                            "application/json",
                        ),
                    }
                } else {
                    self.create_http_response(400, "Bad Request", "text/plain")
                }
            }
            ("GET", "/graphql") => {
                // Simple test response
                let test_response = r#"{"data":{"hello":"Hello from OxiRS GraphQL!"}}"#;
                self.create_http_response(200, test_response, "application/json")
            }
            _ => self.create_http_response(404, "Not Found", "text/plain"),
        }
    }

    fn create_http_response(&self, status: u16, body: &str, content_type: &str) -> String {
        let status_text = match status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            411 => "Length Required",
            413 => "Payload Too Large",
            431 => "Request Header Fields Too Large",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            _ => "Unknown",
        };

        format!(
            "HTTP/1.1 {} {}\r\n\
             Content-Type: {}\r\n\
             Content-Length: {}\r\n\
             Access-Control-Allow-Origin: *\r\n\
             Access-Control-Allow-Headers: Content-Type\r\n\
             Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
             \r\n\
             {}",
            status,
            status_text,
            content_type,
            body.len(),
            body
        )
    }

    async fn execute_graphql_from_json(&self, body: &str) -> Result<String> {
        let request: GraphQLRequest = serde_json::from_str(body)?;

        // Parse the GraphQL document
        let document = self.parse_graphql_document(&request.query)?;

        // Validate the query if validation is enabled
        if self.enable_validation {
            if let Some(ref validator) = self.validator {
                let validation_result = validator.validate(&document)?;

                if !validation_result.is_valid {
                    // Return validation errors as GraphQL errors
                    let errors = validation_result
                        .errors
                        .into_iter()
                        .map(|err| GraphQLErrorResponse {
                            message: err.message,
                            locations: None,
                            path: if err.path.is_empty() {
                                None
                            } else {
                                Some(err.path)
                            },
                            extensions: Some({
                                let mut ext = HashMap::new();
                                ext.insert(
                                    "rule".to_string(),
                                    JsonValue::String(format!("{:?}", err.rule)),
                                );
                                ext
                            }),
                        })
                        .collect();

                    let response = GraphQLResponse {
                        data: None,
                        errors: Some(errors),
                    };

                    return Ok(serde_json::to_string(&response)?);
                }

                // Log validation warnings if any
                for warning in validation_result.warnings {
                    warn!("Query validation warning: {}", warning.message);
                    if let Some(suggestion) = warning.suggestion {
                        warn!("Suggestion: {}", suggestion);
                    }
                }
            }
        }

        // Convert variables to our Value type
        let variables = request
            .variables
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| (k, self.json_to_value(v)))
            .collect();

        // Create execution context
        let context = ExecutionContext::new()
            .with_variables(variables)
            .with_operation_name(request.operation_name.unwrap_or_default());

        // Execute the query
        let result = self.executor.execute(&document, &context).await?;

        // Format the response
        let response = GraphQLResponse {
            data: result.data,
            errors: if result.errors.is_empty() {
                None
            } else {
                Some(
                    result
                        .errors
                        .into_iter()
                        .map(|err| GraphQLErrorResponse {
                            message: err.message,
                            locations: err
                                .locations
                                .into_iter()
                                .map(|loc| Location {
                                    line: loc.line as u32,
                                    column: loc.column as u32,
                                })
                                .collect::<Vec<_>>()
                                .into(),
                            path: if err.path.is_empty() {
                                None
                            } else {
                                Some(err.path)
                            },
                            extensions: if err.extensions.is_empty() {
                                None
                            } else {
                                Some(err.extensions)
                            },
                        })
                        .collect(),
                )
            },
        };

        Ok(serde_json::to_string(&response)?)
    }

    fn parse_graphql_document(&self, query: &str) -> Result<Document> {
        crate::parser::parse_document(query)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn json_to_value(&self, json: JsonValue) -> crate::ast::Value {
        match json {
            JsonValue::Null => crate::ast::Value::NullValue,
            JsonValue::Bool(b) => crate::ast::Value::BooleanValue(b),
            JsonValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    crate::ast::Value::IntValue(i)
                } else if let Some(f) = n.as_f64() {
                    crate::ast::Value::FloatValue(f)
                } else {
                    crate::ast::Value::StringValue(n.to_string())
                }
            }
            JsonValue::String(s) => crate::ast::Value::StringValue(s),
            JsonValue::Array(arr) => crate::ast::Value::ListValue(
                arr.into_iter().map(|v| self.json_to_value(v)).collect(),
            ),
            JsonValue::Object(obj) => crate::ast::Value::ObjectValue(
                obj.into_iter()
                    .map(|(k, v)| (k, self.json_to_value(v)))
                    .collect(),
            ),
        }
    }

    fn get_playground_html(&self) -> String {
        r#"
<!DOCTYPE html>
<html>
<head>
    <title>GraphQL Playground</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .query-section {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        textarea, input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
            box-sizing: border-box;
        }
        textarea {
            height: 120px;
            resize: vertical;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        }
        pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 GraphQL Playground</h1>
        
        <div class="query-section">
            <label for="query">GraphQL Query:</label>
            <textarea id="query" placeholder="Enter your GraphQL query here...">query {
  hello
}</textarea>
        </div>
        
        <div class="query-section">
            <label for="variables">Variables (JSON):</label>
            <textarea id="variables" placeholder='{"key": "value"}'>{}</textarea>
        </div>
        
        <div class="query-section">
            <label for="operationName">Operation Name:</label>
            <input type="text" id="operationName" placeholder="Optional operation name">
        </div>
        
        <button onclick="executeQuery()">Execute Query</button>
        <button onclick="clearResult()">Clear</button>
        
        <div id="result" class="result" style="display: none;">
            <pre id="resultContent"></pre>
        </div>
    </div>

    <script>
        async function executeQuery() {
            const query = document.getElementById('query').value;
            const variables = document.getElementById('variables').value;
            const operationName = document.getElementById('operationName').value;
            
            let parsedVariables = {};
            if (variables.trim()) {
                try {
                    parsedVariables = JSON.parse(variables);
                } catch (e) {
                    showResult('Error parsing variables: ' + e.message);
                    return;
                }
            }
            
            const request = {
                query: query,
                variables: parsedVariables,
                operationName: operationName || null
            };
            
            try {
                const response = await fetch('/graphql', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(request)
                });
                
                const result = await response.json();
                showResult(JSON.stringify(result, null, 2));
            } catch (error) {
                showResult('Network error: ' + error.message);
            }
        }
        
        function showResult(content) {
            document.getElementById('resultContent').textContent = content;
            document.getElementById('result').style.display = 'block';
        }
        
        function clearResult() {
            document.getElementById('result').style.display = 'none';
        }
        
        // Allow Ctrl+Enter to execute query
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                executeQuery();
            }
        });
    </script>
</body>
</html>
        "#
        .to_string()
    }
}

impl Default for Server {
    fn default() -> Self {
        Self::new(Schema::new())
    }
}

/// Find the first occurrence of `needle` inside `haystack`, returning the
/// byte offset of its start.
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Parse the `Content-Length` header (case-insensitively) out of a raw
/// HTTP header block. Returns `None` if the header is absent or malformed.
fn parse_content_length(headers: &str) -> Option<usize> {
    headers.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name.trim().eq_ignore_ascii_case("content-length") {
            value.trim().parse::<usize>().ok()
        } else {
            None
        }
    })
}

#[cfg(test)]
mod http_read_loop_tests {
    use super::*;
    use tokio::io::AsyncWriteExt;
    use tokio::net::{TcpListener, TcpStream};

    #[test]
    fn test_find_subsequence() {
        assert_eq!(find_subsequence(b"abc\r\n\r\ndef", b"\r\n\r\n"), Some(3));
        assert_eq!(find_subsequence(b"no terminator here", b"\r\n\r\n"), None);
        assert_eq!(find_subsequence(b"", b"\r\n\r\n"), None);
    }

    #[test]
    fn test_parse_content_length() {
        let headers = "POST /graphql HTTP/1.1\r\nHost: localhost\r\nContent-Length: 42\r\n";
        assert_eq!(parse_content_length(headers), Some(42));

        let headers_ci = "POST /graphql HTTP/1.1\r\ncontent-length: 7\r\n";
        assert_eq!(parse_content_length(headers_ci), Some(7));

        let headers_missing = "GET / HTTP/1.1\r\nHost: localhost\r\n";
        assert_eq!(parse_content_length(headers_missing), None);
    }

    /// Regression test for the read loop: a POST body larger than a single
    /// 4096-byte TCP read (and split across multiple writes/segments) must
    /// be received in full rather than truncated.
    #[tokio::test]
    async fn test_handle_connection_reads_body_larger_than_one_read() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");

        let server = Arc::new(Server::new(Schema::new()));
        let server_task = {
            let server = Arc::clone(&server);
            tokio::spawn(async move {
                let (stream, peer_addr) = listener.accept().await.expect("accept");
                server.handle_connection(stream, peer_addr).await
            })
        };

        // Build a GraphQL query long enough that the JSON body exceeds the
        // old 4096-byte single-read cap.
        let padding = "x".repeat(8000);
        let query = format!("query {{ hello }} # {padding}");
        let body = serde_json::json!({ "query": query }).to_string();

        let request = format!(
            "POST /graphql HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let mut client = TcpStream::connect(addr).await.expect("connect");
        // Write in two chunks to also exercise the multi-read path.
        let mid = request.len() / 2;
        client
            .write_all(&request.as_bytes()[..mid])
            .await
            .expect("write first half");
        client
            .write_all(&request.as_bytes()[mid..])
            .await
            .expect("write second half");

        let result = server_task.await.expect("task join");
        assert!(result.is_ok(), "handle_connection failed: {result:?}");
    }

    /// A `Content-Length` above the configured cap must be rejected with
    /// `413` instead of being read into memory.
    #[tokio::test]
    async fn test_handle_connection_rejects_oversized_body() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");

        let server = Arc::new(Server::new(Schema::new()).with_max_body_size(16));
        let server_task = {
            let server = Arc::clone(&server);
            tokio::spawn(async move {
                let (stream, peer_addr) = listener.accept().await.expect("accept");
                server.handle_connection(stream, peer_addr).await
            })
        };

        let body = serde_json::json!({ "query": "query { hello }" }).to_string();
        let request = format!(
            "POST /graphql HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let mut client = TcpStream::connect(addr).await.expect("connect");
        client
            .write_all(request.as_bytes())
            .await
            .expect("write request");

        server_task
            .await
            .expect("task join")
            .expect("handle_connection ok");

        use tokio::io::AsyncReadExt;
        let mut response = Vec::new();
        let _ = client.read_to_end(&mut response).await;
        let response = String::from_utf8_lossy(&response);
        assert!(
            response.starts_with("HTTP/1.1 413"),
            "expected 413 response, got: {response}"
        );
    }

    /// A connection that never completes its headers within the configured
    /// read timeout must be dropped rather than hanging the serving task
    /// forever (Slowloris protection).
    #[tokio::test]
    async fn test_handle_connection_times_out_idle_client() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");

        let server =
            Arc::new(Server::new(Schema::new()).with_read_timeout(Duration::from_millis(100)));
        let server_task = {
            let server = Arc::clone(&server);
            tokio::spawn(async move {
                let (stream, peer_addr) = listener.accept().await.expect("accept");
                server.handle_connection(stream, peer_addr).await
            })
        };

        let mut client = TcpStream::connect(addr).await.expect("connect");
        // Send an incomplete header (no terminating blank line) and then
        // just hold the connection open without sending more data.
        client
            .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n")
            .await
            .expect("write partial request");

        let result = tokio::time::timeout(Duration::from_secs(5), server_task)
            .await
            .expect("server task should finish once the read times out")
            .expect("task join");
        assert!(result.is_err(), "expected a timeout error, got: {result:?}");
    }

    /// Regression test: the fully-implemented `rate_limiting` module used
    /// to sit unreferenced -- nothing in the server ever constructed a
    /// `RateLimiter` or consulted it. Once `with_rate_limiting` is used, a
    /// client that exceeds the configured request budget must be rejected
    /// with `429 Too Many Requests` before any GraphQL work happens.
    #[tokio::test]
    async fn test_rate_limiting_rejects_excess_requests() {
        use crate::rate_limiting::RateLimitConfig;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");

        let rate_limit_config = RateLimitConfig {
            max_requests: 1,
            window_seconds: 60,
            ..Default::default()
        };
        let server = Arc::new(Server::new(Schema::new()).with_rate_limiting(rate_limit_config));

        let server_task = {
            let server = Arc::clone(&server);
            tokio::spawn(async move {
                for _ in 0..2 {
                    let (stream, peer_addr) = listener.accept().await.expect("accept");
                    server
                        .handle_connection(stream, peer_addr)
                        .await
                        .expect("handle_connection ok");
                }
            })
        };

        use tokio::io::AsyncReadExt;

        // First request: within budget, should succeed normally.
        let mut client1 = TcpStream::connect(addr).await.expect("connect 1");
        client1
            .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .expect("write request 1");
        let mut response1 = Vec::new();
        let _ = client1.read_to_end(&mut response1).await;
        let response1 = String::from_utf8_lossy(&response1);
        assert!(
            response1.starts_with("HTTP/1.1 200"),
            "expected the first request to succeed, got: {response1}"
        );

        // Second request from the same client: over budget, must be
        // rejected with 429 before GraphQL/playground handling runs.
        let mut client2 = TcpStream::connect(addr).await.expect("connect 2");
        client2
            .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .expect("write request 2");
        let mut response2 = Vec::new();
        let _ = client2.read_to_end(&mut response2).await;
        let response2 = String::from_utf8_lossy(&response2);
        assert!(
            response2.starts_with("HTTP/1.1 429"),
            "expected the second request to be rate limited, got: {response2}"
        );

        server_task.await.expect("task join");
    }
}
