//! GraphQL HTTP server implementation

use crate::ast::Document;
use crate::execution::{ExecutionContext, FieldResolver, QueryExecutor};
use crate::types::Schema;
use crate::validation::{QueryValidator, ValidationConfig};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tracing::{error, info, warn};

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
}

impl Server {
    pub fn new(schema: Schema) -> Self {
        Self {
            executor: QueryExecutor::new(schema),
            enable_playground: true,
            enable_introspection: true,
            validator: None,
            enable_validation: false,
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

    pub fn add_resolver(&mut self, type_name: String, resolver: Arc<dyn FieldResolver>) {
        self.executor.add_resolver(type_name, resolver);
    }

    pub async fn start(self, addr: SocketAddr) -> Result<()> {
        info!("Starting GraphQL server on {}", addr);

        let listener = TcpListener::bind(addr).await?;
        info!("GraphQL server listening on {}", addr);

        let server = Arc::new(self);

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let server = Arc::clone(&server);
                    tokio::spawn(async move {
                        if let Err(e) = server.handle_connection(stream).await {
                            error!("Connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                }
            }
        }
    }

    async fn handle_connection(&self, mut stream: tokio::net::TcpStream) -> Result<()> {
        let mut buffer = [0; 4096];
        let n = stream.read(&mut buffer).await?;

        let request = String::from_utf8_lossy(&buffer[..n]);
        let response = self.process_http_request(&request).await;

        stream.write_all(response.as_bytes()).await?;
        stream.flush().await?;

        Ok(())
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
            500 => "Internal Server Error",
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
