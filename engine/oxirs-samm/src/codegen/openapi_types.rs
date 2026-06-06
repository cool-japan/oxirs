use serde_json::{Map, Value};

/// Configuration for the `x-samm-pagination` OpenAPI extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaginationConfig {
    /// Maximum number of items returned per page.
    pub page_size: usize,
    /// Whether the pagination strategy uses opaque cursors (`true`) or numeric page offsets (`false`).
    pub cursor_based: bool,
    /// Optional HTTP response-header name where the server advertises the total result-set count.
    pub total_count_header: Option<String>,
}

impl Default for PaginationConfig {
    fn default() -> Self {
        Self {
            page_size: 10,
            cursor_based: false,
            total_count_header: None,
        }
    }
}

impl PaginationConfig {
    /// Serialise the config to the `x-samm-pagination` JSON object.
    pub fn to_extension_value(&self) -> Value {
        let mut obj = Map::new();
        obj.insert(
            "pageSize".to_string(),
            Value::Number(serde_json::Number::from(self.page_size)),
        );
        obj.insert("cursorBased".to_string(), Value::Bool(self.cursor_based));
        if let Some(ref header) = self.total_count_header {
            obj.insert(
                "totalCountHeader".to_string(),
                Value::String(header.clone()),
            );
        }
        Value::Object(obj)
    }
}

/// Target OpenAPI specification version for [`OpenApiGenerator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpenApiVersion {
    /// OpenAPI 3.0.3 (default, maximum compatibility).
    #[default]
    V30,
    /// OpenAPI 3.1.0, aligned with JSON Schema 2020-12.
    V31,
}

/// HTTP methods supported in generated path items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    /// HTTP GET
    Get,
    /// HTTP POST
    Post,
    /// HTTP PUT
    Put,
    /// HTTP PATCH
    Patch,
    /// HTTP DELETE
    Delete,
}

impl HttpMethod {
    /// Returns the lowercase string representation of the HTTP method.
    pub fn as_str(self) -> &'static str {
        match self {
            HttpMethod::Get => "get",
            HttpMethod::Post => "post",
            HttpMethod::Put => "put",
            HttpMethod::Patch => "patch",
            HttpMethod::Delete => "delete",
        }
    }
}

/// Configuration for [`OpenApiGenerator`].
#[derive(Debug, Clone)]
pub struct OpenApiOptions {
    /// Target OpenAPI specification version (3.0 or 3.1).
    pub version: OpenApiVersion,
    /// The base path prefix for all generated endpoints.
    pub base_path: String,
    /// API version string embedded in the `info` object.
    pub api_version: String,
    /// Whether to include a GET endpoint for reading the Aspect.
    pub include_get: bool,
    /// Whether to include a POST endpoint for creating Aspect instances.
    pub include_post: bool,
    /// Whether to include a PUT endpoint for updating Aspect instances.
    pub include_put: bool,
    /// Whether to include a DELETE endpoint.
    pub include_delete: bool,
    /// Prefer JSON Schema `$defs` (2020-12) style inside component schemas.
    pub use_defs_keyword: bool,
    /// Language used for description / title lookup.
    pub language: String,
    /// Optional pagination extension configuration.
    pub pagination: Option<PaginationConfig>,
}

impl Default for OpenApiOptions {
    fn default() -> Self {
        Self {
            version: OpenApiVersion::V30,
            base_path: "/api/v1/aspects".to_string(),
            api_version: "1.0.0".to_string(),
            include_get: true,
            include_post: false,
            include_put: false,
            include_delete: false,
            use_defs_keyword: false,
            language: "en".to_string(),
            pagination: None,
        }
    }
}
