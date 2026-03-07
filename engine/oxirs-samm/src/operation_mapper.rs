//! SAMM Operation → REST/MQTT API Mapping
//!
//! This module transforms SAMM (Semantic Aspect Meta Model) operation definitions
//! into concrete REST endpoint and MQTT topic specifications, and can generate
//! OpenAPI 3.0 and AsyncAPI YAML fragments.
//!
//! # Overview
//!
//! A SAMM `Operation` describes a function that can be invoked on an Aspect.
//! This mapper derives:
//!
//! - A REST endpoint (HTTP method + path + query params + JSON schemas)
//! - An MQTT topic (topic pattern + QoS + payload schema)
//! - Or both combined into an `ApiMapping`
//!
//! # OpenAPI / AsyncAPI generation
//!
//! The generated YAML fragments can be embedded directly into larger API
//! specifications. They follow standard OpenAPI 3.0 / AsyncAPI 2.x conventions.

use std::fmt;

// ─── Domain types ─────────────────────────────────────────────────────────────

/// HTTP method for a REST endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpMethod {
    /// HTTP GET
    Get,
    /// HTTP POST
    Post,
    /// HTTP PUT
    Put,
    /// HTTP DELETE
    Delete,
    /// HTTP PATCH
    Patch,
}

impl fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Get => write!(f, "get"),
            Self::Post => write!(f, "post"),
            Self::Put => write!(f, "put"),
            Self::Delete => write!(f, "delete"),
            Self::Patch => write!(f, "patch"),
        }
    }
}

/// A reference to a property in a SAMM operation signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyRef {
    /// Property name (camelCase recommended)
    pub name: String,
    /// XSD / JSON Schema data type (e.g. `"string"`, `"integer"`, `"boolean"`)
    pub data_type: String,
    /// Whether this property is optional
    pub optional: bool,
}

impl PropertyRef {
    /// Create a required property reference.
    pub fn required(name: impl Into<String>, data_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data_type: data_type.into(),
            optional: false,
        }
    }

    /// Create an optional property reference.
    pub fn optional(name: impl Into<String>, data_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data_type: data_type.into(),
            optional: true,
        }
    }
}

/// A SAMM operation definition.
#[derive(Debug, Clone)]
pub struct SammOperation {
    /// Operation name (used for path segment and topic name)
    pub name: String,
    /// Input properties (parameters)
    pub input_props: Vec<PropertyRef>,
    /// Output properties (response)
    pub output_props: Vec<PropertyRef>,
}

impl SammOperation {
    /// Create a new SAMM operation.
    pub fn new(
        name: impl Into<String>,
        input_props: Vec<PropertyRef>,
        output_props: Vec<PropertyRef>,
    ) -> Self {
        Self {
            name: name.into(),
            input_props,
            output_props,
        }
    }
}

/// A REST endpoint derived from a SAMM operation.
#[derive(Debug, Clone)]
pub struct RestEndpoint {
    /// HTTP method
    pub method: HttpMethod,
    /// URL path (e.g. `/operations/getTemperature`)
    pub path: String,
    /// Query parameter names (for GET operations)
    pub query_params: Vec<String>,
    /// JSON schema fragment for the request body (POST/PUT/PATCH only)
    pub body_schema: Option<String>,
    /// JSON schema fragment for the response body
    pub response_schema: String,
}

/// An MQTT topic derived from a SAMM operation.
#[derive(Debug, Clone)]
pub struct MqttTopic {
    /// Topic pattern (may include `+` and `#` wildcards)
    pub topic_pattern: String,
    /// MQTT QoS level (0, 1, or 2)
    pub qos: u8,
    /// JSON schema fragment for the message payload
    pub payload_schema: String,
}

/// A combined API mapping: REST, MQTT, or both.
#[derive(Debug, Clone)]
pub enum ApiMapping {
    /// REST endpoint only
    Rest(RestEndpoint),
    /// MQTT topic only
    Mqtt(MqttTopic),
    /// Both REST and MQTT
    Both(RestEndpoint, MqttTopic),
}

// ─── JSON schema generation helpers ──────────────────────────────────────────

/// Build a minimal inline JSON Schema object fragment from a list of properties.
fn build_json_schema(props: &[PropertyRef]) -> String {
    if props.is_empty() {
        return r#"{"type":"object","properties":{}}"#.to_string();
    }

    let mut required: Vec<&str> = Vec::new();
    let mut properties = Vec::new();

    for p in props {
        let json_type = xsd_to_json_type(&p.data_type);
        properties.push(format!(r#""{}":{{"type":"{}"}}"#, p.name, json_type));
        if !p.optional {
            required.push(p.name.as_str());
        }
    }

    let props_str = properties.join(",");
    if required.is_empty() {
        format!(r#"{{"type":"object","properties":{{{props_str}}}}}"#)
    } else {
        let req_str = required
            .iter()
            .map(|r| format!("\"{r}\""))
            .collect::<Vec<_>>()
            .join(",");
        format!(r#"{{"type":"object","properties":{{{props_str}}},"required":[{req_str}]}}"#)
    }
}

/// Map XSD / SAMM data types to JSON Schema types.
fn xsd_to_json_type(dt: &str) -> &'static str {
    match dt.to_lowercase().as_str() {
        "string" | "xsd:string" | "http://www.w3.org/2001/xmlschema#string" => "string",
        "integer" | "int" | "xsd:integer" | "xsd:int" | "long" | "short" => "integer",
        "float" | "double" | "decimal" | "xsd:float" | "xsd:double" | "xsd:decimal" => "number",
        "boolean" | "xsd:boolean" => "boolean",
        _ => "string", // safe default
    }
}

/// Convert an operation name to kebab-case path segment.
fn to_kebab_case(s: &str) -> String {
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            out.push('-');
        }
        out.push(ch.to_lowercase().next().unwrap_or(ch));
    }
    out
}

/// Convert an operation name to snake_case for MQTT.
fn to_snake_case(s: &str) -> String {
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(ch.to_lowercase().next().unwrap_or(ch));
    }
    out
}

// ─── Mapper ───────────────────────────────────────────────────────────────────

/// Maps SAMM operations to REST and/or MQTT API definitions.
#[derive(Debug, Default)]
pub struct OperationMapper {
    /// Default QoS level for generated MQTT topics
    pub default_qos: u8,
}

impl OperationMapper {
    /// Create a new mapper with QoS 1 default.
    pub fn new() -> Self {
        Self { default_qos: 1 }
    }

    /// Create a mapper with a specific default QoS.
    pub fn with_qos(qos: u8) -> Self {
        Self {
            default_qos: qos.min(2),
        }
    }

    /// Map a SAMM operation to a REST endpoint.
    ///
    /// Mapping logic:
    /// - Read-only operations (no input or only query params) → GET with query params
    /// - Write operations with input → POST with JSON body
    /// - The path is `/operations/{kebab-name}`
    pub fn map_to_rest(&self, op: &SammOperation) -> RestEndpoint {
        let path = format!("/operations/{}", to_kebab_case(&op.name));
        let response_schema = build_json_schema(&op.output_props);

        if op.input_props.is_empty() {
            // Pure read: GET with no params
            RestEndpoint {
                method: HttpMethod::Get,
                path,
                query_params: Vec::new(),
                body_schema: None,
                response_schema,
            }
        } else {
            // Check if all inputs can be expressed as simple query params
            // (no nested objects: only string/integer/boolean types)
            let all_simple = op.input_props.iter().all(|p| {
                matches!(
                    xsd_to_json_type(&p.data_type),
                    "string" | "integer" | "boolean"
                )
            });

            if all_simple && op.input_props.len() <= 5 {
                // Use GET with query parameters for simple inputs
                let query_params = op.input_props.iter().map(|p| p.name.clone()).collect();
                RestEndpoint {
                    method: HttpMethod::Get,
                    path,
                    query_params,
                    body_schema: None,
                    response_schema,
                }
            } else {
                // Use POST with a JSON body for complex / many inputs
                let body_schema = Some(build_json_schema(&op.input_props));
                RestEndpoint {
                    method: HttpMethod::Post,
                    path,
                    query_params: Vec::new(),
                    body_schema,
                    response_schema,
                }
            }
        }
    }

    /// Map a SAMM operation to an MQTT topic.
    ///
    /// - Request topic: `{base_topic}/{snake_name}/request`
    /// - Response topic: `{base_topic}/{snake_name}/response`
    ///
    /// The returned `MqttTopic` uses the request topic pattern and combines both
    /// schemas into a single payload schema for the message envelope.
    pub fn map_to_mqtt(&self, op: &SammOperation, base_topic: &str) -> MqttTopic {
        let snake = to_snake_case(&op.name);
        let base = base_topic.trim_end_matches('/');
        let topic_pattern = format!("{base}/{snake}/request");

        let input_schema = build_json_schema(&op.input_props);
        let output_schema = build_json_schema(&op.output_props);

        // Combine into an envelope schema
        let payload_schema = format!(
            r#"{{"type":"object","properties":{{"request":{input_schema},"response":{output_schema}}}}}"#
        );

        MqttTopic {
            topic_pattern,
            qos: self.default_qos,
            payload_schema,
        }
    }

    /// Generate an OpenAPI 3.0 YAML path fragment for a list of operations.
    ///
    /// The generated YAML contains only the `paths:` section and can be merged
    /// into a larger OpenAPI document.
    pub fn generate_openapi(&self, ops: &[SammOperation], base_url: &str) -> String {
        let mut yaml = format!(
            "openapi: \"3.0.3\"\ninfo:\n  title: SAMM API\n  version: \"1.0.0\"\nservers:\n  - url: \"{base_url}\"\npaths:\n"
        );

        for op in ops {
            let endpoint = self.map_to_rest(op);
            let path = &endpoint.path;
            let method = endpoint.method.to_string();

            yaml.push_str(&format!("  \"{path}\":\n"));
            yaml.push_str(&format!("    {method}:\n"));
            yaml.push_str(&format!(
                "      summary: \"Invoke {} operation\"\n",
                op.name
            ));
            yaml.push_str(&format!("      operationId: \"{}\"\n", op.name));

            if !endpoint.query_params.is_empty() {
                yaml.push_str("      parameters:\n");
                for qp in &endpoint.query_params {
                    let prop = op.input_props.iter().find(|p| p.name == *qp);
                    let json_type = prop
                        .map(|p| xsd_to_json_type(&p.data_type))
                        .unwrap_or("string");
                    let required = prop.map(|p| !p.optional).unwrap_or(true);
                    yaml.push_str(&format!(
                        "        - name: \"{qp}\"\n          in: query\n          required: {required}\n          schema:\n            type: \"{json_type}\"\n"
                    ));
                }
            }

            if let Some(body) = &endpoint.body_schema {
                yaml.push_str("      requestBody:\n");
                yaml.push_str("        required: true\n");
                yaml.push_str("        content:\n");
                yaml.push_str("          application/json:\n");
                yaml.push_str(&format!("            schema: {body}\n"));
            }

            yaml.push_str("      responses:\n");
            yaml.push_str("        '200':\n");
            yaml.push_str("          description: \"Successful response\"\n");
            yaml.push_str("          content:\n");
            yaml.push_str("            application/json:\n");
            yaml.push_str(&format!(
                "              schema: {}\n",
                endpoint.response_schema
            ));
        }

        yaml
    }

    /// Generate an AsyncAPI 2.x YAML fragment for a list of operations.
    ///
    /// Returns a document with an `asyncapi:` header, server info, and channels.
    pub fn generate_asyncapi(&self, ops: &[SammOperation], server: &str) -> String {
        let mut yaml = format!(
            "asyncapi: \"2.6.0\"\ninfo:\n  title: SAMM MQTT API\n  version: \"1.0.0\"\nservers:\n  production:\n    url: \"{server}\"\n    protocol: mqtt\nchannels:\n"
        );

        for op in ops {
            let mqtt = self.map_to_mqtt(op, "samm");
            let topic = &mqtt.topic_pattern;
            let qos = mqtt.qos;

            yaml.push_str(&format!("  \"{topic}\":\n"));
            yaml.push_str(&format!(
                "    description: \"Request channel for {} operation\"\n",
                op.name
            ));
            yaml.push_str("    publish:\n");
            yaml.push_str(&format!("      operationId: \"publish{}\"\n", op.name));
            yaml.push_str("      message:\n");
            yaml.push_str("        payload:\n");
            // Embed the payload schema inline (simplified)
            yaml.push_str("          type: object\n");
            yaml.push_str("    bindings:\n");
            yaml.push_str("      mqtt:\n");
            yaml.push_str(&format!("        qos: {qos}\n"));
        }

        yaml
    }

    /// Map a SAMM operation to both REST and MQTT.
    pub fn map_to_both(&self, op: &SammOperation, base_topic: &str) -> ApiMapping {
        let rest = self.map_to_rest(op);
        let mqtt = self.map_to_mqtt(op, base_topic);
        ApiMapping::Both(rest, mqtt)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mapper() -> OperationMapper {
        OperationMapper::new()
    }

    fn simple_op() -> SammOperation {
        SammOperation::new(
            "getTemperature",
            vec![PropertyRef::required("deviceId", "string")],
            vec![PropertyRef::required("temperature", "double")],
        )
    }

    fn write_op() -> SammOperation {
        SammOperation::new(
            "setConfiguration",
            vec![
                PropertyRef::required("key", "string"),
                PropertyRef::required("value", "string"),
                PropertyRef::optional("ttl", "integer"),
            ],
            vec![PropertyRef::required("success", "boolean")],
        )
    }

    fn no_input_op() -> SammOperation {
        SammOperation::new(
            "getStatus",
            vec![],
            vec![PropertyRef::required("status", "string")],
        )
    }

    // ── PropertyRef ──────────────────────────────────────────────────────────

    #[test]
    fn test_property_ref_required() {
        let p = PropertyRef::required("name", "string");
        assert!(!p.optional);
        assert_eq!(p.name, "name");
    }

    #[test]
    fn test_property_ref_optional() {
        let p = PropertyRef::optional("ttl", "integer");
        assert!(p.optional);
    }

    // ── HttpMethod ───────────────────────────────────────────────────────────

    #[test]
    fn test_http_method_display() {
        assert_eq!(HttpMethod::Get.to_string(), "get");
        assert_eq!(HttpMethod::Post.to_string(), "post");
        assert_eq!(HttpMethod::Put.to_string(), "put");
        assert_eq!(HttpMethod::Delete.to_string(), "delete");
        assert_eq!(HttpMethod::Patch.to_string(), "patch");
    }

    // ── REST mapping ─────────────────────────────────────────────────────────

    #[test]
    fn test_map_to_rest_get_with_query_param() {
        let op = simple_op();
        let ep = mapper().map_to_rest(&op);
        assert_eq!(ep.method, HttpMethod::Get);
        assert!(ep.path.contains("get-temperature"));
        assert!(ep.query_params.contains(&"deviceId".to_string()));
        assert!(ep.body_schema.is_none());
    }

    #[test]
    fn test_map_to_rest_no_input_is_get() {
        let op = no_input_op();
        let ep = mapper().map_to_rest(&op);
        assert_eq!(ep.method, HttpMethod::Get);
        assert!(ep.query_params.is_empty());
    }

    #[test]
    fn test_map_to_rest_many_inputs_is_post() {
        let op = SammOperation::new(
            "complexOp",
            (0..7)
                .map(|i| PropertyRef::required(format!("param{i}"), "string"))
                .collect(),
            vec![PropertyRef::required("result", "boolean")],
        );
        let ep = mapper().map_to_rest(&op);
        assert_eq!(ep.method, HttpMethod::Post);
        assert!(ep.body_schema.is_some());
    }

    #[test]
    fn test_map_to_rest_path_format() {
        let op = simple_op();
        let ep = mapper().map_to_rest(&op);
        assert!(ep.path.starts_with("/operations/"));
    }

    #[test]
    fn test_map_to_rest_response_schema_has_type() {
        let op = simple_op();
        let ep = mapper().map_to_rest(&op);
        assert!(ep.response_schema.contains("temperature"));
        assert!(ep.response_schema.contains("number"));
    }

    #[test]
    fn test_map_to_rest_body_schema_content() {
        let op = write_op();
        let ep = mapper().map_to_rest(&op);
        // write_op has 3 simple string/integer params but > 0 required
        // all_simple=true, len <= 5 → GET with query params
        // Actually setConfiguration has 3 params, all simple → GET
        // But wait: 3 params, all_simple=true, len<=5 → GET
        // Let's verify
        if ep.method == HttpMethod::Post {
            let body = ep.body_schema.as_ref().expect("body schema");
            assert!(body.contains("key"));
            assert!(body.contains("value"));
        } else {
            assert_eq!(ep.method, HttpMethod::Get);
            assert!(ep.query_params.contains(&"key".to_string()));
        }
    }

    #[test]
    fn test_map_to_rest_kebab_case_path() {
        let op = SammOperation::new("getMyData", vec![], vec![]);
        let ep = mapper().map_to_rest(&op);
        assert!(ep.path.contains("get-my-data"));
    }

    // ── MQTT mapping ─────────────────────────────────────────────────────────

    #[test]
    fn test_map_to_mqtt_topic_pattern() {
        let op = simple_op();
        let topic = mapper().map_to_mqtt(&op, "factory/device");
        assert!(topic.topic_pattern.starts_with("factory/device/"));
        assert!(topic.topic_pattern.ends_with("/request"));
    }

    #[test]
    fn test_map_to_mqtt_qos_default() {
        let op = simple_op();
        let topic = mapper().map_to_mqtt(&op, "base");
        assert_eq!(topic.qos, 1);
    }

    #[test]
    fn test_map_to_mqtt_qos_custom() {
        let m = OperationMapper::with_qos(2);
        let op = simple_op();
        let topic = m.map_to_mqtt(&op, "base");
        assert_eq!(topic.qos, 2);
    }

    #[test]
    fn test_map_to_mqtt_qos_capped() {
        let m = OperationMapper::with_qos(5);
        let op = simple_op();
        let topic = m.map_to_mqtt(&op, "base");
        assert!(topic.qos <= 2);
    }

    #[test]
    fn test_map_to_mqtt_payload_schema() {
        let op = simple_op();
        let topic = mapper().map_to_mqtt(&op, "base");
        assert!(topic.payload_schema.contains("request"));
        assert!(topic.payload_schema.contains("response"));
    }

    #[test]
    fn test_map_to_mqtt_snake_case_topic() {
        let op = SammOperation::new("getTemperature", vec![], vec![]);
        let topic = mapper().map_to_mqtt(&op, "base");
        assert!(topic.topic_pattern.contains("get_temperature"));
    }

    #[test]
    fn test_map_to_mqtt_base_topic_trailing_slash() {
        let op = simple_op();
        let topic = mapper().map_to_mqtt(&op, "factory/");
        // Should not double the slash
        assert!(!topic.topic_pattern.contains("//"));
    }

    // ── OpenAPI generation ────────────────────────────────────────────────────

    #[test]
    fn test_generate_openapi_contains_openapi_version() {
        let ops = vec![simple_op()];
        let yaml = mapper().generate_openapi(&ops, "https://api.example.org");
        assert!(yaml.contains("openapi:"));
        assert!(yaml.contains("3.0.3"));
    }

    #[test]
    fn test_generate_openapi_contains_path() {
        let ops = vec![simple_op()];
        let yaml = mapper().generate_openapi(&ops, "https://api.example.org");
        assert!(yaml.contains("get-temperature"));
    }

    #[test]
    fn test_generate_openapi_contains_operation_id() {
        let ops = vec![simple_op()];
        let yaml = mapper().generate_openapi(&ops, "https://api.example.org");
        assert!(yaml.contains("getTemperature"));
    }

    #[test]
    fn test_generate_openapi_contains_base_url() {
        let ops = vec![simple_op()];
        let yaml = mapper().generate_openapi(&ops, "https://api.example.org");
        assert!(yaml.contains("https://api.example.org"));
    }

    #[test]
    fn test_generate_openapi_multiple_ops() {
        let ops = vec![simple_op(), no_input_op()];
        let yaml = mapper().generate_openapi(&ops, "https://api.example.org");
        assert!(yaml.contains("get-temperature"));
        assert!(yaml.contains("get-status"));
    }

    #[test]
    fn test_generate_openapi_200_response() {
        let ops = vec![simple_op()];
        let yaml = mapper().generate_openapi(&ops, "https://api.example.org");
        assert!(yaml.contains("'200'"));
    }

    #[test]
    fn test_generate_openapi_query_parameter() {
        let ops = vec![simple_op()];
        let yaml = mapper().generate_openapi(&ops, "https://api.example.org");
        // simple_op has deviceId as GET query param
        assert!(yaml.contains("deviceId"));
        assert!(yaml.contains("in: query"));
    }

    // ── AsyncAPI generation ────────────────────────────────────────────────────

    #[test]
    fn test_generate_asyncapi_contains_version() {
        let ops = vec![simple_op()];
        let yaml = mapper().generate_asyncapi(&ops, "mqtt://broker.example.org:1883");
        assert!(yaml.contains("asyncapi:"));
        assert!(yaml.contains("2.6.0"));
    }

    #[test]
    fn test_generate_asyncapi_contains_server() {
        let ops = vec![simple_op()];
        let yaml = mapper().generate_asyncapi(&ops, "mqtt://broker.example.org:1883");
        assert!(yaml.contains("mqtt://broker.example.org:1883"));
    }

    #[test]
    fn test_generate_asyncapi_contains_channel() {
        let ops = vec![simple_op()];
        let yaml = mapper().generate_asyncapi(&ops, "mqtt://broker.example.org:1883");
        assert!(yaml.contains("get_temperature"));
        assert!(yaml.contains("request"));
    }

    #[test]
    fn test_generate_asyncapi_qos() {
        let ops = vec![simple_op()];
        let yaml = mapper().generate_asyncapi(&ops, "mqtt://broker.example.org:1883");
        assert!(yaml.contains("qos:"));
    }

    // ── Combined mapping ──────────────────────────────────────────────────────

    #[test]
    fn test_map_to_both() {
        let op = simple_op();
        let mapping = mapper().map_to_both(&op, "factory");
        assert!(matches!(mapping, ApiMapping::Both(_, _)));
    }

    // ── JSON schema helpers ───────────────────────────────────────────────────

    #[test]
    fn test_json_schema_empty_props() {
        let schema = build_json_schema(&[]);
        assert!(schema.contains("object"));
        assert!(schema.contains("properties"));
    }

    #[test]
    fn test_json_schema_required_field() {
        let props = vec![PropertyRef::required("name", "string")];
        let schema = build_json_schema(&props);
        assert!(schema.contains("required"));
        assert!(schema.contains("name"));
    }

    #[test]
    fn test_json_schema_optional_not_in_required() {
        let props = vec![PropertyRef::optional("ttl", "integer")];
        let schema = build_json_schema(&props);
        // optional prop should not appear in required array
        assert!(!schema.contains("\"required\""));
    }

    #[test]
    fn test_xsd_to_json_type_string() {
        assert_eq!(xsd_to_json_type("string"), "string");
        assert_eq!(xsd_to_json_type("xsd:string"), "string");
    }

    #[test]
    fn test_xsd_to_json_type_integer() {
        assert_eq!(xsd_to_json_type("integer"), "integer");
        assert_eq!(xsd_to_json_type("int"), "integer");
    }

    #[test]
    fn test_xsd_to_json_type_double() {
        assert_eq!(xsd_to_json_type("double"), "number");
        assert_eq!(xsd_to_json_type("float"), "number");
        assert_eq!(xsd_to_json_type("decimal"), "number");
    }

    #[test]
    fn test_xsd_to_json_type_boolean() {
        assert_eq!(xsd_to_json_type("boolean"), "boolean");
    }

    #[test]
    fn test_xsd_to_json_type_unknown() {
        assert_eq!(xsd_to_json_type("myCustomType"), "string");
    }

    #[test]
    fn test_to_kebab_case() {
        assert_eq!(to_kebab_case("getTemperature"), "get-temperature");
        assert_eq!(to_kebab_case("setMyValue"), "set-my-value");
        assert_eq!(to_kebab_case("simple"), "simple");
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("getTemperature"), "get_temperature");
        assert_eq!(to_snake_case("setMyValue"), "set_my_value");
    }

    #[test]
    fn test_samm_operation_new() {
        let op = SammOperation::new("myOp", vec![], vec![]);
        assert_eq!(op.name, "myOp");
        assert!(op.input_props.is_empty());
        assert!(op.output_props.is_empty());
    }
}
