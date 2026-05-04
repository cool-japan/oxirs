//! SAMM to OpenAPI Specification Generator
//!
//! Generates [OpenAPI 3.0.3](https://spec.openapis.org/oas/v3.0.3) and
//! [OpenAPI 3.1.0](https://spec.openapis.org/oas/v3.1.0) specification
//! documents from SAMM Aspect Models.  OpenAPI 3.1 aligns with
//! [JSON Schema 2020-12](https://json-schema.org/draft/2020-12/schema).
//!
//! The resulting spec exposes a REST endpoint that returns the Aspect payload,
//! and documents all types as reusable schema components.
//!
//! # Example – OpenAPI 3.0
//!
//! ```rust,no_run
//! use oxirs_samm::parser::parse_aspect_model;
//! use oxirs_samm::codegen::OpenApiGenerator;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let aspect = parse_aspect_model("Movement.ttl").await?;
//! let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
//! let spec = gen.generate(&aspect)?;
//! println!("{}", serde_json::to_string_pretty(&spec)?);
//! # Ok(())
//! # }
//! ```
//!
//! # Example – OpenAPI 3.1 (JSON Schema 2020-12)
//!
//! ```rust,no_run
//! use oxirs_samm::parser::parse_aspect_model;
//! use oxirs_samm::codegen::{OpenApiGenerator, OpenApiOptions, OpenApiVersion};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let aspect = parse_aspect_model("Movement.ttl").await?;
//! let options = OpenApiOptions {
//!     version: OpenApiVersion::V31,
//!     ..OpenApiOptions::default()
//! };
//! let gen = OpenApiGenerator::with_options(options);
//! let spec = gen.generate(&aspect)?;
//! println!("{}", serde_json::to_string_pretty(&spec)?);
//! # Ok(())
//! # }
//! ```

use serde_json::{json, Map, Value};

use crate::error::{Result, SammError};
use crate::metamodel::{
    Aspect, Characteristic, CharacteristicKind, Entity, ModelElement, Property,
};

/// JSON Schema 2020-12 meta-schema URI used as `jsonSchemaDialect` in OpenAPI 3.1 documents.
const JSON_SCHEMA_2020_12_DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";

// ------------------------------------------------------------------ //
//  OpenAPI version selector                                            //
// ------------------------------------------------------------------ //

/// Target OpenAPI specification version for [`OpenApiGenerator`].
///
/// Choosing [`OpenApiVersion::V31`] produces a document that conforms to
/// OpenAPI 3.1.0 and uses JSON Schema 2020-12 as its schema dialect.
/// The key differences from 3.0 are:
///
/// - Version string `"3.1.0"` and a `jsonSchemaDialect` field.
/// - Optional properties use `type: ["T", "null"]` instead of `nullable: true`.
/// - Single-value enumerations use `const` instead of `enum: [value]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpenApiVersion {
    /// OpenAPI 3.0.3 (default, maximum compatibility).
    #[default]
    V30,
    /// OpenAPI 3.1.0, aligned with JSON Schema 2020-12.
    V31,
}

// ------------------------------------------------------------------ //
//  HTTP method vocabulary                                              //
// ------------------------------------------------------------------ //

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
    fn as_str(self) -> &'static str {
        match self {
            HttpMethod::Get => "get",
            HttpMethod::Post => "post",
            HttpMethod::Put => "put",
            HttpMethod::Patch => "patch",
            HttpMethod::Delete => "delete",
        }
    }
}

// ------------------------------------------------------------------ //
//  Configuration                                                       //
// ------------------------------------------------------------------ //

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
        }
    }
}

// ------------------------------------------------------------------ //
//  Generator                                                           //
// ------------------------------------------------------------------ //

/// Generates OpenAPI 3.0.3 specifications from SAMM Aspect Models.
#[derive(Debug, Clone)]
pub struct OpenApiGenerator {
    options: OpenApiOptions,
}

impl OpenApiGenerator {
    /// Create a generator with the given API version and base path.
    pub fn new(version: impl Into<String>, base_path: impl Into<String>) -> Self {
        let options = OpenApiOptions {
            api_version: version.into(),
            base_path: base_path.into(),
            ..OpenApiOptions::default()
        };
        Self { options }
    }

    /// Create a generator from fully-specified [`OpenApiOptions`].
    pub fn with_options(options: OpenApiOptions) -> Self {
        Self { options }
    }

    /// Enable a POST endpoint in the generated spec.
    pub fn with_post(mut self) -> Self {
        self.options.include_post = true;
        self
    }

    /// Enable a PUT endpoint in the generated spec.
    pub fn with_put(mut self) -> Self {
        self.options.include_put = true;
        self
    }

    /// Enable a DELETE endpoint in the generated spec.
    pub fn with_delete(mut self) -> Self {
        self.options.include_delete = true;
        self
    }

    // ---------------------------------------------------------------- //
    //  Public API                                                        //
    // ---------------------------------------------------------------- //

    /// Generate an OpenAPI specification `Value` for the given `aspect`.
    ///
    /// The output version (3.0 or 3.1) is controlled by [`OpenApiOptions::version`].
    pub fn generate(&self, aspect: &Aspect) -> Result<Value> {
        match self.options.version {
            OpenApiVersion::V30 => self.generate_v30(aspect),
            OpenApiVersion::V31 => self.generate_v31(aspect),
        }
    }

    /// Generate an OpenAPI 3.0.3 specification `Value` for the given `aspect`.
    pub fn generate_v30(&self, aspect: &Aspect) -> Result<Value> {
        let aspect_name = aspect.name();
        let path = format!(
            "{}/{}",
            self.options.base_path.trim_end_matches('/'),
            to_kebab_case(&aspect_name)
        );

        let mut spec = Map::new();
        spec.insert("openapi".to_string(), Value::String("3.0.3".to_string()));
        spec.insert("info".to_string(), self.build_info(aspect));
        spec.insert(
            "paths".to_string(),
            json!({ path: self.build_path_item(aspect)? }),
        );
        spec.insert("components".to_string(), self.build_components(aspect)?);

        Ok(Value::Object(spec))
    }

    /// Generate an OpenAPI 3.1.0 specification `Value` for the given `aspect`.
    ///
    /// The generated document:
    /// - Sets `openapi: "3.1.0"` and `jsonSchemaDialect` to the JSON Schema
    ///   2020-12 meta-schema URI.
    /// - Represents optional properties as `type: ["T", "null"]` (or wraps
    ///   non-primitive schemas in `oneOf: [..., {type: "null"}]`).
    /// - Uses `const` for single-value enumerations.
    pub fn generate_v31(&self, aspect: &Aspect) -> Result<Value> {
        let aspect_name = aspect.name();
        let path = format!(
            "{}/{}",
            self.options.base_path.trim_end_matches('/'),
            to_kebab_case(&aspect_name)
        );

        let mut spec = Map::new();
        spec.insert("openapi".to_string(), Value::String("3.1.0".to_string()));
        spec.insert(
            "jsonSchemaDialect".to_string(),
            Value::String(JSON_SCHEMA_2020_12_DIALECT.to_string()),
        );
        spec.insert("info".to_string(), self.build_info(aspect));
        spec.insert(
            "paths".to_string(),
            json!({ path: self.build_path_item(aspect)? }),
        );
        spec.insert("components".to_string(), self.build_components_v31(aspect)?);

        Ok(Value::Object(spec))
    }

    // ---------------------------------------------------------------- //
    //  Private builders                                                  //
    // ---------------------------------------------------------------- //

    fn build_info(&self, aspect: &Aspect) -> Value {
        let aspect_name = aspect.name();
        let title = aspect
            .metadata()
            .get_preferred_name(&self.options.language)
            .map(|s| s.to_string())
            .unwrap_or_else(|| aspect_name.clone());

        let mut info = json!({
            "title": title,
            "version": self.options.api_version,
        });

        if let Some(desc) = aspect.metadata().get_description(&self.options.language) {
            if let Some(obj) = info.as_object_mut() {
                obj.insert("description".to_string(), Value::String(desc.to_string()));
            }
        }

        info
    }

    /// Build the path item object that lists all operations for the aspect path.
    fn build_path_item(&self, aspect: &Aspect) -> Result<Value> {
        let mut item = Map::new();

        let aspect_name = aspect.name();

        if self.options.include_get {
            item.insert(
                HttpMethod::Get.as_str().to_string(),
                self.build_operation(aspect, HttpMethod::Get, &aspect_name)?,
            );
        }
        if self.options.include_post {
            item.insert(
                HttpMethod::Post.as_str().to_string(),
                self.build_operation(aspect, HttpMethod::Post, &aspect_name)?,
            );
        }
        if self.options.include_put {
            item.insert(
                HttpMethod::Put.as_str().to_string(),
                self.build_operation(aspect, HttpMethod::Put, &aspect_name)?,
            );
        }
        if self.options.include_delete {
            item.insert(
                HttpMethod::Delete.as_str().to_string(),
                self.build_operation(aspect, HttpMethod::Delete, &aspect_name)?,
            );
        }

        Ok(Value::Object(item))
    }

    fn build_operation(
        &self,
        aspect: &Aspect,
        method: HttpMethod,
        schema_ref: &str,
    ) -> Result<Value> {
        let (summary, description) = match method {
            HttpMethod::Get => (
                format!("Get {} data", schema_ref),
                format!("Retrieve the current state of the {} aspect", schema_ref),
            ),
            HttpMethod::Post => (
                format!("Create {} instance", schema_ref),
                format!("Create a new {} aspect instance", schema_ref),
            ),
            HttpMethod::Put => (
                format!("Update {} instance", schema_ref),
                format!("Replace the {} aspect instance", schema_ref),
            ),
            HttpMethod::Patch => (
                format!("Patch {} instance", schema_ref),
                format!("Partially update the {} aspect instance", schema_ref),
            ),
            HttpMethod::Delete => (
                format!("Delete {} instance", schema_ref),
                format!("Delete the {} aspect instance", schema_ref),
            ),
        };

        let mut op = Map::new();
        op.insert("summary".to_string(), Value::String(summary));
        op.insert("description".to_string(), Value::String(description));
        op.insert(
            "operationId".to_string(),
            Value::String(format!("{}_{}", method.as_str(), to_camel_case(schema_ref))),
        );
        op.insert(
            "tags".to_string(),
            Value::Array(vec![Value::String(schema_ref.to_string())]),
        );

        // Request body for mutating methods
        if matches!(
            method,
            HttpMethod::Post | HttpMethod::Put | HttpMethod::Patch
        ) {
            op.insert(
                "requestBody".to_string(),
                self.build_request_body(schema_ref),
            );
        }

        op.insert(
            "responses".to_string(),
            self.build_responses(aspect, method, schema_ref),
        );

        Ok(Value::Object(op))
    }

    fn build_request_body(&self, schema_ref: &str) -> Value {
        json!({
            "required": true,
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": format!("#/components/schemas/{}", schema_ref)
                    }
                }
            }
        })
    }

    fn build_responses(&self, _aspect: &Aspect, method: HttpMethod, schema_ref: &str) -> Value {
        let success_code = match method {
            HttpMethod::Post => "201",
            HttpMethod::Delete => "204",
            _ => "200",
        };

        let mut responses = Map::new();

        if method == HttpMethod::Delete {
            responses.insert(
                success_code.to_string(),
                json!({ "description": "Successfully deleted" }),
            );
        } else {
            responses.insert(
                success_code.to_string(),
                json!({
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": format!("#/components/schemas/{}", schema_ref)
                            }
                        }
                    }
                }),
            );
        }

        // Standard error responses
        responses.insert(
            "400".to_string(),
            json!({ "description": "Bad request – invalid input" }),
        );
        responses.insert("404".to_string(), json!({ "description": "Not found" }));
        responses.insert(
            "500".to_string(),
            json!({ "description": "Internal server error" }),
        );

        Value::Object(responses)
    }

    /// Build the `components` section for OpenAPI 3.0.
    fn build_components(&self, aspect: &Aspect) -> Result<Value> {
        let schemas = self.build_schemas(aspect)?;
        Ok(json!({ "schemas": schemas }))
    }

    /// Build the `components` section for OpenAPI 3.1 (JSON Schema 2020-12).
    fn build_components_v31(&self, aspect: &Aspect) -> Result<Value> {
        let schemas = self.build_schemas_v31(aspect)?;
        Ok(json!({ "schemas": schemas }))
    }

    /// Build the `components/schemas` mapping.
    pub fn build_schemas(&self, aspect: &Aspect) -> Result<Value> {
        let mut schemas = Map::new();

        // Primary aspect schema
        let aspect_schema = self.build_aspect_schema(aspect)?;
        schemas.insert(aspect.name(), aspect_schema);

        // Inline entity schemas (SingleEntity references)
        for prop in aspect.properties() {
            if let Some(char) = &prop.characteristic {
                if let CharacteristicKind::SingleEntity { entity_type } = char.kind() {
                    let entity_name = entity_type
                        .split('#')
                        .next_back()
                        .unwrap_or(entity_type.as_str())
                        .to_string();
                    if !schemas.contains_key(&entity_name) {
                        schemas.insert(entity_name, json!({ "type": "object" }));
                    }
                }
            }
        }

        Ok(Value::Object(schemas))
    }

    /// Build the `components/schemas` mapping for OpenAPI 3.1 documents.
    pub fn build_schemas_v31(&self, aspect: &Aspect) -> Result<Value> {
        let mut schemas = Map::new();

        let aspect_schema = self.build_aspect_schema_v31(aspect)?;
        schemas.insert(aspect.name(), aspect_schema);

        // Inline entity schemas (SingleEntity references)
        for prop in aspect.properties() {
            if let Some(char) = &prop.characteristic {
                if let CharacteristicKind::SingleEntity { entity_type } = char.kind() {
                    let entity_name = entity_type
                        .split('#')
                        .next_back()
                        .unwrap_or(entity_type.as_str())
                        .to_string();
                    if !schemas.contains_key(&entity_name) {
                        schemas.insert(entity_name, json!({ "type": "object" }));
                    }
                }
            }
        }

        Ok(Value::Object(schemas))
    }

    fn build_aspect_schema_v31(&self, aspect: &Aspect) -> Result<Value> {
        let mut schema = Map::new();

        schema.insert("type".to_string(), Value::String("object".to_string()));

        if let Some(desc) = aspect.metadata().get_description(&self.options.language) {
            schema.insert("description".to_string(), Value::String(desc.to_string()));
        }

        let (properties_map, required) = self.build_properties_schema_v31(aspect.properties())?;
        schema.insert("properties".to_string(), Value::Object(properties_map));
        if !required.is_empty() {
            schema.insert(
                "required".to_string(),
                Value::Array(required.into_iter().map(Value::String).collect()),
            );
        }

        Ok(Value::Object(schema))
    }

    fn build_properties_schema_v31(
        &self,
        props: &[Property],
    ) -> Result<(Map<String, Value>, Vec<String>)> {
        let mut map = Map::new();
        let mut required = Vec::new();

        for prop in props {
            let name = prop.payload_name.clone().unwrap_or_else(|| prop.name());
            let prop_schema = self.property_schema_v31(prop)?;
            map.insert(name.clone(), prop_schema);
            if !prop.optional {
                required.push(name);
            }
        }

        Ok((map, required))
    }

    /// Build a property schema following OpenAPI 3.1 / JSON Schema 2020-12 rules.
    ///
    /// For optional properties the schema is wrapped to include `"null"` in the
    /// type array (for primitives) or via `oneOf: [..., {type: "null"}]` for
    /// reference/composite schemas that do not have a simple `type` field.
    fn property_schema_v31(&self, prop: &Property) -> Result<Value> {
        let mut s = Map::new();

        if let Some(desc) = prop.metadata().get_description(&self.options.language) {
            s.insert("description".to_string(), Value::String(desc.to_string()));
        }

        if !prop.example_values.is_empty() {
            s.insert(
                "example".to_string(),
                Value::String(prop.example_values.first().cloned().unwrap_or_default()),
            );
        }

        let type_schema = if let Some(char) = &prop.characteristic {
            self.characteristic_schema_v31(char)?
        } else {
            json!({ "type": "string" })
        };

        if prop.optional {
            // Merge the nullability annotation into the schema.
            let nullable_schema = make_nullable_v31(type_schema);
            if let Value::Object(nullable_map) = nullable_schema {
                for (k, v) in nullable_map {
                    s.insert(k, v);
                }
            }
        } else if let Value::Object(type_map) = type_schema {
            for (k, v) in type_map {
                s.insert(k, v);
            }
        }

        Ok(Value::Object(s))
    }

    fn characteristic_schema_v31(&self, char: &Characteristic) -> Result<Value> {
        match char.kind() {
            CharacteristicKind::Trait => {
                let json_type = char
                    .data_type
                    .as_deref()
                    .map(xsd_to_openapi_type)
                    .unwrap_or("string");
                Ok(json!({ "type": json_type }))
            }
            CharacteristicKind::Measurement { unit }
            | CharacteristicKind::Quantifiable { unit } => {
                let json_type = char
                    .data_type
                    .as_deref()
                    .map(xsd_to_openapi_type)
                    .unwrap_or("number");
                Ok(json!({
                    "type": json_type,
                    "description": format!("Value expressed in {}", unit)
                }))
            }
            CharacteristicKind::Duration { unit } => Ok(json!({
                "type": "number",
                "description": format!("Duration in {}", unit)
            })),
            CharacteristicKind::Enumeration { values } => {
                let data_type = char
                    .data_type
                    .as_deref()
                    .map(xsd_to_openapi_type)
                    .unwrap_or("string");
                // In 3.1 / JSON Schema 2020-12: single-value enum → `const`
                if values.len() == 1 {
                    Ok(json!({
                        "type": data_type,
                        "const": values[0]
                    }))
                } else {
                    Ok(json!({
                        "type": data_type,
                        "enum": values
                    }))
                }
            }
            CharacteristicKind::State {
                values,
                default_value,
            } => {
                let data_type = char
                    .data_type
                    .as_deref()
                    .map(xsd_to_openapi_type)
                    .unwrap_or("string");
                let mut s = json!({
                    "type": data_type,
                    "enum": values
                });
                if let (Some(obj), Some(default)) = (s.as_object_mut(), default_value.as_deref()) {
                    obj.insert("default".to_string(), Value::String(default.to_string()));
                }
                Ok(s)
            }
            CharacteristicKind::Collection {
                element_characteristic,
            }
            | CharacteristicKind::List {
                element_characteristic,
            }
            | CharacteristicKind::TimeSeries {
                element_characteristic,
            } => {
                let items = if let Some(inner) = element_characteristic {
                    self.characteristic_schema_v31(inner)?
                } else {
                    json!({})
                };
                Ok(json!({ "type": "array", "items": items }))
            }
            CharacteristicKind::Set {
                element_characteristic,
            }
            | CharacteristicKind::SortedSet {
                element_characteristic,
            } => {
                let items = if let Some(inner) = element_characteristic {
                    self.characteristic_schema_v31(inner)?
                } else {
                    json!({})
                };
                Ok(json!({ "type": "array", "items": items, "uniqueItems": true }))
            }
            CharacteristicKind::Code => {
                let format: Option<&'static str> =
                    char.data_type.as_deref().and_then(xsd_to_openapi_format);
                let mut s = json!({ "type": "string" });
                if let (Some(obj), Some(fmt)) = (s.as_object_mut(), format) {
                    obj.insert("format".to_string(), Value::String(fmt.to_string()));
                }
                Ok(s)
            }
            CharacteristicKind::Either { left, right } => {
                let left_schema = self.characteristic_schema_v31(left)?;
                let right_schema = self.characteristic_schema_v31(right)?;
                Ok(json!({ "oneOf": [left_schema, right_schema] }))
            }
            CharacteristicKind::SingleEntity { entity_type } => {
                let ref_name = entity_type
                    .split('#')
                    .next_back()
                    .unwrap_or(entity_type.as_str());
                Ok(json!({ "$ref": format!("#/components/schemas/{}", ref_name) }))
            }
            CharacteristicKind::StructuredValue { .. } => {
                Ok(json!({ "type": "string", "format": "structured-value" }))
            }
        }
    }

    fn build_aspect_schema(&self, aspect: &Aspect) -> Result<Value> {
        let mut schema = Map::new();

        schema.insert("type".to_string(), Value::String("object".to_string()));

        if let Some(desc) = aspect.metadata().get_description(&self.options.language) {
            schema.insert("description".to_string(), Value::String(desc.to_string()));
        }

        let (properties_map, required) = self.build_properties_schema(aspect.properties())?;
        schema.insert("properties".to_string(), Value::Object(properties_map));
        if !required.is_empty() {
            schema.insert(
                "required".to_string(),
                Value::Array(required.into_iter().map(Value::String).collect()),
            );
        }

        Ok(Value::Object(schema))
    }

    fn build_properties_schema(
        &self,
        props: &[Property],
    ) -> Result<(Map<String, Value>, Vec<String>)> {
        let mut map = Map::new();
        let mut required = Vec::new();

        for prop in props {
            let name = prop.payload_name.clone().unwrap_or_else(|| prop.name());
            let prop_schema = self.property_schema(prop)?;
            map.insert(name.clone(), prop_schema);
            if !prop.optional {
                required.push(name);
            }
        }

        Ok((map, required))
    }

    fn property_schema(&self, prop: &Property) -> Result<Value> {
        let mut s = Map::new();

        if let Some(desc) = prop.metadata().get_description(&self.options.language) {
            s.insert("description".to_string(), Value::String(desc.to_string()));
        }

        if !prop.example_values.is_empty() {
            s.insert(
                "example".to_string(),
                Value::String(prop.example_values.first().cloned().unwrap_or_default()),
            );
        }

        if let Some(char) = &prop.characteristic {
            let type_schema = self.characteristic_schema(char)?;
            if let Value::Object(type_map) = type_schema {
                for (k, v) in type_map {
                    s.insert(k, v);
                }
            }
        } else {
            s.insert("type".to_string(), Value::String("string".to_string()));
        }

        Ok(Value::Object(s))
    }

    fn characteristic_schema(&self, char: &Characteristic) -> Result<Value> {
        match char.kind() {
            CharacteristicKind::Trait => {
                let json_type = char
                    .data_type
                    .as_deref()
                    .map(|dt| xsd_to_openapi_type(dt))
                    .unwrap_or("string");
                Ok(json!({ "type": json_type }))
            }
            CharacteristicKind::Measurement { unit }
            | CharacteristicKind::Quantifiable { unit } => {
                let json_type = char
                    .data_type
                    .as_deref()
                    .map(|dt| xsd_to_openapi_type(dt))
                    .unwrap_or("number");
                Ok(json!({
                    "type": json_type,
                    "description": format!("Value expressed in {}", unit)
                }))
            }
            CharacteristicKind::Duration { unit } => Ok(json!({
                "type": "number",
                "description": format!("Duration in {}", unit)
            })),
            CharacteristicKind::Enumeration { values } => {
                let data_type = char
                    .data_type
                    .as_deref()
                    .map(|dt| xsd_to_openapi_type(dt))
                    .unwrap_or("string");
                Ok(json!({
                    "type": data_type,
                    "enum": values
                }))
            }
            CharacteristicKind::State {
                values,
                default_value,
            } => {
                let data_type = char
                    .data_type
                    .as_deref()
                    .map(|dt| xsd_to_openapi_type(dt))
                    .unwrap_or("string");
                let mut s = json!({
                    "type": data_type,
                    "enum": values
                });
                if let (Some(obj), Some(default)) = (s.as_object_mut(), default_value.as_deref()) {
                    obj.insert("default".to_string(), Value::String(default.to_string()));
                }
                Ok(s)
            }
            CharacteristicKind::Collection {
                element_characteristic,
            }
            | CharacteristicKind::List {
                element_characteristic,
            }
            | CharacteristicKind::TimeSeries {
                element_characteristic,
            } => {
                let items = if let Some(inner) = element_characteristic {
                    self.characteristic_schema(inner)?
                } else {
                    json!({})
                };
                Ok(json!({ "type": "array", "items": items }))
            }
            CharacteristicKind::Set {
                element_characteristic,
            }
            | CharacteristicKind::SortedSet {
                element_characteristic,
            } => {
                let items = if let Some(inner) = element_characteristic {
                    self.characteristic_schema(inner)?
                } else {
                    json!({})
                };
                Ok(json!({ "type": "array", "items": items, "uniqueItems": true }))
            }
            CharacteristicKind::Code => {
                let format: Option<&'static str> =
                    char.data_type.as_deref().and_then(xsd_to_openapi_format);
                let mut s = json!({ "type": "string" });
                if let (Some(obj), Some(fmt)) = (s.as_object_mut(), format) {
                    obj.insert("format".to_string(), Value::String(fmt.to_string()));
                }
                Ok(s)
            }
            CharacteristicKind::Either { left, right } => {
                let left_schema = self.characteristic_schema(left)?;
                let right_schema = self.characteristic_schema(right)?;
                Ok(json!({ "oneOf": [left_schema, right_schema] }))
            }
            CharacteristicKind::SingleEntity { entity_type } => {
                let ref_name = entity_type
                    .split('#')
                    .next_back()
                    .unwrap_or(entity_type.as_str());
                Ok(json!({ "$ref": format!("#/components/schemas/{}", ref_name) }))
            }
            CharacteristicKind::StructuredValue { .. } => {
                Ok(json!({ "type": "string", "format": "structured-value" }))
            }
        }
    }
}

// ------------------------------------------------------------------ //
//  Helper functions                                                    //
// ------------------------------------------------------------------ //

/// Wrap a type schema so that `null` is an accepted value (OpenAPI 3.1 style).
///
/// For schemas with a single `type` field the type is widened to an array that
/// includes `"null"`.  For composite schemas (`$ref`, `oneOf`, `anyOf`, …) an
/// outer `oneOf: [schema, {type: "null"}]` wrapper is used instead.
fn make_nullable_v31(schema: Value) -> Value {
    // Determine whether the schema has a plain `type` string we can widen.
    let has_simple_type = schema
        .as_object()
        .and_then(|m| m.get("type"))
        .map(|t| t.is_string())
        .unwrap_or(false);

    if has_simple_type {
        // Widen: {"type": "string", ...} → {"type": ["string", "null"], ...}
        let mut out = schema.as_object().cloned().unwrap_or_default();
        if let Some(t) = out.remove("type") {
            out.insert("type".to_string(), json!([t, "null"]));
        }
        Value::Object(out)
    } else {
        // Composite or reference schema: wrap with oneOf + {type: "null"}
        json!({ "oneOf": [schema, { "type": "null" }] })
    }
}

/// Map XSD/SAMM data type string to an OpenAPI type string.
fn xsd_to_openapi_type(dt: &str) -> &'static str {
    if dt.ends_with("boolean") {
        return "boolean";
    }
    if dt.ends_with("int")
        || dt.ends_with("integer")
        || dt.ends_with("long")
        || dt.ends_with("short")
        || dt.ends_with("byte")
        || dt.ends_with("unsignedInt")
        || dt.ends_with("unsignedLong")
        || dt.ends_with("unsignedShort")
        || dt.ends_with("positiveInteger")
        || dt.ends_with("nonNegativeInteger")
    {
        return "integer";
    }
    if dt.ends_with("decimal") || dt.ends_with("float") || dt.ends_with("double") {
        return "number";
    }
    "string"
}

/// Map XSD/SAMM data type to an OpenAPI `format` hint (returns `None` for string).
fn xsd_to_openapi_format(dt: &str) -> Option<&'static str> {
    if dt.ends_with("float") {
        return Some("float");
    }
    if dt.ends_with("double") {
        return Some("double");
    }
    if dt.ends_with("int") || dt.ends_with("integer") {
        return Some("int32");
    }
    if dt.ends_with("long") {
        return Some("int64");
    }
    if dt.ends_with("dateTime") || dt.ends_with("dateTimeStamp") {
        return Some("date-time");
    }
    if dt.ends_with("date") {
        return Some("date");
    }
    if dt.ends_with("base64Binary") {
        return Some("byte");
    }
    if dt.ends_with("hexBinary") {
        return Some("binary");
    }
    None
}

/// Convert PascalCase / camelCase to kebab-case for path segments.
fn to_kebab_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('-');
        }
        result.push(ch.to_ascii_lowercase());
    }
    result
}

/// Convert PascalCase or snake_case to camelCase.
fn to_camel_case(s: &str) -> String {
    if s.is_empty() {
        return s.to_string();
    }
    let mut chars = s.chars();
    let first = chars.next().expect("non-empty string").to_ascii_lowercase();
    format!("{}{}", first, chars.as_str())
}

// ------------------------------------------------------------------ //
//  Tests                                                               //
// ------------------------------------------------------------------ //

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};

    fn movement_aspect() -> Aspect {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#Movement".to_string());
        aspect
            .metadata
            .add_preferred_name("en".to_string(), "Movement".to_string());
        aspect
            .metadata
            .add_description("en".to_string(), "Describes movement telemetry".to_string());

        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#SpeedChar".to_string(),
            CharacteristicKind::Measurement {
                unit: "unit:kilometrePerHour".to_string(),
            },
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string());

        let prop =
            Property::new("urn:samm:org.example:1.0.0#speed".to_string()).with_characteristic(char);

        aspect.add_property(prop);
        aspect
    }

    #[test]
    fn test_generate_openapi_version() {
        let aspect = movement_aspect();
        let gen = OpenApiGenerator::new("1.2.3", "/api/v1/aspects");
        let spec = gen.generate(&aspect).expect("generation should succeed");
        assert_eq!(spec["openapi"], "3.0.3");
        assert_eq!(spec["info"]["version"], "1.2.3");
    }

    #[test]
    fn test_path_is_present() {
        let aspect = movement_aspect();
        let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
        let spec = gen.generate(&aspect).expect("generation should succeed");
        let paths = spec["paths"].as_object().expect("paths should be object");
        assert!(!paths.is_empty(), "paths should not be empty");
        let path_key = paths.keys().next().expect("at least one path");
        assert!(
            path_key.contains("movement"),
            "path should contain aspect name"
        );
    }

    #[test]
    fn test_get_operation_present_by_default() {
        let aspect = movement_aspect();
        let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
        let spec = gen.generate(&aspect).expect("generation should succeed");
        let paths = spec["paths"].as_object().expect("paths");
        let path_item = paths.values().next().expect("path item");
        assert!(
            path_item.get("get").is_some(),
            "GET operation should be present"
        );
    }

    #[test]
    fn test_post_not_included_by_default() {
        let aspect = movement_aspect();
        let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
        let spec = gen.generate(&aspect).expect("generation should succeed");
        let paths = spec["paths"].as_object().expect("paths");
        let path_item = paths.values().next().expect("path item");
        assert!(
            path_item.get("post").is_none(),
            "POST should not be present by default"
        );
    }

    #[test]
    fn test_post_included_when_enabled() {
        let aspect = movement_aspect();
        let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects").with_post();
        let spec = gen.generate(&aspect).expect("generation should succeed");
        let paths = spec["paths"].as_object().expect("paths");
        let path_item = paths.values().next().expect("path item");
        assert!(path_item.get("post").is_some(), "POST should be present");
    }

    #[test]
    fn test_components_schemas_contains_aspect() {
        let aspect = movement_aspect();
        let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
        let spec = gen.generate(&aspect).expect("generation should succeed");
        let schemas = spec["components"]["schemas"].as_object().expect("schemas");
        assert!(
            schemas.contains_key("Movement"),
            "schemas should include Movement"
        );
    }

    #[test]
    fn test_aspect_schema_has_properties() {
        let aspect = movement_aspect();
        let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
        let schemas = gen.build_schemas(&aspect).expect("build_schemas");
        assert!(schemas["Movement"]["properties"]["speed"].is_object());
    }

    #[test]
    fn test_aspect_schema_required_field() {
        let aspect = movement_aspect();
        let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
        let schemas = gen.build_schemas(&aspect).expect("build_schemas");
        let required = schemas["Movement"]["required"]
            .as_array()
            .expect("required should be array");
        assert!(required.iter().any(|v| v == "speed"));
    }

    #[test]
    fn test_measurement_type_is_number() {
        let aspect = movement_aspect();
        let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
        let schemas = gen.build_schemas(&aspect).expect("build_schemas");
        assert_eq!(schemas["Movement"]["properties"]["speed"]["type"], "number");
    }

    #[test]
    fn test_info_description_present() {
        let aspect = movement_aspect();
        let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
        let spec = gen.generate(&aspect).expect("generation should succeed");
        assert!(spec["info"]["description"].is_string());
    }

    #[test]
    fn test_response_contains_success_code() {
        let aspect = movement_aspect();
        let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
        let spec = gen.generate(&aspect).expect("generation should succeed");
        let paths = spec["paths"].as_object().expect("paths");
        let path_item = paths.values().next().expect("path item");
        let get_op = &path_item["get"];
        assert!(get_op["responses"]["200"].is_object());
    }

    #[test]
    fn test_delete_responds_204() {
        let aspect = movement_aspect();
        let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects").with_delete();
        let spec = gen.generate(&aspect).expect("generation should succeed");
        let paths = spec["paths"].as_object().expect("paths");
        let path_item = paths.values().next().expect("path item");
        let del_op = &path_item["delete"];
        assert!(del_op["responses"]["204"].is_object());
    }

    #[test]
    fn test_enumeration_generates_enum_schema() {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#StatusEnum".to_string(),
            CharacteristicKind::Enumeration {
                values: vec!["Active".to_string(), "Inactive".to_string()],
            },
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
        let prop = Property::new("urn:samm:org.example:1.0.0#status".to_string())
            .with_characteristic(char);
        aspect.add_property(prop);

        let gen = OpenApiGenerator::new("1.0.0", "/api");
        let schemas = gen.build_schemas(&aspect).expect("build_schemas");
        let status = &schemas["TestAspect"]["properties"]["status"];
        assert!(status["enum"].is_array());
    }

    #[test]
    fn test_to_kebab_case() {
        assert_eq!(to_kebab_case("Movement"), "movement");
        assert_eq!(to_kebab_case("MyAspect"), "my-aspect");
        assert_eq!(to_kebab_case("speed"), "speed");
    }

    #[test]
    fn test_xsd_to_openapi_type_mapping() {
        assert_eq!(
            xsd_to_openapi_type("http://www.w3.org/2001/XMLSchema#boolean"),
            "boolean"
        );
        assert_eq!(
            xsd_to_openapi_type("http://www.w3.org/2001/XMLSchema#int"),
            "integer"
        );
        assert_eq!(
            xsd_to_openapi_type("http://www.w3.org/2001/XMLSchema#float"),
            "number"
        );
        assert_eq!(
            xsd_to_openapi_type("http://www.w3.org/2001/XMLSchema#string"),
            "string"
        );
    }

    // ---------------------------------------------------------------- //
    //  OpenAPI 3.1 tests                                                //
    // ---------------------------------------------------------------- //

    fn v31_gen() -> OpenApiGenerator {
        let options = OpenApiOptions {
            version: OpenApiVersion::V31,
            ..OpenApiOptions::default()
        };
        OpenApiGenerator::with_options(options)
    }

    fn aspect_with_optional_property() -> Aspect {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#SensorReading".to_string());
        aspect
            .metadata
            .add_preferred_name("en".to_string(), "SensorReading".to_string());

        // mandatory numeric property
        let char_mandatory = Characteristic::new(
            "urn:samm:org.example:1.0.0#ValueChar".to_string(),
            CharacteristicKind::Measurement {
                unit: "unit:metre".to_string(),
            },
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string());
        let prop_mandatory = Property::new("urn:samm:org.example:1.0.0#value".to_string())
            .with_characteristic(char_mandatory);
        aspect.add_property(prop_mandatory);

        // optional string label
        let char_optional = Characteristic::new(
            "urn:samm:org.example:1.0.0#LabelChar".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
        let prop_optional = Property::new("urn:samm:org.example:1.0.0#label".to_string())
            .with_characteristic(char_optional)
            .as_optional();
        aspect.add_property(prop_optional);

        aspect
    }

    #[test]
    fn test_openapi31_version_string() {
        let aspect = movement_aspect();
        let gen = v31_gen();
        let spec = gen
            .generate(&aspect)
            .expect("3.1 generation should succeed");
        assert_eq!(spec["openapi"], "3.1.0", "should emit openapi 3.1.0");
    }

    #[test]
    fn test_openapi31_json_schema_dialect() {
        let aspect = movement_aspect();
        let gen = v31_gen();
        let spec = gen
            .generate(&aspect)
            .expect("3.1 generation should succeed");
        let dialect = spec["jsonSchemaDialect"].as_str().unwrap_or("");
        assert!(
            dialect.contains("json-schema.org") || dialect.contains("2020-12"),
            "jsonSchemaDialect should reference JSON Schema 2020-12, got: {}",
            dialect
        );
    }

    #[test]
    fn test_openapi31_no_nullable_keyword() {
        let aspect = aspect_with_optional_property();
        let gen = v31_gen();
        let spec = gen
            .generate(&aspect)
            .expect("3.1 generation should succeed");
        let json = serde_json::to_string(&spec).expect("serialize");
        assert!(
            !json.contains("\"nullable\""),
            "3.1 document must not use the `nullable` keyword"
        );
    }

    #[test]
    fn test_openapi31_type_array_for_optional() {
        let aspect = aspect_with_optional_property();
        let gen = v31_gen();
        let schemas = gen
            .build_schemas_v31(&aspect)
            .expect("build_schemas_v31 should succeed");

        // The optional `label` property should have a type array containing "null"
        let label_schema = &schemas["SensorReading"]["properties"]["label"];
        let type_val = &label_schema["type"];
        assert!(
            type_val.is_array(),
            "optional property type should be an array, got: {}",
            type_val
        );
        let empty = vec![];
        let types: Vec<&str> = type_val
            .as_array()
            .unwrap_or(&empty)
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(
            types.contains(&"null"),
            "optional property type array should contain 'null', got: {:?}",
            types
        );
    }

    #[test]
    fn test_openapi31_mandatory_property_not_nullable() {
        let aspect = aspect_with_optional_property();
        let gen = v31_gen();
        let schemas = gen
            .build_schemas_v31(&aspect)
            .expect("build_schemas_v31 should succeed");

        // The mandatory `value` property should NOT have a type array containing "null"
        let value_schema = &schemas["SensorReading"]["properties"]["value"];
        let type_val = &value_schema["type"];
        // It should be a string (not an array) or at minimum not contain null
        if let Some(arr) = type_val.as_array() {
            let types: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
            assert!(
                !types.contains(&"null"),
                "mandatory property should not include 'null' type, got: {:?}",
                types
            );
        }
        // If it's a string type that's also acceptable (non-nullable)
    }

    #[test]
    fn test_openapi31_const_for_single_value_enum() {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#ConstAspect".to_string());
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#FixedVal".to_string(),
            CharacteristicKind::Enumeration {
                values: vec!["FIXED".to_string()],
            },
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
        let prop = Property::new("urn:samm:org.example:1.0.0#fixedField".to_string())
            .with_characteristic(char);
        aspect.add_property(prop);

        let gen = v31_gen();
        let schemas = gen
            .build_schemas_v31(&aspect)
            .expect("build_schemas_v31 should succeed");
        let field = &schemas["ConstAspect"]["properties"]["fixedField"];
        assert!(
            field["const"].is_string(),
            "single-value enum should use `const` in 3.1, got: {}",
            field
        );
        assert!(
            field.get("enum").is_none(),
            "single-value `const` should not also emit `enum`"
        );
    }

    #[test]
    fn test_openapi31_multi_value_enum_uses_enum_keyword() {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#EnumAspect".to_string());
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#StatusEnum".to_string(),
            CharacteristicKind::Enumeration {
                values: vec!["Active".to_string(), "Inactive".to_string()],
            },
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
        let prop = Property::new("urn:samm:org.example:1.0.0#status".to_string())
            .with_characteristic(char);
        aspect.add_property(prop);

        let gen = v31_gen();
        let schemas = gen
            .build_schemas_v31(&aspect)
            .expect("build_schemas_v31 should succeed");
        let status = &schemas["EnumAspect"]["properties"]["status"];
        assert!(
            status["enum"].is_array(),
            "multi-value enum should still use `enum` keyword in 3.1"
        );
    }

    #[test]
    fn test_openapi30_still_works() {
        // Regression: the 3.0 generator must not be broken by 3.1 changes
        let aspect = movement_aspect();
        let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
        let spec = gen
            .generate(&aspect)
            .expect("3.0 generation should succeed");
        assert_eq!(
            spec["openapi"], "3.0.3",
            "default generator should still emit 3.0.3"
        );
        assert!(
            spec.get("jsonSchemaDialect").is_none(),
            "3.0 document should not have jsonSchemaDialect field"
        );
        let schemas = gen.build_schemas(&aspect).expect("build_schemas");
        assert_eq!(
            schemas["Movement"]["properties"]["speed"]["type"], "number",
            "3.0 schema type should be a plain string"
        );
    }

    #[test]
    fn test_openapi31_ref_optional_uses_one_of() {
        // An optional SingleEntity property cannot use type array — must use oneOf + null
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#RefAspect".to_string());
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#EntityRef".to_string(),
            CharacteristicKind::SingleEntity {
                entity_type: "urn:samm:org.example:1.0.0#MyEntity".to_string(),
            },
        );
        let prop = Property::new("urn:samm:org.example:1.0.0#entity".to_string())
            .with_characteristic(char)
            .as_optional();
        aspect.add_property(prop);

        let gen = v31_gen();
        let schemas = gen
            .build_schemas_v31(&aspect)
            .expect("build_schemas_v31 should succeed");
        let entity_prop = &schemas["RefAspect"]["properties"]["entity"];
        // Should use oneOf wrapping since $ref can't have type array
        assert!(
            entity_prop["oneOf"].is_array(),
            "optional $ref property should be wrapped with oneOf, got: {}",
            entity_prop
        );
    }
}
