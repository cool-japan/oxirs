use serde_json::{json, Map, Value};

use crate::error::{Result, SammError};
use crate::metamodel::{Aspect, Characteristic, CharacteristicKind, ModelElement, Property};

use super::openapi_serializer::{
    aspect_has_collection, make_nullable_v31, to_camel_case, to_kebab_case, xsd_to_openapi_format,
    xsd_to_openapi_type,
};
use super::openapi_types::{HttpMethod, OpenApiOptions, OpenApiVersion, PaginationConfig};

const JSON_SCHEMA_2020_12_DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";

/// Generates OpenAPI 3.0.3 / 3.1.0 specifications from SAMM Aspect Models.
#[derive(Debug, Clone)]
pub struct OpenApiGenerator {
    pub(super) options: OpenApiOptions,
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

    /// Attach a [`PaginationConfig`] to the generator.
    pub fn with_pagination(mut self, config: PaginationConfig) -> Self {
        self.options.pagination = Some(config);
        self
    }

    /// Generate an OpenAPI specification `Value` for the given `aspect`.
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

    fn build_responses(&self, aspect: &Aspect, method: HttpMethod, schema_ref: &str) -> Value {
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
            let mut schema_obj = Map::new();
            schema_obj.insert(
                "$ref".to_string(),
                Value::String(format!("#/components/schemas/{}", schema_ref)),
            );

            if method == HttpMethod::Get {
                if let Some(ref pag) = self.options.pagination {
                    if aspect_has_collection(aspect) {
                        schema_obj
                            .insert("x-samm-pagination".to_string(), pag.to_extension_value());
                    }
                }
            }

            responses.insert(
                success_code.to_string(),
                json!({
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": Value::Object(schema_obj)
                        }
                    }
                }),
            );
        }

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

    fn build_components(&self, aspect: &Aspect) -> Result<Value> {
        let schemas = self.build_schemas(aspect)?;
        Ok(json!({ "schemas": schemas }))
    }

    fn build_components_v31(&self, aspect: &Aspect) -> Result<Value> {
        let schemas = self.build_schemas_v31(aspect)?;
        Ok(json!({ "schemas": schemas }))
    }

    /// Build the `components/schemas` mapping for OpenAPI 3.0.
    pub fn build_schemas(&self, aspect: &Aspect) -> Result<Value> {
        let mut schemas = Map::new();

        let aspect_schema = self.build_aspect_schema(aspect)?;
        schemas.insert(aspect.name(), aspect_schema);

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
