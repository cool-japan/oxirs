//! Schema Introspection and Capability Discovery
//!
//! This module handles GraphQL schema introspection, capability discovery,
//! and federation support detection for remote services.

use anyhow::{anyhow, Result};
use std::collections::HashSet;
use tracing::debug;

use super::types::*;

/// Schema introspection utilities
#[derive(Debug)]
pub struct SchemaIntrospector;

impl SchemaIntrospector {
    /// Advanced schema discovery with introspection
    pub async fn discover_schema_capabilities(
        service_endpoint: &str,
    ) -> Result<SchemaCapabilities> {
        debug!("Discovering schema capabilities for {}", service_endpoint);

        let introspection_query = r#"
            query IntrospectionQuery {
                __schema {
                    queryType { name }
                    mutationType { name }
                    subscriptionType { name }
                    types {
                        ...FullType
                    }
                    directives {
                        name
                        description
                        locations
                        args {
                            ...InputValue
                        }
                    }
                }
            }
            
            fragment FullType on __Type {
                kind
                name
                description
                fields(includeDeprecated: true) {
                    name
                    description
                    args {
                        ...InputValue
                    }
                    type {
                        ...TypeRef
                    }
                    isDeprecated
                    deprecationReason
                }
                inputFields {
                    ...InputValue
                }
                interfaces {
                    ...TypeRef
                }
                enumValues(includeDeprecated: true) {
                    name
                    description
                    isDeprecated
                    deprecationReason
                }
                possibleTypes {
                    ...TypeRef
                }
            }
            
            fragment InputValue on __InputValue {
                name
                description
                type { ...TypeRef }
                defaultValue
            }
            
            fragment TypeRef on __Type {
                kind
                name
                ofType {
                    kind
                    name
                    ofType {
                        kind
                        name
                        ofType {
                            kind
                            name
                            ofType {
                                kind
                                name
                                ofType {
                                    kind
                                    name
                                    ofType {
                                        kind
                                        name
                                        ofType {
                                            kind
                                            name
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        "#;

        let response =
            Self::execute_introspection_query(service_endpoint, introspection_query).await?;
        Self::parse_introspection_response(response)
    }

    /// Execute introspection query against a GraphQL service
    async fn execute_introspection_query(endpoint: &str, query: &str) -> Result<serde_json::Value> {
        let client = reqwest::Client::new();
        let request_body = serde_json::json!({
            "query": query
        });

        let response = client
            .post(endpoint)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Introspection query failed with status: {}",
                response.status()
            ));
        }

        let body: serde_json::Value = response.json().await?;
        Ok(body)
    }

    /// Parse introspection response to extract schema capabilities
    fn parse_introspection_response(response: serde_json::Value) -> Result<SchemaCapabilities> {
        let schema = response["data"]["__schema"]
            .as_object()
            .ok_or_else(|| anyhow!("Invalid introspection response: missing schema"))?;

        let mut capabilities = SchemaCapabilities {
            supports_federation: false,
            supports_subscriptions: false,
            supports_defer_stream: false,
            entity_types: Vec::new(),
            custom_directives: Vec::new(),
            scalar_types: Vec::new(),
            estimated_complexity: 0.0,
        };

        // Check for federation support
        if let Some(directives) = schema["directives"].as_array() {
            for directive in directives {
                if let Some(name) = directive["name"].as_str() {
                    capabilities.custom_directives.push(name.to_string());

                    // Federation directives
                    if matches!(
                        name,
                        "key" | "external" | "requires" | "provides" | "extends"
                    ) {
                        capabilities.supports_federation = true;
                    }
                }
            }
        }

        // Check for subscription support
        if schema["subscriptionType"].is_object() {
            capabilities.supports_subscriptions = true;
        }

        // Analyze types for entities and complexity
        if let Some(types) = schema["types"].as_array() {
            for type_def in types {
                if let Some(type_name) = type_def["name"].as_str() {
                    // Skip GraphQL built-in types
                    if type_name.starts_with("__") {
                        continue;
                    }

                    if let Some(kind) = type_def["kind"].as_str() {
                        match kind {
                            "OBJECT" => {
                                capabilities.estimated_complexity += 1.0;

                                // Check if this could be an entity (has ID field)
                                if let Some(fields) = type_def["fields"].as_array() {
                                    let has_id = fields
                                        .iter()
                                        .any(|field| field["name"].as_str() == Some("id"));

                                    if has_id {
                                        capabilities.entity_types.push(type_name.to_string());
                                    }
                                }
                            }
                            "SCALAR" => {
                                if !matches!(
                                    type_name,
                                    "String" | "Int" | "Float" | "Boolean" | "ID"
                                ) {
                                    capabilities.scalar_types.push(type_name.to_string());
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Check for @defer/@stream support (Apollo Federation 2.0+)
        capabilities.supports_defer_stream = capabilities
            .custom_directives
            .iter()
            .any(|d| matches!(d.as_str(), "defer" | "stream"));

        Ok(capabilities)
    }

    /// Introspect a GraphQL service for Apollo Federation support
    pub async fn introspect_federation_support(
        service_endpoint: &str,
    ) -> Result<FederationServiceInfo> {
        debug!(
            "Introspecting GraphQL service for federation support: {}",
            service_endpoint
        );

        // Query for federation support
        let _federation_query = r#"
            query FederationIntrospection {
                _service {
                    sdl
                }
                __schema {
                    types {
                        name
                        fields {
                            name
                            type {
                                name
                                kind
                            }
                        }
                    }
                }
            }
        "#;

        // In a real implementation, this would make an HTTP request to the service
        // For now, return mock data
        let mock_sdl = r#"
            extend type Query {
                me: User
            }
            
            type User @key(fields: "id") {
                id: ID!
                username: String!
                email: String! @external
            }
        "#;

        Ok(FederationServiceInfo {
            sdl: mock_sdl.to_string(),
            capabilities: FederationCapabilities {
                federation_version: "2.0".to_string(),
                supports_entities: true,
                supports_entity_interfaces: true,
                supports_progressive_override: false,
            },
            entity_types: vec!["User".to_string()],
        })
    }

    /// Parse Apollo Federation directives from a type definition
    pub fn parse_federation_directives(type_def: &TypeDefinition) -> FederationDirectives {
        let mut fed_directives = FederationDirectives {
            key: None,
            external: false,
            requires: None,
            provides: None,
            extends: false,
            shareable: false,
            override_from: None,
            inaccessible: false,
            tags: Vec::new(),
        };

        for directive in &type_def.directives {
            match directive.name.as_str() {
                "key" => {
                    if let Some(fields_arg) = directive.arguments.get("fields") {
                        if let Some(fields) = fields_arg.as_str() {
                            let resolvable = directive
                                .arguments
                                .get("resolvable")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(true);

                            fed_directives.key = Some(KeyDirective {
                                fields: fields.to_string(),
                                resolvable,
                            });
                        }
                    }
                }
                "external" => fed_directives.external = true,
                "requires" => {
                    if let Some(fields_arg) = directive.arguments.get("fields") {
                        if let Some(fields) = fields_arg.as_str() {
                            fed_directives.requires = Some(fields.to_string());
                        }
                    }
                }
                "provides" => {
                    if let Some(fields_arg) = directive.arguments.get("fields") {
                        if let Some(fields) = fields_arg.as_str() {
                            fed_directives.provides = Some(fields.to_string());
                        }
                    }
                }
                "extends" => fed_directives.extends = true,
                "shareable" => fed_directives.shareable = true,
                "override" => {
                    if let Some(from_arg) = directive.arguments.get("from") {
                        if let Some(from) = from_arg.as_str() {
                            fed_directives.override_from = Some(from.to_string());
                        }
                    }
                }
                "inaccessible" => fed_directives.inaccessible = true,
                "tag" => {
                    if let Some(name_arg) = directive.arguments.get("name") {
                        if let Some(tag) = name_arg.as_str() {
                            fed_directives.tags.push(tag.to_string());
                        }
                    }
                }
                _ => {}
            }
        }

        fed_directives
    }

    /// Analyze service compatibility for federation
    pub fn analyze_service_compatibility(
        service_info: &FederationServiceInfo,
        requirements: &FederationRequirements,
    ) -> ServiceCompatibilityReport {
        let mut report = ServiceCompatibilityReport {
            is_compatible: true,
            warnings: Vec::new(),
            missing_features: Vec::new(),
            version_mismatch: false,
        };

        // Check federation version compatibility
        if !Self::is_version_compatible(
            &service_info.capabilities.federation_version,
            &requirements.min_federation_version,
        ) {
            report.is_compatible = false;
            report.version_mismatch = true;
            report.warnings.push(format!(
                "Federation version {} is below minimum required {}",
                service_info.capabilities.federation_version, requirements.min_federation_version
            ));
        }

        // Check required features
        if requirements.requires_entities && !service_info.capabilities.supports_entities {
            report.is_compatible = false;
            report.missing_features.push("entities".to_string());
        }

        if requirements.requires_entity_interfaces
            && !service_info.capabilities.supports_entity_interfaces
        {
            report
                .warnings
                .push("Entity interfaces not supported".to_string());
        }

        if requirements.requires_progressive_override
            && !service_info.capabilities.supports_progressive_override
        {
            report
                .warnings
                .push("Progressive override not supported".to_string());
        }

        // Check entity types
        for required_entity in &requirements.required_entity_types {
            if !service_info.entity_types.contains(required_entity) {
                report.is_compatible = false;
                report
                    .missing_features
                    .push(format!("entity type: {required_entity}"));
            }
        }

        report
    }

    /// Check if federation version is compatible
    fn is_version_compatible(service_version: &str, min_version: &str) -> bool {
        // Simple version comparison (major.minor)
        let service_parts: Vec<&str> = service_version.split('.').collect();
        let min_parts: Vec<&str> = min_version.split('.').collect();

        if service_parts.len() < 2 || min_parts.len() < 2 {
            return false;
        }

        let service_major = service_parts[0].parse::<u32>().unwrap_or(0);
        let service_minor = service_parts[1].parse::<u32>().unwrap_or(0);
        let min_major = min_parts[0].parse::<u32>().unwrap_or(0);
        let min_minor = min_parts[1].parse::<u32>().unwrap_or(0);

        service_major > min_major || (service_major == min_major && service_minor >= min_minor)
    }

    /// Extract schema types and their relationships
    pub fn extract_schema_metadata(response: serde_json::Value) -> Result<SchemaMetadata> {
        let schema = response["data"]["__schema"]
            .as_object()
            .ok_or_else(|| anyhow!("Invalid introspection response"))?;

        let mut metadata = SchemaMetadata {
            types: Vec::new(),
            query_fields: Vec::new(),
            mutation_fields: Vec::new(),
            subscription_fields: Vec::new(),
            directives: Vec::new(),
            entity_types: HashSet::new(),
        };

        // Extract types
        if let Some(types) = schema["types"].as_array() {
            for type_def in types {
                if let Some(type_name) = type_def["name"].as_str() {
                    if !type_name.starts_with("__") {
                        metadata.types.push(type_name.to_string());

                        // Check if it's an entity type (has @key directive)
                        if Self::has_key_directive(type_def) {
                            metadata.entity_types.insert(type_name.to_string());
                        }
                    }
                }
            }
        }

        // Extract query fields
        if let Some(query_type) = schema["queryType"].as_object() {
            if let Some(query_name) = query_type["name"].as_str() {
                // Find the query type in types array and extract fields
                if let Some(types) = schema["types"].as_array() {
                    for type_def in types {
                        if type_def["name"].as_str() == Some(query_name) {
                            if let Some(fields) = type_def["fields"].as_array() {
                                for field in fields {
                                    if let Some(field_name) = field["name"].as_str() {
                                        metadata.query_fields.push(field_name.to_string());
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Extract directives
        if let Some(directives) = schema["directives"].as_array() {
            for directive in directives {
                if let Some(directive_name) = directive["name"].as_str() {
                    metadata.directives.push(directive_name.to_string());
                }
            }
        }

        Ok(metadata)
    }

    /// Check if a type definition has a @key directive
    fn has_key_directive(type_def: &serde_json::Value) -> bool {
        // This is a simplified check - in reality, we'd parse the SDL
        // and look for @key directives
        if let Some(description) = type_def["description"].as_str() {
            description.contains("@key")
        } else {
            false
        }
    }

    /// Generate service capability summary
    pub fn generate_capability_summary(capabilities: &SchemaCapabilities) -> String {
        let mut summary = String::new();

        summary.push_str(&format!(
            "Federation Support: {}\n",
            if capabilities.supports_federation {
                "Yes"
            } else {
                "No"
            }
        ));

        summary.push_str(&format!(
            "Subscriptions: {}\n",
            if capabilities.supports_subscriptions {
                "Yes"
            } else {
                "No"
            }
        ));

        summary.push_str(&format!(
            "Defer/Stream: {}\n",
            if capabilities.supports_defer_stream {
                "Yes"
            } else {
                "No"
            }
        ));

        summary.push_str(&format!(
            "Entity Types: {}\n",
            capabilities.entity_types.len()
        ));
        summary.push_str(&format!(
            "Custom Directives: {}\n",
            capabilities.custom_directives.len()
        ));
        summary.push_str(&format!(
            "Custom Scalars: {}\n",
            capabilities.scalar_types.len()
        ));
        summary.push_str(&format!(
            "Estimated Complexity: {:.1}\n",
            capabilities.estimated_complexity
        ));

        if !capabilities.entity_types.is_empty() {
            summary.push_str("Entity Types: ");
            summary.push_str(&capabilities.entity_types.join(", "));
            summary.push('\n');
        }

        if !capabilities.custom_directives.is_empty() {
            summary.push_str("Custom Directives: ");
            summary.push_str(&capabilities.custom_directives.join(", "));
            summary.push('\n');
        }

        summary
    }
}

/// Federation requirements for service compatibility
#[derive(Debug, Clone)]
pub struct FederationRequirements {
    pub min_federation_version: String,
    pub requires_entities: bool,
    pub requires_entity_interfaces: bool,
    pub requires_progressive_override: bool,
    pub required_entity_types: Vec<String>,
}

impl Default for FederationRequirements {
    fn default() -> Self {
        Self {
            min_federation_version: "2.0".to_string(),
            requires_entities: true,
            requires_entity_interfaces: false,
            requires_progressive_override: false,
            required_entity_types: Vec::new(),
        }
    }
}

/// Service compatibility report
#[derive(Debug, Clone)]
pub struct ServiceCompatibilityReport {
    pub is_compatible: bool,
    pub warnings: Vec<String>,
    pub missing_features: Vec<String>,
    pub version_mismatch: bool,
}

/// Schema metadata extracted from introspection
#[derive(Debug, Clone)]
pub struct SchemaMetadata {
    pub types: Vec<String>,
    pub query_fields: Vec<String>,
    pub mutation_fields: Vec<String>,
    pub subscription_fields: Vec<String>,
    pub directives: Vec<String>,
    pub entity_types: HashSet<String>,
}
