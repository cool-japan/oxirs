//! GraphQL Federation and Schema Stitching
//!
//! This module provides GraphQL federation capabilities, including schema stitching,
//! query planning, and distributed execution across multiple GraphQL services.
//!
//! The module is organized into the following components:
//! - `types`: Type definitions and data structures for GraphQL federation
//! - `core`: Main GraphQLFederation implementation with core operations
//! - `schema_management`: Schema registration, merging, validation, and unified schema creation
//! - `query_processing`: Query parsing, decomposition, and field ownership analysis
//! - `entity_resolution`: Entity resolution, dependency graphs, and advanced federation operations
//! - `translation`: GraphQL to SPARQL translation layer for hybrid query processing

pub mod types;
pub mod core;
pub mod schema_management;
pub mod query_processing;
pub mod entity_resolution;
pub mod translation;

// Re-export main types and structs for public API
pub use types::*;
pub use core::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_graphql_federation_creation() {
        let federation = GraphQLFederation::new();
        assert!(federation.config.enable_schema_stitching);
    }

    #[tokio::test]
    async fn test_schema_registration() {
        let federation = GraphQLFederation::new();

        let schema = FederatedSchema {
            service_id: "test-service".to_string(),
            types: HashMap::new(),
            queries: HashMap::new(),
            mutations: HashMap::new(),
            subscriptions: HashMap::new(),
            directives: HashMap::new(),
        };

        let result = federation
            .register_schema("test-service".to_string(), schema)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_unified_schema_creation() {
        let federation = GraphQLFederation::new();

        // Register an empty schema
        let schema = FederatedSchema {
            service_id: "test-service".to_string(),
            types: HashMap::new(),
            queries: HashMap::new(),
            mutations: HashMap::new(),
            subscriptions: HashMap::new(),
            directives: HashMap::new(),
        };

        federation
            .register_schema("test-service".to_string(), schema)
            .await
            .unwrap();

        let unified_result = federation.create_unified_schema().await;
        assert!(unified_result.is_ok());
    }

    #[tokio::test]
    async fn test_query_parsing() {
        let federation = GraphQLFederation::new();

        let query = r#"
            query GetUser {
                me {
                    id
                    username
                    email
                }
            }
        "#;

        let parsed_result = federation.parse_graphql_query(query);
        assert!(parsed_result.is_ok());

        let parsed_query = parsed_result.unwrap();
        assert!(matches!(parsed_query.operation_type, GraphQLOperationType::Query));
        assert_eq!(parsed_query.operation_name, Some("GetUser".to_string()));
        assert!(!parsed_query.selection_set.is_empty());
    }

    #[tokio::test]
    async fn test_query_complexity_analysis() {
        let federation = GraphQLFederation::new();

        let query = ParsedQuery {
            operation_type: GraphQLOperationType::Query,
            operation_name: Some("TestQuery".to_string()),
            selection_set: vec![
                Selection {
                    name: "user".to_string(),
                    alias: None,
                    arguments: HashMap::new(),
                    selection_set: vec![
                        Selection {
                            name: "id".to_string(),
                            alias: None,
                            arguments: HashMap::new(),
                            selection_set: Vec::new(),
                        },
                        Selection {
                            name: "profile".to_string(),
                            alias: None,
                            arguments: HashMap::new(),
                            selection_set: vec![
                                Selection {
                                    name: "bio".to_string(),
                                    alias: None,
                                    arguments: HashMap::new(),
                                    selection_set: Vec::new(),
                                },
                            ],
                        },
                    ],
                },
            ],
            variables: HashMap::new(),
        };

        let complexity_result = federation.analyze_query_complexity(&query);
        assert!(complexity_result.is_ok());

        let complexity = complexity_result.unwrap();
        assert_eq!(complexity.max_depth, 3);
        assert_eq!(complexity.field_count, 3);
        assert!(complexity.total_complexity > 0);
    }

    #[tokio::test]
    async fn test_federation_directives_parsing() {
        let federation = GraphQLFederation::new();

        let type_def = TypeDefinition {
            name: "User".to_string(),
            description: Some("User entity".to_string()),
            kind: TypeKind::Object {
                fields: HashMap::new(),
            },
            directives: vec![
                Directive {
                    name: "key".to_string(),
                    arguments: {
                        let mut args = HashMap::new();
                        args.insert("fields".to_string(), serde_json::Value::String("id".to_string()));
                        args
                    },
                },
                Directive {
                    name: "external".to_string(),
                    arguments: HashMap::new(),
                },
            ],
        };

        let fed_directives = federation.parse_federation_directives(&type_def);
        assert_eq!(fed_directives.key, Some("id".to_string()));
        assert!(fed_directives.external);
        assert!(!fed_directives.shareable);
    }

    #[tokio::test]
    async fn test_entity_dependency_analysis() {
        let federation = GraphQLFederation::new();

        let entity_a = EntityReference {
            entity_type: "User".to_string(),
            key_fields: vec!["id".to_string()],
            required_fields: vec!["profile_id".to_string()],
            service_id: "user-service".to_string(),
        };

        let entity_b = EntityReference {
            entity_type: "Profile".to_string(),
            key_fields: vec!["profile_id".to_string()],
            required_fields: vec!["bio".to_string()],
            service_id: "profile-service".to_string(),
        };

        let has_dependency = federation.entities_have_dependency(&entity_a, &entity_b);
        assert!(has_dependency.is_ok());
        assert!(has_dependency.unwrap());
    }

    #[tokio::test]
    async fn test_service_query_creation() {
        let federation = GraphQLFederation::new();

        let parsed_query = ParsedQuery {
            operation_type: GraphQLOperationType::Query,
            operation_name: None,
            selection_set: vec![
                Selection {
                    name: "user".to_string(),
                    alias: None,
                    arguments: HashMap::new(),
                    selection_set: Vec::new(),
                },
                Selection {
                    name: "product".to_string(),
                    alias: None,
                    arguments: HashMap::new(),
                    selection_set: Vec::new(),
                },
            ],
            variables: HashMap::new(),
        };

        let ownership = FieldOwnership {
            field_to_service: {
                let mut map = HashMap::new();
                map.insert("user".to_string(), "user-service".to_string());
                map.insert("product".to_string(), "product-service".to_string());
                map
            },
            service_to_fields: {
                let mut map = HashMap::new();
                map.insert("user-service".to_string(), vec!["user".to_string()]);
                map.insert("product-service".to_string(), vec!["product".to_string()]);
                map
            },
        };

        let service_queries = federation.create_service_queries(&parsed_query, &ownership);
        assert!(service_queries.is_ok());

        let queries = service_queries.unwrap();
        assert_eq!(queries.len(), 2);
        
        let user_query = queries.iter().find(|q| q.service_id == "user-service");
        assert!(user_query.is_some());
        assert!(user_query.unwrap().query.contains("user"));
        
        let product_query = queries.iter().find(|q| q.service_id == "product-service");
        assert!(product_query.is_some());
        assert!(product_query.unwrap().query.contains("product"));
    }

    #[tokio::test]
    async fn test_schema_capabilities_analysis() {
        let federation = GraphQLFederation::new();

        let schema = FederatedSchema {
            service_id: "test-service".to_string(),
            types: {
                let mut types = HashMap::new();
                types.insert("User".to_string(), TypeDefinition {
                    name: "User".to_string(),
                    description: None,
                    kind: TypeKind::Object { fields: HashMap::new() },
                    directives: vec![
                        Directive {
                            name: "key".to_string(),
                            arguments: HashMap::new(),
                        },
                    ],
                });
                types
            },
            queries: HashMap::new(),
            mutations: HashMap::new(),
            subscriptions: {
                let mut subs = HashMap::new();
                subs.insert("userUpdated".to_string(), FieldDefinition {
                    name: "userUpdated".to_string(),
                    description: None,
                    field_type: "User".to_string(),
                    arguments: HashMap::new(),
                    directives: Vec::new(),
                });
                subs
            },
            directives: {
                let mut dirs = HashMap::new();
                dirs.insert("key".to_string(), DirectiveDefinition {
                    name: "key".to_string(),
                    description: None,
                    locations: vec![DirectiveLocation::Object],
                    arguments: HashMap::new(),
                    repeatable: false,
                });
                dirs
            },
        };

        federation.register_schema("test-service".to_string(), schema).await.unwrap();

        let capabilities = federation.analyze_schema_capabilities("test-service").await;
        assert!(capabilities.is_ok());

        let caps = capabilities.unwrap();
        assert!(caps.supports_federation);
        assert!(caps.supports_subscriptions);
        assert_eq!(caps.entity_types, vec!["User".to_string()]);
        assert!(caps.estimated_complexity > 0.0);
    }
}