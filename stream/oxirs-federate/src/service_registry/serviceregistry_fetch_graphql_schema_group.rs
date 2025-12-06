//! # ServiceRegistry - fetch_graphql_schema_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use anyhow::{anyhow, Result};
use std::time::Duration;

impl ServiceRegistry {
    /// Fetch GraphQL schema using introspection
    pub(super) async fn fetch_graphql_schema(&self, endpoint_url: &str) -> Result<String> {
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
        let request_body = serde_json::json!({ "query" : introspection_query });
        let response = self
            .http_client
            .post(endpoint_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&request_body)
            .timeout(Duration::from_secs(10))
            .send()
            .await?;
        if response.status().is_success() {
            Ok(response.text().await?)
        } else {
            Err(anyhow!(
                "GraphQL introspection failed: {}",
                response.status()
            ))
        }
    }
}
