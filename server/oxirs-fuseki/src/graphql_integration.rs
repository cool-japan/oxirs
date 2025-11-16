//! GraphQL Integration
//!
//! Provides a GraphQL interface for querying RDF data alongside SPARQL.
//! This module translates GraphQL queries to SPARQL and provides a modern
//! GraphQL API for data access.

use anyhow::Result;
use async_graphql::{
    Context, EmptyMutation, EmptySubscription, ErrorExtensions, Object, Schema, SimpleObject, ID,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info};

/// GraphQL schema for RDF data
pub type OxirsSchema = Schema<QueryRoot, EmptyMutation, EmptySubscription>;

/// Query root for GraphQL
pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// Get information about available datasets
    async fn datasets(&self, ctx: &Context<'_>) -> async_graphql::Result<Vec<Dataset>> {
        let store = ctx
            .data::<Arc<crate::store::Store>>()
            .map_err(|e| e.extend_with(|_, e| e.set("code", "STORE_NOT_FOUND")))?;

        let datasets = store
            .list_datasets()
            .map_err(|e| async_graphql::Error::new(e.to_string()))?;

        Ok(datasets
            .into_iter()
            .map(|name| Dataset {
                id: name.clone().into(),
                name: name.clone(),
                triple_count: get_triple_count(store.clone(), &name),
            })
            .collect())
    }

    /// Get a specific dataset by name
    async fn dataset(
        &self,
        ctx: &Context<'_>,
        name: String,
    ) -> async_graphql::Result<Option<Dataset>> {
        let store = ctx
            .data::<Arc<crate::store::Store>>()
            .map_err(|e| e.extend_with(|_, e| e.set("code", "STORE_NOT_FOUND")))?;

        if dataset_exists(store.clone(), &name) {
            let triple_count = get_triple_count(store.clone(), &name);
            Ok(Some(Dataset {
                id: name.clone().into(),
                name,
                triple_count,
            }))
        } else {
            Ok(None)
        }
    }

    /// Execute a SPARQL query
    async fn sparql_query(
        &self,
        ctx: &Context<'_>,
        dataset: String,
        query: String,
        limit: Option<i32>,
    ) -> async_graphql::Result<QueryResult> {
        let store = ctx
            .data::<Arc<crate::store::Store>>()
            .map_err(|e| e.extend_with(|_, e| e.set("code", "STORE_NOT_FOUND")))?;

        // Add LIMIT if not present and limit is specified
        let query = if let Some(limit) = limit {
            if !query.to_lowercase().contains("limit") {
                format!("{} LIMIT {}", query, limit)
            } else {
                query
            }
        } else {
            query
        };

        let results = store.query_dataset(&query, Some(&dataset)).map_err(|e| {
            error!("SPARQL query failed: {}", e);
            async_graphql::Error::new(format!("Query execution failed: {}", e))
                .extend_with(|_, e| e.set("code", "QUERY_FAILED"))
        })?;

        // Convert to GraphQL response based on query type
        match &results.inner {
            oxirs_core::query::QueryResult::Select {
                variables,
                bindings,
            } => Ok(QueryResult {
                bindings: bindings
                    .iter()
                    .map(|binding| Binding {
                        values: variables
                            .iter()
                            .filter_map(|var| {
                                binding.get(var).map(|term| BindingValue {
                                    variable: var.clone(),
                                    value: term.to_string(),
                                    value_type: match term {
                                        oxirs_core::model::Term::NamedNode(_) => "uri",
                                        oxirs_core::model::Term::BlankNode(_) => "bnode",
                                        oxirs_core::model::Term::Literal(_) => "literal",
                                        oxirs_core::model::Term::Variable(_) => "variable",
                                        oxirs_core::model::Term::QuotedTriple(_) => "quoted-triple",
                                    }
                                    .to_string(),
                                })
                            })
                            .collect(),
                    })
                    .collect(),
                count: bindings.len(),
            }),
            _ => Err(async_graphql::Error::new(
                "Only SELECT queries are supported in GraphQL",
            )),
        }
    }

    /// Get triples from a dataset
    #[allow(clippy::too_many_arguments)]
    async fn triples(
        &self,
        ctx: &Context<'_>,
        dataset: String,
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
        limit: Option<i32>,
        offset: Option<i32>,
    ) -> async_graphql::Result<Vec<Triple>> {
        let store = ctx
            .data::<Arc<crate::store::Store>>()
            .map_err(|e| e.extend_with(|_, e| e.set("code", "STORE_NOT_FOUND")))?;

        // Build SPARQL query
        let s = subject.as_deref().unwrap_or("?s");
        let p = predicate.as_deref().unwrap_or("?p");
        let o = object.as_deref().unwrap_or("?o");

        let mut query = format!("SELECT ?s ?p ?o WHERE {{ {} {} {} }}", s, p, o);

        if let Some(limit) = limit {
            query.push_str(&format!(" LIMIT {}", limit));
        }

        if let Some(offset) = offset {
            query.push_str(&format!(" OFFSET {}", offset));
        }

        let results = store
            .query_dataset(&query, Some(&dataset))
            .map_err(|e| async_graphql::Error::new(format!("Failed to fetch triples: {}", e)))?;

        match &results.inner {
            oxirs_core::query::QueryResult::Select { bindings, .. } => Ok(bindings
                .iter()
                .map(|binding| Triple {
                    subject: binding.get("s").map(|t| t.to_string()).unwrap_or_default(),
                    predicate: binding.get("p").map(|t| t.to_string()).unwrap_or_default(),
                    object: binding.get("o").map(|t| t.to_string()).unwrap_or_default(),
                })
                .collect()),
            _ => Ok(vec![]),
        }
    }

    /// Search for resources
    async fn search(
        &self,
        ctx: &Context<'_>,
        dataset: String,
        query: String,
        limit: Option<i32>,
    ) -> async_graphql::Result<Vec<Resource>> {
        let store = ctx
            .data::<Arc<crate::store::Store>>()
            .map_err(|e| e.extend_with(|_, e| e.set("code", "STORE_NOT_FOUND")))?;

        // Simple text search using SPARQL FILTER and REGEX
        let sparql_query = format!(
            r#"
            SELECT DISTINCT ?subject WHERE {{
                ?subject ?predicate ?object .
                FILTER(REGEX(STR(?subject), "{}", "i") || REGEX(STR(?object), "{}", "i"))
            }} LIMIT {}
            "#,
            query,
            query,
            limit.unwrap_or(10)
        );

        let results = store
            .query_dataset(&sparql_query, Some(&dataset))
            .map_err(|e| async_graphql::Error::new(format!("Search failed: {}", e)))?;

        match &results.inner {
            oxirs_core::query::QueryResult::Select { bindings, .. } => Ok(bindings
                .iter()
                .map(|binding| Resource {
                    uri: binding
                        .get("subject")
                        .map(|t| t.to_string())
                        .unwrap_or_default(),
                    label: None,
                    description: None,
                    properties: vec![],
                })
                .collect()),
            _ => Ok(vec![]),
        }
    }

    /// Get statistics about the system
    async fn statistics(&self, ctx: &Context<'_>) -> async_graphql::Result<Statistics> {
        let store = ctx
            .data::<Arc<crate::store::Store>>()
            .map_err(|e| e.extend_with(|_, e| e.set("code", "STORE_NOT_FOUND")))?;

        let datasets = store
            .list_datasets()
            .map_err(|e| async_graphql::Error::new(e.to_string()))?;
        let total_triples: usize = datasets
            .iter()
            .map(|ds| get_triple_count(store.clone(), ds))
            .sum();

        Ok(Statistics {
            dataset_count: datasets.len() as i32,
            total_triples: total_triples as i32,
            query_count: 0, // Would need metrics integration
            avg_query_time_ms: 0.0,
            uptime_seconds: 0, // Would need startup time tracking
        })
    }
}

// Helper functions
fn dataset_exists(store: Arc<crate::store::Store>, name: &str) -> bool {
    store
        .list_datasets()
        .map(|datasets| datasets.contains(&name.to_string()))
        .unwrap_or(false)
}

fn get_triple_count(store: Arc<crate::store::Store>, dataset_name: &str) -> usize {
    store
        .get_stats(Some(dataset_name))
        .ok()
        .map(|stats| stats.triple_count)
        .unwrap_or(0)
}

/// Dataset representation
#[derive(SimpleObject, Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub id: ID,
    pub name: String,
    pub triple_count: usize,
}

/// Query result
#[derive(SimpleObject, Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub bindings: Vec<Binding>,
    pub count: usize,
}

/// Variable binding
#[derive(SimpleObject, Debug, Clone, Serialize, Deserialize)]
pub struct Binding {
    pub values: Vec<BindingValue>,
}

/// Binding value
#[derive(SimpleObject, Debug, Clone, Serialize, Deserialize)]
pub struct BindingValue {
    pub variable: String,
    pub value: String,
    pub value_type: String,
}

/// RDF Triple
#[derive(SimpleObject, Debug, Clone, Serialize, Deserialize)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Resource representation
#[derive(SimpleObject, Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub uri: String,
    pub label: Option<String>,
    pub description: Option<String>,
    pub properties: Vec<Property>,
}

/// Resource property
#[derive(SimpleObject, Debug, Clone, Serialize, Deserialize)]
pub struct Property {
    pub predicate: String,
    pub value: String,
}

/// System statistics
#[derive(SimpleObject, Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    pub dataset_count: i32,
    pub total_triples: i32,
    pub query_count: i32,
    pub avg_query_time_ms: f64,
    pub uptime_seconds: i64,
}

/// GraphQL service
pub struct GraphQLService {
    schema: OxirsSchema,
    store: Arc<crate::store::Store>,
}

impl GraphQLService {
    /// Create a new GraphQL service
    pub fn new(store: Arc<crate::store::Store>) -> Self {
        let schema = Schema::build(QueryRoot, EmptyMutation, EmptySubscription)
            .data(store.clone())
            .finish();

        Self { schema, store }
    }

    /// Get the GraphQL schema
    pub fn schema(&self) -> &OxirsSchema {
        &self.schema
    }

    /// Execute a GraphQL query
    pub async fn execute(&self, query: &str) -> Result<async_graphql::Response> {
        debug!("Executing GraphQL query: {}", query);

        let result = self.schema.execute(query).await;

        if result.is_err() {
            error!("GraphQL query failed with errors: {:?}", result.errors);
        }

        Ok(result)
    }

    /// Execute a GraphQL query with variables
    pub async fn execute_with_variables(
        &self,
        query: &str,
        variables: serde_json::Value,
    ) -> Result<async_graphql::Response> {
        debug!("Executing GraphQL query with variables: {}", query);

        let result = self
            .schema
            .execute(
                async_graphql::Request::new(query)
                    .variables(async_graphql::Variables::from_json(variables)),
            )
            .await;

        if result.is_err() {
            error!("GraphQL query failed with errors: {:?}", result.errors);
        }

        Ok(result)
    }

    /// Get the GraphQL schema SDL
    pub fn sdl(&self) -> String {
        self.schema.sdl()
    }
}

/// GraphQL query request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_name: Option<String>,
}

/// GraphQL response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub errors: Vec<GraphQLError>,
}

/// GraphQL error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLError {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locations: Option<Vec<Location>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
}

/// Error location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub line: usize,
    pub column: usize,
}

// HTTP Handlers for Axum integration

/// GraphQL query handler (POST /graphql)
pub async fn graphql_handler(
    axum::extract::State(state): axum::extract::State<Arc<crate::server::AppState>>,
    axum::Json(request): axum::Json<GraphQLRequest>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    // Create GraphQL service with store
    let service = GraphQLService::new(Arc::new(state.store.clone()));

    // Execute query
    let variables = request.variables.unwrap_or(serde_json::Value::Null);

    let response = service
        .schema
        .execute(
            async_graphql::Request::new(request.query)
                .variables(async_graphql::Variables::from_json(variables)),
        )
        .await;

    // Convert async-graphql Response to our GraphQLResponse format
    let json_response = GraphQLResponse {
        data: if response.is_ok() {
            Some(serde_json::to_value(&response.data).unwrap_or_default())
        } else {
            None
        },
        errors: response
            .errors
            .into_iter()
            .map(|e| {
                let locations_vec: Option<Vec<Location>> = if !e.locations.is_empty() {
                    Some(
                        e.locations
                            .iter()
                            .map(|loc| Location {
                                line: loc.line,
                                column: loc.column,
                            })
                            .collect(),
                    )
                } else {
                    None
                };

                let path_vec: Option<Vec<String>> = if !e.path.is_empty() {
                    Some(
                        e.path
                            .iter()
                            .map(|segment| match segment {
                                async_graphql::PathSegment::Field(f) => f.to_string(),
                                async_graphql::PathSegment::Index(i) => i.to_string(),
                            })
                            .collect(),
                    )
                } else {
                    None
                };

                GraphQLError {
                    message: e.message,
                    locations: locations_vec,
                    path: path_vec,
                    extensions: if e.extensions.is_some() {
                        e.extensions.and_then(|ext| serde_json::to_value(&ext).ok())
                    } else {
                        None
                    },
                }
            })
            .collect(),
    };

    axum::Json(json_response).into_response()
}

/// GraphQL Playground handler (GET /graphql/playground)
pub async fn graphql_playground() -> axum::response::Html<&'static str> {
    axum::response::Html(
        r#"
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>GraphQL Playground - OxiRS Fuseki</title>
    <style>
        body {
            height: 100%;
            margin: 0;
            width: 100%;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }
        #root {
            height: 100vh;
            width: 100vw;
        }
    </style>
    <script src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/graphql-playground-react/build/static/css/index.css" />
    <script src="https://unpkg.com/graphql-playground-react/build/static/js/middleware.js"></script>
</head>
<body>
    <div id="root"></div>
    <script>
        window.addEventListener('load', function (event) {
            GraphQLPlayground.init(document.getElementById('root'), {
                endpoint: '/graphql',
                settings: {
                    'editor.theme': 'dark',
                    'editor.cursorShape': 'line',
                    'editor.reuseHeaders': true,
                    'tracing.hideTracingResponse': true,
                    'queryPlan.hideQueryPlanResponse': true,
                    'editor.fontSize': 14,
                    'editor.fontFamily': "'Source Code Pro', 'Consolas', 'Inconsolata', 'Droid Sans Mono', 'Monaco', monospace",
                    'request.credentials': 'include',
                },
                tabs: [
                    {
                        endpoint: '/graphql',
                        name: 'OxiRS Fuseki GraphQL',
                        query: '# Welcome to OxiRS Fuseki GraphQL Playground\n# Query RDF data using GraphQL\n\nquery {\n  datasets {\n    name\n    tripleCount\n  }\n}',
                    },
                ],
            })
        })
    </script>
</body>
</html>
"#,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graphql_request_serialization() {
        let request = GraphQLRequest {
            query: "{ datasets { name } }".to_string(),
            variables: None,
            operation_name: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("datasets"));
    }

    #[test]
    fn test_dataset_structure() {
        let dataset = Dataset {
            id: "test".into(),
            name: "test".to_string(),
            triple_count: 100,
        };

        assert_eq!(dataset.name, "test");
        assert_eq!(dataset.triple_count, 100);
    }

    #[test]
    fn test_triple_structure() {
        let triple = Triple {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "http://example.org/object".to_string(),
        };

        assert!(triple.subject.starts_with("http://"));
    }
}
