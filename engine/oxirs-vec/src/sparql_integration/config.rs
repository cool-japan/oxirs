//! Configuration types and defaults for SPARQL vector integration

use crate::similarity::SimilarityMetric;
use serde::{Deserialize, Serialize};

/// SPARQL vector service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorServiceConfig {
    /// Service namespace URI
    pub service_uri: String,
    /// Default similarity threshold
    pub default_threshold: f32,
    /// Default number of results to return
    pub default_limit: usize,
    /// Default similarity metric
    pub default_metric: SimilarityMetric,
    /// Enable caching of vector search results
    pub enable_caching: bool,
    /// Cache size for search results
    pub cache_size: usize,
    /// Enable query optimization
    pub enable_optimization: bool,
    /// Enable result explanations
    pub enable_explanations: bool,
    /// Performance monitoring
    pub enable_monitoring: bool,
}

impl Default for VectorServiceConfig {
    fn default() -> Self {
        Self {
            service_uri: "http://oxirs.org/vec/".to_string(),
            default_threshold: 0.7,
            default_limit: 10,
            default_metric: SimilarityMetric::Cosine,
            enable_caching: true,
            cache_size: 1000,
            enable_optimization: true,
            enable_explanations: false,
            enable_monitoring: false,
        }
    }
}

/// Vector service function registry
#[derive(Debug, Clone)]
pub struct VectorServiceFunction {
    pub name: String,
    pub arity: usize,
    pub description: String,
    pub parameters: Vec<VectorServiceParameter>,
}

#[derive(Debug, Clone)]
pub struct VectorServiceParameter {
    pub name: String,
    pub param_type: VectorParameterType,
    pub required: bool,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum VectorParameterType {
    IRI,
    Literal,
    Vector,
    Number,
    String,
}

/// Vector query optimizer for performance enhancement
#[derive(Debug, Clone)]
pub struct VectorQueryOptimizer {
    pub enable_caching: bool,
    pub enable_parallel_execution: bool,
    pub enable_index_selection: bool,
    pub cost_model: CostModel,
}

#[derive(Debug, Clone)]
pub struct CostModel {
    pub linear_search_cost: f32,
    pub index_search_cost: f32,
    pub embedding_generation_cost: f32,
    pub cache_lookup_cost: f32,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            linear_search_cost: 1.0,
            index_search_cost: 0.1,
            embedding_generation_cost: 10.0,
            cache_lookup_cost: 0.01,
        }
    }
}

impl Default for VectorQueryOptimizer {
    fn default() -> Self {
        Self {
            enable_caching: true,
            enable_parallel_execution: true,
            enable_index_selection: true,
            cost_model: CostModel::default(),
        }
    }
}

/// Vector service argument types
#[derive(Debug, Clone)]
pub enum VectorServiceArg {
    IRI(String),
    Literal(String),
    Number(f32),
    Vector(crate::Vector),
    String(String),
}

/// Vector service result types
#[derive(Debug, Clone)]
pub enum VectorServiceResult {
    Number(f32),
    String(String),
    Vector(crate::Vector),
    Boolean(bool),
    SimilarityList(Vec<(String, f32)>),
    DetailedSimilarityList(Vec<(String, f32, std::collections::HashMap<String, String>)>),
    Clusters(Vec<Vec<String>>),
}

/// Vector query representation
#[derive(Debug, Clone)]
pub struct VectorQuery {
    pub operation_type: String,
    pub args: Vec<VectorServiceArg>,
    pub metadata: std::collections::HashMap<String, String>,
    pub estimated_result_size: Option<usize>,
    pub preferred_index: Option<String>,
    pub use_cache: bool,
    pub parallel_execution: bool,
    pub timeout: Option<std::time::Duration>,
}

impl VectorQuery {
    pub fn new(operation_type: String, args: Vec<VectorServiceArg>) -> Self {
        Self {
            operation_type,
            args,
            metadata: std::collections::HashMap::new(),
            estimated_result_size: None,
            preferred_index: None,
            use_cache: false,
            parallel_execution: false,
            timeout: None,
        }
    }

    pub fn can_parallelize(&self) -> bool {
        matches!(
            self.operation_type.as_str(),
            "similarity_search" | "batch_search" | "cluster_search"
        )
    }

    pub fn cache_key(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.operation_type.hash(&mut hasher);

        // Simple hash of arguments (simplified for this implementation)
        for arg in &self.args {
            match arg {
                VectorServiceArg::IRI(s)
                | VectorServiceArg::Literal(s)
                | VectorServiceArg::String(s) => {
                    s.hash(&mut hasher);
                }
                VectorServiceArg::Number(n) => {
                    n.to_bits().hash(&mut hasher);
                }
                VectorServiceArg::Vector(v) => {
                    v.len().hash(&mut hasher);
                }
            }
        }

        format!("query_{:x}", hasher.finish())
    }
}

/// Vector operation enum for SPARQL queries
#[derive(Debug, Clone)]
pub enum VectorOperation {
    /// Find similar resources to a given resource
    FindSimilar {
        resource: String,
        limit: Option<usize>,
        threshold: Option<f32>,
    },
    /// Calculate similarity between two resources
    CalculateSimilarity {
        resource1: String,
        resource2: String,
    },
    /// Search for resources similar to given text
    SearchText {
        query: String,
        limit: Option<usize>,
        threshold: Option<f32>,
    },
    /// Embed text into a vector
    EmbedText { text: String },
    /// Calculate similarity between two vectors
    VectorSimilarity {
        vector1: crate::Vector,
        vector2: crate::Vector,
    },
    /// Find nearest neighbors for a given vector
    VectorKNN {
        query_vector: crate::Vector,
        k: usize,
        threshold: Option<f32>,
    },
}

impl VectorOperation {
    /// Generate a SPARQL SERVICE query for this operation
    pub fn to_sparql_service_query(&self, service_uri: &str) -> String {
        match self {
            VectorOperation::FindSimilar {
                resource,
                limit,
                threshold,
            } => {
                let limit_clause = limit.map(|l| format!("LIMIT {}", l)).unwrap_or_default();
                let threshold_param = threshold.unwrap_or(0.0);
                format!(
                    r#"
                    SELECT ?resource ?similarity WHERE {{
                        SERVICE <{}> {{
                            SELECT ?resource ?similarity WHERE {{
                                ?resource vec:similar <{}> .
                                ?resource vec:similarity ?similarity .
                                FILTER(?similarity >= {})
                            }}
                            ORDER BY DESC(?similarity)
                            {}
                        }}
                    }}
                    "#,
                    service_uri, resource, threshold_param, limit_clause
                )
            }
            VectorOperation::CalculateSimilarity {
                resource1,
                resource2,
            } => {
                format!(
                    r#"
                    SELECT ?similarity WHERE {{
                        SERVICE <{}> {{
                            SELECT ?similarity WHERE {{
                                BIND(vec:similarity(<{}>, <{}>) AS ?similarity)
                            }}
                        }}
                    }}
                    "#,
                    service_uri, resource1, resource2
                )
            }
            VectorOperation::SearchText {
                query,
                limit,
                threshold,
            } => {
                let limit_clause = limit.map(|l| format!("LIMIT {}", l)).unwrap_or_default();
                let threshold_param = threshold.unwrap_or(0.0);
                format!(
                    r#"
                    SELECT ?resource ?similarity WHERE {{
                        SERVICE <{}> {{
                            SELECT ?resource ?similarity WHERE {{
                                ?resource vec:searchText "{}" .
                                ?resource vec:similarity ?similarity .
                                FILTER(?similarity >= {})
                            }}
                            ORDER BY DESC(?similarity)
                            {}
                        }}
                    }}
                    "#,
                    service_uri, query, threshold_param, limit_clause
                )
            }
            VectorOperation::EmbedText { text } => {
                format!(
                    r#"
                    SELECT ?vector WHERE {{
                        SERVICE <{}> {{
                            SELECT ?vector WHERE {{
                                BIND(vec:embedText("{}") AS ?vector)
                            }}
                        }}
                    }}
                    "#,
                    service_uri, text
                )
            }
            VectorOperation::VectorSimilarity { .. } => {
                format!(
                    r#"
                    SELECT ?similarity WHERE {{
                        SERVICE <{}> {{
                            SELECT ?similarity WHERE {{
                                BIND(vec:vectorSimilarity(?vector1, ?vector2) AS ?similarity)
                            }}
                        }}
                    }}
                    "#,
                    service_uri
                )
            }
            VectorOperation::VectorKNN { k, threshold, .. } => {
                let threshold_param = threshold.unwrap_or(0.0);
                format!(
                    r#"
                    SELECT ?resource ?similarity WHERE {{
                        SERVICE <{}> {{
                            SELECT ?resource ?similarity WHERE {{
                                ?resource vec:knn ?queryVector .
                                ?resource vec:similarity ?similarity .
                                FILTER(?similarity >= {})
                            }}
                            ORDER BY DESC(?similarity)
                            LIMIT {}
                        }}
                    }}
                    "#,
                    service_uri, threshold_param, k
                )
            }
        }
    }
}

/// Vector query result
#[derive(Debug, Clone)]
pub struct VectorQueryResult {
    pub results: Vec<(String, f32)>,
    pub metadata: std::collections::HashMap<String, String>,
    pub execution_time: std::time::Duration,
    pub from_cache: bool,
}

impl VectorQueryResult {
    pub fn new(results: Vec<(String, f32)>, execution_time: std::time::Duration) -> Self {
        Self {
            results,
            metadata: std::collections::HashMap::new(),
            execution_time,
            from_cache: false,
        }
    }

    pub fn with_metadata(mut self, metadata: std::collections::HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn from_cache(mut self) -> Self {
        self.from_cache = true;
        self
    }
}
