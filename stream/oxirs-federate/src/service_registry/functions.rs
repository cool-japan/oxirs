//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::Deserialize;
use url::Url;

#[cfg(test)]
use super::serviceregistry_type::ServiceRegistry;
#[cfg(test)]
use super::types::{
    ConnectionConfig, PerformanceStats, RegistryConfig, SparqlCapabilities, SparqlEndpoint,
    SparqlVersion,
};
#[cfg(test)]
use chrono::Utc;
#[cfg(test)]
use std::collections::{HashMap, HashSet};
pub(super) fn serialize_url<S>(url: &Url, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(url.as_str())
}
pub(super) fn deserialize_url<'de, D>(deserializer: D) -> Result<Url, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Url::parse(&s).map_err(serde::de::Error::custom)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_service_registry_creation() {
        let config = RegistryConfig::default();
        let registry = ServiceRegistry::with_config(config);
        assert_eq!(registry.get_sparql_endpoints().len(), 0);
        assert_eq!(registry.get_graphql_services().len(), 0);
    }
    #[tokio::test]
    async fn test_sparql_endpoint_registration() {
        let config = RegistryConfig::default();
        let _registry = ServiceRegistry::with_config(config);
        let _endpoint = SparqlEndpoint {
            id: "test-endpoint".to_string(),
            name: "Test SPARQL Endpoint".to_string(),
            url: Url::parse("http://localhost:3030/test").expect("valid URL"),
            auth: None,
            capabilities: SparqlCapabilities {
                sparql_version: SparqlVersion::V11,
                result_formats: HashSet::new(),
                graph_formats: HashSet::new(),
                custom_functions: HashSet::new(),
                max_query_complexity: Some(1000),
                supports_federation: true,
                supports_update: false,
                supports_named_graphs: true,
                supports_full_text_search: false,
                supports_geospatial: false,
                supports_rdf_star: false,
                service_description: None,
            },
            statistics: PerformanceStats::default(),
            registered_at: Utc::now(),
            last_access: None,
            metadata: HashMap::new(),
            connection_config: ConnectionConfig::default(),
        };
    }
}
