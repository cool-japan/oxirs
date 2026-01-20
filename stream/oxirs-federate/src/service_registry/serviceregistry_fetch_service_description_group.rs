//! # ServiceRegistry - fetch_service_description_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, HashSet};
use super::serviceregistry_type::ServiceRegistry;

impl ServiceRegistry {
    /// Fetch service description if available
    async fn fetch_service_description(
        &self,
        endpoint_url: &str,
    ) -> Result<ServiceDescription> {
        let sd_query = r#"
            SELECT ?defaultGraph ?namedGraph ?language ?propertyFunction WHERE {
                OPTIONAL { ?service <http://www.w3.org/ns/sparql-service-description#defaultGraph> ?defaultGraph }
                OPTIONAL { ?service <http://www.w3.org/ns/sparql-service-description#namedGraph> ?namedGraph }
                OPTIONAL { ?service <http://www.w3.org/ns/sparql-service-description#languageExtension> ?language }
                OPTIONAL { ?service <http://www.w3.org/ns/sparql-service-description#propertyFeature> ?propertyFunction }
            }
        "#;
        let _response = self.test_sparql_query(endpoint_url, sd_query).await?;
        Ok(ServiceDescription {
            default_graphs: vec![],
            named_graphs: vec![],
            languages: vec!["SPARQL".to_string()],
            property_functions: vec![],
        })
    }
    /// Estimate query complexity limit by testing increasingly complex queries
    async fn estimate_query_complexity_limit(&self, endpoint_url: &str) -> Result<u32> {
        let base_query = "SELECT ?s WHERE { ?s ?p ?o ";
        let mut complexity = 10;
        for i in 1..=10 {
            let mut query = base_query.to_string();
            for j in 0..i * 10 {
                query.push_str(&format!(". ?s{j} ?p{j} ?o{j} "));
            }
            query.push_str("} LIMIT 1");
            if self.test_sparql_query(endpoint_url, &query).await.is_err() {
                break;
            }
            complexity = i * 100;
        }
        Ok(complexity)
    }
}
