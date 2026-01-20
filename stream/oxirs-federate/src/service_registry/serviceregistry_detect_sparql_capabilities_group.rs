//! # ServiceRegistry - detect_sparql_capabilities_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use super::types::{ServiceDescription, SparqlCapabilities, SparqlEndpoint, SparqlVersion};
use anyhow::Result;
use std::time::Duration;
use tracing::{debug, info};

impl ServiceRegistry {
    /// Detect SPARQL endpoint capabilities
    pub(super) async fn detect_sparql_capabilities(
        &self,
        endpoint: &SparqlEndpoint,
    ) -> Result<SparqlCapabilities> {
        debug!(
            "Detecting capabilities for SPARQL endpoint: {}",
            endpoint.url
        );
        let mut capabilities = SparqlCapabilities::default();
        let endpoint_url = endpoint.url.to_string();
        capabilities.sparql_version = self.detect_sparql_version(&endpoint_url).await?;
        capabilities.result_formats = self.detect_result_formats(&endpoint_url).await?;
        capabilities.graph_formats = self.detect_graph_formats(&endpoint_url).await?;
        capabilities.supports_update = self.test_update_support(&endpoint_url).await?;
        capabilities.supports_named_graphs = self.test_named_graph_support(&endpoint_url).await?;
        capabilities.supports_federation = self.test_federation_support(&endpoint_url).await?;
        capabilities.supports_full_text_search = self.test_fulltext_support(&endpoint_url).await?;
        capabilities.supports_geospatial = self.test_geospatial_support(&endpoint_url).await?;
        capabilities.supports_rdf_star = self.test_rdf_star_support(&endpoint_url).await?;
        capabilities.custom_functions = self.discover_custom_functions(&endpoint_url).await?;
        capabilities.service_description = self.fetch_service_description(&endpoint_url).await.ok();
        capabilities.max_query_complexity = self
            .estimate_query_complexity_limit(&endpoint_url)
            .await
            .ok();
        info!(
            "Capability detection completed for {}: {:?}",
            endpoint.url, capabilities
        );
        Ok(capabilities)
    }
    /// Detect SPARQL version by testing specific features
    async fn detect_sparql_version(&self, endpoint_url: &str) -> Result<SparqlVersion> {
        let sparql_12_query = "SELECT (IF(true, 'yes', 'no') AS ?test) WHERE {}";
        if self
            .test_sparql_query(endpoint_url, sparql_12_query)
            .await
            .is_ok()
        {
            return Ok(SparqlVersion::V12);
        }
        let sparql_11_query = "SELECT ?s WHERE { ?s ?p ?o . FILTER EXISTS { ?s ?p ?o } }";
        if self
            .test_sparql_query(endpoint_url, sparql_11_query)
            .await
            .is_ok()
        {
            return Ok(SparqlVersion::V11);
        }
        Ok(SparqlVersion::V10)
    }
    /// Test SPARQL UPDATE support
    async fn test_update_support(&self, endpoint_url: &str) -> Result<bool> {
        let update_query =
            "INSERT DATA { <http://example.org/test> <http://example.org/test> \"test\" }";
        let update_url = format!("{}update", endpoint_url.trim_end_matches('/'));
        let response = self
            .http_client
            .post(&update_url)
            .header("Content-Type", "application/sparql-update")
            .body(update_query)
            .timeout(Duration::from_secs(5))
            .send()
            .await;
        match response {
            Ok(resp) => Ok(resp.status().is_success() || resp.status().as_u16() == 400),
            Err(_) => Ok(false),
        }
    }
    /// Test named graph support
    async fn test_named_graph_support(&self, endpoint_url: &str) -> Result<bool> {
        let query = "SELECT ?g WHERE { GRAPH ?g { ?s ?p ?o } } LIMIT 1";
        Ok(self.test_sparql_query(endpoint_url, query).await.is_ok())
    }
    /// Test federation support (SERVICE clause)
    async fn test_federation_support(&self, endpoint_url: &str) -> Result<bool> {
        let query = "SELECT ?s WHERE { SERVICE <http://dbpedia.org/sparql> { ?s ?p ?o } } LIMIT 1";
        let result = self.test_sparql_query(endpoint_url, query).await;
        match result {
            Ok(_) => Ok(true),
            Err(e) => {
                let error_msg = e.to_string().to_lowercase();
                Ok(error_msg.contains("service")
                    && !error_msg.contains("syntax")
                    && !error_msg.contains("parse"))
            }
        }
    }
    /// Test full-text search capabilities
    async fn test_fulltext_support(&self, endpoint_url: &str) -> Result<bool> {
        let lucene_query = "SELECT ?s WHERE { ?s <http://jena.apache.org/text#query> \"test\" }";
        if self
            .test_sparql_query(endpoint_url, lucene_query)
            .await
            .is_ok()
        {
            return Ok(true);
        }
        let virtuoso_query = "SELECT ?s WHERE { ?s ?p ?o . ?o bif:contains \"test\" }";
        if self
            .test_sparql_query(endpoint_url, virtuoso_query)
            .await
            .is_ok()
        {
            return Ok(true);
        }
        Ok(false)
    }
    /// Test geospatial capabilities
    async fn test_geospatial_support(&self, endpoint_url: &str) -> Result<bool> {
        let geo_query = "SELECT ?s WHERE { ?s <http://www.opengis.net/ont/geosparql#asWKT> ?geo }";
        if self
            .test_sparql_query(endpoint_url, geo_query)
            .await
            .is_ok()
        {
            return Ok(true);
        }
        let virtuoso_geo =
            "SELECT ?s WHERE { ?s ?p ?o . FILTER(bif:st_within(?o, bif:st_point(0, 0), 10)) }";
        if self
            .test_sparql_query(endpoint_url, virtuoso_geo)
            .await
            .is_ok()
        {
            return Ok(true);
        }
        Ok(false)
    }
    /// Test RDF-star support
    async fn test_rdf_star_support(&self, endpoint_url: &str) -> Result<bool> {
        let rdf_star_query = "SELECT ?s WHERE { <<?s ?p ?o>> ?meta ?value }";
        Ok(self
            .test_sparql_query(endpoint_url, rdf_star_query)
            .await
            .is_ok())
    }
    /// Fetch service description if available
    async fn fetch_service_description(&self, endpoint_url: &str) -> Result<ServiceDescription> {
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
