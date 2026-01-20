//! Lineage Query Builder
//!
//! SPARQL query builder for provenance queries

/// Lineage Query Builder
pub struct LineageQueryBuilder;

impl LineageQueryBuilder {
    /// Build SPARQL query for lineage chain
    pub fn build_lineage_query(entity_uri: &str) -> String {
        format!(
            r#"PREFIX prov: <http://www.w3.org/ns/prov#>
SELECT ?derived ?activity ?agent
WHERE {{
  <{entity}> prov:wasDerivedFrom* ?derived .
  OPTIONAL {{ ?derived prov:wasGeneratedBy ?activity }}
  OPTIONAL {{ ?derived prov:wasAttributedTo ?agent }}
}}
"#,
            entity = entity_uri
        )
    }
}
