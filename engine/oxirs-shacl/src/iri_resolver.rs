//! IRI Resolution and Validation Module
//!
//! Provides comprehensive IRI expansion, validation, and resolution capabilities
//! for SHACL validation operations.

use oxirs_core::model::{NamedNode, Term};
use std::collections::HashMap;
use url::Url;

/// Error types for IRI resolution
#[derive(Debug, thiserror::Error)]
pub enum IriResolutionError {
    #[error("Invalid IRI: {0}")]
    InvalidIri(String),

    #[error("Unknown prefix: {0}")]
    UnknownPrefix(String),

    #[error("Invalid relative IRI: {0}")]
    InvalidRelativeIri(String),

    #[error("URL parsing error: {0}")]
    UrlParsing(#[from] url::ParseError),
}

/// IRI resolver with namespace and base IRI support
#[derive(Debug, Clone)]
pub struct IriResolver {
    /// Namespace prefix mappings
    prefixes: HashMap<String, String>,
    /// Base IRI for resolving relative IRIs
    base_iri: Option<String>,
    /// Default namespaces
    default_namespaces: HashMap<String, String>,
}

impl IriResolver {
    /// Create a new IRI resolver
    pub fn new() -> Self {
        let mut resolver = Self {
            prefixes: HashMap::new(),
            base_iri: None,
            default_namespaces: HashMap::new(),
        };

        resolver.add_default_namespaces();
        resolver
    }

    /// Create resolver with base IRI
    pub fn with_base_iri(base_iri: &str) -> Result<Self, IriResolutionError> {
        let mut resolver = Self::new();
        resolver.set_base_iri(base_iri)?;
        Ok(resolver)
    }

    /// Add default namespace prefixes
    fn add_default_namespaces(&mut self) {
        self.default_namespaces.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        self.default_namespaces.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        self.default_namespaces.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        self.default_namespaces.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        self.default_namespaces
            .insert("sh".to_string(), "http://www.w3.org/ns/shacl#".to_string());
        self.default_namespaces
            .insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        self.default_namespaces.insert(
            "dc".to_string(),
            "http://purl.org/dc/elements/1.1/".to_string(),
        );
        self.default_namespaces.insert(
            "dcterms".to_string(),
            "http://purl.org/dc/terms/".to_string(),
        );
        self.default_namespaces.insert(
            "skos".to_string(),
            "http://www.w3.org/2004/02/skos/core#".to_string(),
        );
    }

    /// Set the base IRI for resolving relative IRIs
    pub fn set_base_iri(&mut self, base_iri: &str) -> Result<(), IriResolutionError> {
        // Validate the base IRI
        Url::parse(base_iri)?;
        self.base_iri = Some(base_iri.to_string());
        Ok(())
    }

    /// Add a namespace prefix mapping
    pub fn add_prefix(&mut self, prefix: &str, namespace: &str) -> Result<(), IriResolutionError> {
        // Validate the namespace IRI
        Url::parse(namespace)?;
        self.prefixes
            .insert(prefix.to_string(), namespace.to_string());
        Ok(())
    }

    /// Remove a namespace prefix mapping
    pub fn remove_prefix(&mut self, prefix: &str) {
        self.prefixes.remove(prefix);
    }

    /// Get namespace for a prefix
    pub fn get_namespace(&self, prefix: &str) -> Option<&str> {
        self.prefixes
            .get(prefix)
            .or_else(|| self.default_namespaces.get(prefix))
            .map(|s| s.as_str())
    }

    /// Expand a prefixed name to a full IRI
    pub fn expand_prefixed_name(&self, prefixed_name: &str) -> Result<String, IriResolutionError> {
        if let Some(colon_pos) = prefixed_name.find(':') {
            let prefix = &prefixed_name[..colon_pos];
            let local_name = &prefixed_name[colon_pos + 1..];

            if let Some(namespace) = self.get_namespace(prefix) {
                Ok(format!("{}{}", namespace, local_name))
            } else {
                Err(IriResolutionError::UnknownPrefix(prefix.to_string()))
            }
        } else {
            // No prefix, treat as local name with default namespace if available
            if let Some(base) = &self.base_iri {
                self.resolve_relative_iri(prefixed_name, base)
            } else {
                Err(IriResolutionError::InvalidIri(format!(
                    "No prefix found and no base IRI set: {}",
                    prefixed_name
                )))
            }
        }
    }

    /// Resolve a relative IRI against a base IRI
    pub fn resolve_relative_iri(
        &self,
        relative_iri: &str,
        base_iri: &str,
    ) -> Result<String, IriResolutionError> {
        let base_url = Url::parse(base_iri)?;
        let resolved_url = base_url.join(relative_iri)?;
        Ok(resolved_url.to_string())
    }

    /// Resolve an IRI string to a full IRI
    pub fn resolve_iri(&self, iri: &str) -> Result<String, IriResolutionError> {
        // Check if it's already a full IRI
        if iri.starts_with("http://") || iri.starts_with("https://") || iri.starts_with("urn:") {
            // Validate the IRI
            Url::parse(iri)?;
            Ok(iri.to_string())
        }
        // Check if it's a prefixed name
        else if iri.contains(':') && !iri.starts_with('/') {
            self.expand_prefixed_name(iri)
        }
        // Treat as relative IRI
        else if let Some(base) = &self.base_iri {
            self.resolve_relative_iri(iri, base)
        } else {
            Err(IriResolutionError::InvalidRelativeIri(format!(
                "No base IRI available for relative IRI: {}",
                iri
            )))
        }
    }

    /// Resolve a term to a named node if it's an IRI
    pub fn resolve_term(&self, term: &Term) -> Result<Option<NamedNode>, IriResolutionError> {
        match term {
            Term::NamedNode(named_node) => {
                // Validate existing named node
                let iri_str = named_node.as_str();
                let resolved_iri = self.resolve_iri(iri_str)?;
                Ok(Some(NamedNode::new(resolved_iri).map_err(|e| {
                    IriResolutionError::InvalidIri(format!("Failed to create named node: {}", e))
                })?))
            }
            _ => Ok(None),
        }
    }

    /// Validate an IRI string
    pub fn validate_iri(&self, iri: &str) -> Result<(), IriResolutionError> {
        self.resolve_iri(iri).map(|_| ())
    }

    /// Check if an IRI is absolute
    pub fn is_absolute_iri(&self, iri: &str) -> bool {
        iri.starts_with("http://") || iri.starts_with("https://") || iri.starts_with("urn:")
    }

    /// Check if a string is a prefixed name
    pub fn is_prefixed_name(&self, name: &str) -> bool {
        if let Some(colon_pos) = name.find(':') {
            let prefix = &name[..colon_pos];
            self.get_namespace(prefix).is_some()
        } else {
            false
        }
    }

    /// Get all defined prefixes
    pub fn get_all_prefixes(&self) -> HashMap<String, String> {
        let mut all_prefixes = self.default_namespaces.clone();
        all_prefixes.extend(self.prefixes.clone());
        all_prefixes
    }

    /// Compact an IRI to a prefixed name if possible
    pub fn compact_iri(&self, iri: &str) -> String {
        // Try to find a matching namespace
        for (prefix, namespace) in self.get_all_prefixes() {
            if iri.starts_with(&namespace) {
                let local_name = &iri[namespace.len()..];
                // Check if local name is valid (no spaces, special characters)
                if self.is_valid_local_name(local_name) {
                    return format!("{}:{}", prefix, local_name);
                }
            }
        }

        // Return original IRI if no compaction possible
        iri.to_string()
    }

    /// Check if a local name is valid for compaction
    fn is_valid_local_name(&self, local_name: &str) -> bool {
        !local_name.is_empty()
            && !local_name.contains(' ')
            && !local_name.contains('\t')
            && !local_name.contains('\n')
            && !local_name.contains('\r')
    }

    /// Create a named node from an IRI string with resolution
    pub fn create_named_node(&self, iri: &str) -> Result<NamedNode, IriResolutionError> {
        let resolved_iri = self.resolve_iri(iri)?;
        NamedNode::new(resolved_iri).map_err(|e| {
            IriResolutionError::InvalidIri(format!("Failed to create named node: {}", e))
        })
    }

    /// Bulk resolve a set of IRIs
    pub fn resolve_iris(&self, iris: &[&str]) -> Result<Vec<String>, IriResolutionError> {
        iris.iter().map(|iri| self.resolve_iri(iri)).collect()
    }

    /// Load namespace prefixes from a map
    pub fn load_prefixes(
        &mut self,
        prefixes: HashMap<String, String>,
    ) -> Result<(), IriResolutionError> {
        for (prefix, namespace) in prefixes {
            self.add_prefix(&prefix, &namespace)?;
        }
        Ok(())
    }

    /// Clear all custom prefixes (keeps default namespaces)
    pub fn clear_custom_prefixes(&mut self) {
        self.prefixes.clear();
    }

    /// Reset to default state
    pub fn reset(&mut self) {
        self.prefixes.clear();
        self.base_iri = None;
    }
}

impl Default for IriResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// IRI validation utilities
pub struct IriValidator;

impl IriValidator {
    /// Validate an IRI according to RFC 3987
    pub fn validate_iri_syntax(iri: &str) -> Result<(), IriResolutionError> {
        // Basic validation using URL parser
        Url::parse(iri)?;
        Ok(())
    }

    /// Check if an IRI is a valid HTTP(S) IRI
    pub fn is_http_iri(iri: &str) -> bool {
        iri.starts_with("http://") || iri.starts_with("https://")
    }

    /// Check if an IRI is a valid URN
    pub fn is_urn(iri: &str) -> bool {
        iri.starts_with("urn:")
    }

    /// Extract the scheme from an IRI
    pub fn extract_scheme(iri: &str) -> Option<String> {
        if let Ok(url) = Url::parse(iri) {
            Some(url.scheme().to_string())
        } else {
            None
        }
    }

    /// Extract the host from an HTTP(S) IRI
    pub fn extract_host(iri: &str) -> Option<String> {
        if let Ok(url) = Url::parse(iri) {
            url.host_str().map(|s| s.to_string())
        } else {
            None
        }
    }

    /// Check if an IRI is dereferenceable (HTTP/HTTPS)
    pub fn is_dereferenceable(iri: &str) -> bool {
        Self::is_http_iri(iri)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iri_resolver_creation() {
        let resolver = IriResolver::new();
        assert!(resolver.get_namespace("sh").is_some());
        assert_eq!(
            resolver.get_namespace("sh").unwrap(),
            "http://www.w3.org/ns/shacl#"
        );
    }

    #[test]
    fn test_base_iri_setting() {
        let mut resolver = IriResolver::new();
        assert!(resolver.set_base_iri("http://example.org/").is_ok());
        assert!(resolver.set_base_iri("invalid-iri").is_err());
    }

    #[test]
    fn test_prefix_expansion() {
        let resolver = IriResolver::new();

        let expanded = resolver.expand_prefixed_name("sh:Shape").unwrap();
        assert_eq!(expanded, "http://www.w3.org/ns/shacl#Shape");

        let expanded = resolver.expand_prefixed_name("rdf:type").unwrap();
        assert_eq!(expanded, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    }

    #[test]
    fn test_custom_prefix() {
        let mut resolver = IriResolver::new();
        resolver.add_prefix("ex", "http://example.org/").unwrap();

        let expanded = resolver.expand_prefixed_name("ex:Person").unwrap();
        assert_eq!(expanded, "http://example.org/Person");
    }

    #[test]
    fn test_relative_iri_resolution() {
        let mut resolver = IriResolver::new();
        resolver.set_base_iri("http://example.org/base/").unwrap();

        let resolved = resolver
            .resolve_relative_iri("relative", "http://example.org/base/")
            .unwrap();
        assert_eq!(resolved, "http://example.org/base/relative");

        let resolved = resolver
            .resolve_relative_iri("../other", "http://example.org/base/")
            .unwrap();
        assert_eq!(resolved, "http://example.org/other");
    }

    #[test]
    fn test_iri_resolution() {
        let mut resolver = IriResolver::new();
        resolver.set_base_iri("http://example.org/").unwrap();

        // Absolute IRI
        let resolved = resolver.resolve_iri("http://example.org/test").unwrap();
        assert_eq!(resolved, "http://example.org/test");

        // Prefixed name
        let resolved = resolver.resolve_iri("sh:Shape").unwrap();
        assert_eq!(resolved, "http://www.w3.org/ns/shacl#Shape");

        // Relative IRI
        let resolved = resolver.resolve_iri("relative").unwrap();
        assert_eq!(resolved, "http://example.org/relative");
    }

    #[test]
    fn test_iri_compaction() {
        let resolver = IriResolver::new();

        let compacted = resolver.compact_iri("http://www.w3.org/ns/shacl#Shape");
        assert_eq!(compacted, "sh:Shape");

        let compacted = resolver.compact_iri("http://example.org/unknown");
        assert_eq!(compacted, "http://example.org/unknown");
    }

    #[test]
    fn test_iri_validation() {
        assert!(IriValidator::validate_iri_syntax("http://example.org/").is_ok());
        assert!(IriValidator::validate_iri_syntax("https://example.org/path?query=value").is_ok());
        assert!(
            IriValidator::validate_iri_syntax("urn:uuid:12345678-1234-5678-1234-567812345678")
                .is_ok()
        );
        assert!(IriValidator::validate_iri_syntax("invalid-iri").is_err());
    }

    #[test]
    fn test_iri_type_checking() {
        assert!(IriValidator::is_http_iri("http://example.org/"));
        assert!(IriValidator::is_http_iri("https://example.org/"));
        assert!(!IriValidator::is_http_iri("urn:example"));

        assert!(IriValidator::is_urn(
            "urn:uuid:12345678-1234-5678-1234-567812345678"
        ));
        assert!(!IriValidator::is_urn("http://example.org/"));
    }

    #[test]
    fn test_scheme_and_host_extraction() {
        assert_eq!(
            IriValidator::extract_scheme("http://example.org/path"),
            Some("http".to_string())
        );
        assert_eq!(
            IriValidator::extract_scheme("https://example.org/path"),
            Some("https".to_string())
        );
        assert_eq!(
            IriValidator::extract_scheme("urn:example"),
            Some("urn".to_string())
        );

        assert_eq!(
            IriValidator::extract_host("http://example.org/path"),
            Some("example.org".to_string())
        );
        assert_eq!(IriValidator::extract_host("urn:example"), None);
    }

    #[test]
    fn test_bulk_resolution() {
        let resolver = IriResolver::new();
        let iris = vec!["sh:Shape", "rdf:type", "rdfs:Class"];

        let resolved = resolver.resolve_iris(&iris).unwrap();
        assert_eq!(resolved.len(), 3);
        assert_eq!(resolved[0], "http://www.w3.org/ns/shacl#Shape");
        assert_eq!(
            resolved[1],
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        );
        assert_eq!(resolved[2], "http://www.w3.org/2000/01/rdf-schema#Class");
    }
}
