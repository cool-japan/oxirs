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
        if local_name.is_empty() {
            return false;
        }
        
        // Check for forbidden characters in local names
        let forbidden_chars = [' ', '\t', '\n', '\r', '<', '>', '"', '{', '}', '|', '^', '`', '\\'];
        for &ch in &forbidden_chars {
            if local_name.contains(ch) {
                return false;
            }
        }
        
        // Check for percent-encoded sequences
        if local_name.contains('%') {
            if !self.is_valid_percent_encoding(local_name) {
                return false;
            }
        }
        
        true
    }
    
    /// Validate percent-encoded sequences in local names
    fn is_valid_percent_encoding(&self, s: &str) -> bool {
        let chars: Vec<char> = s.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            if chars[i] == '%' {
                // Need at least 2 more characters for valid percent encoding
                if i + 2 >= chars.len() {
                    return false;
                }
                
                // Check if next two characters are valid hex digits
                let hex1 = chars[i + 1];
                let hex2 = chars[i + 2];
                
                if !hex1.is_ascii_hexdigit() || !hex2.is_ascii_hexdigit() {
                    return false;
                }
                
                i += 3;
            } else {
                i += 1;
            }
        }
        
        true
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
    
    /// Normalize an IRI by removing unnecessary components
    pub fn normalize_iri(&self, iri: &str) -> Result<String, IriResolutionError> {
        let resolved_iri = self.resolve_iri(iri)?;
        let url = Url::parse(&resolved_iri)?;
        
        // Normalize the URL
        let mut normalized = url.clone();
        
        // Remove default port numbers
        match (url.scheme(), url.port()) {
            ("http", Some(80)) | ("https", Some(443)) => {
                normalized.set_port(None).map_err(|_| {
                    IriResolutionError::InvalidIri("Failed to normalize port".to_string())
                })?;
            }
            _ => {}
        }
        
        // Remove fragment if empty
        if url.fragment() == Some("") {
            normalized.set_fragment(None);
        }
        
        // Normalize path (remove redundant . and .. segments)
        let path = normalized.path();
        if path.contains("./") || path.contains("../") {
            let normalized_path = self.normalize_path(path);
            normalized.set_path(&normalized_path);
        }
        
        Ok(normalized.to_string())
    }
    
    /// Normalize a URL path by resolving . and .. segments
    fn normalize_path(&self, path: &str) -> String {
        let segments: Vec<&str> = path.split('/').collect();
        let mut normalized = Vec::new();
        
        for segment in segments {
            match segment {
                "." => continue, // Skip current directory references
                ".." => {
                    // Go up one level if possible
                    if !normalized.is_empty() && normalized.last() != Some(&"..") {
                        normalized.pop();
                    } else if !path.starts_with('/') {
                        // Only add .. for relative paths
                        normalized.push("..");
                    }
                }
                _ => normalized.push(segment),
            }
        }
        
        // Reconstruct path
        let result = normalized.join("/");
        if path.starts_with('/') && !result.starts_with('/') {
            format!("/{}", result)
        } else if path.ends_with('/') && !result.ends_with('/') && !result.is_empty() {
            format!("{}/", result)
        } else {
            result
        }
    }
    
    /// Validate namespace prefix according to XML namespace rules
    pub fn validate_prefix(&self, prefix: &str) -> Result<(), IriResolutionError> {
        if prefix.is_empty() {
            return Err(IriResolutionError::InvalidIri(
                "Prefix cannot be empty".to_string(),
            ));
        }
        
        // Check for valid XML NCName
        if !self.is_valid_ncname(prefix) {
            return Err(IriResolutionError::InvalidIri(format!(
                "Invalid prefix '{}': must be a valid XML NCName",
                prefix
            )));
        }
        
        Ok(())
    }
    
    /// Check if a string is a valid XML NCName (no colon)
    fn is_valid_ncname(&self, name: &str) -> bool {
        if name.is_empty() {
            return false;
        }
        
        // Must start with letter or underscore
        let first_char = name.chars().next().unwrap();
        if !first_char.is_alphabetic() && first_char != '_' {
            return false;
        }
        
        // Rest must be letters, digits, underscores, hyphens, or dots
        for ch in name.chars().skip(1) {
            if !ch.is_alphanumeric() && ch != '_' && ch != '-' && ch != '.' {
                return false;
            }
        }
        
        // Must not contain colon (that's what makes it NCName vs QName)
        !name.contains(':')
    }
    
    /// Get suggestions for common namespace prefixes
    pub fn suggest_prefix_for_namespace(&self, namespace: &str) -> Option<String> {
        // Common namespace to prefix mappings
        let suggestions = [
            ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf"),
            ("http://www.w3.org/2000/01/rdf-schema#", "rdfs"),
            ("http://www.w3.org/2002/07/owl#", "owl"),
            ("http://www.w3.org/2001/XMLSchema#", "xsd"),
            ("http://www.w3.org/ns/shacl#", "sh"),
            ("http://xmlns.com/foaf/0.1/", "foaf"),
            ("http://purl.org/dc/elements/1.1/", "dc"),
            ("http://purl.org/dc/terms/", "dcterms"),
            ("http://www.w3.org/2004/02/skos/core#", "skos"),
            ("http://www.w3.org/2006/vcard/ns#", "vcard"),
            ("http://www.w3.org/2003/01/geo/wgs84_pos#", "geo"),
            ("http://dbpedia.org/ontology/", "dbo"),
            ("http://dbpedia.org/resource/", "dbr"),
            ("http://www.wikidata.org/entity/", "wd"),
        ];
        
        for (ns, prefix) in &suggestions {
            if namespace == *ns {
                return Some(prefix.to_string());
            }
        }
        
        // Try to generate a reasonable prefix from the namespace
        if let Ok(url) = Url::parse(namespace) {
            if let Some(host) = url.host_str() {
                // Extract domain parts and create prefix
                let parts: Vec<&str> = host.split('.').collect();
                if parts.len() >= 2 {
                    let domain_part = parts[parts.len() - 2]; // e.g., "example" from "www.example.org"
                    if self.is_valid_ncname(domain_part) {
                        return Some(domain_part.to_string());
                    }
                }
            }
        }
        
        None
    }
    
    /// Export current prefix mappings for serialization
    pub fn export_prefix_mappings(&self) -> HashMap<String, String> {
        let mut mappings = HashMap::new();
        
        // Include default namespaces
        for (prefix, namespace) in &self.default_namespaces {
            mappings.insert(prefix.clone(), namespace.clone());
        }
        
        // Include custom prefixes (override defaults if necessary)
        for (prefix, namespace) in &self.prefixes {
            mappings.insert(prefix.clone(), namespace.clone());
        }
        
        mappings
    }
    
    /// Import prefix mappings from external source
    pub fn import_prefix_mappings(&mut self, mappings: &HashMap<String, String>) -> Result<Vec<String>, IriResolutionError> {
        let mut warnings = Vec::new();
        
        for (prefix, namespace) in mappings {
            // Validate prefix
            if let Err(e) = self.validate_prefix(prefix) {
                warnings.push(format!("Skipping invalid prefix '{}': {}", prefix, e));
                continue;
            }
            
            // Validate namespace
            if let Err(e) = Url::parse(namespace) {
                warnings.push(format!("Skipping invalid namespace '{}' for prefix '{}': {}", namespace, prefix, e));
                continue;
            }
            
            // Check for conflicts with default namespaces
            if let Some(existing_ns) = self.default_namespaces.get(prefix) {
                if existing_ns != namespace {
                    warnings.push(format!(
                        "Overriding default namespace for prefix '{}': '{}' -> '{}'",
                        prefix, existing_ns, namespace
                    ));
                }
            }
            
            self.prefixes.insert(prefix.clone(), namespace.clone());
        }
        
        Ok(warnings)
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

    #[test]
    fn test_iri_normalization() {
        let resolver = IriResolver::new();
        
        // Test port normalization
        assert_eq!(
            resolver.normalize_iri("http://example.org:80/path").unwrap(),
            "http://example.org/path"
        );
        assert_eq!(
            resolver.normalize_iri("https://example.org:443/path").unwrap(),
            "https://example.org/path"
        );
        
        // Test path normalization
        assert_eq!(
            resolver.normalize_iri("http://example.org/a/./b/../c").unwrap(),
            "http://example.org/a/c"
        );
    }

    #[test]
    fn test_path_normalization() {
        let resolver = IriResolver::new();
        
        assert_eq!(resolver.normalize_path("/a/./b/../c"), "/a/c");
        assert_eq!(resolver.normalize_path("a/./b/../c"), "a/c");
        assert_eq!(resolver.normalize_path("/a/b/c/.."), "/a/b");
        assert_eq!(resolver.normalize_path("../a/b"), "../a/b");
    }

    #[test]
    fn test_prefix_validation() {
        let resolver = IriResolver::new();
        
        // Valid prefixes
        assert!(resolver.validate_prefix("ex").is_ok());
        assert!(resolver.validate_prefix("example").is_ok());
        assert!(resolver.validate_prefix("_private").is_ok());
        assert!(resolver.validate_prefix("my-namespace").is_ok());
        
        // Invalid prefixes
        assert!(resolver.validate_prefix("").is_err());
        assert!(resolver.validate_prefix("123invalid").is_err());
        assert!(resolver.validate_prefix("has:colon").is_err());
        assert!(resolver.validate_prefix("has space").is_err());
    }

    #[test]
    fn test_ncname_validation() {
        let resolver = IriResolver::new();
        
        assert!(resolver.is_valid_ncname("validName"));
        assert!(resolver.is_valid_ncname("_underscore"));
        assert!(resolver.is_valid_ncname("with-hyphen"));
        assert!(resolver.is_valid_ncname("with.dot"));
        assert!(resolver.is_valid_ncname("name123"));
        
        assert!(!resolver.is_valid_ncname(""));
        assert!(!resolver.is_valid_ncname("123startsWithNumber"));
        assert!(!resolver.is_valid_ncname("has:colon"));
        assert!(!resolver.is_valid_ncname("has space"));
    }

    #[test]
    fn test_percent_encoding_validation() {
        let resolver = IriResolver::new();
        
        assert!(resolver.is_valid_percent_encoding("simple"));
        assert!(resolver.is_valid_percent_encoding("with%20space"));
        assert!(resolver.is_valid_percent_encoding("multiple%20%2B%3D"));
        
        assert!(!resolver.is_valid_percent_encoding("incomplete%2"));
        assert!(!resolver.is_valid_percent_encoding("invalid%GG"));
        assert!(!resolver.is_valid_percent_encoding("percent%"));
    }

    #[test]
    fn test_prefix_suggestions() {
        let resolver = IriResolver::new();
        
        assert_eq!(
            resolver.suggest_prefix_for_namespace("http://www.w3.org/ns/shacl#"),
            Some("sh".to_string())
        );
        
        assert_eq!(
            resolver.suggest_prefix_for_namespace("http://xmlns.com/foaf/0.1/"),
            Some("foaf".to_string())
        );
        
        // Test domain-based suggestion
        assert_eq!(
            resolver.suggest_prefix_for_namespace("http://example.org/ontology/"),
            Some("example".to_string())
        );
    }

    #[test]
    fn test_prefix_import_export() {
        let mut resolver = IriResolver::new();
        
        // Export current mappings
        let exported = resolver.export_prefix_mappings();
        assert!(exported.contains_key("sh"));
        assert!(exported.contains_key("rdf"));
        
        // Import new mappings
        let mut new_mappings = HashMap::new();
        new_mappings.insert("test".to_string(), "http://test.example.org/".to_string());
        new_mappings.insert("invalid".to_string(), "not-a-valid-url".to_string());
        
        let warnings = resolver.import_prefix_mappings(&new_mappings).unwrap();
        assert_eq!(warnings.len(), 1); // Should warn about invalid URL
        
        // Check that valid mapping was added
        assert_eq!(
            resolver.get_namespace("test"),
            Some("http://test.example.org/")
        );
    }
}
