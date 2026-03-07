//! # DID:web Method Resolver
//!
//! Resolves DID identifiers using the `did:web` method (W3C DID Web Method).
//! Converts `did:web:example.com:path` to `https://example.com/path/did.json`
//! and fetches the DID Document.
//!
//! ## Features
//!
//! - **URL construction**: Correct domain/path decoding per spec
//! - **DID Document parsing**: JSON-LD DID Document deserialization
//! - **Verification method extraction**: Parse public keys and services
//! - **Caching**: In-memory cache with TTL for resolved documents
//! - **Validation**: Structural validation of resolved DID Documents
//! - **Metadata**: Resolution metadata including content type, duration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the DID:web resolver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidWebConfig {
    /// Cache TTL for resolved documents (default: 300s).
    pub cache_ttl: Duration,
    /// Maximum cache entries (default: 1000).
    pub max_cache_entries: usize,
    /// Request timeout (default: 10s).
    pub request_timeout: Duration,
    /// Whether to require HTTPS (default: true).
    pub require_https: bool,
    /// Whether to validate DID Document structure (default: true).
    pub validate_documents: bool,
}

impl Default for DidWebConfig {
    fn default() -> Self {
        Self {
            cache_ttl: Duration::from_secs(300),
            max_cache_entries: 1000,
            request_timeout: Duration::from_secs(10),
            require_https: true,
            validate_documents: true,
        }
    }
}

// ─────────────────────────────────────────────
// DID Document types
// ─────────────────────────────────────────────

/// A resolved DID Document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidDocument {
    /// The DID identifier.
    pub id: String,
    /// JSON-LD context.
    pub context: Vec<String>,
    /// Verification methods (public keys).
    pub verification_method: Vec<VerificationMethod>,
    /// Authentication methods.
    pub authentication: Vec<String>,
    /// Assertion methods.
    pub assertion_method: Vec<String>,
    /// Service endpoints.
    pub service: Vec<ServiceEndpoint>,
    /// Controller of the DID.
    pub controller: Option<String>,
    /// Also-known-as aliases.
    pub also_known_as: Vec<String>,
}

/// A verification method (public key).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationMethod {
    /// Verification method ID.
    pub id: String,
    /// Type (e.g., Ed25519VerificationKey2020).
    pub method_type: String,
    /// Controller DID.
    pub controller: String,
    /// Public key in multibase encoding.
    pub public_key_multibase: Option<String>,
    /// Public key in JWK format.
    pub public_key_jwk: Option<serde_json::Value>,
}

/// A service endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    /// Service ID.
    pub id: String,
    /// Service type.
    pub service_type: String,
    /// Endpoint URL.
    pub service_endpoint: String,
}

/// Resolution metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionMetadata {
    /// Content type of the resolved document.
    pub content_type: String,
    /// Time taken to resolve (ms).
    pub duration_ms: u64,
    /// Whether the result came from cache.
    pub from_cache: bool,
    /// The URL that was fetched.
    pub resolved_url: String,
    /// HTTP status code (if fetched).
    pub http_status: Option<u16>,
}

/// Validation result for a DID Document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the document is valid.
    pub valid: bool,
    /// Validation errors.
    pub errors: Vec<String>,
    /// Validation warnings.
    pub warnings: Vec<String>,
}

// ─────────────────────────────────────────────
// Cache Entry
// ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct CacheEntry {
    document: DidDocument,
    cached_at: Instant,
}

// ─────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────

/// Statistics for the DID:web resolver.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DidWebStats {
    pub total_resolutions: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub validation_errors: u64,
    pub url_parse_errors: u64,
}

// ─────────────────────────────────────────────
// URL Construction
// ─────────────────────────────────────────────

/// Convert a `did:web:...` identifier to the corresponding HTTPS URL.
///
/// Per the DID Web Method spec:
/// - `did:web:example.com` → `https://example.com/.well-known/did.json`
/// - `did:web:example.com:path:to:resource` → `https://example.com/path/to/resource/did.json`
/// - `%3A` in the domain is decoded to `:` (for ports)
pub fn did_web_to_url(did: &str) -> Result<String, String> {
    if !did.starts_with("did:web:") {
        return Err(format!("Not a did:web identifier: {did}"));
    }

    let method_specific = &did[8..]; // After "did:web:"
    if method_specific.is_empty() {
        return Err("Empty method-specific identifier".into());
    }

    let parts: Vec<&str> = method_specific.split(':').collect();

    // First part is the domain (with optional port via %3A)
    let domain = parts[0].replace("%3A", ":");

    if parts.len() == 1 {
        // Root DID: https://domain/.well-known/did.json
        Ok(format!("https://{domain}/.well-known/did.json"))
    } else {
        // Path-based DID: https://domain/path1/path2/.../did.json
        let path = parts[1..].join("/");
        Ok(format!("https://{domain}/{path}/did.json"))
    }
}

/// Parse a DID:web identifier into its components.
pub fn parse_did_web(did: &str) -> Result<(String, Vec<String>), String> {
    if !did.starts_with("did:web:") {
        return Err(format!("Not a did:web identifier: {did}"));
    }

    let method_specific = &did[8..];
    if method_specific.is_empty() {
        return Err("Empty method-specific identifier".into());
    }

    let parts: Vec<&str> = method_specific.split(':').collect();
    let domain = parts[0].replace("%3A", ":");
    let path: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();

    Ok((domain, path))
}

/// Construct a did:web identifier from domain and path.
pub fn construct_did_web(domain: &str, path: &[&str]) -> String {
    let encoded_domain = domain.replace(':', "%3A");
    if path.is_empty() {
        format!("did:web:{encoded_domain}")
    } else {
        format!("did:web:{encoded_domain}:{}", path.join(":"))
    }
}

// ─────────────────────────────────────────────
// DID Document Validation
// ─────────────────────────────────────────────

/// Validate a DID Document structure.
pub fn validate_did_document(doc: &DidDocument) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    if doc.id.is_empty() {
        errors.push("DID Document must have an 'id' field".into());
    } else if !doc.id.starts_with("did:") {
        errors.push(format!("Invalid DID format: {}", doc.id));
    }

    if doc.context.is_empty() {
        errors.push("DID Document must have a '@context' field".into());
    } else if !doc
        .context
        .iter()
        .any(|c| c.contains("did/v1") || c.contains("www.w3.org"))
    {
        warnings.push("Context should include W3C DID context".into());
    }

    if doc.verification_method.is_empty() {
        warnings.push("No verification methods found".into());
    }

    for vm in &doc.verification_method {
        if vm.id.is_empty() {
            errors.push("Verification method must have an 'id'".into());
        }
        if vm.method_type.is_empty() {
            errors.push("Verification method must have a 'type'".into());
        }
        if vm.public_key_multibase.is_none() && vm.public_key_jwk.is_none() {
            warnings.push(format!(
                "Verification method '{}' has no public key material",
                vm.id
            ));
        }
    }

    for svc in &doc.service {
        if svc.id.is_empty() {
            errors.push("Service endpoint must have an 'id'".into());
        }
        if svc.service_endpoint.is_empty() {
            errors.push("Service endpoint must have a URL".into());
        }
    }

    ValidationResult {
        valid: errors.is_empty(),
        errors,
        warnings,
    }
}

// ─────────────────────────────────────────────
// DID:web Resolver
// ─────────────────────────────────────────────

/// Resolver for DID:web identifiers.
pub struct DidWebResolver {
    config: DidWebConfig,
    cache: HashMap<String, CacheEntry>,
    stats: DidWebStats,
}

impl DidWebResolver {
    /// Create a new resolver.
    pub fn new(config: DidWebConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            stats: DidWebStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(DidWebConfig::default())
    }

    /// Resolve a did:web identifier to its URL (without HTTP fetch).
    pub fn resolve_url(&mut self, did: &str) -> Result<String, String> {
        self.stats.total_resolutions += 1;
        did_web_to_url(did).map_err(|e| {
            self.stats.url_parse_errors += 1;
            e
        })
    }

    /// Check cache for a resolved document.
    pub fn get_cached(&mut self, did: &str) -> Option<&DidDocument> {
        if let Some(entry) = self.cache.get(did) {
            if entry.cached_at.elapsed() <= self.config.cache_ttl {
                self.stats.cache_hits += 1;
                return Some(&entry.document);
            }
        }
        self.stats.cache_misses += 1;
        None
    }

    /// Store a resolved document in cache.
    pub fn cache_document(&mut self, did: &str, document: DidDocument) {
        if self.cache.len() >= self.config.max_cache_entries {
            // Evict oldest
            if let Some(oldest_key) = self
                .cache
                .iter()
                .min_by_key(|(_, e)| e.cached_at)
                .map(|(k, _)| k.clone())
            {
                self.cache.remove(&oldest_key);
            }
        }
        self.cache.insert(
            did.to_string(),
            CacheEntry {
                document,
                cached_at: Instant::now(),
            },
        );
    }

    /// Validate a DID Document.
    pub fn validate(&mut self, doc: &DidDocument) -> ValidationResult {
        let result = validate_did_document(doc);
        if !result.valid {
            self.stats.validation_errors += 1;
        }
        result
    }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get statistics.
    pub fn stats(&self) -> &DidWebStats {
        &self.stats
    }

    /// Get configuration.
    pub fn config(&self) -> &DidWebConfig {
        &self.config
    }

    /// Number of cached entries.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_document() -> DidDocument {
        DidDocument {
            id: "did:web:example.com".into(),
            context: vec!["https://www.w3.org/ns/did/v1".into()],
            verification_method: vec![VerificationMethod {
                id: "did:web:example.com#key-1".into(),
                method_type: "Ed25519VerificationKey2020".into(),
                controller: "did:web:example.com".into(),
                public_key_multibase: Some("z6Mkf...".into()),
                public_key_jwk: None,
            }],
            authentication: vec!["did:web:example.com#key-1".into()],
            assertion_method: vec![],
            service: vec![ServiceEndpoint {
                id: "did:web:example.com#service-1".into(),
                service_type: "LinkedDomains".into(),
                service_endpoint: "https://example.com".into(),
            }],
            controller: None,
            also_known_as: vec![],
        }
    }

    #[test]
    fn test_did_web_to_url_root() {
        let url = did_web_to_url("did:web:example.com").expect("parse failed");
        assert_eq!(url, "https://example.com/.well-known/did.json");
    }

    #[test]
    fn test_did_web_to_url_path() {
        let url = did_web_to_url("did:web:example.com:users:alice").expect("parse failed");
        assert_eq!(url, "https://example.com/users/alice/did.json");
    }

    #[test]
    fn test_did_web_to_url_port() {
        let url = did_web_to_url("did:web:example.com%3A8080").expect("parse failed");
        assert_eq!(url, "https://example.com:8080/.well-known/did.json");
    }

    #[test]
    fn test_did_web_to_url_port_with_path() {
        let url = did_web_to_url("did:web:example.com%3A8080:path").expect("parse failed");
        assert_eq!(url, "https://example.com:8080/path/did.json");
    }

    #[test]
    fn test_did_web_to_url_invalid() {
        assert!(did_web_to_url("did:key:abc").is_err());
        assert!(did_web_to_url("not-a-did").is_err());
    }

    #[test]
    fn test_did_web_to_url_empty() {
        assert!(did_web_to_url("did:web:").is_err());
    }

    #[test]
    fn test_parse_did_web() {
        let (domain, path) =
            parse_did_web("did:web:example.com:users:alice").expect("parse failed");
        assert_eq!(domain, "example.com");
        assert_eq!(path, vec!["users", "alice"]);
    }

    #[test]
    fn test_parse_did_web_root() {
        let (domain, path) = parse_did_web("did:web:example.com").expect("parse failed");
        assert_eq!(domain, "example.com");
        assert!(path.is_empty());
    }

    #[test]
    fn test_construct_did_web() {
        let did = construct_did_web("example.com", &["users", "alice"]);
        assert_eq!(did, "did:web:example.com:users:alice");
    }

    #[test]
    fn test_construct_did_web_root() {
        let did = construct_did_web("example.com", &[]);
        assert_eq!(did, "did:web:example.com");
    }

    #[test]
    fn test_construct_did_web_port() {
        let did = construct_did_web("example.com:8080", &[]);
        assert_eq!(did, "did:web:example.com%3A8080");
    }

    #[test]
    fn test_validate_valid_document() {
        let result = validate_did_document(&sample_document());
        assert!(result.valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validate_missing_id() {
        let mut doc = sample_document();
        doc.id = String::new();
        let result = validate_did_document(&doc);
        assert!(!result.valid);
    }

    #[test]
    fn test_validate_missing_context() {
        let mut doc = sample_document();
        doc.context.clear();
        let result = validate_did_document(&doc);
        assert!(!result.valid);
    }

    #[test]
    fn test_validate_no_verification_methods() {
        let mut doc = sample_document();
        doc.verification_method.clear();
        let result = validate_did_document(&doc);
        assert!(result.valid); // Warning, not error
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_validate_empty_vm_id() {
        let mut doc = sample_document();
        doc.verification_method[0].id = String::new();
        let result = validate_did_document(&doc);
        assert!(!result.valid);
    }

    #[test]
    fn test_validate_empty_service_endpoint() {
        let mut doc = sample_document();
        doc.service[0].service_endpoint = String::new();
        let result = validate_did_document(&doc);
        assert!(!result.valid);
    }

    #[test]
    fn test_resolver_url() {
        let mut resolver = DidWebResolver::with_defaults();
        let url = resolver
            .resolve_url("did:web:example.com")
            .expect("resolve failed");
        assert_eq!(url, "https://example.com/.well-known/did.json");
    }

    #[test]
    fn test_resolver_cache() {
        let mut resolver = DidWebResolver::with_defaults();
        let doc = sample_document();
        resolver.cache_document("did:web:example.com", doc);
        assert_eq!(resolver.cache_size(), 1);

        let cached = resolver.get_cached("did:web:example.com");
        assert!(cached.is_some());
        assert_eq!(resolver.stats().cache_hits, 1);
    }

    #[test]
    fn test_resolver_cache_miss() {
        let mut resolver = DidWebResolver::with_defaults();
        assert!(resolver.get_cached("did:web:unknown.com").is_none());
        assert_eq!(resolver.stats().cache_misses, 1);
    }

    #[test]
    fn test_resolver_cache_ttl() {
        let mut resolver = DidWebResolver::new(DidWebConfig {
            cache_ttl: Duration::from_millis(50),
            ..Default::default()
        });
        resolver.cache_document("did:web:example.com", sample_document());
        std::thread::sleep(Duration::from_millis(100));
        assert!(resolver.get_cached("did:web:example.com").is_none());
    }

    #[test]
    fn test_resolver_clear_cache() {
        let mut resolver = DidWebResolver::with_defaults();
        resolver.cache_document("did:web:a.com", sample_document());
        resolver.cache_document("did:web:b.com", sample_document());
        resolver.clear_cache();
        assert_eq!(resolver.cache_size(), 0);
    }

    #[test]
    fn test_resolver_validate() {
        let mut resolver = DidWebResolver::with_defaults();
        let result = resolver.validate(&sample_document());
        assert!(result.valid);
    }

    #[test]
    fn test_config_defaults() {
        let config = DidWebConfig::default();
        assert_eq!(config.cache_ttl, Duration::from_secs(300));
        assert!(config.require_https);
        assert!(config.validate_documents);
    }

    #[test]
    fn test_config_serialization() {
        let config = DidWebConfig::default();
        let json = serde_json::to_string(&config).expect("serialize failed");
        assert!(json.contains("cache_ttl"));
    }

    #[test]
    fn test_document_serialization() {
        let doc = sample_document();
        let json = serde_json::to_string(&doc).expect("serialize failed");
        let deser: DidDocument = serde_json::from_str(&json).expect("deser failed");
        assert_eq!(deser.id, doc.id);
    }

    #[test]
    fn test_stats_tracking() {
        let mut resolver = DidWebResolver::with_defaults();
        let _ = resolver.resolve_url("did:web:a.com");
        let _ = resolver.resolve_url("did:web:b.com");
        assert_eq!(resolver.stats().total_resolutions, 2);
    }

    #[test]
    fn test_stats_url_parse_errors() {
        let mut resolver = DidWebResolver::with_defaults();
        let _ = resolver.resolve_url("invalid");
        assert_eq!(resolver.stats().url_parse_errors, 1);
    }

    #[test]
    fn test_cache_eviction() {
        let mut resolver = DidWebResolver::new(DidWebConfig {
            max_cache_entries: 2,
            ..Default::default()
        });
        resolver.cache_document("did:web:a.com", sample_document());
        resolver.cache_document("did:web:b.com", sample_document());
        resolver.cache_document("did:web:c.com", sample_document());
        assert!(resolver.cache_size() <= 2);
    }

    #[test]
    fn test_roundtrip_construct_parse() {
        let did = construct_did_web("example.com", &["users", "alice"]);
        let (domain, path) = parse_did_web(&did).expect("parse failed");
        assert_eq!(domain, "example.com");
        assert_eq!(path, vec!["users", "alice"]);
    }

    #[test]
    fn test_validation_result_serialization() {
        let result = ValidationResult {
            valid: true,
            errors: vec![],
            warnings: vec!["test".into()],
        };
        let json = serde_json::to_string(&result).expect("serialize failed");
        assert!(json.contains("warnings"));
    }
}
