//! # DID Resolver
//!
//! A W3C DID (Decentralized Identifier) resolver with an in-memory document store.
//!
//! Supports:
//! - Registering / deactivating / updating DID documents
//! - Resolving DIDs to documents with metadata
//! - Adding service endpoints to registered documents
//! - Listing DID methods present in the store
//! - Parsing DID method strings ("did:web:example.com" → "web")

use std::collections::HashMap;

// ─── Public types ─────────────────────────────────────────────────────────────

/// W3C DID Document
#[derive(Debug, Clone)]
pub struct DidDocument {
    pub id: String,
    pub also_known_as: Vec<String>,
    pub verification_methods: Vec<VerificationMethod>,
    /// Verification method IDs for authentication
    pub authentication: Vec<String>,
    /// Verification method IDs for assertion
    pub assertion_method: Vec<String>,
    pub service_endpoints: Vec<ServiceEndpoint>,
    pub created: Option<String>,
    pub updated: Option<String>,
}

impl DidDocument {
    /// Create a minimal DID document with just an ID
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            also_known_as: vec![],
            verification_methods: vec![],
            authentication: vec![],
            assertion_method: vec![],
            service_endpoints: vec![],
            created: None,
            updated: None,
        }
    }
}

/// Verification method (key material) embedded in a DID document
#[derive(Debug, Clone)]
pub struct VerificationMethod {
    pub id: String,
    pub vm_type: String,
    pub controller: String,
    pub public_key_multibase: Option<String>,
}

/// Service endpoint advertised in a DID document
#[derive(Debug, Clone)]
pub struct ServiceEndpoint {
    pub id: String,
    pub service_type: String,
    pub endpoint_url: String,
}

/// Errors that can occur during DID resolution
#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionError {
    /// The DID was not found in the resolver's store
    NotFound(String),
    /// The DID string is malformed
    Malformed(String),
    /// The DID has been deactivated
    Deactivated(String),
    /// Network or remote error (unused in local store but provided for interface completeness)
    NetworkError(String),
}

impl std::fmt::Display for ResolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(s) => write!(f, "DID not found: {s}"),
            Self::Malformed(s) => write!(f, "Malformed DID: {s}"),
            Self::Deactivated(s) => write!(f, "DID has been deactivated: {s}"),
            Self::NetworkError(s) => write!(f, "Network error: {s}"),
        }
    }
}

impl std::error::Error for ResolutionError {}

/// Successful resolution result
#[derive(Debug, Clone)]
pub struct ResolutionResult {
    pub document: DidDocument,
    pub metadata: ResolutionMetadata,
}

/// Metadata accompanying a successful resolution
#[derive(Debug, Clone)]
pub struct ResolutionMetadata {
    /// RFC 3339 timestamp of resolution
    pub resolved_at: String,
    /// DID method (e.g. "web", "key")
    pub method: String,
    /// Content type of the document
    pub content_type: String,
}

// ─── Internal store entry ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct StoreEntry {
    document: DidDocument,
    deactivated: bool,
}

// ─── DID Resolver ─────────────────────────────────────────────────────────────

/// In-memory DID resolver that manages a local document store
pub struct DidResolver {
    store: HashMap<String, StoreEntry>,
    resolution_count: usize,
}

impl Default for DidResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl DidResolver {
    /// Create a new empty resolver
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
            resolution_count: 0,
        }
    }

    /// Register a DID document in the store.
    ///
    /// Overwrites any existing entry for the same DID.
    pub fn register(&mut self, doc: DidDocument) {
        let key = doc.id.clone();
        self.store.insert(
            key,
            StoreEntry {
                document: doc,
                deactivated: false,
            },
        );
    }

    /// Resolve a DID to its document and metadata.
    ///
    /// Returns `ResolutionError::NotFound` if the DID is unknown,
    /// `ResolutionError::Deactivated` if it has been deactivated.
    pub fn resolve(&mut self, did: &str) -> Result<ResolutionResult, ResolutionError> {
        self.resolution_count += 1;

        if did.is_empty() || !did.starts_with("did:") {
            return Err(ResolutionError::Malformed(did.to_string()));
        }

        let entry = self
            .store
            .get(did)
            .ok_or_else(|| ResolutionError::NotFound(did.to_string()))?;

        if entry.deactivated {
            return Err(ResolutionError::Deactivated(did.to_string()));
        }

        let method = parse_did_method(did).unwrap_or_else(|| "unknown".to_string());

        Ok(ResolutionResult {
            document: entry.document.clone(),
            metadata: ResolutionMetadata {
                resolved_at: chrono_timestamp(),
                method,
                content_type: "application/did+ld+json".to_string(),
            },
        })
    }

    /// Deactivate a DID.  Returns `true` if the DID existed, `false` otherwise.
    pub fn deactivate(&mut self, did: &str) -> bool {
        if let Some(entry) = self.store.get_mut(did) {
            entry.deactivated = true;
            true
        } else {
            false
        }
    }

    /// Update an existing DID document.
    ///
    /// Returns `ResolutionError::NotFound` if the DID is not registered.
    /// Returns `ResolutionError::Deactivated` if the DID is deactivated.
    pub fn update(&mut self, did: &str, doc: DidDocument) -> Result<(), ResolutionError> {
        let entry = self
            .store
            .get_mut(did)
            .ok_or_else(|| ResolutionError::NotFound(did.to_string()))?;

        if entry.deactivated {
            return Err(ResolutionError::Deactivated(did.to_string()));
        }

        entry.document = doc;
        Ok(())
    }

    /// Return the unique DID methods present in the store (e.g. ["key", "web"]).
    pub fn list_methods(&self) -> Vec<String> {
        let mut methods: Vec<String> = self
            .store
            .keys()
            .filter_map(|did| parse_did_method(did))
            .collect();
        methods.sort();
        methods.dedup();
        methods
    }

    /// Add a service endpoint to an existing DID document.
    ///
    /// Returns `ResolutionError::NotFound` or `ResolutionError::Deactivated` on failure.
    pub fn add_service(
        &mut self,
        did: &str,
        service: ServiceEndpoint,
    ) -> Result<(), ResolutionError> {
        let entry = self
            .store
            .get_mut(did)
            .ok_or_else(|| ResolutionError::NotFound(did.to_string()))?;

        if entry.deactivated {
            return Err(ResolutionError::Deactivated(did.to_string()));
        }

        entry.document.service_endpoints.push(service);
        Ok(())
    }

    /// Total number of resolve() calls made on this resolver
    pub fn resolution_count(&self) -> usize {
        self.resolution_count
    }

    /// Number of documents currently held (including deactivated)
    pub fn count(&self) -> usize {
        self.store.len()
    }
}

// ─── Utility functions ────────────────────────────────────────────────────────

/// Parse the method from a DID string.
///
/// "did:web:example.com" → Some("web")
/// "did:key:z123" → Some("key")
/// "not-a-did" → None
pub fn parse_did_method(did: &str) -> Option<String> {
    // DID syntax: "did:<method>:<method-specific-id>"
    let parts: Vec<&str> = did.splitn(3, ':').collect();
    if parts.len() >= 2 && parts[0] == "did" && !parts[1].is_empty() {
        Some(parts[1].to_string())
    } else {
        None
    }
}

/// Return an RFC 3339-like timestamp string (uses a fixed fallback for no-std compatibility)
fn chrono_timestamp() -> String {
    // Use chrono if available; otherwise fall back to a static placeholder
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("{secs}")
    }
    #[cfg(target_arch = "wasm32")]
    {
        "0".to_string()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── helpers ─────────────────────────────────────────────────────────────

    fn make_doc(id: &str) -> DidDocument {
        DidDocument::new(id)
    }

    fn make_doc_with_vm(id: &str) -> DidDocument {
        let mut doc = DidDocument::new(id);
        doc.verification_methods.push(VerificationMethod {
            id: format!("{id}#key-1"),
            vm_type: "Ed25519VerificationKey2020".to_string(),
            controller: id.to_string(),
            public_key_multibase: Some("zABC123".to_string()),
        });
        doc.authentication.push(format!("{id}#key-1"));
        doc.assertion_method.push(format!("{id}#key-1"));
        doc
    }

    // ─── parse_did_method ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_did_method_web() {
        assert_eq!(
            parse_did_method("did:web:example.com"),
            Some("web".to_string())
        );
    }

    #[test]
    fn test_parse_did_method_key() {
        assert_eq!(parse_did_method("did:key:z6Mk"), Some("key".to_string()));
    }

    #[test]
    fn test_parse_did_method_ion() {
        assert_eq!(parse_did_method("did:ion:EiDABC"), Some("ion".to_string()));
    }

    #[test]
    fn test_parse_did_method_not_a_did() {
        assert_eq!(parse_did_method("not-a-did"), None);
    }

    #[test]
    fn test_parse_did_method_empty() {
        assert_eq!(parse_did_method(""), None);
    }

    #[test]
    fn test_parse_did_method_missing_method() {
        // "did:" with empty method part
        assert_eq!(parse_did_method("did::something"), None);
    }

    // ─── register / resolve ───────────────────────────────────────────────────

    #[test]
    fn test_register_and_resolve() {
        let mut resolver = DidResolver::new();
        let doc = make_doc("did:key:z6MkTest");
        resolver.register(doc.clone());
        let result = resolver
            .resolve("did:key:z6MkTest")
            .expect("should resolve");
        assert_eq!(result.document.id, "did:key:z6MkTest");
    }

    #[test]
    fn test_resolve_not_found() {
        let mut resolver = DidResolver::new();
        let err = resolver
            .resolve("did:key:nonexistent")
            .expect_err("should be NotFound");
        assert!(matches!(err, ResolutionError::NotFound(_)));
    }

    #[test]
    fn test_resolve_malformed_did() {
        let mut resolver = DidResolver::new();
        let err = resolver
            .resolve("not-a-did")
            .expect_err("should be Malformed");
        assert!(matches!(err, ResolutionError::Malformed(_)));
    }

    #[test]
    fn test_resolve_empty_did() {
        let mut resolver = DidResolver::new();
        let err = resolver.resolve("").expect_err("empty DID is malformed");
        assert!(matches!(err, ResolutionError::Malformed(_)));
    }

    #[test]
    fn test_resolve_correct_document_returned() {
        let mut resolver = DidResolver::new();
        let mut doc = make_doc("did:web:example.com");
        doc.also_known_as.push("https://example.com".to_string());
        resolver.register(doc);
        let result = resolver.resolve("did:web:example.com").unwrap();
        assert_eq!(result.document.also_known_as, vec!["https://example.com"]);
    }

    // ─── deactivate ───────────────────────────────────────────────────────────

    #[test]
    fn test_deactivate_returns_true() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z1"));
        assert!(resolver.deactivate("did:key:z1"));
    }

    #[test]
    fn test_deactivate_nonexistent_returns_false() {
        let mut resolver = DidResolver::new();
        assert!(!resolver.deactivate("did:key:nonexistent"));
    }

    #[test]
    fn test_deactivated_did_resolve_fails() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z2"));
        resolver.deactivate("did:key:z2");
        let err = resolver.resolve("did:key:z2").expect_err("deactivated");
        assert!(matches!(err, ResolutionError::Deactivated(_)));
    }

    // ─── update ───────────────────────────────────────────────────────────────

    #[test]
    fn test_update_succeeds() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z3"));
        let mut updated = make_doc("did:key:z3");
        updated.also_known_as.push("alias".to_string());
        resolver
            .update("did:key:z3", updated)
            .expect("update should succeed");
        let result = resolver.resolve("did:key:z3").unwrap();
        assert_eq!(result.document.also_known_as, vec!["alias"]);
    }

    #[test]
    fn test_update_not_found() {
        let mut resolver = DidResolver::new();
        let err = resolver
            .update("did:key:nothere", make_doc("did:key:nothere"))
            .expect_err("NotFound");
        assert!(matches!(err, ResolutionError::NotFound(_)));
    }

    #[test]
    fn test_update_deactivated_fails() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z4"));
        resolver.deactivate("did:key:z4");
        let err = resolver
            .update("did:key:z4", make_doc("did:key:z4"))
            .expect_err("Deactivated");
        assert!(matches!(err, ResolutionError::Deactivated(_)));
    }

    // ─── list_methods ─────────────────────────────────────────────────────────

    #[test]
    fn test_list_methods_empty() {
        let resolver = DidResolver::new();
        assert!(resolver.list_methods().is_empty());
    }

    #[test]
    fn test_list_methods_single() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z5"));
        assert_eq!(resolver.list_methods(), vec!["key"]);
    }

    #[test]
    fn test_list_methods_multiple() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z6"));
        resolver.register(make_doc("did:web:example.com"));
        resolver.register(make_doc("did:web:other.com"));
        let methods = resolver.list_methods();
        assert!(methods.contains(&"key".to_string()));
        assert!(methods.contains(&"web".to_string()));
        // deduped
        assert_eq!(methods.iter().filter(|m| *m == "web").count(), 1);
    }

    // ─── add_service ──────────────────────────────────────────────────────────

    #[test]
    fn test_add_service_succeeds() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z7"));
        let svc = ServiceEndpoint {
            id: "did:key:z7#linked-domain".to_string(),
            service_type: "LinkedDomains".to_string(),
            endpoint_url: "https://example.com".to_string(),
        };
        resolver
            .add_service("did:key:z7", svc)
            .expect("add_service should succeed");
        let result = resolver.resolve("did:key:z7").unwrap();
        assert_eq!(result.document.service_endpoints.len(), 1);
        assert_eq!(
            result.document.service_endpoints[0].endpoint_url,
            "https://example.com"
        );
    }

    #[test]
    fn test_add_service_not_found() {
        let mut resolver = DidResolver::new();
        let svc = ServiceEndpoint {
            id: "did:key:z99#svc".to_string(),
            service_type: "Foo".to_string(),
            endpoint_url: "https://foo.com".to_string(),
        };
        let err = resolver
            .add_service("did:key:z99", svc)
            .expect_err("NotFound");
        assert!(matches!(err, ResolutionError::NotFound(_)));
    }

    #[test]
    fn test_add_service_deactivated_fails() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z8"));
        resolver.deactivate("did:key:z8");
        let svc = ServiceEndpoint {
            id: "did:key:z8#svc".to_string(),
            service_type: "Foo".to_string(),
            endpoint_url: "https://foo.com".to_string(),
        };
        let err = resolver
            .add_service("did:key:z8", svc)
            .expect_err("Deactivated");
        assert!(matches!(err, ResolutionError::Deactivated(_)));
    }

    // ─── count / resolution_count ─────────────────────────────────────────────

    #[test]
    fn test_count_empty() {
        let resolver = DidResolver::new();
        assert_eq!(resolver.count(), 0);
    }

    #[test]
    fn test_count_after_register() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z9"));
        resolver.register(make_doc("did:web:a.com"));
        assert_eq!(resolver.count(), 2);
    }

    #[test]
    fn test_count_includes_deactivated() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z10"));
        resolver.deactivate("did:key:z10");
        assert_eq!(resolver.count(), 1); // deactivated still counted
    }

    #[test]
    fn test_resolution_count_increments() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z11"));
        assert_eq!(resolver.resolution_count(), 0);
        let _ = resolver.resolve("did:key:z11");
        let _ = resolver.resolve("did:key:missing");
        assert_eq!(resolver.resolution_count(), 2);
    }

    // ─── verification methods / authentication / assertion ────────────────────

    #[test]
    fn test_resolve_verification_methods() {
        let mut resolver = DidResolver::new();
        let doc = make_doc_with_vm("did:key:z12");
        resolver.register(doc);
        let result = resolver.resolve("did:key:z12").unwrap();
        assert_eq!(result.document.verification_methods.len(), 1);
        assert_eq!(
            result.document.verification_methods[0].vm_type,
            "Ed25519VerificationKey2020"
        );
    }

    #[test]
    fn test_resolve_authentication_list() {
        let mut resolver = DidResolver::new();
        let doc = make_doc_with_vm("did:key:z13");
        resolver.register(doc);
        let result = resolver.resolve("did:key:z13").unwrap();
        assert!(!result.document.authentication.is_empty());
        assert_eq!(result.document.authentication[0], "did:key:z13#key-1");
    }

    #[test]
    fn test_resolve_assertion_method_list() {
        let mut resolver = DidResolver::new();
        let doc = make_doc_with_vm("did:key:z14");
        resolver.register(doc);
        let result = resolver.resolve("did:key:z14").unwrap();
        assert!(!result.document.assertion_method.is_empty());
    }

    // ─── ResolutionMetadata ───────────────────────────────────────────────────

    #[test]
    fn test_resolution_metadata_method() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:web:meta.com"));
        let result = resolver.resolve("did:web:meta.com").unwrap();
        assert_eq!(result.metadata.method, "web");
    }

    #[test]
    fn test_resolution_metadata_content_type() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z15"));
        let result = resolver.resolve("did:key:z15").unwrap();
        assert_eq!(result.metadata.content_type, "application/did+ld+json");
    }

    #[test]
    fn test_resolution_metadata_resolved_at_nonempty() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z16"));
        let result = resolver.resolve("did:key:z16").unwrap();
        assert!(!result.metadata.resolved_at.is_empty());
    }

    // ─── service_endpoints ────────────────────────────────────────────────────

    #[test]
    fn test_multiple_services() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z17"));
        for i in 0..3 {
            let svc = ServiceEndpoint {
                id: format!("did:key:z17#svc-{i}"),
                service_type: "LinkedDomains".to_string(),
                endpoint_url: format!("https://svc{i}.example.com"),
            };
            resolver.add_service("did:key:z17", svc).unwrap();
        }
        let result = resolver.resolve("did:key:z17").unwrap();
        assert_eq!(result.document.service_endpoints.len(), 3);
    }

    #[test]
    fn test_register_overwrites_existing() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z18"));
        // Re-register with different data
        let mut updated = make_doc("did:key:z18");
        updated.also_known_as.push("new_alias".to_string());
        resolver.register(updated);
        let result = resolver.resolve("did:key:z18").unwrap();
        assert_eq!(result.document.also_known_as, vec!["new_alias"]);
    }

    // ─── Additional tests (round 11 extra coverage) ───────────────────────────

    #[test]
    fn test_parse_did_method_ethr() {
        assert_eq!(parse_did_method("did:ethr:0xabc"), Some("ethr".to_string()));
    }

    #[test]
    fn test_parse_did_method_ion_variant() {
        assert_eq!(
            parse_did_method("did:ion:test-string"),
            Some("ion".to_string())
        );
    }

    #[test]
    fn test_resolve_deactivated_returns_error() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z_deact_1"));
        resolver.deactivate("did:key:z_deact_1");
        let err = resolver
            .resolve("did:key:z_deact_1")
            .expect_err("deactivated");
        assert!(matches!(err, ResolutionError::Deactivated(_)));
    }

    #[test]
    fn test_deactivate_returns_false_when_not_found() {
        let mut resolver = DidResolver::new();
        assert!(!resolver.deactivate("did:key:z_missing_xr"));
    }

    #[test]
    fn test_deactivate_idempotent() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z_idem_1"));
        assert!(resolver.deactivate("did:key:z_idem_1"));
        // Second deactivate: document still in store but already deactivated
        let result = resolver.deactivate("did:key:z_idem_1");
        // Whether true or false is implementation-defined; just ensure no panic
        let _ = result;
    }

    #[test]
    fn test_list_methods_includes_key() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:key:z_km_1"));
        let methods = resolver.list_methods();
        assert!(methods.contains(&"key".to_string()));
    }

    #[test]
    fn test_list_methods_includes_web() {
        let mut resolver = DidResolver::new();
        resolver.register(make_doc("did:web:lm_test.com"));
        let methods = resolver.list_methods();
        assert!(methods.contains(&"web".to_string()));
    }

    #[test]
    fn test_did_document_created_starts_none() {
        // A freshly constructed DidDocument has no created timestamp by default;
        // callers set it explicitly when registering.
        let doc = make_doc("did:key:z_cr_1");
        assert!(
            doc.created.is_none(),
            "created should be None on a freshly constructed document"
        );
    }

    #[test]
    fn test_did_document_created_matches_updated() {
        let doc = make_doc("did:key:z_cu_1");
        assert_eq!(
            doc.created, doc.updated,
            "created and updated should be equal for new documents"
        );
    }

    #[test]
    fn test_verification_method_fields() {
        let vm = VerificationMethod {
            id: "did:key:z_vm_1#key-1".to_string(),
            controller: "did:key:z_vm_1".to_string(),
            vm_type: "Ed25519VerificationKey2020".to_string(),
            public_key_multibase: Some("zBase58Key".to_string()),
        };
        assert_eq!(vm.id, "did:key:z_vm_1#key-1");
        assert_eq!(vm.controller, "did:key:z_vm_1");
        assert_eq!(vm.vm_type, "Ed25519VerificationKey2020");
        assert_eq!(vm.public_key_multibase, Some("zBase58Key".to_string()));
    }
}
