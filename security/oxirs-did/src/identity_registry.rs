//! DID identity registry with resolution and caching.
//!
//! Provides an in-memory registry for Decentralised Identifiers (DIDs).
//! Supports registration, resolution, update, deactivation, and method-based
//! lookup — all without network I/O.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// A registered DID entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DidEntry {
    /// The full DID string, e.g. `did:web:example.com`.
    pub did: String,
    /// DID Document serialised as JSON.
    pub document_json: String,
    /// Unix epoch (seconds) when the entry was first created.
    pub created_at: u64,
    /// Unix epoch (seconds) of the last update.
    pub updated_at: u64,
    /// Whether the DID has been deactivated.
    pub deactivated: bool,
}

impl DidEntry {
    /// Create a new active entry.
    pub fn new(did: impl Into<String>, document_json: impl Into<String>, timestamp: u64) -> Self {
        let did = did.into();
        Self {
            did,
            document_json: document_json.into(),
            created_at: timestamp,
            updated_at: timestamp,
            deactivated: false,
        }
    }
}

/// Aggregate statistics for the registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryStats {
    /// Total DID entries (active + deactivated).
    pub total_dids: usize,
    /// Active (non-deactivated) entries.
    pub active_dids: usize,
    /// Deactivated entries.
    pub deactivated_dids: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during registry operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// The requested DID was not found.
    NotFound(String),
    /// A DID with the same identifier already exists.
    AlreadyExists(String),
    /// The DID has been deactivated and cannot be updated.
    Deactivated(String),
    /// The DID string does not conform to the `did:<method>:<id>` format.
    InvalidDid(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::NotFound(d) => write!(f, "DID not found: {d}"),
            RegistryError::AlreadyExists(d) => write!(f, "DID already registered: {d}"),
            RegistryError::Deactivated(d) => write!(f, "DID is deactivated: {d}"),
            RegistryError::InvalidDid(d) => write!(f, "Invalid DID format: {d}"),
        }
    }
}

impl std::error::Error for RegistryError {}

// ─────────────────────────────────────────────────────────────────────────────
// IdentityRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory identity registry for DIDs.
///
/// All operations are synchronous and O(1) or O(n) at worst.
#[derive(Debug, Default)]
pub struct IdentityRegistry {
    /// Primary store: DID → entry.
    entries: HashMap<String, DidEntry>,
    /// Secondary index: method → list of DIDs with that method.
    did_by_method: HashMap<String, Vec<String>>,
}

impl IdentityRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new DID entry.
    ///
    /// # Errors
    /// - `InvalidDid` if `entry.did` is not in `did:<method>:<id>` format.
    /// - `AlreadyExists` if the DID is already registered.
    pub fn register(&mut self, entry: DidEntry) -> Result<(), RegistryError> {
        if !Self::is_valid_did(&entry.did) {
            return Err(RegistryError::InvalidDid(entry.did.clone()));
        }
        if self.entries.contains_key(&entry.did) {
            return Err(RegistryError::AlreadyExists(entry.did.clone()));
        }

        if let Some(method) = Self::parse_method(&entry.did) {
            self.did_by_method
                .entry(method)
                .or_default()
                .push(entry.did.clone());
        }

        self.entries.insert(entry.did.clone(), entry);
        Ok(())
    }

    /// Resolve a DID by its identifier.
    ///
    /// # Errors
    /// - `NotFound` if the DID is not in the registry.
    /// - `Deactivated` if the entry exists but has been deactivated.
    pub fn resolve(&self, did: &str) -> Result<&DidEntry, RegistryError> {
        let entry = self
            .entries
            .get(did)
            .ok_or_else(|| RegistryError::NotFound(did.to_string()))?;

        if entry.deactivated {
            return Err(RegistryError::Deactivated(did.to_string()));
        }

        Ok(entry)
    }

    /// Resolve a DID even if it is deactivated (for audit purposes).
    ///
    /// # Errors
    /// - `NotFound` if the DID is not in the registry.
    pub fn resolve_any(&self, did: &str) -> Result<&DidEntry, RegistryError> {
        self.entries
            .get(did)
            .ok_or_else(|| RegistryError::NotFound(did.to_string()))
    }

    /// Update the DID Document for an existing, active entry.
    ///
    /// # Errors
    /// - `NotFound` if the DID is not registered.
    /// - `Deactivated` if the entry is deactivated.
    pub fn update(
        &mut self,
        did: &str,
        new_document: String,
        timestamp: u64,
    ) -> Result<(), RegistryError> {
        let entry = self
            .entries
            .get_mut(did)
            .ok_or_else(|| RegistryError::NotFound(did.to_string()))?;

        if entry.deactivated {
            return Err(RegistryError::Deactivated(did.to_string()));
        }

        entry.document_json = new_document;
        entry.updated_at = timestamp;
        Ok(())
    }

    /// Mark a DID as deactivated.
    ///
    /// # Errors
    /// - `NotFound` if the DID is not registered.
    /// - `Deactivated` if the entry is already deactivated.
    pub fn deactivate(&mut self, did: &str) -> Result<(), RegistryError> {
        let entry = self
            .entries
            .get_mut(did)
            .ok_or_else(|| RegistryError::NotFound(did.to_string()))?;

        if entry.deactivated {
            return Err(RegistryError::Deactivated(did.to_string()));
        }

        entry.deactivated = true;
        Ok(())
    }

    /// List all entries (active and deactivated) for a given DID method.
    ///
    /// E.g. `list_by_method("web")` returns all `did:web:…` entries.
    pub fn list_by_method(&self, method: &str) -> Vec<&DidEntry> {
        self.did_by_method
            .get(method)
            .map(|dids| {
                dids.iter()
                    .filter_map(|did| self.entries.get(did))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Compute aggregate statistics for the registry.
    pub fn stats(&self) -> RegistryStats {
        let total_dids = self.entries.len();
        let deactivated_dids = self.entries.values().filter(|e| e.deactivated).count();
        let active_dids = total_dids - deactivated_dids;
        RegistryStats {
            total_dids,
            active_dids,
            deactivated_dids,
        }
    }

    // ── Static helpers ────────────────────────────────────────────────────────

    /// Extract the DID method from a DID string.
    ///
    /// `"did:web:example.com"` → `Some("web")`
    /// `"not-a-did"` → `None`
    pub fn parse_method(did: &str) -> Option<String> {
        let parts: Vec<&str> = did.splitn(3, ':').collect();
        if parts.len() >= 2 && parts[0] == "did" && !parts[1].is_empty() {
            Some(parts[1].to_string())
        } else {
            None
        }
    }

    /// Return `true` if `did` conforms to the minimal `did:<method>:<id>` syntax.
    ///
    /// Rules enforced:
    /// - Must start with `"did:"`.
    /// - Method segment must be non-empty and contain only lowercase letters and digits.
    /// - Identifier segment must be non-empty.
    pub fn is_valid_did(did: &str) -> bool {
        let parts: Vec<&str> = did.splitn(3, ':').collect();
        if parts.len() < 3 {
            return false;
        }
        if parts[0] != "did" {
            return false;
        }
        let method = parts[1];
        if method.is_empty() {
            return false;
        }
        // Method must be lowercase letters or digits only
        if !method
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        {
            return false;
        }
        // Identifier segment must be non-empty
        !parts[2].is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(did: &str) -> DidEntry {
        DidEntry::new(did, r#"{"id":"placeholder"}"#, 1_000_000)
    }

    // ── is_valid_did ─────────────────────────────────────────────────────────

    #[test]
    fn test_valid_did_key() {
        assert!(IdentityRegistry::is_valid_did("did:key:z6Mk"));
    }

    #[test]
    fn test_valid_did_web() {
        assert!(IdentityRegistry::is_valid_did("did:web:example.com"));
    }

    #[test]
    fn test_valid_did_ethr() {
        assert!(IdentityRegistry::is_valid_did("did:ethr:0xabc123"));
    }

    #[test]
    fn test_invalid_did_no_prefix() {
        assert!(!IdentityRegistry::is_valid_did("key:z6Mk"));
    }

    #[test]
    fn test_invalid_did_missing_id() {
        assert!(!IdentityRegistry::is_valid_did("did:web:"));
    }

    #[test]
    fn test_invalid_did_missing_method() {
        assert!(!IdentityRegistry::is_valid_did("did::example"));
    }

    #[test]
    fn test_invalid_did_uppercase_method() {
        assert!(!IdentityRegistry::is_valid_did("did:Web:example.com"));
    }

    #[test]
    fn test_invalid_did_empty() {
        assert!(!IdentityRegistry::is_valid_did(""));
    }

    #[test]
    fn test_invalid_did_only_did() {
        assert!(!IdentityRegistry::is_valid_did("did"));
    }

    // ── parse_method ─────────────────────────────────────────────────────────

    #[test]
    fn test_parse_method_web() {
        assert_eq!(
            IdentityRegistry::parse_method("did:web:example.com"),
            Some("web".to_string())
        );
    }

    #[test]
    fn test_parse_method_key() {
        assert_eq!(
            IdentityRegistry::parse_method("did:key:z6Mk"),
            Some("key".to_string())
        );
    }

    #[test]
    fn test_parse_method_invalid() {
        assert_eq!(IdentityRegistry::parse_method("not-a-did"), None);
    }

    #[test]
    fn test_parse_method_missing_id() {
        // "did:web" without identifier
        assert_eq!(
            IdentityRegistry::parse_method("did:web"),
            Some("web".to_string())
        );
    }

    // ── DidEntry::new ─────────────────────────────────────────────────────────

    #[test]
    fn test_entry_new_defaults() {
        let e = DidEntry::new("did:key:z1", "{}", 42);
        assert_eq!(e.did, "did:key:z1");
        assert_eq!(e.created_at, 42);
        assert_eq!(e.updated_at, 42);
        assert!(!e.deactivated);
    }

    #[test]
    fn test_entry_clone() {
        let e = DidEntry::new("did:key:z2", "{}", 100);
        let e2 = e.clone();
        assert_eq!(e, e2);
    }

    // ── register ─────────────────────────────────────────────────────────────

    #[test]
    fn test_register_success() {
        let mut reg = IdentityRegistry::new();
        assert!(reg.register(sample_entry("did:key:z6Mk")).is_ok());
        assert_eq!(reg.stats().total_dids, 1);
    }

    #[test]
    fn test_register_invalid_did() {
        let mut reg = IdentityRegistry::new();
        let e = DidEntry::new("not-a-did", "{}", 0);
        assert!(matches!(reg.register(e), Err(RegistryError::InvalidDid(_))));
    }

    #[test]
    fn test_register_duplicate() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z6Mk")).unwrap();
        let result = reg.register(sample_entry("did:key:z6Mk"));
        assert!(matches!(result, Err(RegistryError::AlreadyExists(_))));
    }

    #[test]
    fn test_register_multiple_methods() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z1")).unwrap();
        reg.register(sample_entry("did:web:example.com")).unwrap();
        assert_eq!(reg.stats().total_dids, 2);
    }

    // ── resolve ───────────────────────────────────────────────────────────────

    #[test]
    fn test_resolve_success() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z6Mk")).unwrap();
        let entry = reg.resolve("did:key:z6Mk").unwrap();
        assert_eq!(entry.did, "did:key:z6Mk");
    }

    #[test]
    fn test_resolve_not_found() {
        let reg = IdentityRegistry::new();
        assert!(matches!(
            reg.resolve("did:key:unknown"),
            Err(RegistryError::NotFound(_))
        ));
    }

    #[test]
    fn test_resolve_deactivated_returns_error() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z6Mk")).unwrap();
        reg.deactivate("did:key:z6Mk").unwrap();
        assert!(matches!(
            reg.resolve("did:key:z6Mk"),
            Err(RegistryError::Deactivated(_))
        ));
    }

    // ── resolve_any ───────────────────────────────────────────────────────────

    #[test]
    fn test_resolve_any_active() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z1")).unwrap();
        assert!(reg.resolve_any("did:key:z1").is_ok());
    }

    #[test]
    fn test_resolve_any_deactivated() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z1")).unwrap();
        reg.deactivate("did:key:z1").unwrap();
        let e = reg.resolve_any("did:key:z1").unwrap();
        assert!(e.deactivated);
    }

    #[test]
    fn test_resolve_any_not_found() {
        let reg = IdentityRegistry::new();
        assert!(matches!(
            reg.resolve_any("did:key:z99"),
            Err(RegistryError::NotFound(_))
        ));
    }

    // ── update ────────────────────────────────────────────────────────────────

    #[test]
    fn test_update_success() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z1")).unwrap();
        reg.update("did:key:z1", r#"{"id":"new"}"#.to_string(), 2_000_000)
            .unwrap();
        let e = reg.resolve("did:key:z1").unwrap();
        assert_eq!(e.document_json, r#"{"id":"new"}"#);
        assert_eq!(e.updated_at, 2_000_000);
    }

    #[test]
    fn test_update_not_found() {
        let mut reg = IdentityRegistry::new();
        assert!(matches!(
            reg.update("did:key:unknown", "{}".to_string(), 0),
            Err(RegistryError::NotFound(_))
        ));
    }

    #[test]
    fn test_update_deactivated() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z1")).unwrap();
        reg.deactivate("did:key:z1").unwrap();
        assert!(matches!(
            reg.update("did:key:z1", "{}".to_string(), 0),
            Err(RegistryError::Deactivated(_))
        ));
    }

    // ── deactivate ────────────────────────────────────────────────────────────

    #[test]
    fn test_deactivate_success() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z1")).unwrap();
        assert!(reg.deactivate("did:key:z1").is_ok());
        let e = reg.resolve_any("did:key:z1").unwrap();
        assert!(e.deactivated);
    }

    #[test]
    fn test_deactivate_not_found() {
        let mut reg = IdentityRegistry::new();
        assert!(matches!(
            reg.deactivate("did:key:unknown"),
            Err(RegistryError::NotFound(_))
        ));
    }

    #[test]
    fn test_deactivate_already_deactivated() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z1")).unwrap();
        reg.deactivate("did:key:z1").unwrap();
        assert!(matches!(
            reg.deactivate("did:key:z1"),
            Err(RegistryError::Deactivated(_))
        ));
    }

    // ── list_by_method ────────────────────────────────────────────────────────

    #[test]
    fn test_list_by_method_empty() {
        let reg = IdentityRegistry::new();
        assert!(reg.list_by_method("web").is_empty());
    }

    #[test]
    fn test_list_by_method_one() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:web:example.com")).unwrap();
        let list = reg.list_by_method("web");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].did, "did:web:example.com");
    }

    #[test]
    fn test_list_by_method_multiple() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z1")).unwrap();
        reg.register(sample_entry("did:key:z2")).unwrap();
        reg.register(sample_entry("did:web:example.com")).unwrap();
        let list = reg.list_by_method("key");
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_list_by_method_includes_deactivated() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z1")).unwrap();
        reg.deactivate("did:key:z1").unwrap();
        let list = reg.list_by_method("key");
        assert_eq!(list.len(), 1);
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let reg = IdentityRegistry::new();
        let s = reg.stats();
        assert_eq!(s.total_dids, 0);
        assert_eq!(s.active_dids, 0);
        assert_eq!(s.deactivated_dids, 0);
    }

    #[test]
    fn test_stats_after_register() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z1")).unwrap();
        reg.register(sample_entry("did:key:z2")).unwrap();
        let s = reg.stats();
        assert_eq!(s.total_dids, 2);
        assert_eq!(s.active_dids, 2);
        assert_eq!(s.deactivated_dids, 0);
    }

    #[test]
    fn test_stats_after_deactivate() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z1")).unwrap();
        reg.register(sample_entry("did:key:z2")).unwrap();
        reg.deactivate("did:key:z1").unwrap();
        let s = reg.stats();
        assert_eq!(s.total_dids, 2);
        assert_eq!(s.active_dids, 1);
        assert_eq!(s.deactivated_dids, 1);
    }

    // ── error display ─────────────────────────────────────────────────────────

    #[test]
    fn test_error_display_not_found() {
        let e = RegistryError::NotFound("did:key:z1".to_string());
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn test_error_display_already_exists() {
        let e = RegistryError::AlreadyExists("did:key:z1".to_string());
        assert!(e.to_string().contains("already"));
    }

    #[test]
    fn test_error_display_deactivated() {
        let e = RegistryError::Deactivated("did:key:z1".to_string());
        assert!(e.to_string().contains("deactivated"));
    }

    #[test]
    fn test_error_display_invalid_did() {
        let e = RegistryError::InvalidDid("bad".to_string());
        assert!(e.to_string().contains("Invalid"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(RegistryError::NotFound("x".to_string()));
        assert!(!e.to_string().is_empty());
    }

    // ── RegistryStats fields ──────────────────────────────────────────────────

    #[test]
    fn test_registry_stats_fields() {
        let s = RegistryStats {
            total_dids: 10,
            active_dids: 7,
            deactivated_dids: 3,
        };
        assert_eq!(s.total_dids, 10);
        assert_eq!(s.active_dids + s.deactivated_dids, s.total_dids);
    }

    // ── RegistryStats clone / eq ──────────────────────────────────────────────

    #[test]
    fn test_registry_stats_clone() {
        let s = RegistryStats {
            total_dids: 5,
            active_dids: 4,
            deactivated_dids: 1,
        };
        let s2 = s.clone();
        assert_eq!(s, s2);
    }

    // ── register many methods ─────────────────────────────────────────────────

    #[test]
    fn test_register_many_dids() {
        let mut reg = IdentityRegistry::new();
        for i in 0..10 {
            reg.register(sample_entry(&format!("did:key:z{i}")))
                .unwrap();
        }
        assert_eq!(reg.stats().total_dids, 10);
    }

    // ── list_by_method returns only matching method ───────────────────────────

    #[test]
    fn test_list_by_method_no_cross_contamination() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z1")).unwrap();
        reg.register(sample_entry("did:web:example.com")).unwrap();
        let web_list = reg.list_by_method("web");
        assert_eq!(web_list.len(), 1);
        assert_eq!(web_list[0].did, "did:web:example.com");
    }

    // ── update preserves created_at ───────────────────────────────────────────

    #[test]
    fn test_update_preserves_created_at() {
        let mut reg = IdentityRegistry::new();
        reg.register(sample_entry("did:key:z1")).unwrap();
        reg.update("did:key:z1", "{}".to_string(), 9_999_999)
            .unwrap();
        let e = reg.resolve("did:key:z1").unwrap();
        assert_eq!(e.created_at, 1_000_000); // unchanged
        assert_eq!(e.updated_at, 9_999_999);
    }

    // ── is_valid_did numeric method ───────────────────────────────────────────

    #[test]
    fn test_valid_did_numeric_method() {
        assert!(IdentityRegistry::is_valid_did("did:123:abc"));
    }

    // ── RegistryError variants equality ──────────────────────────────────────

    #[test]
    fn test_registry_error_eq() {
        assert_eq!(
            RegistryError::NotFound("a".to_string()),
            RegistryError::NotFound("a".to_string())
        );
        assert_ne!(
            RegistryError::NotFound("a".to_string()),
            RegistryError::NotFound("b".to_string())
        );
    }
}
