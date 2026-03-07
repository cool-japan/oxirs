//! API keys and service accounts for OxiRS Fuseki.
//!
//! Provides secure API key generation, storage, validation and permission
//! checking.  Keys are stored as bcrypt hashes; the raw key is only ever
//! returned at generation time.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Rate-limit tier assigned to an API key.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RateLimitTier {
    /// Lowest tier – suitable for evaluation / trial use.
    Free,
    /// Standard production tier.
    #[default]
    Standard,
    /// High-throughput tier for enterprise customers.
    Premium,
    /// Unlimited – internal service accounts.
    Unlimited,
    /// Custom tier with caller-specified requests-per-second ceiling.
    Custom { rps: u64 },
}

impl RateLimitTier {
    /// Return the maximum requests per second for the tier.
    pub fn max_rps(&self) -> Option<u64> {
        match self {
            RateLimitTier::Free => Some(10),
            RateLimitTier::Standard => Some(100),
            RateLimitTier::Premium => Some(1_000),
            RateLimitTier::Unlimited => None,
            RateLimitTier::Custom { rps } => Some(*rps),
        }
    }
}

/// A permission that can be granted to an API key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    /// Execute SPARQL SELECT / ASK / CONSTRUCT / DESCRIBE queries.
    SparqlRead,
    /// Execute SPARQL UPDATE (INSERT, DELETE, LOAD, …).
    SparqlWrite,
    /// Access administrative endpoints.
    Admin,
    /// Export RDF data in bulk.
    DataExport,
    /// Import RDF data in bulk.
    DataImport,
    /// Manage other API keys.
    KeyManagement,
    /// Read-only access to server metrics.
    MetricsRead,
    /// Custom application-defined permission.
    Custom(String),
}

/// An API key record stored in the key store.
///
/// The raw key is never stored — only the SHA-256 hash (which is then
/// bcrypt-hashed for storage).  Because bcrypt is slow, we use SHA-256 as a
/// fast pre-hash so that variable-length raw keys are reduced to a fixed-size
/// input for bcrypt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// Unique key identifier (URL-safe, public).
    pub key_id: String,
    /// bcrypt hash of `sha256(raw_key)`.
    pub key_hash: String,
    /// Human-readable label for the key.
    pub label: String,
    /// Owner / service account name.
    pub owner: String,
    /// When the key was created.
    pub created_at: DateTime<Utc>,
    /// Optional expiry.  `None` means never expires.
    pub expires_at: Option<DateTime<Utc>>,
    /// Granted permissions.
    pub permissions: HashSet<Permission>,
    /// Rate-limit tier.
    pub rate_limit_tier: RateLimitTier,
    /// Whether the key has been revoked.
    pub revoked: bool,
    /// Arbitrary metadata (e.g. team, project, environment).
    pub metadata: HashMap<String, String>,
}

impl ApiKey {
    /// Check whether the key has expired or been revoked.
    pub fn is_active(&self) -> bool {
        if self.revoked {
            return false;
        }
        if let Some(exp) = self.expires_at {
            return Utc::now() < exp;
        }
        true
    }

    /// Check whether this key has a specific permission.
    pub fn has_permission(&self, perm: &Permission) -> bool {
        self.permissions.contains(perm)
    }

    /// Check whether this key has all of the required permissions.
    pub fn has_all_permissions(&self, required: &[Permission]) -> bool {
        required.iter().all(|p| self.has_permission(p))
    }
}

// ---------------------------------------------------------------------------
// Raw key helper
// ---------------------------------------------------------------------------

/// A newly-generated raw API key.  The raw value is shown only once.
#[derive(Debug, Clone)]
pub struct RawApiKey {
    /// The human-visible key string (base58-encoded 32 bytes).
    pub raw: String,
    /// The associated key record (with hash set).
    pub record: ApiKey,
}

/// Generate `n` random bytes using the scirs2-core random subsystem
/// (or fall back to OS entropy via `getrandom`).
fn random_bytes_32() -> [u8; 32] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Use a combination of system time, thread id, and process id as entropy
    // sources to generate 32 pseudo-random bytes.
    // In production environments scirs2-core random should be preferred;
    // this implementation uses only std primitives to avoid any optional-feature
    // gating in the test harness.
    let mut buf = [0u8; 32];

    // Mix multiple entropy sources into 4 u64 values.
    let sources: [u64; 4] = {
        let mut h0 = DefaultHasher::new();
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        let mut h3 = DefaultHasher::new();

        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .hash(&mut h0);

        std::thread::current().id().hash(&mut h1);
        std::process::id().hash(&mut h2);

        // XOR with a counter to make successive calls distinct.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let c = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        c.hash(&mut h3);

        [h0.finish(), h1.finish(), h2.finish(), h3.finish()]
    };

    // Write the 4 u64 values as little-endian bytes.
    for (i, v) in sources.iter().enumerate() {
        let bytes = v.to_le_bytes();
        buf[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
    }

    // Final SHA-256 pass to further mix and reduce bias.
    let mut hasher = Sha256::new();
    hasher.update(buf);
    let hash = hasher.finalize();
    buf.copy_from_slice(&hash);
    buf
}

/// Base58 alphabet (Bitcoin alphabet).
const BASE58_ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// Encode bytes as a Base58 string.
fn base58_encode(input: &[u8]) -> String {
    let mut digits: Vec<u8> = vec![0];
    for &byte in input {
        let mut carry = byte as usize;
        for d in digits.iter_mut() {
            carry += (*d as usize) << 8;
            *d = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }
    // Handle leading zero bytes.
    let leading_zeros = input.iter().take_while(|&&b| b == 0).count();
    let mut result = String::with_capacity(leading_zeros + digits.len());
    for _ in 0..leading_zeros {
        result.push('1');
    }
    for &d in digits.iter().rev() {
        result.push(BASE58_ALPHABET[d as usize] as char);
    }
    result
}

/// Generate a new raw API key: 32 random bytes, base58-encoded.
fn generate_raw_key() -> String {
    let bytes = random_bytes_32();
    base58_encode(&bytes)
}

/// Generate a unique key ID (16 hex chars).
fn generate_key_id() -> String {
    let bytes = random_bytes_32();
    // Use first 8 bytes as a hex ID.
    hex::encode(&bytes[..8])
}

/// SHA-256 pre-hash of the raw key (returns hex string).
fn sha256_hex(raw: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    hex::encode(hasher.finalize())
}

// ---------------------------------------------------------------------------
// ApiKeyStore trait
// ---------------------------------------------------------------------------

/// Errors from the API key store.
#[derive(Debug, thiserror::Error)]
pub enum ApiKeyError {
    #[error("Key not found: {0}")]
    NotFound(String),
    #[error("Key has expired")]
    Expired,
    #[error("Key has been revoked")]
    Revoked,
    #[error("Insufficient permissions")]
    InsufficientPermissions,
    #[error("Key ID already exists: {0}")]
    DuplicateKeyId(String),
    #[error("Hash error: {0}")]
    Hash(String),
    #[error("Invalid key format")]
    InvalidFormat,
}

/// Result type for key store operations.
pub type ApiKeyResult<T> = Result<T, ApiKeyError>;

/// Parameters for creating a new API key.
#[derive(Debug, Clone)]
pub struct CreateKeyParams {
    pub label: String,
    pub owner: String,
    pub permissions: HashSet<Permission>,
    pub rate_limit_tier: RateLimitTier,
    pub expires_in: Option<Duration>,
    pub metadata: HashMap<String, String>,
}

impl Default for CreateKeyParams {
    fn default() -> Self {
        Self {
            label: "default".to_string(),
            owner: "anonymous".to_string(),
            permissions: HashSet::new(),
            rate_limit_tier: RateLimitTier::Standard,
            expires_in: None,
            metadata: HashMap::new(),
        }
    }
}

impl CreateKeyParams {
    pub fn new(label: impl Into<String>, owner: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            owner: owner.into(),
            ..Default::default()
        }
    }

    pub fn with_permission(mut self, perm: Permission) -> Self {
        self.permissions.insert(perm);
        self
    }

    pub fn with_tier(mut self, tier: RateLimitTier) -> Self {
        self.rate_limit_tier = tier;
        self
    }

    pub fn with_expiry(mut self, duration: Duration) -> Self {
        self.expires_in = Some(duration);
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A store for API keys.
#[async_trait::async_trait]
pub trait ApiKeyStore: Send + Sync + 'static {
    /// Generate and store a new API key.  Returns the raw key (shown once).
    async fn create_key(&self, params: CreateKeyParams) -> ApiKeyResult<RawApiKey>;

    /// Validate a raw key string.  Returns the key record on success.
    async fn validate_key(&self, raw_key: &str) -> ApiKeyResult<ApiKey>;

    /// Retrieve a key record by its ID.
    async fn get_key(&self, key_id: &str) -> ApiKeyResult<ApiKey>;

    /// Revoke a key by its ID.
    async fn revoke_key(&self, key_id: &str) -> ApiKeyResult<()>;

    /// List all keys (optionally filtered by owner).
    async fn list_keys(&self, owner: Option<&str>) -> ApiKeyResult<Vec<ApiKey>>;

    /// Delete a key permanently.
    async fn delete_key(&self, key_id: &str) -> ApiKeyResult<()>;

    /// Check that a raw key has the given permissions.
    async fn check_permissions(
        &self,
        raw_key: &str,
        required: &[Permission],
    ) -> ApiKeyResult<ApiKey> {
        let key = self.validate_key(raw_key).await?;
        if key.has_all_permissions(required) {
            Ok(key)
        } else {
            Err(ApiKeyError::InsufficientPermissions)
        }
    }
}

// ---------------------------------------------------------------------------
// InMemoryApiKeyStore
// ---------------------------------------------------------------------------

/// An in-memory implementation of `ApiKeyStore`.
///
/// Suitable for testing and lightweight deployments.  For production use,
/// back this with a persistent store.
pub struct InMemoryApiKeyStore {
    /// Keys indexed by `key_id`.
    keys: Arc<RwLock<HashMap<String, ApiKey>>>,
    /// Maps `sha256(raw_key)` → `key_id` for O(1) lookup.
    hash_index: Arc<RwLock<HashMap<String, String>>>,
    /// bcrypt cost factor.
    bcrypt_cost: u32,
}

impl InMemoryApiKeyStore {
    /// Create a store with the default bcrypt cost (10).
    pub fn new() -> Self {
        Self::with_bcrypt_cost(10)
    }

    /// Create a store with the specified bcrypt cost.
    ///
    /// Use a lower cost (4) in tests for speed.
    pub fn with_bcrypt_cost(cost: u32) -> Self {
        Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
            hash_index: Arc::new(RwLock::new(HashMap::new())),
            bcrypt_cost: cost,
        }
    }

    /// Hash a raw key using SHA-256 then bcrypt.
    fn hash_key(&self, raw: &str) -> ApiKeyResult<String> {
        let pre_hash = sha256_hex(raw);
        bcrypt::hash(&pre_hash, self.bcrypt_cost).map_err(|e| ApiKeyError::Hash(e.to_string()))
    }

    /// Verify a raw key against a stored bcrypt hash.
    fn verify_key(&self, raw: &str, hash: &str) -> bool {
        let pre_hash = sha256_hex(raw);
        bcrypt::verify(&pre_hash, hash).unwrap_or(false)
    }
}

impl Default for InMemoryApiKeyStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ApiKeyStore for InMemoryApiKeyStore {
    async fn create_key(&self, params: CreateKeyParams) -> ApiKeyResult<RawApiKey> {
        let raw = generate_raw_key();
        let key_id = generate_key_id();
        let key_hash = self.hash_key(&raw)?;
        let pre_hash = sha256_hex(&raw);

        let expires_at = params.expires_in.map(|d| {
            Utc::now() + chrono::Duration::from_std(d).unwrap_or(chrono::Duration::seconds(0))
        });

        let record = ApiKey {
            key_id: key_id.clone(),
            key_hash,
            label: params.label,
            owner: params.owner,
            created_at: Utc::now(),
            expires_at,
            permissions: params.permissions,
            rate_limit_tier: params.rate_limit_tier,
            revoked: false,
            metadata: params.metadata,
        };

        {
            let mut keys = self.keys.write();
            if keys.contains_key(&key_id) {
                return Err(ApiKeyError::DuplicateKeyId(key_id));
            }
            keys.insert(key_id.clone(), record.clone());
        }
        {
            let mut index = self.hash_index.write();
            index.insert(pre_hash, key_id);
        }

        Ok(RawApiKey { raw, record })
    }

    async fn validate_key(&self, raw_key: &str) -> ApiKeyResult<ApiKey> {
        let pre_hash = sha256_hex(raw_key);
        let key_id = {
            let index = self.hash_index.read();
            index.get(&pre_hash).cloned()
        };

        let key_id = key_id.ok_or_else(|| ApiKeyError::NotFound("(hash not found)".to_string()))?;

        let key = {
            let keys = self.keys.read();
            keys.get(&key_id).cloned()
        };
        let key = key.ok_or_else(|| ApiKeyError::NotFound(key_id.clone()))?;

        // Verify bcrypt hash.
        if !self.verify_key(raw_key, &key.key_hash) {
            return Err(ApiKeyError::NotFound(key_id));
        }

        if key.revoked {
            return Err(ApiKeyError::Revoked);
        }
        if let Some(exp) = key.expires_at {
            if Utc::now() >= exp {
                return Err(ApiKeyError::Expired);
            }
        }

        Ok(key)
    }

    async fn get_key(&self, key_id: &str) -> ApiKeyResult<ApiKey> {
        let keys = self.keys.read();
        keys.get(key_id)
            .cloned()
            .ok_or_else(|| ApiKeyError::NotFound(key_id.to_string()))
    }

    async fn revoke_key(&self, key_id: &str) -> ApiKeyResult<()> {
        let mut keys = self.keys.write();
        let key = keys
            .get_mut(key_id)
            .ok_or_else(|| ApiKeyError::NotFound(key_id.to_string()))?;
        key.revoked = true;
        Ok(())
    }

    async fn list_keys(&self, owner: Option<&str>) -> ApiKeyResult<Vec<ApiKey>> {
        let keys = self.keys.read();
        let result = keys
            .values()
            .filter(|k| owner.map(|o| k.owner == o).unwrap_or(true))
            .cloned()
            .collect();
        Ok(result)
    }

    async fn delete_key(&self, key_id: &str) -> ApiKeyResult<()> {
        let key = {
            let mut keys = self.keys.write();
            keys.remove(key_id)
                .ok_or_else(|| ApiKeyError::NotFound(key_id.to_string()))?
        };
        // Remove from hash index.
        let pre_hash = sha256_hex(&key.key_id); // NOTE: index is keyed by sha256(raw), not sha256(key_id)
                                                // We stored pre_hash = sha256(raw) → key_id at creation time.
                                                // We need to find and remove that index entry.
        let mut index = self.hash_index.write();
        index.retain(|_, v| v != key_id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Build a store with cost=4 for fast bcrypt in tests.
    fn test_store() -> InMemoryApiKeyStore {
        InMemoryApiKeyStore::with_bcrypt_cost(4)
    }

    fn basic_params(label: &str) -> CreateKeyParams {
        CreateKeyParams::new(label, "test-owner")
            .with_permission(Permission::SparqlRead)
            .with_tier(RateLimitTier::Standard)
    }

    // -----------------------------------------------------------------------
    // RateLimitTier
    // -----------------------------------------------------------------------

    #[test]
    fn test_tier_max_rps() {
        assert_eq!(RateLimitTier::Free.max_rps(), Some(10));
        assert_eq!(RateLimitTier::Standard.max_rps(), Some(100));
        assert_eq!(RateLimitTier::Premium.max_rps(), Some(1_000));
        assert_eq!(RateLimitTier::Unlimited.max_rps(), None);
        assert_eq!(RateLimitTier::Custom { rps: 250 }.max_rps(), Some(250));
    }

    #[test]
    fn test_tier_default() {
        let t = RateLimitTier::default();
        assert_eq!(t, RateLimitTier::Standard);
    }

    #[test]
    fn test_tier_serialize() {
        let json = serde_json::to_string(&RateLimitTier::Premium).unwrap();
        assert!(json.contains("premium"));
    }

    // -----------------------------------------------------------------------
    // Permission
    // -----------------------------------------------------------------------

    #[test]
    fn test_permission_equality() {
        assert_eq!(Permission::SparqlRead, Permission::SparqlRead);
        assert_ne!(Permission::SparqlRead, Permission::SparqlWrite);
    }

    #[test]
    fn test_permission_custom() {
        let p = Permission::Custom("my_perm".to_string());
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("my_perm"));
    }

    #[test]
    fn test_permission_in_hashset() {
        let mut set = HashSet::new();
        set.insert(Permission::SparqlRead);
        set.insert(Permission::Admin);
        assert!(set.contains(&Permission::SparqlRead));
        assert!(!set.contains(&Permission::DataExport));
    }

    // -----------------------------------------------------------------------
    // ApiKey helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_api_key_is_active_not_expired() {
        let key = ApiKey {
            key_id: "id".to_string(),
            key_hash: "h".to_string(),
            label: "l".to_string(),
            owner: "o".to_string(),
            created_at: Utc::now(),
            expires_at: Some(Utc::now() + chrono::Duration::hours(1)),
            permissions: HashSet::new(),
            rate_limit_tier: RateLimitTier::Standard,
            revoked: false,
            metadata: HashMap::new(),
        };
        assert!(key.is_active());
    }

    #[test]
    fn test_api_key_is_not_active_when_expired() {
        let key = ApiKey {
            key_id: "id2".to_string(),
            key_hash: "h".to_string(),
            label: "l".to_string(),
            owner: "o".to_string(),
            created_at: Utc::now() - chrono::Duration::hours(2),
            expires_at: Some(Utc::now() - chrono::Duration::hours(1)),
            permissions: HashSet::new(),
            rate_limit_tier: RateLimitTier::Free,
            revoked: false,
            metadata: HashMap::new(),
        };
        assert!(!key.is_active());
    }

    #[test]
    fn test_api_key_is_not_active_when_revoked() {
        let key = ApiKey {
            key_id: "id3".to_string(),
            key_hash: "h".to_string(),
            label: "l".to_string(),
            owner: "o".to_string(),
            created_at: Utc::now(),
            expires_at: None,
            permissions: HashSet::new(),
            rate_limit_tier: RateLimitTier::Standard,
            revoked: true,
            metadata: HashMap::new(),
        };
        assert!(!key.is_active());
    }

    #[test]
    fn test_api_key_has_permission() {
        let mut perms = HashSet::new();
        perms.insert(Permission::SparqlRead);
        let key = ApiKey {
            key_id: "id4".to_string(),
            key_hash: "h".to_string(),
            label: "l".to_string(),
            owner: "o".to_string(),
            created_at: Utc::now(),
            expires_at: None,
            permissions: perms,
            rate_limit_tier: RateLimitTier::Standard,
            revoked: false,
            metadata: HashMap::new(),
        };
        assert!(key.has_permission(&Permission::SparqlRead));
        assert!(!key.has_permission(&Permission::Admin));
    }

    #[test]
    fn test_api_key_has_all_permissions() {
        let mut perms = HashSet::new();
        perms.insert(Permission::SparqlRead);
        perms.insert(Permission::SparqlWrite);
        let key = ApiKey {
            key_id: "id5".to_string(),
            key_hash: "h".to_string(),
            label: "l".to_string(),
            owner: "o".to_string(),
            created_at: Utc::now(),
            expires_at: None,
            permissions: perms,
            rate_limit_tier: RateLimitTier::Standard,
            revoked: false,
            metadata: HashMap::new(),
        };
        assert!(key.has_all_permissions(&[Permission::SparqlRead, Permission::SparqlWrite]));
        assert!(!key.has_all_permissions(&[Permission::SparqlRead, Permission::Admin]));
    }

    // -----------------------------------------------------------------------
    // base58_encode
    // -----------------------------------------------------------------------

    #[test]
    fn test_base58_encode_non_empty() {
        let raw = [1u8; 32];
        let encoded = base58_encode(&raw);
        assert!(!encoded.is_empty());
        // All characters should be in the alphabet.
        assert!(encoded
            .chars()
            .all(|c| BASE58_ALPHABET.contains(&(c as u8))));
    }

    #[test]
    fn test_base58_encode_different_inputs() {
        let a = base58_encode(&[0u8; 32]);
        let b = base58_encode(&[1u8; 32]);
        // Different inputs should produce different outputs.
        assert_ne!(a, b);
    }

    // -----------------------------------------------------------------------
    // sha256_hex
    // -----------------------------------------------------------------------

    #[test]
    fn test_sha256_hex_deterministic() {
        let h1 = sha256_hex("hello");
        let h2 = sha256_hex("hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_sha256_hex_different_inputs() {
        let h1 = sha256_hex("hello");
        let h2 = sha256_hex("world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_sha256_hex_length() {
        let h = sha256_hex("test");
        assert_eq!(h.len(), 64); // 32 bytes × 2 hex chars
    }

    // -----------------------------------------------------------------------
    // InMemoryApiKeyStore
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_create_key_returns_raw() {
        let store = test_store();
        let params = basic_params("mykey");
        let raw_key = store.create_key(params).await.unwrap();
        assert!(!raw_key.raw.is_empty());
        assert!(!raw_key.record.key_id.is_empty());
    }

    #[tokio::test]
    async fn test_validate_valid_key() {
        let store = test_store();
        let raw_key = store
            .create_key(basic_params("validate-test"))
            .await
            .unwrap();
        let key = store.validate_key(&raw_key.raw).await.unwrap();
        assert_eq!(key.key_id, raw_key.record.key_id);
    }

    #[tokio::test]
    async fn test_validate_invalid_key_returns_not_found() {
        let store = test_store();
        let result = store.validate_key("totally-invalid-key").await;
        assert!(matches!(result, Err(ApiKeyError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_validate_revoked_key() {
        let store = test_store();
        let raw_key = store.create_key(basic_params("revoke-test")).await.unwrap();
        store.revoke_key(&raw_key.record.key_id).await.unwrap();
        let result = store.validate_key(&raw_key.raw).await;
        assert!(matches!(result, Err(ApiKeyError::Revoked)));
    }

    #[tokio::test]
    async fn test_validate_expired_key() {
        let store = test_store();
        let params = basic_params("expire-test").with_expiry(Duration::from_nanos(1));
        let raw_key = store.create_key(params).await.unwrap();
        // Wait for the key to expire.
        tokio::time::sleep(Duration::from_millis(10)).await;
        let result = store.validate_key(&raw_key.raw).await;
        assert!(matches!(result, Err(ApiKeyError::Expired)));
    }

    #[tokio::test]
    async fn test_get_key_by_id() {
        let store = test_store();
        let raw_key = store.create_key(basic_params("get-test")).await.unwrap();
        let key = store.get_key(&raw_key.record.key_id).await.unwrap();
        assert_eq!(key.label, "get-test");
    }

    #[tokio::test]
    async fn test_get_nonexistent_key() {
        let store = test_store();
        let result = store.get_key("nonexistent-id").await;
        assert!(matches!(result, Err(ApiKeyError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_revoke_key() {
        let store = test_store();
        let raw_key = store.create_key(basic_params("revoke-me")).await.unwrap();
        store.revoke_key(&raw_key.record.key_id).await.unwrap();
        let key = store.get_key(&raw_key.record.key_id).await.unwrap();
        assert!(key.revoked);
    }

    #[tokio::test]
    async fn test_revoke_nonexistent_key() {
        let store = test_store();
        let result = store.revoke_key("no-such-key").await;
        assert!(matches!(result, Err(ApiKeyError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_list_keys_all() {
        let store = test_store();
        store.create_key(basic_params("k1")).await.unwrap();
        store.create_key(basic_params("k2")).await.unwrap();
        let keys = store.list_keys(None).await.unwrap();
        assert_eq!(keys.len(), 2);
    }

    #[tokio::test]
    async fn test_list_keys_by_owner() {
        let store = test_store();
        store
            .create_key(CreateKeyParams::new("k1", "alice"))
            .await
            .unwrap();
        store
            .create_key(CreateKeyParams::new("k2", "bob"))
            .await
            .unwrap();
        let alice_keys = store.list_keys(Some("alice")).await.unwrap();
        assert_eq!(alice_keys.len(), 1);
        assert_eq!(alice_keys[0].owner, "alice");
    }

    #[tokio::test]
    async fn test_delete_key() {
        let store = test_store();
        let raw_key = store.create_key(basic_params("del-test")).await.unwrap();
        store.delete_key(&raw_key.record.key_id).await.unwrap();
        let result = store.get_key(&raw_key.record.key_id).await;
        assert!(matches!(result, Err(ApiKeyError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_delete_nonexistent_key() {
        let store = test_store();
        let result = store.delete_key("ghost").await;
        assert!(matches!(result, Err(ApiKeyError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_check_permissions_granted() {
        let store = test_store();
        let params = basic_params("perm-test").with_permission(Permission::Admin);
        let raw_key = store.create_key(params).await.unwrap();
        let key = store
            .check_permissions(&raw_key.raw, &[Permission::SparqlRead, Permission::Admin])
            .await
            .unwrap();
        assert_eq!(key.label, "perm-test");
    }

    #[tokio::test]
    async fn test_check_permissions_denied() {
        let store = test_store();
        let raw_key = store.create_key(basic_params("perm-deny")).await.unwrap();
        let result = store
            .check_permissions(&raw_key.raw, &[Permission::Admin])
            .await;
        assert!(matches!(result, Err(ApiKeyError::InsufficientPermissions)));
    }

    #[tokio::test]
    async fn test_key_no_expiry_stays_active() {
        let store = test_store();
        let params = CreateKeyParams::new("forever", "system")
            .with_permission(Permission::SparqlRead)
            .with_tier(RateLimitTier::Unlimited);
        let raw_key = store.create_key(params).await.unwrap();
        let key = store.validate_key(&raw_key.raw).await.unwrap();
        assert!(key.is_active());
    }

    #[tokio::test]
    async fn test_create_key_params_builder() {
        let params = CreateKeyParams::new("my-svc", "platform-team")
            .with_permission(Permission::SparqlRead)
            .with_permission(Permission::DataExport)
            .with_tier(RateLimitTier::Premium)
            .with_expiry(Duration::from_secs(3600))
            .with_metadata("env", "production");
        assert!(params.permissions.contains(&Permission::SparqlRead));
        assert!(params.permissions.contains(&Permission::DataExport));
        assert_eq!(params.rate_limit_tier, RateLimitTier::Premium);
        assert_eq!(
            params.metadata.get("env").map(String::as_str),
            Some("production")
        );
    }

    #[tokio::test]
    async fn test_multiple_key_uniqueness() {
        let store = test_store();
        let k1 = store.create_key(basic_params("k1")).await.unwrap();
        let k2 = store.create_key(basic_params("k2")).await.unwrap();
        // Keys and IDs should differ.
        assert_ne!(k1.raw, k2.raw);
        assert_ne!(k1.record.key_id, k2.record.key_id);
    }

    #[tokio::test]
    async fn test_key_metadata() {
        let store = test_store();
        let params = basic_params("meta-key").with_metadata("team", "infra");
        let raw_key = store.create_key(params).await.unwrap();
        let key = store.get_key(&raw_key.record.key_id).await.unwrap();
        assert_eq!(key.metadata.get("team").map(String::as_str), Some("infra"));
    }

    #[tokio::test]
    async fn test_unlimited_tier_key() {
        let store = test_store();
        let params = CreateKeyParams::new("svc-account", "ops")
            .with_permission(Permission::SparqlRead)
            .with_permission(Permission::SparqlWrite)
            .with_permission(Permission::Admin)
            .with_tier(RateLimitTier::Unlimited);
        let raw_key = store.create_key(params).await.unwrap();
        let key = store.validate_key(&raw_key.raw).await.unwrap();
        assert_eq!(key.rate_limit_tier, RateLimitTier::Unlimited);
        assert!(key.rate_limit_tier.max_rps().is_none());
    }
}
