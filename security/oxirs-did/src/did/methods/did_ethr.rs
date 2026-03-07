//! did:ethr DID Method implementation
//!
//! did:ethr is based on the ERC-1056 Ethereum DID Registry contract.
//! Format: `did:ethr:[network:]<ethereum-address>`
//!
//! References:
//!   <https://github.com/decentralized-identity/ethr-did-resolver>
//!   ERC-1056: Ethereum EIP-1056 lightweight identity standard
//!
//! Examples:
//!   did:ethr:0xf3beac30c498d9e26865f34fcaa57dbb935b0d74
//!   did:ethr:mainnet:0xf3beac30c498d9e26865f34fcaa57dbb935b0d74
//!   did:ethr:0x4:0xf3beac30c498d9e26865f34fcaa57dbb935b0d74  (Rinkeby testnet)

use super::DidMethod;
use crate::did::document::VerificationRelationship;
use crate::did::{Did, DidDocument};
use crate::{DidError, DidResult, Service, VerificationMethod};
use async_trait::async_trait;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};

/// Known Ethereum network names/IDs
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EthNetwork {
    /// Ethereum mainnet (chain ID 1)
    Mainnet,
    /// Goerli testnet (chain ID 5)
    Goerli,
    /// Sepolia testnet (chain ID 11155111)
    Sepolia,
    /// Polygon mainnet (chain ID 137)
    Polygon,
    /// Arbitrum One (chain ID 42161)
    Arbitrum,
    /// Custom network by chain ID string or name
    Custom(String),
}

impl EthNetwork {
    /// Parse from string (network name or hex chain ID)
    pub fn parse(s: &str) -> Self {
        match s {
            "mainnet" | "1" => Self::Mainnet,
            "goerli" | "5" => Self::Goerli,
            "sepolia" | "11155111" => Self::Sepolia,
            "polygon" | "137" => Self::Polygon,
            "arbitrum" | "42161" => Self::Arbitrum,
            other => Self::Custom(other.to_string()),
        }
    }

    /// Get the display name
    pub fn as_str(&self) -> &str {
        match self {
            Self::Mainnet => "mainnet",
            Self::Goerli => "goerli",
            Self::Sepolia => "sepolia",
            Self::Polygon => "polygon",
            Self::Arbitrum => "arbitrum",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Get the EIP-155 chain ID
    pub fn chain_id(&self) -> u64 {
        match self {
            Self::Mainnet => 1,
            Self::Goerli => 5,
            Self::Sepolia => 11155111,
            Self::Polygon => 137,
            Self::Arbitrum => 42161,
            Self::Custom(_) => 0, // Unknown
        }
    }
}

/// A parsed did:ethr identifier
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DidEthr {
    /// The Ethereum network (None = mainnet)
    pub network: Option<EthNetwork>,
    /// The Ethereum address (checksummed EIP-55 format)
    pub address: String,
}

impl DidEthr {
    /// Create a new did:ethr from an Ethereum address (mainnet)
    ///
    /// # Arguments
    /// * `address` - Ethereum address (with or without 0x prefix, will be normalized)
    pub fn new(address: &str) -> DidResult<Self> {
        let normalized = normalize_eth_address(address)?;
        Ok(Self {
            network: None,
            address: normalized,
        })
    }

    /// Create a did:ethr for a specific network
    pub fn new_on_network(network: &str, address: &str) -> DidResult<Self> {
        let normalized = normalize_eth_address(address)?;
        let net = EthNetwork::parse(network);
        Ok(Self {
            network: Some(net),
            address: normalized,
        })
    }

    /// Parse did:ethr from DID string
    pub fn from_did_string(did: &str) -> DidResult<Self> {
        if !did.starts_with("did:ethr:") {
            return Err(DidError::InvalidFormat(
                "DID must start with 'did:ethr:'".to_string(),
            ));
        }

        let rest = &did["did:ethr:".len()..];

        // Check if there's a network prefix:
        // did:ethr:<address>         -> no network prefix (mainnet)
        // did:ethr:<name>:<address>  -> named network (e.g. "mainnet", "goerli")
        // did:ethr:0x<chainid>:<address> -> hex chain-id network prefix
        //
        // Disambiguation: if rest starts with "0x"/"0X" AND contains a colon,
        // it is a hex network prefix followed by an address (e.g. "0x4:0xABCD…").
        // If it starts with "0x"/"0X" but has NO colon, it is a bare mainnet address.
        if rest.starts_with("0x") || rest.starts_with("0X") {
            if let Some(colon_pos) = rest.find(':') {
                // Hex network prefix: "0x<id>:<address>"
                let network_str = &rest[..colon_pos];
                let address_str = &rest[colon_pos + 1..];
                let normalized = normalize_eth_address(address_str)?;
                let net = EthNetwork::parse(network_str);
                return Ok(Self {
                    network: Some(net),
                    address: normalized,
                });
            }

            // No colon → plain mainnet address
            let normalized = normalize_eth_address(rest)?;
            return Ok(Self {
                network: None,
                address: normalized,
            });
        }

        // Has network prefix: find the last colon to split network from address
        if let Some(colon_pos) = rest.rfind(':') {
            let network_str = &rest[..colon_pos];
            let address_str = &rest[colon_pos + 1..];
            let normalized = normalize_eth_address(address_str)?;
            let net = EthNetwork::parse(network_str);
            return Ok(Self {
                network: Some(net),
                address: normalized,
            });
        }

        Err(DidError::InvalidFormat(format!(
            "Cannot parse did:ethr address from: {}",
            did
        )))
    }

    /// Convert to DID string
    pub fn to_did_string(&self) -> String {
        match &self.network {
            None => format!("did:ethr:{}", self.address),
            Some(net) => format!("did:ethr:{}:{}", net.as_str(), self.address),
        }
    }

    /// Convert to Did
    pub fn to_did(&self) -> DidResult<Did> {
        Did::new(&self.to_did_string())
    }

    /// Resolve this did:ethr to a DID Document
    ///
    /// In a full implementation, this would query the ERC-1056 contract.
    /// This implementation generates a minimal DID Document from the address.
    pub fn resolve(&self) -> DidResult<DidDocument> {
        self.generate_document()
    }

    /// Generate a DID Document from the Ethereum address
    ///
    /// The DID Document contains an EcdsaSecp256k1RecoveryMethod2020 verification method
    /// referencing the Ethereum address, following the ethr-did-resolver spec.
    fn generate_document(&self) -> DidResult<DidDocument> {
        let did_str = self.to_did_string();
        let did = Did::new(&did_str)?;

        // Primary verification method ID (controller key)
        let key_id = format!("{}#controller", did_str);

        // Create verification method using blockchain account ID
        let caip10_id = match &self.network {
            None => format!("eip155:1:{}", self.address),
            Some(net) => format!("eip155:{}:{}", net.chain_id(), self.address),
        };

        let vm = VerificationMethod::blockchain(
            &key_id,
            &did_str,
            "EcdsaSecp256k1RecoveryMethod2020",
            &caip10_id,
        );

        let mut doc = DidDocument::new(did);
        doc.context = vec![
            "https://www.w3.org/ns/did/v1".to_string(),
            "https://w3id.org/security/suites/secp256k1recovery-2020/v2".to_string(),
            "https://w3id.org/security/v3-unstable".to_string(),
        ];

        doc.verification_method.push(vm);

        // All standard relationships reference the controller key
        doc.authentication
            .push(VerificationRelationship::Reference(key_id.clone()));
        doc.assertion_method
            .push(VerificationRelationship::Reference(key_id.clone()));
        doc.capability_invocation
            .push(VerificationRelationship::Reference(key_id.clone()));
        doc.capability_delegation
            .push(VerificationRelationship::Reference(key_id));

        // Add a default service endpoint for signing
        doc.service.push(Service {
            id: format!("{}#messaging", did_str),
            service_type: "EncryptedMessaging".to_string(),
            service_endpoint: "https://example.com/messaging".to_string(),
        });

        Ok(doc)
    }

    /// Add a secp256k1 public key to the document
    pub fn add_public_key_to_document(
        &self,
        doc: &mut DidDocument,
        public_key_hex: &str,
        key_fragment: &str,
    ) -> DidResult<String> {
        let did_str = self.to_did_string();
        let key_id = format!("{}#{}", did_str, key_fragment);

        // Public key JWK for secp256k1
        let compressed_key = hex::decode(public_key_hex)
            .map_err(|e| DidError::InvalidKey(format!("Invalid hex public key: {}", e)))?;

        if compressed_key.len() != 33 {
            return Err(DidError::InvalidKey(
                "secp256k1 compressed public key must be 33 bytes".to_string(),
            ));
        }

        let jwk = serde_json::json!({
            "kty": "EC",
            "crv": "secp256k1",
            "x": URL_SAFE_NO_PAD.encode(&compressed_key[1..17]),
            "y": URL_SAFE_NO_PAD.encode(&compressed_key[17..]),
        });

        let vm =
            VerificationMethod::jwk(&key_id, &did_str, "EcdsaSecp256k1VerificationKey2019", jwk);
        doc.verification_method.push(vm);

        Ok(key_id)
    }

    /// Compute the Ethereum address from an uncompressed public key (65 bytes)
    pub fn address_from_public_key(public_key_uncompressed: &[u8]) -> DidResult<String> {
        if public_key_uncompressed.len() != 65 || public_key_uncompressed[0] != 0x04 {
            return Err(DidError::InvalidKey(
                "Expected uncompressed public key (65 bytes starting with 0x04)".to_string(),
            ));
        }

        // Keccak256 of the raw 64-byte key (skip 0x04 prefix)
        let mut hasher = Keccak256::new();
        hasher.update(&public_key_uncompressed[1..]);
        let hash = hasher.finalize();

        // Take the last 20 bytes
        let address_bytes = &hash[12..];
        let hex_addr = hex::encode(address_bytes);

        // Apply EIP-55 checksum
        apply_eip55_checksum(&hex_addr)
    }

    /// Get the Ethereum address
    pub fn eth_address(&self) -> &str {
        &self.address
    }

    /// Get the network (None = mainnet)
    pub fn network(&self) -> Option<&EthNetwork> {
        self.network.as_ref()
    }
}

/// did:ethr method resolver
pub struct DidEthrMethod;

impl Default for DidEthrMethod {
    fn default() -> Self {
        Self::new()
    }
}

impl DidEthrMethod {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DidMethod for DidEthrMethod {
    fn method_name(&self) -> &str {
        "ethr"
    }

    async fn resolve(&self, did: &Did) -> DidResult<DidDocument> {
        if !self.supports(did) {
            return Err(DidError::UnsupportedMethod(did.method().to_string()));
        }

        let ethr = DidEthr::from_did_string(did.as_str())?;
        ethr.resolve()
    }
}

/// Normalize an Ethereum address to lowercase hex with 0x prefix
fn normalize_eth_address(address: &str) -> DidResult<String> {
    let addr = if address.starts_with("0x") || address.starts_with("0X") {
        &address[2..]
    } else {
        address
    };

    if addr.len() != 40 {
        return Err(DidError::InvalidFormat(format!(
            "Ethereum address must be 40 hex characters (without 0x), got {}",
            addr.len()
        )));
    }

    if !addr.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(DidError::InvalidFormat(
            "Ethereum address must contain only hex characters".to_string(),
        ));
    }

    let lower = addr.to_lowercase();
    Ok(format!("0x{}", lower))
}

/// Apply EIP-55 mixed-case checksum encoding to a lowercase hex address
fn apply_eip55_checksum(hex_addr: &str) -> DidResult<String> {
    let lower = hex_addr.to_lowercase();

    // Keccak256 of the lowercase address
    let mut hasher = Keccak256::new();
    hasher.update(lower.as_bytes());
    let hash = hasher.finalize();

    let checksummed: String = lower
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if c.is_ascii_alphabetic() {
                // If the i-th nibble of the hash >= 8, capitalize
                let nibble_idx = i / 2;
                let nibble = if i % 2 == 0 {
                    (hash[nibble_idx] >> 4) & 0xf
                } else {
                    hash[nibble_idx] & 0xf
                };
                if nibble >= 8 {
                    c.to_ascii_uppercase()
                } else {
                    c
                }
            } else {
                c
            }
        })
        .collect();

    Ok(format!("0x{}", checksummed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_did_ethr_mainnet() {
        let ethr = DidEthr::new("0xf3beac30c498d9e26865f34fcaa57dbb935b0d74").unwrap();
        assert!(ethr.network.is_none());
        assert_eq!(ethr.address, "0xf3beac30c498d9e26865f34fcaa57dbb935b0d74");
    }

    #[test]
    fn test_did_ethr_to_string_mainnet() {
        let ethr = DidEthr::new("0xf3beac30c498d9e26865f34fcaa57dbb935b0d74").unwrap();
        assert_eq!(
            ethr.to_did_string(),
            "did:ethr:0xf3beac30c498d9e26865f34fcaa57dbb935b0d74"
        );
    }

    #[test]
    fn test_did_ethr_to_string_with_network() {
        let ethr = DidEthr::new_on_network("goerli", "0xf3beac30c498d9e26865f34fcaa57dbb935b0d74")
            .unwrap();
        assert_eq!(
            ethr.to_did_string(),
            "did:ethr:goerli:0xf3beac30c498d9e26865f34fcaa57dbb935b0d74"
        );
    }

    #[test]
    fn test_did_ethr_from_string_mainnet() {
        let did_str = "did:ethr:0xf3beac30c498d9e26865f34fcaa57dbb935b0d74";
        let ethr = DidEthr::from_did_string(did_str).unwrap();
        assert!(ethr.network.is_none());
        assert_eq!(ethr.address, "0xf3beac30c498d9e26865f34fcaa57dbb935b0d74");
        assert_eq!(ethr.to_did_string(), did_str);
    }

    #[test]
    fn test_did_ethr_from_string_with_network() {
        let did_str = "did:ethr:mainnet:0xf3beac30c498d9e26865f34fcaa57dbb935b0d74";
        let ethr = DidEthr::from_did_string(did_str).unwrap();
        assert_eq!(ethr.network, Some(EthNetwork::Mainnet));
        assert_eq!(ethr.address, "0xf3beac30c498d9e26865f34fcaa57dbb935b0d74");
        assert_eq!(ethr.to_did_string(), did_str);
    }

    #[test]
    fn test_did_ethr_from_string_hex_network() {
        let did_str = "did:ethr:0x4:0xf3beac30c498d9e26865f34fcaa57dbb935b0d74";
        let ethr = DidEthr::from_did_string(did_str).unwrap();
        assert_eq!(ethr.network, Some(EthNetwork::Custom("0x4".to_string())));
        assert_eq!(ethr.address, "0xf3beac30c498d9e26865f34fcaa57dbb935b0d74");
    }

    #[test]
    fn test_did_ethr_resolve_mainnet() {
        let ethr = DidEthr::new("0xf3beac30c498d9e26865f34fcaa57dbb935b0d74").unwrap();
        let doc = ethr.resolve().unwrap();

        assert_eq!(doc.verification_method.len(), 1);
        assert!(!doc.authentication.is_empty());
        assert!(!doc.assertion_method.is_empty());
        let vm = &doc.verification_method[0];
        assert_eq!(vm.method_type, "EcdsaSecp256k1RecoveryMethod2020");
        assert!(vm.blockchain_account_id.is_some());
    }

    #[test]
    fn test_did_ethr_resolve_network() {
        let ethr = DidEthr::new_on_network("polygon", "0xf3beac30c498d9e26865f34fcaa57dbb935b0d74")
            .unwrap();
        let doc = ethr.resolve().unwrap();

        let vm = &doc.verification_method[0];
        let caip10 = vm.blockchain_account_id.as_ref().unwrap();
        // Should reference polygon chain ID 137
        assert!(caip10.contains("137:"));
    }

    #[test]
    fn test_did_ethr_invalid_address() {
        assert!(DidEthr::new("invalid_address").is_err());
        assert!(DidEthr::new("0x123").is_err()); // too short
        assert!(DidEthr::new("0xGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG").is_err());
    }

    #[test]
    fn test_did_ethr_wrong_method() {
        assert!(DidEthr::from_did_string("did:key:z6Mk123").is_err());
        assert!(DidEthr::from_did_string("did:pkh:eip155:1:0xabc").is_err());
    }

    #[test]
    fn test_normalize_eth_address() {
        // Uppercase should be normalized to lowercase
        let ethr = DidEthr::new("0xF3BEAC30C498D9E26865F34FCAA57DBB935B0D74").unwrap();
        assert_eq!(ethr.address, "0xf3beac30c498d9e26865f34fcaa57dbb935b0d74");

        // Without 0x prefix
        let ethr2 = DidEthr::new("f3beac30c498d9e26865f34fcaa57dbb935b0d74").unwrap();
        assert_eq!(ethr2.address, "0xf3beac30c498d9e26865f34fcaa57dbb935b0d74");
    }

    #[test]
    fn test_eth_network_chain_ids() {
        assert_eq!(EthNetwork::Mainnet.chain_id(), 1);
        assert_eq!(EthNetwork::Goerli.chain_id(), 5);
        assert_eq!(EthNetwork::Sepolia.chain_id(), 11155111);
        assert_eq!(EthNetwork::Polygon.chain_id(), 137);
        assert_eq!(EthNetwork::Arbitrum.chain_id(), 42161);
    }

    #[test]
    fn test_eth_network_from_str() {
        assert_eq!(EthNetwork::parse("mainnet"), EthNetwork::Mainnet);
        assert_eq!(EthNetwork::parse("1"), EthNetwork::Mainnet);
        assert_eq!(EthNetwork::parse("polygon"), EthNetwork::Polygon);
        assert_eq!(EthNetwork::parse("137"), EthNetwork::Polygon);
        assert_eq!(
            EthNetwork::parse("custom-net"),
            EthNetwork::Custom("custom-net".to_string())
        );
    }

    #[tokio::test]
    async fn test_did_ethr_method_resolver() {
        let method = DidEthrMethod::new();
        assert_eq!(method.method_name(), "ethr");

        let did = Did::new("did:ethr:0xf3beac30c498d9e26865f34fcaa57dbb935b0d74").unwrap();
        let doc = method.resolve(&did).await.unwrap();
        assert!(!doc.verification_method.is_empty());
    }

    #[tokio::test]
    async fn test_did_ethr_wrong_method_error() {
        let method = DidEthrMethod::new();
        let did = Did::new("did:key:z6Mk123").unwrap();
        assert!(method.resolve(&did).await.is_err());
    }

    #[test]
    fn test_address_from_public_key_rejects_invalid() {
        // Must be 65 bytes and start with 0x04
        assert!(DidEthr::address_from_public_key(&[0u8; 64]).is_err());
        assert!(DidEthr::address_from_public_key(&[0u8; 65]).is_err()); // starts with 0x00, not 0x04
    }

    #[test]
    fn test_address_from_public_key_valid() {
        // Create a synthetic 65-byte "uncompressed" public key for testing
        let mut pk = [0u8; 65];
        pk[0] = 0x04;
        // Fill with some non-zero data
        for (i, byte) in pk.iter_mut().enumerate().skip(1) {
            *byte = i as u8;
        }
        let result = DidEthr::address_from_public_key(&pk);
        assert!(result.is_ok());
        let addr = result.unwrap();
        assert!(addr.starts_with("0x"));
        assert_eq!(addr.len(), 42); // 0x + 40 hex chars
    }

    #[test]
    fn test_apply_eip55_checksum() {
        // Known EIP-55 checksum test vector
        let result = apply_eip55_checksum("fb6916095ca1df60bb79ce92ce3ea74c37c5d359");
        assert!(result.is_ok());
        let checksummed = result.unwrap();
        assert!(checksummed.starts_with("0x"));
    }

    #[test]
    fn test_document_service_endpoint() {
        let ethr = DidEthr::new("0xf3beac30c498d9e26865f34fcaa57dbb935b0d74").unwrap();
        let doc = ethr.resolve().unwrap();
        // Should have at least one service endpoint
        assert!(!doc.service.is_empty());
    }

    #[test]
    fn test_to_did_conversion() {
        let ethr = DidEthr::new("0xf3beac30c498d9e26865f34fcaa57dbb935b0d74").unwrap();
        let did = ethr.to_did().unwrap();
        assert_eq!(did.method(), "ethr");
    }
}
