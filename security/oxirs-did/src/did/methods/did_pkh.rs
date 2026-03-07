//! did:pkh DID Method implementation
//!
//! did:pkh is a DID method based on blockchain account addresses (EIP-2844 standard).
//! Format: `did:pkh:<CAIP-2 chain namespace>:<CAIP-2 account address>`
//!
//! Examples:
//!   did:pkh:eip155:1:0xab16a96d359ec26a11e2c2b3d8f8b8942d5bfcdb  (Ethereum mainnet)
//!   did:pkh:solana:4sGjMW1sUnHzSxGspuhpqLDx6wiyjNtZ:base58addr  (Solana)
//!   did:pkh:bip122:000000000019d6689c085ae165831e93:bitcoinaddr  (Bitcoin mainnet)

use super::DidMethod;
use crate::did::{Did, DidDocument};
use crate::{DidError, DidResult, Service, VerificationMethod};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Supported blockchain namespaces (CAIP-2)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChainNamespace {
    /// Ethereum and EVM-compatible chains (EIP-155)
    Eip155,
    /// Solana
    Solana,
    /// Bitcoin (BIP-122)
    Bip122,
    /// Cosmos/Tendermint
    Cosmos,
    /// Polkadot
    Polkadot,
    /// Other/custom namespaces
    Other(String),
}

impl ChainNamespace {
    /// Parse from string
    pub fn parse(s: &str) -> Self {
        match s {
            "eip155" => Self::Eip155,
            "solana" => Self::Solana,
            "bip122" => Self::Bip122,
            "cosmos" => Self::Cosmos,
            "polkadot" => Self::Polkadot,
            other => Self::Other(other.to_string()),
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &str {
        match self {
            Self::Eip155 => "eip155",
            Self::Solana => "solana",
            Self::Bip122 => "bip122",
            Self::Cosmos => "cosmos",
            Self::Polkadot => "polkadot",
            Self::Other(s) => s.as_str(),
        }
    }

    /// Get the verification method type for this chain
    pub fn verification_method_type(&self) -> &'static str {
        match self {
            Self::Eip155 => "EcdsaSecp256k1RecoveryMethod2020",
            Self::Solana => "Ed25519VerificationKey2020",
            Self::Bip122 => "EcdsaSecp256k1VerificationKey2019",
            Self::Cosmos => "EcdsaSecp256k1VerificationKey2019",
            Self::Polkadot => "Sr25519VerificationKey2020",
            Self::Other(_) => "BlockchainVerificationMethod2021",
        }
    }

    /// Validate account address format for this namespace
    pub fn validate_address(&self, address: &str) -> DidResult<()> {
        match self {
            Self::Eip155 => validate_eip155_address(address),
            Self::Solana => validate_solana_address(address),
            Self::Bip122 => validate_bitcoin_address(address),
            Self::Cosmos | Self::Polkadot | Self::Other(_) => {
                // Basic validation - non-empty
                if address.is_empty() {
                    return Err(DidError::InvalidFormat(
                        "Account address cannot be empty".to_string(),
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Validate Ethereum address (EIP-55 checksum or lowercase hex)
fn validate_eip155_address(address: &str) -> DidResult<()> {
    if !address.starts_with("0x") && !address.starts_with("0X") {
        return Err(DidError::InvalidFormat(
            "Ethereum address must start with '0x'".to_string(),
        ));
    }

    let hex_part = &address[2..];
    if hex_part.len() != 40 {
        return Err(DidError::InvalidFormat(format!(
            "Ethereum address hex part must be 40 characters, got {}",
            hex_part.len()
        )));
    }

    if !hex_part.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(DidError::InvalidFormat(
            "Ethereum address must contain only hex characters".to_string(),
        ));
    }

    Ok(())
}

/// Validate Solana address (base58 encoded 32-byte public key)
fn validate_solana_address(address: &str) -> DidResult<()> {
    if address.is_empty() {
        return Err(DidError::InvalidFormat(
            "Solana address cannot be empty".to_string(),
        ));
    }

    // Solana addresses are base58-encoded 32-byte public keys (typically 32-44 chars)
    let decoded = bs58::decode(address)
        .into_vec()
        .map_err(|e| DidError::InvalidFormat(format!("Invalid base58 address: {}", e)))?;

    if decoded.len() != 32 {
        return Err(DidError::InvalidFormat(format!(
            "Solana public key must be 32 bytes, got {}",
            decoded.len()
        )));
    }

    Ok(())
}

/// Validate Bitcoin address (basic format check)
fn validate_bitcoin_address(address: &str) -> DidResult<()> {
    if address.is_empty() {
        return Err(DidError::InvalidFormat(
            "Bitcoin address cannot be empty".to_string(),
        ));
    }

    // P2PKH addresses start with '1', P2SH with '3', bech32 with 'bc1'
    let valid_prefix = address.starts_with('1')
        || address.starts_with('3')
        || address.starts_with("bc1")
        || address.starts_with("tb1");

    if !valid_prefix {
        return Err(DidError::InvalidFormat(
            "Bitcoin address must start with '1', '3', 'bc1', or 'tb1'".to_string(),
        ));
    }

    // Length checks
    if address.len() < 25 || address.len() > 62 {
        return Err(DidError::InvalidFormat(format!(
            "Bitcoin address length {} is out of valid range [25, 62]",
            address.len()
        )));
    }

    Ok(())
}

/// did:pkh DID representation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DidPkh {
    /// CAIP-2 chain namespace (e.g., "eip155")
    pub chain_namespace: ChainNamespace,
    /// CAIP-2 chain ID (e.g., "1" for Ethereum mainnet)
    pub chain_id: String,
    /// Blockchain account address (e.g., "0xab16...")
    pub account_address: String,
}

impl DidPkh {
    /// Create a new did:pkh from components
    ///
    /// # Arguments
    /// * `chain_namespace` - The CAIP-2 chain namespace (e.g., "eip155")
    /// * `chain_id` - The CAIP-2 chain ID (e.g., "1" for Ethereum mainnet)
    /// * `account_address` - The blockchain account address
    pub fn new(chain_namespace: &str, chain_id: &str, account_address: &str) -> DidResult<Self> {
        if chain_namespace.is_empty() {
            return Err(DidError::InvalidFormat(
                "Chain namespace cannot be empty".to_string(),
            ));
        }
        if chain_id.is_empty() {
            return Err(DidError::InvalidFormat(
                "Chain ID cannot be empty".to_string(),
            ));
        }
        if account_address.is_empty() {
            return Err(DidError::InvalidFormat(
                "Account address cannot be empty".to_string(),
            ));
        }

        let namespace = ChainNamespace::parse(chain_namespace);

        // Validate address format based on chain namespace
        namespace.validate_address(account_address)?;

        Ok(Self {
            chain_namespace: namespace,
            chain_id: chain_id.to_string(),
            account_address: account_address.to_string(),
        })
    }

    /// Create an Ethereum did:pkh
    pub fn ethereum(chain_id: u64, address: &str) -> DidResult<Self> {
        Self::new("eip155", &chain_id.to_string(), address)
    }

    /// Create a Solana did:pkh
    pub fn solana(address: &str) -> DidResult<Self> {
        // Solana uses a specific chain reference
        Self::new("solana", "4sGjMW1sUnHzSxGspuhpqLDx6wiyjNtZ", address)
    }

    /// Convert to DID string
    pub fn to_did_string(&self) -> String {
        format!(
            "did:pkh:{}:{}:{}",
            self.chain_namespace.as_str(),
            self.chain_id,
            self.account_address
        )
    }

    /// Parse from DID string
    pub fn from_did_string(did: &str) -> DidResult<Self> {
        if !did.starts_with("did:pkh:") {
            return Err(DidError::InvalidFormat(
                "DID must start with 'did:pkh:'".to_string(),
            ));
        }

        let rest = &did["did:pkh:".len()..];
        // Format: <namespace>:<chain_id>:<address>
        // Note: chain_id and address may contain colons in some edge cases,
        // so we split on the first two colons only
        let parts: Vec<&str> = rest.splitn(3, ':').collect();

        if parts.len() != 3 {
            return Err(DidError::InvalidFormat(format!(
                "did:pkh must have format did:pkh:<namespace>:<chain_id>:<address>, got: {}",
                did
            )));
        }

        Self::new(parts[0], parts[1], parts[2])
    }

    /// Convert to a Did object
    pub fn to_did(&self) -> DidResult<Did> {
        Did::new(&self.to_did_string())
    }

    /// Resolve to a DID Document
    pub fn resolve(&self) -> DidResult<DidDocument> {
        self.generate_document()
    }

    /// Generate a DID Document from the blockchain address
    ///
    /// For blockchain addresses, we generate a DID document that references
    /// the account address. Since we don't have the public key directly
    /// (only the address), the verification method uses the blockchain
    /// account as the cryptographic anchor.
    fn generate_document(&self) -> DidResult<DidDocument> {
        let did_str = self.to_did_string();
        let did = Did::new(&did_str)?;
        let key_id = format!("{}#blockchainAccountId", did_str);
        let vm_type = self.chain_namespace.verification_method_type();

        // Construct blockchain account ID in CAIP-10 format
        let blockchain_account_id = format!(
            "{}:{}:{}",
            self.chain_namespace.as_str(),
            self.chain_id,
            self.account_address
        );

        let verification_method =
            VerificationMethod::blockchain(&key_id, &did_str, vm_type, &blockchain_account_id);

        // Build DID Document
        let mut doc = DidDocument::new(did);

        // Add appropriate JSON-LD contexts
        doc.context = vec![
            "https://www.w3.org/ns/did/v1".to_string(),
            "https://w3id.org/security/suites/secp256k1recovery-2020/v2".to_string(),
            "https://w3id.org/security/v3-unstable".to_string(),
        ];

        doc.verification_method.push(verification_method);

        // Add verification relationships
        use crate::did::document::VerificationRelationship;
        doc.authentication
            .push(VerificationRelationship::Reference(key_id.clone()));
        doc.assertion_method
            .push(VerificationRelationship::Reference(key_id.clone()));
        doc.capability_invocation
            .push(VerificationRelationship::Reference(key_id.clone()));
        doc.capability_delegation
            .push(VerificationRelationship::Reference(key_id));

        Ok(doc)
    }

    /// Get the blockchain account identifier in CAIP-10 format
    pub fn caip10_account_id(&self) -> String {
        format!(
            "{}:{}:{}",
            self.chain_namespace.as_str(),
            self.chain_id,
            self.account_address
        )
    }
}

/// did:pkh method resolver
pub struct DidPkhMethod;

impl Default for DidPkhMethod {
    fn default() -> Self {
        Self::new()
    }
}

impl DidPkhMethod {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DidMethod for DidPkhMethod {
    fn method_name(&self) -> &str {
        "pkh"
    }

    async fn resolve(&self, did: &Did) -> DidResult<DidDocument> {
        if !self.supports(did) {
            return Err(DidError::UnsupportedMethod(did.method().to_string()));
        }

        let pkh = DidPkh::from_did_string(did.as_str())?;
        pkh.resolve()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_ethereum_did_pkh() {
        let pkh = DidPkh::ethereum(1, "0xab16a96d359ec26a11e2c2b3d8f8b8942d5bfcdb").unwrap();
        assert_eq!(pkh.chain_namespace, ChainNamespace::Eip155);
        assert_eq!(pkh.chain_id, "1");
        assert_eq!(
            pkh.account_address,
            "0xab16a96d359ec26a11e2c2b3d8f8b8942d5bfcdb"
        );
    }

    #[test]
    fn test_did_pkh_to_string() {
        let pkh = DidPkh::ethereum(1, "0xab16a96d359ec26a11e2c2b3d8f8b8942d5bfcdb").unwrap();
        assert_eq!(
            pkh.to_did_string(),
            "did:pkh:eip155:1:0xab16a96d359ec26a11e2c2b3d8f8b8942d5bfcdb"
        );
    }

    #[test]
    fn test_did_pkh_from_string() {
        let did_str = "did:pkh:eip155:1:0xab16a96d359ec26a11e2c2b3d8f8b8942d5bfcdb";
        let pkh = DidPkh::from_did_string(did_str).unwrap();
        assert_eq!(pkh.chain_namespace, ChainNamespace::Eip155);
        assert_eq!(pkh.chain_id, "1");
        assert_eq!(
            pkh.account_address,
            "0xab16a96d359ec26a11e2c2b3d8f8b8942d5bfcdb"
        );
        assert_eq!(pkh.to_did_string(), did_str);
    }

    #[test]
    fn test_did_pkh_roundtrip() {
        let original = "did:pkh:eip155:137:0xab16a96d359ec26a11e2c2b3d8f8b8942d5bfcdb";
        let pkh = DidPkh::from_did_string(original).unwrap();
        assert_eq!(pkh.to_did_string(), original);
    }

    #[test]
    fn test_invalid_ethereum_address() {
        // Too short
        assert!(DidPkh::ethereum(1, "0xabc").is_err());
        // Missing 0x prefix
        assert!(DidPkh::ethereum(1, "ab16a96d359ec26a11e2c2b3d8f8b8942d5bfcdb").is_err());
        // Wrong length
        assert!(DidPkh::ethereum(1, "0xab16a96d").is_err());
    }

    #[test]
    fn test_invalid_did_pkh_format() {
        assert!(DidPkh::from_did_string("did:key:z6Mk").is_err());
        assert!(DidPkh::from_did_string("did:pkh:eip155:only-two-parts").is_err());
        assert!(DidPkh::from_did_string("not-a-did").is_err());
    }

    #[test]
    fn test_resolve_ethereum_did_pkh() {
        let pkh = DidPkh::ethereum(1, "0xab16a96d359ec26a11e2c2b3d8f8b8942d5bfcdb").unwrap();
        let doc = pkh.resolve().unwrap();

        assert_eq!(doc.verification_method.len(), 1);
        assert!(!doc.authentication.is_empty());
        assert!(!doc.assertion_method.is_empty());
    }

    #[test]
    fn test_caip10_account_id() {
        let pkh = DidPkh::ethereum(1, "0xab16a96d359ec26a11e2c2b3d8f8b8942d5bfcdb").unwrap();
        assert_eq!(
            pkh.caip10_account_id(),
            "eip155:1:0xab16a96d359ec26a11e2c2b3d8f8b8942d5bfcdb"
        );
    }

    #[tokio::test]
    async fn test_did_pkh_method_resolver() {
        let method = DidPkhMethod::new();
        assert_eq!(method.method_name(), "pkh");

        let did = Did::new("did:pkh:eip155:1:0xab16a96d359ec26a11e2c2b3d8f8b8942d5bfcdb").unwrap();
        let doc = method.resolve(&did).await.unwrap();
        assert!(!doc.verification_method.is_empty());
    }

    #[tokio::test]
    async fn test_did_pkh_wrong_method() {
        let method = DidPkhMethod::new();
        let did = Did::new("did:key:z6Mk123").unwrap();
        assert!(method.resolve(&did).await.is_err());
    }

    #[test]
    fn test_bitcoin_did_pkh() {
        let pkh = DidPkh::new(
            "bip122",
            "000000000019d6689c085ae165831e93",
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7Divf",
        )
        .unwrap();
        assert_eq!(pkh.chain_namespace, ChainNamespace::Bip122);
        let did_str = pkh.to_did_string();
        assert!(did_str.starts_with("did:pkh:bip122:"));
    }

    #[test]
    fn test_chain_namespace_verification_method_type() {
        assert_eq!(
            ChainNamespace::Eip155.verification_method_type(),
            "EcdsaSecp256k1RecoveryMethod2020"
        );
        assert_eq!(
            ChainNamespace::Solana.verification_method_type(),
            "Ed25519VerificationKey2020"
        );
    }
}
