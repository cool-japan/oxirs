//! IDS Contract Negotiation and Lifecycle Management
//!
//! Implements automated contract negotiation following IDS Reference Architecture.

use super::policy::odrl_parser::PolicyType;
use super::policy::{OdrlPolicy, Permission};
use super::types::{IdsError, IdsResult, IdsUri, Party, SecurityProfile};
use base64::Engine;
use chrono::{DateTime, Duration, Utc};
use ring::digest::{Context as DigestContext, SHA256};
use ring::rand::SystemRandom;
use ring::signature::{self, Ed25519KeyPair, KeyPair};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Data Contract for IDS data exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataContract {
    /// Contract unique identifier
    #[serde(rename = "@id")]
    pub contract_id: IdsUri,

    /// Contract type
    #[serde(rename = "@type")]
    pub contract_type: String,

    /// Data consumer
    pub consumer: Party,

    /// Data provider
    pub provider: Party,

    /// Target asset description
    pub target_asset: AssetDescription,

    /// Usage policy (ODRL)
    pub usage_policy: OdrlPolicy,

    /// Contract start date
    pub contract_start: DateTime<Utc>,

    /// Contract end date (optional for unlimited)
    pub contract_end: Option<DateTime<Utc>>,

    /// Digital signatures
    pub signatures: Vec<DigitalSignature>,

    /// Contract state
    pub state: ContractState,

    /// Negotiation history
    pub negotiation_history: Vec<NegotiationMessage>,

    /// Created at
    pub created_at: DateTime<Utc>,

    /// Last modified at
    pub modified_at: DateTime<Utc>,
}

/// Asset description in contract
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AssetDescription {
    /// Asset ID
    pub asset_id: IdsUri,

    /// Asset title
    pub title: String,

    /// Asset description
    pub description: Option<String>,

    /// Content type
    pub content_type: Option<String>,

    /// File size (bytes)
    pub file_size: Option<u64>,

    /// Checksum (SHA-256)
    pub checksum: Option<String>,

    /// Asset version
    pub version: Option<String>,

    /// Keywords
    pub keywords: Vec<String>,

    /// Language
    pub language: Option<String>,
}

/// Digital Signature for contract
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalSignature {
    /// Signer party ID
    pub signer: IdsUri,

    /// Signature algorithm (RS256, ES256, etc.)
    pub algorithm: String,

    /// Signature value (base64)
    pub signature: String,

    /// Signing timestamp
    pub signed_at: DateTime<Utc>,

    /// Certificate chain (optional)
    pub certificate_chain: Option<Vec<String>>,
}

/// Contract Signer for creating and verifying digital signatures
pub struct ContractSigner {
    /// Ed25519 key pair for signing
    key_pair: Ed25519KeyPair,
    /// Signer identity
    signer_id: IdsUri,
}

impl ContractSigner {
    /// Create a new contract signer with a generated key pair
    pub fn new(signer_id: IdsUri) -> IdsResult<Self> {
        let rng = SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng)
            .map_err(|e| IdsError::InternalError(format!("Failed to generate key pair: {}", e)))?;

        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref())
            .map_err(|e| IdsError::InternalError(format!("Failed to parse key pair: {}", e)))?;

        Ok(Self {
            key_pair,
            signer_id,
        })
    }

    /// Create a contract signer from an existing PKCS#8 encoded private key
    pub fn from_pkcs8(signer_id: IdsUri, pkcs8_bytes: &[u8]) -> IdsResult<Self> {
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes)
            .map_err(|e| IdsError::InternalError(format!("Failed to parse key pair: {}", e)))?;

        Ok(Self {
            key_pair,
            signer_id,
        })
    }

    /// Get the public key bytes
    pub fn public_key(&self) -> &[u8] {
        self.key_pair.public_key().as_ref()
    }

    /// Get the public key as base64
    pub fn public_key_base64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(self.public_key())
    }

    /// Compute SHA-256 hash of contract content
    fn compute_contract_hash(contract: &DataContract) -> IdsResult<Vec<u8>> {
        let contract_json = serde_json::to_string(contract).map_err(|e| {
            IdsError::SerializationError(format!("Failed to serialize contract: {}", e))
        })?;

        let mut context = DigestContext::new(&SHA256);
        context.update(contract_json.as_bytes());
        Ok(context.finish().as_ref().to_vec())
    }

    /// Sign a contract
    pub fn sign_contract(&self, contract: &DataContract) -> IdsResult<DigitalSignature> {
        let hash = Self::compute_contract_hash(contract)?;
        let sig = self.key_pair.sign(&hash);

        Ok(DigitalSignature {
            signer: self.signer_id.clone(),
            algorithm: "Ed25519".to_string(),
            signature: base64::engine::general_purpose::STANDARD.encode(sig.as_ref()),
            signed_at: Utc::now(),
            certificate_chain: None,
        })
    }

    /// Verify a contract signature
    pub fn verify_signature(
        contract: &DataContract,
        signature: &DigitalSignature,
        public_key: &[u8],
    ) -> IdsResult<bool> {
        if signature.algorithm != "Ed25519" {
            return Err(IdsError::InternalError(format!(
                "Unsupported signature algorithm: {}",
                signature.algorithm
            )));
        }

        let hash = Self::compute_contract_hash(contract)?;

        let sig_bytes = base64::engine::general_purpose::STANDARD
            .decode(&signature.signature)
            .map_err(|e| IdsError::InternalError(format!("Invalid signature encoding: {}", e)))?;

        let public_key = signature::UnparsedPublicKey::new(&signature::ED25519, public_key);

        match public_key.verify(&hash, &sig_bytes) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

/// Signature verification result
#[derive(Debug, Clone)]
pub struct SignatureVerificationResult {
    /// Whether the signature is valid
    pub valid: bool,
    /// Signer ID
    pub signer: IdsUri,
    /// Verification timestamp
    pub verified_at: DateTime<Utc>,
    /// Error message if invalid
    pub error: Option<String>,
}

/// Contract State
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ContractState {
    /// Contract is being negotiated
    Negotiating,

    /// Contract offer pending acceptance
    Offered,

    /// Contract has been accepted (active)
    Accepted,

    /// Contract is active and in use
    Active,

    /// Contract has been suspended
    Suspended,

    /// Contract has been terminated
    Terminated,

    /// Contract has expired
    Expired,

    /// Contract was rejected
    Rejected,
}

impl ContractState {
    /// Check if contract is active (can be used)
    pub fn is_active(&self) -> bool {
        matches!(self, ContractState::Active | ContractState::Accepted)
    }

    /// Check if contract can be modified
    pub fn is_modifiable(&self) -> bool {
        matches!(self, ContractState::Negotiating | ContractState::Offered)
    }

    /// Check if contract is terminated
    pub fn is_terminated(&self) -> bool {
        matches!(
            self,
            ContractState::Terminated | ContractState::Expired | ContractState::Rejected
        )
    }
}

/// Negotiation message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationMessage {
    /// Message ID
    pub message_id: String,

    /// Message type
    pub message_type: NegotiationMessageType,

    /// Sender party
    pub sender: IdsUri,

    /// Receiver party
    pub receiver: IdsUri,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Proposed policy (for offers/counter-offers)
    pub policy: Option<OdrlPolicy>,

    /// Message content
    pub message: Option<String>,
}

/// Negotiation message types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NegotiationMessageType {
    /// Initial contract request
    ContractRequest,

    /// Contract offer
    ContractOffer,

    /// Counter-offer
    CounterOffer,

    /// Contract acceptance
    ContractAcceptance,

    /// Contract rejection
    ContractRejection,

    /// Contract agreement (finalized)
    ContractAgreement,
}

/// Contract Negotiator trait
#[async_trait::async_trait]
pub trait ContractNegotiator: Send + Sync {
    /// Initiate contract negotiation
    async fn initiate_negotiation(&self, offer: ContractOffer) -> IdsResult<NegotiationId>;

    /// Submit counter-offer
    async fn counter_offer(
        &self,
        negotiation_id: NegotiationId,
        counter: ContractOffer,
    ) -> IdsResult<()>;

    /// Accept contract
    async fn accept(&self, negotiation_id: NegotiationId) -> IdsResult<DataContract>;

    /// Reject contract
    async fn reject(&self, negotiation_id: NegotiationId, reason: String) -> IdsResult<()>;

    /// Get negotiation status
    async fn get_status(&self, negotiation_id: NegotiationId) -> IdsResult<NegotiationStatus>;
}

/// Contract Offer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractOffer {
    /// Provider party
    pub provider: Party,

    /// Consumer party
    pub consumer: Party,

    /// Asset being offered
    pub asset: AssetDescription,

    /// Proposed usage policy
    pub policy: OdrlPolicy,

    /// Contract duration (days)
    pub duration_days: Option<i64>,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Negotiation ID
pub type NegotiationId = String;

/// Negotiation Status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationStatus {
    /// Negotiation ID
    pub negotiation_id: NegotiationId,

    /// Current state
    pub state: ContractState,

    /// Number of negotiation rounds
    pub rounds: u32,

    /// Last message timestamp
    pub last_updated: DateTime<Utc>,

    /// Current offer
    pub current_offer: Option<ContractOffer>,

    /// History of messages
    pub messages: Vec<NegotiationMessage>,
}

/// In-memory contract negotiator (for development)
pub struct InMemoryNegotiator {
    negotiations: Arc<RwLock<HashMap<String, NegotiationStatus>>>,
    contracts: Arc<RwLock<HashMap<String, DataContract>>>,
}

impl Default for InMemoryNegotiator {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryNegotiator {
    /// Create a new in-memory negotiator
    pub fn new() -> Self {
        Self {
            negotiations: Arc::new(RwLock::new(HashMap::new())),
            contracts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get contract by ID
    pub async fn get_contract(&self, contract_id: &str) -> Option<DataContract> {
        self.contracts.read().await.get(contract_id).cloned()
    }

    /// List all contracts
    pub async fn list_contracts(&self) -> Vec<DataContract> {
        self.contracts.read().await.values().cloned().collect()
    }

    /// Terminate contract
    pub async fn terminate_contract(&self, contract_id: &str) -> IdsResult<()> {
        let mut contracts = self.contracts.write().await;

        if let Some(contract) = contracts.get_mut(contract_id) {
            contract.state = ContractState::Terminated;
            contract.modified_at = Utc::now();
            Ok(())
        } else {
            Err(IdsError::ContractNotFound(contract_id.to_string()))
        }
    }
}

#[async_trait::async_trait]
impl ContractNegotiator for InMemoryNegotiator {
    async fn initiate_negotiation(&self, offer: ContractOffer) -> IdsResult<NegotiationId> {
        let negotiation_id = Uuid::new_v4().to_string();

        let initial_message = NegotiationMessage {
            message_id: Uuid::new_v4().to_string(),
            message_type: NegotiationMessageType::ContractOffer,
            sender: offer.provider.id.clone(),
            receiver: offer.consumer.id.clone(),
            timestamp: Utc::now(),
            policy: Some(offer.policy.clone()),
            message: None,
        };

        let status = NegotiationStatus {
            negotiation_id: negotiation_id.clone(),
            state: ContractState::Offered,
            rounds: 1,
            last_updated: Utc::now(),
            current_offer: Some(offer),
            messages: vec![initial_message],
        };

        self.negotiations
            .write()
            .await
            .insert(negotiation_id.clone(), status);

        Ok(negotiation_id)
    }

    async fn counter_offer(
        &self,
        negotiation_id: NegotiationId,
        counter: ContractOffer,
    ) -> IdsResult<()> {
        let mut negotiations = self.negotiations.write().await;

        let status = negotiations.get_mut(&negotiation_id).ok_or_else(|| {
            IdsError::NegotiationFailed(format!("Negotiation {} not found", negotiation_id))
        })?;

        if !status.state.is_modifiable() {
            return Err(IdsError::InvalidContractState {
                expected: "Negotiating or Offered".to_string(),
                actual: format!("{:?}", status.state),
            });
        }

        let message = NegotiationMessage {
            message_id: Uuid::new_v4().to_string(),
            message_type: NegotiationMessageType::CounterOffer,
            sender: counter.consumer.id.clone(),
            receiver: counter.provider.id.clone(),
            timestamp: Utc::now(),
            policy: Some(counter.policy.clone()),
            message: None,
        };

        status.messages.push(message);
        status.current_offer = Some(counter);
        status.rounds += 1;
        status.last_updated = Utc::now();

        Ok(())
    }

    async fn accept(&self, negotiation_id: NegotiationId) -> IdsResult<DataContract> {
        let mut negotiations = self.negotiations.write().await;

        let status = negotiations.get_mut(&negotiation_id).ok_or_else(|| {
            IdsError::NegotiationFailed(format!("Negotiation {} not found", negotiation_id))
        })?;

        let offer = status
            .current_offer
            .as_ref()
            .ok_or_else(|| IdsError::NegotiationFailed("No current offer to accept".to_string()))?;

        // Create contract from accepted offer
        let contract_id = IdsUri::new(format!("urn:ids:contract:{}", Uuid::new_v4())).unwrap();

        let now = Utc::now();
        let contract_end = offer.duration_days.map(|days| now + Duration::days(days));

        let contract = DataContract {
            contract_id: contract_id.clone(),
            contract_type: "ids:ContractAgreement".to_string(),
            consumer: offer.consumer.clone(),
            provider: offer.provider.clone(),
            target_asset: offer.asset.clone(),
            usage_policy: offer.policy.clone(),
            contract_start: now,
            contract_end,
            signatures: Vec::new(), // TODO: Add digital signatures
            state: ContractState::Accepted,
            negotiation_history: status.messages.clone(),
            created_at: now,
            modified_at: now,
        };

        // Update negotiation state
        status.state = ContractState::Accepted;
        status.last_updated = now;

        let acceptance_message = NegotiationMessage {
            message_id: Uuid::new_v4().to_string(),
            message_type: NegotiationMessageType::ContractAcceptance,
            sender: offer.consumer.id.clone(),
            receiver: offer.provider.id.clone(),
            timestamp: now,
            policy: None,
            message: Some("Contract accepted".to_string()),
        };

        status.messages.push(acceptance_message);

        // Store contract
        self.contracts
            .write()
            .await
            .insert(contract_id.as_str().to_string(), contract.clone());

        Ok(contract)
    }

    async fn reject(&self, negotiation_id: NegotiationId, reason: String) -> IdsResult<()> {
        let mut negotiations = self.negotiations.write().await;

        let status = negotiations.get_mut(&negotiation_id).ok_or_else(|| {
            IdsError::NegotiationFailed(format!("Negotiation {} not found", negotiation_id))
        })?;

        status.state = ContractState::Rejected;
        status.last_updated = Utc::now();

        let rejection_message = NegotiationMessage {
            message_id: Uuid::new_v4().to_string(),
            message_type: NegotiationMessageType::ContractRejection,
            sender: status
                .current_offer
                .as_ref()
                .map(|o| o.consumer.id.clone())
                .unwrap_or_else(|| IdsUri::new("urn:ids:unknown").unwrap()),
            receiver: status
                .current_offer
                .as_ref()
                .map(|o| o.provider.id.clone())
                .unwrap_or_else(|| IdsUri::new("urn:ids:unknown").unwrap()),
            timestamp: Utc::now(),
            policy: None,
            message: Some(reason),
        };

        status.messages.push(rejection_message);

        Ok(())
    }

    async fn get_status(&self, negotiation_id: NegotiationId) -> IdsResult<NegotiationStatus> {
        self.negotiations
            .read()
            .await
            .get(&negotiation_id)
            .cloned()
            .ok_or_else(|| {
                IdsError::NegotiationFailed(format!("Negotiation {} not found", negotiation_id))
            })
    }
}

/// Contract Manager
pub struct ContractManager {
    negotiator: Arc<dyn ContractNegotiator>,
    contracts: Arc<RwLock<HashMap<String, DataContract>>>,
}

impl ContractManager {
    /// Create a new contract manager
    pub fn new(negotiator: Arc<dyn ContractNegotiator>) -> Self {
        Self {
            negotiator,
            contracts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get contract by string ID
    pub async fn get_contract(&self, contract_id: &str) -> Option<DataContract> {
        self.contracts.read().await.get(contract_id).cloned()
    }

    /// Get contract by IdsUri
    ///
    /// This is the preferred method when you have an IdsUri.
    pub async fn get_contract_uri(&self, contract_id: &IdsUri) -> Option<DataContract> {
        self.contracts
            .read()
            .await
            .get(contract_id.as_str())
            .cloned()
    }

    /// Check if contract is valid for use
    pub async fn is_valid(&self, contract_id: &str) -> bool {
        if let Some(contract) = self.get_contract(contract_id).await {
            // Check state
            if !contract.state.is_active() {
                return false;
            }

            // Check expiration
            if let Some(end) = contract.contract_end {
                if Utc::now() > end {
                    return false;
                }
            }

            true
        } else {
            false
        }
    }

    /// Activate contract after acceptance
    pub async fn activate_contract(&self, contract_id: &str) -> IdsResult<()> {
        let mut contracts = self.contracts.write().await;

        if let Some(contract) = contracts.get_mut(contract_id) {
            if contract.state != ContractState::Accepted {
                return Err(IdsError::InvalidContractState {
                    expected: "Accepted".to_string(),
                    actual: format!("{:?}", contract.state),
                });
            }

            contract.state = ContractState::Active;
            contract.modified_at = Utc::now();
            Ok(())
        } else {
            Err(IdsError::ContractNotFound(contract_id.to_string()))
        }
    }

    /// Get negotiator
    pub fn negotiator(&self) -> Arc<dyn ContractNegotiator> {
        Arc::clone(&self.negotiator)
    }

    /// Get contract by IdsUri
    pub async fn get_contract_by_uri(&self, contract_id: &IdsUri) -> Option<DataContract> {
        self.get_contract(contract_id.as_str()).await
    }

    /// List all contracts
    pub async fn list_contracts(&self) -> Vec<DataContract> {
        self.contracts.read().await.values().cloned().collect()
    }

    /// Initiate contract negotiation
    ///
    /// Creates a new contract offer and starts the negotiation process.
    pub async fn initiate_negotiation(
        &self,
        provider_id: IdsUri,
        asset_id: IdsUri,
        consumer: Party,
    ) -> IdsResult<DataContract> {
        // Get resource info from catalog (if available, otherwise use minimal info)
        let asset = AssetDescription {
            asset_id: asset_id.clone(),
            title: format!("Asset {}", asset_id.as_str()),
            description: None,
            content_type: None,
            file_size: None,
            checksum: None,
            version: None,
            keywords: Vec::new(),
            language: None,
        };

        // Create provider party
        let provider = Party {
            id: provider_id.clone(),
            name: format!("Provider {}", provider_id.as_str()),
            legal_name: None,
            description: None,
            contact: None,
            gaiax_participant_id: None,
        };

        // Create a default usage policy
        let asset_target = super::policy::odrl_parser::AssetTarget {
            uid: asset_id.clone(),
            asset_type: None,
            title: Some(asset.title.clone()),
            description: asset.description.clone(),
        };

        let policy = super::policy::OdrlPolicy {
            uid: IdsUri::new(format!("urn:ids:policy:{}", uuid::Uuid::new_v4()))
                .unwrap_or_else(|_| IdsUri::new("urn:ids:policy:default").expect("valid")),
            policy_type: PolicyType::Offer,
            context: None,
            profile: None,
            assigner: Some(provider.clone()),
            assignee: Some(consumer.clone()),
            targets: vec![asset_target.clone()],
            permissions: vec![Permission {
                uid: None,
                action: super::policy::OdrlAction::Use,
                target: Some(asset_target),
                assigner: Some(provider.clone()),
                assignee: Some(consumer.clone()),
                constraints: Vec::new(),
                duties: Vec::new(),
            }],
            prohibitions: Vec::new(),
            obligations: Vec::new(),
            inherits_from: None,
            conflict: None,
        };

        let offer = ContractOffer {
            provider,
            consumer,
            asset,
            policy,
            duration_days: Some(super::super::ids::DEFAULT_CONTRACT_VALIDITY_DAYS),
            metadata: HashMap::new(),
        };

        // Initiate negotiation via the negotiator
        let negotiation_id = self.negotiator.initiate_negotiation(offer).await?;

        // Accept the contract immediately (for simplified flow)
        let contract = self.negotiator.accept(negotiation_id).await?;

        // Store the contract
        self.contracts
            .write()
            .await
            .insert(contract.contract_id.as_str().to_string(), contract.clone());

        Ok(contract)
    }

    /// Accept a contract
    pub async fn accept_contract(&self, contract_id: &IdsUri) -> IdsResult<DataContract> {
        let mut contracts = self.contracts.write().await;
        let id_str = contract_id.as_str();

        let contract = contracts
            .get_mut(id_str)
            .ok_or_else(|| IdsError::ContractNotFound(id_str.to_string()))?;

        if !contract.state.is_modifiable() && contract.state != ContractState::Offered {
            return Err(IdsError::InvalidContractState {
                expected: "Offered or Negotiating".to_string(),
                actual: format!("{:?}", contract.state),
            });
        }

        contract.state = ContractState::Accepted;
        contract.modified_at = Utc::now();

        // Add acceptance to negotiation history
        contract.negotiation_history.push(NegotiationMessage {
            message_id: uuid::Uuid::new_v4().to_string(),
            message_type: NegotiationMessageType::ContractAcceptance,
            sender: contract.consumer.id.clone(),
            receiver: contract.provider.id.clone(),
            timestamp: Utc::now(),
            policy: None,
            message: Some("Contract accepted".to_string()),
        });

        Ok(contract.clone())
    }

    /// Reject a contract
    pub async fn reject_contract(&self, contract_id: &IdsUri) -> IdsResult<DataContract> {
        let mut contracts = self.contracts.write().await;
        let id_str = contract_id.as_str();

        let contract = contracts
            .get_mut(id_str)
            .ok_or_else(|| IdsError::ContractNotFound(id_str.to_string()))?;

        if contract.state.is_terminated() {
            return Err(IdsError::InvalidContractState {
                expected: "Not terminated".to_string(),
                actual: format!("{:?}", contract.state),
            });
        }

        contract.state = ContractState::Rejected;
        contract.modified_at = Utc::now();

        // Add rejection to negotiation history
        contract.negotiation_history.push(NegotiationMessage {
            message_id: uuid::Uuid::new_v4().to_string(),
            message_type: NegotiationMessageType::ContractRejection,
            sender: contract.consumer.id.clone(),
            receiver: contract.provider.id.clone(),
            timestamp: Utc::now(),
            policy: None,
            message: Some("Contract rejected".to_string()),
        });

        Ok(contract.clone())
    }

    /// Terminate a contract
    pub async fn terminate_contract(&self, contract_id: &IdsUri) -> IdsResult<DataContract> {
        let mut contracts = self.contracts.write().await;
        let id_str = contract_id.as_str();

        let contract = contracts
            .get_mut(id_str)
            .ok_or_else(|| IdsError::ContractNotFound(id_str.to_string()))?;

        if contract.state.is_terminated() {
            return Err(IdsError::InvalidContractState {
                expected: "Not already terminated".to_string(),
                actual: format!("{:?}", contract.state),
            });
        }

        contract.state = ContractState::Terminated;
        contract.modified_at = Utc::now();

        Ok(contract.clone())
    }

    /// Store a contract directly (for imports or external negotiations)
    pub async fn store_contract(&self, contract: DataContract) {
        self.contracts
            .write()
            .await
            .insert(contract.contract_id.as_str().to_string(), contract);
    }

    /// Get contract count
    pub async fn count(&self) -> usize {
        self.contracts.read().await.len()
    }

    /// Check if contract exists
    pub async fn contains(&self, contract_id: &str) -> bool {
        self.contracts.read().await.contains_key(contract_id)
    }

    /// Filter contracts by state
    pub async fn filter_by_state(&self, state: ContractState) -> Vec<DataContract> {
        self.contracts
            .read()
            .await
            .values()
            .filter(|c| c.state == state)
            .cloned()
            .collect()
    }

    /// Get active contracts only
    pub async fn list_active(&self) -> Vec<DataContract> {
        self.contracts
            .read()
            .await
            .values()
            .filter(|c| c.state.is_active())
            .cloned()
            .collect()
    }

    /// Remove expired contracts
    pub async fn cleanup_expired(&self) -> usize {
        let mut contracts = self.contracts.write().await;
        let now = Utc::now();

        let expired: Vec<String> = contracts
            .iter()
            .filter(|(_, c)| c.contract_end.map(|end| now > end).unwrap_or(false))
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired.len();
        for id in expired {
            if let Some(contract) = contracts.get_mut(&id) {
                contract.state = ContractState::Expired;
                contract.modified_at = now;
            }
        }

        count
    }
}

#[cfg(test)]
mod tests {
    use super::super::policy::{OdrlAction, OdrlParser, Permission};
    use super::*;

    fn create_test_party(name: &str) -> Party {
        Party {
            id: IdsUri::new(format!("https://{}.example.org", name)).expect("valid uri"),
            name: name.to_string(),
            legal_name: None,
            description: None,
            contact: None,
            gaiax_participant_id: None,
        }
    }

    fn create_test_asset() -> AssetDescription {
        AssetDescription {
            asset_id: IdsUri::new("https://example.org/data/dataset1").unwrap(),
            title: "Test Dataset".to_string(),
            description: Some("A test dataset for contract negotiation".to_string()),
            content_type: Some("application/json".to_string()),
            file_size: Some(1024),
            checksum: None,
            version: Some("1.0".to_string()),
            keywords: vec!["test".to_string(), "data".to_string()],
            language: Some("en".to_string()),
        }
    }

    #[test]
    fn test_contract_state_checks() {
        assert!(ContractState::Active.is_active());
        assert!(ContractState::Accepted.is_active());
        assert!(!ContractState::Negotiating.is_active());

        assert!(ContractState::Negotiating.is_modifiable());
        assert!(ContractState::Offered.is_modifiable());
        assert!(!ContractState::Accepted.is_modifiable());

        assert!(ContractState::Terminated.is_terminated());
        assert!(ContractState::Expired.is_terminated());
        assert!(!ContractState::Active.is_terminated());
    }

    #[tokio::test]
    async fn test_contract_negotiation_flow() {
        let negotiator = Arc::new(InMemoryNegotiator::new());

        let provider = create_test_party("provider");
        let consumer = create_test_party("consumer");
        let asset = create_test_asset();

        let policy = OdrlParser::create_read_permission(
            IdsUri::new("https://example.org/policy/read").unwrap(),
            provider.clone(),
            consumer.clone(),
            super::super::policy::odrl_parser::AssetTarget {
                uid: asset.asset_id.clone(),
                asset_type: None,
                title: Some(asset.title.clone()),
                description: asset.description.clone(),
            },
        );

        let offer = ContractOffer {
            provider,
            consumer,
            asset,
            policy,
            duration_days: Some(90),
            metadata: HashMap::new(),
        };

        // Initiate negotiation
        let negotiation_id = negotiator.initiate_negotiation(offer).await.unwrap();

        // Check status
        let status = negotiator.get_status(negotiation_id.clone()).await.unwrap();
        assert_eq!(status.state, ContractState::Offered);
        assert_eq!(status.rounds, 1);

        // Accept contract
        let contract = negotiator.accept(negotiation_id.clone()).await.unwrap();
        assert_eq!(contract.state, ContractState::Accepted);
        assert!(contract.contract_end.is_some());
    }

    #[tokio::test]
    async fn test_contract_rejection() {
        let negotiator = Arc::new(InMemoryNegotiator::new());

        let provider = create_test_party("provider");
        let consumer = create_test_party("consumer");
        let asset = create_test_asset();

        let policy = OdrlParser::create_read_permission(
            IdsUri::new("https://example.org/policy/read").unwrap(),
            provider.clone(),
            consumer.clone(),
            super::super::policy::odrl_parser::AssetTarget {
                uid: asset.asset_id.clone(),
                asset_type: None,
                title: Some(asset.title.clone()),
                description: asset.description.clone(),
            },
        );

        let offer = ContractOffer {
            provider,
            consumer,
            asset,
            policy,
            duration_days: None,
            metadata: HashMap::new(),
        };

        let negotiation_id = negotiator.initiate_negotiation(offer).await.unwrap();

        negotiator
            .reject(negotiation_id.clone(), "Terms not acceptable".to_string())
            .await
            .unwrap();

        let status = negotiator.get_status(negotiation_id).await.unwrap();
        assert_eq!(status.state, ContractState::Rejected);
    }

    #[tokio::test]
    async fn test_contract_manager() {
        let negotiator = Arc::new(InMemoryNegotiator::new());
        let manager = ContractManager::new(negotiator.clone());

        let provider = create_test_party("provider");
        let consumer = create_test_party("consumer");
        let asset = create_test_asset();

        let policy = OdrlParser::create_read_permission(
            IdsUri::new("https://example.org/policy/read").unwrap(),
            provider.clone(),
            consumer.clone(),
            super::super::policy::odrl_parser::AssetTarget {
                uid: asset.asset_id.clone(),
                asset_type: None,
                title: Some(asset.title.clone()),
                description: asset.description.clone(),
            },
        );

        let offer = ContractOffer {
            provider,
            consumer,
            asset,
            policy,
            duration_days: Some(30),
            metadata: HashMap::new(),
        };

        let negotiation_id = negotiator.initiate_negotiation(offer).await.unwrap();
        let contract = negotiator.accept(negotiation_id).await.unwrap();

        let contract_id = contract.contract_id.clone();

        // Store contract
        manager
            .contracts
            .write()
            .await
            .insert(contract_id.to_string(), contract);

        // Activate contract
        manager
            .activate_contract(contract_id.as_str())
            .await
            .unwrap();

        // Check validity
        assert!(manager.is_valid(contract_id.as_str()).await);

        let retrieved = manager.get_contract(contract_id.as_str()).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().state, ContractState::Active);
    }

    #[tokio::test]
    async fn test_contract_signing() {
        let negotiator = Arc::new(InMemoryNegotiator::new());

        let provider = create_test_party("provider");
        let consumer = create_test_party("consumer");
        let asset = create_test_asset();

        let policy = OdrlParser::create_read_permission(
            IdsUri::new("https://example.org/policy/read").expect("valid URI"),
            provider.clone(),
            consumer.clone(),
            super::super::policy::odrl_parser::AssetTarget {
                uid: asset.asset_id.clone(),
                asset_type: None,
                title: Some(asset.title.clone()),
                description: asset.description.clone(),
            },
        );

        let offer = ContractOffer {
            provider: provider.clone(),
            consumer,
            asset,
            policy,
            duration_days: Some(90),
            metadata: HashMap::new(),
        };

        // Create and accept contract
        let negotiation_id = negotiator
            .initiate_negotiation(offer)
            .await
            .expect("initiate");
        let mut contract = negotiator.accept(negotiation_id).await.expect("accept");

        // Create signer and sign the contract
        let signer = ContractSigner::new(provider.id.clone()).expect("create signer");
        let signature = signer.sign_contract(&contract).expect("sign contract");

        // Verify the signature before adding it to the contract
        // (the hash is computed from the contract without signatures)
        let is_valid = ContractSigner::verify_signature(&contract, &signature, signer.public_key())
            .expect("verify signature");

        assert!(is_valid);
        assert_eq!(signature.algorithm, "Ed25519");
        assert_eq!(signature.signer, provider.id);

        // Now add the signature to the contract
        contract.signatures.push(signature);
    }

    #[tokio::test]
    async fn test_contract_signature_tampering_detection() {
        let negotiator = Arc::new(InMemoryNegotiator::new());

        let provider = create_test_party("provider");
        let consumer = create_test_party("consumer");
        let asset = create_test_asset();

        let policy = OdrlParser::create_read_permission(
            IdsUri::new("https://example.org/policy/read").expect("valid URI"),
            provider.clone(),
            consumer.clone(),
            super::super::policy::odrl_parser::AssetTarget {
                uid: asset.asset_id.clone(),
                asset_type: None,
                title: Some(asset.title.clone()),
                description: asset.description.clone(),
            },
        );

        let offer = ContractOffer {
            provider: provider.clone(),
            consumer,
            asset,
            policy,
            duration_days: Some(90),
            metadata: HashMap::new(),
        };

        let negotiation_id = negotiator
            .initiate_negotiation(offer)
            .await
            .expect("initiate");
        let mut contract = negotiator.accept(negotiation_id).await.expect("accept");

        // Sign the contract
        let signer = ContractSigner::new(provider.id.clone()).expect("create signer");
        let signature = signer.sign_contract(&contract).expect("sign contract");
        contract.signatures.push(signature.clone());

        // Tamper with the contract
        contract.contract_type = "ids:TamperedContract".to_string();

        // Verification should fail for tampered contract
        let is_valid = ContractSigner::verify_signature(&contract, &signature, signer.public_key())
            .expect("verify signature");

        assert!(!is_valid, "Tampered contract should fail verification");
    }
}
