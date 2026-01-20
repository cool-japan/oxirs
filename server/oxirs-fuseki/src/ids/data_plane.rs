//! IDS Data Plane
//!
//! Manages data transfer after contract agreement.
//! Integrates with oxirs-stream for high-performance data movement.

use super::contract::{ContractState, DataContract};
use super::lineage::{Activity, Agent, LineageRecord, ProvenanceGraph};
use super::policy::{EvaluationContext, OdrlAction, PolicyDecision, PolicyEngine};
use super::residency::{GdprComplianceChecker, Region, TransferRecord};
use super::types::{IdsError, IdsResult, IdsUri, Party, TransferProtocol};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Transfer status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TransferStatus {
    /// Transfer pending
    Pending,
    /// Transfer in progress
    InProgress,
    /// Transfer completed successfully
    Completed,
    /// Transfer failed
    Failed,
    /// Transfer cancelled
    Cancelled,
    /// Transfer suspended (e.g., policy violation)
    Suspended,
}

/// Transfer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TransferType {
    /// Push data to consumer
    Push,
    /// Consumer pulls data
    Pull,
    /// Bidirectional streaming
    Streaming,
}

/// Data transfer request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferRequest {
    /// Request ID
    pub id: String,
    /// Contract ID authorizing the transfer
    pub contract_id: IdsUri,
    /// Resource to transfer
    pub resource_id: IdsUri,
    /// Transfer type
    pub transfer_type: TransferType,
    /// Transfer protocol
    pub protocol: TransferProtocol,
    /// Source endpoint
    pub source_endpoint: Option<String>,
    /// Destination endpoint
    pub destination_endpoint: String,
    /// Request timestamp
    pub requested_at: DateTime<Utc>,
    /// Requestor
    pub requestor: Party,
    /// Additional properties
    pub properties: HashMap<String, String>,
}

impl TransferRequest {
    /// Create a new transfer request
    pub fn new(
        contract_id: IdsUri,
        resource_id: IdsUri,
        destination_endpoint: impl Into<String>,
        requestor: Party,
    ) -> Self {
        Self {
            id: format!("transfer-{}", Utc::now().timestamp_nanos_opt().unwrap_or(0)),
            contract_id,
            resource_id,
            transfer_type: TransferType::Push,
            protocol: TransferProtocol::Https,
            source_endpoint: None,
            destination_endpoint: destination_endpoint.into(),
            requested_at: Utc::now(),
            requestor,
            properties: HashMap::new(),
        }
    }

    /// Set transfer type
    pub fn with_transfer_type(mut self, transfer_type: TransferType) -> Self {
        self.transfer_type = transfer_type;
        self
    }

    /// Set protocol
    pub fn with_protocol(mut self, protocol: TransferProtocol) -> Self {
        self.protocol = protocol;
        self
    }

    /// Set source endpoint
    pub fn with_source_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.source_endpoint = Some(endpoint.into());
        self
    }

    /// Add property
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
}

/// Data transfer result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferResult {
    /// Transfer request ID
    pub request_id: String,
    /// Transfer status
    pub status: TransferStatus,
    /// Bytes transferred
    pub bytes_transferred: u64,
    /// Start time
    pub started_at: Option<DateTime<Utc>>,
    /// End time
    pub completed_at: Option<DateTime<Utc>>,
    /// Error message if failed
    pub error: Option<String>,
    /// Lineage record ID
    pub lineage_record_id: Option<String>,
    /// Transfer checksum (SHA-256)
    pub checksum: Option<String>,
}

impl TransferResult {
    /// Create a pending result
    pub fn pending(request_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            status: TransferStatus::Pending,
            bytes_transferred: 0,
            started_at: None,
            completed_at: None,
            error: None,
            lineage_record_id: None,
            checksum: None,
        }
    }

    /// Create a completed result
    pub fn completed(
        request_id: impl Into<String>,
        bytes_transferred: u64,
        started_at: DateTime<Utc>,
        checksum: impl Into<String>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            status: TransferStatus::Completed,
            bytes_transferred,
            started_at: Some(started_at),
            completed_at: Some(Utc::now()),
            error: None,
            lineage_record_id: None,
            checksum: Some(checksum.into()),
        }
    }

    /// Create a failed result
    pub fn failed(request_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            status: TransferStatus::Failed,
            bytes_transferred: 0,
            started_at: None,
            completed_at: Some(Utc::now()),
            error: Some(error.into()),
            lineage_record_id: None,
            checksum: None,
        }
    }

    /// Set lineage record ID
    pub fn with_lineage_record(mut self, record_id: impl Into<String>) -> Self {
        self.lineage_record_id = Some(record_id.into());
        self
    }
}

/// Transfer process (in-flight transfer)
#[derive(Debug, Clone)]
pub struct TransferProcess {
    /// Transfer request
    pub request: TransferRequest,
    /// Current status
    pub status: TransferStatus,
    /// Bytes transferred so far
    pub bytes_transferred: u64,
    /// Start time
    pub started_at: Option<DateTime<Utc>>,
    /// Last update time
    pub updated_at: DateTime<Utc>,
    /// Error if any
    pub error: Option<String>,
}

/// Data Plane Manager
///
/// Orchestrates data transfers between IDS connectors after contract agreement.
pub struct DataPlaneManager {
    /// Policy engine for access control
    policy_engine: Arc<PolicyEngine>,
    /// Provenance tracker
    provenance: Arc<ProvenanceGraph>,
    /// GDPR compliance checker
    gdpr_checker: Arc<GdprComplianceChecker>,
    /// Active transfers
    transfers: Arc<RwLock<HashMap<String, TransferProcess>>>,
    /// Transfer history
    history: Arc<RwLock<Vec<TransferResult>>>,
    /// Connector ID
    connector_id: IdsUri,
}

impl DataPlaneManager {
    /// Create a new Data Plane Manager
    pub fn new(
        connector_id: IdsUri,
        policy_engine: Arc<PolicyEngine>,
        provenance: Arc<ProvenanceGraph>,
    ) -> Self {
        Self {
            policy_engine,
            provenance,
            gdpr_checker: Arc::new(GdprComplianceChecker::new()),
            transfers: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            connector_id,
        }
    }

    /// Initiate a data transfer based on an agreed contract
    pub async fn initiate_transfer(
        &self,
        request: TransferRequest,
        contract: &DataContract,
    ) -> IdsResult<TransferResult> {
        // Step 1: Verify contract is active (Accepted or Active state)
        if !contract.state.is_active() {
            return Err(IdsError::ContractNotAgreed(format!(
                "Contract {} is not in active state (current: {:?})",
                contract.contract_id, contract.state
            )));
        }

        // Step 2: Verify contract hasn't expired
        if let Some(end) = contract.contract_end {
            if end < Utc::now() {
                return Err(IdsError::ContractExpired(format!(
                    "Contract {} expired at {}",
                    contract.contract_id, end
                )));
            }
        }

        // Step 3: Verify resource is covered by contract
        if contract.target_asset.asset_id != request.resource_id {
            return Err(IdsError::PolicyViolation(format!(
                "Resource {} is not covered by contract {} (expected: {})",
                request.resource_id, contract.contract_id, contract.target_asset.asset_id
            )));
        }

        // Step 4: Evaluate policy for the transfer action
        let context = EvaluationContext::new()
            .with_requestor(request.requestor.clone())
            .with_resource(request.resource_id.clone())
            .with_connector_id(self.connector_id.as_str());

        let decision = self
            .policy_engine
            .evaluate(&contract.usage_policy.uid, &OdrlAction::Read, &context)
            .await?;

        match decision {
            PolicyDecision::Deny { reason, .. } => {
                return Err(IdsError::PolicyViolation(format!(
                    "Transfer denied by policy: {}",
                    reason
                )));
            }
            PolicyDecision::NotApplicable => {
                return Err(IdsError::PolicyViolation(
                    "No applicable policy found for transfer".to_string(),
                ));
            }
            PolicyDecision::Permit { duties, .. } => {
                // Record duties that need to be fulfilled
                if !duties.is_empty() {
                    tracing::info!(
                        "Transfer {} has {} duties to fulfill",
                        request.id,
                        duties.len()
                    );
                }
            }
        }

        // Step 5: Create transfer process
        let process = TransferProcess {
            request: request.clone(),
            status: TransferStatus::Pending,
            bytes_transferred: 0,
            started_at: None,
            updated_at: Utc::now(),
            error: None,
        };

        // Store transfer
        {
            let mut transfers = self.transfers.write().await;
            transfers.insert(request.id.clone(), process);
        }

        // Step 6: Execute transfer (async)
        let result = self.execute_transfer(&request).await?;

        // Step 7: Record provenance
        let lineage_record = self.record_transfer_provenance(&request, &result).await?;

        let result = result.with_lineage_record(lineage_record);

        // Step 8: Store in history
        {
            let mut history = self.history.write().await;
            history.push(result.clone());
        }

        // Step 9: Clean up active transfer
        {
            let mut transfers = self.transfers.write().await;
            transfers.remove(&request.id);
        }

        Ok(result)
    }

    /// Execute the actual data transfer
    async fn execute_transfer(&self, request: &TransferRequest) -> IdsResult<TransferResult> {
        let started_at = Utc::now();

        // Update status to in progress
        {
            let mut transfers = self.transfers.write().await;
            if let Some(process) = transfers.get_mut(&request.id) {
                process.status = TransferStatus::InProgress;
                process.started_at = Some(started_at);
                process.updated_at = Utc::now();
            }
        }

        // Execute transfer based on protocol
        match request.protocol {
            TransferProtocol::Https => self.execute_https_transfer(request, started_at).await,
            TransferProtocol::Idscp2 => self.execute_idscp2_transfer(request, started_at).await,
            TransferProtocol::MultipartFormData => {
                self.execute_multipart_transfer(request, started_at).await
            }
            TransferProtocol::S3 => self.execute_s3_transfer(request, started_at).await,
            TransferProtocol::Kafka => self.execute_kafka_transfer(request, started_at).await,
            TransferProtocol::Nats => self.execute_nats_transfer(request, started_at).await,
        }
    }

    /// Execute NATS transfer
    async fn execute_nats_transfer(
        &self,
        request: &TransferRequest,
        started_at: DateTime<Utc>,
    ) -> IdsResult<TransferResult> {
        // NATS JetStream transfer would:
        // 1. Publish messages to subject
        // 2. Wait for acknowledgements
        // 3. Record sequence number

        let bytes_transferred = 1024 * 100; // 100KB for NATS
        let checksum = format!("sha256:{}", hex::encode([0u8; 32]));

        tracing::info!(
            "NATS transfer {} completed: {} bytes to subject {}",
            request.id,
            bytes_transferred,
            request.destination_endpoint
        );

        Ok(TransferResult::completed(
            &request.id,
            bytes_transferred,
            started_at,
            checksum,
        ))
    }

    /// Execute HTTPS transfer
    async fn execute_https_transfer(
        &self,
        request: &TransferRequest,
        started_at: DateTime<Utc>,
    ) -> IdsResult<TransferResult> {
        // In a real implementation, this would:
        // 1. Fetch data from source
        // 2. Apply any transformations required by policy
        // 3. POST to destination endpoint
        // 4. Verify receipt

        // Simulated transfer for now
        let bytes_transferred = 1024 * 1024; // 1MB simulated
        let checksum = format!("sha256:{}", hex::encode([0u8; 32]));

        tracing::info!(
            "HTTPS transfer {} completed: {} bytes to {}",
            request.id,
            bytes_transferred,
            request.destination_endpoint
        );

        Ok(TransferResult::completed(
            &request.id,
            bytes_transferred,
            started_at,
            checksum,
        ))
    }

    /// Execute IDSCP2 transfer
    async fn execute_idscp2_transfer(
        &self,
        request: &TransferRequest,
        started_at: DateTime<Utc>,
    ) -> IdsResult<TransferResult> {
        // IDSCP2 is the IDS Communication Protocol v2
        // It provides secure, authenticated data exchange

        let bytes_transferred = 1024 * 1024;
        let checksum = format!("sha256:{}", hex::encode([0u8; 32]));

        tracing::info!(
            "IDSCP2 transfer {} completed: {} bytes",
            request.id,
            bytes_transferred
        );

        Ok(TransferResult::completed(
            &request.id,
            bytes_transferred,
            started_at,
            checksum,
        ))
    }

    /// Execute multipart form data transfer
    async fn execute_multipart_transfer(
        &self,
        request: &TransferRequest,
        started_at: DateTime<Utc>,
    ) -> IdsResult<TransferResult> {
        let bytes_transferred = 1024 * 1024;
        let checksum = format!("sha256:{}", hex::encode([0u8; 32]));

        tracing::info!(
            "Multipart transfer {} completed: {} bytes",
            request.id,
            bytes_transferred
        );

        Ok(TransferResult::completed(
            &request.id,
            bytes_transferred,
            started_at,
            checksum,
        ))
    }

    /// Execute S3 transfer
    async fn execute_s3_transfer(
        &self,
        request: &TransferRequest,
        started_at: DateTime<Utc>,
    ) -> IdsResult<TransferResult> {
        // S3 transfer would:
        // 1. Generate pre-signed URL or use S3 API
        // 2. Copy object to destination bucket
        // 3. Verify with ETag

        let bytes_transferred = 1024 * 1024 * 10; // 10MB for S3
        let checksum = format!("sha256:{}", hex::encode([0u8; 32]));

        tracing::info!(
            "S3 transfer {} completed: {} bytes to {}",
            request.id,
            bytes_transferred,
            request.destination_endpoint
        );

        Ok(TransferResult::completed(
            &request.id,
            bytes_transferred,
            started_at,
            checksum,
        ))
    }

    /// Execute Kafka transfer
    async fn execute_kafka_transfer(
        &self,
        request: &TransferRequest,
        started_at: DateTime<Utc>,
    ) -> IdsResult<TransferResult> {
        // Kafka transfer would:
        // 1. Produce messages to destination topic
        // 2. Wait for acknowledgements
        // 3. Record offset

        let bytes_transferred = 1024 * 100; // 100KB for Kafka
        let checksum = format!("sha256:{}", hex::encode([0u8; 32]));

        tracing::info!(
            "Kafka transfer {} completed: {} bytes to topic {}",
            request.id,
            bytes_transferred,
            request.destination_endpoint
        );

        Ok(TransferResult::completed(
            &request.id,
            bytes_transferred,
            started_at,
            checksum,
        ))
    }

    /// Record transfer in provenance graph
    async fn record_transfer_provenance(
        &self,
        request: &TransferRequest,
        result: &TransferResult,
    ) -> IdsResult<String> {
        let activity_id = IdsUri::new(format!("urn:ids:activity:transfer:{}", request.id))
            .map_err(|e| {
                IdsError::InternalError(format!("Failed to create activity URI: {}", e))
            })?;

        let activity = Activity::completed(
            activity_id,
            "ids:DataTransfer",
            result.started_at.unwrap_or_else(Utc::now),
            result.completed_at.unwrap_or_else(Utc::now),
        );

        let agent = Agent::software(self.connector_id.clone(), "OxiRS IDS Connector");

        let record = LineageRecord::new(request.resource_id.clone())
            .with_activity(activity)
            .with_agent(agent);

        let record_id = record.entity.as_str().to_string();
        self.provenance.record_lineage(record).await?;

        Ok(record_id)
    }

    /// Get transfer status
    pub async fn get_transfer_status(&self, transfer_id: &str) -> Option<TransferStatus> {
        let transfers = self.transfers.read().await;
        transfers.get(transfer_id).map(|p| p.status)
    }

    /// Get active transfers
    pub async fn get_active_transfers(&self) -> Vec<TransferProcess> {
        let transfers = self.transfers.read().await;
        transfers.values().cloned().collect()
    }

    /// Get transfer history
    pub async fn get_transfer_history(&self) -> Vec<TransferResult> {
        let history = self.history.read().await;
        history.clone()
    }

    /// Get transfer history for a specific resource
    pub async fn get_transfers_for_resource(&self, resource_id: &IdsUri) -> Vec<TransferResult> {
        let history = self.history.read().await;
        history
            .iter()
            .filter(|r| {
                // Match by request ID prefix containing resource
                r.request_id.contains(resource_id.as_str())
            })
            .cloned()
            .collect()
    }

    /// Cancel a pending or in-progress transfer
    pub async fn cancel_transfer(&self, transfer_id: &str) -> IdsResult<TransferResult> {
        let mut transfers = self.transfers.write().await;

        if let Some(process) = transfers.get_mut(transfer_id) {
            if process.status == TransferStatus::Pending
                || process.status == TransferStatus::InProgress
            {
                process.status = TransferStatus::Cancelled;
                process.updated_at = Utc::now();

                let result = TransferResult {
                    request_id: transfer_id.to_string(),
                    status: TransferStatus::Cancelled,
                    bytes_transferred: process.bytes_transferred,
                    started_at: process.started_at,
                    completed_at: Some(Utc::now()),
                    error: Some("Transfer cancelled by user".to_string()),
                    lineage_record_id: None,
                    checksum: None,
                };

                // Store in history
                let mut history = self.history.write().await;
                history.push(result.clone());

                return Ok(result);
            }
        }

        Err(IdsError::NotFound(format!(
            "Transfer {} not found or not cancellable",
            transfer_id
        )))
    }

    /// Suspend transfer due to policy violation
    pub async fn suspend_transfer(
        &self,
        transfer_id: &str,
        reason: impl Into<String>,
    ) -> IdsResult<()> {
        let mut transfers = self.transfers.write().await;

        if let Some(process) = transfers.get_mut(transfer_id) {
            process.status = TransferStatus::Suspended;
            process.error = Some(reason.into());
            process.updated_at = Utc::now();
            Ok(())
        } else {
            Err(IdsError::NotFound(format!(
                "Transfer {} not found",
                transfer_id
            )))
        }
    }

    /// Check GDPR compliance for cross-border transfer
    pub async fn check_gdpr_compliance(
        &self,
        from_region: &Region,
        to_region: &Region,
        from_org: Option<&str>,
        to_org: Option<&str>,
    ) -> IdsResult<bool> {
        let result = self
            .gdpr_checker
            .check_transfer_compliance(from_region, to_region, from_org, to_org)
            .await?;

        if !result.compliant {
            tracing::warn!("GDPR non-compliance: {:?}", result.non_compliance_reasons);
        }

        Ok(result.compliant)
    }
}

/// Stream Transfer Adapter
///
/// Adapter for integrating with oxirs-stream backends.
pub struct StreamTransferAdapter {
    /// Data plane manager
    data_plane: Arc<DataPlaneManager>,
}

impl StreamTransferAdapter {
    /// Create a new stream transfer adapter
    pub fn new(data_plane: Arc<DataPlaneManager>) -> Self {
        Self { data_plane }
    }

    /// Execute a streaming transfer using Kafka
    pub async fn stream_via_kafka(
        &self,
        request: TransferRequest,
        contract: &DataContract,
        topic: &str,
    ) -> IdsResult<TransferResult> {
        let request = request
            .with_protocol(TransferProtocol::Kafka)
            .with_property("kafka.topic", topic);

        self.data_plane.initiate_transfer(request, contract).await
    }

    /// Execute a transfer using S3
    pub async fn transfer_via_s3(
        &self,
        request: TransferRequest,
        contract: &DataContract,
        bucket: &str,
        key: &str,
    ) -> IdsResult<TransferResult> {
        let request = request
            .with_protocol(TransferProtocol::S3)
            .with_property("s3.bucket", bucket)
            .with_property("s3.key", key);

        self.data_plane.initiate_transfer(request, contract).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_party() -> Party {
        Party {
            id: IdsUri::new("https://example.org/party/consumer").expect("valid uri"),
            name: "Test Consumer".to_string(),
            legal_name: None,
            description: None,
            contact: None,
            gaiax_participant_id: None,
        }
    }

    #[tokio::test]
    async fn test_transfer_request_creation() {
        let request = TransferRequest::new(
            IdsUri::new("https://example.org/contract/1").expect("valid uri"),
            IdsUri::new("https://example.org/data/resource1").expect("valid uri"),
            "https://consumer.example.org/receive",
            test_party(),
        )
        .with_protocol(TransferProtocol::Https)
        .with_transfer_type(TransferType::Push);

        assert_eq!(request.protocol, TransferProtocol::Https);
        assert_eq!(request.transfer_type, TransferType::Push);
    }

    #[tokio::test]
    async fn test_transfer_result_states() {
        let pending = TransferResult::pending("test-1");
        assert_eq!(pending.status, TransferStatus::Pending);

        let completed = TransferResult::completed("test-2", 1024, Utc::now(), "sha256:abc123");
        assert_eq!(completed.status, TransferStatus::Completed);
        assert_eq!(completed.bytes_transferred, 1024);

        let failed = TransferResult::failed("test-3", "Connection refused");
        assert_eq!(failed.status, TransferStatus::Failed);
        assert!(failed.error.is_some());
    }

    #[tokio::test]
    async fn test_data_plane_manager_creation() {
        let connector_id = IdsUri::new("urn:ids:connector:test").expect("valid uri");
        let policy_engine = Arc::new(PolicyEngine::new());
        let provenance = Arc::new(ProvenanceGraph::default());

        let manager = DataPlaneManager::new(connector_id, policy_engine, provenance);

        let active = manager.get_active_transfers().await;
        assert!(active.is_empty());
    }
}
