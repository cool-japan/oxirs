//! IDS Management REST API
//!
//! REST endpoints for IDS connector management, contract negotiation,
//! and data transfer operations.

use super::broker::{BrokerClient, CatalogQuery, ConnectorSelfDescription, RegistrationResult};
use super::catalog::DataResource;
use super::connector::{IdsConnector, IdsConnectorConfig};
use super::contract::{ContractManager, ContractState, DataContract};
use super::data_plane::{
    DataPlaneManager, TransferRequest, TransferResult, TransferStatus, TransferType,
};
use super::identity::DapsClient;
use super::policy::{EvaluationContext, OdrlAction, PolicyDecision, PolicyEngine};
use super::types::{IdsError, IdsResult, IdsUri, Party, SecurityProfile, TransferProtocol};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Json, Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// IDS API State
pub struct IdsApiState {
    /// IDS Connector
    pub connector: Arc<IdsConnector>,
    /// Data Plane Manager
    pub data_plane: Arc<DataPlaneManager>,
    /// Broker Client (optional)
    pub broker_client: Option<Arc<BrokerClient>>,
}

impl IdsApiState {
    /// Create new API state
    pub fn new(connector: Arc<IdsConnector>, data_plane: Arc<DataPlaneManager>) -> Self {
        Self {
            connector,
            data_plane,
            broker_client: None,
        }
    }

    /// Set broker client
    pub fn with_broker(mut self, broker: Arc<BrokerClient>) -> Self {
        self.broker_client = Some(broker);
        self
    }
}

/// Build IDS API router
pub fn ids_router(state: Arc<IdsApiState>) -> Router {
    Router::new()
        // Connector Management
        .route("/connector", get(get_connector_info))
        .route("/connector/self-description", get(get_self_description))
        // Catalog Management
        .route("/catalog", get(get_catalog))
        .route("/catalog/resources", get(list_resources))
        .route("/catalog/resources", post(add_resource))
        .route("/catalog/resources/:id", get(get_resource))
        .route("/catalog/resources/:id", delete(remove_resource))
        // Contract Management
        .route("/contracts", get(list_contracts))
        .route("/contracts", post(initiate_negotiation))
        .route("/contracts/:id", get(get_contract))
        .route("/contracts/:id/accept", post(accept_contract))
        .route("/contracts/:id/reject", post(reject_contract))
        .route("/contracts/:id/terminate", post(terminate_contract))
        // Transfer Management
        .route("/transfers", get(list_transfers))
        .route("/transfers", post(initiate_transfer))
        .route("/transfers/:id", get(get_transfer))
        .route("/transfers/:id/cancel", post(cancel_transfer))
        // Policy Evaluation
        .route("/policy/evaluate", post(evaluate_policy))
        // Broker Integration
        .route("/broker/register", post(register_with_broker))
        .route("/broker/unregister", post(unregister_from_broker))
        .route("/broker/catalog", get(query_broker_catalog))
        .with_state(state)
}

// ===== Request/Response Types =====

/// Connector info response
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectorInfoResponse {
    pub connector_id: String,
    pub title: String,
    pub description: String,
    pub security_profile: SecurityProfile,
    pub supported_protocols: Vec<TransferProtocol>,
    pub version: String,
}

/// Self-description response
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelfDescriptionResponse {
    #[serde(rename = "@context")]
    pub context: serde_json::Value,
    #[serde(rename = "@id")]
    pub id: String,
    #[serde(rename = "@type")]
    pub connector_type: String,
    pub title: String,
    pub description: String,
    pub security_profile: String,
    pub resource_catalog: Vec<ResourceDto>,
    pub version: String,
}

/// Resource DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceDto {
    #[serde(rename = "@id")]
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub content_type: Option<String>,
    pub keywords: Vec<String>,
    pub language: Option<String>,
    pub created: DateTime<Utc>,
    pub modified: DateTime<Utc>,
}

impl From<&DataResource> for ResourceDto {
    fn from(r: &DataResource) -> Self {
        Self {
            id: r.id.as_str().to_string(),
            title: r.title.clone(),
            description: r.description.clone(),
            content_type: r.content_type.clone(),
            keywords: r.keywords.clone(),
            language: r.language.clone(),
            created: r.created(),
            modified: r.modified(),
        }
    }
}

/// Add resource request
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AddResourceRequest {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub content_type: Option<String>,
    pub keywords: Option<Vec<String>>,
    pub language: Option<String>,
}

/// Contract DTO
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContractDto {
    pub contract_id: String,
    pub consumer: PartyDto,
    pub provider: PartyDto,
    pub state: ContractState,
    pub asset_id: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Party DTO
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PartyDto {
    pub id: String,
    pub name: String,
}

impl From<&Party> for PartyDto {
    fn from(p: &Party) -> Self {
        Self {
            id: p.id.as_str().to_string(),
            name: p.name.clone(),
        }
    }
}

/// Initiate negotiation request
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitiateNegotiationRequest {
    pub provider_id: String,
    pub asset_id: String,
    pub consumer_id: String,
    pub consumer_name: String,
    pub offer_id: Option<String>,
}

/// Transfer request DTO
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TransferRequestDto {
    pub contract_id: String,
    pub resource_id: String,
    pub destination_endpoint: String,
    pub protocol: Option<TransferProtocol>,
    pub transfer_type: Option<TransferType>,
}

/// Transfer response DTO
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TransferResponseDto {
    pub transfer_id: String,
    pub status: TransferStatus,
    pub bytes_transferred: u64,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

impl From<&TransferResult> for TransferResponseDto {
    fn from(r: &TransferResult) -> Self {
        Self {
            transfer_id: r.request_id.clone(),
            status: r.status,
            bytes_transferred: r.bytes_transferred,
            started_at: r.started_at,
            completed_at: r.completed_at,
            error: r.error.clone(),
        }
    }
}

/// Policy evaluation request
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PolicyEvaluationRequest {
    pub policy_id: String,
    pub action: String,
    pub resource_id: Option<String>,
    pub connector_id: Option<String>,
    pub purpose: Option<String>,
}

/// Policy evaluation response
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PolicyEvaluationResponse {
    pub permitted: bool,
    pub reason: Option<String>,
    pub duties: Vec<String>,
}

/// Broker registration request
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrokerRegistrationRequest {
    pub broker_url: Option<String>,
}

/// Broker catalog query params
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrokerCatalogQuery {
    pub keyword: Option<String>,
    pub content_type: Option<String>,
    pub limit: Option<u32>,
}

/// API Error response
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: String,
    pub message: String,
    pub status: u16,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(self)).into_response()
    }
}

impl From<IdsError> for ApiError {
    fn from(e: IdsError) -> Self {
        Self {
            error: format!("{:?}", e),
            message: e.to_string(),
            status: e.status_code(),
        }
    }
}

// ===== Handlers =====

/// Get connector information
async fn get_connector_info(
    State(state): State<Arc<IdsApiState>>,
) -> Result<Json<ConnectorInfoResponse>, ApiError> {
    let connector = &state.connector;
    let config = connector.connector_id();

    Ok(Json(ConnectorInfoResponse {
        connector_id: config.as_str().to_string(),
        title: "OxiRS IDS Connector".to_string(),
        description: "Semantic Web Data Space Connector".to_string(),
        security_profile: SecurityProfile::TrustSecurityProfile,
        supported_protocols: vec![
            TransferProtocol::Https,
            TransferProtocol::S3,
            TransferProtocol::Kafka,
        ],
        version: super::IDS_VERSION.to_string(),
    }))
}

/// Get self-description
async fn get_self_description(
    State(state): State<Arc<IdsApiState>>,
) -> Result<Json<SelfDescriptionResponse>, ApiError> {
    let connector = &state.connector;
    let catalog = connector.catalog();
    let resources = catalog.list_all().await;

    Ok(Json(SelfDescriptionResponse {
        context: serde_json::json!({
            "ids": "https://w3id.org/idsa/core/",
            "idsc": "https://w3id.org/idsa/code/"
        }),
        id: connector.connector_id().as_str().to_string(),
        connector_type: "ids:BaseConnector".to_string(),
        title: "OxiRS IDS Connector".to_string(),
        description: "Semantic Web Data Space Connector".to_string(),
        security_profile: SecurityProfile::TrustSecurityProfile.to_uri().to_string(),
        resource_catalog: resources.iter().map(ResourceDto::from).collect(),
        version: super::IDS_VERSION.to_string(),
    }))
}

/// Get catalog
async fn get_catalog(
    State(state): State<Arc<IdsApiState>>,
) -> Result<Json<Vec<ResourceDto>>, ApiError> {
    let catalog = state.connector.catalog();
    let resources = catalog.list_all().await;

    Ok(Json(resources.iter().map(ResourceDto::from).collect()))
}

/// List resources
async fn list_resources(
    State(state): State<Arc<IdsApiState>>,
) -> Result<Json<Vec<ResourceDto>>, ApiError> {
    let catalog = state.connector.catalog();
    let resources = catalog.list_all().await;

    Ok(Json(resources.iter().map(ResourceDto::from).collect()))
}

/// Add resource to catalog
async fn add_resource(
    State(state): State<Arc<IdsApiState>>,
    Json(req): Json<AddResourceRequest>,
) -> Result<Json<ResourceDto>, ApiError> {
    let id = IdsUri::new(&req.id)?;

    // Get the connector's ID as the publisher
    let publisher = state.connector.connector_id().clone();
    let now = Utc::now();

    let resource = DataResource {
        id: id.clone(),
        title: req.title,
        description: req.description,
        content_type: req.content_type,
        keywords: req.keywords.unwrap_or_default(),
        language: req.language,
        publisher,
        distributions: Vec::new(),
        access_url: None,
        download_url: None,
        byte_size: None,
        checksum: None,
        license: None,
        version: None,
        created_at: now,
        modified_at: now,
    };

    let catalog = state.connector.catalog();
    catalog.add_resource(resource.clone()).await;

    Ok(Json(ResourceDto::from(&resource)))
}

/// Get resource by ID
async fn get_resource(
    State(state): State<Arc<IdsApiState>>,
    Path(id): Path<String>,
) -> Result<Json<ResourceDto>, ApiError> {
    let resource_id = IdsUri::new(&id)?;
    let catalog = state.connector.catalog();

    match catalog.get_resource(resource_id.as_str()).await {
        Some(resource) => Ok(Json(ResourceDto::from(&resource))),
        None => Err(ApiError {
            error: "NotFound".to_string(),
            message: format!("Resource {} not found", id),
            status: 404,
        }),
    }
}

/// Remove resource from catalog
async fn remove_resource(
    State(state): State<Arc<IdsApiState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let resource_id = IdsUri::new(&id)?;
    let catalog = state.connector.catalog();

    if catalog.remove_resource(resource_id.as_str()).await {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError {
            error: "NotFound".to_string(),
            message: format!("Resource {} not found", id),
            status: 404,
        })
    }
}

/// List contracts
async fn list_contracts(
    State(state): State<Arc<IdsApiState>>,
) -> Result<Json<Vec<ContractDto>>, ApiError> {
    let contract_manager = state.connector.contract_manager();
    let contracts = contract_manager.list_contracts().await;

    Ok(Json(
        contracts
            .iter()
            .map(|c| ContractDto {
                contract_id: c.contract_id.as_str().to_string(),
                consumer: PartyDto::from(&c.consumer),
                provider: PartyDto::from(&c.provider),
                state: c.state,
                asset_id: c.target_asset.asset_id.as_str().to_string(),
                created_at: c.created_at,
                expires_at: c.contract_end,
            })
            .collect(),
    ))
}

/// Initiate contract negotiation
async fn initiate_negotiation(
    State(state): State<Arc<IdsApiState>>,
    Json(req): Json<InitiateNegotiationRequest>,
) -> Result<Json<ContractDto>, ApiError> {
    let contract_manager = state.connector.contract_manager();

    let provider_id = IdsUri::new(&req.provider_id)?;
    let consumer_id = IdsUri::new(&req.consumer_id)?;
    let asset_id = IdsUri::new(&req.asset_id)?;

    let consumer = Party {
        id: consumer_id,
        name: req.consumer_name.clone(),
        legal_name: None,
        description: None,
        contact: None,
        gaiax_participant_id: None,
    };

    let contract = contract_manager
        .initiate_negotiation(provider_id, asset_id, consumer)
        .await
        .map_err(ApiError::from)?;

    Ok(Json(ContractDto {
        contract_id: contract.contract_id.as_str().to_string(),
        consumer: PartyDto::from(&contract.consumer),
        provider: PartyDto::from(&contract.provider),
        state: contract.state,
        asset_id: contract.target_asset.asset_id.as_str().to_string(),
        created_at: contract.created_at,
        expires_at: contract.contract_end,
    }))
}

/// Get contract by ID
async fn get_contract(
    State(state): State<Arc<IdsApiState>>,
    Path(id): Path<String>,
) -> Result<Json<ContractDto>, ApiError> {
    let contract_id = IdsUri::new(&id)?;
    let contract_manager = state.connector.contract_manager();

    match contract_manager.get_contract_uri(&contract_id).await {
        Some(contract) => Ok(Json(ContractDto {
            contract_id: contract.contract_id.as_str().to_string(),
            consumer: PartyDto::from(&contract.consumer),
            provider: PartyDto::from(&contract.provider),
            state: contract.state,
            asset_id: contract.target_asset.asset_id.as_str().to_string(),
            created_at: contract.created_at,
            expires_at: contract.contract_end,
        })),
        None => Err(ApiError {
            error: "NotFound".to_string(),
            message: format!("Contract {} not found", id),
            status: 404,
        }),
    }
}

/// Accept contract
async fn accept_contract(
    State(state): State<Arc<IdsApiState>>,
    Path(id): Path<String>,
) -> Result<Json<ContractDto>, ApiError> {
    let contract_id = IdsUri::new(&id)?;
    let contract_manager = state.connector.contract_manager();

    let contract = contract_manager
        .accept_contract(&contract_id)
        .await
        .map_err(ApiError::from)?;

    Ok(Json(ContractDto {
        contract_id: contract.contract_id.as_str().to_string(),
        consumer: PartyDto::from(&contract.consumer),
        provider: PartyDto::from(&contract.provider),
        state: contract.state,
        asset_id: contract.target_asset.asset_id.as_str().to_string(),
        created_at: contract.created_at,
        expires_at: contract.contract_end,
    }))
}

/// Reject contract
async fn reject_contract(
    State(state): State<Arc<IdsApiState>>,
    Path(id): Path<String>,
) -> Result<Json<ContractDto>, ApiError> {
    let contract_id = IdsUri::new(&id)?;
    let contract_manager = state.connector.contract_manager();

    let contract = contract_manager
        .reject_contract(&contract_id)
        .await
        .map_err(ApiError::from)?;

    Ok(Json(ContractDto {
        contract_id: contract.contract_id.as_str().to_string(),
        consumer: PartyDto::from(&contract.consumer),
        provider: PartyDto::from(&contract.provider),
        state: contract.state,
        asset_id: contract.target_asset.asset_id.as_str().to_string(),
        created_at: contract.created_at,
        expires_at: contract.contract_end,
    }))
}

/// Terminate contract
async fn terminate_contract(
    State(state): State<Arc<IdsApiState>>,
    Path(id): Path<String>,
) -> Result<Json<ContractDto>, ApiError> {
    let contract_id = IdsUri::new(&id)?;
    let contract_manager = state.connector.contract_manager();

    let contract = contract_manager
        .terminate_contract(&contract_id)
        .await
        .map_err(ApiError::from)?;

    Ok(Json(ContractDto {
        contract_id: contract.contract_id.as_str().to_string(),
        consumer: PartyDto::from(&contract.consumer),
        provider: PartyDto::from(&contract.provider),
        state: contract.state,
        asset_id: contract.target_asset.asset_id.as_str().to_string(),
        created_at: contract.created_at,
        expires_at: contract.contract_end,
    }))
}

/// List transfers
async fn list_transfers(
    State(state): State<Arc<IdsApiState>>,
) -> Result<Json<Vec<TransferResponseDto>>, ApiError> {
    let history = state.data_plane.get_transfer_history().await;

    Ok(Json(
        history.iter().map(TransferResponseDto::from).collect(),
    ))
}

/// Initiate data transfer
async fn initiate_transfer(
    State(state): State<Arc<IdsApiState>>,
    Json(req): Json<TransferRequestDto>,
) -> Result<Json<TransferResponseDto>, ApiError> {
    let contract_id = IdsUri::new(&req.contract_id)?;
    let resource_id = IdsUri::new(&req.resource_id)?;

    // Get the contract
    let contract_manager = state.connector.contract_manager();
    let contract = contract_manager
        .get_contract_uri(&contract_id)
        .await
        .ok_or_else(|| ApiError {
            error: "NotFound".to_string(),
            message: format!("Contract {} not found", contract_id),
            status: 404,
        })?;

    // Create transfer request
    let mut transfer_request = TransferRequest::new(
        contract_id,
        resource_id,
        &req.destination_endpoint,
        contract.consumer.clone(),
    );

    if let Some(protocol) = req.protocol {
        transfer_request = transfer_request.with_protocol(protocol);
    }

    if let Some(transfer_type) = req.transfer_type {
        transfer_request = transfer_request.with_transfer_type(transfer_type);
    }

    // Initiate transfer
    let result = state
        .data_plane
        .initiate_transfer(transfer_request, &contract)
        .await
        .map_err(ApiError::from)?;

    Ok(Json(TransferResponseDto::from(&result)))
}

/// Get transfer by ID
async fn get_transfer(
    State(state): State<Arc<IdsApiState>>,
    Path(id): Path<String>,
) -> Result<Json<TransferResponseDto>, ApiError> {
    let status = state.data_plane.get_transfer_status(&id).await;

    match status {
        Some(status) => Ok(Json(TransferResponseDto {
            transfer_id: id,
            status,
            bytes_transferred: 0,
            started_at: None,
            completed_at: None,
            error: None,
        })),
        None => {
            // Check history
            let history = state.data_plane.get_transfer_history().await;
            match history.iter().find(|r| r.request_id == id) {
                Some(result) => Ok(Json(TransferResponseDto::from(result))),
                None => Err(ApiError {
                    error: "NotFound".to_string(),
                    message: format!("Transfer {} not found", id),
                    status: 404,
                }),
            }
        }
    }
}

/// Cancel transfer
async fn cancel_transfer(
    State(state): State<Arc<IdsApiState>>,
    Path(id): Path<String>,
) -> Result<Json<TransferResponseDto>, ApiError> {
    let result = state
        .data_plane
        .cancel_transfer(&id)
        .await
        .map_err(ApiError::from)?;

    Ok(Json(TransferResponseDto::from(&result)))
}

/// Evaluate policy
async fn evaluate_policy(
    State(state): State<Arc<IdsApiState>>,
    Json(req): Json<PolicyEvaluationRequest>,
) -> Result<Json<PolicyEvaluationResponse>, ApiError> {
    let policy_id = IdsUri::new(&req.policy_id)?;
    let policy_engine = state.connector.policy_engine();

    let action = match req.action.as_str() {
        "read" => OdrlAction::Read,
        "use" => OdrlAction::Use,
        "modify" => OdrlAction::Modify,
        "delete" => OdrlAction::Delete,
        "distribute" => OdrlAction::Distribute,
        _ => OdrlAction::Custom(req.action.clone()),
    };

    let mut context = EvaluationContext::new();

    if let Some(ref resource_id) = req.resource_id {
        if let Ok(uri) = IdsUri::new(resource_id) {
            context = context.with_resource(uri);
        }
    }

    if let Some(ref connector_id) = req.connector_id {
        context = context.with_connector_id(connector_id);
    }

    if let Some(ref purpose) = req.purpose {
        context = context.with_purpose(purpose);
    }

    let decision = policy_engine
        .evaluate(&policy_id, &action, &context)
        .await
        .map_err(ApiError::from)?;

    match decision {
        PolicyDecision::Permit { duties, .. } => Ok(Json(PolicyEvaluationResponse {
            permitted: true,
            reason: None,
            duties: duties.iter().map(|d| format!("{:?}", d.action)).collect(),
        })),
        PolicyDecision::Deny { reason, .. } => Ok(Json(PolicyEvaluationResponse {
            permitted: false,
            reason: Some(reason),
            duties: Vec::new(),
        })),
        PolicyDecision::NotApplicable => Ok(Json(PolicyEvaluationResponse {
            permitted: false,
            reason: Some("No applicable policy found".to_string()),
            duties: Vec::new(),
        })),
    }
}

/// Register with broker
async fn register_with_broker(
    State(state): State<Arc<IdsApiState>>,
    Json(_req): Json<BrokerRegistrationRequest>,
) -> Result<Json<RegistrationResult>, ApiError> {
    let broker = state.broker_client.as_ref().ok_or_else(|| ApiError {
        error: "NoBroker".to_string(),
        message: "No broker client configured".to_string(),
        status: 503,
    })?;

    let connector = &state.connector;
    let catalog = connector.catalog();
    let resources = catalog.list_all().await;

    // Create self-description with minimal Party info
    let curator = Party {
        id: connector.connector_id().clone(),
        name: "OxiRS".to_string(),
        legal_name: None,
        description: None,
        contact: None,
        gaiax_participant_id: None,
    };

    let self_desc = ConnectorSelfDescription::new(
        connector.connector_id().clone(),
        "OxiRS IDS Connector",
        "Semantic Web Data Space Connector",
        curator,
    );

    let result = broker
        .register_connector(&self_desc)
        .await
        .map_err(ApiError::from)?;

    Ok(Json(result))
}

/// Unregister from broker
async fn unregister_from_broker(
    State(state): State<Arc<IdsApiState>>,
) -> Result<StatusCode, ApiError> {
    let broker = state.broker_client.as_ref().ok_or_else(|| ApiError {
        error: "NoBroker".to_string(),
        message: "No broker client configured".to_string(),
        status: 503,
    })?;

    broker
        .unregister_connector()
        .await
        .map_err(ApiError::from)?;

    Ok(StatusCode::NO_CONTENT)
}

/// Query broker catalog
async fn query_broker_catalog(
    State(state): State<Arc<IdsApiState>>,
    Query(params): Query<BrokerCatalogQuery>,
) -> Result<Json<Vec<ResourceDto>>, ApiError> {
    let broker = state.broker_client.as_ref().ok_or_else(|| ApiError {
        error: "NoBroker".to_string(),
        message: "No broker client configured".to_string(),
        status: 503,
    })?;

    let query = CatalogQuery {
        keyword: params.keyword,
        content_type: params.content_type,
        limit: params.limit,
        ..Default::default()
    };

    let resources = broker.query_catalog(&query).await.map_err(ApiError::from)?;

    Ok(Json(
        resources
            .iter()
            .map(|r| ResourceDto {
                id: r.id.as_str().to_string(),
                title: r.title.clone(),
                description: r.description.clone(),
                content_type: r.content_type.clone(),
                keywords: r.keywords.clone(),
                language: r.language.clone(),
                created: r.created_at,
                modified: r.modified_at,
            })
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_dto_conversion() {
        let now = Utc::now();
        let resource = DataResource {
            id: IdsUri::new("urn:ids:resource:1").expect("valid uri"),
            title: "Test Resource".to_string(),
            description: Some("A test resource".to_string()),
            content_type: Some("application/json".to_string()),
            keywords: vec!["test".to_string()],
            language: Some("en".to_string()),
            publisher: IdsUri::new("urn:ids:connector:test").expect("valid uri"),
            distributions: Vec::new(),
            access_url: None,
            download_url: None,
            byte_size: None,
            checksum: None,
            license: None,
            version: None,
            created_at: now,
            modified_at: now,
        };

        let dto = ResourceDto::from(&resource);
        assert_eq!(dto.title, "Test Resource");
        assert_eq!(dto.keywords, vec!["test".to_string()]);
    }

    #[test]
    fn test_api_error_from_ids_error() {
        let ids_error = IdsError::ContractNotFound("test".to_string());
        let api_error = ApiError::from(ids_error);

        assert_eq!(api_error.status, 404);
        assert!(api_error.message.contains("test"));
    }
}
