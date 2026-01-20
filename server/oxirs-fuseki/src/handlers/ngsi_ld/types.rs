//! NGSI-LD v1.6 Type Definitions
//!
//! ETSI GS CIM 009 V1.6.1 compliant types for NGSI-LD API
//! <https://www.etsi.org/deliver/etsi_gs/CIM/001_099/009/01.06.01_60/gs_CIM009v010601p.pdf>
//!
//! Compatible with:
//! - FIWARE Context Broker
//! - Japan PLATEAU Smart City Platform
//! - EU NGSI-LD Reference Implementations

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// NGSI-LD Core Types
// ============================================================================

/// NGSI-LD Entity representation
///
/// An Entity is the core information element in NGSI-LD, representing
/// a thing in the real world (physical or conceptual).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NgsiEntity {
    /// Entity identifier (URI)
    #[serde(rename = "@id")]
    pub id: String,

    /// Entity type (URI)
    #[serde(rename = "@type")]
    pub entity_type: NgsiType,

    /// JSON-LD @context
    #[serde(rename = "@context", skip_serializing_if = "Option::is_none")]
    pub context: Option<NgsiContext>,

    /// Entity scope (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<Vec<String>>,

    /// Entity location (optional geo-property)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<GeoProperty>,

    /// Observation space (optional geo-property)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation_space: Option<GeoProperty>,

    /// Operation space (optional geo-property)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_space: Option<GeoProperty>,

    /// Created at timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,

    /// Modified at timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<DateTime<Utc>>,

    /// Additional properties (flattened into entity)
    #[serde(flatten)]
    pub properties: HashMap<String, NgsiAttribute>,
}

impl NgsiEntity {
    /// Create a new entity with the given ID and type
    pub fn new(id: impl Into<String>, entity_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            entity_type: NgsiType::Single(entity_type.into()),
            context: None,
            scope: None,
            location: None,
            observation_space: None,
            operation_space: None,
            created_at: Some(Utc::now()),
            modified_at: Some(Utc::now()),
            properties: HashMap::new(),
        }
    }

    /// Add a property to the entity
    pub fn with_property(mut self, name: impl Into<String>, property: NgsiProperty) -> Self {
        self.properties
            .insert(name.into(), NgsiAttribute::Property(property));
        self
    }

    /// Add a relationship to the entity
    pub fn with_relationship(
        mut self,
        name: impl Into<String>,
        relationship: NgsiRelationship,
    ) -> Self {
        self.properties
            .insert(name.into(), NgsiAttribute::Relationship(relationship));
        self
    }

    /// Get entity ID without prefix
    pub fn short_id(&self) -> &str {
        self.id
            .rsplit_once(':')
            .map(|(_, id)| id)
            .unwrap_or(&self.id)
    }
}

/// NGSI-LD type (single or multiple)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NgsiType {
    /// Single type
    Single(String),
    /// Multiple types
    Multiple(Vec<String>),
}

impl NgsiType {
    /// Get the primary type
    pub fn primary(&self) -> &str {
        match self {
            NgsiType::Single(t) => t,
            NgsiType::Multiple(ts) => ts.first().map(|s| s.as_str()).unwrap_or(""),
        }
    }

    /// Get all types as a vector
    pub fn all(&self) -> Vec<&str> {
        match self {
            NgsiType::Single(t) => vec![t],
            NgsiType::Multiple(ts) => ts.iter().map(|s| s.as_str()).collect(),
        }
    }
}

/// NGSI-LD @context
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NgsiContext {
    /// URI reference to context
    Uri(String),
    /// Inline context object
    Object(HashMap<String, serde_json::Value>),
    /// Array of context references
    Array(Vec<NgsiContext>),
}

impl Default for NgsiContext {
    fn default() -> Self {
        NgsiContext::Uri("https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld".to_string())
    }
}

// ============================================================================
// NGSI-LD Attribute Types
// ============================================================================

/// NGSI-LD Attribute (Property, Relationship, or GeoProperty)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NgsiAttribute {
    /// Property with value
    Property(NgsiProperty),
    /// Relationship to another entity
    Relationship(NgsiRelationship),
    /// Geographic property
    GeoProperty(GeoProperty),
}

/// NGSI-LD Property
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NgsiProperty {
    /// Type indicator ("Property")
    #[serde(rename = "type")]
    pub property_type: PropertyType,

    /// Property value (any JSON value)
    pub value: serde_json::Value,

    /// Observation timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_at: Option<DateTime<Utc>>,

    /// Unit code (UN/CEFACT)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_code: Option<String>,

    /// Dataset ID for multi-attribute
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,

    /// Instance ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,

    /// Created at timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,

    /// Modified at timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<DateTime<Utc>>,

    /// Nested properties of property
    #[serde(flatten)]
    pub sub_properties: HashMap<String, NgsiAttribute>,
}

impl NgsiProperty {
    /// Create a simple property with a value
    pub fn new(value: impl Into<serde_json::Value>) -> Self {
        Self {
            property_type: PropertyType::Property,
            value: value.into(),
            observed_at: None,
            unit_code: None,
            dataset_id: None,
            instance_id: None,
            created_at: None,
            modified_at: None,
            sub_properties: HashMap::new(),
        }
    }

    /// Create a property with observation timestamp
    pub fn with_observed_at(
        value: impl Into<serde_json::Value>,
        observed_at: DateTime<Utc>,
    ) -> Self {
        Self {
            property_type: PropertyType::Property,
            value: value.into(),
            observed_at: Some(observed_at),
            unit_code: None,
            dataset_id: None,
            instance_id: None,
            created_at: None,
            modified_at: None,
            sub_properties: HashMap::new(),
        }
    }

    /// Add unit code
    pub fn with_unit(mut self, unit_code: impl Into<String>) -> Self {
        self.unit_code = Some(unit_code.into());
        self
    }
}

/// Property type indicator
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PropertyType {
    /// Standard property
    Property,
}

/// NGSI-LD Relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NgsiRelationship {
    /// Type indicator ("Relationship")
    #[serde(rename = "type")]
    pub rel_type: RelationshipType,

    /// Target entity URI
    pub object: String,

    /// Observation timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_at: Option<DateTime<Utc>>,

    /// Dataset ID for multi-attribute
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,

    /// Instance ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,

    /// Created at timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,

    /// Modified at timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<DateTime<Utc>>,

    /// Nested properties of relationship
    #[serde(flatten)]
    pub sub_properties: HashMap<String, NgsiAttribute>,
}

impl NgsiRelationship {
    /// Create a relationship to another entity
    pub fn new(object: impl Into<String>) -> Self {
        Self {
            rel_type: RelationshipType::Relationship,
            object: object.into(),
            observed_at: None,
            dataset_id: None,
            instance_id: None,
            created_at: None,
            modified_at: None,
            sub_properties: HashMap::new(),
        }
    }
}

/// Relationship type indicator
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationshipType {
    /// Standard relationship
    Relationship,
}

// ============================================================================
// GeoProperty Types (GeoJSON)
// ============================================================================

/// NGSI-LD GeoProperty
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeoProperty {
    /// Type indicator ("GeoProperty")
    #[serde(rename = "type")]
    pub geo_type: GeoPropertyType,

    /// GeoJSON value
    pub value: GeoJsonValue,

    /// Observation timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_at: Option<DateTime<Utc>>,

    /// Dataset ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,

    /// Instance ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
}

impl GeoProperty {
    /// Create a point geo-property
    pub fn point(longitude: f64, latitude: f64) -> Self {
        Self {
            geo_type: GeoPropertyType::GeoProperty,
            value: GeoJsonValue::Point(GeoJsonPoint {
                geo_type: "Point".to_string(),
                coordinates: vec![longitude, latitude],
            }),
            observed_at: None,
            dataset_id: None,
            instance_id: None,
        }
    }

    /// Create a polygon geo-property
    pub fn polygon(coordinates: Vec<Vec<Vec<f64>>>) -> Self {
        Self {
            geo_type: GeoPropertyType::GeoProperty,
            value: GeoJsonValue::Polygon(GeoJsonPolygon {
                geo_type: "Polygon".to_string(),
                coordinates,
            }),
            observed_at: None,
            dataset_id: None,
            instance_id: None,
        }
    }
}

/// GeoProperty type indicator
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GeoPropertyType {
    /// Standard geo-property
    GeoProperty,
}

/// GeoJSON value types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GeoJsonValue {
    /// Point geometry
    Point(GeoJsonPoint),
    /// LineString geometry
    LineString(GeoJsonLineString),
    /// Polygon geometry
    Polygon(GeoJsonPolygon),
    /// MultiPoint geometry
    MultiPoint(GeoJsonMultiPoint),
    /// MultiLineString geometry
    MultiLineString(GeoJsonMultiLineString),
    /// MultiPolygon geometry
    MultiPolygon(GeoJsonMultiPolygon),
}

/// GeoJSON Point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoJsonPoint {
    #[serde(rename = "type")]
    pub geo_type: String,
    pub coordinates: Vec<f64>,
}

/// GeoJSON LineString
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoJsonLineString {
    #[serde(rename = "type")]
    pub geo_type: String,
    pub coordinates: Vec<Vec<f64>>,
}

/// GeoJSON Polygon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoJsonPolygon {
    #[serde(rename = "type")]
    pub geo_type: String,
    pub coordinates: Vec<Vec<Vec<f64>>>,
}

/// GeoJSON MultiPoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoJsonMultiPoint {
    #[serde(rename = "type")]
    pub geo_type: String,
    pub coordinates: Vec<Vec<f64>>,
}

/// GeoJSON MultiLineString
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoJsonMultiLineString {
    #[serde(rename = "type")]
    pub geo_type: String,
    pub coordinates: Vec<Vec<Vec<f64>>>,
}

/// GeoJSON MultiPolygon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoJsonMultiPolygon {
    #[serde(rename = "type")]
    pub geo_type: String,
    pub coordinates: Vec<Vec<Vec<Vec<f64>>>>,
}

// ============================================================================
// Subscription Types
// ============================================================================

/// NGSI-LD Subscription
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NgsiSubscription {
    /// Subscription ID
    #[serde(rename = "@id", skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Type ("Subscription")
    #[serde(rename = "@type")]
    pub sub_type: SubscriptionType,

    /// JSON-LD @context
    #[serde(rename = "@context", skip_serializing_if = "Option::is_none")]
    pub context: Option<NgsiContext>,

    /// Subscription name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Subscription description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Entity selectors
    pub entities: Vec<EntitySelector>,

    /// Watched attributes (empty = all)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub watched_attributes: Option<Vec<String>>,

    /// Notification trigger
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notification_trigger: Option<Vec<NotificationTrigger>>,

    /// Query filter (NGSI-LD query language)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q: Option<String>,

    /// Geo-query filter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geo_q: Option<GeoQuery>,

    /// CSF (Context Source Filter)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub csf: Option<String>,

    /// Notification configuration
    pub notification: NotificationParams,

    /// Expiration timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,

    /// Throttling in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub throttling: Option<u64>,

    /// Temporal query
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temporal_q: Option<TemporalQuery>,

    /// Scope query
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope_q: Option<String>,

    /// Language filter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,

    /// Status
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<SubscriptionStatus>,

    /// Is active
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_active: Option<bool>,

    /// Created at
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,

    /// Modified at
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<DateTime<Utc>>,
}

/// Subscription type indicator
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SubscriptionType {
    /// Standard subscription
    Subscription,
}

/// Subscription status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SubscriptionStatus {
    /// Subscription is active
    Active,
    /// Subscription is paused
    Paused,
    /// Subscription has expired
    Expired,
}

/// Entity selector for subscriptions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntitySelector {
    /// Entity ID pattern
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Entity ID pattern (regex)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_pattern: Option<String>,

    /// Entity type
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub entity_type: Option<String>,
}

/// Notification trigger types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum NotificationTrigger {
    /// Triggered on entity creation
    EntityCreated,
    /// Triggered on entity update
    EntityUpdated,
    /// Triggered on entity deletion
    EntityDeleted,
    /// Triggered on attribute creation
    AttributeCreated,
    /// Triggered on attribute update
    AttributeUpdated,
    /// Triggered on attribute deletion
    AttributeDeleted,
}

/// Notification parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NotificationParams {
    /// Attributes to include (empty = all)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Vec<String>>,

    /// Notification format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<NotificationFormat>,

    /// Endpoint configuration
    pub endpoint: NotificationEndpoint,

    /// Show changes only
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_changes: Option<bool>,

    /// Include sysAttrs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sys_attrs: Option<bool>,

    /// Last notification timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_notification: Option<DateTime<Utc>>,

    /// Last success timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_success: Option<DateTime<Utc>>,

    /// Last failure timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_failure: Option<DateTime<Utc>>,

    /// Times sent counter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub times_sent: Option<u64>,

    /// Times failed counter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub times_failed: Option<u64>,
}

/// Notification format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NotificationFormat {
    /// Normalized (full NGSI-LD)
    Normalized,
    /// Key-values (simplified)
    KeyValues,
}

/// Notification endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NotificationEndpoint {
    /// Endpoint URI
    pub uri: String,

    /// Accept header
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accept: Option<String>,

    /// Receiver info (headers, auth)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receiver_info: Option<Vec<KeyValuePair>>,

    /// Notifier info
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notifier_info: Option<Vec<KeyValuePair>>,
}

/// Key-value pair for receiver/notifier info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyValuePair {
    pub key: String,
    pub value: String,
}

// ============================================================================
// Query Types
// ============================================================================

/// Geo-query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeoQuery {
    /// Geometry type
    pub geometry: String,

    /// Coordinates
    pub coordinates: serde_json::Value,

    /// Geo-relationship
    pub georel: String,

    /// Geo-property to query
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geoproperty: Option<String>,
}

/// Temporal query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalQuery {
    /// Time relation
    pub timerel: TimeRelation,

    /// Time property (observedAt, createdAt, modifiedAt)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeproperty: Option<String>,

    /// Time at
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_at: Option<DateTime<Utc>>,

    /// End time at
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time_at: Option<DateTime<Utc>>,
}

/// Time relation for temporal queries
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TimeRelation {
    /// Before time
    Before,
    /// After time
    After,
    /// Between times
    Between,
}

// ============================================================================
// API Query Parameters
// ============================================================================

/// Query parameters for entity operations
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct NgsiQueryParams {
    /// Entity ID filter (comma-separated)
    pub id: Option<String>,

    /// Entity type filter
    #[serde(rename = "type")]
    pub entity_type: Option<String>,

    /// ID pattern (regex)
    pub id_pattern: Option<String>,

    /// Attributes to return (comma-separated)
    pub attrs: Option<String>,

    /// NGSI-LD query expression
    pub q: Option<String>,

    /// Geo-relationship
    pub georel: Option<String>,

    /// Geometry type
    pub geometry: Option<String>,

    /// Coordinates
    pub coordinates: Option<String>,

    /// Geo-property
    pub geoproperty: Option<String>,

    /// CSF query
    pub csf: Option<String>,

    /// Limit
    pub limit: Option<u32>,

    /// Offset
    pub offset: Option<u32>,

    /// Options (keyValues, sysAttrs, count)
    pub options: Option<String>,

    /// Pick specific attribute instances
    pub pick: Option<String>,

    /// Omit specific attribute instances
    pub omit: Option<String>,

    /// Language filter
    pub lang: Option<String>,

    /// Scope query
    #[serde(rename = "scopeQ")]
    pub scope_q: Option<String>,

    /// Local flag
    pub local: Option<bool>,

    /// Data type (entity or entityType)
    #[serde(rename = "datasetId")]
    pub dataset_id: Option<String>,

    /// Delete all flag
    #[serde(rename = "deleteAll")]
    pub delete_all: Option<bool>,
}

impl NgsiQueryParams {
    /// Check if keyValues option is set
    pub fn is_key_values(&self) -> bool {
        self.options
            .as_ref()
            .map(|o| o.contains("keyValues"))
            .unwrap_or(false)
    }

    /// Check if sysAttrs option is set
    pub fn is_sys_attrs(&self) -> bool {
        self.options
            .as_ref()
            .map(|o| o.contains("sysAttrs"))
            .unwrap_or(false)
    }

    /// Check if count option is set
    pub fn is_count(&self) -> bool {
        self.options
            .as_ref()
            .map(|o| o.contains("count"))
            .unwrap_or(false)
    }

    /// Get entity IDs as vector
    pub fn entity_ids(&self) -> Vec<&str> {
        self.id
            .as_ref()
            .map(|s| s.split(',').map(|s| s.trim()).collect())
            .unwrap_or_default()
    }

    /// Get attribute names as vector
    pub fn attribute_names(&self) -> Vec<&str> {
        self.attrs
            .as_ref()
            .map(|s| s.split(',').map(|s| s.trim()).collect())
            .unwrap_or_default()
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// NGSI-LD API error
#[derive(Debug, thiserror::Error)]
pub enum NgsiError {
    #[error("Resource not found: {0}")]
    NotFound(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Already exists: {0}")]
    AlreadyExists(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Unsupported media type: {0}")]
    UnsupportedMediaType(String),

    #[error("Not acceptable: {0}")]
    NotAcceptable(String),

    #[error("Operation not supported: {0}")]
    OperationNotSupported(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("LdContext not available: {0}")]
    LdContextNotAvailable(String),

    #[error("No multi-tenant header: {0}")]
    NoMultiTenantHeader(String),
}

impl NgsiError {
    /// Get HTTP status code for this error
    pub fn status_code(&self) -> u16 {
        match self {
            NgsiError::NotFound(_) => 404,
            NgsiError::BadRequest(_) => 400,
            NgsiError::AlreadyExists(_) => 409,
            NgsiError::InvalidRequest(_) => 400,
            NgsiError::UnsupportedMediaType(_) => 415,
            NgsiError::NotAcceptable(_) => 406,
            NgsiError::OperationNotSupported(_) => 422,
            NgsiError::InternalError(_) => 500,
            NgsiError::LdContextNotAvailable(_) => 503,
            NgsiError::NoMultiTenantHeader(_) => 400,
        }
    }

    /// Get NGSI-LD error type URI
    pub fn error_type(&self) -> &str {
        match self {
            NgsiError::NotFound(_) => "https://uri.etsi.org/ngsi-ld/errors/ResourceNotFound",
            NgsiError::BadRequest(_) => "https://uri.etsi.org/ngsi-ld/errors/BadRequestData",
            NgsiError::AlreadyExists(_) => "https://uri.etsi.org/ngsi-ld/errors/AlreadyExists",
            NgsiError::InvalidRequest(_) => "https://uri.etsi.org/ngsi-ld/errors/InvalidRequest",
            NgsiError::UnsupportedMediaType(_) => {
                "https://uri.etsi.org/ngsi-ld/errors/UnsupportedMediaType"
            }
            NgsiError::NotAcceptable(_) => "https://uri.etsi.org/ngsi-ld/errors/NotAcceptable",
            NgsiError::OperationNotSupported(_) => {
                "https://uri.etsi.org/ngsi-ld/errors/OperationNotSupported"
            }
            NgsiError::InternalError(_) => "https://uri.etsi.org/ngsi-ld/errors/InternalError",
            NgsiError::LdContextNotAvailable(_) => {
                "https://uri.etsi.org/ngsi-ld/errors/LdContextNotAvailable"
            }
            NgsiError::NoMultiTenantHeader(_) => {
                "https://uri.etsi.org/ngsi-ld/errors/NonexistentTenant"
            }
        }
    }
}

impl axum::response::IntoResponse for NgsiError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;
        use axum::response::Json;

        let status =
            StatusCode::from_u16(self.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        let body = serde_json::json!({
            "type": self.error_type(),
            "title": format!("{}", self),
            "status": status.as_u16(),
        });

        (status, [("Content-Type", "application/json")], Json(body)).into_response()
    }
}

// ============================================================================
// Batch Operation Types
// ============================================================================

/// Batch operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchOperationResult {
    /// Successfully processed entity IDs
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub success: Vec<String>,

    /// Failed operations
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub errors: Vec<BatchError>,
}

/// Batch operation error
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchError {
    /// Entity ID that failed
    pub entity_id: String,

    /// Error details
    pub error: ProblemDetails,
}

/// RFC 7807 Problem Details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemDetails {
    /// Error type URI
    #[serde(rename = "type")]
    pub error_type: String,

    /// Human-readable title
    pub title: String,

    /// HTTP status code
    pub status: u16,

    /// Detailed description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,

    /// Instance URI
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngsi_entity_creation() {
        let entity = NgsiEntity::new("urn:ngsi-ld:Vehicle:A123", "Vehicle").with_property(
            "speed",
            NgsiProperty::new(serde_json::json!(80.5)).with_unit("KMH"),
        );

        assert_eq!(entity.id, "urn:ngsi-ld:Vehicle:A123");
        assert_eq!(entity.entity_type.primary(), "Vehicle");
        assert_eq!(entity.short_id(), "A123");
        assert!(entity.properties.contains_key("speed"));
    }

    #[test]
    fn test_ngsi_property_serialization() {
        let prop = NgsiProperty::new(serde_json::json!(25.5)).with_unit("CEL");

        let json = serde_json::to_string(&prop).unwrap();
        assert!(json.contains("\"type\":\"Property\""));
        assert!(json.contains("\"value\":25.5"));
        assert!(json.contains("\"unitCode\":\"CEL\""));
    }

    #[test]
    fn test_ngsi_relationship_serialization() {
        let rel = NgsiRelationship::new("urn:ngsi-ld:Building:B001");

        let json = serde_json::to_string(&rel).unwrap();
        assert!(json.contains("\"type\":\"Relationship\""));
        assert!(json.contains("\"object\":\"urn:ngsi-ld:Building:B001\""));
    }

    #[test]
    fn test_geo_property_point() {
        let geo = GeoProperty::point(139.7673068, 35.6809591);

        let json = serde_json::to_string(&geo).unwrap();
        assert!(json.contains("\"type\":\"GeoProperty\""));
        assert!(json.contains("\"Point\""));
    }

    #[test]
    fn test_ngsi_query_params() {
        let params = NgsiQueryParams {
            id: Some("urn:ngsi-ld:Vehicle:A1,urn:ngsi-ld:Vehicle:A2".to_string()),
            entity_type: Some("Vehicle".to_string()),
            attrs: Some("speed,temperature".to_string()),
            options: Some("keyValues,count".to_string()),
            ..Default::default()
        };

        assert_eq!(
            params.entity_ids(),
            vec!["urn:ngsi-ld:Vehicle:A1", "urn:ngsi-ld:Vehicle:A2"]
        );
        assert_eq!(params.attribute_names(), vec!["speed", "temperature"]);
        assert!(params.is_key_values());
        assert!(params.is_count());
        assert!(!params.is_sys_attrs());
    }

    #[test]
    fn test_ngsi_error_response() {
        let error = NgsiError::NotFound("Entity not found".to_string());
        assert_eq!(error.status_code(), 404);
        assert_eq!(
            error.error_type(),
            "https://uri.etsi.org/ngsi-ld/errors/ResourceNotFound"
        );
    }

    #[test]
    fn test_ngsi_context_default() {
        let ctx = NgsiContext::default();
        if let NgsiContext::Uri(uri) = ctx {
            assert!(uri.contains("ngsi-ld-core-context"));
        } else {
            panic!("Expected URI context");
        }
    }
}
