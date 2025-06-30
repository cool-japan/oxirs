//! NATS Configuration Types
//!
//! This module contains all configuration-related types for the NATS backend.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// NATS-specific configuration with advanced JetStream features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatsConfig {
    pub url: String,
    pub cluster_urls: Option<Vec<String>>,
    pub stream_name: String,
    pub subject_prefix: String,
    pub max_age_seconds: u64,
    pub max_bytes: u64,
    pub replicas: usize,
    pub storage_type: NatsStorageType,
    pub retention_policy: NatsRetentionPolicy,
    pub max_msgs: i64,
    pub max_msg_size: i32,
    pub discard_policy: NatsDiscardPolicy,
    pub duplicate_window: Duration,
    pub consumer_config: NatsConsumerConfig,
    pub auth_config: Option<NatsAuthConfig>,
    pub tls_config: Option<NatsTlsConfig>,
    pub subject_router: Option<SubjectRouter>,
    pub queue_groups: Vec<QueueGroupConfig>,
    pub request_reply_config: Option<RequestReplyConfig>,
    pub enable_clustering: bool,
    pub cluster_name: Option<String>,
}

/// NATS storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NatsStorageType {
    File,
    Memory,
}

/// NATS retention policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NatsRetentionPolicy {
    Limits,
    Interest,
    WorkQueue,
}

/// NATS discard policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NatsDiscardPolicy {
    Old,
    New,
}

/// NATS consumer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatsConsumerConfig {
    pub name: String,
    pub description: String,
    pub deliver_policy: NatsDeliverPolicy,
    pub ack_policy: NatsAckPolicy,
    pub ack_wait: Duration,
    pub max_deliver: i64,
    pub replay_policy: NatsReplayPolicy,
    pub max_ack_pending: i64,
    pub max_waiting: i64,
    pub max_batch: i64,
    pub max_expires: Duration,
    pub flow_control: bool,
    pub heartbeat: Duration,
    pub queue_group: Option<String>,
    pub filter_subjects: Vec<String>,
    pub rate_limit: Option<u64>,
    pub headers_only: bool,
}

/// NATS deliver policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NatsDeliverPolicy {
    All,
    Last,
    New,
    ByStartSequence(u64),
    ByStartTime(DateTime<Utc>),
    LastPerSubject,
}

/// NATS acknowledgment policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NatsAckPolicy {
    None,
    All,
    Explicit,
}

/// NATS replay policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NatsReplayPolicy {
    Instant,
    Original,
}

/// NATS authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatsAuthConfig {
    pub token: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub nkey: Option<String>,
    pub jwt: Option<String>,
    pub creds_file: Option<String>,
}

/// NATS TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatsTlsConfig {
    pub cert_file: Option<String>,
    pub key_file: Option<String>,
    pub ca_file: Option<String>,
    pub verify_cert: bool,
    pub verify_hostname: bool,
}

/// Subject routing configuration for advanced message routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectRouter {
    pub routes: Vec<SubjectRoute>,
    pub wildcard_patterns: Vec<WildcardPattern>,
    pub default_handler: Option<String>,
}

/// Individual subject route
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectRoute {
    pub pattern: String,
    pub handler: String,
    pub priority: u32,
    pub filters: Vec<MessageFilter>,
}

/// Wildcard pattern configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WildcardPattern {
    pub pattern: String,
    pub description: String,
    pub enabled: bool,
}

/// Message filter for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageFilter {
    pub field: String,
    pub operator: FilterOperator,
    pub value: String,
}

/// Filter operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Equals,
    Contains,
    StartsWith,
    EndsWith,
    Regex,
    GreaterThan,
    LessThan,
}

/// Queue group configuration for load balancing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueGroupConfig {
    pub name: String,
    pub subjects: Vec<String>,
    pub max_members: Option<usize>,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub health_check_interval: Duration,
}

/// Load balancing strategies for queue groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    Random,
    WeightedRoundRobin(Vec<u32>),
    Consistent,
}

/// Request-reply pattern configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestReplyConfig {
    pub timeout: Duration,
    pub retries: u32,
    pub retry_delay: Duration,
    pub circuit_breaker: Option<CircuitBreakerConfig>,
}

/// Circuit breaker configuration for request-reply
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub recovery_timeout: Duration,
    pub half_open_max_calls: u32,
}

// Default implementations
impl Default for NatsConfig {
    fn default() -> Self {
        Self {
            url: "nats://localhost:4222".to_string(),
            cluster_urls: None,
            stream_name: "RDF_STREAM".to_string(),
            subject_prefix: "rdf".to_string(),
            max_age_seconds: 86400, // 24 hours
            max_bytes: 1024 * 1024 * 1024, // 1GB
            replicas: 1,
            storage_type: NatsStorageType::File,
            retention_policy: NatsRetentionPolicy::Limits,
            max_msgs: 1_000_000,
            max_msg_size: 1024 * 1024, // 1MB
            discard_policy: NatsDiscardPolicy::Old,
            duplicate_window: Duration::from_secs(120),
            consumer_config: NatsConsumerConfig::default(),
            auth_config: None,
            tls_config: None,
            subject_router: None,
            queue_groups: Vec::new(),
            request_reply_config: None,
            enable_clustering: false,
            cluster_name: None,
        }
    }
}

impl Default for NatsConsumerConfig {
    fn default() -> Self {
        Self {
            name: "rdf_consumer".to_string(),
            description: "RDF Stream Consumer".to_string(),
            deliver_policy: NatsDeliverPolicy::All,
            ack_policy: NatsAckPolicy::Explicit,
            ack_wait: Duration::from_secs(30),
            max_deliver: 3,
            replay_policy: NatsReplayPolicy::Instant,
            max_ack_pending: 1000,
            max_waiting: 512,
            max_batch: 100,
            max_expires: Duration::from_secs(300),
            flow_control: true,
            heartbeat: Duration::from_secs(5),
            queue_group: None,
            filter_subjects: Vec::new(),
            rate_limit: None,
            headers_only: false,
        }
    }
}