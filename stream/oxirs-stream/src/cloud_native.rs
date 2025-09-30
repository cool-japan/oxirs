//! # Cloud-Native Integration Module
//!
//! Comprehensive Kubernetes and service mesh integration for OxiRS Stream,
//! providing enterprise-grade cloud-native deployment, scaling, and management capabilities.
//!
//! This module provides:
//! - Kubernetes Custom Resource Definitions (CRDs) and Operators
//! - Service mesh integration (Istio, Linkerd, Consul Connect)
//! - Auto-scaling with custom metrics
//! - Health checks and observability
//! - Multi-cloud deployment strategies
//! - GitOps integration and CI/CD pipelines

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Cloud-native configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudNativeConfig {
    /// Kubernetes configuration
    pub kubernetes: KubernetesConfig,
    /// Service mesh configuration
    pub service_mesh: ServiceMeshConfig,
    /// Auto-scaling configuration
    pub auto_scaling: AutoScalingConfig,
    /// Observability configuration
    pub observability: ObservabilityConfig,
    /// Multi-cloud configuration
    pub multi_cloud: MultiCloudConfig,
    /// GitOps configuration
    pub gitops: GitOpsConfig,
}

impl Default for CloudNativeConfig {
    fn default() -> Self {
        Self {
            kubernetes: KubernetesConfig::default(),
            service_mesh: ServiceMeshConfig::default(),
            auto_scaling: AutoScalingConfig::default(),
            observability: ObservabilityConfig::default(),
            multi_cloud: MultiCloudConfig::default(),
            gitops: GitOpsConfig::default(),
        }
    }
}

/// Kubernetes configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesConfig {
    /// Enable Kubernetes integration
    pub enabled: bool,
    /// Namespace for OxiRS resources
    pub namespace: String,
    /// Custom Resource Definitions
    pub crds: Vec<CustomResourceDefinition>,
    /// Operator configuration
    pub operator: OperatorConfig,
    /// Health check configuration
    pub health_checks: HealthCheckConfig,
    /// Network policies
    pub network_policies: NetworkPolicyConfig,
    /// Resource quotas
    pub resource_quotas: ResourceQuotaConfig,
}

impl Default for KubernetesConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            namespace: "oxirs".to_string(),
            crds: vec![
                CustomResourceDefinition::stream_processor(),
                CustomResourceDefinition::stream_cluster(),
                CustomResourceDefinition::stream_policy(),
            ],
            operator: OperatorConfig::default(),
            health_checks: HealthCheckConfig::default(),
            network_policies: NetworkPolicyConfig::default(),
            resource_quotas: ResourceQuotaConfig::default(),
        }
    }
}

/// Custom Resource Definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomResourceDefinition {
    pub api_version: String,
    pub kind: String,
    pub metadata: ObjectMetadata,
    pub spec: CRDSpec,
}

impl CustomResourceDefinition {
    /// Create StreamProcessor CRD
    pub fn stream_processor() -> Self {
        Self {
            api_version: "apiextensions.k8s.io/v1".to_string(),
            kind: "CustomResourceDefinition".to_string(),
            metadata: ObjectMetadata {
                name: "streamprocessors.oxirs.io".to_string(),
                namespace: None,
                labels: BTreeMap::from([
                    ("app".to_string(), "oxirs".to_string()),
                    ("component".to_string(), "stream-processor".to_string()),
                ]),
                annotations: BTreeMap::new(),
            },
            spec: CRDSpec {
                group: "oxirs.io".to_string(),
                versions: vec![CRDVersion {
                    name: "v1".to_string(),
                    served: true,
                    storage: true,
                    schema: ResourceSchema::stream_processor_schema(),
                }],
                scope: "Namespaced".to_string(),
                names: CRDNames {
                    plural: "streamprocessors".to_string(),
                    singular: "streamprocessor".to_string(),
                    kind: "StreamProcessor".to_string(),
                    short_names: vec!["sp".to_string()],
                },
            },
        }
    }

    /// Create StreamCluster CRD
    pub fn stream_cluster() -> Self {
        Self {
            api_version: "apiextensions.k8s.io/v1".to_string(),
            kind: "CustomResourceDefinition".to_string(),
            metadata: ObjectMetadata {
                name: "streamclusters.oxirs.io".to_string(),
                namespace: None,
                labels: BTreeMap::from([
                    ("app".to_string(), "oxirs".to_string()),
                    ("component".to_string(), "stream-cluster".to_string()),
                ]),
                annotations: BTreeMap::new(),
            },
            spec: CRDSpec {
                group: "oxirs.io".to_string(),
                versions: vec![CRDVersion {
                    name: "v1".to_string(),
                    served: true,
                    storage: true,
                    schema: ResourceSchema::stream_cluster_schema(),
                }],
                scope: "Namespaced".to_string(),
                names: CRDNames {
                    plural: "streamclusters".to_string(),
                    singular: "streamcluster".to_string(),
                    kind: "StreamCluster".to_string(),
                    short_names: vec!["sc".to_string()],
                },
            },
        }
    }

    /// Create StreamPolicy CRD
    pub fn stream_policy() -> Self {
        Self {
            api_version: "apiextensions.k8s.io/v1".to_string(),
            kind: "CustomResourceDefinition".to_string(),
            metadata: ObjectMetadata {
                name: "streampolicies.oxirs.io".to_string(),
                namespace: None,
                labels: BTreeMap::from([
                    ("app".to_string(), "oxirs".to_string()),
                    ("component".to_string(), "stream-policy".to_string()),
                ]),
                annotations: BTreeMap::new(),
            },
            spec: CRDSpec {
                group: "oxirs.io".to_string(),
                versions: vec![CRDVersion {
                    name: "v1".to_string(),
                    served: true,
                    storage: true,
                    schema: ResourceSchema::stream_policy_schema(),
                }],
                scope: "Namespaced".to_string(),
                names: CRDNames {
                    plural: "streampolicies".to_string(),
                    singular: "streampolicy".to_string(),
                    kind: "StreamPolicy".to_string(),
                    short_names: vec!["spol".to_string()],
                },
            },
        }
    }
}

/// Object metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMetadata {
    pub name: String,
    pub namespace: Option<String>,
    pub labels: BTreeMap<String, String>,
    pub annotations: BTreeMap<String, String>,
}

/// CRD specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CRDSpec {
    pub group: String,
    pub versions: Vec<CRDVersion>,
    pub scope: String,
    pub names: CRDNames,
}

/// CRD version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CRDVersion {
    pub name: String,
    pub served: bool,
    pub storage: bool,
    pub schema: ResourceSchema,
}

/// CRD names
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CRDNames {
    pub plural: String,
    pub singular: String,
    pub kind: String,
    pub short_names: Vec<String>,
}

/// Resource schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSchema {
    pub open_api_v3_schema: OpenAPIV3Schema,
}

impl ResourceSchema {
    /// StreamProcessor schema
    pub fn stream_processor_schema() -> Self {
        Self {
            open_api_v3_schema: OpenAPIV3Schema {
                schema_type: "object".to_string(),
                properties: BTreeMap::from([
                    ("spec".to_string(), SchemaProperty::stream_processor_spec()),
                    ("status".to_string(), SchemaProperty::stream_processor_status()),
                ]),
                required: vec!["spec".to_string()],
            },
        }
    }

    /// StreamCluster schema
    pub fn stream_cluster_schema() -> Self {
        Self {
            open_api_v3_schema: OpenAPIV3Schema {
                schema_type: "object".to_string(),
                properties: BTreeMap::from([
                    ("spec".to_string(), SchemaProperty::stream_cluster_spec()),
                    ("status".to_string(), SchemaProperty::stream_cluster_status()),
                ]),
                required: vec!["spec".to_string()],
            },
        }
    }

    /// StreamPolicy schema
    pub fn stream_policy_schema() -> Self {
        Self {
            open_api_v3_schema: OpenAPIV3Schema {
                schema_type: "object".to_string(),
                properties: BTreeMap::from([
                    ("spec".to_string(), SchemaProperty::stream_policy_spec()),
                    ("status".to_string(), SchemaProperty::stream_policy_status()),
                ]),
                required: vec!["spec".to_string()],
            },
        }
    }
}

/// OpenAPI v3 schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAPIV3Schema {
    #[serde(rename = "type")]
    pub schema_type: String,
    pub properties: BTreeMap<String, SchemaProperty>,
    pub required: Vec<String>,
}

/// Schema property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaProperty {
    #[serde(rename = "type")]
    pub property_type: String,
    pub description: Option<String>,
    pub properties: Option<BTreeMap<String, SchemaProperty>>,
    pub items: Option<Box<SchemaProperty>>,
    pub required: Option<Vec<String>>,
}

impl SchemaProperty {
    /// StreamProcessor spec schema
    pub fn stream_processor_spec() -> Self {
        Self {
            property_type: "object".to_string(),
            description: Some("StreamProcessor specification".to_string()),
            properties: Some(BTreeMap::from([
                ("replicas".to_string(), SchemaProperty {
                    property_type: "integer".to_string(),
                    description: Some("Number of replicas".to_string()),
                    properties: None,
                    items: None,
                    required: None,
                }),
                ("image".to_string(), SchemaProperty {
                    property_type: "string".to_string(),
                    description: Some("Container image".to_string()),
                    properties: None,
                    items: None,
                    required: None,
                }),
                ("config".to_string(), SchemaProperty {
                    property_type: "object".to_string(),
                    description: Some("Configuration".to_string()),
                    properties: None,
                    items: None,
                    required: None,
                }),
            ])),
            items: None,
            required: Some(vec!["replicas".to_string(), "image".to_string()]),
        }
    }

    /// StreamProcessor status schema
    pub fn stream_processor_status() -> Self {
        Self {
            property_type: "object".to_string(),
            description: Some("StreamProcessor status".to_string()),
            properties: Some(BTreeMap::from([
                ("phase".to_string(), SchemaProperty {
                    property_type: "string".to_string(),
                    description: Some("Current phase".to_string()),
                    properties: None,
                    items: None,
                    required: None,
                }),
                ("ready_replicas".to_string(), SchemaProperty {
                    property_type: "integer".to_string(),
                    description: Some("Number of ready replicas".to_string()),
                    properties: None,
                    items: None,
                    required: None,
                }),
            ])),
            items: None,
            required: None,
        }
    }

    /// StreamCluster spec schema
    pub fn stream_cluster_spec() -> Self {
        Self {
            property_type: "object".to_string(),
            description: Some("StreamCluster specification".to_string()),
            properties: Some(BTreeMap::from([
                ("size".to_string(), SchemaProperty {
                    property_type: "integer".to_string(),
                    description: Some("Cluster size".to_string()),
                    properties: None,
                    items: None,
                    required: None,
                }),
                ("storage".to_string(), SchemaProperty {
                    property_type: "object".to_string(),
                    description: Some("Storage configuration".to_string()),
                    properties: None,
                    items: None,
                    required: None,
                }),
            ])),
            items: None,
            required: Some(vec!["size".to_string()]),
        }
    }

    /// StreamCluster status schema
    pub fn stream_cluster_status() -> Self {
        Self {
            property_type: "object".to_string(),
            description: Some("StreamCluster status".to_string()),
            properties: Some(BTreeMap::from([
                ("phase".to_string(), SchemaProperty {
                    property_type: "string".to_string(),
                    description: Some("Current phase".to_string()),
                    properties: None,
                    items: None,
                    required: None,
                }),
                ("nodes".to_string(), SchemaProperty {
                    property_type: "array".to_string(),
                    description: Some("Cluster nodes".to_string()),
                    properties: None,
                    items: Some(Box::new(SchemaProperty {
                        property_type: "object".to_string(),
                        description: None,
                        properties: None,
                        items: None,
                        required: None,
                    })),
                    required: None,
                }),
            ])),
            items: None,
            required: None,
        }
    }

    /// StreamPolicy spec schema
    pub fn stream_policy_spec() -> Self {
        Self {
            property_type: "object".to_string(),
            description: Some("StreamPolicy specification".to_string()),
            properties: Some(BTreeMap::from([
                ("rules".to_string(), SchemaProperty {
                    property_type: "array".to_string(),
                    description: Some("Policy rules".to_string()),
                    properties: None,
                    items: Some(Box::new(SchemaProperty {
                        property_type: "object".to_string(),
                        description: None,
                        properties: None,
                        items: None,
                        required: None,
                    })),
                    required: None,
                }),
            ])),
            items: None,
            required: Some(vec!["rules".to_string()]),
        }
    }

    /// StreamPolicy status schema
    pub fn stream_policy_status() -> Self {
        Self {
            property_type: "object".to_string(),
            description: Some("StreamPolicy status".to_string()),
            properties: Some(BTreeMap::from([
                ("applied".to_string(), SchemaProperty {
                    property_type: "boolean".to_string(),
                    description: Some("Policy applied status".to_string()),
                    properties: None,
                    items: None,
                    required: None,
                }),
            ])),
            items: None,
            required: None,
        }
    }
}

/// Operator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Enable operator
    pub enabled: bool,
    /// Operator image
    pub image: String,
    /// Watch all namespaces
    pub cluster_scoped: bool,
    /// Reconciliation interval
    pub reconcile_interval: Duration,
    /// Leader election configuration
    pub leader_election: LeaderElectionConfig,
}

impl Default for OperatorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            image: "oxirs/operator:latest".to_string(),
            cluster_scoped: false,
            reconcile_interval: Duration::from_secs(30),
            leader_election: LeaderElectionConfig::default(),
        }
    }
}

/// Leader election configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderElectionConfig {
    pub enabled: bool,
    pub lock_name: String,
    pub lease_duration: Duration,
    pub renew_deadline: Duration,
    pub retry_period: Duration,
}

impl Default for LeaderElectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            lock_name: "oxirs-operator-lock".to_string(),
            lease_duration: Duration::from_secs(15),
            renew_deadline: Duration::from_secs(10),
            retry_period: Duration::from_secs(2),
        }
    }
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Enable health checks
    pub enabled: bool,
    /// Liveness probe configuration
    pub liveness_probe: ProbeConfig,
    /// Readiness probe configuration
    pub readiness_probe: ProbeConfig,
    /// Startup probe configuration
    pub startup_probe: ProbeConfig,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            liveness_probe: ProbeConfig {
                path: "/health/live".to_string(),
                port: 8080,
                initial_delay_seconds: 30,
                period_seconds: 10,
                timeout_seconds: 5,
                failure_threshold: 3,
                success_threshold: 1,
            },
            readiness_probe: ProbeConfig {
                path: "/health/ready".to_string(),
                port: 8080,
                initial_delay_seconds: 10,
                period_seconds: 5,
                timeout_seconds: 3,
                failure_threshold: 3,
                success_threshold: 1,
            },
            startup_probe: ProbeConfig {
                path: "/health/startup".to_string(),
                port: 8080,
                initial_delay_seconds: 0,
                period_seconds: 5,
                timeout_seconds: 3,
                failure_threshold: 30,
                success_threshold: 1,
            },
        }
    }
}

/// Probe configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeConfig {
    pub path: String,
    pub port: u16,
    pub initial_delay_seconds: u32,
    pub period_seconds: u32,
    pub timeout_seconds: u32,
    pub failure_threshold: u32,
    pub success_threshold: u32,
}

/// Network policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicyConfig {
    /// Enable network policies
    pub enabled: bool,
    /// Default deny all traffic
    pub default_deny: bool,
    /// Allowed ingress rules
    pub ingress_rules: Vec<NetworkPolicyRule>,
    /// Allowed egress rules
    pub egress_rules: Vec<NetworkPolicyRule>,
}

impl Default for NetworkPolicyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_deny: true,
            ingress_rules: vec![
                NetworkPolicyRule {
                    description: "Allow ingress from service mesh".to_string(),
                    from_selector: Some(LabelSelector {
                        match_labels: BTreeMap::from([
                            ("app".to_string(), "istio-proxy".to_string()),
                        ]),
                    }),
                    ports: vec![NetworkPolicyPort {
                        port: 8080,
                        protocol: "TCP".to_string(),
                    }],
                },
            ],
            egress_rules: vec![
                NetworkPolicyRule {
                    description: "Allow egress to Kubernetes API".to_string(),
                    from_selector: None,
                    ports: vec![NetworkPolicyPort {
                        port: 443,
                        protocol: "TCP".to_string(),
                    }],
                },
            ],
        }
    }
}

/// Network policy rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicyRule {
    pub description: String,
    pub from_selector: Option<LabelSelector>,
    pub ports: Vec<NetworkPolicyPort>,
}

/// Label selector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelSelector {
    pub match_labels: BTreeMap<String, String>,
}

/// Network policy port
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicyPort {
    pub port: u16,
    pub protocol: String,
}

/// Resource quota configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuotaConfig {
    /// Enable resource quotas
    pub enabled: bool,
    /// CPU limit
    pub cpu_limit: String,
    /// Memory limit
    pub memory_limit: String,
    /// Storage limit
    pub storage_limit: String,
    /// Pod limit
    pub pod_limit: u32,
    /// Service limit
    pub service_limit: u32,
}

impl Default for ResourceQuotaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cpu_limit: "10".to_string(),
            memory_limit: "20Gi".to_string(),
            storage_limit: "100Gi".to_string(),
            pod_limit: 50,
            service_limit: 10,
        }
    }
}

/// Service mesh configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMeshConfig {
    /// Service mesh provider
    pub provider: ServiceMeshProvider,
    /// Enable service mesh
    pub enabled: bool,
    /// mTLS configuration
    pub mtls: MutualTLSConfig,
    /// Traffic management
    pub traffic_management: TrafficManagementConfig,
    /// Observability configuration
    pub observability: ServiceMeshObservabilityConfig,
    /// Security policies
    pub security_policies: SecurityPolicyConfig,
}

impl Default for ServiceMeshConfig {
    fn default() -> Self {
        Self {
            provider: ServiceMeshProvider::Istio,
            enabled: true,
            mtls: MutualTLSConfig::default(),
            traffic_management: TrafficManagementConfig::default(),
            observability: ServiceMeshObservabilityConfig::default(),
            security_policies: SecurityPolicyConfig::default(),
        }
    }
}

/// Service mesh providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceMeshProvider {
    Istio,
    Linkerd,
    ConsulConnect,
    OpenServiceMesh,
    Kuma,
}

/// Mutual TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutualTLSConfig {
    /// Enable mTLS
    pub enabled: bool,
    /// mTLS mode
    pub mode: MutualTLSMode,
    /// Certificate authority
    pub ca_provider: CertificateAuthorityProvider,
    /// Certificate rotation interval
    pub cert_rotation_interval: ChronoDuration,
    /// Key size
    pub key_size: u32,
}

impl Default for MutualTLSConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: MutualTLSMode::Strict,
            ca_provider: CertificateAuthorityProvider::Istio,
            cert_rotation_interval: ChronoDuration::days(30),
            key_size: 2048,
        }
    }
}

/// Mutual TLS modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutualTLSMode {
    Disabled,
    Permissive,
    Strict,
}

/// Certificate authority providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CertificateAuthorityProvider {
    Istio,
    CertManager,
    Vault,
    External,
}

/// Traffic management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficManagementConfig {
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Retry configuration
    pub retry: RetryConfig,
    /// Timeout configuration
    pub timeout: TimeoutConfig,
    /// Rate limiting
    pub rate_limiting: ServiceMeshRateLimitConfig,
}

impl Default for TrafficManagementConfig {
    fn default() -> Self {
        Self {
            load_balancing: LoadBalancingStrategy::RoundRobin,
            circuit_breaker: CircuitBreakerConfig::default(),
            retry: RetryConfig::default(),
            timeout: TimeoutConfig::default(),
            rate_limiting: ServiceMeshRateLimitConfig::default(),
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnection,
    Random,
    WeightedRoundRobin,
    ConsistentHash,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Enable circuit breaker
    pub enabled: bool,
    /// Consecutive errors threshold
    pub consecutive_errors: u32,
    /// Error threshold percentage
    pub error_threshold_percentage: f64,
    /// Minimum request threshold
    pub min_request_amount: u32,
    /// Sleep window
    pub sleep_window: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            consecutive_errors: 5,
            error_threshold_percentage: 50.0,
            min_request_amount: 20,
            sleep_window: Duration::from_secs(30),
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Number of retry attempts
    pub attempts: u32,
    /// Per-try timeout
    pub per_try_timeout: Duration,
    /// Retry conditions
    pub retry_on: Vec<RetryCondition>,
    /// Backoff strategy
    pub backoff: BackoffStrategy,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            attempts: 3,
            per_try_timeout: Duration::from_secs(5),
            retry_on: vec![
                RetryCondition::FiveXX,
                RetryCondition::GatewayError,
                RetryCondition::ConnectFailure,
            ],
            backoff: BackoffStrategy::Exponential {
                base_interval: Duration::from_millis(25),
                max_interval: Duration::from_secs(30),
            },
        }
    }
}

/// Retry conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryCondition {
    FiveXX,
    GatewayError,
    ConnectFailure,
    RefusedStream,
    Cancelled,
    DeadlineExceeded,
    ResourceExhausted,
}

/// Backoff strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    Fixed {
        interval: Duration,
    },
    Exponential {
        base_interval: Duration,
        max_interval: Duration,
    },
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Request timeout
    pub request_timeout: Duration,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Stream idle timeout
    pub stream_idle_timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            connection_timeout: Duration::from_secs(10),
            stream_idle_timeout: Duration::from_secs(300),
        }
    }
}

/// Service mesh rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMeshRateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Rate limit per second
    pub requests_per_second: u32,
    /// Burst size
    pub burst_size: u32,
    /// Rate limit headers
    pub fill_interval: Duration,
}

impl Default for ServiceMeshRateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_second: 100,
            burst_size: 200,
            fill_interval: Duration::from_secs(1),
        }
    }
}

/// Service mesh observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMeshObservabilityConfig {
    /// Enable distributed tracing
    pub tracing: TracingConfig,
    /// Enable metrics collection
    pub metrics: MetricsConfig,
    /// Enable access logging
    pub access_logs: AccessLogsConfig,
}

impl Default for ServiceMeshObservabilityConfig {
    fn default() -> Self {
        Self {
            tracing: TracingConfig::default(),
            metrics: MetricsConfig::default(),
            access_logs: AccessLogsConfig::default(),
        }
    }
}

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Enable tracing
    pub enabled: bool,
    /// Tracing provider
    pub provider: TracingProvider,
    /// Sampling rate
    pub sampling_rate: f64,
    /// Trace headers
    pub custom_headers: Vec<String>,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: TracingProvider::Jaeger,
            sampling_rate: 1.0,
            custom_headers: vec![
                "x-request-id".to_string(),
                "x-correlation-id".to_string(),
            ],
        }
    }
}

/// Tracing providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TracingProvider {
    Jaeger,
    Zipkin,
    OpenTelemetry,
    Datadog,
    LightStep,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics
    pub enabled: bool,
    /// Metrics provider
    pub provider: MetricsProvider,
    /// Custom metrics
    pub custom_metrics: Vec<CustomMetric>,
    /// Metrics scraping interval
    pub scrape_interval: Duration,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: MetricsProvider::Prometheus,
            custom_metrics: vec![
                CustomMetric {
                    name: "oxirs_stream_events_total".to_string(),
                    metric_type: MetricType::Counter,
                    description: "Total number of stream events processed".to_string(),
                    labels: vec!["source".to_string(), "type".to_string()],
                },
                CustomMetric {
                    name: "oxirs_stream_latency_seconds".to_string(),
                    metric_type: MetricType::Histogram,
                    description: "Stream processing latency in seconds".to_string(),
                    labels: vec!["operation".to_string()],
                },
            ],
            scrape_interval: Duration::from_secs(30),
        }
    }
}

/// Metrics providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricsProvider {
    Prometheus,
    Datadog,
    NewRelic,
    CloudWatch,
}

/// Custom metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetric {
    pub name: String,
    pub metric_type: MetricType,
    pub description: String,
    pub labels: Vec<String>,
}

/// Metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// Access logs configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessLogsConfig {
    /// Enable access logs
    pub enabled: bool,
    /// Log format
    pub format: AccessLogFormat,
    /// Log level
    pub level: AccessLogLevel,
    /// Include response body
    pub include_response_body: bool,
    /// Include request body
    pub include_request_body: bool,
}

impl Default for AccessLogsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            format: AccessLogFormat::JSON,
            level: AccessLogLevel::Info,
            include_response_body: false,
            include_request_body: false,
        }
    }
}

/// Access log formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessLogFormat {
    JSON,
    Text,
    CEF,
}

/// Access log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessLogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Security policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicyConfig {
    /// Authorization policies
    pub authorization_policies: Vec<AuthorizationPolicy>,
    /// Security constraints
    pub security_constraints: Vec<SecurityConstraint>,
    /// JWT validation
    pub jwt_validation: JWTValidationConfig,
}

impl Default for SecurityPolicyConfig {
    fn default() -> Self {
        Self {
            authorization_policies: vec![
                AuthorizationPolicy {
                    name: "default-deny".to_string(),
                    action: PolicyAction::Deny,
                    rules: vec![
                        PolicyRule {
                            from: Some(vec!["source.ip != \"127.0.0.1\"".to_string()]),
                            to: Some(vec!["destination.service_name == \"oxirs-stream\"".to_string()]),
                            when: None,
                        },
                    ],
                },
            ],
            security_constraints: vec![
                SecurityConstraint {
                    name: "tls-only".to_string(),
                    constraint_type: ConstraintType::Protocol,
                    value: "https".to_string(),
                },
            ],
            jwt_validation: JWTValidationConfig::default(),
        }
    }
}

/// Authorization policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationPolicy {
    pub name: String,
    pub action: PolicyAction,
    pub rules: Vec<PolicyRule>,
}

/// Policy actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyAction {
    Allow,
    Deny,
    Audit,
}

/// Policy rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    pub from: Option<Vec<String>>,
    pub to: Option<Vec<String>>,
    pub when: Option<Vec<String>>,
}

/// Security constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConstraint {
    pub name: String,
    pub constraint_type: ConstraintType,
    pub value: String,
}

/// Constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    Protocol,
    Header,
    Method,
    Path,
}

/// JWT validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JWTValidationConfig {
    /// Enable JWT validation
    pub enabled: bool,
    /// JWT issuer
    pub issuer: String,
    /// JWT audiences
    pub audiences: Vec<String>,
    /// JWKS URI
    pub jwks_uri: String,
    /// Token validation rules
    pub validation_rules: Vec<JWTValidationRule>,
}

impl Default for JWTValidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            issuer: "https://oxirs.example.com".to_string(),
            audiences: vec!["oxirs-api".to_string()],
            jwks_uri: "https://oxirs.example.com/.well-known/jwks.json".to_string(),
            validation_rules: vec![
                JWTValidationRule {
                    claim: "exp".to_string(),
                    operation: ClaimOperation::GreaterThan,
                    value: "now()".to_string(),
                },
            ],
        }
    }
}

/// JWT validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JWTValidationRule {
    pub claim: String,
    pub operation: ClaimOperation,
    pub value: String,
}

/// Claim operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClaimOperation {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    Contains,
    In,
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Horizontal Pod Autoscaler
    pub hpa: HPAConfig,
    /// Vertical Pod Autoscaler
    pub vpa: VPAConfig,
    /// Cluster autoscaler
    pub cluster_autoscaler: ClusterAutoscalerConfig,
    /// Custom metrics for scaling
    pub custom_metrics: Vec<CustomScalingMetric>,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hpa: HPAConfig::default(),
            vpa: VPAConfig::default(),
            cluster_autoscaler: ClusterAutoscalerConfig::default(),
            custom_metrics: vec![
                CustomScalingMetric {
                    name: "stream_events_per_second".to_string(),
                    target_value: 1000.0,
                    metric_type: ScalingMetricType::Value,
                },
                CustomScalingMetric {
                    name: "memory_utilization".to_string(),
                    target_value: 70.0,
                    metric_type: ScalingMetricType::Utilization,
                },
            ],
        }
    }
}

/// Horizontal Pod Autoscaler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HPAConfig {
    /// Enable HPA
    pub enabled: bool,
    /// Minimum replicas
    pub min_replicas: u32,
    /// Maximum replicas
    pub max_replicas: u32,
    /// Target CPU utilization
    pub target_cpu_utilization: f64,
    /// Target memory utilization
    pub target_memory_utilization: f64,
    /// Scale down stabilization window
    pub scale_down_stabilization_window: Duration,
    /// Scale up stabilization window
    pub scale_up_stabilization_window: Duration,
}

impl Default for HPAConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_replicas: 2,
            max_replicas: 100,
            target_cpu_utilization: 70.0,
            target_memory_utilization: 80.0,
            scale_down_stabilization_window: Duration::from_secs(300),
            scale_up_stabilization_window: Duration::from_secs(60),
        }
    }
}

/// Vertical Pod Autoscaler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPAConfig {
    /// Enable VPA
    pub enabled: bool,
    /// Update mode
    pub update_mode: VPAUpdateMode,
    /// Resource policy
    pub resource_policy: VPAResourcePolicy,
}

impl Default for VPAConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default due to potential conflicts with HPA
            update_mode: VPAUpdateMode::Auto,
            resource_policy: VPAResourcePolicy::default(),
        }
    }
}

/// VPA update modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VPAUpdateMode {
    Off,
    Initial,
    Auto,
}

/// VPA resource policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPAResourcePolicy {
    /// Minimum allowed resources
    pub min_allowed: ResourceRequirements,
    /// Maximum allowed resources
    pub max_allowed: ResourceRequirements,
}

impl Default for VPAResourcePolicy {
    fn default() -> Self {
        Self {
            min_allowed: ResourceRequirements {
                cpu: "100m".to_string(),
                memory: "128Mi".to_string(),
            },
            max_allowed: ResourceRequirements {
                cpu: "4".to_string(),
                memory: "8Gi".to_string(),
            },
        }
    }
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu: String,
    pub memory: String,
}

/// Cluster autoscaler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterAutoscalerConfig {
    /// Enable cluster autoscaler
    pub enabled: bool,
    /// Minimum nodes
    pub min_nodes: u32,
    /// Maximum nodes
    pub max_nodes: u32,
    /// Scale down delay after add
    pub scale_down_delay_after_add: Duration,
    /// Scale down unneeded time
    pub scale_down_unneeded_time: Duration,
}

impl Default for ClusterAutoscalerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_nodes: 3,
            max_nodes: 100,
            scale_down_delay_after_add: Duration::from_secs(600),
            scale_down_unneeded_time: Duration::from_secs(600),
        }
    }
}

/// Custom scaling metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomScalingMetric {
    pub name: String,
    pub target_value: f64,
    pub metric_type: ScalingMetricType,
}

/// Scaling metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingMetricType {
    Value,
    AverageValue,
    Utilization,
}

/// Observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Alerting configuration
    pub alerting: AlertingConfig,
    /// Dashboards configuration
    pub dashboards: DashboardConfig,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            monitoring: MonitoringConfig::default(),
            logging: LoggingConfig::default(),
            alerting: AlertingConfig::default(),
            dashboards: DashboardConfig::default(),
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Prometheus configuration
    pub prometheus: PrometheusConfig,
    /// Service monitors
    pub service_monitors: Vec<ServiceMonitor>,
    /// Pod monitors
    pub pod_monitors: Vec<PodMonitor>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            prometheus: PrometheusConfig::default(),
            service_monitors: vec![
                ServiceMonitor {
                    name: "oxirs-stream-monitor".to_string(),
                    namespace: "oxirs".to_string(),
                    selector: LabelSelector {
                        match_labels: BTreeMap::from([
                            ("app".to_string(), "oxirs-stream".to_string()),
                        ]),
                    },
                    endpoints: vec![
                        ServiceMonitorEndpoint {
                            port: "metrics".to_string(),
                            path: "/metrics".to_string(),
                            interval: Duration::from_secs(30),
                        },
                    ],
                },
            ],
            pod_monitors: vec![],
        }
    }
}

/// Prometheus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusConfig {
    /// Enable Prometheus
    pub enabled: bool,
    /// Retention period
    pub retention: Duration,
    /// Storage size
    pub storage_size: String,
    /// Resource requirements
    pub resources: ResourceRequirements,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retention: Duration::from_secs(30 * 24 * 3600), // 30 days
            storage_size: "50Gi".to_string(),
            resources: ResourceRequirements {
                cpu: "2".to_string(),
                memory: "4Gi".to_string(),
            },
        }
    }
}

/// Service monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMonitor {
    pub name: String,
    pub namespace: String,
    pub selector: LabelSelector,
    pub endpoints: Vec<ServiceMonitorEndpoint>,
}

/// Service monitor endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMonitorEndpoint {
    pub port: String,
    pub path: String,
    pub interval: Duration,
}

/// Pod monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodMonitor {
    pub name: String,
    pub namespace: String,
    pub selector: LabelSelector,
    pub pod_metrics_endpoints: Vec<PodMetricsEndpoint>,
}

/// Pod metrics endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodMetricsEndpoint {
    pub port: String,
    pub path: String,
    pub interval: Duration,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log aggregation
    pub aggregation: LogAggregationConfig,
    /// Log forwarding
    pub forwarding: LogForwardingConfig,
    /// Log retention
    pub retention: LogRetentionConfig,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            aggregation: LogAggregationConfig::default(),
            forwarding: LogForwardingConfig::default(),
            retention: LogRetentionConfig::default(),
        }
    }
}

/// Log aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogAggregationConfig {
    /// Enable log aggregation
    pub enabled: bool,
    /// Aggregation provider
    pub provider: LogAggregationProvider,
    /// Buffer size
    pub buffer_size: usize,
    /// Flush interval
    pub flush_interval: Duration,
}

impl Default for LogAggregationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: LogAggregationProvider::Fluentd,
            buffer_size: 64 * 1024, // 64KB
            flush_interval: Duration::from_secs(10),
        }
    }
}

/// Log aggregation providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogAggregationProvider {
    Fluentd,
    Fluent_Bit,
    Logstash,
    Vector,
}

/// Log forwarding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogForwardingConfig {
    /// Enable log forwarding
    pub enabled: bool,
    /// Forwarding destinations
    pub destinations: Vec<LogDestination>,
}

impl Default for LogForwardingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            destinations: vec![
                LogDestination {
                    name: "elasticsearch".to_string(),
                    destination_type: LogDestinationType::Elasticsearch,
                    endpoint: "https://elasticsearch.example.com:9200".to_string(),
                    index: "oxirs-logs".to_string(),
                },
            ],
        }
    }
}

/// Log destination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogDestination {
    pub name: String,
    pub destination_type: LogDestinationType,
    pub endpoint: String,
    pub index: String,
}

/// Log destination types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogDestinationType {
    Elasticsearch,
    Splunk,
    CloudWatch,
    BigQuery,
    S3,
}

/// Log retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRetentionConfig {
    /// Retention period
    pub retention_period: Duration,
    /// Compression enabled
    pub compression: bool,
    /// Archive to cold storage
    pub cold_storage: bool,
}

impl Default for LogRetentionConfig {
    fn default() -> Self {
        Self {
            retention_period: Duration::from_secs(90 * 24 * 3600), // 90 days
            compression: true,
            cold_storage: true,
        }
    }
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Alert manager configuration
    pub alert_manager: AlertManagerConfig,
    /// Alert rules
    pub rules: Vec<AlertRule>,
    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            alert_manager: AlertManagerConfig::default(),
            rules: vec![
                AlertRule {
                    name: "high-cpu-usage".to_string(),
                    expression: "rate(container_cpu_usage_seconds_total[5m]) > 0.8".to_string(),
                    duration: Duration::from_secs(300),
                    severity: AlertSeverity::Warning,
                    summary: "High CPU usage detected".to_string(),
                    description: "CPU usage is above 80% for more than 5 minutes".to_string(),
                },
                AlertRule {
                    name: "high-memory-usage".to_string(),
                    expression: "container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9".to_string(),
                    duration: Duration::from_secs(300),
                    severity: AlertSeverity::Critical,
                    summary: "High memory usage detected".to_string(),
                    description: "Memory usage is above 90% for more than 5 minutes".to_string(),
                },
            ],
            notification_channels: vec![
                NotificationChannel {
                    name: "slack".to_string(),
                    channel_type: NotificationChannelType::Slack,
                    webhook_url: Some("https://hooks.slack.com/services/...".to_string()),
                    email_addresses: None,
                },
            ],
        }
    }
}

/// Alert manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertManagerConfig {
    /// Enable alert manager
    pub enabled: bool,
    /// Storage size
    pub storage_size: String,
    /// Retention period
    pub retention: Duration,
    /// Resource requirements
    pub resources: ResourceRequirements,
}

impl Default for AlertManagerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            storage_size: "10Gi".to_string(),
            retention: Duration::from_secs(30 * 24 * 3600), // 30 days
            resources: ResourceRequirements {
                cpu: "500m".to_string(),
                memory: "1Gi".to_string(),
            },
        }
    }
}

/// Alert rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub expression: String,
    pub duration: Duration,
    pub severity: AlertSeverity,
    pub summary: String,
    pub description: String,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Notification channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    pub name: String,
    pub channel_type: NotificationChannelType,
    pub webhook_url: Option<String>,
    pub email_addresses: Option<Vec<String>>,
}

/// Notification channel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannelType {
    Slack,
    Email,
    PagerDuty,
    Webhook,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Grafana configuration
    pub grafana: GrafanaConfig,
    /// Dashboard definitions
    pub dashboards: Vec<Dashboard>,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            grafana: GrafanaConfig::default(),
            dashboards: vec![
                Dashboard {
                    name: "oxirs-overview".to_string(),
                    title: "OxiRS Stream Overview".to_string(),
                    description: "Overview dashboard for OxiRS Stream".to_string(),
                    panels: vec![
                        DashboardPanel {
                            title: "Events per Second".to_string(),
                            panel_type: PanelType::Graph,
                            query: "rate(oxirs_stream_events_total[1m])".to_string(),
                        },
                        DashboardPanel {
                            title: "Processing Latency".to_string(),
                            panel_type: PanelType::Graph,
                            query: "histogram_quantile(0.95, oxirs_stream_latency_seconds)".to_string(),
                        },
                    ],
                },
            ],
        }
    }
}

/// Grafana configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrafanaConfig {
    /// Enable Grafana
    pub enabled: bool,
    /// Admin user
    pub admin_user: String,
    /// Admin password
    pub admin_password: String,
    /// Persistence configuration
    pub persistence: PersistenceConfig,
    /// Resource requirements
    pub resources: ResourceRequirements,
}

impl Default for GrafanaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            admin_user: "admin".to_string(),
            admin_password: "admin".to_string(), // Should be changed in production
            persistence: PersistenceConfig {
                enabled: true,
                size: "10Gi".to_string(),
                storage_class: "default".to_string(),
            },
            resources: ResourceRequirements {
                cpu: "500m".to_string(),
                memory: "1Gi".to_string(),
            },
        }
    }
}

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    pub enabled: bool,
    pub size: String,
    pub storage_class: String,
}

/// Dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    pub name: String,
    pub title: String,
    pub description: String,
    pub panels: Vec<DashboardPanel>,
}

/// Dashboard panel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPanel {
    pub title: String,
    pub panel_type: PanelType,
    pub query: String,
}

/// Panel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PanelType {
    Graph,
    SingleStat,
    Table,
    Heatmap,
    Logs,
}

/// Multi-cloud configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiCloudConfig {
    /// Enable multi-cloud deployment
    pub enabled: bool,
    /// Primary cloud provider
    pub primary_provider: CloudProvider,
    /// Secondary cloud providers
    pub secondary_providers: Vec<CloudProvider>,
    /// Replication strategy
    pub replication_strategy: ReplicationStrategy,
    /// Failover configuration
    pub failover: MultiCloudFailoverConfig,
}

impl Default for MultiCloudConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            primary_provider: CloudProvider::AWS,
            secondary_providers: vec![CloudProvider::GCP, CloudProvider::Azure],
            replication_strategy: ReplicationStrategy::ActivePassive,
            failover: MultiCloudFailoverConfig::default(),
        }
    }
}

/// Cloud providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudProvider {
    AWS,
    GCP,
    Azure,
    DigitalOcean,
    Linode,
    OnPremise,
}

/// Replication strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationStrategy {
    ActiveActive,
    ActivePassive,
    MultiMaster,
}

/// Multi-cloud failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiCloudFailoverConfig {
    /// Enable automatic failover
    pub auto_failover: bool,
    /// Failover threshold
    pub failover_threshold: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Failback enabled
    pub failback_enabled: bool,
}

impl Default for MultiCloudFailoverConfig {
    fn default() -> Self {
        Self {
            auto_failover: true,
            failover_threshold: Duration::from_secs(300),
            health_check_interval: Duration::from_secs(30),
            failback_enabled: true,
        }
    }
}

/// GitOps configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitOpsConfig {
    /// Enable GitOps
    pub enabled: bool,
    /// GitOps provider
    pub provider: GitOpsProvider,
    /// Repository configuration
    pub repository: RepositoryConfig,
    /// Sync configuration
    pub sync: SyncConfig,
    /// CD pipeline configuration
    pub cd_pipeline: CDPipelineConfig,
}

impl Default for GitOpsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: GitOpsProvider::ArgoCD,
            repository: RepositoryConfig::default(),
            sync: SyncConfig::default(),
            cd_pipeline: CDPipelineConfig::default(),
        }
    }
}

/// GitOps providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GitOpsProvider {
    ArgoCD,
    Flux,
    Tekton,
    Jenkins,
}

/// Repository configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryConfig {
    /// Repository URL
    pub url: String,
    /// Branch
    pub branch: String,
    /// Path
    pub path: String,
    /// Credentials
    pub credentials: GitCredentials,
}

impl Default for RepositoryConfig {
    fn default() -> Self {
        Self {
            url: "https://github.com/oxirs/oxirs-deploy.git".to_string(),
            branch: "main".to_string(),
            path: "kubernetes".to_string(),
            credentials: GitCredentials::default(),
        }
    }
}

/// Git credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitCredentials {
    /// Credential type
    pub credential_type: GitCredentialType,
    /// Username
    pub username: Option<String>,
    /// Password or token
    pub password: Option<String>,
    /// SSH private key
    pub ssh_private_key: Option<String>,
}

impl Default for GitCredentials {
    fn default() -> Self {
        Self {
            credential_type: GitCredentialType::Token,
            username: None,
            password: None,
            ssh_private_key: None,
        }
    }
}

/// Git credential types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GitCredentialType {
    Token,
    UsernamePassword,
    SSH,
}

/// Sync configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Auto sync enabled
    pub auto_sync: bool,
    /// Sync interval
    pub sync_interval: Duration,
    /// Prune resources
    pub prune: bool,
    /// Self heal
    pub self_heal: bool,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            auto_sync: true,
            sync_interval: Duration::from_secs(180),
            prune: true,
            self_heal: true,
        }
    }
}

/// CD pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CDPipelineConfig {
    /// Enable CD pipeline
    pub enabled: bool,
    /// Pipeline stages
    pub stages: Vec<PipelineStage>,
    /// Deployment strategy
    pub deployment_strategy: DeploymentStrategy,
}

impl Default for CDPipelineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            stages: vec![
                PipelineStage {
                    name: "build".to_string(),
                    stage_type: StageType::Build,
                    commands: vec![
                        "docker build -t oxirs/stream:${GIT_COMMIT} .".to_string(),
                        "docker push oxirs/stream:${GIT_COMMIT}".to_string(),
                    ],
                },
                PipelineStage {
                    name: "test".to_string(),
                    stage_type: StageType::Test,
                    commands: vec![
                        "cargo test --all".to_string(),
                        "helm lint charts/oxirs-stream".to_string(),
                    ],
                },
                PipelineStage {
                    name: "deploy-staging".to_string(),
                    stage_type: StageType::Deploy,
                    commands: vec![
                        "helm upgrade --install oxirs-stream-staging charts/oxirs-stream --namespace staging".to_string(),
                    ],
                },
            ],
            deployment_strategy: DeploymentStrategy::RollingUpdate,
        }
    }
}

/// Pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    pub name: String,
    pub stage_type: StageType,
    pub commands: Vec<String>,
}

/// Stage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StageType {
    Build,
    Test,
    Deploy,
    Approve,
}

/// Deployment strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    RollingUpdate,
    BlueGreen,
    Canary,
    Recreate,
}

/// Cloud-native manager
pub struct CloudNativeManager {
    config: CloudNativeConfig,
    kubernetes_client: Arc<dyn KubernetesClient>,
    service_mesh_client: Arc<dyn ServiceMeshClient>,
    metrics: Arc<RwLock<CloudNativeMetrics>>,
}

impl CloudNativeManager {
    /// Create a new cloud-native manager
    pub fn new(
        config: CloudNativeConfig,
        kubernetes_client: Arc<dyn KubernetesClient>,
        service_mesh_client: Arc<dyn ServiceMeshClient>,
    ) -> Self {
        Self {
            config,
            kubernetes_client,
            service_mesh_client,
            metrics: Arc::new(RwLock::new(CloudNativeMetrics::default())),
        }
    }

    /// Deploy OxiRS resources to Kubernetes
    pub async fn deploy(&self) -> Result<()> {
        info!("Starting deployment of OxiRS resources");

        // Create namespace
        self.create_namespace().await?;

        // Deploy CRDs
        self.deploy_crds().await?;

        // Deploy operator
        if self.config.kubernetes.operator.enabled {
            self.deploy_operator().await?;
        }

        // Configure service mesh
        if self.config.service_mesh.enabled {
            self.configure_service_mesh().await?;
        }

        // Setup monitoring
        if self.config.observability.monitoring.prometheus.enabled {
            self.setup_monitoring().await?;
        }

        // Setup auto-scaling
        if self.config.auto_scaling.enabled {
            self.setup_auto_scaling().await?;
        }

        info!("Successfully deployed OxiRS resources");
        Ok(())
    }

    /// Create namespace
    async fn create_namespace(&self) -> Result<()> {
        let namespace = KubernetesResource::Namespace {
            metadata: ObjectMetadata {
                name: self.config.kubernetes.namespace.clone(),
                namespace: None,
                labels: BTreeMap::from([
                    ("app".to_string(), "oxirs".to_string()),
                    ("managed-by".to_string(), "oxirs-operator".to_string()),
                ]),
                annotations: BTreeMap::new(),
            },
        };

        self.kubernetes_client.apply_resource(namespace).await?;
        Ok(())
    }

    /// Deploy CRDs
    async fn deploy_crds(&self) -> Result<()> {
        for crd in &self.config.kubernetes.crds {
            let resource = KubernetesResource::CustomResourceDefinition(crd.clone());
            self.kubernetes_client.apply_resource(resource).await?;
        }
        Ok(())
    }

    /// Deploy operator
    async fn deploy_operator(&self) -> Result<()> {
        let operator_deployment = self.create_operator_deployment();
        self.kubernetes_client.apply_resource(operator_deployment).await?;
        Ok(())
    }

    /// Create operator deployment
    fn create_operator_deployment(&self) -> KubernetesResource {
        KubernetesResource::Deployment {
            metadata: ObjectMetadata {
                name: "oxirs-operator".to_string(),
                namespace: Some(self.config.kubernetes.namespace.clone()),
                labels: BTreeMap::from([
                    ("app".to_string(), "oxirs-operator".to_string()),
                ]),
                annotations: BTreeMap::new(),
            },
            spec: DeploymentSpec {
                replicas: 1,
                selector: LabelSelector {
                    match_labels: BTreeMap::from([
                        ("app".to_string(), "oxirs-operator".to_string()),
                    ]),
                },
                template: PodTemplateSpec {
                    metadata: ObjectMetadata {
                        name: "oxirs-operator".to_string(),
                        namespace: Some(self.config.kubernetes.namespace.clone()),
                        labels: BTreeMap::from([
                            ("app".to_string(), "oxirs-operator".to_string()),
                        ]),
                        annotations: BTreeMap::new(),
                    },
                    spec: PodSpec {
                        containers: vec![
                            Container {
                                name: "operator".to_string(),
                                image: self.config.kubernetes.operator.image.clone(),
                                resources: Some(ResourceRequirements {
                                    cpu: "100m".to_string(),
                                    memory: "128Mi".to_string(),
                                }),
                                env: vec![
                                    EnvVar {
                                        name: "NAMESPACE".to_string(),
                                        value: self.config.kubernetes.namespace.clone(),
                                    },
                                    EnvVar {
                                        name: "RECONCILE_INTERVAL".to_string(),
                                        value: format!("{}s", self.config.kubernetes.operator.reconcile_interval.as_secs()),
                                    },
                                ],
                                ports: vec![
                                    ContainerPort {
                                        name: "metrics".to_string(),
                                        container_port: 8080,
                                        protocol: "TCP".to_string(),
                                    },
                                ],
                            },
                        ],
                        service_account: Some("oxirs-operator".to_string()),
                    },
                },
            },
        }
    }

    /// Configure service mesh
    async fn configure_service_mesh(&self) -> Result<()> {
        match self.config.service_mesh.provider {
            ServiceMeshProvider::Istio => self.configure_istio().await?,
            ServiceMeshProvider::Linkerd => self.configure_linkerd().await?,
            ServiceMeshProvider::ConsulConnect => self.configure_consul_connect().await?,
            _ => {
                warn!("Service mesh provider {:?} not implemented", self.config.service_mesh.provider);
            }
        }
        Ok(())
    }

    /// Configure Istio
    async fn configure_istio(&self) -> Result<()> {
        // Enable sidecar injection
        self.service_mesh_client.enable_sidecar_injection(&self.config.kubernetes.namespace).await?;

        // Configure destination rules
        self.service_mesh_client.apply_destination_rules(&self.config.service_mesh.traffic_management).await?;

        // Configure virtual services
        self.service_mesh_client.apply_virtual_services(&self.config.service_mesh.traffic_management).await?;

        // Configure authorization policies
        self.service_mesh_client.apply_authorization_policies(&self.config.service_mesh.security_policies).await?;

        Ok(())
    }

    /// Configure Linkerd
    async fn configure_linkerd(&self) -> Result<()> {
        // Implement Linkerd configuration
        warn!("Linkerd configuration not yet implemented");
        Ok(())
    }

    /// Configure Consul Connect
    async fn configure_consul_connect(&self) -> Result<()> {
        // Implement Consul Connect configuration
        warn!("Consul Connect configuration not yet implemented");
        Ok(())
    }

    /// Setup monitoring
    async fn setup_monitoring(&self) -> Result<()> {
        // Deploy Prometheus
        self.deploy_prometheus().await?;

        // Deploy service monitors
        self.deploy_service_monitors().await?;

        // Deploy Grafana
        if self.config.observability.dashboards.grafana.enabled {
            self.deploy_grafana().await?;
        }

        Ok(())
    }

    /// Deploy Prometheus
    async fn deploy_prometheus(&self) -> Result<()> {
        let prometheus = KubernetesResource::StatefulSet {
            metadata: ObjectMetadata {
                name: "prometheus".to_string(),
                namespace: Some(self.config.kubernetes.namespace.clone()),
                labels: BTreeMap::from([
                    ("app".to_string(), "prometheus".to_string()),
                ]),
                annotations: BTreeMap::new(),
            },
            spec: StatefulSetSpec {
                replicas: 1,
                selector: LabelSelector {
                    match_labels: BTreeMap::from([
                        ("app".to_string(), "prometheus".to_string()),
                    ]),
                },
                template: PodTemplateSpec {
                    metadata: ObjectMetadata {
                        name: "prometheus".to_string(),
                        namespace: Some(self.config.kubernetes.namespace.clone()),
                        labels: BTreeMap::from([
                            ("app".to_string(), "prometheus".to_string()),
                        ]),
                        annotations: BTreeMap::new(),
                    },
                    spec: PodSpec {
                        containers: vec![
                            Container {
                                name: "prometheus".to_string(),
                                image: "prom/prometheus:latest".to_string(),
                                resources: Some(self.config.observability.monitoring.prometheus.resources.clone()),
                                env: vec![],
                                ports: vec![
                                    ContainerPort {
                                        name: "web".to_string(),
                                        container_port: 9090,
                                        protocol: "TCP".to_string(),
                                    },
                                ],
                            },
                        ],
                        service_account: Some("prometheus".to_string()),
                    },
                },
                volume_claim_templates: vec![
                    VolumeClaimTemplate {
                        metadata: ObjectMetadata {
                            name: "prometheus-storage".to_string(),
                            namespace: Some(self.config.kubernetes.namespace.clone()),
                            labels: BTreeMap::new(),
                            annotations: BTreeMap::new(),
                        },
                        spec: VolumeClaimSpec {
                            access_modes: vec!["ReadWriteOnce".to_string()],
                            resources: ResourceRequirements {
                                cpu: "".to_string(),
                                memory: self.config.observability.monitoring.prometheus.storage_size.clone(),
                            },
                        },
                    },
                ],
            },
        };

        self.kubernetes_client.apply_resource(prometheus).await?;
        Ok(())
    }

    /// Deploy service monitors
    async fn deploy_service_monitors(&self) -> Result<()> {
        for monitor in &self.config.observability.monitoring.service_monitors {
            let service_monitor = KubernetesResource::ServiceMonitor(monitor.clone());
            self.kubernetes_client.apply_resource(service_monitor).await?;
        }
        Ok(())
    }

    /// Deploy Grafana
    async fn deploy_grafana(&self) -> Result<()> {
        let grafana = KubernetesResource::Deployment {
            metadata: ObjectMetadata {
                name: "grafana".to_string(),
                namespace: Some(self.config.kubernetes.namespace.clone()),
                labels: BTreeMap::from([
                    ("app".to_string(), "grafana".to_string()),
                ]),
                annotations: BTreeMap::new(),
            },
            spec: DeploymentSpec {
                replicas: 1,
                selector: LabelSelector {
                    match_labels: BTreeMap::from([
                        ("app".to_string(), "grafana".to_string()),
                    ]),
                },
                template: PodTemplateSpec {
                    metadata: ObjectMetadata {
                        name: "grafana".to_string(),
                        namespace: Some(self.config.kubernetes.namespace.clone()),
                        labels: BTreeMap::from([
                            ("app".to_string(), "grafana".to_string()),
                        ]),
                        annotations: BTreeMap::new(),
                    },
                    spec: PodSpec {
                        containers: vec![
                            Container {
                                name: "grafana".to_string(),
                                image: "grafana/grafana:latest".to_string(),
                                resources: Some(self.config.observability.dashboards.grafana.resources.clone()),
                                env: vec![
                                    EnvVar {
                                        name: "GF_SECURITY_ADMIN_USER".to_string(),
                                        value: self.config.observability.dashboards.grafana.admin_user.clone(),
                                    },
                                    EnvVar {
                                        name: "GF_SECURITY_ADMIN_PASSWORD".to_string(),
                                        value: self.config.observability.dashboards.grafana.admin_password.clone(),
                                    },
                                ],
                                ports: vec![
                                    ContainerPort {
                                        name: "web".to_string(),
                                        container_port: 3000,
                                        protocol: "TCP".to_string(),
                                    },
                                ],
                            },
                        ],
                        service_account: Some("grafana".to_string()),
                    },
                },
            },
        };

        self.kubernetes_client.apply_resource(grafana).await?;
        Ok(())
    }

    /// Setup auto-scaling
    async fn setup_auto_scaling(&self) -> Result<()> {
        if self.config.auto_scaling.hpa.enabled {
            self.setup_hpa().await?;
        }

        if self.config.auto_scaling.vpa.enabled {
            self.setup_vpa().await?;
        }

        Ok(())
    }

    /// Setup HPA
    async fn setup_hpa(&self) -> Result<()> {
        let hpa = KubernetesResource::HorizontalPodAutoscaler {
            metadata: ObjectMetadata {
                name: "oxirs-stream-hpa".to_string(),
                namespace: Some(self.config.kubernetes.namespace.clone()),
                labels: BTreeMap::from([
                    ("app".to_string(), "oxirs-stream".to_string()),
                ]),
                annotations: BTreeMap::new(),
            },
            spec: HPASpec {
                scale_target_ref: ScaleTargetRef {
                    api_version: "apps/v1".to_string(),
                    kind: "Deployment".to_string(),
                    name: "oxirs-stream".to_string(),
                },
                min_replicas: Some(self.config.auto_scaling.hpa.min_replicas),
                max_replicas: self.config.auto_scaling.hpa.max_replicas,
                target_cpu_utilization_percentage: Some(self.config.auto_scaling.hpa.target_cpu_utilization as i32),
                metrics: vec![
                    MetricSpec {
                        metric_type: "Resource".to_string(),
                        resource: Some(ResourceMetricSource {
                            name: "cpu".to_string(),
                            target: MetricTarget {
                                target_type: "Utilization".to_string(),
                                average_utilization: Some(self.config.auto_scaling.hpa.target_cpu_utilization as i32),
                            },
                        }),
                    },
                    MetricSpec {
                        metric_type: "Resource".to_string(),
                        resource: Some(ResourceMetricSource {
                            name: "memory".to_string(),
                            target: MetricTarget {
                                target_type: "Utilization".to_string(),
                                average_utilization: Some(self.config.auto_scaling.hpa.target_memory_utilization as i32),
                            },
                        }),
                    },
                ],
            },
        };

        self.kubernetes_client.apply_resource(hpa).await?;
        Ok(())
    }

    /// Setup VPA
    async fn setup_vpa(&self) -> Result<()> {
        warn!("VPA setup not yet implemented");
        Ok(())
    }

    /// Get metrics
    pub async fn get_metrics(&self) -> CloudNativeMetrics {
        self.metrics.read().await.clone()
    }
}

/// Kubernetes client trait
#[async_trait::async_trait]
pub trait KubernetesClient: Send + Sync {
    async fn apply_resource(&self, resource: KubernetesResource) -> Result<()>;
    async fn delete_resource(&self, resource: KubernetesResource) -> Result<()>;
    async fn get_resource(&self, resource_type: &str, name: &str, namespace: &str) -> Result<Option<KubernetesResource>>;
    async fn list_resources(&self, resource_type: &str, namespace: &str) -> Result<Vec<KubernetesResource>>;
}

/// Service mesh client trait
#[async_trait::async_trait]
pub trait ServiceMeshClient: Send + Sync {
    async fn enable_sidecar_injection(&self, namespace: &str) -> Result<()>;
    async fn apply_destination_rules(&self, config: &TrafficManagementConfig) -> Result<()>;
    async fn apply_virtual_services(&self, config: &TrafficManagementConfig) -> Result<()>;
    async fn apply_authorization_policies(&self, config: &SecurityPolicyConfig) -> Result<()>;
}

/// Kubernetes resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KubernetesResource {
    Namespace {
        metadata: ObjectMetadata,
    },
    CustomResourceDefinition(CustomResourceDefinition),
    Deployment {
        metadata: ObjectMetadata,
        spec: DeploymentSpec,
    },
    StatefulSet {
        metadata: ObjectMetadata,
        spec: StatefulSetSpec,
    },
    Service {
        metadata: ObjectMetadata,
        spec: ServiceSpec,
    },
    ConfigMap {
        metadata: ObjectMetadata,
        data: BTreeMap<String, String>,
    },
    Secret {
        metadata: ObjectMetadata,
        data: BTreeMap<String, Vec<u8>>,
    },
    ServiceMonitor(ServiceMonitor),
    HorizontalPodAutoscaler {
        metadata: ObjectMetadata,
        spec: HPASpec,
    },
}

/// Deployment specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentSpec {
    pub replicas: u32,
    pub selector: LabelSelector,
    pub template: PodTemplateSpec,
}

/// StatefulSet specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatefulSetSpec {
    pub replicas: u32,
    pub selector: LabelSelector,
    pub template: PodTemplateSpec,
    pub volume_claim_templates: Vec<VolumeClaimTemplate>,
}

/// Pod template specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodTemplateSpec {
    pub metadata: ObjectMetadata,
    pub spec: PodSpec,
}

/// Pod specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodSpec {
    pub containers: Vec<Container>,
    pub service_account: Option<String>,
}

/// Container specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Container {
    pub name: String,
    pub image: String,
    pub resources: Option<ResourceRequirements>,
    pub env: Vec<EnvVar>,
    pub ports: Vec<ContainerPort>,
}

/// Environment variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvVar {
    pub name: String,
    pub value: String,
}

/// Container port
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerPort {
    pub name: String,
    pub container_port: u16,
    pub protocol: String,
}

/// Service specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceSpec {
    pub selector: LabelSelector,
    pub ports: Vec<ServicePort>,
    pub service_type: String,
}

/// Service port
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePort {
    pub name: String,
    pub port: u16,
    pub target_port: u16,
    pub protocol: String,
}

/// Volume claim template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeClaimTemplate {
    pub metadata: ObjectMetadata,
    pub spec: VolumeClaimSpec,
}

/// Volume claim specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeClaimSpec {
    pub access_modes: Vec<String>,
    pub resources: ResourceRequirements,
}

/// HPA specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HPASpec {
    pub scale_target_ref: ScaleTargetRef,
    pub min_replicas: Option<u32>,
    pub max_replicas: u32,
    pub target_cpu_utilization_percentage: Option<i32>,
    pub metrics: Vec<MetricSpec>,
}

/// Scale target reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleTargetRef {
    pub api_version: String,
    pub kind: String,
    pub name: String,
}

/// Metric specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSpec {
    #[serde(rename = "type")]
    pub metric_type: String,
    pub resource: Option<ResourceMetricSource>,
}

/// Resource metric source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetricSource {
    pub name: String,
    pub target: MetricTarget,
}

/// Metric target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTarget {
    #[serde(rename = "type")]
    pub target_type: String,
    pub average_utilization: Option<i32>,
}

/// Cloud-native metrics
#[derive(Debug, Clone, Default)]
pub struct CloudNativeMetrics {
    pub deployments_created: u64,
    pub deployments_updated: u64,
    pub deployments_deleted: u64,
    pub pods_running: u64,
    pub service_mesh_enabled: bool,
    pub auto_scaling_events: u64,
    pub last_deployment_time: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_native_config_defaults() {
        let config = CloudNativeConfig::default();
        assert!(config.kubernetes.enabled);
        assert_eq!(config.kubernetes.namespace, "oxirs");
        assert!(config.service_mesh.enabled);
        assert_eq!(config.service_mesh.provider, ServiceMeshProvider::Istio);
        assert!(config.auto_scaling.enabled);
    }

    #[test]
    fn test_custom_resource_definition_creation() {
        let crd = CustomResourceDefinition::stream_processor();
        assert_eq!(crd.kind, "CustomResourceDefinition");
        assert_eq!(crd.spec.group, "oxirs.io");
        assert_eq!(crd.spec.names.kind, "StreamProcessor");
    }

    #[test]
    fn test_kubernetes_config_serialization() {
        let config = KubernetesConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: KubernetesConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(config.namespace, deserialized.namespace);
        assert_eq!(config.enabled, deserialized.enabled);
    }

    #[test]
    fn test_service_mesh_config() {
        let config = ServiceMeshConfig::default();
        assert!(config.enabled);
        assert!(config.mtls.enabled);
        assert_eq!(config.mtls.mode, MutualTLSMode::Strict);
        assert_eq!(config.traffic_management.load_balancing, LoadBalancingStrategy::RoundRobin);
    }

    #[test]
    fn test_auto_scaling_config() {
        let config = AutoScalingConfig::default();
        assert!(config.enabled);
        assert!(config.hpa.enabled);
        assert_eq!(config.hpa.min_replicas, 2);
        assert_eq!(config.hpa.max_replicas, 100);
        assert_eq!(config.hpa.target_cpu_utilization, 70.0);
    }

    #[test]
    fn test_observability_config() {
        let config = ObservabilityConfig::default();
        assert!(config.monitoring.prometheus.enabled);
        assert!(config.dashboards.grafana.enabled);
        assert_eq!(config.dashboards.grafana.admin_user, "admin");
    }

    #[test]
    fn test_gitops_config() {
        let config = GitOpsConfig::default();
        assert!(config.enabled);
        assert_eq!(config.provider, GitOpsProvider::ArgoCD);
        assert!(config.sync.auto_sync);
        assert!(config.sync.prune);
        assert!(config.sync.self_heal);
    }
}