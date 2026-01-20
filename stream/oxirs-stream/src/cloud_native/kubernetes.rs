//! Kubernetes Integration Types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
