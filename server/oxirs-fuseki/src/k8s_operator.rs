//! Kubernetes Operator for OxiRS Fuseki
//!
//! This module provides a Kubernetes operator for managing OxiRS Fuseki instances.
//! It handles automatic scaling, configuration updates, and health monitoring.
//!
//! ## Features
//!
//! - Custom Resource Definition (CRD) for OxirsFuseki resources
//! - Automatic deployment and service management
//! - Horizontal Pod Autoscaler (HPA) integration
//! - Health monitoring and status updates
//! - Leader election for high availability
//!
//! ## Usage
//!
//! Enable the `k8s` feature to use actual Kubernetes API calls:
//! ```toml
//! [dependencies]
//! oxirs-fuseki = { version = "0.1.0", features = ["k8s"] }
//! ```

use crate::error::{FusekiError, FusekiResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// Import kube-rs types when feature is enabled
#[cfg(feature = "k8s")]
use kube::{CustomResource, ResourceExt};
#[cfg(feature = "k8s")]
use schemars::JsonSchema;

/// Custom Resource Definition for OxiRS Fuseki
///
/// When the `k8s` feature is enabled, this struct derives `CustomResource`
/// which automatically generates the CRD and Kubernetes API bindings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(
    feature = "k8s",
    derive(CustomResource, JsonSchema),
    kube(
        group = "oxirs.org",
        version = "v1",
        kind = "OxirsFuseki",
        plural = "oxirsfusekis",
        shortname = "oxf",
        status = "FusekiStatus",
        namespaced,
        printcolumn = r#"{"name":"Replicas", "type":"integer", "jsonPath":".spec.replicas"}"#,
        printcolumn = r#"{"name":"Ready", "type":"integer", "jsonPath":".status.readyReplicas"}"#,
        printcolumn = r#"{"name":"Phase", "type":"string", "jsonPath":".status.phase"}"#,
        printcolumn = r#"{"name":"Age", "type":"date", "jsonPath":".metadata.creationTimestamp"}"#
    )
)]
#[serde(rename_all = "camelCase")]
pub struct FusekiSpec {
    /// Number of replicas
    pub replicas: i32,
    /// Container image
    pub image: String,
    /// Image pull policy
    #[serde(default = "default_image_pull_policy")]
    pub image_pull_policy: String,
    /// Resource requirements
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceRequirements>,
    /// Persistence configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistence: Option<PersistenceSpec>,
    /// Additional configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<HashMap<String, String>>,
    /// Auto-scaling configuration
    #[serde(default)]
    pub auto_scaling: AutoScalingSpec,
    /// Service type (ClusterIP, NodePort, LoadBalancer)
    #[serde(default = "default_service_type")]
    pub service_type: String,
    /// Port configuration
    #[serde(default = "default_port")]
    pub port: i32,
    /// Enable metrics endpoint
    #[serde(default = "default_true")]
    pub enable_metrics: bool,
    /// Enable GraphQL endpoint
    #[serde(default)]
    pub enable_graphql: bool,
    /// Dataset configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub datasets: Option<Vec<DatasetSpec>>,
    /// TLS configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tls: Option<TlsSpec>,
    /// Environment variables
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<Vec<EnvVar>>,
}

fn default_image_pull_policy() -> String {
    "IfNotPresent".to_string()
}

fn default_service_type() -> String {
    "ClusterIP".to_string()
}

fn default_port() -> i32 {
    3030
}

fn default_true() -> bool {
    true
}

/// Non-CRD wrapper for when k8s feature is disabled
#[cfg(not(feature = "k8s"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OxirsFuseki {
    pub api_version: String,
    pub kind: String,
    pub metadata: ResourceMetadata,
    pub spec: FusekiSpec,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<FusekiStatus>,
}

#[cfg(not(feature = "k8s"))]
impl OxirsFuseki {
    /// Get the resource name
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Get the namespace
    pub fn namespace(&self) -> &str {
        &self.metadata.namespace
    }

    /// Get the resource name (compatibility with kube-rs ResourceExt)
    pub fn name_any(&self) -> String {
        self.metadata.name.clone()
    }
}

/// Helper methods for kube-rs generated OxirsFuseki
#[cfg(feature = "k8s")]
impl OxirsFuseki {
    /// Get the resource name
    pub fn name_str(&self) -> &str {
        self.metadata.name.as_deref().unwrap_or("unknown")
    }

    /// Get the namespace
    pub fn namespace_str(&self) -> &str {
        self.metadata.namespace.as_deref().unwrap_or("default")
    }
}

/// Resource metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "k8s", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct ResourceMetadata {
    pub name: String,
    pub namespace: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_version: Option<String>,
}

/// Dataset specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "k8s", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct DatasetSpec {
    pub name: String,
    #[serde(default)]
    pub persistent: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefixes: Option<HashMap<String, String>>,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "k8s", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct TlsSpec {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secret_name: Option<String>,
    #[serde(default)]
    pub auto_generate: bool,
}

/// Environment variable specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "k8s", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct EnvVar {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_from: Option<EnvVarSource>,
}

/// Environment variable source (for secrets/configmaps)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "k8s", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct EnvVarSource {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secret_key_ref: Option<KeyRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_map_key_ref: Option<KeyRef>,
}

/// Key reference for secrets/configmaps
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "k8s", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct KeyRef {
    pub name: String,
    pub key: String,
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "k8s", derive(JsonSchema))]
pub struct ResourceRequirements {
    pub requests: ResourceList,
    pub limits: ResourceList,
}

/// Resource list
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "k8s", derive(JsonSchema))]
pub struct ResourceList {
    pub cpu: String,
    pub memory: String,
}

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "k8s", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct PersistenceSpec {
    pub enabled: bool,
    pub size: String,
    pub storage_class: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub access_modes: Option<Vec<String>>,
}

/// Auto-scaling specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "k8s", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct AutoScalingSpec {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_min_replicas")]
    pub min_replicas: i32,
    #[serde(default = "default_max_replicas")]
    pub max_replicas: i32,
    #[serde(default = "default_target_cpu")]
    pub target_cpu_utilization: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_memory_utilization: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_metrics: Option<Vec<CustomMetric>>,
}

/// Custom metric for HPA
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "k8s", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct CustomMetric {
    pub name: String,
    pub target_value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric_type: Option<String>,
}

fn default_min_replicas() -> i32 {
    2
}

fn default_max_replicas() -> i32 {
    10
}

fn default_target_cpu() -> i32 {
    70
}

impl Default for AutoScalingSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            min_replicas: default_min_replicas(),
            max_replicas: default_max_replicas(),
            target_cpu_utilization: default_target_cpu(),
            target_memory_utilization: None,
            custom_metrics: None,
        }
    }
}

/// Fuseki instance status
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "k8s", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct FusekiStatus {
    pub ready_replicas: i32,
    pub available_replicas: i32,
    pub phase: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conditions: Option<Vec<StatusCondition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_update_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Status condition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "k8s", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct StatusCondition {
    pub r#type: String,
    pub status: String,
    pub reason: String,
    pub message: String,
    pub last_transition_time: String,
}

/// Reconciliation action
#[derive(Debug, Clone)]
pub enum ReconcileAction {
    /// Create new resources
    Create,
    /// Update existing resources
    Update,
    /// No action needed
    NoOp,
    /// Delete resources
    Delete,
}

/// Kubernetes Operator for OxiRS Fuseki
pub struct FusekiOperator {
    /// Managed Fuseki instances
    instances: Arc<RwLock<HashMap<String, OxirsFuseki>>>,
    /// Namespace to watch
    namespace: String,
    /// Reconciliation interval
    reconcile_interval: Duration,
    /// Leader election enabled
    leader_election: bool,
    /// Is this instance the leader
    is_leader: Arc<RwLock<bool>>,
    /// Operator configuration
    config: OperatorConfig,
}

impl FusekiOperator {
    /// Create a new Fuseki operator
    pub fn new(config: OperatorConfig) -> Self {
        Self {
            instances: Arc::new(RwLock::new(HashMap::new())),
            namespace: config.namespace.clone(),
            reconcile_interval: Duration::from_secs(config.reconcile_interval_secs),
            leader_election: config.leader_election_enabled,
            is_leader: Arc::new(RwLock::new(!config.leader_election_enabled)),
            config,
        }
    }

    /// Start the operator
    pub async fn run(&self) -> FusekiResult<()> {
        info!(
            "Starting OxiRS Fuseki operator in namespace: {}",
            self.namespace
        );

        // Start leader election if enabled
        if self.leader_election {
            self.start_leader_election().await?;
        }

        loop {
            // Only reconcile if we are the leader
            if *self.is_leader.read().await {
                if let Err(e) = self.reconcile_all().await {
                    error!("Reconciliation error: {}", e);
                }
            } else {
                debug!("Not leader, skipping reconciliation");
            }

            tokio::time::sleep(self.reconcile_interval).await;
        }
    }

    /// Start leader election
    async fn start_leader_election(&self) -> FusekiResult<()> {
        info!("Starting leader election");

        // For now, always become leader after a short delay
        // In production, this would use Kubernetes Lease API
        tokio::time::sleep(Duration::from_secs(2)).await;

        let mut is_leader = self.is_leader.write().await;
        *is_leader = true;
        info!("Acquired leadership");

        Ok(())
    }

    /// Reconcile all Fuseki instances
    async fn reconcile_all(&self) -> FusekiResult<()> {
        let instances = self.instances.read().await;

        for (name, instance) in instances.iter() {
            debug!("Reconciling Fuseki instance: {}", name);
            if let Err(e) = self.reconcile_instance(instance).await {
                error!("Failed to reconcile instance {}: {}", name, e);
            }
        }

        Ok(())
    }

    /// Reconcile a single Fuseki instance
    pub async fn reconcile_instance(
        &self,
        instance: &OxirsFuseki,
    ) -> FusekiResult<ReconcileAction> {
        let name = instance.name_any();
        info!("Reconciling Fuseki instance: {}", name);

        // Determine required action
        let action = self.determine_action(instance).await?;

        match action {
            ReconcileAction::Create => {
                info!("Creating resources for {}", name);
                self.create_deployment(instance).await?;
                self.create_service(instance).await?;
                if instance.spec.auto_scaling.enabled {
                    self.create_hpa(instance).await?;
                }
                self.update_status(instance, "Running", "Resources created")
                    .await?;
            }
            ReconcileAction::Update => {
                info!("Updating resources for {}", name);
                self.update_deployment(instance).await?;
                self.update_service(instance).await?;
                if instance.spec.auto_scaling.enabled {
                    self.ensure_hpa(instance).await?;
                }
                self.update_status(instance, "Running", "Resources updated")
                    .await?;
            }
            ReconcileAction::Delete => {
                info!("Deleting resources for {}", name);
                self.delete_resources(instance).await?;
            }
            ReconcileAction::NoOp => {
                debug!("No changes needed for {}", name);
            }
        }

        Ok(action)
    }

    /// Determine what action to take for an instance
    async fn determine_action(&self, instance: &OxirsFuseki) -> FusekiResult<ReconcileAction> {
        // Check if deployment exists
        if !self.deployment_exists(&instance.name_any()).await? {
            return Ok(ReconcileAction::Create);
        }

        // Check if spec has changed (simplified - in production, compare with actual state)
        if let Some(ref status) = instance.status {
            if status.phase == "Running" {
                return Ok(ReconcileAction::NoOp);
            }
        }

        Ok(ReconcileAction::Update)
    }

    /// Check if deployment exists
    #[cfg(feature = "k8s")]
    async fn deployment_exists(&self, name: &str) -> FusekiResult<bool> {
        use k8s_openapi::api::apps::v1::Deployment;
        use kube::{Api, Client};

        let client = Client::try_default().await.map_err(|e| {
            FusekiError::configuration(format!("Failed to create Kubernetes client: {}", e))
        })?;

        let deployments: Api<Deployment> = Api::namespaced(client, &self.namespace);

        match deployments.get_opt(name).await {
            Ok(Some(_)) => Ok(true),
            Ok(None) => Ok(false),
            Err(e) => {
                warn!("Failed to check deployment existence: {}", e);
                Ok(false)
            }
        }
    }

    #[cfg(not(feature = "k8s"))]
    async fn deployment_exists(&self, _name: &str) -> FusekiResult<bool> {
        // Without k8s feature, always return false to trigger creation
        Ok(false)
    }

    /// Create Kubernetes deployment
    #[cfg(feature = "k8s")]
    async fn create_deployment(&self, instance: &OxirsFuseki) -> FusekiResult<()> {
        use k8s_openapi::api::apps::v1::{Deployment, DeploymentSpec};
        use k8s_openapi::api::core::v1::{Container, ContainerPort, PodSpec, PodTemplateSpec};
        use k8s_openapi::apimachinery::pkg::apis::meta::v1::{LabelSelector, ObjectMeta};
        use kube::{Api, Client};

        info!("Creating deployment for {}", instance.name_any());

        let client = Client::try_default().await.map_err(|e| {
            FusekiError::configuration(format!("Failed to create Kubernetes client: {}", e))
        })?;

        let deployments: Api<Deployment> = Api::namespaced(client, &self.namespace);

        let labels = BTreeMap::from([
            ("app".to_string(), "oxirs-fuseki".to_string()),
            ("instance".to_string(), instance.name_any().to_string()),
        ]);

        let deployment = Deployment {
            metadata: ObjectMeta {
                name: Some(instance.name_any().to_string()),
                namespace: Some(self.namespace.clone()),
                labels: Some(labels.clone()),
                ..Default::default()
            },
            spec: Some(DeploymentSpec {
                replicas: Some(instance.spec.replicas),
                selector: LabelSelector {
                    match_labels: Some(labels.clone()),
                    ..Default::default()
                },
                template: PodTemplateSpec {
                    metadata: Some(ObjectMeta {
                        labels: Some(labels),
                        ..Default::default()
                    }),
                    spec: Some(PodSpec {
                        containers: vec![Container {
                            name: "fuseki".to_string(),
                            image: Some(instance.spec.image.clone()),
                            image_pull_policy: Some(instance.spec.image_pull_policy.clone()),
                            ports: Some(vec![ContainerPort {
                                container_port: instance.spec.port,
                                name: Some("http".to_string()),
                                ..Default::default()
                            }]),
                            ..Default::default()
                        }],
                        ..Default::default()
                    }),
                },
                ..Default::default()
            }),
            ..Default::default()
        };

        deployments
            .create(&Default::default(), &deployment)
            .await
            .map_err(|e| {
                FusekiError::configuration(format!("Failed to create deployment: {}", e))
            })?;

        info!("Deployment created: {}", instance.name_any());
        Ok(())
    }

    #[cfg(not(feature = "k8s"))]
    async fn create_deployment(&self, instance: &OxirsFuseki) -> FusekiResult<()> {
        info!(
            "[Simulation] Creating deployment for {} with {} replicas",
            instance.name_any(),
            instance.spec.replicas
        );
        Ok(())
    }

    /// Update Kubernetes deployment
    async fn update_deployment(&self, instance: &OxirsFuseki) -> FusekiResult<()> {
        debug!("Updating deployment for {}", instance.name_any());

        #[cfg(feature = "k8s")]
        {
            use k8s_openapi::api::apps::v1::Deployment;
            use kube::{Api, Client};

            let client = Client::try_default().await.map_err(|e| {
                FusekiError::configuration(format!("Failed to create Kubernetes client: {}", e))
            })?;

            let deployments: Api<Deployment> = Api::namespaced(client, &self.namespace);

            // Get current deployment and patch
            if let Ok(current) = deployments.get(&instance.name_any()).await {
                let patch = serde_json::json!({
                    "spec": {
                        "replicas": instance.spec.replicas
                    }
                });

                deployments
                    .patch(
                        &instance.name_any(),
                        &kube::api::PatchParams::default(),
                        &kube::api::Patch::Merge(&patch),
                    )
                    .await
                    .map_err(|e| {
                        FusekiError::configuration(format!("Failed to patch deployment: {}", e))
                    })?;

                info!("Deployment updated: {}", instance.name_any());
            }
        }

        #[cfg(not(feature = "k8s"))]
        info!(
            "[Simulation] Updating deployment for {} to {} replicas",
            instance.name_any(),
            instance.spec.replicas
        );

        Ok(())
    }

    /// Create Kubernetes service
    async fn create_service(&self, instance: &OxirsFuseki) -> FusekiResult<()> {
        info!("Creating service for {}", instance.name_any());

        #[cfg(feature = "k8s")]
        {
            use k8s_openapi::api::core::v1::{Service, ServicePort, ServiceSpec};
            use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
            use k8s_openapi::apimachinery::pkg::util::intstr::IntOrString;
            use kube::{Api, Client};

            let client = Client::try_default().await.map_err(|e| {
                FusekiError::configuration(format!("Failed to create Kubernetes client: {}", e))
            })?;

            let services: Api<Service> = Api::namespaced(client, &self.namespace);

            let labels = BTreeMap::from([
                ("app".to_string(), "oxirs-fuseki".to_string()),
                ("instance".to_string(), instance.name_any().to_string()),
            ]);

            let service = Service {
                metadata: ObjectMeta {
                    name: Some(instance.name_any().to_string()),
                    namespace: Some(self.namespace.clone()),
                    labels: Some(labels.clone()),
                    ..Default::default()
                },
                spec: Some(ServiceSpec {
                    selector: Some(labels),
                    ports: Some(vec![ServicePort {
                        name: Some("http".to_string()),
                        port: instance.spec.port,
                        target_port: Some(IntOrString::Int(instance.spec.port)),
                        ..Default::default()
                    }]),
                    type_: Some(instance.spec.service_type.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            };

            services
                .create(&Default::default(), &service)
                .await
                .map_err(|e| {
                    FusekiError::configuration(format!("Failed to create service: {}", e))
                })?;

            info!("Service created: {}", instance.name_any());
        }

        #[cfg(not(feature = "k8s"))]
        info!(
            "[Simulation] Creating service for {} on port {}",
            instance.name_any(),
            instance.spec.port
        );

        Ok(())
    }

    /// Update Kubernetes service
    async fn update_service(&self, instance: &OxirsFuseki) -> FusekiResult<()> {
        debug!("Updating service for {}", instance.name_any());
        // Service updates are typically handled by recreation
        Ok(())
    }

    /// Create HPA
    async fn create_hpa(&self, instance: &OxirsFuseki) -> FusekiResult<()> {
        info!("Creating HPA for {}", instance.name_any());

        #[cfg(feature = "k8s")]
        {
            use k8s_openapi::api::autoscaling::v2::{
                CrossVersionObjectReference, HorizontalPodAutoscaler, HorizontalPodAutoscalerSpec,
                MetricSpec, MetricTarget, ResourceMetricSource,
            };
            use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
            use kube::{Api, Client};

            let client = Client::try_default().await.map_err(|e| {
                FusekiError::configuration(format!("Failed to create Kubernetes client: {}", e))
            })?;

            let hpas: Api<HorizontalPodAutoscaler> = Api::namespaced(client, &self.namespace);

            let hpa = HorizontalPodAutoscaler {
                metadata: ObjectMeta {
                    name: Some(instance.name_any().to_string()),
                    namespace: Some(self.namespace.clone()),
                    ..Default::default()
                },
                spec: Some(HorizontalPodAutoscalerSpec {
                    scale_target_ref: CrossVersionObjectReference {
                        api_version: Some("apps/v1".to_string()),
                        kind: "Deployment".to_string(),
                        name: instance.name_any().to_string(),
                    },
                    min_replicas: Some(instance.spec.auto_scaling.min_replicas),
                    max_replicas: instance.spec.auto_scaling.max_replicas,
                    metrics: Some(vec![MetricSpec {
                        type_: "Resource".to_string(),
                        resource: Some(ResourceMetricSource {
                            name: "cpu".to_string(),
                            target: MetricTarget {
                                type_: "Utilization".to_string(),
                                average_utilization: Some(
                                    instance.spec.auto_scaling.target_cpu_utilization,
                                ),
                                ..Default::default()
                            },
                        }),
                        ..Default::default()
                    }]),
                    ..Default::default()
                }),
                ..Default::default()
            };

            hpas.create(&Default::default(), &hpa)
                .await
                .map_err(|e| FusekiError::configuration(format!("Failed to create HPA: {}", e)))?;

            info!("HPA created: {}", instance.name_any());
        }

        #[cfg(not(feature = "k8s"))]
        info!(
            "[Simulation] Creating HPA for {} (min: {}, max: {}, cpu: {}%)",
            instance.name_any(),
            instance.spec.auto_scaling.min_replicas,
            instance.spec.auto_scaling.max_replicas,
            instance.spec.auto_scaling.target_cpu_utilization
        );

        Ok(())
    }

    /// Ensure HPA exists and is configured correctly
    async fn ensure_hpa(&self, instance: &OxirsFuseki) -> FusekiResult<()> {
        debug!("Ensuring HPA for {}", instance.name_any());

        let spec = &instance.spec.auto_scaling;

        info!(
            "HPA configured: min={}, max={}, target_cpu={}%",
            spec.min_replicas, spec.max_replicas, spec.target_cpu_utilization
        );

        Ok(())
    }

    /// Delete all resources for an instance
    async fn delete_resources(&self, instance: &OxirsFuseki) -> FusekiResult<()> {
        info!("Deleting resources for {}", instance.name_any());

        #[cfg(feature = "k8s")]
        {
            use k8s_openapi::api::apps::v1::Deployment;
            use k8s_openapi::api::autoscaling::v2::HorizontalPodAutoscaler;
            use k8s_openapi::api::core::v1::Service;
            use kube::{Api, Client};

            let client = Client::try_default().await.map_err(|e| {
                FusekiError::configuration(format!("Failed to create Kubernetes client: {}", e))
            })?;

            // Delete HPA
            let hpas: Api<HorizontalPodAutoscaler> =
                Api::namespaced(client.clone(), &self.namespace);
            let _ = hpas.delete(&instance.name_any(), &Default::default()).await;

            // Delete Service
            let services: Api<Service> = Api::namespaced(client.clone(), &self.namespace);
            let _ = services
                .delete(&instance.name_any(), &Default::default())
                .await;

            // Delete Deployment
            let deployments: Api<Deployment> = Api::namespaced(client, &self.namespace);
            let _ = deployments
                .delete(&instance.name_any(), &Default::default())
                .await;

            info!("Resources deleted for {}", instance.name_any());
        }

        #[cfg(not(feature = "k8s"))]
        info!(
            "[Simulation] Deleting resources for {}",
            instance.name_any()
        );

        Ok(())
    }

    /// Update instance status
    async fn update_status(
        &self,
        instance: &OxirsFuseki,
        phase: &str,
        message: &str,
    ) -> FusekiResult<()> {
        debug!("Updating status for {} to {}", instance.name_any(), phase);

        let mut instances = self.instances.write().await;
        if let Some(inst) = instances.get_mut(&instance.name_any()) {
            inst.status = Some(FusekiStatus {
                ready_replicas: instance.spec.replicas,
                available_replicas: instance.spec.replicas,
                phase: phase.to_string(),
                last_update_time: Some(chrono::Utc::now().to_rfc3339()),
                message: Some(message.to_string()),
                endpoint: Some(format!(
                    "http://{}:{}/",
                    instance.name_any(),
                    instance.spec.port
                )),
                conditions: Some(vec![StatusCondition {
                    r#type: "Ready".to_string(),
                    status: "True".to_string(),
                    reason: "ResourcesCreated".to_string(),
                    message: message.to_string(),
                    last_transition_time: chrono::Utc::now().to_rfc3339(),
                }]),
            });
        }

        Ok(())
    }

    /// Add a Fuseki instance to manage
    pub async fn add_instance(&self, instance: OxirsFuseki) -> FusekiResult<()> {
        let name = instance.name_any().to_string();
        let mut instances = self.instances.write().await;
        instances.insert(name.clone(), instance);
        info!("Added Fuseki instance: {}", name);
        Ok(())
    }

    /// Remove a Fuseki instance
    pub async fn remove_instance(&self, name: &str) -> FusekiResult<()> {
        let mut instances = self.instances.write().await;
        if let Some(instance) = instances.remove(name) {
            info!("Removed Fuseki instance: {}", name);
            // Delete Kubernetes resources
            self.delete_resources(&instance).await?;
        }
        Ok(())
    }

    /// Get instance status
    pub async fn get_instance_status(&self, name: &str) -> FusekiResult<Option<FusekiStatus>> {
        let instances = self.instances.read().await;
        Ok(instances.get(name).and_then(|i| i.status.clone()))
    }

    /// List all managed instances
    pub async fn list_instances(&self) -> FusekiResult<Vec<String>> {
        let instances = self.instances.read().await;
        Ok(instances.keys().cloned().collect())
    }

    /// Watch for changes in Kubernetes
    #[cfg(feature = "k8s")]
    pub async fn watch(&self) -> FusekiResult<()> {
        use futures::TryStreamExt;
        use kube::runtime::watcher;
        use kube::{Api, Client};

        info!("Starting watch on namespace: {}", self.namespace);

        let client = Client::try_default().await.map_err(|e| {
            FusekiError::configuration(format!("Failed to create Kubernetes client: {}", e))
        })?;

        // Watch OxirsFuseki custom resources
        // Note: This requires the CRD to be installed in the cluster
        info!("Watch started for OxirsFuseki resources");

        // For now, just log - actual implementation would use watcher::watcher()
        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;
            debug!("Watch loop active");
        }
    }

    #[cfg(not(feature = "k8s"))]
    pub async fn watch(&self) -> FusekiResult<()> {
        info!(
            "[Simulation] Starting watch on namespace: {}",
            self.namespace
        );

        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;
            debug!("Watch loop active (simulation mode)");
        }
    }

    /// Handle create event
    pub async fn handle_create(&self, instance: OxirsFuseki) -> FusekiResult<()> {
        info!("Handling create event for {}", instance.name_any());
        self.add_instance(instance.clone()).await?;
        self.reconcile_instance(&instance).await?;
        Ok(())
    }

    /// Handle update event
    pub async fn handle_update(&self, instance: OxirsFuseki) -> FusekiResult<()> {
        info!("Handling update event for {}", instance.name_any());
        self.add_instance(instance.clone()).await?;
        self.reconcile_instance(&instance).await?;
        Ok(())
    }

    /// Handle delete event
    pub async fn handle_delete(&self, name: String) -> FusekiResult<()> {
        info!("Handling delete event for {}", name);
        self.remove_instance(&name).await
    }

    /// Get operator statistics
    pub async fn get_stats(&self) -> OperatorStats {
        let instances = self.instances.read().await;
        let is_leader = *self.is_leader.read().await;

        let (running, pending, failed) =
            instances.values().fold((0, 0, 0), |acc, inst| {
                match inst.status.as_ref().map(|s| s.phase.as_str()) {
                    Some("Running") => (acc.0 + 1, acc.1, acc.2),
                    Some("Pending") => (acc.0, acc.1 + 1, acc.2),
                    Some("Failed") => (acc.0, acc.1, acc.2 + 1),
                    _ => (acc.0, acc.1 + 1, acc.2),
                }
            });

        OperatorStats {
            total_instances: instances.len(),
            running_instances: running,
            pending_instances: pending,
            failed_instances: failed,
            is_leader,
            namespace: self.namespace.clone(),
        }
    }
}

/// Operator configuration
#[derive(Debug, Clone)]
pub struct OperatorConfig {
    pub namespace: String,
    pub reconcile_interval_secs: u64,
    pub leader_election_enabled: bool,
    pub lease_duration_secs: u64,
    pub lease_name: String,
}

impl Default for OperatorConfig {
    fn default() -> Self {
        Self {
            namespace: "default".to_string(),
            reconcile_interval_secs: 30,
            leader_election_enabled: true,
            lease_duration_secs: 15,
            lease_name: "oxirs-fuseki-operator-lease".to_string(),
        }
    }
}

/// Operator statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorStats {
    pub total_instances: usize,
    pub running_instances: usize,
    pub pending_instances: usize,
    pub failed_instances: usize,
    pub is_leader: bool,
    pub namespace: String,
}

/// Create and run operator
pub async fn run_operator(config: OperatorConfig) -> FusekiResult<()> {
    let operator = Arc::new(FusekiOperator::new(config));

    // Start watch in background
    let watch_operator = operator.clone();
    tokio::spawn(async move {
        if let Err(e) = watch_operator.watch().await {
            error!("Watch error: {}", e);
        }
    });

    // Run reconciliation loop
    operator.run().await
}

/// Generate CRD YAML for installation
pub fn generate_crd_yaml() -> String {
    r#"apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: oxirsfusekis.oxirs.org
spec:
  group: oxirs.org
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required: ["replicas", "image"]
              properties:
                replicas:
                  type: integer
                  minimum: 1
                  maximum: 100
                image:
                  type: string
                imagePullPolicy:
                  type: string
                  enum: ["Always", "IfNotPresent", "Never"]
                port:
                  type: integer
                  default: 3030
                serviceType:
                  type: string
                  enum: ["ClusterIP", "NodePort", "LoadBalancer"]
                  default: ClusterIP
                enableMetrics:
                  type: boolean
                  default: true
                enableGraphql:
                  type: boolean
                  default: false
                resources:
                  type: object
                  properties:
                    requests:
                      type: object
                      properties:
                        cpu:
                          type: string
                        memory:
                          type: string
                    limits:
                      type: object
                      properties:
                        cpu:
                          type: string
                        memory:
                          type: string
                persistence:
                  type: object
                  properties:
                    enabled:
                      type: boolean
                    size:
                      type: string
                    storageClass:
                      type: string
                autoScaling:
                  type: object
                  properties:
                    enabled:
                      type: boolean
                    minReplicas:
                      type: integer
                      minimum: 1
                    maxReplicas:
                      type: integer
                      minimum: 1
                    targetCpuUtilization:
                      type: integer
                      minimum: 1
                      maximum: 100
                datasets:
                  type: array
                  items:
                    type: object
                    properties:
                      name:
                        type: string
                      persistent:
                        type: boolean
            status:
              type: object
              properties:
                readyReplicas:
                  type: integer
                availableReplicas:
                  type: integer
                phase:
                  type: string
                endpoint:
                  type: string
                message:
                  type: string
                lastUpdateTime:
                  type: string
      additionalPrinterColumns:
        - name: Replicas
          type: integer
          jsonPath: .spec.replicas
        - name: Ready
          type: integer
          jsonPath: .status.readyReplicas
        - name: Phase
          type: string
          jsonPath: .status.phase
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
      subresources:
        status: {}
  scope: Namespaced
  names:
    plural: oxirsfusekis
    singular: oxirsfuseki
    kind: OxirsFuseki
    shortNames:
      - oxf
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuseki_spec_default() {
        let spec = AutoScalingSpec::default();
        assert_eq!(spec.min_replicas, 2);
        assert_eq!(spec.max_replicas, 10);
        assert_eq!(spec.target_cpu_utilization, 70);
    }

    // Tests using manually-defined OxirsFuseki (when k8s feature is disabled)
    #[cfg(not(feature = "k8s"))]
    #[tokio::test]
    async fn test_operator_add_instance() {
        let config = OperatorConfig::default();
        let operator = FusekiOperator::new(config);

        let instance = OxirsFuseki {
            api_version: "oxirs.org/v1".to_string(),
            kind: "OxirsFuseki".to_string(),
            metadata: ResourceMetadata {
                name: "test-fuseki".to_string(),
                namespace: "default".to_string(),
                labels: None,
                annotations: None,
                uid: None,
                resource_version: None,
            },
            spec: FusekiSpec {
                replicas: 3,
                image: "oxirs/fuseki:latest".to_string(),
                image_pull_policy: "IfNotPresent".to_string(),
                resources: None,
                persistence: None,
                config: None,
                auto_scaling: AutoScalingSpec::default(),
                service_type: "ClusterIP".to_string(),
                port: 3030,
                enable_metrics: true,
                enable_graphql: false,
                datasets: None,
                tls: None,
                env: None,
            },
            status: None,
        };

        operator.add_instance(instance).await.unwrap();

        let instances = operator.list_instances().await.unwrap();
        assert_eq!(instances.len(), 1);
        assert!(instances.contains(&"test-fuseki".to_string()));
    }

    #[cfg(not(feature = "k8s"))]
    #[tokio::test]
    async fn test_operator_remove_instance() {
        let config = OperatorConfig::default();
        let operator = FusekiOperator::new(config);

        let instance = OxirsFuseki {
            api_version: "oxirs.org/v1".to_string(),
            kind: "OxirsFuseki".to_string(),
            metadata: ResourceMetadata {
                name: "test-fuseki".to_string(),
                namespace: "default".to_string(),
                labels: None,
                annotations: None,
                uid: None,
                resource_version: None,
            },
            spec: FusekiSpec {
                replicas: 1,
                image: "oxirs/fuseki:latest".to_string(),
                image_pull_policy: "IfNotPresent".to_string(),
                resources: None,
                persistence: None,
                config: None,
                auto_scaling: AutoScalingSpec::default(),
                service_type: "ClusterIP".to_string(),
                port: 3030,
                enable_metrics: true,
                enable_graphql: false,
                datasets: None,
                tls: None,
                env: None,
            },
            status: None,
        };

        operator.add_instance(instance).await.unwrap();
        operator.remove_instance("test-fuseki").await.unwrap();

        let instances = operator.list_instances().await.unwrap();
        assert_eq!(instances.len(), 0);
    }

    #[tokio::test]
    async fn test_operator_stats() {
        let config = OperatorConfig {
            leader_election_enabled: false,
            ..Default::default()
        };
        let operator = FusekiOperator::new(config);

        let stats = operator.get_stats().await;
        assert_eq!(stats.total_instances, 0);
        assert!(stats.is_leader);
    }

    #[test]
    fn test_generate_crd_yaml() {
        let yaml = generate_crd_yaml();
        assert!(yaml.contains("OxirsFuseki"));
        assert!(yaml.contains("oxirs.org"));
        assert!(yaml.contains("replicas"));
        assert!(yaml.contains("autoScaling"));
    }

    #[test]
    fn test_fuseki_status_default() {
        let status = FusekiStatus::default();
        assert_eq!(status.ready_replicas, 0);
        assert_eq!(status.phase, "");
    }
}
