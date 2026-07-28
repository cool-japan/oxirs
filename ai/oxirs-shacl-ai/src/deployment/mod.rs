//! Deployment Strategies and Infrastructure Management
//!
//! This module implements comprehensive deployment strategies including containerization,
//! auto-scaling, monitoring automation, and operational excellence patterns.

pub mod auto_scaling;
pub mod config;
pub mod containerization;
pub mod health;
pub mod load_balancing;
pub mod monitoring;
pub mod orchestration;
pub mod types;
pub mod updates;

use std::time::{Duration, Instant};

use crate::{Result, ShaclAiError};

// Re-export main types for convenience
pub use config::{DeploymentBackend, DeploymentConfig, DeploymentStrategy, EnvironmentType};
pub use types::{
    DeploymentInfo, DeploymentResult, DeploymentSpec, DeploymentStatistics, DeploymentStatus,
    ImageInfo, MonitoringUrl, OrchestrationResult, ScalingRequest, ScalingResult, ScalingType,
    ServiceEndpoint, UpdateResult, UpdateSpec,
};

use auto_scaling::AutoScalingEngine;
use containerization::ContainerizationEngine;
use health::HealthMonitor;
use load_balancing::LoadBalancingManager;
use monitoring::MonitoringAutomation;
use orchestration::OrchestrationEngine;
use updates::UpdateManager;

/// Deployment manager for SHACL-AI systems
#[derive(Debug)]
pub struct DeploymentManager {
    config: DeploymentConfig,
    containerization: ContainerizationEngine,
    orchestration: OrchestrationEngine,
    auto_scaling: AutoScalingEngine,
    monitoring: MonitoringAutomation,
    load_balancer: LoadBalancingManager,
    health_checker: HealthMonitor,
    update_manager: UpdateManager,
    statistics: DeploymentStatistics,
}

impl DeploymentManager {
    /// Create a new deployment manager
    pub fn new() -> Self {
        Self::with_config(DeploymentConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: DeploymentConfig) -> Self {
        Self {
            config,
            containerization: ContainerizationEngine::new(),
            orchestration: OrchestrationEngine::new(),
            auto_scaling: AutoScalingEngine::new(),
            monitoring: MonitoringAutomation::new(),
            load_balancer: LoadBalancingManager::new(),
            health_checker: HealthMonitor::new(),
            update_manager: UpdateManager::new(),
            statistics: DeploymentStatistics::default(),
        }
    }

    /// Deploy a SHACL-AI system against the configured orchestration backend.
    ///
    /// # Fail-loud contract
    ///
    /// This crate does not bundle a live orchestration backend (Kubernetes /
    /// Docker API clients are intentionally excluded to keep the default build
    /// Pure-Rust and free of network side effects). Therefore, with the default
    /// [`DeploymentBackend::None`], this method **never fabricates a successful
    /// deployment**: it validates the spec (returning
    /// [`ShaclAiError::Configuration`] for an invalid spec) and then returns
    /// [`ShaclAiError::Unsupported`], because nothing was actually provisioned.
    ///
    /// To obtain a deployment plan derived from the spec (without applying
    /// anything), call [`Self::plan_deployment`].
    pub async fn deploy_system(
        &mut self,
        deployment_spec: DeploymentSpec,
    ) -> Result<DeploymentResult> {
        tracing::info!("Starting SHACL-AI system deployment");

        // Validate deployment specification (fail loud on invalid input).
        self.validate_deployment_spec(&deployment_spec)?;

        // Dispatch on the configured backend. Because no live backend exists,
        // record the failed attempt and refuse to claim success.
        match self.config.deployment_backend {
            DeploymentBackend::None => {
                self.statistics.total_deployments += 1;
                self.statistics.failed_deployments += 1;
                tracing::warn!(
                    "deploy_system called but no live orchestration backend is configured"
                );
                Err(ShaclAiError::Unsupported(format!(
                    "no live orchestration backend is configured (deployment_backend = None); \
                     nothing was provisioned for '{}' v{}. Configure a real backend or call \
                     plan_deployment() for a dry-run plan.",
                    deployment_spec.name, deployment_spec.version
                )))
            }
        }
    }

    /// Produce a dry-run deployment plan derived from the spec.
    ///
    /// This applies nothing to any infrastructure. Every value in the returned
    /// [`DeploymentResult`] is computed deterministically from `deployment_spec`
    /// and this manager's configuration (status is
    /// [`DeploymentStatus::Planned`]), so it is safe to inspect what *would* be
    /// deployed without any network side effects.
    pub async fn plan_deployment(
        &self,
        deployment_spec: DeploymentSpec,
    ) -> Result<DeploymentResult> {
        let start_time = Instant::now();
        self.validate_deployment_spec(&deployment_spec)?;

        let image_info = if self.config.enable_containerization {
            Some(self.containerization.build_images(&deployment_spec).await?)
        } else {
            None
        };

        let orchestration_result = self.orchestration.setup_cluster(&deployment_spec).await?;

        let deployment_info = self
            .deploy_application(&deployment_spec, &orchestration_result)
            .await?;

        let endpoints = self.extract_service_endpoints(&deployment_spec);
        let monitoring_urls = self.get_monitoring_urls(&deployment_spec);

        Ok(DeploymentResult {
            deployment_id: format!("plan_{}", chrono::Utc::now().timestamp()),
            status: DeploymentStatus::Planned,
            deployment_time: start_time.elapsed(),
            image_info,
            orchestration_result,
            deployment_info,
            endpoints,
            monitoring_urls,
        })
    }

    /// Scale system based on metrics
    pub async fn scale_system(&mut self, scaling_request: ScalingRequest) -> Result<ScalingResult> {
        tracing::info!("Scaling SHACL-AI system: {:?}", scaling_request);

        let scaling_result = match scaling_request.scaling_type {
            ScalingType::HorizontalUp => {
                self.auto_scaling
                    .scale_horizontally_up(&scaling_request)
                    .await?
            }
            ScalingType::HorizontalDown => {
                self.auto_scaling
                    .scale_horizontally_down(&scaling_request)
                    .await?
            }
            ScalingType::VerticalUp => {
                self.auto_scaling
                    .scale_vertically_up(&scaling_request)
                    .await?
            }
            ScalingType::VerticalDown => {
                self.auto_scaling
                    .scale_vertically_down(&scaling_request)
                    .await?
            }
        };

        self.statistics.scaling_events += 1;
        if scaling_request.auto_triggered {
            self.statistics.auto_scaling_triggered += 1;
        }

        Ok(scaling_result)
    }

    /// Update deployment
    pub async fn update_deployment(&mut self, update_spec: UpdateSpec) -> Result<UpdateResult> {
        tracing::info!("Updating SHACL-AI deployment");

        let update_result = self.update_manager.perform_update(&update_spec).await?;

        if update_result.success {
            self.statistics.successful_deployments += 1;
        } else {
            self.statistics.failed_deployments += 1;
            if update_result.rollback_performed {
                self.statistics.rollbacks_performed += 1;
            }
        }

        Ok(update_result)
    }

    /// Get deployment statistics
    pub fn get_statistics(&self) -> &DeploymentStatistics {
        &self.statistics
    }

    // Private helper methods

    /// Validate a deployment specification. Returns
    /// [`ShaclAiError::Configuration`] for any structurally invalid spec so that
    /// bad input fails loudly rather than being silently accepted.
    fn validate_deployment_spec(&self, spec: &DeploymentSpec) -> Result<()> {
        if spec.name.trim().is_empty() {
            return Err(ShaclAiError::Configuration(
                "deployment name must not be empty".to_string(),
            ));
        }
        if spec.version.trim().is_empty() {
            return Err(ShaclAiError::Configuration(
                "deployment version must not be empty".to_string(),
            ));
        }
        if spec.replicas == 0 {
            return Err(ShaclAiError::Configuration(
                "deployment replicas must be >= 1".to_string(),
            ));
        }

        // Resource requests, when present, must be parseable Kubernetes quantities.
        if let Some(cpu) = spec.resources.cpu.as_deref() {
            parse_cpu_millicores(cpu).ok_or_else(|| {
                ShaclAiError::Configuration(format!("invalid CPU resource quantity: '{cpu}'"))
            })?;
        }
        if let Some(memory) = spec.resources.memory.as_deref() {
            parse_memory_bytes(memory).ok_or_else(|| {
                ShaclAiError::Configuration(format!("invalid memory resource quantity: '{memory}'"))
            })?;
        }

        // Networking ports must be non-zero and internally consistent.
        for port in &spec.networking.ports {
            if port.port == 0 || port.target_port == 0 {
                return Err(ShaclAiError::Configuration(format!(
                    "service port '{}' has an invalid port/target_port of 0",
                    port.name
                )));
            }
        }
        if let Some(ingress) = &spec.networking.ingress {
            if ingress.host.trim().is_empty() {
                return Err(ShaclAiError::Configuration(
                    "ingress host must not be empty when ingress is configured".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Derive the planned application layout from the spec (dry-run only).
    async fn deploy_application(
        &self,
        spec: &DeploymentSpec,
        orchestration: &OrchestrationResult,
    ) -> Result<DeploymentInfo> {
        // Service names are derived from the declared network ports; fall back
        // to the deployment name itself when no ports are declared.
        let services: Vec<String> = if spec.networking.ports.is_empty() {
            vec![spec.name.clone()]
        } else {
            spec.networking
                .ports
                .iter()
                .map(|p| format!("{}-{}", spec.name, p.name))
                .collect()
        };

        let pods: Vec<String> = (0..spec.replicas)
            .map(|i| format!("{}-pod-{i}", spec.name))
            .collect();

        Ok(DeploymentInfo {
            deployment_id: format!("{}-{}", spec.name, spec.version),
            namespace: orchestration.namespace.clone(),
            services,
            pods,
            replicas: spec.replicas,
        })
    }

    /// Derive the service endpoints that the spec would expose (dry-run only).
    ///
    /// Endpoints are built from the spec's own networking declaration — the
    /// ingress host when present, otherwise an in-cluster DNS name — never from
    /// hardcoded placeholder domains.
    fn extract_service_endpoints(&self, spec: &DeploymentSpec) -> Vec<ServiceEndpoint> {
        let ingress = spec.networking.ingress.as_ref();
        spec.networking
            .ports
            .iter()
            .map(|port| {
                let (host, scheme, path) = match ingress {
                    Some(ing) => (
                        ing.host.clone(),
                        if ing.tls_enabled { "https" } else { "http" },
                        ing.path.clone(),
                    ),
                    None => (
                        // In-cluster service DNS name (Kubernetes convention).
                        format!(
                            "{}-{}.{}.svc.cluster.local",
                            spec.name, port.name, spec.name
                        ),
                        "http",
                        "/".to_string(),
                    ),
                };
                ServiceEndpoint {
                    service_name: format!("{}-{}", spec.name, port.name),
                    endpoint_url: format!("{scheme}://{host}:{}{path}", port.port),
                    port: port.port,
                    protocol: format!("{:?}", port.protocol),
                }
            })
            .collect()
    }

    /// Derive monitoring URLs from the spec (dry-run only).
    ///
    /// Only returns URLs that can be honestly derived from the spec's ingress
    /// host. If monitoring is disabled or no ingress is declared, returns an
    /// empty vector rather than inventing placeholder dashboards.
    fn get_monitoring_urls(&self, spec: &DeploymentSpec) -> Vec<MonitoringUrl> {
        if !self.config.enable_health_monitoring {
            return Vec::new();
        }
        match spec.networking.ingress.as_ref() {
            Some(ingress) => {
                let scheme = if ingress.tls_enabled { "https" } else { "http" };
                vec![
                    MonitoringUrl {
                        service_name: "Metrics".to_string(),
                        url: format!("{scheme}://{}/metrics", ingress.host),
                    },
                    MonitoringUrl {
                        service_name: "Health".to_string(),
                        url: format!("{scheme}://{}/health", ingress.host),
                    },
                ]
            }
            None => Vec::new(),
        }
    }
}

/// Parse a Kubernetes CPU quantity (e.g. `"500m"`, `"2"`, `"1.5"`) into
/// millicores. Returns `None` for unparseable input.
fn parse_cpu_millicores(value: &str) -> Option<u64> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    if let Some(millis) = value.strip_suffix('m') {
        millis.trim().parse::<u64>().ok()
    } else {
        let cores: f64 = value.parse().ok()?;
        if cores < 0.0 || !cores.is_finite() {
            return None;
        }
        Some((cores * 1000.0).round() as u64)
    }
}

/// Parse a Kubernetes memory quantity (e.g. `"512Mi"`, `"2Gi"`, `"1000000"`)
/// into bytes. Returns `None` for unparseable input.
fn parse_memory_bytes(value: &str) -> Option<u64> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    // Binary (power-of-two) suffixes.
    const BINARY: &[(&str, u64)] = &[
        ("Ki", 1 << 10),
        ("Mi", 1 << 20),
        ("Gi", 1 << 30),
        ("Ti", 1u64 << 40),
        ("Pi", 1u64 << 50),
    ];
    // Decimal (power-of-ten) suffixes.
    const DECIMAL: &[(&str, u64)] = &[
        ("k", 1_000),
        ("M", 1_000_000),
        ("G", 1_000_000_000),
        ("T", 1_000_000_000_000),
        ("P", 1_000_000_000_000_000),
    ];
    for (suffix, multiplier) in BINARY {
        if let Some(num) = value.strip_suffix(suffix) {
            let base: u64 = num.trim().parse().ok()?;
            return base.checked_mul(*multiplier);
        }
    }
    for (suffix, multiplier) in DECIMAL {
        if let Some(num) = value.strip_suffix(suffix) {
            let base: u64 = num.trim().parse().ok()?;
            return base.checked_mul(*multiplier);
        }
    }
    value.parse::<u64>().ok()
}

impl Default for DeploymentManager {
    fn default() -> Self {
        Self::new()
    }
}

// Dry-run planning helpers. These derive the *intended* configuration from the
// spec; they do not build images or provision clusters (that requires a live
// backend, which `deploy_system` gates on).
impl ContainerizationEngine {
    /// Derive the planned image reference from the spec. No image is built.
    async fn build_images(&self, spec: &DeploymentSpec) -> Result<ImageInfo> {
        Ok(ImageInfo {
            // Image reference derived from the deployment name + version.
            image_tag: format!("{}:{}", spec.name, spec.version),
            // Nothing was built, so the size/build-time are honestly zero.
            image_size: 0,
            build_time: Duration::ZERO,
            vulnerabilities: vec![],
        })
    }
}

impl OrchestrationEngine {
    /// Derive the planned cluster layout from the spec. No cluster is created.
    async fn setup_cluster(&self, spec: &DeploymentSpec) -> Result<OrchestrationResult> {
        // Namespace is derived from the environment so plans in different
        // environments do not collide.
        let namespace = format!("{}-{}", spec.name, environment_slug(&spec.environment));
        // Plan enough nodes to host the requested replicas, assuming a
        // conservative packing of 4 pods/node with a 1-node floor.
        let node_count = spec.replicas.div_ceil(4).max(1);
        Ok(OrchestrationResult {
            cluster_name: format!("{}-cluster", spec.name),
            namespace,
            node_count,
            // Nothing was provisioned, so setup time is honestly zero.
            setup_time: Duration::ZERO,
        })
    }
}

impl AutoScalingEngine {
    async fn scale_horizontally_up(&self, _request: &ScalingRequest) -> Result<ScalingResult> {
        // Placeholder implementation
        Ok(ScalingResult {
            success: true,
            previous_replicas: 2,
            new_replicas: 4,
            scaling_time: std::time::Duration::from_secs(60),
            resource_changes: None,
        })
    }

    async fn scale_horizontally_down(&self, _request: &ScalingRequest) -> Result<ScalingResult> {
        // Placeholder implementation
        Ok(ScalingResult {
            success: true,
            previous_replicas: 4,
            new_replicas: 2,
            scaling_time: std::time::Duration::from_secs(30),
            resource_changes: None,
        })
    }

    async fn scale_vertically_up(&self, _request: &ScalingRequest) -> Result<ScalingResult> {
        // Placeholder implementation
        Ok(ScalingResult {
            success: true,
            previous_replicas: 2,
            new_replicas: 2,
            scaling_time: std::time::Duration::from_secs(120),
            resource_changes: Some(types::ResourceRequirements {
                cpu: Some("2000m".to_string()),
                memory: Some("4Gi".to_string()),
            }),
        })
    }

    async fn scale_vertically_down(&self, _request: &ScalingRequest) -> Result<ScalingResult> {
        // Placeholder implementation
        Ok(ScalingResult {
            success: true,
            previous_replicas: 2,
            new_replicas: 2,
            scaling_time: std::time::Duration::from_secs(90),
            resource_changes: Some(types::ResourceRequirements {
                cpu: Some("1000m".to_string()),
                memory: Some("2Gi".to_string()),
            }),
        })
    }
}

/// Return a short, DNS-safe slug for an environment, used to derive namespaces.
fn environment_slug(environment: &EnvironmentType) -> &'static str {
    match environment {
        EnvironmentType::Development => "dev",
        EnvironmentType::Testing => "test",
        EnvironmentType::Staging => "staging",
        EnvironmentType::Production => "prod",
        EnvironmentType::DisasterRecovery => "dr",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deployment_manager_creation() {
        let manager = DeploymentManager::new();
        assert!(manager.config.enable_containerization);
        assert!(manager.config.enable_auto_scaling);
        assert!(manager.config.enable_load_balancing);
    }

    #[test]
    fn test_deployment_config() {
        let config = DeploymentConfig::default();
        assert!(config.enable_health_monitoring);
        assert!(matches!(
            config.deployment_strategy,
            DeploymentStrategy::BlueGreen
        ));
        assert!(matches!(config.environment, EnvironmentType::Production));
    }

    #[test]
    fn test_auto_scaling_config() {
        let config = config::AutoScalingConfig::default();
        assert_eq!(config.min_instances, 2);
        assert_eq!(config.max_instances, 10);
        assert_eq!(config.target_cpu_utilization, 0.7);
    }

    #[test]
    fn test_resource_limits() {
        let limits = config::ResourceLimits::default();
        assert_eq!(limits.cpu_limit, 4.0);
        assert_eq!(limits.memory_limit_mb, 8192);
        assert_eq!(limits.max_concurrent_validations, 1000);
    }

    fn valid_spec() -> DeploymentSpec {
        DeploymentSpec {
            name: "shacl-ai".to_string(),
            version: "0.4.1".to_string(),
            environment: EnvironmentType::Production,
            resources: types::ResourceRequirements {
                cpu: Some("500m".to_string()),
                memory: Some("512Mi".to_string()),
            },
            replicas: 3,
            configuration: std::collections::HashMap::new(),
            volumes: vec![],
            networking: types::NetworkingSpec {
                service_type: types::ServiceType::ClusterIP,
                ports: vec![types::ServicePort {
                    name: "http".to_string(),
                    port: 8080,
                    target_port: 8080,
                    protocol: containerization::Protocol::TCP,
                }],
                ingress: None,
            },
        }
    }

    /// Regression: with the default (no) backend, `deploy_system` must fail
    /// loudly instead of fabricating a successful deployment.
    #[tokio::test]
    async fn regression_deploy_system_fails_loud_without_backend() {
        let mut manager = DeploymentManager::new();
        let result = manager.deploy_system(valid_spec()).await;
        assert!(
            matches!(result, Err(ShaclAiError::Unsupported(_))),
            "expected Unsupported, got {result:?}"
        );
        assert_eq!(manager.get_statistics().failed_deployments, 1);
        assert_eq!(manager.get_statistics().successful_deployments, 0);
    }

    /// Regression: an invalid spec must be rejected up-front.
    #[tokio::test]
    async fn regression_deploy_system_rejects_invalid_spec() {
        let mut manager = DeploymentManager::new();
        let mut spec = valid_spec();
        spec.replicas = 0;
        let result = manager.deploy_system(spec).await;
        assert!(matches!(result, Err(ShaclAiError::Configuration(_))));

        let mut spec = valid_spec();
        spec.resources.memory = Some("not-a-quantity".to_string());
        let result = manager.deploy_system(spec).await;
        assert!(matches!(result, Err(ShaclAiError::Configuration(_))));
    }

    /// Regression: `plan_deployment` yields an honest, spec-derived plan whose
    /// endpoints come from the spec (not hardcoded example.com domains).
    #[tokio::test]
    async fn regression_plan_deployment_is_spec_derived() {
        let manager = DeploymentManager::new();
        let plan = manager
            .plan_deployment(valid_spec())
            .await
            .expect("planning should succeed");
        assert!(matches!(plan.status, DeploymentStatus::Planned));
        assert_eq!(plan.deployment_info.replicas, 3);
        assert_eq!(plan.deployment_info.pods.len(), 3);
        // No fabricated public endpoints.
        for endpoint in &plan.endpoints {
            assert!(
                !endpoint.endpoint_url.contains("example.com"),
                "endpoint must be spec-derived, got {}",
                endpoint.endpoint_url
            );
        }
        // Image tag derived from name:version, size honestly zero (not built).
        let image = plan.image_info.expect("image plan present");
        assert_eq!(image.image_tag, "shacl-ai:0.4.1");
        assert_eq!(image.image_size, 0);
    }

    #[test]
    fn regression_parse_quantities() {
        assert_eq!(parse_cpu_millicores("500m"), Some(500));
        assert_eq!(parse_cpu_millicores("2"), Some(2000));
        assert_eq!(parse_cpu_millicores("bad"), None);
        assert_eq!(parse_memory_bytes("1Ki"), Some(1024));
        assert_eq!(parse_memory_bytes("2Gi"), Some(2 * (1 << 30)));
        assert_eq!(parse_memory_bytes("1000000"), Some(1_000_000));
        assert_eq!(parse_memory_bytes("bad"), None);
    }
}
