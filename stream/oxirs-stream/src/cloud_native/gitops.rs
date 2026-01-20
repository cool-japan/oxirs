//! GitOps Configuration Types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
