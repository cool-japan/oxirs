//! # Security Traits and Primitives
//!
//! Authentication, authorization, audit, and threat detection traits,
//! plus supporting data types used throughout the security framework.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use super::{AuditEvent, AuthMethod, Permission, ThreatSeverity};

/// Authentication provider trait
#[async_trait::async_trait]
pub trait AuthenticationProvider: Send + Sync {
    async fn authenticate(&self, credentials: &Credentials) -> Result<super::SecurityContext>;
    async fn validate_token(&self, token: &str) -> Result<super::SecurityContext>;
    async fn refresh_token(&self, refresh_token: &str) -> Result<(String, String)>; // (access_token, refresh_token)
}

/// Authorization provider trait
#[async_trait::async_trait]
pub trait AuthorizationProvider: Send + Sync {
    async fn authorize(
        &self,
        context: &super::SecurityContext,
        resource: &str,
        action: &Permission,
    ) -> Result<bool>;
    async fn get_user_permissions(&self, user_id: &str) -> Result<HashSet<Permission>>;
    async fn get_user_roles(&self, user_id: &str) -> Result<HashSet<String>>;
}

/// Audit logger trait
#[async_trait::async_trait]
pub trait AuditLogger: Send + Sync {
    async fn log(&self, entry: AuditLogEntry) -> Result<()>;
    async fn query(&self, filter: AuditFilter) -> Result<Vec<AuditLogEntry>>;
}

/// Threat detector trait
#[async_trait::async_trait]
pub trait ThreatDetector: Send + Sync {
    async fn detect_threat(&self, context: &ThreatContext) -> Result<Option<ThreatAlert>>;
    async fn learn_pattern(&self, context: &ThreatContext) -> Result<()>;
}

/// Rate limiter trait
#[async_trait::async_trait]
pub trait RateLimiter: Send + Sync {
    async fn check_limit(&self, identifier: &str) -> Result<()>;
    async fn reset_limit(&self, identifier: &str) -> Result<()>;
}

/// Credentials for authentication
#[derive(Debug, Clone)]
pub enum Credentials {
    ApiKey {
        key: String,
        ip_address: Option<String>,
    },
    JWT {
        token: String,
        ip_address: Option<String>,
    },
    UserPassword {
        username: String,
        password: String,
        ip_address: Option<String>,
    },
    Certificate {
        cert: Vec<u8>,
        ip_address: Option<String>,
    },
}

impl Credentials {
    pub fn identifier(&self) -> String {
        match self {
            Credentials::ApiKey { key, .. } => format!("api_key:{key}"),
            Credentials::JWT { token, .. } => format!("jwt:{token}"),
            Credentials::UserPassword { username, .. } => format!("user:{username}"),
            Credentials::Certificate { .. } => "certificate".to_string(),
        }
    }

    pub fn user_id(&self) -> Option<&str> {
        match self {
            Credentials::UserPassword { username, .. } => Some(username),
            _ => None,
        }
    }

    pub fn ip_address(&self) -> Option<&str> {
        match self {
            Credentials::ApiKey { ip_address, .. }
            | Credentials::JWT { ip_address, .. }
            | Credentials::UserPassword { ip_address, .. }
            | Credentials::Certificate { ip_address, .. } => ip_address.as_deref(),
        }
    }
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub event_type: AuditEvent,
    pub timestamp: DateTime<Utc>,
    pub user_id: Option<String>,
    pub ip_address: Option<String>,
    pub success: bool,
    pub details: String,
}

/// Audit filter for queries
#[derive(Debug, Clone)]
pub struct AuditFilter {
    pub event_types: Option<Vec<AuditEvent>>,
    pub user_id: Option<String>,
    pub ip_address: Option<String>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub success: Option<bool>,
}

/// Threat context
#[derive(Debug, Clone)]
pub struct ThreatContext {
    pub event_type: String,
    pub user_id: Option<String>,
    pub ip_address: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub details: HashMap<String, String>,
}

/// Threat alert
#[derive(Debug, Clone)]
pub struct ThreatAlert {
    pub id: Uuid,
    pub rule_name: String,
    pub severity: ThreatSeverity,
    pub description: String,
    pub context: ThreatContext,
    pub timestamp: DateTime<Utc>,
    pub resolved: bool,
}

/// Security metrics
#[derive(Debug, Clone, Default)]
pub struct SecurityMetrics {
    pub authentication_attempts: u64,
    pub authentication_successes: u64,
    pub authentication_failures: u64,
    pub authorization_checks: u64,
    pub authorization_successes: u64,
    pub authorization_failures: u64,
    pub threat_alerts: u64,
    pub rate_limit_violations: u64,
    pub authentication_latency_ms: f64,
    pub authorization_latency_ms: f64,
}

/// Security context for requests
#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub roles: HashSet<String>,
    pub permissions: HashSet<Permission>,
    pub attributes: HashMap<String, String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub authentication_method: Option<AuthMethod>,
    pub authenticated_at: Option<DateTime<Utc>>,
}

impl SecurityContext {
    /// Check if user has permission
    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.permissions.contains(permission)
    }

    /// Check if user has role
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.contains(role)
    }

    /// Check if user has any of the specified roles
    pub fn has_any_role(&self, roles: &[String]) -> bool {
        roles.iter().any(|role| self.roles.contains(role))
    }

    /// Get attribute value
    pub fn get_attribute(&self, name: &str) -> Option<&String> {
        self.attributes.get(name)
    }
}
