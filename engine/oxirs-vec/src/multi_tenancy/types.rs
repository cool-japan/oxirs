//! Core types for multi-tenancy support

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Result type for multi-tenancy operations
pub type MultiTenancyResult<T> = Result<T, MultiTenancyError>;

/// Errors that can occur in multi-tenancy operations
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum MultiTenancyError {
    #[error("Tenant not found: {tenant_id}")]
    TenantNotFound { tenant_id: String },

    #[error("Tenant already exists: {tenant_id}")]
    TenantAlreadyExists { tenant_id: String },

    #[error("Quota exceeded for tenant {tenant_id}: {resource}")]
    QuotaExceeded { tenant_id: String, resource: String },

    #[error("Rate limit exceeded for tenant {tenant_id}")]
    RateLimitExceeded { tenant_id: String },

    #[error("Access denied for tenant {tenant_id}: {reason}")]
    AccessDenied { tenant_id: String, reason: String },

    #[error("Tenant is suspended: {tenant_id}")]
    TenantSuspended { tenant_id: String },

    #[error("Invalid tenant configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Isolation violation: {message}")]
    IsolationViolation { message: String },

    #[error("Billing error: {message}")]
    BillingError { message: String },

    #[error("Internal error: {message}")]
    InternalError { message: String },
}

/// Context for tenant operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantContext {
    /// Tenant identifier
    pub tenant_id: String,

    /// Request timestamp
    pub timestamp: DateTime<Utc>,

    /// Request metadata
    pub metadata: HashMap<String, String>,

    /// Authentication token (optional)
    pub auth_token: Option<String>,

    /// Client IP address (optional)
    pub client_ip: Option<String>,

    /// User agent (optional)
    pub user_agent: Option<String>,
}

impl TenantContext {
    /// Create a new tenant context
    pub fn new(tenant_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            auth_token: None,
            client_ip: None,
            user_agent: None,
        }
    }

    /// Add metadata to the context
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set authentication token
    pub fn with_auth_token(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    /// Set client IP
    pub fn with_client_ip(mut self, ip: impl Into<String>) -> Self {
        self.client_ip = Some(ip.into());
        self
    }

    /// Set user agent
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }
}

/// Type of tenant operation for billing and metering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TenantOperation {
    /// Vector insertion
    VectorInsert,

    /// Vector search/query
    VectorSearch,

    /// Vector update
    VectorUpdate,

    /// Vector deletion
    VectorDelete,

    /// Index build/rebuild
    IndexBuild,

    /// Batch operation
    BatchOperation,

    /// Embedding generation
    EmbeddingGeneration,

    /// Cross-encoder re-ranking
    Reranking,

    /// Custom operation
    Custom(u32),
}

impl TenantOperation {
    /// Get operation name
    pub fn name(&self) -> &'static str {
        match self {
            Self::VectorInsert => "vector_insert",
            Self::VectorSearch => "vector_search",
            Self::VectorUpdate => "vector_update",
            Self::VectorDelete => "vector_delete",
            Self::IndexBuild => "index_build",
            Self::BatchOperation => "batch_operation",
            Self::EmbeddingGeneration => "embedding_generation",
            Self::Reranking => "reranking",
            Self::Custom(_) => "custom",
        }
    }

    /// Get default cost weight for billing
    pub fn default_cost_weight(&self) -> f64 {
        match self {
            Self::VectorInsert => 1.0,
            Self::VectorSearch => 2.0,
            Self::VectorUpdate => 1.5,
            Self::VectorDelete => 0.5,
            Self::IndexBuild => 10.0,
            Self::BatchOperation => 5.0,
            Self::EmbeddingGeneration => 3.0,
            Self::Reranking => 4.0,
            Self::Custom(_) => 1.0,
        }
    }
}

/// Tenant statistics for monitoring and analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantStatistics {
    /// Tenant ID
    pub tenant_id: String,

    /// Total vectors stored
    pub total_vectors: usize,

    /// Total queries executed
    pub total_queries: u64,

    /// Average query latency (milliseconds)
    pub avg_query_latency_ms: f64,

    /// Peak queries per second
    pub peak_qps: f64,

    /// Current storage usage (bytes)
    pub storage_bytes: u64,

    /// Current memory usage (bytes)
    pub memory_bytes: u64,

    /// Total API calls
    pub api_calls: u64,

    /// Error count
    pub error_count: u64,

    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,

    /// Operation counts by type
    pub operation_counts: HashMap<String, u64>,

    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl TenantStatistics {
    /// Create new statistics for a tenant
    pub fn new(tenant_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            total_vectors: 0,
            total_queries: 0,
            avg_query_latency_ms: 0.0,
            peak_qps: 0.0,
            storage_bytes: 0,
            memory_bytes: 0,
            api_calls: 0,
            error_count: 0,
            last_activity: Utc::now(),
            operation_counts: HashMap::new(),
            custom_metrics: HashMap::new(),
        }
    }

    /// Record an operation
    pub fn record_operation(&mut self, operation: TenantOperation) {
        let op_name = operation.name().to_string();
        *self.operation_counts.entry(op_name).or_insert(0) += 1;
        self.api_calls += 1;
        self.last_activity = Utc::now();
    }

    /// Record a query with latency
    pub fn record_query(&mut self, latency_ms: f64) {
        self.total_queries += 1;
        self.record_operation(TenantOperation::VectorSearch);

        // Update average latency using exponential moving average
        let alpha = 0.1;
        self.avg_query_latency_ms = alpha * latency_ms + (1.0 - alpha) * self.avg_query_latency_ms;
    }

    /// Record an error
    pub fn record_error(&mut self) {
        self.error_count += 1;
        self.last_activity = Utc::now();
    }

    /// Update storage usage
    pub fn update_storage(&mut self, bytes: u64) {
        self.storage_bytes = bytes;
    }

    /// Update memory usage
    pub fn update_memory(&mut self, bytes: u64) {
        self.memory_bytes = bytes;
    }

    /// Set custom metric
    pub fn set_custom_metric(&mut self, key: impl Into<String>, value: f64) {
        self.custom_metrics.insert(key.into(), value);
    }

    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        if self.api_calls == 0 {
            0.0
        } else {
            self.error_count as f64 / self.api_calls as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_context_creation() {
        let ctx = TenantContext::new("tenant1")
            .with_metadata("region", "us-west")
            .with_auth_token("token123")
            .with_client_ip("192.168.1.1");

        assert_eq!(ctx.tenant_id, "tenant1");
        assert_eq!(ctx.metadata.get("region"), Some(&"us-west".to_string()));
        assert_eq!(ctx.auth_token, Some("token123".to_string()));
        assert_eq!(ctx.client_ip, Some("192.168.1.1".to_string()));
    }

    #[test]
    fn test_tenant_operation_cost_weights() {
        assert_eq!(TenantOperation::VectorInsert.default_cost_weight(), 1.0);
        assert_eq!(TenantOperation::VectorSearch.default_cost_weight(), 2.0);
        assert_eq!(TenantOperation::IndexBuild.default_cost_weight(), 10.0);
    }

    #[test]
    fn test_tenant_statistics() {
        let mut stats = TenantStatistics::new("tenant1");

        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.api_calls, 0);

        stats.record_query(100.0);
        assert_eq!(stats.total_queries, 1);
        assert_eq!(stats.api_calls, 1);
        assert!((stats.avg_query_latency_ms - 10.0).abs() < 1.0); // EMA starting from 0

        stats.record_operation(TenantOperation::VectorInsert);
        assert_eq!(stats.api_calls, 2);

        stats.record_error();
        assert_eq!(stats.error_count, 1);
        assert!((stats.error_rate() - 0.5).abs() < 0.01); // 1 error out of 2 calls
    }

    #[test]
    fn test_multitenance_error_display() {
        let error = MultiTenancyError::TenantNotFound {
            tenant_id: "tenant1".to_string(),
        };
        assert!(error.to_string().contains("tenant1"));

        let error = MultiTenancyError::QuotaExceeded {
            tenant_id: "tenant2".to_string(),
            resource: "storage".to_string(),
        };
        assert!(error.to_string().contains("tenant2"));
        assert!(error.to_string().contains("storage"));
    }
}
