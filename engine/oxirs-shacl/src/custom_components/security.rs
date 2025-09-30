//! Security policies and violation handling for custom constraint components
//!
//! This module provides comprehensive security framework for custom components,
//! including sandboxing, resource quotas, and violation detection.

use crate::Severity;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Duration;

/// Security policy for components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    /// Whether component can execute arbitrary SPARQL
    pub allow_arbitrary_sparql: bool,
    /// Whether component can access external resources
    pub allow_external_access: bool,
    /// Maximum execution time allowed
    pub max_execution_time: Option<Duration>,
    /// Maximum memory usage allowed
    pub max_memory_usage: Option<usize>,
    /// Allowed SPARQL operations
    pub allowed_sparql_operations: HashSet<SparqlOperation>,
    /// Trusted component flag
    pub trusted: bool,
    /// Sandboxing level
    pub sandboxing_level: SandboxingLevel,
    /// Resource quotas
    pub resource_quotas: ResourceQuotas,
}

/// SPARQL operations enumeration
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparqlOperation {
    Select,
    Ask,
    Construct,
    Describe,
    Insert,
    Delete,
    Update,
    Service,
}

/// Sandboxing levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SandboxingLevel {
    None,
    Basic,
    Strict,
    Isolation,
}

/// Resource quotas for components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuotas {
    /// Maximum CPU time per execution
    pub max_cpu_time: Option<Duration>,
    /// Maximum number of SPARQL queries per execution
    pub max_sparql_queries: Option<u32>,
    /// Maximum result set size
    pub max_result_size: Option<usize>,
    /// Maximum recursion depth
    pub max_recursion_depth: Option<u32>,
}

/// Security violation
#[derive(Debug, Clone)]
pub struct SecurityViolation {
    /// Violation type
    pub violation_type: SecurityViolationType,
    /// Violation description
    pub description: String,
    /// Severity level
    pub severity: Severity,
}

/// Security violation types
#[derive(Debug, Clone)]
pub enum SecurityViolationType {
    UnauthorizedSparqlOperation,
    ExternalResourceAccess,
    ExecutionTimeExceeded,
    MemoryLimitExceeded,
    RecursionLimitExceeded,
    UntrustedComponentExecution,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            allow_arbitrary_sparql: false,
            allow_external_access: false,
            max_execution_time: Some(Duration::from_secs(30)),
            max_memory_usage: Some(100 * 1024 * 1024), // 100MB
            allowed_sparql_operations: [SparqlOperation::Ask, SparqlOperation::Select]
                .iter()
                .cloned()
                .collect(),
            trusted: false,
            sandboxing_level: SandboxingLevel::Basic,
            resource_quotas: ResourceQuotas {
                max_cpu_time: Some(Duration::from_secs(10)),
                max_sparql_queries: Some(10),
                max_result_size: Some(10000),
                max_recursion_depth: Some(10),
            },
        }
    }
}
