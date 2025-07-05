//! Secure SPARQL Executor with Advanced Security Features
//!
//! This module provides a comprehensive secure SPARQL execution environment that integrates
//! all security features including analysis, rewriting, sandboxing, and policy enforcement.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use oxirs_core::Store;

use crate::{
    constraints::ConstraintContext,
    security::{
        advanced::{
            SecurityPolicyManager, SecurityContext, ExecutionConstraints, SecureExecutionResult,
            Permission, SecurityConstraints,
        },
        SecurityConfig, SparqlSecurityAnalyzer, QueryExecutionSandbox,
    },
    sparql::{SparqlConstraint, EnhancedSparqlExecutor},
    Result, ShaclError,
};

/// Comprehensive secure SPARQL executor with enterprise-grade security
#[derive(Debug)]
pub struct SecureSparqlExecutor {
    /// Advanced security policy manager
    policy_manager: SecurityPolicyManager,
    
    /// Basic security analyzer (legacy compatibility)
    security_analyzer: SparqlSecurityAnalyzer,
    
    /// Enhanced SPARQL executor for function support
    sparql_executor: EnhancedSparqlExecutor,
    
    /// Execution statistics
    execution_stats: ExecutionStatistics,
    
    /// Security configuration
    security_config: SecurityConfig,
}

impl SecureSparqlExecutor {
    /// Create a new secure SPARQL executor
    pub fn new(security_config: SecurityConfig) -> Result<Self> {
        let policy_manager = SecurityPolicyManager::new()?;
        let security_analyzer = SparqlSecurityAnalyzer::new(security_config.clone())?;
        let sparql_executor = EnhancedSparqlExecutor::new();

        Ok(Self {
            policy_manager,
            security_analyzer,
            sparql_executor,
            execution_stats: ExecutionStatistics::new(),
            security_config,
        })
    }

    /// Execute a SPARQL constraint with full security checks and monitoring
    pub fn execute_secure_constraint(
        &mut self,
        constraint: &SparqlConstraint,
        context: &ConstraintContext,
        store: &dyn Store,
        user_context: &str,
        execution_constraints: Option<ExecutionConstraints>,
    ) -> Result<SecureConstraintResult> {
        let start_time = Instant::now();

        // 1. Basic validation
        constraint.validate()?;

        // 2. Execute with full security pipeline
        let execution_result = self.policy_manager.execute_secure_sparql(
            &constraint.query,
            user_context,
            execution_constraints,
        )?;

        // 3. Convert to constraint evaluation result
        let constraint_result = self.convert_to_constraint_result(
            &execution_result,
            constraint,
            context,
            store,
        )?;

        // 4. Update execution statistics
        let total_time = start_time.elapsed();
        self.execution_stats.record_execution(total_time, execution_result.success);

        let exec_time = constraint_result.execution_time;
        Ok(SecureConstraintResult {
            constraint_evaluation: constraint_result,
            security_execution: execution_result,
            total_execution_time: total_time,
            security_overhead: total_time.saturating_sub(exec_time),
        })
    }

    /// Execute a SPARQL query with comprehensive security (public API)
    pub fn execute_secure_query(
        &mut self,
        query: &str,
        user_id: &str,
        permissions: Vec<Permission>,
        constraints: Option<SecurityConstraints>,
        execution_constraints: Option<ExecutionConstraints>,
    ) -> Result<SecureQueryResult> {
        let start_time = Instant::now();

        // Create or get security context
        let security_constraints = constraints.unwrap_or_default();
        let context_id = self.policy_manager.create_security_context(
            user_id,
            permissions,
            security_constraints,
        )?;

        // Execute with security
        let execution_result = self.policy_manager.execute_secure_sparql(
            query,
            &context_id,
            execution_constraints,
        )?;

        // Update statistics
        let total_time = start_time.elapsed();
        self.execution_stats.record_execution(total_time, execution_result.success);

        Ok(SecureQueryResult {
            success: execution_result.success,
            query_result: execution_result.result_data,
            security_analysis: execution_result.security_analysis,
            policy_result: execution_result.policy_result,
            injection_analysis: execution_result.injection_analysis,
            original_query: execution_result.original_query,
            rewritten_query: execution_result.rewritten_query,
            execution_time: execution_result.execution_time,
            context_id,
        })
    }

    /// Execute with legacy sandbox (for backward compatibility)
    pub fn execute_with_sandbox(
        &mut self,
        constraint: &SparqlConstraint,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<LegacyConstraintResult> {
        // Basic security analysis
        let analysis = self.security_analyzer.analyze_query(&constraint.query)?;
        
        if !analysis.is_safe {
            return Err(ShaclError::SecurityViolation(format!(
                "Query failed basic security analysis: {} violations",
                analysis.violations.len()
            )));
        }

        // Create execution sandbox
        let mut sandbox = QueryExecutionSandbox::new(self.security_config.clone());
        sandbox.start_execution()?;

        // Execute constraint using enhanced executor
        let result = self.sparql_executor.execute_constraint_enhanced(
            constraint,
            context,
            store,
        )?;

        // Check sandbox limits
        sandbox.check_execution_limits()?;
        
        // Stop sandbox and get stats
        let execution_stats = sandbox.stop_execution()?;

        Ok(LegacyConstraintResult {
            constraint_result: result,
            security_analysis: analysis,
            execution_stats,
        })
    }

    /// Get comprehensive security metrics
    pub fn get_security_metrics(&self) -> ComprehensiveSecurityMetrics {
        let policy_metrics = self.policy_manager.get_security_metrics();
        let execution_metrics = self.execution_stats.get_metrics();

        let total_queries = execution_metrics.total_executions;
        let avg_overhead = execution_metrics.average_security_overhead;
        let success_rate = execution_metrics.success_rate;
        
        ComprehensiveSecurityMetrics {
            policy_metrics,
            execution_metrics,
            total_queries_analyzed: total_queries,
            average_security_overhead: avg_overhead,
            security_success_rate: success_rate,
        }
    }

    /// Install a security policy
    pub fn install_security_policy(
        &mut self,
        policy: crate::security::advanced::SecurityPolicy,
    ) -> Result<()> {
        self.policy_manager.install_policy(policy)
    }

    /// Get audit events with filtering
    pub fn get_audit_events(
        &self,
        filter: Option<crate::security::advanced::SecurityEventFilter>,
    ) -> Vec<crate::security::advanced::SecurityEvent> {
        self.policy_manager.audit_logger.get_events(filter)
    }

    /// Update global security configuration
    pub fn update_security_config(&mut self, config: SecurityConfig) -> Result<()> {
        self.security_config = config.clone();
        self.security_analyzer = SparqlSecurityAnalyzer::new(config)?;
        Ok(())
    }

    /// Convert execution result to constraint evaluation result
    fn convert_to_constraint_result(
        &self,
        execution_result: &SecureExecutionResult,
        constraint: &SparqlConstraint,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        // In a full implementation, this would interpret the SPARQL results
        // and convert them to constraint evaluation results
        
        if execution_result.success {
            Ok(ConstraintEvaluationResult {
                satisfied: true,
                violations: Vec::new(),
                execution_time: execution_result.execution_time,
                memory_used: execution_result.result_data.memory_used,
                query_rewritten: execution_result.original_query != execution_result.rewritten_query,
            })
        } else {
            Ok(ConstraintEvaluationResult {
                satisfied: false,
                violations: vec![ConstraintViolationDetails {
                    focus_node: context.focus_node.clone(),
                    constraint_component: constraint.clone(),
                    message: "Security violation prevented execution".to_string(),
                    details: HashMap::new(),
                }],
                execution_time: execution_result.execution_time,
                memory_used: execution_result.result_data.memory_used,
                query_rewritten: true,
            })
        }
    }
}

/// Execution statistics tracker
#[derive(Debug)]
struct ExecutionStatistics {
    total_executions: u64,
    successful_executions: u64,
    total_time: Duration,
    total_security_overhead: Duration,
    start_time: Instant,
}

impl ExecutionStatistics {
    fn new() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            total_time: Duration::default(),
            total_security_overhead: Duration::default(),
            start_time: Instant::now(),
        }
    }

    fn record_execution(&mut self, execution_time: Duration, success: bool) {
        self.total_executions += 1;
        self.total_time += execution_time;
        
        if success {
            self.successful_executions += 1;
        }
    }

    fn get_metrics(&self) -> ExecutionMetrics {
        let uptime = self.start_time.elapsed();
        let success_rate = if self.total_executions > 0 {
            (self.successful_executions as f64) / (self.total_executions as f64)
        } else {
            0.0
        };

        let average_execution_time = if self.total_executions > 0 {
            self.total_time / self.total_executions as u32
        } else {
            Duration::default()
        };

        let average_security_overhead = if self.total_executions > 0 {
            self.total_security_overhead / self.total_executions as u32
        } else {
            Duration::default()
        };

        ExecutionMetrics {
            total_executions: self.total_executions,
            successful_executions: self.successful_executions,
            success_rate,
            average_execution_time,
            average_security_overhead,
            uptime,
        }
    }
}

/// Result types for secure execution
#[derive(Debug)]
pub struct SecureConstraintResult {
    pub constraint_evaluation: ConstraintEvaluationResult,
    pub security_execution: SecureExecutionResult,
    pub total_execution_time: Duration,
    pub security_overhead: Duration,
}

#[derive(Debug)]
pub struct SecureQueryResult {
    pub success: bool,
    pub query_result: crate::security::advanced::QueryResult,
    pub security_analysis: crate::security::SecurityAnalysisResult,
    pub policy_result: crate::security::advanced::PolicyAuthorizationResult,
    pub injection_analysis: crate::security::advanced::InjectionAnalysisResult,
    pub original_query: String,
    pub rewritten_query: String,
    pub execution_time: Duration,
    pub context_id: String,
}

#[derive(Debug)]
pub struct LegacyConstraintResult {
    pub constraint_result: crate::constraints::ConstraintEvaluationResult,
    pub security_analysis: crate::security::SecurityAnalysisResult,
    pub execution_stats: crate::security::ExecutionStats,
}

#[derive(Debug, Clone)]
pub struct ConstraintEvaluationResult {
    pub satisfied: bool,
    pub violations: Vec<ConstraintViolationDetails>,
    pub execution_time: Duration,
    pub memory_used: usize,
    pub query_rewritten: bool,
}

#[derive(Debug, Clone)]
pub struct ConstraintViolationDetails {
    pub focus_node: oxirs_core::model::Term,
    pub constraint_component: SparqlConstraint,
    pub message: String,
    pub details: HashMap<String, String>,
}

/// Comprehensive security metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct ComprehensiveSecurityMetrics {
    pub policy_metrics: crate::security::advanced::SecurityMetrics,
    pub execution_metrics: ExecutionMetrics,
    pub total_queries_analyzed: u64,
    pub average_security_overhead: Duration,
    pub security_success_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub success_rate: f64,
    pub average_execution_time: Duration,
    pub average_security_overhead: Duration,
    pub uptime: Duration,
}

/// Factory for creating secure executors with different configurations
pub struct SecureExecutorFactory;

impl SecureExecutorFactory {
    /// Create executor with default security settings
    pub fn create_default() -> Result<SecureSparqlExecutor> {
        SecureSparqlExecutor::new(SecurityConfig::default())
    }

    /// Create executor with strict security settings
    pub fn create_strict() -> Result<SecureSparqlExecutor> {
        let mut config = SecurityConfig::default();
        config.max_execution_time = Duration::from_secs(10);
        config.max_memory_usage = 50 * 1024 * 1024; // 50MB
        config.max_result_count = 1000;
        config.max_complexity_score = 50.0;
        config.enable_injection_detection = true;
        config.enable_sandboxing = true;
        config.enable_security_logging = true;
        
        // Remove potentially dangerous functions
        config.allowed_functions.retain(|f| !["RAND", "NOW"].contains(&f.as_str()));
        
        SecureSparqlExecutor::new(config)
    }

    /// Create executor with relaxed security settings (for trusted environments)
    pub fn create_relaxed() -> Result<SecureSparqlExecutor> {
        let mut config = SecurityConfig::default();
        config.max_execution_time = Duration::from_secs(60);
        config.max_memory_usage = 500 * 1024 * 1024; // 500MB
        config.max_result_count = 50000;
        config.max_complexity_score = 500.0;
        config.enable_injection_detection = true; // Keep injection detection
        config.enable_sandboxing = false;
        
        SecureSparqlExecutor::new(config)
    }

    /// Create executor with custom configuration
    pub fn create_custom(config: SecurityConfig) -> Result<SecureSparqlExecutor> {
        SecureSparqlExecutor::new(config)
    }
}

/// Utility functions for security management
pub mod utils {
    use super::*;

    /// Create a basic user security context with standard permissions
    pub fn create_standard_user_context(
        executor: &mut SecureSparqlExecutor,
        user_id: &str,
    ) -> Result<String> {
        let permissions = vec![
            Permission::ReadData,
            Permission::ExecuteQueries,
        ];
        
        let constraints = SecurityConstraints {
            max_execution_time: Duration::from_secs(30),
            max_memory_usage: 10 * 1024 * 1024, // 10MB
            max_result_count: 1000,
            ..Default::default()
        };

        executor.policy_manager.create_security_context(user_id, permissions, constraints)
    }

    /// Create an admin security context with elevated permissions
    pub fn create_admin_context(
        executor: &mut SecureSparqlExecutor,
        user_id: &str,
    ) -> Result<String> {
        let permissions = vec![
            Permission::ReadData,
            Permission::ReadMetadata,
            Permission::ExecuteQueries,
            Permission::UseAdvancedFunctions,
            Permission::AccessExternalData,
            Permission::ViewAuditLogs,
            Permission::ManagePolicies,
        ];

        let constraints = SecurityConstraints {
            max_execution_time: Duration::from_secs(300), // 5 minutes
            max_memory_usage: 100 * 1024 * 1024, // 100MB
            max_result_count: 50000,
            ..Default::default()
        };

        executor.policy_manager.create_security_context(user_id, permissions, constraints)
    }

    /// Validate a SPARQL query for basic safety (quick check)
    pub fn quick_security_check(query: &str) -> Result<bool> {
        let config = SecurityConfig::default();
        let analyzer = SparqlSecurityAnalyzer::new(config)?;
        let result = analyzer.analyze_query(query)?;
        Ok(result.is_safe)
    }

    /// Sanitize a SPARQL query for safe execution
    pub fn sanitize_query(query: &str) -> Result<String> {
        let config = SecurityConfig::default();
        let analyzer = SparqlSecurityAnalyzer::new(config)?;
        analyzer.sanitize_query(query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::advanced::Permission;

    #[test]
    fn test_secure_executor_creation() {
        let executor = SecureExecutorFactory::create_default();
        assert!(executor.is_ok());
    }

    #[test]
    fn test_factory_configurations() {
        let default_exec = SecureExecutorFactory::create_default().unwrap();
        let strict_exec = SecureExecutorFactory::create_strict().unwrap();
        let relaxed_exec = SecureExecutorFactory::create_relaxed().unwrap();

        // Basic validation that different configurations were created
        assert_eq!(default_exec.security_config.max_execution_time, Duration::from_secs(30));
        assert_eq!(strict_exec.security_config.max_execution_time, Duration::from_secs(10));
        assert_eq!(relaxed_exec.security_config.max_execution_time, Duration::from_secs(60));
    }

    #[test]
    fn test_context_creation_utils() {
        let mut executor = SecureExecutorFactory::create_default().unwrap();
        
        let user_context = utils::create_standard_user_context(&mut executor, "test_user");
        assert!(user_context.is_ok());

        let admin_context = utils::create_admin_context(&mut executor, "admin_user");
        assert!(admin_context.is_ok());
    }

    #[test]
    fn test_quick_security_check() {
        let safe_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let result = utils::quick_security_check(safe_query);
        assert!(result.is_ok());
        assert!(result.unwrap());

        let dangerous_query = "DROP GRAPH <http://example.org/graph>";
        let result = utils::quick_security_check(dangerous_query);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_query_sanitization() {
        let query_with_comments = r"
            SELECT ?s ?p ?o WHERE {
                ?s ?p ?o .  # This is a comment
            }
        ";

        let sanitized = utils::sanitize_query(query_with_comments);
        assert!(sanitized.is_ok());
        assert!(!sanitized.unwrap().contains('#'));
    }

    #[test]
    fn test_execution_statistics() {
        let mut stats = ExecutionStatistics::new();
        
        stats.record_execution(Duration::from_millis(100), true);
        stats.record_execution(Duration::from_millis(200), false);
        stats.record_execution(Duration::from_millis(150), true);

        let metrics = stats.get_metrics();
        assert_eq!(metrics.total_executions, 3);
        assert_eq!(metrics.successful_executions, 2);
        assert_eq!(metrics.success_rate, 2.0 / 3.0);
    }
}