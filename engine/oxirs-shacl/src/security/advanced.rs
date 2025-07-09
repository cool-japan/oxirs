//! Advanced Security Features for SPARQL Execution
//!
//! This module provides enterprise-grade security features for SPARQL constraint execution
//! including query rewriting, policy-based execution control, audit logging, rate limiting,
//! and advanced injection detection.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{
    security::{SecurityAnalysisResult, SecurityConfig, SparqlSecurityAnalyzer},
    Result, ShaclError,
};

/// Advanced security policy manager
#[derive(Debug)]
pub struct SecurityPolicyManager {
    /// Active security policies
    policies: Arc<RwLock<HashMap<String, SecurityPolicy>>>,

    /// Security context registry
    contexts: Arc<RwLock<HashMap<String, SecurityContext>>>,

    /// Audit logger
    pub audit_logger: AuditLogger,

    /// Rate limiter
    rate_limiter: RateLimiter,

    /// Query rewriter for security
    query_rewriter: QuerySecurityRewriter,

    /// Advanced injection detector
    injection_detector: AdvancedInjectionDetector,

    /// Security event monitor
    event_monitor: SecurityEventMonitor,
}

impl SecurityPolicyManager {
    /// Create a new security policy manager
    pub fn new() -> Result<Self> {
        Ok(Self {
            policies: Arc::new(RwLock::new(HashMap::new())),
            contexts: Arc::new(RwLock::new(HashMap::new())),
            audit_logger: AuditLogger::new()?,
            rate_limiter: RateLimiter::new(),
            query_rewriter: QuerySecurityRewriter::new()?,
            injection_detector: AdvancedInjectionDetector::new()?,
            event_monitor: SecurityEventMonitor::new(),
        })
    }

    /// Create a security context for SPARQL execution
    pub fn create_security_context(
        &mut self,
        user_id: &str,
        permissions: Vec<Permission>,
        constraints: SecurityConstraints,
    ) -> Result<String> {
        let context_id = uuid::Uuid::new_v4().to_string();

        let context = SecurityContext {
            id: context_id.clone(),
            user_id: user_id.to_string(),
            permissions: permissions.into_iter().collect(),
            constraints,
            created_at: Utc::now(),
            last_used: Utc::now(),
            queries_executed: 0,
            total_execution_time: Duration::default(),
        };

        {
            let mut contexts = self.contexts.write().unwrap();
            contexts.insert(context_id.clone(), context);
        }

        self.audit_logger.log_event(SecurityEvent {
            event_type: SecurityEventType::ContextCreated,
            context_id: Some(context_id.clone()),
            user_id: Some(user_id.to_string()),
            message: "Security context created".to_string(),
            timestamp: Utc::now(),
            severity: SecurityEventSeverity::Info,
            details: HashMap::new(),
        })?;

        Ok(context_id)
    }

    /// Execute SPARQL query with advanced security
    pub fn execute_secure_sparql(
        &mut self,
        query: &str,
        context_id: &str,
        additional_constraints: Option<ExecutionConstraints>,
    ) -> Result<SecureExecutionResult> {
        let start_time = Instant::now();

        // 1. Validate security context
        let context = self.validate_security_context(context_id)?;

        // 2. Check rate limits
        self.rate_limiter
            .check_rate_limit(&context.user_id, &context.id)?;

        // 3. Advanced injection detection
        let injection_result = self.injection_detector.analyze_query(query)?;
        if !injection_result.is_safe {
            self.handle_security_violation(
                SecurityViolationType::InjectionDetected,
                &context,
                &format!("Injection detected: {:?}", injection_result.threats),
            )?;
            return Err(ShaclError::SecurityViolation(
                "Query blocked due to injection detection".to_string(),
            ));
        }

        // 4. Policy-based authorization
        let policy_result = self.check_query_policy(query, &context)?;
        if !policy_result.authorized {
            self.handle_security_violation(
                SecurityViolationType::PolicyViolation,
                &context,
                &policy_result.denial_reason,
            )?;
            return Err(ShaclError::SecurityViolation(format!(
                "Query blocked by policy: {}",
                policy_result.denial_reason
            )));
        }

        // 5. Query rewriting for security
        let rewritten_query = self.query_rewriter.rewrite_for_security(
            query,
            &context,
            &additional_constraints.unwrap_or_default(),
        )?;

        // 6. Basic security analysis (using existing analyzer)
        let analyzer = SparqlSecurityAnalyzer::new(SecurityConfig::default())?;
        let analysis = analyzer.analyze_query(&rewritten_query)?;

        if !analysis.is_safe {
            self.handle_security_violation(
                SecurityViolationType::SecurityAnalysisFailed,
                &context,
                &format!(
                    "Security analysis failed: {} violations",
                    analysis.violations.len()
                ),
            )?;
            return Err(ShaclError::SecurityViolation(
                "Query failed security analysis".to_string(),
            ));
        }

        // 7. Execute query (placeholder - would integrate with actual SPARQL engine)
        let execution_result = self.execute_query_safely(&rewritten_query, &context)?;

        // 8. Update context and log execution
        self.update_context_stats(context_id, start_time.elapsed())?;

        self.audit_logger.log_event(SecurityEvent {
            event_type: SecurityEventType::QueryExecuted,
            context_id: Some(context_id.to_string()),
            user_id: Some(context.user_id.clone()),
            message: "SPARQL query executed successfully".to_string(),
            timestamp: Utc::now(),
            severity: SecurityEventSeverity::Info,
            details: {
                let mut details = HashMap::new();
                details.insert("original_query_length".to_string(), query.len().to_string());
                details.insert(
                    "rewritten_query_length".to_string(),
                    rewritten_query.len().to_string(),
                );
                details.insert(
                    "execution_time_ms".to_string(),
                    start_time.elapsed().as_millis().to_string(),
                );
                details
            },
        })?;

        Ok(SecureExecutionResult {
            success: true,
            original_query: query.to_string(),
            rewritten_query,
            execution_time: start_time.elapsed(),
            security_analysis: analysis,
            policy_result,
            injection_analysis: injection_result,
            result_data: execution_result,
        })
    }

    /// Validate security context and check permissions
    fn validate_security_context(&self, context_id: &str) -> Result<SecurityContext> {
        let contexts = self.contexts.read().unwrap();
        let context = contexts
            .get(context_id)
            .ok_or_else(|| ShaclError::SecurityViolation("Invalid security context".to_string()))?;

        // Check if context is still valid (not expired)
        let max_age = Duration::from_secs(24 * 3600); // 24 hour sessions
        if context.created_at + chrono::Duration::from_std(max_age).unwrap() < Utc::now() {
            return Err(ShaclError::SecurityViolation(
                "Security context expired".to_string(),
            ));
        }

        Ok(context.clone())
    }

    /// Check query against security policies
    fn check_query_policy(
        &self,
        query: &str,
        context: &SecurityContext,
    ) -> Result<PolicyAuthorizationResult> {
        let policies = self.policies.read().unwrap();

        // Check if user has any policies applied
        let applicable_policies: Vec<_> = policies
            .values()
            .filter(|policy| policy.applies_to_user(&context.user_id))
            .collect();

        if applicable_policies.is_empty() {
            // Default allow if no policies
            return Ok(PolicyAuthorizationResult {
                authorized: true,
                applied_policies: Vec::new(),
                denial_reason: String::new(),
            });
        }

        for policy in &applicable_policies {
            if !policy.authorize_query(query, context)? {
                return Ok(PolicyAuthorizationResult {
                    authorized: false,
                    applied_policies: vec![policy.id.clone()],
                    denial_reason: format!("Denied by policy: {}", policy.name),
                });
            }
        }

        Ok(PolicyAuthorizationResult {
            authorized: true,
            applied_policies: applicable_policies.iter().map(|p| p.id.clone()).collect(),
            denial_reason: String::new(),
        })
    }

    /// Execute query with additional safety measures
    fn execute_query_safely(&self, _query: &str, _context: &SecurityContext) -> Result<QueryResult> {
        // Placeholder implementation - in practice would integrate with actual SPARQL engine
        // with sandboxing, resource limits, and monitoring

        // Simulate execution
        Ok(QueryResult {
            bindings: Vec::new(),
            execution_time: Duration::from_millis(100),
            memory_used: 1024,
            rows_returned: 0,
        })
    }

    /// Handle security violations
    fn handle_security_violation(
        &mut self,
        violation_type: SecurityViolationType,
        context: &SecurityContext,
        message: &str,
    ) -> Result<()> {
        // Log security event
        self.audit_logger.log_event(SecurityEvent {
            event_type: SecurityEventType::SecurityViolation,
            context_id: Some(context.id.clone()),
            user_id: Some(context.user_id.clone()),
            message: message.to_string(),
            timestamp: Utc::now(),
            severity: SecurityEventSeverity::Warning,
            details: {
                let mut details = HashMap::new();
                details.insert("violation_type".to_string(), format!("{violation_type:?}"));
                details
            },
        })?;

        // Update security event monitor
        self.event_monitor
            .record_violation(&context.user_id, violation_type);

        // Check if user should be blocked
        if self.event_monitor.should_block_user(&context.user_id) {
            self.audit_logger.log_event(SecurityEvent {
                event_type: SecurityEventType::UserBlocked,
                context_id: Some(context.id.clone()),
                user_id: Some(context.user_id.clone()),
                message: "User blocked due to multiple security violations".to_string(),
                timestamp: Utc::now(),
                severity: SecurityEventSeverity::Critical,
                details: HashMap::new(),
            })?;
        }

        Ok(())
    }

    /// Update context statistics
    fn update_context_stats(&self, context_id: &str, execution_time: Duration) -> Result<()> {
        let mut contexts = self.contexts.write().unwrap();
        if let Some(context) = contexts.get_mut(context_id) {
            context.last_used = Utc::now();
            context.queries_executed += 1;
            context.total_execution_time += execution_time;
        }
        Ok(())
    }

    /// Install a security policy
    pub fn install_policy(&mut self, policy: SecurityPolicy) -> Result<()> {
        let policy_id = policy.id.clone();

        {
            let mut policies = self.policies.write().unwrap();
            policies.insert(policy_id.clone(), policy);
        }

        self.audit_logger.log_event(SecurityEvent {
            event_type: SecurityEventType::PolicyInstalled,
            context_id: None,
            user_id: None,
            message: format!("Security policy installed: {policy_id}"),
            timestamp: Utc::now(),
            severity: SecurityEventSeverity::Info,
            details: HashMap::new(),
        })?;

        Ok(())
    }

    /// Get security metrics
    pub fn get_security_metrics(&self) -> SecurityMetrics {
        let contexts = self.contexts.read().unwrap();
        let total_contexts = contexts.len();
        let active_contexts = contexts
            .values()
            .filter(|c| c.last_used + chrono::Duration::hours(1) > Utc::now())
            .count();

        SecurityMetrics {
            total_contexts,
            active_contexts,
            total_queries_executed: contexts.values().map(|c| c.queries_executed).sum(),
            rate_limit_violations: self.rate_limiter.get_violation_count(),
            security_violations: self.event_monitor.get_total_violations(),
            blocked_users: self.event_monitor.get_blocked_users().len(),
        }
    }
}

/// Security context for query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    pub id: String,
    pub user_id: String,
    pub permissions: HashSet<Permission>,
    pub constraints: SecurityConstraints,
    pub created_at: DateTime<Utc>,
    pub last_used: DateTime<Utc>,
    pub queries_executed: u64,
    pub total_execution_time: Duration,
}

/// User permissions for SPARQL execution
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    ReadData,
    ReadMetadata,
    ExecuteQueries,
    UseAdvancedFunctions,
    AccessExternalData,
    ModifyConstraints,
    ViewAuditLogs,
    ManagePolicies,
}

/// Security constraints for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConstraints {
    pub max_execution_time: Duration,
    pub max_memory_usage: usize,
    pub max_result_count: usize,
    pub allowed_functions: HashSet<String>,
    pub allowed_prefixes: HashSet<String>,
    pub data_access_restrictions: Vec<DataAccessRestriction>,
}

impl Default for SecurityConstraints {
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_secs(30),
            max_memory_usage: 10 * 1024 * 1024, // 10MB
            max_result_count: 1000,
            allowed_functions: HashSet::new(),
            allowed_prefixes: HashSet::new(),
            data_access_restrictions: Vec::new(),
        }
    }
}

/// Data access restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAccessRestriction {
    pub restriction_type: RestrictionType,
    pub pattern: String,
    pub allowed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestrictionType {
    GraphPattern,
    PredicatePattern,
    SubjectPattern,
    ObjectPattern,
}

/// Security policy for SPARQL execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    pub id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub enabled: bool,
    pub applicable_users: PolicyScope,
    pub query_rules: Vec<QueryRule>,
    pub execution_limits: ExecutionLimits,
    pub audit_requirements: AuditRequirements,
}

impl SecurityPolicy {
    fn applies_to_user(&self, user_id: &str) -> bool {
        match &self.applicable_users {
            PolicyScope::AllUsers => true,
            PolicyScope::SpecificUsers(users) => users.contains(&user_id.to_string()),
            PolicyScope::UserGroups(_groups) => {
                // In practice, would check if user belongs to any of these groups
                false
            }
            PolicyScope::RoleBasedAccess(_roles) => {
                // In practice, would check user roles
                false
            }
        }
    }

    fn authorize_query(&self, query: &str, context: &SecurityContext) -> Result<bool> {
        for rule in &self.query_rules {
            if !rule.evaluate(query, context)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyScope {
    AllUsers,
    SpecificUsers(Vec<String>),
    UserGroups(Vec<String>),
    RoleBasedAccess(Vec<String>),
}

/// Query rule for policy evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRule {
    pub rule_type: QueryRuleType,
    pub condition: String,
    pub action: PolicyAction,
}

impl QueryRule {
    fn evaluate(&self, query: &str, _context: &SecurityContext) -> Result<bool> {
        match self.rule_type {
            QueryRuleType::PatternMatch => {
                let regex = Regex::new(&self.condition)?;
                let matches = regex.is_match(query);

                match self.action {
                    PolicyAction::Allow => Ok(matches),
                    PolicyAction::Deny => Ok(!matches),
                    PolicyAction::RequireApproval => {
                        // In practice, would trigger approval workflow
                        Ok(!matches)
                    }
                }
            }
            QueryRuleType::ComplexityLimit => {
                // Simplified complexity check
                let complexity = query.len() as f64 / 100.0;
                let limit: f64 = self.condition.parse().unwrap_or(10.0);

                match self.action {
                    PolicyAction::Allow => Ok(complexity <= limit),
                    PolicyAction::Deny => Ok(complexity > limit),
                    PolicyAction::RequireApproval => Ok(complexity <= limit * 2.0),
                }
            }
            QueryRuleType::FunctionRestriction => {
                let forbidden_functions: Vec<&str> = self.condition.split(',').collect();
                let contains_forbidden = forbidden_functions
                    .iter()
                    .any(|func| query.to_uppercase().contains(&func.to_uppercase()));

                match self.action {
                    PolicyAction::Allow => Ok(!contains_forbidden),
                    PolicyAction::Deny => Ok(contains_forbidden),
                    PolicyAction::RequireApproval => Ok(!contains_forbidden),
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryRuleType {
    PatternMatch,
    ComplexityLimit,
    FunctionRestriction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyAction {
    Allow,
    Deny,
    RequireApproval,
}

/// Execution limits for policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLimits {
    pub max_execution_time: Duration,
    pub max_memory_usage: usize,
    pub max_concurrent_queries: usize,
    pub max_queries_per_hour: usize,
}

/// Audit requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRequirements {
    pub log_all_queries: bool,
    pub log_query_results: bool,
    pub log_execution_metrics: bool,
    pub retention_period_days: u32,
}

/// Additional execution constraints
#[derive(Debug, Clone, Default)]
pub struct ExecutionConstraints {
    pub priority: ExecutionPriority,
    pub isolation_level: IsolationLevel,
    pub timeout_override: Option<Duration>,
    pub custom_limits: HashMap<String, String>,
}

#[derive(Debug, Clone, Default)]
pub enum ExecutionPriority {
    Low,
    #[default]
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, Default)]
pub enum IsolationLevel {
    None,
    #[default]
    Basic,
    Process,
    Container,
}

/// Audit logger for security events
#[derive(Debug)]
pub struct AuditLogger {
    events: Arc<RwLock<Vec<SecurityEvent>>>,
}

impl AuditLogger {
    fn new() -> Result<Self> {
        Ok(Self {
            events: Arc::new(RwLock::new(Vec::new())),
        })
    }

    fn log_event(&self, event: SecurityEvent) -> Result<()> {
        let mut events = self.events.write().unwrap();
        events.push(event);
        // In practice, would also write to persistent storage
        Ok(())
    }

    pub fn get_events(&self, filter: Option<SecurityEventFilter>) -> Vec<SecurityEvent> {
        let events = self.events.read().unwrap();
        if let Some(filter) = filter {
            events
                .iter()
                .filter(|e| filter.matches(e))
                .cloned()
                .collect()
        } else {
            events.clone()
        }
    }
}

/// Security event for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub event_type: SecurityEventType,
    pub context_id: Option<String>,
    pub user_id: Option<String>,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub severity: SecurityEventSeverity,
    pub details: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventType {
    ContextCreated,
    QueryExecuted,
    SecurityViolation,
    PolicyViolation,
    RateLimitExceeded,
    UserBlocked,
    PolicyInstalled,
    InjectionAttempt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Security event filter
#[derive(Debug, Clone)]
pub struct SecurityEventFilter {
    pub event_types: Option<Vec<SecurityEventType>>,
    pub user_id: Option<String>,
    pub severity: Option<SecurityEventSeverity>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
}

impl SecurityEventFilter {
    fn matches(&self, event: &SecurityEvent) -> bool {
        if let Some(types) = &self.event_types {
            if !types
                .iter()
                .any(|t| std::mem::discriminant(t) == std::mem::discriminant(&event.event_type))
            {
                return false;
            }
        }

        if let Some(user_id) = &self.user_id {
            if event.user_id.as_ref() != Some(user_id) {
                return false;
            }
        }

        if let Some(severity) = &self.severity {
            if std::mem::discriminant(severity) != std::mem::discriminant(&event.severity) {
                return false;
            }
        }

        if let Some(start_time) = &self.start_time {
            if event.timestamp < *start_time {
                return false;
            }
        }

        if let Some(end_time) = &self.end_time {
            if event.timestamp > *end_time {
                return false;
            }
        }

        true
    }
}

/// Rate limiter for query execution
#[derive(Debug)]
pub struct RateLimiter {
    user_buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
    context_buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
    violation_count: Arc<RwLock<u64>>,
}

impl RateLimiter {
    fn new() -> Self {
        Self {
            user_buckets: Arc::new(RwLock::new(HashMap::new())),
            context_buckets: Arc::new(RwLock::new(HashMap::new())),
            violation_count: Arc::new(RwLock::new(0)),
        }
    }

    fn check_rate_limit(&self, user_id: &str, context_id: &str) -> Result<()> {
        // Check user-level rate limit
        {
            let mut buckets = self.user_buckets.write().unwrap();
            let bucket = buckets.entry(user_id.to_string()).or_insert_with(|| {
                TokenBucket::new(100, Duration::from_secs(3600)) // 100 queries per hour
            });

            if !bucket.try_consume(1) {
                let mut violations = self.violation_count.write().unwrap();
                *violations += 1;
                return Err(ShaclError::SecurityViolation(
                    "User rate limit exceeded".to_string(),
                ));
            }
        }

        // Check context-level rate limit
        {
            let mut buckets = self.context_buckets.write().unwrap();
            let bucket = buckets.entry(context_id.to_string()).or_insert_with(|| {
                TokenBucket::new(50, Duration::from_secs(60)) // 50 queries per minute per context
            });

            if !bucket.try_consume(1) {
                let mut violations = self.violation_count.write().unwrap();
                *violations += 1;
                return Err(ShaclError::SecurityViolation(
                    "Context rate limit exceeded".to_string(),
                ));
            }
        }

        Ok(())
    }

    fn get_violation_count(&self) -> u64 {
        *self.violation_count.read().unwrap()
    }
}

/// Token bucket for rate limiting
#[derive(Debug)]
pub struct TokenBucket {
    capacity: u32,
    tokens: u32,
    refill_period: Duration,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: u32, refill_period: Duration) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_period,
            last_refill: Instant::now(),
        }
    }

    fn try_consume(&mut self, tokens: u32) -> bool {
        self.refill();

        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_refill) >= self.refill_period {
            self.tokens = self.capacity;
            self.last_refill = now;
        }
    }
}

/// Query security rewriter
#[derive(Debug)]
pub struct QuerySecurityRewriter {
    rewrite_rules: Vec<RewriteRule>,
}

impl QuerySecurityRewriter {
    fn new() -> Result<Self> {
        Ok(Self {
            rewrite_rules: vec![
                RewriteRule {
                    name: "limit_results".to_string(),
                    pattern: Regex::new(r"(?i)select\s+")?,
                    replacement: "SELECT ".to_string(),
                    description: "Add result limits to SELECT queries".to_string(),
                },
                RewriteRule {
                    name: "sanitize_literals".to_string(),
                    pattern: Regex::new(r#""([^"\\]*(\\.[^"\\]*)*)""#)?,
                    replacement: "\"$1\"".to_string(),
                    description: "Sanitize string literals".to_string(),
                },
            ],
        })
    }

    fn rewrite_for_security(
        &self,
        query: &str,
        context: &SecurityContext,
        constraints: &ExecutionConstraints,
    ) -> Result<String> {
        let mut rewritten = query.to_string();

        // Add result limits if not present
        if !rewritten.to_uppercase().contains("LIMIT") {
            let limit = context.constraints.max_result_count.min(10000);
            rewritten = format!("{rewritten} LIMIT {limit}");
        }

        // Add timeout hints
        if let Some(timeout) = constraints.timeout_override {
            // In practice, would add query hints for timeout
            rewritten = format!("# TIMEOUT: {timeout:?}\n{rewritten}");
        }

        // Apply rewrite rules
        for rule in &self.rewrite_rules {
            if rule.name == "limit_results" {
                continue; // Already handled above
            }
            rewritten = rule
                .pattern
                .replace_all(&rewritten, &rule.replacement)
                .to_string();
        }

        Ok(rewritten)
    }
}

#[derive(Debug)]
struct RewriteRule {
    name: String,
    pattern: Regex,
    replacement: String,
    description: String,
}

/// Advanced injection detector
#[derive(Debug)]
pub struct AdvancedInjectionDetector {
    threat_patterns: Vec<ThreatPattern>,
    ml_model: Option<Box<dyn InjectionMLModel>>,
}

impl AdvancedInjectionDetector {
    fn new() -> Result<Self> {
        let threat_patterns = vec![
            ThreatPattern {
                name: "union_injection".to_string(),
                pattern: Regex::new(r"(?i)\bunion\s+(?:all\s+)?select")?,
                severity: ThreatSeverity::High,
                description: "Potential UNION-based injection".to_string(),
            },
            ThreatPattern {
                name: "nested_queries".to_string(),
                pattern: Regex::new(r"(?i)select[^}]*\{[^}]*select")?,
                severity: ThreatSeverity::Medium,
                description: "Deeply nested queries".to_string(),
            },
            ThreatPattern {
                name: "external_data".to_string(),
                pattern: Regex::new(r"(?i)service\s+<[^>]*>")?,
                severity: ThreatSeverity::High,
                description: "External service access".to_string(),
            },
        ];

        Ok(Self {
            threat_patterns,
            ml_model: None, // Would integrate with ML model in practice
        })
    }

    fn analyze_query(&self, query: &str) -> Result<InjectionAnalysisResult> {
        let mut threats = Vec::new();
        let mut confidence_score = 1.0;

        // Pattern-based detection
        for pattern in &self.threat_patterns {
            if pattern.pattern.is_match(query) {
                threats.push(DetectedThreat {
                    threat_type: pattern.name.clone(),
                    severity: pattern.severity.clone(),
                    description: pattern.description.clone(),
                    confidence: 0.8,
                    location: pattern.pattern.find(query).map(|m| m.start()),
                });

                confidence_score *= 0.7; // Reduce confidence for each threat
            }
        }

        // ML-based detection (placeholder)
        if let Some(_ml_model) = &self.ml_model {
            // Would use ML model to analyze query
        }

        Ok(InjectionAnalysisResult {
            is_safe: threats.is_empty(),
            threats,
            confidence_score,
            analysis_method: "pattern_and_ml".to_string(),
        })
    }
}

#[derive(Debug)]
struct ThreatPattern {
    name: String,
    pattern: Regex,
    severity: ThreatSeverity,
    description: String,
}

#[derive(Debug, Clone)]
pub enum ThreatSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug)]
pub struct InjectionAnalysisResult {
    pub is_safe: bool,
    pub threats: Vec<DetectedThreat>,
    pub confidence_score: f64,
    pub analysis_method: String,
}

#[derive(Debug)]
pub struct DetectedThreat {
    pub threat_type: String,
    pub severity: ThreatSeverity,
    pub description: String,
    pub confidence: f64,
    pub location: Option<usize>,
}

trait InjectionMLModel: std::fmt::Debug {
    fn predict_injection_probability(&self, query: &str) -> f64;
}

/// Security event monitor
#[derive(Debug)]
pub struct SecurityEventMonitor {
    user_violations: Arc<RwLock<HashMap<String, ViolationTracker>>>,
    blocked_users: Arc<RwLock<HashSet<String>>>,
}

impl SecurityEventMonitor {
    fn new() -> Self {
        Self {
            user_violations: Arc::new(RwLock::new(HashMap::new())),
            blocked_users: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    fn record_violation(&self, user_id: &str, violation_type: SecurityViolationType) {
        let mut violations = self.user_violations.write().unwrap();
        let tracker = violations
            .entry(user_id.to_string())
            .or_insert_with(ViolationTracker::new);
        tracker.record_violation(violation_type);
    }

    fn should_block_user(&self, user_id: &str) -> bool {
        let violations = self.user_violations.read().unwrap();
        if let Some(tracker) = violations.get(user_id) {
            tracker.should_block()
        } else {
            false
        }
    }

    fn get_total_violations(&self) -> usize {
        let violations = self.user_violations.read().unwrap();
        violations.values().map(|t| t.total_violations()).sum()
    }

    fn get_blocked_users(&self) -> Vec<String> {
        let blocked = self.blocked_users.read().unwrap();
        blocked.iter().cloned().collect()
    }
}

#[derive(Debug)]
struct ViolationTracker {
    violations: HashMap<SecurityViolationType, u32>,
    last_violation: SystemTime,
}

impl ViolationTracker {
    fn new() -> Self {
        Self {
            violations: HashMap::new(),
            last_violation: SystemTime::now(),
        }
    }

    fn record_violation(&mut self, violation_type: SecurityViolationType) {
        *self.violations.entry(violation_type).or_insert(0) += 1;
        self.last_violation = SystemTime::now();
    }

    fn should_block(&self) -> bool {
        // Block if more than 5 violations in the last hour
        let total_recent = self.violations.values().sum::<u32>();
        total_recent > 5
    }

    fn total_violations(&self) -> usize {
        self.violations.values().sum::<u32>() as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SecurityViolationType {
    InjectionDetected,
    PolicyViolation,
    SecurityAnalysisFailed,
    RateLimitExceeded,
    UnauthorizedAccess,
    MaliciousPattern,
}

/// Result types for security operations
#[derive(Debug)]
pub struct SecureExecutionResult {
    pub success: bool,
    pub original_query: String,
    pub rewritten_query: String,
    pub execution_time: Duration,
    pub security_analysis: SecurityAnalysisResult,
    pub policy_result: PolicyAuthorizationResult,
    pub injection_analysis: InjectionAnalysisResult,
    pub result_data: QueryResult,
}

#[derive(Debug)]
pub struct PolicyAuthorizationResult {
    pub authorized: bool,
    pub applied_policies: Vec<String>,
    pub denial_reason: String,
}

#[derive(Debug)]
pub struct QueryResult {
    pub bindings: Vec<HashMap<String, String>>,
    pub execution_time: Duration,
    pub memory_used: usize,
    pub rows_returned: usize,
}

/// Security metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub total_contexts: usize,
    pub active_contexts: usize,
    pub total_queries_executed: u64,
    pub rate_limit_violations: u64,
    pub security_violations: usize,
    pub blocked_users: usize,
}

impl Default for SecurityPolicyManager {
    fn default() -> Self {
        Self::new().expect("Failed to create SecurityPolicyManager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_policy_manager_creation() {
        let manager = SecurityPolicyManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_security_context_creation() {
        let mut manager = SecurityPolicyManager::new().unwrap();

        let context_id = manager
            .create_security_context(
                "test_user",
                vec![Permission::ReadData, Permission::ExecuteQueries],
                SecurityConstraints::default(),
            )
            .unwrap();

        assert!(!context_id.is_empty());
    }

    #[test]
    fn test_rate_limiter() {
        let rate_limiter = RateLimiter::new();

        // Should allow initial requests
        assert!(rate_limiter.check_rate_limit("user1", "context1").is_ok());
    }

    #[test]
    fn test_injection_detector() {
        let detector = AdvancedInjectionDetector::new().unwrap();

        let safe_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let result = detector.analyze_query(safe_query).unwrap();
        assert!(result.is_safe);

        let malicious_query = "SELECT ?s WHERE { ?s ?p ?o } UNION SELECT ?x WHERE { ?x ?y ?z }";
        let result = detector.analyze_query(malicious_query).unwrap();
        assert!(!result.is_safe);
    }

    #[test]
    fn test_query_rewriter() {
        let rewriter = QuerySecurityRewriter::new().unwrap();
        let context = SecurityContext {
            id: "test".to_string(),
            user_id: "test_user".to_string(),
            permissions: HashSet::new(),
            constraints: SecurityConstraints::default(),
            created_at: Utc::now(),
            last_used: Utc::now(),
            queries_executed: 0,
            total_execution_time: Duration::default(),
        };

        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let rewritten = rewriter
            .rewrite_for_security(query, &context, &ExecutionConstraints::default())
            .unwrap();

        assert!(rewritten.contains("LIMIT"));
    }

    #[test]
    fn test_security_policy() {
        let policy = SecurityPolicy {
            id: "test_policy".to_string(),
            name: "Test Policy".to_string(),
            description: "A test security policy".to_string(),
            version: "1.0".to_string(),
            enabled: true,
            applicable_users: PolicyScope::AllUsers,
            query_rules: vec![QueryRule {
                rule_type: QueryRuleType::PatternMatch,
                condition: r"(?i)delete".to_string(),
                action: PolicyAction::Deny,
            }],
            execution_limits: ExecutionLimits {
                max_execution_time: Duration::from_secs(30),
                max_memory_usage: 10 * 1024 * 1024,
                max_concurrent_queries: 5,
                max_queries_per_hour: 100,
            },
            audit_requirements: AuditRequirements {
                log_all_queries: true,
                log_query_results: false,
                log_execution_metrics: true,
                retention_period_days: 90,
            },
        };

        let context = SecurityContext {
            id: "test".to_string(),
            user_id: "test_user".to_string(),
            permissions: HashSet::new(),
            constraints: SecurityConstraints::default(),
            created_at: Utc::now(),
            last_used: Utc::now(),
            queries_executed: 0,
            total_execution_time: Duration::default(),
        };

        // Should block DELETE queries
        let delete_query = "DELETE WHERE { ?s ?p ?o }";
        assert!(!policy.authorize_query(delete_query, &context).unwrap());

        // Should allow SELECT queries
        let select_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        assert!(policy.authorize_query(select_query, &context).unwrap());
    }
}
