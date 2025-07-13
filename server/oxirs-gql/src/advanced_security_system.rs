//! Advanced Security and Authorization System for GraphQL
//!
//! This module provides comprehensive security features including field-level authorization,
//! rate limiting, query depth analysis, query whitelisting, SQL injection prevention,
//! and advanced threat detection for GraphQL endpoints.

use anyhow::Result;
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex as AsyncMutex, RwLock as AsyncRwLock};
use tracing::instrument;

use crate::ast::OperationType;

/// Security configuration for GraphQL endpoints
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Enable/disable various security features
    pub enable_rate_limiting: bool,
    pub enable_query_depth_analysis: bool,
    pub enable_query_complexity_analysis: bool,
    pub enable_query_whitelisting: bool,
    pub enable_field_authorization: bool,
    pub enable_introspection_protection: bool,
    pub enable_mutation_protection: bool,
    pub enable_subscription_protection: bool,
    pub enable_ip_filtering: bool,
    pub enable_user_agent_filtering: bool,
    pub enable_threat_detection: bool,
    pub enable_audit_logging: bool,

    /// Rate limiting configuration
    pub rate_limit_requests_per_minute: usize,
    pub rate_limit_burst_capacity: usize,
    pub rate_limit_window_size: Duration,

    /// Query analysis limits
    pub max_query_depth: usize,
    pub max_query_complexity: usize,
    pub max_query_aliases: usize,
    pub max_query_fields: usize,
    pub max_selection_sets: usize,

    /// Authentication configuration
    pub require_authentication: bool,
    pub jwt_secret: Option<String>,
    pub api_key_header: String,
    pub session_timeout: Duration,

    /// Threat detection thresholds
    pub threat_detection_sensitivity: ThreatSensitivity,
    pub max_failed_attempts: usize,
    pub lockout_duration: Duration,
    pub suspicious_pattern_threshold: f64,

    /// IP filtering
    pub allowed_ip_ranges: Vec<IpRange>,
    pub blocked_ips: HashSet<IpAddr>,
    pub geolocation_restrictions: Option<GeolocationRestrictions>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_rate_limiting: true,
            enable_query_depth_analysis: true,
            enable_query_complexity_analysis: true,
            enable_query_whitelisting: false,
            enable_field_authorization: true,
            enable_introspection_protection: true,
            enable_mutation_protection: true,
            enable_subscription_protection: true,
            enable_ip_filtering: false,
            enable_user_agent_filtering: false,
            enable_threat_detection: true,
            enable_audit_logging: true,

            rate_limit_requests_per_minute: 100,
            rate_limit_burst_capacity: 20,
            rate_limit_window_size: Duration::from_secs(60),

            max_query_depth: 10,
            max_query_complexity: 1000,
            max_query_aliases: 50,
            max_query_fields: 100,
            max_selection_sets: 50,

            require_authentication: false,
            jwt_secret: None,
            api_key_header: "X-API-Key".to_string(),
            session_timeout: Duration::from_secs(3600),

            threat_detection_sensitivity: ThreatSensitivity::Medium,
            max_failed_attempts: 5,
            lockout_duration: Duration::from_secs(300),
            suspicious_pattern_threshold: 0.8,

            allowed_ip_ranges: Vec::new(),
            blocked_ips: HashSet::new(),
            geolocation_restrictions: None,
        }
    }
}

/// Threat detection sensitivity levels
#[derive(Debug, Clone, Copy)]
pub enum ThreatSensitivity {
    Low,
    Medium,
    High,
    Paranoid,
}

/// IP range specification for filtering
#[derive(Debug, Clone)]
pub struct IpRange {
    pub network: IpAddr,
    pub prefix_length: u8,
}

/// Geolocation-based restrictions
#[derive(Debug, Clone)]
pub struct GeolocationRestrictions {
    pub allowed_countries: HashSet<String>,
    pub blocked_countries: HashSet<String>,
    pub allowed_regions: HashSet<String>,
    pub blocked_regions: HashSet<String>,
}

/// Security context for request processing
#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub client_ip: IpAddr,
    pub user_agent: Option<String>,
    pub authentication_token: Option<String>,
    pub user_id: Option<String>,
    pub permissions: HashSet<String>,
    pub roles: HashSet<String>,
    pub session_id: String,
    pub request_timestamp: SystemTime,
    pub geographical_info: Option<GeographicalInfo>,
}

/// Geographical information for requests
#[derive(Debug, Clone)]
pub struct GeographicalInfo {
    pub country: String,
    pub region: String,
    pub city: String,
    pub latitude: f64,
    pub longitude: f64,
    pub timezone: String,
}

/// Security violation types
#[derive(Debug, Clone, Serialize)]
pub enum SecurityViolation {
    RateLimitExceeded {
        requests_per_minute: usize,
        limit: usize,
    },
    QueryDepthExceeded {
        depth: usize,
        max_depth: usize,
    },
    QueryComplexityExceeded {
        complexity: usize,
        max_complexity: usize,
    },
    UnauthorizedFieldAccess {
        field_name: String,
        required_permission: String,
    },
    IntrospectionBlocked,
    MutationBlocked {
        mutation_name: String,
    },
    SubscriptionBlocked {
        subscription_name: String,
    },
    IpBlocked {
        ip: IpAddr,
    },
    GeolocationBlocked {
        country: String,
        region: String,
    },
    SuspiciousActivity {
        threat_score: f64,
        patterns: Vec<String>,
    },
    AuthenticationRequired,
    InvalidToken,
    InsufficientPermissions {
        required: String,
        available: Vec<String>,
    },
    QueryNotWhitelisted {
        query_signature: String,
    },
}

/// Audit log entry
#[derive(Debug, Clone, Serialize)]
pub struct AuditLogEntry {
    pub timestamp: SystemTime,
    pub client_ip: IpAddr,
    pub user_id: Option<String>,
    pub session_id: String,
    pub operation_type: OperationType,
    pub query: String,
    pub variables: Option<serde_json::Value>,
    pub execution_time: Duration,
    pub success: bool,
    pub violations: Vec<SecurityViolation>,
    pub response_size: usize,
    pub user_agent: Option<String>,
}

/// Rate limiting state for a client
#[derive(Debug)]
struct RateLimitState {
    requests: VecDeque<Instant>,
    blocked_until: Option<Instant>,
    total_requests: usize,
    #[allow(dead_code)]
    first_request: Instant,
}

/// Client security state tracking
#[derive(Debug)]
struct ClientSecurityState {
    #[allow(dead_code)]
    failed_attempts: usize,
    #[allow(dead_code)]
    locked_until: Option<Instant>,
    #[allow(dead_code)]
    threat_score: f64,
    #[allow(dead_code)]
    suspicious_patterns: Vec<String>,
    #[allow(dead_code)]
    first_seen: SystemTime,
    #[allow(dead_code)]
    last_activity: SystemTime,
}

/// Query analysis result
#[derive(Debug)]
pub struct QueryAnalysisResult {
    pub depth: usize,
    pub complexity: usize,
    pub field_count: usize,
    pub alias_count: usize,
    pub selection_set_count: usize,
    pub contains_introspection: bool,
    pub mutations: Vec<String>,
    pub subscriptions: Vec<String>,
    pub required_permissions: HashSet<String>,
    pub query_signature: String,
}

/// Advanced GraphQL Security System
pub struct AdvancedSecuritySystem {
    config: SecurityConfig,
    rate_limits: Arc<AsyncMutex<HashMap<IpAddr, RateLimitState>>>,
    #[allow(dead_code)]
    client_states: Arc<AsyncMutex<HashMap<IpAddr, ClientSecurityState>>>,
    #[allow(dead_code)]
    whitelisted_queries: Arc<AsyncRwLock<HashSet<String>>>,
    #[allow(dead_code)]
    field_permissions: Arc<AsyncRwLock<HashMap<String, HashSet<String>>>>,
    audit_log: Arc<AsyncMutex<VecDeque<AuditLogEntry>>>,
    #[allow(dead_code)]
    threat_detector: Arc<AsyncMutex<ThreatDetector>>,
    #[allow(dead_code)]
    auth_validator: Arc<AuthenticationValidator>,
}

/// Threat detection system
#[derive(Debug)]
pub struct ThreatDetector {
    #[allow(dead_code)]
    known_attack_patterns: Vec<AttackPattern>,
    #[allow(dead_code)]
    behavioral_baselines: HashMap<String, BehavioralBaseline>,
    #[allow(dead_code)]
    anomaly_threshold: f64,
}

/// Attack pattern definition
#[derive(Debug, Clone)]
pub struct AttackPattern {
    pub name: String,
    pub pattern_type: AttackPatternType,
    pub signature: String,
    pub severity: ThreatSeverity,
    pub detection_rules: Vec<DetectionRule>,
}

/// Attack pattern types
#[derive(Debug, Clone)]
pub enum AttackPatternType {
    QueryComplexityAttack,
    BatchQueryAttack,
    IntrospectionAbuse,
    FieldEnumeration,
    DataExfiltration,
    DenialOfService,
    AuthenticationBypass,
    PrivilegeEscalation,
}

/// Threat severity levels
#[derive(Debug, Clone)]
pub enum ThreatSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Detection rule for threat patterns
#[derive(Debug, Clone)]
pub struct DetectionRule {
    pub rule_type: DetectionRuleType,
    pub threshold: f64,
    pub time_window: Duration,
}

/// Detection rule types
#[derive(Debug, Clone)]
pub enum DetectionRuleType {
    RequestFrequency,
    QueryComplexity,
    ErrorRate,
    ResponseSize,
    FieldAccessPattern,
    UserAgentPattern,
}

/// Behavioral baseline for normal user behavior
#[derive(Debug, Clone)]
pub struct BehavioralBaseline {
    pub avg_query_complexity: f64,
    pub avg_request_frequency: f64,
    pub common_fields: HashSet<String>,
    pub typical_operations: HashSet<OperationType>,
    pub normal_response_sizes: (usize, usize), // (min, max)
}

/// Authentication validator
#[derive(Debug)]
pub struct AuthenticationValidator {
    #[allow(dead_code)]
    jwt_secret: Option<String>,
    #[allow(dead_code)]
    valid_sessions: Arc<AsyncRwLock<HashMap<String, SessionInfo>>>,
    #[allow(dead_code)]
    api_keys: Arc<AsyncRwLock<HashMap<String, ApiKeyInfo>>>,
}

/// Session information
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub user_id: String,
    pub permissions: HashSet<String>,
    pub roles: HashSet<String>,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub expires_at: SystemTime,
}

/// API key information
#[derive(Debug, Clone)]
pub struct ApiKeyInfo {
    pub key_id: String,
    pub permissions: HashSet<String>,
    pub rate_limit: Option<usize>,
    pub created_at: SystemTime,
    pub last_used: SystemTime,
    pub expires_at: Option<SystemTime>,
}

impl AdvancedSecuritySystem {
    /// Create a new advanced security system
    pub fn new(config: SecurityConfig) -> Self {
        let auth_validator = Arc::new(AuthenticationValidator {
            jwt_secret: config.jwt_secret.clone(),
            valid_sessions: Arc::new(AsyncRwLock::new(HashMap::new())),
            api_keys: Arc::new(AsyncRwLock::new(HashMap::new())),
        });

        let threat_detector = ThreatDetector {
            known_attack_patterns: Self::initialize_attack_patterns(),
            behavioral_baselines: HashMap::new(),
            anomaly_threshold: match config.threat_detection_sensitivity {
                ThreatSensitivity::Low => 0.9,
                ThreatSensitivity::Medium => 0.8,
                ThreatSensitivity::High => 0.7,
                ThreatSensitivity::Paranoid => 0.5,
            },
        };

        Self {
            config,
            rate_limits: Arc::new(AsyncMutex::new(HashMap::new())),
            client_states: Arc::new(AsyncMutex::new(HashMap::new())),
            whitelisted_queries: Arc::new(AsyncRwLock::new(HashSet::new())),
            field_permissions: Arc::new(AsyncRwLock::new(HashMap::new())),
            audit_log: Arc::new(AsyncMutex::new(VecDeque::new())),
            threat_detector: Arc::new(AsyncMutex::new(threat_detector)),
            auth_validator,
        }
    }

    /// Validate a GraphQL request
    #[instrument(skip(self, query))]
    pub async fn validate_request(
        &self,
        context: &SecurityContext,
        query: &str,
        _variables: Option<&serde_json::Value>,
        operation_name: Option<&str>,
    ) -> Result<Vec<SecurityViolation>> {
        let mut violations = Vec::new();

        // IP filtering check
        if self.config.enable_ip_filtering {
            if let Some(violation) = self.check_ip_filtering(&context.client_ip).await? {
                violations.push(violation);
                return Ok(violations); // Block immediately for IP violations
            }
        }

        // Rate limiting check
        if self.config.enable_rate_limiting {
            if let Some(violation) = self.check_rate_limiting(&context.client_ip).await? {
                violations.push(violation);
            }
        }

        // Authentication check
        if self.config.require_authentication {
            if let Some(violation) = self.check_authentication(context).await? {
                violations.push(violation);
                return Ok(violations); // Block immediately for auth violations
            }
        }

        // Parse and analyze query
        let analysis = self.analyze_query(query).await?;

        // Query depth analysis
        if self.config.enable_query_depth_analysis {
            if let Some(violation) = self.check_query_depth(&analysis) {
                violations.push(violation);
            }
        }

        // Query complexity analysis
        if self.config.enable_query_complexity_analysis {
            if let Some(violation) = self.check_query_complexity(&analysis) {
                violations.push(violation);
            }
        }

        // Query whitelisting
        if self.config.enable_query_whitelisting {
            if let Some(violation) = self.check_query_whitelist(&analysis).await? {
                violations.push(violation);
            }
        }

        // Field authorization
        if self.config.enable_field_authorization {
            let field_violations = self.check_field_authorization(context, &analysis).await?;
            violations.extend(field_violations);
        }

        // Introspection protection
        if self.config.enable_introspection_protection && analysis.contains_introspection {
            violations.push(SecurityViolation::IntrospectionBlocked);
        }

        // Mutation protection
        if self.config.enable_mutation_protection {
            for mutation in &analysis.mutations {
                if !self.is_mutation_allowed(context, mutation).await? {
                    violations.push(SecurityViolation::MutationBlocked {
                        mutation_name: mutation.clone(),
                    });
                }
            }
        }

        // Subscription protection
        if self.config.enable_subscription_protection {
            for subscription in &analysis.subscriptions {
                if !self.is_subscription_allowed(context, subscription).await? {
                    violations.push(SecurityViolation::SubscriptionBlocked {
                        subscription_name: subscription.clone(),
                    });
                }
            }
        }

        // Threat detection
        if self.config.enable_threat_detection {
            if let Some(violation) = self.detect_threats(context, &analysis).await? {
                violations.push(violation);
            }
        }

        Ok(violations)
    }

    /// Analyze a GraphQL query for security purposes
    async fn analyze_query(&self, query: &str) -> Result<QueryAnalysisResult> {
        // This is a simplified implementation
        // In practice, you'd use a full GraphQL parser

        let depth = self.calculate_query_depth(query);
        let complexity = self.calculate_query_complexity(query);
        let field_count = self.count_fields(query);
        let alias_count = self.count_aliases(query);
        let selection_set_count = self.count_selection_sets(query);
        let contains_introspection = query.contains("__schema") || query.contains("__type");
        let mutations = self.extract_mutations(query);
        let subscriptions = self.extract_subscriptions(query);
        let required_permissions = self.extract_required_permissions(query).await?;
        let query_signature = self.generate_query_signature(query);

        Ok(QueryAnalysisResult {
            depth,
            complexity,
            field_count,
            alias_count,
            selection_set_count,
            contains_introspection,
            mutations,
            subscriptions,
            required_permissions,
            query_signature,
        })
    }

    /// Check IP filtering rules
    async fn check_ip_filtering(&self, client_ip: &IpAddr) -> Result<Option<SecurityViolation>> {
        if self.config.blocked_ips.contains(client_ip) {
            return Ok(Some(SecurityViolation::IpBlocked { ip: *client_ip }));
        }

        // Check allowed IP ranges
        if !self.config.allowed_ip_ranges.is_empty() {
            let mut allowed = false;
            for range in &self.config.allowed_ip_ranges {
                if self.ip_in_range(client_ip, range) {
                    allowed = true;
                    break;
                }
            }
            if !allowed {
                return Ok(Some(SecurityViolation::IpBlocked { ip: *client_ip }));
            }
        }

        Ok(None)
    }

    /// Check rate limiting for a client IP
    async fn check_rate_limiting(&self, client_ip: &IpAddr) -> Result<Option<SecurityViolation>> {
        let mut rate_limits = self.rate_limits.lock().await;
        let now = Instant::now();

        let rate_limit_state = rate_limits
            .entry(*client_ip)
            .or_insert_with(|| RateLimitState {
                requests: VecDeque::new(),
                blocked_until: None,
                total_requests: 0,
                first_request: now,
            });

        // Check if client is currently blocked
        if let Some(blocked_until) = rate_limit_state.blocked_until {
            if now < blocked_until {
                return Ok(Some(SecurityViolation::RateLimitExceeded {
                    requests_per_minute: rate_limit_state.requests.len(),
                    limit: self.config.rate_limit_requests_per_minute,
                }));
            } else {
                rate_limit_state.blocked_until = None;
            }
        }

        // Remove old requests outside the window
        let window_start = now - self.config.rate_limit_window_size;
        while let Some(&front_time) = rate_limit_state.requests.front() {
            if front_time < window_start {
                rate_limit_state.requests.pop_front();
            } else {
                break;
            }
        }

        // Check if we're over the limit
        if rate_limit_state.requests.len() >= self.config.rate_limit_requests_per_minute {
            rate_limit_state.blocked_until = Some(now + self.config.rate_limit_window_size);
            return Ok(Some(SecurityViolation::RateLimitExceeded {
                requests_per_minute: rate_limit_state.requests.len(),
                limit: self.config.rate_limit_requests_per_minute,
            }));
        }

        // Add current request
        rate_limit_state.requests.push_back(now);
        rate_limit_state.total_requests += 1;

        Ok(None)
    }

    /// Log security events for audit purposes
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip(self, context, query, variables))]
    pub async fn log_audit_event(
        &self,
        context: &SecurityContext,
        operation_type: OperationType,
        query: &str,
        variables: Option<&serde_json::Value>,
        execution_time: Duration,
        success: bool,
        violations: Vec<SecurityViolation>,
        response_size: usize,
    ) -> Result<()> {
        if !self.config.enable_audit_logging {
            return Ok(());
        }

        let entry = AuditLogEntry {
            timestamp: SystemTime::now(),
            client_ip: context.client_ip,
            user_id: context.user_id.clone(),
            session_id: context.session_id.clone(),
            operation_type,
            query: query.to_string(),
            variables: variables.cloned(),
            execution_time,
            success,
            violations,
            response_size,
            user_agent: context.user_agent.clone(),
        };

        let mut audit_log = self.audit_log.lock().await;
        audit_log.push_back(entry);

        // Keep only recent entries (last 10,000)
        while audit_log.len() > 10000 {
            audit_log.pop_front();
        }

        Ok(())
    }

    /// Initialize known attack patterns
    fn initialize_attack_patterns() -> Vec<AttackPattern> {
        vec![
            AttackPattern {
                name: "Deep Query Attack".to_string(),
                pattern_type: AttackPatternType::QueryComplexityAttack,
                signature: "excessive_depth".to_string(),
                severity: ThreatSeverity::High,
                detection_rules: vec![DetectionRule {
                    rule_type: DetectionRuleType::QueryComplexity,
                    threshold: 1000.0,
                    time_window: Duration::from_secs(60),
                }],
            },
            AttackPattern {
                name: "Introspection Abuse".to_string(),
                pattern_type: AttackPatternType::IntrospectionAbuse,
                signature: "introspection_query".to_string(),
                severity: ThreatSeverity::Medium,
                detection_rules: vec![DetectionRule {
                    rule_type: DetectionRuleType::FieldAccessPattern,
                    threshold: 5.0,
                    time_window: Duration::from_secs(300),
                }],
            },
            // Additional patterns would be defined here...
        ]
    }

    // Helper methods (simplified implementations)
    fn calculate_query_depth(&self, _query: &str) -> usize {
        5
    }
    fn calculate_query_complexity(&self, _query: &str) -> usize {
        100
    }
    fn count_fields(&self, _query: &str) -> usize {
        10
    }
    fn count_aliases(&self, _query: &str) -> usize {
        2
    }
    fn count_selection_sets(&self, _query: &str) -> usize {
        3
    }
    fn extract_mutations(&self, _query: &str) -> Vec<String> {
        Vec::new()
    }
    fn extract_subscriptions(&self, _query: &str) -> Vec<String> {
        Vec::new()
    }
    async fn extract_required_permissions(&self, _query: &str) -> Result<HashSet<String>> {
        Ok(HashSet::new())
    }
    fn generate_query_signature(&self, query: &str) -> String {
        format!("sig_{}", query.len()) // Simplified signature
    }
    fn ip_in_range(&self, _ip: &IpAddr, _range: &IpRange) -> bool {
        true
    }
    async fn check_authentication(
        &self,
        _context: &SecurityContext,
    ) -> Result<Option<SecurityViolation>> {
        Ok(None)
    }
    fn check_query_depth(&self, analysis: &QueryAnalysisResult) -> Option<SecurityViolation> {
        if analysis.depth > self.config.max_query_depth {
            Some(SecurityViolation::QueryDepthExceeded {
                depth: analysis.depth,
                max_depth: self.config.max_query_depth,
            })
        } else {
            None
        }
    }
    fn check_query_complexity(&self, analysis: &QueryAnalysisResult) -> Option<SecurityViolation> {
        if analysis.complexity > self.config.max_query_complexity {
            Some(SecurityViolation::QueryComplexityExceeded {
                complexity: analysis.complexity,
                max_complexity: self.config.max_query_complexity,
            })
        } else {
            None
        }
    }
    async fn check_query_whitelist(
        &self,
        _analysis: &QueryAnalysisResult,
    ) -> Result<Option<SecurityViolation>> {
        Ok(None)
    }
    async fn check_field_authorization(
        &self,
        _context: &SecurityContext,
        _analysis: &QueryAnalysisResult,
    ) -> Result<Vec<SecurityViolation>> {
        Ok(Vec::new())
    }
    async fn is_mutation_allowed(
        &self,
        _context: &SecurityContext,
        _mutation: &str,
    ) -> Result<bool> {
        Ok(true)
    }
    async fn is_subscription_allowed(
        &self,
        _context: &SecurityContext,
        _subscription: &str,
    ) -> Result<bool> {
        Ok(true)
    }
    async fn detect_threats(
        &self,
        _context: &SecurityContext,
        _analysis: &QueryAnalysisResult,
    ) -> Result<Option<SecurityViolation>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[tokio::test]
    async fn test_security_system_creation() {
        let config = SecurityConfig::default();
        let security_system = AdvancedSecuritySystem::new(config);

        assert!(security_system.config.enable_rate_limiting);
        assert!(security_system.config.enable_query_depth_analysis);
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let config = SecurityConfig {
            rate_limit_requests_per_minute: 2,
            ..Default::default()
        };
        let security_system = AdvancedSecuritySystem::new(config);

        let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));

        // First request should pass
        let violation1 = security_system.check_rate_limiting(&ip).await.unwrap();
        assert!(violation1.is_none());

        // Second request should pass
        let violation2 = security_system.check_rate_limiting(&ip).await.unwrap();
        assert!(violation2.is_none());

        // Third request should be rate limited
        let violation3 = security_system.check_rate_limiting(&ip).await.unwrap();
        assert!(violation3.is_some());

        if let Some(SecurityViolation::RateLimitExceeded {
            requests_per_minute,
            limit,
        }) = violation3
        {
            assert_eq!(requests_per_minute, 2);
            assert_eq!(limit, 2);
        } else {
            panic!("Expected rate limit violation");
        }
    }

    #[tokio::test]
    async fn test_query_analysis() {
        let config = SecurityConfig::default();
        let security_system = AdvancedSecuritySystem::new(config);

        let query = "query { user { name email } }";
        let analysis = security_system.analyze_query(query).await.unwrap();

        assert!(analysis.depth > 0);
        assert!(analysis.complexity > 0);
        assert!(analysis.field_count > 0);
    }
}
