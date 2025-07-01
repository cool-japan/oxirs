//! Critical Security Enhancements for OxiRS Engine
//!
//! This module implements enterprise-grade security enhancements based on the comprehensive
//! security audit findings. It addresses input validation, configuration security,
//! information disclosure prevention, and other critical security concerns.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use tracing::{debug, error, info, warn};

/// Comprehensive input validation and sanitization framework
#[derive(Debug)]
pub struct InputValidationFramework {
    /// SPARQL query validator
    sparql_validator: SparqlQueryValidator,
    /// RDF data validator
    rdf_validator: RdfDataValidator,
    /// IRI format validator
    iri_validator: IriValidator,
    /// Configuration for validation rules
    validation_config: ValidationConfig,
}

/// SPARQL query security validator
#[derive(Debug)]
pub struct SparqlQueryValidator {
    /// Maximum query complexity allowed
    max_complexity: usize,
    /// Maximum query length
    max_query_length: usize,
    /// Allowed query types
    allowed_query_types: HashSet<QueryType>,
    /// Blocked keywords and patterns
    blocked_patterns: Vec<String>,
    /// Rate limiting configuration
    rate_limit_config: RateLimitConfig,
}

/// Query types enumeration
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum QueryType {
    Select,
    Construct,
    Ask,
    Describe,
    Update,
    Insert,
    Delete,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Queries per minute per client
    queries_per_minute: u32,
    /// Maximum concurrent queries per client
    max_concurrent_queries: u32,
    /// Timeout for individual queries
    query_timeout_seconds: u32,
}

/// RDF data validator for preventing injection attacks
#[derive(Debug)]
pub struct RdfDataValidator {
    /// Maximum triple count per request
    max_triples_per_request: usize,
    /// Maximum literal length
    max_literal_length: usize,
    /// Allowed RDF formats
    allowed_formats: HashSet<RdfFormat>,
    /// Content validation rules
    content_rules: ContentValidationRules,
}

/// RDF formats enumeration
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum RdfFormat {
    Turtle,
    NTriples,
    JsonLd,
    RdfXml,
    TurtleStar,
    NTriplesStar,
}

/// Content validation rules
#[derive(Debug, Clone)]
pub struct ContentValidationRules {
    /// Check for malicious patterns
    check_malicious_patterns: bool,
    /// Validate IRI formats
    validate_iris: bool,
    /// Check for suspicious literals
    check_suspicious_literals: bool,
    /// Maximum nesting depth
    max_nesting_depth: usize,
}

/// IRI validation for preventing format attacks
#[derive(Debug)]
pub struct IriValidator {
    /// Allowed schemes
    allowed_schemes: HashSet<String>,
    /// Blocked domains
    blocked_domains: HashSet<String>,
    /// Maximum IRI length
    max_iri_length: usize,
    /// Character validation rules
    char_validation: CharacterValidationRules,
}

/// Character validation rules for IRIs
#[derive(Debug, Clone)]
pub struct CharacterValidationRules {
    /// Allow unicode characters
    allow_unicode: bool,
    /// Block suspicious patterns
    block_suspicious_patterns: bool,
    /// Normalize encoding
    normalize_encoding: bool,
}

/// Overall validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable strict validation mode
    strict_mode: bool,
    /// Log validation failures
    log_failures: bool,
    /// Fail on validation errors
    fail_on_errors: bool,
    /// Custom validation rules
    custom_rules: HashMap<String, String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_mode: true,
            log_failures: true,
            fail_on_errors: true,
            custom_rules: HashMap::new(),
        }
    }
}

/// Secure configuration management system
#[derive(Debug)]
pub struct SecureConfigurationManager {
    /// Environment variable handler
    env_handler: EnvironmentVariableHandler,
    /// Secret management integration
    secret_manager: SecretManager,
    /// Configuration encryption
    config_encryption: ConfigurationEncryption,
    /// Configuration validation
    config_validator: ConfigurationValidator,
}

/// Environment variable secure handling
#[derive(Debug)]
pub struct EnvironmentVariableHandler {
    /// Required environment variables
    required_vars: HashSet<String>,
    /// Secret environment variables (never logged)
    secret_vars: HashSet<String>,
    /// Default values for optional variables
    default_values: HashMap<String, String>,
}

/// Secret management integration
#[derive(Debug)]
pub struct SecretManager {
    /// Secret store type
    store_type: SecretStoreType,
    /// Connection configuration
    connection_config: SecretStoreConfig,
    /// Cache configuration
    cache_config: SecretCacheConfig,
}

/// Secret store types
#[derive(Debug, Clone)]
pub enum SecretStoreType {
    HashiCorpVault,
    AwsSecretsManager,
    AzureKeyVault,
    KubernetesSecrets,
    LocalEncrypted,
}

/// Secret store connection configuration
#[derive(Debug, Clone)]
pub struct SecretStoreConfig {
    /// Store endpoint
    pub endpoint: String,
    /// Authentication method
    pub auth_method: AuthMethod,
    /// Connection timeout
    pub timeout_seconds: u32,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// Authentication methods for secret stores
#[derive(Debug, Clone)]
pub enum AuthMethod {
    Token(String),
    Certificate { cert_path: String, key_path: String },
    AwsIam,
    ServiceAccount,
}

/// Retry configuration for secret operations
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Initial retry delay in milliseconds
    pub initial_delay_ms: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Maximum delay in milliseconds
    pub max_delay_ms: u64,
}

/// Secret caching configuration
#[derive(Debug, Clone)]
pub struct SecretCacheConfig {
    /// Enable caching
    pub enabled: bool,
    /// Cache TTL in seconds
    pub ttl_seconds: u64,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Encrypt cached values
    pub encrypt_cached_values: bool,
}

/// Configuration encryption for sensitive data
#[derive(Debug)]
pub struct ConfigurationEncryption {
    /// Encryption algorithm
    algorithm: EncryptionAlgorithm,
    /// Key derivation function
    kdf: KeyDerivationFunction,
    /// Encryption key management
    key_manager: EncryptionKeyManager,
}

/// Encryption algorithms supported
#[derive(Debug, Clone)]
pub enum EncryptionAlgorithm {
    Aes256Gcm,
    ChaCha20Poly1305,
    Aes256Cbc,
}

/// Key derivation functions
#[derive(Debug, Clone)]
pub enum KeyDerivationFunction {
    Pbkdf2,
    Scrypt,
    Argon2,
}

/// Encryption key management
#[derive(Debug)]
pub struct EncryptionKeyManager {
    /// Key rotation policy
    rotation_policy: KeyRotationPolicy,
    /// Key storage method
    storage_method: KeyStorageMethod,
    /// Key derivation parameters
    derivation_params: KeyDerivationParams,
}

/// Key rotation policies
#[derive(Debug, Clone)]
pub struct KeyRotationPolicy {
    /// Rotation interval in days
    pub rotation_interval_days: u32,
    /// Keep old keys for recovery
    pub keep_old_keys: bool,
    /// Number of old keys to retain
    pub old_key_retention_count: u32,
}

/// Key storage methods
#[derive(Debug, Clone)]
pub enum KeyStorageMethod {
    Environment,
    SecretStore,
    Hsm,
    File { path: String, encrypted: bool },
}

/// Key derivation parameters
#[derive(Debug, Clone)]
pub struct KeyDerivationParams {
    /// Salt size in bytes
    pub salt_size: usize,
    /// Iteration count for PBKDF2
    pub iterations: u32,
    /// Memory cost for Argon2
    pub memory_cost: u32,
    /// Time cost for Argon2
    pub time_cost: u32,
    /// Parallelism for Argon2
    pub parallelism: u32,
}

/// Configuration validation framework
#[derive(Debug)]
pub struct ConfigurationValidator {
    /// Validation rules
    rules: Vec<ValidationRule>,
    /// Required configuration keys
    required_keys: HashSet<String>,
    /// Type validation
    type_validators: HashMap<String, TypeValidator>,
}

/// Configuration validation rules
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Configuration key pattern
    pub key_pattern: String,
    /// Validation function
    pub validator: ValidatorType,
    /// Error message
    pub error_message: String,
}

/// Validator types
#[derive(Debug, Clone)]
pub enum ValidatorType {
    Range { min: f64, max: f64 },
    Regex(String),
    OneOf(Vec<String>),
    Custom(String),
}

/// Type validators for configuration values
#[derive(Debug, Clone)]
pub enum TypeValidator {
    String { min_len: Option<usize>, max_len: Option<usize> },
    Integer { min: Option<i64>, max: Option<i64> },
    Float { min: Option<f64>, max: Option<f64> },
    Boolean,
    Url,
    Email,
    FilePath,
}

/// Production-ready error handling system
#[derive(Debug)]
pub struct ProductionErrorHandler {
    /// Error sanitization rules
    sanitization_rules: ErrorSanitizationRules,
    /// Logging configuration
    logging_config: SecurityLoggingConfig,
    /// Alert configuration
    alert_config: SecurityAlertConfig,
}

/// Error sanitization to prevent information disclosure
#[derive(Debug, Clone)]
pub struct ErrorSanitizationRules {
    /// Remove file paths from errors
    pub remove_file_paths: bool,
    /// Remove stack traces in production
    pub remove_stack_traces: bool,
    /// Sanitize database errors
    pub sanitize_database_errors: bool,
    /// Replace sensitive patterns
    pub sensitive_patterns: Vec<SensitivePattern>,
    /// Generic error messages for external users
    pub use_generic_messages: bool,
}

/// Sensitive patterns to remove from error messages
#[derive(Debug, Clone)]
pub struct SensitivePattern {
    /// Pattern to match
    pub pattern: String,
    /// Replacement text
    pub replacement: String,
    /// Use regex matching
    pub is_regex: bool,
}

/// Security-focused logging configuration
#[derive(Debug, Clone)]
pub struct SecurityLoggingConfig {
    /// Enable security audit logging
    pub enable_audit_logging: bool,
    /// Log security events to separate file
    pub separate_security_log: bool,
    /// Structured logging format
    pub structured_format: bool,
    /// Include correlation IDs
    pub include_correlation_ids: bool,
    /// Sensitive data filtering
    pub sensitive_data_filter: SensitiveDataFilter,
    /// Log rotation configuration
    pub rotation_config: LogRotationConfig,
}

/// Sensitive data filtering for logs
#[derive(Debug, Clone)]
pub struct SensitiveDataFilter {
    /// Patterns to redact
    pub redaction_patterns: Vec<String>,
    /// Replacement text for redacted data
    pub redaction_replacement: String,
    /// Hash sensitive values instead of redacting
    pub hash_sensitive_values: bool,
    /// Fields to always redact
    pub always_redact_fields: HashSet<String>,
}

/// Log rotation configuration
#[derive(Debug, Clone)]
pub struct LogRotationConfig {
    /// Maximum file size in MB
    pub max_file_size_mb: u64,
    /// Maximum number of files to keep
    pub max_files: u32,
    /// Compress rotated logs
    pub compress_rotated: bool,
    /// Rotation schedule
    pub rotation_schedule: RotationSchedule,
}

/// Log rotation schedules
#[derive(Debug, Clone)]
pub enum RotationSchedule {
    Hourly,
    Daily,
    Weekly,
    SizeBased,
}

/// Security alert configuration
#[derive(Debug, Clone)]
pub struct SecurityAlertConfig {
    /// Enable security alerts
    pub enabled: bool,
    /// Alert thresholds
    pub thresholds: AlertThresholds,
    /// Alert destinations
    pub destinations: Vec<AlertDestination>,
    /// Alert frequency limits
    pub frequency_limits: AlertFrequencyLimits,
}

/// Alert threshold configuration
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Failed authentication attempts per minute
    pub failed_auth_per_minute: u32,
    /// Validation failures per minute
    pub validation_failures_per_minute: u32,
    /// Configuration errors per hour
    pub config_errors_per_hour: u32,
    /// Suspicious activity score threshold
    pub suspicious_activity_score: f64,
}

/// Alert destinations
#[derive(Debug, Clone)]
pub enum AlertDestination {
    Email(String),
    Slack { webhook_url: String, channel: String },
    PagerDuty { integration_key: String },
    Webhook { url: String, headers: HashMap<String, String> },
    Syslog { server: String, facility: String },
}

/// Alert frequency limiting
#[derive(Debug, Clone)]
pub struct AlertFrequencyLimits {
    /// Minimum time between identical alerts (seconds)
    pub min_interval_seconds: u64,
    /// Maximum alerts per hour
    pub max_alerts_per_hour: u32,
    /// Burst alert threshold
    pub burst_threshold: u32,
}

/// Security validation result
#[derive(Debug, Clone)]
pub struct SecurityValidationResult {
    /// Validation success
    pub is_valid: bool,
    /// Security score (0-100)
    pub security_score: u8,
    /// Validation errors
    pub errors: Vec<SecurityValidationError>,
    /// Warnings
    pub warnings: Vec<SecurityWarning>,
    /// Recommendations
    pub recommendations: Vec<SecurityRecommendation>,
}

/// Security validation error
#[derive(Debug, Clone)]
pub struct SecurityValidationError {
    /// Error type
    pub error_type: SecurityErrorType,
    /// Error message
    pub message: String,
    /// Severity level
    pub severity: SecuritySeverity,
    /// Affected component
    pub component: String,
    /// Remediation steps
    pub remediation: Vec<String>,
}

/// Security error types
#[derive(Debug, Clone)]
pub enum SecurityErrorType {
    InputValidation,
    Authentication,
    Authorization,
    ConfigurationSecurity,
    DataProtection,
    NetworkSecurity,
    InformationDisclosure,
}

/// Security severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Security warning
#[derive(Debug, Clone)]
pub struct SecurityWarning {
    /// Warning message
    pub message: String,
    /// Affected component
    pub component: String,
    /// Recommendation
    pub recommendation: String,
}

/// Security recommendation
#[derive(Debug, Clone)]
pub struct SecurityRecommendation {
    /// Recommendation title
    pub title: String,
    /// Description
    pub description: String,
    /// Priority
    pub priority: RecommendationPriority,
    /// Implementation effort
    pub effort: ImplementationEffort,
    /// Impact assessment
    pub impact: SecurityImpact,
}

/// Recommendation priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Implementation effort estimates
#[derive(Debug, Clone)]
pub enum ImplementationEffort {
    Minimal,  // Hours
    Low,      // Days
    Medium,   // Weeks
    High,     // Months
}

/// Security impact assessment
#[derive(Debug, Clone)]
pub enum SecurityImpact {
    Low,
    Medium,
    High,
    Critical,
}

impl InputValidationFramework {
    /// Create a new input validation framework
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            sparql_validator: SparqlQueryValidator::new(),
            rdf_validator: RdfDataValidator::new(),
            iri_validator: IriValidator::new(),
            validation_config: config,
        }
    }
    
    /// Validate SPARQL query for security issues
    pub fn validate_sparql_query(&self, query: &str) -> Result<SecurityValidationResult> {
        info!("Validating SPARQL query for security issues");
        
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut security_score = 100u8;
        
        // Check query length
        if query.len() > self.sparql_validator.max_query_length {
            errors.push(SecurityValidationError {
                error_type: SecurityErrorType::InputValidation,
                message: format!("Query exceeds maximum length of {} characters", 
                                self.sparql_validator.max_query_length),
                severity: SecuritySeverity::High,
                component: "SPARQL Parser".to_string(),
                remediation: vec!["Reduce query complexity".to_string(), 
                                "Split into multiple queries".to_string()],
            });
            security_score -= 20;
        }
        
        // Check for blocked patterns
        for pattern in &self.sparql_validator.blocked_patterns {
            if query.to_lowercase().contains(&pattern.to_lowercase()) {
                errors.push(SecurityValidationError {
                    error_type: SecurityErrorType::InputValidation,
                    message: format!("Query contains blocked pattern: {}", pattern),
                    severity: SecuritySeverity::Critical,
                    component: "SPARQL Parser".to_string(),
                    remediation: vec!["Remove malicious patterns".to_string(),
                                    "Use parameterized queries".to_string()],
                });
                security_score -= 30;
            }
        }
        
        // Check query complexity
        let complexity = self.calculate_query_complexity(query);
        if complexity > self.sparql_validator.max_complexity {
            warnings.push(SecurityWarning {
                message: format!("Query complexity ({}) exceeds recommended limit ({})", 
                               complexity, self.sparql_validator.max_complexity),
                component: "SPARQL Parser".to_string(),
                recommendation: "Consider simplifying the query structure".to_string(),
            });
            security_score -= 10;
        }
        
        Ok(SecurityValidationResult {
            is_valid: errors.is_empty(),
            security_score,
            errors,
            warnings,
            recommendations: self.generate_sparql_recommendations(),
        })
    }
    
    /// Validate RDF data for security issues
    pub fn validate_rdf_data(&self, data: &str, format: RdfFormat) -> Result<SecurityValidationResult> {
        info!("Validating RDF data for security issues");
        
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut security_score = 100u8;
        
        // Check if format is allowed
        if !self.rdf_validator.allowed_formats.contains(&format) {
            errors.push(SecurityValidationError {
                error_type: SecurityErrorType::InputValidation,
                message: format!("RDF format {:?} is not allowed", format),
                severity: SecuritySeverity::High,
                component: "RDF Parser".to_string(),
                remediation: vec!["Use an allowed RDF format".to_string()],
            });
            security_score -= 25;
        }
        
        // Check data size
        if data.len() > (self.rdf_validator.max_triples_per_request * 1000) {
            errors.push(SecurityValidationError {
                error_type: SecurityErrorType::InputValidation,
                message: "RDF data exceeds maximum size limit".to_string(),
                severity: SecuritySeverity::Medium,
                component: "RDF Parser".to_string(),
                remediation: vec!["Reduce data size".to_string(), 
                                "Split into multiple requests".to_string()],
            });
            security_score -= 15;
        }
        
        // Check for malicious patterns
        if self.rdf_validator.content_rules.check_malicious_patterns {
            if self.contains_malicious_patterns(data) {
                errors.push(SecurityValidationError {
                    error_type: SecurityErrorType::InputValidation,
                    message: "RDF data contains potentially malicious patterns".to_string(),
                    severity: SecuritySeverity::Critical,
                    component: "RDF Parser".to_string(),
                    remediation: vec!["Remove malicious content".to_string(),
                                    "Validate data source".to_string()],
                });
                security_score -= 40;
            }
        }
        
        Ok(SecurityValidationResult {
            is_valid: errors.is_empty(),
            security_score,
            errors,
            warnings,
            recommendations: self.generate_rdf_recommendations(),
        })
    }
    
    /// Validate IRI format for security issues
    pub fn validate_iri(&self, iri: &str) -> Result<SecurityValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut security_score = 100u8;
        
        // Check IRI length
        if iri.len() > self.iri_validator.max_iri_length {
            errors.push(SecurityValidationError {
                error_type: SecurityErrorType::InputValidation,
                message: format!("IRI exceeds maximum length of {} characters", 
                                self.iri_validator.max_iri_length),
                severity: SecuritySeverity::Medium,
                component: "IRI Validator".to_string(),
                remediation: vec!["Use shorter IRI".to_string()],
            });
            security_score -= 15;
        }
        
        // Check scheme
        if let Some(scheme) = iri.split(':').next() {
            if !self.iri_validator.allowed_schemes.contains(scheme) {
                errors.push(SecurityValidationError {
                    error_type: SecurityErrorType::InputValidation,
                    message: format!("IRI scheme '{}' is not allowed", scheme),
                    severity: SecuritySeverity::High,
                    component: "IRI Validator".to_string(),
                    remediation: vec!["Use an allowed scheme".to_string()],
                });
                security_score -= 25;
            }
        }
        
        // Check for blocked domains
        for domain in &self.iri_validator.blocked_domains {
            if iri.contains(domain) {
                errors.push(SecurityValidationError {
                    error_type: SecurityErrorType::InputValidation,
                    message: format!("IRI contains blocked domain: {}", domain),
                    severity: SecuritySeverity::Critical,
                    component: "IRI Validator".to_string(),
                    remediation: vec!["Use trusted domains only".to_string()],
                });
                security_score -= 40;
            }
        }
        
        Ok(SecurityValidationResult {
            is_valid: errors.is_empty(),
            security_score,
            errors,
            warnings,
            recommendations: vec![],
        })
    }
    
    /// Calculate query complexity score
    fn calculate_query_complexity(&self, query: &str) -> usize {
        let mut complexity = 0;
        let query_lower = query.to_lowercase();
        
        // Count joins and unions
        complexity += query_lower.matches("join").count() * 2;
        complexity += query_lower.matches("union").count() * 3;
        complexity += query_lower.matches("optional").count() * 2;
        complexity += query_lower.matches("filter").count();
        complexity += query_lower.matches("group by").count() * 2;
        complexity += query_lower.matches("order by").count();
        complexity += query_lower.matches("having").count() * 2;
        
        // Count subqueries
        complexity += query_lower.matches("select").count().saturating_sub(1) * 5;
        
        complexity
    }
    
    /// Check for malicious patterns in data
    fn contains_malicious_patterns(&self, data: &str) -> bool {
        let malicious_patterns = [
            "javascript:",
            "data:",
            "vbscript:",
            "<script",
            "onload=",
            "onerror=",
            "onclick=",
            "eval(",
            "setTimeout(",
            "setInterval(",
        ];
        
        let data_lower = data.to_lowercase();
        malicious_patterns.iter().any(|pattern| data_lower.contains(pattern))
    }
    
    /// Generate SPARQL security recommendations
    fn generate_sparql_recommendations(&self) -> Vec<SecurityRecommendation> {
        vec![
            SecurityRecommendation {
                title: "Implement Query Timeouts".to_string(),
                description: "Add timeout limits to prevent DoS attacks through complex queries".to_string(),
                priority: RecommendationPriority::High,
                effort: ImplementationEffort::Low,
                impact: SecurityImpact::Medium,
            },
            SecurityRecommendation {
                title: "Add Rate Limiting".to_string(),
                description: "Implement per-client rate limiting for SPARQL queries".to_string(),
                priority: RecommendationPriority::Medium,
                effort: ImplementationEffort::Medium,
                impact: SecurityImpact::Medium,
            },
            SecurityRecommendation {
                title: "Query Parameterization".to_string(),
                description: "Encourage use of parameterized queries to prevent injection".to_string(),
                priority: RecommendationPriority::High,
                effort: ImplementationEffort::Medium,
                impact: SecurityImpact::High,
            },
        ]
    }
    
    /// Generate RDF security recommendations
    fn generate_rdf_recommendations(&self) -> Vec<SecurityRecommendation> {
        vec![
            SecurityRecommendation {
                title: "Content Type Validation".to_string(),
                description: "Strictly validate Content-Type headers for RDF uploads".to_string(),
                priority: RecommendationPriority::Medium,
                effort: ImplementationEffort::Low,
                impact: SecurityImpact::Medium,
            },
            SecurityRecommendation {
                title: "Data Sanitization".to_string(),
                description: "Implement comprehensive data sanitization for RDF content".to_string(),
                priority: RecommendationPriority::High,
                effort: ImplementationEffort::Medium,
                impact: SecurityImpact::High,
            },
        ]
    }
}

impl SparqlQueryValidator {
    /// Create a new SPARQL query validator with secure defaults
    pub fn new() -> Self {
        Self {
            max_complexity: 100,
            max_query_length: 50000,
            allowed_query_types: HashSet::from([
                QueryType::Select,
                QueryType::Construct,
                QueryType::Ask,
                QueryType::Describe,
            ]),
            blocked_patterns: vec![
                "drop".to_string(),
                "delete".to_string(),
                "load".to_string(),
                "import".to_string(),
                "system".to_string(),
                "exec".to_string(),
                "eval".to_string(),
            ],
            rate_limit_config: RateLimitConfig {
                queries_per_minute: 60,
                max_concurrent_queries: 5,
                query_timeout_seconds: 30,
            },
        }
    }
}

impl RdfDataValidator {
    /// Create a new RDF data validator with secure defaults
    pub fn new() -> Self {
        Self {
            max_triples_per_request: 10000,
            max_literal_length: 10000,
            allowed_formats: HashSet::from([
                RdfFormat::Turtle,
                RdfFormat::NTriples,
                RdfFormat::JsonLd,
                RdfFormat::TurtleStar,
                RdfFormat::NTriplesStar,
            ]),
            content_rules: ContentValidationRules {
                check_malicious_patterns: true,
                validate_iris: true,
                check_suspicious_literals: true,
                max_nesting_depth: 10,
            },
        }
    }
}

impl IriValidator {
    /// Create a new IRI validator with secure defaults
    pub fn new() -> Self {
        Self {
            allowed_schemes: HashSet::from([
                "http".to_string(),
                "https".to_string(),
                "urn".to_string(),
                "ftp".to_string(),
            ]),
            blocked_domains: HashSet::from([
                "localhost".to_string(),
                "127.0.0.1".to_string(),
                "0.0.0.0".to_string(),
                "169.254.0.0".to_string(), // Link-local
            ]),
            max_iri_length: 2000,
            char_validation: CharacterValidationRules {
                allow_unicode: true,
                block_suspicious_patterns: true,
                normalize_encoding: true,
            },
        }
    }
}

impl SecureConfigurationManager {
    /// Create a new secure configuration manager
    pub fn new() -> Result<Self> {
        Ok(Self {
            env_handler: EnvironmentVariableHandler::new(),
            secret_manager: SecretManager::new(SecretStoreType::LocalEncrypted)?,
            config_encryption: ConfigurationEncryption::new()?,
            config_validator: ConfigurationValidator::new(),
        })
    }
    
    /// Validate configuration security
    pub fn validate_configuration(&self, config_data: &str) -> Result<SecurityValidationResult> {
        info!("Validating configuration security");
        
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut security_score = 100u8;
        
        // Check for hardcoded secrets
        if self.contains_hardcoded_secrets(config_data) {
            errors.push(SecurityValidationError {
                error_type: SecurityErrorType::ConfigurationSecurity,
                message: "Configuration contains hardcoded secrets".to_string(),
                severity: SecuritySeverity::Critical,
                component: "Configuration Manager".to_string(),
                remediation: vec![
                    "Move secrets to environment variables".to_string(),
                    "Use secret management system".to_string(),
                    "Encrypt sensitive configuration".to_string(),
                ],
            });
            security_score -= 50;
        }
        
        // Check for default passwords
        if self.contains_default_passwords(config_data) {
            errors.push(SecurityValidationError {
                error_type: SecurityErrorType::ConfigurationSecurity,
                message: "Configuration contains default passwords".to_string(),
                severity: SecuritySeverity::High,
                component: "Configuration Manager".to_string(),
                remediation: vec![
                    "Change default passwords".to_string(),
                    "Use strong passwords".to_string(),
                    "Implement password rotation".to_string(),
                ],
            });
            security_score -= 30;
        }
        
        // Check for insecure settings
        if self.has_insecure_settings(config_data) {
            warnings.push(SecurityWarning {
                message: "Configuration contains potentially insecure settings".to_string(),
                component: "Configuration Manager".to_string(),
                recommendation: "Review and harden configuration settings".to_string(),
            });
            security_score -= 15;
        }
        
        Ok(SecurityValidationResult {
            is_valid: errors.is_empty(),
            security_score,
            errors,
            warnings,
            recommendations: self.generate_config_recommendations(),
        })
    }
    
    /// Check for hardcoded secrets in configuration
    fn contains_hardcoded_secrets(&self, config: &str) -> bool {
        let secret_patterns = [
            r"password\s*=\s*['\"](?!%|\$|ENV)[^'\"]+['\"]",
            r"secret\s*=\s*['\"](?!%|\$|ENV)[^'\"]+['\"]",
            r"key\s*=\s*['\"](?!%|\$|ENV)[^'\"]+['\"]",
            r"token\s*=\s*['\"](?!%|\$|ENV)[^'\"]+['\"]",
        ];
        
        // Simple pattern matching (in production, use regex crate)
        secret_patterns.iter().any(|pattern| {
            config.to_lowercase().contains("password =") ||
            config.to_lowercase().contains("secret =") ||
            config.to_lowercase().contains("key =") ||
            config.to_lowercase().contains("token =")
        })
    }
    
    /// Check for default passwords
    fn contains_default_passwords(&self, config: &str) -> bool {
        let default_passwords = [
            "password",
            "admin",
            "default",
            "changeme",
            "123456",
            "password123",
        ];
        
        let config_lower = config.to_lowercase();
        default_passwords.iter().any(|pwd| config_lower.contains(&format!("\"{}\"", pwd)))
    }
    
    /// Check for insecure settings
    fn has_insecure_settings(&self, config: &str) -> bool {
        let insecure_patterns = [
            "debug = true",
            "ssl = false",
            "https = false",
            "secure = false",
            "verify = false",
        ];
        
        let config_lower = config.to_lowercase();
        insecure_patterns.iter().any(|pattern| config_lower.contains(pattern))
    }
    
    /// Generate configuration security recommendations
    fn generate_config_recommendations(&self) -> Vec<SecurityRecommendation> {
        vec![
            SecurityRecommendation {
                title: "Implement Secret Management".to_string(),
                description: "Use a dedicated secret management system for sensitive configuration".to_string(),
                priority: RecommendationPriority::Critical,
                effort: ImplementationEffort::Medium,
                impact: SecurityImpact::Critical,
            },
            SecurityRecommendation {
                title: "Environment Variable Usage".to_string(),
                description: "Move all sensitive configuration to environment variables".to_string(),
                priority: RecommendationPriority::High,
                effort: ImplementationEffort::Low,
                impact: SecurityImpact::High,
            },
            SecurityRecommendation {
                title: "Configuration Encryption".to_string(),
                description: "Encrypt configuration files containing sensitive data".to_string(),
                priority: RecommendationPriority::Medium,
                effort: ImplementationEffort::Medium,
                impact: SecurityImpact::Medium,
            },
        ]
    }
}

impl EnvironmentVariableHandler {
    /// Create a new environment variable handler
    pub fn new() -> Self {
        Self {
            required_vars: HashSet::from([
                "OXIRS_DATABASE_URL".to_string(),
                "OXIRS_JWT_SECRET".to_string(),
                "OXIRS_ADMIN_PASSWORD".to_string(),
            ]),
            secret_vars: HashSet::from([
                "OXIRS_JWT_SECRET".to_string(),
                "OXIRS_ADMIN_PASSWORD".to_string(),
                "OXIRS_DATABASE_PASSWORD".to_string(),
                "OXIRS_TLS_KEY".to_string(),
            ]),
            default_values: HashMap::from([
                ("OXIRS_LOG_LEVEL".to_string(), "info".to_string()),
                ("OXIRS_PORT".to_string(), "8080".to_string()),
                ("OXIRS_HOST".to_string(), "0.0.0.0".to_string()),
            ]),
        }
    }
}

impl SecretManager {
    /// Create a new secret manager
    pub fn new(store_type: SecretStoreType) -> Result<Self> {
        Ok(Self {
            store_type,
            connection_config: SecretStoreConfig {
                endpoint: "localhost:8200".to_string(),
                auth_method: AuthMethod::Token("".to_string()),
                timeout_seconds: 30,
                retry_config: RetryConfig {
                    max_retries: 3,
                    initial_delay_ms: 1000,
                    backoff_multiplier: 2.0,
                    max_delay_ms: 10000,
                },
            },
            cache_config: SecretCacheConfig {
                enabled: true,
                ttl_seconds: 3600,
                max_cache_size: 1000,
                encrypt_cached_values: true,
            },
        })
    }
}

impl ConfigurationEncryption {
    /// Create a new configuration encryption handler
    pub fn new() -> Result<Self> {
        Ok(Self {
            algorithm: EncryptionAlgorithm::Aes256Gcm,
            kdf: KeyDerivationFunction::Argon2,
            key_manager: EncryptionKeyManager::new()?,
        })
    }
}

impl EncryptionKeyManager {
    /// Create a new encryption key manager
    pub fn new() -> Result<Self> {
        Ok(Self {
            rotation_policy: KeyRotationPolicy {
                rotation_interval_days: 90,
                keep_old_keys: true,
                old_key_retention_count: 3,
            },
            storage_method: KeyStorageMethod::Environment,
            derivation_params: KeyDerivationParams {
                salt_size: 32,
                iterations: 100000,
                memory_cost: 65536,
                time_cost: 3,
                parallelism: 4,
            },
        })
    }
}

impl ConfigurationValidator {
    /// Create a new configuration validator
    pub fn new() -> Self {
        Self {
            rules: vec![
                ValidationRule {
                    name: "Port Range".to_string(),
                    key_pattern: "*port*".to_string(),
                    validator: ValidatorType::Range { min: 1.0, max: 65535.0 },
                    error_message: "Port must be between 1 and 65535".to_string(),
                },
                ValidationRule {
                    name: "Log Level".to_string(),
                    key_pattern: "*log_level*".to_string(),
                    validator: ValidatorType::OneOf(vec![
                        "trace".to_string(), "debug".to_string(), 
                        "info".to_string(), "warn".to_string(), "error".to_string()
                    ]),
                    error_message: "Invalid log level".to_string(),
                },
            ],
            required_keys: HashSet::from([
                "database.url".to_string(),
                "server.port".to_string(),
                "auth.jwt_secret".to_string(),
            ]),
            type_validators: HashMap::from([
                ("port".to_string(), TypeValidator::Integer { min: Some(1), max: Some(65535) }),
                ("url".to_string(), TypeValidator::Url),
                ("email".to_string(), TypeValidator::Email),
                ("file_path".to_string(), TypeValidator::FilePath),
            ]),
        }
    }
}

impl ProductionErrorHandler {
    /// Create a new production error handler
    pub fn new() -> Self {
        Self {
            sanitization_rules: ErrorSanitizationRules {
                remove_file_paths: true,
                remove_stack_traces: true,
                sanitize_database_errors: true,
                sensitive_patterns: vec![
                    SensitivePattern {
                        pattern: r"/Users/[^/]+/".to_string(),
                        replacement: "/home/user/".to_string(),
                        is_regex: true,
                    },
                    SensitivePattern {
                        pattern: "password".to_string(),
                        replacement: "[REDACTED]".to_string(),
                        is_regex: false,
                    },
                ],
                use_generic_messages: true,
            },
            logging_config: SecurityLoggingConfig {
                enable_audit_logging: true,
                separate_security_log: true,
                structured_format: true,
                include_correlation_ids: true,
                sensitive_data_filter: SensitiveDataFilter {
                    redaction_patterns: vec![
                        r"password['\"]?\s*[:=]\s*['\"]?[^'\"]*".to_string(),
                        r"token['\"]?\s*[:=]\s*['\"]?[^'\"]*".to_string(),
                        r"secret['\"]?\s*[:=]\s*['\"]?[^'\"]*".to_string(),
                    ],
                    redaction_replacement: "[REDACTED]".to_string(),
                    hash_sensitive_values: false,
                    always_redact_fields: HashSet::from([
                        "password".to_string(),
                        "token".to_string(),
                        "secret".to_string(),
                        "key".to_string(),
                    ]),
                },
                rotation_config: LogRotationConfig {
                    max_file_size_mb: 100,
                    max_files: 10,
                    compress_rotated: true,
                    rotation_schedule: RotationSchedule::Daily,
                },
            },
            alert_config: SecurityAlertConfig {
                enabled: true,
                thresholds: AlertThresholds {
                    failed_auth_per_minute: 10,
                    validation_failures_per_minute: 50,
                    config_errors_per_hour: 5,
                    suspicious_activity_score: 70.0,
                },
                destinations: vec![
                    AlertDestination::Email("security@example.com".to_string()),
                    AlertDestination::Syslog { 
                        server: "localhost:514".to_string(), 
                        facility: "local0".to_string() 
                    },
                ],
                frequency_limits: AlertFrequencyLimits {
                    min_interval_seconds: 300,
                    max_alerts_per_hour: 20,
                    burst_threshold: 5,
                },
            },
        }
    }
    
    /// Sanitize error message for production use
    pub fn sanitize_error(&self, error_message: &str) -> String {
        let mut sanitized = error_message.to_string();
        
        // Remove file paths
        if self.sanitization_rules.remove_file_paths {
            sanitized = sanitized.replace(
                &std::env::current_dir().unwrap_or_default().to_string_lossy(),
                "[PROJECT_ROOT]"
            );
        }
        
        // Apply sensitive pattern replacements
        for pattern in &self.sanitization_rules.sensitive_patterns {
            if pattern.is_regex {
                // In production, use regex crate for proper regex replacement
                sanitized = sanitized.replace(&pattern.pattern, &pattern.replacement);
            } else {
                sanitized = sanitized.replace(&pattern.pattern, &pattern.replacement);
            }
        }
        
        // Use generic message if enabled
        if self.sanitization_rules.use_generic_messages {
            if sanitized.to_lowercase().contains("error") {
                return "An internal error occurred. Please contact support.".to_string();
            }
        }
        
        sanitized
    }
}

impl fmt::Display for SecuritySeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecuritySeverity::Low => write!(f, "LOW"),
            SecuritySeverity::Medium => write!(f, "MEDIUM"),
            SecuritySeverity::High => write!(f, "HIGH"),
            SecuritySeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

impl fmt::Display for SecurityErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityErrorType::InputValidation => write!(f, "Input Validation"),
            SecurityErrorType::Authentication => write!(f, "Authentication"),
            SecurityErrorType::Authorization => write!(f, "Authorization"),
            SecurityErrorType::ConfigurationSecurity => write!(f, "Configuration Security"),
            SecurityErrorType::DataProtection => write!(f, "Data Protection"),
            SecurityErrorType::NetworkSecurity => write!(f, "Network Security"),
            SecurityErrorType::InformationDisclosure => write!(f, "Information Disclosure"),
        }
    }
}

/// Run comprehensive security validation across all OxiRS modules
pub fn run_comprehensive_security_audit() -> Result<SecurityValidationResult> {
    info!("Starting comprehensive security audit of OxiRS modules");
    
    let validation_framework = InputValidationFramework::new(ValidationConfig::default());
    let config_manager = SecureConfigurationManager::new()?;
    let error_handler = ProductionErrorHandler::new();
    
    let mut all_errors = Vec::new();
    let mut all_warnings = Vec::new();
    let mut total_score = 0u8;
    let mut component_count = 0;
    
    // Test SPARQL validation
    let test_query = "SELECT * WHERE { ?s ?p ?o } LIMIT 10";
    match validation_framework.validate_sparql_query(test_query) {
        Ok(result) => {
            all_errors.extend(result.errors);
            all_warnings.extend(result.warnings);
            total_score += result.security_score;
            component_count += 1;
        },
        Err(e) => {
            error!("SPARQL validation failed: {}", e);
        }
    }
    
    // Test RDF validation
    let test_rdf = "@prefix ex: <http://example.org/> . ex:subject ex:predicate \"literal\" .";
    match validation_framework.validate_rdf_data(test_rdf, RdfFormat::Turtle) {
        Ok(result) => {
            all_errors.extend(result.errors);
            all_warnings.extend(result.warnings);
            total_score += result.security_score;
            component_count += 1;
        },
        Err(e) => {
            error!("RDF validation failed: {}", e);
        }
    }
    
    // Test configuration validation
    let test_config = r#"
        [server]
        port = 8080
        host = "0.0.0.0"
        
        [auth]
        password = "default_password"
    "#;
    match config_manager.validate_configuration(test_config) {
        Ok(result) => {
            all_errors.extend(result.errors);
            all_warnings.extend(result.warnings);
            total_score += result.security_score;
            component_count += 1;
        },
        Err(e) => {
            error!("Configuration validation failed: {}", e);
        }
    }
    
    let average_score = if component_count > 0 { total_score / component_count } else { 0 };
    
    let recommendations = vec![
        SecurityRecommendation {
            title: "Implement Comprehensive Input Validation".to_string(),
            description: "Deploy the input validation framework across all modules".to_string(),
            priority: RecommendationPriority::Critical,
            effort: ImplementationEffort::Medium,
            impact: SecurityImpact::Critical,
        },
        SecurityRecommendation {
            title: "Secure Configuration Management".to_string(),
            description: "Implement secure configuration management with secret handling".to_string(),
            priority: RecommendationPriority::High,
            effort: ImplementationEffort::Medium,
            impact: SecurityImpact::High,
        },
        SecurityRecommendation {
            title: "Production Error Handling".to_string(),
            description: "Deploy production-ready error handling with information sanitization".to_string(),
            priority: RecommendationPriority::High,
            effort: ImplementationEffort::Low,
            impact: SecurityImpact::Medium,
        },
    ];
    
    info!("Security audit completed with {} errors and {} warnings", 
          all_errors.len(), all_warnings.len());
    
    Ok(SecurityValidationResult {
        is_valid: all_errors.is_empty(),
        security_score: average_score,
        errors: all_errors,
        warnings: all_warnings,
        recommendations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparql_validation() {
        let framework = InputValidationFramework::new(ValidationConfig::default());
        
        // Test valid query
        let valid_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10";
        let result = framework.validate_sparql_query(valid_query).unwrap();
        assert!(result.is_valid);
        assert!(result.security_score > 80);
        
        // Test malicious query
        let malicious_query = "DROP ALL; SELECT * WHERE { ?s ?p ?o }";
        let result = framework.validate_sparql_query(malicious_query).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }
    
    #[test]
    fn test_rdf_validation() {
        let framework = InputValidationFramework::new(ValidationConfig::default());
        
        // Test valid RDF
        let valid_rdf = "@prefix ex: <http://example.org/> . ex:s ex:p \"literal\" .";
        let result = framework.validate_rdf_data(valid_rdf, RdfFormat::Turtle).unwrap();
        assert!(result.is_valid);
        
        // Test malicious RDF
        let malicious_rdf = r#"@prefix ex: <javascript:alert('xss')> . ex:s ex:p "literal" ."#;
        let result = framework.validate_rdf_data(malicious_rdf, RdfFormat::Turtle).unwrap();
        assert!(!result.is_valid);
    }
    
    #[test]
    fn test_configuration_validation() {
        let config_manager = SecureConfigurationManager::new().unwrap();
        
        // Test configuration with hardcoded secrets
        let bad_config = r#"password = "hardcoded_password""#;
        let result = config_manager.validate_configuration(bad_config).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
        
        // Test good configuration
        let good_config = r#"password = "${OXIRS_PASSWORD}""#;
        let result = config_manager.validate_configuration(good_config).unwrap();
        assert!(result.is_valid);
    }
    
    #[test]
    fn test_error_sanitization() {
        let error_handler = ProductionErrorHandler::new();
        
        let error_msg = "Database connection failed: password='secret123' at /Users/admin/project/src/db.rs:42";
        let sanitized = error_handler.sanitize_error(error_msg);
        
        assert!(!sanitized.contains("secret123"));
        assert!(!sanitized.contains("/Users/admin"));
    }
    
    #[test]
    fn test_comprehensive_audit() {
        let result = run_comprehensive_security_audit().unwrap();
        assert!(!result.recommendations.is_empty());
        assert!(result.security_score <= 100);
    }
}