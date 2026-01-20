//! Security Module for OxiRS AI Components
//!
//! Provides comprehensive security features including:
//! - Secure credential management with encryption
//! - Input validation and sanitization
//! - Rate limiting and throttling
//! - Audit logging for security events
//! - Memory-safe operations
//! - DDoS protection

pub mod credentials;
pub mod input_validation;
pub mod rate_limiting;
pub mod audit_log;
pub mod memory_protection;
pub mod encryption;

pub use credentials::{CredentialManager, SecureCredential};
pub use input_validation::{InputValidator, ValidationResult};
pub use rate_limiting::{RateLimiter, RateLimitConfig};
pub use audit_log::{AuditLogger, SecurityEvent};
pub use memory_protection::{SecureMemory, MemoryGuard};
pub use encryption::{Encryptor, EncryptionConfig};

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Security configuration for AI modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable credential encryption
    pub encrypt_credentials: bool,

    /// Enable input validation
    pub validate_inputs: bool,

    /// Enable rate limiting
    pub enable_rate_limiting: bool,

    /// Enable audit logging
    pub enable_audit_log: bool,

    /// Maximum input size (bytes)
    pub max_input_size: usize,

    /// Maximum tokens per request
    pub max_tokens_per_request: usize,

    /// Rate limit: requests per minute
    pub requests_per_minute: usize,

    /// Rate limit: tokens per minute
    pub tokens_per_minute: usize,

    /// Enable memory protection
    pub enable_memory_protection: bool,

    /// Memory limit for embeddings (MB)
    pub embedding_memory_limit_mb: usize,

    /// Enable DDoS protection
    pub enable_ddos_protection: bool,

    /// IP whitelist (empty = allow all)
    pub ip_whitelist: Vec<String>,

    /// IP blacklist
    pub ip_blacklist: Vec<String>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            encrypt_credentials: true,
            validate_inputs: true,
            enable_rate_limiting: true,
            enable_audit_log: true,
            max_input_size: 1024 * 1024, // 1MB
            max_tokens_per_request: 4096,
            requests_per_minute: 60,
            tokens_per_minute: 100_000,
            enable_memory_protection: true,
            embedding_memory_limit_mb: 1024, // 1GB
            enable_ddos_protection: true,
            ip_whitelist: vec![],
            ip_blacklist: vec![],
        }
    }
}

/// Security manager coordinating all security features
pub struct SecurityManager {
    config: SecurityConfig,
    credential_manager: CredentialManager,
    input_validator: InputValidator,
    rate_limiter: RateLimiter,
    audit_logger: AuditLogger,
    memory_guard: MemoryGuard,
}

impl SecurityManager {
    pub fn new(config: SecurityConfig) -> Result<Self> {
        let credential_manager = CredentialManager::new(config.encrypt_credentials)?;
        let input_validator = InputValidator::new(
            config.max_input_size,
            config.max_tokens_per_request,
        );
        let rate_limiter = RateLimiter::new(RateLimitConfig {
            requests_per_minute: config.requests_per_minute,
            tokens_per_minute: config.tokens_per_minute,
            enable_burst: true,
            burst_size: config.requests_per_minute / 10,
        });
        let audit_logger = AuditLogger::new()?;
        let memory_guard = MemoryGuard::new(config.embedding_memory_limit_mb * 1024 * 1024);

        Ok(Self {
            config,
            credential_manager,
            input_validator,
            rate_limiter,
            audit_logger,
            memory_guard,
        })
    }

    /// Validate and sanitize user input
    pub fn validate_input(&self, input: &str, user_id: &str) -> Result<ValidationResult> {
        if !self.config.validate_inputs {
            return Ok(ValidationResult::valid(input.to_string()));
        }

        // Log validation attempt
        self.audit_logger.log_event(SecurityEvent::InputValidation {
            user_id: user_id.to_string(),
            input_length: input.len(),
            timestamp: chrono::Utc::now(),
        })?;

        self.input_validator.validate(input)
    }

    /// Check rate limit for user
    pub fn check_rate_limit(&self, user_id: &str, tokens: usize) -> Result<bool> {
        if !self.config.enable_rate_limiting {
            return Ok(true);
        }

        let allowed = self.rate_limiter.check_limit(user_id, tokens)?;

        if !allowed {
            self.audit_logger.log_event(SecurityEvent::RateLimitExceeded {
                user_id: user_id.to_string(),
                timestamp: chrono::Utc::now(),
            })?;
        }

        Ok(allowed)
    }

    /// Store credential securely
    pub fn store_credential(&mut self, provider: &str, api_key: &str) -> Result<()> {
        self.audit_logger.log_event(SecurityEvent::CredentialStored {
            provider: provider.to_string(),
            timestamp: chrono::Utc::now(),
        })?;

        self.credential_manager.store(provider, api_key)
    }

    /// Retrieve credential securely
    pub fn retrieve_credential(&self, provider: &str) -> Result<Option<SecureCredential>> {
        self.credential_manager.retrieve(provider)
    }

    /// Check memory usage
    pub fn check_memory_usage(&self, additional_bytes: usize) -> Result<bool> {
        if !self.config.enable_memory_protection {
            return Ok(true);
        }

        Ok(self.memory_guard.can_allocate(additional_bytes))
    }

    /// Check IP address
    pub fn check_ip(&self, ip: &str) -> Result<bool> {
        if !self.config.enable_ddos_protection {
            return Ok(true);
        }

        // Check blacklist first
        if self.config.ip_blacklist.contains(&ip.to_string()) {
            self.audit_logger.log_event(SecurityEvent::IpBlocked {
                ip: ip.to_string(),
                reason: "blacklisted".to_string(),
                timestamp: chrono::Utc::now(),
            })?;
            return Ok(false);
        }

        // If whitelist is configured, check it
        if !self.config.ip_whitelist.is_empty() {
            let allowed = self.config.ip_whitelist.contains(&ip.to_string());
            if !allowed {
                self.audit_logger.log_event(SecurityEvent::IpBlocked {
                    ip: ip.to_string(),
                    reason: "not whitelisted".to_string(),
                    timestamp: chrono::Utc::now(),
                })?;
            }
            return Ok(allowed);
        }

        Ok(true)
    }

    /// Get audit log for security analysis
    pub fn get_audit_events(&self, start_time: chrono::DateTime<chrono::Utc>) -> Result<Vec<SecurityEvent>> {
        self.audit_logger.get_events_since(start_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_manager_creation() {
        let config = SecurityConfig::default();
        let manager = SecurityManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_input_validation() {
        let config = SecurityConfig::default();
        let manager = SecurityManager::new(config).unwrap();

        let result = manager.validate_input("SELECT * FROM users", "test_user");
        assert!(result.is_ok());
    }

    #[test]
    fn test_rate_limiting() {
        let config = SecurityConfig::default();
        let manager = SecurityManager::new(config).unwrap();

        // First request should be allowed
        let result = manager.check_rate_limit("test_user", 100);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_memory_check() {
        let config = SecurityConfig::default();
        let manager = SecurityManager::new(config).unwrap();

        // Small allocation should be allowed
        let result = manager.check_memory_usage(1024);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_ip_whitelist() {
        let mut config = SecurityConfig::default();
        config.ip_whitelist = vec!["127.0.0.1".to_string()];
        let manager = SecurityManager::new(config).unwrap();

        // Whitelisted IP should be allowed
        assert!(manager.check_ip("127.0.0.1").unwrap());

        // Non-whitelisted IP should be blocked
        assert!(!manager.check_ip("192.168.1.1").unwrap());
    }

    #[test]
    fn test_ip_blacklist() {
        let mut config = SecurityConfig::default();
        config.ip_blacklist = vec!["10.0.0.1".to_string()];
        let manager = SecurityManager::new(config).unwrap();

        // Blacklisted IP should be blocked
        assert!(!manager.check_ip("10.0.0.1").unwrap());

        // Non-blacklisted IP should be allowed
        assert!(manager.check_ip("127.0.0.1").unwrap());
    }
}
