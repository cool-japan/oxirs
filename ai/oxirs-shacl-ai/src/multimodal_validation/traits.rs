//! Trait definitions for multi-modal validation

use async_trait::async_trait;

use crate::{Result, ShaclAiError};
use super::types::*;

/// Trait for text content validation
#[async_trait]
pub trait TextValidator: Send + Sync + std::fmt::Debug {
    /// Validate text content
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>>;
    
    /// Get validator name
    fn name(&self) -> &str;
    
    /// Get validator description
    fn description(&self) -> &str;
    
    /// Check if validator supports the given content
    fn supports_content(&self, content: &MultiModalContent) -> bool {
        matches!(content.content_type, ContentType::Text | ContentType::Composite)
    }
}

/// Trait for image content validation
#[async_trait]
pub trait ImageValidator: Send + Sync + std::fmt::Debug {
    /// Validate image content
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>>;
    
    /// Get validator name
    fn name(&self) -> &str;
    
    /// Get validator description
    fn description(&self) -> &str;
    
    /// Check if validator supports the given content
    fn supports_content(&self, content: &MultiModalContent) -> bool {
        matches!(content.content_type, ContentType::Image | ContentType::Composite)
    }
}

/// Trait for audio content validation
#[async_trait]
pub trait AudioValidator: Send + Sync + std::fmt::Debug {
    /// Validate audio content
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>>;
    
    /// Get validator name
    fn name(&self) -> &str;
    
    /// Get validator description
    fn description(&self) -> &str;
    
    /// Check if validator supports the given content
    fn supports_content(&self, content: &MultiModalContent) -> bool {
        matches!(content.content_type, ContentType::Audio | ContentType::Composite)
    }
}

/// Trait for video content validation
#[async_trait]
pub trait VideoValidator: Send + Sync + std::fmt::Debug {
    /// Validate video content
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>>;
    
    /// Get validator name
    fn name(&self) -> &str;
    
    /// Get validator description
    fn description(&self) -> &str;
    
    /// Check if validator supports the given content
    fn supports_content(&self, content: &MultiModalContent) -> bool {
        matches!(content.content_type, ContentType::Video | ContentType::Composite)
    }
}

/// Trait for document content validation
#[async_trait]
pub trait DocumentValidator: Send + Sync + std::fmt::Debug {
    /// Validate document content
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>>;
    
    /// Get validator name
    fn name(&self) -> &str;
    
    /// Get validator description
    fn description(&self) -> &str;
    
    /// Check if validator supports the given content
    fn supports_content(&self, content: &MultiModalContent) -> bool {
        matches!(content.content_type, ContentType::Document | ContentType::Composite)
    }
}

/// Trait for semantic analysis of content
#[async_trait]
pub trait SemanticAnalyzer: Send + Sync + std::fmt::Debug {
    /// Analyze content semantically
    async fn analyze(&self, content: &MultiModalContent) -> Result<Option<AnalysisResult>>;
    
    /// Get analyzer name
    fn name(&self) -> &str;
    
    /// Get analyzer description
    fn description(&self) -> &str;
    
    /// Check if analyzer supports the given content
    fn supports_content(&self, content: &MultiModalContent) -> bool {
        true // Most semantic analyzers can work with any content type
    }
}

/// Trait for cross-modal validation
#[async_trait]
pub trait CrossModalValidator: Send + Sync + std::fmt::Debug {
    /// Validate across multiple content types
    async fn validate_cross_modal(
        &self,
        content_analyses: &[ContentAnalysis],
    ) -> Result<Vec<ValidationResult>>;
    
    /// Get validator name
    fn name(&self) -> &str;
    
    /// Get validator description
    fn description(&self) -> &str;
    
    /// Get required content types for cross-modal validation
    fn required_content_types(&self) -> Vec<ContentType>;
}

/// Trait for content loading
#[async_trait]
pub trait ContentLoader: Send + Sync + std::fmt::Debug {
    /// Load content from a reference
    async fn load_content(&self, content_ref: &MultiModalContentRef) -> Result<MultiModalContent>;
    
    /// Check if loader supports the given content reference
    fn supports_content_ref(&self, content_ref: &MultiModalContentRef) -> bool;
    
    /// Get loader name
    fn name(&self) -> &str;
}

/// Trait for content preprocessing
#[async_trait]
pub trait ContentPreprocessor: Send + Sync + std::fmt::Debug {
    /// Preprocess content before validation
    async fn preprocess(&self, content: &MultiModalContent) -> Result<MultiModalContent>;
    
    /// Check if preprocessor supports the given content
    fn supports_content(&self, content: &MultiModalContent) -> bool;
    
    /// Get preprocessor name
    fn name(&self) -> &str;
}

/// Trait for content caching
#[async_trait]
pub trait ContentCache: Send + Sync + std::fmt::Debug {
    /// Get cached content
    async fn get(&self, key: &str) -> Result<Option<CachedContent>>;
    
    /// Store content in cache
    async fn set(&self, key: &str, content: &CachedContent) -> Result<()>;
    
    /// Remove content from cache
    async fn remove(&self, key: &str) -> Result<()>;
    
    /// Clear all cached content
    async fn clear(&self) -> Result<()>;
    
    /// Get cache statistics
    async fn stats(&self) -> Result<CacheStats>;
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Number of cached items
    pub items: u64,
    /// Total cache size in bytes
    pub size: u64,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

/// Trait for validation reporting
pub trait ValidationReporter: Send + Sync + std::fmt::Debug {
    /// Generate validation report
    fn generate_report(&self, report: &MultiModalValidationReport) -> Result<String>;
    
    /// Get reporter name
    fn name(&self) -> &str;
    
    /// Get supported output formats
    fn supported_formats(&self) -> Vec<String>;
}

/// Trait for validation metrics
pub trait ValidationMetrics: Send + Sync + std::fmt::Debug {
    /// Record validation metrics
    fn record_validation(&self, content_type: &ContentType, duration: std::time::Duration, success: bool);
    
    /// Record analysis metrics
    fn record_analysis(&self, analyzer: &str, duration: std::time::Duration, confidence: f64);
    
    /// Get validation statistics
    fn get_statistics(&self) -> ValidationStatistics;
    
    /// Reset metrics
    fn reset(&self);
}

/// Trait for content quality assessment
#[async_trait]
pub trait QualityAssessor: Send + Sync + std::fmt::Debug {
    /// Assess content quality
    async fn assess_quality(&self, content: &MultiModalContent, analysis: &ContentAnalysis) -> Result<QualityAssessment>;
    
    /// Get quality thresholds
    fn get_thresholds(&self) -> QualityThresholds;
    
    /// Set quality thresholds
    fn set_thresholds(&self, thresholds: QualityThresholds);
}

/// Quality assessment result
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f64,
    /// Individual quality dimensions
    pub dimensions: std::collections::HashMap<String, f64>,
    /// Quality issues found
    pub issues: Vec<QualityIssue>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Quality issue
#[derive(Debug, Clone)]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: String,
    /// Issue description
    pub description: String,
    /// Issue severity
    pub severity: QualitySeverity,
    /// Confidence in issue detection
    pub confidence: f64,
}

/// Quality severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualitySeverity {
    /// Low severity issue
    Low,
    /// Medium severity issue
    Medium,
    /// High severity issue
    High,
    /// Critical severity issue
    Critical,
}

/// Quality thresholds
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Minimum overall quality score
    pub min_overall_score: f64,
    /// Minimum scores for individual dimensions
    pub min_dimension_scores: std::collections::HashMap<String, f64>,
    /// Maximum allowed critical issues
    pub max_critical_issues: u32,
    /// Maximum allowed high severity issues
    pub max_high_issues: u32,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_overall_score: 0.7,
            min_dimension_scores: std::collections::HashMap::new(),
            max_critical_issues: 0,
            max_high_issues: 2,
        }
    }
}

/// Trait for validation rule management
#[async_trait]
pub trait ValidationRuleManager: Send + Sync + std::fmt::Debug {
    /// Add a validation rule
    async fn add_rule(&self, rule: ValidationRule) -> Result<()>;
    
    /// Remove a validation rule
    async fn remove_rule(&self, rule_id: &str) -> Result<()>;
    
    /// Update a validation rule
    async fn update_rule(&self, rule: ValidationRule) -> Result<()>;
    
    /// Get all validation rules
    async fn get_rules(&self) -> Result<Vec<ValidationRule>>;
    
    /// Get rules for a specific content type
    async fn get_rules_for_content_type(&self, content_type: &ContentType) -> Result<Vec<ValidationRule>>;
    
    /// Validate content against all applicable rules
    async fn validate_with_rules(&self, content: &MultiModalContent) -> Result<Vec<ValidationResult>>;
}

/// Validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule identifier
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Applicable content types
    pub content_types: Vec<ContentType>,
    /// Rule conditions
    pub conditions: Vec<RuleCondition>,
    /// Rule actions
    pub actions: Vec<RuleAction>,
    /// Rule priority
    pub priority: u32,
    /// Whether rule is enabled
    pub enabled: bool,
}

/// Rule condition
#[derive(Debug, Clone)]
pub struct RuleCondition {
    /// Condition type
    pub condition_type: String,
    /// Condition parameters
    pub parameters: std::collections::HashMap<String, String>,
    /// Whether condition should be negated
    pub negate: bool,
}

/// Rule action
#[derive(Debug, Clone)]
pub struct RuleAction {
    /// Action type
    pub action_type: String,
    /// Action parameters
    pub parameters: std::collections::HashMap<String, String>,
}

/// Trait for plugin management
#[async_trait]
pub trait PluginManager: Send + Sync + std::fmt::Debug {
    /// Load a plugin
    async fn load_plugin(&self, plugin_path: &str) -> Result<()>;
    
    /// Unload a plugin
    async fn unload_plugin(&self, plugin_id: &str) -> Result<()>;
    
    /// List loaded plugins
    async fn list_plugins(&self) -> Result<Vec<PluginInfo>>;
    
    /// Get plugin by ID
    async fn get_plugin(&self, plugin_id: &str) -> Result<Option<PluginInfo>>;
}

/// Plugin information
#[derive(Debug, Clone)]
pub struct PluginInfo {
    /// Plugin ID
    pub id: String,
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin author
    pub author: String,
    /// Supported content types
    pub supported_content_types: Vec<ContentType>,
    /// Plugin capabilities
    pub capabilities: Vec<String>,
}

impl std::fmt::Display for QualitySeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QualitySeverity::Low => write!(f, "Low"),
            QualitySeverity::Medium => write!(f, "Medium"),
            QualitySeverity::High => write!(f, "High"),
            QualitySeverity::Critical => write!(f, "Critical"),
        }
    }
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            items: 0,
            size: 0,
            hit_rate: 0.0,
        }
    }
}