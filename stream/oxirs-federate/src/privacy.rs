//! Privacy Protection and Compliance Module
//!
//! This module provides comprehensive privacy protection features for federated query processing,
//! including differential privacy, data anonymization, GDPR compliance, and privacy-preserving joins.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock as AsyncRwLock;
use uuid::Uuid;

/// Privacy-related errors
#[derive(Error, Debug)]
pub enum PrivacyError {
    #[error("Privacy budget exhausted: {0}")]
    BudgetExhausted(String),
    #[error("Insufficient privacy level: required {required}, current {current}")]
    InsufficientPrivacy { required: f64, current: f64 },
    #[error("GDPR compliance violation: {0}")]
    GdprViolation(String),
    #[error("Data anonymization failed: {0}")]
    AnonymizationFailed(String),
    #[error("Privacy policy not found: {0}")]
    PolicyNotFound(String),
    #[error("Sensitive data detected: {0}")]
    SensitiveDataDetected(String),
}

/// Privacy protection level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyLevel {
    /// No privacy protection
    None,
    /// Basic anonymization
    Basic,
    /// k-anonymity protection
    KAnonymity,
    /// l-diversity protection
    LDiversity,
    /// t-closeness protection
    TCloseness,
    /// Differential privacy
    DifferentialPrivacy,
    /// Maximum privacy protection
    Maximum,
}

impl PrivacyLevel {
    /// Get numeric privacy score
    pub fn score(&self) -> f64 {
        match self {
            PrivacyLevel::None => 0.0,
            PrivacyLevel::Basic => 0.2,
            PrivacyLevel::KAnonymity => 0.4,
            PrivacyLevel::LDiversity => 0.6,
            PrivacyLevel::TCloseness => 0.8,
            PrivacyLevel::DifferentialPrivacy => 0.9,
            PrivacyLevel::Maximum => 1.0,
        }
    }
}

/// Differential privacy parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialPrivacyConfig {
    /// Privacy budget (epsilon)
    pub epsilon: f64,
    /// Differential privacy delta
    pub delta: f64,
    /// Sensitivity of the query
    pub sensitivity: f64,
    /// Noise mechanism
    pub mechanism: NoiseMechanism,
    /// Budget tracking window
    pub budget_window: Duration,
}

impl Default for DifferentialPrivacyConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-5,
            sensitivity: 1.0,
            mechanism: NoiseMechanism::Laplace,
            budget_window: Duration::from_hours(24),
        }
    }
}

/// Noise mechanisms for differential privacy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NoiseMechanism {
    /// Laplace mechanism
    Laplace,
    /// Gaussian mechanism
    Gaussian,
    /// Exponential mechanism
    Exponential,
    /// Sparse Vector Technique
    SparseVector,
}

/// K-anonymity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KAnonymityConfig {
    /// Minimum group size (k value)
    pub k: usize,
    /// Quasi-identifiers to consider
    pub quasi_identifiers: Vec<String>,
    /// Suppression threshold
    pub suppression_threshold: f64,
    /// Generalization hierarchy
    pub generalization_rules: HashMap<String, Vec<String>>,
}

impl Default for KAnonymityConfig {
    fn default() -> Self {
        Self {
            k: 5,
            quasi_identifiers: vec![],
            suppression_threshold: 0.1,
            generalization_rules: HashMap::new(),
        }
    }
}

/// L-diversity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LDiversityConfig {
    /// L value for diversity requirement
    pub l: usize,
    /// Sensitive attributes
    pub sensitive_attributes: Vec<String>,
    /// Diversity measure
    pub diversity_measure: DiversityMeasure,
}

/// Diversity measures
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DiversityMeasure {
    /// Distinct l-diversity
    Distinct,
    /// Entropy l-diversity
    Entropy,
    /// Recursive (c,l)-diversity
    Recursive,
}

/// GDPR compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GdprConfig {
    /// Data retention period
    pub retention_period: Duration,
    /// Consent requirements
    pub consent_required: bool,
    /// Data subject rights
    pub subject_rights: Vec<DataSubjectRight>,
    /// Lawful basis for processing
    pub lawful_basis: LawfulBasis,
    /// Data protection impact assessment required
    pub dpia_required: bool,
}

/// Data subject rights under GDPR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSubjectRight {
    /// Right to access
    Access,
    /// Right to rectification
    Rectification,
    /// Right to erasure (right to be forgotten)
    Erasure,
    /// Right to restrict processing
    RestrictProcessing,
    /// Right to data portability
    DataPortability,
    /// Right to object
    Object,
}

/// Lawful basis for data processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LawfulBasis {
    /// Consent
    Consent,
    /// Contract
    Contract,
    /// Legal obligation
    LegalObligation,
    /// Vital interests
    VitalInterests,
    /// Public task
    PublicTask,
    /// Legitimate interests
    LegitimateInterests,
}

/// Privacy policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyPolicy {
    /// Policy ID
    pub id: String,
    /// Policy name
    pub name: String,
    /// Required privacy level
    pub required_level: PrivacyLevel,
    /// Differential privacy config
    pub dp_config: Option<DifferentialPrivacyConfig>,
    /// K-anonymity config
    pub k_anonymity_config: Option<KAnonymityConfig>,
    /// L-diversity config
    pub l_diversity_config: Option<LDiversityConfig>,
    /// GDPR config
    pub gdpr_config: Option<GdprConfig>,
    /// Sensitive data patterns
    pub sensitive_patterns: Vec<String>,
    /// Data anonymization rules
    pub anonymization_rules: Vec<AnonymizationRule>,
    /// Privacy-preserving join settings
    pub join_settings: PrivacyPreservingJoinConfig,
}

/// Data anonymization rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymizationRule {
    /// Field pattern to match
    pub field_pattern: String,
    /// Anonymization technique
    pub technique: AnonymizationTechnique,
    /// Configuration for the technique
    pub config: serde_json::Value,
}

/// Anonymization techniques
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnonymizationTechnique {
    /// Replace with fixed value
    Suppression,
    /// Generalization to broader category
    Generalization,
    /// Add random noise
    Perturbation,
    /// Replace with synthetic data
    Substitution,
    /// Hash-based pseudonymization
    Hashing,
    /// Format-preserving encryption
    Encryption,
    /// Data masking
    Masking,
}

/// Privacy-preserving join configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyPreservingJoinConfig {
    /// Enable secure multi-party computation
    pub enable_smpc: bool,
    /// Enable homomorphic encryption
    pub enable_he: bool,
    /// Enable private set intersection
    pub enable_psi: bool,
    /// Minimum trust level required
    pub min_trust_level: f64,
    /// Join result anonymization
    pub result_anonymization: bool,
}

impl Default for PrivacyPreservingJoinConfig {
    fn default() -> Self {
        Self {
            enable_smpc: false,
            enable_he: false,
            enable_psi: true,
            min_trust_level: 0.7,
            result_anonymization: true,
        }
    }
}

/// Privacy budget tracker
#[derive(Debug)]
pub struct PrivacyBudgetTracker {
    /// Total available budget
    total_budget: f64,
    /// Used budget within window
    used_budget: Arc<RwLock<HashMap<String, (f64, SystemTime)>>>,
    /// Budget window duration
    window_duration: Duration,
}

impl PrivacyBudgetTracker {
    /// Create new budget tracker
    pub fn new(total_budget: f64, window_duration: Duration) -> Self {
        Self {
            total_budget,
            used_budget: Arc::new(RwLock::new(HashMap::new())),
            window_duration,
        }
    }

    /// Check if budget is available
    pub fn check_budget(&self, user_id: &str, requested: f64) -> Result<(), PrivacyError> {
        let current_time = SystemTime::now();
        let mut budget_map = self.used_budget.write().unwrap();

        // Clean expired entries
        budget_map.retain(|_, (_, timestamp)| {
            current_time.duration_since(*timestamp).unwrap_or(Duration::ZERO) <= self.window_duration
        });

        // Calculate used budget for user
        let used = budget_map.get(user_id).map(|(budget, _)| *budget).unwrap_or(0.0);

        if used + requested > self.total_budget {
            return Err(PrivacyError::BudgetExhausted(
                format!("User {} would exceed budget: used={}, requested={}, total={}", 
                       user_id, used, requested, self.total_budget)
            ));
        }

        Ok(())
    }

    /// Consume budget
    pub fn consume_budget(&self, user_id: &str, amount: f64) -> Result<(), PrivacyError> {
        self.check_budget(user_id, amount)?;
        
        let current_time = SystemTime::now();
        let mut budget_map = self.used_budget.write().unwrap();
        
        budget_map.entry(user_id.to_string()).and_modify(|(budget, timestamp)| {
            *budget += amount;
            *timestamp = current_time;
        }).or_insert((amount, current_time));

        Ok(())
    }

    /// Get remaining budget for user
    pub fn remaining_budget(&self, user_id: &str) -> f64 {
        let budget_map = self.used_budget.read().unwrap();
        let used = budget_map.get(user_id).map(|(budget, _)| *budget).unwrap_or(0.0);
        (self.total_budget - used).max(0.0)
    }
}

/// Privacy manager for federated queries
#[derive(Debug)]
pub struct PrivacyManager {
    /// Privacy policies by ID
    policies: Arc<AsyncRwLock<HashMap<String, PrivacyPolicy>>>,
    /// Budget tracker
    budget_tracker: Arc<PrivacyBudgetTracker>,
    /// Sensitive data detector
    sensitive_detector: SensitiveDataDetector,
    /// GDPR compliance checker
    gdpr_checker: GdprComplianceChecker,
    /// Data anonymizer
    anonymizer: DataAnonymizer,
}

impl PrivacyManager {
    /// Create new privacy manager
    pub fn new(total_budget: f64, budget_window: Duration) -> Self {
        Self {
            policies: Arc::new(AsyncRwLock::new(HashMap::new())),
            budget_tracker: Arc::new(PrivacyBudgetTracker::new(total_budget, budget_window)),
            sensitive_detector: SensitiveDataDetector::new(),
            gdpr_checker: GdprComplianceChecker::new(),
            anonymizer: DataAnonymizer::new(),
        }
    }

    /// Register privacy policy
    pub async fn register_policy(&self, policy: PrivacyPolicy) {
        let mut policies = self.policies.write().await;
        policies.insert(policy.id.clone(), policy);
    }

    /// Apply privacy protection to query
    pub async fn apply_privacy_protection(
        &self,
        query: &str,
        user_id: &str,
        policy_id: &str,
        data: &mut serde_json::Value,
    ) -> Result<PrivacyProtectionResult, PrivacyError> {
        let policies = self.policies.read().await;
        let policy = policies.get(policy_id)
            .ok_or_else(|| PrivacyError::PolicyNotFound(policy_id.to_string()))?;

        // Check sensitive data
        self.sensitive_detector.detect_sensitive_data(data, &policy.sensitive_patterns)?;

        // Apply differential privacy if configured
        let mut protection_result = PrivacyProtectionResult::default();
        
        if let Some(dp_config) = &policy.dp_config {
            self.budget_tracker.consume_budget(user_id, dp_config.epsilon)?;
            protection_result.differential_privacy_applied = true;
            protection_result.epsilon_used = dp_config.epsilon;
            
            // Apply noise based on mechanism
            self.apply_differential_privacy(data, dp_config)?;
        }

        // Apply anonymization
        if !policy.anonymization_rules.is_empty() {
            self.anonymizer.anonymize_data(data, &policy.anonymization_rules)?;
            protection_result.anonymization_applied = true;
        }

        // Check GDPR compliance
        if let Some(gdpr_config) = &policy.gdpr_config {
            self.gdpr_checker.check_compliance(data, gdpr_config)?;
            protection_result.gdpr_compliant = true;
        }

        // Apply k-anonymity if configured
        if let Some(k_config) = &policy.k_anonymity_config {
            self.apply_k_anonymity(data, k_config)?;
            protection_result.k_anonymity_applied = true;
            protection_result.k_value = Some(k_config.k);
        }

        // Apply l-diversity if configured
        if let Some(l_config) = &policy.l_diversity_config {
            self.apply_l_diversity(data, l_config)?;
            protection_result.l_diversity_applied = true;
            protection_result.l_value = Some(l_config.l);
        }

        protection_result.privacy_level_achieved = policy.required_level;
        Ok(protection_result)
    }

    /// Apply differential privacy noise
    fn apply_differential_privacy(
        &self,
        data: &mut serde_json::Value,
        config: &DifferentialPrivacyConfig,
    ) -> Result<(), PrivacyError> {
        match config.mechanism {
            NoiseMechanism::Laplace => {
                self.apply_laplace_noise(data, config.sensitivity / config.epsilon)?;
            }
            NoiseMechanism::Gaussian => {
                let sigma = (2.0 * config.sensitivity.powi(2) * 
                           (1.25 / config.delta).ln()) / config.epsilon.powi(2);
                self.apply_gaussian_noise(data, sigma.sqrt())?;
            }
            _ => {
                return Err(PrivacyError::AnonymizationFailed(
                    "Unsupported noise mechanism".to_string()
                ));
            }
        }
        Ok(())
    }

    /// Apply Laplace noise to numeric values
    fn apply_laplace_noise(&self, data: &mut serde_json::Value, scale: f64) -> Result<(), PrivacyError> {
        use rand::distributions::{Distribution, Uniform};
        use rand::thread_rng;

        let mut rng = thread_rng();
        let uniform = Uniform::new(0.0, 1.0);

        self.apply_noise_recursive(data, &mut |value| {
            if let Some(num) = value.as_f64() {
                let u1 = uniform.sample(&mut rng);
                let u2 = uniform.sample(&mut rng);
                let noise = scale * (u1 - 0.5).signum() * (1.0 - 2.0 * u2.min(1.0 - u2)).ln();
                *value = serde_json::Value::Number(
                    serde_json::Number::from_f64(num + noise).unwrap_or_else(|| serde_json::Number::from(0))
                );
            }
        });

        Ok(())
    }

    /// Apply Gaussian noise to numeric values
    fn apply_gaussian_noise(&self, data: &mut serde_json::Value, sigma: f64) -> Result<(), PrivacyError> {
        use rand::distributions::{Distribution, Normal};
        use rand::thread_rng;

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, sigma).map_err(|e| {
            PrivacyError::AnonymizationFailed(format!("Invalid Gaussian parameters: {}", e))
        })?;

        self.apply_noise_recursive(data, &mut |value| {
            if let Some(num) = value.as_f64() {
                let noise = normal.sample(&mut rng);
                *value = serde_json::Value::Number(
                    serde_json::Number::from_f64(num + noise).unwrap_or_else(|| serde_json::Number::from(0))
                );
            }
        });

        Ok(())
    }

    /// Apply noise function recursively to data
    fn apply_noise_recursive<F>(&self, data: &mut serde_json::Value, noise_fn: &mut F)
    where
        F: FnMut(&mut serde_json::Value),
    {
        match data {
            serde_json::Value::Object(map) => {
                for value in map.values_mut() {
                    self.apply_noise_recursive(value, noise_fn);
                }
            }
            serde_json::Value::Array(arr) => {
                for value in arr.iter_mut() {
                    self.apply_noise_recursive(value, noise_fn);
                }
            }
            serde_json::Value::Number(_) => {
                noise_fn(data);
            }
            _ => {}
        }
    }

    /// Apply k-anonymity protection
    fn apply_k_anonymity(
        &self,
        data: &mut serde_json::Value,
        config: &KAnonymityConfig,
    ) -> Result<(), PrivacyError> {
        // Implementation for k-anonymity would involve:
        // 1. Group records by quasi-identifiers
        // 2. Suppress or generalize records in groups < k
        // 3. Apply generalization hierarchy
        
        // For now, implement basic generalization
        if let serde_json::Value::Array(records) = data {
            for record in records.iter_mut() {
                if let serde_json::Value::Object(map) = record {
                    for field in &config.quasi_identifiers {
                        if let Some(value) = map.get_mut(field) {
                            self.generalize_value(value, field, &config.generalization_rules);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply l-diversity protection
    fn apply_l_diversity(
        &self,
        data: &mut serde_json::Value,
        config: &LDiversityConfig,
    ) -> Result<(), PrivacyError> {
        // Implementation for l-diversity would involve:
        // 1. Check diversity of sensitive attributes within equivalence classes
        // 2. Suppress records that don't meet l-diversity requirement
        
        // Basic implementation - ensure minimum diversity
        if let serde_json::Value::Array(records) = data {
            let mut diversity_groups: HashMap<String, HashSet<String>> = HashMap::new();
            
            for record in records.iter() {
                if let serde_json::Value::Object(map) = record {
                    for sensitive_attr in &config.sensitive_attributes {
                        if let Some(value) = map.get(sensitive_attr) {
                            let group_key = self.get_equivalence_class_key(map);
                            diversity_groups.entry(group_key)
                                .or_insert_with(HashSet::new)
                                .insert(value.to_string());
                        }
                    }
                }
            }

            // Remove groups that don't meet l-diversity requirement
            records.retain(|record| {
                if let serde_json::Value::Object(map) = record {
                    let group_key = self.get_equivalence_class_key(map);
                    diversity_groups.get(&group_key)
                        .map(|group| group.len() >= config.l)
                        .unwrap_or(false)
                } else {
                    false
                }
            });
        }

        Ok(())
    }

    /// Get equivalence class key for record
    fn get_equivalence_class_key(&self, record: &serde_json::Map<String, serde_json::Value>) -> String {
        // Create key based on quasi-identifiers
        let mut key_parts = Vec::new();
        for (field, value) in record.iter() {
            if field.contains("quasi") || field.contains("id") {
                key_parts.push(format!("{}:{}", field, value));
            }
        }
        key_parts.sort();
        key_parts.join("|")
    }

    /// Generalize value based on rules
    fn generalize_value(
        &self,
        value: &mut serde_json::Value,
        field: &str,
        rules: &HashMap<String, Vec<String>>,
    ) {
        if let Some(hierarchy) = rules.get(field) {
            if let Some(str_value) = value.as_str() {
                // Find position in hierarchy and move up one level
                if let Some(pos) = hierarchy.iter().position(|h| h == str_value) {
                    if pos > 0 {
                        *value = serde_json::Value::String(hierarchy[pos - 1].clone());
                    }
                }
            }
        }
    }

    /// Create privacy-preserving join
    pub async fn create_privacy_preserving_join(
        &self,
        left_data: &serde_json::Value,
        right_data: &serde_json::Value,
        join_config: &PrivacyPreservingJoinConfig,
        join_keys: &[String],
    ) -> Result<serde_json::Value, PrivacyError> {
        if join_config.enable_psi {
            self.private_set_intersection_join(left_data, right_data, join_keys).await
        } else if join_config.enable_smpc {
            self.secure_multiparty_join(left_data, right_data, join_keys).await
        } else {
            // Fallback to basic anonymized join
            self.anonymized_join(left_data, right_data, join_keys).await
        }
    }

    /// Perform private set intersection join
    async fn private_set_intersection_join(
        &self,
        left_data: &serde_json::Value,
        right_data: &serde_json::Value,
        join_keys: &[String],
    ) -> Result<serde_json::Value, PrivacyError> {
        // Implementation of PSI-based join
        // This would involve cryptographic protocols for secure intersection
        
        // For now, implement a simplified version
        let mut result = Vec::new();
        
        if let (serde_json::Value::Array(left_records), serde_json::Value::Array(right_records)) = 
            (left_data, right_data) {
            
            for left_record in left_records {
                for right_record in right_records {
                    if self.records_match_on_keys(left_record, right_record, join_keys) {
                        let mut joined_record = left_record.clone();
                        if let (serde_json::Value::Object(left_map), serde_json::Value::Object(right_map)) = 
                            (&mut joined_record, right_record) {
                            
                            for (key, value) in right_map {
                                if !join_keys.contains(key) {
                                    left_map.insert(format!("right_{}", key), value.clone());
                                }
                            }
                        }
                        result.push(joined_record);
                    }
                }
            }
        }
        
        Ok(serde_json::Value::Array(result))
    }

    /// Perform secure multiparty computation join
    async fn secure_multiparty_join(
        &self,
        _left_data: &serde_json::Value,
        _right_data: &serde_json::Value,
        _join_keys: &[String],
    ) -> Result<serde_json::Value, PrivacyError> {
        // Implementation would involve SMPC protocols
        Err(PrivacyError::AnonymizationFailed(
            "SMPC join not yet implemented".to_string()
        ))
    }

    /// Perform anonymized join
    async fn anonymized_join(
        &self,
        left_data: &serde_json::Value,
        right_data: &serde_json::Value,
        join_keys: &[String],
    ) -> Result<serde_json::Value, PrivacyError> {
        // Standard join with post-processing anonymization
        let mut result = Vec::new();
        
        if let (serde_json::Value::Array(left_records), serde_json::Value::Array(right_records)) = 
            (left_data, right_data) {
            
            for left_record in left_records {
                for right_record in right_records {
                    if self.records_match_on_keys(left_record, right_record, join_keys) {
                        let mut joined_record = left_record.clone();
                        if let (serde_json::Value::Object(left_map), serde_json::Value::Object(right_map)) = 
                            (&mut joined_record, right_record) {
                            
                            for (key, value) in right_map {
                                if !join_keys.contains(key) {
                                    left_map.insert(format!("right_{}", key), value.clone());
                                }
                            }
                        }
                        result.push(joined_record);
                    }
                }
            }
        }
        
        Ok(serde_json::Value::Array(result))
    }

    /// Check if records match on join keys
    fn records_match_on_keys(
        &self,
        left: &serde_json::Value,
        right: &serde_json::Value,
        join_keys: &[String],
    ) -> bool {
        if let (serde_json::Value::Object(left_map), serde_json::Value::Object(right_map)) = 
            (left, right) {
            
            for key in join_keys {
                if left_map.get(key) != right_map.get(key) {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
}

/// Result of privacy protection application
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PrivacyProtectionResult {
    /// Whether differential privacy was applied
    pub differential_privacy_applied: bool,
    /// Epsilon value used for differential privacy
    pub epsilon_used: f64,
    /// Whether anonymization was applied
    pub anonymization_applied: bool,
    /// Whether GDPR compliance was checked
    pub gdpr_compliant: bool,
    /// Whether k-anonymity was applied
    pub k_anonymity_applied: bool,
    /// K value used
    pub k_value: Option<usize>,
    /// Whether l-diversity was applied
    pub l_diversity_applied: bool,
    /// L value used
    pub l_value: Option<usize>,
    /// Privacy level achieved
    pub privacy_level_achieved: PrivacyLevel,
    /// Warning messages
    pub warnings: Vec<String>,
}

/// Sensitive data detector
#[derive(Debug)]
pub struct SensitiveDataDetector {
    /// Sensitive patterns (regex)
    patterns: Vec<regex::Regex>,
}

impl SensitiveDataDetector {
    /// Create new sensitive data detector
    pub fn new() -> Self {
        let patterns = vec![
            // Email patterns
            regex::Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap(),
            // Phone number patterns
            regex::Regex::new(r"\b\d{3}-\d{3}-\d{4}\b").unwrap(),
            // SSN patterns
            regex::Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
            // Credit card patterns
            regex::Regex::new(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b").unwrap(),
        ];

        Self { patterns }
    }

    /// Detect sensitive data in value
    pub fn detect_sensitive_data(
        &self,
        data: &serde_json::Value,
        custom_patterns: &[String],
    ) -> Result<(), PrivacyError> {
        let mut custom_regexes = Vec::new();
        for pattern in custom_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                custom_regexes.push(regex);
            }
        }

        self.check_value_recursive(data, &custom_regexes)?;
        Ok(())
    }

    /// Check value recursively for sensitive data
    fn check_value_recursive(
        &self,
        value: &serde_json::Value,
        custom_regexes: &[regex::Regex],
    ) -> Result<(), PrivacyError> {
        match value {
            serde_json::Value::String(s) => {
                for pattern in &self.patterns {
                    if pattern.is_match(s) {
                        return Err(PrivacyError::SensitiveDataDetected(
                            format!("Sensitive pattern detected: {}", pattern.as_str())
                        ));
                    }
                }
                for pattern in custom_regexes {
                    if pattern.is_match(s) {
                        return Err(PrivacyError::SensitiveDataDetected(
                            format!("Custom sensitive pattern detected: {}", pattern.as_str())
                        ));
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    self.check_value_recursive(item, custom_regexes)?;
                }
            }
            serde_json::Value::Object(map) => {
                for value in map.values() {
                    self.check_value_recursive(value, custom_regexes)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// GDPR compliance checker
#[derive(Debug)]
pub struct GdprComplianceChecker {
    /// Audit log
    audit_log: Arc<RwLock<Vec<GdprAuditEntry>>>,
}

/// GDPR audit entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GdprAuditEntry {
    /// Timestamp
    pub timestamp: SystemTime,
    /// User ID
    pub user_id: String,
    /// Action taken
    pub action: String,
    /// Lawful basis
    pub lawful_basis: LawfulBasis,
    /// Data processed
    pub data_description: String,
}

impl GdprComplianceChecker {
    /// Create new GDPR compliance checker
    pub fn new() -> Self {
        Self {
            audit_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Check GDPR compliance
    pub fn check_compliance(
        &self,
        _data: &serde_json::Value,
        config: &GdprConfig,
    ) -> Result<(), PrivacyError> {
        // Check retention period
        // Check consent requirements
        // Check lawful basis
        // Log audit entry
        
        if config.consent_required {
            // Would check for valid consent in real implementation
        }

        if config.dpia_required {
            // Would verify DPIA completion
        }

        Ok(())
    }

    /// Log GDPR audit entry
    pub fn log_audit_entry(&self, entry: GdprAuditEntry) {
        let mut log = self.audit_log.write().unwrap();
        log.push(entry);
    }

    /// Get audit log
    pub fn get_audit_log(&self) -> Vec<GdprAuditEntry> {
        self.audit_log.read().unwrap().clone()
    }
}

/// Data anonymizer
#[derive(Debug)]
pub struct DataAnonymizer;

impl DataAnonymizer {
    /// Create new data anonymizer
    pub fn new() -> Self {
        Self
    }

    /// Anonymize data according to rules
    pub fn anonymize_data(
        &self,
        data: &mut serde_json::Value,
        rules: &[AnonymizationRule],
    ) -> Result<(), PrivacyError> {
        for rule in rules {
            self.apply_anonymization_rule(data, rule)?;
        }
        Ok(())
    }

    /// Apply single anonymization rule
    fn apply_anonymization_rule(
        &self,
        data: &mut serde_json::Value,
        rule: &AnonymizationRule,
    ) -> Result<(), PrivacyError> {
        let field_regex = regex::Regex::new(&rule.field_pattern)
            .map_err(|e| PrivacyError::AnonymizationFailed(format!("Invalid regex: {}", e)))?;

        self.apply_rule_recursive(data, &field_regex, rule)?;
        Ok(())
    }

    /// Apply rule recursively
    fn apply_rule_recursive(
        &self,
        value: &mut serde_json::Value,
        field_regex: &regex::Regex,
        rule: &AnonymizationRule,
    ) -> Result<(), PrivacyError> {
        match value {
            serde_json::Value::Object(map) => {
                for (key, val) in map.iter_mut() {
                    if field_regex.is_match(key) {
                        self.apply_technique(val, &rule.technique, &rule.config)?;
                    } else {
                        self.apply_rule_recursive(val, field_regex, rule)?;
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr.iter_mut() {
                    self.apply_rule_recursive(item, field_regex, rule)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Apply anonymization technique
    fn apply_technique(
        &self,
        value: &mut serde_json::Value,
        technique: &AnonymizationTechnique,
        _config: &serde_json::Value,
    ) -> Result<(), PrivacyError> {
        match technique {
            AnonymizationTechnique::Suppression => {
                *value = serde_json::Value::String("***".to_string());
            }
            AnonymizationTechnique::Generalization => {
                if let Some(s) = value.as_str() {
                    *value = serde_json::Value::String(
                        s.chars().take(3).collect::<String>() + "***"
                    );
                }
            }
            AnonymizationTechnique::Hashing => {
                if let Some(s) = value.as_str() {
                    use std::collections::hash_map::DefaultHasher;
                    let mut hasher = DefaultHasher::new();
                    s.hash(&mut hasher);
                    let hash = hasher.finish();
                    *value = serde_json::Value::String(format!("hash_{:x}", hash));
                }
            }
            AnonymizationTechnique::Masking => {
                if let Some(s) = value.as_str() {
                    let masked = s.chars()
                        .enumerate()
                        .map(|(i, c)| if i < s.len() / 2 { c } else { '*' })
                        .collect::<String>();
                    *value = serde_json::Value::String(masked);
                }
            }
            _ => {
                return Err(PrivacyError::AnonymizationFailed(
                    format!("Technique {:?} not implemented", technique)
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_privacy_manager_creation() {
        let manager = PrivacyManager::new(10.0, Duration::from_hours(24));
        assert!(manager.budget_tracker.remaining_budget("test_user") == 10.0);
    }

    #[tokio::test]
    async fn test_budget_tracking() {
        let tracker = PrivacyBudgetTracker::new(5.0, Duration::from_hours(1));
        
        // Test budget consumption
        assert!(tracker.consume_budget("user1", 2.0).is_ok());
        assert!(tracker.remaining_budget("user1") == 3.0);
        
        // Test budget exhaustion
        assert!(tracker.consume_budget("user1", 4.0).is_err());
    }

    #[test]
    fn test_sensitive_data_detection() {
        let detector = SensitiveDataDetector::new();
        let data = serde_json::json!({
            "email": "test@example.com",
            "phone": "123-456-7890"
        });
        
        assert!(detector.detect_sensitive_data(&data, &[]).is_err());
    }

    #[test]
    fn test_data_anonymization() {
        let anonymizer = DataAnonymizer::new();
        let mut data = serde_json::json!({
            "name": "John Doe",
            "email": "john@example.com"
        });
        
        let rules = vec![
            AnonymizationRule {
                field_pattern: "name".to_string(),
                technique: AnonymizationTechnique::Suppression,
                config: serde_json::Value::Null,
            }
        ];
        
        assert!(anonymizer.anonymize_data(&mut data, &rules).is_ok());
        assert_eq!(data["name"], "***");
    }
}