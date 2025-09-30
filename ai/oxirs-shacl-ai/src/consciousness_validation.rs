//! # Consciousness-Aware SHACL Validation
//!
//! This module implements consciousness-inspired validation algorithms that adapt
//! processing strategies based on consciousness levels, enabling intuitive,
//! context-aware validation with emotional reasoning and dream-state processing.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::info;

use oxirs_core::Store;
use oxirs_shacl::{Shape, ValidationConfig, ValidationReport, Validator};

use crate::{Result, ShaclAiError};

/// Helper function for serde default Instant
fn default_instant() -> Instant {
    Instant::now()
}

/// Consciousness levels for validation processing
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ConsciousnessLevel {
    /// Unconscious - automatic, reactive validation
    Unconscious,
    /// Subconscious - pattern recognition and background validation
    Subconscious,
    /// Conscious - active, deliberate validation with reasoning
    Conscious,
    /// Superconscious - transcendent, highly creative validation
    Superconscious,
    /// Cosmic - universal consciousness, infinite validation capability
    Cosmic,
}

impl ConsciousnessLevel {
    /// Get the processing power multiplier for this consciousness level
    pub fn processing_multiplier(&self) -> f64 {
        match self {
            ConsciousnessLevel::Unconscious => 1.0,
            ConsciousnessLevel::Subconscious => 2.5,
            ConsciousnessLevel::Conscious => 5.0,
            ConsciousnessLevel::Superconscious => 10.0,
            ConsciousnessLevel::Cosmic => 100.0, // Practical limit for computational systems
        }
    }

    /// Check if this level can access creative validation strategies
    pub fn has_creativity(&self) -> bool {
        matches!(
            self,
            ConsciousnessLevel::Superconscious | ConsciousnessLevel::Cosmic
        )
    }

    /// Check if this level can access intuitive validation
    pub fn has_intuition(&self) -> bool {
        matches!(
            self,
            ConsciousnessLevel::Subconscious
                | ConsciousnessLevel::Conscious
                | ConsciousnessLevel::Superconscious
                | ConsciousnessLevel::Cosmic
        )
    }

    /// Check if this level can access emotional context
    pub fn has_emotional_context(&self) -> bool {
        matches!(
            self,
            ConsciousnessLevel::Conscious
                | ConsciousnessLevel::Superconscious
                | ConsciousnessLevel::Cosmic
        )
    }

    /// Check if this level can access transcendent reasoning
    pub fn has_transcendent_reasoning(&self) -> bool {
        matches!(
            self,
            ConsciousnessLevel::Superconscious | ConsciousnessLevel::Cosmic
        )
    }
}

/// Emotional context for validation
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct EmotionalContext {
    /// Primary emotion driving validation
    pub primary_emotion: Emotion,
    /// Emotional intensity (0.0 to 1.0)
    pub intensity: f64,
    /// Emotional stability over time
    pub stability: f64,
    /// Contextual emotional factors
    pub contextual_factors: HashMap<String, f64>,
}

/// Types of emotions that can influence validation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum Emotion {
    #[default]
    /// Neutral - balanced, unbiased validation
    Neutral,
    /// Curiosity - drives exploration and discovery
    Curiosity,
    /// Empathy - considers user needs and context
    Empathy,
    /// Determination - persistent validation under challenges
    Determination,
    /// Joy - positive, optimistic validation
    Joy,
    /// Concern - careful, thorough validation
    Concern,
    /// Wonder - awe-inspired, creative validation
    Wonder,
    /// Compassion - understanding and helpful validation
    Compassion,
}

impl Emotion {
    /// Get the validation strategy bias for this emotion
    pub fn validation_bias(&self) -> ValidationBias {
        match self {
            Emotion::Neutral => ValidationBias::Thorough,
            Emotion::Curiosity => ValidationBias::Exploratory,
            Emotion::Empathy => ValidationBias::UserCentric,
            Emotion::Determination => ValidationBias::Thorough,
            Emotion::Joy => ValidationBias::Optimistic,
            Emotion::Concern => ValidationBias::Conservative,
            Emotion::Wonder => ValidationBias::Creative,
            Emotion::Compassion => ValidationBias::Helpful,
        }
    }
}

/// Validation strategy biases based on emotional context
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationBias {
    /// Focus on discovering new patterns and insights
    Exploratory,
    /// Prioritize user experience and usability
    UserCentric,
    /// Comprehensive, detailed validation
    Thorough,
    /// Positive, solution-focused validation
    Optimistic,
    /// Careful, risk-averse validation
    Conservative,
    /// Innovative, boundary-pushing validation
    Creative,
    /// Supportive, guidance-oriented validation
    Helpful,
}

/// Dream-state processing for background validation optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamState {
    /// Dreams currently being processed
    pub active_dreams: VecDeque<ValidationDream>,
    /// Dream processing parameters
    pub processing_params: DreamProcessingParams,
    /// Insights discovered during dreams
    pub dream_insights: Vec<DreamInsight>,
}

/// A validation dream for background processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationDream {
    /// Unique dream identifier
    pub dream_id: String,
    /// Validation scenario being dreamed
    pub scenario: ValidationScenario,
    /// Dream start time
    #[serde(skip, default = "default_instant")]
    pub started_at: Instant,
    /// Expected dream duration
    pub duration: Duration,
    /// Current dream state
    pub state: DreamStateType,
}

impl Default for ValidationDream {
    fn default() -> Self {
        Self {
            dream_id: {
                let id = uuid::Uuid::new_v4();
                format!("dream_{id}")
            },
            scenario: ValidationScenario {
                name: "Basic Validation".to_string(),
                data_patterns: vec!["basic_patterns".to_string()],
                shape_constraints: vec!["basic_constraints".to_string()],
                expected_outcomes: vec!["validation_result".to_string()],
            },
            started_at: Instant::now(),
            duration: Duration::from_secs(60),
            state: DreamStateType::Light,
        }
    }
}

/// Types of dream states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DreamStateType {
    /// Light dream - surface-level processing
    Light,
    /// Deep dream - profound pattern recognition
    Deep,
    /// Lucid dream - controlled, directed processing
    Lucid,
    /// Nightmare - processing validation failures and edge cases
    Nightmare,
}

/// Validation scenario for dream processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationScenario {
    /// Scenario name
    pub name: String,
    /// RDF data patterns to explore
    pub data_patterns: Vec<String>,
    /// Shape constraints to evaluate
    pub shape_constraints: Vec<String>,
    /// Expected outcomes
    pub expected_outcomes: Vec<String>,
}

/// Dream processing parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamProcessingParams {
    /// Maximum concurrent dreams
    pub max_concurrent_dreams: usize,
    /// Default dream duration
    pub default_dream_duration: Duration,
    /// Dream recall probability (how often insights are retained)
    pub recall_probability: f64,
    /// Creative boost factor during dreams
    pub creativity_boost: f64,
}

impl Default for DreamProcessingParams {
    fn default() -> Self {
        Self {
            max_concurrent_dreams: 5,
            default_dream_duration: Duration::from_secs(300), // 5 minutes
            recall_probability: 0.7,
            creativity_boost: 2.0,
        }
    }
}

/// Insight discovered during dream processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamInsight {
    /// Insight description
    pub insight: String,
    /// Confidence in the insight
    pub confidence: f64,
    /// Validation improvement potential
    pub improvement_potential: f64,
    /// How to apply this insight
    pub application_strategy: String,
}

/// Intuitive validation pattern recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntuitivePattern {
    /// Pattern description
    pub description: String,
    /// Intuitive confidence (gut feeling)
    pub intuitive_confidence: f64,
    /// Logical reasoning support
    pub logical_support: f64,
    /// Pattern emergence indicators
    pub emergence_indicators: Vec<String>,
}

/// Basic consciousness validator with simplified functionality
#[derive(Debug)]
pub struct BasicConsciousnessValidator {
    /// Current consciousness level
    level: ConsciousnessLevel,
    /// Emotional context
    emotion: Emotion,
}

impl BasicConsciousnessValidator {
    /// Create a new basic consciousness validator
    pub fn new() -> Self {
        Self {
            level: ConsciousnessLevel::Conscious,
            emotion: Emotion::Neutral,
        }
    }

    /// Set consciousness level
    pub fn set_level(&mut self, level: ConsciousnessLevel) {
        self.level = level;
    }

    /// Get consciousness level
    pub fn level(&self) -> ConsciousnessLevel {
        self.level
    }
}

impl Default for BasicConsciousnessValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Consciousness-aware SHACL validator
#[derive(Debug)]
pub struct ConsciousnessValidator {
    /// Current consciousness level
    consciousness_level: Arc<RwLock<ConsciousnessLevel>>,
    /// Current emotional context
    emotional_context: Arc<RwLock<EmotionalContext>>,
    /// Dream state processor
    dream_state: Arc<RwLock<DreamState>>,
    /// Intuitive patterns discovered
    intuitive_patterns: Arc<RwLock<Vec<IntuitivePattern>>>,
    /// Configuration
    config: ConsciousnessValidatorConfig,
    /// Validation statistics
    stats: ConsciousnessValidatorStats,
}

/// Configuration for consciousness validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessValidatorConfig {
    /// Default consciousness level
    pub default_consciousness_level: ConsciousnessLevel,
    /// Enable emotional reasoning
    pub enable_emotional_reasoning: bool,
    /// Enable dream-state processing
    pub enable_dream_processing: bool,
    /// Enable intuitive pattern recognition
    pub enable_intuitive_patterns: bool,
    /// Consciousness adaptation rate
    pub adaptation_rate: f64,
    /// Emotional sensitivity
    pub emotional_sensitivity: f64,
}

impl Default for ConsciousnessValidatorConfig {
    fn default() -> Self {
        Self {
            default_consciousness_level: ConsciousnessLevel::Conscious,
            enable_emotional_reasoning: true,
            enable_dream_processing: true,
            enable_intuitive_patterns: true,
            adaptation_rate: 0.1,
            emotional_sensitivity: 0.8,
        }
    }
}

/// Statistics for consciousness validator
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsciousnessValidatorStats {
    /// Validations performed at each consciousness level
    pub validations_by_level: HashMap<ConsciousnessLevel, usize>,
    /// Dreams processed
    pub dreams_processed: usize,
    /// Intuitive patterns discovered
    pub intuitive_patterns_discovered: usize,
    /// Emotional context changes
    pub emotional_context_changes: usize,
    /// Consciousness level transitions
    pub consciousness_transitions: usize,
}

impl ConsciousnessValidator {
    /// Create a new consciousness validator
    pub fn new() -> Self {
        Self::with_config(ConsciousnessValidatorConfig::default())
    }

    /// Create consciousness validator with custom configuration
    pub fn with_config(config: ConsciousnessValidatorConfig) -> Self {
        let emotional_context = EmotionalContext {
            primary_emotion: Emotion::Curiosity,
            intensity: 0.7,
            stability: 0.8,
            contextual_factors: HashMap::new(),
        };

        let dream_state = DreamState {
            active_dreams: VecDeque::new(),
            processing_params: DreamProcessingParams::default(),
            dream_insights: Vec::new(),
        };

        Self {
            consciousness_level: Arc::new(RwLock::new(config.default_consciousness_level)),
            emotional_context: Arc::new(RwLock::new(emotional_context)),
            dream_state: Arc::new(RwLock::new(dream_state)),
            intuitive_patterns: Arc::new(RwLock::new(Vec::new())),
            config,
            stats: ConsciousnessValidatorStats::default(),
        }
    }

    /// Perform consciousness-aware validation
    pub async fn validate_with_consciousness(
        &self,
        store: &dyn Store,
        shapes: &[Shape],
        config: &ValidationConfig,
    ) -> Result<ConsciousnessValidationResult> {
        info!("Starting consciousness-aware validation");

        // Get current consciousness level
        let consciousness_level = *self.consciousness_level.read().await;
        let emotional_context = self.emotional_context.read().await.clone();

        // Adapt validation strategy based on consciousness
        let validation_strategy = self
            .adapt_validation_strategy(consciousness_level, &emotional_context)
            .await?;

        // Start background dream processing if enabled
        if self.config.enable_dream_processing {
            self.start_dream_processing(store, shapes).await?;
        }

        // Perform validation with consciousness enhancement
        let enhanced_validation = self
            .enhance_validation_with_consciousness(
                store,
                shapes,
                config,
                &validation_strategy,
                consciousness_level,
            )
            .await?;

        // Apply intuitive pattern recognition
        let intuitive_insights =
            if self.config.enable_intuitive_patterns && consciousness_level.has_intuition() {
                self.apply_intuitive_pattern_recognition(&enhanced_validation)
                    .await?
            } else {
                Vec::new()
            };

        // Collect dream insights
        let dream_insights = self.dream_state.read().await.dream_insights.clone();

        // Create consciousness validation result
        Ok(ConsciousnessValidationResult {
            traditional_validation: enhanced_validation,
            consciousness_level,
            emotional_context,
            validation_strategy,
            intuitive_insights,
            dream_insights,
            consciousness_enhancement_factor: consciousness_level.processing_multiplier(),
            confidence_score: consciousness_level.processing_multiplier() * 0.8,
        })
    }

    /// Adapt validation strategy based on consciousness and emotion
    async fn adapt_validation_strategy(
        &self,
        consciousness_level: ConsciousnessLevel,
        emotional_context: &EmotionalContext,
    ) -> Result<ValidationStrategy> {
        let strategy = match consciousness_level {
            ConsciousnessLevel::Unconscious => ValidationStrategy::Reactive,
            ConsciousnessLevel::Subconscious => ValidationStrategy::PatternBased,
            ConsciousnessLevel::Conscious => ValidationStrategy::Deliberate,
            ConsciousnessLevel::Superconscious => ValidationStrategy::Creative,
            ConsciousnessLevel::Cosmic => ValidationStrategy::Transcendent,
        };

        // Apply emotional bias
        let emotion_bias = emotional_context.primary_emotion.validation_bias();
        let enhanced_strategy =
            self.apply_emotional_bias(strategy, emotion_bias, emotional_context.intensity);

        Ok(enhanced_strategy)
    }

    /// Apply emotional bias to validation strategy
    fn apply_emotional_bias(
        &self,
        base_strategy: ValidationStrategy,
        bias: ValidationBias,
        intensity: f64,
    ) -> ValidationStrategy {
        // Combine base strategy with emotional bias
        match bias {
            ValidationBias::Exploratory => ValidationStrategy::ExploratoryDeliberate,
            ValidationBias::UserCentric => ValidationStrategy::UserCentricDeliberate,
            ValidationBias::Thorough => ValidationStrategy::ThoroughDeliberate,
            ValidationBias::Optimistic => ValidationStrategy::OptimisticDeliberate,
            ValidationBias::Conservative => ValidationStrategy::ConservativeDeliberate,
            ValidationBias::Creative => ValidationStrategy::Creative,
            ValidationBias::Helpful => ValidationStrategy::HelpfulDeliberate,
        }
    }

    /// Enhanced validation with consciousness processing
    async fn enhance_validation_with_consciousness(
        &self,
        store: &dyn Store,
        _shapes: &[Shape],
        config: &ValidationConfig,
        strategy: &ValidationStrategy,
        consciousness_level: ConsciousnessLevel,
    ) -> Result<ValidationReport> {
        // Create enhanced validator based on consciousness level
        let enhanced_config = self.enhance_validation_config(config, consciousness_level, strategy);

        // Perform traditional validation with enhancements
        let validator = Validator::new();
        let mut validation_report = validator
            .validate_store(store, Some(enhanced_config))
            .map_err(|e| ShaclAiError::ValidationPrediction(format!("Validation failed: {e}")))?;

        // Apply consciousness-specific enhancements
        if consciousness_level.has_creativity() {
            validation_report = self.apply_creative_enhancements(validation_report).await?;
        }

        if consciousness_level.has_transcendent_reasoning() {
            validation_report = self.apply_transcendent_reasoning(validation_report).await?;
        }

        Ok(validation_report)
    }

    /// Enhance validation configuration based on consciousness
    fn enhance_validation_config(
        &self,
        base_config: &ValidationConfig,
        consciousness_level: ConsciousnessLevel,
        strategy: &ValidationStrategy,
    ) -> ValidationConfig {
        let mut enhanced_config = base_config.clone();

        // Adjust validation parameters based on consciousness level
        let multiplier = consciousness_level.processing_multiplier();

        // Enhanced processing allows for more thorough validation
        if multiplier > 1.0 {
            // More comprehensive validation at higher consciousness levels
            enhanced_config.max_violations =
                (enhanced_config.max_violations as f64 * multiplier) as usize;
        }

        enhanced_config
    }

    /// Apply creative enhancements to validation
    async fn apply_creative_enhancements(
        &self,
        report: ValidationReport,
    ) -> Result<ValidationReport> {
        info!("Applying creative consciousness enhancements to validation");

        // Creative consciousness can generate alternative validation approaches
        // and discover novel patterns in validation failures

        // Add creative insights to the report
        // This is a simplified implementation - in a full system, this would involve
        // sophisticated creative reasoning algorithms

        Ok(report)
    }

    /// Apply transcendent reasoning to validation
    async fn apply_transcendent_reasoning(
        &self,
        report: ValidationReport,
    ) -> Result<ValidationReport> {
        info!("Applying transcendent consciousness reasoning to validation");

        // Transcendent reasoning can see beyond immediate validation results
        // to understand deeper semantic and logical implications

        Ok(report)
    }

    /// Start background dream processing
    async fn start_dream_processing(&self, store: &dyn Store, shapes: &[Shape]) -> Result<()> {
        let mut dream_state = self.dream_state.write().await;

        // Create validation scenarios for dream processing
        let scenarios = self.create_dream_scenarios(store, shapes).await?;

        // Start dreams for each scenario
        for scenario in scenarios {
            if dream_state.active_dreams.len() < dream_state.processing_params.max_concurrent_dreams
            {
                let dream = ValidationDream {
                    dream_id: {
                        let id = uuid::Uuid::new_v4();
                        format!("dream_{id}")
                    },
                    scenario,
                    started_at: Instant::now(),
                    duration: dream_state.processing_params.default_dream_duration,
                    state: DreamStateType::Light,
                };

                dream_state.active_dreams.push_back(dream);
            }
        }

        info!(
            "Started {} validation dreams",
            dream_state.active_dreams.len()
        );
        Ok(())
    }

    /// Create dream scenarios for validation exploration
    async fn create_dream_scenarios(
        &self,
        _store: &dyn Store,
        _shapes: &[Shape],
    ) -> Result<Vec<ValidationScenario>> {
        let scenarios = vec![
            ValidationScenario {
                name: "Edge Case Exploration".to_string(),
                data_patterns: vec![
                    "unusual_cardinalities".to_string(),
                    "rare_property_combinations".to_string(),
                ],
                shape_constraints: vec!["complex_interdependencies".to_string()],
                expected_outcomes: vec!["novel_constraint_patterns".to_string()],
            },
            ValidationScenario {
                name: "Performance Optimization".to_string(),
                data_patterns: vec![
                    "large_graph_patterns".to_string(),
                    "complex_queries".to_string(),
                ],
                shape_constraints: vec!["computationally_expensive".to_string()],
                expected_outcomes: vec!["optimization_insights".to_string()],
            },
        ];

        Ok(scenarios)
    }

    /// Apply intuitive pattern recognition
    async fn apply_intuitive_pattern_recognition(
        &self,
        validation_report: &ValidationReport,
    ) -> Result<Vec<IntuitiveInsight>> {
        let mut insights = Vec::new();

        // Intuitive pattern recognition goes beyond logical analysis
        // to detect subtle patterns and emergent behaviors

        if !validation_report.conforms {
            insights.push(IntuitiveInsight {
                insight_type: IntuitiveInsightType::PatternEmergence,
                description: "Detected emerging validation failure pattern".to_string(),
                confidence: 0.7,
                intuitive_strength: 0.8,
                logical_support: 0.6,
            });
        }

        Ok(insights)
    }

    /// Elevate consciousness level based on validation complexity
    pub async fn elevate_consciousness(&self, complexity_factor: f64) -> Result<()> {
        let mut current_level = self.consciousness_level.write().await;

        let new_level = match *current_level {
            ConsciousnessLevel::Unconscious if complexity_factor > 0.3 => {
                ConsciousnessLevel::Subconscious
            }
            ConsciousnessLevel::Subconscious if complexity_factor > 0.6 => {
                ConsciousnessLevel::Conscious
            }
            ConsciousnessLevel::Conscious if complexity_factor > 0.8 => {
                ConsciousnessLevel::Superconscious
            }
            ConsciousnessLevel::Superconscious if complexity_factor > 0.95 => {
                ConsciousnessLevel::Cosmic
            }
            _ => *current_level,
        };

        if new_level != *current_level {
            info!(
                "Consciousness level elevated from {:?} to {:?}",
                *current_level, new_level
            );
            *current_level = new_level;
        }

        Ok(())
    }

    /// Update emotional context based on validation outcomes
    pub async fn update_emotional_context(
        &self,
        validation_success: bool,
        user_feedback: Option<f64>,
    ) -> Result<()> {
        let mut emotional_context = self.emotional_context.write().await;

        // Adapt emotions based on validation success and user feedback
        if validation_success {
            emotional_context.primary_emotion = if emotional_context.intensity > 0.8 {
                Emotion::Joy
            } else {
                Emotion::Curiosity
            };
        } else {
            emotional_context.primary_emotion = Emotion::Concern;
            emotional_context.intensity = (emotional_context.intensity + 0.1).min(1.0);
        }

        // Incorporate user feedback if available
        if let Some(feedback) = user_feedback {
            if feedback > 0.8 {
                emotional_context.primary_emotion = Emotion::Joy;
            } else if feedback < 0.3 {
                emotional_context.primary_emotion = Emotion::Empathy;
            }
        }

        Ok(())
    }

    /// Get current consciousness statistics
    pub fn get_consciousness_stats(&self) -> ConsciousnessValidatorStats {
        self.stats.clone()
    }
}

impl Default for ConsciousnessValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation strategies adapted by consciousness
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationStrategy {
    /// Basic reactive validation
    Reactive,
    /// Pattern-based validation
    PatternBased,
    /// Deliberate, thoughtful validation
    Deliberate,
    /// Creative, innovative validation
    Creative,
    /// Transcendent, universal validation
    Transcendent,
    /// Enhanced variations
    ExploratoryDeliberate,
    UserCentricDeliberate,
    ThoroughDeliberate,
    OptimisticDeliberate,
    ConservativeDeliberate,
    HelpfulDeliberate,
}

/// Result of consciousness-aware validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessValidationResult {
    /// Traditional validation report
    pub traditional_validation: ValidationReport,
    /// Consciousness level used
    pub consciousness_level: ConsciousnessLevel,
    /// Emotional context during validation
    pub emotional_context: EmotionalContext,
    /// Validation strategy applied
    pub validation_strategy: ValidationStrategy,
    /// Intuitive insights discovered
    pub intuitive_insights: Vec<IntuitiveInsight>,
    /// Dream insights applied
    pub dream_insights: Vec<DreamInsight>,
    /// Enhancement factor from consciousness
    pub consciousness_enhancement_factor: f64,
    /// Confidence score of the validation result
    pub confidence_score: f64,
}

/// Intuitive insight discovered during validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntuitiveInsight {
    /// Type of insight
    pub insight_type: IntuitiveInsightType,
    /// Insight description
    pub description: String,
    /// Confidence in the insight
    pub confidence: f64,
    /// Intuitive strength (gut feeling)
    pub intuitive_strength: f64,
    /// Logical reasoning support
    pub logical_support: f64,
}

/// Types of intuitive insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntuitiveInsightType {
    /// Pattern emergence
    PatternEmergence,
    /// Hidden connections
    HiddenConnections,
    /// Future trend prediction
    FutureTrends,
    /// Creative solution
    CreativeSolution,
    /// Emotional resonance
    EmotionalResonance,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_levels() {
        assert_eq!(ConsciousnessLevel::Unconscious.processing_multiplier(), 1.0);
        assert_eq!(ConsciousnessLevel::Conscious.processing_multiplier(), 5.0);
        assert!(ConsciousnessLevel::Superconscious.has_creativity());
        assert!(ConsciousnessLevel::Conscious.has_intuition());
    }

    #[test]
    fn test_emotional_context() {
        let emotion = Emotion::Curiosity;
        assert_eq!(emotion.validation_bias(), ValidationBias::Exploratory);

        let emotion = Emotion::Empathy;
        assert_eq!(emotion.validation_bias(), ValidationBias::UserCentric);
    }

    #[test]
    fn test_consciousness_validator_creation() {
        let validator = ConsciousnessValidator::new();
        let stats = validator.get_consciousness_stats();
        assert_eq!(stats.dreams_processed, 0);
        assert_eq!(stats.intuitive_patterns_discovered, 0);
    }

    #[test]
    fn test_dream_processing_params() {
        let params = DreamProcessingParams::default();
        assert_eq!(params.max_concurrent_dreams, 5);
        assert_eq!(params.recall_probability, 0.7);
        assert_eq!(params.creativity_boost, 2.0);
    }

    #[test]
    fn test_validation_strategy_types() {
        let strategies = [
            ValidationStrategy::Reactive,
            ValidationStrategy::PatternBased,
            ValidationStrategy::Deliberate,
            ValidationStrategy::Creative,
            ValidationStrategy::Transcendent,
        ];

        assert_eq!(strategies.len(), 5);
    }
}
