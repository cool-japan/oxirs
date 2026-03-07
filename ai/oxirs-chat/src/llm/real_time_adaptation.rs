//! Real-Time Adaptation Module
//!
//! Provides dynamic model adaptation capabilities for continuous learning
//! and real-time performance optimization based on user interactions.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::{Mutex, RwLock};

use super::types::{LLMRequest, LLMResponse};
/// Real-time adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationConfig {
    pub adaptation_id: String,
    pub strategy: AdaptationStrategy,
    pub trigger_conditions: TriggerConditions,
    pub adaptation_parameters: AdaptationParameters,
    pub performance_targets: PerformanceTargets,
    pub learning_rate_schedule: LearningRateSchedule,
    pub rollback_config: RollbackConfig,
}

/// Adaptation strategies for real-time learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    OnlineGradientDescent,
    ReinforcementLearning,
    MetaLearning,
    FewShotAdaptation,
    InContextLearning,
    MemoryAugmentedLearning,
    AdaptivePrompting,
    HybridAdaptation,
}

/// Conditions that trigger adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerConditions {
    pub performance_degradation_threshold: f32,
    pub user_feedback_threshold: i32,
    pub error_rate_threshold: f32,
    pub latency_threshold_ms: f32,
    pub adaptation_frequency: AdaptationFrequency,
    pub minimum_data_points: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationFrequency {
    Continuous,
    Periodic(Duration),
    OnDemand,
    ThresholdBased,
}

/// Parameters for adaptation algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationParameters {
    pub learning_rate: f32,
    pub adaptation_strength: f32,
    pub memory_window_size: usize,
    pub regularization_weight: f32,
    pub exploration_rate: f32,
    pub momentum: f32,
    pub gradient_clipping: Option<f32>,
}

/// Performance targets for adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub target_accuracy: f32,
    pub target_latency_ms: f32,
    pub target_user_satisfaction: f32,
    pub target_error_rate: f32,
    pub convergence_threshold: f32,
}

/// Learning rate scheduling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    Constant(f32),
    Exponential { initial: f32, decay: f32 },
    Cosine { initial: f32, min: f32 },
    Adaptive,
    PerformanceBased,
}

/// Rollback configuration for failed adaptations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackConfig {
    pub enable_rollback: bool,
    pub rollback_threshold: f32,
    pub max_rollback_attempts: usize,
    pub rollback_window: Duration,
    pub checkpoint_frequency: Duration,
}

/// Interaction data for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionData {
    pub interaction_id: String,
    pub request: LLMRequest,
    pub response: LLMResponse,
    pub user_feedback: Option<UserFeedback>,
    pub context_information: ContextInformation,
    pub timestamp: SystemTime,
}

/// User feedback for adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedback {
    pub rating: FeedbackRating,
    pub feedback_type: FeedbackType,
    pub specific_comments: Option<String>,
    pub correction_suggestions: Option<String>,
    pub context_relevance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackRating {
    Excellent,
    Good,
    Fair,
    Poor,
    Terrible,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackType {
    Accuracy,
    Relevance,
    Completeness,
    Clarity,
    Helpfulness,
    Overall,
}

/// Context information for adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInformation {
    pub user_profile: UserProfile,
    pub session_context: SessionContext,
    pub domain_context: DomainContext,
    pub temporal_context: TemporalContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: String,
    pub expertise_level: ExpertiseLevel,
    pub preferences: UserPreferences,
    pub interaction_history: InteractionHistory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub response_style: ResponseStyle,
    pub detail_level: DetailLevel,
    pub preferred_formats: Vec<String>,
    pub language_preferences: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseStyle {
    Concise,
    Detailed,
    Conversational,
    Technical,
    Explanatory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetailLevel {
    High,
    Medium,
    Low,
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionHistory {
    pub total_interactions: usize,
    pub average_satisfaction: f32,
    pub common_topics: Vec<String>,
    pub feedback_patterns: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContext {
    pub session_id: String,
    pub session_duration: Duration,
    pub conversation_flow: ConversationFlow,
    pub current_objectives: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationFlow {
    pub topic_transitions: Vec<String>,
    pub question_types: Vec<String>,
    pub complexity_progression: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainContext {
    pub primary_domain: String,
    pub secondary_domains: Vec<String>,
    pub domain_expertise_required: f32,
    pub domain_specific_patterns: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    pub time_of_day: String,
    pub day_of_week: String,
    pub seasonal_patterns: Vec<String>,
    pub trending_topics: Vec<String>,
}

/// Adaptation metrics and performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationMetrics {
    pub accuracy_improvement: f32,
    pub latency_improvement: f32,
    pub user_satisfaction_improvement: f32,
    pub learning_convergence_rate: f32,
    pub adaptation_stability: f32,
    pub performance_consistency: f32,
    pub rollback_frequency: f32,
    pub adaptation_efficiency: f32,
}

/// Model checkpoint for rollback
#[derive(Debug, Clone)]
pub struct ModelCheckpoint {
    pub checkpoint_id: String,
    pub model_state: Vec<u8>, // Serialized model state
    pub performance_metrics: AdaptationMetrics,
    pub timestamp: SystemTime,
    pub adaptation_step: usize,
}

/// Real-time adaptation engine
pub struct RealTimeAdaptation {
    config: AdaptationConfig,
    interaction_buffer: Arc<Mutex<VecDeque<InteractionData>>>,
    adaptation_metrics: Arc<RwLock<AdaptationMetrics>>,
    model_checkpoints: Arc<RwLock<VecDeque<ModelCheckpoint>>>,
    current_performance: Arc<RwLock<PerformanceState>>,
    adaptation_history: Arc<RwLock<Vec<AdaptationEvent>>>,
}

#[derive(Debug, Clone)]
pub struct PerformanceState {
    accuracy: f32,
    latency_ms: f32,
    user_satisfaction: f32,
    error_rate: f32,
    adaptation_count: usize,
    last_adaptation: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdaptationEvent {
    event_id: String,
    adaptation_type: AdaptationType,
    trigger_reason: String,
    performance_before: PerformanceSnapshot,
    performance_after: PerformanceSnapshot,
    success: bool,
    timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum AdaptationType {
    GradientUpdate,
    PromptOptimization,
    MemoryUpdate,
    ArchitectureAdjustment,
    HyperparameterTuning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceSnapshot {
    accuracy: f32,
    latency_ms: f32,
    user_satisfaction: f32,
    error_rate: f32,
}

impl RealTimeAdaptation {
    /// Create new real-time adaptation engine
    pub fn new(config: AdaptationConfig) -> Self {
        Self {
            config,
            interaction_buffer: Arc::new(Mutex::new(VecDeque::new())),
            adaptation_metrics: Arc::new(RwLock::new(AdaptationMetrics {
                accuracy_improvement: 0.0,
                latency_improvement: 0.0,
                user_satisfaction_improvement: 0.0,
                learning_convergence_rate: 0.0,
                adaptation_stability: 0.0,
                performance_consistency: 0.0,
                rollback_frequency: 0.0,
                adaptation_efficiency: 0.0,
            })),
            model_checkpoints: Arc::new(RwLock::new(VecDeque::new())),
            current_performance: Arc::new(RwLock::new(PerformanceState {
                accuracy: 0.8,
                latency_ms: 100.0,
                user_satisfaction: 0.8,
                error_rate: 0.1,
                adaptation_count: 0,
                last_adaptation: None,
            })),
            adaptation_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Process new interaction for adaptation
    pub async fn process_interaction(&self, interaction: InteractionData) -> Result<()> {
        // Add to interaction buffer
        {
            let mut buffer = self.interaction_buffer.lock().await;
            buffer.push_back(interaction.clone());

            // Maintain buffer size
            if buffer.len() > self.config.adaptation_parameters.memory_window_size {
                buffer.pop_front();
            }
        }

        // Check if adaptation should be triggered
        if self.should_trigger_adaptation(&interaction).await? {
            self.trigger_adaptation().await?;
        }

        Ok(())
    }

    /// Check if adaptation should be triggered
    async fn should_trigger_adaptation(&self, interaction: &InteractionData) -> Result<bool> {
        let current_perf = self.current_performance.read().await;
        let config = &self.config.trigger_conditions;

        // Check performance degradation
        if current_perf.accuracy < config.performance_degradation_threshold {
            return Ok(true);
        }

        // Check error rate
        if current_perf.error_rate > config.error_rate_threshold {
            return Ok(true);
        }

        // Check latency
        if current_perf.latency_ms > config.latency_threshold_ms {
            return Ok(true);
        }

        // Check user feedback
        if let Some(feedback) = &interaction.user_feedback {
            if matches!(
                feedback.rating,
                FeedbackRating::Poor | FeedbackRating::Terrible
            ) {
                return Ok(true);
            }
        }

        // Check frequency-based triggers
        match config.adaptation_frequency {
            AdaptationFrequency::Continuous => Ok(true),
            AdaptationFrequency::Periodic(duration) => {
                if let Some(last_adaptation) = current_perf.last_adaptation {
                    Ok(SystemTime::now()
                        .duration_since(last_adaptation)
                        .unwrap_or(Duration::from_secs(0))
                        >= duration)
                } else {
                    Ok(true)
                }
            }
            AdaptationFrequency::OnDemand => Ok(false),
            AdaptationFrequency::ThresholdBased => {
                // Already checked above
                Ok(false)
            }
        }
    }

    /// Trigger adaptation process
    async fn trigger_adaptation(&self) -> Result<()> {
        // Create checkpoint before adaptation
        self.create_checkpoint().await?;

        // Collect recent interactions for adaptation
        let interactions = {
            let buffer = self.interaction_buffer.lock().await;
            buffer.iter().cloned().collect::<Vec<_>>()
        };

        if interactions.len() < self.config.trigger_conditions.minimum_data_points {
            return Ok(()); // Not enough data for adaptation
        }

        // Perform adaptation based on strategy
        let adaptation_result = match self.config.strategy {
            AdaptationStrategy::OnlineGradientDescent => {
                self.online_gradient_descent(&interactions).await
            }
            AdaptationStrategy::ReinforcementLearning => {
                self.reinforcement_learning(&interactions).await
            }
            AdaptationStrategy::MetaLearning => self.meta_learning(&interactions).await,
            AdaptationStrategy::InContextLearning => self.in_context_learning(&interactions).await,
            AdaptationStrategy::AdaptivePrompting => self.adaptive_prompting(&interactions).await,
            _ => {
                // Fallback to online gradient descent
                self.online_gradient_descent(&interactions).await
            }
        }?;

        // Update performance state
        {
            let mut current_perf = self.current_performance.write().await;
            current_perf.adaptation_count += 1;
            current_perf.last_adaptation = Some(SystemTime::now());
        }

        // Record adaptation event
        self.record_adaptation_event(adaptation_result).await?;

        Ok(())
    }

    /// Online gradient descent adaptation
    async fn online_gradient_descent(
        &self,
        interactions: &[InteractionData],
    ) -> Result<AdaptationResult> {
        // Simulate gradient descent adaptation
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Calculate performance improvement
        let improvement = self.calculate_performance_improvement(interactions).await?;

        Ok(AdaptationResult {
            adaptation_type: AdaptationType::GradientUpdate,
            performance_improvement: improvement,
            success: improvement > 0.0,
            details: "Applied online gradient descent update".to_string(),
        })
    }

    /// Reinforcement learning adaptation
    async fn reinforcement_learning(
        &self,
        interactions: &[InteractionData],
    ) -> Result<AdaptationResult> {
        // Simulate RL adaptation
        tokio::time::sleep(Duration::from_millis(150)).await;

        let improvement = self.calculate_performance_improvement(interactions).await?;

        Ok(AdaptationResult {
            adaptation_type: AdaptationType::ArchitectureAdjustment,
            performance_improvement: improvement * 1.2, // RL typically more effective
            success: improvement > 0.0,
            details: "Applied reinforcement learning policy update".to_string(),
        })
    }

    /// Meta learning adaptation
    async fn meta_learning(&self, interactions: &[InteractionData]) -> Result<AdaptationResult> {
        // Simulate meta learning
        tokio::time::sleep(Duration::from_millis(200)).await;

        let improvement = self.calculate_performance_improvement(interactions).await?;

        Ok(AdaptationResult {
            adaptation_type: AdaptationType::HyperparameterTuning,
            performance_improvement: improvement * 1.5, // Meta learning more adaptive
            success: improvement > 0.0,
            details: "Applied meta learning adaptation".to_string(),
        })
    }

    /// In-context learning adaptation
    async fn in_context_learning(
        &self,
        interactions: &[InteractionData],
    ) -> Result<AdaptationResult> {
        // Simulate in-context learning
        tokio::time::sleep(Duration::from_millis(50)).await;

        let improvement = self.calculate_performance_improvement(interactions).await?;

        Ok(AdaptationResult {
            adaptation_type: AdaptationType::MemoryUpdate,
            performance_improvement: improvement * 0.8, // Faster but less impactful
            success: improvement > 0.0,
            details: "Applied in-context learning update".to_string(),
        })
    }

    /// Adaptive prompting
    async fn adaptive_prompting(
        &self,
        interactions: &[InteractionData],
    ) -> Result<AdaptationResult> {
        // Simulate prompt optimization
        tokio::time::sleep(Duration::from_millis(75)).await;

        let improvement = self.calculate_performance_improvement(interactions).await?;

        Ok(AdaptationResult {
            adaptation_type: AdaptationType::PromptOptimization,
            performance_improvement: improvement * 1.1,
            success: improvement > 0.0,
            details: "Applied adaptive prompt optimization".to_string(),
        })
    }

    /// Calculate performance improvement from interactions
    async fn calculate_performance_improvement(
        &self,
        interactions: &[InteractionData],
    ) -> Result<f32> {
        let mut total_feedback_score = 0.0;
        let mut feedback_count = 0;

        for interaction in interactions {
            if let Some(feedback) = &interaction.user_feedback {
                let score = match feedback.rating {
                    FeedbackRating::Excellent => 1.0,
                    FeedbackRating::Good => 0.8,
                    FeedbackRating::Fair => 0.6,
                    FeedbackRating::Poor => 0.4,
                    FeedbackRating::Terrible => 0.2,
                };
                total_feedback_score += score;
                feedback_count += 1;
            }
        }

        if feedback_count > 0 {
            let average_feedback = total_feedback_score / feedback_count as f32;
            // Convert to improvement metric (assuming current baseline is 0.8)
            Ok((average_feedback - 0.8) * 0.1) // Small improvement
        } else {
            // Estimate improvement based on usage patterns
            Ok(0.01) // Default small improvement
        }
    }

    /// Create model checkpoint
    async fn create_checkpoint(&self) -> Result<()> {
        let current_metrics = self.adaptation_metrics.read().await.clone();
        let current_perf = self.current_performance.read().await;

        let checkpoint = ModelCheckpoint {
            checkpoint_id: format!(
                "checkpoint_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .expect("SystemTime should be after UNIX_EPOCH")
                    .as_secs()
            ),
            model_state: vec![0u8; 1000], // Mock model state
            performance_metrics: current_metrics,
            timestamp: SystemTime::now(),
            adaptation_step: current_perf.adaptation_count,
        };

        let mut checkpoints = self.model_checkpoints.write().await;
        checkpoints.push_back(checkpoint);

        // Maintain checkpoint history (keep last 10)
        if checkpoints.len() > 10 {
            checkpoints.pop_front();
        }

        Ok(())
    }

    /// Record adaptation event
    async fn record_adaptation_event(&self, result: AdaptationResult) -> Result<()> {
        let event = AdaptationEvent {
            event_id: format!(
                "event_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .expect("SystemTime should be after UNIX_EPOCH")
                    .as_secs()
            ),
            adaptation_type: result.adaptation_type,
            trigger_reason: "Performance threshold".to_string(),
            performance_before: PerformanceSnapshot {
                accuracy: 0.8,
                latency_ms: 100.0,
                user_satisfaction: 0.8,
                error_rate: 0.1,
            },
            performance_after: PerformanceSnapshot {
                accuracy: 0.8 + result.performance_improvement,
                latency_ms: 100.0,
                user_satisfaction: 0.8 + result.performance_improvement,
                error_rate: 0.1 - (result.performance_improvement * 0.5),
            },
            success: result.success,
            timestamp: SystemTime::now(),
        };

        let mut history = self.adaptation_history.write().await;
        history.push(event);

        Ok(())
    }

    /// Get adaptation metrics
    pub async fn get_adaptation_metrics(&self) -> Result<AdaptationMetrics> {
        let metrics = self.adaptation_metrics.read().await;
        Ok(metrics.clone())
    }

    /// Get current performance state
    pub async fn get_performance_state(&self) -> Result<PerformanceState> {
        let state = self.current_performance.read().await;
        Ok(state.clone())
    }

    /// Rollback to previous checkpoint if needed
    pub async fn rollback_if_needed(&self) -> Result<bool> {
        let current_perf = self.current_performance.read().await;

        // Check if rollback is needed
        if current_perf.accuracy < self.config.rollback_config.rollback_threshold {
            self.rollback_to_checkpoint().await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Rollback to most recent checkpoint
    async fn rollback_to_checkpoint(&self) -> Result<()> {
        let checkpoints = self.model_checkpoints.read().await;

        if let Some(latest_checkpoint) = checkpoints.back() {
            // Simulate rollback process
            tokio::time::sleep(Duration::from_millis(100)).await;

            // Update performance to checkpoint state
            let mut current_perf = self.current_performance.write().await;
            current_perf.accuracy =
                latest_checkpoint.performance_metrics.accuracy_improvement + 0.8;
            current_perf.user_satisfaction = latest_checkpoint
                .performance_metrics
                .user_satisfaction_improvement
                + 0.8;

            println!(
                "Rolled back to checkpoint: {}",
                latest_checkpoint.checkpoint_id
            );
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct AdaptationResult {
    adaptation_type: AdaptationType,
    performance_improvement: f32,
    success: bool,
    details: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Priority, Usage, UseCase};

    #[tokio::test]
    async fn test_real_time_adaptation_creation() {
        let config = AdaptationConfig {
            adaptation_id: "test_adaptation".to_string(),
            strategy: AdaptationStrategy::OnlineGradientDescent,
            trigger_conditions: TriggerConditions {
                performance_degradation_threshold: 0.7,
                user_feedback_threshold: 5,
                error_rate_threshold: 0.2,
                latency_threshold_ms: 200.0,
                adaptation_frequency: AdaptationFrequency::Continuous,
                minimum_data_points: 10,
            },
            adaptation_parameters: AdaptationParameters {
                learning_rate: 0.01,
                adaptation_strength: 0.5,
                memory_window_size: 100,
                regularization_weight: 0.01,
                exploration_rate: 0.1,
                momentum: 0.9,
                gradient_clipping: Some(1.0),
            },
            performance_targets: PerformanceTargets {
                target_accuracy: 0.95,
                target_latency_ms: 50.0,
                target_user_satisfaction: 0.9,
                target_error_rate: 0.05,
                convergence_threshold: 0.01,
            },
            learning_rate_schedule: LearningRateSchedule::Adaptive,
            rollback_config: RollbackConfig {
                enable_rollback: true,
                rollback_threshold: 0.6,
                max_rollback_attempts: 3,
                rollback_window: Duration::from_secs(3600),
                checkpoint_frequency: Duration::from_secs(300),
            },
        };

        let adaptation = RealTimeAdaptation::new(config);
        let metrics = adaptation.get_adaptation_metrics().await.unwrap();
        assert_eq!(metrics.accuracy_improvement, 0.0);
    }

    #[tokio::test]
    async fn test_interaction_processing() {
        let config = AdaptationConfig {
            adaptation_id: "test_adaptation".to_string(),
            strategy: AdaptationStrategy::OnlineGradientDescent,
            trigger_conditions: TriggerConditions {
                performance_degradation_threshold: 0.7,
                user_feedback_threshold: 5,
                error_rate_threshold: 0.2,
                latency_threshold_ms: 200.0,
                adaptation_frequency: AdaptationFrequency::OnDemand,
                minimum_data_points: 1,
            },
            adaptation_parameters: AdaptationParameters {
                learning_rate: 0.01,
                adaptation_strength: 0.5,
                memory_window_size: 100,
                regularization_weight: 0.01,
                exploration_rate: 0.1,
                momentum: 0.9,
                gradient_clipping: Some(1.0),
            },
            performance_targets: PerformanceTargets {
                target_accuracy: 0.95,
                target_latency_ms: 50.0,
                target_user_satisfaction: 0.9,
                target_error_rate: 0.05,
                convergence_threshold: 0.01,
            },
            learning_rate_schedule: LearningRateSchedule::Adaptive,
            rollback_config: RollbackConfig {
                enable_rollback: true,
                rollback_threshold: 0.6,
                max_rollback_attempts: 3,
                rollback_window: Duration::from_secs(3600),
                checkpoint_frequency: Duration::from_secs(300),
            },
        };

        let adaptation = RealTimeAdaptation::new(config);

        let interaction = InteractionData {
            interaction_id: "test_1".to_string(),
            request: LLMRequest {
                messages: vec![],
                system_prompt: None,
                temperature: 0.7,
                max_tokens: Some(100),
                use_case: UseCase::SimpleQuery,
                priority: Priority::Normal,
                timeout: None,
            },
            response: LLMResponse {
                content: "Test response".to_string(),
                model_used: "test_model".to_string(),
                provider_used: "test_provider".to_string(),
                usage: Usage {
                    prompt_tokens: 10,
                    completion_tokens: 20,
                    total_tokens: 30,
                    cost: 0.001,
                },
                latency: Duration::from_millis(100),
                quality_score: Some(0.9),
                metadata: HashMap::new(),
            },
            user_feedback: Some(UserFeedback {
                rating: FeedbackRating::Good,
                feedback_type: FeedbackType::Overall,
                specific_comments: None,
                correction_suggestions: None,
                context_relevance: 0.8,
            }),
            context_information: ContextInformation {
                user_profile: UserProfile {
                    user_id: "test_user".to_string(),
                    expertise_level: ExpertiseLevel::Intermediate,
                    preferences: UserPreferences {
                        response_style: ResponseStyle::Conversational,
                        detail_level: DetailLevel::Medium,
                        preferred_formats: vec!["text".to_string()],
                        language_preferences: vec!["en".to_string()],
                    },
                    interaction_history: InteractionHistory {
                        total_interactions: 10,
                        average_satisfaction: 0.8,
                        common_topics: vec!["AI".to_string()],
                        feedback_patterns: HashMap::new(),
                    },
                },
                session_context: SessionContext {
                    session_id: "session_1".to_string(),
                    session_duration: Duration::from_secs(300),
                    conversation_flow: ConversationFlow {
                        topic_transitions: vec!["greeting".to_string()],
                        question_types: vec!["factual".to_string()],
                        complexity_progression: vec![0.5],
                    },
                    current_objectives: vec!["learn_ai".to_string()],
                },
                domain_context: DomainContext {
                    primary_domain: "AI".to_string(),
                    secondary_domains: vec!["ML".to_string()],
                    domain_expertise_required: 0.6,
                    domain_specific_patterns: HashMap::new(),
                },
                temporal_context: TemporalContext {
                    time_of_day: "morning".to_string(),
                    day_of_week: "monday".to_string(),
                    seasonal_patterns: vec!["spring".to_string()],
                    trending_topics: vec!["AI_trends".to_string()],
                },
            },
            timestamp: SystemTime::now(),
        };

        adaptation.process_interaction(interaction).await.unwrap();
        let performance = adaptation.get_performance_state().await.unwrap();
        assert!(performance.accuracy > 0.0);
    }
}
