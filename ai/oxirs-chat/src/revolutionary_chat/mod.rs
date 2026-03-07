//! Revolutionary Chat Optimization System
//!
//! Provides advanced optimization, coordination, and analysis capabilities
//! for high-performance conversational AI systems.
//!
//! Note: Full implementation pending scirs2-core API stabilization.
//! Currently exposes configuration types and stub implementations.

pub mod config;
pub mod coordinator;
pub mod emotional;
pub mod intent;
pub mod optimizer;
pub mod patterns;
pub mod performance;
pub mod statistics;

pub use config::{
    AdvancedStatisticsConfig, ChatPerformanceTargets, ConversationAnalysisConfig,
    CoordinationStrategy, RevolutionaryChatConfig, UnifiedOptimizationConfig,
};

pub use coordinator::{CoordinationEvent, UnifiedChatCoordinator, UserBehaviorSummary};
pub use emotional::EmotionalState;
pub use intent::{IntentPredictor, IntentType, PredictedIntent};
pub use optimizer::RevolutionaryChatOptimizer;
pub use patterns::DetectedPattern;
pub use performance::PerformanceCorrelation;
pub use statistics::AdvancedChatStatisticsCollector;
