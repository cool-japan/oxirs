//! Consciousness Model Configuration
//!
//! Provides configuration structures for consciousness processing settings.

/// Configuration for consciousness processing features
#[derive(Debug, Clone)]
pub struct ConsciousnessConfig {
    pub enabled: bool,
    pub memory_retention_hours: u64,
    pub emotional_adaptation: bool,
    pub insight_generation: bool,
}

impl Default for ConsciousnessConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            memory_retention_hours: 24,
            emotional_adaptation: true,
            insight_generation: true,
        }
    }
}

/// Configuration for the consciousness model components
#[derive(Debug, Clone)]
pub struct ConsciousnessModelConfig {
    pub neural_simulation_enabled: bool,
    pub memory_layers: usize,
    pub attention_points: usize,
    pub emotional_regulation: bool,
    pub consciousness_stream_length: usize,
}

impl Default for ConsciousnessModelConfig {
    fn default() -> Self {
        Self {
            neural_simulation_enabled: true,
            memory_layers: 3,
            attention_points: 20,
            emotional_regulation: true,
            consciousness_stream_length: 100,
        }
    }
}
