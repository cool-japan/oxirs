//! Emotional state tracking for revolutionary chat optimization

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Tracks the emotional state of a conversation participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    /// Positive/negative emotion valence (-1.0 to 1.0)
    pub valence: f64,
    /// Energy level / arousal (0.0 to 1.0)
    pub arousal: f64,
    /// Level of curiosity (0.0 to 1.0)
    pub curiosity: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// When this state was measured
    #[serde(skip)]
    pub timestamp: Option<SystemTime>,
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            curiosity: 0.5,
            confidence: 0.5,
            timestamp: Some(SystemTime::now()),
        }
    }
}

impl EmotionalState {
    /// Create a new neutral emotional state
    pub fn neutral() -> Self {
        Self::default()
    }

    /// Create a positive emotional state
    pub fn positive() -> Self {
        Self {
            valence: 0.7,
            arousal: 0.6,
            curiosity: 0.7,
            confidence: 0.8,
            timestamp: Some(SystemTime::now()),
        }
    }

    /// Blend this state with another (weighted average)
    pub fn blend(&self, other: &EmotionalState, weight: f64) -> EmotionalState {
        let w = weight.clamp(0.0, 1.0);
        let w_inv = 1.0 - w;
        EmotionalState {
            valence: self.valence * w_inv + other.valence * w,
            arousal: self.arousal * w_inv + other.arousal * w,
            curiosity: self.curiosity * w_inv + other.curiosity * w,
            confidence: self.confidence * w_inv + other.confidence * w,
            timestamp: Some(SystemTime::now()),
        }
    }

    /// Returns true if the emotional state is positive overall
    pub fn is_positive(&self) -> bool {
        self.valence > 0.2
    }

    /// Returns true if engagement (curiosity + arousal) is high
    pub fn is_engaged(&self) -> bool {
        (self.curiosity + self.arousal) / 2.0 > 0.6
    }
}

/// A history entry of emotional states over time
#[derive(Debug, Clone)]
pub struct EmotionalHistory {
    states: Vec<EmotionalState>,
    max_size: usize,
}

impl EmotionalHistory {
    /// Create a new emotional history with maximum capacity
    pub fn new(max_size: usize) -> Self {
        Self {
            states: Vec::new(),
            max_size,
        }
    }

    /// Record a new emotional state
    pub fn record(&mut self, state: EmotionalState) {
        self.states.push(state);
        if self.states.len() > self.max_size {
            self.states.remove(0);
        }
    }

    /// Get the trend in valence (positive = improving)
    pub fn valence_trend(&self) -> f64 {
        if self.states.len() < 2 {
            return 0.0;
        }
        let recent_n = self.states.len().min(5);
        let recent_avg: f64 = self.states[self.states.len() - recent_n..]
            .iter()
            .map(|s| s.valence)
            .sum::<f64>()
            / recent_n as f64;
        let earlier_avg: f64 = self.states[..self.states.len() - recent_n]
            .iter()
            .map(|s| s.valence)
            .sum::<f64>()
            / (self.states.len() - recent_n) as f64;
        recent_avg - earlier_avg
    }

    /// Get the most recent emotional state
    pub fn current(&self) -> Option<&EmotionalState> {
        self.states.last()
    }
}
