//! Conversation pattern recognition for revolutionary chat optimization

use serde::{Deserialize, Serialize};

/// A detected pattern in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Name of the detected pattern
    pub pattern_name: String,
    /// Confidence in this detection (0.0 to 1.0)
    pub confidence: f64,
    /// Start index of the pattern in the message sequence
    pub start_index: usize,
    /// End index of the pattern in the message sequence
    pub end_index: usize,
    /// Strength of the pattern signal
    pub pattern_strength: f64,
}

/// Types of conversation patterns
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// User is asking clarifying questions
    Clarification,
    /// User is exploring a topic in depth
    DeepDive,
    /// User is trying different approaches
    Exploration,
    /// User is providing feedback
    FeedbackLoop,
    /// User is following a structured workflow
    StructuredWorkflow,
    /// User is expressing frustration
    Frustration,
    /// User has reached a goal state
    GoalAchieved,
    /// Conversation is drifting off-topic
    TopicDrift,
}

impl PatternType {
    /// Get the human-readable name of this pattern type
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Clarification => "Clarification",
            Self::DeepDive => "Deep Dive",
            Self::Exploration => "Exploration",
            Self::FeedbackLoop => "Feedback Loop",
            Self::StructuredWorkflow => "Structured Workflow",
            Self::Frustration => "Frustration",
            Self::GoalAchieved => "Goal Achieved",
            Self::TopicDrift => "Topic Drift",
        }
    }
}

/// Pattern recognizer that identifies conversation patterns
#[derive(Debug)]
pub struct PatternRecognizer {
    min_confidence: f64,
    window_size: usize,
}

impl PatternRecognizer {
    /// Create a new pattern recognizer
    pub fn new(min_confidence: f64, window_size: usize) -> Self {
        Self {
            min_confidence,
            window_size,
        }
    }

    /// Detect patterns in a sequence of message texts
    pub fn detect_patterns(&self, messages: &[String]) -> Vec<DetectedPattern> {
        let mut patterns = Vec::new();

        if messages.len() < 2 {
            return patterns;
        }

        // Simple heuristic pattern detection
        let len = messages.len();
        let window_end = len.min(self.window_size);

        // Check for clarification pattern (multiple questions)
        let question_count = messages[..window_end]
            .iter()
            .filter(|m| m.contains('?'))
            .count();

        if question_count >= 2 {
            let confidence = (question_count as f64 / window_end as f64).min(1.0);
            if confidence >= self.min_confidence {
                patterns.push(DetectedPattern {
                    pattern_name: PatternType::Clarification.as_str().to_string(),
                    confidence,
                    start_index: 0,
                    end_index: window_end.saturating_sub(1),
                    pattern_strength: confidence,
                });
            }
        }

        // Check for exploration pattern (variety of topics)
        let unique_words: std::collections::HashSet<&str> = messages[..window_end]
            .iter()
            .flat_map(|m| m.split_whitespace())
            .collect();

        if unique_words.len() > 50 {
            let confidence = ((unique_words.len() - 50) as f64 / 100.0).min(1.0);
            if confidence >= self.min_confidence {
                patterns.push(DetectedPattern {
                    pattern_name: PatternType::Exploration.as_str().to_string(),
                    confidence,
                    start_index: 0,
                    end_index: window_end.saturating_sub(1),
                    pattern_strength: confidence,
                });
            }
        }

        patterns
    }
}

impl Default for PatternRecognizer {
    fn default() -> Self {
        Self::new(0.5, 20)
    }
}
