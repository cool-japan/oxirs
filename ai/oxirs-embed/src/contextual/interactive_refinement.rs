//! Interactive refinement for contextual embeddings

use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Interactive context from user feedback
#[derive(Debug, Clone)]
pub struct InteractiveContext {
    pub session_id: String,
    pub feedback_history: Vec<UserFeedback>,
    pub adaptation_score: f32,
    pub confidence_level: f32,
}

/// User feedback for interactive refinement
#[derive(Debug, Clone)]
pub struct UserFeedback {
    pub feedback_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub feedback_type: FeedbackType,
    pub score: f32,
    pub text_feedback: Option<String>,
}

/// Types of user feedback
#[derive(Debug, Clone)]
pub enum FeedbackType {
    Relevance,
    Quality,
    Preference,
    Correction,
    Satisfaction,
}