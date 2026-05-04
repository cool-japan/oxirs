//! Interactive feedback loop for graph-based RAG retrieval refinement.
//!
//! Users can mark triples as relevant (positive feedback) or irrelevant
//! (negative feedback).  The `FeedbackSession` adjusts per-triple weights
//! multiplicatively so that subsequent retrievals favour positively-rated
//! triples and suppress negatively-rated ones.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Feedback types
// ─────────────────────────────────────────────────────────────────────────────

/// The valence of a feedback signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackKind {
    Positive,
    Negative,
}

/// A single feedback event from the user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEvent {
    /// Fingerprint of the triple: `"{subject}|{predicate}|{object}"`.
    pub triple_key: String,
    /// Whether the user found this triple useful.
    pub kind: FeedbackKind,
    /// Optional textual note from the user.
    pub note: Option<String>,
}

impl FeedbackEvent {
    /// Construct a feedback event for the given triple.
    pub fn new(subject: &str, predicate: &str, object: &str, kind: FeedbackKind) -> Self {
        Self {
            triple_key: triple_key(subject, predicate, object),
            kind,
            note: None,
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

fn triple_key(s: &str, p: &str, o: &str) -> String {
    format!("{}|{}|{}", s, p, o)
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Hyperparameters for weight adjustment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackConfig {
    /// Multiplicative boost applied per positive feedback event.
    pub positive_factor: f64,
    /// Multiplicative penalty applied per negative feedback event.
    pub negative_factor: f64,
    /// Minimum allowed weight (prevents complete suppression).
    pub min_weight: f64,
    /// Maximum allowed weight (prevents runaway amplification).
    pub max_weight: f64,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            positive_factor: 1.5,
            negative_factor: 0.6,
            min_weight: 0.01,
            max_weight: 10.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FeedbackSession
// ─────────────────────────────────────────────────────────────────────────────

/// Maintains per-triple learned weights across a user session.
///
/// Weights start at 1.0 and are adjusted multiplicatively with each
/// feedback signal.  Use [`FeedbackSession::apply_weights`] to rescore a result set
/// before the next retrieval round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSession {
    /// Per-triple multiplicative weight, keyed by triple fingerprint.
    weights: HashMap<String, f64>,
    /// Ordered history of all feedback events.
    history: Vec<FeedbackEvent>,
    /// Configuration hyperparameters.
    pub config: FeedbackConfig,
}

impl FeedbackSession {
    pub fn new() -> Self {
        Self::with_config(FeedbackConfig::default())
    }

    pub fn with_config(config: FeedbackConfig) -> Self {
        Self {
            weights: HashMap::new(),
            history: Vec::new(),
            config,
        }
    }

    /// Record a feedback event and update the triple's weight.
    pub fn record(&mut self, event: FeedbackEvent) {
        let factor = match event.kind {
            FeedbackKind::Positive => self.config.positive_factor,
            FeedbackKind::Negative => self.config.negative_factor,
        };
        let w = self.weights.entry(event.triple_key.clone()).or_insert(1.0);
        *w = (*w * factor).clamp(self.config.min_weight, self.config.max_weight);
        self.history.push(event);
    }

    /// Convenience: record a positive signal for a triple.
    pub fn like(&mut self, subject: &str, predicate: &str, object: &str) {
        self.record(FeedbackEvent::new(
            subject,
            predicate,
            object,
            FeedbackKind::Positive,
        ));
    }

    /// Convenience: record a negative signal for a triple.
    pub fn dislike(&mut self, subject: &str, predicate: &str, object: &str) {
        self.record(FeedbackEvent::new(
            subject,
            predicate,
            object,
            FeedbackKind::Negative,
        ));
    }

    /// Return the learned weight for a triple (1.0 if unseen).
    pub fn weight(&self, subject: &str, predicate: &str, object: &str) -> f64 {
        let key = triple_key(subject, predicate, object);
        *self.weights.get(&key).unwrap_or(&1.0)
    }

    /// Apply learned weights to a set of `(triple_key, score)` pairs.
    ///
    /// The adjusted score is `original_score * weight`.
    pub fn apply_weights(&self, scores: &HashMap<String, f64>) -> HashMap<String, f64> {
        scores
            .iter()
            .map(|(k, &v)| {
                let w = self.weights.get(k).copied().unwrap_or(1.0);
                (k.clone(), v * w)
            })
            .collect()
    }

    /// Return the full event history.
    pub fn history(&self) -> &[FeedbackEvent] {
        &self.history
    }

    /// Count positive feedback events.
    pub fn positive_count(&self) -> usize {
        self.history
            .iter()
            .filter(|e| e.kind == FeedbackKind::Positive)
            .count()
    }

    /// Count negative feedback events.
    pub fn negative_count(&self) -> usize {
        self.history
            .iter()
            .filter(|e| e.kind == FeedbackKind::Negative)
            .count()
    }

    /// Reset all learned weights and clear history.
    pub fn reset(&mut self) {
        self.weights.clear();
        self.history.clear();
    }
}

impl Default for FeedbackSession {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_weight_is_one() {
        let session = FeedbackSession::new();
        assert_eq!(session.weight("A", "p", "B"), 1.0);
    }

    #[test]
    fn test_positive_feedback_increases_weight() {
        let mut session = FeedbackSession::new();
        session.like("A", "p", "B");
        assert!(session.weight("A", "p", "B") > 1.0);
    }

    #[test]
    fn test_negative_feedback_decreases_weight() {
        let mut session = FeedbackSession::new();
        session.dislike("A", "p", "B");
        assert!(session.weight("A", "p", "B") < 1.0);
    }

    #[test]
    fn test_repeated_positive_capped_at_max() {
        let mut session = FeedbackSession::new();
        for _ in 0..100 {
            session.like("A", "p", "B");
        }
        assert!(session.weight("A", "p", "B") <= session.config.max_weight);
    }

    #[test]
    fn test_repeated_negative_capped_at_min() {
        let mut session = FeedbackSession::new();
        for _ in 0..100 {
            session.dislike("A", "p", "B");
        }
        assert!(session.weight("A", "p", "B") >= session.config.min_weight);
    }

    #[test]
    fn test_apply_weights() {
        let mut session = FeedbackSession::new();
        session.like("A", "knows", "B");
        let key = "A|knows|B".to_string();
        let mut scores = HashMap::new();
        scores.insert(key.clone(), 0.5);
        let adjusted = session.apply_weights(&scores);
        assert!(adjusted[&key] > 0.5);
    }

    #[test]
    fn test_event_history() {
        let mut session = FeedbackSession::new();
        session.like("X", "p", "Y");
        session.dislike("X", "q", "Z");
        assert_eq!(session.history().len(), 2);
        assert_eq!(session.positive_count(), 1);
        assert_eq!(session.negative_count(), 1);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut session = FeedbackSession::new();
        session.like("A", "p", "B");
        session.reset();
        assert_eq!(session.weight("A", "p", "B"), 1.0);
        assert_eq!(session.history().len(), 0);
    }

    #[test]
    fn test_feedback_event_with_note() {
        let event =
            FeedbackEvent::new("A", "p", "B", FeedbackKind::Positive).with_note("very relevant");
        assert_eq!(event.note.as_deref(), Some("very relevant"));
    }

    #[test]
    fn test_custom_config() {
        let config = FeedbackConfig {
            positive_factor: 2.0,
            negative_factor: 0.5,
            min_weight: 0.1,
            max_weight: 5.0,
        };
        let mut session = FeedbackSession::with_config(config);
        session.like("A", "p", "B");
        assert_eq!(session.weight("A", "p", "B"), 2.0);
    }
}
