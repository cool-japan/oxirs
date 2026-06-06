//! Interactive feedback loop for graph-based RAG retrieval refinement.
//!
//! Users can mark triples as relevant (positive feedback) or irrelevant
//! (negative feedback).  The `FeedbackSession` adjusts per-triple weights
//! multiplicatively so that subsequent retrievals favour positively-rated
//! triples and suppress negatively-rated ones.
//!
//! # v0.4.0 additions
//!
//! [`TripleRelevanceFeedback`] provides a seahash-keyed, multiplicative-weight
//! session with a typed [`Relevance`] enum and a sorted `apply_to_scores`
//! method, suitable for single-session adaptive retrieval.

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

// ─────────────────────────────────────────────────────────────────────────────
// v0.4.0: TripleRelevanceFeedback (seahash-keyed adaptive weights)
// ─────────────────────────────────────────────────────────────────────────────

/// Relevance signal for a triple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Relevance {
    /// User found this triple useful.
    Positive,
    /// User found this triple not useful.
    Negative,
    /// User has no preference (resets weight to 1.0).
    Neutral,
}

/// Unique identifier for a triple — seahash of `"{s}|{p}|{o}"`.
pub type TripleId = u64;

/// Compute a [`TripleId`] for the given triple components.
fn triple_id(s: &str, p: &str, o: &str) -> TripleId {
    seahash::hash(format!("{s}|{p}|{o}").as_bytes())
}

/// Session-scoped adaptive feedback for triple retrieval.
///
/// Weights start at 1.0 and are updated multiplicatively on each
/// [`Relevance::Positive`] or [`Relevance::Negative`] signal.
/// [`Relevance::Neutral`] resets the weight to exactly 1.0.
///
/// Use [`TripleRelevanceFeedback::apply_to_scores`] to re-rank a scored
/// result list before serving the next retrieval round.
pub struct TripleRelevanceFeedback {
    positive: std::collections::HashSet<TripleId>,
    negative: std::collections::HashSet<TripleId>,
    weights: std::collections::HashMap<TripleId, f64>,
}

impl TripleRelevanceFeedback {
    /// Positive weight multiplier per feedback event.
    const POSITIVE_FACTOR: f64 = 1.5;
    /// Negative weight multiplier per feedback event.
    const NEGATIVE_FACTOR: f64 = 0.5;
    /// Minimum weight (prevents complete suppression).
    const MIN_WEIGHT: f64 = 0.1;
    /// Maximum weight (prevents runaway amplification).
    const MAX_WEIGHT: f64 = 2.0;

    pub fn new() -> Self {
        Self {
            positive: std::collections::HashSet::new(),
            negative: std::collections::HashSet::new(),
            weights: std::collections::HashMap::new(),
        }
    }

    /// Record a relevance signal for the triple `(subject, predicate, object)`.
    ///
    /// - `Positive`: weight *= 1.5, capped at 2.0
    /// - `Negative`: weight *= 0.5, floored at 0.1
    /// - `Neutral`:  weight reset to 1.0
    pub fn record_feedback(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        signal: Relevance,
    ) {
        let id = triple_id(subject, predicate, object);
        match signal {
            Relevance::Positive => {
                let current = self.weights.get(&id).copied().unwrap_or(1.0);
                let next = (current * Self::POSITIVE_FACTOR).min(Self::MAX_WEIGHT);
                self.weights.insert(id, next);
                self.positive.insert(id);
                self.negative.remove(&id);
            }
            Relevance::Negative => {
                let current = self.weights.get(&id).copied().unwrap_or(1.0);
                let next = (current * Self::NEGATIVE_FACTOR).max(Self::MIN_WEIGHT);
                self.weights.insert(id, next);
                self.negative.insert(id);
                self.positive.remove(&id);
            }
            Relevance::Neutral => {
                self.weights.insert(id, 1.0);
                self.positive.remove(&id);
                self.negative.remove(&id);
            }
        }
    }

    /// Get the multiplicative weight for a triple (default 1.0 if no feedback).
    pub fn weight_of(&self, subject: &str, predicate: &str, object: &str) -> f64 {
        let id = triple_id(subject, predicate, object);
        self.weights.get(&id).copied().unwrap_or(1.0)
    }

    /// Apply weights to a scored list of triples and return re-weighted scores
    /// sorted descending.
    ///
    /// `scores`: `Vec<((subject, predicate, object), raw_score)>`
    pub fn apply_to_scores(
        &self,
        scores: Vec<((String, String, String), f64)>,
    ) -> Vec<((String, String, String), f64)> {
        let mut weighted: Vec<((String, String, String), f64)> = scores
            .into_iter()
            .map(|((s, p, o), raw)| {
                let w = self.weight_of(&s, &p, &o);
                ((s, p, o), raw * w)
            })
            .collect();
        weighted.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        weighted
    }

    /// Clear all feedback and weights for this session.
    pub fn reset(&mut self) {
        self.positive.clear();
        self.negative.clear();
        self.weights.clear();
    }
}

impl Default for TripleRelevanceFeedback {
    fn default() -> Self {
        Self::new()
    }
}

// ─── TripleRelevanceFeedback tests ────────────────────────────────────────────

#[cfg(test)]
mod triple_relevance_tests {
    use super::{Relevance, TripleRelevanceFeedback};

    #[test]
    fn test_positive_feedback_boosts_score() {
        let mut session = TripleRelevanceFeedback::new();
        session.record_feedback("A", "p", "B", Relevance::Positive);
        let raw = 0.5_f64;
        let boosted = session.weight_of("A", "p", "B") * raw;
        assert!(
            boosted > raw,
            "positive feedback should boost score: {boosted} vs {raw}"
        );
    }

    #[test]
    fn test_negative_feedback_reduces_score() {
        let mut session = TripleRelevanceFeedback::new();
        session.record_feedback("A", "p", "B", Relevance::Negative);
        let raw = 0.5_f64;
        let reduced = session.weight_of("A", "p", "B") * raw;
        assert!(
            reduced < raw,
            "negative feedback should reduce score: {reduced} vs {raw}"
        );
    }

    #[test]
    fn test_neutral_no_feedback_leaves_score_unchanged() {
        let session = TripleRelevanceFeedback::new();
        // No feedback recorded → weight is 1.0.
        let raw = 0.42_f64;
        let result = session.weight_of("X", "q", "Y") * raw;
        assert!(
            (result - raw).abs() < 1e-12,
            "neutral (no feedback) should leave score unchanged: {result} vs {raw}"
        );
    }

    #[test]
    fn test_repeated_positive_capped_at_max() {
        let mut session = TripleRelevanceFeedback::new();
        for _ in 0..100 {
            session.record_feedback("A", "p", "B", Relevance::Positive);
        }
        let w = session.weight_of("A", "p", "B");
        assert!(
            w <= 2.0,
            "repeated positive feedback must not exceed 2.0, got {w}"
        );
    }

    #[test]
    fn test_repeated_negative_floored_at_min() {
        let mut session = TripleRelevanceFeedback::new();
        for _ in 0..100 {
            session.record_feedback("A", "p", "B", Relevance::Negative);
        }
        let w = session.weight_of("A", "p", "B");
        assert!(
            w >= 0.1,
            "repeated negative feedback must not go below 0.1, got {w}"
        );
    }

    #[test]
    fn test_reset_clears_all_weights() {
        let mut session = TripleRelevanceFeedback::new();
        session.record_feedback("A", "p", "B", Relevance::Positive);
        session.reset();
        let w = session.weight_of("A", "p", "B");
        assert!(
            (w - 1.0).abs() < 1e-12,
            "after reset, weight should be 1.0, got {w}"
        );
    }

    #[test]
    fn test_apply_to_scores_sorted_descending() {
        let mut session = TripleRelevanceFeedback::new();
        session.record_feedback("A", "p", "B", Relevance::Positive);
        // "A|p|B" gets weight 1.5, "X|q|Y" stays 1.0.
        let scores = vec![
            (("X".into(), "q".into(), "Y".into()), 0.8_f64),
            (("A".into(), "p".into(), "B".into()), 0.5_f64),
        ];
        let result = session.apply_to_scores(scores);
        // A|p|B: 0.5 * 1.5 = 0.75; X|q|Y: 0.8 * 1.0 = 0.8 → X first.
        assert_eq!(result.len(), 2, "should return same number of triples");
        let (_, first_score) = &result[0];
        let (_, second_score) = &result[1];
        assert!(
            first_score >= second_score,
            "results should be sorted descending: {first_score} >= {second_score}"
        );
    }
}
