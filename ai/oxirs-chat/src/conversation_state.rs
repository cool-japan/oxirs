//! Conversation state machine for multi-turn chat interactions.
//!
//! Tracks the lifecycle of a conversation through well-defined phases,
//! enforcing valid transitions and recording the full transition history.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Phase and transition types
// ---------------------------------------------------------------------------

/// The current phase of a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConversationPhase {
    /// Initial greeting / session start.
    Greeting,
    /// Gathering additional information from the user.
    Clarification,
    /// Performing computation or retrieval behind the scenes.
    Processing,
    /// Delivering a response to the user.
    Responding,
    /// Waiting for the user to confirm or reject a proposal.
    WaitingForConfirmation,
    /// The conversation has finished successfully.
    Completed,
    /// The conversation ended due to an error.
    Error,
}

/// A recorded phase transition.
#[derive(Debug, Clone)]
pub struct StateTransition {
    /// The phase before the transition.
    pub from: ConversationPhase,
    /// The phase after the transition.
    pub to: ConversationPhase,
    /// A human-readable description of what caused the transition.
    pub trigger: String,
    /// Millisecond timestamp of the transition.
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// ConversationState
// ---------------------------------------------------------------------------

/// State machine for a single conversation session.
pub struct ConversationState {
    id: String,
    phase: ConversationPhase,
    history: Vec<StateTransition>,
    context: HashMap<String, String>,
    created_at: u64,
    turn_count: usize,
}

impl ConversationState {
    /// Create a new conversation in the [`ConversationPhase::Greeting`] phase.
    pub fn new(id: impl Into<String>, now_ms: u64) -> Self {
        Self {
            id: id.into(),
            phase: ConversationPhase::Greeting,
            history: Vec::new(),
            context: HashMap::new(),
            created_at: now_ms,
            turn_count: 0,
        }
    }

    /// Attempt a transition to `to`, recording the transition with `trigger`.
    ///
    /// Returns `true` when the transition is allowed; `false` when `to` is not
    /// a valid successor to the current phase.
    pub fn transition(
        &mut self,
        to: ConversationPhase,
        trigger: impl Into<String>,
        now_ms: u64,
    ) -> bool {
        let allowed = Self::allowed_transitions(&self.phase);
        if !allowed.contains(&to) {
            return false;
        }
        let t = StateTransition {
            from: self.phase.clone(),
            to: to.clone(),
            trigger: trigger.into(),
            timestamp: now_ms,
        };
        self.history.push(t);
        self.phase = to;
        true
    }

    /// The current phase.
    pub fn current_phase(&self) -> &ConversationPhase {
        &self.phase
    }

    /// Store a context key-value pair.
    pub fn set_context(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.context.insert(key.into(), value.into());
    }

    /// Retrieve a context value by key.
    pub fn get_context(&self, key: &str) -> Option<&str> {
        self.context.get(key).map(String::as_str)
    }

    /// Increment the conversation turn counter.
    pub fn increment_turn(&mut self) {
        self.turn_count += 1;
    }

    /// The number of turns recorded so far.
    pub fn turn_count(&self) -> usize {
        self.turn_count
    }

    /// The full history of phase transitions in chronological order.
    pub fn history(&self) -> &[StateTransition] {
        &self.history
    }

    /// Whether the conversation is in a terminal phase (Completed or Error).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.phase,
            ConversationPhase::Completed | ConversationPhase::Error
        )
    }

    /// The set of phases that can be reached from `phase`.
    pub fn allowed_transitions(phase: &ConversationPhase) -> Vec<ConversationPhase> {
        match phase {
            ConversationPhase::Greeting => vec![
                ConversationPhase::Clarification,
                ConversationPhase::Processing,
                ConversationPhase::Error,
            ],
            ConversationPhase::Clarification => vec![
                ConversationPhase::Processing,
                ConversationPhase::Clarification,
                ConversationPhase::Error,
            ],
            ConversationPhase::Processing => vec![
                ConversationPhase::Responding,
                ConversationPhase::WaitingForConfirmation,
                ConversationPhase::Error,
            ],
            ConversationPhase::Responding => vec![
                ConversationPhase::Greeting,
                ConversationPhase::Clarification,
                ConversationPhase::Processing,
                ConversationPhase::Completed,
                ConversationPhase::Error,
            ],
            ConversationPhase::WaitingForConfirmation => vec![
                ConversationPhase::Processing,
                ConversationPhase::Clarification,
                ConversationPhase::Completed,
                ConversationPhase::Error,
            ],
            ConversationPhase::Completed => vec![],
            ConversationPhase::Error => vec![],
        }
    }

    /// Elapsed time since the conversation was created.
    pub fn duration_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.created_at)
    }

    /// The conversation's unique identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Millisecond timestamp at which the conversation was created.
    pub fn created_at(&self) -> u64 {
        self.created_at
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn new_state(id: &str) -> ConversationState {
        ConversationState::new(id, 0)
    }

    // --- creation ---

    #[test]
    fn test_new_starts_in_greeting() {
        let s = new_state("sess1");
        assert_eq!(s.current_phase(), &ConversationPhase::Greeting);
    }

    #[test]
    fn test_new_id_preserved() {
        let s = new_state("my-conversation-id");
        assert_eq!(s.id(), "my-conversation-id");
    }

    #[test]
    fn test_new_created_at_stored() {
        let s = ConversationState::new("s", 42_000);
        assert_eq!(s.created_at(), 42_000);
    }

    #[test]
    fn test_new_empty_history() {
        let s = new_state("s");
        assert!(s.history().is_empty());
    }

    #[test]
    fn test_new_turn_count_zero() {
        let s = new_state("s");
        assert_eq!(s.turn_count(), 0);
    }

    // --- valid transitions ---

    #[test]
    fn test_greeting_to_processing_valid() {
        let mut s = new_state("s");
        assert!(s.transition(ConversationPhase::Processing, "user sent query", 100));
        assert_eq!(s.current_phase(), &ConversationPhase::Processing);
    }

    #[test]
    fn test_greeting_to_clarification_valid() {
        let mut s = new_state("s");
        assert!(s.transition(ConversationPhase::Clarification, "needs more info", 100));
        assert_eq!(s.current_phase(), &ConversationPhase::Clarification);
    }

    #[test]
    fn test_processing_to_responding_valid() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "t", 100);
        assert!(s.transition(ConversationPhase::Responding, "done", 200));
        assert_eq!(s.current_phase(), &ConversationPhase::Responding);
    }

    #[test]
    fn test_responding_to_completed_valid() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "t", 100);
        s.transition(ConversationPhase::Responding, "t", 200);
        assert!(s.transition(ConversationPhase::Completed, "done", 300));
        assert_eq!(s.current_phase(), &ConversationPhase::Completed);
    }

    #[test]
    fn test_clarification_to_processing_valid() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Clarification, "t", 100);
        assert!(s.transition(ConversationPhase::Processing, "got info", 200));
    }

    #[test]
    fn test_clarification_to_clarification_valid() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Clarification, "t", 100);
        assert!(s.transition(ConversationPhase::Clarification, "still need info", 200));
    }

    #[test]
    fn test_any_phase_to_error_valid() {
        let phases = [
            ConversationPhase::Greeting,
            ConversationPhase::Clarification,
            ConversationPhase::Processing,
            ConversationPhase::Responding,
            ConversationPhase::WaitingForConfirmation,
        ];
        for phase in &phases {
            let allowed = ConversationState::allowed_transitions(phase);
            assert!(
                allowed.contains(&ConversationPhase::Error),
                "Error should be reachable from {phase:?}"
            );
        }
    }

    // --- invalid transitions ---

    #[test]
    fn test_greeting_to_completed_invalid() {
        let mut s = new_state("s");
        assert!(!s.transition(ConversationPhase::Completed, "skip", 100));
        assert_eq!(s.current_phase(), &ConversationPhase::Greeting);
    }

    #[test]
    fn test_completed_to_any_invalid() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "t", 10);
        s.transition(ConversationPhase::Responding, "t", 20);
        s.transition(ConversationPhase::Completed, "t", 30);
        assert!(!s.transition(ConversationPhase::Greeting, "restart", 40));
        assert_eq!(s.current_phase(), &ConversationPhase::Completed);
    }

    #[test]
    fn test_error_to_any_invalid() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Error, "crash", 100);
        assert!(!s.transition(ConversationPhase::Greeting, "retry", 200));
    }

    #[test]
    fn test_invalid_transition_does_not_record_history() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Completed, "skip", 100);
        assert!(s.history().is_empty());
    }

    // --- history ---

    #[test]
    fn test_history_records_transition() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "user asked", 500);
        assert_eq!(s.history().len(), 1);
        let t = &s.history()[0];
        assert_eq!(t.from, ConversationPhase::Greeting);
        assert_eq!(t.to, ConversationPhase::Processing);
        assert_eq!(t.trigger, "user asked");
        assert_eq!(t.timestamp, 500);
    }

    #[test]
    fn test_history_grows_with_transitions() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "t1", 100);
        s.transition(ConversationPhase::Responding, "t2", 200);
        s.transition(ConversationPhase::Completed, "t3", 300);
        assert_eq!(s.history().len(), 3);
    }

    #[test]
    fn test_history_chronological() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "a", 100);
        s.transition(ConversationPhase::Responding, "b", 200);
        assert!(s.history()[0].timestamp <= s.history()[1].timestamp);
    }

    // --- context ---

    #[test]
    fn test_set_and_get_context() {
        let mut s = new_state("s");
        s.set_context("user_name", "Alice");
        assert_eq!(s.get_context("user_name"), Some("Alice"));
    }

    #[test]
    fn test_get_context_missing_key_returns_none() {
        let s = new_state("s");
        assert!(s.get_context("nonexistent").is_none());
    }

    #[test]
    fn test_set_context_overwrite() {
        let mut s = new_state("s");
        s.set_context("k", "v1");
        s.set_context("k", "v2");
        assert_eq!(s.get_context("k"), Some("v2"));
    }

    #[test]
    fn test_context_multiple_keys() {
        let mut s = new_state("s");
        s.set_context("a", "1");
        s.set_context("b", "2");
        s.set_context("c", "3");
        assert_eq!(s.get_context("a"), Some("1"));
        assert_eq!(s.get_context("b"), Some("2"));
        assert_eq!(s.get_context("c"), Some("3"));
    }

    // --- turn_count ---

    #[test]
    fn test_increment_turn() {
        let mut s = new_state("s");
        s.increment_turn();
        assert_eq!(s.turn_count(), 1);
        s.increment_turn();
        assert_eq!(s.turn_count(), 2);
    }

    // --- is_terminal ---

    #[test]
    fn test_is_terminal_completed() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "t", 10);
        s.transition(ConversationPhase::Responding, "t", 20);
        s.transition(ConversationPhase::Completed, "t", 30);
        assert!(s.is_terminal());
    }

    #[test]
    fn test_is_terminal_error() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Error, "crash", 10);
        assert!(s.is_terminal());
    }

    #[test]
    fn test_is_not_terminal_processing() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "t", 10);
        assert!(!s.is_terminal());
    }

    #[test]
    fn test_is_not_terminal_greeting() {
        let s = new_state("s");
        assert!(!s.is_terminal());
    }

    // --- allowed_transitions ---

    #[test]
    fn test_allowed_from_completed_is_empty() {
        let allowed = ConversationState::allowed_transitions(&ConversationPhase::Completed);
        assert!(allowed.is_empty());
    }

    #[test]
    fn test_allowed_from_error_is_empty() {
        let allowed = ConversationState::allowed_transitions(&ConversationPhase::Error);
        assert!(allowed.is_empty());
    }

    #[test]
    fn test_allowed_from_greeting_non_empty() {
        let allowed = ConversationState::allowed_transitions(&ConversationPhase::Greeting);
        assert!(!allowed.is_empty());
    }

    #[test]
    fn test_allowed_from_waiting_for_confirmation() {
        let allowed =
            ConversationState::allowed_transitions(&ConversationPhase::WaitingForConfirmation);
        assert!(allowed.contains(&ConversationPhase::Processing));
        assert!(allowed.contains(&ConversationPhase::Completed));
        assert!(allowed.contains(&ConversationPhase::Error));
    }

    // --- duration_ms ---

    #[test]
    fn test_duration_ms_zero_when_same_time() {
        let s = ConversationState::new("s", 5000);
        assert_eq!(s.duration_ms(5000), 0);
    }

    #[test]
    fn test_duration_ms_elapsed() {
        let s = ConversationState::new("s", 1000);
        assert_eq!(s.duration_ms(4000), 3000);
    }

    #[test]
    fn test_duration_ms_saturating_subtraction() {
        let s = ConversationState::new("s", 5000);
        // now_ms before created_at — should not panic
        assert_eq!(s.duration_ms(1000), 0);
    }

    // --- combined scenarios ---

    #[test]
    fn test_full_happy_path() {
        let mut s = ConversationState::new("conv1", 0);
        assert!(s.transition(ConversationPhase::Processing, "query received", 100));
        s.increment_turn();
        s.set_context("query", "What is Jena?");
        assert!(s.transition(ConversationPhase::Responding, "answer ready", 200));
        s.increment_turn();
        assert!(s.transition(ConversationPhase::Completed, "user satisfied", 300));

        assert!(s.is_terminal());
        assert_eq!(s.turn_count(), 2);
        assert_eq!(s.get_context("query"), Some("What is Jena?"));
        assert_eq!(s.history().len(), 3);
    }

    #[test]
    fn test_error_path_from_processing() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "start", 100);
        s.transition(ConversationPhase::Error, "timeout", 500);
        assert!(s.is_terminal());
        assert!(!s.transition(ConversationPhase::Greeting, "retry", 600));
    }

    #[test]
    fn test_greeting_to_error_valid() {
        let mut s = new_state("s");
        assert!(s.transition(ConversationPhase::Error, "auth failed", 50));
        assert!(s.is_terminal());
    }

    #[test]
    fn test_responding_back_to_greeting() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "t", 1);
        s.transition(ConversationPhase::Responding, "t", 2);
        assert!(s.transition(ConversationPhase::Greeting, "new topic", 3));
        assert_eq!(s.current_phase(), &ConversationPhase::Greeting);
    }

    #[test]
    fn test_processing_to_waiting_for_confirmation() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "t", 1);
        assert!(s.transition(
            ConversationPhase::WaitingForConfirmation,
            "confirm action?",
            2
        ));
        assert_eq!(
            s.current_phase(),
            &ConversationPhase::WaitingForConfirmation
        );
    }

    #[test]
    fn test_phase_equality() {
        assert_eq!(ConversationPhase::Greeting, ConversationPhase::Greeting);
        assert_ne!(ConversationPhase::Greeting, ConversationPhase::Error);
    }

    // --- additional coverage ---

    #[test]
    fn test_waiting_for_confirmation_to_completed_valid() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "t", 1);
        s.transition(ConversationPhase::WaitingForConfirmation, "confirm?", 2);
        assert!(s.transition(ConversationPhase::Completed, "confirmed", 3));
        assert!(s.is_terminal());
    }

    #[test]
    fn test_waiting_for_confirmation_to_clarification_valid() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "t", 1);
        s.transition(ConversationPhase::WaitingForConfirmation, "confirm?", 2);
        assert!(s.transition(ConversationPhase::Clarification, "unclear", 3));
        assert_eq!(s.current_phase(), &ConversationPhase::Clarification);
    }

    #[test]
    fn test_transition_trigger_string_stored() {
        let mut s = new_state("s");
        let trigger = "user sent hello";
        s.transition(ConversationPhase::Processing, trigger, 100);
        assert_eq!(s.history()[0].trigger, trigger);
    }

    #[test]
    fn test_processing_to_clarification_invalid() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "t", 1);
        // Processing → Clarification is NOT in allowed list
        assert!(!s.transition(ConversationPhase::Clarification, "ask more", 2));
    }

    #[test]
    fn test_responding_to_processing_valid() {
        let mut s = new_state("s");
        s.transition(ConversationPhase::Processing, "t", 1);
        s.transition(ConversationPhase::Responding, "t", 2);
        assert!(s.transition(ConversationPhase::Processing, "follow-up", 3));
        assert_eq!(s.current_phase(), &ConversationPhase::Processing);
    }
}
