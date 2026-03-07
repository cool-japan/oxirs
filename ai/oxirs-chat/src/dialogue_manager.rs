//! Dialogue state machine for multi-turn conversations.
//!
//! Implements a keyword-based intent detector and a finite-state machine that
//! transitions between greeting, information-gathering, processing, responding,
//! clarifying, and closing states.

/// States the dialogue can be in at any point.
#[derive(Debug, Clone, PartialEq)]
pub enum DialogueState {
    /// Initial greeting state
    Greeting,
    /// Actively gathering information on a topic
    Gathering {
        topic: String,
        questions_asked: usize,
    },
    /// Processing the gathered information
    Processing,
    /// Presenting a response to the user
    Responding,
    /// Asking the user to clarify something
    Clarifying(String),
    /// Conversation is winding down
    Closing,
    /// An error has occurred
    Error(String),
}

/// Intent detected from user input.
#[derive(Debug, Clone, PartialEq)]
pub enum UserIntent {
    Greet,
    Query(String),
    Clarify(String),
    Confirm,
    Deny,
    Goodbye,
    Unknown,
}

/// A single turn of dialogue with before/after states and the response given.
#[derive(Debug, Clone)]
pub struct DialogueTurn {
    pub user_input: String,
    pub intent: UserIntent,
    pub state_before: DialogueState,
    pub state_after: DialogueState,
    pub response: String,
}

/// A record of a state transition triggered by a user intent.
#[derive(Debug, Clone)]
pub struct StateTransition {
    pub from: DialogueState,
    pub to: DialogueState,
    pub trigger: UserIntent,
}

/// Multi-turn dialogue manager with a finite-state machine.
pub struct DialogueManager {
    state: DialogueState,
    history: Vec<DialogueTurn>,
}

impl DialogueManager {
    /// Create a new dialogue manager starting in the Greeting state.
    pub fn new() -> Self {
        Self {
            state: DialogueState::Greeting,
            history: Vec::new(),
        }
    }

    /// Process a single user input turn:
    /// 1. Detect intent
    /// 2. Transition state
    /// 3. Generate response
    /// 4. Record the turn in history
    pub fn process_turn(&mut self, input: &str) -> DialogueTurn {
        let intent = Self::detect_intent(input);
        let state_before = self.state.clone();

        let (new_state, response) = self.transition(&state_before, &intent, input);
        self.state = new_state.clone();

        let turn = DialogueTurn {
            user_input: input.to_string(),
            intent,
            state_before,
            state_after: new_state,
            response,
        };
        self.history.push(turn.clone());
        turn
    }

    /// Current dialogue state.
    pub fn current_state(&self) -> &DialogueState {
        &self.state
    }

    /// Full turn history.
    pub fn history(&self) -> &[DialogueTurn] {
        &self.history
    }

    /// Number of turns processed so far.
    pub fn turn_count(&self) -> usize {
        self.history.len()
    }

    /// Reset the dialogue back to the initial Greeting state.
    pub fn reset(&mut self) {
        self.state = DialogueState::Greeting;
        self.history.clear();
    }

    /// Keyword-based intent detection.
    ///
    /// Rules (first match wins):
    /// - Whole-word "hello" or "hi"          → Greet
    /// - Whole-word "bye" or "goodbye"       → Goodbye
    /// - Contains "what do you mean"         → Clarify
    /// - Contains "?"                        → Query
    /// - Whole-word "yes" or "ok"            → Confirm
    /// - Whole-word "no"                     → Deny
    /// - Otherwise                           → Unknown
    ///
    /// Whole-word matching prevents false positives such as detecting "hi" inside
    /// "something" or "no" inside "know".
    pub fn detect_intent(input: &str) -> UserIntent {
        let lower = input.to_lowercase();

        if Self::contains_word(&lower, "hello") || Self::contains_word(&lower, "hi") {
            return UserIntent::Greet;
        }
        if Self::contains_word(&lower, "bye") || Self::contains_word(&lower, "goodbye") {
            return UserIntent::Goodbye;
        }
        if lower.contains("what do you mean") {
            return UserIntent::Clarify(input.to_string());
        }
        if lower.contains('?') {
            return UserIntent::Query(input.to_string());
        }
        if Self::contains_word(&lower, "yes") || Self::contains_word(&lower, "ok") {
            return UserIntent::Confirm;
        }
        if Self::contains_word(&lower, "no") {
            return UserIntent::Deny;
        }
        UserIntent::Unknown
    }

    /// Return true if `text` contains `word` as a whole word (surrounded by
    /// non-alphanumeric characters or at the start/end of the string).
    fn contains_word(text: &str, word: &str) -> bool {
        let mut start = 0;
        while let Some(pos) = text[start..].find(word) {
            let abs_pos = start + pos;
            let before_ok = abs_pos == 0
                || !text
                    .as_bytes()
                    .get(abs_pos - 1)
                    .copied()
                    .unwrap_or(b' ')
                    .is_ascii_alphanumeric();
            let after_pos = abs_pos + word.len();
            let after_ok = after_pos >= text.len()
                || !text
                    .as_bytes()
                    .get(after_pos)
                    .copied()
                    .unwrap_or(b' ')
                    .is_ascii_alphanumeric();
            if before_ok && after_ok {
                return true;
            }
            start = abs_pos + 1;
        }
        false
    }

    // --- private state machine ---

    /// Compute the next state and response given the current state and detected intent.
    fn transition(
        &self,
        current: &DialogueState,
        intent: &UserIntent,
        input: &str,
    ) -> (DialogueState, String) {
        // Goodbye is handled from any state
        if *intent == UserIntent::Goodbye {
            return (
                DialogueState::Closing,
                "Goodbye! Have a great day.".to_string(),
            );
        }

        match current {
            DialogueState::Greeting => match intent {
                UserIntent::Greet => (
                    DialogueState::Greeting,
                    "Hello! How can I help you today?".to_string(),
                ),
                UserIntent::Query(q) => (
                    DialogueState::Gathering {
                        topic: Self::extract_topic(q),
                        questions_asked: 1,
                    },
                    "I'll look into that. Can you tell me more?".to_string(),
                ),
                UserIntent::Clarify(c) => (
                    DialogueState::Clarifying(c.clone()),
                    "Let me clarify that for you.".to_string(),
                ),
                _ => (
                    DialogueState::Greeting,
                    "I'm here to help. What would you like to know?".to_string(),
                ),
            },

            DialogueState::Gathering {
                topic,
                questions_asked,
            } => match intent {
                UserIntent::Query(q) => {
                    let new_topic = Self::extract_topic(q);
                    (
                        DialogueState::Processing,
                        format!("Processing your query about '{new_topic}'..."),
                    )
                }
                UserIntent::Clarify(c) => (
                    DialogueState::Clarifying(c.clone()),
                    "Let me clarify that point.".to_string(),
                ),
                UserIntent::Confirm => (
                    DialogueState::Processing,
                    format!("Great, processing your request about '{topic}'..."),
                ),
                _ => (
                    DialogueState::Gathering {
                        topic: topic.clone(),
                        questions_asked: questions_asked + 1,
                    },
                    format!(
                        "Could you provide more detail about '{topic}'? (Question {})",
                        questions_asked + 1
                    ),
                ),
            },

            DialogueState::Processing => {
                // Automatically transition to Responding
                (
                    DialogueState::Responding,
                    format!("Here is what I found regarding your request: '{input}'."),
                )
            }

            DialogueState::Responding => match intent {
                UserIntent::Query(q) => {
                    let topic = Self::extract_topic(q);
                    (
                        DialogueState::Gathering {
                            topic: topic.clone(),
                            questions_asked: 1,
                        },
                        format!("Sure, let me gather information about '{topic}'."),
                    )
                }
                UserIntent::Clarify(c) => (
                    DialogueState::Clarifying(c.clone()),
                    "I'll clarify that point further.".to_string(),
                ),
                UserIntent::Confirm => (
                    DialogueState::Responding,
                    "Glad that was helpful! Is there anything else?".to_string(),
                ),
                UserIntent::Deny => (
                    DialogueState::Gathering {
                        topic: String::new(),
                        questions_asked: 0,
                    },
                    "I apologise. Could you rephrase your question?".to_string(),
                ),
                _ => (
                    DialogueState::Responding,
                    "Is there anything else I can help you with?".to_string(),
                ),
            },

            DialogueState::Clarifying(_) => match intent {
                UserIntent::Query(q) => {
                    let topic = Self::extract_topic(q);
                    (
                        DialogueState::Gathering {
                            topic: topic.clone(),
                            questions_asked: 1,
                        },
                        format!("Let me gather information about '{topic}'."),
                    )
                }
                UserIntent::Confirm => (
                    DialogueState::Responding,
                    "Thank you for confirming. Here's my response.".to_string(),
                ),
                _ => (
                    DialogueState::Clarifying(input.to_string()),
                    "I need a bit more context to clarify properly.".to_string(),
                ),
            },

            DialogueState::Closing => (
                DialogueState::Closing,
                "The conversation has ended. Please start a new one.".to_string(),
            ),

            DialogueState::Error(e) => (
                DialogueState::Error(e.clone()),
                format!("An error occurred: {e}. Please try again."),
            ),
        }
    }

    /// Extract a short topic label from a query string.
    fn extract_topic(query: &str) -> String {
        // Use the first 5 words of the query as the topic label, stripped of '?'
        let words: Vec<&str> = query
            .split_whitespace()
            .map(|w| w.trim_end_matches('?'))
            .filter(|w| !w.is_empty())
            .take(5)
            .collect();
        if words.is_empty() {
            "general".to_string()
        } else {
            words.join(" ")
        }
    }
}

impl Default for DialogueManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Intent detection tests ---

    #[test]
    fn test_intent_greet_hello() {
        assert_eq!(
            DialogueManager::detect_intent("hello there"),
            UserIntent::Greet
        );
    }

    #[test]
    fn test_intent_greet_hi() {
        assert_eq!(DialogueManager::detect_intent("hi!"), UserIntent::Greet);
    }

    #[test]
    fn test_intent_goodbye_bye() {
        assert_eq!(DialogueManager::detect_intent("bye"), UserIntent::Goodbye);
    }

    #[test]
    fn test_intent_goodbye_full() {
        assert_eq!(
            DialogueManager::detect_intent("goodbye friend"),
            UserIntent::Goodbye
        );
    }

    #[test]
    fn test_intent_query() {
        let intent = DialogueManager::detect_intent("What is SPARQL?");
        matches!(intent, UserIntent::Query(_));
    }

    #[test]
    fn test_intent_clarify() {
        let intent = DialogueManager::detect_intent("what do you mean by that?");
        matches!(intent, UserIntent::Clarify(_));
    }

    #[test]
    fn test_intent_confirm_yes() {
        assert_eq!(
            DialogueManager::detect_intent("yes please"),
            UserIntent::Confirm
        );
    }

    #[test]
    fn test_intent_confirm_ok() {
        assert_eq!(
            DialogueManager::detect_intent("ok, go ahead"),
            UserIntent::Confirm
        );
    }

    #[test]
    fn test_intent_deny() {
        assert_eq!(
            DialogueManager::detect_intent("no, that is wrong"),
            UserIntent::Deny
        );
    }

    #[test]
    fn test_intent_unknown() {
        assert_eq!(
            DialogueManager::detect_intent("tell me something interesting"),
            UserIntent::Unknown
        );
    }

    #[test]
    fn test_intent_case_insensitive() {
        assert_eq!(DialogueManager::detect_intent("HELLO"), UserIntent::Greet);
    }

    // --- Initial state test ---

    #[test]
    fn test_initial_state_is_greeting() {
        let dm = DialogueManager::new();
        assert_eq!(*dm.current_state(), DialogueState::Greeting);
    }

    // --- State transition tests ---

    #[test]
    fn test_greeting_plus_greet_stays_greeting() {
        let mut dm = DialogueManager::new();
        dm.process_turn("hello");
        assert_eq!(*dm.current_state(), DialogueState::Greeting);
    }

    #[test]
    fn test_greeting_plus_query_goes_gathering() {
        let mut dm = DialogueManager::new();
        dm.process_turn("What is RDF?");
        matches!(*dm.current_state(), DialogueState::Gathering { .. });
    }

    #[test]
    fn test_gathering_plus_query_goes_processing() {
        let mut dm = DialogueManager::new();
        dm.process_turn("What is RDF?");
        dm.process_turn("Can you explain more?");
        assert_eq!(*dm.current_state(), DialogueState::Processing);
    }

    #[test]
    fn test_processing_goes_responding_on_any_input() {
        let mut dm = DialogueManager::new();
        dm.process_turn("What is SPARQL?");
        dm.process_turn("Tell me more?");
        dm.process_turn("anything");
        assert_eq!(*dm.current_state(), DialogueState::Responding);
    }

    #[test]
    fn test_responding_plus_query_goes_gathering() {
        let mut dm = DialogueManager::new();
        dm.process_turn("What is SPARQL?");
        dm.process_turn("More details?");
        dm.process_turn("now respond");
        dm.process_turn("What about OWL?");
        matches!(*dm.current_state(), DialogueState::Gathering { .. });
    }

    #[test]
    fn test_any_state_goodbye_goes_closing() {
        let mut dm = DialogueManager::new();
        dm.process_turn("goodbye");
        assert_eq!(*dm.current_state(), DialogueState::Closing);
    }

    #[test]
    fn test_gathering_plus_goodbye_goes_closing() {
        let mut dm = DialogueManager::new();
        dm.process_turn("What is SPARQL?");
        dm.process_turn("bye");
        assert_eq!(*dm.current_state(), DialogueState::Closing);
    }

    // --- History tests ---

    #[test]
    fn test_turn_history_empty_initially() {
        let dm = DialogueManager::new();
        assert_eq!(dm.turn_count(), 0);
    }

    #[test]
    fn test_turn_count_increments() {
        let mut dm = DialogueManager::new();
        dm.process_turn("hello");
        dm.process_turn("What is OWL?");
        assert_eq!(dm.turn_count(), 2);
    }

    #[test]
    fn test_history_records_turns() {
        let mut dm = DialogueManager::new();
        dm.process_turn("hello");
        let turn = &dm.history()[0];
        assert_eq!(turn.user_input, "hello");
        assert_eq!(turn.intent, UserIntent::Greet);
    }

    #[test]
    fn test_history_state_before_after() {
        let mut dm = DialogueManager::new();
        dm.process_turn("hello");
        let turn = &dm.history()[0];
        assert_eq!(turn.state_before, DialogueState::Greeting);
        assert_eq!(turn.state_after, DialogueState::Greeting);
    }

    #[test]
    fn test_history_response_non_empty() {
        let mut dm = DialogueManager::new();
        let turn = dm.process_turn("hello");
        assert!(!turn.response.is_empty());
    }

    // --- Reset tests ---

    #[test]
    fn test_reset_clears_history() {
        let mut dm = DialogueManager::new();
        dm.process_turn("hello");
        dm.reset();
        assert_eq!(dm.turn_count(), 0);
    }

    #[test]
    fn test_reset_returns_to_greeting() {
        let mut dm = DialogueManager::new();
        dm.process_turn("What is SPARQL?");
        dm.reset();
        assert_eq!(*dm.current_state(), DialogueState::Greeting);
    }

    // --- Multi-turn dialogue sequence ---

    #[test]
    fn test_full_conversation_sequence() {
        let mut dm = DialogueManager::new();
        let t1 = dm.process_turn("hello");
        assert_eq!(t1.intent, UserIntent::Greet);

        let t2 = dm.process_turn("What is SHACL?");
        matches!(t2.intent, UserIntent::Query(_));

        let t3 = dm.process_turn("Tell me more about constraints?");
        matches!(t3.intent, UserIntent::Query(_));

        let t4 = dm.process_turn("something");
        assert_eq!(*dm.current_state(), DialogueState::Responding);
        assert!(!t4.response.is_empty());

        dm.process_turn("goodbye");
        assert_eq!(*dm.current_state(), DialogueState::Closing);
    }

    // --- Default impl ---

    #[test]
    fn test_default_same_as_new() {
        let dm = DialogueManager::default();
        assert_eq!(*dm.current_state(), DialogueState::Greeting);
        assert_eq!(dm.turn_count(), 0);
    }

    // --- Clarify from Greeting ---

    #[test]
    fn test_greeting_clarify() {
        let mut dm = DialogueManager::new();
        dm.process_turn("what do you mean");
        matches!(*dm.current_state(), DialogueState::Clarifying(_));
    }

    // --- Closing state is sticky ---

    #[test]
    fn test_closing_state_stays_closed() {
        let mut dm = DialogueManager::new();
        dm.process_turn("bye");
        dm.process_turn("hello");
        assert_eq!(*dm.current_state(), DialogueState::Closing);
    }
}
