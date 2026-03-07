//! Intent prediction for revolutionary chat optimization

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Types of user intent
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentType {
    /// User is asking a question
    Question,
    /// User is making a request
    Request,
    /// User is providing information
    Information,
    /// User is seeking clarification
    Clarification,
    /// User is providing feedback
    Feedback,
    /// User wants to end the conversation
    Goodbye,
    /// Unknown or ambiguous intent
    Unknown,
}

impl IntentType {
    /// Get a human-readable label for this intent type
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Question => "question",
            Self::Request => "request",
            Self::Information => "information",
            Self::Clarification => "clarification",
            Self::Feedback => "feedback",
            Self::Goodbye => "goodbye",
            Self::Unknown => "unknown",
        }
    }
}

/// A predicted intent for a message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedIntent {
    /// The predicted intent type
    pub intent_type: IntentType,
    /// Confidence in this prediction (0.0 to 1.0)
    pub confidence: f64,
    /// The original message text
    pub message_text: String,
    /// Secondary intent (if applicable)
    pub secondary_intent: Option<IntentType>,
    /// Keywords that contributed to this prediction
    pub contributing_keywords: Vec<String>,
}

impl PredictedIntent {
    /// Create a new predicted intent
    pub fn new(intent_type: IntentType, confidence: f64, message_text: String) -> Self {
        Self {
            intent_type,
            confidence,
            message_text,
            secondary_intent: None,
            contributing_keywords: Vec::new(),
        }
    }
}

/// Predicts user intent from message text
#[derive(Debug)]
pub struct IntentPredictor {
    intent_history: VecDeque<PredictedIntent>,
    max_history: usize,
}

impl IntentPredictor {
    /// Create a new intent predictor
    pub fn new() -> Self {
        Self {
            intent_history: VecDeque::with_capacity(100),
            max_history: 100,
        }
    }

    /// Predict the intent of a single message
    pub fn predict(&mut self, text: &str) -> Result<PredictedIntent> {
        let intent = self.predict_single_intent(text);
        self.intent_history.push_back(intent.clone());

        // Keep history manageable
        while self.intent_history.len() > self.max_history {
            self.intent_history.pop_front();
        }

        Ok(intent)
    }

    /// Predict intents for a batch of messages
    pub fn predict_batch(&mut self, messages: &[String]) -> Result<Vec<PredictedIntent>> {
        let mut results = Vec::with_capacity(messages.len());
        for message in messages {
            results.push(self.predict(message)?);
        }
        Ok(results)
    }

    /// Get the recent intent history
    pub fn recent_intents(&self, limit: usize) -> Vec<&PredictedIntent> {
        let start = if self.intent_history.len() > limit {
            self.intent_history.len() - limit
        } else {
            0
        };
        self.intent_history.iter().skip(start).collect()
    }

    // Private: classify text using simple heuristics
    fn predict_single_intent(&self, text: &str) -> PredictedIntent {
        let text_lower = text.to_lowercase();

        let (intent_type, confidence, keywords) = if text_lower.contains("bye")
            || text_lower.contains("goodbye")
            || text_lower.contains("exit")
            || text_lower.contains("quit")
        {
            (IntentType::Goodbye, 0.90, vec!["farewell".to_string()])
        } else if text_lower.starts_with("please ")
            || text_lower.starts_with("can you ")
            || text_lower.starts_with("could you ")
            || text_lower.starts_with("show me ")
            || text_lower.starts_with("give me ")
            || text_lower.starts_with("find ")
        {
            (IntentType::Request, 0.80, vec!["request".to_string()])
        } else if text_lower.contains("thank")
            || text_lower.contains("great")
            || text_lower.contains("good")
            || text_lower.contains("nice")
            || text_lower.contains("bad")
            || text_lower.contains("wrong")
            || text_lower.contains("incorrect")
        {
            (IntentType::Feedback, 0.70, vec!["feedback".to_string()])
        } else if text_lower.ends_with('?')
            || text_lower.starts_with("what ")
            || text_lower.starts_with("how ")
            || text_lower.starts_with("why ")
            || text_lower.starts_with("when ")
            || text_lower.starts_with("where ")
            || text_lower.starts_with("who ")
            || text_lower.starts_with("which ")
        {
            let mut kw = vec!["question".to_string()];
            if text_lower.starts_with("how ") {
                kw.push("how".to_string());
            }
            (IntentType::Question, 0.85, kw)
        } else if text_lower.starts_with("i ")
            || text_lower.starts_with("the ")
            || text_lower.starts_with("this ")
            || text_lower.starts_with("my ")
        {
            (
                IntentType::Information,
                0.60,
                vec!["information".to_string()],
            )
        } else {
            (IntentType::Unknown, 0.40, Vec::new())
        };

        let mut intent = PredictedIntent::new(intent_type, confidence, text.to_string());
        intent.contributing_keywords = keywords;
        intent
    }
}

impl Default for IntentPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_question_detection() {
        let mut predictor = IntentPredictor::new();
        let intent = predictor.predict("What is SPARQL?").expect("predict");
        assert_eq!(intent.intent_type, IntentType::Question);
        assert!(intent.confidence > 0.5);
    }

    #[test]
    fn test_request_detection() {
        let mut predictor = IntentPredictor::new();
        let intent = predictor
            .predict("Can you show me an example?")
            .expect("predict");
        assert_eq!(intent.intent_type, IntentType::Request);
    }

    #[test]
    fn test_goodbye_detection() {
        let mut predictor = IntentPredictor::new();
        let intent = predictor.predict("goodbye, thanks!").expect("predict");
        assert_eq!(intent.intent_type, IntentType::Goodbye);
    }

    #[test]
    fn test_feedback_detection() {
        let mut predictor = IntentPredictor::new();
        let intent = predictor
            .predict("That's wrong, please fix it")
            .expect("predict");
        assert_eq!(intent.intent_type, IntentType::Feedback);
    }

    #[test]
    fn test_intent_history() {
        let mut predictor = IntentPredictor::new();
        predictor.predict("What is RDF?").expect("predict 1");
        predictor.predict("Show me examples").expect("predict 2");
        predictor.predict("Thank you").expect("predict 3");

        let history = predictor.recent_intents(3);
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_batch_prediction() {
        let mut predictor = IntentPredictor::new();
        let messages = vec![
            "What is SPARQL?".to_string(),
            "Please show me an example".to_string(),
        ];
        let intents = predictor.predict_batch(&messages).expect("batch predict");
        assert_eq!(intents.len(), 2);
    }

    #[test]
    fn test_intent_type_display() {
        assert_eq!(IntentType::Question.as_str(), "question");
        assert_eq!(IntentType::Request.as_str(), "request");
        assert_eq!(IntentType::Unknown.as_str(), "unknown");
    }
}
