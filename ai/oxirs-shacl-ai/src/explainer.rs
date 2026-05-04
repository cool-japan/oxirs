//! Plain-English explanation of SHACL constraint violations via LLM.
//!
//! [`ConstraintExplainer`] wraps any [`CompletionProvider`] and translates
//! raw SHACL violation reports into human-readable explanations suitable for
//! display to non-technical users.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_shacl_ai::explainer::ConstraintExplainer;
//! use oxirs_shacl_ai::llm::LocalProvider;
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = Arc::new(LocalProvider::new());
//! let explainer = ConstraintExplainer::new(provider);
//! let explanation = explainer
//!     .explain("sh:minCount violation: foaf:name must appear at least once")
//!     .await?;
//! assert!(!explanation.is_empty());
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use crate::llm::prompt::ShaclPrompts;
use crate::llm::provider::{CompletionProvider, CompletionRequest};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by [`ConstraintExplainer::explain`].
#[derive(Debug, thiserror::Error)]
pub enum ExplainerError {
    /// The LLM provider returned an error.
    #[error("LLM provider error: {0}")]
    Provider(String),
}

// ---------------------------------------------------------------------------
// Explainer
// ---------------------------------------------------------------------------

/// Translates SHACL violation summaries into plain-English explanations.
pub struct ConstraintExplainer {
    provider: Arc<dyn CompletionProvider>,
    model: String,
}

impl ConstraintExplainer {
    /// Create an explainer using `provider` and the `"local"` model identifier.
    pub fn new(provider: Arc<dyn CompletionProvider>) -> Self {
        Self {
            provider,
            model: "local".to_string(),
        }
    }

    /// Override the model identifier forwarded to the provider.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Explain a SHACL violation report in plain English.
    ///
    /// `violation_summary` should be a concise description of the violation,
    /// e.g. `"sh:minCount violation: foaf:name must appear at least once on
    /// node <http://example.org/Alice>"`.
    pub async fn explain(&self, violation_summary: &str) -> Result<String, ExplainerError> {
        let messages = ShaclPrompts::violation_explanation_prompt(violation_summary);
        let request = CompletionRequest {
            model: self.model.clone(),
            messages,
            max_tokens: Some(500),
            temperature: Some(0.3),
        };

        let response = self
            .provider
            .complete(&request)
            .await
            .map_err(|e| ExplainerError::Provider(e.to_string()))?;

        Ok(response.content)
    }

    /// Suggest concrete fixes for the given violation.
    pub async fn suggest_fix(&self, violation_summary: &str) -> Result<String, ExplainerError> {
        let messages = ShaclPrompts::fix_suggestion_prompt(violation_summary);
        let request = CompletionRequest {
            model: self.model.clone(),
            messages,
            max_tokens: Some(500),
            temperature: Some(0.3),
        };

        let response = self
            .provider
            .complete(&request)
            .await
            .map_err(|e| ExplainerError::Provider(e.to_string()))?;

        Ok(response.content)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LocalProvider;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_explain_non_empty() {
        let provider = Arc::new(LocalProvider::new());
        let explainer = ConstraintExplainer::new(provider);
        let result = explainer
            .explain("sh:minCount violation: foaf:name must appear at least once")
            .await
            .expect("explain should succeed");
        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn test_suggest_fix_non_empty() {
        let provider = Arc::new(LocalProvider::new());
        let explainer = ConstraintExplainer::new(provider);
        let result = explainer
            .suggest_fix("sh:maxCount violation: ex:email must appear at most once")
            .await
            .expect("suggest_fix should succeed");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_with_model_overrides() {
        let provider = Arc::new(LocalProvider::new());
        let explainer = ConstraintExplainer::new(provider).with_model("claude-3-5-sonnet");
        assert_eq!(explainer.model, "claude-3-5-sonnet");
    }
}
