//! Deterministic local LLM provider — no network calls, no API keys.
//!
//! `LocalProvider` is the default provider used in tests and offline
//! environments.  All responses are computed deterministically from the
//! prompt content, so the same input always yields the same output.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;

use super::provider::{
    Capabilities, CompletionProvider, CompletionRequest, CompletionResponse, LlmError,
};

/// A deterministic, network-free LLM provider for testing and offline use.
///
/// Responses are looked up by a hash of the message content.  If no canned
/// response is registered for the prompt, a generic fallback is returned.
///
/// # Example
///
/// ```rust
/// use oxirs_shacl_ai::llm::{LocalProvider, CompletionRequest, CompletionProvider, Message, Role};
/// use std::sync::Arc;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let provider = LocalProvider::new();
/// let request = CompletionRequest {
///     model: "local".to_string(),
///     messages: vec![
///         Message { role: Role::User, content: "Hello".to_string() },
///     ],
///     max_tokens: Some(100),
///     temperature: Some(0.0),
/// };
/// let response = provider.complete(&request).await?;
/// assert!(!response.content.is_empty());
/// # Ok(())
/// # }
/// ```
pub struct LocalProvider {
    /// Canned responses keyed by prompt hash.
    responses: Arc<HashMap<u64, String>>,
    /// Canned responses keyed by a substring contained in the prompt.
    keyword_responses: Arc<HashMap<String, String>>,
    /// Dimensionality of the embeddings produced by [`LocalProvider::embed`].
    pub embedding_dim: usize,
    /// Capabilities reported by this provider.
    pub caps: Capabilities,
}

impl LocalProvider {
    /// Create a `LocalProvider` with a small set of built-in canned responses.
    pub fn new() -> Self {
        Self {
            responses: Arc::new(HashMap::new()),
            keyword_responses: Arc::new(HashMap::new()),
            embedding_dim: 64,
            caps: Capabilities {
                supports_tools: false,
                supports_embeddings: true,
                supports_streaming: false,
                max_context_tokens: 4096,
            },
        }
    }

    /// Create a `LocalProvider` with custom keyword-keyed canned responses.
    ///
    /// `responses` maps a keyword (substring of the last user message content)
    /// to the string that will be returned as the completion content.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_shacl_ai::llm::{LocalProvider, CompletionRequest, CompletionProvider, Message, Role};
    /// use std::collections::HashMap;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut canned = HashMap::new();
    /// canned.insert(
    ///     "foaf:name".to_string(),
    ///     r#"{"node_shape":{"target_class":"foaf:Person","properties":[]}}"#.to_string(),
    /// );
    /// let provider = LocalProvider::with_responses(canned);
    /// let request = CompletionRequest {
    ///     model: "local".to_string(),
    ///     messages: vec![
    ///         Message { role: Role::User, content: "constraint on foaf:name".to_string() },
    ///     ],
    ///     max_tokens: Some(200),
    ///     temperature: Some(0.0),
    /// };
    /// let response = provider.complete(&request).await?;
    /// assert!(response.content.contains("foaf:Person"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_responses(responses: HashMap<String, String>) -> Self {
        Self {
            responses: Arc::new(HashMap::new()),
            keyword_responses: Arc::new(responses),
            embedding_dim: 64,
            caps: Capabilities {
                supports_tools: false,
                supports_embeddings: true,
                supports_streaming: false,
                max_context_tokens: 4096,
            },
        }
    }

    /// Hash a slice of messages for use as a response lookup key.
    fn hash_messages(messages: &[super::provider::Message]) -> u64 {
        let mut h = DefaultHasher::new();
        for m in messages {
            m.content.hash(&mut h);
        }
        h.finish()
    }

    /// Produce a deterministic unit-normalised embedding for `text`.
    ///
    /// The embedding is computed by accumulating character values into a fixed-
    /// length buffer, then L2-normalising.  The same text always produces the
    /// same vector; different texts produce different vectors (with high
    /// probability for typical NLP inputs).
    pub fn canned_embedding(text: &str, dim: usize) -> Vec<f32> {
        let mut v = vec![0_f32; dim];
        for (i, c) in text.chars().enumerate() {
            v[i % dim] += c as u32 as f32 * 0.001;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    /// Look up a canned response for the given messages.
    ///
    /// Resolution order:
    /// 1. Exact hash match in `self.responses`.
    /// 2. First keyword match in `self.keyword_responses` (checked against the
    ///    last user message content, case-insensitively).
    /// 3. Generic fallback.
    fn lookup(&self, messages: &[super::provider::Message]) -> String {
        // Exact hash match
        let h = Self::hash_messages(messages);
        if let Some(canned) = self.responses.get(&h) {
            return canned.clone();
        }

        // Keyword match against the last user message
        let last_user_content = messages
            .iter()
            .rev()
            .find(|m| m.role == super::provider::Role::User)
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let lower = last_user_content.to_lowercase();
        for (keyword, response) in self.keyword_responses.as_ref() {
            if lower.contains(keyword.to_lowercase().as_str()) {
                return response.clone();
            }
        }

        // Generic fallback
        format!(
            "LocalProvider: deterministic response for {} messages",
            messages.len()
        )
    }
}

impl Default for LocalProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CompletionProvider for LocalProvider {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let content = self.lookup(&request.messages);
        Ok(CompletionResponse {
            content,
            model: "local".to_string(),
            usage: None,
        })
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        let dim = self.embedding_dim;
        Ok(texts
            .iter()
            .map(|t| Self::canned_embedding(t, dim))
            .collect())
    }

    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn name(&self) -> &str {
        "local"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::provider::{Message, Role};

    fn make_messages(content: &str) -> Vec<Message> {
        vec![Message {
            role: Role::User,
            content: content.to_string(),
        }]
    }

    #[tokio::test]
    async fn test_local_provider_returns_non_empty() {
        let p = LocalProvider::new();
        let req = CompletionRequest {
            model: "local".to_string(),
            messages: make_messages("Hello"),
            max_tokens: Some(100),
            temperature: Some(0.0),
        };
        let resp = p.complete(&req).await.expect("local provider never fails");
        assert!(!resp.content.is_empty());
        assert_eq!(resp.model, "local");
    }

    #[tokio::test]
    async fn test_local_provider_deterministic() {
        let p = LocalProvider::new();
        let messages = make_messages("determinism test");
        let req = CompletionRequest {
            model: "local".to_string(),
            messages: messages.clone(),
            max_tokens: Some(100),
            temperature: Some(0.0),
        };
        let r1 = p.complete(&req).await.expect("ok");
        let r2 = p.complete(&req).await.expect("ok");
        assert_eq!(r1.content, r2.content);
    }

    #[tokio::test]
    async fn test_local_provider_keyword_match() {
        let mut canned = HashMap::new();
        canned.insert("foaf:name".to_string(), "canned-response".to_string());
        let p = LocalProvider::with_responses(canned);
        let req = CompletionRequest {
            model: "local".to_string(),
            messages: make_messages("constraint about foaf:name"),
            max_tokens: Some(50),
            temperature: Some(0.0),
        };
        let resp = p.complete(&req).await.expect("ok");
        assert_eq!(resp.content, "canned-response");
    }

    #[tokio::test]
    async fn test_local_provider_embed_length() {
        let p = LocalProvider::new();
        let texts = vec!["hello".to_string(), "world".to_string()];
        let embeddings = p.embed(&texts).await.expect("ok");
        assert_eq!(embeddings.len(), 2);
        for emb in &embeddings {
            assert_eq!(emb.len(), p.embedding_dim);
        }
    }

    #[test]
    fn test_canned_embedding_normalised() {
        let emb = LocalProvider::canned_embedding("test", 8);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "embedding not normalised: {norm}"
        );
    }

    #[test]
    fn test_capabilities_valid() {
        let p = LocalProvider::new();
        let caps = p.capabilities();
        assert!(caps.max_context_tokens > 0);
        assert!(caps.supports_embeddings);
    }
}
