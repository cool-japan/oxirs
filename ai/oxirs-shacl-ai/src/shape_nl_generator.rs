//! Natural-language to SHACL shape generator backed by an LLM provider.
//!
//! [`ShapeNlGenerator`] accepts a free-text constraint description and
//! returns one or more candidate [`ProposedShape`] values that can be used
//! as SHACL node-shape templates.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_shacl_ai::shape_nl_generator::ShapeNlGenerator;
//! use oxirs_shacl_ai::llm::LocalProvider;
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = Arc::new(LocalProvider::new());
//! let gen = ShapeNlGenerator::new(provider);
//! // LocalProvider returns a generic fallback; real providers would parse JSON.
//! let _ = gen.propose("every person must have a foaf:name").await;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use serde::Deserialize;

use crate::llm::prompt::ShaclPrompts;
use crate::llm::provider::{CompletionProvider, CompletionRequest};

// ---------------------------------------------------------------------------
// Internal deserialisation types (mirror of the expected LLM JSON output)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct GeneratedShapeEnvelope {
    node_shape: GeneratedNodeShapeDto,
}

#[derive(Debug, Deserialize)]
struct GeneratedNodeShapeDto {
    target_class: String,
    properties: Vec<GeneratedPropertyDto>,
}

#[derive(Debug, Deserialize)]
struct GeneratedPropertyDto {
    path: String,
    min_count: Option<u32>,
    max_count: Option<u32>,
    datatype: Option<String>,
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A SHACL property constraint proposed by the natural-language generator.
///
/// Named `NlPropertyConstraint` to avoid collision with
/// [`crate::shape::PropertyConstraint`].
#[derive(Debug, Clone)]
pub struct NlPropertyConstraint {
    /// Predicate path (e.g. `"foaf:name"` or `"http://schema.org/name"`).
    pub path: String,
    /// Minimum cardinality, if any.
    pub min_count: Option<u32>,
    /// Maximum cardinality, if any.
    pub max_count: Option<u32>,
    /// Datatype URI, if any.
    pub datatype: Option<String>,
}

/// A candidate SHACL node shape proposed by the generator.
#[derive(Debug, Clone)]
pub struct ProposedShape {
    /// The RDF class this shape targets (e.g. `"foaf:Person"`).
    pub target_class: String,
    /// Property constraints included in the shape.
    pub property_constraints: Vec<NlPropertyConstraint>,
}

impl ProposedShape {
    fn from_dto(dto: GeneratedShapeEnvelope) -> Self {
        ProposedShape {
            target_class: dto.node_shape.target_class,
            property_constraints: dto
                .node_shape
                .properties
                .into_iter()
                .map(|p| NlPropertyConstraint {
                    path: p.path,
                    min_count: p.min_count,
                    max_count: p.max_count,
                    datatype: p.datatype,
                })
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by [`ShapeNlGenerator::propose`].
#[derive(Debug, thiserror::Error)]
pub enum GeneratorError {
    /// The LLM provider returned an error.
    #[error("LLM provider error: {0}")]
    Provider(String),

    /// The LLM output could not be parsed as a `ProposedShape`.
    #[error("failed to parse LLM output: {0}")]
    ParseError(String),
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

/// LLM-backed generator that converts natural-language constraint descriptions
/// into candidate SHACL shapes.
pub struct ShapeNlGenerator {
    provider: Arc<dyn CompletionProvider>,
    model: String,
}

impl ShapeNlGenerator {
    /// Create a generator using `provider` and the `"local"` model identifier.
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

    /// Propose candidate SHACL shapes for the given natural-language description.
    ///
    /// Returns a [`Vec`] so that future multi-shot generation (asking the LLM
    /// for *N* alternatives) can be accommodated without breaking the API.
    pub async fn propose(
        &self,
        constraint_text: &str,
    ) -> Result<Vec<ProposedShape>, GeneratorError> {
        let messages = ShaclPrompts::shape_generation_prompt(constraint_text);
        let request = CompletionRequest {
            model: self.model.clone(),
            messages,
            max_tokens: Some(1000),
            temperature: Some(0.2),
        };

        let response = self
            .provider
            .complete(&request)
            .await
            .map_err(|e| GeneratorError::Provider(e.to_string()))?;

        // Attempt to parse the JSON envelope from the LLM response.
        let envelope: GeneratedShapeEnvelope =
            serde_json::from_str(&response.content).map_err(|e| {
                GeneratorError::ParseError(format!(
                    "LLM output not valid JSON: {e}\nContent: {}",
                    response.content
                ))
            })?;

        Ok(vec![ProposedShape::from_dto(envelope)])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{CompletionProvider, LocalProvider};
    use std::collections::HashMap;

    const VALID_SHAPE_JSON: &str = r#"{
        "node_shape": {
            "target_class": "foaf:Person",
            "properties": [
                {"path": "foaf:name", "min_count": 1, "max_count": null, "datatype": "xsd:string"}
            ]
        }
    }"#;

    #[tokio::test]
    async fn test_propose_with_json_provider() {
        let mut canned = HashMap::new();
        canned.insert("person must have".to_string(), VALID_SHAPE_JSON.to_string());
        let provider = Arc::new(LocalProvider::with_responses(canned));
        let gen = ShapeNlGenerator::new(provider);

        let shapes = gen
            .propose("Every person must have a foaf:name")
            .await
            .expect("should succeed with valid JSON response");

        assert_eq!(shapes.len(), 1);
        assert_eq!(shapes[0].target_class, "foaf:Person");
        assert_eq!(shapes[0].property_constraints.len(), 1);
        let prop = &shapes[0].property_constraints[0];
        assert_eq!(prop.path, "foaf:name");
        assert_eq!(prop.min_count, Some(1));
    }

    #[tokio::test]
    async fn test_propose_parse_error_on_non_json() {
        // LocalProvider default returns a non-JSON fallback string
        let provider = Arc::new(LocalProvider::new());
        let gen = ShapeNlGenerator::new(provider);

        let result = gen
            .propose("something that won't match any canned response")
            .await;
        // The default response is not valid JSON, so we expect a ParseError
        assert!(matches!(result, Err(GeneratorError::ParseError(_))));
    }

    #[test]
    fn test_proposed_shape_has_correct_fields() {
        let envelope = GeneratedShapeEnvelope {
            node_shape: GeneratedNodeShapeDto {
                target_class: "ex:Book".to_string(),
                properties: vec![GeneratedPropertyDto {
                    path: "ex:title".to_string(),
                    min_count: Some(1),
                    max_count: Some(1),
                    datatype: Some("xsd:string".to_string()),
                }],
            },
        };
        let shape = ProposedShape::from_dto(envelope);
        assert_eq!(shape.target_class, "ex:Book");
        assert_eq!(
            shape.property_constraints[0].datatype.as_deref(),
            Some("xsd:string")
        );
    }

    #[test]
    fn test_with_model_overrides() {
        let provider = Arc::new(LocalProvider::new());
        let gen = ShapeNlGenerator::new(provider).with_model("gpt-4o");
        assert_eq!(gen.model, "gpt-4o");
    }
}
