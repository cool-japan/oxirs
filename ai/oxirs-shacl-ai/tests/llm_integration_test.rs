//! Integration tests for the LLM provider abstraction and related components.
//!
//! All tests run offline using `LocalProvider` — no API keys or network access required.

use std::collections::HashMap;
use std::sync::Arc;

use oxirs_shacl_ai::llm::{CompletionProvider, CompletionRequest, LocalProvider, Message, Role};
use oxirs_shacl_ai::shape_nl_generator::GeneratorError;
use oxirs_shacl_ai::{ConstraintExplainer, ShapeNlGenerator};

// ---------------------------------------------------------------------------
// LocalProvider tests
// ---------------------------------------------------------------------------

/// The same prompt must always produce the same response.
#[tokio::test]
async fn test_local_provider_deterministic() {
    let provider = LocalProvider::new();
    let messages = vec![Message {
        role: Role::User,
        content: "determinism check".to_string(),
    }];
    let req = CompletionRequest {
        model: "local".to_string(),
        messages: messages.clone(),
        max_tokens: Some(100),
        temperature: Some(0.0),
    };
    let r1 = provider.complete(&req).await.expect("first call ok");
    let r2 = provider.complete(&req).await.expect("second call ok");
    assert_eq!(
        r1.content, r2.content,
        "LocalProvider must be deterministic"
    );
}

/// `embed` must return as many vectors as texts, each of non-zero length.
#[tokio::test]
async fn test_local_provider_embedding() {
    let provider = LocalProvider::new();
    let texts = vec!["hello".to_string(), "world".to_string()];
    let embeddings = provider.embed(&texts).await.expect("embed ok");
    assert_eq!(embeddings.len(), 2, "one embedding per text");
    for emb in &embeddings {
        assert!(!emb.is_empty(), "embedding vector must be non-empty");
    }
}

/// The capabilities struct must have plausible values.
#[tokio::test]
async fn test_local_provider_capabilities() {
    let provider = LocalProvider::new();
    let caps = provider.capabilities();
    assert!(
        caps.max_context_tokens > 0,
        "max_context_tokens must be positive"
    );
    assert!(
        caps.supports_embeddings,
        "LocalProvider must support embeddings"
    );
}

/// Embeddings for different texts must differ.
#[tokio::test]
async fn test_local_provider_different_texts_differ() {
    let provider = LocalProvider::new();
    let texts = vec!["apple".to_string(), "orange".to_string()];
    let embeddings = provider.embed(&texts).await.expect("ok");
    assert_ne!(
        embeddings[0], embeddings[1],
        "different texts should produce different embeddings"
    );
}

// ---------------------------------------------------------------------------
// ConstraintExplainer tests
// ---------------------------------------------------------------------------

/// `explain` must return a non-empty string.
#[tokio::test]
async fn test_explainer_non_empty() {
    let provider = Arc::new(LocalProvider::new());
    let explainer = ConstraintExplainer::new(provider);
    let explanation = explainer
        .explain("sh:minCount violation: foaf:name must appear at least once")
        .await
        .expect("explain ok");
    assert!(!explanation.is_empty(), "explanation must not be empty");
}

/// `explain` result is deterministic (same violation → same text).
#[tokio::test]
async fn test_explainer_deterministic() {
    let provider = Arc::new(LocalProvider::new());
    let explainer = ConstraintExplainer::new(provider);
    let v = "sh:maxCount violation: ex:email must appear at most once";
    let e1 = explainer.explain(v).await.expect("ok");
    let e2 = explainer.explain(v).await.expect("ok");
    assert_eq!(e1, e2);
}

// ---------------------------------------------------------------------------
// ShapeNlGenerator tests
// ---------------------------------------------------------------------------

/// When the provider is configured to return valid JSON, `propose` must parse
/// it into a `ProposedShape` with correct fields.
#[tokio::test]
async fn test_shape_generator_parse() {
    let valid_json = r#"{
        "node_shape": {
            "target_class": "foaf:Person",
            "properties": [
                {
                    "path": "foaf:name",
                    "min_count": 1,
                    "max_count": null,
                    "datatype": "xsd:string"
                }
            ]
        }
    }"#;

    let mut canned = HashMap::new();
    // The keyword "person must have" will match our query below.
    canned.insert("person must have".to_string(), valid_json.to_string());
    let provider = Arc::new(LocalProvider::with_responses(canned));
    let gen = ShapeNlGenerator::new(provider);

    let shapes = gen
        .propose("every person must have a foaf:name")
        .await
        .expect("propose ok");

    assert_eq!(shapes.len(), 1);
    let shape = &shapes[0];
    assert_eq!(shape.target_class, "foaf:Person");
    assert_eq!(shape.property_constraints.len(), 1);

    let prop = &shape.property_constraints[0];
    assert_eq!(prop.path, "foaf:name");
    assert_eq!(prop.min_count, Some(1));
    assert_eq!(prop.max_count, None);
    assert_eq!(prop.datatype.as_deref(), Some("xsd:string"));
}

/// When the provider returns non-JSON, `propose` must return a `ParseError`.
#[tokio::test]
async fn test_shape_generator_parse_error_on_bad_response() {
    let provider = Arc::new(LocalProvider::new()); // returns non-JSON fallback
    let gen = ShapeNlGenerator::new(provider);
    let result = gen.propose("no canned response for this input xyz42").await;
    assert!(
        matches!(result, Err(GeneratorError::ParseError(_))),
        "expected ParseError, got: {result:?}"
    );
}

/// `with_model` must override the model field.
#[tokio::test]
async fn test_shape_generator_with_model() {
    let valid_json = r#"{"node_shape":{"target_class":"ex:Book","properties":[]}}"#;
    let mut canned = HashMap::new();
    canned.insert("book".to_string(), valid_json.to_string());
    let provider = Arc::new(LocalProvider::with_responses(canned));
    let gen = ShapeNlGenerator::new(provider).with_model("gpt-4o");

    // The model name override should not break the logic.
    let shapes = gen.propose("a book resource").await.expect("ok");
    assert_eq!(shapes[0].target_class, "ex:Book");
}
