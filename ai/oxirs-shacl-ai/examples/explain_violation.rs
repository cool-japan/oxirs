//! Example: explain a SHACL violation using the local (offline) LLM provider.

use oxirs_shacl_ai::{CompletionProvider, ConstraintExplainer, LocalProvider};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use the offline LocalProvider — no API key required.
    let provider = Arc::new(LocalProvider::new());

    // Explain a SHACL violation in plain English.
    let explainer = ConstraintExplainer::new(provider.clone());
    let explanation = explainer
        .explain("sh:minCount violation: foaf:name must appear at least once on node <http://example.org/Alice>")
        .await?;
    println!("Explanation: {explanation}");

    // Demonstrate provider capabilities.
    let caps = provider.capabilities();
    println!("Provider: {}", provider.name());
    println!("  supports_embeddings: {}", caps.supports_embeddings);
    println!("  max_context_tokens:  {}", caps.max_context_tokens);

    Ok(())
}
