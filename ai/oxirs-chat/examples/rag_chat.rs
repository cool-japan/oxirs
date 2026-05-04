//! RAG (Retrieval-Augmented Generation) Chat Example
//!
//! Demonstrates the RAG retrieval pipeline directly: embed a query, retrieve
//! semantically relevant triples from the knowledge graph, and generate a
//! grounded answer. Distinct from `kg_chat.rs` which uses the high-level
//! `OxiRSChat` facade; this example works directly with `rag::RagEngine` to
//! show the retrieval internals.
//!
//! Run with:
//!   cargo run --example rag_chat
//!
//! Optional environment variables:
//!   OPENAI_API_KEY     - Use OpenAI for response generation
//!   ANTHROPIC_API_KEY  - Use Anthropic for response generation

use anyhow::Result;
use oxirs_chat::{
    rag::{RagConfig, RagEngine, RetrievalConfig},
    ChatConfig, OxiRSChat,
};
use oxirs_core::{
    model::{Literal, NamedNode, Triple},
    ConcreteStore, Store,
};
use std::sync::Arc;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

/// Populate the store with document-chunk-style triples that simulate a
/// research paper corpus indexed as RDF.
fn populate_doc_corpus(store: &mut ConcreteStore) -> Result<()> {
    let has_abstract = NamedNode::new("http://purl.org/dc/terms/abstract")?;
    let has_title = NamedNode::new("http://purl.org/dc/terms/title")?;
    let has_subject = NamedNode::new("http://purl.org/dc/terms/subject")?;
    let part_of = NamedNode::new("http://purl.org/dc/terms/isPartOf")?;

    // Paper 1 — RDF overview
    let paper1 = NamedNode::new("http://example.org/paper/rdf-overview")?;
    let journal = NamedNode::new("http://example.org/journal/semantic-web")?;
    store.insert_triple(Triple::new(
        paper1.clone(),
        has_title.clone(),
        Literal::new("An Overview of the Resource Description Framework"),
    ))?;
    store.insert_triple(Triple::new(
        paper1.clone(),
        has_abstract.clone(),
        Literal::new(
            "RDF is a standard model for data interchange on the Web. \
             RDF has features that facilitate data merging even if the underlying \
             schemas differ, and it specifically supports the evolution of schemas \
             over time without requiring all the data consumers to be changed.",
        ),
    ))?;
    store.insert_triple(Triple::new(
        paper1.clone(),
        has_subject.clone(),
        Literal::new("Semantic Web, Linked Data, Knowledge Graphs"),
    ))?;
    store.insert_triple(Triple::new(paper1, part_of.clone(), journal.clone()))?;

    // Paper 2 — SPARQL
    let paper2 = NamedNode::new("http://example.org/paper/sparql-tutorial")?;
    store.insert_triple(Triple::new(
        paper2.clone(),
        has_title.clone(),
        Literal::new("SPARQL 1.1 Query Language for RDF"),
    ))?;
    store.insert_triple(Triple::new(
        paper2.clone(),
        has_abstract.clone(),
        Literal::new(
            "SPARQL is the standard query language for RDF. It enables querying, \
             updating, and federating over graph-structured data. \
             SPARQL 1.1 adds aggregation, property paths, subqueries, and \
             federated query capabilities.",
        ),
    ))?;
    store.insert_triple(Triple::new(
        paper2.clone(),
        has_subject.clone(),
        Literal::new("SPARQL, Query Language, RDF"),
    ))?;
    store.insert_triple(Triple::new(paper2, part_of.clone(), journal.clone()))?;

    // Paper 3 — Knowledge Graph Embeddings
    let paper3 = NamedNode::new("http://example.org/paper/kg-embeddings")?;
    let ai_journal = NamedNode::new("http://example.org/journal/ai-research")?;
    store.insert_triple(Triple::new(
        paper3.clone(),
        has_title.clone(),
        Literal::new("Knowledge Graph Embedding Methods"),
    ))?;
    store.insert_triple(Triple::new(
        paper3.clone(),
        has_abstract.clone(),
        Literal::new(
            "Knowledge graph embedding methods such as TransE, DistMult, ComplEx, \
             and RotatE learn low-dimensional representations of entities and \
             relations. These embeddings support link prediction, entity \
             classification, and question answering over knowledge graphs.",
        ),
    ))?;
    store.insert_triple(Triple::new(
        paper3.clone(),
        has_subject.clone(),
        Literal::new("Machine Learning, Knowledge Graphs, Embeddings"),
    ))?;
    store.insert_triple(Triple::new(paper3, part_of, ai_journal))?;

    info!("Document corpus: {} triples inserted", store.len()?);
    Ok(())
}

/// Show retrieved context from the RAG engine directly.
async fn run_rag_retrieval(rag_engine: &mut RagEngine, query: &str) -> Result<()> {
    println!("\nQuery: {query}");
    println!("Retrieving context from RAG engine...");

    let context = rag_engine.retrieve(query).await?;

    println!("  Context score:     {:.3}", context.context_score);
    println!("  Semantic results:  {}", context.semantic_results.len());
    println!("  Extracted entities: {}", context.extracted_entities.len());
    println!("  Assembly time:     {:?}", context.assembly_time);

    if !context.semantic_results.is_empty() {
        println!("  Top result: {}", context.semantic_results[0].triple);
    }
    if !context.extracted_entities.is_empty() {
        let names: Vec<&str> = context
            .extracted_entities
            .iter()
            .take(3)
            .map(|e| e.text.as_str())
            .collect();
        println!("  Entities: {:?}", names);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialise logging.
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== OxiRS Chat — RAG Chat Example ===");

    // 1. Build and populate the document-chunk RDF store.
    let mut raw_store = ConcreteStore::new()?;
    populate_doc_corpus(&mut raw_store)?;
    let store: Arc<dyn Store> = Arc::new(raw_store);

    // 2. Demonstrate direct RAG engine access with custom retrieval settings.
    info!("--- Part A: Direct RagEngine retrieval ---");
    {
        let rag_config = RagConfig {
            retrieval: RetrievalConfig {
                max_results: 5,
                similarity_threshold: 0.5,
                graph_traversal_depth: 2,
                enable_entity_expansion: true,
                enable_quantum_enhancement: false,
                enable_consciousness_integration: false,
            },
            ..RagConfig::default()
        };

        let mut rag_engine = RagEngine::new(rag_config, store.clone());
        rag_engine.initialize().await?;

        let queries = [
            "What is SPARQL and how does it query graphs?",
            "Knowledge graph embeddings for link prediction",
            "RDF data interchange and schema evolution",
        ];

        for query in &queries {
            if let Err(e) = run_rag_retrieval(&mut rag_engine, query).await {
                warn!("RAG retrieval failed for '{}': {}", query, e);
            }
        }
    }

    // 3. Use the full OxiRSChat facade with the same store to show end-to-end.
    info!("\n--- Part B: End-to-end OxiRSChat with RAG ---");
    {
        let chat_config = ChatConfig {
            max_context_tokens: 8000,
            sliding_window_size: 20,
            enable_context_compression: true,
            temperature: 0.7,
            max_tokens: 2000,
            timeout_seconds: 30,
            enable_topic_tracking: true,
            enable_sentiment_analysis: true,
            enable_intent_detection: true,
        };

        let chat = OxiRSChat::new(chat_config, store).await?;
        let session_id = "rag_chat_demo".to_string();
        let _session = chat.create_session(session_id.clone()).await?;

        let nl_queries = [
            "Explain what SPARQL property paths do.",
            "How are TransE and RotatE different embedding approaches?",
        ];

        for query in &nl_queries {
            println!("\nYou: {query}");
            match chat.process_message(&session_id, query.to_string()).await {
                Ok(response) => {
                    println!("Assistant: {}", response.content.to_text());
                    if let Some(meta) = &response.metadata {
                        if let Some(ms) = meta.processing_time_ms {
                            println!("  [processed in {}ms]", ms);
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Note: LLM call failed ({}). Set OPENAI_API_KEY or \
                         ANTHROPIC_API_KEY to enable full response generation.",
                        e
                    );
                }
            }
        }

        // Export session for inspection.
        match chat.export_session(&session_id).await {
            Ok(data) => {
                info!(
                    "Session exported: id={}, {} messages",
                    data.id,
                    data.messages.len()
                );
            }
            Err(e) => warn!("Session export failed: {}", e),
        }
    }

    info!("Done.");
    Ok(())
}
