//! Basic usage example for OxiRS

use oxirs_core::Store;
use oxirs_core::parser::{Parser, RdfFormat};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();

    // Create a new RDF store
    let mut store = Store::new()?;
    
    // Parse some RDF data
    let parser = Parser::new(RdfFormat::Turtle);
    let turtle_data = r#"
        @prefix ex: <http://example.org/> .
        
        ex:alice ex:knows ex:bob .
        ex:bob ex:age 30 .
        ex:alice ex:name "Alice" .
    "#;
    
    let graph = parser.parse_str(turtle_data)?;
    println!("Parsed {} triples", graph.len());
    
    // Insert data into store
    store.insert("http://example.org/alice", "http://example.org/knows", "http://example.org/bob")?;
    store.insert("http://example.org/bob", "http://example.org/age", "30")?;
    store.insert("http://example.org/alice", "http://example.org/name", "Alice")?;
    
    // Query the store
    let sparql_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
    let results = store.query(sparql_query)?;
    println!("Query executed successfully");
    
    println!("Basic OxiRS usage completed!");
    Ok(())
}