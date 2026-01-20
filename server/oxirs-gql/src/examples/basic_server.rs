//! Basic GraphQL Server Example
//!
//! This example demonstrates how to set up a basic GraphQL server
//! with RDF integration using OxiRS GraphQL.

use anyhow::Result;
use oxirs_gql::{GraphQLConfig, GraphQLServer, RdfStore};
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting OxiRS GraphQL Basic Server Example");

    // Create RDF store
    let store = Arc::new(RdfStore::new()?);
    
    // Load some sample data
    load_sample_data(&store).await?;

    // Configure GraphQL server
    let config = GraphQLConfig {
        enable_introspection: true,
        enable_playground: true,
        max_query_depth: Some(10),
        max_query_complexity: Some(1000),
        enable_query_validation: true,
        ..Default::default()
    };

    // Create and configure server
    let server = GraphQLServer::new(store.clone())
        .with_config(config);

    info!("GraphQL server configured");
    info!("GraphQL Playground will be available at http://127.0.0.1:4000/playground");
    info!("GraphQL endpoint available at http://127.0.0.1:4000/graphql");

    // Start the server
    server.start("127.0.0.1:4000").await?;

    Ok(())
}

/// Load sample RDF data into the store
async fn load_sample_data(store: &Arc<RdfStore>) -> Result<()> {
    info!("Loading sample RDF data");

    // Create a mutable reference to the store for data loading
    let mut store_mut = RdfStore::new()?;

    // Add some sample triples
    store_mut.insert_triple(
        "http://example.org/person/1",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://example.org/Person",
    )?;

    store_mut.insert_triple(
        "http://example.org/person/1",
        "http://example.org/name",
        "\"John Doe\"",
    )?;

    store_mut.insert_triple(
        "http://example.org/person/1",
        "http://example.org/age",
        "\"30\"",
    )?;

    store_mut.insert_triple(
        "http://example.org/person/2",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://example.org/Person",
    )?;

    store_mut.insert_triple(
        "http://example.org/person/2",
        "http://example.org/name",
        "\"Jane Smith\"",
    )?;

    store_mut.insert_triple(
        "http://example.org/person/2",
        "http://example.org/age",
        "\"28\"",
    )?;

    info!("Sample data loaded successfully");
    Ok(())
}