//! GraphQL Demo for OxiRS
//!
//! This example demonstrates the GraphQL functionality of OxiRS, including:
//! - GraphQL schema generation from RDF ontologies
//! - GraphQL query parsing and execution
//! - RDF-specific scalar types
//! - GraphQL Playground web interface

use anyhow::Result;
use oxirs_gql::{
    parser::parse_document,
    rdf_scalars::RdfScalars,
    schema::{SchemaGenerationConfig, SchemaGenerator},
    GraphQLConfig, GraphQLServer, RdfStore,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 OxiRS GraphQL Demo");
    println!("====================");

    // 1. Create an RDF store
    println!("📦 Creating RDF store...");
    let store = Arc::new(RdfStore::new()?);

    // 2. Generate GraphQL schema from RDF ontology
    println!("🏗️  Generating GraphQL schema from RDF ontology...");
    let schema_generator = SchemaGenerator::new().with_config(SchemaGenerationConfig::default());

    // Generate schema from FOAF ontology (mock)
    let schema_sdl = schema_generator
        .generate_from_ontology("http://xmlns.com/foaf/0.1/")
        .await?;
    println!("📄 Generated GraphQL Schema:");
    println!("{schema_sdl}");

    // 3. Parse a sample GraphQL query
    println!("\n🔍 Parsing GraphQL query...");
    let sample_query = r#"
    query GetPeople($limit: Int) {
      people(limit: $limit) {
        id
        uri
        name
        email
        knows(limit: 3) {
          name
        }
      }
    }
    "#;

    match parse_document(sample_query) {
        Ok(document) => {
            println!("✅ Successfully parsed GraphQL query");
            println!("   Operation: {:?}", document.definitions[0]);
        }
        Err(e) => {
            println!("❌ Failed to parse query: {e}");
        }
    }

    // 4. Demonstrate RDF scalar types
    println!("\n🔧 Testing RDF scalar types...");
    let iri_scalar = RdfScalars::iri();
    let literal_scalar = RdfScalars::literal();
    let datetime_scalar = RdfScalars::datetime();

    println!("   - IRI scalar: {}", iri_scalar.name);
    println!("   - Literal scalar: {}", literal_scalar.name);
    println!("   - DateTime scalar: {}", datetime_scalar.name);

    // 5. Create and configure GraphQL server
    println!("\n🌐 Starting GraphQL server...");
    let config = GraphQLConfig::default();

    let server = GraphQLServer::new(store).with_config(config);

    println!("🎯 Server configuration:");
    println!("   - Playground enabled: ✅");
    println!("   - Introspection enabled: ✅");
    println!("   - Max query depth: 10");
    println!("   - Max query complexity: 1000");

    // 6. Start the server
    println!("\n🚀 Starting server on http://localhost:4000");
    println!("📊 GraphQL Playground: http://localhost:4000/");
    println!("🔗 GraphQL endpoint: http://localhost:4000/graphql");
    println!("\n💡 Try these sample queries in the playground:");
    println!("   {{ hello }}");
    println!("   {{ version }}");
    println!("   {{ triples }}");
    println!("   {{ subjects(limit: 5) }}");

    server.start("127.0.0.1:4000").await?;

    Ok(())
}
