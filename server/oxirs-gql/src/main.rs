//! OxiRS GraphQL Server Binary

use clap::Parser;
use oxirs_gql::{GraphQLServer, GraphQLConfig, RdfStore};
// use oxirs_gql::juniper_server::{JuniperGraphQLServer, GraphQLServerBuilder, GraphQLServerConfig};
use std::sync::Arc;
use std::net::SocketAddr;

#[derive(Parser)]
#[command(name = "oxirs-gql")]
#[command(about = "OxiRS GraphQL server")]
struct Args {
    /// Server port
    #[arg(short, long, default_value = "4000")]
    port: u16,

    /// Server host
    #[arg(long, default_value = "localhost")]
    host: String,

    /// Dataset path
    #[arg(short, long)]
    dataset: Option<String>,

    /// RDF data file to load
    #[arg(short, long)]
    file: Option<String>,

    /// RDF format (turtle, ntriples, rdfxml, jsonld)
    #[arg(long, default_value = "turtle")]
    format: String,

    /// Enable GraphQL playground
    #[arg(long)]
    playground: bool,

    /// Enable introspection
    #[arg(long, default_value = "true")]
    introspection: bool,

    /// Use the new Juniper-based GraphQL server (default: true)
    #[arg(long, default_value = "true")]
    use_juniper: bool,

    /// Enable GraphiQL interface
    #[arg(long, default_value = "true")]
    graphiql: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    let store = if let Some(dataset_path) = args.dataset {
        RdfStore::open(dataset_path)?
    } else {
        RdfStore::new()?
    };
    
    // Load RDF data file if provided
    if let Some(file_path) = args.file {
        println!("Loading RDF data from {} (format: {})", file_path, args.format);
        store.load_file(&file_path, &args.format)?;
        
        // Print some basic stats
        let count = store.triple_count()?;
        println!("Loaded {} triples", count);
        
        // Show a few sample subjects
        let subjects = store.get_subjects(Some(5))?;
        if !subjects.is_empty() {
            println!("Sample subjects:");
            for subject in subjects {
                println!("  {}", subject);
            }
        }
    }
    
    let addr: SocketAddr = if args.host == "localhost" {
        format!("127.0.0.1:{}", args.port).parse()?
    } else {
        format!("{}:{}", args.host, args.port).parse()?
    };
    let store_arc = Arc::new(store);
    
    println!("🚀 Starting OxiRS GraphQL server on http://{}", addr);
    
    // Use the core GraphQL implementation
    println!("🔧 Using core GraphQL implementation with optimization features");
    
    let config = GraphQLConfig {
        enable_playground: args.playground,
        enable_introspection: args.introspection,
        ..Default::default()
    };
    
    let server = GraphQLServer::new(store_arc)
        .with_config(config);
    
    if args.playground {
        println!("📊 GraphQL Playground available at http://{}/", addr);
    }
    println!("🔍 GraphQL endpoint: http://{}/graphql", addr);
    
    server.start(&addr.to_string()).await?;
    
    Ok(())
}