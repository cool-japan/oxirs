//! Server demo for OxiRS

use oxirs_fuseki::Server;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();

    println!("Starting OxiRS Fuseki server demo...");
    
    // Create and configure server
    let server = Server::builder()
        .host("localhost")
        .port(3030)
        .build()
        .await?;
    
    println!("Server configured, starting on http://localhost:3030");
    println!("Press Ctrl+C to stop the server");
    
    // Run the server
    server.run().await?;
    
    Ok(())
}