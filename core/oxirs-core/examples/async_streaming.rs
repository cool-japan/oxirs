//! Example demonstrating async streaming RDF parsing and serialization

#[cfg(not(feature = "async-tokio"))]
fn main() {
    eprintln!("This example requires the 'async-tokio' feature. Run with: cargo run --example async_streaming --features async-tokio");
}

#[cfg(feature = "async-tokio")]
use oxirs_core::{
    io::{
        AsyncRdfParser, AsyncRdfSerializer, AsyncStreamingConfig, AsyncStreamingParser,
        AsyncStreamingSerializer, ProgressCallback, StreamingProgress,
    },
    model::*,
    parser::RdfFormat,
};
#[cfg(feature = "async-tokio")]
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
#[cfg(feature = "async-tokio")]
use tokio::fs::File;
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[cfg(feature = "async-tokio")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Parse N-Triples with progress reporting
    println!("Example 1: Parsing N-Triples with progress reporting");
    parse_with_progress().await?;

    // Example 2: Parse with cancellation
    println!("\nExample 2: Parsing with cancellation");
    parse_with_cancellation().await?;

    // Example 3: Serialize with streaming
    println!("\nExample 3: Streaming serialization");
    serialize_with_streaming().await?;

    // Example 4: Parse large file with custom configuration
    println!("\nExample 4: Parse with custom configuration");
    parse_with_custom_config().await?;

    Ok(())
}

#[cfg(feature = "async-tokio")]
async fn parse_with_progress() -> Result<(), Box<dyn std::error::Error>> {
    let ntriples_data = r#"<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
<http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" .
<http://example.org/charlie> <http://xmlns.com/foaf/0.1/name> "Charlie" .
"#;

    let parser = AsyncStreamingParser::new(RdfFormat::NTriples);
    let reader = std::io::Cursor::new(ntriples_data.as_bytes());

    let progress_callback: ProgressCallback = Box::new(|progress: &StreamingProgress| {
        println!(
            "Progress: {} bytes, {} items processed",
            progress.bytes_processed, progress.items_processed
        );
        if let Some(rate) = progress.items_per_second {
            println!("Processing rate: {:.2} items/second", rate);
        }
    });

    let quads = parser
        .parse_async(
            reader,
            AsyncStreamingConfig::default(),
            Some(progress_callback),
            None,
        )
        .await?;

    println!("Parsed {} quads", quads.len());
    for quad in quads.iter().take(3) {
        println!("  {}", quad);
    }

    Ok(())
}

#[cfg(feature = "async-tokio")]
async fn parse_with_cancellation() -> Result<(), Box<dyn std::error::Error>> {
    // Create a large dataset
    let mut ntriples_data = String::new();
    for i in 0..100 {
        ntriples_data.push_str(&format!(
            "<http://example.org/item{}> <http://example.org/value> \"{}\" .\n",
            i, i
        ));
    }

    let parser = AsyncStreamingParser::new(RdfFormat::NTriples);
    let reader = std::io::Cursor::new(ntriples_data.as_bytes());

    let cancel_token = Arc::new(AtomicBool::new(false));
    let cancel_token_clone = cancel_token.clone();

    // Cancel after 50ms
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        cancel_token_clone.store(true, Ordering::Relaxed);
        println!("Cancellation requested!");
    });

    let processed = Arc::new(AtomicUsize::new(0));
    let processed_clone = processed.clone();
    let result = parser
        .parse_with_handler_async(
            reader,
            |_quad| {
                let processed = processed_clone.clone();
                async move {
                    processed.fetch_add(1, Ordering::Relaxed);
                    // Simulate some processing time
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                    Ok(())
                }
            },
            AsyncStreamingConfig::default(),
            None,
            Some(cancel_token),
        )
        .await;

    match result {
        Ok(()) => println!(
            "Parsing completed. Processed {} items",
            processed.load(Ordering::Relaxed)
        ),
        Err(e) => println!(
            "Parsing cancelled or failed: {}. Processed {} items",
            e,
            processed.load(Ordering::Relaxed)
        ),
    }

    Ok(())
}

#[cfg(feature = "async-tokio")]
async fn serialize_with_streaming() -> Result<(), Box<dyn std::error::Error>> {
    // Create some test data
    let mut quads = Vec::new();
    for i in 0..10 {
        let subject = NamedNode::new(&format!("http://example.org/item{}", i))?;
        let predicate = NamedNode::new("http://example.org/value")?;
        let object = Literal::new(format!("Value {}", i));
        let triple = Triple::new(subject, predicate, object);
        quads.push(Quad::from_triple(triple));
    }

    let serializer = AsyncStreamingSerializer::new(RdfFormat::NTriples);
    let mut output = Vec::new();

    let progress_callback: ProgressCallback = Box::new(|progress: &StreamingProgress| {
        println!(
            "Serialization progress: {} items, {} bytes written",
            progress.items_processed, progress.bytes_processed
        );
    });

    serializer
        .serialize_quads_async(
            &mut output,
            quads.into_iter(),
            AsyncStreamingConfig::default(),
            Some(progress_callback),
            None,
        )
        .await?;

    println!("\nSerialized output ({} bytes):", output.len());
    println!("{}", String::from_utf8_lossy(&output));

    Ok(())
}

#[cfg(feature = "async-tokio")]
async fn parse_with_custom_config() -> Result<(), Box<dyn std::error::Error>> {
    let invalid_ntriples = r#"<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
INVALID LINE HERE
<http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" .
ANOTHER INVALID LINE
<http://example.org/charlie> <http://xmlns.com/foaf/0.1/name> "Charlie" .
"#;

    let parser = AsyncStreamingParser::new(RdfFormat::NTriples);
    let reader = std::io::Cursor::new(invalid_ntriples.as_bytes());

    // Configure to ignore errors
    let mut config = AsyncStreamingConfig::default();
    config.ignore_errors = true;
    config.chunk_size = 1024; // Small chunks for demonstration
    config.buffer_size = 4096;

    let progress_callback: ProgressCallback = Box::new(|progress: &StreamingProgress| {
        if progress.errors_encountered > 0 {
            println!(
                "Progress: {} items parsed, {} errors encountered",
                progress.items_processed, progress.errors_encountered
            );
        }
    });

    let quads = parser
        .parse_async(reader, config, Some(progress_callback), None)
        .await?;

    println!("\nParsed {} valid quads despite errors:", quads.len());
    for quad in &quads {
        println!("  {}", quad);
    }

    Ok(())
}
