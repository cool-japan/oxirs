//! Example demonstrating zero-copy serialization for efficient RDF processing

use bytes::Buf;
use bytes::BytesMut;
use oxirs_core::io::{
    MmapReader, MmapWriter, ZeroCopyDeserialize, ZeroCopySerialize, ZeroCopyTerm, ZeroCopyTriple,
};
use oxirs_core::model::*;
use std::fs;
use std::time::Instant;
use tempfile::TempDir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Zero-Copy Serialization Example ===\n");

    // Example 1: Basic serialization/deserialization
    println!("Example 1: Basic Serialization");
    basic_example()?;

    // Example 2: Memory-mapped files
    println!("\nExample 2: Memory-Mapped Files");
    mmap_example()?;

    // Example 3: Streaming with bytes
    println!("\nExample 3: Streaming with Bytes");
    bytes_example()?;

    // Example 4: Performance comparison
    println!("\nExample 4: Performance Comparison");
    performance_example()?;

    // Example 5: Large dataset processing
    println!("\nExample 5: Large Dataset Processing");
    large_dataset_example()?;

    Ok(())
}

fn basic_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create some RDF data
    let triple = Triple::new(
        NamedNode::new("http://example.org/alice")?,
        NamedNode::new("http://example.org/knows")?,
        NamedNode::new("http://example.org/bob")?,
    );

    // Serialize to bytes
    let mut buffer = Vec::new();
    triple.serialize_to(&mut buffer)?;

    println!("Serialized triple size: {} bytes", buffer.len());
    println!("Calculated size: {} bytes", triple.serialized_size());

    // Deserialize with zero-copy
    let (deserialized, remaining) = ZeroCopyTriple::deserialize_from(&buffer)?;
    assert!(remaining.is_empty());

    // Access data without allocation
    if let ZeroCopyTerm::NamedNode(iri) = &deserialized.subject {
        println!("Subject: {}", iri.as_str());
    }
    println!("Predicate: {}", deserialized.predicate.as_str());
    if let ZeroCopyTerm::NamedNode(iri) = &deserialized.object {
        println!("Object: {}", iri.as_str());
    }

    Ok(())
}

fn mmap_example() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join("data.rdf.bin");

    // Write data using memory-mapped file
    {
        let mut writer = MmapWriter::new(&file_path, 1024 * 1024)?; // 1MB capacity

        println!("Writing 1000 triples...");
        let start = Instant::now();

        for i in 0..1000 {
            let triple = Triple::new(
                NamedNode::new(format!("http://example.org/person/{i}"))?,
                NamedNode::new("http://example.org/name")?,
                Literal::new(format!("Person {i}")),
            );
            writer.write_triple(&triple)?;
        }

        writer.finalize()?;
        println!("Write time: {:?}", start.elapsed());
    }

    // Read data using memory-mapped file
    {
        let reader = MmapReader::new(&file_path)?;

        println!("Reading triples with zero-copy...");
        let start = Instant::now();

        let mut count = 0;
        for result in reader.iter_triples() {
            let triple = result?;
            // Data is accessed directly from mapped memory
            count += 1;

            if count <= 3 {
                if let ZeroCopyTerm::NamedNode(iri) = &triple.subject {
                    println!("  Subject: {}", iri.as_str())
                }
            }
        }

        println!("Read {} triples in {:?}", count, start.elapsed());

        let file_size = fs::metadata(&file_path)?.len();
        println!("File size: {} KB", file_size / 1024);
    }

    Ok(())
}

fn bytes_example() -> Result<(), Box<dyn std::error::Error>> {
    // Using bytes crate for efficient buffer management
    let mut buf = BytesMut::with_capacity(1024);

    // Serialize multiple terms
    let terms = vec![
        Term::NamedNode(NamedNode::new("http://example.org/resource")?),
        Term::Literal(Literal::new("Simple literal")),
        Term::Literal(Literal::new_language_tagged_literal("Bonjour", "fr")?),
        Term::Literal(Literal::new_typed(
            "42",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?,
        )),
        Term::BlankNode(BlankNode::new("b1")?),
    ];

    println!("Serializing terms:");
    for term in &terms {
        term.serialize_to_bytes(&mut buf);
        println!("  {} -> {} bytes", term, term.serialized_size());
    }

    // Freeze buffer for reading
    let mut bytes = buf.freeze();
    println!("\nTotal buffer size: {} bytes", bytes.len());

    // Deserialize terms
    println!("\nDeserializing terms:");
    let mut count = 0;
    while bytes.remaining() > 0 {
        let term = ZeroCopyTerm::deserialize_from_bytes(&mut bytes)?;
        match term {
            ZeroCopyTerm::NamedNode(iri) => {
                println!("  NamedNode: {}", iri.as_str());
            }
            ZeroCopyTerm::Literal(lit) => {
                print!("  Literal: \"{}\"", lit.value());
                if let Some(lang) = lit.language() {
                    print!("@{lang}");
                } else if let Some(dt) = lit.datatype() {
                    print!("^^<{}>", dt.as_str());
                }
                println!();
            }
            ZeroCopyTerm::BlankNode(b) => {
                println!("  BlankNode: _:{}", b.id());
            }
            ZeroCopyTerm::Variable(v) => {
                println!("  Variable: ?{}", v.as_str());
            }
            ZeroCopyTerm::QuotedTriple(_qt) => {
                println!("  QuotedTriple: << <subject> <predicate> <object> >>");
            }
        }
        count += 1;
    }
    println!("Deserialized {count} terms");

    Ok(())
}

fn performance_example() -> Result<(), Box<dyn std::error::Error>> {
    let iterations = 10000;
    let test_triple = Triple::new(
        NamedNode::new("http://example.org/subject")?,
        NamedNode::new("http://example.org/predicate")?,
        Literal::new("This is a test object value"),
    );

    // Measure serialization performance
    let mut total_size = 0;
    let start = Instant::now();

    for _ in 0..iterations {
        let mut buf = Vec::with_capacity(test_triple.serialized_size());
        test_triple.serialize_to(&mut buf)?;
        total_size += buf.len();
    }

    let serialize_time = start.elapsed();
    let serialize_rate = iterations as f64 / serialize_time.as_secs_f64();

    println!("Serialization performance:");
    println!("  Iterations: {iterations}");
    println!("  Time: {serialize_time:?}");
    println!("  Rate: {serialize_rate:.0} triples/second");
    println!(
        "  Throughput: {:.2} MB/second",
        (total_size as f64 / 1024.0 / 1024.0) / serialize_time.as_secs_f64()
    );

    // Prepare data for deserialization test
    let mut buf = Vec::new();
    test_triple.serialize_to(&mut buf)?;

    // Measure deserialization performance
    let start = Instant::now();

    for _ in 0..iterations {
        let (_, _) = ZeroCopyTriple::deserialize_from(&buf)?;
    }

    let deserialize_time = start.elapsed();
    let deserialize_rate = iterations as f64 / deserialize_time.as_secs_f64();

    println!("\nDeserialization performance (zero-copy):");
    println!("  Iterations: {iterations}");
    println!("  Time: {deserialize_time:?}");
    println!("  Rate: {:.0} triples/second", deserialize_rate);

    Ok(())
}

fn large_dataset_example() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join("large_dataset.rdf.bin");

    let num_triples = 100_000;

    // Generate and write large dataset
    {
        println!("Generating {} triples...", num_triples);
        let start = Instant::now();

        let mut writer = MmapWriter::new(&file_path, 50 * 1024 * 1024)?; // 50MB

        for i in 0..num_triples {
            let triple = Triple::new(
                NamedNode::new(format!("http://example.org/entity/{}", i))?,
                NamedNode::new(format!("http://example.org/property/{}", i % 100))?,
                match i % 3 {
                    0 => Term::Literal(Literal::new(format!("Value {}", i))),
                    1 => Term::NamedNode(NamedNode::new(format!("http://example.org/ref/{}", i))?),
                    _ => Term::BlankNode(BlankNode::new(format!("b{}", i))?),
                },
            );
            writer.write_triple(&triple)?;
        }

        writer.finalize()?;
        println!("Generation time: {:?}", start.elapsed());
    }

    // Process dataset with zero-copy
    {
        let reader = MmapReader::new(&file_path)?;
        let file_size = fs::metadata(&file_path)?.len();

        println!("\nProcessing dataset:");
        println!("  File size: {:.2} MB", file_size as f64 / 1024.0 / 1024.0);

        let start = Instant::now();

        // Count different types of objects
        let mut literal_count = 0;
        let mut named_node_count = 0;
        let mut blank_node_count = 0;

        for result in reader.iter_triples() {
            let triple = result?;
            match &triple.object {
                ZeroCopyTerm::Literal(_) => literal_count += 1,
                ZeroCopyTerm::NamedNode(_) => named_node_count += 1,
                ZeroCopyTerm::BlankNode(_) => blank_node_count += 1,
                _ => {}
            }
        }

        let process_time = start.elapsed();
        let process_rate = num_triples as f64 / process_time.as_secs_f64();

        println!("  Processing time: {process_time:?}");
        println!("  Processing rate: {process_rate:.0} triples/second");
        println!("  Object types:");
        println!("    Literals: {literal_count}");
        println!("    Named nodes: {named_node_count}");
        println!("    Blank nodes: {blank_node_count}");
    }

    Ok(())
}
