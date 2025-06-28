//! Tests for zero-copy serialization

use bytes::{Bytes, BytesMut};
use oxirs_core::io::{
    MmapReader, MmapWriter, ZeroCopyDeserialize, ZeroCopySerialize, ZeroCopyStr, ZeroCopyTerm,
    ZeroCopyTriple,
};
use oxirs_core::model::*;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_basic_serialization() {
    // Test NamedNode
    let node = Term::NamedNode(NamedNode::new("http://example.org/test").unwrap());
    let mut buf = Vec::new();
    node.serialize_to(&mut buf).unwrap();

    let (deserialized, rest) = ZeroCopyTerm::deserialize_from(&buf).unwrap();
    assert!(rest.is_empty());

    match deserialized {
        ZeroCopyTerm::NamedNode(iri) => {
            assert_eq!(iri.as_str(), "http://example.org/test");
        }
        _ => panic!("Expected NamedNode"),
    }

    // Test BlankNode
    let blank = Term::BlankNode(BlankNode::new("b1").unwrap());
    let mut buf = Vec::new();
    blank.serialize_to(&mut buf).unwrap();

    let (deserialized, rest) = ZeroCopyTerm::deserialize_from(&buf).unwrap();
    assert!(rest.is_empty());

    match deserialized {
        ZeroCopyTerm::BlankNode(b) => {
            assert_eq!(b.id(), "b1");
        }
        _ => panic!("Expected BlankNode"),
    }

    // Test simple Literal
    let literal = Term::Literal(Literal::new("Hello, World!"));
    let mut buf = Vec::new();
    literal.serialize_to(&mut buf).unwrap();

    let (deserialized, rest) = ZeroCopyTerm::deserialize_from(&buf).unwrap();
    assert!(rest.is_empty());

    match deserialized {
        ZeroCopyTerm::Literal(lit) => {
            assert_eq!(lit.value(), "Hello, World!");
            assert_eq!(lit.language(), None);
            assert!(lit.datatype().is_none());
        }
        _ => panic!("Expected Literal"),
    }
}

#[test]
fn test_language_tagged_literal() {
    let literal = Term::Literal(Literal::new_language_tagged_literal("Bonjour", "fr").unwrap());
    let mut buf = Vec::new();
    literal.serialize_to(&mut buf).unwrap();

    let (deserialized, rest) = ZeroCopyTerm::deserialize_from(&buf).unwrap();
    assert!(rest.is_empty());

    match deserialized {
        ZeroCopyTerm::Literal(lit) => {
            assert_eq!(lit.value(), "Bonjour");
            assert_eq!(lit.language(), Some("fr"));
            assert!(lit.datatype().is_none());
        }
        _ => panic!("Expected Literal"),
    }
}

#[test]
fn test_typed_literal() {
    let literal = Term::Literal(Literal::new_typed(
        "42",
        NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
    ));
    let mut buf = Vec::new();
    literal.serialize_to(&mut buf).unwrap();

    let (deserialized, rest) = ZeroCopyTerm::deserialize_from(&buf).unwrap();
    assert!(rest.is_empty());

    match deserialized {
        ZeroCopyTerm::Literal(lit) => {
            assert_eq!(lit.value(), "42");
            assert_eq!(lit.language(), None);
            assert_eq!(
                lit.datatype().unwrap().as_str(),
                "http://www.w3.org/2001/XMLSchema#integer"
            );
        }
        _ => panic!("Expected Literal"),
    }
}

#[test]
fn test_triple_serialization() {
    let triple = Triple::new(
        NamedNode::new("http://example.org/alice").unwrap(),
        NamedNode::new("http://example.org/knows").unwrap(),
        NamedNode::new("http://example.org/bob").unwrap(),
    );

    let mut buf = Vec::new();
    triple.serialize_to(&mut buf).unwrap();

    let (deserialized, rest) = ZeroCopyTriple::deserialize_from(&buf).unwrap();
    assert!(rest.is_empty());

    match &deserialized.subject {
        ZeroCopyTerm::NamedNode(iri) => assert_eq!(iri.as_str(), "http://example.org/alice"),
        _ => panic!("Expected NamedNode subject"),
    }

    assert_eq!(deserialized.predicate.as_str(), "http://example.org/knows");

    match &deserialized.object {
        ZeroCopyTerm::NamedNode(iri) => assert_eq!(iri.as_str(), "http://example.org/bob"),
        _ => panic!("Expected NamedNode object"),
    }
}

#[test]
fn test_bytes_serialization() {
    let triple = Triple::new(
        NamedNode::new("http://example.org/subject").unwrap(),
        NamedNode::new("http://example.org/predicate").unwrap(),
        Literal::new("Object Value"),
    );

    let mut buf = BytesMut::with_capacity(triple.serialized_size());
    triple.serialize_to_bytes(&mut buf);

    let mut bytes = buf.freeze();
    let deserialized = ZeroCopyTriple::deserialize_from_bytes(&mut bytes).unwrap();

    match &deserialized.subject {
        ZeroCopyTerm::NamedNode(iri) => assert_eq!(iri.as_str(), "http://example.org/subject"),
        _ => panic!("Expected NamedNode subject"),
    }

    match &deserialized.object {
        ZeroCopyTerm::Literal(lit) => assert_eq!(lit.value(), "Object Value"),
        _ => panic!("Expected Literal object"),
    }
}

#[test]
fn test_multiple_triples() {
    let triples = vec![
        Triple::new(
            NamedNode::new("http://example.org/s1").unwrap(),
            NamedNode::new("http://example.org/p1").unwrap(),
            Literal::new("Object 1"),
        ),
        Triple::new(
            NamedNode::new("http://example.org/s2").unwrap(),
            NamedNode::new("http://example.org/p2").unwrap(),
            Literal::new("Object 2"),
        ),
        Triple::new(
            NamedNode::new("http://example.org/s3").unwrap(),
            NamedNode::new("http://example.org/p3").unwrap(),
            Literal::new("Object 3"),
        ),
    ];

    let mut buf = Vec::new();
    for triple in &triples {
        triple.serialize_to(&mut buf).unwrap();
    }

    // Deserialize all triples
    let mut data = &buf[..];
    let mut deserialized = Vec::new();

    while !data.is_empty() {
        let (triple, rest) = ZeroCopyTriple::deserialize_from(data).unwrap();
        deserialized.push(triple);
        data = rest;
    }

    assert_eq!(deserialized.len(), 3);

    for (i, triple) in deserialized.iter().enumerate() {
        match &triple.subject {
            ZeroCopyTerm::NamedNode(iri) => {
                assert_eq!(iri.as_str(), format!("http://example.org/s{}", i + 1));
            }
            _ => panic!("Expected NamedNode subject"),
        }
    }
}

#[test]
fn test_mmap_writer_reader() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test.rdf.bin");

    // Write triples
    {
        let mut writer = MmapWriter::new(&file_path, 1024 * 1024).unwrap(); // 1MB

        for i in 0..100 {
            let triple = Triple::new(
                NamedNode::new(&format!("http://example.org/subject{}", i)).unwrap(),
                NamedNode::new("http://example.org/predicate").unwrap(),
                Literal::new(&format!("Object {}", i)),
            );
            writer.write_triple(&triple).unwrap();
        }

        writer.finalize().unwrap();
    }

    // Read triples
    {
        let reader = MmapReader::new(&file_path).unwrap();
        let mut count = 0;

        for (i, result) in reader.iter_triples().enumerate() {
            let triple = result.unwrap();

            match &triple.subject {
                ZeroCopyTerm::NamedNode(iri) => {
                    assert_eq!(iri.as_str(), format!("http://example.org/subject{}", i));
                }
                _ => panic!("Expected NamedNode subject"),
            }

            match &triple.object {
                ZeroCopyTerm::Literal(lit) => {
                    assert_eq!(lit.value(), format!("Object {}", i));
                }
                _ => panic!("Expected Literal object"),
            }

            count += 1;
        }

        assert_eq!(count, 100);
    }
}

#[test]
fn test_serialized_size() {
    let node = Term::NamedNode(NamedNode::new("http://example.org/test").unwrap());
    let expected_size = 1 + 4 + "http://example.org/test".len();
    assert_eq!(node.serialized_size(), expected_size);

    let literal = Term::Literal(Literal::new("Hello"));
    let expected_size = 1 + 4 + "Hello".len();
    assert_eq!(literal.serialized_size(), expected_size);

    let lang_literal =
        Term::Literal(Literal::new_language_tagged_literal("Bonjour", "fr").unwrap());
    let expected_size = 1 + 4 + "Bonjour".len() + 4 + "fr".len();
    assert_eq!(lang_literal.serialized_size(), expected_size);
}

#[test]
fn test_zero_copy_str() {
    let borrowed = ZeroCopyStr::new_borrowed("Hello");
    assert_eq!(borrowed.as_str(), "Hello");

    let owned = ZeroCopyStr::new_owned("World".to_string());
    assert_eq!(owned.as_str(), "World");

    let converted = borrowed.into_owned();
    assert_eq!(converted, "Hello");
}

#[test]
fn test_large_dataset() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("large.rdf.bin");

    // Write many triples
    {
        let mut writer = MmapWriter::new(&file_path, 10 * 1024 * 1024).unwrap(); // 10MB

        for i in 0..10000 {
            let triple = Triple::new(
                NamedNode::new(&format!("http://example.org/entity/{}", i)).unwrap(),
                NamedNode::new("http://example.org/type").unwrap(),
                NamedNode::new(&format!("http://example.org/class/{}", i % 100)).unwrap(),
            );
            writer.write_triple(&triple).unwrap();
        }

        writer.finalize().unwrap();
    }

    // Verify file size is reasonable
    let metadata = fs::metadata(&file_path).unwrap();
    assert!(metadata.len() < 1024 * 1024); // Should be well under 1MB for 10k triples

    // Read and verify
    {
        let reader = MmapReader::new(&file_path).unwrap();
        let count = reader.iter_triples().count();
        assert_eq!(count, 10000);
    }
}

#[test]
fn test_utf8_strings() {
    let strings = vec![
        "Hello, World!",
        "Здравствуй, мир!",
        "你好，世界！",
        "مرحبا بالعالم!",
        "🌍🌎🌏",
    ];

    for s in strings {
        let literal = Term::Literal(Literal::new(s));
        let mut buf = Vec::new();
        literal.serialize_to(&mut buf).unwrap();

        let (deserialized, _) = ZeroCopyTerm::deserialize_from(&buf).unwrap();
        match deserialized {
            ZeroCopyTerm::Literal(lit) => assert_eq!(lit.value(), s),
            _ => panic!("Expected Literal"),
        }
    }
}
