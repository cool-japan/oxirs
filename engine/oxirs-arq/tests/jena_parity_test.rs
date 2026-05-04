//! Integration tests for the Apache Jena feature parity matrix.
//!
//! These tests verify that the embedded catalog parses correctly and that
//! the expected gaps (JenaText, Assembler, Jena Rule Language) are catalogued.

use oxirs_arq::jena_parity::{generate_jena_report, load_catalog, JenaCategory, JenaStatus};

/// The catalog must parse without error.
#[test]
fn test_load_catalog_succeeds() {
    let matrix = load_catalog().expect("embedded jena_catalog.toml should parse without error");
    assert!(!matrix.is_empty(), "catalog must not be empty");
}

/// The catalog must contain entries.
#[test]
fn test_catalog_not_empty() {
    let matrix = load_catalog().expect("catalog");
    let total: usize = matrix.values().map(|v| v.len()).sum();
    assert!(
        total >= 40,
        "catalog must contain at least 40 entries; got {total}"
    );
}

/// Every expected non-Other category must be represented.
#[test]
fn test_all_categories_represented() {
    let matrix = load_catalog().expect("catalog");
    let required = [
        JenaCategory::SparqlEngine,
        JenaCategory::RdfFormats,
        JenaCategory::StorageBackends,
        JenaCategory::Inference,
        JenaCategory::Validation,
        JenaCategory::Spatial,
        JenaCategory::HttpServer,
        JenaCategory::TextSearch,
        JenaCategory::Assembler,
        JenaCategory::GraphApi,
        JenaCategory::Security,
        JenaCategory::Tooling,
    ];
    for cat in &required {
        assert!(matrix.contains_key(cat), "missing category: {cat:?}");
        assert!(!matrix[cat].is_empty(), "category {cat:?} has no entries");
    }
}

/// JenaText gap must be catalogued as Missing.
#[test]
fn test_text_search_entries_are_missing() {
    let matrix = load_catalog().expect("catalog");
    let entries = matrix
        .get(&JenaCategory::TextSearch)
        .expect("TextSearch category must be present");
    assert!(
        !entries.is_empty(),
        "TextSearch category must have at least one entry"
    );
    // All TextSearch entries should be Missing (no OxiRS implementation yet)
    for entry in entries {
        assert_eq!(
            entry.status,
            JenaStatus::Missing,
            "TextSearch entry '{}' should be Missing, got {:?}",
            entry.name,
            entry.status
        );
    }
}

/// Assembler gap must be catalogued as Missing.
#[test]
fn test_assembler_entries_are_missing() {
    let matrix = load_catalog().expect("catalog");
    let entries = matrix
        .get(&JenaCategory::Assembler)
        .expect("Assembler category must be present");
    assert!(
        !entries.is_empty(),
        "Assembler category must have at least one entry"
    );
    for entry in entries {
        assert_eq!(
            entry.status,
            JenaStatus::Missing,
            "Assembler entry '{}' should be Missing, got {:?}",
            entry.name,
            entry.status
        );
    }
}

/// Jena Rule Language (JRL) gap must be catalogued as Missing.
#[test]
fn test_jena_rule_language_is_missing() {
    let matrix = load_catalog().expect("catalog");
    let inference = matrix
        .get(&JenaCategory::Inference)
        .expect("Inference category must be present");
    let jrl = inference
        .iter()
        .find(|e| e.name.contains("Jena Rule Language") || e.name.contains("JRL"))
        .expect("Jena Rule Language entry must exist in Inference category");
    assert_eq!(
        jrl.status,
        JenaStatus::Missing,
        "JRL entry should be Missing; got {:?}",
        jrl.status
    );
}

/// The generated report must contain headings for all categories.
#[test]
fn test_report_contains_all_categories() {
    let matrix = load_catalog().expect("catalog");
    let report = generate_jena_report(&matrix);

    let expected_headings = [
        "SPARQL Engine",
        "RDF Formats",
        "Storage Backends",
        "Inference",
        "Validation",
        "Spatial",
        "HTTP Server",
        "Full-Text Search",
        "Dataset Assembler",
        "Graph API",
        "Security",
        "Tooling",
    ];
    for heading in &expected_headings {
        assert!(
            report.contains(heading),
            "report missing heading: {heading}"
        );
    }
}

/// The report must be substantially long (more than 2 KB).
#[test]
fn test_report_length_nontrivial() {
    let matrix = load_catalog().expect("catalog");
    let report = generate_jena_report(&matrix);
    assert!(
        report.len() > 2048,
        "report is suspiciously short: {} bytes",
        report.len()
    );
}

/// SDB storage backend must be OutOfScope (not Missing).
#[test]
fn test_sdb_is_out_of_scope() {
    let matrix = load_catalog().expect("catalog");
    let backends = matrix
        .get(&JenaCategory::StorageBackends)
        .expect("StorageBackends must be present");
    let sdb = backends
        .iter()
        .find(|e| e.name.contains("SDB") || e.jena_component.contains("jena-sdb"))
        .expect("SDB entry must exist in StorageBackends");
    assert_eq!(
        sdb.status,
        JenaStatus::OutOfScope,
        "SDB should be OutOfScope per Pure Rust Policy; got {:?}",
        sdb.status
    );
}

/// Every implemented entry must have an oxirs_module field.
#[test]
fn test_implemented_entries_have_module() {
    let matrix = load_catalog().expect("catalog");
    for (cat, entries) in &matrix {
        for entry in entries {
            if entry.status == JenaStatus::Implemented {
                assert!(
                    entry.oxirs_module.is_some(),
                    "category {cat:?} entry '{}' is Implemented but has no oxirs_module",
                    entry.name
                );
            }
        }
    }
}
