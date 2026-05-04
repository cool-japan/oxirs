//! Integration tests for the ESMF SDK 2.x parity matrix.

use oxirs_samm::parity::{generate_report, load_catalog, ImplStatus};

#[test]
fn test_catalog_parses() {
    let matrix = load_catalog().expect("catalog should parse");
    assert!(
        !matrix.is_empty(),
        "matrix should have at least one category"
    );
}

#[test]
fn test_all_categories_have_entries() {
    let matrix = load_catalog().expect("catalog should parse");
    let total: usize = matrix.values().map(|v| v.len()).sum();
    assert!(
        total >= 15,
        "catalog should have at least 15 entries, got {total}"
    );
}

#[test]
fn test_status_enum_coverage() {
    let matrix = load_catalog().expect("catalog should parse");
    let has_implemented = matrix
        .values()
        .flatten()
        .any(|e| e.status == ImplStatus::Implemented);
    let has_missing = matrix
        .values()
        .flatten()
        .any(|e| e.status == ImplStatus::Missing);
    assert!(
        has_implemented,
        "should have at least one implemented entry"
    );
    assert!(has_missing, "should have at least one missing entry");
}

#[test]
fn test_report_contains_all_categories() {
    let matrix = load_catalog().expect("catalog should parse");
    let report = generate_report(&matrix);
    assert!(report.contains("Aspect"), "report should mention Aspect");
    assert!(
        report.contains("Validation") || report.contains("validation"),
        "report should mention Validation"
    );
}

#[test]
fn test_implemented_entries_have_oxirs_module() {
    let matrix = load_catalog().expect("catalog should parse");
    for entries in matrix.values() {
        for entry in entries {
            if entry.status == ImplStatus::Implemented {
                assert!(
                    entry.oxirs_module.is_some(),
                    "Implemented entry '{}' must reference an oxirs_module",
                    entry.name
                );
            }
        }
    }
}

#[test]
fn test_report_summary_table_present() {
    let matrix = load_catalog().expect("catalog should parse");
    let report = generate_report(&matrix);
    assert!(
        report.contains("## Summary"),
        "Summary section must be present"
    );
    assert!(
        report.contains("## Detailed Parity Matrix"),
        "Detailed section must be present"
    );
    // Summary table header row must appear
    assert!(
        report.contains("| Category |"),
        "Summary table header must appear"
    );
}

#[test]
fn test_esmf_references_are_non_empty() {
    let matrix = load_catalog().expect("catalog should parse");
    for entries in matrix.values() {
        for entry in entries {
            assert!(
                !entry.esmf_reference.is_empty(),
                "entry '{}' must have a non-empty esmf_reference",
                entry.name
            );
        }
    }
}

#[test]
fn test_partial_entries_may_lack_module() {
    // Partial entries are allowed to have no oxirs_module — only Implemented must have one.
    // This test just ensures we don't accidentally enforce a stricter rule.
    let matrix = load_catalog().expect("catalog should parse");
    let partial_without_module = matrix
        .values()
        .flatten()
        .filter(|e| e.status == ImplStatus::Partial && e.oxirs_module.is_none())
        .count();
    // Simply assert the test runs; partial entries may or may not have a module.
    let _ = partial_without_module;
}
