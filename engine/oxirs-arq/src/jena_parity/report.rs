//! Markdown report generator for the Apache Jena parity matrix.
//!
//! Call [`generate_jena_report`] to produce a complete markdown document
//! summarising the implementation status across all Jena feature categories.

use crate::jena_parity::matrix::{JenaCategory, JenaEntry, JenaParityMatrix, JenaStatus};

/// Human-readable heading for each [`JenaCategory`].
fn category_heading(cat: &JenaCategory) -> &'static str {
    match cat {
        JenaCategory::SparqlEngine => "SPARQL Engine (ARQ)",
        JenaCategory::RdfFormats => "RDF Formats",
        JenaCategory::StorageBackends => "Storage Backends",
        JenaCategory::Inference => "Inference & Reasoning",
        JenaCategory::Validation => "Validation (SHACL)",
        JenaCategory::Spatial => "Spatial / GeoSPARQL",
        JenaCategory::HttpServer => "HTTP Server (Fuseki)",
        JenaCategory::TextSearch => "Full-Text Search (JenaText)",
        JenaCategory::Assembler => "Dataset Assembler",
        JenaCategory::GraphApi => "Graph API",
        JenaCategory::Security => "Security",
        JenaCategory::Tooling => "Tooling",
        JenaCategory::Other => "Other",
    }
}

/// Stable display order for categories in the report.
fn category_order() -> Vec<JenaCategory> {
    vec![
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
    ]
}

/// Badge string for a status value.
fn status_badge(status: &JenaStatus) -> &'static str {
    match status {
        JenaStatus::Implemented => "Implemented",
        JenaStatus::Partial => "Partial",
        JenaStatus::Missing => "Missing",
        JenaStatus::OutOfScope => "Out of Scope",
    }
}

/// Count features by status across a slice of entries.
///
/// Returns `(implemented, partial, missing, out_of_scope)`.
fn count_by_status(entries: &[JenaEntry]) -> (usize, usize, usize, usize) {
    let implemented = entries
        .iter()
        .filter(|e| e.status == JenaStatus::Implemented)
        .count();
    let partial = entries
        .iter()
        .filter(|e| e.status == JenaStatus::Partial)
        .count();
    let missing = entries
        .iter()
        .filter(|e| e.status == JenaStatus::Missing)
        .count();
    let out_of_scope = entries
        .iter()
        .filter(|e| e.status == JenaStatus::OutOfScope)
        .count();
    (implemented, partial, missing, out_of_scope)
}

/// Generate a full markdown parity report from a [`JenaParityMatrix`].
///
/// The report contains:
/// 1. A summary table with per-category counts.
/// 2. A detailed section for every category with one row per feature.
///
/// The returned `String` is suitable for printing to stdout or writing to a
/// documentation file.
pub fn generate_jena_report(matrix: &JenaParityMatrix) -> String {
    let mut out = String::with_capacity(8192);

    out.push_str("# Apache Jena Feature Parity Report — OxiRS\n\n");
    out.push_str("> **Generated automatically** by `cargo run --bin jena_parity_report`.\n");
    out.push_str("> Do not edit by hand — regenerate after updating `jena_catalog.toml`.\n\n");
    out.push_str("---\n\n");

    // ── Summary table ────────────────────────────────────────────────────
    out.push_str("## Summary\n\n");
    out.push_str("| Category | Implemented | Partial | Missing | Out of Scope | Total |\n");
    out.push_str("|---|---|---|---|---|---|\n");

    let mut grand_impl = 0usize;
    let mut grand_part = 0usize;
    let mut grand_miss = 0usize;
    let mut grand_oos = 0usize;

    for cat in category_order() {
        let heading = category_heading(&cat);
        let entries = match matrix.get(&cat) {
            Some(e) => e.as_slice(),
            None => &[],
        };
        let (imp, part, miss, oos) = count_by_status(entries);
        let total = imp + part + miss + oos;
        grand_impl += imp;
        grand_part += part;
        grand_miss += miss;
        grand_oos += oos;
        out.push_str(&format!(
            "| {heading} | {imp} | {part} | {miss} | {oos} | {total} |\n"
        ));
    }

    // Also collect any Other categories that may exist in the matrix
    if let Some(entries) = matrix.get(&JenaCategory::Other) {
        let heading = category_heading(&JenaCategory::Other);
        let (imp, part, miss, oos) = count_by_status(entries);
        let total = imp + part + miss + oos;
        grand_impl += imp;
        grand_part += part;
        grand_miss += miss;
        grand_oos += oos;
        out.push_str(&format!(
            "| {heading} | {imp} | {part} | {miss} | {oos} | {total} |\n"
        ));
    }

    let grand_total = grand_impl + grand_part + grand_miss + grand_oos;
    out.push_str(&format!(
        "| **Total** | **{grand_impl}** | **{grand_part}** | **{grand_miss}** | **{grand_oos}** | **{grand_total}** |\n"
    ));
    out.push('\n');

    // ── Detailed per-category sections ──────────────────────────────────
    out.push_str("## Detailed Parity Matrix\n\n");

    for cat in category_order() {
        let heading = category_heading(&cat);
        let entries = match matrix.get(&cat) {
            Some(e) => e.as_slice(),
            None => continue,
        };
        if entries.is_empty() {
            continue;
        }

        out.push_str(&format!("### {heading}\n\n"));
        out.push_str(
            "| Feature | Jena Component | Jena Class / API | OxiRS Module | Status | Notes |\n",
        );
        out.push_str("|---|---|---|---|---|---|\n");

        for entry in entries {
            let module = entry
                .oxirs_module
                .as_deref()
                .map(|m| format!("`{m}`"))
                .unwrap_or_else(|| "—".to_string());
            let badge = status_badge(&entry.status);
            let name = entry.name.replace('|', "\\|");
            let component = entry.jena_component.replace('|', "\\|");
            let class_api = entry.jena_class_or_api.replace('|', "\\|");
            let notes = entry.notes.replace('|', "\\|");
            out.push_str(&format!(
                "| {name} | `{component}` | `{class_api}` | {module} | {badge} | {notes} |\n"
            ));
        }
        out.push('\n');
    }

    // Any Other category
    if let Some(entries) = matrix.get(&JenaCategory::Other) {
        let heading = category_heading(&JenaCategory::Other);
        out.push_str(&format!("### {heading}\n\n"));
        out.push_str(
            "| Feature | Jena Component | Jena Class / API | OxiRS Module | Status | Notes |\n",
        );
        out.push_str("|---|---|---|---|---|---|\n");
        for entry in entries {
            let module = entry
                .oxirs_module
                .as_deref()
                .map(|m| format!("`{m}`"))
                .unwrap_or_else(|| "—".to_string());
            let badge = status_badge(&entry.status);
            let name = entry.name.replace('|', "\\|");
            let component = entry.jena_component.replace('|', "\\|");
            let class_api = entry.jena_class_or_api.replace('|', "\\|");
            let notes = entry.notes.replace('|', "\\|");
            out.push_str(&format!(
                "| {name} | `{component}` | `{class_api}` | {module} | {badge} | {notes} |\n"
            ));
        }
        out.push('\n');
    }

    out.push_str("---\n\n");
    out.push_str(
        "*Report generated by `oxirs-arq` — Apache Jena feature parity matrix (2026-05-01).*\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jena_parity::matrix::{JenaCategory, JenaEntry, JenaParityMatrix, JenaStatus};
    use std::collections::HashMap;

    fn sample_matrix() -> JenaParityMatrix {
        let mut m = HashMap::new();
        m.insert(
            JenaCategory::SparqlEngine,
            vec![
                JenaEntry {
                    name: "SPARQL 1.1 SELECT".to_string(),
                    jena_component: "jena-arq".to_string(),
                    jena_class_or_api: "QueryExecutionFactory.create()".to_string(),
                    oxirs_module: Some("oxirs_arq::executor".to_string()),
                    status: JenaStatus::Implemented,
                    notes: "Full SPARQL 1.1 compliance.".to_string(),
                },
                JenaEntry {
                    name: "ARQ Property Functions".to_string(),
                    jena_component: "jena-arq".to_string(),
                    jena_class_or_api: "PropertyFunctionRegistry".to_string(),
                    oxirs_module: None,
                    status: JenaStatus::Partial,
                    notes: "Built-ins implemented; registration API planned.".to_string(),
                },
            ],
        );
        m.insert(
            JenaCategory::TextSearch,
            vec![JenaEntry {
                name: "JenaText SPARQL integration".to_string(),
                jena_component: "jena-text".to_string(),
                jena_class_or_api: "TextDatasetFactory".to_string(),
                oxirs_module: None,
                status: JenaStatus::Missing,
                notes: "text:query property function not yet implemented.".to_string(),
            }],
        );
        m.insert(
            JenaCategory::StorageBackends,
            vec![JenaEntry {
                name: "SDB (SQL-backed store)".to_string(),
                jena_component: "jena-sdb".to_string(),
                jena_class_or_api: "SDBFactory".to_string(),
                oxirs_module: None,
                status: JenaStatus::OutOfScope,
                notes: "Pure Rust Policy: SQL-backed stores not planned.".to_string(),
            }],
        );
        m
    }

    #[test]
    fn test_report_contains_headings() {
        let report = generate_jena_report(&sample_matrix());
        assert!(
            report.contains("Apache Jena Feature Parity Report"),
            "title missing"
        );
        assert!(report.contains("## Summary"), "summary section missing");
        assert!(
            report.contains("## Detailed Parity Matrix"),
            "detail section missing"
        );
    }

    #[test]
    fn test_report_contains_categories() {
        let report = generate_jena_report(&sample_matrix());
        assert!(
            report.contains("SPARQL Engine"),
            "SPARQL Engine section missing"
        );
        assert!(
            report.contains("Full-Text Search"),
            "TextSearch section missing"
        );
    }

    #[test]
    fn test_report_contains_status_badges() {
        let report = generate_jena_report(&sample_matrix());
        assert!(report.contains("Implemented"), "implemented badge missing");
        assert!(report.contains("Missing"), "missing badge missing");
        assert!(report.contains("Partial"), "partial badge missing");
        assert!(
            report.contains("Out of Scope"),
            "out_of_scope badge missing"
        );
    }

    #[test]
    fn test_report_empty_matrix() {
        let report = generate_jena_report(&HashMap::new());
        assert!(
            report.contains("Apache Jena Feature Parity Report"),
            "title should appear even for empty matrix"
        );
    }

    #[test]
    fn test_count_by_status_four_tuple() {
        let entries = sample_matrix()
            .remove(&JenaCategory::SparqlEngine)
            .unwrap_or_default();
        let (imp, part, miss, oos) = count_by_status(&entries);
        assert_eq!(imp, 1);
        assert_eq!(part, 1);
        assert_eq!(miss, 0);
        assert_eq!(oos, 0);
    }

    #[test]
    fn test_grand_total_in_report() {
        let report = generate_jena_report(&sample_matrix());
        assert!(report.contains("**Total**"), "grand total row missing");
    }
}
