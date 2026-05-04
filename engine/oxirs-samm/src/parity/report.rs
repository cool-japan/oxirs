//! Markdown report generator for the ESMF SDK 2.x parity matrix.
//!
//! Call [`generate_report`] to produce a complete markdown document
//! summarising the implementation status across all feature categories.

use crate::parity::matrix::{FeatureCategory, FeatureEntry, ImplStatus, ParityMatrix};

/// Human-readable heading for each [`FeatureCategory`].
fn category_heading(cat: &FeatureCategory) -> &'static str {
    match cat {
        FeatureCategory::AspectModeling => "Aspect Modeling",
        FeatureCategory::Validation => "Validation",
        FeatureCategory::CodeGeneration => "Code Generation",
        FeatureCategory::OpenApiEmission => "OpenAPI Emission",
        FeatureCategory::JsonLdProfiles => "JSON-LD Profiles",
        FeatureCategory::ModelResolution => "Model Resolution",
        FeatureCategory::CommandLineTooling => "Command-Line Tooling",
        FeatureCategory::Other(_) => "Other",
    }
}

/// Stable display order for categories in the report.
fn category_order() -> Vec<FeatureCategory> {
    vec![
        FeatureCategory::AspectModeling,
        FeatureCategory::Validation,
        FeatureCategory::CodeGeneration,
        FeatureCategory::OpenApiEmission,
        FeatureCategory::JsonLdProfiles,
        FeatureCategory::ModelResolution,
        FeatureCategory::CommandLineTooling,
    ]
}

/// Emoji / badge string for a status value.
fn status_badge(status: &ImplStatus) -> &'static str {
    match status {
        ImplStatus::Implemented => "✅ Implemented",
        ImplStatus::Partial => "⚠️ Partial",
        ImplStatus::Missing => "❌ Missing",
    }
}

/// Count features by status across a slice of entries.
fn count_by_status(entries: &[FeatureEntry]) -> (usize, usize, usize) {
    let implemented = entries
        .iter()
        .filter(|e| e.status == ImplStatus::Implemented)
        .count();
    let partial = entries
        .iter()
        .filter(|e| e.status == ImplStatus::Partial)
        .count();
    let missing = entries
        .iter()
        .filter(|e| e.status == ImplStatus::Missing)
        .count();
    (implemented, partial, missing)
}

/// Generate a full markdown parity report from a [`ParityMatrix`].
///
/// The report contains:
/// 1. A summary table with per-category counts.
/// 2. A detailed section for every category with one row per feature.
///
/// The returned `String` is suitable for writing directly to a `.md` file
/// (e.g. `docs/esmf_parity.md`).
pub fn generate_report(matrix: &ParityMatrix) -> String {
    let mut out = String::with_capacity(4096);

    out.push_str("# ESMF SDK 2.x Parity Report — oxirs-samm\n\n");
    out.push_str("> **Generated automatically** by `cargo run --bin parity_report`.\n");
    out.push_str("> Do not edit by hand — regenerate after updating `esmf_catalog.toml`.\n\n");
    out.push_str("---\n\n");

    // ── Summary table ────────────────────────────────────────────────────
    out.push_str("## Summary\n\n");
    out.push_str("| Category | ✅ Implemented | ⚠️ Partial | ❌ Missing | Total |\n");
    out.push_str("|---|---|---|---|---|\n");

    let mut grand_impl = 0usize;
    let mut grand_part = 0usize;
    let mut grand_miss = 0usize;

    for cat in category_order() {
        let heading = category_heading(&cat);
        let entries = match matrix.get(&cat) {
            Some(e) => e.as_slice(),
            None => &[],
        };
        let (imp, part, miss) = count_by_status(entries);
        let total = imp + part + miss;
        grand_impl += imp;
        grand_part += part;
        grand_miss += miss;
        out.push_str(&format!(
            "| {heading} | {imp} | {part} | {miss} | {total} |\n"
        ));
    }

    // Also collect any Other categories that may exist in the matrix
    for (cat, entries) in matrix {
        if matches!(cat, FeatureCategory::Other(_)) {
            let heading = category_heading(cat);
            let (imp, part, miss) = count_by_status(entries);
            let total = imp + part + miss;
            grand_impl += imp;
            grand_part += part;
            grand_miss += miss;
            out.push_str(&format!(
                "| {heading} | {imp} | {part} | {miss} | {total} |\n"
            ));
        }
    }

    let grand_total = grand_impl + grand_part + grand_miss;
    out.push_str(&format!(
        "| **Total** | **{grand_impl}** | **{grand_part}** | **{grand_miss}** | **{grand_total}** |\n"
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
        out.push_str("| Feature | Status | oxirs-samm Module | ESMF Reference | Notes |\n");
        out.push_str("|---|---|---|---|---|\n");

        for entry in entries {
            let module = entry
                .oxirs_module
                .as_deref()
                .map(|m| format!("`{m}`"))
                .unwrap_or_else(|| "—".to_string());
            let badge = status_badge(&entry.status);
            // Escape pipe characters in fields to avoid breaking the table
            let name = entry.name.replace('|', "\\|");
            let notes = entry.notes.replace('|', "\\|");
            out.push_str(&format!(
                "| {name} | {badge} | {module} | [spec]({}) | {notes} |\n",
                entry.esmf_reference
            ));
        }
        out.push('\n');
    }

    // Any Other categories
    for (cat, entries) in matrix {
        if matches!(cat, FeatureCategory::Other(_)) {
            let heading = category_heading(cat);
            out.push_str(&format!("### {heading}\n\n"));
            out.push_str("| Feature | Status | oxirs-samm Module | ESMF Reference | Notes |\n");
            out.push_str("|---|---|---|---|---|\n");
            for entry in entries {
                let module = entry
                    .oxirs_module
                    .as_deref()
                    .map(|m| format!("`{m}`"))
                    .unwrap_or_else(|| "—".to_string());
                let badge = status_badge(&entry.status);
                let name = entry.name.replace('|', "\\|");
                let notes = entry.notes.replace('|', "\\|");
                out.push_str(&format!(
                    "| {name} | {badge} | {module} | [spec]({}) | {notes} |\n",
                    entry.esmf_reference
                ));
            }
            out.push('\n');
        }
    }

    out.push_str("---\n\n");
    out.push_str("*Report generated by `oxirs-samm` — ESMF SDK 2.x parity matrix.*\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parity::matrix::{FeatureCategory, FeatureEntry, ImplStatus};
    use std::collections::HashMap;

    fn sample_matrix() -> ParityMatrix {
        let mut m = HashMap::new();
        m.insert(
            FeatureCategory::AspectModeling,
            vec![
                FeatureEntry {
                    name: "Aspect definition".to_string(),
                    status: ImplStatus::Implemented,
                    oxirs_module: Some("metamodel::aspect".to_string()),
                    esmf_reference: "https://example.com/aspect".to_string(),
                    notes: "Core aspect.".to_string(),
                },
                FeatureEntry {
                    name: "Either characteristic".to_string(),
                    status: ImplStatus::Missing,
                    oxirs_module: None,
                    esmf_reference: "https://example.com/either".to_string(),
                    notes: "Not yet implemented.".to_string(),
                },
            ],
        );
        m.insert(
            FeatureCategory::Validation,
            vec![FeatureEntry {
                name: "SHACL validation".to_string(),
                status: ImplStatus::Implemented,
                oxirs_module: Some("validator::shacl_validator".to_string()),
                esmf_reference: "https://example.com/shacl".to_string(),
                notes: "Full SHACL support.".to_string(),
            }],
        );
        m
    }

    #[test]
    fn test_report_contains_headings() {
        let report = generate_report(&sample_matrix());
        assert!(
            report.contains("ESMF SDK 2.x Parity Report"),
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
        let report = generate_report(&sample_matrix());
        assert!(
            report.contains("Aspect Modeling"),
            "Aspect Modeling missing"
        );
        assert!(
            report.contains("Validation") || report.contains("validation"),
            "Validation missing"
        );
    }

    #[test]
    fn test_report_contains_status_badges() {
        let report = generate_report(&sample_matrix());
        assert!(report.contains("Implemented"), "implemented badge missing");
        assert!(report.contains("Missing"), "missing badge missing");
    }

    #[test]
    fn test_report_empty_matrix() {
        let report = generate_report(&HashMap::new());
        assert!(
            report.contains("ESMF SDK 2.x Parity Report"),
            "title should appear even for empty matrix"
        );
    }
}
