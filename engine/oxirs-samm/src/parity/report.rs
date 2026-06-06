//! Markdown report generator for the ESMF SDK 2.x parity matrix.
//!
//! Call [`generate_report`] to produce a complete markdown document
//! summarising the implementation status across all feature categories.

use crate::parity::matrix::{FeatureCategory, FeatureEntry, FeatureStatus, ParityMatrix};

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

/// Badge string for a status value.
fn status_badge(status: &FeatureStatus) -> &'static str {
    match status {
        FeatureStatus::Done => "Done",
        FeatureStatus::Partial => "Partial",
        FeatureStatus::Missing => "Missing",
    }
}

/// Count features by status across a slice of entries.
/// Returns `(done, partial, missing)`.
fn count_by_status(entries: &[FeatureEntry]) -> (usize, usize, usize) {
    let done = entries
        .iter()
        .filter(|e| e.status == FeatureStatus::Done)
        .count();
    let partial = entries
        .iter()
        .filter(|e| e.status == FeatureStatus::Partial)
        .count();
    let missing = entries
        .iter()
        .filter(|e| e.status == FeatureStatus::Missing)
        .count();
    (done, partial, missing)
}

/// Generate a full markdown parity report from a [`ParityMatrix`].
///
/// The report contains:
/// 1. An H1 title "ESMF SDK 2.x Parity Report".
/// 2. A summary table with per-category counts at the top.
/// 3. Per-category H2 sections showing Done / Partial / Missing entries with
///    a symbol marker per row.
///
/// The returned `String` is suitable for writing directly to a `.md` file
/// (e.g. `docs/esmf_parity.md`).
pub fn generate_report(matrix: &ParityMatrix) -> String {
    let mut out = String::with_capacity(8192);

    out.push_str("# ESMF SDK 2.x Parity Report — oxirs-samm\n\n");
    out.push_str("> **Generated automatically** by `cargo run --bin parity_report`.\n");
    out.push_str("> Do not edit by hand — regenerate after updating `esmf_catalog.toml`.\n\n");
    out.push_str("---\n\n");

    // ── Summary table ──────────────────────────────────────────────────────
    out.push_str("## Summary\n\n");
    out.push_str("| Category | Done | Partial | Missing | Total | Coverage |\n");
    out.push_str("|---|---|---|---|---|---|\n");

    let mut grand_done = 0usize;
    let mut grand_part = 0usize;
    let mut grand_miss = 0usize;

    for cat in category_order() {
        let heading = category_heading(&cat);
        let entries = match matrix.get(&cat) {
            Some(e) => e,
            None => &[],
        };
        let (done, part, miss) = count_by_status(entries);
        let total = done + part + miss;
        grand_done += done;
        grand_part += part;
        grand_miss += miss;
        let coverage = if total == 0 {
            0.0
        } else {
            (done as f64 + part as f64 * 0.5) / total as f64 * 100.0
        };
        out.push_str(&format!(
            "| {heading} | {done} | {part} | {miss} | {total} | {coverage:.0}% |\n"
        ));
    }

    // Also collect any Other categories that may exist in the matrix
    for (cat, entries) in matrix.iter() {
        if matches!(cat, FeatureCategory::Other(_)) {
            let heading = category_heading(cat);
            let (done, part, miss) = count_by_status(entries);
            let total = done + part + miss;
            grand_done += done;
            grand_part += part;
            grand_miss += miss;
            let coverage = if total == 0 {
                0.0
            } else {
                (done as f64 + part as f64 * 0.5) / total as f64 * 100.0
            };
            out.push_str(&format!(
                "| {heading} | {done} | {part} | {miss} | {total} | {coverage:.0}% |\n"
            ));
        }
    }

    let grand_total = grand_done + grand_part + grand_miss;
    let grand_coverage = if grand_total == 0 {
        0.0
    } else {
        (grand_done as f64 + grand_part as f64 * 0.5) / grand_total as f64 * 100.0
    };
    out.push_str(&format!(
        "| **Total** | **{grand_done}** | **{grand_part}** | **{grand_miss}** | **{grand_total}** | **{grand_coverage:.0}%** |\n"
    ));
    out.push('\n');

    // ── Detailed per-category sections ─────────────────────────────────────
    out.push_str("## Detailed Parity Matrix\n\n");

    for cat in category_order() {
        let heading = category_heading(&cat);
        let entries = match matrix.get(&cat) {
            Some(e) => e,
            None => continue,
        };
        if entries.is_empty() {
            continue;
        }

        out.push_str(&format!("### {heading}\n\n"));
        out.push_str("| Status | Feature | Description | oxirs-samm Module | Notes |\n");
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
            let desc = entry.description.replace('|', "\\|");
            let notes = entry.notes.as_deref().unwrap_or("—").replace('|', "\\|");
            out.push_str(&format!(
                "| {badge} | {name} | {desc} | {module} | {notes} |\n"
            ));
        }
        out.push('\n');
    }

    // Any Other categories
    for (cat, entries) in matrix.iter() {
        if matches!(cat, FeatureCategory::Other(_)) {
            let heading = category_heading(cat);
            out.push_str(&format!("### {heading}\n\n"));
            out.push_str("| Status | Feature | Description | oxirs-samm Module | Notes |\n");
            out.push_str("|---|---|---|---|---|\n");
            for entry in entries {
                let module = entry
                    .oxirs_module
                    .as_deref()
                    .map(|m| format!("`{m}`"))
                    .unwrap_or_else(|| "—".to_string());
                let badge = status_badge(&entry.status);
                let name = entry.name.replace('|', "\\|");
                let desc = entry.description.replace('|', "\\|");
                let notes = entry.notes.as_deref().unwrap_or("—").replace('|', "\\|");
                out.push_str(&format!(
                    "| {badge} | {name} | {desc} | {module} | {notes} |\n"
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
    use crate::parity::matrix::{FeatureCategory, FeatureEntry, FeatureStatus, ParityMatrix};

    fn sample_matrix() -> ParityMatrix {
        let mut m = ParityMatrix::new();
        m.add_entry(
            FeatureCategory::AspectModeling,
            FeatureEntry {
                name: "Aspect definition".to_string(),
                description: "Core aspect element.".to_string(),
                status: FeatureStatus::Done,
                oxirs_module: Some("metamodel::aspect".to_string()),
                notes: None,
            },
        )
        .expect("first entry");
        m.add_entry(
            FeatureCategory::AspectModeling,
            FeatureEntry {
                name: "Either characteristic".to_string(),
                description: "Union type.".to_string(),
                status: FeatureStatus::Missing,
                oxirs_module: None,
                notes: Some("Not yet implemented.".to_string()),
            },
        )
        .expect("second entry");
        m.add_entry(
            FeatureCategory::Validation,
            FeatureEntry {
                name: "SHACL validation".to_string(),
                description: "SHACL shape validation.".to_string(),
                status: FeatureStatus::Done,
                oxirs_module: Some("validator::shacl_validator".to_string()),
                notes: None,
            },
        )
        .expect("validation entry");
        m
    }

    #[test]
    fn test_report_contains_headings() {
        let report = generate_report(&sample_matrix());
        assert!(
            report.contains("ESMF SDK"),
            "ESMF SDK title missing from report"
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
            "Aspect Modeling heading missing"
        );
        assert!(
            report.contains("Validation") || report.contains("validation"),
            "Validation heading missing"
        );
    }

    #[test]
    fn test_report_contains_status_markers() {
        let report = generate_report(&sample_matrix());
        assert!(report.contains("Done"), "done marker missing");
        assert!(report.contains("Missing"), "missing marker missing");
    }

    #[test]
    fn test_report_empty_matrix() {
        let report = generate_report(&ParityMatrix::new());
        assert!(
            report.contains("ESMF SDK"),
            "title should appear even for empty matrix"
        );
    }

    #[test]
    fn test_report_summary_table_present() {
        let report = generate_report(&sample_matrix());
        assert!(
            report.contains("| Category |"),
            "summary table header missing"
        );
    }
}
