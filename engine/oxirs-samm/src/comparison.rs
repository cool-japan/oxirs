//! Model Comparison and Diff Utilities
//!
//! This module provides comprehensive utilities for comparing SAMM Aspect Models
//! and generating detailed diff reports. Useful for version control, change tracking,
//! and model evolution analysis.
//!
//! # Features
//!
//! - **Property-level comparison** - Detect added, removed, and modified properties
//! - **Characteristic comparison** - Deep comparison of characteristic changes
//! - **Metadata comparison** - Track changes in names, descriptions, and metadata
//! - **Structural comparison** - Detect changes in model structure and relationships
//! - **Diff report generation** - Generate human-readable change reports
//! - **Visual comparison** - Generate side-by-side visual diagrams showing changes
//! - **HTML diff reports** - Create interactive HTML reports with embedded visualizations
//! - **Mermaid.js comparison** - GitHub-friendly comparison diagrams with change highlighting
//!
//! # Examples
//!
//! ```rust
//! use oxirs_samm::comparison::ModelComparison;
//! use oxirs_samm::metamodel::Aspect;
//!
//! # fn example(old_aspect: &Aspect, new_aspect: &Aspect) {
//! // Compare two versions of a model
//! let comparison = ModelComparison::compare(old_aspect, new_aspect);
//!
//! // Check for changes
//! if comparison.has_changes() {
//!     println!("Properties added: {}", comparison.properties_added.len());
//!     println!("Properties removed: {}", comparison.properties_removed.len());
//!     println!("Properties modified: {}", comparison.properties_modified.len());
//! }
//!
//! // Get detailed diff report
//! let report = comparison.generate_report();
//! println!("{}", report);
//! # }
//! ```

use crate::metamodel::{Aspect, Characteristic, ModelElement, Property};
use std::collections::{HashMap, HashSet};

/// Result of comparing two SAMM Aspect Models
///
/// Contains detailed information about all differences between two models.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelComparison {
    /// URN of the old aspect
    pub old_urn: String,
    /// URN of the new aspect
    pub new_urn: String,
    /// Properties that were added in the new model
    pub properties_added: Vec<String>,
    /// Properties that were removed from the old model
    pub properties_removed: Vec<String>,
    /// Properties that were modified (URN to change description)
    pub properties_modified: HashMap<String, PropertyChange>,
    /// Metadata changes
    pub metadata_changes: Vec<MetadataChange>,
    /// Operations added
    pub operations_added: Vec<String>,
    /// Operations removed
    pub operations_removed: Vec<String>,
    /// Overall change summary
    pub summary: String,
}

/// Detailed change information for a property
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyChange {
    /// Property URN
    pub urn: String,
    /// Whether optional flag changed
    pub optional_changed: bool,
    /// Old optional value
    pub old_optional: bool,
    /// New optional value
    pub new_optional: bool,
    /// Whether characteristic changed
    pub characteristic_changed: bool,
    /// Description of characteristic change
    pub characteristic_change_description: Option<String>,
    /// Metadata changes for this property
    pub metadata_changes: Vec<String>,
}

/// Metadata change description
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataChange {
    /// Element URN
    pub element_urn: String,
    /// Type of metadata that changed
    pub change_type: MetadataChangeType,
    /// Description of the change
    pub description: String,
}

/// Types of metadata changes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataChangeType {
    /// Preferred name changed
    PreferredName,
    /// Description changed
    Description,
    /// See reference changed
    SeeReference,
    /// URN changed
    UrnChanged,
}

impl ModelComparison {
    /// Compares two SAMM Aspect Models and returns detailed differences
    ///
    /// # Arguments
    ///
    /// * `old` - The original/old version of the aspect
    /// * `new` - The updated/new version of the aspect
    ///
    /// # Returns
    ///
    /// A `ModelComparison` containing all detected differences
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::comparison::ModelComparison;
    /// # use oxirs_samm::metamodel::Aspect;
    /// # fn example(old: &Aspect, new: &Aspect) {
    /// let comparison = ModelComparison::compare(old, new);
    /// if comparison.has_changes() {
    ///     println!("Model has changed!");
    /// }
    /// # }
    /// ```
    pub fn compare(old: &Aspect, new: &Aspect) -> Self {
        let old_urn = old.urn().to_string();
        let new_urn = new.urn().to_string();

        // Build property maps for comparison
        let old_props: HashMap<_, _> = old
            .properties()
            .iter()
            .map(|p| (p.urn().to_string(), p))
            .collect();

        let new_props: HashMap<_, _> = new
            .properties()
            .iter()
            .map(|p| (p.urn().to_string(), p))
            .collect();

        // Find added and removed properties
        let old_urns: HashSet<_> = old_props.keys().cloned().collect();
        let new_urns: HashSet<_> = new_props.keys().cloned().collect();

        let properties_added: Vec<_> = new_urns.difference(&old_urns).cloned().collect();
        let properties_removed: Vec<_> = old_urns.difference(&new_urns).cloned().collect();

        // Find modified properties
        let mut properties_modified = HashMap::new();
        for urn in old_urns.intersection(&new_urns) {
            if let (Some(old_prop), Some(new_prop)) = (old_props.get(urn), new_props.get(urn)) {
                if let Some(change) = Self::compare_properties(old_prop, new_prop) {
                    properties_modified.insert(urn.clone(), change);
                }
            }
        }

        // Compare metadata
        let metadata_changes = Self::compare_metadata(old, new);

        // Compare operations
        let old_ops: HashSet<_> = old
            .operations()
            .iter()
            .map(|o| o.urn().to_string())
            .collect();
        let new_ops: HashSet<_> = new
            .operations()
            .iter()
            .map(|o| o.urn().to_string())
            .collect();

        let operations_added: Vec<_> = new_ops.difference(&old_ops).cloned().collect();
        let operations_removed: Vec<_> = old_ops.difference(&new_ops).cloned().collect();

        // Generate summary
        let summary = Self::generate_summary(
            &properties_added,
            &properties_removed,
            &properties_modified,
            &operations_added,
            &operations_removed,
            &metadata_changes,
        );

        Self {
            old_urn,
            new_urn,
            properties_added,
            properties_removed,
            properties_modified,
            metadata_changes,
            operations_added,
            operations_removed,
            summary,
        }
    }

    /// Compares two properties and returns changes if any
    fn compare_properties(old: &Property, new: &Property) -> Option<PropertyChange> {
        let mut has_changes = false;
        let mut metadata_changes = Vec::new();

        // Check optional flag
        let optional_changed = old.optional != new.optional;
        if optional_changed {
            has_changes = true;
        }

        // Check characteristic changes
        let characteristic_changed =
            !Self::characteristics_equal(&old.characteristic, &new.characteristic);
        let characteristic_change_description = if characteristic_changed {
            has_changes = true;
            Some(Self::describe_characteristic_change(
                &old.characteristic,
                &new.characteristic,
            ))
        } else {
            None
        };

        // Check metadata changes
        if old.metadata.preferred_names != new.metadata.preferred_names {
            has_changes = true;
            metadata_changes.push("Preferred names changed".to_string());
        }

        if old.metadata.descriptions != new.metadata.descriptions {
            has_changes = true;
            metadata_changes.push("Descriptions changed".to_string());
        }

        if old.payload_name != new.payload_name {
            has_changes = true;
            metadata_changes.push(format!(
                "Payload name changed: {:?} -> {:?}",
                old.payload_name, new.payload_name
            ));
        }

        if has_changes {
            Some(PropertyChange {
                urn: old.urn().to_string(),
                optional_changed,
                old_optional: old.optional,
                new_optional: new.optional,
                characteristic_changed,
                characteristic_change_description,
                metadata_changes,
            })
        } else {
            None
        }
    }

    /// Checks if two characteristics are equal
    fn characteristics_equal(old: &Option<Characteristic>, new: &Option<Characteristic>) -> bool {
        match (old, new) {
            (None, None) => true,
            (Some(_), None) | (None, Some(_)) => false,
            (Some(old_char), Some(new_char)) => {
                old_char.urn() == new_char.urn()
                    && old_char.data_type == new_char.data_type
                    && old_char.kind() == new_char.kind()
            }
        }
    }

    /// Describes the change in characteristics
    fn describe_characteristic_change(
        old: &Option<Characteristic>,
        new: &Option<Characteristic>,
    ) -> String {
        match (old, new) {
            (None, None) => "No change".to_string(),
            (None, Some(new_char)) => format!("Added characteristic: {}", new_char.urn()),
            (Some(old_char), None) => format!("Removed characteristic: {}", old_char.urn()),
            (Some(old_char), Some(new_char)) => {
                if old_char.urn() != new_char.urn() {
                    format!("Changed from {} to {}", old_char.urn(), new_char.urn())
                } else if old_char.data_type != new_char.data_type {
                    format!(
                        "Data type changed: {:?} -> {:?}",
                        old_char.data_type, new_char.data_type
                    )
                } else if old_char.kind() != new_char.kind() {
                    format!(
                        "Characteristic kind changed: {:?} -> {:?}",
                        old_char.kind(),
                        new_char.kind()
                    )
                } else {
                    "Characteristic details changed".to_string()
                }
            }
        }
    }

    /// Compares metadata between two aspects
    fn compare_metadata(old: &Aspect, new: &Aspect) -> Vec<MetadataChange> {
        let mut changes = Vec::new();

        // Check URN change
        if old.urn() != new.urn() {
            changes.push(MetadataChange {
                element_urn: old.urn().to_string(),
                change_type: MetadataChangeType::UrnChanged,
                description: format!("URN changed from {} to {}", old.urn(), new.urn()),
            });
        }

        // Check preferred names
        if old.metadata().preferred_names != new.metadata().preferred_names {
            changes.push(MetadataChange {
                element_urn: old.urn().to_string(),
                change_type: MetadataChangeType::PreferredName,
                description: "Preferred names changed".to_string(),
            });
        }

        // Check descriptions
        if old.metadata().descriptions != new.metadata().descriptions {
            changes.push(MetadataChange {
                element_urn: old.urn().to_string(),
                change_type: MetadataChangeType::Description,
                description: "Descriptions changed".to_string(),
            });
        }

        // Check see references
        if old.metadata().see_refs != new.metadata().see_refs {
            changes.push(MetadataChange {
                element_urn: old.urn().to_string(),
                change_type: MetadataChangeType::SeeReference,
                description: "See references changed".to_string(),
            });
        }

        changes
    }

    /// Generates a summary of all changes
    fn generate_summary(
        added: &[String],
        removed: &[String],
        modified: &HashMap<String, PropertyChange>,
        ops_added: &[String],
        ops_removed: &[String],
        metadata: &[MetadataChange],
    ) -> String {
        let total_changes = added.len()
            + removed.len()
            + modified.len()
            + ops_added.len()
            + ops_removed.len()
            + metadata.len();

        if total_changes == 0 {
            return "No changes detected".to_string();
        }

        let mut summary = format!("{} total changes detected:\n", total_changes);

        if !added.is_empty() {
            summary.push_str(&format!("  - {} properties added\n", added.len()));
        }
        if !removed.is_empty() {
            summary.push_str(&format!("  - {} properties removed\n", removed.len()));
        }
        if !modified.is_empty() {
            summary.push_str(&format!("  - {} properties modified\n", modified.len()));
        }
        if !ops_added.is_empty() {
            summary.push_str(&format!("  - {} operations added\n", ops_added.len()));
        }
        if !ops_removed.is_empty() {
            summary.push_str(&format!("  - {} operations removed\n", ops_removed.len()));
        }
        if !metadata.is_empty() {
            summary.push_str(&format!("  - {} metadata changes\n", metadata.len()));
        }

        summary
    }

    /// Checks if there are any changes between the models
    ///
    /// # Returns
    ///
    /// `true` if any changes were detected, `false` otherwise
    pub fn has_changes(&self) -> bool {
        !self.properties_added.is_empty()
            || !self.properties_removed.is_empty()
            || !self.properties_modified.is_empty()
            || !self.operations_added.is_empty()
            || !self.operations_removed.is_empty()
            || !self.metadata_changes.is_empty()
    }

    /// Generates a detailed human-readable diff report
    ///
    /// # Returns
    ///
    /// A formatted string containing the full diff report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# SAMM Model Comparison Report\n\n");
        report.push_str(&format!("Old Model: {}\n", self.old_urn));
        report.push_str(&format!("New Model: {}\n\n", self.new_urn));

        report.push_str(&format!("## Summary\n\n{}\n", self.summary));

        if !self.properties_added.is_empty() {
            report.push_str("\n## Properties Added\n\n");
            for prop in &self.properties_added {
                report.push_str(&format!("  + {}\n", prop));
            }
        }

        if !self.properties_removed.is_empty() {
            report.push_str("\n## Properties Removed\n\n");
            for prop in &self.properties_removed {
                report.push_str(&format!("  - {}\n", prop));
            }
        }

        if !self.properties_modified.is_empty() {
            report.push_str("\n## Properties Modified\n\n");
            for (urn, change) in &self.properties_modified {
                report.push_str(&format!("  ~ {}\n", urn));
                if change.optional_changed {
                    report.push_str(&format!(
                        "    - Optional: {} -> {}\n",
                        change.old_optional, change.new_optional
                    ));
                }
                if let Some(ref desc) = change.characteristic_change_description {
                    report.push_str(&format!("    - Characteristic: {}\n", desc));
                }
                for meta_change in &change.metadata_changes {
                    report.push_str(&format!("    - {}\n", meta_change));
                }
            }
        }

        if !self.operations_added.is_empty() {
            report.push_str("\n## Operations Added\n\n");
            for op in &self.operations_added {
                report.push_str(&format!("  + {}\n", op));
            }
        }

        if !self.operations_removed.is_empty() {
            report.push_str("\n## Operations Removed\n\n");
            for op in &self.operations_removed {
                report.push_str(&format!("  - {}\n", op));
            }
        }

        if !self.metadata_changes.is_empty() {
            report.push_str("\n## Metadata Changes\n\n");
            for change in &self.metadata_changes {
                report.push_str(&format!(
                    "  ~ {}: {}\n",
                    change.element_urn, change.description
                ));
            }
        }

        report
    }

    /// Checks if the changes are breaking (backwards incompatible)
    ///
    /// Breaking changes include:
    /// - Removing properties
    /// - Changing required properties to optional (data loss risk)
    /// - Changing characteristic types
    /// - Removing operations
    ///
    /// # Returns
    ///
    /// `true` if breaking changes detected, `false` otherwise
    pub fn has_breaking_changes(&self) -> bool {
        // Removed properties are breaking
        if !self.properties_removed.is_empty() {
            return true;
        }

        // Removed operations are breaking
        if !self.operations_removed.is_empty() {
            return true;
        }

        // Check for characteristic type changes (breaking)
        for change in self.properties_modified.values() {
            if change.characteristic_changed {
                return true;
            }
            // Changing from required to optional can be breaking in some contexts
            if change.optional_changed && !change.old_optional && change.new_optional {
                return true;
            }
        }

        false
    }

    /// Gets a list of all breaking changes with descriptions
    ///
    /// # Returns
    ///
    /// Vector of breaking change descriptions
    pub fn get_breaking_changes(&self) -> Vec<String> {
        let mut breaking = Vec::new();

        for prop in &self.properties_removed {
            breaking.push(format!("Property removed: {}", prop));
        }

        for op in &self.operations_removed {
            breaking.push(format!("Operation removed: {}", op));
        }

        for (urn, change) in &self.properties_modified {
            if change.characteristic_changed {
                breaking.push(format!("Property characteristic changed: {}", urn));
            }
            if change.optional_changed && !change.old_optional && change.new_optional {
                breaking.push(format!(
                    "Property changed from required to optional: {}",
                    urn
                ));
            }
        }

        breaking
    }

    /// Generates an HTML diff report with side-by-side visual comparison
    ///
    /// Creates an interactive HTML document showing before/after diagrams
    /// with detailed change highlighting and statistics.
    ///
    /// # Arguments
    ///
    /// * `old_aspect` - The original version of the aspect
    /// * `new_aspect` - The updated version of the aspect
    ///
    /// # Returns
    ///
    /// HTML string containing the complete diff report
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oxirs_samm::comparison::ModelComparison;
    /// # use oxirs_samm::metamodel::Aspect;
    /// # fn example(old: &Aspect, new: &Aspect) -> Result<(), Box<dyn std::error::Error>> {
    /// let comparison = ModelComparison::compare(old, new);
    /// let html_report = comparison.generate_visual_diff_html(old, new)?;
    /// std::fs::write("diff_report.html", html_report)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate_visual_diff_html(
        &self,
        old_aspect: &Aspect,
        new_aspect: &Aspect,
    ) -> Result<String, crate::error::SammError> {
        use crate::generators::diagram::{generate_diagram, DiagramFormat, DiagramStyle};

        let style = DiagramStyle::default();

        // Generate diagrams for both versions
        let old_diagram = generate_diagram(old_aspect, DiagramFormat::Mermaid(style.clone()))?;
        let new_diagram = generate_diagram(new_aspect, DiagramFormat::Mermaid(style))?;

        // Build HTML report
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
        html.push_str("  <meta charset=\"UTF-8\">\n");
        html.push_str(
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        html.push_str(&format!(
            "  <title>Model Comparison: {} vs {}</title>\n",
            old_aspect.name(),
            new_aspect.name()
        ));
        html.push_str("  <script src=\"https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js\"></script>\n");
        html.push_str("  <style>\n");
        html.push_str("    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }\n");
        html.push_str("    .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }\n");
        html.push_str(
            "    h1 { color: #333; border-bottom: 3px solid #4682B4; padding-bottom: 10px; }\n",
        );
        html.push_str("    h2 { color: #555; margin-top: 30px; }\n");
        html.push_str("    .comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }\n");
        html.push_str("    .diagram-panel { background: white; padding: 20px; border-radius: 4px; border: 2px solid #ddd; }\n");
        html.push_str("    .diagram-panel h3 { margin-top: 0; text-align: center; padding: 10px; border-radius: 4px; }\n");
        html.push_str("    .old-version { border-color: #ff6b6b; }\n");
        html.push_str("    .old-version h3 { background: #ffe0e0; color: #c92a2a; }\n");
        html.push_str("    .new-version { border-color: #51cf66; }\n");
        html.push_str("    .new-version h3 { background: #e0ffe0; color: #2f9e44; }\n");
        html.push_str("    .change-summary { background: #f8f9fa; padding: 20px; border-radius: 4px; margin: 20px 0; border-left: 4px solid #4682B4; }\n");
        html.push_str("    .change-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }\n");
        html.push_str("    .stat-card { background: white; padding: 15px; border-radius: 4px; text-align: center; border: 1px solid #dee2e6; }\n");
        html.push_str("    .stat-value { font-size: 32px; font-weight: bold; }\n");
        html.push_str("    .stat-label { color: #666; font-size: 14px; margin-top: 5px; }\n");
        html.push_str("    .added { color: #2f9e44; }\n");
        html.push_str("    .removed { color: #c92a2a; }\n");
        html.push_str("    .modified { color: #f08c00; }\n");
        html.push_str("    .unchanged { color: #495057; }\n");
        html.push_str("    .change-list { list-style: none; padding: 0; }\n");
        html.push_str("    .change-list li { padding: 8px; margin: 4px 0; border-radius: 4px; }\n");
        html.push_str(
            "    .change-list li.added { background: #e0ffe0; border-left: 3px solid #2f9e44; }\n",
        );
        html.push_str("    .change-list li.removed { background: #ffe0e0; border-left: 3px solid #c92a2a; }\n");
        html.push_str("    .change-list li.modified { background: #fff4e6; border-left: 3px solid #f08c00; }\n");
        html.push_str("    .breaking-warning { background: #fff3cd; border: 2px solid #ffc107; padding: 15px; border-radius: 4px; margin: 20px 0; }\n");
        html.push_str("    .breaking-warning h3 { margin-top: 0; color: #856404; }\n");
        html.push_str("  </style>\n</head>\n<body>\n");

        html.push_str("  <div class=\"container\">\n");
        html.push_str("    <h1>📊 Model Comparison Report</h1>\n");
        html.push_str(&format!(
            "    <p><strong>Old Version:</strong> {} | <strong>New Version:</strong> {}</p>\n",
            old_aspect.name(),
            new_aspect.name()
        ));

        // Change statistics
        html.push_str("    <div class=\"change-stats\">\n");
        html.push_str("      <div class=\"stat-card\">\n");
        html.push_str(&format!(
            "        <div class=\"stat-value added\">+{}</div>\n",
            self.properties_added.len()
        ));
        html.push_str("        <div class=\"stat-label\">Properties Added</div>\n");
        html.push_str("      </div>\n");
        html.push_str("      <div class=\"stat-card\">\n");
        html.push_str(&format!(
            "        <div class=\"stat-value removed\">-{}</div>\n",
            self.properties_removed.len()
        ));
        html.push_str("        <div class=\"stat-label\">Properties Removed</div>\n");
        html.push_str("      </div>\n");
        html.push_str("      <div class=\"stat-card\">\n");
        html.push_str(&format!(
            "        <div class=\"stat-value modified\">~{}</div>\n",
            self.properties_modified.len()
        ));
        html.push_str("        <div class=\"stat-label\">Properties Modified</div>\n");
        html.push_str("      </div>\n");
        html.push_str("      <div class=\"stat-card\">\n");
        html.push_str(&format!(
            "        <div class=\"stat-value {}\">{}</div>\n",
            if self.has_breaking_changes() {
                "removed"
            } else {
                "unchanged"
            },
            if self.has_breaking_changes() {
                "YES"
            } else {
                "NO"
            }
        ));
        html.push_str("        <div class=\"stat-label\">Breaking Changes</div>\n");
        html.push_str("      </div>\n");
        html.push_str("    </div>\n");

        // Breaking changes warning
        if self.has_breaking_changes() {
            html.push_str("    <div class=\"breaking-warning\">\n");
            html.push_str("      <h3>⚠️ Breaking Changes Detected</h3>\n");
            html.push_str("      <ul>\n");
            for change in self.get_breaking_changes() {
                html.push_str(&format!("        <li>{}</li>\n", change));
            }
            html.push_str("      </ul>\n");
            html.push_str("    </div>\n");
        }

        // Side-by-side comparison
        html.push_str("    <h2>Visual Comparison</h2>\n");
        html.push_str("    <div class=\"comparison-grid\">\n");

        // Old version
        html.push_str("      <div class=\"diagram-panel old-version\">\n");
        html.push_str("        <h3>📤 Old Version</h3>\n");
        html.push_str("        <div class=\"mermaid\">\n");
        html.push_str(&old_diagram);
        html.push_str("        </div>\n");
        html.push_str("      </div>\n");

        // New version
        html.push_str("      <div class=\"diagram-panel new-version\">\n");
        html.push_str("        <h3>📥 New Version</h3>\n");
        html.push_str("        <div class=\"mermaid\">\n");
        html.push_str(&new_diagram);
        html.push_str("        </div>\n");
        html.push_str("      </div>\n");
        html.push_str("    </div>\n");

        // Detailed change summary
        html.push_str("    <h2>Detailed Changes</h2>\n");
        html.push_str("    <div class=\"change-summary\">\n");
        html.push_str(&format!("      <pre>{}</pre>\n", self.summary));
        html.push_str("    </div>\n");

        // Properties added
        if !self.properties_added.is_empty() {
            html.push_str("    <h3>✅ Properties Added</h3>\n");
            html.push_str("    <ul class=\"change-list\">\n");
            for prop in &self.properties_added {
                html.push_str(&format!("      <li class=\"added\">{}</li>\n", prop));
            }
            html.push_str("    </ul>\n");
        }

        // Properties removed
        if !self.properties_removed.is_empty() {
            html.push_str("    <h3>❌ Properties Removed</h3>\n");
            html.push_str("    <ul class=\"change-list\">\n");
            for prop in &self.properties_removed {
                html.push_str(&format!("      <li class=\"removed\">{}</li>\n", prop));
            }
            html.push_str("    </ul>\n");
        }

        // Properties modified
        if !self.properties_modified.is_empty() {
            html.push_str("    <h3>🔄 Properties Modified</h3>\n");
            html.push_str("    <ul class=\"change-list\">\n");
            for (urn, change) in &self.properties_modified {
                html.push_str(&format!(
                    "      <li class=\"modified\"><strong>{}</strong><br/>",
                    urn
                ));
                if change.optional_changed {
                    html.push_str(&format!(
                        "        Optional: {} → {}<br/>",
                        change.old_optional, change.new_optional
                    ));
                }
                if let Some(desc) = &change.characteristic_change_description {
                    html.push_str(&format!("        {}<br/>", desc));
                }
                for meta_change in &change.metadata_changes {
                    html.push_str(&format!("        {}<br/>", meta_change));
                }
                html.push_str("      </li>\n");
            }
            html.push_str("    </ul>\n");
        }

        // Footer
        html.push_str("    <hr style=\"margin-top: 40px;\">\n");
        html.push_str("    <p style=\"text-align: center; color: #666; font-size: 12px;\">\n");
        html.push_str("      Generated by OxiRS SAMM • Model Comparison Report\n");
        html.push_str("    </p>\n");
        html.push_str("  </div>\n");

        html.push_str("  <script>\n");
        html.push_str("    mermaid.initialize({ startOnLoad: true, theme: 'default' });\n");
        html.push_str("  </script>\n");
        html.push_str("</body>\n</html>\n");

        Ok(html)
    }

    /// Generates a Mermaid.js comparison diagram highlighting changes
    ///
    /// Creates a single Mermaid diagram that shows both versions with
    /// color-coding to indicate added, removed, and modified elements.
    ///
    /// # Arguments
    ///
    /// * `old_aspect` - The original version of the aspect
    /// * `new_aspect` - The updated version of the aspect
    ///
    /// # Returns
    ///
    /// Mermaid diagram string with change annotations
    pub fn generate_mermaid_comparison(
        &self,
        old_aspect: &Aspect,
        new_aspect: &Aspect,
    ) -> Result<String, crate::error::SammError> {
        use crate::metamodel::ModelElement;

        let mut mermaid = String::new();
        mermaid.push_str("graph LR\n");

        // Create nodes for both aspects
        mermaid.push_str(&format!("    Old[\"{} (Old)\"]\n", old_aspect.name()));
        mermaid.push_str("    style Old fill:#ffe0e0,stroke:#c92a2a,stroke-width:3px\n\n");

        mermaid.push_str(&format!("    New[\"{} (New)\"]\n", new_aspect.name()));
        mermaid.push_str("    style New fill:#e0ffe0,stroke:#2f9e44,stroke-width:3px\n\n");

        // Add properties with change indicators
        let old_props: std::collections::HashMap<_, _> = old_aspect
            .properties()
            .iter()
            .map(|p| (p.urn().to_string(), p))
            .collect();

        let new_props: std::collections::HashMap<_, _> = new_aspect
            .properties()
            .iter()
            .map(|p| (p.urn().to_string(), p))
            .collect();

        // Properties in old version
        for prop in old_aspect.properties() {
            let prop_id = sanitize_id(prop.urn());
            let prop_name = prop.name();

            if self.properties_removed.contains(&prop.urn().to_string()) {
                // Removed property
                mermaid.push_str(&format!("    {}[\"❌ {}\"]\n", prop_id, prop_name));
                mermaid.push_str(&format!("    Old --> {}\n", prop_id));
                mermaid.push_str(&format!(
                    "    style {} fill:#ffe0e0,stroke:#c92a2a\n",
                    prop_id
                ));
            } else if self.properties_modified.contains_key(prop.urn()) {
                // Modified property
                mermaid.push_str(&format!("    {}[\"🔄 {}\"]\n", prop_id, prop_name));
                mermaid.push_str(&format!("    Old --> {}\n", prop_id));
                mermaid.push_str(&format!("    {} --> New\n", prop_id));
                mermaid.push_str(&format!(
                    "    style {} fill:#fff4e6,stroke:#f08c00\n",
                    prop_id
                ));
            } else {
                // Unchanged property
                mermaid.push_str(&format!("    {}[\"{}\"]\n", prop_id, prop_name));
                mermaid.push_str(&format!("    Old --> {}\n", prop_id));
                mermaid.push_str(&format!("    {} --> New\n", prop_id));
                mermaid.push_str(&format!("    style {} fill:#f0f0f0,stroke:#999\n", prop_id));
            }
        }

        // Added properties (only in new version)
        for added in &self.properties_added {
            if let Some(prop) = new_props.get(added) {
                let prop_id = sanitize_id(prop.urn());
                let prop_name = prop.name();
                mermaid.push_str(&format!("    {}[\"✅ {}\"]\n", prop_id, prop_name));
                mermaid.push_str(&format!("    New --> {}\n", prop_id));
                mermaid.push_str(&format!(
                    "    style {} fill:#e0ffe0,stroke:#2f9e44\n",
                    prop_id
                ));
            }
        }

        Ok(mermaid)
    }

    /// Export comparison as JSON
    ///
    /// Returns a JSON representation of the comparison result for programmatic processing.
    ///
    /// # Returns
    ///
    /// JSON string containing all changes
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::comparison::ModelComparison;
    /// # use oxirs_samm::metamodel::Aspect;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let aspect1 = Aspect::new("test1".to_string());
    /// # let aspect2 = Aspect::new("test2".to_string());
    /// let comparison = ModelComparison::compare(&aspect1, &aspect2);
    /// let json = comparison.export_json()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn export_json(&self) -> Result<String, crate::error::SammError> {
        use serde_json::json;

        let property_changes: Vec<_> = self
            .properties_modified
            .iter()
            .map(|(urn, change)| {
                json!({
                    "urn": urn,
                    "optional_changed": change.optional_changed,
                    "old_optional": change.old_optional,
                    "new_optional": change.new_optional,
                    "characteristic_changed": change.characteristic_changed,
                    "characteristic_change_description": change.characteristic_change_description,
                    "metadata_changes": change.metadata_changes,
                })
            })
            .collect();

        let metadata_changes: Vec<_> = self
            .metadata_changes
            .iter()
            .map(|change| {
                json!({
                    "element_urn": change.element_urn,
                    "change_type": format!("{:?}", change.change_type),
                    "description": change.description,
                })
            })
            .collect();

        let result = json!({
            "has_changes": self.has_changes(),
            "properties_added": self.properties_added,
            "properties_removed": self.properties_removed,
            "property_changes": property_changes,
            "metadata_changes": metadata_changes,
            "operations_added": self.operations_added,
            "operations_removed": self.operations_removed,
            "summary": self.summary,
        });

        serde_json::to_string_pretty(&result)
            .map_err(|e| crate::error::SammError::ParseError(format!("JSON export error: {}", e)))
    }

    /// Export comparison as Markdown table
    ///
    /// Returns a Markdown-formatted table of all changes for documentation.
    ///
    /// # Returns
    ///
    /// Markdown string with change summary
    pub fn export_markdown_table(&self) -> String {
        let mut md = String::new();

        md.push_str("# Model Comparison\n\n");

        if !self.has_changes() {
            md.push_str("*No changes detected*\n");
            return md;
        }

        // Property changes
        if !self.properties_added.is_empty()
            || !self.properties_removed.is_empty()
            || !self.properties_modified.is_empty()
        {
            md.push_str("## Property Changes\n\n");
            md.push_str("| Property | Change Type | Details |\n");
            md.push_str("|----------|-------------|----------|\n");

            for prop_urn in &self.properties_added {
                md.push_str(&format!("| `{}` | ✅ Added | New property |\n", prop_urn));
            }

            for prop_urn in &self.properties_removed {
                md.push_str(&format!(
                    "| `{}` | ❌ Removed | Deleted property |\n",
                    prop_urn
                ));
            }

            for (urn, change) in &self.properties_modified {
                let mut details = Vec::new();
                if change.optional_changed {
                    details.push(format!(
                        "optional: {} → {}",
                        change.old_optional, change.new_optional
                    ));
                }
                if change.characteristic_changed {
                    if let Some(desc) = &change.characteristic_change_description {
                        details.push(format!("characteristic: {}", desc));
                    }
                }
                let details_str = if details.is_empty() {
                    "Modified".to_string()
                } else {
                    details.join(", ")
                };
                md.push_str(&format!("| `{}` | 🔄 Modified | {} |\n", urn, details_str));
            }

            md.push('\n');
        }

        // Metadata changes
        if !self.metadata_changes.is_empty() {
            md.push_str("## Metadata Changes\n\n");
            md.push_str("| Element | Type | Description |\n");
            md.push_str("|---------|------|-------------|\n");

            for change in &self.metadata_changes {
                md.push_str(&format!(
                    "| `{}` | {:?} | {} |\n",
                    change.element_urn, change.change_type, change.description
                ));
            }

            md.push('\n');
        }

        // Operations
        if !self.operations_added.is_empty() || !self.operations_removed.is_empty() {
            md.push_str("## Operation Changes\n\n");
            md.push_str("| Operation | Change Type |\n");
            md.push_str("|-----------|-------------|\n");

            for op in &self.operations_added {
                md.push_str(&format!("| `{}` | ✅ Added |\n", op));
            }

            for op in &self.operations_removed {
                md.push_str(&format!("| `{}` | ❌ Removed |\n", op));
            }

            md.push('\n');
        }

        md
    }

    /// Export comparison as CSV
    ///
    /// Returns a CSV representation for import into spreadsheet tools.
    ///
    /// # Returns
    ///
    /// CSV string with all changes
    pub fn export_csv(&self) -> String {
        let mut csv = String::new();

        csv.push_str("Element Type,Element URN,Change Type,Field,Old Value,New Value\n");

        for prop_urn in &self.properties_added {
            csv.push_str(&format!("Property,{},Added,,,\n", prop_urn));
        }

        for prop_urn in &self.properties_removed {
            csv.push_str(&format!("Property,{},Removed,,,\n", prop_urn));
        }

        for (urn, change) in &self.properties_modified {
            if change.optional_changed {
                csv.push_str(&format!(
                    "Property,{},Modified,optional,{},{}\n",
                    urn, change.old_optional, change.new_optional
                ));
            }
            if change.characteristic_changed {
                csv.push_str(&format!(
                    "Property,{},Modified,characteristic,,{}\n",
                    urn,
                    change
                        .characteristic_change_description
                        .as_deref()
                        .unwrap_or("")
                ));
            }
        }

        for change in &self.metadata_changes {
            csv.push_str(&format!(
                "Metadata,{},{:?},,{},\n",
                change.element_urn, change.change_type, change.description
            ));
        }

        for op in &self.operations_added {
            csv.push_str(&format!("Operation,{},Added,,,\n", op));
        }

        for op in &self.operations_removed {
            csv.push_str(&format!("Operation,{},Removed,,,\n", op));
        }

        csv
    }
}

fn sanitize_id(urn: &str) -> String {
    urn.replace([':', '#', '.', '-', ' ', '/'], "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::Property;

    #[test]
    fn test_compare_identical_models() {
        let aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let comparison = ModelComparison::compare(&aspect1, &aspect2);

        assert!(!comparison.has_changes());
        assert!(comparison.properties_added.is_empty());
        assert!(comparison.properties_removed.is_empty());
        assert!(comparison.properties_modified.is_empty());
    }

    #[test]
    fn test_detect_added_property() {
        let aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let mut aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        aspect2.add_property(Property::new("urn:samm:test:1.0.0#newProp".to_string()));

        let comparison = ModelComparison::compare(&aspect1, &aspect2);

        assert!(comparison.has_changes());
        assert_eq!(comparison.properties_added.len(), 1);
        assert!(comparison
            .properties_added
            .contains(&"urn:samm:test:1.0.0#newProp".to_string()));
    }

    #[test]
    fn test_detect_removed_property() {
        let mut aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        aspect1.add_property(Property::new("urn:samm:test:1.0.0#oldProp".to_string()));

        let comparison = ModelComparison::compare(&aspect1, &aspect2);

        assert!(comparison.has_changes());
        assert_eq!(comparison.properties_removed.len(), 1);
        assert!(comparison
            .properties_removed
            .contains(&"urn:samm:test:1.0.0#oldProp".to_string()));
    }

    #[test]
    fn test_detect_optional_change() {
        let mut aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let mut aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let prop1 = Property::new("urn:samm:test:1.0.0#prop1".to_string());
        let mut prop2 = Property::new("urn:samm:test:1.0.0#prop1".to_string());
        prop2.optional = true;

        aspect1.add_property(prop1);
        aspect2.add_property(prop2);

        let comparison = ModelComparison::compare(&aspect1, &aspect2);

        assert!(comparison.has_changes());
        assert_eq!(comparison.properties_modified.len(), 1);

        let change = comparison
            .properties_modified
            .get("urn:samm:test:1.0.0#prop1")
            .expect("operation should succeed");
        assert!(change.optional_changed);
        assert!(!change.old_optional);
        assert!(change.new_optional);
    }

    #[test]
    fn test_breaking_changes_detection() {
        let mut aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        aspect1.add_property(Property::new("urn:samm:test:1.0.0#removedProp".to_string()));

        let comparison = ModelComparison::compare(&aspect1, &aspect2);

        assert!(comparison.has_breaking_changes());
        let breaking = comparison.get_breaking_changes();
        assert_eq!(breaking.len(), 1);
        assert!(breaking[0].contains("removed"));
    }

    #[test]
    fn test_generate_report() {
        let mut aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let mut aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        aspect1.add_property(Property::new("urn:samm:test:1.0.0#oldProp".to_string()));
        aspect2.add_property(Property::new("urn:samm:test:1.0.0#newProp".to_string()));

        let comparison = ModelComparison::compare(&aspect1, &aspect2);
        let report = comparison.generate_report();

        assert!(report.contains("SAMM Model Comparison Report"));
        assert!(report.contains("Properties Added"));
        assert!(report.contains("Properties Removed"));
    }

    #[test]
    fn test_metadata_changes_detection() {
        let mut aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let mut aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        aspect1
            .metadata
            .add_preferred_name("en".to_string(), "Old Name".to_string());
        aspect2
            .metadata
            .add_preferred_name("en".to_string(), "New Name".to_string());

        let comparison = ModelComparison::compare(&aspect1, &aspect2);

        assert!(comparison.has_changes());
        assert!(!comparison.metadata_changes.is_empty());
    }

    #[test]
    fn test_no_breaking_changes_for_additions() {
        let aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let mut aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        aspect2.add_property(Property::new("urn:samm:test:1.0.0#newProp".to_string()));

        let comparison = ModelComparison::compare(&aspect1, &aspect2);

        assert!(comparison.has_changes());
        assert!(!comparison.has_breaking_changes());
    }

    #[test]
    fn test_generate_visual_diff_html() {
        let mut aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let mut aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        aspect1.add_property(Property::new("urn:samm:test:1.0.0#oldProp".to_string()));
        aspect2.add_property(Property::new("urn:samm:test:1.0.0#newProp".to_string()));

        let comparison = ModelComparison::compare(&aspect1, &aspect2);
        let html = comparison.generate_visual_diff_html(&aspect1, &aspect2);

        assert!(html.is_ok());
        let html_content = html.expect("operation should succeed");
        assert!(html_content.contains("<!DOCTYPE html>"));
        assert!(html_content.contains("Model Comparison Report"));
        assert!(html_content.contains("mermaid"));
        assert!(html_content.contains("Old Version"));
        assert!(html_content.contains("New Version"));
        assert!(html_content.contains("Properties Added"));
        assert!(html_content.contains("Properties Removed"));
    }

    #[test]
    fn test_generate_visual_diff_html_with_breaking_changes() {
        let mut aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        aspect1.add_property(Property::new(
            "urn:samm:test:1.0.0#requiredProp".to_string(),
        ));

        let comparison = ModelComparison::compare(&aspect1, &aspect2);
        let html = comparison.generate_visual_diff_html(&aspect1, &aspect2);

        assert!(html.is_ok());
        let html_content = html.expect("operation should succeed");
        assert!(html_content.contains("Breaking Changes Detected"));
        assert!(html_content.contains("breaking-warning"));
    }

    #[test]
    fn test_generate_mermaid_comparison() {
        let mut aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let mut aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        aspect1.add_property(Property::new("urn:samm:test:1.0.0#oldProp".to_string()));
        aspect2.add_property(Property::new("urn:samm:test:1.0.0#newProp".to_string()));

        let comparison = ModelComparison::compare(&aspect1, &aspect2);
        let mermaid = comparison.generate_mermaid_comparison(&aspect1, &aspect2);

        assert!(mermaid.is_ok());
        let mermaid_content = mermaid.expect("operation should succeed");
        assert!(mermaid_content.contains("graph LR"));
        assert!(mermaid_content.contains("Old"));
        assert!(mermaid_content.contains("New"));
        assert!(mermaid_content.contains("❌")); // removed marker
        assert!(mermaid_content.contains("✅")); // added marker
    }

    #[test]
    fn test_generate_mermaid_comparison_with_modifications() {
        let mut aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let mut aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let prop1 = Property::new("urn:samm:test:1.0.0#prop1".to_string());
        let mut prop2 = Property::new("urn:samm:test:1.0.0#prop1".to_string());
        prop2.optional = true;

        aspect1.add_property(prop1);
        aspect2.add_property(prop2);

        let comparison = ModelComparison::compare(&aspect1, &aspect2);
        let mermaid = comparison.generate_mermaid_comparison(&aspect1, &aspect2);

        assert!(mermaid.is_ok());
        let mermaid_content = mermaid.expect("operation should succeed");
        assert!(mermaid_content.contains("🔄")); // modified marker
        assert!(mermaid_content.contains("fill:#fff4e6")); // modified color
    }

    #[test]
    fn test_generate_mermaid_comparison_unchanged_properties() {
        let mut aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let mut aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let prop1 = Property::new("urn:samm:test:1.0.0#unchangedProp".to_string());
        let prop2 = Property::new("urn:samm:test:1.0.0#unchangedProp".to_string());

        aspect1.add_property(prop1);
        aspect2.add_property(prop2);

        let comparison = ModelComparison::compare(&aspect1, &aspect2);
        let mermaid = comparison.generate_mermaid_comparison(&aspect1, &aspect2);

        assert!(mermaid.is_ok());
        let mermaid_content = mermaid.expect("operation should succeed");
        // Unchanged properties should be present and connected to both versions
        assert!(mermaid_content.contains("unchangedProp"));
        assert!(mermaid_content.contains("fill:#f0f0f0")); // unchanged color
    }

    #[test]
    fn test_visual_diff_html_structure() {
        let aspect1 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let aspect2 = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let comparison = ModelComparison::compare(&aspect1, &aspect2);
        let html = comparison.generate_visual_diff_html(&aspect1, &aspect2);

        assert!(html.is_ok());
        let html_content = html.expect("operation should succeed");

        // Check for essential HTML elements
        assert!(html_content.contains("<html"));
        assert!(html_content.contains("</html>"));
        assert!(html_content.contains("<head>"));
        assert!(html_content.contains("</head>"));
        assert!(html_content.contains("<body>"));
        assert!(html_content.contains("</body>"));
        assert!(html_content.contains("comparison-grid"));
        assert!(html_content.contains("stat-card"));
        assert!(html_content.contains("mermaid.initialize"));
    }

    #[test]
    fn test_export_json() {
        let mut aspect1 = Aspect::new("urn:samm:test:1.0.0#Aspect1".to_string());
        let mut prop1 = Property::new("urn:samm:test:1.0.0#prop1".to_string());
        prop1.optional = false;
        aspect1.add_property(prop1);

        let mut aspect2 = Aspect::new("urn:samm:test:1.0.0#Aspect2".to_string());
        let mut prop2 = Property::new("urn:samm:test:1.0.0#prop2".to_string());
        prop2.optional = true;
        aspect2.add_property(prop2);

        let comparison = ModelComparison::compare(&aspect1, &aspect2);
        let json = comparison.export_json().expect("comparison should succeed");

        assert!(json.contains("property_changes"));
        assert!(json.contains("metadata_changes"));
    }

    #[test]
    fn test_export_markdown_table() {
        let mut aspect1 = Aspect::new("urn:samm:test:1.0.0#Aspect1".to_string());
        let prop1 = Property::new("urn:samm:test:1.0.0#prop1".to_string());
        aspect1.add_property(prop1);

        let aspect2 = Aspect::new("urn:samm:test:1.0.0#Aspect2".to_string());

        let comparison = ModelComparison::compare(&aspect1, &aspect2);
        let markdown = comparison.export_markdown_table();

        assert!(markdown.contains("Property"));
        assert!(markdown.contains("Change Type"));
        assert!(markdown.contains("|"));
    }
}
