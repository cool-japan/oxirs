//! RDF Patch serialization.
//!
//! Exports: [`PatchSerializer`].

use super::patch_types::{PatchChange, RdfPatch};

// ─── PatchSerializer ─────────────────────────────────────────────────────────

/// Serializes [`RdfPatch`] documents to the RDF Patch text format
pub struct PatchSerializer;

impl PatchSerializer {
    /// Serialize an [`RdfPatch`] to a string
    pub fn serialize(patch: &RdfPatch) -> String {
        let mut out = String::new();
        for header in &patch.headers {
            out.push_str(&header.to_string());
            out.push('\n');
        }
        if !patch.headers.is_empty() && !patch.changes.is_empty() {
            out.push('\n');
        }
        for change in &patch.changes {
            out.push_str(&change.to_string());
            out.push('\n');
        }
        out
    }

    /// Serialize a single [`PatchChange`] line (no newline appended)
    pub fn serialize_change(change: &PatchChange) -> String {
        change.to_string()
    }
}
