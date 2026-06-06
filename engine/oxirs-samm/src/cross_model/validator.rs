//! Cross-model reference validator.
//!
//! Given a [`CrossModelRegistry`] populated with multiple SAMM model namespaces,
//! a [`CrossModelValidator`] checks whether all cross-model URN references in a
//! set of property descriptors resolve to known entries.
//!
//! # Example
//!
//! ```rust
//! use oxirs_samm::cross_model::{
//!     CrossModelRegistry, CrossModelValidator, CrossModelReference, ModelEntry,
//! };
//!
//! let mut registry = CrossModelRegistry::new();
//! registry.register_model(ModelEntry {
//!     namespace: "urn:samm:org.example.b:1.0.0".to_string(),
//!     file_path: None,
//!     exported_urns: vec!["urn:samm:org.example.b:1.0.0#VelocityChar".to_string()],
//! }).unwrap();
//!
//! let refs = CrossModelValidator::extract_cross_references(
//!     "urn:samm:org.example.a:1.0.0",
//!     &[("speed dataType", "urn:samm:org.example.b:1.0.0#VelocityChar")],
//! );
//!
//! let validator = CrossModelValidator::new(&registry);
//! let report = validator.validate("urn:samm:org.example.a:1.0.0", &refs);
//! assert!(report.is_valid());
//! ```

use super::CrossModelRegistry;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single cross-model URN reference found inside a source model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossModelReference {
    /// Namespace (model) that contains this reference.
    pub source_namespace: String,

    /// The URN being referenced (must live in a different namespace).
    pub target_urn: String,

    /// Human-readable label for where this reference appears, e.g.
    /// `"property 'temperature' dataType"`.
    pub context: String,
}

/// The outcome of a [`CrossModelValidator::validate`] call.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// All references that were checked (resolved + unresolved).
    pub checked_references: Vec<CrossModelReference>,

    /// References that could not be resolved against the registry.
    pub unresolved: Vec<CrossModelReference>,

    /// Number of references that resolved successfully.
    pub resolved_count: usize,
}

impl ValidationReport {
    /// `true` iff every checked reference resolved successfully.
    pub fn is_valid(&self) -> bool {
        self.unresolved.is_empty()
    }

    /// A human-readable one-line summary of the validation result.
    pub fn summary(&self) -> String {
        let total = self.checked_references.len();
        if self.is_valid() {
            format!("cross-model validation passed: {total} reference(s) resolved")
        } else {
            format!(
                "cross-model validation FAILED: {}/{total} reference(s) unresolved",
                self.unresolved.len()
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

/// Validates cross-model URN references against a [`CrossModelRegistry`].
///
/// The validator is bound to the registry for its lifetime (`'r`) so that
/// multiple validation passes can share a single registry without cloning.
pub struct CrossModelValidator<'r> {
    registry: &'r CrossModelRegistry,
}

impl<'r> CrossModelValidator<'r> {
    /// Create a new validator bound to `registry`.
    pub fn new(registry: &'r CrossModelRegistry) -> Self {
        Self { registry }
    }

    /// Validate that every reference in `references` resolves to an entry in
    /// the registry.
    ///
    /// References whose `source_namespace` matches their target model are
    /// already expected to have been filtered out by
    /// [`extract_cross_references`](CrossModelValidator::extract_cross_references);
    /// they are validated here as well so that the function is safe to call
    /// with any slice.
    pub fn validate(
        &self,
        source_namespace: &str,
        references: &[CrossModelReference],
    ) -> ValidationReport {
        let mut checked = Vec::with_capacity(references.len());
        let mut unresolved = Vec::new();

        for reference in references {
            // Skip references from a different source than the one we are
            // currently validating — the caller may have mixed references from
            // several sources; we only check what belongs to this namespace.
            if reference.source_namespace != source_namespace {
                continue;
            }

            checked.push(reference.clone());

            if self.registry.resolve_urn(&reference.target_urn).is_none() {
                unresolved.push(reference.clone());
            }
        }

        let resolved_count = checked.len() - unresolved.len();

        ValidationReport {
            checked_references: checked,
            unresolved,
            resolved_count,
        }
    }

    /// Extract all URN references from `properties` that point *outside*
    /// `source_namespace`.
    ///
    /// `properties` is a slice of `(context_label, urn_value)` pairs, where
    /// `context_label` is a human-readable description of where the reference
    /// appears (e.g. `"property 'speed' dataType"`).
    ///
    /// Only values that start with `urn:samm:` and whose model prefix (the
    /// part before `#`) differs from `source_namespace` are included in the
    /// result.  Non-URN values (literals, xsd types, etc.) are silently
    /// skipped.
    pub fn extract_cross_references(
        source_namespace: &str,
        properties: &[(&str, &str)],
    ) -> Vec<CrossModelReference> {
        properties
            .iter()
            .filter_map(|(ctx, urn)| {
                if !is_samm_urn(urn) {
                    return None;
                }
                // Derive the model prefix from the URN (everything before `#`).
                let target_ns = model_namespace(urn);
                if target_ns == source_namespace {
                    return None; // same-namespace reference — not cross-model
                }
                Some(CrossModelReference {
                    source_namespace: source_namespace.to_string(),
                    target_urn: urn.to_string(),
                    context: ctx.to_string(),
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Return the model-level namespace prefix for a SAMM URN.
///
/// For `urn:samm:org.example:1.0.0#TypeName` this returns
/// `"urn:samm:org.example:1.0.0"`.  For URNs without a `#` fragment the
/// whole string is returned unchanged.
fn model_namespace(urn: &str) -> &str {
    match urn.rfind('#') {
        Some(pos) => &urn[..pos],
        None => urn,
    }
}

/// `true` iff `value` looks like a SAMM URN (starts with `urn:samm:`).
fn is_samm_urn(value: &str) -> bool {
    value.starts_with("urn:samm:")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cross_model::{CrossModelRegistry, ModelEntry};

    fn registry_with(namespace: &str, urns: &[&str]) -> CrossModelRegistry {
        let mut reg = CrossModelRegistry::new();
        reg.register_model(ModelEntry {
            namespace: namespace.to_string(),
            file_path: None,
            exported_urns: urns.iter().map(|u| u.to_string()).collect(),
        })
        .unwrap();
        reg
    }

    // 8 — empty references → report is_valid, 0 unresolved.
    #[test]
    fn test_validate_no_cross_refs() {
        let registry = CrossModelRegistry::new();
        let validator = CrossModelValidator::new(&registry);

        let report = validator.validate("urn:samm:org.example.a:1.0.0", &[]);
        assert!(report.is_valid());
        assert_eq!(report.unresolved.len(), 0);
        assert_eq!(report.resolved_count, 0);
    }

    // 9 — all references point to known URNs → valid.
    #[test]
    fn test_validate_all_resolved() {
        let registry = registry_with(
            "urn:samm:org.example.b:1.0.0",
            &["urn:samm:org.example.b:1.0.0#SpeedChar"],
        );
        let validator = CrossModelValidator::new(&registry);

        let refs = vec![CrossModelReference {
            source_namespace: "urn:samm:org.example.a:1.0.0".to_string(),
            target_urn: "urn:samm:org.example.b:1.0.0#SpeedChar".to_string(),
            context: "property 'speed' dataType".to_string(),
        }];

        let report = validator.validate("urn:samm:org.example.a:1.0.0", &refs);
        assert!(report.is_valid());
        assert_eq!(report.resolved_count, 1);
        assert!(report.unresolved.is_empty());
    }

    // 10 — one reference points to unknown URN → not valid, 1 unresolved.
    #[test]
    fn test_validate_unresolved_ref() {
        let registry = CrossModelRegistry::new(); // empty — nothing registered
        let validator = CrossModelValidator::new(&registry);

        let refs = vec![CrossModelReference {
            source_namespace: "urn:samm:org.example.a:1.0.0".to_string(),
            target_urn: "urn:samm:org.example.b:1.0.0#GhostType".to_string(),
            context: "property 'ghost' characteristic".to_string(),
        }];

        let report = validator.validate("urn:samm:org.example.a:1.0.0", &refs);
        assert!(!report.is_valid());
        assert_eq!(report.unresolved.len(), 1);
        assert_eq!(
            report.unresolved[0].target_urn,
            "urn:samm:org.example.b:1.0.0#GhostType"
        );
    }

    // 11 — refs in the same namespace as source are excluded from the result.
    #[test]
    fn test_extract_cross_references_filters_same_namespace() {
        let source = "urn:samm:org.example.a:1.0.0";
        let props = &[
            (
                "localProp dataType",
                "urn:samm:org.example.a:1.0.0#LocalChar",
            ),
            ("description", "not a samm urn"),
        ];

        let refs = CrossModelValidator::extract_cross_references(source, props);
        assert!(
            refs.is_empty(),
            "same-namespace and non-URN values must be filtered out"
        );
    }

    // 12 — refs in a different namespace are included.
    #[test]
    fn test_extract_cross_references_includes_external() {
        let source = "urn:samm:org.example.a:1.0.0";
        let props = &[
            (
                "temperature dataType",
                "urn:samm:org.example.b:1.0.0#TemperatureChar",
            ),
            ("local", "urn:samm:org.example.a:1.0.0#LocalChar"),
            ("literal value", "some-plain-string"),
        ];

        let refs = CrossModelValidator::extract_cross_references(source, props);
        assert_eq!(refs.len(), 1);
        assert_eq!(
            refs[0].target_urn,
            "urn:samm:org.example.b:1.0.0#TemperatureChar"
        );
        assert_eq!(refs[0].context, "temperature dataType");
    }

    // 13 — summary string is non-empty and reflects validity.
    #[test]
    fn test_validation_report_summary() {
        let registry = CrossModelRegistry::new();
        let validator = CrossModelValidator::new(&registry);
        let report = validator.validate("urn:samm:org.example.a:1.0.0", &[]);

        let summary = report.summary();
        assert!(!summary.is_empty(), "summary must not be blank");
        // With no references the report is valid → summary should say "passed"
        assert!(
            summary.contains("passed"),
            "empty-reference report summary should say 'passed': {summary}"
        );

        // Now a failing report.
        let failing_refs = vec![CrossModelReference {
            source_namespace: "urn:samm:org.example.a:1.0.0".to_string(),
            target_urn: "urn:samm:org.example.z:1.0.0#Missing".to_string(),
            context: "ctx".to_string(),
        }];
        let failing_report = validator.validate("urn:samm:org.example.a:1.0.0", &failing_refs);
        let failing_summary = failing_report.summary();
        assert!(
            failing_summary.contains("FAILED"),
            "failing summary should contain 'FAILED': {failing_summary}"
        );
    }

    // 14 — validating two sources independently works correctly.
    #[test]
    fn test_validate_multiple_sources() {
        // Model B exports two types; model C is unknown.
        let mut registry = CrossModelRegistry::new();
        registry
            .register_model(ModelEntry {
                namespace: "urn:samm:org.example.b:1.0.0".to_string(),
                file_path: None,
                exported_urns: vec![
                    "urn:samm:org.example.b:1.0.0#TypeX".to_string(),
                    "urn:samm:org.example.b:1.0.0#TypeY".to_string(),
                ],
            })
            .unwrap();

        let validator = CrossModelValidator::new(&registry);

        // Source A: references TypeX in B (resolved) and Missing in C (unresolved).
        let refs_a = vec![
            CrossModelReference {
                source_namespace: "urn:samm:org.example.a:1.0.0".to_string(),
                target_urn: "urn:samm:org.example.b:1.0.0#TypeX".to_string(),
                context: "a→b TypeX".to_string(),
            },
            CrossModelReference {
                source_namespace: "urn:samm:org.example.a:1.0.0".to_string(),
                target_urn: "urn:samm:org.example.c:1.0.0#Missing".to_string(),
                context: "a→c Missing".to_string(),
            },
        ];

        // Source D: references TypeY in B (resolved).
        let refs_d = vec![CrossModelReference {
            source_namespace: "urn:samm:org.example.d:1.0.0".to_string(),
            target_urn: "urn:samm:org.example.b:1.0.0#TypeY".to_string(),
            context: "d→b TypeY".to_string(),
        }];

        let report_a = validator.validate("urn:samm:org.example.a:1.0.0", &refs_a);
        assert!(!report_a.is_valid(), "A should have 1 unresolved");
        assert_eq!(report_a.unresolved.len(), 1);
        assert_eq!(report_a.resolved_count, 1);

        let report_d = validator.validate("urn:samm:org.example.d:1.0.0", &refs_d);
        assert!(report_d.is_valid(), "D should fully resolve");
        assert_eq!(report_d.resolved_count, 1);
    }
}
