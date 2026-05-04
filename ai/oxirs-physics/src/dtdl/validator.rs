//! DTDL v3 Semantic Validator
//!
//! Validates a parsed [`DtdlInterface`] against DTDL v3 semantic constraints
//! that cannot be expressed purely in the JSON schema:
//!
//! - The `@id` DTMI must be syntactically valid.
//! - The `@type` must resolve to `"Interface"`.
//! - All content element names must be non-empty.
//! - Content elements that carry their own `@id` must have valid DTMIs.
//! - Relationship `target` DTMIs must be valid when present.
//! - Component `schema` DTMIs must be valid.
//!
//! The validator returns a `Vec<DtdlValidationError>` so that all issues are
//! reported in one pass rather than failing on the first error.

use super::types::{primary_type, DtdlContent, DtdlInterface, DtdlValidationError};

/// Validate a parsed [`DtdlInterface`] against DTDL v3 constraints.
///
/// Returns all validation errors found; an empty `Vec` means the interface
/// is structurally valid per DTDL v3.
pub fn validate(iface: &DtdlInterface) -> Vec<DtdlValidationError> {
    let mut errors = Vec::new();

    // 1. DTMI validity
    if let Err(e) = iface.id.validate() {
        errors.push(e);
    }

    // 2. @type must be (or include) "Interface"
    if primary_type(&iface.element_type) != Some("Interface") {
        errors.push(DtdlValidationError::MissingField {
            field: "@type must be 'Interface'",
        });
    }

    // 3. Content element validation
    for content in iface.contents.as_deref().unwrap_or(&[]) {
        validate_content(content, &mut errors);
    }

    errors
}

fn validate_content(content: &DtdlContent, errors: &mut Vec<DtdlValidationError>) {
    match content {
        DtdlContent::Telemetry(t) => {
            if t.name.trim().is_empty() {
                errors.push(DtdlValidationError::MissingField {
                    field: "Telemetry.name must not be empty",
                });
            }
            if let Some(id) = &t.id {
                if let Err(e) = id.validate() {
                    errors.push(e);
                }
            }
        }

        DtdlContent::Property(p) => {
            if p.name.trim().is_empty() {
                errors.push(DtdlValidationError::MissingField {
                    field: "Property.name must not be empty",
                });
            }
            if let Some(id) = &p.id {
                if let Err(e) = id.validate() {
                    errors.push(e);
                }
            }
        }

        DtdlContent::Command(c) => {
            if c.name.trim().is_empty() {
                errors.push(DtdlValidationError::MissingField {
                    field: "Command.name must not be empty",
                });
            }
        }

        DtdlContent::Component(comp) => {
            if comp.name.trim().is_empty() {
                errors.push(DtdlValidationError::MissingField {
                    field: "Component.name must not be empty",
                });
            }
            if let Err(e) = comp.schema.validate() {
                errors.push(e);
            }
        }

        DtdlContent::Relationship(rel) => {
            if rel.name.trim().is_empty() {
                errors.push(DtdlValidationError::MissingField {
                    field: "Relationship.name must not be empty",
                });
            }
            if let Some(target) = &rel.target {
                if let Err(e) = target.validate() {
                    errors.push(e);
                }
            }
        }
    }
}

/// Check whether an interface passes all validation constraints.
///
/// Convenience wrapper over [`validate`] that returns `true` iff there are
/// no errors.
pub fn is_valid(iface: &DtdlInterface) -> bool {
    validate(iface).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtdl::parser::parse_dtdl_interface;
    use crate::dtdl::types::Dtmi;

    #[test]
    fn valid_minimal_interface() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Valid;1"
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let errors = validate(&iface);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn invalid_dtmi_caught() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Valid;1"
        }"#;
        let mut iface = parse_dtdl_interface(json).expect("parse");
        iface.id = Dtmi("bad-id".into());
        let errors = validate(&iface);
        assert!(!errors.is_empty(), "expected DTMI error");
        assert!(errors
            .iter()
            .any(|e| matches!(e, DtdlValidationError::InvalidDtmi { .. })));
    }

    #[test]
    fn valid_telemetry_property_command() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Full;1",
            "contents": [
                { "@type": "Telemetry", "name": "temp", "schema": "double" },
                { "@type": "Property", "name": "target", "schema": "double", "writable": true },
                { "@type": "Command", "name": "reset" }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let errors = validate(&iface);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn is_valid_wrapper() {
        let json = r#"{"@type":"Interface","@id":"dtmi:t:X;1"}"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        assert!(is_valid(&iface));
    }

    #[test]
    fn valid_relationship_with_target() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Building;1",
            "contents": [
                { "@type": "Relationship", "name": "contains", "target": "dtmi:test:Room;1" }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let errors = validate(&iface);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn valid_component_with_dtmi_schema() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Building;1",
            "contents": [
                { "@type": "Component", "name": "thermostat", "schema": "dtmi:test:Thermo;1" }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let errors = validate(&iface);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn multiple_content_elements_all_validated() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Multi;1",
            "contents": [
                { "@type": "Telemetry", "name": "a", "schema": "double" },
                { "@type": "Property", "name": "b", "schema": "string" },
                { "@type": "Command", "name": "c" },
                { "@type": "Relationship", "name": "d" }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let errors = validate(&iface);
        assert!(errors.is_empty(), "{errors:?}");
    }
}
