//! SAMM to JSON-LD Generator
//!
//! Generates JSON-LD (Linked Data) from SAMM Aspect models.
//! Provides RDF serialization with @context and semantic references.

use crate::error::SammError;
use crate::metamodel::{Aspect, ModelElement};

/// Generate JSON-LD from SAMM Aspect
pub fn generate_jsonld(aspect: &Aspect) -> Result<String, SammError> {
    let aspect_name = aspect.name();
    let aspect_urn = &aspect.metadata.urn;

    let mut jsonld = String::new();
    jsonld.push_str("{\n");
    jsonld.push_str("  \"@context\": {\n");

    // SAMM namespace context
    jsonld.push_str("    \"samm\": \"urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#\",\n");
    jsonld.push_str("    \"samm-c\": \"urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#\",\n");
    jsonld.push_str("    \"samm-e\": \"urn:samm:org.eclipse.esmf.samm:entity:2.3.0#\",\n");
    jsonld.push_str("    \"xsd\": \"http://www.w3.org/2001/XMLSchema#\",\n");
    jsonld.push_str("    \"rdfs\": \"http://www.w3.org/2000/01/rdf-schema#\",\n");

    // Aspect-specific context
    jsonld.push_str(&format!("    \"aspect\": \"{}\",\n", aspect_urn));

    // Property contexts
    for (i, prop) in aspect.properties().iter().enumerate() {
        let prop_urn = &prop.metadata.urn;
        let comma = if i == aspect.properties().len() - 1
            && aspect.operations().is_empty()
            && aspect.events().is_empty()
        {
            ""
        } else {
            ","
        };
        jsonld.push_str(&format!(
            "    \"{}\": \"{}\"{}\n",
            to_camel_case(&prop.name()),
            prop_urn,
            comma
        ));
    }

    // Operation contexts
    for (i, op) in aspect.operations().iter().enumerate() {
        let op_urn = &op.metadata.urn;
        let comma = if i == aspect.operations().len() - 1 && aspect.events().is_empty() {
            ""
        } else {
            ","
        };
        jsonld.push_str(&format!(
            "    \"{}\": \"{}\"{}\n",
            to_camel_case(&op.name()),
            op_urn,
            comma
        ));
    }

    // Event contexts
    for (i, event) in aspect.events().iter().enumerate() {
        let event_urn = &event.metadata.urn;
        let comma = if i == aspect.events().len() - 1 {
            ""
        } else {
            ","
        };
        jsonld.push_str(&format!(
            "    \"{}\": \"{}\"{}\n",
            to_camel_case(&event.name()),
            event_urn,
            comma
        ));
    }

    jsonld.push_str("  },\n");

    // Aspect data
    jsonld.push_str(&format!("  \"@id\": \"{}\",\n", aspect_urn));
    jsonld.push_str("  \"@type\": \"samm:Aspect\",\n");
    jsonld.push_str(&format!(
        "  \"samm:name\": \"{}\",\n",
        aspect
            .metadata
            .get_preferred_name("en")
            .unwrap_or(&aspect_name)
    ));

    if let Some(desc) = aspect.metadata.get_description("en") {
        jsonld.push_str(&format!("  \"samm:description\": \"{}\",\n", desc));
    }

    // Properties
    if !aspect.properties().is_empty() {
        jsonld.push_str("  \"samm:properties\": [\n");
        for (i, prop) in aspect.properties().iter().enumerate() {
            jsonld.push_str("    {\n");
            jsonld.push_str(&format!("      \"@id\": \"{}\",\n", prop.metadata.urn));
            jsonld.push_str("      \"@type\": \"samm:Property\",\n");
            jsonld.push_str(&format!("      \"samm:name\": \"{}\",\n", prop.name()));

            if let Some(char) = &prop.characteristic {
                if let Some(dt) = &char.data_type {
                    jsonld.push_str(&format!(
                        "      \"samm:dataType\": \"{}\",\n",
                        dt.split('#').next_back().unwrap_or("string")
                    ));
                }
            }

            jsonld.push_str(&format!("      \"samm:optional\": {}", prop.optional));

            let comma = if i < aspect.properties().len() - 1 {
                ","
            } else {
                ""
            };
            jsonld.push_str(&format!("\n    }}{}\n", comma));
        }
        jsonld.push_str("  ]");

        if !aspect.operations().is_empty() || !aspect.events().is_empty() {
            jsonld.push(',');
        }
        jsonld.push('\n');
    }

    // Operations
    if !aspect.operations().is_empty() {
        jsonld.push_str("  \"samm:operations\": [\n");
        for (i, op) in aspect.operations().iter().enumerate() {
            jsonld.push_str("    {\n");
            jsonld.push_str(&format!("      \"@id\": \"{}\",\n", op.metadata.urn));
            jsonld.push_str("      \"@type\": \"samm:Operation\",\n");
            jsonld.push_str(&format!("      \"samm:name\": \"{}\"", op.name()));

            let comma = if i < aspect.operations().len() - 1 {
                ","
            } else {
                ""
            };
            jsonld.push_str(&format!("\n    }}{}\n", comma));
        }
        jsonld.push_str("  ]");

        if !aspect.events().is_empty() {
            jsonld.push(',');
        }
        jsonld.push('\n');
    }

    // Events
    if !aspect.events().is_empty() {
        jsonld.push_str("  \"samm:events\": [\n");
        for (i, event) in aspect.events().iter().enumerate() {
            jsonld.push_str("    {\n");
            jsonld.push_str(&format!("      \"@id\": \"{}\",\n", event.metadata.urn));
            jsonld.push_str("      \"@type\": \"samm:Event\",\n");
            jsonld.push_str(&format!("      \"samm:name\": \"{}\"", event.name()));

            let comma = if i < aspect.events().len() - 1 {
                ","
            } else {
                ""
            };
            jsonld.push_str(&format!("\n    }}{}\n", comma));
        }
        jsonld.push_str("  ]\n");
    }

    jsonld.push_str("}\n");
    Ok(jsonld)
}

/// Convert snake_case to camelCase
fn to_camel_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;

    for ch in s.chars() {
        if ch == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(
                ch.to_uppercase()
                    .next()
                    .expect("uppercase should produce a character"),
            );
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camel_case_conversion() {
        assert_eq!(to_camel_case("movement_aspect"), "movementAspect");
        assert_eq!(to_camel_case("position"), "position");
        assert_eq!(to_camel_case("current_speed"), "currentSpeed");
    }
}
