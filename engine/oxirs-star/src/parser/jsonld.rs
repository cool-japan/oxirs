//! JSON-LD-star processing utilities.
//!
//! This module provides functions for parsing JSON-LD-star format, an extension
//! of JSON-LD that supports RDF-star quoted triples and annotations.

use crate::model::{BlankNode, Literal, NamedNode, StarGraph, StarQuad, StarTerm, StarTriple};
use crate::{StarError, StarResult};

use super::context::ParseContext;

/// Process a JSON-LD-star value recursively.
///
/// This function handles JSON-LD values including objects, arrays, and primitives,
/// converting them into RDF-star triples and adding them to the graph.
///
/// # Arguments
///
/// * `value` - The JSON value to process
/// * `graph` - The RDF-star graph to populate
/// * `context` - The parsing context for prefix resolution and blank node generation
///
/// # Returns
///
/// `Ok(())` on success, or an error if processing fails
pub(super) fn process_jsonld_star_value(
    value: &serde_json::Value,
    graph: &mut StarGraph,
    context: &mut ParseContext,
) -> StarResult<()> {
    match value {
        serde_json::Value::Object(obj) => {
            process_jsonld_star_object(obj, graph, context)?;
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                process_jsonld_star_value(item, graph, context)?;
            }
        }
        _ => {
            // Skip primitive values at top level
        }
    }
    Ok(())
}

/// Process a JSON-LD-star object.
///
/// This function converts a JSON-LD object into RDF-star triples. It handles:
/// - `@id` for IRI subjects
/// - Blank node generation for objects without `@id`
/// - RDF-star annotations via `@annotation` extension
/// - Property values with appropriate type conversions
///
/// # Arguments
///
/// * `obj` - The JSON object to process
/// * `graph` - The RDF-star graph to populate
/// * `context` - The parsing context
///
/// # Returns
///
/// `Ok(())` on success, or an error if processing fails
pub(super) fn process_jsonld_star_object(
    obj: &serde_json::Map<String, serde_json::Value>,
    graph: &mut StarGraph,
    context: &mut ParseContext,
) -> StarResult<()> {
    // Check for quoted triple annotation (RDF-star extension)
    if obj.contains_key("@annotation") {
        return process_quoted_triple_annotation(obj, graph, context);
    }

    // Extract subject (either @id or generate blank node)
    let subject = if let Some(id_value) = obj.get("@id") {
        match id_value {
            serde_json::Value::String(id) => StarTerm::iri(id)?,
            _ => return Err(StarError::parse_error("@id must be a string".to_string())),
        }
    } else {
        StarTerm::BlankNode(BlankNode {
            id: context.next_blank_node(),
        })
    };

    // Process properties
    for (key, value) in obj {
        if key.starts_with('@') {
            // Skip JSON-LD keywords for now
            continue;
        }

        let predicate = StarTerm::iri(key)?;

        // Process property values
        process_property_values(&subject, &predicate, value, graph, context)?;
    }

    Ok(())
}

/// Process property values (which may be arrays or single values).
///
/// JSON-LD allows properties to have either single values or arrays of values.
/// This function handles both cases.
///
/// # Arguments
///
/// * `subject` - The subject term of the triple
/// * `predicate` - The predicate term of the triple
/// * `value` - The JSON value (single or array)
/// * `graph` - The RDF-star graph to populate
/// * `context` - The parsing context
///
/// # Returns
///
/// `Ok(())` on success, or an error if processing fails
pub(super) fn process_property_values(
    subject: &StarTerm,
    predicate: &StarTerm,
    value: &serde_json::Value,
    graph: &mut StarGraph,
    context: &mut ParseContext,
) -> StarResult<()> {
    match value {
        serde_json::Value::Array(arr) => {
            for item in arr {
                create_triple_from_value(subject, predicate, item, graph, context)?;
            }
        }
        _ => {
            create_triple_from_value(subject, predicate, value, graph, context)?;
        }
    }
    Ok(())
}

/// Create a triple from a JSON-LD value.
///
/// This function converts a JSON-LD value into an RDF-star object term and creates
/// a triple. It handles:
/// - String values (IRIs or plain literals)
/// - Objects with `@id`, `@value`, `@type`, `@language`
/// - Numbers and booleans with XSD datatypes
/// - Nested objects (creating blank nodes)
///
/// # Arguments
///
/// * `subject` - The subject term of the triple
/// * `predicate` - The predicate term of the triple
/// * `value` - The JSON value to convert to an object term
/// * `graph` - The RDF-star graph to populate
/// * `context` - The parsing context
///
/// # Returns
///
/// `Ok(())` on success, or an error if processing fails
pub(super) fn create_triple_from_value(
    subject: &StarTerm,
    predicate: &StarTerm,
    value: &serde_json::Value,
    graph: &mut StarGraph,
    context: &mut ParseContext,
) -> StarResult<()> {
    let object = match value {
        serde_json::Value::String(s) => {
            if s.starts_with("http://") || s.starts_with("https://") || s.contains(':') {
                StarTerm::iri(s)?
            } else {
                StarTerm::Literal(Literal {
                    value: s.clone(),
                    datatype: None,
                    language: None,
                })
            }
        }
        serde_json::Value::Object(obj) => {
            if let Some(id_value) = obj.get("@id") {
                // Reference to another resource
                if let serde_json::Value::String(id) = id_value {
                    StarTerm::iri(id)?
                } else {
                    return Err(StarError::parse_error("@id must be a string".to_string()));
                }
            } else if let Some(value_str) = obj.get("@value") {
                // Typed literal
                let value_str = match value_str {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    _ => return Err(StarError::parse_error("Invalid @value".to_string())),
                };

                let datatype = obj
                    .get("@type")
                    .and_then(|v| v.as_str())
                    .map(|s| NamedNode { iri: s.to_string() });

                let language = obj
                    .get("@language")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                StarTerm::Literal(Literal {
                    value: value_str,
                    language,
                    datatype,
                })
            } else {
                // Nested object - process recursively and return blank node
                let blank_node = StarTerm::BlankNode(BlankNode {
                    id: context.next_blank_node(),
                });
                process_jsonld_star_object(obj, graph, context)?;
                blank_node
            }
        }
        serde_json::Value::Number(n) => StarTerm::Literal(Literal {
            value: n.to_string(),
            datatype: Some(NamedNode {
                iri: "http://www.w3.org/2001/XMLSchema#decimal".to_string(),
            }),
            language: None,
        }),
        serde_json::Value::Bool(b) => StarTerm::Literal(Literal {
            value: b.to_string(),
            datatype: Some(NamedNode {
                iri: "http://www.w3.org/2001/XMLSchema#boolean".to_string(),
            }),
            language: None,
        }),
        _ => {
            return Err(StarError::parse_error(
                "Unsupported JSON value type".to_string(),
            ));
        }
    };

    // Create and insert quad
    let quad = StarQuad::new(subject.clone(), predicate.clone(), object, None);
    graph.insert_quad(quad)?;

    Ok(())
}

/// Process quoted triple annotation (RDF-star extension for JSON-LD).
///
/// This function handles the `@annotation` extension for JSON-LD-star, which allows
/// annotating quoted triples with metadata properties.
///
/// # Arguments
///
/// * `obj` - The JSON object containing the annotation
/// * `graph` - The RDF-star graph to populate
/// * `context` - The parsing context
///
/// # Returns
///
/// `Ok(())` on success, or an error if processing fails
pub(super) fn process_quoted_triple_annotation(
    obj: &serde_json::Map<String, serde_json::Value>,
    graph: &mut StarGraph,
    context: &mut ParseContext,
) -> StarResult<()> {
    let annotation = obj
        .get("@annotation")
        .ok_or_else(|| StarError::parse_error("Missing @annotation".to_string()))?;

    match annotation {
        serde_json::Value::Object(ann_obj) => {
            // Extract the quoted triple
            let subject_val = ann_obj.get("subject").ok_or_else(|| {
                StarError::parse_error("Missing subject in annotation".to_string())
            })?;
            let predicate_val = ann_obj.get("predicate").ok_or_else(|| {
                StarError::parse_error("Missing predicate in annotation".to_string())
            })?;
            let object_val = ann_obj.get("object").ok_or_else(|| {
                StarError::parse_error("Missing object in annotation".to_string())
            })?;

            let subject = json_value_to_star_term(subject_val)?;
            let predicate = json_value_to_star_term(predicate_val)?;
            let object = json_value_to_star_term(object_val)?;

            // Create quoted triple
            let quoted_triple =
                StarTerm::QuotedTriple(Box::new(StarTriple::new(subject, predicate, object)));

            // Process annotation properties
            for (prop_key, prop_value) in obj {
                if prop_key == "@annotation" {
                    continue;
                }

                let annotation_predicate = StarTerm::iri(prop_key)?;
                process_property_values(
                    &quoted_triple,
                    &annotation_predicate,
                    prop_value,
                    graph,
                    context,
                )?;
            }
        }
        _ => {
            return Err(StarError::parse_error(
                "@annotation must be an object".to_string(),
            ));
        }
    }

    Ok(())
}

/// Convert JSON value to StarTerm.
///
/// This function converts a JSON value into an RDF-star term. It handles:
/// - Blank nodes (strings starting with `_:`)
/// - IRIs (all other strings)
/// - Objects with `@id` or `@value`
///
/// # Arguments
///
/// * `value` - The JSON value to convert
///
/// # Returns
///
/// The converted StarTerm, or an error if conversion fails
pub(super) fn json_value_to_star_term(value: &serde_json::Value) -> StarResult<StarTerm> {
    match value {
        serde_json::Value::String(s) => {
            if s.starts_with("_:") {
                Ok(StarTerm::BlankNode(BlankNode { id: s.clone() }))
            } else {
                StarTerm::iri(s)
            }
        }
        serde_json::Value::Object(obj) => {
            if let Some(id) = obj.get("@id").and_then(|v| v.as_str()) {
                StarTerm::iri(id)
            } else if let Some(value) = obj.get("@value").and_then(|v| v.as_str()) {
                let datatype = obj
                    .get("@type")
                    .and_then(|v| v.as_str())
                    .map(|s| NamedNode { iri: s.to_string() });
                let language = obj
                    .get("@language")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                Ok(StarTerm::Literal(Literal {
                    value: value.to_string(),
                    datatype,
                    language,
                }))
            } else {
                Err(StarError::parse_error("Invalid object term".to_string()))
            }
        }
        _ => Err(StarError::parse_error("Invalid term value".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_value_to_star_term_iri() {
        let value = serde_json::Value::String("http://example.org/resource".to_string());
        let term = json_value_to_star_term(&value).unwrap();
        assert!(matches!(term, StarTerm::NamedNode(_)));
    }

    #[test]
    fn test_json_value_to_star_term_blank_node() {
        let value = serde_json::Value::String("_:b1".to_string());
        let term = json_value_to_star_term(&value).unwrap();
        assert!(matches!(term, StarTerm::BlankNode(_)));
    }

    #[test]
    fn test_json_value_to_star_term_literal() {
        let mut obj = serde_json::Map::new();
        obj.insert(
            "@value".to_string(),
            serde_json::Value::String("test".to_string()),
        );
        let value = serde_json::Value::Object(obj);
        let term = json_value_to_star_term(&value).unwrap();
        assert!(matches!(term, StarTerm::Literal(_)));
    }
}
