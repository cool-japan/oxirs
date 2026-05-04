//! Re-extract a [`PhysicsState`] from RDF triples.
//!
//! This is the symmetric counterpart to
//! `crate::sync::state_to_rdf::StateToRdfWriter`. After the writer has
//! materialised a state into the RDF graph, an external client may update
//! some of those properties; this extractor rebuilds the in-memory
//! [`PhysicsState`] so that the simulator can be re-initialised.
//!
//! The extractor takes a flat property map (typically obtained from a
//! SPARQL `CONSTRUCT` over the state graph) rather than a live
//! `RdfStore` to keep the module deterministic and easy to test.
//! `BidirectionalSync` plugs an RdfStore in front via a closure.

use std::collections::HashMap;

use crate::error::{PhysicsError, PhysicsResult};

use super::state_to_rdf::{PhysicsState, PhysicsStateValue};

/// One row of a typed property map produced by an external SPARQL query.
///
/// `predicate` is the local part of a `phys:*` predicate, `literal` is the
/// raw textual literal, `datatype` is the optional `xsd:*` datatype URI.
#[derive(Debug, Clone, PartialEq)]
pub struct RdfPropertyRow {
    /// Local predicate name (the suffix after `phys:`).
    pub predicate: String,
    /// Lexical form of the RDF literal.
    pub literal: String,
    /// Datatype URI (e.g. `http://www.w3.org/2001/XMLSchema#double`).
    pub datatype: Option<String>,
}

/// Re-extract a state from a list of RDF property rows.
#[derive(Debug, Clone, Default)]
pub struct RdfToStateExtractor;

/// Output of an extractor pass.
#[derive(Debug, Clone)]
pub struct RdfToStateOutput {
    /// Re-built physics state.
    pub state: PhysicsState,
    /// Predicates that the extractor could not parse and chose to skip.
    /// Tests assert this is empty for a healthy round-trip.
    pub skipped: Vec<String>,
}

impl RdfToStateExtractor {
    /// Construct a new extractor.
    pub fn new() -> Self {
        Self
    }

    /// Re-extract a state for `entity_iri` at `step` from `rows`.
    ///
    /// `rows` should contain only triples for the target state node; the
    /// caller is responsible for the SPARQL filtering.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::ParameterExtraction`] when `entity_iri` is
    /// empty.
    pub fn extract(
        &self,
        entity_iri: impl Into<String>,
        step: u64,
        rows: &[RdfPropertyRow],
    ) -> PhysicsResult<RdfToStateOutput> {
        let entity = entity_iri.into();
        if entity.is_empty() {
            return Err(PhysicsError::ParameterExtraction(
                "entity_iri must not be empty".to_string(),
            ));
        }
        let mut state = PhysicsState::new(&entity);
        state.step = step;

        let mut skipped = Vec::new();
        let mut values: HashMap<String, PhysicsStateValue> = HashMap::new();
        for row in rows {
            // Reserved core predicates injected by the writer — never make
            // it back into `values`.
            match row.predicate.as_str() {
                "step" | "timestamp" | "simulatesEntity" => continue,
                _ => {}
            }
            match parse_value(row) {
                Some(v) => {
                    values.insert(row.predicate.clone(), v);
                }
                None => skipped.push(row.predicate.clone()),
            }
        }
        state.values = values;

        Ok(RdfToStateOutput { state, skipped })
    }
}

fn parse_value(row: &RdfPropertyRow) -> Option<PhysicsStateValue> {
    let datatype = row.datatype.as_deref().unwrap_or("");
    if datatype.ends_with("#double")
        || datatype.ends_with("#float")
        || datatype.ends_with("#decimal")
    {
        return row
            .literal
            .parse::<f64>()
            .ok()
            .map(PhysicsStateValue::Scalar);
    }
    if datatype.ends_with("#integer") || datatype.ends_with("#int") || datatype.ends_with("#long") {
        return row
            .literal
            .parse::<i64>()
            .ok()
            .map(PhysicsStateValue::Integer);
    }
    if datatype.ends_with("#boolean") {
        return match row.literal.as_str() {
            "true" | "1" => Some(PhysicsStateValue::Bool(true)),
            "false" | "0" => Some(PhysicsStateValue::Bool(false)),
            _ => None,
        };
    }
    if datatype.ends_with("#string") || datatype.is_empty() {
        // Vectors are encoded as comma-separated strings by the writer; if
        // every comma-separated chunk parses as f64 we re-hydrate as a
        // vector, otherwise keep the original string.
        let trimmed = row.literal.trim();
        if trimmed.contains(',') {
            let parts: Result<Vec<f64>, _> = trimmed
                .split(',')
                .map(|s| s.trim().parse::<f64>())
                .collect();
            if let Ok(v) = parts {
                return Some(PhysicsStateValue::Vector(v));
            }
        }
        return Some(PhysicsStateValue::Text(row.literal.clone()));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn double_row(p: &str, lit: &str) -> RdfPropertyRow {
        RdfPropertyRow {
            predicate: p.to_string(),
            literal: lit.to_string(),
            datatype: Some("http://www.w3.org/2001/XMLSchema#double".to_string()),
        }
    }

    fn boolean_row(p: &str, lit: &str) -> RdfPropertyRow {
        RdfPropertyRow {
            predicate: p.to_string(),
            literal: lit.to_string(),
            datatype: Some("http://www.w3.org/2001/XMLSchema#boolean".to_string()),
        }
    }

    fn string_row(p: &str, lit: &str) -> RdfPropertyRow {
        RdfPropertyRow {
            predicate: p.to_string(),
            literal: lit.to_string(),
            datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
        }
    }

    fn integer_row(p: &str, lit: &str) -> RdfPropertyRow {
        RdfPropertyRow {
            predicate: p.to_string(),
            literal: lit.to_string(),
            datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
        }
    }

    #[test]
    fn extract_basic_state() {
        let rows = vec![
            double_row("temperature", "298.15"),
            double_row("voltage", "3.71"),
            boolean_row("is_charging", "true"),
            string_row("label", "running"),
            integer_row("cycles", "42"),
        ];
        let extractor = RdfToStateExtractor::new();
        let out = extractor
            .extract("urn:example:battery:001", 7, &rows)
            .expect("extract should succeed");
        assert_eq!(out.state.entity_iri, "urn:example:battery:001");
        assert_eq!(out.state.step, 7);
        assert_eq!(out.state.values.len(), 5);
        assert!(matches!(
            out.state.values.get("temperature"),
            Some(PhysicsStateValue::Scalar(_))
        ));
        assert!(matches!(
            out.state.values.get("is_charging"),
            Some(PhysicsStateValue::Bool(true))
        ));
        assert!(matches!(
            out.state.values.get("cycles"),
            Some(PhysicsStateValue::Integer(42))
        ));
        assert!(out.skipped.is_empty());
    }

    #[test]
    fn extract_rejects_empty_entity_iri() {
        let extractor = RdfToStateExtractor::new();
        let result = extractor.extract("", 0, &[]);
        assert!(matches!(result, Err(PhysicsError::ParameterExtraction(_))));
    }

    #[test]
    fn extract_skips_unparseable_double() {
        let rows = vec![double_row("voltage", "not-a-number")];
        let extractor = RdfToStateExtractor::new();
        let out = extractor
            .extract("urn:example:e", 0, &rows)
            .expect("extract should succeed");
        assert_eq!(out.skipped, vec!["voltage".to_string()]);
        assert!(out.state.values.is_empty());
    }

    #[test]
    fn extract_recovers_vector_from_csv_string() {
        let rows = vec![string_row("velocity", "1.0,2.0,3.0")];
        let out = RdfToStateExtractor::new()
            .extract("urn:example:e", 0, &rows)
            .expect("extract should succeed");
        match out.state.values.get("velocity") {
            Some(PhysicsStateValue::Vector(v)) => assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]),
            other => panic!("expected vector, got {other:?}"),
        }
    }

    #[test]
    fn extract_keeps_textual_string() {
        let rows = vec![string_row("status", "running")];
        let out = RdfToStateExtractor::new()
            .extract("urn:example:e", 0, &rows)
            .expect("extract should succeed");
        match out.state.values.get("status") {
            Some(PhysicsStateValue::Text(s)) => assert_eq!(s, "running"),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn extract_skips_core_predicates() {
        let rows = vec![
            integer_row("step", "12"),
            string_row("timestamp", "2026-04-30T00:00:00Z"),
            string_row("simulatesEntity", "urn:example:e"),
            double_row("voltage", "3.71"),
        ];
        let out = RdfToStateExtractor::new()
            .extract("urn:example:e", 12, &rows)
            .expect("extract should succeed");
        assert_eq!(out.state.values.len(), 1);
        assert!(out.state.values.contains_key("voltage"));
    }

    #[test]
    fn boolean_supports_alternate_lexical_forms() {
        let rows = vec![
            boolean_row("is_charging", "1"),
            boolean_row("is_running", "false"),
        ];
        let out = RdfToStateExtractor::new()
            .extract("urn:example:e", 0, &rows)
            .expect("extract should succeed");
        assert!(matches!(
            out.state.values.get("is_charging"),
            Some(PhysicsStateValue::Bool(true))
        ));
        assert!(matches!(
            out.state.values.get("is_running"),
            Some(PhysicsStateValue::Bool(false))
        ));
    }
}
