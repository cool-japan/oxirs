//! RDF Dataset Canonicalization (RDFC-1.0)
//!
//! Simplified implementation of the W3C RDF Dataset Canonicalization Algorithm.
//!
//! See: <https://www.w3.org/TR/rdf-canon/>

use super::{RdfTerm, RdfTriple};
use crate::DidResult;
use std::collections::{BTreeMap, HashMap};

/// Canonicalize a set of RDF triples
pub fn canonicalize_graph(triples: &[RdfTriple]) -> DidResult<String> {
    let canonicalizer = RdfCanonicalizer::new();
    canonicalizer.canonicalize(triples)
}

/// RDF Canonicalizer
struct RdfCanonicalizer {
    /// Blank node identifier mapping
    blank_node_map: HashMap<String, String>,
    /// Counter for canonical blank node IDs
    blank_node_counter: usize,
}

impl RdfCanonicalizer {
    fn new() -> Self {
        Self {
            blank_node_map: HashMap::new(),
            blank_node_counter: 0,
        }
    }

    /// Canonicalize triples to N-Quads format
    fn canonicalize(&self, triples: &[RdfTriple]) -> DidResult<String> {
        // Step 1: Identify blank nodes and create canonical identifiers
        let mut canonicalizer = Self::new();

        // Collect all blank nodes
        for triple in triples {
            if let RdfTerm::BlankNode(id) = &triple.subject {
                canonicalizer.get_or_create_canonical_id(id);
            }
            if let RdfTerm::BlankNode(id) = &triple.object {
                canonicalizer.get_or_create_canonical_id(id);
            }
        }

        // Step 2: Serialize each triple in canonical form
        let mut nquads: Vec<String> = triples
            .iter()
            .map(|t| canonicalizer.triple_to_nquad(t))
            .collect();

        // Step 3: Sort lexicographically
        nquads.sort();

        // Step 4: Join with newlines
        Ok(nquads.join("\n"))
    }

    /// Get or create a canonical blank node ID
    fn get_or_create_canonical_id(&mut self, original_id: &str) -> String {
        if let Some(canonical) = self.blank_node_map.get(original_id) {
            canonical.clone()
        } else {
            let canonical = format!("_:c14n{}", self.blank_node_counter);
            self.blank_node_counter += 1;
            self.blank_node_map
                .insert(original_id.to_string(), canonical.clone());
            canonical
        }
    }

    /// Convert a triple to N-Quad format
    fn triple_to_nquad(&self, triple: &RdfTriple) -> String {
        let subject = self.term_to_nquad(&triple.subject);
        let predicate = format!("<{}>", triple.predicate);
        let object = self.term_to_nquad(&triple.object);

        format!("{} {} {} .", subject, predicate, object)
    }

    /// Convert an RDF term to N-Quad format
    fn term_to_nquad(&self, term: &RdfTerm) -> String {
        match term {
            RdfTerm::Iri(iri) => format!("<{}>", iri),
            RdfTerm::BlankNode(id) => self
                .blank_node_map
                .get(id)
                .cloned()
                .unwrap_or_else(|| format!("_:{}", id)),
            RdfTerm::Literal {
                value,
                datatype,
                language,
            } => {
                let escaped = escape_string(value);
                if let Some(lang) = language {
                    format!("\"{}\"@{}", escaped, lang)
                } else if let Some(dt) = datatype {
                    if dt == "http://www.w3.org/2001/XMLSchema#string" {
                        format!("\"{}\"", escaped)
                    } else {
                        format!("\"{}\"^^<{}>", escaped, dt)
                    }
                } else {
                    format!("\"{}\"", escaped)
                }
            }
        }
    }
}

/// Escape special characters in a string for N-Quads
fn escape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04X}", c as u32));
            }
            _ => result.push(c),
        }
    }
    result
}

/// Hash a canonical form using SHA-256
pub fn hash_canonical(canonical: &str) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(canonical.as_bytes());
    hex::encode(hash)
}

/// Full RDFC-1.0 implementation with hash
pub fn canonicalize_and_hash(triples: &[RdfTriple]) -> DidResult<(String, String)> {
    let canonical = canonicalize_graph(triples)?;
    let hash = hash_canonical(&canonical);
    Ok((canonical, hash))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonicalize_simple() {
        let triples = vec![RdfTriple::iri(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        )];

        let canonical = canonicalize_graph(&triples).unwrap();
        assert!(canonical.contains("<http://example.org/s>"));
        assert!(canonical.contains("<http://example.org/p>"));
        assert!(canonical.contains("<http://example.org/o>"));
    }

    #[test]
    fn test_canonicalize_sorting() {
        let triples = vec![
            RdfTriple::iri(
                "http://example.org/z",
                "http://example.org/p",
                "http://example.org/o",
            ),
            RdfTriple::iri(
                "http://example.org/a",
                "http://example.org/p",
                "http://example.org/o",
            ),
        ];

        let canonical = canonicalize_graph(&triples).unwrap();
        let lines: Vec<&str> = canonical.lines().collect();

        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("/a>"));
        assert!(lines[1].contains("/z>"));
    }

    #[test]
    fn test_canonicalize_literal() {
        let triples = vec![RdfTriple::literal(
            "http://example.org/s",
            "http://example.org/p",
            "Hello, World!",
            Some("http://www.w3.org/2001/XMLSchema#string"),
        )];

        let canonical = canonicalize_graph(&triples).unwrap();
        assert!(canonical.contains("\"Hello, World!\""));
    }

    #[test]
    fn test_escape_string() {
        assert_eq!(escape_string("hello"), "hello");
        assert_eq!(escape_string("hello\nworld"), "hello\\nworld");
        assert_eq!(escape_string("say \"hello\""), "say \\\"hello\\\"");
    }

    #[test]
    fn test_hash() {
        let canonical = "<http://a> <http://b> <http://c> .";
        let hash = hash_canonical(canonical);
        assert_eq!(hash.len(), 64); // SHA-256 hex = 64 chars
    }
}
