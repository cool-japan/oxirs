//! SPARQL query parser
//!
//! This is a placeholder implementation that will be enhanced with full
//! SPARQL 1.1 parsing capabilities in future iterations.

use crate::model::{BlankNode, Literal, NamedNode, Variable};
use crate::query::algebra::{AlgebraTriplePattern, TermPattern as AlgebraTermPattern};
use crate::query::sparql_algebra::{GraphPattern, TermPattern, TriplePattern};
use crate::query::sparql_query::Query;
use crate::OxirsError;
use std::collections::HashMap;

/// A SPARQL parser
#[derive(Debug, Clone, Default)]
pub struct SparqlParser {
    base_iri: Option<NamedNode>,
    prefixes: HashMap<String, NamedNode>,
}

impl SparqlParser {
    /// Creates a new SPARQL parser
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the base IRI for resolving relative IRIs
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, OxirsError> {
        self.base_iri = Some(NamedNode::new(base_iri.into())?);
        Ok(self)
    }

    /// Adds a prefix mapping
    pub fn with_prefix(
        mut self,
        prefix: impl Into<String>,
        iri: impl Into<String>,
    ) -> Result<Self, OxirsError> {
        self.prefixes
            .insert(prefix.into(), NamedNode::new(iri.into())?);
        Ok(self)
    }

    /// Parses a SPARQL query string - alias for parse_query
    pub fn parse(&self, query: &str) -> Result<Query, OxirsError> {
        self.parse_query(query)
    }

    /// Parses a SPARQL query string
    pub fn parse_query(&self, query: &str) -> Result<Query, OxirsError> {
        // This is a simplified parser for demonstration
        // Full implementation would use a proper parser generator

        let query = query.trim();

        // Very basic SELECT query detection
        if query.to_uppercase().starts_with("SELECT") {
            self.parse_select_query(query)
        } else if query.to_uppercase().starts_with("CONSTRUCT") {
            self.parse_construct_query(query)
        } else if query.to_uppercase().starts_with("ASK") {
            self.parse_ask_query(query)
        } else if query.to_uppercase().starts_with("DESCRIBE") {
            self.parse_describe_query(query)
        } else {
            Err(OxirsError::Parse(
                "Unsupported query form. Query must start with SELECT, CONSTRUCT, ASK, or DESCRIBE"
                    .to_string(),
            ))
        }
    }

    // Private helper methods for parsing different query forms

    fn parse_select_query(&self, query: &str) -> Result<Query, OxirsError> {
        // Extract WHERE clause (simplified parsing)
        let where_start = query
            .to_uppercase()
            .find("WHERE")
            .ok_or_else(|| OxirsError::Parse("SELECT query must have WHERE clause".to_string()))?;

        // Parse WHERE clause (simplified - just extract triple patterns)
        let pattern = self.parse_where_clause(&query[where_start + 5..])?;

        Ok(Query::Select {
            dataset: None,
            pattern,
            base_iri: self.base_iri.as_ref().map(|iri| iri.as_str().to_string()),
        })
    }

    fn parse_construct_query(&self, query: &str) -> Result<Query, OxirsError> {
        // Find CONSTRUCT template and WHERE clause
        let construct_start = query.to_uppercase().find("CONSTRUCT").unwrap() + 9;
        let where_start = query.to_uppercase().find("WHERE").ok_or_else(|| {
            OxirsError::Parse("CONSTRUCT query must have WHERE clause".to_string())
        })?;

        // Parse template (simplified - just get the content between braces)
        let construct_clause = query[construct_start..where_start].trim();
        let algebra_template = self.parse_construct_template(construct_clause)?;
        let template: Vec<TriplePattern> = algebra_template
            .iter()
            .map(|p| self.convert_triple_pattern(p))
            .collect();

        // Parse WHERE clause
        let pattern = self.parse_where_clause(&query[where_start + 5..])?;

        Ok(Query::Construct {
            template,
            dataset: None,
            pattern,
            base_iri: self.base_iri.as_ref().map(|iri| iri.as_str().to_string()),
        })
    }

    fn parse_ask_query(&self, query: &str) -> Result<Query, OxirsError> {
        let where_start = query
            .to_uppercase()
            .find("WHERE")
            .ok_or_else(|| OxirsError::Parse("ASK query must have WHERE clause".to_string()))?;

        let pattern = self.parse_where_clause(&query[where_start + 5..])?;

        Ok(Query::Ask {
            dataset: None,
            pattern,
            base_iri: self.base_iri.as_ref().map(|iri| iri.as_str().to_string()),
        })
    }

    fn parse_describe_query(&self, query: &str) -> Result<Query, OxirsError> {
        let where_start = query.to_uppercase().find("WHERE").ok_or_else(|| {
            OxirsError::Parse("DESCRIBE query must have WHERE clause".to_string())
        })?;

        let pattern = self.parse_where_clause(&query[where_start + 5..])?;

        Ok(Query::Describe {
            dataset: None,
            pattern,
            base_iri: self.base_iri.as_ref().map(|iri| iri.as_str().to_string()),
        })
    }

    fn parse_construct_template(
        &self,
        template_text: &str,
    ) -> Result<Vec<AlgebraTriplePattern>, OxirsError> {
        let content = template_text.trim();
        if !content.starts_with('{') || !content.ends_with('}') {
            return Err(OxirsError::Parse(
                "CONSTRUCT template must be enclosed in {}".to_string(),
            ));
        }

        let content = content[1..content.len() - 1].trim();
        let mut triple_patterns: Vec<AlgebraTriplePattern> = Vec::new();

        // Split by periods (very naive approach)
        for triple_str in content.split('.') {
            let triple_str = triple_str.trim();
            if triple_str.is_empty() {
                continue;
            }

            // Parse triple pattern (subject predicate object)
            let parts: Vec<&str> = triple_str.split_whitespace().collect();
            if parts.len() != 3 {
                return Err(OxirsError::Parse(format!(
                    "Invalid triple pattern: '{triple_str}'"
                )));
            }

            let subject = self.parse_term_pattern(parts[0])?;
            let predicate = self.parse_term_pattern(parts[1])?;
            let object = self.parse_term_pattern(parts[2])?;

            // Validate subject pattern (literals can't be subjects)
            if matches!(subject, TermPattern::Literal(_)) {
                return Err(OxirsError::Parse("Literals cannot be subjects".to_string()));
            }

            // Validate predicate pattern (only named nodes and variables allowed)
            if !matches!(
                predicate,
                TermPattern::NamedNode(_) | TermPattern::Variable(_)
            ) {
                return Err(OxirsError::Parse(
                    "Predicates must be named nodes or variables".to_string(),
                ));
            }

            // Convert sparql_algebra::TermPattern to algebra::TermPattern
            let algebra_subject = self.convert_to_algebra_term(&subject)?;
            let algebra_predicate = self.convert_to_algebra_term(&predicate)?;
            let algebra_object = self.convert_to_algebra_term(&object)?;

            triple_patterns.push(AlgebraTriplePattern::new(
                algebra_subject,
                algebra_predicate,
                algebra_object,
            ));
        }

        Ok(triple_patterns)
    }

    // Helper method to convert sparql_algebra::TermPattern to algebra::TermPattern
    fn convert_to_algebra_term(
        &self,
        term: &TermPattern,
    ) -> Result<AlgebraTermPattern, OxirsError> {
        match term {
            TermPattern::NamedNode(n) => Ok(AlgebraTermPattern::NamedNode(n.clone())),
            TermPattern::BlankNode(b) => Ok(AlgebraTermPattern::BlankNode(b.clone())),
            TermPattern::Literal(l) => Ok(AlgebraTermPattern::Literal(l.clone())),
            TermPattern::Variable(v) => Ok(AlgebraTermPattern::Variable(v.clone())),
            #[cfg(feature = "sparql-12")]
            TermPattern::Triple(_) => Err(OxirsError::Parse(
                "Quoted triples not supported in construct templates".to_string(),
            )),
        }
    }

    fn parse_where_clause(&self, where_text: &str) -> Result<GraphPattern, OxirsError> {
        // Very simplified parsing - just extract basic triple patterns
        let content = where_text.trim();
        if !content.starts_with('{') || !content.ends_with('}') {
            return Err(OxirsError::Parse(
                "WHERE clause must be enclosed in {}".to_string(),
            ));
        }

        let content = content[1..content.len() - 1].trim();
        let mut triple_patterns: Vec<TriplePattern> = Vec::new();

        // Split by periods (very naive approach)
        for triple_str in content.split('.') {
            let triple_str = triple_str.trim();
            if triple_str.is_empty() {
                continue;
            }

            // Parse triple pattern (subject predicate object)
            let parts: Vec<&str> = triple_str.split_whitespace().collect();
            if parts.len() != 3 {
                return Err(OxirsError::Parse(format!(
                    "Invalid triple pattern: '{triple_str}'"
                )));
            }

            let subject = self.parse_term_pattern(parts[0])?;
            let predicate = self.parse_term_pattern(parts[1])?;
            let object = self.parse_term_pattern(parts[2])?;

            triple_patterns.push(TriplePattern::new(subject, predicate, object));
        }

        Ok(GraphPattern::Bgp {
            patterns: triple_patterns,
        })
    }

    fn parse_term_pattern(&self, term: &str) -> Result<TermPattern, OxirsError> {
        if term.starts_with('?') || term.starts_with('$') {
            Variable::new(term).map(TermPattern::Variable)
        } else if term.starts_with('<') && term.ends_with('>') {
            let iri = &term[1..term.len() - 1];
            NamedNode::new(iri).map(TermPattern::NamedNode)
        } else if term.starts_with('"') && term.ends_with('"') {
            let value = &term[1..term.len() - 1];
            Ok(TermPattern::Literal(Literal::new(value)))
        } else if term.starts_with("_:") {
            BlankNode::new(term).map(TermPattern::BlankNode)
        } else if let Some(colon_pos) = term.find(':') {
            // Prefixed name
            let prefix = &term[..colon_pos];
            let local = &term[colon_pos + 1..];

            if let Some(namespace) = self.prefixes.get(prefix) {
                let iri = format!("{}{}", namespace.as_str(), local);
                NamedNode::new(iri).map(TermPattern::NamedNode)
            } else {
                Err(OxirsError::Parse(format!("Unknown prefix: {prefix}")))
            }
        } else {
            Err(OxirsError::Parse(format!("Invalid term pattern: {term}")))
        }
    }

    /// Convert algebra TermPattern to sparql_algebra TermPattern
    fn convert_term_pattern(&self, term: &AlgebraTermPattern) -> TermPattern {
        match term {
            AlgebraTermPattern::NamedNode(n) => TermPattern::NamedNode(n.clone()),
            AlgebraTermPattern::BlankNode(b) => TermPattern::BlankNode(b.clone()),
            AlgebraTermPattern::Literal(l) => TermPattern::Literal(l.clone()),
            AlgebraTermPattern::Variable(v) => TermPattern::Variable(v.clone()),
        }
    }

    /// Convert AlgebraTriplePattern to sparql_algebra TriplePattern
    fn convert_triple_pattern(&self, pattern: &AlgebraTriplePattern) -> TriplePattern {
        TriplePattern::new(
            self.convert_term_pattern(&pattern.subject),
            self.convert_term_pattern(&pattern.predicate),
            self.convert_term_pattern(&pattern.object),
        )
    }

    /// Convert sparql_algebra TermPattern back to algebra TermPattern
    fn convert_term_pattern_back(&self, term: &TermPattern) -> AlgebraTermPattern {
        match term {
            TermPattern::NamedNode(n) => AlgebraTermPattern::NamedNode(n.clone()),
            TermPattern::BlankNode(b) => AlgebraTermPattern::BlankNode(b.clone()),
            TermPattern::Literal(l) => AlgebraTermPattern::Literal(l.clone()),
            TermPattern::Variable(v) => AlgebraTermPattern::Variable(v.clone()),
            #[cfg(feature = "sparql-12")]
            TermPattern::Triple(_) => {
                // Triple patterns in term position are not yet supported
                todo!("Triple patterns in term position are not yet fully implemented")
            }
        }
    }

    /// Convert sparql_algebra TriplePattern back to AlgebraTriplePattern
    pub fn convert_triple_pattern_back(&self, pattern: &TriplePattern) -> AlgebraTriplePattern {
        AlgebraTriplePattern::new(
            self.convert_term_pattern_back(&pattern.subject),
            self.convert_term_pattern_back(&pattern.predicate),
            self.convert_term_pattern_back(&pattern.object),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select_query() {
        let parser = SparqlParser::new();
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }";
        let result = parser.parse_query(query);
        assert!(result.is_ok());

        if let Ok(Query::Select { pattern, .. }) = result {
            match pattern {
                GraphPattern::Bgp { patterns } => {
                    assert_eq!(patterns.len(), 1);
                    // Verify it's a triple pattern with variables
                    let triple = &patterns[0];
                    assert!(matches!(triple.subject, TermPattern::Variable(_)));
                    assert!(matches!(triple.predicate, TermPattern::Variable(_)));
                    assert!(matches!(triple.object, TermPattern::Variable(_)));
                }
                _ => panic!("Expected BGP pattern"),
            }
        } else {
            panic!("Expected SELECT query");
        }
    }

    #[test]
    fn test_ask_query() {
        let parser = SparqlParser::new();
        let query = "ASK WHERE { ?s ?p ?o . }";
        let result = parser.parse_query(query);
        assert!(result.is_ok());

        if let Ok(Query::Ask { pattern, .. }) = result {
            match pattern {
                GraphPattern::Bgp { patterns } => {
                    assert_eq!(patterns.len(), 1);
                }
                _ => panic!("Expected BGP pattern"),
            }
        } else {
            panic!("Expected ASK query");
        }
    }

    #[test]
    fn test_construct_query() {
        let parser = SparqlParser::new();
        let query = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o . }";
        let result = parser.parse_query(query);
        assert!(result.is_ok());

        if let Ok(Query::Construct {
            template, pattern, ..
        }) = result
        {
            assert_eq!(template.len(), 1);
            match pattern {
                GraphPattern::Bgp { patterns } => {
                    assert_eq!(patterns.len(), 1);
                }
                _ => panic!("Expected BGP pattern"),
            }
        } else {
            panic!("Expected CONSTRUCT query");
        }
    }

    #[test]
    fn test_parse_with_prefix() {
        let parser = SparqlParser::new()
            .with_prefix("ex", "http://example.org/")
            .unwrap();

        let query = "SELECT ?s WHERE { ex:subject ?p ?o . }";
        let result = parser.parse_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_query() {
        let parser = SparqlParser::new();
        let query = "INVALID QUERY";
        let result = parser.parse_query(query);
        assert!(result.is_err());
    }
}
