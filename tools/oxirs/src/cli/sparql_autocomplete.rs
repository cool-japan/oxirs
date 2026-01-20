//! SPARQL autocomplete module
//!
//! Provides context-aware autocomplete for SPARQL queries in interactive mode.

use super::completion::{CompletionContext, CompletionItem, CompletionProvider, CompletionType};
use std::collections::HashMap;

/// SPARQL keywords for autocomplete
const SPARQL_KEYWORDS: &[&str] = &[
    "SELECT",
    "DISTINCT",
    "REDUCED",
    "WHERE",
    "FILTER",
    "OPTIONAL",
    "UNION",
    "MINUS",
    "GRAPH",
    "SERVICE",
    "BIND",
    "VALUES",
    "LIMIT",
    "OFFSET",
    "ORDER",
    "BY",
    "GROUP",
    "HAVING",
    "ASC",
    "DESC",
    "PREFIX",
    "BASE",
    "ASK",
    "CONSTRUCT",
    "DESCRIBE",
    "INSERT",
    "DELETE",
    "LOAD",
    "CLEAR",
    "DROP",
    "CREATE",
    "ADD",
    "MOVE",
    "COPY",
    "WITH",
    "DATA",
    "SILENT",
    "DEFAULT",
    "NAMED",
    "ALL",
    "USING",
    "EXISTS",
    "NOT",
    "AS",
    "FROM",
];

/// SPARQL functions for autocomplete
const SPARQL_FUNCTIONS: &[&str] = &[
    // String functions
    "STR",
    "LANG",
    "LANGMATCHES",
    "DATATYPE",
    "STRLEN",
    "SUBSTR",
    "UCASE",
    "LCASE",
    "STRSTARTS",
    "STRENDS",
    "CONTAINS",
    "STRBEFORE",
    "STRAFTER",
    "ENCODE_FOR_URI",
    "CONCAT",
    "REPLACE",
    "STRLANG",
    "STRDT",
    "UUID",
    "STRUUID",
    // Numeric functions
    "ABS",
    "ROUND",
    "CEIL",
    "FLOOR",
    "RAND",
    // Date/Time functions
    "NOW",
    "YEAR",
    "MONTH",
    "DAY",
    "HOURS",
    "MINUTES",
    "SECONDS",
    "TIMEZONE",
    "TZ",
    // Hash functions
    "MD5",
    "SHA1",
    "SHA256",
    "SHA512",
    // Type checking
    "isIRI",
    "isURI",
    "isBLANK",
    "isLITERAL",
    "isNUMERIC",
    "BOUND",
    // Logical functions
    "IF",
    "COALESCE",
    "sameTerm",
    "IN",
    "NOT IN",
    // Aggregate functions
    "COUNT",
    "SUM",
    "MIN",
    "MAX",
    "AVG",
    "SAMPLE",
    "GROUP_CONCAT",
    // SPARQL 1.1 functions
    "REGEX",
];

/// Common RDF prefixes
const COMMON_PREFIXES: &[(&str, &str)] = &[
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("foaf", "http://xmlns.com/foaf/0.1/"),
    ("dc", "http://purl.org/dc/elements/1.1/"),
    ("dcterms", "http://purl.org/dc/terms/"),
    ("skos", "http://www.w3.org/2004/02/skos/core#"),
    ("schema", "http://schema.org/"),
    ("dcat", "http://www.w3.org/ns/dcat#"),
    ("void", "http://rdfs.org/ns/void#"),
    ("vcard", "http://www.w3.org/2006/vcard/ns#"),
    ("prov", "http://www.w3.org/ns/prov#"),
    ("time", "http://www.w3.org/2006/time#"),
    ("geo", "http://www.w3.org/2003/01/geo/wgs84_pos#"),
];

/// Common RDF properties
const COMMON_PROPERTIES: &[&str] = &[
    "rdf:type",
    "rdfs:label",
    "rdfs:comment",
    "rdfs:seeAlso",
    "rdfs:isDefinedBy",
    "rdfs:subClassOf",
    "rdfs:subPropertyOf",
    "rdfs:domain",
    "rdfs:range",
    "owl:sameAs",
    "owl:equivalentClass",
    "owl:equivalentProperty",
    "foaf:name",
    "foaf:knows",
    "foaf:mbox",
    "foaf:homepage",
    "foaf:depiction",
    "dc:title",
    "dc:creator",
    "dc:subject",
    "dc:description",
    "dc:date",
    "dcterms:created",
    "dcterms:modified",
    "dcterms:issued",
    "skos:prefLabel",
    "skos:altLabel",
    "skos:broader",
    "skos:narrower",
];

/// Query template suggestions
const QUERY_TEMPLATES: &[(&str, &str)] = &[
    (
        "SELECT basic",
        "SELECT ?s ?p ?o\nWHERE {\n  ?s ?p ?o\n}\nLIMIT 10",
    ),
    (
        "SELECT with filter",
        "SELECT ?s ?label\nWHERE {\n  ?s rdfs:label ?label .\n  FILTER(LANG(?label) = \"en\")\n}\nLIMIT 10",
    ),
    (
        "SELECT with optional",
        "SELECT ?person ?name ?email\nWHERE {\n  ?person foaf:name ?name .\n  OPTIONAL { ?person foaf:mbox ?email }\n}\nLIMIT 10",
    ),
    (
        "COUNT query",
        "SELECT (COUNT(?s) AS ?count)\nWHERE {\n  ?s ?p ?o\n}",
    ),
    (
        "ASK query",
        "ASK WHERE {\n  ?s ?p ?o\n}",
    ),
    (
        "CONSTRUCT query",
        "CONSTRUCT {\n  ?s ?p ?o\n}\nWHERE {\n  ?s ?p ?o\n}\nLIMIT 100",
    ),
    (
        "DESCRIBE query",
        "DESCRIBE ?resource\nWHERE {\n  ?resource ?p ?o\n}\nLIMIT 1",
    ),
];

/// SPARQL autocomplete provider
pub struct SparqlAutocompleteProvider {
    /// Custom prefixes defined by user
    custom_prefixes: HashMap<String, String>,
    /// Recently used variables
    recent_variables: Vec<String>,
}

impl SparqlAutocompleteProvider {
    /// Create a new SPARQL autocomplete provider
    pub fn new() -> Self {
        Self {
            custom_prefixes: HashMap::new(),
            recent_variables: Vec::new(),
        }
    }

    /// Add a custom prefix
    pub fn add_prefix(&mut self, prefix: String, uri: String) {
        self.custom_prefixes.insert(prefix, uri);
    }

    /// Add a variable that was recently used
    pub fn add_variable(&mut self, var: String) {
        if !self.recent_variables.contains(&var) {
            self.recent_variables.push(var);
            // Keep only last 50 variables
            if self.recent_variables.len() > 50 {
                self.recent_variables.remove(0);
            }
        }
    }

    /// Get completions for the current word
    fn get_word_completions(&self, word: &str) -> Vec<CompletionItem> {
        let mut completions = Vec::new();
        let word_upper = word.to_uppercase();

        // Check if it looks like a variable
        if word.starts_with('?') || word.starts_with('$') {
            // Suggest recent variables
            for var in &self.recent_variables {
                if var.to_uppercase().starts_with(&word_upper) {
                    completions.push(CompletionItem {
                        replacement: var.clone(),
                        display: var.clone(),
                        description: Some("Recent variable".to_string()),
                        completion_type: CompletionType::Variable,
                    });
                }
            }
            return completions;
        }

        // Check if it looks like a prefixed name
        if word.contains(':') {
            let parts: Vec<&str> = word.split(':').collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let local = parts[1].to_uppercase();

                // Suggest common properties with this prefix
                for property in COMMON_PROPERTIES {
                    if property.starts_with(prefix) && property.to_uppercase().contains(&local) {
                        completions.push(CompletionItem {
                            replacement: property.to_string(),
                            display: property.to_string(),
                            description: Some("Common property".to_string()),
                            completion_type: CompletionType::Value,
                        });
                    }
                }
            }
            return completions;
        }

        // Keywords
        for keyword in SPARQL_KEYWORDS {
            if keyword.starts_with(&word_upper) {
                completions.push(CompletionItem {
                    replacement: keyword.to_string(),
                    display: keyword.to_string(),
                    description: Some("SPARQL keyword".to_string()),
                    completion_type: CompletionType::Command,
                });
            }
        }

        // Functions
        for function in SPARQL_FUNCTIONS {
            if function.to_uppercase().starts_with(&word_upper) {
                completions.push(CompletionItem {
                    replacement: format!("{}()", function),
                    display: format!("{}()", function),
                    description: Some("SPARQL function".to_string()),
                    completion_type: CompletionType::Value,
                });
            }
        }

        // Prefix suggestions (for PREFIX declarations)
        if word.is_empty() || word.len() < 4 {
            for (prefix, uri) in COMMON_PREFIXES {
                if prefix.starts_with(word) {
                    completions.push(CompletionItem {
                        replacement: format!("PREFIX {}: <{}>", prefix, uri),
                        display: format!("{}: <{}>", prefix, uri),
                        description: Some("Common prefix".to_string()),
                        completion_type: CompletionType::Value,
                    });
                }
            }
        }

        completions
    }

    /// Get query template completions
    fn get_template_completions(&self) -> Vec<CompletionItem> {
        QUERY_TEMPLATES
            .iter()
            .map(|(name, template)| CompletionItem {
                replacement: template.to_string(),
                display: name.to_string(),
                description: Some("Query template".to_string()),
                completion_type: CompletionType::Command,
            })
            .collect()
    }

    /// Analyze query context to determine what completions are relevant
    fn analyze_context(&self, query: &str, position: usize) -> QueryContext {
        let before_cursor = &query[..position];

        // Check if we're in a string literal
        let in_string = before_cursor.matches('"').count() % 2 == 1;
        if in_string {
            return QueryContext::StringLiteral;
        }

        // Check if we're in a comment
        if let Some(last_line_start) = before_cursor.rfind('\n') {
            let current_line = &before_cursor[last_line_start..];
            if current_line.contains('#') {
                return QueryContext::Comment;
            }
        }

        // Check if we're in a WHERE clause
        let in_where = before_cursor.to_uppercase().contains("WHERE")
            && !before_cursor.to_uppercase().ends_with("WHERE");

        // Check if we're in a FILTER clause
        let in_filter = before_cursor.to_uppercase().contains("FILTER")
            && before_cursor.matches('{').count() > before_cursor.matches('}').count();

        // Check if we're after PREFIX
        if before_cursor.to_uppercase().ends_with("PREFIX") {
            return QueryContext::PrefixDeclaration;
        }

        // Determine context
        if query.trim().is_empty() {
            QueryContext::Empty
        } else if in_filter {
            QueryContext::Filter
        } else if in_where {
            QueryContext::TriplePattern
        } else {
            QueryContext::QueryClause
        }
    }
}

impl Default for SparqlAutocompleteProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CompletionProvider for SparqlAutocompleteProvider {
    fn get_completions(&self, context: &CompletionContext) -> Vec<CompletionItem> {
        let query = format!("{} {}", context.args.join(" "), context.current_word);
        let query_context = self.analyze_context(&query, query.len());

        match query_context {
            QueryContext::Empty => {
                // Suggest query templates and keywords
                let mut completions = self.get_template_completions();
                completions.extend(self.get_word_completions(""));
                completions
            }
            QueryContext::PrefixDeclaration => {
                // Suggest common prefixes
                COMMON_PREFIXES
                    .iter()
                    .map(|(prefix, uri)| CompletionItem {
                        replacement: format!("{}: <{}>", prefix, uri),
                        display: format!("{}: <{}>", prefix, uri),
                        description: Some("Common prefix".to_string()),
                        completion_type: CompletionType::Value,
                    })
                    .collect()
            }
            QueryContext::Filter => {
                // Suggest functions and operators
                self.get_word_completions(&context.current_word)
                    .into_iter()
                    .filter(|c| {
                        matches!(c.completion_type, CompletionType::Value)
                            || SPARQL_FUNCTIONS.contains(&c.replacement.as_str())
                    })
                    .collect()
            }
            QueryContext::TriplePattern => {
                // Suggest variables and common properties
                self.get_word_completions(&context.current_word)
            }
            QueryContext::QueryClause => {
                // Suggest keywords
                self.get_word_completions(&context.current_word)
                    .into_iter()
                    .filter(|c| matches!(c.completion_type, CompletionType::Command))
                    .collect()
            }
            QueryContext::StringLiteral | QueryContext::Comment => {
                // No completions in strings or comments
                Vec::new()
            }
        }
    }
}

/// Query context for determining relevant completions
#[derive(Debug, Clone, Copy, PartialEq)]
enum QueryContext {
    Empty,
    PrefixDeclaration,
    QueryClause,
    TriplePattern,
    Filter,
    StringLiteral,
    Comment,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparql_autocomplete_keywords() {
        let provider = SparqlAutocompleteProvider::new();
        let context = CompletionContext {
            command: None,
            subcommand: None,
            args: vec![],
            current_word: "SEL".to_string(),
            position: 3,
        };

        let completions = provider.get_word_completions(&context.current_word);
        assert!(completions.iter().any(|c| c.replacement == "SELECT"));
    }

    #[test]
    fn test_sparql_autocomplete_functions() {
        let provider = SparqlAutocompleteProvider::new();
        let context = CompletionContext {
            command: None,
            subcommand: None,
            args: vec![],
            current_word: "CONC".to_string(),
            position: 4,
        };

        let completions = provider.get_word_completions(&context.current_word);
        assert!(completions.iter().any(|c| c.replacement == "CONCAT()"));
    }

    #[test]
    fn test_sparql_autocomplete_variables() {
        let mut provider = SparqlAutocompleteProvider::new();
        provider.add_variable("?person".to_string());
        provider.add_variable("?name".to_string());

        let context = CompletionContext {
            command: None,
            subcommand: None,
            args: vec![],
            current_word: "?p".to_string(),
            position: 2,
        };

        let completions = provider.get_word_completions(&context.current_word);
        assert!(completions.iter().any(|c| c.replacement == "?person"));
    }

    #[test]
    fn test_sparql_autocomplete_prefixes() {
        let provider = SparqlAutocompleteProvider::new();
        let completions = provider.get_word_completions("fo");

        assert!(completions.iter().any(|c| c.display.contains("foaf")));
    }

    #[test]
    fn test_query_context_empty() {
        let provider = SparqlAutocompleteProvider::new();
        let context = provider.analyze_context("", 0);
        assert_eq!(context, QueryContext::Empty);
    }

    #[test]
    fn test_query_context_where() {
        let provider = SparqlAutocompleteProvider::new();
        let query = "SELECT ?s WHERE { ?s ?p ?o";
        let context = provider.analyze_context(query, query.len());
        assert_eq!(context, QueryContext::TriplePattern);
    }

    #[test]
    fn test_query_context_filter() {
        let provider = SparqlAutocompleteProvider::new();
        let query = "SELECT ?s WHERE { ?s ?p ?o . FILTER(";
        let context = provider.analyze_context(query, query.len());
        assert_eq!(context, QueryContext::Filter);
    }

    #[test]
    fn test_query_templates() {
        let provider = SparqlAutocompleteProvider::new();
        let templates = provider.get_template_completions();
        assert!(!templates.is_empty());
        assert!(templates.iter().any(|t| t.display.contains("SELECT")));
    }

    #[test]
    fn test_add_custom_prefix() {
        let mut provider = SparqlAutocompleteProvider::new();
        provider.add_prefix("custom".to_string(), "http://example.org/".to_string());
        assert_eq!(
            provider.custom_prefixes.get("custom"),
            Some(&"http://example.org/".to_string())
        );
    }

    #[test]
    fn test_add_variable() {
        let mut provider = SparqlAutocompleteProvider::new();
        provider.add_variable("?test".to_string());
        assert!(provider.recent_variables.contains(&"?test".to_string()));
    }

    #[test]
    fn test_variable_limit() {
        let mut provider = SparqlAutocompleteProvider::new();
        // Add 60 variables to test the limit
        for i in 0..60 {
            provider.add_variable(format!("?var{}", i));
        }
        // Should keep only the last 50
        assert_eq!(provider.recent_variables.len(), 50);
        assert!(!provider.recent_variables.contains(&"?var0".to_string()));
        assert!(provider.recent_variables.contains(&"?var59".to_string()));
    }
}
