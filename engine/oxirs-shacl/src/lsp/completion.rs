//! Code completion provider for SHACL shapes.
//!
//! Provides intelligent code completion for:
//! - SHACL properties (sh:targetClass, sh:property, etc.)
//! - SHACL classes (sh:NodeShape, sh:PropertyShape, etc.)
//! - Constraint types (sh:minCount, sh:maxCount, etc.)
//! - Data types (xsd:string, xsd:integer, etc.)

use std::sync::Arc;

use tower_lsp::lsp_types::{
    CompletionItem, CompletionItemKind, CompletionResponse, Documentation, InsertTextFormat,
    MarkupContent, MarkupKind, Position,
};

use oxirs_core::ConcreteStore;

/// Completion provider for SHACL shapes
pub struct CompletionProvider {
    _store: Arc<ConcreteStore>,
}

impl CompletionProvider {
    /// Create a new completion provider
    pub fn new(store: Arc<ConcreteStore>) -> Self {
        Self { _store: store }
    }

    /// Provide completions for a given position in the document
    pub async fn provide_completions(&self, text: &str, position: Position) -> CompletionResponse {
        let mut completions = Vec::new();

        // Get context at position (what prefix is being typed)
        let context = self.get_completion_context(text, position);

        match context.as_str() {
            "sh:" => {
                // SHACL vocabulary completions
                completions.extend(self.get_shacl_property_completions());
                completions.extend(self.get_shacl_class_completions());
            }
            "xsd:" => {
                // XSD datatype completions
                completions.extend(self.get_xsd_datatype_completions());
            }
            "rdf:" => {
                // RDF vocabulary completions
                completions.extend(self.get_rdf_completions());
            }
            "rdfs:" => {
                // RDFS vocabulary completions
                completions.extend(self.get_rdfs_completions());
            }
            _ => {
                // General completions
                completions.extend(self.get_general_completions());
            }
        }

        CompletionResponse::Array(completions)
    }

    /// Get completion context (prefix being typed)
    fn get_completion_context(&self, text: &str, position: Position) -> String {
        // Extract text up to cursor position
        let lines: Vec<&str> = text.lines().collect();
        if position.line as usize >= lines.len() {
            return String::new();
        }

        let line = lines[position.line as usize];
        let cursor_pos = position.character as usize;

        if cursor_pos > line.len() {
            return String::new();
        }

        // Get word/prefix before cursor
        let before_cursor = &line[..cursor_pos];

        // Check for namespace prefixes
        if let Some(colon_pos) = before_cursor.rfind(':') {
            if colon_pos > 0 {
                let prefix_start = before_cursor[..colon_pos]
                    .rfind(|c: char| !c.is_alphanumeric() && c != '_')
                    .map(|p| p + 1)
                    .unwrap_or(0);
                return before_cursor[prefix_start..=colon_pos].to_string();
            }
        }

        String::new()
    }

    /// Get SHACL property completions
    fn get_shacl_property_completions(&self) -> Vec<CompletionItem> {
        vec![
            self.create_completion(
                "targetClass",
                "Target class for shape",
                "sh:targetClass <ClassIRI>",
            ),
            self.create_completion(
                "targetNode",
                "Specific target node",
                "sh:targetNode <NodeIRI>",
            ),
            self.create_completion(
                "targetSubjectsOf",
                "Subjects of property",
                "sh:targetSubjectsOf <PropertyIRI>",
            ),
            self.create_completion(
                "targetObjectsOf",
                "Objects of property",
                "sh:targetObjectsOf <PropertyIRI>",
            ),
            self.create_completion(
                "property",
                "Property shape",
                "sh:property [ sh:path <PropertyIRI> ]",
            ),
            self.create_completion("path", "Property path", "sh:path <PropertyIRI>"),
            self.create_completion("datatype", "Value datatype", "sh:datatype xsd:string"),
            self.create_completion("class", "Value class", "sh:class <ClassIRI>"),
            self.create_completion("minCount", "Minimum cardinality", "sh:minCount 1"),
            self.create_completion("maxCount", "Maximum cardinality", "sh:maxCount 1"),
            self.create_completion("minLength", "Minimum string length", "sh:minLength 1"),
            self.create_completion("maxLength", "Maximum string length", "sh:maxLength 100"),
            self.create_completion(
                "minInclusive",
                "Minimum value (inclusive)",
                "sh:minInclusive 0",
            ),
            self.create_completion(
                "maxInclusive",
                "Maximum value (inclusive)",
                "sh:maxInclusive 100",
            ),
            self.create_completion(
                "minExclusive",
                "Minimum value (exclusive)",
                "sh:minExclusive 0",
            ),
            self.create_completion(
                "maxExclusive",
                "Maximum value (exclusive)",
                "sh:maxExclusive 100",
            ),
            self.create_completion("pattern", "Regex pattern", "sh:pattern \"^[A-Z]+$\""),
            self.create_completion(
                "languageIn",
                "Language tags",
                "sh:languageIn (\"en\" \"fr\")",
            ),
            self.create_completion("nodeKind", "Node kind constraint", "sh:nodeKind sh:IRI"),
            self.create_completion("in", "Enumeration", "sh:in (\"value1\" \"value2\")"),
            self.create_completion("hasValue", "Fixed value", "sh:hasValue \"specificValue\""),
            self.create_completion("node", "Node shape", "sh:node <ShapeIRI>"),
            self.create_completion("not", "Negation", "sh:not <ShapeIRI>"),
            self.create_completion("and", "Conjunction", "sh:and (<Shape1> <Shape2>)"),
            self.create_completion("or", "Disjunction", "sh:or (<Shape1> <Shape2>)"),
            self.create_completion(
                "xone",
                "Exclusive disjunction",
                "sh:xone (<Shape1> <Shape2>)",
            ),
            self.create_completion(
                "message",
                "Validation message",
                "sh:message \"Custom error message\"",
            ),
            self.create_completion(
                "severity",
                "Validation severity",
                "sh:severity sh:Violation",
            ),
            self.create_completion("deactivated", "Deactivate shape", "sh:deactivated true"),
            self.create_completion("name", "Shape name", "sh:name \"Shape Name\""),
            self.create_completion(
                "description",
                "Shape description",
                "sh:description \"Description\"",
            ),
        ]
    }

    /// Get SHACL class completions
    fn get_shacl_class_completions(&self) -> Vec<CompletionItem> {
        vec![
            self.create_completion("NodeShape", "Node shape class", "a sh:NodeShape"),
            self.create_completion(
                "PropertyShape",
                "Property shape class",
                "a sh:PropertyShape",
            ),
            self.create_completion("IRI", "IRI node kind", "sh:nodeKind sh:IRI"),
            self.create_completion("BlankNode", "Blank node kind", "sh:nodeKind sh:BlankNode"),
            self.create_completion("Literal", "Literal node kind", "sh:nodeKind sh:Literal"),
            self.create_completion(
                "BlankNodeOrIRI",
                "Blank node or IRI",
                "sh:nodeKind sh:BlankNodeOrIRI",
            ),
            self.create_completion(
                "BlankNodeOrLiteral",
                "Blank node or literal",
                "sh:nodeKind sh:BlankNodeOrLiteral",
            ),
            self.create_completion(
                "IRIOrLiteral",
                "IRI or literal",
                "sh:nodeKind sh:IRIOrLiteral",
            ),
            self.create_completion(
                "Violation",
                "Violation severity",
                "sh:severity sh:Violation",
            ),
            self.create_completion("Warning", "Warning severity", "sh:severity sh:Warning"),
            self.create_completion("Info", "Info severity", "sh:severity sh:Info"),
        ]
    }

    /// Get XSD datatype completions
    fn get_xsd_datatype_completions(&self) -> Vec<CompletionItem> {
        vec![
            self.create_completion("string", "String datatype", "xsd:string"),
            self.create_completion("boolean", "Boolean datatype", "xsd:boolean"),
            self.create_completion("integer", "Integer datatype", "xsd:integer"),
            self.create_completion("decimal", "Decimal datatype", "xsd:decimal"),
            self.create_completion("float", "Float datatype", "xsd:float"),
            self.create_completion("double", "Double datatype", "xsd:double"),
            self.create_completion("dateTime", "DateTime datatype", "xsd:dateTime"),
            self.create_completion("date", "Date datatype", "xsd:date"),
            self.create_completion("time", "Time datatype", "xsd:time"),
            self.create_completion("duration", "Duration datatype", "xsd:duration"),
            self.create_completion("anyURI", "URI datatype", "xsd:anyURI"),
            self.create_completion("base64Binary", "Base64 binary datatype", "xsd:base64Binary"),
            self.create_completion("hexBinary", "Hex binary datatype", "xsd:hexBinary"),
        ]
    }

    /// Get RDF vocabulary completions
    fn get_rdf_completions(&self) -> Vec<CompletionItem> {
        vec![
            self.create_completion("type", "RDF type", "rdf:type <ClassIRI>"),
            self.create_completion("Property", "RDF Property", "rdf:Property"),
            self.create_completion("List", "RDF List", "rdf:List"),
            self.create_completion("nil", "Empty list", "rdf:nil"),
        ]
    }

    /// Get RDFS vocabulary completions
    fn get_rdfs_completions(&self) -> Vec<CompletionItem> {
        vec![
            self.create_completion("label", "RDFS label", "rdfs:label \"Label\""),
            self.create_completion("comment", "RDFS comment", "rdfs:comment \"Comment\""),
            self.create_completion(
                "subClassOf",
                "Subclass relationship",
                "rdfs:subClassOf <ClassIRI>",
            ),
            self.create_completion(
                "subPropertyOf",
                "Subproperty relationship",
                "rdfs:subPropertyOf <PropertyIRI>",
            ),
            self.create_completion("domain", "Property domain", "rdfs:domain <ClassIRI>"),
            self.create_completion("range", "Property range", "rdfs:range <ClassIRI>"),
        ]
    }

    /// Get general completions
    fn get_general_completions(&self) -> Vec<CompletionItem> {
        vec![
            self.create_completion(
                "@prefix",
                "Prefix declaration",
                "@prefix sh: <http://www.w3.org/ns/shacl#> .",
            ),
            self.create_completion(
                "@base",
                "Base URI declaration",
                "@base <http://example.org/> .",
            ),
            self.create_completion("a", "RDF type shorthand", "a sh:NodeShape"),
        ]
    }

    /// Create a completion item
    fn create_completion(
        &self,
        label: &str,
        description: &str,
        insert_text: &str,
    ) -> CompletionItem {
        CompletionItem {
            label: label.to_string(),
            kind: Some(CompletionItemKind::PROPERTY),
            detail: Some(description.to_string()),
            documentation: Some(Documentation::MarkupContent(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!(
                    "**{}**\n\n{}\n\n```turtle\n{}\n```",
                    label, description, insert_text
                ),
            })),
            insert_text: Some(insert_text.to_string()),
            insert_text_format: Some(InsertTextFormat::PLAIN_TEXT),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_completion_provider() {
        let store = Arc::new(ConcreteStore::new().unwrap());
        let provider = CompletionProvider::new(store);

        let text = "sh:";
        let position = Position::new(0, 3);

        let completions = provider.provide_completions(text, position).await;

        match completions {
            CompletionResponse::Array(items) => {
                assert!(!items.is_empty());
            }
            _ => panic!("Expected array response"),
        }
    }

    #[test]
    fn test_completion_context_extraction() {
        let store = Arc::new(ConcreteStore::new().unwrap());
        let provider = CompletionProvider::new(store);

        let context = provider.get_completion_context("sh:", Position::new(0, 3));
        assert_eq!(context, "sh:");

        let context = provider.get_completion_context("  xsd:", Position::new(0, 6));
        assert_eq!(context, "xsd:");
    }
}
