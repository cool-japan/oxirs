//! Hover information provider for SHACL shapes.
//!
//! Provides documentation and type information when hovering over SHACL properties.

use std::sync::Arc;

use tower_lsp::lsp_types::{Hover, HoverContents, MarkupContent, MarkupKind, Position};

use oxirs_core::ConcreteStore;

/// Hover provider for SHACL shapes
pub struct HoverProvider {
    _store: Arc<ConcreteStore>,
}

impl HoverProvider {
    /// Create a new hover provider
    pub fn new(store: Arc<ConcreteStore>) -> Self {
        Self { _store: store }
    }

    /// Provide hover information for a given position
    pub async fn provide_hover(&self, text: &str, position: Position) -> Option<Hover> {
        // Get word at position
        let word = self.get_word_at_position(text, position)?;

        // Get hover content for the word
        let content = self.get_hover_content(&word)?;

        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: content,
            }),
            range: None,
        })
    }

    /// Get word at cursor position
    fn get_word_at_position(&self, text: &str, position: Position) -> Option<String> {
        let lines: Vec<&str> = text.lines().collect();
        if position.line as usize >= lines.len() {
            return None;
        }

        let line = lines[position.line as usize];
        let cursor_pos = position.character as usize;

        if cursor_pos > line.len() {
            return None;
        }

        // Find word boundaries
        let start = line[..cursor_pos]
            .rfind(|c: char| !c.is_alphanumeric() && c != ':' && c != '_')
            .map(|p| p + 1)
            .unwrap_or(0);

        let end = line[cursor_pos..]
            .find(|c: char| !c.is_alphanumeric() && c != ':' && c != '_')
            .map(|p| cursor_pos + p)
            .unwrap_or(line.len());

        if start < end {
            Some(line[start..end].to_string())
        } else {
            None
        }
    }

    /// Get hover content for a SHACL term
    fn get_hover_content(&self, term: &str) -> Option<String> {
        // Match against known SHACL vocabulary
        match term {
            // Core shape types
            "sh:NodeShape" => Some(self.format_hover(
                "NodeShape",
                "A node shape that validates nodes in the data graph",
                "```turtle\nex:PersonShape\n  a sh:NodeShape ;\n  sh:targetClass ex:Person ;\n  sh:property [...] .\n```"
            )),
            "sh:PropertyShape" => Some(self.format_hover(
                "PropertyShape",
                "A property shape that validates property values",
                "```turtle\nex:PersonShape\n  sh:property [\n    a sh:PropertyShape ;\n    sh:path ex:name ;\n    sh:datatype xsd:string ;\n  ] .\n```"
            )),

            // Target definitions
            "sh:targetClass" => Some(self.format_hover(
                "targetClass",
                "Specifies the class of nodes to validate",
                "```turtle\nsh:targetClass ex:Person\n```\nValidates all instances of ex:Person"
            )),
            "sh:targetNode" => Some(self.format_hover(
                "targetNode",
                "Specifies specific nodes to validate",
                "```turtle\nsh:targetNode ex:JohnDoe\n```\nValidates only the specified node"
            )),
            "sh:targetSubjectsOf" => Some(self.format_hover(
                "targetSubjectsOf",
                "Validates nodes that are subjects of a property",
                "```turtle\nsh:targetSubjectsOf ex:hasAge\n```\nValidates all nodes with ex:hasAge property"
            )),
            "sh:targetObjectsOf" => Some(self.format_hover(
                "targetObjectsOf",
                "Validates nodes that are objects of a property",
                "```turtle\nsh:targetObjectsOf ex:knows\n```\nValidates all nodes referenced by ex:knows"
            )),

            // Cardinality constraints
            "sh:minCount" => Some(self.format_hover(
                "minCount",
                "Minimum number of values (inclusive)",
                "```turtle\nsh:minCount 1 ;\n```\nRequires at least 1 value"
            )),
            "sh:maxCount" => Some(self.format_hover(
                "maxCount",
                "Maximum number of values (inclusive)",
                "```turtle\nsh:maxCount 1 ;\n```\nAllows at most 1 value"
            )),

            // Value type constraints
            "sh:datatype" => Some(self.format_hover(
                "datatype",
                "Specifies the required datatype for literal values",
                "```turtle\nsh:datatype xsd:string ;\n```\nRequires string values"
            )),
            "sh:class" => Some(self.format_hover(
                "class",
                "Specifies the required class for node values",
                "```turtle\nsh:class ex:Person ;\n```\nRequires values to be instances of ex:Person"
            )),
            "sh:nodeKind" => Some(self.format_hover(
                "nodeKind",
                "Specifies the kind of node (IRI, BlankNode, Literal)",
                "```turtle\nsh:nodeKind sh:IRI ;\n```\nRequires values to be IRIs"
            )),

            // String constraints
            "sh:minLength" => Some(self.format_hover(
                "minLength",
                "Minimum string length (inclusive)",
                "```turtle\nsh:minLength 3 ;\n```\nRequires at least 3 characters"
            )),
            "sh:maxLength" => Some(self.format_hover(
                "maxLength",
                "Maximum string length (inclusive)",
                "```turtle\nsh:maxLength 100 ;\n```\nAllows at most 100 characters"
            )),
            "sh:pattern" => Some(self.format_hover(
                "pattern",
                "Regular expression pattern for string matching",
                "```turtle\nsh:pattern \"^[A-Z][a-z]+$\" ;\n```\nRequires capitalized words"
            )),

            // Numeric constraints
            "sh:minInclusive" => Some(self.format_hover(
                "minInclusive",
                "Minimum numeric value (inclusive)",
                "```turtle\nsh:minInclusive 0 ;\n```\nRequires value >= 0"
            )),
            "sh:maxInclusive" => Some(self.format_hover(
                "maxInclusive",
                "Maximum numeric value (inclusive)",
                "```turtle\nsh:maxInclusive 100 ;\n```\nRequires value <= 100"
            )),
            "sh:minExclusive" => Some(self.format_hover(
                "minExclusive",
                "Minimum numeric value (exclusive)",
                "```turtle\nsh:minExclusive 0 ;\n```\nRequires value > 0"
            )),
            "sh:maxExclusive" => Some(self.format_hover(
                "maxExclusive",
                "Maximum numeric value (exclusive)",
                "```turtle\nsh:maxExclusive 100 ;\n```\nRequires value < 100"
            )),

            // Logical constraints
            "sh:and" => Some(self.format_hover(
                "and",
                "Conjunction - all shapes must be valid",
                "```turtle\nsh:and (\n  [sh:datatype xsd:string]\n  [sh:minLength 3]\n) ;\n```"
            )),
            "sh:or" => Some(self.format_hover(
                "or",
                "Disjunction - at least one shape must be valid",
                "```turtle\nsh:or (\n  [sh:datatype xsd:string]\n  [sh:datatype xsd:integer]\n) ;\n```"
            )),
            "sh:not" => Some(self.format_hover(
                "not",
                "Negation - shape must NOT be valid",
                "```turtle\nsh:not [\n  sh:hasValue \"forbidden\"\n] ;\n```"
            )),
            "sh:xone" => Some(self.format_hover(
                "xone",
                "Exclusive OR - exactly one shape must be valid",
                "```turtle\nsh:xone (\n  [sh:datatype xsd:string]\n  [sh:datatype xsd:integer]\n) ;\n```"
            )),

            // Other constraints
            "sh:in" => Some(self.format_hover(
                "in",
                "Enumeration - value must be in list",
                "```turtle\nsh:in (\"red\" \"green\" \"blue\") ;\n```\nOnly allows listed values"
            )),
            "sh:hasValue" => Some(self.format_hover(
                "hasValue",
                "Requires a specific value to be present",
                "```turtle\nsh:hasValue \"required value\" ;\n```"
            )),

            // Severity levels
            "sh:Violation" => Some(self.format_hover(
                "Violation",
                "Error-level severity",
                "```turtle\nsh:severity sh:Violation\n```\nValidation failure is an error"
            )),
            "sh:Warning" => Some(self.format_hover(
                "Warning",
                "Warning-level severity",
                "```turtle\nsh:severity sh:Warning\n```\nValidation failure is a warning"
            )),
            "sh:Info" => Some(self.format_hover(
                "Info",
                "Info-level severity",
                "```turtle\nsh:severity sh:Info\n```\nValidation failure is informational"
            )),

            _ => None,
        }
    }

    /// Format hover content
    fn format_hover(&self, title: &str, description: &str, example: &str) -> String {
        format!(
            "**{}**\n\n{}\n\n### Example\n\n{}",
            title, description, example
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hover_provider() {
        let store = Arc::new(ConcreteStore::new().unwrap());
        let provider = HoverProvider::new(store);

        let text = "sh:targetClass ex:Person";
        let position = Position::new(0, 5); // On "targetClass"

        let hover = provider.provide_hover(text, position).await;
        assert!(hover.is_some());
    }

    #[test]
    fn test_word_extraction() {
        let store = Arc::new(ConcreteStore::new().unwrap());
        let provider = HoverProvider::new(store);

        let word = provider.get_word_at_position("sh:targetClass", Position::new(0, 5));
        assert_eq!(word, Some("sh:targetClass".to_string()));
    }
}
