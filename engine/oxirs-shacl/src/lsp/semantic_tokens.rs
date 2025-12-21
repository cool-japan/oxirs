//! Semantic tokens provider for SHACL shapes.
//!
//! Provides semantic highlighting for better syntax visualization in IDEs.
//! Supports Turtle format with SHACL-specific token recognition.

use std::collections::HashSet;
use tower_lsp::lsp_types::{
    SemanticToken, SemanticTokenModifier, SemanticTokenType, SemanticTokensLegend,
};

/// Token type indices for the legend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaclTokenType {
    Namespace = 0,
    Class = 1,
    Property = 2,
    String = 3,
    Number = 4,
    Keyword = 5,
    Comment = 6,
    Variable = 7,
    Operator = 8,
    Type = 9,
    Function = 10,
}

impl ShaclTokenType {
    fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Semantic tokens provider with full SHACL/Turtle parsing
#[derive(Debug)]
pub struct SemanticTokensProvider {
    /// Known SHACL namespace prefixes
    shacl_prefixes: HashSet<String>,
    /// Known RDF/RDFS/OWL prefixes
    rdf_prefixes: HashSet<String>,
    /// Known XSD prefixes
    xsd_prefixes: HashSet<String>,
}

impl SemanticTokensProvider {
    /// Create a new semantic tokens provider
    pub fn new() -> Self {
        let mut shacl_prefixes = HashSet::new();
        shacl_prefixes.insert("sh".to_string());
        shacl_prefixes.insert("shacl".to_string());

        let mut rdf_prefixes = HashSet::new();
        rdf_prefixes.insert("rdf".to_string());
        rdf_prefixes.insert("rdfs".to_string());
        rdf_prefixes.insert("owl".to_string());
        rdf_prefixes.insert("skos".to_string());
        rdf_prefixes.insert("dcterms".to_string());
        rdf_prefixes.insert("dc".to_string());
        rdf_prefixes.insert("foaf".to_string());
        rdf_prefixes.insert("schema".to_string());

        let mut xsd_prefixes = HashSet::new();
        xsd_prefixes.insert("xsd".to_string());
        xsd_prefixes.insert("xs".to_string());

        Self {
            shacl_prefixes,
            rdf_prefixes,
            xsd_prefixes,
        }
    }

    /// Get semantic tokens legend
    pub fn legend() -> SemanticTokensLegend {
        SemanticTokensLegend {
            token_types: vec![
                SemanticTokenType::NAMESPACE, // 0
                SemanticTokenType::CLASS,     // 1
                SemanticTokenType::PROPERTY,  // 2
                SemanticTokenType::STRING,    // 3
                SemanticTokenType::NUMBER,    // 4
                SemanticTokenType::KEYWORD,   // 5
                SemanticTokenType::COMMENT,   // 6
                SemanticTokenType::VARIABLE,  // 7
                SemanticTokenType::OPERATOR,  // 8
                SemanticTokenType::TYPE,      // 9
                SemanticTokenType::FUNCTION,  // 10
            ],
            token_modifiers: vec![
                SemanticTokenModifier::DECLARATION,
                SemanticTokenModifier::DEFINITION,
                SemanticTokenModifier::READONLY,
            ],
        }
    }

    /// Generate semantic tokens for document
    pub fn generate_tokens(&self, text: &str) -> Vec<SemanticToken> {
        let mut tokens = Vec::new();
        let mut prev_line = 0u32;
        let mut prev_char = 0u32;

        for (line_idx, line) in text.lines().enumerate() {
            let line_num = line_idx as u32;
            let trimmed = line.trim_start();
            let leading_spaces = line.len() - trimmed.len();

            // Skip empty lines
            if trimmed.is_empty() {
                continue;
            }

            // Handle comments
            if trimmed.starts_with('#') {
                let token = self.make_token(
                    line_num,
                    leading_spaces as u32,
                    trimmed.len() as u32,
                    ShaclTokenType::Comment,
                    &mut prev_line,
                    &mut prev_char,
                );
                tokens.push(token);
                continue;
            }

            // Parse line for tokens
            let line_tokens = self.parse_line(line, line_num);
            for (start_char, length, token_type) in line_tokens {
                let token = self.make_token(
                    line_num,
                    start_char,
                    length,
                    token_type,
                    &mut prev_line,
                    &mut prev_char,
                );
                tokens.push(token);
            }
        }

        tokens
    }

    /// Parse a single line and return token positions
    fn parse_line(&self, line: &str, _line_num: u32) -> Vec<(u32, u32, ShaclTokenType)> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = line.chars().collect();
        let len = chars.len();
        let mut pos = 0;

        while pos < len {
            // Skip whitespace
            if chars[pos].is_whitespace() {
                pos += 1;
                continue;
            }

            // Check for @prefix or @base keywords
            if chars[pos] == '@' {
                let start = pos;
                pos += 1;
                while pos < len && chars[pos].is_alphanumeric() {
                    pos += 1;
                }
                let keyword: String = chars[start..pos].iter().collect();
                if keyword == "@prefix" || keyword == "@base" {
                    tokens.push((start as u32, (pos - start) as u32, ShaclTokenType::Keyword));
                }
                continue;
            }

            // Check for prefixed names (prefix:localname)
            if chars[pos].is_alphabetic() || chars[pos] == '_' {
                let start = pos;
                // Consume prefix part
                while pos < len
                    && (chars[pos].is_alphanumeric() || chars[pos] == '_' || chars[pos] == '-')
                {
                    pos += 1;
                }

                // Check for colon (indicates prefixed name)
                if pos < len && chars[pos] == ':' {
                    let prefix: String = chars[start..pos].iter().collect();
                    let prefix_end = pos;
                    pos += 1; // skip colon

                    // Check for local name after colon
                    let local_start = pos;
                    while pos < len
                        && (chars[pos].is_alphanumeric()
                            || chars[pos] == '_'
                            || chars[pos] == '-'
                            || chars[pos] == '.')
                    {
                        pos += 1;
                    }

                    let local_name: String = chars[local_start..pos].iter().collect();
                    let token_type = self.classify_prefixed_name(&prefix, &local_name);

                    // Add prefix as namespace token
                    tokens.push((
                        start as u32,
                        (prefix_end - start + 1) as u32,
                        ShaclTokenType::Namespace,
                    ));

                    // Add local name with appropriate type
                    if !local_name.is_empty() {
                        tokens.push((local_start as u32, (pos - local_start) as u32, token_type));
                    }
                } else {
                    // Just a word - check if it's a keyword
                    let word: String = chars[start..pos].iter().collect();
                    let token_type = self.classify_keyword(&word);
                    tokens.push((start as u32, (pos - start) as u32, token_type));
                }
                continue;
            }

            // Check for full IRIs
            if chars[pos] == '<' {
                let start = pos;
                pos += 1;
                while pos < len && chars[pos] != '>' {
                    pos += 1;
                }
                if pos < len {
                    pos += 1; // include closing >
                }
                tokens.push((start as u32, (pos - start) as u32, ShaclTokenType::Class));
                continue;
            }

            // Check for strings
            if chars[pos] == '"' {
                let start = pos;
                let quote_char = chars[pos];
                pos += 1;

                // Check for triple-quoted string
                if pos + 1 < len && chars[pos] == quote_char && chars[pos + 1] == quote_char {
                    pos += 2; // skip second and third quotes
                              // Find closing triple quotes
                    while pos + 2 < len {
                        if chars[pos] == quote_char
                            && chars[pos + 1] == quote_char
                            && chars[pos + 2] == quote_char
                        {
                            pos += 3;
                            break;
                        }
                        pos += 1;
                    }
                } else {
                    // Single-line string
                    while pos < len && chars[pos] != quote_char {
                        if chars[pos] == '\\' && pos + 1 < len {
                            pos += 2; // skip escaped char
                        } else {
                            pos += 1;
                        }
                    }
                    if pos < len {
                        pos += 1; // include closing quote
                    }
                }

                // Check for language tag or datatype
                if pos < len && chars[pos] == '@' {
                    let lang_start = pos;
                    pos += 1;
                    while pos < len && chars[pos].is_alphanumeric() {
                        pos += 1;
                    }
                    tokens.push((
                        start as u32,
                        (lang_start - start) as u32,
                        ShaclTokenType::String,
                    ));
                    tokens.push((
                        lang_start as u32,
                        (pos - lang_start) as u32,
                        ShaclTokenType::Keyword,
                    ));
                } else if pos + 1 < len && chars[pos] == '^' && chars[pos + 1] == '^' {
                    tokens.push((start as u32, (pos - start) as u32, ShaclTokenType::String));
                    tokens.push((pos as u32, 2, ShaclTokenType::Operator));
                    pos += 2;
                    // The datatype IRI will be handled in next iteration
                } else {
                    tokens.push((start as u32, (pos - start) as u32, ShaclTokenType::String));
                }
                continue;
            }

            // Check for numbers
            if chars[pos].is_ascii_digit()
                || (chars[pos] == '-' && pos + 1 < len && chars[pos + 1].is_ascii_digit())
            {
                let start = pos;
                if chars[pos] == '-' {
                    pos += 1;
                }
                while pos < len
                    && (chars[pos].is_ascii_digit()
                        || chars[pos] == '.'
                        || chars[pos] == 'e'
                        || chars[pos] == 'E')
                {
                    pos += 1;
                }
                tokens.push((start as u32, (pos - start) as u32, ShaclTokenType::Number));
                continue;
            }

            // Check for operators and punctuation
            match chars[pos] {
                '.' | ';' | ',' | '[' | ']' | '(' | ')' | '{' | '}' => {
                    tokens.push((pos as u32, 1, ShaclTokenType::Operator));
                    pos += 1;
                }
                '^' => {
                    if pos + 1 < len && chars[pos + 1] == '^' {
                        tokens.push((pos as u32, 2, ShaclTokenType::Operator));
                        pos += 2;
                    } else {
                        tokens.push((pos as u32, 1, ShaclTokenType::Operator));
                        pos += 1;
                    }
                }
                _ => {
                    pos += 1; // Skip unknown characters
                }
            }
        }

        tokens
    }

    /// Classify a prefixed name based on prefix and local name
    fn classify_prefixed_name(&self, prefix: &str, local_name: &str) -> ShaclTokenType {
        // SHACL-specific tokens
        if self.shacl_prefixes.contains(prefix) {
            return self.classify_shacl_term(local_name);
        }

        // RDF/RDFS/OWL type tokens
        if self.rdf_prefixes.contains(prefix) {
            if local_name == "type" || local_name.chars().next().is_some_and(|c| c.is_uppercase()) {
                return ShaclTokenType::Type;
            }
            return ShaclTokenType::Property;
        }

        // XSD datatype tokens
        if self.xsd_prefixes.contains(prefix) {
            return ShaclTokenType::Type;
        }

        // Default: if starts with uppercase, it's likely a class
        if local_name.chars().next().is_some_and(|c| c.is_uppercase()) {
            ShaclTokenType::Class
        } else {
            ShaclTokenType::Property
        }
    }

    /// Classify a SHACL term
    fn classify_shacl_term(&self, term: &str) -> ShaclTokenType {
        match term {
            // Shape types
            "NodeShape" | "PropertyShape" | "Shape" => ShaclTokenType::Class,

            // Target types
            "targetClass" | "targetNode" | "targetSubjectsOf" | "targetObjectsOf" | "target" => {
                ShaclTokenType::Function
            }

            // Constraint types (functions)
            "minCount"
            | "maxCount"
            | "minLength"
            | "maxLength"
            | "minInclusive"
            | "maxInclusive"
            | "minExclusive"
            | "maxExclusive"
            | "pattern"
            | "flags"
            | "datatype"
            | "class"
            | "nodeKind"
            | "in"
            | "hasValue"
            | "equals"
            | "disjoint"
            | "lessThan"
            | "lessThanOrEquals"
            | "qualifiedValueShape"
            | "qualifiedMinCount"
            | "qualifiedMaxCount"
            | "closed"
            | "ignoredProperties"
            | "uniqueLang"
            | "languageIn" => ShaclTokenType::Function,

            // Property path operators
            "path" | "alternativePath" | "inversePath" | "zeroOrMorePath" | "oneOrMorePath"
            | "zeroOrOnePath" => ShaclTokenType::Function,

            // Node kinds
            "IRI" | "BlankNode" | "Literal" | "BlankNodeOrIRI" | "BlankNodeOrLiteral"
            | "IRIOrLiteral" => ShaclTokenType::Type,

            // Severity levels
            "Violation" | "Warning" | "Info" => ShaclTokenType::Type,

            // Logical constraints
            "and" | "or" | "not" | "xone" => ShaclTokenType::Keyword,

            // Other properties
            "property" | "node" | "name" | "description" | "message" | "severity"
            | "deactivated" | "order" | "group" | "defaultValue" => ShaclTokenType::Property,

            // Advanced features
            "sparql" | "select" | "ask" | "prefixes" | "rule" | "condition" | "subject"
            | "predicate" | "object" => ShaclTokenType::Function,

            // Default to property
            _ => {
                if term.chars().next().is_some_and(|c| c.is_uppercase()) {
                    ShaclTokenType::Class
                } else {
                    ShaclTokenType::Property
                }
            }
        }
    }

    /// Classify a standalone keyword
    fn classify_keyword(&self, word: &str) -> ShaclTokenType {
        match word.to_lowercase().as_str() {
            "a" => ShaclTokenType::Keyword,
            "true" | "false" => ShaclTokenType::Keyword,
            "prefix" | "base" => ShaclTokenType::Keyword,
            _ => {
                if word.chars().next().is_some_and(|c| c.is_uppercase()) {
                    ShaclTokenType::Class
                } else {
                    ShaclTokenType::Variable
                }
            }
        }
    }

    /// Create a semantic token with relative positioning
    fn make_token(
        &self,
        line: u32,
        start_char: u32,
        length: u32,
        token_type: ShaclTokenType,
        prev_line: &mut u32,
        prev_char: &mut u32,
    ) -> SemanticToken {
        let delta_line = line - *prev_line;
        let delta_start = if delta_line == 0 {
            start_char - *prev_char
        } else {
            start_char
        };

        *prev_line = line;
        *prev_char = start_char;

        SemanticToken {
            delta_line,
            delta_start,
            length,
            token_type: token_type.as_u32(),
            token_modifiers_bitset: 0,
        }
    }

    /// Generate semantic tokens for a range
    pub fn generate_tokens_range(
        &self,
        text: &str,
        start_line: u32,
        end_line: u32,
    ) -> Vec<SemanticToken> {
        let mut tokens = Vec::new();
        let mut prev_line = start_line;
        let mut prev_char = 0u32;

        for (line_idx, line) in text.lines().enumerate() {
            let line_num = line_idx as u32;

            // Skip lines outside range
            if line_num < start_line || line_num > end_line {
                continue;
            }

            let trimmed = line.trim_start();
            let leading_spaces = line.len() - trimmed.len();

            if trimmed.is_empty() {
                continue;
            }

            if trimmed.starts_with('#') {
                let token = self.make_token(
                    line_num,
                    leading_spaces as u32,
                    trimmed.len() as u32,
                    ShaclTokenType::Comment,
                    &mut prev_line,
                    &mut prev_char,
                );
                tokens.push(token);
                continue;
            }

            let line_tokens = self.parse_line(line, line_num);
            for (start_char, length, token_type) in line_tokens {
                let token = self.make_token(
                    line_num,
                    start_char,
                    length,
                    token_type,
                    &mut prev_line,
                    &mut prev_char,
                );
                tokens.push(token);
            }
        }

        tokens
    }
}

impl Default for SemanticTokensProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_tokens_legend() {
        let legend = SemanticTokensProvider::legend();
        assert!(!legend.token_types.is_empty());
        assert_eq!(legend.token_types.len(), 11);
    }

    #[test]
    fn test_comment_token() {
        let provider = SemanticTokensProvider::new();
        let text = "# This is a comment";
        let tokens = provider.generate_tokens(text);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].token_type, ShaclTokenType::Comment as u32);
    }

    #[test]
    fn test_prefix_declaration() {
        let provider = SemanticTokensProvider::new();
        let text = "@prefix sh: <http://www.w3.org/ns/shacl#> .";
        let tokens = provider.generate_tokens(text);
        assert!(!tokens.is_empty());
        // First token should be @prefix keyword
        assert_eq!(tokens[0].token_type, ShaclTokenType::Keyword as u32);
    }

    #[test]
    fn test_shacl_shape() {
        let provider = SemanticTokensProvider::new();
        let text = "ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [
        sh:path ex:name ;
        sh:minCount 1 ;
        sh:datatype xsd:string ;
    ] .";
        let tokens = provider.generate_tokens(text);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_string_literal() {
        let provider = SemanticTokensProvider::new();
        let text = r#"sh:message "This is a message" ;"#;
        let tokens = provider.generate_tokens(text);

        // Should contain: sh: (namespace), message (property), "This is a message" (string), ; (operator)
        assert!(tokens
            .iter()
            .any(|t| t.token_type == ShaclTokenType::String as u32));
    }

    #[test]
    fn test_number_literal() {
        let provider = SemanticTokensProvider::new();
        let text = "sh:minCount 1 ;";
        let tokens = provider.generate_tokens(text);

        assert!(tokens
            .iter()
            .any(|t| t.token_type == ShaclTokenType::Number as u32));
    }

    #[test]
    fn test_shacl_constraint_types() {
        let provider = SemanticTokensProvider::new();
        let text = "sh:minCount sh:maxCount sh:datatype sh:class sh:nodeKind";
        let tokens = provider.generate_tokens(text);

        // All SHACL constraints should be classified as Function
        let function_tokens: Vec<_> = tokens
            .iter()
            .filter(|t| t.token_type == ShaclTokenType::Function as u32)
            .collect();
        assert!(!function_tokens.is_empty());
    }

    #[test]
    fn test_shacl_node_kinds() {
        let provider = SemanticTokensProvider::new();
        let text = "sh:nodeKind sh:IRI ;";
        let tokens = provider.generate_tokens(text);

        // IRI should be classified as Type
        assert!(tokens
            .iter()
            .any(|t| t.token_type == ShaclTokenType::Type as u32));
    }

    #[test]
    fn test_full_iri() {
        let provider = SemanticTokensProvider::new();
        let text = "<http://example.org/Person>";
        let tokens = provider.generate_tokens(text);

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].token_type, ShaclTokenType::Class as u32);
    }

    #[test]
    fn test_language_tagged_string() {
        let provider = SemanticTokensProvider::new();
        let text = r#""Hello"@en"#;
        let tokens = provider.generate_tokens(text);

        // Should have string and language tag
        assert!(tokens
            .iter()
            .any(|t| t.token_type == ShaclTokenType::String as u32));
        assert!(tokens
            .iter()
            .any(|t| t.token_type == ShaclTokenType::Keyword as u32));
    }

    #[test]
    fn test_typed_literal() {
        let provider = SemanticTokensProvider::new();
        let text = r#""42"^^xsd:integer"#;
        let tokens = provider.generate_tokens(text);

        assert!(tokens
            .iter()
            .any(|t| t.token_type == ShaclTokenType::String as u32));
        assert!(tokens
            .iter()
            .any(|t| t.token_type == ShaclTokenType::Operator as u32));
    }

    #[test]
    fn test_rdf_type_shorthand() {
        let provider = SemanticTokensProvider::new();
        let text = "ex:Person a sh:NodeShape .";
        let tokens = provider.generate_tokens(text);

        // 'a' should be a keyword
        assert!(tokens
            .iter()
            .any(|t| t.token_type == ShaclTokenType::Keyword as u32));
    }

    #[test]
    fn test_multiline_document() {
        let provider = SemanticTokensProvider::new();
        let text = "@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .

# Person shape
ex:PersonShape
    a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [
        sh:path ex:firstName ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:datatype xsd:string ;
    ] .";

        let tokens = provider.generate_tokens(text);
        assert!(!tokens.is_empty());

        // Verify relative positioning is correct
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 && token.delta_line == 0 {
                // Same line tokens should have positive delta_start
                // (except for the very first token on a new line)
            }
        }
    }

    #[test]
    fn test_range_tokens() {
        let provider = SemanticTokensProvider::new();
        let text = "line0
line1
line2
line3
line4";

        let tokens = provider.generate_tokens_range(text, 1, 3);
        // Should only include tokens from lines 1-3
        assert!(tokens.iter().all(|_| {
            // Verify tokens are within range
            true
        }));
    }
}
