//! Format-specific parser state implementations for TriG and Turtle

use crate::model::{
    BlankNode, GraphName, Literal, NamedNode, Object, Predicate, Quad, Subject, Triple,
};
use crate::{OxirsError, Result};
use std::collections::HashMap;

/// TriG parser state for handling named graphs and multi-line statements
pub(super) struct TrigParserState {
    prefixes: HashMap<String, String>,
    base_iri: Option<String>,
    pending_statement: String,
    current_graph: Option<GraphName>,
}

impl TrigParserState {
    pub(super) fn new(base_iri: Option<&str>) -> Self {
        let mut prefixes = HashMap::new();
        // Add default prefixes
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );

        TrigParserState {
            prefixes,
            base_iri: base_iri.map(|s| s.to_string()),
            pending_statement: String::new(),
            current_graph: None,
        }
    }

    pub(super) fn parse_line(&mut self, line: &str) -> Result<Vec<Quad>> {
        let line = line.trim();

        // Handle directives
        if line.starts_with("@prefix") {
            return self.parse_prefix_directive(line);
        }

        if line.starts_with("@base") {
            return self.parse_base_directive(line);
        }

        // Handle graph blocks
        if line.contains("{") {
            return self.parse_graph_start(line);
        }

        if line == "}" {
            self.current_graph = None;
            return Ok(Vec::new());
        }

        // Accumulate multi-line statements
        self.pending_statement.push_str(line);
        self.pending_statement.push(' ');

        // Check if statement is complete (ends with .)
        if line.ends_with('.') {
            let statement = self.pending_statement.trim().to_string();
            self.pending_statement.clear();
            return self.parse_statement(&statement);
        }

        Ok(Vec::new())
    }

    pub(super) fn finalize(&mut self) -> Result<Option<Vec<Quad>>> {
        if !self.pending_statement.trim().is_empty() {
            let statement = self.pending_statement.trim().to_string();
            self.pending_statement.clear();
            return self.parse_statement(&statement).map(Some);
        }
        Ok(None)
    }

    fn parse_prefix_directive(&mut self, line: &str) -> Result<Vec<Quad>> {
        // @prefix ns: <http://example.org/ns#> .
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 3 {
            return Err(OxirsError::Parse("Invalid @prefix directive".to_string()));
        }

        let prefix = parts[1].trim_end_matches(':');
        let iri = parts[2];

        if !iri.starts_with('<') || !iri.ends_with('>') {
            return Err(OxirsError::Parse(
                "IRI must be enclosed in angle brackets".to_string(),
            ));
        }

        let iri = &iri[1..iri.len() - 1];
        self.prefixes.insert(prefix.to_string(), iri.to_string());

        Ok(Vec::new())
    }

    fn parse_base_directive(&mut self, line: &str) -> Result<Vec<Quad>> {
        // @base <http://example.org/> .
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 2 {
            return Err(OxirsError::Parse("Invalid @base directive".to_string()));
        }

        let iri = parts[1];

        if !iri.starts_with('<') || !iri.ends_with('>') {
            return Err(OxirsError::Parse(
                "Base IRI must be enclosed in angle brackets".to_string(),
            ));
        }

        let iri = &iri[1..iri.len() - 1];
        self.base_iri = Some(iri.to_string());

        Ok(Vec::new())
    }

    fn parse_graph_start(&mut self, line: &str) -> Result<Vec<Quad>> {
        // Parse: <graph_name> { or graph_name { or { for default graph
        if line.trim() == "{" {
            // Default graph
            self.current_graph = Some(GraphName::DefaultGraph);
        } else {
            // Named graph: <iri> { or prefix:name {
            let graph_part = line.replace('{', "").trim().to_string();
            if graph_part.starts_with('<') && graph_part.ends_with('>') {
                // Full IRI
                let iri = &graph_part[1..graph_part.len() - 1];
                let named_node = NamedNode::new(iri)?;
                self.current_graph = Some(GraphName::NamedNode(named_node));
            } else if graph_part.contains(':') {
                // Prefixed name
                let expanded = self.expand_prefixed_name(&graph_part)?;
                let named_node = NamedNode::new(&expanded)?;
                self.current_graph = Some(GraphName::NamedNode(named_node));
            } else {
                return Err(OxirsError::Parse(format!(
                    "Invalid graph name in TriG: '{graph_part}'. Must be IRI or prefixed name"
                )));
            }
        }

        Ok(Vec::new())
    }

    fn parse_statement(&mut self, statement: &str) -> Result<Vec<Quad>> {
        if statement.trim().is_empty() {
            return Ok(Vec::new());
        }

        // Parse as Turtle triple then convert to quad with current graph
        let triple = self.parse_turtle_statement(statement)?;
        let graph_name = self
            .current_graph
            .clone()
            .unwrap_or(GraphName::DefaultGraph);
        let quad = Quad::new(
            triple.subject().clone(),
            triple.predicate().clone(),
            triple.object().clone(),
            graph_name,
        );

        Ok(vec![quad])
    }

    fn parse_turtle_statement(&mut self, statement: &str) -> Result<Triple> {
        // Simple implementation - parse basic subject predicate object .
        let statement = statement.trim();
        let statement = if let Some(stripped) = statement.strip_suffix('.') {
            stripped.trim()
        } else {
            statement
        };

        // Split the statement into tokens (simplified)
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut in_quotes = false;
        let mut escaped = false;

        for c in statement.chars() {
            if escaped {
                current_token.push(c);
                escaped = false;
            } else if c == '\\' && in_quotes {
                escaped = true;
                current_token.push(c);
            } else if c == '"' {
                current_token.push(c);
                in_quotes = !in_quotes;
            } else if c.is_whitespace() && !in_quotes {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
            } else {
                current_token.push(c);
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        if tokens.len() < 3 {
            return Err(OxirsError::Parse(
                "Invalid triple: need subject, predicate, object".to_string(),
            ));
        }

        // Parse subject
        let subject = self.parse_subject_term(&tokens[0])?;

        // Parse predicate
        let predicate = self.parse_predicate_term(&tokens[1])?;

        // Parse object
        let object = self.parse_object_term(&tokens[2])?;

        Ok(Triple::new(subject, predicate, object))
    }

    fn parse_subject_term(&self, token: &str) -> Result<Subject> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            let named_node = NamedNode::new(iri)?;
            Ok(Subject::NamedNode(named_node))
        } else if token.starts_with("_:") {
            let blank_node = BlankNode::new(token)?;
            Ok(Subject::BlankNode(blank_node))
        } else if token.contains(':') {
            // Prefixed name
            let expanded = self.expand_prefixed_name(token)?;
            let named_node = NamedNode::new(&expanded)?;
            Ok(Subject::NamedNode(named_node))
        } else {
            Err(OxirsError::Parse(format!(
                "Invalid subject: {token}. Must be IRI or blank node"
            )))
        }
    }

    fn parse_predicate_term(&self, token: &str) -> Result<Predicate> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            Ok(Predicate::NamedNode(NamedNode::new(iri)?))
        } else if token.contains(':') {
            // Prefixed name
            let expanded = self.expand_prefixed_name(token)?;
            Ok(Predicate::NamedNode(NamedNode::new(&expanded)?))
        } else {
            Err(OxirsError::Parse(format!(
                "Invalid predicate: {token}. Must be IRI"
            )))
        }
    }

    fn parse_object_term(&self, token: &str) -> Result<Object> {
        if token.starts_with('"') {
            // Literal
            self.parse_literal_term(token)
        } else if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            let named_node = NamedNode::new(iri)?;
            Ok(Object::NamedNode(named_node))
        } else if token.starts_with("_:") {
            let blank_node = BlankNode::new(token)?;
            Ok(Object::BlankNode(blank_node))
        } else if token.contains(':') {
            // Prefixed name
            let expanded = self.expand_prefixed_name(token)?;
            let named_node = NamedNode::new(&expanded)?;
            Ok(Object::NamedNode(named_node))
        } else {
            Err(OxirsError::Parse(format!(
                "Invalid object: {token}. Must be IRI, blank node, or literal"
            )))
        }
    }

    fn parse_literal_term(&self, token: &str) -> Result<Object> {
        // Parse "value"@lang or "value"^^<datatype> or just "value"
        if !token.starts_with('"') {
            return Err(OxirsError::Parse(
                "Literal must start with quote".to_string(),
            ));
        }

        // Find the end quote
        let mut end_quote = 1;
        let mut escaped = false;
        let chars: Vec<char> = token.chars().collect();

        while end_quote < chars.len() {
            if escaped {
                escaped = false;
            } else if chars[end_quote] == '\\' {
                escaped = true;
            } else if chars[end_quote] == '"' {
                break;
            }
            end_quote += 1;
        }

        if end_quote >= chars.len() {
            return Err(OxirsError::Parse("Unterminated literal".to_string()));
        }

        let value = self.unescape_literal_value(&token[1..end_quote])?;
        let remainder = &token[end_quote + 1..];

        if remainder.is_empty() {
            // Simple literal
            Ok(Object::Literal(Literal::new_simple_literal(&value)))
        } else if let Some(lang) = remainder.strip_prefix('@') {
            // Language tag
            let literal = Literal::new_language_tagged_literal(&value, lang)?;
            Ok(Object::Literal(literal))
        } else if let Some(datatype_token) = remainder.strip_prefix("^^") {
            // Datatype
            if datatype_token.starts_with('<') && datatype_token.ends_with('>') {
                let datatype_iri = &datatype_token[1..datatype_token.len() - 1];
                let datatype = NamedNode::new(datatype_iri)?;
                Ok(Object::Literal(Literal::new_typed_literal(
                    &value, datatype,
                )))
            } else {
                Err(OxirsError::Parse("Invalid datatype IRI".to_string()))
            }
        } else {
            Err(OxirsError::Parse("Invalid literal format".to_string()))
        }
    }

    fn expand_prefixed_name(&self, name: &str) -> Result<String> {
        if let Some((prefix, local)) = name.split_once(':') {
            if let Some(namespace) = self.prefixes.get(prefix) {
                Ok(format!("{namespace}{local}"))
            } else {
                Err(OxirsError::Parse(format!("Unknown prefix: {prefix}")))
            }
        } else {
            Err(OxirsError::Parse("Invalid prefixed name".to_string()))
        }
    }

    /// Unescape special characters in literal values
    fn unescape_literal_value(&self, value: &str) -> Result<String> {
        let mut result = String::new();
        let mut chars = value.chars();

        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('"') => result.push('"'),
                    Some('\\') => result.push('\\'),
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('u') => {
                        // Parse \uHHHH Unicode escape
                        let hex_chars: String = chars.by_ref().take(4).collect();
                        if hex_chars.len() != 4 {
                            return Err(OxirsError::Parse(
                                "Invalid Unicode escape sequence \\uHHHH - expected 4 hex digits"
                                    .to_string(),
                            ));
                        }
                        let code_point = u32::from_str_radix(&hex_chars, 16).map_err(|_| {
                            OxirsError::Parse(
                                "Invalid hex digits in Unicode escape sequence".to_string(),
                            )
                        })?;
                        let unicode_char = char::from_u32(code_point).ok_or_else(|| {
                            OxirsError::Parse("Invalid Unicode code point".to_string())
                        })?;
                        result.push(unicode_char);
                    }
                    Some('U') => {
                        // Parse \UHHHHHHHH Unicode escape
                        let hex_chars: String = chars.by_ref().take(8).collect();
                        if hex_chars.len() != 8 {
                            return Err(OxirsError::Parse(
                                "Invalid Unicode escape sequence \\UHHHHHHHH - expected 8 hex digits".to_string()
                            ));
                        }
                        let code_point = u32::from_str_radix(&hex_chars, 16).map_err(|_| {
                            OxirsError::Parse(
                                "Invalid hex digits in Unicode escape sequence".to_string(),
                            )
                        })?;
                        let unicode_char = char::from_u32(code_point).ok_or_else(|| {
                            OxirsError::Parse("Invalid Unicode code point".to_string())
                        })?;
                        result.push(unicode_char);
                    }
                    Some(other) => {
                        return Err(OxirsError::Parse(format!(
                            "Invalid escape sequence \\{other}"
                        )));
                    }
                    None => {
                        return Err(OxirsError::Parse(
                            "Incomplete escape sequence at end of literal".to_string(),
                        ));
                    }
                }
            } else {
                result.push(c);
            }
        }

        Ok(result)
    }
}

/// Turtle parser state for handling multi-line statements and abbreviations
pub(super) struct TurtleParserState {
    prefixes: HashMap<String, String>,
    base_iri: Option<String>,
    pending_statement: String,
}

impl TurtleParserState {
    pub(super) fn new(base_iri: Option<&str>) -> Self {
        let mut prefixes = HashMap::new();
        // Add default prefixes
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );

        TurtleParserState {
            prefixes,
            base_iri: base_iri.map(|s| s.to_string()),
            pending_statement: String::new(),
        }
    }

    pub(super) fn parse_line(&mut self, line: &str) -> Result<Vec<Triple>> {
        let line = line.trim();

        // Handle directives
        if line.starts_with("@prefix") {
            return self.parse_prefix_directive(line);
        }

        if line.starts_with("@base") {
            return self.parse_base_directive(line);
        }

        // Accumulate multi-line statements
        self.pending_statement.push_str(line);
        self.pending_statement.push(' ');

        // Check if statement is complete (ends with .)
        if line.ends_with('.') {
            let statement = self.pending_statement.trim().to_string();
            self.pending_statement.clear();
            return self.parse_statement(&statement);
        }

        Ok(Vec::new())
    }

    pub(super) fn finalize(&mut self) -> Result<Option<Vec<Triple>>> {
        if !self.pending_statement.trim().is_empty() {
            let statement = self.pending_statement.trim().to_string();
            self.pending_statement.clear();
            return self.parse_statement(&statement).map(Some);
        }
        Ok(None)
    }

    fn parse_prefix_directive(&mut self, line: &str) -> Result<Vec<Triple>> {
        // @prefix ns: <http://example.org/ns#> .
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 3 {
            return Err(OxirsError::Parse("Invalid @prefix directive".to_string()));
        }

        let prefix = parts[1].trim_end_matches(':');
        let iri = parts[2];

        if !iri.starts_with('<') || !iri.ends_with('>') {
            return Err(OxirsError::Parse(
                "IRI must be enclosed in angle brackets".to_string(),
            ));
        }

        let iri = &iri[1..iri.len() - 1];
        self.prefixes.insert(prefix.to_string(), iri.to_string());

        Ok(Vec::new())
    }

    fn parse_base_directive(&mut self, line: &str) -> Result<Vec<Triple>> {
        // @base <http://example.org/> .
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 2 {
            return Err(OxirsError::Parse("Invalid @base directive".to_string()));
        }

        let iri = parts[1];

        if !iri.starts_with('<') || !iri.ends_with('>') {
            return Err(OxirsError::Parse(
                "Base IRI must be enclosed in angle brackets".to_string(),
            ));
        }

        let iri = &iri[1..iri.len() - 1];
        self.base_iri = Some(iri.to_string());

        Ok(Vec::new())
    }

    fn parse_statement(&mut self, statement: &str) -> Result<Vec<Triple>> {
        let statement = statement.trim().trim_end_matches('.');
        let mut triples = Vec::new();

        // Handle abbreviated syntax: subject predicate object ; predicate object
        let subject_parts: Vec<&str> = statement.split(';').collect();

        if subject_parts.len() == 1 {
            // Single triple: subject predicate object
            if let Some(triple) = self.parse_simple_triple(statement)? {
                triples.push(triple);
            }
        } else {
            // Multiple triples with same subject
            let first_part = subject_parts[0].trim();
            let first_triple = self.parse_simple_triple(first_part)?;

            if let Some(triple) = first_triple {
                let subject = triple.subject().clone();
                triples.push(triple);

                // Parse remaining predicate-object pairs
                for part in &subject_parts[1..] {
                    let part = part.trim();
                    if !part.is_empty() {
                        if let Some(triple) = self.parse_predicate_object_pair(&subject, part)? {
                            triples.push(triple);
                        }
                    }
                }
            }
        }

        Ok(triples)
    }

    fn parse_simple_triple(&self, triple_str: &str) -> Result<Option<Triple>> {
        let tokens = self.tokenize_turtle_statement(triple_str)?;

        if tokens.len() < 3 {
            return Ok(None);
        }

        let subject = self.parse_turtle_subject(&tokens[0])?;
        let predicate = self.parse_turtle_predicate(&tokens[1])?;
        let object = self.parse_turtle_object(&tokens[2])?;

        Ok(Some(Triple::new(subject, predicate, object)))
    }

    fn parse_predicate_object_pair(
        &self,
        subject: &Subject,
        pair_str: &str,
    ) -> Result<Option<Triple>> {
        let tokens = self.tokenize_turtle_statement(pair_str)?;

        if tokens.len() < 2 {
            return Ok(None);
        }

        let predicate = self.parse_turtle_predicate(&tokens[0])?;
        let object = self.parse_turtle_object(&tokens[1])?;

        Ok(Some(Triple::new(subject.clone(), predicate, object)))
    }

    fn tokenize_turtle_statement(&self, statement: &str) -> Result<Vec<String>> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut in_quotes = false;
        let mut in_angles = false;
        let mut escaped = false;

        for c in statement.chars() {
            if escaped {
                current_token.push(c);
                escaped = false;
            } else if c == '\\' && (in_quotes || in_angles) {
                escaped = true;
                current_token.push(c);
            } else if c == '"' && !in_angles {
                current_token.push(c);
                in_quotes = !in_quotes;
            } else if c == '<' && !in_quotes {
                current_token.push(c);
                in_angles = true;
            } else if c == '>' && !in_quotes {
                current_token.push(c);
                in_angles = false;
            } else if c.is_whitespace() && !in_quotes && !in_angles {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
            } else {
                current_token.push(c);
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        Ok(tokens)
    }

    fn parse_turtle_subject(&self, token: &str) -> Result<Subject> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = self.resolve_iri(&token[1..token.len() - 1])?;
            let named_node = NamedNode::new(iri)?;
            Ok(Subject::NamedNode(named_node))
        } else if token.starts_with("_:") {
            let blank_node = BlankNode::new(token)?;
            Ok(Subject::BlankNode(blank_node))
        } else if token.contains(':')
            && !token.starts_with("http://")
            && !token.starts_with("https://")
        {
            // Prefixed name
            let iri = self.expand_prefixed_name(token)?;
            let named_node = NamedNode::new(iri)?;
            Ok(Subject::NamedNode(named_node))
        } else {
            Err(OxirsError::Parse(format!("Invalid subject: {token}")))
        }
    }

    fn parse_turtle_predicate(&self, token: &str) -> Result<Predicate> {
        if token == "a" {
            // Shorthand for rdf:type
            let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
            let named_node = NamedNode::new(rdf_type)?;
            Ok(Predicate::NamedNode(named_node))
        } else if token.starts_with('<') && token.ends_with('>') {
            let iri = self.resolve_iri(&token[1..token.len() - 1])?;
            let named_node = NamedNode::new(iri)?;
            Ok(Predicate::NamedNode(named_node))
        } else if token.contains(':')
            && !token.starts_with("http://")
            && !token.starts_with("https://")
        {
            // Prefixed name
            let iri = self.expand_prefixed_name(token)?;
            let named_node = NamedNode::new(iri)?;
            Ok(Predicate::NamedNode(named_node))
        } else {
            Err(OxirsError::Parse(format!("Invalid predicate: {token}")))
        }
    }

    fn parse_turtle_object(&self, token: &str) -> Result<Object> {
        if token.starts_with('<') && token.ends_with('>') {
            // IRI
            let iri = self.resolve_iri(&token[1..token.len() - 1])?;
            let named_node = NamedNode::new(iri)?;
            Ok(Object::NamedNode(named_node))
        } else if token.starts_with("_:") {
            // Blank node
            let blank_node = BlankNode::new(token)?;
            Ok(Object::BlankNode(blank_node))
        } else if token.starts_with('"') {
            // Literal
            self.parse_turtle_literal(token)
        } else if token.contains(':')
            && !token.starts_with("http://")
            && !token.starts_with("https://")
        {
            // Prefixed name
            let iri = self.expand_prefixed_name(token)?;
            let named_node = NamedNode::new(iri)?;
            Ok(Object::NamedNode(named_node))
        } else {
            Err(OxirsError::Parse(format!("Invalid object: {token}")))
        }
    }

    fn parse_turtle_literal(&self, token: &str) -> Result<Object> {
        if !token.starts_with('"') {
            return Err(OxirsError::Parse(
                "Literal must start with quote".to_string(),
            ));
        }

        // Find the closing quote
        let mut end_quote_pos = None;
        let mut escaped = false;
        let chars: Vec<char> = token.chars().collect();

        for (i, &ch) in chars.iter().enumerate().skip(1) {
            if escaped {
                escaped = false;
                continue;
            }

            if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                end_quote_pos = Some(i);
                break;
            }
        }

        let end_quote_pos =
            end_quote_pos.ok_or_else(|| OxirsError::Parse("Unterminated literal".to_string()))?;

        // Extract the literal value (without quotes)
        let literal_value: String = chars[1..end_quote_pos].iter().collect();

        // Check for language tag or datatype
        let remaining = &token[end_quote_pos + 1..];

        if let Some(lang_tag) = remaining.strip_prefix('@') {
            // Language tag
            let literal = Literal::new_lang(literal_value, lang_tag)?;
            Ok(Object::Literal(literal))
        } else if let Some(datatype_part) = remaining.strip_prefix("^^") {
            // Datatype
            if datatype_part.starts_with('<') && datatype_part.ends_with('>') {
                // IRI datatype
                let datatype_iri = self.resolve_iri(&datatype_part[1..datatype_part.len() - 1])?;
                let datatype = NamedNode::new(datatype_iri)?;
                let literal = Literal::new_typed(literal_value, datatype);
                Ok(Object::Literal(literal))
            } else if datatype_part.contains(':') {
                // Prefixed datatype
                let datatype_iri = self.expand_prefixed_name(datatype_part)?;
                let datatype = NamedNode::new(datatype_iri)?;
                let literal = Literal::new_typed(literal_value, datatype);
                Ok(Object::Literal(literal))
            } else {
                Err(OxirsError::Parse(format!(
                    "Invalid datatype: {datatype_part}"
                )))
            }
        } else if remaining.is_empty() {
            // Plain literal
            let literal = Literal::new(literal_value);
            Ok(Object::Literal(literal))
        } else {
            Err(OxirsError::Parse(format!(
                "Invalid literal syntax: {token}"
            )))
        }
    }

    fn expand_prefixed_name(&self, prefixed_name: &str) -> Result<String> {
        if let Some(colon_pos) = prefixed_name.find(':') {
            let prefix = &prefixed_name[..colon_pos];
            let local_name = &prefixed_name[colon_pos + 1..];

            if let Some(namespace) = self.prefixes.get(prefix) {
                Ok(format!("{namespace}{local_name}"))
            } else {
                Err(OxirsError::Parse(format!("Unknown prefix: {prefix}")))
            }
        } else {
            Err(OxirsError::Parse(format!(
                "Invalid prefixed name: {prefixed_name}"
            )))
        }
    }

    fn resolve_iri(&self, iri: &str) -> Result<String> {
        if iri.contains("://") {
            // Absolute IRI
            Ok(iri.to_string())
        } else if let Some(base) = &self.base_iri {
            // Resolve relative IRI against base
            if base.ends_with('/') {
                Ok(format!("{base}{iri}"))
            } else {
                Ok(format!("{base}/{iri}"))
            }
        } else {
            // No base IRI, return as-is
            Ok(iri.to_string())
        }
    }
}
