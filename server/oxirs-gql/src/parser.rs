//! GraphQL query parser implementation
//!
//! This module provides a GraphQL parser that converts GraphQL query strings
//! into AST nodes for execution.

use crate::ast::*;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::iter::Peekable;
use std::str::Chars;

/// GraphQL parser for converting query strings to AST
pub struct Parser {
    chars: Peekable<Chars<'static>>,
    position: usize,
    line: usize,
    column: usize,
}

impl Parser {
    pub fn new(input: &str) -> Self {
        // SAFETY: We ensure the string lives long enough for parsing
        let input: &'static str = unsafe { std::mem::transmute(input) };
        Self {
            chars: input.chars().peekable(),
            position: 0,
            line: 1,
            column: 1,
        }
    }

    /// Parse a GraphQL document
    pub fn parse_document(&mut self) -> Result<Document> {
        self.skip_whitespace();
        let mut definitions = Vec::new();

        while self.chars.peek().is_some() {
            definitions.push(self.parse_definition()?);
            self.skip_whitespace();
        }

        if definitions.is_empty() {
            return Err(anyhow!("Empty document"));
        }

        Ok(Document { definitions })
    }

    fn parse_definition(&mut self) -> Result<Definition> {
        self.skip_whitespace();

        if self.peek_keyword("query")
            || self.peek_keyword("mutation")
            || self.peek_keyword("subscription")
            || self.peek_char('{')
        {
            Ok(Definition::Operation(self.parse_operation_definition()?))
        } else if self.peek_keyword("fragment") {
            Ok(Definition::Fragment(self.parse_fragment_definition()?))
        } else {
            Err(anyhow!("Unexpected definition at line {}", self.line))
        }
    }

    fn parse_operation_definition(&mut self) -> Result<OperationDefinition> {
        self.skip_whitespace();

        let operation_type = if self.peek_char('{') {
            OperationType::Query // Shorthand query
        } else {
            self.parse_operation_type()?
        };

        let name = if self.peek_name() && !self.peek_char('{') {
            Some(self.parse_name()?)
        } else {
            None
        };

        let variable_definitions = if self.peek_char('(') {
            self.parse_variable_definitions()?
        } else {
            Vec::new()
        };

        let directives = self.parse_directives()?;
        let selection_set = self.parse_selection_set()?;

        Ok(OperationDefinition {
            operation_type,
            name,
            variable_definitions,
            directives,
            selection_set,
        })
    }

    fn parse_operation_type(&mut self) -> Result<OperationType> {
        let keyword = self.parse_name()?;
        match keyword.as_str() {
            "query" => Ok(OperationType::Query),
            "mutation" => Ok(OperationType::Mutation),
            "subscription" => Ok(OperationType::Subscription),
            _ => Err(anyhow!("Invalid operation type: {}", keyword)),
        }
    }

    fn parse_variable_definitions(&mut self) -> Result<Vec<VariableDefinition>> {
        self.expect_char('(')?;
        let mut definitions = Vec::new();

        while !self.peek_char(')') {
            definitions.push(self.parse_variable_definition()?);
            if self.peek_char(',') {
                self.next_char();
            }
            self.skip_whitespace();
        }

        self.expect_char(')')?;
        Ok(definitions)
    }

    fn parse_variable_definition(&mut self) -> Result<VariableDefinition> {
        let variable = self.parse_variable()?;
        self.expect_char(':')?;
        let type_ = self.parse_type()?;

        let default_value = if self.peek_char('=') {
            self.next_char();
            Some(self.parse_value()?)
        } else {
            None
        };

        let directives = self.parse_directives()?;

        Ok(VariableDefinition {
            variable,
            type_,
            default_value,
            directives,
        })
    }

    fn parse_variable(&mut self) -> Result<Variable> {
        self.expect_char('$')?;
        let name = self.parse_name()?;
        Ok(Variable { name })
    }

    fn parse_type(&mut self) -> Result<Type> {
        self.skip_whitespace();
        let mut type_ = if self.peek_char('[') {
            self.next_char(); // consume '['
            let inner_type = self.parse_type()?;
            self.expect_char(']')?;
            Type::ListType(Box::new(inner_type))
        } else {
            let name = self.parse_name()?;
            Type::NamedType(name)
        };

        if self.peek_char('!') {
            self.next_char();
            type_ = Type::NonNullType(Box::new(type_));
        }

        Ok(type_)
    }

    fn parse_selection_set(&mut self) -> Result<SelectionSet> {
        self.expect_char('{')?;
        let mut selections = Vec::new();

        while !self.peek_char('}') {
            selections.push(self.parse_selection()?);
            self.skip_whitespace();
        }

        self.expect_char('}')?;
        Ok(SelectionSet { selections })
    }

    fn parse_selection(&mut self) -> Result<Selection> {
        self.skip_whitespace();

        if self.peek_keyword("...") {
            self.consume_keyword("...")?;
            if self.peek_keyword("on") {
                Ok(Selection::InlineFragment(self.parse_inline_fragment()?))
            } else {
                Ok(Selection::FragmentSpread(self.parse_fragment_spread()?))
            }
        } else {
            Ok(Selection::Field(self.parse_field()?))
        }
    }

    fn parse_field(&mut self) -> Result<Field> {
        let mut name = self.parse_name()?;
        let mut alias = None;

        // Check for alias
        if self.peek_char(':') {
            self.next_char();
            alias = Some(name);
            name = self.parse_name()?;
        }

        let arguments = if self.peek_char('(') {
            self.parse_arguments()?
        } else {
            Vec::new()
        };

        let directives = self.parse_directives()?;

        let selection_set = if self.peek_char('{') {
            Some(self.parse_selection_set()?)
        } else {
            None
        };

        Ok(Field {
            alias,
            name,
            arguments,
            directives,
            selection_set,
        })
    }

    fn parse_arguments(&mut self) -> Result<Vec<Argument>> {
        self.expect_char('(')?;
        let mut arguments = Vec::new();

        while !self.peek_char(')') {
            arguments.push(self.parse_argument()?);
            if self.peek_char(',') {
                self.next_char();
            }
            self.skip_whitespace();
        }

        self.expect_char(')')?;
        Ok(arguments)
    }

    fn parse_argument(&mut self) -> Result<Argument> {
        let name = self.parse_name()?;
        self.expect_char(':')?;
        let value = self.parse_value()?;

        Ok(Argument { name, value })
    }

    fn parse_value(&mut self) -> Result<Value> {
        self.skip_whitespace();

        if self.peek_char('$') {
            Ok(Value::Variable(self.parse_variable()?))
        } else if self.peek_char('"') {
            Ok(Value::StringValue(self.parse_string()?))
        } else if self.peek_char('[') {
            Ok(Value::ListValue(self.parse_list_value()?))
        } else if self.peek_char('{') {
            Ok(Value::ObjectValue(self.parse_object_value()?))
        } else if self.peek_keyword("null") {
            self.consume_keyword("null")?;
            Ok(Value::NullValue)
        } else if self.peek_keyword("true") {
            self.consume_keyword("true")?;
            Ok(Value::BooleanValue(true))
        } else if self.peek_keyword("false") {
            self.consume_keyword("false")?;
            Ok(Value::BooleanValue(false))
        } else if self.peek_numeric() {
            self.parse_numeric_value()
        } else {
            // Enum value
            let name = self.parse_name()?;
            Ok(Value::EnumValue(name))
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.expect_char('"')?;
        let mut value = String::new();

        while !self.peek_char('"') {
            if self.peek_char('\\') {
                self.next_char(); // consume '\'
                match self.next_char() {
                    Some('"') => value.push('"'),
                    Some('\\') => value.push('\\'),
                    Some('/') => value.push('/'),
                    Some('b') => value.push('\u{0008}'),
                    Some('f') => value.push('\u{000C}'),
                    Some('n') => value.push('\n'),
                    Some('r') => value.push('\r'),
                    Some('t') => value.push('\t'),
                    Some('u') => {
                        // Unicode escape sequence
                        let mut hex = String::new();
                        for _ in 0..4 {
                            if let Some(ch) = self.next_char() {
                                hex.push(ch);
                            } else {
                                return Err(anyhow!("Incomplete unicode escape"));
                            }
                        }
                        let code = u32::from_str_radix(&hex, 16)
                            .map_err(|_| anyhow!("Invalid unicode escape"))?;
                        if let Some(unicode_char) = std::char::from_u32(code) {
                            value.push(unicode_char);
                        } else {
                            return Err(anyhow!("Invalid unicode code point"));
                        }
                    }
                    Some(ch) => return Err(anyhow!("Invalid escape sequence: \\{}", ch)),
                    None => return Err(anyhow!("Unterminated string")),
                }
            } else {
                match self.next_char() {
                    Some(ch) => value.push(ch),
                    None => return Err(anyhow!("Unterminated string")),
                }
            }
        }

        self.expect_char('"')?;
        Ok(value)
    }

    fn parse_list_value(&mut self) -> Result<Vec<Value>> {
        self.expect_char('[')?;
        let mut values = Vec::new();

        while !self.peek_char(']') {
            values.push(self.parse_value()?);
            if self.peek_char(',') {
                self.next_char();
            }
            self.skip_whitespace();
        }

        self.expect_char(']')?;
        Ok(values)
    }

    fn parse_object_value(&mut self) -> Result<HashMap<String, Value>> {
        self.expect_char('{')?;
        let mut object = HashMap::new();

        while !self.peek_char('}') {
            let name = self.parse_name()?;
            self.expect_char(':')?;
            let value = self.parse_value()?;
            object.insert(name, value);

            if self.peek_char(',') {
                self.next_char();
            }
            self.skip_whitespace();
        }

        self.expect_char('}')?;
        Ok(object)
    }

    fn parse_numeric_value(&mut self) -> Result<Value> {
        let mut number = String::new();

        // Handle negative sign
        if self.peek_char('-') {
            number.push(self.next_char().unwrap());
        }

        // Parse integer part
        while let Some(&ch) = self.chars.peek() {
            if ch.is_ascii_digit() {
                number.push(self.next_char().unwrap());
            } else {
                break;
            }
        }

        // Check for decimal point
        if self.peek_char('.') {
            number.push(self.next_char().unwrap());

            while let Some(&ch) = self.chars.peek() {
                if ch.is_ascii_digit() {
                    number.push(self.next_char().unwrap());
                } else {
                    break;
                }
            }

            // Parse as float
            number
                .parse::<f64>()
                .map(Value::FloatValue)
                .map_err(|_| anyhow!("Invalid float: {}", number))
        } else {
            // Check for exponent
            if self.peek_char('e') || self.peek_char('E') {
                number.push(self.next_char().unwrap());

                if self.peek_char('+') || self.peek_char('-') {
                    number.push(self.next_char().unwrap());
                }

                while let Some(&ch) = self.chars.peek() {
                    if ch.is_ascii_digit() {
                        number.push(self.next_char().unwrap());
                    } else {
                        break;
                    }
                }

                // Parse as float
                number
                    .parse::<f64>()
                    .map(Value::FloatValue)
                    .map_err(|_| anyhow!("Invalid float: {}", number))
            } else {
                // Parse as integer
                number
                    .parse::<i64>()
                    .map(Value::IntValue)
                    .map_err(|_| anyhow!("Invalid integer: {}", number))
            }
        }
    }

    fn parse_fragment_definition(&mut self) -> Result<FragmentDefinition> {
        self.consume_keyword("fragment")?;
        let name = self.parse_name()?;
        self.consume_keyword("on")?;
        let type_condition = self.parse_name()?;
        let directives = self.parse_directives()?;
        let selection_set = self.parse_selection_set()?;

        Ok(FragmentDefinition {
            name,
            type_condition,
            directives,
            selection_set,
        })
    }

    fn parse_fragment_spread(&mut self) -> Result<FragmentSpread> {
        let fragment_name = self.parse_name()?;
        let directives = self.parse_directives()?;

        Ok(FragmentSpread {
            fragment_name,
            directives,
        })
    }

    fn parse_inline_fragment(&mut self) -> Result<InlineFragment> {
        let type_condition = if self.peek_keyword("on") {
            self.consume_keyword("on")?;
            Some(self.parse_name()?)
        } else {
            None
        };

        let directives = self.parse_directives()?;
        let selection_set = self.parse_selection_set()?;

        Ok(InlineFragment {
            type_condition,
            directives,
            selection_set,
        })
    }

    fn parse_directives(&mut self) -> Result<Vec<Directive>> {
        let mut directives = Vec::new();

        while self.peek_char('@') {
            directives.push(self.parse_directive()?);
        }

        Ok(directives)
    }

    fn parse_directive(&mut self) -> Result<Directive> {
        self.expect_char('@')?;
        let name = self.parse_name()?;

        let arguments = if self.peek_char('(') {
            self.parse_arguments()?
        } else {
            Vec::new()
        };

        Ok(Directive { name, arguments })
    }

    fn parse_name(&mut self) -> Result<String> {
        self.skip_whitespace();
        let mut name = String::new();

        // First character must be letter or underscore
        match self.chars.peek() {
            Some(&ch) if ch.is_alphabetic() || ch == '_' => {
                name.push(self.next_char().unwrap());
            }
            _ => return Err(anyhow!("Expected name at line {}", self.line)),
        }

        // Subsequent characters can be letters, digits, or underscores
        while let Some(&ch) = self.chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                name.push(self.next_char().unwrap());
            } else {
                break;
            }
        }

        Ok(name)
    }

    // Helper methods

    fn next_char(&mut self) -> Option<char> {
        if let Some(ch) = self.chars.next() {
            self.position += 1;
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            Some(ch)
        } else {
            None
        }
    }

    fn peek_char(&mut self, expected: char) -> bool {
        self.skip_whitespace();
        self.chars.peek() == Some(&expected)
    }

    fn peek_name(&mut self) -> bool {
        self.skip_whitespace();
        matches!(self.chars.peek(), Some(&ch) if ch.is_alphabetic() || ch == '_')
    }

    fn peek_numeric(&mut self) -> bool {
        self.skip_whitespace();
        matches!(self.chars.peek(), Some(&ch) if ch.is_ascii_digit() || ch == '-')
    }

    fn peek_keyword(&mut self, keyword: &str) -> bool {
        self.skip_whitespace();

        let chars: Vec<char> = self.chars.clone().take(keyword.len()).collect();
        let word: String = chars.into_iter().collect();

        if word == keyword {
            // Check that it's not part of a larger identifier
            let next_chars: Vec<char> = self.chars.clone().skip(keyword.len()).take(1).collect();
            if let Some(&next_ch) = next_chars.first() {
                !next_ch.is_alphanumeric() && next_ch != '_'
            } else {
                true
            }
        } else {
            false
        }
    }

    fn consume_keyword(&mut self, keyword: &str) -> Result<()> {
        self.skip_whitespace();

        for expected_ch in keyword.chars() {
            match self.next_char() {
                Some(ch) if ch == expected_ch => continue,
                Some(ch) => return Err(anyhow!("Expected '{}', found '{}'", expected_ch, ch)),
                None => return Err(anyhow!("Unexpected end of input")),
            }
        }

        Ok(())
    }

    fn expect_char(&mut self, expected: char) -> Result<()> {
        self.skip_whitespace();
        match self.next_char() {
            Some(ch) if ch == expected => Ok(()),
            Some(ch) => Err(anyhow!("Expected '{}', found '{}'", expected, ch)),
            None => Err(anyhow!("Expected '{}', found end of input", expected)),
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(&ch) = self.chars.peek() {
            if ch.is_whitespace() {
                self.next_char();
            } else if ch == '#' {
                // Skip comments
                while let Some(&ch) = self.chars.peek() {
                    self.next_char();
                    if ch == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }
}

/// Parse a GraphQL document from a string
pub fn parse_document(input: &str) -> Result<Document> {
    let mut parser = Parser::new(input);
    parser.parse_document()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_query() {
        let query = "{ hello }";
        let doc = parse_document(query).unwrap();

        assert_eq!(doc.definitions.len(), 1);
        if let Definition::Operation(op) = &doc.definitions[0] {
            assert!(matches!(op.operation_type, OperationType::Query));
            assert_eq!(op.selection_set.selections.len(), 1);
        }
    }

    #[test]
    fn test_named_query() {
        let query = "query GetUser { user { id name } }";
        let doc = parse_document(query).unwrap();

        assert_eq!(doc.definitions.len(), 1);
        if let Definition::Operation(op) = &doc.definitions[0] {
            assert!(matches!(op.operation_type, OperationType::Query));
            assert_eq!(op.name, Some("GetUser".to_string()));
        }
    }

    #[test]
    fn test_query_with_arguments() {
        let query = r#"{ user(id: "123") { name } }"#;
        let doc = parse_document(query).unwrap();

        assert_eq!(doc.definitions.len(), 1);
        if let Definition::Operation(op) = &doc.definitions[0] {
            if let Selection::Field(field) = &op.selection_set.selections[0] {
                assert_eq!(field.name, "user");
                assert_eq!(field.arguments.len(), 1);
                assert_eq!(field.arguments[0].name, "id");
            }
        }
    }

    #[test]
    fn test_query_with_variables() {
        let query = "query GetUser($id: ID!) { user(id: $id) { name } }";
        let doc = parse_document(query).unwrap();

        assert_eq!(doc.definitions.len(), 1);
        if let Definition::Operation(op) = &doc.definitions[0] {
            assert_eq!(op.variable_definitions.len(), 1);
            assert_eq!(op.variable_definitions[0].variable.name, "id");
        }
    }

    #[test]
    fn test_fragment() {
        let query = "fragment UserFields on User { id name email }";
        let doc = parse_document(query).unwrap();

        assert_eq!(doc.definitions.len(), 1);
        if let Definition::Fragment(frag) = &doc.definitions[0] {
            assert_eq!(frag.name, "UserFields");
            assert_eq!(frag.type_condition, "User");
            assert_eq!(frag.selection_set.selections.len(), 3);
        }
    }
}
