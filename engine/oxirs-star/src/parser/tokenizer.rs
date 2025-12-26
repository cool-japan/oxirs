//! Tokenization utilities for RDF-star parsing.
//!
//! This module provides low-level tokenization functions used by the StarParser
//! to process RDF-star syntax including quoted triples, literals, and graph blocks.

use anyhow::Result;

use super::context::TrigParserState;

/// Strip inline comments from a line, respecting string boundaries.
///
/// Comments start with `#` outside of strings. This function handles:
/// - Escaped characters within strings (`\"`)
/// - Proper string boundary detection
/// - UTF-8 character boundaries
///
/// # Examples
///
/// ```ignore
/// # use oxirs_star::parser::tokenizer::strip_inline_comment;
/// assert_eq!(strip_inline_comment("ex:foo ex:bar \"value\" . # comment"), "ex:foo ex:bar \"value\" . ");
/// assert_eq!(strip_inline_comment("ex:foo ex:bar \"value # not a comment\" ."), "ex:foo ex:bar \"value # not a comment\" .");
/// ```
pub fn strip_inline_comment(line: &str) -> &str {
    let mut in_string = false;
    let mut escape_next = false;
    let mut byte_index = 0;

    for ch in line.chars() {
        if escape_next {
            escape_next = false;
            byte_index += ch.len_utf8();
            continue;
        }

        match ch {
            '\\' if in_string => {
                escape_next = true;
                byte_index += ch.len_utf8();
            }
            '"' => {
                in_string = !in_string;
                byte_index += ch.len_utf8();
            }
            '#' if !in_string => {
                // Found comment start outside of string
                return &line[..byte_index];
            }
            _ => {
                byte_index += ch.len_utf8();
            }
        }
    }

    // No comment found
    line
}

/// Check if a Turtle statement is complete (ends with '.').
///
/// This function handles:
/// - Directive completeness (`@prefix`, `@base` must end with `.`)
/// - Quoted triple nesting depth tracking (`<< ... >>`)
/// - Annotation block depth tracking (`{| ... |}`)
/// - String literal boundaries
/// - Escape sequences within strings
///
/// # Arguments
///
/// * `statement` - The statement to check for completeness
///
/// # Returns
///
/// `true` if the statement is complete, `false` otherwise
pub fn is_complete_turtle_statement(statement: &str) -> bool {
    let trimmed = statement.trim();

    // Directives are complete when they end with '.'
    if trimmed.starts_with("@prefix") || trimmed.starts_with("@base") {
        return trimmed.ends_with('.');
    }

    // Check if statement ends with '.' and is balanced
    let mut in_string = false;
    let mut escape_next = false;
    let mut quoted_triple_depth: i32 = 0;
    let mut annotation_depth: i32 = 0;
    let mut chars = statement.chars().peekable();

    while let Some(ch) = chars.next() {
        if escape_next {
            escape_next = false;
            continue;
        }

        match ch {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '<' if !in_string => {
                // Check for quoted triple start
                if chars.peek() == Some(&'<') {
                    chars.next(); // consume second '<'
                    quoted_triple_depth += 1;
                }
            }
            '>' if !in_string && quoted_triple_depth > 0 => {
                // Check for quoted triple end
                if chars.peek() == Some(&'>') {
                    chars.next(); // consume second '>'
                    quoted_triple_depth = quoted_triple_depth.saturating_sub(1);
                }
            }
            '{' if !in_string && quoted_triple_depth == 0 => {
                // Check for annotation block start {|
                if chars.peek() == Some(&'|') {
                    chars.next(); // consume '|'
                    annotation_depth += 1;
                }
            }
            '|' if !in_string && quoted_triple_depth == 0 && annotation_depth > 0 => {
                // Check for annotation block end |}
                if chars.peek() == Some(&'}') {
                    chars.next(); // consume '}'
                    annotation_depth = annotation_depth.saturating_sub(1);
                }
            }
            '.' if !in_string && quoted_triple_depth == 0 && annotation_depth == 0 => {
                // Check if this is a period in a number (e.g., 0.9)
                // If the next character is a digit, this is part of a number, not a statement terminator
                if chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                    // This is part of a numeric literal, not a statement terminator
                    continue;
                }
                // Statement ends with a dot and we're not in any nested structure
                return true;
            }
            _ => {}
        }
    }

    false
}

/// Check if a TriG statement is complete (enhanced version).
///
/// This function handles:
/// - Directive completeness (`@prefix`, `@base`)
/// - Graph block opening/closing (`{ ... }`)
/// - Brace depth tracking for nested structures
/// - Quoted triple nesting (`<< ... >>`)
/// - String literal boundaries
/// - State transitions for graph block parsing
///
/// # Arguments
///
/// * `statement` - The statement to check for completeness
/// * `state` - Mutable reference to the TriG parser state
///
/// # Returns
///
/// `true` if the statement is complete, `false` otherwise
pub fn is_complete_trig_statement(statement: &str, state: &mut TrigParserState) -> bool {
    let trimmed = statement.trim();

    // Handle directives (always complete on one line)
    if trimmed.starts_with("@prefix") || trimmed.starts_with("@base") {
        return trimmed.ends_with('.');
    }

    // Handle graph block closing
    if trimmed == "}" {
        return true;
    }

    let mut brace_count: i32 = 0;
    let mut in_string = false;
    let mut escape_next = false;
    let mut quoted_triple_depth: i32 = 0;
    let mut chars = statement.chars().peekable();

    while let Some(ch) = chars.next() {
        if escape_next {
            escape_next = false;
            continue;
        }

        match ch {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '<' if !in_string => {
                // Check for quoted triple start
                if chars.peek() == Some(&'<') {
                    chars.next(); // consume second '<'
                    quoted_triple_depth += 1;
                }
            }
            '>' if !in_string && quoted_triple_depth > 0 => {
                // Check for quoted triple end
                if chars.peek() == Some(&'>') {
                    chars.next(); // consume second '>'
                    quoted_triple_depth = quoted_triple_depth.saturating_sub(1);
                }
            }
            '{' if !in_string && quoted_triple_depth == 0 => {
                brace_count += 1;
                if !state.in_graph_block {
                    state.parsing_graph_name = false;
                }
            }
            '}' if !in_string && quoted_triple_depth == 0 => {
                brace_count = brace_count.saturating_sub(1);
            }
            '.' if !in_string && quoted_triple_depth == 0 && brace_count == 0 => {
                // Check if this is a period in a number (e.g., 0.9)
                // If the next character is a digit, this is part of a number, not a statement terminator
                if chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                    // This is part of a numeric literal, not a statement terminator
                    continue;
                }
                // Statement ends with a dot and we're not in any nested structure
                return true;
            }
            _ => {}
        }
    }

    // Special handling for graph declarations
    if brace_count > 0 && !state.in_graph_block {
        // We have an opening brace - this is a complete graph declaration start
        return true;
    }

    // Check for complete graph block
    if brace_count == 0 && state.in_graph_block && trimmed.ends_with('}') {
        return true;
    }

    false
}

/// Tokenize a triple pattern into its constituent terms.
///
/// This function splits a triple pattern (subject predicate object) into
/// individual tokens while respecting:
/// - Quoted triple boundaries (`<< ... >>`)
/// - String literal boundaries (`"..."`)
/// - Escape sequences within strings
/// - Whitespace as token separators (only at depth 0)
///
/// # Arguments
///
/// * `pattern` - The triple pattern string to tokenize
///
/// # Returns
///
/// A vector of token strings representing the terms
///
/// # Examples
///
/// ```ignore
/// let tokens = tokenize_triple("ex:alice ex:knows ex:bob")?;
/// assert_eq!(tokens.len(), 3);
/// ```
pub fn tokenize_triple(pattern: &str) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current_token = String::new();
    let mut chars = pattern.chars().peekable();
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;

    while let Some(ch) = chars.next() {
        if escape_next {
            current_token.push(ch);
            escape_next = false;
            continue;
        }

        match ch {
            '\\' if in_string => {
                escape_next = true;
                current_token.push(ch);
            }
            '"' => {
                in_string = !in_string;
                current_token.push(ch);
            }
            '<' if !in_string && chars.peek() == Some(&'<') => {
                // Start of quoted triple
                chars.next(); // consume second '<'
                depth += 1;
                current_token.push_str("<<");
            }
            '>' if !in_string && chars.peek() == Some(&'>') => {
                // End of quoted triple
                chars.next(); // consume second '>'
                depth -= 1;
                current_token.push_str(">>");
            }
            ' ' | '\t' if !in_string && depth == 0 => {
                // Whitespace at top level - end of token
                if !current_token.trim().is_empty() {
                    tokens.push(current_token.trim().to_string());
                    current_token.clear();
                }
            }
            _ => {
                current_token.push(ch);
            }
        }
    }

    // Add final token
    if !current_token.trim().is_empty() {
        tokens.push(current_token.trim().to_string());
    }

    Ok(tokens)
}

/// Tokenize a quad pattern (similar to triple but allows 4 terms).
///
/// This function splits a quad pattern (subject predicate object graph) into
/// individual tokens while respecting the same boundaries as `tokenize_triple`.
///
/// # Arguments
///
/// * `pattern` - The quad pattern string to tokenize
///
/// # Returns
///
/// A vector of token strings representing the terms
///
/// # Examples
///
/// ```ignore
/// let tokens = tokenize_quad("ex:alice ex:knows ex:bob ex:socialGraph")?;
/// assert_eq!(tokens.len(), 4);
/// ```
pub fn tokenize_quad(pattern: &str) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current_token = String::new();
    let mut chars = pattern.chars().peekable();
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;

    while let Some(ch) = chars.next() {
        if escape_next {
            current_token.push(ch);
            escape_next = false;
            continue;
        }

        match ch {
            '\\' if in_string => {
                escape_next = true;
                current_token.push(ch);
            }
            '"' => {
                in_string = !in_string;
                current_token.push(ch);
            }
            '<' if !in_string && chars.peek() == Some(&'<') => {
                // Start of quoted triple
                chars.next(); // consume second '<'
                depth += 1;
                current_token.push_str("<<");
            }
            '>' if !in_string && chars.peek() == Some(&'>') => {
                // End of quoted triple
                chars.next(); // consume second '>'
                depth -= 1;
                current_token.push_str(">>");
            }
            ' ' | '\t' if !in_string && depth == 0 => {
                // Whitespace at top level - end of token
                if !current_token.trim().is_empty() {
                    tokens.push(current_token.trim().to_string());
                    current_token.clear();
                }
            }
            _ => {
                current_token.push(ch);
            }
        }
    }

    // Add final token
    if !current_token.trim().is_empty() {
        tokens.push(current_token.trim().to_string());
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_inline_comment() {
        assert_eq!(strip_inline_comment("ex:foo ex:bar ."), "ex:foo ex:bar .");
        assert_eq!(
            strip_inline_comment("ex:foo ex:bar . # comment"),
            "ex:foo ex:bar . "
        );
        assert_eq!(
            strip_inline_comment("ex:foo ex:bar \"value # not a comment\" ."),
            "ex:foo ex:bar \"value # not a comment\" ."
        );
        assert_eq!(
            strip_inline_comment("ex:foo ex:bar \"value\\\" # still in string\" . # comment"),
            "ex:foo ex:bar \"value\\\" # still in string\" . "
        );
    }

    #[test]
    fn test_is_complete_turtle_statement() {
        // Complete statements
        assert!(is_complete_turtle_statement("ex:alice ex:knows ex:bob ."));
        assert!(is_complete_turtle_statement(
            "@prefix ex: <http://example.org/> ."
        ));

        // Incomplete statements
        assert!(!is_complete_turtle_statement("ex:alice ex:knows ex:bob"));
        assert!(!is_complete_turtle_statement(
            "@prefix ex: <http://example.org/>"
        ));

        // Quoted triples
        assert!(is_complete_turtle_statement(
            "<< ex:alice ex:knows ex:bob >> ex:certainty 0.9 ."
        ));
        assert!(!is_complete_turtle_statement(
            "<< ex:alice ex:knows ex:bob >> ex:certainty 0.9"
        ));

        // Annotation blocks
        assert!(is_complete_turtle_statement(
            "ex:alice ex:knows ex:bob {| ex:since 2020 |} ."
        ));
        assert!(!is_complete_turtle_statement(
            "ex:alice ex:knows ex:bob {| ex:since 2020 |}"
        ));
    }

    #[test]
    fn test_tokenize_triple() {
        // Simple triple
        let tokens = tokenize_triple("ex:alice ex:knows ex:bob").unwrap();
        assert_eq!(tokens, vec!["ex:alice", "ex:knows", "ex:bob"]);

        // Triple with quoted triple
        let tokens = tokenize_triple("<< ex:alice ex:knows ex:bob >> ex:certainty 0.9").unwrap();
        assert_eq!(
            tokens,
            vec!["<< ex:alice ex:knows ex:bob >>", "ex:certainty", "0.9"]
        );

        // Triple with literal
        let tokens = tokenize_triple("ex:alice ex:name \"Alice Wonder\"").unwrap();
        assert_eq!(tokens, vec!["ex:alice", "ex:name", "\"Alice Wonder\""]);
    }

    #[test]
    fn test_tokenize_quad() {
        // Simple quad
        let tokens = tokenize_quad("ex:alice ex:knows ex:bob ex:graph1").unwrap();
        assert_eq!(tokens, vec!["ex:alice", "ex:knows", "ex:bob", "ex:graph1"]);

        // Quad with quoted triple
        let tokens =
            tokenize_quad("<< ex:alice ex:knows ex:bob >> ex:certainty 0.9 ex:graph1").unwrap();
        assert_eq!(
            tokens,
            vec![
                "<< ex:alice ex:knows ex:bob >>",
                "ex:certainty",
                "0.9",
                "ex:graph1"
            ]
        );
    }

    #[test]
    fn test_is_complete_trig_statement() {
        let mut state = TrigParserState::new();

        // Complete statements
        assert!(is_complete_trig_statement(
            "ex:alice ex:knows ex:bob .",
            &mut state
        ));
        assert!(is_complete_trig_statement(
            "@prefix ex: <http://example.org/> .",
            &mut state
        ));

        // Graph block opening
        state = TrigParserState::new();
        assert!(is_complete_trig_statement("ex:graph1 {", &mut state));

        // Graph block closing
        state = TrigParserState::new();
        state.in_graph_block = true;
        assert!(is_complete_trig_statement("}", &mut state));

        // Incomplete statements
        state = TrigParserState::new();
        assert!(!is_complete_trig_statement(
            "ex:alice ex:knows ex:bob",
            &mut state
        ));
    }
}
