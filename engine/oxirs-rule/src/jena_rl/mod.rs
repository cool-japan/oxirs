//! Jena Rule Language (JRL) parser for `.rules` files.
//!
//! Parses Apache Jena's `.rules` syntax and lowers it to the native
//! `oxirs-rule` [`Rule`] type, enabling migration from Jena-based projects.
//!
//! # Jena Rule Language format
//!
//! ```text
//! # Comments start with #
//! @prefix ex: <http://example.org/> .
//!
//! # Forward rule: conditions -> consequences
//! [rule1: (?a ex:p ?b) (?b ex:q ?c) -> (?a ex:r ?c)]
//!
//! # Unnamed rule (name synthesised as jrl_rule_0, jrl_rule_1, …)
//! [(?x rdf:type ex:Person) -> (?x rdf:type ex:Human)]
//!
//! # Backward rule: head <- body
//! [anc: (?x ex:ancestor ?z) <- (?x ex:parent ?z)]
//! ```
//!
//! Variables are `?name`.  IRIs are either `<http://...>` or `prefix:local`.
//! Well-known prefixes (`rdf`, `rdfs`, `xsd`, `owl`) are pre-populated and
//! do not need explicit `@prefix` declarations.
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::jena_rl;
//!
//! let src = r#"
//! @prefix ex: <http://example.org/> .
//! [parent2ancestor: (?x ex:parent ?y) -> (?x ex:ancestor ?y)]
//! "#;
//!
//! let rules = jena_rl::parse_and_lower(src).expect("should parse");
//! assert_eq!(rules.len(), 1);
//! assert_eq!(rules[0].name, "parent2ancestor");
//! ```

pub mod lexer;
pub mod lowering;
pub mod parser;

pub use lowering::{lower_rule_set, LoweringError};
pub use parser::{JrlParseError, JrlRuleSet};

use crate::Rule;

/// Parse a Jena Rule Language string into an intermediate `JrlRuleSet`.
///
/// The returned rule set preserves all JRL-level details (backward flag,
/// prefix map, etc.) and can be lowered with [`lower_rule_set`].
pub fn parse_jrl(input: &str) -> Result<JrlRuleSet, JrlParseError> {
    let tokens = lexer::Lexer::tokenize(input).map_err(|e| JrlParseError {
        message: format!("lex error: {}", e),
        token_index: 0,
    })?;
    parser::parse(&tokens)
}

/// Parse a JRL string and lower it directly to oxirs-rule [`Rule`] objects.
///
/// This is the primary entry point for most callers.
pub fn parse_and_lower(input: &str) -> Result<Vec<Rule>, LoweringError> {
    let rule_set = parse_jrl(input)?;
    lower_rule_set(&rule_set)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_jrl_empty_returns_empty() {
        let rs = parse_jrl("").expect("empty input should succeed");
        assert!(rs.rules.is_empty());
    }

    #[test]
    fn test_parse_and_lower_roundtrip() {
        // Use full IRIs or rdf: (default prefix) to avoid ex: resolution failures.
        let src = "[r: (?x rdf:type rdfs:Class) -> (?x rdf:type owl:Class)]";
        let rules = parse_and_lower(src).expect("should succeed");
        assert_eq!(rules.len(), 1);
    }

    #[test]
    fn test_parse_jrl_retains_prefix_map() {
        let src = "@prefix foo: <http://foo.org/> .\n[r: (?x foo:bar ?y) -> (?x foo:baz ?y)]";
        let rs = parse_jrl(src).expect("should parse");
        assert!(rs.prefixes.contains_key("foo"));
    }

    #[test]
    fn test_parse_and_lower_with_prefixes() {
        let src = "@prefix ex: <http://example.org/> .\n[rule1: (?a ex:p ?b) -> (?a ex:q ?b)]";
        let rules = parse_and_lower(src).expect("should succeed");
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].name, "rule1");
    }
}
