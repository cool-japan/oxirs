// Lowering: JrlRuleSet → Vec<oxirs_rule::Rule>
//
// Mapping strategy:
//   JrlTerm::Variable(s)    → Term::Variable(s)
//   JrlTerm::Iri(s)         → Term::Constant(fully-expanded IRI string)
//   JrlTerm::Literal(l)     → Term::Literal(lexical form)
//
//   JrlAtom::Triple         → RuleAtom::Triple
//   JrlAtom::Builtin "notEqual"      → RuleAtom::NotEqual
//   JrlAtom::Builtin "lessThan"      → RuleAtom::LessThan
//   JrlAtom::Builtin "greaterThan"   → RuleAtom::GreaterThan
//   JrlAtom::Builtin (all others)    → RuleAtom::Builtin { name, args }
//
// IRIs in prefix:local form use the sentinel "prefix:<p>:<l>" from the parser
// and are expanded here using the prefix map.

use std::collections::HashMap;
use std::fmt;

use super::parser::{JrlAtom, JrlLiteral, JrlRule, JrlRuleSet, JrlTerm};
use crate::{Rule, RuleAtom, Term};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum LoweringError {
    ParseError(super::parser::JrlParseError),
    UnresolvablePrefix {
        prefix: String,
    },
    UnsupportedAtom {
        atom: String,
    },
    ArityError {
        builtin: String,
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoweringError::ParseError(e) => write!(f, "parse error: {}", e),
            LoweringError::UnresolvablePrefix { prefix } => {
                write!(f, "unresolvable prefix: `{}`", prefix)
            }
            LoweringError::UnsupportedAtom { atom } => {
                write!(f, "unsupported atom: {}", atom)
            }
            LoweringError::ArityError {
                builtin,
                expected,
                got,
            } => {
                write!(
                    f,
                    "builtin `{}` expects {} args, got {}",
                    builtin, expected, got
                )
            }
        }
    }
}

impl std::error::Error for LoweringError {}

impl From<super::parser::JrlParseError> for LoweringError {
    fn from(e: super::parser::JrlParseError) -> Self {
        LoweringError::ParseError(e)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Lower a complete `JrlRuleSet` into the native oxirs-rule `Rule` vector.
pub fn lower_rule_set(rule_set: &JrlRuleSet) -> Result<Vec<Rule>, LoweringError> {
    let mut rules = Vec::with_capacity(rule_set.rules.len());
    for jrl_rule in &rule_set.rules {
        rules.push(lower_rule(jrl_rule, &rule_set.prefixes)?);
    }
    Ok(rules)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn lower_rule(rule: &JrlRule, prefixes: &HashMap<String, String>) -> Result<Rule, LoweringError> {
    let name = rule
        .name
        .clone()
        .unwrap_or_else(|| "jrl_rule_unnamed".to_string());

    let body = lower_atoms(&rule.conditions, prefixes)?;
    let head = lower_atoms(&rule.consequences, prefixes)?;

    Ok(Rule { name, body, head })
}

fn lower_atoms(
    atoms: &[JrlAtom],
    prefixes: &HashMap<String, String>,
) -> Result<Vec<RuleAtom>, LoweringError> {
    atoms.iter().map(|a| lower_atom(a, prefixes)).collect()
}

fn lower_atom(
    atom: &JrlAtom,
    prefixes: &HashMap<String, String>,
) -> Result<RuleAtom, LoweringError> {
    match atom {
        JrlAtom::Triple {
            subject,
            predicate,
            object,
        } => Ok(RuleAtom::Triple {
            subject: lower_term(subject, prefixes)?,
            predicate: lower_term(predicate, prefixes)?,
            object: lower_term(object, prefixes)?,
        }),

        JrlAtom::Builtin { name, args } => lower_builtin(name, args, prefixes),
    }
}

/// Map known builtins to typed `RuleAtom` variants; everything else → `Builtin`.
fn lower_builtin(
    name: &str,
    args: &[JrlTerm],
    prefixes: &HashMap<String, String>,
) -> Result<RuleAtom, LoweringError> {
    match name {
        "notEqual" => {
            ensure_arity(name, 2, args.len())?;
            Ok(RuleAtom::NotEqual {
                left: lower_term(&args[0], prefixes)?,
                right: lower_term(&args[1], prefixes)?,
            })
        }
        "lessThan" => {
            ensure_arity(name, 2, args.len())?;
            Ok(RuleAtom::LessThan {
                left: lower_term(&args[0], prefixes)?,
                right: lower_term(&args[1], prefixes)?,
            })
        }
        "greaterThan" => {
            ensure_arity(name, 2, args.len())?;
            Ok(RuleAtom::GreaterThan {
                left: lower_term(&args[0], prefixes)?,
                right: lower_term(&args[1], prefixes)?,
            })
        }
        _ => {
            // Generic builtin (equal, sum, product, print, …)
            let lowered_args: Result<Vec<Term>, LoweringError> =
                args.iter().map(|t| lower_term(t, prefixes)).collect();
            Ok(RuleAtom::Builtin {
                name: name.to_string(),
                args: lowered_args?,
            })
        }
    }
}

fn ensure_arity(name: &str, expected: usize, got: usize) -> Result<(), LoweringError> {
    if got != expected {
        Err(LoweringError::ArityError {
            builtin: name.to_string(),
            expected,
            got,
        })
    } else {
        Ok(())
    }
}

fn lower_term(term: &JrlTerm, prefixes: &HashMap<String, String>) -> Result<Term, LoweringError> {
    match term {
        JrlTerm::Variable(name) => Ok(Term::Variable(name.clone())),
        JrlTerm::Iri(iri) => {
            let expanded = expand_iri_sentinel(iri, prefixes)?;
            Ok(Term::Constant(expanded))
        }
        JrlTerm::Literal(lit) => Ok(Term::Literal(literal_to_string(lit))),
    }
}

/// Expand the IRI sentinel `prefix:<prefix>:<local>` or return plain IRI unchanged.
fn expand_iri_sentinel(
    iri: &str,
    prefixes: &HashMap<String, String>,
) -> Result<String, LoweringError> {
    if let Some(rest) = iri.strip_prefix("prefix:") {
        // Format is `prefix:<p>:<local>` where `<p>` may contain `:` in the local
        // Split on first `:` only.
        if let Some(colon_pos) = rest.find(':') {
            let prefix = &rest[..colon_pos];
            let local = &rest[colon_pos + 1..];
            return expand_prefix(prefix, local, prefixes);
        }
        // Malformed sentinel — return as-is with a log
        return Ok(iri.to_string());
    }
    Ok(iri.to_string())
}

/// Expand `prefix:local` using the prefix map.
pub fn expand_prefix(
    prefix: &str,
    local: &str,
    prefixes: &HashMap<String, String>,
) -> Result<String, LoweringError> {
    match prefixes.get(prefix) {
        Some(ns) => Ok(format!("{}{}", ns, local)),
        None => Err(LoweringError::UnresolvablePrefix {
            prefix: prefix.to_string(),
        }),
    }
}

fn literal_to_string(lit: &JrlLiteral) -> String {
    match lit {
        JrlLiteral::String(s) => s.clone(),
        JrlLiteral::Integer(n) => n.to_string(),
        JrlLiteral::Float(f) => f.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jena_rl::lexer::Lexer;
    use crate::jena_rl::parser::parse;

    const EX_PREFIX: &str = "@prefix ex: <http://example.org/> .\n";

    fn parse_and_lower_str(input: &str) -> Vec<Rule> {
        // Prepend the `ex:` prefix declaration so all tests can use `ex:` freely.
        let src = format!("{}{}", EX_PREFIX, input);
        let toks = Lexer::tokenize(&src).expect("lex ok");
        let rs = parse(&toks).expect("parse ok");
        lower_rule_set(&rs).expect("lower ok")
    }

    #[test]
    fn test_lower_simple_triple_rule() {
        let rules = parse_and_lower_str("[r: (?a rdf:type ex:Person) -> (?a rdf:type ex:Human)]");
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].name, "r");
        assert_eq!(rules[0].body.len(), 1);
        assert_eq!(rules[0].head.len(), 1);
        // Subject should be a Variable
        match &rules[0].body[0] {
            RuleAtom::Triple { subject, .. } => {
                assert!(matches!(subject, Term::Variable(_)), "expected Variable");
            }
            _ => panic!("expected Triple atom"),
        }
    }

    #[test]
    fn test_lower_prefix_expansion() {
        let input = "@prefix ex: <http://example.org/> .\n[r: (?x ex:p ?y) -> (?x ex:q ?y)]";
        let toks = Lexer::tokenize(input).expect("lex ok");
        let rs = parse(&toks).expect("parse ok");
        let rules = lower_rule_set(&rs).expect("lower ok");
        match &rules[0].body[0] {
            RuleAtom::Triple { predicate, .. } => {
                assert_eq!(
                    predicate,
                    &Term::Constant("http://example.org/p".to_string())
                );
            }
            _ => panic!("expected Triple"),
        }
    }

    #[test]
    fn test_lower_not_equal_builtin() {
        let rules =
            parse_and_lower_str("[r: (?x ex:age ?a) (notEqual ?a 0) -> (?x ex:alive true)]");
        let body = &rules[0].body;
        let has_not_equal = body.iter().any(|a| matches!(a, RuleAtom::NotEqual { .. }));
        assert!(has_not_equal);
    }

    #[test]
    fn test_lower_less_than_builtin() {
        let rules =
            parse_and_lower_str("[r: (?x ex:score ?s) (lessThan ?s 50) -> (?x ex:failing true)]");
        let has_lt = rules[0]
            .body
            .iter()
            .any(|a| matches!(a, RuleAtom::LessThan { .. }));
        assert!(has_lt);
    }

    #[test]
    fn test_lower_greater_than_builtin() {
        let rules = parse_and_lower_str(
            "[r: (?x ex:score ?s) (greaterThan ?s 90) -> (?x ex:distinction true)]",
        );
        let has_gt = rules[0]
            .body
            .iter()
            .any(|a| matches!(a, RuleAtom::GreaterThan { .. }));
        assert!(has_gt);
    }

    #[test]
    fn test_lower_generic_builtin_stays_builtin() {
        let rules = parse_and_lower_str(
            "[r: (?x ex:v ?a) (?x ex:v ?b) (sum ?a ?b ?c) -> (?x ex:total ?c)]",
        );
        let has_sum = rules[0]
            .body
            .iter()
            .any(|a| matches!(a, RuleAtom::Builtin { name, .. } if name == "sum"));
        assert!(has_sum);
    }

    #[test]
    fn test_lower_literal_becomes_term_literal() {
        let rules = parse_and_lower_str(r#"[r: (?x ex:name "Alice") -> (?x rdf:type ex:Person)]"#);
        match &rules[0].body[0] {
            RuleAtom::Triple { object, .. } => {
                assert_eq!(*object, Term::Literal("Alice".to_string()));
            }
            _ => panic!("expected Triple"),
        }
    }

    #[test]
    fn test_lower_rdf_type_uses_default_prefix() {
        // `rdf:type` should expand to the standard RDF IRI without explicit @prefix
        let rules = parse_and_lower_str("[r: (?x rdf:type ex:Foo) -> (?x rdf:type ex:Bar)]");
        match &rules[0].body[0] {
            RuleAtom::Triple { predicate, .. } => {
                assert_eq!(
                    predicate,
                    &Term::Constant("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
                );
            }
            _ => panic!("expected Triple"),
        }
    }

    #[test]
    fn test_lower_integer_literal() {
        let rules = parse_and_lower_str("[r: (?x ex:count 5) -> (?x rdf:type ex:Five)]");
        match &rules[0].body[0] {
            RuleAtom::Triple { object, .. } => {
                assert_eq!(*object, Term::Literal("5".to_string()));
            }
            _ => panic!("expected Triple"),
        }
    }

    #[test]
    fn test_lower_multiple_rules() {
        let input = "[r1: (?a ex:p ?b) -> (?a ex:q ?b)]\n[r2: (?x ex:q ?y) -> (?x ex:r ?y)]";
        let rules = parse_and_lower_str(input);
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].name, "r1");
        assert_eq!(rules[1].name, "r2");
    }
}
