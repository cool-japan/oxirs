//! # N3 Logic Parser
//!
//! Parses Notation3 rule documents into `N3Rule` structures.
//! Handles `@prefix`, `@forAll`, `@forSome`, and `{ ... } => { ... }` rules.

use anyhow::{anyhow, Result};

use super::n3logic_types::{N3BuiltIn, N3Formula, N3Rule, N3Term};

pub struct N3Parser;

impl N3Parser {
    pub fn parse(input: &str) -> Result<Vec<N3Rule>> {
        let mut rules = Vec::new();
        let mut universals: Vec<String> = Vec::new();
        let mut existentials: Vec<String> = Vec::new();
        let text = input.trim();
        let mut pos = 0;

        while pos < text.len() {
            pos = Self::skip_ws(text, pos);
            if pos >= text.len() {
                break;
            }

            if text[pos..].starts_with("@prefix") {
                pos = Self::skip_to_dot(text, pos);
                continue;
            }
            if text[pos..].starts_with("@forAll") {
                let (vars, np) = Self::parse_quantifier(text, pos + 7)?;
                universals.extend(vars);
                pos = np;
                continue;
            }
            if text[pos..].starts_with("@forSome") {
                let (vars, np) = Self::parse_quantifier(text, pos + 8)?;
                existentials.extend(vars);
                pos = np;
                continue;
            }
            if text[pos..].starts_with('{') {
                let (rule, np) = Self::parse_rule_at(text, pos)?;
                rules.push(
                    rule.with_universals(universals.clone())
                        .with_existentials(existentials.clone()),
                );
                pos = Self::skip_ws(text, np);
                if pos < text.len() && text.as_bytes()[pos] == b'.' {
                    pos += 1;
                }
                continue;
            }
            pos = Self::skip_to_dot(text, pos);
        }
        Ok(rules)
    }

    pub fn parse_rule(input: &str) -> Result<N3Rule> {
        let (rule, _) = Self::parse_rule_at(input.trim(), 0)?;
        Ok(rule)
    }

    fn parse_rule_at(text: &str, start: usize) -> Result<(N3Rule, usize)> {
        let mut pos = start;
        if pos >= text.len() || text.as_bytes()[pos] != b'{' {
            return Err(anyhow!("Expected '{{' at position {}", pos));
        }
        let (ant, np) = Self::parse_graph(text, pos)?;
        pos = Self::skip_ws(text, np);
        if !text[pos..].starts_with("=>") {
            return Err(anyhow!("Expected '=>' at position {}", pos));
        }
        pos += 2;
        pos = Self::skip_ws(text, pos);
        if pos >= text.len() || text.as_bytes()[pos] != b'{' {
            return Err(anyhow!("Expected '{{' after '=>' at position {}", pos));
        }
        let (cons, np2) = Self::parse_graph(text, pos)?;
        Ok((N3Rule::new(ant, cons), np2))
    }

    pub(super) fn parse_graph(text: &str, start: usize) -> Result<(Vec<N3Formula>, usize)> {
        let mut pos = start + 1;
        let mut formulas = Vec::new();
        loop {
            pos = Self::skip_ws(text, pos);
            if pos >= text.len() {
                return Err(anyhow!("Unterminated '{{'"));
            }
            if text.as_bytes()[pos] == b'}' {
                pos += 1;
                break;
            }
            if text.as_bytes()[pos] == b'.' {
                pos += 1;
                continue;
            }
            let (formula, np) = Self::parse_formula(text, pos)?;
            formulas.push(formula);
            pos = Self::skip_ws(text, np);
            if pos < text.len() && text.as_bytes()[pos] == b'.' {
                pos += 1;
            }
        }
        Ok((formulas, pos))
    }

    fn parse_formula(text: &str, start: usize) -> Result<(N3Formula, usize)> {
        if text[start..].starts_with('{') {
            let (sub, np) = Self::parse_graph(text, start)?;
            return Ok((N3Formula::Graph(sub), np));
        }
        let (subject, p2) = Self::parse_term(text, start)?;
        let (predicate, p3) = Self::parse_term(text, Self::skip_ws(text, p2))?;
        let (object, p4) = Self::parse_term(text, Self::skip_ws(text, p3))?;
        let formula = Self::maybe_builtin(&predicate, subject, object);
        Ok((formula, p4))
    }

    fn maybe_builtin(predicate: &N3Term, subject: N3Term, object: N3Term) -> N3Formula {
        let pred = match predicate {
            N3Term::Iri(s) => s.as_str(),
            _ => {
                return N3Formula::Triple {
                    subject,
                    predicate: predicate.clone(),
                    object,
                }
            }
        };
        match pred {
            "math:greaterThan" | "http://www.w3.org/2000/10/swap/math#greaterThan" => {
                N3Formula::BuiltIn(N3BuiltIn::MathGreaterThan {
                    left: subject,
                    right: object,
                })
            }
            "math:lessThan" | "http://www.w3.org/2000/10/swap/math#lessThan" => {
                N3Formula::BuiltIn(N3BuiltIn::MathLessThan {
                    left: subject,
                    right: object,
                })
            }
            "math:equalTo" | "http://www.w3.org/2000/10/swap/math#equalTo" => {
                N3Formula::BuiltIn(N3BuiltIn::MathEqualTo {
                    left: subject,
                    right: object,
                })
            }
            "math:sum" | "http://www.w3.org/2000/10/swap/math#sum" => {
                N3Formula::BuiltIn(N3BuiltIn::MathSum {
                    args: vec![subject],
                    result: object,
                })
            }
            "math:difference" | "http://www.w3.org/2000/10/swap/math#difference" => {
                N3Formula::BuiltIn(N3BuiltIn::MathDifference {
                    args: vec![subject],
                    result: object,
                })
            }
            "math:product" | "http://www.w3.org/2000/10/swap/math#product" => {
                N3Formula::BuiltIn(N3BuiltIn::MathProduct {
                    args: vec![subject],
                    result: object,
                })
            }
            "math:quotient" | "http://www.w3.org/2000/10/swap/math#quotient" => {
                N3Formula::BuiltIn(N3BuiltIn::MathQuotient {
                    args: vec![subject],
                    result: object,
                })
            }
            "string:concatenation" | "http://www.w3.org/2000/10/swap/string#concatenation" => {
                N3Formula::BuiltIn(N3BuiltIn::StringConcatenation {
                    args: vec![subject],
                    result: object,
                })
            }
            "string:length" | "http://www.w3.org/2000/10/swap/string#length" => {
                N3Formula::BuiltIn(N3BuiltIn::StringLength {
                    input: subject,
                    result: object,
                })
            }
            "string:contains" | "http://www.w3.org/2000/10/swap/string#contains" => {
                N3Formula::BuiltIn(N3BuiltIn::StringContains {
                    subject,
                    substring: object,
                })
            }
            "log:equal" | "http://www.w3.org/2000/10/swap/log#equal" => {
                N3Formula::BuiltIn(N3BuiltIn::LogEqual {
                    left: subject,
                    right: object,
                })
            }
            "log:notEqual" | "http://www.w3.org/2000/10/swap/log#notEqual" => {
                N3Formula::BuiltIn(N3BuiltIn::LogNotEqual {
                    left: subject,
                    right: object,
                })
            }
            _ => N3Formula::Triple {
                subject,
                predicate: predicate.clone(),
                object,
            },
        }
    }

    pub(super) fn parse_term(text: &str, start: usize) -> Result<(N3Term, usize)> {
        if start >= text.len() {
            return Err(anyhow!("Unexpected end of input"));
        }
        let b = text.as_bytes()[start];

        if b == b'<' {
            let end = text[start + 1..]
                .find('>')
                .ok_or_else(|| anyhow!("Unterminated IRI"))?;
            return Ok((
                N3Term::Iri(text[start + 1..start + 1 + end].to_string()),
                start + 1 + end + 1,
            ));
        }

        if b == b'"' {
            let mut i = start + 1;
            let mut value = String::new();
            while i < text.len() {
                let c = text.as_bytes()[i];
                if c == b'\\' && i + 1 < text.len() {
                    i += 1;
                    value.push(text.as_bytes()[i] as char);
                    i += 1;
                } else if c == b'"' {
                    i += 1;
                    break;
                } else {
                    value.push(c as char);
                    i += 1;
                }
            }
            let mut lang = None;
            let mut datatype = None;
            if i < text.len() {
                if text[i..].starts_with("^^<") {
                    i += 3;
                    let de = text[i..]
                        .find('>')
                        .ok_or_else(|| anyhow!("Unterminated datatype IRI"))?;
                    datatype = Some(text[i..i + de].to_string());
                    i += de + 1;
                } else if i < text.len() && text.as_bytes()[i] == b'@' {
                    i += 1;
                    let ls = i;
                    while i < text.len()
                        && (text.as_bytes()[i].is_ascii_alphabetic() || text.as_bytes()[i] == b'-')
                    {
                        i += 1;
                    }
                    lang = Some(text[ls..i].to_string());
                }
            }
            return Ok((
                N3Term::Literal {
                    value,
                    datatype,
                    lang,
                },
                i,
            ));
        }

        if text[start..].starts_with("_:") {
            let s = start + 2;
            let mut i = s;
            while i < text.len()
                && (text.as_bytes()[i].is_ascii_alphanumeric() || text.as_bytes()[i] == b'_')
            {
                i += 1;
            }
            return Ok((N3Term::BlankNode(text[s..i].to_string()), i));
        }

        if b == b'?' {
            let s = start + 1;
            let mut i = s;
            while i < text.len()
                && (text.as_bytes()[i].is_ascii_alphanumeric() || text.as_bytes()[i] == b'_')
            {
                i += 1;
            }
            return Ok((N3Term::Variable(text[s..i].to_string()), i));
        }

        if b == b'{' {
            let (fmls, np) = Self::parse_graph(text, start)?;
            return Ok((N3Term::NestedFormula(Box::new(fmls)), np));
        }

        if b.is_ascii_digit()
            || (b == b'-' && start + 1 < text.len() && text.as_bytes()[start + 1].is_ascii_digit())
        {
            let mut i = start;
            if b == b'-' {
                i += 1;
            }
            while i < text.len()
                && (text.as_bytes()[i].is_ascii_digit() || text.as_bytes()[i] == b'.')
            {
                i += 1;
            }
            return Ok((
                N3Term::Literal {
                    value: text[start..i].to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".to_string()),
                    lang: None,
                },
                i,
            ));
        }

        if b.is_ascii_alphabetic() || b == b':' || b == b'_' {
            let mut i = start;
            while i < text.len() {
                let c = text.as_bytes()[i];
                if c.is_ascii_alphanumeric() || c == b'_' || c == b':' || c == b'-' || c == b'.' {
                    i += 1;
                } else {
                    break;
                }
            }
            while i > start && text.as_bytes()[i - 1] == b'.' {
                i -= 1;
            }
            let token = text[start..i].to_string();
            if token == "a" {
                return Ok((
                    N3Term::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
                    i,
                ));
            }
            return Ok((N3Term::Iri(token), i));
        }

        Err(anyhow!(
            "Cannot parse term at position {}: '{}'",
            start,
            &text[start..std::cmp::min(start + 20, text.len())]
        ))
    }

    fn parse_quantifier(text: &str, start: usize) -> Result<(Vec<String>, usize)> {
        let mut pos = Self::skip_ws(text, start);
        let mut vars = Vec::new();
        while pos < text.len() && text.as_bytes()[pos] != b'.' {
            pos = Self::skip_ws(text, pos);
            if pos >= text.len() || text.as_bytes()[pos] == b'.' {
                break;
            }
            if text.as_bytes()[pos] == b',' {
                pos += 1;
                continue;
            }
            let (term, np) = Self::parse_term(text, pos)?;
            if let Some(s) = term.value_str() {
                vars.push(s.to_string());
            }
            pos = np;
        }
        if pos < text.len() && text.as_bytes()[pos] == b'.' {
            pos += 1;
        }
        Ok((vars, pos))
    }

    pub(super) fn skip_ws(text: &str, mut pos: usize) -> usize {
        while pos < text.len() {
            let c = text.as_bytes()[pos];
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                pos += 1;
            } else if text[pos..].starts_with('#') {
                while pos < text.len() && text.as_bytes()[pos] != b'\n' {
                    pos += 1;
                }
            } else {
                break;
            }
        }
        pos
    }

    fn skip_to_dot(text: &str, mut pos: usize) -> usize {
        while pos < text.len() && text.as_bytes()[pos] != b'.' {
            pos += 1;
        }
        if pos < text.len() {
            pos += 1;
        }
        pos
    }
}
