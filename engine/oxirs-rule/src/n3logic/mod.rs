//! # Notation3 (N3) Logic Support
//!
//! N3 extends Turtle with logical rules using `@forAll`, `@forSome`, and `=>` (implies).
//! Used in reasoning systems like Notation3Py, EYE reasoner, etc.

use std::collections::HashMap;
use std::fmt;

use anyhow::{anyhow, Result};

// ── Public types ──────────────────────────────────────────────────────────────

/// A Notation3 rule: antecedent (LHS) implies consequent (RHS).
#[derive(Debug, Clone, PartialEq)]
pub struct N3Rule {
    pub antecedent: Vec<N3Formula>,
    pub consequent: Vec<N3Formula>,
    pub universals: Vec<String>,
    pub existentials: Vec<String>,
}

impl N3Rule {
    pub fn new(antecedent: Vec<N3Formula>, consequent: Vec<N3Formula>) -> Self {
        Self {
            antecedent,
            consequent,
            universals: Vec::new(),
            existentials: Vec::new(),
        }
    }

    pub fn with_universals(mut self, vars: Vec<String>) -> Self {
        self.universals = vars;
        self
    }

    pub fn with_existentials(mut self, vars: Vec<String>) -> Self {
        self.existentials = vars;
        self
    }
}

impl fmt::Display for N3Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "N3Rule({} antecedent(s) => {} consequent(s))",
            self.antecedent.len(),
            self.consequent.len()
        )
    }
}

/// An N3 formula: triple, nested graph, or built-in.
#[derive(Debug, Clone, PartialEq)]
pub enum N3Formula {
    Triple {
        subject: N3Term,
        predicate: N3Term,
        object: N3Term,
    },
    Graph(Vec<N3Formula>),
    BuiltIn(N3BuiltIn),
}

/// An N3 term.
#[derive(Debug, Clone, PartialEq)]
pub enum N3Term {
    Iri(String),
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    BlankNode(String),
    Variable(String),
    Universal(String),
    NestedFormula(Box<Vec<N3Formula>>),
}

impl N3Term {
    pub fn value_str(&self) -> Option<&str> {
        match self {
            N3Term::Iri(s) | N3Term::BlankNode(s) | N3Term::Variable(s) | N3Term::Universal(s) => {
                Some(s.as_str())
            }
            N3Term::Literal { value, .. } => Some(value.as_str()),
            N3Term::NestedFormula(_) => None,
        }
    }

    pub fn is_variable(&self) -> bool {
        matches!(self, N3Term::Variable(_) | N3Term::Universal(_))
    }
}

impl fmt::Display for N3Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            N3Term::Iri(s) => write!(f, "<{}>", s),
            N3Term::Literal {
                value,
                datatype,
                lang,
            } => {
                write!(f, "\"{}\"", value)?;
                if let Some(dt) = datatype {
                    write!(f, "^^<{}>", dt)?;
                }
                if let Some(l) = lang {
                    write!(f, "@{}", l)?;
                }
                Ok(())
            }
            N3Term::BlankNode(s) => write!(f, "_:{}", s),
            N3Term::Variable(s) => write!(f, "?{}", s),
            N3Term::Universal(s) => write!(f, "!{}", s),
            N3Term::NestedFormula(_) => write!(f, "{{ ... }}"),
        }
    }
}

/// N3 built-in operations.
#[derive(Debug, Clone, PartialEq)]
pub enum N3BuiltIn {
    MathSum {
        args: Vec<N3Term>,
        result: N3Term,
    },
    MathDifference {
        args: Vec<N3Term>,
        result: N3Term,
    },
    MathProduct {
        args: Vec<N3Term>,
        result: N3Term,
    },
    MathQuotient {
        args: Vec<N3Term>,
        result: N3Term,
    },
    MathGreaterThan {
        left: N3Term,
        right: N3Term,
    },
    MathLessThan {
        left: N3Term,
        right: N3Term,
    },
    MathEqualTo {
        left: N3Term,
        right: N3Term,
    },
    StringConcatenation {
        args: Vec<N3Term>,
        result: N3Term,
    },
    StringLength {
        input: N3Term,
        result: N3Term,
    },
    StringContains {
        subject: N3Term,
        substring: N3Term,
    },
    LogImplies {
        antecedent: Box<N3Formula>,
        consequent: Box<N3Formula>,
    },
    LogConcludes {
        graph: N3Term,
        formula: Box<N3Formula>,
    },
    LogEqual {
        left: N3Term,
        right: N3Term,
    },
    LogNotEqual {
        left: N3Term,
        right: N3Term,
    },
}

/// A ground RDF triple.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Triple {
    pub fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            subject: s.into(),
            predicate: p.into(),
            object: o.into(),
        }
    }
}

/// Variable binding map.
pub type Bindings = HashMap<String, N3Term>;

// ── Parser ─────────────────────────────────────────────────────────────────────

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

    fn parse_graph(text: &str, start: usize) -> Result<(Vec<N3Formula>, usize)> {
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

    fn parse_term(text: &str, start: usize) -> Result<(N3Term, usize)> {
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

    fn skip_ws(text: &str, mut pos: usize) -> usize {
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

// ── Engine ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct N3Engine {
    pub rules: Vec<N3Rule>,
    pub facts: Vec<Triple>,
}

impl N3Engine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_rule(&mut self, rule: N3Rule) {
        self.rules.push(rule);
    }

    pub fn assert_fact(&mut self, triple: Triple) {
        if !self.facts.contains(&triple) {
            self.facts.push(triple);
        }
    }

    pub fn run(&mut self) -> Result<Vec<Triple>> {
        self.run_bounded(usize::MAX)
    }

    pub fn run_bounded(&mut self, max_iter: usize) -> Result<Vec<Triple>> {
        for _ in 0..max_iter {
            let new_triples = self.derive_one_pass()?;
            if new_triples.is_empty() {
                break;
            }
            for t in new_triples {
                if !self.facts.contains(&t) {
                    self.facts.push(t);
                }
            }
        }
        Ok(self.facts.clone())
    }

    pub fn is_derivable(&self, triple: &Triple) -> bool {
        if self.facts.contains(triple) {
            return true;
        }
        for rule in &self.rules {
            if self.apply_rule(rule, &self.facts).contains(triple) {
                return true;
            }
        }
        false
    }

    fn derive_one_pass(&self) -> Result<Vec<Triple>> {
        let mut new_triples = Vec::new();
        for rule in &self.rules {
            for t in self.apply_rule(rule, &self.facts) {
                if !self.facts.contains(&t) && !new_triples.contains(&t) {
                    new_triples.push(t);
                }
            }
        }
        Ok(new_triples)
    }

    fn apply_rule(&self, rule: &N3Rule, facts: &[Triple]) -> Vec<Triple> {
        let mut results = Vec::new();
        let all_bindings = Self::match_all_formulas(&rule.antecedent, facts, &Bindings::new());
        for bindings in all_bindings {
            for formula in &rule.consequent {
                if let Some(t) = Self::instantiate_formula(formula, &bindings) {
                    results.push(t);
                }
            }
        }
        results
    }

    fn match_all_formulas(
        formulas: &[N3Formula],
        facts: &[Triple],
        current: &Bindings,
    ) -> Vec<Bindings> {
        if formulas.is_empty() {
            return vec![current.clone()];
        }
        let first = &formulas[0];
        let rest = &formulas[1..];
        let mut results = Vec::new();
        match first {
            N3Formula::Triple {
                subject,
                predicate,
                object,
            } => {
                for fact in facts {
                    let mut b = current.clone();
                    if Self::unify(subject, &fact.subject, &mut b)
                        && Self::unify(predicate, &fact.predicate, &mut b)
                        && Self::unify(object, &fact.object, &mut b)
                    {
                        results.extend(Self::match_all_formulas(rest, facts, &b));
                    }
                }
            }
            N3Formula::BuiltIn(bi) => {
                let mut b = current.clone();
                if Self::evaluate_builtin(bi, &mut b) {
                    results.extend(Self::match_all_formulas(rest, facts, &b));
                }
            }
            N3Formula::Graph(_) => {
                results.extend(Self::match_all_formulas(rest, facts, current));
            }
        }
        results
    }

    fn unify(pattern: &N3Term, ground: &str, bindings: &mut Bindings) -> bool {
        match pattern {
            N3Term::Variable(v) | N3Term::Universal(v) => {
                if let Some(existing) = bindings.get(v.as_str()) {
                    existing.value_str().map(|s| s == ground).unwrap_or(false)
                } else {
                    bindings.insert(v.clone(), N3Term::Iri(ground.to_string()));
                    true
                }
            }
            other => other.value_str().map(|s| s == ground).unwrap_or(false),
        }
    }

    pub fn evaluate_builtin(bi: &N3BuiltIn, bindings: &mut Bindings) -> bool {
        match bi {
            N3BuiltIn::MathSum { args, result } => {
                let vals: Option<Vec<f64>> = args
                    .iter()
                    .map(|a| Self::resolve_num(a, bindings))
                    .collect();
                vals.map(|vs| Self::bind_num(result, vs.iter().sum(), bindings))
                    .unwrap_or(false)
            }
            N3BuiltIn::MathDifference { args, result } => {
                if args.len() < 2 {
                    return false;
                }
                match (
                    Self::resolve_num(&args[0], bindings),
                    Self::resolve_num(&args[1], bindings),
                ) {
                    (Some(a), Some(b)) => Self::bind_num(result, a - b, bindings),
                    _ => false,
                }
            }
            N3BuiltIn::MathProduct { args, result } => {
                let vals: Option<Vec<f64>> = args
                    .iter()
                    .map(|a| Self::resolve_num(a, bindings))
                    .collect();
                vals.map(|vs| Self::bind_num(result, vs.iter().product(), bindings))
                    .unwrap_or(false)
            }
            N3BuiltIn::MathQuotient { args, result } => {
                if args.len() < 2 {
                    return false;
                }
                match (
                    Self::resolve_num(&args[0], bindings),
                    Self::resolve_num(&args[1], bindings),
                ) {
                    (Some(a), Some(b)) if b != 0.0 => Self::bind_num(result, a / b, bindings),
                    _ => false,
                }
            }
            N3BuiltIn::MathGreaterThan { left, right } => {
                match (
                    Self::resolve_num(left, bindings),
                    Self::resolve_num(right, bindings),
                ) {
                    (Some(l), Some(r)) => l > r,
                    _ => match (
                        Self::resolve_str(left, bindings),
                        Self::resolve_str(right, bindings),
                    ) {
                        (Some(l), Some(r)) => l > r,
                        _ => false,
                    },
                }
            }
            N3BuiltIn::MathLessThan { left, right } => {
                match (
                    Self::resolve_num(left, bindings),
                    Self::resolve_num(right, bindings),
                ) {
                    (Some(l), Some(r)) => l < r,
                    _ => match (
                        Self::resolve_str(left, bindings),
                        Self::resolve_str(right, bindings),
                    ) {
                        (Some(l), Some(r)) => l < r,
                        _ => false,
                    },
                }
            }
            N3BuiltIn::MathEqualTo { left, right } => {
                match (
                    Self::resolve_num(left, bindings),
                    Self::resolve_num(right, bindings),
                ) {
                    (Some(l), Some(r)) => (l - r).abs() < 1e-12,
                    _ => match (
                        Self::resolve_str(left, bindings),
                        Self::resolve_str(right, bindings),
                    ) {
                        (Some(l), Some(r)) => l == r,
                        _ => false,
                    },
                }
            }
            N3BuiltIn::StringConcatenation { args, result } => {
                let parts: Option<Vec<String>> = args
                    .iter()
                    .map(|a| Self::resolve_str(a, bindings))
                    .collect();
                parts
                    .map(|ps| Self::bind_str(result, &ps.concat(), bindings))
                    .unwrap_or(false)
            }
            N3BuiltIn::StringLength { input, result } => Self::resolve_str(input, bindings)
                .map(|s| Self::bind_num(result, s.len() as f64, bindings))
                .unwrap_or(false),
            N3BuiltIn::StringContains { subject, substring } => {
                match (
                    Self::resolve_str(subject, bindings),
                    Self::resolve_str(substring, bindings),
                ) {
                    (Some(s), Some(sub)) => s.contains(sub.as_str()),
                    _ => false,
                }
            }
            N3BuiltIn::LogEqual { left, right } => {
                match (
                    Self::resolve_str(left, bindings),
                    Self::resolve_str(right, bindings),
                ) {
                    (Some(l), Some(r)) => l == r,
                    _ => false,
                }
            }
            N3BuiltIn::LogNotEqual { left, right } => {
                match (
                    Self::resolve_str(left, bindings),
                    Self::resolve_str(right, bindings),
                ) {
                    (Some(l), Some(r)) => l != r,
                    _ => false,
                }
            }
            N3BuiltIn::LogImplies { .. } | N3BuiltIn::LogConcludes { .. } => true,
        }
    }

    fn resolve_num(term: &N3Term, bindings: &Bindings) -> Option<f64> {
        Self::resolve_str(term, bindings).and_then(|s| s.parse().ok())
    }

    fn resolve_str(term: &N3Term, bindings: &Bindings) -> Option<String> {
        match term {
            N3Term::Variable(v) | N3Term::Universal(v) => bindings
                .get(v.as_str())
                .and_then(|t| t.value_str().map(|s| s.to_string())),
            N3Term::Literal { value, .. } => Some(value.clone()),
            N3Term::Iri(s) | N3Term::BlankNode(s) => Some(s.clone()),
            N3Term::NestedFormula(_) => None,
        }
    }

    fn bind_num(term: &N3Term, value: f64, bindings: &mut Bindings) -> bool {
        let s = if value.fract() == 0.0 && value.abs() < 1e15 {
            format!("{}", value as i64)
        } else {
            format!("{}", value)
        };
        match term {
            N3Term::Variable(v) | N3Term::Universal(v) => {
                if let Some(existing) = bindings.get(v.as_str()) {
                    existing
                        .value_str()
                        .and_then(|es| es.parse::<f64>().ok())
                        .map(|ev| (ev - value).abs() < 1e-12)
                        .unwrap_or(false)
                } else {
                    bindings.insert(
                        v.clone(),
                        N3Term::Literal {
                            value: s,
                            datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".to_string()),
                            lang: None,
                        },
                    );
                    true
                }
            }
            N3Term::Literal { value: lv, .. } => lv
                .parse::<f64>()
                .map(|v2| (v2 - value).abs() < 1e-12)
                .unwrap_or(false),
            _ => false,
        }
    }

    fn bind_str(term: &N3Term, value: &str, bindings: &mut Bindings) -> bool {
        match term {
            N3Term::Variable(v) | N3Term::Universal(v) => {
                if let Some(existing) = bindings.get(v.as_str()) {
                    existing.value_str().map(|s| s == value).unwrap_or(false)
                } else {
                    bindings.insert(
                        v.clone(),
                        N3Term::Literal {
                            value: value.to_string(),
                            datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                            lang: None,
                        },
                    );
                    true
                }
            }
            N3Term::Literal { value: lv, .. } => lv == value,
            _ => false,
        }
    }

    fn instantiate_formula(formula: &N3Formula, bindings: &Bindings) -> Option<Triple> {
        if let N3Formula::Triple {
            subject,
            predicate,
            object,
        } = formula
        {
            let s = Self::inst_term(subject, bindings)?;
            let p = Self::inst_term(predicate, bindings)?;
            let o = Self::inst_term(object, bindings)?;
            Some(Triple::new(s, p, o))
        } else {
            None
        }
    }

    fn inst_term(term: &N3Term, bindings: &Bindings) -> Option<String> {
        match term {
            N3Term::Variable(v) | N3Term::Universal(v) => bindings
                .get(v.as_str())
                .and_then(|t| t.value_str().map(|s| s.to_string())),
            N3Term::Iri(s) | N3Term::BlankNode(s) => Some(s.clone()),
            N3Term::Literal { value, .. } => Some(value.clone()),
            N3Term::NestedFormula(_) => None,
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: make a simple triple-pattern formula
    fn tf(s: &str, p: &str, o: &str) -> N3Formula {
        N3Formula::Triple {
            subject: if let Some(stripped) = s.strip_prefix('?') {
                N3Term::Variable(stripped.to_string())
            } else {
                N3Term::Iri(s.to_string())
            },
            predicate: if let Some(stripped) = p.strip_prefix('?') {
                N3Term::Variable(stripped.to_string())
            } else {
                N3Term::Iri(p.to_string())
            },
            object: if let Some(stripped) = o.strip_prefix('?') {
                N3Term::Variable(stripped.to_string())
            } else {
                N3Term::Iri(o.to_string())
            },
        }
    }

    // ── Parser tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_parse_simple_rule() {
        let rules = N3Parser::parse("{ ?a :hasAge ?n } => { ?a a :Adult } .").unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].antecedent.len(), 1);
        assert_eq!(rules[0].consequent.len(), 1);
    }

    #[test]
    fn test_parse_rule_method() {
        let rule = N3Parser::parse_rule("{ ?a :hasAge ?n } => { ?a a :Adult }").unwrap();
        assert_eq!(rule.antecedent.len(), 1);
        assert_eq!(rule.consequent.len(), 1);
    }

    #[test]
    fn test_parse_rule_antecedent_has_variable_subject() {
        let rules = N3Parser::parse("{ ?x :knows ?y } => { ?x :acquaintance ?y } .").unwrap();
        match &rules[0].antecedent[0] {
            N3Formula::Triple { subject, .. } => {
                assert!(matches!(subject, N3Term::Variable(v) if v == "x"))
            }
            _ => panic!("expected triple"),
        }
    }

    #[test]
    fn test_parse_rule_antecedent_has_variable_object() {
        let rules = N3Parser::parse("{ ?x :knows ?y } => { ?x :acquaintance ?y } .").unwrap();
        match &rules[0].antecedent[0] {
            N3Formula::Triple { object, .. } => {
                assert!(matches!(object, N3Term::Variable(v) if v == "y"))
            }
            _ => panic!("expected triple"),
        }
    }

    #[test]
    fn test_parse_multiple_antecedents() {
        let n3 = "{ ?a :hasAge ?n . ?a a :Person } => { ?a :isAdult true } .";
        let rules = N3Parser::parse(n3).unwrap();
        assert_eq!(rules[0].antecedent.len(), 2);
    }

    #[test]
    fn test_parse_multiple_rules() {
        let n3 =
            "{ ?a :hasAge ?n } => { ?a a :Person } . { ?a a :Person } => { ?a :hasType :Human } .";
        let rules = N3Parser::parse(n3).unwrap();
        assert_eq!(rules.len(), 2);
    }

    #[test]
    fn test_parse_forall_declaration() {
        let n3 = "@forAll :x, :n . { :x :hasAge :n } => { :x a :Adult } .";
        let rules = N3Parser::parse(n3).unwrap();
        assert!(!rules[0].universals.is_empty());
    }

    #[test]
    fn test_parse_forsome_declaration() {
        let n3 = "@forSome :y . { ?x :knows :y } => { ?x :hasFriend :y } .";
        let rules = N3Parser::parse(n3).unwrap();
        assert!(!rules[0].existentials.is_empty());
    }

    #[test]
    fn test_parse_universal_count() {
        let n3 = "@forAll :x, :y . { :x :p :y } => { :y :q :x } .";
        let rules = N3Parser::parse(n3).unwrap();
        assert_eq!(rules[0].universals.len(), 2);
    }

    #[test]
    fn test_parse_iri_terms() {
        let n3 = "{ <http://a.org/x> <http://a.org/p> <http://a.org/y> } => { <http://a.org/x> <http://a.org/q> <http://a.org/y> } .";
        let rules = N3Parser::parse(n3).unwrap();
        match &rules[0].antecedent[0] {
            N3Formula::Triple { subject, .. } => {
                assert!(matches!(subject, N3Term::Iri(s) if s == "http://a.org/x"))
            }
            _ => panic!("expected triple"),
        }
    }

    #[test]
    fn test_parse_string_literal() {
        let rules = N3Parser::parse(r#"{ ?x :name "Alice" } => { ?x a :Named } ."#).unwrap();
        match &rules[0].antecedent[0] {
            N3Formula::Triple { object, .. } => {
                assert!(matches!(object, N3Term::Literal { value, .. } if value == "Alice"))
            }
            _ => panic!("expected triple"),
        }
    }

    #[test]
    fn test_parse_number_literal() {
        let rules = N3Parser::parse("{ ?x :age 42 } => { ?x a :MiddleAged } .").unwrap();
        match &rules[0].antecedent[0] {
            N3Formula::Triple { object, .. } => {
                assert!(matches!(object, N3Term::Literal { value, .. } if value == "42"))
            }
            _ => panic!("expected triple"),
        }
    }

    #[test]
    fn test_parse_blank_node() {
        let rules = N3Parser::parse("{ _:b1 :p _:b2 } => { _:b1 :q _:b2 } .").unwrap();
        match &rules[0].antecedent[0] {
            N3Formula::Triple { subject, .. } => {
                assert!(matches!(subject, N3Term::BlankNode(s) if s == "b1"))
            }
            _ => panic!("expected triple"),
        }
    }

    #[test]
    fn test_parse_rdf_type_shorthand() {
        let rules = N3Parser::parse("{ ?x a :Person } => { ?x a :Being } .").unwrap();
        match &rules[0].antecedent[0] {
            N3Formula::Triple { predicate, .. } => {
                assert!(matches!(predicate, N3Term::Iri(p) if p.contains("type")))
            }
            _ => panic!("expected triple"),
        }
    }

    #[test]
    fn test_parse_math_greater_than() {
        let rules = N3Parser::parse("{ ?x math:greaterThan 18 } => { ?x a :Adult } .").unwrap();
        assert!(matches!(
            &rules[0].antecedent[0],
            N3Formula::BuiltIn(N3BuiltIn::MathGreaterThan { .. })
        ));
    }

    #[test]
    fn test_parse_math_less_than() {
        let rules = N3Parser::parse("{ ?x math:lessThan 18 } => { ?x a :Minor } .").unwrap();
        assert!(matches!(
            &rules[0].antecedent[0],
            N3Formula::BuiltIn(N3BuiltIn::MathLessThan { .. })
        ));
    }

    #[test]
    fn test_parse_log_equal() {
        let rules = N3Parser::parse("{ ?x log:equal ?y } => { ?x :same ?y } .").unwrap();
        assert!(matches!(
            &rules[0].antecedent[0],
            N3Formula::BuiltIn(N3BuiltIn::LogEqual { .. })
        ));
    }

    #[test]
    fn test_parse_log_not_equal() {
        let rules = N3Parser::parse("{ ?x log:notEqual ?y } => { ?x :different ?y } .").unwrap();
        assert!(matches!(
            &rules[0].antecedent[0],
            N3Formula::BuiltIn(N3BuiltIn::LogNotEqual { .. })
        ));
    }

    #[test]
    fn test_parse_string_length_builtin() {
        let rules =
            N3Parser::parse(r#"{ ?x string:length ?n } => { ?x :hasLength ?n } ."#).unwrap();
        assert!(matches!(
            &rules[0].antecedent[0],
            N3Formula::BuiltIn(N3BuiltIn::StringLength { .. })
        ));
    }

    #[test]
    fn test_parse_empty_document() {
        let rules = N3Parser::parse("").unwrap();
        assert_eq!(rules.len(), 0);
    }

    #[test]
    fn test_parse_with_comments() {
        let rules = N3Parser::parse("# comment\n{ ?x :p ?y } => { ?x :q ?y } .").unwrap();
        assert_eq!(rules.len(), 1);
    }

    #[test]
    fn test_parse_consequent_variable_order() {
        let rules = N3Parser::parse("{ ?a :parent ?b } => { ?b :child ?a } .").unwrap();
        match &rules[0].consequent[0] {
            N3Formula::Triple { subject, .. } => {
                assert!(matches!(subject, N3Term::Variable(v) if v == "b"))
            }
            _ => panic!("expected triple"),
        }
    }

    #[test]
    fn test_parse_with_prefix_declaration() {
        let n3 = "@prefix ex: <http://example.org/> . { ?x ex:p ?y } => { ?x ex:q ?y } .";
        let rules = N3Parser::parse(n3).unwrap();
        assert_eq!(rules.len(), 1);
    }

    // ── Engine: basic forward chaining ───────────────────────────────────────

    #[test]
    fn test_engine_simple_derivation() {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":p", "?y")],
            vec![tf("?x", ":q", "?y")],
        ));
        engine.assert_fact(Triple::new("a", ":p", "b"));
        let facts = engine.run().unwrap();
        assert!(facts
            .iter()
            .any(|t| t.subject == "a" && t.predicate == ":q" && t.object == "b"));
    }

    #[test]
    fn test_engine_derive_inverse_relation() {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":parent", "?y")],
            vec![tf("?y", ":child", "?x")],
        ));
        engine.assert_fact(Triple::new("alice", ":parent", "bob"));
        let facts = engine.run().unwrap();
        assert!(facts
            .iter()
            .any(|t| t.subject == "bob" && t.predicate == ":child" && t.object == "alice"));
    }

    #[test]
    fn test_engine_chain_two_rules() {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":parent", "?y")],
            vec![tf("?x", ":ancestor", "?y")],
        ));
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":ancestor", "?y"), tf("?y", ":parent", "?z")],
            vec![tf("?x", ":ancestor", "?z")],
        ));
        engine.assert_fact(Triple::new("alice", ":parent", "bob"));
        engine.assert_fact(Triple::new("bob", ":parent", "carol"));
        let facts = engine.run().unwrap();
        assert!(
            facts
                .iter()
                .any(|t| t.subject == "alice" && t.predicate == ":ancestor" && t.object == "carol"),
            "got: {:?}",
            facts
        );
    }

    #[test]
    fn test_engine_bounded_terminates() {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":linked", "?y")],
            vec![tf("?y", ":linked", "?x")],
        ));
        engine.assert_fact(Triple::new("a", ":linked", "b"));
        let facts = engine.run_bounded(5).unwrap();
        assert!(!facts.is_empty());
    }

    #[test]
    fn test_engine_fixpoint_stable() {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":p", "?y")],
            vec![tf("?x", ":q", "?y")],
        ));
        engine.assert_fact(Triple::new("x1", ":p", "y1"));
        let c1 = engine.run_bounded(1).unwrap().len();
        let mut engine2 = N3Engine::new();
        engine2.add_rule(N3Rule::new(
            vec![tf("?x", ":p", "?y")],
            vec![tf("?x", ":q", "?y")],
        ));
        engine2.assert_fact(Triple::new("x1", ":p", "y1"));
        let c100 = engine2.run_bounded(100).unwrap().len();
        assert_eq!(c1, c100);
    }

    #[test]
    fn test_engine_is_derivable_from_fact() {
        let mut engine = N3Engine::new();
        engine.assert_fact(Triple::new("a", ":p", "b"));
        assert!(engine.is_derivable(&Triple::new("a", ":p", "b")));
    }

    #[test]
    fn test_engine_is_not_derivable() {
        let engine = N3Engine::new();
        assert!(!engine.is_derivable(&Triple::new("a", ":p", "b")));
    }

    #[test]
    fn test_engine_is_derivable_via_rule() {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?a", ":knows", "?b")],
            vec![tf("?a", ":acquaintance", "?b")],
        ));
        engine.assert_fact(Triple::new("alice", ":knows", "bob"));
        assert!(engine.is_derivable(&Triple::new("alice", ":acquaintance", "bob")));
    }

    #[test]
    fn test_engine_is_not_derivable_no_match() {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?a", ":knows", "?b")],
            vec![tf("?a", ":acquaintance", "?b")],
        ));
        engine.assert_fact(Triple::new("alice", ":knows", "bob"));
        assert!(!engine.is_derivable(&Triple::new("carol", ":acquaintance", "dave")));
    }

    #[test]
    fn test_engine_no_rules_returns_facts() {
        let mut engine = N3Engine::new();
        engine.assert_fact(Triple::new("a", ":p", "b"));
        let facts = engine.run().unwrap();
        assert_eq!(facts.len(), 1);
    }

    #[test]
    fn test_engine_no_matching_rule() {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":never", "?y")],
            vec![tf("?x", ":derived", "?y")],
        ));
        engine.assert_fact(Triple::new("a", ":different", "b"));
        let facts = engine.run().unwrap();
        assert_eq!(facts.len(), 1);
    }

    #[test]
    fn test_engine_multiple_facts_derived() {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":parent", "?y")],
            vec![tf("?x", ":ancestor", "?y")],
        ));
        engine.assert_fact(Triple::new("alice", ":parent", "bob"));
        engine.assert_fact(Triple::new("carol", ":parent", "dave"));
        let facts = engine.run().unwrap();
        assert!(facts
            .iter()
            .any(|t| t.subject == "alice" && t.predicate == ":ancestor"));
        assert!(facts
            .iter()
            .any(|t| t.subject == "carol" && t.predicate == ":ancestor"));
    }

    #[test]
    fn test_engine_zero_iterations() {
        let mut engine = N3Engine::new();
        engine.assert_fact(Triple::new("a", ":p", "b"));
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":p", "?y")],
            vec![tf("?x", ":q", "?y")],
        ));
        let facts = engine.run_bounded(0).unwrap();
        assert_eq!(facts.len(), 1);
    }

    #[test]
    fn test_engine_variable_binding_consistency() {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":type", "?t"), tf("?y", ":type", "?t")],
            vec![tf("?x", ":sameTypeAs", "?y")],
        ));
        engine.assert_fact(Triple::new("alice", ":type", "Human"));
        engine.assert_fact(Triple::new("bob", ":type", "Human"));
        engine.assert_fact(Triple::new("rex", ":type", "Dog"));
        let facts = engine.run().unwrap();
        assert!(
            facts
                .iter()
                .any(|t| t.subject == "alice" && t.predicate == ":sameTypeAs" && t.object == "bob"),
            "got: {:?}",
            facts
        );
        assert!(!facts
            .iter()
            .any(|t| t.subject == "alice" && t.predicate == ":sameTypeAs" && t.object == "rex"));
    }

    #[test]
    fn test_engine_three_antecedent_chain() {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![
                tf("?a", ":p1", "?b"),
                tf("?b", ":p2", "?c"),
                tf("?c", ":p3", "?d"),
            ],
            vec![tf("?a", ":chain", "?d")],
        ));
        engine.assert_fact(Triple::new("n1", ":p1", "n2"));
        engine.assert_fact(Triple::new("n2", ":p2", "n3"));
        engine.assert_fact(Triple::new("n3", ":p3", "n4"));
        let facts = engine.run().unwrap();
        assert!(
            facts
                .iter()
                .any(|t| t.subject == "n1" && t.predicate == ":chain" && t.object == "n4"),
            "got: {:?}",
            facts
        );
    }

    // ── Math built-ins ───────────────────────────────────────────────────────

    fn lit(v: &str) -> N3Term {
        N3Term::Literal {
            value: v.to_string(),
            datatype: None,
            lang: None,
        }
    }

    #[test]
    fn test_math_gt_passes() {
        let bi = N3BuiltIn::MathGreaterThan {
            left: lit("25"),
            right: lit("18"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_gt_fails() {
        let bi = N3BuiltIn::MathGreaterThan {
            left: lit("10"),
            right: lit("18"),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_lt_passes() {
        let bi = N3BuiltIn::MathLessThan {
            left: lit("10"),
            right: lit("18"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_lt_fails() {
        let bi = N3BuiltIn::MathLessThan {
            left: lit("25"),
            right: lit("18"),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_sum_binds_result() {
        let bi = N3BuiltIn::MathSum {
            args: vec![lit("3"), lit("4")],
            result: N3Term::Variable("r".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(matches!(b.get("r"), Some(N3Term::Literal { value, .. }) if value == "7"));
    }

    #[test]
    fn test_math_sum_checks_correct_result() {
        let bi = N3BuiltIn::MathSum {
            args: vec![lit("3"), lit("4")],
            result: lit("7"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_sum_wrong_result() {
        let bi = N3BuiltIn::MathSum {
            args: vec![lit("3"), lit("4")],
            result: lit("8"),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_difference() {
        let bi = N3BuiltIn::MathDifference {
            args: vec![lit("10"), lit("3")],
            result: N3Term::Variable("r".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(matches!(b.get("r"), Some(N3Term::Literal { value, .. }) if value == "7"));
    }

    #[test]
    fn test_math_product_binds_result() {
        let bi = N3BuiltIn::MathProduct {
            args: vec![lit("3"), lit("4")],
            result: N3Term::Variable("r".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(matches!(b.get("r"), Some(N3Term::Literal { value, .. }) if value == "12"));
    }

    #[test]
    fn test_math_product_checks_result() {
        let bi = N3BuiltIn::MathProduct {
            args: vec![lit("3"), lit("4")],
            result: lit("12"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_quotient() {
        let bi = N3BuiltIn::MathQuotient {
            args: vec![lit("10"), lit("2")],
            result: N3Term::Variable("r".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(matches!(b.get("r"), Some(N3Term::Literal { value, .. }) if value == "5"));
    }

    #[test]
    fn test_math_quotient_div_zero() {
        let bi = N3BuiltIn::MathQuotient {
            args: vec![lit("10"), lit("0")],
            result: N3Term::Variable("r".to_string()),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_equal_to_passes() {
        let bi = N3BuiltIn::MathEqualTo {
            left: lit("5"),
            right: lit("5"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_equal_to_fails() {
        let bi = N3BuiltIn::MathEqualTo {
            left: lit("5"),
            right: lit("6"),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    // ── String built-ins ─────────────────────────────────────────────────────

    #[test]
    fn test_string_concat_binds() {
        let bi = N3BuiltIn::StringConcatenation {
            args: vec![lit("Hello"), lit(" World")],
            result: N3Term::Variable("r".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(
            matches!(b.get("r"), Some(N3Term::Literal { value, .. }) if value == "Hello World")
        );
    }

    #[test]
    fn test_string_concat_checks() {
        let bi = N3BuiltIn::StringConcatenation {
            args: vec![lit("foo"), lit("bar")],
            result: lit("foobar"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_string_length_binds() {
        let bi = N3BuiltIn::StringLength {
            input: lit("hello"),
            result: N3Term::Variable("len".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(matches!(b.get("len"), Some(N3Term::Literal { value, .. }) if value == "5"));
    }

    #[test]
    fn test_string_length_empty() {
        let bi = N3BuiltIn::StringLength {
            input: lit(""),
            result: N3Term::Variable("l".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(matches!(b.get("l"), Some(N3Term::Literal { value, .. }) if value == "0"));
    }

    #[test]
    fn test_string_contains_passes() {
        let bi = N3BuiltIn::StringContains {
            subject: lit("Hello World"),
            substring: lit("World"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_string_contains_fails() {
        let bi = N3BuiltIn::StringContains {
            subject: lit("Hello"),
            substring: lit("World"),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    // ── Log built-ins ────────────────────────────────────────────────────────

    #[test]
    fn test_log_equal_same() {
        let bi = N3BuiltIn::LogEqual {
            left: N3Term::Iri("http://x.org/a".to_string()),
            right: N3Term::Iri("http://x.org/a".to_string()),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_log_equal_different() {
        let bi = N3BuiltIn::LogEqual {
            left: N3Term::Iri("http://x.org/a".to_string()),
            right: N3Term::Iri("http://x.org/b".to_string()),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_log_not_equal_different() {
        let bi = N3BuiltIn::LogNotEqual {
            left: N3Term::Iri("http://x.org/a".to_string()),
            right: N3Term::Iri("http://x.org/b".to_string()),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_log_not_equal_same() {
        let bi = N3BuiltIn::LogNotEqual {
            left: N3Term::Iri("http://x.org/x".to_string()),
            right: N3Term::Iri("http://x.org/x".to_string()),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    // ── Universal variable tests ─────────────────────────────────────────────

    #[test]
    fn test_universal_vars_collected() {
        let n3 = "@forAll :x, :y . { :x :knows :y } => { :y :knownBy :x } .";
        let rules = N3Parser::parse(n3).unwrap();
        assert_eq!(rules[0].universals.len(), 2);
    }

    #[test]
    fn test_existential_vars_collected() {
        let n3 = "@forSome :z . { ?x :knows ?y } => { ?x :connects :z } .";
        let rules = N3Parser::parse(n3).unwrap();
        assert_eq!(rules[0].existentials.len(), 1);
    }

    #[test]
    fn test_universal_term_in_rule_fires() {
        let rule = N3Rule::new(
            vec![N3Formula::Triple {
                subject: N3Term::Universal("x".to_string()),
                predicate: N3Term::Iri(":classOf".to_string()),
                object: N3Term::Universal("y".to_string()),
            }],
            vec![N3Formula::Triple {
                subject: N3Term::Universal("y".to_string()),
                predicate: N3Term::Iri(":memberOf".to_string()),
                object: N3Term::Universal("x".to_string()),
            }],
        )
        .with_universals(vec!["x".to_string(), "y".to_string()]);
        let mut engine = N3Engine::new();
        engine.add_rule(rule);
        engine.assert_fact(Triple::new("Animal", ":classOf", "Dog"));
        let facts = engine.run().unwrap();
        assert!(
            facts
                .iter()
                .any(|t| t.subject == "Dog" && t.predicate == ":memberOf" && t.object == "Animal"),
            "got: {:?}",
            facts
        );
    }

    // ── N3Term display / helpers ─────────────────────────────────────────────

    #[test]
    fn test_term_display_iri() {
        assert_eq!(
            format!("{}", N3Term::Iri("http://x.org".to_string())),
            "<http://x.org>"
        );
    }

    #[test]
    fn test_term_display_variable() {
        assert_eq!(format!("{}", N3Term::Variable("x".to_string())), "?x");
    }

    #[test]
    fn test_term_display_literal() {
        assert_eq!(
            format!(
                "{}",
                N3Term::Literal {
                    value: "hi".to_string(),
                    datatype: None,
                    lang: None
                }
            ),
            "\"hi\""
        );
    }

    #[test]
    fn test_term_display_blank_node() {
        assert_eq!(format!("{}", N3Term::BlankNode("b1".to_string())), "_:b1");
    }

    #[test]
    fn test_term_display_universal() {
        assert_eq!(format!("{}", N3Term::Universal("u".to_string())), "!u");
    }

    #[test]
    fn test_term_is_variable_true() {
        assert!(N3Term::Variable("v".to_string()).is_variable());
        assert!(N3Term::Universal("u".to_string()).is_variable());
    }

    #[test]
    fn test_term_is_variable_false() {
        assert!(!N3Term::Iri("x".to_string()).is_variable());
        assert!(!N3Term::BlankNode("b".to_string()).is_variable());
    }

    // ── N3Rule builder / display ─────────────────────────────────────────────

    #[test]
    fn test_rule_with_universals() {
        let r = N3Rule::new(vec![], vec![]).with_universals(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(r.universals.len(), 2);
    }

    #[test]
    fn test_rule_with_existentials() {
        let r = N3Rule::new(vec![], vec![]).with_existentials(vec!["z".to_string()]);
        assert_eq!(r.existentials.len(), 1);
    }

    #[test]
    fn test_rule_display() {
        let r = N3Rule::new(vec![], vec![]);
        assert!(format!("{}", r).contains("N3Rule"));
    }

    // ── Triple ───────────────────────────────────────────────────────────────

    #[test]
    fn test_triple_equality() {
        assert_eq!(Triple::new("a", "b", "c"), Triple::new("a", "b", "c"));
    }

    #[test]
    fn test_triple_inequality() {
        assert_ne!(Triple::new("a", "b", "c"), Triple::new("a", "b", "d"));
    }

    #[test]
    fn test_engine_no_duplicate_facts() {
        let mut engine = N3Engine::new();
        engine.assert_fact(Triple::new("a", "p", "b"));
        engine.assert_fact(Triple::new("a", "p", "b"));
        assert_eq!(engine.facts.len(), 1);
    }

    #[test]
    fn test_engine_multiple_facts_distinct() {
        let mut engine = N3Engine::new();
        engine.assert_fact(Triple::new("a", "p", "b"));
        engine.assert_fact(Triple::new("b", "p", "c"));
        engine.assert_fact(Triple::new("c", "p", "d"));
        assert_eq!(engine.facts.len(), 3);
    }
}
