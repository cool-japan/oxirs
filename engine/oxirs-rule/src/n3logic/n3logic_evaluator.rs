//! # N3 Logic Engine / Evaluator
//!
//! The `N3Engine` forward-chaining rule engine: applies N3 rules to facts,
//! evaluates built-in predicates, and derives new triples at fixpoint.

use anyhow::Result;

use super::n3logic_types::{Bindings, N3BuiltIn, N3Formula, N3Rule, N3Term, Triple};

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
