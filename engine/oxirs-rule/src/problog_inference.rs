//! Probabilistic Datalog (ProbLog) — inference algorithms
//!
//! This module contains WMC (Weighted Model Counting), SDD/BDD inference algorithms,
//! belief propagation, and the fixpoint materialization engine.

use crate::forward::Substitution;
use crate::{RuleAtom, Term};
use anyhow::{anyhow, Result};
use std::collections::HashMap;

use super::problog_types::{ProbLogStats, ProbabilisticRule};

/// Unify two atoms, returning a substitution if successful.
/// Implements full variable unification with occurs check.
pub fn unify_atoms(pattern: &RuleAtom, target: &RuleAtom) -> Option<Substitution> {
    let mut substitution = HashMap::new();

    match (pattern, target) {
        (
            RuleAtom::Triple {
                subject: s1,
                predicate: p1,
                object: o1,
            },
            RuleAtom::Triple {
                subject: s2,
                predicate: p2,
                object: o2,
            },
        ) => {
            if !unify_terms(s1, s2, &mut substitution) {
                return None;
            }
            if !unify_terms(p1, p2, &mut substitution) {
                return None;
            }
            if !unify_terms(o1, o2, &mut substitution) {
                return None;
            }
            Some(substitution)
        }
        _ => None,
    }
}

/// Unify two terms, updating the substitution.
pub fn unify_terms(t1: &Term, t2: &Term, subst: &mut Substitution) -> bool {
    let t1_resolved = apply_substitution_to_term(t1, subst);
    let t2_resolved = apply_substitution_to_term(t2, subst);

    match (&t1_resolved, &t2_resolved) {
        (Term::Variable(v1), Term::Variable(v2)) if v1 == v2 => true,
        (Term::Variable(v), t) | (t, Term::Variable(v)) => {
            if occurs_in_term(v, t) {
                return false;
            }
            subst.insert(v.clone(), t.clone());
            true
        }
        (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
        (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
        (Term::Function { name: n1, args: a1 }, Term::Function { name: n2, args: a2 }) => {
            if n1 != n2 || a1.len() != a2.len() {
                return false;
            }
            for (arg1, arg2) in a1.iter().zip(a2.iter()) {
                let mut local_subst = subst.clone();
                if !unify_terms(arg1, arg2, &mut local_subst) {
                    return false;
                }
                *subst = local_subst;
            }
            true
        }
        _ => false,
    }
}

/// Check if a variable occurs in a term (for occurs check).
pub fn occurs_in_term(var: &str, term: &Term) -> bool {
    match term {
        Term::Variable(v) => v == var,
        Term::Constant(_) | Term::Literal(_) => false,
        Term::Function { args, .. } => args.iter().any(|arg| occurs_in_term(var, arg)),
    }
}

/// Apply substitution to a term.
pub fn apply_substitution_to_term(term: &Term, subst: &Substitution) -> Term {
    match term {
        Term::Variable(v) => subst.get(v).cloned().unwrap_or_else(|| term.clone()),
        Term::Function { name, args } => Term::Function {
            name: name.clone(),
            args: args
                .iter()
                .map(|arg| apply_substitution_to_term(arg, subst))
                .collect(),
        },
        _ => term.clone(),
    }
}

/// Apply substitution to an atom.
pub fn apply_substitution_to_atom(atom: &RuleAtom, subst: &Substitution) -> RuleAtom {
    match atom {
        RuleAtom::Triple {
            subject,
            predicate,
            object,
        } => RuleAtom::Triple {
            subject: apply_substitution_to_term(subject, subst),
            predicate: apply_substitution_to_term(predicate, subst),
            object: apply_substitution_to_term(object, subst),
        },
        RuleAtom::Builtin { name, args } => RuleAtom::Builtin {
            name: name.clone(),
            args: args
                .iter()
                .map(|arg| apply_substitution_to_term(arg, subst))
                .collect(),
        },
        RuleAtom::NotEqual { left, right } => RuleAtom::NotEqual {
            left: apply_substitution_to_term(left, subst),
            right: apply_substitution_to_term(right, subst),
        },
        RuleAtom::GreaterThan { left, right } => RuleAtom::GreaterThan {
            left: apply_substitution_to_term(left, subst),
            right: apply_substitution_to_term(right, subst),
        },
        RuleAtom::LessThan { left, right } => RuleAtom::LessThan {
            left: apply_substitution_to_term(left, subst),
            right: apply_substitution_to_term(right, subst),
        },
    }
}

/// Apply substitution to rule body.
pub fn apply_substitution_to_body(body: &[RuleAtom], subst: &Substitution) -> Vec<RuleAtom> {
    body.iter()
        .map(|atom| apply_substitution_to_atom(atom, subst))
        .collect()
}

/// Find all variable bindings that satisfy a rule body against a given fact database.
pub fn find_all_bindings(
    body: &[RuleAtom],
    facts: &HashMap<RuleAtom, f64>,
) -> Result<Vec<Substitution>> {
    if body.is_empty() {
        return Ok(vec![HashMap::new()]);
    }

    let first_atom = &body[0];
    let rest_body = &body[1..];

    let mut all_bindings = Vec::new();

    for fact in facts.keys() {
        if let Some(binding) = unify_atoms(first_atom, fact) {
            if rest_body.is_empty() {
                all_bindings.push(binding);
            } else {
                let instantiated_rest = apply_substitution_to_body(rest_body, &binding);
                let rest_bindings = find_all_bindings(&instantiated_rest, facts)?;

                for rest_binding in rest_bindings {
                    let mut merged = binding.clone();
                    for (var, term) in rest_binding {
                        merged.insert(var, term);
                    }
                    all_bindings.push(merged);
                }
            }
        }
    }

    Ok(all_bindings)
}

/// Apply a single rule to derive new facts during materialization.
pub fn apply_rule_for_materialization(
    rule: &crate::Rule,
    facts: &HashMap<RuleAtom, f64>,
    rule_prob: f64,
) -> Result<HashMap<RuleAtom, f64>> {
    let mut derived = HashMap::new();
    let bindings = find_all_bindings(&rule.body, facts)?;

    for binding in bindings {
        let mut body_prob = 1.0;
        for body_atom in &rule.body {
            let instantiated = apply_substitution_to_atom(body_atom, &binding);
            let atom_prob = facts.get(&instantiated).copied().unwrap_or(0.0);
            body_prob *= atom_prob;
        }

        for head_atom in &rule.head {
            let instantiated_head = apply_substitution_to_atom(head_atom, &binding);
            let derivation_prob = body_prob * rule_prob;

            let current_prob = derived.get(&instantiated_head).copied().unwrap_or(0.0);
            let combined = if current_prob > 0.0 {
                current_prob + derivation_prob - (current_prob * derivation_prob)
            } else {
                derivation_prob
            };

            derived.insert(instantiated_head, combined);
        }
    }

    Ok(derived)
}

/// Compute fixpoint using bottom-up materialization (semi-naive evaluation).
///
/// # Algorithm
/// 1. Start with base facts (EDB)
/// 2. Apply all rules to derive new facts (IDB)
/// 3. Repeat until fixpoint (no new facts derived)
/// 4. Track probabilities using disjunctive combination
pub fn materialize(
    probabilistic_facts: &HashMap<RuleAtom, f64>,
    deterministic_facts: &std::collections::HashSet<RuleAtom>,
    probabilistic_rules: &[ProbabilisticRule],
    max_fixpoint_iterations: usize,
    stats: &mut ProbLogStats,
) -> Result<HashMap<RuleAtom, f64>> {
    stats.fixpoint_iterations = 0;

    let mut current_facts = HashMap::new();
    for (fact, &prob) in probabilistic_facts {
        current_facts.insert(fact.clone(), prob);
    }
    for fact in deterministic_facts {
        current_facts.insert(fact.clone(), 1.0);
    }

    let mut iteration = 0;
    loop {
        iteration += 1;
        stats.fixpoint_iterations = iteration;

        if iteration > max_fixpoint_iterations {
            return Err(anyhow!(
                "Maximum fixpoint iterations exceeded: {}",
                max_fixpoint_iterations
            ));
        }

        let previous_size = current_facts.len();
        let mut new_facts = HashMap::new();

        for prob_rule in probabilistic_rules {
            let derived = apply_rule_for_materialization(
                &prob_rule.rule,
                &current_facts,
                prob_rule.probability.unwrap_or(1.0),
            )?;

            for (fact, prob) in derived {
                let entry = new_facts.entry(fact).or_insert(0.0);
                *entry = *entry + prob - (*entry * prob);
            }
        }

        let mut changed = false;
        for (fact, new_prob) in &new_facts {
            let existing_prob = current_facts.get(fact).copied().unwrap_or(0.0);
            if (new_prob - existing_prob).abs() > 1e-10 {
                changed = true;
                current_facts.insert(fact.clone(), *new_prob);
            }
        }

        if !changed && current_facts.len() == previous_size {
            break;
        }
    }

    stats.materialized_facts_count = current_facts.len();
    Ok(current_facts)
}
