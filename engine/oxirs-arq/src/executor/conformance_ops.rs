//! Conformance Operations
//!
//! Implementations for SPARQL algebra operators needed for SPARQL 1.1 conformance:
//! - MINUS (set difference)
//! - EXTEND (BIND)
//! - VALUES (inline data)
//! - Property paths (evaluation via path module)

use crate::algebra::{Algebra, Binding, Solution, Term, Variable};
use anyhow::Result;
use std::collections::{HashSet, VecDeque};

use super::dataset::{convert_property_path, Dataset, DatasetPathAdapter};
use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Execute a MINUS pattern
    ///
    /// SPARQL MINUS removes solutions from P1 that are compatible with any
    /// solution in P2 (based on shared variables).
    pub fn execute_minus(
        &self,
        left: &Algebra,
        right: &Algebra,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        let left_solutions = self.execute_serial(left, dataset)?;
        let right_solutions = self.execute_serial(right, dataset)?;

        if right_solutions.is_empty() {
            return Ok(left_solutions);
        }

        let mut result = Solution::new();
        for left_binding in &left_solutions {
            let left_vars: HashSet<&Variable> = left_binding.keys().collect();

            let mut should_remove = false;
            for right_binding in &right_solutions {
                let right_vars: HashSet<&Variable> = right_binding.keys().collect();
                let shared_vars: Vec<&&Variable> = left_vars.intersection(&right_vars).collect();

                if shared_vars.is_empty() {
                    // Disjoint variables: SPARQL spec says MINUS removes nothing
                    continue;
                }

                // Check compatibility on shared variables
                let mut compatible = true;
                for shared_var in &shared_vars {
                    let left_val = left_binding.get(**shared_var);
                    let right_val = right_binding.get(**shared_var);
                    if left_val != right_val {
                        compatible = false;
                        break;
                    }
                }
                if compatible {
                    should_remove = true;
                    break;
                }
            }

            if !should_remove {
                result.push(left_binding.clone());
            }
        }
        Ok(result)
    }

    /// Execute an EXTEND (BIND) pattern
    ///
    /// Extends each solution with a new variable binding computed by evaluating expr.
    pub fn execute_extend(
        &self,
        pattern: &Algebra,
        variable: &Variable,
        expr: &crate::algebra::Expression,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        let pattern_solutions = self.execute_serial(pattern, dataset)?;
        let mut result = Solution::new();

        for mut binding in pattern_solutions {
            match self.evaluate_expression(expr, &binding) {
                Ok(value) => {
                    binding.insert(variable.clone(), value);
                }
                Err(_) => {
                    // In SPARQL, BIND with an error: the binding is still included
                    // but the variable is unbound (we just don't add it)
                }
            }
            result.push(binding);
        }
        Ok(result)
    }

    /// Execute a VALUES clause
    ///
    /// Returns the inline data as solutions.
    pub fn execute_values(
        &self,
        _variables: &[Variable],
        bindings: &[Binding],
    ) -> Result<Solution> {
        Ok(bindings.to_vec())
    }

    /// Execute a property path pattern
    ///
    /// Evaluates property paths against the dataset using BFS/DFS traversal.
    pub fn execute_property_path(
        &self,
        subject: &Term,
        path: &crate::algebra::PropertyPath,
        object: &Term,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        let adapter = DatasetPathAdapter::new(dataset);
        let path_pp = convert_property_path(path)?;

        let mut result = Solution::new();

        match (subject, object) {
            (Term::Variable(s_var), Term::Variable(o_var)) => {
                // Both variable: enumerate all subjects AND objects as potential starting nodes.
                // This is needed for paths like inverse (^pred) where the "subjects" in path
                // evaluation are the objects in the graph, and vice versa.
                let mut candidates: HashSet<Term> = HashSet::new();
                for s in dataset.subjects()? {
                    candidates.insert(s);
                }
                for o in dataset.objects()? {
                    candidates.insert(o);
                }
                let mut seen_pairs: HashSet<(Term, Term)> = HashSet::new();
                for s in candidates {
                    let reachable = evaluate_path_from(&path_pp, &s, &adapter)?;
                    for o in reachable {
                        let pair = (s.clone(), o.clone());
                        if seen_pairs.insert(pair) {
                            let mut binding = Binding::new();
                            binding.insert(s_var.clone(), s.clone());
                            binding.insert(o_var.clone(), o);
                            result.push(binding);
                        }
                    }
                }
            }
            (Term::Variable(s_var), concrete_obj) => {
                // Subject variable, concrete object: find all subjects that can reach object
                // We enumerate all subjects AND all objects to handle terminal nodes
                // (nodes that appear only as objects, not as subjects)
                let mut candidates: HashSet<Term> = HashSet::new();
                for s in dataset.subjects()? {
                    candidates.insert(s);
                }
                // Also include all objects so terminal nodes (like leaves in hierarchy)
                // are considered as potential starting points
                for o in dataset.objects()? {
                    candidates.insert(o);
                }
                for s in candidates {
                    let reachable = evaluate_path_from(&path_pp, &s, &adapter)?;
                    if reachable.contains(concrete_obj) {
                        let mut binding = Binding::new();
                        binding.insert(s_var.clone(), s);
                        result.push(binding);
                    }
                }
            }
            (concrete_subj, Term::Variable(o_var)) => {
                // Concrete subject, object variable: find all reachable objects
                let reachable = evaluate_path_from(&path_pp, concrete_subj, &adapter)?;
                for o in reachable {
                    let mut binding = Binding::new();
                    binding.insert(o_var.clone(), o);
                    result.push(binding);
                }
            }
            (concrete_subj, concrete_obj) => {
                // Both concrete: check if path exists
                let reachable = evaluate_path_from(&path_pp, concrete_subj, &adapter)?;
                if reachable.contains(concrete_obj) {
                    result.push(Binding::new());
                }
            }
        }
        Ok(result)
    }
}

/// Evaluate a property path from a given subject, returning all reachable objects
fn evaluate_path_from(
    path: &crate::path::PropertyPath,
    subject: &Term,
    adapter: &DatasetPathAdapter<'_>,
) -> Result<HashSet<Term>> {
    use crate::path::PathDataset;
    use crate::path::PropertyPath as PP;

    let mut result = HashSet::new();

    match path {
        PP::Direct(pred) => {
            let objects = adapter.find_outgoing(subject, pred)?;
            result.extend(objects);
        }
        PP::Inverse(inner) => {
            // For inverse: find nodes x such that inner(x, subject)
            // We need to find all nodes from which subject is reachable via inner
            // For a direct inner path, this is efficient:
            if let PP::Direct(pred) = inner.as_ref() {
                let incoming = adapter.find_incoming(pred, subject)?;
                result.extend(incoming);
            } else {
                // For complex inner paths, we enumerate all predicates
                let all_preds = adapter.get_predicates()?;
                for pred in &all_preds {
                    let candidates = adapter.find_incoming(pred, subject)?;
                    for candidate in candidates {
                        let forward = evaluate_path_from(inner, &candidate, adapter)?;
                        if forward.contains(subject) {
                            result.insert(candidate);
                        }
                    }
                }
            }
        }
        PP::Sequence(left, right) => {
            let intermediates = evaluate_path_from(left, subject, adapter)?;
            for mid in &intermediates {
                let right_results = evaluate_path_from(right, mid, adapter)?;
                result.extend(right_results);
            }
        }
        PP::Alternative(left, right) => {
            result.extend(evaluate_path_from(left, subject, adapter)?);
            result.extend(evaluate_path_from(right, subject, adapter)?);
        }
        PP::ZeroOrMore(inner) => {
            // Include subject itself (zero hops)
            result.insert(subject.clone());
            // BFS from subject
            let mut queue = VecDeque::new();
            queue.push_back(subject.clone());
            let mut seen = HashSet::new();
            seen.insert(subject.clone());

            while let Some(current) = queue.pop_front() {
                let next_nodes = evaluate_path_from(inner, &current, adapter)?;
                for next in next_nodes {
                    if !seen.contains(&next) {
                        seen.insert(next.clone());
                        result.insert(next.clone());
                        queue.push_back(next);
                    }
                }
            }
        }
        PP::OneOrMore(inner) => {
            // BFS without including subject itself
            let mut queue = VecDeque::new();
            let mut seen = HashSet::new();
            seen.insert(subject.clone());

            let immediate = evaluate_path_from(inner, subject, adapter)?;
            for node in immediate {
                if !seen.contains(&node) {
                    seen.insert(node.clone());
                    result.insert(node.clone());
                    queue.push_back(node);
                }
            }

            while let Some(current) = queue.pop_front() {
                let next_nodes = evaluate_path_from(inner, &current, adapter)?;
                for next in next_nodes {
                    if !seen.contains(&next) {
                        seen.insert(next.clone());
                        result.insert(next.clone());
                        queue.push_back(next);
                    }
                }
            }
        }
        PP::ZeroOrOne(inner) => {
            // Include subject itself (zero hops)
            result.insert(subject.clone());
            // Include direct successors (one hop)
            result.extend(evaluate_path_from(inner, subject, adapter)?);
        }
        PP::NegatedPropertySet(excluded_preds) => {
            let all_preds = adapter.get_predicates()?;
            let excluded_set: HashSet<&Term> = excluded_preds.iter().collect();

            for pred in &all_preds {
                if excluded_set.is_empty() || !excluded_set.contains(pred) {
                    let objects = adapter.find_outgoing(subject, pred)?;
                    result.extend(objects);
                }
            }
        }
    }

    Ok(result)
}
