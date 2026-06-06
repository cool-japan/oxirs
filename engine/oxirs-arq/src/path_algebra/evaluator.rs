use std::collections::{HashSet, VecDeque};

use super::ast::{Iri, NpsItem, PathDirection, PropertyPath};
use super::PathAlgebraError;

type TripleOracle = Box<dyn Fn(&str, &str, &str) -> bool + Send + Sync>;
type FwdPredicateOracle = Box<dyn Fn(&str) -> Vec<String> + Send + Sync>;
type RevPredicateOracle = Box<dyn Fn(&str) -> Vec<String> + Send + Sync>;
type NeighbourOracle = Box<dyn Fn(&str, &str, PathDirection) -> Vec<String> + Send + Sync>;

/// Breadth-first SPARQL 1.2 property path algebra evaluator.
///
/// Driven by four graph oracles to support both forward and inverse traversal
/// without coupling to any concrete store type:
///
/// - `triple_oracle(s, p, o)` — tests existence of a single triple
/// - `fwd_predicate_oracle(node)` — lists all `p` where `(node, p, ?)` exists
/// - `rev_predicate_oracle(node)` — lists all `p` where `(?, p, node)` exists
/// - `neighbour_oracle(node, p, dir)` — lists endpoints of `(node, p, ?)` or `(?, p, node)`
pub struct PathAlgebraEvaluator {
    _triple_oracle: TripleOracle,
    fwd_predicate_oracle: FwdPredicateOracle,
    rev_predicate_oracle: RevPredicateOracle,
    neighbour_oracle: NeighbourOracle,
}

impl PathAlgebraEvaluator {
    /// Construct a new evaluator with four oracles.
    ///
    /// # Arguments
    /// * `triple_oracle` — returns `true` iff triple `(s, p, o)` exists
    /// * `fwd_predicate_oracle` — returns all `p` such that `(node, p, ?)` exists
    /// * `rev_predicate_oracle` — returns all `p` such that `(?, p, node)` exists
    /// * `neighbour_oracle` — for `Forward` returns all `o` for `(node, p, o)`;
    ///   for `Backward` returns all `s` for `(s, p, node)`
    pub fn new(
        triple_oracle: impl Fn(&str, &str, &str) -> bool + Send + Sync + 'static,
        fwd_predicate_oracle: impl Fn(&str) -> Vec<String> + Send + Sync + 'static,
        rev_predicate_oracle: impl Fn(&str) -> Vec<String> + Send + Sync + 'static,
        neighbour_oracle: impl Fn(&str, &str, PathDirection) -> Vec<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            _triple_oracle: Box::new(triple_oracle),
            fwd_predicate_oracle: Box::new(fwd_predicate_oracle),
            rev_predicate_oracle: Box::new(rev_predicate_oracle),
            neighbour_oracle: Box::new(neighbour_oracle),
        }
    }

    /// Evaluate a property path starting from `start` in the given direction.
    ///
    /// Returns all reachable endpoint nodes.  The result is sorted and deduplicated.
    /// Cycle protection is implemented via a visited-set in every recursive
    /// transitive step.
    pub fn evaluate(
        &self,
        start: &str,
        path: &PropertyPath,
        direction: PathDirection,
    ) -> Result<Vec<String>, PathAlgebraError> {
        let mut visited: HashSet<String> = HashSet::new();
        let results = self.eval_path(start, path, direction, &mut visited)?;
        let mut deduped: Vec<String> = results.into_iter().collect();
        deduped.sort_unstable();
        Ok(deduped)
    }

    /// Evaluate a negated property set from `start` in `direction`.
    ///
    /// Returns all nodes reachable in one step whose predicate is NOT in the NPS.
    ///
    /// SPARQL 1.2 §18.1.8: for a Forward NPS evaluation, `Inverse(p)` items in the
    /// set block the backward one-step traversal of `p`; `Forward(p)` items block
    /// the forward one-step traversal of `p`.
    pub fn evaluate_nps(
        &self,
        start: &str,
        nps: &[NpsItem],
        direction: PathDirection,
    ) -> Result<Vec<String>, PathAlgebraError> {
        let blocked_forward: HashSet<&str> = nps
            .iter()
            .filter_map(|item| {
                if let NpsItem::Forward(iri) = item {
                    Some(iri.as_str())
                } else {
                    None
                }
            })
            .collect();
        let blocked_inverse: HashSet<&str> = nps
            .iter()
            .filter_map(|item| {
                if let NpsItem::Inverse(iri) = item {
                    Some(iri.as_str())
                } else {
                    None
                }
            })
            .collect();

        let mut results = HashSet::new();

        match direction {
            PathDirection::Forward => {
                let predicates = (self.fwd_predicate_oracle)(start);
                for pred in &predicates {
                    if blocked_forward.contains(pred.as_str()) {
                        continue;
                    }
                    for obj in (self.neighbour_oracle)(start, pred, PathDirection::Forward) {
                        results.insert(obj);
                    }
                }
            }
            PathDirection::Backward => {
                // For backward NPS: enumerate predicates where `start` is the object,
                // and skip those blocked by Inverse items.
                let predicates = (self.rev_predicate_oracle)(start);
                for pred in &predicates {
                    if blocked_inverse.contains(pred.as_str()) {
                        continue;
                    }
                    for subj in (self.neighbour_oracle)(start, pred, PathDirection::Backward) {
                        results.insert(subj);
                    }
                }
            }
        }

        let mut out: Vec<String> = results.into_iter().collect();
        out.sort_unstable();
        Ok(out)
    }

    fn eval_path(
        &self,
        start: &str,
        path: &PropertyPath,
        direction: PathDirection,
        visited: &mut HashSet<String>,
    ) -> Result<HashSet<String>, PathAlgebraError> {
        match path {
            PropertyPath::Link(iri) => self.eval_link(start, iri, direction),

            PropertyPath::Inverse(inner) => {
                let flipped = match direction {
                    PathDirection::Forward => PathDirection::Backward,
                    PathDirection::Backward => PathDirection::Forward,
                };
                self.eval_path(start, inner, flipped, visited)
            }

            PropertyPath::Sequence(left, right) => {
                let mid_set = self.eval_path(start, left, direction, visited)?;
                let mut result = HashSet::new();
                for mid in &mid_set {
                    let mut inner_visited = visited.clone();
                    let end_set = self.eval_path(mid, right, direction, &mut inner_visited)?;
                    result.extend(end_set);
                }
                Ok(result)
            }

            PropertyPath::Alternative(left, right) => {
                let mut result = self.eval_path(start, left, direction, visited)?;
                let right_set = self.eval_path(start, right, direction, visited)?;
                result.extend(right_set);
                Ok(result)
            }

            PropertyPath::ZeroOrMore(inner) => {
                self.eval_transitive_closure(start, inner, direction, true)
            }

            PropertyPath::OneOrMore(inner) => {
                self.eval_transitive_closure(start, inner, direction, false)
            }

            PropertyPath::Optional(inner) => {
                let mut result = HashSet::new();
                result.insert(start.to_string());
                let one_step = self.eval_path(start, inner, direction, visited)?;
                result.extend(one_step);
                Ok(result)
            }

            PropertyPath::NegatedPropertySet(nps_items) => {
                let nodes = self.evaluate_nps(start, nps_items, direction)?;
                Ok(nodes.into_iter().collect())
            }
        }
    }

    /// BFS transitive closure implementing `*` (include_start=true) and `+` (include_start=false).
    ///
    /// Cycle protection via a globally shared visited set across BFS steps.
    fn eval_transitive_closure(
        &self,
        start: &str,
        path: &PropertyPath,
        direction: PathDirection,
        include_start: bool,
    ) -> Result<HashSet<String>, PathAlgebraError> {
        let mut result: HashSet<String> = HashSet::new();
        if include_start {
            result.insert(start.to_string());
        }

        let mut queue: VecDeque<String> = VecDeque::new();
        let mut globally_visited: HashSet<String> = HashSet::new();
        globally_visited.insert(start.to_string());
        queue.push_back(start.to_string());

        while let Some(current) = queue.pop_front() {
            let mut step_visited = HashSet::new();
            let neighbours = self.eval_path(&current, path, direction, &mut step_visited)?;
            for neighbour in neighbours {
                if !globally_visited.contains(&neighbour) {
                    globally_visited.insert(neighbour.clone());
                    result.insert(neighbour.clone());
                    queue.push_back(neighbour);
                }
            }
        }
        Ok(result)
    }

    fn eval_link(
        &self,
        start: &str,
        iri: &Iri,
        direction: PathDirection,
    ) -> Result<HashSet<String>, PathAlgebraError> {
        let neighbours = (self.neighbour_oracle)(start, iri, direction);
        Ok(neighbours.into_iter().collect())
    }
}
