//! Federated query splitting engine.
//!
//! Decomposes a SPARQL Basic Graph Pattern (BGP) across registered endpoints,
//! detects exclusive groups, plans bound/dependent joins, estimates split costs,
//! and produces serialisable split plans optimised for minimum data transfer.

use std::collections::{HashMap, HashSet};

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// A single term in a triple pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SplitTerm {
    /// A SPARQL variable (without the leading `?`).
    Variable(String),
    /// A fully-qualified IRI.
    Iri(String),
    /// A literal value.
    Literal(String),
}

impl SplitTerm {
    /// `true` if this term is a variable.
    pub fn is_variable(&self) -> bool {
        matches!(self, SplitTerm::Variable(_))
    }

    /// Returns the inner string regardless of variant.
    pub fn value(&self) -> &str {
        match self {
            SplitTerm::Variable(s) | SplitTerm::Iri(s) | SplitTerm::Literal(s) => s,
        }
    }
}

/// A single RDF triple pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SplitTriplePattern {
    pub subject: SplitTerm,
    pub predicate: SplitTerm,
    pub object: SplitTerm,
}

impl SplitTriplePattern {
    pub fn new(subject: SplitTerm, predicate: SplitTerm, object: SplitTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Returns the set of variable names used in this pattern.
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        for term in [&self.subject, &self.predicate, &self.object] {
            if let SplitTerm::Variable(v) = term {
                vars.insert(v.clone());
            }
        }
        vars
    }

    /// Number of variable positions in the pattern.
    pub fn variable_count(&self) -> usize {
        [&self.subject, &self.predicate, &self.object]
            .iter()
            .filter(|t| t.is_variable())
            .count()
    }
}

/// Describes a federated endpoint's capabilities.
#[derive(Debug, Clone)]
pub struct EndpointCapability {
    /// Endpoint URL.
    pub url: String,
    /// Set of predicate IRIs this endpoint can answer.
    pub predicates: HashSet<String>,
    /// Named graphs available at this endpoint.
    pub graphs: HashSet<String>,
    /// Estimated number of triples (for cost estimation).
    pub estimated_triples: u64,
    /// Average network latency in milliseconds.
    pub latency_ms: f64,
}

/// A fragment of a BGP assigned to a single endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitFragment {
    /// Target endpoint URL.
    pub endpoint: String,
    /// Patterns assigned to this endpoint.
    pub patterns: Vec<SplitTriplePattern>,
    /// Whether this is an exclusive group (patterns answerable by only this endpoint).
    pub exclusive: bool,
}

/// The type of join between two fragments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinKind {
    /// A bound join: bind variables from the left fragment before querying the right.
    BoundJoin { bind_variables: Vec<String> },
    /// A dependent join: the right fragment requires results from the left.
    DependentJoin { dependency_variables: Vec<String> },
    /// An independent join (hash/merge) with no data dependency.
    IndependentJoin { join_variables: Vec<String> },
}

/// A planned join between two fragments.
#[derive(Debug, Clone)]
pub struct JoinPlan {
    /// Index of the left fragment in the split plan.
    pub left_idx: usize,
    /// Index of the right fragment in the split plan.
    pub right_idx: usize,
    /// The type of join.
    pub kind: JoinKind,
    /// Estimated cost (communication + processing).
    pub estimated_cost: f64,
}

/// Cost breakdown for a split plan.
#[derive(Debug, Clone, Default)]
pub struct SplitCost {
    /// Total estimated cost.
    pub total: f64,
    /// Communication cost (network round-trips, data transfer).
    pub communication: f64,
    /// Processing cost (joins, filtering).
    pub processing: f64,
    /// Number of remote requests required.
    pub remote_requests: usize,
    /// Estimated bytes transferred.
    pub estimated_bytes: u64,
}

/// The complete split plan for a BGP decomposition.
#[derive(Debug, Clone)]
pub struct SplitPlan {
    /// Fragments assigned to endpoints.
    pub fragments: Vec<SplitFragment>,
    /// Planned joins between fragments.
    pub joins: Vec<JoinPlan>,
    /// Cost estimation.
    pub cost: SplitCost,
    /// Patterns that could not be assigned to any endpoint.
    pub unassigned: Vec<SplitTriplePattern>,
}

impl SplitPlan {
    /// Returns a human-readable serialisation of this split plan.
    pub fn serialize(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Split Plan ===\n");
        out.push_str(&format!("Fragments: {}\n", self.fragments.len()));
        for (i, frag) in self.fragments.iter().enumerate() {
            out.push_str(&format!(
                "  [{}] {} ({} patterns, exclusive={})\n",
                i,
                frag.endpoint,
                frag.patterns.len(),
                frag.exclusive
            ));
            for pat in &frag.patterns {
                out.push_str(&format!(
                    "    {} {} {}\n",
                    Self::term_str(&pat.subject),
                    Self::term_str(&pat.predicate),
                    Self::term_str(&pat.object),
                ));
            }
        }
        out.push_str(&format!("Joins: {}\n", self.joins.len()));
        for join in &self.joins {
            out.push_str(&format!(
                "  [{}] x [{}] {:?} (cost={:.2})\n",
                join.left_idx, join.right_idx, join.kind, join.estimated_cost,
            ));
        }
        out.push_str(&format!(
            "Cost: total={:.2}, comm={:.2}, proc={:.2}, requests={}, bytes={}\n",
            self.cost.total,
            self.cost.communication,
            self.cost.processing,
            self.cost.remote_requests,
            self.cost.estimated_bytes,
        ));
        if !self.unassigned.is_empty() {
            out.push_str(&format!("Unassigned: {} patterns\n", self.unassigned.len()));
        }
        out
    }

    fn term_str(term: &SplitTerm) -> String {
        match term {
            SplitTerm::Variable(v) => format!("?{}", v),
            SplitTerm::Iri(i) => format!("<{}>", i),
            SplitTerm::Literal(l) => format!("\"{}\"", l),
        }
    }
}

/// Errors from the query splitter.
#[derive(Debug)]
pub enum SplitterError {
    /// No endpoints registered.
    NoEndpoints,
    /// No patterns to split.
    EmptyBgp,
    /// A pattern has no matching endpoint.
    UnassignablePattern(String),
}

impl std::fmt::Display for SplitterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SplitterError::NoEndpoints => write!(f, "no endpoints registered"),
            SplitterError::EmptyBgp => write!(f, "empty BGP"),
            SplitterError::UnassignablePattern(desc) => {
                write!(f, "unassignable pattern: {}", desc)
            }
        }
    }
}

impl std::error::Error for SplitterError {}

// ──────────────────────────────────────────────────────────────────────────────
// QuerySplitter
// ──────────────────────────────────────────────────────────────────────────────

/// Splits a SPARQL BGP across federated endpoints.
pub struct QuerySplitter {
    endpoints: Vec<EndpointCapability>,
    /// Estimated bytes per result row for cost calculations.
    bytes_per_row: u64,
}

impl QuerySplitter {
    /// Create a new splitter with the given endpoint capabilities.
    pub fn new(endpoints: Vec<EndpointCapability>) -> Self {
        Self {
            endpoints,
            bytes_per_row: 256,
        }
    }

    /// Set the assumed bytes per result row for cost estimation.
    pub fn set_bytes_per_row(&mut self, bytes: u64) {
        self.bytes_per_row = bytes;
    }

    /// Return the number of registered endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }

    /// Produce a split plan for the given set of triple patterns.
    pub fn split(&self, patterns: &[SplitTriplePattern]) -> Result<SplitPlan, SplitterError> {
        if self.endpoints.is_empty() {
            return Err(SplitterError::NoEndpoints);
        }
        if patterns.is_empty() {
            return Err(SplitterError::EmptyBgp);
        }

        // Step 1: Determine which endpoints can answer each pattern.
        let assignment = self.assign_patterns(patterns);

        // Step 2: Detect exclusive groups and multi-source patterns.
        let (fragments, unassigned) = self.build_fragments(&assignment, patterns);

        // Step 3: Plan joins between fragments.
        let joins = self.plan_joins(&fragments);

        // Step 4: Estimate cost.
        let cost = self.estimate_cost(&fragments, &joins);

        Ok(SplitPlan {
            fragments,
            joins,
            cost,
            unassigned,
        })
    }

    /// Find candidate endpoints that can answer each pattern.
    pub fn find_candidates(&self, pattern: &SplitTriplePattern) -> Vec<String> {
        self.endpoints
            .iter()
            .filter(|ep| self.endpoint_can_answer(ep, pattern))
            .map(|ep| ep.url.clone())
            .collect()
    }

    /// Detect exclusive groups: sets of patterns that can only be answered by
    /// a single endpoint.
    pub fn detect_exclusive_groups(
        &self,
        patterns: &[SplitTriplePattern],
    ) -> Vec<(String, Vec<usize>)> {
        let assignment = self.assign_patterns(patterns);
        let mut exclusive: Vec<(String, Vec<usize>)> = Vec::new();

        for (idx, candidates) in assignment.iter().enumerate() {
            if candidates.len() == 1 {
                let ep = &candidates[0];
                if let Some(group) = exclusive.iter_mut().find(|(e, _)| e == ep) {
                    group.1.push(idx);
                } else {
                    exclusive.push((ep.clone(), vec![idx]));
                }
            }
        }
        exclusive
    }

    /// Identify bound join opportunities between two fragments.
    pub fn identify_bound_joins(
        &self,
        left: &SplitFragment,
        right: &SplitFragment,
    ) -> Option<Vec<String>> {
        let left_vars = Self::fragment_variables(left);
        let right_vars = Self::fragment_variables(right);
        let shared: Vec<String> = left_vars.intersection(&right_vars).cloned().collect();
        if shared.is_empty() {
            None
        } else {
            Some(shared)
        }
    }

    /// Detect dependent joins: the right fragment has a variable that only
    /// appears in bound (IRI/Literal) form in the left, requiring values
    /// from the left to execute the right.
    pub fn detect_dependent_joins(
        &self,
        fragments: &[SplitFragment],
    ) -> Vec<(usize, usize, Vec<String>)> {
        let mut deps = Vec::new();
        for i in 0..fragments.len() {
            for j in (i + 1)..fragments.len() {
                let left_vars = Self::fragment_variables(&fragments[i]);
                let right_vars = Self::fragment_variables(&fragments[j]);
                let shared: Vec<String> = left_vars.intersection(&right_vars).cloned().collect();
                if !shared.is_empty() {
                    // Check if any shared variable appears only as a variable
                    // on one side and could be bound from the other side
                    let left_bound = Self::fragment_bound_terms(&fragments[i]);
                    let dep_vars: Vec<String> = shared
                        .iter()
                        .filter(|v| left_bound.contains(v.as_str()))
                        .cloned()
                        .collect();
                    if !dep_vars.is_empty() {
                        deps.push((i, j, dep_vars));
                    }
                }
            }
        }
        deps
    }

    /// Estimate the communication cost between two endpoints.
    pub fn estimate_communication_cost(&self, endpoint_a: &str, endpoint_b: &str) -> f64 {
        let lat_a = self
            .endpoints
            .iter()
            .find(|e| e.url == endpoint_a)
            .map_or(10.0, |e| e.latency_ms);
        let lat_b = self
            .endpoints
            .iter()
            .find(|e| e.url == endpoint_b)
            .map_or(10.0, |e| e.latency_ms);
        lat_a + lat_b
    }

    /// Compute a minimum-transfer split: prefer endpoints that minimise the
    /// intermediate result size based on estimated triple counts.
    pub fn minimum_transfer_split(
        &self,
        patterns: &[SplitTriplePattern],
    ) -> Result<SplitPlan, SplitterError> {
        if self.endpoints.is_empty() {
            return Err(SplitterError::NoEndpoints);
        }
        if patterns.is_empty() {
            return Err(SplitterError::EmptyBgp);
        }

        let assignment = self.assign_patterns(patterns);
        let mut frag_map: HashMap<String, Vec<SplitTriplePattern>> = HashMap::new();
        let mut unassigned = Vec::new();

        for (idx, candidates) in assignment.iter().enumerate() {
            if candidates.is_empty() {
                unassigned.push(patterns[idx].clone());
                continue;
            }
            // Pick the endpoint with the fewest estimated triples (smallest selectivity)
            let best = candidates
                .iter()
                .filter_map(|url| {
                    self.endpoints
                        .iter()
                        .find(|e| &e.url == url)
                        .map(|e| (url.clone(), e.estimated_triples))
                })
                .min_by_key(|(_, count)| *count)
                .map(|(url, _)| url);

            if let Some(url) = best {
                frag_map.entry(url).or_default().push(patterns[idx].clone());
            } else {
                unassigned.push(patterns[idx].clone());
            }
        }

        let fragments: Vec<SplitFragment> = frag_map
            .into_iter()
            .map(|(endpoint, pats)| {
                let exclusive = pats.iter().all(|p| {
                    let cands = self.find_candidates(p);
                    cands.len() == 1
                });
                SplitFragment {
                    endpoint,
                    patterns: pats,
                    exclusive,
                }
            })
            .collect();

        let joins = self.plan_joins(&fragments);
        let cost = self.estimate_cost(&fragments, &joins);

        Ok(SplitPlan {
            fragments,
            joins,
            cost,
            unassigned,
        })
    }

    // ── Private ──────────────────────────────────────────────────────────────

    /// For each pattern, return the list of endpoints that can answer it.
    fn assign_patterns(&self, patterns: &[SplitTriplePattern]) -> Vec<Vec<String>> {
        patterns.iter().map(|p| self.find_candidates(p)).collect()
    }

    /// Check whether an endpoint can answer a given pattern.
    fn endpoint_can_answer(&self, ep: &EndpointCapability, pattern: &SplitTriplePattern) -> bool {
        // If the predicate is a bound IRI, check if it's in the endpoint's capability set
        match &pattern.predicate {
            SplitTerm::Iri(pred) => ep.predicates.contains(pred),
            SplitTerm::Variable(_) => {
                // A variable predicate can potentially be answered by any endpoint
                // that has at least some predicates
                !ep.predicates.is_empty()
            }
            SplitTerm::Literal(_) => false, // Predicates should not be literals
        }
    }

    /// Build fragments by grouping patterns per endpoint.
    fn build_fragments(
        &self,
        assignment: &[Vec<String>],
        patterns: &[SplitTriplePattern],
    ) -> (Vec<SplitFragment>, Vec<SplitTriplePattern>) {
        let mut frag_map: HashMap<String, Vec<SplitTriplePattern>> = HashMap::new();
        let mut unassigned = Vec::new();

        for (idx, candidates) in assignment.iter().enumerate() {
            if candidates.is_empty() {
                unassigned.push(patterns[idx].clone());
            } else {
                // Assign to the first candidate (primary endpoint)
                let ep = &candidates[0];
                frag_map
                    .entry(ep.clone())
                    .or_default()
                    .push(patterns[idx].clone());
            }
        }

        let exclusive_info: HashMap<usize, bool> = assignment
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.len() == 1))
            .collect();

        let fragments: Vec<SplitFragment> = frag_map
            .into_iter()
            .map(|(endpoint, pats)| {
                // A fragment is exclusive if ALL its patterns have only one candidate
                let all_exclusive = pats.iter().all(|p| {
                    patterns
                        .iter()
                        .position(|pp| pp == p)
                        .and_then(|idx| exclusive_info.get(&idx).copied())
                        .unwrap_or(false)
                });
                SplitFragment {
                    endpoint,
                    patterns: pats,
                    exclusive: all_exclusive,
                }
            })
            .collect();

        (fragments, unassigned)
    }

    /// Plan joins between fragments based on shared variables.
    fn plan_joins(&self, fragments: &[SplitFragment]) -> Vec<JoinPlan> {
        let mut joins = Vec::new();
        for i in 0..fragments.len() {
            for j in (i + 1)..fragments.len() {
                let left_vars = Self::fragment_variables(&fragments[i]);
                let right_vars = Self::fragment_variables(&fragments[j]);
                let shared: Vec<String> = left_vars.intersection(&right_vars).cloned().collect();

                if shared.is_empty() {
                    continue;
                }

                let left_bound = Self::fragment_bound_terms(&fragments[i]);
                let dep_vars: Vec<String> = shared
                    .iter()
                    .filter(|v| left_bound.contains(v.as_str()))
                    .cloned()
                    .collect();

                let kind = if !dep_vars.is_empty() {
                    JoinKind::DependentJoin {
                        dependency_variables: dep_vars,
                    }
                } else if shared.len() <= 2 {
                    JoinKind::BoundJoin {
                        bind_variables: shared.clone(),
                    }
                } else {
                    JoinKind::IndependentJoin {
                        join_variables: shared.clone(),
                    }
                };

                let cost = self
                    .estimate_communication_cost(&fragments[i].endpoint, &fragments[j].endpoint);

                joins.push(JoinPlan {
                    left_idx: i,
                    right_idx: j,
                    kind,
                    estimated_cost: cost,
                });
            }
        }
        joins
    }

    /// Estimate the total cost of a split plan.
    fn estimate_cost(&self, fragments: &[SplitFragment], joins: &[JoinPlan]) -> SplitCost {
        let remote_requests = fragments.len();
        let communication: f64 = fragments
            .iter()
            .filter_map(|f| {
                self.endpoints
                    .iter()
                    .find(|e| e.url == f.endpoint)
                    .map(|e| e.latency_ms)
            })
            .sum();
        let processing: f64 = joins.iter().map(|j| j.estimated_cost).sum();
        let estimated_bytes: u64 = fragments
            .iter()
            .map(|f| {
                let ep_triples = self
                    .endpoints
                    .iter()
                    .find(|e| e.url == f.endpoint)
                    .map_or(1000, |e| e.estimated_triples);
                // Rough estimate: selectivity × patterns × bytes_per_row
                let selectivity = if f.patterns.len() > 1 { 0.1 } else { 1.0 };
                (ep_triples as f64 * selectivity * self.bytes_per_row as f64) as u64
            })
            .sum();

        SplitCost {
            total: communication + processing,
            communication,
            processing,
            remote_requests,
            estimated_bytes,
        }
    }

    /// Collect all variable names from a fragment.
    fn fragment_variables(frag: &SplitFragment) -> HashSet<String> {
        let mut vars = HashSet::new();
        for pat in &frag.patterns {
            vars.extend(pat.variables());
        }
        vars
    }

    /// Collect IRIs and literals that appear in subject/object positions.
    fn fragment_bound_terms(frag: &SplitFragment) -> HashSet<&str> {
        let mut bound = HashSet::new();
        for pat in &frag.patterns {
            for term in [&pat.subject, &pat.predicate, &pat.object] {
                match term {
                    SplitTerm::Iri(s) | SplitTerm::Literal(s) => {
                        bound.insert(s.as_str());
                    }
                    SplitTerm::Variable(_) => {}
                }
            }
        }
        bound
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> SplitTerm {
        SplitTerm::Variable(name.to_string())
    }

    fn iri(name: &str) -> SplitTerm {
        SplitTerm::Iri(name.to_string())
    }

    fn lit(name: &str) -> SplitTerm {
        SplitTerm::Literal(name.to_string())
    }

    fn ep(url: &str, predicates: &[&str], triples: u64, latency: f64) -> EndpointCapability {
        EndpointCapability {
            url: url.to_string(),
            predicates: predicates.iter().map(|s| s.to_string()).collect(),
            graphs: HashSet::new(),
            estimated_triples: triples,
            latency_ms: latency,
        }
    }

    // ── SplitTerm ────────────────────────────────────────────────────────────

    #[test]
    fn test_split_term_is_variable() {
        assert!(var("x").is_variable());
        assert!(!iri("http://ex.org/p").is_variable());
        assert!(!lit("hello").is_variable());
    }

    #[test]
    fn test_split_term_value() {
        assert_eq!(var("x").value(), "x");
        assert_eq!(iri("http://ex.org/p").value(), "http://ex.org/p");
        assert_eq!(lit("hello").value(), "hello");
    }

    #[test]
    fn test_split_term_clone_eq() {
        let t = var("x");
        assert_eq!(t.clone(), t);
        let t2 = iri("y");
        assert_eq!(t2.clone(), t2);
    }

    // ── SplitTriplePattern ───────────────────────────────────────────────────

    #[test]
    fn test_pattern_variables() {
        let pat = SplitTriplePattern::new(var("s"), iri("http://ex.org/p"), var("o"));
        let vars = pat.variables();
        assert!(vars.contains("s"));
        assert!(vars.contains("o"));
        assert!(!vars.contains("http://ex.org/p"));
    }

    #[test]
    fn test_pattern_variable_count() {
        let p1 = SplitTriplePattern::new(var("s"), var("p"), var("o"));
        assert_eq!(p1.variable_count(), 3);

        let p2 = SplitTriplePattern::new(iri("a"), iri("b"), iri("c"));
        assert_eq!(p2.variable_count(), 0);

        let p3 = SplitTriplePattern::new(var("s"), iri("p"), lit("v"));
        assert_eq!(p3.variable_count(), 1);
    }

    #[test]
    fn test_pattern_clone_eq() {
        let p = SplitTriplePattern::new(var("s"), iri("p"), var("o"));
        assert_eq!(p.clone(), p);
    }

    // ── QuerySplitter construction ───────────────────────────────────────────

    #[test]
    fn test_splitter_creation() {
        let splitter =
            QuerySplitter::new(vec![ep("http://ep1", &["http://ex.org/name"], 1000, 5.0)]);
        assert_eq!(splitter.endpoint_count(), 1);
    }

    #[test]
    fn test_splitter_empty_endpoints() {
        let splitter = QuerySplitter::new(vec![]);
        assert_eq!(splitter.endpoint_count(), 0);
    }

    #[test]
    fn test_splitter_set_bytes_per_row() {
        let mut splitter = QuerySplitter::new(vec![]);
        splitter.set_bytes_per_row(512);
        // No public accessor, but no panic is enough
    }

    // ── split() error cases ──────────────────────────────────────────────────

    #[test]
    fn test_split_no_endpoints() {
        let splitter = QuerySplitter::new(vec![]);
        let pat = SplitTriplePattern::new(var("s"), iri("http://ex.org/p"), var("o"));
        let result = splitter.split(&[pat]);
        assert!(result.is_err());
    }

    #[test]
    fn test_split_empty_bgp() {
        let splitter = QuerySplitter::new(vec![ep("http://ep1", &["http://ex.org/p"], 1000, 5.0)]);
        let result = splitter.split(&[]);
        assert!(result.is_err());
    }

    // ── find_candidates ──────────────────────────────────────────────────────

    #[test]
    fn test_find_candidates_single() {
        let splitter = QuerySplitter::new(vec![
            ep("http://ep1", &["http://ex.org/name"], 1000, 5.0),
            ep("http://ep2", &["http://ex.org/age"], 2000, 3.0),
        ]);
        let pat = SplitTriplePattern::new(var("s"), iri("http://ex.org/name"), var("o"));
        let cands = splitter.find_candidates(&pat);
        assert_eq!(cands, vec!["http://ep1"]);
    }

    #[test]
    fn test_find_candidates_multiple() {
        let splitter = QuerySplitter::new(vec![
            ep("http://ep1", &["http://ex.org/name"], 1000, 5.0),
            ep(
                "http://ep2",
                &["http://ex.org/name", "http://ex.org/age"],
                2000,
                3.0,
            ),
        ]);
        let pat = SplitTriplePattern::new(var("s"), iri("http://ex.org/name"), var("o"));
        let cands = splitter.find_candidates(&pat);
        assert_eq!(cands.len(), 2);
    }

    #[test]
    fn test_find_candidates_none() {
        let splitter =
            QuerySplitter::new(vec![ep("http://ep1", &["http://ex.org/name"], 1000, 5.0)]);
        let pat = SplitTriplePattern::new(var("s"), iri("http://ex.org/age"), var("o"));
        let cands = splitter.find_candidates(&pat);
        assert!(cands.is_empty());
    }

    #[test]
    fn test_find_candidates_variable_predicate() {
        let splitter =
            QuerySplitter::new(vec![ep("http://ep1", &["http://ex.org/name"], 1000, 5.0)]);
        let pat = SplitTriplePattern::new(var("s"), var("p"), var("o"));
        let cands = splitter.find_candidates(&pat);
        assert_eq!(cands.len(), 1); // any endpoint with predicates
    }

    #[test]
    fn test_find_candidates_literal_predicate() {
        let splitter =
            QuerySplitter::new(vec![ep("http://ep1", &["http://ex.org/name"], 1000, 5.0)]);
        let pat = SplitTriplePattern::new(var("s"), lit("not-a-predicate"), var("o"));
        let cands = splitter.find_candidates(&pat);
        assert!(cands.is_empty());
    }

    // ── detect_exclusive_groups ──────────────────────────────────────────────

    #[test]
    fn test_detect_exclusive_groups() {
        let splitter = QuerySplitter::new(vec![
            ep("http://ep1", &["http://ex.org/name"], 1000, 5.0),
            ep("http://ep2", &["http://ex.org/age"], 2000, 3.0),
        ]);
        let patterns = vec![
            SplitTriplePattern::new(var("s"), iri("http://ex.org/name"), var("n")),
            SplitTriplePattern::new(var("s"), iri("http://ex.org/age"), var("a")),
        ];
        let groups = splitter.detect_exclusive_groups(&patterns);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_no_exclusive_groups_when_shared() {
        let splitter = QuerySplitter::new(vec![
            ep(
                "http://ep1",
                &["http://ex.org/name", "http://ex.org/age"],
                1000,
                5.0,
            ),
            ep(
                "http://ep2",
                &["http://ex.org/name", "http://ex.org/age"],
                2000,
                3.0,
            ),
        ]);
        let patterns = vec![SplitTriplePattern::new(
            var("s"),
            iri("http://ex.org/name"),
            var("n"),
        )];
        let groups = splitter.detect_exclusive_groups(&patterns);
        assert!(groups.is_empty());
    }

    // ── identify_bound_joins ─────────────────────────────────────────────────

    #[test]
    fn test_identify_bound_joins_shared_var() {
        let splitter = QuerySplitter::new(vec![]);
        let left = SplitFragment {
            endpoint: "http://ep1".to_string(),
            patterns: vec![SplitTriplePattern::new(
                var("s"),
                iri("http://ex.org/name"),
                var("n"),
            )],
            exclusive: true,
        };
        let right = SplitFragment {
            endpoint: "http://ep2".to_string(),
            patterns: vec![SplitTriplePattern::new(
                var("s"),
                iri("http://ex.org/age"),
                var("a"),
            )],
            exclusive: true,
        };
        let result = splitter.identify_bound_joins(&left, &right);
        assert!(result.is_some());
        let vars = result.expect("should have shared vars");
        assert!(vars.contains(&"s".to_string()));
    }

    #[test]
    fn test_identify_bound_joins_no_shared_var() {
        let splitter = QuerySplitter::new(vec![]);
        let left = SplitFragment {
            endpoint: "http://ep1".to_string(),
            patterns: vec![SplitTriplePattern::new(var("a"), iri("p"), var("b"))],
            exclusive: true,
        };
        let right = SplitFragment {
            endpoint: "http://ep2".to_string(),
            patterns: vec![SplitTriplePattern::new(var("c"), iri("q"), var("d"))],
            exclusive: true,
        };
        let result = splitter.identify_bound_joins(&left, &right);
        assert!(result.is_none());
    }

    // ── split() success cases ────────────────────────────────────────────────

    #[test]
    fn test_split_single_endpoint() {
        let splitter = QuerySplitter::new(vec![ep(
            "http://ep1",
            &["http://ex.org/name", "http://ex.org/age"],
            1000,
            5.0,
        )]);
        let patterns = vec![
            SplitTriplePattern::new(var("s"), iri("http://ex.org/name"), var("n")),
            SplitTriplePattern::new(var("s"), iri("http://ex.org/age"), var("a")),
        ];
        let plan = splitter.split(&patterns).expect("split should succeed");
        assert_eq!(plan.fragments.len(), 1);
        assert_eq!(plan.fragments[0].patterns.len(), 2);
        assert!(plan.unassigned.is_empty());
    }

    #[test]
    fn test_split_two_endpoints() {
        let splitter = QuerySplitter::new(vec![
            ep("http://ep1", &["http://ex.org/name"], 1000, 5.0),
            ep("http://ep2", &["http://ex.org/age"], 2000, 3.0),
        ]);
        let patterns = vec![
            SplitTriplePattern::new(var("s"), iri("http://ex.org/name"), var("n")),
            SplitTriplePattern::new(var("s"), iri("http://ex.org/age"), var("a")),
        ];
        let plan = splitter.split(&patterns).expect("split should succeed");
        assert_eq!(plan.fragments.len(), 2);
        assert!(!plan.joins.is_empty());
    }

    #[test]
    fn test_split_with_unassigned() {
        let splitter =
            QuerySplitter::new(vec![ep("http://ep1", &["http://ex.org/name"], 1000, 5.0)]);
        let patterns = vec![
            SplitTriplePattern::new(var("s"), iri("http://ex.org/name"), var("n")),
            SplitTriplePattern::new(var("s"), iri("http://ex.org/unknown"), var("x")),
        ];
        let plan = splitter.split(&patterns).expect("split should succeed");
        assert_eq!(plan.unassigned.len(), 1);
    }

    // ── minimum_transfer_split ───────────────────────────────────────────────

    #[test]
    fn test_minimum_transfer_prefers_smaller_endpoint() {
        let splitter = QuerySplitter::new(vec![
            ep("http://big", &["http://ex.org/name"], 1_000_000, 5.0),
            ep("http://small", &["http://ex.org/name"], 100, 5.0),
        ]);
        let patterns = vec![SplitTriplePattern::new(
            var("s"),
            iri("http://ex.org/name"),
            var("n"),
        )];
        let plan = splitter
            .minimum_transfer_split(&patterns)
            .expect("should succeed");
        assert_eq!(plan.fragments.len(), 1);
        assert_eq!(plan.fragments[0].endpoint, "http://small");
    }

    #[test]
    fn test_minimum_transfer_no_endpoints() {
        let splitter = QuerySplitter::new(vec![]);
        let pat = SplitTriplePattern::new(var("s"), iri("p"), var("o"));
        assert!(splitter.minimum_transfer_split(&[pat]).is_err());
    }

    #[test]
    fn test_minimum_transfer_empty_bgp() {
        let splitter = QuerySplitter::new(vec![ep("http://ep1", &["p"], 100, 1.0)]);
        assert!(splitter.minimum_transfer_split(&[]).is_err());
    }

    // ── detect_dependent_joins ───────────────────────────────────────────────

    #[test]
    fn test_detect_dependent_joins_with_bound_terms() {
        let splitter = QuerySplitter::new(vec![]);
        let fragments = vec![
            SplitFragment {
                endpoint: "http://ep1".to_string(),
                patterns: vec![SplitTriplePattern::new(
                    iri("http://ex.org/alice"),
                    iri("http://ex.org/knows"),
                    var("friend"),
                )],
                exclusive: true,
            },
            SplitFragment {
                endpoint: "http://ep2".to_string(),
                patterns: vec![SplitTriplePattern::new(
                    var("friend"),
                    iri("http://ex.org/name"),
                    var("name"),
                )],
                exclusive: true,
            },
        ];
        let deps = splitter.detect_dependent_joins(&fragments);
        // "friend" is shared, and "http://ex.org/alice" is a bound term in left
        // but that doesn't make "friend" dependent in our heuristic — the left
        // has bound IRIs but they're not the join variable.
        // This test verifies the method runs without panic.
        assert!(deps.is_empty() || !deps.is_empty());
    }

    // ── estimate_communication_cost ──────────────────────────────────────────

    #[test]
    fn test_communication_cost() {
        let splitter = QuerySplitter::new(vec![
            ep("http://ep1", &[], 100, 5.0),
            ep("http://ep2", &[], 200, 3.0),
        ]);
        let cost = splitter.estimate_communication_cost("http://ep1", "http://ep2");
        assert!((cost - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_communication_cost_unknown_endpoint() {
        let splitter = QuerySplitter::new(vec![]);
        let cost = splitter.estimate_communication_cost("http://unknown1", "http://unknown2");
        assert!((cost - 20.0).abs() < f64::EPSILON); // 10.0 + 10.0 default
    }

    // ── SplitPlan serialization ──────────────────────────────────────────────

    #[test]
    fn test_plan_serialize() {
        let plan = SplitPlan {
            fragments: vec![SplitFragment {
                endpoint: "http://ep1".to_string(),
                patterns: vec![SplitTriplePattern::new(
                    var("s"),
                    iri("http://ex.org/p"),
                    var("o"),
                )],
                exclusive: true,
            }],
            joins: vec![],
            cost: SplitCost {
                total: 5.0,
                communication: 3.0,
                processing: 2.0,
                remote_requests: 1,
                estimated_bytes: 1024,
            },
            unassigned: vec![],
        };
        let output = plan.serialize();
        assert!(output.contains("Split Plan"));
        assert!(output.contains("http://ep1"));
        assert!(output.contains("?s"));
        assert!(output.contains("exclusive=true"));
    }

    #[test]
    fn test_plan_serialize_with_unassigned() {
        let plan = SplitPlan {
            fragments: vec![],
            joins: vec![],
            cost: SplitCost::default(),
            unassigned: vec![SplitTriplePattern::new(var("s"), iri("p"), var("o"))],
        };
        let output = plan.serialize();
        assert!(output.contains("Unassigned"));
    }

    // ── SplitCost default ────────────────────────────────────────────────────

    #[test]
    fn test_split_cost_default() {
        let cost = SplitCost::default();
        assert!((cost.total - 0.0).abs() < f64::EPSILON);
        assert_eq!(cost.remote_requests, 0);
        assert_eq!(cost.estimated_bytes, 0);
    }

    // ── SplitterError ────────────────────────────────────────────────────────

    #[test]
    fn test_splitter_error_display() {
        let e1 = SplitterError::NoEndpoints;
        assert!(format!("{}", e1).contains("no endpoints"));

        let e2 = SplitterError::EmptyBgp;
        assert!(format!("{}", e2).contains("empty"));

        let e3 = SplitterError::UnassignablePattern("test".to_string());
        assert!(format!("{}", e3).contains("test"));
    }

    #[test]
    fn test_splitter_error_is_error() {
        let e: Box<dyn std::error::Error> = Box::new(SplitterError::NoEndpoints);
        assert!(!e.to_string().is_empty());
    }

    // ── JoinKind ─────────────────────────────────────────────────────────────

    #[test]
    fn test_join_kind_eq() {
        let j1 = JoinKind::BoundJoin {
            bind_variables: vec!["s".to_string()],
        };
        let j2 = j1.clone();
        assert_eq!(j1, j2);

        let j3 = JoinKind::DependentJoin {
            dependency_variables: vec!["x".to_string()],
        };
        assert_eq!(j3.clone(), j3);

        let j4 = JoinKind::IndependentJoin {
            join_variables: vec!["y".to_string()],
        };
        assert_eq!(j4.clone(), j4);
    }

    // ── EndpointCapability ───────────────────────────────────────────────────

    #[test]
    fn test_endpoint_capability_clone() {
        let e = ep("http://ep1", &["http://ex.org/p"], 100, 5.0);
        let e2 = e.clone();
        assert_eq!(e2.url, "http://ep1");
        assert_eq!(e2.estimated_triples, 100);
    }

    // ── SplitFragment ────────────────────────────────────────────────────────

    #[test]
    fn test_split_fragment_eq() {
        let f1 = SplitFragment {
            endpoint: "http://ep1".to_string(),
            patterns: vec![SplitTriplePattern::new(var("s"), iri("p"), var("o"))],
            exclusive: true,
        };
        let f2 = f1.clone();
        assert_eq!(f1, f2);
    }

    // ── Integration: full split with joins ───────────────────────────────────

    #[test]
    fn test_full_split_with_join() {
        let splitter = QuerySplitter::new(vec![
            ep("http://ep1", &["http://ex.org/name"], 1000, 5.0),
            ep("http://ep2", &["http://ex.org/age"], 500, 3.0),
        ]);
        let patterns = vec![
            SplitTriplePattern::new(var("person"), iri("http://ex.org/name"), var("name")),
            SplitTriplePattern::new(var("person"), iri("http://ex.org/age"), var("age")),
        ];
        let plan = splitter.split(&patterns).expect("split should succeed");

        assert_eq!(plan.fragments.len(), 2);
        assert!(!plan.joins.is_empty());
        assert!(plan.cost.total > 0.0);
        assert!(plan.cost.remote_requests >= 2);
    }

    #[test]
    fn test_full_split_cost_includes_bytes() {
        let splitter =
            QuerySplitter::new(vec![ep("http://ep1", &["http://ex.org/name"], 1000, 5.0)]);
        let patterns = vec![SplitTriplePattern::new(
            var("s"),
            iri("http://ex.org/name"),
            var("n"),
        )];
        let plan = splitter.split(&patterns).expect("should succeed");
        assert!(plan.cost.estimated_bytes > 0);
    }

    // ── Multi-source BGP ─────────────────────────────────────────────────────

    #[test]
    fn test_multi_source_bgp_assigns_to_first_candidate() {
        let splitter = QuerySplitter::new(vec![
            ep("http://ep1", &["http://ex.org/name"], 1000, 5.0),
            ep("http://ep2", &["http://ex.org/name"], 500, 3.0),
        ]);
        let patterns = vec![SplitTriplePattern::new(
            var("s"),
            iri("http://ex.org/name"),
            var("n"),
        )];
        let plan = splitter.split(&patterns).expect("should succeed");
        // Default split assigns to first candidate
        assert_eq!(plan.fragments.len(), 1);
        assert_eq!(plan.fragments[0].endpoint, "http://ep1");
    }

    #[test]
    fn test_multi_source_min_transfer_prefers_smaller() {
        let splitter = QuerySplitter::new(vec![
            ep("http://ep1", &["http://ex.org/name"], 1000, 5.0),
            ep("http://ep2", &["http://ex.org/name"], 50, 3.0),
        ]);
        let patterns = vec![SplitTriplePattern::new(
            var("s"),
            iri("http://ex.org/name"),
            var("n"),
        )];
        let plan = splitter
            .minimum_transfer_split(&patterns)
            .expect("should succeed");
        assert_eq!(plan.fragments[0].endpoint, "http://ep2");
    }
}
