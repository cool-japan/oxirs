//! Render a physics state vector to RDF triples.
//!
//! The writer takes a [`PhysicsState`] (a snapshot of the simulator at one
//! step) and emits a SPARQL `INSERT DATA` block that materialises the state
//! into an RDF graph. The same machinery is used to compute incremental
//! "diff-only" updates against a previous snapshot so that only changed
//! properties are written downstream.
//!
//! The rendering is deterministic: triples are emitted in lexicographic key
//! order so two equal states always produce byte-identical output.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Possible value kinds that a physics state property can take.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhysicsStateValue {
    /// 64-bit floating point scalar.
    Scalar(f64),
    /// Vector of `f64`.
    Vector(Vec<f64>),
    /// Boolean.
    Bool(bool),
    /// Plain UTF-8 string (e.g. an enumerated state label).
    Text(String),
    /// Integer.
    Integer(i64),
}

impl PhysicsStateValue {
    /// Format the value as a SPARQL literal suitable for `INSERT DATA`.
    pub fn to_sparql_literal(&self) -> String {
        match self {
            Self::Scalar(v) => format!("\"{v}\"^^xsd:double"),
            Self::Vector(v) => {
                let parts: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
                format!("\"{}\"^^xsd:string", parts.join(","))
            }
            Self::Bool(b) => format!("\"{b}\"^^xsd:boolean"),
            Self::Text(s) => {
                let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
                format!("\"{escaped}\"^^xsd:string")
            }
            Self::Integer(i) => format!("\"{i}\"^^xsd:integer"),
        }
    }

    /// Compare two values for "approximate equality" — used by [`state_diff`]
    /// to suppress noise for floating-point fluctuations beneath the
    /// configured tolerance.
    pub fn approx_eq(&self, other: &Self, tol: f64) -> bool {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => approx_eq_scalar(*a, *b, tol),
            (Self::Vector(a), Self::Vector(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| approx_eq_scalar(*x, *y, tol))
            }
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Integer(a), Self::Integer(b)) => a == b,
            _ => false,
        }
    }
}

fn approx_eq_scalar(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a == b {
        return true;
    }
    (a - b).abs() <= tol
}

/// A snapshot of the simulator state at a single step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicsState {
    /// The entity URN this state belongs to.
    pub entity_iri: String,
    /// Simulation step counter (monotonically increasing).
    pub step: u64,
    /// Wall-clock time the snapshot was taken.
    pub timestamp: DateTime<Utc>,
    /// Property name → value map. Use a deterministic key set so that
    /// rendering and diffing are reproducible.
    pub values: HashMap<String, PhysicsStateValue>,
}

impl PhysicsState {
    /// Create an empty state for `entity_iri` at `step = 0`.
    pub fn new(entity_iri: impl Into<String>) -> Self {
        Self {
            entity_iri: entity_iri.into(),
            step: 0,
            timestamp: Utc::now(),
            values: HashMap::new(),
        }
    }

    /// Insert or overwrite a property value, returning `&mut self` for
    /// chaining.
    pub fn with(mut self, key: impl Into<String>, value: PhysicsStateValue) -> Self {
        self.values.insert(key.into(), value);
        self
    }

    /// Set a scalar property in place.
    pub fn set_scalar(&mut self, key: impl Into<String>, value: f64) {
        self.values
            .insert(key.into(), PhysicsStateValue::Scalar(value));
    }
}

/// Difference between two physics states.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StateDiff {
    /// Properties present in the new state but not the old one.
    pub added: BTreeMap<String, PhysicsStateValue>,
    /// Properties whose value changed (above the configured tolerance).
    pub changed: BTreeMap<String, (PhysicsStateValue, PhysicsStateValue)>,
    /// Properties present in the old state but removed in the new one.
    pub removed: BTreeMap<String, PhysicsStateValue>,
}

impl StateDiff {
    /// Total number of changes recorded.
    pub fn total_changes(&self) -> usize {
        self.added.len() + self.changed.len() + self.removed.len()
    }

    /// Returns `true` when the diff is empty (states are equivalent).
    pub fn is_empty(&self) -> bool {
        self.total_changes() == 0
    }
}

/// Compute the diff between `old` and `new` state values using `tol` for
/// floating-point comparison.
///
/// Properties whose value is approximately equal are excluded so that
/// numerical noise does not produce spurious updates.
pub fn state_diff(old: &PhysicsState, new: &PhysicsState, tol: f64) -> StateDiff {
    let mut diff = StateDiff::default();
    for (k, v_new) in &new.values {
        match old.values.get(k) {
            None => {
                diff.added.insert(k.clone(), v_new.clone());
            }
            Some(v_old) => {
                if !v_old.approx_eq(v_new, tol) {
                    diff.changed
                        .insert(k.clone(), (v_old.clone(), v_new.clone()));
                }
            }
        }
    }
    for (k, v_old) in &old.values {
        if !new.values.contains_key(k) {
            diff.removed.insert(k.clone(), v_old.clone());
        }
    }
    diff
}

/// Configuration for the state-to-RDF writer.
#[derive(Debug, Clone)]
pub struct StateGraphConfig {
    /// Physics namespace prefix for predicates.
    pub physics_prefix: String,
    /// Named-graph IRI under which state triples are stored. When `None`
    /// the writer emits triples into the default graph.
    pub named_graph: Option<String>,
    /// Tolerance used when comparing scalar / vector values for
    /// "approximate equality" during diffs.
    pub tolerance: f64,
}

impl Default for StateGraphConfig {
    fn default() -> Self {
        Self {
            physics_prefix: "http://oxirs.org/physics#".to_string(),
            named_graph: Some("http://oxirs.org/physics/state".to_string()),
            tolerance: 1e-9,
        }
    }
}

/// Renders a [`PhysicsState`] (or a [`StateDiff`]) as RDF using SPARQL
/// `INSERT DATA` blocks.
#[derive(Debug, Clone)]
pub struct StateToRdfWriter {
    config: StateGraphConfig,
}

impl StateToRdfWriter {
    /// Build a writer with default configuration.
    pub fn new() -> Self {
        Self {
            config: StateGraphConfig::default(),
        }
    }

    /// Build a writer with a custom configuration.
    pub fn with_config(config: StateGraphConfig) -> Self {
        Self { config }
    }

    /// Currently configured namespace prefix.
    pub fn physics_prefix(&self) -> &str {
        &self.config.physics_prefix
    }

    /// Render the full state — every property in `state` is emitted.
    pub fn render_full(&self, state: &PhysicsState) -> String {
        let mut triples = Vec::new();
        let state_node = self.state_iri(state);

        triples.push(format!("<{state_node}> a phys:State ."));
        triples.push(format!(
            "<{state_node}> phys:simulatesEntity <{}> .",
            state.entity_iri
        ));
        triples.push(format!(
            "<{state_node}> phys:step \"{}\"^^xsd:integer .",
            state.step
        ));
        triples.push(format!(
            "<{state_node}> phys:timestamp \"{}\"^^xsd:dateTime .",
            state.timestamp.to_rfc3339()
        ));

        let ordered: BTreeMap<&String, &PhysicsStateValue> = state.values.iter().collect();
        for (key, value) in ordered {
            triples.push(format!(
                "<{state_node}> phys:{key} {} .",
                value.to_sparql_literal()
            ));
        }

        self.wrap(&triples)
    }

    /// Render only the *changes* between `previous` and `current`. Returns
    /// `None` when the diff is empty so that callers can short-circuit and
    /// avoid issuing a no-op update.
    pub fn render_diff(&self, previous: &PhysicsState, current: &PhysicsState) -> Option<String> {
        let diff = state_diff(previous, current, self.config.tolerance);
        if diff.is_empty() {
            return None;
        }
        let state_node = self.state_iri(current);

        let mut delete_triples = Vec::new();
        let mut insert_triples = Vec::new();
        for (k, old_v) in &diff.removed {
            delete_triples.push(format!(
                "<{state_node}> phys:{k} {} .",
                old_v.to_sparql_literal()
            ));
        }
        for (k, (old_v, _new_v)) in &diff.changed {
            delete_triples.push(format!(
                "<{state_node}> phys:{k} {} .",
                old_v.to_sparql_literal()
            ));
        }
        for (k, v_new) in &diff.added {
            insert_triples.push(format!(
                "<{state_node}> phys:{k} {} .",
                v_new.to_sparql_literal()
            ));
        }
        for (k, (_old_v, new_v)) in &diff.changed {
            insert_triples.push(format!(
                "<{state_node}> phys:{k} {} .",
                new_v.to_sparql_literal()
            ));
        }

        Some(self.wrap_modify(&delete_triples, &insert_triples))
    }

    /// Compute a stable IRI for the given state snapshot.
    pub fn state_iri(&self, state: &PhysicsState) -> String {
        format!(
            "{}state/{}/{}",
            self.config.physics_prefix,
            sanitize(&state.entity_iri),
            state.step
        )
    }

    fn wrap(&self, triples: &[String]) -> String {
        let body = triples.join("\n        ");
        let inner = match &self.config.named_graph {
            Some(g) => format!("GRAPH <{g}> {{\n        {body}\n    }}"),
            None => body,
        };
        format!(
            "PREFIX phys: <{phys}>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n\nINSERT DATA {{\n    {inner}\n}}\n",
            phys = self.config.physics_prefix
        )
    }

    fn wrap_modify(&self, delete_triples: &[String], insert_triples: &[String]) -> String {
        let delete_body = delete_triples.join("\n        ");
        let insert_body = insert_triples.join("\n        ");
        let (delete_inner, insert_inner) = match &self.config.named_graph {
            Some(g) => (
                format!("GRAPH <{g}> {{\n        {delete_body}\n    }}"),
                format!("GRAPH <{g}> {{\n        {insert_body}\n    }}"),
            ),
            None => (delete_body.clone(), insert_body.clone()),
        };
        let delete_block = if delete_triples.is_empty() {
            String::new()
        } else {
            format!("DELETE DATA {{\n    {delete_inner}\n}};\n")
        };
        let insert_block = if insert_triples.is_empty() {
            String::new()
        } else {
            format!("INSERT DATA {{\n    {insert_inner}\n}}\n")
        };
        format!(
            "PREFIX phys: <{phys}>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n\n{delete_block}{insert_block}",
            phys = self.config.physics_prefix
        )
    }
}

impl Default for StateToRdfWriter {
    fn default() -> Self {
        Self::new()
    }
}

fn sanitize(iri: &str) -> String {
    iri.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> PhysicsState {
        let mut s = PhysicsState::new("urn:example:battery:001");
        s.step = 12;
        s.set_scalar("temperature", 298.15);
        s.set_scalar("voltage", 3.71);
        s.values
            .insert("is_charging".to_string(), PhysicsStateValue::Bool(true));
        s.values.insert(
            "label".to_string(),
            PhysicsStateValue::Text("running".to_string()),
        );
        s
    }

    #[test]
    fn full_render_contains_core_predicates() {
        let writer = StateToRdfWriter::new();
        let s = sample_state();
        let out = writer.render_full(&s);
        assert!(out.contains("phys:State"));
        assert!(out.contains("phys:simulatesEntity"));
        assert!(out.contains("urn:example:battery:001"));
        assert!(out.contains("phys:step"));
        assert!(out.contains("phys:temperature"));
        assert!(out.contains("phys:voltage"));
        assert!(out.contains("phys:is_charging"));
        assert!(out.contains("phys:label"));
    }

    #[test]
    fn full_render_is_deterministic() {
        let writer = StateToRdfWriter::new();
        let s1 = sample_state();
        let s2 = sample_state();
        let r1 = writer.render_full(&s1);
        let r2 = writer.render_full(&s2);
        // The state contains the same keys in arbitrary order but rendering
        // must be identical because the writer sorts keys.
        assert_eq!(
            r1.replace(s1.timestamp.to_rfc3339().as_str(), "<TS>"),
            r2.replace(s2.timestamp.to_rfc3339().as_str(), "<TS>")
        );
    }

    #[test]
    fn diff_empty_returns_none() {
        let writer = StateToRdfWriter::new();
        let prev = sample_state();
        let curr = prev.clone();
        assert!(writer.render_diff(&prev, &curr).is_none());
    }

    #[test]
    fn diff_only_changed_emitted() {
        let writer = StateToRdfWriter::new();
        let mut prev = sample_state();
        let mut curr = prev.clone();
        // change voltage but leave temperature untouched
        curr.set_scalar("voltage", 3.95);
        prev.set_scalar("voltage", 3.71);
        let q = writer
            .render_diff(&prev, &curr)
            .expect("diff should be non-empty");
        assert!(q.contains("phys:voltage"));
        // changed property must appear in DELETE+INSERT, untouched must not
        assert!(!q.contains("phys:temperature"));
    }

    #[test]
    fn diff_added_property_emitted() {
        let writer = StateToRdfWriter::new();
        let prev = PhysicsState::new("urn:example:e1");
        let mut curr = PhysicsState::new("urn:example:e1");
        curr.set_scalar("pressure", 101_325.0);
        let q = writer.render_diff(&prev, &curr).expect("non-empty");
        assert!(q.contains("phys:pressure"));
        assert!(q.contains("INSERT DATA"));
        assert!(!q.contains("DELETE DATA"));
    }

    #[test]
    fn diff_removed_property_emitted_as_delete() {
        let writer = StateToRdfWriter::new();
        let mut prev = PhysicsState::new("urn:example:e1");
        prev.set_scalar("pressure", 101_325.0);
        let curr = PhysicsState::new("urn:example:e1");
        let q = writer.render_diff(&prev, &curr).expect("non-empty");
        assert!(q.contains("phys:pressure"));
        assert!(q.contains("DELETE DATA"));
    }

    #[test]
    fn approx_eq_below_tolerance_is_not_a_change() {
        let writer = StateToRdfWriter::new();
        let mut prev = PhysicsState::new("urn:example:e1");
        let mut curr = PhysicsState::new("urn:example:e1");
        prev.set_scalar("voltage", 3.700_000_000_001);
        curr.set_scalar("voltage", 3.700_000_000_002);
        // default tolerance 1e-9 → these are equal
        assert!(writer.render_diff(&prev, &curr).is_none());
    }

    #[test]
    fn vector_equality_uses_pointwise_tolerance() {
        let v1 = PhysicsStateValue::Vector(vec![1.0, 2.0, 3.0]);
        let v2 = PhysicsStateValue::Vector(vec![1.0 + 1e-12, 2.0, 3.0 - 1e-12]);
        assert!(v1.approx_eq(&v2, 1e-9));
        let v3 = PhysicsStateValue::Vector(vec![1.0, 2.0, 4.0]);
        assert!(!v1.approx_eq(&v3, 1e-9));
    }

    #[test]
    fn state_iri_is_sanitized() {
        let writer = StateToRdfWriter::new();
        let s = PhysicsState {
            entity_iri: "urn:example:e/1?x".to_string(),
            step: 0,
            timestamp: Utc::now(),
            values: HashMap::new(),
        };
        let iri = writer.state_iri(&s);
        // The sanitised entity portion must contain no colons or slashes —
        // even though the namespace prefix ("http://oxirs.org/...") does.
        let prefix = writer.physics_prefix();
        assert!(iri.starts_with(prefix));
        let suffix = &iri[prefix.len()..];
        assert!(!suffix.contains(':'));
        assert!(!suffix.contains('?'));
        assert!(suffix.contains("urn_example_e_1_x"));
    }

    #[test]
    fn integer_values_render_as_xsd_integer() {
        let v = PhysicsStateValue::Integer(42);
        let lit = v.to_sparql_literal();
        assert!(lit.contains("xsd:integer"));
        assert!(lit.contains("42"));
    }

    #[test]
    fn diff_total_changes_counts_all_kinds() {
        let mut prev = PhysicsState::new("urn:example:e");
        let mut curr = PhysicsState::new("urn:example:e");
        prev.set_scalar("a", 1.0);
        prev.set_scalar("b", 2.0);
        curr.set_scalar("a", 1.5); // changed
        curr.set_scalar("c", 3.0); // added
                                   // b is removed
        let d = state_diff(&prev, &curr, 1e-9);
        assert_eq!(d.added.len(), 1);
        assert_eq!(d.changed.len(), 1);
        assert_eq!(d.removed.len(), 1);
        assert_eq!(d.total_changes(), 3);
    }
}
