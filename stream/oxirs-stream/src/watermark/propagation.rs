//! Watermark propagation across operator topologies.
//!
//! A streaming pipeline is a directed acyclic graph of operators.  Watermarks
//! flow from sources through operators to sinks.  At each operator, the
//! *output* watermark is the **minimum** of all *input* watermarks — this is
//! the contract that lets downstream operators close windows safely.
//!
//! This module provides:
//!
//! * [`OperatorId`] — typed wrapper around an operator name.
//! * [`OperatorWatermarkAggregator`] — per-operator aggregator that
//!   combines multiple input watermarks into one output watermark using the
//!   minimum rule.  Built on top of [`super::WatermarkAligner`].
//! * [`WatermarkPropagator`] — operator-graph-wide propagator that drives
//!   updates from upstream sources to downstream sinks while enforcing
//!   monotonicity per output edge.  Returns
//!   [`StreamError::WatermarkViolation`] if any operator emits a non-monotonic
//!   watermark.
//!
//! Contract
//!
//! 1. **Per-operator output non-decreasing**: each operator's emitted
//!    watermark must be ≥ its previously emitted watermark.
//! 2. **Aggregation is min**: an operator with N upstream sources sees the
//!    minimum of the N inputs as its output watermark.
//! 3. **Cycles are not supported**: the topology must be acyclic.
//!
//! Returns
//!
//! * `Ok(Some(global_watermark))` after a successful update.
//! * `Ok(None)` if no operator output watermark could yet be determined
//!   (e.g. a downstream operator without all of its upstream inputs reporting).
//! * `Err(StreamError::WatermarkViolation { … })` on monotonicity violation.

use std::collections::HashMap;

use crate::error::StreamError;

use super::WatermarkAligner;

// ─── OperatorId ──────────────────────────────────────────────────────────────

/// Unique identifier of an operator within a topology.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OperatorId(pub String);

impl OperatorId {
    /// Construct a new identifier.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Borrow the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<S: Into<String>> From<S> for OperatorId {
    fn from(s: S) -> Self {
        OperatorId(s.into())
    }
}

// ─── Operator-level aggregator ───────────────────────────────────────────────

/// Aggregates per-input watermarks for a single operator.
///
/// Each operator wraps a [`WatermarkAligner`] keyed by upstream operator id.
/// The output watermark is the global minimum across all inputs and is
/// monotonically non-decreasing.
pub struct OperatorWatermarkAggregator {
    operator_id: OperatorId,
    inputs: WatermarkAligner,
    last_emitted: Option<i64>,
}

impl OperatorWatermarkAggregator {
    /// Create a new aggregator for `operator_id`.
    pub fn new(operator_id: impl Into<OperatorId>) -> Self {
        Self {
            operator_id: operator_id.into(),
            inputs: WatermarkAligner::new(),
            last_emitted: None,
        }
    }

    /// Update the watermark received from `input` upstream.
    ///
    /// Returns the new operator output watermark if all inputs have reported
    /// at least once.  If any input is missing returns `None`.
    pub fn observe(
        &mut self,
        input: &OperatorId,
        watermark_ms: i64,
        expected_inputs: usize,
    ) -> Result<Option<i64>, StreamError> {
        self.inputs.update(input.as_str(), watermark_ms);

        if self.inputs.source_count() < expected_inputs {
            return Ok(None);
        }

        let candidate = self.inputs.global_watermark();
        if let Some(prev) = self.last_emitted {
            if candidate < prev {
                return Err(StreamError::WatermarkViolation {
                    operator_id: self.operator_id.0.clone(),
                    reason: format!("candidate output watermark {candidate} < previous {prev}"),
                });
            }
        }
        self.last_emitted = Some(candidate);
        Ok(Some(candidate))
    }

    /// Most recently emitted output watermark, if any.
    pub fn last_emitted(&self) -> Option<i64> {
        self.last_emitted
    }

    /// Operator identifier.
    pub fn operator_id(&self) -> &OperatorId {
        &self.operator_id
    }
}

// ─── Topology-level propagator ───────────────────────────────────────────────

/// Topology-wide watermark propagator.
///
/// The topology is described by `add_edge(upstream, downstream)`.  Source
/// watermarks are pushed via [`WatermarkPropagator::push_source`]; the
/// propagator then fans them through the topology and returns the resulting
/// global watermark — the minimum across all sinks.
pub struct WatermarkPropagator {
    // upstream → downstream edges
    downstream_of: HashMap<OperatorId, Vec<OperatorId>>,
    // downstream → upstream edges (so we can compute expected_inputs)
    upstream_of: HashMap<OperatorId, Vec<OperatorId>>,
    // every operator known to the topology
    operators: HashMap<OperatorId, OperatorWatermarkAggregator>,
    // sinks (no downstream).
    sinks: Vec<OperatorId>,
}

impl Default for WatermarkPropagator {
    fn default() -> Self {
        Self::new()
    }
}

impl WatermarkPropagator {
    /// Create an empty propagator.
    pub fn new() -> Self {
        Self {
            downstream_of: HashMap::new(),
            upstream_of: HashMap::new(),
            operators: HashMap::new(),
            sinks: Vec::new(),
        }
    }

    /// Register an operator.  Idempotent.
    pub fn add_operator(&mut self, operator: OperatorId) {
        self.operators
            .entry(operator.clone())
            .or_insert_with(|| OperatorWatermarkAggregator::new(operator));
    }

    /// Register a directed edge `upstream → downstream`.
    ///
    /// Both endpoints are auto-added if not already present.
    pub fn add_edge(&mut self, upstream: OperatorId, downstream: OperatorId) {
        self.add_operator(upstream.clone());
        self.add_operator(downstream.clone());
        self.downstream_of
            .entry(upstream.clone())
            .or_default()
            .push(downstream.clone());
        self.upstream_of
            .entry(downstream)
            .or_default()
            .push(upstream);
        self.recompute_sinks();
    }

    fn recompute_sinks(&mut self) {
        self.sinks = self
            .operators
            .keys()
            .filter(|op| !self.downstream_of.contains_key(*op))
            .cloned()
            .collect();
    }

    /// Push a source watermark.  The source is treated as having a single
    /// virtual upstream (itself) so the aggregator's monotonicity logic
    /// applies directly.
    ///
    /// Returns the new global watermark (minimum across all sinks), or
    /// `Ok(None)` if any sink is still missing inputs.
    pub fn push_source(
        &mut self,
        source: &OperatorId,
        watermark_ms: i64,
    ) -> Result<Option<i64>, StreamError> {
        // Treat the source as both the operator and its own only upstream.
        if !self.operators.contains_key(source) {
            self.add_operator(source.clone());
        }
        let agg = self
            .operators
            .get_mut(source)
            .expect("source aggregator just added");
        agg.observe(source, watermark_ms, 1)?;

        // BFS down the graph, propagating watermarks.
        let mut frontier: Vec<OperatorId> =
            self.downstream_of.get(source).cloned().unwrap_or_default();

        while let Some(op) = frontier.pop() {
            // Compute the new candidate watermark for `op` from all its
            // upstreams' last_emitted.  If any upstream hasn't emitted yet,
            // skip — `op` cannot move yet.
            let upstreams: Vec<OperatorId> = self.upstream_of.get(&op).cloned().unwrap_or_default();
            if upstreams.is_empty() {
                continue;
            }

            // Collect (upstream_id, watermark_value) snapshots first so we
            // don't overlap a mutable borrow on `self.operators`.
            let mut readings: Vec<(OperatorId, i64)> = Vec::with_capacity(upstreams.len());
            let mut all_ready = true;
            for u in &upstreams {
                match self.operators.get(u).and_then(|a| a.last_emitted()) {
                    Some(v) => readings.push((u.clone(), v)),
                    None => {
                        all_ready = false;
                        break;
                    }
                }
            }
            if !all_ready {
                continue;
            }

            // Feed each upstream's current watermark into `op`'s aggregator.
            let n_inputs = upstreams.len();
            for (u, wm) in &readings {
                let agg = self
                    .operators
                    .get_mut(&op)
                    .expect("operator known to topology");
                agg.observe(u, *wm, n_inputs)?;
            }

            // Schedule descendants.
            if let Some(ds) = self.downstream_of.get(&op) {
                frontier.extend(ds.iter().cloned());
            }
        }

        Ok(self.global_watermark())
    }

    /// The global watermark: the minimum across all sinks' last_emitted.
    /// `None` if any sink hasn't emitted yet.
    pub fn global_watermark(&self) -> Option<i64> {
        if self.sinks.is_empty() {
            // No sinks ⇒ global = min across all operators that have emitted.
            let mut min_v: Option<i64> = None;
            for (_, agg) in self.operators.iter() {
                let v = agg.last_emitted()?;
                min_v = Some(min_v.map(|m| m.min(v)).unwrap_or(v));
            }
            return min_v;
        }
        let mut min_v: Option<i64> = None;
        for sink in &self.sinks {
            let v = self.operators.get(sink).and_then(|a| a.last_emitted())?;
            min_v = Some(min_v.map(|m| m.min(v)).unwrap_or(v));
        }
        min_v
    }

    /// Return the last-emitted output watermark for `op`, if any.
    pub fn watermark_of(&self, op: &OperatorId) -> Option<i64> {
        self.operators.get(op).and_then(|a| a.last_emitted())
    }

    /// Number of operators in the topology.
    pub fn operator_count(&self) -> usize {
        self.operators.len()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn op(name: &str) -> OperatorId {
        OperatorId::new(name)
    }

    // ── OperatorWatermarkAggregator ──────────────────────────────────────────

    #[test]
    fn aggregator_emits_min_across_inputs() {
        let mut agg = OperatorWatermarkAggregator::new("merge");
        let a = op("a");
        let b = op("b");
        // Only one input reported → not enough yet.
        assert_eq!(agg.observe(&a, 1_000, 2).unwrap(), None);
        // Both inputs reported → emit min.
        let out = agg.observe(&b, 800, 2).unwrap();
        assert_eq!(out, Some(800));
    }

    #[test]
    fn aggregator_is_monotonic() {
        let mut agg = OperatorWatermarkAggregator::new("op");
        let a = op("a");
        agg.observe(&a, 5_000, 1).unwrap();
        // Lower watermark must violate monotonicity.
        let err = agg.observe(&a, 4_000, 1).expect_err("monotonic");
        match err {
            StreamError::WatermarkViolation { operator_id, .. } => {
                assert_eq!(operator_id, "op");
            }
            other => panic!("expected WatermarkViolation, got {other:?}"),
        }
    }

    #[test]
    fn aggregator_equal_watermark_is_ok() {
        let mut agg = OperatorWatermarkAggregator::new("op");
        let a = op("a");
        agg.observe(&a, 1_000, 1).unwrap();
        // Equal is allowed.
        assert_eq!(agg.observe(&a, 1_000, 1).unwrap(), Some(1_000));
    }

    // ── WatermarkPropagator ──────────────────────────────────────────────────

    #[test]
    fn propagator_single_source_to_single_sink() {
        let mut p = WatermarkPropagator::new();
        let s = op("source");
        let snk = op("sink");
        p.add_edge(s.clone(), snk.clone());

        let g = p.push_source(&s, 1_000).unwrap();
        assert_eq!(g, Some(1_000));
        assert_eq!(p.watermark_of(&s), Some(1_000));
        assert_eq!(p.watermark_of(&snk), Some(1_000));
    }

    #[test]
    fn propagator_two_sources_take_min() {
        // src_a → join, src_b → join, join → sink
        let mut p = WatermarkPropagator::new();
        let a = op("a");
        let b = op("b");
        let j = op("j");
        let s = op("sink");
        p.add_edge(a.clone(), j.clone());
        p.add_edge(b.clone(), j.clone());
        p.add_edge(j.clone(), s.clone());

        // First source reports 1000.
        let g = p.push_source(&a, 1_000).unwrap();
        // join cannot move yet (b not reported).
        assert_eq!(p.watermark_of(&j), None);
        assert_eq!(g, None);

        // Second source reports 700.
        let g = p.push_source(&b, 700).unwrap();
        // join takes min(1000, 700) = 700.
        assert_eq!(p.watermark_of(&j), Some(700));
        assert_eq!(p.watermark_of(&s), Some(700));
        assert_eq!(g, Some(700));
    }

    #[test]
    fn propagator_global_watermark_is_min_across_sinks() {
        // src → sink_a (sink_a is one sink)
        // src → sink_b (sink_b is another sink)
        let mut p = WatermarkPropagator::new();
        let s = op("src");
        let sa = op("sa");
        let sb = op("sb");
        p.add_edge(s.clone(), sa.clone());
        p.add_edge(s.clone(), sb.clone());

        let g = p.push_source(&s, 5_000).unwrap();
        assert_eq!(g, Some(5_000));
        assert_eq!(p.watermark_of(&sa), Some(5_000));
        assert_eq!(p.watermark_of(&sb), Some(5_000));
    }

    #[test]
    fn propagator_is_monotonic_across_topology() {
        let mut p = WatermarkPropagator::new();
        let s = op("src");
        let snk = op("snk");
        p.add_edge(s.clone(), snk.clone());
        p.push_source(&s, 1_000).unwrap();
        // Decrease must fail.
        let err = p.push_source(&s, 500).expect_err("monotonic");
        assert!(matches!(err, StreamError::WatermarkViolation { .. }));
    }

    #[test]
    fn propagator_no_topology_treats_each_op_as_sink() {
        let mut p = WatermarkPropagator::new();
        let s = op("solo");
        // No edges; "solo" is its own sink.
        let g = p.push_source(&s, 42).unwrap();
        assert_eq!(g, Some(42));
        assert_eq!(p.operator_count(), 1);
    }
}
