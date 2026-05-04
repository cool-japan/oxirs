//! Periodic bidirectional sync orchestrator.
//!
//! [`BidirectionalSync`] takes a [`PhysicsState`] producer (the simulator)
//! and an `RdfPropertyRow` consumer (the RDF graph) and drives a periodic
//! round-trip:
//!
//! 1. Snapshot the simulator state at step `t`.
//! 2. Emit a SPARQL `INSERT DATA` (or `DELETE/INSERT`) covering only the
//!    properties that changed since the last snapshot.
//! 3. Optionally pull external updates back from RDF and re-extract a
//!    [`PhysicsState`] for re-initialisation.
//!
//! The orchestrator never touches a live RDF store directly — callers
//! provide closures, which keeps the module easy to test and avoids
//! mandatory `RdfStore` dependencies.

use std::time::{Duration, Instant};

use crate::error::{PhysicsError, PhysicsResult};

use super::rdf_to_state::{RdfPropertyRow, RdfToStateExtractor, RdfToStateOutput};
use super::state_to_rdf::{
    state_diff, PhysicsState, StateDiff, StateGraphConfig, StateToRdfWriter,
};

/// Direction of a single sync invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncDirection {
    /// Push physics state → RDF graph.
    StateToRdf,
    /// Pull RDF graph → physics state.
    RdfToState,
    /// Skipped — sync interval has not elapsed yet.
    Skipped,
}

/// Configuration for the bidirectional orchestrator.
#[derive(Debug, Clone)]
pub struct BidirectionalSyncConfig {
    /// Minimum interval between sync passes. The orchestrator returns
    /// [`SyncDirection::Skipped`] when called before the interval elapses.
    pub min_interval: Duration,
    /// Whether the orchestrator should emit a *full* snapshot the first
    /// time it observes a state. Subsequent passes always use diffs.
    pub initial_full_snapshot: bool,
    /// State graph configuration used by the embedded writer.
    pub state_graph: StateGraphConfig,
}

impl Default for BidirectionalSyncConfig {
    fn default() -> Self {
        Self {
            min_interval: Duration::from_millis(100),
            initial_full_snapshot: true,
            state_graph: StateGraphConfig::default(),
        }
    }
}

/// Outcome of a single push/pull pass.
#[derive(Debug, Clone)]
pub struct BidirectionalSyncReport {
    /// Direction the orchestrator actually executed.
    pub direction: SyncDirection,
    /// Diff produced for the state-to-RDF leg (empty if direction was
    /// [`SyncDirection::Skipped`] or [`SyncDirection::RdfToState`]).
    pub diff: StateDiff,
    /// Optional re-extracted state from the RDF leg.
    pub re_extracted: Option<PhysicsState>,
    /// SPARQL query string emitted to the consumer (when applicable).
    pub sparql: Option<String>,
}

/// Periodic bidirectional sync orchestrator.
pub struct BidirectionalSync {
    config: BidirectionalSyncConfig,
    writer: StateToRdfWriter,
    extractor: RdfToStateExtractor,
    /// Last full snapshot we observed; updated after every push.
    last_snapshot: Option<PhysicsState>,
    /// Wall-clock time of the most recent push or pull.
    last_sync_at: Option<Instant>,
    /// Whether we have ever pushed an initial snapshot.
    has_pushed_initial: bool,
}

impl BidirectionalSync {
    /// Build a new orchestrator from `config`.
    pub fn new(config: BidirectionalSyncConfig) -> Self {
        let writer = StateToRdfWriter::with_config(config.state_graph.clone());
        Self {
            config,
            writer,
            extractor: RdfToStateExtractor::new(),
            last_snapshot: None,
            last_sync_at: None,
            has_pushed_initial: false,
        }
    }

    /// Convenience constructor with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BidirectionalSyncConfig::default())
    }

    /// Read-only access to the embedded writer (e.g. for inspecting the
    /// configured prefix in tests).
    pub fn writer(&self) -> &StateToRdfWriter {
        &self.writer
    }

    /// Returns `true` when the configured interval has elapsed since the
    /// last sync.
    pub fn ready(&self) -> bool {
        match (self.last_sync_at, self.config.min_interval) {
            (None, _) => true,
            (Some(t), interval) => t.elapsed() >= interval,
        }
    }

    /// Manually mark the orchestrator as just-synced (mainly for tests).
    pub fn touch(&mut self) {
        self.last_sync_at = Some(Instant::now());
    }

    /// Push the supplied `state` to RDF, returning the SPARQL query the
    /// caller should execute. The very first call returns a full snapshot
    /// (when `initial_full_snapshot` is true), subsequent calls return only
    /// diffs.
    ///
    /// Returns [`SyncDirection::Skipped`] when the configured interval has
    /// not yet elapsed.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::Internal`] when the supplied state is
    /// malformed (empty entity IRI).
    pub fn push_state(&mut self, state: &PhysicsState) -> PhysicsResult<BidirectionalSyncReport> {
        if state.entity_iri.is_empty() {
            return Err(PhysicsError::Internal(
                "PhysicsState.entity_iri must not be empty".to_string(),
            ));
        }
        if !self.ready() {
            return Ok(BidirectionalSyncReport {
                direction: SyncDirection::Skipped,
                diff: StateDiff::default(),
                re_extracted: None,
                sparql: None,
            });
        }

        let initial_due = self.config.initial_full_snapshot && !self.has_pushed_initial;
        let (sparql, diff) = if initial_due || self.last_snapshot.is_none() {
            let q = self.writer.render_full(state);
            self.has_pushed_initial = true;
            (Some(q), StateDiff::default())
        } else {
            let prev = match self.last_snapshot.as_ref() {
                Some(p) => p,
                None => {
                    // Defensive: should be unreachable thanks to the branch
                    // condition above.
                    return Err(PhysicsError::Internal(
                        "missing previous snapshot for diff".to_string(),
                    ));
                }
            };
            let d = state_diff(prev, state, self.config.state_graph.tolerance);
            let q = self.writer.render_diff(prev, state);
            (q, d)
        };

        // Update bookkeeping
        self.last_snapshot = Some(state.clone());
        self.last_sync_at = Some(Instant::now());

        Ok(BidirectionalSyncReport {
            direction: SyncDirection::StateToRdf,
            diff,
            re_extracted: None,
            sparql,
        })
    }

    /// Pull the most recent property rows for `entity_iri` at `step` from
    /// the supplied closure and re-extract a [`PhysicsState`].
    ///
    /// # Errors
    ///
    /// Bubbles up any error returned by `fetch_rows` and any failure from
    /// the inner extractor.
    pub fn pull_state<F>(
        &mut self,
        entity_iri: &str,
        step: u64,
        fetch_rows: F,
    ) -> PhysicsResult<BidirectionalSyncReport>
    where
        F: FnOnce(&str, u64) -> PhysicsResult<Vec<RdfPropertyRow>>,
    {
        if !self.ready() {
            return Ok(BidirectionalSyncReport {
                direction: SyncDirection::Skipped,
                diff: StateDiff::default(),
                re_extracted: None,
                sparql: None,
            });
        }
        let rows = fetch_rows(entity_iri, step)?;
        let RdfToStateOutput { state, skipped } =
            self.extractor.extract(entity_iri, step, &rows)?;
        if !skipped.is_empty() {
            tracing::debug!(
                "bidirectional sync skipped {} unparseable predicates: {:?}",
                skipped.len(),
                skipped
            );
        }
        let diff = match self.last_snapshot.as_ref() {
            Some(prev) => state_diff(prev, &state, self.config.state_graph.tolerance),
            None => StateDiff::default(),
        };

        self.last_snapshot = Some(state.clone());
        self.last_sync_at = Some(Instant::now());

        Ok(BidirectionalSyncReport {
            direction: SyncDirection::RdfToState,
            diff,
            re_extracted: Some(state),
            sparql: None,
        })
    }

    /// Reset internal state so the next push emits a full snapshot again.
    pub fn reset(&mut self) {
        self.last_snapshot = None;
        self.last_sync_at = None;
        self.has_pushed_initial = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    fn entity() -> &'static str {
        "urn:example:battery:001"
    }

    fn state_at(step: u64, voltage: f64, temperature: f64) -> PhysicsState {
        let mut s = PhysicsState::new(entity());
        s.step = step;
        s.set_scalar("voltage", voltage);
        s.set_scalar("temperature", temperature);
        s
    }

    #[test]
    fn first_push_emits_full_snapshot() {
        let mut sync = BidirectionalSync::with_defaults();
        let s = state_at(0, 3.7, 298.0);
        let report = sync.push_state(&s).expect("push should succeed");
        assert_eq!(report.direction, SyncDirection::StateToRdf);
        let q = report.sparql.expect("SPARQL must be produced");
        assert!(q.contains("INSERT DATA"));
        assert!(q.contains("phys:State"));
        assert!(q.contains(entity()));
    }

    #[test]
    fn second_push_emits_diff_only() {
        let mut sync = BidirectionalSync::new(BidirectionalSyncConfig {
            min_interval: Duration::from_millis(0),
            ..Default::default()
        });
        let s0 = state_at(0, 3.7, 298.0);
        let r0 = sync.push_state(&s0).expect("push should succeed");
        assert!(r0.diff.is_empty());

        let s1 = state_at(1, 3.95, 298.0); // only voltage changed
        sleep(Duration::from_millis(1));
        let r1 = sync.push_state(&s1).expect("second push should succeed");
        assert_eq!(r1.direction, SyncDirection::StateToRdf);
        assert_eq!(r1.diff.changed.len(), 1);
        assert!(r1.diff.changed.contains_key("voltage"));
        let q = r1.sparql.expect("non-empty diff must produce SPARQL");
        assert!(q.contains("phys:voltage"));
        assert!(!q.contains("phys:temperature"));
    }

    #[test]
    fn skipped_when_interval_not_elapsed() {
        let mut sync = BidirectionalSync::new(BidirectionalSyncConfig {
            min_interval: Duration::from_secs(10),
            ..Default::default()
        });
        let s0 = state_at(0, 3.7, 298.0);
        sync.push_state(&s0).expect("first push");
        // Push immediately again — orchestrator must skip.
        let r = sync.push_state(&s0).expect("second push");
        assert_eq!(r.direction, SyncDirection::Skipped);
        assert!(r.sparql.is_none());
    }

    #[test]
    fn empty_diff_yields_no_sparql() {
        let mut sync = BidirectionalSync::new(BidirectionalSyncConfig {
            min_interval: Duration::from_millis(0),
            ..Default::default()
        });
        let s = state_at(0, 3.7, 298.0);
        sync.push_state(&s).expect("first push");
        sleep(Duration::from_millis(1));
        let r = sync.push_state(&s).expect("second push");
        assert_eq!(r.direction, SyncDirection::StateToRdf);
        assert!(r.diff.is_empty());
        // No diff means no SPARQL.
        assert!(r.sparql.is_none());
    }

    #[test]
    fn pull_state_round_trips() {
        let mut sync = BidirectionalSync::new(BidirectionalSyncConfig {
            min_interval: Duration::from_millis(0),
            ..Default::default()
        });
        let result = sync.pull_state(entity(), 0, |_, _| {
            Ok(vec![
                RdfPropertyRow {
                    predicate: "voltage".to_string(),
                    literal: "3.71".to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#double".to_string()),
                },
                RdfPropertyRow {
                    predicate: "temperature".to_string(),
                    literal: "298.15".to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#double".to_string()),
                },
            ])
        });
        let report = result.expect("pull should succeed");
        assert_eq!(report.direction, SyncDirection::RdfToState);
        let s = report.re_extracted.expect("must produce a state");
        assert_eq!(s.entity_iri, entity());
        assert_eq!(s.values.len(), 2);
    }

    #[test]
    fn reset_clears_last_snapshot() {
        let mut sync = BidirectionalSync::new(BidirectionalSyncConfig {
            min_interval: Duration::from_millis(0),
            ..Default::default()
        });
        let s0 = state_at(0, 3.7, 298.0);
        sync.push_state(&s0).expect("first push");
        sync.reset();
        sleep(Duration::from_millis(1));
        let r = sync.push_state(&s0).expect("post-reset push");
        // After reset, the next push acts as a fresh initial snapshot.
        assert!(r.sparql.expect("sparql").contains("phys:State"));
    }

    #[test]
    fn empty_entity_iri_rejected() {
        let mut sync = BidirectionalSync::with_defaults();
        let mut s = PhysicsState::new("");
        s.set_scalar("voltage", 3.7);
        let r = sync.push_state(&s);
        assert!(matches!(r, Err(PhysicsError::Internal(_))));
    }
}
