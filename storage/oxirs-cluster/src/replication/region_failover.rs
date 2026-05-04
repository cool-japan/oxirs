//! Region failover state machine for active-active geo deployments.
//!
//! When a primary region becomes unreachable the controller demotes it,
//! promotes the next-best secondary from the configured tier, replays the
//! outstanding writes that the failed region had not yet shipped, and finally
//! re-admits the original region as a secondary once it recovers.
//!
//! The state machine is deliberately *event-driven* rather than tied to a
//! particular transport, so it can be unit-tested without spinning up an
//! actual cluster.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::active_active_geo::{ActiveActiveGeoConfig, RegionId};

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors emitted by the failover controller.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RegionFailoverError {
    /// The target region is not in the configured deployment.
    #[error("Region '{0}' is not part of the deployment")]
    UnknownRegion(RegionId),

    /// A demotion was requested but no eligible secondary could be promoted.
    #[error("No eligible secondary region available for promotion (failed='{failed}')")]
    NoSecondaryAvailable { failed: RegionId },

    /// A state transition was requested in an inconsistent state.
    #[error("Illegal transition from {from:?} for region '{region}'")]
    IllegalTransition { region: RegionId, from: RegionState },

    /// Lock contention or poisoning.
    #[error("Failover state lock poisoned: {0}")]
    LockPoisoned(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// State / Role
// ─────────────────────────────────────────────────────────────────────────────

/// Functional role a region plays at a given moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionRole {
    /// Region accepts local writes and serves as a fanout source.
    Primary,
    /// Region is hot-standby — receiving writes asynchronously, available
    /// for promotion.
    Secondary,
    /// Region was promoted from secondary in response to a failure of the
    /// previous primary.
    PromotedSecondary,
    /// Region was demoted due to suspected outage or partition.
    DemotedPrimary,
}

/// Health-driven state of a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionState {
    /// Region is healthy and reachable.
    Healthy,
    /// Region is suspected to be unreachable; failover may be in progress.
    Suspect,
    /// Region is confirmed offline.
    Failed,
    /// Region has been demoted but not yet been promoted to secondary
    /// (catch-up in progress).
    Recovering,
}

/// Single failover event recorded by the controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailoverEvent {
    /// A region was marked Suspect.
    Suspect { region: RegionId, at: Instant },
    /// A region was demoted (Primary → DemotedPrimary).
    Demoted { region: RegionId, at: Instant },
    /// A region was promoted (Secondary → PromotedSecondary).
    Promoted {
        region: RegionId,
        replaced: RegionId,
        at: Instant,
    },
    /// Outstanding writes were replayed to a recovered region.
    Replayed {
        region: RegionId,
        replayed: usize,
        at: Instant,
    },
    /// A region was readmitted as Secondary.
    Readmitted { region: RegionId, at: Instant },
}

// ─────────────────────────────────────────────────────────────────────────────
// Pending replay buffer
// ─────────────────────────────────────────────────────────────────────────────

/// A single outstanding write that the controller still needs to replay to
/// a recovering region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutstandingWrite {
    /// Region whose log holds the write.
    pub origin_region: RegionId,
    /// Sequence number assigned in the origin region's log.
    pub seq: u64,
    /// Key affected by the write.
    pub key: String,
    /// Opaque value/payload.
    pub value: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Controller
// ─────────────────────────────────────────────────────────────────────────────

/// Region failover controller.
///
/// Maintains the (state, role) pair for every region in the deployment plus
/// a small ring buffer of recent events for diagnostics.
#[derive(Debug)]
pub struct RegionFailoverController {
    inner: Arc<Mutex<FailoverInner>>,
    config: ActiveActiveGeoConfig,
    suspect_after: Duration,
    history_capacity: usize,
}

#[derive(Debug)]
struct FailoverInner {
    state: BTreeMap<RegionId, RegionState>,
    role: BTreeMap<RegionId, RegionRole>,
    last_seen: HashMap<RegionId, Instant>,
    pending_replay: HashMap<RegionId, VecDeque<OutstandingWrite>>,
    history: VecDeque<FailoverEvent>,
}

impl RegionFailoverController {
    /// Build a controller for the given configuration. All regions start
    /// `Healthy`; the local region is the only `Primary`, all others are
    /// `Secondary`.
    pub fn new(config: ActiveActiveGeoConfig) -> Self {
        Self::with_options(config, Duration::from_secs(15), 256)
    }

    /// Variant of [`Self::new`] that lets the caller override the suspect
    /// timeout (how long without a heartbeat before we promote a region to
    /// `Suspect`) and the in-memory event history capacity.
    pub fn with_options(
        config: ActiveActiveGeoConfig,
        suspect_after: Duration,
        history_capacity: usize,
    ) -> Self {
        let mut state_map = BTreeMap::new();
        let mut role_map = BTreeMap::new();
        let mut last_seen = HashMap::new();
        let mut pending_replay = HashMap::new();
        let now = Instant::now();
        for r in &config.regions {
            state_map.insert(r.clone(), RegionState::Healthy);
            let role = if r == &config.local_region {
                RegionRole::Primary
            } else {
                RegionRole::Secondary
            };
            role_map.insert(r.clone(), role);
            last_seen.insert(r.clone(), now);
            pending_replay.insert(r.clone(), VecDeque::new());
        }
        let inner = FailoverInner {
            state: state_map,
            role: role_map,
            last_seen,
            pending_replay,
            history: VecDeque::with_capacity(history_capacity),
        };
        Self {
            inner: Arc::new(Mutex::new(inner)),
            config,
            suspect_after,
            history_capacity,
        }
    }

    /// Record a heartbeat from `region`. Resets the suspect timer.
    pub fn heartbeat(&self, region: &RegionId) -> Result<(), RegionFailoverError> {
        let mut inner = self.lock_inner()?;
        if !inner.state.contains_key(region) {
            return Err(RegionFailoverError::UnknownRegion(region.clone()));
        }
        inner.last_seen.insert(region.clone(), Instant::now());
        // If the region was Suspect or Failed, transition back to Recovering.
        let current = inner
            .state
            .get(region)
            .copied()
            .unwrap_or(RegionState::Healthy);
        if matches!(current, RegionState::Suspect | RegionState::Failed) {
            inner.state.insert(region.clone(), RegionState::Recovering);
        } else {
            inner.state.insert(region.clone(), RegionState::Healthy);
        }
        Ok(())
    }

    /// Drive the suspect-detector forward. Any region whose last heartbeat
    /// is older than `suspect_after` becomes `Suspect`.
    pub fn tick(&self) -> Result<Vec<FailoverEvent>, RegionFailoverError> {
        let mut inner = self.lock_inner()?;
        let now = Instant::now();
        let mut emitted = Vec::new();
        let suspects: Vec<RegionId> = inner
            .last_seen
            .iter()
            .filter_map(|(r, t)| {
                if now.duration_since(*t) >= self.suspect_after {
                    Some(r.clone())
                } else {
                    None
                }
            })
            .collect();
        for region in suspects {
            let state = inner
                .state
                .get(&region)
                .copied()
                .unwrap_or(RegionState::Healthy);
            if matches!(state, RegionState::Healthy) {
                inner.state.insert(region.clone(), RegionState::Suspect);
                let ev = FailoverEvent::Suspect {
                    region: region.clone(),
                    at: now,
                };
                self.push_event(&mut inner, ev.clone());
                emitted.push(ev);
            }
        }
        Ok(emitted)
    }

    /// Mark a region failed and trigger a primary→secondary promotion.
    ///
    /// Returns the (`failed`, `promoted`) pair on success.
    pub fn demote_and_promote(
        &self,
        failed: &RegionId,
    ) -> Result<(RegionId, RegionId), RegionFailoverError> {
        let mut inner = self.lock_inner()?;
        if !inner.state.contains_key(failed) {
            return Err(RegionFailoverError::UnknownRegion(failed.clone()));
        }
        let current_role = inner
            .role
            .get(failed)
            .copied()
            .ok_or_else(|| RegionFailoverError::UnknownRegion(failed.clone()))?;
        if !matches!(
            current_role,
            RegionRole::Primary | RegionRole::PromotedSecondary
        ) {
            return Err(RegionFailoverError::IllegalTransition {
                region: failed.clone(),
                from: inner
                    .state
                    .get(failed)
                    .copied()
                    .unwrap_or(RegionState::Healthy),
            });
        }

        // Choose a healthy secondary as replacement, walking through the
        // primary tier ordering when defined.
        let replacement = self.pick_replacement(&inner, failed).ok_or_else(|| {
            RegionFailoverError::NoSecondaryAvailable {
                failed: failed.clone(),
            }
        })?;

        inner
            .role
            .insert(failed.clone(), RegionRole::DemotedPrimary);
        inner.state.insert(failed.clone(), RegionState::Failed);
        inner
            .role
            .insert(replacement.clone(), RegionRole::PromotedSecondary);
        inner
            .state
            .insert(replacement.clone(), RegionState::Healthy);

        let now = Instant::now();
        self.push_event(
            &mut inner,
            FailoverEvent::Demoted {
                region: failed.clone(),
                at: now,
            },
        );
        self.push_event(
            &mut inner,
            FailoverEvent::Promoted {
                region: replacement.clone(),
                replaced: failed.clone(),
                at: now,
            },
        );
        Ok((failed.clone(), replacement))
    }

    /// Buffer outstanding writes that need to be replayed to `region` once
    /// it comes back online.
    pub fn buffer_replay_writes<I: IntoIterator<Item = OutstandingWrite>>(
        &self,
        region: &RegionId,
        writes: I,
    ) -> Result<(), RegionFailoverError> {
        let mut inner = self.lock_inner()?;
        let q = inner
            .pending_replay
            .get_mut(region)
            .ok_or_else(|| RegionFailoverError::UnknownRegion(region.clone()))?;
        for w in writes {
            q.push_back(w);
        }
        Ok(())
    }

    /// Drain (and replay) all outstanding writes targeted at `region`,
    /// returning them to the caller.
    pub fn replay_outstanding(
        &self,
        region: &RegionId,
    ) -> Result<Vec<OutstandingWrite>, RegionFailoverError> {
        let mut inner = self.lock_inner()?;
        let q = inner
            .pending_replay
            .get_mut(region)
            .ok_or_else(|| RegionFailoverError::UnknownRegion(region.clone()))?;
        let drained: Vec<OutstandingWrite> = q.drain(..).collect();
        let count = drained.len();
        let now = Instant::now();
        if count > 0 {
            self.push_event(
                &mut inner,
                FailoverEvent::Replayed {
                    region: region.clone(),
                    replayed: count,
                    at: now,
                },
            );
        }
        Ok(drained)
    }

    /// Re-admit a region that has finished catch-up as a `Secondary`.
    pub fn readmit(&self, region: &RegionId) -> Result<(), RegionFailoverError> {
        let mut inner = self.lock_inner()?;
        if !inner.state.contains_key(region) {
            return Err(RegionFailoverError::UnknownRegion(region.clone()));
        }
        inner.role.insert(region.clone(), RegionRole::Secondary);
        inner.state.insert(region.clone(), RegionState::Healthy);
        inner.last_seen.insert(region.clone(), Instant::now());
        let ev = FailoverEvent::Readmitted {
            region: region.clone(),
            at: Instant::now(),
        };
        self.push_event(&mut inner, ev);
        Ok(())
    }

    /// Inspect a region's current role.
    pub fn role(&self, region: &RegionId) -> Result<RegionRole, RegionFailoverError> {
        let inner = self.lock_inner()?;
        inner
            .role
            .get(region)
            .copied()
            .ok_or_else(|| RegionFailoverError::UnknownRegion(region.clone()))
    }

    /// Inspect a region's current state.
    pub fn state(&self, region: &RegionId) -> Result<RegionState, RegionFailoverError> {
        let inner = self.lock_inner()?;
        inner
            .state
            .get(region)
            .copied()
            .ok_or_else(|| RegionFailoverError::UnknownRegion(region.clone()))
    }

    /// Number of writes still buffered for replay to `region`.
    pub fn pending_replay_len(&self, region: &RegionId) -> Result<usize, RegionFailoverError> {
        let inner = self.lock_inner()?;
        Ok(inner
            .pending_replay
            .get(region)
            .map(|q| q.len())
            .unwrap_or(0))
    }

    /// Read-only event history (oldest first).
    pub fn history(&self) -> Result<Vec<FailoverEvent>, RegionFailoverError> {
        let inner = self.lock_inner()?;
        Ok(inner.history.iter().cloned().collect())
    }

    /// Configuration in use.
    pub fn config(&self) -> &ActiveActiveGeoConfig {
        &self.config
    }

    /// Suspect timeout currently configured.
    pub fn suspect_after(&self) -> Duration {
        self.suspect_after
    }

    fn pick_replacement(&self, inner: &FailoverInner, failed: &RegionId) -> Option<RegionId> {
        // Prefer regions from the configured "primary" tier, in order.
        let tier_order = self
            .config
            .primary_tier
            .get("primary")
            .cloned()
            .unwrap_or_else(|| self.config.regions.clone());
        tier_order.into_iter().find(|r| {
            r != failed
                && matches!(
                    inner.state.get(r).copied().unwrap_or(RegionState::Healthy),
                    RegionState::Healthy | RegionState::Recovering
                )
                && matches!(
                    inner.role.get(r).copied().unwrap_or(RegionRole::Secondary),
                    RegionRole::Secondary
                )
        })
    }

    fn push_event(&self, inner: &mut FailoverInner, event: FailoverEvent) {
        if inner.history.len() == self.history_capacity {
            inner.history.pop_front();
        }
        inner.history.push_back(event);
    }

    fn lock_inner(&self) -> Result<std::sync::MutexGuard<'_, FailoverInner>, RegionFailoverError> {
        self.inner
            .lock()
            .map_err(|e| RegionFailoverError::LockPoisoned(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn config() -> ActiveActiveGeoConfig {
        ActiveActiveGeoConfig::multi_region(
            "us-east-1",
            vec![
                "us-east-1".to_string(),
                "eu-west-1".to_string(),
                "ap-northeast-1".to_string(),
            ],
        )
    }

    #[test]
    fn fresh_controller_has_local_primary() {
        let ctl = RegionFailoverController::new(config());
        assert_eq!(
            ctl.role(&"us-east-1".into()).expect("role"),
            RegionRole::Primary
        );
        assert_eq!(
            ctl.role(&"eu-west-1".into()).expect("role"),
            RegionRole::Secondary
        );
        assert_eq!(
            ctl.state(&"us-east-1".into()).expect("state"),
            RegionState::Healthy
        );
    }

    #[test]
    fn tick_marks_quiet_region_suspect() {
        let ctl = RegionFailoverController::with_options(config(), Duration::from_millis(20), 32);
        thread::sleep(Duration::from_millis(40));
        let events = ctl.tick().expect("tick");
        // Local region should also be marked suspect since we never refreshed.
        assert!(!events.is_empty());
        let suspects = events
            .iter()
            .filter_map(|e| match e {
                FailoverEvent::Suspect { region, .. } => Some(region.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!suspects.is_empty());
    }

    #[test]
    fn heartbeat_clears_suspect_state() {
        let ctl = RegionFailoverController::with_options(config(), Duration::from_millis(20), 32);
        thread::sleep(Duration::from_millis(40));
        ctl.tick().expect("tick");
        assert_eq!(
            ctl.state(&"eu-west-1".into()).expect("state"),
            RegionState::Suspect
        );
        ctl.heartbeat(&"eu-west-1".into()).expect("heartbeat");
        assert_eq!(
            ctl.state(&"eu-west-1".into()).expect("state"),
            RegionState::Recovering
        );
    }

    #[test]
    fn demote_and_promote_picks_secondary() {
        let ctl = RegionFailoverController::new(config());
        let (failed, promoted) = ctl
            .demote_and_promote(&"us-east-1".into())
            .expect("demote+promote");
        assert_eq!(failed, "us-east-1");
        assert!(["us-east-1", "eu-west-1", "ap-northeast-1"].contains(&promoted.as_str()));
        assert_eq!(ctl.role(&failed).expect("role"), RegionRole::DemotedPrimary);
        assert_eq!(
            ctl.role(&promoted).expect("role"),
            RegionRole::PromotedSecondary
        );
    }

    #[test]
    fn demote_with_no_secondaries_errors() {
        let cfg = ActiveActiveGeoConfig::single_region("solo");
        let ctl = RegionFailoverController::new(cfg);
        let res = ctl.demote_and_promote(&"solo".into());
        assert!(matches!(
            res,
            Err(RegionFailoverError::NoSecondaryAvailable { .. })
        ));
    }

    #[test]
    fn demote_unknown_region_errors() {
        let ctl = RegionFailoverController::new(config());
        let res = ctl.demote_and_promote(&"mars-1".into());
        assert!(matches!(res, Err(RegionFailoverError::UnknownRegion(_))));
    }

    #[test]
    fn replay_outstanding_drains_buffer() {
        let ctl = RegionFailoverController::new(config());
        let writes = vec![
            OutstandingWrite {
                origin_region: "us-east-1".into(),
                seq: 1,
                key: "k1".into(),
                value: "v1".into(),
            },
            OutstandingWrite {
                origin_region: "us-east-1".into(),
                seq: 2,
                key: "k2".into(),
                value: "v2".into(),
            },
        ];
        ctl.buffer_replay_writes(&"eu-west-1".into(), writes.clone())
            .expect("buffer");
        assert_eq!(ctl.pending_replay_len(&"eu-west-1".into()).expect("len"), 2);
        let drained = ctl.replay_outstanding(&"eu-west-1".into()).expect("replay");
        assert_eq!(drained.len(), 2);
        assert_eq!(drained, writes);
        assert_eq!(ctl.pending_replay_len(&"eu-west-1".into()).expect("len"), 0);
    }

    #[test]
    fn full_failover_cycle() {
        let ctl = RegionFailoverController::new(config());
        // Buffer some outstanding writes to be replayed when EU comes back.
        ctl.buffer_replay_writes(
            &"eu-west-1".into(),
            vec![OutstandingWrite {
                origin_region: "us-east-1".into(),
                seq: 1,
                key: "k".into(),
                value: "v".into(),
            }],
        )
        .expect("buffer");
        // Failure of US.
        let (failed, promoted) = ctl
            .demote_and_promote(&"us-east-1".into())
            .expect("failover");
        assert_eq!(failed, "us-east-1");
        assert_ne!(promoted, "us-east-1");
        // Recovery: heartbeat and readmit US as secondary.
        ctl.heartbeat(&"us-east-1".into()).expect("heartbeat");
        ctl.readmit(&"us-east-1".into()).expect("readmit");
        assert_eq!(
            ctl.role(&"us-east-1".into()).expect("role"),
            RegionRole::Secondary
        );
        assert_eq!(
            ctl.state(&"us-east-1".into()).expect("state"),
            RegionState::Healthy
        );
        // Replay outstanding writes.
        let drained = ctl.replay_outstanding(&"eu-west-1".into()).expect("replay");
        assert_eq!(drained.len(), 1);
        // History should record the demote / promote / replay / readmit.
        let hist = ctl.history().expect("history");
        assert!(hist
            .iter()
            .any(|e| matches!(e, FailoverEvent::Demoted { .. })));
        assert!(hist
            .iter()
            .any(|e| matches!(e, FailoverEvent::Promoted { .. })));
        assert!(hist
            .iter()
            .any(|e| matches!(e, FailoverEvent::Replayed { .. })));
        assert!(hist
            .iter()
            .any(|e| matches!(e, FailoverEvent::Readmitted { .. })));
    }
}
