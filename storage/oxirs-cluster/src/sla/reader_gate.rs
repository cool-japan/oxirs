//! Reader-side SLA gate for read-replica serving.
//!
//! Mirrors [`super::proposer_gate::SlaProposerGate`] but targets the
//! read-replica request path. The read-replica handler invokes
//! [`SlaReaderGate::try_acquire`] before serving a read; on success the
//! handler proceeds, on rejection it returns a `429`-equivalent to the
//! upstream caller.
//!
//! In contrast to the proposer gate, reads typically have a different
//! cost profile — large, expensive scans may cost more than a single
//! token. The gate therefore exposes [`SlaReaderGate::try_acquire_with_cost`]
//! for callers that have a non-default cost model.

use std::sync::Arc;

use oxirs_core::sla::SlaClass;

use super::admission::{ClusterAdmissionController, SlaError};

// ─────────────────────────────────────────────────────────────────────────────
// Outcome
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of [`SlaReaderGate::try_acquire`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReaderOutcome {
    /// Read was admitted — caller may proceed.
    Admitted,
    /// Read was rejected (rate or concurrency cap exceeded).
    Rejected(SlaError),
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate
// ─────────────────────────────────────────────────────────────────────────────

/// SLA-aware read-replica gate.
#[derive(Debug, Clone)]
pub struct SlaReaderGate {
    controller: Arc<ClusterAdmissionController>,
}

impl SlaReaderGate {
    /// Build a new gate sharing the given admission controller.
    pub fn new(controller: Arc<ClusterAdmissionController>) -> Self {
        Self { controller }
    }

    /// Reference to the underlying admission controller.
    pub fn controller(&self) -> &Arc<ClusterAdmissionController> {
        &self.controller
    }

    /// Attempt to admit a single read for the given class.
    pub fn try_acquire(&self, class: SlaClass) -> Result<(), SlaError> {
        self.controller.acquire_permit(class)
    }

    /// Attempt to admit, returning a structured outcome.
    pub fn admit(&self, class: SlaClass) -> ReaderOutcome {
        match self.controller.acquire_permit(class) {
            Ok(()) => ReaderOutcome::Admitted,
            Err(e) => ReaderOutcome::Rejected(e),
        }
    }

    /// Attempt to admit a read whose cost is `cost` tokens (rather than
    /// the default 1.0 from the per-class quota).
    ///
    /// The concurrency cap still applies — only the token-bucket cost is
    /// scaled. A successful admission still increments the in-flight
    /// counter by 1; the caller must release exactly once via
    /// [`Self::release`].
    pub fn try_acquire_with_cost(&self, class: SlaClass, cost: f64) -> Result<(), SlaError> {
        self.controller.acquire_permit_with_cost(class, cost)
    }

    /// Release a previously acquired permit.
    pub fn release(&self, class: SlaClass) -> Result<(), SlaError> {
        self.controller.release_permit(class)
    }

    /// Scoped helper.
    pub fn scoped<F, T>(&self, class: SlaClass, f: F) -> Result<T, SlaError>
    where
        F: FnOnce() -> T,
    {
        self.controller.scoped_admit(class, f)
    }

    /// Number of currently in-flight reads for `class`.
    pub fn in_flight(&self, class: SlaClass) -> Result<usize, SlaError> {
        self.controller.in_flight(class)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sla::admission::{SlaAdmissionConfig, SlaClassQuota};
    use std::collections::HashMap;

    fn small_gate() -> SlaReaderGate {
        let mut quotas = HashMap::new();
        quotas.insert(
            SlaClass::Silver,
            SlaClassQuota {
                max_qps: None,
                max_concurrent: 2,
                token_cost: 1.0,
            },
        );
        let controller = Arc::new(ClusterAdmissionController::new(SlaAdmissionConfig {
            quotas,
        }));
        SlaReaderGate::new(controller)
    }

    #[test]
    fn admit_succeeds_within_cap() {
        let gate = small_gate();
        assert_eq!(gate.admit(SlaClass::Silver), ReaderOutcome::Admitted);
        assert_eq!(gate.admit(SlaClass::Silver), ReaderOutcome::Admitted);
    }

    #[test]
    fn admit_blocked_after_cap() {
        let gate = small_gate();
        gate.try_acquire(SlaClass::Silver).expect("1");
        gate.try_acquire(SlaClass::Silver).expect("2");
        let outcome = gate.admit(SlaClass::Silver);
        match outcome {
            ReaderOutcome::Rejected(SlaError::ConcurrencyCapExceeded { limit, .. }) => {
                assert_eq!(limit, 2);
            }
            other => panic!("unexpected outcome: {:?}", other),
        }
    }

    #[test]
    fn cost_aware_admit_drains_bucket() {
        let gate = small_gate();
        // Silver bucket capacity = 20; spend 10 in one shot.
        gate.try_acquire_with_cost(SlaClass::Silver, 10.0)
            .expect("expensive read");
        assert_eq!(gate.in_flight(SlaClass::Silver).expect("count"), 1);
        gate.release(SlaClass::Silver).expect("release");
    }

    #[test]
    fn cost_aware_admit_rejects_unregistered() {
        let gate = small_gate();
        let res = gate.try_acquire_with_cost(SlaClass::Bronze, 1.0);
        assert!(matches!(res, Err(SlaError::ClassNotRegistered { .. })));
    }

    #[test]
    fn scoped_releases_on_normal_return() {
        let gate = small_gate();
        gate.scoped(SlaClass::Silver, || ()).expect("scoped");
        assert_eq!(gate.in_flight(SlaClass::Silver).expect("count"), 0);
    }
}
