//! Proposer-side SLA gate for Raft log writes.
//!
//! Wraps a [`ClusterAdmissionController`] and exposes a thin interface that
//! the Raft proposer (or any caller producing log entries) can use to
//! gate writes.
//!
//! Usage pattern:
//!
//! ```ignore
//! let gate = SlaProposerGate::new(controller);
//! gate.try_acquire(SlaClass::Gold)?;
//! // ... actually call ConsensusManager::propose_command(...) ...
//! gate.release(SlaClass::Gold);
//! ```
//!
//! Or use the scoped variant:
//!
//! ```ignore
//! gate.scoped(SlaClass::Gold, || { /* propose */ })?;
//! ```

use std::sync::Arc;

use oxirs_core::sla::SlaClass;

use super::admission::{ClusterAdmissionController, SlaError};

// ─────────────────────────────────────────────────────────────────────────────
// Outcome
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of [`SlaProposerGate::try_acquire`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProposerOutcome {
    /// The propose was admitted — caller may proceed.
    Admitted,
    /// The propose was rejected (rate or concurrency cap exceeded).
    Rejected(SlaError),
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate
// ─────────────────────────────────────────────────────────────────────────────

/// SLA-aware Raft log proposer gate.
#[derive(Debug, Clone)]
pub struct SlaProposerGate {
    controller: Arc<ClusterAdmissionController>,
}

impl SlaProposerGate {
    /// Build a new gate sharing the given admission controller.
    pub fn new(controller: Arc<ClusterAdmissionController>) -> Self {
        Self { controller }
    }

    /// Reference to the underlying admission controller.
    pub fn controller(&self) -> &Arc<ClusterAdmissionController> {
        &self.controller
    }

    /// Attempt to admit a single propose for the given class.
    ///
    /// Caller must invoke [`Self::release`] (or use [`Self::scoped`]) once
    /// the propose has finished.
    pub fn try_acquire(&self, class: SlaClass) -> Result<(), SlaError> {
        self.controller.acquire_permit(class)
    }

    /// Attempt to admit, returning a structured outcome instead of an
    /// error.  Useful when the caller wants to log/measure rejections in
    /// addition to surfacing them.
    pub fn admit(&self, class: SlaClass) -> ProposerOutcome {
        match self.controller.acquire_permit(class) {
            Ok(()) => ProposerOutcome::Admitted,
            Err(e) => ProposerOutcome::Rejected(e),
        }
    }

    /// Release a previously acquired permit.
    pub fn release(&self, class: SlaClass) -> Result<(), SlaError> {
        self.controller.release_permit(class)
    }

    /// Scoped helper: acquire-call-release in a single call.
    pub fn scoped<F, T>(&self, class: SlaClass, f: F) -> Result<T, SlaError>
    where
        F: FnOnce() -> T,
    {
        self.controller.scoped_admit(class, f)
    }

    /// Number of currently in-flight proposes for `class`.
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

    fn small_gate() -> SlaProposerGate {
        let mut quotas = HashMap::new();
        quotas.insert(
            SlaClass::Bronze,
            SlaClassQuota {
                max_qps: None,
                max_concurrent: 1,
                token_cost: 1.0,
            },
        );
        let controller = Arc::new(ClusterAdmissionController::new(SlaAdmissionConfig {
            quotas,
        }));
        SlaProposerGate::new(controller)
    }

    #[test]
    fn admitted_outcome_for_first_request() {
        let gate = small_gate();
        assert_eq!(gate.admit(SlaClass::Bronze), ProposerOutcome::Admitted);
    }

    #[test]
    fn second_request_blocked_by_concurrency_cap() {
        let gate = small_gate();
        gate.try_acquire(SlaClass::Bronze).expect("first");
        let outcome = gate.admit(SlaClass::Bronze);
        match outcome {
            ProposerOutcome::Rejected(SlaError::ConcurrencyCapExceeded { limit, .. }) => {
                assert_eq!(limit, 1);
            }
            other => panic!("unexpected outcome: {:?}", other),
        }
    }

    #[test]
    fn release_frees_slot() {
        let gate = small_gate();
        gate.try_acquire(SlaClass::Bronze).expect("first");
        gate.release(SlaClass::Bronze).expect("release");
        gate.try_acquire(SlaClass::Bronze).expect("second");
    }

    #[test]
    fn scoped_runs_callback_and_releases() {
        let gate = small_gate();
        let result = gate.scoped(SlaClass::Bronze, || 42).expect("scoped");
        assert_eq!(result, 42);
        assert_eq!(gate.in_flight(SlaClass::Bronze).expect("count"), 0);
    }

    #[test]
    fn unregistered_class_rejected() {
        // Bronze is registered, but Gold is not.
        let gate = small_gate();
        let outcome = gate.admit(SlaClass::Gold);
        assert!(matches!(
            outcome,
            ProposerOutcome::Rejected(SlaError::ClassNotRegistered { .. })
        ));
    }
}
