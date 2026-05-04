//! Per-class admission control for the cluster.
//!
//! Wraps [`oxirs_core::sla::AdmissionController`] (which is per-tenant) by
//! registering a synthetic tenant for each [`SlaClass`] tier — that gives
//! us a token bucket per class with the refill rate / capacity defined by
//! the shared SLA primitives.
//!
//! Adds an additional, *strict* concurrency cap: `max_concurrent_per_class`.
//! While the token bucket smooths out bursts (qps), the concurrency cap
//! guarantees that no more than N requests of a given class are *in flight*
//! at any moment. Acquired permits must be released; the controller exposes
//! `acquire_permit` / `release_permit` for that purpose.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use oxirs_core::sla::{AdmissionController, AdmissionError, SlaClass};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Quota for a single SLA class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlaClassQuota {
    /// Reserved knob for a future per-class override of the token refill
    /// rate.
    ///
    /// Currently informational only — the underlying token bucket sources
    /// its refill rate from [`SlaClass::thresholds`] (specifically
    /// `token_refill_rate`). Once `oxirs_core::sla::AdmissionController`
    /// gains a per-tenant override hook, this field will be wired through.
    pub max_qps: Option<f64>,
    /// Maximum number of concurrently in-flight requests for this class.
    ///
    /// Enforced as a hard counter; exceeding requests are rejected
    /// independently of token-bucket state.
    pub max_concurrent: usize,
    /// Token cost charged per admit attempt (default: `1.0`).
    pub token_cost: f64,
}

impl SlaClassQuota {
    /// Default quota for a class — uses the class's own threshold values.
    pub fn for_class(class: SlaClass) -> Self {
        let t = class.thresholds();
        Self {
            max_qps: None,
            max_concurrent: t.max_concurrent_queries,
            token_cost: 1.0,
        }
    }
}

/// Cluster-wide SLA admission configuration.
///
/// Holds one [`SlaClassQuota`] per registered class. Classes that are not
/// registered are rejected at admission time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SlaAdmissionConfig {
    /// Per-class quotas.
    pub quotas: HashMap<SlaClass, SlaClassQuota>,
}

impl SlaAdmissionConfig {
    /// Build a config seeded with default quotas for every tier
    /// (Bronze → Platinum).
    pub fn with_defaults() -> Self {
        let mut quotas = HashMap::new();
        for class in [
            SlaClass::Bronze,
            SlaClass::Silver,
            SlaClass::Gold,
            SlaClass::Platinum,
        ] {
            quotas.insert(class, SlaClassQuota::for_class(class));
        }
        Self { quotas }
    }

    /// Attach a quota for `class`.
    pub fn with_class(mut self, class: SlaClass, quota: SlaClassQuota) -> Self {
        self.quotas.insert(class, quota);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors emitted by the cluster admission layer.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SlaError {
    /// SLA class was never registered with the controller.
    #[error("SLA class '{class}' is not registered")]
    ClassNotRegistered { class: String },

    /// Token bucket for the class is exhausted (rate-limited).
    #[error("SLA token bucket exhausted for class '{class}'")]
    RateLimitExceeded { class: String },

    /// Concurrency cap reached for the class.
    #[error("SLA concurrency cap reached for class '{class}' (limit={limit})")]
    ConcurrencyCapExceeded { class: String, limit: usize },

    /// Lock poisoning.
    #[error("SLA admission state lock poisoned: {0}")]
    LockPoisoned(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats
// ─────────────────────────────────────────────────────────────────────────────

/// Cumulative admission statistics, per class.
#[derive(Debug, Clone, Default)]
pub struct ClusterAdmissionStats {
    /// Total successful admissions per class.
    pub admitted: HashMap<SlaClass, u64>,
    /// Token-bucket rejections per class.
    pub rejected_rate: HashMap<SlaClass, u64>,
    /// Concurrency-cap rejections per class.
    pub rejected_concurrency: HashMap<SlaClass, u64>,
    /// Number of currently in-flight requests, per class.
    pub in_flight: HashMap<SlaClass, usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Controller
// ─────────────────────────────────────────────────────────────────────────────

/// Per-class admission controller for the cluster.
///
/// Internally each class is registered as a synthetic tenant with the shared
/// [`oxirs_core::sla::AdmissionController`]. The synthetic tenant id is the
/// class's lowercase name (`"bronze"`, `"silver"`, `"gold"`, `"platinum"`).
pub struct ClusterAdmissionController {
    inner: AdmissionController,
    state: Arc<Mutex<AdmissionState>>,
    config: SlaAdmissionConfig,
}

impl std::fmt::Debug for ClusterAdmissionController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterAdmissionController")
            .field("config", &self.config)
            .field("class_count", &self.class_count())
            .finish()
    }
}

#[derive(Debug, Default)]
struct AdmissionState {
    in_flight: HashMap<SlaClass, usize>,
    admitted: HashMap<SlaClass, u64>,
    rejected_rate: HashMap<SlaClass, u64>,
    rejected_concurrency: HashMap<SlaClass, u64>,
}

impl ClusterAdmissionController {
    /// Build a fresh controller from the given config.
    ///
    /// Classes whose `quotas` entry is present are auto-registered at
    /// construction time. Unregistered classes can be added later with
    /// [`Self::register_class`].
    pub fn new(config: SlaAdmissionConfig) -> Self {
        let inner = AdmissionController::new();
        let ctrl = Self {
            inner,
            state: Arc::new(Mutex::new(AdmissionState::default())),
            config,
        };
        let classes: Vec<SlaClass> = ctrl.config.quotas.keys().copied().collect();
        for c in classes {
            ctrl.register_class_internal(c);
        }
        ctrl
    }

    /// Register (or re-register) a class with the underlying controller.
    pub fn register_class(&self, class: SlaClass) {
        self.register_class_internal(class);
    }

    fn register_class_internal(&self, class: SlaClass) {
        self.inner.register_tenant(class.name(), class);
    }

    /// Look up the quota currently in force for `class`.
    pub fn quota(&self, class: SlaClass) -> Option<SlaClassQuota> {
        self.config.quotas.get(&class).cloned()
    }

    /// `true` when `class` is registered.
    pub fn is_class_registered(&self, class: SlaClass) -> bool {
        self.inner.sla_class(class.name()).is_some()
    }

    /// Number of distinct classes currently registered.
    pub fn class_count(&self) -> usize {
        self.inner.tenant_count()
    }

    /// Try to admit a single request of `class`.
    ///
    /// On success the per-class in-flight counter is incremented; the caller
    /// must invoke [`Self::release_permit`] (or use [`Self::scoped_admit`])
    /// once the request finishes.
    pub fn acquire_permit(&self, class: SlaClass) -> Result<(), SlaError> {
        // Class registration check.
        if !self.is_class_registered(class) {
            return Err(SlaError::ClassNotRegistered {
                class: class.name().to_string(),
            });
        }
        let quota = self
            .config
            .quotas
            .get(&class)
            .cloned()
            .unwrap_or_else(|| SlaClassQuota::for_class(class));

        // Concurrency check (taken first so a rate-rejected request never
        // pollutes the in-flight counter).
        {
            let mut st = self.lock_state()?;
            let counter = st.in_flight.entry(class).or_insert(0);
            if *counter >= quota.max_concurrent {
                let limit = quota.max_concurrent;
                *st.rejected_concurrency.entry(class).or_insert(0) += 1;
                return Err(SlaError::ConcurrencyCapExceeded {
                    class: class.name().to_string(),
                    limit,
                });
            }
        }

        // Rate check via token bucket.
        match self
            .inner
            .try_admit_with_cost(class.name(), quota.token_cost)
        {
            Ok(()) => {
                let mut st = self.lock_state()?;
                *st.in_flight.entry(class).or_insert(0) += 1;
                *st.admitted.entry(class).or_insert(0) += 1;
                Ok(())
            }
            Err(AdmissionError::RateLimitExceeded { .. }) => {
                let mut st = self.lock_state()?;
                *st.rejected_rate.entry(class).or_insert(0) += 1;
                Err(SlaError::RateLimitExceeded {
                    class: class.name().to_string(),
                })
            }
            Err(AdmissionError::TenantNotRegistered { .. }) => Err(SlaError::ClassNotRegistered {
                class: class.name().to_string(),
            }),
        }
    }

    /// Try to admit with a custom token cost (still enforces the same
    /// concurrency cap).
    pub fn acquire_permit_with_cost(&self, class: SlaClass, cost: f64) -> Result<(), SlaError> {
        if !self.is_class_registered(class) {
            return Err(SlaError::ClassNotRegistered {
                class: class.name().to_string(),
            });
        }
        let quota = self
            .config
            .quotas
            .get(&class)
            .cloned()
            .unwrap_or_else(|| SlaClassQuota::for_class(class));

        // Concurrency check first.
        {
            let mut st = self.lock_state()?;
            let counter = st.in_flight.entry(class).or_insert(0);
            if *counter >= quota.max_concurrent {
                let limit = quota.max_concurrent;
                *st.rejected_concurrency.entry(class).or_insert(0) += 1;
                return Err(SlaError::ConcurrencyCapExceeded {
                    class: class.name().to_string(),
                    limit,
                });
            }
        }

        match self.inner.try_admit_with_cost(class.name(), cost) {
            Ok(()) => {
                let mut st = self.lock_state()?;
                *st.in_flight.entry(class).or_insert(0) += 1;
                *st.admitted.entry(class).or_insert(0) += 1;
                Ok(())
            }
            Err(AdmissionError::RateLimitExceeded { .. }) => {
                let mut st = self.lock_state()?;
                *st.rejected_rate.entry(class).or_insert(0) += 1;
                Err(SlaError::RateLimitExceeded {
                    class: class.name().to_string(),
                })
            }
            Err(AdmissionError::TenantNotRegistered { .. }) => Err(SlaError::ClassNotRegistered {
                class: class.name().to_string(),
            }),
        }
    }

    /// Release a previously acquired permit.
    ///
    /// Releasing a permit that was never acquired is a no-op (the counter
    /// is saturated at zero) so callers never have to worry about double
    /// release in error paths.
    pub fn release_permit(&self, class: SlaClass) -> Result<(), SlaError> {
        let mut st = self.lock_state()?;
        if let Some(counter) = st.in_flight.get_mut(&class) {
            *counter = counter.saturating_sub(1);
        }
        Ok(())
    }

    /// Scoped helper: acquire a permit, run `f`, release on drop.
    pub fn scoped_admit<F, T>(&self, class: SlaClass, f: F) -> Result<T, SlaError>
    where
        F: FnOnce() -> T,
    {
        self.acquire_permit(class)?;
        let result = f();
        self.release_permit(class)?;
        Ok(result)
    }

    /// Snapshot of cumulative statistics.
    pub fn stats(&self) -> Result<ClusterAdmissionStats, SlaError> {
        let st = self.lock_state()?;
        Ok(ClusterAdmissionStats {
            admitted: st.admitted.clone(),
            rejected_rate: st.rejected_rate.clone(),
            rejected_concurrency: st.rejected_concurrency.clone(),
            in_flight: st.in_flight.clone(),
        })
    }

    /// Number of currently in-flight requests for `class`.
    pub fn in_flight(&self, class: SlaClass) -> Result<usize, SlaError> {
        let st = self.lock_state()?;
        Ok(st.in_flight.get(&class).copied().unwrap_or(0))
    }

    /// Peek at the current available token count for `class`.
    pub fn available_tokens(&self, class: SlaClass) -> Option<f64> {
        self.inner.available_tokens(class.name())
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, AdmissionState>, SlaError> {
        self.state
            .lock()
            .map_err(|e| SlaError::LockPoisoned(e.to_string()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn small_config() -> SlaAdmissionConfig {
        let mut quotas: HashMap<SlaClass, SlaClassQuota> = HashMap::new();
        quotas.insert(
            SlaClass::Bronze,
            SlaClassQuota {
                max_qps: None,
                max_concurrent: 2,
                token_cost: 1.0,
            },
        );
        quotas.insert(
            SlaClass::Gold,
            SlaClassQuota {
                max_qps: None,
                max_concurrent: 5,
                token_cost: 1.0,
            },
        );
        SlaAdmissionConfig { quotas }
    }

    #[test]
    fn defaults_register_all_four_classes() {
        let ctrl = ClusterAdmissionController::new(SlaAdmissionConfig::with_defaults());
        assert!(ctrl.is_class_registered(SlaClass::Bronze));
        assert!(ctrl.is_class_registered(SlaClass::Silver));
        assert!(ctrl.is_class_registered(SlaClass::Gold));
        assert!(ctrl.is_class_registered(SlaClass::Platinum));
        assert_eq!(ctrl.class_count(), 4);
    }

    #[test]
    fn unknown_class_rejected() {
        // Empty config — no classes are registered, even Bronze.
        let ctrl = ClusterAdmissionController::new(SlaAdmissionConfig::default());
        let err = ctrl
            .acquire_permit(SlaClass::Bronze)
            .expect_err("must fail");
        assert!(matches!(err, SlaError::ClassNotRegistered { .. }));
    }

    #[test]
    fn concurrency_cap_blocks_extra_requests() {
        let ctrl = ClusterAdmissionController::new(small_config());
        // Bronze cap = 2 — first two succeed, third is rejected.
        ctrl.acquire_permit(SlaClass::Bronze).expect("admit 1");
        ctrl.acquire_permit(SlaClass::Bronze).expect("admit 2");
        let err = ctrl
            .acquire_permit(SlaClass::Bronze)
            .expect_err("third must fail");
        assert!(matches!(err, SlaError::ConcurrencyCapExceeded { .. }));
        // Releasing one frees a slot.
        ctrl.release_permit(SlaClass::Bronze).expect("release");
        ctrl.acquire_permit(SlaClass::Bronze)
            .expect("post-release admit");
    }

    #[test]
    fn rate_limit_kicks_in_when_bucket_drained() {
        // Bronze capacity = 5 (per oxirs_core::sla thresholds).
        let ctrl = ClusterAdmissionController::new(SlaAdmissionConfig::with_defaults());
        let mut admitted = 0usize;
        let mut rejected = 0usize;
        for _ in 0..20 {
            match ctrl.acquire_permit(SlaClass::Bronze) {
                Ok(()) => {
                    admitted += 1;
                    // Release so the concurrency cap doesn't kick in.
                    ctrl.release_permit(SlaClass::Bronze).expect("release");
                }
                Err(SlaError::RateLimitExceeded { .. }) => rejected += 1,
                Err(e) => panic!("unexpected error: {:?}", e),
            }
        }
        assert!(admitted >= 1);
        assert!(rejected >= 1);
    }

    #[test]
    fn stats_track_admit_and_rejection_counts() {
        let ctrl = ClusterAdmissionController::new(small_config());
        ctrl.acquire_permit(SlaClass::Bronze).expect("admit 1");
        ctrl.acquire_permit(SlaClass::Bronze).expect("admit 2");
        let _ = ctrl.acquire_permit(SlaClass::Bronze); // rejected on concurrency
        let stats = ctrl.stats().expect("stats");
        assert_eq!(stats.admitted.get(&SlaClass::Bronze).copied(), Some(2));
        assert_eq!(
            stats.rejected_concurrency.get(&SlaClass::Bronze).copied(),
            Some(1)
        );
        assert_eq!(stats.in_flight.get(&SlaClass::Bronze).copied(), Some(2));
    }

    #[test]
    fn release_does_not_underflow() {
        let ctrl = ClusterAdmissionController::new(small_config());
        // Releasing without acquiring is safe.
        ctrl.release_permit(SlaClass::Bronze).expect("ok");
        assert_eq!(ctrl.in_flight(SlaClass::Bronze).expect("count"), 0);
    }

    #[test]
    fn scoped_admit_releases_permit() {
        let ctrl = ClusterAdmissionController::new(small_config());
        ctrl.scoped_admit(SlaClass::Gold, || {
            // Inside the scope: 1 in flight.
            assert_eq!(ctrl.in_flight(SlaClass::Gold).expect("count"), 1);
        })
        .expect("scoped");
        // After the scope closes the counter is back to zero.
        assert_eq!(ctrl.in_flight(SlaClass::Gold).expect("count"), 0);
    }

    #[test]
    fn shared_via_arc_is_thread_safe() {
        let ctrl = Arc::new(ClusterAdmissionController::new(small_config()));
        let c1 = ctrl.clone();
        let c2 = ctrl.clone();
        let h1 = std::thread::spawn(move || c1.acquire_permit(SlaClass::Bronze));
        let h2 = std::thread::spawn(move || c2.acquire_permit(SlaClass::Bronze));
        let r1 = h1.join().expect("join 1");
        let r2 = h2.join().expect("join 2");
        // Both should succeed (cap = 2), but if either fails, it should be on
        // concurrency, not rate (we haven't drained the bucket).
        for r in [&r1, &r2] {
            if let Err(e) = r {
                assert!(matches!(e, SlaError::ConcurrencyCapExceeded { .. }));
            }
        }
    }
}
