//! Token-bucket admission controller — one bucket per tenant.
//!
//! Each tenant is registered with an [`SlaClass`]; the controller maintains a
//! leaky-bucket that refills at the rate specified by
//! [`super::thresholds::SlaThresholds::token_refill_rate`] and can burst up to
//! [`super::thresholds::SlaThresholds::token_bucket_capacity`] tokens.
//!
//! A call to [`AdmissionController::try_admit`] deducts 1.0 token and returns
//! `Ok(())` on success or an [`AdmissionError`] on rejection.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use thiserror::Error;

use super::class::SlaClass;

// ─────────────────────────────────────────────────────────────────────────────
// AdmissionError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`AdmissionController::try_admit`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AdmissionError {
    /// The tenant exhausted its token bucket and cannot accept more requests
    /// at this time.
    #[error("Rate limit exceeded for tenant '{tenant_id}'")]
    RateLimitExceeded {
        /// The tenant identifier whose bucket is empty.
        tenant_id: String,
    },

    /// The tenant has not been registered with [`AdmissionController::register_tenant`].
    #[error("Tenant '{tenant_id}' is not registered with the admission controller")]
    TenantNotRegistered {
        /// The tenant identifier that was not found in the registry.
        tenant_id: String,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// TokenBucket (private)
// ─────────────────────────────────────────────────────────────────────────────

struct TokenBucket {
    tokens: f64,
    capacity: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
    sla_class: SlaClass,
}

impl TokenBucket {
    fn new(sla: SlaClass) -> Self {
        let t = sla.thresholds();
        TokenBucket {
            tokens: t.token_bucket_capacity,
            capacity: t.token_bucket_capacity,
            refill_rate: t.token_refill_rate,
            last_refill: Instant::now(),
            sla_class: sla,
        }
    }

    /// Bring the bucket up to date with elapsed wall time.
    fn refill(&mut self) {
        let elapsed_secs = self.last_refill.elapsed().as_secs_f64();
        self.tokens = (self.tokens + elapsed_secs * self.refill_rate).min(self.capacity);
        self.last_refill = Instant::now();
    }

    /// Attempt to consume `cost` tokens.  Returns `true` if admitted.
    fn try_consume(&mut self, cost: f64) -> bool {
        self.refill();
        if self.tokens >= cost {
            self.tokens -= cost;
            true
        } else {
            false
        }
    }

    fn sla_class(&self) -> SlaClass {
        self.sla_class
    }

    /// Current token level (after lazy refill).
    fn available_tokens(&mut self) -> f64 {
        self.refill();
        self.tokens
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdmissionController
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe admission controller backed by per-tenant token buckets.
///
/// ```rust
/// use oxirs_core::sla::{SlaClass, AdmissionController};
///
/// let ctrl = AdmissionController::new();
/// ctrl.register_tenant("premium_user", SlaClass::Platinum);
/// assert!(ctrl.try_admit("premium_user").is_ok());
/// ```
#[derive(Clone)]
pub struct AdmissionController {
    buckets: Arc<Mutex<HashMap<String, TokenBucket>>>,
}

impl Default for AdmissionController {
    fn default() -> Self {
        Self::new()
    }
}

impl AdmissionController {
    /// Create an empty controller (no tenants registered).
    pub fn new() -> Self {
        AdmissionController {
            buckets: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register (or re-register) a tenant with the given SLA class.
    ///
    /// Re-registering an existing tenant resets its bucket to full capacity.
    pub fn register_tenant(&self, tenant_id: &str, sla: SlaClass) {
        let mut buckets = self.buckets.lock().unwrap_or_else(|e| e.into_inner());
        buckets.insert(tenant_id.to_owned(), TokenBucket::new(sla));
    }

    /// Try to admit one query unit (cost = 1.0 token) for `tenant_id`.
    ///
    /// Returns `Ok(())` when admitted, `Err(AdmissionError)` otherwise.
    pub fn try_admit(&self, tenant_id: &str) -> Result<(), AdmissionError> {
        let mut buckets = self.buckets.lock().unwrap_or_else(|e| e.into_inner());
        match buckets.get_mut(tenant_id) {
            Some(bucket) => {
                if bucket.try_consume(1.0) {
                    Ok(())
                } else {
                    Err(AdmissionError::RateLimitExceeded {
                        tenant_id: tenant_id.to_owned(),
                    })
                }
            }
            None => Err(AdmissionError::TenantNotRegistered {
                tenant_id: tenant_id.to_owned(),
            }),
        }
    }

    /// Try to admit a request with a custom token cost.
    pub fn try_admit_with_cost(&self, tenant_id: &str, cost: f64) -> Result<(), AdmissionError> {
        let mut buckets = self.buckets.lock().unwrap_or_else(|e| e.into_inner());
        match buckets.get_mut(tenant_id) {
            Some(bucket) => {
                if bucket.try_consume(cost) {
                    Ok(())
                } else {
                    Err(AdmissionError::RateLimitExceeded {
                        tenant_id: tenant_id.to_owned(),
                    })
                }
            }
            None => Err(AdmissionError::TenantNotRegistered {
                tenant_id: tenant_id.to_owned(),
            }),
        }
    }

    /// Return the SLA class for a registered tenant, if any.
    pub fn sla_class(&self, tenant_id: &str) -> Option<SlaClass> {
        let mut buckets = self.buckets.lock().unwrap_or_else(|e| e.into_inner());
        buckets.get_mut(tenant_id).map(|b| b.sla_class())
    }

    /// Return the current available token count for `tenant_id`.
    ///
    /// Returns `None` when the tenant is not registered.
    pub fn available_tokens(&self, tenant_id: &str) -> Option<f64> {
        let mut buckets = self.buckets.lock().unwrap_or_else(|e| e.into_inner());
        buckets.get_mut(tenant_id).map(|b| b.available_tokens())
    }

    /// Return the number of registered tenants.
    pub fn tenant_count(&self) -> usize {
        self.buckets.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Deregister a tenant.  Returns `true` if the tenant existed.
    pub fn deregister_tenant(&self, tenant_id: &str) -> bool {
        self.buckets
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(tenant_id)
            .is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admit_registered_tenant() {
        let ctrl = AdmissionController::new();
        ctrl.register_tenant("t1", SlaClass::Platinum);
        // Platinum has capacity=200; first few requests trivially succeed
        for _ in 0..10 {
            assert!(ctrl.try_admit("t1").is_ok());
        }
    }

    #[test]
    fn test_reject_unknown_tenant() {
        let ctrl = AdmissionController::new();
        let err = ctrl
            .try_admit("ghost")
            .expect_err("ghost is not registered");
        assert!(matches!(err, AdmissionError::TenantNotRegistered { .. }));
    }

    #[test]
    fn test_bronze_bucket_exhausts() {
        let ctrl = AdmissionController::new();
        ctrl.register_tenant("bronze", SlaClass::Bronze);

        // Bronze capacity = 5; drain it then check rejection
        let mut admitted = 0usize;
        let mut rejected = 0usize;
        for _ in 0..10 {
            if ctrl.try_admit("bronze").is_ok() {
                admitted += 1;
            } else {
                rejected += 1;
            }
        }
        assert!(admitted > 0, "should admit at least the first few");
        assert!(
            rejected > 0,
            "should eventually reject once tokens depleted"
        );
    }

    #[test]
    fn test_deregister_tenant() {
        let ctrl = AdmissionController::new();
        ctrl.register_tenant("t2", SlaClass::Gold);
        assert_eq!(ctrl.tenant_count(), 1);
        assert!(ctrl.deregister_tenant("t2"));
        assert_eq!(ctrl.tenant_count(), 0);
        let err = ctrl
            .try_admit("t2")
            .expect_err("t2 was deregistered, should reject");
        assert!(matches!(err, AdmissionError::TenantNotRegistered { .. }));
    }

    #[test]
    fn test_sla_class_query() {
        let ctrl = AdmissionController::new();
        ctrl.register_tenant("s", SlaClass::Silver);
        assert_eq!(ctrl.sla_class("s"), Some(SlaClass::Silver));
        assert_eq!(ctrl.sla_class("nonexistent"), None);
    }

    #[test]
    fn test_custom_cost_admit() {
        let ctrl = AdmissionController::new();
        ctrl.register_tenant("gold", SlaClass::Gold);
        // Gold capacity = 50; consume in one big gulp
        assert!(ctrl.try_admit_with_cost("gold", 45.0).is_ok());
        // Only 5 tokens left; a cost-of-10 should fail
        assert!(ctrl.try_admit_with_cost("gold", 10.0).is_err());
    }
}
