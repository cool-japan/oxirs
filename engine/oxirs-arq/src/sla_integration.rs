//! SLA-aware admission control integration for the ARQ query executor.
//!
//! [`ArqSlaGate`] is the entry point that wraps [`oxirs_core::sla::AdmissionController`]
//! and [`oxirs_core::sla::PriorityDispatcher`] with ARQ-specific logic:
//!
//! * Translates rejected admission attempts into typed [`ArqSlaError`] values.
//! * Looks up tenant SLA classes via [`TenantConfigRegistry`].
//! * Applies per-tenant overrides (custom token cost, soft-vs-strict mode).
//! * Provides a synchronous "admit and execute" wrapper that funnels work
//!   through the priority dispatcher.
//!
//! The gate is designed to be wired into [`crate::executor::QueryExecutor`]
//! before each `execute()` call.  See `tests/sla_admission.rs` for an
//! end-to-end integration example.

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use oxirs_core::sla::{
    AdmissionController, AdmissionError, PrioritizedQuery, PriorityDispatcher, SlaClass,
};
use thiserror::Error;

use crate::tenant_config::TenantConfigRegistry;

// ─────────────────────────────────────────────────────────────────────────────
// ArqSlaError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors raised by [`ArqSlaGate`] when a query cannot proceed.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ArqSlaError {
    /// The tenant has not been registered with the gate.
    #[error("tenant '{tenant_id}' is not registered with the ARQ SLA gate")]
    TenantNotRegistered {
        /// Identifier of the unknown tenant.
        tenant_id: String,
    },

    /// The tenant's token bucket is exhausted.
    #[error("SLA exceeded for tenant '{tenant_id}' (class {class}): {detail}")]
    SlaExceeded {
        /// Identifier of the tenant whose budget is exhausted.
        tenant_id: String,
        /// SLA tier that was being enforced.
        class: SlaClassDisplay,
        /// Free-form description of the rejection reason.
        detail: String,
    },

    /// A queued query was dropped or the dispatcher was shut down.
    #[error("priority dispatcher closed for tenant '{tenant_id}'")]
    DispatcherClosed {
        /// Identifier of the tenant whose query was dropped.
        tenant_id: String,
    },
}

/// Wrapper around [`SlaClass`] that gives `Display` and PEq impls suitable
/// for error formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlaClassDisplay(pub SlaClass);

impl fmt::Display for SlaClassDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.name())
    }
}

impl From<SlaClass> for SlaClassDisplay {
    fn from(class: SlaClass) -> Self {
        SlaClassDisplay(class)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdmittedQuery
// ─────────────────────────────────────────────────────────────────────────────

/// Hand-back struct returned by [`ArqSlaGate::admit`] on a successful gate
/// pass.  The caller consumes the inner payload after performing the query.
#[derive(Debug)]
pub struct AdmittedQuery<P> {
    /// Tenant the work belongs to.
    pub tenant_id: String,
    /// SLA class enforced by the gate.
    pub sla_class: SlaClass,
    /// Per-query timeout (if the tenant config specified one).
    pub timeout: Option<Duration>,
    /// The payload (typically the algebra to execute).
    pub payload: P,
}

// ─────────────────────────────────────────────────────────────────────────────
// ArqSlaGate
// ─────────────────────────────────────────────────────────────────────────────

/// Front door for ARQ tenant admission control.
///
/// Internally wraps:
///
/// * an [`AdmissionController`] (token-bucket gate),
/// * a [`PriorityDispatcher`] (max-heap by SLA tier), and
/// * the [`TenantConfigRegistry`] that drives both.
///
/// The gate is `Clone`-cheap; share one instance across executor threads.
#[derive(Clone)]
pub struct ArqSlaGate {
    registry: TenantConfigRegistry,
    controller: AdmissionController,
    dispatcher: Arc<Mutex<PriorityDispatcher<DispatcherSlot>>>,
}

#[derive(Debug)]
struct DispatcherSlot {
    label: String,
}

impl ArqSlaGate {
    /// Build a gate driven by the given registry.
    ///
    /// Tenants present in the registry at construction time are pre-registered
    /// with the underlying [`AdmissionController`] so the very first admission
    /// attempt for each tenant has a fresh full bucket.
    pub fn new(registry: TenantConfigRegistry) -> Self {
        let controller = AdmissionController::new();
        for tenant_id in registry.tenant_ids() {
            if let Some(class) = registry.sla_class(&tenant_id) {
                controller.register_tenant(&tenant_id, class);
            }
        }
        Self {
            registry,
            controller,
            dispatcher: Arc::new(Mutex::new(PriorityDispatcher::new())),
        }
    }

    /// Register a tenant on the fly (forwards into both the registry and the
    /// admission controller).
    pub fn register_tenant(&self, config: crate::tenant_config::TenantConfig) {
        self.controller
            .register_tenant(&config.tenant_id, config.sla_class);
        self.registry.register(config);
    }

    /// Convenience: peek at the snapshot of the underlying registry.
    pub fn registry(&self) -> &TenantConfigRegistry {
        &self.registry
    }

    /// Convenience: peek at the underlying admission controller.
    pub fn controller(&self) -> &AdmissionController {
        &self.controller
    }

    /// Try to admit a query for `tenant_id`.
    ///
    /// On acceptance, the payload is returned wrapped in an [`AdmittedQuery`].
    /// On rejection, the call returns either [`ArqSlaError::TenantNotRegistered`]
    /// or [`ArqSlaError::SlaExceeded`] depending on the reason.
    pub fn admit<P>(&self, tenant_id: &str, payload: P) -> Result<AdmittedQuery<P>, ArqSlaError> {
        let config =
            self.registry
                .get(tenant_id)
                .ok_or_else(|| ArqSlaError::TenantNotRegistered {
                    tenant_id: tenant_id.to_owned(),
                })?;

        match self
            .controller
            .try_admit_with_cost(tenant_id, config.max_query_cost)
        {
            Ok(()) => Ok(AdmittedQuery {
                tenant_id: tenant_id.to_owned(),
                sla_class: config.sla_class,
                timeout: config.query_timeout,
                payload,
            }),
            Err(err) => {
                if !config.strict_admission {
                    // Soft mode: pass through with the SLA tier still attached
                    // but no token consumed.  Useful for canary tenants.
                    Ok(AdmittedQuery {
                        tenant_id: tenant_id.to_owned(),
                        sla_class: config.sla_class,
                        timeout: config.query_timeout,
                        payload,
                    })
                } else {
                    Err(self.translate_admission_error(&config.tenant_id, config.sla_class, err))
                }
            }
        }
    }

    fn translate_admission_error(
        &self,
        tenant_id: &str,
        class: SlaClass,
        err: AdmissionError,
    ) -> ArqSlaError {
        match err {
            AdmissionError::RateLimitExceeded { .. } => ArqSlaError::SlaExceeded {
                tenant_id: tenant_id.to_owned(),
                class: class.into(),
                detail: "token bucket exhausted".to_owned(),
            },
            AdmissionError::TenantNotRegistered { .. } => {
                // Should not happen because admit() registers via registry,
                // but handle defensively.
                ArqSlaError::TenantNotRegistered {
                    tenant_id: tenant_id.to_owned(),
                }
            }
        }
    }

    /// Enqueue an admitted query into the priority dispatcher.
    ///
    /// Returns the SLA tier under which the query was enqueued.  Use
    /// [`Self::next_dispatch`] to drain the queue.
    pub fn enqueue<P>(&self, admitted: &AdmittedQuery<P>, label: impl Into<String>) {
        let mut dispatcher = self.dispatcher.lock().unwrap_or_else(|e| e.into_inner());
        dispatcher.enqueue(
            admitted.tenant_id.clone(),
            admitted.sla_class,
            DispatcherSlot {
                label: label.into(),
            },
        );
    }

    /// Pop the next pending query (highest priority first).
    ///
    /// Returns `None` if the dispatcher is empty.  This API is sufficient for
    /// a single-thread executor; multi-threaded callers should pair it with
    /// their own work-stealing loop.
    pub fn next_dispatch(&self) -> Option<DispatchedQuery> {
        let mut dispatcher = self.dispatcher.lock().unwrap_or_else(|e| e.into_inner());
        dispatcher.dequeue().map(
            |prioritized: PrioritizedQuery<DispatcherSlot>| DispatchedQuery {
                tenant_id: prioritized.tenant_id,
                priority: prioritized.priority,
                label: prioritized.payload.label,
            },
        )
    }

    /// Number of queries currently waiting in the dispatcher.
    pub fn pending_count(&self) -> usize {
        let dispatcher = self.dispatcher.lock().unwrap_or_else(|e| e.into_inner());
        dispatcher.len()
    }
}

/// Outcome of [`ArqSlaGate::next_dispatch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchedQuery {
    /// Tenant that submitted the query.
    pub tenant_id: String,
    /// Numeric SLA priority used by the dispatcher.
    pub priority: u8,
    /// Free-form label set by the caller of [`ArqSlaGate::enqueue`].
    pub label: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tenant_config::TenantConfig;

    fn registry_with(tenants: &[(&str, SlaClass)]) -> TenantConfigRegistry {
        let registry = TenantConfigRegistry::new();
        for (id, class) in tenants {
            registry.register(TenantConfig::new(*id, *class));
        }
        registry
    }

    #[test]
    fn test_admit_unregistered_tenant_fails() {
        let gate = ArqSlaGate::new(TenantConfigRegistry::new());
        let err = gate
            .admit("ghost", "select * { ?s ?p ?o }".to_string())
            .expect_err("unregistered tenants must be rejected");
        assert!(matches!(err, ArqSlaError::TenantNotRegistered { .. }));
    }

    #[test]
    fn test_admit_registered_tenant_succeeds() {
        let registry = registry_with(&[("alpha", SlaClass::Platinum)]);
        let gate = ArqSlaGate::new(registry);
        let admitted = gate
            .admit("alpha", "query".to_string())
            .expect("Platinum bucket should admit first request");
        assert_eq!(admitted.tenant_id, "alpha");
        assert_eq!(admitted.sla_class, SlaClass::Platinum);
        assert_eq!(admitted.payload, "query");
    }

    #[test]
    fn test_bronze_exhaustion_returns_sla_exceeded() {
        let registry = registry_with(&[("budget", SlaClass::Bronze)]);
        let gate = ArqSlaGate::new(registry);

        // Bronze capacity = 5, max_query_cost = 1 by default.  Drain it.
        let mut last_err: Option<ArqSlaError> = None;
        for _ in 0..50 {
            if let Err(err) = gate.admit("budget", ()) {
                last_err = Some(err);
                break;
            }
        }
        let err = last_err.expect("Bronze tenant must eventually be rejected");
        assert!(
            matches!(err, ArqSlaError::SlaExceeded { ref class, .. } if class.0 == SlaClass::Bronze)
        );
    }

    #[test]
    fn test_soft_admission_passes_through_after_exhaustion() {
        let registry = TenantConfigRegistry::new();
        registry
            .register(TenantConfig::new("canary", SlaClass::Bronze).with_strict_admission(false));
        let gate = ArqSlaGate::new(registry);

        // Drain the bucket
        for _ in 0..20 {
            let _ = gate.admit("canary", ()).expect("soft admission must pass");
        }
    }

    #[test]
    fn test_dispatcher_orders_by_sla() {
        let registry = registry_with(&[
            ("free", SlaClass::Bronze),
            ("paid", SlaClass::Gold),
            ("vip", SlaClass::Platinum),
        ]);
        let gate = ArqSlaGate::new(registry);

        let bronze = gate.admit("free", "b").expect("admit bronze");
        let gold = gate.admit("paid", "g").expect("admit gold");
        let platinum = gate.admit("vip", "p").expect("admit platinum");

        gate.enqueue(&bronze, "bronze-job");
        gate.enqueue(&gold, "gold-job");
        gate.enqueue(&platinum, "platinum-job");
        assert_eq!(gate.pending_count(), 3);

        let first = gate
            .next_dispatch()
            .expect("dispatcher must yield platinum first");
        assert_eq!(first.tenant_id, "vip");
        assert_eq!(first.label, "platinum-job");

        let second = gate.next_dispatch().expect("then gold");
        assert_eq!(second.tenant_id, "paid");

        let third = gate.next_dispatch().expect("then bronze");
        assert_eq!(third.tenant_id, "free");

        assert!(gate.next_dispatch().is_none());
    }

    #[test]
    fn test_register_tenant_via_gate() {
        let gate = ArqSlaGate::new(TenantConfigRegistry::new());
        gate.register_tenant(TenantConfig::new("late", SlaClass::Gold));
        let admitted = gate
            .admit("late", "q".to_string())
            .expect("late-registered tenant must admit");
        assert_eq!(admitted.sla_class, SlaClass::Gold);
    }

    #[test]
    fn test_max_query_cost_drains_bucket_faster() {
        let registry = TenantConfigRegistry::new();
        registry.register(TenantConfig::new("greedy", SlaClass::Silver).with_max_query_cost(10.0));
        let gate = ArqSlaGate::new(registry);

        // Silver capacity = 20.  At cost = 10, two should pass and the third
        // should be rejected.
        assert!(gate.admit("greedy", ()).is_ok());
        assert!(gate.admit("greedy", ()).is_ok());
        let err = gate
            .admit("greedy", ())
            .expect_err("third high-cost call must be rejected");
        assert!(matches!(err, ArqSlaError::SlaExceeded { .. }));
    }

    #[test]
    fn test_query_timeout_propagates() {
        let registry = TenantConfigRegistry::new();
        registry.register(
            TenantConfig::new("timed", SlaClass::Gold)
                .with_query_timeout(Duration::from_millis(750)),
        );
        let gate = ArqSlaGate::new(registry);
        let admitted = gate.admit("timed", ()).expect("admit");
        assert_eq!(admitted.timeout, Some(Duration::from_millis(750)));
    }

    #[test]
    fn test_dispatcher_pending_count_tracks_enqueue_dequeue() {
        let registry = registry_with(&[("a", SlaClass::Silver)]);
        let gate = ArqSlaGate::new(registry);
        let admitted = gate.admit("a", ()).expect("admit");
        gate.enqueue(&admitted, "j1");
        gate.enqueue(&admitted, "j2");
        assert_eq!(gate.pending_count(), 2);
        let _ = gate.next_dispatch();
        assert_eq!(gate.pending_count(), 1);
        let _ = gate.next_dispatch();
        assert_eq!(gate.pending_count(), 0);
    }

    #[test]
    fn test_sla_class_display_format() {
        let display: SlaClassDisplay = SlaClass::Platinum.into();
        assert_eq!(format!("{display}"), "platinum");
    }

    #[test]
    fn test_pre_registered_in_registry_admitted_immediately() {
        let registry = TenantConfigRegistry::new();
        registry.register(TenantConfig::new("preexisting", SlaClass::Gold));
        let gate = ArqSlaGate::new(registry);
        // First admission should succeed because gate's `new()` registered the
        // tenant with the AdmissionController during construction.
        let admitted = gate
            .admit("preexisting", ())
            .expect("pre-registered tenant must be in admission controller");
        assert_eq!(admitted.sla_class, SlaClass::Gold);
    }
}
