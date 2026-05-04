//! Compatibility shim — the admission controller has moved to
//! [`oxirs_core::sla::admission_controller`].
//!
//! The `From<AdmissionError> for MultiTenancyError` conversion stays here
//! (rather than in `oxirs-core`) because [`MultiTenancyError`] is local to
//! `oxirs-vec` and Rust's orphan rule forbids implementing a foreign trait for
//! a foreign error.

use crate::multi_tenancy::types::MultiTenancyError;

pub use oxirs_core::sla::{AdmissionController, AdmissionError};

impl From<AdmissionError> for MultiTenancyError {
    fn from(err: AdmissionError) -> Self {
        match err {
            AdmissionError::RateLimitExceeded { tenant_id } => {
                MultiTenancyError::RateLimitExceeded { tenant_id }
            }
            AdmissionError::TenantNotRegistered { tenant_id } => {
                MultiTenancyError::TenantNotFound { tenant_id }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::sla::SlaClass;

    #[test]
    fn test_admission_error_converts_to_multi_tenancy_error() {
        let err = AdmissionError::RateLimitExceeded {
            tenant_id: "x".into(),
        };
        let mt_err: MultiTenancyError = err.into();
        assert!(matches!(
            mt_err,
            MultiTenancyError::RateLimitExceeded { .. }
        ));

        let err = AdmissionError::TenantNotRegistered {
            tenant_id: "y".into(),
        };
        let mt_err: MultiTenancyError = err.into();
        assert!(matches!(mt_err, MultiTenancyError::TenantNotFound { .. }));
    }

    #[test]
    fn test_admission_controller_re_exported() {
        // Smoke test to confirm the re-exported types behave correctly through
        // the shim path.
        let ctrl = AdmissionController::new();
        ctrl.register_tenant("vec_tenant", SlaClass::Silver);
        assert!(ctrl.try_admit("vec_tenant").is_ok());
    }
}
