//! Data Residency and GDPR Compliance
//!
//! Enforces regional data placement rules and GDPR Article 44-49 compliance.

pub mod gdpr_compliance;
pub mod region_policy;

pub use gdpr_compliance::{
    Article49Derogation, GdprArticle, GdprComplianceChecker, Safeguard, TransferComplianceResult,
    TransferRecord,
};
pub use region_policy::{Region, ResidencyEnforcer, ResidencyPolicy};
