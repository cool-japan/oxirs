//! SOC2/GDPR-compliant structured audit trail for OxiRS.
//!
//! This module provides a complete audit logging framework suitable for
//! enterprise deployments that require SOC2 Type II or GDPR compliance:
//!
//! - [`AuditEvent`] — the core immutable record type.
//! - [`AuditLogger`] / [`InMemoryAuditLogger`] / [`JsonLineAuditLogger`] /
//!   [`CompositeAuditLogger`] — pluggable sinks.
//! - [`AuditFilter`] / [`AuditQuery`] / [`AuditQueryable`] — structured
//!   query and pagination over event collections.
//! - [`GdprService`] / [`DataSubjectReport`] — Article 15 (subject access)
//!   and Article 17 (right to erasure / pseudonymisation) operations.
//!
//! # Quick Start
//!
//! ```rust
//! use oxirs_core::audit::{
//!     AuditEvent, AuditEventKind, AuditOutcome, AuditActor, AuditResource,
//!     AuditLogger, InMemoryAuditLogger,
//! };
//! use oxirs_core::audit::event::ActorType;
//!
//! let logger = InMemoryAuditLogger::new();
//!
//! let actor = AuditActor {
//!     actor_id: "user-123".to_string(),
//!     actor_type: ActorType::User,
//!     ip_address: Some("10.0.0.1".to_string()),
//!     session_id: Some("sess-abc".to_string()),
//! };
//!
//! let resource = AuditResource {
//!     resource_type: "dataset".to_string(),
//!     resource_id: "ds-main".to_string(),
//!     tenant_id: Some("acme".to_string()),
//! };
//!
//! let event = AuditEvent::new(
//!     AuditEventKind::DataAccess,
//!     "sparql.select",
//!     actor,
//!     resource,
//!     AuditOutcome::Success,
//! )
//! .with_duration(42)
//! .with_metadata("rows_returned", "150");
//!
//! logger.log(event).expect("log failed");
//! assert_eq!(logger.len(), 1);
//! ```

pub mod event;
pub mod filter;
pub mod gdpr;
pub mod logger;

pub use event::{ActorType, AuditActor, AuditEvent, AuditEventKind, AuditOutcome, AuditResource};
pub use filter::{AuditFilter, AuditQuery, AuditQueryable, SortOrder};
pub use gdpr::{DataSubjectReport, GdprService};
pub use logger::{
    AuditLogError, AuditLogger, CompositeAuditLogger, InMemoryAuditLogger, JsonLineAuditLogger,
};
