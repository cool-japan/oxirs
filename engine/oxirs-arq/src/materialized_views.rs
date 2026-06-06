//! Materialized Views for Query Optimization (facade).
//!
//! This module provides comprehensive materialized view support including:
//! - View definition and storage
//! - Query rewriting to utilize materialized views
//! - Incremental view maintenance
//! - Cost-based view selection
//! - Automatic view recommendations
//!
//! The implementation is split across sibling modules (declared in `lib.rs`)
//! and re-exported here so the original public API is preserved:
//!
//! - [`materialized_views_types`](crate::materialized_views_types) — config,
//!   view, view-data, metadata, dependency, cost, and recommendation type
//!   definitions.
//! - [`materialized_views_storage`](crate::materialized_views_storage) —
//!   [`ViewStorage`] (two-tier memory + disk persistence), JSON-safe
//!   serialisation for [`ViewData`], and the [`QueryRewriter`] / [`ViewIndex`]
//!   used during query-rewrite matching.
//! - [`materialized_views_scheduler`](crate::materialized_views_scheduler) —
//!   [`MaintenanceScheduler`] task queue and [`ViewRecommendationEngine`]
//!   that proposes new views based on query patterns.
//! - [`materialized_views_manager`](crate::materialized_views_manager) —
//!   high-level [`MaterializedViewManager`] orchestrating view creation,
//!   update (incremental and full), usage statistics, and recommendation
//!   retrieval.

pub use crate::materialized_views_types::*;
// Storage / rewriter / scheduler / manager modules only contain `impl` blocks
// for the types defined in `materialized_views_types`, so they have no items
// to re-export here — but we still want them compiled into the crate.
#[allow(unused_imports)]
use crate::materialized_views_manager as _materialized_views_manager_impls;
#[allow(unused_imports)]
use crate::materialized_views_scheduler as _materialized_views_scheduler_impls;
#[allow(unused_imports)]
use crate::materialized_views_storage as _materialized_views_storage_impls;
