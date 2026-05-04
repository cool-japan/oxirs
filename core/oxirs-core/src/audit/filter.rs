//! Audit event filtering and query engine.

use std::cmp::Reverse;

use chrono::{DateTime, Utc};

use super::event::{AuditEvent, AuditEventKind};

/// Predicate that selects a subset of audit events.
///
/// All set fields are ANDed together. Unset fields match any value.
#[derive(Debug, Clone, Default)]
pub struct AuditFilter {
    /// Match only events of this kind.
    pub kind: Option<AuditEventKind>,
    /// Match only events whose `actor.actor_id` equals this value.
    pub actor_id: Option<String>,
    /// Match only events whose `resource.resource_id` equals this value.
    pub resource_id: Option<String>,
    /// Match only events whose `resource.tenant_id` equals this value.
    pub tenant_id: Option<String>,
    /// Match only events at or after this timestamp (inclusive).
    pub from: Option<DateTime<Utc>>,
    /// Match only events before or at this timestamp (inclusive).
    pub until: Option<DateTime<Utc>>,
    /// Match only events whose `data_subject_id` equals this value.
    pub data_subject_id: Option<String>,
    /// Match only events whose `action` starts with this prefix.
    ///
    /// For example, `"sparql."` matches `"sparql.select"` and `"sparql.update"`.
    pub action_prefix: Option<String>,
}

impl AuditFilter {
    /// Return `true` if `event` satisfies every set predicate.
    pub fn matches(&self, event: &AuditEvent) -> bool {
        if let Some(ref kind) = self.kind {
            if &event.kind != kind {
                return false;
            }
        }
        if let Some(ref actor_id) = self.actor_id {
            if &event.actor.actor_id != actor_id {
                return false;
            }
        }
        if let Some(ref resource_id) = self.resource_id {
            if &event.resource.resource_id != resource_id {
                return false;
            }
        }
        if let Some(ref tenant_id) = self.tenant_id {
            if event.resource.tenant_id.as_deref() != Some(tenant_id.as_str()) {
                return false;
            }
        }
        if let Some(from) = self.from {
            if event.timestamp < from {
                return false;
            }
        }
        if let Some(until) = self.until {
            if event.timestamp > until {
                return false;
            }
        }
        if let Some(ref subject) = self.data_subject_id {
            if event.data_subject_id.as_deref() != Some(subject.as_str()) {
                return false;
            }
        }
        if let Some(ref prefix) = self.action_prefix {
            if !event.action.starts_with(prefix.as_str()) {
                return false;
            }
        }
        true
    }
}

/// Sort direction for query results.
#[derive(Debug, Clone, Default)]
pub enum SortOrder {
    /// Oldest events first (ascending timestamp). This is the default.
    #[default]
    Ascending,
    /// Newest events first (descending timestamp).
    Descending,
}

/// A paginated, sorted query over a collection of audit events.
///
/// Processing order: **filter → sort → offset → limit**.
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    /// Predicate applied first to narrow the event set.
    pub filter: AuditFilter,
    /// Maximum number of events to return after offset is applied.
    pub limit: Option<usize>,
    /// Number of matching events to skip before returning results.
    pub offset: Option<usize>,
    /// Sort order applied before offset/limit pagination.
    pub sort: SortOrder,
}

/// Extension trait that adds structured querying to slices of audit events.
pub trait AuditQueryable {
    /// Execute an [`AuditQuery`] against this collection and return matching events.
    fn query(&self, query: &AuditQuery) -> Vec<AuditEvent>;
}

impl AuditQueryable for Vec<AuditEvent> {
    fn query(&self, query: &AuditQuery) -> Vec<AuditEvent> {
        // 1. Filter
        let mut matched: Vec<AuditEvent> = self
            .iter()
            .filter(|e| query.filter.matches(e))
            .cloned()
            .collect();

        // 2. Sort by timestamp
        match query.sort {
            SortOrder::Ascending => matched.sort_by_key(|e| e.timestamp),
            SortOrder::Descending => matched.sort_by_key(|e| Reverse(e.timestamp)),
        }

        // 3. Offset
        let start = query.offset.unwrap_or(0).min(matched.len());
        let sliced = matched.into_iter().skip(start);

        // 4. Limit
        match query.limit {
            Some(n) => sliced.take(n).collect(),
            None => sliced.collect(),
        }
    }
}
