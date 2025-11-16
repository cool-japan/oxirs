//! Annotation lifecycle management for RDF-star
//!
//! This module provides comprehensive lifecycle management for annotations,
//! tracking states, transitions, approvals, and archival processes.
//!
//! # Features
//!
//! - **State management** - Draft, Active, Deprecated, Archived states
//! - **Transition workflows** - Approval workflows with validation
//! - **Audit trails** - Complete history of state changes
//! - **Retention policies** - Automatic archival and deletion
//! - **Approval chains** - Multi-level approval processes
//! - **Version control** - Track annotation evolution
//!
//! # Lifecycle States
//!
//! ```text
//! Draft → Review → Active → Deprecated → Archived → Deleted
//!   ↓       ↓        ↓         ↓           ↓
//! Reject  Reject  Update   Archive     Delete
//! ```
//!
//! # Examples
//!
//! ```rust
//! use oxirs_star::annotation_lifecycle::{LifecycleManager, AnnotationState};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut manager = LifecycleManager::new();
//!
//! // Create annotation in draft state
//! // let id = manager.create_draft(annotation)?;
//!
//! // Submit for review
//! // manager.submit_for_review(id, "user1")?;
//!
//! // Approve and activate
//! // manager.approve(id, "approver1")?;
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, span, Level};

use crate::StarResult;

/// Annotation lifecycle state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnnotationState {
    /// Initial draft state
    Draft,
    /// Submitted for review
    UnderReview,
    /// Active and in use
    Active,
    /// Deprecated but still accessible
    Deprecated,
    /// Archived (read-only)
    Archived,
    /// Marked for deletion
    PendingDeletion,
    /// Deleted (tombstone)
    Deleted,
    /// Rejected during review
    Rejected,
}

/// State transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Previous state
    pub from_state: AnnotationState,

    /// New state
    pub to_state: AnnotationState,

    /// Transition timestamp
    pub timestamp: DateTime<Utc>,

    /// User who initiated transition
    pub initiated_by: String,

    /// Reason for transition
    pub reason: Option<String>,

    /// Approval chain (if applicable)
    pub approvals: Vec<Approval>,
}

/// Approval record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Approval {
    /// Approver identifier
    pub approver: String,

    /// Approval timestamp
    pub timestamp: DateTime<Utc>,

    /// Approval decision
    pub approved: bool,

    /// Comments
    pub comments: Option<String>,
}

/// Lifecycle metadata for an annotation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleMetadata {
    /// Unique lifecycle ID
    pub lifecycle_id: String,

    /// Current state
    pub current_state: AnnotationState,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Created by
    pub created_by: String,

    /// Last modified timestamp
    pub last_modified: DateTime<Utc>,

    /// Last modified by
    pub last_modified_by: String,

    /// State transition history
    pub transitions: Vec<StateTransition>,

    /// Scheduled archival date
    pub archival_date: Option<DateTime<Utc>>,

    /// Scheduled deletion date
    pub deletion_date: Option<DateTime<Utc>>,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Version number
    pub version: u32,
}

/// Retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Deprecate after this duration
    pub deprecation_period: Option<Duration>,

    /// Archive after this duration
    pub archival_period: Option<Duration>,

    /// Delete after this duration
    pub deletion_period: Option<Duration>,

    /// Require approval before deletion
    pub require_deletion_approval: bool,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            deprecation_period: Some(Duration::days(365)), // 1 year
            archival_period: Some(Duration::days(730)),    // 2 years
            deletion_period: Some(Duration::days(1825)),   // 5 years
            require_deletion_approval: true,
        }
    }
}

/// Lifecycle manager
pub struct LifecycleManager {
    /// Lifecycle metadata indexed by annotation ID
    metadata: HashMap<String, LifecycleMetadata>,

    /// Retention policy
    retention_policy: RetentionPolicy,

    /// Approval requirements by state transition
    approval_requirements: HashMap<(AnnotationState, AnnotationState), usize>,

    /// Statistics
    stats: LifecycleStatistics,
}

/// Statistics for lifecycle operations
#[derive(Debug, Clone, Default)]
pub struct LifecycleStatistics {
    /// Total annotations managed
    pub total_annotations: usize,

    /// By state counts
    pub draft_count: usize,
    pub under_review_count: usize,
    pub active_count: usize,
    pub deprecated_count: usize,
    pub archived_count: usize,
    pub pending_deletion_count: usize,
    pub deleted_count: usize,
    pub rejected_count: usize,

    /// Transition counts
    pub total_transitions: usize,
    pub approvals_granted: usize,
    pub approvals_rejected: usize,
}

impl LifecycleManager {
    /// Create a new lifecycle manager
    pub fn new() -> Self {
        let mut approval_requirements = HashMap::new();

        // Define approval requirements for state transitions
        approval_requirements.insert((AnnotationState::Draft, AnnotationState::Active), 1);
        approval_requirements.insert((AnnotationState::UnderReview, AnnotationState::Active), 1);
        approval_requirements.insert((AnnotationState::Active, AnnotationState::Deprecated), 0);
        approval_requirements.insert((AnnotationState::Deprecated, AnnotationState::Archived), 0);
        approval_requirements.insert(
            (AnnotationState::Archived, AnnotationState::PendingDeletion),
            1,
        );

        Self {
            metadata: HashMap::new(),
            retention_policy: RetentionPolicy::default(),
            approval_requirements,
            stats: LifecycleStatistics::default(),
        }
    }

    /// Create annotation in draft state
    pub fn create_draft(
        &mut self,
        annotation_id: String,
        created_by: String,
    ) -> StarResult<String> {
        let span = span!(Level::DEBUG, "create_draft");
        let _enter = span.enter();

        let lifecycle_id = format!("lifecycle_{}", uuid::Uuid::new_v4());

        let metadata = LifecycleMetadata {
            lifecycle_id: lifecycle_id.clone(),
            current_state: AnnotationState::Draft,
            created_at: Utc::now(),
            created_by,
            last_modified: Utc::now(),
            last_modified_by: "system".to_string(),
            transitions: Vec::new(),
            archival_date: None,
            deletion_date: None,
            tags: Vec::new(),
            version: 1,
        };

        self.metadata.insert(annotation_id, metadata);
        self.stats.total_annotations += 1;
        self.stats.draft_count += 1;

        debug!("Created annotation lifecycle: {}", lifecycle_id);
        Ok(lifecycle_id)
    }

    /// Submit annotation for review
    pub fn submit_for_review(&mut self, annotation_id: &str, submitted_by: &str) -> StarResult<()> {
        self.transition_state(
            annotation_id,
            AnnotationState::UnderReview,
            submitted_by,
            Some("Submitted for review"),
        )
    }

    /// Approve annotation
    pub fn approve(
        &mut self,
        annotation_id: &str,
        approver: &str,
        comments: Option<String>,
    ) -> StarResult<()> {
        let metadata = self
            .metadata
            .get_mut(annotation_id)
            .ok_or_else(|| crate::StarError::invalid_quoted_triple("Annotation not found"))?;

        // Add approval
        if let Some(last_transition) = metadata.transitions.last_mut() {
            last_transition.approvals.push(Approval {
                approver: approver.to_string(),
                timestamp: Utc::now(),
                approved: true,
                comments,
            });

            self.stats.approvals_granted += 1;

            // Check if enough approvals
            let required = self
                .approval_requirements
                .get(&(last_transition.from_state, last_transition.to_state))
                .copied()
                .unwrap_or(0);

            if last_transition.approvals.len() >= required {
                // Transition to active
                self.transition_state(annotation_id, AnnotationState::Active, approver, None)?;
            }
        }

        Ok(())
    }

    /// Reject annotation
    pub fn reject(
        &mut self,
        annotation_id: &str,
        rejector: &str,
        reason: Option<String>,
    ) -> StarResult<()> {
        let metadata = self
            .metadata
            .get_mut(annotation_id)
            .ok_or_else(|| crate::StarError::invalid_quoted_triple("Annotation not found"))?;

        if let Some(last_transition) = metadata.transitions.last_mut() {
            last_transition.approvals.push(Approval {
                approver: rejector.to_string(),
                timestamp: Utc::now(),
                approved: false,
                comments: reason.clone(),
            });

            self.stats.approvals_rejected += 1;
        }

        self.transition_state(
            annotation_id,
            AnnotationState::Rejected,
            rejector,
            reason.as_deref(),
        )
    }

    /// Deprecate annotation
    pub fn deprecate(
        &mut self,
        annotation_id: &str,
        deprecated_by: &str,
        reason: Option<String>,
    ) -> StarResult<()> {
        self.transition_state(
            annotation_id,
            AnnotationState::Deprecated,
            deprecated_by,
            reason.as_deref(),
        )
    }

    /// Archive annotation
    pub fn archive(&mut self, annotation_id: &str, archived_by: &str) -> StarResult<()> {
        self.transition_state(
            annotation_id,
            AnnotationState::Archived,
            archived_by,
            Some("Archived"),
        )
    }

    /// Mark for deletion
    pub fn mark_for_deletion(&mut self, annotation_id: &str, requested_by: &str) -> StarResult<()> {
        self.transition_state(
            annotation_id,
            AnnotationState::PendingDeletion,
            requested_by,
            Some("Marked for deletion"),
        )
    }

    /// Delete annotation (tombstone)
    pub fn delete(&mut self, annotation_id: &str, deleted_by: &str) -> StarResult<()> {
        self.transition_state(
            annotation_id,
            AnnotationState::Deleted,
            deleted_by,
            Some("Deleted"),
        )
    }

    /// Transition to a new state
    fn transition_state(
        &mut self,
        annotation_id: &str,
        new_state: AnnotationState,
        initiated_by: &str,
        reason: Option<&str>,
    ) -> StarResult<()> {
        let span = span!(Level::DEBUG, "transition_state");
        let _enter = span.enter();

        // Get old state first (without holding mutable borrow)
        let old_state = self
            .metadata
            .get(annotation_id)
            .ok_or_else(|| crate::StarError::invalid_quoted_triple("Annotation not found"))?
            .current_state;

        // Validate transition
        self.validate_transition(old_state, new_state)?;

        // Calculate archival date if needed
        let archival_date = if new_state == AnnotationState::Active {
            self.retention_policy
                .deprecation_period
                .map(|d| Utc::now() + d)
        } else {
            None
        };

        // Update metadata (scoped to drop borrow before calling update_state_counts)
        {
            let metadata = self
                .metadata
                .get_mut(annotation_id)
                .ok_or_else(|| crate::StarError::invalid_quoted_triple("Annotation not found"))?;

            // Create transition record
            let transition = StateTransition {
                from_state: old_state,
                to_state: new_state,
                timestamp: Utc::now(),
                initiated_by: initiated_by.to_string(),
                reason: reason.map(|s| s.to_string()),
                approvals: Vec::new(),
            };

            metadata.transitions.push(transition);
            metadata.current_state = new_state;
            metadata.last_modified = Utc::now();
            metadata.last_modified_by = initiated_by.to_string();

            // Set archival date if needed
            if let Some(date) = archival_date {
                metadata.archival_date = Some(date);
            }
        }

        // Update statistics (after dropping metadata borrow)
        self.update_state_counts(old_state, new_state);
        self.stats.total_transitions += 1;

        debug!(
            "Transitioned annotation {} from {:?} to {:?}",
            annotation_id, old_state, new_state
        );

        Ok(())
    }

    fn validate_transition(&self, from: AnnotationState, to: AnnotationState) -> StarResult<()> {
        // Define allowed transitions
        let allowed = match (from, to) {
            (AnnotationState::Draft, AnnotationState::UnderReview) => true,
            (AnnotationState::Draft, AnnotationState::Active) => true,
            (AnnotationState::UnderReview, AnnotationState::Active) => true,
            (AnnotationState::UnderReview, AnnotationState::Rejected) => true,
            (AnnotationState::Active, AnnotationState::Deprecated) => true,
            (AnnotationState::Active, AnnotationState::Archived) => true,
            (AnnotationState::Deprecated, AnnotationState::Archived) => true,
            (AnnotationState::Deprecated, AnnotationState::Active) => true, // Reactivate
            (AnnotationState::Archived, AnnotationState::PendingDeletion) => true,
            (AnnotationState::PendingDeletion, AnnotationState::Deleted) => true,
            (AnnotationState::PendingDeletion, AnnotationState::Archived) => true, // Cancel deletion
            _ => false,
        };

        if !allowed {
            return Err(crate::StarError::invalid_quoted_triple(format!(
                "Invalid state transition from {:?} to {:?}",
                from, to
            )));
        }

        Ok(())
    }

    fn update_state_counts(&mut self, old_state: AnnotationState, new_state: AnnotationState) {
        // Decrement old state
        match old_state {
            AnnotationState::Draft => {
                self.stats.draft_count = self.stats.draft_count.saturating_sub(1)
            }
            AnnotationState::UnderReview => {
                self.stats.under_review_count = self.stats.under_review_count.saturating_sub(1)
            }
            AnnotationState::Active => {
                self.stats.active_count = self.stats.active_count.saturating_sub(1)
            }
            AnnotationState::Deprecated => {
                self.stats.deprecated_count = self.stats.deprecated_count.saturating_sub(1)
            }
            AnnotationState::Archived => {
                self.stats.archived_count = self.stats.archived_count.saturating_sub(1)
            }
            AnnotationState::PendingDeletion => {
                self.stats.pending_deletion_count =
                    self.stats.pending_deletion_count.saturating_sub(1)
            }
            AnnotationState::Deleted => {
                self.stats.deleted_count = self.stats.deleted_count.saturating_sub(1)
            }
            AnnotationState::Rejected => {
                self.stats.rejected_count = self.stats.rejected_count.saturating_sub(1)
            }
        }

        // Increment new state
        match new_state {
            AnnotationState::Draft => self.stats.draft_count += 1,
            AnnotationState::UnderReview => self.stats.under_review_count += 1,
            AnnotationState::Active => self.stats.active_count += 1,
            AnnotationState::Deprecated => self.stats.deprecated_count += 1,
            AnnotationState::Archived => self.stats.archived_count += 1,
            AnnotationState::PendingDeletion => self.stats.pending_deletion_count += 1,
            AnnotationState::Deleted => self.stats.deleted_count += 1,
            AnnotationState::Rejected => self.stats.rejected_count += 1,
        }
    }

    /// Apply retention policies (scheduled task)
    pub fn apply_retention_policies(&mut self) -> StarResult<Vec<String>> {
        let span = span!(Level::INFO, "apply_retention_policies");
        let _enter = span.enter();

        let mut processed = Vec::new();
        let now = Utc::now();

        for (annotation_id, metadata) in &self.metadata {
            // Check for scheduled archival
            if let Some(archival_date) = metadata.archival_date {
                if now >= archival_date && metadata.current_state == AnnotationState::Active {
                    processed.push(annotation_id.clone());
                }
            }

            // Check for scheduled deletion
            if let Some(deletion_date) = metadata.deletion_date {
                if now >= deletion_date && metadata.current_state == AnnotationState::Archived {
                    processed.push(annotation_id.clone());
                }
            }
        }

        info!(
            "Applied retention policies to {} annotations",
            processed.len()
        );

        Ok(processed)
    }

    /// Get lifecycle metadata
    pub fn get_metadata(&self, annotation_id: &str) -> Option<&LifecycleMetadata> {
        self.metadata.get(annotation_id)
    }

    /// Get statistics
    pub fn statistics(&self) -> &LifecycleStatistics {
        &self.stats
    }

    /// Get annotations by state
    pub fn get_by_state(&self, state: AnnotationState) -> Vec<String> {
        self.metadata
            .iter()
            .filter(|(_, meta)| meta.current_state == state)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Set retention policy
    pub fn set_retention_policy(&mut self, policy: RetentionPolicy) {
        self.retention_policy = policy;
    }
}

impl Default for LifecycleManager {
    fn default() -> Self {
        Self::new()
    }
}

// Use uuid crate for generating unique IDs
mod uuid {
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(1);

    pub struct Uuid;

    impl Uuid {
        pub fn new_v4() -> String {
            let counter = COUNTER.fetch_add(1, Ordering::SeqCst);
            format!("{:016x}", counter)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_draft() {
        let mut manager = LifecycleManager::new();

        let lifecycle_id = manager
            .create_draft("ann1".to_string(), "user1".to_string())
            .unwrap();

        assert!(!lifecycle_id.is_empty());
        assert_eq!(manager.statistics().draft_count, 1);
    }

    #[test]
    fn test_submit_for_review() {
        let mut manager = LifecycleManager::new();

        manager
            .create_draft("ann1".to_string(), "user1".to_string())
            .unwrap();
        manager.submit_for_review("ann1", "user1").unwrap();

        let metadata = manager.get_metadata("ann1").unwrap();
        assert_eq!(metadata.current_state, AnnotationState::UnderReview);
    }

    #[test]
    fn test_approve() {
        let mut manager = LifecycleManager::new();

        manager
            .create_draft("ann1".to_string(), "user1".to_string())
            .unwrap();
        manager.submit_for_review("ann1", "user1").unwrap();
        manager.approve("ann1", "approver1", None).unwrap();

        let metadata = manager.get_metadata("ann1").unwrap();
        assert_eq!(metadata.current_state, AnnotationState::Active);
        assert_eq!(manager.statistics().active_count, 1);
    }

    #[test]
    fn test_reject() {
        let mut manager = LifecycleManager::new();

        manager
            .create_draft("ann1".to_string(), "user1".to_string())
            .unwrap();
        manager.submit_for_review("ann1", "user1").unwrap();
        manager
            .reject("ann1", "approver1", Some("Not ready".to_string()))
            .unwrap();

        let metadata = manager.get_metadata("ann1").unwrap();
        assert_eq!(metadata.current_state, AnnotationState::Rejected);
    }

    #[test]
    fn test_full_lifecycle() {
        let mut manager = LifecycleManager::new();

        // Create
        manager
            .create_draft("ann1".to_string(), "user1".to_string())
            .unwrap();

        // Review
        manager.submit_for_review("ann1", "user1").unwrap();

        // Approve
        manager.approve("ann1", "approver1", None).unwrap();

        // Deprecate
        manager
            .deprecate("ann1", "user1", Some("Outdated".to_string()))
            .unwrap();

        // Archive
        manager.archive("ann1", "user1").unwrap();

        // Mark for deletion
        manager.mark_for_deletion("ann1", "admin").unwrap();

        // Delete
        manager.delete("ann1", "admin").unwrap();

        let metadata = manager.get_metadata("ann1").unwrap();
        assert_eq!(metadata.current_state, AnnotationState::Deleted);
        assert_eq!(metadata.transitions.len(), 6); // 6 transitions (create doesn't count)
    }

    #[test]
    fn test_invalid_transition() {
        let mut manager = LifecycleManager::new();

        manager
            .create_draft("ann1".to_string(), "user1".to_string())
            .unwrap();

        // Try to delete directly from draft (invalid)
        let result = manager.delete("ann1", "user1");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_by_state() {
        let mut manager = LifecycleManager::new();

        manager
            .create_draft("ann1".to_string(), "user1".to_string())
            .unwrap();
        manager
            .create_draft("ann2".to_string(), "user1".to_string())
            .unwrap();
        manager
            .create_draft("ann3".to_string(), "user1".to_string())
            .unwrap();

        manager.submit_for_review("ann1", "user1").unwrap();

        let drafts = manager.get_by_state(AnnotationState::Draft);
        assert_eq!(drafts.len(), 2);

        let under_review = manager.get_by_state(AnnotationState::UnderReview);
        assert_eq!(under_review.len(), 1);
    }

    #[test]
    fn test_reactivate_deprecated() {
        let mut manager = LifecycleManager::new();

        manager
            .create_draft("ann1".to_string(), "user1".to_string())
            .unwrap();
        manager.submit_for_review("ann1", "user1").unwrap();
        manager.approve("ann1", "approver1", None).unwrap();
        manager.deprecate("ann1", "user1", None).unwrap();

        // Reactivate
        manager
            .transition_state(
                "ann1",
                AnnotationState::Active,
                "user1",
                Some("Reactivated"),
            )
            .unwrap();

        let metadata = manager.get_metadata("ann1").unwrap();
        assert_eq!(metadata.current_state, AnnotationState::Active);
    }
}
