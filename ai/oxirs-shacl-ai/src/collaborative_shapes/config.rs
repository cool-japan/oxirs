//! Configuration for collaborative shape development

use serde::{Deserialize, Serialize};

/// Configuration for collaborative development
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeConfig {
    /// Enable real-time collaboration
    pub enable_real_time: bool,
    
    /// Maximum concurrent users per workspace
    pub max_concurrent_users: usize,
    
    /// Enable conflict detection
    pub enable_conflict_detection: bool,
    
    /// Automatic conflict resolution threshold
    pub auto_resolution_confidence: f64,
    
    /// Enable peer review system
    pub enable_peer_review: bool,
    
    /// Minimum reviewers required for approval
    pub min_reviewers: usize,
    
    /// Enable shape sharing and library
    pub enable_shape_library: bool,
    
    /// Workspace session timeout (seconds)
    pub session_timeout_seconds: u64,
    
    /// Enable activity tracking
    pub enable_activity_tracking: bool,
    
    /// Enable notifications
    pub enable_notifications: bool,
}

impl Default for CollaborativeConfig {
    fn default() -> Self {
        Self {
            enable_real_time: true,
            max_concurrent_users: 10,
            enable_conflict_detection: true,
            auto_resolution_confidence: 0.8,
            enable_peer_review: true,
            min_reviewers: 2,
            enable_shape_library: true,
            session_timeout_seconds: 3600, // 1 hour
            enable_activity_tracking: true,
            enable_notifications: true,
        }
    }
}

impl CollaborativeConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable all features
    pub fn all_enabled() -> Self {
        Self {
            enable_real_time: true,
            max_concurrent_users: 50,
            enable_conflict_detection: true,
            auto_resolution_confidence: 0.9,
            enable_peer_review: true,
            min_reviewers: 1,
            enable_shape_library: true,
            session_timeout_seconds: 7200, // 2 hours
            enable_activity_tracking: true,
            enable_notifications: true,
        }
    }

    /// Minimal configuration for basic collaboration
    pub fn minimal() -> Self {
        Self {
            enable_real_time: true,
            max_concurrent_users: 5,
            enable_conflict_detection: false,
            auto_resolution_confidence: 0.5,
            enable_peer_review: false,
            min_reviewers: 0,
            enable_shape_library: false,
            session_timeout_seconds: 1800, // 30 minutes
            enable_activity_tracking: false,
            enable_notifications: false,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_concurrent_users == 0 {
            return Err("max_concurrent_users must be greater than 0".to_string());
        }

        if self.auto_resolution_confidence < 0.0 || self.auto_resolution_confidence > 1.0 {
            return Err("auto_resolution_confidence must be between 0.0 and 1.0".to_string());
        }

        if self.enable_peer_review && self.min_reviewers == 0 {
            return Err("min_reviewers must be greater than 0 when peer review is enabled".to_string());
        }

        if self.session_timeout_seconds < 60 {
            return Err("session_timeout_seconds must be at least 60 seconds".to_string());
        }

        Ok(())
    }
}

/// Workspace settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSettings {
    pub is_public: bool,
    pub allow_external_contributions: bool,
    pub require_approval_for_changes: bool,
    pub enable_auto_save: bool,
    pub save_interval_seconds: u64,
    pub max_shape_versions: usize,
    pub enable_branching: bool,
}

impl Default for WorkspaceSettings {
    fn default() -> Self {
        Self {
            is_public: false,
            allow_external_contributions: true,
            require_approval_for_changes: true,
            enable_auto_save: true,
            save_interval_seconds: 300, // 5 minutes
            max_shape_versions: 50,
            enable_branching: true,
        }
    }
}