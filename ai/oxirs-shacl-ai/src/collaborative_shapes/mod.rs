//! Collaborative Shape Development
//!
//! This module implements comprehensive collaborative features for shape development,
//! including multi-user management, real-time collaboration, conflict resolution,
//! and shape reusability.

pub mod config;
pub mod types;
pub mod workspace;
pub mod real_time;
pub mod conflict_resolution;
pub mod library;
pub mod review;

pub use config::*;
pub use types::*;
pub use workspace::*;
pub use real_time::*;
pub use conflict_resolution::*;
pub use library::*;
pub use review::*;

use crate::{Result, ShaclAiError};

/// Collaborative shape development system
#[derive(Debug)]
pub struct CollaborativeShapeSystem {
    config: CollaborativeConfig,
    workspace_manager: WorkspaceManager,
    real_time_engine: RealTimeCollaborationEngine,
    conflict_resolution: AdvancedConflictResolution,
    shape_library: CollaborativeShapeLibrary,
    review_system: ShapeReviewSystem,
    statistics: CollaborativeStatistics,
}

impl CollaborativeShapeSystem {
    /// Create new collaborative shape system
    pub fn new() -> Self {
        Self::with_config(CollaborativeConfig::default())
    }
    
    /// Create with custom configuration
    pub fn with_config(config: CollaborativeConfig) -> Self {
        Self {
            workspace_manager: WorkspaceManager::new(),
            real_time_engine: RealTimeCollaborationEngine::new(),
            conflict_resolution: AdvancedConflictResolution::new(),
            shape_library: CollaborativeShapeLibrary::new(),
            review_system: ShapeReviewSystem::new(),
            statistics: CollaborativeStatistics::default(),
            config,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &CollaborativeConfig {
        &self.config
    }

    /// Get workspace manager
    pub fn workspace_manager(&self) -> &WorkspaceManager {
        &self.workspace_manager
    }

    /// Get mutable workspace manager
    pub fn workspace_manager_mut(&mut self) -> &mut WorkspaceManager {
        &mut self.workspace_manager
    }

    /// Get real-time engine
    pub fn real_time_engine(&self) -> &RealTimeCollaborationEngine {
        &self.real_time_engine
    }

    /// Get mutable real-time engine
    pub fn real_time_engine_mut(&mut self) -> &mut RealTimeCollaborationEngine {
        &mut self.real_time_engine
    }

    /// Get conflict resolution
    pub fn conflict_resolution(&self) -> &AdvancedConflictResolution {
        &self.conflict_resolution
    }

    /// Get mutable conflict resolution
    pub fn conflict_resolution_mut(&mut self) -> &mut AdvancedConflictResolution {
        &mut self.conflict_resolution
    }

    /// Get shape library
    pub fn shape_library(&self) -> &CollaborativeShapeLibrary {
        &self.shape_library
    }

    /// Get mutable shape library
    pub fn shape_library_mut(&mut self) -> &mut CollaborativeShapeLibrary {
        &mut self.shape_library
    }

    /// Get review system
    pub fn review_system(&self) -> &ShapeReviewSystem {
        &self.review_system
    }

    /// Get mutable review system
    pub fn review_system_mut(&mut self) -> &mut ShapeReviewSystem {
        &mut self.review_system
    }

    /// Get statistics
    pub fn statistics(&self) -> &CollaborativeStatistics {
        &self.statistics
    }

    /// Get mutable statistics
    pub fn statistics_mut(&mut self) -> &mut CollaborativeStatistics {
        &mut self.statistics
    }

    /// Initialize the collaborative system
    pub async fn initialize(&mut self) -> Result<()> {
        // Initialize all subsystems
        self.workspace_manager.initialize().await?;
        self.real_time_engine.initialize().await?;
        self.conflict_resolution.initialize().await?;
        self.shape_library.initialize().await?;
        self.review_system.initialize().await?;
        
        Ok(())
    }

    /// Shutdown the collaborative system
    pub async fn shutdown(&mut self) -> Result<()> {
        // Shutdown all subsystems
        self.real_time_engine.shutdown().await?;
        self.workspace_manager.shutdown().await?;
        
        Ok(())
    }
}

impl Default for CollaborativeShapeSystem {
    fn default() -> Self {
        Self::new()
    }
}