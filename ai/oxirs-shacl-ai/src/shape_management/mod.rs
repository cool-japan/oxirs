//! Shape Management Module
//!
//! This module provides intelligent shape management capabilities including
//! version control, optimization, collaboration, and reusability features.

pub mod collaboration;
pub mod library;
pub mod optimization;
pub mod reusability;
pub mod version_control;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use oxirs_core::{model::Term, Store};
use oxirs_shacl::{Shape, ShapeId, ValidationReport};

use crate::{
    shape::{PropertyConstraint, Shape as AiShape, ShapeMetrics},
    Result, ShaclAiError,
};

// Re-export important types
pub use collaboration::*;
pub use library::*;
pub use optimization::*;
pub use reusability::*;
pub use version_control::*;

/// Configuration for shape management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeManagementConfig {
    /// Enable automatic versioning
    pub enable_auto_versioning: bool,

    /// Enable optimization recommendations
    pub enable_optimization: bool,

    /// Enable collaboration features
    pub enable_collaboration: bool,

    /// Enable shape reusability analysis
    pub enable_reusability: bool,

    /// Version retention policy (number of versions to keep)
    pub version_retention_count: usize,

    /// Compatibility check strictness (0.0 to 1.0)
    pub compatibility_strictness: f64,

    /// Optimization threshold for automatic recommendations
    pub optimization_threshold: f64,

    /// Collaboration timeout in seconds
    pub collaboration_timeout_secs: u64,

    /// Maximum concurrent shape operations
    pub max_concurrent_operations: usize,

    /// Cache size for shape operations
    pub cache_size: usize,

    /// Performance monitoring settings
    pub performance_monitoring: PerformanceMonitoringConfig,
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoringConfig {
    pub enable_metrics: bool,
    pub metrics_interval_secs: u64,
    pub enable_profiling: bool,
    pub max_profile_samples: usize,
}

impl Default for ShapeManagementConfig {
    fn default() -> Self {
        Self {
            enable_auto_versioning: true,
            enable_optimization: true,
            enable_collaboration: true,
            enable_reusability: true,
            version_retention_count: 10,
            compatibility_strictness: 0.8,
            optimization_threshold: 0.7,
            collaboration_timeout_secs: 300,
            max_concurrent_operations: 10,
            cache_size: 1000,
            performance_monitoring: PerformanceMonitoringConfig {
                enable_metrics: true,
                metrics_interval_secs: 60,
                enable_profiling: false,
                max_profile_samples: 1000,
            },
        }
    }
}

/// Intelligent shape management system
#[derive(Debug)]
pub struct IntelligentShapeManager {
    config: ShapeManagementConfig,
    version_control: ShapeVersionControl,
    optimizer: ShapeOptimizer,
    collaboration_engine: CollaborationEngine,
    reusability_manager: ReusabilityManager,
    shape_library: ShapeLibrary,
    statistics: ShapeManagementStatistics,
}

/// Statistics for shape management operations
#[derive(Debug, Default)]
pub struct ShapeManagementStatistics {
    pub shapes_managed: AtomicUsize,
    pub versions_created: AtomicUsize,
    pub optimizations_applied: AtomicUsize,
    pub collaborations_facilitated: AtomicUsize,
    pub patterns_reused: AtomicUsize,
    pub conflicts_resolved: AtomicUsize,
    pub average_optimization_improvement: f64,
    pub collaboration_efficiency: f64,
}

impl IntelligentShapeManager {
    /// Create a new intelligent shape manager
    pub fn new() -> Self {
        Self::with_config(ShapeManagementConfig::default())
    }

    /// Create a new intelligent shape manager with custom configuration
    pub fn with_config(config: ShapeManagementConfig) -> Self {
        Self {
            version_control: ShapeVersionControl::new(),
            optimizer: ShapeOptimizer::new(),
            collaboration_engine: CollaborationEngine::new(),
            reusability_manager: ReusabilityManager::new(),
            shape_library: ShapeLibrary::new(),
            statistics: ShapeManagementStatistics::default(),
            config,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ShapeManagementConfig {
        &self.config
    }

    /// Get statistics
    pub fn statistics(&self) -> &ShapeManagementStatistics {
        &self.statistics
    }

    /// Get version control system
    pub fn version_control(&self) -> &ShapeVersionControl {
        &self.version_control
    }

    /// Get mutable version control system
    pub fn version_control_mut(&mut self) -> &mut ShapeVersionControl {
        &mut self.version_control
    }

    /// Get optimizer
    pub fn optimizer(&self) -> &ShapeOptimizer {
        &self.optimizer
    }

    /// Get mutable optimizer
    pub fn optimizer_mut(&mut self) -> &mut ShapeOptimizer {
        &mut self.optimizer
    }

    /// Get collaboration engine
    pub fn collaboration_engine(&self) -> &CollaborationEngine {
        &self.collaboration_engine
    }

    /// Get mutable collaboration engine
    pub fn collaboration_engine_mut(&mut self) -> &mut CollaborationEngine {
        &mut self.collaboration_engine
    }

    /// Get reusability manager
    pub fn reusability_manager(&self) -> &ReusabilityManager {
        &self.reusability_manager
    }

    /// Get mutable reusability manager
    pub fn reusability_manager_mut(&mut self) -> &mut ReusabilityManager {
        &mut self.reusability_manager
    }

    /// Get shape library
    pub fn shape_library(&self) -> &ShapeLibrary {
        &self.shape_library
    }

    /// Get mutable shape library
    pub fn shape_library_mut(&mut self) -> &mut ShapeLibrary {
        &mut self.shape_library
    }
}

impl Default for IntelligentShapeManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization opportunity for shape improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    /// Unique identifier for the opportunity
    pub id: String,
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Expected improvement score (0.0 to 1.0)
    pub expected_improvement: f64,
    /// Confidence level in the recommendation
    pub confidence: f64,
    /// Description of the optimization
    pub description: String,
    /// Effort required to implement
    pub effort_level: EffortLevel,
    /// Priority of the optimization
    pub priority: OptimizationPriority,
}

/// Types of shape optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Constraint simplification
    ConstraintSimplification,
    /// Path optimization
    PathOptimization,
    /// Target refinement
    TargetRefinement,
    /// Property grouping
    PropertyGrouping,
    /// Redundancy removal
    RedundancyRemoval,
    /// Performance optimization
    PerformanceOptimization,
    /// Custom optimization
    Custom(String),
}

impl std::fmt::Display for OptimizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizationType::ConstraintSimplification => write!(f, "constraint_simplification"),
            OptimizationType::PathOptimization => write!(f, "path_optimization"),
            OptimizationType::TargetRefinement => write!(f, "target_refinement"),
            OptimizationType::PropertyGrouping => write!(f, "property_grouping"),
            OptimizationType::RedundancyRemoval => write!(f, "redundancy_removal"),
            OptimizationType::PerformanceOptimization => write!(f, "performance_optimization"),
            OptimizationType::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

impl OptimizationType {
    pub fn as_str(&self) -> &str {
        match self {
            OptimizationType::ConstraintSimplification => "constraint_simplification",
            OptimizationType::PathOptimization => "path_optimization",
            OptimizationType::TargetRefinement => "target_refinement",
            OptimizationType::PropertyGrouping => "property_grouping",
            OptimizationType::RedundancyRemoval => "redundancy_removal",
            OptimizationType::PerformanceOptimization => "performance_optimization",
            OptimizationType::Custom(_) => "custom",
        }
    }
}

/// Effort level for implementing optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Priority of optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationPriority {
    Critical,
    High,
    Medium,
    Low,
}
