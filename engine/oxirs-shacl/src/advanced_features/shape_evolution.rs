//! Shape Evolution & Version Tracking
//!
//! This module provides advanced shape evolution tracking, versioning, and lifecycle management.
//! It enables monitoring how shapes change over time and maintaining historical shape versions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

use crate::{Result, ShaclError, Shape, ShapeId};

/// Shape evolution event types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeEvolutionEvent {
    /// Shape was created
    Created,
    /// Constraint added
    ConstraintAdded(String),
    /// Constraint removed
    ConstraintRemoved(String),
    /// Constraint modified
    ConstraintModified(String),
    /// Target added
    TargetAdded,
    /// Target removed
    TargetRemoved,
    /// Property added
    PropertyAdded(String),
    /// Property removed
    PropertyRemoved(String),
    /// Severity changed
    SeverityChanged,
    /// Shape deactivated
    Deactivated,
    /// Shape reactivated
    Reactivated,
    /// Shape merged with another
    Merged(ShapeId),
    /// Shape split into multiple shapes
    Split(Vec<ShapeId>),
    /// Custom event
    Custom(String),
}

/// Shape version entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeVersion {
    /// Version number (semantic versioning: major.minor.patch)
    pub version: String,
    /// Shape at this version
    pub shape: Shape,
    /// Timestamp of this version
    pub timestamp: DateTime<Utc>,
    /// Event that caused this version
    pub event: ShapeEvolutionEvent,
    /// Description of changes
    pub change_description: String,
    /// Author of changes
    pub author: Option<String>,
}

/// Shape evolution tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeEvolutionTracker {
    /// Shape ID being tracked
    pub shape_id: ShapeId,
    /// Version history (oldest to newest)
    pub versions: VecDeque<ShapeVersion>,
    /// Maximum versions to retain
    pub max_versions: usize,
    /// Current version
    pub current_version: String,
}

impl ShapeEvolutionTracker {
    /// Create a new evolution tracker
    pub fn new(shape_id: ShapeId, initial_shape: Shape) -> Self {
        let version = "1.0.0".to_string();
        let mut versions = VecDeque::new();

        versions.push_back(ShapeVersion {
            version: version.clone(),
            shape: initial_shape,
            timestamp: Utc::now(),
            event: ShapeEvolutionEvent::Created,
            change_description: "Initial shape creation".to_string(),
            author: None,
        });

        Self {
            shape_id,
            versions,
            max_versions: 100,
            current_version: version,
        }
    }

    /// Record a new shape version
    pub fn record_version(
        &mut self,
        shape: Shape,
        event: ShapeEvolutionEvent,
        description: String,
        author: Option<String>,
    ) -> Result<()> {
        // Parse current version and increment
        let new_version = self.increment_version(&event)?;

        let version = ShapeVersion {
            version: new_version.clone(),
            shape,
            timestamp: Utc::now(),
            event,
            change_description: description,
            author,
        };

        // Add new version
        self.versions.push_back(version);
        self.current_version = new_version;

        // Trim old versions if needed
        while self.versions.len() > self.max_versions {
            self.versions.pop_front();
        }

        Ok(())
    }

    /// Increment version based on event type
    fn increment_version(&self, event: &ShapeEvolutionEvent) -> Result<String> {
        let parts: Vec<&str> = self.current_version.split('.').collect();
        if parts.len() != 3 {
            return Err(ShaclError::Configuration(
                "Invalid version format".to_string(),
            ));
        }

        let major: u32 = parts[0]
            .parse()
            .map_err(|_| ShaclError::Configuration("Invalid major version".to_string()))?;
        let minor: u32 = parts[1]
            .parse()
            .map_err(|_| ShaclError::Configuration("Invalid minor version".to_string()))?;
        let patch: u32 = parts[2]
            .parse()
            .map_err(|_| ShaclError::Configuration("Invalid patch version".to_string()))?;

        // Determine version increment based on event severity
        let new_version = match event {
            ShapeEvolutionEvent::Created => "1.0.0".to_string(),
            ShapeEvolutionEvent::Merged(_) | ShapeEvolutionEvent::Split(_) => {
                format!("{}.0.0", major + 1)
            }
            ShapeEvolutionEvent::ConstraintAdded(_)
            | ShapeEvolutionEvent::ConstraintRemoved(_)
            | ShapeEvolutionEvent::PropertyAdded(_)
            | ShapeEvolutionEvent::PropertyRemoved(_) => format!("{}.{}.0", major, minor + 1),
            _ => format!("{}.{}.{}", major, minor, patch + 1),
        };

        Ok(new_version)
    }

    /// Get a specific version
    pub fn get_version(&self, version: &str) -> Option<&ShapeVersion> {
        self.versions.iter().find(|v| v.version == version)
    }

    /// Get the current shape
    pub fn current_shape(&self) -> Option<&Shape> {
        self.versions.back().map(|v| &v.shape)
    }

    /// Get all versions
    pub fn all_versions(&self) -> Vec<&ShapeVersion> {
        self.versions.iter().collect()
    }

    /// Get evolution timeline
    pub fn timeline(&self) -> Vec<(DateTime<Utc>, &ShapeEvolutionEvent, &str)> {
        self.versions
            .iter()
            .map(|v| (v.timestamp, &v.event, v.version.as_str()))
            .collect()
    }

    /// Rollback to a previous version
    pub fn rollback(&mut self, version: &str) -> Result<Shape> {
        let target_version = self
            .versions
            .iter()
            .find(|v| v.version == version)
            .ok_or_else(|| ShaclError::Configuration(format!("Version {} not found", version)))?;

        let shape = target_version.shape.clone();

        // Record rollback event
        self.record_version(
            shape.clone(),
            ShapeEvolutionEvent::Custom(format!("Rollback to {}", version)),
            format!("Rolled back to version {}", version),
            None,
        )?;

        Ok(shape)
    }

    /// Compare two versions
    pub fn compare_versions(&self, v1: &str, v2: &str) -> Result<Vec<ShapeDifference>> {
        let version1 = self
            .get_version(v1)
            .ok_or_else(|| ShaclError::Configuration(format!("Version {} not found", v1)))?;
        let version2 = self
            .get_version(v2)
            .ok_or_else(|| ShaclError::Configuration(format!("Version {} not found", v2)))?;

        Ok(compare_shapes(&version1.shape, &version2.shape))
    }
}

/// Difference between two shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShapeDifference {
    /// Constraint added
    ConstraintAdded(String),
    /// Constraint removed
    ConstraintRemoved(String),
    /// Constraint value changed
    ConstraintChanged {
        constraint: String,
        old_value: String,
        new_value: String,
    },
    /// Target added or removed
    TargetDifference(String),
    /// Severity changed
    SeverityChanged { old: String, new: String },
    /// Other property changed
    PropertyChanged {
        property: String,
        old_value: String,
        new_value: String,
    },
}

/// Compare two shapes and return differences
pub fn compare_shapes(shape1: &Shape, shape2: &Shape) -> Vec<ShapeDifference> {
    let mut differences = Vec::new();

    // Compare constraints
    for (id, _constraint) in &shape1.constraints {
        if !shape2.constraints.contains_key(id) {
            differences.push(ShapeDifference::ConstraintRemoved(id.as_str().to_string()));
        }
    }

    for (id, _constraint) in &shape2.constraints {
        if !shape1.constraints.contains_key(id) {
            differences.push(ShapeDifference::ConstraintAdded(id.as_str().to_string()));
        }
    }

    // Compare severity
    if shape1.severity != shape2.severity {
        differences.push(ShapeDifference::SeverityChanged {
            old: format!("{:?}", shape1.severity),
            new: format!("{:?}", shape2.severity),
        });
    }

    // Compare targets
    if shape1.targets.len() != shape2.targets.len() {
        differences.push(ShapeDifference::TargetDifference(format!(
            "Target count changed: {} -> {}",
            shape1.targets.len(),
            shape2.targets.len()
        )));
    }

    differences
}

/// Shape evolution registry
pub struct ShapeEvolutionRegistry {
    /// Trackers by shape ID
    trackers: HashMap<ShapeId, ShapeEvolutionTracker>,
}

impl ShapeEvolutionRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            trackers: HashMap::new(),
        }
    }

    /// Register a shape for evolution tracking
    pub fn register(&mut self, shape: Shape) {
        let shape_id = shape.id.clone();
        let tracker = ShapeEvolutionTracker::new(shape_id.clone(), shape);
        self.trackers.insert(shape_id, tracker);
    }

    /// Get a tracker
    pub fn get_tracker(&self, shape_id: &ShapeId) -> Option<&ShapeEvolutionTracker> {
        self.trackers.get(shape_id)
    }

    /// Get a mutable tracker
    pub fn get_tracker_mut(&mut self, shape_id: &ShapeId) -> Option<&mut ShapeEvolutionTracker> {
        self.trackers.get_mut(shape_id)
    }

    /// Record an evolution event
    pub fn record_event(
        &mut self,
        shape_id: &ShapeId,
        shape: Shape,
        event: ShapeEvolutionEvent,
        description: String,
        author: Option<String>,
    ) -> Result<()> {
        let tracker = self
            .trackers
            .get_mut(shape_id)
            .ok_or_else(|| ShaclError::Configuration("Shape not registered".to_string()))?;

        tracker.record_version(shape, event, description, author)
    }

    /// Get all tracked shapes
    pub fn tracked_shapes(&self) -> Vec<&ShapeId> {
        self.trackers.keys().collect()
    }
}

impl Default for ShapeEvolutionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ShapeType;

    #[test]
    fn test_evolution_tracker_creation() {
        let shape_id = ShapeId::new("test:shape".to_string());
        let shape = Shape::new(shape_id.clone(), ShapeType::NodeShape);

        let tracker = ShapeEvolutionTracker::new(shape_id, shape);

        assert_eq!(tracker.current_version, "1.0.0");
        assert_eq!(tracker.versions.len(), 1);
    }

    #[test]
    fn test_version_increment() {
        let shape_id = ShapeId::new("test:shape".to_string());
        let shape = Shape::new(shape_id.clone(), ShapeType::NodeShape);

        let mut tracker = ShapeEvolutionTracker::new(shape_id.clone(), shape.clone());

        tracker
            .record_version(
                shape.clone(),
                ShapeEvolutionEvent::ConstraintAdded("test".to_string()),
                "Added constraint".to_string(),
                None,
            )
            .unwrap();

        assert_eq!(tracker.current_version, "1.1.0");
    }

    #[test]
    fn test_rollback() {
        let shape_id = ShapeId::new("test:shape".to_string());
        let shape = Shape::new(shape_id.clone(), ShapeType::NodeShape);

        let mut tracker = ShapeEvolutionTracker::new(shape_id.clone(), shape.clone());

        // Create a few versions
        tracker
            .record_version(
                shape.clone(),
                ShapeEvolutionEvent::ConstraintAdded("test".to_string()),
                "Added constraint".to_string(),
                None,
            )
            .unwrap();

        // Rollback to 1.0.0
        let result = tracker.rollback("1.0.0");
        assert!(result.is_ok());
    }

    #[test]
    fn test_evolution_registry() {
        let mut registry = ShapeEvolutionRegistry::new();

        let shape_id = ShapeId::new("test:shape".to_string());
        let shape = Shape::new(shape_id.clone(), ShapeType::NodeShape);

        registry.register(shape.clone());

        assert!(registry.get_tracker(&shape_id).is_some());
        assert_eq!(registry.tracked_shapes().len(), 1);
    }

    #[test]
    fn test_compare_shapes() {
        let shape_id = ShapeId::new("test:shape".to_string());
        let shape1 = Shape::new(shape_id.clone(), ShapeType::NodeShape);
        let mut shape2 = shape1.clone();

        shape2.severity = crate::Severity::Warning;

        let diffs = compare_shapes(&shape1, &shape2);
        assert!(!diffs.is_empty());
    }
}
