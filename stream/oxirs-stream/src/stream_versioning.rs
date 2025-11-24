//! Stream Versioning and Time-Travel Queries
//!
//! This module provides comprehensive stream versioning capabilities with
//! time-travel query support for historical data analysis and replay.
//!
//! # Features
//!
//! - **Version Management**: Track and manage stream data versions
//! - **Time-Travel Queries**: Query historical stream states
//! - **Snapshot Management**: Create and restore point-in-time snapshots
//! - **Branching**: Create branches for what-if analysis
//! - **Diff Operations**: Compare versions and generate changesets
//! - **Retention Policies**: Automatic version cleanup and archival

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

use crate::error::StreamError;

/// Version identifier type
pub type VersionId = u64;

/// Branch identifier type
pub type BranchId = String;

/// Stream versioning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningConfig {
    /// Maximum number of versions to retain
    pub max_versions: usize,
    /// Maximum age for version retention
    pub max_age: Duration,
    /// Enable automatic snapshots
    pub auto_snapshot: bool,
    /// Snapshot interval
    pub snapshot_interval: Duration,
    /// Enable compression for old versions
    pub compress_old_versions: bool,
    /// Compression threshold (versions older than this get compressed)
    pub compression_threshold: Duration,
    /// Enable branching support
    pub enable_branching: bool,
    /// Maximum number of branches
    pub max_branches: usize,
}

impl Default for VersioningConfig {
    fn default() -> Self {
        Self {
            max_versions: 1000,
            max_age: Duration::from_secs(86400 * 7), // 7 days
            auto_snapshot: true,
            snapshot_interval: Duration::from_secs(3600), // 1 hour
            compress_old_versions: true,
            compression_threshold: Duration::from_secs(86400), // 1 day
            enable_branching: true,
            max_branches: 10,
        }
    }
}

/// Version metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadata {
    /// Version ID
    pub version_id: VersionId,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Parent version (None for root)
    pub parent_version: Option<VersionId>,
    /// Branch this version belongs to
    pub branch_id: BranchId,
    /// Version description
    pub description: String,
    /// Number of events in this version
    pub event_count: usize,
    /// Size in bytes
    pub size_bytes: usize,
    /// Is this version compressed
    pub is_compressed: bool,
    /// Custom tags
    pub tags: HashMap<String, String>,
}

/// A versioned event with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedEvent<T> {
    /// The event data
    pub data: T,
    /// Version this event was added in
    pub version_id: VersionId,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event sequence number within version
    pub sequence: u64,
    /// Is this event deleted in a later version
    pub is_deleted: bool,
    /// Version where this event was deleted (if applicable)
    pub deleted_in_version: Option<VersionId>,
}

/// A snapshot of stream state at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot<T> {
    /// Snapshot identifier
    pub snapshot_id: String,
    /// Version at snapshot time
    pub version_id: VersionId,
    /// Snapshot timestamp
    pub timestamp: SystemTime,
    /// Events in the snapshot
    pub events: Vec<VersionedEvent<T>>,
    /// Snapshot metadata
    pub metadata: HashMap<String, String>,
    /// Size in bytes
    pub size_bytes: usize,
}

/// Branch information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    /// Branch identifier
    pub branch_id: BranchId,
    /// Branch name
    pub name: String,
    /// Base version (where branch was created)
    pub base_version: VersionId,
    /// Current head version
    pub head_version: VersionId,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last update timestamp
    pub updated_at: SystemTime,
    /// Branch description
    pub description: String,
    /// Is this the main branch
    pub is_main: bool,
}

/// Time-travel query specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeTravelQuery {
    /// Target version or timestamp
    pub target: TimeTravelTarget,
    /// Branch to query
    pub branch_id: Option<BranchId>,
    /// Filter predicate
    pub filter: Option<String>,
    /// Projection fields
    pub projection: Option<Vec<String>>,
    /// Result limit
    pub limit: Option<usize>,
    /// Include deleted events
    pub include_deleted: bool,
}

/// Target for time-travel query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeTravelTarget {
    /// Specific version ID
    Version(VersionId),
    /// Point in time
    Timestamp(SystemTime),
    /// Relative time (e.g., 1 hour ago)
    RelativeTime(Duration),
    /// Latest version
    Latest,
    /// Specific snapshot
    Snapshot(String),
}

/// Diff between two versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff<T> {
    /// Source version
    pub from_version: VersionId,
    /// Target version
    pub to_version: VersionId,
    /// Added events
    pub added: Vec<VersionedEvent<T>>,
    /// Deleted events
    pub deleted: Vec<VersionedEvent<T>>,
    /// Modified events (old value, new value)
    pub modified: Vec<(VersionedEvent<T>, VersionedEvent<T>)>,
    /// Number of unchanged events
    pub unchanged_count: usize,
}

/// Change operation type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChangeType {
    /// Event added
    Add,
    /// Event deleted
    Delete,
    /// Event modified
    Modify,
}

/// A single change in a changeset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Change<T> {
    /// Change type
    pub change_type: ChangeType,
    /// Event data
    pub data: T,
    /// Previous value (for modifications)
    pub previous: Option<T>,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Changeset between versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Changeset<T> {
    /// Source version
    pub from_version: VersionId,
    /// Target version
    pub to_version: VersionId,
    /// List of changes
    pub changes: Vec<Change<T>>,
    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Versioning statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VersioningStats {
    /// Total number of versions
    pub total_versions: usize,
    /// Total number of events across all versions
    pub total_events: usize,
    /// Total storage size
    pub total_size_bytes: usize,
    /// Number of snapshots
    pub snapshot_count: usize,
    /// Number of branches
    pub branch_count: usize,
    /// Oldest version timestamp
    pub oldest_version: Option<SystemTime>,
    /// Newest version timestamp
    pub newest_version: Option<SystemTime>,
    /// Average events per version
    pub avg_events_per_version: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Number of time-travel queries executed
    pub time_travel_queries: u64,
    /// Average query latency
    pub avg_query_latency_ms: f64,
}

/// Stream versioning manager
pub struct StreamVersioning<T>
where
    T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    /// Configuration
    config: VersioningConfig,
    /// Current version
    current_version: Arc<RwLock<VersionId>>,
    /// Version metadata index
    versions: Arc<RwLock<BTreeMap<VersionId, VersionMetadata>>>,
    /// Event storage by version
    events: Arc<RwLock<HashMap<VersionId, Vec<VersionedEvent<T>>>>>,
    /// Snapshots
    snapshots: Arc<RwLock<HashMap<String, Snapshot<T>>>>,
    /// Branches
    branches: Arc<RwLock<HashMap<BranchId, Branch>>>,
    /// Current branch
    current_branch: Arc<RwLock<BranchId>>,
    /// Statistics
    stats: Arc<RwLock<VersioningStats>>,
    /// Last snapshot time
    last_snapshot: Arc<RwLock<Instant>>,
    /// Query latencies for stats
    query_latencies: Arc<RwLock<VecDeque<f64>>>,
}

impl<T> StreamVersioning<T>
where
    T: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    /// Create a new stream versioning manager
    pub fn new(config: VersioningConfig) -> Self {
        let main_branch = Branch {
            branch_id: "main".to_string(),
            name: "Main Branch".to_string(),
            base_version: 0,
            head_version: 0,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            description: "Main development branch".to_string(),
            is_main: true,
        };

        let mut branches = HashMap::new();
        branches.insert("main".to_string(), main_branch);

        let initial_version = VersionMetadata {
            version_id: 0,
            created_at: SystemTime::now(),
            parent_version: None,
            branch_id: "main".to_string(),
            description: "Initial version".to_string(),
            event_count: 0,
            size_bytes: 0,
            is_compressed: false,
            tags: HashMap::new(),
        };

        let mut versions = BTreeMap::new();
        versions.insert(0, initial_version);

        Self {
            config,
            current_version: Arc::new(RwLock::new(0)),
            versions: Arc::new(RwLock::new(versions)),
            events: Arc::new(RwLock::new(HashMap::new())),
            snapshots: Arc::new(RwLock::new(HashMap::new())),
            branches: Arc::new(RwLock::new(branches)),
            current_branch: Arc::new(RwLock::new("main".to_string())),
            stats: Arc::new(RwLock::new(VersioningStats::default())),
            last_snapshot: Arc::new(RwLock::new(Instant::now())),
            query_latencies: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
        }
    }

    /// Create a new version with events
    pub async fn create_version(
        &self,
        events: Vec<T>,
        description: &str,
    ) -> Result<VersionId, StreamError> {
        let mut current = self.current_version.write().await;
        let mut versions = self.versions.write().await;
        let mut event_storage = self.events.write().await;
        let mut branches = self.branches.write().await;
        let current_branch = self.current_branch.read().await.clone();

        let new_version_id = *current + 1;

        // Create versioned events
        let versioned_events: Vec<VersionedEvent<T>> = events
            .into_iter()
            .enumerate()
            .map(|(i, data)| VersionedEvent {
                data,
                version_id: new_version_id,
                timestamp: SystemTime::now(),
                sequence: i as u64,
                is_deleted: false,
                deleted_in_version: None,
            })
            .collect();

        let event_count = versioned_events.len();
        let size_bytes = self.estimate_size(&versioned_events);

        // Create version metadata
        let metadata = VersionMetadata {
            version_id: new_version_id,
            created_at: SystemTime::now(),
            parent_version: Some(*current),
            branch_id: current_branch.clone(),
            description: description.to_string(),
            event_count,
            size_bytes,
            is_compressed: false,
            tags: HashMap::new(),
        };

        // Store version
        versions.insert(new_version_id, metadata);
        event_storage.insert(new_version_id, versioned_events);

        // Update branch head
        if let Some(branch) = branches.get_mut(&current_branch) {
            branch.head_version = new_version_id;
            branch.updated_at = SystemTime::now();
        }

        *current = new_version_id;

        // Update stats
        self.update_stats_after_create(event_count, size_bytes)
            .await;

        // Apply retention policy
        self.apply_retention_policy(&mut versions, &mut event_storage)?;

        // Drop locks before calling maybe_create_auto_snapshot which acquires its own locks
        drop(versions);
        drop(event_storage);
        drop(branches);
        drop(current);

        // Check if we need to create auto-snapshot
        if self.config.auto_snapshot {
            self.maybe_create_auto_snapshot().await?;
        }

        Ok(new_version_id)
    }

    /// Add events to current version
    pub async fn add_events(&self, events: Vec<T>) -> Result<usize, StreamError> {
        let current = *self.current_version.read().await;
        let mut event_storage = self.events.write().await;

        let entry = event_storage.entry(current).or_insert_with(Vec::new);
        let start_sequence = entry.len() as u64;

        let versioned_events: Vec<VersionedEvent<T>> = events
            .into_iter()
            .enumerate()
            .map(|(i, data)| VersionedEvent {
                data,
                version_id: current,
                timestamp: SystemTime::now(),
                sequence: start_sequence + i as u64,
                is_deleted: false,
                deleted_in_version: None,
            })
            .collect();

        let count = versioned_events.len();
        entry.extend(versioned_events);

        // Update version metadata
        let mut versions = self.versions.write().await;
        if let Some(metadata) = versions.get_mut(&current) {
            metadata.event_count = entry.len();
            metadata.size_bytes = self.estimate_size(entry);
        }

        Ok(count)
    }

    /// Execute a time-travel query
    pub async fn time_travel_query(
        &self,
        query: TimeTravelQuery,
    ) -> Result<Vec<VersionedEvent<T>>, StreamError> {
        let start = Instant::now();

        let target_version = self.resolve_target(&query.target).await?;
        let branch_id = query.branch_id.unwrap_or_else(|| "main".to_string());

        // Collect all events up to target version for the branch
        let versions = self.versions.read().await;
        let event_storage = self.events.read().await;

        let mut result_events = Vec::new();

        for (version_id, metadata) in versions.iter() {
            if *version_id > target_version {
                break;
            }

            if metadata.branch_id != branch_id {
                continue;
            }

            if let Some(events) = event_storage.get(version_id) {
                for event in events {
                    // Skip deleted events unless requested
                    if event.is_deleted && !query.include_deleted {
                        if let Some(deleted_version) = event.deleted_in_version {
                            if deleted_version <= target_version {
                                continue;
                            }
                        }
                    }

                    result_events.push(event.clone());
                }
            }
        }

        // Apply limit
        if let Some(limit) = query.limit {
            result_events.truncate(limit);
        }

        // Record query latency
        let latency = start.elapsed().as_secs_f64() * 1000.0;
        self.record_query_latency(latency).await;

        Ok(result_events)
    }

    /// Get events at a specific version
    pub async fn get_at_version(
        &self,
        version_id: VersionId,
    ) -> Result<Vec<VersionedEvent<T>>, StreamError> {
        let query = TimeTravelQuery {
            target: TimeTravelTarget::Version(version_id),
            branch_id: None,
            filter: None,
            projection: None,
            limit: None,
            include_deleted: false,
        };

        self.time_travel_query(query).await
    }

    /// Get events at a specific timestamp
    pub async fn get_at_timestamp(
        &self,
        timestamp: SystemTime,
    ) -> Result<Vec<VersionedEvent<T>>, StreamError> {
        let query = TimeTravelQuery {
            target: TimeTravelTarget::Timestamp(timestamp),
            branch_id: None,
            filter: None,
            projection: None,
            limit: None,
            include_deleted: false,
        };

        self.time_travel_query(query).await
    }

    /// Create a snapshot
    pub async fn create_snapshot(&self, name: &str) -> Result<String, StreamError> {
        let current = *self.current_version.read().await;
        let events = self.get_at_version(current).await?;

        let snapshot_id = format!("{}_{}", name, current);
        let size_bytes = self.estimate_size(&events);

        let snapshot = Snapshot {
            snapshot_id: snapshot_id.clone(),
            version_id: current,
            timestamp: SystemTime::now(),
            events,
            metadata: HashMap::new(),
            size_bytes,
        };

        let mut snapshots = self.snapshots.write().await;
        snapshots.insert(snapshot_id.clone(), snapshot);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.snapshot_count += 1;

        Ok(snapshot_id)
    }

    /// Restore from snapshot
    pub async fn restore_snapshot(&self, snapshot_id: &str) -> Result<VersionId, StreamError> {
        let snapshots = self.snapshots.read().await;
        let snapshot = snapshots
            .get(snapshot_id)
            .ok_or_else(|| StreamError::NotFound(format!("Snapshot not found: {}", snapshot_id)))?
            .clone();
        drop(snapshots);

        // Create a new version from snapshot
        let events: Vec<T> = snapshot.events.into_iter().map(|e| e.data).collect();

        self.create_version(events, &format!("Restored from snapshot: {}", snapshot_id))
            .await
    }

    /// Create a new branch
    pub async fn create_branch(
        &self,
        name: &str,
        description: &str,
    ) -> Result<BranchId, StreamError> {
        if !self.config.enable_branching {
            return Err(StreamError::Configuration(
                "Branching is not enabled".to_string(),
            ));
        }

        let mut branches = self.branches.write().await;

        if branches.len() >= self.config.max_branches {
            return Err(StreamError::ResourceExhausted(
                "Maximum number of branches reached".to_string(),
            ));
        }

        let branch_id = format!("branch_{}", uuid::Uuid::new_v4());
        let current = *self.current_version.read().await;

        let branch = Branch {
            branch_id: branch_id.clone(),
            name: name.to_string(),
            base_version: current,
            head_version: current,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            description: description.to_string(),
            is_main: false,
        };

        branches.insert(branch_id.clone(), branch);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.branch_count += 1;

        Ok(branch_id)
    }

    /// Switch to a different branch
    pub async fn switch_branch(&self, branch_id: &str) -> Result<(), StreamError> {
        let branches = self.branches.read().await;

        if !branches.contains_key(branch_id) {
            return Err(StreamError::NotFound(format!(
                "Branch not found: {}",
                branch_id
            )));
        }

        let head_version = branches.get(branch_id).unwrap().head_version;
        drop(branches);

        let mut current_branch = self.current_branch.write().await;
        let mut current_version = self.current_version.write().await;

        *current_branch = branch_id.to_string();
        *current_version = head_version;

        Ok(())
    }

    /// Merge a branch into the current branch
    pub async fn merge_branch(&self, source_branch_id: &str) -> Result<VersionId, StreamError> {
        let branches = self.branches.read().await;
        let current_branch_id = self.current_branch.read().await.clone();

        let source_branch = branches
            .get(source_branch_id)
            .ok_or_else(|| {
                StreamError::NotFound(format!("Branch not found: {}", source_branch_id))
            })?
            .clone();

        let target_branch = branches
            .get(&current_branch_id)
            .ok_or_else(|| {
                StreamError::NotFound(format!("Branch not found: {}", current_branch_id))
            })?
            .clone();

        drop(branches);

        // Get events from source branch since divergence
        let source_events = self
            .get_branch_events_since(source_branch_id, source_branch.base_version)
            .await?;

        // Create merge version
        let events: Vec<T> = source_events.into_iter().map(|e| e.data).collect();

        self.create_version(
            events,
            &format!("Merge {} into {}", source_branch.name, target_branch.name),
        )
        .await
    }

    /// Get diff between two versions
    pub async fn diff(
        &self,
        from_version: VersionId,
        to_version: VersionId,
    ) -> Result<VersionDiff<T>, StreamError> {
        let from_events = self.get_at_version(from_version).await?;
        let to_events = self.get_at_version(to_version).await?;

        // Build event maps by sequence for comparison
        let from_map: HashMap<u64, &VersionedEvent<T>> =
            from_events.iter().map(|e| (e.sequence, e)).collect();
        let to_map: HashMap<u64, &VersionedEvent<T>> =
            to_events.iter().map(|e| (e.sequence, e)).collect();

        let mut added = Vec::new();
        let mut deleted = Vec::new();
        let mut modified = Vec::new();
        let mut unchanged_count = 0;

        // Find added and modified
        for (seq, event) in &to_map {
            if let Some(from_event) = from_map.get(seq) {
                // Check if modified (simplified - just compare version)
                if event.version_id != from_event.version_id {
                    modified.push(((*from_event).clone(), (*event).clone()));
                } else {
                    unchanged_count += 1;
                }
            } else {
                added.push((*event).clone());
            }
        }

        // Find deleted
        for (seq, event) in &from_map {
            if !to_map.contains_key(seq) {
                deleted.push((*event).clone());
            }
        }

        Ok(VersionDiff {
            from_version,
            to_version,
            added,
            deleted,
            modified,
            unchanged_count,
        })
    }

    /// Generate changeset between versions
    pub async fn generate_changeset(
        &self,
        from_version: VersionId,
        to_version: VersionId,
    ) -> Result<Changeset<T>, StreamError> {
        let diff = self.diff(from_version, to_version).await?;

        let mut changes = Vec::new();

        // Add changes for added events
        for event in diff.added {
            changes.push(Change {
                change_type: ChangeType::Add,
                data: event.data,
                previous: None,
                timestamp: event.timestamp,
            });
        }

        // Add changes for deleted events
        for event in diff.deleted {
            changes.push(Change {
                change_type: ChangeType::Delete,
                data: event.data,
                previous: None,
                timestamp: SystemTime::now(),
            });
        }

        // Add changes for modified events
        for (old, new) in diff.modified {
            changes.push(Change {
                change_type: ChangeType::Modify,
                data: new.data,
                previous: Some(old.data),
                timestamp: new.timestamp,
            });
        }

        Ok(Changeset {
            from_version,
            to_version,
            changes,
            created_at: SystemTime::now(),
        })
    }

    /// Get version history
    pub async fn get_version_history(&self) -> Vec<VersionMetadata> {
        let versions = self.versions.read().await;
        versions.values().cloned().collect()
    }

    /// Get all branches
    pub async fn get_branches(&self) -> Vec<Branch> {
        let branches = self.branches.read().await;
        branches.values().cloned().collect()
    }

    /// Get current version
    pub async fn current_version(&self) -> VersionId {
        *self.current_version.read().await
    }

    /// Get current branch
    pub async fn current_branch(&self) -> BranchId {
        self.current_branch.read().await.clone()
    }

    /// Get statistics
    pub async fn get_stats(&self) -> VersioningStats {
        self.stats.read().await.clone()
    }

    /// Tag a version
    pub async fn tag_version(
        &self,
        version_id: VersionId,
        key: &str,
        value: &str,
    ) -> Result<(), StreamError> {
        let mut versions = self.versions.write().await;

        if let Some(metadata) = versions.get_mut(&version_id) {
            metadata.tags.insert(key.to_string(), value.to_string());
            Ok(())
        } else {
            Err(StreamError::NotFound(format!(
                "Version not found: {}",
                version_id
            )))
        }
    }

    /// Find versions by tag
    pub async fn find_by_tag(&self, key: &str, value: &str) -> Vec<VersionId> {
        let versions = self.versions.read().await;

        versions
            .iter()
            .filter(|(_, m)| m.tags.get(key).map(|v| v == value).unwrap_or(false))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Delete a branch
    pub async fn delete_branch(&self, branch_id: &str) -> Result<(), StreamError> {
        let mut branches = self.branches.write().await;

        if let Some(branch) = branches.get(branch_id) {
            if branch.is_main {
                return Err(StreamError::InvalidOperation(
                    "Cannot delete main branch".to_string(),
                ));
            }
        } else {
            return Err(StreamError::NotFound(format!(
                "Branch not found: {}",
                branch_id
            )));
        }

        branches.remove(branch_id);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.branch_count = stats.branch_count.saturating_sub(1);

        Ok(())
    }

    /// Compact old versions
    pub async fn compact(&self) -> Result<usize, StreamError> {
        let mut event_storage = self.events.write().await;
        let mut versions = self.versions.write().await;

        let threshold = SystemTime::now() - self.config.compression_threshold;
        let mut compacted_count = 0;

        for (version_id, metadata) in versions.iter_mut() {
            if metadata.created_at < threshold && !metadata.is_compressed {
                // In a real implementation, we would compress the events
                // For now, we just mark them as compressed
                metadata.is_compressed = true;
                compacted_count += 1;

                // Optionally reduce storage (simplified)
                if let Some(events) = event_storage.get_mut(version_id) {
                    // In reality, we'd compress the serialized form
                    metadata.size_bytes = self.estimate_size(events) / 2; // Simulated compression
                }
            }
        }

        Ok(compacted_count)
    }

    // Private helper methods

    fn resolve_target<'a>(
        &'a self,
        target: &'a TimeTravelTarget,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<VersionId, StreamError>> + Send + 'a>,
    > {
        Box::pin(async move {
            match target {
                TimeTravelTarget::Version(v) => Ok(*v),
                TimeTravelTarget::Latest => Ok(*self.current_version.read().await),
                TimeTravelTarget::Timestamp(ts) => {
                    let versions = self.versions.read().await;
                    let mut best_version = 0;

                    for (version_id, metadata) in versions.iter() {
                        if metadata.created_at <= *ts {
                            best_version = *version_id;
                        } else {
                            break;
                        }
                    }

                    Ok(best_version)
                }
                TimeTravelTarget::RelativeTime(duration) => {
                    let target_time = SystemTime::now() - *duration;
                    self.resolve_target(&TimeTravelTarget::Timestamp(target_time))
                        .await
                }
                TimeTravelTarget::Snapshot(snapshot_id) => {
                    let snapshots = self.snapshots.read().await;
                    snapshots
                        .get(snapshot_id)
                        .map(|s| s.version_id)
                        .ok_or_else(|| {
                            StreamError::NotFound(format!("Snapshot not found: {}", snapshot_id))
                        })
                }
            }
        })
    }

    async fn get_branch_events_since(
        &self,
        branch_id: &str,
        since_version: VersionId,
    ) -> Result<Vec<VersionedEvent<T>>, StreamError> {
        let versions = self.versions.read().await;
        let event_storage = self.events.read().await;

        let mut result = Vec::new();

        for (version_id, metadata) in versions.iter() {
            if *version_id <= since_version {
                continue;
            }

            if metadata.branch_id != branch_id {
                continue;
            }

            if let Some(events) = event_storage.get(version_id) {
                result.extend(events.clone());
            }
        }

        Ok(result)
    }

    fn estimate_size<S: Serialize>(&self, data: &S) -> usize {
        // Rough estimate using serialization
        serde_json::to_vec(data).map(|v| v.len()).unwrap_or(0)
    }

    async fn update_stats_after_create(&self, event_count: usize, size_bytes: usize) {
        let mut stats = self.stats.write().await;
        stats.total_versions += 1;
        stats.total_events += event_count;
        stats.total_size_bytes += size_bytes;
        stats.newest_version = Some(SystemTime::now());

        if stats.oldest_version.is_none() {
            stats.oldest_version = Some(SystemTime::now());
        }

        if stats.total_versions > 0 {
            stats.avg_events_per_version = stats.total_events as f64 / stats.total_versions as f64;
        }
    }

    async fn record_query_latency(&self, latency_ms: f64) {
        let mut latencies = self.query_latencies.write().await;
        latencies.push_back(latency_ms);

        if latencies.len() > 1000 {
            latencies.pop_front();
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.time_travel_queries += 1;

        if !latencies.is_empty() {
            stats.avg_query_latency_ms = latencies.iter().sum::<f64>() / latencies.len() as f64;
        }
    }

    async fn maybe_create_auto_snapshot(&self) -> Result<(), StreamError> {
        let last = *self.last_snapshot.read().await;

        if last.elapsed() >= self.config.snapshot_interval {
            let current = *self.current_version.read().await;
            let name = format!("auto_{}", current);
            self.create_snapshot(&name).await?;

            let mut last_snapshot = self.last_snapshot.write().await;
            *last_snapshot = Instant::now();
        }

        Ok(())
    }

    fn apply_retention_policy(
        &self,
        versions: &mut BTreeMap<VersionId, VersionMetadata>,
        event_storage: &mut HashMap<VersionId, Vec<VersionedEvent<T>>>,
    ) -> Result<(), StreamError> {
        // Check max versions
        while versions.len() > self.config.max_versions {
            if let Some((&oldest_id, _)) = versions.iter().next() {
                // Don't delete version 0
                if oldest_id == 0 {
                    break;
                }

                versions.remove(&oldest_id);
                event_storage.remove(&oldest_id);
            } else {
                break;
            }
        }

        // Check max age
        let cutoff = SystemTime::now() - self.config.max_age;
        let mut to_remove = Vec::new();

        for (version_id, metadata) in versions.iter() {
            if *version_id == 0 {
                continue; // Keep initial version
            }

            if metadata.created_at < cutoff {
                to_remove.push(*version_id);
            }
        }

        for version_id in to_remove {
            versions.remove(&version_id);
            event_storage.remove(&version_id);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestEvent {
        id: u64,
        value: String,
    }

    #[tokio::test]
    async fn test_create_version() {
        let versioning = StreamVersioning::<TestEvent>::new(VersioningConfig::default());

        let events = vec![
            TestEvent {
                id: 1,
                value: "first".to_string(),
            },
            TestEvent {
                id: 2,
                value: "second".to_string(),
            },
        ];

        let version_id = versioning
            .create_version(events, "Test version")
            .await
            .unwrap();

        assert_eq!(version_id, 1);
        assert_eq!(versioning.current_version().await, 1);
    }

    #[tokio::test]
    async fn test_time_travel_query() {
        let versioning = StreamVersioning::<TestEvent>::new(VersioningConfig::default());

        // Create version 1
        let events1 = vec![TestEvent {
            id: 1,
            value: "v1".to_string(),
        }];
        versioning
            .create_version(events1, "Version 1")
            .await
            .unwrap();

        // Create version 2
        let events2 = vec![TestEvent {
            id: 2,
            value: "v2".to_string(),
        }];
        versioning
            .create_version(events2, "Version 2")
            .await
            .unwrap();

        // Query version 1
        let result = versioning.get_at_version(1).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].data.id, 1);

        // Query version 2
        let result = versioning.get_at_version(2).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_snapshot_create_and_restore() {
        let versioning = StreamVersioning::<TestEvent>::new(VersioningConfig::default());

        let events = vec![TestEvent {
            id: 1,
            value: "snapshot_test".to_string(),
        }];
        versioning
            .create_version(events, "Test version")
            .await
            .unwrap();

        // Create snapshot
        let snapshot_id = versioning.create_snapshot("test_snap").await.unwrap();
        assert!(snapshot_id.contains("test_snap"));

        // Add more events
        versioning
            .create_version(
                vec![TestEvent {
                    id: 2,
                    value: "after_snap".to_string(),
                }],
                "After snapshot",
            )
            .await
            .unwrap();

        // Restore snapshot
        let restored_version = versioning.restore_snapshot(&snapshot_id).await.unwrap();
        assert!(restored_version > 0);
    }

    #[tokio::test]
    async fn test_branching() {
        let versioning = StreamVersioning::<TestEvent>::new(VersioningConfig::default());

        // Create initial version
        versioning
            .create_version(
                vec![TestEvent {
                    id: 1,
                    value: "main".to_string(),
                }],
                "Initial",
            )
            .await
            .unwrap();

        // Create branch
        let branch_id = versioning
            .create_branch("feature", "Feature branch")
            .await
            .unwrap();

        // Switch to branch
        versioning.switch_branch(&branch_id).await.unwrap();
        assert_eq!(versioning.current_branch().await, branch_id);

        // Create version on branch
        versioning
            .create_version(
                vec![TestEvent {
                    id: 2,
                    value: "feature".to_string(),
                }],
                "Feature work",
            )
            .await
            .unwrap();

        // Switch back to main
        versioning.switch_branch("main").await.unwrap();
        assert_eq!(versioning.current_branch().await, "main");
    }

    #[tokio::test]
    async fn test_diff() {
        let versioning = StreamVersioning::<TestEvent>::new(VersioningConfig::default());

        // Version 1
        versioning
            .create_version(
                vec![TestEvent {
                    id: 1,
                    value: "v1".to_string(),
                }],
                "V1",
            )
            .await
            .unwrap();

        // Version 2
        versioning
            .create_version(
                vec![
                    TestEvent {
                        id: 1,
                        value: "v1".to_string(),
                    },
                    TestEvent {
                        id: 2,
                        value: "v2".to_string(),
                    },
                ],
                "V2",
            )
            .await
            .unwrap();

        let diff = versioning.diff(1, 2).await.unwrap();
        assert_eq!(diff.from_version, 1);
        assert_eq!(diff.to_version, 2);
        assert!(!diff.added.is_empty());
    }

    #[tokio::test]
    async fn test_changeset() {
        let versioning = StreamVersioning::<TestEvent>::new(VersioningConfig::default());

        versioning
            .create_version(
                vec![TestEvent {
                    id: 1,
                    value: "initial".to_string(),
                }],
                "Initial",
            )
            .await
            .unwrap();

        versioning
            .create_version(
                vec![
                    TestEvent {
                        id: 1,
                        value: "initial".to_string(),
                    },
                    TestEvent {
                        id: 2,
                        value: "added".to_string(),
                    },
                ],
                "Added",
            )
            .await
            .unwrap();

        let changeset = versioning.generate_changeset(1, 2).await.unwrap();
        assert!(!changeset.changes.is_empty());

        let add_changes: Vec<_> = changeset
            .changes
            .iter()
            .filter(|c| c.change_type == ChangeType::Add)
            .collect();
        assert!(!add_changes.is_empty());
    }

    #[tokio::test]
    async fn test_tagging() {
        let versioning = StreamVersioning::<TestEvent>::new(VersioningConfig::default());

        let version_id = versioning
            .create_version(
                vec![TestEvent {
                    id: 1,
                    value: "tagged".to_string(),
                }],
                "Tagged version",
            )
            .await
            .unwrap();

        versioning
            .tag_version(version_id, "release", "v1.0.0")
            .await
            .unwrap();

        let found = versioning.find_by_tag("release", "v1.0.0").await;
        assert!(found.contains(&version_id));
    }

    #[tokio::test]
    async fn test_relative_time_query() {
        let versioning = StreamVersioning::<TestEvent>::new(VersioningConfig::default());

        versioning
            .create_version(
                vec![TestEvent {
                    id: 1,
                    value: "recent".to_string(),
                }],
                "Recent",
            )
            .await
            .unwrap();

        // Query with relative time
        let query = TimeTravelQuery {
            target: TimeTravelTarget::RelativeTime(Duration::from_secs(0)),
            branch_id: None,
            filter: None,
            projection: None,
            limit: None,
            include_deleted: false,
        };

        let result = versioning.time_travel_query(query).await.unwrap();
        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn test_stats() {
        let versioning = StreamVersioning::<TestEvent>::new(VersioningConfig::default());

        versioning
            .create_version(
                vec![TestEvent {
                    id: 1,
                    value: "test".to_string(),
                }],
                "Test",
            )
            .await
            .unwrap();

        let stats = versioning.get_stats().await;
        assert!(stats.total_versions >= 1);
        assert!(stats.total_events >= 1);
    }

    #[tokio::test]
    async fn test_compact() {
        let config = VersioningConfig {
            compression_threshold: Duration::from_secs(0), // Immediate compression
            ..Default::default()
        };

        let versioning = StreamVersioning::<TestEvent>::new(config);

        versioning
            .create_version(
                vec![TestEvent {
                    id: 1,
                    value: "compact_test".to_string(),
                }],
                "To compact",
            )
            .await
            .unwrap();

        let _compacted = versioning.compact().await.unwrap();
        // Successfully compacted (number of events retained)
    }

    #[tokio::test]
    async fn test_delete_branch() {
        let versioning = StreamVersioning::<TestEvent>::new(VersioningConfig::default());

        let branch_id = versioning
            .create_branch("to_delete", "Will be deleted")
            .await
            .unwrap();

        versioning.delete_branch(&branch_id).await.unwrap();

        let branches = versioning.get_branches().await;
        assert!(!branches.iter().any(|b| b.branch_id == branch_id));
    }

    #[tokio::test]
    async fn test_cannot_delete_main_branch() {
        let versioning = StreamVersioning::<TestEvent>::new(VersioningConfig::default());

        let result = versioning.delete_branch("main").await;
        assert!(result.is_err());
    }
    #[tokio::test]
    async fn test_retention_policy_concurrency() {
        // Configure to trigger retention policy immediately
        let config = VersioningConfig {
            max_versions: 1,
            ..Default::default()
        };
        let versioning = StreamVersioning::<TestEvent>::new(config);

        // Create version 1
        versioning.create_version(vec![], "v1").await.unwrap();

        // Create version 2 - this should trigger retention policy and deadlock
        // because create_version holds locks and calls apply_retention_policy which wants locks
        versioning.create_version(vec![], "v2").await.unwrap();
    }
}
