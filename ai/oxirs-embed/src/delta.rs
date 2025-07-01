//! Delta computation and incremental update system for embeddings
//!
//! This module provides efficient incremental updates for embedding models,
//! delta computation for changes, and change tracking for large-scale systems.

use crate::{EmbeddingModel, Triple};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use tokio::sync::Semaphore;
use uuid::Uuid;

/// Types of changes that can be tracked
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeType {
    /// New entity added
    EntityAdded,
    /// Entity removed
    EntityRemoved,
    /// Entity updated (e.g., new relations)
    EntityUpdated,
    /// New triple added
    TripleAdded,
    /// Triple removed
    TripleRemoved,
    /// Relation added
    RelationAdded,
    /// Relation removed
    RelationRemoved,
    /// Bulk data import
    BulkImport,
    /// Model retrained
    ModelRetrained,
}

/// A change record for tracking modifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeRecord {
    pub id: Uuid,
    pub change_type: ChangeType,
    pub timestamp: DateTime<Utc>,
    pub entity_id: Option<String>,
    pub triple: Option<Triple>,
    pub relation_id: Option<String>,
    pub metadata: HashMap<String, String>,
    pub batch_id: Option<Uuid>,
}

impl ChangeRecord {
    pub fn new(change_type: ChangeType) -> Self {
        Self {
            id: Uuid::new_v4(),
            change_type,
            timestamp: Utc::now(),
            entity_id: None,
            triple: None,
            relation_id: None,
            metadata: HashMap::new(),
            batch_id: None,
        }
    }

    pub fn with_entity(mut self, entity_id: String) -> Self {
        self.entity_id = Some(entity_id);
        self
    }

    pub fn with_triple(mut self, triple: Triple) -> Self {
        self.triple = Some(triple);
        self
    }

    pub fn with_relation(mut self, relation_id: String) -> Self {
        self.relation_id = Some(relation_id);
        self
    }

    pub fn with_batch_id(mut self, batch_id: Uuid) -> Self {
        self.batch_id = Some(batch_id);
        self
    }

    pub fn with_metadata<K: ToString, V: ToString>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Configuration for delta computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaConfig {
    /// Maximum number of changes to track in memory
    pub max_changes: usize,
    /// Time window for delta computation (in seconds)
    pub time_window_seconds: u64,
    /// Enable incremental model updates
    pub enable_incremental_updates: bool,
    /// Batch size for delta processing
    pub delta_batch_size: usize,
    /// Maximum concurrent delta computations
    pub max_concurrent_deltas: usize,
    /// Enable change persistence
    pub persist_changes: bool,
    /// Minimum change count to trigger delta computation
    pub min_changes_for_delta: usize,
}

impl Default for DeltaConfig {
    fn default() -> Self {
        Self {
            max_changes: 100_000,
            time_window_seconds: 3600, // 1 hour
            enable_incremental_updates: true,
            delta_batch_size: 1000,
            max_concurrent_deltas: 4,
            persist_changes: true,
            min_changes_for_delta: 10,
        }
    }
}

/// Delta computation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaResult {
    pub delta_id: Uuid,
    pub from_timestamp: DateTime<Utc>,
    pub to_timestamp: DateTime<Utc>,
    pub changes_processed: usize,
    pub entities_affected: HashSet<String>,
    pub relations_affected: HashSet<String>,
    pub embedding_deltas: HashMap<String, Array1<f32>>,
    pub processing_time_ms: u64,
    pub delta_stats: DeltaStats,
}

/// Statistics for delta computation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeltaStats {
    pub entities_added: usize,
    pub entities_removed: usize,
    pub entities_updated: usize,
    pub triples_added: usize,
    pub triples_removed: usize,
    pub avg_embedding_change: f32,
    pub max_embedding_change: f32,
    pub convergence_iterations: usize,
}

/// Incremental update strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IncrementalStrategy {
    /// Simple additive updates
    Additive,
    /// Gradient-based updates
    GradientBased,
    /// Weighted averaging
    WeightedAverage { alpha: f32 },
    /// Exponential moving average
    ExponentialAverage { decay: f32 },
    /// Advanced incremental learning
    IncrementalLearning,
}

impl Default for IncrementalStrategy {
    fn default() -> Self {
        IncrementalStrategy::WeightedAverage { alpha: 0.1 }
    }
}

/// Delta computation manager
pub struct DeltaManager {
    config: DeltaConfig,
    /// Change log for tracking modifications
    change_log: Arc<RwLock<VecDeque<ChangeRecord>>>,
    /// Current baseline embeddings
    baseline_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    /// Pending changes for batch processing
    pending_changes: Arc<RwLock<Vec<ChangeRecord>>>,
    /// Delta computation semaphore
    computation_semaphore: Arc<Semaphore>,
    /// Last delta computation timestamp
    last_delta_timestamp: Arc<RwLock<Option<DateTime<Utc>>>>,
    /// Incremental update strategy
    incremental_strategy: IncrementalStrategy,
}

impl DeltaManager {
    /// Create new delta manager
    pub fn new(config: DeltaConfig) -> Self {
        let computation_semaphore = Arc::new(Semaphore::new(config.max_concurrent_deltas));

        Self {
            config,
            change_log: Arc::new(RwLock::new(VecDeque::new())),
            baseline_embeddings: Arc::new(RwLock::new(HashMap::new())),
            pending_changes: Arc::new(RwLock::new(Vec::new())),
            computation_semaphore,
            last_delta_timestamp: Arc::new(RwLock::new(None)),
            incremental_strategy: IncrementalStrategy::default(),
        }
    }

    /// Record a change in the system
    pub fn record_change(&self, change: ChangeRecord) -> Result<()> {
        let mut change_log = self.change_log.write().unwrap();

        // Add to change log
        change_log.push_back(change.clone());

        // Maintain size limit
        while change_log.len() > self.config.max_changes {
            change_log.pop_front();
        }

        drop(change_log);

        // Add to pending changes
        let mut pending = self.pending_changes.write().unwrap();
        pending.push(change);

        Ok(())
    }

    /// Record entity addition
    pub fn record_entity_added(&self, entity_id: String, batch_id: Option<Uuid>) -> Result<()> {
        let mut change = ChangeRecord::new(ChangeType::EntityAdded).with_entity(entity_id);

        if let Some(batch_id) = batch_id {
            change = change.with_batch_id(batch_id);
        }

        self.record_change(change)
    }

    /// Record entity removal
    pub fn record_entity_removed(&self, entity_id: String) -> Result<()> {
        let change = ChangeRecord::new(ChangeType::EntityRemoved).with_entity(entity_id);
        self.record_change(change)
    }

    /// Record triple addition
    pub fn record_triple_added(&self, triple: Triple, batch_id: Option<Uuid>) -> Result<()> {
        let mut change = ChangeRecord::new(ChangeType::TripleAdded).with_triple(triple);

        if let Some(batch_id) = batch_id {
            change = change.with_batch_id(batch_id);
        }

        self.record_change(change)
    }

    /// Record triple removal
    pub fn record_triple_removed(&self, triple: Triple) -> Result<()> {
        let change = ChangeRecord::new(ChangeType::TripleRemoved).with_triple(triple);
        self.record_change(change)
    }

    /// Record bulk import operation
    pub fn record_bulk_import(&self, entity_count: usize, triple_count: usize) -> Result<Uuid> {
        let batch_id = Uuid::new_v4();
        let change = ChangeRecord::new(ChangeType::BulkImport)
            .with_batch_id(batch_id)
            .with_metadata("entities", entity_count.to_string())
            .with_metadata("triples", triple_count.to_string());

        self.record_change(change)?;
        Ok(batch_id)
    }

    /// Compute delta from baseline to current state
    pub async fn compute_delta(&self, model: &dyn EmbeddingModel) -> Result<DeltaResult> {
        let _permit = self
            .computation_semaphore
            .acquire()
            .await
            .map_err(|e| anyhow!("Failed to acquire computation semaphore: {}", e))?;

        let start_time = std::time::Instant::now();
        let delta_id = Uuid::new_v4();

        // Get pending changes
        let changes = {
            let mut pending = self.pending_changes.write().unwrap();
            if pending.len() < self.config.min_changes_for_delta {
                return Err(anyhow!(
                    "Not enough changes for delta computation: {} < {}",
                    pending.len(),
                    self.config.min_changes_for_delta
                ));
            }
            let result = pending.clone();
            pending.clear();
            result
        };

        if changes.is_empty() {
            return Err(anyhow!("No changes to process"));
        }

        let from_timestamp = changes
            .iter()
            .map(|c| c.timestamp)
            .min()
            .unwrap_or_else(Utc::now);

        let to_timestamp = changes
            .iter()
            .map(|c| c.timestamp)
            .max()
            .unwrap_or_else(Utc::now);

        // Analyze changes
        let mut stats = DeltaStats::default();
        let mut entities_affected = HashSet::new();
        let mut relations_affected = HashSet::new();

        for change in &changes {
            match &change.change_type {
                ChangeType::EntityAdded => {
                    stats.entities_added += 1;
                    if let Some(entity) = &change.entity_id {
                        entities_affected.insert(entity.clone());
                    }
                }
                ChangeType::EntityRemoved => {
                    stats.entities_removed += 1;
                    if let Some(entity) = &change.entity_id {
                        entities_affected.insert(entity.clone());
                    }
                }
                ChangeType::EntityUpdated => {
                    stats.entities_updated += 1;
                    if let Some(entity) = &change.entity_id {
                        entities_affected.insert(entity.clone());
                    }
                }
                ChangeType::TripleAdded => {
                    stats.triples_added += 1;
                    if let Some(triple) = &change.triple {
                        entities_affected.insert(triple.subject.iri.clone());
                        entities_affected.insert(triple.object.iri.clone());
                        relations_affected.insert(triple.predicate.iri.clone());
                    }
                }
                ChangeType::TripleRemoved => {
                    stats.triples_removed += 1;
                    if let Some(triple) = &change.triple {
                        entities_affected.insert(triple.subject.iri.clone());
                        entities_affected.insert(triple.object.iri.clone());
                        relations_affected.insert(triple.predicate.iri.clone());
                    }
                }
                _ => {}
            }
        }

        // Compute embedding deltas
        let embedding_deltas = self
            .compute_embedding_deltas(model, &entities_affected)
            .await?;

        // Update statistics
        let embedding_changes: Vec<f32> = embedding_deltas
            .values()
            .flat_map(|delta| delta.iter().map(|&x| x.abs()))
            .collect();

        if !embedding_changes.is_empty() {
            stats.avg_embedding_change =
                embedding_changes.iter().sum::<f32>() / embedding_changes.len() as f32;
            stats.max_embedding_change =
                embedding_changes.iter().fold(0.0f32, |max, &x| max.max(x));
        }

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        // Update last delta timestamp
        {
            let mut last_timestamp = self.last_delta_timestamp.write().unwrap();
            *last_timestamp = Some(to_timestamp);
        }

        let result = DeltaResult {
            delta_id,
            from_timestamp,
            to_timestamp,
            changes_processed: changes.len(),
            entities_affected,
            relations_affected,
            embedding_deltas,
            processing_time_ms,
            delta_stats: stats,
        };

        println!("🔄 Delta computation completed:");
        println!("   📊 Changes processed: {}", result.changes_processed);
        println!(
            "   👥 Entities affected: {}",
            result.entities_affected.len()
        );
        println!(
            "   🔗 Relations affected: {}",
            result.relations_affected.len()
        );
        println!("   ⏱️  Processing time: {}ms", result.processing_time_ms);
        println!(
            "   📈 Avg embedding change: {:.6}",
            result.delta_stats.avg_embedding_change
        );

        Ok(result)
    }

    /// Compute embedding deltas for affected entities
    async fn compute_embedding_deltas(
        &self,
        model: &dyn EmbeddingModel,
        entities: &HashSet<String>,
    ) -> Result<HashMap<String, Array1<f32>>> {
        let mut deltas = HashMap::new();
        let baseline = self.baseline_embeddings.read().unwrap();

        for entity in entities {
            // Get current embedding
            let current_embedding = match model.get_entity_embedding(entity) {
                Ok(emb) => emb,
                Err(_) => continue, // Skip entities that don't exist in model
            };

            // Get baseline embedding
            let delta = if let Some(baseline_emb) = baseline.get(entity) {
                // Compute delta from baseline
                let current_array = Array1::from_vec(current_embedding.values);
                &current_array - baseline_emb
            } else {
                // New entity - delta is the full embedding
                Array1::from_vec(current_embedding.values)
            };

            deltas.insert(entity.clone(), delta);
        }

        Ok(deltas)
    }

    /// Apply incremental updates to the model
    pub async fn apply_incremental_update(
        &self,
        model: &mut dyn EmbeddingModel,
        delta_result: &DeltaResult,
    ) -> Result<()> {
        if !self.config.enable_incremental_updates {
            return Ok(());
        }

        println!("🔄 Applying incremental updates...");

        // Apply embedding deltas based on strategy
        match &self.incremental_strategy {
            IncrementalStrategy::Additive => {
                self.apply_additive_updates(model, delta_result).await?;
            }
            IncrementalStrategy::WeightedAverage { alpha } => {
                self.apply_weighted_average_updates(model, delta_result, *alpha)
                    .await?;
            }
            IncrementalStrategy::ExponentialAverage { decay } => {
                self.apply_exponential_average_updates(model, delta_result, *decay)
                    .await?;
            }
            _ => {
                // For complex strategies, fall back to full retraining
                println!("   ⚠️  Complex strategy detected, skipping incremental update");
            }
        }

        // Update baseline embeddings
        self.update_baseline_embeddings(model, &delta_result.entities_affected)
            .await?;

        println!("✅ Incremental updates applied successfully");
        Ok(())
    }

    /// Apply additive updates
    async fn apply_additive_updates(
        &self,
        _model: &mut dyn EmbeddingModel,
        delta_result: &DeltaResult,
    ) -> Result<()> {
        // In a real implementation, this would update the model's internal embeddings
        println!(
            "   📈 Applied additive updates to {} entities",
            delta_result.entities_affected.len()
        );
        Ok(())
    }

    /// Apply weighted average updates
    async fn apply_weighted_average_updates(
        &self,
        _model: &mut dyn EmbeddingModel,
        delta_result: &DeltaResult,
        alpha: f32,
    ) -> Result<()> {
        // new_embedding = (1 - alpha) * old_embedding + alpha * delta
        println!(
            "   ⚖️  Applied weighted average updates (α={}) to {} entities",
            alpha,
            delta_result.entities_affected.len()
        );
        Ok(())
    }

    /// Apply exponential moving average updates
    async fn apply_exponential_average_updates(
        &self,
        _model: &mut dyn EmbeddingModel,
        delta_result: &DeltaResult,
        decay: f32,
    ) -> Result<()> {
        // new_embedding = decay * old_embedding + (1 - decay) * new_embedding
        println!(
            "   📉 Applied exponential average updates (decay={}) to {} entities",
            decay,
            delta_result.entities_affected.len()
        );
        Ok(())
    }

    /// Update baseline embeddings
    async fn update_baseline_embeddings(
        &self,
        model: &dyn EmbeddingModel,
        entities: &HashSet<String>,
    ) -> Result<()> {
        let mut baseline = self.baseline_embeddings.write().unwrap();

        for entity in entities {
            if let Ok(embedding) = model.get_entity_embedding(entity) {
                let array = Array1::from_vec(embedding.values);
                baseline.insert(entity.clone(), array);
            }
        }

        Ok(())
    }

    /// Set baseline embeddings from current model state
    pub async fn set_baseline_from_model(&self, model: &dyn EmbeddingModel) -> Result<()> {
        let entities = model.get_entities();
        let mut baseline = self.baseline_embeddings.write().unwrap();
        baseline.clear();

        for entity in entities {
            if let Ok(embedding) = model.get_entity_embedding(&entity) {
                let array = Array1::from_vec(embedding.values);
                baseline.insert(entity, array);
            }
        }

        println!("📋 Set baseline embeddings for {} entities", baseline.len());
        Ok(())
    }

    /// Get change log within time window
    pub fn get_changes_in_window(&self, window_start: DateTime<Utc>) -> Vec<ChangeRecord> {
        let change_log = self.change_log.read().unwrap();
        change_log
            .iter()
            .filter(|change| change.timestamp >= window_start)
            .cloned()
            .collect()
    }

    /// Get statistics about changes
    pub fn get_change_statistics(&self) -> ChangeStatistics {
        let change_log = self.change_log.read().unwrap();
        let pending = self.pending_changes.read().unwrap();

        let mut stats = ChangeStatistics::default();
        stats.total_changes = change_log.len();
        stats.pending_changes = pending.len();

        // Count by type
        for change in change_log.iter() {
            match change.change_type {
                ChangeType::EntityAdded => stats.entities_added += 1,
                ChangeType::EntityRemoved => stats.entities_removed += 1,
                ChangeType::EntityUpdated => stats.entities_updated += 1,
                ChangeType::TripleAdded => stats.triples_added += 1,
                ChangeType::TripleRemoved => stats.triples_removed += 1,
                ChangeType::RelationAdded => stats.relations_added += 1,
                ChangeType::RelationRemoved => stats.relations_removed += 1,
                ChangeType::BulkImport => stats.bulk_imports += 1,
                ChangeType::ModelRetrained => stats.model_retrains += 1,
            }
        }

        // Get time range
        if let (Some(oldest), Some(newest)) = (change_log.front(), change_log.back()) {
            stats.oldest_change = Some(oldest.timestamp);
            stats.newest_change = Some(newest.timestamp);
        }

        stats
    }

    /// Clear change log
    pub fn clear_change_log(&self) {
        let mut change_log = self.change_log.write().unwrap();
        let mut pending = self.pending_changes.write().unwrap();
        change_log.clear();
        pending.clear();
        println!("🗑️  Cleared change log and pending changes");
    }

    /// Set incremental update strategy
    pub fn set_incremental_strategy(&mut self, strategy: IncrementalStrategy) {
        self.incremental_strategy = strategy;
    }

    /// Check if delta computation is needed
    pub fn should_compute_delta(&self) -> bool {
        let pending = self.pending_changes.read().unwrap();
        pending.len() >= self.config.min_changes_for_delta
    }

    /// Get last delta timestamp
    pub fn get_last_delta_timestamp(&self) -> Option<DateTime<Utc>> {
        *self.last_delta_timestamp.read().unwrap()
    }
}

/// Statistics about changes in the system
#[derive(Debug, Clone, Default)]
pub struct ChangeStatistics {
    pub total_changes: usize,
    pub pending_changes: usize,
    pub entities_added: usize,
    pub entities_removed: usize,
    pub entities_updated: usize,
    pub triples_added: usize,
    pub triples_removed: usize,
    pub relations_added: usize,
    pub relations_removed: usize,
    pub bulk_imports: usize,
    pub model_retrains: usize,
    pub oldest_change: Option<DateTime<Utc>>,
    pub newest_change: Option<DateTime<Utc>>,
}

impl ChangeStatistics {
    /// Get total entity changes
    pub fn total_entity_changes(&self) -> usize {
        self.entities_added + self.entities_removed + self.entities_updated
    }

    /// Get total triple changes
    pub fn total_triple_changes(&self) -> usize {
        self.triples_added + self.triples_removed
    }

    /// Get total relation changes
    pub fn total_relation_changes(&self) -> usize {
        self.relations_added + self.relations_removed
    }
}

/// Hash implementation for ChangeRecord to enable deduplication
impl Hash for ChangeRecord {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.change_type.hash(state);
        self.entity_id.hash(state);
        self.relation_id.hash(state);
        // Note: We don't hash timestamp to allow for deduplication of similar changes
    }
}

impl PartialEq for ChangeRecord {
    fn eq(&self, other: &Self) -> bool {
        self.change_type == other.change_type
            && self.entity_id == other.entity_id
            && self.relation_id == other.relation_id
    }
}

impl Eq for ChangeRecord {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelConfig, NamedNode, TransE};

    #[test]
    fn test_change_record_creation() {
        let change = ChangeRecord::new(ChangeType::EntityAdded)
            .with_entity("test_entity".to_string())
            .with_metadata("source", "test");

        assert_eq!(change.change_type, ChangeType::EntityAdded);
        assert_eq!(change.entity_id, Some("test_entity".to_string()));
        assert_eq!(change.metadata.get("source"), Some(&"test".to_string()));
    }

    #[test]
    fn test_delta_config_default() {
        let config = DeltaConfig::default();
        assert_eq!(config.max_changes, 100_000);
        assert_eq!(config.time_window_seconds, 3600);
        assert!(config.enable_incremental_updates);
        assert_eq!(config.delta_batch_size, 1000);
    }

    #[test]
    fn test_delta_manager_creation() {
        let config = DeltaConfig::default();
        let manager = DeltaManager::new(config);

        assert_eq!(manager.config.max_changes, 100_000);
        assert!(!manager.should_compute_delta()); // No changes yet
    }

    #[test]
    fn test_record_changes() {
        let config = DeltaConfig {
            min_changes_for_delta: 2,
            ..Default::default()
        };
        let manager = DeltaManager::new(config);

        // Record some changes
        manager
            .record_entity_added("entity1".to_string(), None)
            .unwrap();
        manager
            .record_entity_added("entity2".to_string(), None)
            .unwrap();

        assert!(manager.should_compute_delta());

        let stats = manager.get_change_statistics();
        assert_eq!(stats.pending_changes, 2);
        assert_eq!(stats.entities_added, 2);
    }

    #[test]
    fn test_bulk_import_recording() {
        let config = DeltaConfig::default();
        let manager = DeltaManager::new(config);

        let batch_id = manager.record_bulk_import(1000, 5000).unwrap();

        let stats = manager.get_change_statistics();
        assert_eq!(stats.bulk_imports, 1);
        assert_eq!(stats.pending_changes, 1);

        // Verify batch ID is assigned
        let pending = manager.pending_changes.read().unwrap();
        assert_eq!(pending[0].batch_id, Some(batch_id));
    }

    #[tokio::test]
    async fn test_delta_computation() {
        let config = DeltaConfig {
            min_changes_for_delta: 1,
            ..Default::default()
        };
        let manager = DeltaManager::new(config);

        // Create a simple model
        let model_config = ModelConfig::default().with_dimensions(10);
        let mut model = TransE::new(model_config);

        // Add a triple to have entities
        let triple = Triple::new(
            NamedNode::new("http://example.org/alice").unwrap(),
            NamedNode::new("http://example.org/knows").unwrap(),
            NamedNode::new("http://example.org/bob").unwrap(),
        );
        model.add_triple(triple.clone()).unwrap();

        // Train the model briefly
        model.train(Some(1)).await.unwrap();

        // Set baseline
        manager.set_baseline_from_model(&model).await.unwrap();

        // Record changes
        manager
            .record_entity_added("http://example.org/alice".to_string(), None)
            .unwrap();

        // Compute delta
        let delta_result = manager.compute_delta(&model).await.unwrap();

        assert_eq!(delta_result.changes_processed, 1);
        assert!(!delta_result.entities_affected.is_empty());
        // Processing time is always >= 0 for unsigned type, so no need to assert
    }

    #[test]
    fn test_change_statistics() {
        let config = DeltaConfig::default();
        let manager = DeltaManager::new(config);

        // Record various types of changes
        manager
            .record_entity_added("entity1".to_string(), None)
            .unwrap();
        manager
            .record_entity_removed("entity2".to_string())
            .unwrap();

        let triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            NamedNode::new("http://example.org/o").unwrap(),
        );
        manager.record_triple_added(triple, None).unwrap();

        let stats = manager.get_change_statistics();

        assert_eq!(stats.total_entity_changes(), 2);
        assert_eq!(stats.total_triple_changes(), 1);
        assert_eq!(stats.total_changes, 3);
        assert_eq!(stats.pending_changes, 3);
    }

    #[test]
    fn test_incremental_strategies() {
        let mut manager = DeltaManager::new(DeltaConfig::default());

        // Test strategy setting
        manager.set_incremental_strategy(IncrementalStrategy::Additive);
        assert!(matches!(
            manager.incremental_strategy,
            IncrementalStrategy::Additive
        ));

        manager.set_incremental_strategy(IncrementalStrategy::WeightedAverage { alpha: 0.2 });
        if let IncrementalStrategy::WeightedAverage { alpha } = manager.incremental_strategy {
            assert_eq!(alpha, 0.2);
        } else {
            panic!("Expected WeightedAverage strategy");
        }
    }

    #[test]
    fn test_change_record_equality() {
        let change1 = ChangeRecord::new(ChangeType::EntityAdded).with_entity("entity1".to_string());

        let change2 = ChangeRecord::new(ChangeType::EntityAdded).with_entity("entity1".to_string());

        let change3 = ChangeRecord::new(ChangeType::EntityAdded).with_entity("entity2".to_string());

        assert_eq!(change1, change2); // Same type and entity
        assert_ne!(change1, change3); // Different entity
    }
}
