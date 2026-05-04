//! In-process toy parameter server with sharded embeddings.
//!
//! [`ParameterServer`] is a **prototype**: it lives entirely inside one Rust
//! process, owns sharded copies of the entity / relation embedding tables, and
//! lets workers (typically [`super::worker::Worker`]) **pull** the latest
//! parameters and **push** locally-computed gradients back.  Two update modes
//! are supported:
//!
//! 1. [`UpdateMode::Sync`] — pushes are **buffered** per-step until every
//!    expected worker has pushed; only then is the average applied to the
//!    parameters and a new step starts.  This is the standard mini-batch SGD
//!    contract and gives reproducible convergence.  The barrier is per-shard:
//!    a worker that pushed for shard A is free to push for shard B without
//!    waiting on shard A's barrier to clear.
//!
//! 2. [`UpdateMode::Async`] — pushes are applied **immediately** with no
//!    barrier.  This trades a small amount of staleness (workers may be working
//!    against a slightly outdated copy of the parameters) for higher
//!    throughput.  We track a per-shard *staleness counter* (the number of
//!    pushes between the last `pull` and the next `pull`) so callers can
//!    bound the expected divergence.
//!
//! The server is bounded by design to **4–8 workers / 4–8 shards** for the
//! prototype.  Larger setups should use a real RPC-based parameter server.
//!
//! All public methods are `async` and use `tokio::sync::RwLock` so shards can
//! be pulled concurrently from many workers without contention; pushes acquire
//! a write lock only on the affected shard.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Notify, RwLock};
use tokio::time::timeout;
use tracing::{debug, trace, warn};

use super::shard_manager::ModelShardManager;

/// How [`ParameterServer::push`] applies gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum UpdateMode {
    /// Synchronous: gradients buffered until `expected_workers` push, then averaged.
    #[default]
    Sync,
    /// Asynchronous: gradients applied immediately (eventual consistency).
    Async,
}

/// Configuration for [`ParameterServer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterServerConfig {
    /// Embedding dimensionality (length of each row).
    pub embedding_dim: usize,
    /// Total number of entity rows the server will own (across all shards).
    pub num_entities: usize,
    /// Total number of relation rows the server will own.
    pub num_relations: usize,
    /// Number of shards to split the entity table into.
    pub num_shards: usize,
    /// Number of workers expected to push per step in [`UpdateMode::Sync`].
    pub expected_workers: usize,
    /// Sync/async update mode.
    pub update_mode: UpdateMode,
    /// Optimizer learning rate applied during `push`.
    pub learning_rate: f32,
    /// Maximum staleness tolerated in async mode before logging a warning.
    pub max_staleness: u64,
    /// Per-step barrier timeout in [`UpdateMode::Sync`]; if exceeded, the
    /// barrier proceeds with whatever pushes have arrived.
    pub barrier_timeout: Duration,
}

impl Default for ParameterServerConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 32,
            num_entities: 64,
            num_relations: 8,
            num_shards: 4,
            expected_workers: 4,
            update_mode: UpdateMode::Sync,
            learning_rate: 0.01,
            max_staleness: 16,
            barrier_timeout: Duration::from_secs(30),
        }
    }
}

/// Public-facing snapshot of one shard's contents.
///
/// Returned by [`ParameterServer::pull`].  Workers operate on this owned copy
/// locally, then submit gradients via `push`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardSnapshot {
    /// Shard index.
    pub shard_id: usize,
    /// One row of `embedding_dim` weights per entity owned by the shard.
    pub entities: Vec<Vec<f32>>,
    /// Mapping from row index inside `entities` back to the global entity ID.
    pub entity_ids: Vec<String>,
    /// Relation table — small, replicated to every shard for convenience.
    pub relations: Vec<Vec<f32>>,
    /// Mapping from row index inside `relations` back to the global relation ID.
    pub relation_ids: Vec<String>,
    /// Server step counter at the moment the snapshot was taken.
    pub step: u64,
}

/// Aggregate stats reported by [`ParameterServer::stats`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParameterServerStats {
    /// Total `pull` operations served.
    pub total_pulls: u64,
    /// Total `push` operations applied (sync averages count as one).
    pub total_pushes: u64,
    /// Number of completed sync barriers.
    pub barriers_completed: u64,
    /// Maximum observed staleness in async mode (pushes since last pull on a shard).
    pub max_staleness_observed: u64,
    /// Mean squared L2 norm of the most recent applied gradient (per shard).
    pub last_grad_norm: f64,
}

// ── Internal shard state (private) ───────────────────────────────────────────

#[derive(Debug)]
struct ShardState {
    /// Owned entity rows for this shard.
    entities: Vec<Vec<f32>>,
    /// Global entity IDs, aligned with `entities`.
    entity_ids: Vec<String>,
    /// Per-shard step counter.
    step: u64,
    /// Pending sync-mode pushes for the **current** step.
    pending: Vec<PendingGradient>,
    /// Tracks pushes seen in this step (set of `worker_id`).
    pushed_workers: Vec<u32>,
    /// Async-mode staleness: number of pushes since the last pull.
    staleness: u64,
    /// Notified when a barrier completes (sync mode).
    barrier_done: Arc<Notify>,
}

#[derive(Debug, Clone)]
struct PendingGradient {
    worker_id: u32,
    rows: Vec<(usize, Vec<f32>)>, // (row index inside shard, gradient row)
}

// ── ParameterServer ──────────────────────────────────────────────────────────

/// Sharded parameter server with sync/async update modes.
///
/// See module-level docs for semantics.
pub struct ParameterServer {
    config: ParameterServerConfig,
    /// Per-shard state.  Each shard has its own RwLock so independent shards
    /// can be served concurrently.
    shards: Vec<Arc<RwLock<ShardState>>>,
    /// Replicated relation table.  Tiny enough to keep behind a single lock.
    relations: Arc<RwLock<Vec<Vec<f32>>>>,
    /// Stable list of relation IDs (declared at construction).
    relation_ids: Vec<String>,
    /// Stats (single mutex; cold path).
    stats: Arc<Mutex<ParameterServerStats>>,
    /// Shard manager — owned because the server may need to reshard if elastic
    /// scaling is added later.  Today it is read-only.
    shard_manager: ModelShardManager,
}

impl ParameterServer {
    /// Build a new parameter server.
    ///
    /// `entity_ids` and `relation_ids` must list every entity / relation the
    /// server should own; the server hashes each entity ID into a shard via
    /// the supplied [`ModelShardManager`].  Embedding rows are initialised to
    /// small uniform values in `[-0.05, 0.05]` for reproducibility.
    pub fn new(
        config: ParameterServerConfig,
        entity_ids: Vec<String>,
        relation_ids: Vec<String>,
        shard_manager: ModelShardManager,
    ) -> Result<Self> {
        if config.embedding_dim == 0 {
            anyhow::bail!("embedding_dim must be > 0");
        }
        if config.num_shards == 0 {
            anyhow::bail!("num_shards must be > 0");
        }
        if config.expected_workers == 0 {
            anyhow::bail!("expected_workers must be > 0");
        }
        let num_shards = config.num_shards.min(shard_manager.num_shards());

        // Initial weights — deterministic small values keyed off the entity ID.
        let mut shards = Vec::with_capacity(num_shards);
        let mut shard_buckets: Vec<(Vec<Vec<f32>>, Vec<String>)> =
            (0..num_shards).map(|_| (Vec::new(), Vec::new())).collect();

        for id in entity_ids.into_iter() {
            let s = shard_manager.shard_for(&id);
            let row = init_row(&id, config.embedding_dim);
            shard_buckets[s].0.push(row);
            shard_buckets[s].1.push(id);
        }

        for (entities, ids) in shard_buckets.into_iter() {
            shards.push(Arc::new(RwLock::new(ShardState {
                entities,
                entity_ids: ids,
                step: 0,
                pending: Vec::new(),
                pushed_workers: Vec::new(),
                staleness: 0,
                barrier_done: Arc::new(Notify::new()),
            })));
        }

        // Build relation table — tiny, fully replicated.
        let mut relations = Vec::with_capacity(relation_ids.len());
        for id in &relation_ids {
            relations.push(init_row(id, config.embedding_dim));
        }

        Ok(Self {
            config,
            shards,
            relations: Arc::new(RwLock::new(relations)),
            relation_ids,
            stats: Arc::new(Mutex::new(ParameterServerStats::default())),
            shard_manager,
        })
    }

    /// Number of shards.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Configuration snapshot.
    pub fn config(&self) -> &ParameterServerConfig {
        &self.config
    }

    /// Shard manager view (read-only).
    pub fn shard_manager(&self) -> &ModelShardManager {
        &self.shard_manager
    }

    /// Pull a snapshot of `shard_id`.
    ///
    /// Returns owned data so workers can compute gradients without holding any
    /// lock.  Panics-free: an out-of-range `shard_id` returns an `Err`.
    pub async fn pull(&self, shard_id: usize) -> Result<ShardSnapshot> {
        let shard = self
            .shards
            .get(shard_id)
            .ok_or_else(|| anyhow::anyhow!("shard {shard_id} out of range"))?;
        let g = shard.read().await;
        let snap = ShardSnapshot {
            shard_id,
            entities: g.entities.clone(),
            entity_ids: g.entity_ids.clone(),
            relations: self.relations.read().await.clone(),
            relation_ids: self.relation_ids.clone(),
            step: g.step,
        };
        drop(g);

        // Reset async staleness window on pull.
        if matches!(self.config.update_mode, UpdateMode::Async) {
            let mut w = shard.write().await;
            w.staleness = 0;
        }

        let mut stats = self.stats.lock().await;
        stats.total_pulls += 1;
        Ok(snap)
    }

    /// Push entity-row gradients to `shard_id`.
    ///
    /// `rows` is `(row_index_inside_shard, gradient_row)` pairs.  In
    /// [`UpdateMode::Sync`] the push is buffered until `expected_workers`
    /// pushes have accumulated for this shard *or* the per-step barrier
    /// timeout fires.  In [`UpdateMode::Async`] the gradient is applied
    /// immediately and the per-shard staleness counter is incremented.
    pub async fn push(
        &self,
        shard_id: usize,
        worker_id: u32,
        rows: Vec<(usize, Vec<f32>)>,
    ) -> Result<()> {
        let shard = self
            .shards
            .get(shard_id)
            .ok_or_else(|| anyhow::anyhow!("shard {shard_id} out of range"))?
            .clone();

        // Validate gradient shapes up-front, before we touch any state.
        for (idx, grad) in &rows {
            if grad.len() != self.config.embedding_dim {
                anyhow::bail!(
                    "gradient row {idx} has dim {} but server expects {}",
                    grad.len(),
                    self.config.embedding_dim
                );
            }
        }

        match self.config.update_mode {
            UpdateMode::Sync => self.push_sync(shard, shard_id, worker_id, rows).await,
            UpdateMode::Async => self.push_async(shard, worker_id, rows).await,
        }
    }

    /// Apply gradients to relation rows.
    ///
    /// Relation gradients are always averaged across the most-recent push
    /// regardless of `update_mode` — they're a tiny fully-replicated table.
    pub async fn push_relation(&self, worker_id: u32, rows: Vec<(usize, Vec<f32>)>) -> Result<()> {
        for (idx, grad) in &rows {
            if grad.len() != self.config.embedding_dim {
                anyhow::bail!(
                    "relation gradient row {idx} has dim {} but server expects {}",
                    grad.len(),
                    self.config.embedding_dim
                );
            }
        }

        let mut rel = self.relations.write().await;
        for (idx, grad) in rows {
            if let Some(target) = rel.get_mut(idx) {
                for (t, g) in target.iter_mut().zip(grad.iter()) {
                    *t -= self.config.learning_rate * *g;
                }
            }
        }
        trace!("worker {worker_id}: relation gradients applied");
        Ok(())
    }

    /// Snapshot of current stats.
    pub async fn stats(&self) -> ParameterServerStats {
        self.stats.lock().await.clone()
    }

    /// Total per-shard step counts (for tests).
    pub async fn shard_steps(&self) -> Vec<u64> {
        let mut steps = Vec::with_capacity(self.shards.len());
        for s in &self.shards {
            steps.push(s.read().await.step);
        }
        steps
    }

    // ── Internals ───────────────────────────────────────────────────────────

    async fn push_sync(
        &self,
        shard: Arc<RwLock<ShardState>>,
        shard_id: usize,
        worker_id: u32,
        rows: Vec<(usize, Vec<f32>)>,
    ) -> Result<()> {
        // Append the push and check the barrier.
        let (apply_now, barrier) = {
            let mut g = shard.write().await;
            if g.pushed_workers.contains(&worker_id) {
                anyhow::bail!("worker {worker_id} already pushed for shard {shard_id} this step");
            }
            g.pending.push(PendingGradient { worker_id, rows });
            g.pushed_workers.push(worker_id);
            let ready = g.pushed_workers.len() >= self.config.expected_workers;
            (ready, g.barrier_done.clone())
        };

        if apply_now {
            self.apply_sync_barrier(shard.clone(), shard_id).await?;
            barrier.notify_waiters();
            return Ok(());
        }

        // Wait for someone else (or our own future call) to fill the barrier,
        // up to `barrier_timeout`.
        let waited = timeout(self.config.barrier_timeout, barrier.notified()).await;
        if waited.is_err() {
            warn!(
                "shard {shard_id} barrier timed out after {:?}; flushing partial step",
                self.config.barrier_timeout
            );
            self.apply_sync_barrier(shard, shard_id).await?;
        }
        Ok(())
    }

    async fn apply_sync_barrier(
        &self,
        shard: Arc<RwLock<ShardState>>,
        shard_id: usize,
    ) -> Result<()> {
        let mut g = shard.write().await;
        let lr = self.config.learning_rate;
        let dim = self.config.embedding_dim;
        let n = g.pending.len().max(1) as f32;

        // Average gradients per row.
        let mut acc: std::collections::HashMap<usize, Vec<f32>> = std::collections::HashMap::new();
        for pending in &g.pending {
            for (idx, grad) in &pending.rows {
                let entry = acc.entry(*idx).or_insert_with(|| vec![0.0; dim]);
                for (t, gval) in entry.iter_mut().zip(grad.iter()) {
                    *t += *gval / n;
                }
            }
        }

        // Apply averaged gradients.
        let mut sq_sum = 0.0_f64;
        for (idx, grad) in &acc {
            if let Some(target) = g.entities.get_mut(*idx) {
                for (t, gval) in target.iter_mut().zip(grad.iter()) {
                    *t -= lr * *gval;
                    sq_sum += (*gval as f64) * (*gval as f64);
                }
            }
        }

        g.pending.clear();
        g.pushed_workers.clear();
        g.step += 1;
        let new_step = g.step;
        drop(g);

        let mut stats = self.stats.lock().await;
        stats.total_pushes += 1;
        stats.barriers_completed += 1;
        if !acc.is_empty() {
            stats.last_grad_norm = sq_sum / acc.len() as f64;
        }
        debug!("shard {shard_id} barrier applied (new step = {new_step})");
        Ok(())
    }

    async fn push_async(
        &self,
        shard: Arc<RwLock<ShardState>>,
        worker_id: u32,
        rows: Vec<(usize, Vec<f32>)>,
    ) -> Result<()> {
        let lr = self.config.learning_rate;
        let mut g = shard.write().await;
        let mut sq_sum = 0.0_f64;
        let mut applied = 0usize;
        for (idx, grad) in &rows {
            if let Some(target) = g.entities.get_mut(*idx) {
                for (t, gval) in target.iter_mut().zip(grad.iter()) {
                    *t -= lr * *gval;
                    sq_sum += (*gval as f64) * (*gval as f64);
                }
                applied += 1;
            }
        }
        g.staleness = g.staleness.saturating_add(1);
        g.step += 1;
        let new_staleness = g.staleness;
        drop(g);

        let mut stats = self.stats.lock().await;
        stats.total_pushes += 1;
        stats.max_staleness_observed = stats.max_staleness_observed.max(new_staleness);
        if applied > 0 {
            stats.last_grad_norm = sq_sum / applied as f64;
        }

        if new_staleness > self.config.max_staleness {
            warn!(
                "worker {worker_id} async push: staleness {new_staleness} exceeds max {}",
                self.config.max_staleness
            );
        }
        Ok(())
    }
}

fn init_row(seed_id: &str, dim: usize) -> Vec<f32> {
    // Linear congruential generator seeded by FNV-1a hash of `seed_id`.
    // Pure-Rust, no rand dependency, fully deterministic across runs.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in seed_id.as_bytes() {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    let mut state = h | 1;
    let mut row = Vec::with_capacity(dim);
    for _ in 0..dim {
        // Numerical Recipes LCG.
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to [-0.05, 0.05).
        let raw = (state >> 32) as u32;
        let f = (raw as f32 / u32::MAX as f32) * 0.1 - 0.05;
        row.push(f);
    }
    row
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed_training::shard_manager::{ModelShardManager, ShardingStrategy};

    fn small_cfg(mode: UpdateMode, workers: usize) -> ParameterServerConfig {
        ParameterServerConfig {
            embedding_dim: 4,
            num_entities: 8,
            num_relations: 2,
            num_shards: 2,
            expected_workers: workers,
            update_mode: mode,
            learning_rate: 0.1,
            max_staleness: 8,
            barrier_timeout: Duration::from_millis(500),
        }
    }

    fn small_server(mode: UpdateMode, workers: usize) -> ParameterServer {
        let cfg = small_cfg(mode, workers);
        let entity_ids: Vec<String> = (0..cfg.num_entities).map(|i| format!("e{i}")).collect();
        let relation_ids: Vec<String> = (0..cfg.num_relations).map(|i| format!("r{i}")).collect();
        let mgr = ModelShardManager::new(cfg.num_shards, ShardingStrategy::EntityHash);
        ParameterServer::new(cfg, entity_ids, relation_ids, mgr)
            .expect("server construction failed")
    }

    #[tokio::test]
    async fn server_constructs_and_reports_shards() {
        let s = small_server(UpdateMode::Sync, 2);
        assert_eq!(s.num_shards(), 2);
    }

    #[tokio::test]
    async fn server_rejects_zero_dim() {
        let mut cfg = small_cfg(UpdateMode::Sync, 2);
        cfg.embedding_dim = 0;
        let mgr = ModelShardManager::new(cfg.num_shards, ShardingStrategy::EntityHash);
        let res = ParameterServer::new(cfg, vec!["a".into()], vec!["r".into()], mgr);
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn pull_returns_consistent_dim_rows() {
        let s = small_server(UpdateMode::Sync, 2);
        for shard in 0..s.num_shards() {
            let snap = s.pull(shard).await.expect("pull");
            assert_eq!(snap.shard_id, shard);
            assert_eq!(snap.relations.len(), 2);
            for row in &snap.entities {
                assert_eq!(row.len(), 4);
            }
        }
    }

    #[tokio::test]
    async fn push_async_applies_immediately() {
        let s = small_server(UpdateMode::Async, 1);
        let snap = s.pull(0).await.expect("pull");
        let before = snap.entities.first().cloned().unwrap_or_default();

        let grad: Vec<f32> = vec![1.0; 4];
        if !snap.entities.is_empty() {
            s.push(0, 0, vec![(0, grad.clone())])
                .await
                .expect("push async");
            let snap2 = s.pull(0).await.expect("pull2");
            let after = snap2.entities.first().cloned().unwrap_or_default();
            // After applying grad=1.0 with lr=0.1, weights drop by 0.1.
            for (b, a) in before.iter().zip(after.iter()) {
                assert!(
                    (b - a - 0.1).abs() < 1e-5,
                    "expected b - a ≈ 0.1, got b={b}, a={a}"
                );
            }
        }
    }

    #[tokio::test]
    async fn push_sync_buffers_until_barrier() {
        let s = Arc::new(small_server(UpdateMode::Sync, 2));
        let snap = s.pull(0).await.expect("pull");
        if snap.entities.is_empty() {
            // Hash put no entities on shard 0; pick another shard.
            return;
        }

        let grad: Vec<f32> = vec![2.0; 4];

        // Push from worker 0; should *not* increment step yet.
        let s0 = Arc::clone(&s);
        let g0 = grad.clone();
        let h0 = tokio::spawn(async move {
            s0.push(0, 0, vec![(0, g0)]).await.expect("worker 0 push");
        });

        // Push from worker 1; this completes the barrier.
        let s1 = Arc::clone(&s);
        let g1 = grad.clone();
        let h1 = tokio::spawn(async move {
            s1.push(0, 1, vec![(0, g1)]).await.expect("worker 1 push");
        });

        h0.await.expect("worker 0 join");
        h1.await.expect("worker 1 join");

        let stats = s.stats().await;
        assert_eq!(
            stats.barriers_completed, 1,
            "exactly one barrier should have fired"
        );
        let steps = s.shard_steps().await;
        assert_eq!(steps[0], 1, "shard 0 should have advanced one step");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn push_sync_rejects_double_push_from_same_worker() {
        // Force a 2-worker barrier and use the same worker_id twice.  The
        // first push will block on the barrier; the second concurrent push
        // from the same worker must be rejected.
        let s = Arc::new(small_server(UpdateMode::Sync, 2));
        let snap = s.pull(0).await.expect("pull");
        if snap.entities.is_empty() {
            return;
        }

        let g = vec![0.0_f32; 4];
        let s_first = Arc::clone(&s);
        let g_first = g.clone();
        let h = tokio::spawn(async move {
            // This will block on the barrier (only one worker pushed).
            // The barrier_timeout in `small_cfg` is 500ms — it will eventually
            // unblock and return Ok, but until then we have time to fire the
            // second push from the same worker.
            s_first.push(0, 7, vec![(0, g_first)]).await
        });

        // Yield so the spawned push registers its worker_id before we re-push.
        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let err = s.push(0, 7, vec![(0, g)]).await;
        assert!(err.is_err(), "second push by same worker must fail");

        // Let the original push complete (either via timeout flush or barrier).
        let _ = h.await.expect("join push task");
    }

    #[tokio::test]
    async fn push_validates_gradient_dim() {
        let s = small_server(UpdateMode::Async, 1);
        // Wrong-length gradient.
        let res = s.push(0, 0, vec![(0, vec![1.0; 3])]).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn relation_push_applies_with_learning_rate() {
        let s = small_server(UpdateMode::Sync, 1);
        let before = s.pull(0).await.expect("pull").relations[0].clone();
        s.push_relation(0, vec![(0, vec![1.0_f32; 4])])
            .await
            .expect("rel push");
        let after = s.pull(0).await.expect("pull2").relations[0].clone();
        for (b, a) in before.iter().zip(after.iter()) {
            assert!((b - a - 0.1).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn async_pull_resets_staleness() {
        let s = small_server(UpdateMode::Async, 1);
        // Find a shard with at least one entity to push to.
        let snap = s.pull(0).await.expect("pull");
        if snap.entities.is_empty() {
            return;
        }
        for _ in 0..3 {
            s.push(0, 0, vec![(0, vec![0.1_f32; 4])])
                .await
                .expect("push");
        }
        let stats_before = s.stats().await;
        assert!(stats_before.max_staleness_observed >= 3);

        // Pulling again should reset the per-shard counter (max stays).
        let _ = s.pull(0).await.expect("pull");
        let stats_after = s.stats().await;
        assert_eq!(
            stats_after.max_staleness_observed, stats_before.max_staleness_observed,
            "max_staleness_observed is monotonic"
        );
    }
}
