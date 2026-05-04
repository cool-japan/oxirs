//! Distributed training worker that pulls latest params, computes local
//! gradients on a TransE-shaped objective, and pushes them back to a
//! [`super::parameter_server::ParameterServer`].
//!
//! The worker implements the classic parameter-server inner loop:
//!
//! ```text
//! for step in 0..max_steps {
//!     for shard in shards_this_worker_owns {
//!         snap = ps.pull(shard)
//!         (loss, grads) = local_step(snap, mini_batch)
//!         ps.push(shard, grads)
//!     }
//! }
//! ```
//!
//! The loss is a margin-ranking loss on TransE: `score(h,r,t) = ||h + r - t||`.
//! Concretely, for each positive triple `(h, r, t)` we sample a negative tail
//! `t'` from the same shard's entities and minimise
//! `max(0, margin + score(h,r,t) - score(h,r,t'))`.  This is a small but
//! genuinely non-trivial signal — sufficient to drive convergence on a toy
//! graph and to validate that the parameter-server plumbing actually moves
//! parameters in the right direction.
//!
//! Workers are intentionally **stateless** between iterations: every step pulls
//! fresh parameters from the server.  This is wasteful but matches the
//! prototype contract (and makes tests trivially reproducible).

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, trace};

use super::parameter_server::{ParameterServer, ShardSnapshot, UpdateMode};

/// `(row_index_in_shard, gradient_row)` — what the worker pushes per row.
type GradRow = (usize, Vec<f32>);

/// Output of [`Worker::local_step`]: mean loss, entity row gradients,
/// relation row gradients, and sample count.
type LocalStepOutput = (f64, Vec<GradRow>, Vec<GradRow>, usize);

/// One TransE-shaped sample: `(head, relation, tail)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleSample {
    /// Head entity ID.
    pub head: String,
    /// Relation IRI.
    pub relation: String,
    /// Tail entity ID.
    pub tail: String,
}

impl TripleSample {
    /// Convenience constructor.
    pub fn new(
        head: impl Into<String>,
        relation: impl Into<String>,
        tail: impl Into<String>,
    ) -> Self {
        Self {
            head: head.into(),
            relation: relation.into(),
            tail: tail.into(),
        }
    }
}

/// Per-iteration loss reported by [`Worker::run`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerLoss {
    /// Worker rank.
    pub worker_id: u32,
    /// Sequence of mean-batch losses, one entry per pulled shard.
    pub history: Vec<f64>,
    /// Sum of all losses across the run.
    pub total_loss: f64,
    /// Number of (h,r,t) triples that contributed to the loss.
    pub samples: usize,
}

impl WorkerLoss {
    /// Mean of the recorded losses.
    pub fn mean(&self) -> f64 {
        if self.history.is_empty() {
            0.0
        } else {
            self.total_loss / self.history.len() as f64
        }
    }
}

/// Worker configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Worker rank (must be unique within a parameter-server cohort).
    pub worker_id: u32,
    /// Maximum number of training iterations the worker will perform.
    pub max_steps: usize,
    /// TransE margin (γ).
    pub margin: f32,
    /// L2 regularisation coefficient.  `0.0` disables.
    pub l2_reg: f32,
    /// Random seed for deterministic negative sampling.
    pub seed: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            worker_id: 0,
            max_steps: 50,
            margin: 1.0,
            l2_reg: 0.0,
            seed: 1,
        }
    }
}

/// In-process distributed-training worker.
pub struct Worker {
    config: WorkerConfig,
    server: Arc<ParameterServer>,
    /// Triples this worker is responsible for.  Each triple is routed to the
    /// shard owning its head entity at training time.
    samples: Vec<TripleSample>,
    /// LCG state for deterministic negative sampling.
    rng_state: u64,
}

impl Worker {
    /// Build a new worker.
    pub fn new(
        config: WorkerConfig,
        server: Arc<ParameterServer>,
        samples: Vec<TripleSample>,
    ) -> Self {
        let rng_state = config.seed | 1;
        Self {
            config,
            server,
            samples,
            rng_state,
        }
    }

    /// Worker configuration view.
    pub fn config(&self) -> &WorkerConfig {
        &self.config
    }

    /// Run the worker's training loop.
    ///
    /// Returns the per-step loss history.  This is `async` and re-entrant:
    /// callers typically `tokio::spawn` one task per worker and `join_all`
    /// them.
    pub async fn run(mut self) -> Result<WorkerLoss> {
        let mut loss = WorkerLoss {
            worker_id: self.config.worker_id,
            ..Default::default()
        };
        let started = Instant::now();

        // Group samples by shard ownership of the head entity, refreshed each step.
        for step in 0..self.config.max_steps {
            // Build per-shard groups using the current shard manager mapping.
            // We materialise indices (not references) so the immutable borrow
            // of `self.samples` is dropped before the per-shard loop, which
            // re-borrows `self` mutably to mutate the RNG.
            let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
            for (i, s) in self.samples.iter().enumerate() {
                let shard = self.server.shard_manager().shard_for(&s.head);
                groups.entry(shard).or_default().push(i);
            }

            for (shard_id, indices) in groups {
                let snap = self.server.pull(shard_id).await?;
                // Clone the small per-shard sample slice into an owned vec so
                // we can mutably borrow `self` (for RNG state) inside
                // `local_step` without conflicting with the immutable borrow
                // of `self.samples`.
                let shard_samples: Vec<TripleSample> =
                    indices.iter().map(|&i| self.samples[i].clone()).collect();
                let sample_refs: Vec<&TripleSample> = shard_samples.iter().collect();
                let (mean_loss, grads, rel_grads, n) = self.local_step(&snap, &sample_refs)?;
                if !rel_grads.is_empty() {
                    self.server
                        .push_relation(self.config.worker_id, rel_grads)
                        .await?;
                }
                if !grads.is_empty() {
                    self.server
                        .push(shard_id, self.config.worker_id, grads)
                        .await?;
                }
                loss.history.push(mean_loss);
                loss.total_loss += mean_loss;
                loss.samples += n;
                trace!(
                    worker = self.config.worker_id,
                    step,
                    shard = shard_id,
                    samples = n,
                    loss = mean_loss,
                    "worker step done"
                );
            }
        }

        debug!(
            worker = self.config.worker_id,
            elapsed_ms = started.elapsed().as_millis() as u64,
            mean_loss = loss.mean(),
            "worker finished"
        );
        Ok(loss)
    }

    /// Compute the local gradient batch for one shard's worth of samples.
    ///
    /// Returns `(mean_loss, entity_gradient_rows, relation_gradient_rows,
    /// sample_count)`.  Gradient rows are `(row_index, gradient_vector)`
    /// pairs ready to be pushed back to the server.  The entity rows are
    /// indexed inside the shard; the relation rows use the global relation
    /// table index because relations are fully replicated.
    ///
    /// Implementation note: we use a *closed-form* derivative of the margin
    /// loss with respect to head/tail rows.  Both entity- and relation-row
    /// gradients are returned to the caller, which is responsible for
    /// applying them via [`ParameterServer::push`] /
    /// [`ParameterServer::push_relation`] in the desired order.
    fn local_step(
        &mut self,
        snap: &ShardSnapshot,
        samples: &[&TripleSample],
    ) -> Result<LocalStepOutput> {
        if snap.entities.is_empty() || samples.is_empty() {
            return Ok((0.0, Vec::new(), Vec::new(), 0));
        }

        let dim = snap.entities[0].len();
        let entity_index: HashMap<&str, usize> = snap
            .entity_ids
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        let relation_index: HashMap<&str, usize> = snap
            .relation_ids
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();

        // Accumulate gradients per row in the shard.
        let mut grad_acc: HashMap<usize, Vec<f32>> = HashMap::new();
        // Accumulate relation gradients (replicated table → idx is global).
        let mut rel_grad: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut total_loss = 0.0_f64;
        let mut counted = 0usize;

        for s in samples {
            // We can only train on triples whose **head** lives on this shard.
            let h_idx = match entity_index.get(s.head.as_str()) {
                Some(&i) => i,
                None => continue,
            };
            let r_idx = match relation_index.get(s.relation.as_str()) {
                Some(&i) => i,
                None => continue,
            };

            // Tail may live elsewhere; we still get a useful gradient on the
            // head row by treating the tail as constant.  If the tail does
            // happen to live on this shard we update both.
            let t_idx_local = entity_index.get(s.tail.as_str()).copied();
            let head = &snap.entities[h_idx];
            let rel = &snap.relations[r_idx];

            // For tail, prefer the shard's own copy if available, otherwise
            // we fabricate a vector by looking up the relation row → that's
            // a pragmatic toy choice; the prototype is intentionally not a
            // full distributed embedding lookup.
            let tail_vec: Vec<f32> = match t_idx_local {
                Some(i) => snap.entities[i].clone(),
                None => snap.relations[r_idx].clone(),
            };

            // Sample a negative tail t' from this shard's entities.
            let neg_idx = self.next_index(snap.entities.len());
            let neg = &snap.entities[neg_idx];

            // Score(h, r, t) = ||h + r - t||₂  (using f32 throughout).
            let pos_diff: Vec<f32> = head
                .iter()
                .zip(rel.iter())
                .zip(tail_vec.iter())
                .map(|((h, r), t)| h + r - t)
                .collect();
            let neg_diff: Vec<f32> = head
                .iter()
                .zip(rel.iter())
                .zip(neg.iter())
                .map(|((h, r), n)| h + r - n)
                .collect();

            let pos_score = l2_norm(&pos_diff);
            let neg_score = l2_norm(&neg_diff);
            let margin = self.config.margin;
            let raw_loss = (margin + pos_score - neg_score).max(0.0);
            total_loss += raw_loss as f64;
            counted += 1;

            // Subgradient when raw_loss > 0:
            //   ∂L/∂h = (pos_diff/||pos_diff||) - (neg_diff/||neg_diff||)
            //   ∂L/∂r = same
            //   ∂L/∂t = -pos_diff/||pos_diff||
            //   ∂L/∂t' = neg_diff/||neg_diff||
            if raw_loss > 0.0 {
                let pos_norm = pos_score.max(1e-6);
                let neg_norm = neg_score.max(1e-6);

                let grad_h: Vec<f32> = pos_diff
                    .iter()
                    .zip(neg_diff.iter())
                    .map(|(p, n)| p / pos_norm - n / neg_norm)
                    .collect();
                let grad_r = grad_h.clone();
                let grad_t: Vec<f32> = pos_diff.iter().map(|p| -p / pos_norm).collect();
                let grad_neg: Vec<f32> = neg_diff.iter().map(|n| n / neg_norm).collect();

                accumulate_grad(&mut grad_acc, h_idx, &grad_h, dim);
                if let Some(ti) = t_idx_local {
                    accumulate_grad(&mut grad_acc, ti, &grad_t, dim);
                }
                accumulate_grad(&mut grad_acc, neg_idx, &grad_neg, dim);
                accumulate_grad(&mut rel_grad, r_idx, &grad_r, dim);
            }

            // Optional L2 regularisation on the head row.
            if self.config.l2_reg > 0.0 {
                let entry = grad_acc.entry(h_idx).or_insert_with(|| vec![0.0; dim]);
                for (e, h) in entry.iter_mut().zip(head.iter()) {
                    *e += self.config.l2_reg * *h;
                }
            }
        }

        let mean_loss = if counted == 0 {
            0.0
        } else {
            total_loss / counted as f64
        };
        let grads: Vec<(usize, Vec<f32>)> = grad_acc.into_iter().collect();
        let rel_grads: Vec<(usize, Vec<f32>)> = rel_grad.into_iter().collect();
        Ok((mean_loss, grads, rel_grads, counted))
    }

    fn next_index(&mut self, n: usize) -> usize {
        // LCG step (Numerical Recipes).
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.rng_state >> 32) as usize) % n.max(1)
    }
}

/// Run multiple workers concurrently and gather their losses.
///
/// `workers` is consumed; each worker is spawned on its own tokio task.
pub async fn run_workers(workers: Vec<Worker>) -> Result<Vec<WorkerLoss>> {
    let mut handles = Vec::with_capacity(workers.len());
    for w in workers {
        handles.push(tokio::spawn(async move { w.run().await }));
    }
    let mut out = Vec::with_capacity(handles.len());
    for h in handles {
        match h.await {
            Ok(Ok(loss)) => out.push(loss),
            Ok(Err(e)) => return Err(e),
            Err(join_err) => return Err(anyhow::anyhow!("worker join failed: {join_err}")),
        }
    }
    Ok(out)
}

/// Pretty-print server state for debugging (used by examples).
pub async fn describe_server(server: &ParameterServer) -> String {
    let stats = server.stats().await;
    let steps = server.shard_steps().await;
    let mode = match server.config().update_mode {
        UpdateMode::Sync => "sync",
        UpdateMode::Async => "async",
    };
    format!(
        "ParameterServer[mode={mode}, shards={}, total_pulls={}, total_pushes={}, barriers={}, steps={steps:?}]",
        server.num_shards(),
        stats.total_pulls,
        stats.total_pushes,
        stats.barriers_completed,
    )
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn accumulate_grad(target: &mut HashMap<usize, Vec<f32>>, idx: usize, grad: &[f32], dim: usize) {
    let entry = target.entry(idx).or_insert_with(|| vec![0.0; dim]);
    for (e, g) in entry.iter_mut().zip(grad.iter()) {
        *e += *g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed_training::parameter_server::{
        ParameterServer, ParameterServerConfig, UpdateMode,
    };
    use crate::distributed_training::shard_manager::{ModelShardManager, ShardingStrategy};

    fn build_server(workers: usize, mode: UpdateMode) -> Arc<ParameterServer> {
        let cfg = ParameterServerConfig {
            embedding_dim: 8,
            num_entities: 8,
            num_relations: 2,
            num_shards: 2,
            expected_workers: workers,
            update_mode: mode,
            learning_rate: 0.05,
            max_staleness: 16,
            barrier_timeout: std::time::Duration::from_millis(500),
        };
        let entity_ids: Vec<String> = (0..cfg.num_entities).map(|i| format!("e{i}")).collect();
        let relation_ids: Vec<String> = (0..cfg.num_relations).map(|i| format!("r{i}")).collect();
        let mgr = ModelShardManager::new(cfg.num_shards, ShardingStrategy::EntityHash);
        Arc::new(
            ParameterServer::new(cfg, entity_ids, relation_ids, mgr)
                .expect("server construction failed"),
        )
    }

    fn small_triples() -> Vec<TripleSample> {
        vec![
            TripleSample::new("e0", "r0", "e1"),
            TripleSample::new("e2", "r0", "e3"),
            TripleSample::new("e4", "r1", "e5"),
            TripleSample::new("e6", "r1", "e7"),
        ]
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn worker_runs_async_and_records_loss() {
        let server = build_server(1, UpdateMode::Async);
        let cfg = WorkerConfig {
            worker_id: 0,
            max_steps: 5,
            margin: 1.0,
            l2_reg: 0.0,
            seed: 7,
        };
        let w = Worker::new(cfg, Arc::clone(&server), small_triples());
        let loss = w.run().await.expect("worker run failed");
        assert_eq!(loss.worker_id, 0);
        assert!(
            !loss.history.is_empty(),
            "worker should record at least one loss entry"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn four_workers_async_complete() {
        let server = build_server(1, UpdateMode::Async);
        let mut ws = Vec::new();
        for i in 0..4 {
            ws.push(Worker::new(
                WorkerConfig {
                    worker_id: i,
                    max_steps: 3,
                    margin: 1.0,
                    l2_reg: 1e-4,
                    seed: 1 + i as u64,
                },
                Arc::clone(&server),
                small_triples(),
            ));
        }
        let losses = run_workers(ws).await.expect("workers failed");
        assert_eq!(losses.len(), 4);
        for l in &losses {
            assert!(l.history.iter().all(|x| x.is_finite()));
        }
    }

    #[tokio::test]
    async fn describe_server_renders() {
        let s = build_server(1, UpdateMode::Async);
        let desc = describe_server(&s).await;
        assert!(desc.contains("mode=async"));
    }
}
