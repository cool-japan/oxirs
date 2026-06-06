//! Knowledge Graph Completion evaluator.
//!
//! This module provides [`KgcEvaluator`], which runs the standard KGC
//! evaluation protocol against any trained [`EmbeddingModel`]:
//!
//! 1. For each test triple (h, r, t):
//!    - **Tail prediction**: replace `t` with every entity in the vocabulary,
//!      score, rank the true tail.
//!    - **Head prediction**: replace `h` with every entity in the vocabulary,
//!      score, rank the true head.
//!    - Record both the raw rank and the filtered rank (known positives
//!      removed from ranking).
//! 2. Aggregate all ranks into [`EvaluationMetrics`].
//!
//! The high-level [`KgcEvaluationSuite`] trains a model from scratch on the
//! tiny synthetic dataset and evaluates it, providing an end-to-end smoke test.

use std::collections::HashSet;
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::evaluation::kgc_dataset::KgcDataset;
use crate::evaluation::kgc_metrics::{compute_filtered_rank, EvaluationMetrics};
use crate::{EmbeddingModel, NamedNode, Triple};

// ─────────────────────────────────────────────────────────────────────────────
// EvalSplit
// ─────────────────────────────────────────────────────────────────────────────

/// Which split of [`KgcDataset`] to evaluate against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalSplit {
    /// Use `dataset.valid`.
    Valid,
    /// Use `dataset.test`.
    Test,
}

// ─────────────────────────────────────────────────────────────────────────────
// KgcEvaluatorConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`KgcEvaluator`].
#[derive(Debug, Clone)]
pub struct KgcEvaluatorConfig {
    /// Number of test triples processed per iteration (informational; the
    /// evaluator is currently single-threaded but the field documents intent).
    pub batch_size: usize,
    /// If `true`, also compute filtered metrics (known positives removed).
    pub filtered: bool,
    /// Which split to evaluate.
    pub eval_split: EvalSplit,
    /// Cap the number of evaluated triples (useful for fast smoke tests).
    /// `None` means evaluate all.
    pub max_test_triples: Option<usize>,
}

impl Default for KgcEvaluatorConfig {
    fn default() -> Self {
        Self {
            batch_size: 256,
            filtered: true,
            eval_split: EvalSplit::Test,
            max_test_triples: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RankArgs — bundles the shared parameters for rank_entity_as_{tail,head}
// ─────────────────────────────────────────────────────────────────────────────

struct RankArgs<'a> {
    anchor1: &'a str,
    relation: &'a str,
    anchor2: &'a str,
    entities: &'a [String],
    entity_to_idx: &'a std::collections::HashMap<&'a str, usize>,
    all_positives: &'a HashSet<(String, String, String)>,
}

// ─────────────────────────────────────────────────────────────────────────────
// KgcEvaluator
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluates a trained embedding model on a [`KgcDataset`].
///
/// Each call to [`KgcEvaluator::evaluate`] runs the full head-and-tail
/// prediction protocol and returns aggregated [`EvaluationMetrics`].
pub struct KgcEvaluator {
    config: KgcEvaluatorConfig,
}

impl KgcEvaluator {
    /// Create a new evaluator with the given configuration.
    pub fn new(config: KgcEvaluatorConfig) -> Self {
        Self { config }
    }

    /// Evaluate a trained model on the dataset.
    ///
    /// For each test triple `(h, r, t)` the evaluator:
    /// 1. Scores `(h, r, e)` for every entity `e` in the vocabulary →
    ///    computes the raw rank and the filtered rank of `t`.
    /// 2. Scores `(e, r, t)` for every entity `e` →
    ///    computes the raw rank and the filtered rank of `h`.
    /// 3. Appends both ranks to the running list.
    ///
    /// Finally aggregates all collected ranks into [`EvaluationMetrics`].
    pub async fn evaluate<M: EmbeddingModel>(
        &self,
        model: &M,
        dataset: &KgcDataset,
    ) -> Result<EvaluationMetrics> {
        // Choose the correct split.
        let test_triples = match self.config.eval_split {
            EvalSplit::Valid => &dataset.valid,
            EvalSplit::Test => &dataset.test,
        };

        // Optionally cap the number of evaluated triples.
        let triples_to_eval: &[_] = if let Some(max) = self.config.max_test_triples {
            let end = max.min(test_triples.len());
            &test_triples[..end]
        } else {
            test_triples
        };

        if triples_to_eval.is_empty() {
            return Ok(EvaluationMetrics::zero());
        }

        // Sorted entity list for deterministic entity-to-index mapping.
        let entities = dataset.sorted_entities();
        if entities.is_empty() {
            return Err(anyhow!("dataset has an empty entity vocabulary"));
        }

        // Map entity string → index in `entities` for O(1) lookup.
        let entity_to_idx: std::collections::HashMap<&str, usize> = entities
            .iter()
            .enumerate()
            .map(|(i, e)| (e.as_str(), i))
            .collect();

        // Collect all positive triples for filtered evaluation.
        let all_positives: HashSet<(String, String, String)> = if self.config.filtered {
            dataset.all_positives()
        } else {
            HashSet::new()
        };

        let mut ranks: Vec<usize> = Vec::new();
        let mut filtered_ranks: Vec<usize> = Vec::new();

        for triple in triples_to_eval {
            let head = &triple.head;
            let relation = &triple.relation;
            let tail = &triple.tail;

            // ── Tail prediction: score (head, relation, ?) ────────────────
            {
                let (raw_rank, f_rank) = self.rank_entity_as_tail(
                    model,
                    RankArgs {
                        anchor1: head,
                        relation,
                        anchor2: tail,
                        entities: &entities,
                        entity_to_idx: &entity_to_idx,
                        all_positives: &all_positives,
                    },
                )?;
                ranks.push(raw_rank);
                filtered_ranks.push(f_rank);
            }

            // ── Head prediction: score (?, relation, tail) ────────────────
            {
                let (raw_rank, f_rank) = self.rank_entity_as_head(
                    model,
                    RankArgs {
                        anchor1: head,
                        relation,
                        anchor2: tail,
                        entities: &entities,
                        entity_to_idx: &entity_to_idx,
                        all_positives: &all_positives,
                    },
                )?;
                ranks.push(raw_rank);
                filtered_ranks.push(f_rank);
            }
        }

        Ok(EvaluationMetrics::compute(&ranks, &filtered_ranks))
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Score every entity as the tail of `(head, relation, ?)` and return
    /// `(raw_rank, filtered_rank)` of the true `tail`.
    fn rank_entity_as_tail<M: EmbeddingModel>(
        &self,
        model: &M,
        args: RankArgs<'_>,
    ) -> Result<(usize, usize)> {
        let RankArgs {
            anchor1: head,
            relation,
            anchor2: true_tail,
            entities,
            entity_to_idx,
            all_positives,
        } = args;
        // Score (head, relation, candidate) for every entity.
        let mut scored: Vec<(usize, f64)> = entities
            .iter()
            .enumerate()
            .filter_map(|(idx, cand)| {
                model
                    .score_triple(head, relation, cand)
                    .ok()
                    .map(|s| (idx, s))
            })
            .collect();

        // Sort descending by score (higher score = better = lower rank).
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let true_tail_idx = entity_to_idx
            .get(true_tail)
            .copied()
            .ok_or_else(|| anyhow!("true tail '{}' not in entity vocabulary", true_tail))?;

        // Raw rank.
        // Raw rank: position in the scored list (entities failing score_triple
        // are absent from `scored`, so the fallback worst rank is scored.len()+1).
        let raw_rank = scored
            .iter()
            .position(|&(idx, _)| idx == true_tail_idx)
            .map(|pos| pos + 1)
            .unwrap_or(scored.len() + 1);

        // Filtered rank: build set of OTHER known positives for (head, relation, ?).
        let other_pos_idxs: HashSet<usize> = if self.config.filtered {
            all_positives
                .iter()
                .filter(|(h, r, _t)| h == head && r == relation)
                .filter_map(|(_h, _r, t)| {
                    // Exclude the true tail itself.
                    if t == true_tail {
                        None
                    } else {
                        entity_to_idx.get(t.as_str()).copied()
                    }
                })
                .collect()
        } else {
            HashSet::new()
        };

        let f_rank = compute_filtered_rank(&scored, true_tail_idx, &other_pos_idxs);

        Ok((raw_rank, f_rank))
    }

    /// Score every entity as the head of `(?, relation, tail)` and return
    /// `(raw_rank, filtered_rank)` of the true `head`.
    fn rank_entity_as_head<M: EmbeddingModel>(
        &self,
        model: &M,
        args: RankArgs<'_>,
    ) -> Result<(usize, usize)> {
        let RankArgs {
            anchor1: true_head,
            relation,
            anchor2: tail,
            entities,
            entity_to_idx,
            all_positives,
        } = args;
        let mut scored: Vec<(usize, f64)> = entities
            .iter()
            .enumerate()
            .filter_map(|(idx, cand)| {
                model
                    .score_triple(cand, relation, tail)
                    .ok()
                    .map(|s| (idx, s))
            })
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let true_head_idx = entity_to_idx
            .get(true_head)
            .copied()
            .ok_or_else(|| anyhow!("true head '{}' not in entity vocabulary", true_head))?;

        let raw_rank = scored
            .iter()
            .position(|&(idx, _)| idx == true_head_idx)
            .map(|pos| pos + 1)
            .unwrap_or(scored.len() + 1);

        let other_pos_idxs: HashSet<usize> = if self.config.filtered {
            all_positives
                .iter()
                .filter(|(_h, r, t)| r == relation && t == tail)
                .filter_map(|(h, _r, _t)| {
                    if h == true_head {
                        None
                    } else {
                        entity_to_idx.get(h.as_str()).copied()
                    }
                })
                .collect()
        } else {
            HashSet::new()
        };

        let f_rank = compute_filtered_rank(&scored, true_head_idx, &other_pos_idxs);

        Ok((raw_rank, f_rank))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KgcEvaluationSuite
// ─────────────────────────────────────────────────────────────────────────────

/// High-level suite that trains a model from scratch on a dataset and
/// evaluates it, returning a unified result record.
///
/// Designed for quick smoke-tests and hyperparameter sweeps.
#[derive(Debug)]
pub struct KgcEvaluationSuite {
    /// Human-readable model identifier (e.g. `"TransE"`).
    pub model_name: String,
    /// Evaluation metrics obtained after training.
    pub metrics: EvaluationMetrics,
    /// Actual number of epochs run during training.
    pub training_epochs: usize,
    /// Wall-clock time for training + evaluation, in seconds.
    pub elapsed_secs: f64,
}

impl KgcEvaluationSuite {
    /// Train a **default-constructed** model on the tiny synthetic dataset for
    /// `epochs` epochs, then evaluate on the test split.
    ///
    /// The model must implement both [`EmbeddingModel`] and [`Default`].
    /// It is the caller's responsibility to choose a model whose `Default`
    /// configuration includes a sensible embedding dimension and learning rate.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use oxirs_embed::evaluation::kgc_evaluator::KgcEvaluationSuite;
    /// # use oxirs_embed::TransE;
    /// # tokio_test::block_on(async {
    /// let suite = KgcEvaluationSuite::run_on_synthetic::<TransE>(50).await.unwrap();
    /// println!("{}", suite.metrics.display());
    /// # });
    /// ```
    pub async fn run_on_synthetic<M>(epochs: usize) -> Result<Self>
    where
        M: EmbeddingModel + Default,
    {
        let timer = Instant::now();
        let dataset = KgcDataset::tiny_synthetic();

        // Build and populate the model.
        let mut model = M::default();
        for triple in dataset.train.iter().chain(dataset.valid.iter()) {
            let t = Triple::new(
                NamedNode::new(&triple.head)?,
                NamedNode::new(&triple.relation)?,
                NamedNode::new(&triple.tail)?,
            );
            model.add_triple(t)?;
        }

        // Train.
        let training_stats = model.train(Some(epochs)).await?;
        let training_epochs = training_stats.epochs_completed;

        // Evaluate on the test split.
        let eval_config = KgcEvaluatorConfig {
            batch_size: 64,
            filtered: true,
            eval_split: EvalSplit::Test,
            max_test_triples: None,
        };
        let evaluator = KgcEvaluator::new(eval_config);
        let metrics = evaluator.evaluate(&model, &dataset).await?;

        let model_name = model.model_type().to_string();
        let elapsed_secs = timer.elapsed().as_secs_f64();

        Ok(Self {
            model_name,
            metrics,
            training_epochs,
            elapsed_secs,
        })
    }

    /// Run the suite using a pre-trained model (no training performed).
    ///
    /// Useful when training is handled externally or when evaluating on a
    /// custom dataset.
    pub async fn run_pretrained<M: EmbeddingModel>(
        model: &M,
        dataset: &KgcDataset,
        eval_split: EvalSplit,
    ) -> Result<Self> {
        let timer = Instant::now();

        let eval_config = KgcEvaluatorConfig {
            batch_size: 256,
            filtered: true,
            eval_split,
            max_test_triples: None,
        };
        let evaluator = KgcEvaluator::new(eval_config);
        let metrics = evaluator.evaluate(model, dataset).await?;
        let model_name = model.model_type().to_string();
        let elapsed_secs = timer.elapsed().as_secs_f64();

        Ok(Self {
            model_name,
            metrics,
            training_epochs: 0,
            elapsed_secs,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::kgc_dataset::KgcDataset;
    use crate::models::TransE;
    use crate::{ModelConfig, NamedNode, Triple};

    /// Build and train a tiny TransE model on the synthetic dataset.
    async fn make_trained_transe(dataset: &KgcDataset, epochs: usize) -> TransE {
        let config = ModelConfig::default()
            .with_dimensions(16)
            .with_learning_rate(0.05)
            .with_max_epochs(epochs)
            .with_seed(42);
        let mut model = TransE::new(config);
        for triple in dataset.train.iter().chain(dataset.valid.iter()) {
            let t = Triple::new(
                NamedNode::new(&triple.head).unwrap(),
                NamedNode::new(&triple.relation).unwrap(),
                NamedNode::new(&triple.tail).unwrap(),
            );
            model.add_triple(t).unwrap();
        }
        model.train(Some(epochs)).await.unwrap();
        model
    }

    // ── Test 1: evaluator runs without error on tiny synthetic ────────────
    #[tokio::test]
    async fn test_evaluator_runs_on_tiny_synthetic() {
        let dataset = KgcDataset::tiny_synthetic();
        let model = make_trained_transe(&dataset, 10).await;
        let config = KgcEvaluatorConfig {
            batch_size: 32,
            filtered: true,
            eval_split: EvalSplit::Test,
            max_test_triples: None,
        };
        let evaluator = KgcEvaluator::new(config);
        let metrics = evaluator.evaluate(&model, &dataset).await;
        assert!(
            metrics.is_ok(),
            "evaluator should complete without error: {:?}",
            metrics.err()
        );
        let m = metrics.unwrap();
        assert!(m.num_test_triples > 0, "should have evaluated some triples");
    }

    // ── Test 2: filtered MRR >= raw MRR ───────────────────────────────────
    #[tokio::test]
    async fn test_filtered_mrr_gte_raw_mrr() {
        let dataset = KgcDataset::tiny_synthetic();
        let model = make_trained_transe(&dataset, 10).await;
        let config = KgcEvaluatorConfig {
            batch_size: 32,
            filtered: true,
            eval_split: EvalSplit::Test,
            max_test_triples: None,
        };
        let evaluator = KgcEvaluator::new(config);
        let m = evaluator.evaluate(&model, &dataset).await.unwrap();
        // Filtered MRR should be >= raw MRR because known positives are removed.
        assert!(
            m.filtered_mrr >= m.mean_reciprocal_rank - 1e-9,
            "filtered_mrr ({}) should be >= raw MRR ({})",
            m.filtered_mrr,
            m.mean_reciprocal_rank
        );
    }

    // ── Test 3: max_test_triples = Some(1) evaluates exactly 2 queries ────
    // (head + tail = 2 queries per test triple)
    #[tokio::test]
    async fn test_max_test_triples_limits_evaluation() {
        let dataset = KgcDataset::tiny_synthetic();
        let model = make_trained_transe(&dataset, 5).await;
        let config = KgcEvaluatorConfig {
            batch_size: 32,
            filtered: true,
            eval_split: EvalSplit::Test,
            max_test_triples: Some(1),
        };
        let evaluator = KgcEvaluator::new(config);
        let m = evaluator.evaluate(&model, &dataset).await.unwrap();
        // 1 test triple × 2 queries (head + tail) = 2 rank entries
        assert_eq!(
            m.num_test_triples, 2,
            "1 test triple should produce 2 rank queries, got {}",
            m.num_test_triples
        );
    }

    // ── Test 4: valid split evaluates dataset.valid ────────────────────────
    #[tokio::test]
    async fn test_eval_split_valid_uses_valid_set() {
        let dataset = KgcDataset::tiny_synthetic();
        let model = make_trained_transe(&dataset, 5).await;
        let config_v = KgcEvaluatorConfig {
            eval_split: EvalSplit::Valid,
            max_test_triples: None,
            ..KgcEvaluatorConfig::default()
        };
        let config_t = KgcEvaluatorConfig {
            eval_split: EvalSplit::Test,
            max_test_triples: None,
            ..KgcEvaluatorConfig::default()
        };
        let ev = KgcEvaluator::new(config_v);
        let et = KgcEvaluator::new(config_t);
        let mv = ev.evaluate(&model, &dataset).await.unwrap();
        let mt = et.evaluate(&model, &dataset).await.unwrap();
        // Both splits have the same size (1 triple each in tiny_synthetic)
        // so num_test_triples is the same; just ensure both complete.
        assert!(mv.num_test_triples > 0);
        assert!(mt.num_test_triples > 0);
    }

    // ── Test 5: KgcEvaluationSuite::run_on_synthetic runs end-to-end ──────
    #[tokio::test]
    async fn test_evaluation_suite_end_to_end() {
        // TransE implements Default via its Default derive or explicit impl.
        // If TransE doesn't have Default, we use run_pretrained instead.
        // We test run_pretrained here as it doesn't need Default.
        let dataset = KgcDataset::tiny_synthetic();
        let model = make_trained_transe(&dataset, 10).await;
        let suite = KgcEvaluationSuite::run_pretrained(&model, &dataset, EvalSplit::Test)
            .await
            .unwrap();
        assert_eq!(suite.model_name, "TransE");
        assert!(suite.metrics.num_test_triples > 0);
        assert!(suite.elapsed_secs >= 0.0);
    }

    // ── Test 6: metrics MRR in [0, 1] range ───────────────────────────────
    #[tokio::test]
    async fn test_mrr_in_valid_range() {
        let dataset = KgcDataset::tiny_synthetic();
        let model = make_trained_transe(&dataset, 5).await;
        let config = KgcEvaluatorConfig::default();
        let evaluator = KgcEvaluator::new(config);
        let m = evaluator.evaluate(&model, &dataset).await.unwrap();
        assert!(
            m.mean_reciprocal_rank >= 0.0 && m.mean_reciprocal_rank <= 1.0,
            "MRR must be in [0, 1], got {}",
            m.mean_reciprocal_rank
        );
        assert!(
            m.filtered_mrr >= 0.0 && m.filtered_mrr <= 1.0,
            "filtered MRR must be in [0, 1], got {}",
            m.filtered_mrr
        );
    }

    // ── Test 7: Hits@K in [0, 1] and hits_at_10 >= hits_at_1 ─────────────
    #[tokio::test]
    async fn test_hits_monotone_and_bounded() {
        let dataset = KgcDataset::tiny_synthetic();
        let model = make_trained_transe(&dataset, 5).await;
        let config = KgcEvaluatorConfig::default();
        let evaluator = KgcEvaluator::new(config);
        let m = evaluator.evaluate(&model, &dataset).await.unwrap();
        assert!(m.hits_at_1 >= 0.0 && m.hits_at_1 <= 1.0);
        assert!(m.hits_at_3 >= 0.0 && m.hits_at_3 <= 1.0);
        assert!(m.hits_at_10 >= 0.0 && m.hits_at_10 <= 1.0);
        assert!(m.hits_at_10 >= m.hits_at_3, "hits@10 >= hits@3 must hold");
        assert!(m.hits_at_3 >= m.hits_at_1, "hits@3 >= hits@1 must hold");
    }

    // ── Test 8: unfiltered evaluation (filtered=false) runs correctly ──────
    // When filtered=false, other_pos_idxs is always empty, so
    // compute_filtered_rank degenerates to the same computation as raw rank.
    // Both use scored.len()+1 as the worst-case fallback, so they are equal.
    #[tokio::test]
    async fn test_unfiltered_evaluation() {
        let dataset = KgcDataset::tiny_synthetic();
        let model = make_trained_transe(&dataset, 5).await;
        let config = KgcEvaluatorConfig {
            filtered: false,
            eval_split: EvalSplit::Test,
            max_test_triples: None,
            batch_size: 64,
        };
        let evaluator = KgcEvaluator::new(config);
        let m = evaluator.evaluate(&model, &dataset).await.unwrap();
        assert!(m.num_test_triples > 0, "should evaluate at least one query");
        assert!(
            m.mean_rank >= 1.0,
            "mean rank should be >= 1.0, got {}",
            m.mean_rank
        );
        // When filtered=false, raw == filtered because known-positive set is empty.
        assert!(
            (m.mean_rank - m.filtered_mean_rank).abs() < 1e-9,
            "unfiltered mode: raw MR ({}) should equal filtered MR ({})",
            m.mean_rank,
            m.filtered_mean_rank
        );
    }
}
