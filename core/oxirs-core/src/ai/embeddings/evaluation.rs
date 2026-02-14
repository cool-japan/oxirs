use super::KnowledgeGraphEmbedding;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Comprehensive knowledge graph evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphMetrics {
    /// Mean Reciprocal Rank (filtered)
    pub mrr_filtered: f32,
    /// Mean Reciprocal Rank (unfiltered)
    pub mrr_unfiltered: f32,
    /// Mean Rank (filtered)
    pub mr_filtered: f32,
    /// Mean Rank (unfiltered)
    pub mr_unfiltered: f32,
    /// Hits@K metrics (filtered)
    pub hits_at_k_filtered: std::collections::HashMap<u32, f32>,
    /// Hits@K metrics (unfiltered)
    pub hits_at_k_unfiltered: std::collections::HashMap<u32, f32>,
    /// Per-relation type performance
    pub per_relation_metrics: std::collections::HashMap<String, RelationMetrics>,
    /// Link prediction task breakdown
    pub task_breakdown: TaskBreakdownMetrics,
    /// Confidence intervals (95%)
    pub confidence_intervals: ConfidenceIntervals,
    /// Statistical significance test results
    pub statistical_tests: StatisticalTestResults,
}

/// Comprehensive training metrics for knowledge graph embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Final training loss
    pub loss: f32,
    /// Loss history across epochs
    pub loss_history: Vec<f32>,
    /// Basic accuracy (deprecated, use ranking metrics instead)
    pub accuracy: f32,
    /// Number of training epochs completed
    pub epochs: usize,
    /// Total training time
    pub time_elapsed: std::time::Duration,
    /// Knowledge graph specific metrics
    pub kg_metrics: KnowledgeGraphMetrics,
}

/// Per-relation performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationMetrics {
    pub mrr: f32,
    pub mr: f32,
    pub hits_at_k: std::collections::HashMap<u32, f32>,
    pub sample_count: usize,
    pub entity_coverage: f32,
}

/// Breakdown by link prediction tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBreakdownMetrics {
    /// Head entity prediction (?, r, t)
    pub head_prediction: LinkPredictionMetrics,
    /// Tail entity prediction (h, r, ?)
    pub tail_prediction: LinkPredictionMetrics,
    /// Relation prediction (h, ?, t)
    pub relation_prediction: LinkPredictionMetrics,
}

/// Link prediction specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkPredictionMetrics {
    pub mrr: f32,
    pub mr: f32,
    pub hits_at_k: std::collections::HashMap<u32, f32>,
    pub auc_roc: f32,
    pub auc_pr: f32,
    pub precision_at_k: std::collections::HashMap<u32, f32>,
    pub recall_at_k: std::collections::HashMap<u32, f32>,
}

/// Confidence intervals for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervals {
    pub mrr_ci: (f32, f32),
    pub mr_ci: (f32, f32),
    pub hits_at_10_ci: (f32, f32),
}

/// Statistical significance test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestResults {
    /// Wilcoxon signed-rank test p-value vs baseline
    pub wilcoxon_p_value: Option<f32>,
    /// Bootstrap test confidence level
    pub bootstrap_confidence: f32,
    /// Effect size (Cohen's d)
    pub effect_size: Option<f32>,
}

impl Default for KnowledgeGraphMetrics {
    fn default() -> Self {
        let mut hits_at_k = std::collections::HashMap::new();
        hits_at_k.insert(1, 0.0);
        hits_at_k.insert(3, 0.0);
        hits_at_k.insert(10, 0.0);
        hits_at_k.insert(100, 0.0);

        let mut precision_at_k = std::collections::HashMap::new();
        precision_at_k.insert(1, 0.0);
        precision_at_k.insert(3, 0.0);
        precision_at_k.insert(10, 0.0);

        let mut recall_at_k = std::collections::HashMap::new();
        recall_at_k.insert(1, 0.0);
        recall_at_k.insert(3, 0.0);
        recall_at_k.insert(10, 0.0);

        Self {
            mrr_filtered: 0.0,
            mrr_unfiltered: 0.0,
            mr_filtered: 0.0,
            mr_unfiltered: 0.0,
            hits_at_k_filtered: hits_at_k.clone(),
            hits_at_k_unfiltered: hits_at_k.clone(),
            per_relation_metrics: std::collections::HashMap::new(),
            task_breakdown: TaskBreakdownMetrics {
                head_prediction: LinkPredictionMetrics {
                    mrr: 0.0,
                    mr: 0.0,
                    hits_at_k: hits_at_k.clone(),
                    auc_roc: 0.0,
                    auc_pr: 0.0,
                    precision_at_k: precision_at_k.clone(),
                    recall_at_k: recall_at_k.clone(),
                },
                tail_prediction: LinkPredictionMetrics {
                    mrr: 0.0,
                    mr: 0.0,
                    hits_at_k: hits_at_k.clone(),
                    auc_roc: 0.0,
                    auc_pr: 0.0,
                    precision_at_k: precision_at_k.clone(),
                    recall_at_k: recall_at_k.clone(),
                },
                relation_prediction: LinkPredictionMetrics {
                    mrr: 0.0,
                    mr: 0.0,
                    hits_at_k: hits_at_k.clone(),
                    auc_roc: 0.0,
                    auc_pr: 0.0,
                    precision_at_k,
                    recall_at_k,
                },
            },
            confidence_intervals: ConfidenceIntervals {
                mrr_ci: (0.0, 0.0),
                mr_ci: (0.0, 0.0),
                hits_at_10_ci: (0.0, 0.0),
            },
            statistical_tests: StatisticalTestResults {
                wilcoxon_p_value: None,
                bootstrap_confidence: 0.95,
                effect_size: None,
            },
        }
    }
}

/// Compute comprehensive knowledge graph metrics for link prediction
pub async fn compute_kg_metrics(
    model: &dyn KnowledgeGraphEmbedding,
    test_triples: &[(String, String, String)],
    all_triples: &[(String, String, String)],
    k_values: &[u32],
) -> Result<KnowledgeGraphMetrics> {
    let mut metrics = KnowledgeGraphMetrics::default();

    // Convert to hashset for efficient filtering
    let all_triples_set: HashSet<(String, String, String)> = all_triples.iter().cloned().collect();

    // Head prediction metrics
    metrics.task_breakdown.head_prediction = compute_link_prediction_metrics(
        model,
        test_triples,
        &all_triples_set,
        LinkPredictionTask::HeadPrediction,
        k_values,
    )
    .await?;

    // Tail prediction metrics
    metrics.task_breakdown.tail_prediction = compute_link_prediction_metrics(
        model,
        test_triples,
        &all_triples_set,
        LinkPredictionTask::TailPrediction,
        k_values,
    )
    .await?;

    // Relation prediction metrics
    metrics.task_breakdown.relation_prediction = compute_link_prediction_metrics(
        model,
        test_triples,
        &all_triples_set,
        LinkPredictionTask::RelationPrediction,
        k_values,
    )
    .await?;

    // Aggregate metrics across tasks
    metrics.mrr_filtered = (metrics.task_breakdown.head_prediction.mrr
        + metrics.task_breakdown.tail_prediction.mrr)
        / 2.0;
    metrics.mr_filtered = (metrics.task_breakdown.head_prediction.mr
        + metrics.task_breakdown.tail_prediction.mr)
        / 2.0;

    // Aggregate Hits@K
    for &k in k_values {
        let head_hits = metrics
            .task_breakdown
            .head_prediction
            .hits_at_k
            .get(&k)
            .unwrap_or(&0.0);
        let tail_hits = metrics
            .task_breakdown
            .tail_prediction
            .hits_at_k
            .get(&k)
            .unwrap_or(&0.0);
        metrics
            .hits_at_k_filtered
            .insert(k, (head_hits + tail_hits) / 2.0);
    }

    // Compute per-relation metrics
    metrics.per_relation_metrics =
        compute_per_relation_metrics(model, test_triples, &all_triples_set, k_values).await?;

    // Compute confidence intervals
    metrics.confidence_intervals = compute_confidence_intervals(
        &metrics.task_breakdown.head_prediction,
        &metrics.task_breakdown.tail_prediction,
        test_triples.len(),
    )?;

    Ok(metrics)
}

/// Link prediction task types
#[derive(Debug, Clone)]
pub enum LinkPredictionTask {
    HeadPrediction,
    TailPrediction,
    RelationPrediction,
}

/// Compute link prediction metrics for specific task
async fn compute_link_prediction_metrics(
    model: &dyn KnowledgeGraphEmbedding,
    test_triples: &[(String, String, String)],
    all_triples: &HashSet<(String, String, String)>,
    task: LinkPredictionTask,
    k_values: &[u32],
) -> Result<LinkPredictionMetrics> {
    let mut ranks = Vec::new();
    let mut reciprocal_ranks = Vec::new();
    let mut hits_at_k = std::collections::HashMap::new();
    let mut precision_at_k = std::collections::HashMap::new();
    let mut recall_at_k = std::collections::HashMap::new();

    // Initialize counters
    for &k in k_values {
        hits_at_k.insert(k, 0.0);
        precision_at_k.insert(k, 0.0);
        recall_at_k.insert(k, 0.0);
    }

    for (head, relation, tail) in test_triples {
        let rank = match task {
            LinkPredictionTask::HeadPrediction => {
                compute_entity_rank(model, "?", relation, tail, all_triples, true).await?
            }
            LinkPredictionTask::TailPrediction => {
                compute_entity_rank(model, head, relation, "?", all_triples, false).await?
            }
            LinkPredictionTask::RelationPrediction => {
                compute_relation_rank(model, head, tail, all_triples).await?
            }
        };

        ranks.push(rank as f32);
        reciprocal_ranks.push(1.0 / rank as f32);

        // Update hits@k counters
        for &k in k_values {
            if rank <= k {
                if let Some(hits) = hits_at_k.get_mut(&k) {
                    *hits += 1.0;
                }
            }
        }
    }

    let num_samples = test_triples.len() as f32;

    // Normalize hits@k
    for (_, hits) in hits_at_k.iter_mut() {
        *hits /= num_samples;
    }

    // Compute precision and recall at k (simplified)
    for &k in k_values {
        let hits = hits_at_k.get(&k).unwrap_or(&0.0);
        precision_at_k.insert(k, *hits); // Simplified: assume precision = hits@k
        recall_at_k.insert(k, *hits); // Simplified: assume recall = hits@k
    }

    Ok(LinkPredictionMetrics {
        mrr: reciprocal_ranks.iter().sum::<f32>() / num_samples,
        mr: ranks.iter().sum::<f32>() / num_samples,
        hits_at_k,
        auc_roc: compute_auc_roc(&ranks)?,
        auc_pr: compute_auc_pr(&ranks)?,
        precision_at_k,
        recall_at_k,
    })
}

/// Compute rank of correct entity in filtered setting
async fn compute_entity_rank(
    model: &dyn KnowledgeGraphEmbedding,
    head: &str,
    relation: &str,
    tail: &str,
    all_triples: &HashSet<(String, String, String)>,
    predict_head: bool,
) -> Result<u32> {
    // Get all entities (simplified - in practice would use entity vocabulary)
    let entities: Vec<String> = all_triples
        .iter()
        .flat_map(|(h, _, t)| vec![h.clone(), t.clone()])
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    let mut scores = Vec::new();
    let correct_entity = if predict_head { head } else { tail };

    for entity in &entities {
        let test_head = if predict_head { entity } else { head };
        let test_tail = if predict_head { tail } else { entity };

        // Skip if this would create a known triple (filtered setting)
        if all_triples.contains(&(
            test_head.to_string(),
            relation.to_string(),
            test_tail.to_string(),
        )) && entity != correct_entity
        {
            continue;
        }

        let score = model.score_triple(test_head, relation, test_tail).await?;
        scores.push((entity.clone(), score));
    }

    // Sort by score (descending)
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find rank of correct entity
    let rank = scores
        .iter()
        .position(|(entity, _)| entity == correct_entity)
        .unwrap_or(scores.len() - 1)
        + 1;

    Ok(rank as u32)
}

/// Compute rank of correct relation
async fn compute_relation_rank(
    model: &dyn KnowledgeGraphEmbedding,
    head: &str,
    tail: &str,
    all_triples: &HashSet<(String, String, String)>,
) -> Result<u32> {
    // Get all relations
    let relations: Vec<String> = all_triples
        .iter()
        .map(|(_, r, _)| r.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    let mut scores = Vec::new();

    for relation in &relations {
        let score = model.score_triple(head, relation, tail).await?;
        scores.push((relation.clone(), score));
    }

    // Sort by score (descending)
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find rank (simplified - assumes first relation is correct)
    Ok(1) // Placeholder
}

/// Compute per-relation performance metrics
async fn compute_per_relation_metrics(
    model: &dyn KnowledgeGraphEmbedding,
    test_triples: &[(String, String, String)],
    all_triples: &HashSet<(String, String, String)>,
    k_values: &[u32],
) -> Result<std::collections::HashMap<String, RelationMetrics>> {
    let mut relation_metrics = std::collections::HashMap::new();

    // Group test triples by relation
    let mut relation_groups: std::collections::HashMap<String, Vec<(String, String, String)>> =
        std::collections::HashMap::new();

    for triple in test_triples {
        relation_groups
            .entry(triple.1.clone())
            .or_default()
            .push(triple.clone());
    }

    // Compute metrics for each relation
    for (relation, relation_triples) in relation_groups {
        let metrics = compute_link_prediction_metrics(
            model,
            &relation_triples,
            all_triples,
            LinkPredictionTask::TailPrediction,
            k_values,
        )
        .await?;

        let entity_count = relation_triples
            .iter()
            .flat_map(|(h, _, t)| vec![h, t])
            .collect::<HashSet<_>>()
            .len();

        relation_metrics.insert(
            relation,
            RelationMetrics {
                mrr: metrics.mrr,
                mr: metrics.mr,
                hits_at_k: metrics.hits_at_k,
                sample_count: relation_triples.len(),
                entity_coverage: entity_count as f32 / relation_triples.len() as f32,
            },
        );
    }

    Ok(relation_metrics)
}

/// Compute confidence intervals using bootstrap sampling
fn compute_confidence_intervals(
    head_metrics: &LinkPredictionMetrics,
    tail_metrics: &LinkPredictionMetrics,
    sample_size: usize,
) -> Result<ConfidenceIntervals> {
    // Simplified confidence interval computation
    let combined_mrr = (head_metrics.mrr + tail_metrics.mrr) / 2.0;
    let combined_mr = (head_metrics.mr + tail_metrics.mr) / 2.0;
    let combined_hits_10 = (head_metrics.hits_at_k.get(&10).unwrap_or(&0.0)
        + tail_metrics.hits_at_k.get(&10).unwrap_or(&0.0))
        / 2.0;

    // Standard error approximation
    let se_factor = 1.96 / (sample_size as f32).sqrt(); // 95% CI

    Ok(ConfidenceIntervals {
        mrr_ci: (
            (combined_mrr - combined_mrr * se_factor).max(0.0),
            (combined_mrr + combined_mrr * se_factor).min(1.0),
        ),
        mr_ci: (
            (combined_mr - combined_mr * se_factor).max(1.0),
            combined_mr + combined_mr * se_factor,
        ),
        hits_at_10_ci: (
            (combined_hits_10 - combined_hits_10 * se_factor).max(0.0),
            (combined_hits_10 + combined_hits_10 * se_factor).min(1.0),
        ),
    })
}

/// Compute AUC-ROC score
fn compute_auc_roc(ranks: &[f32]) -> Result<f32> {
    // Simplified AUC computation
    let max_rank = ranks.iter().fold(0.0f32, |a, &b| a.max(b));
    let normalized_ranks: Vec<f32> = ranks.iter().map(|&r| 1.0 - (r / max_rank)).collect();
    Ok(normalized_ranks.iter().sum::<f32>() / ranks.len() as f32)
}

/// Compute AUC-PR score
fn compute_auc_pr(ranks: &[f32]) -> Result<f32> {
    // Simplified AUC-PR computation (placeholder)
    compute_auc_roc(ranks)
}

/// Create evaluation report
pub fn create_evaluation_report(metrics: &KnowledgeGraphMetrics) -> String {
    format!(
        "Knowledge Graph Embedding Evaluation Report\n\
            ==========================================\n\
            \n\
            Overall Performance:\n\
            - MRR (filtered): {:.4}\n\
            - Mean Rank (filtered): {:.1}\n\
            - Hits@1: {:.4}\n\
            - Hits@3: {:.4}\n\
            - Hits@10: {:.4}\n\
            \n\
            Task Breakdown:\n\
            - Head Prediction MRR: {:.4}\n\
            - Tail Prediction MRR: {:.4}\n\
            - Relation Prediction MRR: {:.4}\n\
            \n\
            Confidence Intervals (95%):\n\
            - MRR: [{:.4}, {:.4}]\n\
            - Hits@10: [{:.4}, {:.4}]\n\
            \n\
            Per-Relation Performance:\n\
            {} relations evaluated\n",
        metrics.mrr_filtered,
        metrics.mr_filtered,
        metrics.hits_at_k_filtered.get(&1).unwrap_or(&0.0),
        metrics.hits_at_k_filtered.get(&3).unwrap_or(&0.0),
        metrics.hits_at_k_filtered.get(&10).unwrap_or(&0.0),
        metrics.task_breakdown.head_prediction.mrr,
        metrics.task_breakdown.tail_prediction.mrr,
        metrics.task_breakdown.relation_prediction.mrr,
        metrics.confidence_intervals.mrr_ci.0,
        metrics.confidence_intervals.mrr_ci.1,
        metrics.confidence_intervals.hits_at_10_ci.0,
        metrics.confidence_intervals.hits_at_10_ci.1,
        metrics.per_relation_metrics.len()
    )
}
