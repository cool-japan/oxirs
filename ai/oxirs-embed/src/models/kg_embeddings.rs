//! Knowledge Graph Embedding algorithms for link prediction.
//!
//! These algorithms learn dense vector representations of entities and
//! relations in a knowledge graph to predict missing triples. All math
//! is implemented with plain `Vec<f64>` to keep compilation simple and
//! avoid external numerical-library dependencies in this file.
//!
//! Implemented algorithms:
//! * **TransE**   – translating embeddings; score = −‖h + r − t‖₂
//! * **DistMult** – bilinear diagonal model; score = Σ(hᵢ · rᵢ · tᵢ)
//! * **RotatE**   – complex-space rotation; score = −‖h ∘ r − t‖₂

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by KG embedding operations.
#[derive(Debug)]
pub enum KgError {
    /// The model has not been trained yet.
    NotTrained,
    /// An entity ID is out of range.
    UnknownEntity(EntityId),
    /// A relation ID is out of range.
    UnknownRelation(RelationId),
    /// The embedding dimension is zero or otherwise invalid.
    InvalidDimension,
    /// No triples were provided for training.
    NoTrainingData,
    /// A numerical issue occurred (NaN / Inf).
    NumericalError(String),
    /// top-k is zero.
    InvalidTopK,
}

impl fmt::Display for KgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KgError::NotTrained => write!(f, "model has not been trained"),
            KgError::UnknownEntity(id) => write!(f, "unknown entity id {id}"),
            KgError::UnknownRelation(id) => write!(f, "unknown relation id {id}"),
            KgError::InvalidDimension => write!(f, "embedding dimension must be > 0"),
            KgError::NoTrainingData => write!(f, "no training triples provided"),
            KgError::NumericalError(msg) => write!(f, "numerical error: {msg}"),
            KgError::InvalidTopK => write!(f, "top_k must be > 0"),
        }
    }
}

impl std::error::Error for KgError {}

/// Convenience alias.
pub type KgResult<T> = Result<T, KgError>;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Index of an entity in the embedding table.
pub type EntityId = usize;
/// Index of a relation in the embedding table.
pub type RelationId = usize;

/// A single (head, relation, tail) triple.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KgTriple {
    pub head: EntityId,
    pub relation: RelationId,
    pub tail: EntityId,
}

impl KgTriple {
    /// Construct a new triple.
    pub fn new(head: EntityId, relation: RelationId, tail: EntityId) -> Self {
        Self {
            head,
            relation,
            tail,
        }
    }
}

/// Hyper-parameters shared by all KG embedding trainers.
#[derive(Debug, Clone)]
pub struct KgEmbeddingConfig {
    /// Dimensionality of entity and relation vectors (e.g. 50, 100, 200).
    pub embedding_dim: usize,
    /// SGD learning rate.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub num_epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Number of negative samples generated per positive triple.
    pub neg_samples: usize,
    /// Margin γ used in max-margin / hinge loss (primarily TransE).
    pub margin: f64,
    /// L2 regularisation coefficient.
    pub regularization: f64,
    /// Fixed seed for reproducibility (simple LCG).
    pub seed: u64,
}

impl Default for KgEmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 50,
            learning_rate: 0.01,
            num_epochs: 100,
            batch_size: 32,
            neg_samples: 1,
            margin: 1.0,
            regularization: 1e-4,
            seed: 42,
        }
    }
}

/// Trained embedding tables together with string→id look-ups.
#[derive(Debug, Clone)]
pub struct KgEmbeddings {
    /// Entity embedding matrix: `entity_embeddings[entity_id]` is a `dim`-vector.
    pub entity_embeddings: Vec<Vec<f64>>,
    /// Relation embedding matrix: `relation_embeddings[relation_id]` is a `dim`-vector.
    pub relation_embeddings: Vec<Vec<f64>>,
    /// Map from entity string to numeric id.
    pub entity_to_id: HashMap<String, EntityId>,
    /// Map from relation string to numeric id.
    pub relation_to_id: HashMap<String, RelationId>,
}

/// Loss and convergence information collected during training.
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Mean loss recorded at the end of each epoch.
    pub losses: Vec<f64>,
    /// Loss at the final epoch.
    pub final_loss: f64,
    /// Total number of epochs that were actually run.
    pub epochs_trained: usize,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Shared interface for all KG embedding models.
pub trait KgModel {
    /// Score a (head, relation, tail) triple. Higher ⟹ more plausible.
    fn score(&self, triple: &KgTriple) -> KgResult<f64>;

    /// Rank all entities as possible tails; returns top-`k` (entity, score) pairs.
    fn predict_tail(
        &self,
        head: EntityId,
        relation: RelationId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>>;

    /// Rank all entities as possible heads; returns top-`k` (entity, score) pairs.
    fn predict_head(
        &self,
        relation: RelationId,
        tail: EntityId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>>;
}

// ---------------------------------------------------------------------------
// Minimal deterministic pseudo-random number generator (LCG)
// ---------------------------------------------------------------------------

/// A tiny, dependency-free LCG used for reproducible weight initialisation
/// and negative-sample corruption. Not suitable for cryptographic use.
#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Next value in [0, 1).
    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `[0, n)`.
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_f64() * n as f64) as usize % n
    }
}

// ---------------------------------------------------------------------------
// Helper math on plain Vec<f64>
// ---------------------------------------------------------------------------

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn l2_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn clamp_vec(v: &mut [f64], lo: f64, hi: f64) {
    for x in v.iter_mut() {
        *x = x.clamp(lo, hi);
    }
}

fn normalize_vec(v: &mut [f64]) {
    let norm = l2_norm(v);
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Negative sampling
// ---------------------------------------------------------------------------

/// Generate a corrupted triple by replacing head OR tail randomly.
fn corrupt_triple(
    triple: &KgTriple,
    num_entities: usize,
    positive_set: &std::collections::HashSet<(usize, usize, usize)>,
    rng: &mut Lcg,
) -> KgTriple {
    // Try up to 20 times to find a genuinely negative triple.
    for _ in 0..20 {
        let corrupt_head = rng.next_usize(2) == 0;
        let candidate = if corrupt_head {
            let new_head = rng.next_usize(num_entities);
            KgTriple::new(new_head, triple.relation, triple.tail)
        } else {
            let new_tail = rng.next_usize(num_entities);
            KgTriple::new(triple.head, triple.relation, new_tail)
        };
        if !positive_set.contains(&(candidate.head, candidate.relation, candidate.tail)) {
            return candidate;
        }
    }
    // Fallback: return candidate even if it overlaps (rare with large entity sets).
    let new_tail = (triple.tail + 1) % num_entities;
    KgTriple::new(triple.head, triple.relation, new_tail)
}

// ---------------------------------------------------------------------------
// TransE
// ---------------------------------------------------------------------------

/// **TransE** – translating embeddings.
///
/// The scoring function is `−‖h + r − t‖₂`.  
/// Training uses max-margin (hinge) loss with SGD and entity-norm projection.
#[derive(Debug, Clone)]
pub struct TransE {
    pub config: KgEmbeddingConfig,
    pub embeddings: Option<KgEmbeddings>,
    num_entities: usize,
    num_relations: usize,
}

impl TransE {
    /// Create a new, untrained TransE model.
    pub fn new(config: KgEmbeddingConfig) -> Self {
        Self {
            config,
            embeddings: None,
            num_entities: 0,
            num_relations: 0,
        }
    }

    /// Train on the provided triples.
    ///
    /// `num_entities` and `num_relations` define the size of the embedding
    /// tables; IDs in `triples` must be in `[0, num_entities)` /
    /// `[0, num_relations)`.
    pub fn train(
        &mut self,
        triples: &[KgTriple],
        num_entities: usize,
        num_relations: usize,
    ) -> KgResult<TrainingHistory> {
        if triples.is_empty() {
            return Err(KgError::NoTrainingData);
        }
        if self.config.embedding_dim == 0 {
            return Err(KgError::InvalidDimension);
        }
        self.num_entities = num_entities;
        self.num_relations = num_relations;

        let dim = self.config.embedding_dim;
        let mut rng = Lcg::new(self.config.seed);

        // Initialise entity embeddings uniformly in [-6/√dim, 6/√dim].
        let bound = 6.0 / (dim as f64).sqrt();
        let mut ent_emb: Vec<Vec<f64>> = (0..num_entities)
            .map(|_| {
                let mut v: Vec<f64> = (0..dim)
                    .map(|_| (rng.next_f64() * 2.0 - 1.0) * bound)
                    .collect();
                normalize_vec(&mut v);
                v
            })
            .collect();

        // Initialise relation embeddings uniformly in [-6/√dim, 6/√dim].
        let mut rel_emb: Vec<Vec<f64>> = (0..num_relations)
            .map(|_| {
                (0..dim)
                    .map(|_| (rng.next_f64() * 2.0 - 1.0) * bound)
                    .collect()
            })
            .collect();

        // Build positive-triple look-up for negative sampling.
        let positive_set: std::collections::HashSet<(usize, usize, usize)> = triples
            .iter()
            .map(|t| (t.head, t.relation, t.tail))
            .collect();

        let lr = self.config.learning_rate;
        let margin = self.config.margin;
        let reg = self.config.regularization;
        let mut losses = Vec::with_capacity(self.config.num_epochs);

        for _epoch in 0..self.config.num_epochs {
            let mut epoch_loss = 0.0_f64;
            let mut count = 0usize;

            for pos in triples {
                for _ in 0..self.config.neg_samples {
                    let neg = corrupt_triple(pos, num_entities, &positive_set, &mut rng);

                    let h_pos = &ent_emb[pos.head];
                    let r = &rel_emb[pos.relation];
                    let t_pos = &ent_emb[pos.tail];
                    let h_neg = &ent_emb[neg.head];
                    let t_neg = &ent_emb[neg.tail];

                    // Compute h+r−t for positive and negative.
                    let pos_diff: Vec<f64> = (0..dim).map(|i| h_pos[i] + r[i] - t_pos[i]).collect();
                    let neg_diff: Vec<f64> = (0..dim).map(|i| h_neg[i] + r[i] - t_neg[i]).collect();

                    let d_pos = l2_norm(&pos_diff);
                    let d_neg = l2_norm(&neg_diff);

                    let loss = (margin + d_pos - d_neg).max(0.0);
                    epoch_loss += loss;
                    count += 1;

                    if loss > 0.0 {
                        // Gradient of L2 norm: ∂‖v‖/∂vᵢ = vᵢ/‖v‖.
                        let grad_pos: Vec<f64> = if d_pos > 1e-12 {
                            pos_diff.iter().map(|x| x / d_pos).collect()
                        } else {
                            vec![0.0; dim]
                        };
                        let grad_neg: Vec<f64> = if d_neg > 1e-12 {
                            neg_diff.iter().map(|x| x / d_neg).collect()
                        } else {
                            vec![0.0; dim]
                        };

                        // Update positive triple components.
                        for i in 0..dim {
                            let g = grad_pos[i];
                            ent_emb[pos.head][i] -= lr * (g + reg * ent_emb[pos.head][i]);
                            rel_emb[pos.relation][i] -= lr * (g + reg * rel_emb[pos.relation][i]);
                            ent_emb[pos.tail][i] += lr * (g - reg * ent_emb[pos.tail][i]);
                        }

                        // Update negative triple components.
                        for i in 0..dim {
                            let g = grad_neg[i];
                            ent_emb[neg.head][i] += lr * (g + reg * ent_emb[neg.head][i]);
                            ent_emb[neg.tail][i] -= lr * (g - reg * ent_emb[neg.tail][i]);
                        }
                    }

                    // Project entity embeddings back to unit sphere.
                    normalize_vec(&mut ent_emb[pos.head]);
                    normalize_vec(&mut ent_emb[pos.tail]);
                    normalize_vec(&mut ent_emb[neg.head]);
                    normalize_vec(&mut ent_emb[neg.tail]);
                }
            }

            let mean_loss = if count > 0 {
                epoch_loss / count as f64
            } else {
                0.0
            };
            losses.push(mean_loss);
        }

        let final_loss = losses.last().copied().unwrap_or(0.0);
        let epochs_trained = losses.len();

        self.embeddings = Some(KgEmbeddings {
            entity_embeddings: ent_emb,
            relation_embeddings: rel_emb,
            entity_to_id: HashMap::new(),
            relation_to_id: HashMap::new(),
        });

        Ok(TrainingHistory {
            losses,
            final_loss,
            epochs_trained,
        })
    }

    /// Score a triple: −‖h + r − t‖₂  (higher = more plausible).
    pub fn score(&self, triple: &KgTriple) -> KgResult<f64> {
        let emb = self.embeddings.as_ref().ok_or(KgError::NotTrained)?;
        let h = emb
            .entity_embeddings
            .get(triple.head)
            .ok_or(KgError::UnknownEntity(triple.head))?;
        let r = emb
            .relation_embeddings
            .get(triple.relation)
            .ok_or(KgError::UnknownRelation(triple.relation))?;
        let t = emb
            .entity_embeddings
            .get(triple.tail)
            .ok_or(KgError::UnknownEntity(triple.tail))?;

        Ok(-Self::score_fn(h, r, t))
    }

    /// Rank all entities as candidate tails; return top-`k`.
    pub fn predict_tail(
        &self,
        head: EntityId,
        relation: RelationId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>> {
        if top_k == 0 {
            return Err(KgError::InvalidTopK);
        }
        let emb = self.embeddings.as_ref().ok_or(KgError::NotTrained)?;
        let h = emb
            .entity_embeddings
            .get(head)
            .ok_or(KgError::UnknownEntity(head))?;
        let r = emb
            .relation_embeddings
            .get(relation)
            .ok_or(KgError::UnknownRelation(relation))?;

        let mut scored: Vec<(EntityId, f64)> = emb
            .entity_embeddings
            .iter()
            .enumerate()
            .map(|(id, t)| (id, -Self::score_fn(h, r, t)))
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Rank all entities as candidate heads; return top-`k`.
    pub fn predict_head(
        &self,
        relation: RelationId,
        tail: EntityId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>> {
        if top_k == 0 {
            return Err(KgError::InvalidTopK);
        }
        let emb = self.embeddings.as_ref().ok_or(KgError::NotTrained)?;
        let r = emb
            .relation_embeddings
            .get(relation)
            .ok_or(KgError::UnknownRelation(relation))?;
        let t = emb
            .entity_embeddings
            .get(tail)
            .ok_or(KgError::UnknownEntity(tail))?;

        let mut scored: Vec<(EntityId, f64)> = emb
            .entity_embeddings
            .iter()
            .enumerate()
            .map(|(id, h)| (id, -Self::score_fn(h, r, t)))
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Project all entity embeddings onto the unit sphere.
    pub fn normalize_entities(&mut self) {
        if let Some(ref mut emb) = self.embeddings {
            for v in emb.entity_embeddings.iter_mut() {
                normalize_vec(v);
            }
        }
    }

    /// TransE distance: ‖h + r − t‖₂.
    fn score_fn(h: &[f64], r: &[f64], t: &[f64]) -> f64 {
        let diff: Vec<f64> = (0..h.len()).map(|i| h[i] + r[i] - t[i]).collect();
        l2_norm(&diff)
    }
}

impl KgModel for TransE {
    fn score(&self, triple: &KgTriple) -> KgResult<f64> {
        self.score(triple)
    }

    fn predict_tail(
        &self,
        head: EntityId,
        relation: RelationId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>> {
        self.predict_tail(head, relation, top_k)
    }

    fn predict_head(
        &self,
        relation: RelationId,
        tail: EntityId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>> {
        self.predict_head(relation, tail, top_k)
    }
}

// ---------------------------------------------------------------------------
// DistMult
// ---------------------------------------------------------------------------

/// **DistMult** – bilinear diagonal scoring.
///
/// Scoring: `Σ(hᵢ · rᵢ · tᵢ)`.  
/// Trained with softplus (logistic) loss and SGD.
#[derive(Debug, Clone)]
pub struct DistMult {
    pub config: KgEmbeddingConfig,
    pub embeddings: Option<KgEmbeddings>,
    num_entities: usize,
    num_relations: usize,
}

impl DistMult {
    /// Create a new, untrained DistMult model.
    pub fn new(config: KgEmbeddingConfig) -> Self {
        Self {
            config,
            embeddings: None,
            num_entities: 0,
            num_relations: 0,
        }
    }

    /// Train using negative-sampling and softplus loss.
    pub fn train(
        &mut self,
        triples: &[KgTriple],
        num_entities: usize,
        num_relations: usize,
    ) -> KgResult<TrainingHistory> {
        if triples.is_empty() {
            return Err(KgError::NoTrainingData);
        }
        if self.config.embedding_dim == 0 {
            return Err(KgError::InvalidDimension);
        }
        self.num_entities = num_entities;
        self.num_relations = num_relations;

        let dim = self.config.embedding_dim;
        let mut rng = Lcg::new(self.config.seed);
        let bound = 1.0 / (dim as f64).sqrt();

        let mut ent_emb: Vec<Vec<f64>> = (0..num_entities)
            .map(|_| {
                (0..dim)
                    .map(|_| (rng.next_f64() * 2.0 - 1.0) * bound)
                    .collect()
            })
            .collect();
        let mut rel_emb: Vec<Vec<f64>> = (0..num_relations)
            .map(|_| {
                (0..dim)
                    .map(|_| (rng.next_f64() * 2.0 - 1.0) * bound)
                    .collect()
            })
            .collect();

        let positive_set: std::collections::HashSet<(usize, usize, usize)> = triples
            .iter()
            .map(|t| (t.head, t.relation, t.tail))
            .collect();

        let lr = self.config.learning_rate;
        let reg = self.config.regularization;
        let mut losses = Vec::with_capacity(self.config.num_epochs);

        for _epoch in 0..self.config.num_epochs {
            let mut epoch_loss = 0.0_f64;
            let mut count = 0usize;

            for pos in triples {
                // Positive sample loss: −log σ(score_pos)
                {
                    let s = Self::score_fn(
                        &ent_emb[pos.head],
                        &rel_emb[pos.relation],
                        &ent_emb[pos.tail],
                    );
                    let sig = sigmoid(s);
                    let loss = -sig.ln().max(-100.0);
                    epoch_loss += loss;
                    count += 1;

                    // Gradient: -(1 - σ(s)) · ∂s/∂params
                    let g = -(1.0 - sig);
                    for i in 0..dim {
                        let h_i = ent_emb[pos.head][i];
                        let r_i = rel_emb[pos.relation][i];
                        let t_i = ent_emb[pos.tail][i];
                        ent_emb[pos.head][i] -= lr * (g * r_i * t_i + reg * h_i);
                        rel_emb[pos.relation][i] -= lr * (g * h_i * t_i + reg * r_i);
                        ent_emb[pos.tail][i] -= lr * (g * h_i * r_i + reg * t_i);
                    }
                    clamp_vec(&mut ent_emb[pos.head], -10.0, 10.0);
                    clamp_vec(&mut rel_emb[pos.relation], -10.0, 10.0);
                    clamp_vec(&mut ent_emb[pos.tail], -10.0, 10.0);
                }

                for _ in 0..self.config.neg_samples {
                    let neg = corrupt_triple(pos, num_entities, &positive_set, &mut rng);
                    let s = Self::score_fn(
                        &ent_emb[neg.head],
                        &rel_emb[neg.relation],
                        &ent_emb[neg.tail],
                    );
                    let sig = sigmoid(-s);
                    let loss = -sig.ln().max(-100.0);
                    epoch_loss += loss;
                    count += 1;

                    let g = 1.0 - sig; // ∂loss/∂s = σ(s) - 0
                    for i in 0..dim {
                        let h_i = ent_emb[neg.head][i];
                        let r_i = rel_emb[neg.relation][i];
                        let t_i = ent_emb[neg.tail][i];
                        ent_emb[neg.head][i] -= lr * (g * r_i * t_i + reg * h_i);
                        rel_emb[neg.relation][i] -= lr * (g * h_i * t_i + reg * r_i);
                        ent_emb[neg.tail][i] -= lr * (g * h_i * r_i + reg * t_i);
                    }
                    clamp_vec(&mut ent_emb[neg.head], -10.0, 10.0);
                    clamp_vec(&mut ent_emb[neg.tail], -10.0, 10.0);
                }
            }

            let mean_loss = if count > 0 {
                epoch_loss / count as f64
            } else {
                0.0
            };
            losses.push(mean_loss);
        }

        let final_loss = losses.last().copied().unwrap_or(0.0);
        let epochs_trained = losses.len();

        self.embeddings = Some(KgEmbeddings {
            entity_embeddings: ent_emb,
            relation_embeddings: rel_emb,
            entity_to_id: HashMap::new(),
            relation_to_id: HashMap::new(),
        });

        Ok(TrainingHistory {
            losses,
            final_loss,
            epochs_trained,
        })
    }

    /// Score a triple: Σ(hᵢ · rᵢ · tᵢ).
    pub fn score(&self, triple: &KgTriple) -> KgResult<f64> {
        let emb = self.embeddings.as_ref().ok_or(KgError::NotTrained)?;
        let h = emb
            .entity_embeddings
            .get(triple.head)
            .ok_or(KgError::UnknownEntity(triple.head))?;
        let r = emb
            .relation_embeddings
            .get(triple.relation)
            .ok_or(KgError::UnknownRelation(triple.relation))?;
        let t = emb
            .entity_embeddings
            .get(triple.tail)
            .ok_or(KgError::UnknownEntity(triple.tail))?;
        Ok(Self::score_fn(h, r, t))
    }

    /// Rank all entities as candidate tails; return top-`k`.
    pub fn predict_tail(
        &self,
        head: EntityId,
        relation: RelationId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>> {
        if top_k == 0 {
            return Err(KgError::InvalidTopK);
        }
        let emb = self.embeddings.as_ref().ok_or(KgError::NotTrained)?;
        let h = emb
            .entity_embeddings
            .get(head)
            .ok_or(KgError::UnknownEntity(head))?;
        let r = emb
            .relation_embeddings
            .get(relation)
            .ok_or(KgError::UnknownRelation(relation))?;

        let mut scored: Vec<(EntityId, f64)> = emb
            .entity_embeddings
            .iter()
            .enumerate()
            .map(|(id, t)| (id, Self::score_fn(h, r, t)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Rank all entities as candidate heads; return top-`k`.
    pub fn predict_head(
        &self,
        relation: RelationId,
        tail: EntityId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>> {
        if top_k == 0 {
            return Err(KgError::InvalidTopK);
        }
        let emb = self.embeddings.as_ref().ok_or(KgError::NotTrained)?;
        let r = emb
            .relation_embeddings
            .get(relation)
            .ok_or(KgError::UnknownRelation(relation))?;
        let t = emb
            .entity_embeddings
            .get(tail)
            .ok_or(KgError::UnknownEntity(tail))?;

        let mut scored: Vec<(EntityId, f64)> = emb
            .entity_embeddings
            .iter()
            .enumerate()
            .map(|(id, h)| (id, Self::score_fn(h, r, t)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// DistMult scoring: Σ(hᵢ · rᵢ · tᵢ).
    fn score_fn(h: &[f64], r: &[f64], t: &[f64]) -> f64 {
        h.iter()
            .zip(r.iter())
            .zip(t.iter())
            .map(|((hi, ri), ti)| hi * ri * ti)
            .sum()
    }
}

impl KgModel for DistMult {
    fn score(&self, triple: &KgTriple) -> KgResult<f64> {
        self.score(triple)
    }

    fn predict_tail(
        &self,
        head: EntityId,
        relation: RelationId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>> {
        self.predict_tail(head, relation, top_k)
    }

    fn predict_head(
        &self,
        relation: RelationId,
        tail: EntityId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>> {
        self.predict_head(relation, tail, top_k)
    }
}

// ---------------------------------------------------------------------------
// RotatE
// ---------------------------------------------------------------------------

/// **RotatE** – knowledge-graph embedding by relational rotation in ℂ.
///
/// Each entity is embedded in ℂ^(d/2); each relation is a phase vector
/// θ ∈ ℝ^(d/2).  Scoring: `−‖h ∘ r − t‖` where `∘` is element-wise
/// complex multiplication with r as a unit-modulus complex number
/// (e^{iθ}).
#[derive(Debug, Clone)]
pub struct RotatE {
    pub config: KgEmbeddingConfig,
    /// Real parts of entity embeddings: `real[entity_id][k]`.
    pub entity_re: Option<Vec<Vec<f64>>>,
    /// Imaginary parts of entity embeddings: `imag[entity_id][k]`.
    pub entity_im: Option<Vec<Vec<f64>>>,
    /// Relation phase vectors: `phases[relation_id][k]`.
    pub relation_phases: Option<Vec<Vec<f64>>>,
    num_entities: usize,
    num_relations: usize,
}

impl RotatE {
    /// Create a new, untrained RotatE model.
    pub fn new(config: KgEmbeddingConfig) -> Self {
        Self {
            config,
            entity_re: None,
            entity_im: None,
            relation_phases: None,
            num_entities: 0,
            num_relations: 0,
        }
    }

    /// Train using negative-sampling and max-margin loss.
    pub fn train(
        &mut self,
        triples: &[KgTriple],
        num_entities: usize,
        num_relations: usize,
    ) -> KgResult<TrainingHistory> {
        if triples.is_empty() {
            return Err(KgError::NoTrainingData);
        }
        if self.config.embedding_dim == 0 {
            return Err(KgError::InvalidDimension);
        }

        self.num_entities = num_entities;
        self.num_relations = num_relations;

        // Half-dimension: d/2 complex components per embedding.
        let half_dim = (self.config.embedding_dim + 1) / 2;
        let mut rng = Lcg::new(self.config.seed);
        let pi = std::f64::consts::PI;

        // Entity embeddings: unit-modulus complex (cos θ, sin θ).
        let mut ent_re: Vec<Vec<f64>> = (0..num_entities)
            .map(|_| (0..half_dim).map(|_| rng.next_f64() * 2.0 - 1.0).collect())
            .collect();
        let mut ent_im: Vec<Vec<f64>> = (0..num_entities)
            .map(|_| (0..half_dim).map(|_| rng.next_f64() * 2.0 - 1.0).collect())
            .collect();

        // Normalise to unit modulus initially.
        for i in 0..num_entities {
            for k in 0..half_dim {
                let norm = (ent_re[i][k].powi(2) + ent_im[i][k].powi(2))
                    .sqrt()
                    .max(1e-12);
                ent_re[i][k] /= norm;
                ent_im[i][k] /= norm;
            }
        }

        // Relation phases in (−π, π).
        let mut rel_phases: Vec<Vec<f64>> = (0..num_relations)
            .map(|_| {
                (0..half_dim)
                    .map(|_| (rng.next_f64() * 2.0 - 1.0) * pi)
                    .collect()
            })
            .collect();

        let positive_set: std::collections::HashSet<(usize, usize, usize)> = triples
            .iter()
            .map(|t| (t.head, t.relation, t.tail))
            .collect();

        let lr = self.config.learning_rate;
        let margin = self.config.margin;
        let reg = self.config.regularization;
        let mut losses = Vec::with_capacity(self.config.num_epochs);

        for _epoch in 0..self.config.num_epochs {
            let mut epoch_loss = 0.0_f64;
            let mut count = 0usize;

            for pos in triples {
                for _ in 0..self.config.neg_samples {
                    let neg = corrupt_triple(pos, num_entities, &positive_set, &mut rng);

                    let d_pos = Self::dist_fn(
                        &ent_re[pos.head],
                        &ent_im[pos.head],
                        &rel_phases[pos.relation],
                        &ent_re[pos.tail],
                        &ent_im[pos.tail],
                    );
                    let d_neg = Self::dist_fn(
                        &ent_re[neg.head],
                        &ent_im[neg.head],
                        &rel_phases[neg.relation],
                        &ent_re[neg.tail],
                        &ent_im[neg.tail],
                    );

                    let loss = (margin + d_pos - d_neg).max(0.0);
                    epoch_loss += loss;
                    count += 1;

                    if loss > 0.0 && d_pos > 1e-12 {
                        // Gradient of ‖h∘r − t‖ w.r.t. phases and entity components.
                        let r_re: Vec<f64> = rel_phases[pos.relation]
                            .iter()
                            .map(|&ph| ph.cos())
                            .collect();
                        let r_im: Vec<f64> = rel_phases[pos.relation]
                            .iter()
                            .map(|&ph| ph.sin())
                            .collect();

                        for k in 0..half_dim {
                            let (res_re, res_im) = Self::complex_multiply(
                                ent_re[pos.head][k],
                                ent_im[pos.head][k],
                                r_re[k],
                                r_im[k],
                            );
                            let err_re = res_re - ent_re[pos.tail][k];
                            let err_im = res_im - ent_im[pos.tail][k];

                            // Gradients (positive sample – push apart).
                            let g_scale = 1.0 / d_pos;

                            // d(dist)/d(h_re_k)
                            let d_h_re = g_scale * (err_re * r_re[k] + err_im * r_im[k]);
                            // d(dist)/d(h_im_k)
                            let d_h_im = g_scale * (err_im * r_re[k] - err_re * r_im[k]);
                            // d(dist)/d(phase_k) = g * (-h_re*sin + h_im*cos)*err_re + (h_re*cos + h_im*(-sin))*err_im
                            let d_ph = g_scale
                                * ((-ent_re[pos.head][k] * r_im[k]
                                    + ent_im[pos.head][k] * r_re[k])
                                    * err_re
                                    + (-ent_re[pos.head][k] * r_re[k]
                                        - ent_im[pos.head][k] * r_im[k])
                                        * err_im);
                            // d(dist)/d(t_re_k)
                            let d_t_re = g_scale * (-err_re);
                            // d(dist)/d(t_im_k)
                            let d_t_im = g_scale * (-err_im);

                            ent_re[pos.head][k] -= lr * (d_h_re + reg * ent_re[pos.head][k]);
                            ent_im[pos.head][k] -= lr * (d_h_im + reg * ent_im[pos.head][k]);
                            rel_phases[pos.relation][k] -=
                                lr * (d_ph + reg * rel_phases[pos.relation][k]);
                            ent_re[pos.tail][k] -= lr * (d_t_re + reg * ent_re[pos.tail][k]);
                            ent_im[pos.tail][k] -= lr * (d_t_im + reg * ent_im[pos.tail][k]);
                        }

                        // Keep phases bounded in (−2π, 2π).
                        for ph in rel_phases[pos.relation].iter_mut() {
                            *ph = ph.clamp(-2.0 * pi, 2.0 * pi);
                        }
                    }
                }
            }

            let mean_loss = if count > 0 {
                epoch_loss / count as f64
            } else {
                0.0
            };
            losses.push(mean_loss);
        }

        let final_loss = losses.last().copied().unwrap_or(0.0);
        let epochs_trained = losses.len();

        self.entity_re = Some(ent_re);
        self.entity_im = Some(ent_im);
        self.relation_phases = Some(rel_phases);

        Ok(TrainingHistory {
            losses,
            final_loss,
            epochs_trained,
        })
    }

    /// Score a triple: `−‖h ∘ e^{iθ} − t‖`.
    pub fn score(&self, triple: &KgTriple) -> KgResult<f64> {
        let ent_re = self.entity_re.as_ref().ok_or(KgError::NotTrained)?;
        let ent_im = self.entity_im.as_ref().ok_or(KgError::NotTrained)?;
        let phases = self.relation_phases.as_ref().ok_or(KgError::NotTrained)?;

        let h_re = ent_re
            .get(triple.head)
            .ok_or(KgError::UnknownEntity(triple.head))?;
        let h_im = ent_im
            .get(triple.head)
            .ok_or(KgError::UnknownEntity(triple.head))?;
        let ph = phases
            .get(triple.relation)
            .ok_or(KgError::UnknownRelation(triple.relation))?;
        let t_re = ent_re
            .get(triple.tail)
            .ok_or(KgError::UnknownEntity(triple.tail))?;
        let t_im = ent_im
            .get(triple.tail)
            .ok_or(KgError::UnknownEntity(triple.tail))?;

        Ok(-Self::dist_fn(h_re, h_im, ph, t_re, t_im))
    }

    /// Rank all entities as candidate tails; return top-`k`.
    pub fn predict_tail(
        &self,
        head: EntityId,
        relation: RelationId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>> {
        if top_k == 0 {
            return Err(KgError::InvalidTopK);
        }
        let ent_re = self.entity_re.as_ref().ok_or(KgError::NotTrained)?;
        let ent_im = self.entity_im.as_ref().ok_or(KgError::NotTrained)?;
        let phases = self.relation_phases.as_ref().ok_or(KgError::NotTrained)?;

        let h_re = ent_re.get(head).ok_or(KgError::UnknownEntity(head))?;
        let h_im = ent_im.get(head).ok_or(KgError::UnknownEntity(head))?;
        let ph = phases
            .get(relation)
            .ok_or(KgError::UnknownRelation(relation))?;

        let num = ent_re.len();
        let mut scored: Vec<(EntityId, f64)> = (0..num)
            .map(|id| (id, -Self::dist_fn(h_re, h_im, ph, &ent_re[id], &ent_im[id])))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Rank all entities as candidate heads; return top-`k`.
    pub fn predict_head(
        &self,
        relation: RelationId,
        tail: EntityId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>> {
        if top_k == 0 {
            return Err(KgError::InvalidTopK);
        }
        let ent_re = self.entity_re.as_ref().ok_or(KgError::NotTrained)?;
        let ent_im = self.entity_im.as_ref().ok_or(KgError::NotTrained)?;
        let phases = self.relation_phases.as_ref().ok_or(KgError::NotTrained)?;

        let ph = phases
            .get(relation)
            .ok_or(KgError::UnknownRelation(relation))?;
        let t_re = ent_re.get(tail).ok_or(KgError::UnknownEntity(tail))?;
        let t_im = ent_im.get(tail).ok_or(KgError::UnknownEntity(tail))?;

        let num = ent_re.len();
        // For head prediction we find h such that h ∘ r ≈ t,
        // i.e. h ≈ t ∘ r̄ (conjugate rotation).
        // Score is still computed with the standard formula.
        let mut scored: Vec<(EntityId, f64)> = (0..num)
            .map(|id| (id, -Self::dist_fn(&ent_re[id], &ent_im[id], ph, t_re, t_im)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Distance: ‖h ∘ e^{iθ} − t‖.
    fn dist_fn(h_re: &[f64], h_im: &[f64], phases: &[f64], t_re: &[f64], t_im: &[f64]) -> f64 {
        let sum_sq: f64 = phases
            .iter()
            .enumerate()
            .map(|(k, &ph)| {
                let (res_re, res_im) = Self::complex_multiply(h_re[k], h_im[k], ph.cos(), ph.sin());
                (res_re - t_re[k]).powi(2) + (res_im - t_im[k]).powi(2)
            })
            .sum();
        sum_sq.sqrt()
    }

    /// Complex multiplication: (a + ib)(c + id) = (ac − bd) + i(ad + bc).
    pub fn complex_multiply(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> (f64, f64) {
        (a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re)
    }
}

impl KgModel for RotatE {
    fn score(&self, triple: &KgTriple) -> KgResult<f64> {
        self.score(triple)
    }

    fn predict_tail(
        &self,
        head: EntityId,
        relation: RelationId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>> {
        self.predict_tail(head, relation, top_k)
    }

    fn predict_head(
        &self,
        relation: RelationId,
        tail: EntityId,
        top_k: usize,
    ) -> KgResult<Vec<(EntityId, f64)>> {
        self.predict_head(relation, tail, top_k)
    }
}

// ---------------------------------------------------------------------------
// Link prediction evaluation metrics
// ---------------------------------------------------------------------------

/// Standard link-prediction evaluation metrics.
pub struct LinkPredictionEvaluator;

impl LinkPredictionEvaluator {
    /// Hits@K: fraction of test triples for which the true tail entity
    /// appears in the top-`k` predictions.
    pub fn hits_at_k(model: &dyn KgModel, test_triples: &[KgTriple], k: usize) -> f64 {
        if test_triples.is_empty() || k == 0 {
            return 0.0;
        }
        let hits: usize = test_triples
            .iter()
            .filter(|t| {
                model
                    .predict_tail(t.head, t.relation, k)
                    .map(|preds| preds.iter().any(|(eid, _)| *eid == t.tail))
                    .unwrap_or(false)
            })
            .count();
        hits as f64 / test_triples.len() as f64
    }

    /// Mean Rank: average rank of the true tail entity across all test triples
    /// (lower is better).
    pub fn mean_rank(model: &dyn KgModel, test_triples: &[KgTriple], num_entities: usize) -> f64 {
        if test_triples.is_empty() {
            return 0.0;
        }
        let total: usize = test_triples
            .iter()
            .map(|t| {
                model
                    .predict_tail(t.head, t.relation, num_entities)
                    .map(|preds| {
                        preds
                            .iter()
                            .position(|(eid, _)| *eid == t.tail)
                            .map(|p| p + 1)
                            .unwrap_or(num_entities + 1)
                    })
                    .unwrap_or(num_entities + 1)
            })
            .sum();
        total as f64 / test_triples.len() as f64
    }

    /// Mean Reciprocal Rank: average of 1/rank for the true tail entity
    /// (higher is better; max = 1.0).
    pub fn mrr(model: &dyn KgModel, test_triples: &[KgTriple], num_entities: usize) -> f64 {
        if test_triples.is_empty() {
            return 0.0;
        }
        let sum: f64 = test_triples
            .iter()
            .map(|t| {
                model
                    .predict_tail(t.head, t.relation, num_entities)
                    .map(|preds| {
                        preds
                            .iter()
                            .position(|(eid, _)| *eid == t.tail)
                            .map(|p| 1.0 / (p as f64 + 1.0))
                            .unwrap_or(0.0)
                    })
                    .unwrap_or(0.0)
            })
            .sum();
        sum / test_triples.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Serialise `KgEmbeddings` to a simple CSV-like byte string.
///
/// Format:
/// ```text
/// ENTITIES <n>
/// <dim values per line>
/// RELATIONS <m>
/// <dim values per line>
/// ```
pub fn serialize_embeddings(emb: &KgEmbeddings) -> Vec<u8> {
    let mut out = String::new();
    out.push_str(&format!("ENTITIES {}\n", emb.entity_embeddings.len()));
    for row in &emb.entity_embeddings {
        let line: Vec<String> = row.iter().map(|x| format!("{x:.8}")).collect();
        out.push_str(&line.join(","));
        out.push('\n');
    }
    out.push_str(&format!("RELATIONS {}\n", emb.relation_embeddings.len()));
    for row in &emb.relation_embeddings {
        let line: Vec<String> = row.iter().map(|x| format!("{x:.8}")).collect();
        out.push_str(&line.join(","));
        out.push('\n');
    }
    out.into_bytes()
}

/// Deserialise `KgEmbeddings` from the format produced by
/// [`serialize_embeddings`].
pub fn deserialize_embeddings(data: &[u8]) -> Result<KgEmbeddings, KgError> {
    let text = std::str::from_utf8(data)
        .map_err(|e| KgError::NumericalError(format!("utf8 error: {e}")))?;
    let mut lines = text.lines();

    let parse_section_header = |line: &str, prefix: &str| -> Result<usize, KgError> {
        let rest = line
            .strip_prefix(prefix)
            .ok_or_else(|| KgError::NumericalError(format!("expected '{prefix}', got '{line}'")))?;
        rest.trim()
            .parse::<usize>()
            .map_err(|e| KgError::NumericalError(e.to_string()))
    };

    let parse_row = |line: &str| -> Result<Vec<f64>, KgError> {
        line.split(',')
            .map(|s| {
                s.trim()
                    .parse::<f64>()
                    .map_err(|e| KgError::NumericalError(e.to_string()))
            })
            .collect()
    };

    let ent_header = lines
        .next()
        .ok_or(KgError::NumericalError("empty data".into()))?;
    let num_ent = parse_section_header(ent_header, "ENTITIES ")?;
    let mut entity_embeddings = Vec::with_capacity(num_ent);
    for _ in 0..num_ent {
        let line = lines
            .next()
            .ok_or(KgError::NumericalError("truncated entity data".into()))?;
        entity_embeddings.push(parse_row(line)?);
    }

    let rel_header = lines
        .next()
        .ok_or(KgError::NumericalError("missing RELATIONS header".into()))?;
    let num_rel = parse_section_header(rel_header, "RELATIONS ")?;
    let mut relation_embeddings = Vec::with_capacity(num_rel);
    for _ in 0..num_rel {
        let line = lines
            .next()
            .ok_or(KgError::NumericalError("truncated relation data".into()))?;
        relation_embeddings.push(parse_row(line)?);
    }

    Ok(KgEmbeddings {
        entity_embeddings,
        relation_embeddings,
        entity_to_id: HashMap::new(),
        relation_to_id: HashMap::new(),
    })
}

// ---------------------------------------------------------------------------
// Private sigmoid
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ===========================================================================
// Tests
// ===========================================================================

// ===========================================================================
// Tests (in separate file to keep this file under 2000 lines)
// ===========================================================================

#[cfg(test)]
#[path = "kg_embeddings_tests.rs"]
mod tests;
