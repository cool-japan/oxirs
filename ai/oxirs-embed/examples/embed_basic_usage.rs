//! Basic usage example for oxirs-embed
//!
//! This example demonstrates the core functionality of the embedding system.

use oxirs_embed::{EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Triple, Vector};
use std::collections::HashMap;
use uuid::Uuid;

// Simple TransE implementation for demonstration
pub struct SimpleTransE {
    model_id: Uuid,
    config: ModelConfig,
    entity_to_id: HashMap<String, usize>,
    relation_to_id: HashMap<String, usize>,
    triples: Vec<(usize, usize, usize)>,
    entity_embeddings: Vec<Vec<f64>>,
    relation_embeddings: Vec<Vec<f64>>,
    is_trained: bool,
}

impl SimpleTransE {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            model_id: Uuid::new_v4(),
            config,
            entity_to_id: HashMap::new(),
            relation_to_id: HashMap::new(),
            triples: Vec::new(),
            entity_embeddings: Vec::new(),
            relation_embeddings: Vec::new(),
            is_trained: false,
        }
    }

    fn get_or_create_entity_id(&mut self, entity: String) -> usize {
        if let Some(&id) = self.entity_to_id.get(&entity) {
            id
        } else {
            let id = self.entity_embeddings.len();
            self.entity_to_id.insert(entity, id);

            // Initialize random embedding
            let embedding: Vec<f64> = (0..self.config.dimensions)
                .map(|_| (rand::random::<f64>() - 0.5) * 0.1)
                .collect();
            self.entity_embeddings.push(embedding);

            id
        }
    }

    fn get_or_create_relation_id(&mut self, relation: String) -> usize {
        if let Some(&id) = self.relation_to_id.get(&relation) {
            id
        } else {
            let id = self.relation_embeddings.len();
            self.relation_to_id.insert(relation, id);

            // Initialize random embedding
            let embedding: Vec<f64> = (0..self.config.dimensions)
                .map(|_| (rand::random::<f64>() - 0.5) * 0.1)
                .collect();
            self.relation_embeddings.push(embedding);

            id
        }
    }

    fn score_triple_ids(&self, subject_id: usize, predicate_id: usize, object_id: usize) -> f64 {
        if subject_id >= self.entity_embeddings.len()
            || object_id >= self.entity_embeddings.len()
            || predicate_id >= self.relation_embeddings.len()
        {
            return f64::NEG_INFINITY;
        }

        let h = &self.entity_embeddings[subject_id];
        let r = &self.relation_embeddings[predicate_id];
        let t = &self.entity_embeddings[object_id];

        // TransE score: -||h + r - t||
        let mut distance = 0.0;
        for i in 0..self.config.dimensions {
            let diff = h[i] + r[i] - t[i];
            distance += diff * diff;
        }

        -distance.sqrt()
    }

    fn train_epoch(&mut self) -> f64 {
        if self.triples.is_empty() {
            return 0.0;
        }

        let mut epoch_loss = 0.0;
        let mut updates = 0;

        for &(s, p, o) in &self.triples {
            // Generate negative sample by corrupting object
            let neg_o = rand::random::<usize>() % self.entity_embeddings.len();

            if neg_o != o {
                // Make sure it's actually negative
                let pos_score = self.score_triple_ids(s, p, o);
                let neg_score = self.score_triple_ids(s, p, neg_o);

                // Margin loss: max(0, margin + neg_score - pos_score)
                let margin = 1.0;
                let loss = (margin + neg_score - pos_score).max(0.0);
                epoch_loss += loss;

                if loss > 0.0 {
                    // Simple gradient update (simplified)
                    let update_magnitude = self.config.learning_rate * 0.1;

                    // Update embeddings slightly
                    for i in 0..self.config.dimensions {
                        // Positive triple: decrease distance
                        let h_val = self.entity_embeddings[s][i];
                        let r_val = self.relation_embeddings[p][i];
                        let t_val = self.entity_embeddings[o][i];
                        let diff = h_val + r_val - t_val;

                        if diff > 0.0 {
                            self.entity_embeddings[s][i] -= update_magnitude;
                            self.relation_embeddings[p][i] -= update_magnitude;
                            self.entity_embeddings[o][i] += update_magnitude;
                        } else {
                            self.entity_embeddings[s][i] += update_magnitude;
                            self.relation_embeddings[p][i] += update_magnitude;
                            self.entity_embeddings[o][i] -= update_magnitude;
                        }
                    }
                    updates += 1;
                }
            }
        }

        if updates > 0 {
            epoch_loss / updates as f64
        } else {
            0.0
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for SimpleTransE {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn model_id(&self) -> &Uuid {
        &self.model_id
    }

    fn model_type(&self) -> &'static str {
        "SimpleTransE"
    }

    fn add_triple(&mut self, triple: Triple) -> anyhow::Result<()> {
        let subject_id = self.get_or_create_entity_id(triple.subject.iri);
        let predicate_id = self.get_or_create_relation_id(triple.predicate.iri);
        let object_id = self.get_or_create_entity_id(triple.object.iri);

        self.triples.push((subject_id, predicate_id, object_id));
        Ok(())
    }

    async fn train(&mut self, epochs: Option<usize>) -> anyhow::Result<TrainingStats> {
        let start_time = std::time::Instant::now();
        let epochs = epochs.unwrap_or(self.config.max_epochs);

        if self.triples.is_empty() {
            return Err(anyhow::anyhow!("No training data available"));
        }

        let mut loss_history = Vec::new();

        for epoch in 0..epochs {
            let epoch_loss = self.train_epoch();
            loss_history.push(epoch_loss);

            if epoch % 10 == 0 {
                println!("Epoch {}: loss = {:.6}", epoch, epoch_loss);
            }

            if epoch > 10 && epoch_loss < 1e-6 {
                break;
            }
        }

        self.is_trained = true;
        let training_time = start_time.elapsed().as_secs_f64();

        Ok(TrainingStats {
            epochs_completed: loss_history.len(),
            final_loss: loss_history.last().copied().unwrap_or(0.0),
            training_time_seconds: training_time,
            convergence_achieved: loss_history.last().copied().unwrap_or(f64::INFINITY) < 1e-6,
            loss_history,
        })
    }

    fn get_entity_embedding(&self, entity: &str) -> anyhow::Result<Vector> {
        if let Some(&id) = self.entity_to_id.get(entity) {
            let embedding = &self.entity_embeddings[id];
            let values: Vec<f32> = embedding.iter().map(|&x| x as f32).collect();
            Ok(Vector::new(values))
        } else {
            Err(anyhow::anyhow!("Entity not found: {}", entity))
        }
    }

    fn get_relation_embedding(&self, relation: &str) -> anyhow::Result<Vector> {
        if let Some(&id) = self.relation_to_id.get(relation) {
            let embedding = &self.relation_embeddings[id];
            let values: Vec<f32> = embedding.iter().map(|&x| x as f32).collect();
            Ok(Vector::new(values))
        } else {
            Err(anyhow::anyhow!("Relation not found: {}", relation))
        }
    }

    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> anyhow::Result<f64> {
        let subject_id = self
            .entity_to_id
            .get(subject)
            .ok_or_else(|| anyhow::anyhow!("Subject not found: {}", subject))?;
        let predicate_id = self
            .relation_to_id
            .get(predicate)
            .ok_or_else(|| anyhow::anyhow!("Predicate not found: {}", predicate))?;
        let object_id = self
            .entity_to_id
            .get(object)
            .ok_or_else(|| anyhow::anyhow!("Object not found: {}", object))?;

        Ok(self.score_triple_ids(*subject_id, *predicate_id, *object_id))
    }

    fn get_entities(&self) -> Vec<String> {
        self.entity_to_id.keys().cloned().collect()
    }

    fn get_relations(&self) -> Vec<String> {
        self.relation_to_id.keys().cloned().collect()
    }

    fn clear(&mut self) {
        self.entity_to_id.clear();
        self.relation_to_id.clear();
        self.triples.clear();
        self.entity_embeddings.clear();
        self.relation_embeddings.clear();
        self.is_trained = false;
    }

    fn is_trained(&self) -> bool {
        self.is_trained
    }

    fn predict_objects(
        &self,
        subject: &str,
        predicate: &str,
        k: usize,
    ) -> anyhow::Result<Vec<(String, f64)>> {
        let subject_id = self
            .entity_to_id
            .get(subject)
            .ok_or_else(|| anyhow::anyhow!("Subject not found: {}", subject))?;
        let predicate_id = self
            .relation_to_id
            .get(predicate)
            .ok_or_else(|| anyhow::anyhow!("Predicate not found: {}", predicate))?;

        let mut scores = Vec::new();
        for (entity, &entity_id) in &self.entity_to_id {
            let score = self.score_triple_ids(*subject_id, *predicate_id, entity_id);
            scores.push((entity.clone(), score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        Ok(scores)
    }

    fn predict_subjects(
        &self,
        predicate: &str,
        object: &str,
        k: usize,
    ) -> anyhow::Result<Vec<(String, f64)>> {
        let predicate_id = self
            .relation_to_id
            .get(predicate)
            .ok_or_else(|| anyhow::anyhow!("Predicate not found: {}", predicate))?;
        let object_id = self
            .entity_to_id
            .get(object)
            .ok_or_else(|| anyhow::anyhow!("Object not found: {}", object))?;

        let mut scores = Vec::new();
        for (entity, &entity_id) in &self.entity_to_id {
            let score = self.score_triple_ids(entity_id, *predicate_id, *object_id);
            scores.push((entity.clone(), score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        Ok(scores)
    }

    fn predict_relations(
        &self,
        subject: &str,
        object: &str,
        k: usize,
    ) -> anyhow::Result<Vec<(String, f64)>> {
        let subject_id = self
            .entity_to_id
            .get(subject)
            .ok_or_else(|| anyhow::anyhow!("Subject not found: {}", subject))?;
        let object_id = self
            .entity_to_id
            .get(object)
            .ok_or_else(|| anyhow::anyhow!("Object not found: {}", object))?;

        let mut scores = Vec::new();
        for (relation, &relation_id) in &self.relation_to_id {
            let score = self.score_triple_ids(*subject_id, relation_id, *object_id);
            scores.push((relation.clone(), score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        Ok(scores)
    }

    fn get_stats(&self) -> ModelStats {
        ModelStats {
            num_entities: self.entity_to_id.len(),
            num_relations: self.relation_to_id.len(),
            num_triples: self.triples.len(),
            dimensions: self.config.dimensions,
            is_trained: self.is_trained,
            model_type: "SimpleTransE".to_string(),
            creation_time: chrono::Utc::now(),
            last_training_time: if self.is_trained {
                Some(chrono::Utc::now())
            } else {
                None
            },
        }
    }

    fn save(&self, _path: &str) -> anyhow::Result<()> {
        // Simplified save implementation
        Ok(())
    }

    fn load(&mut self, _path: &str) -> anyhow::Result<()> {
        // Simplified load implementation
        Ok(())
    }

    async fn encode(&self, _texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Err(anyhow::anyhow!("Knowledge graph embedding model does not support text encoding"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 OxiRS Embed - Basic Usage Example");
    println!("=====================================");

    // Create a simple TransE model
    let config = ModelConfig {
        dimensions: 50,
        learning_rate: 0.01,
        max_epochs: 50,
        ..Default::default()
    };
    let mut model = SimpleTransE::new(config);

    println!("\n📊 Adding training data...");

    // Add some simple triples
    model.add_triple(Triple::new(
        oxirs_embed::NamedNode {
            iri: "alice".to_string(),
        },
        oxirs_embed::NamedNode {
            iri: "knows".to_string(),
        },
        oxirs_embed::NamedNode {
            iri: "bob".to_string(),
        },
    ))?;

    model.add_triple(Triple::new(
        oxirs_embed::NamedNode {
            iri: "bob".to_string(),
        },
        oxirs_embed::NamedNode {
            iri: "knows".to_string(),
        },
        oxirs_embed::NamedNode {
            iri: "charlie".to_string(),
        },
    ))?;

    model.add_triple(Triple::new(
        oxirs_embed::NamedNode {
            iri: "alice".to_string(),
        },
        oxirs_embed::NamedNode {
            iri: "likes".to_string(),
        },
        oxirs_embed::NamedNode {
            iri: "charlie".to_string(),
        },
    ))?;

    model.add_triple(Triple::new(
        oxirs_embed::NamedNode {
            iri: "charlie".to_string(),
        },
        oxirs_embed::NamedNode {
            iri: "works_at".to_string(),
        },
        oxirs_embed::NamedNode {
            iri: "company".to_string(),
        },
    ))?;

    println!("✅ Added {} triples", model.triples.len());
    println!("✅ Found {} entities", model.entity_embeddings.len());
    println!("✅ Found {} relations", model.relation_embeddings.len());

    println!("\n🏋️ Training model...");
    let stats = model.train(Some(30)).await?;
    println!("✅ Training completed. Final loss: {:.6}", stats.final_loss);
    println!(
        "✅ Training time: {:.2} seconds",
        stats.training_time_seconds
    );
    println!("✅ Epochs completed: {}", stats.epochs_completed);

    println!("\n🧮 Testing embeddings...");

    // Test entity embeddings
    let alice_emb = model.get_entity_embedding("alice")?;
    let bob_emb = model.get_entity_embedding("bob")?;
    println!("✅ Alice embedding: {} dimensions", alice_emb.values.len());
    println!("✅ Bob embedding: {} dimensions", bob_emb.values.len());

    // Test relation embeddings
    let knows_emb = model.get_relation_embedding("knows")?;
    let likes_emb = model.get_relation_embedding("likes")?;
    println!(
        "✅ 'knows' embedding: {} dimensions",
        knows_emb.values.len()
    );
    println!(
        "✅ 'likes' embedding: {} dimensions",
        likes_emb.values.len()
    );

    println!("\n🎯 Testing triple scoring...");

    // Test scoring
    let score1 = model.score_triple("alice", "knows", "bob")?;
    let score2 = model.score_triple("alice", "knows", "charlie")?;
    let score3 = model.score_triple("bob", "likes", "alice")?;

    println!("📈 Score(alice, knows, bob): {:.4}", score1);
    println!("📈 Score(alice, knows, charlie): {:.4}", score2);
    println!("📈 Score(bob, likes, alice): {:.4}", score3);

    // Similarity test
    let similarity = alice_emb
        .values
        .iter()
        .zip(&bob_emb.values)
        .map(|(a, b)| a * b)
        .sum::<f32>();
    println!("🔗 Alice-Bob similarity: {:.4}", similarity);

    println!("\n🎉 Example completed successfully!");
    println!(
        "🚀 OxiRS Embed is working with {} dimensions",
        alice_emb.values.len()
    );

    Ok(())
}
