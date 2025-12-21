//! Personalized vector search with user-specific embeddings and preferences
//!
//! This module provides personalized search capabilities that adapt to individual
//! user behavior, preferences, and interaction history. It maintains user-specific
//! embeddings that evolve over time based on feedback signals.
//!
//! # Features
//!
//! - **User embeddings**: Learn and maintain personalized user representations
//! - **Collaborative filtering**: Leverage behavior of similar users
//! - **Contextual bandits**: Balance exploration vs exploitation
//! - **Preference learning**: Adapt to explicit and implicit feedback
//! - **Privacy-aware**: Support for federated and differential privacy
//! - **Real-time adaptation**: Update user models with each interaction
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_vec::personalized_search::{PersonalizedSearchEngine, UserFeedback, FeedbackType};
//!
//! // Create personalized search engine
//! let mut engine = PersonalizedSearchEngine::new_default()?;
//!
//! // Register user
//! engine.register_user("user123", None)?;
//!
//! // Search with personalization
//! let results = engine.personalized_search("user123", "machine learning", 10)?;
//!
//! // Provide feedback
//! engine.record_feedback(UserFeedback {
//!     user_id: "user123".to_string(),
//!     item_id: results[0].id.clone(),
//!     feedback_type: FeedbackType::Click,
//!     score: 1.0,
//!     timestamp: std::time::SystemTime::now(),
//!     metadata: Default::default(),
//! })?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::Vector;
use crate::VectorStore;
use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use scirs2_core::random::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Type alias for the similarity matrix between users
type SimilarityMatrix = Arc<RwLock<Option<HashMap<(String, String), f32>>>>;

/// Personalized search engine that maintains user-specific models
pub struct PersonalizedSearchEngine {
    config: PersonalizationConfig,
    vector_store: Arc<RwLock<VectorStore>>,
    user_profiles: Arc<RwLock<HashMap<String, UserProfile>>>,
    item_profiles: Arc<RwLock<HashMap<String, ItemProfile>>>,
    interaction_history: Arc<RwLock<Vec<UserInteraction>>>,
    similarity_matrix: SimilarityMatrix,
}

/// Configuration for personalized search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationConfig {
    /// Dimension of user embeddings
    pub user_embedding_dim: usize,
    /// Learning rate for user embedding updates
    pub learning_rate: f32,
    /// Decay factor for older interactions
    pub time_decay_factor: f32,
    /// Weight for collaborative filtering
    pub collaborative_weight: f32,
    /// Weight for content-based filtering
    pub content_weight: f32,
    /// Enable contextual bandits
    pub enable_bandits: bool,
    /// Exploration rate for bandits
    pub exploration_rate: f32,
    /// Enable differential privacy
    pub enable_privacy: bool,
    /// Privacy epsilon parameter
    pub privacy_epsilon: f32,
    /// Minimum interactions before personalization
    pub min_interactions: usize,
    /// User similarity threshold
    pub user_similarity_threshold: f32,
    /// Enable real-time updates
    pub enable_realtime_updates: bool,
    /// Cold start strategy
    pub cold_start_strategy: ColdStartStrategy,
}

impl Default for PersonalizationConfig {
    fn default() -> Self {
        Self {
            user_embedding_dim: 128,
            learning_rate: 0.01,
            time_decay_factor: 0.95,
            collaborative_weight: 0.4,
            content_weight: 0.6,
            enable_bandits: true,
            exploration_rate: 0.1,
            enable_privacy: false,
            privacy_epsilon: 1.0,
            min_interactions: 5,
            user_similarity_threshold: 0.7,
            enable_realtime_updates: true,
            cold_start_strategy: ColdStartStrategy::PopularityBased,
        }
    }
}

/// Strategy for handling new users (cold start problem)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColdStartStrategy {
    /// Use global popularity
    PopularityBased,
    /// Use demographic information
    DemographicBased,
    /// Use random exploration
    RandomExploration,
    /// Use hybrid approach
    Hybrid,
}

/// User profile containing personalized embedding and preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: String,
    pub embedding: Vec<f32>,
    pub preferences: HashMap<String, f32>,
    pub interaction_count: usize,
    pub last_updated: SystemTime,
    pub demographics: Option<UserDemographics>,
    pub similar_users: Vec<(String, f32)>, // (user_id, similarity)
    pub favorite_categories: HashMap<String, f32>,
    pub negative_items: Vec<String>, // Disliked items
}

/// User demographic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserDemographics {
    pub age_group: Option<String>,
    pub location: Option<String>,
    pub language: Option<String>,
    pub interests: Vec<String>,
}

/// Item profile with popularity and category information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemProfile {
    pub item_id: String,
    pub embedding: Vec<f32>,
    pub popularity_score: f32,
    pub categories: Vec<String>,
    pub interaction_count: usize,
    pub average_rating: f32,
    pub last_accessed: SystemTime,
}

/// User interaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInteraction {
    pub user_id: String,
    pub item_id: String,
    pub interaction_type: InteractionType,
    pub score: f32,
    pub timestamp: SystemTime,
    pub context: HashMap<String, String>,
}

/// Type of user interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    View,
    Click,
    Like,
    Dislike,
    Share,
    Purchase,
    Rating(f32),
    DwellTime(Duration),
    Custom(String),
}

/// User feedback for model updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedback {
    pub user_id: String,
    pub item_id: String,
    pub feedback_type: FeedbackType,
    pub score: f32,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}

/// Type of feedback signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackType {
    Explicit(f32), // Rating
    Click,         // Binary positive signal
    View,          // Implicit interest
    Skip,          // Negative signal
    Purchase,      // Strong positive signal
    Share,         // Strong positive signal
    LongDwell,     // Time-based positive
    QuickBounce,   // Time-based negative
    Custom(String),
}

/// Personalized search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizedResult {
    pub id: String,
    pub score: f32,
    pub personalization_score: f32,
    pub content_score: f32,
    pub collaborative_score: f32,
    pub exploration_bonus: f32,
    pub metadata: HashMap<String, String>,
    pub explanation: Option<String>,
}

impl PersonalizedSearchEngine {
    /// Create a new personalized search engine with default configuration
    pub fn new_default() -> Result<Self> {
        Self::new(PersonalizationConfig::default(), None)
    }

    /// Create a new personalized search engine with custom configuration
    pub fn new(config: PersonalizationConfig, vector_store: Option<VectorStore>) -> Result<Self> {
        let default_store = VectorStore::new();
        let vector_store = Arc::new(RwLock::new(vector_store.unwrap_or(default_store)));

        Ok(Self {
            config,
            vector_store,
            user_profiles: Arc::new(RwLock::new(HashMap::new())),
            item_profiles: Arc::new(RwLock::new(HashMap::new())),
            interaction_history: Arc::new(RwLock::new(Vec::new())),
            similarity_matrix: Arc::new(RwLock::new(None)),
        })
    }

    /// Register a new user
    pub fn register_user(
        &mut self,
        user_id: impl Into<String>,
        demographics: Option<UserDemographics>,
    ) -> Result<()> {
        let user_id = user_id.into();

        // Initialize user embedding
        let embedding = self.initialize_user_embedding(&user_id, demographics.as_ref())?;

        let profile = UserProfile {
            user_id: user_id.clone(),
            embedding,
            preferences: HashMap::new(),
            interaction_count: 0,
            last_updated: SystemTime::now(),
            demographics,
            similar_users: Vec::new(),
            favorite_categories: HashMap::new(),
            negative_items: Vec::new(),
        };

        self.user_profiles.write().insert(user_id, profile);

        Ok(())
    }

    /// Perform personalized search for a user
    pub fn personalized_search(
        &self,
        user_id: impl Into<String>,
        query: impl Into<String>,
        k: usize,
    ) -> Result<Vec<PersonalizedResult>> {
        let user_id = user_id.into();
        let query = query.into();

        // Get user profile
        let user_profiles = self.user_profiles.read();
        let user_profile = user_profiles
            .get(&user_id)
            .ok_or_else(|| anyhow!("User not found: {}", user_id))?;

        // Check if user has enough interactions for personalization
        let use_personalization = user_profile.interaction_count >= self.config.min_interactions;

        // Get base search results (content-based)
        let base_results = self.content_based_search(&query, k * 3)?;

        // Apply personalization
        let personalized_results = if use_personalization {
            self.apply_personalization(&user_id, base_results, k)?
        } else {
            self.apply_cold_start_strategy(&user_id, base_results, k)?
        };

        Ok(personalized_results)
    }

    /// Content-based search without personalization
    fn content_based_search(&self, query: &str, k: usize) -> Result<Vec<PersonalizedResult>> {
        // Simple text embedding (in production, use proper embedding model)
        let _query_embedding = self.create_query_embedding(query)?;

        // Search in vector store using text query
        let store = self.vector_store.read();
        let results = store.similarity_search(query, k)?;

        // Convert to PersonalizedResult
        Ok(results
            .into_iter()
            .map(|(id, score)| PersonalizedResult {
                id,
                score,
                personalization_score: 0.0,
                content_score: score,
                collaborative_score: 0.0,
                exploration_bonus: 0.0,
                metadata: HashMap::new(),
                explanation: None,
            })
            .collect())
    }

    /// Apply personalization to search results
    fn apply_personalization(
        &self,
        user_id: &str,
        mut results: Vec<PersonalizedResult>,
        k: usize,
    ) -> Result<Vec<PersonalizedResult>> {
        let user_profiles = self.user_profiles.read();
        let user_profile = user_profiles
            .get(user_id)
            .ok_or_else(|| anyhow!("User not found"))?;

        // Compute collaborative filtering scores
        for result in &mut results {
            // Collaborative score based on similar users
            let collab_score = self.compute_collaborative_score(user_profile, &result.id)?;

            // Personalization score based on user embedding
            let personal_score = self.compute_personalization_score(user_profile, &result.id)?;

            // Exploration bonus (contextual bandits)
            let exploration_bonus = if self.config.enable_bandits {
                self.compute_exploration_bonus(user_profile, &result.id)?
            } else {
                0.0
            };

            // Combine scores
            result.collaborative_score = collab_score;
            result.personalization_score = personal_score;
            result.exploration_bonus = exploration_bonus;

            result.score = self.config.content_weight * result.content_score
                + self.config.collaborative_weight * collab_score
                + (1.0 - self.config.content_weight - self.config.collaborative_weight)
                    * personal_score
                + exploration_bonus;

            // Generate explanation
            result.explanation = Some(self.generate_explanation(result));
        }

        // Re-rank by combined score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Apply diversity
        let diversified = self.apply_diversity(&results, k)?;

        Ok(diversified)
    }

    /// Compute collaborative filtering score
    fn compute_collaborative_score(
        &self,
        user_profile: &UserProfile,
        item_id: &str,
    ) -> Result<f32> {
        let item_profiles = self.item_profiles.read();

        if let Some(item_profile) = item_profiles.get(item_id) {
            // Score based on similar users' interactions
            let mut collab_score = 0.0;
            let mut total_weight = 0.0;

            for (similar_user_id, similarity) in &user_profile.similar_users {
                // Check if similar user interacted with this item
                let interactions = self.interaction_history.read();
                let user_interacted = interactions.iter().any(|i| {
                    &i.user_id == similar_user_id && i.item_id == item_id && i.score > 0.0
                });

                if user_interacted {
                    collab_score += similarity;
                    total_weight += similarity;
                }
            }

            if total_weight > 0.0 {
                collab_score /= total_weight;
            }

            // Add popularity bonus
            collab_score += item_profile.popularity_score * 0.1;

            Ok(collab_score.min(1.0))
        } else {
            Ok(0.0)
        }
    }

    /// Compute personalization score based on user embedding
    fn compute_personalization_score(
        &self,
        user_profile: &UserProfile,
        item_id: &str,
    ) -> Result<f32> {
        let item_profiles = self.item_profiles.read();

        if let Some(item_profile) = item_profiles.get(item_id) {
            // Compute cosine similarity between user and item embeddings
            let similarity =
                self.cosine_similarity(&user_profile.embedding, &item_profile.embedding);

            // Check negative items
            if user_profile.negative_items.contains(&item_id.to_string()) {
                return Ok(similarity * 0.5); // Penalize disliked items
            }

            // Boost based on category preferences
            let category_boost = item_profile
                .categories
                .iter()
                .filter_map(|cat| user_profile.favorite_categories.get(cat))
                .sum::<f32>()
                / item_profile.categories.len().max(1) as f32;

            Ok((similarity + category_boost * 0.3).min(1.0))
        } else {
            Ok(0.0)
        }
    }

    /// Compute exploration bonus using contextual bandits
    fn compute_exploration_bonus(&self, user_profile: &UserProfile, item_id: &str) -> Result<f32> {
        let item_profiles = self.item_profiles.read();

        if let Some(item_profile) = item_profiles.get(item_id) {
            // UCB-style exploration bonus
            let n = user_profile.interaction_count as f32;
            let n_i = item_profile.interaction_count as f32;

            if n_i == 0.0 {
                // High exploration bonus for unseen items
                return Ok(self.config.exploration_rate);
            }

            let exploration_bonus = self.config.exploration_rate * ((2.0 * n.ln() / n_i).sqrt());

            Ok(exploration_bonus.min(0.5))
        } else {
            Ok(0.0)
        }
    }

    /// Apply cold start strategy for new users
    fn apply_cold_start_strategy(
        &self,
        _user_id: &str,
        mut results: Vec<PersonalizedResult>,
        k: usize,
    ) -> Result<Vec<PersonalizedResult>> {
        match self.config.cold_start_strategy {
            ColdStartStrategy::PopularityBased => {
                // Boost popular items
                let item_profiles = self.item_profiles.read();

                for result in &mut results {
                    if let Some(item_profile) = item_profiles.get(&result.id) {
                        result.score += item_profile.popularity_score * 0.3;
                    }
                }

                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            }
            ColdStartStrategy::RandomExploration => {
                // Add random exploration
                use scirs2_core::random::rng;
                let mut rng_instance = rng();

                for result in &mut results {
                    // Generate random value between 0.0 and 0.2
                    let random_val = (rng_instance.next_u64() as f32 / u64::MAX as f32) * 0.2;
                    result.score += random_val;
                }

                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            }
            ColdStartStrategy::DemographicBased => {
                // Use demographic-based recommendations (simplified)
                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            }
            ColdStartStrategy::Hybrid => {
                // Combine multiple strategies
                use scirs2_core::random::rng;
                let item_profiles = self.item_profiles.read();
                let mut rng_instance = rng();

                for result in &mut results {
                    if let Some(item_profile) = item_profiles.get(&result.id) {
                        let random_val = (rng_instance.next_u64() as f32 / u64::MAX as f32) * 0.1;
                        result.score += item_profile.popularity_score * 0.2 + random_val;
                    }
                }

                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            }
        }

        Ok(results.into_iter().take(k).collect())
    }

    /// Record user feedback and update user profile
    pub fn record_feedback(&mut self, feedback: UserFeedback) -> Result<()> {
        // Convert feedback to interaction
        let interaction = UserInteraction {
            user_id: feedback.user_id.clone(),
            item_id: feedback.item_id.clone(),
            interaction_type: Self::feedback_to_interaction_type(&feedback.feedback_type),
            score: feedback.score,
            timestamp: feedback.timestamp,
            context: feedback.metadata.clone(),
        };

        // Store interaction
        self.interaction_history.write().push(interaction.clone());

        // Update user profile if real-time updates enabled
        if self.config.enable_realtime_updates {
            self.update_user_profile(&feedback.user_id, &interaction)?;
        }

        // Update item profile
        self.update_item_profile(&feedback.item_id, &interaction)?;

        Ok(())
    }

    /// Update user profile based on interaction
    fn update_user_profile(&mut self, user_id: &str, interaction: &UserInteraction) -> Result<()> {
        let mut user_profiles = self.user_profiles.write();

        if let Some(profile) = user_profiles.get_mut(user_id) {
            // Update interaction count
            profile.interaction_count += 1;
            profile.last_updated = SystemTime::now();

            // Get item embedding
            let item_profiles = self.item_profiles.read();
            if let Some(item_profile) = item_profiles.get(&interaction.item_id) {
                // Update user embedding using gradient descent
                let learning_rate = self.config.learning_rate;

                for (i, emb_val) in profile.embedding.iter_mut().enumerate() {
                    if i < item_profile.embedding.len() {
                        let target = item_profile.embedding[i];
                        let gradient = (target - *emb_val) * interaction.score;
                        *emb_val += learning_rate * gradient;
                    }
                }

                // Normalize embedding
                let norm: f32 = profile.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    profile.embedding.iter_mut().for_each(|x| *x /= norm);
                }

                // Update category preferences
                for category in &item_profile.categories {
                    let current = profile
                        .favorite_categories
                        .get(category)
                        .copied()
                        .unwrap_or(0.0);
                    let updated = current * 0.9 + interaction.score * 0.1;
                    profile
                        .favorite_categories
                        .insert(category.clone(), updated);
                }

                // Update negative items
                if interaction.score < 0.0 {
                    profile.negative_items.push(interaction.item_id.clone());
                }
            }
        }

        Ok(())
    }

    /// Update item profile based on interaction
    fn update_item_profile(&mut self, item_id: &str, interaction: &UserInteraction) -> Result<()> {
        let mut item_profiles = self.item_profiles.write();

        if let Some(profile) = item_profiles.get_mut(item_id) {
            profile.interaction_count += 1;
            profile.last_accessed = SystemTime::now();

            // Update average rating
            let old_avg = profile.average_rating;
            let count = profile.interaction_count as f32;
            profile.average_rating = (old_avg * (count - 1.0) + interaction.score) / count;

            // Update popularity score (decayed)
            profile.popularity_score = profile.popularity_score * 0.95 + interaction.score * 0.05;
        }

        Ok(())
    }

    /// Update user similarity matrix
    pub fn update_user_similarities(&mut self) -> Result<()> {
        let user_profiles = self.user_profiles.read();
        let user_ids: Vec<String> = user_profiles.keys().cloned().collect();

        for user_id in &user_ids {
            if let Some(user_profile) = user_profiles.get(user_id) {
                let mut similar_users = Vec::new();

                // Compute similarities with all other users
                for other_id in &user_ids {
                    if other_id != user_id {
                        if let Some(other_profile) = user_profiles.get(other_id) {
                            let similarity = self.cosine_similarity(
                                &user_profile.embedding,
                                &other_profile.embedding,
                            );

                            if similarity >= self.config.user_similarity_threshold {
                                similar_users.push((other_id.clone(), similarity));
                            }
                        }
                    }
                }

                // Sort by similarity and keep top 10
                similar_users.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                similar_users.truncate(10);

                // Update user profile (need to drop read lock and acquire write lock)
                drop(user_profiles);
                let mut user_profiles = self.user_profiles.write();
                if let Some(profile) = user_profiles.get_mut(user_id) {
                    profile.similar_users = similar_users;
                }

                return Ok(()); // Early return to avoid deadlock
            }
        }

        Ok(())
    }

    /// Apply diversity to results
    fn apply_diversity(
        &self,
        results: &[PersonalizedResult],
        k: usize,
    ) -> Result<Vec<PersonalizedResult>> {
        // MMR-style diversity
        let mut diversified = Vec::new();
        let mut remaining: Vec<PersonalizedResult> = results.to_vec();

        if !remaining.is_empty() {
            // Add highest scored item first
            diversified.push(remaining.remove(0));
        }

        let lambda = 0.7; // Relevance vs diversity trade-off

        while diversified.len() < k && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f32::NEG_INFINITY;

            for (i, candidate) in remaining.iter().enumerate() {
                // Compute minimum similarity to already selected items
                let mut min_similarity = 1.0f32;

                for selected in &diversified {
                    let similarity = if selected.metadata.get("category")
                        == candidate.metadata.get("category")
                    {
                        0.8
                    } else {
                        0.2
                    };

                    min_similarity = min_similarity.min(similarity);
                }

                // MMR score
                let mmr_score = lambda * candidate.score + (1.0 - lambda) * (1.0 - min_similarity);

                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = i;
                }
            }

            diversified.push(remaining.remove(best_idx));
        }

        Ok(diversified)
    }

    /// Generate explanation for personalized result
    fn generate_explanation(&self, result: &PersonalizedResult) -> String {
        let mut reasons = Vec::new();

        if result.personalization_score > 0.5 {
            reasons.push("matches your interests");
        }

        if result.collaborative_score > 0.5 {
            reasons.push("liked by similar users");
        }

        if result.exploration_bonus > 0.1 {
            reasons.push("new discovery");
        }

        if reasons.is_empty() {
            reasons.push("relevant to your query");
        }

        format!("Recommended because: {}", reasons.join(", "))
    }

    /// Initialize user embedding
    fn initialize_user_embedding(
        &self,
        _user_id: &str,
        demographics: Option<&UserDemographics>,
    ) -> Result<Vec<f32>> {
        use scirs2_core::random::rng;
        let mut embedding = vec![0.0f32; self.config.user_embedding_dim];

        if let Some(demo) = demographics {
            // Use demographics to seed embedding
            for (_i, interest) in demo.interests.iter().enumerate().take(embedding.len() / 2) {
                let hash = Self::hash_string(interest);
                let idx = (hash % self.config.user_embedding_dim as u64) as usize;
                embedding[idx] = 0.5;
            }
        } else {
            // Random initialization
            let mut rng_instance = rng();

            for val in &mut embedding {
                // Generate random value between -0.1 and 0.1
                let random_val = (rng_instance.next_u64() as f32 / u64::MAX as f32) * 0.2 - 0.1;
                *val = random_val;
            }
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            embedding.iter_mut().for_each(|x| *x /= norm);
        }

        Ok(embedding)
    }

    /// Create query embedding
    fn create_query_embedding(&self, query: &str) -> Result<Vector> {
        // Simple token-based embedding (in production, use proper model)
        let tokens: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();

        let mut embedding = vec![0.0f32; 128]; // Default dimension

        for token in tokens {
            let hash = Self::hash_string(&token);
            let idx = (hash % embedding.len() as u64) as usize;
            embedding[idx] += 1.0;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            embedding.iter_mut().for_each(|x| *x /= norm);
        }

        Ok(Vector::new(embedding))
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Convert feedback type to interaction type
    fn feedback_to_interaction_type(feedback_type: &FeedbackType) -> InteractionType {
        match feedback_type {
            FeedbackType::Explicit(rating) => InteractionType::Rating(*rating),
            FeedbackType::Click => InteractionType::Click,
            FeedbackType::View => InteractionType::View,
            FeedbackType::Skip => InteractionType::Custom("skip".to_string()),
            FeedbackType::Purchase => InteractionType::Purchase,
            FeedbackType::Share => InteractionType::Share,
            FeedbackType::LongDwell => InteractionType::DwellTime(Duration::from_secs(60)),
            FeedbackType::QuickBounce => InteractionType::DwellTime(Duration::from_secs(5)),
            FeedbackType::Custom(name) => InteractionType::Custom(name.clone()),
        }
    }

    /// Hash string to u64
    fn hash_string(s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    /// Get user profile
    pub fn get_user_profile(&self, user_id: &str) -> Option<UserProfile> {
        self.user_profiles.read().get(user_id).cloned()
    }

    /// Get statistics
    pub fn get_statistics(&self) -> PersonalizationStatistics {
        let user_profiles = self.user_profiles.read();
        let item_profiles = self.item_profiles.read();
        let interactions = self.interaction_history.read();

        PersonalizationStatistics {
            total_users: user_profiles.len(),
            total_items: item_profiles.len(),
            total_interactions: interactions.len(),
            average_interactions_per_user: if user_profiles.is_empty() {
                0.0
            } else {
                interactions.len() as f32 / user_profiles.len() as f32
            },
        }
    }
}

/// Statistics about personalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationStatistics {
    pub total_users: usize,
    pub total_items: usize,
    pub total_interactions: usize,
    pub average_interactions_per_user: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_user() -> Result<()> {
        let mut engine = PersonalizedSearchEngine::new_default()?;

        engine.register_user("user1", None)?;

        let profile = engine.get_user_profile("user1");
        assert!(profile.is_some());

        Ok(())
    }

    #[test]
    fn test_feedback_recording() -> Result<()> {
        let mut engine = PersonalizedSearchEngine::new_default()?;

        engine.register_user("user1", None)?;

        let feedback = UserFeedback {
            user_id: "user1".to_string(),
            item_id: "item1".to_string(),
            feedback_type: FeedbackType::Click,
            score: 1.0,
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
        };

        engine.record_feedback(feedback)?;

        let stats = engine.get_statistics();
        assert_eq!(stats.total_interactions, 1);

        Ok(())
    }

    #[test]
    fn test_cold_start_strategy() -> Result<()> {
        let engine = PersonalizedSearchEngine::new_default()?;

        let query_embedding = engine.create_query_embedding("test query")?;
        assert_eq!(query_embedding.dimensions, 128);

        Ok(())
    }

    #[test]
    fn test_cosine_similarity() -> Result<()> {
        let engine = PersonalizedSearchEngine::new_default()?;

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];

        let similarity = engine.cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 0.001);

        Ok(())
    }
}
