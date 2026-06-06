//! Configuration types for enterprise knowledge analyzer.

/// Configuration for enterprise knowledge analysis
#[derive(Debug, Clone)]
pub struct EnterpriseConfig {
    /// Maximum number of products to track
    pub max_products: usize,
    /// Maximum number of employees to track
    pub max_employees: usize,
    /// Maximum number of customers to track
    pub max_customers: usize,
    /// Product recommendation refresh interval (hours)
    pub product_recommendation_refresh_hours: u64,
    /// Employee skill analysis interval (hours)
    pub skill_analysis_interval_hours: u64,
    /// Market analysis interval (hours)
    pub market_analysis_interval_hours: u64,
    /// Enable real-time customer behavior tracking
    pub enable_real_time_customer_tracking: bool,
    /// Minimum interaction threshold for recommendations
    pub min_interaction_threshold: u32,
    /// Embedding dimension
    pub embedding_dimension: usize,
    /// Recommendation system config
    pub recommendation_config: RecommendationConfig,
}

impl Default for EnterpriseConfig {
    fn default() -> Self {
        Self {
            max_products: 500_000,
            max_employees: 50_000,
            max_customers: 1_000_000,
            product_recommendation_refresh_hours: 6,
            skill_analysis_interval_hours: 24,
            market_analysis_interval_hours: 12,
            enable_real_time_customer_tracking: true,
            min_interaction_threshold: 3,
            embedding_dimension: 256,
            recommendation_config: RecommendationConfig::default(),
        }
    }
}

/// Configuration for recommendation systems
#[derive(Debug, Clone)]
pub struct RecommendationConfig {
    /// Number of recommendations to generate
    pub num_recommendations: usize,
    /// Similarity threshold for recommendations
    pub similarity_threshold: f64,
    /// Diversity factor (0.0 = pure similarity, 1.0 = pure diversity)
    pub diversity_factor: f64,
    /// Enable collaborative filtering
    pub enable_collaborative_filtering: bool,
    /// Enable content-based filtering
    pub enable_content_based_filtering: bool,
    /// Enable hybrid recommendations
    pub enable_hybrid: bool,
    /// Cold start strategy
    pub cold_start_strategy: ColdStartStrategy,
}

impl Default for RecommendationConfig {
    fn default() -> Self {
        Self {
            num_recommendations: 10,
            similarity_threshold: 0.3,
            diversity_factor: 0.2,
            enable_collaborative_filtering: true,
            enable_content_based_filtering: true,
            enable_hybrid: true,
            cold_start_strategy: ColdStartStrategy::PopularityBased,
        }
    }
}

/// Cold start strategy for new users/items
#[derive(Debug, Clone)]
pub enum ColdStartStrategy {
    PopularityBased,
    ContentBased,
    DemographicBased,
    RandomSampling,
}
