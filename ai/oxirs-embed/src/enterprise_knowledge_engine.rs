//! Recommendation engine, market analysis, and enterprise metrics types.

use crate::enterprise_knowledge_product::CategoryPerformance;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Recommendation engine
#[derive(Debug, Clone)]
pub struct RecommendationEngine {
    /// Engine type
    pub engine_type: RecommendationEngineType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Performance metrics
    pub performance: RecommendationPerformance,
    /// Last update
    pub last_update: DateTime<Utc>,
}

/// Types of recommendation engines
#[derive(Debug, Clone)]
pub enum RecommendationEngineType {
    CollaborativeFiltering,
    ContentBased,
    MatrixFactorization,
    DeepLearning,
    Hybrid,
}

/// Recommendation engine performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationPerformance {
    /// Precision at K
    pub precision_at_k: HashMap<u32, f64>,
    /// Recall at K
    pub recall_at_k: HashMap<u32, f64>,
    /// NDCG scores
    pub ndcg_scores: HashMap<u32, f64>,
    /// Click-through rate
    pub click_through_rate: f64,
    /// Conversion rate
    pub conversion_rate: f64,
    /// Revenue impact
    pub revenue_impact: f64,
}

/// Market analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketAnalysis {
    /// Performance by category
    pub category_performance: HashMap<String, CategoryPerformance>,
    /// Trending products
    pub trending_products: Vec<String>,
    /// Customer segment distribution
    pub segment_distribution: HashMap<String, u32>,
    /// Market opportunities
    pub market_opportunities: Vec<String>,
    /// Competitive landscape
    pub competitive_landscape: HashMap<String, f64>,
    /// Market forecast
    pub forecast: HashMap<String, f64>,
}

/// Enterprise analytics and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnterpriseMetrics {
    /// Total products
    pub total_products: usize,
    /// Total employees
    pub total_employees: usize,
    /// Total customers
    pub total_customers: usize,
    /// Total revenue
    pub total_revenue: f64,
    /// Average customer satisfaction
    pub avg_customer_satisfaction: f64,
    /// Employee engagement score
    pub employee_engagement: f64,
    /// Organizational efficiency
    pub organizational_efficiency: f64,
    /// Innovation index
    pub innovation_index: f64,
    /// Top performing products
    pub top_products: Vec<String>,
    /// Top performing employees
    pub top_employees: Vec<String>,
    /// High-value customers
    pub high_value_customers: Vec<String>,
}
