//! Product, category, and sales-related types for enterprise knowledge.

use crate::Vector;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Product embedding with business context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductEmbedding {
    /// Product unique identifier
    pub product_id: String,
    /// Product name
    pub name: String,
    /// Product description
    pub description: String,
    /// Product category
    pub category: String,
    /// Subcategories
    pub subcategories: Vec<String>,
    /// Product features
    pub features: Vec<ProductFeature>,
    /// Price information
    pub price: f64,
    /// Availability status
    pub availability: ProductAvailability,
    /// Sales metrics
    pub sales_metrics: SalesMetrics,
    /// Customer ratings
    pub ratings: CustomerRatings,
    /// Product embedding vector
    pub embedding: Vector,
    /// Similar products
    pub similar_products: Vec<String>,
    /// Market position score
    pub market_position: f64,
    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Product feature information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductFeature {
    /// Feature name
    pub feature_name: String,
    /// Feature value
    pub feature_value: String,
    /// Feature type
    pub feature_type: FeatureType,
    /// Feature importance score
    pub importance_score: f64,
}

/// Types of product features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureType {
    Categorical,
    Numerical,
    Boolean,
    Text,
    List,
}

/// Product availability status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProductAvailability {
    InStock(u32),
    OutOfStock,
    Discontinued,
    PreOrder(DateTime<Utc>),
    Limited(u32),
}

/// Sales metrics for products
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalesMetrics {
    /// Total units sold
    pub units_sold: u64,
    /// Revenue generated
    pub revenue: f64,
    /// Sales velocity (units per day)
    pub sales_velocity: f64,
    /// Conversion rate
    pub conversion_rate: f64,
    /// Return rate
    pub return_rate: f64,
    /// Profit margin
    pub profit_margin: f64,
}

/// Customer ratings and reviews
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerRatings {
    /// Average rating (1-5)
    pub average_rating: f64,
    /// Total number of reviews
    pub review_count: u32,
    /// Rating distribution
    pub rating_distribution: HashMap<u8, u32>,
    /// Sentiment score (-1 to 1)
    pub sentiment_score: f64,
}

/// Category hierarchy structure
#[derive(Debug, Clone)]
pub struct CategoryHierarchy {
    /// Category tree
    pub categories: HashMap<String, Category>,
    /// Category relationships
    pub parent_child: HashMap<String, Vec<String>>,
    /// Category embeddings
    pub category_embeddings: HashMap<String, Vector>,
}

/// Category information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Category {
    /// Category ID
    pub category_id: String,
    /// Category name
    pub name: String,
    /// Parent category
    pub parent: Option<String>,
    /// Child categories
    pub children: Vec<String>,
    /// Products in this category
    pub products: Vec<String>,
    /// Category attributes
    pub attributes: HashMap<String, String>,
    /// Category performance metrics
    pub performance: CategoryPerformance,
}

/// Category performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryPerformance {
    /// Total sales
    pub total_sales: f64,
    /// Product count
    pub product_count: u32,
    /// Average rating
    pub average_rating: f64,
    /// Growth rate
    pub growth_rate: f64,
    /// Market share
    pub market_share: f64,
}
