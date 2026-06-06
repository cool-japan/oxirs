//! Customer, purchase, preference, and recommendation types for enterprise knowledge.

use crate::Vector;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Customer embedding with behavior context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerEmbedding {
    /// Customer unique identifier
    pub customer_id: String,
    /// Customer name (anonymized if needed)
    pub name: String,
    /// Customer segment
    pub segment: CustomerSegment,
    /// Purchase history
    pub purchase_history: Vec<Purchase>,
    /// Preferences
    pub preferences: CustomerPreferences,
    /// Behavior metrics
    pub behavior_metrics: BehaviorMetrics,
    /// Customer embedding vector
    pub embedding: Vector,
    /// Predicted lifetime value
    pub predicted_ltv: f64,
    /// Churn risk
    pub churn_risk: f64,
    /// Recommended products
    pub recommendations: Vec<ProductRecommendation>,
    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Customer segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CustomerSegment {
    HighValue,
    Regular,
    Occasional,
    NewCustomer,
    AtRisk,
    Churned,
}

/// Purchase information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Purchase {
    /// Product ID
    pub product_id: String,
    /// Purchase date
    pub purchase_date: DateTime<Utc>,
    /// Quantity
    pub quantity: u32,
    /// Price paid
    pub price: f64,
    /// Channel used
    pub channel: PurchaseChannel,
    /// Satisfaction rating
    pub satisfaction: Option<u8>,
}

/// Purchase channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PurchaseChannel {
    Online,
    InStore,
    Mobile,
    Phone,
    ThirdParty,
}

/// Customer preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerPreferences {
    /// Preferred categories
    pub preferred_categories: Vec<String>,
    /// Price sensitivity
    pub price_sensitivity: f64,
    /// Brand loyalty
    pub brand_loyalty: HashMap<String, f64>,
    /// Preferred channels
    pub preferred_channels: Vec<PurchaseChannel>,
    /// Communication preferences
    pub communication_preferences: CommunicationPreferences,
}

/// Communication preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPreferences {
    /// Email opt-in
    pub email_opt_in: bool,
    /// SMS opt-in
    pub sms_opt_in: bool,
    /// Frequency preference
    pub frequency: CommunicationFrequency,
    /// Content preferences
    pub content_types: Vec<String>,
}

/// Communication frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationFrequency {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Never,
}

/// Customer behavior metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorMetrics {
    /// Visit frequency
    pub visit_frequency: f64,
    /// Average session duration (minutes)
    pub avg_session_duration: f64,
    /// Pages/products viewed per session
    pub avg_products_viewed: f64,
    /// Cart abandonment rate
    pub cart_abandonment_rate: f64,
    /// Return visit rate
    pub return_visit_rate: f64,
    /// Referral rate
    pub referral_rate: f64,
}

/// Product recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductRecommendation {
    /// Product ID
    pub product_id: String,
    /// Recommendation score
    pub score: f64,
    /// Reason for recommendation
    pub reason: RecommendationReason,
    /// Confidence level
    pub confidence: f64,
    /// Expected revenue impact
    pub expected_revenue: f64,
}

/// Reasons for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationReason {
    SimilarProducts,
    CustomersBought,
    PopularInCategory,
    PersonalizedPreference,
    TrendingNow,
    SeasonalRecommendation,
}
