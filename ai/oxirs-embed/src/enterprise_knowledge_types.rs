//! Enterprise Knowledge Types
//!
//! Type definitions for enterprise knowledge graphs: entities, embeddings,
//! metrics, enumerations, and configuration structures.

use crate::Vector;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    InStock(u32), // quantity
    OutOfStock,
    Discontinued,
    PreOrder(DateTime<Utc>), // available date
    Limited(u32),            // limited quantity
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

/// Employee embedding with professional context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmployeeEmbedding {
    /// Employee unique identifier
    pub employee_id: String,
    /// Employee name
    pub name: String,
    /// Job title
    pub job_title: String,
    /// Department
    pub department: String,
    /// Team
    pub team: String,
    /// Skills
    pub skills: Vec<Skill>,
    /// Experience level
    pub experience_level: ExperienceLevel,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Project history
    pub project_history: Vec<ProjectParticipation>,
    /// Collaboration network
    pub collaborators: Vec<String>,
    /// Employee embedding vector
    pub embedding: Vector,
    /// Career progression predictions
    pub career_predictions: CareerPredictions,
    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Skill information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Skill name
    pub skill_name: String,
    /// Skill category
    pub category: SkillCategory,
    /// Proficiency level (1-10)
    pub proficiency_level: u8,
    /// Years of experience
    pub years_experience: f64,
    /// Skill importance in role
    pub role_importance: f64,
    /// Market demand score
    pub market_demand: f64,
}

/// Skill categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SkillCategory {
    Technical,
    Leadership,
    Communication,
    Analytical,
    Creative,
    Domain,
    Language,
    Tools,
}

/// Experience levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperienceLevel {
    Junior,
    Mid,
    Senior,
    Lead,
    Principal,
    Executive,
}

/// Performance metrics for employees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Overall performance score (1-10)
    pub overall_score: f64,
    /// Goal achievement rate
    pub goal_achievement_rate: f64,
    /// Project completion rate
    pub project_completion_rate: f64,
    /// Collaboration score
    pub collaboration_score: f64,
    /// Innovation score
    pub innovation_score: f64,
    /// Leadership score
    pub leadership_score: f64,
}

/// Project participation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectParticipation {
    /// Project ID
    pub project_id: String,
    /// Project name
    pub project_name: String,
    /// Role in project
    pub role: String,
    /// Start date
    pub start_date: DateTime<Utc>,
    /// End date
    pub end_date: Option<DateTime<Utc>>,
    /// Project outcome
    pub outcome: ProjectOutcome,
    /// Contribution score
    pub contribution_score: f64,
}

/// Project outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProjectOutcome {
    Successful,
    PartiallySuccessful,
    Failed,
    Cancelled,
    Ongoing,
}

/// Career progression predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CareerPredictions {
    /// Promotion likelihood (0-1)
    pub promotion_likelihood: f64,
    /// Predicted next role
    pub next_role: String,
    /// Skills to develop
    pub skills_to_develop: Vec<String>,
    /// Career path recommendations
    pub career_paths: Vec<String>,
    /// Retention risk (0-1)
    pub retention_risk: f64,
}

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

/// Organizational structure
#[derive(Debug, Clone)]
pub struct OrganizationalStructure {
    /// Departments
    pub departments: HashMap<String, Department>,
    /// Teams within departments
    pub teams: HashMap<String, Team>,
    /// Reporting relationships
    pub reporting_structure: HashMap<String, Vec<String>>,
    /// Cross-functional projects
    pub projects: HashMap<String, Project>,
}

/// Department information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Department {
    /// Department ID
    pub department_id: String,
    /// Department name
    pub name: String,
    /// Department head
    pub head: String,
    /// Employees
    pub employees: Vec<String>,
    /// Teams
    pub teams: Vec<String>,
    /// Budget
    pub budget: f64,
    /// Performance metrics
    pub performance: DepartmentPerformance,
}

/// Department performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepartmentPerformance {
    /// Budget utilization
    pub budget_utilization: f64,
    /// Goal achievement
    pub goal_achievement: f64,
    /// Employee satisfaction
    pub employee_satisfaction: f64,
    /// Productivity score
    pub productivity_score: f64,
    /// Innovation index
    pub innovation_index: f64,
}

/// Team information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Team {
    /// Team ID
    pub team_id: String,
    /// Team name
    pub name: String,
    /// Team lead
    pub lead: String,
    /// Team members
    pub members: Vec<String>,
    /// Department
    pub department: String,
    /// Team skills
    pub team_skills: Vec<Skill>,
    /// Team performance
    pub performance: TeamPerformance,
}

/// Team performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamPerformance {
    /// Collaboration score
    pub collaboration_score: f64,
    /// Delivery performance
    pub delivery_performance: f64,
    /// Quality metrics
    pub quality_score: f64,
    /// Innovation score
    pub innovation_score: f64,
    /// Team satisfaction
    pub team_satisfaction: f64,
}

/// Project information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    /// Project ID
    pub project_id: String,
    /// Project name
    pub name: String,
    /// Project description
    pub description: String,
    /// Project manager
    pub manager: String,
    /// Team members
    pub team_members: Vec<String>,
    /// Start date
    pub start_date: DateTime<Utc>,
    /// End date
    pub end_date: Option<DateTime<Utc>>,
    /// Budget
    pub budget: f64,
    /// Status
    pub status: ProjectStatus,
    /// Required skills
    pub required_skills: Vec<String>,
    /// Performance metrics
    pub performance: ProjectPerformance,
}

/// Project status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProjectStatus {
    Planning,
    InProgress,
    OnHold,
    Completed,
    Cancelled,
}

/// Project performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectPerformance {
    /// Progress percentage
    pub progress_percentage: f64,
    /// Budget utilization
    pub budget_utilization: f64,
    /// Timeline adherence
    pub timeline_adherence: f64,
    /// Quality score
    pub quality_score: f64,
    /// Stakeholder satisfaction
    pub stakeholder_satisfaction: f64,
}

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
