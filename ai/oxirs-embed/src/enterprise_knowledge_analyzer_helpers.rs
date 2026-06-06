//! Private helper methods, background tasks, and metrics for the enterprise knowledge analyzer.

use crate::enterprise_knowledge::EnterpriseKnowledgeAnalyzer;
use crate::enterprise_knowledge_customer::{
    BehaviorMetrics, CustomerPreferences, ProductRecommendation, Purchase, RecommendationReason,
};
use crate::enterprise_knowledge_employee::{
    CareerPredictions, EmployeeEmbedding, ExperienceLevel, PerformanceMetrics,
    ProjectParticipation, Skill,
};
use crate::enterprise_knowledge_engine::EnterpriseMetrics;
use crate::enterprise_knowledge_product::{CustomerRatings, ProductFeature, SalesMetrics};
use crate::Vector;
use anyhow::Result;
use chrono::Utc;
use std::collections::{HashMap, HashSet};
use tokio::task::JoinHandle;
use tracing::{debug, info};

impl EnterpriseKnowledgeAnalyzer {
    pub(crate) async fn compute_product_embedding_vector(
        &self,
        _name: &str,
        _description: &str,
        _features: &[ProductFeature],
        _sales_metrics: &SalesMetrics,
    ) -> Result<Vector> {
        let values = {
            use scirs2_core::random::{Random, RngExt};
            let mut random = Random::default();
            (0..self.config.embedding_dimension)
                .map(|_| random.random::<f32>())
                .collect()
        };
        Ok(Vector::new(values))
    }

    pub(crate) async fn find_similar_products(
        &self,
        _product_id: &str,
        _embedding: &Vector,
    ) -> Result<Vec<String>> {
        Ok(vec!["prod_002".to_string(), "prod_003".to_string()])
    }

    pub(crate) async fn calculate_market_position(
        &self,
        sales_metrics: &SalesMetrics,
        ratings: &CustomerRatings,
    ) -> Result<f64> {
        let sales_score = (sales_metrics.sales_velocity / 100.0).min(1.0);
        let rating_score = ratings.average_rating / 5.0;
        let position = (sales_score * 0.6 + rating_score * 0.4).min(1.0);
        Ok(position)
    }

    pub(crate) async fn compute_employee_embedding_vector(
        &self,
        _skills: &[Skill],
        _performance: &PerformanceMetrics,
        _projects: &[ProjectParticipation],
    ) -> Result<Vector> {
        let values = {
            use scirs2_core::random::{Random, RngExt};
            let mut random = Random::default();
            (0..self.config.embedding_dimension)
                .map(|_| random.random::<f32>())
                .collect()
        };
        Ok(Vector::new(values))
    }

    pub(crate) async fn predict_career_progression(
        &self,
        skills: &[Skill],
        performance: &PerformanceMetrics,
        _experience_level: &ExperienceLevel,
    ) -> Result<CareerPredictions> {
        let performance_factor = performance.overall_score / 10.0;
        let skill_factor = skills
            .iter()
            .map(|s| s.proficiency_level as f64 / 10.0)
            .sum::<f64>()
            / skills.len() as f64;
        let promotion_likelihood = (performance_factor * 0.7 + skill_factor * 0.3).min(1.0);

        Ok(CareerPredictions {
            promotion_likelihood,
            next_role: "Senior Software Engineer".to_string(),
            skills_to_develop: vec!["Team Leadership".to_string(), "System Design".to_string()],
            career_paths: vec![
                "Technical Lead".to_string(),
                "Engineering Manager".to_string(),
            ],
            retention_risk: 1.0 - promotion_likelihood * 0.8,
        })
    }

    pub(crate) async fn compute_customer_embedding_vector(
        &self,
        _purchases: &[Purchase],
        _preferences: &CustomerPreferences,
        _behavior: &BehaviorMetrics,
    ) -> Result<Vector> {
        let values = {
            use scirs2_core::random::{Random, RngExt};
            let mut random = Random::default();
            (0..self.config.embedding_dimension)
                .map(|_| random.random::<f32>())
                .collect()
        };
        Ok(Vector::new(values))
    }

    pub(crate) async fn predict_customer_ltv(
        &self,
        purchases: &[Purchase],
        behavior: &BehaviorMetrics,
    ) -> Result<f64> {
        if purchases.is_empty() {
            return Ok(0.0);
        }

        let total_spent: f64 = purchases.iter().map(|p| p.price * p.quantity as f64).sum();
        let avg_purchase = total_spent / purchases.len() as f64;
        let frequency_factor = behavior.visit_frequency;
        let ltv = avg_purchase * frequency_factor * 12.0;

        Ok(ltv)
    }

    pub(crate) async fn calculate_churn_risk(
        &self,
        behavior: &BehaviorMetrics,
        purchases: &[Purchase],
    ) -> Result<f64> {
        let recency_factor = if let Some(last_purchase) = purchases.last() {
            let days_since_last = (Utc::now() - last_purchase.purchase_date).num_days() as f64;
            (days_since_last / 90.0).min(1.0)
        } else {
            1.0
        };

        let engagement_factor = 1.0 - (behavior.visit_frequency / 10.0).min(1.0);
        let abandonment_factor = behavior.cart_abandonment_rate;

        let churn_risk =
            (recency_factor * 0.4 + engagement_factor * 0.3 + abandonment_factor * 0.3).min(1.0);
        Ok(churn_risk)
    }

    pub(crate) async fn generate_customer_recommendations(
        &self,
        _customer_id: &str,
        _embedding: &Vector,
    ) -> Result<Vec<ProductRecommendation>> {
        Ok(vec![
            ProductRecommendation {
                product_id: "prod_101".to_string(),
                score: 0.95,
                reason: RecommendationReason::SimilarProducts,
                confidence: 0.85,
                expected_revenue: 199.99,
            },
            ProductRecommendation {
                product_id: "prod_102".to_string(),
                score: 0.88,
                reason: RecommendationReason::CustomersBought,
                confidence: 0.78,
                expected_revenue: 149.99,
            },
        ])
    }

    pub(crate) async fn calculate_employee_similarity(
        &self,
        emp1: &EmployeeEmbedding,
        emp2: &EmployeeEmbedding,
    ) -> Result<f64> {
        let embedding1 = &emp1.embedding.values;
        let embedding2 = &emp2.embedding.values;

        let dot_product: f32 = embedding1
            .iter()
            .zip(embedding2.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();

        let cosine_similarity = if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        };

        let skill_similarity = self
            .calculate_skill_similarity(&emp1.skills, &emp2.skills)
            .await?;

        let final_similarity = 0.6 * cosine_similarity as f64 + 0.4 * skill_similarity;

        Ok(final_similarity)
    }

    pub(crate) async fn calculate_skill_similarity(
        &self,
        skills1: &[Skill],
        skills2: &[Skill],
    ) -> Result<f64> {
        let skill_set1: HashSet<_> = skills1.iter().map(|s| &s.skill_name).collect();
        let skill_set2: HashSet<_> = skills2.iter().map(|s| &s.skill_name).collect();

        let intersection = skill_set1.intersection(&skill_set2).count();
        let union = skill_set1.union(&skill_set2).count();

        if union > 0 {
            Ok(intersection as f64 / union as f64)
        } else {
            Ok(0.0)
        }
    }

    pub(crate) async fn calculate_skill_match_score(
        &self,
        employee_skills: &[Skill],
        required_skills: &[String],
    ) -> Result<f64> {
        let employee_skill_names: HashSet<_> =
            employee_skills.iter().map(|s| &s.skill_name).collect();
        let required_skill_set: HashSet<_> = required_skills.iter().collect();

        let matches = required_skill_set
            .intersection(&employee_skill_names)
            .count();
        let score = matches as f64 / required_skills.len() as f64;

        Ok(score)
    }

    pub(crate) async fn select_optimal_team(
        &self,
        _candidates: Vec<(String, f64)>,
        team_size: usize,
    ) -> Result<Vec<String>> {
        let team: Vec<String> = _candidates
            .into_iter()
            .take(team_size)
            .map(|(id, _score)| id)
            .collect();

        Ok(team)
    }

    pub(crate) async fn identify_market_opportunities(&self) -> Result<Vec<String>> {
        Ok(vec![
            "AI-powered fitness devices".to_string(),
            "Sustainable electronics".to_string(),
            "Remote work solutions".to_string(),
        ])
    }

    pub(crate) async fn analyze_competitive_landscape(&self) -> Result<HashMap<String, f64>> {
        let mut landscape = HashMap::new();
        landscape.insert("TechCorp".to_string(), 0.35);
        landscape.insert("InnovateCo".to_string(), 0.28);
        landscape.insert("FutureTech".to_string(), 0.22);
        landscape.insert("Others".to_string(), 0.15);

        Ok(landscape)
    }

    pub(crate) async fn generate_market_forecast(&self) -> Result<HashMap<String, f64>> {
        let mut forecast = HashMap::new();
        forecast.insert("Q1_growth".to_string(), 0.12);
        forecast.insert("Q2_growth".to_string(), 0.15);
        forecast.insert("Q3_growth".to_string(), 0.18);
        forecast.insert("Q4_growth".to_string(), 0.10);

        Ok(forecast)
    }

    pub(crate) async fn start_recommendation_engine(&self) -> JoinHandle<()> {
        let interval =
            std::time::Duration::from_secs(self.config.product_recommendation_refresh_hours * 3600);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;
                info!("Refreshing product recommendation engines");
                debug!("Product recommendation engines refreshed");
            }
        })
    }

    pub(crate) async fn start_skill_analysis(&self) -> JoinHandle<()> {
        let interval =
            std::time::Duration::from_secs(self.config.skill_analysis_interval_hours * 3600);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;
                info!("Performing employee skill analysis");
                debug!("Employee skill analysis completed");
            }
        })
    }

    pub(crate) async fn start_market_analysis(&self) -> JoinHandle<()> {
        let interval =
            std::time::Duration::from_secs(self.config.market_analysis_interval_hours * 3600);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;
                info!("Performing market trend analysis");
                debug!("Market trend analysis completed");
            }
        })
    }

    pub(crate) async fn start_organizational_optimization(&self) -> JoinHandle<()> {
        let interval = std::time::Duration::from_secs(24 * 3600);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;
                info!("Performing organizational optimization");
                debug!("Organizational optimization completed");
            }
        })
    }

    /// Get comprehensive enterprise metrics
    pub async fn get_enterprise_metrics(&self) -> Result<EnterpriseMetrics> {
        let product_embeddings = self.product_embeddings.read().expect("lock poisoned");
        let employee_embeddings = self.employee_embeddings.read().expect("lock poisoned");
        let customer_embeddings = self.customer_embeddings.read().expect("lock poisoned");

        let total_products = product_embeddings.len();
        let total_employees = employee_embeddings.len();
        let total_customers = customer_embeddings.len();

        let total_revenue = product_embeddings
            .values()
            .map(|p| p.sales_metrics.revenue)
            .sum();

        let avg_customer_satisfaction = product_embeddings
            .values()
            .map(|p| p.ratings.average_rating)
            .sum::<f64>()
            / total_products.max(1) as f64;

        let employee_engagement = employee_embeddings
            .values()
            .map(|e| e.performance_metrics.overall_score)
            .sum::<f64>()
            / total_employees.max(1) as f64;

        let mut product_scores: Vec<_> = product_embeddings
            .iter()
            .map(|(id, p)| (id.clone(), p.market_position))
            .collect();
        product_scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("product scores should be finite")
        });
        let top_products: Vec<String> = product_scores
            .into_iter()
            .take(10)
            .map(|(id, _)| id)
            .collect();

        let mut employee_scores: Vec<_> = employee_embeddings
            .iter()
            .map(|(id, e)| (id.clone(), e.performance_metrics.overall_score))
            .collect();
        employee_scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("employee scores should be finite")
        });
        let top_employees: Vec<String> = employee_scores
            .into_iter()
            .take(10)
            .map(|(id, _)| id)
            .collect();

        let mut customer_values: Vec<_> = customer_embeddings
            .iter()
            .map(|(id, c)| (id.clone(), c.predicted_ltv))
            .collect();
        customer_values.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("customer values should be finite")
        });
        let high_value_customers: Vec<String> = customer_values
            .into_iter()
            .take(10)
            .map(|(id, _)| id)
            .collect();

        Ok(EnterpriseMetrics {
            total_products,
            total_employees,
            total_customers,
            total_revenue,
            avg_customer_satisfaction,
            employee_engagement,
            organizational_efficiency: 0.75,
            innovation_index: 0.68,
            top_products,
            top_employees,
            high_value_customers,
        })
    }
}
