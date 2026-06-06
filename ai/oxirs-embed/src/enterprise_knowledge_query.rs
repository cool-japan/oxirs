//! Enterprise Knowledge Query
//!
//! Knowledge querying: entity lookup, path queries, semantic similarity search,
//! market analysis, team composition optimization, and enterprise metrics.

use super::enterprise_knowledge_builder::EnterpriseKnowledgeAnalyzer;
use super::enterprise_knowledge_types::*;
use anyhow::Result;
use chrono::Utc;
use std::collections::{HashMap, HashSet};
use tracing::info;

impl EnterpriseKnowledgeAnalyzer {
    /// Get product recommendations for a customer
    pub async fn recommend_products(
        &self,
        customer_id: &str,
        num_recommendations: usize,
    ) -> Result<Vec<ProductRecommendation>> {
        let customer_embedding = self.generate_customer_embedding(customer_id).await?;

        if !customer_embedding.recommendations.is_empty()
            && customer_embedding.last_updated > Utc::now() - chrono::Duration::hours(6)
        {
            return Ok(customer_embedding
                .recommendations
                .into_iter()
                .take(num_recommendations)
                .collect());
        }

        self.generate_customer_recommendations(customer_id, &customer_embedding.embedding)
            .await
    }

    /// Find similar employees based on skills and experience
    pub async fn find_similar_employees(
        &self,
        employee_id: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let target_embedding = self.generate_employee_embedding(employee_id).await?;
        let embeddings = {
            let guard = self.employee_embeddings.read().expect("lock poisoned");
            guard.clone()
        };

        let mut similarities = Vec::new();

        for (other_id, other_embedding) in embeddings.iter() {
            if other_id != employee_id {
                let similarity = self
                    .calculate_employee_similarity(&target_embedding, other_embedding)
                    .await?;
                similarities.push((other_id.clone(), similarity));
            }
        }

        similarities.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("similarity scores should be finite")
        });
        similarities.truncate(k);

        Ok(similarities)
    }

    /// Optimize team composition for a project
    pub async fn optimize_team_composition(
        &self,
        _project_id: &str,
        required_skills: &[String],
    ) -> Result<Vec<String>> {
        let employees = {
            let guard = self.employee_embeddings.read().expect("lock poisoned");
            guard.clone()
        };
        let mut candidates = Vec::new();

        for (employee_id, employee) in employees.iter() {
            let skill_match_score = self
                .calculate_skill_match_score(&employee.skills, required_skills)
                .await?;
            candidates.push((employee_id.clone(), skill_match_score));
        }

        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("candidate scores should be finite")
        });

        let optimal_team = self.select_optimal_team(candidates, 5).await?;

        Ok(optimal_team)
    }

    /// Analyze market trends and opportunities
    pub async fn analyze_market_trends(&self) -> Result<MarketAnalysis> {
        let products = {
            let guard = self.product_embeddings.read().expect("lock poisoned");
            guard.clone()
        };
        let customers = {
            let guard = self.customer_embeddings.read().expect("lock poisoned");
            guard.clone()
        };

        let mut category_performance: HashMap<String, CategoryPerformance> = HashMap::new();
        let mut trending_products = Vec::new();

        for (product_id, product) in products.iter() {
            let performance = category_performance
                .entry(product.category.clone())
                .or_insert(CategoryPerformance {
                    total_sales: 0.0,
                    product_count: 0,
                    average_rating: 0.0,
                    growth_rate: 0.0,
                    market_share: 0.0,
                });

            performance.total_sales += product.sales_metrics.revenue;
            performance.product_count += 1;
            performance.average_rating += product.ratings.average_rating;

            if product.sales_metrics.sales_velocity > 20.0 {
                trending_products.push(product_id.clone());
            }
        }

        for performance in category_performance.values_mut() {
            if performance.product_count > 0 {
                performance.average_rating /= performance.product_count as f64;
            }
        }

        let mut segment_analysis = HashMap::new();
        for customer in customers.values() {
            let segment_name = format!("{:?}", customer.segment);
            let count = segment_analysis.entry(segment_name).or_insert(0u32);
            *count += 1;
        }

        Ok(MarketAnalysis {
            category_performance,
            trending_products,
            segment_distribution: segment_analysis,
            market_opportunities: self.identify_market_opportunities().await?,
            competitive_landscape: self.analyze_competitive_landscape().await?,
            forecast: self.generate_market_forecast().await?,
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

    // ===== QUERY HELPER METHODS =====

    async fn calculate_employee_similarity(
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

    async fn calculate_skill_similarity(
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

    async fn calculate_skill_match_score(
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

    async fn select_optimal_team(
        &self,
        candidates: Vec<(String, f64)>,
        team_size: usize,
    ) -> Result<Vec<String>> {
        let team: Vec<String> = candidates
            .into_iter()
            .take(team_size)
            .map(|(id, _score)| id)
            .collect();

        Ok(team)
    }

    /// Log query statistics for observability
    pub fn log_query_stats(&self) {
        let product_count = self
            .product_embeddings
            .read()
            .map(|g| g.len())
            .unwrap_or(0);
        let employee_count = self
            .employee_embeddings
            .read()
            .map(|g| g.len())
            .unwrap_or(0);
        let customer_count = self
            .customer_embeddings
            .read()
            .map(|g| g.len())
            .unwrap_or(0);

        info!(
            products = product_count,
            employees = employee_count,
            customers = customer_count,
            "Enterprise knowledge query stats"
        );
    }
}
