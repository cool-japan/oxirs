//! Public entity-generation and analysis methods of `EnterpriseKnowledgeAnalyzer`.

use crate::enterprise_knowledge::EnterpriseKnowledgeAnalyzer;
use crate::enterprise_knowledge_customer::{
    BehaviorMetrics, CommunicationFrequency, CommunicationPreferences, CustomerEmbedding,
    CustomerPreferences, CustomerSegment, ProductRecommendation, Purchase, PurchaseChannel,
};
use crate::enterprise_knowledge_employee::{
    EmployeeEmbedding, ExperienceLevel, PerformanceMetrics, ProjectOutcome, ProjectParticipation,
    Skill, SkillCategory,
};
use crate::enterprise_knowledge_engine::MarketAnalysis;
use crate::enterprise_knowledge_product::{
    CategoryPerformance, CustomerRatings, FeatureType, ProductAvailability, ProductEmbedding,
    ProductFeature, SalesMetrics,
};
use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;
use tracing::info;

impl EnterpriseKnowledgeAnalyzer {
    /// Start background analysis tasks
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting enterprise knowledge analysis system");

        let recommendation_task = self.start_recommendation_engine().await;
        self.analysis_tasks.push(recommendation_task);

        let skill_analysis_task = self.start_skill_analysis().await;
        self.analysis_tasks.push(skill_analysis_task);

        let market_analysis_task = self.start_market_analysis().await;
        self.analysis_tasks.push(market_analysis_task);

        let org_optimization_task = self.start_organizational_optimization().await;
        self.analysis_tasks.push(org_optimization_task);

        info!("Enterprise knowledge analysis system started successfully");
        Ok(())
    }

    /// Stop analysis tasks
    pub async fn stop(&mut self) {
        info!("Stopping enterprise knowledge analysis system");

        for task in self.analysis_tasks.drain(..) {
            task.abort();
        }

        info!("Enterprise knowledge analysis system stopped");
    }

    /// Generate product embedding with business features
    pub async fn generate_product_embedding(&self, product_id: &str) -> Result<ProductEmbedding> {
        {
            let embeddings = self.product_embeddings.read().expect("lock poisoned");
            if let Some(existing) = embeddings.get(product_id) {
                return Ok(existing.clone());
            }
        }

        info!("Generating product embedding for: {}", product_id);

        let name = format!("Product_{product_id}");
        let description = format!("Description for product {product_id}");
        let category = "Electronics".to_string();
        let subcategories = vec!["Smartphones".to_string(), "Mobile".to_string()];

        let features = vec![
            ProductFeature {
                feature_name: "Brand".to_string(),
                feature_value: "TechCorp".to_string(),
                feature_type: FeatureType::Categorical,
                importance_score: 0.9,
            },
            ProductFeature {
                feature_name: "Price".to_string(),
                feature_value: "299.99".to_string(),
                feature_type: FeatureType::Numerical,
                importance_score: 0.8,
            },
        ];

        let price = 299.99;
        let availability = ProductAvailability::InStock(100);

        let sales_metrics = SalesMetrics {
            units_sold: 1500,
            revenue: 449_985.0,
            sales_velocity: 25.5,
            conversion_rate: 0.12,
            return_rate: 0.03,
            profit_margin: 0.35,
        };

        let mut rating_distribution = HashMap::new();
        rating_distribution.insert(5, 120);
        rating_distribution.insert(4, 80);
        rating_distribution.insert(3, 30);
        rating_distribution.insert(2, 10);
        rating_distribution.insert(1, 5);

        let ratings = CustomerRatings {
            average_rating: 4.2,
            review_count: 245,
            rating_distribution,
            sentiment_score: 0.7,
        };

        let embedding = self
            .compute_product_embedding_vector(&name, &description, &features, &sales_metrics)
            .await?;

        let similar_products = self.find_similar_products(product_id, &embedding).await?;

        let market_position = self
            .calculate_market_position(&sales_metrics, &ratings)
            .await?;

        let product_embedding = ProductEmbedding {
            product_id: product_id.to_string(),
            name,
            description,
            category,
            subcategories,
            features,
            price,
            availability,
            sales_metrics,
            ratings,
            embedding,
            similar_products,
            market_position,
            last_updated: Utc::now(),
        };

        {
            let mut embeddings = self.product_embeddings.write().expect("lock poisoned");
            embeddings.insert(product_id.to_string(), product_embedding.clone());
        }

        info!(
            "Generated product embedding for {} with market position: {:.3}",
            product_id, market_position
        );
        Ok(product_embedding)
    }

    /// Generate employee embedding with skills and performance
    pub async fn generate_employee_embedding(
        &self,
        employee_id: &str,
    ) -> Result<EmployeeEmbedding> {
        {
            let embeddings = self.employee_embeddings.read().expect("lock poisoned");
            if let Some(existing) = embeddings.get(employee_id) {
                return Ok(existing.clone());
            }
        }

        info!("Generating employee embedding for: {}", employee_id);

        let name = format!("Employee_{employee_id}");
        let job_title = "Software Engineer".to_string();
        let department = "Engineering".to_string();
        let team = "Backend Team".to_string();

        let skills = vec![
            Skill {
                skill_name: "Python".to_string(),
                category: SkillCategory::Technical,
                proficiency_level: 8,
                years_experience: 5.0,
                role_importance: 0.9,
                market_demand: 0.85,
            },
            Skill {
                skill_name: "Leadership".to_string(),
                category: SkillCategory::Leadership,
                proficiency_level: 6,
                years_experience: 2.0,
                role_importance: 0.6,
                market_demand: 0.9,
            },
        ];

        let experience_level = ExperienceLevel::Mid;

        let performance_metrics = PerformanceMetrics {
            overall_score: 8.2,
            goal_achievement_rate: 0.92,
            project_completion_rate: 0.95,
            collaboration_score: 8.5,
            innovation_score: 7.8,
            leadership_score: 6.5,
        };

        let project_history = vec![ProjectParticipation {
            project_id: "proj_001".to_string(),
            project_name: "Customer Portal".to_string(),
            role: "Backend Developer".to_string(),
            start_date: Utc::now() - chrono::Duration::days(365),
            end_date: Some(Utc::now() - chrono::Duration::days(300)),
            outcome: ProjectOutcome::Successful,
            contribution_score: 8.5,
        }];

        let collaborators = vec!["emp_002".to_string(), "emp_003".to_string()];

        let embedding = self
            .compute_employee_embedding_vector(&skills, &performance_metrics, &project_history)
            .await?;

        let career_predictions = self
            .predict_career_progression(&skills, &performance_metrics, &experience_level)
            .await?;

        let employee_embedding = EmployeeEmbedding {
            employee_id: employee_id.to_string(),
            name,
            job_title,
            department,
            team,
            skills,
            experience_level,
            performance_metrics,
            project_history,
            collaborators,
            embedding,
            career_predictions,
            last_updated: Utc::now(),
        };

        {
            let mut embeddings = self.employee_embeddings.write().expect("lock poisoned");
            embeddings.insert(employee_id.to_string(), employee_embedding.clone());
        }

        info!(
            "Generated employee embedding for {} with promotion likelihood: {:.3}",
            employee_id, employee_embedding.career_predictions.promotion_likelihood
        );
        Ok(employee_embedding)
    }

    /// Generate customer embedding with behavior and preferences
    pub async fn generate_customer_embedding(
        &self,
        customer_id: &str,
    ) -> Result<CustomerEmbedding> {
        {
            let embeddings = self.customer_embeddings.read().expect("lock poisoned");
            if let Some(existing) = embeddings.get(customer_id) {
                return Ok(existing.clone());
            }
        }

        info!("Generating customer embedding for: {}", customer_id);

        let name = format!("Customer_{customer_id}");
        let segment = CustomerSegment::Regular;

        let purchase_history = vec![
            Purchase {
                product_id: "prod_001".to_string(),
                purchase_date: Utc::now() - chrono::Duration::days(30),
                quantity: 1,
                price: 299.99,
                channel: PurchaseChannel::Online,
                satisfaction: Some(4),
            },
            Purchase {
                product_id: "prod_002".to_string(),
                purchase_date: Utc::now() - chrono::Duration::days(60),
                quantity: 2,
                price: 149.99,
                channel: PurchaseChannel::InStore,
                satisfaction: Some(5),
            },
        ];

        let mut brand_loyalty = HashMap::new();
        brand_loyalty.insert("TechCorp".to_string(), 0.8);
        brand_loyalty.insert("InnovateCo".to_string(), 0.6);

        let preferences = CustomerPreferences {
            preferred_categories: vec!["Electronics".to_string(), "Books".to_string()],
            price_sensitivity: 0.6,
            brand_loyalty,
            preferred_channels: vec![PurchaseChannel::Online, PurchaseChannel::Mobile],
            communication_preferences: CommunicationPreferences {
                email_opt_in: true,
                sms_opt_in: false,
                frequency: CommunicationFrequency::Weekly,
                content_types: vec!["Promotions".to_string(), "NewProducts".to_string()],
            },
        };

        let behavior_metrics = BehaviorMetrics {
            visit_frequency: 2.5,
            avg_session_duration: 12.5,
            avg_products_viewed: 8.2,
            cart_abandonment_rate: 0.25,
            return_visit_rate: 0.7,
            referral_rate: 0.1,
        };

        let embedding = self
            .compute_customer_embedding_vector(&purchase_history, &preferences, &behavior_metrics)
            .await?;

        let predicted_ltv = self
            .predict_customer_ltv(&purchase_history, &behavior_metrics)
            .await?;

        let churn_risk = self
            .calculate_churn_risk(&behavior_metrics, &purchase_history)
            .await?;

        let recommendations = self
            .generate_customer_recommendations(customer_id, &embedding)
            .await?;

        let customer_embedding = CustomerEmbedding {
            customer_id: customer_id.to_string(),
            name,
            segment,
            purchase_history,
            preferences,
            behavior_metrics,
            embedding,
            predicted_ltv,
            churn_risk,
            recommendations,
            last_updated: Utc::now(),
        };

        {
            let mut embeddings = self.customer_embeddings.write().expect("lock poisoned");
            embeddings.insert(customer_id.to_string(), customer_embedding.clone());
        }

        info!(
            "Generated customer embedding for {} with LTV: ${:.2} and churn risk: {:.3}",
            customer_id, predicted_ltv, churn_risk
        );
        Ok(customer_embedding)
    }

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

        let mut category_performance = HashMap::new();
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
            let count = segment_analysis.entry(segment_name).or_insert(0);
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
}
