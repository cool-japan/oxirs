//! Tests for the enterprise knowledge analyzer.

#[cfg(test)]
mod tests {
    use crate::enterprise_knowledge::EnterpriseKnowledgeAnalyzer;
    use crate::enterprise_knowledge_config::EnterpriseConfig;

    #[tokio::test]
    async fn test_enterprise_analyzer_creation() {
        let config = EnterpriseConfig::default();
        let analyzer = EnterpriseKnowledgeAnalyzer::new(config);

        assert_eq!(
            analyzer
                .product_embeddings
                .read()
                .expect("should succeed")
                .len(),
            0
        );
        assert_eq!(
            analyzer
                .employee_embeddings
                .read()
                .expect("should succeed")
                .len(),
            0
        );
        assert_eq!(
            analyzer
                .customer_embeddings
                .read()
                .expect("should succeed")
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn test_product_embedding_generation() {
        let config = EnterpriseConfig::default();
        let analyzer = EnterpriseKnowledgeAnalyzer::new(config);

        let result = analyzer.generate_product_embedding("test_product").await;
        assert!(result.is_ok());

        let embedding = result.expect("should succeed");
        assert_eq!(embedding.product_id, "test_product");
        assert!(embedding.market_position >= 0.0);
        assert!(embedding.market_position <= 1.0);
        assert_eq!(embedding.embedding.values.len(), 256);
    }

    #[tokio::test]
    async fn test_employee_embedding_generation() {
        let config = EnterpriseConfig::default();
        let analyzer = EnterpriseKnowledgeAnalyzer::new(config);

        let result = analyzer.generate_employee_embedding("test_employee").await;
        assert!(result.is_ok());

        let embedding = result.expect("should succeed");
        assert_eq!(embedding.employee_id, "test_employee");
        assert!(embedding.career_predictions.promotion_likelihood >= 0.0);
        assert!(embedding.career_predictions.promotion_likelihood <= 1.0);
    }

    #[tokio::test]
    async fn test_customer_embedding_generation() {
        let config = EnterpriseConfig::default();
        let analyzer = EnterpriseKnowledgeAnalyzer::new(config);

        let result = analyzer.generate_customer_embedding("test_customer").await;
        assert!(result.is_ok());

        let embedding = result.expect("should succeed");
        assert_eq!(embedding.customer_id, "test_customer");
        assert!(embedding.predicted_ltv >= 0.0);
        assert!(embedding.churn_risk >= 0.0);
        assert!(embedding.churn_risk <= 1.0);
    }

    #[tokio::test]
    async fn test_product_recommendations() {
        let config = EnterpriseConfig::default();
        let analyzer = EnterpriseKnowledgeAnalyzer::new(config);

        let _customer = analyzer
            .generate_customer_embedding("test_customer")
            .await
            .expect("should succeed");

        let recommendations = analyzer.recommend_products("test_customer", 5).await;
        assert!(recommendations.is_ok());

        let recs = recommendations.expect("should succeed");
        assert!(!recs.is_empty());
        assert!(recs.len() <= 5);

        for rec in &recs {
            assert!(rec.score >= 0.0);
            assert!(rec.score <= 1.0);
            assert!(rec.confidence >= 0.0);
            assert!(rec.confidence <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_market_analysis() {
        let config = EnterpriseConfig::default();
        let analyzer = EnterpriseKnowledgeAnalyzer::new(config);

        let _product = analyzer
            .generate_product_embedding("test_product")
            .await
            .expect("should succeed");
        let _customer = analyzer
            .generate_customer_embedding("test_customer")
            .await
            .expect("should succeed");

        let analysis = analyzer.analyze_market_trends().await;
        assert!(analysis.is_ok());

        let market_analysis = analysis.expect("should succeed");
        assert!(!market_analysis.competitive_landscape.is_empty());
        assert!(!market_analysis.forecast.is_empty());
    }

    #[tokio::test]
    async fn test_enterprise_metrics() {
        let config = EnterpriseConfig::default();
        let analyzer = EnterpriseKnowledgeAnalyzer::new(config);

        let _product = analyzer
            .generate_product_embedding("test_product")
            .await
            .expect("should succeed");
        let _employee = analyzer
            .generate_employee_embedding("test_employee")
            .await
            .expect("should succeed");
        let _customer = analyzer
            .generate_customer_embedding("test_customer")
            .await
            .expect("should succeed");

        let metrics = analyzer.get_enterprise_metrics().await;
        assert!(metrics.is_ok());

        let enterprise_metrics = metrics.expect("should succeed");
        assert_eq!(enterprise_metrics.total_products, 1);
        assert_eq!(enterprise_metrics.total_employees, 1);
        assert_eq!(enterprise_metrics.total_customers, 1);
        assert!(enterprise_metrics.total_revenue >= 0.0);
    }
}
