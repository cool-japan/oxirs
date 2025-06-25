//! Tests for advanced subquery optimization

use oxirs_fuseki::subquery_optimizer::*;
use std::collections::HashMap;

#[tokio::test]
async fn test_exists_to_semi_join_optimization() {
    let optimizer = AdvancedSubqueryOptimizer::new();
    
    let query = r#"
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person
        WHERE {
            ?person foaf:name ?name .
            FILTER EXISTS {
                ?person foaf:knows ?friend .
                ?friend foaf:age ?age .
                FILTER(?age > 18)
            }
        }
    "#;
    
    let result = optimizer.optimize(query).await.unwrap();
    
    assert!(!result.rewrites_applied.is_empty());
    assert!(result.rewrites_applied.iter().any(|r| r.id == "exists_to_semi_join"));
    assert!(result.estimated_cost_reduction > 0.0);
}

#[tokio::test]
async fn test_not_exists_to_anti_join_optimization() {
    let optimizer = AdvancedSubqueryOptimizer::new();
    
    let query = r#"
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person
        WHERE {
            ?person foaf:name ?name .
            FILTER NOT EXISTS {
                ?person foaf:knows ?criminal .
                ?criminal a :Criminal
            }
        }
    "#;
    
    let result = optimizer.optimize(query).await.unwrap();
    
    assert!(!result.rewrites_applied.is_empty());
    assert!(result.rewrites_applied.iter().any(|r| r.id == "not_exists_to_anti_join"));
}

#[tokio::test]
async fn test_simple_subquery_pullup() {
    let optimizer = AdvancedSubqueryOptimizer::new();
    
    let query = r#"
        SELECT ?name ?age
        WHERE {
            {
                SELECT ?person ?name ?age
                WHERE {
                    ?person foaf:name ?name ;
                            foaf:age ?age .
                }
            }
            FILTER(?age > 21)
        }
    "#;
    
    let result = optimizer.optimize(query).await.unwrap();
    
    // Simple subquery should be pulled up
    assert!(result.rewrites_applied.iter().any(|r| r.id == "simple_subquery_pullup"));
}

#[tokio::test]
async fn test_in_subquery_to_join() {
    let optimizer = AdvancedSubqueryOptimizer::new();
    
    let query = r#"
        SELECT ?product
        WHERE {
            ?product :price ?price .
            FILTER(?product IN (
                SELECT ?p WHERE {
                    ?p :category :Electronics ;
                       :rating ?r .
                    FILTER(?r > 4.0)
                }
            ))
        }
    "#;
    
    let result = optimizer.optimize(query).await.unwrap();
    
    assert!(result.rewrites_applied.iter().any(|r| r.id == "in_to_join"));
}

#[tokio::test]
async fn test_correlated_subquery_decorrelation() {
    let optimizer = AdvancedSubqueryOptimizer::new();
    
    let query = r#"
        SELECT ?dept (COUNT(?emp) AS ?count)
        WHERE {
            ?dept a :Department .
            ?emp :department ?dept .
            FILTER EXISTS {
                SELECT ?salary WHERE {
                    ?emp :salary ?salary .
                    FILTER(?salary > (
                        SELECT (AVG(?s) AS ?avg)
                        WHERE {
                            ?e :department ?dept ;
                               :salary ?s .
                        }
                    ))
                }
            }
        }
        GROUP BY ?dept
    "#;
    
    let result = optimizer.optimize(query).await.unwrap();
    
    // Should attempt to decorrelate the nested correlated subquery
    assert!(result.rewrites_applied.iter().any(|r| r.id == "decorrelate_simple"));
}

#[tokio::test]
async fn test_scalar_subquery_optimization() {
    let optimizer = AdvancedSubqueryOptimizer::new();
    
    let query = r#"
        SELECT ?product ?price
        WHERE {
            ?product :price ?price .
            FILTER(?price < (
                SELECT (AVG(?p) AS ?avgPrice)
                WHERE { ?prod :price ?p }
            ))
        }
    "#;
    
    let result = optimizer.optimize(query).await.unwrap();
    
    // Scalar subquery should be optimized
    assert!(result.estimated_cost_reduction > 0.0);
}

#[tokio::test]
async fn test_multiple_subquery_optimization() {
    let optimizer = AdvancedSubqueryOptimizer::new();
    
    let query = r#"
        SELECT ?person ?totalSpent
        WHERE {
            ?person foaf:name ?name .
            
            # First subquery
            FILTER EXISTS {
                ?person :hasPurchased ?product .
                ?product :category :Electronics
            }
            
            # Second subquery
            FILTER NOT EXISTS {
                ?person :hasReturned ?item .
                ?item :reason :Defective
            }
            
            # Scalar subquery
            BIND((
                SELECT (SUM(?amount) AS ?total)
                WHERE {
                    ?person :purchase ?p .
                    ?p :amount ?amount
                }
            ) AS ?totalSpent)
        }
    "#;
    
    let result = optimizer.optimize(query).await.unwrap();
    
    // Should optimize multiple subqueries
    assert!(result.rewrites_applied.len() >= 2);
}

#[tokio::test]
async fn test_materialization_manager() {
    let manager = MaterializationManager::new();
    
    let subquery = SubqueryInfo {
        id: "test_sq".to_string(),
        query_text: "SELECT * WHERE { ?s ?p ?o }".to_string(),
        subquery_type: SubqueryType::Scalar,
        is_correlated: false,
        outer_vars: vec![],
        estimated_size: 1000,
        estimated_selectivity: 0.1,
        estimated_cost: 150.0,
        filter_count: 0,
        join_count: 0,
        outer_cardinality: 1,
        dependencies: vec![],
    };
    
    // First execution - should execute and potentially cache
    let results1 = manager.get_or_materialize(&subquery, || {
        Ok(vec![
            HashMap::from([("s".to_string(), serde_json::json!("subject1"))]),
            HashMap::from([("s".to_string(), serde_json::json!("subject2"))]),
        ])
    }).await.unwrap();
    
    assert_eq!(results1.len(), 2);
    
    // Second execution - should hit cache
    let results2 = manager.get_or_materialize(&subquery, || {
        // This shouldn't be called if cache hit
        panic!("Should not execute - should use cache");
    }).await.unwrap();
    
    assert_eq!(results2.len(), 2);
}

#[tokio::test]
async fn test_execution_strategy_selection() {
    let selector = ExecutionStrategySelector::new();
    
    // EXISTS subquery should use semi-join
    let exists_subquery = SubqueryInfo {
        id: "exists_sq".to_string(),
        query_text: "?s ?p ?o".to_string(),
        subquery_type: SubqueryType::Exists,
        is_correlated: false,
        outer_vars: vec![],
        estimated_size: 100,
        estimated_selectivity: 0.5,
        estimated_cost: 50.0,
        filter_count: 0,
        join_count: 0,
        outer_cardinality: 1,
        dependencies: vec![],
    };
    
    let strategy = selector.select_strategy(&exists_subquery).unwrap();
    match strategy {
        ExecutionStrategy::SemiJoin => {},
        _ => panic!("Expected SemiJoin strategy for EXISTS"),
    }
    
    // Correlated scalar subquery should use correlated execution
    let correlated_subquery = SubqueryInfo {
        id: "corr_sq".to_string(),
        query_text: "SELECT ?x WHERE { ?x :relatedTo ?outer }".to_string(),
        subquery_type: SubqueryType::Scalar,
        is_correlated: true,
        outer_vars: vec!["?outer".to_string()],
        estimated_size: 10,
        estimated_selectivity: 0.1,
        estimated_cost: 20.0,
        filter_count: 0,
        join_count: 1,
        outer_cardinality: 100,
        dependencies: vec![],
    };
    
    let strategy = selector.select_strategy(&correlated_subquery).unwrap();
    match strategy {
        ExecutionStrategy::CorrelatedExecution => {},
        _ => panic!("Expected CorrelatedExecution strategy for correlated subquery"),
    }
}

#[tokio::test]
async fn test_cost_estimation() {
    let estimator = SubqueryCostEstimator::new();
    
    let subquery = SubqueryInfo {
        id: "cost_sq".to_string(),
        query_text: "SELECT * WHERE { ?s ?p ?o . ?o ?p2 ?o2 }".to_string(),
        subquery_type: SubqueryType::From,
        is_correlated: false,
        outer_vars: vec![],
        estimated_size: 1000,
        estimated_selectivity: 0.3,
        estimated_cost: 100.0,
        filter_count: 2,
        join_count: 1,
        outer_cardinality: 1,
        dependencies: vec![],
    };
    
    // Test different execution strategies
    let materialize_cost = estimator.estimate_cost(&subquery, &ExecutionStrategy::MaterializeOnce).await.unwrap();
    let join_cost = estimator.estimate_cost(&subquery, &ExecutionStrategy::JoinConversion).await.unwrap();
    let semi_join_cost = estimator.estimate_cost(&subquery, &ExecutionStrategy::SemiJoin).await.unwrap();
    
    // Semi-join should be cheaper than materialize for this case
    assert!(semi_join_cost < materialize_cost);
    // Join conversion should also be reasonably efficient
    assert!(join_cost < materialize_cost);
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_complex_query_optimization() {
        let optimizer = AdvancedSubqueryOptimizer::new();
        
        let complex_query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/>
            
            SELECT ?person ?name ?friendCount ?avgAge
            WHERE {
                ?person foaf:name ?name .
                
                # Correlated scalar subquery
                {
                    SELECT (COUNT(?friend) AS ?friendCount)
                    WHERE {
                        ?person foaf:knows ?friend .
                        FILTER EXISTS {
                            ?friend foaf:age ?age .
                            FILTER(?age >= 18)
                        }
                    }
                }
                
                # Another scalar subquery
                {
                    SELECT (AVG(?friendAge) AS ?avgAge)
                    WHERE {
                        ?person foaf:knows ?f .
                        ?f foaf:age ?friendAge
                    }
                }
                
                # Filter with subquery
                FILTER(?friendCount > (
                    SELECT (AVG(?c) AS ?avgCount)
                    WHERE {
                        ?p foaf:name ?n .
                        {
                            SELECT ?p (COUNT(?f) AS ?c)
                            WHERE { ?p foaf:knows ?f }
                            GROUP BY ?p
                        }
                    }
                ))
                
                # NOT EXISTS optimization
                FILTER NOT EXISTS {
                    ?person ex:bannedFrom ?location .
                    ?location ex:type ex:PublicSpace
                }
            }
            ORDER BY DESC(?friendCount)
            LIMIT 10
        "#;
        
        let result = optimizer.optimize(complex_query).await.unwrap();
        
        // Should apply multiple optimizations
        assert!(result.rewrites_applied.len() > 1);
        assert!(result.estimated_cost_reduction > 0.2); // At least 20% improvement
        
        // Check that specific optimizations were applied
        let rewrite_ids: Vec<_> = result.rewrites_applied.iter().map(|r| &r.id).collect();
        assert!(rewrite_ids.contains(&&"exists_to_semi_join".to_string()));
        assert!(rewrite_ids.contains(&&"not_exists_to_anti_join".to_string()));
    }
    
    #[tokio::test]
    async fn test_optimization_with_sparql_star() {
        let optimizer = AdvancedSubqueryOptimizer::new();
        
        // Query with SPARQL-star features and subqueries
        let query = r#"
            PREFIX ex: <http://example.org/>
            
            SELECT ?stmt ?confidence
            WHERE {
                ?stmt a rdf:Statement .
                << ?s ex:claims ?o >> ex:confidence ?confidence .
                
                FILTER EXISTS {
                    SELECT ?source WHERE {
                        << ?s ex:claims ?o >> ex:source ?source .
                        ?source ex:reliability ?r .
                        FILTER(?r > 0.8)
                    }
                }
                
                FILTER(?confidence > (
                    SELECT (AVG(?c) AS ?avgConf)
                    WHERE {
                        << ?subj ex:claims ?obj >> ex:confidence ?c
                    }
                ))
            }
        "#;
        
        let result = optimizer.optimize(query).await;
        
        // Should handle SPARQL-star with subqueries
        assert!(result.is_ok());
        if let Ok(opt) = result {
            assert!(!opt.rewrites_applied.is_empty());
        }
    }
}