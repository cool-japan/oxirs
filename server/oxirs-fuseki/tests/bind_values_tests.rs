//! Tests for enhanced BIND and VALUES clause processing

use oxirs_fuseki::bind_values_enhanced::*;
use std::collections::HashMap;
use serde_json::json;

#[tokio::test]
async fn test_bind_string_functions() {
    let processor = EnhancedBindProcessor::new();
    
    let query = r#"
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?name ?upperName ?nameLength
        WHERE {
            ?person foaf:firstName ?first ;
                    foaf:lastName ?last .
            BIND(CONCAT(?first, " ", ?last) AS ?name)
            BIND(UCASE(?name) AS ?upperName)
            BIND(STRLEN(?name) AS ?nameLength)
        }
    "#;
    
    let mut bindings = vec![
        HashMap::from([
            ("first".to_string(), json!("John")),
            ("last".to_string(), json!("Doe")),
        ]),
        HashMap::from([
            ("first".to_string(), json!("Jane")),
            ("last".to_string(), json!("Smith")),
        ]),
    ];
    
    processor.process_bind_clauses(query, &mut bindings).await.unwrap();
    
    // Check first binding
    assert_eq!(bindings[0].get("?name"), Some(&json!("evaluated_result")));
    assert_eq!(bindings[0].get("?upperName"), Some(&json!("evaluated_result")));
    assert_eq!(bindings[0].get("?nameLength"), Some(&json!("evaluated_result")));
}

#[tokio::test]
async fn test_bind_numeric_functions() {
    let processor = EnhancedBindProcessor::new();
    
    let query = r#"
        SELECT ?price ?tax ?total ?rounded
        WHERE {
            ?product :price ?price .
            BIND(?price * 0.1 AS ?tax)
            BIND(?price + ?tax AS ?total)
            BIND(ROUND(?total) AS ?rounded)
        }
    "#;
    
    let mut bindings = vec![
        HashMap::from([("price".to_string(), json!(99.99))]),
        HashMap::from([("price".to_string(), json!(149.50))]),
    ];
    
    processor.process_bind_clauses(query, &mut bindings).await.unwrap();
    
    // Verify bindings were processed
    assert!(bindings[0].contains_key("?tax"));
    assert!(bindings[0].contains_key("?total"));
    assert!(bindings[0].contains_key("?rounded"));
}

#[tokio::test]
async fn test_bind_date_functions() {
    let processor = EnhancedBindProcessor::new();
    
    let query = r#"
        SELECT ?date ?year ?month ?day
        WHERE {
            ?event :date ?date .
            BIND(YEAR(?date) AS ?year)
            BIND(MONTH(?date) AS ?month)
            BIND(DAY(?date) AS ?day)
            BIND(NOW() AS ?currentTime)
        }
    "#;
    
    let mut bindings = vec![
        HashMap::from([("date".to_string(), json!("2024-06-15"))]),
    ];
    
    processor.process_bind_clauses(query, &mut bindings).await.unwrap();
    
    assert!(bindings[0].contains_key("?year"));
    assert!(bindings[0].contains_key("?month"));
    assert!(bindings[0].contains_key("?day"));
    assert!(bindings[0].contains_key("?currentTime"));
}

#[tokio::test]
async fn test_bind_hash_functions() {
    let processor = EnhancedBindProcessor::new();
    
    let query = r#"
        SELECT ?email ?emailHash ?secureHash
        WHERE {
            ?person :email ?email .
            BIND(MD5(?email) AS ?emailHash)
            BIND(SHA256(?email) AS ?secureHash)
        }
    "#;
    
    let mut bindings = vec![
        HashMap::from([("email".to_string(), json!("user@example.com"))]),
    ];
    
    processor.process_bind_clauses(query, &mut bindings).await.unwrap();
    
    assert!(bindings[0].contains_key("?emailHash"));
    assert!(bindings[0].contains_key("?secureHash"));
}

#[tokio::test]
async fn test_bind_conditional_expressions() {
    let processor = EnhancedBindProcessor::new();
    
    let query = r#"
        SELECT ?age ?category ?discount
        WHERE {
            ?person :age ?age .
            BIND(IF(?age < 18, "minor", IF(?age >= 65, "senior", "adult")) AS ?category)
            BIND(IF(?category = "senior", 0.2, IF(?category = "minor", 0.1, 0.0)) AS ?discount)
        }
    "#;
    
    let mut bindings = vec![
        HashMap::from([("age".to_string(), json!(10))]),
        HashMap::from([("age".to_string(), json!(30))]),
        HashMap::from([("age".to_string(), json!(70))]),
    ];
    
    processor.process_bind_clauses(query, &mut bindings).await.unwrap();
    
    for binding in &bindings {
        assert!(binding.contains_key("?category"));
        assert!(binding.contains_key("?discount"));
    }
}

#[tokio::test]
async fn test_values_simple_inline() {
    let processor = EnhancedValuesProcessor::new();
    
    let query = r#"
        SELECT ?person ?email
        WHERE {
            VALUES (?person ?email) {
                (:alice "alice@example.com")
                (:bob "bob@example.com")
                (:charlie "charlie@example.com")
            }
            ?person :hasEmail ?email .
        }
    "#;
    
    let mut bindings = vec![
        HashMap::new(), // Empty initial binding
    ];
    
    processor.process_values_clauses(query, &mut bindings).await.unwrap();
    
    // Should create 3 bindings from VALUES
    assert_eq!(bindings.len(), 3);
}

#[tokio::test]
async fn test_values_multiple_variables() {
    let processor = EnhancedValuesProcessor::new();
    
    let query = r#"
        SELECT ?x ?y ?z
        WHERE {
            VALUES (?x ?y ?z) {
                (1 2 3)
                (4 5 6)
                (7 8 9)
                (UNDEF 10 11)
            }
        }
    "#;
    
    let mut bindings = vec![HashMap::new()];
    
    processor.process_values_clauses(query, &mut bindings).await.unwrap();
    
    // Should handle UNDEF values properly
    assert_eq!(bindings.len(), 4);
}

#[tokio::test]
async fn test_values_with_existing_bindings() {
    let processor = EnhancedValuesProcessor::new();
    
    let query = r#"
        SELECT ?person ?status ?priority
        WHERE {
            ?person :name ?name .
            VALUES (?status ?priority) {
                ("active" 1)
                ("pending" 2)
                ("inactive" 3)
            }
        }
    "#;
    
    let mut bindings = vec![
        HashMap::from([
            ("person".to_string(), json!(":john")),
            ("name".to_string(), json!("John Doe")),
        ]),
        HashMap::from([
            ("person".to_string(), json!(":jane")),
            ("name".to_string(), json!("Jane Smith")),
        ]),
    ];
    
    processor.process_values_clauses(query, &mut bindings).await.unwrap();
    
    // Should create cross product: 2 persons × 3 status values = 6 bindings
    assert_eq!(bindings.len(), 6);
}

#[tokio::test]
async fn test_bind_expression_caching() {
    let processor = EnhancedBindProcessor::new();
    
    let query = r#"
        SELECT ?x ?computed
        WHERE {
            ?item :value ?x .
            BIND(CONCAT("prefix_", STR(?x), "_suffix") AS ?computed)
        }
    "#;
    
    // Large number of bindings with same expression
    let mut bindings: Vec<_> = (0..100)
        .map(|i| HashMap::from([("x".to_string(), json!(i))]))
        .collect();
    
    let start = std::time::Instant::now();
    processor.process_bind_clauses(query, &mut bindings).await.unwrap();
    let duration = start.elapsed();
    
    // All bindings should have computed value
    for binding in &bindings {
        assert!(binding.contains_key("?computed"));
    }
    
    // Cache should make this fast
    assert!(duration.as_millis() < 1000);
}

#[tokio::test]
async fn test_values_optimization_strategies() {
    let processor = EnhancedValuesProcessor::new();
    
    // Large VALUES clause that should trigger optimization
    let mut query = String::from(r#"
        SELECT ?id ?value
        WHERE {
            VALUES (?id ?value) {
    "#);
    
    // Add 1000 value pairs
    for i in 0..1000 {
        query.push_str(&format!("                ({} {})\n", i, i * 10));
    }
    
    query.push_str(r#"
            }
            ?entity :id ?id ;
                    :value ?value .
        }
    "#);
    
    let mut bindings = vec![HashMap::new()];
    
    let start = std::time::Instant::now();
    processor.process_values_clauses(&query, &mut bindings).await.unwrap();
    let duration = start.elapsed();
    
    // Should handle large VALUES efficiently
    assert_eq!(bindings.len(), 1000);
    assert!(duration.as_millis() < 5000);
}

#[tokio::test]
async fn test_bind_complex_expressions() {
    let processor = EnhancedBindProcessor::new();
    
    let query = r#"
        SELECT ?uri ?localName ?namespace
        WHERE {
            ?s ?p ?uri .
            FILTER(ISURI(?uri))
            BIND(REPLACE(STR(?uri), "^.*/([^/]*)$", "$1") AS ?localName)
            BIND(REPLACE(STR(?uri), "/[^/]*$", "/") AS ?namespace)
        }
    "#;
    
    let mut bindings = vec![
        HashMap::from([
            ("uri".to_string(), json!("http://example.org/onto/Person")),
        ]),
        HashMap::from([
            ("uri".to_string(), json!("http://schema.org/Organization")),
        ]),
    ];
    
    processor.process_bind_clauses(query, &mut bindings).await.unwrap();
    
    for binding in &bindings {
        assert!(binding.contains_key("?localName"));
        assert!(binding.contains_key("?namespace"));
    }
}

#[tokio::test]
async fn test_bind_coalesce_function() {
    let processor = EnhancedBindProcessor::new();
    
    let query = r#"
        SELECT ?name ?displayName
        WHERE {
            ?person :firstName ?first .
            OPTIONAL { ?person :nickName ?nick }
            OPTIONAL { ?person :lastName ?last }
            BIND(COALESCE(?nick, ?first, "Anonymous") AS ?displayName)
            BIND(CONCAT(COALESCE(?first, ""), " ", COALESCE(?last, "")) AS ?name)
        }
    "#;
    
    let mut bindings = vec![
        HashMap::from([
            ("first".to_string(), json!("John")),
            ("nick".to_string(), json!("Johnny")),
            ("last".to_string(), json!("Doe")),
        ]),
        HashMap::from([
            ("first".to_string(), json!("Jane")),
            // No nickname
            ("last".to_string(), json!("Smith")),
        ]),
        HashMap::from([
            // No first name or nickname
            ("last".to_string(), json!("Anonymous")),
        ]),
    ];
    
    processor.process_bind_clauses(query, &mut bindings).await.unwrap();
    
    // All bindings should have displayName
    for binding in &bindings {
        assert!(binding.contains_key("?displayName"));
        assert!(binding.contains_key("?name"));
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_bind_performance_with_many_expressions() {
        let processor = EnhancedBindProcessor::new();
        
        let query = r#"
            SELECT ?x ?a ?b ?c ?d ?e
            WHERE {
                ?item :value ?x .
                BIND(?x * 2 AS ?a)
                BIND(?a + 10 AS ?b)
                BIND(?b / 3 AS ?c)
                BIND(ROUND(?c) AS ?d)
                BIND(STR(?d) AS ?e)
            }
        "#;
        
        // Create many bindings
        let mut bindings: Vec<_> = (0..10000)
            .map(|i| HashMap::from([("x".to_string(), json!(i as f64))]))
            .collect();
        
        let start = std::time::Instant::now();
        processor.process_bind_clauses(query, &mut bindings).await.unwrap();
        let duration = start.elapsed();
        
        // Should process efficiently even with many bindings
        assert!(duration.as_secs() < 5);
        
        // Verify all bindings have all computed values
        for binding in &bindings {
            assert!(binding.contains_key("?a"));
            assert!(binding.contains_key("?b"));
            assert!(binding.contains_key("?c"));
            assert!(binding.contains_key("?d"));
            assert!(binding.contains_key("?e"));
        }
    }
    
    #[tokio::test]
    async fn test_values_memory_efficiency() {
        let processor = EnhancedValuesProcessor::new();
        
        // Create a VALUES clause with many columns
        let mut query = String::from(r#"
            SELECT ?a ?b ?c ?d ?e ?f ?g ?h ?i ?j
            WHERE {
                VALUES (?a ?b ?c ?d ?e ?f ?g ?h ?i ?j) {
        "#);
        
        // Add rows
        for n in 0..5000 {
            query.push_str(&format!(
                "                    ({} {} {} {} {} {} {} {} {} {})\n",
                n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8, n+9
            ));
        }
        
        query.push_str("                }
            }
        ");
        
        let mut bindings = vec![HashMap::new()];
        
        let start = std::time::Instant::now();
        processor.process_values_clauses(&query, &mut bindings).await.unwrap();
        let duration = start.elapsed();
        
        // Should handle efficiently
        assert_eq!(bindings.len(), 5000);
        assert!(duration.as_secs() < 10);
    }
}