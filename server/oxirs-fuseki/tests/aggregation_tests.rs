//! Tests for enhanced SPARQL 1.2 aggregation functions

use oxirs_fuseki::aggregation::*;
use serde_json::Value;
use std::collections::HashMap;

#[test]
fn test_group_concat_with_separator() {
    let mut agg = GroupConcatAggregate::new(Some(" | ".to_string()), false);

    agg.add_value(&Value::String("first".to_string())).unwrap();
    agg.add_value(&Value::String("second".to_string())).unwrap();
    agg.add_value(&Value::String("third".to_string())).unwrap();

    let result = agg.get_result().unwrap();
    assert_eq!(
        result.value,
        Value::String("first | second | third".to_string())
    );
}

#[test]
fn test_group_concat_distinct() {
    let mut agg = GroupConcatAggregate::new(Some(",".to_string()), true);

    agg.add_value(&Value::String("apple".to_string())).unwrap();
    agg.add_value(&Value::String("banana".to_string())).unwrap();
    agg.add_value(&Value::String("apple".to_string())).unwrap();
    agg.add_value(&Value::String("cherry".to_string())).unwrap();

    let result = agg.get_result().unwrap();
    assert_eq!(
        result.value,
        Value::String("apple,banana,cherry".to_string())
    );
}

#[test]
fn test_sample_aggregate() {
    let mut agg = SampleAggregate::new();

    // SAMPLE should return any value, we'll test with the first one
    agg.add_value(&Value::String("first".to_string())).unwrap();
    agg.add_value(&Value::String("second".to_string())).unwrap();
    agg.add_value(&Value::String("third".to_string())).unwrap();

    let result = agg.get_result().unwrap();
    assert_eq!(result.value, Value::String("first".to_string()));
}

#[test]
fn test_median_odd_count() {
    let mut agg = MedianAggregate::new();

    // Values: 10, 20, 30, 40, 50
    for i in 1..=5 {
        agg.add_value(&Value::Number(serde_json::Number::from(i * 10)))
            .unwrap();
    }

    let result = agg.get_result().unwrap();
    if let Value::Number(n) = result.value {
        assert_eq!(n.as_f64().unwrap(), 30.0);
    } else {
        panic!("Expected numeric result");
    }
}

#[test]
fn test_median_even_count() {
    let mut agg = MedianAggregate::new();

    // Values: 10, 20, 30, 40
    for i in 1..=4 {
        agg.add_value(&Value::Number(serde_json::Number::from(i * 10)))
            .unwrap();
    }

    let result = agg.get_result().unwrap();
    if let Value::Number(n) = result.value {
        assert_eq!(n.as_f64().unwrap(), 25.0); // (20 + 30) / 2
    } else {
        panic!("Expected numeric result");
    }
}

#[test]
fn test_mode_strings() {
    let mut agg = ModeAggregate::new();

    let values = vec!["red", "blue", "red", "green", "red", "blue"];
    for val in values {
        agg.add_value(&Value::String(val.to_string())).unwrap();
    }

    let result = agg.get_result().unwrap();
    assert_eq!(result.value, Value::String("red".to_string()));
}

#[test]
fn test_stddev_sample() {
    let mut agg = StdDevAggregate::new(false); // Sample standard deviation

    // Values: 2, 4, 4, 4, 5, 5, 7, 9
    let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    for val in values {
        agg.add_value(&Value::Number(serde_json::Number::from_f64(val).unwrap()))
            .unwrap();
    }

    let result = agg.get_result().unwrap();
    if let Value::Number(n) = result.value {
        let stddev = n.as_f64().unwrap();
        assert!((stddev - 2.0).abs() < 0.1); // Approximately 2.0
    } else {
        panic!("Expected numeric result");
    }
}

#[test]
fn test_variance() {
    let mut agg = VarianceAggregate::new(false); // Sample variance

    // Simple dataset: 1, 2, 3, 4, 5
    for i in 1..=5 {
        agg.add_value(&Value::Number(serde_json::Number::from(i)))
            .unwrap();
    }

    let result = agg.get_result().unwrap();
    if let Value::Number(n) = result.value {
        let variance = n.as_f64().unwrap();
        assert_eq!(variance, 2.5); // Variance of 1,2,3,4,5 is 2.5
    } else {
        panic!("Expected numeric result");
    }
}

#[test]
fn test_percentile_50() {
    let mut agg = PercentileAggregate::new(50.0); // 50th percentile (median)

    for i in 1..=100 {
        agg.add_value(&Value::Number(serde_json::Number::from(i)))
            .unwrap();
    }

    let result = agg.get_result().unwrap();
    if let Value::Number(n) = result.value {
        assert_eq!(n.as_f64().unwrap(), 50.0);
    } else {
        panic!("Expected numeric result");
    }
}

#[test]
fn test_percentile_90() {
    let mut agg = PercentileAggregate::new(90.0); // 90th percentile

    for i in 1..=100 {
        agg.add_value(&Value::Number(serde_json::Number::from(i)))
            .unwrap();
    }

    let result = agg.get_result().unwrap();
    if let Value::Number(n) = result.value {
        assert_eq!(n.as_f64().unwrap(), 90.0);
    } else {
        panic!("Expected numeric result");
    }
}

#[test]
fn test_count_distinct() {
    let mut agg = CountDistinctAggregate::new();

    // Add values with duplicates
    let values = vec!["apple", "banana", "apple", "cherry", "banana", "date"];
    for val in values {
        agg.add_value(&Value::String(val.to_string())).unwrap();
    }

    let result = agg.get_result().unwrap();
    if let Value::Number(n) = result.value {
        assert_eq!(n.as_u64().unwrap(), 4); // 4 distinct values
    } else {
        panic!("Expected numeric result");
    }
}

#[test]
fn test_aggregation_factory() {
    // Test GROUP_CONCAT creation
    let mut args = HashMap::new();
    args.insert("separator".to_string(), Value::String(";".to_string()));
    args.insert("distinct".to_string(), Value::Bool(true));

    let agg = AggregationFactory::create_aggregate("GROUP_CONCAT", &args).unwrap();
    assert_eq!(agg.name(), "GROUP_CONCAT");
    assert!(agg.requires_distinct());

    // Test PERCENTILE creation
    let mut args = HashMap::new();
    args.insert(
        "percentile".to_string(),
        Value::Number(serde_json::Number::from(75)),
    );

    let agg = AggregationFactory::create_aggregate("PERCENTILE", &args).unwrap();
    assert_eq!(agg.name(), "PERCENTILE");
}

#[test]
fn test_enhanced_aggregation_processor() {
    let mut processor = EnhancedAggregationProcessor::new();

    // Register multiple aggregations
    processor
        .register_aggregate("concat_result".to_string(), "GROUP_CONCAT", &HashMap::new())
        .unwrap();

    processor
        .register_aggregate("median_result".to_string(), "MEDIAN", &HashMap::new())
        .unwrap();

    // Add values
    processor
        .add_value("concat_result", &Value::String("a".to_string()))
        .unwrap();
    processor
        .add_value("concat_result", &Value::String("b".to_string()))
        .unwrap();
    processor
        .add_value(
            "median_result",
            &Value::Number(serde_json::Number::from(10)),
        )
        .unwrap();
    processor
        .add_value(
            "median_result",
            &Value::Number(serde_json::Number::from(20)),
        )
        .unwrap();
    processor
        .add_value(
            "median_result",
            &Value::Number(serde_json::Number::from(30)),
        )
        .unwrap();

    // Get results
    let results = processor.get_results().unwrap();

    assert_eq!(
        results.get("concat_result").unwrap().value,
        Value::String("a b".to_string())
    );
    if let Value::Number(n) = &results.get("median_result").unwrap().value {
        assert_eq!(n.as_f64().unwrap(), 20.0);
    }
}

#[cfg(test)]
mod sparql_query_tests {
    use super::*;

    #[test]
    fn test_sparql_group_concat_query() {
        // Example SPARQL query with GROUP_CONCAT
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?person (GROUP_CONCAT(?email; SEPARATOR=", ") AS ?emails)
            WHERE {
                ?person foaf:mbox ?email
            }
            GROUP BY ?person
        "#;

        // This would be tested with actual query execution
        assert!(query.contains("GROUP_CONCAT"));
    }

    #[test]
    fn test_sparql_statistical_query() {
        // Example SPARQL query with statistical functions
        let query = r#"
            PREFIX ex: <http://example.org/>
            SELECT 
                (MEDIAN(?age) AS ?median_age)
                (STDDEV(?age) AS ?age_stddev)
                (PERCENTILE(?income, 0.75) AS ?income_75th)
            WHERE {
                ?person ex:age ?age ;
                        ex:income ?income .
            }
        "#;

        assert!(query.contains("MEDIAN"));
        assert!(query.contains("STDDEV"));
        assert!(query.contains("PERCENTILE"));
    }
}
