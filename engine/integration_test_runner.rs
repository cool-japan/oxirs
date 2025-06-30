use std::time::Duration;

mod integration_tests;
use integration_tests::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting OxiRS Engine Integration Tests...\n");

    // Create test suite with custom configuration
    let config = IntegrationTestConfig {
        parallel_execution: true,
        test_timeout: Duration::from_secs(30),
        memory_threshold_mb: 512,
        performance_thresholds: PerformanceThresholds {
            sparql_query_max_ms: 500,
            shacl_validation_max_ms: 1000,
            vector_search_max_ms: 50,
            rule_inference_max_ms: 250,
            rdf_star_max_ms: 100,
        },
        verbose_logging: true,
    };

    let mut suite = OxirsIntegrationTestSuite::with_config(config);

    // Run all tests
    let report = suite.run_all_tests()?;
    
    println!("\n📊 Integration Test Report");
    println!("==========================");
    println!("Total Tests: {}", report.summary.total_tests);
    println!("Passed: {}", report.summary.passed_tests);
    println!("Failed: {}", report.summary.failed_tests);
    println!("Success Rate: {:.1}%", report.summary.success_rate);
    println!("Total Execution Time: {:?}", report.summary.total_execution_time);
    println!("Performance Score: {:.1}/100", report.summary.performance_score);

    println!("\n📈 Module Results:");
    for (module, results) in &report.module_results {
        println!("  {}: {}/{} tests passed ({:.1}% success, {:.1} avg score)", 
            module, 
            results.passed_tests, 
            results.total_tests,
            (results.passed_tests as f64 / results.total_tests as f64) * 100.0,
            results.average_performance_score
        );
    }

    println!("\n🔗 Integration Coverage: {:.1}%", report.integration_analysis.coverage_percentage);
    println!("Integration Points Tested: {}", report.integration_analysis.integration_points_tested.len());

    if !report.integration_analysis.missing_integration_points.is_empty() {
        println!("\n⚠️  Missing Integration Points:");
        for point in &report.integration_analysis.missing_integration_points {
            println!("  - {}", point);
        }
    }

    if !report.recommendations.is_empty() {
        println!("\n💡 Recommendations:");
        for rec in &report.recommendations {
            println!("  {} [{}]: {}", 
                match rec.priority {
                    RecommendationPriority::Critical => "🔴",
                    RecommendationPriority::High => "🟠",
                    RecommendationPriority::Medium => "🟡",
                    RecommendationPriority::Low => "🟢",
                },
                rec.category,
                rec.description
            );
        }
    }

    println!("\n✅ Integration tests completed successfully!");
    Ok(())
}