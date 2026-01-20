//! Federation and Aggregates Demo
//!
//! This example demonstrates the new features added to oxirs-arq:
//! 1. HTTP-based federation with SPARQL endpoints
//! 2. SPARQL service description query for endpoint discovery
//! 3. Aggregate support in interactive query builder
//! 4. CardinalityEstimator integration with JIT compiler
//!
//! Run with: cargo run --example federation_and_aggregates_demo --all-features

use anyhow::Result;
use oxirs_arq::algebra::{Aggregate, Algebra, Expression, Variable};
use oxirs_arq::federation::{
    EndpointCapabilities, EndpointDiscovery, FederationConfig, FederationExecutor,
};
use oxirs_arq::interactive_query_builder::{InteractiveQueryBuilder, PatternBuilder};
use oxirs_arq::jit_compiler::{JitCompilerConfig, QueryJitCompiler};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== OxiRS-ARQ: Federation & Aggregates Demo ===\n");

    // 1. Demonstrate Federation Configuration
    demo_federation_config();

    // 2. Demonstrate Endpoint Discovery
    demo_endpoint_discovery();

    // 3. Demonstrate Interactive Query Builder with Aggregates
    demo_aggregate_queries()?;

    // 4. Demonstrate JIT Compiler with Cardinality Estimation
    demo_jit_compilation()?;

    // 5. Demonstrate Federation Executor
    demo_federation_executor().await?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn demo_federation_config() {
    println!("1. Federation Configuration");
    println!("   Creating federation config with custom settings...");

    let config = FederationConfig {
        max_concurrent_requests: 16,
        max_retries: 3,
        enable_caching: true,
        enable_health_monitoring: true,
        ..Default::default()
    };

    println!(
        "   ✓ Max concurrent requests: {}",
        config.max_concurrent_requests
    );
    println!("   ✓ Max retries: {}", config.max_retries);
    println!("   ✓ Caching enabled: {}", config.enable_caching);
    println!(
        "   ✓ Health monitoring: {}\n",
        config.enable_health_monitoring
    );
}

fn demo_endpoint_discovery() {
    println!("2. Endpoint Discovery");
    println!("   Registering SPARQL endpoints...");

    let discovery = EndpointDiscovery::new();

    // Register DBpedia endpoint
    discovery.register_endpoint(EndpointCapabilities {
        endpoint: "https://dbpedia.org/sparql".to_string(),
        sparql_version: "1.1".to_string(),
        result_formats: vec![
            "application/sparql-results+json".to_string(),
            "application/sparql-results+xml".to_string(),
        ],
        max_query_complexity: Some(10000),
        supports_federation: true,
        supports_rdf_star: false,
        named_graphs: vec!["http://dbpedia.org".to_string()],
    });

    // Register Wikidata endpoint
    discovery.register_endpoint(EndpointCapabilities {
        endpoint: "https://query.wikidata.org/sparql".to_string(),
        sparql_version: "1.1".to_string(),
        result_formats: vec!["application/sparql-results+json".to_string()],
        max_query_complexity: Some(5000),
        supports_federation: false,
        supports_rdf_star: false,
        named_graphs: vec![],
    });

    println!("   ✓ Registered 2 endpoints");

    // Find endpoints with federation support
    use oxirs_arq::federation::EndpointCriteria;
    let federated = discovery.find_endpoints(EndpointCriteria {
        supports_federation: Some(true),
        ..Default::default()
    });

    println!(
        "   ✓ Found {} endpoints with federation support\n",
        federated.len()
    );
}

fn demo_aggregate_queries() -> Result<()> {
    println!("3. Interactive Query Builder with Aggregates");
    println!("   Building queries with COUNT, SUM, AVG aggregates...\n");

    // Example 1: COUNT aggregate
    println!("   Example 1: COUNT aggregate");
    let count_query = InteractiveQueryBuilder::new()
        .select_all()
        .prefix("foaf", "http://xmlns.com/foaf/0.1/")
        .r#where(
            PatternBuilder::new()
                .subject_var("person")?
                .predicate_iri("http://xmlns.com/foaf/0.1/name")?
                .object_var("name")?,
        )
        .count("totalPeople", None, false)?
        .build_string()?;

    println!("   Query:");
    for line in count_query.lines() {
        println!("     {}", line);
    }

    // Example 2: Multiple aggregates (COUNT, SUM, AVG)
    println!("\n   Example 2: Multiple aggregates (COUNT, SUM, AVG)");
    let multi_agg_algebra = InteractiveQueryBuilder::new()
        .select_all()
        .r#where(
            PatternBuilder::new()
                .subject_var("product")?
                .predicate_var("p")?
                .object_var("o")?,
        )
        .count("productCount", None, false)?
        .sum(
            "totalPrice",
            Expression::Variable(Variable::new_unchecked("price")),
            false,
        )?
        .avg(
            "avgPrice",
            Expression::Variable(Variable::new_unchecked("price")),
            false,
        )?
        .build_algebra()?;

    match &multi_agg_algebra {
        Algebra::Group { aggregates, .. } => {
            println!("   ✓ Generated query with {} aggregates:", aggregates.len());
            for (var, agg) in aggregates {
                let agg_type = match agg {
                    Aggregate::Count { .. } => "COUNT",
                    Aggregate::Sum { .. } => "SUM",
                    Aggregate::Avg { .. } => "AVG",
                    Aggregate::Min { .. } => "MIN",
                    Aggregate::Max { .. } => "MAX",
                    Aggregate::Sample { .. } => "SAMPLE",
                    Aggregate::GroupConcat { .. } => "GROUP_CONCAT",
                };
                println!("     - {} ({})", var.name(), agg_type);
            }
        }
        _ => println!("   Note: Aggregates wrapped in query algebra"),
    }

    // Example 3: GROUP_CONCAT
    println!("\n   Example 3: GROUP_CONCAT aggregate");
    let _concat_algebra = InteractiveQueryBuilder::new()
        .select_all()
        .r#where(
            PatternBuilder::new()
                .subject_var("author")?
                .predicate_var("p")?
                .object_var("o")?,
        )
        .group_concat(
            "allBooks",
            Expression::Variable(Variable::new_unchecked("bookTitle")),
            false,
            Some("; ".to_string()),
        )?
        .build_algebra()?;

    println!("   ✓ GROUP_CONCAT with custom separator ';'");
    println!();

    Ok(())
}

fn demo_jit_compilation() -> Result<()> {
    println!("4. JIT Compiler with Cardinality Estimation");
    println!("   Compiling SPARQL query with integrated cardinality estimator...\n");

    // Create JIT compiler with cardinality estimation
    let config = JitCompilerConfig::default();
    let mut compiler = QueryJitCompiler::new(config)?;

    // Create a simple query
    let query = InteractiveQueryBuilder::new()
        .select(vec!["s", "p", "o"])?
        .r#where(
            PatternBuilder::new()
                .subject_var("s")?
                .predicate_var("p")?
                .object_var("o")?,
        )
        .limit(100)
        .build_algebra()?;

    // Compile the query
    println!("   Compiling query algebra...");
    let compiled = compiler.compile(&query)?;

    println!("   ✓ Query compiled successfully");
    println!("   ✓ Query ID: {}", compiled.id);
    println!(
        "   ✓ Compilation time: {:?}",
        compiled.compiled_at.elapsed()
    );
    println!(
        "   ✓ Cardinality estimation integrated in {} operations",
        compiled.plan.operations.len()
    );
    println!();

    Ok(())
}

async fn demo_federation_executor() -> Result<()> {
    println!("5. Federation Executor");
    println!("   Creating federation executor with HTTP support...\n");

    let config = FederationConfig::default();
    let executor = FederationExecutor::new(config);

    // Get statistics
    let stats = executor.statistics();
    println!("   Initial Statistics:");
    println!("     Total requests: {}", stats.total_requests);
    println!("     Active endpoints: {}", stats.active_endpoints);
    println!("     Healthy endpoints: {}", stats.healthy_endpoints);

    // Demonstrate query decomposition
    use oxirs_arq::algebra::{Term, TriplePattern};
    let patterns = vec![TriplePattern {
        subject: Term::Variable(Variable::new_unchecked("s")),
        predicate: Term::Variable(Variable::new_unchecked("p")),
        object: Term::Variable(Variable::new_unchecked("o")),
    }];

    let _bgp_query = Algebra::Bgp(patterns);

    // Note: In a real scenario, you would wrap this in a SERVICE pattern
    // and use execute_service() to make actual HTTP requests to SPARQL endpoints
    println!("\n   ✓ Federation executor ready for HTTP SPARQL queries");
    println!("   ✓ Supports:");
    println!("     - HTTP POST with SPARQL query strings");
    println!("     - SPARQL JSON results parsing");
    println!("     - Connection pooling and retry logic");
    println!("     - Endpoint health monitoring");
    println!("     - Query result caching");
    println!("     - W3C SPARQL Service Description protocol");

    Ok(())
}
