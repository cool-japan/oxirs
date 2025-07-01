# OxiRS-Star Migration Guide

[![Documentation](https://docs.rs/oxirs-star/badge.svg)](https://docs.rs/oxirs-star)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**Comprehensive migration guide for moving from other RDF stores to OxiRS-Star, including automated migration tools, compatibility layers, and best practices.**

## Table of Contents

- [Migration Overview](#migration-overview)
- [From Apache Jena](#from-apache-jena)
- [From Oxigraph](#from-oxigraph)
- [From Virtuoso](#from-virtuoso)
- [From AllegroGraph](#from-allegrograph)
- [From Stardog](#from-stardog)
- [From RDF4J](#from-rdf4j)
- [Migration Tools](#migration-tools)
- [Data Conversion](#data-conversion)
- [Query Migration](#query-migration)
- [Application Integration](#application-integration)
- [Performance Validation](#performance-validation)
- [Rollback Strategies](#rollback-strategies)

## Migration Overview

### Pre-Migration Assessment

```rust
use oxirs_star::migration::{MigrationAssessment, SourceStore, CompatibilityAnalyzer};

// Assess current RDF store for migration readiness
let assessment = MigrationAssessment::new()
    .analyze_source_store(SourceStore::ApacheJena)
    .scan_data_directory("/path/to/jena/data")
    .analyze_query_logs("/path/to/query.log")
    .check_application_dependencies();

let report = assessment.generate_report()?;

println!("🔍 Migration Assessment Report");
println!("├─ Source store: {}", report.source_store);
println!("├─ Data size: {} GB", report.data_size_gb);
println!("├─ Triple count: {} million", report.triple_count / 1_000_000);
println!("├─ RDF-star usage: {}%", report.rdf_star_usage_percent);
println!("├─ Custom extensions: {}", report.custom_extensions.len());
println!("├─ Migration complexity: {:?}", report.complexity);
println!("└─ Estimated migration time: {:?}", report.estimated_duration);

// Compatibility analysis
let compatibility = CompatibilityAnalyzer::new()
    .analyze_queries(&report.query_patterns)
    .analyze_features(&report.used_features)
    .analyze_apis(&report.application_apis);

println!("\n📊 Compatibility Analysis:");
println!("├─ SPARQL queries: {:.1}% compatible", compatibility.sparql_compatibility * 100.0);
println!("├─ Features: {:.1}% supported", compatibility.feature_compatibility * 100.0);
println!("├─ APIs: {:.1}% compatible", compatibility.api_compatibility * 100.0);
println!("└─ Migration risk: {:?}", compatibility.risk_level);

if compatibility.risk_level == RiskLevel::High {
    println!("⚠️  High-risk migration detected. Consider gradual migration strategy.");
}
```

### Migration Strategy Selection

```rust
use oxirs_star::migration::{MigrationStrategy, MigrationPlanner};

// Choose appropriate migration strategy
let strategy = match (report.data_size_gb, report.complexity, compatibility.risk_level) {
    (size, _, _) if size < 1.0 => MigrationStrategy::DirectMigration,
    (_, Complexity::Low, RiskLevel::Low) => MigrationStrategy::DirectMigration,
    (_, Complexity::Medium, _) => MigrationStrategy::GradualMigration,
    (_, Complexity::High, _) | (_, _, RiskLevel::High) => MigrationStrategy::HybridApproach,
    _ => MigrationStrategy::PhasedMigration,
};

println!("🎯 Recommended Migration Strategy: {:?}", strategy);

let planner = MigrationPlanner::with_strategy(strategy);
let plan = planner.create_migration_plan(&report)?;

println!("\n📋 Migration Plan:");
for (i, phase) in plan.phases.iter().enumerate() {
    println!("  Phase {}: {} (Duration: {:?})", i + 1, phase.name, phase.estimated_duration);
    for task in &phase.tasks {
        println!("    - {}", task.description);
    }
}
```

## From Apache Jena

### Jena TDB Migration

```rust
use oxirs_star::migration::jena::{JenaTdbMigrator, JenaConfig};

// Configure Jena TDB migration
let jena_config = JenaConfig {
    tdb_directory: "/path/to/jena/tdb".to_string(),
    include_named_graphs: true,
    preserve_blank_nodes: true,
    convert_reification_to_rdf_star: true,
    batch_size: 100_000,
    parallel_processing: true,
};

let migrator = JenaTdbMigrator::with_config(jena_config);

// Pre-migration validation
let validation_result = migrator.validate_source_data()?;
if !validation_result.is_valid {
    println!("❌ Source data validation failed:");
    for issue in validation_result.issues {
        println!("  - {}: {}", issue.severity, issue.description);
    }
    return Err("Source data validation failed".into());
}

// Execute migration
println!("🚀 Starting Jena TDB migration...");
let migration_result = migrator.migrate_to_oxirs_star(
    &mut target_store,
    |progress| {
        println!("Migration progress: {:.1}% ({} triples processed)", 
            progress.percentage, progress.triples_processed);
    }
)?;

println!("✅ Migration completed successfully!");
println!("├─ Triples migrated: {}", migration_result.triples_migrated);
println!("├─ Named graphs: {}", migration_result.named_graphs_migrated);
println!("├─ RDF-star triples: {}", migration_result.rdf_star_triples_created);
println!("├─ Duration: {:?}", migration_result.total_duration);
println!("└─ Errors: {}", migration_result.errors.len());

// Handle migration errors
if !migration_result.errors.is_empty() {
    println!("⚠️  Migration completed with errors:");
    for error in migration_result.errors {
        println!("  Line {}: {}", error.line_number, error.message);
    }
}
```

### Jena Fuseki Migration

```rust
use oxirs_star::migration::jena::{FusekiMigrator, FusekiEndpoint};

// Migrate from Jena Fuseki via SPARQL endpoint
let fuseki_endpoint = FusekiEndpoint {
    url: "http://localhost:3030/dataset/sparql".to_string(),
    update_url: Some("http://localhost:3030/dataset/update".to_string()),
    username: Some("admin".to_string()),
    password: Some("password".to_string()),
    timeout: std::time::Duration::from_secs(300),
};

let fuseki_migrator = FusekiMigrator::new(fuseki_endpoint);

// Discover datasets
let datasets = fuseki_migrator.discover_datasets().await?;
println!("Found {} datasets in Fuseki:", datasets.len());
for dataset in &datasets {
    println!("  - {}: {} triples", dataset.name, dataset.triple_count);
}

// Migrate specific dataset
let dataset_to_migrate = "my_dataset";
println!("🔄 Migrating dataset: {}", dataset_to_migrate);

let migration_config = FusekiMigrationConfig {
    chunk_size: 10_000,
    max_concurrent_requests: 5,
    include_inferred_triples: false,
    preserve_graph_structure: true,
};

let result = fuseki_migrator.migrate_dataset(
    dataset_to_migrate,
    &mut target_store,
    migration_config,
).await?;

println!("✅ Fuseki dataset migration completed!");
println!("  Data migrated: {} GB", result.data_size_gb);
println!("  Migration rate: {:.0} triples/sec", result.migration_rate);
```

### Jena API Migration

```rust
use oxirs_star::migration::jena::{JenaApiMigrator, ApiCompatibilityLayer};

// Create compatibility layer for existing Jena code
let compat_layer = ApiCompatibilityLayer::new();

// Map common Jena patterns to OxiRS-Star
impl ApiCompatibilityLayer {
    // Jena Model → OxiRS-Star Store
    pub fn migrate_model_operations(&self, jena_code: &str) -> Result<String> {
        let oxirs_code = jena_code
            .replace("Model model = ModelFactory.createDefaultModel()", 
                     "let mut store = StarStore::new()")
            .replace("model.add(stmt)", 
                     "store.insert(&triple)?")
            .replace("model.listStatements()", 
                     "store.iter()")
            .replace("Resource resource = model.createResource(uri)", 
                     "let resource = StarTerm::iri(uri)?");
        
        Ok(oxirs_code)
    }
    
    // Jena Query → SPARQL-star
    pub fn migrate_query_operations(&self, jena_query: &str) -> Result<String> {
        let sparql_star = if jena_query.contains("ReificationVocab") {
            // Convert reification to RDF-star
            self.convert_reification_to_star(jena_query)?
        } else {
            jena_query.to_string()
        };
        
        Ok(sparql_star)
    }
}

// Automated code migration
let migrator = JenaApiMigrator::new(compat_layer);
let java_source_dir = "/path/to/java/src";
let rust_output_dir = "/path/to/rust/src";

let migration_stats = migrator.migrate_source_code(java_source_dir, rust_output_dir)?;

println!("📝 Code Migration Results:");
println!("├─ Java files processed: {}", migration_stats.files_processed);
println!("├─ Rust files generated: {}", migration_stats.files_generated);
println!("├─ API calls migrated: {}", migration_stats.api_calls_migrated);
println!("├─ Manual review needed: {}", migration_stats.manual_review_items);
println!("└─ Migration confidence: {:.1}%", migration_stats.confidence_percent);
```

## From Oxigraph

### Oxigraph Store Migration

```rust
use oxirs_star::migration::oxigraph::{OxigraphMigrator, OxigraphStore};

// Migrate from Oxigraph store
let oxigraph_store = OxigraphStore::open("/path/to/oxigraph/store")?;
let migrator = OxigraphMigrator::new();

// Analyze Oxigraph data
let analysis = migrator.analyze_oxigraph_store(&oxigraph_store)?;
println!("📊 Oxigraph Store Analysis:");
println!("├─ Total triples: {}", analysis.total_triples);
println!("├─ Named graphs: {}", analysis.named_graphs);
println!("├─ Store size: {} MB", analysis.store_size_mb);
println!("└─ Estimated migration time: {:?}", analysis.estimated_migration_time);

// Direct migration (leverages similar architecture)
let mut target_store = StarStore::new();
let migration_result = migrator.direct_migrate(
    &oxigraph_store, 
    &mut target_store,
    |progress| {
        println!("Progress: {:.1}% - {} triples migrated", 
            progress.percentage, progress.triples_migrated);
    }
)?;

println!("✅ Oxigraph migration completed!");
println!("  Speed: {:.0} triples/sec", migration_result.triples_per_second);
println!("  Data integrity: {}% verified", migration_result.integrity_check_percent);

// Validate migrated data
let validation = migrator.validate_migration(&oxigraph_store, &target_store)?;
if validation.is_complete {
    println!("✅ Migration validation passed");
} else {
    println!("⚠️  Validation issues found:");
    for issue in validation.issues {
        println!("  - {}", issue.description);
    }
}
```

## From Virtuoso

### Virtuoso Universal Server Migration

```rust
use oxirs_star::migration::virtuoso::{VirtuosoMigrator, VirtuosoConnection};

// Connect to Virtuoso Universal Server
let virtuoso_connection = VirtuosoConnection {
    host: "localhost".to_string(),
    port: 1111,
    username: "dba".to_string(),
    password: "dba".to_string(),
    database: "DB".to_string(),
    ssl: false,
};

let virtuoso_migrator = VirtuosoMigrator::connect(virtuoso_connection).await?;

// Discover Virtuoso graphs
let graphs = virtuoso_migrator.list_graphs().await?;
println!("📋 Virtuoso Graphs:");
for graph in &graphs {
    println!("  - {}: {} triples", graph.name, graph.triple_count);
}

// Export data from Virtuoso
let export_config = VirtuosoExportConfig {
    format: ExportFormat::NQuads,
    include_system_graphs: false,
    batch_size: 50_000,
    compress_output: true,
};

println!("📤 Exporting data from Virtuoso...");
let exported_file = virtuoso_migrator.export_all_graphs(
    "/tmp/virtuoso_export.nq.gz",
    export_config,
).await?;

println!("✅ Export completed: {}", exported_file);

// Import into OxiRS-Star
let mut target_store = StarStore::new();
let import_result = target_store.import_nquads_file(&exported_file)?;

println!("📥 Import to OxiRS-Star completed:");
println!("├─ Triples imported: {}", import_result.triples_imported);
println!("├─ Named graphs: {}", import_result.named_graphs_imported);
println!("└─ Import time: {:?}", import_result.duration);

// Migrate Virtuoso-specific features
let feature_migrator = VirtuosoFeatureMigrator::new();

// Convert Virtuoso text indexing to OxiRS full-text search
if virtuoso_migrator.has_text_indexing().await? {
    println!("🔍 Migrating text indexing...");
    feature_migrator.migrate_text_indexing(&mut target_store).await?;
}

// Convert Virtuoso inference rules
if virtuoso_migrator.has_inference_rules().await? {
    println!("🧠 Migrating inference rules...");
    let rules = virtuoso_migrator.export_inference_rules().await?;
    feature_migrator.migrate_inference_rules(&rules, &mut target_store).await?;
}
```

## From AllegroGraph

### AllegroGraph Migration

```rust
use oxirs_star::migration::allegrograph::{AllegroGraphMigrator, AGConnection};

// Connect to AllegroGraph
let ag_connection = AGConnection {
    server_url: "http://localhost:10035".to_string(),
    repository: "my_repository".to_string(),
    username: Some("user".to_string()),
    password: Some("password".to_string()),
};

let ag_migrator = AllegroGraphMigrator::connect(ag_connection).await?;

// Analyze AllegroGraph repository
let repo_info = ag_migrator.get_repository_info().await?;
println!("📊 AllegroGraph Repository Info:");
println!("├─ Triple count: {}", repo_info.triple_count);
println!("├─ Repository size: {} MB", repo_info.size_mb);
println!("├─ Contexts: {}", repo_info.contexts.len());
println!("├─ Free text indices: {}", repo_info.freetext_indices.len());
println!("└─ Geospatial indices: {}", repo_info.geospatial_indices.len());

// Export data in chunks
let export_config = AllegroGraphExportConfig {
    format: ExportFormat::NTriples,
    chunk_size: 100_000,
    include_contexts: true,
    export_directory: "/tmp/ag_export".to_string(),
};

println!("📤 Exporting AllegroGraph data...");
let export_result = ag_migrator.export_repository(export_config).await?;

println!("✅ Export completed:");
println!("├─ Files created: {}", export_result.files_created);
println!("├─ Total size: {} GB", export_result.total_size_gb);
println!("└─ Export time: {:?}", export_result.duration);

// Import to OxiRS-Star with AllegroGraph feature mapping
let ag_feature_mapper = AllegroGraphFeatureMapper::new();
let mut target_store = StarStore::new();

for export_file in export_result.files {
    println!("📥 Importing file: {}", export_file.path);
    
    // Map AllegroGraph contexts to named graphs
    let context_mapping = ag_feature_mapper.map_contexts(&export_file.contexts)?;
    
    target_store.import_with_context_mapping(&export_file.path, context_mapping)?;
}

// Migrate AllegroGraph-specific features
println!("🔧 Migrating AllegroGraph features...");

// Free text indexing
if !repo_info.freetext_indices.is_empty() {
    ag_feature_mapper.migrate_freetext_indices(&repo_info.freetext_indices, &mut target_store)?;
}

// Geospatial data
if !repo_info.geospatial_indices.is_empty() {
    ag_feature_mapper.migrate_geospatial_indices(&repo_info.geospatial_indices, &mut target_store)?;
}

// Social network analysis features
if repo_info.has_sna_features {
    ag_feature_mapper.migrate_sna_features(&mut target_store)?;
}
```

## From Stardog

### Stardog Migration

```rust
use oxirs_star::migration::stardog::{StardogMigrator, StardogConnection};

// Connect to Stardog
let stardog_connection = StardogConnection {
    server_url: "http://localhost:5820".to_string(),
    database: "my_database".to_string(),
    username: "admin".to_string(),
    password: "admin".to_string(),
};

let stardog_migrator = StardogMigrator::connect(stardog_connection).await?;

// Analyze Stardog database
let db_info = stardog_migrator.get_database_info().await?;
println!("📊 Stardog Database Info:");
println!("├─ Size: {} triples", db_info.size);
println!("├─ Named graphs: {}", db_info.named_graphs.len());
println!("├─ Reasoning enabled: {}", db_info.reasoning_enabled);
println!("├─ Full-text search: {}", db_info.fulltext_enabled);
println!("└─ Stored queries: {}", db_info.stored_queries.len());

// Export Stardog data
println!("📤 Exporting Stardog database...");
let export_result = stardog_migrator.export_database(
    ExportFormat::Turtle,
    "/tmp/stardog_export.ttl"
).await?;

println!("✅ Export completed: {} MB", export_result.file_size_mb);

// Import to OxiRS-Star
let mut target_store = StarStore::new();
let import_result = target_store.import_turtle_file("/tmp/stardog_export.ttl")?;

println!("📥 Import completed: {} triples", import_result.triples_imported);

// Migrate Stardog-specific features
let feature_migrator = StardogFeatureMigrator::new();

// Migrate reasoning rules
if db_info.reasoning_enabled {
    println!("🧠 Migrating reasoning rules...");
    let rules = stardog_migrator.export_reasoning_rules().await?;
    let oxirs_rules = feature_migrator.convert_stardog_rules_to_oxirs(&rules)?;
    
    // Apply rules to OxiRS rule engine
    let mut rule_engine = oxirs_rule::RuleEngine::new();
    for rule in oxirs_rules {
        rule_engine.add_rule(rule)?;
    }
    
    // Integrate with store
    target_store.attach_rule_engine(rule_engine)?;
}

// Migrate stored queries
if !db_info.stored_queries.is_empty() {
    println!("💾 Migrating stored queries...");
    let query_migrator = StardogQueryMigrator::new();
    
    for stored_query in db_info.stored_queries {
        let oxirs_query = query_migrator.convert_query(&stored_query)?;
        target_store.store_query(&stored_query.name, &oxirs_query)?;
    }
}

// Migrate security model
if db_info.has_security_model {
    println!("🔐 Migrating security model...");
    let security_migrator = StardogSecurityMigrator::new();
    let security_config = security_migrator.migrate_security_model(&db_info).await?;
    target_store.apply_security_config(security_config)?;
}
```

## From RDF4J

### RDF4J Repository Migration

```rust
use oxirs_star::migration::rdf4j::{Rdf4jMigrator, Rdf4jRepository};

// Connect to RDF4J repository
let rdf4j_repo = Rdf4jRepository {
    server_url: "http://localhost:8080/rdf4j-server".to_string(),
    repository_id: "my_repository".to_string(),
    username: Some("admin".to_string()),
    password: Some("password".to_string()),
};

let rdf4j_migrator = Rdf4jMigrator::connect(rdf4j_repo).await?;

// Get repository metadata
let repo_metadata = rdf4j_migrator.get_repository_metadata().await?;
println!("📊 RDF4J Repository Metadata:");
println!("├─ Type: {}", repo_metadata.repository_type);
println!("├─ Size: {} statements", repo_metadata.statement_count);
println!("├─ Contexts: {}", repo_metadata.contexts.len());
println!("├─ Namespaces: {}", repo_metadata.namespaces.len());
println!("└─ Last modified: {}", repo_metadata.last_modified);

// Stream data from RDF4J
println!("🌊 Streaming data from RDF4J...");
let mut target_store = StarStore::new();
let mut statements_migrated = 0;

let statement_stream = rdf4j_migrator.stream_all_statements().await?;
tokio::pin!(statement_stream);

while let Some(statement_batch) = statement_stream.next().await {
    let batch = statement_batch?;
    
    // Convert RDF4J statements to OxiRS-Star triples
    let oxirs_triples: Vec<StarTriple> = batch
        .into_iter()
        .map(|stmt| rdf4j_migrator.convert_statement_to_triple(stmt))
        .collect::<Result<Vec<_>, _>>()?;
    
    target_store.insert_batch(&oxirs_triples)?;
    statements_migrated += oxirs_triples.len();
    
    if statements_migrated % 100_000 == 0 {
        println!("  Migrated {} statements...", statements_migrated);
    }
}

println!("✅ RDF4J migration completed: {} statements", statements_migrated);

// Migrate RDF4J-specific features
let feature_migrator = Rdf4jFeatureMigrator::new();

// Migrate namespace prefixes
let namespaces = rdf4j_migrator.get_namespaces().await?;
for (prefix, namespace) in namespaces {
    target_store.add_namespace_prefix(&prefix, &namespace)?;
}

// Migrate custom functions (if any)
let custom_functions = rdf4j_migrator.discover_custom_functions().await?;
if !custom_functions.is_empty() {
    println!("🔧 Found {} custom functions", custom_functions.len());
    for func in custom_functions {
        let oxirs_function = feature_migrator.convert_custom_function(&func)?;
        target_store.register_custom_function(oxirs_function)?;
    }
}
```

## Migration Tools

### Automated Migration Suite

```rust
use oxirs_star::migration::{MigrationSuite, MigrationOrchestrator};

// Comprehensive migration orchestrator
let orchestrator = MigrationOrchestrator::new()
    .add_source_adapter(SourceAdapter::Jena)
    .add_source_adapter(SourceAdapter::Virtuoso)
    .add_source_adapter(SourceAdapter::Stardog)
    .add_source_adapter(SourceAdapter::RDF4J)
    .add_source_adapter(SourceAdapter::AllegroGraph)
    .add_source_adapter(SourceAdapter::Oxigraph);

// Auto-detect source store type
let source_path = "/path/to/source/store";
let detected_source = orchestrator.detect_source_type(source_path)?;
println!("🔍 Detected source store: {:?}", detected_source.store_type);
println!("  Confidence: {:.1}%", detected_source.confidence * 100.0);

// Create migration plan
let migration_config = MigrationConfig {
    source_path: source_path.to_string(),
    target_store: &mut target_store,
    batch_size: 100_000,
    parallel_workers: num_cpus::get(),
    validate_data: true,
    create_backups: true,
    dry_run: false,
};

let migration_plan = orchestrator.create_migration_plan(
    detected_source.store_type,
    migration_config
)?;

println!("📋 Migration Plan Created:");
println!("├─ Estimated duration: {:?}", migration_plan.estimated_duration);
println!("├─ Data size: {} GB", migration_plan.estimated_data_size_gb);
println!("├─ Memory required: {} GB", migration_plan.memory_required_gb);
println!("├─ Phases: {}", migration_plan.phases.len());
println!("└─ Risk assessment: {:?}", migration_plan.risk_level);

// Execute migration
if migration_plan.risk_level != RiskLevel::High {
    println!("🚀 Starting automated migration...");
    let result = orchestrator.execute_migration(migration_plan).await?;
    
    println!("✅ Migration completed successfully!");
    println!("├─ Duration: {:?}", result.actual_duration);
    println!("├─ Triples migrated: {}", result.triples_migrated);
    println!("├─ Errors: {}", result.errors.len());
    println!("├─ Warnings: {}", result.warnings.len());
    println!("└─ Data integrity: {:.2}%", result.integrity_score * 100.0);
} else {
    println!("⚠️  High-risk migration detected. Manual intervention required.");
    
    // Provide detailed recommendations
    for recommendation in migration_plan.recommendations {
        println!("💡 {}", recommendation.description);
        println!("   Priority: {:?}", recommendation.priority);
    }
}
```

### CLI Migration Tool

```bash
# OxiRS-Star migration CLI tool
oxirs-migrate --help

# Auto-detect and migrate
oxirs-migrate auto \
    --source /path/to/source/store \
    --target /path/to/oxirs/store \
    --batch-size 100000 \
    --parallel 8 \
    --validate

# Source-specific migrations
oxirs-migrate jena \
    --tdb-directory /path/to/jena/tdb \
    --target /path/to/oxirs/store \
    --convert-reification \
    --preserve-blanks

oxirs-migrate virtuoso \
    --host localhost \
    --port 1111 \
    --user dba \
    --password dba \
    --target /path/to/oxirs/store \
    --include-inference

oxirs-migrate stardog \
    --server http://localhost:5820 \
    --database my_db \
    --user admin \
    --password admin \
    --target /path/to/oxirs/store \
    --migrate-reasoning

# Dry run mode
oxirs-migrate auto \
    --source /path/to/source \
    --target /path/to/target \
    --dry-run \
    --report migration_report.json

# Resume interrupted migration
oxirs-migrate resume \
    --checkpoint /tmp/migration_checkpoint.json \
    --target /path/to/oxirs/store
```

## Data Conversion

### RDF-star Conversion

```rust
use oxirs_star::conversion::{RdfStarConverter, ConversionStrategy};

// Convert reified RDF to RDF-star
let converter = RdfStarConverter::new()
    .with_strategy(ConversionStrategy::Aggressive)
    .enable_pattern_detection()
    .enable_validation();

// Detect reification patterns
let reification_patterns = converter.detect_reification_patterns(&source_store)?;
println!("🔍 Detected reification patterns:");
for pattern in &reification_patterns {
    println!("  - {}: {} instances", pattern.description, pattern.instance_count);
}

// Convert reification to RDF-star
let conversion_result = converter.convert_reification_to_star(
    &source_store,
    &reification_patterns,
    |progress| {
        println!("Conversion progress: {:.1}%", progress.percentage);
    }
)?;

println!("✅ RDF-star conversion completed:");
println!("├─ Reified triples converted: {}", conversion_result.reified_triples_converted);
println!("├─ RDF-star triples created: {}", conversion_result.star_triples_created);
println!("├─ Space savings: {:.1}%", conversion_result.space_savings_percent);
println!("└─ Conversion accuracy: {:.2}%", conversion_result.accuracy_percent);

// Validate conversion
let validation = converter.validate_conversion(&source_store, &target_store)?;
if validation.is_semantically_equivalent {
    println!("✅ Conversion preserves semantic meaning");
} else {
    println!("⚠️  Semantic differences detected:");
    for diff in validation.semantic_differences {
        println!("  - {}", diff.description);
    }
}
```

### Format Conversion

```rust
use oxirs_star::conversion::{FormatConverter, ConversionOptions};

// Convert between RDF formats during migration
let format_converter = FormatConverter::new();

// Batch format conversion
let conversion_options = ConversionOptions {
    source_format: RdfFormat::RdfXml,
    target_format: RdfFormat::TurtleStar,
    preserve_formatting: false,
    optimize_output: true,
    validate_syntax: true,
};

let conversion_result = format_converter.convert_directory(
    "/path/to/source/rdf_xml",
    "/path/to/target/turtle_star",
    conversion_options,
)?;

println!("📄 Format conversion completed:");
println!("├─ Files converted: {}", conversion_result.files_converted);
println!("├─ Total size: {} GB", conversion_result.total_size_gb);
println!("├─ Compression ratio: {:.2}:1", conversion_result.compression_ratio);
println!("└─ Conversion time: {:?}", conversion_result.duration);

// Streaming format conversion for large files
let streaming_converter = format_converter.streaming();
let large_file = "/path/to/large_dataset.rdf";
let converted_file = "/path/to/large_dataset.ttls";

streaming_converter.convert_file_streaming(
    large_file,
    converted_file,
    RdfFormat::RdfXml,
    RdfFormat::TurtleStar,
    |progress| {
        println!("Streaming conversion: {:.1}% - {} MB processed", 
            progress.percentage, progress.megabytes_processed);
    }
)?;
```

## Query Migration

### SPARQL Query Conversion

```rust
use oxirs_star::migration::query::{QueryMigrator, QueryAnalyzer};

// Analyze existing queries for migration compatibility
let query_analyzer = QueryAnalyzer::new();
let query_files = glob::glob("/path/to/queries/*.sparql")?;

let mut analysis_results = Vec::new();

for query_file in query_files {
    let query_file = query_file?;
    let query_content = std::fs::read_to_string(&query_file)?;
    
    let analysis = query_analyzer.analyze_query(&query_content)?;
    analysis_results.push((query_file, analysis));
}

// Summarize analysis
let mut total_queries = 0;
let mut compatible_queries = 0;
let mut migration_required = 0;

for (file_path, analysis) in &analysis_results {
    total_queries += 1;
    
    match analysis.compatibility {
        CompatibilityLevel::FullyCompatible => {
            compatible_queries += 1;
            println!("✅ {}: Fully compatible", file_path.display());
        },
        CompatibilityLevel::MinorChanges => {
            println!("🔄 {}: Minor changes needed", file_path.display());
            migration_required += 1;
        },
        CompatibilityLevel::MajorChanges => {
            println!("⚠️  {}: Major changes needed", file_path.display());
            migration_required += 1;
        },
        CompatibilityLevel::Incompatible => {
            println!("❌ {}: Incompatible", file_path.display());
            migration_required += 1;
        },
    }
    
    if !analysis.issues.is_empty() {
        for issue in &analysis.issues {
            println!("    - {}: {}", issue.severity, issue.description);
        }
    }
}

println!("\n📊 Query Analysis Summary:");
println!("├─ Total queries: {}", total_queries);
println!("├─ Fully compatible: {} ({:.1}%)", compatible_queries, 
    compatible_queries as f64 / total_queries as f64 * 100.0);
println!("└─ Requiring migration: {} ({:.1}%)", migration_required,
    migration_required as f64 / total_queries as f64 * 100.0);

// Migrate queries that need changes
let query_migrator = QueryMigrator::new();

for (file_path, analysis) in analysis_results {
    if analysis.compatibility != CompatibilityLevel::FullyCompatible {
        let original_query = std::fs::read_to_string(&file_path)?;
        
        match query_migrator.migrate_query(&original_query, &analysis) {
            Ok(migrated_query) => {
                let output_path = file_path.with_extension("sparql.migrated");
                std::fs::write(&output_path, &migrated_query)?;
                println!("✅ Migrated: {} → {}", file_path.display(), output_path.display());
            },
            Err(e) => {
                println!("❌ Failed to migrate {}: {}", file_path.display(), e);
            }
        }
    }
}
```

### Application Integration

```rust
use oxirs_star::migration::application::{ApplicationMigrator, CodeAnalyzer};

// Analyze application code for required changes
let code_analyzer = CodeAnalyzer::new()
    .add_language(Language::Java)
    .add_language(Language::Python)
    .add_language(Language::JavaScript)
    .add_language(Language::Rust);

let app_directory = "/path/to/application/src";
let analysis = code_analyzer.analyze_directory(app_directory)?;

println!("🔍 Application Code Analysis:");
println!("├─ Source files analyzed: {}", analysis.files_analyzed);
println!("├─ RDF store dependencies: {}", analysis.rdf_dependencies.len());
println!("├─ SPARQL queries found: {}", analysis.embedded_queries.len());
println!("├─ API calls identified: {}", analysis.api_calls.len());
println!("└─ Migration complexity: {:?}", analysis.complexity);

// Generate migration recommendations
let app_migrator = ApplicationMigrator::new();
let recommendations = app_migrator.generate_recommendations(&analysis)?;

println!("\n💡 Migration Recommendations:");
for rec in recommendations {
    println!("  {}: {}", rec.category, rec.description);
    println!("    Effort: {:?}", rec.effort_level);
    if let Some(code_example) = rec.code_example {
        println!("    Example: {}", code_example);
    }
}

// Generate adapter code
let adapter_config = AdapterConfig {
    target_language: Language::Rust,
    preserve_api_compatibility: true,
    generate_documentation: true,
    include_examples: true,
};

let adapters = app_migrator.generate_adapters(&analysis, adapter_config)?;

for adapter in adapters {
    let output_path = format!("/path/to/adapters/{}.rs", adapter.name);
    std::fs::write(&output_path, &adapter.code)?;
    println!("📝 Generated adapter: {}", output_path);
}
```

## Performance Validation

### Migration Performance Testing

```rust
use oxirs_star::migration::validation::{PerformanceValidator, BenchmarkSuite};

// Compare performance before and after migration
let validator = PerformanceValidator::new();

// Benchmark source store
println!("📊 Benchmarking source store...");
let source_benchmark = validator.benchmark_store(&source_store, &benchmark_queries)?;

// Benchmark target store
println!("📊 Benchmarking migrated store...");
let target_benchmark = validator.benchmark_store(&target_store, &benchmark_queries)?;

// Compare results
let comparison = validator.compare_benchmarks(&source_benchmark, &target_benchmark)?;

println!("⚡ Performance Comparison:");
println!("├─ Query throughput: {:.1}x improvement", comparison.throughput_ratio);
println!("├─ Query latency: {:.1}x improvement", comparison.latency_ratio);
println!("├─ Memory usage: {:.1}x improvement", comparison.memory_ratio);
println!("├─ Storage efficiency: {:.1}x improvement", comparison.storage_ratio);
println!("└─ Overall performance: {:.1}x improvement", comparison.overall_ratio);

if comparison.overall_ratio < 1.0 {
    println!("⚠️  Performance regression detected!");
    println!("Recommendations:");
    for rec in comparison.optimization_recommendations {
        println!("  - {}", rec.description);
    }
} else {
    println!("✅ Migration performance validation passed");
}

// Stress testing
println!("🏋️  Running stress tests...");
let stress_test_config = StressTestConfig {
    concurrent_users: 100,
    queries_per_user: 1000,
    data_size_multiplier: 10.0,
    duration: std::time::Duration::from_secs(300), // 5 minutes
};

let stress_results = validator.run_stress_test(&target_store, stress_test_config)?;

println!("🏋️  Stress Test Results:");
println!("├─ Peak throughput: {} queries/sec", stress_results.peak_throughput);
println!("├─ Average latency: {:?}", stress_results.average_latency);
println!("├─ Error rate: {:.2}%", stress_results.error_rate * 100.0);
println!("├─ Memory stability: {}", if stress_results.memory_stable { "✅" } else { "❌" });
println!("└─ Performance degradation: {:.1}%", stress_results.degradation_percent);
```

## Rollback Strategies

### Migration Rollback

```rust
use oxirs_star::migration::rollback::{RollbackManager, CheckpointManager};

// Create rollback manager with checkpoints
let rollback_manager = RollbackManager::new()
    .with_checkpoint_interval(std::time::Duration::from_secs(300)) // 5 minutes
    .enable_incremental_backups()
    .enable_transaction_logging();

// Create pre-migration checkpoint
let pre_migration_checkpoint = rollback_manager.create_checkpoint(
    "pre_migration",
    &source_store,
)?;

println!("📸 Pre-migration checkpoint created: {}", pre_migration_checkpoint.id);

// Execute migration with rollback capability
let migration_result = rollback_manager.execute_with_rollback(
    || {
        // Migration logic here
        migrator.migrate(&source_store, &mut target_store)
    },
    RollbackPolicy::RollbackOnError
)?;

match migration_result {
    MigrationResult::Success(result) => {
        println!("✅ Migration completed successfully");
        
        // Validate migration
        let validation = validator.validate_migration(&source_store, &target_store)?;
        
        if validation.is_valid {
            // Commit migration
            rollback_manager.commit_migration()?;
            println!("✅ Migration committed");
        } else {
            // Rollback due to validation failure
            println!("❌ Validation failed, rolling back...");
            rollback_manager.rollback_to_checkpoint(&pre_migration_checkpoint)?;
            println!("🔄 Rollback completed");
        }
    },
    MigrationResult::Error(e) => {
        println!("❌ Migration failed: {}", e);
        println!("🔄 Rolling back to pre-migration state...");
        
        rollback_manager.rollback_to_checkpoint(&pre_migration_checkpoint)?;
        println!("✅ Rollback completed successfully");
    }
}

// Cleanup old checkpoints
rollback_manager.cleanup_old_checkpoints(
    std::time::Duration::from_days(7) // Keep checkpoints for 7 days
)?;
```

This comprehensive migration guide provides complete coverage for migrating from major RDF stores to OxiRS-Star, with automated tools, validation processes, and safety mechanisms to ensure successful transitions.