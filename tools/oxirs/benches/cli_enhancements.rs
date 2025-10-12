//! Performance benchmarks for CLI enhancements
//!
//! Benchmarks for query explain, templates, and history features

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs::commands::{cache, history, templates};
use std::collections::HashMap;

fn bench_template_rendering(c: &mut Criterion) {
    let mut group = c.benchmark_group("template_rendering");

    // Get a simple template
    let template = templates::get_template("select-all").unwrap();

    group.bench_function("render_select_all", |b| {
        b.iter(|| {
            let params = HashMap::new();
            black_box(templates::render_template(&template, &params).unwrap())
        });
    });

    // Get a template with parameters
    let template_with_params = templates::get_template("select-by-type").unwrap();

    group.bench_function("render_with_params", |b| {
        let mut params = HashMap::new();
        params.insert(
            "type_iri".to_string(),
            "http://xmlns.com/foaf/0.1/Person".to_string(),
        );
        params.insert("limit".to_string(), "100".to_string());

        b.iter(|| black_box(templates::render_template(&template_with_params, &params).unwrap()));
    });

    group.finish();
}

fn bench_template_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("template_lookup");

    group.bench_function("get_template_by_name", |b| {
        b.iter(|| black_box(templates::get_template("select-by-type")));
    });

    group.bench_function("list_all_templates", |b| {
        b.iter(|| black_box(templates::get_all_templates()));
    });

    group.bench_function("list_by_category", |b| {
        b.iter(|| {
            black_box(templates::list_templates(Some(
                templates::TemplateCategory::Basic,
            )))
        });
    });

    group.finish();
}

fn bench_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_operations");

    let cache = cache::QueryCache::new(300, 1000);
    let dataset = "test_dataset";
    let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10";
    let result = r#"{"results": []}"#.to_string();

    // Pre-populate cache for hit testing
    cache.set(dataset, query, result.clone());

    group.bench_function("cache_hit", |b| {
        b.iter(|| black_box(cache.get(dataset, query)));
    });

    group.bench_function("cache_miss", |b| {
        b.iter(|| black_box(cache.get(dataset, "non_existent_query")));
    });

    group.bench_function("cache_set", |b| {
        let query_new = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 50";
        b.iter(|| {
            cache.set(dataset, query_new, result.clone());
        });
    });

    group.finish();
}

fn bench_history_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("history_operations");

    use std::env::temp_dir;
    let temp_file = temp_dir().join("bench_history.json");
    let mut history = history::QueryHistory::new(temp_file.clone(), 1000);

    // Pre-populate with some entries
    for i in 0..100 {
        let entry = history::HistoryEntry {
            id: i,
            timestamp: chrono::Utc::now(),
            dataset: "test_dataset".to_string(),
            query: format!("SELECT ?s WHERE {{ ?s ?p ?o }} LIMIT {}", i),
            execution_time_ms: Some(10.5),
            result_count: Some(42),
            success: true,
            error: None,
        };
        history.add_entry(entry).ok();
    }

    group.bench_function("add_entry", |b| {
        b.iter(|| {
            let entry = history::HistoryEntry {
                id: 0,
                timestamp: chrono::Utc::now(),
                dataset: "test_dataset".to_string(),
                query: "SELECT ?s ?p ?o WHERE { ?s ?p ?o }".to_string(),
                execution_time_ms: Some(5.2),
                result_count: Some(10),
                success: true,
                error: None,
            };
            black_box(history.add_entry(entry))
        });
    });

    group.bench_function("get_entry", |b| {
        b.iter(|| black_box(history.get_entry(50)));
    });

    group.bench_function("search", |b| {
        b.iter(|| black_box(history.search("SELECT")));
    });

    group.bench_function("recent_n", |b| {
        b.iter(|| black_box(history.recent(20)));
    });

    // Cleanup
    std::fs::remove_file(temp_file).ok();

    group.finish();
}

fn bench_cache_key_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_key_generation");

    group.bench_function("generate_cache_key", |b| {
        let dataset = "test_dataset";
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10";

        b.iter(|| {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            dataset.hash(&mut hasher);
            query.hash(&mut hasher);
            black_box(format!("{:x}", hasher.finish()))
        });
    });

    group.finish();
}

fn bench_template_categories(c: &mut Criterion) {
    let mut group = c.benchmark_group("template_categories");

    let categories = vec![
        templates::TemplateCategory::Basic,
        templates::TemplateCategory::Advanced,
        templates::TemplateCategory::Analytics,
        templates::TemplateCategory::Aggregation,
        templates::TemplateCategory::PropertyPaths,
        templates::TemplateCategory::Federation,
    ];

    for category in categories {
        group.bench_with_input(
            BenchmarkId::new("filter_by_category", format!("{:?}", category)),
            &category,
            |b, cat| {
                b.iter(|| black_box(templates::list_templates(Some(cat.clone()))));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_template_rendering,
    bench_template_lookup,
    bench_cache_operations,
    bench_history_operations,
    bench_cache_key_generation,
    bench_template_categories
);

criterion_main!(benches);
