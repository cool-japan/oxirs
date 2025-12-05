//! Benchmarks for Dashboard Analytics Performance
//!
//! Tests the performance of dashboard analytics export functionality
//! including CSV and Excel generation.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxirs_chat::dashboard::{
    DashboardAnalytics, DashboardConfig, ExportFormat, QueryRecord, QueryType, TimeRange,
};
use std::time::Duration;

fn setup_dashboard_with_data() -> DashboardAnalytics {
    let config = DashboardConfig::default();
    let dashboard = DashboardAnalytics::new(config);

    // Populate with sample data for realistic benchmarking
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // Add query records
        for i in 0..1000 {
            dashboard
                .record_query(QueryRecord {
                    query_id: format!("query_{}", i),
                    query_type: match i % 4 {
                        0 => QueryType::NaturalLanguage,
                        1 => QueryType::Sparql,
                        2 => QueryType::VectorSearch,
                        _ => QueryType::Hybrid,
                    },
                    execution_time_ms: 50 + (i % 200),
                    result_count: (10 + (i % 100)) as usize,
                    success: i % 10 != 0, // 90% success rate
                    timestamp: chrono::Utc::now(),
                    error: if i % 10 == 0 {
                        Some("Sample error".to_string())
                    } else {
                        None
                    },
                })
                .await;
        }

        // Add user activity
        for i in 0..100 {
            dashboard
                .update_user_activity(format!("user_{}", i), (i % 50) as u64)
                .await;
        }

        // Add health metrics
        for i in 0..100 {
            dashboard
                .update_health(
                    50.0 + (i as f64 % 30.0),
                    512.0 + (i as f64 * 2.0),
                    10 + (i % 20),
                )
                .await;
        }
    });

    dashboard
}

fn bench_csv_export(c: &mut Criterion) {
    let dashboard = setup_dashboard_with_data();
    let time_range = TimeRange::last_days(7);

    c.bench_function("dashboard_csv_export", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    dashboard
                        .export_data(ExportFormat::Csv, time_range)
                        .await
                        .unwrap(),
                )
            })
        });
    });
}

fn bench_excel_export(c: &mut Criterion) {
    let dashboard = setup_dashboard_with_data();
    let time_range = TimeRange::last_days(7);

    c.bench_function("dashboard_excel_export", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    dashboard
                        .export_data(ExportFormat::Excel, time_range)
                        .await
                        .unwrap(),
                )
            })
        });
    });
}

fn bench_json_export(c: &mut Criterion) {
    let dashboard = setup_dashboard_with_data();
    let time_range = TimeRange::last_days(7);

    c.bench_function("dashboard_json_export", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    dashboard
                        .export_data(ExportFormat::Json, time_range)
                        .await
                        .unwrap(),
                )
            })
        });
    });
}

fn bench_query_analytics(c: &mut Criterion) {
    let dashboard = setup_dashboard_with_data();
    let time_range = TimeRange::last_hours(24);

    c.bench_function("dashboard_query_analytics", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async { black_box(dashboard.get_query_analytics(time_range).await) })
        });
    });
}

fn bench_user_analytics(c: &mut Criterion) {
    let dashboard = setup_dashboard_with_data();
    let time_range = TimeRange::last_hours(24);

    c.bench_function("dashboard_user_analytics", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| rt.block_on(async { black_box(dashboard.get_user_analytics(time_range).await) }));
    });
}

fn bench_health_analytics(c: &mut Criterion) {
    let dashboard = setup_dashboard_with_data();
    let time_range = TimeRange::last_hours(24);

    c.bench_function("dashboard_health_analytics", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async { black_box(dashboard.get_health_analytics(time_range).await) })
        });
    });
}

fn bench_overview(c: &mut Criterion) {
    let dashboard = setup_dashboard_with_data();

    c.bench_function("dashboard_overview", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| rt.block_on(async { black_box(dashboard.get_overview().await) }));
    });
}

criterion_group! {
    name = dashboard_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(5));
    targets = bench_csv_export,
        bench_excel_export,
        bench_json_export,
        bench_query_analytics,
        bench_user_analytics,
        bench_health_analytics,
        bench_overview
}

criterion_main!(dashboard_benches);
