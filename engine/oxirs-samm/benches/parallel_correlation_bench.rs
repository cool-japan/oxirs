//! Benchmarks for parallel code generation and batch correlation matrix.
//!
//! # Design notes on parameter interpretation
//!
//! **"N generators"** — `ParallelGenerator::generate_all` runs a fixed set of
//! 10 built-in generators; there is no API to subset them.  The varying
//! parameter here is instead the *aspect complexity* (number of properties:
//! 4, 8, 16) so the parallel vs. sequential comparison reflects real work
//! variation, not an artificial generator count.
//!
//! **"control variables 1, 3, 5"** — `partial_correlation_matrix` accepts a
//! single `control_idx: usize`.  The bench calls the function with three
//! different control indices (0, 2, 4) to measure throughput across different
//! column positions in the matrix, which is the closest meaningful mapping of
//! the parameterisation requested.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_samm::analytics::batch_matrix::BatchCorrelationMatrix;
use oxirs_samm::generators::parallel::ParallelGenerator;
use oxirs_samm::generators::{
    generate_aas, generate_dtdl, generate_graphql, generate_java, generate_jsonld,
    generate_payload, generate_python, generate_scala, generate_sql, generate_typescript,
    AasFormat, JavaOptions, PythonOptions, ScalaOptions, SqlDialect, TsOptions,
};
use oxirs_samm::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};
use oxirs_samm::validation::batch::BatchValidator;
use std::hint::black_box;
use std::time::Duration;

// ============================================================================
//  Shared helpers
// ============================================================================

/// Build a test aspect with `n` properties.
fn make_aspect(n: usize) -> Aspect {
    let mut aspect = Aspect::new(format!("urn:samm:org.example:1.0.0#BenchAspect{}", n));
    for i in 0..n {
        let char = Characteristic::new(
            format!("urn:samm:org.example:1.0.0#Char{}", i),
            CharacteristicKind::Trait,
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
        let prop = Property::new(format!("urn:samm:org.example:1.0.0#prop{}", i))
            .with_characteristic(char);
        aspect.add_property(prop);
    }
    aspect
}

/// Build a minimal valid aspect (one property, camelCase fragment).
fn make_valid_aspect(urn_fragment: &str) -> Aspect {
    let ns = "urn:samm:org.example:1.0.0";
    let mut aspect = Aspect::new(format!("{}#{}", ns, urn_fragment));
    let char = Characteristic::new(
        format!("{}#speedChar{}", ns, urn_fragment),
        CharacteristicKind::Measurement {
            unit: "unit:kilometre".to_string(),
        },
    )
    .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string());
    let prop = Property::new(format!("{}#speed{}", ns, urn_fragment)).with_characteristic(char);
    aspect.add_property(prop);
    aspect
}

/// Synthesize `n` feature columns each with `obs` observations.
/// Column `i` is `sin(i * j)` so no two are identical.
fn make_columns(n: usize, obs: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..obs)
                .map(|j| ((i + 1) as f64 * j as f64).sin())
                .collect()
        })
        .collect()
}

// ============================================================================
//  ParallelGenerator benchmarks
// ============================================================================

/// Compare rayon parallel (generate_all) vs sequential invocation of every
/// generator for aspects with 4, 8, and 16 properties.
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");
    group.measurement_time(Duration::from_secs(15));

    for &n_props in &[4_usize, 8, 16] {
        let aspect = make_aspect(n_props);
        let gen = ParallelGenerator::new();

        // Parallel: all generators via rayon
        group.bench_with_input(BenchmarkId::new("parallel", n_props), &aspect, |b, asp| {
            b.iter(|| {
                let result = gen.generate_all(black_box(asp));
                black_box(result.outputs.len())
            });
        });

        // Sequential: same generators called one-by-one
        group.bench_with_input(
            BenchmarkId::new("sequential", n_props),
            &aspect,
            |b, asp| {
                b.iter(|| {
                    let mut count = 0usize;
                    let _ = generate_graphql(black_box(asp)).ok();
                    count += 1;
                    let _ = generate_typescript(black_box(asp), TsOptions::default()).ok();
                    count += 1;
                    let _ = generate_python(black_box(asp), PythonOptions::default()).ok();
                    count += 1;
                    let _ = generate_java(black_box(asp), JavaOptions::default()).ok();
                    count += 1;
                    let _ = generate_scala(black_box(asp), ScalaOptions::default()).ok();
                    count += 1;
                    let _ = generate_sql(black_box(asp), SqlDialect::PostgreSql).ok();
                    count += 1;
                    let _ = generate_jsonld(black_box(asp)).ok();
                    count += 1;
                    let _ = generate_payload(black_box(asp), false).ok();
                    count += 1;
                    let _ = generate_dtdl(black_box(asp)).ok();
                    count += 1;
                    let _ = generate_aas(black_box(asp), AasFormat::Json).ok();
                    count += 1;
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

/// Measure throughput (aspects per second) for concurrent generation at
/// varying property counts: 1, 2, 4, 8, 16 properties.
fn bench_parallel_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_throughput");
    group.measurement_time(Duration::from_secs(12));

    for &n_props in &[1_usize, 2, 4, 8, 16] {
        let aspect = make_aspect(n_props);
        let gen = ParallelGenerator::new();

        // Throughput = one aspect processed through all generators per iteration.
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("aspects_per_sec", n_props),
            &aspect,
            |b, asp| {
                b.iter(|| {
                    let result = gen.generate_all(black_box(asp));
                    black_box(result.outputs.len())
                });
            },
        );
    }

    group.finish();
}

/// Single large complex aspect (50 properties) through `generate_all`.
fn bench_parallel_large_aspect(c: &mut Criterion) {
    let aspect = make_aspect(50);
    let gen = ParallelGenerator::new();

    c.bench_function("parallel_large_aspect_50props", |b| {
        b.iter(|| {
            let result = gen.generate_all(black_box(&aspect));
            black_box(result.outputs.len())
        });
    });
}

// ============================================================================
//  BatchCorrelationMatrix benchmarks
// ============================================================================

/// Correlation matrix for N variables with M=1000 samples: N = 10, 50, 100, 500.
fn bench_correlation_matrix_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation_matrix_size");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(20);

    for &n_vars in &[10_usize, 50, 100, 500] {
        let cols = make_columns(n_vars, 1000);
        let refs: Vec<&[f64]> = cols.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements(n_vars as u64));
        group.bench_with_input(BenchmarkId::new("n_vars", n_vars), &refs, |b, samples| {
            b.iter(|| {
                let mat = BatchCorrelationMatrix::compute(black_box(samples), None)
                    .expect("valid inputs must not fail");
                black_box(mat.significant_pairs.len())
            });
        });
    }

    group.finish();
}

/// Partial correlation with different control indices (0, 2, 4) on a 20-variable
/// dataset with 500 observations.  Models the cost of varying the control
/// column position across three distinct parameterisations.
fn bench_partial_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("partial_correlation_control_idx");
    group.measurement_time(Duration::from_secs(12));

    let n_vars = 20_usize;
    let cols = make_columns(n_vars, 500);
    let refs: Vec<&[f64]> = cols.iter().map(|v| v.as_slice()).collect();

    // Three control positions: 0 (first), 2 (mid-low), 4 (mid)
    for &ctrl_idx in &[0_usize, 2, 4] {
        group.bench_with_input(
            BenchmarkId::new("control_idx", ctrl_idx),
            &refs,
            |b, samples| {
                b.iter(|| {
                    let mat = BatchCorrelationMatrix::partial_correlation_matrix(
                        black_box(samples),
                        black_box(ctrl_idx),
                    )
                    .expect("valid inputs must not fail");
                    black_box(mat.significant_pairs.len())
                });
            },
        );
    }

    group.finish();
}

/// Throughput benchmark: matrices per second for typical N=20, M=500.
fn bench_correlation_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation_throughput");
    group.measurement_time(Duration::from_secs(12));

    let cols = make_columns(20, 500);
    let refs: Vec<&[f64]> = cols.iter().map(|v| v.as_slice()).collect();

    group.throughput(Throughput::Elements(1));
    group.bench_function("matrices_per_sec_n20_m500", |b| {
        b.iter(|| {
            let mat = BatchCorrelationMatrix::compute(black_box(&refs[..]), None)
                .expect("valid inputs must not fail");
            black_box(mat.significant_pairs.len())
        });
    });

    group.finish();
}

// ============================================================================
//  BatchValidator benchmarks
// ============================================================================

/// validate_batch for batch sizes 10, 100, 1000, 10000.
fn bench_batch_validate_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_validate_sizes");
    // Large batches are slow — give them more time and reduce samples.
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    let validator = BatchValidator::new();

    for &batch_size in &[10_usize, 100, 1000, 10000] {
        // Pre-build all aspects outside the bench loop.
        let aspects: Vec<Aspect> = (0..batch_size)
            .map(|i| make_valid_aspect(&format!("Aspect{}", i)))
            .collect();
        let aspect_refs: Vec<&Aspect> = aspects.iter().collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &aspect_refs,
            |b, refs| {
                b.iter(|| {
                    let reports = validator.validate_batch(black_box(refs));
                    black_box(reports.len())
                });
            },
        );
    }

    group.finish();
}

/// GPU (dispatch overhead + CPU fallback) vs CPU-only on the same batch.
///
/// In a standard test environment the GPU backend is unavailable, so the
/// `with_gpu(true)` path measures the dispatch attempt overhead before
/// the transparent CPU fallback.
fn bench_batch_gpu_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_gpu_vs_cpu");
    group.measurement_time(Duration::from_secs(15));

    let batch_size = 100_usize;
    let aspects: Vec<Aspect> = (0..batch_size)
        .map(|i| make_valid_aspect(&format!("GpuVsCpu{}", i)))
        .collect();
    let aspect_refs: Vec<&Aspect> = aspects.iter().collect();

    let cpu_validator = BatchValidator::new().with_gpu(false);
    let gpu_validator = BatchValidator::new().with_gpu(true);

    group.throughput(Throughput::Elements(batch_size as u64));

    group.bench_function("cpu_only", |b| {
        b.iter(|| {
            let reports = cpu_validator.validate_batch(black_box(&aspect_refs));
            black_box(reports.len())
        });
    });

    group.bench_function("gpu_dispatch_with_cpu_fallback", |b| {
        b.iter(|| {
            let reports = gpu_validator.validate_batch(black_box(&aspect_refs));
            black_box(reports.len())
        });
    });

    group.finish();
}

// ============================================================================
//  Criterion registration
// ============================================================================

criterion_group! {
    name = parallel_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(50);
    targets =
        bench_parallel_vs_sequential,
        bench_parallel_throughput,
        bench_parallel_large_aspect
}

criterion_group! {
    name = correlation_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(50);
    targets =
        bench_correlation_matrix_size,
        bench_partial_correlation,
        bench_correlation_throughput
}

criterion_group! {
    name = batch_validator_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(50);
    targets =
        bench_batch_validate_sizes,
        bench_batch_gpu_vs_cpu
}

criterion_main!(
    parallel_benches,
    correlation_benches,
    batch_validator_benches
);
