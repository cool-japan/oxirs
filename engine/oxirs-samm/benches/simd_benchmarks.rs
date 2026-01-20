use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_samm::simd_ops::*;

/// Benchmark URN validation for single URN
fn bench_validate_single_urn(c: &mut Criterion) {
    let urn = "urn:samm:org.example.domain:1.0.0#MyAspect";

    c.bench_function("validate_single_urn", |b| {
        b.iter(|| validate_urns_batch(black_box(&[urn])))
    });
}

/// Benchmark URN validation for batch of URNs
fn bench_validate_batch_urns(c: &mut Criterion) {
    let mut group = c.benchmark_group("validate_batch_urns");

    for batch_size in [10, 100, 1000, 10000].iter() {
        let urns: Vec<&str> = (0..*batch_size)
            .map(|i| match i % 4 {
                0 => "urn:samm:org.example:1.0.0#Aspect",
                1 => "urn:samm:com.company.product:2.1.3#Property",
                2 => "urn:samm:io.sample.test:1.5.0#Characteristic",
                _ => "urn:samm:net.test.domain:3.0.0#Entity",
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| b.iter(|| validate_urns_batch(black_box(&urns))),
        );
    }

    group.finish();
}

/// Benchmark character counting with SIMD
fn bench_count_char_simd(c: &mut Criterion) {
    let text = "urn:samm:org.example.domain.subdomain:1.0.0#MyAspect";

    c.bench_function("count_char_simd_colon", |b| {
        b.iter(|| count_char_simd(black_box(text), black_box(':')))
    });

    c.bench_function("count_char_simd_dot", |b| {
        b.iter(|| count_char_simd(black_box(text), black_box('.')))
    });
}

/// Benchmark namespace extraction
fn bench_extract_namespace(c: &mut Criterion) {
    let urn = "urn:samm:org.example.domain.subdomain:1.0.0#MyAspect";

    c.bench_function("extract_namespace_fast", |b| {
        b.iter(|| extract_namespace_fast(black_box(urn)))
    });
}

/// Benchmark version extraction
fn bench_extract_version(c: &mut Criterion) {
    let urn = "urn:samm:org.example.domain.subdomain:1.0.0#MyAspect";

    c.bench_function("extract_version_fast", |b| {
        b.iter(|| extract_version_fast(black_box(urn)))
    });
}

/// Benchmark element extraction
fn bench_extract_element(c: &mut Criterion) {
    let urn = "urn:samm:org.example.domain.subdomain:1.0.0#MyAspect";

    c.bench_function("extract_element_fast", |b| {
        b.iter(|| extract_element_fast(black_box(urn)))
    });
}

/// Benchmark batch URN part extraction
fn bench_extract_urn_parts_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("extract_urn_parts_batch");

    for batch_size in [10, 100, 1000].iter() {
        let urns: Vec<&str> = (0..*batch_size)
            .map(|i| match i % 3 {
                0 => "urn:samm:org.example:1.0.0#Aspect",
                1 => "urn:samm:com.test:2.0.0#Property",
                _ => "urn:samm:io.sample:1.5.0#Characteristic",
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| b.iter(|| extract_urn_parts_batch(black_box(&urns))),
        );
    }

    group.finish();
}

/// Benchmark finding URNs in large text
fn bench_find_urns_in_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_urns_in_text");

    // Small text (1 KB)
    let small_text = r#"
        The aspect urn:samm:org.example:1.0.0#MyAspect has properties.
        Property urn:samm:org.example:1.0.0#temperature is required.
        Another property urn:samm:org.example:1.0.0#pressure is optional.
    "#
    .repeat(20);

    group.bench_function("small_text_1kb", |b| {
        b.iter(|| find_urns_in_text(black_box(&small_text)))
    });

    // Medium text (10 KB)
    let medium_text = small_text.repeat(10);

    group.bench_function("medium_text_10kb", |b| {
        b.iter(|| find_urns_in_text(black_box(&medium_text)))
    });

    // Large text (100 KB)
    let large_text = medium_text.repeat(10);

    group.bench_function("large_text_100kb", |b| {
        b.iter(|| find_urns_in_text(black_box(&large_text)))
    });

    group.finish();
}

/// Benchmark comparison: SIMD char counting vs standard iteration
fn bench_simd_vs_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_standard");

    let text = "urn:samm:org.example.domain.subdomain:1.0.0#MyAspect".repeat(100);

    group.bench_function("simd_count", |b| {
        b.iter(|| count_char_simd(black_box(&text), black_box(':')))
    });

    group.bench_function("standard_count", |b| {
        b.iter(|| black_box(&text).chars().filter(|&c| c == ':').count())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_validate_single_urn,
    bench_validate_batch_urns,
    bench_count_char_simd,
    bench_extract_namespace,
    bench_extract_version,
    bench_extract_element,
    bench_extract_urn_parts_batch,
    bench_find_urns_in_text,
    bench_simd_vs_standard,
);

criterion_main!(benches);
