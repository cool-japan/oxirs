//! Benchmark suite for new features in oxirs-ttl (Beta.2)
//!
//! This benchmark suite covers:
//! - IRI normalization (RFC 3987 Section 5.3)
//! - N3 serialization (variables, formulas, implications)
//!
//! Run with: cargo bench -p oxirs-ttl --bench new_features_benchmarks

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_core::model::NamedNode;
use oxirs_ttl::formats::n3_parser::N3Document;
use oxirs_ttl::formats::n3_serializer::N3Serializer;
use oxirs_ttl::formats::n3_types::{N3Formula, N3Implication, N3Statement, N3Term, N3Variable};
use oxirs_ttl::toolkit::iri_normalizer::{iris_equivalent, normalize_iri, normalize_iri_cow};

// ============================================================================
// IRI Normalization Benchmarks
// ============================================================================

/// Benchmark IRI normalization with various IRI types
fn bench_iri_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("iri_normalization");

    // Simple IRI (already normalized)
    group.bench_function("simple_already_normalized", |b| {
        let iri = "http://example.org/path";
        b.iter(|| normalize_iri(iri).unwrap());
    });

    // IRI with uppercase scheme and host
    group.bench_function("case_normalization", |b| {
        let iri = "HTTP://EXAMPLE.ORG/Path";
        b.iter(|| normalize_iri(iri).unwrap());
    });

    // IRI with percent-encoding
    group.bench_function("percent_encoding", |b| {
        let iri = "http://example.org/%7Euser/path%2Fto%2Fresource";
        b.iter(|| normalize_iri(iri).unwrap());
    });

    // IRI with dot segments
    group.bench_function("path_normalization", |b| {
        let iri = "http://example.org/a/./b/../c/./d";
        b.iter(|| normalize_iri(iri).unwrap());
    });

    // IRI with default port
    group.bench_function("default_port_removal", |b| {
        let iri = "http://example.org:80/path";
        b.iter(|| normalize_iri(iri).unwrap());
    });

    // Complex IRI (all normalizations)
    group.bench_function("complex_normalization", |b| {
        let iri = "HTTP://EXAMPLE.ORG:80/%7Euser/a/./b/../c?query=%41%42%43#fragment";
        b.iter(|| normalize_iri(iri).unwrap());
    });

    // IPv6 address
    group.bench_function("ipv6_normalization", |b| {
        let iri = "http://[2001:DB8::1]/path";
        b.iter(|| normalize_iri(iri).unwrap());
    });

    // Userinfo component
    group.bench_function("userinfo_normalization", |b| {
        let iri = "http://user:pass@example.org/path";
        b.iter(|| normalize_iri(iri).unwrap());
    });

    group.finish();
}

/// Benchmark IRI normalization with Cow optimization
fn bench_iri_normalization_cow(c: &mut Criterion) {
    let mut group = c.benchmark_group("iri_normalization_cow");

    // Already normalized (should return Borrowed)
    group.bench_function("cow_already_normalized", |b| {
        let iri = "http://example.org/path";
        b.iter(|| normalize_iri_cow(iri).unwrap());
    });

    // Needs normalization (should return Owned)
    group.bench_function("cow_needs_normalization", |b| {
        let iri = "HTTP://EXAMPLE.ORG/path";
        b.iter(|| normalize_iri_cow(iri).unwrap());
    });

    group.finish();
}

/// Benchmark IRI equivalence checking
fn bench_iri_equivalence(c: &mut Criterion) {
    let mut group = c.benchmark_group("iri_equivalence");

    group.bench_function("equivalent_simple", |b| {
        let iri1 = "http://example.org/path";
        let iri2 = "HTTP://EXAMPLE.ORG/path";
        b.iter(|| iris_equivalent(iri1, iri2).unwrap());
    });

    group.bench_function("equivalent_complex", |b| {
        let iri1 = "http://example.org:80/a/./b/../c";
        let iri2 = "HTTP://EXAMPLE.ORG/a/c";
        b.iter(|| iris_equivalent(iri1, iri2).unwrap());
    });

    group.finish();
}

/// Benchmark batch IRI normalization
fn bench_batch_iri_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_iri_normalization");

    // Generate various IRI types
    let iris = [
        "http://example.org/path1",
        "HTTP://EXAMPLE.ORG/path2",
        "http://example.org:80/path3",
        "http://example.org/%7Euser/path4",
        "http://example.org/a/./b/../c/path5",
        "HTTP://EXAMPLE.ORG:80/%7Euser/a/./b/../c/path6",
        "http://[2001:db8::1]/path7",
        "http://user:pass@example.org/path8",
        "https://example.org:443/path9",
        "ftp://example.org:21/path10",
    ];

    for size in [10, 100, 1_000].iter() {
        let batch: Vec<_> = (0..*size).map(|i| iris[i % iris.len()]).collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                batch
                    .iter()
                    .map(|iri| normalize_iri(iri).unwrap())
                    .collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

// ============================================================================
// N3 Serialization Benchmarks
// ============================================================================

/// Generate test N3 statements with variables
fn generate_n3_statements(count: usize) -> Vec<N3Statement> {
    (0..count)
        .map(|i| {
            N3Statement::new(
                N3Term::Variable(N3Variable::universal(&format!("x{}", i))),
                N3Term::NamedNode(NamedNode::new_unchecked(format!(
                    "http://example.org/predicate{}",
                    i
                ))),
                N3Term::Variable(N3Variable::universal(&format!("y{}", i))),
            )
        })
        .collect()
}

/// Generate test N3 implications
fn generate_n3_implications(count: usize) -> Vec<N3Implication> {
    (0..count)
        .map(|i| {
            let mut antecedent = N3Formula::new();
            antecedent.add_statement(N3Statement::new(
                N3Term::Variable(N3Variable::universal(&format!("x{}", i))),
                N3Term::NamedNode(NamedNode::new_unchecked(format!(
                    "http://example.org/parent{}",
                    i
                ))),
                N3Term::Variable(N3Variable::universal(&format!("y{}", i))),
            ));

            let mut consequent = N3Formula::new();
            consequent.add_statement(N3Statement::new(
                N3Term::Variable(N3Variable::universal(&format!("y{}", i))),
                N3Term::NamedNode(NamedNode::new_unchecked(format!(
                    "http://example.org/child{}",
                    i
                ))),
                N3Term::Variable(N3Variable::universal(&format!("x{}", i))),
            ));

            N3Implication::new(antecedent, consequent)
        })
        .collect()
}

/// Benchmark N3 statement serialization
fn bench_n3_statement_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("n3_statement_serialization");

    for size in [10, 100, 1_000].iter() {
        let statements = generate_n3_statements(*size);
        let serializer = N3Serializer::new();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut buffer = Vec::new();
                for stmt in &statements {
                    serializer.serialize_statement(stmt, &mut buffer).unwrap();
                }
                buffer
            });
        });
    }

    group.finish();
}

/// Benchmark N3 implication serialization
fn bench_n3_implication_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("n3_implication_serialization");

    for size in [10, 100, 1_000].iter() {
        let implications = generate_n3_implications(*size);
        let serializer = N3Serializer::new();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut buffer = Vec::new();
                for impl_ in &implications {
                    serializer
                        .serialize_implication(impl_, &mut buffer)
                        .unwrap();
                }
                buffer
            });
        });
    }

    group.finish();
}

/// Benchmark full N3 document serialization
fn bench_n3_document_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("n3_document_serialization");

    for size in [10, 100, 1_000].iter() {
        let mut doc = N3Document::new();
        doc.add_prefix("ex".to_string(), "http://example.org/".to_string());
        doc.set_base_iri("http://example.org/base".to_string());

        // Add statements
        let statements = generate_n3_statements(*size / 2);
        for stmt in statements {
            doc.add_statement(stmt);
        }

        // Add implications
        let implications = generate_n3_implications(*size / 2);
        for impl_ in implications {
            doc.add_implication(impl_);
        }

        let serializer = N3Serializer::new();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut buffer = Vec::new();
                serializer.serialize_document(&doc, &mut buffer).unwrap();
                buffer
            });
        });
    }

    group.finish();
}

/// Benchmark N3 formula serialization
fn bench_n3_formula_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("n3_formula_serialization");

    // Empty formula
    group.bench_function("empty_formula", |b| {
        let formula = N3Formula::new();
        let serializer = N3Serializer::new();
        b.iter(|| {
            let mut buffer = Vec::new();
            serializer.serialize_formula(&formula, &mut buffer).unwrap();
            buffer
        });
    });

    // Formula with statements
    for size in [1, 10, 100].iter() {
        let mut formula = N3Formula::new();
        let statements = generate_n3_statements(*size);
        for stmt in statements {
            formula.add_statement(stmt);
        }

        let serializer = N3Serializer::new();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut buffer = Vec::new();
                serializer.serialize_formula(&formula, &mut buffer).unwrap();
                buffer
            });
        });
    }

    group.finish();
}

/// Benchmark N3 serialization with complex nested structures
fn bench_n3_nested_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("n3_nested_serialization");

    // Nested formulas
    group.bench_function("nested_formulas", |b| {
        let mut outer_formula = N3Formula::new();
        let mut inner_formula = N3Formula::new();
        inner_formula.add_statement(N3Statement::new(
            N3Term::Variable(N3Variable::universal("x")),
            N3Term::NamedNode(NamedNode::new_unchecked("http://example.org/predicate")),
            N3Term::Variable(N3Variable::universal("y")),
        ));
        outer_formula.add_statement(N3Statement::new(
            N3Term::Formula(Box::new(inner_formula)),
            N3Term::NamedNode(NamedNode::new_unchecked("http://example.org/source")),
            N3Term::NamedNode(NamedNode::new_unchecked("http://example.org/document")),
        ));

        let serializer = N3Serializer::new();
        b.iter(|| {
            let mut buffer = Vec::new();
            serializer
                .serialize_formula(&outer_formula, &mut buffer)
                .unwrap();
            buffer
        });
    });

    group.finish();
}

criterion_group!(
    iri_benches,
    bench_iri_normalization,
    bench_iri_normalization_cow,
    bench_iri_equivalence,
    bench_batch_iri_normalization
);

criterion_group!(
    n3_benches,
    bench_n3_statement_serialization,
    bench_n3_implication_serialization,
    bench_n3_document_serialization,
    bench_n3_formula_serialization,
    bench_n3_nested_serialization
);

criterion_main!(iri_benches, n3_benches);
