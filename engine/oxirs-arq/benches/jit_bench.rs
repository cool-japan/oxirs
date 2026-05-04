//! Benchmarks for JIT phase b — Cranelift filter expression codegen.
//!
//! Run with:
//!   cargo bench -p oxirs-arq --features jit --bench jit_bench

#![cfg(feature = "jit")]

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use oxirs_arq::jit::{BinOp, BuiltinFunc, FilterCompiler, FilterExpr, JitFilterCache, VarIndexMap};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn range_filter_expr() -> (FilterExpr, VarIndexMap) {
    // (?x > 0.0) && (?x < 1000.0) — a common numeric range filter
    let mut vm = VarIndexMap::new();
    vm.insert("x".to_string(), 0);
    let gt_zero = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(0.0)),
    };
    let lt_thousand = FilterExpr::BinOp {
        op: BinOp::Lt,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(1000.0)),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::And,
        left: Box::new(gt_zero),
        right: Box::new(lt_thousand),
    };
    (expr, vm)
}

fn complex_filter_expr() -> (FilterExpr, VarIndexMap) {
    // ABS(?x - ?y) <= 50.0 && (?x * ?y) > 100.0
    let mut vm = VarIndexMap::new();
    vm.insert("x".to_string(), 0);
    vm.insert("y".to_string(), 1);

    let diff = FilterExpr::BinOp {
        op: BinOp::Sub,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Variable("y".to_string())),
    };
    let abs_diff = FilterExpr::Builtin {
        func: BuiltinFunc::Abs,
        arg: Box::new(diff),
    };
    let close = FilterExpr::BinOp {
        op: BinOp::Le,
        left: Box::new(abs_diff),
        right: Box::new(FilterExpr::Literal(50.0)),
    };
    let product = FilterExpr::BinOp {
        op: BinOp::Mul,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Variable("y".to_string())),
    };
    let large_product = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(product),
        right: Box::new(FilterExpr::Literal(100.0)),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::And,
        left: Box::new(close),
        right: Box::new(large_product),
    };
    (expr, vm)
}

// ---------------------------------------------------------------------------
// bench_jit_numeric_filter: JIT vs manual interpreted (1 call per iteration)
// ---------------------------------------------------------------------------

fn bench_jit_numeric_filter(c: &mut Criterion) {
    let (expr, vm) = range_filter_expr();
    let compiled = FilterCompiler::new()
        .compile(&expr, vm)
        .expect("compile ok")
        .expect("in supported subset");

    let mut binding = HashMap::new();
    binding.insert("x".to_string(), 500.0f64);

    let mut group = c.benchmark_group("jit_filter");
    group.bench_function("jit_eval_range_filter", |b| {
        b.iter(|| black_box(compiled.evaluate(black_box(&binding))));
    });

    // Reference: interpreted equivalent (manual Rust, representing interpreter baseline)
    group.bench_function("interpreted_range_filter", |b| {
        b.iter(|| {
            let x: f64 = black_box(500.0f64);
            black_box(x > 0.0 && x < 1000.0)
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// bench_jit_compile_time: time to compile a complex filter expression
// ---------------------------------------------------------------------------

fn bench_jit_compile_time(c: &mut Criterion) {
    let (expr, vm) = complex_filter_expr();
    c.bench_function("jit_compile_complex_filter", |b| {
        b.iter(|| {
            let expr_clone = black_box(expr.clone());
            let vm_clone = black_box(vm.clone());
            FilterCompiler::new()
                .compile(&expr_clone, vm_clone)
                .expect("compile ok")
        });
    });
}

// ---------------------------------------------------------------------------
// bench_jit_cache_lookup: hot-path cache lookup latency
// ---------------------------------------------------------------------------

fn bench_jit_cache_lookup(c: &mut Criterion) {
    let cache = JitFilterCache::new(128).expect("cache init");
    let (expr, vm) = range_filter_expr();
    cache
        .compile_and_insert(1234, &expr, vm)
        .expect("compile ok");

    c.bench_function("jit_cache_lookup_hit", |b| {
        b.iter(|| black_box(cache.get(black_box(1234_u64))));
    });

    c.bench_function("jit_cache_lookup_miss", |b| {
        b.iter(|| black_box(cache.get(black_box(9999_u64))));
    });
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_jit_numeric_filter,
    bench_jit_compile_time,
    bench_jit_cache_lookup,
);
criterion_main!(benches);
