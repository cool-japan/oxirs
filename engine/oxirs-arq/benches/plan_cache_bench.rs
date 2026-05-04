//! Benchmark for the algebra-level JIT plan cache (phase a).

use criterion::{criterion_group, criterion_main, Criterion};
use oxirs_arq::algebra::{Algebra, Term, TriplePattern, Variable};
use oxirs_arq::plan_cache::{compute_fingerprint, PlanCache};
use oxirs_core::model::NamedNode;

fn make_bgp(s_name: &str, o_name: &str) -> Algebra {
    let pred = Term::Iri(NamedNode::new_unchecked("http://example.org/p"));
    let s = Term::Variable(Variable::new(s_name).expect("valid var"));
    let o = Term::Variable(Variable::new(o_name).expect("valid var"));
    Algebra::Bgp(vec![TriplePattern {
        subject: s,
        predicate: pred,
        object: o,
    }])
}

fn bench_cache_hit(c: &mut Criterion) {
    let cache: PlanCache<String> = PlanCache::new(1024);
    cache.insert(42, "cached_plan".to_string());
    c.bench_function("plan_cache_hit", |b| {
        b.iter(|| {
            let _ = cache.get(42);
        });
    });
}

fn bench_cache_miss(c: &mut Criterion) {
    let cache: PlanCache<String> = PlanCache::new(1024);
    c.bench_function("plan_cache_miss", |b| {
        b.iter(|| {
            let _ = cache.get(99999);
        });
    });
}

fn bench_fingerprint(c: &mut Criterion) {
    let plan = make_bgp("subject", "object");
    c.bench_function("compute_fingerprint_simple_bgp", |b| {
        b.iter(|| compute_fingerprint(&plan));
    });
}

fn bench_cache_insert(c: &mut Criterion) {
    c.bench_function("plan_cache_insert", |b| {
        let cache: PlanCache<String> = PlanCache::new(1024);
        let mut key = 0u64;
        b.iter(|| {
            cache.insert(key, "plan".to_string());
            key = key.wrapping_add(1);
        });
    });
}

criterion_group!(
    benches,
    bench_cache_hit,
    bench_cache_miss,
    bench_fingerprint,
    bench_cache_insert
);
criterion_main!(benches);
