//! Integration tests for the algebra-level JIT plan cache (phase a).
//!
//! Tests cover:
//! - LRU eviction capacity + access-order refresh
//! - `PlanCache` hit/miss counters
//! - `PlanCache` capacity eviction
//! - `PlanCache` invalidate_all
//! - Concurrent read stress (8 threads)
//! - Fingerprint variable normalisation (renamed-variable plans collide)
//! - Structurally distinct plans do NOT collide
//! - Optimizer integration (plan cache enabled → cache hits observed)

use oxirs_arq::algebra::{Algebra, Expression, Term, TriplePattern, Variable};
use oxirs_arq::optimizer::{Optimizer, OptimizerConfig};
use oxirs_arq::plan_cache::{compute_fingerprint, LruEviction, PlanCache};
use oxirs_core::model::NamedNode;

// ---------------------------------------------------------------------------
// LruEviction tests
// ---------------------------------------------------------------------------

#[test]
fn test_lru_eviction_capacity() {
    let mut lru = LruEviction::new(3);
    assert!(lru.on_insert(1).is_none());
    assert!(lru.on_insert(2).is_none());
    assert!(lru.on_insert(3).is_none());
    let evicted = lru.on_insert(4);
    assert_eq!(evicted, Some(1)); // LRU is key 1
}

#[test]
fn test_lru_eviction_access_refreshes_order() {
    let mut lru = LruEviction::new(3);
    lru.on_insert(1);
    lru.on_insert(2);
    lru.on_insert(3);
    lru.on_access(1); // Now 1 is MRU; LRU is 2
    let evicted = lru.on_insert(4);
    assert_eq!(evicted, Some(2));
}

#[test]
fn test_lru_eviction_no_duplicates() {
    let mut lru = LruEviction::new(3);
    lru.on_insert(1);
    lru.on_insert(1);
    lru.on_insert(1);
    assert_eq!(lru.len(), 1);
}

// ---------------------------------------------------------------------------
// PlanCache basic tests
// ---------------------------------------------------------------------------

#[test]
fn test_plan_cache_hit_and_miss() {
    let cache: PlanCache<String> = PlanCache::new(10);
    cache.insert(42, "plan-a".to_string());
    assert_eq!(cache.get(42).as_deref(), Some("plan-a"));
    assert!(cache.get(99).is_none());
    let (hits, misses, _) = cache.stats();
    assert_eq!(hits, 1);
    assert_eq!(misses, 1);
}

#[test]
fn test_plan_cache_capacity() {
    let cache: PlanCache<u32> = PlanCache::new(3);
    cache.insert(1, 10);
    cache.insert(2, 20);
    cache.insert(3, 30);
    cache.insert(4, 40); // Should evict key 1
    assert!(cache.get(1).is_none(), "key 1 should be evicted");
    assert!(cache.get(4).is_some(), "key 4 should be present");
}

#[test]
fn test_plan_cache_invalidate_all() {
    let cache: PlanCache<String> = PlanCache::new(10);
    cache.insert(1, "plan".to_string());
    assert!(cache.get(1).is_some());
    cache.invalidate_all();
    assert!(cache.get(1).is_none());
    assert_eq!(cache.schema_version(), 1);
}

#[test]
fn test_plan_cache_eviction_counter() {
    let cache: PlanCache<u32> = PlanCache::new(2);
    cache.insert(1, 1);
    cache.insert(2, 2);
    cache.insert(3, 3); // evicts key 1
    let (_, _, evictions) = cache.stats();
    assert_eq!(evictions, 1);
}

#[test]
fn test_plan_cache_concurrent_reads() {
    use std::sync::Arc;
    use std::thread;

    let cache: Arc<PlanCache<u32>> = Arc::new(PlanCache::new(100));
    for i in 0u64..50 {
        cache.insert(i, i as u32);
    }

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let c = Arc::clone(&cache);
            thread::spawn(move || {
                for i in 0u64..50 {
                    let _ = c.get(i);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread should not panic");
    }

    let (hits, _, _) = cache.stats();
    assert!(hits > 0);
}

#[test]
fn test_plan_cache_clone_shares_state() {
    let cache: PlanCache<u32> = PlanCache::new(10);
    let clone = cache.clone();
    cache.insert(7, 42);
    assert_eq!(clone.get(7), Some(42));
}

// ---------------------------------------------------------------------------
// Fingerprint tests
// ---------------------------------------------------------------------------

fn pred() -> Term {
    Term::Iri(NamedNode::new_unchecked("http://example.org/p"))
}

fn var_term(name: &str) -> Term {
    Term::Variable(Variable::new(name).expect("valid var"))
}

fn bgp(s: &str, o: &str) -> Algebra {
    Algebra::Bgp(vec![TriplePattern {
        subject: var_term(s),
        predicate: pred(),
        object: var_term(o),
    }])
}

#[test]
fn test_fingerprint_renamed_variables_collide() {
    let p1 = bgp("x", "y");
    let p2 = bgp("a", "b");
    assert_eq!(
        compute_fingerprint(&p1),
        compute_fingerprint(&p2),
        "structurally identical plans with different variable names should collide"
    );
}

#[test]
fn test_fingerprint_structurally_distinct_do_not_collide() {
    let p1 = bgp("x", "y");
    let p2 = Algebra::Join {
        left: Box::new(bgp("x", "y")),
        right: Box::new(bgp("y", "z")),
    };
    assert_ne!(
        compute_fingerprint(&p1),
        compute_fingerprint(&p2),
        "structurally different plans must have different fingerprints"
    );
}

#[test]
fn test_fingerprint_deterministic() {
    let p = bgp("s", "o");
    let fp1 = compute_fingerprint(&p);
    let fp2 = compute_fingerprint(&p);
    assert_eq!(fp1, fp2);
}

#[test]
fn test_fingerprint_empty_bgp_stable() {
    let p = Algebra::Bgp(vec![]);
    assert_eq!(compute_fingerprint(&p), compute_fingerprint(&p));
}

/// Regression test: `Expression::Function { name: "foo" }` and
/// `Expression::Function { name: "bar" }` must NOT collide.
///
/// Before the needle was anchored to `"Variable { name: \""`, the shorter
/// needle `"name: \""` also matched the `name` field of `Function`, causing
/// `FILTER(foo(?x))` and `FILTER(bar(?x))` to produce the same fingerprint
/// even though they represent distinct query plans.
#[test]
fn test_fingerprint_function_name_not_normalised() {
    // FILTER(?x, foo(?x)) vs FILTER(?x, bar(?x)) — same structure, different function name
    let inner_bgp = bgp("x", "y");

    let foo_filter = Algebra::Filter {
        pattern: Box::new(inner_bgp.clone()),
        condition: Expression::Function {
            name: "foo".to_string(),
            args: vec![Expression::Variable(Variable::new("x").expect("valid var"))],
        },
    };

    let bar_filter = Algebra::Filter {
        pattern: Box::new(inner_bgp),
        condition: Expression::Function {
            name: "bar".to_string(),
            args: vec![Expression::Variable(Variable::new("x").expect("valid var"))],
        },
    };

    assert_ne!(
        compute_fingerprint(&foo_filter),
        compute_fingerprint(&bar_filter),
        "FILTER(foo(?x)) and FILTER(bar(?x)) must have different fingerprints — \
         function names must not be normalised away by the variable normaliser"
    );
}

// ---------------------------------------------------------------------------
// Optimizer integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_optimizer_plan_cache_hit_skips_optimization() {
    let mut optimizer = Optimizer::new(OptimizerConfig::default()).with_plan_cache_capacity(128);
    assert!(optimizer.has_plan_cache());

    let plan = bgp("s", "o");

    // First call — cache miss.
    let result1 = optimizer
        .optimize(plan.clone())
        .expect("optimize should succeed");

    // Second call — should be a cache hit.
    let result2 = optimizer
        .optimize(plan.clone())
        .expect("optimize should succeed");

    // Results should be structurally equal.
    assert_eq!(format!("{result1:?}"), format!("{result2:?}"));

    let (hits, misses, _) = optimizer.plan_cache_stats();
    assert_eq!(hits, 1, "second call should be a cache hit");
    assert_eq!(misses, 1, "first call should be a cache miss");
}

#[test]
fn test_optimizer_plan_cache_renamed_vars_cache_hit() {
    let mut optimizer = Optimizer::new(OptimizerConfig::default()).with_plan_cache_capacity(128);

    let plan_xy = bgp("x", "y");
    let plan_ab = bgp("a", "b");

    optimizer
        .optimize(plan_xy)
        .expect("optimize should succeed");
    optimizer
        .optimize(plan_ab)
        .expect("optimize should succeed");

    let (hits, misses, _) = optimizer.plan_cache_stats();
    assert_eq!(hits, 1, "renamed-variable plan should be a cache hit");
    assert_eq!(misses, 1, "first plan should be a miss");
}

#[test]
fn test_optimizer_plan_cache_invalidate() {
    let mut optimizer = Optimizer::new(OptimizerConfig::default()).with_plan_cache_capacity(128);

    let plan = bgp("s", "o");
    optimizer.optimize(plan.clone()).expect("optimize ok");
    optimizer.invalidate_plan_cache();
    optimizer.optimize(plan).expect("optimize ok");

    // After invalidation the second call should also be a miss.
    let (hits, misses, _) = optimizer.plan_cache_stats();
    assert_eq!(hits, 0, "no hits after invalidation");
    assert_eq!(misses, 2, "both calls should be misses");
}

#[test]
fn test_optimizer_without_cache_works_normally() {
    let mut optimizer = Optimizer::new(OptimizerConfig::default());
    assert!(!optimizer.has_plan_cache());

    let plan = bgp("s", "o");
    let result = optimizer.optimize(plan).expect("optimize should succeed");
    // No panic, no error.
    let _ = result;

    let (hits, misses, _) = optimizer.plan_cache_stats();
    assert_eq!(hits, 0);
    assert_eq!(misses, 0);
}
