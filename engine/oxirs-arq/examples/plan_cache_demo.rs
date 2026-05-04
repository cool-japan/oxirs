//! Demonstration of the algebra-level JIT plan cache (phase a).
//!
//! Shows how the `PlanCache` is used via the `Optimizer` API and
//! how repeated structurally-identical queries (even with different
//! variable names) are served from cache without re-optimizing.

use oxirs_arq::algebra::{Algebra, Term, TriplePattern, Variable};
use oxirs_arq::optimizer::{Optimizer, OptimizerConfig};
use oxirs_arq::plan_cache::compute_fingerprint;
use oxirs_core::model::NamedNode;

fn make_bgp(s: &str, p_iri: &str, o: &str) -> Algebra {
    Algebra::Bgp(vec![TriplePattern {
        subject: Term::Variable(Variable::new(s).expect("valid var")),
        predicate: Term::Iri(NamedNode::new_unchecked(p_iri)),
        object: Term::Variable(Variable::new(o).expect("valid var")),
    }])
}

fn main() {
    println!("=== OxiRS ARQ — Algebra-level Plan Cache Demo ===\n");

    // Build an optimizer with a 1024-entry plan cache.
    let mut optimizer = Optimizer::new(OptimizerConfig::default()).with_plan_cache_capacity(1024);

    println!(
        "Plan cache enabled: {}",
        if optimizer.has_plan_cache() {
            "yes"
        } else {
            "no"
        }
    );

    // -----------------------------------------------------------------------
    // Query 1 — first time (cache miss)
    // -----------------------------------------------------------------------
    let q1 = make_bgp("s", "http://example.org/knows", "o");
    let fp1 = compute_fingerprint(&q1);
    println!("\n[1] Query 1 fingerprint: {fp1:016x}");

    let result1 = optimizer.optimize(q1.clone()).expect("optimize q1");
    let (hits, misses, evictions) = optimizer.plan_cache_stats();
    println!("    After first optimize: hits={hits}, misses={misses}, evictions={evictions}");

    // -----------------------------------------------------------------------
    // Query 2 — same structure, different variable names (cache hit)
    // -----------------------------------------------------------------------
    let q2 = make_bgp("subject", "http://example.org/knows", "object");
    let fp2 = compute_fingerprint(&q2);
    println!("\n[2] Query 2 fingerprint: {fp2:016x}");
    println!(
        "    Fingerprints match: {}",
        if fp1 == fp2 { "yes" } else { "no" }
    );

    let result2 = optimizer.optimize(q2.clone()).expect("optimize q2");
    let (hits, misses, evictions) = optimizer.plan_cache_stats();
    println!("    After second optimize: hits={hits}, misses={misses}, evictions={evictions}");

    // -----------------------------------------------------------------------
    // Query 3 — structurally different plan (cache miss)
    // -----------------------------------------------------------------------
    let q3 = Algebra::Join {
        left: Box::new(make_bgp("s", "http://example.org/knows", "o")),
        right: Box::new(make_bgp("o", "http://example.org/likes", "t")),
    };
    let fp3 = compute_fingerprint(&q3);
    println!("\n[3] Query 3 fingerprint: {fp3:016x}");
    println!(
        "    Fingerprint differs from q1: {}",
        if fp3 != fp1 { "yes" } else { "no" }
    );

    let _result3 = optimizer.optimize(q3).expect("optimize q3");
    let (hits, misses, evictions) = optimizer.plan_cache_stats();
    println!("    After third optimize: hits={hits}, misses={misses}, evictions={evictions}");

    // -----------------------------------------------------------------------
    // Schema invalidation demo
    // -----------------------------------------------------------------------
    println!("\n[4] Invalidating plan cache (simulating schema change)...");
    optimizer.invalidate_plan_cache();

    let _result4 = optimizer
        .optimize(q1)
        .expect("optimize q1 after invalidation");
    let (hits, misses, evictions) = optimizer.plan_cache_stats();
    println!(
        "    After invalidation + re-optimize: hits={hits}, misses={misses}, evictions={evictions}"
    );

    // -----------------------------------------------------------------------
    // Results equality sanity check
    // -----------------------------------------------------------------------
    println!(
        "\n[5] Result equality check (q1 == q2): {}",
        if format!("{result1:?}") == format!("{result2:?}") {
            "match"
        } else {
            "differ (unexpected)"
        }
    );

    println!("\n=== Demo complete ===");
}
