//! Example demonstrating arena-based memory management for RDF data

use oxirs_core::model::*;
use oxirs_core::store::{ConcurrentArena, GraphArena, LocalArena, ScopedArena};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Arena-Based Memory Management Example ===\n");

    example_local_arena()?;
    example_concurrent_arena()?;
    example_graph_arena()?;
    example_scoped_arena()?;
    benchmark_arena_vs_heap()?;

    Ok(())
}

fn example_local_arena() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Local Arena Allocation");
    println!("{}", "-".repeat(40));

    let arena = LocalArena::with_capacity(4096);

    // Allocate strings
    let s1 = arena.alloc_str("http://example.org/subject");
    let s2 = arena.alloc_str("http://example.org/predicate");
    let s3 = arena.alloc_str("Object literal value");

    println!("Allocated strings:");
    println!("  s1: {}", s1.as_str());
    println!("  s2: {}", s2.as_str());
    println!("  s3: {}", s3.as_str());

    // Allocate terms
    let term1 = Term::NamedNode(NamedNode::new("http://example.org/resource")?);
    let term2 = Term::Literal(Literal::new("Hello, Arena!"));
    let term3 = Term::BlankNode(BlankNode::new("b1")?);

    let arena_term1 = arena.alloc_term(&term1);
    let arena_term2 = arena.alloc_term(&term2);
    let arena_term3 = arena.alloc_term(&term3);

    println!("\nAllocated terms:");
    if let oxirs_core::store::ArenaTerm::NamedNode(s) = arena_term1 {
        println!("  NamedNode: {}", s.as_str());
    }
    if let oxirs_core::store::ArenaTerm::Literal { value, .. } = arena_term2 {
        println!("  Literal: {}", value.as_str());
    }
    if let oxirs_core::store::ArenaTerm::BlankNode(s) = arena_term3 {
        println!("  BlankNode: {}", s.as_str());
    }

    println!("\nTotal allocated: {} bytes", arena.allocated_bytes());
    println!();

    Ok(())
}

fn example_concurrent_arena() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Concurrent Arena Allocation");
    println!("{}", "-".repeat(40));

    let arena = Arc::new(ConcurrentArena::new(1024));

    // Simulate concurrent allocation
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let arena = Arc::clone(&arena);
            std::thread::spawn(move || {
                for i in 0..10 {
                    let uri = format!("http://example.org/thread{thread_id}/resource{i}");
                    let allocated = arena.alloc_str(&uri);
                    if i == 0 {
                        println!("Thread {thread_id}: allocated '{allocated}'");
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    println!("\nTotal allocated: {} bytes", arena.total_allocated());
    println!("Number of arenas: {}", arena.arena_count());
    println!();

    Ok(())
}

fn example_graph_arena() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Graph Arena with Term Caching");
    println!("{}", "-".repeat(40));

    let arena = GraphArena::with_capacity(8192);

    // Create some terms - duplicates will be cached
    let terms = vec![
        Term::NamedNode(NamedNode::new("http://example.org/Person")?),
        Term::NamedNode(NamedNode::new("http://example.org/name")?),
        Term::Literal(Literal::new("Alice")),
        Term::NamedNode(NamedNode::new("http://example.org/Person")?), // Duplicate
        Term::Literal(Literal::new("Bob")),
        Term::NamedNode(NamedNode::new("http://example.org/name")?), // Duplicate
    ];

    println!("Allocating {} terms (with duplicates):", terms.len());
    for (i, term) in terms.iter().enumerate() {
        let allocated = arena.alloc_term(term);
        match (&term, &allocated) {
            (Term::NamedNode(n), _) => println!("  [{}] NamedNode: {}", i, n.as_str()),
            (Term::Literal(l), _) => println!("  [{}] Literal: {}", i, l.value()),
            _ => {}
        }
    }

    println!("\nUnique terms cached: {}", arena.cached_terms());
    println!("Total allocated: {} bytes", arena.allocated_bytes());

    // Allocate a triple
    let triple = Triple::new(
        NamedNode::new("http://example.org/alice")?,
        NamedNode::new("http://example.org/knows")?,
        NamedNode::new("http://example.org/bob")?,
    );

    let arena_triple = arena.alloc_triple(&triple);
    println!("\nAllocated triple:");
    if let oxirs_core::store::ArenaTerm::NamedNode(s) = arena_triple.subject {
        print!("  Subject: {}", s.as_str());
    }
    print!(" -> {}", arena_triple.predicate.as_str());
    if let oxirs_core::store::ArenaTerm::NamedNode(s) = arena_triple.object {
        println!(" -> Object: {}", s.as_str());
    }

    println!();
    Ok(())
}

fn example_scoped_arena() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 4: Scoped Arena for Temporary Allocations");
    println!("{}", "-".repeat(40));

    let parent_arena = LocalArena::new();

    // Initial allocation in parent
    parent_arena.alloc_str("parent allocation");
    let initial_bytes = parent_arena.allocated_bytes();
    println!("Parent arena initial: {initial_bytes} bytes");

    // Create a scope for temporary allocations
    {
        let scoped = ScopedArena::new(&parent_arena);

        // Allocate in the scope
        for i in 0..5 {
            scoped.alloc_str(&format!("scoped allocation {i}"));
        }

        println!("Scoped allocations: {} bytes", scoped.scope_allocated());
    }

    println!(
        "Parent arena after scope: {} bytes",
        parent_arena.allocated_bytes()
    );
    println!();

    Ok(())
}

fn benchmark_arena_vs_heap() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 5: Arena vs Heap Allocation Performance");
    println!("{}", "-".repeat(40));

    const NUM_ALLOCATIONS: usize = 10_000;

    // Benchmark heap allocation
    let start = Instant::now();
    let mut heap_allocated = Vec::with_capacity(NUM_ALLOCATIONS);
    for i in 0..NUM_ALLOCATIONS {
        let term = Term::NamedNode(NamedNode::new(format!("http://example.org/resource{i}"))?);
        heap_allocated.push(term);
    }
    let heap_duration = start.elapsed();

    // Benchmark arena allocation
    let arena = LocalArena::with_capacity(1024 * 1024); // 1MB
    let start = Instant::now();
    for i in 0..NUM_ALLOCATIONS {
        let term = Term::NamedNode(NamedNode::new(format!("http://example.org/resource{i}"))?);
        arena.alloc_term(&term);
    }
    let arena_duration = start.elapsed();

    println!("Allocations: {NUM_ALLOCATIONS}");
    println!("Heap allocation time: {heap_duration:?}");
    println!("Arena allocation time: {arena_duration:?}");
    println!("Arena allocated bytes: {}", arena.allocated_bytes());

    let speedup = heap_duration.as_secs_f64() / arena_duration.as_secs_f64();
    println!("Arena speedup: {speedup:.2}x faster");

    // Memory fragmentation comparison
    println!("\nMemory characteristics:");
    println!("  Heap: {NUM_ALLOCATIONS} individual allocations (fragmented)");
    println!("  Arena: 1 contiguous allocation (cache-friendly)");

    Ok(())
}
