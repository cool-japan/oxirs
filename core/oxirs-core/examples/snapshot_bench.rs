//! Cold-start measurement harness for the frozen mmap snapshot format.
//!
//! Usage (each `load*` invocation is a separate process so `/usr/bin/time -v`
//! attributes peak RSS and wall time to exactly one load path):
//!
//! ```text
//!   snapshot_bench load  <dataset_dir>   # RdfStore::open (uses snapshot if present)
//!   snapshot_bench build <dataset_dir>   # bake snapshot.oxsnap next to data.nq
//! ```
//!
//! `<dataset_dir>` must contain `data.nq`. `load` prints the quad COUNT and a few
//! fixed spot-query counts so the snapshot and traditional paths can be compared.

use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use oxirs_core::model::{NamedNode, Predicate};
use oxirs_core::RdfStore;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("");
    let dir = match args.get(2) {
        Some(d) => d.clone(),
        None => {
            eprintln!("usage: snapshot_bench <load|build> <dataset_dir>");
            std::process::exit(2);
        }
    };

    match mode {
        "build" => {
            let started = Instant::now();
            let path = RdfStore::build_snapshot(&dir)?;
            println!(
                "built snapshot {} in {:.3}s",
                path.display(),
                started.elapsed().as_secs_f64()
            );
        }
        "load" => {
            let started = Instant::now();
            let store = RdfStore::open(&dir)?;
            let elapsed = started.elapsed();
            let count = store.len()?;
            println!("COUNT={count}");
            println!("cold_start_seconds={:.3}", elapsed.as_secs_f64());

            // Order-independent content digest: fold a per-quad Debug hash (which
            // captures every term field — IRI, literal value/language/datatype,
            // blank/variant) with wrapping addition (commutative, so no sort). The
            // traditional and snapshot loads MUST print the same digest — a real
            // "identical terms, not just identical counts" proof.
            let mut digest: u64 = 0;
            for quad in store.quads()? {
                let mut hasher = DefaultHasher::new();
                format!("{quad:?}").hash(&mut hasher);
                digest = digest.wrapping_add(hasher.finish());
            }
            println!("quad_digest=0x{digest:016x}");

            // A few fixed spot queries: predicate-led scans over predicates that
            // appear in the wik dataset. Counts must match across load paths.
            for predicate in [
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://www.w3.org/2004/02/skos/core#prefLabel",
                "http://www.w3.org/2004/02/skos/core#broader",
            ] {
                let p = Predicate::NamedNode(NamedNode::new(predicate)?);
                let n = store.query_quads(None, Some(&p), None, None)?.len();
                println!("predicate_count[{predicate}]={n}");
            }
        }
        other => {
            eprintln!("unknown mode {other:?}; use load|build");
            std::process::exit(2);
        }
    }
    Ok(())
}
