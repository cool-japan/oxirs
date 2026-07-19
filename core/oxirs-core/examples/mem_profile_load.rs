//! Memory-footprint measurement harness for the persistent store load path.
//!
//! Opens a `Persistent` [`RdfStore`] over an existing `data.nq` dataset directory
//! and reports the interned in-memory footprint plus the process resident set,
//! so the before/after of the `rdf_store` memory-optimization work can be
//! measured on a real dataset.
//!
//! Usage:
//! ```text
//!   cargo run --release --example mem_profile_load -- <dataset_dir> [--shrink]
//!   cargo run --release --features dhat-heap --example mem_profile_load -- <dataset_dir> [--shrink]
//! ```
//!
//! * `<dataset_dir>` must contain a `data.nq` file. Use a *copy* of the source
//!   dataset — `RdfStore::open` opens an append writer and may normalize the
//!   trailing newline, so never point this at a read-only source of record.
//! * `--shrink` calls [`RdfStore::shrink_to_fit`] after the load to measure the
//!   effect of releasing over-provisioned dictionary capacity.
//!
//! Two independent numbers are reported and should be gathered in *separate*
//! runs:
//! * **Steady RSS** — `VmRSS` sampled from `/proc/self/status` after the load
//!   completes and while the store is still alive. This is the number the
//!   optimization targets. Run a *plain* `--release` build (no `dhat-heap`) for a
//!   realistic figure.
//! * **dhat heap stats** — with `--features dhat-heap`, the peak
//!   heap-bytes-requested reported by the pure-Rust dhat profiler. dhat perturbs
//!   allocation, so do not read RSS from a dhat run.
//!
//! Wrap the whole run in `/usr/bin/time -v` to also capture Maximum resident set
//! size (peak RSS).

use anyhow::{Context, Result};
use oxirs_core::RdfStore;
use std::time::Instant;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Read `VmRSS` (resident set size) for the current process from
/// `/proc/self/status`, in kibibytes. Returns `None` off Linux or on any read
/// error.
fn read_vm_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kib: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())?;
            return Some(kib);
        }
    }
    None
}

/// Peak `VmHWM` (high-water-mark resident set) for the current process, in KiB.
fn read_vm_hwm_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kib: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())?;
            return Some(kib);
        }
    }
    None
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<()> {
    #[cfg(feature = "dhat-heap")]
    let _dhat = dhat::Profiler::new_heap();

    let mut args = std::env::args().skip(1);
    let dataset_dir = args
        .next()
        .context("usage: mem_profile_load <dataset_dir> [--shrink]")?;
    let shrink = args.any(|a| a == "--shrink");

    println!("== mem_profile_load ==");
    println!("dataset dir : {dataset_dir}");
    println!("shrink      : {shrink}");
    #[cfg(feature = "dhat-heap")]
    println!("dhat-heap   : ON (RSS figures from this run are NOT representative)");

    let rss_before = read_vm_rss_kib().unwrap_or(0);
    let start = Instant::now();

    let store = RdfStore::open(&dataset_dir).context("open persistent store")?;

    let load_elapsed = start.elapsed();
    let len = store.len().context("len")?;
    let rss_after_load = read_vm_rss_kib().unwrap_or(0);

    let (size_est, rss_after_shrink, shrink_elapsed) = if shrink {
        let s = Instant::now();
        store.shrink_to_fit().context("shrink_to_fit")?;
        let el = s.elapsed();
        (
            store.interned_size_estimate().unwrap_or(0),
            read_vm_rss_kib().unwrap_or(0),
            Some(el),
        )
    } else {
        (
            store.interned_size_estimate().unwrap_or(0),
            rss_after_load,
            None,
        )
    };

    let peak_hwm = read_vm_hwm_kib().unwrap_or(0);

    println!("---");
    println!("quads loaded        : {len}");
    println!("load time           : {:.2}s", load_elapsed.as_secs_f64());
    if let Some(el) = shrink_elapsed {
        println!("shrink time         : {:.3}s", el.as_secs_f64());
    }
    println!(
        "interned size_est   : {} bytes ({:.1} MiB) = {:.0} bytes/quad",
        size_est,
        mib(size_est as u64),
        if len > 0 {
            size_est as f64 / len as f64
        } else {
            0.0
        }
    );
    println!("RSS before open     : {:.1} MiB", mib(rss_before * 1024));
    println!(
        "RSS after load      : {:.1} MiB ({:.0} bytes/quad)",
        mib(rss_after_load * 1024),
        if len > 0 {
            (rss_after_load * 1024) as f64 / len as f64
        } else {
            0.0
        }
    );
    if shrink {
        println!(
            "RSS after shrink    : {:.1} MiB ({:.0} bytes/quad)",
            mib(rss_after_shrink * 1024),
            if len > 0 {
                (rss_after_shrink * 1024) as f64 / len as f64
            } else {
                0.0
            }
        );
    }
    println!("peak RSS (VmHWM)    : {:.1} MiB", mib(peak_hwm * 1024));

    #[cfg(feature = "dhat-heap")]
    {
        let stats = dhat::HeapStats::get();
        println!("---dhat heap---");
        println!(
            "dhat max heap bytes : {} ({:.1} MiB) in {} blocks",
            stats.max_bytes,
            mib(stats.max_bytes as u64),
            stats.max_blocks
        );
        println!(
            "dhat total alloc    : {} bytes over {} allocations",
            stats.total_bytes, stats.total_blocks
        );
    }

    // Keep the store alive until after every measurement above.
    std::hint::black_box(&store);
    Ok(())
}
