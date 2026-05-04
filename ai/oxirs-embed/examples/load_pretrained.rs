//! Demonstration of the model zoo API.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example load_pretrained -p oxirs-embed
//! ```
//!
//! Note: the built-in catalog entries carry `sha256 = "PLACEHOLDER"` and
//! point to `file:///seeds/…` paths that do not ship inside the crate.
//! This example shows the listing and search APIs, which work without any
//! checkpoint files on disk.

use oxirs_embed::model_zoo::{ModelZoo, ModelZooLoader};

fn main() -> anyhow::Result<()> {
    // ------------------------------------------------------------------
    // 1. List all catalog entries
    // ------------------------------------------------------------------
    println!("=== Built-in Model Zoo Catalog ===");
    let zoo = ModelZoo::registry();
    let mut entries = zoo.list();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    for m in &entries {
        println!(
            "  {:30}  type={:12}  dataset={:10}  dims={:3}  license={}",
            m.name, m.model_type, m.dataset, m.dimensions, m.license
        );
    }
    println!();

    // ------------------------------------------------------------------
    // 2. Search by keyword
    // ------------------------------------------------------------------
    println!("=== Search: 'transe' ===");
    let mut results = zoo.search("transe");
    results.sort_by(|a, b| a.name.cmp(&b.name));
    for m in &results {
        println!("  {} ({})", m.name, m.dataset);
    }
    println!();

    println!("=== Search: 'FB15k' ===");
    let mut results = zoo.search("FB15k");
    results.sort_by(|a, b| a.name.cmp(&b.name));
    for m in &results {
        println!("  {} — {}", m.name, m.citation);
    }
    println!();

    // ------------------------------------------------------------------
    // 3. Attempt to load a model (will fail because the seed files are
    //    not shipped; this illustrates the error type).
    // ------------------------------------------------------------------
    println!("=== Load attempt (expects IO error — seed file not on disk) ===");
    let loader = ModelZooLoader::new(std::env::temp_dir()).accept_license();
    match loader.load("transe-fb15k237") {
        Ok(_model) => println!("  Model loaded successfully."),
        Err(e) => println!("  Expected error: {e}"),
    }

    println!("\nDone.");
    Ok(())
}
