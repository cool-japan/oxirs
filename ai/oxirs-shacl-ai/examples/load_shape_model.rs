//! Example: browse the built-in shape-model zoo and (optionally) load a
//! checkpoint from disk.
//!
//! Run with:
//! ```sh
//! cargo run --example load_shape_model
//! ```

fn main() -> anyhow::Result<()> {
    use oxirs_shacl_ai::model_zoo::{ShapeModelZoo, ShapeModelZooLoader};

    // List all bundled models, sorted by name.
    println!("=== Built-in shape-model zoo ===");
    for m in ShapeModelZoo::registry().list() {
        println!(
            "  {name:<30} type={mtype:<18} hidden={h:<4} license={lic}",
            name = m.name,
            mtype = m.model_type,
            h = m.hidden_dim,
            lic = m.license,
        );
    }
    println!();

    // Filter by architecture family.
    for family in ["GAT", "GraphSAGE", "Graphormer", "GraphTransformer"] {
        let hits = ShapeModelZoo::registry().by_model_type(family);
        println!("{family} models: {}", hits.len());
    }
    println!();

    // Search by keyword.
    let matches = ShapeModelZoo::registry().search("lubm");
    println!("Models mentioning 'lubm': {}", matches.len());
    println!();

    // Demonstrate loading (would succeed if seed bytes existed at the given
    // path, which they do not in a fresh checkout — this is expected).
    let loader = ShapeModelZooLoader::new(std::path::Path::new("/nonexistent/seeds"));
    match loader.load("gat-shacl-base") {
        Ok(loaded) => {
            println!(
                "Loaded '{}': {} bytes",
                loaded.manifest.name,
                loaded.weights.len()
            );
        }
        Err(e) => {
            println!("(expected in CI — no seed files on disk): {e}");
        }
    }

    Ok(())
}
