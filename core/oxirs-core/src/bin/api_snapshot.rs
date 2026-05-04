//! Snapshot tool for the oxirs-core public API surface.
//!
//! Parses `src/lib.rs` at build-time (via `env!("CARGO_MANIFEST_DIR")`) and
//! prints the API surface as pretty-printed JSON to stdout.
//!
//! # Usage
//!
//! ```text
//! cargo run -p oxirs-core --bin api_snapshot --quiet > core/oxirs-core/api_baseline.json
//! ```
//!
//! Redirect to `api_baseline.json` and commit the result.  The CI test
//! `api_surface_stable` will then enforce that no breaking changes are made.

fn main() -> anyhow::Result<()> {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let lib_path = manifest_dir.join("src").join("lib.rs");

    let surface = oxirs_core::api_surface::parse_lib(&lib_path)?;
    let json = serde_json::to_string_pretty(&surface)?;
    println!("{json}");
    Ok(())
}
