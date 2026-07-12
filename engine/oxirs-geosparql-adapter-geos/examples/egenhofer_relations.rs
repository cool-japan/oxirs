//! Egenhofer topological relations example (relocated from oxirs-geosparql under
//! the Pure Rust Policy v2 quarantine of the GEOS C FFI).
//!
//! `eh_meet`/`eh_inside`/`eh_contains` come from this GEOS adapter (they need a
//! geometric boundary); the other five Egenhofer relations are Pure Rust and come
//! from the parent crate.
//!
//! Run with: cargo run -p oxirs-geosparql-adapter-geos --example egenhofer_relations
//! (requires the GEOS C library to be installed).

use oxirs_geosparql::error::Result;
use oxirs_geosparql::functions::egenhofer::{
    eh_covered_by, eh_covers, eh_disjoint, eh_equals, eh_overlap,
};
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql_adapter_geos::{eh_contains, eh_inside, eh_meet};

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL Egenhofer Relations Example (GEOS adapter) ===\n");

    // 1. eh_equals (Pure Rust)
    let poly1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;
    let poly2 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;
    println!("1. eh_equals    -> {}", eh_equals(&poly1, &poly2)?);

    // 2. eh_disjoint (Pure Rust)
    let a = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))")?;
    let b = Geometry::from_wkt("POLYGON((3 3, 5 3, 5 5, 3 5, 3 3))")?;
    println!("2. eh_disjoint  -> {}", eh_disjoint(&a, &b)?);

    // 3. eh_meet (GEOS adapter)
    let m1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))")?;
    let m2 = Geometry::from_wkt("POLYGON((2 0, 4 0, 4 2, 2 2, 2 0))")?;
    println!("3. eh_meet      -> {}", eh_meet(&m1, &m2)?);

    // 4. eh_overlap (Pure Rust)
    let o1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;
    let o2 = Geometry::from_wkt("POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))")?;
    println!("4. eh_overlap   -> {}", eh_overlap(&o1, &o2)?);

    // 5/6. eh_covers / eh_covered_by (Pure Rust)
    let large = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
    let small = Geometry::from_wkt("POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))")?;
    println!("5. eh_covers    -> {}", eh_covers(&large, &small)?);
    println!("6. eh_coveredBy -> {}", eh_covered_by(&small, &large)?);

    // 7/8. eh_inside / eh_contains (GEOS adapter)
    let outer = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
    let inner = Geometry::from_wkt("POLYGON((3 3, 7 3, 7 7, 3 7, 3 3))")?;
    println!("7. eh_inside    -> {}", eh_inside(&inner, &outer)?);
    println!("8. eh_contains  -> {}", eh_contains(&outer, &inner)?);

    println!("\n=== Example completed successfully! ===");
    Ok(())
}
