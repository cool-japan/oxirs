//! RCC8 (Region Connection Calculus) relations example (relocated from
//! oxirs-geosparql under the Pure Rust Policy v2 quarantine of the GEOS C FFI).
//!
//! `rcc8_ec`/`rcc8_tpp`/`rcc8_tppi`/`rcc8_ntpp`/`rcc8_ntppi` come from this GEOS
//! adapter (they need a geometric boundary); `rcc8_eq`/`rcc8_dc`/`rcc8_po` are Pure
//! Rust and come from the parent crate.
//!
//! Run with: cargo run -p oxirs-geosparql-adapter-geos --example rcc8_relations
//! (requires the GEOS C library to be installed).

use oxirs_geosparql::error::Result;
use oxirs_geosparql::functions::rcc8::{rcc8_dc, rcc8_eq, rcc8_po};
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql_adapter_geos::{rcc8_ec, rcc8_ntpp, rcc8_ntppi, rcc8_tpp, rcc8_tppi};

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL RCC8 Relations Example (GEOS adapter) ===\n");

    // EQ / DC / PO (Pure Rust)
    let r1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;
    let r2 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;
    println!("EQ    -> {}", rcc8_eq(&r1, &r2)?);

    let d1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))")?;
    let d2 = Geometry::from_wkt("POLYGON((4 4, 6 4, 6 6, 4 6, 4 4))")?;
    println!("DC    -> {}", rcc8_dc(&d1, &d2)?);

    let p1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;
    let p2 = Geometry::from_wkt("POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))")?;
    println!("PO    -> {}", rcc8_po(&p1, &p2)?);

    // EC (GEOS adapter)
    let e1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))")?;
    let e2 = Geometry::from_wkt("POLYGON((2 0, 4 0, 4 2, 2 2, 2 0))")?;
    println!("EC    -> {}", rcc8_ec(&e1, &e2)?);

    // TPP / TPPi (GEOS adapter)
    let outer = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
    let inner = Geometry::from_wkt("POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))")?;
    println!("TPP   -> {}", rcc8_tpp(&inner, &outer)?);
    println!("TPPi  -> {}", rcc8_tppi(&outer, &inner)?);

    // NTPP / NTPPi (GEOS adapter)
    let big = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
    let tiny = Geometry::from_wkt("POLYGON((3 3, 7 3, 7 7, 3 7, 3 3))")?;
    println!("NTPP  -> {}", rcc8_ntpp(&tiny, &big)?);
    println!("NTPPi -> {}", rcc8_ntppi(&big, &tiny)?);

    println!("\n=== Example completed successfully! ===");
    Ok(())
}
