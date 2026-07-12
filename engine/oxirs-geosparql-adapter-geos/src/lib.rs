//! # oxirs-geosparql-adapter-geos
//!
//! Quarantined **GEOS** (the [GEOS](https://libgeos.org/) C library) computational-
//! geometry backend for [`oxirs_geosparql`].
//!
//! This crate isolates the [`geos`] crate — which pulls the `geos-sys` C FFI
//! (raw bindings to the GEOS C library) — into a `publish = false` adapter so that
//! the published `oxirs-geosparql` crate keeps a 100% Pure-Rust `--all-features`
//! dependency surface, per the **COOLJAPAN Pure Rust Policy v2**.
//!
//! ## What lives here
//!
//! `oxirs-geosparql`'s default (and only) build is Pure Rust: it computes spatial
//! relations with the `geo`/`wkt` crates and (optionally) the Pure-Rust
//! `geo-buffer` straight-skeleton buffer (Polygon/MultiPolygon only). GEOS used to
//! be an optional, non-default `geos-backend` feature providing the operations the
//! Pure-Rust path does not fully cover. Those operations now live here:
//!
//! - [`buffer`] / [`buffer_with_params`]: robust buffering for **all** geometry
//!   types (the Pure-Rust buffer only handles Polygon/MultiPolygon) plus the full
//!   OGC cap/join styles (flat/square caps, mitre/bevel joins, mitre limit,
//!   quadrant segments).
//! - [`boundary`]: OGC Simple Features boundary via GEOS.
//! - Egenhofer boundary-dependent relations: [`eh_meet`], [`eh_inside`],
//!   [`eh_contains`].
//! - RCC8 boundary-dependent relations: [`rcc8_ec`], [`rcc8_tpp`], [`rcc8_tppi`],
//!   [`rcc8_ntpp`], [`rcc8_ntppi`].
//!
//! All functions operate on [`oxirs_geosparql::geometry::Geometry`] and reuse the
//! parent's public Pure-Rust helpers (`bbox_utils`, `simple_features`,
//! `geometric_operations::intersection`) so results stay consistent with the
//! parent's pure relations.
//!
//! ## Build behavior
//!
//! The [`geos`] crate links the system GEOS C library (`libgeos`) and discovers it
//! via `geos-config` / `pkg-config`. Building this adapter therefore requires GEOS
//! to be installed on the host:
//!
//! - macOS:   `brew install geos`
//! - Ubuntu:  `sudo apt-get install libgeos-dev`
//! - Fedora:  `sudo dnf install geos-devel`
//!
//! [`geos`]: https://crates.io/crates/geos

pub mod buffer;
pub mod egenhofer;
pub mod rcc8;

pub use buffer::{boundary, buffer, buffer_with_params};
pub use egenhofer::{eh_contains, eh_inside, eh_meet};
pub use rcc8::{rcc8_ec, rcc8_ntpp, rcc8_ntppi, rcc8_tpp, rcc8_tppi};
