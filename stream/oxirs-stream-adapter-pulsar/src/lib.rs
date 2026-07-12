//! Quarantined Apache Pulsar backend adapter for `oxirs-stream`.
//!
//! # Why this crate exists
//!
//! Under the COOLJAPAN **Pure Rust Policy v2** (purity is measured on the full
//! `--all-features` dependency closure), a *published* crate must not drag in
//! C FFI. The real Pulsar backend uses the [`pulsar`] crate, whose default
//! runtime pulls in `native-tls` and whose default `compression` feature pulls
//! in `lz4` → `lz4-sys` (and `zstd`). Critically, pulsar 6.x exposes **no**
//! Pure-Rust TLS configuration — every runtime feature requires `native-tls`,
//! `aws-lc-rs`, or `ring` — so a feature swap to rustls is impossible. To keep
//! the published `oxirs-stream` surface 100% Pure Rust while *preserving* the
//! real Pulsar capability, the live `pulsar`-backed backend has been
//! **quarantined** into this crate.
//!
//! This crate is `publish = false`: it never ships to crates.io, so its C FFI
//! dependency never appears in the published Pure-Rust surface. Binaries that
//! actually want a Pulsar backend depend on this crate directly.
//!
//! # Relationship to `oxirs-stream`
//!
//! [`PulsarProducer`] / [`PulsarConsumer`] are constructed from
//! [`oxirs_stream::StreamConfig`] with an
//! [`oxirs_stream::StreamBackendType::Pulsar`] backend selector, and consume /
//! emit [`oxirs_stream::StreamEvent`] values — identical to the published types.
//!
//! [`pulsar`]: https://docs.rs/pulsar

/// Pulsar producer/consumer/admin implementation (named `backend` rather than
/// `pulsar` to avoid colliding with the external `pulsar` crate).
pub mod backend;

pub use backend::*;
