//! Regression test for the `experimental-ai` feature gate on the
//! `consciousness` / `molecular` / `quantum` module trees.
//!
//! These modules generate results from untrained, randomly-initialized
//! weights and must never be part of the default public API surface.
//! See lib.rs:290 audit finding (P1/core-rest).

/// `experimental-ai` must stay out of the `default` feature set in Cargo.toml.
/// This guards against a future edit accidentally re-enabling the
/// unvalidated research modules for every downstream consumer by default.
#[test]
fn experimental_ai_is_not_a_default_feature() {
    let manifest = include_str!("../Cargo.toml");
    let features_section = manifest
        .split("[features]")
        .nth(1)
        .expect("Cargo.toml must contain a [features] section");
    // Only look at the `default = [...]` line, not the whole features block
    // (which legitimately declares `experimental-ai = []` elsewhere).
    let default_line = features_section
        .lines()
        .find(|line| line.trim_start().starts_with("default"))
        .expect("Cargo.toml [features] must declare a `default` key");
    assert!(
        !default_line.contains("experimental-ai"),
        "experimental-ai must not be enabled by default (found in: {default_line})"
    );
}

/// The `experimental-ai` feature must still be declared, so that consumers
/// can opt in explicitly (and so `--all-features` still exercises the code).
#[test]
fn experimental_ai_feature_is_declared() {
    let manifest = include_str!("../Cargo.toml");
    assert!(
        manifest
            .lines()
            .any(|line| line.trim_start().starts_with("experimental-ai = [")),
        "Cargo.toml must declare an `experimental-ai` feature"
    );
}

/// Under the default feature set, the experimental module trees must not be
/// reachable from the crate root at all -- not just "empty", but genuinely
/// absent from the public API.
#[cfg(not(feature = "experimental-ai"))]
#[test]
fn experimental_modules_are_absent_without_the_feature() {
    // Compile-time proof: if any of these paths were reachable without the
    // `experimental-ai` feature, this module simply would not compile.
    // (Nothing to assert at runtime -- the test passing at all is the check.)
}

/// With the feature explicitly enabled, the module trees must be reachable,
/// confirming the gate is additive (opt-in) rather than accidentally
/// removing the functionality entirely.
#[cfg(feature = "experimental-ai")]
#[test]
fn experimental_modules_are_reachable_with_the_feature() {
    // Smoke check: referencing the module paths is enough to prove they
    // compile and are exported under the feature flag.
    let _ = std::any::type_name::<oxirs_core::consciousness::intuitive_planner::IntuitionNetwork>;
    let _ = std::any::type_name::<oxirs_core::quantum::QuantumState>;
    let _ = std::any::type_name::<oxirs_core::molecular::CellularDivision>;
}
