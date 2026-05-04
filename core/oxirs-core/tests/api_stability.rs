//! API stability regression tests for oxirs-core.
//!
//! These tests guard against unintentional breaking changes in the public API
//! surface of `oxirs-core`.  A committed `api_baseline.json` snapshot is
//! compared against the current `src/lib.rs` on every test run.
//!
//! To update the baseline after an *intentional* API change, run:
//! ```text
//! cargo run -p oxirs-core --bin api_snapshot --quiet \
//!     > core/oxirs-core/api_baseline.json
//! ```

use oxirs_core::api_surface::{diff_surfaces, parse_lib, ApiSurface, TypeSig};
use std::path::Path;

/// Guard: current `src/lib.rs` must not have any items removed relative to the
/// committed baseline.  Additive changes (new items) are always allowed.
#[test]
fn api_surface_stable() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lib_path = manifest_dir.join("src").join("lib.rs");
    let baseline_path = manifest_dir.join("api_baseline.json");

    // Load committed baseline.
    let baseline_json = std::fs::read_to_string(&baseline_path).unwrap_or_else(|_| {
        panic!(
            "api_baseline.json not found at {} — \
             run `cargo run -p oxirs-core --bin api_snapshot --quiet \
             > core/oxirs-core/api_baseline.json`",
            baseline_path.display()
        )
    });
    let baseline: ApiSurface = serde_json::from_str(&baseline_json)
        .unwrap_or_else(|e| panic!("api_baseline.json is malformed: {e}"));

    // Parse current surface.
    let current =
        parse_lib(&lib_path).unwrap_or_else(|e| panic!("failed to parse src/lib.rs: {e}"));

    // Diff and assert no breaking changes.
    let diff = diff_surfaces(&baseline, &current);
    assert!(
        !diff.is_breaking,
        "API breaking change detected:\n{diff}\n\
         If this change is intentional, regenerate the baseline:\n\
         cargo run -p oxirs-core --bin api_snapshot --quiet \
         > core/oxirs-core/api_baseline.json"
    );
}

/// Parse a synthetic fixture file and verify that the surface extraction
/// captures the correct items.
#[test]
fn api_surface_parse_fixture() {
    let dir = std::env::temp_dir().join("oxirs_api_surface_test");
    std::fs::create_dir_all(&dir).expect("failed to create temp dir");
    let fixture = dir.join("fixture_lib.rs");

    std::fs::write(
        &fixture,
        r#"
        pub struct MyStruct;
        pub fn my_fn() {}
        pub trait MyTrait {}
        pub mod my_mod {}
        pub(crate) struct Hidden;
        "#,
    )
    .expect("failed to write fixture");

    let surface = parse_lib(&fixture).expect("parse failed");

    assert!(
        surface.types.iter().any(|t| t.name == "MyStruct"),
        "MyStruct should be in surface; got: {:?}",
        surface.types
    );
    assert!(
        surface.fns.iter().any(|f| f.name == "my_fn"),
        "my_fn should be in surface; got: {:?}",
        surface.fns
    );
    assert!(
        surface.traits.iter().any(|t| t.name == "MyTrait"),
        "MyTrait should be in surface; got: {:?}",
        surface.traits
    );
    assert!(
        surface.modules.iter().any(|m| m == "my_mod"),
        "my_mod should be in surface; got: {:?}",
        surface.modules
    );
    assert!(
        !surface.types.iter().any(|t| t.name == "Hidden"),
        "pub(crate) item should NOT be in surface; got: {:?}",
        surface.types
    );
}

/// A removed type must be reported as a breaking change.
#[test]
fn api_surface_breaking_change_detected() {
    let baseline = ApiSurface {
        types: vec![TypeSig {
            name: "Foo".into(),
            kind: "struct".into(),
            generics: String::new(),
        }],
        ..Default::default()
    };
    let current = ApiSurface::default(); // Foo has been removed.

    let diff = diff_surfaces(&baseline, &current);

    assert!(
        diff.is_breaking,
        "removing a type must be a breaking change"
    );
    assert!(
        diff.removed_types.contains(&"Foo".into()),
        "removed_types should contain Foo; got: {:?}",
        diff.removed_types
    );
}

/// Adding new items to the surface must never be considered breaking.
#[test]
fn api_surface_additive_change_allowed() {
    let baseline = ApiSurface::default();
    let current = ApiSurface {
        types: vec![TypeSig {
            name: "Bar".into(),
            kind: "struct".into(),
            generics: String::new(),
        }],
        ..Default::default()
    };

    let diff = diff_surfaces(&baseline, &current);

    assert!(
        !diff.is_breaking,
        "adding items must not be a breaking change; diff: {diff}"
    );
}

/// The `ApiSurface` type must round-trip through JSON losslessly.
#[test]
fn api_surface_json_roundtrip() {
    let surface = ApiSurface {
        types: vec![TypeSig {
            name: "X".into(),
            kind: "enum".into(),
            generics: "<T>".into(),
        }],
        ..Default::default()
    };

    let json = serde_json::to_string(&surface).expect("serialization failed");
    let back: ApiSurface = serde_json::from_str(&json).expect("deserialization failed");

    assert_eq!(back.types[0].name, "X");
    assert_eq!(back.types[0].kind, "enum");
    assert_eq!(back.types[0].generics, "<T>");
}

/// A changed function signature must be reported as a breaking change.
#[test]
fn api_surface_changed_fn_is_breaking() {
    use oxirs_core::api_surface::FnSig;

    let baseline = ApiSurface {
        fns: vec![FnSig {
            name: "do_thing".into(),
            signature: "fn do_thing () -> u32".into(),
        }],
        ..Default::default()
    };
    let current = ApiSurface {
        fns: vec![FnSig {
            name: "do_thing".into(),
            signature: "fn do_thing (x : u32) -> u32".into(), // parameter added
        }],
        ..Default::default()
    };

    let diff = diff_surfaces(&baseline, &current);

    assert!(
        diff.is_breaking,
        "a changed function signature must be breaking"
    );
    assert_eq!(
        diff.changed_fns.len(),
        1,
        "changed_fns should have one entry"
    );
}

/// A removed module must be reported as a breaking change.
#[test]
fn api_surface_removed_module_is_breaking() {
    let baseline = ApiSurface {
        modules: vec!["my_mod".into()],
        ..Default::default()
    };
    let current = ApiSurface::default();

    let diff = diff_surfaces(&baseline, &current);

    assert!(diff.is_breaking, "removing a module must be breaking");
    assert!(
        diff.removed_modules.contains(&"my_mod".into()),
        "removed_modules should contain my_mod; got: {:?}",
        diff.removed_modules
    );
}
