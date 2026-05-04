//! API surface tracking for oxirs-core.
//!
//! Parses the public API surface from `lib.rs` using `syn`, compares against a
//! committed JSON baseline, and reports breaking changes.  Only top-level `pub`
//! items (not `pub(crate)`) and non-`#[doc(hidden)]` items are included.
//!
//! # Workflow
//!
//! 1. **Generate baseline** (after an intentional API change):
//!    ```text
//!    cargo run -p oxirs-core --bin api_snapshot --quiet > core/oxirs-core/api_baseline.json
//!    ```
//! 2. **CI guard**: `tests/api_stability.rs` loads the baseline and asserts the
//!    current surface matches it — any removed or changed item fails the test.

use std::path::Path;

use quote::{quote, ToTokens};
use serde::{Deserialize, Serialize};
use syn::{File, Item, Visibility};

// ─── Data types ──────────────────────────────────────────────────────────────

/// A public struct, enum, or type alias in the API surface.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TypeSig {
    /// Item name.
    pub name: String,
    /// One of `"struct"`, `"enum"`, or `"type"`.
    pub kind: String,
    /// Stringified generics clause, e.g. `"< T : Clone >"`.
    pub generics: String,
}

/// A public free function in the API surface.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FnSig {
    /// Function name.
    pub name: String,
    /// Stringified function signature (no body), e.g. `"fn foo (x : u32) -> bool"`.
    pub signature: String,
}

/// A public trait in the API surface.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraitSig {
    /// Trait name.
    pub name: String,
    /// Stringified generics clause.
    pub generics: String,
    /// Stringified super-trait bounds.
    pub supertraits: String,
}

/// The collected public API surface of a single Rust source file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ApiSurface {
    /// Public types (structs, enums, type aliases).
    pub types: Vec<TypeSig>,
    /// Public free functions.
    pub fns: Vec<FnSig>,
    /// Public traits.
    pub traits: Vec<TraitSig>,
    /// Public modules (`pub mod`).
    pub modules: Vec<String>,
    /// Public re-exports (`pub use`).
    pub uses: Vec<String>,
    /// Public constants and statics.
    pub constants: Vec<String>,
}

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors that can arise during API surface operations.
#[derive(Debug, thiserror::Error)]
pub enum ApiSurfaceError {
    /// File I/O error.
    #[error("failed to read source file: {0}")]
    Io(#[from] std::io::Error),

    /// `syn` parse failure.
    #[error("failed to parse Rust source: {0}")]
    Parse(String),

    /// Baseline JSON file missing.
    #[error(
        "baseline file not found at {0} — run \
         `cargo run --bin api_snapshot --quiet > api_baseline.json` to generate it"
    )]
    BaselineNotFound(String),

    /// Baseline JSON malformed.
    #[error("failed to parse baseline JSON: {0}")]
    BaselineJson(String),
}

// ─── Parsing ─────────────────────────────────────────────────────────────────

/// Parse the public API surface from a Rust source file.
///
/// Only top-level items with `pub` (not `pub(crate)`, `pub(super)`, etc.)
/// visibility are included.  Items annotated with `#[doc(hidden)]` are
/// excluded.
pub fn parse_lib(path: &Path) -> Result<ApiSurface, ApiSurfaceError> {
    let src = std::fs::read_to_string(path)?;
    let file: File = syn::parse_str(&src).map_err(|e| ApiSurfaceError::Parse(e.to_string()))?;
    let mut surface = ApiSurface::default();
    collect_items(&file.items, &mut surface);
    Ok(surface)
}

/// Returns `true` if the visibility is exactly `pub` (not `pub(crate)` etc.).
fn is_public(vis: &Visibility) -> bool {
    matches!(vis, Visibility::Public(_))
}

/// Returns `true` if any attribute is `#[doc(hidden)]`.
fn is_doc_hidden(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        // Stringify the whole attribute token stream and look for "hidden".
        // This correctly handles `#[doc(hidden)]` regardless of spacing.
        a.path().is_ident("doc") && a.to_token_stream().to_string().contains("hidden")
    })
}

/// Collect public, non-hidden top-level items into `surface`.
fn collect_items(items: &[Item], surface: &mut ApiSurface) {
    for item in items {
        match item {
            Item::Fn(f) if is_public(&f.vis) && !is_doc_hidden(&f.attrs) => {
                let sig = &f.sig;
                surface.fns.push(FnSig {
                    name: f.sig.ident.to_string(),
                    signature: quote!(#sig).to_string(),
                });
            }
            Item::Struct(s) if is_public(&s.vis) && !is_doc_hidden(&s.attrs) => {
                let generics = &s.generics;
                surface.types.push(TypeSig {
                    name: s.ident.to_string(),
                    kind: "struct".into(),
                    generics: quote!(#generics).to_string(),
                });
            }
            Item::Enum(e) if is_public(&e.vis) && !is_doc_hidden(&e.attrs) => {
                let generics = &e.generics;
                surface.types.push(TypeSig {
                    name: e.ident.to_string(),
                    kind: "enum".into(),
                    generics: quote!(#generics).to_string(),
                });
            }
            Item::Trait(t) if is_public(&t.vis) && !is_doc_hidden(&t.attrs) => {
                let generics = &t.generics;
                let supertraits = &t.supertraits;
                surface.traits.push(TraitSig {
                    name: t.ident.to_string(),
                    generics: quote!(#generics).to_string(),
                    supertraits: quote!(#supertraits).to_string(),
                });
            }
            Item::Type(ty) if is_public(&ty.vis) && !is_doc_hidden(&ty.attrs) => {
                let generics = &ty.generics;
                surface.types.push(TypeSig {
                    name: ty.ident.to_string(),
                    kind: "type".into(),
                    generics: quote!(#generics).to_string(),
                });
            }
            Item::Mod(m) if is_public(&m.vis) && !is_doc_hidden(&m.attrs) => {
                surface.modules.push(m.ident.to_string());
            }
            Item::Use(u) if is_public(&u.vis) => {
                let tree = &u.tree;
                surface.uses.push(quote!(#tree).to_string());
            }
            Item::Const(c) if is_public(&c.vis) && !is_doc_hidden(&c.attrs) => {
                surface.constants.push(c.ident.to_string());
            }
            Item::Static(s) if is_public(&s.vis) && !is_doc_hidden(&s.attrs) => {
                surface.constants.push(s.ident.to_string());
            }
            _ => {}
        }
    }
}

// ─── Diffing ─────────────────────────────────────────────────────────────────

/// Compute the diff between a `baseline` surface and the `current` surface.
///
/// Breaking changes are: removed items, changed signatures, relocated types.
/// Non-breaking changes are: added items (allowed).
pub fn diff_surfaces(baseline: &ApiSurface, current: &ApiSurface) -> SurfaceDiff {
    let mut diff = SurfaceDiff::default();

    // Check for removed or changed types.
    for t in &baseline.types {
        match current.types.iter().find(|x| x.name == t.name) {
            None => diff.removed_types.push(t.name.clone()),
            Some(cur) if cur != t => diff.changed_types.push((t.clone(), cur.clone())),
            _ => {}
        }
    }

    // Check for removed or changed functions.
    for f in &baseline.fns {
        match current.fns.iter().find(|x| x.name == f.name) {
            None => diff.removed_fns.push(f.name.clone()),
            Some(cur) if cur != f => diff.changed_fns.push((f.clone(), cur.clone())),
            _ => {}
        }
    }

    // Check for removed traits.
    for t in &baseline.traits {
        if !current.traits.iter().any(|x| x.name == t.name) {
            diff.removed_traits.push(t.name.clone());
        }
    }

    // Check for removed modules.
    for m in &baseline.modules {
        if !current.modules.contains(m) {
            diff.removed_modules.push(m.clone());
        }
    }

    diff.is_breaking = !diff.removed_types.is_empty()
        || !diff.removed_fns.is_empty()
        || !diff.removed_traits.is_empty()
        || !diff.removed_modules.is_empty()
        || !diff.changed_types.is_empty()
        || !diff.changed_fns.is_empty();

    diff
}

/// The result of comparing two API surfaces.
#[derive(Debug, Default)]
pub struct SurfaceDiff {
    /// Names of types that were present in the baseline but are now missing.
    pub removed_types: Vec<String>,
    /// Types whose signature changed between baseline and current.
    pub changed_types: Vec<(TypeSig, TypeSig)>,
    /// Names of functions that were removed.
    pub removed_fns: Vec<String>,
    /// Functions whose signature changed.
    pub changed_fns: Vec<(FnSig, FnSig)>,
    /// Names of traits that were removed.
    pub removed_traits: Vec<String>,
    /// Names of modules that were removed.
    pub removed_modules: Vec<String>,
    /// `true` if any breaking change was detected.
    pub is_breaking: bool,
}

impl std::fmt::Display for SurfaceDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.is_breaking {
            return write!(f, "No breaking changes detected.");
        }

        writeln!(f, "BREAKING CHANGES DETECTED:")?;
        for name in &self.removed_types {
            writeln!(f, "  REMOVED type: {name}")?;
        }
        for name in &self.removed_fns {
            writeln!(f, "  REMOVED fn: {name}")?;
        }
        for name in &self.removed_traits {
            writeln!(f, "  REMOVED trait: {name}")?;
        }
        for name in &self.removed_modules {
            writeln!(f, "  REMOVED module: {name}")?;
        }
        for (old, new) in &self.changed_types {
            writeln!(f, "  CHANGED type {}: {:?} -> {:?}", old.name, old, new)?;
        }
        for (old, new) in &self.changed_fns {
            writeln!(f, "  CHANGED fn {}: {:?} -> {:?}", old.name, old, new)?;
        }
        writeln!(
            f,
            "\nTo update the baseline after intentional API changes, run:\
             \n  cargo run -p oxirs-core --bin api_snapshot --quiet \
             > core/oxirs-core/api_baseline.json"
        )
    }
}
