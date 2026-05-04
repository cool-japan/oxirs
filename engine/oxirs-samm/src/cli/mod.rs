//! Command-line interface support for the `samm` binary.
//!
//! This module provides the sub-command implementations used by the
//! `src/bin/samm.rs` binary.  Currently exposed sub-commands:
//!
//! | Sub-command | Module | Description |
//! |-------------|--------|-------------|
//! | `generate`  | `generate` | Invoke SAMM code generators from the CLI |

/// `samm generate` sub-command implementation.
pub mod generate;
