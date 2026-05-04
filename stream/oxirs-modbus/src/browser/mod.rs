//! Terminal UI register browser for Modbus devices.
//!
//! This module provides an interactive TUI (using [`ratatui`]) that lets you
//! browse all four Modbus register spaces in real time:
//!
//! - **Coils** (FC01/FC05/FC15) — 1-bit read-write
//! - **Discrete Inputs** (FC02) — 1-bit read-only
//! - **Holding Registers** (FC03/FC06/FC16) — 16-bit read-write
//! - **Input Registers** (FC04) — 16-bit read-only
//!
//! Activate with `--features tui` in Cargo.
//!
//! # Quick start
//!
//! ```no_run
//! # #[cfg(feature = "tui")]
//! # {
//! use oxirs_modbus::browser::{RegisterBrowser, BrowserConfig, BrowserError};
//! use oxirs_modbus::browser::state::{RegisterTab, RegisterValue};
//!
//! let mut browser = RegisterBrowser::new(BrowserConfig::default());
//! browser.load_snapshot(
//!     RegisterTab::HoldingRegisters,
//!     vec![
//!         (40, RegisterValue::U16(1024), Some("Motor speed".to_string())),
//!         (41, RegisterValue::F32(23.75), Some("Temperature".to_string())),
//!     ],
//! );
//! browser.run()?;
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod app;
pub mod state;
pub mod ui;
pub mod widgets;

pub use app::{BrowserConfig, BrowserError, RegisterBrowser};
pub use state::{BrowserState, FilteredView, RegisterTab};
