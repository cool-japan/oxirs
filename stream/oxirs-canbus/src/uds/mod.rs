//! UDS (Unified Diagnostic Services) – ISO 14229 implementation
//!
//! Provides a complete UDS implementation over ISO-TP (ISO 15765-2) framing,
//! covering all standard service IDs, negative response codes, and an async
//! [`UdsClient`] for communicating with ECUs over a CAN channel.
//!
//! # Protocol Stack
//!
//! ```text
//! UDS Service layer  (this module)
//!        │
//!        ▼
//! ISO 15765-2 (ISO-TP) transport  ← IsoTpCodec
//!        │
//!        ▼
//! CAN 2.0 frames  (oxirs_canbus CanFrame)
//! ```
//!
//! # Example
//!
//! ```rust
//! use oxirs_canbus::uds::{UdsRequest, UdsServiceId};
//!
//! let req = UdsRequest::new(UdsServiceId::ReadDataByIdentifier)
//!     .with_data(vec![0xF1, 0x90]);
//! assert_eq!(req.service_id, UdsServiceId::ReadDataByIdentifier);
//! ```

pub mod session_manager;
pub mod uds_codec;
pub mod uds_services;
mod uds_tests;
pub mod uds_types;

pub use uds_codec::*;
pub use uds_services::*;
pub use uds_types::*;
