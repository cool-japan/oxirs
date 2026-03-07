//! J1939 protocol implementation modules
//!
//! This module provides advanced J1939 protocol support beyond the base
//! transport protocol already implemented in `protocol::j1939`.
//!
//! # Submodules
//!
//! - `diagnostics` -- SAE J1939/73 Diagnostic Messages (DM1, DM2, DM11, DM13)
//! - `transport_protocol` -- Enhanced SAE J1939-21 Transport Protocol with
//!   full RTS/CTS, BAM, EOM and Abort message support

pub mod diagnostics;
pub mod transport_protocol;

// Re-export key types for convenience
pub use diagnostics::{
    known_spn_description, DiagnosticEvent, DiagnosticTroubleCode, Dm11Request, Dm13Message,
    Dm1Message, Dm2Message, Dm3Request, HoldSignal, LampStatus,
};

pub use transport_protocol::{
    AbortReason, TpControlMessage, TpDataTransfer, TpReassembler, TP_CM_PGN, TP_DT_PGN,
};
