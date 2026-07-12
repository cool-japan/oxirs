//! Register auto-discovery for unknown Modbus devices.
//!
//! This module implements a three-stage pipeline:
//!
//! 1. **Driver** ([`driver`]) — walks FC 0x03 / FC 0x04 over a configurable
//!    address window, collecting raw 16-bit register values.  Uses the
//!    [`driver::ModbusAccess`] trait so it can be driven by a real TCP/RTU
//!    connection or an in-process mock.
//!
//! 2. **Inference** ([`inference`]) — applies a set of heuristics (Float32
//!    pairing, boolean, signed/unsigned 16-bit, scaled integer) to assign an
//!    [`inference::InferredType`] and [`inference::ConfidenceLevel`] to each
//!    discovered register.
//!
//! 3. **Emitter** ([`emitter`]) — converts inference results into a
//!    [`emitter::CandidateRegisterMap`] that can be serialised to JSON or a
//!    YAML-like human-readable format, and consumed by
//!    `crate::register_validator`.
//!
//! [`driver`]: crate::discovery::driver
//! [`driver::ModbusAccess`]: crate::discovery::driver::ModbusAccess
//! [`inference`]: crate::discovery::inference
//! [`inference::InferredType`]: crate::discovery::inference::InferredType
//! [`inference::ConfidenceLevel`]: crate::discovery::inference::ConfidenceLevel
//! [`emitter`]: crate::discovery::emitter
//! [`emitter::CandidateRegisterMap`]: crate::discovery::emitter::CandidateRegisterMap
//!
//! # Quick Start
//!
//! ```no_run
//! use oxirs_modbus::discovery::{DiscoveryDriver, DiscoveryConfig, driver::ModbusAccess, driver::DiscoveryError};
//! use oxirs_modbus::discovery::inference::TypeInferrer;
//! use oxirs_modbus::discovery::emitter::RegisterMapEmitter;
//!
//! struct AllZero;
//! impl ModbusAccess for AllZero {
//!     fn read_holding_registers(&mut self, _a: u16, c: u16) -> Result<Vec<u16>, DiscoveryError> { Ok(vec![0; c as usize]) }
//!     fn read_input_registers(&mut self, _a: u16, c: u16) -> Result<Vec<u16>, DiscoveryError> { Ok(vec![0; c as usize]) }
//!     fn read_coils(&mut self, _a: u16, c: u16) -> Result<Vec<bool>, DiscoveryError> { Ok(vec![false; c as usize]) }
//!     fn read_discrete_inputs(&mut self, _a: u16, c: u16) -> Result<Vec<bool>, DiscoveryError> { Ok(vec![false; c as usize]) }
//! }
//!
//! let cfg = DiscoveryConfig { address_start: 0, address_end: 4, read_batch_size: 4, ..Default::default() };
//! let mut driver = DiscoveryDriver::new(AllZero, cfg);
//! let discovered = driver.scan().expect("scan failed");
//! let holding: Vec<_> = discovered.into_iter()
//!     .filter(|r| matches!(r.function_code, oxirs_modbus::discovery::driver::DiscoveryFunctionCode::ReadHoldingRegisters))
//!     .collect();
//! let total = holding.len();
//! let inferred = TypeInferrer::infer_batch(&holding);
//! let map = RegisterMapEmitter::emit(1, &inferred, total);
//! let _json = RegisterMapEmitter::to_json(&map).expect("JSON serialization failed");
//! ```

/// Discovery driver and `ModbusAccess` trait.
pub mod driver;
/// Candidate register map emitter (JSON / YAML-like output).
pub mod emitter;
/// Type inference heuristics for discovered register values.
pub mod inference;

pub use driver::{DiscoveredRegister, DiscoveryConfig, DiscoveryDriver, DiscoveryError};
pub use emitter::RegisterMapEmitter;
pub use inference::{ConfidenceLevel, InferenceResult, InferredType, TypeInferrer};
