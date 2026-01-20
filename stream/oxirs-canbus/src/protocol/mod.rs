//! CANbus protocol handling and frame parsing
//!
//! This module provides:
//! - CAN frame types and parsing (`frame`)
//! - SocketCAN client for Linux (`socketcan_client`)
//! - J1939 protocol implementation (`j1939`)
//! - Common J1939 PGN decoders (`j1939_pgns`)

pub mod frame;
pub mod j1939;
pub mod j1939_pgns;

#[cfg(target_os = "linux")]
pub mod socketcan_client;

// Frame types
pub use frame::{CanFrame, CanId};

// J1939 protocol
pub use j1939::{
    AddressManager, DeviceInfo, J1939Header, J1939Message, J1939Processor, Pgn, Priority,
    TransportProtocol,
};

// J1939 PGNs
pub use j1939_pgns::{
    // Decoders
    AmbDecoder,
    CcvsDecoder,
    DecodedPgn,
    DecodedSignal,
    Eec1Decoder,
    Eec2Decoder,
    Eflp1Decoder,
    Et1Decoder,
    LfeDecoder,
    PgnDecoder,
    PgnRegistry,
    PgnValue,
    Vep1Decoder,
    // PGN constants
    PGN_AMB,
    PGN_CCVS,
    PGN_CI,
    PGN_DD,
    PGN_EBC1,
    PGN_EEC1,
    PGN_EEC2,
    PGN_EFLP1,
    PGN_ET1,
    PGN_ETC1,
    PGN_ETC2,
    PGN_HRWS,
    PGN_LFC,
    PGN_LFE,
    PGN_SOFT,
    PGN_VEP1,
    PGN_VW,
};

// SocketCAN (Linux only)
#[cfg(target_os = "linux")]
pub use socketcan_client::{CanFdClient, CanStatistics, CanbusClient};
