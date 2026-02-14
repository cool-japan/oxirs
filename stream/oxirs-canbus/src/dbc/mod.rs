//! DBC file parser and signal decoder
//!
//! Parses Vector CANdb++ DBC files to extract message and signal definitions,
//! and provides signal extraction and decoding from CAN frame data.
//!
//! # Features
//!
//! - **DBC Parsing** (`parser`): Parse DBC files for message/signal definitions
//! - **Signal Decoding** (`signal`): Extract and decode signals from CAN frames
//!
//! # Example
//!
//! ```no_run
//! use oxirs_canbus::dbc::{parse_dbc, SignalDecoder};
//!
//! let dbc_content = r#"
//! BO_ 2024 EngineData: 8 Engine
//!  SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8031.875] "rpm" Dashboard
//! "#;
//!
//! let db = parse_dbc(dbc_content).expect("DBC parsing should succeed");
//! let decoder = SignalDecoder::new(&db);
//!
//! // Decode a CAN frame
//! let frame_data = [0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
//! let values = decoder.decode_message(2024, &frame_data).unwrap();
//! println!("Engine Speed: {:?} rpm", values["EngineSpeed"]);
//! ```

pub mod parser;
pub mod signal;

// Re-export parser types
pub use parser::{
    // Functions
    parse_dbc,
    parse_dbc_file,
    // Core types
    AttributeDefinition,
    AttributeObjectType,
    AttributeValue,
    AttributeValueType,
    ByteOrder,
    DbcDatabase,
    DbcMessage,
    DbcNode,
    DbcParser,
    DbcSignal,
    MultiplexerType,
    ValueType,
};

// Re-export signal decoder
pub use signal::{
    DecodedSignalValue, SignalDecoder, SignalEncoder, SignalExtractionError, SignalValue,
};
