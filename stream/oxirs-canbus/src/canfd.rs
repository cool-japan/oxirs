//! CAN FD (Flexible Data-rate) support for OxiRS
//!
//! CAN FD extends CAN 2.0 with:
//! - Extended payload: up to 64 bytes (vs 8 for CAN 2.0)
//! - Bit Rate Switching (BRS): data phase at higher bitrate
//! - Error State Indicator (ESI): transmitter error state
//!
//! # Standards
//! - ISO 11898-1:2015 (CAN FD)
//! - SAE J2284-4 (CAN FD for automotive)
//!
//! # Example
//!
//! ```rust
//! use oxirs_canbus::canfd::{CanFdFrame, CanFdFlags, CanFdDecoder, CanFdEncoder};
//!
//! // Create a CAN FD frame with 12-byte payload
//! let mut frame = CanFdFrame::new(0x123, vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C])
//!     .expect("valid CAN FD frame");
//! frame.flags.brs = true; // Enable bit rate switching
//!
//! // Encode to wire bytes
//! let encoded = CanFdEncoder::encode(&frame).expect("valid encoding");
//!
//! // Decode back
//! let decoded = CanFdDecoder::decode(&encoded).expect("valid decoding");
//! assert_eq!(decoded.can_id, frame.can_id);
//! ```

use crate::error::{CanbusError, CanbusResult};

// CAN FD valid Data Length Code values and their corresponding byte lengths
// Per ISO 11898-1:2015 Table 1
const CANFD_DLC_TO_LEN: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64];

/// Maximum payload size for CAN FD frames (ISO 11898-1:2015)
pub const CANFD_MAX_PAYLOAD: usize = 64;

/// Maximum payload size for CAN 2.0 frames
pub const CAN20_MAX_PAYLOAD: usize = 8;

/// CAN FD frame flags
///
/// These flags are carried in the frame control field of CAN FD frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CanFdFlags {
    /// Bit Rate Switching: if true, the data phase uses a higher bitrate
    pub brs: bool,
    /// Error State Indicator: reflects the transmitter's error state
    /// (false = error active, true = error passive)
    pub esi: bool,
    /// FD Frame marker: distinguishes CAN FD from CAN 2.0 (always true for CAN FD)
    pub fdf: bool,
}

impl CanFdFlags {
    /// Create flags for a standard CAN FD frame (BRS off, ESI off)
    pub fn standard() -> Self {
        Self {
            brs: false,
            esi: false,
            fdf: true,
        }
    }

    /// Create flags for a CAN FD frame with bit rate switching enabled
    pub fn with_brs() -> Self {
        Self {
            brs: true,
            esi: false,
            fdf: true,
        }
    }

    /// Encode flags to a single byte bitmask
    ///
    /// Bit layout: `[7:3]` reserved, `[2]` ESI, `[1]` BRS, `[0]` FDF
    pub fn to_byte(&self) -> u8 {
        let mut b = 0u8;
        if self.fdf {
            b |= 0x01;
        }
        if self.brs {
            b |= 0x02;
        }
        if self.esi {
            b |= 0x04;
        }
        b
    }

    /// Decode flags from a single byte bitmask
    pub fn from_byte(b: u8) -> Self {
        Self {
            fdf: (b & 0x01) != 0,
            brs: (b & 0x02) != 0,
            esi: (b & 0x04) != 0,
        }
    }
}

/// A CAN FD frame with extended payload support (up to 64 bytes)
///
/// CAN FD frames are compatible with CAN 2.0 at the identifier level
/// but use a different frame format in the data phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanFdFrame {
    /// CAN identifier (11-bit standard or 29-bit extended)
    pub can_id: u32,
    /// Whether this uses a 29-bit extended identifier
    pub is_extended: bool,
    /// CAN FD specific flags (BRS, ESI, FDF)
    pub flags: CanFdFlags,
    /// Data payload (0..=64 bytes)
    pub data: Vec<u8>,
}

impl CanFdFrame {
    /// Create a new CAN FD frame
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The CAN ID exceeds 29 bits (0x1FFFFFFF)
    /// - The data exceeds 64 bytes
    pub fn new(can_id: u32, data: Vec<u8>) -> CanbusResult<Self> {
        if can_id > 0x1FFF_FFFF {
            return Err(CanbusError::InvalidCanId(can_id));
        }
        if data.len() > CANFD_MAX_PAYLOAD {
            return Err(CanbusError::FrameTooLarge(data.len()));
        }

        let is_extended = can_id > 0x7FF;

        Ok(Self {
            can_id,
            is_extended,
            flags: CanFdFlags {
                fdf: true,
                brs: false,
                esi: false,
            },
            data,
        })
    }

    /// Create a CAN FD frame with bit rate switching enabled
    pub fn new_with_brs(can_id: u32, data: Vec<u8>) -> CanbusResult<Self> {
        let mut frame = Self::new(can_id, data)?;
        frame.flags.brs = true;
        Ok(frame)
    }

    /// Create a CAN FD frame with all flags specified
    pub fn new_with_flags(can_id: u32, data: Vec<u8>, flags: CanFdFlags) -> CanbusResult<Self> {
        if can_id > 0x1FFF_FFFF {
            return Err(CanbusError::InvalidCanId(can_id));
        }
        if data.len() > CANFD_MAX_PAYLOAD {
            return Err(CanbusError::FrameTooLarge(data.len()));
        }

        let is_extended = can_id > 0x7FF;

        Ok(Self {
            can_id,
            is_extended,
            flags,
            data,
        })
    }

    /// Get the Data Length Code (DLC) for this frame
    ///
    /// Returns the encoded DLC value (0..=15) corresponding to the
    /// payload length, per ISO 11898-1:2015 Table 1.
    pub fn dlc(&self) -> u8 {
        payload_len_to_dlc(self.data.len())
    }

    /// Get the actual payload length in bytes
    pub fn payload_len(&self) -> usize {
        self.data.len()
    }

    /// Check if this frame fits in a CAN 2.0 payload (8 bytes or less)
    pub fn is_can20_compatible(&self) -> bool {
        self.data.len() <= CAN20_MAX_PAYLOAD
    }

    /// Get the minimum CAN FD payload length that can hold `dlc` encoded bytes
    pub fn dlc_to_len(dlc: u8) -> usize {
        if dlc < 16 {
            CANFD_DLC_TO_LEN[dlc as usize] as usize
        } else {
            64
        }
    }
}

/// Convert a byte length to the CAN FD DLC encoding
///
/// Per ISO 11898-1:2015:
/// - 0..=8: DLC = length
/// - 9..=12: DLC = 9
/// - 13..=16: DLC = 10
/// - 17..=20: DLC = 11
/// - 21..=24: DLC = 12
/// - 25..=32: DLC = 13
/// - 33..=48: DLC = 14
/// - 49..=64: DLC = 15
pub fn payload_len_to_dlc(len: usize) -> u8 {
    match len {
        0..=8 => len as u8,
        9..=12 => 9,
        13..=16 => 10,
        17..=20 => 11,
        21..=24 => 12,
        25..=32 => 13,
        33..=48 => 14,
        _ => 15, // 49..=64
    }
}

/// Round up a payload length to the nearest valid CAN FD length
///
/// CAN FD supports: 0-8, 12, 16, 20, 24, 32, 48, 64 bytes
pub fn round_up_to_canfd_len(len: usize) -> usize {
    match len {
        0..=8 => len,
        9..=12 => 12,
        13..=16 => 16,
        17..=20 => 20,
        21..=24 => 24,
        25..=32 => 32,
        33..=48 => 48,
        _ => 64,
    }
}

/// Wire format for encoded CAN FD frames
///
/// Layout:
/// - Bytes 0..=3: CAN ID (LE) with flags in high bits
///   - Bit 31: extended ID flag
///   - Bits 30:0: CAN ID
/// - Byte 4: flags byte (FDF|BRS|ESI)
/// - Byte 5: DLC (0..=15)
/// - Bytes 6..: payload (variable, 0..=64 bytes)
pub const CANFD_WIRE_HEADER_SIZE: usize = 6;

/// Encoder for CAN FD frames to wire format
pub struct CanFdEncoder;

impl CanFdEncoder {
    /// Encode a CAN FD frame to a byte vector (wire format)
    ///
    /// # Layout
    /// ```text
    /// [can_id: u32 LE] [flags: u8] [dlc: u8] [data...]
    /// ```
    /// The high bit of can_id encodes the extended-frame flag.
    pub fn encode(frame: &CanFdFrame) -> CanbusResult<Vec<u8>> {
        let dlc = frame.dlc();
        let mut buf = Vec::with_capacity(CANFD_WIRE_HEADER_SIZE + frame.data.len());

        // Encode CAN ID with extended flag in bit 31
        let mut id_word = frame.can_id & 0x1FFF_FFFF;
        if frame.is_extended {
            id_word |= 0x8000_0000;
        }
        buf.extend_from_slice(&id_word.to_le_bytes());

        // Flags byte
        buf.push(frame.flags.to_byte());

        // DLC
        buf.push(dlc);

        // Payload
        buf.extend_from_slice(&frame.data);

        Ok(buf)
    }
}

/// Decoder for CAN FD frames from wire format
pub struct CanFdDecoder;

impl CanFdDecoder {
    /// Decode a CAN FD frame from its wire representation
    ///
    /// # Errors
    ///
    /// Returns an error if the byte slice is too short or contains
    /// an invalid DLC/payload combination.
    pub fn decode(data: &[u8]) -> CanbusResult<CanFdFrame> {
        if data.len() < CANFD_WIRE_HEADER_SIZE {
            return Err(CanbusError::Config(format!(
                "CAN FD wire data too short: {} bytes (minimum {})",
                data.len(),
                CANFD_WIRE_HEADER_SIZE
            )));
        }

        // Parse CAN ID and extended flag
        let id_word = u32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| CanbusError::Config("Failed to parse CAN FD ID word".to_string()))?,
        );

        let is_extended = (id_word & 0x8000_0000) != 0;
        let can_id = id_word & 0x1FFF_FFFF;

        // Parse flags
        let flags = CanFdFlags::from_byte(data[4]);

        // Parse DLC
        let dlc = data[5];
        if dlc > 15 {
            return Err(CanbusError::Config(format!(
                "Invalid CAN FD DLC: {} (max 15)",
                dlc
            )));
        }

        let expected_payload_len = CanFdFrame::dlc_to_len(dlc);

        // Validate total length
        if data.len() < CANFD_WIRE_HEADER_SIZE + expected_payload_len {
            return Err(CanbusError::Config(format!(
                "CAN FD payload truncated: DLC {} requires {} bytes, got {}",
                dlc,
                expected_payload_len,
                data.len() - CANFD_WIRE_HEADER_SIZE
            )));
        }

        let payload =
            data[CANFD_WIRE_HEADER_SIZE..CANFD_WIRE_HEADER_SIZE + expected_payload_len].to_vec();

        Ok(CanFdFrame {
            can_id,
            is_extended,
            flags,
            data: payload,
        })
    }
}

/// Statistics for CAN FD bus monitoring
#[derive(Debug, Clone, Default)]
pub struct CanFdStats {
    /// Total frames received
    pub frames_received: u64,
    /// Frames with BRS (bit rate switching) enabled
    pub brs_frames: u64,
    /// Frames with ESI (error state indicator) set
    pub esi_frames: u64,
    /// Frames with payload > 8 bytes (utilizing FD advantage)
    pub extended_payload_frames: u64,
    /// CAN 2.0 compatible frames (payload <= 8 bytes)
    pub can20_compatible_frames: u64,
    /// Total payload bytes received
    pub total_payload_bytes: u64,
}

impl CanFdStats {
    /// Create a new stats counter
    pub fn new() -> Self {
        Self::default()
    }

    /// Update stats from a received frame
    pub fn record_frame(&mut self, frame: &CanFdFrame) {
        self.frames_received += 1;
        if frame.flags.brs {
            self.brs_frames += 1;
        }
        if frame.flags.esi {
            self.esi_frames += 1;
        }
        if frame.data.len() > CAN20_MAX_PAYLOAD {
            self.extended_payload_frames += 1;
        } else {
            self.can20_compatible_frames += 1;
        }
        self.total_payload_bytes += frame.data.len() as u64;
    }

    /// Average payload size
    pub fn average_payload_bytes(&self) -> f64 {
        if self.frames_received == 0 {
            0.0
        } else {
            self.total_payload_bytes as f64 / self.frames_received as f64
        }
    }

    /// BRS frame ratio (0.0 to 1.0)
    pub fn brs_ratio(&self) -> f64 {
        if self.frames_received == 0 {
            0.0
        } else {
            self.brs_frames as f64 / self.frames_received as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- CanFdFlags tests ----

    #[test]
    fn test_canfd_flags_default() {
        let flags = CanFdFlags::default();
        assert!(!flags.brs);
        assert!(!flags.esi);
        assert!(!flags.fdf);
    }

    #[test]
    fn test_canfd_flags_standard() {
        let flags = CanFdFlags::standard();
        assert!(!flags.brs);
        assert!(!flags.esi);
        assert!(flags.fdf);
    }

    #[test]
    fn test_canfd_flags_with_brs() {
        let flags = CanFdFlags::with_brs();
        assert!(flags.brs);
        assert!(!flags.esi);
        assert!(flags.fdf);
    }

    #[test]
    fn test_canfd_flags_to_byte_fdf_only() {
        let flags = CanFdFlags {
            fdf: true,
            brs: false,
            esi: false,
        };
        assert_eq!(flags.to_byte(), 0x01);
    }

    #[test]
    fn test_canfd_flags_to_byte_brs_fdf() {
        let flags = CanFdFlags {
            fdf: true,
            brs: true,
            esi: false,
        };
        assert_eq!(flags.to_byte(), 0x03);
    }

    #[test]
    fn test_canfd_flags_to_byte_all_set() {
        let flags = CanFdFlags {
            fdf: true,
            brs: true,
            esi: true,
        };
        assert_eq!(flags.to_byte(), 0x07);
    }

    #[test]
    fn test_canfd_flags_roundtrip() {
        for byte in 0u8..8 {
            let flags = CanFdFlags::from_byte(byte);
            assert_eq!(flags.to_byte(), byte);
        }
    }

    // ---- CanFdFrame creation tests ----

    #[test]
    fn test_canfd_frame_new_standard_id() {
        let frame = CanFdFrame::new(0x123, vec![0x01, 0x02, 0x03]).expect("valid frame");
        assert_eq!(frame.can_id, 0x123);
        assert!(!frame.is_extended);
        assert_eq!(frame.data.len(), 3);
    }

    #[test]
    fn test_canfd_frame_new_extended_id() {
        let frame = CanFdFrame::new(0x1FFFFFFF, vec![0xAA, 0xBB]).expect("valid frame");
        assert_eq!(frame.can_id, 0x1FFFFFFF);
        assert!(frame.is_extended);
    }

    #[test]
    fn test_canfd_frame_invalid_id() {
        let result = CanFdFrame::new(0x2000_0000, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_canfd_frame_too_large() {
        let data = vec![0u8; 65]; // 65 bytes > 64 byte limit
        let result = CanFdFrame::new(0x100, data);
        assert!(result.is_err());
    }

    #[test]
    fn test_canfd_frame_max_payload() {
        let data: Vec<u8> = (0..64).collect();
        let frame = CanFdFrame::new(0x100, data.clone()).expect("valid frame");
        assert_eq!(frame.payload_len(), 64);
        assert_eq!(frame.dlc(), 15);
        assert!(!frame.is_can20_compatible());
    }

    #[test]
    fn test_canfd_frame_empty_payload() {
        let frame = CanFdFrame::new(0x100, vec![]).expect("valid frame");
        assert_eq!(frame.payload_len(), 0);
        assert_eq!(frame.dlc(), 0);
        assert!(frame.is_can20_compatible());
    }

    #[test]
    fn test_canfd_frame_new_with_brs() {
        let frame = CanFdFrame::new_with_brs(0x100, vec![0x01]).expect("valid frame");
        assert!(frame.flags.brs);
        assert!(frame.flags.fdf);
        assert!(!frame.flags.esi);
    }

    #[test]
    fn test_canfd_frame_can20_compatible() {
        let frame8 = CanFdFrame::new(0x100, vec![0u8; 8]).expect("valid frame");
        assert!(frame8.is_can20_compatible());

        let frame9 = CanFdFrame::new(0x100, vec![0u8; 9]).expect("valid frame");
        assert!(!frame9.is_can20_compatible());
    }

    // ---- DLC encoding tests ----

    #[test]
    fn test_payload_len_to_dlc_basic() {
        assert_eq!(payload_len_to_dlc(0), 0);
        assert_eq!(payload_len_to_dlc(1), 1);
        assert_eq!(payload_len_to_dlc(8), 8);
    }

    #[test]
    fn test_payload_len_to_dlc_extended() {
        assert_eq!(payload_len_to_dlc(9), 9);
        assert_eq!(payload_len_to_dlc(12), 9);
        assert_eq!(payload_len_to_dlc(13), 10);
        assert_eq!(payload_len_to_dlc(16), 10);
        assert_eq!(payload_len_to_dlc(32), 13);
        assert_eq!(payload_len_to_dlc(48), 14);
        assert_eq!(payload_len_to_dlc(64), 15);
    }

    #[test]
    fn test_canfd_dlc_to_len() {
        assert_eq!(CanFdFrame::dlc_to_len(0), 0);
        assert_eq!(CanFdFrame::dlc_to_len(8), 8);
        assert_eq!(CanFdFrame::dlc_to_len(9), 12);
        assert_eq!(CanFdFrame::dlc_to_len(10), 16);
        assert_eq!(CanFdFrame::dlc_to_len(13), 32);
        assert_eq!(CanFdFrame::dlc_to_len(14), 48);
        assert_eq!(CanFdFrame::dlc_to_len(15), 64);
    }

    #[test]
    fn test_round_up_to_canfd_len() {
        assert_eq!(round_up_to_canfd_len(0), 0);
        assert_eq!(round_up_to_canfd_len(8), 8);
        assert_eq!(round_up_to_canfd_len(9), 12);
        assert_eq!(round_up_to_canfd_len(12), 12);
        assert_eq!(round_up_to_canfd_len(13), 16);
        assert_eq!(round_up_to_canfd_len(32), 32);
        assert_eq!(round_up_to_canfd_len(33), 48);
        assert_eq!(round_up_to_canfd_len(64), 64);
        assert_eq!(round_up_to_canfd_len(65), 64);
    }

    // ---- Encoder/Decoder tests ----

    #[test]
    fn test_canfd_encode_decode_roundtrip_standard_id() {
        let frame = CanFdFrame::new(0x123, vec![0xDE, 0xAD, 0xBE, 0xEF]).expect("valid frame");
        let encoded = CanFdEncoder::encode(&frame).expect("valid encoding");
        let decoded = CanFdDecoder::decode(&encoded).expect("valid decoding");

        assert_eq!(decoded.can_id, frame.can_id);
        assert_eq!(decoded.is_extended, frame.is_extended);
        assert_eq!(decoded.data, frame.data);
    }

    #[test]
    fn test_canfd_encode_decode_roundtrip_extended_id() {
        let frame = CanFdFrame::new(0x0CF00400, vec![0x01, 0x02, 0x03, 0x04]).expect("valid frame");
        let encoded = CanFdEncoder::encode(&frame).expect("valid encoding");
        let decoded = CanFdDecoder::decode(&encoded).expect("valid decoding");

        assert_eq!(decoded.can_id, 0x0CF00400);
        assert!(decoded.is_extended);
        assert_eq!(decoded.data, vec![0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_canfd_encode_decode_with_brs_and_esi() {
        let flags = CanFdFlags {
            fdf: true,
            brs: true,
            esi: true,
        };
        let frame =
            CanFdFrame::new_with_flags(0x100, vec![0xAA, 0xBB], flags).expect("valid frame");
        let encoded = CanFdEncoder::encode(&frame).expect("valid encoding");
        let decoded = CanFdDecoder::decode(&encoded).expect("valid decoding");

        assert!(decoded.flags.brs);
        assert!(decoded.flags.esi);
        assert!(decoded.flags.fdf);
    }

    #[test]
    fn test_canfd_encode_decode_max_payload() {
        let data: Vec<u8> = (0..64).collect();
        let frame = CanFdFrame::new(0x1FFFFFFF, data.clone()).expect("valid frame");
        let encoded = CanFdEncoder::encode(&frame).expect("valid encoding");
        let decoded = CanFdDecoder::decode(&encoded).expect("valid decoding");

        assert_eq!(decoded.data, data);
        assert_eq!(decoded.can_id, 0x1FFFFFFF);
    }

    #[test]
    fn test_canfd_encode_decode_12_byte_payload() {
        // 12 bytes is a valid CAN FD length
        let data = vec![0x11u8; 12];
        let frame = CanFdFrame::new(0x200, data.clone()).expect("valid frame");
        let encoded = CanFdEncoder::encode(&frame).expect("valid encoding");
        let decoded = CanFdDecoder::decode(&encoded).expect("valid decoding");

        assert_eq!(decoded.data, data);
        assert_eq!(decoded.dlc(), 9); // DLC=9 maps to 12 bytes
    }

    #[test]
    fn test_canfd_decode_truncated_data() {
        let result = CanFdDecoder::decode(&[0x00, 0x01, 0x00, 0x00]);
        assert!(result.is_err());
    }

    #[test]
    fn test_canfd_encode_wire_header_size() {
        let frame = CanFdFrame::new(0x100, vec![0x01, 0x02]).expect("valid frame");
        let encoded = CanFdEncoder::encode(&frame).expect("valid encoding");
        assert_eq!(encoded.len(), CANFD_WIRE_HEADER_SIZE + 2);
    }

    // ---- CanFdStats tests ----

    #[test]
    fn test_canfd_stats_initial() {
        let stats = CanFdStats::new();
        assert_eq!(stats.frames_received, 0);
        assert_eq!(stats.brs_frames, 0);
        assert_eq!(stats.average_payload_bytes(), 0.0);
    }

    #[test]
    fn test_canfd_stats_record_frames() {
        let mut stats = CanFdStats::new();

        let frame1 = CanFdFrame::new(0x100, vec![0x01, 0x02, 0x03]).expect("valid frame");
        let frame2 = CanFdFrame::new_with_brs(0x200, vec![0u8; 32]).expect("valid frame");
        let mut frame3 = CanFdFrame::new(0x300, vec![0u8; 64]).expect("valid frame");
        frame3.flags.esi = true;

        stats.record_frame(&frame1);
        stats.record_frame(&frame2);
        stats.record_frame(&frame3);

        assert_eq!(stats.frames_received, 3);
        assert_eq!(stats.brs_frames, 1);
        assert_eq!(stats.esi_frames, 1);
        assert_eq!(stats.extended_payload_frames, 2); // 32 and 64 byte frames
        assert_eq!(stats.can20_compatible_frames, 1); // 3 byte frame
        assert_eq!(stats.total_payload_bytes, 3 + 32 + 64);
    }

    #[test]
    fn test_canfd_stats_brs_ratio() {
        let mut stats = CanFdStats::new();

        let brs_frame = CanFdFrame::new_with_brs(0x100, vec![0x01]).expect("valid frame");
        let normal_frame = CanFdFrame::new(0x200, vec![0x01]).expect("valid frame");

        stats.record_frame(&brs_frame);
        stats.record_frame(&normal_frame);

        assert!((stats.brs_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_canfd_stats_average_payload() {
        let mut stats = CanFdStats::new();

        let frame8 = CanFdFrame::new(0x100, vec![0u8; 8]).expect("valid frame");
        let frame16 = CanFdFrame::new(0x200, vec![0u8; 16]).expect("valid frame");

        stats.record_frame(&frame8);
        stats.record_frame(&frame16);

        assert!((stats.average_payload_bytes() - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_canfd_frame_dlc_values_roundtrip() {
        // Test all valid CAN FD lengths
        let valid_lens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64];
        for &len in &valid_lens {
            let data = vec![0xABu8; len];
            let frame = CanFdFrame::new(0x100, data).expect("valid frame");
            let dlc = frame.dlc();
            let decoded_len = CanFdFrame::dlc_to_len(dlc);
            assert_eq!(
                decoded_len, len,
                "length {} roundtrip failed (DLC={})",
                len, dlc
            );
        }
    }
}
