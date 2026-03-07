//! CAN recording file formats
//!
//! Supports industry-standard recording formats for offline analysis:
//!
//! - **ASC**: Vector CANalyzer text format (human-readable, widely supported)
//! - **BLF**: Binary Logging File (compact binary, Vector proprietary, simplified implementation)
//!
//! # ASC Format Example
//!
//! ```text
//! date Mon Feb 26 10:00:00 2026
//! base hex  timestamps absolute
//! no internal events logged
//!    0.000125 1  0CF00400  Rx   d 8 00 00 00 00 00 00 00 00
//!    0.001000 1  18FEF100  Tx   d 4 DE AD BE EF
//! ```
//!
//! # BLF Format
//!
//! Simplified binary format with magic header and fixed-size objects:
//!
//! ```text
//! [Magic: "BLF0200\x00"][file_size: u64][object_count: u32][start_ts_ns: u64]
//! [Object: timestamp_ns: u64][channel: u16][frame_id: u32][dlc: u8][data: [u8; 64]]
//! ```

use crate::error::{CanbusError, CanbusResult};

// ============================================================================
// ASC Format
// ============================================================================

/// CAN frame transmission direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Frame received from the bus
    Rx,
    /// Frame transmitted to the bus
    Tx,
}

impl Direction {
    /// Convert to ASC format string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rx => "Rx",
            Self::Tx => "Tx",
        }
    }

    /// Parse from ASC format direction abbreviation ("Rx" or "Tx")
    pub fn parse_asc(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rx" => Some(Self::Rx),
            "tx" => Some(Self::Tx),
            _ => None,
        }
    }
}

/// A single CAN frame record in ASC format
///
/// Corresponds to a line in a Vector CANalyzer .asc file:
/// `<timestamp> <channel> <frame_id> <direction> d <dlc> <data bytes...>`
#[derive(Debug, Clone, PartialEq)]
pub struct AscRecord {
    /// Absolute timestamp in seconds since log start
    pub timestamp: f64,
    /// CAN channel number (1-based)
    pub channel: u8,
    /// CAN frame ID (11-bit standard or 29-bit extended)
    pub frame_id: u32,
    /// Data payload bytes
    pub data: Vec<u8>,
    /// Frame direction (Rx/Tx)
    pub direction: Direction,
    /// Data Length Code
    pub dlc: u8,
}

impl AscRecord {
    /// Create a new ASC record
    pub fn new(
        timestamp: f64,
        channel: u8,
        frame_id: u32,
        data: Vec<u8>,
        direction: Direction,
    ) -> Self {
        let dlc = data.len() as u8;
        Self {
            timestamp,
            channel,
            frame_id,
            data,
            direction,
            dlc,
        }
    }

    /// Check if this is an extended (29-bit) frame ID
    ///
    /// Heuristic: if the raw ID is > 0x7FF, it's extended
    pub fn is_extended_id(&self) -> bool {
        self.frame_id > 0x7FF
    }
}

/// Parser for Vector CANalyzer ASC files
///
/// Parses the line-based text format used by CANalyzer, CANdb++, and compatible tools.
pub struct AscParser;

impl AscParser {
    /// Parse ASC text content into a list of CAN frame records
    ///
    /// Skips comment/header lines (lines starting with keywords like "date", "base", "no ", etc.)
    /// Parses data lines matching the pattern:
    /// `<timestamp> <channel> <frame_id> <direction> d <dlc> <data...>`
    pub fn parse(text: &str) -> CanbusResult<Vec<AscRecord>> {
        let mut records = Vec::new();

        for (line_num, line) in text.lines().enumerate() {
            let trimmed = line.trim();

            // Skip empty lines and header/comment lines
            if trimmed.is_empty()
                || trimmed.starts_with("date")
                || trimmed.starts_with("base")
                || trimmed.starts_with("no ")
                || trimmed.starts_with("//")
                || trimmed.starts_with(';')
                || trimmed.starts_with("Begin")
                || trimmed.starts_with("End")
                || trimmed.starts_with("Statistics")
                || trimmed.starts_with("internal")
            {
                continue;
            }

            // Try to parse a data line
            if let Some(record) = Self::parse_data_line(trimmed) {
                records.push(record);
            } else {
                // Silently skip unrecognized lines (ASC files can have various formats)
                let _ = line_num; // suppress unused warning
            }
        }

        Ok(records)
    }

    /// Parse a single data line
    ///
    /// Expected format: `<timestamp> <channel> <id> <dir> d <dlc> [data bytes...]`
    fn parse_data_line(line: &str) -> Option<AscRecord> {
        let tokens: Vec<&str> = line.split_whitespace().collect();

        // Minimum tokens: timestamp + channel + id + direction + "d" + dlc = 6
        if tokens.len() < 6 {
            return None;
        }

        let timestamp: f64 = tokens[0].parse().ok()?;
        let channel: u8 = tokens[1].parse().ok()?;

        // Frame ID may be hex (no 0x prefix in ASC format)
        let frame_id = u32::from_str_radix(tokens[2], 16).ok()?;

        let direction = Direction::parse_asc(tokens[3])?;

        // tokens[4] should be "d" (data frame marker)
        if tokens[4].to_lowercase() != "d" {
            return None;
        }

        let dlc: u8 = tokens[5].parse().ok()?;

        // Parse data bytes
        let mut data: Vec<u8> = Vec::with_capacity(dlc as usize);
        for i in 0..(dlc as usize) {
            let byte_idx = 6 + i;
            if byte_idx < tokens.len() {
                let byte = u8::from_str_radix(tokens[byte_idx], 16).ok()?;
                data.push(byte);
            }
        }

        Some(AscRecord {
            timestamp,
            channel,
            frame_id,
            data,
            direction,
            dlc,
        })
    }
}

/// Writer for Vector CANalyzer ASC format
pub struct AscWriter;

impl AscWriter {
    /// Generate ASC format text from a list of CAN records
    pub fn write(records: &[AscRecord]) -> String {
        Self::write_with_header(records, "Mon Jan 01 00:00:00 2026")
    }

    /// Generate ASC format text with a custom header date string
    pub fn write_with_header(records: &[AscRecord], date_str: &str) -> String {
        let mut output = String::new();

        // ASC file header
        output.push_str(&format!("date {}\n", date_str));
        output.push_str("base hex  timestamps absolute\n");
        output.push_str("no internal events logged\n");

        // Data lines
        for record in records {
            let data_hex: Vec<String> = record.data.iter().map(|b| format!("{:02X}", b)).collect();
            let data_str = data_hex.join(" ");

            output.push_str(&format!(
                "   {:.6} {}  {:08X}  {}   d {} {}\n",
                record.timestamp,
                record.channel,
                record.frame_id,
                record.direction.as_str(),
                record.dlc,
                data_str
            ));
        }

        output
    }
}

// ============================================================================
// BLF Format (simplified)
// ============================================================================

/// Magic bytes for BLF file header
pub const BLF_MAGIC: &[u8; 8] = b"BLF0200\x00";

/// BLF header size in bytes
pub const BLF_HEADER_SIZE: usize = 28; // 8 (magic) + 8 (file_size) + 4 (count) + 8 (start_ts)

/// BLF object record size in bytes
pub const BLF_OBJECT_SIZE: usize = 79; // 8 (ts_ns) + 2 (channel) + 4 (frame_id) + 1 (dlc) + 64 (data)

/// A single CAN frame record in BLF (binary) format
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlfRecord {
    /// Timestamp in nanoseconds since epoch or log start
    pub timestamp_ns: u64,
    /// CAN channel number (1-based)
    pub channel: u16,
    /// CAN frame ID
    pub frame_id: u32,
    /// Data Length Code (0..=8 for CAN 2.0, 0..=64 for CAN FD)
    pub dlc: u8,
    /// Data payload (up to 64 bytes for CAN FD)
    pub data: Vec<u8>,
}

impl BlfRecord {
    /// Create a new BLF record
    pub fn new(timestamp_ns: u64, channel: u16, frame_id: u32, data: Vec<u8>) -> Self {
        let dlc = data.len().min(64) as u8;
        Self {
            timestamp_ns,
            channel,
            frame_id,
            dlc,
            data,
        }
    }

    /// Convert this BLF record to an AscRecord for cross-format conversion
    pub fn to_asc_record(&self) -> AscRecord {
        let timestamp_secs = self.timestamp_ns as f64 / 1_000_000_000.0;
        AscRecord {
            timestamp: timestamp_secs,
            channel: self.channel as u8,
            frame_id: self.frame_id,
            data: self.data.clone(),
            direction: Direction::Rx,
            dlc: self.dlc,
        }
    }
}

/// Writer for simplified BLF binary format
///
/// Binary layout:
/// - Header (28 bytes): magic(8) + file_size(8 LE) + object_count(4 LE) + start_ts_ns(8 LE)
/// - Objects (79 bytes each): timestamp_ns(8) + channel(2) + frame_id(4) + dlc(1) + data(64)
pub struct BlfWriter;

impl BlfWriter {
    /// Encode a list of BLF records to binary bytes
    pub fn write(records: &[BlfRecord]) -> Vec<u8> {
        let object_count = records.len() as u32;
        let file_size = (BLF_HEADER_SIZE + records.len() * BLF_OBJECT_SIZE) as u64;
        let start_ts_ns: u64 = records.first().map(|r| r.timestamp_ns).unwrap_or(0);

        let mut buf = Vec::with_capacity(file_size as usize);

        // Write header
        buf.extend_from_slice(BLF_MAGIC);
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(&object_count.to_le_bytes());
        buf.extend_from_slice(&start_ts_ns.to_le_bytes());

        // Write objects
        for record in records {
            buf.extend_from_slice(&record.timestamp_ns.to_le_bytes());
            buf.extend_from_slice(&record.channel.to_le_bytes());
            buf.extend_from_slice(&record.frame_id.to_le_bytes());
            buf.push(record.dlc);

            // Data field is fixed 64 bytes, zero-padded
            let mut data_field = [0u8; 64];
            let copy_len = record.data.len().min(64);
            data_field[..copy_len].copy_from_slice(&record.data[..copy_len]);
            buf.extend_from_slice(&data_field);
        }

        buf
    }
}

/// Parser for simplified BLF binary format
pub struct BlfParser;

impl BlfParser {
    /// Parse binary BLF data into a list of CAN frame records
    pub fn parse(data: &[u8]) -> CanbusResult<Vec<BlfRecord>> {
        // Validate header
        if data.len() < BLF_HEADER_SIZE {
            return Err(CanbusError::Config(format!(
                "BLF data too short: {} bytes (need at least {})",
                data.len(),
                BLF_HEADER_SIZE
            )));
        }

        // Verify magic
        if &data[..8] != BLF_MAGIC {
            return Err(CanbusError::Config("Invalid BLF magic bytes".to_string()));
        }

        // Parse header fields
        let file_size = u64::from_le_bytes(
            data[8..16]
                .try_into()
                .map_err(|_| CanbusError::Config("Failed to read BLF file_size".to_string()))?,
        );

        let object_count = u32::from_le_bytes(
            data[16..20]
                .try_into()
                .map_err(|_| CanbusError::Config("Failed to read BLF object_count".to_string()))?,
        ) as usize;

        // Validate file size
        let expected_size = BLF_HEADER_SIZE + object_count * BLF_OBJECT_SIZE;
        if data.len() < expected_size {
            return Err(CanbusError::Config(format!(
                "BLF file truncated: expected {} bytes for {} objects, got {}",
                expected_size,
                object_count,
                data.len()
            )));
        }

        // Ignore start_ts_ns (bytes 20..28) — used in header for metadata only
        let _ = file_size;

        // Parse objects
        let mut records = Vec::with_capacity(object_count);
        for i in 0..object_count {
            let offset = BLF_HEADER_SIZE + i * BLF_OBJECT_SIZE;
            let obj = &data[offset..offset + BLF_OBJECT_SIZE];

            let timestamp_ns = u64::from_le_bytes(obj[0..8].try_into().map_err(|_| {
                CanbusError::Config(format!("Failed to parse BLF object {} timestamp", i))
            })?);

            let channel = u16::from_le_bytes(obj[8..10].try_into().map_err(|_| {
                CanbusError::Config(format!("Failed to parse BLF object {} channel", i))
            })?);

            let frame_id = u32::from_le_bytes(obj[10..14].try_into().map_err(|_| {
                CanbusError::Config(format!("Failed to parse BLF object {} frame_id", i))
            })?);

            let dlc = obj[14];
            let data_slice = &obj[15..79];
            let actual_len = (dlc as usize).min(64);
            let data = data_slice[..actual_len].to_vec();

            records.push(BlfRecord {
                timestamp_ns,
                channel,
                frame_id,
                dlc,
                data,
            });
        }

        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Direction tests ----

    #[test]
    fn test_direction_as_str() {
        assert_eq!(Direction::Rx.as_str(), "Rx");
        assert_eq!(Direction::Tx.as_str(), "Tx");
    }

    #[test]
    fn test_direction_from_str() {
        assert_eq!(Direction::parse_asc("Rx"), Some(Direction::Rx));
        assert_eq!(Direction::parse_asc("rx"), Some(Direction::Rx));
        assert_eq!(Direction::parse_asc("TX"), Some(Direction::Tx));
        assert_eq!(Direction::parse_asc("Tx"), Some(Direction::Tx));
        assert_eq!(Direction::parse_asc("unknown"), None);
    }

    // ---- AscRecord tests ----

    #[test]
    fn test_asc_record_new() {
        let rec = AscRecord::new(1.5, 1, 0x0CF00400, vec![0x01, 0x02], Direction::Rx);
        assert_eq!(rec.timestamp, 1.5);
        assert_eq!(rec.channel, 1);
        assert_eq!(rec.frame_id, 0x0CF00400);
        assert_eq!(rec.data, vec![0x01, 0x02]);
        assert_eq!(rec.dlc, 2);
        assert_eq!(rec.direction, Direction::Rx);
    }

    #[test]
    fn test_asc_record_is_extended_id() {
        let rec = AscRecord::new(0.0, 1, 0x0CF00400, vec![], Direction::Rx);
        assert!(rec.is_extended_id());

        let rec = AscRecord::new(0.0, 1, 0x123, vec![], Direction::Rx);
        assert!(!rec.is_extended_id());
    }

    // ---- AscParser tests ----

    #[test]
    fn test_asc_parser_basic() {
        let asc = "date Mon Feb 26 10:00:00 2026\n\
                   base hex  timestamps absolute\n\
                   no internal events logged\n\
                   0.000125 1 0CF00400 Rx d 8 00 00 00 00 00 00 00 00\n";

        let records = AscParser::parse(asc).expect("parse should succeed");
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!((r.timestamp - 0.000125).abs() < 1e-9);
        assert_eq!(r.channel, 1);
        assert_eq!(r.frame_id, 0x0CF00400);
        assert_eq!(r.direction, Direction::Rx);
        assert_eq!(r.dlc, 8);
        assert_eq!(r.data, vec![0u8; 8]);
    }

    #[test]
    fn test_asc_parser_multiple_records() {
        let asc = "date Mon Feb 26 10:00:00 2026\n\
                   base hex  timestamps absolute\n\
                   0.001000 1 0CF00400 Rx d 4 DE AD BE EF\n\
                   0.002000 2 18FEF100 Tx d 3 AA BB CC\n\
                   0.003000 1 00000123 Rx d 2 11 22\n";

        let records = AscParser::parse(asc).expect("parse should succeed");
        assert_eq!(records.len(), 3);

        assert!((records[0].timestamp - 0.001000).abs() < 1e-9);
        assert_eq!(records[0].data, vec![0xDE, 0xAD, 0xBE, 0xEF]);

        assert_eq!(records[1].channel, 2);
        assert_eq!(records[1].direction, Direction::Tx);
        assert_eq!(records[1].data, vec![0xAA, 0xBB, 0xCC]);

        assert_eq!(records[2].frame_id, 0x123);
        assert_eq!(records[2].dlc, 2);
    }

    #[test]
    fn test_asc_parser_empty_input() {
        let records = AscParser::parse("").expect("parse should succeed");
        assert!(records.is_empty());
    }

    #[test]
    fn test_asc_parser_skips_header_lines() {
        let asc = "date Mon Feb 26 10:00:00 2026\n\
                   base hex timestamps absolute\n\
                   no internal events logged\n\
                   // Comment line\n\
                   ; Another comment\n\
                   Begin Triggerblock\n\
                   0.000001 1 00000100 Rx d 1 AB\n\
                   End TriggerBlock\n";

        let records = AscParser::parse(asc).expect("parse should succeed");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].data, vec![0xAB]);
    }

    #[test]
    fn test_asc_parser_zero_dlc() {
        let asc = "0.000500 1 00000200 Rx d 0\n";
        let records = AscParser::parse(asc).expect("parse should succeed");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].dlc, 0);
        assert!(records[0].data.is_empty());
    }

    #[test]
    fn test_asc_parser_full_8_byte_frame() {
        let asc = "1.234567 2 0CF00400 Tx d 8 11 22 33 44 55 66 77 88\n";
        let records = AscParser::parse(asc).expect("parse should succeed");
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert_eq!(r.dlc, 8);
        assert_eq!(r.data, vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
    }

    // ---- AscWriter tests ----

    #[test]
    fn test_asc_writer_produces_parseable_output() {
        let records = vec![
            AscRecord::new(0.001, 1, 0x0CF00400, vec![0x01, 0x02, 0x03], Direction::Rx),
            AscRecord::new(0.002, 1, 0x18FEF100, vec![0xFF, 0x00], Direction::Tx),
        ];

        let asc_text = AscWriter::write(&records);

        // Should be parseable back to the same records
        let parsed = AscParser::parse(&asc_text).expect("parse should succeed");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].frame_id, 0x0CF00400);
        assert_eq!(parsed[0].data, vec![0x01, 0x02, 0x03]);
        assert_eq!(parsed[1].direction, Direction::Tx);
    }

    #[test]
    fn test_asc_writer_header_format() {
        let records = vec![AscRecord::new(1.0, 1, 0x123, vec![0xAB], Direction::Rx)];
        let asc_text = AscWriter::write(&records);

        assert!(asc_text.contains("date "));
        assert!(asc_text.contains("base hex"));
        assert!(asc_text.contains("no internal events logged"));
    }

    #[test]
    fn test_asc_writer_custom_header() {
        let records = vec![];
        let asc_text = AscWriter::write_with_header(&records, "Mon Feb 26 10:00:00 2026");
        assert!(asc_text.contains("date Mon Feb 26 10:00:00 2026"));
    }

    #[test]
    fn test_asc_writer_empty_records() {
        let records = vec![];
        let asc_text = AscWriter::write(&records);
        // Should have header but no data lines
        let parsed = AscParser::parse(&asc_text).expect("parse should succeed");
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_asc_roundtrip() {
        let original = vec![
            AscRecord::new(0.0001, 1, 0x100, vec![0xAA, 0xBB], Direction::Rx),
            AscRecord::new(0.0002, 1, 0x200, vec![0x11, 0x22, 0x33], Direction::Tx),
            AscRecord::new(0.0003, 2, 0x0CF00400, vec![0; 8], Direction::Rx),
        ];

        let text = AscWriter::write(&original);
        let parsed = AscParser::parse(&text).expect("parse should succeed");

        assert_eq!(parsed.len(), original.len());
        for (orig, parsed) in original.iter().zip(parsed.iter()) {
            assert!((orig.timestamp - parsed.timestamp).abs() < 1e-6);
            assert_eq!(orig.channel, parsed.channel);
            assert_eq!(orig.frame_id, parsed.frame_id);
            assert_eq!(orig.data, parsed.data);
            assert_eq!(orig.direction, parsed.direction);
            assert_eq!(orig.dlc, parsed.dlc);
        }
    }

    // ---- BlfRecord tests ----

    #[test]
    fn test_blf_record_new() {
        let rec = BlfRecord::new(1_000_000_000, 1, 0x0CF00400, vec![0xDE, 0xAD]);
        assert_eq!(rec.timestamp_ns, 1_000_000_000);
        assert_eq!(rec.channel, 1);
        assert_eq!(rec.frame_id, 0x0CF00400);
        assert_eq!(rec.dlc, 2);
        assert_eq!(rec.data, vec![0xDE, 0xAD]);
    }

    #[test]
    fn test_blf_record_to_asc_record() {
        let blf = BlfRecord::new(500_000_000, 2, 0x18FEF100, vec![0x01, 0x02]);
        let asc = blf.to_asc_record();
        assert!((asc.timestamp - 0.5).abs() < 1e-9);
        assert_eq!(asc.channel, 2);
        assert_eq!(asc.frame_id, 0x18FEF100);
    }

    // ---- BlfWriter tests ----

    #[test]
    fn test_blf_writer_magic_header() {
        let records = vec![BlfRecord::new(0, 1, 0x123, vec![0x01])];
        let bytes = BlfWriter::write(&records);

        assert_eq!(&bytes[..8], b"BLF0200\x00");
    }

    #[test]
    fn test_blf_writer_object_count_in_header() {
        let records = vec![
            BlfRecord::new(0, 1, 0x100, vec![0x01]),
            BlfRecord::new(1000, 1, 0x200, vec![0x02]),
            BlfRecord::new(2000, 1, 0x300, vec![0x03]),
        ];
        let bytes = BlfWriter::write(&records);

        // Object count is at bytes 16..20 (little-endian u32)
        let count = u32::from_le_bytes(bytes[16..20].try_into().expect("valid slice"));
        assert_eq!(count, 3);
    }

    #[test]
    fn test_blf_writer_file_size_in_header() {
        let records = vec![BlfRecord::new(0, 1, 0x100, vec![0x01])];
        let bytes = BlfWriter::write(&records);

        let expected_size = (BLF_HEADER_SIZE + BLF_OBJECT_SIZE) as u64;
        let file_size = u64::from_le_bytes(bytes[8..16].try_into().expect("valid slice"));
        assert_eq!(file_size, expected_size);
    }

    #[test]
    fn test_blf_writer_empty_records() {
        let bytes = BlfWriter::write(&[]);
        assert_eq!(bytes.len(), BLF_HEADER_SIZE);
        // Object count should be 0
        let count = u32::from_le_bytes(bytes[16..20].try_into().expect("valid slice"));
        assert_eq!(count, 0);
    }

    // ---- BlfParser tests ----

    #[test]
    fn test_blf_roundtrip_single_record() {
        let original = vec![BlfRecord::new(
            1_234_567_890,
            1,
            0x0CF00400,
            vec![0xAA, 0xBB, 0xCC],
        )];

        let bytes = BlfWriter::write(&original);
        let parsed = BlfParser::parse(&bytes).expect("parse should succeed");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].timestamp_ns, original[0].timestamp_ns);
        assert_eq!(parsed[0].channel, original[0].channel);
        assert_eq!(parsed[0].frame_id, original[0].frame_id);
        assert_eq!(parsed[0].dlc, original[0].dlc);
        assert_eq!(parsed[0].data, original[0].data);
    }

    #[test]
    fn test_blf_roundtrip_multiple_records() {
        let original = vec![
            BlfRecord::new(0, 1, 0x100, vec![0x11, 0x22]),
            BlfRecord::new(1_000_000, 1, 0x200, vec![0x33, 0x44, 0x55]),
            BlfRecord::new(
                2_000_000,
                2,
                0x0CF00400,
                vec![0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0],
            ),
        ];

        let bytes = BlfWriter::write(&original);
        let parsed = BlfParser::parse(&bytes).expect("parse should succeed");

        assert_eq!(parsed.len(), 3);
        for (i, (orig, parsed)) in original.iter().zip(parsed.iter()).enumerate() {
            assert_eq!(orig.timestamp_ns, parsed.timestamp_ns, "record {}", i);
            assert_eq!(orig.frame_id, parsed.frame_id, "record {}", i);
            assert_eq!(orig.data, parsed.data, "record {}", i);
        }
    }

    #[test]
    fn test_blf_parser_empty_data_error() {
        let result = BlfParser::parse(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_blf_parser_invalid_magic() {
        let mut bytes = BlfWriter::write(&[]);
        // Corrupt the magic bytes
        bytes[0] = b'X';
        let result = BlfParser::parse(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_blf_parser_truncated_data() {
        let records = vec![BlfRecord::new(0, 1, 0x100, vec![0x01])];
        let mut bytes = BlfWriter::write(&records);
        // Truncate the file mid-object
        bytes.truncate(bytes.len() - 10);
        let result = BlfParser::parse(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_blf_record_full_can_fd() {
        // 64-byte CAN FD frame
        let data: Vec<u8> = (0..64u8).collect();
        let rec = BlfRecord::new(0, 1, 0x1FFFFFFF, data.clone());
        assert_eq!(rec.dlc, 64);

        let bytes = BlfWriter::write(&[rec]);
        let parsed = BlfParser::parse(&bytes).expect("parse should succeed");
        assert_eq!(parsed[0].data, data);
    }

    #[test]
    fn test_blf_start_timestamp_in_header() {
        let records = vec![
            BlfRecord::new(999_000_000, 1, 0x100, vec![0x01]),
            BlfRecord::new(1_000_000_000, 1, 0x200, vec![0x02]),
        ];
        let bytes = BlfWriter::write(&records);
        // Start timestamp at bytes 20..28
        let start_ts = u64::from_le_bytes(bytes[20..28].try_into().expect("valid slice"));
        assert_eq!(start_ts, 999_000_000);
    }
}
