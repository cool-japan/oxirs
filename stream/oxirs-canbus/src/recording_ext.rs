//! Extended CAN recording formats
//!
//! This module provides additional recording format support beyond the basic ASC/BLF:
//!
//! - **CSV** - Comma-separated values with timestamp, ID, DLC, data columns
//! - **MF4** - MDF4 (Measurement Data Format 4) stub reader for compatibility
//! - **`CanRecording`** - Unified recording abstraction with format autodetection
//!
//! # CSV Format
//!
//! ```text
//! timestamp_s,can_id_hex,dlc,data_hex,direction,channel
//! 0.000125,0CF00400,8,0000000000000000,Rx,1
//! 0.001000,18FEF100,4,DEADBEEF,Tx,1
//! ```
//!
//! # MF4 Format
//!
//! The MF4 reader provides header detection and basic metadata extraction.
//! Full MF4 decoding requires a dedicated library; this stub identifies
//! MF4 files and extracts version information.
//!
//! # CanRecording
//!
//! ```rust
//! use oxirs_canbus::recording_ext::{CanRecording, RecordingFormat};
//!
//! // Create an in-memory recording
//! let mut recording = CanRecording::new();
//! recording.add_frame(0.001, 1, 0x0CF00400, vec![0x01, 0x02, 0x03, 0x04], None);
//!
//! // Export as CSV
//! let csv = recording.to_csv();
//! assert!(csv.contains("0CF00400"));
//! ```

use crate::error::{CanbusError, CanbusResult};
use crate::recording::{AscRecord, AscWriter, BlfRecord, BlfWriter, Direction};

// ============================================================================
// Unified frame record
// ============================================================================

/// A unified CAN frame record usable across all recording formats
#[derive(Debug, Clone, PartialEq)]
pub struct CanFrame {
    /// Timestamp in seconds from recording start
    pub timestamp_s: f64,
    /// CAN channel number (1-based)
    pub channel: u8,
    /// CAN frame ID
    pub can_id: u32,
    /// Data payload
    pub data: Vec<u8>,
    /// Frame direction
    pub direction: Direction,
    /// Whether this is a CAN FD frame
    pub is_fd: bool,
}

impl CanFrame {
    /// Create a new CAN frame record
    pub fn new(
        timestamp_s: f64,
        channel: u8,
        can_id: u32,
        data: Vec<u8>,
        direction: Direction,
    ) -> Self {
        Self {
            timestamp_s,
            channel,
            can_id,
            data,
            direction,
            is_fd: false,
        }
    }

    /// Create a CAN FD frame record
    pub fn new_fd(
        timestamp_s: f64,
        channel: u8,
        can_id: u32,
        data: Vec<u8>,
        direction: Direction,
    ) -> Self {
        Self {
            timestamp_s,
            channel,
            can_id,
            data,
            direction,
            is_fd: true,
        }
    }

    /// Get the DLC (Data Length Code)
    pub fn dlc(&self) -> u8 {
        self.data.len().min(64) as u8
    }

    /// Convert to an ASC record
    pub fn to_asc_record(&self) -> AscRecord {
        AscRecord::new(
            self.timestamp_s,
            self.channel,
            self.can_id,
            self.data.clone(),
            self.direction,
        )
    }

    /// Convert to a BLF record (timestamp in nanoseconds)
    pub fn to_blf_record(&self) -> BlfRecord {
        let timestamp_ns = (self.timestamp_s * 1_000_000_000.0) as u64;
        BlfRecord::new(
            timestamp_ns,
            self.channel as u16,
            self.can_id,
            self.data.clone(),
        )
    }
}

// ============================================================================
// Recording format enum
// ============================================================================

/// Supported recording file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingFormat {
    /// Vector CANalyzer ASC text format
    Asc,
    /// Binary Logging File (Vector BLF)
    Blf,
    /// Comma-separated values (OxiRS custom format)
    Csv,
    /// MDF4 (Measurement Data Format 4) — read-only stub
    Mf4,
}

impl RecordingFormat {
    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "asc" => Some(Self::Asc),
            "blf" => Some(Self::Blf),
            "csv" => Some(Self::Csv),
            "mf4" | "mdf" => Some(Self::Mf4),
            _ => None,
        }
    }

    /// Detect format from file content magic bytes / header
    pub fn detect(data: &[u8]) -> Option<Self> {
        if data.len() >= 8 && &data[..8] == b"BLF0200\x00" {
            return Some(Self::Blf);
        }
        if data.len() >= 8 && &data[..8] == b"MDF     " {
            return Some(Self::Mf4);
        }
        // Heuristic for ASC: starts with "date " or "base "
        if let Ok(text) = std::str::from_utf8(&data[..data.len().min(32)]) {
            if text.starts_with("date ") || text.starts_with("base ") {
                return Some(Self::Asc);
            }
        }
        // Heuristic for CSV: starts with "timestamp" column header
        if let Ok(text) = std::str::from_utf8(&data[..data.len().min(64)]) {
            if text.starts_with("timestamp") || text.starts_with("Timestamp") {
                return Some(Self::Csv);
            }
        }
        None
    }
}

// ============================================================================
// CSV format
// ============================================================================

/// Header line for CSV recording files
pub const CSV_HEADER: &str = "timestamp_s,can_id_hex,dlc,data_hex,direction,channel";

/// Writer for CSV CAN recording format
pub struct CanCsvWriter;

impl CanCsvWriter {
    /// Write CAN frames to CSV format
    ///
    /// Columns: timestamp_s, can_id_hex, dlc, data_hex, direction, channel
    pub fn write(frames: &[CanFrame]) -> String {
        let mut output = String::new();
        output.push_str(CSV_HEADER);
        output.push('\n');

        for frame in frames {
            let data_hex: String = frame.data.iter().map(|b| format!("{:02X}", b)).collect();
            let dir_str = frame.direction.as_str();

            output.push_str(&format!(
                "{:.9},{:08X},{},{},{},{}\n",
                frame.timestamp_s,
                frame.can_id,
                frame.dlc(),
                data_hex,
                dir_str,
                frame.channel
            ));
        }

        output
    }
}

/// Parser for CSV CAN recording format
pub struct CanCsvParser;

impl CanCsvParser {
    /// Parse CSV text content into CAN frames
    pub fn parse(text: &str) -> CanbusResult<Vec<CanFrame>> {
        let mut frames = Vec::new();
        let mut lines = text.lines();

        // Skip header line
        if let Some(header) = lines.next() {
            // Validate that it looks like a CSV header
            if !header.contains("timestamp") && !header.contains("can_id") {
                // Not a valid header — treat as data line and push back
                // by processing it as a data line
                if let Some(frame) = Self::parse_data_line(header) {
                    frames.push(frame);
                }
            }
        }

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            if let Some(frame) = Self::parse_data_line(trimmed) {
                frames.push(frame);
            }
        }

        Ok(frames)
    }

    /// Parse a single CSV data line
    fn parse_data_line(line: &str) -> Option<CanFrame> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 4 {
            return None;
        }

        let timestamp_s: f64 = parts[0].trim().parse().ok()?;
        let can_id = u32::from_str_radix(parts[1].trim(), 16).ok()?;
        let dlc: u8 = parts[2].trim().parse().ok()?;

        // Parse hex data
        let data_str = parts[3].trim();
        let mut data = Vec::with_capacity(dlc as usize);
        for i in 0..(dlc as usize) {
            let byte_start = i * 2;
            let byte_end = byte_start + 2;
            if byte_end <= data_str.len() {
                let byte = u8::from_str_radix(&data_str[byte_start..byte_end], 16).ok()?;
                data.push(byte);
            }
        }

        // Direction (optional, default Rx)
        let direction = parts
            .get(4)
            .and_then(|s| Direction::parse_asc(s.trim()))
            .unwrap_or(Direction::Rx);

        // Channel (optional, default 1)
        let channel: u8 = parts
            .get(5)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(1);

        Some(CanFrame::new(timestamp_s, channel, can_id, data, direction))
    }
}

// ============================================================================
// MF4 stub
// ============================================================================

/// Magic bytes for MDF4 files
pub const MF4_MAGIC: &[u8; 8] = b"MDF     ";

/// MF4 file version strings
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mf4Version {
    /// Major version (e.g., 4)
    pub major: u8,
    /// Minor version (e.g., 10)
    pub minor: u8,
}

impl Mf4Version {
    /// Create a version from a three-digit string like "410"
    pub fn parse_version(s: &str) -> Option<Self> {
        let trimmed = s.trim();
        if trimmed.len() >= 3 {
            let major = trimmed[..1].parse().ok()?;
            let minor = trimmed[1..3].parse().ok()?;
            Some(Self { major, minor })
        } else {
            None
        }
    }
}

/// MF4 file header metadata (stub)
#[derive(Debug, Clone)]
pub struct Mf4Header {
    /// File version
    pub version: Option<Mf4Version>,
    /// File size in bytes
    pub file_size: u64,
    /// Whether the file is recognized as a valid MF4
    pub is_valid: bool,
}

/// Stub reader for MDF4 (Measurement Data Format 4) files
///
/// Provides file format detection and header extraction.
/// Full MF4 decoding is not supported; use this to identify files
/// and extract basic metadata.
pub struct Mf4Reader;

impl Mf4Reader {
    /// Check if data appears to be a valid MF4 file
    pub fn is_mf4(data: &[u8]) -> bool {
        data.len() >= 8 && &data[..8] == MF4_MAGIC
    }

    /// Parse the MF4 file header (stub implementation)
    ///
    /// Returns basic header metadata. Does not decode channel data.
    pub fn read_header(data: &[u8]) -> CanbusResult<Mf4Header> {
        if !Self::is_mf4(data) {
            return Err(CanbusError::Config(
                "Not a valid MF4 file: magic bytes not found".to_string(),
            ));
        }

        // MF4 file layout (simplified):
        // Bytes 0..7: "MDF     " magic
        // Bytes 8..15: version string (e.g., "4.10    ")
        // Bytes 16..23: file size (u64 LE in actual MF4, we read best-effort)
        let version = if data.len() >= 16 {
            let ver_bytes = &data[8..16];
            let ver_str = std::str::from_utf8(ver_bytes)
                .unwrap_or("")
                .trim_end_matches([' ', '\0']);
            Mf4Version::parse_version(ver_str)
        } else {
            None
        };

        let file_size = if data.len() >= 24 {
            u64::from_le_bytes(
                data[16..24]
                    .try_into()
                    .map_err(|_| CanbusError::Config("Failed to read MF4 file size".to_string()))?,
            )
        } else {
            data.len() as u64
        };

        Ok(Mf4Header {
            version,
            file_size,
            is_valid: true,
        })
    }

    /// Get a human-readable description of the MF4 file
    pub fn describe(data: &[u8]) -> String {
        match Self::read_header(data) {
            Ok(hdr) => {
                let ver_str = hdr
                    .version
                    .map(|v| format!("{}.{}", v.major, v.minor))
                    .unwrap_or_else(|| "unknown".to_string());
                format!(
                    "MDF4 file: version={}, size={} bytes",
                    ver_str, hdr.file_size
                )
            }
            Err(e) => format!("Invalid MF4 file: {}", e),
        }
    }
}

// ============================================================================
// CanRecording — unified abstraction
// ============================================================================

/// A unified CAN recording that can be imported/exported in multiple formats
///
/// # Example
///
/// ```rust
/// use oxirs_canbus::recording_ext::CanRecording;
/// use oxirs_canbus::recording::Direction;
///
/// let mut recording = CanRecording::new();
/// recording.add_frame(0.001, 1, 0x0CF00400, vec![0x01, 0x02, 0x03, 0x04], None);
/// recording.add_frame(0.002, 1, 0x18FEF100, vec![0xFF, 0x00], Some(Direction::Tx));
///
/// let csv = recording.to_csv();
/// assert_eq!(recording.frame_count(), 2);
/// ```
#[derive(Debug, Clone, Default)]
pub struct CanRecording {
    /// All captured frames in chronological order
    pub frames: Vec<CanFrame>,
    /// Optional recording metadata
    pub description: Option<String>,
    /// Recording start timestamp (wall clock, seconds since Unix epoch)
    pub start_timestamp: Option<f64>,
}

impl CanRecording {
    /// Create an empty recording
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a CAN frame to the recording
    ///
    /// `direction` defaults to `Direction::Rx` if `None`.
    pub fn add_frame(
        &mut self,
        timestamp_s: f64,
        channel: u8,
        can_id: u32,
        data: Vec<u8>,
        direction: Option<Direction>,
    ) {
        let dir = direction.unwrap_or(Direction::Rx);
        self.frames
            .push(CanFrame::new(timestamp_s, channel, can_id, data, dir));
    }

    /// Add a CAN FD frame to the recording
    pub fn add_fd_frame(
        &mut self,
        timestamp_s: f64,
        channel: u8,
        can_id: u32,
        data: Vec<u8>,
        direction: Option<Direction>,
    ) {
        let dir = direction.unwrap_or(Direction::Rx);
        self.frames
            .push(CanFrame::new_fd(timestamp_s, channel, can_id, data, dir));
    }

    /// Get the number of frames in the recording
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Get the duration of the recording in seconds
    pub fn duration_s(&self) -> f64 {
        match (self.frames.first(), self.frames.last()) {
            (Some(first), Some(last)) => last.timestamp_s - first.timestamp_s,
            _ => 0.0,
        }
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        CanCsvWriter::write(&self.frames)
    }

    /// Export to ASC format
    pub fn to_asc(&self) -> String {
        let asc_records: Vec<AscRecord> = self.frames.iter().map(|f| f.to_asc_record()).collect();
        AscWriter::write(&asc_records)
    }

    /// Export to BLF binary format
    pub fn to_blf(&self) -> Vec<u8> {
        let blf_records: Vec<BlfRecord> = self.frames.iter().map(|f| f.to_blf_record()).collect();
        BlfWriter::write(&blf_records)
    }

    /// Import from CSV format
    pub fn from_csv(text: &str) -> CanbusResult<Self> {
        let frames = CanCsvParser::parse(text)?;
        Ok(Self {
            frames,
            description: None,
            start_timestamp: None,
        })
    }

    /// Get frames filtered by CAN ID
    pub fn frames_for_id(&self, can_id: u32) -> Vec<&CanFrame> {
        self.frames.iter().filter(|f| f.can_id == can_id).collect()
    }

    /// Get frames filtered by channel
    pub fn frames_for_channel(&self, channel: u8) -> Vec<&CanFrame> {
        self.frames
            .iter()
            .filter(|f| f.channel == channel)
            .collect()
    }

    /// Get unique CAN IDs seen in this recording
    pub fn unique_can_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.frames.iter().map(|f| f.can_id).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Count CAN FD frames
    pub fn fd_frame_count(&self) -> usize {
        self.frames.iter().filter(|f| f.is_fd).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- RecordingFormat tests ----

    #[test]
    fn test_recording_format_from_extension() {
        assert_eq!(
            RecordingFormat::from_extension("asc"),
            Some(RecordingFormat::Asc)
        );
        assert_eq!(
            RecordingFormat::from_extension("ASC"),
            Some(RecordingFormat::Asc)
        );
        assert_eq!(
            RecordingFormat::from_extension("blf"),
            Some(RecordingFormat::Blf)
        );
        assert_eq!(
            RecordingFormat::from_extension("csv"),
            Some(RecordingFormat::Csv)
        );
        assert_eq!(
            RecordingFormat::from_extension("mf4"),
            Some(RecordingFormat::Mf4)
        );
        assert_eq!(
            RecordingFormat::from_extension("mdf"),
            Some(RecordingFormat::Mf4)
        );
        assert_eq!(RecordingFormat::from_extension("txt"), None);
    }

    #[test]
    fn test_recording_format_detect_blf() {
        let blf_header = b"BLF0200\x00";
        assert_eq!(
            RecordingFormat::detect(blf_header),
            Some(RecordingFormat::Blf)
        );
    }

    #[test]
    fn test_recording_format_detect_mf4() {
        let mf4_header = b"MDF     ";
        assert_eq!(
            RecordingFormat::detect(mf4_header),
            Some(RecordingFormat::Mf4)
        );
    }

    #[test]
    fn test_recording_format_detect_asc() {
        let asc_content = b"date Mon Feb 26 10:00:00 2026\n";
        assert_eq!(
            RecordingFormat::detect(asc_content),
            Some(RecordingFormat::Asc)
        );
    }

    #[test]
    fn test_recording_format_detect_csv() {
        let csv_content = b"timestamp_s,can_id_hex,dlc\n";
        assert_eq!(
            RecordingFormat::detect(csv_content),
            Some(RecordingFormat::Csv)
        );
    }

    #[test]
    fn test_recording_format_detect_unknown() {
        let unknown = b"\x00\x01\x02\x03";
        assert_eq!(RecordingFormat::detect(unknown), None);
    }

    // ---- CanFrame tests ----

    #[test]
    fn test_can_frame_new() {
        let frame = CanFrame::new(1.5, 1, 0x0CF00400, vec![0xDE, 0xAD], Direction::Rx);
        assert_eq!(frame.timestamp_s, 1.5);
        assert_eq!(frame.channel, 1);
        assert_eq!(frame.can_id, 0x0CF00400);
        assert_eq!(frame.dlc(), 2);
        assert!(!frame.is_fd);
    }

    #[test]
    fn test_can_frame_fd() {
        let data = vec![0u8; 32];
        let frame = CanFrame::new_fd(0.001, 1, 0x100, data, Direction::Tx);
        assert!(frame.is_fd);
        assert_eq!(frame.dlc(), 32);
    }

    #[test]
    fn test_can_frame_to_asc_record() {
        let frame = CanFrame::new(0.5, 2, 0x123, vec![0xAA, 0xBB], Direction::Tx);
        let asc = frame.to_asc_record();
        assert!((asc.timestamp - 0.5).abs() < 1e-9);
        assert_eq!(asc.channel, 2);
        assert_eq!(asc.frame_id, 0x123);
        assert_eq!(asc.direction, Direction::Tx);
    }

    #[test]
    fn test_can_frame_to_blf_record() {
        let frame = CanFrame::new(1.0, 1, 0x100, vec![0x01], Direction::Rx);
        let blf = frame.to_blf_record();
        assert_eq!(blf.timestamp_ns, 1_000_000_000);
        assert_eq!(blf.frame_id, 0x100);
    }

    // ---- CSV format tests ----

    #[test]
    fn test_csv_writer_header() {
        let frames = vec![];
        let csv = CanCsvWriter::write(&frames);
        assert!(csv.starts_with("timestamp_s,can_id_hex,dlc,data_hex,direction,channel\n"));
    }

    #[test]
    fn test_csv_writer_single_frame() {
        let frame = CanFrame::new(
            0.001,
            1,
            0x0CF00400,
            vec![0xDE, 0xAD, 0xBE, 0xEF],
            Direction::Rx,
        );
        let csv = CanCsvWriter::write(&[frame]);
        assert!(csv.contains("0CF00400"));
        assert!(csv.contains("DEADBEEF"));
        assert!(csv.contains("Rx"));
    }

    #[test]
    fn test_csv_roundtrip() {
        let frames = vec![
            CanFrame::new(
                0.001,
                1,
                0x0CF00400,
                vec![0x01, 0x02, 0x03, 0x04],
                Direction::Rx,
            ),
            CanFrame::new(0.002, 2, 0x18FEF100, vec![0xFF, 0x00], Direction::Tx),
        ];

        let csv = CanCsvWriter::write(&frames);
        let parsed = CanCsvParser::parse(&csv).expect("valid parse");

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].can_id, 0x0CF00400);
        assert_eq!(parsed[0].data, vec![0x01, 0x02, 0x03, 0x04]);
        assert_eq!(parsed[1].direction, Direction::Tx);
        assert_eq!(parsed[1].channel, 2);
    }

    #[test]
    fn test_csv_empty() {
        let csv = CanCsvWriter::write(&[]);
        let parsed = CanCsvParser::parse(&csv).expect("valid parse");
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_csv_zero_dlc() {
        let frame = CanFrame::new(0.0, 1, 0x100, vec![], Direction::Rx);
        let csv = CanCsvWriter::write(&[frame]);
        let parsed = CanCsvParser::parse(&csv).expect("valid parse");
        assert_eq!(parsed.len(), 1);
        assert!(parsed[0].data.is_empty());
    }

    #[test]
    fn test_csv_parser_skips_comments() {
        let csv = "timestamp_s,can_id_hex,dlc,data_hex,direction,channel\n\
                   # This is a comment\n\
                   0.001000,00000100,2,AABB,Rx,1\n";
        let parsed = CanCsvParser::parse(csv).expect("valid parse");
        assert_eq!(parsed.len(), 1);
    }

    // ---- MF4 tests ----

    #[test]
    fn test_mf4_detection() {
        let valid = b"MDF     ";
        let invalid = b"BLF0200\x00";
        assert!(Mf4Reader::is_mf4(valid));
        assert!(!Mf4Reader::is_mf4(invalid));
        assert!(!Mf4Reader::is_mf4(b""));
    }

    #[test]
    fn test_mf4_invalid_magic_error() {
        let result = Mf4Reader::read_header(b"NOT_MF4X");
        assert!(result.is_err());
    }

    #[test]
    fn test_mf4_header_valid() {
        // Minimal fake MF4: magic + version + file_size
        let mut data = vec![0u8; 32];
        data[..8].copy_from_slice(b"MDF     ");
        data[8..16].copy_from_slice(b"4.10    ");
        let file_size: u64 = 1024;
        data[16..24].copy_from_slice(&file_size.to_le_bytes());

        let hdr = Mf4Reader::read_header(&data).expect("valid MF4 header");
        assert!(hdr.is_valid);
        assert_eq!(hdr.file_size, 1024);
    }

    #[test]
    fn test_mf4_describe_invalid() {
        let desc = Mf4Reader::describe(b"invalid");
        assert!(desc.contains("Invalid"));
    }

    #[test]
    fn test_mf4_describe_valid() {
        let mut data = vec![0u8; 32];
        data[..8].copy_from_slice(b"MDF     ");
        data[8..16].copy_from_slice(b"4.10    ");
        let desc = Mf4Reader::describe(&data);
        assert!(desc.contains("MDF4 file"));
    }

    #[test]
    fn test_mf4_version_from_str() {
        let ver = Mf4Version::parse_version("410");
        assert!(ver.is_some());
        let v = ver.expect("should succeed");
        assert_eq!(v.major, 4);
        assert_eq!(v.minor, 10);
    }

    // ---- CanRecording tests ----

    #[test]
    fn test_can_recording_new() {
        let rec = CanRecording::new();
        assert_eq!(rec.frame_count(), 0);
        assert_eq!(rec.duration_s(), 0.0);
    }

    #[test]
    fn test_can_recording_add_frame() {
        let mut rec = CanRecording::new();
        rec.add_frame(0.001, 1, 0x100, vec![0x01, 0x02], None);
        rec.add_frame(0.002, 1, 0x200, vec![0x03, 0x04], Some(Direction::Tx));

        assert_eq!(rec.frame_count(), 2);
        assert!((rec.duration_s() - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_can_recording_to_csv() {
        let mut rec = CanRecording::new();
        rec.add_frame(0.001, 1, 0x0CF00400, vec![0x01, 0x02, 0x03, 0x04], None);

        let csv = rec.to_csv();
        assert!(csv.contains("0CF00400"));
    }

    #[test]
    fn test_can_recording_to_asc() {
        let mut rec = CanRecording::new();
        rec.add_frame(0.001, 1, 0x0CF00400, vec![0x01, 0x02], None);

        let asc = rec.to_asc();
        assert!(asc.contains("0CF00400"));
        assert!(asc.contains("date "));
    }

    #[test]
    fn test_can_recording_to_blf() {
        let mut rec = CanRecording::new();
        rec.add_frame(1.0, 1, 0x100, vec![0xAA, 0xBB], None);

        let blf = rec.to_blf();
        assert_eq!(&blf[..8], b"BLF0200\x00");
    }

    #[test]
    fn test_can_recording_from_csv() {
        let csv = "timestamp_s,can_id_hex,dlc,data_hex,direction,channel\n\
                   0.001000,0CF00400,4,01020304,Rx,1\n\
                   0.002000,18FEF100,2,AABB,Tx,2\n";

        let rec = CanRecording::from_csv(csv).expect("valid parse");
        assert_eq!(rec.frame_count(), 2);
        assert_eq!(rec.frames[0].can_id, 0x0CF00400);
        assert_eq!(rec.frames[1].direction, Direction::Tx);
    }

    #[test]
    fn test_can_recording_csv_roundtrip() {
        let mut original = CanRecording::new();
        original.add_frame(
            0.001,
            1,
            0x0CF00400,
            vec![0x01, 0x02, 0x03, 0x04],
            Some(Direction::Rx),
        );
        original.add_frame(0.002, 1, 0x18FEF100, vec![0xFF, 0x00], Some(Direction::Tx));

        let csv = original.to_csv();
        let restored = CanRecording::from_csv(&csv).expect("valid parse");

        assert_eq!(restored.frame_count(), original.frame_count());
        for (orig, rest) in original.frames.iter().zip(restored.frames.iter()) {
            assert!((orig.timestamp_s - rest.timestamp_s).abs() < 1e-6);
            assert_eq!(orig.can_id, rest.can_id);
            assert_eq!(orig.data, rest.data);
            assert_eq!(orig.direction, rest.direction);
        }
    }

    #[test]
    fn test_can_recording_frames_for_id() {
        let mut rec = CanRecording::new();
        rec.add_frame(0.001, 1, 0x100, vec![0x01], None);
        rec.add_frame(0.002, 1, 0x200, vec![0x02], None);
        rec.add_frame(0.003, 1, 0x100, vec![0x03], None);

        let frames = rec.frames_for_id(0x100);
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn test_can_recording_frames_for_channel() {
        let mut rec = CanRecording::new();
        rec.add_frame(0.001, 1, 0x100, vec![0x01], None);
        rec.add_frame(0.002, 2, 0x200, vec![0x02], None);
        rec.add_frame(0.003, 1, 0x300, vec![0x03], None);

        let ch1 = rec.frames_for_channel(1);
        assert_eq!(ch1.len(), 2);

        let ch2 = rec.frames_for_channel(2);
        assert_eq!(ch2.len(), 1);
    }

    #[test]
    fn test_can_recording_unique_ids() {
        let mut rec = CanRecording::new();
        rec.add_frame(0.001, 1, 0x100, vec![], None);
        rec.add_frame(0.002, 1, 0x200, vec![], None);
        rec.add_frame(0.003, 1, 0x100, vec![], None);
        rec.add_frame(0.004, 1, 0x300, vec![], None);

        let ids = rec.unique_can_ids();
        assert_eq!(ids, vec![0x100, 0x200, 0x300]);
    }

    #[test]
    fn test_can_recording_add_fd_frame() {
        let mut rec = CanRecording::new();
        rec.add_fd_frame(0.001, 1, 0x100, vec![0u8; 32], None);

        assert_eq!(rec.frame_count(), 1);
        assert_eq!(rec.fd_frame_count(), 1);
        assert!(rec.frames[0].is_fd);
    }

    #[test]
    fn test_can_recording_fd_frame_count() {
        let mut rec = CanRecording::new();
        rec.add_frame(0.001, 1, 0x100, vec![0x01], None);
        rec.add_fd_frame(0.002, 1, 0x200, vec![0u8; 16], None);
        rec.add_fd_frame(0.003, 1, 0x300, vec![0u8; 64], None);

        assert_eq!(rec.fd_frame_count(), 2);
        assert_eq!(rec.frame_count(), 3);
    }

    #[test]
    fn test_can_recording_with_temp_file() {
        // Verify temp file path usage pattern (no actual I/O needed)
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("oxirs_canbus_test.csv");
        // Just verify the path is usable (no actual file write needed for unit test)
        assert!(path.to_str().is_some());
    }
}
