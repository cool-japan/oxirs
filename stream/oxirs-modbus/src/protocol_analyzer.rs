//! Modbus protocol frame analysis and statistics.
//!
//! Provides frame ingestion, function-code metadata lookup, exception
//! detection, request/response pairing by transaction ID, and aggregate
//! statistics collection (histograms, averages, error counts).

use std::collections::HashMap;
use std::fmt;

// ── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by the protocol analyzer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalyzerError {
    /// A frame supplied to `ingest` was structurally invalid.
    InvalidFrame(String),
}

impl fmt::Display for AnalyzerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFrame(msg) => write!(f, "invalid Modbus frame: {msg}"),
        }
    }
}

impl std::error::Error for AnalyzerError {}

// ── FunctionCategory ─────────────────────────────────────────────────────────

/// High-level category for a Modbus function code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionCategory {
    /// Read operations (FC01, FC02, FC03, FC04).
    Read,
    /// Write operations (FC05, FC06, FC15, FC16).
    Write,
    /// Diagnostic / loopback operations (FC08).
    Diagnostic,
    /// Any function code not otherwise classified.
    Other,
}

impl fmt::Display for FunctionCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read => write!(f, "Read"),
            Self::Write => write!(f, "Write"),
            Self::Diagnostic => write!(f, "Diagnostic"),
            Self::Other => write!(f, "Other"),
        }
    }
}

// ── FunctionCodeInfo ─────────────────────────────────────────────────────────

/// Metadata about a Modbus function code.
#[derive(Debug, Clone)]
pub struct FunctionCodeInfo {
    /// The raw function code byte.
    pub code: u8,
    /// Human-readable name.
    pub name: &'static str,
    /// High-level category.
    pub category: FunctionCategory,
}

// ── ModbusFrame ──────────────────────────────────────────────────────────────

/// A captured Modbus TCP frame (MBAP header + PDU data).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModbusFrame {
    /// Modbus TCP transaction identifier.
    pub transaction_id: u16,
    /// Unit (slave) identifier.
    pub unit_id: u8,
    /// Function code (may have exception bit set).
    pub function_code: u8,
    /// PDU data bytes (excluding function code).
    pub data: Vec<u8>,
    /// `true` if this is a client request; `false` if a server response.
    pub is_request: bool,
    /// Capture timestamp in milliseconds since Unix epoch.
    pub timestamp_ms: u64,
}

impl ModbusFrame {
    /// Create a new `ModbusFrame`.
    pub fn new(
        transaction_id: u16,
        unit_id: u8,
        function_code: u8,
        data: Vec<u8>,
        is_request: bool,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            transaction_id,
            unit_id,
            function_code,
            data,
            is_request,
            timestamp_ms,
        }
    }
}

// ── AnalysisStats ────────────────────────────────────────────────────────────

/// Aggregate statistics produced by `ProtocolAnalyzer::analyze`.
#[derive(Debug, Clone)]
pub struct AnalysisStats {
    /// Total frames ingested.
    pub total_frames: usize,
    /// Number of request frames (is_request == true).
    pub request_count: usize,
    /// Number of response frames (is_request == false).
    pub response_count: usize,
    /// Number of exception / error frames.
    pub error_count: usize,
    /// Average number of data bytes per frame.
    pub avg_data_bytes: f64,
    /// How many times each function code appeared.
    pub function_histogram: HashMap<u8, usize>,
}

// ── ProtocolAnalyzer ─────────────────────────────────────────────────────────

/// Stateful Modbus protocol frame analyzer.
///
/// Frames are fed via `ingest`. Call `analyze` at any time to get a snapshot
/// of the current statistics.
pub struct ProtocolAnalyzer {
    frames: Vec<ModbusFrame>,
}

impl ProtocolAnalyzer {
    /// Create a new, empty `ProtocolAnalyzer`.
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Ingest a captured frame.
    pub fn ingest(&mut self, frame: ModbusFrame) {
        self.frames.push(frame);
    }

    /// Compute and return aggregate statistics over all ingested frames.
    pub fn analyze(&self) -> AnalysisStats {
        let total_frames = self.frames.len();
        let mut request_count = 0usize;
        let mut response_count = 0usize;
        let mut error_count = 0usize;
        let mut total_bytes = 0usize;
        let mut function_histogram: HashMap<u8, usize> = HashMap::new();

        for frame in &self.frames {
            if frame.is_request {
                request_count += 1;
            } else {
                response_count += 1;
            }
            if Self::is_exception(frame) {
                error_count += 1;
            }
            total_bytes += frame.data.len();
            *function_histogram.entry(frame.function_code).or_insert(0) += 1;
        }

        let avg_data_bytes = if total_frames == 0 {
            0.0
        } else {
            total_bytes as f64 / total_frames as f64
        };

        AnalysisStats {
            total_frames,
            request_count,
            response_count,
            error_count,
            avg_data_bytes,
            function_histogram,
        }
    }

    /// Return metadata for a known function code.
    ///
    /// Covers FC 01-06, 08, 15-16. Any other value returns `"Unknown"`.
    pub fn function_info(code: u8) -> FunctionCodeInfo {
        match code {
            0x01 => FunctionCodeInfo {
                code,
                name: "Read Coil Status",
                category: FunctionCategory::Read,
            },
            0x02 => FunctionCodeInfo {
                code,
                name: "Read Discrete Inputs",
                category: FunctionCategory::Read,
            },
            0x03 => FunctionCodeInfo {
                code,
                name: "Read Holding Registers",
                category: FunctionCategory::Read,
            },
            0x04 => FunctionCodeInfo {
                code,
                name: "Read Input Registers",
                category: FunctionCategory::Read,
            },
            0x05 => FunctionCodeInfo {
                code,
                name: "Write Single Coil",
                category: FunctionCategory::Write,
            },
            0x06 => FunctionCodeInfo {
                code,
                name: "Write Single Register",
                category: FunctionCategory::Write,
            },
            0x08 => FunctionCodeInfo {
                code,
                name: "Diagnostics",
                category: FunctionCategory::Diagnostic,
            },
            0x0F => FunctionCodeInfo {
                code,
                name: "Write Multiple Coils",
                category: FunctionCategory::Write,
            },
            0x10 => FunctionCodeInfo {
                code,
                name: "Write Multiple Registers",
                category: FunctionCategory::Write,
            },
            _ => FunctionCodeInfo {
                code,
                name: "Unknown",
                category: FunctionCategory::Other,
            },
        }
    }

    /// Return `true` if `frame` carries an exception response.
    ///
    /// In Modbus, the server sets bit 7 of the function code to signal an
    /// exception: `(function_code & 0x80) != 0`.
    pub fn is_exception(frame: &ModbusFrame) -> bool {
        frame.function_code & 0x80 != 0
    }

    /// Match request/response pairs by `transaction_id`.
    ///
    /// Returns clones of matched pairs `(request, response)`.
    /// Only the first request and first response sharing a transaction ID are
    /// included (subsequent duplicates are ignored).
    pub fn find_request_response_pairs(&self) -> Vec<(ModbusFrame, ModbusFrame)> {
        let mut requests: HashMap<u16, &ModbusFrame> = HashMap::new();
        let mut responses: HashMap<u16, &ModbusFrame> = HashMap::new();

        for frame in &self.frames {
            let tid = frame.transaction_id;
            if frame.is_request {
                requests.entry(tid).or_insert(frame);
            } else {
                responses.entry(tid).or_insert(frame);
            }
        }

        let mut pairs = Vec::new();
        for (tid, req) in &requests {
            if let Some(resp) = responses.get(tid) {
                pairs.push(((*req).clone(), (*resp).clone()));
            }
        }
        // Sort by transaction_id for deterministic output.
        pairs.sort_by_key(|(req, _)| req.transaction_id);
        pairs
    }

    /// Return references to all exception frames.
    pub fn error_frames(&self) -> Vec<&ModbusFrame> {
        self.frames
            .iter()
            .filter(|f| Self::is_exception(f))
            .collect()
    }

    /// Total frames ingested.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

impl Default for ProtocolAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn req(tid: u16, fc: u8, data: &[u8]) -> ModbusFrame {
        ModbusFrame::new(tid, 1, fc, data.to_vec(), true, tid as u64 * 1000)
    }

    fn resp(tid: u16, fc: u8, data: &[u8]) -> ModbusFrame {
        ModbusFrame::new(tid, 1, fc, data.to_vec(), false, tid as u64 * 1000 + 100)
    }

    fn exc(tid: u16, fc: u8) -> ModbusFrame {
        // Exception response: fc | 0x80
        ModbusFrame::new(tid, 1, fc | 0x80, vec![0x02], false, 9999)
    }

    // ── ModbusFrame construction ────────────────────────────────────────────

    #[test]
    fn test_modbus_frame_new() {
        let f = req(1, 0x03, b"hello");
        assert_eq!(f.transaction_id, 1);
        assert_eq!(f.function_code, 0x03);
        assert!(f.is_request);
        assert_eq!(f.data, b"hello");
    }

    #[test]
    fn test_modbus_frame_response() {
        let f = resp(2, 0x03, &[0x04, 0x00, 0x0A]);
        assert!(!f.is_request);
    }

    // ── ingest + analyze ────────────────────────────────────────────────────

    #[test]
    fn test_analyze_empty_buffer() {
        let analyzer = ProtocolAnalyzer::new();
        let s = analyzer.analyze();
        assert_eq!(s.total_frames, 0);
        assert_eq!(s.request_count, 0);
        assert_eq!(s.response_count, 0);
        assert_eq!(s.error_count, 0);
        assert_eq!(s.avg_data_bytes, 0.0);
        assert!(s.function_histogram.is_empty());
    }

    #[test]
    fn test_analyze_counts_total_frames() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(1, 0x03, &[0x00, 0x01, 0x00, 0x0A]));
        a.ingest(resp(1, 0x03, &[0x14, 0x00, 0x01]));
        let s = a.analyze();
        assert_eq!(s.total_frames, 2);
    }

    #[test]
    fn test_analyze_request_response_counts() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(1, 0x01, &[0x00, 0x00, 0x00, 0x08]));
        a.ingest(resp(1, 0x01, &[0x01, 0xFF]));
        let s = a.analyze();
        assert_eq!(s.request_count, 1);
        assert_eq!(s.response_count, 1);
    }

    #[test]
    fn test_analyze_error_count() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(exc(1, 0x03));
        a.ingest(exc(2, 0x06));
        a.ingest(req(3, 0x03, &[]));
        let s = a.analyze();
        assert_eq!(s.error_count, 2);
    }

    #[test]
    fn test_analyze_avg_data_bytes() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(1, 0x03, &[0x00, 0x01])); // 2 bytes
        a.ingest(resp(1, 0x03, &[0x04, 0x00, 0x01, 0x00, 0x02])); // 5 bytes
        let s = a.analyze();
        // avg = (2 + 5) / 2 = 3.5
        assert!((s.avg_data_bytes - 3.5).abs() < 1e-9);
    }

    #[test]
    fn test_analyze_function_histogram() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(1, 0x03, &[]));
        a.ingest(req(2, 0x03, &[]));
        a.ingest(req(3, 0x06, &[]));
        let s = a.analyze();
        assert_eq!(s.function_histogram[&0x03], 2);
        assert_eq!(s.function_histogram[&0x06], 1);
    }

    // ── function_info ──────────────────────────────────────────────────────

    #[test]
    fn test_function_info_fc01() {
        let info = ProtocolAnalyzer::function_info(0x01);
        assert_eq!(info.code, 0x01);
        assert_eq!(info.name, "Read Coil Status");
        assert_eq!(info.category, FunctionCategory::Read);
    }

    #[test]
    fn test_function_info_fc02() {
        let info = ProtocolAnalyzer::function_info(0x02);
        assert_eq!(info.category, FunctionCategory::Read);
        assert_eq!(info.name, "Read Discrete Inputs");
    }

    #[test]
    fn test_function_info_fc03() {
        let info = ProtocolAnalyzer::function_info(0x03);
        assert_eq!(info.category, FunctionCategory::Read);
        assert_eq!(info.name, "Read Holding Registers");
    }

    #[test]
    fn test_function_info_fc04() {
        let info = ProtocolAnalyzer::function_info(0x04);
        assert_eq!(info.category, FunctionCategory::Read);
    }

    #[test]
    fn test_function_info_fc05() {
        let info = ProtocolAnalyzer::function_info(0x05);
        assert_eq!(info.category, FunctionCategory::Write);
        assert_eq!(info.name, "Write Single Coil");
    }

    #[test]
    fn test_function_info_fc06() {
        let info = ProtocolAnalyzer::function_info(0x06);
        assert_eq!(info.category, FunctionCategory::Write);
        assert_eq!(info.name, "Write Single Register");
    }

    #[test]
    fn test_function_info_fc08_diagnostic() {
        let info = ProtocolAnalyzer::function_info(0x08);
        assert_eq!(info.category, FunctionCategory::Diagnostic);
        assert_eq!(info.name, "Diagnostics");
    }

    #[test]
    fn test_function_info_fc15() {
        let info = ProtocolAnalyzer::function_info(0x0F);
        assert_eq!(info.category, FunctionCategory::Write);
        assert_eq!(info.name, "Write Multiple Coils");
    }

    #[test]
    fn test_function_info_fc16() {
        let info = ProtocolAnalyzer::function_info(0x10);
        assert_eq!(info.category, FunctionCategory::Write);
        assert_eq!(info.name, "Write Multiple Registers");
    }

    #[test]
    fn test_function_info_unknown() {
        let info = ProtocolAnalyzer::function_info(0xFF);
        assert_eq!(info.name, "Unknown");
        assert_eq!(info.category, FunctionCategory::Other);
    }

    #[test]
    fn test_function_info_fc07_unknown() {
        // FC 07 is not in our known list.
        let info = ProtocolAnalyzer::function_info(0x07);
        assert_eq!(info.name, "Unknown");
    }

    // ── is_exception ───────────────────────────────────────────────────────

    #[test]
    fn test_is_exception_true_for_0x83() {
        let f = exc(1, 0x03);
        assert!(ProtocolAnalyzer::is_exception(&f));
    }

    #[test]
    fn test_is_exception_false_for_normal_fc() {
        let f = req(1, 0x03, &[]);
        assert!(!ProtocolAnalyzer::is_exception(&f));
    }

    #[test]
    fn test_is_exception_boundary_0x80() {
        let f = ModbusFrame::new(1, 1, 0x80, vec![], false, 0);
        assert!(ProtocolAnalyzer::is_exception(&f));
    }

    #[test]
    fn test_is_exception_0x7f_not_exception() {
        let f = ModbusFrame::new(1, 1, 0x7F, vec![], false, 0);
        assert!(!ProtocolAnalyzer::is_exception(&f));
    }

    // ── find_request_response_pairs ────────────────────────────────────────

    #[test]
    fn test_pairs_matches_by_transaction_id() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(10, 0x03, &[0x00, 0x01]));
        a.ingest(resp(10, 0x03, &[0x02, 0x00, 0x01]));
        let pairs = a.find_request_response_pairs();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0.transaction_id, 10);
        assert_eq!(pairs[0].1.transaction_id, 10);
    }

    #[test]
    fn test_pairs_empty_when_no_match() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(1, 0x03, &[]));
        a.ingest(resp(2, 0x03, &[])); // different tid
        let pairs = a.find_request_response_pairs();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_pairs_multiple_matched() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(1, 0x01, &[]));
        a.ingest(resp(1, 0x01, &[]));
        a.ingest(req(2, 0x06, &[]));
        a.ingest(resp(2, 0x06, &[]));
        let pairs = a.find_request_response_pairs();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn test_pairs_request_only_not_included() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(5, 0x03, &[]));
        let pairs = a.find_request_response_pairs();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_pairs_response_only_not_included() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(resp(5, 0x03, &[]));
        let pairs = a.find_request_response_pairs();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_pairs_sorted_by_transaction_id() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(3, 0x03, &[]));
        a.ingest(resp(3, 0x03, &[]));
        a.ingest(req(1, 0x06, &[]));
        a.ingest(resp(1, 0x06, &[]));
        a.ingest(req(2, 0x01, &[]));
        a.ingest(resp(2, 0x01, &[]));
        let pairs = a.find_request_response_pairs();
        let tids: Vec<u16> = pairs.iter().map(|(r, _)| r.transaction_id).collect();
        assert_eq!(tids, vec![1, 2, 3]);
    }

    // ── error_frames ───────────────────────────────────────────────────────

    #[test]
    fn test_error_frames_empty() {
        let a = ProtocolAnalyzer::new();
        assert!(a.error_frames().is_empty());
    }

    #[test]
    fn test_error_frames_returns_exceptions() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(1, 0x03, &[]));
        a.ingest(exc(2, 0x03));
        let ef = a.error_frames();
        assert_eq!(ef.len(), 1);
        assert_eq!(ef[0].transaction_id, 2);
    }

    #[test]
    fn test_error_frames_multiple() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(exc(1, 0x01));
        a.ingest(exc(2, 0x03));
        a.ingest(req(3, 0x06, &[]));
        assert_eq!(a.error_frames().len(), 2);
    }

    // ── frame_count ────────────────────────────────────────────────────────

    #[test]
    fn test_frame_count_increments_on_ingest() {
        let mut a = ProtocolAnalyzer::new();
        assert_eq!(a.frame_count(), 0);
        a.ingest(req(1, 0x01, &[]));
        assert_eq!(a.frame_count(), 1);
        a.ingest(resp(1, 0x01, &[]));
        assert_eq!(a.frame_count(), 2);
    }

    // ── FunctionCategory Display ───────────────────────────────────────────

    #[test]
    fn test_function_category_display() {
        assert_eq!(FunctionCategory::Read.to_string(), "Read");
        assert_eq!(FunctionCategory::Write.to_string(), "Write");
        assert_eq!(FunctionCategory::Diagnostic.to_string(), "Diagnostic");
        assert_eq!(FunctionCategory::Other.to_string(), "Other");
    }

    // ── AnalyzerError Display ──────────────────────────────────────────────

    #[test]
    fn test_analyzer_error_display() {
        let e = AnalyzerError::InvalidFrame("bad length".to_string());
        assert!(e.to_string().contains("bad length"));
    }

    // ── histogram edge case: exception codes in histogram ──────────────────

    #[test]
    fn test_histogram_includes_exception_codes() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(exc(1, 0x03)); // FC = 0x83
        let s = a.analyze();
        assert!(s.function_histogram.contains_key(&0x83));
    }

    // ── default impl ───────────────────────────────────────────────────────

    #[test]
    fn test_default_creates_empty_analyzer() {
        let a = ProtocolAnalyzer::default();
        assert_eq!(a.frame_count(), 0);
    }

    // ── avg_data_bytes with zero-byte frames ──────────────────────────────

    #[test]
    fn test_analyze_avg_data_zero_byte_frames() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(1, 0x03, &[]));
        a.ingest(resp(1, 0x03, &[]));
        let s = a.analyze();
        assert_eq!(s.avg_data_bytes, 0.0);
    }

    // ── multiple FC types in histogram ────────────────────────────────────

    #[test]
    fn test_histogram_multiple_function_codes() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(1, 0x01, &[]));
        a.ingest(req(2, 0x02, &[]));
        a.ingest(req(3, 0x03, &[]));
        a.ingest(req(4, 0x04, &[]));
        let s = a.analyze();
        assert_eq!(s.function_histogram.len(), 4);
    }

    // ── pairs: duplicate request/response shares same pair ────────────────

    #[test]
    fn test_pairs_duplicate_requests_uses_first() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(5, 0x06, &[0x00, 0x01, 0x00, 0xFF]));
        a.ingest(req(5, 0x06, &[0x00, 0x02, 0x00, 0xFF])); // duplicate tid
        a.ingest(resp(5, 0x06, &[0x00, 0x01, 0x00, 0xFF]));
        let pairs = a.find_request_response_pairs();
        // Only one pair despite two requests.
        assert_eq!(pairs.len(), 1);
    }

    // ── function_info code field matches input ────────────────────────────

    #[test]
    fn test_function_info_code_field_matches() {
        for fc in [0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x08, 0x0F, 0x10] {
            let info = ProtocolAnalyzer::function_info(fc);
            assert_eq!(info.code, fc);
        }
    }

    // ── exception frame has non-zero data ─────────────────────────────────

    #[test]
    fn test_exception_frame_contains_exception_code() {
        let f = exc(1, 0x03);
        // Exception byte in data should be 0x02 (Illegal Data Address).
        assert!(!f.data.is_empty());
        assert_eq!(f.data[0], 0x02);
    }

    // ── analyze response_count includes exceptions ────────────────────────

    #[test]
    fn test_analyze_exception_counted_as_response() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(req(1, 0x03, &[]));
        a.ingest(exc(1, 0x03)); // exception is a response
        let s = a.analyze();
        assert_eq!(s.request_count, 1);
        assert_eq!(s.response_count, 1);
        assert_eq!(s.error_count, 1);
    }

    // ── error_frames references are stable ───────────────────────────────

    #[test]
    fn test_error_frames_references_unit_ids() {
        let mut a = ProtocolAnalyzer::new();
        let mut ef1 = exc(10, 0x01);
        ef1.unit_id = 5;
        a.ingest(ef1);
        let frames = a.error_frames();
        assert_eq!(frames[0].unit_id, 5);
    }

    // ── function categories are correct for all known codes ───────────────

    #[test]
    fn test_read_codes_category() {
        for fc in [0x01u8, 0x02, 0x03, 0x04] {
            assert_eq!(
                ProtocolAnalyzer::function_info(fc).category,
                FunctionCategory::Read
            );
        }
    }

    #[test]
    fn test_write_codes_category() {
        for fc in [0x05u8, 0x06, 0x0F, 0x10] {
            assert_eq!(
                ProtocolAnalyzer::function_info(fc).category,
                FunctionCategory::Write
            );
        }
    }

    // ── timestamp ordering preserved via error_frames ────────────────────

    #[test]
    fn test_frames_total_count_reflects_ingest_order() {
        let mut a = ProtocolAnalyzer::new();
        a.ingest(ModbusFrame::new(1, 1, 0x03, vec![], true, 100));
        a.ingest(ModbusFrame::new(2, 1, 0x03, vec![], true, 200));
        a.ingest(ModbusFrame::new(3, 1, 0x03, vec![], true, 50));
        // Three frames were ingested regardless of timestamps.
        assert_eq!(a.frame_count(), 3);
        // All are requests, no exceptions.
        let s = a.analyze();
        assert_eq!(s.total_frames, 3);
        assert_eq!(s.request_count, 3);
        assert_eq!(s.error_count, 0);
    }
}
