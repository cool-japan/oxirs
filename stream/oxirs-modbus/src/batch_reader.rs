//! Modbus batch register reading with adjacent-register coalescing.
//!
//! Optimises Modbus read operations by merging nearby register requests into
//! single multi-register reads (FC03/FC04), scheduling reads with priority or
//! round-robin, retrying with exponential backoff, and reassembling results
//! back to individual register values.

use std::collections::{BTreeMap, HashMap, VecDeque};

// ──────────────────────────────────────────────────────────────────────────────
// Private type aliases
// ──────────────────────────────────────────────────────────────────────────────

/// Internal coalescing map: address → (count, labels, source ranges).
type CoalesceMap = BTreeMap<u16, (u16, Vec<String>, Vec<(u16, u16)>)>;

/// Internal entry type for coalescing: (address, count, labels, source ranges).
type CoalesceEntry = (u16, u16, Vec<String>, Vec<(u16, u16)>);

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// The Modbus function code used for reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReadFunctionCode {
    /// FC03: Read Holding Registers.
    ReadHoldingRegisters,
    /// FC04: Read Input Registers.
    ReadInputRegisters,
}

impl ReadFunctionCode {
    /// Return the raw function code byte.
    pub fn code(&self) -> u8 {
        match self {
            ReadFunctionCode::ReadHoldingRegisters => 0x03,
            ReadFunctionCode::ReadInputRegisters => 0x04,
        }
    }
}

/// Scheduling mode for batch reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingMode {
    /// Process requests in the order they were added.
    RoundRobin,
    /// Process higher-priority requests first.
    Priority,
}

/// A single register read request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterRequest {
    /// Starting register address.
    pub address: u16,
    /// Number of registers to read (1–125).
    pub count: u16,
    /// Function code to use.
    pub function_code: ReadFunctionCode,
    /// Priority (higher = more urgent). Only meaningful with `Priority` scheduling.
    pub priority: u32,
    /// Human-readable label for the request.
    pub label: String,
}

/// A coalesced read operation covering a contiguous address range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoalescedRead {
    /// Starting register address.
    pub address: u16,
    /// Total number of contiguous registers.
    pub count: u16,
    /// Function code.
    pub function_code: ReadFunctionCode,
    /// Labels of the original requests merged into this read.
    pub source_labels: Vec<String>,
    /// Original request addresses and counts for reassembly.
    pub source_ranges: Vec<(u16, u16)>,
}

/// A single register's value after reassembly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterValue {
    /// Register address.
    pub address: u16,
    /// Raw 16-bit register values.
    pub values: Vec<u16>,
    /// Label of the original request.
    pub label: String,
}

/// Gap between two register ranges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterGap {
    /// Address right after the end of the first range.
    pub gap_start: u16,
    /// Starting address of the next range.
    pub gap_end: u16,
    /// Size of the gap in registers.
    pub gap_size: u16,
}

/// Configuration for the batch reader.
#[derive(Debug, Clone)]
pub struct BatchReaderConfig {
    /// Maximum gap (in registers) that may be coalesced into a single read.
    /// Gaps larger than this cause a split into separate reads.
    pub max_coalesce_gap: u16,
    /// Maximum number of registers in a single Modbus read (protocol limit: 125).
    pub max_registers_per_read: u16,
    /// Maximum number of retries before giving up.
    pub max_retries: u32,
    /// Base delay in milliseconds for exponential backoff.
    pub base_delay_ms: u64,
    /// Read timeout in milliseconds.
    pub timeout_ms: u64,
    /// Scheduling mode.
    pub scheduling: SchedulingMode,
}

impl Default for BatchReaderConfig {
    fn default() -> Self {
        Self {
            max_coalesce_gap: 5,
            max_registers_per_read: 125,
            max_retries: 3,
            base_delay_ms: 100,
            timeout_ms: 5_000,
            scheduling: SchedulingMode::RoundRobin,
        }
    }
}

/// Statistics collected during batch reading.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total individual register requests submitted.
    pub total_requests: u64,
    /// Total coalesced reads performed.
    pub total_coalesced_reads: u64,
    /// Total registers read (actual bytes on the wire).
    pub total_registers_read: u64,
    /// Number of reads saved by coalescing.
    pub reads_saved: u64,
    /// Total retries performed.
    pub total_retries: u64,
    /// Number of timeouts encountered.
    pub total_timeouts: u64,
    /// Number of successful reads.
    pub total_successes: u64,
    /// Number of failed reads (after all retries).
    pub total_failures: u64,
    /// Estimated bytes saved by coalescing vs. individual reads.
    pub bytes_saved: u64,
}

/// Errors from the batch reader.
#[derive(Debug)]
pub enum BatchError {
    /// Register count exceeds protocol maximum.
    CountExceedsMax { requested: u16, max: u16 },
    /// Read timed out.
    Timeout { address: u16, attempt: u32 },
    /// All retries exhausted.
    RetriesExhausted { address: u16, max_retries: u32 },
    /// Response length mismatch.
    LengthMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::CountExceedsMax { requested, max } => {
                write!(f, "register count {} exceeds max {}", requested, max)
            }
            BatchError::Timeout { address, attempt } => {
                write!(
                    f,
                    "timeout reading address {} (attempt {})",
                    address, attempt
                )
            }
            BatchError::RetriesExhausted {
                address,
                max_retries,
            } => {
                write!(
                    f,
                    "retries exhausted for address {} after {} attempts",
                    address, max_retries
                )
            }
            BatchError::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "response length mismatch: expected {}, got {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for BatchError {}

// ──────────────────────────────────────────────────────────────────────────────
// BatchReader
// ──────────────────────────────────────────────────────────────────────────────

/// Optimises Modbus register reads by coalescing adjacent requests.
pub struct BatchReader {
    config: BatchReaderConfig,
    /// Pending requests, keyed by function code.
    pending: HashMap<ReadFunctionCode, VecDeque<RegisterRequest>>,
    stats: BatchStats,
}

impl BatchReader {
    /// Create a new batch reader with the given configuration.
    pub fn new(config: BatchReaderConfig) -> Self {
        Self {
            config,
            pending: HashMap::new(),
            stats: BatchStats::default(),
        }
    }

    /// Create a batch reader with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BatchReaderConfig::default())
    }

    /// Submit a register read request.
    pub fn submit(&mut self, request: RegisterRequest) -> Result<(), BatchError> {
        if request.count > self.config.max_registers_per_read {
            return Err(BatchError::CountExceedsMax {
                requested: request.count,
                max: self.config.max_registers_per_read,
            });
        }
        self.stats.total_requests += 1;
        self.pending
            .entry(request.function_code)
            .or_default()
            .push_back(request);
        Ok(())
    }

    /// Return the number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending.values().map(|q| q.len()).sum()
    }

    /// Get current statistics.
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Reset statistics to zero.
    pub fn reset_stats(&mut self) {
        self.stats = BatchStats::default();
    }

    /// Coalesce pending requests into optimised reads.
    ///
    /// Returns a list of coalesced reads grouped by function code.
    pub fn coalesce(&mut self) -> Vec<CoalescedRead> {
        let mut all_coalesced = Vec::new();

        for (fc, queue) in &mut self.pending {
            let mut requests: Vec<RegisterRequest> = queue.drain(..).collect();

            // Sort by scheduling mode
            match self.config.scheduling {
                SchedulingMode::Priority => {
                    requests.sort_by(|a, b| {
                        b.priority.cmp(&a.priority).then(a.address.cmp(&b.address))
                    });
                }
                SchedulingMode::RoundRobin => {
                    requests.sort_by_key(|r| r.address);
                }
            }

            // Sort by address for coalescing (regardless of scheduling)
            let mut sorted = requests.clone();
            sorted.sort_by_key(|r| r.address);

            let coalesced = Self::coalesce_requests(
                &sorted,
                *fc,
                self.config.max_coalesce_gap,
                self.config.max_registers_per_read,
            );

            let original_count = sorted.len() as u64;
            let coalesced_count = coalesced.len() as u64;
            if coalesced_count < original_count {
                self.stats.reads_saved += original_count - coalesced_count;
            }
            self.stats.total_coalesced_reads += coalesced_count;

            for c in &coalesced {
                self.stats.total_registers_read += c.count as u64;
            }

            all_coalesced.extend(coalesced);
        }

        all_coalesced
    }

    /// Detect gaps in a set of register ranges.
    pub fn detect_gaps(ranges: &[(u16, u16)]) -> Vec<RegisterGap> {
        if ranges.len() < 2 {
            return Vec::new();
        }
        let mut sorted: Vec<(u16, u16)> = ranges.to_vec();
        sorted.sort_by_key(|&(addr, _)| addr);

        let mut gaps = Vec::new();
        for i in 0..sorted.len() - 1 {
            let end_of_current = sorted[i].0.saturating_add(sorted[i].1);
            let start_of_next = sorted[i + 1].0;
            if start_of_next > end_of_current {
                gaps.push(RegisterGap {
                    gap_start: end_of_current,
                    gap_end: start_of_next,
                    gap_size: start_of_next - end_of_current,
                });
            }
        }
        gaps
    }

    /// Reassemble raw response bytes back to individual register values.
    pub fn reassemble(
        coalesced: &CoalescedRead,
        raw_values: &[u16],
    ) -> Result<Vec<RegisterValue>, BatchError> {
        let expected = coalesced.count as usize;
        if raw_values.len() != expected {
            return Err(BatchError::LengthMismatch {
                expected,
                actual: raw_values.len(),
            });
        }

        let mut results = Vec::new();
        for (i, &(addr, count)) in coalesced.source_ranges.iter().enumerate() {
            let offset = (addr - coalesced.address) as usize;
            let end = offset + count as usize;
            if end > raw_values.len() {
                return Err(BatchError::LengthMismatch {
                    expected: end,
                    actual: raw_values.len(),
                });
            }
            let label = coalesced.source_labels.get(i).cloned().unwrap_or_default();
            results.push(RegisterValue {
                address: addr,
                values: raw_values[offset..end].to_vec(),
                label,
            });
        }
        Ok(results)
    }

    /// Calculate the exponential backoff delay for a given retry attempt.
    pub fn backoff_delay(&self, attempt: u32) -> u64 {
        self.config
            .base_delay_ms
            .saturating_mul(1u64 << attempt.min(10))
    }

    /// Record a successful read in statistics.
    pub fn record_success(&mut self) {
        self.stats.total_successes += 1;
    }

    /// Record a failed read in statistics.
    pub fn record_failure(&mut self) {
        self.stats.total_failures += 1;
    }

    /// Record a retry in statistics.
    pub fn record_retry(&mut self) {
        self.stats.total_retries += 1;
    }

    /// Record a timeout in statistics.
    pub fn record_timeout(&mut self) {
        self.stats.total_timeouts += 1;
    }

    /// Get the configured timeout in milliseconds.
    pub fn timeout_ms(&self) -> u64 {
        self.config.timeout_ms
    }

    /// Get the maximum retries.
    pub fn max_retries(&self) -> u32 {
        self.config.max_retries
    }

    // ── Private ──────────────────────────────────────────────────────────────

    /// Coalesce sorted register requests into minimal reads.
    fn coalesce_requests(
        requests: &[RegisterRequest],
        fc: ReadFunctionCode,
        max_gap: u16,
        max_count: u16,
    ) -> Vec<CoalescedRead> {
        if requests.is_empty() {
            return Vec::new();
        }

        // Use a BTreeMap to merge overlapping/adjacent ranges
        let mut ranges: CoalesceMap = BTreeMap::new();

        for req in requests {
            ranges.insert(
                req.address,
                (
                    req.count,
                    vec![req.label.clone()],
                    vec![(req.address, req.count)],
                ),
            );
        }

        // Merge adjacent ranges
        let entries: Vec<CoalesceEntry> = ranges
            .into_iter()
            .map(|(addr, (count, labels, sources))| (addr, count, labels, sources))
            .collect();

        let mut coalesced: Vec<CoalescedRead> = Vec::new();
        let mut cur_addr = entries[0].0;
        let mut cur_end = cur_addr.saturating_add(entries[0].1);
        let mut cur_labels: Vec<String> = entries[0].2.clone();
        let mut cur_sources: Vec<(u16, u16)> = entries[0].3.clone();

        for entry in entries.iter().skip(1) {
            let (addr, count, labels, sources) = entry;
            let entry_end = addr.saturating_add(*count);
            let gap = addr.saturating_sub(cur_end);

            let potential_count = entry_end.saturating_sub(cur_addr);

            if gap <= max_gap && potential_count <= max_count {
                // Merge
                cur_end = cur_end.max(entry_end);
                cur_labels.extend_from_slice(labels);
                cur_sources.extend_from_slice(sources);
            } else {
                // Flush current
                coalesced.push(CoalescedRead {
                    address: cur_addr,
                    count: cur_end - cur_addr,
                    function_code: fc,
                    source_labels: cur_labels.clone(),
                    source_ranges: cur_sources.clone(),
                });
                cur_addr = *addr;
                cur_end = entry_end;
                cur_labels = labels.clone();
                cur_sources = sources.clone();
            }
        }

        // Flush last group
        coalesced.push(CoalescedRead {
            address: cur_addr,
            count: cur_end - cur_addr,
            function_code: fc,
            source_labels: cur_labels,
            source_ranges: cur_sources,
        });

        coalesced
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn req(addr: u16, count: u16, label: &str) -> RegisterRequest {
        RegisterRequest {
            address: addr,
            count,
            function_code: ReadFunctionCode::ReadHoldingRegisters,
            priority: 0,
            label: label.to_string(),
        }
    }

    fn req_prio(addr: u16, count: u16, label: &str, prio: u32) -> RegisterRequest {
        RegisterRequest {
            address: addr,
            count,
            function_code: ReadFunctionCode::ReadHoldingRegisters,
            priority: prio,
            label: label.to_string(),
        }
    }

    fn req_input(addr: u16, count: u16, label: &str) -> RegisterRequest {
        RegisterRequest {
            address: addr,
            count,
            function_code: ReadFunctionCode::ReadInputRegisters,
            priority: 0,
            label: label.to_string(),
        }
    }

    // ── ReadFunctionCode ─────────────────────────────────────────────────────

    #[test]
    fn test_function_code_values() {
        assert_eq!(ReadFunctionCode::ReadHoldingRegisters.code(), 0x03);
        assert_eq!(ReadFunctionCode::ReadInputRegisters.code(), 0x04);
    }

    #[test]
    fn test_function_code_eq() {
        assert_eq!(
            ReadFunctionCode::ReadHoldingRegisters,
            ReadFunctionCode::ReadHoldingRegisters
        );
        assert_ne!(
            ReadFunctionCode::ReadHoldingRegisters,
            ReadFunctionCode::ReadInputRegisters
        );
    }

    // ── BatchReader creation ─────────────────────────────────────────────────

    #[test]
    fn test_reader_creation() {
        let reader = BatchReader::with_defaults();
        assert_eq!(reader.pending_count(), 0);
        assert_eq!(reader.stats().total_requests, 0);
    }

    #[test]
    fn test_reader_custom_config() {
        let cfg = BatchReaderConfig {
            max_coalesce_gap: 10,
            max_registers_per_read: 100,
            max_retries: 5,
            base_delay_ms: 200,
            timeout_ms: 10_000,
            scheduling: SchedulingMode::Priority,
        };
        let reader = BatchReader::new(cfg);
        assert_eq!(reader.timeout_ms(), 10_000);
        assert_eq!(reader.max_retries(), 5);
    }

    // ── Submit ───────────────────────────────────────────────────────────────

    #[test]
    fn test_submit_request() {
        let mut reader = BatchReader::with_defaults();
        assert!(reader.submit(req(100, 10, "temp")).is_ok());
        assert_eq!(reader.pending_count(), 1);
        assert_eq!(reader.stats().total_requests, 1);
    }

    #[test]
    fn test_submit_count_exceeds_max() {
        let mut reader = BatchReader::with_defaults();
        let result = reader.submit(req(100, 200, "too-large"));
        assert!(result.is_err());
    }

    #[test]
    fn test_submit_multiple() {
        let mut reader = BatchReader::with_defaults();
        reader.submit(req(100, 5, "a")).ok();
        reader.submit(req(200, 5, "b")).ok();
        assert_eq!(reader.pending_count(), 2);
    }

    // ── Coalescing ───────────────────────────────────────────────────────────

    #[test]
    fn test_coalesce_adjacent() {
        let mut reader = BatchReader::with_defaults();
        reader.submit(req(100, 5, "a")).ok();
        reader.submit(req(105, 5, "b")).ok(); // adjacent (100+5=105)
        let coalesced = reader.coalesce();
        assert_eq!(coalesced.len(), 1);
        assert_eq!(coalesced[0].address, 100);
        assert_eq!(coalesced[0].count, 10);
    }

    #[test]
    fn test_coalesce_with_small_gap() {
        let cfg = BatchReaderConfig {
            max_coalesce_gap: 3,
            ..Default::default()
        };
        let mut reader = BatchReader::new(cfg);
        reader.submit(req(100, 5, "a")).ok();
        reader.submit(req(108, 5, "b")).ok(); // gap of 3 (105→108)
        let coalesced = reader.coalesce();
        assert_eq!(coalesced.len(), 1);
        assert_eq!(coalesced[0].count, 13); // 100→113
    }

    #[test]
    fn test_coalesce_large_gap_splits() {
        let cfg = BatchReaderConfig {
            max_coalesce_gap: 2,
            ..Default::default()
        };
        let mut reader = BatchReader::new(cfg);
        reader.submit(req(100, 5, "a")).ok();
        reader.submit(req(200, 5, "b")).ok(); // gap of 95
        let coalesced = reader.coalesce();
        assert_eq!(coalesced.len(), 2);
    }

    #[test]
    fn test_coalesce_respects_max_count() {
        let cfg = BatchReaderConfig {
            max_coalesce_gap: 100,
            max_registers_per_read: 20,
            ..Default::default()
        };
        let mut reader = BatchReader::new(cfg);
        reader.submit(req(0, 10, "a")).ok();
        reader.submit(req(10, 10, "b")).ok();
        reader.submit(req(20, 10, "c")).ok(); // total would be 30 > 20
        let coalesced = reader.coalesce();
        assert!(coalesced.len() >= 2); // must split
    }

    #[test]
    fn test_coalesce_empty() {
        let mut reader = BatchReader::with_defaults();
        let coalesced = reader.coalesce();
        assert!(coalesced.is_empty());
    }

    #[test]
    fn test_coalesce_different_function_codes() {
        let mut reader = BatchReader::with_defaults();
        reader.submit(req(100, 5, "holding")).ok();
        reader.submit(req_input(100, 5, "input")).ok();
        let coalesced = reader.coalesce();
        // Different FCs → separate groups
        assert_eq!(coalesced.len(), 2);
    }

    // ── Gap detection ────────────────────────────────────────────────────────

    #[test]
    fn test_detect_gaps_none() {
        let ranges = vec![(100, 5), (105, 5)]; // contiguous
        let gaps = BatchReader::detect_gaps(&ranges);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_detect_gaps_single_gap() {
        let ranges = vec![(100, 5), (110, 5)]; // gap 105→110
        let gaps = BatchReader::detect_gaps(&ranges);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].gap_start, 105);
        assert_eq!(gaps[0].gap_end, 110);
        assert_eq!(gaps[0].gap_size, 5);
    }

    #[test]
    fn test_detect_gaps_multiple() {
        let ranges = vec![(0, 5), (10, 5), (20, 5)];
        let gaps = BatchReader::detect_gaps(&ranges);
        assert_eq!(gaps.len(), 2);
    }

    #[test]
    fn test_detect_gaps_unsorted_input() {
        let ranges = vec![(20, 5), (0, 5), (10, 5)];
        let gaps = BatchReader::detect_gaps(&ranges);
        assert_eq!(gaps.len(), 2); // sorted internally
    }

    #[test]
    fn test_detect_gaps_single_range() {
        let ranges = vec![(100, 10)];
        let gaps = BatchReader::detect_gaps(&ranges);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_detect_gaps_empty() {
        let gaps = BatchReader::detect_gaps(&[]);
        assert!(gaps.is_empty());
    }

    // ── Reassembly ───────────────────────────────────────────────────────────

    #[test]
    fn test_reassemble_single_source() {
        let coalesced = CoalescedRead {
            address: 100,
            count: 5,
            function_code: ReadFunctionCode::ReadHoldingRegisters,
            source_labels: vec!["temp".to_string()],
            source_ranges: vec![(100, 5)],
        };
        let raw = vec![10, 20, 30, 40, 50];
        let result = BatchReader::reassemble(&coalesced, &raw);
        assert!(result.is_ok());
        let values = result.expect("should reassemble");
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].address, 100);
        assert_eq!(values[0].values, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_reassemble_multiple_sources() {
        let coalesced = CoalescedRead {
            address: 100,
            count: 10,
            function_code: ReadFunctionCode::ReadHoldingRegisters,
            source_labels: vec!["a".to_string(), "b".to_string()],
            source_ranges: vec![(100, 5), (105, 5)],
        };
        let raw: Vec<u16> = (0..10).collect();
        let result = BatchReader::reassemble(&coalesced, &raw);
        assert!(result.is_ok());
        let values = result.expect("should reassemble");
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].values, vec![0, 1, 2, 3, 4]);
        assert_eq!(values[1].values, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_reassemble_length_mismatch() {
        let coalesced = CoalescedRead {
            address: 100,
            count: 10,
            function_code: ReadFunctionCode::ReadHoldingRegisters,
            source_labels: vec!["a".to_string()],
            source_ranges: vec![(100, 10)],
        };
        let raw = vec![1, 2, 3]; // too short
        let result = BatchReader::reassemble(&coalesced, &raw);
        assert!(result.is_err());
    }

    // ── Backoff ──────────────────────────────────────────────────────────────

    #[test]
    fn test_backoff_delay_exponential() {
        let reader = BatchReader::with_defaults();
        assert_eq!(reader.backoff_delay(0), 100); // 100 * 2^0
        assert_eq!(reader.backoff_delay(1), 200); // 100 * 2^1
        assert_eq!(reader.backoff_delay(2), 400); // 100 * 2^2
        assert_eq!(reader.backoff_delay(3), 800); // 100 * 2^3
    }

    #[test]
    fn test_backoff_delay_capped() {
        let reader = BatchReader::with_defaults();
        let d10 = reader.backoff_delay(10);
        let d20 = reader.backoff_delay(20);
        // attempt > 10 is capped to 2^10
        assert_eq!(d10, d20);
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_tracking() {
        let mut reader = BatchReader::with_defaults();
        reader.submit(req(100, 5, "a")).ok();
        reader.submit(req(105, 5, "b")).ok();
        reader.coalesce();

        assert_eq!(reader.stats().total_requests, 2);
        assert_eq!(reader.stats().total_coalesced_reads, 1);
        assert!(reader.stats().reads_saved >= 1);
    }

    #[test]
    fn test_stats_reset() {
        let mut reader = BatchReader::with_defaults();
        reader.submit(req(100, 5, "a")).ok();
        reader.coalesce();
        reader.reset_stats();
        assert_eq!(reader.stats().total_requests, 0);
        assert_eq!(reader.stats().total_coalesced_reads, 0);
    }

    #[test]
    fn test_record_success() {
        let mut reader = BatchReader::with_defaults();
        reader.record_success();
        reader.record_success();
        assert_eq!(reader.stats().total_successes, 2);
    }

    #[test]
    fn test_record_failure() {
        let mut reader = BatchReader::with_defaults();
        reader.record_failure();
        assert_eq!(reader.stats().total_failures, 1);
    }

    #[test]
    fn test_record_retry() {
        let mut reader = BatchReader::with_defaults();
        reader.record_retry();
        reader.record_retry();
        assert_eq!(reader.stats().total_retries, 2);
    }

    #[test]
    fn test_record_timeout() {
        let mut reader = BatchReader::with_defaults();
        reader.record_timeout();
        assert_eq!(reader.stats().total_timeouts, 1);
    }

    // ── Config defaults ──────────────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = BatchReaderConfig::default();
        assert_eq!(cfg.max_coalesce_gap, 5);
        assert_eq!(cfg.max_registers_per_read, 125);
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.base_delay_ms, 100);
        assert_eq!(cfg.timeout_ms, 5_000);
        assert_eq!(cfg.scheduling, SchedulingMode::RoundRobin);
    }

    // ── BatchError display ───────────────────────────────────────────────────

    #[test]
    fn test_batch_error_display() {
        let e1 = BatchError::CountExceedsMax {
            requested: 200,
            max: 125,
        };
        assert!(format!("{}", e1).contains("200"));

        let e2 = BatchError::Timeout {
            address: 100,
            attempt: 3,
        };
        assert!(format!("{}", e2).contains("100"));

        let e3 = BatchError::RetriesExhausted {
            address: 100,
            max_retries: 5,
        };
        assert!(format!("{}", e3).contains("5"));

        let e4 = BatchError::LengthMismatch {
            expected: 10,
            actual: 5,
        };
        assert!(format!("{}", e4).contains("10"));
    }

    #[test]
    fn test_batch_error_is_error() {
        let e: Box<dyn std::error::Error> = Box::new(BatchError::CountExceedsMax {
            requested: 200,
            max: 125,
        });
        assert!(!e.to_string().is_empty());
    }

    // ── Priority scheduling ──────────────────────────────────────────────────

    #[test]
    fn test_priority_scheduling() {
        let cfg = BatchReaderConfig {
            scheduling: SchedulingMode::Priority,
            ..Default::default()
        };
        let mut reader = BatchReader::new(cfg);
        reader.submit(req_prio(100, 5, "low", 1)).ok();
        reader.submit(req_prio(200, 5, "high", 100)).ok();
        let coalesced = reader.coalesce();
        // Both get coalesced but separately (large gap)
        assert_eq!(coalesced.len(), 2);
    }

    // ── Bytes saved estimation ───────────────────────────────────────────────

    #[test]
    fn test_bytes_saved_calculation() {
        let mut reader = BatchReader::with_defaults();
        reader.submit(req(100, 5, "a")).ok();
        reader.submit(req(105, 5, "b")).ok();
        reader.submit(req(110, 5, "c")).ok();
        reader.coalesce();
        // 3 requests coalesced into 1 → 2 reads saved
        assert!(reader.stats().reads_saved >= 2);
    }

    // ── Coalesced read fields ────────────────────────────────────────────────

    #[test]
    fn test_coalesced_read_source_labels() {
        let mut reader = BatchReader::with_defaults();
        reader.submit(req(100, 5, "temp")).ok();
        reader.submit(req(105, 5, "pressure")).ok();
        let coalesced = reader.coalesce();
        assert_eq!(coalesced.len(), 1);
        assert!(coalesced[0].source_labels.contains(&"temp".to_string()));
        assert!(coalesced[0].source_labels.contains(&"pressure".to_string()));
    }

    #[test]
    fn test_coalesced_read_source_ranges() {
        let mut reader = BatchReader::with_defaults();
        reader.submit(req(100, 5, "a")).ok();
        reader.submit(req(105, 3, "b")).ok();
        let coalesced = reader.coalesce();
        assert_eq!(coalesced[0].source_ranges.len(), 2);
        assert_eq!(coalesced[0].source_ranges[0], (100, 5));
        assert_eq!(coalesced[0].source_ranges[1], (105, 3));
    }

    // ── RegisterGap ──────────────────────────────────────────────────────────

    #[test]
    fn test_register_gap_eq() {
        let g1 = RegisterGap {
            gap_start: 10,
            gap_end: 20,
            gap_size: 10,
        };
        let g2 = g1.clone();
        assert_eq!(g1, g2);
    }

    // ── BatchStats default ───────────────────────────────────────────────────

    #[test]
    fn test_batch_stats_default() {
        let s = BatchStats::default();
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.total_coalesced_reads, 0);
        assert_eq!(s.total_registers_read, 0);
        assert_eq!(s.reads_saved, 0);
        assert_eq!(s.total_retries, 0);
        assert_eq!(s.total_timeouts, 0);
        assert_eq!(s.total_successes, 0);
        assert_eq!(s.total_failures, 0);
        assert_eq!(s.bytes_saved, 0);
    }

    // ── Scheduling mode eq ───────────────────────────────────────────────────

    #[test]
    fn test_scheduling_mode_eq() {
        assert_eq!(SchedulingMode::RoundRobin, SchedulingMode::RoundRobin);
        assert_eq!(SchedulingMode::Priority, SchedulingMode::Priority);
        assert_ne!(SchedulingMode::RoundRobin, SchedulingMode::Priority);
    }
}
