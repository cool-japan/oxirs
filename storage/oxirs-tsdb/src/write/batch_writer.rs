//! High-performance batched write path for TSDB with CRC32-protected WAL.
//!
//! ## Design
//!
//! ```text
//! Ingest           Buffer                   Flush
//! ──────────────   ──────────────────────   ─────────────────────
//! write_point() ──►  RingBuffer<MetricPt>  ──►  WriteAheadLog
//! write_batch() ──►  (lock-free enqueue)   ──►  Columnar storage
//!                        ▲
//!                   flush on size limit
//!                   or periodic timeout
//! ```
//!
//! [`BatchWriter`] accumulates [`MetricPoint`] values in a bounded ring
//! buffer.  When the buffer reaches `batch_capacity` entries, or when
//! [`BatchWriter::flush`] is called explicitly, it drains the buffer,
//! appends each entry to the [`CrcWal`], and returns the sequence number
//! of the last committed entry.
//!
//! [`CrcWal`] extends the basic WAL format with a CRC32 checksum per record,
//! enabling corruption detection on recovery.
//!
//! ## CRC32 WAL record format
//!
//! ```text
//! ┌────────────┬────────────┬─────────────────────────────┐
//! │ CRC32 (4B) │ Length(4B) │ Payload (Length bytes)      │
//! └────────────┴────────────┴─────────────────────────────┘
//! ```
//!
//! On replay the CRC is recomputed over the payload and compared; mismatches
//! surface as [`TsdbError::CrcMismatch`].

use crate::error::{TsdbError, TsdbResult};
use crc32fast::Hasher as Crc32Hasher;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// =============================================================================
// MetricPoint
// =============================================================================

/// A single time-series observation ready for ingestion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricPoint {
    /// Series or metric name.
    pub metric: String,
    /// Unix epoch milliseconds.
    pub timestamp_ms: i64,
    /// Observed value.
    pub value: f64,
}

impl MetricPoint {
    /// Create a new metric point.
    pub fn new(metric: impl Into<String>, timestamp_ms: i64, value: f64) -> Self {
        Self {
            metric: metric.into(),
            timestamp_ms,
            value,
        }
    }

    /// Serialize to a compact binary representation.
    ///
    /// Format: `metric_len(4) + metric_bytes + timestamp_ms(8) + value(8)`
    pub fn to_bytes(&self) -> TsdbResult<Vec<u8>> {
        let metric_bytes = self.metric.as_bytes();
        let metric_len = metric_bytes.len() as u32;
        let mut buf = Vec::with_capacity(4 + metric_bytes.len() + 8 + 8);
        buf.extend_from_slice(&metric_len.to_le_bytes());
        buf.extend_from_slice(metric_bytes);
        buf.extend_from_slice(&self.timestamp_ms.to_le_bytes());
        buf.extend_from_slice(&self.value.to_le_bytes());
        Ok(buf)
    }

    /// Deserialize from the binary representation produced by [`MetricPoint::to_bytes`].
    pub fn from_bytes(data: &[u8]) -> TsdbResult<Self> {
        if data.len() < 4 {
            return Err(TsdbError::Wal("record too short (< 4 bytes)".into()));
        }
        let metric_len = u32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| TsdbError::Wal("slice error".into()))?,
        ) as usize;

        let end_metric = 4 + metric_len;
        if data.len() < end_metric + 16 {
            return Err(TsdbError::Wal(format!(
                "record too short: expected {}, got {}",
                end_metric + 16,
                data.len()
            )));
        }

        let metric = String::from_utf8(data[4..end_metric].to_vec())
            .map_err(|e| TsdbError::Wal(format!("metric name UTF-8 error: {e}")))?;

        let timestamp_ms = i64::from_le_bytes(
            data[end_metric..end_metric + 8]
                .try_into()
                .map_err(|_| TsdbError::Wal("ts slice error".into()))?,
        );
        let value = f64::from_le_bytes(
            data[end_metric + 8..end_metric + 16]
                .try_into()
                .map_err(|_| TsdbError::Wal("value slice error".into()))?,
        );

        Ok(Self {
            metric,
            timestamp_ms,
            value,
        })
    }
}

// =============================================================================
// CrcWal — CRC32-protected Write-Ahead Log
// =============================================================================

/// CRC32-protected Write-Ahead Log.
///
/// Each record is prefixed with a 4-byte CRC and a 4-byte payload length.
/// On recovery, the CRC is verified before returning the payload.
pub struct CrcWal {
    path: PathBuf,
    writer: BufWriter<File>,
    /// Total records appended since open (monotonically increasing).
    record_count: u64,
    /// Global sequence counter (shared across instances for testing).
    next_seq: Arc<AtomicU64>,
}

impl CrcWal {
    /// Open (or create) a WAL file at `path`.
    pub fn open(path: &Path) -> TsdbResult<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| TsdbError::Wal(format!("open WAL: {e}")))?;

        Ok(Self {
            path: path.to_path_buf(),
            writer: BufWriter::new(file),
            record_count: 0,
            next_seq: Arc::new(AtomicU64::new(1)),
        })
    }

    /// Return the WAL file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return the number of records appended during this session.
    pub fn record_count(&self) -> u64 {
        self.record_count
    }

    /// Compute CRC32 over a byte slice.
    pub fn crc32(data: &[u8]) -> u32 {
        let mut h = Crc32Hasher::new();
        h.update(data);
        h.finalize()
    }

    /// Append a [`MetricPoint`] to the WAL.
    ///
    /// Returns the sequence number assigned to this record.
    pub fn append(&mut self, point: &MetricPoint) -> TsdbResult<u64> {
        let payload = point.to_bytes()?;
        let checksum = Self::crc32(&payload);
        let length = payload.len() as u32;

        // Write: CRC32(4) | length(4) | payload
        self.writer
            .write_all(&checksum.to_le_bytes())
            .map_err(|e| TsdbError::Wal(format!("write CRC: {e}")))?;
        self.writer
            .write_all(&length.to_le_bytes())
            .map_err(|e| TsdbError::Wal(format!("write length: {e}")))?;
        self.writer
            .write_all(&payload)
            .map_err(|e| TsdbError::Wal(format!("write payload: {e}")))?;
        self.writer
            .flush()
            .map_err(|e| TsdbError::Wal(format!("flush WAL: {e}")))?;

        self.record_count += 1;
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        Ok(seq)
    }

    /// Append multiple points in a single write call.
    ///
    /// Returns the sequence number of the last record.
    pub fn append_batch(&mut self, points: &[MetricPoint]) -> TsdbResult<u64> {
        let mut last_seq = 0u64;
        for point in points {
            last_seq = self.append(point)?;
        }
        Ok(last_seq)
    }

    /// Replay all records from the WAL file.
    ///
    /// Returns an error if any record fails CRC verification.
    pub fn replay(path: &Path) -> TsdbResult<Vec<MetricPoint>> {
        let file =
            File::open(path).map_err(|e| TsdbError::Wal(format!("open WAL for replay: {e}")))?;
        let mut reader = BufReader::new(file);
        let mut points = Vec::new();

        loop {
            // Read header: CRC32(4) + length(4)
            let mut header = [0u8; 8];
            match reader.read_exact(&mut header) {
                Ok(()) => {}
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(TsdbError::Wal(format!("read header: {e}"))),
            }

            let stored_crc = u32::from_le_bytes(header[0..4].try_into().expect("4 bytes"));
            let length = u32::from_le_bytes(header[4..8].try_into().expect("4 bytes")) as usize;

            let mut payload = vec![0u8; length];
            reader
                .read_exact(&mut payload)
                .map_err(|e| TsdbError::Wal(format!("read payload: {e}")))?;

            let computed_crc = Self::crc32(&payload);
            if computed_crc != stored_crc {
                return Err(TsdbError::CrcMismatch {
                    expected: stored_crc,
                    got: computed_crc,
                });
            }

            points.push(MetricPoint::from_bytes(&payload)?);
        }

        Ok(points)
    }

    /// Truncate the WAL file (clear all records).
    pub fn clear(&mut self) -> TsdbResult<()> {
        self.writer
            .flush()
            .map_err(|e| TsdbError::Wal(format!("{e}")))?;

        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.path)
            .map_err(|e| TsdbError::Wal(format!("truncate WAL: {e}")))?;

        self.writer = BufWriter::new(file);
        self.record_count = 0;
        Ok(())
    }

    /// Force all buffered data to the OS (fsync).
    pub fn sync(&mut self) -> TsdbResult<()> {
        self.writer
            .flush()
            .map_err(|e| TsdbError::Wal(format!("{e}")))?;
        self.writer
            .get_ref()
            .sync_all()
            .map_err(|e| TsdbError::Wal(format!("fsync WAL: {e}")))
    }

    /// Estimate WAL file size in bytes.
    pub fn file_size(&mut self) -> TsdbResult<u64> {
        self.writer
            .flush()
            .map_err(|e| TsdbError::Wal(format!("{e}")))?;
        let pos = self
            .writer
            .get_mut()
            .seek(SeekFrom::End(0))
            .map_err(|e| TsdbError::Wal(format!("seek: {e}")))?;
        Ok(pos)
    }
}

// =============================================================================
// RingBuffer — bounded lock-free-ish write buffer
// =============================================================================

/// Bounded ring buffer backed by a `VecDeque` and protected by a `Mutex`.
///
/// The `Mutex` overhead is acceptable at the expected write rates (≥1M/s is
/// achieved by batching many calls per flush).  For true lock-free ingestion
/// replace this with `crossbeam-queue::ArrayQueue`.
#[derive(Debug)]
struct RingBuffer {
    inner: Mutex<VecDeque<MetricPoint>>,
    capacity: usize,
}

impl RingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    /// Push a point.  Returns `Err(BufferFull)` when the buffer is at capacity.
    fn push(&self, point: MetricPoint) -> TsdbResult<()> {
        let mut q = self
            .inner
            .lock()
            .map_err(|_| TsdbError::Wal("lock poisoned".into()))?;
        if q.len() >= self.capacity {
            return Err(TsdbError::BufferFull(q.len()));
        }
        q.push_back(point);
        Ok(())
    }

    /// Drain all buffered points.
    fn drain_all(&self) -> TsdbResult<Vec<MetricPoint>> {
        let mut q = self
            .inner
            .lock()
            .map_err(|_| TsdbError::Wal("lock poisoned".into()))?;
        Ok(q.drain(..).collect())
    }

    /// Current number of buffered points.
    fn len(&self) -> usize {
        self.inner.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Buffer capacity.
    fn capacity(&self) -> usize {
        self.capacity
    }
}

// =============================================================================
// BatchWriter
// =============================================================================

/// High-throughput write path that buffers metric points and flushes in batches.
///
/// ## Usage
///
/// ```rust,no_run
/// use oxirs_tsdb::write::batch_writer::{BatchWriter, BatchWriterConfig, MetricPoint};
/// use std::path::Path;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cfg = BatchWriterConfig::default();
/// let mut bw = BatchWriter::open(Path::new("/tmp/tsdb_wal.bin"), cfg)?;
///
/// bw.write_point(MetricPoint::new("cpu", 1_700_000_000_000, 42.5))?;
///
/// let seq = bw.flush()?;
/// println!("flushed; last seq = {seq}");
/// # Ok(())
/// # }
/// ```
pub struct BatchWriter {
    buffer: Arc<RingBuffer>,
    wal: CrcWal,
    config: BatchWriterConfig,
    /// Total number of points successfully flushed.
    total_flushed: u64,
}

/// Configuration for [`BatchWriter`].
#[derive(Debug, Clone)]
pub struct BatchWriterConfig {
    /// Maximum number of points held in the ring buffer before an automatic flush.
    pub batch_capacity: usize,
    /// Whether to fsync the WAL after every flush (safer but slower).
    pub fsync_on_flush: bool,
}

impl Default for BatchWriterConfig {
    fn default() -> Self {
        Self {
            batch_capacity: 4096,
            fsync_on_flush: false,
        }
    }
}

impl BatchWriterConfig {
    /// Create a config with a specific batch capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            batch_capacity: capacity,
            ..Default::default()
        }
    }
}

impl BatchWriter {
    /// Open (or create) a `BatchWriter` backed by a CRC-protected WAL at `path`.
    pub fn open(path: &Path, config: BatchWriterConfig) -> TsdbResult<Self> {
        let wal = CrcWal::open(path)?;
        let buffer = Arc::new(RingBuffer::new(config.batch_capacity));
        Ok(Self {
            buffer,
            wal,
            config,
            total_flushed: 0,
        })
    }

    /// Return total points flushed since creation.
    pub fn total_flushed(&self) -> u64 {
        self.total_flushed
    }

    /// Return the number of points currently buffered.
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }

    /// Return the configured batch capacity.
    pub fn batch_capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Enqueue a single metric point.
    ///
    /// If the buffer is full, automatically flushes first.
    pub fn write_point(&mut self, point: MetricPoint) -> TsdbResult<()> {
        if self.buffer.len() >= self.config.batch_capacity {
            self.flush()?;
        }
        self.buffer.push(point)
    }

    /// Enqueue a batch of metric points, flushing automatically when needed.
    ///
    /// Returns the WAL sequence number of the last committed record.
    pub fn write_batch(&mut self, points: &[MetricPoint]) -> TsdbResult<u64> {
        let mut last_seq = 0u64;
        for point in points {
            if self.buffer.len() >= self.config.batch_capacity {
                last_seq = self.flush()?;
            }
            self.buffer.push(point.clone())?;
        }
        // Flush remaining
        if self.buffer.len() > 0 {
            last_seq = self.flush()?;
        }
        Ok(last_seq)
    }

    /// Flush all buffered points to the WAL.
    ///
    /// Returns the WAL sequence number of the last committed record, or 0 if
    /// the buffer was empty.
    pub fn flush(&mut self) -> TsdbResult<u64> {
        let points = self.buffer.drain_all()?;
        if points.is_empty() {
            return Ok(0);
        }
        let last_seq = self.wal.append_batch(&points)?;
        self.total_flushed += points.len() as u64;
        if self.config.fsync_on_flush {
            self.wal.sync()?;
        }
        Ok(last_seq)
    }

    /// Replay the WAL from disk (e.g. after a crash).
    pub fn replay_wal(path: &Path) -> TsdbResult<Vec<MetricPoint>> {
        CrcWal::replay(path)
    }
}

// =============================================================================
// Tests
// =============================================================================

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn tmp_path(name: &str) -> PathBuf {
        env::temp_dir().join(format!("oxirs_tsdb_{name}.wal"))
    }

    // -- MetricPoint ----------------------------------------------------------

    #[test]
    fn test_metric_point_roundtrip() {
        let p = MetricPoint::new("temperature", 1_700_000_000_000, 23.7);
        let bytes = p.to_bytes().expect("serialize");
        let back = MetricPoint::from_bytes(&bytes).expect("deserialize");
        assert_eq!(p.metric, back.metric);
        assert_eq!(p.timestamp_ms, back.timestamp_ms);
        assert!((p.value - back.value).abs() < 1e-12);
    }

    #[test]
    fn test_metric_point_unicode_metric_name() {
        let p = MetricPoint::new("温度センサー", 42, 100.0);
        let bytes = p.to_bytes().expect("serialize unicode");
        let back = MetricPoint::from_bytes(&bytes).expect("deserialize unicode");
        assert_eq!(back.metric, "温度センサー");
    }

    #[test]
    fn test_metric_point_from_bytes_too_short() {
        let result = MetricPoint::from_bytes(&[0u8; 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_metric_point_clone_and_eq() {
        let p = MetricPoint::new("x", 0, 0.0);
        assert_eq!(p, p.clone());
    }

    #[test]
    fn test_metric_point_empty_metric_name() {
        let p = MetricPoint::new("", 1000, 42.0);
        let bytes = p.to_bytes().expect("serialize empty name");
        let back = MetricPoint::from_bytes(&bytes).expect("deserialize empty name");
        assert_eq!(back.metric, "");
        assert_eq!(back.timestamp_ms, 1000);
    }

    #[test]
    fn test_metric_point_negative_value() {
        let p = MetricPoint::new("temp", 1000, -40.5);
        let bytes = p.to_bytes().expect("serialize neg");
        let back = MetricPoint::from_bytes(&bytes).expect("deserialize neg");
        assert!((back.value - (-40.5)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metric_point_zero_timestamp() {
        let p = MetricPoint::new("x", 0, 0.0);
        let bytes = p.to_bytes().expect("serialize zero ts");
        let back = MetricPoint::from_bytes(&bytes).expect("deserialize zero ts");
        assert_eq!(back.timestamp_ms, 0);
    }

    #[test]
    fn test_metric_point_max_timestamp() {
        let p = MetricPoint::new("x", i64::MAX, 1.0);
        let bytes = p.to_bytes().expect("serialize max ts");
        let back = MetricPoint::from_bytes(&bytes).expect("deserialize max ts");
        assert_eq!(back.timestamp_ms, i64::MAX);
    }

    #[test]
    fn test_metric_point_special_float_values() {
        let p_inf = MetricPoint::new("x", 0, f64::INFINITY);
        let bytes = p_inf.to_bytes().expect("serialize inf");
        let back = MetricPoint::from_bytes(&bytes).expect("deserialize inf");
        assert!(back.value.is_infinite() && back.value.is_sign_positive());

        let p_nan = MetricPoint::new("x", 0, f64::NAN);
        let bytes = p_nan.to_bytes().expect("serialize nan");
        let back = MetricPoint::from_bytes(&bytes).expect("deserialize nan");
        assert!(back.value.is_nan());
    }

    #[test]
    fn test_metric_point_serde_json_roundtrip() {
        let p = MetricPoint::new("latency", 1_700_000_000_000, 2.5);
        let json = serde_json::to_string(&p).expect("json serialize");
        let back: MetricPoint = serde_json::from_str(&json).expect("json deserialize");
        assert_eq!(p, back);
    }

    // -- CrcWal ---------------------------------------------------------------

    #[test]
    fn test_crc_wal_append_and_replay() {
        let path = tmp_path("append_replay");
        let _ = std::fs::remove_file(&path);

        let pts = vec![
            MetricPoint::new("cpu", 1_000, 55.0),
            MetricPoint::new("mem", 2_000, 8192.0),
            MetricPoint::new("disk", 3_000, 0.75),
        ];

        {
            let mut wal = CrcWal::open(&path).expect("open");
            for p in &pts {
                wal.append(p).expect("append");
            }
        }

        let replayed = CrcWal::replay(&path).expect("replay");
        assert_eq!(replayed.len(), 3);
        assert_eq!(replayed[0].metric, "cpu");
        assert_eq!(replayed[1].metric, "mem");
        assert_eq!(replayed[2].metric, "disk");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_crc_wal_crc_mismatch_detected() {
        let path = tmp_path("crc_corrupt");
        let _ = std::fs::remove_file(&path);

        {
            let mut wal = CrcWal::open(&path).expect("open");
            wal.append(&MetricPoint::new("x", 0, 1.0)).expect("append");
        }

        // Corrupt the stored CRC (flip byte 0)
        {
            let mut data = std::fs::read(&path).expect("read");
            data[0] ^= 0xFF;
            std::fs::write(&path, &data).expect("write corrupt");
        }

        let result = CrcWal::replay(&path);
        assert!(
            matches!(result, Err(TsdbError::CrcMismatch { .. })),
            "expected CrcMismatch, got: {result:?}"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_crc_wal_clear() {
        let path = tmp_path("clear");
        let _ = std::fs::remove_file(&path);

        let mut wal = CrcWal::open(&path).expect("open");
        wal.append(&MetricPoint::new("y", 1, 2.0)).expect("append");
        assert_eq!(wal.record_count(), 1);

        wal.clear().expect("clear");
        assert_eq!(wal.record_count(), 0);

        let replayed = CrcWal::replay(&path).expect("replay after clear");
        assert!(replayed.is_empty());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_crc_wal_record_count() {
        let path = tmp_path("record_count");
        let _ = std::fs::remove_file(&path);

        let mut wal = CrcWal::open(&path).expect("open");
        for i in 0..10u32 {
            wal.append(&MetricPoint::new(format!("s{i}"), i as i64, i as f64))
                .expect("append");
        }
        assert_eq!(wal.record_count(), 10);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_crc_wal_append_batch() {
        let path = tmp_path("batch");
        let _ = std::fs::remove_file(&path);

        let pts: Vec<_> = (0..5).map(|i| MetricPoint::new("b", i, i as f64)).collect();

        {
            let mut wal = CrcWal::open(&path).expect("open");
            let seq = wal.append_batch(&pts).expect("batch");
            assert!(seq > 0);
        }

        let replayed = CrcWal::replay(&path).expect("replay");
        assert_eq!(replayed.len(), 5);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_crc32_known_value() {
        let crc = CrcWal::crc32(b"");
        assert_eq!(crc, 0x0000_0000);
    }

    #[test]
    fn test_crc32_non_empty() {
        let a = CrcWal::crc32(b"hello");
        let b = CrcWal::crc32(b"hello");
        assert_eq!(a, b, "CRC must be deterministic");
        let c = CrcWal::crc32(b"world");
        assert_ne!(a, c, "different data must produce different CRC");
    }

    #[test]
    fn test_crc_wal_file_size_grows() {
        let path = tmp_path("size_grows");
        let _ = std::fs::remove_file(&path);

        let mut wal = CrcWal::open(&path).expect("open");
        let before = wal.file_size().expect("size");

        wal.append(&MetricPoint::new("s", 0, 0.0)).expect("append");
        let after = wal.file_size().expect("size after");

        assert!(after > before, "WAL must grow after append");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_crc_wal_multiple_sessions() {
        let path = tmp_path("multi_session");
        let _ = std::fs::remove_file(&path);

        // Session 1: write 3 records
        {
            let mut wal = CrcWal::open(&path).expect("open");
            for i in 0..3 {
                wal.append(&MetricPoint::new("s1", i, i as f64))
                    .expect("append");
            }
        }

        // Session 2: write 2 more records (append mode)
        {
            let mut wal = CrcWal::open(&path).expect("reopen");
            for i in 3..5 {
                wal.append(&MetricPoint::new("s2", i, i as f64))
                    .expect("append");
            }
        }

        // Verify all 5 records are present
        let replayed = CrcWal::replay(&path).expect("replay");
        assert_eq!(replayed.len(), 5);
        assert_eq!(replayed[0].metric, "s1");
        assert_eq!(replayed[3].metric, "s2");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_crc_wal_large_metric_name() {
        let path = tmp_path("large_name");
        let _ = std::fs::remove_file(&path);

        let large_name = "x".repeat(10_000);
        let mut wal = CrcWal::open(&path).expect("open");
        wal.append(&MetricPoint::new(&large_name, 0, 0.0))
            .expect("append large name");

        let replayed = CrcWal::replay(&path).expect("replay");
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].metric.len(), 10_000);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_crc_wal_sync() {
        let path = tmp_path("sync_test");
        let _ = std::fs::remove_file(&path);

        let mut wal = CrcWal::open(&path).expect("open");
        wal.append(&MetricPoint::new("s", 0, 1.0)).expect("append");
        wal.sync().expect("sync should succeed");

        let replayed = CrcWal::replay(&path).expect("replay");
        assert_eq!(replayed.len(), 1);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_crc_wal_path_accessor() {
        let path = tmp_path("path_access");
        let _ = std::fs::remove_file(&path);

        let wal = CrcWal::open(&path).expect("open");
        assert_eq!(wal.path(), path.as_path());

        let _ = std::fs::remove_file(path);
    }

    // -- RingBuffer -----------------------------------------------------------

    #[test]
    fn test_ring_buffer_capacity_enforced() {
        let rb = RingBuffer::new(3);
        rb.push(MetricPoint::new("a", 0, 0.0)).expect("push 1");
        rb.push(MetricPoint::new("b", 1, 1.0)).expect("push 2");
        rb.push(MetricPoint::new("c", 2, 2.0)).expect("push 3");
        let result = rb.push(MetricPoint::new("d", 3, 3.0));
        assert!(
            matches!(result, Err(TsdbError::BufferFull(_))),
            "should be BufferFull"
        );
    }

    #[test]
    fn test_ring_buffer_drain_all() {
        let rb = RingBuffer::new(10);
        for i in 0..5 {
            rb.push(MetricPoint::new("x", i, i as f64)).expect("push");
        }
        let drained = rb.drain_all().expect("drain");
        assert_eq!(drained.len(), 5);
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn test_ring_buffer_drain_preserves_order() {
        let rb = RingBuffer::new(10);
        for i in 0..5 {
            rb.push(MetricPoint::new(format!("m{i}"), i, i as f64))
                .expect("push");
        }
        let drained = rb.drain_all().expect("drain");
        for (i, p) in drained.iter().enumerate() {
            assert_eq!(p.metric, format!("m{i}"));
            assert_eq!(p.timestamp_ms, i as i64);
        }
    }

    #[test]
    fn test_ring_buffer_capacity_accessor() {
        let rb = RingBuffer::new(42);
        assert_eq!(rb.capacity(), 42);
    }

    #[test]
    fn test_ring_buffer_len_after_push_and_drain() {
        let rb = RingBuffer::new(10);
        assert_eq!(rb.len(), 0);
        rb.push(MetricPoint::new("a", 0, 0.0)).expect("push");
        assert_eq!(rb.len(), 1);
        rb.push(MetricPoint::new("b", 1, 1.0)).expect("push");
        assert_eq!(rb.len(), 2);
        let _ = rb.drain_all().expect("drain");
        assert_eq!(rb.len(), 0);
    }

    // -- BatchWriter ----------------------------------------------------------

    #[test]
    fn test_batch_writer_write_and_flush() {
        let path = tmp_path("bw_flush");
        let _ = std::fs::remove_file(&path);

        let cfg = BatchWriterConfig::with_capacity(10);
        let mut bw = BatchWriter::open(&path, cfg).expect("open");

        for i in 0..5 {
            bw.write_point(MetricPoint::new("cpu", i, i as f64))
                .expect("write");
        }
        let seq = bw.flush().expect("flush");
        assert!(seq > 0);
        assert_eq!(bw.total_flushed(), 5);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_batch_writer_auto_flush_on_capacity() {
        let path = tmp_path("bw_auto");
        let _ = std::fs::remove_file(&path);

        let cfg = BatchWriterConfig::with_capacity(3);
        let mut bw = BatchWriter::open(&path, cfg).expect("open");

        for i in 0..4i64 {
            bw.write_point(MetricPoint::new("x", i, i as f64))
                .expect("write");
        }

        assert!(bw.total_flushed() >= 3);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_batch_writer_write_batch() {
        let path = tmp_path("bw_write_batch");
        let _ = std::fs::remove_file(&path);

        let pts: Vec<_> = (0..20i64)
            .map(|i| MetricPoint::new(format!("s{i}"), i, i as f64))
            .collect();

        let cfg = BatchWriterConfig::with_capacity(8);
        let mut bw = BatchWriter::open(&path, cfg).expect("open");

        let seq = bw.write_batch(&pts).expect("write_batch");
        assert!(seq > 0);
        assert_eq!(bw.total_flushed(), 20);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_batch_writer_replay_after_crash() {
        let path = tmp_path("bw_crash_recovery");
        let _ = std::fs::remove_file(&path);

        let pts: Vec<_> = (0..10i64)
            .map(|i| MetricPoint::new("temp", i * 1_000, 20.0 + i as f64))
            .collect();

        {
            let cfg = BatchWriterConfig::default();
            let mut bw = BatchWriter::open(&path, cfg).expect("open");
            bw.write_batch(&pts).expect("write");
        }

        let recovered = BatchWriter::replay_wal(&path).expect("replay");
        assert_eq!(recovered.len(), 10);
        assert_eq!(recovered[0].metric, "temp");
        assert!((recovered[5].value - 25.0).abs() < 1e-9);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_batch_writer_flush_empty_buffer() {
        let path = tmp_path("bw_empty_flush");
        let _ = std::fs::remove_file(&path);

        let mut bw = BatchWriter::open(&path, Default::default()).expect("open");
        let seq = bw.flush().expect("flush empty");
        assert_eq!(seq, 0, "flushing empty buffer should return seq 0");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_batch_writer_config_default() {
        let cfg = BatchWriterConfig::default();
        assert_eq!(cfg.batch_capacity, 4096);
        assert!(!cfg.fsync_on_flush);
    }

    #[test]
    fn test_batch_writer_buffered_count() {
        let path = tmp_path("bw_count");
        let _ = std::fs::remove_file(&path);

        let cfg = BatchWriterConfig::with_capacity(100);
        let mut bw = BatchWriter::open(&path, cfg).expect("open");

        assert_eq!(bw.buffered_count(), 0);
        bw.write_point(MetricPoint::new("a", 0, 0.0))
            .expect("write");
        assert_eq!(bw.buffered_count(), 1);

        let _ = std::fs::remove_file(path);
    }

    // -- Additional BatchWriter tests for high-performance write path ---------

    #[test]
    fn test_batch_writer_fsync_on_flush() {
        let path = tmp_path("bw_fsync");
        let _ = std::fs::remove_file(&path);

        let cfg = BatchWriterConfig {
            batch_capacity: 10,
            fsync_on_flush: true,
        };
        let mut bw = BatchWriter::open(&path, cfg).expect("open");

        bw.write_point(MetricPoint::new("x", 0, 1.0))
            .expect("write");
        let seq = bw.flush().expect("flush with fsync");
        assert!(seq > 0);

        // Verify data survived fsync
        let recovered = BatchWriter::replay_wal(&path).expect("replay");
        assert_eq!(recovered.len(), 1);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_batch_writer_large_batch() {
        let path = tmp_path("bw_large");
        let _ = std::fs::remove_file(&path);

        let pts: Vec<_> = (0..1000i64)
            .map(|i| MetricPoint::new("sensor", i * 100, i as f64 * 0.1))
            .collect();

        let cfg = BatchWriterConfig::with_capacity(100);
        let mut bw = BatchWriter::open(&path, cfg).expect("open");
        let seq = bw.write_batch(&pts).expect("write large batch");
        assert!(seq > 0);
        assert_eq!(bw.total_flushed(), 1000);

        let recovered = BatchWriter::replay_wal(&path).expect("replay");
        assert_eq!(recovered.len(), 1000);
        assert!((recovered[999].value - 99.9).abs() < 1e-9);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_batch_writer_multiple_flush_cycles() {
        let path = tmp_path("bw_multi_flush");
        let _ = std::fs::remove_file(&path);

        let cfg = BatchWriterConfig::with_capacity(5);
        let mut bw = BatchWriter::open(&path, cfg).expect("open");

        // Flush cycle 1
        for i in 0..5 {
            bw.write_point(MetricPoint::new("a", i, i as f64))
                .expect("write");
        }
        bw.flush().expect("flush 1");
        assert_eq!(bw.total_flushed(), 5);

        // Flush cycle 2
        for i in 5..10 {
            bw.write_point(MetricPoint::new("b", i, i as f64))
                .expect("write");
        }
        bw.flush().expect("flush 2");
        assert_eq!(bw.total_flushed(), 10);

        let recovered = BatchWriter::replay_wal(&path).expect("replay");
        assert_eq!(recovered.len(), 10);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_batch_writer_batch_capacity_accessor() {
        let path = tmp_path("bw_cap");
        let _ = std::fs::remove_file(&path);

        let cfg = BatchWriterConfig::with_capacity(256);
        let bw = BatchWriter::open(&path, cfg).expect("open");
        assert_eq!(bw.batch_capacity(), 256);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_batch_writer_interleaved_write_and_flush() {
        let path = tmp_path("bw_interleave");
        let _ = std::fs::remove_file(&path);

        let cfg = BatchWriterConfig::with_capacity(50);
        let mut bw = BatchWriter::open(&path, cfg).expect("open");

        // Write-flush-write-flush pattern
        bw.write_point(MetricPoint::new("a", 0, 1.0)).expect("w1");
        bw.flush().expect("f1");
        bw.write_point(MetricPoint::new("b", 1, 2.0)).expect("w2");
        bw.write_point(MetricPoint::new("c", 2, 3.0)).expect("w3");
        bw.flush().expect("f2");

        assert_eq!(bw.total_flushed(), 3);

        let recovered = BatchWriter::replay_wal(&path).expect("replay");
        assert_eq!(recovered.len(), 3);
        assert_eq!(recovered[0].metric, "a");
        assert_eq!(recovered[1].metric, "b");
        assert_eq!(recovered[2].metric, "c");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_batch_writer_config_with_capacity() {
        let cfg = BatchWriterConfig::with_capacity(512);
        assert_eq!(cfg.batch_capacity, 512);
        assert!(!cfg.fsync_on_flush); // defaults preserved
    }
}
