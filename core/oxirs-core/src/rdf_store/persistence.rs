//! Durable, append-based persistence for the [`Persistent`](super::StorageBackend::Persistent)
//! storage backend.
//!
//! ## Design
//!
//! The on-disk format stays plain N-Quads (`data.nq`), one statement per line,
//! so the file is human-readable and interoperable. Durability and performance
//! are achieved without ever rewriting the whole file per insert:
//!
//! * **Inserts append.** Each new quad is serialized to a single N-Quads line
//!   and appended through a shared [`BufWriter`] opened in append mode. Loading
//!   `N` quads is therefore `O(N)`, not the previous `O(N^2)` full-file rewrite
//!   per insert.
//! * **Deletions compact.** An append log cannot express a deletion, so
//!   `remove`/`clear`/`drop-graph` mutate memory and set a dirty flag. On an
//!   explicit [`flush`](PersistentState::flush) or on `Drop`, if the store is
//!   dirty it is compacted: the live quads are written to `data.nq.tmp`,
//!   `sync_all`'d, and atomically `rename`d over `data.nq`, after which a fresh
//!   append handle is opened. Deletions become durable across a reopen.
//! * **Crash safety.** Appended lines are line-atomic; a torn trailing line
//!   (from a crash mid-append) is detected and truncated on load. Compaction is
//!   all-or-nothing thanks to the temp-file + `sync_all` + atomic `rename`.
//! * **Fail-safe load.** `File::open` errors on an existing file propagate
//!   (rather than silently yielding an empty store that a later rewrite would
//!   make permanent). Mid-file parse errors are counted and logged, and the
//!   first compaction is refused when the load reported errors so unreadable
//!   data is never overwritten.

use super::storage::MemoryStorage;
use crate::model::Quad;
use crate::parser::RdfFormat;
use crate::serializer::Serializer;
use crate::{OxirsError, Result};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

/// Controls how often appended data is forced to stable storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// `fsync` the append log once every `n` append operations (and on every
    /// explicit flush / bulk write / compaction). `EveryN(1)` fsyncs on every
    /// insert (maximum durability, lowest throughput).
    EveryN(u64),
    /// Never `fsync` on individual appends; only on explicit
    /// [`flush`](PersistentState::flush), bulk writes, and compaction.
    OnFlush,
}

impl Default for SyncPolicy {
    fn default() -> Self {
        SyncPolicy::EveryN(1000)
    }
}

/// Per-store persistence state for the `Persistent` backend.
///
/// Held behind an `Arc` inside `StorageBackend::Persistent`; all methods take
/// `&self` and use interior mutability so both the `&self` `Store`-trait path
/// and the `&mut self` inherent path share one append writer.
#[derive(Debug)]
pub struct PersistentState {
    /// Dataset directory.
    path: PathBuf,
    /// `<path>/data.nq`.
    data_file: PathBuf,
    /// Append writer over `data.nq` (append mode). `None` only transiently
    /// during compaction.
    append: Mutex<Option<BufWriter<File>>>,
    /// Set when an in-memory deletion has not yet been compacted to disk.
    dirty: AtomicBool,
    /// Appends since the last `fsync`.
    ops_since_sync: AtomicU64,
    /// Sync policy.
    sync_policy: SyncPolicy,
    /// Set when the initial load could not fully read/parse `data.nq`. While
    /// set, compaction is refused so partially-read data is never overwritten.
    load_had_errors: AtomicBool,
}

impl PersistentState {
    /// Open (or create) the append log for the dataset directory `path`.
    pub fn open(path: PathBuf, sync_policy: SyncPolicy, load_had_errors: bool) -> Result<Self> {
        std::fs::create_dir_all(&path).map_err(|e| {
            OxirsError::Io(format!(
                "Failed to create directory {}: {e}",
                path.display()
            ))
        })?;
        let data_file = path.join("data.nq");
        let writer = Self::open_append_writer(&data_file)?;
        Ok(Self {
            path,
            data_file,
            append: Mutex::new(Some(writer)),
            dirty: AtomicBool::new(false),
            ops_since_sync: AtomicU64::new(0),
            sync_policy,
            load_had_errors: AtomicBool::new(load_had_errors),
        })
    }

    /// Dataset directory.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn open_append_writer(data_file: &Path) -> Result<BufWriter<File>> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(data_file)
            .map_err(|e| {
                OxirsError::Io(format!(
                    "Failed to open append log {}: {e}",
                    data_file.display()
                ))
            })?;
        Ok(BufWriter::new(file))
    }

    /// Mark the store dirty because an in-memory mutation (delete/clear/drop)
    /// cannot be expressed by the append log and needs a compaction to persist.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Whether a compaction is pending.
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::SeqCst)
    }

    /// Append a single already-serialized N-Quads line (without trailing
    /// newline). Applies the sync policy.
    pub fn append_line(&self, line: &str) -> Result<()> {
        let mut guard = self
            .append
            .lock()
            .map_err(|e| OxirsError::Store(format!("append lock poisoned: {e}")))?;
        if guard.is_none() {
            *guard = Some(Self::open_append_writer(&self.data_file)?);
        }
        let writer = guard
            .as_mut()
            .ok_or_else(|| OxirsError::Store("append writer unavailable".to_string()))?;
        writeln!(writer, "{line}")
            .map_err(|e| OxirsError::Io(format!("Failed to append quad: {e}")))?;
        let ops = self.ops_since_sync.fetch_add(1, Ordering::SeqCst) + 1;
        if let SyncPolicy::EveryN(n) = self.sync_policy {
            if n > 0 && ops % n == 0 {
                Self::flush_writer(writer)?;
                self.ops_since_sync.store(0, Ordering::SeqCst);
            }
        }
        Ok(())
    }

    /// Append many serialized N-Quads lines under a single lock with exactly
    /// one `fsync` at the end, regardless of sync policy. Used by the batched
    /// bulk-insert path.
    pub fn append_lines(&self, lines: &[String]) -> Result<()> {
        if lines.is_empty() {
            return Ok(());
        }
        let mut guard = self
            .append
            .lock()
            .map_err(|e| OxirsError::Store(format!("append lock poisoned: {e}")))?;
        if guard.is_none() {
            *guard = Some(Self::open_append_writer(&self.data_file)?);
        }
        let writer = guard
            .as_mut()
            .ok_or_else(|| OxirsError::Store("append writer unavailable".to_string()))?;
        for line in lines {
            writeln!(writer, "{line}")
                .map_err(|e| OxirsError::Io(format!("Failed to append quad: {e}")))?;
        }
        Self::flush_writer(writer)?;
        self.ops_since_sync.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// Flush the userspace buffer to the OS and `fsync` the file.
    fn flush_writer(writer: &mut BufWriter<File>) -> Result<()> {
        writer
            .flush()
            .map_err(|e| OxirsError::Io(format!("Failed to flush append log: {e}")))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|e| OxirsError::Io(format!("Failed to fsync append log: {e}")))?;
        Ok(())
    }

    /// Make all pending state durable. If the store is dirty (has uncompacted
    /// deletions) it is compacted; otherwise the append buffer is flushed and
    /// `fsync`'d.
    pub fn flush(&self, storage: &MemoryStorage) -> Result<()> {
        if self.is_dirty() {
            self.compact(storage)
        } else {
            let mut guard = self
                .append
                .lock()
                .map_err(|e| OxirsError::Store(format!("append lock poisoned: {e}")))?;
            if let Some(writer) = guard.as_mut() {
                Self::flush_writer(writer)?;
            }
            self.ops_since_sync.store(0, Ordering::SeqCst);
            Ok(())
        }
    }

    /// Atomically rewrite `data.nq` from the live in-memory quads, then reopen
    /// a fresh append handle. This is the only destructive path; it refuses to
    /// run when the initial load reported errors.
    pub fn compact(&self, storage: &MemoryStorage) -> Result<()> {
        if self.load_had_errors.load(Ordering::SeqCst) {
            return Err(OxirsError::Store(format!(
                "Refusing to compact {}: the initial load reported parse/open errors, so \
                 rewriting the file could discard data that failed to load. Fix or remove the \
                 corrupt data.nq and reopen the store.",
                self.data_file.display()
            )));
        }

        let tmp_file = self.data_file.with_extension("nq.tmp");
        {
            let file = File::create(&tmp_file).map_err(|e| {
                OxirsError::Io(format!("Failed to create {}: {e}", tmp_file.display()))
            })?;
            let mut writer = BufWriter::new(file);
            let serializer = Serializer::new(RdfFormat::NQuads);
            for quad in &storage.quads {
                let line = serializer.serialize_quad_to_nquads(quad)?;
                writeln!(writer, "{line}").map_err(|e| {
                    OxirsError::Io(format!("Failed to write {}: {e}", tmp_file.display()))
                })?;
            }
            writer.flush().map_err(|e| {
                OxirsError::Io(format!("Failed to flush {}: {e}", tmp_file.display()))
            })?;
            writer.get_ref().sync_all().map_err(|e| {
                OxirsError::Io(format!("Failed to fsync {}: {e}", tmp_file.display()))
            })?;
        }

        // Close the current append handle before the rename so every platform
        // (notably Windows) can replace the file.
        let mut guard = self
            .append
            .lock()
            .map_err(|e| OxirsError::Store(format!("append lock poisoned: {e}")))?;
        *guard = None;

        std::fs::rename(&tmp_file, &self.data_file).map_err(|e| {
            // Best-effort reopen so the store stays writable even if rename failed.
            if let Ok(w) = Self::open_append_writer(&self.data_file) {
                *guard = Some(w);
            }
            OxirsError::Io(format!(
                "Failed to atomically replace {}: {e}",
                self.data_file.display()
            ))
        })?;

        *guard = Some(Self::open_append_writer(&self.data_file)?);
        self.dirty.store(false, Ordering::SeqCst);
        self.ops_since_sync.store(0, Ordering::SeqCst);
        Ok(())
    }
}

/// Load a `MemoryStorage` from an existing N-Quads `data.nq` file.
///
/// Returns the populated storage and a flag indicating whether any mid-file
/// parse errors were encountered (which callers use to refuse the first
/// destructive rewrite). A single torn trailing line — the hallmark of a crash
/// mid-append — is dropped and truncated from the file. `File::open` failures on
/// the existing file are propagated rather than silently swallowed.
pub fn load_from_disk(data_file: &Path) -> Result<(MemoryStorage, bool)> {
    let mut storage = MemoryStorage::new();

    let mut file = File::open(data_file)
        .map_err(|e| OxirsError::Io(format!("Failed to open {}: {e}", data_file.display())))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| OxirsError::Io(format!("Failed to read {}: {e}", data_file.display())))?;
    drop(file);

    if bytes.is_empty() {
        return Ok((storage, false));
    }

    let ends_with_newline = bytes.last() == Some(&b'\n');
    let segments: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
    let seg_count = segments.len();

    let mut had_errors = false;
    let mut torn_trailing: Option<usize> = None;

    for (idx, seg) in segments.iter().enumerate() {
        let is_last_segment = idx + 1 == seg_count;
        if seg.iter().all(|b| b.is_ascii_whitespace()) {
            continue;
        }
        match parse_nquads_line(seg) {
            Ok(quads) => {
                for quad in quads {
                    storage.insert_quad(quad);
                }
            }
            Err(msg) => {
                if is_last_segment && !ends_with_newline {
                    tracing::warn!(
                        "Dropping torn trailing line ({} bytes) in {}: {msg}",
                        seg.len(),
                        data_file.display()
                    );
                    torn_trailing = Some(seg.len());
                } else {
                    had_errors = true;
                    tracing::warn!(
                        "Skipping unparseable line {} in {}: {msg}",
                        idx + 1,
                        data_file.display()
                    );
                }
            }
        }
    }

    // Normalize the on-disk file so subsequent appends never merge into the last
    // line: either truncate a torn trailing line, or add a missing final newline.
    if let Some(torn_len) = torn_trailing {
        let new_len = bytes.len().saturating_sub(torn_len) as u64;
        match OpenOptions::new().write(true).open(data_file) {
            Ok(f) => {
                if let Err(e) = f.set_len(new_len) {
                    tracing::error!(
                        "Failed to truncate torn trailing line in {}: {e}",
                        data_file.display()
                    );
                }
            }
            Err(e) => tracing::error!(
                "Failed to open {} to truncate torn trailing line: {e}",
                data_file.display()
            ),
        }
    } else if !ends_with_newline {
        // The last valid line lacked a trailing newline; add one.
        match OpenOptions::new().append(true).open(data_file) {
            Ok(mut f) => {
                if let Err(e) = f.write_all(b"\n") {
                    tracing::error!(
                        "Failed to normalize trailing newline in {}: {e}",
                        data_file.display()
                    );
                }
            }
            Err(e) => tracing::error!(
                "Failed to open {} to normalize trailing newline: {e}",
                data_file.display()
            ),
        }
    }

    Ok((storage, had_errors))
}

/// Parse a single N-Quads line into zero or more quads. Blank/comment lines
/// yield an empty vector; a genuine parse failure yields `Err`.
fn parse_nquads_line(line: &[u8]) -> std::result::Result<Vec<Quad>, String> {
    // Note: `crate::format::RdfFormat` (the parser's format enum) is a distinct
    // type from `crate::parser::RdfFormat` (the serializer's) imported above.
    use crate::format::{RdfFormat as FormatRdfFormat, RdfParser};

    match line.iter().position(|&b| !b.is_ascii_whitespace()) {
        None => return Ok(Vec::new()),
        Some(i) if line[i] == b'#' => return Ok(Vec::new()),
        _ => {}
    }

    let parser = RdfParser::new(FormatRdfFormat::NQuads);
    let mut quads = Vec::new();
    for item in parser.for_slice(line) {
        match item {
            Ok(quad) => quads.push(quad),
            Err(e) => return Err(e.to_string()),
        }
    }
    if quads.is_empty() {
        return Err("line produced no quads".to_string());
    }
    Ok(quads)
}
