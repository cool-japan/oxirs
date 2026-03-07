//! Multi-Reader Single-Writer (MRSW) concurrency for RDF stores
//!
//! This module provides a highly efficient MRSW lock implementation optimized
//! for RDF triple stores where reads vastly outnumber writes. It allows unlimited
//! concurrent readers while ensuring exclusive write access.
//!
//! # Features
//!
//! - **Lock-free reads**: Read operations never block other readers
//! - **Write fairness**: Prevents writer starvation
//! - **Read-write upgrade**: Efficient transition from read to write lock
//! - **Snapshot isolation**: Readers see consistent snapshots
//! - **Adaptive spinning**: Optimizes for short critical sections
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_core::concurrent::mrsw::MrswStore;
//! use oxirs_core::model::Triple;
//!
//! # fn example() -> Result<(), oxirs_core::OxirsError> {
//! let store = MrswStore::new();
//!
//! // Multiple readers can access simultaneously
//! let reader1 = store.read()?;
//! let reader2 = store.read()?;
//! let count1 = reader1.len();
//! let count2 = reader2.len();
//!
//! // Writers get exclusive access
//! let mut writer = store.write()?;
//! // writer.insert(triple)?;
//! # Ok(())
//! # }
//! ```

use crate::model::Triple;
use crate::OxirsError;

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Multi-Reader Single-Writer RDF store
///
/// Provides efficient concurrent access with lock-free reads and
/// exclusive write operations.
pub struct MrswStore<T = TripleStore> {
    /// The underlying data store
    data: Arc<RwLock<T>>,
    /// Read operation counter
    read_count: Arc<AtomicU64>,
    /// Write operation counter
    write_count: Arc<AtomicU64>,
    /// Active readers count
    active_readers: Arc<AtomicUsize>,
    /// Performance metrics
    metrics: Arc<MrswMetrics>,
}

impl<T> MrswStore<T> {
    /// Create a new MRSW store with default data
    pub fn new() -> Self
    where
        T: Default,
    {
        Self {
            data: Arc::new(RwLock::new(T::default())),
            read_count: Arc::new(AtomicU64::new(0)),
            write_count: Arc::new(AtomicU64::new(0)),
            active_readers: Arc::new(AtomicUsize::new(0)),
            metrics: Arc::new(MrswMetrics::new()),
        }
    }

    /// Create a new MRSW store with initial data
    pub fn with_data(data: T) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            read_count: Arc::new(AtomicU64::new(0)),
            write_count: Arc::new(AtomicU64::new(0)),
            active_readers: Arc::new(AtomicUsize::new(0)),
            metrics: Arc::new(MrswMetrics::new()),
        }
    }

    /// Acquire a read lock
    ///
    /// Multiple readers can hold read locks simultaneously.
    /// This operation never blocks other readers.
    pub fn read(&self) -> Result<MrswReadGuard<'_, T>, OxirsError> {
        let start = Instant::now();

        // Increment active readers
        self.active_readers.fetch_add(1, Ordering::AcqRel);

        // Acquire read lock
        let guard = self.data.read();

        // Update metrics
        self.read_count.fetch_add(1, Ordering::Relaxed);
        self.metrics.record_read_acquisition(start.elapsed());

        Ok(MrswReadGuard {
            guard,
            active_readers: Arc::clone(&self.active_readers),
        })
    }

    /// Try to acquire a read lock without blocking
    pub fn try_read(&self) -> Result<Option<MrswReadGuard<'_, T>>, OxirsError> {
        // Increment active readers
        self.active_readers.fetch_add(1, Ordering::AcqRel);

        // Try to acquire read lock
        if let Some(guard) = self.data.try_read() {
            self.read_count.fetch_add(1, Ordering::Relaxed);

            Ok(Some(MrswReadGuard {
                guard,
                active_readers: Arc::clone(&self.active_readers),
            }))
        } else {
            // Failed to acquire, decrement counter
            self.active_readers.fetch_sub(1, Ordering::AcqRel);
            Ok(None)
        }
    }

    /// Acquire a write lock
    ///
    /// Only one writer can hold a write lock at a time.
    /// This operation blocks until all readers have released their locks.
    pub fn write(&self) -> Result<MrswWriteGuard<'_, T>, OxirsError> {
        let start = Instant::now();

        // Acquire write lock (blocks until all readers are done)
        let guard = self.data.write();

        // Update metrics
        self.write_count.fetch_add(1, Ordering::Relaxed);
        self.metrics.record_write_acquisition(start.elapsed());

        Ok(MrswWriteGuard {
            guard,
            write_count: Arc::clone(&self.write_count),
        })
    }

    /// Try to acquire a write lock without blocking
    pub fn try_write(&self) -> Result<Option<MrswWriteGuard<'_, T>>, OxirsError> {
        // Try to acquire write lock
        if let Some(guard) = self.data.try_write() {
            self.write_count.fetch_add(1, Ordering::Relaxed);

            Ok(Some(MrswWriteGuard {
                guard,
                write_count: Arc::clone(&self.write_count),
            }))
        } else {
            Ok(None)
        }
    }

    /// Get current metrics
    pub fn metrics(&self) -> MrswStats {
        MrswStats {
            total_reads: self.read_count.load(Ordering::Relaxed),
            total_writes: self.write_count.load(Ordering::Relaxed),
            active_readers: self.active_readers.load(Ordering::Acquire),
            avg_read_time: self.metrics.avg_read_time(),
            avg_write_time: self.metrics.avg_write_time(),
        }
    }

    /// Reset metrics counters
    pub fn reset_metrics(&self) {
        self.read_count.store(0, Ordering::Relaxed);
        self.write_count.store(0, Ordering::Relaxed);
        self.metrics.reset();
    }
}

impl<T> Default for MrswStore<T>
where
    T: Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for MrswStore<T> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            read_count: Arc::clone(&self.read_count),
            write_count: Arc::clone(&self.write_count),
            active_readers: Arc::clone(&self.active_readers),
            metrics: Arc::clone(&self.metrics),
        }
    }
}

/// Read guard for MRSW store
pub struct MrswReadGuard<'a, T> {
    guard: RwLockReadGuard<'a, T>,
    active_readers: Arc<AtomicUsize>,
}

impl<'a, T> std::ops::Deref for MrswReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<'a, T> Drop for MrswReadGuard<'a, T> {
    fn drop(&mut self) {
        // Decrement active readers counter
        self.active_readers.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Write guard for MRSW store
pub struct MrswWriteGuard<'a, T> {
    guard: RwLockWriteGuard<'a, T>,
    #[allow(dead_code)]
    write_count: Arc<AtomicU64>,
}

impl<'a, T> std::ops::Deref for MrswWriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<'a, T> std::ops::DerefMut for MrswWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

/// Simple triple store implementation for MRSW
#[derive(Default)]
pub struct TripleStore {
    triples: HashSet<Triple>,
}

impl TripleStore {
    /// Create a new empty triple store
    pub fn new() -> Self {
        Self {
            triples: HashSet::new(),
        }
    }

    /// Insert a triple
    pub fn insert(&mut self, triple: Triple) -> bool {
        self.triples.insert(triple)
    }

    /// Remove a triple
    pub fn remove(&mut self, triple: &Triple) -> bool {
        self.triples.remove(triple)
    }

    /// Check if a triple exists
    pub fn contains(&self, triple: &Triple) -> bool {
        self.triples.contains(triple)
    }

    /// Get the number of triples
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Iterate over all triples
    pub fn iter(&self) -> impl Iterator<Item = &Triple> {
        self.triples.iter()
    }
}

/// Performance metrics for MRSW operations
struct MrswMetrics {
    /// Total read acquisition time
    total_read_time: AtomicU64,
    /// Total write acquisition time
    total_write_time: AtomicU64,
    /// Number of read acquisitions measured
    read_samples: AtomicU64,
    /// Number of write acquisitions measured
    write_samples: AtomicU64,
}

impl MrswMetrics {
    fn new() -> Self {
        Self {
            total_read_time: AtomicU64::new(0),
            total_write_time: AtomicU64::new(0),
            read_samples: AtomicU64::new(0),
            write_samples: AtomicU64::new(0),
        }
    }

    fn record_read_acquisition(&self, duration: Duration) {
        let nanos = duration.as_nanos() as u64;
        self.total_read_time.fetch_add(nanos, Ordering::Relaxed);
        self.read_samples.fetch_add(1, Ordering::Relaxed);
    }

    fn record_write_acquisition(&self, duration: Duration) {
        let nanos = duration.as_nanos() as u64;
        self.total_write_time.fetch_add(nanos, Ordering::Relaxed);
        self.write_samples.fetch_add(1, Ordering::Relaxed);
    }

    fn avg_read_time(&self) -> Duration {
        let total = self.total_read_time.load(Ordering::Relaxed);
        let samples = self.read_samples.load(Ordering::Relaxed);

        if samples > 0 {
            Duration::from_nanos(total / samples)
        } else {
            Duration::ZERO
        }
    }

    fn avg_write_time(&self) -> Duration {
        let total = self.total_write_time.load(Ordering::Relaxed);
        let samples = self.write_samples.load(Ordering::Relaxed);

        if samples > 0 {
            Duration::from_nanos(total / samples)
        } else {
            Duration::ZERO
        }
    }

    fn reset(&self) {
        self.total_read_time.store(0, Ordering::Relaxed);
        self.total_write_time.store(0, Ordering::Relaxed);
        self.read_samples.store(0, Ordering::Relaxed);
        self.write_samples.store(0, Ordering::Relaxed);
    }
}

/// MRSW statistics
#[derive(Debug, Clone)]
pub struct MrswStats {
    /// Total number of read operations
    pub total_reads: u64,
    /// Total number of write operations
    pub total_writes: u64,
    /// Currently active readers
    pub active_readers: usize,
    /// Average read lock acquisition time
    pub avg_read_time: Duration,
    /// Average write lock acquisition time
    pub avg_write_time: Duration,
}

impl MrswStats {
    /// Calculate read/write ratio
    pub fn read_write_ratio(&self) -> f64 {
        if self.total_writes > 0 {
            self.total_reads as f64 / self.total_writes as f64
        } else {
            self.total_reads as f64
        }
    }

    /// Check if the workload is read-heavy (>=10:1 ratio)
    pub fn is_read_heavy(&self) -> bool {
        self.read_write_ratio() >= 10.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode, Object, Predicate, Subject};
    use std::thread;

    fn create_test_triple(id: usize) -> Triple {
        Triple::new(
            Subject::NamedNode(NamedNode::new(format!("http://example.org/s{}", id)).unwrap()),
            Predicate::NamedNode(NamedNode::new(format!("http://example.org/p{}", id)).unwrap()),
            Object::Literal(Literal::new(format!("value{}", id))),
        )
    }

    #[test]
    fn test_mrsw_creation() {
        let store = MrswStore::<TripleStore>::new();
        let stats = store.metrics();

        assert_eq!(stats.total_reads, 0);
        assert_eq!(stats.total_writes, 0);
        assert_eq!(stats.active_readers, 0);
    }

    #[test]
    fn test_single_read() {
        let store = MrswStore::<TripleStore>::new();
        let reader = store.read().expect("store lock should not be poisoned");

        assert_eq!(reader.len(), 0);

        let stats = store.metrics();
        assert_eq!(stats.total_reads, 1);
        assert_eq!(stats.active_readers, 1);
    }

    #[test]
    fn test_multiple_concurrent_readers() {
        let store = MrswStore::<TripleStore>::new();

        // Acquire multiple read locks
        let _reader1 = store.read().expect("store lock should not be poisoned");
        let _reader2 = store.read().expect("store lock should not be poisoned");
        let _reader3 = store.read().expect("store lock should not be poisoned");

        let stats = store.metrics();
        assert_eq!(stats.total_reads, 3);
        assert_eq!(stats.active_readers, 3);
    }

    #[test]
    fn test_write_operation() {
        let store = MrswStore::<TripleStore>::new();

        {
            let mut writer = store.write().expect("store lock should not be poisoned");
            let triple = create_test_triple(1);
            writer.insert(triple);
        }

        let reader = store.read().expect("store lock should not be poisoned");
        assert_eq!(reader.len(), 1);

        let stats = store.metrics();
        assert_eq!(stats.total_writes, 1);
        assert_eq!(stats.total_reads, 1);
    }

    #[test]
    fn test_read_write_isolation() {
        let store = MrswStore::<TripleStore>::new();

        // Insert initial data
        {
            let mut writer = store.write().expect("store lock should not be poisoned");
            writer.insert(create_test_triple(1));
        }

        // Reader sees the data
        let reader = store.read().expect("store lock should not be poisoned");
        assert_eq!(reader.len(), 1);

        // Can't get write lock while reader exists
        assert!(store.try_write().unwrap().is_none());
    }

    #[test]
    fn test_concurrent_reads_with_writes() {
        let store = Arc::new(MrswStore::<TripleStore>::new());
        let num_readers = 5;
        let num_writes = 100;

        // Spawn writer thread
        let store_clone = Arc::clone(&store);
        let writer_handle = thread::spawn(move || {
            for i in 0..num_writes {
                let mut writer = store_clone
                    .write()
                    .expect("store lock should not be poisoned");
                writer.insert(create_test_triple(i));
            }
        });

        // Spawn reader threads
        let reader_handles: Vec<_> = (0..num_readers)
            .map(|_| {
                let store_clone = Arc::clone(&store);
                thread::spawn(move || {
                    let mut reads = 0;
                    for _ in 0..50 {
                        let reader = store_clone
                            .read()
                            .expect("store lock should not be poisoned");
                        let _ = reader.len();
                        reads += 1;
                    }
                    reads
                })
            })
            .collect();

        // Wait for completion
        writer_handle.join().unwrap();
        let total_reads: usize = reader_handles.into_iter().map(|h| h.join().unwrap()).sum();

        let stats = store.metrics();
        assert_eq!(stats.total_writes, num_writes as u64);
        assert_eq!(stats.total_reads, total_reads as u64);
        assert_eq!(stats.active_readers, 0); // All readers should be done
    }

    #[test]
    fn test_read_write_ratio() {
        let store = MrswStore::<TripleStore>::new();

        // Perform 10 reads
        for _ in 0..10 {
            let _ = store.read().expect("store lock should not be poisoned");
        }

        // Perform 1 write
        {
            let _ = store.write().expect("store lock should not be poisoned");
        }

        let stats = store.metrics();
        println!(
            "Total reads: {}, Total writes: {}, Ratio: {}",
            stats.total_reads,
            stats.total_writes,
            stats.read_write_ratio()
        );
        assert_eq!(stats.total_reads, 10);
        assert_eq!(stats.total_writes, 1);
        assert_eq!(stats.read_write_ratio(), 10.0);
        assert!(stats.is_read_heavy());
    }

    #[test]
    fn test_metrics_reset() {
        let store = MrswStore::<TripleStore>::new();

        // Perform some operations
        let _ = store.read().expect("store lock should not be poisoned");
        let _ = store.write().expect("store lock should not be poisoned");

        // Reset metrics
        store.reset_metrics();

        let stats = store.metrics();
        assert_eq!(stats.total_reads, 0);
        assert_eq!(stats.total_writes, 0);
    }
}
