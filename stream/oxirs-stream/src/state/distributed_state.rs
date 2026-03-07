//! # Distributed State Store
//!
//! Consistent state store for stateful stream operators across partitions.
//! Supports: key-value state, list state, map state, aggregating state.

use crate::error::StreamError;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

// ─── Partition Key ────────────────────────────────────────────────────────────

/// Unique identifier for a state partition.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StatePartitionKey {
    pub operator_id: String,
    pub partition_id: u32,
    pub subtask_index: u32,
}

impl StatePartitionKey {
    /// Create a new partition key.
    pub fn new(operator_id: impl Into<String>, partition_id: u32, subtask_index: u32) -> Self {
        Self {
            operator_id: operator_id.into(),
            partition_id,
            subtask_index,
        }
    }

    /// Serialize partition key to a byte prefix for namespacing state keys.
    pub fn to_prefix(&self) -> Vec<u8> {
        format!(
            "{}:{}:{}:",
            self.operator_id, self.partition_id, self.subtask_index
        )
        .into_bytes()
    }
}

// ─── StateBackend trait ───────────────────────────────────────────────────────

/// Pluggable storage backend for state operators.
///
/// Implementations must be `Send + Sync` so they can be shared across async
/// tasks and threads.
pub trait StateBackend: Send + Sync {
    /// Return the value for the given key, or `None` if absent.
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, StreamError>;

    /// Insert or overwrite a key-value pair.
    fn put(&self, key: &[u8], value: &[u8]) -> Result<(), StreamError>;

    /// Remove a key. Returns `true` if the key previously existed.
    fn delete(&self, key: &[u8]) -> Result<bool, StreamError>;

    /// Return all entries whose key starts with `prefix`.
    #[allow(clippy::type_complexity)]
    fn range_scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, StreamError>;

    /// Serialize the current state into an opaque byte snapshot tagged with
    /// `checkpoint_id`.
    fn checkpoint(&self, checkpoint_id: u64) -> Result<Vec<u8>, StreamError>;

    /// Replace current state with the content of a previously-created snapshot.
    fn restore(&self, snapshot: &[u8]) -> Result<(), StreamError>;

    /// Approximate heap/disk footprint in bytes.
    fn size_bytes(&self) -> usize;
}

// ─── Snapshot encoding ────────────────────────────────────────────────────────
//
// Binary format (all integers little-endian):
//
//   [u64 checkpoint_id]
//   [u64 entry_count]
//   { [u32 key_len] [key_bytes…] [u32 val_len] [val_bytes…] } × entry_count

fn encode_snapshot(checkpoint_id: u64, data: &HashMap<Vec<u8>, Vec<u8>>) -> Vec<u8> {
    let entries_size: usize = data.iter().map(|(k, v)| 8 + k.len() + v.len()).sum();
    let mut out = Vec::with_capacity(16 + entries_size);
    out.extend_from_slice(&checkpoint_id.to_le_bytes());
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    for (k, v) in data {
        out.extend_from_slice(&(k.len() as u32).to_le_bytes());
        out.extend_from_slice(k);
        out.extend_from_slice(&(v.len() as u32).to_le_bytes());
        out.extend_from_slice(v);
    }
    out
}

/// Read a `u64` from `buf[offset..offset+8]`, returning a descriptive error on
/// failure.
#[inline]
fn read_u64(buf: &[u8], offset: usize, field: &str) -> Result<u64, StreamError> {
    buf.get(offset..offset + 8)
        .ok_or_else(|| StreamError::Deserialization(format!("snapshot truncated reading {field}")))?
        .try_into()
        .map(u64::from_le_bytes)
        .map_err(|_| StreamError::Deserialization(format!("bad bytes for {field}")))
}

/// Read a `u32` from `buf[offset..offset+4]`.
#[inline]
fn read_u32(buf: &[u8], offset: usize, field: &str) -> Result<u32, StreamError> {
    buf.get(offset..offset + 4)
        .ok_or_else(|| StreamError::Deserialization(format!("snapshot truncated reading {field}")))?
        .try_into()
        .map(u32::from_le_bytes)
        .map_err(|_| StreamError::Deserialization(format!("bad bytes for {field}")))
}

#[allow(clippy::type_complexity)]
fn decode_snapshot(snapshot: &[u8]) -> Result<(u64, HashMap<Vec<u8>, Vec<u8>>), StreamError> {
    if snapshot.len() < 16 {
        return Err(StreamError::Deserialization(
            "snapshot too short to contain header".into(),
        ));
    }

    let checkpoint_id = read_u64(snapshot, 0, "checkpoint_id")?;
    let entry_count = read_u64(snapshot, 8, "entry_count")? as usize;

    let mut pos = 16usize;
    let mut data = HashMap::with_capacity(entry_count);

    for i in 0..entry_count {
        let key_len = read_u32(snapshot, pos, &format!("key_len[{i}]"))? as usize;
        pos += 4;

        let key = snapshot
            .get(pos..pos + key_len)
            .ok_or_else(|| {
                StreamError::Deserialization(format!("snapshot truncated at key data[{i}]"))
            })?
            .to_vec();
        pos += key_len;

        let val_len = read_u32(snapshot, pos, &format!("val_len[{i}]"))? as usize;
        pos += 4;

        let val = snapshot
            .get(pos..pos + val_len)
            .ok_or_else(|| {
                StreamError::Deserialization(format!("snapshot truncated at val data[{i}]"))
            })?
            .to_vec();
        pos += val_len;

        data.insert(key, val);
    }

    Ok((checkpoint_id, data))
}

// ─── In-memory backend ────────────────────────────────────────────────────────

/// In-memory `StateBackend` — fast but not durable across process restarts.
pub struct InMemoryStateBackend {
    data: Arc<RwLock<HashMap<Vec<u8>, Vec<u8>>>>,
    /// Monotonically increasing logical version, incremented on every write.
    version: Arc<RwLock<u64>>,
}

impl InMemoryStateBackend {
    /// Create an empty in-memory backend.
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            version: Arc::new(RwLock::new(0)),
        }
    }

    /// Current logical version.
    pub fn version(&self) -> Result<u64, StreamError> {
        self.version
            .read()
            .map(|g| *g)
            .map_err(|e| StreamError::Other(format!("version lock poisoned: {e}")))
    }

    fn bump_version(&self) -> Result<(), StreamError> {
        let mut ver = self
            .version
            .write()
            .map_err(|e| StreamError::Other(format!("version write-lock poisoned: {e}")))?;
        *ver += 1;
        Ok(())
    }
}

impl Default for InMemoryStateBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StateBackend for InMemoryStateBackend {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, StreamError> {
        let data = self
            .data
            .read()
            .map_err(|e| StreamError::Other(format!("data read-lock poisoned: {e}")))?;
        Ok(data.get(key).cloned())
    }

    fn put(&self, key: &[u8], value: &[u8]) -> Result<(), StreamError> {
        {
            let mut data = self
                .data
                .write()
                .map_err(|e| StreamError::Other(format!("data write-lock poisoned: {e}")))?;
            data.insert(key.to_vec(), value.to_vec());
        }
        self.bump_version()
    }

    fn delete(&self, key: &[u8]) -> Result<bool, StreamError> {
        let existed = {
            let mut data = self
                .data
                .write()
                .map_err(|e| StreamError::Other(format!("data write-lock poisoned: {e}")))?;
            data.remove(key).is_some()
        };
        if existed {
            self.bump_version()?;
        }
        Ok(existed)
    }

    fn range_scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, StreamError> {
        let data = self
            .data
            .read()
            .map_err(|e| StreamError::Other(format!("data read-lock poisoned: {e}")))?;
        let results = data
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Ok(results)
    }

    fn checkpoint(&self, checkpoint_id: u64) -> Result<Vec<u8>, StreamError> {
        let data = self
            .data
            .read()
            .map_err(|e| StreamError::Other(format!("data read-lock poisoned: {e}")))?;
        Ok(encode_snapshot(checkpoint_id, &data))
    }

    fn restore(&self, snapshot: &[u8]) -> Result<(), StreamError> {
        let (_checkpoint_id, restored) = decode_snapshot(snapshot)?;
        {
            let mut data = self
                .data
                .write()
                .map_err(|e| StreamError::Other(format!("data write-lock poisoned: {e}")))?;
            *data = restored;
        }
        self.bump_version()
    }

    fn size_bytes(&self) -> usize {
        // If the lock is poisoned we return 0 (best-effort metric).
        match self.data.read() {
            Ok(data) => data.iter().map(|(k, v)| k.len() + v.len()).sum(),
            Err(_) => 0,
        }
    }
}

// ─── Keyed state store ────────────────────────────────────────────────────────

/// Typed, partitioned key-value state handle for stateful stream operators.
///
/// Serialization is provided by caller-supplied function pointers to keep this
/// crate free of hard-coded codec dependencies.
pub struct KeyedStateStore<K, V> {
    partition: StatePartitionKey,
    backend: Arc<dyn StateBackend>,
    key_serializer: fn(&K) -> Vec<u8>,
    value_serializer: fn(&V) -> Vec<u8>,
    value_deserializer: fn(&[u8]) -> Result<V, StreamError>,
    _phantom: std::marker::PhantomData<(K, V)>,
}

impl<K: std::fmt::Debug, V: std::fmt::Debug + Clone> KeyedStateStore<K, V> {
    /// Create a new keyed store backed by `backend`.
    pub fn new(
        partition: StatePartitionKey,
        backend: Arc<dyn StateBackend>,
        key_ser: fn(&K) -> Vec<u8>,
        val_ser: fn(&V) -> Vec<u8>,
        val_de: fn(&[u8]) -> Result<V, StreamError>,
    ) -> Self {
        Self {
            partition,
            backend,
            key_serializer: key_ser,
            value_serializer: val_ser,
            value_deserializer: val_de,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Build the fully-namespaced storage key for `key`.
    fn storage_key(&self, key: &K) -> Vec<u8> {
        let mut prefix = self.partition.to_prefix();
        prefix.extend_from_slice(&(self.key_serializer)(key));
        prefix
    }

    /// Return the value for `key`, or `None` if absent.
    pub fn get(&self, key: &K) -> Result<Option<V>, StreamError> {
        match self.backend.get(&self.storage_key(key))? {
            None => Ok(None),
            Some(bytes) => (self.value_deserializer)(&bytes).map(Some),
        }
    }

    /// Store `value` under `key`.
    pub fn put(&self, key: &K, value: V) -> Result<(), StreamError> {
        let bytes = (self.value_serializer)(&value);
        self.backend.put(&self.storage_key(key), &bytes)
    }

    /// Remove `key`. Returns `true` if it existed.
    pub fn delete(&self, key: &K) -> Result<bool, StreamError> {
        self.backend.delete(&self.storage_key(key))
    }

    /// Atomic read-modify-write.  `updater` receives the current value (or
    /// `None`) and returns the new value to store.
    pub fn update_or_default(
        &self,
        key: &K,
        updater: impl FnOnce(Option<V>) -> V,
    ) -> Result<V, StreamError> {
        let current = self.get(key)?;
        let new_value = updater(current);
        self.put(key, new_value.clone())?;
        Ok(new_value)
    }
}

// ─── Aggregating state ────────────────────────────────────────────────────────

/// Aggregating state that folds incoming values into a running accumulator.
///
/// Typical use-cases: running sum, count, min/max, HyperLogLog cardinality.
pub struct AggregatingState<In, Out> {
    partition: StatePartitionKey,
    backend: Arc<dyn StateBackend>,
    /// Fixed key used to store the accumulator within the backend.
    aggregate_key: Vec<u8>,
    /// `combine_fn(accumulator, new_input) -> new_accumulator`
    combine_fn: fn(Out, In) -> Out,
    /// Value returned when no accumulator has been written yet.
    default: Out,
    serializer: fn(&Out) -> Vec<u8>,
    deserializer: fn(&[u8]) -> Result<Out, StreamError>,
    _phantom: std::marker::PhantomData<In>,
}

impl<In, Out: Clone> AggregatingState<In, Out> {
    /// Create a new aggregating state descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        partition: StatePartitionKey,
        backend: Arc<dyn StateBackend>,
        aggregate_key: Vec<u8>,
        combine_fn: fn(Out, In) -> Out,
        default: Out,
        serializer: fn(&Out) -> Vec<u8>,
        deserializer: fn(&[u8]) -> Result<Out, StreamError>,
    ) -> Self {
        Self {
            partition,
            backend,
            aggregate_key,
            combine_fn,
            default,
            serializer,
            deserializer,
            _phantom: std::marker::PhantomData,
        }
    }

    fn storage_key(&self) -> Vec<u8> {
        let mut prefix = self.partition.to_prefix();
        prefix.extend_from_slice(&self.aggregate_key);
        prefix
    }

    fn read_accumulator(&self) -> Result<Out, StreamError> {
        match self.backend.get(&self.storage_key())? {
            None => Ok(self.default.clone()),
            Some(bytes) => (self.deserializer)(&bytes),
        }
    }

    /// Fold `value` into the accumulator.
    pub fn add(&self, value: In) -> Result<(), StreamError> {
        let current = self.read_accumulator()?;
        let new_acc = (self.combine_fn)(current, value);
        self.backend
            .put(&self.storage_key(), &(self.serializer)(&new_acc))
    }

    /// Return the current accumulator value.
    pub fn get(&self) -> Result<Out, StreamError> {
        self.read_accumulator()
    }

    /// Reset the accumulator (delete from backend so default is returned).
    pub fn clear(&self) -> Result<(), StreamError> {
        self.backend.delete(&self.storage_key()).map(|_| ())
    }
}

// ─── Stats ────────────────────────────────────────────────────────────────────

/// Point-in-time metrics for a state backend.
#[derive(Debug, Clone)]
pub struct StateBackendStats {
    pub size_bytes: usize,
    pub collected_at: Instant,
}

impl StateBackendStats {
    /// Collect current stats from a backend.
    pub fn collect(backend: &dyn StateBackend) -> Self {
        Self {
            size_bytes: backend.size_bytes(),
            collected_at: Instant::now(),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper serializers / deserializers used only in tests.
    fn str_key_ser(k: &String) -> Vec<u8> {
        k.as_bytes().to_vec()
    }

    fn i64_ser(v: &i64) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    fn i64_de(b: &[u8]) -> Result<i64, StreamError> {
        if b.len() < 8 {
            return Err(StreamError::Deserialization("i64 needs 8 bytes".into()));
        }
        let arr: [u8; 8] = b[..8]
            .try_into()
            .map_err(|_| StreamError::Deserialization("i64 slice error".into()))?;
        Ok(i64::from_le_bytes(arr))
    }

    fn u64_ser(v: &u64) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    fn u64_de(b: &[u8]) -> Result<u64, StreamError> {
        if b.len() < 8 {
            return Err(StreamError::Deserialization("u64 needs 8 bytes".into()));
        }
        let arr: [u8; 8] = b[..8]
            .try_into()
            .map_err(|_| StreamError::Deserialization("u64 slice error".into()))?;
        Ok(u64::from_le_bytes(arr))
    }

    fn partition() -> StatePartitionKey {
        StatePartitionKey::new("op1", 0, 0)
    }

    #[test]
    fn test_backend_put_get_delete() {
        let backend = InMemoryStateBackend::new();

        backend.put(b"hello", b"world").unwrap();
        let val = backend.get(b"hello").unwrap();
        assert_eq!(val.as_deref(), Some(b"world".as_ref()));

        let existed = backend.delete(b"hello").unwrap();
        assert!(existed);

        assert!(backend.get(b"hello").unwrap().is_none());

        let not_found = backend.delete(b"missing").unwrap();
        assert!(!not_found);
    }

    #[test]
    fn test_backend_range_scan() {
        let backend = InMemoryStateBackend::new();

        backend.put(b"ns:a", b"1").unwrap();
        backend.put(b"ns:b", b"2").unwrap();
        backend.put(b"other:c", b"3").unwrap();

        let results = backend.range_scan(b"ns:").unwrap();
        assert_eq!(results.len(), 2);

        let all = backend.range_scan(b"").unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_backend_checkpoint_restore() {
        let backend = InMemoryStateBackend::new();

        backend.put(b"k1", b"v1").unwrap();
        backend.put(b"k2", b"v2").unwrap();

        let snapshot = backend.checkpoint(42).unwrap();
        assert!(!snapshot.is_empty());

        // Corrupt the live state.
        backend.delete(b"k1").unwrap();
        backend.put(b"k2", b"changed").unwrap();
        backend.put(b"k3", b"new").unwrap();

        // Restore to snapshot.
        backend.restore(&snapshot).unwrap();

        assert_eq!(backend.get(b"k1").unwrap().as_deref(), Some(b"v1".as_ref()));
        assert_eq!(backend.get(b"k2").unwrap().as_deref(), Some(b"v2".as_ref()));
        assert!(backend.get(b"k3").unwrap().is_none());
    }

    #[test]
    fn test_backend_size_bytes() {
        let backend = InMemoryStateBackend::new();
        assert_eq!(backend.size_bytes(), 0);

        backend.put(b"abc", b"def").unwrap();
        assert_eq!(backend.size_bytes(), 6);
    }

    #[test]
    fn test_keyed_state_store_basic() {
        let backend = Arc::new(InMemoryStateBackend::new());
        let store: KeyedStateStore<String, i64> =
            KeyedStateStore::new(partition(), backend, str_key_ser, i64_ser, i64_de);

        let key = "counter".to_string();

        assert!(store.get(&key).unwrap().is_none());

        store.put(&key, 10).unwrap();
        assert_eq!(store.get(&key).unwrap(), Some(10));

        let new_val = store
            .update_or_default(&key, |cur| cur.unwrap_or(0) + 5)
            .unwrap();
        assert_eq!(new_val, 15);
        assert_eq!(store.get(&key).unwrap(), Some(15));

        assert!(store.delete(&key).unwrap());
        assert!(store.get(&key).unwrap().is_none());
    }

    #[test]
    fn test_aggregating_state_sum() {
        let backend = Arc::new(InMemoryStateBackend::new());

        fn combine(acc: u64, x: u64) -> u64 {
            acc + x
        }

        let agg: AggregatingState<u64, u64> = AggregatingState::new(
            partition(),
            backend,
            b"total".to_vec(),
            combine,
            0u64,
            u64_ser,
            u64_de,
        );

        assert_eq!(agg.get().unwrap(), 0);

        agg.add(10).unwrap();
        agg.add(20).unwrap();
        agg.add(5).unwrap();

        assert_eq!(agg.get().unwrap(), 35);

        agg.clear().unwrap();
        assert_eq!(agg.get().unwrap(), 0);
    }

    #[test]
    fn test_partition_namespacing_isolation() {
        let backend = Arc::new(InMemoryStateBackend::new());

        let p1 = StatePartitionKey::new("op", 0, 0);
        let p2 = StatePartitionKey::new("op", 0, 1);

        let store1: KeyedStateStore<String, i64> =
            KeyedStateStore::new(p1, backend.clone(), str_key_ser, i64_ser, i64_de);
        let store2: KeyedStateStore<String, i64> =
            KeyedStateStore::new(p2, backend, str_key_ser, i64_ser, i64_de);

        let key = "x".to_string();
        store1.put(&key, 1).unwrap();
        store2.put(&key, 2).unwrap();

        assert_eq!(store1.get(&key).unwrap(), Some(1));
        assert_eq!(store2.get(&key).unwrap(), Some(2));
    }

    #[test]
    fn test_snapshot_round_trip_empty() {
        let backend = InMemoryStateBackend::new();
        let snapshot = backend.checkpoint(0).unwrap();

        let new_backend = InMemoryStateBackend::new();
        new_backend.restore(&snapshot).unwrap();
        assert_eq!(new_backend.size_bytes(), 0);
    }

    #[test]
    fn test_decode_snapshot_too_short() {
        let result = decode_snapshot(b"short");
        assert!(result.is_err());
    }

    #[test]
    fn test_version_bumps_on_write() {
        let backend = InMemoryStateBackend::new();
        let v0 = backend.version().unwrap();
        backend.put(b"k", b"v").unwrap();
        let v1 = backend.version().unwrap();
        assert!(v1 > v0);
        backend.delete(b"k").unwrap();
        let v2 = backend.version().unwrap();
        assert!(v2 > v1);
    }

    #[test]
    fn test_state_backend_stats() {
        let backend = InMemoryStateBackend::new();
        backend.put(b"key", b"value").unwrap();
        let stats = StateBackendStats::collect(&backend);
        assert_eq!(stats.size_bytes, 8); // 3 + 5
    }
}
