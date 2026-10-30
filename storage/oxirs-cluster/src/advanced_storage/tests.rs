//! Tests for [`super`]: WAL recovery, snapshots and encryption at rest.

use super::*;
use tempfile::TempDir;

async fn create_test_storage() -> (AdvancedStorageBackend, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig {
        data_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    let storage = AdvancedStorageBackend::new(config).await.unwrap();
    (storage, temp_dir)
}

#[tokio::test]
async fn test_raft_state_storage() {
    let (storage, _temp_dir) = create_test_storage().await;

    let state = RaftState {
        current_term: 1,
        voted_for: Some(1),
        log: vec![],
        commit_index: 0,
        last_applied: 0,
    };

    // Store state
    storage.store_raft_state(state.clone()).await.unwrap();

    // Load state
    let loaded_state = storage.load_raft_state().await.unwrap().unwrap();
    assert_eq!(state.current_term, loaded_state.current_term);
    assert_eq!(state.voted_for, loaded_state.voted_for);
}

#[tokio::test]
async fn test_snapshot_creation_and_loading() {
    let (storage, _temp_dir) = create_test_storage().await;

    let data = b"test snapshot data";
    let metadata = storage
        .create_snapshot(10, 1, vec![1, 2, 3], data)
        .await
        .unwrap();

    assert_eq!(metadata.last_included_index, 10);
    assert_eq!(metadata.last_included_term, 1);
    assert_eq!(metadata.configuration, vec![1, 2, 3]);

    // Load snapshot
    let loaded_data = storage.load_snapshot(10, 1).await.unwrap().unwrap();
    assert_eq!(&loaded_data, data);
}

#[tokio::test]
async fn test_cache_functionality() {
    let (storage, _temp_dir) = create_test_storage().await;

    let state = RaftState {
        current_term: 1,
        voted_for: Some(1),
        log: vec![],
        commit_index: 0,
        last_applied: 0,
    };

    // Store and load multiple times to test caching
    storage.store_raft_state(state.clone()).await.unwrap();

    for _ in 0..5 {
        let _loaded_state = storage.load_raft_state().await.unwrap().unwrap();
    }

    let stats = storage.stats().await;
    assert!(stats.cache_hit_ratio > 0.0);
}

#[tokio::test]
async fn test_corruption_detection() {
    let (storage, temp_dir) = create_test_storage().await;

    let data = b"test data";
    let _metadata = storage.create_snapshot(1, 1, vec![1], data).await.unwrap();

    // Corrupt the snapshot file
    let snapshot_path = temp_dir.path().join("snapshot-1-1.dat");
    let mut corrupted_data = fs::read(&snapshot_path).await.unwrap();
    corrupted_data[0] ^= 0xFF; // Flip bits to corrupt
    fs::write(&snapshot_path, corrupted_data).await.unwrap();

    // Loading should detect corruption
    let result = storage.load_snapshot(1, 1).await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("checksum verification failed"));
}

#[tokio::test]
async fn test_wal_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig {
        data_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    let state = RaftState {
        current_term: 5,
        voted_for: Some(2),
        log: vec![],
        commit_index: 0,
        last_applied: 0,
    };

    // Create storage and store state
    {
        let storage = AdvancedStorageBackend::new(config.clone()).await.unwrap();
        storage.store_raft_state(state.clone()).await.unwrap();
    }

    // Create new storage instance (simulates restart)
    let storage = AdvancedStorageBackend::new(config).await.unwrap();
    let recovered_state = storage.load_raft_state().await.unwrap().unwrap();

    assert_eq!(state.current_term, recovered_state.current_term);
    assert_eq!(state.voted_for, recovered_state.voted_for);
}

fn encrypted_config(dir: &Path, key: [u8; 32]) -> StorageConfig {
    StorageConfig {
        data_dir: dir.to_path_buf(),
        enable_encryption: true,
        encryption_key: Some(key),
        ..Default::default()
    }
}

fn sample_state(term: u64) -> RaftState {
    RaftState {
        current_term: term,
        voted_for: Some(2),
        log: vec![],
        commit_index: 0,
        last_applied: 0,
    }
}

#[tokio::test]
async fn test_encrypted_wal_recovers_with_same_key() {
    let temp_dir = TempDir::new().unwrap();
    let config = encrypted_config(temp_dir.path(), [7u8; 32]);
    {
        let storage = AdvancedStorageBackend::new(config.clone()).await.unwrap();
        storage.store_raft_state(sample_state(5)).await.unwrap();
    }

    let wal = std::fs::read(temp_dir.path().join("wal.log")).unwrap();
    assert!(at_rest::is_sealed(&wal[4..]), "WAL record is not sealed");

    let storage = AdvancedStorageBackend::new(config).await.unwrap();
    let recovered = storage.load_raft_state().await.unwrap().unwrap();
    assert_eq!(recovered.current_term, 5);
    assert_eq!(recovered.voted_for, Some(2));
}

#[tokio::test]
async fn test_encrypted_wal_rejects_other_key_and_mode_mismatch() {
    let temp_dir = TempDir::new().unwrap();
    {
        let storage = AdvancedStorageBackend::new(encrypted_config(temp_dir.path(), [7u8; 32]))
            .await
            .unwrap();
        storage.store_raft_state(sample_state(5)).await.unwrap();
    }

    // Another key must not silently recover an empty log.
    let other_key = AdvancedStorageBackend::new(encrypted_config(temp_dir.path(), [8u8; 32])).await;
    assert!(other_key.is_err());

    // Neither may a backend with encryption switched off.
    let plaintext_mode = AdvancedStorageBackend::new(StorageConfig {
        data_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    })
    .await;
    assert!(plaintext_mode.is_err());

    // And an existing plaintext WAL is not mixed with encrypted records.
    let plain_dir = TempDir::new().unwrap();
    {
        let storage = AdvancedStorageBackend::new(StorageConfig {
            data_dir: plain_dir.path().to_path_buf(),
            ..Default::default()
        })
        .await
        .unwrap();
        storage.store_raft_state(sample_state(1)).await.unwrap();
    }
    let upgraded = AdvancedStorageBackend::new(encrypted_config(plain_dir.path(), [7u8; 32])).await;
    assert!(upgraded.is_err());
}

#[tokio::test]
async fn test_encrypted_snapshot_round_trip_and_tamper_detection() {
    let temp_dir = TempDir::new().unwrap();
    let storage = AdvancedStorageBackend::new(encrypted_config(temp_dir.path(), [7u8; 32]))
        .await
        .unwrap();
    let data = b"snapshot payload with enough repetition repetition repetition";

    storage
        .create_snapshot(10, 2, vec![1, 2], data)
        .await
        .unwrap();
    let snapshot_path = temp_dir.path().join("snapshot-10-2.dat");
    let on_disk = std::fs::read(&snapshot_path).unwrap();
    assert!(at_rest::is_sealed(&on_disk));
    assert_eq!(
        storage.load_snapshot(10, 2).await.unwrap().unwrap(),
        data.to_vec()
    );

    let mut tampered = on_disk.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0x01;
    std::fs::write(&snapshot_path, &tampered).unwrap();
    assert!(storage.load_snapshot(10, 2).await.is_err());
}

#[tokio::test]
async fn test_encryption_settings_must_be_consistent() {
    let temp_dir = TempDir::new().unwrap();
    let flag_only = StorageConfig {
        data_dir: temp_dir.path().join("a"),
        enable_encryption: true,
        ..Default::default()
    };
    assert!(AdvancedStorageBackend::new(flag_only).await.is_err());

    let key_only = StorageConfig {
        data_dir: temp_dir.path().join("b"),
        encryption_key: Some([7u8; 32]),
        ..Default::default()
    };
    assert!(AdvancedStorageBackend::new(key_only).await.is_err());
}

#[test]
fn test_storage_config_debug_redacts_key() {
    let config = StorageConfig {
        enable_encryption: true,
        encryption_key: Some([0xAB; 32]),
        ..Default::default()
    };
    let printed = format!("{config:?}");
    assert!(printed.contains("<redacted>"));
    assert!(!printed.contains("171"), "key bytes leaked: {printed}");
}

// ---------------------------------------------------------------------------
// WAL recovery: compressed records, torn tails, snapshots
// ---------------------------------------------------------------------------

/// A Raft state whose WAL record is far above the 1 KiB compression threshold.
fn large_state(term: u64) -> RaftState {
    use crate::network::LogEntry;
    use crate::raft::RdfCommand;
    RaftState {
        current_term: term,
        voted_for: Some(3),
        log: (1..=64)
            .map(|i| {
                LogEntry::new(
                    i,
                    term,
                    RdfCommand::Insert {
                        subject: format!("http://example.org/subject/{i}"),
                        predicate: "http://example.org/predicate".to_string(),
                        object: format!("object number {i} with some padding text"),
                    },
                )
            })
            .collect(),
        commit_index: 64,
        last_applied: 64,
    }
}

fn plain_config(dir: &Path, enable_compression: bool) -> StorageConfig {
    StorageConfig {
        data_dir: dir.to_path_buf(),
        enable_compression,
        ..Default::default()
    }
}

async fn reopen(config: &StorageConfig) -> AdvancedStorageBackend {
    AdvancedStorageBackend::new(config.clone())
        .await
        .expect("storage must reopen")
}

async fn current_state(storage: &AdvancedStorageBackend) -> RaftState {
    storage
        .load_raft_state()
        .await
        .expect("load_raft_state")
        .expect("a state was stored")
}

/// Records over 1 KiB are LZ4-compressed, but recovery used to read every
/// record as uncompressed. It stopped at the first large one, so a restart
/// came back with a stale state, and every later append was lost as well.
#[tokio::test]
async fn test_wal_recovers_large_records_and_later_appends() {
    let dir = TempDir::new().unwrap();
    let config = plain_config(dir.path(), true);
    {
        let storage = reopen(&config).await;
        storage.store_raft_state(sample_state(1)).await.unwrap();
        storage.store_raft_state(large_state(2)).await.unwrap();
        storage.store_raft_state(sample_state(3)).await.unwrap();
    }
    {
        let storage = reopen(&config).await;
        assert_eq!(current_state(&storage).await.current_term, 3);
        storage.store_raft_state(large_state(4)).await.unwrap();
    }
    let storage = reopen(&config).await;
    let recovered = current_state(&storage).await;
    assert_eq!(recovered.current_term, 4);
    assert_eq!(recovered.log.len(), 64);
}

/// A MessagePack `WalEntry` record starts with an array marker, which can
/// never be mistaken for the LZ4 frame magic of a compressed record.
#[tokio::test]
async fn test_wal_record_framing_is_unambiguous() {
    let dir = TempDir::new().unwrap();
    {
        let storage = reopen(&plain_config(dir.path(), true)).await;
        storage.store_raft_state(sample_state(1)).await.unwrap();
        storage.store_raft_state(large_state(2)).await.unwrap();
    }
    let wal = std::fs::read(dir.path().join("wal.log")).unwrap();
    let first_len = u32::from_le_bytes(wal[..4].try_into().unwrap()) as usize;
    let first = &wal[4..4 + first_len];
    let second = &wal[8 + first_len..];
    assert_eq!(first[0], 0x94, "a small record is not a 4-element array");
    assert_eq!(
        second[..4],
        oxiarc_lz4::LZ4_FRAME_MAGIC.to_le_bytes(),
        "a large record is not an LZ4 frame"
    );
}

/// The record checksum used to cover the compressed bytes, so changing
/// `enable_compression` between restarts stopped recovery.
#[tokio::test]
async fn test_wal_recovers_across_a_compression_change() {
    let dir = TempDir::new().unwrap();
    {
        let storage = reopen(&plain_config(dir.path(), true)).await;
        storage.store_raft_state(large_state(2)).await.unwrap();
    }
    {
        let storage = reopen(&plain_config(dir.path(), false)).await;
        assert_eq!(current_state(&storage).await.current_term, 2);
        storage.store_raft_state(large_state(3)).await.unwrap();
    }
    let storage = reopen(&plain_config(dir.path(), true)).await;
    assert_eq!(current_state(&storage).await.current_term, 3);
}

/// WALs written by released versions (checksum over the compressed bytes of
/// large records) must still recover.
#[tokio::test]
async fn test_wal_written_by_earlier_releases_recovers() {
    use crate::serialization::CompressionAlgorithm;
    let dir = TempDir::new().unwrap();
    let mut serializer = MessageSerializer::with_config(SerializationConfig {
        compression: CompressionAlgorithm::Lz4,
        ..Default::default()
    });
    let mut wal = Vec::new();
    let states = [
        sample_state(1),
        large_state(2),
        sample_state(3),
        large_state(4),
    ];
    for (sequence, state) in (0u64..).zip(states) {
        // The append logic of releases up to 0.4.1.
        let entry = WalEntry {
            sequence,
            timestamp: 1_700_000_000,
            operation: WalOperation::WriteRaftState(state),
            checksum: String::new(),
        };
        let content = serializer.serialize(&entry).unwrap();
        let checksum = hex::encode(Sha256::digest(&content.payload));
        let record = serializer
            .serialize(&WalEntry { checksum, ..entry })
            .unwrap()
            .payload;
        wal.extend_from_slice(&(record.len() as u32).to_le_bytes());
        wal.extend_from_slice(&record);
    }
    std::fs::write(dir.path().join("wal.log"), &wal).unwrap();

    let storage = reopen(&plain_config(dir.path(), true)).await;
    let recovered = current_state(&storage).await;
    assert_eq!(recovered.current_term, 4);
    assert_eq!(recovered.log.len(), 64);
}

/// Names of the files that hold torn WAL tails set aside by recovery.
fn torn_tail_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("wal.log.torn-"))
        })
        .collect();
    files.sort();
    files
}

/// A crash in the middle of an append leaves a torn final record. Opening
/// the store used to fail outright; now the torn bytes are set aside and
/// the records written afterwards stay readable.
#[tokio::test]
async fn test_wal_torn_tail_is_set_aside() {
    let dir = TempDir::new().unwrap();
    let config = plain_config(dir.path(), true);
    {
        let storage = reopen(&config).await;
        storage.store_raft_state(sample_state(1)).await.unwrap();
        storage.store_raft_state(sample_state(2)).await.unwrap();
    }
    let wal_path = dir.path().join("wal.log");
    let good = std::fs::read(&wal_path).unwrap();
    let torn: Vec<u8> = [500u32.to_le_bytes().as_slice(), &[0x94, 1, 2, 3]].concat();
    std::fs::write(&wal_path, [good.as_slice(), &torn].concat()).unwrap();

    {
        let storage = reopen(&config).await;
        assert_eq!(current_state(&storage).await.current_term, 2);
        storage.store_raft_state(sample_state(3)).await.unwrap();
    }
    let set_aside = torn_tail_files(dir.path());
    assert_eq!(set_aside.len(), 1, "the torn bytes were not preserved");
    assert!(std::fs::read(&set_aside[0]).unwrap() == torn);

    let storage = reopen(&config).await;
    assert_eq!(current_state(&storage).await.current_term, 3);
}

/// A complete but damaged final record (a write that never reached the
/// disk intact) is treated like a torn tail.
#[tokio::test]
async fn test_wal_damaged_final_record_is_set_aside() {
    let dir = TempDir::new().unwrap();
    let config = plain_config(dir.path(), false);
    {
        let storage = reopen(&config).await;
        storage.store_raft_state(sample_state(1)).await.unwrap();
        storage.store_raft_state(sample_state(2)).await.unwrap();
    }
    let wal_path = dir.path().join("wal.log");
    let mut wal = std::fs::read(&wal_path).unwrap();
    let last = wal.len() - 1;
    wal[last] ^= 0xFF;
    std::fs::write(&wal_path, &wal).unwrap();

    let storage = reopen(&config).await;
    assert_eq!(current_state(&storage).await.current_term, 1);
    assert_eq!(torn_tail_files(dir.path()).len(), 1);
}

/// Flip one byte inside record `index` (0-based) of the WAL in `dir`.
fn damage_record(dir: &Path, index: usize) -> Vec<u8> {
    let wal_path = dir.join("wal.log");
    let mut wal = std::fs::read(&wal_path).unwrap();
    let mut offset = 0;
    for _ in 0..index {
        let len = u32::from_le_bytes(wal[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4 + len;
    }
    let len = u32::from_le_bytes(wal[offset..offset + 4].try_into().unwrap()) as usize;
    wal[offset + 4 + len / 2] ^= 0xFF;
    std::fs::write(&wal_path, &wal).unwrap();
    wal
}

/// A damaged record followed by more records is corruption, not a torn
/// write. Recovery used to stop there and silently drop everything after it.
#[tokio::test]
async fn test_wal_mid_log_corruption_is_an_error() {
    let dir = TempDir::new().unwrap();
    let config = plain_config(dir.path(), false);
    {
        let storage = reopen(&config).await;
        for term in 1..=3 {
            storage.store_raft_state(sample_state(term)).await.unwrap();
        }
    }
    let damaged = damage_record(dir.path(), 1);

    assert!(AdvancedStorageBackend::new(config).await.is_err());
    assert!(
        std::fs::read(dir.path().join("wal.log")).unwrap() == damaged,
        "recovery modified a WAL it could not read"
    );
    assert!(torn_tail_files(dir.path()).is_empty());
}

/// The same holds for encrypted records that fail authentication.
#[tokio::test]
async fn test_encrypted_wal_mid_log_tampering_is_an_error() {
    let dir = TempDir::new().unwrap();
    let config = encrypted_config(dir.path(), [7u8; 32]);
    {
        let storage = reopen(&config).await;
        for term in 1..=3 {
            storage.store_raft_state(sample_state(term)).await.unwrap();
        }
    }
    damage_record(dir.path(), 1);
    assert!(AdvancedStorageBackend::new(config).await.is_err());
}

/// Large records are compressed before they are sealed; both must be undone.
#[tokio::test]
async fn test_encrypted_wal_recovers_large_records() {
    let dir = TempDir::new().unwrap();
    let config = encrypted_config(dir.path(), [7u8; 32]);
    {
        let storage = reopen(&config).await;
        storage.store_raft_state(large_state(5)).await.unwrap();
        storage.store_raft_state(sample_state(6)).await.unwrap();
    }
    let storage = reopen(&config).await;
    assert_eq!(current_state(&storage).await.current_term, 6);
}

/// Snapshot metadata is recorded in the WAL, but recovery used to ignore it,
/// so no snapshot could be loaded after a restart.
#[tokio::test]
async fn test_snapshot_loads_after_restart() {
    let dir = TempDir::new().unwrap();
    let config = plain_config(dir.path(), true);
    let data = vec![42u8; 4096];
    {
        let storage = reopen(&config).await;
        storage
            .create_snapshot(7, 2, vec![1, 2], &data)
            .await
            .unwrap();
    }
    let storage = reopen(&config).await;
    let loaded = storage.load_snapshot(7, 2).await.unwrap();
    assert!(
        loaded == Some(data),
        "the snapshot did not load after a restart"
    );
}

/// A WAL record for `state` whose checksum is SHA-256 over
/// `checksum_input(content encoding)` and whose payload is
/// `encode(final encoding)`, to reproduce what earlier releases wrote.
fn crafted_record(
    sequence: u64,
    state: RaftState,
    checksum_input: impl Fn(&[u8]) -> Vec<u8>,
    encode: impl Fn(&[u8]) -> Vec<u8>,
) -> Vec<u8> {
    let entry = WalEntry {
        sequence,
        timestamp: 1_700_000_000,
        operation: WalOperation::WriteRaftState(state),
        checksum: String::new(),
    };
    let content = rmp_serde::to_vec(&entry).unwrap();
    let checksum = hex::encode(Sha256::digest(checksum_input(&content)));
    let payload = encode(&rmp_serde::to_vec(&WalEntry { checksum, ..entry }).unwrap());
    let mut record = (payload.len() as u32).to_le_bytes().to_vec();
    record.extend_from_slice(&payload);
    record
}

fn unchanged(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

/// oxiarc-lz4 releases before 0.3.6 compressed differently, so the checksum
/// that oxirs-cluster 0.2.1 to 0.3.x stored for a large record cannot be
/// reproduced. The LZ4 frame's own content checksum vouches for it instead.
#[tokio::test]
async fn test_wal_from_older_compressors_recovers_on_the_frame_checksum() {
    let dir = TempDir::new().unwrap();
    // A checksum over bytes that today's compressor does not produce.
    let other_compressor = |content: &[u8]| oxiarc_lz4::compress_hc(content).unwrap();
    let frame = |encoded: &[u8]| oxiarc_lz4::compress(encoded).unwrap();
    let wal = [
        crafted_record(0, large_state(1), other_compressor, frame),
        crafted_record(1, sample_state(2), unchanged, unchanged),
        crafted_record(2, large_state(3), other_compressor, frame),
    ]
    .concat();
    std::fs::write(dir.path().join("wal.log"), &wal).unwrap();

    let storage = reopen(&plain_config(dir.path(), true)).await;
    let recovered = current_state(&storage).await;
    assert_eq!(recovered.current_term, 3);
    assert_eq!(recovered.log.len(), 64);
}

/// Without a frame content checksum nothing vouches for such a record.
#[tokio::test]
async fn test_unverifiable_frame_without_content_checksum_is_damage() {
    let dir = TempDir::new().unwrap();
    let other_compressor = |content: &[u8]| oxiarc_lz4::compress_hc(content).unwrap();
    let unchecked_frame = |encoded: &[u8]| {
        oxiarc_lz4::compress_with_options(
            encoded,
            oxiarc_lz4::FrameDescriptor::new().with_content_checksum(false),
        )
        .unwrap()
    };
    let wal = [
        crafted_record(0, large_state(1), other_compressor, unchecked_frame),
        crafted_record(1, sample_state(2), unchanged, unchanged),
    ]
    .concat();
    std::fs::write(dir.path().join("wal.log"), &wal).unwrap();

    assert!(AdvancedStorageBackend::new(plain_config(dir.path(), true))
        .await
        .is_err());
}

/// Releases up to 0.2.0 compressed large records with lz4_flex, as a bare
/// LZ4 block behind a size prefix. Their checksums cannot be verified, so
/// such a log is refused with an explanation, wherever the record sits, and
/// left untouched.
#[tokio::test]
async fn test_wal_from_lz4_flex_releases_is_refused() {
    let dir = TempDir::new().unwrap();
    let size_prefixed_block = |bytes: &[u8]| {
        let mut out = (bytes.len() as u32).to_le_bytes().to_vec();
        out.extend(oxiarc_lz4::compress_block(bytes).unwrap());
        out
    };
    let wal = [
        crafted_record(0, sample_state(1), unchanged, unchanged),
        crafted_record(1, large_state(2), size_prefixed_block, size_prefixed_block),
    ]
    .concat();
    let wal_path = dir.path().join("wal.log");
    std::fs::write(&wal_path, &wal).unwrap();

    let error = match AdvancedStorageBackend::new(plain_config(dir.path(), true)).await {
        Ok(_) => panic!("a WAL with unverifiable lz4_flex records was replayed"),
        Err(e) => e.to_string(),
    };
    assert!(error.contains("0.2.0 or earlier"), "{error}");
    assert!(
        std::fs::read(&wal_path).unwrap() == wal,
        "the WAL was modified"
    );
    assert!(torn_tail_files(dir.path()).is_empty());
}

/// A snapshot comes back exactly as it was stored. Whether a snapshot was
/// compressed was not recorded, so loading tried to decompress every
/// snapshot and kept the result whenever that worked: with compression off,
/// a snapshot whose bytes happened to form an LZ4 frame came back
/// decompressed.
#[tokio::test]
async fn test_snapshot_bytes_come_back_as_stored() {
    let dir = TempDir::new().unwrap();
    let config = plain_config(dir.path(), false);
    let storage = reopen(&config).await;
    let data = oxiarc_lz4::compress(&[7u8; 2048]).unwrap();
    storage.create_snapshot(3, 1, vec![1], &data).await.unwrap();

    let loaded = storage.load_snapshot(3, 1).await.unwrap();
    assert!(loaded == Some(data), "the snapshot changed on the way back");
}

/// Every encoding round-trips, with and without encryption at rest: small
/// snapshots as one LZ4 frame, large ones as parallel LZ4 blocks, and data
/// that compression would not shrink as it is.
#[tokio::test]
async fn test_compressed_snapshots_round_trip() {
    let compressible_large: Vec<u8> = (0..3 * 1024 * 1024 + 123)
        .map(|i| (i % 251) as u8)
        .collect();
    let incompressible: Vec<u8> = (0..4096u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    let cases = [vec![42u8; 4096], compressible_large, incompressible];

    for encrypted in [false, true] {
        let dir = TempDir::new().unwrap();
        let config = if encrypted {
            StorageConfig {
                enable_compression: true,
                ..encrypted_config(dir.path(), [9u8; 32])
            }
        } else {
            plain_config(dir.path(), true)
        };
        let storage = reopen(&config).await;
        for (index, data) in cases.iter().enumerate() {
            let index = index as u64 + 1;
            storage
                .create_snapshot(index, 1, vec![1], data)
                .await
                .unwrap();
            let loaded = storage.load_snapshot(index, 1).await.unwrap();
            assert!(
                loaded.as_ref() == Some(data),
                "snapshot {index} (encrypted: {encrypted}) did not round-trip"
            );
        }
        // And after a restart.
        drop(storage);
        let storage = reopen(&config).await;
        for (index, data) in cases.iter().enumerate() {
            let loaded = storage.load_snapshot(index as u64 + 1, 1).await.unwrap();
            assert!(
                loaded.as_ref() == Some(data),
                "snapshot {index} after restart"
            );
        }
    }
}

/// Recovery decompresses an entry into at most `MAX_WAL_RECORD_SIZE` bytes.
/// A compressed entry larger than that could be written but never read back,
/// so it is journaled uncompressed instead.
#[test]
fn test_entry_too_large_to_decompress_is_journaled_uncompressed() {
    let entry = WalEntry {
        sequence: 1,
        timestamp: 0,
        operation: WalOperation::WriteRaftState(large_state(2)),
        checksum: String::new(),
    };
    let entry = WalEntry {
        checksum: wal_entry_checksum(&entry).unwrap(),
        ..entry
    };
    let mut serializer = MessageSerializer::with_config(SerializationConfig {
        compression: CompressionAlgorithm::Lz4,
        ..Default::default()
    });
    let lz4_magic = oxiarc_lz4::LZ4_FRAME_MAGIC.to_le_bytes();

    // Within the limit the compressed payload is kept.
    let serialized = serializer.serialize(&entry).unwrap();
    assert_ne!(serialized.compression, CompressionAlgorithm::None);
    let limit = serialized.original_size;
    let payload = journal_payload(serialized, limit, || {
        Err(anyhow!("must not re-encode an entry within the limit"))
    })
    .unwrap();
    assert!(payload.starts_with(&lz4_magic));

    // Beyond it the entry is written uncompressed, and it still reads back.
    let serialized = serializer.serialize(&entry).unwrap();
    let limit = serialized.original_size - 1;
    let payload = journal_payload(serialized, limit, || Ok(rmp_serde::to_vec(&entry)?)).unwrap();
    assert!(!payload.starts_with(&lz4_magic));
    let decoded = decode_wal_payload(&payload)
        .unwrap()
        .expect("the uncompressed entry decodes and verifies");
    assert_eq!(decoded.sequence, 1);
}
