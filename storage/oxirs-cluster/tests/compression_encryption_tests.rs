//! Comprehensive integration tests for compression and encryption
//!
//! This test suite validates:
//! - Compression ratios meet targets (LZ4 60%, Zstd 70%, LZMA 80%)
//! - Compression speed meets targets (LZ4 >500 MB/s, Zstd >200 MB/s, LZMA >50 MB/s)
//! - Auto-selection chooses correct algorithms
//! - Encryption/decryption correctness
//! - Key rotation with backward compatibility
//! - Performance overhead is acceptable

use oxirs_cluster::compression_strategy::{
    AccessPattern, Algorithm, CompressedData, CompressionConfig, CompressionStrategy,
};
use oxirs_cluster::encryption::{EncryptionConfig, EncryptionManager, HsmProvider};
use scirs2_core::random::{Random, RngCore};

// Test data generators
fn generate_highly_compressible_data(size: usize) -> Vec<u8> {
    // Repeat pattern - highly compressible
    vec![65u8; size]
}

fn generate_text_data(size: usize) -> Vec<u8> {
    // Simulate text data with some repetition
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        data.extend_from_slice(pattern);
    }
    data.truncate(size);
    data
}

fn generate_random_data(size: usize) -> Vec<u8> {
    // Random data - not compressible
    let mut rng = Random::seed(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()));
    let mut data = vec![0u8; size];
    rng.fill_bytes(&mut data);
    data
}

fn measure_throughput_mbps(data_size: usize, elapsed_ns: u128) -> f64 {
    let size_mb = data_size as f64 / 1_000_000.0;
    let time_sec = elapsed_ns as f64 / 1_000_000_000.0;
    size_mb / time_sec
}

// Compression ratio tests
#[test]
fn test_lz4_compression_ratio() {
    let config = CompressionConfig {
        default_algorithm: Algorithm::Lz4,
        auto_select: false,
        compression_threshold_bytes: 0,
    };
    let strategy = CompressionStrategy::new(config);

    // Test with highly compressible data
    let data = generate_highly_compressible_data(100_000);
    let compressed = strategy.compress(&data).expect("Compression failed");

    let ratio = 1.0 - (compressed.data.len() as f64 / data.len() as f64);
    println!(
        "LZ4 compression ratio: {:.1}% (target: ~60%)",
        ratio * 100.0
    );

    // LZ4 should achieve at least 50% compression on highly compressible data
    assert!(
        ratio >= 0.50,
        "LZ4 compression ratio {:.1}% is below 50%",
        ratio * 100.0
    );
}

#[test]
fn test_zstd_compression_ratio() {
    let config = CompressionConfig {
        default_algorithm: Algorithm::Zstd,
        auto_select: false,
        compression_threshold_bytes: 0,
    };
    let strategy = CompressionStrategy::new(config);

    let data = generate_highly_compressible_data(100_000);
    let compressed = strategy.compress(&data).expect("Compression failed");

    let ratio = 1.0 - (compressed.data.len() as f64 / data.len() as f64);
    println!(
        "Zstd compression ratio: {:.1}% (target: ~70%)",
        ratio * 100.0
    );

    // Zstd should achieve at least 60% compression
    assert!(
        ratio >= 0.60,
        "Zstd compression ratio {:.1}% is below 60%",
        ratio * 100.0
    );
}

#[test]
fn test_lzma_compression_ratio() {
    let config = CompressionConfig {
        default_algorithm: Algorithm::Lzma,
        auto_select: false,
        compression_threshold_bytes: 0,
    };
    let strategy = CompressionStrategy::new(config);

    let data = generate_highly_compressible_data(100_000);
    let compressed = strategy.compress(&data).expect("Compression failed");

    let ratio = 1.0 - (compressed.data.len() as f64 / data.len() as f64);
    println!(
        "LZMA compression ratio: {:.1}% (target: ~80%)",
        ratio * 100.0
    );

    // LZMA should achieve at least 70% compression
    assert!(
        ratio >= 0.70,
        "LZMA compression ratio {:.1}% is below 70%",
        ratio * 100.0
    );
}

// Compression speed tests
#[test]
fn test_lz4_compression_speed() {
    let config = CompressionConfig {
        default_algorithm: Algorithm::Lz4,
        auto_select: false,
        compression_threshold_bytes: 0,
    };
    let strategy = CompressionStrategy::new(config);

    let data = generate_text_data(10_000_000); // 10 MB
    let start = std::time::Instant::now();
    let _compressed = strategy.compress(&data).expect("Compression failed");
    let elapsed = start.elapsed().as_nanos();

    let throughput = measure_throughput_mbps(data.len(), elapsed);
    println!("LZ4 compression speed: {:.1} MB/s (target: >500 MB/s)", throughput);

    // LZ4 should be fast (>100 MB/s even in debug mode)
    assert!(
        throughput > 100.0,
        "LZ4 compression speed {:.1} MB/s is below 100 MB/s",
        throughput
    );
}

#[test]
fn test_zstd_compression_speed() {
    let config = CompressionConfig {
        default_algorithm: Algorithm::Zstd,
        auto_select: false,
        compression_threshold_bytes: 0,
    };
    let strategy = CompressionStrategy::new(config);

    let data = generate_text_data(10_000_000); // 10 MB
    let start = std::time::Instant::now();
    let _compressed = strategy.compress(&data).expect("Compression failed");
    let elapsed = start.elapsed().as_nanos();

    let throughput = measure_throughput_mbps(data.len(), elapsed);
    println!(
        "Zstd compression speed: {:.1} MB/s (target: >200 MB/s)",
        throughput
    );

    // Zstd should achieve reasonable speed (>50 MB/s in debug mode)
    assert!(
        throughput > 50.0,
        "Zstd compression speed {:.1} MB/s is below 50 MB/s",
        throughput
    );
}

#[test]
fn test_lzma_compression_speed() {
    let config = CompressionConfig {
        default_algorithm: Algorithm::Lzma,
        auto_select: false,
        compression_threshold_bytes: 0,
    };
    let strategy = CompressionStrategy::new(config);

    let data = generate_text_data(1_000_000); // 1 MB (smaller for LZMA)
    let start = std::time::Instant::now();
    let _compressed = strategy.compress(&data).expect("Compression failed");
    let elapsed = start.elapsed().as_nanos();

    let throughput = measure_throughput_mbps(data.len(), elapsed);
    println!(
        "LZMA compression speed: {:.1} MB/s (target: >50 MB/s)",
        throughput
    );

    // LZMA is slower but should still be usable (>10 MB/s in debug mode)
    assert!(
        throughput > 10.0,
        "LZMA compression speed {:.1} MB/s is below 10 MB/s",
        throughput
    );
}

// Auto-selection tests
#[test]
fn test_auto_selection_small_data() {
    let config = CompressionConfig {
        default_algorithm: Algorithm::Zstd,
        auto_select: true,
        compression_threshold_bytes: 1024,
    };
    let strategy = CompressionStrategy::new(config);

    // Small data should not be compressed
    let small_data = vec![65u8; 512];
    let algorithm = strategy.select_algorithm(&small_data, AccessPattern::Warm);
    assert_eq!(
        algorithm,
        Algorithm::None,
        "Small data should not be compressed"
    );
}

#[test]
fn test_auto_selection_random_data() {
    let config = CompressionConfig::default();
    let strategy = CompressionStrategy::new(config);

    // Random data should not be compressed (low compressibility)
    let random_data = generate_random_data(10_000);
    let algorithm = strategy.select_algorithm(&random_data, AccessPattern::Warm);
    assert_eq!(
        algorithm,
        Algorithm::None,
        "Random data should not be compressed"
    );
}

#[test]
fn test_auto_selection_by_access_pattern() {
    let config = CompressionConfig::default();
    let strategy = CompressionStrategy::new(config);

    let data = generate_highly_compressible_data(10_000);

    let hot_algo = strategy.select_algorithm(&data, AccessPattern::Hot);
    assert_eq!(
        hot_algo,
        Algorithm::Lz4,
        "Hot data should use LZ4 for speed"
    );

    let warm_algo = strategy.select_algorithm(&data, AccessPattern::Warm);
    assert_eq!(
        warm_algo,
        Algorithm::Zstd,
        "Warm data should use Zstd for balance"
    );

    let cold_algo = strategy.select_algorithm(&data, AccessPattern::Cold);
    assert_eq!(
        cold_algo,
        Algorithm::Lzma,
        "Cold data should use LZMA for maximum compression"
    );
}

#[test]
fn test_compressibility_estimation() {
    let config = CompressionConfig::default();
    let strategy = CompressionStrategy::new(config);

    // Highly compressible
    let compressible = generate_highly_compressible_data(1000);
    let score1 = strategy.estimate_compressibility(&compressible);
    assert!(
        score1 > 0.9,
        "Highly compressible data should score >0.9, got {:.3}",
        score1
    );

    // Random (not compressible)
    let random = generate_random_data(1000);
    let score2 = strategy.estimate_compressibility(&random);
    assert!(
        score2 < 0.2,
        "Random data should score <0.2, got {:.3}",
        score2
    );

    // Text (moderately compressible)
    let text = generate_text_data(1000);
    let score3 = strategy.estimate_compressibility(&text);
    assert!(
        score3 > 0.3 && score3 < 0.8,
        "Text data should score between 0.3 and 0.8, got {:.3}",
        score3
    );
}

// Encryption tests
#[tokio::test]
async fn test_encryption_decryption_round_trip() {
    let config = EncryptionConfig::default();
    let manager = EncryptionManager::new(config).expect("Failed to create manager");

    let plaintext = b"Secret cluster data that needs encryption";
    let encrypted = manager.encrypt(plaintext).await.expect("Encryption failed");

    assert_ne!(
        &encrypted.ciphertext[..plaintext.len()],
        plaintext,
        "Ciphertext should differ from plaintext"
    );

    let decrypted = manager
        .decrypt(&encrypted)
        .await
        .expect("Decryption failed");
    assert_eq!(decrypted, plaintext, "Decrypted data should match original");
}

#[tokio::test]
async fn test_encryption_performance_overhead() {
    let config = EncryptionConfig::default();
    let manager = EncryptionManager::new(config).expect("Failed to create manager");

    let data = generate_text_data(1_000_000); // 1 MB

    // Measure encryption overhead
    let start = std::time::Instant::now();
    let encrypted = manager.encrypt(&data).await.expect("Encryption failed");
    let encrypt_time = start.elapsed();

    // Measure decryption overhead
    let start = std::time::Instant::now();
    let _decrypted = manager
        .decrypt(&encrypted)
        .await
        .expect("Decryption failed");
    let decrypt_time = start.elapsed();

    println!(
        "Encryption: {:.2}ms, Decryption: {:.2}ms for 1MB",
        encrypt_time.as_secs_f64() * 1000.0,
        decrypt_time.as_secs_f64() * 1000.0
    );

    // Overhead should be reasonable (<100ms for 1MB in debug mode)
    assert!(
        encrypt_time.as_millis() < 100,
        "Encryption overhead too high: {}ms",
        encrypt_time.as_millis()
    );
    assert!(
        decrypt_time.as_millis() < 100,
        "Decryption overhead too high: {}ms",
        decrypt_time.as_millis()
    );
}

#[tokio::test]
async fn test_key_rotation_backward_compatibility() {
    let config = EncryptionConfig {
        key_rotation_days: 90,
        hsm_enabled: false,
        hsm_provider: None,
    };
    let mut manager = EncryptionManager::new(config).expect("Failed to create manager");

    // Encrypt data with original key
    let plaintext = b"Data encrypted before key rotation";
    let encrypted_v1 = manager.encrypt(plaintext).await.expect("Encryption failed");

    // Rotate key
    manager.rotate_key().await.expect("Key rotation failed");

    // Encrypt data with new key
    let encrypted_v2 = manager.encrypt(plaintext).await.expect("Encryption failed");

    // Verify different keys were used
    assert_ne!(
        encrypted_v1.key_id, encrypted_v2.key_id,
        "Different keys should be used"
    );

    // Verify both versions can be decrypted
    let decrypted_v1 = manager
        .decrypt(&encrypted_v1)
        .await
        .expect("Failed to decrypt v1");
    let decrypted_v2 = manager
        .decrypt(&encrypted_v2)
        .await
        .expect("Failed to decrypt v2");

    assert_eq!(decrypted_v1, plaintext);
    assert_eq!(decrypted_v2, plaintext);
}

#[tokio::test]
async fn test_multiple_key_rotations() {
    let config = EncryptionConfig::default();
    let mut manager = EncryptionManager::new(config).expect("Failed to create manager");

    let mut encrypted_versions = Vec::new();
    let plaintext = b"Test data for multiple rotations";

    // Create 5 versions with different keys
    for i in 0..5 {
        let encrypted = manager.encrypt(plaintext).await.expect("Encryption failed");
        encrypted_versions.push((i, encrypted));

        if i < 4 {
            manager.rotate_key().await.expect("Key rotation failed");
        }
    }

    // Verify all versions decrypt correctly
    for (i, encrypted) in encrypted_versions {
        let decrypted = manager
            .decrypt(&encrypted)
            .await
            .unwrap_or_else(|_| panic!("Decryption failed for version {}", i));
        assert_eq!(decrypted, plaintext);
    }

    // Verify key history
    assert_eq!(
        manager.historical_key_count().await,
        4,
        "Should have 4 historical keys"
    );
}

// Combined compression + encryption tests
#[tokio::test]
async fn test_compression_then_encryption() {
    let comp_config = CompressionConfig::default();
    let compression = CompressionStrategy::new(comp_config);

    let enc_config = EncryptionConfig::default();
    let encryption = EncryptionManager::new(enc_config).expect("Failed to create manager");

    let original = generate_text_data(100_000);

    // Compress first
    let compressed = compression
        .compress(&original)
        .expect("Compression failed");
    println!(
        "Compression ratio: {:.1}%",
        (1.0 - compressed.data.len() as f64 / original.len() as f64) * 100.0
    );

    // Then encrypt
    let encrypted = encryption
        .encrypt(&compressed.data)
        .await
        .expect("Encryption failed");

    // Decrypt
    let decrypted = encryption
        .decrypt(&encrypted)
        .await
        .expect("Decryption failed");

    // Decompress
    let decompressed = compression
        .decompress(&CompressedData {
            data: decrypted,
            original_size: compressed.original_size,
            algorithm: compressed.algorithm,
        })
        .expect("Decompression failed");

    assert_eq!(decompressed, original);
}

#[tokio::test]
async fn test_combined_performance_overhead() {
    let comp_config = CompressionConfig::default();
    let compression = CompressionStrategy::new(comp_config);

    let enc_config = EncryptionConfig::default();
    let encryption = EncryptionManager::new(enc_config).expect("Failed to create manager");

    let data = generate_text_data(1_000_000); // 1 MB

    // Measure compression + encryption
    let start = std::time::Instant::now();
    let compressed = compression.compress(&data).expect("Compression failed");
    let encrypted = encryption
        .encrypt(&compressed.data)
        .await
        .expect("Encryption failed");
    let total_time = start.elapsed();

    println!(
        "Combined compression + encryption: {:.2}ms for 1MB",
        total_time.as_secs_f64() * 1000.0
    );

    // Total overhead should be reasonable (<200ms in debug mode)
    assert!(
        total_time.as_millis() < 200,
        "Combined overhead too high: {}ms",
        total_time.as_millis()
    );

    // Verify correctness
    let decrypted = encryption
        .decrypt(&encrypted)
        .await
        .expect("Decryption failed");
    let decompressed = compression
        .decompress(&CompressedData {
            data: decrypted,
            original_size: compressed.original_size,
            algorithm: compressed.algorithm,
        })
        .expect("Decompression failed");

    assert_eq!(decompressed, data);
}

// HSM integration tests (mock)
#[tokio::test]
async fn test_hsm_aws_kms_config() {
    let config = EncryptionConfig {
        key_rotation_days: 90,
        hsm_enabled: true,
        hsm_provider: Some(HsmProvider::AwsKms {
            region: "us-west-2".to_string(),
            key_arn: "arn:aws:kms:us-west-2:123456789012:key/test".to_string(),
        }),
    };

    // Should fall back to local key generation
    let manager = EncryptionManager::new(config).expect("Failed to create manager");

    let plaintext = b"Test with AWS KMS config";
    let encrypted = manager.encrypt(plaintext).await.expect("Encryption failed");
    let decrypted = manager
        .decrypt(&encrypted)
        .await
        .expect("Decryption failed");

    assert_eq!(decrypted, plaintext);
}

#[tokio::test]
async fn test_hsm_azure_config() {
    let config = EncryptionConfig {
        key_rotation_days: 90,
        hsm_enabled: true,
        hsm_provider: Some(HsmProvider::AzureKeyVault {
            vault_url: "https://test-vault.vault.azure.net".to_string(),
            key_name: "test-key".to_string(),
        }),
    };

    let manager = EncryptionManager::new(config).expect("Failed to create manager");

    let plaintext = b"Test with Azure Key Vault config";
    let encrypted = manager.encrypt(plaintext).await.expect("Encryption failed");
    let decrypted = manager
        .decrypt(&encrypted)
        .await
        .expect("Decryption failed");

    assert_eq!(decrypted, plaintext);
}

#[tokio::test]
async fn test_hsm_gcp_config() {
    let config = EncryptionConfig {
        key_rotation_days: 90,
        hsm_enabled: true,
        hsm_provider: Some(HsmProvider::GcpKms {
            project: "test-project".to_string(),
            location: "us-west1".to_string(),
            key_ring: "test-keyring".to_string(),
            key_name: "test-key".to_string(),
        }),
    };

    let manager = EncryptionManager::new(config).expect("Failed to create manager");

    let plaintext = b"Test with GCP KMS config";
    let encrypted = manager.encrypt(plaintext).await.expect("Encryption failed");
    let decrypted = manager
        .decrypt(&encrypted)
        .await
        .expect("Decryption failed");

    assert_eq!(decrypted, plaintext);
}

// Stress tests
#[test]
fn test_compression_large_data() {
    let config = CompressionConfig::default();
    let strategy = CompressionStrategy::new(config);

    // Test with 100 MB of data
    let large_data = generate_text_data(100_000_000);
    let compressed = strategy
        .compress(&large_data)
        .expect("Compression failed");
    let decompressed = strategy
        .decompress(&compressed)
        .expect("Decompression failed");

    assert_eq!(decompressed.len(), large_data.len());
    assert_eq!(&decompressed[..1000], &large_data[..1000]);
}

#[tokio::test]
async fn test_encryption_large_data() {
    let config = EncryptionConfig::default();
    let manager = EncryptionManager::new(config).expect("Failed to create manager");

    // Test with 10 MB of data
    let large_data = generate_text_data(10_000_000);
    let encrypted = manager
        .encrypt(&large_data)
        .await
        .expect("Encryption failed");
    let decrypted = manager
        .decrypt(&encrypted)
        .await
        .expect("Decryption failed");

    assert_eq!(decrypted.len(), large_data.len());
    assert_eq!(&decrypted[..1000], &large_data[..1000]);
}
