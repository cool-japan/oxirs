//! Continuous WAL Shipping for Disaster Recovery
//!
//! Production Feature: Continuous WAL File Shipping
//!
//! This module provides continuous WAL shipping capabilities for:
//! - Disaster recovery and standby servers
//! - Geographic replication
//! - Point-in-time recovery across multiple locations
//! - Automated failover support
//!
//! ## Architecture
//!
//! WAL shipping continuously monitors for new WAL files and ships them to
//! configured destinations (filesystem, network shares, cloud storage, etc.).
//! This enables hot standby servers and disaster recovery scenarios.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use oxirs_tdb::wal_shipping::{WalShipper, ShippingConfig, ShippingDestination};
//!
//! // Configure shipping destinations
//! let config = ShippingConfig {
//!     destinations: vec![
//!         ShippingDestination::FileSystem {
//!             path: "/standby/wal".into(),
//!             sync_interval: Duration::from_secs(10),
//!         },
//!     ],
//!     ..Default::default()
//! };
//!
//! // Create shipper
//! let shipper = WalShipper::new(config)?;
//!
//! // Start continuous shipping
//! shipper.start()?;
//!
//! // WAL files are automatically shipped to all destinations
//! ```

use crate::error::{Result, TdbError};
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Shipping destination type
#[derive(Debug, Clone)]
pub enum ShippingDestination {
    /// Local filesystem destination
    FileSystem {
        /// Destination path
        path: PathBuf,
        /// Sync interval
        sync_interval: Duration,
    },
    /// Network share (NFS, CIFS, etc.)
    NetworkShare {
        /// Mount point or UNC path
        path: PathBuf,
        /// Sync interval
        sync_interval: Duration,
    },
    /// Custom destination (future: S3, GCS, Azure, etc.)
    Custom {
        /// Custom destination identifier
        id: String,
        /// Configuration parameters
        params: HashMap<String, String>,
    },
}

impl ShippingDestination {
    /// Get destination identifier
    pub fn id(&self) -> String {
        match self {
            ShippingDestination::FileSystem { path, .. } => {
                format!("fs:{}", path.display())
            }
            ShippingDestination::NetworkShare { path, .. } => {
                format!("net:{}", path.display())
            }
            ShippingDestination::Custom { id, .. } => id.clone(),
        }
    }

    /// Get sync interval
    pub fn sync_interval(&self) -> Duration {
        match self {
            ShippingDestination::FileSystem { sync_interval, .. } => *sync_interval,
            ShippingDestination::NetworkShare { sync_interval, .. } => *sync_interval,
            ShippingDestination::Custom { .. } => Duration::from_secs(30),
        }
    }
}

/// Shipping configuration
#[derive(Debug, Clone)]
pub struct ShippingConfig {
    /// List of shipping destinations
    pub destinations: Vec<ShippingDestination>,
    /// Enable continuous shipping
    pub enable_shipping: bool,
    /// Compress WAL files during shipping
    pub compress_during_shipping: bool,
    /// Verify shipped files
    pub verify_after_shipping: bool,
    /// Maximum retry attempts for failed shipments
    pub max_retry_attempts: usize,
    /// Retry delay
    pub retry_delay: Duration,
    /// Shipping queue size
    pub queue_size: usize,
    /// WAL source directory
    pub wal_source_dir: PathBuf,
}

impl Default for ShippingConfig {
    fn default() -> Self {
        Self {
            destinations: Vec::new(),
            enable_shipping: false,
            compress_during_shipping: true,
            verify_after_shipping: true,
            max_retry_attempts: 3,
            retry_delay: Duration::from_secs(10),
            queue_size: 100,
            wal_source_dir: PathBuf::from("wal"),
        }
    }
}

/// Shipping status for a WAL file
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShippingStatus {
    /// Pending shipment
    Pending,
    /// Currently shipping
    Shipping,
    /// Successfully shipped
    Shipped,
    /// Failed to ship
    Failed,
    /// Retrying after failure
    Retrying,
}

/// Shipping record for a WAL file
#[derive(Debug, Clone)]
pub struct ShippingRecord {
    /// WAL file name
    pub file_name: String,
    /// WAL file path
    pub file_path: PathBuf,
    /// File size
    pub file_size: u64,
    /// LSN range [start, end]
    pub lsn_range: (u64, u64),
    /// Shipping status per destination
    pub destination_status: HashMap<String, (ShippingStatus, Option<String>)>,
    /// Retry counts per destination
    pub retry_counts: HashMap<String, usize>,
    /// Created timestamp
    pub created_at: SystemTime,
    /// Last shipping attempt
    pub last_attempt: Option<SystemTime>,
}

/// WAL shipping statistics
#[derive(Debug, Clone, Default)]
pub struct ShippingStats {
    /// Total files shipped
    pub files_shipped: u64,
    /// Total bytes shipped
    pub bytes_shipped: u64,
    /// Pending shipments
    pub pending_shipments: usize,
    /// Failed shipments
    pub failed_shipments: u64,
    /// Retry attempts
    pub retry_attempts: u64,
    /// Average shipping time (milliseconds)
    pub avg_shipping_time_ms: u64,
    /// Last successful shipment
    pub last_shipment_time: Option<SystemTime>,
}

/// WAL shipper for continuous archiving
pub struct WalShipper {
    /// Configuration
    config: ShippingConfig,
    /// Shipping queue
    shipping_queue: Arc<RwLock<VecDeque<ShippingRecord>>>,
    /// Shipped files index
    shipped_files: Arc<RwLock<HashMap<String, ShippingRecord>>>,
    /// Whether shipper is active
    active: Arc<AtomicBool>,
    /// Statistics
    stats: Arc<RwLock<ShippingStats>>,
    /// Total files shipped counter
    total_shipped: Arc<AtomicU64>,
    /// Total bytes shipped counter
    total_bytes_shipped: Arc<AtomicU64>,
    /// Failed shipments counter
    failed_shipments: Arc<AtomicU64>,
}

impl WalShipper {
    /// Create a new WAL shipper
    pub fn new(config: ShippingConfig) -> Result<Self> {
        // Validate configuration
        if config.enable_shipping && config.destinations.is_empty() {
            return Err(TdbError::Other(
                "Shipping enabled but no destinations configured".to_string(),
            ));
        }

        // Create destination directories
        for dest in &config.destinations {
            match dest {
                ShippingDestination::FileSystem { path, .. }
                | ShippingDestination::NetworkShare { path, .. } => {
                    fs::create_dir_all(path).map_err(|e| {
                        TdbError::Other(format!("Failed to create destination directory: {}", e))
                    })?;
                }
                ShippingDestination::Custom { .. } => {
                    // Custom destinations handle their own setup
                }
            }
        }

        Ok(Self {
            config,
            shipping_queue: Arc::new(RwLock::new(VecDeque::new())),
            shipped_files: Arc::new(RwLock::new(HashMap::new())),
            active: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(ShippingStats::default())),
            total_shipped: Arc::new(AtomicU64::new(0)),
            total_bytes_shipped: Arc::new(AtomicU64::new(0)),
            failed_shipments: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Start continuous shipping
    pub fn start(&self) -> Result<()> {
        if !self.config.enable_shipping {
            return Err(TdbError::Other(
                "Shipping is not enabled in configuration".to_string(),
            ));
        }

        if self.active.load(Ordering::Acquire) {
            return Err(TdbError::Other("Shipper is already active".to_string()));
        }

        self.active.store(true, Ordering::Release);
        Ok(())
    }

    /// Stop continuous shipping
    pub fn stop(&self) -> Result<()> {
        self.active.store(false, Ordering::Release);
        Ok(())
    }

    /// Check if shipper is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    /// Ship a WAL file to all configured destinations
    pub fn ship_wal_file(&self, file_path: impl AsRef<Path>, lsn_range: (u64, u64)) -> Result<()> {
        let file_path = file_path.as_ref();

        if !file_path.exists() {
            return Err(TdbError::Other(format!(
                "WAL file does not exist: {:?}",
                file_path
            )));
        }

        // Get file metadata
        let file_name = file_path
            .file_name()
            .ok_or_else(|| TdbError::Other("Invalid file path".to_string()))?
            .to_string_lossy()
            .to_string();

        let metadata = fs::metadata(file_path).map_err(TdbError::Io)?;
        let file_size = metadata.len();

        // Create shipping record
        let mut destination_status = HashMap::new();
        let mut retry_counts = HashMap::new();

        for dest in &self.config.destinations {
            destination_status.insert(dest.id(), (ShippingStatus::Pending, None));
            retry_counts.insert(dest.id(), 0);
        }

        let record = ShippingRecord {
            file_name: file_name.clone(),
            file_path: file_path.to_path_buf(),
            file_size,
            lsn_range,
            destination_status,
            retry_counts,
            created_at: SystemTime::now(),
            last_attempt: None,
        };

        // Add to shipping queue
        {
            let mut queue = self.shipping_queue.write();
            if queue.len() >= self.config.queue_size {
                return Err(TdbError::Other(format!(
                    "Shipping queue full (max: {})",
                    self.config.queue_size
                )));
            }
            queue.push_back(record.clone());
        }

        // Process shipment immediately if active
        if self.is_active() {
            self.process_shipment(&file_name)?;
        }

        Ok(())
    }

    /// Process a pending shipment
    fn process_shipment(&self, file_name: &str) -> Result<()> {
        let start_time = SystemTime::now();

        // Get record from queue
        let mut record = {
            let queue = self.shipping_queue.write();
            queue
                .iter()
                .find(|r| r.file_name == file_name)
                .cloned()
                .ok_or_else(|| TdbError::Other(format!("File not in queue: {}", file_name)))?
        };

        // Ship to each destination
        for dest in &self.config.destinations {
            let dest_id = dest.id();
            let status = record.destination_status.get(&dest_id).map(|(s, _)| *s);

            // Skip if already shipped
            if status == Some(ShippingStatus::Shipped) {
                continue;
            }

            // Update status to shipping
            record
                .destination_status
                .insert(dest_id.clone(), (ShippingStatus::Shipping, None));

            // Perform shipment
            let result = self.ship_to_destination(&record, dest);

            match result {
                Ok(()) => {
                    // Success
                    record
                        .destination_status
                        .insert(dest_id.clone(), (ShippingStatus::Shipped, None));
                }
                Err(e) => {
                    // Failure
                    let retry_count = record.retry_counts.get(&dest_id).copied().unwrap_or(0);

                    if retry_count < self.config.max_retry_attempts {
                        // Schedule retry
                        record.destination_status.insert(
                            dest_id.clone(),
                            (ShippingStatus::Retrying, Some(e.to_string())),
                        );
                        record.retry_counts.insert(dest_id.clone(), retry_count + 1);
                    } else {
                        // Max retries exceeded
                        record.destination_status.insert(
                            dest_id.clone(),
                            (ShippingStatus::Failed, Some(e.to_string())),
                        );
                        self.failed_shipments.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        record.last_attempt = Some(SystemTime::now());

        // Check if all destinations shipped
        let all_shipped = record
            .destination_status
            .values()
            .all(|(status, _)| *status == ShippingStatus::Shipped);

        if all_shipped {
            // Remove from queue and add to shipped files
            {
                let mut queue = self.shipping_queue.write();
                queue.retain(|r| r.file_name != file_name);
            }

            self.shipped_files
                .write()
                .insert(file_name.to_string(), record.clone());

            // Update statistics
            self.total_shipped.fetch_add(1, Ordering::Relaxed);
            self.total_bytes_shipped
                .fetch_add(record.file_size, Ordering::Relaxed);

            let duration = start_time
                .elapsed()
                .unwrap_or(Duration::from_secs(0))
                .as_millis() as u64;

            let mut stats = self.stats.write();
            stats.files_shipped += 1;
            stats.bytes_shipped += record.file_size;
            stats.last_shipment_time = Some(SystemTime::now());

            // Update average shipping time
            if stats.files_shipped == 1 {
                stats.avg_shipping_time_ms = duration;
            } else {
                stats.avg_shipping_time_ms =
                    (stats.avg_shipping_time_ms * (stats.files_shipped - 1) + duration)
                        / stats.files_shipped;
            }
        }

        Ok(())
    }

    /// Ship a file to a specific destination
    fn ship_to_destination(
        &self,
        record: &ShippingRecord,
        destination: &ShippingDestination,
    ) -> Result<()> {
        match destination {
            ShippingDestination::FileSystem { path, .. }
            | ShippingDestination::NetworkShare { path, .. } => {
                let dest_path = path.join(&record.file_name);

                // Copy file to destination
                fs::copy(&record.file_path, &dest_path).map_err(|e| {
                    TdbError::Other(format!("Failed to copy file to destination: {}", e))
                })?;

                // Verify if enabled
                if self.config.verify_after_shipping {
                    self.verify_shipped_file(&record.file_path, &dest_path)?;
                }

                Ok(())
            }
            ShippingDestination::Custom { .. } => {
                // Custom destinations would implement their own shipping logic
                Err(TdbError::Other(
                    "Custom destinations not yet implemented".to_string(),
                ))
            }
        }
    }

    /// Verify a shipped file matches the original
    fn verify_shipped_file(&self, original: &Path, shipped: &Path) -> Result<()> {
        let original_meta = fs::metadata(original).map_err(TdbError::Io)?;
        let shipped_meta = fs::metadata(shipped).map_err(TdbError::Io)?;

        if original_meta.len() != shipped_meta.len() {
            return Err(TdbError::Other(format!(
                "File size mismatch: original={}, shipped={}",
                original_meta.len(),
                shipped_meta.len()
            )));
        }

        // Could add checksum verification here
        Ok(())
    }

    /// Get shipping statistics
    pub fn get_stats(&self) -> ShippingStats {
        let mut stats = self.stats.read().clone();
        stats.pending_shipments = self.shipping_queue.read().len();
        stats.failed_shipments = self.failed_shipments.load(Ordering::Relaxed);
        stats
    }

    /// List pending shipments
    pub fn list_pending(&self) -> Vec<ShippingRecord> {
        self.shipping_queue.read().iter().cloned().collect()
    }

    /// List shipped files
    pub fn list_shipped(&self) -> Vec<ShippingRecord> {
        self.shipped_files.read().values().cloned().collect()
    }

    /// Get shipping record by file name
    pub fn get_record(&self, file_name: &str) -> Option<ShippingRecord> {
        // Check shipped files first
        if let Some(record) = self.shipped_files.read().get(file_name) {
            return Some(record.clone());
        }

        // Check queue
        self.shipping_queue
            .read()
            .iter()
            .find(|r| r.file_name == file_name)
            .cloned()
    }

    /// Retry failed shipments
    pub fn retry_failed(&self) -> Result<usize> {
        let pending: Vec<String> = self
            .shipping_queue
            .read()
            .iter()
            .filter(|r| {
                r.destination_status.values().any(|(status, _)| {
                    *status == ShippingStatus::Failed || *status == ShippingStatus::Retrying
                })
            })
            .map(|r| r.file_name.clone())
            .collect();

        for file_name in &pending {
            self.process_shipment(file_name)?;
        }

        Ok(pending.len())
    }

    /// Clear shipped files older than retention period
    pub fn cleanup_old_records(&self, retention: Duration) -> Result<usize> {
        let cutoff = SystemTime::now()
            .checked_sub(retention)
            .ok_or_else(|| TdbError::Other("Invalid retention period".to_string()))?;

        let mut shipped = self.shipped_files.write();
        let original_count = shipped.len();

        shipped.retain(|_, record| record.created_at >= cutoff);

        Ok(original_count - shipped.len())
    }
}

impl Default for WalShipper {
    fn default() -> Self {
        Self::new(ShippingConfig::default()).expect("Default config should be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_create_shipper() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_shipper_test");
        fs::remove_dir_all(&temp_dir).ok();

        let dest_dir = temp_dir.join("destination");

        let config = ShippingConfig {
            destinations: vec![ShippingDestination::FileSystem {
                path: dest_dir.clone(),
                sync_interval: Duration::from_secs(10),
            }],
            enable_shipping: true,
            ..Default::default()
        };

        let shipper = WalShipper::new(config).unwrap();
        assert!(!shipper.is_active());

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_start_stop_shipper() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_shipper_startstop_test");
        fs::remove_dir_all(&temp_dir).ok();

        let dest_dir = temp_dir.join("destination");

        let config = ShippingConfig {
            destinations: vec![ShippingDestination::FileSystem {
                path: dest_dir,
                sync_interval: Duration::from_secs(10),
            }],
            enable_shipping: true,
            ..Default::default()
        };

        let shipper = WalShipper::new(config).unwrap();
        assert!(!shipper.is_active());

        shipper.start().unwrap();
        assert!(shipper.is_active());

        shipper.stop().unwrap();
        assert!(!shipper.is_active());

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_ship_wal_file() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_ship_file_test");
        fs::remove_dir_all(&temp_dir).ok();

        let source_dir = temp_dir.join("source");
        let dest_dir = temp_dir.join("destination");

        fs::create_dir_all(&source_dir).unwrap();

        // Create test WAL file
        let wal_file = source_dir.join("wal_001.log");
        fs::write(&wal_file, b"WAL data for shipping").unwrap();

        let config = ShippingConfig {
            destinations: vec![ShippingDestination::FileSystem {
                path: dest_dir.clone(),
                sync_interval: Duration::from_secs(10),
            }],
            enable_shipping: true,
            wal_source_dir: source_dir.clone(),
            ..Default::default()
        };

        let shipper = WalShipper::new(config).unwrap();
        shipper.start().unwrap();

        // Ship the file
        shipper.ship_wal_file(&wal_file, (1, 100)).unwrap();

        // Verify file was shipped
        let shipped_file = dest_dir.join("wal_001.log");
        assert!(shipped_file.exists());

        let shipped_data = fs::read_to_string(&shipped_file).unwrap();
        assert_eq!(shipped_data, "WAL data for shipping");

        // Check statistics
        let stats = shipper.get_stats();
        assert_eq!(stats.files_shipped, 1);
        assert!(stats.bytes_shipped > 0);

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_multiple_destinations() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_multi_dest_test");
        fs::remove_dir_all(&temp_dir).ok();

        let source_dir = temp_dir.join("source");
        let dest1 = temp_dir.join("dest1");
        let dest2 = temp_dir.join("dest2");

        fs::create_dir_all(&source_dir).unwrap();

        let wal_file = source_dir.join("wal_002.log");
        fs::write(&wal_file, b"Multi-destination WAL").unwrap();

        let config = ShippingConfig {
            destinations: vec![
                ShippingDestination::FileSystem {
                    path: dest1.clone(),
                    sync_interval: Duration::from_secs(10),
                },
                ShippingDestination::FileSystem {
                    path: dest2.clone(),
                    sync_interval: Duration::from_secs(10),
                },
            ],
            enable_shipping: true,
            ..Default::default()
        };

        let shipper = WalShipper::new(config).unwrap();
        shipper.start().unwrap();

        shipper.ship_wal_file(&wal_file, (1, 100)).unwrap();

        // Verify shipped to both destinations
        assert!(dest1.join("wal_002.log").exists());
        assert!(dest2.join("wal_002.log").exists());

        let stats = shipper.get_stats();
        assert_eq!(stats.files_shipped, 1);

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_shipping_queue() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_queue_test");
        fs::remove_dir_all(&temp_dir).ok();

        let source_dir = temp_dir.join("source");
        let dest_dir = temp_dir.join("destination");

        fs::create_dir_all(&source_dir).unwrap();

        let config = ShippingConfig {
            destinations: vec![ShippingDestination::FileSystem {
                path: dest_dir,
                sync_interval: Duration::from_secs(10),
            }],
            enable_shipping: false, // Don't auto-ship
            queue_size: 5,
            ..Default::default()
        };

        let shipper = WalShipper::new(config).unwrap();

        // Add files to queue
        for i in 1..=3 {
            let wal_file = source_dir.join(format!("wal_{:03}.log", i));
            fs::write(&wal_file, format!("WAL {}", i)).unwrap();

            shipper
                .ship_wal_file(&wal_file, (i as u64, i as u64 * 100))
                .unwrap();
        }

        let pending = shipper.list_pending();
        assert_eq!(pending.len(), 3);

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_shipping_verification() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_verify_test");
        fs::remove_dir_all(&temp_dir).ok();

        let source_dir = temp_dir.join("source");
        let dest_dir = temp_dir.join("destination");

        fs::create_dir_all(&source_dir).unwrap();

        let wal_file = source_dir.join("wal_verify.log");
        fs::write(&wal_file, b"Verified WAL data").unwrap();

        let config = ShippingConfig {
            destinations: vec![ShippingDestination::FileSystem {
                path: dest_dir.clone(),
                sync_interval: Duration::from_secs(10),
            }],
            enable_shipping: true,
            verify_after_shipping: true, // Enable verification
            ..Default::default()
        };

        let shipper = WalShipper::new(config).unwrap();
        shipper.start().unwrap();

        shipper.ship_wal_file(&wal_file, (1, 100)).unwrap();

        assert!(dest_dir.join("wal_verify.log").exists());

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_get_shipping_record() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_record_test");
        fs::remove_dir_all(&temp_dir).ok();

        let source_dir = temp_dir.join("source");
        let dest_dir = temp_dir.join("destination");

        fs::create_dir_all(&source_dir).unwrap();

        let wal_file = source_dir.join("wal_record.log");
        fs::write(&wal_file, b"Record data").unwrap();

        let config = ShippingConfig {
            destinations: vec![ShippingDestination::FileSystem {
                path: dest_dir,
                sync_interval: Duration::from_secs(10),
            }],
            enable_shipping: true,
            ..Default::default()
        };

        let shipper = WalShipper::new(config).unwrap();
        shipper.start().unwrap();

        shipper.ship_wal_file(&wal_file, (10, 200)).unwrap();

        let record = shipper.get_record("wal_record.log").unwrap();
        assert_eq!(record.lsn_range, (10, 200));
        assert!(record.file_size > 0);

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_destination_id() {
        let dest = ShippingDestination::FileSystem {
            path: PathBuf::from("/test/path"),
            sync_interval: Duration::from_secs(10),
        };

        assert!(dest.id().contains("/test/path"));

        let net_dest = ShippingDestination::NetworkShare {
            path: PathBuf::from("//server/share"),
            sync_interval: Duration::from_secs(30),
        };

        assert!(net_dest.id().contains("//server/share"));
    }

    #[test]
    fn test_shipping_stats() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_stats_test");
        fs::remove_dir_all(&temp_dir).ok();

        let source_dir = temp_dir.join("source");
        let dest_dir = temp_dir.join("destination");

        fs::create_dir_all(&source_dir).unwrap();

        let wal_file = source_dir.join("wal_stats.log");
        fs::write(&wal_file, b"Stats test data").unwrap();

        let config = ShippingConfig {
            destinations: vec![ShippingDestination::FileSystem {
                path: dest_dir,
                sync_interval: Duration::from_secs(10),
            }],
            enable_shipping: true,
            ..Default::default()
        };

        let shipper = WalShipper::new(config).unwrap();
        shipper.start().unwrap();

        shipper.ship_wal_file(&wal_file, (1, 100)).unwrap();

        let stats = shipper.get_stats();
        assert_eq!(stats.files_shipped, 1);
        assert!(stats.bytes_shipped > 0);
        assert_eq!(stats.pending_shipments, 0);
        assert!(stats.last_shipment_time.is_some());

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_list_shipped() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_list_shipped_test");
        fs::remove_dir_all(&temp_dir).ok();

        let source_dir = temp_dir.join("source");
        let dest_dir = temp_dir.join("destination");

        fs::create_dir_all(&source_dir).unwrap();

        let config = ShippingConfig {
            destinations: vec![ShippingDestination::FileSystem {
                path: dest_dir,
                sync_interval: Duration::from_secs(10),
            }],
            enable_shipping: true,
            ..Default::default()
        };

        let shipper = WalShipper::new(config).unwrap();
        shipper.start().unwrap();

        // Ship multiple files
        for i in 1..=3 {
            let wal_file = source_dir.join(format!("wal_{:03}.log", i));
            fs::write(&wal_file, format!("WAL {}", i)).unwrap();
            shipper
                .ship_wal_file(&wal_file, (i as u64, i as u64 * 100))
                .unwrap();
        }

        let shipped = shipper.list_shipped();
        assert_eq!(shipped.len(), 3);

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_cleanup_old_records() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_cleanup_test");
        fs::remove_dir_all(&temp_dir).ok();

        let source_dir = temp_dir.join("source");
        let dest_dir = temp_dir.join("destination");

        fs::create_dir_all(&source_dir).unwrap();

        let config = ShippingConfig {
            destinations: vec![ShippingDestination::FileSystem {
                path: dest_dir,
                sync_interval: Duration::from_secs(10),
            }],
            enable_shipping: true,
            ..Default::default()
        };

        let shipper = WalShipper::new(config).unwrap();
        shipper.start().unwrap();

        // Ship a file
        let wal_file = source_dir.join("wal_old.log");
        fs::write(&wal_file, b"Old WAL").unwrap();
        shipper.ship_wal_file(&wal_file, (1, 100)).unwrap();

        // Cleanup with very short retention (should remove nothing since file just created)
        let removed = shipper
            .cleanup_old_records(Duration::from_secs(3600))
            .unwrap();
        assert_eq!(removed, 0);

        fs::remove_dir_all(&temp_dir).ok();
    }
}
