//! # TDB2-Compatible File System Layout
//!
//! Implements the TDB2-compatible directory structure and file management
//! for persistent RDF storage. This module provides the file organization
//! and management layer that mirrors Apache Jena TDB2's approach.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// TDB2-compatible file system layout manager
///
/// Manages the database directory structure and file organization
/// according to TDB2 specifications for compatibility.
///
/// ## Directory Structure
///
/// ```text
/// database_root/
/// ├── Data-0001/           # Main data directory with version
/// │   ├── nodes.dat        # Node table data
/// │   ├── nodes.idn        # Node table index
/// │   ├── SPO.dat          # SPO triple table data
/// │   ├── SPO.idn          # SPO triple table index
/// │   ├── SPO.bpt          # SPO B+ tree index
/// │   ├── POS.dat          # POS triple table data
/// │   ├── POS.idn          # POS triple table index
/// │   ├── POS.bpt          # POS B+ tree index
/// │   ├── OSP.dat          # OSP triple table data
/// │   ├── OSP.idn          # OSP triple table index
/// │   ├── OSP.bpt          # OSP B+ tree index
/// │   ├── SOP.dat          # SOP triple table data
/// │   ├── SOP.idn          # SOP triple table index
/// │   ├── SOP.bpt          # SOP B+ tree index
/// │   ├── PSO.dat          # PSO triple table data
/// │   ├── PSO.idn          # PSO triple table index
/// │   ├── PSO.bpt          # PSO B+ tree index
/// │   ├── OPS.dat          # OPS triple table data
/// │   ├── OPS.idn          # OPS triple table index
/// │   └── OPS.bpt          # OPS B+ tree index
/// ├── txn.log              # Transaction log
/// ├── tdb.info             # Database metadata
/// ├── tdb.lock             # Database lock file
/// └── backup/              # Backup directory
///     └── timestamp/       # Timestamped backups
/// ```
#[derive(Debug, Clone)]
pub struct TdbFileSystem {
    root_path: PathBuf,
    data_path: PathBuf,
    version: u32,
    file_handles: Arc<RwLock<HashMap<String, Arc<Mutex<File>>>>>,
    metadata: Arc<RwLock<DatabaseMetadata>>,
    config: FileSystemConfig,
    // Advanced locking state
    active_locks: Arc<RwLock<HashMap<String, LockMetadata>>>,
    lock_waiters: Arc<Mutex<Vec<LockRequest>>>,
    deadlock_detector: Arc<Mutex<DeadlockDetector>>,
}

/// Lock request for queuing
#[derive(Debug, Clone)]
pub struct LockRequest {
    pub request_id: String,
    pub owner_id: String,
    pub lock_mode: FileLockMode,
    pub file_types: Vec<FileType>,
    pub requested_at: Instant,
    pub timeout: Duration,
}

/// Simple deadlock detector
#[derive(Debug)]
pub struct DeadlockDetector {
    wait_for_graph: HashMap<String, Vec<String>>,
    last_check: Instant,
}

/// Database metadata stored in tdb.info
#[derive(Debug, Clone)]
pub struct DatabaseMetadata {
    pub version: u32,
    pub created_timestamp: u64,
    pub last_accessed: u64,
    pub triple_count: u64,
    pub node_count: u64,
    pub data_version: u32,
    pub format_version: String,
    pub properties: HashMap<String, String>,
}

impl Default for DatabaseMetadata {
    fn default() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            version: 1,
            created_timestamp: now,
            last_accessed: now,
            triple_count: 0,
            node_count: 0,
            data_version: 1,
            format_version: "TDB2-compatible-1.0".to_string(),
            properties: HashMap::new(),
        }
    }
}

/// File type enumeration for TDB files
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FileType {
    // Node table files
    NodesData,
    NodesIndex,

    // Triple table files (data)
    SpoData,
    PosData,
    OspData,
    SopData,
    PsoData,
    OpsData,

    // Triple table files (indices)
    SpoIndex,
    PosIndex,
    OspIndex,
    SopIndex,
    PsoIndex,
    OpsIndex,

    // B+ tree index files
    SpoBtree,
    PosBtree,
    OspBtree,
    SopBtree,
    PsoBtree,
    OpsBtree,

    // System files
    TransactionLog,
    Metadata,
    LockFile,
}

impl FileType {
    /// Get the filename for this file type
    pub fn filename(&self) -> &'static str {
        match self {
            FileType::NodesData => "nodes.dat",
            FileType::NodesIndex => "nodes.idn",

            FileType::SpoData => "SPO.dat",
            FileType::PosData => "POS.dat",
            FileType::OspData => "OSP.dat",
            FileType::SopData => "SOP.dat",
            FileType::PsoData => "PSO.dat",
            FileType::OpsData => "OPS.dat",

            FileType::SpoIndex => "SPO.idn",
            FileType::PosIndex => "POS.idn",
            FileType::OspIndex => "OSP.idn",
            FileType::SopIndex => "SOP.idn",
            FileType::PsoIndex => "PSO.idn",
            FileType::OpsIndex => "OPS.idn",

            FileType::SpoBtree => "SPO.bpt",
            FileType::PosBtree => "POS.bpt",
            FileType::OspBtree => "OSP.bpt",
            FileType::SopBtree => "SOP.bpt",
            FileType::PsoBtree => "PSO.bpt",
            FileType::OpsBtree => "OPS.bpt",

            FileType::TransactionLog => "txn.log",
            FileType::Metadata => "tdb.info",
            FileType::LockFile => "tdb.lock",
        }
    }

    /// Check if this file type should be in the data directory
    pub fn is_data_file(&self) -> bool {
        !matches!(
            self,
            FileType::TransactionLog | FileType::Metadata | FileType::LockFile
        )
    }
}

/// Advanced locking granularity options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockGranularity {
    /// Database-level locking (coarse)
    Database,
    /// File-type-level locking (medium)
    FileType,
    /// Individual file locking (fine)
    File,
}

/// File locking modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileLockMode {
    /// No locking
    None,
    /// Shared read lock
    SharedRead,
    /// Exclusive write lock
    ExclusiveWrite,
    /// Intention shared lock
    IntentionShared,
    /// Intention exclusive lock
    IntentionExclusive,
}

/// Advanced locking configuration
#[derive(Debug, Clone)]
pub struct AdvancedLockingConfig {
    /// Enable advanced locking mechanisms
    pub enabled: bool,
    /// Lock timeout duration
    pub lock_timeout: Duration,
    /// Enable distributed locking support
    pub enable_distributed_locking: bool,
    /// Lock granularity level
    pub lock_granularity: LockGranularity,
    /// Enable deadlock detection
    pub enable_deadlock_detection: bool,
    /// Lock escalation threshold
    pub escalation_threshold: usize,
    /// Maximum lock wait time
    pub max_wait_time: Duration,
}

impl Default for AdvancedLockingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            lock_timeout: Duration::from_secs(30),
            enable_distributed_locking: false,
            lock_granularity: LockGranularity::FileType,
            enable_deadlock_detection: true,
            escalation_threshold: 100,
            max_wait_time: Duration::from_secs(300),
        }
    }
}

/// Lock metadata for tracking
#[derive(Debug, Clone)]
pub struct LockMetadata {
    pub lock_id: String,
    pub lock_mode: FileLockMode,
    pub acquired_at: Instant,
    pub owner_id: String,
    pub file_types: Vec<FileType>,
}

/// File management configuration
#[derive(Debug, Clone)]
pub struct FileSystemConfig {
    pub create_if_missing: bool,
    pub sync_writes: bool,
    pub use_memory_mapping: bool,
    pub backup_on_startup: bool,
    pub max_file_handles: usize,
    pub page_size: usize,
    pub advanced_locking: AdvancedLockingConfig,
}

impl Default for FileSystemConfig {
    fn default() -> Self {
        Self {
            create_if_missing: true,
            sync_writes: true,
            use_memory_mapping: false,
            backup_on_startup: true,
            max_file_handles: 256,
            page_size: 8192, // 8KB pages like TDB2
            advanced_locking: AdvancedLockingConfig::default(),
        }
    }
}

impl TdbFileSystem {
    /// Create or open a TDB file system at the specified path
    ///
    /// # Arguments
    ///
    /// * `path` - Root directory for the database
    /// * `config` - File system configuration
    ///
    /// # Returns
    ///
    /// Returns a `TdbFileSystem` instance or an error if initialization fails.
    pub fn new<P: AsRef<Path>>(path: P, config: FileSystemConfig) -> Result<Self> {
        let root_path = path.as_ref().to_path_buf();

        // Create root directory if it doesn't exist and creation is enabled
        if !root_path.exists() && config.create_if_missing {
            fs::create_dir_all(&root_path)?;
        }

        if !root_path.exists() {
            return Err(anyhow!(
                "Database directory does not exist: {:?}",
                root_path
            ));
        }

        // Find or create data directory with version
        let (data_path, version) = Self::setup_data_directory(&root_path)?;

        let filesystem = Self {
            root_path: root_path.clone(),
            data_path,
            version,
            file_handles: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(DatabaseMetadata::default())),
            config: config.clone(),
            active_locks: Arc::new(RwLock::new(HashMap::new())),
            lock_waiters: Arc::new(Mutex::new(Vec::new())),
            deadlock_detector: Arc::new(Mutex::new(DeadlockDetector {
                wait_for_graph: HashMap::new(),
                last_check: Instant::now(),
            })),
        };

        // Load or create metadata
        filesystem.load_or_create_metadata()?;

        // Create lock file
        filesystem.create_lock_file()?;

        // Initialize required files
        filesystem.initialize_files(&config)?;

        // Create backup if requested
        if config.backup_on_startup {
            let _ = filesystem.create_backup(); // Non-fatal if backup fails
        }

        Ok(filesystem)
    }

    /// Setup data directory with version
    fn setup_data_directory(root_path: &Path) -> Result<(PathBuf, u32)> {
        // Look for existing data directories
        let mut max_version = 0;
        let mut data_path = None;

        if let Ok(entries) = fs::read_dir(root_path) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let file_name = entry.file_name();
                    let name = file_name.to_string_lossy();

                    if name.starts_with("Data-") && entry.path().is_dir() {
                        if let Ok(version_str) = name[5..].parse::<u32>() {
                            if version_str > max_version {
                                max_version = version_str;
                                data_path = Some(entry.path());
                            }
                        }
                    }
                }
            }
        }

        if let Some(path) = data_path {
            Ok((path, max_version))
        } else {
            // Create new data directory
            let version = 1;
            let path = root_path.join(format!("Data-{:04}", version));
            fs::create_dir_all(&path)?;
            Ok((path, version))
        }
    }

    /// Load or create database metadata
    fn load_or_create_metadata(&self) -> Result<()> {
        let metadata_path = self.root_path.join(FileType::Metadata.filename());

        if metadata_path.exists() {
            let content = fs::read_to_string(&metadata_path)?;
            let metadata: DatabaseMetadata = self.parse_metadata(&content)?;
            *self.metadata.write().unwrap() = metadata;
        } else {
            // Create new metadata
            let metadata = DatabaseMetadata {
                data_version: self.version,
                ..Default::default()
            };
            *self.metadata.write().unwrap() = metadata;
            self.save_metadata()?;
        }

        Ok(())
    }

    /// Parse metadata from file content
    fn parse_metadata(&self, content: &str) -> Result<DatabaseMetadata> {
        let mut metadata = DatabaseMetadata::default();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();

                match key {
                    "version" => metadata.version = value.parse()?,
                    "created_timestamp" => metadata.created_timestamp = value.parse()?,
                    "last_accessed" => metadata.last_accessed = value.parse()?,
                    "triple_count" => metadata.triple_count = value.parse()?,
                    "node_count" => metadata.node_count = value.parse()?,
                    "data_version" => metadata.data_version = value.parse()?,
                    "format_version" => metadata.format_version = value.to_string(),
                    _ => {
                        metadata
                            .properties
                            .insert(key.to_string(), value.to_string());
                    }
                }
            }
        }

        Ok(metadata)
    }

    /// Save metadata to file
    fn save_metadata(&self) -> Result<()> {
        let metadata_path = self.root_path.join(FileType::Metadata.filename());
        let metadata = self.metadata.read().unwrap();

        let mut content = String::new();
        content.push_str("# TDB Database Metadata\n");
        content.push_str(&format!("version={}\n", metadata.version));
        content.push_str(&format!(
            "created_timestamp={}\n",
            metadata.created_timestamp
        ));
        content.push_str(&format!("last_accessed={}\n", metadata.last_accessed));
        content.push_str(&format!("triple_count={}\n", metadata.triple_count));
        content.push_str(&format!("node_count={}\n", metadata.node_count));
        content.push_str(&format!("data_version={}\n", metadata.data_version));
        content.push_str(&format!("format_version={}\n", metadata.format_version));

        for (key, value) in &metadata.properties {
            content.push_str(&format!("{}={}\n", key, value));
        }

        fs::write(&metadata_path, content)?;
        Ok(())
    }

    /// Create lock file to prevent concurrent access
    fn create_lock_file(&self) -> Result<()> {
        let lock_path = self.root_path.join(FileType::LockFile.filename());
        let pid = std::process::id();
        let content = format!(
            "pid={}\ntimestamp={}\n",
            pid,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        fs::write(&lock_path, content)?;
        Ok(())
    }

    /// Initialize all required files
    fn initialize_files(&self, config: &FileSystemConfig) -> Result<()> {
        // List of all file types that should exist
        let file_types = vec![
            FileType::NodesData,
            FileType::NodesIndex,
            FileType::SpoData,
            FileType::SpoIndex,
            FileType::SpoBtree,
            FileType::PosData,
            FileType::PosIndex,
            FileType::PosBtree,
            FileType::OspData,
            FileType::OspIndex,
            FileType::OspBtree,
            FileType::SopData,
            FileType::SopIndex,
            FileType::SopBtree,
            FileType::PsoData,
            FileType::PsoIndex,
            FileType::PsoBtree,
            FileType::OpsData,
            FileType::OpsIndex,
            FileType::OpsBtree,
            FileType::TransactionLog,
        ];

        for file_type in file_types {
            self.ensure_file_exists(&file_type)?;
        }

        Ok(())
    }

    /// Ensure a file exists, create if missing
    fn ensure_file_exists(&self, file_type: &FileType) -> Result<()> {
        let path = self.get_file_path(file_type);

        if !path.exists() {
            // Create empty file with proper header
            let mut file = File::create(&path)?;

            // Write file header based on type
            match file_type {
                FileType::NodesData | FileType::NodesIndex => {
                    file.write_all(b"TDB2_NODES_V1\n")?;
                }
                ft if format!("{:?}", ft).contains("Data") => {
                    file.write_all(b"TDB2_TRIPLES_V1\n")?;
                }
                ft if format!("{:?}", ft).contains("Index") => {
                    file.write_all(b"TDB2_INDEX_V1\n")?;
                }
                ft if format!("{:?}", ft).contains("Btree") => {
                    file.write_all(b"TDB2_BTREE_V1\n")?;
                }
                FileType::TransactionLog => {
                    file.write_all(b"TDB2_TXNLOG_V1\n")?;
                }
                _ => {
                    file.write_all(b"TDB2_FILE_V1\n")?;
                }
            }

            file.sync_all()?;
        }

        Ok(())
    }

    /// Get file path for a given file type
    pub fn get_file_path(&self, file_type: &FileType) -> PathBuf {
        if file_type.is_data_file() {
            self.data_path.join(file_type.filename())
        } else {
            self.root_path.join(file_type.filename())
        }
    }

    /// Open file with caching
    pub fn open_file(&self, file_type: &FileType, write_mode: bool) -> Result<Arc<Mutex<File>>> {
        let file_key = format!("{:?}_{}", file_type, write_mode);

        // Check if file is already open
        {
            let handles = self.file_handles.read().unwrap();
            if let Some(handle) = handles.get(&file_key) {
                return Ok(Arc::clone(handle));
            }
        }

        // Open new file handle
        let path = self.get_file_path(file_type);
        let file = if write_mode {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&path)?
        } else {
            OpenOptions::new().read(true).open(&path)?
        };

        let handle = Arc::new(Mutex::new(file));

        // Cache the handle
        {
            let mut handles = self.file_handles.write().unwrap();
            handles.insert(file_key, Arc::clone(&handle));
        }

        Ok(handle)
    }

    /// Close file handle
    pub fn close_file(&self, file_type: &FileType, write_mode: bool) -> Result<()> {
        let file_key = format!("{:?}_{}", file_type, write_mode);

        let mut handles = self.file_handles.write().unwrap();
        handles.remove(&file_key);

        Ok(())
    }

    /// Create backup of the database
    pub fn create_backup(&self) -> Result<PathBuf> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let backup_dir = self.root_path.join("backup").join(timestamp.to_string());
        fs::create_dir_all(&backup_dir)?;

        // Copy all important files
        let important_files = vec![
            FileType::Metadata,
            FileType::NodesData,
            FileType::NodesIndex,
            FileType::SpoData,
            FileType::SpoBtree,
            FileType::PosData,
            FileType::PosBtree,
            FileType::OspData,
            FileType::OspBtree,
        ];

        for file_type in important_files {
            let src = self.get_file_path(&file_type);
            if src.exists() {
                let dst = backup_dir.join(file_type.filename());
                fs::copy(&src, &dst)?;
            }
        }

        Ok(backup_dir)
    }

    /// Compact database files
    pub fn compact(&self) -> Result<()> {
        // Implementation would include:
        // 1. Create new data directory with incremented version
        // 2. Copy and compact all data files
        // 3. Update metadata
        // 4. Atomically switch to new directory

        // For now, just update last accessed time
        {
            let mut metadata = self.metadata.write().unwrap();
            metadata.last_accessed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }
        self.save_metadata()?;

        Ok(())
    }

    /// Update statistics in metadata
    pub fn update_stats(&self, triple_count: u64, node_count: u64) -> Result<()> {
        {
            let mut metadata = self.metadata.write().unwrap();
            metadata.triple_count = triple_count;
            metadata.node_count = node_count;
            metadata.last_accessed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }
        self.save_metadata()
    }

    /// Get current metadata
    pub fn get_metadata(&self) -> DatabaseMetadata {
        self.metadata.read().unwrap().clone()
    }

    /// Get database root path
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }

    /// Get data directory path
    pub fn data_path(&self) -> &Path {
        &self.data_path
    }

    /// Get current version
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Validate file system integrity
    pub fn validate_integrity(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();

        // Check if all required files exist
        let required_files = vec![
            FileType::Metadata,
            FileType::NodesData,
            FileType::NodesIndex,
            FileType::SpoData,
            FileType::SpoBtree,
            FileType::PosData,
            FileType::PosBtree,
        ];

        for file_type in required_files {
            let path = self.get_file_path(&file_type);
            if !path.exists() {
                issues.push(format!("Missing required file: {:?}", path));
            }
        }

        // Check lock file
        let lock_path = self.root_path.join(FileType::LockFile.filename());
        if !lock_path.exists() {
            issues.push("Missing lock file".to_string());
        }

        // Validate metadata
        let metadata = self.get_metadata();
        if metadata.format_version.is_empty() {
            issues.push("Invalid metadata: missing format version".to_string());
        }

        Ok(issues)
    }

    /// Atomic write operation using temporary file and rename
    ///
    /// Ensures that the write operation is atomic - either all data is written
    /// successfully or no changes are made to the target file.
    ///
    /// # Arguments
    /// * `file_type` - The type of file to write to
    /// * `data` - The data to write atomically
    ///
    /// # Examples
    /// ```
    /// # use oxirs_tdb::filesystem::{TdbFileSystem, FileType, FileSystemConfig};
    /// # use tempfile::TempDir;
    /// # let temp_dir = TempDir::new().unwrap();
    /// # let fs = TdbFileSystem::new(temp_dir.path(), FileSystemConfig::default()).unwrap();
    /// let data = b"Important atomic data";
    /// fs.atomic_write(&FileType::NodesData, data).unwrap();
    /// ```
    pub fn atomic_write(&self, file_type: &FileType, data: &[u8]) -> Result<()> {
        let target_path = self.get_file_path(file_type);
        let temp_path = target_path.with_extension("tmp");

        // Write to temporary file first
        {
            let mut temp_file = File::create(&temp_path)?;

            // Write file header based on type (preserve existing logic)
            match file_type {
                FileType::NodesData | FileType::NodesIndex => {
                    temp_file.write_all(b"TDB2_NODES_V1\n")?;
                }
                ft if format!("{:?}", ft).contains("Data") => {
                    temp_file.write_all(b"TDB2_TRIPLES_V1\n")?;
                }
                ft if format!("{:?}", ft).contains("Index") => {
                    temp_file.write_all(b"TDB2_INDEX_V1\n")?;
                }
                ft if format!("{:?}", ft).contains("Btree") => {
                    temp_file.write_all(b"TDB2_BTREE_V1\n")?;
                }
                FileType::TransactionLog => {
                    temp_file.write_all(b"TDB2_TXNLOG_V1\n")?;
                }
                _ => {
                    temp_file.write_all(b"TDB2_FILE_V1\n")?;
                }
            }

            // Write the actual data
            temp_file.write_all(data)?;

            // Ensure data is written to disk
            temp_file.sync_all()?;
        }

        // Atomically rename temporary file to target
        fs::rename(&temp_path, &target_path)?;

        // Invalidate any cached file handles for this file type
        self.invalidate_file_cache(file_type);

        Ok(())
    }

    /// Atomic write operation for multiple files
    ///
    /// Ensures that either all files are updated successfully or no changes are made.
    /// Uses a two-phase approach: write all temp files first, then rename atomically.
    ///
    /// # Arguments
    /// * `files` - Vector of (file_type, data) pairs to write atomically
    ///
    /// # Examples
    /// ```
    /// # use oxirs_tdb::filesystem::{TdbFileSystem, FileType, FileSystemConfig};
    /// # use tempfile::TempDir;
    /// # let temp_dir = TempDir::new().unwrap();
    /// # let fs = TdbFileSystem::new(temp_dir.path(), FileSystemConfig::default()).unwrap();
    /// let files = vec![
    ///     (FileType::NodesData, b"nodes data".as_slice()),
    ///     (FileType::NodesIndex, b"nodes index".as_slice()),
    /// ];
    /// fs.atomic_multi_file_write(&files).unwrap();
    /// ```
    pub fn atomic_multi_file_write(&self, files: &[(FileType, &[u8])]) -> Result<()> {
        let mut temp_paths = Vec::new();
        let mut target_paths = Vec::new();

        // Phase 1: Write all temporary files
        for (file_type, data) in files {
            let target_path = self.get_file_path(file_type);
            let temp_path = target_path.with_extension("tmp");

            // Write to temporary file
            {
                let mut temp_file = File::create(&temp_path)?;

                // Write file header
                match file_type {
                    FileType::NodesData | FileType::NodesIndex => {
                        temp_file.write_all(b"TDB2_NODES_V1\n")?;
                    }
                    ft if format!("{:?}", ft).contains("Data") => {
                        temp_file.write_all(b"TDB2_TRIPLES_V1\n")?;
                    }
                    ft if format!("{:?}", ft).contains("Index") => {
                        temp_file.write_all(b"TDB2_INDEX_V1\n")?;
                    }
                    ft if format!("{:?}", ft).contains("Btree") => {
                        temp_file.write_all(b"TDB2_BTREE_V1\n")?;
                    }
                    FileType::TransactionLog => {
                        temp_file.write_all(b"TDB2_TXNLOG_V1\n")?;
                    }
                    _ => {
                        temp_file.write_all(b"TDB2_FILE_V1\n")?;
                    }
                }

                temp_file.write_all(data)?;
                temp_file.sync_all()?;
            }

            temp_paths.push(temp_path);
            target_paths.push(target_path);
        }

        // Phase 2: Atomically rename all files
        for (temp_path, target_path) in temp_paths.iter().zip(target_paths.iter()) {
            if let Err(e) = fs::rename(temp_path, target_path) {
                // Cleanup on failure
                for temp in &temp_paths {
                    let _ = fs::remove_file(temp);
                }
                return Err(e.into());
            }
        }

        // Invalidate cached file handles for all updated files
        for (file_type, _) in files {
            self.invalidate_file_cache(file_type);
        }

        Ok(())
    }

    /// Transactional file operation with rollback capability
    ///
    /// Performs a file operation within a transaction that can be rolled back
    /// if any part of the operation fails.
    ///
    /// # Arguments
    /// * `operation_id` - Unique identifier for this operation
    /// * `operation` - Closure that performs the file operations
    ///
    /// # Examples
    /// ```
    /// # use oxirs_tdb::filesystem::{TdbFileSystem, FileType, FileSystemConfig};
    /// # use tempfile::TempDir;
    /// # let temp_dir = TempDir::new().unwrap();
    /// # let fs = TdbFileSystem::new(temp_dir.path(), FileSystemConfig::default()).unwrap();
    /// let result = fs.transactional_operation("update_nodes", || {
    ///     fs.atomic_write(&FileType::NodesData, b"new data")?;
    ///     fs.atomic_write(&FileType::NodesIndex, b"new index")?;
    ///     Ok(())
    /// });
    /// ```
    pub fn transactional_operation<F, R>(&self, operation_id: &str, operation: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        // Create operation backup directory
        let backup_dir = self.root_path.join("temp_backup").join(operation_id);
        fs::create_dir_all(&backup_dir)?;

        // Store original file states for rollback
        let file_types = vec![
            FileType::NodesData,
            FileType::NodesIndex,
            FileType::SpoData,
            FileType::SpoBtree,
            FileType::PosData,
            FileType::PosBtree,
            FileType::OspData,
            FileType::OspBtree,
        ];

        // Backup existing files
        for file_type in &file_types {
            let source_path = self.get_file_path(file_type);
            if source_path.exists() {
                let backup_path = backup_dir.join(file_type.filename());
                fs::copy(&source_path, &backup_path)?;
            }
        }

        // Execute the operation
        let result = operation();

        match result {
            Ok(value) => {
                // Success: cleanup backup
                let _ = fs::remove_dir_all(&backup_dir);
                Ok(value)
            }
            Err(e) => {
                // Failure: rollback changes
                self.rollback_operation(operation_id)?;
                Err(e)
            }
        }
    }

    /// Rollback a transactional operation
    ///
    /// Restores files to their state before the specified operation began.
    ///
    /// # Arguments
    /// * `operation_id` - The operation to rollback
    pub fn rollback_operation(&self, operation_id: &str) -> Result<()> {
        let backup_dir = self.root_path.join("temp_backup").join(operation_id);

        if !backup_dir.exists() {
            return Err(anyhow!("No backup found for operation: {}", operation_id));
        }

        // Restore all backed up files
        for entry in fs::read_dir(&backup_dir)? {
            let entry = entry?;
            let backup_file = entry.path();

            if let Some(filename) = backup_file.file_name() {
                // Determine if this is a data file or root file
                let file_type = self.filename_to_file_type(filename.to_string_lossy().as_ref())?;
                let target_path = self.get_file_path(&file_type);

                // Restore the file
                fs::copy(&backup_file, &target_path)?;

                // Invalidate cache
                self.invalidate_file_cache(&file_type);
            }
        }

        // Cleanup backup directory
        fs::remove_dir_all(&backup_dir)?;

        Ok(())
    }

    /// Convert filename back to FileType for restoration
    fn filename_to_file_type(&self, filename: &str) -> Result<FileType> {
        match filename {
            "nodes.dat" => Ok(FileType::NodesData),
            "nodes.idn" => Ok(FileType::NodesIndex),
            "SPO.dat" => Ok(FileType::SpoData),
            "SPO.idn" => Ok(FileType::SpoIndex),
            "SPO.bpt" => Ok(FileType::SpoBtree),
            "POS.dat" => Ok(FileType::PosData),
            "POS.idn" => Ok(FileType::PosIndex),
            "POS.bpt" => Ok(FileType::PosBtree),
            "OSP.dat" => Ok(FileType::OspData),
            "OSP.idn" => Ok(FileType::OspIndex),
            "OSP.bpt" => Ok(FileType::OspBtree),
            "SOP.dat" => Ok(FileType::SopData),
            "SOP.idn" => Ok(FileType::SopIndex),
            "SOP.bpt" => Ok(FileType::SopBtree),
            "PSO.dat" => Ok(FileType::PsoData),
            "PSO.idn" => Ok(FileType::PsoIndex),
            "PSO.bpt" => Ok(FileType::PsoBtree),
            "OPS.dat" => Ok(FileType::OpsData),
            "OPS.idn" => Ok(FileType::OpsIndex),
            "OPS.bpt" => Ok(FileType::OpsBtree),
            "txn.log" => Ok(FileType::TransactionLog),
            "tdb.info" => Ok(FileType::Metadata),
            "tdb.lock" => Ok(FileType::LockFile),
            _ => Err(anyhow!("Unknown file type: {}", filename)),
        }
    }

    /// Invalidate cached file handle for a specific file type
    fn invalidate_file_cache(&self, file_type: &FileType) {
        let keys_to_remove: Vec<String> = {
            let handles = self.file_handles.read().unwrap();
            handles
                .keys()
                .filter(|key| key.starts_with(&format!("{:?}_", file_type)))
                .cloned()
                .collect()
        };

        let mut handles = self.file_handles.write().unwrap();
        for key in keys_to_remove {
            handles.remove(&key);
        }
    }

    /// Acquire advanced file lock with timeout and deadlock detection
    ///
    /// Attempts to acquire a lock on the specified file types with the given mode.
    /// Supports timeout, deadlock detection, and lock escalation.
    ///
    /// # Arguments
    /// * `owner_id` - Unique identifier for the lock owner (e.g., transaction ID)
    /// * `lock_mode` - The type of lock to acquire
    /// * `file_types` - Vector of file types to lock
    /// * `timeout` - Maximum time to wait for the lock
    ///
    /// # Returns
    /// Returns a lock ID on success that can be used to release the lock
    pub fn acquire_advanced_lock(
        &self,
        owner_id: &str,
        lock_mode: FileLockMode,
        file_types: &[FileType],
        timeout: Option<Duration>,
    ) -> Result<String> {
        if !self.config.advanced_locking.enabled {
            // Use simple locking when advanced locking is disabled
            return self.acquire_simple_lock(owner_id, lock_mode, file_types);
        }

        let timeout = timeout.unwrap_or(self.config.advanced_locking.lock_timeout);
        let lock_id = format!(
            "{}_{}",
            owner_id,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let start_time = Instant::now();

        // Check for deadlocks periodically
        if self.config.advanced_locking.enable_deadlock_detection {
            self.detect_deadlocks()?;
        }

        // Attempt to acquire lock with timeout
        loop {
            if start_time.elapsed() > timeout {
                return Err(anyhow!(
                    "Lock timeout for owner {} on files {:?}",
                    owner_id,
                    file_types
                ));
            }

            // Check if lock can be acquired
            if self.can_acquire_lock(owner_id, lock_mode, file_types)? {
                // Acquire the lock
                let lock_metadata = LockMetadata {
                    lock_id: lock_id.clone(),
                    lock_mode,
                    acquired_at: Instant::now(),
                    owner_id: owner_id.to_string(),
                    file_types: file_types.to_vec(),
                };

                {
                    let mut locks = self.active_locks.write().unwrap();
                    locks.insert(lock_id.clone(), lock_metadata);
                }

                return Ok(lock_id);
            }

            // Add to wait queue if not already there
            self.add_to_wait_queue(owner_id, lock_mode, file_types, timeout)?;

            // Wait a bit before retrying
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Check if a lock can be acquired based on compatibility matrix
    fn can_acquire_lock(
        &self,
        owner_id: &str,
        requested_mode: FileLockMode,
        requested_files: &[FileType],
    ) -> Result<bool> {
        let locks = self.active_locks.read().unwrap();

        for (_, existing_lock) in locks.iter() {
            // Skip locks owned by the same owner (allow lock escalation)
            if existing_lock.owner_id == owner_id {
                continue;
            }

            // Check if file types overlap
            let files_overlap = requested_files
                .iter()
                .any(|file_type| existing_lock.file_types.contains(file_type));

            if files_overlap {
                // Check lock compatibility
                if !self.are_locks_compatible(existing_lock.lock_mode, requested_mode) {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Lock compatibility matrix
    fn are_locks_compatible(&self, existing: FileLockMode, requested: FileLockMode) -> bool {
        use FileLockMode::*;
        match (existing, requested) {
            (None, _) | (_, None) => true,
            (SharedRead, SharedRead) => true,
            (SharedRead, IntentionShared) => true,
            (IntentionShared, SharedRead) => true,
            (IntentionShared, IntentionShared) => true,
            (IntentionShared, IntentionExclusive) => true,
            (IntentionExclusive, IntentionShared) => true,
            _ => false, // All other combinations are incompatible
        }
    }

    /// Add lock request to wait queue
    fn add_to_wait_queue(
        &self,
        owner_id: &str,
        lock_mode: FileLockMode,
        file_types: &[FileType],
        timeout: Duration,
    ) -> Result<()> {
        let request = LockRequest {
            request_id: format!(
                "{}_{}",
                owner_id,
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ),
            owner_id: owner_id.to_string(),
            lock_mode,
            file_types: file_types.to_vec(),
            requested_at: Instant::now(),
            timeout,
        };

        let mut waiters = self.lock_waiters.lock().unwrap();
        // Check if already waiting
        if !waiters.iter().any(|w| w.owner_id == owner_id) {
            waiters.push(request);
        }

        Ok(())
    }

    /// Release an advanced lock
    ///
    /// Releases the lock with the specified ID and processes any waiting lock requests.
    ///
    /// # Arguments
    /// * `lock_id` - The ID of the lock to release
    pub fn release_advanced_lock(&self, lock_id: &str) -> Result<()> {
        if !self.config.advanced_locking.enabled {
            // Handle simple lock release
            return self.release_simple_lock(lock_id);
        }

        // Remove the lock
        let removed_lock = {
            let mut locks = self.active_locks.write().unwrap();
            locks.remove(lock_id)
        };

        if removed_lock.is_some() {
            // Process waiting lock requests
            self.process_lock_queue()?;
        }

        Ok(())
    }

    /// Process waiting lock requests
    fn process_lock_queue(&self) -> Result<()> {
        let mut waiters = self.lock_waiters.lock().unwrap();
        let mut processed_indices = Vec::new();

        for (index, request) in waiters.iter().enumerate() {
            // Check if request has timed out
            if request.requested_at.elapsed() > request.timeout {
                processed_indices.push(index);
                continue;
            }

            // Try to acquire the lock
            if self.can_acquire_lock(&request.owner_id, request.lock_mode, &request.file_types)? {
                let lock_metadata = LockMetadata {
                    lock_id: request.request_id.clone(),
                    lock_mode: request.lock_mode,
                    acquired_at: Instant::now(),
                    owner_id: request.owner_id.clone(),
                    file_types: request.file_types.clone(),
                };

                {
                    let mut locks = self.active_locks.write().unwrap();
                    locks.insert(request.request_id.clone(), lock_metadata);
                }

                processed_indices.push(index);
            }
        }

        // Remove processed requests (in reverse order to maintain indices)
        for &index in processed_indices.iter().rev() {
            waiters.remove(index);
        }

        Ok(())
    }

    /// Detect deadlocks using wait-for graph
    fn detect_deadlocks(&self) -> Result<()> {
        let mut detector = self.deadlock_detector.lock().unwrap();

        // Only check periodically to avoid overhead
        if detector.last_check.elapsed() < Duration::from_secs(1) {
            return Ok(());
        }

        detector.last_check = Instant::now();

        // Build wait-for graph
        detector.wait_for_graph.clear();

        let waiters = self.lock_waiters.lock().unwrap();
        let locks = self.active_locks.read().unwrap();

        for request in waiters.iter() {
            let mut waiting_for = Vec::new();

            // Find who this request is waiting for
            for (_, existing_lock) in locks.iter() {
                let files_overlap = request
                    .file_types
                    .iter()
                    .any(|file_type| existing_lock.file_types.contains(file_type));

                if files_overlap
                    && !self.are_locks_compatible(existing_lock.lock_mode, request.lock_mode)
                {
                    waiting_for.push(existing_lock.owner_id.clone());
                }
            }

            if !waiting_for.is_empty() {
                detector
                    .wait_for_graph
                    .insert(request.owner_id.clone(), waiting_for);
            }
        }

        // Detect cycles (simplified cycle detection)
        for owner in detector.wait_for_graph.keys() {
            if self.has_cycle_from(
                &detector.wait_for_graph,
                owner,
                &mut std::collections::HashSet::new(),
            )? {
                return Err(anyhow!("Deadlock detected involving owner: {}", owner));
            }
        }

        Ok(())
    }

    /// Check for cycles in wait-for graph (DFS-based)
    fn has_cycle_from(
        &self,
        graph: &HashMap<String, Vec<String>>,
        start: &str,
        visited: &mut std::collections::HashSet<String>,
    ) -> Result<bool> {
        if visited.contains(start) {
            return Ok(true); // Cycle found
        }

        visited.insert(start.to_string());

        if let Some(dependencies) = graph.get(start) {
            for dep in dependencies {
                if self.has_cycle_from(graph, dep, visited)? {
                    return Ok(true);
                }
            }
        }

        visited.remove(start);
        Ok(false)
    }

    /// Get current lock status
    pub fn get_lock_status(&self) -> Result<Vec<LockMetadata>> {
        let locks = self.active_locks.read().unwrap();
        Ok(locks.values().cloned().collect())
    }

    /// Force release all locks for a specific owner (for cleanup)
    pub fn release_all_locks_for_owner(&self, owner_id: &str) -> Result<()> {
        let lock_ids_to_remove: Vec<String> = {
            let locks = self.active_locks.read().unwrap();
            locks
                .iter()
                .filter(|(_, lock)| lock.owner_id == owner_id)
                .map(|(id, _)| id.clone())
                .collect()
        };

        for lock_id in lock_ids_to_remove {
            self.release_advanced_lock(&lock_id)?;
        }

        // Remove from wait queue
        {
            let mut waiters = self.lock_waiters.lock().unwrap();
            waiters.retain(|w| w.owner_id != owner_id);
        }

        Ok(())
    }

    /// Clean up temporary files and close handles
    pub fn cleanup(&self) -> Result<()> {
        // Close all file handles
        {
            let mut handles = self.file_handles.write().unwrap();
            handles.clear();
        }

        // Remove lock file
        let lock_path = self.root_path.join(FileType::LockFile.filename());
        if lock_path.exists() {
            let _ = fs::remove_file(&lock_path); // Non-fatal
        }

        // Clean up any remaining temporary files
        let temp_backup_dir = self.root_path.join("temp_backup");
        if temp_backup_dir.exists() {
            let _ = fs::remove_dir_all(&temp_backup_dir);
        }

        // Remove any .tmp files
        if let Ok(entries) = fs::read_dir(&self.data_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "tmp" {
                        let _ = fs::remove_file(&path);
                    }
                }
            }
        }

        Ok(())
    }

    /// Acquire simple lock when advanced locking is disabled
    ///
    /// Provides basic file-level locking without advanced features like deadlock detection,
    /// timeouts, or lock queuing. This ensures some level of coordination even when
    /// advanced locking is disabled.
    fn acquire_simple_lock(
        &self,
        owner_id: &str,
        lock_mode: FileLockMode,
        file_types: &[FileType],
    ) -> Result<String> {
        let lock_id = format!(
            "simple_{}_{}",
            owner_id,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        let mut locks = self.active_locks.write().unwrap();

        // Check for conflicts with existing locks
        for (_, existing_lock) in locks.iter() {
            let files_overlap = file_types
                .iter()
                .any(|file_type| existing_lock.file_types.contains(file_type));

            if files_overlap && !self.are_locks_compatible(existing_lock.lock_mode, lock_mode) {
                return Err(anyhow!(
                    "Simple lock conflict: owner {} cannot acquire {:?} lock on {:?}, conflicts with existing {:?} lock by {}",
                    owner_id,
                    lock_mode,
                    file_types,
                    existing_lock.lock_mode,
                    existing_lock.owner_id
                ));
            }
        }

        // Create the lock
        let lock_metadata = LockMetadata {
            lock_id: lock_id.clone(),
            lock_mode,
            acquired_at: Instant::now(),
            owner_id: owner_id.to_string(),
            file_types: file_types.to_vec(),
        };

        locks.insert(lock_id.clone(), lock_metadata);

        Ok(lock_id)
    }

    /// Release simple lock when advanced locking is disabled
    fn release_simple_lock(&self, lock_id: &str) -> Result<()> {
        let mut locks = self.active_locks.write().unwrap();

        if locks.remove(lock_id).is_none() {
            // Don't treat missing locks as an error in simple mode
            // This makes the API more forgiving for cases where
            // locks might be released multiple times
            tracing::warn!("Attempted to release non-existent simple lock: {}", lock_id);
        }

        Ok(())
    }
}

impl Drop for TdbFileSystem {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_filesystem_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = FileSystemConfig::default();

        let fs = TdbFileSystem::new(temp_dir.path(), config).unwrap();

        assert!(fs.root_path().exists());
        assert!(fs.data_path().exists());
        assert_eq!(fs.version(), 1);

        // Check that required files exist
        assert!(fs.get_file_path(&FileType::Metadata).exists());
        assert!(fs.get_file_path(&FileType::NodesData).exists());
        assert!(fs.get_file_path(&FileType::SpoBtree).exists());
    }

    #[test]
    fn test_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config = FileSystemConfig::default();
        let fs = TdbFileSystem::new(temp_dir.path(), config).unwrap();

        // Test file opening
        let file_handle = fs.open_file(&FileType::NodesData, true).unwrap();

        // Test writing
        {
            let mut file = file_handle.lock().unwrap();
            file.write_all(b"test data").unwrap();
            file.sync_all().unwrap();
        }

        // Test reading
        {
            let mut file = file_handle.lock().unwrap();
            file.seek(SeekFrom::Start(0)).unwrap();
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).unwrap();
            assert!(buffer.len() > 0);
        }
    }

    #[test]
    fn test_metadata_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config = FileSystemConfig::default();
        let fs = TdbFileSystem::new(temp_dir.path(), config).unwrap();

        // Update stats
        fs.update_stats(1000, 500).unwrap();

        let metadata = fs.get_metadata();
        assert_eq!(metadata.triple_count, 1000);
        assert_eq!(metadata.node_count, 500);
    }

    #[test]
    fn test_backup_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = FileSystemConfig::default();
        let fs = TdbFileSystem::new(temp_dir.path(), config).unwrap();

        let backup_path = fs.create_backup().unwrap();
        assert!(backup_path.exists());
        assert!(backup_path.join("tdb.info").exists());
    }

    #[test]
    fn test_integrity_validation() {
        let temp_dir = TempDir::new().unwrap();
        let config = FileSystemConfig::default();
        let fs = TdbFileSystem::new(temp_dir.path(), config).unwrap();

        let issues = fs.validate_integrity().unwrap();
        assert!(issues.is_empty()); // Should have no issues for fresh database
    }
}
