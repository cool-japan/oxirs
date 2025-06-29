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
use std::time::{SystemTime, UNIX_EPOCH};

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

/// File management configuration
#[derive(Debug, Clone)]
pub struct FileSystemConfig {
    pub create_if_missing: bool,
    pub sync_writes: bool,
    pub use_memory_mapping: bool,
    pub backup_on_startup: bool,
    pub max_file_handles: usize,
    pub page_size: usize,
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
