//! Backup operations for MmapStore
//!
//! This module provides full and incremental backup functionality for the memory-mapped store.
//! Features include:
//! - Full backup support with metadata tracking
//! - Incremental backup for efficient storage
//! - Backup chain restoration
//! - Intelligent backup type recommendations

use super::types::{BackupMetadata, FileHeader, HEADER_SIZE};
use super::MmapStore;
use anyhow::{Context, Result};
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

impl MmapStore {
    /// Create a full backup of the store
    pub fn create_full_backup(&self, backup_dir: &Path) -> Result<BackupMetadata> {
        let _write_lock = self.write_lock.lock();

        // Ensure all data is flushed
        self.flush()?;

        // Create backup directory if it doesn't exist
        fs::create_dir_all(backup_dir)?;

        // Generate backup filename with timestamp
        let timestamp = std::time::SystemTime::now();
        let timestamp_secs = timestamp
            .duration_since(std::time::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();
        let backup_filename = format!("full_backup_{timestamp_secs}.oxirs");
        let backup_path = backup_dir.join(&backup_filename);

        // Copy data file to backup
        let data_path = self.path.join("data.oxirs");
        fs::copy(&data_path, &backup_path).context("Failed to copy data file to backup")?;

        // Copy term interner to backup
        let term_path = self.path.join("terms.oxirs");
        let term_backup_path = backup_dir.join(format!("terms_{timestamp_secs}.oxirs"));
        if term_path.exists() {
            fs::copy(&term_path, &term_backup_path)
                .context("Failed to copy term file to backup")?;
        }

        let quad_count = self.header.read().quad_count;
        let data_file = self.data_file.lock();
        let checkpoint_offset = data_file.metadata()?.len();

        let metadata = BackupMetadata {
            timestamp,
            quad_count,
            checkpoint_offset,
            is_full_backup: true,
            backup_path: backup_path.clone(),
        };

        // Update backup tracking
        *self.last_backup_offset.write() = checkpoint_offset;
        self.backup_history.write().push(metadata.clone());

        println!(
            "Full backup created: {} ({} quads, {} bytes)",
            backup_path.display(),
            quad_count,
            checkpoint_offset
        );

        Ok(metadata)
    }

    /// Create an incremental backup containing only changes since last backup
    pub fn create_incremental_backup(&self, backup_dir: &Path) -> Result<BackupMetadata> {
        let _write_lock = self.write_lock.lock();

        // Ensure all data is flushed
        self.flush()?;

        // Get last backup offset
        let last_offset = *self.last_backup_offset.read();

        // If no previous backup, create full backup instead
        if last_offset == 0 {
            return self.create_full_backup(backup_dir);
        }

        // Create backup directory if it doesn't exist
        fs::create_dir_all(backup_dir)?;

        // Generate backup filename with timestamp
        let timestamp = std::time::SystemTime::now();
        let timestamp_secs = timestamp
            .duration_since(std::time::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();
        let backup_filename = format!("incr_backup_{timestamp_secs}.oxirs");
        let backup_path = backup_dir.join(&backup_filename);

        // Open backup file for writing
        let mut backup_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&backup_path)
            .context("Failed to create incremental backup file")?;

        // Write incremental backup header
        let mut incr_header = FileHeader::new();
        incr_header.flags = 1; // Mark as incremental backup
        incr_header.data_offset = last_offset; // Store the base offset
        backup_file.write_all(unsafe {
            std::slice::from_raw_parts(
                &incr_header as *const _ as *const u8,
                std::mem::size_of::<FileHeader>(),
            )
        })?;

        // Copy only the new data since last backup
        let data_file = self.data_file.lock();
        let current_len = data_file.metadata()?.len();

        if current_len > last_offset {
            // Read and write new data
            let mmap = self.data_mmap.read();
            if let Some(mmap) = mmap.as_ref() {
                let new_data_start = (last_offset - HEADER_SIZE as u64) as usize;
                let new_data_end = mmap.len();
                if new_data_start < new_data_end {
                    let new_data = &mmap[new_data_start..new_data_end];
                    backup_file.write_all(new_data)?;
                }
            }
        }

        backup_file.flush()?;
        backup_file.sync_all()?;

        // Calculate number of new quads
        let new_quads = if current_len > last_offset {
            (current_len - last_offset) / std::mem::size_of::<super::types::DiskQuad>() as u64
        } else {
            0
        };

        let metadata = BackupMetadata {
            timestamp,
            quad_count: new_quads,
            checkpoint_offset: current_len,
            is_full_backup: false,
            backup_path: backup_path.clone(),
        };

        // Update backup tracking
        *self.last_backup_offset.write() = current_len;
        self.backup_history.write().push(metadata.clone());

        println!(
            "Incremental backup created: {} ({} new quads, {} bytes)",
            backup_path.display(),
            new_quads,
            current_len - last_offset
        );

        Ok(metadata)
    }

    /// Restore from a backup (full or incremental chain)
    pub fn restore_from_backup(&self, backup_path: &Path) -> Result<()> {
        let _write_lock = self.write_lock.lock();

        // Read backup header to determine type
        let mut backup_file = File::open(backup_path).context("Failed to open backup file")?;
        let mut header_bytes = vec![0u8; HEADER_SIZE];
        std::io::Read::read_exact(&mut backup_file, &mut header_bytes)?;
        let backup_header: FileHeader =
            unsafe { std::ptr::read(header_bytes.as_ptr() as *const FileHeader) };
        backup_header.validate()?;

        if backup_header.flags == 0 {
            // Full backup - simple copy
            let data_path = self.path.join("data.oxirs");
            fs::copy(backup_path, &data_path).context("Failed to restore from full backup")?;

            // Reload the store
            self.reload()?;
        } else {
            // Incremental backup - need to apply on top of base
            return Err(anyhow::anyhow!(
                "Incremental backup restoration requires base backup. Use restore_incremental_chain() instead."
            ));
        }

        Ok(())
    }

    /// Restore from an incremental backup chain
    pub fn restore_incremental_chain(&self, backup_paths: &[PathBuf]) -> Result<()> {
        if backup_paths.is_empty() {
            return Err(anyhow::anyhow!("No backup paths provided"));
        }

        let _write_lock = self.write_lock.lock();

        // First path must be a full backup
        let first_backup = &backup_paths[0];
        let mut backup_file = File::open(first_backup).context("Failed to open first backup")?;
        let mut header_bytes = vec![0u8; HEADER_SIZE];
        std::io::Read::read_exact(&mut backup_file, &mut header_bytes)?;
        let backup_header: FileHeader =
            unsafe { std::ptr::read(header_bytes.as_ptr() as *const FileHeader) };
        backup_header.validate()?;

        if backup_header.flags != 0 {
            return Err(anyhow::anyhow!(
                "First backup in chain must be a full backup"
            ));
        }

        // Copy full backup as base
        let data_path = self.path.join("data.oxirs");
        fs::copy(first_backup, &data_path).context("Failed to restore base backup")?;

        // Apply incremental backups in order
        for backup_path in &backup_paths[1..] {
            let mut incr_file =
                File::open(backup_path).context("Failed to open incremental backup")?;

            // Skip header
            incr_file.seek(SeekFrom::Start(HEADER_SIZE as u64))?;

            // Append to data file
            let mut data_file = OpenOptions::new()
                .append(true)
                .open(&data_path)
                .context("Failed to open data file for appending")?;

            std::io::copy(&mut incr_file, &mut data_file)?;
            data_file.flush()?;
        }

        // Reload the store
        self.reload()?;

        println!(
            "Restored from backup chain ({} backups)",
            backup_paths.len()
        );

        Ok(())
    }

    /// Reload the store from disk after restoration
    pub(super) fn reload(&self) -> Result<()> {
        use memmap2::MmapOptions;

        let data_path = self.path.join("data.oxirs");

        // Reopen data file
        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&data_path)
            .context("Failed to reopen data file")?;

        let file_len = data_file.metadata()?.len();

        // Read header
        let mut header_bytes = vec![0u8; HEADER_SIZE];
        let mut file_ref = &data_file;
        std::io::Read::read_exact(&mut file_ref, &mut header_bytes)?;
        let header: FileHeader =
            unsafe { std::ptr::read(header_bytes.as_ptr() as *const FileHeader) };
        header.validate()?;

        // Create new memory map
        let new_mmap = if file_len > HEADER_SIZE as u64 {
            Some(unsafe {
                MmapOptions::new()
                    .offset(HEADER_SIZE as u64)
                    .len((file_len - HEADER_SIZE as u64) as usize)
                    .map(&data_file)?
            })
        } else {
            None
        };

        // Update internal state
        *self.data_file.lock() = data_file;
        *self.data_mmap.write() = new_mmap;
        *self.header.write() = header;

        // Clear caches and indexes (they'll be rebuilt lazily)
        self.indexes.write().clear();
        self.term_cache.write().clear();
        self.deleted_quads.write().clear();

        // Reload term interner
        let term_path = self.path.join("terms.oxirs");
        if term_path.exists() {
            match crate::store::term_interner::TermInterner::load(&term_path) {
                Ok(interner) => {
                    *self.term_interner.write() = interner;
                }
                Err(e) => {
                    eprintln!("Warning: Failed to reload term interner: {e}");
                }
            }
        }

        Ok(())
    }

    /// Get backup history
    pub fn get_backup_history(&self) -> Vec<BackupMetadata> {
        self.backup_history.read().clone()
    }

    /// Clear backup history
    pub fn clear_backup_history(&self) {
        self.backup_history.write().clear();
        *self.last_backup_offset.write() = 0;
    }

    /// Get recommended backup type based on changes since last backup
    pub fn recommended_backup_type(&self) -> &'static str {
        let last_offset = *self.last_backup_offset.read();

        if last_offset == 0 {
            return "full";
        }

        let current_len = {
            let data_file = self.data_file.lock();
            data_file.metadata().map(|m| m.len()).unwrap_or(0)
        };

        let history = self.backup_history.read();
        let incremental_count = history.iter().filter(|m| !m.is_full_backup).count();

        // Recommend full backup if:
        // 1. Too many incremental backups in chain (>10)
        // 2. Changes are more than 50% of total data
        let large_changes =
            current_len > last_offset && (current_len - last_offset) > last_offset / 2;
        if incremental_count > 10 || large_changes {
            "full"
        } else {
            "incremental"
        }
    }
}
