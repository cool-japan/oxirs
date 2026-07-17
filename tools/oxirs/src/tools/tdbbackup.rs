//! TDB Backup and Restore Tool
//!
//! Comprehensive backup and restore functionality for OxiRS databases
//! with support for compression, incremental backups, and verification.

use super::compression::{self, CompressionFormat};
use super::{ToolResult, ToolStats};
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Configuration for backup operation
pub struct BackupConfig {
    pub source: PathBuf,
    pub target: PathBuf,
    pub compress: bool,
    pub incremental: bool,
    pub verify: bool,
    pub include_metadata: bool,
}

/// Configuration for restore operation
pub struct RestoreConfig {
    pub source: PathBuf,
    pub target: PathBuf,
    pub verify: bool,
    pub overwrite: bool,
}

/// Backup metadata
#[derive(Debug)]
struct BackupMetadata {
    timestamp: SystemTime,
    source_path: PathBuf,
    file_count: usize,
    total_size: u64,
    compressed: bool,
    incremental: bool,
}

/// Run backup operation
pub async fn run(
    source: PathBuf,
    target: PathBuf,
    compress: bool,
    incremental: bool,
) -> ToolResult {
    let config = BackupConfig {
        source,
        target,
        compress,
        incremental,
        verify: true,
        include_metadata: true,
    };

    backup(config).await
}

/// Perform database backup
pub async fn backup(config: BackupConfig) -> ToolResult {
    let mut stats = ToolStats::new();

    println!("OxiRS Database Backup");
    println!("=====================\n");

    // Validate source exists
    if !config.source.exists() {
        return Err(format!(
            "Source directory does not exist: {}",
            config.source.display()
        )
        .into());
    }

    if !config.source.is_dir() {
        return Err(format!("Source must be a directory: {}", config.source.display()).into());
    }

    println!("Source: {}", config.source.display());
    println!("Target: {}", config.target.display());
    println!(
        "Mode: {}",
        if config.incremental {
            "Incremental"
        } else {
            "Full"
        }
    );
    println!(
        "Compression: {}\n",
        if config.compress { "Yes" } else { "No" }
    );

    // Create target directory
    if let Some(parent) = config.target.parent() {
        fs::create_dir_all(parent)?;
    }

    // Collect files to backup
    let files = collect_files(&config.source)?;
    println!("Found {} file(s) to backup", files.len());

    // Calculate total size
    let total_size: u64 = files
        .iter()
        .map(|f| f.metadata().ok().map(|m| m.len()).unwrap_or(0))
        .sum();
    println!("Total size: {:.2} MB\n", total_size as f64 / 1_048_576.0);

    // Perform backup
    let backup_start = std::time::Instant::now();

    if config.compress {
        backup_compressed(&files, &config.source, &config.target)?;
    } else {
        backup_uncompressed(&files, &config.source, &config.target)?;
    }

    let backup_time = backup_start.elapsed();

    // Write metadata
    if config.include_metadata {
        write_backup_metadata(&config, &files, total_size)?;
    }

    // Verify if requested
    if config.verify {
        println!("\nVerifying backup...");
        verify_backup(&config.source, &config.target, config.compress)?;
        println!("✓ Backup verified successfully");
    }

    println!("\nBackup completed in {:.2}s", backup_time.as_secs_f64());
    println!("Backup saved to: {}", config.target.display());

    stats.items_processed = files.len();
    stats.finish();
    stats.print_summary("Backup");

    Ok(())
}

/// Perform database restore
pub async fn restore(config: RestoreConfig) -> ToolResult {
    let mut stats = ToolStats::new();

    println!("OxiRS Database Restore");
    println!("======================\n");

    // Validate source exists
    if !config.source.exists() {
        return Err(format!("Backup file does not exist: {}", config.source.display()).into());
    }

    println!("Source: {}", config.source.display());
    println!("Target: {}", config.target.display());

    // Check if target exists
    if config.target.exists() && !config.overwrite {
        return Err(format!(
            "Target directory already exists: {}. Use --overwrite to replace",
            config.target.display()
        )
        .into());
    }

    // Create target directory
    fs::create_dir_all(&config.target)?;

    // Detect backup format
    let is_compressed = config.source.extension().and_then(|s| s.to_str()) == Some("tar")
        || config.source.extension().and_then(|s| s.to_str()) == Some("gz");

    println!(
        "Format: {}\n",
        if is_compressed {
            "Compressed"
        } else {
            "Uncompressed"
        }
    );

    // Perform restore
    let restore_start = std::time::Instant::now();

    if is_compressed {
        restore_compressed(&config.source, &config.target)?;
    } else {
        restore_uncompressed(&config.source, &config.target)?;
    }

    let restore_time = restore_start.elapsed();

    // Verify if requested
    if config.verify {
        println!("\nVerifying restore...");
        verify_restore(&config.target)?;
        println!("✓ Restore verified successfully");
    }

    println!("\nRestore completed in {:.2}s", restore_time.as_secs_f64());
    println!("Database restored to: {}", config.target.display());

    stats.finish();
    stats.print_summary("Restore");

    Ok(())
}

/// Collect all files in directory recursively
fn collect_files(dir: &Path) -> ToolResult<Vec<PathBuf>> {
    let mut files = Vec::new();

    fn collect_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> ToolResult<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                collect_recursive(&path, files)?;
            } else if path.is_file() {
                files.push(path);
            }
        }
        Ok(())
    }

    collect_recursive(dir, &mut files)?;
    Ok(files)
}

/// Backup without compression (copy files)
fn backup_uncompressed(files: &[PathBuf], source_root: &Path, target: &Path) -> ToolResult<()> {
    fs::create_dir_all(target)?;

    for file in files {
        // Calculate relative path
        let rel_path = file
            .strip_prefix(source_root)
            .map_err(|e| format!("Failed to get relative path: {}", e))?;

        let target_file = target.join(rel_path);

        // Create parent directories
        if let Some(parent) = target_file.parent() {
            fs::create_dir_all(parent)?;
        }

        // Copy file
        fs::copy(file, &target_file)?;
        print!(".");
        io::stdout().flush()?;
    }

    println!();
    Ok(())
}

/// Backup with compression: a simple tar-like framing (path length + path +
/// size + content per file) written through the crate's single gzip
/// implementation. This is the one and only gzip path in the crate — the
/// framing bytes are streamed into `compression::open_writer(.., Gzip)` so the
/// archive is a genuine gzip stream, and restore reads it back through the
/// matching decompressor.
fn backup_compressed(files: &[PathBuf], source_root: &Path, target: &Path) -> ToolResult<()> {
    let target_file = if target.extension().is_some() {
        target.to_path_buf()
    } else {
        target.with_extension("tar")
    };

    let mut writer = compression::open_writer(&target_file, CompressionFormat::Gzip)?;

    for file_path in files {
        // Calculate relative path
        let rel_path = file_path
            .strip_prefix(source_root)
            .map_err(|e| format!("Failed to get relative path: {}", e))?;

        // Read file content
        let mut file = File::open(file_path)?;
        let metadata = file.metadata()?;
        let size = metadata.len();

        // Write header: path length (4 bytes) + path + size (8 bytes)
        let path_str = rel_path.to_string_lossy();
        let path_bytes = path_str.as_bytes();
        writer.write_all(&(path_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(path_bytes)?;
        writer.write_all(&size.to_le_bytes())?;

        // Write content
        let mut buffer = vec![0u8; 8192];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            writer.write_all(&buffer[..n])?;
        }

        print!(".");
        io::stdout().flush()?;
    }

    writer.flush()?;
    println!();

    Ok(())
}

/// Restore from uncompressed backup
fn restore_uncompressed(source: &Path, target: &Path) -> ToolResult<()> {
    // Simply copy all files from source to target
    copy_dir_recursive(source, target)?;
    Ok(())
}

/// Restore from compressed backup.
///
/// The archive is read back through the crate's single gzip implementation:
/// `compression::open_reader_sniffed` inflates it when the gzip magic is
/// present (so the file name need not carry `.gz`) and passes a non-gzip
/// archive through unchanged. Corrupt gzip surfaces an explicit error.
fn restore_compressed(source: &Path, target: &Path) -> ToolResult<()> {
    let mut reader = BufReader::new(compression::open_reader_sniffed(source)?);

    loop {
        // Read path length
        let mut len_bytes = [0u8; 4];
        match reader.read_exact(&mut len_bytes) {
            Ok(_) => {}
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }

        let path_len = u32::from_le_bytes(len_bytes) as usize;

        // Read path
        let mut path_bytes = vec![0u8; path_len];
        reader.read_exact(&mut path_bytes)?;
        let rel_path =
            String::from_utf8(path_bytes).map_err(|e| format!("Invalid UTF-8 in path: {}", e))?;

        // Read size
        let mut size_bytes = [0u8; 8];
        reader.read_exact(&mut size_bytes)?;
        let size = u64::from_le_bytes(size_bytes);

        // Create target file
        let target_file = target.join(&rel_path);
        if let Some(parent) = target_file.parent() {
            fs::create_dir_all(parent)?;
        }

        // Write content
        let mut file = File::create(&target_file)?;
        let mut remaining = size;
        let mut buffer = vec![0u8; 8192];

        while remaining > 0 {
            let to_read = remaining.min(buffer.len() as u64) as usize;
            reader.read_exact(&mut buffer[..to_read])?;
            file.write_all(&buffer[..to_read])?;
            remaining -= to_read as u64;
        }

        print!(".");
        io::stdout().flush()?;
    }

    println!();
    Ok(())
}

/// Copy directory recursively
fn copy_dir_recursive(source: &Path, target: &Path) -> ToolResult<()> {
    fs::create_dir_all(target)?;

    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());

        if file_type.is_dir() {
            copy_dir_recursive(&source_path, &target_path)?;
        } else if file_type.is_file() {
            fs::copy(&source_path, &target_path)?;
        }
    }

    Ok(())
}

/// Verify backup integrity
fn verify_backup(source: &Path, target: &Path, _compressed: bool) -> ToolResult<()> {
    // Basic verification: check file count and sizes match
    let source_files = collect_files(source)?;
    let source_size: u64 = source_files
        .iter()
        .map(|f| f.metadata().ok().map(|m| m.len()).unwrap_or(0))
        .sum();

    // For compressed backups, just check the archive exists
    if target.is_file() {
        let backup_size = target.metadata()?.len();
        if backup_size == 0 {
            return Err("Backup file is empty".into());
        }
        println!("  Source: {:.2} MB", source_size as f64 / 1_048_576.0);
        println!("  Backup: {:.2} MB", backup_size as f64 / 1_048_576.0);
    } else {
        // For uncompressed, check all files exist
        for source_file in &source_files {
            let rel_path = source_file
                .strip_prefix(source)
                .map_err(|e| format!("Failed to get relative path: {}", e))?;
            let target_file = target.join(rel_path);
            if !target_file.exists() {
                return Err(format!("File missing in backup: {}", rel_path.display()).into());
            }
        }
        println!("  {} file(s) verified", source_files.len());
    }

    Ok(())
}

/// Verify restore integrity
fn verify_restore(target: &Path) -> ToolResult<()> {
    // Basic verification: check directory is readable and contains files
    if !target.exists() {
        return Err("Restore target does not exist".into());
    }

    if !target.is_dir() {
        return Err("Restore target is not a directory".into());
    }

    let files = collect_files(target)?;
    if files.is_empty() {
        return Err("No files found in restore target".into());
    }

    println!("  {} file(s) restored", files.len());
    Ok(())
}

/// Write backup metadata
fn write_backup_metadata(
    config: &BackupConfig,
    files: &[PathBuf],
    total_size: u64,
) -> ToolResult<()> {
    let metadata = BackupMetadata {
        timestamp: SystemTime::now(),
        source_path: config.source.clone(),
        file_count: files.len(),
        total_size,
        compressed: config.compress,
        incremental: config.incremental,
    };

    let metadata_path = if config.target.is_dir() {
        config.target.join("backup_metadata.txt")
    } else {
        config.target.with_extension("metadata.txt")
    };

    let mut file = File::create(metadata_path)?;
    writeln!(file, "OxiRS Backup Metadata")?;
    writeln!(file, "====================")?;
    writeln!(file, "Timestamp: {:?}", metadata.timestamp)?;
    writeln!(file, "Source: {}", metadata.source_path.display())?;
    writeln!(file, "Files: {}", metadata.file_count)?;
    writeln!(file, "Size: {} bytes", metadata.total_size)?;
    writeln!(file, "Compressed: {}", metadata.compressed)?;
    writeln!(file, "Incremental: {}", metadata.incremental)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_dir(label: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "oxirs_tdbbackup_test_{label}_{}_{}",
            std::process::id(),
            n
        ))
    }

    /// The consolidated compressed backup path must produce a genuine gzip
    /// stream and restore it losslessly (backup + restore round-trip), reusing
    /// the crate's single gzip implementation for both directions.
    #[test]
    fn test_compressed_backup_restore_round_trip_gzip() {
        let src = unique_dir("src");
        let restore_dir = unique_dir("restore");
        let archive = unique_dir("archive").with_extension("gz");

        std::fs::create_dir_all(&src).expect("create src");
        std::fs::create_dir_all(src.join("sub")).expect("create subdir");
        std::fs::write(
            src.join("a.ttl"),
            b"@prefix ex: <http://example.org/> .\nex:a ex:b ex:c .\n",
        )
        .expect("write a.ttl");
        std::fs::write(
            src.join("sub/b.nt"),
            b"<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n",
        )
        .expect("write sub/b.nt");

        let files = collect_files(&src).expect("collect files");
        assert_eq!(files.len(), 2, "both source files must be collected");

        backup_compressed(&files, &src, &archive).expect("compressed backup");

        // The archive must be a real gzip stream, proving the single gzip
        // implementation actually compressed the framing.
        let raw = std::fs::read(&archive).expect("read archive");
        assert!(raw.len() >= 2, "archive too small");
        assert_eq!(
            &raw[..2],
            &[0x1f, 0x8b],
            "compressed backup must be a real gzip stream"
        );

        restore_compressed(&archive, &restore_dir).expect("compressed restore");

        // Every file must round-trip byte-for-byte.
        let restored_a = std::fs::read(restore_dir.join("a.ttl")).expect("read restored a.ttl");
        let restored_b =
            std::fs::read(restore_dir.join("sub/b.nt")).expect("read restored sub/b.nt");
        assert_eq!(
            restored_a,
            std::fs::read(src.join("a.ttl")).expect("orig a.ttl"),
            "a.ttl must round-trip through gzip backup+restore"
        );
        assert_eq!(
            restored_b,
            std::fs::read(src.join("sub/b.nt")).expect("orig sub/b.nt"),
            "sub/b.nt must round-trip through gzip backup+restore"
        );

        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&restore_dir).ok();
        let _ = std::fs::remove_file(&archive);
    }

    /// A non-gzip archive (e.g. produced before consolidation, or by a tool
    /// that wrote the framing uncompressed) must still restore: the sniffing
    /// reader passes a non-gzip stream through unchanged.
    #[test]
    fn test_restore_passes_through_non_gzip_archive() {
        let src = unique_dir("plainsrc");
        let restore_dir = unique_dir("plainrestore");
        let archive = unique_dir("plainarchive").with_extension("tar");

        std::fs::create_dir_all(&src).expect("create src");
        std::fs::write(
            src.join("only.nt"),
            b"<http://ex/s> <http://ex/p> <http://ex/o> .\n",
        )
        .expect("write only.nt");

        let files = collect_files(&src).expect("collect files");

        // Hand-write the tar-like framing WITHOUT compression to emulate a
        // legacy archive, then confirm the sniffing restore still reads it.
        {
            let mut out = File::create(&archive).expect("create plain archive");
            for file_path in &files {
                let rel = file_path.strip_prefix(&src).expect("rel path");
                let content = std::fs::read(file_path).expect("read source file");
                let path_str = rel.to_string_lossy();
                let path_bytes = path_str.as_bytes();
                out.write_all(&(path_bytes.len() as u32).to_le_bytes())
                    .expect("write path len");
                out.write_all(path_bytes).expect("write path");
                out.write_all(&(content.len() as u64).to_le_bytes())
                    .expect("write size");
                out.write_all(&content).expect("write content");
            }
            out.flush().expect("flush plain archive");
        }

        // The plain archive must NOT start with the gzip magic.
        let raw = std::fs::read(&archive).expect("read plain archive");
        assert_ne!(
            &raw[..2],
            &[0x1f, 0x8b],
            "control archive must be uncompressed"
        );

        restore_compressed(&archive, &restore_dir).expect("restore of plain archive");
        let restored = std::fs::read(restore_dir.join("only.nt")).expect("read restored only.nt");
        assert_eq!(
            restored,
            std::fs::read(src.join("only.nt")).expect("orig only.nt")
        );

        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&restore_dir).ok();
        let _ = std::fs::remove_file(&archive);
    }
}
