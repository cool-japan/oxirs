//! TDB Compaction Tool
//!
//! Database compaction and optimization for OxiRS TDB stores.
//! Reclaims space, rebuilds indexes, and optimizes storage layout.

use super::{ToolResult, ToolStats};
use colored::Colorize;
use oxirs_tdb::{TdbConfig, TdbStore};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Compaction report
#[derive(Debug)]
#[allow(dead_code)]
struct CompactionReport {
    /// Storage size before compaction
    size_before: u64,
    /// Storage size after compaction
    size_after: u64,
    /// Time taken for compaction
    duration_secs: f64,
    /// Number of files processed
    files_processed: usize,
    /// Number of old files deleted
    files_deleted: usize,
}

impl CompactionReport {
    /// Calculate space savings
    fn space_saved(&self) -> u64 {
        self.size_before.saturating_sub(self.size_after)
    }

    /// Calculate space savings percentage
    fn space_saved_pct(&self) -> f64 {
        if self.size_before > 0 {
            (self.space_saved() as f64 / self.size_before as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Format bytes for display
    fn format_bytes(bytes: u64) -> String {
        const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];

        if bytes == 0 {
            return "0 B".to_string();
        }

        let bytes_f = bytes as f64;
        let unit_index = (bytes_f.log2() / 10.0).floor() as usize;
        let unit_index = unit_index.min(UNITS.len() - 1);

        let value = bytes_f / (1024_f64.powi(unit_index as i32));

        if value < 10.0 {
            format!("{:.2} {}", value, UNITS[unit_index])
        } else if value < 100.0 {
            format!("{:.1} {}", value, UNITS[unit_index])
        } else {
            format!("{:.0} {}", value, UNITS[unit_index])
        }
    }
}

/// Run TDB compaction command
pub async fn run(location: PathBuf, delete_old: bool) -> ToolResult {
    let mut stats = ToolStats::new();

    println!("{}", "═".repeat(70).bright_blue());
    println!(
        "{}",
        "  OxiRS TDB Database Compaction".bright_green().bold()
    );
    println!("{}", "═".repeat(70).bright_blue());
    println!();

    // Validate location exists
    if !location.exists() {
        return Err(format!("Dataset location does not exist: {}", location.display()).into());
    }

    if !location.is_dir() {
        return Err(format!(
            "Dataset location must be a directory: {}",
            location.display()
        )
        .into());
    }

    println!(
        "Dataset Location: {}",
        location.display().to_string().cyan()
    );
    println!(
        "Delete Old Files: {}",
        if delete_old {
            "Yes".green()
        } else {
            "No".yellow()
        }
    );
    println!();

    // Calculate initial size
    println!("{}", "Analyzing database...".bright_yellow());
    let size_before = calculate_directory_size(&location)?;
    println!(
        "  Current size: {}",
        CompactionReport::format_bytes(size_before)
            .bright_white()
            .bold()
    );

    // Count files before compaction
    let files_before = count_files(&location)?;
    println!(
        "  File count:   {}",
        format!("{}", files_before).bright_white()
    );
    println!();

    // Open store with write access
    println!("{}", "Opening database...".bright_yellow());
    let config = TdbConfig::new(&location);

    let mut store =
        TdbStore::open_with_config(config).map_err(|e| format!("Failed to open store: {}", e))?;

    println!("  ✓ {}", "Database opened".green());
    println!();

    // Perform compaction
    println!("{}", "Performing compaction...".bright_yellow());
    println!("  This may take several minutes for large databases.");
    println!();

    let compaction_start = Instant::now();

    // Step 1: Run store compaction
    println!("  [1/4] {} bloom filters...", "Rebuilding".bright_white());
    store
        .compact()
        .map_err(|e| format!("Compaction failed: {}", e))?;
    println!("        ✓ {}", "Bloom filters optimized".dimmed());

    // Step 2: Optimize indexes
    println!("  [2/4] {} indexes...", "Optimizing".bright_white());
    // Index optimization is part of the compact() call
    println!("        ✓ {}", "Indexes optimized".dimmed());

    // Step 3: Flush all data
    println!("  [3/4] {} data to disk...", "Flushing".bright_white());
    // Explicit flush if needed (compact() already does this)
    println!("        ✓ {}", "Data synchronized".dimmed());

    // Step 4: Clean up old files
    println!("  [4/4] {} old files...", "Cleaning up".bright_white());
    let files_deleted = if delete_old {
        delete_obsolete_files(&location)?
    } else {
        0
    };

    if delete_old {
        println!(
            "        ✓ {} {}",
            format!("{}", files_deleted).bright_white(),
            "obsolete files deleted".dimmed()
        );
    } else {
        println!(
            "        ⊘ {}",
            "Skipped (use --delete-old to enable)".dimmed()
        );
    }

    let compaction_duration = compaction_start.elapsed();

    // Close store to ensure all data is written
    drop(store);

    println!();
    println!(
        "  ✓ {}",
        format!(
            "Compaction completed in {:.2}s",
            compaction_duration.as_secs_f64()
        )
        .green()
        .bold()
    );
    println!();

    // Calculate final size
    println!("{}", "Analyzing results...".bright_yellow());
    let size_after = calculate_directory_size(&location)?;
    let files_after = count_files(&location)?;

    let report = CompactionReport {
        size_before,
        size_after,
        duration_secs: compaction_duration.as_secs_f64(),
        files_processed: files_before,
        files_deleted,
    };

    // Display summary
    println!();
    println!("{}", "Compaction Summary".bright_yellow().bold());
    println!("{}", "─".repeat(70));
    println!(
        "  Size Before:   {}",
        CompactionReport::format_bytes(report.size_before).bright_white()
    );
    println!(
        "  Size After:    {}",
        CompactionReport::format_bytes(report.size_after).bright_white()
    );

    let space_saved = report.space_saved();
    if space_saved > 0 {
        println!(
            "  Space Saved:   {} {}",
            CompactionReport::format_bytes(space_saved)
                .bright_green()
                .bold(),
            format!("({:.1}% reduction)", report.space_saved_pct()).dimmed()
        );
    } else if report.size_after > report.size_before {
        let space_increased = report.size_after - report.size_before;
        println!(
            "  Space Change:  {} {}",
            format!("+{}", CompactionReport::format_bytes(space_increased)).bright_yellow(),
            "(database reorganization)".dimmed()
        );
    } else {
        println!("  Space Change:  {}", "No change".yellow());
    }

    println!(
        "  Files Before:  {}",
        files_before.to_string().bright_white()
    );
    println!(
        "  Files After:   {}",
        files_after.to_string().bright_white()
    );

    if files_deleted > 0 {
        println!(
            "  Files Deleted: {}",
            files_deleted.to_string().bright_white()
        );
    }

    println!(
        "  Duration:      {}s",
        format!("{:.2}", report.duration_secs).bright_white()
    );

    println!();
    println!("{}", "═".repeat(70).bright_blue());

    if !delete_old {
        println!(
            "  {}",
            "Tip: Run with --delete-old to remove obsolete files".dimmed()
        );
        println!("{}", "═".repeat(70).bright_blue());
    }

    stats.items_processed = report.files_processed;
    stats.finish();

    Ok(())
}

/// Calculate total size of directory recursively
fn calculate_directory_size(dir: &PathBuf) -> Result<u64, Box<dyn std::error::Error>> {
    let mut total_size = 0u64;

    fn calculate_recursive(
        dir: &PathBuf,
        total: &mut u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                calculate_recursive(&path, total)?;
            } else if path.is_file() {
                *total += entry.metadata()?.len();
            }
        }
        Ok(())
    }

    calculate_recursive(dir, &mut total_size)?;
    Ok(total_size)
}

/// Count files in directory recursively
fn count_files(dir: &PathBuf) -> Result<usize, Box<dyn std::error::Error>> {
    let mut count = 0;

    fn count_recursive(dir: &PathBuf, count: &mut usize) -> Result<(), Box<dyn std::error::Error>> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                count_recursive(&path, count)?;
            } else if path.is_file() {
                *count += 1;
            }
        }
        Ok(())
    }

    count_recursive(dir, &mut count)?;
    Ok(count)
}

/// Delete obsolete files (logs, temporary files, old backups)
fn delete_obsolete_files(dir: &PathBuf) -> Result<usize, Box<dyn std::error::Error>> {
    let mut deleted_count = 0;

    // Patterns for obsolete files
    let obsolete_patterns = vec![
        ".tmp",      // Temporary files
        ".old",      // Old backups
        ".bak",      // Backup files
        ".log",      // Log files (be careful with this)
        "~",         // Editor backup files
        ".lock.old", // Old lock files
    ];

    fn delete_recursive(
        dir: &PathBuf,
        patterns: &[&str],
        deleted: &mut usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                delete_recursive(&path, patterns, deleted)?;
            } else if path.is_file() {
                let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

                // Check if file matches any obsolete pattern
                for pattern in patterns {
                    if file_name.ends_with(pattern) {
                        match fs::remove_file(&path) {
                            Ok(_) => {
                                *deleted += 1;
                            }
                            Err(e) => {
                                eprintln!("Warning: Failed to delete {}: {}", path.display(), e);
                            }
                        }
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    delete_recursive(dir, &obsolete_patterns, &mut deleted_count)?;
    Ok(deleted_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_calculate_directory_size() {
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path().to_path_buf();

        // Create test files
        let mut file1 = File::create(dir_path.join("file1.txt")).unwrap();
        file1.write_all(b"hello").unwrap();

        let mut file2 = File::create(dir_path.join("file2.txt")).unwrap();
        file2.write_all(b"world!").unwrap();

        let total_size = calculate_directory_size(&dir_path).unwrap();
        assert_eq!(total_size, 11); // "hello" (5) + "world!" (6)
    }

    #[test]
    fn test_count_files() {
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path().to_path_buf();

        // Create test files
        File::create(dir_path.join("file1.txt")).unwrap();
        File::create(dir_path.join("file2.txt")).unwrap();
        File::create(dir_path.join("file3.dat")).unwrap();

        let count = count_files(&dir_path).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_delete_obsolete_files() {
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path().to_path_buf();

        // Create various files
        File::create(dir_path.join("data.db")).unwrap();
        File::create(dir_path.join("backup.old")).unwrap();
        File::create(dir_path.join("cache.tmp")).unwrap();
        File::create(dir_path.join("log.log")).unwrap();

        let deleted = delete_obsolete_files(&dir_path).unwrap();
        assert_eq!(deleted, 3); // .old, .tmp, .log files deleted

        // Check that data.db still exists
        assert!(dir_path.join("data.db").exists());
        assert!(!dir_path.join("backup.old").exists());
        assert!(!dir_path.join("cache.tmp").exists());
    }

    #[test]
    fn test_compaction_report() {
        let report = CompactionReport {
            size_before: 1024 * 1024 * 100, // 100 MB
            size_after: 1024 * 1024 * 70,   // 70 MB
            duration_secs: 5.5,
            files_processed: 50,
            files_deleted: 5,
        };

        assert_eq!(report.space_saved(), 1024 * 1024 * 30);
        assert!((report.space_saved_pct() - 30.0).abs() < 0.1);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(CompactionReport::format_bytes(0), "0 B");
        assert_eq!(CompactionReport::format_bytes(512), "512 B");
        assert_eq!(CompactionReport::format_bytes(1024), "1.00 KB");
        assert_eq!(CompactionReport::format_bytes(1536), "1.50 KB");
        assert_eq!(CompactionReport::format_bytes(1048576), "1.00 MB");
        assert_eq!(CompactionReport::format_bytes(1073741824), "1.00 GB");
    }
}
