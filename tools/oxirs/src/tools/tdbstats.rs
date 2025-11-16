//! TDB Statistics Tool
//!
//! Comprehensive database statistics reporting for OxiRS TDB stores.
//! Provides detailed insights into storage, indexes, transactions, and performance.

use super::{ToolResult, ToolStats};
use oxirs_tdb::{TdbConfig, TdbStore};
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Statistics output format
#[derive(Debug, Clone, Copy)]
pub enum StatsFormat {
    /// Human-readable text format
    Text,
    /// JSON format for programmatic access
    Json,
    /// CSV format for spreadsheet import
    Csv,
}

impl FromStr for StatsFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" | "txt" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            "csv" => Ok(Self::Csv),
            _ => Err(format!("Unknown format: {}. Supported: text, json, csv", s)),
        }
    }
}

/// Run TDB statistics command
pub async fn run(location: PathBuf, detailed: bool, format: String) -> ToolResult {
    let mut stats = ToolStats::new();

    // Parse output format
    let output_format = format
        .parse::<StatsFormat>()
        .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)))?;

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

    // Open store
    let config = TdbConfig::new(&location);

    let store =
        TdbStore::open_with_config(config).map_err(|e| format!("Failed to open store: {}", e))?;

    // Collect basic statistics
    let basic_stats = store.stats();

    // Collect enhanced statistics if detailed mode
    let enhanced_stats = if detailed {
        Some(store.enhanced_stats())
    } else {
        None
    };

    // Display statistics based on format
    match output_format {
        StatsFormat::Text => {
            display_text_stats(&location, &basic_stats, enhanced_stats.as_ref(), detailed)?;
        }
        StatsFormat::Json => {
            display_json_stats(&location, &basic_stats, enhanced_stats.as_ref())?;
        }
        StatsFormat::Csv => {
            display_csv_stats(&location, &basic_stats, enhanced_stats.as_ref())?;
        }
    }

    stats.items_processed = basic_stats.triple_count;
    stats.finish();

    Ok(())
}

/// Display statistics in human-readable text format
fn display_text_stats(
    location: &Path,
    basic: &oxirs_tdb::TdbStats,
    enhanced: Option<&oxirs_tdb::store::TdbEnhancedStats>,
    detailed: bool,
) -> ToolResult {
    use colored::*;

    println!("{}", "═".repeat(70).bright_blue());
    println!(
        "{}",
        "  OxiRS TDB Database Statistics".bright_green().bold()
    );
    println!("{}", "═".repeat(70).bright_blue());
    println!();

    // Basic information
    println!("{}", "Dataset Information".bright_yellow().bold());
    println!("{}", "─".repeat(70));
    println!("  Location: {}", location.display().to_string().cyan());
    println!();

    // Triple statistics
    println!("{}", "RDF Data Statistics".bright_yellow().bold());
    println!("{}", "─".repeat(70));
    println!(
        "  Triple Count:    {} {}",
        format_number(basic.triple_count).bright_white().bold(),
        "triples".dimmed()
    );
    println!(
        "  Dictionary Size: {} {}",
        format_number(basic.dictionary_size).bright_white().bold(),
        "unique terms".dimmed()
    );

    // Compression ratio estimate
    if basic.dictionary_size > 0 {
        let compression_ratio = basic.triple_count as f64 / basic.dictionary_size as f64;
        println!(
            "  Compression:     {:.2}x {}",
            compression_ratio.to_string().bright_white().bold(),
            "(triples per term)".dimmed()
        );
    }
    println!();

    // Bloom filter statistics
    if let Some(ref bloom_stats) = basic.bloom_filter_stats {
        println!("{}", "Bloom Filter Statistics".bright_yellow().bold());
        println!("{}", "─".repeat(70));
        println!(
            "  Filter Size:     {} {}",
            format_bytes(bloom_stats.size as u64).bright_white(),
            format!("({} bits)", format_number(bloom_stats.size * 8)).dimmed()
        );
        println!(
            "  Estimated FPR:   {:.4}%",
            (bloom_stats.estimated_fpr * 100.0)
                .to_string()
                .bright_white()
        );
        println!(
            "  Hash Functions:  {}",
            format_number(bloom_stats.num_hashes).bright_white()
        );
        println!();
    }

    // Prefix compression statistics
    if let Some(ref comp_stats) = basic.compression_stats {
        println!("{}", "Prefix Compression Statistics".bright_yellow().bold());
        println!("{}", "─".repeat(70));
        println!(
            "  Unique Prefixes: {}",
            format_number(comp_stats.num_prefixes).bright_white().bold()
        );
        println!(
            "  Prefix Storage:  {}",
            format_bytes(comp_stats.total_prefix_bytes as u64).bright_white()
        );
        println!();
    }

    // Enhanced statistics (detailed mode)
    if detailed {
        if let Some(enhanced) = enhanced {
            // Buffer pool statistics
            use std::sync::atomic::Ordering;
            println!("{}", "Buffer Pool Statistics".bright_yellow().bold());
            println!("{}", "─".repeat(70));
            let total_fetches = enhanced.buffer_pool.total_fetches.load(Ordering::Relaxed) as usize;
            let cache_hits = enhanced.buffer_pool.cache_hits.load(Ordering::Relaxed) as usize;
            let cache_misses = enhanced.buffer_pool.cache_misses.load(Ordering::Relaxed) as usize;
            let evictions = enhanced.buffer_pool.evictions.load(Ordering::Relaxed) as usize;

            println!(
                "  Total Fetches:   {}",
                format_number(total_fetches).bright_white()
            );
            println!(
                "  Hit Rate:        {:.2}%",
                (enhanced.buffer_pool.hit_rate() * 100.0)
                    .to_string()
                    .bright_white()
                    .bold()
            );
            println!(
                "  Cache Hits:      {}",
                format_number(cache_hits).bright_white()
            );
            println!(
                "  Cache Misses:    {}",
                format_number(cache_misses).bright_white()
            );
            println!(
                "  Evictions:       {}",
                format_number(evictions).bright_white()
            );
            println!();

            // Storage statistics
            println!("{}", "Storage Statistics".bright_yellow().bold());
            println!("{}", "─".repeat(70));
            println!(
                "  Total Size:      {}",
                format_bytes(enhanced.storage.total_size_bytes)
                    .bright_white()
                    .bold()
            );
            println!(
                "  Pages Allocated: {}",
                format_number(enhanced.storage.pages_allocated).bright_white()
            );
            println!(
                "  Page Size:       {}",
                format_bytes(enhanced.storage.page_size as u64).bright_white()
            );
            println!(
                "  Memory Usage:    {}",
                format_bytes(enhanced.storage.memory_usage_bytes as u64).bright_white()
            );
            println!();

            // Transaction statistics
            println!("{}", "Transaction Statistics".bright_yellow().bold());
            println!("{}", "─".repeat(70));
            println!(
                "  Active Txns:     {}",
                format_number(enhanced.transaction.active_transactions).bright_white()
            );
            println!(
                "  WAL Enabled:     {}",
                if enhanced.transaction.wal_enabled {
                    "Yes".green()
                } else {
                    "No".yellow()
                }
            );
            if enhanced.transaction.wal_enabled {
                println!(
                    "  WAL Size:        {}",
                    format_bytes(enhanced.transaction.wal_size_bytes).bright_white()
                );
            }
            println!();

            // Index statistics
            println!("{}", "Index Statistics".bright_yellow().bold());
            println!("{}", "─".repeat(70));
            println!(
                "  SPO Index:       {} {}",
                format_number(enhanced.index.spo_entries).bright_white(),
                "entries".dimmed()
            );
            println!(
                "  POS Index:       {} {}",
                format_number(enhanced.index.pos_entries).bright_white(),
                "entries".dimmed()
            );
            println!(
                "  OSP Index:       {} {}",
                format_number(enhanced.index.osp_entries).bright_white(),
                "entries".dimmed()
            );
            println!(
                "  Total Entries:   {}",
                format_number(enhanced.index.total_entries())
                    .bright_white()
                    .bold()
            );
            println!();
        }
    }

    println!("{}", "═".repeat(70).bright_blue());

    if !detailed {
        println!(
            "{}",
            "  Tip: Use --detailed flag for comprehensive statistics".dimmed()
        );
        println!("{}", "═".repeat(70).bright_blue());
    }

    Ok(())
}

/// Display statistics in JSON format
fn display_json_stats(
    location: &Path,
    basic: &oxirs_tdb::TdbStats,
    enhanced: Option<&oxirs_tdb::store::TdbEnhancedStats>,
) -> ToolResult {
    use serde_json::json;

    let mut stats_json = json!({
        "dataset": {
            "location": location.display().to_string(),
        },
        "triples": {
            "count": basic.triple_count,
            "dictionary_size": basic.dictionary_size,
        }
    });

    // Add bloom filter stats if available
    if let Some(ref bloom_stats) = basic.bloom_filter_stats {
        stats_json["bloom_filter"] = json!({
            "size": bloom_stats.size,
            "num_hashes": bloom_stats.num_hashes,
            "count": bloom_stats.count,
            "bits_set": bloom_stats.bits_set,
            "load_factor": bloom_stats.load_factor,
            "estimated_fpr": bloom_stats.estimated_fpr,
        });
    }

    // Add compression stats if available
    if let Some(ref comp_stats) = basic.compression_stats {
        stats_json["compression"] = json!({
            "num_prefixes": comp_stats.num_prefixes,
            "total_prefix_bytes": comp_stats.total_prefix_bytes,
        });
    }

    // Add enhanced stats if available
    if let Some(enhanced) = enhanced {
        use std::sync::atomic::Ordering;
        stats_json["buffer_pool"] = json!({
            "total_fetches": enhanced.buffer_pool.total_fetches.load(Ordering::Relaxed),
            "hit_rate": enhanced.buffer_pool.hit_rate(),
            "cache_hits": enhanced.buffer_pool.cache_hits.load(Ordering::Relaxed),
            "cache_misses": enhanced.buffer_pool.cache_misses.load(Ordering::Relaxed),
            "evictions": enhanced.buffer_pool.evictions.load(Ordering::Relaxed),
        });

        stats_json["storage"] = json!({
            "total_size_bytes": enhanced.storage.total_size_bytes,
            "pages_allocated": enhanced.storage.pages_allocated,
            "page_size": enhanced.storage.page_size,
            "memory_usage_bytes": enhanced.storage.memory_usage_bytes,
        });

        stats_json["transaction"] = json!({
            "active_transactions": enhanced.transaction.active_transactions,
            "wal_enabled": enhanced.transaction.wal_enabled,
            "wal_size_bytes": enhanced.transaction.wal_size_bytes,
        });

        stats_json["index"] = json!({
            "spo_entries": enhanced.index.spo_entries,
            "pos_entries": enhanced.index.pos_entries,
            "osp_entries": enhanced.index.osp_entries,
            "total_entries": enhanced.index.total_entries(),
        });
    }

    println!("{}", serde_json::to_string_pretty(&stats_json)?);

    Ok(())
}

/// Display statistics in CSV format
fn display_csv_stats(
    location: &Path,
    basic: &oxirs_tdb::TdbStats,
    enhanced: Option<&oxirs_tdb::store::TdbEnhancedStats>,
) -> ToolResult {
    println!("metric,value,unit");
    println!("location,\"{}\",path", location.display());
    println!("triple_count,{},triples", basic.triple_count);
    println!("dictionary_size,{},terms", basic.dictionary_size);

    // Bloom filter statistics
    if let Some(ref bloom_stats) = basic.bloom_filter_stats {
        println!("bloom_size,{},bytes", bloom_stats.size);
        println!("bloom_num_hashes,{},count", bloom_stats.num_hashes);
        println!("bloom_count,{},items", bloom_stats.count);
        println!("bloom_bits_set,{},bits", bloom_stats.bits_set);
        println!("bloom_load_factor,{:.4},ratio", bloom_stats.load_factor);
        println!(
            "bloom_estimated_fpr,{:.6},percent",
            bloom_stats.estimated_fpr * 100.0
        );
    }

    // Compression statistics
    if let Some(ref comp_stats) = basic.compression_stats {
        println!("compression_num_prefixes,{},count", comp_stats.num_prefixes);
        println!(
            "compression_prefix_bytes,{},bytes",
            comp_stats.total_prefix_bytes
        );
    }

    // Enhanced statistics
    if let Some(enhanced) = enhanced {
        use std::sync::atomic::Ordering;
        println!(
            "buffer_total_fetches,{},count",
            enhanced.buffer_pool.total_fetches.load(Ordering::Relaxed)
        );
        println!(
            "buffer_hit_rate,{:.4},percent",
            enhanced.buffer_pool.hit_rate() * 100.0
        );
        println!(
            "buffer_cache_hits,{},count",
            enhanced.buffer_pool.cache_hits.load(Ordering::Relaxed)
        );
        println!(
            "buffer_cache_misses,{},count",
            enhanced.buffer_pool.cache_misses.load(Ordering::Relaxed)
        );
        println!(
            "buffer_evictions,{},count",
            enhanced.buffer_pool.evictions.load(Ordering::Relaxed)
        );

        println!(
            "storage_total_size,{},bytes",
            enhanced.storage.total_size_bytes
        );
        println!("storage_pages,{},pages", enhanced.storage.pages_allocated);
        println!("storage_page_size,{},bytes", enhanced.storage.page_size);
        println!(
            "storage_memory,{},bytes",
            enhanced.storage.memory_usage_bytes
        );

        println!(
            "txn_active,{},count",
            enhanced.transaction.active_transactions
        );
        println!(
            "txn_wal_enabled,{},boolean",
            enhanced.transaction.wal_enabled
        );
        println!("txn_wal_size,{},bytes", enhanced.transaction.wal_size_bytes);

        println!("index_spo,{},entries", enhanced.index.spo_entries);
        println!("index_pos,{},entries", enhanced.index.pos_entries);
        println!("index_osp,{},entries", enhanced.index.osp_entries);
        println!("index_total,{},entries", enhanced.index.total_entries());
    }

    Ok(())
}

/// Format number with thousands separators
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();

    for (count, c) in s.chars().rev().enumerate() {
        if count > 0 && count % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }

    result.chars().rev().collect()
}

/// Format bytes in human-readable format
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1234567), "1,234,567");
        assert_eq!(format_number(1000000000), "1,000,000,000");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_stats_format_parse() {
        assert!(matches!(
            "text".parse::<StatsFormat>(),
            Ok(StatsFormat::Text)
        ));
        assert!(matches!(
            "json".parse::<StatsFormat>(),
            Ok(StatsFormat::Json)
        ));
        assert!(matches!("csv".parse::<StatsFormat>(), Ok(StatsFormat::Csv)));
        assert!("invalid".parse::<StatsFormat>().is_err());
    }
}
