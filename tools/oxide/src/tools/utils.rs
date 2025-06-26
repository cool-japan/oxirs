//! Common utilities for CLI tools

use super::ToolResult;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;

/// Detect RDF format from file extension
pub fn detect_rdf_format(file: &Path) -> String {
    if let Some(ext) = file.extension().and_then(|s| s.to_str()) {
        match ext.to_lowercase().as_str() {
            "ttl" | "turtle" => "turtle".to_string(),
            "nt" | "ntriples" => "ntriples".to_string(),
            "rdf" | "xml" => "rdfxml".to_string(),
            "jsonld" | "json-ld" => "jsonld".to_string(),
            "trig" => "trig".to_string(),
            "nq" | "nquads" => "nquads".to_string(),
            "owl" => "rdfxml".to_string(), // OWL is typically RDF/XML
            _ => "turtle".to_string(),     // Default fallback
        }
    } else {
        "turtle".to_string() // Default fallback
    }
}

/// Check if a format is supported for input
pub fn is_supported_input_format(format: &str) -> bool {
    matches!(
        format,
        "turtle" | "ntriples" | "rdfxml" | "jsonld" | "trig" | "nquads"
    )
}

/// Check if a format is supported for output
pub fn is_supported_output_format(format: &str) -> bool {
    matches!(
        format,
        "turtle" | "ntriples" | "rdfxml" | "jsonld" | "trig" | "nquads"
    )
}

/// Check if a SPARQL results format is supported
pub fn is_supported_results_format(format: &str) -> bool {
    matches!(format, "table" | "csv" | "tsv" | "json" | "xml")
}

/// Read file content or stdin if path is "-"
pub fn read_input(path: &Path) -> ToolResult<String> {
    if path.to_str() == Some("-") {
        // Read from stdin
        let stdin = io::stdin();
        let mut content = String::new();
        for line in stdin.lock().lines() {
            content.push_str(&line?);
            content.push('\n');
        }
        Ok(content)
    } else {
        Ok(fs::read_to_string(path)?)
    }
}

/// Write content to file or stdout if path is None or "-"
pub fn write_output(content: &str, path: Option<&Path>) -> ToolResult<()> {
    match path {
        None => {
            // Write to stdout
            io::stdout().write_all(content.as_bytes())?;
            io::stdout().flush()?;
        }
        Some(path) if path.to_str() == Some("-") => {
            // Write to stdout
            io::stdout().write_all(content.as_bytes())?;
            io::stdout().flush()?;
        }
        Some(path) => {
            // Write to file
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, content)?;
        }
    }
    Ok(())
}

/// Format file size in human-readable form
pub fn format_file_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Format duration in human-readable form
pub fn format_duration(duration: std::time::Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    let millis = duration.subsec_millis();

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}.{}s", minutes, seconds, millis / 100)
    } else if seconds > 0 {
        format!("{}.{}s", seconds, millis / 100)
    } else {
        format!("{}ms", millis)
    }
}

/// Validate and normalize IRI
pub fn validate_iri(iri: &str) -> Result<String, String> {
    // Basic IRI validation - in a real implementation, this would be more comprehensive
    if iri.is_empty() {
        return Err("IRI cannot be empty".to_string());
    }

    // Check for basic IRI structure
    if iri.contains(' ') {
        return Err("IRI cannot contain spaces".to_string());
    }

    // Basic scheme check
    if !iri.contains(':') {
        return Err("IRI must contain a scheme".to_string());
    }

    // Normalize by trimming whitespace
    Ok(iri.trim().to_string())
}

/// Validate language tag (RFC 5646)
pub fn validate_language_tag(tag: &str) -> Result<String, String> {
    // Basic language tag validation - in a real implementation, this would follow RFC 5646
    if tag.is_empty() {
        return Err("Language tag cannot be empty".to_string());
    }

    // Basic pattern: language[-script][-region][-variant]
    let normalized = tag.to_lowercase();

    // Check for valid characters (letters, numbers, hyphens)
    if !normalized
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-')
    {
        return Err("Language tag contains invalid characters".to_string());
    }

    // Should not start or end with hyphen
    if normalized.starts_with('-') || normalized.ends_with('-') {
        return Err("Language tag cannot start or end with hyphen".to_string());
    }

    // Should not have consecutive hyphens
    if normalized.contains("--") {
        return Err("Language tag cannot have consecutive hyphens".to_string());
    }

    Ok(normalized)
}

/// URL encode string
pub fn url_encode(input: &str) -> String {
    // Simple URL encoding - in practice, would use a proper URL encoding library
    input
        .chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            ' ' => "+".to_string(),
            _ => format!("%{:02X}", c as u8),
        })
        .collect()
}

/// URL decode string
pub fn url_decode(input: &str) -> Result<String, String> {
    let mut result = String::new();
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '+' => result.push(' '),
            '%' => {
                // Decode hex sequence
                let hex1 = chars
                    .next()
                    .ok_or("Invalid URL encoding: incomplete hex sequence")?;
                let hex2 = chars
                    .next()
                    .ok_or("Invalid URL encoding: incomplete hex sequence")?;

                let hex_str = format!("{}{}", hex1, hex2);
                let byte = u8::from_str_radix(&hex_str, 16)
                    .map_err(|_| "Invalid URL encoding: invalid hex sequence")?;

                result.push(byte as char);
            }
            _ => result.push(c),
        }
    }

    Ok(result)
}

/// Check if file exists and is readable
pub fn check_file_readable(path: &Path) -> ToolResult<()> {
    if !path.exists() {
        return Err(format!("File does not exist: {}", path.display()).into());
    }

    if !path.is_file() {
        return Err(format!("Path is not a file: {}", path.display()).into());
    }

    // Try to open file to check readability
    fs::File::open(path).map_err(|e| format!("Cannot read file {}: {}", path.display(), e))?;

    Ok(())
}

/// Get file extension
pub fn get_file_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
}

/// Progress indicator for long-running operations
pub struct ProgressIndicator {
    last_update: std::time::Instant,
    interval: std::time::Duration,
}

impl ProgressIndicator {
    pub fn new() -> Self {
        Self {
            last_update: std::time::Instant::now(),
            interval: std::time::Duration::from_millis(500), // Update every 500ms
        }
    }

    pub fn update(&mut self, current: usize, total: Option<usize>) {
        let now = std::time::Instant::now();
        if now.duration_since(self.last_update) >= self.interval {
            match total {
                Some(total) if total > 0 => {
                    let percent = (current as f64 / total as f64 * 100.0) as u8;
                    print!("\rProgress: {}/{} ({}%)", current, total, percent);
                }
                _ => {
                    print!("\rProcessed: {}", current);
                }
            }
            io::stdout().flush().unwrap_or(());
            self.last_update = now;
        }
    }

    pub fn finish(&self, total: usize) {
        println!("\rCompleted: {} items", total);
    }
}

impl Default for ProgressIndicator {
    fn default() -> Self {
        Self::new()
    }
}
