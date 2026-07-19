//! Compression Support for RDF Files
//!
//! Support for reading and writing compressed RDF files with gzip compression.

use super::ToolResult;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Cursor, Read, Write};
use std::path::Path;

/// Default gzip compression level used by [`open_writer`] (mirrors the
/// historical `flate2::Compression::default()` which maps to level 6).
const GZIP_DEFAULT_LEVEL: u8 = 6;

/// Streaming gzip writer adapter backed by [`oxiarc_deflate`].
///
/// `oxiarc-deflate` exposes a one-shot `gzip_compress` API rather than an
/// incremental encoder, so this adapter buffers all written bytes in memory
/// and compresses them into the underlying file when [`flush`](Write::flush)
/// is called (and once more on drop, in case the caller forgot to flush).
struct GzipBufWriter {
    inner: File,
    buffer: Vec<u8>,
    level: u8,
    /// Set once the buffered data has been compressed and written so that a
    /// `flush()` followed by `drop` does not emit the gzip stream twice.
    finished: bool,
}

impl GzipBufWriter {
    fn new(inner: File, level: u8) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
            level,
            finished: false,
        }
    }

    /// Compress the accumulated buffer and write the gzip stream to the file.
    fn finish(&mut self) -> io::Result<()> {
        if self.finished {
            return Ok(());
        }
        let compressed = oxiarc_deflate::gzip_compress(&self.buffer, self.level).map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("gzip compression failed: {e}"),
            )
        })?;
        self.inner.write_all(&compressed)?;
        self.inner.flush()?;
        self.finished = true;
        Ok(())
    }
}

impl Write for GzipBufWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.finish()
    }
}

impl Drop for GzipBufWriter {
    fn drop(&mut self) {
        // Best-effort: ensure buffered data is emitted even without an explicit
        // flush. Errors during drop cannot be propagated, so they are ignored;
        // callers that need the result must call `flush()` explicitly.
        let _ = self.finish();
    }
}

/// Compression formats supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionFormat {
    None,
    Gzip,
    Bzip2,
    Xz,
}

impl CompressionFormat {
    /// Detect compression format from file extension
    pub fn from_path(path: &Path) -> Self {
        let path_str = path.to_string_lossy().to_lowercase();

        if path_str.ends_with(".gz") || path_str.ends_with(".gzip") {
            CompressionFormat::Gzip
        } else if path_str.ends_with(".bz2") || path_str.ends_with(".bzip2") {
            CompressionFormat::Bzip2
        } else if path_str.ends_with(".xz") || path_str.ends_with(".lzma") {
            CompressionFormat::Xz
        } else {
            CompressionFormat::None
        }
    }

    /// Get file extension for this format
    pub fn extension(&self) -> &str {
        match self {
            CompressionFormat::None => "",
            CompressionFormat::Gzip => ".gz",
            CompressionFormat::Bzip2 => ".bz2",
            CompressionFormat::Xz => ".xz",
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &str {
        match self {
            CompressionFormat::None => "none",
            CompressionFormat::Gzip => "gzip",
            CompressionFormat::Bzip2 => "bzip2",
            CompressionFormat::Xz => "xz",
        }
    }
}

/// Compressed reader wrapper.
///
/// The boxed variants are `Send` so a `CompressedReader` can be fed directly
/// into oxirs-core's streaming parsers (`RdfParser::for_reader` /
/// `FormatHandler::parse_triples`), both of which require
/// `Read + Send + 'static`.
pub enum CompressedReader {
    None(BufReader<File>),
    Gzip(Box<dyn Read + Send>),
    Bzip2(Box<dyn Read + Send>),
    Xz(Box<dyn Read + Send>),
}

impl Read for CompressedReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            CompressedReader::None(r) => r.read(buf),
            CompressedReader::Gzip(r) => r.read(buf),
            CompressedReader::Bzip2(r) => r.read(buf),
            CompressedReader::Xz(r) => r.read(buf),
        }
    }
}

/// Compressed writer wrapper
pub enum CompressedWriter {
    None(BufWriter<File>),
    Gzip(Box<dyn Write>),
    Bzip2(Box<dyn Write>),
    Xz(Box<dyn Write>),
}

impl Write for CompressedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            CompressedWriter::None(w) => w.write(buf),
            CompressedWriter::Gzip(w) => w.write(buf),
            CompressedWriter::Bzip2(w) => w.write(buf),
            CompressedWriter::Xz(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            CompressedWriter::None(w) => w.flush(),
            CompressedWriter::Gzip(w) => w.flush(),
            CompressedWriter::Bzip2(w) => w.flush(),
            CompressedWriter::Xz(w) => w.flush(),
        }
    }
}

/// Open a file for reading with automatic decompression
pub fn open_reader(path: &Path) -> ToolResult<CompressedReader> {
    let format = CompressionFormat::from_path(path);
    let file = File::open(path)?;

    match format {
        CompressionFormat::None => Ok(CompressedReader::None(BufReader::new(file))),

        CompressionFormat::Gzip => {
            // Pure-Rust gzip decompression via oxiarc-deflate. The crate offers
            // a one-shot API, so the whole file is read and inflated up front,
            // then served from an in-memory cursor.
            let mut compressed = Vec::new();
            BufReader::new(file).read_to_end(&mut compressed)?;
            let decompressed = oxiarc_deflate::gzip_decompress(&compressed)
                .map_err(|e| format!("gzip decompression failed: {e}"))?;
            Ok(CompressedReader::Gzip(Box::new(Cursor::new(decompressed))))
        }

        CompressionFormat::Bzip2 => {
            // Simulate bzip2 decompression (in production, use bzip2 crate)
            let reader = BufReader::new(file);
            Ok(CompressedReader::Bzip2(Box::new(reader)))
        }

        CompressionFormat::Xz => {
            // Simulate xz decompression (in production, use xz2 crate)
            let reader = BufReader::new(file);
            Ok(CompressedReader::Xz(Box::new(reader)))
        }
    }
}

/// Open a file for reading, decompressing gzip transparently based on the
/// file's *content* (the gzip magic bytes `0x1f 0x8b`) rather than its name.
///
/// This exists for callers whose compressed artifact may not carry a `.gz`
/// file name (for example an archive whose name was chosen by the user): it
/// guarantees a gzip stream is inflated and a non-gzip stream is passed
/// through verbatim, while still routing every gzip inflate through the one
/// `oxiarc_deflate` implementation in this crate. Corrupt gzip input yields an
/// explicit error.
pub fn open_reader_sniffed(path: &Path) -> ToolResult<Box<dyn Read + Send>> {
    let mut all = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut all)?;

    if all.len() >= 2 && all[0] == 0x1f && all[1] == 0x8b {
        let decompressed = oxiarc_deflate::gzip_decompress(&all)
            .map_err(|e| format!("gzip decompression failed: {e}"))?;
        Ok(Box::new(Cursor::new(decompressed)))
    } else {
        Ok(Box::new(Cursor::new(all)))
    }
}

/// Open a file for writing with automatic compression
pub fn open_writer(path: &Path, format: CompressionFormat) -> ToolResult<CompressedWriter> {
    let file = File::create(path)?;

    match format {
        CompressionFormat::None => Ok(CompressedWriter::None(BufWriter::new(file))),

        CompressionFormat::Gzip => {
            // Pure-Rust gzip compression via oxiarc-deflate. The adapter buffers
            // writes and emits the gzip stream on flush/drop (level 6, matching
            // the previous flate2 default).
            let writer = GzipBufWriter::new(file, GZIP_DEFAULT_LEVEL);
            Ok(CompressedWriter::Gzip(Box::new(writer)))
        }

        CompressionFormat::Bzip2 => {
            // Simulate bzip2 compression (in production, use bzip2 crate)
            let writer = BufWriter::new(file);
            Ok(CompressedWriter::Bzip2(Box::new(writer)))
        }

        CompressionFormat::Xz => {
            // Simulate xz compression (in production, use xz2 crate)
            let writer = BufWriter::new(file);
            Ok(CompressedWriter::Xz(Box::new(writer)))
        }
    }
}

/// Compress a file
pub fn compress_file(
    input: &Path,
    output: &Path,
    format: CompressionFormat,
) -> ToolResult<CompressionStats> {
    let start = std::time::Instant::now();

    println!("Compressing: {}", input.display());
    println!("Output: {}", output.display());
    println!("Format: {}\n", format.name());

    // Open input
    let mut input_file = File::open(input)?;
    let input_size = input_file.metadata()?.len();

    // Open output with compression
    let mut output_writer = open_writer(output, format)?;

    // Copy with compression
    let mut buffer = vec![0u8; 65536]; // 64KB buffer
    let mut total_read = 0u64;

    loop {
        let n = input_file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        output_writer.write_all(&buffer[..n])?;
        total_read += n as u64;

        // Progress indicator
        let percent = (total_read as f64 / input_size as f64 * 100.0) as u8;
        print!("\rProgress: {}%", percent);
        io::stdout().flush()?;
    }

    output_writer.flush()?;
    println!();

    // Get output size
    let output_size = std::fs::metadata(output)?.len();

    let duration = start.elapsed();
    let ratio = if input_size > 0 {
        output_size as f64 / input_size as f64
    } else {
        0.0
    };

    let stats = CompressionStats {
        input_size,
        output_size,
        ratio,
        duration,
    };

    println!("\nCompression Statistics:");
    println!("  Input:  {:.2} MB", input_size as f64 / 1_048_576.0);
    println!("  Output: {:.2} MB", output_size as f64 / 1_048_576.0);
    println!("  Ratio:  {:.1}%", ratio * 100.0);
    println!("  Time:   {:.2}s", duration.as_secs_f64());

    if input_size > 0 {
        let saved = input_size.saturating_sub(output_size);
        println!("  Saved:  {:.2} MB", saved as f64 / 1_048_576.0);
    }

    Ok(stats)
}

/// Decompress a file
pub fn decompress_file(input: &Path, output: &Path) -> ToolResult<DecompressionStats> {
    let format = CompressionFormat::from_path(input);

    if format == CompressionFormat::None {
        return Err("Input file is not compressed".into());
    }

    let start = std::time::Instant::now();

    println!("Decompressing: {}", input.display());
    println!("Output: {}", output.display());
    println!("Format: {}\n", format.name());

    // Open input with decompression
    let mut input_reader = open_reader(input)?;
    let input_size = std::fs::metadata(input)?.len();

    // Open output
    let mut output_file = BufWriter::new(File::create(output)?);

    // Copy with decompression
    let mut buffer = vec![0u8; 65536]; // 64KB buffer
    let mut total_written = 0u64;
    let mut last_progress = 0u8;

    loop {
        let n = input_reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        output_file.write_all(&buffer[..n])?;
        total_written += n as u64;

        // Progress indicator (approximate)
        if input_size > 0 {
            // Note: This is approximate since we don't know decompressed size
            let progress = ((total_written / 10) as f64).min(100.0) as u8;
            if progress > last_progress {
                print!("\rDecompressing...");
                io::stdout().flush()?;
                last_progress = progress;
            }
        }
    }

    output_file.flush()?;
    println!();

    // Get output size
    let output_size = std::fs::metadata(output)?.len();

    let duration = start.elapsed();

    let stats = DecompressionStats {
        input_size,
        output_size,
        duration,
    };

    println!("\nDecompression Statistics:");
    println!("  Input:  {:.2} MB", input_size as f64 / 1_048_576.0);
    println!("  Output: {:.2} MB", output_size as f64 / 1_048_576.0);
    println!("  Time:   {:.2}s", duration.as_secs_f64());

    Ok(stats)
}

/// Compression statistics
#[derive(Debug)]
pub struct CompressionStats {
    pub input_size: u64,
    pub output_size: u64,
    pub ratio: f64,
    pub duration: std::time::Duration,
}

/// Decompression statistics
#[derive(Debug)]
pub struct DecompressionStats {
    pub input_size: u64,
    pub output_size: u64,
    pub duration: std::time::Duration,
}

/// Strip a single trailing compression extension so callers can run their own
/// (name-based) RDF format detection on the *inner* file name.
///
/// Examples: `data.ttl.gz` -> `data.ttl`, `data.nt.bz2` -> `data.nt`,
/// `data.ttl` -> `data.ttl` (unchanged). This is what makes `.gz` handling
/// transparent: the RDF format is decided by the inner extension, not by the
/// compression wrapper.
pub fn strip_compression_suffix(path: &Path) -> std::path::PathBuf {
    if CompressionFormat::from_path(path) == CompressionFormat::None {
        return path.to_path_buf();
    }
    // The compression extension is the final dotted component; dropping it
    // reveals the inner name (e.g. `data.ttl.gz` -> `data.ttl`).
    path.with_extension("")
}

/// Detect RDF format from compressed file path
/// Examples: "data.ttl.gz" -> "turtle", "data.nt.bz2" -> "ntriples"
pub fn detect_rdf_format_compressed(path: &Path) -> Option<String> {
    let path_str = path.to_string_lossy().to_lowercase();

    // Remove compression extensions
    let path_without_compression = path_str
        .trim_end_matches(".gz")
        .trim_end_matches(".gzip")
        .trim_end_matches(".bz2")
        .trim_end_matches(".bzip2")
        .trim_end_matches(".xz")
        .trim_end_matches(".lzma");

    // Detect RDF format from remaining extension
    if let Some(ext) = path_without_compression.split('.').next_back() {
        let format = match ext {
            "ttl" | "turtle" => "turtle",
            "nt" | "ntriples" => "ntriples",
            "rdf" | "xml" => "rdfxml",
            "jsonld" | "json-ld" => "jsonld",
            "trig" => "trig",
            "nq" | "nquads" => "nquads",
            _ => return None,
        };
        Some(format.to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_compression_format() {
        assert_eq!(
            CompressionFormat::from_path(Path::new("data.ttl.gz")),
            CompressionFormat::Gzip
        );
        assert_eq!(
            CompressionFormat::from_path(Path::new("data.nt.bz2")),
            CompressionFormat::Bzip2
        );
        assert_eq!(
            CompressionFormat::from_path(Path::new("data.rdf.xz")),
            CompressionFormat::Xz
        );
        assert_eq!(
            CompressionFormat::from_path(Path::new("data.ttl")),
            CompressionFormat::None
        );
    }

    #[test]
    fn test_detect_rdf_format_compressed() {
        assert_eq!(
            detect_rdf_format_compressed(Path::new("data.ttl.gz")),
            Some("turtle".to_string())
        );
        assert_eq!(
            detect_rdf_format_compressed(Path::new("data.nt.bz2")),
            Some("ntriples".to_string())
        );
        assert_eq!(
            detect_rdf_format_compressed(Path::new("data.rdf.xz")),
            Some("rdfxml".to_string())
        );
        assert_eq!(
            detect_rdf_format_compressed(Path::new("data.ttl")),
            Some("turtle".to_string())
        );
    }

    #[test]
    fn test_compression_format_extension() {
        assert_eq!(CompressionFormat::Gzip.extension(), ".gz");
        assert_eq!(CompressionFormat::Bzip2.extension(), ".bz2");
        assert_eq!(CompressionFormat::Xz.extension(), ".xz");
        assert_eq!(CompressionFormat::None.extension(), "");
    }

    #[test]
    fn test_compression_format_name() {
        assert_eq!(CompressionFormat::Gzip.name(), "gzip");
        assert_eq!(CompressionFormat::Bzip2.name(), "bzip2");
        assert_eq!(CompressionFormat::Xz.name(), "xz");
        assert_eq!(CompressionFormat::None.name(), "none");
    }

    /// Round-trip through the gzip writer/reader (oxiarc-deflate backed) and
    /// confirm the on-disk bytes are a real gzip stream (magic 0x1f 0x8b) that
    /// decompresses back to the original payload.
    #[test]
    fn test_gzip_writer_reader_round_trip() {
        use std::env::temp_dir;

        let unique = format!(
            "{}_{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let path = temp_dir().join(format!("oxirs_gzip_rt_{unique}.ttl.gz"));

        // Use a repetitive payload large enough to exercise the compressor.
        let payload = "<urn:s> <urn:p> <urn:o> .\n".repeat(2000);

        // Write through the gzip writer.
        {
            let mut writer =
                open_writer(&path, CompressionFormat::Gzip).expect("failed to open gzip writer");
            writer
                .write_all(payload.as_bytes())
                .expect("failed to write payload");
            writer.flush().expect("failed to flush gzip writer");
        }

        // The raw file must start with the gzip magic bytes.
        let raw = std::fs::read(&path).expect("failed to read gzip file");
        assert!(raw.len() >= 2, "gzip output too small");
        assert_eq!(&raw[..2], &[0x1f, 0x8b], "missing gzip magic header");
        assert!(
            raw.len() < payload.len(),
            "compressed output ({}) should be smaller than input ({})",
            raw.len(),
            payload.len()
        );

        // Read it back through the gzip reader.
        let mut reader = open_reader(&path).expect("failed to open gzip reader");
        let mut decompressed = String::new();
        reader
            .read_to_string(&mut decompressed)
            .expect("failed to read decompressed payload");
        assert_eq!(decompressed, payload, "round-trip payload mismatch");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_strip_compression_suffix() {
        assert_eq!(
            strip_compression_suffix(Path::new("data.ttl.gz")),
            Path::new("data.ttl")
        );
        assert_eq!(
            strip_compression_suffix(Path::new("data.nt.bz2")),
            Path::new("data.nt")
        );
        assert_eq!(
            strip_compression_suffix(Path::new("data.rdf.xz")),
            Path::new("data.rdf")
        );
        // No compression extension: returned unchanged.
        assert_eq!(
            strip_compression_suffix(Path::new("data.ttl")),
            Path::new("data.ttl")
        );
        // Bare compression extension.
        assert_eq!(
            strip_compression_suffix(Path::new("data.gz")),
            Path::new("data")
        );
    }

    /// Corrupt gzip input must surface an explicit decompression error rather
    /// than returning empty/garbage data (fail-loud contract).
    #[test]
    fn test_corrupt_gzip_input_errors() {
        use std::env::temp_dir;

        let unique = format!(
            "{}_{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let path = temp_dir().join(format!("oxirs_gzip_corrupt_{unique}.ttl.gz"));

        // Gzip magic header followed by garbage: looks like gzip by name and by
        // the first two bytes, but the stream is not a valid gzip member.
        std::fs::write(&path, [0x1f, 0x8b, 0x08, 0x00, 0xde, 0xad, 0xbe, 0xef])
            .expect("failed to write corrupt gzip file");

        let result = open_reader(&path);
        // Either open_reader itself fails during eager decompression, or the
        // first read does — in both cases the caller sees an explicit error and
        // never valid-looking empty content.
        match result {
            Err(_) => { /* explicit error at open time: good */ }
            Ok(mut reader) => {
                let mut buf = Vec::new();
                let read_result = reader.read_to_end(&mut buf);
                assert!(
                    read_result.is_err() || buf.is_empty(),
                    "corrupt gzip must not yield fabricated content"
                );
                // The intended contract is an explicit error; assert we did not
                // silently succeed with bytes.
                assert!(
                    read_result.is_err(),
                    "corrupt gzip must produce an explicit read error"
                );
            }
        }

        let _ = std::fs::remove_file(&path);
    }

    /// Ensure a payload written by the gzip writer can be decompressed directly
    /// by `oxiarc_deflate::gzip_decompress`, proving format compatibility.
    #[test]
    fn test_gzip_writer_oxiarc_format_compatible() {
        use std::env::temp_dir;

        let unique = format!(
            "{}_{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let path = temp_dir().join(format!("oxirs_gzip_fmt_{unique}.nt.gz"));
        let payload = b"hello oxirs gzip format compatibility check\n";

        {
            let mut writer =
                open_writer(&path, CompressionFormat::Gzip).expect("failed to open gzip writer");
            writer.write_all(payload).expect("failed to write payload");
            writer.flush().expect("failed to flush gzip writer");
        }

        let raw = std::fs::read(&path).expect("failed to read gzip file");
        let decompressed =
            oxiarc_deflate::gzip_decompress(&raw).expect("oxiarc failed to decompress");
        assert_eq!(decompressed, payload);

        let _ = std::fs::remove_file(&path);
    }
}
