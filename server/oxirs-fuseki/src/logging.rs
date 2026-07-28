//! Logging initialization wired to [`crate::config::LoggingConfig`].
//!
//! Previously the server always called `tracing_subscriber::fmt::init()`, which
//! ignored every configured logging setting — an operator asking for JSON logs
//! rotated to a file silently got default text logs on stdout. This module
//! honours the config for real:
//!
//! * `format`: `Text` (full), `Json`, or `Compact`.
//! * `output`: `Stdout`, `Stderr`, `File`, or `Both` (stdout **and** file).
//! * `file_config`: size-based rotation (`max_size_mb`, `max_files`) with
//!   optional gzip compression of rotated segments (Pure-Rust `oxiarc-deflate`).
//! * `level`: an `EnvFilter` directive (falls back to `RUST_LOG`, then `info`).

use crate::config::{FileLogConfig, LogFormat, LogOutput, LoggingConfig};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use tracing_subscriber::EnvFilter;

/// A file writer that rotates when the active file exceeds a byte threshold,
/// keeping at most `max_files` historical segments and optionally gzip-compressing
/// rotated segments.
struct RotatingFile {
    path: PathBuf,
    file: File,
    written: u64,
    max_bytes: u64,
    max_files: usize,
    compress: bool,
}

impl RotatingFile {
    fn open(config: &FileLogConfig) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.path)?;
        let written = file.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(Self {
            path: config.path.clone(),
            file,
            written,
            max_bytes: config.max_size_mb.saturating_mul(1024 * 1024).max(1),
            max_files: config.max_files.max(1),
            compress: config.compress,
        })
    }

    /// Rotate `path` → `path.1[.gz]`, shifting existing segments up and dropping
    /// any beyond `max_files`.
    fn rotate(&mut self) -> io::Result<()> {
        let ext = if self.compress { "gz" } else { "" };
        let seg = |n: usize| -> PathBuf {
            if ext.is_empty() {
                PathBuf::from(format!("{}.{n}", self.path.display()))
            } else {
                PathBuf::from(format!("{}.{n}.{ext}", self.path.display()))
            }
        };

        // Drop the oldest, then shift each segment up by one.
        let oldest = seg(self.max_files);
        if oldest.exists() {
            std::fs::remove_file(&oldest)?;
        }
        for n in (1..self.max_files).rev() {
            let from = seg(n);
            if from.exists() {
                std::fs::rename(&from, seg(n + 1))?;
            }
        }

        // Move the active file to segment 1 (compressing if requested), then
        // reopen a fresh active file.
        if self.compress {
            let raw = std::fs::read(&self.path)?;
            let compressed = oxiarc_deflate::gzip_compress(&raw, 6)
                .map_err(|e| io::Error::other(format!("gzip rotate failed: {e}")))?;
            std::fs::write(seg(1), compressed)?;
            std::fs::remove_file(&self.path)?;
        } else {
            std::fs::rename(&self.path, seg(1))?;
        }

        self.file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        self.written = 0;
        Ok(())
    }
}

impl Write for RotatingFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.written + buf.len() as u64 > self.max_bytes && self.written > 0 {
            self.rotate()?;
        }
        let n = self.file.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

/// A cloneable handle onto a shared [`RotatingFile`], usable as an `io::Write`
/// sink for the tracing writer closure.
#[derive(Clone)]
struct RotatingHandle(Arc<Mutex<RotatingFile>>);

impl Write for RotatingHandle {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self.0.lock() {
            Ok(mut f) => f.write(buf),
            Err(_) => Ok(buf.len()),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self.0.lock() {
            Ok(mut f) => f.flush(),
            Err(_) => Ok(()),
        }
    }
}

/// Writes to stdout AND a rotating file simultaneously (`LogOutput::Both`).
struct TeeWriter {
    file: RotatingHandle,
}

impl Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let _ = io::stdout().write_all(buf);
        let _ = self.file.clone().write_all(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        let _ = io::stdout().flush();
        self.file.flush()
    }
}

/// Build the byte-sink [`BoxMakeWriter`] for the configured output. Returns an
/// error only when a file sink is requested but cannot be opened (fail-loud:
/// the operator asked for file logging and must know it failed).
fn make_writer(config: &LoggingConfig) -> io::Result<BoxMakeWriter> {
    let open_file = |fc: &FileLogConfig| -> io::Result<RotatingHandle> {
        if let Some(parent) = fc.path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        Ok(RotatingHandle(Arc::new(Mutex::new(RotatingFile::open(
            fc,
        )?))))
    };

    Ok(match config.output {
        LogOutput::Stdout => BoxMakeWriter::new(io::stdout),
        LogOutput::Stderr => BoxMakeWriter::new(io::stderr),
        LogOutput::File => {
            let fc = require_file_config(config)?;
            let handle = open_file(fc)?;
            BoxMakeWriter::new(move || handle.clone())
        }
        LogOutput::Both => {
            let fc = require_file_config(config)?;
            let handle = open_file(fc)?;
            BoxMakeWriter::new(move || TeeWriter {
                file: handle.clone(),
            })
        }
    })
}

/// `File`/`Both` output requires `file_config`; its absence is a fail-loud error.
fn require_file_config(config: &LoggingConfig) -> io::Result<&FileLogConfig> {
    config.file_config.as_ref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "logging.output requests file output but logging.file_config is not set",
        )
    })
}

/// Build the level filter from `logging.level`, falling back to `RUST_LOG` then
/// `info` when the directive is empty/invalid.
fn make_filter(level: &str) -> EnvFilter {
    EnvFilter::try_new(level)
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new("info"))
}

/// Initialize the global tracing subscriber from the logging config.
///
/// Applies format, output routing (including file rotation/compression) and the
/// level filter. A file-output failure surfaces as an error rather than silently
/// downgrading to stdout.
pub fn init(config: &LoggingConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let filter = make_filter(&config.level);
    let writer = make_writer(config)?;
    // ANSI colors only make sense on a terminal; disable for file/both sinks.
    let ansi = matches!(config.output, LogOutput::Stdout | LogOutput::Stderr);

    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(writer)
        .with_ansi(ansi);

    match config.format {
        LogFormat::Json => builder.json().try_init()?,
        LogFormat::Compact => builder.compact().try_init()?,
        LogFormat::Text => builder.try_init()?,
    }
    Ok(())
}

/// Path of the Nth rotated segment (test helper mirror of [`RotatingFile::rotate`]).
#[cfg(test)]
fn segment_path(base: &Path, n: usize, compress: bool) -> PathBuf {
    if compress {
        PathBuf::from(format!("{}.{n}.gz", base.display()))
    } else {
        PathBuf::from(format!("{}.{n}", base.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_log_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "oxirs_fuseki_log_test_{}_{}",
            std::process::id(),
            name
        ));
        p
    }

    #[test]
    fn regression_rotating_file_rotates_and_caps_segments() {
        let base = temp_log_path("rotate");
        let _ = std::fs::remove_file(&base);
        for n in 1..=4 {
            let _ = std::fs::remove_file(segment_path(&base, n, false));
        }

        let fc = FileLogConfig {
            path: base.clone(),
            max_size_mb: 1, // 1 MiB threshold
            max_files: 2,
            compress: false,
        };
        let mut rf = RotatingFile::open(&fc).expect("open");
        // Force a small threshold so we exercise rotation deterministically.
        rf.max_bytes = 64;

        let chunk = vec![b'x'; 40];
        for _ in 0..10 {
            rf.write_all(&chunk).expect("write");
        }
        rf.flush().expect("flush");

        // The active file exists and at least one rotated segment was created,
        // but never more than max_files.
        assert!(base.exists(), "active log file must exist");
        assert!(
            segment_path(&base, 1, false).exists(),
            "first rotated segment must exist"
        );
        assert!(
            !segment_path(&base, 3, false).exists(),
            "segments beyond max_files must not exist"
        );

        // Cleanup.
        let _ = std::fs::remove_file(&base);
        for n in 1..=4 {
            let _ = std::fs::remove_file(segment_path(&base, n, false));
        }
    }

    #[test]
    fn regression_file_output_requires_file_config() {
        let config = LoggingConfig {
            level: "info".to_string(),
            format: LogFormat::Text,
            output: LogOutput::File,
            file_config: None,
        };
        assert!(make_writer(&config).is_err());
    }
}
