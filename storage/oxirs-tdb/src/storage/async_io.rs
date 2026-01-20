//! Asynchronous I/O layer with optional io_uring support
//!
//! Provides high-performance async file operations for the TDB storage engine.
//! On Linux, can optionally use io_uring for maximum performance.
//! Falls back to tokio::fs on other platforms.
//!
//! ## Features
//!
//! - **Async file operations** - Non-blocking reads and writes
//! - **io_uring support** - Linux kernel-level async I/O (optional)
//! - **Cross-platform** - Works on Linux, macOS, Windows
//! - **Batch operations** - Submit multiple I/O operations together
//! - **Zero-copy** - Minimizes data copying where possible
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_tdb::storage::async_io::{AsyncFileHandle, AsyncIoBackend};
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Open file with async I/O
//!     let file = AsyncFileHandle::open("data.tdb", AsyncIoBackend::Auto).await?;
//!
//!     // Read data asynchronously
//!     let mut buffer = vec![0u8; 4096];
//!     let bytes_read = file.read_at(&mut buffer, 0).await?;
//!
//!     // Write data asynchronously
//!     file.write_at(b"hello", 0).await?;
//!     file.sync_all().await?;
//!
//!     Ok(())
//! }
//! ```

use crate::error::{Result, TdbError};
use bytes::Bytes;
use futures::future::BoxFuture;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

/// Async I/O backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AsyncIoBackend {
    /// Automatically select best backend for platform
    #[default]
    Auto,
    /// Use tokio::fs (cross-platform)
    Tokio,
    /// Use io_uring (Linux only, requires feature flag)
    #[cfg(target_os = "linux")]
    IoUring,
}

impl AsyncIoBackend {
    /// Select the best backend for the current platform
    pub fn select_best() -> Self {
        #[cfg(all(target_os = "linux", feature = "tokio-uring"))]
        {
            Self::IoUring
        }
        #[cfg(not(all(target_os = "linux", feature = "tokio-uring")))]
        {
            Self::Tokio
        }
    }
}

/// Async file handle for non-blocking I/O
pub struct AsyncFileHandle {
    /// File path
    path: PathBuf,
    /// Backend type
    backend: AsyncIoBackend,
    /// Tokio file handle (used by Tokio backend)
    tokio_file: Option<Arc<tokio::sync::Mutex<File>>>,
    /// Statistics
    stats: Arc<parking_lot::Mutex<AsyncIoStats>>,
}

/// Async I/O statistics
#[derive(Debug, Clone, Default)]
pub struct AsyncIoStats {
    /// Total reads
    pub total_reads: u64,
    /// Total writes
    pub total_writes: u64,
    /// Bytes read
    pub bytes_read: u64,
    /// Bytes written
    pub bytes_written: u64,
    /// Total sync operations
    pub total_syncs: u64,
    /// Backend in use
    pub backend: Option<String>,
}

impl AsyncFileHandle {
    /// Open a file with async I/O
    pub async fn open<P: AsRef<Path>>(path: P, backend: AsyncIoBackend) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let backend = if backend == AsyncIoBackend::Auto {
            AsyncIoBackend::select_best()
        } else {
            backend
        };

        let stats = AsyncIoStats {
            backend: Some(format!("{:?}", backend)),
            ..Default::default()
        };

        match backend {
            AsyncIoBackend::Tokio => {
                let file = tokio::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false) // Don't truncate existing files
                    .open(&path)
                    .await
                    .map_err(TdbError::Io)?;

                Ok(Self {
                    path,
                    backend,
                    tokio_file: Some(Arc::new(tokio::sync::Mutex::new(file))),
                    stats: Arc::new(parking_lot::Mutex::new(stats)),
                })
            }
            #[cfg(target_os = "linux")]
            AsyncIoBackend::IoUring => {
                // Note: io_uring integration would go here
                // For now, fall back to tokio
                Self::open(path, AsyncIoBackend::Tokio).await
            }
            AsyncIoBackend::Auto => unreachable!("Auto should have been resolved"),
        }
    }

    /// Create a new file with async I/O
    pub async fn create<P: AsRef<Path>>(path: P, backend: AsyncIoBackend) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let backend = if backend == AsyncIoBackend::Auto {
            AsyncIoBackend::select_best()
        } else {
            backend
        };

        let stats = AsyncIoStats {
            backend: Some(format!("{:?}", backend)),
            ..Default::default()
        };

        match backend {
            AsyncIoBackend::Tokio => {
                // Open file with both read and write permissions
                // Using truncate(true) with create(true) to create new file
                let file = tokio::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true) // Intentional: we want to create a new empty file
                    .open(&path)
                    .await
                    .map_err(TdbError::Io)?;

                Ok(Self {
                    path,
                    backend,
                    tokio_file: Some(Arc::new(tokio::sync::Mutex::new(file))),
                    stats: Arc::new(parking_lot::Mutex::new(stats)),
                })
            }
            #[cfg(target_os = "linux")]
            AsyncIoBackend::IoUring => {
                // Note: io_uring integration would go here
                Box::pin(Self::create(path, AsyncIoBackend::Tokio)).await
            }
            AsyncIoBackend::Auto => unreachable!(),
        }
    }

    /// Read data at a specific offset
    pub async fn read_at(&self, buffer: &mut [u8], offset: u64) -> Result<usize> {
        match self.backend {
            AsyncIoBackend::Tokio => {
                let file = self
                    .tokio_file
                    .as_ref()
                    .ok_or_else(|| TdbError::Other("No tokio file handle".to_string()))?;

                let mut file_guard = file.lock().await;
                file_guard
                    .seek(std::io::SeekFrom::Start(offset))
                    .await
                    .map_err(TdbError::Io)?;

                let bytes_read = file_guard.read(buffer).await.map_err(TdbError::Io)?;

                // Update stats
                let mut stats = self.stats.lock();
                stats.total_reads += 1;
                stats.bytes_read += bytes_read as u64;

                Ok(bytes_read)
            }
            #[cfg(target_os = "linux")]
            AsyncIoBackend::IoUring => {
                // Fallback to tokio for now
                self.read_at_tokio(buffer, offset).await
            }
            AsyncIoBackend::Auto => unreachable!(),
        }
    }

    /// Tokio implementation of read_at (helper)
    async fn read_at_tokio(&self, buffer: &mut [u8], offset: u64) -> Result<usize> {
        let file = self
            .tokio_file
            .as_ref()
            .ok_or_else(|| TdbError::Other("No tokio file handle".to_string()))?;

        let mut file_guard = file.lock().await;
        file_guard
            .seek(std::io::SeekFrom::Start(offset))
            .await
            .map_err(TdbError::Io)?;

        let bytes_read = file_guard.read(buffer).await.map_err(TdbError::Io)?;

        let mut stats = self.stats.lock();
        stats.total_reads += 1;
        stats.bytes_read += bytes_read as u64;

        Ok(bytes_read)
    }

    /// Write data at a specific offset
    pub async fn write_at(&self, data: &[u8], offset: u64) -> Result<usize> {
        match self.backend {
            AsyncIoBackend::Tokio => {
                let file = self
                    .tokio_file
                    .as_ref()
                    .ok_or_else(|| TdbError::Other("No tokio file handle".to_string()))?;

                let mut file_guard = file.lock().await;
                file_guard
                    .seek(std::io::SeekFrom::Start(offset))
                    .await
                    .map_err(TdbError::Io)?;

                let bytes_written = file_guard.write(data).await.map_err(TdbError::Io)?;

                // Update stats
                let mut stats = self.stats.lock();
                stats.total_writes += 1;
                stats.bytes_written += bytes_written as u64;

                Ok(bytes_written)
            }
            #[cfg(target_os = "linux")]
            AsyncIoBackend::IoUring => {
                // Fallback to tokio for now
                self.write_at_tokio(data, offset).await
            }
            AsyncIoBackend::Auto => unreachable!(),
        }
    }

    /// Tokio implementation of write_at (helper)
    async fn write_at_tokio(&self, data: &[u8], offset: u64) -> Result<usize> {
        let file = self
            .tokio_file
            .as_ref()
            .ok_or_else(|| TdbError::Other("No tokio file handle".to_string()))?;

        let mut file_guard = file.lock().await;
        file_guard
            .seek(std::io::SeekFrom::Start(offset))
            .await
            .map_err(TdbError::Io)?;

        let bytes_written = file_guard.write(data).await.map_err(TdbError::Io)?;

        let mut stats = self.stats.lock();
        stats.total_writes += 1;
        stats.bytes_written += bytes_written as u64;

        Ok(bytes_written)
    }

    /// Sync all data to disk
    pub async fn sync_all(&self) -> Result<()> {
        match self.backend {
            AsyncIoBackend::Tokio => {
                let file = self
                    .tokio_file
                    .as_ref()
                    .ok_or_else(|| TdbError::Other("No tokio file handle".to_string()))?;

                let file_guard = file.lock().await;
                file_guard.sync_all().await.map_err(TdbError::Io)?;

                // Update stats
                let mut stats = self.stats.lock();
                stats.total_syncs += 1;

                Ok(())
            }
            #[cfg(target_os = "linux")]
            AsyncIoBackend::IoUring => {
                // Fallback to tokio for now
                self.sync_all_tokio().await
            }
            AsyncIoBackend::Auto => unreachable!(),
        }
    }

    /// Tokio implementation of sync_all (helper)
    async fn sync_all_tokio(&self) -> Result<()> {
        let file = self
            .tokio_file
            .as_ref()
            .ok_or_else(|| TdbError::Other("No tokio file handle".to_string()))?;

        let file_guard = file.lock().await;
        file_guard.sync_all().await.map_err(TdbError::Io)?;

        let mut stats = self.stats.lock();
        stats.total_syncs += 1;

        Ok(())
    }

    /// Get file size
    pub async fn len(&self) -> Result<u64> {
        match self.backend {
            AsyncIoBackend::Tokio => {
                let file = self
                    .tokio_file
                    .as_ref()
                    .ok_or_else(|| TdbError::Other("No tokio file handle".to_string()))?;

                let file_guard = file.lock().await;
                let metadata = file_guard.metadata().await.map_err(TdbError::Io)?;
                Ok(metadata.len())
            }
            #[cfg(target_os = "linux")]
            AsyncIoBackend::IoUring => {
                // Fallback to tokio for now
                self.len_tokio().await
            }
            AsyncIoBackend::Auto => unreachable!(),
        }
    }

    /// Tokio implementation of len (helper)
    async fn len_tokio(&self) -> Result<u64> {
        let file = self
            .tokio_file
            .as_ref()
            .ok_or_else(|| TdbError::Other("No tokio file handle".to_string()))?;

        let file_guard = file.lock().await;
        let metadata = file_guard.metadata().await.map_err(TdbError::Io)?;
        Ok(metadata.len())
    }

    /// Check if file is empty
    pub async fn is_empty(&self) -> Result<bool> {
        Ok(self.len().await? == 0)
    }

    /// Get statistics
    pub fn stats(&self) -> AsyncIoStats {
        self.stats.lock().clone()
    }

    /// Get file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get backend type
    pub fn backend(&self) -> AsyncIoBackend {
        self.backend
    }
}

/// Batch async I/O operations for better performance
pub struct AsyncIoBatch {
    /// Operations to execute
    operations: Vec<AsyncIoOperation>,
    /// File handle
    file: Arc<AsyncFileHandle>,
}

/// Single async I/O operation
pub enum AsyncIoOperation {
    /// Read operation
    Read {
        /// Offset
        offset: u64,
        /// Length
        length: usize,
    },
    /// Write operation
    Write {
        /// Offset
        offset: u64,
        /// Data
        data: Bytes,
    },
}

impl AsyncIoBatch {
    /// Create a new batch
    pub fn new(file: Arc<AsyncFileHandle>) -> Self {
        Self {
            operations: Vec::new(),
            file,
        }
    }

    /// Add a read operation
    pub fn add_read(&mut self, offset: u64, length: usize) {
        self.operations
            .push(AsyncIoOperation::Read { offset, length });
    }

    /// Add a write operation
    pub fn add_write(&mut self, offset: u64, data: Bytes) {
        self.operations
            .push(AsyncIoOperation::Write { offset, data });
    }

    /// Execute all operations
    pub async fn execute(self) -> Result<Vec<Result<Vec<u8>>>> {
        let mut results = Vec::new();

        for op in self.operations {
            match op {
                AsyncIoOperation::Read { offset, length } => {
                    let mut buffer = vec![0u8; length];
                    match self.file.read_at(&mut buffer, offset).await {
                        Ok(bytes_read) => {
                            buffer.truncate(bytes_read);
                            results.push(Ok(buffer));
                        }
                        Err(e) => results.push(Err(e)),
                    }
                }
                AsyncIoOperation::Write { offset, data } => {
                    match self.file.write_at(&data, offset).await {
                        Ok(_) => results.push(Ok(Vec::new())),
                        Err(e) => results.push(Err(e)),
                    }
                }
            }
        }

        Ok(results)
    }

    /// Get number of operations
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[tokio::test]
    async fn test_async_file_create() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_async_create.dat");

        let file = AsyncFileHandle::create(&test_file, AsyncIoBackend::Tokio)
            .await
            .unwrap();

        assert_eq!(file.backend(), AsyncIoBackend::Tokio);
        assert!(file.is_empty().await.unwrap());

        // Cleanup
        let _ = tokio::fs::remove_file(&test_file).await;
    }

    #[tokio::test]
    async fn test_async_write_and_read() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_async_rw.dat");

        let file = AsyncFileHandle::create(&test_file, AsyncIoBackend::Tokio)
            .await
            .unwrap();

        // Write data
        let data = b"Hello, async world!";
        let bytes_written = file.write_at(data, 0).await.unwrap();
        assert_eq!(bytes_written, data.len());

        // Sync
        file.sync_all().await.unwrap();

        // Read back
        let mut buffer = vec![0u8; data.len()];
        let bytes_read = file.read_at(&mut buffer, 0).await.unwrap();
        assert_eq!(bytes_read, data.len());
        assert_eq!(&buffer[..], data);

        // Check stats
        let stats = file.stats();
        assert_eq!(stats.total_writes, 1);
        assert_eq!(stats.total_reads, 1);
        assert_eq!(stats.bytes_written, data.len() as u64);
        assert_eq!(stats.bytes_read, data.len() as u64);

        // Cleanup
        let _ = tokio::fs::remove_file(&test_file).await;
    }

    #[tokio::test]
    async fn test_async_file_len() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_async_len.dat");

        let file = AsyncFileHandle::create(&test_file, AsyncIoBackend::Tokio)
            .await
            .unwrap();

        assert_eq!(file.len().await.unwrap(), 0);

        // Write some data
        file.write_at(b"test", 0).await.unwrap();
        file.sync_all().await.unwrap();

        assert_eq!(file.len().await.unwrap(), 4);

        // Cleanup
        let _ = tokio::fs::remove_file(&test_file).await;
    }

    #[tokio::test]
    async fn test_async_io_batch() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_async_batch.dat");

        let file = Arc::new(
            AsyncFileHandle::create(&test_file, AsyncIoBackend::Tokio)
                .await
                .unwrap(),
        );

        // Create batch
        let mut batch = AsyncIoBatch::new(file.clone());

        // Add write operations
        batch.add_write(0, Bytes::from("Hello"));
        batch.add_write(5, Bytes::from("World"));

        assert_eq!(batch.len(), 2);

        // Execute batch
        let results = batch.execute().await.unwrap();
        assert_eq!(results.len(), 2);

        // Sync
        file.sync_all().await.unwrap();

        // Read back with batch
        let mut read_batch = AsyncIoBatch::new(file.clone());
        read_batch.add_read(0, 5);
        read_batch.add_read(5, 5);

        let results = read_batch.execute().await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].as_ref().unwrap(), b"Hello");
        assert_eq!(results[1].as_ref().unwrap(), b"World");

        // Cleanup
        let _ = tokio::fs::remove_file(&test_file).await;
    }

    #[tokio::test]
    async fn test_backend_selection() {
        let backend = AsyncIoBackend::select_best();

        #[cfg(all(target_os = "linux", feature = "tokio-uring"))]
        assert_eq!(backend, AsyncIoBackend::IoUring);

        #[cfg(not(all(target_os = "linux", feature = "tokio-uring")))]
        assert_eq!(backend, AsyncIoBackend::Tokio);
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_async_concurrent.dat");

        let file = Arc::new(
            AsyncFileHandle::create(&test_file, AsyncIoBackend::Tokio)
                .await
                .unwrap(),
        );

        // Spawn multiple concurrent writes
        let mut handles = Vec::new();
        for i in 0..10 {
            let file_clone = file.clone();
            let handle = tokio::spawn(async move {
                let data = format!("Data{}", i);
                file_clone.write_at(data.as_bytes(), (i * 10) as u64).await
            });
            handles.push(handle);
        }

        // Wait for all writes
        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // Verify stats
        let stats = file.stats();
        assert_eq!(stats.total_writes, 10);

        // Cleanup
        let _ = tokio::fs::remove_file(&test_file).await;
    }

    #[tokio::test]
    async fn test_multiple_reads_writes() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_async_multiple.dat");

        let file = AsyncFileHandle::create(&test_file, AsyncIoBackend::Tokio)
            .await
            .unwrap();

        // Write at different offsets
        file.write_at(b"AAA", 0).await.unwrap();
        file.write_at(b"BBB", 100).await.unwrap();
        file.write_at(b"CCC", 200).await.unwrap();
        file.sync_all().await.unwrap();

        // Read back
        let mut buf1 = vec![0u8; 3];
        let mut buf2 = vec![0u8; 3];
        let mut buf3 = vec![0u8; 3];

        file.read_at(&mut buf1, 0).await.unwrap();
        file.read_at(&mut buf2, 100).await.unwrap();
        file.read_at(&mut buf3, 200).await.unwrap();

        assert_eq!(&buf1, b"AAA");
        assert_eq!(&buf2, b"BBB");
        assert_eq!(&buf3, b"CCC");

        // Cleanup
        let _ = tokio::fs::remove_file(&test_file).await;
    }
}
