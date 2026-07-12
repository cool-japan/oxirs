//! Streaming compression/decompression surface compatible with `zstd::stream`.
//!
//! [`Encoder`] buffers all written bytes and emits a single STANDARD Zstandard
//! frame on [`Encoder::finish`]. [`Decoder`] eagerly decompresses its source on
//! construction and then serves the plaintext through [`std::io::Read`]. Both
//! the free [`encode_all`]/[`decode_all`] helpers and the reader/writer adapters
//! are backed by [`oxiarc_zstd`].

use std::marker::PhantomData;

/// Read all of `source`, compress it, and return the resulting frame.
///
/// Non-positive `level` values fall back to [`crate::DEFAULT_COMPRESSION_LEVEL`].
pub fn encode_all<R: std::io::Read>(mut source: R, level: i32) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    source.read_to_end(&mut buf)?;
    let eff = if level <= 0 { crate::DEFAULT_COMPRESSION_LEVEL } else { level };
    oxiarc_zstd::encode_all(&buf, eff).map_err(std::io::Error::other)
}

/// Read all of `source`, decompress it, and return the resulting payload.
pub fn decode_all<R: std::io::Read>(mut source: R) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    source.read_to_end(&mut buf)?;
    oxiarc_zstd::decode_all(&buf).map_err(std::io::Error::other)
}

/// Streaming encoder compatible with `zstd::stream::Encoder`.
///
/// Bytes written through [`std::io::Write`] are buffered until [`finish`](Self::finish)
/// is called, at which point a single Zstandard frame is emitted to the wrapped
/// writer.
pub struct Encoder<'a, W: std::io::Write> {
    writer: W,
    buffer: Vec<u8>,
    level: i32,
    _marker: PhantomData<&'a ()>,
}

impl<'a, W: std::io::Write> Encoder<'a, W> {
    /// Create an encoder that will write its frame to `writer`.
    ///
    /// Non-positive `level` values fall back to [`crate::DEFAULT_COMPRESSION_LEVEL`].
    pub fn new(writer: W, level: i32) -> std::io::Result<Self> {
        Ok(Self { writer, buffer: Vec::new(), level, _marker: PhantomData })
    }

    /// Compress all buffered bytes, write the frame to the wrapped writer, and
    /// return the writer.
    pub fn finish(mut self) -> std::io::Result<W> {
        let eff = if self.level <= 0 { crate::DEFAULT_COMPRESSION_LEVEL } else { self.level };
        let out = oxiarc_zstd::encode_all(&self.buffer, eff).map_err(std::io::Error::other)?;
        self.writer.write_all(&out)?;
        Ok(self.writer)
    }
}

impl<'a, W: std::io::Write> std::io::Write for Encoder<'a, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Streaming decoder compatible with `zstd::stream::Decoder`.
///
/// The source is fully read and decompressed on construction; the plaintext is
/// then served incrementally through [`std::io::Read`].
pub struct Decoder<'a, R: std::io::Read> {
    buffer: Vec<u8>,
    pos: usize,
    _marker: PhantomData<&'a R>,
}

impl<'a, R: std::io::Read> Decoder<'a, R> {
    /// Read all of `reader`, decompress it, and return a decoder positioned at
    /// the start of the plaintext.
    pub fn new(mut reader: R) -> std::io::Result<Self> {
        let mut raw = Vec::new();
        reader.read_to_end(&mut raw)?;
        let buffer = oxiarc_zstd::decode_all(&raw).map_err(std::io::Error::other)?;
        Ok(Self { buffer, pos: 0, _marker: PhantomData })
    }
}

impl<'a, R: std::io::Read> std::io::Read for Decoder<'a, R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let remaining = &self.buffer[self.pos..];
        let n = remaining.len().min(out.len());
        out[..n].copy_from_slice(&remaining[..n]);
        self.pos += n;
        Ok(n)
    }
}
