//! Pure-Rust `zstd`-API shim backed by [`oxiarc_zstd`].
//!
//! This crate is **NOT** the upstream `zstd` crate. It is a drop-in,
//! source-compatible shim that re-implements the subset of the `zstd` public
//! surface — `bulk`, `stream`, and the small `zstd_safe`-style helpers — that
//! downstream consumers (tantivy, parquet, pulsar, wasmtime) actually use, on
//! top of the Pure-Rust [`oxiarc_zstd`] encoder/decoder.
//!
//! It is wired into the oxirs build via a local `[patch.crates-io]` override so
//! that the C-backed `zstd-sys` dependency is eliminated, satisfying the
//! COOLJAPAN Pure Rust Policy v2.
//!
//! Compressed output is a STANDARD Zstandard frame (magic `28 B5 2F FD`) with
//! `Frame_Content_Size` written, and decompression consumes standard frames, so
//! data produced by this shim interoperates with conforming Zstandard
//! implementations and vice versa.

/// Default compression level used when callers pass a non-positive level.
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 3;

/// Inclusive range of compression levels supported by this shim.
pub fn compression_level_range() -> std::ops::RangeInclusive<i32> {
    1..=22
}

pub mod bulk;
pub mod stream;
pub mod zstd_safe;

pub use stream::{decode_all, encode_all, Decoder, Encoder};

#[cfg(test)]
mod tests {
    #[test]
    fn bulk_roundtrip_appends() {
        let data = b"the quick brown fox jumps over the lazy dog, repeated".repeat(20);

        let mut comp = crate::bulk::Compressor::new(3).expect("new");
        let mut dst = vec![0xAAu8; 3];
        let pre = dst.len();
        let n = comp.compress_to_buffer(&data, &mut dst).expect("compress");
        assert_eq!(dst.len(), pre + n);

        let mut dec = crate::bulk::Decompressor::new().expect("dec");
        // `Decompressor::decompress_to_buffer` bounds allocation by the spare
        // capacity already reserved in `destination` (decompression-bomb
        // guard), mirroring the real `zstd::bulk::Decompressor` contract, so
        // callers must reserve up front rather than relying on unbounded
        // auto-growth.
        let mut out = Vec::with_capacity(data.len());
        dec.decompress_to_buffer(&dst[pre..], &mut out).expect("decompress");
        assert_eq!(out, data);

        let compressed = crate::bulk::compress(&data, 3).expect("free compress");
        let restored = crate::bulk::decompress(&compressed, data.len()).expect("free decompress");
        assert_eq!(restored, data);
    }

    #[test]
    fn bulk_slice_buffers() {
        let src = (0..5000u32).map(|i| (i % 251) as u8).collect::<Vec<u8>>();
        let mut cdst = vec![0u8; crate::zstd_safe::compress_bound(src.len())];
        let n = crate::bulk::compress_to_buffer(&src, &mut cdst, 3).expect("c");
        let mut ddst = vec![0u8; src.len()];
        let m = crate::bulk::decompress_to_buffer(&cdst[..n], &mut ddst).expect("d");
        assert_eq!(m, src.len());
        assert_eq!(ddst[..m], src[..]);
    }

    #[test]
    fn stream_encode_decode_all() {
        let data = b"stream payload ".repeat(100);
        let c = crate::encode_all(&data[..], 3).expect("e");
        let d = crate::decode_all(&c[..]).expect("d");
        assert_eq!(d, data);

        let writer: Vec<u8> = Vec::new();
        let mut enc = crate::stream::Encoder::new(writer, 3).expect("enc new");
        {
            use std::io::Write;
            enc.write_all(&data).expect("write_all");
        }
        let framed = enc.finish().expect("finish");

        let mut dec = crate::stream::Decoder::new(&framed[..]).expect("dec new");
        let mut out = Vec::new();
        {
            use std::io::Read;
            dec.read_to_end(&mut out).expect("read_to_end");
        }
        assert_eq!(out, data);
    }

    #[test]
    fn frame_content_size() {
        for &size in &[10usize, 200, 1000, 50_000, 100_000] {
            let payload = (0..size).map(|i| (i % 251) as u8).collect::<Vec<u8>>();
            let comp = oxiarc_zstd::compress_with_level(&payload, 3).expect("c");
            assert_eq!(
                crate::zstd_safe::get_frame_content_size(&comp).expect("fcs"),
                Some(size as u64),
                "size {size}"
            );
        }

        assert!(crate::zstd_safe::get_frame_content_size(&[0, 1, 2, 3, 4]).is_err());
        assert!(crate::zstd_safe::get_frame_content_size(&[]).is_err());
    }

    #[test]
    fn compress_bound_holds() {
        for &n in &[0usize, 1, 1000, 1 << 20] {
            let src = (0..n).map(|i| (i % 251) as u8).collect::<Vec<u8>>();
            let c = oxiarc_zstd::compress_with_level(&src, 3).expect("c");
            assert!(crate::zstd_safe::compress_bound(n) >= c.len(), "n {n}");
        }
    }

    #[test]
    fn upper_bound_some() {
        let payload = b"hello upper bound".repeat(10);
        let comp = oxiarc_zstd::compress_with_level(&payload, 3).expect("c");
        assert_eq!(crate::bulk::Decompressor::upper_bound(&comp), Some(payload.len()));
    }
}
