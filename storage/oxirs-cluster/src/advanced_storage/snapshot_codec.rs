//! On-disk encoding of snapshot payloads.
//!
//! Since 0.5.0 every payload starts with a header that names its encoding,
//! so a snapshot is decoded exactly the way it was written:
//!
//! ```text
//! b"OXCLSNP1" | encoding (1 byte) | body
//! ```
//!
//! | encoding | body |
//! |---|---|
//! | 0 | the snapshot bytes as given |
//! | 1 | one LZ4 frame |
//! | 2 | LZ4 blocks of [`CHUNK_SIZE`] input bytes each, after a table: the original length (u64 LE), the block count (u32 LE) and the compressed length of each block (u32 LE) |
//!
//! The header sits inside the at-rest seal when encryption is enabled.
//!
//! Releases up to 0.4.1 wrote the body without a header and recorded
//! nowhere whether it was compressed. Such payloads are recognised by their
//! shape: an LZ4 frame by its magic number, the block layout only when its
//! table accounts for every byte, and anything else as raw bytes. A raw
//! snapshot that happens to have one of those shapes is ambiguous by
//! construction; payloads with a header never are.

use anyhow::{anyhow, Result};
use std::borrow::Cow;

/// Marks a payload written in the format described above.
const SNAPSHOT_MAGIC: &[u8; 8] = b"OXCLSNP1";
/// Snapshots at least this large are compressed as parallel LZ4 blocks.
const PARALLEL_THRESHOLD: usize = 1024 * 1024;
/// Input bytes per LZ4 block in the block layout.
const CHUNK_SIZE: usize = 256 * 1024;
/// Largest snapshot that a single LZ4 frame may decode to.
const MAX_FRAME_OUTPUT: usize = 100 * 1024 * 1024;
/// Bytes of the block table before the per-block lengths.
const TABLE_HEADER_LEN: usize = 12;

/// How the body after the header is encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Encoding {
    Raw = 0,
    Lz4Frame = 1,
    Lz4Blocks = 2,
}

impl Encoding {
    fn from_byte(byte: u8) -> Option<Self> {
        [Self::Raw, Self::Lz4Frame, Self::Lz4Blocks]
            .into_iter()
            .find(|encoding| *encoding as u8 == byte)
    }
}

/// Encode snapshot `data` for disk. With `compress`, the data is
/// LZ4-compressed unless that would not make it smaller.
pub(super) fn encode(data: &[u8], compress: bool) -> Vec<u8> {
    let (encoding, body) = if compress {
        compress_payload(data)
    } else {
        (Encoding::Raw, Cow::Borrowed(data))
    };
    let mut payload = Vec::with_capacity(SNAPSHOT_MAGIC.len() + 1 + body.len());
    payload.extend_from_slice(SNAPSHOT_MAGIC);
    payload.push(encoding as u8);
    payload.extend_from_slice(&body);
    payload
}

/// Decode a payload written by [`encode`] or by a release up to 0.4.1.
///
/// # Errors
///
/// Fails when the header is truncated or names an unknown encoding, or when
/// a compressed body does not decompress to the length it records.
pub(super) fn decode(payload: &[u8]) -> Result<Vec<u8>> {
    let Some(rest) = payload.strip_prefix(SNAPSHOT_MAGIC.as_slice()) else {
        return decode_legacy(payload);
    };
    let (&encoding, body) = rest
        .split_first()
        .ok_or_else(|| anyhow!("snapshot payload ends inside its header"))?;
    match Encoding::from_byte(encoding) {
        Some(Encoding::Raw) => Ok(body.to_vec()),
        Some(Encoding::Lz4Frame) => decompress_frame(body),
        Some(Encoding::Lz4Blocks) => match parse_blocks(body) {
            Some(layout) => decompress_blocks(&layout),
            None => Err(anyhow!(
                "snapshot block table does not match the data that follows it"
            )),
        },
        None => Err(anyhow!("snapshot payload has unknown encoding {encoding}")),
    }
}

/// Decode a payload without a header, as written up to 0.4.1.
fn decode_legacy(payload: &[u8]) -> Result<Vec<u8>> {
    if payload.starts_with(&oxiarc_lz4::LZ4_FRAME_MAGIC.to_le_bytes()) {
        return decompress_frame(payload);
    }
    match parse_blocks(payload) {
        Some(layout) => decompress_blocks(&layout),
        None => Ok(payload.to_vec()),
    }
}

/// Compress `data`, falling back to the raw bytes when compression fails or
/// does not make the data smaller.
fn compress_payload(data: &[u8]) -> (Encoding, Cow<'_, [u8]>) {
    let compressed = if data.len() < PARALLEL_THRESHOLD {
        oxiarc_lz4::compress(data)
            .map(|body| (Encoding::Lz4Frame, body))
            .map_err(|e| anyhow!("{e}"))
    } else {
        compress_blocks(data).map(|body| (Encoding::Lz4Blocks, body))
    };
    match compressed {
        Ok((encoding, body)) if body.len() < data.len() => (encoding, Cow::Owned(body)),
        Ok(_) => (Encoding::Raw, Cow::Borrowed(data)),
        Err(e) => {
            tracing::warn!("snapshot compression failed, storing it uncompressed: {e}");
            (Encoding::Raw, Cow::Borrowed(data))
        }
    }
}

/// Compress `data` as LZ4 blocks of [`CHUNK_SIZE`] input bytes, in parallel.
fn compress_blocks(data: &[u8]) -> Result<Vec<u8>> {
    use rayon::prelude::*;

    let blocks = data
        .par_chunks(CHUNK_SIZE)
        .map(|chunk| oxiarc_lz4::compress_block(chunk).map_err(|e| anyhow!("{e}")))
        .collect::<Result<Vec<_>>>()?;

    let compressed_len: usize = blocks.iter().map(Vec::len).sum();
    let mut body = Vec::with_capacity(TABLE_HEADER_LEN + 4 * blocks.len() + compressed_len);
    body.extend_from_slice(&u64::try_from(data.len())?.to_le_bytes());
    body.extend_from_slice(&u32::try_from(blocks.len())?.to_le_bytes());
    for block in &blocks {
        body.extend_from_slice(&u32::try_from(block.len())?.to_le_bytes());
    }
    for block in &blocks {
        body.extend_from_slice(block);
    }
    Ok(body)
}

/// A body in the block layout, split into its blocks.
struct BlockLayout<'a> {
    original_len: usize,
    blocks: Vec<&'a [u8]>,
}

/// Split `body` into its LZ4 blocks, or return `None` when its table does
/// not describe exactly the bytes that follow it.
fn parse_blocks(body: &[u8]) -> Option<BlockLayout<'_>> {
    let original_len = u64::from_le_bytes(body.get(..8)?.try_into().ok()?);
    let original_len = usize::try_from(original_len).ok()?;
    let count = u32::from_le_bytes(body.get(8..TABLE_HEADER_LEN)?.try_into().ok()?);
    let count = usize::try_from(count).ok()?;
    // Every block but the last holds CHUNK_SIZE input bytes.
    let expected_count = original_len.checked_add(CHUNK_SIZE - 1)? / CHUNK_SIZE;
    if count == 0 || count != expected_count {
        return None;
    }

    let table_end = count.checked_mul(4)?.checked_add(TABLE_HEADER_LEN)?;
    let table = body.get(TABLE_HEADER_LEN..table_end)?;
    let mut rest = body.get(table_end..)?;
    let mut blocks = Vec::with_capacity(count);
    for entry in table.chunks_exact(4) {
        let len = usize::try_from(u32::from_le_bytes(entry.try_into().ok()?)).ok()?;
        if len > rest.len() {
            return None;
        }
        let (block, tail) = rest.split_at(len);
        blocks.push(block);
        rest = tail;
    }
    rest.is_empty().then_some(BlockLayout {
        original_len,
        blocks,
    })
}

/// Decompress the blocks of `layout` in parallel and check that each one
/// yields exactly the input bytes it was made from.
fn decompress_blocks(layout: &BlockLayout<'_>) -> Result<Vec<u8>> {
    use rayon::prelude::*;

    let last = layout.blocks.len() - 1;
    let parts = layout
        .blocks
        .par_iter()
        .enumerate()
        .map(|(index, block)| {
            let expected = if index == last {
                layout.original_len - last * CHUNK_SIZE
            } else {
                CHUNK_SIZE
            };
            let part = oxiarc_lz4::decompress_block(block, expected)
                .map_err(|e| anyhow!("snapshot block {index} does not decompress: {e}"))?;
            if part.len() != expected {
                return Err(anyhow!(
                    "snapshot block {index} decompresses to {} bytes instead of {expected}",
                    part.len()
                ));
            }
            Ok(part)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(parts.concat())
}

/// Decompress one LZ4 frame.
fn decompress_frame(body: &[u8]) -> Result<Vec<u8>> {
    oxiarc_lz4::decompress(body, MAX_FRAME_OUTPUT)
        .map_err(|e| anyhow!("snapshot LZ4 frame does not decompress: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What releases up to 0.4.1 wrote for `data` with compression enabled.
    fn legacy_compressed(data: &[u8]) -> Vec<u8> {
        if data.len() < PARALLEL_THRESHOLD {
            return oxiarc_lz4::compress(data).unwrap();
        }
        let blocks: Vec<Vec<u8>> = data
            .chunks(CHUNK_SIZE)
            .map(|chunk| oxiarc_lz4::compress_block(chunk).unwrap())
            .collect();
        let mut out = (data.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
        for block in &blocks {
            out.extend_from_slice(&(block.len() as u32).to_le_bytes());
        }
        for block in &blocks {
            out.extend_from_slice(block);
        }
        out
    }

    fn large_data() -> Vec<u8> {
        (0..2 * 1024 * 1024 + 77).map(|i| (i % 253) as u8).collect()
    }

    #[test]
    fn test_every_encoding_round_trips() {
        let small = vec![1u8; 10_000];
        let large = large_data();
        for (data, encoding) in [
            (&small, Encoding::Lz4Frame),
            (&large, Encoding::Lz4Blocks),
            (&vec![5u8; 3], Encoding::Raw),
        ] {
            let payload = encode(data, true);
            assert_eq!(payload[SNAPSHOT_MAGIC.len()], encoding as u8);
            assert!(decode(&payload).unwrap() == *data);
        }
        let payload = encode(&small, false);
        assert_eq!(payload[SNAPSHOT_MAGIC.len()], Encoding::Raw as u8);
        assert!(decode(&payload).unwrap() == small);
    }

    #[test]
    fn test_payloads_from_earlier_releases_decode() {
        let small = vec![9u8; 5000];
        let large = large_data();
        assert!(decode(&legacy_compressed(&small)).unwrap() == small);
        assert!(decode(&legacy_compressed(&large)).unwrap() == large);
        // Raw payloads (compression off) come back unchanged.
        assert!(decode(b"raw snapshot bytes").unwrap() == b"raw snapshot bytes");
        assert!(decode(&large).unwrap() == large);
    }

    #[test]
    fn test_damaged_payloads_are_errors() {
        let mut frame = encode(&[3u8; 20_000], true);
        let body_start = SNAPSHOT_MAGIC.len() + 1;
        for byte in &mut frame[body_start + 8..] {
            *byte = 0xA5;
        }
        assert!(decode(&frame).is_err(), "a damaged frame decoded");

        let mut blocks = encode(&large_data(), true);
        let len = blocks.len();
        blocks.truncate(len - 1);
        assert!(decode(&blocks).is_err(), "a truncated block body decoded");

        let mut unknown = encode(b"data", false);
        unknown[SNAPSHOT_MAGIC.len()] = 9;
        assert!(decode(&unknown).is_err(), "an unknown encoding decoded");
        assert!(decode(SNAPSHOT_MAGIC).is_err(), "a bare header decoded");

        let mut legacy_frame = legacy_compressed(&[4u8; 30_000]);
        let len = legacy_frame.len();
        legacy_frame[len / 2..].fill(0x5A);
        assert!(
            decode(&legacy_frame).is_err(),
            "a damaged legacy frame decoded"
        );
    }
}
