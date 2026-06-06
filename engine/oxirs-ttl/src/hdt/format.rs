//! # HDT Binary Format Constants and Utilities
//!
//! Binary format constants, variable-byte integer encoding/decoding, CRC
//! checksums, and section-header parsing for the HDT 1.0 specification.
//!
//! ## Variable-Byte Integer (vbyte) Encoding
//!
//! Each 7-bit group is stored in one byte, LSB first.  The high bit of each
//! byte signals continuation: `1` means more bytes follow, `0` means the
//! current byte is the last one.
//!
//! ```
//! // Value 300 = 0b_0000001_0101100
//! // byte 0: 0b1_0101100  (7 low bits, continue)
//! // byte 1: 0b0_0000010  (7 high bits, stop)
//! assert_eq!(oxirs_ttl::hdt::format::write_vbyte(300), vec![0b1010_1100, 0b0000_0010]);
//! ```

use std::io::{self, Read};

use super::HdtError;

// ---------------------------------------------------------------------------
// Magic / section identifiers
// ---------------------------------------------------------------------------

/// HDT file magic bytes: `$HDT\r\n\0` (7 bytes).
pub const HDT_MAGIC: &[u8] = b"$HDT\r\n\x00";

/// HDT version 1 section identifier cookie: `$HDTv1\0`.
pub const HDT_SECTION_V1: &[u8] = b"$HDTv1\x00";

/// Section type identifier for the header section.
pub const SECTION_HEADER: &[u8] = b"Header";

/// Section type identifier for the dictionary section.
pub const SECTION_DICTIONARY: &[u8] = b"Dictionary";

/// Section type identifier for the triples section.
pub const SECTION_TRIPLES: &[u8] = b"Triples";

/// Front-coding dictionary suffix terminator constant.
pub const CIT_SUFFIX: u8 = 0x00;

// ---------------------------------------------------------------------------
// Variable-byte integer encoding
// ---------------------------------------------------------------------------

/// Decode a variable-byte integer from `reader`.
///
/// Each byte contributes 7 bits (LSB first).  The high bit indicates
/// continuation: 1 = more bytes follow; 0 = last byte.
///
/// # Errors
/// Returns `io::Error` on read failure or if the value would overflow `u64`.
pub fn read_vbyte<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf)?;
        let byte = buf[0];
        // 7 data bits from this byte
        let bits = (byte & 0x7F) as u64;
        result |= bits << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            // High bit clear → last byte
            break;
        }
        if shift >= 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "vbyte: value overflow (> 63 bits)",
            ));
        }
    }
    Ok(result)
}

/// Decode a variable-byte integer from a byte slice, returning `(value, bytes_consumed)`.
///
/// This is a convenience wrapper over a cursor for callers that already hold
/// the bytes in memory.
///
/// # Errors
/// Returns `HdtError::Io` on truncated input.
pub fn read_vbyte_slice(data: &[u8]) -> Result<(u64, usize), HdtError> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    for (i, &byte) in data.iter().enumerate() {
        let bits = (byte & 0x7F) as u64;
        result |= bits << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        if shift >= 64 {
            return Err(HdtError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "vbyte: value overflow",
            )));
        }
    }
    Err(HdtError::Io(io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "vbyte: unexpected end of data",
    )))
}

/// Encode `value` as a variable-byte integer.
///
/// Each byte stores 7 bits of the value (LSB first).  The high bit of each
/// byte is set to 1 for all bytes except the last.
pub fn write_vbyte(mut value: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte); // high bit clear → last byte
            break;
        } else {
            out.push(byte | 0x80); // high bit set → more bytes follow
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Section header
// ---------------------------------------------------------------------------

/// Read an HDT section header from a byte slice at the given offset.
///
/// Format:
/// ```text
/// "$HDT"  4 bytes literal cookie
/// type    null-terminated ASCII section-type string
/// size    u64 LE (8 bytes) — byte length of the section body
/// crc8    1 byte — consumed but not validated here
/// ```
///
/// Returns `(section_type, byte_length)` and advances `*offset` past the
/// header bytes.
///
/// # Errors
/// Returns `HdtError::InvalidSection` if the cookie is absent or the data
/// is truncated.
pub fn read_section_header(data: &[u8], offset: &mut usize) -> Result<(String, u64), HdtError> {
    const COOKIE: &[u8] = b"$HDT";
    if *offset + COOKIE.len() > data.len() {
        return Err(HdtError::InvalidSection {
            name: format!("data truncated before section cookie at offset {}", offset),
        });
    }
    if &data[*offset..*offset + COOKIE.len()] != COOKIE {
        return Err(HdtError::InvalidSection {
            name: format!(
                "expected $HDT cookie at offset {}, got {:?}",
                offset,
                &data[*offset..*offset + COOKIE.len().min(data.len() - *offset)]
            ),
        });
    }
    *offset += COOKIE.len();

    // Null-terminated section-type string
    let type_start = *offset;
    let type_end = data[*offset..]
        .iter()
        .position(|b| *b == 0)
        .map(|p| *offset + p)
        .ok_or_else(|| HdtError::InvalidSection {
            name: "section type string not null-terminated".to_owned(),
        })?;
    let section_type = std::str::from_utf8(&data[type_start..type_end])
        .map_err(|e| HdtError::InvalidSection {
            name: format!("section type UTF-8 error: {}", e),
        })?
        .to_owned();
    *offset = type_end + 1; // skip the null terminator

    // 8-byte LE size
    if *offset + 8 > data.len() {
        return Err(HdtError::InvalidSection {
            name: "data truncated reading section size".to_owned(),
        });
    }
    let size = u64::from_le_bytes(
        data[*offset..*offset + 8]
            .try_into()
            .map_err(|_| HdtError::InvalidSection {
                name: "cannot read 8-byte size field".to_owned(),
            })?,
    );
    *offset += 8;

    // CRC byte — consumed but not validated
    if *offset >= data.len() {
        return Err(HdtError::InvalidSection {
            name: "data truncated reading section CRC".to_owned(),
        });
    }
    *offset += 1;

    Ok((section_type, size))
}

// ---------------------------------------------------------------------------
// CRC checksums
// ---------------------------------------------------------------------------

/// Compute CRC-16/CCITT-FALSE over `data`.
///
/// Parameters: poly = 0x1021, init = 0xFFFF, RefIn = false, RefOut = false,
/// XorOut = 0x0000.
///
/// Known value: `compute_crc16(b"123456789") == 0x29B1`.
pub fn compute_crc16(data: &[u8]) -> u16 {
    const POLY: u16 = 0x1021;
    let mut crc: u16 = 0xFFFF;
    for &byte in data {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ POLY;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// Compute CRC-32/ISO-HDLC over `data`.
///
/// Parameters: poly = 0x04C11DB7 (reflected = 0xEDB88320), init = 0xFFFFFFFF,
/// RefIn = true, RefOut = true, XorOut = 0xFFFFFFFF.
///
/// Known value: `compute_crc32(b"123456789") == 0xCBF43926`.
pub fn compute_crc32(data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB8_8320; // reflected polynomial
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod format_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_write_vbyte_zero() {
        assert_eq!(write_vbyte(0), vec![0x00]);
    }

    #[test]
    fn test_write_vbyte_one() {
        assert_eq!(write_vbyte(1), vec![0x01]);
    }

    #[test]
    fn test_write_vbyte_127() {
        assert_eq!(write_vbyte(127), vec![0x7F]);
    }

    #[test]
    fn test_write_vbyte_128() {
        // 128 = 0b1000_0000 requires 2 bytes: [0x80, 0x01]
        assert_eq!(write_vbyte(128), vec![0x80, 0x01]);
    }

    #[test]
    fn test_write_vbyte_300() {
        // 300 = 0b1_0010_1100 → bytes [0b1010_1100, 0b0000_0010]
        assert_eq!(write_vbyte(300), vec![0b1010_1100, 0b0000_0010]);
    }

    #[test]
    fn test_read_vbyte_single_byte() {
        let mut cur = Cursor::new(vec![0x2A]);
        assert_eq!(read_vbyte(&mut cur).unwrap(), 42);
    }

    #[test]
    fn test_read_vbyte_multi_byte() {
        let encoded = write_vbyte(300);
        let mut cur = Cursor::new(encoded);
        assert_eq!(read_vbyte(&mut cur).unwrap(), 300);
    }

    #[test]
    fn test_vbyte_roundtrip() {
        for v in [0u64, 1, 127, 128, 255, 1024, 16383, 16384, u32::MAX as u64] {
            let encoded = write_vbyte(v);
            let mut cur = Cursor::new(encoded);
            assert_eq!(read_vbyte(&mut cur).unwrap(), v);
        }
    }

    #[test]
    fn test_crc16_known_value() {
        assert_eq!(compute_crc16(b"123456789"), 0x29B1);
    }

    #[test]
    fn test_crc32_known_value() {
        assert_eq!(compute_crc32(b"123456789"), 0xCBF4_3926);
    }
}
