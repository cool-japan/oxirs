//! SWRL Encoding Built-in Functions
//!
//! This module implements encoding/decoding operations for SWRL rules including:
//! - Hashing: hash
//! - Base64: base64_encode, base64_decode
//! - URI operations: encode_uri, decode_uri, resolve_uri

use anyhow::Result;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::super::types::SwrlArgument;
use super::utils::*;

pub(crate) fn builtin_hash(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "hash requires exactly 2 arguments: input, result"
        ));
    }

    let input = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    let hash_value = hasher.finish();

    Ok(hash_value.to_string() == expected)
}

pub(crate) fn builtin_base64_encode(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("base64Encode requires exactly 2 arguments"));
    }

    let input = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple base64 encoding using standard library
    let encoded = base64_encode_simple(&input);
    Ok(encoded == expected)
}

pub(crate) fn builtin_base64_decode(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("base64Decode requires exactly 2 arguments"));
    }

    let input = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple base64 decoding
    match base64_decode_simple(&input) {
        Ok(decoded) => Ok(decoded == expected),
        Err(_) => Ok(false),
    }
}

pub(crate) fn builtin_encode_uri(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "encodeURI requires exactly 2 arguments: input, result"
        ));
    }

    let input = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple URL encoding
    let encoded: String = input
        .chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            _ => format!("%{:02X}", c as u8),
        })
        .collect();

    Ok(encoded == expected)
}

pub(crate) fn builtin_decode_uri(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "decodeURI requires exactly 2 arguments: input, result"
        ));
    }

    let input = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple URL decoding
    let mut decoded = String::new();
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                decoded.push(byte as char);
            } else {
                decoded.push(c);
                decoded.push_str(&hex);
            }
        } else {
            decoded.push(c);
        }
    }

    Ok(decoded == expected)
}

pub(crate) fn builtin_resolve_uri(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "resolveURI requires exactly 3 arguments: base, relative, result"
        ));
    }

    let base = extract_string_value(&args[0])?;
    let relative = extract_string_value(&args[1])?;
    let expected = extract_string_value(&args[2])?;

    // Simple URI resolution
    let result = if relative.starts_with("http://") || relative.starts_with("https://") {
        relative
    } else if relative.starts_with('/') {
        // Extract scheme and host from base
        if let Some(idx) = base.find("://") {
            let scheme_host = &base[..idx + 3];
            if let Some(host_end) = base[idx + 3..].find('/') {
                format!(
                    "{}{}{}",
                    scheme_host,
                    &base[idx + 3..idx + 3 + host_end],
                    relative
                )
            } else {
                format!("{}{}{}", scheme_host, &base[idx + 3..], relative)
            }
        } else {
            relative
        }
    } else {
        // Relative to current path
        if let Some(last_slash) = base.rfind('/') {
            format!("{}/{}", &base[..last_slash], relative)
        } else {
            relative
        }
    };

    Ok(result == expected)
}

// Helper functions for base64

fn base64_encode_simple(input: &str) -> String {
    const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = input.as_bytes();
    let mut result = String::new();

    for chunk in bytes.chunks(3) {
        let mut buf = [0u8; 3];
        for (i, &b) in chunk.iter().enumerate() {
            buf[i] = b;
        }

        result.push(BASE64_CHARS[(buf[0] >> 2) as usize] as char);
        result.push(BASE64_CHARS[(((buf[0] & 0x03) << 4) | (buf[1] >> 4)) as usize] as char);

        if chunk.len() > 1 {
            result.push(BASE64_CHARS[(((buf[1] & 0x0f) << 2) | (buf[2] >> 6)) as usize] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(BASE64_CHARS[(buf[2] & 0x3f) as usize] as char);
        } else {
            result.push('=');
        }
    }

    result
}

fn base64_decode_simple(input: &str) -> Result<String> {
    let input = input.trim_end_matches('=');
    let mut bytes = Vec::new();

    for chunk in input.as_bytes().chunks(4) {
        let vals: Vec<u8> = chunk
            .iter()
            .map(|&b| match b {
                b'A'..=b'Z' => b - b'A',
                b'a'..=b'z' => b - b'a' + 26,
                b'0'..=b'9' => b - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                _ => 0,
            })
            .collect();

        if !vals.is_empty() {
            bytes.push((vals[0] << 2) | (vals.get(1).unwrap_or(&0) >> 4));
        }
        if vals.len() > 2 {
            bytes.push((vals[1] << 4) | (vals[2] >> 2));
        }
        if vals.len() > 3 {
            bytes.push((vals[2] << 6) | vals[3]);
        }
    }

    String::from_utf8(bytes).map_err(|e| anyhow::anyhow!("UTF-8 decode error: {}", e))
}
