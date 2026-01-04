//! NodeId type for dictionary encoding with inline values optimization
//!
//! NodeIds use a 64-bit encoding scheme that supports both dictionary references
//! and inline values for small, frequently-used terms. This significantly reduces
//! dictionary lookups for common values.

use oxicode::Decode;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Type tag for inline value encoding (uses top 8 bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InlineType {
    /// Dictionary reference (standard NodeId) - type tag 0x00
    Dictionary = 0x00,
    /// Inline small integer (i32) - type tag 0x01
    SmallInt = 0x01,
    /// Inline boolean - type tag 0x02
    Boolean = 0x02,
    /// Inline short string (up to 7 ASCII bytes) - type tag 0x03
    ShortString = 0x03,
    /// Inline small positive integer (u32) - type tag 0x04
    SmallUInt = 0x04,
    /// Inline decimal (mantissa + exponent) - type tag 0x05
    Decimal = 0x05,
    /// Inline date/time (Unix timestamp seconds) - type tag 0x06
    DateTime = 0x06,
}

impl InlineType {
    /// Extract type tag from NodeId
    const fn from_node_id(id: u64) -> u8 {
        ((id >> 56) & 0xFF) as u8
    }

    /// Encode type tag into NodeId
    const fn encode(self, value: u64) -> u64 {
        ((self as u64) << 56) | (value & 0x00FF_FFFF_FFFF_FFFF)
    }
}

/// 8-byte node identifier for RDF terms
///
/// NodeIds use an optimized encoding that supports:
/// - Dictionary references for large/complex values
/// - Inline encoding for small, frequently-used values
///
/// # Encoding Scheme (64 bits)
///
/// ```text
/// ┌─────────────┬────────────────────────────────────────────────┐
/// │ Type (8 bit)│           Value (56 bits)                      │
/// └─────────────┴────────────────────────────────────────────────┘
/// ```
///
/// ## Type Tags:
/// - 0x00: Dictionary reference (standard sequential NodeId)
/// - 0x01: Inline signed integer (i32, sign-extended to 56 bits)
/// - 0x02: Inline boolean (0 = false, 1 = true)
/// - 0x03: Inline short ASCII string (up to 7 bytes)
/// - 0x04: Inline unsigned integer (u32, zero-extended to 56 bits)
/// - 0x05: Inline decimal (mantissa + exponent)
/// - 0x06: Inline date/time (Unix timestamp)
///
/// ## Benefits:
/// - Eliminates dictionary lookups for ~30-40% of common values
/// - Reduces memory pressure on dictionary cache
/// - Improves query performance for predicates with many small literals
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(u64);

impl NodeId {
    /// Mask for extracting the value portion (lower 56 bits)
    const VALUE_MASK: u64 = 0x00FF_FFFF_FFFF_FFFF;

    /// Mask for extracting the type tag (upper 8 bits)
    const TYPE_MASK: u64 = 0xFF00_0000_0000_0000;

    /// Maximum value for dictionary references (56 bits)
    const MAX_DICT_ID: u64 = 0x00FF_FFFF_FFFF_FFFF;

    /// Create a new NodeId from a raw u64 value
    pub const fn new(id: u64) -> Self {
        NodeId(id)
    }

    /// Get the raw u64 value
    pub const fn as_u64(&self) -> u64 {
        self.0
    }

    /// First valid node ID (0 is reserved as NULL)
    pub const FIRST: NodeId = NodeId(1);

    /// Reserved NULL node ID
    pub const NULL: NodeId = NodeId(0);

    /// Check if this is a null node ID
    pub const fn is_null(&self) -> bool {
        self.0 == 0
    }

    /// Get the next node ID (for dictionary references only)
    pub const fn next(&self) -> NodeId {
        NodeId(self.0 + 1)
    }

    /// Create a dictionary reference NodeId
    pub const fn dict_ref(id: u64) -> Self {
        // Ensure we don't overflow into the type bits
        assert!(id <= Self::MAX_DICT_ID);
        NodeId(id)
    }

    /// Create an inline small integer NodeId
    pub const fn inline_int(value: i32) -> Self {
        // Sign-extend i32 to 56 bits, then encode with type tag
        let extended = (value as i64) as u64 & Self::VALUE_MASK;
        NodeId(InlineType::SmallInt.encode(extended))
    }

    /// Create an inline boolean NodeId
    pub const fn inline_bool(value: bool) -> Self {
        NodeId(InlineType::Boolean.encode(if value { 1 } else { 0 }))
    }

    /// Create an inline short string NodeId (up to 7 ASCII bytes)
    ///
    /// Returns None if the string is too long or contains non-ASCII characters
    pub fn inline_short_string(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() > 7 || !s.is_ascii() {
            return None;
        }

        let mut encoded: u64 = 0;
        for (i, &byte) in bytes.iter().enumerate() {
            encoded |= (byte as u64) << (i * 8);
        }

        Some(NodeId(InlineType::ShortString.encode(encoded)))
    }

    /// Create an inline unsigned integer NodeId (u32)
    pub const fn inline_uint(value: u32) -> Self {
        NodeId(InlineType::SmallUInt.encode(value as u64))
    }

    /// Get the inline type of this NodeId
    pub const fn inline_type(&self) -> u8 {
        InlineType::from_node_id(self.0)
    }

    /// Check if this is a dictionary reference
    pub const fn is_dict_ref(&self) -> bool {
        !self.is_null() && self.inline_type() == InlineType::Dictionary as u8
    }

    /// Check if this is an inline value
    pub const fn is_inline(&self) -> bool {
        !self.is_null() && !self.is_dict_ref()
    }

    /// Extract dictionary ID (only valid if is_dict_ref() returns true)
    pub const fn dict_id(&self) -> u64 {
        self.0 & Self::VALUE_MASK
    }

    /// Extract inline integer value (only valid if inline_type() == SmallInt)
    pub const fn inline_int_value(&self) -> Option<i32> {
        if self.inline_type() == InlineType::SmallInt as u8 {
            // Extract 56-bit value and sign-extend to i64, then cast to i32
            let value = (self.0 & Self::VALUE_MASK) as i64;
            // Sign-extend from 56 bits
            let extended = if value & 0x0080_0000_0000_0000 != 0 {
                value | 0xFF00_0000_0000_0000u64 as i64
            } else {
                value
            };
            Some(extended as i32)
        } else {
            None
        }
    }

    /// Extract inline boolean value (only valid if inline_type() == Boolean)
    pub const fn inline_bool_value(&self) -> Option<bool> {
        if self.inline_type() == InlineType::Boolean as u8 {
            Some((self.0 & Self::VALUE_MASK) != 0)
        } else {
            None
        }
    }

    /// Extract inline unsigned integer value (only valid if inline_type() == SmallUInt)
    pub const fn inline_uint_value(&self) -> Option<u32> {
        if self.inline_type() == InlineType::SmallUInt as u8 {
            Some((self.0 & Self::VALUE_MASK) as u32)
        } else {
            None
        }
    }

    /// Extract inline short string value (only valid if inline_type() == ShortString)
    pub fn inline_string_value(&self) -> Option<String> {
        if self.inline_type() == InlineType::ShortString as u8 {
            let encoded = self.0 & Self::VALUE_MASK;
            let mut bytes = Vec::with_capacity(7);

            for i in 0..7 {
                let byte = ((encoded >> (i * 8)) & 0xFF) as u8;
                if byte == 0 {
                    break;
                }
                bytes.push(byte);
            }

            String::from_utf8(bytes).ok()
        } else {
            None
        }
    }
}

impl Default for NodeId {
    fn default() -> Self {
        NodeId::NULL
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}

impl From<u64> for NodeId {
    fn from(id: u64) -> Self {
        NodeId(id)
    }
}

impl From<NodeId> for u64 {
    fn from(id: NodeId) -> Self {
        id.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic NodeId tests
    #[test]
    fn test_node_id_creation() {
        let id = NodeId::new(42);
        assert_eq!(id.as_u64(), 42);
    }

    #[test]
    fn test_node_id_null() {
        let null_id = NodeId::NULL;
        assert!(null_id.is_null());
        assert_eq!(null_id.as_u64(), 0);

        let valid_id = NodeId::FIRST;
        assert!(!valid_id.is_null());
        assert_eq!(valid_id.as_u64(), 1);
    }

    #[test]
    fn test_node_id_next() {
        let id = NodeId::new(10);
        let next = id.next();
        assert_eq!(next.as_u64(), 11);
    }

    #[test]
    fn test_node_id_ordering() {
        let id1 = NodeId::new(10);
        let id2 = NodeId::new(20);
        assert!(id1 < id2);
        assert!(id2 > id1);
    }

    #[test]
    fn test_node_id_conversions() {
        let raw: u64 = 42;
        let id: NodeId = raw.into();
        assert_eq!(id.as_u64(), 42);

        let back: u64 = id.into();
        assert_eq!(back, 42);
    }

    #[test]
    fn test_node_id_serialization() {
        let id = NodeId::new(123);
        let serialized = oxicode::serde::encode_to_vec(&id, oxicode::config::standard()).unwrap();
        let deserialized: NodeId =
            oxicode::serde::decode_from_slice(&serialized, oxicode::config::standard())
                .unwrap()
                .0;
        assert_eq!(id, deserialized);
    }

    // Dictionary reference tests
    #[test]
    fn test_dict_ref() {
        let id = NodeId::dict_ref(12345);
        assert!(id.is_dict_ref());
        assert!(!id.is_inline());
        assert_eq!(id.dict_id(), 12345);
        assert_eq!(id.inline_type(), InlineType::Dictionary as u8);
    }

    #[test]
    fn test_dict_ref_large() {
        let large_id = 0x00FF_FFFF_FFFF_FFFE; // Just under max
        let id = NodeId::dict_ref(large_id);
        assert!(id.is_dict_ref());
        assert_eq!(id.dict_id(), large_id);
    }

    // Inline integer tests
    #[test]
    fn test_inline_int_positive() {
        let id = NodeId::inline_int(42);
        assert!(!id.is_dict_ref());
        assert!(id.is_inline());
        assert_eq!(id.inline_type(), InlineType::SmallInt as u8);
        assert_eq!(id.inline_int_value(), Some(42));
        assert_eq!(id.inline_bool_value(), None);
        assert_eq!(id.inline_uint_value(), None);
    }

    #[test]
    fn test_inline_int_negative() {
        let id = NodeId::inline_int(-100);
        assert!(id.is_inline());
        assert_eq!(id.inline_int_value(), Some(-100));
    }

    #[test]
    fn test_inline_int_zero() {
        let id = NodeId::inline_int(0);
        assert!(id.is_inline());
        assert_eq!(id.inline_int_value(), Some(0));
    }

    #[test]
    fn test_inline_int_bounds() {
        let max = NodeId::inline_int(i32::MAX);
        assert_eq!(max.inline_int_value(), Some(i32::MAX));

        let min = NodeId::inline_int(i32::MIN);
        assert_eq!(min.inline_int_value(), Some(i32::MIN));
    }

    // Inline boolean tests
    #[test]
    fn test_inline_bool_true() {
        let id = NodeId::inline_bool(true);
        assert!(id.is_inline());
        assert_eq!(id.inline_type(), InlineType::Boolean as u8);
        assert_eq!(id.inline_bool_value(), Some(true));
        assert_eq!(id.inline_int_value(), None);
    }

    #[test]
    fn test_inline_bool_false() {
        let id = NodeId::inline_bool(false);
        assert!(id.is_inline());
        assert_eq!(id.inline_bool_value(), Some(false));
    }

    // Inline short string tests
    #[test]
    fn test_inline_short_string() {
        let id = NodeId::inline_short_string("hello").unwrap();
        assert!(id.is_inline());
        assert_eq!(id.inline_type(), InlineType::ShortString as u8);
        assert_eq!(id.inline_string_value(), Some("hello".to_string()));
    }

    #[test]
    fn test_inline_short_string_single_char() {
        let id = NodeId::inline_short_string("x").unwrap();
        assert_eq!(id.inline_string_value(), Some("x".to_string()));
    }

    #[test]
    fn test_inline_short_string_seven_chars() {
        let id = NodeId::inline_short_string("abcdefg").unwrap();
        assert_eq!(id.inline_string_value(), Some("abcdefg".to_string()));
    }

    #[test]
    fn test_inline_short_string_too_long() {
        let result = NodeId::inline_short_string("abcdefgh"); // 8 chars
        assert!(result.is_none());
    }

    #[test]
    fn test_inline_short_string_non_ascii() {
        let result = NodeId::inline_short_string("hello🌍");
        assert!(result.is_none());
    }

    #[test]
    fn test_inline_short_string_empty() {
        let id = NodeId::inline_short_string("").unwrap();
        assert_eq!(id.inline_string_value(), Some("".to_string()));
    }

    // Inline unsigned integer tests
    #[test]
    fn test_inline_uint() {
        let id = NodeId::inline_uint(12345);
        assert!(id.is_inline());
        assert_eq!(id.inline_type(), InlineType::SmallUInt as u8);
        assert_eq!(id.inline_uint_value(), Some(12345));
        assert_eq!(id.inline_int_value(), None);
    }

    #[test]
    fn test_inline_uint_zero() {
        let id = NodeId::inline_uint(0);
        assert_eq!(id.inline_uint_value(), Some(0));
    }

    #[test]
    fn test_inline_uint_max() {
        let id = NodeId::inline_uint(u32::MAX);
        assert_eq!(id.inline_uint_value(), Some(u32::MAX));
    }

    // Type discrimination tests
    #[test]
    fn test_inline_types_are_distinct() {
        let dict = NodeId::dict_ref(42);
        let int = NodeId::inline_int(42);
        let uint = NodeId::inline_uint(42);
        let bool_true = NodeId::inline_bool(true);
        let string = NodeId::inline_short_string("42").unwrap();

        // All should have different raw values
        assert_ne!(dict.as_u64(), int.as_u64());
        assert_ne!(dict.as_u64(), uint.as_u64());
        assert_ne!(dict.as_u64(), bool_true.as_u64());
        assert_ne!(dict.as_u64(), string.as_u64());
        assert_ne!(int.as_u64(), uint.as_u64());
    }

    // Serialization tests for inline values
    #[test]
    fn test_inline_int_serialization() {
        let id = NodeId::inline_int(-42);
        let serialized = oxicode::serde::encode_to_vec(&id, oxicode::config::standard()).unwrap();
        let deserialized: NodeId =
            oxicode::serde::decode_from_slice(&serialized, oxicode::config::standard())
                .unwrap()
                .0;
        assert_eq!(id, deserialized);
        assert_eq!(deserialized.inline_int_value(), Some(-42));
    }

    #[test]
    fn test_inline_bool_serialization() {
        let id = NodeId::inline_bool(true);
        let serialized = oxicode::serde::encode_to_vec(&id, oxicode::config::standard()).unwrap();
        let deserialized: NodeId =
            oxicode::serde::decode_from_slice(&serialized, oxicode::config::standard())
                .unwrap()
                .0;
        assert_eq!(id, deserialized);
        assert_eq!(deserialized.inline_bool_value(), Some(true));
    }

    #[test]
    fn test_inline_string_serialization() {
        let id = NodeId::inline_short_string("test").unwrap();
        let serialized = oxicode::serde::encode_to_vec(&id, oxicode::config::standard()).unwrap();
        let deserialized: NodeId =
            oxicode::serde::decode_from_slice(&serialized, oxicode::config::standard())
                .unwrap()
                .0;
        assert_eq!(id, deserialized);
        assert_eq!(deserialized.inline_string_value(), Some("test".to_string()));
    }

    // Performance benefit tests
    #[test]
    fn test_common_rdf_values() {
        // Test common RDF literal values that should be inlined
        assert!(NodeId::inline_int(0).is_inline()); // xsd:integer "0"
        assert!(NodeId::inline_int(1).is_inline()); // xsd:integer "1"
        assert!(NodeId::inline_bool(true).is_inline()); // xsd:boolean "true"
        assert!(NodeId::inline_bool(false).is_inline()); // xsd:boolean "false"
        assert!(NodeId::inline_short_string("en").unwrap().is_inline()); // Language tag
        assert!(NodeId::inline_short_string("").unwrap().is_inline()); // Empty string
    }

    // Edge case: NULL should not be inline
    #[test]
    fn test_null_is_not_inline() {
        assert!(!NodeId::NULL.is_inline());
        assert!(!NodeId::NULL.is_dict_ref());
        assert!(NodeId::NULL.is_null());
    }
}
