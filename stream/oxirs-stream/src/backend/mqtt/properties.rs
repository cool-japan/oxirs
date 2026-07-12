//! MQTT 5.0 Property Codec
//!
//! Encodes and decodes MQTT 5.0 message properties per the MQTT 5.0 specification
//! (section 2.2.2). Properties are encoded as a Variable Byte Integer (VarInt)
//! length prefix followed by a sequence of property identifier + value pairs.

use super::types::MqttMessageProperties;
use crate::error::{StreamError, StreamResult};
use std::collections::HashMap;

/// MQTT 5.0 property identifiers per the MQTT 5.0 specification (section 2.2.2)
pub mod property_id {
    pub const PAYLOAD_FORMAT_INDICATOR: u8 = 0x01;
    pub const MESSAGE_EXPIRY_INTERVAL: u8 = 0x02;
    pub const CONTENT_TYPE: u8 = 0x03;
    pub const RESPONSE_TOPIC: u8 = 0x08;
    pub const CORRELATION_DATA: u8 = 0x09;
    pub const SUBSCRIPTION_IDENTIFIER: u8 = 0x0B;
    pub const TOPIC_ALIAS: u8 = 0x23;
    pub const USER_PROPERTY: u8 = 0x26;
}

/// Encode a Variable Byte Integer into `buf`.
///
/// The MQTT 5.0 VarInt encoding uses 7 bits per byte; the MSB is a continuation flag.
/// Values up to 268,435,455 (0x0FFFFFFF) are supported (4-byte maximum).
fn encode_varint(mut value: u64, buf: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value > 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Decode a Variable Byte Integer from `data`.
///
/// Returns `(value, bytes_consumed)`. Fails if the data is truncated or
/// the VarInt exceeds the 4-byte maximum defined by the MQTT 5.0 spec.
fn decode_varint(data: &[u8]) -> StreamResult<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    let mut idx = 0;

    loop {
        if idx >= data.len() {
            return Err(StreamError::Deserialization(
                "MQTT VarInt: unexpected end of data".to_string(),
            ));
        }
        if shift > 21 {
            return Err(StreamError::Deserialization(
                "MQTT VarInt: value too large (exceeds 4-byte maximum)".to_string(),
            ));
        }

        let byte = data[idx];
        idx += 1;

        value |= ((byte & 0x7F) as u64) << shift;
        shift += 7;

        if byte & 0x80 == 0 {
            break;
        }
    }

    Ok((value, idx))
}

/// Encode a UTF-8 string as MQTT binary data: 2-byte big-endian length followed by the UTF-8 bytes.
fn encode_utf8_string(s: &str, buf: &mut Vec<u8>) {
    let bytes = s.as_bytes();
    let len = bytes.len() as u16;
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(bytes);
}

/// Decode a UTF-8 string from MQTT binary data.
///
/// Reads a 2-byte big-endian length, then that many UTF-8 bytes.
/// Returns `(string, total_bytes_consumed)`.
fn decode_utf8_string(data: &[u8]) -> StreamResult<(String, usize)> {
    if data.len() < 2 {
        return Err(StreamError::Deserialization(
            "MQTT UTF-8 string: insufficient data for length field".to_string(),
        ));
    }
    let len = u16::from_be_bytes([data[0], data[1]]) as usize;
    let total = 2 + len;
    if data.len() < total {
        return Err(StreamError::Deserialization(format!(
            "MQTT UTF-8 string: expected {} bytes, got {}",
            total,
            data.len()
        )));
    }
    let s = std::str::from_utf8(&data[2..total]).map_err(|e| {
        StreamError::Deserialization(format!("MQTT UTF-8 string: invalid UTF-8: {}", e))
    })?;
    Ok((s.to_string(), total))
}

/// Encode binary data as MQTT format: 2-byte big-endian length followed by raw bytes.
fn encode_binary_data(bytes: &[u8], buf: &mut Vec<u8>) {
    let len = bytes.len() as u16;
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(bytes);
}

/// Decode binary data from MQTT format.
///
/// Reads a 2-byte big-endian length, then that many bytes.
/// Returns `(data, total_bytes_consumed)`.
fn decode_binary_data(data: &[u8]) -> StreamResult<(Vec<u8>, usize)> {
    if data.len() < 2 {
        return Err(StreamError::Deserialization(
            "MQTT binary data: insufficient data for length field".to_string(),
        ));
    }
    let len = u16::from_be_bytes([data[0], data[1]]) as usize;
    let total = 2 + len;
    if data.len() < total {
        return Err(StreamError::Deserialization(format!(
            "MQTT binary data: expected {} bytes, got {}",
            total,
            data.len()
        )));
    }
    Ok((data[2..total].to_vec(), total))
}

/// Encode [`MqttMessageProperties`] into MQTT 5.0 binary format.
///
/// Returns bytes starting with the VarInt length prefix, followed by all
/// property TLV (type-length-value) entries as defined in MQTT 5.0 section 2.2.2.
pub fn encode_mqtt5_properties(props: &MqttMessageProperties) -> Vec<u8> {
    use property_id::*;

    let mut body = Vec::new();

    if let Some(pfi) = props.payload_format_indicator {
        body.push(PAYLOAD_FORMAT_INDICATOR);
        body.push(pfi);
    }

    if let Some(mei) = props.message_expiry_interval {
        body.push(MESSAGE_EXPIRY_INTERVAL);
        body.extend_from_slice(&mei.to_be_bytes());
    }

    if let Some(ref ct) = props.content_type {
        body.push(CONTENT_TYPE);
        encode_utf8_string(ct, &mut body);
    }

    if let Some(ref rt) = props.response_topic {
        body.push(RESPONSE_TOPIC);
        encode_utf8_string(rt, &mut body);
    }

    if let Some(ref cd) = props.correlation_data {
        body.push(CORRELATION_DATA);
        encode_binary_data(cd, &mut body);
    }

    if let Some(si) = props.subscription_identifier {
        body.push(SUBSCRIPTION_IDENTIFIER);
        encode_varint(si as u64, &mut body);
    }

    if let Some(ta) = props.topic_alias {
        body.push(TOPIC_ALIAS);
        body.extend_from_slice(&ta.to_be_bytes());
    }

    // User Properties: each pair is encoded separately with its own property id
    for (key, val) in &props.user_properties {
        body.push(USER_PROPERTY);
        encode_utf8_string(key, &mut body);
        encode_utf8_string(val, &mut body);
    }

    let mut result = Vec::new();
    encode_varint(body.len() as u64, &mut result);
    result.extend_from_slice(&body);
    result
}

/// Decode MQTT 5.0 properties from a binary slice.
///
/// The slice must start with a Variable Byte Integer that gives the total
/// length of the property section (as defined in MQTT 5.0 section 2.2.2).
///
/// Returns `(MqttMessageProperties, bytes_consumed)` where `bytes_consumed`
/// is the number of bytes read from the start of `data` (including the VarInt prefix).
///
/// # Errors
///
/// Returns [`StreamError::Deserialization`] if the data is malformed or truncated.
pub fn decode_mqtt5_properties(data: &[u8]) -> StreamResult<(MqttMessageProperties, usize)> {
    use property_id::*;

    // Decode the VarInt length prefix
    let (prop_len, varint_size) = decode_varint(data)?;
    let prop_len = prop_len as usize;

    let total_consumed = varint_size + prop_len;
    if data.len() < total_consumed {
        return Err(StreamError::Deserialization(format!(
            "MQTT properties: declared length {} but only {} bytes available",
            prop_len,
            data.len() - varint_size
        )));
    }

    let prop_data = &data[varint_size..total_consumed];

    let mut props = MqttMessageProperties {
        payload_format_indicator: None,
        message_expiry_interval: None,
        topic_alias: None,
        response_topic: None,
        correlation_data: None,
        user_properties: HashMap::new(),
        subscription_identifier: None,
        content_type: None,
    };

    let mut pos = 0;
    while pos < prop_len {
        if pos >= prop_data.len() {
            return Err(StreamError::Deserialization(
                "MQTT properties: unexpected end while reading property id".to_string(),
            ));
        }
        let prop_id = prop_data[pos];
        pos += 1;

        match prop_id {
            PAYLOAD_FORMAT_INDICATOR => {
                if pos >= prop_data.len() {
                    return Err(StreamError::Deserialization(
                        "MQTT properties: truncated Payload Format Indicator".to_string(),
                    ));
                }
                props.payload_format_indicator = Some(prop_data[pos]);
                pos += 1;
            }
            MESSAGE_EXPIRY_INTERVAL => {
                if pos + 4 > prop_data.len() {
                    return Err(StreamError::Deserialization(
                        "MQTT properties: truncated Message Expiry Interval".to_string(),
                    ));
                }
                let val = u32::from_be_bytes([
                    prop_data[pos],
                    prop_data[pos + 1],
                    prop_data[pos + 2],
                    prop_data[pos + 3],
                ]);
                props.message_expiry_interval = Some(val);
                pos += 4;
            }
            CONTENT_TYPE => {
                let (s, consumed) = decode_utf8_string(&prop_data[pos..])?;
                props.content_type = Some(s);
                pos += consumed;
            }
            RESPONSE_TOPIC => {
                let (s, consumed) = decode_utf8_string(&prop_data[pos..])?;
                props.response_topic = Some(s);
                pos += consumed;
            }
            CORRELATION_DATA => {
                let (bytes, consumed) = decode_binary_data(&prop_data[pos..])?;
                props.correlation_data = Some(bytes);
                pos += consumed;
            }
            SUBSCRIPTION_IDENTIFIER => {
                let (val, consumed) = decode_varint(&prop_data[pos..])?;
                props.subscription_identifier = Some(val as u32);
                pos += consumed;
            }
            TOPIC_ALIAS => {
                if pos + 2 > prop_data.len() {
                    return Err(StreamError::Deserialization(
                        "MQTT properties: truncated Topic Alias".to_string(),
                    ));
                }
                let val = u16::from_be_bytes([prop_data[pos], prop_data[pos + 1]]);
                props.topic_alias = Some(val);
                pos += 2;
            }
            USER_PROPERTY => {
                let (key, k_consumed) = decode_utf8_string(&prop_data[pos..])?;
                pos += k_consumed;
                let (val, v_consumed) = decode_utf8_string(&prop_data[pos..])?;
                pos += v_consumed;
                props.user_properties.insert(key, val);
            }
            unknown => {
                return Err(StreamError::Deserialization(format!(
                    "MQTT properties: unknown property identifier 0x{:02X}",
                    unknown
                )));
            }
        }
    }

    Ok((props, total_consumed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_full_props() -> MqttMessageProperties {
        let mut user_properties = HashMap::new();
        user_properties.insert("x-trace-id".to_string(), "abc123".to_string());
        user_properties.insert("x-source".to_string(), "sensor-node-7".to_string());

        MqttMessageProperties {
            payload_format_indicator: Some(1),
            message_expiry_interval: Some(3600),
            topic_alias: Some(42),
            response_topic: Some("response/sensor/reply".to_string()),
            correlation_data: Some(vec![0xDE, 0xAD, 0xBE, 0xEF]),
            user_properties,
            subscription_identifier: Some(5),
            content_type: Some("application/json".to_string()),
        }
    }

    #[test]
    fn test_encode_decode_all_fields() {
        let original = make_full_props();
        let encoded = encode_mqtt5_properties(&original);
        assert!(!encoded.is_empty());

        let (decoded, consumed) = decode_mqtt5_properties(&encoded).expect("decode should succeed");
        assert_eq!(consumed, encoded.len(), "should consume all bytes");

        assert_eq!(
            decoded.payload_format_indicator,
            original.payload_format_indicator
        );
        assert_eq!(
            decoded.message_expiry_interval,
            original.message_expiry_interval
        );
        assert_eq!(decoded.topic_alias, original.topic_alias);
        assert_eq!(decoded.response_topic, original.response_topic);
        assert_eq!(decoded.correlation_data, original.correlation_data);
        assert_eq!(
            decoded.subscription_identifier,
            original.subscription_identifier
        );
        assert_eq!(decoded.content_type, original.content_type);
        assert_eq!(
            decoded.user_properties.len(),
            original.user_properties.len()
        );
        for (k, v) in &original.user_properties {
            assert_eq!(decoded.user_properties.get(k), Some(v));
        }
    }

    #[test]
    fn test_encode_decode_empty() {
        let original = MqttMessageProperties {
            payload_format_indicator: None,
            message_expiry_interval: None,
            topic_alias: None,
            response_topic: None,
            correlation_data: None,
            user_properties: HashMap::new(),
            subscription_identifier: None,
            content_type: None,
        };

        let encoded = encode_mqtt5_properties(&original);
        // Empty properties: just the VarInt 0x00
        assert_eq!(encoded, vec![0x00]);

        let (decoded, consumed) = decode_mqtt5_properties(&encoded).expect("decode should succeed");
        assert_eq!(consumed, 1);
        assert!(decoded.payload_format_indicator.is_none());
        assert!(decoded.message_expiry_interval.is_none());
        assert!(decoded.topic_alias.is_none());
        assert!(decoded.response_topic.is_none());
        assert!(decoded.correlation_data.is_none());
        assert!(decoded.subscription_identifier.is_none());
        assert!(decoded.content_type.is_none());
        assert!(decoded.user_properties.is_empty());
    }

    #[test]
    fn test_encode_decode_user_properties_multiple() {
        let mut user_properties = HashMap::new();
        user_properties.insert("alpha".to_string(), "one".to_string());
        user_properties.insert("beta".to_string(), "two".to_string());
        user_properties.insert("gamma".to_string(), "three".to_string());

        let original = MqttMessageProperties {
            payload_format_indicator: None,
            message_expiry_interval: None,
            topic_alias: None,
            response_topic: None,
            correlation_data: None,
            user_properties,
            subscription_identifier: None,
            content_type: None,
        };

        let encoded = encode_mqtt5_properties(&original);
        let (decoded, _) = decode_mqtt5_properties(&encoded).expect("decode should succeed");

        assert_eq!(decoded.user_properties.len(), 3);
        assert_eq!(
            decoded.user_properties.get("alpha"),
            Some(&"one".to_string())
        );
        assert_eq!(
            decoded.user_properties.get("beta"),
            Some(&"two".to_string())
        );
        assert_eq!(
            decoded.user_properties.get("gamma"),
            Some(&"three".to_string())
        );
    }

    #[test]
    fn test_encode_decode_content_type_response_topic() {
        let original = MqttMessageProperties {
            payload_format_indicator: None,
            message_expiry_interval: None,
            topic_alias: None,
            response_topic: Some("response/topic/v2".to_string()),
            correlation_data: None,
            user_properties: HashMap::new(),
            subscription_identifier: None,
            content_type: Some("application/cbor".to_string()),
        };

        let encoded = encode_mqtt5_properties(&original);
        let (decoded, consumed) = decode_mqtt5_properties(&encoded).expect("decode should succeed");

        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.response_topic, original.response_topic);
        assert_eq!(decoded.content_type, original.content_type);
        assert!(decoded.payload_format_indicator.is_none());
        assert!(decoded.topic_alias.is_none());
    }

    #[test]
    fn test_encode_decode_correlation_data_binary() {
        // Correlation data with non-ASCII bytes including 0x00
        let binary_data = vec![0x00, 0xFF, 0x01, 0x80, 0x7F, 0x00, 0xAB];
        let original = MqttMessageProperties {
            payload_format_indicator: None,
            message_expiry_interval: None,
            topic_alias: None,
            response_topic: None,
            correlation_data: Some(binary_data.clone()),
            user_properties: HashMap::new(),
            subscription_identifier: None,
            content_type: None,
        };

        let encoded = encode_mqtt5_properties(&original);
        let (decoded, consumed) = decode_mqtt5_properties(&encoded).expect("decode should succeed");

        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.correlation_data, Some(binary_data));
    }

    #[test]
    fn test_varint_encoding_large_values() {
        // 128: needs 2 bytes (0x80, 0x01)
        let mut buf = Vec::new();
        encode_varint(128, &mut buf);
        assert_eq!(buf, vec![0x80, 0x01]);
        let (val, consumed) = decode_varint(&buf).expect("decode varint 128");
        assert_eq!(val, 128);
        assert_eq!(consumed, 2);

        // 16383: needs 2 bytes (0xFF, 0x7F)
        let mut buf = Vec::new();
        encode_varint(16383, &mut buf);
        assert_eq!(buf, vec![0xFF, 0x7F]);
        let (val, consumed) = decode_varint(&buf).expect("decode varint 16383");
        assert_eq!(val, 16383);
        assert_eq!(consumed, 2);

        // 2097151: needs 3 bytes (0xFF, 0xFF, 0x7F)
        let mut buf = Vec::new();
        encode_varint(2_097_151, &mut buf);
        assert_eq!(buf, vec![0xFF, 0xFF, 0x7F]);
        let (val, consumed) = decode_varint(&buf).expect("decode varint 2097151");
        assert_eq!(val, 2_097_151);
        assert_eq!(consumed, 3);
    }

    #[test]
    fn test_partial_data_error() {
        // Provide truncated data: declare 10 bytes of properties but only supply 3
        let mut data = Vec::new();
        encode_varint(10, &mut data); // claims 10 bytes
        data.extend_from_slice(&[0x01, 0x02, 0x03]); // only 3 bytes

        let result = decode_mqtt5_properties(&data);
        assert!(result.is_err(), "should fail on truncated data");
        if let Err(StreamError::Deserialization(msg)) = result {
            assert!(
                msg.contains("declared length")
                    || msg.contains("expected")
                    || msg.contains("insufficient"),
                "error message should describe the problem: {}",
                msg
            );
        } else {
            panic!("expected StreamError::Deserialization");
        }
    }
}
