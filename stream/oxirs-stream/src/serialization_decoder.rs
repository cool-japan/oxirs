//! Delta compression / decompression for event streams.
//!
//! This module provides the [`DeltaCompressor`] which calculates and applies
//! deltas between successive `StreamEvent` values using one of several
//! algorithms: XOR, prefix, dictionary, or LZ4-based diffs.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::serialization_types::{DeltaCompressedEvent, DeltaCompressionType, EventDelta};
use crate::StreamEvent;

/// Delta compression support for event streams
pub struct DeltaCompressor {
    /// Previous event states for delta calculation
    previous_states: Arc<RwLock<HashMap<String, StreamEvent>>>,
    /// Compression algorithm to use
    compression_type: DeltaCompressionType,
    /// Maximum states to keep in memory
    max_states: usize,
}

impl DeltaCompressor {
    /// Create a new delta compressor
    pub fn new(compression_type: DeltaCompressionType, max_states: usize) -> Self {
        Self {
            previous_states: Arc::new(RwLock::new(HashMap::new())),
            compression_type,
            max_states,
        }
    }

    /// Compress event using delta compression
    pub async fn compress_delta(
        &self,
        event: &StreamEvent,
        event_id: &str,
    ) -> Result<DeltaCompressedEvent> {
        let mut states = self.previous_states.write().await;

        // Clean up old states if we exceed the limit
        if states.len() >= self.max_states {
            let keys_to_remove: Vec<String> = states
                .keys()
                .take(states.len() - self.max_states + 1)
                .cloned()
                .collect();
            for key in keys_to_remove {
                states.remove(&key);
            }
        }

        let delta = if let Some(previous) = states.get(event_id) {
            self.calculate_delta(previous, event)?
        } else {
            // First event, store as full event
            EventDelta::Full(Box::new(event.clone()))
        };

        // Update state
        states.insert(event_id.to_string(), event.clone());

        Ok(DeltaCompressedEvent {
            event_id: event_id.to_string(),
            delta,
            compression_type: self.compression_type,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Calculate delta between two events
    fn calculate_delta(&self, previous: &StreamEvent, current: &StreamEvent) -> Result<EventDelta> {
        match self.compression_type {
            DeltaCompressionType::Xor => self.calculate_xor_delta(previous, current),
            DeltaCompressionType::Prefix => self.calculate_prefix_delta(previous, current),
            DeltaCompressionType::Dictionary => self.calculate_dictionary_delta(previous, current),
            DeltaCompressionType::Lz4Delta => self.calculate_lz4_delta(previous, current),
        }
    }

    /// XOR-based delta compression
    fn calculate_xor_delta(
        &self,
        previous: &StreamEvent,
        current: &StreamEvent,
    ) -> Result<EventDelta> {
        let prev_bytes = serde_json::to_vec(previous)?;
        let curr_bytes = serde_json::to_vec(current)?;

        if prev_bytes.len() != curr_bytes.len() {
            // If sizes differ, store as full event
            return Ok(EventDelta::Full(Box::new(current.clone())));
        }

        let xor_bytes: Vec<u8> = prev_bytes
            .iter()
            .zip(curr_bytes.iter())
            .map(|(a, b)| a ^ b)
            .collect();

        Ok(EventDelta::Xor(xor_bytes))
    }

    /// Prefix compression for string fields
    fn calculate_prefix_delta(
        &self,
        previous: &StreamEvent,
        current: &StreamEvent,
    ) -> Result<EventDelta> {
        let prev_json = serde_json::to_value(previous)?;
        let curr_json = serde_json::to_value(current)?;

        let diff = self.calculate_json_prefix_diff(&prev_json, &curr_json)?;
        Ok(EventDelta::Prefix(diff))
    }

    /// Dictionary-based compression
    fn calculate_dictionary_delta(
        &self,
        previous: &StreamEvent,
        current: &StreamEvent,
    ) -> Result<EventDelta> {
        let prev_strings = self.extract_strings_from_event(previous);
        let curr_strings = self.extract_strings_from_event(current);

        let mut dictionary = HashMap::new();
        let mut dict_id = 0u16;

        // Build dictionary from common strings
        for string in &prev_strings {
            if curr_strings.contains(string) && !dictionary.contains_key(string) {
                dictionary.insert(string.clone(), dict_id);
                dict_id += 1;
            }
        }

        // Replace strings with dictionary IDs
        let compressed_event = self.replace_strings_with_ids(current, &dictionary)?;

        Ok(EventDelta::Dictionary {
            dictionary,
            compressed_event,
        })
    }

    /// LZ4-based delta compression
    fn calculate_lz4_delta(
        &self,
        previous: &StreamEvent,
        current: &StreamEvent,
    ) -> Result<EventDelta> {
        let prev_bytes = serde_json::to_vec(previous)?;
        let curr_bytes = serde_json::to_vec(current)?;

        // Simple delta: store additions and removals
        let diff_bytes = self.calculate_byte_diff(&prev_bytes, &curr_bytes);
        let compressed = oxiarc_lz4::compress(&diff_bytes)
            .map_err(|e| anyhow!("LZ4 compression failed: {}", e))?;

        Ok(EventDelta::Lz4(compressed))
    }

    /// Calculate JSON prefix differences
    fn calculate_json_prefix_diff(
        &self,
        prev: &serde_json::Value,
        curr: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        match (prev, curr) {
            (serde_json::Value::Object(prev_obj), serde_json::Value::Object(curr_obj)) => {
                let mut diff = serde_json::Map::new();
                for (key, curr_val) in curr_obj {
                    if let Some(prev_val) = prev_obj.get(key) {
                        if prev_val != curr_val {
                            diff.insert(key.clone(), curr_val.clone());
                        }
                    } else {
                        diff.insert(key.clone(), curr_val.clone());
                    }
                }
                Ok(serde_json::Value::Object(diff))
            }
            _ => Ok(curr.clone()),
        }
    }

    /// Extract all strings from an event
    fn extract_strings_from_event(&self, event: &StreamEvent) -> Vec<String> {
        let mut strings = Vec::new();
        if let Ok(json) = serde_json::to_value(event) {
            Self::extract_strings_from_json(&json, &mut strings);
        }
        strings
    }

    /// Recursively extract strings from JSON value
    fn extract_strings_from_json(value: &serde_json::Value, strings: &mut Vec<String>) {
        match value {
            serde_json::Value::String(s) => strings.push(s.clone()),
            serde_json::Value::Array(arr) => {
                for item in arr {
                    Self::extract_strings_from_json(item, strings);
                }
            }
            serde_json::Value::Object(obj) => {
                for (_, val) in obj {
                    Self::extract_strings_from_json(val, strings);
                }
            }
            _ => {}
        }
    }

    /// Replace strings with dictionary IDs
    fn replace_strings_with_ids(
        &self,
        event: &StreamEvent,
        dictionary: &HashMap<String, u16>,
    ) -> Result<serde_json::Value> {
        let mut json = serde_json::to_value(event)?;
        Self::replace_strings_in_json(&mut json, dictionary);
        Ok(json)
    }

    /// Recursively replace strings in JSON
    fn replace_strings_in_json(value: &mut serde_json::Value, dictionary: &HashMap<String, u16>) {
        match value {
            serde_json::Value::String(s) => {
                if let Some(&id) = dictionary.get(s) {
                    *value = serde_json::Value::Number(serde_json::Number::from(id));
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    Self::replace_strings_in_json(item, dictionary);
                }
            }
            serde_json::Value::Object(obj) => {
                for val in obj.values_mut() {
                    Self::replace_strings_in_json(val, dictionary);
                }
            }
            _ => {}
        }
    }

    /// Calculate byte-level differences
    fn calculate_byte_diff(&self, prev: &[u8], curr: &[u8]) -> Vec<u8> {
        // Simple implementation - could be enhanced with more sophisticated diff algorithms
        let mut diff = Vec::new();

        // Store length difference
        diff.extend_from_slice(&(curr.len() as u32).to_le_bytes());
        diff.extend_from_slice(&(prev.len() as u32).to_le_bytes());

        // Store the current bytes (simplified)
        diff.extend_from_slice(curr);

        diff
    }

    /// Decompress delta-compressed event
    pub async fn decompress_delta(
        &self,
        compressed: &DeltaCompressedEvent,
        previous_event: Option<&StreamEvent>,
    ) -> Result<StreamEvent> {
        match &compressed.delta {
            EventDelta::Full(event) => Ok((**event).clone()),
            EventDelta::Xor(xor_bytes) => {
                if let Some(prev) = previous_event {
                    let prev_bytes = serde_json::to_vec(prev)?;
                    if prev_bytes.len() == xor_bytes.len() {
                        let restored_bytes: Vec<u8> = prev_bytes
                            .iter()
                            .zip(xor_bytes.iter())
                            .map(|(a, b)| a ^ b)
                            .collect();
                        let event = serde_json::from_slice(&restored_bytes)?;
                        Ok(event)
                    } else {
                        Err(anyhow!("XOR delta length mismatch"))
                    }
                } else {
                    Err(anyhow!("Previous event required for XOR decompression"))
                }
            }
            EventDelta::Prefix(diff) => {
                if let Some(prev) = previous_event {
                    let mut prev_json = serde_json::to_value(prev)?;
                    self.apply_json_diff(&mut prev_json, diff)?;
                    let event = serde_json::from_value(prev_json)?;
                    Ok(event)
                } else {
                    Err(anyhow!("Previous event required for prefix decompression"))
                }
            }
            EventDelta::Dictionary {
                dictionary,
                compressed_event,
            } => {
                let mut restored_json = compressed_event.clone();
                let reverse_dict: HashMap<u16, String> =
                    dictionary.iter().map(|(k, &v)| (v, k.clone())).collect();
                Self::restore_strings_from_ids(&mut restored_json, &reverse_dict);
                let event = serde_json::from_value(restored_json)?;
                Ok(event)
            }
            EventDelta::Lz4(compressed_bytes) => {
                let decompressed = oxiarc_lz4::decompress(compressed_bytes, 100 * 1024 * 1024)
                    .map_err(|e| anyhow!("LZ4 decompression failed: {}", e))?;
                // Restore from diff (simplified - would need more sophisticated restoration)
                let event = serde_json::from_slice(&decompressed)?;
                Ok(event)
            }
        }
    }

    /// Apply JSON diff to base JSON
    fn apply_json_diff(
        &self,
        base: &mut serde_json::Value,
        diff: &serde_json::Value,
    ) -> Result<()> {
        if let (Some(base_obj), Some(diff_obj)) = (base.as_object_mut(), diff.as_object()) {
            for (key, diff_val) in diff_obj {
                base_obj.insert(key.clone(), diff_val.clone());
            }
        } else {
            *base = diff.clone();
        }
        Ok(())
    }

    /// Restore strings from dictionary IDs
    fn restore_strings_from_ids(
        value: &mut serde_json::Value,
        reverse_dict: &HashMap<u16, String>,
    ) {
        match value {
            serde_json::Value::Number(n) => {
                if let Some(id) = n.as_u64() {
                    if let Some(string) = reverse_dict.get(&(id as u16)) {
                        *value = serde_json::Value::String(string.clone());
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    Self::restore_strings_from_ids(item, reverse_dict);
                }
            }
            serde_json::Value::Object(obj) => {
                for val in obj.values_mut() {
                    Self::restore_strings_from_ids(val, reverse_dict);
                }
            }
            _ => {}
        }
    }
}
