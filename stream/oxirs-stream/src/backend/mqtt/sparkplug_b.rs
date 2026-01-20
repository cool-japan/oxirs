//! Sparkplug B Protocol Buffer Types
//!
//! Eclipse Sparkplug B is an MQTT-based industrial IoT data format.
//! This module provides the generated protobuf types for Sparkplug B payloads.

// Include the generated protobuf code at module level
include!(concat!(env!("OUT_DIR"), "/org.eclipse.tahu.protobuf.rs"));
