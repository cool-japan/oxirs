//! HTTP Protocol Optimization
//!
//! Provides HTTP/2 and HTTP/3 support with optimizations for SPARQL workloads.
//! Includes connection multiplexing, header compression, and server push capabilities.

use crate::error::{FusekiError, FusekiResult};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info};

/// HTTP protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpProtocolConfig {
    /// Enable HTTP/2
    pub http2_enabled: bool,
    /// Enable HTTP/3 (QUIC)
    pub http3_enabled: bool,
    /// HTTP/2 initial connection window size
    pub http2_initial_connection_window_size: u32,
    /// HTTP/2 initial stream window size
    pub http2_initial_stream_window_size: u32,
    /// HTTP/2 max concurrent streams
    pub http2_max_concurrent_streams: u32,
    /// HTTP/2 max frame size
    pub http2_max_frame_size: u32,
    /// HTTP/2 keep alive interval
    pub http2_keep_alive_interval: Duration,
    /// HTTP/2 keep alive timeout
    pub http2_keep_alive_timeout: Duration,
    /// Enable server push for SPARQL results
    pub enable_server_push: bool,
    /// Enable header compression (HPACK for HTTP/2, QPACK for HTTP/3)
    pub enable_header_compression: bool,
}

impl Default for HttpProtocolConfig {
    fn default() -> Self {
        Self {
            http2_enabled: true,
            http3_enabled: false,                              // Experimental
            http2_initial_connection_window_size: 1024 * 1024, // 1MB
            http2_initial_stream_window_size: 256 * 1024,      // 256KB
            http2_max_concurrent_streams: 100,
            http2_max_frame_size: 16384, // 16KB (default)
            http2_keep_alive_interval: Duration::from_secs(60),
            http2_keep_alive_timeout: Duration::from_secs(20),
            enable_server_push: true,
            enable_header_compression: true,
        }
    }
}

/// HTTP/2 connection manager
pub struct Http2Manager {
    config: HttpProtocolConfig,
}

impl Http2Manager {
    /// Create a new HTTP/2 manager
    pub fn new(config: HttpProtocolConfig) -> Self {
        Self { config }
    }

    /// Build HTTP/2 configuration for Hyper
    pub fn build_http2_config(&self) -> Http2Config {
        info!("Configuring HTTP/2 protocol");

        Http2Config {
            initial_connection_window_size: self.config.http2_initial_connection_window_size,
            initial_stream_window_size: self.config.http2_initial_stream_window_size,
            max_concurrent_streams: self.config.http2_max_concurrent_streams,
            max_frame_size: self.config.http2_max_frame_size,
            keep_alive_interval: self.config.http2_keep_alive_interval,
            keep_alive_timeout: self.config.http2_keep_alive_timeout,
            enable_connect_protocol: false,
            max_send_buffer_size: 1024 * 1024, // 1MB
        }
    }

    /// Check if server push is enabled
    pub fn is_server_push_enabled(&self) -> bool {
        self.config.enable_server_push
    }

    /// Optimize for SPARQL query patterns
    pub fn optimize_for_sparql(&mut self) {
        debug!("Optimizing HTTP/2 for SPARQL workloads");

        // Increase stream window for large result sets
        self.config.http2_initial_stream_window_size = 512 * 1024; // 512KB

        // Allow more concurrent streams for federated queries
        self.config.http2_max_concurrent_streams = 200;

        // Larger frame size for bulk data transfer
        self.config.http2_max_frame_size = 32768; // 32KB
    }
}

/// HTTP/2 configuration
#[derive(Debug, Clone)]
pub struct Http2Config {
    pub initial_connection_window_size: u32,
    pub initial_stream_window_size: u32,
    pub max_concurrent_streams: u32,
    pub max_frame_size: u32,
    pub keep_alive_interval: Duration,
    pub keep_alive_timeout: Duration,
    pub enable_connect_protocol: bool,
    pub max_send_buffer_size: usize,
}

/// HTTP/3 (QUIC) manager
pub struct Http3Manager {
    config: HttpProtocolConfig,
}

impl Http3Manager {
    /// Create a new HTTP/3 manager
    pub fn new(config: HttpProtocolConfig) -> Self {
        Self { config }
    }

    /// Check if HTTP/3 is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.http3_enabled
    }

    /// Build HTTP/3 configuration
    pub fn build_http3_config(&self) -> FusekiResult<Http3Config> {
        if !self.config.http3_enabled {
            return Err(FusekiError::configuration(
                "HTTP/3 is not enabled".to_string(),
            ));
        }

        info!("Configuring HTTP/3 (QUIC) protocol");

        Ok(Http3Config {
            max_idle_timeout: Duration::from_secs(300),
            max_udp_payload_size: 1200,
            initial_max_data: 10 * 1024 * 1024, // 10MB
            initial_max_stream_data_bidi_local: 1024 * 1024,
            initial_max_stream_data_bidi_remote: 1024 * 1024,
            initial_max_stream_data_uni: 1024 * 1024,
            initial_max_streams_bidi: 100,
            initial_max_streams_uni: 100,
            disable_active_migration: false,
        })
    }
}

/// HTTP/3 configuration
#[derive(Debug, Clone)]
pub struct Http3Config {
    pub max_idle_timeout: Duration,
    pub max_udp_payload_size: u16,
    pub initial_max_data: u64,
    pub initial_max_stream_data_bidi_local: u64,
    pub initial_max_stream_data_bidi_remote: u64,
    pub initial_max_stream_data_uni: u64,
    pub initial_max_streams_bidi: u64,
    pub initial_max_streams_uni: u64,
    pub disable_active_migration: bool,
}

/// Server push helper for SPARQL results
pub struct ServerPushHelper;

impl ServerPushHelper {
    /// Determine if result should be server-pushed
    pub fn should_push_result(result_size: usize, query_complexity: f64) -> bool {
        // Push small to medium results that are complex queries
        result_size < 100_000 && query_complexity > 0.5
    }

    /// Generate push promise for SPARQL result
    pub fn create_push_promise(result_uri: &str) -> PushPromise {
        PushPromise {
            uri: result_uri.to_string(),
            headers: vec![
                (
                    "content-type".to_string(),
                    "application/sparql-results+json".to_string(),
                ),
                ("cache-control".to_string(), "no-cache".to_string()),
            ],
        }
    }
}

/// Server push promise
#[derive(Debug, Clone)]
pub struct PushPromise {
    pub uri: String,
    pub headers: Vec<(String, String)>,
}

/// Protocol negotiation utilities
pub struct ProtocolNegotiation;

impl ProtocolNegotiation {
    /// Negotiate protocol version with client
    pub fn negotiate(client_protocols: &[&str]) -> Protocol {
        for protocol in client_protocols {
            match *protocol {
                "h3" | "h3-29" => return Protocol::Http3,
                "h2" => return Protocol::Http2,
                "http/1.1" => return Protocol::Http11,
                _ => continue,
            }
        }
        Protocol::Http11 // Default fallback
    }

    /// Get ALPN protocols in order of preference
    pub fn alpn_protocols() -> Vec<Vec<u8>> {
        vec![
            b"h3".to_vec(),       // HTTP/3
            b"h2".to_vec(),       // HTTP/2
            b"http/1.1".to_vec(), // HTTP/1.1
        ]
    }
}

/// Supported HTTP protocol versions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Http11,
    Http2,
    Http3,
}

impl Protocol {
    /// Get protocol name
    pub fn name(&self) -> &'static str {
        match self {
            Protocol::Http11 => "HTTP/1.1",
            Protocol::Http2 => "HTTP/2",
            Protocol::Http3 => "HTTP/3",
        }
    }

    /// Check if protocol supports multiplexing
    pub fn supports_multiplexing(&self) -> bool {
        matches!(self, Protocol::Http2 | Protocol::Http3)
    }

    /// Check if protocol supports server push
    pub fn supports_server_push(&self) -> bool {
        matches!(self, Protocol::Http2 | Protocol::Http3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http2_config_default() {
        let config = HttpProtocolConfig::default();
        assert!(config.http2_enabled);
        assert!(!config.http3_enabled);
        assert_eq!(config.http2_max_concurrent_streams, 100);
    }

    #[test]
    fn test_http2_manager() {
        let config = HttpProtocolConfig::default();
        let manager = Http2Manager::new(config);
        let http2_config = manager.build_http2_config();

        assert_eq!(http2_config.max_concurrent_streams, 100);
        assert_eq!(http2_config.max_frame_size, 16384);
    }

    #[test]
    fn test_sparql_optimization() {
        let config = HttpProtocolConfig::default();
        let mut manager = Http2Manager::new(config);

        manager.optimize_for_sparql();
        let http2_config = manager.build_http2_config();

        assert_eq!(http2_config.initial_stream_window_size, 512 * 1024);
        assert_eq!(http2_config.max_concurrent_streams, 200);
    }

    #[test]
    fn test_protocol_negotiation() {
        let protocols = ["h2", "http/1.1"];
        let negotiated = ProtocolNegotiation::negotiate(&protocols);
        assert_eq!(negotiated, Protocol::Http2);

        let protocols = ["http/1.1"];
        let negotiated = ProtocolNegotiation::negotiate(&protocols);
        assert_eq!(negotiated, Protocol::Http11);
    }

    #[test]
    fn test_server_push_decision() {
        assert!(ServerPushHelper::should_push_result(50_000, 0.8));
        assert!(!ServerPushHelper::should_push_result(150_000, 0.8));
        assert!(!ServerPushHelper::should_push_result(50_000, 0.3));
    }

    #[test]
    fn test_protocol_capabilities() {
        assert!(Protocol::Http2.supports_multiplexing());
        assert!(Protocol::Http3.supports_multiplexing());
        assert!(!Protocol::Http11.supports_multiplexing());

        assert!(Protocol::Http2.supports_server_push());
        assert!(!Protocol::Http11.supports_server_push());
    }
}
