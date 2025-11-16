//! DDoS Protection
//!
//! Provides comprehensive DDoS protection including:
//! - Rate limiting at multiple levels
//! - IP-based blocking and whitelisting
//! - Challenge-response mechanisms
//! - Traffic pattern analysis
//! - Automatic mitigation

use crate::error::{FusekiError, FusekiResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// DDoS protection manager
pub struct DDoSProtectionManager {
    config: DDoSProtectionConfig,
    /// IP tracking for rate limiting
    ip_tracker: Arc<DashMap<IpAddr, IpTrackingInfo>>,
    /// Blocked IPs
    blocked_ips: Arc<DashMap<IpAddr, BlockInfo>>,
    /// Whitelisted IPs
    whitelisted_ips: Arc<DashMap<IpAddr, ()>>,
    /// Attack detection
    attack_detector: Arc<AttackDetector>,
}

/// DDoS protection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DDoSProtectionConfig {
    /// Enable DDoS protection
    pub enabled: bool,
    /// Requests per second per IP (normal traffic)
    pub requests_per_second: u32,
    /// Burst allowance
    pub burst_size: u32,
    /// Block duration for violators (seconds)
    pub block_duration_secs: u64,
    /// Enable auto-blocking
    pub auto_block: bool,
    /// Enable challenge-response
    pub enable_challenge: bool,
    /// Connection limit per IP
    pub max_connections_per_ip: u32,
    /// Enable traffic analysis
    pub enable_traffic_analysis: bool,
}

impl Default for DDoSProtectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_second: 100,
            burst_size: 200,
            block_duration_secs: 300, // 5 minutes
            auto_block: true,
            enable_challenge: false,
            max_connections_per_ip: 10,
            enable_traffic_analysis: true,
        }
    }
}

/// IP tracking information
#[derive(Debug, Clone)]
struct IpTrackingInfo {
    request_count: u64,
    last_request: Instant,
    window_start: Instant,
    requests_in_window: u32,
    connection_count: u32,
    violation_count: u32,
}

impl Default for IpTrackingInfo {
    fn default() -> Self {
        Self {
            request_count: 0,
            last_request: Instant::now(),
            window_start: Instant::now(),
            requests_in_window: 0,
            connection_count: 0,
            violation_count: 0,
        }
    }
}

/// Block information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockInfo {
    pub ip: IpAddr,
    pub blocked_at: DateTime<Utc>,
    pub reason: BlockReason,
    pub violation_count: u32,
}

/// Reasons for blocking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockReason {
    RateLimitExceeded,
    TooManyConnections,
    SuspiciousPattern,
    ManualBlock,
}

/// Attack detection
struct AttackDetector {
    /// Traffic patterns
    patterns: DashMap<String, TrafficPattern>,
}

/// Traffic pattern analysis
#[derive(Debug, Clone)]
struct TrafficPattern {
    request_rate: f64,
    error_rate: f64,
    last_updated: Instant,
}

impl DDoSProtectionManager {
    /// Create a new DDoS protection manager
    pub fn new(config: DDoSProtectionConfig) -> Self {
        Self {
            config,
            ip_tracker: Arc::new(DashMap::new()),
            blocked_ips: Arc::new(DashMap::new()),
            whitelisted_ips: Arc::new(DashMap::new()),
            attack_detector: Arc::new(AttackDetector {
                patterns: DashMap::new(),
            }),
        }
    }

    /// Check if request is allowed
    pub async fn check_request(&self, ip: IpAddr) -> FusekiResult<RequestDecision> {
        if !self.config.enabled {
            return Ok(RequestDecision::Allow);
        }

        // Check whitelist
        if self.whitelisted_ips.contains_key(&ip) {
            debug!("IP {} is whitelisted", ip);
            return Ok(RequestDecision::Allow);
        }

        // Check if IP is blocked
        if let Some(block_info) = self.blocked_ips.get(&ip) {
            let block_age = (Utc::now() - block_info.blocked_at)
                .to_std()
                .unwrap_or(Duration::from_secs(0));
            if block_age < Duration::from_secs(self.config.block_duration_secs) {
                debug!("IP {} is blocked for {:?}", ip, block_age);
                return Ok(RequestDecision::Block {
                    reason: block_info.reason,
                    retry_after: (Duration::from_secs(self.config.block_duration_secs) - block_age)
                        .as_secs(),
                });
            } else {
                // Block expired, remove it
                self.blocked_ips.remove(&ip);
            }
        }

        // Track request and check limits
        let should_block_rate_limit: bool;
        let should_block_connections: bool;

        {
            // Scope the tracker lock to avoid deadlock when calling block_ip
            let mut tracker = self
                .ip_tracker
                .entry(ip)
                .or_insert_with(IpTrackingInfo::default);

            tracker.request_count += 1;
            tracker.last_request = Instant::now();

            // Check rate limit
            let window_age = tracker.window_start.elapsed();
            if window_age >= Duration::from_secs(1) {
                // New window
                tracker.window_start = Instant::now();
                tracker.requests_in_window = 1;
                should_block_rate_limit = false;
            } else {
                tracker.requests_in_window += 1;

                // Check if rate limit exceeded
                if tracker.requests_in_window > self.config.requests_per_second {
                    tracker.violation_count += 1;

                    if self.config.auto_block && tracker.violation_count >= 3 {
                        warn!("Auto-blocking IP {} due to rate limit violations", ip);
                        should_block_rate_limit = true;
                    } else {
                        // Rate limited but not blocked yet
                        return Ok(RequestDecision::RateLimit { retry_after: 1 });
                    }
                } else {
                    should_block_rate_limit = false;
                }
            }

            // Check connection limit
            should_block_connections = tracker.connection_count
                > self.config.max_connections_per_ip
                && self.config.auto_block;

            if should_block_connections {
                warn!(
                    "IP {} exceeded connection limit: {}",
                    ip, tracker.connection_count
                );
            }
        } // Drop the tracker lock here

        // Now call block_ip without holding the tracker lock
        if should_block_rate_limit {
            self.block_ip(ip, BlockReason::RateLimitExceeded).await?;
            return Ok(RequestDecision::Block {
                reason: BlockReason::RateLimitExceeded,
                retry_after: self.config.block_duration_secs,
            });
        }

        if should_block_connections {
            self.block_ip(ip, BlockReason::TooManyConnections).await?;
            return Ok(RequestDecision::Block {
                reason: BlockReason::TooManyConnections,
                retry_after: self.config.block_duration_secs,
            });
        }

        Ok(RequestDecision::Allow)
    }

    /// Register connection for IP
    pub async fn register_connection(&self, ip: IpAddr) {
        if !self.config.enabled {
            return;
        }

        let mut tracker = self
            .ip_tracker
            .entry(ip)
            .or_insert_with(IpTrackingInfo::default);

        tracker.connection_count += 1;
    }

    /// Unregister connection for IP
    pub async fn unregister_connection(&self, ip: IpAddr) {
        if !self.config.enabled {
            return;
        }

        if let Some(mut tracker) = self.ip_tracker.get_mut(&ip) {
            if tracker.connection_count > 0 {
                tracker.connection_count -= 1;
            }
        }
    }

    /// Block an IP address
    pub async fn block_ip(&self, ip: IpAddr, reason: BlockReason) -> FusekiResult<()> {
        let violation_count = self
            .ip_tracker
            .get(&ip)
            .map(|t| t.violation_count)
            .unwrap_or(0);

        let block_info = BlockInfo {
            ip,
            blocked_at: Utc::now(),
            reason,
            violation_count,
        };

        self.blocked_ips.insert(ip, block_info);
        info!("Blocked IP {} (reason: {:?})", ip, reason);

        Ok(())
    }

    /// Unblock an IP address
    pub async fn unblock_ip(&self, ip: IpAddr) -> FusekiResult<()> {
        self.blocked_ips.remove(&ip);
        info!("Unblocked IP {}", ip);
        Ok(())
    }

    /// Add IP to whitelist
    pub async fn whitelist_ip(&self, ip: IpAddr) -> FusekiResult<()> {
        self.whitelisted_ips.insert(ip, ());
        info!("Whitelisted IP {}", ip);
        Ok(())
    }

    /// Remove IP from whitelist
    pub async fn remove_from_whitelist(&self, ip: IpAddr) -> FusekiResult<()> {
        self.whitelisted_ips.remove(&ip);
        info!("Removed IP {} from whitelist", ip);
        Ok(())
    }

    /// Get blocked IPs
    pub fn get_blocked_ips(&self) -> Vec<BlockInfo> {
        self.blocked_ips
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get protection statistics
    pub fn get_statistics(&self) -> ProtectionStatistics {
        let total_ips = self.ip_tracker.len();
        let blocked_ips = self.blocked_ips.len();
        let whitelisted_ips = self.whitelisted_ips.len();

        let mut total_requests = 0u64;
        let mut total_violations = 0u32;

        for entry in self.ip_tracker.iter() {
            total_requests += entry.value().request_count;
            total_violations += entry.value().violation_count;
        }

        ProtectionStatistics {
            enabled: self.config.enabled,
            total_ips_tracked: total_ips,
            blocked_ips_count: blocked_ips,
            whitelisted_ips_count: whitelisted_ips,
            total_requests,
            total_violations,
            requests_per_second_limit: self.config.requests_per_second,
        }
    }

    /// Analyze traffic patterns
    pub async fn analyze_traffic(&self) -> TrafficAnalysisReport {
        if !self.config.enable_traffic_analysis {
            return TrafficAnalysisReport::default();
        }

        // Calculate overall request rate
        let now = Instant::now();
        let total_requests: u32 = self
            .ip_tracker
            .iter()
            .filter(|entry| entry.value().window_start.elapsed() < Duration::from_secs(60))
            .map(|entry| entry.value().requests_in_window)
            .sum();

        let request_rate = total_requests as f64 / 60.0;

        // Detect anomalies
        let anomalies = self.detect_anomalies();

        TrafficAnalysisReport {
            timestamp: now,
            overall_request_rate: request_rate,
            active_ips: self.ip_tracker.len(),
            anomalies_detected: anomalies.len(),
            anomalies,
        }
    }

    /// Detect traffic anomalies
    fn detect_anomalies(&self) -> Vec<TrafficAnomaly> {
        let mut anomalies = Vec::new();

        for entry in self.ip_tracker.iter() {
            let tracker = entry.value();

            // Check for suspicious patterns
            if tracker.requests_in_window > self.config.requests_per_second * 5 {
                anomalies.push(TrafficAnomaly {
                    ip: *entry.key(),
                    anomaly_type: AnomalyType::HighRequestRate,
                    severity: Severity::High,
                    description: format!(
                        "Request rate {} exceeds normal by 5x",
                        tracker.requests_in_window
                    ),
                });
            }

            if tracker.connection_count > self.config.max_connections_per_ip {
                anomalies.push(TrafficAnomaly {
                    ip: *entry.key(),
                    anomaly_type: AnomalyType::TooManyConnections,
                    severity: Severity::Medium,
                    description: format!("{} concurrent connections", tracker.connection_count),
                });
            }
        }

        anomalies
    }
}

/// Request decision
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestDecision {
    /// Allow the request
    Allow,
    /// Block the request
    Block {
        reason: BlockReason,
        retry_after: u64,
    },
    /// Rate limit the request
    RateLimit { retry_after: u64 },
    /// Challenge the client
    Challenge { challenge_type: ChallengeType },
}

/// Challenge types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChallengeType {
    Captcha,
    ProofOfWork,
    Cookie,
}

/// Protection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtectionStatistics {
    pub enabled: bool,
    pub total_ips_tracked: usize,
    pub blocked_ips_count: usize,
    pub whitelisted_ips_count: usize,
    pub total_requests: u64,
    pub total_violations: u32,
    pub requests_per_second_limit: u32,
}

/// Traffic analysis report
#[derive(Debug, Clone)]
pub struct TrafficAnalysisReport {
    pub timestamp: Instant,
    pub overall_request_rate: f64,
    pub active_ips: usize,
    pub anomalies_detected: usize,
    pub anomalies: Vec<TrafficAnomaly>,
}

impl Default for TrafficAnalysisReport {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            overall_request_rate: 0.0,
            active_ips: 0,
            anomalies_detected: 0,
            anomalies: Vec::new(),
        }
    }
}

/// Traffic anomaly
#[derive(Debug, Clone)]
pub struct TrafficAnomaly {
    pub ip: IpAddr,
    pub anomaly_type: AnomalyType,
    pub severity: Severity,
    pub description: String,
}

/// Anomaly types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    HighRequestRate,
    TooManyConnections,
    SuspiciousPattern,
}

/// Severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[tokio::test]
    async fn test_ddos_protection() {
        let config = DDoSProtectionConfig {
            enabled: true,
            requests_per_second: 10,
            burst_size: 20,
            block_duration_secs: 5, // Reduced from 60 for faster tests
            auto_block: true,
            enable_challenge: false,
            max_connections_per_ip: 5,
            enable_traffic_analysis: true,
        };

        let manager = DDoSProtectionManager::new(config);
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));

        // First request should be allowed
        let decision = manager.check_request(ip).await.unwrap();
        assert_eq!(decision, RequestDecision::Allow);

        // Simulate many requests rapidly (within same window)
        // Make 10 requests to trigger rate limiting without auto-blocking
        for _ in 0..10 {
            let _ = manager.check_request(ip).await;
        }

        // Should be rate limited (11th request exceeds limit of 10)
        let decision = manager.check_request(ip).await.unwrap();
        assert!(matches!(decision, RequestDecision::RateLimit { .. }));
    }

    #[tokio::test]
    async fn test_whitelist() {
        let config = DDoSProtectionConfig::default();
        let manager = DDoSProtectionManager::new(config);
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));

        manager.whitelist_ip(ip).await.unwrap();

        // Whitelisted IP should always be allowed
        for _ in 0..1000 {
            let decision = manager.check_request(ip).await.unwrap();
            assert_eq!(decision, RequestDecision::Allow);
        }
    }

    #[tokio::test]
    async fn test_block_ip() {
        let config = DDoSProtectionConfig::default();
        let manager = DDoSProtectionManager::new(config);
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));

        manager
            .block_ip(ip, BlockReason::ManualBlock)
            .await
            .unwrap();

        let decision = manager.check_request(ip).await.unwrap();
        assert!(matches!(decision, RequestDecision::Block { .. }));
    }

    #[tokio::test]
    async fn test_statistics() {
        let config = DDoSProtectionConfig::default();
        let manager = DDoSProtectionManager::new(config);

        let ip1 = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));
        let ip2 = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 2));

        manager.check_request(ip1).await.unwrap();
        manager.check_request(ip2).await.unwrap();

        let stats = manager.get_statistics();
        assert_eq!(stats.total_ips_tracked, 2);
        assert_eq!(stats.total_requests, 2);
    }
}
