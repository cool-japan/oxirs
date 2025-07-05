//! Byzantine behavior detection systems and security components

#![allow(dead_code)]

use super::types::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

/// Byzantine behavior detection system with advanced threat detection
pub struct ByzantineDetector {
    /// Suspected Byzantine nodes
    suspected_nodes: HashSet<NodeId>,

    /// Message timing anomalies with detailed analysis
    timing_anomalies: HashMap<NodeId, TimingAnalysis>,

    /// Signature verification failures
    signature_failures: HashMap<NodeId, usize>,

    /// Inconsistent message patterns
    inconsistent_patterns: HashMap<NodeId, usize>,

    /// Detection threshold
    detection_threshold: usize,

    /// Network partition detection
    partition_detector: PartitionDetector,

    /// Message replay attack detection
    replay_detector: ReplayDetector,

    /// Equivocation detection (sending different messages for same view/sequence)
    equivocation_detector: EquivocationDetector,

    /// Resource exhaustion attack detection
    resource_monitor: ResourceMonitor,

    /// Collusion detection between nodes
    collusion_detector: CollusionDetector,
}

/// Advanced timing analysis for Byzantine detection
#[derive(Debug, Clone)]
pub struct TimingAnalysis {
    /// Recent message timestamps
    message_times: VecDeque<Instant>,
    /// Average response time
    avg_response_time: Duration,
    /// Standard deviation of response times
    response_time_stddev: Duration,
    /// Suspicious timing patterns count
    suspicious_patterns: usize,
}

/// Network partition detection system
#[derive(Debug, Clone)]
pub struct PartitionDetector {
    /// Last communication time with each node
    last_communication: HashMap<NodeId, Instant>,
    /// Suspected partitioned nodes
    partitioned_nodes: HashSet<NodeId>,
    /// Partition timeout threshold
    partition_timeout: Duration,
}

/// Replay attack detection system
#[derive(Debug, Clone)]
pub struct ReplayDetector {
    /// Recently seen message hashes with timestamps
    seen_messages: HashMap<Vec<u8>, Instant>,
    /// Replay attack threshold
    replay_window: Duration,
    /// Detected replay attempts
    replay_attempts: HashMap<NodeId, usize>,
}

/// Equivocation detection system
#[derive(Debug, Clone)]
pub struct EquivocationDetector {
    /// Messages per view/sequence from each node
    node_messages: HashMap<NodeId, HashMap<(ViewNumber, SequenceNumber), Vec<Vec<u8>>>>,
    /// Detected equivocations
    equivocations: HashMap<NodeId, usize>,
}

/// Resource exhaustion monitoring
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ResourceMonitor {
    /// Message rate per node
    message_rates: HashMap<NodeId, VecDeque<Instant>>,
    /// Rate limit threshold (messages per second)
    rate_limit: f64,
    /// Memory usage tracking
    memory_usage: HashMap<NodeId, usize>,
    /// Detected resource attacks
    resource_attacks: HashMap<NodeId, usize>,
}

/// Collusion detection between Byzantine nodes
#[derive(Debug, Clone)]
pub struct CollusionDetector {
    /// Coordinated behavior patterns
    coordination_patterns: HashMap<Vec<NodeId>, usize>,
    /// Simultaneous actions tracking
    simultaneous_actions: VecDeque<(Instant, Vec<NodeId>)>,
    /// Collusion threshold
    collusion_threshold: usize,
}

impl PartitionDetector {
    pub fn new() -> Self {
        Self {
            last_communication: HashMap::new(),
            partitioned_nodes: HashSet::new(),
            partition_timeout: Duration::from_secs(30),
        }
    }
}

impl ReplayDetector {
    pub fn new() -> Self {
        Self {
            seen_messages: HashMap::new(),
            replay_window: Duration::from_secs(60), // 1 minute window
            replay_attempts: HashMap::new(),
        }
    }
}

impl EquivocationDetector {
    pub fn new() -> Self {
        Self {
            node_messages: HashMap::new(),
            equivocations: HashMap::new(),
        }
    }
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {
            message_rates: HashMap::new(),
            rate_limit: 100.0, // 100 messages per second default limit
            memory_usage: HashMap::new(),
            resource_attacks: HashMap::new(),
        }
    }
}

impl CollusionDetector {
    pub fn new() -> Self {
        Self {
            coordination_patterns: HashMap::new(),
            simultaneous_actions: VecDeque::new(),
            collusion_threshold: 5, // 5 coordinated actions trigger suspicion
        }
    }
}

impl ByzantineDetector {
    pub fn new(detection_threshold: usize) -> Self {
        Self {
            suspected_nodes: HashSet::new(),
            timing_anomalies: HashMap::new(),
            signature_failures: HashMap::new(),
            inconsistent_patterns: HashMap::new(),
            detection_threshold,
            partition_detector: PartitionDetector::new(),
            replay_detector: ReplayDetector::new(),
            equivocation_detector: EquivocationDetector::new(),
            resource_monitor: ResourceMonitor::new(),
            collusion_detector: CollusionDetector::new(),
        }
    }

    /// Report advanced timing anomaly with detailed analysis
    pub fn report_timing_anomaly(&mut self, node_id: NodeId, response_time: Duration) {
        let now = Instant::now();

        // First, update/create the timing analysis
        {
            let analysis = self
                .timing_anomalies
                .entry(node_id)
                .or_insert_with(|| TimingAnalysis {
                    message_times: VecDeque::new(),
                    avg_response_time: Duration::from_millis(100), // Default
                    response_time_stddev: Duration::from_millis(50),
                    suspicious_patterns: 0,
                });

            analysis.message_times.push_back(now);

            // Keep only recent timing data (last 100 messages)
            while analysis.message_times.len() > 100 {
                analysis.message_times.pop_front();
            }
        }

        // Update statistics (separate borrow)
        self.update_timing_statistics(node_id, response_time);

        // Detect suspicious patterns and update if needed
        let is_suspicious = self.detect_timing_attack(node_id, response_time);
        if is_suspicious {
            if let Some(analysis) = self.timing_anomalies.get_mut(&node_id) {
                analysis.suspicious_patterns += 1;
                if analysis.suspicious_patterns >= self.detection_threshold {
                    self.suspected_nodes.insert(node_id);
                    tracing::warn!("Node {} suspected of timing attacks", node_id);
                }
            }
        }
    }

    /// Detect potential timing-based attacks
    fn detect_timing_attack(&self, node_id: NodeId, response_time: Duration) -> bool {
        if let Some(analysis) = self.timing_anomalies.get(&node_id) {
            // Check for extremely fast responses (potential pre-computation)
            if response_time < Duration::from_millis(1) {
                return true;
            }

            // Check for extremely slow responses (potential DoS)
            if response_time > analysis.avg_response_time + 3 * analysis.response_time_stddev {
                return true;
            }

            // Check for suspiciously regular timing (potential automation)
            if analysis.message_times.len() >= 10 {
                let intervals: Vec<_> = analysis
                    .message_times
                    .iter()
                    .zip(analysis.message_times.iter().skip(1))
                    .map(|(a, b)| b.duration_since(*a))
                    .collect();

                // If all intervals are too similar, it's suspicious
                if let (Some(&min), Some(&max)) = (intervals.iter().min(), intervals.iter().max()) {
                    if max - min < Duration::from_millis(10) && intervals.len() >= 5 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Update timing statistics for a node
    fn update_timing_statistics(&mut self, node_id: NodeId, response_time: Duration) {
        if let Some(analysis) = self.timing_anomalies.get_mut(&node_id) {
            // Simple exponential moving average
            let alpha = 0.1;
            let new_time = response_time.as_millis() as f64;
            let old_avg = analysis.avg_response_time.as_millis() as f64;
            let new_avg = alpha * new_time + (1.0 - alpha) * old_avg;
            analysis.avg_response_time = Duration::from_millis(new_avg as u64);
        }
    }

    pub fn report_signature_failure(&mut self, node_id: NodeId) {
        *self.signature_failures.entry(node_id).or_default() += 1;
        if self.signature_failures[&node_id] >= self.detection_threshold {
            self.suspected_nodes.insert(node_id);
            tracing::warn!("Node {} suspected due to signature failures", node_id);
        }
    }

    pub fn report_inconsistent_pattern(&mut self, node_id: NodeId) {
        *self.inconsistent_patterns.entry(node_id).or_default() += 1;
        if self.inconsistent_patterns[&node_id] >= self.detection_threshold {
            self.suspected_nodes.insert(node_id);
            tracing::warn!("Node {} suspected due to inconsistent patterns", node_id);
        }
    }

    /// Check for message replay attacks
    pub fn check_replay_attack(&mut self, node_id: NodeId, message_hash: Vec<u8>) -> bool {
        let now = Instant::now();

        // Clean old entries
        self.replay_detector
            .seen_messages
            .retain(|_, &mut timestamp| {
                now.duration_since(timestamp) <= self.replay_detector.replay_window
            });

        // Check if message was seen recently
        if let Some(&timestamp) = self.replay_detector.seen_messages.get(&message_hash) {
            if now.duration_since(timestamp) <= self.replay_detector.replay_window {
                *self
                    .replay_detector
                    .replay_attempts
                    .entry(node_id)
                    .or_default() += 1;
                if self.replay_detector.replay_attempts[&node_id] >= self.detection_threshold {
                    self.suspected_nodes.insert(node_id);
                    tracing::warn!("Node {} suspected of replay attacks", node_id);
                }
                return true;
            }
        }

        self.replay_detector.seen_messages.insert(message_hash, now);
        false
    }

    /// Detect equivocation (sending different messages for same view/sequence)
    pub fn check_equivocation(
        &mut self,
        node_id: NodeId,
        view: ViewNumber,
        sequence: SequenceNumber,
        message_hash: Vec<u8>,
    ) -> bool {
        let messages = self
            .equivocation_detector
            .node_messages
            .entry(node_id)
            .or_default()
            .entry((view, sequence))
            .or_default();

        // Check if we've seen a different message for this view/sequence
        if !messages.is_empty() && !messages.contains(&message_hash) {
            *self
                .equivocation_detector
                .equivocations
                .entry(node_id)
                .or_default() += 1;
            if self.equivocation_detector.equivocations[&node_id] >= self.detection_threshold {
                self.suspected_nodes.insert(node_id);
                tracing::warn!("Node {} suspected of equivocation", node_id);
            }
            return true;
        }

        messages.push(message_hash);
        false
    }

    /// Monitor resource usage for DoS attacks
    pub fn monitor_resource_usage(&mut self, node_id: NodeId) -> bool {
        let now = Instant::now();
        let rates = self
            .resource_monitor
            .message_rates
            .entry(node_id)
            .or_default();

        rates.push_back(now);

        // Keep only messages from the last second
        while let Some(&front_time) = rates.front() {
            if now.duration_since(front_time) > Duration::from_secs(1) {
                rates.pop_front();
            } else {
                break;
            }
        }

        // Check if rate exceeds threshold
        let current_rate = rates.len() as f64;
        if current_rate > self.resource_monitor.rate_limit {
            *self
                .resource_monitor
                .resource_attacks
                .entry(node_id)
                .or_default() += 1;
            if self.resource_monitor.resource_attacks[&node_id] >= self.detection_threshold {
                self.suspected_nodes.insert(node_id);
                tracing::warn!("Node {} suspected of resource exhaustion attack", node_id);
            }
            return true;
        }

        false
    }

    /// Detect potential collusion between nodes
    pub fn check_collusion(&mut self, coordinating_nodes: Vec<NodeId>) {
        if coordinating_nodes.len() >= 2 {
            let now = Instant::now();

            // Record simultaneous action
            self.collusion_detector
                .simultaneous_actions
                .push_back((now, coordinating_nodes.clone()));

            // Clean old entries (keep last hour)
            while let Some((timestamp, _)) = self.collusion_detector.simultaneous_actions.front() {
                if now.duration_since(*timestamp) > Duration::from_secs(3600) {
                    self.collusion_detector.simultaneous_actions.pop_front();
                } else {
                    break;
                }
            }

            // Check for repeated coordination
            *self
                .collusion_detector
                .coordination_patterns
                .entry(coordinating_nodes.clone())
                .or_default() += 1;

            if self.collusion_detector.coordination_patterns[&coordinating_nodes]
                >= self.collusion_detector.collusion_threshold
            {
                for &node_id in &coordinating_nodes {
                    self.suspected_nodes.insert(node_id);
                }
                tracing::warn!(
                    "Suspected collusion detected between nodes: {:?}",
                    coordinating_nodes
                );
            }
        }
    }

    /// Check network partition status
    pub fn check_network_partition(&mut self, node_id: NodeId) {
        let now = Instant::now();
        self.partition_detector
            .last_communication
            .insert(node_id, now);

        // Check for partitioned nodes
        for (&id, &last_time) in &self.partition_detector.last_communication {
            if now.duration_since(last_time) > self.partition_detector.partition_timeout {
                self.partition_detector.partitioned_nodes.insert(id);
            } else {
                self.partition_detector.partitioned_nodes.remove(&id);
            }
        }
    }

    /// Get comprehensive threat assessment
    pub fn get_threat_assessment(&self, node_id: NodeId) -> ThreatLevel {
        let mut score = 0;

        if self.suspected_nodes.contains(&node_id) {
            score += 10;
        }

        if let Some(failures) = self.signature_failures.get(&node_id) {
            score += failures * 2;
        }

        if let Some(patterns) = self.inconsistent_patterns.get(&node_id) {
            score += patterns;
        }

        if let Some(replays) = self.replay_detector.replay_attempts.get(&node_id) {
            score += replays * 3;
        }

        if let Some(equivocations) = self.equivocation_detector.equivocations.get(&node_id) {
            score += equivocations * 5;
        }

        if let Some(attacks) = self.resource_monitor.resource_attacks.get(&node_id) {
            score += attacks;
        }

        match score {
            0..=2 => ThreatLevel::Low,
            3..=7 => ThreatLevel::Medium,
            8..=15 => ThreatLevel::High,
            _ => ThreatLevel::Critical,
        }
    }

    pub fn is_suspected(&self, node_id: NodeId) -> bool {
        self.suspected_nodes.contains(&node_id)
    }

    pub fn get_suspected_nodes(&self) -> &HashSet<NodeId> {
        &self.suspected_nodes
    }

    pub fn is_partitioned(&self, node_id: NodeId) -> bool {
        self.partition_detector.partitioned_nodes.contains(&node_id)
    }
}
