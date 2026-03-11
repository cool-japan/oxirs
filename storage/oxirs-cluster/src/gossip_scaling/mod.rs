//! Large-scale cluster gossip scaling for 1000+ node deployments.
//!
//! This module provides adaptive gossip protocols optimised for massive cluster
//! sizes by combining:
//!
//! - **FanoutController**: adaptive fanout that responds to cluster size and message loss rate.
//! - **GossipPartitioner**: hierarchical zone/rack-aware partitioning to confine gossip rounds.
//! - **GossipMessageCompressor**: zstd-backed compression pipeline for gossip payloads.
//! - **ClusterMembershipIndex**: O(log N) consistent-hash ring for fast member lookup.

use crate::error::{ClusterError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::debug;

// ─────────────────────────────────────────────
//  Node identity
// ─────────────────────────────────────────────

/// Unique identifier for a cluster member.
pub type NodeId = u64;

/// Logical zone or datacenter label used for partitioning.
pub type ZoneLabel = String;

/// Full address of a cluster member (host + port).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemberAddr {
    /// Hostname or IP address.
    pub host: String,
    /// TCP/UDP port.
    pub port: u16,
}

impl MemberAddr {
    /// Creates a new member address.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
        }
    }
}

impl std::fmt::Display for MemberAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.host, self.port)
    }
}

/// Metadata attached to each cluster member.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberMeta {
    /// Unique node identifier.
    pub id: NodeId,
    /// Network address.
    pub addr: MemberAddr,
    /// Zone (datacenter, rack, availability zone …).
    pub zone: ZoneLabel,
    /// Wall-clock timestamp of last successful heartbeat (Unix seconds).
    pub last_seen_unix: u64,
    /// Whether the member is currently considered alive.
    pub alive: bool,
}

impl MemberMeta {
    /// Creates live metadata with the current timestamp.
    pub fn new(id: NodeId, addr: MemberAddr, zone: ZoneLabel) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        Self {
            id,
            addr,
            zone,
            last_seen_unix: now,
            alive: true,
        }
    }
}

// ─────────────────────────────────────────────
//  FanoutController
// ─────────────────────────────────────────────

/// Configuration for the adaptive fanout controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanoutConfig {
    /// Minimum fanout regardless of cluster size.
    pub min_fanout: usize,
    /// Maximum fanout cap to avoid flooding.
    pub max_fanout: usize,
    /// Desired gossip convergence time in rounds (used to compute baseline fanout).
    pub target_rounds: usize,
    /// Fraction of detected message loss that triggers a fanout increase.
    pub loss_increase_threshold: f64,
}

impl Default for FanoutConfig {
    fn default() -> Self {
        Self {
            min_fanout: 3,
            max_fanout: 12,
            target_rounds: 5,
            loss_increase_threshold: 0.05,
        }
    }
}

/// Adaptively controls gossip fanout for large clusters.
///
/// The baseline fanout is computed as:
///
/// ```text
/// fanout = ceil(log2(cluster_size) * scale_factor)
/// ```
///
/// and is further adjusted upward when message loss is detected.
pub struct FanoutController {
    config: FanoutConfig,
    cluster_size: AtomicUsize,
    current_fanout: AtomicUsize,
    loss_rate: Arc<RwLock<f64>>,
    rounds_completed: AtomicU64,
}

impl FanoutController {
    /// Creates a new fanout controller with the given configuration.
    pub fn new(config: FanoutConfig) -> Result<Self> {
        if config.min_fanout == 0 {
            return Err(ClusterError::Config("min_fanout must be >= 1".into()));
        }
        if config.max_fanout < config.min_fanout {
            return Err(ClusterError::Config(
                "max_fanout must be >= min_fanout".into(),
            ));
        }
        let initial = config.min_fanout;
        Ok(Self {
            config,
            cluster_size: AtomicUsize::new(1),
            current_fanout: AtomicUsize::new(initial),
            loss_rate: Arc::new(RwLock::new(0.0)),
            rounds_completed: AtomicU64::new(0),
        })
    }

    /// Updates the controller with the current cluster size and recomputes fanout.
    pub async fn update_cluster_size(&self, size: usize) {
        self.cluster_size.store(size, Ordering::Relaxed);
        self.recompute_fanout().await;
    }

    /// Reports the observed message loss rate for the most recent gossip round
    /// and adjusts fanout accordingly.
    pub async fn report_loss_rate(&self, loss: f64) {
        {
            let mut lr = self.loss_rate.write().await;
            *lr = loss.clamp(0.0, 1.0);
        }
        self.rounds_completed.fetch_add(1, Ordering::Relaxed);
        self.recompute_fanout().await;
    }

    /// Returns the currently recommended fanout.
    pub fn fanout(&self) -> usize {
        self.current_fanout.load(Ordering::Relaxed)
    }

    /// Recomputes the adaptive fanout from cluster size and loss rate.
    async fn recompute_fanout(&self) {
        let size = self.cluster_size.load(Ordering::Relaxed).max(1);
        let loss = *self.loss_rate.read().await;

        // Baseline: log2(N) rounded up, scaled by target rounds.
        let log2 = (size as f64).log2().ceil() as usize;
        let baseline = (log2.max(1) * 2) / self.config.target_rounds.max(1);
        let baseline = baseline.max(self.config.min_fanout);

        // Increase fanout if loss exceeds threshold.
        let adjusted = if loss >= self.config.loss_increase_threshold {
            let boost = ((loss / self.config.loss_increase_threshold) as usize).max(1);
            baseline.saturating_add(boost)
        } else {
            baseline
        };

        let clamped = adjusted
            .max(self.config.min_fanout)
            .min(self.config.max_fanout);

        let old = self.current_fanout.swap(clamped, Ordering::Relaxed);
        if old != clamped {
            debug!(
                old,
                new = clamped,
                cluster_size = size,
                loss,
                "Fanout adjusted"
            );
        }
    }
}

// ─────────────────────────────────────────────
//  GossipPartitioner
// ─────────────────────────────────────────────

/// Hierarchical gossip partitioner that restricts early gossip rounds to local
/// zones and expands to cross-zone gossip in later rounds.
pub struct GossipPartitioner {
    /// All members, grouped by zone label.
    zones: Arc<RwLock<HashMap<ZoneLabel, Vec<NodeId>>>>,
    /// Flat member metadata store.
    members: Arc<RwLock<HashMap<NodeId, MemberMeta>>>,
}

impl GossipPartitioner {
    /// Creates a new partitioner with no members.
    pub fn new() -> Self {
        Self {
            zones: Arc::new(RwLock::new(HashMap::new())),
            members: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Registers a member with zone metadata.
    pub async fn add_member(&self, meta: MemberMeta) {
        let zone = meta.zone.clone();
        let id = meta.id;
        {
            let mut zones = self.zones.write().await;
            zones.entry(zone).or_default().push(id);
        }
        let mut members = self.members.write().await;
        members.insert(id, meta);
    }

    /// Removes a member from the partitioner.
    pub async fn remove_member(&self, id: NodeId) {
        let zone = {
            let mut members = self.members.write().await;
            members.remove(&id).map(|m| m.zone)
        };
        if let Some(z) = zone {
            let mut zones = self.zones.write().await;
            if let Some(list) = zones.get_mut(&z) {
                list.retain(|&x| x != id);
                if list.is_empty() {
                    zones.remove(&z);
                }
            }
        }
    }

    /// Selects gossip targets for a given source node and round number.
    ///
    /// - Round 0–1: intra-zone gossip only (fast local convergence).
    /// - Round 2+: cross-zone gossip (global convergence).
    ///
    /// Returns up to `fanout` target node IDs.
    pub async fn select_targets(
        &self,
        source_id: NodeId,
        round: u32,
        fanout: usize,
    ) -> Vec<NodeId> {
        if fanout == 0 {
            return vec![];
        }

        let members = self.members.read().await;
        let zones = self.zones.read().await;

        // Determine source zone.
        let source_zone = members
            .get(&source_id)
            .map(|m| m.zone.clone())
            .unwrap_or_default();

        let candidates: Vec<NodeId> = if round < 2 && !source_zone.is_empty() {
            // Intra-zone candidates.
            zones
                .get(&source_zone)
                .map(|ids| ids.iter().copied().filter(|&x| x != source_id).collect())
                .unwrap_or_default()
        } else {
            // All alive members excluding source.
            members
                .values()
                .filter(|m| m.alive && m.id != source_id)
                .map(|m| m.id)
                .collect()
        };

        // Deterministic pseudo-random selection using the source ID and round
        // as a seed (no external RNG dependency).
        let mut selected = HashSet::with_capacity(fanout);
        let seed_base = source_id
            .wrapping_mul(6364136223846793005)
            .wrapping_add(round as u64);
        let n = candidates.len();
        if n == 0 {
            return vec![];
        }
        let mut attempts = 0_usize;
        while selected.len() < fanout && attempts < n * 2 {
            let idx = (seed_base.wrapping_add(attempts as u64) % n as u64) as usize;
            selected.insert(candidates[idx]);
            attempts += 1;
        }
        selected.into_iter().collect()
    }

    /// Returns the number of known zones.
    pub async fn zone_count(&self) -> usize {
        self.zones.read().await.len()
    }

    /// Returns the total number of registered members.
    pub async fn member_count(&self) -> usize {
        self.members.read().await.len()
    }

    /// Lists all zone labels.
    pub async fn zones(&self) -> Vec<ZoneLabel> {
        self.zones.read().await.keys().cloned().collect()
    }
}

impl Default for GossipPartitioner {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
//  GossipMessageCompressor
// ─────────────────────────────────────────────

/// Compression statistics for the gossip compressor.
#[derive(Debug, Default)]
pub struct CompressionStats {
    total_bytes_in: AtomicU64,
    total_bytes_out: AtomicU64,
    compress_calls: AtomicU64,
    decompress_calls: AtomicU64,
}

impl CompressionStats {
    /// Compression ratio (output / input). Returns 1.0 for no data.
    pub fn ratio(&self) -> f64 {
        let inn = self.total_bytes_in.load(Ordering::Relaxed);
        let out = self.total_bytes_out.load(Ordering::Relaxed);
        if inn == 0 {
            1.0
        } else {
            out as f64 / inn as f64
        }
    }
}

/// Zstd-backed compressor for gossip message payloads.
///
/// Payloads below `min_compress_bytes` are transmitted uncompressed to avoid
/// the overhead of compressing tiny messages.
pub struct GossipMessageCompressor {
    /// Compression level (1–22; default 3 for speed).
    level: i32,
    /// Minimum payload size for which compression is applied.
    min_compress_bytes: usize,
    stats: Arc<CompressionStats>,
}

impl GossipMessageCompressor {
    /// Creates a new compressor with the given zstd level and threshold.
    pub fn new(level: i32, min_compress_bytes: usize) -> Result<Self> {
        if !(1..=22).contains(&level) {
            return Err(ClusterError::Config(
                "zstd level must be in range 1–22".into(),
            ));
        }
        Ok(Self {
            level,
            min_compress_bytes,
            stats: Arc::new(CompressionStats::default()),
        })
    }

    /// Compresses `data` if it exceeds the threshold; otherwise returns it unchanged.
    ///
    /// The returned buffer is prefixed with a 1-byte tag:
    /// - `0x00` = uncompressed
    /// - `0x01` = zstd compressed
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.stats
            .total_bytes_in
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        self.stats.compress_calls.fetch_add(1, Ordering::Relaxed);

        let result = if data.len() < self.min_compress_bytes {
            let mut out = Vec::with_capacity(data.len() + 1);
            out.push(0x00_u8);
            out.extend_from_slice(data);
            out
        } else {
            let compressed = oxiarc_zstd::encode_all(data, self.level)
                .map_err(|e| ClusterError::Compression(e.to_string()))?;
            let mut out = Vec::with_capacity(compressed.len() + 1);
            out.push(0x01_u8);
            out.extend_from_slice(&compressed);
            out
        };

        self.stats
            .total_bytes_out
            .fetch_add(result.len() as u64, Ordering::Relaxed);
        Ok(result)
    }

    /// Decompresses a buffer produced by [`GossipMessageCompressor::compress`].
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.stats.decompress_calls.fetch_add(1, Ordering::Relaxed);

        let (tag, payload) = data
            .split_first()
            .ok_or_else(|| ClusterError::Compression("empty gossip message".into()))?;

        match tag {
            0x00 => Ok(payload.to_vec()),
            0x01 => oxiarc_zstd::decode_all(payload)
                .map_err(|e| ClusterError::Compression(e.to_string())),
            other => Err(ClusterError::Compression(format!(
                "unknown compression tag: {:#x}",
                other
            ))),
        }
    }

    /// Returns aggregate compression statistics.
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }
}

// ─────────────────────────────────────────────
//  ClusterMembershipIndex
// ─────────────────────────────────────────────

/// A consistent-hash ring providing O(log N) membership lookup.
///
/// Each node is mapped onto the ring using a lightweight FNV-1a-inspired hash
/// (no external dependency).  Virtual nodes (`vnodes_per_member`) improve key
/// distribution across heterogeneous clusters.
pub struct ClusterMembershipIndex {
    /// Ring: hash token → NodeId (sorted BTreeMap for O(log N) successor lookup).
    ring: Arc<RwLock<BTreeMap<u64, NodeId>>>,
    /// Member metadata store.
    members: Arc<RwLock<HashMap<NodeId, MemberMeta>>>,
    /// Number of virtual ring tokens per physical node.
    vnodes_per_member: usize,
    /// Counter for monitoring.
    lookup_count: AtomicU64,
}

impl ClusterMembershipIndex {
    /// Creates a new index with the specified virtual node count.
    pub fn new(vnodes_per_member: usize) -> Result<Self> {
        if vnodes_per_member == 0 {
            return Err(ClusterError::Config(
                "vnodes_per_member must be >= 1".into(),
            ));
        }
        Ok(Self {
            ring: Arc::new(RwLock::new(BTreeMap::new())),
            members: Arc::new(RwLock::new(HashMap::new())),
            vnodes_per_member,
            lookup_count: AtomicU64::new(0),
        })
    }

    /// Adds a member to the ring.
    pub async fn add_member(&self, meta: MemberMeta) {
        let id = meta.id;
        {
            let mut ring = self.ring.write().await;
            for v in 0..self.vnodes_per_member {
                let token = Self::hash_token(id, v);
                ring.insert(token, id);
            }
        }
        let mut members = self.members.write().await;
        members.insert(id, meta);
    }

    /// Removes a member from the ring.
    pub async fn remove_member(&self, id: NodeId) {
        {
            let mut ring = self.ring.write().await;
            for v in 0..self.vnodes_per_member {
                let token = Self::hash_token(id, v);
                ring.remove(&token);
            }
        }
        let mut members = self.members.write().await;
        members.remove(&id);
    }

    /// Finds the responsible node for a given key using a clockwise ring lookup.
    ///
    /// Returns the NodeId of the successor node, or `None` if the cluster is empty.
    pub async fn lookup(&self, key: &[u8]) -> Option<NodeId> {
        self.lookup_count.fetch_add(1, Ordering::Relaxed);
        let key_hash = Self::hash_key(key);
        let ring = self.ring.read().await;
        if ring.is_empty() {
            return None;
        }
        // Clockwise successor: find the first token >= key_hash, or wrap around.
        ring.range(key_hash..)
            .next()
            .or_else(|| ring.iter().next())
            .map(|(_, &id)| id)
    }

    /// Returns the N closest nodes (replicas) for a key.
    pub async fn lookup_replicas(&self, key: &[u8], n: usize) -> Vec<NodeId> {
        if n == 0 {
            return vec![];
        }
        let key_hash = Self::hash_key(key);
        let ring = self.ring.read().await;
        let member_count = {
            let members = self.members.read().await;
            members.len()
        };
        let effective_n = n.min(member_count);
        let mut seen = HashSet::with_capacity(effective_n);
        let mut result = Vec::with_capacity(effective_n);

        // Walk the ring clockwise starting from key_hash.
        let forward = ring.range(key_hash..).map(|(_, &id)| id);
        let wrap = ring.range(..key_hash).map(|(_, &id)| id);
        for id in forward.chain(wrap) {
            if seen.insert(id) {
                result.push(id);
                if result.len() == effective_n {
                    break;
                }
            }
        }
        result
    }

    /// Returns the total number of registered members.
    pub async fn member_count(&self) -> usize {
        self.members.read().await.len()
    }

    /// Returns the total number of lookup operations performed.
    pub fn lookup_count(&self) -> u64 {
        self.lookup_count.load(Ordering::Relaxed)
    }

    /// Returns metadata for a specific member.
    pub async fn get_member(&self, id: NodeId) -> Option<MemberMeta> {
        self.members.read().await.get(&id).cloned()
    }

    /// Lists all alive member IDs.
    pub async fn alive_members(&self) -> Vec<NodeId> {
        let members = self.members.read().await;
        members.values().filter(|m| m.alive).map(|m| m.id).collect()
    }

    /// FNV-1a inspired hash of (node_id, vnode_index).
    fn hash_token(node_id: NodeId, vnode: usize) -> u64 {
        const PRIME: u64 = 1_099_511_628_211;
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in node_id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
        for b in (vnode as u64).to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
        h
    }

    /// FNV-1a hash of a byte slice.
    fn hash_key(key: &[u8]) -> u64 {
        const PRIME: u64 = 1_099_511_628_211;
        let mut h: u64 = 14_695_981_039_346_656_037;
        for &b in key {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
        h
    }
}

// ─────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_meta(id: NodeId, zone: &str) -> MemberMeta {
        MemberMeta::new(
            id,
            MemberAddr::new("127.0.0.1", 8000 + id as u16),
            zone.into(),
        )
    }

    // ── FanoutController ─────────────────────

    #[tokio::test]
    async fn test_fanout_controller_defaults() {
        let ctrl = FanoutController::new(FanoutConfig::default()).expect("new");
        assert!(ctrl.fanout() >= 3);
    }

    #[tokio::test]
    async fn test_fanout_controller_scales_with_cluster_size() {
        let ctrl = FanoutController::new(FanoutConfig::default()).expect("new");
        ctrl.update_cluster_size(10).await;
        let small = ctrl.fanout();
        ctrl.update_cluster_size(1000).await;
        let large = ctrl.fanout();
        // Larger cluster should trigger equal or higher fanout.
        assert!(large >= small);
    }

    #[tokio::test]
    async fn test_fanout_controller_clamps_to_max() {
        let cfg = FanoutConfig {
            min_fanout: 2,
            max_fanout: 6,
            ..Default::default()
        };
        let ctrl = FanoutController::new(cfg).expect("new");
        ctrl.update_cluster_size(1_000_000).await;
        assert!(ctrl.fanout() <= 6);
    }

    #[tokio::test]
    async fn test_fanout_controller_loss_increases_fanout() {
        let cfg = FanoutConfig {
            min_fanout: 3,
            max_fanout: 12,
            loss_increase_threshold: 0.05,
            target_rounds: 5,
        };
        let ctrl = FanoutController::new(cfg).expect("new");
        ctrl.update_cluster_size(100).await;
        let no_loss = ctrl.fanout();
        ctrl.report_loss_rate(0.25).await;
        let with_loss = ctrl.fanout();
        // Loss should push fanout >= no-loss fanout.
        assert!(with_loss >= no_loss);
    }

    // ── GossipPartitioner ────────────────────

    #[tokio::test]
    async fn test_gossip_partitioner_add_members() {
        let p = GossipPartitioner::new();
        p.add_member(make_meta(1, "zone-a")).await;
        p.add_member(make_meta(2, "zone-a")).await;
        p.add_member(make_meta(3, "zone-b")).await;
        assert_eq!(p.member_count().await, 3);
        assert_eq!(p.zone_count().await, 2);
    }

    #[tokio::test]
    async fn test_gossip_partitioner_remove_member() {
        let p = GossipPartitioner::new();
        p.add_member(make_meta(1, "zone-a")).await;
        p.add_member(make_meta(2, "zone-a")).await;
        p.remove_member(2).await;
        assert_eq!(p.member_count().await, 1);
    }

    #[tokio::test]
    async fn test_gossip_partitioner_intra_zone_round0() {
        let p = GossipPartitioner::new();
        for i in 1..=5_u64 {
            p.add_member(make_meta(i, "zone-a")).await;
        }
        for i in 6..=10_u64 {
            p.add_member(make_meta(i, "zone-b")).await;
        }
        let targets = p.select_targets(1, 0, 3).await;
        // All zone-a members except node 1.
        assert!(!targets.contains(&1));
        assert!(targets.len() <= 3);
    }

    // ── GossipMessageCompressor ──────────────

    #[test]
    fn test_compressor_roundtrip_small() {
        let c = GossipMessageCompressor::new(3, 512).expect("new");
        let data = b"hello gossip world";
        let compressed = c.compress(data).expect("compress");
        let decompressed = c.decompress(&compressed).expect("decompress");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compressor_roundtrip_large() {
        let c = GossipMessageCompressor::new(3, 16).expect("new");
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let compressed = c.compress(&data).expect("compress");
        let decompressed = c.decompress(&compressed).expect("decompress");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compressor_ratio_improves_for_compressible_data() {
        let c = GossipMessageCompressor::new(3, 16).expect("new");
        let data: Vec<u8> = vec![0xAB; 4096]; // Highly compressible.
        let compressed = c.compress(&data).expect("compress");
        assert!(
            compressed.len() < data.len(),
            "compressed should be smaller"
        );
    }

    #[test]
    fn test_compressor_invalid_level() {
        assert!(GossipMessageCompressor::new(0, 16).is_err());
        assert!(GossipMessageCompressor::new(23, 16).is_err());
    }

    // ── ClusterMembershipIndex ───────────────

    #[tokio::test]
    async fn test_membership_index_add_lookup() {
        let idx = ClusterMembershipIndex::new(10).expect("new");
        for i in 1..=5_u64 {
            idx.add_member(make_meta(i, "zone-a")).await;
        }
        assert_eq!(idx.member_count().await, 5);
        let node = idx.lookup(b"some-rdf-triple-key").await;
        assert!(node.is_some());
    }

    #[tokio::test]
    async fn test_membership_index_remove_member() {
        let idx = ClusterMembershipIndex::new(5).expect("new");
        idx.add_member(make_meta(1, "z1")).await;
        idx.add_member(make_meta(2, "z1")).await;
        idx.remove_member(1).await;
        assert_eq!(idx.member_count().await, 1);
    }

    #[tokio::test]
    async fn test_membership_index_replica_count() {
        let idx = ClusterMembershipIndex::new(10).expect("new");
        for i in 1..=5_u64 {
            idx.add_member(make_meta(i, "zone-a")).await;
        }
        let replicas = idx.lookup_replicas(b"test-key", 3).await;
        assert_eq!(replicas.len(), 3);
        // All returned replicas should be distinct.
        let set: HashSet<_> = replicas.iter().copied().collect();
        assert_eq!(set.len(), 3);
    }

    #[tokio::test]
    async fn test_membership_index_lookup_empty() {
        let idx = ClusterMembershipIndex::new(5).expect("new");
        let node = idx.lookup(b"key").await;
        assert!(node.is_none());
    }

    #[tokio::test]
    async fn test_membership_index_lookup_counter() {
        let idx = ClusterMembershipIndex::new(5).expect("new");
        idx.add_member(make_meta(1, "z")).await;
        idx.lookup(b"k1").await;
        idx.lookup(b"k2").await;
        assert_eq!(idx.lookup_count(), 2);
    }

    #[tokio::test]
    async fn test_membership_index_alive_members() {
        let idx = ClusterMembershipIndex::new(5).expect("new");
        idx.add_member(make_meta(1, "z")).await;
        idx.add_member(make_meta(2, "z")).await;
        let alive = idx.alive_members().await;
        assert_eq!(alive.len(), 2);
    }
}
