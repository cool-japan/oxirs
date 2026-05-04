//! Cross-region anti-entropy via Merkle tree comparison.
//!
//! Extends the existing intra-cluster [`crate::merkle_tree::MerkleTree`] with
//! a thin wrapper that holds one tree per region and produces a structured
//! divergence report that the active-active replicator can use to drive
//! catch-up shipments.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use super::active_active_geo::RegionId;
use crate::error::{ClusterError, Result};
use crate::merkle_tree::{MerkleComparison, MerkleTree};

// ─────────────────────────────────────────────────────────────────────────────
// Divergence types
// ─────────────────────────────────────────────────────────────────────────────

/// Divergence between the local region tree and one peer region tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionDivergence {
    /// Region that the comparison was made against.
    pub peer_region: RegionId,
    /// Keys present locally but missing on the peer.
    pub keys_missing_on_peer: BTreeSet<String>,
    /// Keys present on the peer but missing locally.
    pub keys_missing_locally: BTreeSet<String>,
    /// Keys present in both with conflicting hashes.
    pub keys_with_conflicts: BTreeSet<String>,
}

impl RegionDivergence {
    /// `true` when the two regions contain exactly the same keys with
    /// identical hashes.
    pub fn is_in_sync(&self) -> bool {
        self.keys_missing_on_peer.is_empty()
            && self.keys_missing_locally.is_empty()
            && self.keys_with_conflicts.is_empty()
    }

    /// Total number of divergent keys (sum of all three categories).
    pub fn total_divergent_keys(&self) -> usize {
        self.keys_missing_on_peer.len()
            + self.keys_missing_locally.len()
            + self.keys_with_conflicts.len()
    }
}

/// Aggregate divergence across all regions versus the local one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossRegionDivergence {
    /// Region that produced this report (the "local" side of every diff).
    pub local_region: RegionId,
    /// Per-peer divergence breakdown.
    pub per_peer: BTreeMap<RegionId, RegionDivergence>,
}

impl CrossRegionDivergence {
    /// `true` when every peer is in sync with the local region.
    pub fn is_in_sync(&self) -> bool {
        self.per_peer.values().all(|d| d.is_in_sync())
    }

    /// Iterator over peer regions that have at least one divergent key.
    pub fn divergent_peers(&self) -> impl Iterator<Item = &RegionId> {
        self.per_peer
            .iter()
            .filter(|(_, d)| !d.is_in_sync())
            .map(|(r, _)| r)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-region anti-entropy controller
// ─────────────────────────────────────────────────────────────────────────────

/// Holds one Merkle tree per region and runs anti-entropy comparisons.
///
/// `local_region` is the region whose data this node owns; the per-peer
/// trees represent the latest snapshot received from peer regions (for
/// example by gossip or an explicit anti-entropy round). Comparisons are
/// always *local vs peer*, so the divergence report is naturally rooted at
/// the local region.
#[derive(Debug)]
pub struct CrossRegionAntiEntropy {
    local_region: RegionId,
    inner: Arc<Mutex<AntiEntropyState>>,
}

#[derive(Debug)]
struct AntiEntropyState {
    local_tree: Arc<MerkleTree>,
    peer_trees: BTreeMap<RegionId, Arc<MerkleTree>>,
}

impl CrossRegionAntiEntropy {
    /// Build a fresh controller seeded with an empty local tree.
    pub fn new(local_region: impl Into<RegionId>) -> Self {
        Self {
            local_region: local_region.into(),
            inner: Arc::new(Mutex::new(AntiEntropyState {
                local_tree: Arc::new(MerkleTree::new()),
                peer_trees: BTreeMap::new(),
            })),
        }
    }

    /// Identifier of the local region.
    pub fn local_region(&self) -> &RegionId {
        &self.local_region
    }

    /// Replace the local tree with a fresh handle (e.g. after rebuild).
    pub async fn set_local_tree(&self, tree: Arc<MerkleTree>) {
        let mut st = self.inner.lock().await;
        st.local_tree = tree;
    }

    /// Get a clonable handle to the local tree for direct manipulation.
    pub async fn local_tree(&self) -> Arc<MerkleTree> {
        self.inner.lock().await.local_tree.clone()
    }

    /// Insert (or overwrite) the snapshot tree for a peer region.
    pub async fn upsert_peer_tree(&self, region: RegionId, tree: Arc<MerkleTree>) -> Result<()> {
        if region == self.local_region {
            return Err(ClusterError::Config(format!(
                "Cannot register the local region '{}' as a peer",
                region
            )));
        }
        let mut st = self.inner.lock().await;
        st.peer_trees.insert(region, tree);
        Ok(())
    }

    /// Remove the snapshot for a peer region.
    pub async fn drop_peer_tree(&self, region: &RegionId) -> bool {
        let mut st = self.inner.lock().await;
        st.peer_trees.remove(region).is_some()
    }

    /// Compute divergence against a single peer region.
    pub async fn compare_with(&self, peer_region: &RegionId) -> Result<RegionDivergence> {
        let st = self.inner.lock().await;
        let peer_tree = st.peer_trees.get(peer_region).cloned().ok_or_else(|| {
            ClusterError::Config(format!("Unknown peer region '{}'", peer_region))
        })?;
        let local_tree = st.local_tree.clone();
        drop(st);
        Self::diff_into(peer_region, &local_tree, &peer_tree).await
    }

    /// Compute divergence across all registered peer regions.
    pub async fn compare_all(&self) -> Result<CrossRegionDivergence> {
        let (local_tree, peer_trees) = {
            let st = self.inner.lock().await;
            (
                st.local_tree.clone(),
                st.peer_trees
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Vec<_>>(),
            )
        };
        let mut per_peer = BTreeMap::new();
        for (region, tree) in peer_trees {
            let div = Self::diff_into(&region, &local_tree, &tree).await?;
            per_peer.insert(region, div);
        }
        Ok(CrossRegionDivergence {
            local_region: self.local_region.clone(),
            per_peer,
        })
    }

    async fn diff_into(
        peer_region: &RegionId,
        local: &MerkleTree,
        peer: &MerkleTree,
    ) -> Result<RegionDivergence> {
        match local.compare(peer).await {
            MerkleComparison::Identical => Ok(RegionDivergence {
                peer_region: peer_region.clone(),
                keys_missing_on_peer: BTreeSet::new(),
                keys_missing_locally: BTreeSet::new(),
                keys_with_conflicts: BTreeSet::new(),
            }),
            // The intra-cluster Merkle compare is asymmetric — it labels keys
            // from the perspective of `self` (the receiver of `.compare()`).
            //
            // Internally:
            //   • `extra_keys` — present in `self` (local) but not in `other` (peer).
            //   • `missing_keys` — present in `other` (peer) but not in `self` (local).
            //   • `conflicting_keys` — present in both with different hashes.
            //
            // Translate that into the cross-region naming we surface to callers.
            MerkleComparison::Different {
                missing_keys,
                extra_keys,
                conflicting_keys,
            } => Ok(RegionDivergence {
                peer_region: peer_region.clone(),
                keys_missing_on_peer: extra_keys.into_iter().collect(),
                keys_missing_locally: missing_keys.into_iter().collect(),
                keys_with_conflicts: conflicting_keys.into_iter().collect(),
            }),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn populated_tree(pairs: &[(&str, &str)]) -> Arc<MerkleTree> {
        let tree = Arc::new(MerkleTree::new());
        for (k, v) in pairs {
            tree.insert(k.to_string(), v).await;
        }
        tree
    }

    #[tokio::test]
    async fn identical_trees_report_no_divergence() {
        let ae = CrossRegionAntiEntropy::new("us-east-1");
        let local = populated_tree(&[("k1", "v1"), ("k2", "v2")]).await;
        let peer = populated_tree(&[("k1", "v1"), ("k2", "v2")]).await;
        ae.set_local_tree(local).await;
        ae.upsert_peer_tree("eu-west-1".into(), peer)
            .await
            .expect("upsert");
        let div = ae.compare_with(&"eu-west-1".into()).await.expect("compare");
        assert!(div.is_in_sync());
    }

    #[tokio::test]
    async fn detects_keys_missing_on_peer() {
        let ae = CrossRegionAntiEntropy::new("us-east-1");
        let local = populated_tree(&[("k1", "v1"), ("k2", "v2")]).await;
        let peer = populated_tree(&[("k1", "v1")]).await;
        ae.set_local_tree(local).await;
        ae.upsert_peer_tree("eu-west-1".into(), peer)
            .await
            .expect("upsert");
        let div = ae.compare_with(&"eu-west-1".into()).await.expect("compare");
        assert_eq!(
            div.keys_missing_on_peer,
            ["k2".to_string()].into_iter().collect()
        );
        assert!(div.keys_missing_locally.is_empty());
        assert!(div.keys_with_conflicts.is_empty());
    }

    #[tokio::test]
    async fn detects_keys_missing_locally() {
        let ae = CrossRegionAntiEntropy::new("us-east-1");
        let local = populated_tree(&[("k1", "v1")]).await;
        let peer = populated_tree(&[("k1", "v1"), ("k2", "v2")]).await;
        ae.set_local_tree(local).await;
        ae.upsert_peer_tree("eu-west-1".into(), peer)
            .await
            .expect("upsert");
        let div = ae.compare_with(&"eu-west-1".into()).await.expect("compare");
        assert_eq!(
            div.keys_missing_locally,
            ["k2".to_string()].into_iter().collect()
        );
    }

    #[tokio::test]
    async fn detects_conflicting_keys() {
        let ae = CrossRegionAntiEntropy::new("us-east-1");
        let local = populated_tree(&[("k1", "v1")]).await;
        let peer = populated_tree(&[("k1", "v2")]).await;
        ae.set_local_tree(local).await;
        ae.upsert_peer_tree("eu-west-1".into(), peer)
            .await
            .expect("upsert");
        let div = ae.compare_with(&"eu-west-1".into()).await.expect("compare");
        assert_eq!(
            div.keys_with_conflicts,
            ["k1".to_string()].into_iter().collect()
        );
    }

    #[tokio::test]
    async fn compare_all_aggregates_per_peer() {
        let ae = CrossRegionAntiEntropy::new("us-east-1");
        ae.set_local_tree(populated_tree(&[("k1", "v1"), ("k2", "v2")]).await)
            .await;
        ae.upsert_peer_tree(
            "eu-west-1".into(),
            populated_tree(&[("k1", "v1"), ("k2", "v2")]).await,
        )
        .await
        .expect("upsert eu");
        ae.upsert_peer_tree(
            "ap-northeast-1".into(),
            populated_tree(&[("k1", "v1")]).await,
        )
        .await
        .expect("upsert ap");
        let report = ae.compare_all().await.expect("compare_all");
        assert_eq!(report.local_region, "us-east-1");
        assert!(report.per_peer.contains_key("eu-west-1"));
        assert!(report.per_peer.contains_key("ap-northeast-1"));
        assert!(report.per_peer["eu-west-1"].is_in_sync());
        assert!(!report.per_peer["ap-northeast-1"].is_in_sync());
        assert!(!report.is_in_sync());
        let divergent: Vec<_> = report.divergent_peers().cloned().collect();
        assert_eq!(divergent, vec!["ap-northeast-1".to_string()]);
    }

    #[tokio::test]
    async fn registering_local_region_as_peer_fails() {
        let ae = CrossRegionAntiEntropy::new("us-east-1");
        let res = ae
            .upsert_peer_tree("us-east-1".into(), Arc::new(MerkleTree::new()))
            .await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn comparing_unknown_peer_fails() {
        let ae = CrossRegionAntiEntropy::new("us-east-1");
        let res = ae.compare_with(&"mars-1".into()).await;
        assert!(res.is_err());
    }
}
