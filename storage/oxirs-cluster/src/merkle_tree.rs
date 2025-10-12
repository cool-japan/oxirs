//! # Merkle Tree Data Integrity Verification
//!
//! Efficient data integrity verification using Merkle trees for distributed
//! RDF triple storage. Enables fast comparison and synchronization between nodes.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Merkle tree hash type
pub type MerkleHash = [u8; 32];

/// Convert bytes to hex string (for debugging)
#[allow(dead_code)]
fn hash_to_hex(hash: &MerkleHash) -> String {
    hex::encode(hash)
}

/// Merkle tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MerkleNode {
    /// Leaf node containing data hash
    Leaf { hash: MerkleHash, data_key: String },
    /// Internal node with left and right children
    Internal {
        hash: MerkleHash,
        left: Box<MerkleNode>,
        right: Box<MerkleNode>,
    },
}

impl MerkleNode {
    /// Get the hash of this node
    pub fn hash(&self) -> &MerkleHash {
        match self {
            MerkleNode::Leaf { hash, .. } => hash,
            MerkleNode::Internal { hash, .. } => hash,
        }
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        matches!(self, MerkleNode::Leaf { .. })
    }

    /// Get the depth of the tree rooted at this node
    pub fn depth(&self) -> usize {
        match self {
            MerkleNode::Leaf { .. } => 0,
            MerkleNode::Internal { left, right, .. } => {
                1 + std::cmp::max(left.depth(), right.depth())
            }
        }
    }
}

/// Merkle tree for data integrity verification
#[derive(Debug, Clone)]
pub struct MerkleTree {
    root: Arc<RwLock<Option<MerkleNode>>>,
    leaves: Arc<RwLock<BTreeMap<String, MerkleHash>>>,
    stats: Arc<RwLock<MerkleTreeStats>>,
}

/// Merkle tree statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MerkleTreeStats {
    /// Total number of leaves
    pub leaf_count: usize,
    /// Tree depth
    pub depth: usize,
    /// Total verifications performed
    pub total_verifications: u64,
    /// Successful verifications
    pub successful_verifications: u64,
    /// Failed verifications
    pub failed_verifications: u64,
    /// Total tree rebuilds
    pub total_rebuilds: u64,
}

/// Merkle proof for a data item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    /// Data key being proved
    pub data_key: String,
    /// Leaf hash
    pub leaf_hash: MerkleHash,
    /// Path from leaf to root (sibling hash, is_left_sibling)
    pub path: Vec<(MerkleHash, bool)>,
    /// Root hash
    pub root_hash: MerkleHash,
}

impl MerkleTree {
    /// Create a new empty Merkle tree
    pub fn new() -> Self {
        Self {
            root: Arc::new(RwLock::new(None)),
            leaves: Arc::new(RwLock::new(BTreeMap::new())),
            stats: Arc::new(RwLock::new(MerkleTreeStats::default())),
        }
    }

    /// Hash a data item
    fn hash_data(data: &str) -> MerkleHash {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        hasher.finalize().into()
    }

    /// Hash two child hashes together
    fn hash_nodes(left: &MerkleHash, right: &MerkleHash) -> MerkleHash {
        let mut hasher = Sha256::new();
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().into()
    }

    /// Insert a data item
    pub async fn insert(&self, key: String, data: &str) {
        let hash = Self::hash_data(data);

        let mut leaves = self.leaves.write().await;
        leaves.insert(key, hash);

        drop(leaves);

        // Rebuild tree after insertion
        self.rebuild().await;
    }

    /// Remove a data item
    pub async fn remove(&self, key: &str) {
        let mut leaves = self.leaves.write().await;
        leaves.remove(key);

        drop(leaves);

        // Rebuild tree after removal
        self.rebuild().await;
    }

    /// Build Merkle tree from current leaves
    async fn rebuild(&self) {
        let leaves = self.leaves.read().await;

        if leaves.is_empty() {
            *self.root.write().await = None;

            let mut stats = self.stats.write().await;
            stats.leaf_count = 0;
            stats.depth = 0;
            stats.total_rebuilds += 1;

            return;
        }

        // Create leaf nodes sorted by key
        let mut nodes: Vec<MerkleNode> = leaves
            .iter()
            .map(|(key, hash)| MerkleNode::Leaf {
                hash: *hash,
                data_key: key.clone(),
            })
            .collect();

        // Build tree bottom-up
        while nodes.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in nodes.chunks(2) {
                if chunk.len() == 2 {
                    // Combine two nodes
                    let hash = Self::hash_nodes(chunk[0].hash(), chunk[1].hash());
                    next_level.push(MerkleNode::Internal {
                        hash,
                        left: Box::new(chunk[0].clone()),
                        right: Box::new(chunk[1].clone()),
                    });
                } else {
                    // Odd node, promote it
                    next_level.push(chunk[0].clone());
                }
            }

            nodes = next_level;
        }

        let root_node = nodes.into_iter().next();

        let depth = root_node.as_ref().map(|n| n.depth()).unwrap_or(0);

        *self.root.write().await = root_node;

        let mut stats = self.stats.write().await;
        stats.leaf_count = leaves.len();
        stats.depth = depth;
        stats.total_rebuilds += 1;
    }

    /// Get the root hash
    pub async fn root_hash(&self) -> Option<MerkleHash> {
        self.root.read().await.as_ref().map(|node| *node.hash())
    }

    /// Verify data integrity
    pub async fn verify(&self, key: &str, data: &str) -> bool {
        let hash = Self::hash_data(data);

        let leaves = self.leaves.read().await;
        let result = leaves
            .get(key)
            .map(|stored_hash| *stored_hash == hash)
            .unwrap_or(false);

        let mut stats = self.stats.write().await;
        stats.total_verifications += 1;

        if result {
            stats.successful_verifications += 1;
        } else {
            stats.failed_verifications += 1;
        }

        result
    }

    /// Generate a Merkle proof for a data item
    pub async fn generate_proof(&self, key: &str) -> Option<MerkleProof> {
        let leaves = self.leaves.read().await;
        let leaf_hash = *leaves.get(key)?;

        let root = self.root.read().await;
        let root_node = root.as_ref()?;
        let root_hash = *root_node.hash();

        // Find path from leaf to root
        let path = self.find_proof_path(root_node, key);

        Some(MerkleProof {
            data_key: key.to_string(),
            leaf_hash,
            path,
            root_hash,
        })
    }

    /// Find the proof path for a key
    fn find_proof_path(&self, node: &MerkleNode, key: &str) -> Vec<(MerkleHash, bool)> {
        match node {
            MerkleNode::Leaf { data_key, .. } => {
                if data_key == key {
                    Vec::new()
                } else {
                    Vec::new()
                }
            }
            MerkleNode::Internal { left, right, .. } => {
                // Check if key is in left subtree
                if self.contains_key(left, key) {
                    let mut path = self.find_proof_path(left, key);
                    // Sibling (right) is on the right, so is_left_sibling = false
                    path.push((*right.hash(), false));
                    path
                } else {
                    let mut path = self.find_proof_path(right, key);
                    // Sibling (left) is on the left, so is_left_sibling = true
                    path.push((*left.hash(), true));
                    path
                }
            }
        }
    }

    /// Check if a node's subtree contains a key
    fn contains_key(&self, node: &MerkleNode, key: &str) -> bool {
        match node {
            MerkleNode::Leaf { data_key, .. } => data_key == key,
            MerkleNode::Internal { left, right, .. } => {
                self.contains_key(left, key) || self.contains_key(right, key)
            }
        }
    }

    /// Verify a Merkle proof
    pub fn verify_proof(&self, proof: &MerkleProof, data: &str) -> bool {
        let computed_hash = Self::hash_data(data);

        if computed_hash != proof.leaf_hash {
            return false;
        }

        // Recompute root hash from leaf and path
        let mut current_hash = proof.leaf_hash;

        for (sibling_hash, is_left_sibling) in &proof.path {
            current_hash = if *is_left_sibling {
                // Sibling is on the left, current is on the right
                Self::hash_nodes(sibling_hash, &current_hash)
            } else {
                // Sibling is on the right, current is on the left
                Self::hash_nodes(&current_hash, sibling_hash)
            };
        }

        current_hash == proof.root_hash
    }

    /// Compare with another Merkle tree
    pub async fn compare(&self, other: &MerkleTree) -> MerkleComparison {
        let our_root = self.root_hash().await;
        let their_root = other.root_hash().await;

        if our_root == their_root {
            return MerkleComparison::Identical;
        }

        // Find differences
        let our_leaves = self.leaves.read().await;
        let their_leaves = other.leaves.read().await;

        let mut missing_keys = Vec::new();
        let mut extra_keys = Vec::new();
        let mut conflicting_keys = Vec::new();

        // Find keys in our tree but not in theirs
        for key in our_leaves.keys() {
            if !their_leaves.contains_key(key) {
                extra_keys.push(key.clone());
            }
        }

        // Find keys in their tree but not in ours, and conflicts
        for (key, their_hash) in their_leaves.iter() {
            if let Some(our_hash) = our_leaves.get(key) {
                if our_hash != their_hash {
                    conflicting_keys.push(key.clone());
                }
            } else {
                missing_keys.push(key.clone());
            }
        }

        MerkleComparison::Different {
            missing_keys,
            extra_keys,
            conflicting_keys,
        }
    }

    /// Get statistics
    pub async fn get_stats(&self) -> MerkleTreeStats {
        self.stats.read().await.clone()
    }

    /// Get all leaf keys
    pub async fn get_keys(&self) -> Vec<String> {
        self.leaves.read().await.keys().cloned().collect()
    }

    /// Get number of leaves
    pub async fn len(&self) -> usize {
        self.leaves.read().await.len()
    }

    /// Check if tree is empty
    pub async fn is_empty(&self) -> bool {
        self.leaves.read().await.is_empty()
    }

    /// Clear the tree
    pub async fn clear(&self) {
        self.leaves.write().await.clear();
        *self.root.write().await = None;

        let mut stats = self.stats.write().await;
        stats.leaf_count = 0;
        stats.depth = 0;
    }
}

impl Default for MerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of comparing two Merkle trees
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MerkleComparison {
    /// Trees are identical
    Identical,
    /// Trees are different
    Different {
        /// Keys present in other tree but missing from ours
        missing_keys: Vec<String>,
        /// Keys present in our tree but not in theirs
        extra_keys: Vec<String>,
        /// Keys present in both but with different hashes
        conflicting_keys: Vec<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_merkle_tree_creation() {
        let tree = MerkleTree::new();
        assert!(tree.is_empty().await);
        assert_eq!(tree.len().await, 0);
        assert!(tree.root_hash().await.is_none());
    }

    #[tokio::test]
    async fn test_insert_and_verify() {
        let tree = MerkleTree::new();

        tree.insert("key1".to_string(), "value1").await;
        tree.insert("key2".to_string(), "value2").await;

        assert_eq!(tree.len().await, 2);
        assert!(tree.root_hash().await.is_some());

        assert!(tree.verify("key1", "value1").await);
        assert!(tree.verify("key2", "value2").await);
        assert!(!tree.verify("key1", "wrong_value").await);
    }

    #[tokio::test]
    async fn test_remove() {
        let tree = MerkleTree::new();

        tree.insert("key1".to_string(), "value1").await;
        tree.insert("key2".to_string(), "value2").await;

        assert_eq!(tree.len().await, 2);

        tree.remove("key1").await;

        assert_eq!(tree.len().await, 1);
        assert!(!tree.verify("key1", "value1").await);
        assert!(tree.verify("key2", "value2").await);
    }

    #[tokio::test]
    async fn test_root_hash_changes() {
        let tree = MerkleTree::new();

        tree.insert("key1".to_string(), "value1").await;
        let hash1 = tree.root_hash().await;

        tree.insert("key2".to_string(), "value2").await;
        let hash2 = tree.root_hash().await;

        assert_ne!(hash1, hash2);
    }

    #[tokio::test]
    async fn test_merkle_proof() {
        let tree = MerkleTree::new();

        tree.insert("key1".to_string(), "value1").await;
        tree.insert("key2".to_string(), "value2").await;
        tree.insert("key3".to_string(), "value3").await;

        let proof = tree.generate_proof("key2").await;
        assert!(proof.is_some());

        let proof = proof.unwrap();
        assert_eq!(proof.data_key, "key2");

        // Verify the proof
        assert!(tree.verify_proof(&proof, "value2"));
        assert!(!tree.verify_proof(&proof, "wrong_value"));
    }

    #[tokio::test]
    async fn test_compare_identical_trees() {
        let tree1 = MerkleTree::new();
        let tree2 = MerkleTree::new();

        tree1.insert("key1".to_string(), "value1").await;
        tree1.insert("key2".to_string(), "value2").await;

        tree2.insert("key1".to_string(), "value1").await;
        tree2.insert("key2".to_string(), "value2").await;

        let comparison = tree1.compare(&tree2).await;
        assert_eq!(comparison, MerkleComparison::Identical);
    }

    #[tokio::test]
    async fn test_compare_different_trees() {
        let tree1 = MerkleTree::new();
        let tree2 = MerkleTree::new();

        tree1.insert("key1".to_string(), "value1").await;
        tree1.insert("key2".to_string(), "value2").await;

        tree2.insert("key2".to_string(), "value2").await;
        tree2.insert("key3".to_string(), "value3").await;

        let comparison = tree1.compare(&tree2).await;

        match comparison {
            MerkleComparison::Different {
                missing_keys,
                extra_keys,
                conflicting_keys,
            } => {
                assert_eq!(missing_keys, vec!["key3"]);
                assert_eq!(extra_keys, vec!["key1"]);
                assert!(conflicting_keys.is_empty());
            }
            _ => panic!("Expected different trees"),
        }
    }

    #[tokio::test]
    async fn test_compare_conflicting_trees() {
        let tree1 = MerkleTree::new();
        let tree2 = MerkleTree::new();

        tree1.insert("key1".to_string(), "value1").await;
        tree2.insert("key1".to_string(), "different_value").await;

        let comparison = tree1.compare(&tree2).await;

        match comparison {
            MerkleComparison::Different {
                missing_keys,
                extra_keys,
                conflicting_keys,
            } => {
                assert!(missing_keys.is_empty());
                assert!(extra_keys.is_empty());
                assert_eq!(conflicting_keys, vec!["key1"]);
            }
            _ => panic!("Expected different trees"),
        }
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let tree = MerkleTree::new();

        tree.insert("key1".to_string(), "value1").await;
        tree.insert("key2".to_string(), "value2").await;

        tree.verify("key1", "value1").await;
        tree.verify("key2", "wrong_value").await;

        let stats = tree.get_stats().await;
        assert_eq!(stats.leaf_count, 2);
        assert_eq!(stats.total_verifications, 2);
        assert_eq!(stats.successful_verifications, 1);
        assert_eq!(stats.failed_verifications, 1);
        assert!(stats.total_rebuilds > 0);
    }

    #[tokio::test]
    async fn test_clear() {
        let tree = MerkleTree::new();

        tree.insert("key1".to_string(), "value1").await;
        tree.insert("key2".to_string(), "value2").await;

        assert_eq!(tree.len().await, 2);

        tree.clear().await;

        assert_eq!(tree.len().await, 0);
        assert!(tree.is_empty().await);
        assert!(tree.root_hash().await.is_none());
    }

    #[tokio::test]
    async fn test_large_tree() {
        let tree = MerkleTree::new();

        // Insert 100 items
        for i in 0..100 {
            tree.insert(format!("key{}", i), &format!("value{}", i))
                .await;
        }

        assert_eq!(tree.len().await, 100);

        let stats = tree.get_stats().await;
        assert_eq!(stats.leaf_count, 100);
        assert!(stats.depth > 0);

        // Verify all items
        for i in 0..100 {
            assert!(
                tree.verify(&format!("key{}", i), &format!("value{}", i))
                    .await
            );
        }
    }
}
