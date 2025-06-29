//! Immutable storage with content-addressable blocks
//!
//! This module provides an immutable, content-addressable storage system
//! inspired by Git and IPFS, optimized for RDF data integrity and versioning.

use crate::model::{Triple, TriplePattern};
use crate::OxirsError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Immutable storage configuration
#[derive(Debug, Clone)]
pub struct ImmutableConfig {
    /// Base path for immutable storage
    pub path: PathBuf,
    /// Block size in bytes
    pub block_size: usize,
    /// Enable deduplication
    pub deduplication: bool,
    /// Merkle tree depth
    pub merkle_depth: usize,
    /// Garbage collection policy
    pub gc_policy: GarbageCollectionPolicy,
}

impl Default for ImmutableConfig {
    fn default() -> Self {
        ImmutableConfig {
            path: PathBuf::from("/var/oxirs/immutable"),
            block_size: 4096,
            deduplication: true,
            merkle_depth: 4,
            gc_policy: GarbageCollectionPolicy::default(),
        }
    }
}

/// Garbage collection policy
#[derive(Debug, Clone)]
pub struct GarbageCollectionPolicy {
    /// Enable automatic GC
    pub auto_gc: bool,
    /// GC threshold (percentage of unreachable blocks)
    pub threshold: f64,
    /// Minimum age for GC eligibility (hours)
    pub min_age_hours: u32,
}

impl Default for GarbageCollectionPolicy {
    fn default() -> Self {
        GarbageCollectionPolicy {
            auto_gc: true,
            threshold: 0.2,
            min_age_hours: 24,
        }
    }
}

/// Content hash type
pub type ContentHash = [u8; 32];

/// Immutable storage engine
pub struct ImmutableStorage {
    config: ImmutableConfig,
    /// Block store
    blocks: Arc<RwLock<BlockStore>>,
    /// Merkle tree index
    merkle_tree: Arc<RwLock<MerkleTree>>,
    /// Reference tracking
    references: Arc<RwLock<ReferenceTracker>>,
    /// Deduplication index
    dedup_index: Arc<RwLock<DeduplicationIndex>>,
    /// Statistics
    stats: Arc<RwLock<ImmutableStats>>,
}

/// Block store for content-addressable storage
struct BlockStore {
    /// Path to block storage
    path: PathBuf,
    /// In-memory cache of recent blocks
    cache: lru::LruCache<ContentHash, Block>,
    /// Block metadata index
    metadata: HashMap<ContentHash, BlockMetadata>,
}

/// Immutable block
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Block {
    /// Block hash (content address)
    hash: ContentHash,
    /// Block type
    block_type: BlockType,
    /// Block data
    data: Vec<u8>,
    /// References to other blocks
    references: Vec<ContentHash>,
}

/// Block type
#[derive(Debug, Clone, Serialize, Deserialize)]
enum BlockType {
    /// Triple data block
    TripleData,
    /// Index block
    Index,
    /// Merkle tree node
    MerkleNode,
    /// Commit object
    Commit,
    /// Manifest block
    Manifest,
}

/// Block metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlockMetadata {
    /// Creation timestamp
    created_at: chrono::DateTime<chrono::Utc>,
    /// Block size in bytes
    size: usize,
    /// Compression type
    compression: Option<String>,
    /// Reference count
    ref_count: u32,
}

/// Merkle tree for integrity verification
struct MerkleTree {
    /// Root hash
    root: Option<ContentHash>,
    /// Tree nodes
    nodes: HashMap<ContentHash, MerkleNode>,
    /// Depth of tree
    depth: usize,
}

/// Merkle tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MerkleNode {
    /// Node hash
    hash: ContentHash,
    /// Left child hash
    left: Option<ContentHash>,
    /// Right child hash
    right: Option<ContentHash>,
    /// Data hash (for leaf nodes)
    data: Option<ContentHash>,
}

/// Reference tracker for garbage collection
struct ReferenceTracker {
    /// Forward references (block -> referenced blocks)
    forward_refs: HashMap<ContentHash, HashSet<ContentHash>>,
    /// Backward references (block -> blocks that reference it)
    backward_refs: HashMap<ContentHash, HashSet<ContentHash>>,
    /// Root blocks (entry points)
    roots: HashSet<ContentHash>,
}

/// Deduplication index
struct DeduplicationIndex {
    /// Content fingerprint to hash mapping
    fingerprints: HashMap<u64, Vec<ContentHash>>,
    /// Triple to block mapping
    triple_blocks: HashMap<Triple, ContentHash>,
}

/// Immutable storage statistics
#[derive(Debug, Default)]
struct ImmutableStats {
    total_blocks: u64,
    unique_blocks: u64,
    total_size: u64,
    dedup_savings: u64,
    gc_reclaimed: u64,
}

/// Commit object for versioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    /// Commit hash
    pub hash: ContentHash,
    /// Parent commits
    pub parents: Vec<ContentHash>,
    /// Root of data tree
    pub tree: ContentHash,
    /// Author information
    pub author: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Commit message
    pub message: String,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl ImmutableStorage {
    /// Create new immutable storage
    pub async fn new(config: ImmutableConfig) -> Result<Self, OxirsError> {
        std::fs::create_dir_all(&config.path)?;
        std::fs::create_dir_all(config.path.join("blocks"))?;
        std::fs::create_dir_all(config.path.join("refs"))?;

        let cache_size = 1000; // Number of blocks to cache

        Ok(ImmutableStorage {
            config: config.clone(),
            blocks: Arc::new(RwLock::new(BlockStore {
                path: config.path.join("blocks"),
                cache: lru::LruCache::new(
                    std::num::NonZeroUsize::new(cache_size)
                        .unwrap_or(std::num::NonZeroUsize::new(1000).unwrap()),
                ),
                metadata: HashMap::new(),
            })),
            merkle_tree: Arc::new(RwLock::new(MerkleTree {
                root: None,
                nodes: HashMap::new(),
                depth: config.merkle_depth,
            })),
            references: Arc::new(RwLock::new(ReferenceTracker {
                forward_refs: HashMap::new(),
                backward_refs: HashMap::new(),
                roots: HashSet::new(),
            })),
            dedup_index: Arc::new(RwLock::new(DeduplicationIndex {
                fingerprints: HashMap::new(),
                triple_blocks: HashMap::new(),
            })),
            stats: Arc::new(RwLock::new(ImmutableStats::default())),
        })
    }

    /// Store triples as immutable blocks
    pub async fn store_triples(
        &self,
        triples: &[Triple],
        message: &str,
    ) -> Result<Commit, OxirsError> {
        let mut block_hashes = Vec::new();
        let mut blocks_guard = self.blocks.write().await;
        let mut dedup_guard = self.dedup_index.write().await;

        // Process triples in chunks
        for chunk in triples.chunks(100) {
            // Check for deduplication
            if self.config.deduplication {
                let mut unique_triples = Vec::new();
                for triple in chunk {
                    if !dedup_guard.triple_blocks.contains_key(triple) {
                        unique_triples.push(triple.clone());
                    }
                }

                if !unique_triples.is_empty() {
                    let block = self.create_triple_block(&unique_triples)?;
                    let hash = block.hash;

                    // Update deduplication index
                    for triple in unique_triples {
                        dedup_guard.triple_blocks.insert(triple, hash);
                    }

                    // Store block
                    self.store_block(&mut blocks_guard, block).await?;
                    block_hashes.push(hash);
                }
            } else {
                let block = self.create_triple_block(chunk)?;
                let hash = block.hash;
                self.store_block(&mut blocks_guard, block).await?;
                block_hashes.push(hash);
            }
        }

        // Create index blocks if needed
        let index_blocks = self.create_index_blocks(&block_hashes)?;
        for block in index_blocks {
            self.store_block(&mut blocks_guard, block).await?;
        }

        // Build Merkle tree
        let tree_root = self.build_merkle_tree(&block_hashes).await?;

        // Create commit
        let commit = Commit {
            hash: self.compute_commit_hash(&tree_root, message),
            parents: vec![], // Would get from current HEAD
            tree: tree_root,
            author: "system".to_string(),
            timestamp: chrono::Utc::now(),
            message: message.to_string(),
            metadata: HashMap::new(),
        };

        // Store commit block
        let commit_block = Block {
            hash: commit.hash,
            block_type: BlockType::Commit,
            data: bincode::serialize(&commit)?,
            references: vec![tree_root],
        };
        self.store_block(&mut blocks_guard, commit_block).await?;

        // Update references
        self.update_references(&commit).await?;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_blocks += block_hashes.len() as u64 + 1;
        stats.unique_blocks += block_hashes.len() as u64 + 1;

        Ok(commit)
    }

    /// Read triples from a commit
    pub async fn read_commit(&self, commit_hash: ContentHash) -> Result<Vec<Triple>, OxirsError> {
        let blocks = self.blocks.read().await;

        // Load commit block
        let commit_block = self.load_block(&blocks, &commit_hash).await?;
        let commit: Commit = bincode::deserialize(&commit_block.data)?;

        // Traverse tree to find triple blocks
        let triple_blocks = self.find_triple_blocks(&commit.tree).await?;

        // Load and decode triples
        let mut all_triples = Vec::new();
        for block_hash in triple_blocks {
            let block = self.load_block(&blocks, &block_hash).await?;
            let triples: Vec<Triple> = bincode::deserialize(&block.data)?;
            all_triples.extend(triples);
        }

        Ok(all_triples)
    }

    /// Query triples with pattern matching
    pub async fn query_triples(
        &self,
        pattern: &TriplePattern,
        commit_hash: Option<ContentHash>,
    ) -> Result<Vec<Triple>, OxirsError> {
        // If commit specified, query from that version
        let commit = if let Some(hash) = commit_hash {
            hash
        } else {
            // Get latest commit (HEAD)
            self.get_head().await?
        };

        let all_triples = self.read_commit(commit).await?;

        // Filter by pattern
        Ok(all_triples
            .into_iter()
            .filter(|triple| pattern.matches(triple))
            .collect())
    }

    /// Verify integrity of storage
    pub async fn verify_integrity(&self) -> Result<IntegrityReport, OxirsError> {
        let blocks = self.blocks.read().await;
        let merkle = self.merkle_tree.read().await;

        let mut report = IntegrityReport {
            total_blocks: 0,
            verified_blocks: 0,
            corrupted_blocks: Vec::new(),
            missing_blocks: Vec::new(),
            merkle_valid: true,
        };

        // Verify all blocks
        for (hash, metadata) in &blocks.metadata {
            report.total_blocks += 1;

            if let Ok(block) = self.load_block(&blocks, hash).await {
                // Verify hash
                if self.compute_hash(&block.data) == *hash {
                    report.verified_blocks += 1;
                } else {
                    report.corrupted_blocks.push(*hash);
                }
            } else {
                report.missing_blocks.push(*hash);
            }
        }

        // Verify Merkle tree
        if let Some(root) = merkle.root {
            report.merkle_valid = self.verify_merkle_tree(&merkle, root).await?;
        }

        Ok(report)
    }

    /// Run garbage collection
    pub async fn garbage_collect(&self) -> Result<GCReport, OxirsError> {
        let mut refs = self.references.write().await;
        let mut blocks = self.blocks.write().await;

        let mut report = GCReport {
            total_blocks: blocks.metadata.len(),
            reachable_blocks: 0,
            collected_blocks: 0,
            reclaimed_bytes: 0,
        };

        // Mark phase - find all reachable blocks
        let mut reachable = HashSet::new();
        let mut to_visit: Vec<_> = refs.roots.iter().cloned().collect();

        while let Some(hash) = to_visit.pop() {
            if reachable.insert(hash) {
                if let Some(children) = refs.forward_refs.get(&hash) {
                    to_visit.extend(children.iter().cloned());
                }
            }
        }

        report.reachable_blocks = reachable.len();

        // Sweep phase - remove unreachable blocks
        let unreachable: Vec<_> = blocks
            .metadata
            .keys()
            .filter(|hash| !reachable.contains(*hash))
            .cloned()
            .collect();

        for hash in unreachable {
            if let Some(metadata) = blocks.metadata.remove(&hash) {
                report.collected_blocks += 1;
                report.reclaimed_bytes += metadata.size;

                // Remove from disk
                let block_path = blocks.path.join(hex::encode(hash));
                let _ = std::fs::remove_file(block_path);

                // Remove from cache
                blocks.cache.pop(&hash);

                // Remove references
                refs.forward_refs.remove(&hash);
                for (_, back_refs) in refs.backward_refs.iter_mut() {
                    back_refs.remove(&hash);
                }
            }
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.gc_reclaimed += report.reclaimed_bytes as u64;

        Ok(report)
    }

    /// Create a triple block
    fn create_triple_block(&self, triples: &[Triple]) -> Result<Block, OxirsError> {
        let data = bincode::serialize(triples)?;
        let hash = self.compute_hash(&data);

        Ok(Block {
            hash,
            block_type: BlockType::TripleData,
            data,
            references: Vec::new(),
        })
    }

    /// Create index blocks for efficient querying
    fn create_index_blocks(&self, data_blocks: &[ContentHash]) -> Result<Vec<Block>, OxirsError> {
        // Simplified - would create actual index structures
        Ok(Vec::new())
    }

    /// Build Merkle tree from blocks
    async fn build_merkle_tree(&self, blocks: &[ContentHash]) -> Result<ContentHash, OxirsError> {
        let mut merkle = self.merkle_tree.write().await;

        // Build tree bottom-up
        let mut current_level = blocks.to_vec();

        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in current_level.chunks(2) {
                let left = chunk[0];
                let right = chunk.get(1).cloned().unwrap_or(left);

                let combined = [left.as_slice(), right.as_slice()].concat();
                let parent_hash = self.compute_hash(&combined);

                let node = MerkleNode {
                    hash: parent_hash,
                    left: Some(left),
                    right: Some(right),
                    data: None,
                };

                merkle.nodes.insert(parent_hash, node);
                next_level.push(parent_hash);
            }

            current_level = next_level;
        }

        let root = current_level[0];
        merkle.root = Some(root);
        Ok(root)
    }

    /// Store a block
    async fn store_block(&self, blocks: &mut BlockStore, block: Block) -> Result<(), OxirsError> {
        let hash = block.hash;
        let size = block.data.len();

        // Write to disk
        let block_path = blocks.path.join(hex::encode(hash));
        let compressed = self.compress_block(&block)?;
        std::fs::write(block_path, compressed)?;

        // Update metadata
        blocks.metadata.insert(
            hash,
            BlockMetadata {
                created_at: chrono::Utc::now(),
                size,
                compression: Some("zstd".to_string()),
                ref_count: 0,
            },
        );

        // Add to cache
        blocks.cache.put(hash, block);

        Ok(())
    }

    /// Load a block
    async fn load_block(
        &self,
        blocks: &BlockStore,
        hash: &ContentHash,
    ) -> Result<Block, OxirsError> {
        // Check cache first
        if let Some(block) = blocks.cache.peek(hash) {
            return Ok(block.clone());
        }

        // Load from disk
        let block_path = blocks.path.join(hex::encode(hash));
        let compressed = std::fs::read(block_path)?;
        let block = self.decompress_block(&compressed)?;

        Ok(block)
    }

    /// Find all triple blocks under a tree
    async fn find_triple_blocks(
        &self,
        tree_hash: &ContentHash,
    ) -> Result<Vec<ContentHash>, OxirsError> {
        // Simplified - would traverse tree structure
        Ok(vec![*tree_hash])
    }

    /// Update reference tracking
    async fn update_references(&self, commit: &Commit) -> Result<(), OxirsError> {
        let mut refs = self.references.write().await;

        // Add commit as root
        refs.roots.insert(commit.hash);

        // Add forward reference
        refs.forward_refs
            .entry(commit.hash)
            .or_insert_with(HashSet::new)
            .insert(commit.tree);

        // Add backward reference
        refs.backward_refs
            .entry(commit.tree)
            .or_insert_with(HashSet::new)
            .insert(commit.hash);

        Ok(())
    }

    /// Get current HEAD commit
    async fn get_head(&self) -> Result<ContentHash, OxirsError> {
        // Simplified - would read from refs/heads/main
        let refs = self.references.read().await;
        refs.roots
            .iter()
            .next()
            .cloned()
            .ok_or_else(|| OxirsError::Store("No commits found".to_string()))
    }

    /// Verify Merkle tree integrity
    async fn verify_merkle_tree(
        &self,
        merkle: &MerkleTree,
        root: ContentHash,
    ) -> Result<bool, OxirsError> {
        // Simplified verification
        Ok(merkle.nodes.contains_key(&root))
    }

    /// Compute SHA256 hash
    fn compute_hash(&self, data: &[u8]) -> ContentHash {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }

    /// Compute commit hash
    fn compute_commit_hash(&self, tree: &ContentHash, message: &str) -> ContentHash {
        let mut hasher = Sha256::new();
        hasher.update(tree);
        hasher.update(message.as_bytes());
        hasher.finalize().into()
    }

    /// Compress block data
    fn compress_block(&self, block: &Block) -> Result<Vec<u8>, OxirsError> {
        let serialized = bincode::serialize(block)?;
        Ok(zstd::encode_all(&serialized[..], 3)?)
    }

    /// Decompress block data
    fn decompress_block(&self, data: &[u8]) -> Result<Block, OxirsError> {
        let decompressed = zstd::decode_all(data)?;
        Ok(bincode::deserialize(&decompressed)?)
    }
}

/// Integrity verification report
#[derive(Debug)]
pub struct IntegrityReport {
    pub total_blocks: usize,
    pub verified_blocks: usize,
    pub corrupted_blocks: Vec<ContentHash>,
    pub missing_blocks: Vec<ContentHash>,
    pub merkle_valid: bool,
}

/// Garbage collection report
#[derive(Debug)]
pub struct GCReport {
    pub total_blocks: usize,
    pub reachable_blocks: usize,
    pub collected_blocks: usize,
    pub reclaimed_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};

    #[tokio::test]
    async fn test_immutable_storage() {
        let config = ImmutableConfig {
            path: PathBuf::from("/tmp/oxirs_immutable_test"),
            ..Default::default()
        };

        let storage = ImmutableStorage::new(config).await.unwrap();

        // Create test triples
        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/s1").unwrap(),
                NamedNode::new("http://example.org/p1").unwrap(),
                crate::model::Object::NamedNode(NamedNode::new("http://example.org/o1").unwrap()),
            ),
            Triple::new(
                NamedNode::new("http://example.org/s2").unwrap(),
                NamedNode::new("http://example.org/p2").unwrap(),
                crate::model::Object::Literal(crate::model::Literal::new("test")),
            ),
        ];

        // Store triples
        let commit = storage
            .store_triples(&triples, "Initial commit")
            .await
            .unwrap();

        // Read back triples
        let loaded = storage.read_commit(commit.hash).await.unwrap();
        assert_eq!(loaded.len(), 2);

        // Query with pattern
        let pattern = TriplePattern::new(
            Some(crate::model::SubjectPattern::NamedNode(
                NamedNode::new("http://example.org/s1").unwrap(),
            )),
            None,
            None,
        );
        let results = storage
            .query_triples(&pattern, Some(commit.hash))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_deduplication() {
        let config = ImmutableConfig {
            path: PathBuf::from("/tmp/oxirs_immutable_dedup"),
            deduplication: true,
            ..Default::default()
        };

        let storage = ImmutableStorage::new(config).await.unwrap();

        // Store same triple multiple times
        let triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            crate::model::Object::Literal(crate::model::Literal::new("value")),
        );

        let triples = vec![triple.clone(), triple.clone(), triple.clone()];

        // Should deduplicate
        let commit = storage.store_triples(&triples, "Dedup test").await.unwrap();

        let stats = storage.stats.read().await;
        // Should only store one unique block for the triples
        assert!(stats.unique_blocks < 3);
    }
}
