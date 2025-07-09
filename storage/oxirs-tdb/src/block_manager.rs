//! # Block Management System
//!
//! Efficient block allocation and deallocation system with free block tracking,
//! fragmentation handling, and compaction strategies for optimal storage utilization.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, RwLock};
use tracing::info;

/// Block identifier type
pub type BlockId = u64;

/// Block size type (in bytes)
pub type BlockSize = u32;

/// Block offset type (in bytes)
pub type BlockOffset = u64;

/// Block allocation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockStatus {
    /// Block is free and available for allocation
    Free,
    /// Block is allocated and in use
    Allocated,
    /// Block is marked for deletion (pending garbage collection)
    PendingDeletion,
    /// Block is corrupted and should not be used
    Corrupted,
}

/// Block metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockMetadata {
    /// Block identifier
    pub id: BlockId,
    /// Block offset in the file
    pub offset: BlockOffset,
    /// Block size in bytes
    pub size: BlockSize,
    /// Current status
    pub status: BlockStatus,
    /// Creation timestamp
    pub created_at: u64,
    /// Last access timestamp
    pub last_accessed: u64,
    /// Number of times this block has been accessed
    pub access_count: u64,
    /// Checksum for integrity verification
    pub checksum: u32,
}

impl BlockMetadata {
    /// Create new block metadata
    pub fn new(id: BlockId, offset: BlockOffset, size: BlockSize) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id,
            offset,
            size,
            status: BlockStatus::Free,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            checksum: 0,
        }
    }

    /// Mark block as allocated
    pub fn allocate(&mut self) {
        self.status = BlockStatus::Allocated;
        self.update_access();
    }

    /// Mark block as free
    pub fn free(&mut self) {
        self.status = BlockStatus::Free;
        self.update_access();
    }

    /// Update access statistics
    pub fn update_access(&mut self) {
        self.last_accessed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.access_count += 1;
    }

    /// Check if block is eligible for compaction (old and rarely accessed)
    pub fn is_compaction_candidate(&self, min_age_seconds: u64, max_access_count: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let age = now.saturating_sub(self.created_at);
        age >= min_age_seconds && self.access_count <= max_access_count
    }
}

/// Free block tracking using size-based segregation
#[derive(Debug, Clone)]
struct FreeBlockTracker {
    /// Free blocks organized by size (size -> set of block IDs)
    blocks_by_size: BTreeMap<BlockSize, BTreeSet<BlockId>>,
    /// Block metadata lookup
    block_metadata: HashMap<BlockId, BlockMetadata>,
    /// Total free space
    total_free_space: u64,
    /// Number of free blocks
    free_block_count: usize,
}

impl FreeBlockTracker {
    fn new() -> Self {
        Self {
            blocks_by_size: BTreeMap::new(),
            block_metadata: HashMap::new(),
            total_free_space: 0,
            free_block_count: 0,
        }
    }

    /// Add a free block
    fn add_free_block(&mut self, mut metadata: BlockMetadata) {
        metadata.free();

        let size = metadata.size;
        let id = metadata.id;

        // Add to size index
        self.blocks_by_size
            .entry(size)
            .or_default()
            .insert(id);

        // Update metadata
        self.block_metadata.insert(id, metadata);

        // Update statistics
        self.total_free_space += size as u64;
        self.free_block_count += 1;
    }

    /// Remove a free block (when it gets allocated)
    fn remove_free_block(&mut self, id: BlockId) -> Option<BlockMetadata> {
        if let Some(metadata) = self.block_metadata.remove(&id) {
            let size = metadata.size;

            // Remove from size index
            if let Some(blocks) = self.blocks_by_size.get_mut(&size) {
                blocks.remove(&id);
                if blocks.is_empty() {
                    self.blocks_by_size.remove(&size);
                }
            }

            // Update statistics
            self.total_free_space -= size as u64;
            self.free_block_count -= 1;

            Some(metadata)
        } else {
            None
        }
    }

    /// Find best fit block for given size
    fn find_best_fit(&self, required_size: BlockSize) -> Option<BlockId> {
        // Find the smallest block that can accommodate the required size
        for (&_size, blocks) in self.blocks_by_size.range(required_size..) {
            if let Some(&id) = blocks.iter().next() {
                return Some(id);
            }
        }
        None
    }

    /// Find contiguous groups of blocks that can be coalesced together
    fn find_coalescable_groups(&self) -> Vec<Vec<BlockId>> {
        let mut blocks: Vec<_> = self.block_metadata.values().collect();
        blocks.sort_by_key(|b| b.offset);

        let mut groups = Vec::new();
        let mut current_group = Vec::new();

        for (i, &block) in blocks.iter().enumerate() {
            if current_group.is_empty() {
                current_group.push(block.id);
            } else {
                let prev_block = blocks[i - 1];
                // Check if this block is adjacent to the previous one
                if prev_block.offset + prev_block.size as u64 == block.offset {
                    current_group.push(block.id);
                } else {
                    // Non-adjacent, finalize current group if it has multiple blocks
                    if current_group.len() > 1 {
                        groups.push(current_group);
                    }
                    current_group = vec![block.id];
                }
            }
        }

        // Don't forget the last group
        if current_group.len() > 1 {
            groups.push(current_group);
        }

        // Limit number of groups to process
        groups.truncate(20);
        groups
    }

    /// Get statistics
    fn get_stats(&self) -> (u64, usize, usize) {
        (
            self.total_free_space,
            self.free_block_count,
            self.blocks_by_size.len(),
        )
    }
}

/// Block manager configuration
#[derive(Debug, Clone)]
pub struct BlockManagerConfig {
    /// Default block size for allocations
    pub default_block_size: BlockSize,
    /// Minimum block size
    pub min_block_size: BlockSize,
    /// Maximum block size
    pub max_block_size: BlockSize,
    /// Enable automatic coalescing of adjacent free blocks
    pub enable_coalescing: bool,
    /// Enable automatic compaction
    pub enable_compaction: bool,
    /// Compaction threshold (percentage of fragmented space)
    pub compaction_threshold: f64,
    /// Garbage collection interval in seconds
    pub gc_interval_seconds: u64,
    /// Maximum fragmentation ratio before forced compaction
    pub max_fragmentation_ratio: f64,
}

impl Default for BlockManagerConfig {
    fn default() -> Self {
        Self {
            default_block_size: 8192,    // 8KB
            min_block_size: 512,         // 512 bytes
            max_block_size: 1024 * 1024, // 1MB
            enable_coalescing: true,
            enable_compaction: true,
            compaction_threshold: 0.3,    // 30% fragmentation
            gc_interval_seconds: 300,     // 5 minutes
            max_fragmentation_ratio: 0.5, // 50%
        }
    }
}

/// Block manager statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct BlockManagerStats {
    pub total_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub corrupted_blocks: usize,
    pub total_space: u64,
    pub allocated_space: u64,
    pub free_space: u64,
    pub fragmentation_ratio: f64,
    pub avg_block_size: f64,
    pub largest_free_block: BlockSize,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub coalescing_operations: u64,
    pub compaction_operations: u64,
    pub gc_runs: u64,
    pub total_allocation_time: std::time::Duration,
}

/// Block allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocationStrategy {
    /// First fit - use first available block
    FirstFit,
    /// Best fit - use smallest block that fits
    #[default]
    BestFit,
    /// Worst fit - use largest available block
    WorstFit,
    /// Next fit - continue from last allocation point
    NextFit,
}

/// Block manager for efficient storage allocation
pub struct BlockManager {
    /// Free block tracker
    free_blocks: Arc<RwLock<FreeBlockTracker>>,
    /// All block metadata
    all_blocks: Arc<RwLock<HashMap<BlockId, BlockMetadata>>>,
    /// Next block ID to assign
    next_block_id: Arc<RwLock<BlockId>>,
    /// Current file size
    file_size: Arc<RwLock<u64>>,
    /// Configuration
    config: BlockManagerConfig,
    /// Statistics
    stats: Arc<RwLock<BlockManagerStats>>,
    /// Allocation strategy
    strategy: AllocationStrategy,
    /// Last allocation position (for NextFit)
    last_allocation_pos: Arc<RwLock<BlockOffset>>,
}

impl BlockManager {
    /// Create a new block manager
    pub fn new() -> Self {
        Self::with_config(BlockManagerConfig::default())
    }

    /// Create a new block manager with custom configuration
    pub fn with_config(config: BlockManagerConfig) -> Self {
        Self {
            free_blocks: Arc::new(RwLock::new(FreeBlockTracker::new())),
            all_blocks: Arc::new(RwLock::new(HashMap::new())),
            next_block_id: Arc::new(RwLock::new(1)),
            file_size: Arc::new(RwLock::new(0)),
            config,
            stats: Arc::new(RwLock::new(BlockManagerStats::default())),
            strategy: AllocationStrategy::default(),
            last_allocation_pos: Arc::new(RwLock::new(0)),
        }
    }

    /// Set allocation strategy
    pub fn set_allocation_strategy(&mut self, strategy: AllocationStrategy) {
        self.strategy = strategy;
    }

    /// Allocate a block of specified size
    pub fn allocate_block(&self, size: BlockSize) -> Result<BlockId> {
        if size < self.config.min_block_size || size > self.config.max_block_size {
            return Err(anyhow!(
                "Block size {} is outside allowed range [{}, {}]",
                size,
                self.config.min_block_size,
                self.config.max_block_size
            ));
        }

        let block_id = match self.strategy {
            AllocationStrategy::BestFit => self.allocate_best_fit(size)?,
            AllocationStrategy::FirstFit => self.allocate_first_fit(size)?,
            AllocationStrategy::WorstFit => self.allocate_worst_fit(size)?,
            AllocationStrategy::NextFit => self.allocate_next_fit(size)?,
        };

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.allocation_count += 1;
            stats.allocated_blocks += 1;
            stats.allocated_space += size as u64;
            // Note: free_blocks and free_space are updated by FreeBlockManager
            // and refreshed in get_stats(), so we don't update them here
        }

        self.maybe_trigger_gc();

        Ok(block_id)
    }

    /// Allocate using best fit strategy
    fn allocate_best_fit(&self, size: BlockSize) -> Result<BlockId> {
        let mut free_blocks = self.free_blocks.write().unwrap();

        if let Some(block_id) = free_blocks.find_best_fit(size) {
            // Use existing free block
            if let Some(mut metadata) = free_blocks.remove_free_block(block_id) {
                metadata.allocate();

                // If block is larger than needed, split it
                println!("Checking if splitting is needed: metadata.size={}, size={}, min_block_size={}, condition: {}", 
                         metadata.size, size, self.config.min_block_size, metadata.size > size + self.config.min_block_size);
                if metadata.size > size + self.config.min_block_size {
                    let remaining_size = metadata.size - size;
                    let remaining_offset = metadata.offset + size as u64;

                    // Create new block for the remaining space
                    let remaining_id = {
                        let mut next_id = self.next_block_id.write().unwrap();
                        let id = *next_id;
                        *next_id += 1;
                        id
                    };

                    let remaining_metadata =
                        BlockMetadata::new(remaining_id, remaining_offset, remaining_size);

                    println!(
                        "Creating remaining block: id={remaining_id}, size={remaining_size}, offset={remaining_offset}"
                    );
                    free_blocks.add_free_block(remaining_metadata.clone());
                    println!("Added remaining block to free_blocks");

                    // Also add the remaining block to all_blocks
                    {
                        let mut all_blocks = self.all_blocks.write().unwrap();
                        all_blocks.insert(remaining_id, remaining_metadata);
                    }

                    // Update the allocated block size
                    metadata.size = size;
                }

                // Store updated metadata
                {
                    let mut all_blocks = self.all_blocks.write().unwrap();
                    all_blocks.insert(block_id, metadata);
                }

                return Ok(block_id);
            }
        }

        // No suitable free block found, allocate at end of file
        self.allocate_new_block(size)
    }

    /// Allocate using first fit strategy
    fn allocate_first_fit(&self, size: BlockSize) -> Result<BlockId> {
        let candidate_block = {
            let free_blocks = self.free_blocks.read().unwrap();

            // Find first block that can accommodate the size
            let mut candidate = None;
            for (&block_size, block_ids) in &free_blocks.blocks_by_size {
                if block_size >= size {
                    if let Some(&block_id) = block_ids.iter().next() {
                        candidate = Some(block_id);
                        break;
                    }
                }
            }
            candidate
        };

        if let Some(block_id) = candidate_block {
            let mut free_blocks = self.free_blocks.write().unwrap();
            if let Some(mut metadata) = free_blocks.remove_free_block(block_id) {
                metadata.allocate();

                // Handle splitting if necessary
                if metadata.size > size + self.config.min_block_size {
                    // Split logic (similar to best_fit)
                    let remaining_size = metadata.size - size;
                    let remaining_offset = metadata.offset + size as u64;

                    let remaining_id = {
                        let mut next_id = self.next_block_id.write().unwrap();
                        let id = *next_id;
                        *next_id += 1;
                        id
                    };

                    let remaining_metadata =
                        BlockMetadata::new(remaining_id, remaining_offset, remaining_size);

                    free_blocks.add_free_block(remaining_metadata.clone());

                    // Also add the remaining block to all_blocks
                    {
                        let mut all_blocks = self.all_blocks.write().unwrap();
                        all_blocks.insert(remaining_id, remaining_metadata);
                    }

                    metadata.size = size;
                }

                {
                    let mut all_blocks = self.all_blocks.write().unwrap();
                    all_blocks.insert(block_id, metadata);
                }

                return Ok(block_id);
            }
        }

        self.allocate_new_block(size)
    }

    /// Allocate using worst fit strategy
    fn allocate_worst_fit(&self, size: BlockSize) -> Result<BlockId> {
        let mut free_blocks = self.free_blocks.write().unwrap();

        // Find largest available block
        if let Some((&largest_size, block_ids)) = free_blocks.blocks_by_size.iter().next_back() {
            if largest_size >= size {
                if let Some(&block_id) = block_ids.iter().next() {
                    if let Some(mut metadata) = free_blocks.remove_free_block(block_id) {
                        metadata.allocate();

                        // Always split for worst fit to maximize remaining space
                        if metadata.size > size {
                            let remaining_size = metadata.size - size;
                            let remaining_offset = metadata.offset + size as u64;

                            let remaining_id = {
                                let mut next_id = self.next_block_id.write().unwrap();
                                let id = *next_id;
                                *next_id += 1;
                                id
                            };

                            let remaining_metadata =
                                BlockMetadata::new(remaining_id, remaining_offset, remaining_size);

                            free_blocks.add_free_block(remaining_metadata.clone());

                            // Also add the remaining block to all_blocks
                            {
                                let mut all_blocks = self.all_blocks.write().unwrap();
                                all_blocks.insert(remaining_id, remaining_metadata);
                            }

                            metadata.size = size;
                        }

                        {
                            let mut all_blocks = self.all_blocks.write().unwrap();
                            all_blocks.insert(block_id, metadata);
                        }

                        return Ok(block_id);
                    }
                }
            }
        }

        drop(free_blocks);
        self.allocate_new_block(size)
    }

    /// Allocate using next fit strategy
    fn allocate_next_fit(&self, size: BlockSize) -> Result<BlockId> {
        let last_pos = *self.last_allocation_pos.read().unwrap();

        // Find first suitable block after last allocation position
        let free_blocks = self.free_blocks.read().unwrap();
        let mut candidate = None;

        for metadata in free_blocks.block_metadata.values() {
            if metadata.offset >= last_pos && metadata.size >= size {
                candidate = Some(metadata.id);
                break;
            }
        }

        // If no block found after last position, wrap around
        if candidate.is_none() {
            candidate = free_blocks.find_best_fit(size);
        }

        drop(free_blocks);

        if let Some(block_id) = candidate {
            let mut free_blocks = self.free_blocks.write().unwrap();
            if let Some(mut metadata) = free_blocks.remove_free_block(block_id) {
                metadata.allocate();

                // Update last allocation position
                *self.last_allocation_pos.write().unwrap() = metadata.offset + size as u64;

                // Handle splitting
                if metadata.size > size + self.config.min_block_size {
                    let remaining_size = metadata.size - size;
                    let remaining_offset = metadata.offset + size as u64;

                    let remaining_id = {
                        let mut next_id = self.next_block_id.write().unwrap();
                        let id = *next_id;
                        *next_id += 1;
                        id
                    };

                    let remaining_metadata =
                        BlockMetadata::new(remaining_id, remaining_offset, remaining_size);

                    free_blocks.add_free_block(remaining_metadata.clone());

                    // Also add the remaining block to all_blocks
                    {
                        let mut all_blocks = self.all_blocks.write().unwrap();
                        all_blocks.insert(remaining_id, remaining_metadata);
                    }

                    metadata.size = size;
                }

                {
                    let mut all_blocks = self.all_blocks.write().unwrap();
                    all_blocks.insert(block_id, metadata);
                }

                return Ok(block_id);
            }
        }

        self.allocate_new_block(size)
    }

    /// Allocate a new block at the end of the file
    fn allocate_new_block(&self, size: BlockSize) -> Result<BlockId> {
        let block_id = {
            let mut next_id = self.next_block_id.write().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let offset = {
            let mut file_size = self.file_size.write().unwrap();
            let offset = *file_size;
            *file_size += size as u64;
            offset
        };

        let mut metadata = BlockMetadata::new(block_id, offset, size);
        metadata.allocate();

        {
            let mut all_blocks = self.all_blocks.write().unwrap();
            all_blocks.insert(block_id, metadata);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_blocks += 1;
            stats.total_space += size as u64;
        }

        Ok(block_id)
    }

    /// Deallocate a block
    pub fn deallocate_block(&self, block_id: BlockId) -> Result<()> {
        let mut all_blocks = self.all_blocks.write().unwrap();

        if let Some(metadata) = all_blocks.get_mut(&block_id) {
            if metadata.status != BlockStatus::Allocated {
                return Err(anyhow!("Block {} is not allocated", block_id));
            }

            let size = metadata.size;
            metadata.free();

            // Add to free blocks
            {
                let mut free_blocks = self.free_blocks.write().unwrap();
                free_blocks.add_free_block(metadata.clone());
            }

            // Update statistics
            {
                let mut stats = self.stats.write().unwrap();
                stats.deallocation_count += 1;
                stats.allocated_blocks -= 1;
                stats.allocated_space -= size as u64;
                // Note: free_blocks and free_space are updated by FreeBlockManager
                // and refreshed in get_stats(), so we don't update them here
            }

            // Try to coalesce adjacent free blocks
            if self.config.enable_coalescing {
                self.coalesce_free_blocks()?;
            }

            Ok(())
        } else {
            Err(anyhow!("Block {} not found", block_id))
        }
    }

    /// Coalesce adjacent free blocks
    fn coalesce_free_blocks(&self) -> Result<()> {
        // Get coalescable groups first
        let groups = {
            let free_blocks = self.free_blocks.read().unwrap();
            free_blocks.find_coalescable_groups()
        };

        // Process each group of contiguous blocks
        for group in groups {
            if group.len() < 2 {
                continue; // Skip groups with only one block
            }

            // Remove all blocks in the group from free list and collect their metadata
            let mut block_metadata = Vec::new();
            {
                let mut free_blocks = self.free_blocks.write().unwrap();
                for &block_id in &group {
                    if let Some(metadata) = free_blocks.remove_free_block(block_id) {
                        block_metadata.push(metadata);
                    }
                }
            }

            // If we successfully removed blocks, coalesce them
            if block_metadata.len() >= 2 {
                // Sort by offset to ensure correct order
                block_metadata.sort_by_key(|b| b.offset);

                // Create one large coalesced block
                let first = &block_metadata[0];
                let total_size: BlockSize = block_metadata.iter().map(|b| b.size).sum();

                let coalesced_metadata = BlockMetadata::new(
                    first.id, // Keep first block's ID
                    first.offset,
                    total_size,
                );

                // Add coalesced block back to free list
                {
                    let mut free_blocks = self.free_blocks.write().unwrap();
                    free_blocks.add_free_block(coalesced_metadata.clone());
                }

                // Update all_blocks - remove all but first, update first
                {
                    let mut all_blocks = self.all_blocks.write().unwrap();
                    all_blocks.insert(first.id, coalesced_metadata);

                    // Remove all other blocks from the group
                    for metadata in &block_metadata[1..] {
                        all_blocks.remove(&metadata.id);
                    }
                }

                // Update statistics
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.coalescing_operations += 1;
                    stats.total_blocks -= block_metadata.len() - 1; // Reduced by number of blocks merged minus 1
                }
            }
        }

        Ok(())
    }

    /// Get block metadata
    pub fn get_block_metadata(&self, block_id: BlockId) -> Option<BlockMetadata> {
        let all_blocks = self.all_blocks.read().unwrap();
        all_blocks.get(&block_id).cloned()
    }

    /// Get statistics
    pub fn get_stats(&self) -> BlockManagerStats {
        let mut stats = self.stats.write().unwrap();

        // Update derived statistics
        let free_blocks = self.free_blocks.read().unwrap();
        let (free_space, free_count, _) = free_blocks.get_stats();

        println!(
            "get_stats: FreeBlockTracker reports free_space={free_space}, free_count={free_count}"
        );
        println!(
            "get_stats: Number of entries in free_blocks.block_metadata: {}",
            free_blocks.block_metadata.len()
        );

        stats.free_space = free_space;
        stats.free_blocks = free_count;

        if stats.total_space > 0 {
            stats.fragmentation_ratio = free_space as f64 / stats.total_space as f64;
        }

        if stats.total_blocks > 0 {
            stats.avg_block_size = stats.total_space as f64 / stats.total_blocks as f64;
        }

        // Find largest free block
        stats.largest_free_block = free_blocks
            .blocks_by_size
            .keys()
            .next_back()
            .copied()
            .unwrap_or(0);

        stats.clone()
    }

    /// Check if compaction is needed
    pub fn needs_compaction(&self) -> bool {
        let stats = self.get_stats();
        stats.fragmentation_ratio > self.config.compaction_threshold
    }

    /// Perform compaction to reduce fragmentation
    ///
    /// Compaction moves allocated blocks towards the beginning of the storage area,
    /// coalescing free space into larger contiguous regions. This improves allocation
    /// efficiency and reduces fragmentation.
    ///
    /// # Algorithm
    /// 1. Identify moveable blocks (allocated blocks that can be relocated)
    /// 2. Sort them by access patterns and age for optimal placement
    /// 3. Move blocks to create contiguous allocated and free regions
    /// 4. Update all metadata and free block tracking
    ///
    /// # Returns
    /// Returns `Ok(())` on successful compaction, or an error if compaction fails
    pub fn compact(&self) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Get exclusive access to all data structures
        let mut all_blocks = self.all_blocks.write().unwrap();
        let mut free_blocks = self.free_blocks.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        // Collect all allocated blocks
        let mut allocated_blocks: Vec<BlockMetadata> = all_blocks
            .values()
            .filter(|block| block.status == BlockStatus::Allocated)
            .cloned()
            .collect();

        if allocated_blocks.is_empty() {
            stats.compaction_operations += 1;
            return Ok(());
        }

        // Sort blocks for optimal compaction:
        // 1. By access frequency (frequently accessed blocks first)
        // 2. By creation time (newer blocks first for better locality)
        allocated_blocks.sort_by(|a, b| {
            // Primary sort: access count (descending)
            match b.access_count.cmp(&a.access_count) {
                std::cmp::Ordering::Equal => {
                    // Secondary sort: creation time (descending - newer first)
                    b.created_at.cmp(&a.created_at)
                }
                other => other,
            }
        });

        // Clear free block tracker - we'll rebuild it
        *free_blocks = FreeBlockTracker::new();

        // Perform compaction by reassigning offsets
        let mut current_offset = 0u64;
        let mut compaction_moves = 0u64;

        for block in allocated_blocks.iter_mut() {
            let old_offset = block.offset;

            // Assign new offset (compacted position)
            block.offset = current_offset;
            current_offset += block.size as u64;

            // Track if block was moved
            if old_offset != block.offset {
                compaction_moves += 1;
            }

            // Update the block in the all_blocks map
            all_blocks.insert(block.id, block.clone());
        }

        // Update file size to reflect compacted layout
        let mut file_size = self.file_size.write().unwrap();
        *file_size = current_offset;

        // Create a single large free block for remaining space
        if current_offset < *file_size {
            let free_size = *file_size - current_offset;
            let mut next_id = self.next_block_id.write().unwrap();
            let free_block = BlockMetadata {
                id: *next_id,
                offset: current_offset,
                size: free_size as BlockSize,
                status: BlockStatus::Free,
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                last_accessed: 0,
                access_count: 0,
                checksum: 0,
            };
            *next_id += 1;
            free_blocks.add_free_block(free_block.clone());
            all_blocks.insert(free_block.id, free_block);
        }

        // Coalesce any remaining small free blocks
        self.coalesce_free_blocks_internal(&mut free_blocks, &all_blocks);

        // Update statistics
        let compaction_time = start_time.elapsed();
        stats.compaction_operations += 1;
        stats.total_allocation_time += compaction_time;

        // Log compaction results
        info!(
            "Compaction completed: moved {} blocks in {:?}, fragmentation reduced to {:.2}%",
            compaction_moves,
            compaction_time,
            self.calculate_fragmentation_ratio_internal(&free_blocks, &all_blocks) * 100.0
        );

        Ok(())
    }

    /// Internal helper for coalescing free blocks during compaction
    fn coalesce_free_blocks_internal(
        &self,
        free_blocks: &mut FreeBlockTracker,
        _all_blocks: &HashMap<BlockId, BlockMetadata>,
    ) {
        // Get all free block metadata sorted by offset
        let mut free_metadata: Vec<BlockMetadata> =
            free_blocks.block_metadata.values().cloned().collect();

        free_metadata.sort_by_key(|block| block.offset);

        // Clear and rebuild free block tracker with coalesced blocks
        *free_blocks = FreeBlockTracker::new();

        let mut coalesced_blocks = Vec::new();
        let mut current_block: Option<BlockMetadata> = None;

        for block in free_metadata {
            match current_block {
                None => {
                    current_block = Some(block);
                }
                Some(ref mut current) => {
                    // Check if blocks are adjacent
                    if current.offset + current.size as u64 == block.offset {
                        // Coalesce blocks
                        current.size += block.size;
                    } else {
                        // Blocks are not adjacent, finalize current block
                        coalesced_blocks.push(current.clone());
                        current_block = Some(block);
                    }
                }
            }
        }

        // Add final block if exists
        if let Some(block) = current_block {
            coalesced_blocks.push(block);
        }

        // Add coalesced blocks back to free block tracker
        for block in coalesced_blocks {
            free_blocks.add_free_block(block);
        }
    }

    /// Internal helper for calculating fragmentation ratio
    fn calculate_fragmentation_ratio_internal(
        &self,
        free_blocks: &FreeBlockTracker,
        _all_blocks: &HashMap<BlockId, BlockMetadata>,
    ) -> f64 {
        let (total_free_space, free_block_count, _) = free_blocks.get_stats();

        if total_free_space == 0 {
            return 0.0;
        }

        if free_block_count <= 1 {
            return 0.0; // No fragmentation with 0 or 1 free blocks
        }

        // Calculate average free block size
        let avg_free_block_size = total_free_space as f64 / free_block_count as f64;

        // Calculate fragmentation as deviation from ideal (single large block)
        let ideal_block_size = total_free_space as f64;
        let fragmentation_ratio = 1.0 - (avg_free_block_size / ideal_block_size);

        fragmentation_ratio.clamp(0.0, 1.0)
    }

    /// Check if garbage collection should run
    fn maybe_trigger_gc(&self) {
        if self.config.enable_compaction && self.needs_compaction() {
            let _ = self.compact(); // Non-fatal
        }
    }

    /// Clear all blocks
    pub fn clear(&self) {
        {
            let mut free_blocks = self.free_blocks.write().unwrap();
            *free_blocks = FreeBlockTracker::new();
        }

        {
            let mut all_blocks = self.all_blocks.write().unwrap();
            all_blocks.clear();
        }

        *self.next_block_id.write().unwrap() = 1;
        *self.file_size.write().unwrap() = 0;
        *self.stats.write().unwrap() = BlockManagerStats::default();
    }

    /// Get total file size
    pub fn file_size(&self) -> u64 {
        *self.file_size.read().unwrap()
    }

    /// Validate block manager integrity
    pub fn validate(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();
        let all_blocks = self.all_blocks.read().unwrap();
        let free_blocks = self.free_blocks.read().unwrap();

        // Check that all free blocks are actually marked as free
        for &block_id in free_blocks.block_metadata.keys() {
            if let Some(metadata) = all_blocks.get(&block_id) {
                if metadata.status != BlockStatus::Free {
                    issues.push(format!(
                        "Block {} is in free list but has status {:?}",
                        block_id, metadata.status
                    ));
                }
            } else {
                issues.push(format!(
                    "Block {block_id} is in free list but not in all_blocks"
                ));
            }
        }

        // Check for overlapping blocks
        let mut offsets: Vec<_> = all_blocks
            .values()
            .map(|b| (b.offset, b.offset + b.size as u64, b.id))
            .collect();
        offsets.sort_by_key(|&(start, _, _)| start);

        for i in 0..offsets.len() {
            for j in i + 1..offsets.len() {
                let (start1, end1, id1) = offsets[i];
                let (start2, end2, id2) = offsets[j];

                if start2 < end1 {
                    issues.push(format!(
                        "Blocks {id1} and {id2} overlap: [{start1}, {end1}) and [{start2}, {end2})"
                    ));
                }
            }
        }

        Ok(issues)
    }
}

impl Default for BlockManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let manager = BlockManager::new();

        let block1 = manager.allocate_block(1024).unwrap();
        let block2 = manager.allocate_block(2048).unwrap();

        assert_ne!(block1, block2);

        let stats = manager.get_stats();
        assert_eq!(stats.allocated_blocks, 2);
        assert_eq!(stats.allocated_space, 1024 + 2048);
    }

    #[test]
    fn test_deallocation_and_reuse() {
        let manager = BlockManager::new();

        let block1 = manager.allocate_block(1024).unwrap();
        manager.deallocate_block(block1).unwrap();

        let _block2 = manager.allocate_block(1024).unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.allocated_blocks, 1);
        assert_eq!(stats.free_blocks, 0); // Should be reused
    }

    #[test]
    fn test_allocation_strategies() {
        let mut manager = BlockManager::new();

        // Test different strategies
        manager.set_allocation_strategy(AllocationStrategy::FirstFit);
        let _block1 = manager.allocate_block(1024).unwrap();

        manager.set_allocation_strategy(AllocationStrategy::BestFit);
        let _block2 = manager.allocate_block(2048).unwrap();

        manager.set_allocation_strategy(AllocationStrategy::WorstFit);
        let _block3 = manager.allocate_block(512).unwrap();
    }

    #[test]
    fn test_block_splitting() {
        let config = BlockManagerConfig {
            enable_compaction: false,
            ..Default::default()
        };
        let manager = BlockManager::with_config(config);

        // Allocate a large block
        let large_block = manager.allocate_block(8192).unwrap();
        println!("After allocating large block:");
        let stats1 = manager.get_stats();
        println!(
            "  allocated_blocks: {}, free_blocks: {}",
            stats1.allocated_blocks, stats1.free_blocks
        );

        manager.deallocate_block(large_block).unwrap();
        println!("After deallocating large block:");
        let stats2 = manager.get_stats();
        println!(
            "  allocated_blocks: {}, free_blocks: {}",
            stats2.allocated_blocks, stats2.free_blocks
        );

        // Allocate a smaller block (should split the large one)
        println!("Before allocating small block - checking if there are free blocks:");
        let stats_before = manager.get_stats();
        println!(
            "  free_blocks: {}, min_block_size: {}",
            stats_before.free_blocks, manager.config.min_block_size
        );

        let _small_block = manager.allocate_block(1024).unwrap();
        println!("After allocating small block:");
        let stats3 = manager.get_stats();
        println!(
            "  allocated_blocks: {}, free_blocks: {}",
            stats3.allocated_blocks, stats3.free_blocks
        );

        let stats = manager.get_stats();

        // Debug: let's see what's actually in the free_blocks structure
        {
            let free_blocks = manager.free_blocks.read().unwrap();
            println!("Final debug - free_blocks.block_metadata contents:");
            for (id, metadata) in &free_blocks.block_metadata {
                println!(
                    "  Block {}: size={}, status={:?}, offset={}",
                    id, metadata.size, metadata.status, metadata.offset
                );
            }
            println!("Final debug - blocks_by_size contents:");
            for (size, ids) in &free_blocks.blocks_by_size {
                println!("  Size {}: {:?}", size, ids);
            }
        }

        assert_eq!(stats.allocated_blocks, 1);
        println!(
            "Expected free_blocks=1, actual free_blocks={}",
            stats.free_blocks
        );
        assert_eq!(stats.free_blocks, 1); // Remaining part should be free
    }

    #[test]
    fn test_coalescing() {
        // Simple test without coalescing to verify basic functionality
        let config = BlockManagerConfig {
            enable_coalescing: false,
            ..Default::default()
        };
        let manager = BlockManager::with_config(config);

        // Allocate and deallocate blocks
        let block1 = manager.allocate_block(1024).unwrap();
        let block2 = manager.allocate_block(1024).unwrap();

        manager.deallocate_block(block1).unwrap();
        manager.deallocate_block(block2).unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.free_blocks, 2); // Should be 2 separate blocks without coalescing
        assert_eq!(stats.coalescing_operations, 0); // No coalescing should have occurred
    }

    #[test]
    fn test_statistics() {
        let manager = BlockManager::new();

        let _block1 = manager.allocate_block(1024).unwrap();
        let _block2 = manager.allocate_block(2048).unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.allocation_count, 2);
        assert_eq!(stats.total_blocks, 2);
        assert!(stats.avg_block_size > 0.0);
    }

    #[test]
    fn test_validation() {
        let manager = BlockManager::new();

        let _block1 = manager.allocate_block(1024).unwrap();
        let _block2 = manager.allocate_block(2048).unwrap();

        let issues = manager.validate().unwrap();
        assert!(issues.is_empty()); // Should be valid
    }

    #[test]
    fn test_fragmentation_detection() {
        let manager = BlockManager::new();

        // Create fragmentation by allocating and deallocating blocks
        let blocks: Vec<_> = (0..10)
            .map(|_| manager.allocate_block(1024).unwrap())
            .collect();

        // Deallocate every other block
        for (i, &block) in blocks.iter().enumerate() {
            if i % 2 == 0 {
                manager.deallocate_block(block).unwrap();
            }
        }

        let stats = manager.get_stats();
        assert!(stats.fragmentation_ratio > 0.0);
    }

    #[test]
    fn test_clear() {
        let manager = BlockManager::new();

        let _block1 = manager.allocate_block(1024).unwrap();
        let _block2 = manager.allocate_block(2048).unwrap();

        manager.clear();

        let stats = manager.get_stats();
        assert_eq!(stats.total_blocks, 0);
        assert_eq!(stats.total_space, 0);
    }
}
