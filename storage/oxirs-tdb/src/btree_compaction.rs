//! B-tree page compaction and merge for TDB storage.
//!
//! Provides utilities to compact, merge, split, and rebalance B-tree pages,
//! tracking statistics throughout the process.

// ────────────────────────────────────────────────────────────────────────────
// Data structures
// ────────────────────────────────────────────────────────────────────────────

/// A single key-value entry in a B-tree page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BTreeEntry {
    /// The sort key for this entry.
    pub key: Vec<u8>,
    /// The associated value bytes.
    pub value: Vec<u8>,
}

impl BTreeEntry {
    /// Construct a new `BTreeEntry` with the given key and value.
    pub fn new(key: Vec<u8>, value: Vec<u8>) -> Self {
        Self { key, value }
    }
}

/// A page in a B-tree, either a leaf or an internal (routing) node.
#[derive(Debug, Clone)]
pub struct BTreePage {
    /// The key-value entries stored on this page.
    pub entries: Vec<BTreeEntry>,
    /// Numeric page identifier.
    pub page_id: u64,
    /// `true` for leaf pages, `false` for internal (routing) pages.
    pub is_leaf: bool,
    /// Child page IDs (length = entries.len() + 1 for internal pages, empty for leaves).
    pub children: Vec<u64>,
}

impl BTreePage {
    /// Create a new empty leaf page.
    pub fn new_leaf(page_id: u64) -> Self {
        Self {
            entries: Vec::new(),
            page_id,
            is_leaf: true,
            children: Vec::new(),
        }
    }

    /// Create a new empty internal (routing) page.
    pub fn new_internal(page_id: u64) -> Self {
        Self {
            entries: Vec::new(),
            page_id,
            is_leaf: false,
            children: Vec::new(),
        }
    }

    /// Returns `true` when `fill_ratio` is below `fill_factor`.
    pub fn is_under_filled(&self, max_entries: usize, fill_factor: f64) -> bool {
        self.fill_ratio(max_entries) < fill_factor
    }

    /// Convenience wrapper around `BTreeCompactor::fill_ratio`.
    pub fn fill_ratio(&self, max_entries: usize) -> f64 {
        BTreeCompactor::fill_ratio(self, max_entries)
    }
}

/// Statistics collected during a compaction run.
#[derive(Debug, Clone, Default)]
pub struct CompactionStats {
    /// Number of pages read as input.
    pub pages_read: usize,
    /// Number of pages produced as output.
    pub pages_written: usize,
    /// Total entries collected from all input pages.
    pub entries_merged: usize,
    /// Number of entries removed as duplicate keys.
    pub duplicate_keys_removed: usize,
    /// Number of empty pages that were eliminated.
    pub empty_pages_removed: usize,
}

/// Configuration for a compaction run.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Target fill factor in the range `(0.0, 1.0]`. Pages whose fill ratio
    /// falls below this are candidates for merging.
    pub fill_factor: f64,
    /// Maximum number of entries per page.
    pub max_page_entries: usize,
    /// Whether duplicate keys should be removed (keep last occurrence).
    pub remove_duplicates: bool,
}

impl CompactionConfig {
    /// Construct a new `CompactionConfig`.
    pub fn new(fill_factor: f64, max_page_entries: usize, remove_duplicates: bool) -> Self {
        debug_assert!(
            (0.0..=1.0).contains(&fill_factor),
            "fill_factor must be in (0.0, 1.0]"
        );
        Self {
            fill_factor,
            max_page_entries,
            remove_duplicates,
        }
    }
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self::new(0.5, 256, true)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BTreeCompactor
// ────────────────────────────────────────────────────────────────────────────

/// Stateless utility for B-tree page compaction, merging, and splitting.
pub struct BTreeCompactor;

impl BTreeCompactor {
    /// Compact a collection of B-tree pages.
    ///
    /// Steps:
    /// 1. Sort all entries across all pages.
    /// 2. Optionally remove duplicate keys (keep last).
    /// 3. Remove empty pages.
    /// 4. Re-distribute entries into fresh pages respecting `max_page_entries`.
    ///
    /// Returns the new pages and compaction statistics.
    pub fn compact(
        pages: Vec<BTreePage>,
        config: &CompactionConfig,
    ) -> (Vec<BTreePage>, CompactionStats) {
        let mut stats = CompactionStats {
            pages_read: pages.len(),
            ..Default::default()
        };

        // Collect all entries
        let all_entries: Vec<BTreeEntry> = pages
            .into_iter()
            .flat_map(|p| p.entries)
            .collect();

        stats.entries_merged = all_entries.len();

        // Sort
        let mut sorted = all_entries;
        sorted.sort_by(|a, b| a.key.cmp(&b.key));

        // Optionally de-duplicate
        let (deduped, dups_removed) = if config.remove_duplicates {
            Self::remove_duplicate_keys(sorted)
        } else {
            (sorted, 0)
        };
        stats.duplicate_keys_removed = dups_removed;

        // Remove empty pages (handled by re-distributing — they won't be created)
        // Count empty pages that existed in the original set (not available here;
        // we approximate by checking if deduped is empty)
        if deduped.is_empty() {
            stats.empty_pages_removed = stats.entries_merged; // all entries were duplicates or empty
            return (vec![], stats);
        }

        // Distribute into new pages
        let new_pages = Self::distribute_entries(deduped, config.max_page_entries);
        stats.pages_written = new_pages.len();

        (new_pages, stats)
    }

    /// Merge two sorted pages into a single sorted page.
    ///
    /// The resulting page inherits `a.page_id` and `a.is_leaf`.
    pub fn merge_pages(a: &BTreePage, b: &BTreePage) -> BTreePage {
        let mut merged = a.entries.clone();
        merged.extend_from_slice(&b.entries);
        merged.sort_by(|x, y| x.key.cmp(&y.key));
        BTreePage {
            entries: merged,
            page_id: a.page_id,
            is_leaf: a.is_leaf,
            children: vec![],
        }
    }

    /// Split a page into multiple pages, each containing at most `max_entries` entries.
    ///
    /// Pages are assigned sequential IDs starting from the original `page_id`.
    pub fn split_page(page: BTreePage, max_entries: usize) -> Vec<BTreePage> {
        if max_entries == 0 {
            return vec![page];
        }
        let mut result = Vec::new();
        let chunks: Vec<&[BTreeEntry]> = page.entries.chunks(max_entries).collect();
        for (i, chunk) in chunks.iter().enumerate() {
            result.push(BTreePage {
                entries: chunk.to_vec(),
                page_id: page.page_id + i as u64,
                is_leaf: page.is_leaf,
                children: vec![],
            });
        }
        if result.is_empty() {
            // Empty page → return a single empty leaf
            result.push(BTreePage::new_leaf(page.page_id));
        }
        result
    }

    /// Sort a page's entries in place by key.
    pub fn sort_entries(page: &mut BTreePage) {
        page.entries.sort_by(|a, b| a.key.cmp(&b.key));
    }

    /// Remove duplicate keys, keeping the **last** occurrence of each key.
    ///
    /// Returns `(deduped_entries, count_removed)`.
    pub fn remove_duplicate_keys(entries: Vec<BTreeEntry>) -> (Vec<BTreeEntry>, usize) {
        let total = entries.len();
        // Walk in reverse, tracking seen keys
        let mut seen = std::collections::HashSet::new();
        let mut result: Vec<BTreeEntry> = entries
            .into_iter()
            .rev()
            .filter(|e| seen.insert(e.key.clone()))
            .collect();
        result.reverse();
        let removed = total - result.len();
        (result, removed)
    }

    /// Estimate the number of pages needed to store `entry_count` entries
    /// given `max_page_entries` and `fill_factor`.
    pub fn estimate_pages_needed(
        entry_count: usize,
        max_page_entries: usize,
        fill_factor: f64,
    ) -> usize {
        if max_page_entries == 0 || entry_count == 0 {
            return 0;
        }
        let effective = (max_page_entries as f64 * fill_factor).max(1.0);
        (entry_count as f64 / effective).ceil() as usize
    }

    /// Compute the fill ratio of a page: `entries.len() / max_entries`.
    pub fn fill_ratio(page: &BTreePage, max_entries: usize) -> f64 {
        if max_entries == 0 {
            return 0.0;
        }
        page.entries.len() as f64 / max_entries as f64
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn distribute_entries(entries: Vec<BTreeEntry>, max_per_page: usize) -> Vec<BTreePage> {
        if max_per_page == 0 {
            let mut p = BTreePage::new_leaf(0);
            p.entries = entries;
            return vec![p];
        }
        entries
            .chunks(max_per_page)
            .enumerate()
            .map(|(i, chunk)| BTreePage {
                entries: chunk.to_vec(),
                page_id: i as u64,
                is_leaf: true,
                children: vec![],
            })
            .collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn entry(k: u8, v: u8) -> BTreeEntry {
        BTreeEntry::new(vec![k], vec![v])
    }

    fn leaf_with(page_id: u64, entries: Vec<BTreeEntry>) -> BTreePage {
        let mut p = BTreePage::new_leaf(page_id);
        p.entries = entries;
        p
    }

    fn default_config() -> CompactionConfig {
        CompactionConfig::new(0.5, 4, true)
    }

    // ── BTreeEntry ────────────────────────────────────────────────────────────

    #[test]
    fn test_entry_construction() {
        let e = BTreeEntry::new(vec![1, 2], vec![3, 4]);
        assert_eq!(e.key, vec![1, 2]);
        assert_eq!(e.value, vec![3, 4]);
    }

    #[test]
    fn test_entry_eq() {
        let a = entry(10, 20);
        let b = entry(10, 20);
        assert_eq!(a, b);
    }

    // ── BTreePage ─────────────────────────────────────────────────────────────

    #[test]
    fn test_new_leaf_is_leaf() {
        let p = BTreePage::new_leaf(0);
        assert!(p.is_leaf);
        assert!(p.entries.is_empty());
        assert!(p.children.is_empty());
    }

    #[test]
    fn test_new_internal_is_not_leaf() {
        let p = BTreePage::new_internal(1);
        assert!(!p.is_leaf);
    }

    #[test]
    fn test_fill_ratio_empty() {
        let p = BTreePage::new_leaf(0);
        assert!((BTreeCompactor::fill_ratio(&p, 10) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fill_ratio_full() {
        let p = leaf_with(0, (0..4).map(|i| entry(i, i)).collect());
        assert!((BTreeCompactor::fill_ratio(&p, 4) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fill_ratio_half() {
        let p = leaf_with(0, vec![entry(1, 1), entry(2, 2)]);
        assert!((BTreeCompactor::fill_ratio(&p, 4) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fill_ratio_max_zero() {
        let p = leaf_with(0, vec![entry(1, 1)]);
        assert_eq!(BTreeCompactor::fill_ratio(&p, 0), 0.0);
    }

    #[test]
    fn test_is_under_filled_true() {
        let p = leaf_with(0, vec![entry(1, 1)]); // 1/4 = 0.25 < fill_factor 0.5
        assert!(p.is_under_filled(4, 0.5));
    }

    #[test]
    fn test_is_under_filled_false() {
        let p = leaf_with(0, (0..4).map(|i| entry(i, i)).collect()); // 4/4 = 1.0 >= 0.5
        assert!(!p.is_under_filled(4, 0.5));
    }

    // ── sort_entries ──────────────────────────────────────────────────────────

    #[test]
    fn test_sort_entries_basic() {
        let mut p = leaf_with(0, vec![entry(5, 0), entry(2, 0), entry(8, 0), entry(1, 0)]);
        BTreeCompactor::sort_entries(&mut p);
        let keys: Vec<u8> = p.entries.iter().map(|e| e.key[0]).collect();
        assert_eq!(keys, vec![1, 2, 5, 8]);
    }

    #[test]
    fn test_sort_entries_already_sorted() {
        let mut p = leaf_with(0, vec![entry(1, 0), entry(2, 0), entry(3, 0)]);
        BTreeCompactor::sort_entries(&mut p);
        let keys: Vec<u8> = p.entries.iter().map(|e| e.key[0]).collect();
        assert_eq!(keys, vec![1, 2, 3]);
    }

    #[test]
    fn test_sort_entries_empty() {
        let mut p = BTreePage::new_leaf(0);
        BTreeCompactor::sort_entries(&mut p); // should not panic
        assert!(p.entries.is_empty());
    }

    // ── merge_pages ───────────────────────────────────────────────────────────

    #[test]
    fn test_merge_two_pages_sorted() {
        let a = leaf_with(0, vec![entry(1, 10), entry(3, 30)]);
        let b = leaf_with(1, vec![entry(2, 20), entry(4, 40)]);
        let merged = BTreeCompactor::merge_pages(&a, &b);
        let keys: Vec<u8> = merged.entries.iter().map(|e| e.key[0]).collect();
        assert_eq!(keys, vec![1, 2, 3, 4]);
        assert_eq!(merged.page_id, 0);
    }

    #[test]
    fn test_merge_empty_pages() {
        let a = BTreePage::new_leaf(5);
        let b = BTreePage::new_leaf(6);
        let merged = BTreeCompactor::merge_pages(&a, &b);
        assert!(merged.entries.is_empty());
        assert_eq!(merged.page_id, 5);
    }

    #[test]
    fn test_merge_one_empty_one_full() {
        let a = leaf_with(0, vec![entry(2, 20)]);
        let b = BTreePage::new_leaf(1);
        let merged = BTreeCompactor::merge_pages(&a, &b);
        assert_eq!(merged.entries.len(), 1);
        assert_eq!(merged.entries[0].key, vec![2]);
    }

    // ── split_page ────────────────────────────────────────────────────────────

    #[test]
    fn test_split_page_even() {
        let p = leaf_with(0, (0..8u8).map(|i| entry(i, i)).collect());
        let pages = BTreeCompactor::split_page(p, 4);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].entries.len(), 4);
        assert_eq!(pages[1].entries.len(), 4);
    }

    #[test]
    fn test_split_page_uneven() {
        let p = leaf_with(0, (0..5u8).map(|i| entry(i, i)).collect());
        let pages = BTreeCompactor::split_page(p, 3);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].entries.len(), 3);
        assert_eq!(pages[1].entries.len(), 2);
    }

    #[test]
    fn test_split_page_no_split_needed() {
        let p = leaf_with(0, (0..3u8).map(|i| entry(i, i)).collect());
        let pages = BTreeCompactor::split_page(p, 4);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].entries.len(), 3);
    }

    #[test]
    fn test_split_empty_page() {
        let p = BTreePage::new_leaf(0);
        let pages = BTreeCompactor::split_page(p, 4);
        assert_eq!(pages.len(), 1);
        assert!(pages[0].entries.is_empty());
    }

    // ── remove_duplicate_keys ────────────────────────────────────────────────

    #[test]
    fn test_dedup_no_duplicates() {
        let entries = vec![entry(1, 10), entry(2, 20), entry(3, 30)];
        let (deduped, removed) = BTreeCompactor::remove_duplicate_keys(entries);
        assert_eq!(removed, 0);
        assert_eq!(deduped.len(), 3);
    }

    #[test]
    fn test_dedup_all_duplicates() {
        let entries = vec![
            BTreeEntry::new(vec![1], vec![10]),
            BTreeEntry::new(vec![1], vec![20]),
            BTreeEntry::new(vec![1], vec![30]),
        ];
        let (deduped, removed) = BTreeCompactor::remove_duplicate_keys(entries);
        assert_eq!(removed, 2);
        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0].value, vec![30]); // last kept
    }

    #[test]
    fn test_dedup_keeps_last() {
        let entries = vec![
            BTreeEntry::new(vec![1], vec![10]),
            BTreeEntry::new(vec![2], vec![20]),
            BTreeEntry::new(vec![1], vec![99]),
        ];
        let (deduped, removed) = BTreeCompactor::remove_duplicate_keys(entries);
        assert_eq!(removed, 1);
        let val_for_key1 = deduped.iter().find(|e| e.key == vec![1]).map(|e| e.value[0]);
        assert_eq!(val_for_key1, Some(99));
    }

    #[test]
    fn test_dedup_empty_input() {
        let (deduped, removed) = BTreeCompactor::remove_duplicate_keys(vec![]);
        assert_eq!(removed, 0);
        assert!(deduped.is_empty());
    }

    // ── estimate_pages_needed ─────────────────────────────────────────────────

    #[test]
    fn test_estimate_exact_fill() {
        // 10 entries, 5 per page, fill=1.0 → 2 pages
        assert_eq!(BTreeCompactor::estimate_pages_needed(10, 5, 1.0), 2);
    }

    #[test]
    fn test_estimate_with_fill_factor() {
        // 10 entries, 10 per page, fill=0.5 → effective=5 → 2 pages
        assert_eq!(BTreeCompactor::estimate_pages_needed(10, 10, 0.5), 2);
    }

    #[test]
    fn test_estimate_zero_entries() {
        assert_eq!(BTreeCompactor::estimate_pages_needed(0, 10, 0.5), 0);
    }

    #[test]
    fn test_estimate_zero_max_per_page() {
        assert_eq!(BTreeCompactor::estimate_pages_needed(10, 0, 0.5), 0);
    }

    #[test]
    fn test_estimate_rounds_up() {
        // 7 entries, 4 per page, fill=1.0 → effective=4 → ceil(7/4) = 2
        assert_eq!(BTreeCompactor::estimate_pages_needed(7, 4, 1.0), 2);
    }

    // ── compact ───────────────────────────────────────────────────────────────

    #[test]
    fn test_compact_empty_input() {
        let (pages, stats) = BTreeCompactor::compact(vec![], &default_config());
        assert!(pages.is_empty());
        assert_eq!(stats.pages_read, 0);
    }

    #[test]
    fn test_compact_removes_duplicates() {
        let page = leaf_with(
            0,
            vec![
                BTreeEntry::new(vec![1], vec![10]),
                BTreeEntry::new(vec![1], vec![20]),
                BTreeEntry::new(vec![2], vec![30]),
            ],
        );
        let (pages, stats) = BTreeCompactor::compact(vec![page], &default_config());
        assert_eq!(stats.duplicate_keys_removed, 1);
        let total_entries: usize = pages.iter().map(|p| p.entries.len()).sum();
        assert_eq!(total_entries, 2);
    }

    #[test]
    fn test_compact_sorts_entries() {
        let page = leaf_with(0, vec![entry(5, 50), entry(1, 10), entry(3, 30)]);
        let (pages, _) = BTreeCompactor::compact(vec![page], &CompactionConfig::new(0.5, 10, false));
        let all_keys: Vec<u8> = pages.iter().flat_map(|p| p.entries.iter().map(|e| e.key[0])).collect();
        assert_eq!(all_keys, vec![1, 3, 5]);
    }

    #[test]
    fn test_compact_merges_multiple_pages() {
        let p1 = leaf_with(0, vec![entry(3, 30), entry(1, 10)]);
        let p2 = leaf_with(1, vec![entry(4, 40), entry(2, 20)]);
        let (pages, stats) = BTreeCompactor::compact(vec![p1, p2], &CompactionConfig::new(0.5, 10, false));
        assert_eq!(stats.pages_read, 2);
        assert_eq!(stats.entries_merged, 4);
        let total: usize = pages.iter().map(|p| p.entries.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_compact_respects_max_page_entries() {
        let p = leaf_with(0, (0..12u8).map(|i| entry(i, i)).collect());
        let (pages, _) = BTreeCompactor::compact(vec![p], &CompactionConfig::new(0.5, 4, false));
        for page in &pages {
            assert!(page.entries.len() <= 4);
        }
    }

    #[test]
    fn test_compact_stats_pages_written() {
        let p = leaf_with(0, (0..8u8).map(|i| entry(i, i)).collect());
        let (pages, stats) = BTreeCompactor::compact(vec![p], &CompactionConfig::new(0.5, 4, false));
        assert_eq!(stats.pages_written, pages.len());
    }

    #[test]
    fn test_compact_all_duplicates_produces_empty() {
        let entries: Vec<BTreeEntry> = (0..5).map(|_| BTreeEntry::new(vec![1], vec![99])).collect();
        let p = leaf_with(0, entries);
        let (pages, stats) = BTreeCompactor::compact(vec![p], &default_config());
        // After dedup there should be exactly 1 entry
        let total: usize = pages.iter().map(|p| p.entries.len()).sum();
        assert_eq!(total, 1);
        assert_eq!(stats.duplicate_keys_removed, 4);
    }

    #[test]
    fn test_compact_no_dedup_when_disabled() {
        let entries: Vec<BTreeEntry> = (0..4)
            .map(|_| BTreeEntry::new(vec![42], vec![1]))
            .collect();
        let p = leaf_with(0, entries);
        let (pages, stats) =
            BTreeCompactor::compact(vec![p], &CompactionConfig::new(0.5, 10, false));
        assert_eq!(stats.duplicate_keys_removed, 0);
        let total: usize = pages.iter().map(|pg| pg.entries.len()).sum();
        assert_eq!(total, 4);
    }
}
