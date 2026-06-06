//! Garbage collection / vacuum logic for the MVCC version store.
//!
//! The `vacuum_versions` function prunes version entries that are no longer
//! visible to any active transaction, keeping only the data required by
//! current and future snapshots.

use std::collections::{HashSet, VecDeque};

use super::transaction::{TxId, VersionedEntry, TX_ID_COMMITTED};

/// Remove obsolete version entries from a single key's version chain.
///
/// Returns the number of entries removed.
///
/// # Parameters
/// - `versions` — mutable reference to the ordered (oldest-first) version chain
/// - `watermark` — the current low-water mark (min `start_tx_id` among active txns)
/// - `committed` — set of all committed transaction IDs
pub fn vacuum_versions(
    versions: &mut VecDeque<VersionedEntry>,
    watermark: TxId,
    committed: &HashSet<TxId>,
) -> usize {
    let mut removed = 0usize;
    let mut keep: VecDeque<VersionedEntry> = VecDeque::new();
    let mut latest_live_kept_at_watermark = false;

    // Drain into a temp vec and walk newest-first so we can track shadowing.
    let mut temp: Vec<VersionedEntry> = versions.drain(..).collect();
    temp.reverse(); // newest first

    for entry in temp {
        let creator_committed =
            entry.created_tx_id == TX_ID_COMMITTED || committed.contains(&entry.created_tx_id);
        let creator_old = entry.created_tx_id <= watermark && creator_committed;

        let deleter_committed_and_old = entry
            .deleted_tx_id
            .map(|d| d <= watermark && committed.contains(&d))
            .unwrap_or(false);

        if creator_old && deleter_committed_and_old {
            // Fully obsolete — remove
            removed += 1;
            continue;
        }

        if creator_old && entry.deleted_tx_id.is_none() && latest_live_kept_at_watermark {
            // A newer live version already shadows this one at the watermark.
            removed += 1;
            continue;
        }

        // Mark that we have kept a live version that is old enough to be the
        // representative at the watermark.
        if entry.deleted_tx_id.is_none() && creator_committed && creator_old {
            latest_live_kept_at_watermark = true;
        }

        keep.push_back(entry);
    }

    // Restore oldest-first order
    let mut restored: Vec<VersionedEntry> = keep.into_iter().collect();
    restored.reverse();
    *versions = restored.into();

    removed
}
