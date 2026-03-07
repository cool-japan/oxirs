//! # Temporal RDF – Version Control and Time-Travel Queries
//!
//! This module provides v0.3.0 temporal capabilities for OxiRS TDB:
//!
//! - **`version_store`** – Append-only versioned triple storage.  Every insert
//!   and delete is timestamped; no record is ever discarded.
//! - **`snapshot`** – Immutable point-in-time views of the dataset.
//! - **`time_travel`** – Query the store as it existed at any past timestamp,
//!   with optional triple-pattern filtering.
//! - **`changelog`** – Ordered audit log of all changes with per-transaction
//!   and time-range filtering.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxirs_tdb::temporal::version_store::TemporalVersionStore;
//! use oxirs_tdb::temporal::time_travel::{TimeTravelQuery, TriplePattern};
//! use chrono::Utc;
//!
//! let mut store = TemporalVersionStore::new();
//! store.insert_now("Alice", "knows", "Bob").unwrap();
//!
//! let snap = TimeTravelQuery::at(Utc::now())
//!     .execute(&store)
//!     .unwrap();
//!
//! assert_eq!(snap.len(), 1);
//! ```

pub mod changelog;
pub mod snapshot;
pub mod time_travel;
pub mod version_store;

pub use changelog::{ChangeEntry, ChangeLog, ChangeLogStats, ChangeOperation};
pub use snapshot::TemporalSnapshot;
pub use time_travel::{TimeTravelQuery, TimeTravelQueryBuilder, TriplePattern};
pub use version_store::{TemporalDiff, TemporalVersionStore, TripleKey, VersionedTriple};

#[cfg(test)]
mod integration_tests {
    //! Integration tests exercising multiple temporal sub-modules together.

    use super::*;
    use chrono::{TimeZone, Utc};

    fn ts(y: i32, m: u32, d: u32) -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    /// Build a store and changelog that share a consistent history.
    fn build_scenario() -> (TemporalVersionStore, ChangeLog) {
        let mut store = TemporalVersionStore::new();
        let mut log = ChangeLog::new();

        let txn1 = store
            .insert_at("Alice", "knows", "Bob", ts(2024, 1, 1))
            .unwrap();
        log.record_insert(
            VersionedTriple::new("Alice", "knows", "Bob", ts(2024, 1, 1), txn1),
            txn1,
        );

        let txn2 = store
            .insert_at("Alice", "knows", "Carol", ts(2024, 6, 1))
            .unwrap();
        log.record_insert(
            VersionedTriple::new("Alice", "knows", "Carol", ts(2024, 6, 1), txn2),
            txn2,
        );

        let txn3 = store
            .insert_at("Bob", "likes", "Rust", ts(2024, 3, 1))
            .unwrap();
        log.record_insert(
            VersionedTriple::new("Bob", "likes", "Rust", ts(2024, 3, 1), txn3),
            txn3,
        );

        store
            .delete_at("Alice", "knows", "Bob", ts(2024, 9, 1))
            .unwrap();
        log.record_delete(
            VersionedTriple::new("Alice", "knows", "Bob", ts(2024, 1, 1), txn1),
            txn1,
        );

        (store, log)
    }

    #[test]
    fn test_insert_then_query_at_past_timestamp() {
        let (store, _) = build_scenario();
        let snap = TimeTravelQuery::at(ts(2024, 2, 1)).execute(&store).unwrap();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap.triples()[0].subject, "Alice");
    }

    #[test]
    fn test_snapshot_after_all_inserts_before_delete() {
        let (store, _) = build_scenario();
        let snap = TimeTravelQuery::at(ts(2024, 7, 1)).execute(&store).unwrap();
        assert_eq!(snap.len(), 3);
    }

    #[test]
    fn test_snapshot_after_delete() {
        let (store, _) = build_scenario();
        let snap = TimeTravelQuery::at(ts(2024, 10, 1))
            .execute(&store)
            .unwrap();
        assert_eq!(snap.len(), 2);
        assert!(snap
            .triples()
            .iter()
            .all(|t| !(t.subject == "Alice" && t.object == "Bob")));
    }

    #[test]
    fn test_history_of_triple() {
        let (store, _) = build_scenario();
        let hist = store.history("Alice", "knows", "Bob");
        assert_eq!(hist.len(), 1);
        assert!(hist[0].valid_to.is_some());
    }

    #[test]
    fn test_diff_range() {
        let (store, _) = build_scenario();
        let diff = store.diff(ts(2024, 5, 1), ts(2024, 10, 1));
        // Carol was added in June (in range), Bob was deleted in Sept (in range)
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added[0].object, "Carol");
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed[0].object, "Bob");
    }

    #[test]
    fn test_snapshot_diff_against() {
        let (store, _) = build_scenario();
        let snap_early = TimeTravelQuery::at(ts(2024, 2, 1)).execute(&store).unwrap();
        let snap_late = TimeTravelQuery::at(ts(2024, 7, 1)).execute(&store).unwrap();

        let (added, removed) = snap_early.diff_against(&snap_late);
        assert_eq!(added.len(), 2); // Carol + Bob-likes-Rust added between Feb and July
        assert!(removed.is_empty());
    }

    #[test]
    fn test_changelog_total_entries() {
        let (_, log) = build_scenario();
        assert_eq!(log.len(), 4); // 3 inserts + 1 delete
    }

    #[test]
    fn test_time_travel_builder_with_predicate_filter() {
        let (store, _) = build_scenario();
        let q = TimeTravelQueryBuilder::new()
            .as_of(ts(2024, 7, 1))
            .predicate("knows")
            .build()
            .unwrap();
        let snap = q.execute(&store).unwrap();
        assert_eq!(snap.len(), 2);
    }

    #[test]
    fn test_version_count_after_operations() {
        let (store, _) = build_scenario();
        // 3 distinct triple keys, 3 version records (1 per key)
        assert_eq!(store.triple_key_count(), 3);
        assert_eq!(store.version_count(), 3);
    }

    #[test]
    fn test_current_after_delete() {
        let (store, _) = build_scenario();
        let current = store.current();
        assert_eq!(current.len(), 2);
        assert!(current
            .iter()
            .all(|t| !(t.subject == "Alice" && t.object == "Bob")));
    }

    #[test]
    fn test_delete_already_deleted_returns_error() {
        let mut store = TemporalVersionStore::new();
        store.insert_at("s", "p", "o", ts(2024, 1, 1)).unwrap();
        store.delete_at("s", "p", "o", ts(2024, 6, 1)).unwrap();
        // Second delete should fail – no live version
        let result = store.delete_at("s", "p", "o", ts(2024, 9, 1));
        assert!(result.is_err());
    }
}
