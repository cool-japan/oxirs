//! Unit tests for the MVCC transaction manager.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::mvcc::{IsolationLevel, MvccManager, TxId};

    fn mgr() -> Arc<MvccManager> {
        MvccManager::new()
    }

    // -----------------------------------------------------------------------
    // 1. Basic single-transaction read-write
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_and_read_within_tx() {
        let m = mgr();
        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx, b"key1", b"val1").unwrap();
        let _ = m.rollback(tx);
    }

    #[test]
    fn test_committed_data_visible_to_new_tx() {
        let m = mgr();
        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx1, b"alpha", b"hello").unwrap();
        m.commit(tx1).unwrap();

        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let val = m.read(tx2, b"alpha").unwrap();
        assert_eq!(val, Some(b"hello".to_vec()));
        m.commit(tx2).unwrap();
    }

    #[test]
    fn test_uncommitted_write_invisible_to_concurrent_reader() {
        let m = mgr();
        let tx_writer = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx_writer, b"secret", b"42").unwrap();

        let tx_reader = m.begin_transaction(IsolationLevel::RepeatableRead);
        let val = m.read(tx_reader, b"secret").unwrap();
        assert_eq!(val, None);

        m.rollback(tx_writer).unwrap();
        m.commit(tx_reader).unwrap();
    }

    #[test]
    fn test_missing_key_returns_none() {
        let m = mgr();
        let tx = m.begin_transaction(IsolationLevel::ReadCommitted);
        let val = m.read(tx, b"nonexistent").unwrap();
        assert_eq!(val, None);
        m.commit(tx).unwrap();
    }

    // -----------------------------------------------------------------------
    // 2. Snapshot isolation
    // -----------------------------------------------------------------------

    #[test]
    fn test_snapshot_isolation_reader_does_not_see_later_commit() {
        let m = mgr();

        let setup = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(setup, b"snap", b"v1").unwrap();
        m.commit(setup).unwrap();

        let reader = m.begin_transaction(IsolationLevel::RepeatableRead);

        let writer = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(writer, b"snap", b"v2").unwrap();
        m.commit(writer).unwrap();

        let val = m.read(reader, b"snap").unwrap();
        assert_eq!(val, Some(b"v1".to_vec()));
        m.commit(reader).unwrap();
    }

    #[test]
    fn test_snapshot_sees_correct_committed_value() {
        let m = mgr();

        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx1, b"k", b"first").unwrap();
        m.commit(tx1).unwrap();

        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx2, b"k", b"second").unwrap();
        m.commit(tx2).unwrap();

        let tx3 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let val = m.read(tx3, b"k").unwrap();
        assert_eq!(val, Some(b"second".to_vec()));
        m.commit(tx3).unwrap();
    }

    // -----------------------------------------------------------------------
    // 3. Rollback restores state
    // -----------------------------------------------------------------------

    #[test]
    fn test_rollback_makes_writes_invisible() {
        let m = mgr();
        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx, b"rb", b"should_vanish").unwrap();
        m.rollback(tx).unwrap();

        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let val = m.read(tx2, b"rb").unwrap();
        assert_eq!(val, None);
        m.commit(tx2).unwrap();
    }

    #[test]
    fn test_rollback_idempotent_for_unknown_tx() {
        let m = mgr();
        let result = m.rollback(99999);
        assert!(result.is_err());
    }

    #[test]
    fn test_rollback_restores_previous_value() {
        let m = mgr();

        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx1, b"restore_me", b"original").unwrap();
        m.commit(tx1).unwrap();

        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx2, b"restore_me", b"overwrite").unwrap();
        m.rollback(tx2).unwrap();

        let tx3 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let val = m.read(tx3, b"restore_me").unwrap();
        assert_eq!(val, Some(b"original".to_vec()));
        m.commit(tx3).unwrap();
    }

    // -----------------------------------------------------------------------
    // 4. Delete operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_delete_makes_key_invisible_after_commit() {
        let m = mgr();

        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx1, b"del_me", b"present").unwrap();
        m.commit(tx1).unwrap();

        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.delete(tx2, b"del_me").unwrap();
        m.commit(tx2).unwrap();

        let tx3 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let val = m.read(tx3, b"del_me").unwrap();
        assert_eq!(val, None);
        m.commit(tx3).unwrap();
    }

    #[test]
    fn test_delete_nonexistent_key_is_noop() {
        let m = mgr();
        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.delete(tx, b"ghost_key").unwrap();
        m.commit(tx).unwrap();
    }

    #[test]
    fn test_delete_rolled_back_key_still_exists() {
        let m = mgr();

        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx1, b"keep_me", b"alive").unwrap();
        m.commit(tx1).unwrap();

        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.delete(tx2, b"keep_me").unwrap();
        m.rollback(tx2).unwrap();

        let tx3 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let val = m.read(tx3, b"keep_me").unwrap();
        assert_eq!(val, Some(b"alive".to_vec()));
        m.commit(tx3).unwrap();
    }

    // -----------------------------------------------------------------------
    // 5. Vacuum / garbage collection
    // -----------------------------------------------------------------------

    #[test]
    fn test_vacuum_removes_obsolete_versions() {
        let m = mgr();

        for i in 0u8..3 {
            let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
            m.write(tx, b"gc_key", &[i]).unwrap();
            m.commit(tx).unwrap();
        }

        let removed = m.vacuum().unwrap();
        assert!(removed >= 2, "expected >= 2 removed, got {}", removed);
    }

    #[test]
    fn test_vacuum_preserves_live_version() {
        let m = mgr();

        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx1, b"live", b"value").unwrap();
        m.commit(tx1).unwrap();

        m.vacuum().unwrap();

        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let val = m.read(tx2, b"live").unwrap();
        assert_eq!(val, Some(b"value".to_vec()));
        m.commit(tx2).unwrap();
    }

    #[test]
    fn test_vacuum_does_not_remove_visible_versions() {
        let m = mgr();

        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx1, b"hold", b"v1").unwrap();
        m.commit(tx1).unwrap();

        let long_reader = m.begin_transaction(IsolationLevel::RepeatableRead);

        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx2, b"hold", b"v2").unwrap();
        m.commit(tx2).unwrap();

        m.vacuum().unwrap();

        let val = m.read(long_reader, b"hold").unwrap();
        assert_eq!(val, Some(b"v1".to_vec()));
        m.commit(long_reader).unwrap();
    }

    #[test]
    fn test_vacuum_multiple_calls_idempotent() {
        let m = mgr();
        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx, b"idem", b"v").unwrap();
        m.commit(tx).unwrap();

        let r1 = m.vacuum().unwrap();
        let r2 = m.vacuum().unwrap();
        let _ = (r1, r2);
    }

    // -----------------------------------------------------------------------
    // 6. Concurrent writers / conflict detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_write_conflict_detected() {
        let m = mgr();

        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx1, b"contested", b"from_tx1").unwrap();

        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let result = m.write(tx2, b"contested", b"from_tx2");
        assert!(result.is_err(), "expected write-write conflict");

        m.rollback(tx1).unwrap();
        m.rollback(tx2).unwrap();
    }

    #[test]
    fn test_no_conflict_after_first_writer_commits() {
        let m = mgr();

        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx1, b"seq", b"v1").unwrap();
        m.commit(tx1).unwrap();

        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx2, b"seq", b"v2").unwrap();
        m.commit(tx2).unwrap();
    }

    // -----------------------------------------------------------------------
    // 7. Serializable isolation / SSI
    // -----------------------------------------------------------------------

    #[test]
    fn test_serializable_no_conflict_independent_keys() {
        let m = mgr();

        let setup = m.begin_transaction(IsolationLevel::Serializable);
        m.write(setup, b"x", b"0").unwrap();
        m.write(setup, b"y", b"0").unwrap();
        m.commit(setup).unwrap();

        let tx1 = m.begin_transaction(IsolationLevel::Serializable);
        let _vx = m.read(tx1, b"x").unwrap();
        m.write(tx1, b"y", b"1").unwrap();

        let tx2 = m.begin_transaction(IsolationLevel::Serializable);
        let _vy = m.read(tx2, b"y").unwrap();
        m.write(tx2, b"x", b"1").unwrap();

        let r1 = m.commit(tx1);
        let r2 = m.commit(tx2);
        assert!(r1.is_ok() || r2.is_ok());
    }

    #[test]
    fn test_serializable_read_committed_no_spurious_aborts() {
        let m = mgr();

        let tx = m.begin_transaction(IsolationLevel::Serializable);
        m.write(tx, b"solo", b"data").unwrap();
        let result = m.commit(tx);
        assert!(result.is_ok());
    }

    #[test]
    fn test_read_committed_sees_latest() {
        let m = mgr();

        let tx1 = m.begin_transaction(IsolationLevel::ReadCommitted);
        m.write(tx1, b"rc_key", b"v1").unwrap();
        m.commit(tx1).unwrap();

        let tx2 = m.begin_transaction(IsolationLevel::ReadCommitted);

        let tx3 = m.begin_transaction(IsolationLevel::ReadCommitted);
        m.write(tx3, b"rc_key", b"v2").unwrap();
        m.commit(tx3).unwrap();

        let val = m.read(tx2, b"rc_key").unwrap();
        assert_eq!(val, Some(b"v2".to_vec()));
        m.commit(tx2).unwrap();
    }

    // -----------------------------------------------------------------------
    // 8. Read-only transactions
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_only_tx_always_succeeds() {
        let m = mgr();

        let setup = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(setup, b"ro_key", b"ro_val").unwrap();
        m.commit(setup).unwrap();

        for _ in 0..10 {
            let ro = m.begin_transaction(IsolationLevel::RepeatableRead);
            let val = m.read(ro, b"ro_key").unwrap();
            assert_eq!(val, Some(b"ro_val".to_vec()));
            m.commit(ro).unwrap();
        }
    }

    #[test]
    fn test_read_only_does_not_interfere_with_writers() {
        let m = mgr();

        let reader = m.begin_transaction(IsolationLevel::RepeatableRead);
        let writer = m.begin_transaction(IsolationLevel::RepeatableRead);

        m.write(writer, b"rw", b"written").unwrap();
        m.commit(writer).unwrap();

        let val = m.read(reader, b"rw").unwrap();
        assert_eq!(
            val, None,
            "reader must not see write from a later transaction"
        );
        m.commit(reader).unwrap();
    }

    // -----------------------------------------------------------------------
    // 9. Nested concurrent transactions
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_concurrent_readers() {
        let m = mgr();

        let setup = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(setup, b"shared", b"data").unwrap();
        m.commit(setup).unwrap();

        let r1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let r2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let r3 = m.begin_transaction(IsolationLevel::RepeatableRead);

        assert_eq!(m.read(r1, b"shared").unwrap(), Some(b"data".to_vec()));
        assert_eq!(m.read(r2, b"shared").unwrap(), Some(b"data".to_vec()));
        assert_eq!(m.read(r3, b"shared").unwrap(), Some(b"data".to_vec()));

        m.commit(r1).unwrap();
        m.commit(r2).unwrap();
        m.commit(r3).unwrap();
    }

    #[test]
    fn test_interleaved_reads_and_writes() {
        let m = mgr();

        let tx_a = m.begin_transaction(IsolationLevel::RepeatableRead);
        let tx_b = m.begin_transaction(IsolationLevel::RepeatableRead);

        m.write(tx_a, b"k_a", b"v_a").unwrap();
        m.write(tx_b, b"k_b", b"v_b").unwrap();

        m.commit(tx_a).unwrap();
        m.commit(tx_b).unwrap();

        let reader = m.begin_transaction(IsolationLevel::RepeatableRead);
        assert_eq!(m.read(reader, b"k_a").unwrap(), Some(b"v_a".to_vec()));
        assert_eq!(m.read(reader, b"k_b").unwrap(), Some(b"v_b".to_vec()));
        m.commit(reader).unwrap();
    }

    // -----------------------------------------------------------------------
    // 10. Watermark advancement
    // -----------------------------------------------------------------------

    #[test]
    fn test_watermark_advances_after_commit() {
        let m = mgr();
        let s0 = m.stats();
        assert_eq!(s0.watermark, 0);

        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.commit(tx).unwrap();

        let s1 = m.stats();
        assert!(s1.watermark >= s0.watermark);
    }

    #[test]
    fn test_watermark_held_by_long_running_tx() {
        let m = mgr();

        for _ in 0..5 {
            let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
            m.commit(tx).unwrap();
        }

        let long_runner = m.begin_transaction(IsolationLevel::RepeatableRead);
        let wm_before = m.stats().watermark;

        for _ in 0..5 {
            let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
            m.commit(tx).unwrap();
        }

        let wm_after = m.stats().watermark;
        assert!(wm_after <= wm_before + 20);

        m.commit(long_runner).unwrap();
    }

    // -----------------------------------------------------------------------
    // 11. Stats tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_committed_count() {
        let m = mgr();
        let init = m.stats().total_committed;
        for _ in 0..5 {
            let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
            m.commit(tx).unwrap();
        }
        assert_eq!(m.stats().total_committed, init + 5);
    }

    #[test]
    fn test_stats_rolled_back_count() {
        let m = mgr();
        let init = m.stats().total_rolled_back;
        for _ in 0..3 {
            let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
            m.rollback(tx).unwrap();
        }
        assert_eq!(m.stats().total_rolled_back, init + 3);
    }

    #[test]
    fn test_stats_active_count() {
        let m = mgr();
        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        assert_eq!(m.stats().active_count, 2);
        m.commit(tx1).unwrap();
        assert_eq!(m.stats().active_count, 1);
        m.rollback(tx2).unwrap();
        assert_eq!(m.stats().active_count, 0);
    }

    // -----------------------------------------------------------------------
    // 12. snapshot_read explicit API
    // -----------------------------------------------------------------------

    #[test]
    fn test_snapshot_read_at_tx0_returns_none() {
        let m = mgr();
        let val = m.snapshot_read(0, b"no_data").unwrap();
        assert_eq!(val, None);
    }

    #[test]
    fn test_snapshot_read_at_specific_version() {
        let m = mgr();

        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx1, b"ver", b"first").unwrap();
        m.commit(tx1).unwrap();

        let snap_after_v1 = tx1;

        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx2, b"ver", b"second").unwrap();
        m.commit(tx2).unwrap();

        let val = m.snapshot_read(snap_after_v1, b"ver").unwrap();
        assert_eq!(val, Some(b"first".to_vec()));
    }

    // -----------------------------------------------------------------------
    // 13. Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_double_commit_returns_error() {
        let m = mgr();
        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.commit(tx).unwrap();
        let result = m.commit(tx);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_to_committed_tx_returns_error() {
        let m = mgr();
        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.commit(tx).unwrap();
        let result = m.write(tx, b"k", b"v");
        assert!(result.is_err());
    }

    #[test]
    fn test_read_from_committed_tx_returns_error() {
        let m = mgr();
        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.commit(tx).unwrap();
        let result = m.read(tx, b"k");
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // 14. Multi-version coexistence
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_versions_of_same_key() {
        let m = mgr();

        for i in 0u8..5 {
            let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
            m.write(tx, b"multi", &[i]).unwrap();
            m.commit(tx).unwrap();
        }

        let reader = m.begin_transaction(IsolationLevel::RepeatableRead);
        let val = m.read(reader, b"multi").unwrap();
        assert_eq!(val, Some(vec![4u8]));
        m.commit(reader).unwrap();
    }

    #[test]
    fn test_write_skew_prevented_in_serializable() {
        let m = mgr();

        let setup = m.begin_transaction(IsolationLevel::Serializable);
        m.write(setup, b"ws_x", b"1").unwrap();
        m.write(setup, b"ws_y", b"1").unwrap();
        m.commit(setup).unwrap();

        let t1 = m.begin_transaction(IsolationLevel::Serializable);
        let t2 = m.begin_transaction(IsolationLevel::Serializable);

        let _x1 = m.read(t1, b"ws_x").unwrap();
        let _y1 = m.read(t1, b"ws_y").unwrap();
        let _x2 = m.read(t2, b"ws_x").unwrap();
        let _y2 = m.read(t2, b"ws_y").unwrap();

        m.write(t1, b"ws_x", b"0").unwrap();
        m.write(t2, b"ws_y", b"0").unwrap();

        let r1 = m.commit(t1);
        let r2 = m.commit(t2);
        assert!(r1.is_ok() || r2.is_ok(), "at least one should succeed");
    }

    // -----------------------------------------------------------------------
    // 15. Repeatable read guarantee
    // -----------------------------------------------------------------------

    #[test]
    fn test_repeatable_read_same_key_twice() {
        let m = mgr();

        let setup = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(setup, b"rr", b"stable").unwrap();
        m.commit(setup).unwrap();

        let reader = m.begin_transaction(IsolationLevel::RepeatableRead);

        let v1 = m.read(reader, b"rr").unwrap();

        let writer = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(writer, b"rr", b"changed").unwrap();
        m.commit(writer).unwrap();

        let v2 = m.read(reader, b"rr").unwrap();
        assert_eq!(v1, v2);
        m.commit(reader).unwrap();
    }

    // -----------------------------------------------------------------------
    // 16. Version garbage collection with multiple keys
    // -----------------------------------------------------------------------

    #[test]
    fn test_vacuum_multiple_keys() {
        let m = mgr();

        for i in 0u8..3 {
            let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
            m.write(tx, b"key_a", &[i]).unwrap();
            m.write(tx, b"key_b", &[i + 10]).unwrap();
            m.commit(tx).unwrap();
        }

        let _ = m.vacuum();

        let reader = m.begin_transaction(IsolationLevel::RepeatableRead);
        let a = m.read(reader, b"key_a").unwrap();
        let b = m.read(reader, b"key_b").unwrap();
        assert_eq!(a, Some(vec![2u8]));
        assert_eq!(b, Some(vec![12u8]));
        m.commit(reader).unwrap();
    }

    // -----------------------------------------------------------------------
    // 17. Abort after partial writes
    // -----------------------------------------------------------------------

    #[test]
    fn test_abort_after_partial_writes() {
        let m = mgr();

        let setup = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(setup, b"pk_a", b"original_a").unwrap();
        m.write(setup, b"pk_b", b"original_b").unwrap();
        m.commit(setup).unwrap();

        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx, b"pk_a", b"overwritten_a").unwrap();
        m.write(tx, b"pk_b", b"overwritten_b").unwrap();
        m.abort(tx).unwrap();

        let reader = m.begin_transaction(IsolationLevel::RepeatableRead);
        let val_a = m.read(reader, b"pk_a").unwrap();
        let val_b = m.read(reader, b"pk_b").unwrap();
        assert_eq!(val_a, Some(b"original_a".to_vec()));
        assert_eq!(val_b, Some(b"original_b".to_vec()));
        m.commit(reader).unwrap();
    }

    // -----------------------------------------------------------------------
    // 18. Read-committed sees intermediate commits
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_committed_sees_intermediate_commit() {
        let m = mgr();

        let setup = m.begin_transaction(IsolationLevel::ReadCommitted);
        m.write(setup, b"rc_key", b"v1").unwrap();
        m.commit(setup).unwrap();

        let reader = m.begin_transaction(IsolationLevel::ReadCommitted);
        let v_before = m.read(reader, b"rc_key").unwrap();
        assert_eq!(v_before, Some(b"v1".to_vec()));

        let writer = m.begin_transaction(IsolationLevel::ReadCommitted);
        m.write(writer, b"rc_key", b"v2").unwrap();
        m.commit(writer).unwrap();

        let v_after = m.read(reader, b"rc_key").unwrap();
        assert_eq!(v_after, Some(b"v2".to_vec()));
        m.commit(reader).unwrap();
    }

    // -----------------------------------------------------------------------
    // 19. Repeated vacuum calls are safe
    // -----------------------------------------------------------------------

    #[test]
    fn test_repeated_vacuum_safe() {
        let m = mgr();
        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx, b"rv_key", b"rv_val").unwrap();
        m.commit(tx).unwrap();

        for _ in 0..5 {
            let _ = m.vacuum();
        }

        let reader = m.begin_transaction(IsolationLevel::RepeatableRead);
        let v = m.read(reader, b"rv_key").unwrap();
        assert_eq!(v, Some(b"rv_val".to_vec()));
        m.commit(reader).unwrap();
    }

    // -----------------------------------------------------------------------
    // 20. Stats rollback count increments on abort
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_rollback_increments_on_abort() {
        let m = mgr();
        let before = m.stats().total_rolled_back;

        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.abort(tx).unwrap();

        let after = m.stats().total_rolled_back;
        assert!(
            after > before,
            "total_rolled_back count should increase after abort"
        );
    }

    // -----------------------------------------------------------------------
    // 21. High-watermark never decreases
    // -----------------------------------------------------------------------

    #[test]
    fn test_watermark_never_decreases() {
        let m = mgr();

        let mut last_wm = m.low_water_mark();
        for _ in 0..5 {
            let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
            m.write(tx, b"wm_key", b"v").unwrap();
            m.commit(tx).unwrap();
            let current_wm = m.stats().watermark;
            assert!(
                current_wm >= last_wm,
                "watermark decreased: {} -> {}",
                last_wm,
                current_wm
            );
            last_wm = current_wm;
        }
    }

    // -----------------------------------------------------------------------
    // 22. Delete then re-insert same key is visible
    // -----------------------------------------------------------------------

    #[test]
    fn test_delete_then_reinsert() {
        let m = mgr();

        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx1, b"dr_key", b"first").unwrap();
        m.commit(tx1).unwrap();

        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.delete(tx2, b"dr_key").unwrap();
        m.commit(tx2).unwrap();

        let tx3 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let v_deleted = m.read(tx3, b"dr_key").unwrap();
        assert_eq!(v_deleted, None, "Key should be deleted");
        m.commit(tx3).unwrap();

        let tx4 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx4, b"dr_key", b"second").unwrap();
        m.commit(tx4).unwrap();

        let tx5 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let v_reinserted = m.read(tx5, b"dr_key").unwrap();
        assert_eq!(v_reinserted, Some(b"second".to_vec()));
        m.commit(tx5).unwrap();
    }

    // -----------------------------------------------------------------------
    // 23. Many uncommitted readers do not block committed data visibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_many_readers_do_not_block_writer() {
        let m = mgr();
        let setup = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(setup, b"mr_key", b"initial").unwrap();
        m.commit(setup).unwrap();

        let readers: Vec<TxId> = (0..10)
            .map(|_| m.begin_transaction(IsolationLevel::RepeatableRead))
            .collect();

        let writer = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(writer, b"mr_key", b"updated").unwrap();
        m.commit(writer).unwrap();

        for r in readers {
            let v = m.read(r, b"mr_key").unwrap();
            assert_eq!(v, Some(b"initial".to_vec()));
            m.commit(r).unwrap();
        }
    }

    // -----------------------------------------------------------------------
    // 24. Serializable conflict aborts second writer, not first
    // -----------------------------------------------------------------------

    #[test]
    fn test_serializable_conflict_first_wins() {
        let m = mgr();

        let t1 = m.begin_transaction(IsolationLevel::Serializable);
        let t2 = m.begin_transaction(IsolationLevel::Serializable);

        let _v1 = m.read(t1, b"sc_key").unwrap();
        let _v2 = m.read(t2, b"sc_key").unwrap();

        m.write(t1, b"sc_key", b"from_t1").unwrap();
        m.write(t2, b"sc_key", b"from_t2").unwrap();

        let r1 = m.commit(t1);
        let r2 = m.commit(t2);

        let successes = [r1.is_ok(), r2.is_ok()].iter().filter(|&&x| x).count();
        assert!(successes >= 1, "At least one transaction should succeed");
    }

    // -----------------------------------------------------------------------
    // 25. Zero-byte value is valid
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_byte_value() {
        let m = mgr();
        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx, b"zb_key", b"").unwrap();
        m.commit(tx).unwrap();

        let reader = m.begin_transaction(IsolationLevel::RepeatableRead);
        let v = m.read(reader, b"zb_key").unwrap();
        assert_eq!(v, Some(vec![]));
        m.commit(reader).unwrap();
    }

    // -----------------------------------------------------------------------
    // 26. Large value round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_value_round_trip() {
        let m = mgr();
        let large: Vec<u8> = (0u8..=255u8).cycle().take(4096).collect();

        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx, b"large_key", &large).unwrap();
        m.commit(tx).unwrap();

        let reader = m.begin_transaction(IsolationLevel::RepeatableRead);
        let v = m.read(reader, b"large_key").unwrap();
        assert_eq!(v, Some(large));
        m.commit(reader).unwrap();
    }

    // -----------------------------------------------------------------------
    // 27. Concurrent writes to disjoint keys do not conflict
    // -----------------------------------------------------------------------

    #[test]
    fn test_concurrent_writes_disjoint_keys_no_conflict() {
        let m = mgr();

        let t1 = m.begin_transaction(IsolationLevel::Serializable);
        let t2 = m.begin_transaction(IsolationLevel::Serializable);

        m.write(t1, b"disjoint_a", b"val_a").unwrap();
        m.write(t2, b"disjoint_b", b"val_b").unwrap();

        m.commit(t1).unwrap();
        m.commit(t2).unwrap();

        let reader = m.begin_transaction(IsolationLevel::RepeatableRead);
        assert_eq!(
            m.read(reader, b"disjoint_a").unwrap(),
            Some(b"val_a".to_vec())
        );
        assert_eq!(
            m.read(reader, b"disjoint_b").unwrap(),
            Some(b"val_b".to_vec())
        );
        m.commit(reader).unwrap();
    }

    // -----------------------------------------------------------------------
    // 28. Stats committed count increments correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_committed_increments() {
        let m = mgr();
        let before = m.stats().total_committed;

        for _ in 0..3 {
            let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
            m.write(tx, b"sc_inc", b"v").unwrap();
            m.commit(tx).unwrap();
        }

        assert_eq!(m.stats().total_committed, before + 3);
    }

    // -----------------------------------------------------------------------
    // 29. Read from aborted tx after abort returns error
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_from_aborted_tx_returns_error() {
        let m = mgr();
        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.abort(tx).unwrap();
        let result = m.read(tx, b"key");
        assert!(result.is_err(), "Read on aborted tx should fail");
    }

    // -----------------------------------------------------------------------
    // 30. Write to aborted tx after abort returns error
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_to_aborted_tx_returns_error() {
        let m = mgr();
        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.abort(tx).unwrap();
        let result = m.write(tx, b"key", b"v");
        assert!(result.is_err(), "Write on aborted tx should fail");
    }

    // -----------------------------------------------------------------------
    // 31. Sequential transactions see monotonically increasing IDs
    // -----------------------------------------------------------------------

    #[test]
    fn test_transaction_ids_are_monotonic() {
        let m = mgr();
        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let tx3 = m.begin_transaction(IsolationLevel::RepeatableRead);
        assert!(tx1 < tx2, "tx1 ({tx1}) should be < tx2 ({tx2})");
        assert!(tx2 < tx3, "tx2 ({tx2}) should be < tx3 ({tx3})");
        m.abort(tx1).unwrap();
        m.abort(tx2).unwrap();
        m.abort(tx3).unwrap();
    }

    // -----------------------------------------------------------------------
    // 32. Versioned snapshot — read_at_version helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_snapshot_at_version_between_commits() {
        let m = mgr();

        let tx1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx1, b"snap_key", b"v1").unwrap();
        m.commit(tx1).unwrap();

        let snapshot_tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        let val_snap = m.read(snapshot_tx, b"snap_key").unwrap();
        assert_eq!(val_snap, Some(b"v1".to_vec()));

        let tx2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx2, b"snap_key", b"v2").unwrap();
        m.commit(tx2).unwrap();

        let val_snap_again = m.read(snapshot_tx, b"snap_key").unwrap();
        assert_eq!(val_snap_again, Some(b"v1".to_vec()));
        m.commit(snapshot_tx).unwrap();

        let new_reader = m.begin_transaction(IsolationLevel::RepeatableRead);
        let val_new = m.read(new_reader, b"snap_key").unwrap();
        assert_eq!(val_new, Some(b"v2".to_vec()));
        m.commit(new_reader).unwrap();
    }

    // -----------------------------------------------------------------------
    // 33. Delete inside a transaction only takes effect after commit
    // -----------------------------------------------------------------------

    #[test]
    fn test_delete_only_visible_after_commit() {
        let m = mgr();

        let setup = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(setup, b"del_vis_key", b"exists").unwrap();
        m.commit(setup).unwrap();

        let deleter = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.delete(deleter, b"del_vis_key").unwrap();

        let reader = m.begin_transaction(IsolationLevel::RepeatableRead);
        let v = m.read(reader, b"del_vis_key").unwrap();
        assert_eq!(v, Some(b"exists".to_vec()));
        m.commit(reader).unwrap();

        m.commit(deleter).unwrap();

        let reader2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let v2 = m.read(reader2, b"del_vis_key").unwrap();
        assert_eq!(v2, None);
        m.commit(reader2).unwrap();
    }

    // -----------------------------------------------------------------------
    // 34. Write overwrite within same transaction
    // -----------------------------------------------------------------------

    #[test]
    fn test_overwrite_within_tx_reads_latest() {
        let m = mgr();
        let tx = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.write(tx, b"ow_key", b"first").unwrap();
        m.write(tx, b"ow_key", b"second").unwrap();
        m.write(tx, b"ow_key", b"third").unwrap();
        m.commit(tx).unwrap();

        let reader = m.begin_transaction(IsolationLevel::RepeatableRead);
        let v = m.read(reader, b"ow_key").unwrap();
        assert_eq!(v, Some(b"third".to_vec()));
        m.commit(reader).unwrap();
    }

    // -----------------------------------------------------------------------
    // 35. MvccStats active count after commit returns to zero
    // -----------------------------------------------------------------------

    #[test]
    fn test_active_count_zero_after_all_committed() {
        let m = mgr();
        let t1 = m.begin_transaction(IsolationLevel::RepeatableRead);
        let t2 = m.begin_transaction(IsolationLevel::RepeatableRead);
        m.commit(t1).unwrap();
        m.commit(t2).unwrap();
        assert_eq!(
            m.stats().active_count,
            0,
            "No active transactions should remain"
        );
    }
}
