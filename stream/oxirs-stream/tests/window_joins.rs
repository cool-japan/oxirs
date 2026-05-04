//! Integration test: watermark-driven window joins.
//!
//! Exercises the W2-S6 window-join semantics:
//!
//! * Tumbling × Tumbling: events join only when they share a pane.
//! * Tumbling × Sliding: a right event in any sliding pane that overlaps
//!   the left tumbling pane joins.
//! * Session × Session: two sessions with the same key join when their
//!   gap-extended intervals overlap and both have closed under the
//!   watermark.
//! * Allowed-lateness budget extends pane/session lifetime.
//! * Late events arriving past the budget are dropped.

use oxirs_stream::window::{
    SessionSessionJoin, SessionSessionJoinConfig, TumblingSlidingJoin, TumblingSlidingJoinConfig,
    TumblingTumblingJoin, TumblingTumblingJoinConfig,
};

#[test]
fn tumbling_tumbling_smoke_test() {
    let cfg = TumblingTumblingJoinConfig::new(1_000);
    let mut j: TumblingTumblingJoin<&str, &str> = TumblingTumblingJoin::new(cfg);

    // pane [0,1000)
    j.push_left("k1".into(), 100, "L0");
    let r = j.push_right("k1".into(), 800, "R0");
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].pane_end_ms, 1_000);

    // pane [1000,2000)
    j.push_left("k2".into(), 1_500, "L1");
    let r = j.push_right("k2".into(), 1_900, "R1");
    assert_eq!(r.len(), 1);

    // Different panes — no cross-pane join.
    let none = j.push_right("k1".into(), 1_500, "Rmiss");
    assert!(none.is_empty());

    // Watermark closes pane 0.
    let purged = j.advance_watermark(1_001);
    assert_eq!(purged, 1);
    // Late event for closed pane is dropped.
    let drop = j.push_left("k1".into(), 500, "Late");
    assert!(drop.is_empty());
    assert_eq!(j.stats().late_events_dropped, 1);
}

#[test]
fn tumbling_sliding_overlapping_panes() {
    // left tumbling 1000ms; right sliding 1000/500.
    let cfg = TumblingSlidingJoinConfig::new(1_000, 1_000, 500);
    let mut j: TumblingSlidingJoin<&str, &str> = TumblingSlidingJoin::new(cfg);

    // Buffer right event at ts=300; lives in panes -500 and 0.
    let r0 = j.push_right("k".into(), 300, "R0");
    assert!(r0.is_empty());
    // Left at ts=200 — pane [0,1000) — picks up R0 exactly once (de-duped).
    let r1 = j.push_left("k".into(), 200, "L0");
    assert_eq!(r1.len(), 1);

    // Watermark past everything → purge.
    let purged = j.advance_watermark(10_000);
    assert!(purged >= 2);
    assert_eq!(j.pane_count(), 0);
}

#[test]
fn tumbling_sliding_late_event_dropped() {
    let cfg = TumblingSlidingJoinConfig::new(1_000, 1_000, 500);
    let mut j: TumblingSlidingJoin<&str, &str> = TumblingSlidingJoin::new(cfg);
    j.advance_watermark(10_000);
    let r = j.push_right("k".into(), 500, "Late");
    assert!(r.is_empty());
    assert_eq!(j.stats().late_events_dropped, 1);
}

#[test]
fn session_session_overlapping_emits_cross_product() {
    let cfg = SessionSessionJoinConfig::new(500);
    let mut j: SessionSessionJoin<&str, &str> = SessionSessionJoin::new(cfg);

    // Left session: 100, 200, 300 → end=800.
    j.push_left("k".into(), 100, "L0");
    j.push_left("k".into(), 200, "L1");
    j.push_left("k".into(), 300, "L2");

    // Right session: 250, 350 → end=850.
    j.push_right("k".into(), 250, "R0");
    j.push_right("k".into(), 350, "R1");

    // Both sessions close at wm ≥ 850.
    let out = j.advance_watermark(900);
    assert_eq!(out.len(), 6, "3 left × 2 right = 6");
    assert_eq!(j.session_count(), 0);
}

#[test]
fn session_session_non_overlapping_emits_nothing() {
    let cfg = SessionSessionJoinConfig::new(50);
    let mut j: SessionSessionJoin<&str, &str> = SessionSessionJoin::new(cfg);
    j.push_left("k".into(), 100, "L"); // session ends 150
    j.push_right("k".into(), 1_000, "R"); // session ends 1050
    let out = j.advance_watermark(2_000);
    assert!(out.is_empty());
}

#[test]
fn session_session_separate_keys_isolated() {
    let cfg = SessionSessionJoinConfig::new(500);
    let mut j: SessionSessionJoin<&str, &str> = SessionSessionJoin::new(cfg);
    j.push_left("a".into(), 100, "La");
    j.push_right("b".into(), 200, "Rb");
    let out = j.advance_watermark(2_000);
    assert!(out.is_empty());
}

#[test]
fn tumbling_tumbling_allowed_lateness_extends_window() {
    let cfg = TumblingTumblingJoinConfig::new(1_000).with_lateness(500);
    let mut j: TumblingTumblingJoin<&str, &str> = TumblingTumblingJoin::new(cfg);
    j.push_left("k".into(), 100, "L0");
    j.advance_watermark(1_499); // closes at 1500
                                // late but within budget
    let out = j.push_right("k".into(), 600, "R0");
    assert_eq!(out.len(), 1);
    j.advance_watermark(1_501);
    let out = j.push_right("k".into(), 700, "Drop");
    assert!(out.is_empty());
    assert_eq!(j.stats().late_events_dropped, 1);
}
