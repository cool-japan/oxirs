//! Integration test: watermark propagation across operator topologies.
//!
//! Verifies the W2-S6 contract:
//!
//! 1. Watermarks are non-decreasing per operator output edge.
//! 2. An operator's output watermark is the minimum of its input watermarks.
//! 3. The global watermark is the minimum across all sinks.
//! 4. Late events arriving past the watermark trigger
//!    [`oxirs_stream::watermark::LateDataDecision::Drop`] when the policy is
//!    `Drop`, or `Reassign(watermark)` within the allowed-lateness budget.

use oxirs_stream::error::StreamError;
use oxirs_stream::watermark::{
    AllowedLatenessTracker, LateDataDecision, LateDataHandler, LateDataPolicy, OperatorId,
    SideOutputRouter, WatermarkAligner, WatermarkGenerator, WatermarkPropagator,
};

#[test]
fn diamond_topology_global_watermark_is_min_across_paths() {
    //          src
    //          / \
    //         a   b
    //          \ /
    //          join
    //           |
    //          sink
    let mut p = WatermarkPropagator::new();
    let src = OperatorId::new("src");
    let a = OperatorId::new("a");
    let b = OperatorId::new("b");
    let j = OperatorId::new("join");
    let sink = OperatorId::new("sink");

    p.add_edge(src.clone(), a.clone());
    p.add_edge(src.clone(), b.clone());
    p.add_edge(a.clone(), j.clone());
    p.add_edge(b.clone(), j.clone());
    p.add_edge(j, sink.clone());

    let g = p.push_source(&src, 5_000).expect("ok");
    assert_eq!(g, Some(5_000));
    assert_eq!(p.watermark_of(&sink), Some(5_000));
}

#[test]
fn merge_two_independent_sources_takes_minimum() {
    // src1 → m, src2 → m, m → sink
    let mut p = WatermarkPropagator::new();
    let s1 = OperatorId::new("s1");
    let s2 = OperatorId::new("s2");
    let m = OperatorId::new("merge");
    let snk = OperatorId::new("sink");
    p.add_edge(s1.clone(), m.clone());
    p.add_edge(s2.clone(), m.clone());
    p.add_edge(m.clone(), snk.clone());

    p.push_source(&s1, 9_000).expect("ok");
    let g = p.push_source(&s2, 4_000).expect("ok");
    assert_eq!(g, Some(4_000));
    assert_eq!(p.watermark_of(&m), Some(4_000));
}

#[test]
fn monotonicity_violation_returns_error() {
    let mut p = WatermarkPropagator::new();
    let s = OperatorId::new("s");
    let snk = OperatorId::new("snk");
    p.add_edge(s.clone(), snk);
    p.push_source(&s, 1_000).expect("ok");
    let err = p.push_source(&s, 500).expect_err("monotonic violation");
    assert!(matches!(err, StreamError::WatermarkViolation { .. }));
}

#[test]
fn watermark_aligner_minimum_across_sources() {
    let mut a = WatermarkAligner::new();
    a.update("src1", 10_000);
    a.update("src2", 5_000);
    a.update("src3", 7_500);
    assert_eq!(a.global_watermark(), 5_000);
    assert_eq!(a.source_count(), 3);
    assert!(a.all_beyond(4_999));
    assert!(!a.all_beyond(5_000));
}

#[test]
fn watermark_generator_emits_at_threshold_and_tracks_max() {
    let mut g = WatermarkGenerator::new(100).with_advance_threshold(5);
    let mut last = None;
    for i in 0..5 {
        last = g.observe(i * 1_000);
    }
    let wm = last.expect("must emit at threshold");
    assert_eq!(wm.timestamp, 3_900); // 4000 - 100
    assert_eq!(wm.advance_count, 1);
}

#[test]
fn late_event_dropped_under_drop_policy() {
    let mut handler = LateDataHandler::new(LateDataPolicy::Drop);
    let on_time = handler.handle(5_000, 4_000);
    assert_eq!(on_time, LateDataDecision::Process);
    let late = handler.handle(1_000, 5_000);
    assert_eq!(late, LateDataDecision::Drop);
    assert_eq!(handler.late_event_count, 1);
}

#[test]
fn late_event_reassigned_within_allowed_lateness() {
    let mut handler = LateDataHandler::new(LateDataPolicy::Reassign {
        max_lateness_ms: 1_000,
    });
    let d = handler.handle(4_500, 5_000);
    assert_eq!(d, LateDataDecision::Reassign(5_000));
}

#[test]
fn late_event_routed_to_side_output() {
    let mut handler = LateDataHandler::new(LateDataPolicy::SideOutput {
        channel: "late-events".to_string(),
    });
    let d = handler.handle(1_000, 5_000);
    assert_eq!(d, LateDataDecision::SideOutput);
}

#[test]
fn allowed_lateness_tracker_governs_window_open_state() {
    let mut t = AllowedLatenessTracker::new();
    t.register("w1", 1_000, 500);
    assert!(t.is_open("w1", 1_400));
    assert!(!t.is_open("w1", 1_500));
    let evicted = t.evict_closed(2_000);
    assert!(evicted.contains(&"w1".to_string()));
    assert_eq!(t.len(), 0);
}

#[test]
fn side_output_router_buffers_per_channel() {
    let mut router: SideOutputRouter<i32> = SideOutputRouter::new();
    router.push("late", 1);
    router.push("late", 2);
    router.push("dlq", 7);
    let drained = router.drain("late");
    assert_eq!(drained, vec![1, 2]);
    assert_eq!(router.len("late"), 0);
    assert_eq!(router.len("dlq"), 1);
}
