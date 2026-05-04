//! Integration tests for bidirectional state synchronisation.

use std::time::Duration;

use oxirs_physics::sync::{
    state_diff, BidirectionalSync, BidirectionalSyncConfig, PhysicsState, PhysicsStateValue,
    StateGraphConfig, StateToRdfWriter, SyncDirection,
};

fn entity() -> &'static str {
    "urn:example:battery:001"
}

fn state_at(step: u64, voltage: f64, temperature: f64) -> PhysicsState {
    let mut s = PhysicsState::new(entity());
    s.step = step;
    s.set_scalar("voltage", voltage);
    s.set_scalar("temperature", temperature);
    s
}

#[test]
fn full_render_round_trips_through_diff_when_no_changes() {
    let writer = StateToRdfWriter::new();
    let s = state_at(0, 3.7, 298.15);
    let q = writer.render_full(&s);
    assert!(q.contains("INSERT DATA"));
    assert!(q.contains("phys:State"));

    // No-change diff must produce nothing.
    let d = state_diff(&s, &s, 1e-9);
    assert!(d.is_empty());
}

#[test]
fn diff_emits_only_changed_property() {
    let writer = StateToRdfWriter::new();
    let s0 = state_at(0, 3.7, 298.15);
    let s1 = state_at(1, 3.95, 298.15); // only voltage changed

    let q = writer.render_diff(&s0, &s1).expect("non-empty diff");
    assert!(q.contains("phys:voltage"));
    // Temperature was unchanged, must not appear at all.
    assert!(!q.contains("phys:temperature"));
}

#[test]
fn writer_supports_default_graph() {
    let cfg = StateGraphConfig {
        named_graph: None,
        ..Default::default()
    };
    let writer = StateToRdfWriter::with_config(cfg);
    let q = writer.render_full(&state_at(0, 3.7, 298.15));
    // No GRAPH wrapper when named graph is None.
    assert!(!q.contains("GRAPH <"));
    assert!(q.contains("INSERT DATA"));
}

#[test]
fn bidirectional_sync_full_round_trip() {
    let mut sync = BidirectionalSync::new(BidirectionalSyncConfig {
        min_interval: Duration::from_millis(0),
        ..Default::default()
    });

    // Step 1: push initial state — full snapshot.
    let s0 = state_at(0, 3.7, 298.15);
    let r0 = sync.push_state(&s0).expect("push 0");
    assert_eq!(r0.direction, SyncDirection::StateToRdf);
    assert!(r0
        .sparql
        .expect("must produce SPARQL")
        .contains("phys:State"));

    // Step 2: push diff — only voltage changes.
    std::thread::sleep(Duration::from_millis(2));
    let s1 = state_at(1, 3.95, 298.15);
    let r1 = sync.push_state(&s1).expect("push 1");
    assert_eq!(r1.diff.changed.len(), 1);
    assert!(r1.diff.changed.contains_key("voltage"));

    // Step 3: pull a state from RDF mock — re-extract.
    std::thread::sleep(Duration::from_millis(2));
    let r2 = sync
        .pull_state(entity(), 2, |entity, step| {
            assert_eq!(entity, "urn:example:battery:001");
            assert_eq!(step, 2);
            Ok(vec![
                oxirs_physics::sync::rdf_to_state::RdfPropertyRow {
                    predicate: "voltage".to_string(),
                    literal: "4.10".to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#double".to_string()),
                },
                oxirs_physics::sync::rdf_to_state::RdfPropertyRow {
                    predicate: "temperature".to_string(),
                    literal: "299.0".to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#double".to_string()),
                },
            ])
        })
        .expect("pull 2");
    assert_eq!(r2.direction, SyncDirection::RdfToState);
    let pulled = r2.re_extracted.expect("must re-extract");
    assert_eq!(pulled.entity_iri, entity());
    assert_eq!(pulled.values.len(), 2);
    match pulled.values.get("voltage") {
        Some(PhysicsStateValue::Scalar(v)) => assert!((*v - 4.10).abs() < 1e-9),
        other => panic!("expected scalar voltage, got {other:?}"),
    }
}

#[test]
fn min_interval_skips_premature_pushes() {
    let mut sync = BidirectionalSync::new(BidirectionalSyncConfig {
        min_interval: Duration::from_secs(60), // way longer than test runtime
        ..Default::default()
    });
    let s = state_at(0, 3.7, 298.15);
    let r0 = sync.push_state(&s).expect("first push");
    assert_eq!(r0.direction, SyncDirection::StateToRdf);
    let r1 = sync.push_state(&s).expect("second push");
    assert_eq!(r1.direction, SyncDirection::Skipped);
    assert!(r1.sparql.is_none());
}

#[test]
fn vector_property_round_trips_through_csv_string_encoding() {
    let mut s = PhysicsState::new(entity());
    s.values.insert(
        "velocity".to_string(),
        PhysicsStateValue::Vector(vec![1.0, -2.0, 3.5]),
    );
    let writer = StateToRdfWriter::new();
    let full = writer.render_full(&s);
    // Vector is encoded as comma-separated string.
    assert!(full.contains("\"1,-2,3.5\""));

    // The reverse extraction recovers it as a Vector value.
    let row = oxirs_physics::sync::rdf_to_state::RdfPropertyRow {
        predicate: "velocity".to_string(),
        literal: "1,-2,3.5".to_string(),
        datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
    };
    let out = oxirs_physics::sync::RdfToStateExtractor::new()
        .extract(entity(), 0, &[row])
        .expect("extract ok");
    match out.state.values.get("velocity") {
        Some(PhysicsStateValue::Vector(v)) => {
            assert_eq!(v.as_slice(), &[1.0, -2.0, 3.5]);
        }
        other => panic!("expected vector, got {other:?}"),
    }
}

#[test]
fn empty_diff_after_first_push_produces_no_sparql() {
    let mut sync = BidirectionalSync::new(BidirectionalSyncConfig {
        min_interval: Duration::from_millis(0),
        ..Default::default()
    });
    let s = state_at(0, 3.7, 298.15);
    sync.push_state(&s).expect("first push");
    std::thread::sleep(Duration::from_millis(2));
    let r = sync.push_state(&s).expect("second push, no changes");
    assert_eq!(r.direction, SyncDirection::StateToRdf);
    assert!(r.sparql.is_none());
    assert!(r.diff.is_empty());
}
