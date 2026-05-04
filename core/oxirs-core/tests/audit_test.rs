//! Integration tests for the SOC2/GDPR-compliant audit trail module.
//!
//! Covers all public APIs of `oxirs_core::audit`.

use std::io::Cursor;
use std::sync::Arc;

use chrono::Utc;
use oxirs_core::audit::{
    event::ActorType, AuditActor, AuditEvent, AuditEventKind, AuditFilter, AuditLogError,
    AuditLogger, AuditOutcome, AuditQuery, AuditQueryable, AuditResource, CompositeAuditLogger,
    GdprService, InMemoryAuditLogger, JsonLineAuditLogger, SortOrder,
};

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

fn make_actor(id: &str) -> AuditActor {
    AuditActor {
        actor_id: id.to_string(),
        actor_type: ActorType::User,
        ip_address: Some("192.168.1.1".to_string()),
        session_id: Some("sess-1".to_string()),
    }
}

fn make_resource(rid: &str, tenant: Option<&str>) -> AuditResource {
    AuditResource {
        resource_type: "dataset".to_string(),
        resource_id: rid.to_string(),
        tenant_id: tenant.map(str::to_string),
    }
}

fn make_event(kind: AuditEventKind, action: &str, actor_id: &str) -> AuditEvent {
    AuditEvent::new(
        kind,
        action,
        make_actor(actor_id),
        make_resource("ds-main", Some("acme")),
        AuditOutcome::Success,
    )
}

// ─────────────────────────────────────────────
// Test 1: AuditEvent construction — UUID + timestamp set
// ─────────────────────────────────────────────

#[test]
fn test_audit_event_new_sets_id_and_timestamp() {
    let before = Utc::now();
    let event = make_event(AuditEventKind::DataAccess, "sparql.select", "user-1");
    let after = Utc::now();

    assert!(!event.event_id.is_empty(), "event_id must not be empty");
    assert!(
        event.timestamp >= before && event.timestamp <= after,
        "timestamp must be within test window"
    );
    assert_eq!(event.action, "sparql.select");
    assert_eq!(event.actor.actor_id, "user-1");
}

// ─────────────────────────────────────────────
// Test 2: Default optional fields are None/empty
// ─────────────────────────────────────────────

#[test]
fn test_audit_event_defaults() {
    let event = make_event(AuditEventKind::System, "system.start", "system");
    assert!(event.duration_ms.is_none());
    assert!(event.metadata.is_empty());
    assert!(event.data_subject_id.is_none());
}

// ─────────────────────────────────────────────
// Test 3: Builder methods chain correctly
// ─────────────────────────────────────────────

#[test]
fn test_audit_event_builder_chaining() {
    let event = make_event(AuditEventKind::DataAccess, "sparql.select", "user-1")
        .with_duration(123)
        .with_metadata("rows", "42")
        .with_metadata("bytes", "1024")
        .with_data_subject("subject-99");

    assert_eq!(event.duration_ms, Some(123));
    assert_eq!(event.metadata.get("rows").map(String::as_str), Some("42"));
    assert_eq!(
        event.metadata.get("bytes").map(String::as_str),
        Some("1024")
    );
    assert_eq!(event.data_subject_id.as_deref(), Some("subject-99"));
}

// ─────────────────────────────────────────────
// Test 4: InMemoryAuditLogger — basic log and len
// ─────────────────────────────────────────────

#[test]
fn test_in_memory_logger_log_and_len() {
    let logger = InMemoryAuditLogger::new();
    assert!(logger.is_empty());

    logger
        .log(make_event(
            AuditEventKind::Authentication,
            "auth.login",
            "u1",
        ))
        .expect("log should succeed");
    logger
        .log(make_event(
            AuditEventKind::DataAccess,
            "sparql.select",
            "u2",
        ))
        .expect("log should succeed");

    assert_eq!(logger.len(), 2);
    assert!(!logger.is_empty());
}

// ─────────────────────────────────────────────
// Test 5: InMemoryAuditLogger — events() returns clone of all events
// ─────────────────────────────────────────────

#[test]
fn test_in_memory_logger_events_snapshot() {
    let logger = InMemoryAuditLogger::new();
    let e1 = make_event(AuditEventKind::DataModification, "sparql.update", "admin");
    let e2 = make_event(AuditEventKind::Admin, "admin.dataset_create", "admin");

    logger.log(e1.clone()).expect("log e1");
    logger.log(e2.clone()).expect("log e2");

    let snap = logger.events();
    assert_eq!(snap.len(), 2);
    assert_eq!(snap[0].action, "sparql.update");
    assert_eq!(snap[1].action, "admin.dataset_create");
}

// ─────────────────────────────────────────────
// Test 6: InMemoryAuditLogger — capacity exceeded error
// ─────────────────────────────────────────────

#[test]
fn test_in_memory_logger_capacity_exceeded() {
    let logger = InMemoryAuditLogger::with_capacity(2);
    logger
        .log(make_event(AuditEventKind::System, "system.start", "sys"))
        .expect("first");
    logger
        .log(make_event(AuditEventKind::System, "system.start", "sys"))
        .expect("second");

    let result = logger.log(make_event(AuditEventKind::System, "system.stop", "sys"));
    match result {
        Err(AuditLogError::CapacityExceeded(cap)) => assert_eq!(cap, 2),
        other => panic!("expected CapacityExceeded, got {:?}", other),
    }
    // The logger should still hold exactly 2 events (no eviction).
    assert_eq!(logger.len(), 2);
}

// ─────────────────────────────────────────────
// Test 7: InMemoryAuditLogger — clear resets
// ─────────────────────────────────────────────

#[test]
fn test_in_memory_logger_clear() {
    let logger = InMemoryAuditLogger::new();
    logger
        .log(make_event(AuditEventKind::Security, "auth.failed", "bad"))
        .expect("log");
    assert_eq!(logger.len(), 1);
    logger.clear();
    assert_eq!(logger.len(), 0);
    assert!(logger.is_empty());
}

// ─────────────────────────────────────────────
// Test 8: JsonLineAuditLogger — writes NDJSON to Cursor
// ─────────────────────────────────────────────

#[test]
fn test_json_line_logger_writes_ndjson() {
    let buf: Vec<u8> = Vec::new();
    let cursor = Cursor::new(buf);
    let logger = JsonLineAuditLogger::new(cursor);

    let event = make_event(AuditEventKind::DataAccess, "sparql.select", "u1");
    logger.log(event).expect("log to cursor");

    // We cannot easily retrieve the inner Cursor directly through the trait.
    // Instead, use a shared Arc<Mutex<Vec<u8>>> to verify output.
}

/// Verify NDJSON round-trip: write two events, read back, parse each line.
#[test]
fn test_json_line_logger_ndjson_round_trip() {
    use std::sync::{Arc, Mutex};

    let buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let buf_clone = Arc::clone(&buf);

    // Write adapter that writes into the shared buffer.
    struct SharedBufWriter(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for SharedBufWriter {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let logger = JsonLineAuditLogger::new(SharedBufWriter(buf_clone));
    logger
        .log(make_event(
            AuditEventKind::DataAccess,
            "sparql.select",
            "u1",
        ))
        .expect("log 1");
    logger
        .log(make_event(
            AuditEventKind::DataModification,
            "sparql.update",
            "u2",
        ))
        .expect("log 2");

    let raw = buf.lock().unwrap_or_else(|p| p.into_inner()).clone();
    let text = std::str::from_utf8(&raw).expect("valid utf8");

    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "must be 2 NDJSON lines");

    // Each line must parse as a valid AuditEvent JSON object.
    let parsed_1: serde_json::Value = serde_json::from_str(lines[0]).expect("line 1 is valid JSON");
    assert_eq!(parsed_1["action"], "sparql.select");

    let parsed_2: serde_json::Value = serde_json::from_str(lines[1]).expect("line 2 is valid JSON");
    assert_eq!(parsed_2["action"], "sparql.update");
}

// ─────────────────────────────────────────────
// Test 10: JsonLineAuditLogger — to_file uses append semantics
// ─────────────────────────────────────────────

#[test]
fn test_json_line_logger_to_file_append() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("oxirs_audit_test_{}.ndjson", uuid::Uuid::new_v4()));

    // Write first event.
    {
        let logger = JsonLineAuditLogger::to_file(&path).expect("create logger");
        logger
            .log(make_event(AuditEventKind::System, "system.start", "sys"))
            .expect("log 1");
    }

    // Write second event in a new logger instance (simulates restart).
    {
        let logger = JsonLineAuditLogger::to_file(&path).expect("reopen logger");
        logger
            .log(make_event(AuditEventKind::System, "system.stop", "sys"))
            .expect("log 2");
    }

    let content = std::fs::read_to_string(&path).expect("read file");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2, "append mode must preserve both lines");

    // Clean up.
    let _ = std::fs::remove_file(&path);
}

// ─────────────────────────────────────────────
// Test 11: CompositeAuditLogger — fans out to all children
// ─────────────────────────────────────────────

#[test]
fn test_composite_logger_fans_out() {
    let l1: Arc<InMemoryAuditLogger> = Arc::new(InMemoryAuditLogger::new());
    let l2: Arc<InMemoryAuditLogger> = Arc::new(InMemoryAuditLogger::new());

    let composite = CompositeAuditLogger::new(vec![
        Arc::clone(&l1) as Arc<dyn AuditLogger>,
        Arc::clone(&l2) as Arc<dyn AuditLogger>,
    ]);

    composite
        .log(make_event(
            AuditEventKind::Authentication,
            "auth.login",
            "u1",
        ))
        .expect("composite log");

    assert_eq!(l1.len(), 1);
    assert_eq!(l2.len(), 1);
}

// ─────────────────────────────────────────────
// Test 12: AuditFilter — empty filter matches everything
// ─────────────────────────────────────────────

#[test]
fn test_audit_filter_empty_matches_all() {
    let filter = AuditFilter::default();
    let event = make_event(AuditEventKind::DataAccess, "sparql.select", "u1");
    assert!(filter.matches(&event));
}

// ─────────────────────────────────────────────
// Test 13: AuditFilter — kind filter
// ─────────────────────────────────────────────

#[test]
fn test_audit_filter_by_kind() {
    let filter = AuditFilter {
        kind: Some(AuditEventKind::Authentication),
        ..Default::default()
    };

    let auth_event = make_event(AuditEventKind::Authentication, "auth.login", "u1");
    let data_event = make_event(AuditEventKind::DataAccess, "sparql.select", "u1");

    assert!(filter.matches(&auth_event));
    assert!(!filter.matches(&data_event));
}

// ─────────────────────────────────────────────
// Test 14: AuditFilter — actor_id filter
// ─────────────────────────────────────────────

#[test]
fn test_audit_filter_by_actor_id() {
    let filter = AuditFilter {
        actor_id: Some("alice".to_string()),
        ..Default::default()
    };

    let alice = make_event(AuditEventKind::DataAccess, "sparql.select", "alice");
    let bob = make_event(AuditEventKind::DataAccess, "sparql.select", "bob");

    assert!(filter.matches(&alice));
    assert!(!filter.matches(&bob));
}

// ─────────────────────────────────────────────
// Test 15: AuditFilter — time range filter
// ─────────────────────────────────────────────

#[test]
fn test_audit_filter_time_range() {
    use chrono::Duration;

    let now = Utc::now();
    let past = now - Duration::hours(1);
    let future = now + Duration::hours(1);

    let event = make_event(AuditEventKind::DataAccess, "sparql.select", "u1");

    let filter_ok = AuditFilter {
        from: Some(past),
        until: Some(future),
        ..Default::default()
    };
    assert!(filter_ok.matches(&event));

    let filter_before = AuditFilter {
        until: Some(past),
        ..Default::default()
    };
    assert!(!filter_before.matches(&event));

    let filter_after = AuditFilter {
        from: Some(future),
        ..Default::default()
    };
    assert!(!filter_after.matches(&event));
}

// ─────────────────────────────────────────────
// Test 16: AuditFilter — data_subject_id filter
// ─────────────────────────────────────────────

#[test]
fn test_audit_filter_by_data_subject() {
    let filter = AuditFilter {
        data_subject_id: Some("subject-42".to_string()),
        ..Default::default()
    };

    let matching = make_event(AuditEventKind::DataAccess, "sparql.select", "admin")
        .with_data_subject("subject-42");
    let non_matching = make_event(AuditEventKind::DataAccess, "sparql.select", "admin")
        .with_data_subject("subject-99");

    assert!(filter.matches(&matching));
    assert!(!filter.matches(&non_matching));
}

// ─────────────────────────────────────────────
// Test 17: AuditFilter — action_prefix filter
// ─────────────────────────────────────────────

#[test]
fn test_audit_filter_action_prefix() {
    let filter = AuditFilter {
        action_prefix: Some("sparql.".to_string()),
        ..Default::default()
    };

    let sparql = make_event(AuditEventKind::DataAccess, "sparql.select", "u1");
    let admin = make_event(AuditEventKind::Admin, "admin.user_create", "u1");

    assert!(filter.matches(&sparql));
    assert!(!filter.matches(&admin));
}

// ─────────────────────────────────────────────
// Test 18: AuditQuery — limit and offset on Vec
// ─────────────────────────────────────────────

#[test]
fn test_audit_query_limit_offset() {
    let events: Vec<AuditEvent> = (0u32..10)
        .map(|i| {
            make_event(
                AuditEventKind::DataAccess,
                "sparql.select",
                &format!("u{i}"),
            )
        })
        .collect();

    let q = AuditQuery {
        filter: AuditFilter::default(),
        limit: Some(3),
        offset: Some(2),
        sort: SortOrder::Ascending,
    };

    let result = events.query(&q);
    assert_eq!(result.len(), 3, "limit of 3 after offset of 2");
}

// ─────────────────────────────────────────────
// Test 19: AuditQuery — descending sort
// ─────────────────────────────────────────────

#[test]
fn test_audit_query_sort_descending() {
    use chrono::Duration;

    // Create events with distinct, known timestamps.
    let base = Utc::now();
    let mut e1 = make_event(AuditEventKind::DataAccess, "sparql.select", "u1");
    let mut e2 = make_event(AuditEventKind::DataAccess, "sparql.select", "u2");
    let mut e3 = make_event(AuditEventKind::DataAccess, "sparql.select", "u3");
    e1.timestamp = base - Duration::seconds(2);
    e2.timestamp = base - Duration::seconds(1);
    e3.timestamp = base;

    let events = vec![e1, e2, e3];

    let q = AuditQuery {
        sort: SortOrder::Descending,
        ..Default::default()
    };
    let result = events.query(&q);
    assert_eq!(result.len(), 3);
    assert!(
        result[0].timestamp > result[1].timestamp,
        "descending: first must be newer"
    );
    assert!(
        result[1].timestamp > result[2].timestamp,
        "descending: second must be newer than third"
    );
}

// ─────────────────────────────────────────────
// Test 20: GdprService::data_subject_report
// ─────────────────────────────────────────────

#[test]
fn test_gdpr_data_subject_report() {
    let events = vec![
        make_event(AuditEventKind::DataAccess, "sparql.select", "admin").with_data_subject("alice"),
        make_event(AuditEventKind::DataModification, "sparql.update", "admin")
            .with_data_subject("bob"),
        make_event(AuditEventKind::DataAccess, "sparql.select", "admin").with_data_subject("alice"),
    ];

    let report = GdprService::data_subject_report(&events, "alice");

    assert_eq!(report.data_subject_id, "alice");
    assert_eq!(report.event_count, 2);
    assert_eq!(report.events.len(), 2);
    // Non-alice events excluded.
    for e in &report.events {
        assert_eq!(e.data_subject_id.as_deref(), Some("alice"));
    }
}

// ─────────────────────────────────────────────
// Test 21: GdprService::pseudonymise — modifies matching, not others
// ─────────────────────────────────────────────

#[test]
fn test_gdpr_pseudonymise_selective() {
    let mut events = vec![
        make_event(AuditEventKind::DataAccess, "sparql.select", "alice-actor")
            .with_data_subject("alice"),
        make_event(AuditEventKind::DataAccess, "sparql.select", "bob-actor")
            .with_data_subject("bob"),
    ];

    let count = GdprService::pseudonymise(&mut events, "alice");
    assert_eq!(count, 1, "only alice's event should be pseudonymised");

    // Alice's fields replaced.
    let alice_ev = &events[0];
    assert_eq!(alice_ev.actor.actor_id, "[redacted]");
    assert_eq!(alice_ev.actor.ip_address.as_deref(), Some("[redacted]"));
    assert_eq!(alice_ev.actor.session_id.as_deref(), Some("[redacted]"));
    assert_eq!(alice_ev.data_subject_id.as_deref(), Some("[redacted]"));

    // Bob's event untouched.
    let bob_ev = &events[1];
    assert_eq!(bob_ev.actor.actor_id, "bob-actor");
    assert_ne!(bob_ev.actor.ip_address.as_deref(), Some("[redacted]"));
    assert_eq!(bob_ev.data_subject_id.as_deref(), Some("bob"));
}

// ─────────────────────────────────────────────
// Test 22: GdprService::pseudonymise — no matching events returns 0
// ─────────────────────────────────────────────

#[test]
fn test_gdpr_pseudonymise_no_match() {
    let mut events = vec![
        make_event(AuditEventKind::DataAccess, "sparql.select", "u1").with_data_subject("charlie"),
    ];
    let count = GdprService::pseudonymise(&mut events, "alice");
    assert_eq!(count, 0);
    // Charlie's event is bit-identical.
    assert_eq!(events[0].actor.actor_id, "u1");
}

// ─────────────────────────────────────────────
// Test 23: AuditOutcome serde round-trip
// ─────────────────────────────────────────────

#[test]
fn test_audit_outcome_serde_round_trip() {
    let outcomes = vec![
        AuditOutcome::Success,
        AuditOutcome::Failure {
            reason: "permission denied".to_string(),
        },
        AuditOutcome::PartialSuccess {
            details: "3/5 graphs written".to_string(),
        },
    ];

    for outcome in outcomes {
        let json = serde_json::to_string(&outcome).expect("serialize");
        let back: AuditOutcome = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, outcome);
    }
}

// ─────────────────────────────────────────────
// Test 24: AuditEventKind exhaustive match (compile-time sanity)
// ─────────────────────────────────────────────

#[test]
fn test_audit_event_kind_exhaustive() {
    let kinds = [
        AuditEventKind::Authentication,
        AuditEventKind::Authorization,
        AuditEventKind::DataAccess,
        AuditEventKind::DataModification,
        AuditEventKind::Admin,
        AuditEventKind::Security,
        AuditEventKind::System,
    ];

    for kind in &kinds {
        // Ensure every variant is nameable and serialisable.
        let label = match kind {
            AuditEventKind::Authentication => "authentication",
            AuditEventKind::Authorization => "authorization",
            AuditEventKind::DataAccess => "data_access",
            AuditEventKind::DataModification => "data_modification",
            AuditEventKind::Admin => "admin",
            AuditEventKind::Security => "security",
            AuditEventKind::System => "system",
        };
        let json = serde_json::to_string(kind).expect("serialize kind");
        assert!(
            json.contains(label),
            "serialized kind must contain '{label}', got '{json}'"
        );
    }
}

// ─────────────────────────────────────────────
// Test 25: DataSubjectReport::to_json produces valid JSON
// ─────────────────────────────────────────────

#[test]
fn test_data_subject_report_to_json() {
    let events = vec![
        make_event(AuditEventKind::DataAccess, "sparql.select", "admin").with_data_subject("diana"),
    ];
    let report = GdprService::data_subject_report(&events, "diana");

    let json = report.to_json().expect("to_json must succeed");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("must be valid JSON");

    assert_eq!(parsed["data_subject_id"], "diana");
    assert_eq!(parsed["event_count"], 1);
    assert!(parsed["events"].is_array());
    assert!(parsed["generated_at"].is_string());
}

// ─────────────────────────────────────────────
// Test 26: AuditFilter — resource_id filter
// ─────────────────────────────────────────────

#[test]
fn test_audit_filter_by_resource_id() {
    let filter = AuditFilter {
        resource_id: Some("ds-main".to_string()),
        ..Default::default()
    };

    let event_main = AuditEvent::new(
        AuditEventKind::DataAccess,
        "sparql.select",
        make_actor("u1"),
        make_resource("ds-main", None),
        AuditOutcome::Success,
    );
    let event_other = AuditEvent::new(
        AuditEventKind::DataAccess,
        "sparql.select",
        make_actor("u1"),
        make_resource("ds-secondary", None),
        AuditOutcome::Success,
    );

    assert!(filter.matches(&event_main));
    assert!(!filter.matches(&event_other));
}

// ─────────────────────────────────────────────
// Test 27: AuditFilter — tenant_id filter
// ─────────────────────────────────────────────

#[test]
fn test_audit_filter_by_tenant_id() {
    let filter = AuditFilter {
        tenant_id: Some("acme".to_string()),
        ..Default::default()
    };

    let acme_event = make_event(AuditEventKind::DataAccess, "sparql.select", "u1");
    // default make_resource has tenant="acme"
    assert!(filter.matches(&acme_event));

    let other_event = AuditEvent::new(
        AuditEventKind::DataAccess,
        "sparql.select",
        make_actor("u1"),
        make_resource("ds-main", Some("globex")),
        AuditOutcome::Success,
    );
    assert!(!filter.matches(&other_event));
}

// ─────────────────────────────────────────────
// Test 28: InMemoryAuditLogger used as AuditQueryable via events()
// ─────────────────────────────────────────────

#[test]
fn test_in_memory_logger_queryable_integration() {
    let logger = InMemoryAuditLogger::new();
    for i in 0u32..5 {
        logger
            .log(make_event(
                AuditEventKind::DataAccess,
                "sparql.select",
                &format!("u{i}"),
            ))
            .expect("log");
    }
    logger
        .log(make_event(AuditEventKind::Admin, "admin.create", "admin"))
        .expect("log admin");

    let snap = logger.events();
    let q = AuditQuery {
        filter: AuditFilter {
            kind: Some(AuditEventKind::DataAccess),
            ..Default::default()
        },
        limit: Some(3),
        ..Default::default()
    };
    let result = snap.query(&q);
    assert_eq!(result.len(), 3);
    for e in &result {
        assert_eq!(e.kind, AuditEventKind::DataAccess);
    }
}

// ─────────────────────────────────────────────
// Test 29: CompositeAuditLogger — still delivers to second logger on first error
// ─────────────────────────────────────────────

#[test]
fn test_composite_logger_delivers_all_on_partial_error() {
    // l1 has capacity 0 → always fails.
    let l1: Arc<InMemoryAuditLogger> = Arc::new(InMemoryAuditLogger::with_capacity(0));
    let l2: Arc<InMemoryAuditLogger> = Arc::new(InMemoryAuditLogger::new());

    let composite = CompositeAuditLogger::new(vec![
        Arc::clone(&l1) as Arc<dyn AuditLogger>,
        Arc::clone(&l2) as Arc<dyn AuditLogger>,
    ]);

    let result = composite.log(make_event(
        AuditEventKind::Authentication,
        "auth.login",
        "u1",
    ));

    // l1 failed → composite returns an error.
    assert!(result.is_err(), "composite must propagate l1's error");
    // l2 still received the event.
    assert_eq!(l2.len(), 1, "l2 must receive the event despite l1 failure");
}

// ─────────────────────────────────────────────
// Test 30: GdprService::data_subject_report — empty result for unknown subject
// ─────────────────────────────────────────────

#[test]
fn test_gdpr_report_empty_for_unknown_subject() {
    let events = vec![
        make_event(AuditEventKind::DataAccess, "sparql.select", "admin").with_data_subject("alice"),
    ];
    let report = GdprService::data_subject_report(&events, "nobody");
    assert_eq!(report.event_count, 0);
    assert!(report.events.is_empty());
    assert_eq!(report.data_subject_id, "nobody");
}
