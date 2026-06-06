//! Audit log export HTTP handlers for OxiRS Fuseki.
//!
//! Provides structured, authenticated export of the in-memory audit trail
//! accumulated by `oxirs_core::audit::InMemoryAuditLogger`.
//!
//! # Endpoints
//!
//! - `GET /$/audit/log`       — filtered event export (JSON / JSONL / CSV)
//! - `GET /$/audit/log/stats` — aggregate statistics over all stored events
//!
//! # Access control
//!
//! Both endpoints require the caller to hold either:
//! - `Permission::ReadAudit`, **or**
//! - `Permission::Admin` / `Permission::GlobalAdmin`
//!
//! The `Authorization: Bearer <token>` header is the only supported credential
//! path (same as all other protected endpoints in this crate).  If the header
//! is absent the handler returns HTTP 401; if the permission check fails it
//! returns HTTP 403.
//!
//! # Query parameters for `GET /$/audit/log`
//!
//! | Parameter | Type   | Description |
//! |-----------|--------|-------------|
//! | `from`    | string | ISO-8601 lower bound (inclusive) |
//! | `to`      | string | ISO-8601 upper bound (inclusive) |
//! | `actor`   | string | Filter by `actor.actor_id` |
//! | `action`  | string | Filter: event `action` must start with this prefix |
//! | `limit`   | usize  | Cap number of returned events (default: all) |
//! | `format`  | string | `json` (default), `jsonl`, or `csv` |

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use chrono::{DateTime, Utc};
use oxirs_core::audit::AuditEvent;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::auth::types::Permission;
use crate::server::AppState;

// ─────────────────────────────────────────────────────────────────────────────
// Query / request types
// ─────────────────────────────────────────────────────────────────────────────

/// Query parameters accepted by `GET /$/audit/log`.
#[derive(Debug, Deserialize)]
pub struct AuditQuery {
    /// ISO-8601 lower bound timestamp (inclusive).
    pub from: Option<String>,
    /// ISO-8601 upper bound timestamp (inclusive).
    pub to: Option<String>,
    /// Filter by actor identifier (exact match).
    pub actor: Option<String>,
    /// Filter by action prefix (e.g. `"sparql."` matches `"sparql.select"`).
    pub action: Option<String>,
    /// Maximum number of events to return (default: all).
    pub limit: Option<usize>,
    /// Output format: `json` (default), `jsonl`, or `csv`.
    pub format: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Output format enum
// ─────────────────────────────────────────────────────────────────────────────

/// Supported export formats for the audit log endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditFormat {
    /// RFC-8259 JSON array — `Content-Type: application/json`.
    Json,
    /// Newline-delimited JSON (NDJSON) — `Content-Type: application/x-ndjson`.
    Jsonl,
    /// RFC-4180 comma-separated values — `Content-Type: text/csv`.
    Csv,
}

impl AuditFormat {
    /// Parse the `format` query-parameter value (case-insensitive).
    ///
    /// Returns `None` if the value is unrecognised.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "json" => Some(Self::Json),
            "jsonl" | "ndjson" => Some(Self::Jsonl),
            "csv" => Some(Self::Csv),
            _ => None,
        }
    }

    /// MIME type string for the `Content-Type` response header.
    pub fn content_type(self) -> &'static str {
        match self {
            Self::Json => "application/json",
            Self::Jsonl => "application/x-ndjson",
            Self::Csv => "text/csv; charset=utf-8",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Response types
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics over all stored audit events.
#[derive(Debug, Serialize)]
pub struct AuditStats {
    /// Total number of events in the logger.
    pub total_events: usize,
    /// Number of events per high-level kind (e.g. `"data_access"`, `"admin"`).
    pub events_by_kind: HashMap<String, usize>,
    /// Number of events per unique actor.
    pub events_by_actor: HashMap<String, usize>,
    /// Number of events per unique action label.
    pub events_by_action: HashMap<String, usize>,
    /// Timestamp of the oldest stored event, if any.
    pub oldest_event_ts: Option<DateTime<Utc>>,
    /// Timestamp of the newest stored event, if any.
    pub newest_event_ts: Option<DateTime<Utc>>,
    /// Count of events whose outcome is `Success`.
    pub success_count: usize,
    /// Count of events whose outcome is `Failure`.
    pub failure_count: usize,
    /// Count of events whose outcome is `PartialSuccess`.
    pub partial_success_count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: parse ISO-8601 timestamp
// ─────────────────────────────────────────────────────────────────────────────

/// Parse an ISO-8601 string into a UTC `DateTime`.
///
/// Accepts both RFC-3339 (with time zone) and the common `YYYY-MM-DDTHH:MM:SS`
/// format by appending `Z` when no time-zone offset is present.
pub fn parse_ts(s: &str) -> Result<DateTime<Utc>, String> {
    // Try full RFC-3339 first.
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.with_timezone(&Utc));
    }
    // Attempt naive datetime with an appended "Z".
    let s_with_z = if s.contains('T') {
        format!("{}Z", s.trim_end_matches('Z'))
    } else {
        // Date-only "YYYY-MM-DD" → midnight UTC.
        format!("{}T00:00:00Z", s)
    };
    DateTime::parse_from_rfc3339(&s_with_z)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| format!("invalid timestamp '{}': {}", s, e))
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: filter events
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the caller-supplied filter parameters to a snapshot of audit events.
///
/// Filtering is ANDed: every non-`None` parameter must match.
pub fn filter_events(
    events: Vec<AuditEvent>,
    from: Option<DateTime<Utc>>,
    to: Option<DateTime<Utc>>,
    actor: Option<&str>,
    action_prefix: Option<&str>,
) -> Vec<AuditEvent> {
    events
        .into_iter()
        .filter(|e| {
            if let Some(from_ts) = from {
                if e.timestamp < from_ts {
                    return false;
                }
            }
            if let Some(to_ts) = to {
                if e.timestamp > to_ts {
                    return false;
                }
            }
            if let Some(actor_id) = actor {
                if e.actor.actor_id != actor_id {
                    return false;
                }
            }
            if let Some(prefix) = action_prefix {
                if !e.action.starts_with(prefix) {
                    return false;
                }
            }
            true
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: apply limit (most-recent N events)
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a maximum-count cap, returning the *most recent* `n` events.
///
/// Events are assumed to be in chronological order (oldest first); this
/// function retains the tail.
pub fn limit_recent(mut events: Vec<AuditEvent>, limit: Option<usize>) -> Vec<AuditEvent> {
    if let Some(n) = limit {
        let len = events.len();
        if len > n {
            events.drain(0..(len - n));
        }
    }
    events
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: format serialisers
// ─────────────────────────────────────────────────────────────────────────────

/// Serialise events as a RFC-8259 JSON array.
pub fn format_as_json(events: &[AuditEvent]) -> Result<String, String> {
    serde_json::to_string(events).map_err(|e| format!("JSON serialisation error: {}", e))
}

/// Serialise events as newline-delimited JSON (one object per line).
pub fn format_as_jsonl(events: &[AuditEvent]) -> Result<String, String> {
    let mut buf = String::new();
    for event in events {
        let line = serde_json::to_string(event)
            .map_err(|e| format!("JSONL serialisation error: {}", e))?;
        buf.push_str(&line);
        buf.push('\n');
    }
    Ok(buf)
}

/// Serialise events as RFC-4180 CSV with a header row.
///
/// String fields containing commas, double-quotes, or newlines are enclosed in
/// double-quotes and internal double-quotes are doubled (RFC-4180 §2).
pub fn format_as_csv(events: &[AuditEvent]) -> Result<String, String> {
    let mut buf = String::new();
    // Header row.
    buf.push_str(
        "event_id,timestamp,kind,action,actor_id,actor_type,resource_type,resource_id,outcome,duration_ms\r\n",
    );
    for e in events {
        let outcome_str = match &e.outcome {
            oxirs_core::audit::AuditOutcome::Success => "success".to_string(),
            oxirs_core::audit::AuditOutcome::Failure { reason } => {
                format!("failure:{}", reason)
            }
            oxirs_core::audit::AuditOutcome::PartialSuccess { details } => {
                format!("partial_success:{}", details)
            }
        };
        let kind_str = format!("{:?}", e.kind).to_ascii_lowercase();
        let actor_type_str = format!("{:?}", e.actor.actor_type).to_ascii_lowercase();
        let duration_str = e.duration_ms.map(|d| d.to_string()).unwrap_or_default();

        let fields: [&str; 10] = [
            &e.event_id,
            &e.timestamp.to_rfc3339(),
            &kind_str,
            &e.action,
            &e.actor.actor_id,
            &actor_type_str,
            &e.resource.resource_type,
            &e.resource.resource_id,
            &outcome_str,
            &duration_str,
        ];

        let row: Vec<String> = fields.iter().map(|f| csv_escape(f)).collect();
        buf.push_str(&row.join(","));
        buf.push_str("\r\n");
    }
    Ok(buf)
}

/// Escape a single CSV field per RFC-4180.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute aggregate statistics over a slice of events.
pub fn compute_stats(events: &[AuditEvent]) -> AuditStats {
    let mut events_by_kind: HashMap<String, usize> = HashMap::new();
    let mut events_by_actor: HashMap<String, usize> = HashMap::new();
    let mut events_by_action: HashMap<String, usize> = HashMap::new();
    let mut oldest: Option<DateTime<Utc>> = None;
    let mut newest: Option<DateTime<Utc>> = None;
    let mut success_count = 0usize;
    let mut failure_count = 0usize;
    let mut partial_success_count = 0usize;

    for e in events {
        // Use serde's snake_case rename so the key matches the JSON wire
        // representation of `AuditEventKind` (e.g. `DataAccess` → `data_access`).
        let kind_str = serde_json::to_value(&e.kind)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_else(|| format!("{:?}", e.kind).to_ascii_lowercase());
        *events_by_kind.entry(kind_str).or_insert(0) += 1;
        *events_by_actor.entry(e.actor.actor_id.clone()).or_insert(0) += 1;
        *events_by_action.entry(e.action.clone()).or_insert(0) += 1;

        let ts = e.timestamp;
        oldest = Some(oldest.map_or(ts, |prev| prev.min(ts)));
        newest = Some(newest.map_or(ts, |prev| prev.max(ts)));

        match &e.outcome {
            oxirs_core::audit::AuditOutcome::Success => success_count += 1,
            oxirs_core::audit::AuditOutcome::Failure { .. } => failure_count += 1,
            oxirs_core::audit::AuditOutcome::PartialSuccess { .. } => partial_success_count += 1,
        }
    }

    AuditStats {
        total_events: events.len(),
        events_by_kind,
        events_by_actor,
        events_by_action,
        oldest_event_ts: oldest,
        newest_event_ts: newest,
        success_count,
        failure_count,
        partial_success_count,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: permission check
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the bearer token from the `Authorization` header.
fn extract_bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| {
            s.strip_prefix("Bearer ")
                .or_else(|| s.strip_prefix("bearer "))
        })
}

/// Perform the audit-read permission check.
///
/// Returns `Ok(())` if the request is authorised, or an HTTP error response
/// that should be returned directly to the caller.
///
/// The error variant is boxed because `Response` is large (>128 bytes) and
/// boxing keeps the `Result` size small on the success path.
pub fn check_audit_permission(state: &AppState, headers: &HeaderMap) -> Result<(), Box<Response>> {
    // If auth is not required by configuration, allow all callers.
    if !state.config.security.auth_required {
        debug!("Auth not required — allowing audit log access");
        return Ok(());
    }

    let token = match extract_bearer(headers) {
        Some(t) => t,
        None => {
            warn!("Audit log request rejected: missing Authorization header");
            return Err(Box::new(error_response(
                StatusCode::UNAUTHORIZED,
                "missing_auth",
                "Authorization: Bearer <token> header is required",
            )));
        }
    };

    let auth_service = match &state.auth_service {
        Some(svc) => svc,
        None => {
            warn!("Audit log request rejected: auth service unavailable");
            return Err(Box::new(error_response(
                StatusCode::UNAUTHORIZED,
                "auth_unavailable",
                "Authentication service is not configured",
            )));
        }
    };

    let validation = match auth_service.validate_jwt_token(token) {
        Ok(v) => v,
        Err(_) => {
            warn!("Audit log request rejected: invalid or expired token");
            return Err(Box::new(error_response(
                StatusCode::UNAUTHORIZED,
                "invalid_token",
                "Token is invalid or expired",
            )));
        }
    };

    let has_permission = validation.user.permissions.contains(&Permission::ReadAudit)
        || validation.user.permissions.contains(&Permission::Admin)
        || validation
            .user
            .permissions
            .contains(&Permission::GlobalAdmin)
        || validation
            .user
            .roles
            .iter()
            .any(|r| r == "admin" || r == "auditor");

    if !has_permission {
        warn!(
            "Audit log access denied for user '{}'",
            validation.user.username
        );
        return Err(Box::new(error_response(
            StatusCode::FORBIDDEN,
            "permission_denied",
            "ReadAudit permission (or admin role) is required to access audit logs",
        )));
    }

    Ok(())
}

/// Build a plain-JSON error response body.
fn error_response(status: StatusCode, code: &str, message: &str) -> Response {
    let body = serde_json::json!({
        "error": code,
        "message": message,
    });
    let json_body = match serde_json::to_string(&body) {
        Ok(s) => s,
        Err(_) => format!(r#"{{"error":"{}","message":"{}"}}"#, code, message),
    };
    let mut resp = Response::new(json_body.into());
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    resp
}

// ─────────────────────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────────────────────

/// `GET /$/audit/log` — export filtered audit events.
///
/// Requires `Permission::ReadAudit` or admin role.
///
/// Returns a body in the requested format (JSON, JSONL, or CSV) with an
/// appropriate `Content-Type` header.
pub async fn get_audit_log(
    State(state): State<Arc<AppState>>,
    Query(params): Query<AuditQuery>,
    headers: HeaderMap,
) -> impl IntoResponse {
    // 1. Authorisation.
    if let Err(resp) = check_audit_permission(&state, &headers) {
        return *resp;
    }

    // 2. Resolve output format; default to JSON.
    let s = params.format.as_deref().unwrap_or("json");
    let fmt = match AuditFormat::parse(s) {
        Some(f) => f,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_format",
                &format!("unknown format '{}'; supported values: json, jsonl, csv", s),
            );
        }
    };

    // 3. Parse optional timestamp bounds.
    let from_ts = match params.from.as_deref().map(parse_ts).transpose() {
        Ok(v) => v,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, "invalid_from_timestamp", &e);
        }
    };
    let to_ts = match params.to.as_deref().map(parse_ts).transpose() {
        Ok(v) => v,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, "invalid_to_timestamp", &e);
        }
    };

    // 4. Retrieve events from the logger.
    let raw_events: Vec<AuditEvent> = state.audit_logger.events();

    // 5. Apply filters.
    let filtered = filter_events(
        raw_events,
        from_ts,
        to_ts,
        params.actor.as_deref(),
        params.action.as_deref(),
    );

    // 6. Apply limit (most-recent N).
    let limited = limit_recent(filtered, params.limit);

    // 7. Serialise.
    let body = match fmt {
        AuditFormat::Json => format_as_json(&limited),
        AuditFormat::Jsonl => format_as_jsonl(&limited),
        AuditFormat::Csv => format_as_csv(&limited),
    };

    let body = match body {
        Ok(s) => s,
        Err(e) => {
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, "serialisation_error", &e);
        }
    };

    // 8. Build response with correct Content-Type.
    let content_type = fmt.content_type();
    let mut resp = Response::new(body.into());
    resp.headers_mut().insert(
        "content-type",
        match HeaderValue::from_str(content_type) {
            Ok(v) => v,
            Err(_) => HeaderValue::from_static("application/octet-stream"),
        },
    );
    resp
}

/// `GET /$/audit/log/stats` — aggregate statistics over all stored events.
///
/// Requires `Permission::ReadAudit` or admin role.
pub async fn get_audit_stats(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    // 1. Authorisation.
    if let Err(resp) = check_audit_permission(&state, &headers) {
        return *resp;
    }

    // 2. Snapshot all events.
    let events = state.audit_logger.events();

    // 3. Compute and serialise statistics.
    let stats = compute_stats(&events);
    match serde_json::to_string(&stats) {
        Ok(body) => {
            let mut resp = Response::new(body.into());
            resp.headers_mut()
                .insert("content-type", HeaderValue::from_static("application/json"));
            resp
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "serialisation_error",
            &e.to_string(),
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Datelike as _;
    use chrono::Timelike as _;
    use chrono::Utc;
    use oxirs_core::audit::{
        event::ActorType, AuditActor, AuditEvent, AuditEventKind, AuditLogger as CoreAuditLogger,
        AuditOutcome, AuditResource, InMemoryAuditLogger,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_event(
        kind: AuditEventKind,
        action: &str,
        actor_id: &str,
        outcome: AuditOutcome,
    ) -> AuditEvent {
        AuditEvent::new(
            kind,
            action,
            AuditActor {
                actor_id: actor_id.to_string(),
                actor_type: ActorType::User,
                ip_address: Some("127.0.0.1".to_string()),
                session_id: None,
            },
            AuditResource {
                resource_type: "dataset".to_string(),
                resource_id: "ds-main".to_string(),
                tenant_id: None,
            },
            outcome,
        )
    }

    fn make_logger_with_events(n: usize) -> InMemoryAuditLogger {
        let logger = InMemoryAuditLogger::new();
        for i in 0..n {
            let kind = if i % 2 == 0 {
                AuditEventKind::DataAccess
            } else {
                AuditEventKind::Admin
            };
            let outcome = if i % 3 == 0 {
                AuditOutcome::Failure {
                    reason: "test".to_string(),
                }
            } else {
                AuditOutcome::Success
            };
            logger
                .log(make_event(kind, &format!("action.{}", i), "alice", outcome))
                .expect("log should succeed");
        }
        logger
    }

    // -----------------------------------------------------------------------
    // parse_ts
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_ts_rfc3339() {
        let dt = parse_ts("2026-01-15T12:00:00Z").expect("valid RFC-3339");
        assert_eq!(dt.year(), 2026);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 15);
    }

    #[test]
    fn test_parse_ts_with_offset() {
        let dt = parse_ts("2026-05-17T09:30:00+09:00").expect("with TZ offset");
        // Converted to UTC: 09:30 JST = 00:30 UTC
        assert_eq!(dt.hour(), 0);
        assert_eq!(dt.minute(), 30);
    }

    #[test]
    fn test_parse_ts_date_only() {
        let dt = parse_ts("2026-05-17").expect("date-only");
        assert_eq!(dt.year(), 2026);
        assert_eq!(dt.month(), 5);
        assert_eq!(dt.day(), 17);
        assert_eq!(dt.hour(), 0);
    }

    #[test]
    fn test_parse_ts_naive_no_z() {
        let dt = parse_ts("2026-03-10T08:00:00").expect("naive datetime");
        assert_eq!(dt.year(), 2026);
    }

    #[test]
    fn test_parse_ts_invalid() {
        assert!(parse_ts("not-a-date").is_err());
        assert!(parse_ts("2026-99-99").is_err());
    }

    // -----------------------------------------------------------------------
    // AuditFormat
    // -----------------------------------------------------------------------

    #[test]
    fn test_audit_format_parse() {
        assert_eq!(AuditFormat::parse("json"), Some(AuditFormat::Json));
        assert_eq!(AuditFormat::parse("JSON"), Some(AuditFormat::Json));
        assert_eq!(AuditFormat::parse("jsonl"), Some(AuditFormat::Jsonl));
        assert_eq!(AuditFormat::parse("ndjson"), Some(AuditFormat::Jsonl));
        assert_eq!(AuditFormat::parse("csv"), Some(AuditFormat::Csv));
        assert_eq!(AuditFormat::parse("unknown"), None);
    }

    #[test]
    fn test_audit_format_content_type() {
        assert_eq!(AuditFormat::Json.content_type(), "application/json");
        assert!(AuditFormat::Jsonl.content_type().contains("ndjson"));
        assert!(AuditFormat::Csv.content_type().contains("text/csv"));
    }

    // -----------------------------------------------------------------------
    // filter_events
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_events_no_filter_passes_all() {
        let logger = make_logger_with_events(10);
        let events = logger.events();
        let result = filter_events(events.clone(), None, None, None, None);
        assert_eq!(result.len(), events.len());
    }

    #[test]
    fn test_filter_events_by_actor() {
        let logger = InMemoryAuditLogger::new();
        logger
            .log(make_event(
                AuditEventKind::DataAccess,
                "sparql.select",
                "alice",
                AuditOutcome::Success,
            ))
            .expect("log");
        logger
            .log(make_event(
                AuditEventKind::DataAccess,
                "sparql.select",
                "bob",
                AuditOutcome::Success,
            ))
            .expect("log");

        let events = logger.events();
        let result = filter_events(events, None, None, Some("alice"), None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].actor.actor_id, "alice");
    }

    #[test]
    fn test_filter_events_by_action_prefix() {
        let logger = InMemoryAuditLogger::new();
        logger
            .log(make_event(
                AuditEventKind::DataAccess,
                "sparql.select",
                "alice",
                AuditOutcome::Success,
            ))
            .expect("log");
        logger
            .log(make_event(
                AuditEventKind::Admin,
                "admin.config",
                "alice",
                AuditOutcome::Success,
            ))
            .expect("log");

        let events = logger.events();
        let result = filter_events(events, None, None, None, Some("sparql."));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].action, "sparql.select");
    }

    #[test]
    fn test_filter_events_by_timestamp_range() {
        let logger = InMemoryAuditLogger::new();
        // Force specific timestamps by using with_duration as a proxy —
        // events will all have Utc::now() timestamps.  We just verify that
        // from > any possible timestamp drops all events.
        logger
            .log(make_event(
                AuditEventKind::DataAccess,
                "sparql.select",
                "alice",
                AuditOutcome::Success,
            ))
            .expect("log");

        let events = logger.events();
        let future = Utc::now() + chrono::Duration::hours(1);
        let result = filter_events(events, Some(future), None, None, None);
        assert_eq!(result.len(), 0, "from in future should exclude all events");
    }

    // -----------------------------------------------------------------------
    // limit_recent
    // -----------------------------------------------------------------------

    #[test]
    fn test_limit_recent_no_limit() {
        let logger = make_logger_with_events(20);
        let events = logger.events();
        let result = limit_recent(events.clone(), None);
        assert_eq!(result.len(), events.len());
    }

    #[test]
    fn test_limit_recent_smaller_than_total() {
        let logger = make_logger_with_events(20);
        let events = logger.events();
        let result = limit_recent(events, Some(5));
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_limit_recent_larger_than_total() {
        let logger = make_logger_with_events(5);
        let events = logger.events();
        let result = limit_recent(events, Some(100));
        assert_eq!(result.len(), 5);
    }

    // -----------------------------------------------------------------------
    // format_as_json
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_as_json_empty() {
        let out = format_as_json(&[]).expect("serialise empty");
        assert_eq!(out, "[]");
    }

    #[test]
    fn test_format_as_json_is_valid_json_array() {
        let logger = make_logger_with_events(3);
        let events = logger.events();
        let out = format_as_json(&events).expect("serialise");
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().map(|a| a.len()), Some(3));
    }

    // -----------------------------------------------------------------------
    // format_as_jsonl
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_as_jsonl_empty() {
        let out = format_as_jsonl(&[]).expect("serialise empty");
        assert!(out.is_empty());
    }

    #[test]
    fn test_format_as_jsonl_line_count() {
        let logger = make_logger_with_events(4);
        let events = logger.events();
        let out = format_as_jsonl(&events).expect("serialise");
        assert_eq!(out.lines().count(), 4);
        // Each line must be a valid JSON object.
        for line in out.lines() {
            let v: serde_json::Value = serde_json::from_str(line).expect("valid JSON per line");
            assert!(v.is_object());
        }
    }

    // -----------------------------------------------------------------------
    // format_as_csv
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_as_csv_has_header() {
        let out = format_as_csv(&[]).expect("serialise empty");
        assert!(out.contains("event_id,timestamp,kind,action"));
    }

    #[test]
    fn test_format_as_csv_row_count() {
        let logger = make_logger_with_events(5);
        let events = logger.events();
        let out = format_as_csv(&events).expect("serialise");
        // Header + 5 data rows; each row ends with CRLF.
        let lines: Vec<&str> = out.split("\r\n").filter(|s| !s.is_empty()).collect();
        assert_eq!(lines.len(), 6); // 1 header + 5 data
    }

    #[test]
    fn test_csv_escape_no_special_chars() {
        assert_eq!(csv_escape("hello"), "hello");
    }

    #[test]
    fn test_csv_escape_with_comma() {
        let esc = csv_escape("hello,world");
        assert_eq!(esc, "\"hello,world\"");
    }

    #[test]
    fn test_csv_escape_with_quote() {
        let esc = csv_escape("say \"hi\"");
        assert_eq!(esc, "\"say \"\"hi\"\"\"");
    }

    // -----------------------------------------------------------------------
    // compute_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_stats_empty() {
        let stats = compute_stats(&[]);
        assert_eq!(stats.total_events, 0);
        assert!(stats.oldest_event_ts.is_none());
        assert!(stats.newest_event_ts.is_none());
        assert_eq!(stats.success_count, 0);
        assert_eq!(stats.failure_count, 0);
    }

    #[test]
    fn test_compute_stats_counts() {
        let logger = make_logger_with_events(9);
        let events = logger.events();
        let stats = compute_stats(&events);
        assert_eq!(stats.total_events, 9);
        // 9 events split between DataAccess (even indices) and Admin (odd)
        // → 5 DataAccess (0,2,4,6,8) + 4 Admin (1,3,5,7)
        assert_eq!(
            stats
                .events_by_kind
                .get("data_access")
                .copied()
                .unwrap_or(0),
            5
        );
        assert_eq!(stats.events_by_kind.get("admin").copied().unwrap_or(0), 4);
        // success/failure: failure at indices 0,3,6 → 3 failures, 6 successes
        assert_eq!(stats.failure_count, 3);
        assert_eq!(stats.success_count, 6);
    }

    #[test]
    fn test_compute_stats_timestamps_present() {
        let logger = make_logger_with_events(3);
        let events = logger.events();
        let stats = compute_stats(&events);
        assert!(stats.oldest_event_ts.is_some());
        assert!(stats.newest_event_ts.is_some());
        // oldest <= newest
        if let (Some(oldest), Some(newest)) = (stats.oldest_event_ts, stats.newest_event_ts) {
            assert!(oldest <= newest);
        }
    }

    // -----------------------------------------------------------------------
    // extract_bearer
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_bearer_present() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static("Bearer mytoken123"),
        );
        assert_eq!(extract_bearer(&headers), Some("mytoken123"));
    }

    #[test]
    fn test_extract_bearer_missing() {
        let headers = HeaderMap::new();
        assert_eq!(extract_bearer(&headers), None);
    }

    #[test]
    fn test_extract_bearer_lowercase_prefix() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("bearer tok42"));
        assert_eq!(extract_bearer(&headers), Some("tok42"));
    }

    #[test]
    fn test_extract_bearer_not_bearer_scheme() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static("Basic dXNlcjpwYXNz"),
        );
        assert_eq!(extract_bearer(&headers), None);
    }
}
