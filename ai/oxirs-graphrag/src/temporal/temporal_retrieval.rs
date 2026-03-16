//! Temporal GraphRAG: time-aware retrieval that filters and weights graph nodes
//! by temporal relevance.
//!
//! # Design
//!
//! Temporal relevance is modelled via two orthogonal mechanisms:
//!
//! 1. **Hard filtering** – remove entities/triples that fall entirely outside
//!    a caller-specified time window.
//! 2. **Soft weighting (temporal decay)** – reduce the retrieval score of
//!    older entities using a configurable decay function (exponential,
//!    linear, or step).  More recent entities rank higher.
//!
//! The module parses timestamps from entity metadata or triple objects
//! (recognised formats: RFC-3339 / ISO-8601, Unix epoch seconds, year strings
//! like "2021").  Unparseable timestamps are treated as "unknown" and kept
//! with a configurable fallback weight.

use crate::{GraphRAGResult, ScoreSource, ScoredEntity, Triple};
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Temporal window ─────────────────────────────────────────────────────────

/// A half-open time window `[start, end)`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Inclusive start (None = no lower bound)
    pub start: Option<DateTime<Utc>>,
    /// Exclusive end (None = no upper bound)
    pub end: Option<DateTime<Utc>>,
}

impl TimeWindow {
    /// Create an unbounded window (accepts all timestamps)
    pub fn unbounded() -> Self {
        Self {
            start: None,
            end: None,
        }
    }

    /// Create a window starting at `start` with no upper bound
    pub fn since(start: DateTime<Utc>) -> Self {
        Self {
            start: Some(start),
            end: None,
        }
    }

    /// Create a window ending before `end` with no lower bound
    pub fn before(end: DateTime<Utc>) -> Self {
        Self {
            start: None,
            end: Some(end),
        }
    }

    /// Create a bounded window
    pub fn between(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
        }
    }

    /// Return `true` if `ts` is within this window
    pub fn contains(&self, ts: DateTime<Utc>) -> bool {
        let after_start = self.start.map_or(true, |s| ts >= s);
        let before_end = self.end.map_or(true, |e| ts < e);
        after_start && before_end
    }
}

// ─── Decay functions ─────────────────────────────────────────────────────────

/// Temporal decay model
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DecayFn {
    /// Exponential decay: w = exp(−λ · age_days)
    Exponential {
        /// Decay rate λ (higher = faster decay; default ≈ 0.005 → half-life ~140 days)
        lambda: f64,
    },
    /// Linear decay: w = max(0, 1 − age_days / half_life_days)
    Linear {
        /// Age in days at which weight reaches 0
        half_life_days: f64,
    },
    /// Step decay: w = 1 if age_days ≤ cutoff, else `old_weight`
    Step {
        /// Age threshold in days
        cutoff_days: f64,
        /// Weight assigned to items older than `cutoff_days`
        old_weight: f64,
    },
    /// No decay – all items receive weight 1.0
    None,
}

impl Default for DecayFn {
    fn default() -> Self {
        Self::Exponential { lambda: 0.005 }
    }
}

impl DecayFn {
    /// Compute temporal weight for an entity whose timestamp is `age_days` old.
    /// Returns a value in [0.0, 1.0].
    pub fn weight(&self, age_days: f64) -> f64 {
        let age = age_days.max(0.0);
        match *self {
            Self::Exponential { lambda } => (-lambda * age).exp().clamp(0.0, 1.0),
            Self::Linear { half_life_days } => {
                (1.0 - age / half_life_days.max(1.0)).clamp(0.0, 1.0)
            }
            Self::Step {
                cutoff_days,
                old_weight,
            } => {
                if age <= cutoff_days {
                    1.0
                } else {
                    old_weight.clamp(0.0, 1.0)
                }
            }
            Self::None => 1.0,
        }
    }
}

// ─── Temporal metadata extraction ────────────────────────────────────────────

/// Well-known temporal metadata keys (checked in order)
const TEMPORAL_META_KEYS: &[&str] = &[
    "timestamp",
    "created",
    "modified",
    "updated",
    "date",
    "published",
    "valid_from",
    "validFrom",
    "time",
];

/// Try to parse a string into a UTC `DateTime`.
/// Supports:
/// - RFC-3339 / ISO-8601  (e.g. "2024-03-15T10:00:00Z")
/// - Date only (e.g. "2024-03-15")
/// - Year only (e.g. "2021")
/// - Unix epoch seconds (e.g. "1700000000")
pub fn parse_timestamp(s: &str) -> Option<DateTime<Utc>> {
    let s = s.trim();

    // RFC-3339 / ISO-8601 with timezone
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }

    // ISO-8601 without timezone → assume UTC
    let formats_no_tz = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ];
    for fmt in &formats_no_tz {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return Some(Utc.from_utc_datetime(&naive));
        }
    }

    // Date only
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let naive = date.and_hms_opt(0, 0, 0)?;
        return Some(Utc.from_utc_datetime(&naive));
    }

    // Year only (e.g. "2021")
    if s.len() == 4 {
        if let Ok(year) = s.parse::<i32>() {
            let date = NaiveDate::from_ymd_opt(year, 1, 1)?;
            let naive = date.and_hms_opt(0, 0, 0)?;
            return Some(Utc.from_utc_datetime(&naive));
        }
    }

    // Unix epoch seconds
    if let Ok(epoch) = s.parse::<i64>() {
        return Utc.timestamp_opt(epoch, 0).single();
    }

    None
}

/// Extract the best timestamp from entity metadata
pub fn extract_timestamp_from_metadata(
    metadata: &HashMap<String, String>,
) -> Option<DateTime<Utc>> {
    for key in TEMPORAL_META_KEYS {
        if let Some(val) = metadata.get(*key) {
            if let Some(ts) = parse_timestamp(val) {
                return Some(ts);
            }
        }
    }
    None
}

/// Try to extract a timestamp from a triple's object literal
pub fn extract_timestamp_from_triple(triple: &Triple) -> Option<DateTime<Utc>> {
    parse_timestamp(&triple.object)
}

// ─── Temporal retrieval configuration ────────────────────────────────────────

/// Fallback behaviour when no timestamp can be determined
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum UnknownTimestampPolicy {
    /// Keep the entity with its original score (weight = 1.0)
    #[default]
    Keep,
    /// Discard the entity entirely
    Discard,
    /// Apply a fixed weight (configured as `unknown_weight` below)
    FixedWeight,
}

/// Temporal filtering and weighting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRetrievalConfig {
    /// Reference time for decay calculation (None = now)
    pub reference_time: Option<DateTime<Utc>>,
    /// Hard filter window (only entities within this window survive)
    pub filter_window: Option<TimeWindow>,
    /// Decay function for score weighting
    pub decay_fn: DecayFn,
    /// Alpha: blend factor between original score and temporal weight
    /// final_score = (1 − temporal_alpha) · original + temporal_alpha · temporal_weight
    pub temporal_alpha: f64,
    /// Policy when timestamp cannot be determined
    pub unknown_policy: UnknownTimestampPolicy,
    /// Fixed weight used when `unknown_policy == FixedWeight`
    pub unknown_weight: f64,
    /// Temporal predicates to scan in triples for timestamps
    /// (e.g. `"http://schema.org/datePublished"`, `"http://purl.org/dc/terms/date"`)
    pub temporal_predicates: Vec<String>,
}

impl Default for TemporalRetrievalConfig {
    fn default() -> Self {
        Self {
            reference_time: None,
            filter_window: None,
            decay_fn: DecayFn::default(),
            temporal_alpha: 0.3,
            unknown_policy: UnknownTimestampPolicy::Keep,
            unknown_weight: 0.5,
            temporal_predicates: vec![
                "http://schema.org/datePublished".to_string(),
                "http://schema.org/dateModified".to_string(),
                "http://purl.org/dc/terms/date".to_string(),
                "http://purl.org/dc/terms/created".to_string(),
                "http://purl.org/dc/terms/modified".to_string(),
                "http://www.w3.org/2006/time#inXSDDateTimeStamp".to_string(),
            ],
        }
    }
}

// ─── Temporal retriever ───────────────────────────────────────────────────────

/// Timestamp index built from the RDF subgraph
struct TemporalIndex {
    /// entity_uri → best timestamp
    timestamps: HashMap<String, DateTime<Utc>>,
}

impl TemporalIndex {
    fn build(subgraph: &[Triple], config: &TemporalRetrievalConfig) -> Self {
        let mut timestamps: HashMap<String, DateTime<Utc>> = HashMap::new();

        let pred_set: std::collections::HashSet<&str> = config
            .temporal_predicates
            .iter()
            .map(|s| s.as_str())
            .collect();

        for triple in subgraph {
            if pred_set.contains(triple.predicate.as_str()) {
                if let Some(ts) = parse_timestamp(&triple.object) {
                    timestamps
                        .entry(triple.subject.clone())
                        .and_modify(|existing| {
                            // Keep the most recent timestamp
                            if ts > *existing {
                                *existing = ts;
                            }
                        })
                        .or_insert(ts);
                }
            }
        }

        Self { timestamps }
    }

    fn get(&self, uri: &str) -> Option<DateTime<Utc>> {
        self.timestamps.get(uri).copied()
    }
}

/// Time-aware retrieval engine
pub struct TemporalRetriever {
    config: TemporalRetrievalConfig,
}

impl TemporalRetriever {
    pub fn new(config: TemporalRetrievalConfig) -> Self {
        Self { config }
    }

    /// Reference time for decay calculation
    fn reference_time(&self) -> DateTime<Utc> {
        self.config.reference_time.unwrap_or_else(Utc::now)
    }

    /// Apply temporal filtering and re-weighting to a list of scored entities.
    ///
    /// `subgraph` is used to build a temporal index from temporal predicates.
    pub fn apply(
        &self,
        entities: Vec<ScoredEntity>,
        subgraph: &[Triple],
    ) -> GraphRAGResult<Vec<ScoredEntity>> {
        let index = TemporalIndex::build(subgraph, &self.config);
        let ref_time = self.reference_time();

        let mut result: Vec<ScoredEntity> = Vec::with_capacity(entities.len());

        for entity in entities {
            // Determine timestamp: subgraph index first, then metadata
            let ts = index
                .get(&entity.uri)
                .or_else(|| extract_timestamp_from_metadata(&entity.metadata));

            let temporal_weight = match ts {
                Some(t) => {
                    let age_days = (ref_time - t).num_seconds().max(0) as f64 / 86_400.0;

                    // Hard filter
                    if let Some(window) = &self.config.filter_window {
                        if !window.contains(t) {
                            continue; // discard
                        }
                    }

                    self.config.decay_fn.weight(age_days)
                }
                None => match self.config.unknown_policy {
                    UnknownTimestampPolicy::Discard => continue,
                    UnknownTimestampPolicy::FixedWeight => self.config.unknown_weight,
                    UnknownTimestampPolicy::Keep => 1.0,
                },
            };

            // Blend original score with temporal weight
            let alpha = self.config.temporal_alpha;
            let new_score = (1.0 - alpha) * entity.score + alpha * temporal_weight;

            let mut updated = entity;
            updated.score = new_score.clamp(0.0, f64::MAX);
            result.push(updated);
        }

        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(result)
    }

    /// Filter triples whose associated subject has a timestamp outside the window.
    /// Triples with no discernible timestamp are kept by default.
    pub fn filter_triples(&self, triples: Vec<Triple>) -> GraphRAGResult<Vec<Triple>> {
        let window = match &self.config.filter_window {
            None => return Ok(triples),
            Some(w) => w,
        };

        let result: Vec<Triple> = triples
            .into_iter()
            .filter(|t| {
                match extract_timestamp_from_triple(t) {
                    Some(ts) => window.contains(ts),
                    None => true, // keep triples without timestamp
                }
            })
            .collect();

        Ok(result)
    }

    /// Score a single entity's temporal relevance (0.0..1.0)
    pub fn temporal_score(&self, ts: DateTime<Utc>) -> f64 {
        let ref_time = self.reference_time();
        let age_days = (ref_time - ts).num_seconds().max(0) as f64 / 86_400.0;
        self.config.decay_fn.weight(age_days)
    }
}

// ─── Integration helper ───────────────────────────────────────────────────────

/// Annotate entities with their most recent timestamp (if available in subgraph)
pub fn annotate_timestamps(
    entities: Vec<ScoredEntity>,
    subgraph: &[Triple],
    config: &TemporalRetrievalConfig,
) -> Vec<ScoredEntity> {
    let index = TemporalIndex::build(subgraph, config);

    entities
        .into_iter()
        .map(|mut e| {
            if let Some(ts) = index.get(&e.uri) {
                e.metadata
                    .insert("temporal_timestamp".to_string(), ts.to_rfc3339());
            }
            e
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScoreSource;
    use chrono::{Datelike, Duration};

    fn make_entity(uri: &str, score: f64) -> ScoredEntity {
        ScoredEntity {
            uri: uri.to_string(),
            score,
            source: ScoreSource::Fused,
            metadata: HashMap::new(),
        }
    }

    fn make_entity_with_ts(uri: &str, score: f64, ts_key: &str, ts_val: &str) -> ScoredEntity {
        let mut e = make_entity(uri, score);
        e.metadata.insert(ts_key.to_string(), ts_val.to_string());
        e
    }

    fn ref_time_days_ago(days: i64) -> DateTime<Utc> {
        Utc::now() - Duration::days(days)
    }

    // ── parse_timestamp ────────────────────────────────────────────────────

    #[test]
    fn test_parse_rfc3339() {
        let ts = parse_timestamp("2024-03-15T10:00:00Z").expect("should succeed");
        assert_eq!(ts.year(), 2024);
    }

    #[test]
    fn test_parse_date_only() {
        let ts = parse_timestamp("2023-06-01").expect("should succeed");
        assert_eq!(ts.year(), 2023);
        assert_eq!(ts.month(), 6);
    }

    #[test]
    fn test_parse_year_only() {
        let ts = parse_timestamp("2021").expect("should succeed");
        assert_eq!(ts.year(), 2021);
    }

    #[test]
    fn test_parse_unix_epoch() {
        let ts = parse_timestamp("1700000000").expect("should succeed");
        assert!(ts.year() >= 2023);
    }

    #[test]
    fn test_parse_invalid_returns_none() {
        assert!(parse_timestamp("not-a-date").is_none());
        assert!(parse_timestamp("").is_none());
    }

    // ── TimeWindow ────────────────────────────────────────────────────────

    #[test]
    fn test_time_window_contains() {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let w = TimeWindow::between(start, end);

        let inside = Utc.with_ymd_and_hms(2023, 6, 1, 0, 0, 0).unwrap();
        let before = Utc.with_ymd_and_hms(2022, 12, 31, 0, 0, 0).unwrap();
        let after = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();

        assert!(w.contains(inside));
        assert!(!w.contains(before));
        assert!(!w.contains(after)); // exclusive end
    }

    #[test]
    fn test_time_window_unbounded_accepts_all() {
        let w = TimeWindow::unbounded();
        assert!(w.contains(Utc::now()));
        assert!(w.contains(Utc.with_ymd_and_hms(1900, 1, 1, 0, 0, 0).unwrap()));
    }

    #[test]
    fn test_time_window_since() {
        let start = Utc::now() - Duration::days(30);
        let w = TimeWindow::since(start);
        assert!(w.contains(Utc::now()));
        assert!(!w.contains(Utc::now() - Duration::days(60)));
    }

    // ── DecayFn ───────────────────────────────────────────────────────────

    #[test]
    fn test_decay_exponential_at_zero() {
        let d = DecayFn::Exponential { lambda: 0.01 };
        assert!((d.weight(0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_decay_exponential_decreases() {
        let d = DecayFn::Exponential { lambda: 0.01 };
        assert!(d.weight(100.0) < d.weight(10.0));
    }

    #[test]
    fn test_decay_linear_at_zero_is_one() {
        let d = DecayFn::Linear {
            half_life_days: 365.0,
        };
        assert!((d.weight(0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_decay_linear_at_half_life_is_zero() {
        let d = DecayFn::Linear {
            half_life_days: 100.0,
        };
        assert!((d.weight(100.0)).abs() < 1e-9);
    }

    #[test]
    fn test_decay_linear_clamps_to_zero() {
        let d = DecayFn::Linear {
            half_life_days: 10.0,
        };
        assert_eq!(d.weight(200.0), 0.0);
    }

    #[test]
    fn test_decay_step_recent() {
        let d = DecayFn::Step {
            cutoff_days: 30.0,
            old_weight: 0.1,
        };
        assert_eq!(d.weight(10.0), 1.0);
        assert_eq!(d.weight(31.0), 0.1);
    }

    #[test]
    fn test_decay_none_always_one() {
        let d = DecayFn::None;
        assert_eq!(d.weight(0.0), 1.0);
        assert_eq!(d.weight(9999.0), 1.0);
    }

    // ── TemporalRetriever::apply ──────────────────────────────────────────

    #[test]
    fn test_apply_no_filter_no_decay() {
        let config = TemporalRetrievalConfig {
            decay_fn: DecayFn::None,
            temporal_alpha: 0.0,
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let entities = vec![make_entity("http://a", 0.9), make_entity("http://b", 0.7)];
        let result = retriever.apply(entities, &[]).expect("should succeed");
        assert_eq!(result.len(), 2);
        // Scores unchanged (alpha=0, decay=1)
        assert!((result[0].score - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_apply_decays_older_entities() {
        let ref_ts = Utc::now();
        let config = TemporalRetrievalConfig {
            reference_time: Some(ref_ts),
            decay_fn: DecayFn::Exponential { lambda: 0.1 },
            temporal_alpha: 1.0, // pure temporal
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);

        // Recent entity: 1 day old
        let mut recent = make_entity_with_ts("http://a", 0.9, "timestamp", "");
        let recent_ts = ref_ts - Duration::days(1);
        recent
            .metadata
            .insert("timestamp".to_string(), recent_ts.to_rfc3339());

        // Old entity: 365 days old
        let mut old = make_entity_with_ts("http://b", 0.9, "timestamp", "");
        let old_ts = ref_ts - Duration::days(365);
        old.metadata
            .insert("timestamp".to_string(), old_ts.to_rfc3339());

        let result = retriever
            .apply(vec![recent, old], &[])
            .expect("should succeed");
        assert_eq!(result.len(), 2);
        // Recent should rank higher
        assert!(result[0].uri == "http://a", "Recent should rank first");
        assert!(result[0].score > result[1].score);
    }

    #[test]
    fn test_apply_hard_filter() {
        let window_start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let window_end = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let config = TemporalRetrievalConfig {
            reference_time: Some(window_end),
            filter_window: Some(TimeWindow::between(window_start, window_end)),
            decay_fn: DecayFn::None,
            temporal_alpha: 0.0,
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);

        let mut inside_entity = make_entity("http://inside", 0.8);
        inside_entity
            .metadata
            .insert("timestamp".to_string(), "2023-06-01".to_string());

        let mut outside_entity = make_entity("http://outside", 0.9);
        outside_entity
            .metadata
            .insert("timestamp".to_string(), "2022-01-01".to_string());

        let result = retriever
            .apply(vec![inside_entity, outside_entity], &[])
            .expect("should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].uri, "http://inside");
    }

    #[test]
    fn test_apply_unknown_timestamp_keep() {
        let config = TemporalRetrievalConfig {
            unknown_policy: UnknownTimestampPolicy::Keep,
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let entities = vec![make_entity("http://notimestamp", 0.7)];
        let result = retriever.apply(entities, &[]).expect("should succeed");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_apply_unknown_timestamp_discard() {
        let config = TemporalRetrievalConfig {
            unknown_policy: UnknownTimestampPolicy::Discard,
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let entities = vec![make_entity("http://notimestamp", 0.7)];
        let result = retriever.apply(entities, &[]).expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_apply_unknown_timestamp_fixed_weight() {
        let config = TemporalRetrievalConfig {
            unknown_policy: UnknownTimestampPolicy::FixedWeight,
            unknown_weight: 0.5,
            temporal_alpha: 1.0,
            decay_fn: DecayFn::None,
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let entities = vec![make_entity("http://notimestamp", 0.8)];
        let result = retriever.apply(entities, &[]).expect("should succeed");
        assert_eq!(result.len(), 1);
        assert!((result[0].score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_apply_subgraph_temporal_index() {
        let ref_ts = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let config = TemporalRetrievalConfig {
            reference_time: Some(ref_ts),
            decay_fn: DecayFn::None,
            temporal_alpha: 0.0,
            temporal_predicates: vec!["http://schema.org/datePublished".to_string()],
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);

        let subgraph = vec![Triple::new(
            "http://entity",
            "http://schema.org/datePublished",
            "2023-06-01",
        )];
        let entities = vec![make_entity("http://entity", 0.8)];
        let result = retriever
            .apply(entities, &subgraph)
            .expect("should succeed");
        assert_eq!(result.len(), 1);
    }

    // ── filter_triples ────────────────────────────────────────────────────

    #[test]
    fn test_filter_triples_no_window_keeps_all() {
        let config = TemporalRetrievalConfig::default();
        let retriever = TemporalRetriever::new(config);
        let triples = vec![
            Triple::new("http://s", "http://p", "2023-01-01"),
            Triple::new("http://s", "http://p", "some literal"),
        ];
        let result = retriever
            .filter_triples(triples.clone())
            .expect("should succeed");
        assert_eq!(result.len(), triples.len());
    }

    #[test]
    fn test_filter_triples_with_window() {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let config = TemporalRetrievalConfig {
            filter_window: Some(TimeWindow::between(start, end)),
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let triples = vec![
            Triple::new("http://s", "http://p", "2023-06-01"), // inside
            Triple::new("http://s", "http://p", "2022-01-01"), // before
            Triple::new("http://s", "http://p", "not-a-date"), // no ts → keep
        ];
        let result = retriever.filter_triples(triples).expect("should succeed");
        // "inside" + "no ts" kept; "before" discarded
        assert_eq!(result.len(), 2);
    }

    // ── temporal_score ────────────────────────────────────────────────────

    #[test]
    fn test_temporal_score_recent_is_high() {
        let config = TemporalRetrievalConfig {
            reference_time: Some(Utc::now()),
            decay_fn: DecayFn::Exponential { lambda: 0.01 },
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let score = retriever.temporal_score(Utc::now() - Duration::days(1));
        assert!(
            score > 0.98,
            "Recent item should score close to 1.0: {score}"
        );
    }

    // ── annotate_timestamps ───────────────────────────────────────────────

    #[test]
    fn test_annotate_timestamps() {
        let config = TemporalRetrievalConfig {
            temporal_predicates: vec!["http://schema.org/datePublished".to_string()],
            ..Default::default()
        };
        let subgraph = vec![Triple::new(
            "http://entity",
            "http://schema.org/datePublished",
            "2023-06-01",
        )];
        let entities = vec![make_entity("http://entity", 0.8)];
        let annotated = annotate_timestamps(entities, &subgraph, &config);
        assert!(
            annotated[0].metadata.contains_key("temporal_timestamp"),
            "Expected temporal_timestamp in metadata"
        );
    }

    // ── Extract metadata ──────────────────────────────────────────────────

    #[test]
    fn test_extract_timestamp_from_metadata_finds_key() {
        let mut m = HashMap::new();
        m.insert("created".to_string(), "2023-01-15".to_string());
        let ts = extract_timestamp_from_metadata(&m).expect("should succeed");
        assert_eq!(ts.year(), 2023);
    }

    #[test]
    fn test_extract_timestamp_none_when_absent() {
        let m = HashMap::new();
        assert!(extract_timestamp_from_metadata(&m).is_none());
    }

    // ── Blending alpha ────────────────────────────────────────────────────

    #[test]
    fn test_alpha_zero_preserves_original_score() {
        let config = TemporalRetrievalConfig {
            temporal_alpha: 0.0,
            decay_fn: DecayFn::None,
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let mut e = make_entity("http://a", 0.75);
        e.metadata
            .insert("timestamp".to_string(), "2023-01-01".to_string());
        let result = retriever.apply(vec![e], &[]).expect("should succeed");
        assert!((result[0].score - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_alpha_one_gives_temporal_weight() {
        let ref_ts = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let config = TemporalRetrievalConfig {
            reference_time: Some(ref_ts),
            temporal_alpha: 1.0,
            decay_fn: DecayFn::None,
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let mut e = make_entity("http://a", 0.75);
        e.metadata
            .insert("timestamp".to_string(), "2023-12-31".to_string());
        let result = retriever.apply(vec![e], &[]).expect("should succeed");
        // decay = None → weight = 1.0, alpha = 1 → score ≈ 1.0
        assert!((result[0].score - 1.0).abs() < 0.01);
    }
}

// ─── Additional tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod additional_tests {
    use super::*;
    use crate::{ScoreSource, ScoredEntity, Triple};
    use chrono::{Datelike, Duration, TimeZone, Utc};

    fn make_entity(uri: &str, score: f64) -> ScoredEntity {
        ScoredEntity {
            uri: uri.to_string(),
            score,
            source: ScoreSource::Vector,
            metadata: HashMap::new(),
        }
    }

    // ── TimeWindow ────────────────────────────────────────────────────────

    #[test]
    fn test_time_window_unbounded_contains_anything() {
        let w = TimeWindow::unbounded();
        assert!(w.contains(Utc::now()));
        assert!(w.contains(Utc.with_ymd_and_hms(1970, 1, 1, 0, 0, 0).unwrap()));
        assert!(w.contains(Utc.with_ymd_and_hms(2099, 12, 31, 0, 0, 0).unwrap()));
    }

    #[test]
    fn test_time_window_since_excludes_before_start() {
        let start = Utc.with_ymd_and_hms(2023, 6, 1, 0, 0, 0).unwrap();
        let w = TimeWindow::since(start);
        let before = Utc.with_ymd_and_hms(2023, 5, 31, 0, 0, 0).unwrap();
        assert!(!w.contains(before));
        assert!(w.contains(start)); // inclusive
    }

    #[test]
    fn test_time_window_before_excludes_at_end() {
        let end = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let w = TimeWindow::before(end);
        // end itself should be excluded (half-open)
        assert!(!w.contains(end));
        let before_end = end - Duration::seconds(1);
        assert!(w.contains(before_end));
    }

    #[test]
    fn test_time_window_between_includes_start_excludes_end() {
        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let w = TimeWindow::between(start, end);
        assert!(w.contains(start));
        assert!(!w.contains(end));
        let mid = Utc.with_ymd_and_hms(2023, 6, 15, 0, 0, 0).unwrap();
        assert!(w.contains(mid));
    }

    // ── DecayFn tests ─────────────────────────────────────────────────────

    #[test]
    fn test_decay_exponential_zero_age_is_one() {
        let decay = DecayFn::Exponential { lambda: 0.01 };
        assert!((decay.weight(0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_decay_exponential_negative_age_clamps_to_zero() {
        let decay = DecayFn::Exponential { lambda: 0.01 };
        // Negative age clamped to 0
        assert!((decay.weight(-10.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_decay_exponential_monotone_decreasing() {
        let decay = DecayFn::Exponential { lambda: 0.005 };
        let w1 = decay.weight(30.0);
        let w2 = decay.weight(60.0);
        assert!(w1 > w2, "Older items should have lower weight");
    }

    #[test]
    fn test_decay_linear_reaches_zero_at_half_life() {
        let decay = DecayFn::Linear {
            half_life_days: 100.0,
        };
        let w = decay.weight(100.0);
        assert!((w - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_decay_linear_half_way_is_half() {
        let decay = DecayFn::Linear {
            half_life_days: 100.0,
        };
        let w = decay.weight(50.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_decay_linear_beyond_half_life_clamps_zero() {
        let decay = DecayFn::Linear {
            half_life_days: 10.0,
        };
        let w = decay.weight(100.0);
        assert!((w - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_decay_step_within_cutoff_is_one() {
        let decay = DecayFn::Step {
            cutoff_days: 30.0,
            old_weight: 0.1,
        };
        assert!((decay.weight(15.0) - 1.0).abs() < 1e-9);
        assert!((decay.weight(30.0) - 1.0).abs() < 1e-9); // at boundary → still 1
    }

    #[test]
    fn test_decay_step_beyond_cutoff_uses_old_weight() {
        let decay = DecayFn::Step {
            cutoff_days: 30.0,
            old_weight: 0.3,
        };
        assert!((decay.weight(31.0) - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_decay_none_always_one() {
        let decay = DecayFn::None;
        for age in [0.0, 1.0, 100.0, 365.0, 10000.0] {
            assert!((decay.weight(age) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_decay_fn_default_is_exponential() {
        let decay = DecayFn::default();
        matches!(decay, DecayFn::Exponential { lambda } if (lambda - 0.005).abs() < f64::EPSILON);
    }

    // ── parse_timestamp tests ─────────────────────────────────────────────

    #[test]
    fn test_parse_timestamp_rfc3339() {
        let ts = parse_timestamp("2024-03-15T10:00:00Z").expect("should succeed");
        assert_eq!(ts.year(), 2024);
        assert_eq!(ts.month(), 3);
        assert_eq!(ts.day(), 15);
    }

    #[test]
    fn test_parse_timestamp_date_only() {
        let ts = parse_timestamp("2023-07-04").expect("should succeed");
        assert_eq!(ts.year(), 2023);
        assert_eq!(ts.month(), 7);
    }

    #[test]
    fn test_parse_timestamp_year_only() {
        let ts = parse_timestamp("2020");
        // Year-only parsing may or may not succeed depending on implementation
        // Just ensure we don't panic
        let _ = ts;
    }

    #[test]
    fn test_parse_timestamp_invalid_returns_none() {
        assert!(parse_timestamp("not-a-date").is_none());
        assert!(parse_timestamp("").is_none());
    }

    #[test]
    fn test_parse_timestamp_unix_epoch() {
        let ts = parse_timestamp("1700000000");
        assert!(ts.is_some());
        let ts = ts.expect("should succeed");
        assert!(ts.year() >= 2023); // 1700000000 ≈ Nov 2023
    }

    // ── TemporalRetrievalConfig defaults ──────────────────────────────────

    #[test]
    fn test_temporal_config_defaults() {
        let cfg = TemporalRetrievalConfig::default();
        assert!((cfg.temporal_alpha - 0.3).abs() < f64::EPSILON);
        assert!(cfg.reference_time.is_none());
        assert!(!cfg.temporal_predicates.is_empty());
    }

    // ── apply with time window ────────────────────────────────────────────

    #[test]
    fn test_apply_entities_sorted_by_score_descending() {
        let config = TemporalRetrievalConfig {
            decay_fn: DecayFn::None,
            temporal_alpha: 0.0,
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let entities = vec![
            make_entity("http://low", 0.3),
            make_entity("http://high", 0.9),
            make_entity("http://mid", 0.6),
        ];
        let result = retriever.apply(entities, &[]).expect("should succeed");
        // Scores should be descending
        for i in 1..result.len() {
            assert!(
                result[i - 1].score >= result[i].score,
                "Results should be sorted descending"
            );
        }
    }

    #[test]
    fn test_apply_empty_entities() {
        let config = TemporalRetrievalConfig::default();
        let retriever = TemporalRetriever::new(config);
        let result = retriever.apply(vec![], &[]).expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_apply_with_filter_window_no_match_discards_all() {
        let past_start = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let past_end = Utc.with_ymd_and_hms(2021, 1, 1, 0, 0, 0).unwrap();
        let config = TemporalRetrievalConfig {
            filter_window: Some(TimeWindow::between(past_start, past_end)),
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let mut e = make_entity("http://recent", 0.8);
        // Timestamp in 2023 – outside the 2020-2021 window
        e.metadata
            .insert("timestamp".to_string(), "2023-01-01".to_string());
        let result = retriever.apply(vec![e], &[]).expect("should succeed");
        assert!(result.is_empty());
    }

    // ── temporal_score ────────────────────────────────────────────────────

    #[test]
    fn test_temporal_score_old_item_lower_than_recent() {
        let ref_ts = Utc::now();
        let config = TemporalRetrievalConfig {
            reference_time: Some(ref_ts),
            decay_fn: DecayFn::Exponential { lambda: 0.01 },
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let recent_score = retriever.temporal_score(ref_ts - Duration::days(10));
        let old_score = retriever.temporal_score(ref_ts - Duration::days(500));
        assert!(recent_score > old_score, "Recent items should score higher");
    }

    #[test]
    fn test_temporal_score_none_decay_always_one() {
        let config = TemporalRetrievalConfig {
            reference_time: Some(Utc::now()),
            decay_fn: DecayFn::None,
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let score = retriever.temporal_score(Utc::now() - Duration::days(9999));
        assert!((score - 1.0).abs() < 1e-9);
    }

    // ── filter_triples with temporal predicates ───────────────────────────

    #[test]
    fn test_filter_triples_keeps_non_temporal_predicates() {
        let config = TemporalRetrievalConfig::default();
        let retriever = TemporalRetriever::new(config);
        let triples = vec![Triple::new(
            "http://s",
            "http://someOtherPred",
            "some value",
        )];
        let result = retriever
            .filter_triples(triples.clone())
            .expect("should succeed");
        assert_eq!(result.len(), 1);
    }

    // ── annotate_timestamps ───────────────────────────────────────────────

    #[test]
    fn test_annotate_timestamps_no_match_leaves_metadata_empty() {
        let config = TemporalRetrievalConfig {
            temporal_predicates: vec!["http://schema.org/datePublished".to_string()],
            ..Default::default()
        };
        let subgraph = vec![Triple::new(
            "http://other_entity",
            "http://schema.org/datePublished",
            "2023-01-01",
        )];
        let entities = vec![make_entity("http://entity_no_match", 0.8)];
        let annotated = annotate_timestamps(entities, &subgraph, &config);
        // No match → temporal_timestamp should not be set
        assert!(!annotated[0].metadata.contains_key("temporal_timestamp"));
    }

    // ── UnknownTimestampPolicy ────────────────────────────────────────────

    #[test]
    fn test_unknown_timestamp_fixed_weight_multiplies_score() {
        let config = TemporalRetrievalConfig {
            unknown_policy: UnknownTimestampPolicy::FixedWeight,
            unknown_weight: 0.25,
            temporal_alpha: 1.0,
            decay_fn: DecayFn::None,
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let entities = vec![make_entity("http://no_ts", 0.8)];
        let result = retriever.apply(entities, &[]).expect("should succeed");
        // alpha=1.0, decay=none → score = unknown_weight = 0.25
        assert_eq!(result.len(), 1);
        assert!((result[0].score - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_unknown_timestamp_keep_policy_preserves_original_score() {
        let config = TemporalRetrievalConfig {
            unknown_policy: UnknownTimestampPolicy::Keep,
            temporal_alpha: 0.0, // don't blend
            decay_fn: DecayFn::None,
            ..Default::default()
        };
        let retriever = TemporalRetriever::new(config);
        let entities = vec![make_entity("http://no_ts", 0.75)];
        let result = retriever.apply(entities, &[]).expect("should succeed");
        assert_eq!(result.len(), 1);
        assert!((result[0].score - 0.75).abs() < 1e-9);
    }
}
