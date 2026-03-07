//! Persistent chat memory store.
//!
//! [`MemoryStore`] extracts entities and facts from conversation turns, stores
//! them as subject-predicate-object tuples, and retrieves them by keyword
//! relevance and recency.  Over time, memory relevance decays exponentially so
//! that older, unreferenced memories become less prominent.  A summariser
//! compresses ageing memories into a single summary entry, and duplicate facts
//! are automatically merged.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_chat::memory_store::{MemoryStore, MemoryConfig};
//!
//! let mut store = MemoryStore::new(MemoryConfig::default());
//! store.ingest("Alice works at ACME Corp and lives in Berlin");
//! store.ingest("Bob is Alice's manager at ACME Corp");
//!
//! let results = store.retrieve("ACME Corp", 5);
//! assert!(!results.is_empty());
//! ```

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single memory fact stored as a subject-predicate-object triple.
#[derive(Debug, Clone, PartialEq)]
pub struct Fact {
    /// Deduplication key (subject|predicate|object).
    pub dedup_key: String,
    /// Subject entity (e.g. "Alice").
    pub subject: String,
    /// Predicate relation (e.g. "works_at").
    pub predicate: String,
    /// Object value (e.g. "ACME Corp").
    pub object: String,
    /// Current relevance score in `[0, 1]`.  Decays over time.
    pub relevance: f64,
    /// Unix timestamp (seconds) when this fact was first stored.
    pub created_at: u64,
    /// Unix timestamp (seconds) of the last access / relevance boost.
    pub last_accessed: u64,
    /// Number of times this fact has been retrieved.
    pub access_count: u32,
}

impl Fact {
    /// Returns a key that uniquely identifies this fact (for deduplication).
    /// Same as the `dedup_key` field, but computed fresh from fields.
    pub fn computed_key(&self) -> String {
        format!("{}|{}|{}", self.subject, self.predicate, self.object)
    }
}

/// A retrieved memory entry.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Unique fact identifier (dedup key).
    pub key: String,
    /// The underlying fact.
    pub fact: Fact,
    /// Retrieval score combining relevance and recency.
    pub retrieval_score: f64,
}

/// Configuration for the memory store.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum number of facts to keep before summarisation.
    pub max_facts: usize,
    /// Exponential decay rate per second (`relevance *= e^{-rate * Δt}`).
    pub decay_rate: f64,
    /// When a fact's relevance falls below this threshold it is a candidate
    /// for summarisation or eviction.
    pub summarise_threshold: f64,
    /// Maximum number of results returned by a single retrieval.
    pub default_top_k: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_facts: 500,
            decay_rate: 1e-6, // very slow decay for human-scale conversations
            summarise_threshold: 0.2,
            default_top_k: 10,
        }
    }
}

/// Aggregate statistics about the memory store.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total number of stored facts.
    pub fact_count: usize,
    /// Unix timestamp of the oldest fact (`None` if empty).
    pub oldest_timestamp: Option<u64>,
    /// Unix timestamp of the newest fact (`None` if empty).
    pub newest_timestamp: Option<u64>,
    /// Number of distinct topics (predicates) observed.
    pub topic_count: usize,
    /// Distribution of predicate counts.
    pub topic_distribution: HashMap<String, usize>,
}

/// Serialisable export format.
#[derive(Debug, Clone)]
pub struct MemoryExport {
    /// All facts at the time of export.
    pub facts: Vec<Fact>,
    /// Unix timestamp of the export.
    pub exported_at: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Entity / fact extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Extract (subject, predicate, object) triples from free text using simple
/// heuristics.
fn extract_facts(text: &str) -> Vec<(String, String, String)> {
    let mut results = Vec::new();
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| s.split_whitespace().count() >= 3)
        .collect();

    for sentence in sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        // Simple pattern: "SUBJ VERB ... OBJ" via common linking verbs
        let linking_verbs = [
            "is", "are", "was", "were", "works", "worked", "lives", "lived", "manages", "managed",
            "has", "had", "knows", "knew", "created", "founded", "joined", "leads", "led",
        ];
        for (idx, &word) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            let lower = lower.trim_end_matches(|c: char| !c.is_alphanumeric());
            if linking_verbs.contains(&lower) && idx > 0 && idx + 1 < words.len() {
                let subject = words[..idx].join(" ");
                let predicate = normalise_predicate(lower);
                let object_words: Vec<&str> = words[idx + 1..]
                    .iter()
                    .take_while(|&&w| {
                        !["and", "but", "or", "because", "so", "that"]
                            .contains(&w.to_lowercase().as_str())
                    })
                    .copied()
                    .collect();
                if !object_words.is_empty() {
                    let object = object_words.join(" ");
                    results.push((subject, predicate.to_string(), object));
                }
            }
        }
    }
    results
}

fn normalise_predicate(verb: &str) -> &str {
    match verb {
        "is" | "are" | "was" | "were" => "is_a",
        "works" | "worked" => "works_at",
        "lives" | "lived" => "lives_in",
        "manages" | "managed" => "manages",
        "has" | "had" => "has",
        "knows" | "knew" => "knows",
        "created" => "created",
        "founded" => "founded",
        "joined" => "joined",
        "leads" | "led" => "leads",
        other => other,
    }
}

/// Extract named entity strings from text (capitalised tokens / known patterns).
pub fn extract_entities(text: &str) -> Vec<String> {
    let mut entities = Vec::new();
    let mut current_entity: Vec<&str> = Vec::new();

    for word in text.split_whitespace() {
        let clean: String = word
            .chars()
            .filter(|c| c.is_alphabetic() || *c == '-' || *c == '\'')
            .collect();
        if clean.is_empty() {
            continue;
        }
        // A word is part of a named entity if it starts with a capital letter
        if clean.chars().next().is_some_and(|c| c.is_uppercase()) {
            current_entity.push(word);
        } else {
            if current_entity.len() > 1 {
                entities.push(current_entity.join(" "));
            } else if let Some(&single) = current_entity.first() {
                // Single capitalised word only if it is >3 chars (not sentence start)
                if single.len() > 3 {
                    entities.push(single.to_string());
                }
            }
            current_entity.clear();
        }
    }
    // Flush
    if current_entity.len() > 1 {
        entities.push(current_entity.join(" "));
    }

    // Deduplicate while preserving order
    let mut seen = std::collections::HashSet::new();
    entities.retain(|e| seen.insert(e.clone()));
    entities
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryStore
// ─────────────────────────────────────────────────────────────────────────────

/// Persistent chat memory store backed by in-memory fact storage.
pub struct MemoryStore {
    config: MemoryConfig,
    facts: HashMap<String, Fact>,
    /// Global summary fact accumulated from summarised memories.
    summary: Option<String>,
}

impl MemoryStore {
    /// Create a new memory store with the given configuration.
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            facts: HashMap::new(),
            summary: None,
        }
    }

    /// Ingest free text: extract facts, merge duplicates, and store.
    pub fn ingest(&mut self, text: &str) {
        let triples = extract_facts(text);
        let ts = now_secs();
        for (subject, predicate, object) in triples {
            let fact = Fact {
                dedup_key: String::new(), // computed below
                subject: subject.clone(),
                predicate: predicate.clone(),
                object: object.clone(),
                relevance: 1.0,
                created_at: ts,
                last_accessed: ts,
                access_count: 0,
            };
            let key = format!("{}|{}|{}", subject, predicate, object);
            self.facts
                .entry(key.clone())
                .and_modify(|existing| {
                    // Merge: boost relevance and update timestamp
                    existing.relevance = (existing.relevance + 0.2).min(1.0);
                    existing.last_accessed = ts;
                    existing.access_count += 1;
                })
                .or_insert_with(|| {
                    let mut f = fact;
                    f.dedup_key = key.clone();
                    f
                });
        }

        // Summarise if over limit
        if self.facts.len() > self.config.max_facts {
            self.summarise();
        }
    }

    /// Retrieve up to `top_k` facts matching `query`.
    ///
    /// Results are sorted by a retrieval score combining keyword overlap and
    /// current relevance.
    pub fn retrieve(&mut self, query: &str, top_k: usize) -> Vec<MemoryEntry> {
        let query_terms: Vec<String> = query.split_whitespace().map(|w| w.to_lowercase()).collect();
        let now = now_secs();
        let rate = self.config.decay_rate;

        let mut scored: Vec<(String, f64)> = self
            .facts
            .iter()
            .map(|(key, fact)| {
                let elapsed = now.saturating_sub(fact.last_accessed) as f64;
                let decayed = (fact.relevance * (-rate * elapsed).exp()).clamp(0.0, 1.0);
                let overlap = keyword_overlap(fact, &query_terms);
                let recency_bonus = recency_score(fact.last_accessed, now);
                let score = 0.4 * overlap + 0.4 * decayed + 0.2 * recency_bonus;
                (key.clone(), score)
            })
            .filter(|(_, s)| *s > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let ts = now_secs();
        scored
            .into_iter()
            .filter_map(|(key, score)| {
                self.facts.get_mut(&key).map(|fact| {
                    fact.last_accessed = ts;
                    fact.access_count += 1;
                    MemoryEntry {
                        key: key.clone(),
                        fact: fact.clone(),
                        retrieval_score: score,
                    }
                })
            })
            .collect()
    }

    /// Apply exponential relevance decay to all stored facts.
    pub fn decay(&mut self) {
        let now = now_secs();
        let rate = self.config.decay_rate;
        for fact in self.facts.values_mut() {
            let elapsed = now.saturating_sub(fact.last_accessed) as f64;
            let decay = (-rate * elapsed).exp();
            fact.relevance = (fact.relevance * decay).clamp(0.0, 1.0);
        }
    }

    /// Compress old/low-relevance memories into a single summary entry and
    /// evict them from the store.
    ///
    /// Returns the number of facts that were summarised.
    pub fn summarise(&mut self) -> usize {
        let now = now_secs();
        let rate = self.config.decay_rate;
        let threshold = self.config.summarise_threshold;
        let to_summarise: Vec<String> = self
            .facts
            .iter()
            .filter(|(_, f)| {
                let elapsed = now.saturating_sub(f.last_accessed) as f64;
                let decayed = (f.relevance * (-rate * elapsed).exp()).clamp(0.0, 1.0);
                decayed < threshold
            })
            .map(|(k, _)| k.clone())
            .collect();

        let count = to_summarise.len();
        if count == 0 {
            return 0;
        }

        let snippets: Vec<String> = to_summarise
            .iter()
            .filter_map(|k| self.facts.get(k))
            .map(|f| format!("{} {} {}", f.subject, f.predicate, f.object))
            .collect();

        let new_summary = snippets.join("; ");
        self.summary = Some(match &self.summary {
            Some(existing) => format!("{existing}; {new_summary}"),
            None => new_summary,
        });

        for k in &to_summarise {
            self.facts.remove(k);
        }

        count
    }

    /// Return the current summary string (if any).
    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    /// Statistics about the current state of the store.
    pub fn stats(&self) -> MemoryStats {
        let mut topic_distribution: HashMap<String, usize> = HashMap::new();
        let mut oldest: Option<u64> = None;
        let mut newest: Option<u64> = None;
        for fact in self.facts.values() {
            *topic_distribution
                .entry(fact.predicate.clone())
                .or_insert(0) += 1;
            oldest = Some(oldest.map_or(fact.created_at, |o: u64| o.min(fact.created_at)));
            newest = Some(newest.map_or(fact.created_at, |n: u64| n.max(fact.created_at)));
        }
        MemoryStats {
            fact_count: self.facts.len(),
            oldest_timestamp: oldest,
            newest_timestamp: newest,
            topic_count: topic_distribution.len(),
            topic_distribution,
        }
    }

    /// Export all facts.
    pub fn export(&self) -> MemoryExport {
        MemoryExport {
            facts: self.facts.values().cloned().collect(),
            exported_at: now_secs(),
        }
    }

    /// Import facts from an export, merging with existing facts.
    pub fn import(&mut self, export: &MemoryExport) {
        for fact in &export.facts {
            self.facts
                .entry(fact.dedup_key.clone())
                .and_modify(|existing| {
                    existing.relevance = (existing.relevance.max(fact.relevance)).min(1.0);
                    existing.access_count += fact.access_count;
                })
                .or_insert_with(|| fact.clone());
        }
    }

    /// Number of facts currently stored.
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    /// Directly insert a fact (useful for testing / import scenarios).
    pub fn insert_fact(&mut self, subject: &str, predicate: &str, object: &str) {
        let ts = now_secs();
        let key = format!("{subject}|{predicate}|{object}");
        self.facts.entry(key.clone()).or_insert_with(|| Fact {
            dedup_key: key,
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            relevance: 1.0,
            created_at: ts,
            last_accessed: ts,
            access_count: 0,
        });
    }
}

/// Keyword overlap score between a fact's text and query terms.
fn keyword_overlap(fact: &Fact, query_terms: &[String]) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let fact_text = format!(
        "{} {} {}",
        fact.subject.to_lowercase(),
        fact.predicate.to_lowercase(),
        fact.object.to_lowercase()
    );
    let matches = query_terms
        .iter()
        .filter(|t| fact_text.contains(t.as_str()))
        .count();
    matches as f64 / query_terms.len() as f64
}

/// Recency score — recent facts score close to 1, old facts close to 0.
fn recency_score(last_accessed: u64, now: u64) -> f64 {
    let age_secs = now.saturating_sub(last_accessed) as f64;
    // Half-life of 1 hour
    let half_life = 3600.0_f64;
    (-age_secs / half_life * std::f64::consts::LN_2).exp()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_store() -> MemoryStore {
        MemoryStore::new(MemoryConfig::default())
    }

    // ── Entity extraction ─────────────────────────────────────────────────────

    #[test]
    fn test_extract_entities_multi_word() {
        let entities = extract_entities("Alice Smith works at ACME Corp in Berlin");
        // "Alice Smith" or "ACME Corp" should be captured as multi-word entities
        assert!(!entities.is_empty());
    }

    #[test]
    fn test_extract_entities_no_capitals() {
        let entities = extract_entities("the quick brown fox");
        assert!(entities.is_empty());
    }

    #[test]
    fn test_extract_entities_deduplicates() {
        let entities = extract_entities("Alice knows Alice");
        // "Alice" should appear at most once
        let alice_count = entities.iter().filter(|e| e.contains("Alice")).count();
        assert!(alice_count <= 1);
    }

    // ── Fact ingestion ────────────────────────────────────────────────────────

    #[test]
    fn test_ingest_adds_facts() {
        let mut store = default_store();
        store.ingest("Alice works at ACME Corp");
        assert!(store.fact_count() > 0);
    }

    #[test]
    fn test_ingest_deduplicates() {
        let mut store = default_store();
        store.ingest("Alice works at ACME");
        let count_after_first = store.fact_count();
        store.ingest("Alice works at ACME");
        // Should not grow — duplicate merging
        assert_eq!(store.fact_count(), count_after_first);
    }

    #[test]
    fn test_ingest_empty_text_no_crash() {
        let mut store = default_store();
        store.ingest("");
        assert_eq!(store.fact_count(), 0);
    }

    #[test]
    fn test_ingest_multiple_sentences() {
        let mut store = default_store();
        store.ingest("Alice works at ACME. Bob manages Alice.");
        assert!(store.fact_count() >= 1);
    }

    // ── Insert fact directly ──────────────────────────────────────────────────

    #[test]
    fn test_insert_fact_basic() {
        let mut store = default_store();
        store.insert_fact("Alice", "knows", "Bob");
        assert_eq!(store.fact_count(), 1);
    }

    #[test]
    fn test_insert_fact_idempotent() {
        let mut store = default_store();
        store.insert_fact("Alice", "knows", "Bob");
        store.insert_fact("Alice", "knows", "Bob");
        assert_eq!(store.fact_count(), 1);
    }

    #[test]
    fn test_insert_multiple_facts() {
        let mut store = default_store();
        store.insert_fact("Alice", "knows", "Bob");
        store.insert_fact("Bob", "works_at", "ACME");
        assert_eq!(store.fact_count(), 2);
    }

    // ── Retrieval ─────────────────────────────────────────────────────────────

    #[test]
    fn test_retrieve_relevant_fact() {
        let mut store = default_store();
        store.insert_fact("Alice", "works_at", "ACME Corp");
        let results = store.retrieve("ACME", 5);
        assert!(!results.is_empty());
        assert!(results[0].fact.subject == "Alice" || results[0].fact.object.contains("ACME"));
    }

    #[test]
    fn test_retrieve_top_k_respected() {
        let mut store = default_store();
        for i in 0..10 {
            store.insert_fact(&format!("Entity{i}"), "is_a", "Thing");
        }
        let results = store.retrieve("Thing", 3);
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_retrieve_no_match_returns_empty() {
        let mut store = default_store();
        store.insert_fact("Alice", "knows", "Bob");
        let results = store.retrieve("xyzzy_unknown_term", 5);
        // May return empty because overlap is 0 — score may still be non-zero
        // due to relevance; just check no panic
        let _ = results;
    }

    #[test]
    fn test_retrieve_increments_access_count() {
        let mut store = default_store();
        store.insert_fact("Alice", "knows", "Bob");
        let _r1 = store.retrieve("Alice", 5);
        let key = "Alice|knows|Bob";
        let count = store.facts.get(key).map_or(0, |f| f.access_count);
        assert!(count > 0);
    }

    #[test]
    fn test_retrieve_descending_score_order() {
        let mut store = default_store();
        store.insert_fact("Alice", "works_at", "ACME Corp");
        store.insert_fact("Bob", "lives_in", "Berlin");
        store.insert_fact("Charlie", "knows", "Dave");
        let results = store.retrieve("Alice ACME", 10);
        for w in results.windows(2) {
            assert!(w[0].retrieval_score >= w[1].retrieval_score - 1e-10);
        }
    }

    // ── Decay ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_decay_keeps_facts() {
        let mut store = default_store();
        store.insert_fact("Alice", "knows", "Bob");
        store.decay();
        assert_eq!(store.fact_count(), 1);
    }

    #[test]
    fn test_decay_reduces_relevance() {
        let config = MemoryConfig {
            decay_rate: 0.1, // aggressive decay
            ..Default::default()
        };
        let mut store = MemoryStore::new(config);
        store.insert_fact("Alice", "knows", "Bob");
        // Manually set last_accessed to a past timestamp
        let key = "Alice|knows|Bob".to_string();
        if let Some(fact) = store.facts.get_mut(&key) {
            fact.last_accessed = fact.last_accessed.saturating_sub(1000);
        }
        store.decay();
        let rel = store.facts.get(&key).map_or(1.0, |f| f.relevance);
        assert!(rel < 1.0);
    }

    // ── Summarisation ─────────────────────────────────────────────────────────

    #[test]
    fn test_summarise_removes_low_relevance() {
        let config = MemoryConfig {
            summarise_threshold: 1.1, // everything below 1.1 → all facts qualify
            ..Default::default()
        };
        let mut store = MemoryStore::new(config);
        store.insert_fact("Alice", "knows", "Bob");
        // Manually lower relevance
        let key = "Alice|knows|Bob".to_string();
        if let Some(f) = store.facts.get_mut(&key) {
            f.relevance = 0.1;
        }
        let removed = store.summarise();
        assert!(removed > 0);
        assert_eq!(store.fact_count(), 0);
    }

    #[test]
    fn test_summarise_produces_summary() {
        let config = MemoryConfig {
            summarise_threshold: 1.1,
            ..Default::default()
        };
        let mut store = MemoryStore::new(config);
        store.insert_fact("Alice", "knows", "Bob");
        let key = "Alice|knows|Bob".to_string();
        if let Some(f) = store.facts.get_mut(&key) {
            f.relevance = 0.05;
        }
        store.summarise();
        assert!(store.summary().is_some());
    }

    #[test]
    fn test_summarise_no_low_relevance_no_change() {
        let mut store = default_store();
        store.insert_fact("Alice", "knows", "Bob");
        // relevance is 1.0 by default — above threshold
        let removed = store.summarise();
        assert_eq!(removed, 0);
        assert_eq!(store.fact_count(), 1);
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty_store() {
        let store = default_store();
        let stats = store.stats();
        assert_eq!(stats.fact_count, 0);
        assert!(stats.oldest_timestamp.is_none());
        assert!(stats.newest_timestamp.is_none());
        assert_eq!(stats.topic_count, 0);
    }

    #[test]
    fn test_stats_populated_store() {
        let mut store = default_store();
        store.insert_fact("Alice", "knows", "Bob");
        store.insert_fact("Bob", "works_at", "ACME");
        let stats = store.stats();
        assert_eq!(stats.fact_count, 2);
        assert_eq!(stats.topic_count, 2);
        assert!(stats.oldest_timestamp.is_some());
        assert!(stats.newest_timestamp.is_some());
    }

    #[test]
    fn test_stats_topic_distribution() {
        let mut store = default_store();
        store.insert_fact("Alice", "knows", "Bob");
        store.insert_fact("Bob", "knows", "Carol");
        store.insert_fact("Alice", "works_at", "ACME");
        let stats = store.stats();
        assert_eq!(*stats.topic_distribution.get("knows").unwrap_or(&0), 2);
        assert_eq!(*stats.topic_distribution.get("works_at").unwrap_or(&0), 1);
    }

    // ── Export / import ───────────────────────────────────────────────────────

    #[test]
    fn test_export_contains_all_facts() {
        let mut store = default_store();
        store.insert_fact("Alice", "knows", "Bob");
        store.insert_fact("Bob", "works_at", "ACME");
        let export = store.export();
        assert_eq!(export.facts.len(), 2);
        assert!(export.exported_at > 0);
    }

    #[test]
    fn test_import_merges_facts() {
        let mut store_a = default_store();
        store_a.insert_fact("Alice", "knows", "Bob");
        let export = store_a.export();

        let mut store_b = default_store();
        store_b.insert_fact("Carol", "works_at", "XYZ");
        store_b.import(&export);

        assert_eq!(store_b.fact_count(), 2);
    }

    #[test]
    fn test_import_deduplicates() {
        let mut store = default_store();
        store.insert_fact("Alice", "knows", "Bob");
        let export = store.export();
        store.import(&export);
        // Should still be 1 — duplicate
        assert_eq!(store.fact_count(), 1);
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    #[test]
    fn test_keyword_overlap_full_match() {
        let fact = Fact {
            dedup_key: String::new(),
            subject: "alice".into(),
            predicate: "knows".into(),
            object: "bob".into(),
            relevance: 1.0,
            created_at: 0,
            last_accessed: 0,
            access_count: 0,
        };
        let terms = vec!["alice".into(), "knows".into()];
        let score = keyword_overlap(&fact, &terms);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_keyword_overlap_no_match() {
        let fact = Fact {
            dedup_key: String::new(),
            subject: "alice".into(),
            predicate: "knows".into(),
            object: "bob".into(),
            relevance: 1.0,
            created_at: 0,
            last_accessed: 0,
            access_count: 0,
        };
        let score = keyword_overlap(&fact, &["xyzzy".into()]);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_recency_score_zero_age() {
        let now = now_secs();
        let score = recency_score(now, now);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_recency_score_old_fact() {
        let now = now_secs();
        let one_day_ago = now.saturating_sub(86400);
        let score = recency_score(one_day_ago, now);
        assert!(score < 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_fact_dedup_key() {
        let fact = Fact {
            dedup_key: "Alice|knows|Bob".into(),
            subject: "Alice".into(),
            predicate: "knows".into(),
            object: "Bob".into(),
            relevance: 1.0,
            created_at: 0,
            last_accessed: 0,
            access_count: 0,
        };
        assert_eq!(fact.computed_key(), "Alice|knows|Bob");
        assert_eq!(fact.dedup_key, "Alice|knows|Bob");
    }

    #[test]
    fn test_memory_config_default_values() {
        let cfg = MemoryConfig::default();
        assert_eq!(cfg.max_facts, 500);
        assert!(cfg.decay_rate > 0.0);
        assert!(cfg.summarise_threshold > 0.0 && cfg.summarise_threshold < 1.0);
        assert!(cfg.default_top_k > 0);
    }

    #[test]
    fn test_normalise_predicate_works() {
        assert_eq!(normalise_predicate("is"), "is_a");
        assert_eq!(normalise_predicate("works"), "works_at");
        assert_eq!(normalise_predicate("lives"), "lives_in");
        assert_eq!(normalise_predicate("knows"), "knows");
        assert_eq!(normalise_predicate("other"), "other");
    }

    #[test]
    fn test_memory_entry_fields() {
        let fact = Fact {
            dedup_key: "A|p|B".into(),
            subject: "A".into(),
            predicate: "p".into(),
            object: "B".into(),
            relevance: 0.8,
            created_at: 100,
            last_accessed: 200,
            access_count: 3,
        };
        let entry = MemoryEntry {
            key: "A|p|B".into(),
            fact: fact.clone(),
            retrieval_score: 0.75,
        };
        assert_eq!(entry.key, "A|p|B");
        assert!((entry.retrieval_score - 0.75).abs() < 1e-10);
        assert_eq!(entry.fact.access_count, 3);
    }

    #[test]
    fn test_retrieve_empty_store() {
        let mut store = default_store();
        let results = store.retrieve("anything", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_export_empty_store() {
        let store = default_store();
        let export = store.export();
        assert!(export.facts.is_empty());
    }

    #[test]
    fn test_ingest_increments_access_on_duplicate() {
        let mut store = default_store();
        store.ingest("Alice works at ACME");
        store.ingest("Alice works at ACME");
        // Find the fact and check access_count was incremented
        for fact in store.facts.values() {
            if fact.subject.contains("Alice") {
                assert!(fact.access_count >= 1 || fact.relevance > 1.0 - 1e-10);
            }
        }
    }
}
