//! Predicate-based filtering implementation
//!
//! Filters services based on predicate presence using bloom filters

use crate::source_selection::types::*;
use crate::ServiceRegistry;
use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(feature = "caching")]
use bloom::{BloomFilter, ASMS};

#[cfg(not(feature = "caching"))]
use super::types::cache_stubs::BloomFilter;

impl PredicateBasedFilter {
    pub fn new(_config: &SourceSelectionConfig) -> Self {
        Self {
            service_filters: Arc::new(RwLock::new(HashMap::new())),
            last_update: Arc::new(RwLock::new(Utc::now())),
        }
    }

    pub async fn filter_by_predicates(
        &self,
        patterns: &[TriplePattern],
        _registry: &ServiceRegistry,
    ) -> Result<HashMap<String, f64>> {
        let mut matches = HashMap::new();
        let filters = self.service_filters.read().await;

        for (service_endpoint, filter) in filters.iter() {
            let mut match_score = 0.0;
            let mut total_patterns = 0.0;

            for pattern in patterns {
                total_patterns += 1.0;

                // Check predicate membership
                if filter.predicate_filter.contains(&pattern.predicate) {
                    match_score += 1.0;
                }

                // Check subject membership (if not variable)
                if !pattern.subject.starts_with('?')
                    && filter.subject_filter.contains(&pattern.subject)
                {
                    match_score += 0.5;
                }

                // Check object membership (if not variable)
                if !pattern.object.starts_with('?')
                    && filter.object_filter.contains(&pattern.object)
                {
                    match_score += 0.5;
                }
            }

            if total_patterns > 0.0 {
                let final_score = match_score / total_patterns;
                if final_score > 0.0 {
                    matches.insert(service_endpoint.clone(), final_score);
                }
            }
        }

        Ok(matches)
    }

    pub async fn update_filters(
        &self,
        service_endpoint: &str,
        triples: &[(String, String, String)],
    ) -> Result<()> {
        let capacity = triples.len().max(1000) as u32;
        let mut predicate_filter = BloomFilter::with_rate(0.01, capacity);
        let mut subject_filter = BloomFilter::with_rate(0.01, capacity);
        let mut object_filter = BloomFilter::with_rate(0.01, capacity);
        let mut type_filter = BloomFilter::with_rate(0.01, capacity);

        for (s, p, o) in triples {
            predicate_filter.insert(p);
            subject_filter.insert(s);
            object_filter.insert(o);

            // Add type information if available
            if p == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
                type_filter.insert(o);
            }
        }

        let filters = ServiceBloomFilters {
            predicate_filter,
            subject_filter,
            object_filter,
            type_filter,
            last_updated: Utc::now(),
            estimated_elements: triples.len(),
        };

        self.service_filters
            .write()
            .await
            .insert(service_endpoint.to_string(), filters);
        *self.last_update.write().await = Utc::now();

        Ok(())
    }
}
