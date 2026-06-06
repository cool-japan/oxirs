//! Serialization, IRI construction, and SPARQL-like query layer over
//! physics simulation RDF triples.

use crate::rdf::physics_rdf_mapper::{extract_double_literal, sanitize_iri_fragment, PhysicsToRdf};
use crate::rdf::physics_rdf_types::{Triple, NS_PHYS, NS_RDF, NS_SOSA, NS_SSN};
use crate::simulation::result_injection::SimulationResult;
use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// SparqlPhysicsQuery
// ──────────────────────────────────────────────────────────────────────────────

/// Executes SPARQL-like queries over an in-memory triple index built from
/// physics simulation result triples.
pub struct SparqlPhysicsQuery {
    /// Indexed triple store: subject → list of (predicate, object).
    store: HashMap<String, Vec<(String, String)>>,
    /// All triples (for full-scan queries).
    triples: Vec<Triple>,
}

impl SparqlPhysicsQuery {
    /// Build the query index from a slice of triples.
    pub fn new(triples: &[Triple]) -> Self {
        let mut store: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for t in triples {
            store
                .entry(t.subject.clone())
                .or_default()
                .push((t.predicate.clone(), t.object.clone()));
        }
        Self {
            store,
            triples: triples.to_vec(),
        }
    }

    /// Build from a simulation result using [`PhysicsToRdf`].
    pub fn from_result(result: &SimulationResult) -> Self {
        let converter = PhysicsToRdf::new();
        let triples = converter.convert(result);
        Self::new(&triples)
    }

    /// Return the maximum `sosa:hasSimpleResult` for "temperature" observations.
    pub fn get_max_temperature(&self) -> Option<f64> {
        self.get_max_for_property("temperature")
    }

    /// Return the maximum `sosa:hasSimpleResult` for observations of the named property.
    pub fn get_max_for_property(&self, property: &str) -> Option<f64> {
        let prop_iri = format!("<{}{}>", NS_PHYS, sanitize_iri_fragment(property));
        let observed_prop_pred = format!("<{}observedProperty>", NS_SOSA);
        let simple_result_pred = format!("<{}hasSimpleResult>", NS_SOSA);

        let obs_subjects: Vec<&String> = self
            .triples
            .iter()
            .filter(|t| t.predicate == observed_prop_pred && t.object == prop_iri)
            .map(|t| &t.subject)
            .collect();

        let mut max_val: Option<f64> = None;
        for subj in obs_subjects {
            if let Some(preds) = self.store.get(subj) {
                for (pred, obj) in preds {
                    if pred == &simple_result_pred {
                        if let Some(v) = extract_double_literal(obj) {
                            max_val = Some(match max_val {
                                None => v,
                                Some(cur) => cur.max(v),
                            });
                        }
                    }
                }
            }
        }
        max_val
    }

    /// Return all observations within a simulation time range `[t_start, t_end]`.
    ///
    /// Returns a list of `(property_iri, value)` pairs.
    pub fn get_observations_in_range(&self, t_start: f64, t_end: f64) -> Vec<(String, f64)> {
        let sim_time_pred = format!("<{}simTime>", NS_PHYS);
        let has_obs_pred = format!("<{}hasObservation>", NS_SSN);
        let simple_result_pred = format!("<{}hasSimpleResult>", NS_SOSA);
        let observed_prop_pred = format!("<{}observedProperty>", NS_SOSA);

        // Find state IRIs whose simTime is in [t_start, t_end]
        let in_range_states: Vec<String> = self
            .triples
            .iter()
            .filter(|t| t.predicate == sim_time_pred)
            .filter_map(|t| {
                extract_double_literal(&t.object).and_then(|sim_t| {
                    if sim_t >= t_start && sim_t <= t_end {
                        Some(t.subject.clone())
                    } else {
                        None
                    }
                })
            })
            .collect();

        let mut results = Vec::new();

        for state_subj in &in_range_states {
            if let Some(preds) = self.store.get(state_subj) {
                let obs_iris: Vec<String> = preds
                    .iter()
                    .filter(|(p, _)| p == &has_obs_pred)
                    .map(|(_, o)| o.clone())
                    .collect();

                for obs_iri in obs_iris {
                    if let Some(obs_preds) = self.store.get(&obs_iri) {
                        let prop = obs_preds
                            .iter()
                            .find(|(p, _)| p == &observed_prop_pred)
                            .map(|(_, o)| o.clone())
                            .unwrap_or_default();
                        let value = obs_preds
                            .iter()
                            .find(|(p, _)| p == &simple_result_pred)
                            .and_then(|(_, o)| extract_double_literal(o));

                        if let Some(v) = value {
                            results.push((prop, v));
                        }
                    }
                }
            }
        }
        results
    }

    /// Return the minimum `sosa:hasSimpleResult` for the named property.
    pub fn get_min_for_property(&self, property: &str) -> Option<f64> {
        let prop_iri = format!("<{}{}>", NS_PHYS, sanitize_iri_fragment(property));
        let observed_prop_pred = format!("<{}observedProperty>", NS_SOSA);
        let simple_result_pred = format!("<{}hasSimpleResult>", NS_SOSA);

        let obs_subjects: Vec<&String> = self
            .triples
            .iter()
            .filter(|t| t.predicate == observed_prop_pred && t.object == prop_iri)
            .map(|t| &t.subject)
            .collect();

        let mut min_val: Option<f64> = None;
        for subj in obs_subjects {
            if let Some(preds) = self.store.get(subj) {
                for (pred, obj) in preds {
                    if pred == &simple_result_pred {
                        if let Some(v) = extract_double_literal(obj) {
                            min_val = Some(match min_val {
                                None => v,
                                Some(cur) => cur.min(v),
                            });
                        }
                    }
                }
            }
        }
        min_val
    }

    /// Return the mean `sosa:hasSimpleResult` for the named property.
    pub fn get_mean_for_property(&self, property: &str) -> Option<f64> {
        let prop_iri = format!("<{}{}>", NS_PHYS, sanitize_iri_fragment(property));
        let observed_prop_pred = format!("<{}observedProperty>", NS_SOSA);
        let simple_result_pred = format!("<{}hasSimpleResult>", NS_SOSA);

        let obs_subjects: Vec<&String> = self
            .triples
            .iter()
            .filter(|t| t.predicate == observed_prop_pred && t.object == prop_iri)
            .map(|t| &t.subject)
            .collect();

        let mut sum = 0.0_f64;
        let mut count = 0usize;
        for subj in obs_subjects {
            if let Some(preds) = self.store.get(subj) {
                for (pred, obj) in preds {
                    if pred == &simple_result_pred {
                        if let Some(v) = extract_double_literal(obj) {
                            sum += v;
                            count += 1;
                        }
                    }
                }
            }
        }
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }

    /// Count total `sosa:Observation` triples in the store.
    pub fn count_observations(&self) -> usize {
        let obs_type = format!("<{}Observation>", NS_SOSA);
        let rdf_type = format!("<{}type>", NS_RDF);
        self.triples
            .iter()
            .filter(|t| t.predicate == rdf_type && t.object == obs_type)
            .count()
    }

    /// Return all distinct `sosa:observedProperty` IRIs.
    pub fn list_observed_properties(&self) -> Vec<String> {
        let observed_prop_pred = format!("<{}observedProperty>", NS_SOSA);
        let mut seen = std::collections::HashSet::new();
        self.triples
            .iter()
            .filter(|t| t.predicate == observed_prop_pred)
            .map(|t| t.object.trim().to_string())
            .filter(|o| seen.insert(o.clone()))
            .collect()
    }

    /// Return all `(property_iri, value)` pairs for a given property.
    pub fn all_values_for_property(&self, property: &str) -> Vec<f64> {
        let prop_iri = format!("<{}{}>", NS_PHYS, sanitize_iri_fragment(property));
        let observed_prop_pred = format!("<{}observedProperty>", NS_SOSA);
        let simple_result_pred = format!("<{}hasSimpleResult>", NS_SOSA);

        let obs_subjects: Vec<&String> = self
            .triples
            .iter()
            .filter(|t| t.predicate == observed_prop_pred && t.object == prop_iri)
            .map(|t| &t.subject)
            .collect();

        let mut values = Vec::new();
        for subj in obs_subjects {
            if let Some(preds) = self.store.get(subj) {
                for (pred, obj) in preds {
                    if pred == &simple_result_pred {
                        if let Some(v) = extract_double_literal(obj) {
                            values.push(v);
                        }
                    }
                }
            }
        }
        values
    }
}
