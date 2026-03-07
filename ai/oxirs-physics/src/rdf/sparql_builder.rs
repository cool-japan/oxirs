//! SPARQL Query Builder for Physics Entity Properties
//!
//! Constructs well-formed SPARQL 1.1 SELECT and UPDATE queries for reading and
//! writing physical entity properties from/to an RDF triplestore.
//!
//! # Design goals
//!
//! * **Zero string panics**: all IRI and literal escaping is handled by helper
//!   functions; no `unwrap()` anywhere.
//! * **Incremental composition**: callers chain `with_property` calls before
//!   calling `build_select_query` or `build_update_query`.
//! * **Namespace-aware**: a configurable prefix map allows compact CURIE-style
//!   output.
//! * **SPARQL 1.1 compliance**: both SELECT and UPDATE (INSERT DATA) forms are
//!   standards-compliant.

use crate::error::{PhysicsError, PhysicsResult};
use crate::simulation::result_injection::SimulationResult;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ──────────────────────────────────────────────────────────────────────────────
// Physics property vocabulary
// ──────────────────────────────────────────────────────────────────────────────

/// Well-known physics properties that can be queried from the triplestore.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PhysicsProperty {
    /// Object mass (SI: kg)
    Mass,
    /// Linear velocity (SI: m/s, expressed as a vector magnitude or component)
    Velocity,
    /// Thermodynamic temperature (SI: K)
    Temperature,
    /// Spatial position (may be a vector triple)
    Position,
    /// Applied force (SI: N)
    Force,
    /// Total mechanical / thermal energy (SI: J)
    Energy,
    /// Electric power (SI: W)
    Power,
    /// Pressure (SI: Pa)
    Pressure,
    /// Angular velocity (SI: rad/s)
    AngularVelocity,
    /// Moment of inertia (SI: kg⋅m²)
    MomentOfInertia,
    /// User-supplied predicate URI or local name
    Custom(String),
}

impl PhysicsProperty {
    /// Return the local name used as a SPARQL variable and as the tail of the
    /// default physics predicate IRI.
    pub fn local_name(&self) -> &str {
        match self {
            Self::Mass => "mass",
            Self::Velocity => "velocity",
            Self::Temperature => "temperature",
            Self::Position => "position",
            Self::Force => "force",
            Self::Energy => "energy",
            Self::Power => "power",
            Self::Pressure => "pressure",
            Self::AngularVelocity => "angularVelocity",
            Self::MomentOfInertia => "momentOfInertia",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Return the SPARQL variable name for this property (prefixed with `?`).
    pub fn sparql_var(&self) -> String {
        format!("?{}", self.local_name())
    }

    /// Build the predicate IRI fragment relative to a namespace prefix.
    pub fn predicate_iri(&self, namespace: &str) -> String {
        format!("{}{}", namespace, self.local_name())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Prefix map helpers
// ──────────────────────────────────────────────────────────────────────────────

/// A lightweight ordered prefix map for SPARQL preamble generation.
///
/// Ordered so that the generated `PREFIX` declarations are deterministic.
#[derive(Debug, Clone)]
pub struct PrefixMap {
    entries: Vec<(String, String)>, // (prefix_name, namespace_iri)
}

impl Default for PrefixMap {
    fn default() -> Self {
        let mut pm = Self {
            entries: Vec::new(),
        };
        pm.insert("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        pm.insert("rdfs", "http://www.w3.org/2000/01/rdf-schema#");
        pm.insert("xsd", "http://www.w3.org/2001/XMLSchema#");
        pm.insert("phys", "http://oxirs.org/physics#");
        pm.insert("prov", "http://www.w3.org/ns/prov#");
        pm.insert("qudt", "http://qudt.org/schema/qudt/");
        pm
    }
}

impl PrefixMap {
    /// Create an empty prefix map (no defaults).
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Insert or overwrite a prefix entry.
    pub fn insert(&mut self, prefix: impl Into<String>, namespace: impl Into<String>) {
        let p = prefix.into();
        let n = namespace.into();
        // Replace if already present
        if let Some(entry) = self.entries.iter_mut().find(|(k, _)| *k == p) {
            entry.1 = n;
        } else {
            self.entries.push((p, n));
        }
    }

    /// Look up the full IRI for a prefix.
    pub fn get_namespace(&self, prefix: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(k, _)| k == prefix)
            .map(|(_, v)| v.as_str())
    }

    /// Emit SPARQL `PREFIX` declarations for all entries.
    pub fn to_sparql_preamble(&self) -> String {
        let mut out = String::new();
        for (prefix, ns) in &self.entries {
            let _ = writeln!(out, "PREFIX {}: <{}>", prefix, ns);
        }
        out
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SPARQL SELECT query builder
// ──────────────────────────────────────────────────────────────────────────────

/// Builder for SPARQL SELECT queries that retrieve physical entity properties.
///
/// # Example
///
/// ```rust
/// use oxirs_physics::rdf::sparql_builder::{PhysicsPropertyQuery, PhysicsProperty};
///
/// let query = PhysicsPropertyQuery::new("urn:example:Turbine#001")
///     .with_property(PhysicsProperty::Mass)
///     .with_property(PhysicsProperty::Temperature)
///     .build_select_query();
///
/// assert!(query.contains("SELECT"));
/// assert!(query.contains("?mass"));
/// assert!(query.contains("?temperature"));
/// ```
#[derive(Debug, Clone)]
pub struct PhysicsPropertyQuery {
    /// IRI of the entity to query (absolute URI).
    entity_uri: String,
    /// Properties to project in the SELECT clause.
    properties: Vec<PhysicsProperty>,
    /// Prefix map for compact output.
    prefixes: PrefixMap,
    /// Physics namespace (tail is used as predicate IRI base).
    physics_namespace: String,
    /// If `true`, OPTIONAL patterns are emitted so that entities with partial
    /// property coverage still produce a result row.
    optional_patterns: bool,
    /// If `true`, also retrieve the `phys:unit` annotation for each value.
    include_units: bool,
}

impl PhysicsPropertyQuery {
    /// Create a new query builder for the given entity URI.
    pub fn new(entity_uri: impl Into<String>) -> Self {
        let physics_ns = "http://oxirs.org/physics#".to_string();
        let mut prefixes = PrefixMap::default();
        prefixes.insert("phys", &physics_ns);

        Self {
            entity_uri: entity_uri.into(),
            properties: Vec::new(),
            prefixes,
            physics_namespace: physics_ns,
            optional_patterns: true,
            include_units: true,
        }
    }

    /// Add a property to retrieve.
    pub fn with_property(mut self, property: PhysicsProperty) -> Self {
        self.properties.push(property);
        self
    }

    /// Add multiple properties at once.
    pub fn with_properties(
        mut self,
        properties: impl IntoIterator<Item = PhysicsProperty>,
    ) -> Self {
        self.properties.extend(properties);
        self
    }

    /// Override the physics namespace used for predicates (default: `http://oxirs.org/physics#`).
    pub fn with_physics_namespace(mut self, ns: impl Into<String>) -> Self {
        let ns = ns.into();
        self.prefixes.insert("phys", &ns);
        self.physics_namespace = ns;
        self
    }

    /// Control whether property patterns are OPTIONAL (default: `true`).
    pub fn with_optional_patterns(mut self, optional: bool) -> Self {
        self.optional_patterns = optional;
        self
    }

    /// Control whether unit annotations are retrieved (default: `true`).
    pub fn with_units(mut self, include: bool) -> Self {
        self.include_units = include;
        self
    }

    /// Build the SPARQL SELECT query string.
    ///
    /// The generated query retrieves the value of each registered property from
    /// the triplestore for the configured entity URI.  When `optional_patterns`
    /// is `true` (the default), each triple pattern is wrapped in `OPTIONAL {}`
    /// so that missing properties do not eliminate the entity row.
    ///
    /// # Returns
    ///
    /// A UTF-8 SPARQL 1.1 query string.
    pub fn build_select_query(&self) -> String {
        let mut q = String::with_capacity(512);

        // Preamble
        q.push_str(&self.prefixes.to_sparql_preamble());
        q.push('\n');

        // SELECT clause
        q.push_str("SELECT ?entity");
        for prop in &self.properties {
            let _ = write!(q, " {}", prop.sparql_var());
            if self.include_units {
                let _ = write!(q, " ?{}Unit", prop.local_name());
            }
        }
        q.push_str(" WHERE {\n");

        // Entity binding
        let _ = writeln!(q, "  BIND(<{}> AS ?entity)", escape_iri(&self.entity_uri));

        // Property patterns
        for prop in &self.properties {
            let predicate = self.predicate_curie(prop);
            let var = prop.sparql_var();
            let unit_var = format!("?{}Unit", prop.local_name());

            if self.optional_patterns {
                q.push_str("  OPTIONAL {\n");
                let _ = writeln!(q, "    ?entity {} {} .", predicate, var);
                if self.include_units {
                    let _ = writeln!(q, "    OPTIONAL {{ {} phys:unit {} . }}", var, unit_var);
                }
                q.push_str("  }\n");
            } else {
                let _ = writeln!(q, "  ?entity {} {} .", predicate, var);
                if self.include_units {
                    // No OPTIONAL wrapper when optional_patterns is false
                    let _ = writeln!(q, "  {} phys:unit {} .", var, unit_var);
                }
            }
        }

        q.push('}');
        q
    }

    /// Build a SPARQL UPDATE (INSERT DATA) query to write simulation results
    /// for the configured entity URI.
    ///
    /// For each scalar quantity in `result.derived_quantities`, a triple of the
    /// form:
    /// ```sparql
    ///   <entity> phys:<key> "<value>"^^xsd:double .
    /// ```
    /// is inserted into the default graph.  Additionally, meta-triples recording
    /// the simulation run ID and timestamp are inserted.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::ResultInjection`] if the entity URI is invalid or
    /// the update string cannot be constructed.
    pub fn build_update_query(&self, result: &SimulationResult) -> PhysicsResult<String> {
        if result.entity_iri.is_empty() {
            return Err(PhysicsError::ResultInjection(
                "Cannot build UPDATE query: entity IRI is empty".to_string(),
            ));
        }

        let mut q = String::with_capacity(1024);
        q.push_str(&self.prefixes.to_sparql_preamble());
        q.push('\n');
        q.push_str("INSERT DATA {\n");

        let entity_iri = escape_iri(&result.entity_iri);
        let run_id = sparql_string_literal(&result.simulation_run_id);
        let timestamp = sparql_string_literal(&result.timestamp.to_rfc3339());

        // Meta triples
        let _ = writeln!(q, "  <{}> phys:simulationRunId {} .", entity_iri, run_id);
        let _ = writeln!(
            q,
            "  <{}> phys:simulationTimestamp {} .",
            entity_iri, timestamp
        );
        let _ = writeln!(
            q,
            "  <{}> phys:converged \"{}\"^^xsd:boolean .",
            entity_iri, result.convergence_info.converged
        );
        let _ = writeln!(
            q,
            "  <{}> phys:iterations \"{}\"^^xsd:integer .",
            entity_iri, result.convergence_info.iterations
        );

        // Derived scalar quantities
        for (key, &val) in &result.derived_quantities {
            let safe_key = sanitize_local_name(key);
            if safe_key.is_empty() {
                continue;
            }
            let _ = writeln!(
                q,
                "  <{}> phys:{} \"{}\"^^xsd:double .",
                entity_iri, safe_key, val
            );
        }

        // Latest state vector (last time step)
        if let Some(last_state) = result.state_trajectory.last() {
            let _ = writeln!(
                q,
                "  <{}> phys:simulationTime \"{}\"^^xsd:double .",
                entity_iri, last_state.time
            );
            for (state_key, &state_val) in &last_state.state {
                let safe_key = sanitize_local_name(state_key);
                if safe_key.is_empty() {
                    continue;
                }
                let _ = writeln!(
                    q,
                    "  <{}> phys:finalState_{} \"{}\"^^xsd:double .",
                    entity_iri, safe_key, state_val
                );
            }
        }

        q.push('}');
        Ok(q)
    }

    // ── private helpers ──────────────────────────────────────────────────────

    /// Return the SPARQL representation of a predicate (CURIE or full IRI).
    fn predicate_curie(&self, prop: &PhysicsProperty) -> String {
        // Use "phys:" CURIE for standard physics namespace
        if self.physics_namespace == "http://oxirs.org/physics#" {
            format!("phys:{}", prop.local_name())
        } else {
            format!("<{}{}>", self.physics_namespace, prop.local_name())
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SPARQL UPDATE generator (standalone)
// ──────────────────────────────────────────────────────────────────────────────

/// Generate a SPARQL DELETE/INSERT (MODIFY) query that replaces old property
/// values with new ones.
///
/// This is a separate free function for callers that do not use the builder.
///
/// # Arguments
///
/// * `entity_uri` – Subject entity IRI.
/// * `property`   – Physics property to update.
/// * `new_value`  – New literal value (will be typed as `xsd:double`).
/// * `unit_str`   – Optional unit annotation written as `phys:unit`.
pub fn build_property_replace_query(
    entity_uri: &str,
    property: &PhysicsProperty,
    new_value: f64,
    unit_str: Option<&str>,
) -> PhysicsResult<String> {
    if entity_uri.is_empty() {
        return Err(PhysicsError::RdfQuery(
            "Entity URI must not be empty".to_string(),
        ));
    }

    let mut q = String::with_capacity(512);
    let preamble = PrefixMap::default().to_sparql_preamble();
    q.push_str(&preamble);
    q.push('\n');

    let predicate = format!("phys:{}", property.local_name());
    let entity = escape_iri(entity_uri);

    // DELETE old values
    let _ = writeln!(q, "DELETE {{ <{}> {} ?oldValue . }}", entity, predicate);
    q.push_str("WHERE  { OPTIONAL { ");
    let _ = write!(q, "<{}> {} ?oldValue . ", entity, predicate);
    q.push_str("} }\n");
    q.push(';');
    q.push('\n');

    // INSERT new value
    q.push_str("INSERT DATA {\n");
    let _ = writeln!(
        q,
        "  <{}> {} \"{}\"^^xsd:double .",
        entity, predicate, new_value
    );
    if let Some(unit) = unit_str {
        let _ = writeln!(
            q,
            "  <{}> phys:unit {} .",
            entity,
            sparql_string_literal(unit)
        );
    }
    q.push('}');

    Ok(q)
}

// ──────────────────────────────────────────────────────────────────────────────
// Batch SPARQL SELECT builder
// ──────────────────────────────────────────────────────────────────────────────

/// Build a SPARQL SELECT query that retrieves the same set of properties for
/// **multiple** entities in a single round-trip using `VALUES`.
///
/// # Arguments
///
/// * `entity_uris`  – Slice of entity URI strings.
/// * `properties`   – Physics properties to retrieve.
/// * `physics_ns`   – Physics namespace (default: `http://oxirs.org/physics#`).
pub fn build_batch_select_query(
    entity_uris: &[&str],
    properties: &[PhysicsProperty],
    physics_ns: Option<&str>,
) -> PhysicsResult<String> {
    if entity_uris.is_empty() {
        return Err(PhysicsError::RdfQuery(
            "At least one entity URI is required".to_string(),
        ));
    }
    if properties.is_empty() {
        return Err(PhysicsError::RdfQuery(
            "At least one property must be specified".to_string(),
        ));
    }

    let ns = physics_ns.unwrap_or("http://oxirs.org/physics#");
    let prefixes = {
        let mut pm = PrefixMap::default();
        pm.insert("phys", ns);
        pm
    };

    let mut q = String::with_capacity(1024);
    q.push_str(&prefixes.to_sparql_preamble());
    q.push('\n');

    // SELECT header
    q.push_str("SELECT ?entity");
    for prop in properties {
        let _ = write!(q, " {}", prop.sparql_var());
    }
    q.push_str(" WHERE {\n");

    // VALUES clause for entity bindings
    q.push_str("  VALUES ?entity {\n");
    for uri in entity_uris {
        let _ = writeln!(q, "    <{}>", escape_iri(uri));
    }
    q.push_str("  }\n");

    // Optional property patterns
    for prop in properties {
        let predicate = if ns == "http://oxirs.org/physics#" {
            format!("phys:{}", prop.local_name())
        } else {
            format!("<{}{}>", ns, prop.local_name())
        };
        let _ = writeln!(
            q,
            "  OPTIONAL {{ ?entity {} {} . }}",
            predicate,
            prop.sparql_var()
        );
    }

    q.push('}');
    Ok(q)
}

// ──────────────────────────────────────────────────────────────────────────────
// Provenance query builder
// ──────────────────────────────────────────────────────────────────────────────

/// Build a SPARQL SELECT query that retrieves simulation provenance information
/// for a given entity using the W3C PROV ontology.
pub fn build_provenance_query(entity_uri: &str) -> PhysicsResult<String> {
    if entity_uri.is_empty() {
        return Err(PhysicsError::RdfQuery(
            "Entity URI must not be empty for provenance query".to_string(),
        ));
    }

    let prefixes = PrefixMap::default();
    let mut q = String::with_capacity(512);
    q.push_str(&prefixes.to_sparql_preamble());
    q.push('\n');

    let entity = escape_iri(entity_uri);

    let _ = write!(
        q,
        r#"SELECT ?activity ?softwareName ?softwareVersion ?startTime ?endTime WHERE {{
  BIND(<{entity}> AS ?entity)
  OPTIONAL {{
    ?entity prov:wasGeneratedBy ?activity .
    OPTIONAL {{ ?activity prov:used ?softwareName . }}
    OPTIONAL {{ ?activity prov:atTime ?startTime . }}
    OPTIONAL {{ ?activity phys:softwareVersion ?softwareVersion . }}
    OPTIONAL {{ ?activity prov:endedAtTime ?endTime . }}
  }}
}}"#,
        entity = entity
    );

    Ok(q)
}

// ──────────────────────────────────────────────────────────────────────────────
// String escaping helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Escape a URI for use inside `<…>` in SPARQL.
///
/// Per SPARQL 1.1 spec, only `>`, `{`, `}`, `|`, `\`, `^`, `` ` ``, and
/// characters < U+0020 are forbidden inside IRI references.  We escape `>`
/// and the most common dangerous characters.
fn escape_iri(iri: &str) -> String {
    iri.replace('>', "%3E")
        .replace('{', "%7B")
        .replace('}', "%7D")
        .replace('|', "%7C")
        .replace('\\', "%5C")
        .replace('^', "%5E")
        .replace('`', "%60")
}

/// Wrap a string in double quotes and escape internal double-quote / backslash.
fn sparql_string_literal(s: &str) -> String {
    let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{}\"", escaped)
}

/// Sanitize an arbitrary string so it can be used as a local name in a CURIE or
/// as a SPARQL variable identifier.
///
/// Replaces any character that is not alphanumeric, `_`, or `-` with `_`.
fn sanitize_local_name(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Simulation result round-trip helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Parse SPARQL binding result rows (as a map of variable → value string) into
/// a [`HashMap`] of property name to numeric value.
///
/// This mirrors what a real SPARQL client would return and is used in tests.
pub fn extract_property_values(
    bindings: &[HashMap<String, String>],
    properties: &[PhysicsProperty],
) -> HashMap<String, f64> {
    let mut result = HashMap::new();
    for row in bindings {
        for prop in properties {
            let var_name = prop.local_name();
            if let Some(val_str) = row.get(var_name) {
                if let Ok(val) = val_str.trim().parse::<f64>() {
                    result.insert(var_name.to_string(), val);
                }
            }
        }
    }
    result
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::result_injection::{
        ConvergenceInfo, SimulationProvenance, SimulationResult, StateVector,
    };
    use chrono::Utc;
    use std::collections::HashMap;

    fn make_test_result() -> SimulationResult {
        let mut derived = HashMap::new();
        derived.insert("max_temperature".to_string(), 450.0);
        derived.insert("total_heat_flux".to_string(), 1250.0);

        let mut state = HashMap::new();
        state.insert("temperature".to_string(), 380.5);
        let trajectory = vec![StateVector { time: 100.0, state }];

        SimulationResult {
            entity_iri: "urn:example:battery:001".to_string(),
            simulation_run_id: "run-abc-123".to_string(),
            timestamp: Utc::now(),
            state_trajectory: trajectory,
            derived_quantities: derived,
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations: 200,
                final_residual: 1e-8,
            },
            provenance: SimulationProvenance {
                software: "OxiRS Physics".to_string(),
                version: "0.2.0".to_string(),
                parameters_hash: "abc123".to_string(),
                executed_at: Utc::now(),
                execution_time_ms: 1500,
            },
        }
    }

    // ── build_select_query ───────────────────────────────────────────────────

    #[test]
    fn test_select_query_contains_select_keyword() {
        let q = PhysicsPropertyQuery::new("urn:example:motor:1")
            .with_property(PhysicsProperty::Mass)
            .build_select_query();
        assert!(q.contains("SELECT"), "SELECT keyword missing");
    }

    #[test]
    fn test_select_query_binds_entity() {
        let q = PhysicsPropertyQuery::new("urn:example:motor:1")
            .with_property(PhysicsProperty::Mass)
            .build_select_query();
        assert!(q.contains("?entity"), "?entity variable missing");
        assert!(q.contains("urn:example:motor:1"), "entity URI missing");
    }

    #[test]
    fn test_select_query_projects_all_properties() {
        let q = PhysicsPropertyQuery::new("urn:example:turbine:99")
            .with_property(PhysicsProperty::Mass)
            .with_property(PhysicsProperty::Temperature)
            .with_property(PhysicsProperty::Velocity)
            .build_select_query();
        assert!(q.contains("?mass"), "?mass missing");
        assert!(q.contains("?temperature"), "?temperature missing");
        assert!(q.contains("?velocity"), "?velocity missing");
    }

    #[test]
    fn test_select_query_optional_patterns() {
        let q_opt = PhysicsPropertyQuery::new("urn:example:e:1")
            .with_property(PhysicsProperty::Force)
            .with_optional_patterns(true)
            .build_select_query();
        assert!(q_opt.contains("OPTIONAL"), "OPTIONAL missing");

        let q_req = PhysicsPropertyQuery::new("urn:example:e:2")
            .with_property(PhysicsProperty::Force)
            .with_optional_patterns(false)
            .build_select_query();
        assert!(
            !q_req.contains("OPTIONAL"),
            "OPTIONAL should not be present"
        );
    }

    #[test]
    fn test_select_query_prefix_declarations() {
        let q = PhysicsPropertyQuery::new("urn:example:e:1")
            .with_property(PhysicsProperty::Energy)
            .build_select_query();
        assert!(q.contains("PREFIX phys:"), "phys prefix missing");
        assert!(q.contains("PREFIX xsd:"), "xsd prefix missing");
    }

    #[test]
    fn test_select_query_no_properties() {
        let q = PhysicsPropertyQuery::new("urn:example:e:0").build_select_query();
        assert!(
            q.contains("SELECT"),
            "SELECT missing even with no properties"
        );
        assert!(q.contains("WHERE"), "WHERE missing");
    }

    #[test]
    fn test_select_query_custom_namespace() {
        let q = PhysicsPropertyQuery::new("urn:example:e:3")
            .with_physics_namespace("http://example.org/custom-physics#")
            .with_property(PhysicsProperty::Mass)
            .with_optional_patterns(false)
            .build_select_query();
        assert!(
            q.contains("http://example.org/custom-physics#mass"),
            "custom namespace not applied"
        );
    }

    #[test]
    fn test_select_query_unit_annotations() {
        let q = PhysicsPropertyQuery::new("urn:example:e:1")
            .with_property(PhysicsProperty::Mass)
            .with_units(true)
            .build_select_query();
        assert!(q.contains("?massUnit"), "massUnit variable missing");

        let q_no_units = PhysicsPropertyQuery::new("urn:example:e:1")
            .with_property(PhysicsProperty::Mass)
            .with_units(false)
            .build_select_query();
        assert!(
            !q_no_units.contains("?massUnit"),
            "massUnit should be absent"
        );
    }

    // ── build_update_query ───────────────────────────────────────────────────

    #[test]
    fn test_update_query_contains_insert_data() {
        let result = make_test_result();
        let q = PhysicsPropertyQuery::new(&result.entity_iri)
            .build_update_query(&result)
            .expect("update query failed");
        assert!(q.contains("INSERT DATA"), "INSERT DATA missing");
    }

    #[test]
    fn test_update_query_contains_entity_iri() {
        let result = make_test_result();
        let q = PhysicsPropertyQuery::new(&result.entity_iri)
            .build_update_query(&result)
            .expect("update query failed");
        assert!(
            q.contains("urn:example:battery:001"),
            "entity IRI missing from UPDATE"
        );
    }

    #[test]
    fn test_update_query_contains_derived_quantities() {
        let result = make_test_result();
        let q = PhysicsPropertyQuery::new(&result.entity_iri)
            .build_update_query(&result)
            .expect("update query failed");
        assert!(
            q.contains("max_temperature") || q.contains("max_temperature"),
            "derived quantity missing"
        );
        assert!(q.contains("xsd:double"), "xsd:double type missing");
    }

    #[test]
    fn test_update_query_empty_entity_iri_is_error() {
        let mut result = make_test_result();
        result.entity_iri = String::new();
        let err = PhysicsPropertyQuery::new("").build_update_query(&result);
        assert!(err.is_err(), "expected error for empty entity IRI");
    }

    #[test]
    fn test_update_query_contains_convergence_info() {
        let result = make_test_result();
        let q = PhysicsPropertyQuery::new(&result.entity_iri)
            .build_update_query(&result)
            .expect("update query failed");
        assert!(
            q.contains("converged") || q.contains("iterations"),
            "convergence info missing"
        );
    }

    // ── build_batch_select_query ─────────────────────────────────────────────

    #[test]
    fn test_batch_select_query_multiple_entities() {
        let uris = ["urn:example:motor:1", "urn:example:motor:2"];
        let props = [PhysicsProperty::Mass, PhysicsProperty::Temperature];
        let q = build_batch_select_query(&uris, &props, None).expect("batch query failed");
        assert!(q.contains("VALUES"), "VALUES clause missing");
        assert!(q.contains("urn:example:motor:1"), "first entity missing");
        assert!(q.contains("urn:example:motor:2"), "second entity missing");
        assert!(q.contains("?mass"), "?mass missing");
        assert!(q.contains("?temperature"), "?temperature missing");
    }

    #[test]
    fn test_batch_select_query_empty_entities_is_error() {
        let result = build_batch_select_query(&[], &[PhysicsProperty::Mass], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_select_query_empty_properties_is_error() {
        let result = build_batch_select_query(&["urn:example:e:1"], &[], None);
        assert!(result.is_err());
    }

    // ── build_property_replace_query ─────────────────────────────────────────

    #[test]
    fn test_replace_query_delete_insert_structure() {
        let q = build_property_replace_query(
            "urn:example:pump:5",
            &PhysicsProperty::Temperature,
            310.0,
            Some("K"),
        )
        .expect("replace query failed");
        assert!(q.contains("DELETE"), "DELETE missing");
        assert!(q.contains("INSERT DATA"), "INSERT DATA missing");
        assert!(q.contains("310"), "new value missing");
    }

    #[test]
    fn test_replace_query_empty_uri_is_error() {
        let err = build_property_replace_query("", &PhysicsProperty::Mass, 10.0, None);
        assert!(err.is_err());
    }

    // ── build_provenance_query ───────────────────────────────────────────────

    #[test]
    fn test_provenance_query_structure() {
        let q = build_provenance_query("urn:example:entity:7").expect("prov query failed");
        assert!(q.contains("SELECT"), "SELECT missing");
        assert!(q.contains("prov:wasGeneratedBy"), "prov predicate missing");
    }

    #[test]
    fn test_provenance_query_empty_uri_is_error() {
        assert!(build_provenance_query("").is_err());
    }

    // ── extract_property_values ──────────────────────────────────────────────

    #[test]
    fn test_extract_property_values_happy_path() {
        let mut row = HashMap::new();
        row.insert("mass".to_string(), "75.0".to_string());
        row.insert("temperature".to_string(), "300.0".to_string());

        let props = [PhysicsProperty::Mass, PhysicsProperty::Temperature];
        let extracted = extract_property_values(&[row], &props);

        assert_eq!(extracted.get("mass"), Some(&75.0));
        assert_eq!(extracted.get("temperature"), Some(&300.0));
    }

    #[test]
    fn test_extract_property_values_missing_key() {
        let row: HashMap<String, String> = HashMap::new();
        let props = [PhysicsProperty::Velocity];
        let extracted = extract_property_values(&[row], &props);
        assert!(!extracted.contains_key("velocity"));
    }

    // ── sanitize_local_name ──────────────────────────────────────────────────

    #[test]
    fn test_sanitize_local_name_spaces() {
        assert_eq!(sanitize_local_name("max temperature"), "max_temperature");
    }

    #[test]
    fn test_sanitize_local_name_special_chars() {
        assert_eq!(sanitize_local_name("heat/flux"), "heat_flux");
    }

    // ── escape_iri ────────────────────────────────────────────────────────────

    #[test]
    fn test_escape_iri_no_op_clean_uri() {
        assert_eq!(escape_iri("urn:example:foo"), "urn:example:foo");
    }

    #[test]
    fn test_escape_iri_angle_bracket() {
        assert!(escape_iri("urn:example:foo>bar").contains("%3E"));
    }

    // ── PrefixMap ────────────────────────────────────────────────────────────

    #[test]
    fn test_prefix_map_default_has_rdf() {
        let pm = PrefixMap::default();
        assert!(pm.get_namespace("rdf").is_some());
        assert!(pm.get_namespace("xsd").is_some());
        assert!(pm.get_namespace("phys").is_some());
    }

    #[test]
    fn test_prefix_map_insert_overwrite() {
        let mut pm = PrefixMap::default();
        pm.insert("phys", "http://custom.org/phys#");
        assert_eq!(pm.get_namespace("phys"), Some("http://custom.org/phys#"));
    }

    #[test]
    fn test_prefix_map_preamble_format() {
        let pm = PrefixMap::empty();
        let preamble = pm.to_sparql_preamble();
        assert!(preamble.is_empty());

        let mut pm2 = PrefixMap::empty();
        pm2.insert("ex", "http://example.org/");
        let p2 = pm2.to_sparql_preamble();
        assert!(p2.contains("PREFIX ex:"));
        assert!(p2.contains("http://example.org/"));
    }

    // ── PhysicsProperty helpers ───────────────────────────────────────────────

    #[test]
    fn test_physics_property_local_names() {
        assert_eq!(PhysicsProperty::Mass.local_name(), "mass");
        assert_eq!(PhysicsProperty::Temperature.local_name(), "temperature");
        assert_eq!(
            PhysicsProperty::AngularVelocity.local_name(),
            "angularVelocity"
        );
    }

    #[test]
    fn test_physics_property_sparql_var() {
        assert_eq!(PhysicsProperty::Mass.sparql_var(), "?mass");
        assert_eq!(PhysicsProperty::Energy.sparql_var(), "?energy");
    }

    #[test]
    fn test_physics_property_custom() {
        let prop = PhysicsProperty::Custom("viscosity".to_string());
        assert_eq!(prop.local_name(), "viscosity");
        assert_eq!(prop.sparql_var(), "?viscosity");
    }
}
