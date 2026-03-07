//! NGSI-LD Query to SPARQL Translation
//!
//! Translates NGSI-LD query expressions to SPARQL queries.

use super::converter::NGSI_LD_PREFIX;
use super::types::{GeoQuery, NgsiError, NgsiQueryParams, TemporalQuery};

/// Query translator for NGSI-LD to SPARQL
pub struct NgsiQueryTranslator {
    /// Default graph name
    default_graph: Option<String>,
}

impl Default for NgsiQueryTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl NgsiQueryTranslator {
    /// Create a new query translator
    pub fn new() -> Self {
        Self {
            default_graph: None,
        }
    }

    /// Set the default graph
    pub fn with_graph(mut self, graph: impl Into<String>) -> Self {
        self.default_graph = Some(graph.into());
        self
    }

    /// Translate NGSI-LD query parameters to SPARQL SELECT
    pub fn translate_query(&self, params: &NgsiQueryParams) -> Result<String, NgsiError> {
        let mut sparql = String::new();

        // Prefixes
        sparql.push_str("PREFIX ngsi-ld: <https://uri.etsi.org/ngsi-ld/>\n");
        sparql.push_str("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n");
        sparql.push_str("PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n\n");

        // SELECT clause
        sparql.push_str("SELECT DISTINCT ?entity ?type ?property ?value\n");

        // FROM clause if graph specified
        if let Some(ref graph) = self.default_graph {
            sparql.push_str(&format!("FROM <{}>\n", graph));
        }

        // WHERE clause
        sparql.push_str("WHERE {\n");

        // Entity type filter
        if let Some(ref entity_type) = params.entity_type {
            let type_uri = self.expand_type(entity_type);
            sparql.push_str(&format!("  ?entity rdf:type <{}> .\n", type_uri));
        } else {
            sparql.push_str("  ?entity rdf:type ?type .\n");
        }

        // Entity ID filter
        if let Some(ref id) = params.id {
            let ids: Vec<&str> = id.split(',').map(|s| s.trim()).collect();
            if ids.len() == 1 {
                sparql.push_str(&format!("  FILTER(?entity = <{}>)\n", ids[0]));
            } else {
                let values: String = ids
                    .iter()
                    .map(|id| format!("<{}>", id))
                    .collect::<Vec<_>>()
                    .join(" ");
                sparql.push_str(&format!("  FILTER(?entity IN ({}))\n", values));
            }
        }

        // ID pattern filter
        if let Some(ref pattern) = params.id_pattern {
            sparql.push_str(&format!("  FILTER(REGEX(STR(?entity), \"{}\"))\n", pattern));
        }

        // NGSI-LD query expression (q parameter)
        if let Some(ref q) = params.q {
            let filter = self.translate_q_expression(q)?;
            sparql.push_str(&format!("  {}\n", filter));
        }

        // Optional: get all properties
        sparql.push_str("  OPTIONAL {\n");
        sparql.push_str("    ?entity ?property ?propNode .\n");
        sparql.push_str("    ?propNode ngsi-ld:hasValue ?value .\n");
        sparql.push_str("  }\n");

        sparql.push_str("}\n");

        // Pagination
        if let Some(limit) = params.limit {
            sparql.push_str(&format!("LIMIT {}\n", limit));
        }
        if let Some(offset) = params.offset {
            sparql.push_str(&format!("OFFSET {}\n", offset));
        }

        Ok(sparql)
    }

    /// Translate NGSI-LD query for a specific entity
    pub fn translate_entity_query(&self, entity_id: &str) -> Result<String, NgsiError> {
        let mut sparql = String::new();

        sparql.push_str("PREFIX ngsi-ld: <https://uri.etsi.org/ngsi-ld/>\n");
        sparql.push_str("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n\n");

        sparql.push_str("SELECT ?predicate ?object\n");

        if let Some(ref graph) = self.default_graph {
            sparql.push_str(&format!("FROM <{}>\n", graph));
        }

        sparql.push_str("WHERE {\n");
        sparql.push_str(&format!("  <{}> ?predicate ?object .\n", entity_id));
        sparql.push_str("}\n");

        Ok(sparql)
    }

    /// Translate geo-query to SPARQL with GeoSPARQL
    pub fn translate_geo_query(&self, geo_q: &GeoQuery) -> Result<String, NgsiError> {
        let geo_property = geo_q
            .geoproperty
            .as_deref()
            .unwrap_or("https://uri.etsi.org/ngsi-ld/location");

        // Parse georel
        let georel_parts: Vec<&str> = geo_q.georel.split(';').collect();
        let relation = georel_parts.first().unwrap_or(&"near");

        let filter = match *relation {
            "near" => {
                // Extract maxDistance
                let max_distance = georel_parts
                    .iter()
                    .find(|p| p.starts_with("maxDistance=="))
                    .and_then(|p| p.strip_prefix("maxDistance=="))
                    .and_then(|v| v.parse::<f64>().ok())
                    .unwrap_or(1000.0);

                format!(
                    "FILTER(geof:distance(?geo, \"{}\"^^geo:wktLiteral) < {})",
                    self.geojson_to_wkt(&geo_q.geometry, &geo_q.coordinates)?,
                    max_distance
                )
            }
            "within" => {
                format!(
                    "FILTER(geof:sfWithin(?geo, \"{}\"^^geo:wktLiteral))",
                    self.geojson_to_wkt(&geo_q.geometry, &geo_q.coordinates)?
                )
            }
            "contains" => {
                format!(
                    "FILTER(geof:sfContains(?geo, \"{}\"^^geo:wktLiteral))",
                    self.geojson_to_wkt(&geo_q.geometry, &geo_q.coordinates)?
                )
            }
            "intersects" => {
                format!(
                    "FILTER(geof:sfIntersects(?geo, \"{}\"^^geo:wktLiteral))",
                    self.geojson_to_wkt(&geo_q.geometry, &geo_q.coordinates)?
                )
            }
            "equals" => {
                format!(
                    "FILTER(geof:sfEquals(?geo, \"{}\"^^geo:wktLiteral))",
                    self.geojson_to_wkt(&geo_q.geometry, &geo_q.coordinates)?
                )
            }
            "disjoint" => {
                format!(
                    "FILTER(geof:sfDisjoint(?geo, \"{}\"^^geo:wktLiteral))",
                    self.geojson_to_wkt(&geo_q.geometry, &geo_q.coordinates)?
                )
            }
            "overlaps" => {
                format!(
                    "FILTER(geof:sfOverlaps(?geo, \"{}\"^^geo:wktLiteral))",
                    self.geojson_to_wkt(&geo_q.geometry, &geo_q.coordinates)?
                )
            }
            _ => {
                return Err(NgsiError::InvalidRequest(format!(
                    "Unknown geo-relation: {}",
                    relation
                )))
            }
        };

        let mut sparql = String::new();
        sparql.push_str(&format!("?entity <{}> ?geoNode .\n", geo_property));
        sparql.push_str("?geoNode ngsi-ld:hasValue ?geo .\n");
        sparql.push_str(&format!("{}\n", filter));

        Ok(sparql)
    }

    /// Translate temporal query to SPARQL
    pub fn translate_temporal_query(
        &self,
        temporal_q: &TemporalQuery,
    ) -> Result<String, NgsiError> {
        let time_property = temporal_q.timeproperty.as_deref().unwrap_or("observedAt");

        let time_pred = format!("{}{}", NGSI_LD_PREFIX, time_property);

        let filter = match temporal_q.timerel {
            super::types::TimeRelation::Before => {
                if let Some(time_at) = temporal_q.time_at {
                    format!("FILTER(?time < \"{}\"^^xsd:dateTime)", time_at.to_rfc3339())
                } else {
                    return Err(NgsiError::InvalidRequest(
                        "timeAt required for 'before'".to_string(),
                    ));
                }
            }
            super::types::TimeRelation::After => {
                if let Some(time_at) = temporal_q.time_at {
                    format!("FILTER(?time > \"{}\"^^xsd:dateTime)", time_at.to_rfc3339())
                } else {
                    return Err(NgsiError::InvalidRequest(
                        "timeAt required for 'after'".to_string(),
                    ));
                }
            }
            super::types::TimeRelation::Between => {
                let start = temporal_q.time_at.ok_or_else(|| {
                    NgsiError::InvalidRequest("timeAt required for 'between'".to_string())
                })?;
                let end = temporal_q.end_time_at.ok_or_else(|| {
                    NgsiError::InvalidRequest("endTimeAt required for 'between'".to_string())
                })?;
                format!(
                    "FILTER(?time >= \"{}\"^^xsd:dateTime && ?time <= \"{}\"^^xsd:dateTime)",
                    start.to_rfc3339(),
                    end.to_rfc3339()
                )
            }
        };

        let mut sparql = String::new();
        sparql.push_str(&format!("?propNode <{}> ?time .\n", time_pred));
        sparql.push_str(&format!("{}\n", filter));

        Ok(sparql)
    }

    /// Translate NGSI-LD q expression to SPARQL FILTER
    fn translate_q_expression(&self, q: &str) -> Result<String, NgsiError> {
        // Parse simple expressions: attr==value, attr!=value, attr>value, etc.
        // Also handle: attr1==value1;attr2==value2 (AND) and attr1==value1|attr2==value2 (OR)

        let mut filters = Vec::new();

        // Split by semicolon (AND) first
        for and_part in q.split(';') {
            let mut or_filters = Vec::new();

            // Split by pipe (OR)
            for or_part in and_part.split('|') {
                if let Some(filter) = self.parse_simple_expression(or_part.trim())? {
                    or_filters.push(filter);
                }
            }

            if !or_filters.is_empty() {
                if or_filters.len() == 1 {
                    filters.push(or_filters.pop().expect("or_filters should not be empty"));
                } else {
                    filters.push(format!("({})", or_filters.join(" || ")));
                }
            }
        }

        if filters.is_empty() {
            Ok(String::new())
        } else if filters.len() == 1 {
            Ok(format!(
                "FILTER({})",
                filters.pop().expect("filters should not be empty")
            ))
        } else {
            Ok(format!(
                "FILTER({} && {})",
                filters[0],
                filters[1..].join(" && ")
            ))
        }
    }

    /// Parse a simple comparison expression
    fn parse_simple_expression(&self, expr: &str) -> Result<Option<String>, NgsiError> {
        // Handle different operators
        let operators = ["==", "!=", ">=", "<=", ">", "<", "~="];

        for op in operators {
            if let Some((attr, value)) = expr.split_once(op) {
                let attr = attr.trim();
                let value = value.trim();
                let prop_var = format!("?{}", attr.replace(['.', '-'], "_"));

                // Generate pattern to bind the property value
                let pattern = format!(
                    "?entity <{}{}> ?{}Node . ?{}Node ngsi-ld:hasValue {}",
                    NGSI_LD_PREFIX,
                    attr,
                    attr.replace(['.', '-'], "_"),
                    attr.replace(['.', '-'], "_"),
                    prop_var
                );

                let comparison = match op {
                    "==" => format!("{} = {}", prop_var, self.format_value(value)),
                    "!=" => format!("{} != {}", prop_var, self.format_value(value)),
                    ">=" => format!("{} >= {}", prop_var, self.format_value(value)),
                    "<=" => format!("{} <= {}", prop_var, self.format_value(value)),
                    ">" => format!("{} > {}", prop_var, self.format_value(value)),
                    "<" => format!("{} < {}", prop_var, self.format_value(value)),
                    "~=" => format!("REGEX(STR({}), \"{}\")", prop_var, value),
                    _ => continue,
                };

                return Ok(Some(format!(
                    "EXISTS {{ {} . FILTER({}) }}",
                    pattern, comparison
                )));
            }
        }

        // Handle unary expressions (attr - property exists)
        if !expr.contains(['=', '>', '<', '~']) {
            let attr = expr.trim();
            if !attr.is_empty() {
                return Ok(Some(format!(
                    "EXISTS {{ ?entity <{}{}> ?_ }}",
                    NGSI_LD_PREFIX, attr
                )));
            }
        }

        Ok(None)
    }

    /// Format a value for SPARQL
    fn format_value(&self, value: &str) -> String {
        // Try to detect type - numeric, boolean, or already quoted values are passed through
        if value.parse::<i64>().is_ok()
            || value.parse::<f64>().is_ok()
            || value == "true"
            || value == "false"
            || (value.starts_with('"') && value.ends_with('"'))
        {
            value.to_string()
        } else {
            format!("\"{}\"", value)
        }
    }

    /// Expand a type name to full URI
    fn expand_type(&self, type_name: &str) -> String {
        if type_name.contains("://") || type_name.starts_with("urn:") {
            type_name.to_string()
        } else {
            format!("{}{}", NGSI_LD_PREFIX, type_name)
        }
    }

    /// Convert GeoJSON to WKT
    fn geojson_to_wkt(
        &self,
        geometry: &str,
        coordinates: &serde_json::Value,
    ) -> Result<String, NgsiError> {
        match geometry.to_uppercase().as_str() {
            "POINT" => {
                if let Some(arr) = coordinates.as_array() {
                    if arr.len() >= 2 {
                        let lon = arr[0].as_f64().unwrap_or(0.0);
                        let lat = arr[1].as_f64().unwrap_or(0.0);
                        return Ok(format!("POINT({} {})", lon, lat));
                    }
                }
                Err(NgsiError::InvalidRequest(
                    "Invalid Point coordinates".to_string(),
                ))
            }
            "POLYGON" => {
                if let Some(rings) = coordinates.as_array() {
                    if let Some(ring) = rings.first().and_then(|r| r.as_array()) {
                        let coords: Vec<String> = ring
                            .iter()
                            .filter_map(|c| {
                                c.as_array().map(|arr| {
                                    let lon = arr.first().and_then(|v| v.as_f64()).unwrap_or(0.0);
                                    let lat = arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
                                    format!("{} {}", lon, lat)
                                })
                            })
                            .collect();
                        return Ok(format!("POLYGON(({}))", coords.join(", ")));
                    }
                }
                Err(NgsiError::InvalidRequest(
                    "Invalid Polygon coordinates".to_string(),
                ))
            }
            _ => Err(NgsiError::InvalidRequest(format!(
                "Unsupported geometry type: {}",
                geometry
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_simple_query() {
        let translator = NgsiQueryTranslator::new();
        let params = NgsiQueryParams {
            entity_type: Some("Vehicle".to_string()),
            limit: Some(10),
            ..Default::default()
        };

        let sparql = translator.translate_query(&params).unwrap();

        assert!(sparql.contains("SELECT DISTINCT"));
        assert!(sparql.contains("rdf:type"));
        assert!(sparql.contains("Vehicle"));
        assert!(sparql.contains("LIMIT 10"));
    }

    #[test]
    fn test_translate_entity_query() {
        let translator = NgsiQueryTranslator::new();
        let sparql = translator
            .translate_entity_query("urn:ngsi-ld:Vehicle:A123")
            .unwrap();

        assert!(sparql.contains("SELECT ?predicate ?object"));
        assert!(sparql.contains("urn:ngsi-ld:Vehicle:A123"));
    }

    #[test]
    fn test_translate_q_expression() {
        let translator = NgsiQueryTranslator::new();

        let filter = translator.translate_q_expression("speed==80").unwrap();
        assert!(filter.contains("FILTER"));
        assert!(filter.contains("speed"));
    }

    #[test]
    fn test_geojson_to_wkt() {
        let translator = NgsiQueryTranslator::new();

        let wkt = translator
            .geojson_to_wkt("Point", &serde_json::json!([139.7673, 35.6809]))
            .unwrap();
        assert_eq!(wkt, "POINT(139.7673 35.6809)");
    }
}
