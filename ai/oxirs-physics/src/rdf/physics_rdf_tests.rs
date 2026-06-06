#[cfg(test)]
mod tests {
    use crate::rdf::physics_rdf_mapper::{
        extract_double_literal, qudt_unit_for, sanitize_iri_fragment, strip_literal_quotes,
        PhysicsToRdf, RdfToPhysics,
    };
    use crate::rdf::physics_rdf_serializer::SparqlPhysicsQuery;
    use crate::rdf::physics_rdf_types::{
        Triple, NS_EX, NS_PHYS, NS_PROV, NS_QUDT, NS_RDF, NS_RDFS, NS_SOSA, NS_SSN, NS_UNIT, NS_XSD,
    };
    use crate::simulation::result_injection::{
        ConvergenceInfo, SimulationProvenance, SimulationResult, StateVector,
    };
    use chrono::Utc;
    use std::collections::HashMap;

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn make_state(time: f64, temperature: f64, pressure: f64) -> StateVector {
        let mut state = HashMap::new();
        state.insert("temperature".to_string(), temperature);
        state.insert("pressure".to_string(), pressure);
        StateVector { time, state }
    }

    fn make_result() -> SimulationResult {
        let trajectory = vec![
            make_state(0.0, 300.0, 101325.0),
            make_state(1.0, 350.0, 101325.0),
            make_state(2.0, 400.0, 102000.0),
        ];
        let mut derived = HashMap::new();
        derived.insert("max_temperature".to_string(), 400.0);
        derived.insert("pressure_drop".to_string(), 675.0);

        SimulationResult {
            entity_iri: "urn:example:reactor:1".to_string(),
            simulation_run_id: "run-abc-123".to_string(),
            timestamp: Utc::now(),
            state_trajectory: trajectory,
            derived_quantities: derived,
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations: 42,
                final_residual: 1e-8,
            },
            provenance: SimulationProvenance {
                software: "OxiRS-Physics".to_string(),
                version: "0.3.0".to_string(),
                parameters_hash: "abc123".to_string(),
                executed_at: Utc::now(),
                execution_time_ms: 1500,
            },
        }
    }

    fn make_bc_triples() -> Vec<Triple> {
        let rdf_type = format!("<{}type>", NS_RDF);
        let bc_iri = "<http://oxirs.org/example/physics#bc_inlet>";
        vec![
            Triple::new(bc_iri, rdf_type, format!("<{}BoundaryCondition>", NS_EX)),
            Triple::new(bc_iri, format!("<{}bcType>", NS_PHYS), "\"inlet\""),
            Triple::new(bc_iri, format!("<{}bcProperty>", NS_PHYS), "\"velocity\""),
            Triple::new(
                bc_iri,
                format!("<{}bcValue>", NS_PHYS),
                format!("\"1.5\"^^<{}double>", NS_XSD),
            ),
            Triple::new(bc_iri, format!("<{}bcUnit>", NS_PHYS), "\"M-PER-SEC\""),
        ]
    }

    fn make_material_triples() -> Vec<Triple> {
        let rdf_type = format!("<{}type>", NS_RDF);
        let mat_iri = "<http://oxirs.org/example/physics#material_steel>";
        vec![
            Triple::new(mat_iri, rdf_type, format!("<{}Material>", NS_EX)),
            Triple::new(mat_iri, format!("<{}label>", NS_RDFS), "\"Steel\""),
            Triple::new(
                mat_iri,
                format!("<{}value>", NS_PHYS),
                format!("\"50.2\"^^<{}double>", NS_XSD),
            ),
            Triple::new(mat_iri, format!("<{}unit>", NS_PHYS), "\"W-PER-M-K\""),
        ]
    }

    // ── PhysicsToRdf tests ────────────────────────────────────────────────────

    #[test]
    fn test_convert_produces_triples() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        assert!(!triples.is_empty(), "expected non-empty triples");
    }

    #[test]
    fn test_convert_contains_sosa_observations() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let obs_type = format!("<{}Observation>", NS_SOSA);
        let rdf_type_pred = format!("<{}type>", NS_RDF);
        let obs_count = triples
            .iter()
            .filter(|t| t.predicate == rdf_type_pred && t.object == obs_type)
            .count();
        // 3 timesteps × 2 properties + 2 derived = ≥8 observations
        assert!(
            obs_count >= 8,
            "expected ≥8 SOSA observations, got {obs_count}"
        );
    }

    #[test]
    fn test_convert_contains_digital_twin() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let dt_type = format!("<{}DigitalTwin>", NS_EX);
        let rdf_type_pred = format!("<{}type>", NS_RDF);
        let dt_count = triples
            .iter()
            .filter(|t| t.predicate == rdf_type_pred && t.object == dt_type)
            .count();
        assert_eq!(dt_count, 1, "expected exactly one DigitalTwin type triple");
    }

    #[test]
    fn test_convert_contains_qudt_units() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let qudt_unit_pred = format!("<{}unit>", NS_QUDT);
        let unit_count = triples
            .iter()
            .filter(|t| t.predicate == qudt_unit_pred)
            .count();
        assert!(unit_count > 0, "expected QUDT unit triples");
    }

    #[test]
    fn test_to_turtle_contains_prefixes() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let turtle = converter.to_turtle(&result);
        assert!(turtle.contains("@prefix sosa:"), "missing sosa prefix");
        assert!(turtle.contains("@prefix qudt:"), "missing qudt prefix");
        assert!(turtle.contains("@prefix prov:"), "missing prov prefix");
    }

    #[test]
    fn test_to_subject_map_groups_correctly() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let map = converter.to_subject_map(&result);
        assert!(!map.is_empty(), "expected non-empty subject map");
        for v in map.values() {
            assert!(
                !v.is_empty(),
                "subject group should have at least one triple"
            );
        }
    }

    #[test]
    fn test_convert_no_digital_twin() {
        let converter = PhysicsToRdf {
            include_digital_twin: false,
            ..PhysicsToRdf::new()
        };
        let result = make_result();
        let triples = converter.convert(&result);
        let dt_type = format!("<{}DigitalTwin>", NS_EX);
        let rdf_type_pred = format!("<{}type>", NS_RDF);
        let dt_count = triples
            .iter()
            .filter(|t| t.predicate == rdf_type_pred && t.object == dt_type)
            .count();
        assert_eq!(dt_count, 0);
    }

    #[test]
    fn test_convert_no_provenance() {
        let converter = PhysicsToRdf {
            include_provenance: false,
            ..PhysicsToRdf::new()
        };
        let result = make_result();
        let triples = converter.convert(&result);
        let activity_type = format!("<{}Activity>", NS_PROV);
        let rdf_type_pred = format!("<{}type>", NS_RDF);
        let prov_count = triples
            .iter()
            .filter(|t| t.predicate == rdf_type_pred && t.object == activity_type)
            .count();
        assert_eq!(prov_count, 0);
    }

    #[test]
    fn test_convert_contains_observed_property() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let obs_prop_pred = format!("<{}observedProperty>", NS_SOSA);
        assert!(triples.iter().any(|t| t.predicate == obs_prop_pred));
    }

    #[test]
    fn test_convert_has_simple_result_values() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let sr_pred = format!("<{}hasSimpleResult>", NS_SOSA);
        let values: Vec<f64> = triples
            .iter()
            .filter(|t| t.predicate == sr_pred)
            .filter_map(|t| extract_double_literal(&t.object))
            .collect();
        assert!(!values.is_empty(), "expected hasSimpleResult values");
        assert!(values.contains(&300.0), "expected temperature 300.0");
    }

    #[test]
    fn test_convert_contains_prov_activity() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let activity_type = format!("<{}Activity>", NS_PROV);
        let rdf_type_pred = format!("<{}type>", NS_RDF);
        assert!(
            triples
                .iter()
                .any(|t| t.predicate == rdf_type_pred && t.object == activity_type),
            "expected prov:Activity triple"
        );
    }

    // ── RdfToPhysics tests ────────────────────────────────────────────────────

    #[test]
    fn test_extract_boundary_conditions_basic() {
        let parser = RdfToPhysics::new();
        let triples = make_bc_triples();
        let bcs = parser
            .extract_boundary_conditions(&triples)
            .expect("should succeed");
        assert_eq!(bcs.len(), 1);
        let bc = &bcs[0];
        assert_eq!(bc.condition_type, "inlet");
        assert_eq!(bc.property, "velocity");
        assert!((bc.value - 1.5).abs() < 1e-10);
        assert_eq!(bc.unit, "M-PER-SEC");
    }

    #[test]
    fn test_extract_material_properties() {
        let parser = RdfToPhysics::new();
        let triples = make_material_triples();
        let mats = parser
            .extract_material_properties(&triples)
            .expect("should succeed");
        assert_eq!(mats.len(), 1);
        let mat = &mats[0];
        assert_eq!(mat.name, "Steel");
        assert!((mat.value - 50.2).abs() < 1e-10);
        assert_eq!(mat.unit, "W-PER-M-K");
    }

    #[test]
    fn test_extract_no_bcs_lenient() {
        let parser = RdfToPhysics {
            lenient: true,
            ..RdfToPhysics::default()
        };
        let triples = vec![Triple::new("<ex:foo>", "<ex:bar>", "<ex:baz>")];
        let bcs = parser
            .extract_boundary_conditions(&triples)
            .expect("should succeed");
        assert!(bcs.is_empty());
    }

    #[test]
    fn test_extract_no_bcs_strict() {
        let parser = RdfToPhysics {
            lenient: false,
            ..RdfToPhysics::default()
        };
        let triples = vec![Triple::new("<ex:foo>", "<ex:bar>", "<ex:baz>")];
        assert!(parser.extract_boundary_conditions(&triples).is_err());
    }

    #[test]
    fn test_extract_observations_from_converted() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let parser = RdfToPhysics::new();
        let obs = parser.extract_observations(&triples);
        assert!(
            !obs.is_empty(),
            "expected observations from converted result"
        );
    }

    // ── Roundtrip test ────────────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_observations_queryable() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let query = SparqlPhysicsQuery::new(&triples);
        let obs_count = query.count_observations();
        // 3 timesteps × 2 props + 2 derived = 8
        assert!(obs_count >= 8, "expected ≥8 observations, got {obs_count}");
    }

    // ── SparqlPhysicsQuery tests ──────────────────────────────────────────────

    #[test]
    fn test_get_max_temperature() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let max_temp = query.get_max_temperature();
        assert!(max_temp.is_some());
        let max = max_temp.expect("should succeed");
        assert!((max - 400.0).abs() < 1e-6, "expected 400.0, got {max}");
    }

    #[test]
    fn test_get_min_temperature() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let min_temp = query.get_min_for_property("temperature");
        assert!(min_temp.is_some());
        let min = min_temp.expect("should succeed");
        assert!((min - 300.0).abs() < 1e-6, "expected 300.0, got {min}");
    }

    #[test]
    fn test_get_mean_for_property() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let mean = query.get_mean_for_property("temperature");
        assert!(mean.is_some());
        // Mean of 300, 350, 400 = 350
        let m = mean.expect("should succeed");
        assert!((m - 350.0).abs() < 1e-6, "expected 350.0, got {m}");
    }

    #[test]
    fn test_get_observations_in_range() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let obs = query.get_observations_in_range(0.5, 1.5);
        // simTime=1.0 is in [0.5, 1.5]
        assert!(!obs.is_empty(), "expected observations in range");
    }

    #[test]
    fn test_get_observations_out_of_range() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let obs = query.get_observations_in_range(10.0, 20.0);
        assert!(
            obs.is_empty(),
            "expected no observations outside time range"
        );
    }

    #[test]
    fn test_list_observed_properties() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let props = query.list_observed_properties();
        assert!(!props.is_empty());
    }

    #[test]
    fn test_count_observations() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        assert!(query.count_observations() >= 8);
    }

    #[test]
    fn test_from_result_constructor() {
        let result = make_result();
        let query = SparqlPhysicsQuery::from_result(&result);
        assert!(query.count_observations() >= 8);
    }

    #[test]
    fn test_max_for_unknown_property_returns_none() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        assert!(query.get_max_for_property("nonexistent_xyz").is_none());
    }

    #[test]
    fn test_all_values_for_property() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let temps = query.all_values_for_property("temperature");
        assert_eq!(temps.len(), 3, "expected 3 temperature observations");
    }

    // ── Utility helper tests ──────────────────────────────────────────────────

    #[test]
    fn test_sanitize_iri_fragment() {
        assert_eq!(sanitize_iri_fragment("run-abc-123"), "run-abc-123");
        assert_eq!(sanitize_iri_fragment("urn:foo:bar"), "urn_foo_bar");
        assert_eq!(sanitize_iri_fragment("a b c"), "a_b_c");
    }

    #[test]
    fn test_extract_double_literal() {
        assert_eq!(extract_double_literal("\"1.5\"^^xsd:double"), Some(1.5));
        assert_eq!(
            extract_double_literal("\"300.0\"^^<http://www.w3.org/2001/XMLSchema#double>"),
            Some(300.0)
        );
        assert_eq!(extract_double_literal("\"notanumber\""), None);
    }

    #[test]
    fn test_strip_literal_quotes() {
        assert_eq!(strip_literal_quotes("\"hello\""), "hello");
        assert_eq!(strip_literal_quotes("\"1.5\"^^xsd:double"), "1.5");
    }

    #[test]
    fn test_triple_statement() {
        let t = Triple::new("<ex:s>", "<ex:p>", "<ex:o>");
        let stmt = t.to_turtle_statement();
        assert!(stmt.contains("<ex:s>"));
        assert!(stmt.contains("<ex:p>"));
        assert!(stmt.contains("<ex:o>"));
    }

    #[test]
    fn test_qudt_unit_for_known() {
        assert_eq!(qudt_unit_for("temperature"), "DEG_C");
        assert_eq!(qudt_unit_for("pressure"), "PA");
        assert_eq!(qudt_unit_for("mass"), "KiloGM");
    }

    #[test]
    fn test_qudt_unit_for_unknown() {
        assert_eq!(qudt_unit_for("some_made_up_quantity"), "UNITLESS");
    }

    // ── Additional PhysicsToRdf tests ─────────────────────────────────────────

    #[test]
    fn test_physics_to_rdf_no_provenance() {
        let conv = PhysicsToRdf {
            include_provenance: false,
            include_digital_twin: true,
            include_units: true,
            base_iri: NS_EX.to_string(),
        };
        let result = make_result();
        let triples = conv.convert(&result);
        let has_prov = triples.iter().any(|t| t.predicate.contains("prov#"));
        assert!(!has_prov, "expected no provenance triples");
    }

    #[test]
    fn test_physics_to_rdf_no_digital_twin() {
        let conv = PhysicsToRdf {
            include_provenance: true,
            include_digital_twin: false,
            include_units: true,
            base_iri: NS_EX.to_string(),
        };
        let result = make_result();
        let triples = conv.convert(&result);
        let has_dt = triples.iter().any(|t| t.object.contains("DigitalTwin"));
        assert!(!has_dt, "expected no DigitalTwin triples when disabled");
    }

    #[test]
    fn test_physics_to_rdf_no_units() {
        let conv = PhysicsToRdf {
            include_provenance: true,
            include_digital_twin: true,
            include_units: false,
            base_iri: NS_EX.to_string(),
        };
        let result = make_result();
        let triples = conv.convert(&result);
        let has_unit = triples
            .iter()
            .any(|t| t.predicate.contains("qudt.org/schema/qudt/unit"));
        assert!(!has_unit, "expected no qudt:unit triples when disabled");
    }

    #[test]
    fn test_to_turtle_contains_prefix() {
        let conv = PhysicsToRdf::new();
        let result = make_result();
        let turtle = conv.to_turtle(&result);
        assert!(
            turtle.contains("@prefix sosa:"),
            "turtle must have sosa prefix"
        );
        assert!(
            turtle.contains("@prefix qudt:"),
            "turtle must have qudt prefix"
        );
        assert!(
            turtle.contains("@prefix prov:"),
            "turtle must have prov prefix"
        );
    }

    #[test]
    fn test_to_subject_map_has_digital_twin() {
        let conv = PhysicsToRdf::new();
        let result = make_result();
        let map = conv.to_subject_map(&result);
        let has_dt_key = map.keys().any(|k| k.contains("dt_"));
        assert!(has_dt_key, "subject map must contain a dt_ key");
    }

    #[test]
    fn test_roundtrip_observation_count() {
        let conv = PhysicsToRdf::new();
        let result = make_result();
        let triples = conv.convert(&result);
        let obs_count = triples
            .iter()
            .filter(|t| t.object.contains("Observation>"))
            .count();
        // 3 time steps × 2 properties + 2 derived = 8 observations
        assert!(
            obs_count >= 8,
            "expected at least 8 observations, got {obs_count}"
        );
    }

    #[test]
    fn test_rdf_to_physics_extract_bc_strict_empty_error() {
        let parser = RdfToPhysics {
            phys_ns: NS_PHYS.to_string(),
            lenient: false,
        };
        let triples: Vec<Triple> = vec![];
        let result = parser.extract_boundary_conditions(&triples);
        assert!(result.is_err(), "strict mode must error on empty BC list");
    }

    #[test]
    fn test_rdf_to_physics_bc_type_and_value() {
        let parser = RdfToPhysics::new();
        let triples = make_bc_triples();
        let bcs = parser
            .extract_boundary_conditions(&triples)
            .expect("should succeed");
        assert_eq!(bcs.len(), 1);
        assert_eq!(bcs[0].condition_type, "inlet");
        assert_eq!(bcs[0].property, "velocity");
        assert!((bcs[0].value - 1.5).abs() < 1e-10);
        assert_eq!(bcs[0].unit, "M-PER-SEC");
    }

    #[test]
    fn test_rdf_to_physics_material_property_extraction() {
        let parser = RdfToPhysics::new();
        let triples = make_material_triples();
        let mats = parser
            .extract_material_properties(&triples)
            .expect("should succeed");
        assert!(!mats.is_empty(), "expected at least one material property");
        assert_eq!(mats[0].unit, "W-PER-M-K");
        assert!((mats[0].value - 50.2).abs() < 1e-6);
    }

    #[test]
    fn test_physics_to_rdf_roundtrip_extract_observations() {
        let conv = PhysicsToRdf::new();
        let result = make_result();
        let triples = conv.convert(&result);

        let parser = RdfToPhysics::new();
        let obs = parser.extract_observations(&triples);
        assert!(
            !obs.is_empty(),
            "should extract at least one observation from roundtrip"
        );
    }

    #[test]
    fn test_sparql_query_get_max_temperature() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let max = query.get_max_temperature();
        assert!(max.is_some());
        let v = max.expect("should succeed");
        assert!((v - 400.0).abs() < 1e-6, "expected max temp 400.0, got {v}");
    }

    #[test]
    fn test_sparql_query_get_min_for_property() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let min = query.get_min_for_property("temperature");
        assert!(min.is_some());
        let v = min.expect("should succeed");
        assert!((v - 300.0).abs() < 1e-6, "expected min temp 300.0, got {v}");
    }

    #[test]
    fn test_sparql_query_mean_for_property() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let mean = query.get_mean_for_property("pressure");
        assert!(mean.is_some());
        // pressure values: 101325, 101325, 102000 → mean ≈ 101550
        let m = mean.expect("should succeed");
        assert!(m > 101000.0 && m < 103000.0, "unexpected mean pressure {m}");
    }

    #[test]
    fn test_triple_eq() {
        let t1 = Triple::new("<ex:s>", "<ex:p>", "<ex:o>");
        let t2 = Triple::new("<ex:s>", "<ex:p>", "<ex:o>");
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_triple_clone() {
        let t = Triple::new("<ex:s>", "<ex:p>", "<ex:o>");
        let c = t.clone();
        assert_eq!(t, c);
    }

    #[test]
    fn test_namespace_constants_non_empty() {
        use crate::rdf::physics_rdf_types::{NS_RDF, NS_RDFS};
        assert!(!NS_SOSA.is_empty());
        assert!(!NS_SSN.is_empty());
        assert!(!NS_QUDT.is_empty());
        assert!(!NS_UNIT.is_empty());
        assert!(!NS_EX.is_empty());
        assert!(!NS_PHYS.is_empty());
        assert!(!NS_PROV.is_empty());
        assert!(!NS_XSD.is_empty());
        assert!(!NS_RDF.is_empty());
        assert!(!NS_RDFS.is_empty());
    }

    #[test]
    fn test_rdf_bc_iri_preserved() {
        let parser = RdfToPhysics::new();
        let triples = make_bc_triples();
        let bcs = parser
            .extract_boundary_conditions(&triples)
            .expect("should succeed");
        assert!(!bcs[0].iri.is_empty(), "BC IRI should be non-empty");
        assert!(
            bcs[0].iri.contains("bc_inlet"),
            "IRI should contain bc_inlet"
        );
    }

    #[test]
    fn test_physics_to_rdf_empty_trajectory() {
        let conv = PhysicsToRdf::new();
        let result = SimulationResult {
            entity_iri: "urn:example:empty:1".to_string(),
            simulation_run_id: "run-empty".to_string(),
            timestamp: chrono::Utc::now(),
            state_trajectory: vec![],
            derived_quantities: HashMap::new(),
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations: 0,
                final_residual: 0.0,
            },
            provenance: SimulationProvenance {
                software: "OxiRS".to_string(),
                version: "0.2.0".to_string(),
                parameters_hash: "0".to_string(),
                executed_at: chrono::Utc::now(),
                execution_time_ms: 0,
            },
        };
        let triples = conv.convert(&result);
        // Digital twin triple should still be generated
        let has_dt = triples.iter().any(|t| t.object.contains("DigitalTwin"));
        assert!(
            has_dt,
            "should produce DigitalTwin triple even with empty trajectory"
        );
    }

    #[test]
    fn test_physics_to_rdf_default_base_iri_is_ex() {
        let conv = PhysicsToRdf::default();
        assert_eq!(conv.base_iri, NS_EX);
    }

    #[test]
    fn test_to_turtle_newlines_per_triple() {
        let conv = PhysicsToRdf::new();
        let result = make_result();
        let turtle = conv.to_turtle(&result);
        // Each triple ends with " .\n"
        let triple_count = turtle.matches(" .").count();
        assert!(
            triple_count >= 10,
            "expected at least 10 triples in turtle, got {triple_count}"
        );
    }

    #[test]
    fn test_qudt_unit_velocity() {
        assert_eq!(qudt_unit_for("velocity"), "M-PER-SEC");
        assert_eq!(qudt_unit_for("velocity_x"), "M-PER-SEC");
        assert_eq!(qudt_unit_for("velocity_y"), "M-PER-SEC");
    }

    #[test]
    fn test_qudt_unit_various() {
        assert_eq!(qudt_unit_for("density"), "KiloGM-PER-M3");
        assert_eq!(qudt_unit_for("voltage"), "V");
        assert_eq!(qudt_unit_for("entropy"), "J-PER-K");
        assert_eq!(qudt_unit_for("frequency"), "HZ");
    }

    #[test]
    fn test_rdf_material_empty_ok() {
        let parser = RdfToPhysics::new();
        let triples: Vec<Triple> = vec![];
        let mats = parser
            .extract_material_properties(&triples)
            .expect("should succeed");
        assert!(mats.is_empty(), "empty triples -> empty material list");
    }
}
