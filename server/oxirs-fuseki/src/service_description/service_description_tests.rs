//! Unit tests for the service_description module.

#[cfg(test)]
mod tests {
    use crate::service_description::service_description_serializer::escape_turtle_string;
    use crate::service_description::{
        EntailmentRegime, SdFeature, SdInputFormat, SdLanguage, SdResultFormat, ServiceDescription,
        RDF_NS, SD_NS,
    };

    fn default_sd() -> ServiceDescription {
        ServiceDescription::default_for_oxirs("http://localhost:3030/sparql", "default")
    }

    // ---- Construction tests -----------------------------------------------

    #[test]
    fn test_default_oxirs_endpoint_url() {
        let sd = default_sd();
        assert_eq!(sd.endpoint_url, "http://localhost:3030/sparql");
    }

    #[test]
    fn test_default_oxirs_dataset_name() {
        let sd = default_sd();
        assert_eq!(sd.dataset_name, "default");
    }

    #[test]
    fn test_default_oxirs_has_label() {
        let sd = default_sd();
        assert!(sd.label.is_some());
        assert!(sd.label.as_ref().unwrap().contains("OxiRS"));
    }

    #[test]
    fn test_default_oxirs_has_description() {
        let sd = default_sd();
        assert!(sd.description.is_some());
        assert!(sd.description.as_ref().unwrap().contains("Rust"));
    }

    #[test]
    fn test_default_has_sparql11_query() {
        let sd = default_sd();
        assert!(sd.supports_sparql11());
    }

    #[test]
    fn test_default_has_sparql12_query() {
        let sd = default_sd();
        assert!(sd.supports_sparql12());
    }

    #[test]
    fn test_default_has_sparql11_update() {
        let sd = default_sd();
        assert!(sd.supports_update());
    }

    #[test]
    fn test_default_has_sparql10_query() {
        let sd = default_sd();
        assert!(sd.supports_language(SdLanguage::Sparql10Query.as_iri()));
    }

    #[test]
    fn test_default_feature_basic_federated_query() {
        let sd = default_sd();
        assert!(sd.has_feature(SdFeature::BasicFederatedQuery.as_iri()));
    }

    #[test]
    fn test_default_feature_union_default_graph() {
        let sd = default_sd();
        assert!(sd.has_feature(SdFeature::UnionDefaultGraph.as_iri()));
    }

    #[test]
    fn test_default_feature_dereferences_uris() {
        let sd = default_sd();
        assert!(sd.has_feature(SdFeature::DereferencesUris.as_iri()));
    }

    #[test]
    fn test_default_feature_empty_graphs() {
        let sd = default_sd();
        assert!(sd.has_feature(SdFeature::EmptyGraphs.as_iri()));
    }

    #[test]
    fn test_default_feature_rdf_star() {
        let sd = default_sd();
        assert!(sd.has_feature(SdFeature::RdfStar.as_iri()));
    }

    #[test]
    fn test_default_feature_sparql_star() {
        let sd = default_sd();
        assert!(sd.has_feature(SdFeature::SparqlStar.as_iri()));
    }

    #[test]
    fn test_default_result_format_json() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/sparql-results+json"));
    }

    #[test]
    fn test_default_result_format_xml() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/sparql-results+xml"));
    }

    #[test]
    fn test_default_result_format_csv() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("text/csv"));
    }

    #[test]
    fn test_default_result_format_tsv() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("text/tab-separated-values"));
    }

    #[test]
    fn test_default_result_format_turtle() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("text/turtle"));
    }

    #[test]
    fn test_default_result_format_ntriples() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/n-triples"));
    }

    #[test]
    fn test_default_result_format_nquads() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/n-quads"));
    }

    #[test]
    fn test_default_result_format_trig() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/trig"));
    }

    #[test]
    fn test_default_result_format_jsonld() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/ld+json"));
    }

    #[test]
    fn test_default_result_format_rdfxml() {
        let sd = default_sd();
        assert!(sd.supports_result_mime("application/rdf+xml"));
    }

    #[test]
    fn test_default_input_format_turtle() {
        let sd = default_sd();
        assert!(sd.accepts_input_mime("text/turtle"));
    }

    #[test]
    fn test_default_input_format_jsonld() {
        let sd = default_sd();
        assert!(sd.accepts_input_mime("application/ld+json"));
    }

    #[test]
    fn test_default_entailment_simple() {
        let sd = default_sd();
        assert!(sd.supports_entailment(&EntailmentRegime::Simple));
    }

    #[test]
    fn test_default_entailment_rdf() {
        let sd = default_sd();
        assert!(sd.supports_entailment(&EntailmentRegime::Rdf));
    }

    #[test]
    fn test_default_entailment_rdfs() {
        let sd = default_sd();
        assert!(sd.supports_entailment(&EntailmentRegime::Rdfs));
    }

    #[test]
    fn test_result_format_count_minimum() {
        let sd = default_sd();
        assert!(sd.result_format_count() >= 10);
    }

    #[test]
    fn test_input_format_count_minimum() {
        let sd = default_sd();
        assert!(sd.input_format_count() >= 6);
    }

    // ---- Turtle output tests -----------------------------------------------

    #[test]
    fn test_turtle_starts_with_prefix() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("@prefix sd:"));
    }

    #[test]
    fn test_turtle_contains_rdf_prefix() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("@prefix rdf:"));
    }

    #[test]
    fn test_turtle_contains_rdfs_prefix() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("@prefix rdfs:"));
    }

    #[test]
    fn test_turtle_contains_service_type() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("a sd:Service"));
    }

    #[test]
    fn test_turtle_contains_endpoint_url() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("http://localhost:3030/sparql"));
    }

    #[test]
    fn test_turtle_contains_sd_endpoint() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:endpoint"));
    }

    #[test]
    fn test_turtle_contains_sparql11_query_language() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:supportedLanguage"));
        assert!(turtle.contains("SPARQL11Query"));
    }

    #[test]
    fn test_turtle_contains_sparql12_query_language() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("SPARQL12Query"));
    }

    #[test]
    fn test_turtle_contains_result_format() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:resultFormat"));
    }

    #[test]
    fn test_turtle_contains_input_format() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:inputFormat"));
    }

    #[test]
    fn test_turtle_contains_feature() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:feature"));
    }

    #[test]
    fn test_turtle_contains_entailment_regime() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:defaultEntailmentRegime"));
    }

    #[test]
    fn test_turtle_contains_dataset_block() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:defaultDataset"));
        assert!(turtle.contains("sd:Dataset"));
    }

    #[test]
    fn test_turtle_contains_default_graph() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("sd:defaultGraph"));
        assert!(turtle.contains("sd:Graph"));
    }

    #[test]
    fn test_turtle_contains_label() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("rdfs:label"));
        assert!(turtle.contains("OxiRS"));
    }

    #[test]
    fn test_turtle_contains_comment() {
        let turtle = default_sd().to_turtle();
        assert!(turtle.contains("rdfs:comment"));
    }

    #[test]
    fn test_turtle_dataset_name_embedded() {
        let sd =
            ServiceDescription::default_for_oxirs("http://example.org/sparql", "my-knowledge-base");
        let turtle = sd.to_turtle();
        assert!(turtle.contains("my-knowledge-base"));
    }

    #[test]
    fn test_turtle_nonempty() {
        let turtle = default_sd().to_turtle();
        assert!(!turtle.is_empty());
        assert!(turtle.len() > 200);
    }

    #[test]
    fn test_turtle_extension_functions_included() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "test")
            .with_language(SdLanguage::Sparql11Query)
            .with_extension_function("http://example.org/functions/myFunc")
            .build();
        let turtle = sd.to_turtle();
        assert!(turtle.contains("sd:extensionFunction"));
        assert!(turtle.contains("myFunc"));
    }

    // ---- JSON-LD output tests -----------------------------------------------

    #[test]
    fn test_jsonld_is_object() {
        let val = default_sd().to_json_ld();
        assert!(val.is_object());
    }

    #[test]
    fn test_jsonld_has_context() {
        let val = default_sd().to_json_ld();
        assert!(val.get("@context").is_some());
    }

    #[test]
    fn test_jsonld_context_has_sd() {
        let val = default_sd().to_json_ld();
        let ctx = val["@context"].as_object().expect("context is object");
        assert!(ctx.contains_key("sd"));
    }

    #[test]
    fn test_jsonld_context_has_rdfs() {
        let val = default_sd().to_json_ld();
        let ctx = val["@context"].as_object().expect("context is object");
        assert!(ctx.contains_key("rdfs"));
    }

    #[test]
    fn test_jsonld_has_graph() {
        let val = default_sd().to_json_ld();
        assert!(val.get("@graph").is_some());
    }

    #[test]
    fn test_jsonld_graph_nonempty() {
        let val = default_sd().to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        assert!(!graph.is_empty());
    }

    #[test]
    fn test_jsonld_service_has_id() {
        let val = default_sd().to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        let service = &graph[0];
        assert!(service.get("@id").is_some());
    }

    #[test]
    fn test_jsonld_service_id_is_endpoint_url() {
        let sd = ServiceDescription::default_for_oxirs("http://localhost:3030/sparql", "default");
        let val = sd.to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        let id = graph[0]["@id"].as_str().expect("id is string");
        assert_eq!(id, "http://localhost:3030/sparql");
    }

    #[test]
    fn test_jsonld_service_has_type() {
        let val = default_sd().to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        let service_type = graph[0]["@type"].as_str().expect("@type is string");
        assert!(service_type.contains("Service"));
    }

    #[test]
    fn test_jsonld_has_supported_language() {
        let val = default_sd().to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        let service = &graph[0];
        let lang_key = "http://www.w3.org/ns/sparql-service-description#supportedLanguage";
        assert!(service.get(lang_key).is_some());
    }

    #[test]
    fn test_jsonld_has_result_format() {
        let val = default_sd().to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        let service = &graph[0];
        let fmt_key = "http://www.w3.org/ns/sparql-service-description#resultFormat";
        assert!(service.get(fmt_key).is_some());
    }

    #[test]
    fn test_jsonld_has_feature() {
        let val = default_sd().to_json_ld();
        let graph = val["@graph"].as_array().expect("graph is array");
        let service = &graph[0];
        let feat_key = "http://www.w3.org/ns/sparql-service-description#feature";
        assert!(service.get(feat_key).is_some());
    }

    // ---- RDF/XML output tests -----------------------------------------------

    #[test]
    fn test_rdfxml_starts_with_xml_declaration() {
        let xml = default_sd().to_rdf_xml();
        assert!(xml.starts_with("<?xml"));
    }

    #[test]
    fn test_rdfxml_contains_rdf_rdf() {
        let xml = default_sd().to_rdf_xml();
        assert!(xml.contains("<rdf:RDF"));
    }

    #[test]
    fn test_rdfxml_contains_service_element() {
        let xml = default_sd().to_rdf_xml();
        assert!(xml.contains("<sd:Service"));
    }

    #[test]
    fn test_rdfxml_contains_endpoint_url() {
        let xml = default_sd().to_rdf_xml();
        assert!(xml.contains("http://localhost:3030/sparql"));
    }

    // ---- Builder pattern tests ----------------------------------------------

    #[test]
    fn test_builder_minimal() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "ds").build();
        assert_eq!(sd.endpoint_url, "http://example.org/sparql");
        assert_eq!(sd.dataset_name, "ds");
        assert!(sd.features.is_empty());
    }

    #[test]
    fn test_builder_with_single_feature() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "ds")
            .with_feature(SdFeature::EmptyGraphs)
            .build();
        assert_eq!(sd.features.len(), 1);
        assert!(sd.has_feature(SdFeature::EmptyGraphs.as_iri()));
    }

    #[test]
    fn test_builder_with_multiple_features() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "ds")
            .with_features([SdFeature::EmptyGraphs, SdFeature::BasicFederatedQuery])
            .build();
        assert_eq!(sd.features.len(), 2);
    }

    #[test]
    fn test_builder_with_extension_function() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "ds")
            .with_extension_function("http://example.org/fn/custom")
            .build();
        assert_eq!(sd.extension_functions.len(), 1);
        assert_eq!(sd.extension_functions[0], "http://example.org/fn/custom");
    }

    #[test]
    fn test_builder_with_entailment_owl_rl() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "ds")
            .with_entailment_regime(EntailmentRegime::Owl2Rl)
            .build();
        assert!(sd.supports_entailment(&EntailmentRegime::Owl2Rl));
    }

    #[test]
    fn test_builder_label_and_description() {
        let sd = ServiceDescription::builder("http://example.org/sparql", "ds")
            .with_label("My Endpoint")
            .with_description("A test endpoint")
            .build();
        assert_eq!(sd.label.as_deref(), Some("My Endpoint"));
        assert_eq!(sd.description.as_deref(), Some("A test endpoint"));
    }

    // ---- Merge tests --------------------------------------------------------

    #[test]
    fn test_merge_adds_missing_features() {
        let mut sd1 = ServiceDescription::builder("http://a.org/sparql", "a")
            .with_feature(SdFeature::EmptyGraphs)
            .build();
        let sd2 = ServiceDescription::builder("http://b.org/sparql", "b")
            .with_feature(SdFeature::BasicFederatedQuery)
            .build();
        sd1.merge(sd2);
        assert_eq!(sd1.features.len(), 2);
    }

    #[test]
    fn test_merge_no_duplicate_features() {
        let mut sd1 = ServiceDescription::builder("http://a.org/sparql", "a")
            .with_feature(SdFeature::EmptyGraphs)
            .build();
        let sd2 = ServiceDescription::builder("http://b.org/sparql", "b")
            .with_feature(SdFeature::EmptyGraphs)
            .build();
        sd1.merge(sd2);
        assert_eq!(sd1.features.len(), 1);
    }

    #[test]
    fn test_merge_adds_missing_languages() {
        let mut sd1 = ServiceDescription::builder("http://a.org/sparql", "a")
            .with_language(SdLanguage::Sparql11Query)
            .build();
        let sd2 = ServiceDescription::builder("http://b.org/sparql", "b")
            .with_language(SdLanguage::Sparql11Update)
            .build();
        sd1.merge(sd2);
        assert_eq!(sd1.supported_languages.len(), 2);
    }

    #[test]
    fn test_merge_adds_extension_functions() {
        let mut sd1 = ServiceDescription::builder("http://a.org/sparql", "a")
            .with_extension_function("http://example.org/fn/f1")
            .build();
        let sd2 = ServiceDescription::builder("http://b.org/sparql", "b")
            .with_extension_function("http://example.org/fn/f2")
            .build();
        sd1.merge(sd2);
        assert_eq!(sd1.extension_functions.len(), 2);
    }

    // ---- IRI / MIME type tests ----------------------------------------------

    #[test]
    fn test_feature_iris_nonempty() {
        let features = [
            SdFeature::DereferencesUris,
            SdFeature::UnionDefaultGraph,
            SdFeature::RequiresDataset,
            SdFeature::EmptyGraphs,
            SdFeature::BasicFederatedQuery,
            SdFeature::RdfStar,
            SdFeature::SparqlStar,
            SdFeature::ConstraintValidation,
            SdFeature::TimeoutHints,
        ];
        for feat in features {
            assert!(
                !feat.as_iri().is_empty(),
                "Feature IRI empty for {:?}",
                feat
            );
        }
    }

    #[test]
    fn test_language_iris_nonempty() {
        let langs = [
            SdLanguage::Sparql10Query,
            SdLanguage::Sparql11Query,
            SdLanguage::Sparql11Update,
            SdLanguage::Sparql12Query,
            SdLanguage::Sparql12Update,
        ];
        for lang in langs {
            assert!(
                !lang.as_iri().is_empty(),
                "Language IRI empty for {:?}",
                lang
            );
        }
    }

    #[test]
    fn test_result_format_mime_nonempty() {
        let fmts = [
            SdResultFormat::SparqlResultsJson,
            SdResultFormat::SparqlResultsXml,
            SdResultFormat::SparqlResultsCsv,
            SdResultFormat::SparqlResultsTsv,
            SdResultFormat::Turtle,
            SdResultFormat::NTriples,
            SdResultFormat::NQuads,
            SdResultFormat::TriG,
            SdResultFormat::JsonLd,
            SdResultFormat::RdfXml,
            SdResultFormat::N3,
            SdResultFormat::LdPatch,
        ];
        for fmt in fmts {
            assert!(!fmt.mime_type().is_empty(), "MIME type empty for {:?}", fmt);
        }
    }

    #[test]
    fn test_input_format_mime_nonempty() {
        let fmts = [
            SdInputFormat::Turtle,
            SdInputFormat::NTriples,
            SdInputFormat::NQuads,
            SdInputFormat::TriG,
            SdInputFormat::JsonLd,
            SdInputFormat::RdfXml,
            SdInputFormat::N3,
        ];
        for fmt in fmts {
            assert!(!fmt.mime_type().is_empty(), "MIME type empty for {:?}", fmt);
        }
    }

    #[test]
    fn test_entailment_regime_iris_nonempty() {
        let regimes = [
            EntailmentRegime::Simple,
            EntailmentRegime::Rdf,
            EntailmentRegime::Rdfs,
            EntailmentRegime::Owl2Direct,
            EntailmentRegime::Owl2Rl,
            EntailmentRegime::Owl2El,
            EntailmentRegime::Owl2Ql,
            EntailmentRegime::DEntailment,
        ];
        for regime in regimes {
            assert!(
                !regime.as_iri().is_empty(),
                "Entailment regime IRI empty for {:?}",
                regime
            );
        }
    }

    #[test]
    fn test_escape_turtle_string_quotes() {
        let result = escape_turtle_string("say \"hello\"");
        assert!(result.contains("\\\""));
        assert!(!result.contains("say \"hello\""));
    }

    #[test]
    fn test_escape_turtle_string_newlines() {
        let result = escape_turtle_string("line1\nline2");
        assert!(result.contains("\\n"));
    }

    #[test]
    fn test_sd_ns_constant() {
        assert_eq!(SD_NS, "http://www.w3.org/ns/sparql-service-description#");
    }

    #[test]
    fn test_rdf_ns_constant() {
        assert_eq!(RDF_NS, "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
    }
}
