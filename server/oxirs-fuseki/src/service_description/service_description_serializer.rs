//! Serializers for `ServiceDescription`: Turtle, JSON-LD, RDF/XML, N-Triples.

use serde_json::json;

use super::service_description_types::ServiceDescription;

// ────────────────────────────────────────────────────────────────────────────
// Serializers on ServiceDescription
// ────────────────────────────────────────────────────────────────────────────

impl ServiceDescription {
    /// Generate the Service Description as Turtle RDF
    ///
    /// Returns a conforming Turtle document per the W3C SD specification.
    pub fn to_turtle(&self) -> String {
        let mut out = String::with_capacity(4096);

        // Prefixes
        out.push_str("@prefix sd: <http://www.w3.org/ns/sparql-service-description#> .\n");
        out.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
        out.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n");
        out.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n");
        out.push_str("@prefix void: <http://rdfs.org/ns/void#> .\n\n");

        // Service node
        out.push_str(&format!("<{}>\n", self.endpoint_url));
        out.push_str("    a sd:Service ;\n");

        // Label
        if let Some(ref lbl) = self.label {
            out.push_str(&format!(
                "    rdfs:label \"{}\" ;\n",
                escape_turtle_string(lbl)
            ));
        }

        // Description
        if let Some(ref desc) = self.description {
            out.push_str(&format!(
                "    rdfs:comment \"{}\" ;\n",
                escape_turtle_string(desc)
            ));
        }

        // Endpoint triple
        out.push_str(&format!("    sd:endpoint <{}> ;\n", self.endpoint_url));

        // Supported languages
        for lang in &self.supported_languages {
            out.push_str(&format!("    sd:supportedLanguage <{}> ;\n", lang.as_iri()));
        }

        // Result formats
        for fmt in &self.result_formats {
            out.push_str(&format!("    sd:resultFormat <{}> ;\n", fmt.as_iri()));
        }

        // Input formats
        for fmt in &self.input_formats {
            out.push_str(&format!("    sd:inputFormat <{}> ;\n", fmt.as_iri()));
        }

        // Features
        for feat in &self.features {
            out.push_str(&format!("    sd:feature <{}> ;\n", feat.as_iri()));
        }

        // Extension functions
        for func_iri in &self.extension_functions {
            out.push_str(&format!("    sd:extensionFunction <{}> ;\n", func_iri));
        }

        // Entailment regimes (via default entailment regime)
        for regime in &self.entailment_regimes {
            out.push_str(&format!(
                "    sd:defaultEntailmentRegime <{}> ;\n",
                regime.as_iri()
            ));
        }

        // Dataset block
        out.push_str(&format!(
            "    sd:defaultDataset [\n        a sd:Dataset ;\n        sd:defaultGraph [ a sd:Graph ] ;\n        rdfs:label \"{}\"\n    ] .\n",
            escape_turtle_string(&self.dataset_name)
        ));

        out
    }

    /// Generate the Service Description as JSON-LD
    ///
    /// Returns a `serde_json::Value` conforming to the JSON-LD representation
    /// of the W3C SD vocabulary.
    pub fn to_json_ld(&self) -> serde_json::Value {
        let supported_languages: Vec<serde_json::Value> = self
            .supported_languages
            .iter()
            .map(|l| json!({ "@id": l.as_iri() }))
            .collect();

        let result_formats: Vec<serde_json::Value> = self
            .result_formats
            .iter()
            .map(|f| json!({ "@id": f.as_iri() }))
            .collect();

        let input_formats: Vec<serde_json::Value> = self
            .input_formats
            .iter()
            .map(|f| json!({ "@id": f.as_iri() }))
            .collect();

        let features: Vec<serde_json::Value> = self
            .features
            .iter()
            .map(|f| json!({ "@id": f.as_iri() }))
            .collect();

        let extension_functions: Vec<serde_json::Value> = self
            .extension_functions
            .iter()
            .map(|iri| json!({ "@id": iri }))
            .collect();

        let entailment_regimes: Vec<serde_json::Value> = self
            .entailment_regimes
            .iter()
            .map(|r| json!({ "@id": r.as_iri() }))
            .collect();

        let mut service_obj = serde_json::Map::new();
        service_obj.insert("@id".into(), json!(self.endpoint_url));
        service_obj.insert(
            "@type".into(),
            json!("http://www.w3.org/ns/sparql-service-description#Service"),
        );
        service_obj.insert(
            "http://www.w3.org/ns/sparql-service-description#endpoint".into(),
            json!([{ "@id": self.endpoint_url }]),
        );

        if let Some(ref lbl) = self.label {
            service_obj.insert(
                "http://www.w3.org/2000/01/rdf-schema#label".into(),
                json!([{ "@value": lbl }]),
            );
        }

        if let Some(ref desc) = self.description {
            service_obj.insert(
                "http://www.w3.org/2000/01/rdf-schema#comment".into(),
                json!([{ "@value": desc }]),
            );
        }

        if !supported_languages.is_empty() {
            service_obj.insert(
                "http://www.w3.org/ns/sparql-service-description#supportedLanguage".into(),
                json!(supported_languages),
            );
        }

        if !result_formats.is_empty() {
            service_obj.insert(
                "http://www.w3.org/ns/sparql-service-description#resultFormat".into(),
                json!(result_formats),
            );
        }

        if !input_formats.is_empty() {
            service_obj.insert(
                "http://www.w3.org/ns/sparql-service-description#inputFormat".into(),
                json!(input_formats),
            );
        }

        if !features.is_empty() {
            service_obj.insert(
                "http://www.w3.org/ns/sparql-service-description#feature".into(),
                json!(features),
            );
        }

        if !extension_functions.is_empty() {
            service_obj.insert(
                "http://www.w3.org/ns/sparql-service-description#extensionFunction".into(),
                json!(extension_functions),
            );
        }

        if !entailment_regimes.is_empty() {
            service_obj.insert(
                "http://www.w3.org/ns/sparql-service-description#defaultEntailmentRegime".into(),
                json!(entailment_regimes),
            );
        }

        // Dataset description
        let dataset = json!({
            "@type": "http://www.w3.org/ns/sparql-service-description#Dataset",
            "http://www.w3.org/ns/sparql-service-description#defaultGraph": [
                {
                    "@type": "http://www.w3.org/ns/sparql-service-description#Graph"
                }
            ],
            "http://www.w3.org/2000/01/rdf-schema#label": [
                { "@value": self.dataset_name }
            ]
        });

        service_obj.insert(
            "http://www.w3.org/ns/sparql-service-description#defaultDataset".into(),
            json!([dataset]),
        );

        json!({
            "@context": {
                "sd": "http://www.w3.org/ns/sparql-service-description#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "xsd": "http://www.w3.org/2001/XMLSchema#"
            },
            "@graph": [serde_json::Value::Object(service_obj)]
        })
    }

    /// Generate Service Description as RDF/XML
    ///
    /// Returns a valid RDF/XML serialization suitable for returning
    /// from a SPARQL endpoint when `application/rdf+xml` is requested.
    pub fn to_rdf_xml(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        out.push('\n');
        out.push_str(r#"<rdf:RDF"#);
        out.push_str("\n    xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"");
        out.push_str("\n    xmlns:rdfs=\"http://www.w3.org/2000/01/rdf-schema#\"");
        out.push_str("\n    xmlns:sd=\"http://www.w3.org/ns/sparql-service-description#\">\n\n");

        out.push_str(&format!(
            "  <sd:Service rdf:about=\"{}\">\n",
            escape_xml_attr(&self.endpoint_url)
        ));

        if let Some(ref lbl) = self.label {
            out.push_str(&format!(
                "    <rdfs:label>{}</rdfs:label>\n",
                escape_xml_string(lbl)
            ));
        }
        if let Some(ref desc) = self.description {
            out.push_str(&format!(
                "    <rdfs:comment>{}</rdfs:comment>\n",
                escape_xml_string(desc)
            ));
        }

        out.push_str(&format!(
            "    <sd:endpoint rdf:resource=\"{}\"/>\n",
            escape_xml_attr(&self.endpoint_url)
        ));

        for lang in &self.supported_languages {
            out.push_str(&format!(
                "    <sd:supportedLanguage rdf:resource=\"{}\"/>\n",
                escape_xml_attr(lang.as_iri())
            ));
        }

        for fmt in &self.result_formats {
            out.push_str(&format!(
                "    <sd:resultFormat rdf:resource=\"{}\"/>\n",
                escape_xml_attr(fmt.as_iri())
            ));
        }

        for fmt in &self.input_formats {
            out.push_str(&format!(
                "    <sd:inputFormat rdf:resource=\"{}\"/>\n",
                escape_xml_attr(fmt.as_iri())
            ));
        }

        for feat in &self.features {
            out.push_str(&format!(
                "    <sd:feature rdf:resource=\"{}\"/>\n",
                escape_xml_attr(feat.as_iri())
            ));
        }

        for func_iri in &self.extension_functions {
            out.push_str(&format!(
                "    <sd:extensionFunction rdf:resource=\"{}\"/>\n",
                escape_xml_attr(func_iri)
            ));
        }

        for regime in &self.entailment_regimes {
            out.push_str(&format!(
                "    <sd:defaultEntailmentRegime rdf:resource=\"{}\"/>\n",
                escape_xml_attr(regime.as_iri())
            ));
        }

        out.push_str(&format!(
            "    <sd:defaultDataset>\n      <sd:Dataset>\n        <sd:defaultGraph><sd:Graph/></sd:defaultGraph>\n        <rdfs:label>{}</rdfs:label>\n      </sd:Dataset>\n    </sd:defaultDataset>\n",
            escape_xml_string(&self.dataset_name)
        ));

        out.push_str("  </sd:Service>\n</rdf:RDF>\n");
        out
    }
}

// ────────────────────────────────────────────────────────────────────────────
// String escaping helpers
// ────────────────────────────────────────────────────────────────────────────

/// Escape a string for use inside Turtle string literals
pub fn escape_turtle_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Escape a string for use inside XML attribute values
pub(super) fn escape_xml_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Escape a string for use inside XML element text content
pub(super) fn escape_xml_string(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
