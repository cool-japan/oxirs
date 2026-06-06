//! Vocabulary metadata serializers.
//!
//! Supported response formats are HTML (browser-friendly default), JSON-LD
//! (machine-friendly, default for linked-data clients) and Turtle (compact
//! semantic-web format). The serializers operate purely on
//! [`VocabularyMetadata`] and have no external dependencies, avoiding extra
//! workspace crates.

use super::metadata::VocabularyMetadata;

/// Negotiable response formats for the vocabulary publishing endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabFormat {
    Html,
    JsonLd,
    Turtle,
}

impl VocabFormat {
    /// IANA media type for the response format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            VocabFormat::Html => "text/html",
            VocabFormat::JsonLd => "application/ld+json",
            VocabFormat::Turtle => "text/turtle",
        }
    }
}

/// Serialize the metadata to the requested format.
pub fn serialize_metadata(metadata: &VocabularyMetadata, format: VocabFormat) -> String {
    match format {
        VocabFormat::Html => serialize_html(metadata),
        VocabFormat::JsonLd => serialize_jsonld(metadata),
        VocabFormat::Turtle => serialize_turtle(metadata),
    }
}

fn serialize_html(m: &VocabularyMetadata) -> String {
    format!(
        r#"<!DOCTYPE html><html><head><title>{title}</title></head><body><h1>{title}</h1><p>{desc}</p><dl><dt>Namespace</dt><dd>{ns}</dd><dt>Concepts</dt><dd>{count}</dd></dl></body></html>"#,
        title = html_escape(&m.title),
        desc = html_escape(&m.description),
        ns = html_escape(&m.namespace),
        count = m.concept_count
    )
}

fn serialize_jsonld(m: &VocabularyMetadata) -> String {
    format!(
        r#"{{"@context":"http://www.w3.org/ns/dcat","@id":"{}","dcterms:title":"{}","dcterms:description":"{}","void:vocabulary":"{}","void:entities":{}}}"#,
        m.id, m.title, m.description, m.namespace, m.concept_count
    )
}

fn serialize_turtle(m: &VocabularyMetadata) -> String {
    let mut out = String::new();
    out.push_str("@prefix dcterms: <http://purl.org/dc/terms/> .\n");
    out.push_str("@prefix void: <http://rdfs.org/ns/void#> .\n");
    out.push_str("@prefix dcat: <http://www.w3.org/ns/dcat#> .\n\n");
    out.push_str(&format!("<{}> a dcat:Dataset ;\n", m.id));
    out.push_str(&format!(
        "  dcterms:title \"{}\" ;\n",
        m.title.replace('"', "\\\"")
    ));
    out.push_str(&format!(
        "  dcterms:description \"{}\" ;\n",
        m.description.replace('"', "\\\"")
    ));
    out.push_str(&format!("  void:vocabulary <{}> ;\n", m.namespace));
    out.push_str(&format!("  void:entities {} .\n", m.concept_count));
    out
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
