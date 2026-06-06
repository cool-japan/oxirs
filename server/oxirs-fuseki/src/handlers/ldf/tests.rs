#![cfg(test)]

use super::formats::*;
use super::pagination::*;
use super::response::*;
use super::triple_pattern::*;
use super::TpfQueryParams;

fn make_params() -> TpfQueryParams {
    TpfQueryParams {
        subject: None,
        predicate: None,
        object: None,
        page: None,
        page_size: None,
    }
}

#[test]
fn test_parse_unbound_query() {
    let q = parse_tpf_query(&make_params()).unwrap();
    assert!(q.is_unbound());
    assert_eq!(q.count_bound(), 0);
}

#[test]
fn test_parse_subject_bound() {
    let mut p = make_params();
    p.subject = Some("http://example.org/a".into());
    let q = parse_tpf_query(&p).unwrap();
    assert!(!q.is_unbound());
    assert_eq!(q.count_bound(), 1);
}

#[test]
fn test_parse_invalid_subject() {
    let mut p = make_params();
    p.subject = Some("not-an-iri".into());
    assert!(parse_tpf_query(&p).is_err());
}

#[test]
fn test_parse_invalid_predicate() {
    let mut p = make_params();
    p.predicate = Some("not-an-iri".into());
    assert!(parse_tpf_query(&p).is_err());
}

#[test]
fn test_parse_blank_node_subject() {
    let mut p = make_params();
    p.subject = Some("_:b0".into());
    let q = parse_tpf_query(&p).unwrap();
    assert_eq!(q.subject.as_deref(), Some("_:b0"));
}

#[test]
fn test_parse_empty_strings_become_unbound() {
    let mut p = make_params();
    p.subject = Some(String::new());
    p.predicate = Some(String::new());
    let q = parse_tpf_query(&p).unwrap();
    assert!(q.is_unbound());
}

#[test]
fn test_pagination_defaults() {
    let pg = PaginationParams::from_params(&make_params());
    assert_eq!(pg.page, 1);
    assert_eq!(pg.page_size, DEFAULT_PAGE_SIZE);
}

#[test]
fn test_pagination_clamping_upper() {
    let mut p = make_params();
    p.page_size = Some(100_000);
    let pg = PaginationParams::from_params(&p);
    assert_eq!(pg.page_size, MAX_PAGE_SIZE);
}

#[test]
fn test_pagination_clamping_zero() {
    let mut p = make_params();
    p.page_size = Some(0);
    let pg = PaginationParams::from_params(&p);
    assert_eq!(pg.page_size, 1);
}

#[test]
fn test_pagination_page_floor() {
    let mut p = make_params();
    p.page = Some(0);
    let pg = PaginationParams::from_params(&p);
    assert_eq!(pg.page, 1);
}

#[test]
fn test_pagination_offset() {
    let pg = PaginationParams {
        page: 3,
        page_size: 50,
    };
    assert_eq!(pg.offset(), 100);
}

#[test]
fn test_pagination_metadata() {
    let pg = PaginationParams {
        page: 2,
        page_size: 10,
    };
    let m = PaginationMetadata::new(&pg, 35);
    assert_eq!(m.total_pages, 4);
    assert!(m.has_next);
    assert!(m.has_previous);
}

#[test]
fn test_pagination_empty() {
    let pg = PaginationParams {
        page: 1,
        page_size: 10,
    };
    let m = PaginationMetadata::new(&pg, 0);
    assert_eq!(m.total_pages, 0);
    assert!(!m.has_next);
    assert!(!m.has_previous);
}

#[test]
fn test_pagination_exact_boundary() {
    let pg = PaginationParams {
        page: 1,
        page_size: 10,
    };
    let m = PaginationMetadata::new(&pg, 10);
    assert_eq!(m.total_pages, 1);
    assert!(!m.has_next);
    assert!(!m.has_previous);
}

#[test]
fn test_format_default_turtle() {
    let headers = axum::http::HeaderMap::new();
    assert_eq!(negotiate_format(&headers), LdfFormat::Turtle);
}

#[test]
fn test_format_jsonld() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("accept", "application/ld+json".parse().unwrap());
    assert_eq!(negotiate_format(&headers), LdfFormat::JsonLd);
}

#[test]
fn test_format_ntriples() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("accept", "application/n-triples".parse().unwrap());
    assert_eq!(negotiate_format(&headers), LdfFormat::NTriples);
}

#[test]
fn test_format_unknown_falls_back_to_turtle() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("accept", "text/html,application/xhtml+xml".parse().unwrap());
    assert_eq!(negotiate_format(&headers), LdfFormat::Turtle);
}

#[test]
fn test_mime_types() {
    assert_eq!(LdfFormat::Turtle.mime_type(), "text/turtle");
    assert_eq!(LdfFormat::JsonLd.mime_type(), "application/ld+json");
    assert_eq!(LdfFormat::NTriples.mime_type(), "application/n-triples");
}

#[test]
fn test_serialize_turtle_includes_metadata() {
    let pg = PaginationParams::default();
    let metadata = PaginationMetadata::new(&pg, 17);
    let response = TpfResponse {
        query: TpfQuery::default(),
        triples: vec![ResponseTriple {
            subject: "http://example.org/s".into(),
            predicate: "http://example.org/p".into(),
            object: "\"hello\"".into(),
        }],
        metadata,
        fragment_uri: "/ldf?page=1".into(),
    };
    let serialized = serialize_response(&response, LdfFormat::Turtle);
    assert!(serialized.contains("hydra:totalItems 17"));
    assert!(serialized.contains("http://example.org/s"));
}

#[test]
fn test_serialize_jsonld_payload() {
    let pg = PaginationParams::default();
    let metadata = PaginationMetadata::new(&pg, 3);
    let response = TpfResponse {
        query: TpfQuery::default(),
        triples: Vec::new(),
        metadata,
        fragment_uri: "/ldf?page=1".into(),
    };
    let serialized = serialize_response(&response, LdfFormat::JsonLd);
    assert!(serialized.contains("\"totalItems\": 3"));
    assert!(serialized.contains("hydra/context.jsonld"));
}

#[test]
fn test_serialize_ntriples_skips_metadata() {
    let pg = PaginationParams::default();
    let metadata = PaginationMetadata::new(&pg, 1);
    let response = TpfResponse {
        query: TpfQuery::default(),
        triples: vec![ResponseTriple {
            subject: "http://example.org/s".into(),
            predicate: "http://example.org/p".into(),
            object: "<http://example.org/o>".into(),
        }],
        metadata,
        fragment_uri: "/ldf?page=1".into(),
    };
    let serialized = serialize_response(&response, LdfFormat::NTriples);
    assert!(serialized.contains("http://example.org/s"));
    assert!(!serialized.contains("hydra:"));
}
