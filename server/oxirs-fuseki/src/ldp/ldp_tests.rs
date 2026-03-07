#[cfg(test)]
mod tests {
    use super::*;

    fn service() -> LdpService {
        LdpService::new("http://example.org/ldp")
    }

    fn post_resource(svc: &mut LdpService, slug: Option<&str>) -> String {
        let slug_header = slug.map(|s| ("Slug".to_string(), s.to_string()));
        let headers: Vec<(String, String)> = [
            Some(("Content-Type".to_string(), "text/turtle".to_string())),
            slug_header,
        ]
        .into_iter()
        .flatten()
        .collect();

        let req = LdpRequest {
            method: HttpMethod::Post,
            iri: "http://example.org/ldp".to_string(),
            headers,
            body: Some(b"<> a <http://example.org/Thing> .".to_vec()),
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 201, "expected 201 Created");
        resp.header("location").unwrap().to_string()
    }

    // LdpResourceType

    #[test]
    fn resource_type_link_iri_rdf_source() {
        assert_eq!(
            LdpResourceType::RdfSource.link_type_iri(),
            "http://www.w3.org/ns/ldp#RDFSource"
        );
    }

    #[test]
    fn resource_type_link_iri_non_rdf() {
        assert_eq!(
            LdpResourceType::NonRdfSource.link_type_iri(),
            "http://www.w3.org/ns/ldp#NonRDFSource"
        );
    }

    #[test]
    fn resource_type_link_iri_basic_container() {
        assert_eq!(
            LdpResourceType::BasicContainer.link_type_iri(),
            "http://www.w3.org/ns/ldp#BasicContainer"
        );
    }

    #[test]
    fn resource_type_link_iri_direct_container() {
        let rt = LdpResourceType::DirectContainer {
            membership_resource: "http://example.org/res".to_string(),
            has_member_relation: "http://www.w3.org/ns/ldp#member".to_string(),
        };
        assert_eq!(
            rt.link_type_iri(),
            "http://www.w3.org/ns/ldp#DirectContainer"
        );
    }

    #[test]
    fn resource_type_link_iri_indirect_container() {
        let rt = LdpResourceType::IndirectContainer {
            membership_resource: "http://example.org/res".to_string(),
            has_member_relation: "http://www.w3.org/ns/ldp#member".to_string(),
            inserted_content_relation: "http://www.w3.org/ns/ldp#MemberSubject".to_string(),
        };
        assert_eq!(
            rt.link_type_iri(),
            "http://www.w3.org/ns/ldp#IndirectContainer"
        );
    }

    #[test]
    fn resource_type_is_container_true_for_containers() {
        assert!(LdpResourceType::BasicContainer.is_container());
        assert!(LdpResourceType::Container.is_container());
        assert!(LdpResourceType::DirectContainer {
            membership_resource: "x".to_string(),
            has_member_relation: "y".to_string(),
        }
        .is_container());
        assert!(LdpResourceType::IndirectContainer {
            membership_resource: "x".to_string(),
            has_member_relation: "y".to_string(),
            inserted_content_relation: "z".to_string(),
        }
        .is_container());
    }

    #[test]
    fn resource_type_is_container_false_for_non_containers() {
        assert!(!LdpResourceType::RdfSource.is_container());
        assert!(!LdpResourceType::NonRdfSource.is_container());
    }

    // HttpMethod

    #[test]
    fn http_method_parse_all() {
        assert_eq!(HttpMethod::parse("GET"), Some(HttpMethod::Get));
        assert_eq!(HttpMethod::parse("get"), Some(HttpMethod::Get));
        assert_eq!(HttpMethod::parse("HEAD"), Some(HttpMethod::Head));
        assert_eq!(HttpMethod::parse("POST"), Some(HttpMethod::Post));
        assert_eq!(HttpMethod::parse("PUT"), Some(HttpMethod::Put));
        assert_eq!(HttpMethod::parse("DELETE"), Some(HttpMethod::Delete));
        assert_eq!(HttpMethod::parse("PATCH"), Some(HttpMethod::Patch));
        assert_eq!(HttpMethod::parse("OPTIONS"), Some(HttpMethod::Options));
        assert_eq!(HttpMethod::parse("UNKNOWN"), None);
    }

    #[test]
    fn http_method_as_str() {
        assert_eq!(HttpMethod::Get.as_str(), "GET");
        assert_eq!(HttpMethod::Post.as_str(), "POST");
        assert_eq!(HttpMethod::Delete.as_str(), "DELETE");
    }

    // PreferHeader

    #[test]
    fn prefer_header_parse_return_representation() {
        let prefs = PreferHeader::parse("return=representation");
        assert!(prefs.contains(&PreferHeader::ReturnRepresentation));
    }

    #[test]
    fn prefer_header_parse_return_minimal() {
        let prefs = PreferHeader::parse("return=minimal");
        assert!(prefs.contains(&PreferHeader::ReturnMinimal));
    }

    #[test]
    fn prefer_header_parse_include_containment() {
        let prefs = PreferHeader::parse(
            "return=representation; include=\"http://www.w3.org/ns/ldp#PreferContainment\"",
        );
        assert!(prefs.contains(&PreferHeader::IncludeContainment));
        assert!(prefs.contains(&PreferHeader::ReturnRepresentation));
    }

    #[test]
    fn prefer_header_parse_omit_containment() {
        let prefs = PreferHeader::parse(
            "return=representation; omit=\"http://www.w3.org/ns/ldp#PreferContainment\"",
        );
        assert!(prefs.contains(&PreferHeader::OmitContainment));
    }

    #[test]
    fn prefer_header_parse_omit_membership() {
        let prefs = PreferHeader::parse(
            "return=representation; omit=\"http://www.w3.org/ns/ldp#PreferMembership\"",
        );
        assert!(prefs.contains(&PreferHeader::OmitMembership));
    }

    #[test]
    fn prefer_header_parse_include_minimal_container() {
        let prefs = PreferHeader::parse(
            "return=representation; include=\"http://www.w3.org/ns/ldp#PreferMinimalContainer\"",
        );
        assert!(prefs.contains(&PreferHeader::IncludeMinimalContainer));
    }

    #[test]
    fn prefer_header_parse_empty() {
        let prefs = PreferHeader::parse("");
        assert!(prefs.is_empty());
    }

    // LdpRequest

    #[test]
    fn ldp_request_get_helper() {
        let req = LdpRequest::get("http://example.org/");
        assert_eq!(req.method, HttpMethod::Get);
        assert_eq!(req.iri, "http://example.org/");
        assert!(req.body.is_none());
    }

    #[test]
    fn ldp_request_header_lookup_case_insensitive() {
        let req = LdpRequest {
            method: HttpMethod::Get,
            iri: "http://example.org/".to_string(),
            headers: vec![("Content-Type".to_string(), "text/turtle".to_string())],
            body: None,
        };
        assert_eq!(req.header("content-type"), Some("text/turtle"));
        assert_eq!(req.header("CONTENT-TYPE"), Some("text/turtle"));
        assert_eq!(req.header("x-missing"), None);
    }

    #[test]
    fn ldp_request_slug_and_content_type() {
        let req = LdpRequest {
            method: HttpMethod::Post,
            iri: "http://example.org/ldp".to_string(),
            headers: vec![
                ("Content-Type".to_string(), "text/turtle".to_string()),
                ("Slug".to_string(), "my-resource".to_string()),
            ],
            body: Some(b"<> a <http://example.org/T> .".to_vec()),
        };
        assert_eq!(req.slug(), Some("my-resource"));
        assert_eq!(req.content_type(), Some("text/turtle"));
        assert_eq!(req.if_match(), None);
    }

    // LdpResponse

    #[test]
    fn ldp_response_ok_status_and_body() {
        let resp = LdpResponse::ok("hello");
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body.as_deref(), Some(b"hello".as_ref()));
    }

    #[test]
    fn ldp_response_created_has_location() {
        let resp = LdpResponse::created("http://example.org/resource/1");
        assert_eq!(resp.status, 201);
        assert_eq!(
            resp.header("location"),
            Some("http://example.org/resource/1")
        );
    }

    #[test]
    fn ldp_response_no_content() {
        let resp = LdpResponse::no_content();
        assert_eq!(resp.status, 204);
        assert!(resp.body.is_none());
    }

    #[test]
    fn ldp_response_not_found() {
        let resp = LdpResponse::not_found();
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn ldp_response_method_not_allowed_with_allow_header() {
        let resp = LdpResponse::method_not_allowed(&["GET", "HEAD"]);
        assert_eq!(resp.status, 405);
        let allow = resp.header("allow").unwrap();
        assert!(allow.contains("GET"));
        assert!(allow.contains("HEAD"));
    }

    #[test]
    fn ldp_response_precondition_failed() {
        let resp = LdpResponse::precondition_failed();
        assert_eq!(resp.status, 412);
    }

    #[test]
    fn ldp_response_conflict_contains_reason() {
        let resp = LdpResponse::conflict("container is not empty");
        assert_eq!(resp.status, 409);
        let body = std::str::from_utf8(resp.body.as_deref().unwrap_or_default()).unwrap();
        assert!(body.contains("container is not empty"));
    }

    #[test]
    fn ldp_response_unsupported_media_type() {
        let resp = LdpResponse::unsupported_media_type();
        assert_eq!(resp.status, 415);
    }

    #[test]
    fn ldp_response_with_etag() {
        let resp = LdpResponse::ok("body").with_etag("abc123");
        let etag = resp.header("etag").unwrap();
        assert!(etag.contains("abc123"));
    }

    #[test]
    fn ldp_response_with_allow() {
        let resp = LdpResponse::ok("body").with_allow(&["GET", "POST"]);
        let allow = resp.header("allow").unwrap();
        assert!(allow.contains("GET"));
        assert!(allow.contains("POST"));
    }

    #[test]
    fn ldp_response_with_accept_post() {
        let resp =
            LdpResponse::ok("body").with_accept_post(&["text/turtle", "application/ld+json"]);
        let ap = resp.header("accept-post").unwrap();
        assert!(ap.contains("text/turtle"));
    }

    #[test]
    fn ldp_response_with_ldp_type_basic_container() {
        let resp = LdpResponse::ok("body").with_ldp_type(&LdpResourceType::BasicContainer);
        let links = resp.all_headers("link");
        assert!(
            links.iter().any(|l| l.contains("ldp#BasicContainer")),
            "expected BasicContainer link"
        );
        assert!(
            links.iter().any(|l| l.contains("ldp#Resource")),
            "expected Resource link"
        );
    }

    #[test]
    fn ldp_response_with_ldp_type_rdf_source() {
        let resp = LdpResponse::ok("body").with_ldp_type(&LdpResourceType::RdfSource);
        let links = resp.all_headers("link");
        assert!(links.iter().any(|l| l.contains("ldp#RDFSource")));
    }

    #[test]
    fn ldp_response_with_content_type() {
        let resp = LdpResponse::ok("body").with_content_type("text/turtle");
        assert_eq!(resp.header("content-type"), Some("text/turtle"));
    }

    #[test]
    fn ldp_response_gone_status() {
        let resp = LdpResponse::gone();
        assert_eq!(resp.status, 410);
    }

    // LdpMember

    #[test]
    fn ldp_member_new_sets_iri_and_etag() {
        let m = LdpMember::new("http://example.org/res/1", LdpResourceType::RdfSource);
        assert_eq!(m.iri, "http://example.org/res/1");
        assert!(!m.etag.is_empty());
    }

    // LdpContainer

    #[test]
    fn ldp_container_new_basic() {
        let c = LdpContainer::new_basic("http://example.org/ldp");
        assert_eq!(c.iri, "http://example.org/ldp");
        assert_eq!(c.resource_type, LdpResourceType::BasicContainer);
        assert!(c.members.is_empty());
    }

    #[test]
    fn ldp_container_new_direct() {
        let c = LdpContainer::new_direct(
            "http://example.org/ldp/d",
            "http://example.org/set",
            "http://www.w3.org/ns/ldp#member",
        );
        assert!(matches!(
            c.resource_type,
            LdpResourceType::DirectContainer { .. }
        ));
    }

    #[test]
    fn ldp_container_new_indirect() {
        let c = LdpContainer::new_indirect(
            "http://example.org/ldp/i",
            "http://example.org/set",
            "http://www.w3.org/ns/ldp#member",
            "http://www.w3.org/ns/ldp#MemberSubject",
        );
        assert!(matches!(
            c.resource_type,
            LdpResourceType::IndirectContainer { .. }
        ));
    }

    #[test]
    fn ldp_container_add_member_success() {
        let mut c = LdpContainer::new_basic("http://example.org/c");
        let m = LdpMember::new("http://example.org/c/1", LdpResourceType::RdfSource);
        assert!(c.add_member(m).is_ok());
        assert_eq!(c.members.len(), 1);
    }

    #[test]
    fn ldp_container_add_member_duplicate_returns_error() {
        let mut c = LdpContainer::new_basic("http://example.org/c");
        let m1 = LdpMember::new("http://example.org/c/1", LdpResourceType::RdfSource);
        let m2 = LdpMember::new("http://example.org/c/1", LdpResourceType::RdfSource);
        assert!(c.add_member(m1).is_ok());
        assert!(c.add_member(m2).is_err());
    }

    #[test]
    fn ldp_container_remove_member_success() {
        let mut c = LdpContainer::new_basic("http://example.org/c");
        let m = LdpMember::new("http://example.org/c/1", LdpResourceType::RdfSource);
        c.add_member(m).unwrap();
        assert!(c.remove_member("http://example.org/c/1").is_ok());
        assert!(c.members.is_empty());
    }

    #[test]
    fn ldp_container_remove_member_not_found() {
        let mut c = LdpContainer::new_basic("http://example.org/c");
        assert!(matches!(
            c.remove_member("http://example.org/c/x"),
            Err(LdpError::NotFound(_))
        ));
    }

    #[test]
    fn ldp_container_get_member() {
        let mut c = LdpContainer::new_basic("http://example.org/c");
        let m = LdpMember::new("http://example.org/c/1", LdpResourceType::RdfSource);
        c.add_member(m).unwrap();
        assert!(c.get_member("http://example.org/c/1").is_some());
        assert!(c.get_member("http://example.org/c/none").is_none());
    }

    #[test]
    fn ldp_container_generate_iri_with_slug() {
        let mut c = LdpContainer::new_basic("http://example.org/c");
        let iri = c.generate_iri(Some("my-resource"));
        assert_eq!(iri, "http://example.org/c/my-resource");
    }

    #[test]
    fn ldp_container_generate_iri_without_slug() {
        let mut c = LdpContainer::new_basic("http://example.org/c");
        let iri = c.generate_iri(None);
        assert!(iri.starts_with("http://example.org/c/"));
    }

    #[test]
    fn ldp_container_generate_iri_slug_collision_appends_counter() {
        let mut c = LdpContainer::new_basic("http://example.org/c");
        let m = LdpMember::new("http://example.org/c/item", LdpResourceType::RdfSource);
        c.add_member(m).unwrap();
        let iri = c.generate_iri(Some("item"));
        assert_ne!(iri, "http://example.org/c/item");
    }

    #[test]
    fn ldp_container_to_turtle_contains_type() {
        let c = LdpContainer::new_basic("http://example.org/c");
        let ttl = c.to_turtle(&[]);
        assert!(ttl.contains("ldp:BasicContainer"));
        assert!(ttl.contains("http://example.org/c"));
    }

    #[test]
    fn ldp_container_to_turtle_containment_triples() {
        let mut c = LdpContainer::new_basic("http://example.org/c");
        let m = LdpMember::new("http://example.org/c/1", LdpResourceType::RdfSource);
        c.add_member(m).unwrap();
        let ttl = c.to_turtle(&[]);
        assert!(ttl.contains("ldp:contains"), "should include containment");
        assert!(ttl.contains("http://example.org/c/1"));
    }

    #[test]
    fn ldp_container_to_turtle_omit_containment() {
        let mut c = LdpContainer::new_basic("http://example.org/c");
        let m = LdpMember::new("http://example.org/c/1", LdpResourceType::RdfSource);
        c.add_member(m).unwrap();
        let ttl = c.to_turtle(&[PreferHeader::OmitContainment]);
        assert!(
            !ttl.contains("ldp:contains"),
            "containment should be omitted"
        );
    }

    #[test]
    fn ldp_container_to_turtle_minimal_container() {
        let mut c = LdpContainer::new_basic("http://example.org/c");
        let m = LdpMember::new("http://example.org/c/1", LdpResourceType::RdfSource);
        c.add_member(m).unwrap();
        let ttl = c.to_turtle(&[PreferHeader::IncludeMinimalContainer]);
        assert!(!ttl.contains("ldp:contains"));
    }

    #[test]
    fn ldp_container_to_jsonld_contains_members() {
        let mut c = LdpContainer::new_basic("http://example.org/c");
        let m = LdpMember::new("http://example.org/c/1", LdpResourceType::RdfSource);
        c.add_member(m).unwrap();
        let jsonld = c.to_jsonld();
        assert!(jsonld.contains("ldp:contains"));
        assert!(jsonld.contains("http://example.org/c/1"));
        assert!(jsonld.contains("ldp:BasicContainer"));
    }

    #[test]
    fn ldp_container_to_jsonld_empty() {
        let c = LdpContainer::new_basic("http://example.org/c");
        let jsonld = c.to_jsonld();
        assert!(jsonld.contains("ldp:contains"));
        assert!(jsonld.contains("[]"));
    }

    #[test]
    fn ldp_container_etag_changes_with_members() {
        let mut c = LdpContainer::new_basic("http://example.org/c");
        let etag_before = c.etag();
        let m = LdpMember::new("http://example.org/c/1", LdpResourceType::RdfSource);
        c.add_member(m).unwrap();
        let etag_after = c.etag();
        assert_ne!(etag_before, etag_after);
    }

    #[test]
    fn ldp_container_direct_to_turtle_includes_membership_props() {
        let c = LdpContainer::new_direct(
            "http://example.org/ldp/d",
            "http://example.org/set",
            "http://www.w3.org/ns/ldp#member",
        );
        let ttl = c.to_turtle(&[]);
        assert!(ttl.contains("ldp:membershipResource"));
        assert!(ttl.contains("ldp:hasMemberRelation"));
    }

    #[test]
    fn ldp_container_direct_to_turtle_omit_membership() {
        let c = LdpContainer::new_direct(
            "http://example.org/ldp/d",
            "http://example.org/set",
            "http://www.w3.org/ns/ldp#member",
        );
        let ttl = c.to_turtle(&[PreferHeader::OmitMembership]);
        assert!(!ttl.contains("ldp:membershipResource"));
    }

    // LdpService -- GET

    #[test]
    fn service_get_root_container_200() {
        let mut svc = service();
        let req = LdpRequest::get("http://example.org/ldp");
        let resp = svc.handle(req);
        assert_eq!(resp.status, 200);
    }

    #[test]
    fn service_get_root_container_has_link_header() {
        let mut svc = service();
        let req = LdpRequest::get("http://example.org/ldp");
        let resp = svc.handle(req);
        let links = resp.all_headers("link");
        assert!(links.iter().any(|l| l.contains("ldp#BasicContainer")));
    }

    #[test]
    fn service_get_root_container_content_type_turtle() {
        let mut svc = service();
        let req = LdpRequest::get("http://example.org/ldp");
        let resp = svc.handle(req);
        let ct = resp.header("content-type").unwrap_or("");
        assert!(ct.contains("text/turtle"));
    }

    #[test]
    fn service_get_root_has_etag() {
        let mut svc = service();
        let req = LdpRequest::get("http://example.org/ldp");
        let resp = svc.handle(req);
        assert!(resp.header("etag").is_some());
    }

    #[test]
    fn service_get_non_existent_404() {
        let mut svc = service();
        let req = LdpRequest::get("http://example.org/ldp/missing");
        let resp = svc.handle(req);
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn service_get_contains_containment_triples_after_post() {
        let mut svc = service();
        post_resource(&mut svc, None);
        let req = LdpRequest::get("http://example.org/ldp");
        let resp = svc.handle(req);
        let body = std::str::from_utf8(resp.body.as_deref().unwrap_or_default()).unwrap();
        assert!(body.contains("ldp:contains"));
    }

    #[test]
    fn service_get_prefer_omit_containment() {
        let mut svc = service();
        post_resource(&mut svc, None);
        let req = LdpRequest {
            method: HttpMethod::Get,
            iri: "http://example.org/ldp".to_string(),
            headers: vec![(
                "Prefer".to_string(),
                "return=representation; omit=\"http://www.w3.org/ns/ldp#PreferContainment\""
                    .to_string(),
            )],
            body: None,
        };
        let resp = svc.handle(req);
        let body = std::str::from_utf8(resp.body.as_deref().unwrap_or_default()).unwrap();
        assert!(
            !body.contains("ldp:contains"),
            "containment must be omitted"
        );
    }

    // LdpService -- HEAD

    #[test]
    fn service_head_existing_resource_200_no_body() {
        let mut svc = service();
        let iri = post_resource(&mut svc, None);
        let req = LdpRequest {
            method: HttpMethod::Head,
            iri,
            headers: Vec::new(),
            body: None,
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 200);
        assert!(resp.body.is_none());
    }

    #[test]
    fn service_head_non_existent_404() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Head,
            iri: "http://example.org/ldp/none".to_string(),
            headers: Vec::new(),
            body: None,
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 404);
    }

    // LdpService -- POST

    #[test]
    fn service_post_creates_resource_201() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Post,
            iri: "http://example.org/ldp".to_string(),
            headers: vec![("Content-Type".to_string(), "text/turtle".to_string())],
            body: Some(b"<> a <http://example.org/T> .".to_vec()),
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 201);
    }

    #[test]
    fn service_post_returns_location_header() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Post,
            iri: "http://example.org/ldp".to_string(),
            headers: vec![("Content-Type".to_string(), "text/turtle".to_string())],
            body: Some(b"<> a <http://example.org/T> .".to_vec()),
        };
        let resp = svc.handle(req);
        assert!(resp.header("location").is_some());
    }

    #[test]
    fn service_post_with_slug_uses_slug_in_iri() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Post,
            iri: "http://example.org/ldp".to_string(),
            headers: vec![
                ("Content-Type".to_string(), "text/turtle".to_string()),
                ("Slug".to_string(), "my-thing".to_string()),
            ],
            body: Some(b"<> a <http://example.org/T> .".to_vec()),
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 201);
        let loc = resp.header("location").unwrap();
        assert!(loc.contains("my-thing"), "IRI should contain slug");
    }

    #[test]
    fn service_post_without_slug_auto_generates_iri() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Post,
            iri: "http://example.org/ldp".to_string(),
            headers: vec![("Content-Type".to_string(), "text/turtle".to_string())],
            body: Some(b"<> a <http://example.org/T> .".to_vec()),
        };
        let resp = svc.handle(req);
        let loc = resp.header("location").unwrap();
        assert!(
            loc.starts_with("http://example.org/ldp/"),
            "IRI should be under container"
        );
    }

    #[test]
    fn service_post_to_non_existent_container_404() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Post,
            iri: "http://example.org/ldp/missing-container".to_string(),
            headers: vec![("Content-Type".to_string(), "text/turtle".to_string())],
            body: Some(b"<> a <http://example.org/T> .".to_vec()),
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn service_post_unsupported_content_type_415() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Post,
            iri: "http://example.org/ldp".to_string(),
            headers: vec![("Content-Type".to_string(), "text/plain".to_string())],
            body: Some(b"hello".to_vec()),
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 415);
    }

    #[test]
    fn service_post_jsonld_accepted() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Post,
            iri: "http://example.org/ldp".to_string(),
            headers: vec![(
                "Content-Type".to_string(),
                "application/ld+json".to_string(),
            )],
            body: Some(b"{\"@id\": \"http://example.org/r\"}".to_vec()),
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 201);
    }

    #[test]
    fn service_post_resource_accessible_via_get() {
        let mut svc = service();
        let iri = post_resource(&mut svc, Some("my-res"));
        let resp = svc.handle(LdpRequest::get(&iri));
        assert_eq!(resp.status, 200);
    }

    #[test]
    fn service_post_multiple_resources_unique_iris() {
        let mut svc = service();
        let iri1 = post_resource(&mut svc, None);
        let iri2 = post_resource(&mut svc, None);
        assert_ne!(iri1, iri2);
    }

    // LdpService -- PUT

    #[test]
    fn service_put_new_resource_201() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Put,
            iri: "http://example.org/ldp/new-res".to_string(),
            headers: vec![("Content-Type".to_string(), "text/turtle".to_string())],
            body: Some(b"<> a <http://example.org/T> .".to_vec()),
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 201);
    }

    #[test]
    fn service_put_new_resource_accessible_via_get() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Put,
            iri: "http://example.org/ldp/new-res".to_string(),
            headers: vec![("Content-Type".to_string(), "text/turtle".to_string())],
            body: Some(b"<> a <http://example.org/T> .".to_vec()),
        };
        svc.handle(req);
        let resp = svc.handle(LdpRequest::get("http://example.org/ldp/new-res"));
        assert_eq!(resp.status, 200);
    }

    #[test]
    fn service_put_update_with_matching_etag_204() {
        let mut svc = service();
        let iri = post_resource(&mut svc, Some("upd"));
        let get_resp = svc.handle(LdpRequest::get(&iri));
        let etag_raw = get_resp.header("etag").unwrap().to_string();
        let etag = etag_raw.trim_matches('"').to_string();
        let put_req = LdpRequest {
            method: HttpMethod::Put,
            iri: iri.clone(),
            headers: vec![
                ("Content-Type".to_string(), "text/turtle".to_string()),
                ("If-Match".to_string(), etag.clone()),
            ],
            body: Some(b"<> a <http://example.org/Updated> .".to_vec()),
        };
        let resp = svc.handle(put_req);
        assert_eq!(resp.status, 204);
    }

    #[test]
    fn service_put_update_with_wrong_etag_412() {
        let mut svc = service();
        let iri = post_resource(&mut svc, Some("etag-test"));
        let put_req = LdpRequest {
            method: HttpMethod::Put,
            iri: iri.clone(),
            headers: vec![
                ("Content-Type".to_string(), "text/turtle".to_string()),
                ("If-Match".to_string(), "wrong-etag".to_string()),
            ],
            body: Some(b"<> a <http://example.org/Updated> .".to_vec()),
        };
        let resp = svc.handle(put_req);
        assert_eq!(resp.status, 412);
    }

    #[test]
    fn service_put_with_if_match_non_existent_412() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Put,
            iri: "http://example.org/ldp/nope".to_string(),
            headers: vec![
                ("Content-Type".to_string(), "text/turtle".to_string()),
                ("If-Match".to_string(), "some-etag".to_string()),
            ],
            body: Some(b"body".to_vec()),
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 412);
    }

    // LdpService -- DELETE

    #[test]
    fn service_delete_resource_204() {
        let mut svc = service();
        let iri = post_resource(&mut svc, Some("to-delete"));
        let req = LdpRequest {
            method: HttpMethod::Delete,
            iri: iri.clone(),
            headers: Vec::new(),
            body: None,
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 204);
    }

    #[test]
    fn service_delete_resource_then_get_410_or_404() {
        let mut svc = service();
        let iri = post_resource(&mut svc, Some("gone-res"));
        let del = LdpRequest {
            method: HttpMethod::Delete,
            iri: iri.clone(),
            headers: Vec::new(),
            body: None,
        };
        svc.handle(del);
        let get = svc.handle(LdpRequest::get(&iri));
        assert!(
            get.status == 404 || get.status == 410,
            "expected 404 or 410, got {}",
            get.status
        );
    }

    #[test]
    fn service_delete_non_existent_404() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Delete,
            iri: "http://example.org/ldp/not-here".to_string(),
            headers: Vec::new(),
            body: None,
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn service_delete_removes_from_container_members() {
        let mut svc = service();
        let iri = post_resource(&mut svc, None);
        let before = svc.handle(LdpRequest::get("http://example.org/ldp"));
        let body_before = std::str::from_utf8(before.body.as_deref().unwrap_or_default()).unwrap();
        assert!(body_before.contains(&iri));
        let del = LdpRequest {
            method: HttpMethod::Delete,
            iri: iri.clone(),
            headers: Vec::new(),
            body: None,
        };
        svc.handle(del);
        let after = svc.handle(LdpRequest::get("http://example.org/ldp"));
        let body_after = std::str::from_utf8(after.body.as_deref().unwrap_or_default()).unwrap();
        assert!(
            !body_after.contains(&iri),
            "deleted resource should not appear in container"
        );
    }

    #[test]
    fn service_delete_non_empty_container_409() {
        let mut svc = service();
        let sub = LdpContainer::new_basic("http://example.org/ldp/sub");
        svc.register_container(sub);
        let inner = LdpContainer::new_basic("http://example.org/ldp/sub/inner");
        svc.register_container(inner.clone());
        let member = LdpMember::new(inner.iri.clone(), LdpResourceType::BasicContainer);
        svc.containers
            .get_mut("http://example.org/ldp/sub")
            .unwrap()
            .add_member(member)
            .unwrap();

        let req = LdpRequest {
            method: HttpMethod::Delete,
            iri: "http://example.org/ldp/sub".to_string(),
            headers: Vec::new(),
            body: None,
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 409);
    }

    // LdpService -- OPTIONS

    #[test]
    fn service_options_container_returns_allow_header() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Options,
            iri: "http://example.org/ldp".to_string(),
            headers: Vec::new(),
            body: None,
        };
        let resp = svc.handle(req);
        let allow = resp.header("allow").unwrap_or("");
        assert!(allow.contains("GET"));
        assert!(allow.contains("POST"));
    }

    #[test]
    fn service_options_container_has_accept_post() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Options,
            iri: "http://example.org/ldp".to_string(),
            headers: Vec::new(),
            body: None,
        };
        let resp = svc.handle(req);
        assert!(resp.header("accept-post").is_some());
    }

    #[test]
    fn service_options_resource_returns_allow_without_post() {
        let mut svc = service();
        let iri = post_resource(&mut svc, Some("opts-res"));
        let req = LdpRequest {
            method: HttpMethod::Options,
            iri: iri.clone(),
            headers: Vec::new(),
            body: None,
        };
        let resp = svc.handle(req);
        let allow = resp.header("allow").unwrap_or("");
        assert!(allow.contains("GET"));
        assert!(!allow.contains("POST"));
    }

    #[test]
    fn service_options_status_204() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Options,
            iri: "http://example.org/ldp".to_string(),
            headers: Vec::new(),
            body: None,
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 204);
    }

    // LdpService -- PATCH

    #[test]
    fn service_patch_existing_resource_204() {
        let mut svc = service();
        let iri = post_resource(&mut svc, Some("patch-me"));
        let req = LdpRequest {
            method: HttpMethod::Patch,
            iri: iri.clone(),
            headers: vec![(
                "Content-Type".to_string(),
                "application/sparql-update".to_string(),
            )],
            body: Some(
                b"INSERT DATA { <http://example.org/s> <http://example.org/p> \"v\" . }".to_vec(),
            ),
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 204);
    }

    #[test]
    fn service_patch_non_existent_404() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Patch,
            iri: "http://example.org/ldp/no-such".to_string(),
            headers: vec![(
                "Content-Type".to_string(),
                "application/sparql-update".to_string(),
            )],
            body: Some(b"DELETE WHERE { ?s ?p ?o . }".to_vec()),
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 404);
    }

    // LdpError

    #[test]
    fn ldp_error_display_not_found() {
        let e = LdpError::NotFound("x".to_string());
        assert!(e.to_string().contains("Not Found"));
    }

    #[test]
    fn ldp_error_display_conflict() {
        let e = LdpError::Conflict("dup".to_string());
        assert!(e.to_string().contains("Conflict"));
    }

    #[test]
    fn ldp_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(LdpError::Internal("oops".to_string()));
        assert!(e.to_string().contains("Internal"));
    }

    // LdpResource

    #[test]
    fn ldp_resource_new_rdf_source() {
        let r = LdpResource::new_rdf_source("http://example.org/r1", b"body".to_vec());
        assert_eq!(r.content_type, "text/turtle");
        assert_eq!(r.resource_type, LdpResourceType::RdfSource);
        assert!(!r.etag.is_empty());
    }

    #[test]
    fn ldp_resource_new_non_rdf() {
        let r = LdpResource::new_non_rdf("http://example.org/img", "image/png", vec![0u8; 100]);
        assert_eq!(r.content_type, "image/png");
        assert_eq!(r.resource_type, LdpResourceType::NonRdfSource);
    }

    #[test]
    fn ldp_resource_update_body_changes_etag() {
        let mut r = LdpResource::new_rdf_source("http://example.org/r", b"old".to_vec());
        let old_etag = r.etag.clone();
        r.update_body(b"new body".to_vec(), "text/turtle".to_string());
        assert_ne!(r.etag, old_etag);
    }

    // LdpService -- register_container

    #[test]
    fn service_register_container_and_get() {
        let mut svc = service();
        let sub = LdpContainer::new_basic("http://example.org/ldp/sub");
        svc.register_container(sub);
        let resp = svc.handle(LdpRequest::get("http://example.org/ldp/sub"));
        assert_eq!(resp.status, 200);
    }

    #[test]
    fn service_register_direct_container_turtle_has_membership() {
        let mut svc = service();
        let dc = LdpContainer::new_direct(
            "http://example.org/ldp/dc",
            "http://example.org/set",
            "http://www.w3.org/ns/ldp#member",
        );
        svc.register_container(dc);
        let resp = svc.handle(LdpRequest::get("http://example.org/ldp/dc"));
        let body = std::str::from_utf8(resp.body.as_deref().unwrap_or_default()).unwrap();
        assert!(body.contains("ldp:membershipResource"));
    }

    // compute_etag_str

    #[test]
    fn compute_etag_str_deterministic() {
        let e1 = compute_etag_str("test");
        let e2 = compute_etag_str("test");
        assert_eq!(e1, e2);
    }

    #[test]
    fn compute_etag_str_different_inputs_different_outputs() {
        let e1 = compute_etag_str("input_a");
        let e2 = compute_etag_str("input_b");
        assert_ne!(e1, e2);
    }

    // LdpService -- base_url

    #[test]
    fn service_base_url_is_set() {
        let svc = LdpService::new("http://example.org/ldp");
        assert_eq!(svc.base_url, "http://example.org/ldp");
    }

    #[test]
    fn service_root_container_exists_at_base_url() {
        let svc = LdpService::new("http://example.org/ldp");
        assert!(svc.containers.contains_key("http://example.org/ldp"));
    }

    // LdpService -- content-type variants

    #[test]
    fn service_post_n_triples_accepted() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Post,
            iri: "http://example.org/ldp".to_string(),
            headers: vec![(
                "Content-Type".to_string(),
                "application/n-triples".to_string(),
            )],
            body: Some(b"<http://a> <http://b> <http://c> .".to_vec()),
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 201);
    }

    #[test]
    fn service_post_rdf_xml_accepted() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Post,
            iri: "http://example.org/ldp".to_string(),
            headers: vec![(
                "Content-Type".to_string(),
                "application/rdf+xml".to_string(),
            )],
            body: Some(b"<rdf:RDF/>".to_vec()),
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 201);
    }

    #[test]
    fn service_post_binary_as_non_rdf() {
        let mut svc = service();
        let req = LdpRequest {
            method: HttpMethod::Post,
            iri: "http://example.org/ldp".to_string(),
            headers: vec![(
                "Content-Type".to_string(),
                "application/octet-stream".to_string(),
            )],
            body: Some(vec![0u8, 1, 2, 3]),
        };
        let resp = svc.handle(req);
        assert_eq!(resp.status, 201);
    }

    // LdpOperation variants

    #[test]
    fn ldp_operation_get_resource_fields() {
        let op = LdpOperation::GetResource {
            iri: "http://example.org/r".to_string(),
            prefer: vec![PreferHeader::ReturnRepresentation],
        };
        if let LdpOperation::GetResource { iri, prefer } = op {
            assert_eq!(iri, "http://example.org/r");
            assert_eq!(prefer.len(), 1);
        }
    }

    #[test]
    fn ldp_operation_delete_resource_fields() {
        let op = LdpOperation::DeleteResource {
            iri: "http://example.org/r".to_string(),
        };
        if let LdpOperation::DeleteResource { iri } = op {
            assert_eq!(iri, "http://example.org/r");
        }
    }
}
