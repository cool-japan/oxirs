//! # Linked Data Platform (LDP) W3C Recommendation Implementation
//!
//! Provides RESTful HTTP access to RDF resources as per:
//! <https://www.w3.org/TR/ldp/>
//!
//! ## Supported Container Types
//! - `ldp:BasicContainer` - simple flat container
//! - `ldp:DirectContainer` - container with direct membership relations
//! - `ldp:IndirectContainer` - container with indirect membership via content relation
//!
//! ## Supported Operations
//! GET, HEAD, POST, PUT, DELETE, PATCH, OPTIONS on both resources and containers.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// --- Constants ---

/// W3C LDP namespace
pub const LDP_NS: &str = "http://www.w3.org/ns/ldp#";
/// RDF namespace
pub const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
/// Dublin Core Terms namespace
pub const DCT_NS: &str = "http://purl.org/dc/terms/";

// --- Resource Types ---

/// LDP resource type classification per the W3C LDP specification.
#[derive(Debug, Clone, PartialEq)]
pub enum LdpResourceType {
    /// `ldp:RDFSource` - any RDF document
    RdfSource,
    /// `ldp:NonRDFSource` - binary / non-RDF content
    NonRdfSource,
    /// `ldp:Container` - abstract base for all containers
    Container,
    /// `ldp:BasicContainer` - simple flat container
    BasicContainer,
    /// `ldp:DirectContainer` - container with membership relations
    DirectContainer {
        membership_resource: String,
        has_member_relation: String,
    },
    /// `ldp:IndirectContainer` - indirect membership via a content relation
    IndirectContainer {
        membership_resource: String,
        has_member_relation: String,
        inserted_content_relation: String,
    },
}

impl LdpResourceType {
    /// Return the primary `Link` rel type IRI for this resource type.
    pub fn link_type_iri(&self) -> &'static str {
        match self {
            LdpResourceType::RdfSource => "http://www.w3.org/ns/ldp#RDFSource",
            LdpResourceType::NonRdfSource => "http://www.w3.org/ns/ldp#NonRDFSource",
            LdpResourceType::Container => "http://www.w3.org/ns/ldp#Container",
            LdpResourceType::BasicContainer => "http://www.w3.org/ns/ldp#BasicContainer",
            LdpResourceType::DirectContainer { .. } => "http://www.w3.org/ns/ldp#DirectContainer",
            LdpResourceType::IndirectContainer { .. } => {
                "http://www.w3.org/ns/ldp#IndirectContainer"
            }
        }
    }

    /// Returns `true` if this type is any variety of container.
    pub fn is_container(&self) -> bool {
        matches!(
            self,
            LdpResourceType::Container
                | LdpResourceType::BasicContainer
                | LdpResourceType::DirectContainer { .. }
                | LdpResourceType::IndirectContainer { .. }
        )
    }
}

// --- HTTP Method ---

/// Supported HTTP methods for LDP interactions.
#[derive(Debug, Clone, PartialEq)]
pub enum HttpMethod {
    Get,
    Head,
    Post,
    Put,
    Delete,
    Patch,
    Options,
}

impl HttpMethod {
    /// Parse a method string (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "GET" => Some(HttpMethod::Get),
            "HEAD" => Some(HttpMethod::Head),
            "POST" => Some(HttpMethod::Post),
            "PUT" => Some(HttpMethod::Put),
            "DELETE" => Some(HttpMethod::Delete),
            "PATCH" => Some(HttpMethod::Patch),
            "OPTIONS" => Some(HttpMethod::Options),
            _ => None,
        }
    }

    /// Return the canonical string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            HttpMethod::Get => "GET",
            HttpMethod::Head => "HEAD",
            HttpMethod::Post => "POST",
            HttpMethod::Put => "PUT",
            HttpMethod::Delete => "DELETE",
            HttpMethod::Patch => "PATCH",
            HttpMethod::Options => "OPTIONS",
        }
    }
}

// --- LDP Operations ---

/// Typed representation of an LDP operation.
#[derive(Debug, Clone)]
pub enum LdpOperation {
    GetResource {
        iri: String,
        prefer: Vec<PreferHeader>,
    },
    HeadResource {
        iri: String,
    },
    PostToContainer {
        container_iri: String,
        content_type: String,
        body: Vec<u8>,
        slug: Option<String>,
    },
    PutResource {
        iri: String,
        content_type: String,
        body: Vec<u8>,
        if_match: Option<String>,
    },
    DeleteResource {
        iri: String,
    },
    PatchResource {
        iri: String,
        patch_content_type: String,
        body: Vec<u8>,
    },
    OptionsResource {
        iri: String,
    },
}

// --- Prefer Header ---

/// LDP `Prefer` header values for controlling response content.
#[derive(Debug, Clone, PartialEq)]
pub enum PreferHeader {
    ReturnRepresentation,
    ReturnMinimal,
    /// Include `ldp:PreferContainment` triples
    IncludeContainment,
    /// Omit `ldp:PreferContainment` triples
    OmitContainment,
    /// Include `ldp:PreferMembership` triples
    IncludeMembership,
    /// Omit `ldp:PreferMembership` triples
    OmitMembership,
    /// Include `ldp:PreferMinimalContainer`
    IncludeMinimalContainer,
}

impl PreferHeader {
    /// Parse the value portion of a `Prefer:` header.
    pub fn parse(value: &str) -> Vec<Self> {
        let mut result = Vec::new();
        for part in value.split(';') {
            let trimmed = part.trim();
            if trimmed == "return=representation" {
                result.push(PreferHeader::ReturnRepresentation);
            } else if trimmed == "return=minimal" {
                result.push(PreferHeader::ReturnMinimal);
            } else if trimmed.starts_with("include=") {
                let iris = trimmed.trim_start_matches("include=").trim_matches('"');
                for iri in iris.split_whitespace() {
                    match iri {
                        "http://www.w3.org/ns/ldp#PreferContainment" => {
                            result.push(PreferHeader::IncludeContainment);
                        }
                        "http://www.w3.org/ns/ldp#PreferMembership" => {
                            result.push(PreferHeader::IncludeMembership);
                        }
                        "http://www.w3.org/ns/ldp#PreferMinimalContainer" => {
                            result.push(PreferHeader::IncludeMinimalContainer);
                        }
                        _ => {}
                    }
                }
            } else if trimmed.starts_with("omit=") {
                let iris = trimmed.trim_start_matches("omit=").trim_matches('"');
                for iri in iris.split_whitespace() {
                    match iri {
                        "http://www.w3.org/ns/ldp#PreferContainment" => {
                            result.push(PreferHeader::OmitContainment);
                        }
                        "http://www.w3.org/ns/ldp#PreferMembership" => {
                            result.push(PreferHeader::OmitMembership);
                        }
                        _ => {}
                    }
                }
            }
        }
        result
    }
}

// --- Request / Response ---

/// An inbound LDP HTTP request.
#[derive(Debug, Clone)]
pub struct LdpRequest {
    pub method: HttpMethod,
    pub iri: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
}

impl LdpRequest {
    /// Create a simple GET request.
    pub fn get(iri: impl Into<String>) -> Self {
        LdpRequest {
            method: HttpMethod::Get,
            iri: iri.into(),
            headers: Vec::new(),
            body: None,
        }
    }

    /// Retrieve the first value for a given header name (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        let lower = name.to_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k.to_lowercase() == lower)
            .map(|(_, v)| v.as_str())
    }

    /// Collect all parsed `Prefer` header preferences.
    pub fn prefer_headers(&self) -> Vec<PreferHeader> {
        self.headers
            .iter()
            .filter(|(k, _)| k.to_lowercase() == "prefer")
            .flat_map(|(_, v)| PreferHeader::parse(v))
            .collect()
    }

    /// Return the `Slug` header value if present.
    pub fn slug(&self) -> Option<&str> {
        self.header("slug")
    }

    /// Return the `Content-Type` header value if present.
    pub fn content_type(&self) -> Option<&str> {
        self.header("content-type")
    }

    /// Return the `If-Match` header value if present.
    pub fn if_match(&self) -> Option<&str> {
        self.header("if-match")
    }
}

/// An outbound LDP HTTP response.
#[derive(Debug, Clone)]
pub struct LdpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
}

impl LdpResponse {
    // --- Status helpers ---

    /// 200 OK with the given body.
    pub fn ok(body: impl Into<Vec<u8>>) -> Self {
        LdpResponse {
            status: 200,
            headers: Vec::new(),
            body: Some(body.into()),
        }
    }

    /// 201 Created with a `Location` header pointing to `iri`.
    pub fn created(iri: &str) -> Self {
        let mut resp = LdpResponse {
            status: 201,
            headers: Vec::new(),
            body: None,
        };
        resp.headers.push(("Location".to_string(), iri.to_string()));
        resp
    }

    /// 204 No Content.
    pub fn no_content() -> Self {
        LdpResponse {
            status: 204,
            headers: Vec::new(),
            body: None,
        }
    }

    /// 404 Not Found.
    pub fn not_found() -> Self {
        LdpResponse {
            status: 404,
            headers: Vec::new(),
            body: Some(b"404 Not Found".to_vec()),
        }
    }

    /// 410 Gone.
    pub fn gone() -> Self {
        LdpResponse {
            status: 410,
            headers: Vec::new(),
            body: Some(b"410 Gone".to_vec()),
        }
    }

    /// 405 Method Not Allowed with an `Allow` header listing `allowed` methods.
    pub fn method_not_allowed(allowed: &[&str]) -> Self {
        let mut resp = LdpResponse {
            status: 405,
            headers: Vec::new(),
            body: Some(b"405 Method Not Allowed".to_vec()),
        };
        resp.headers.push(("Allow".to_string(), allowed.join(", ")));
        resp
    }

    /// 412 Precondition Failed (ETag mismatch).
    pub fn precondition_failed() -> Self {
        LdpResponse {
            status: 412,
            headers: Vec::new(),
            body: Some(b"412 Precondition Failed".to_vec()),
        }
    }

    /// 409 Conflict.
    pub fn conflict(reason: &str) -> Self {
        LdpResponse {
            status: 409,
            headers: Vec::new(),
            body: Some(format!("409 Conflict: {reason}").into_bytes()),
        }
    }

    /// 415 Unsupported Media Type.
    pub fn unsupported_media_type() -> Self {
        LdpResponse {
            status: 415,
            headers: Vec::new(),
            body: Some(b"415 Unsupported Media Type".to_vec()),
        }
    }

    // --- Header builder helpers ---

    /// Append `Link` headers declaring the LDP type of this resource.
    ///
    /// Always adds the base `ldp:Resource` link plus the specific type link.
    pub fn with_ldp_type(mut self, resource_type: &LdpResourceType) -> Self {
        self.headers.push((
            "Link".to_string(),
            "<http://www.w3.org/ns/ldp#Resource>; rel=\"type\"".to_string(),
        ));
        let specific = resource_type.link_type_iri();
        self.headers
            .push(("Link".to_string(), format!("<{specific}>; rel=\"type\"")));
        if matches!(
            resource_type,
            LdpResourceType::BasicContainer
                | LdpResourceType::DirectContainer { .. }
                | LdpResourceType::IndirectContainer { .. }
        ) {
            self.headers.push((
                "Link".to_string(),
                "<http://www.w3.org/ns/ldp#Container>; rel=\"type\"".to_string(),
            ));
            self.headers.push((
                "Link".to_string(),
                "<http://www.w3.org/ns/ldp#RDFSource>; rel=\"type\"".to_string(),
            ));
        }
        self
    }

    /// Append an `ETag` header.
    pub fn with_etag(mut self, etag: &str) -> Self {
        self.headers
            .push(("ETag".to_string(), format!("\"{etag}\"")));
        self
    }

    /// Append an `Allow` header listing the given HTTP methods.
    pub fn with_allow(mut self, methods: &[&str]) -> Self {
        self.headers.push(("Allow".to_string(), methods.join(", ")));
        self
    }

    /// Append an `Accept-Post` header (returned by containers).
    pub fn with_accept_post(mut self, content_types: &[&str]) -> Self {
        self.headers
            .push(("Accept-Post".to_string(), content_types.join(", ")));
        self
    }

    /// Append a `Content-Type` header.
    pub fn with_content_type(mut self, ct: &str) -> Self {
        self.headers
            .push(("Content-Type".to_string(), ct.to_string()));
        self
    }

    // --- Accessors ---

    /// Return the first value for the given header name (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        let lower = name.to_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k.to_lowercase() == lower)
            .map(|(_, v)| v.as_str())
    }

    /// Collect all values for the given header name (case-insensitive).
    pub fn all_headers(&self, name: &str) -> Vec<&str> {
        let lower = name.to_lowercase();
        self.headers
            .iter()
            .filter(|(k, _)| k.to_lowercase() == lower)
            .map(|(_, v)| v.as_str())
            .collect()
    }
}

// --- LDP Member ---

/// A single member entry inside an LDP container.
#[derive(Debug, Clone)]
pub struct LdpMember {
    pub iri: String,
    pub resource_type: LdpResourceType,
    pub created_at: SystemTime,
    pub etag: String,
}

impl LdpMember {
    /// Create a new member with a computed ETag based on the IRI and timestamp.
    pub fn new(iri: impl Into<String>, resource_type: LdpResourceType) -> Self {
        let iri = iri.into();
        let created_at = SystemTime::now();
        let etag = compute_etag(&iri, &created_at);
        LdpMember {
            iri,
            resource_type,
            created_at,
            etag,
        }
    }
}

// --- LDP Container ---

/// An LDP Container manages a collection of member resources.
#[derive(Debug, Clone)]
pub struct LdpContainer {
    pub iri: String,
    pub resource_type: LdpResourceType,
    pub members: Vec<LdpMember>,
    /// Additional RDF properties as `(subject, predicate, object)` triples.
    pub properties: Vec<(String, String, String)>,
    /// Slug counter used to generate unique IRIs.
    slug_counter: u64,
}

impl LdpContainer {
    // --- Constructors ---

    /// Create a new `ldp:BasicContainer`.
    pub fn new_basic(iri: impl Into<String>) -> Self {
        let iri = iri.into();
        LdpContainer {
            iri,
            resource_type: LdpResourceType::BasicContainer,
            members: Vec::new(),
            properties: Vec::new(),
            slug_counter: 0,
        }
    }

    /// Create a new `ldp:DirectContainer`.
    pub fn new_direct(
        iri: impl Into<String>,
        membership_resource: impl Into<String>,
        has_member_relation: impl Into<String>,
    ) -> Self {
        LdpContainer {
            iri: iri.into(),
            resource_type: LdpResourceType::DirectContainer {
                membership_resource: membership_resource.into(),
                has_member_relation: has_member_relation.into(),
            },
            members: Vec::new(),
            properties: Vec::new(),
            slug_counter: 0,
        }
    }

    /// Create a new `ldp:IndirectContainer`.
    pub fn new_indirect(
        iri: impl Into<String>,
        membership_resource: impl Into<String>,
        has_member_relation: impl Into<String>,
        inserted_content_relation: impl Into<String>,
    ) -> Self {
        LdpContainer {
            iri: iri.into(),
            resource_type: LdpResourceType::IndirectContainer {
                membership_resource: membership_resource.into(),
                has_member_relation: has_member_relation.into(),
                inserted_content_relation: inserted_content_relation.into(),
            },
            members: Vec::new(),
            properties: Vec::new(),
            slug_counter: 0,
        }
    }

    // --- Member management ---

    /// Add a member to this container.
    ///
    /// Returns an error if a member with the same IRI already exists.
    pub fn add_member(&mut self, member: LdpMember) -> Result<(), LdpError> {
        if self.members.iter().any(|m| m.iri == member.iri) {
            return Err(LdpError::Conflict(format!(
                "member '{}' already exists",
                member.iri
            )));
        }
        self.members.push(member);
        Ok(())
    }

    /// Remove the member with the given IRI.
    ///
    /// Returns an error if no such member exists.
    pub fn remove_member(&mut self, iri: &str) -> Result<(), LdpError> {
        let pos = self
            .members
            .iter()
            .position(|m| m.iri == iri)
            .ok_or_else(|| LdpError::NotFound(format!("member '{iri}' not found")))?;
        self.members.remove(pos);
        Ok(())
    }

    /// Find and return a reference to the member with the given IRI.
    pub fn get_member(&self, iri: &str) -> Option<&LdpMember> {
        self.members.iter().find(|m| m.iri == iri)
    }

    // --- IRI generation ---

    /// Generate a unique member IRI.
    pub fn generate_iri(&mut self, slug: Option<&str>) -> String {
        let base = self.iri.trim_end_matches('/');
        match slug {
            Some(s) => {
                let candidate = format!("{base}/{s}");
                if self.members.iter().any(|m| m.iri == candidate) {
                    self.slug_counter += 1;
                    format!("{base}/{s}_{}", self.slug_counter)
                } else {
                    candidate
                }
            }
            None => {
                self.slug_counter += 1;
                format!("{base}/{}", self.slug_counter)
            }
        }
    }

    // --- Serialisation ---

    /// Serialise the container as Turtle, respecting `Prefer` headers.
    pub fn to_turtle(&self, prefer: &[PreferHeader]) -> String {
        let omit_containment = prefer.contains(&PreferHeader::OmitContainment);
        let omit_membership = prefer.contains(&PreferHeader::OmitMembership);
        let minimal = prefer.contains(&PreferHeader::IncludeMinimalContainer);

        let mut out = String::new();

        out.push_str("@prefix ldp: <http://www.w3.org/ns/ldp#> .\n");
        out.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
        out.push_str("@prefix dct: <http://purl.org/dc/terms/> .\n");
        out.push('\n');

        out.push_str(&format!("<{}>\n", self.iri));

        let type_iri = match &self.resource_type {
            LdpResourceType::BasicContainer => "ldp:BasicContainer",
            LdpResourceType::DirectContainer { .. } => "ldp:DirectContainer",
            LdpResourceType::IndirectContainer { .. } => "ldp:IndirectContainer",
            _ => "ldp:Container",
        };
        out.push_str(&format!("    a {type_iri} ;\n"));
        out.push_str("    a ldp:Container ;\n");
        out.push_str("    a ldp:RDFSource ;\n");

        if let LdpResourceType::DirectContainer {
            membership_resource,
            has_member_relation,
        } = &self.resource_type
        {
            if !omit_membership {
                out.push_str(&format!(
                    "    ldp:membershipResource <{membership_resource}> ;\n"
                ));
                out.push_str(&format!(
                    "    ldp:hasMemberRelation <{has_member_relation}> ;\n"
                ));
            }
        }

        if let LdpResourceType::IndirectContainer {
            membership_resource,
            has_member_relation,
            inserted_content_relation,
        } = &self.resource_type
        {
            if !omit_membership {
                out.push_str(&format!(
                    "    ldp:membershipResource <{membership_resource}> ;\n"
                ));
                out.push_str(&format!(
                    "    ldp:hasMemberRelation <{has_member_relation}> ;\n"
                ));
                out.push_str(&format!(
                    "    ldp:insertedContentRelation <{inserted_content_relation}> ;\n"
                ));
            }
        }

        for (s, p, o) in &self.properties {
            if s == &self.iri {
                out.push_str(&format!("    <{p}> \"{o}\" ;\n"));
            }
        }

        if !omit_containment && !minimal {
            for member in &self.members {
                out.push_str(&format!("    ldp:contains <{}> ;\n", member.iri));
            }
        }

        if out.trim_end().ends_with(';') {
            let trimmed = out.trim_end().trim_end_matches(';').trim_end();
            out = format!("{trimmed} .\n");
        } else {
            out.push_str("    .\n");
        }

        out
    }

    /// Serialise the container as JSON-LD (compact form).
    pub fn to_jsonld(&self) -> String {
        let mut obj = String::new();
        obj.push_str("{\n");
        obj.push_str("  \"@context\": {\n");
        obj.push_str("    \"ldp\": \"http://www.w3.org/ns/ldp#\",\n");
        obj.push_str("    \"rdf\": \"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"\n");
        obj.push_str("  },\n");
        obj.push_str(&format!("  \"@id\": \"{}\",\n", self.iri));

        let type_iri = match &self.resource_type {
            LdpResourceType::BasicContainer => "ldp:BasicContainer",
            LdpResourceType::DirectContainer { .. } => "ldp:DirectContainer",
            LdpResourceType::IndirectContainer { .. } => "ldp:IndirectContainer",
            _ => "ldp:Container",
        };
        obj.push_str(&format!(
            "  \"@type\": [\"{type_iri}\", \"ldp:Container\", \"ldp:RDFSource\"],\n"
        ));

        if !self.members.is_empty() {
            obj.push_str("  \"ldp:contains\": [\n");
            let last = self.members.len() - 1;
            for (i, m) in self.members.iter().enumerate() {
                let comma = if i < last { "," } else { "" };
                obj.push_str(&format!("    {{\"@id\": \"{}\"}}{comma}\n", m.iri));
            }
            obj.push_str("  ]\n");
        } else {
            obj.push_str("  \"ldp:contains\": []\n");
        }

        obj.push_str("}\n");
        obj
    }

    /// Compute an ETag for the container based on its current state.
    pub fn etag(&self) -> String {
        let member_iris: Vec<&str> = self.members.iter().map(|m| m.iri.as_str()).collect();
        let state = format!("{}{}", self.iri, member_iris.join(","));
        compute_etag_str(&state)
    }
}

// --- LDP Resource ---

/// A non-container LDP resource (RDF or non-RDF).
#[derive(Debug, Clone)]
pub struct LdpResource {
    pub iri: String,
    pub content_type: String,
    pub body: Vec<u8>,
    pub etag: String,
    pub resource_type: LdpResourceType,
}

impl LdpResource {
    /// Create a new RDF source resource.
    pub fn new_rdf_source(iri: impl Into<String>, body: impl Into<Vec<u8>>) -> Self {
        let iri = iri.into();
        let body = body.into();
        let etag = compute_etag_str(&format!("{}{:?}", iri, body.len()));
        LdpResource {
            iri,
            content_type: "text/turtle".to_string(),
            body,
            etag,
            resource_type: LdpResourceType::RdfSource,
        }
    }

    /// Create a new non-RDF source resource (binary).
    pub fn new_non_rdf(
        iri: impl Into<String>,
        content_type: impl Into<String>,
        body: impl Into<Vec<u8>>,
    ) -> Self {
        let iri = iri.into();
        let body = body.into();
        let etag = compute_etag_str(&format!("{}{:?}", iri, body.len()));
        LdpResource {
            content_type: content_type.into(),
            iri,
            body,
            etag,
            resource_type: LdpResourceType::NonRdfSource,
        }
    }

    /// Update the body and regenerate the ETag.
    pub fn update_body(&mut self, body: impl Into<Vec<u8>>, content_type: impl Into<String>) {
        self.body = body.into();
        self.content_type = content_type.into();
        self.etag = compute_etag_str(&format!("{}{:?}", self.iri, self.body.len()));
    }
}

// --- Error ---

/// Errors that may arise during LDP operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LdpError {
    NotFound(String),
    Conflict(String),
    PreconditionFailed(String),
    UnsupportedMediaType(String),
    MethodNotAllowed(String),
    InvalidIri(String),
    Internal(String),
}

impl std::fmt::Display for LdpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LdpError::NotFound(msg) => write!(f, "Not Found: {msg}"),
            LdpError::Conflict(msg) => write!(f, "Conflict: {msg}"),
            LdpError::PreconditionFailed(msg) => write!(f, "Precondition Failed: {msg}"),
            LdpError::UnsupportedMediaType(msg) => write!(f, "Unsupported Media Type: {msg}"),
            LdpError::MethodNotAllowed(msg) => write!(f, "Method Not Allowed: {msg}"),
            LdpError::InvalidIri(msg) => write!(f, "Invalid IRI: {msg}"),
            LdpError::Internal(msg) => write!(f, "Internal Error: {msg}"),
        }
    }
}

impl std::error::Error for LdpError {}

// --- LDP Service ---

/// Accepted RDF content types for LDP POST/PUT.
const ACCEPTED_RDF_TYPES: &[&str] = &[
    "text/turtle",
    "application/ld+json",
    "application/n-triples",
    "application/rdf+xml",
    "text/n3",
];

/// The main LDP service handler.
///
/// Maintains an in-memory registry of containers and resources and processes
/// `LdpRequest` values into `LdpResponse` values.
pub struct LdpService {
    pub base_url: String,
    pub containers: HashMap<String, LdpContainer>,
    pub resources: HashMap<String, LdpResource>,
    /// IRIs that have been deleted (for 410 Gone responses).
    deleted_iris: std::collections::HashSet<String>,
}

impl LdpService {
    // --- Constructor ---

    /// Create a new `LdpService` with the given base URL.
    ///
    /// A root `ldp:BasicContainer` is automatically created at `base_url`.
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        let root = LdpContainer::new_basic(base_url.clone());
        let mut containers = HashMap::new();
        containers.insert(base_url.clone(), root);
        LdpService {
            base_url,
            containers,
            resources: HashMap::new(),
            deleted_iris: std::collections::HashSet::new(),
        }
    }

    // --- Public API ---

    /// Handle an incoming LDP request and return an appropriate response.
    pub fn handle(&mut self, request: LdpRequest) -> LdpResponse {
        let prefer = request.prefer_headers();
        let iri = request.iri.clone();

        match &request.method {
            HttpMethod::Get => self.handle_get(&iri, prefer),
            HttpMethod::Head => self.handle_head(&iri),
            HttpMethod::Post => {
                let content_type = request.content_type().unwrap_or("text/turtle").to_string();
                let body = request.body.clone().unwrap_or_default();
                let slug = request.slug().map(str::to_string);
                self.handle_post(&iri, &content_type, &body, slug.as_deref())
            }
            HttpMethod::Put => {
                let content_type = request.content_type().unwrap_or("text/turtle").to_string();
                let body = request.body.clone().unwrap_or_default();
                let if_match = request.if_match().map(str::to_string);
                self.handle_put(&iri, &content_type, &body, if_match.as_deref())
            }
            HttpMethod::Delete => self.handle_delete(&iri),
            HttpMethod::Patch => {
                let patch_ct = request
                    .content_type()
                    .unwrap_or("application/sparql-update")
                    .to_string();
                let body = request.body.clone().unwrap_or_default();
                self.handle_patch(&iri, &patch_ct, &body)
            }
            HttpMethod::Options => self.handle_options(&iri),
        }
    }

    // --- Private handlers ---

    fn handle_get(&self, iri: &str, prefer: Vec<PreferHeader>) -> LdpResponse {
        if let Some(container) = self.containers.get(iri) {
            let body = container.to_turtle(&prefer);
            let etag = container.etag();
            LdpResponse::ok(body)
                .with_content_type("text/turtle")
                .with_etag(&etag)
                .with_ldp_type(&container.resource_type.clone())
                .with_allow(&["GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS"])
                .with_accept_post(ACCEPTED_RDF_TYPES)
        } else if let Some(resource) = self.resources.get(iri) {
            LdpResponse::ok(resource.body.clone())
                .with_content_type(&resource.content_type.clone())
                .with_etag(&resource.etag.clone())
                .with_ldp_type(&resource.resource_type.clone())
                .with_allow(&["GET", "HEAD", "PUT", "DELETE", "OPTIONS"])
        } else if self.deleted_iris.contains(iri) {
            LdpResponse::gone()
        } else {
            LdpResponse::not_found()
        }
    }

    fn handle_head(&self, iri: &str) -> LdpResponse {
        let mut resp = self.handle_get(iri, Vec::new());
        resp.body = None;
        resp
    }

    fn handle_post(
        &mut self,
        container_iri: &str,
        content_type: &str,
        body: &[u8],
        slug: Option<&str>,
    ) -> LdpResponse {
        let base_ct = content_type.split(';').next().unwrap_or("").trim();
        if !ACCEPTED_RDF_TYPES.contains(&base_ct) && base_ct != "application/octet-stream" {
            return LdpResponse::unsupported_media_type();
        }

        if !self.containers.contains_key(container_iri) {
            return LdpResponse::not_found();
        }

        let new_iri = {
            let container = self
                .containers
                .get_mut(container_iri)
                .expect("checked above");
            container.generate_iri(slug)
        };

        let resource_type = if ACCEPTED_RDF_TYPES.contains(&base_ct) {
            LdpResourceType::RdfSource
        } else {
            LdpResourceType::NonRdfSource
        };

        let resource = LdpResource {
            iri: new_iri.clone(),
            content_type: content_type.to_string(),
            body: body.to_vec(),
            etag: compute_etag_str(&format!("{new_iri}{}", body.len())),
            resource_type: resource_type.clone(),
        };

        self.resources.insert(new_iri.clone(), resource);

        let member = LdpMember::new(new_iri.clone(), resource_type);
        let container = self
            .containers
            .get_mut(container_iri)
            .expect("checked above");
        let _ = container.add_member(member);

        LdpResponse::created(&new_iri).with_ldp_type(&LdpResourceType::RdfSource)
    }

    fn handle_put(
        &mut self,
        iri: &str,
        content_type: &str,
        body: &[u8],
        if_match: Option<&str>,
    ) -> LdpResponse {
        if let Some(expected_etag) = if_match {
            let expected = expected_etag.trim_matches('"');

            if let Some(existing) = self.resources.get(iri) {
                if existing.etag != expected {
                    return LdpResponse::precondition_failed();
                }
                let mut updated = existing.clone();
                updated.update_body(body.to_vec(), content_type.to_string());
                let updated_etag = updated.etag.clone();
                self.resources.insert(iri.to_string(), updated);
                return LdpResponse::no_content().with_etag(&updated_etag);
            } else if let Some(container) = self.containers.get(iri) {
                if container.etag() != expected {
                    return LdpResponse::precondition_failed();
                }
                return LdpResponse::no_content();
            } else {
                return LdpResponse::precondition_failed();
            }
        }

        let is_new = !self.resources.contains_key(iri) && !self.containers.contains_key(iri);

        let resource_type = {
            let base_ct = content_type.split(';').next().unwrap_or("").trim();
            if ACCEPTED_RDF_TYPES.contains(&base_ct) {
                LdpResourceType::RdfSource
            } else {
                LdpResourceType::NonRdfSource
            }
        };

        let etag = compute_etag_str(&format!("{iri}{}", body.len()));

        let resource = LdpResource {
            iri: iri.to_string(),
            content_type: content_type.to_string(),
            body: body.to_vec(),
            etag: etag.clone(),
            resource_type,
        };

        self.resources.insert(iri.to_string(), resource);
        self.deleted_iris.remove(iri);

        if is_new {
            LdpResponse::created(iri).with_etag(&etag)
        } else {
            LdpResponse::no_content().with_etag(&etag)
        }
    }

    fn handle_delete(&mut self, iri: &str) -> LdpResponse {
        if self.resources.remove(iri).is_some() {
            self.deleted_iris.insert(iri.to_string());
            self.remove_member_from_containers(iri);
            return LdpResponse::no_content();
        }
        if let Some(container) = self.containers.get(iri) {
            if !container.members.is_empty() {
                return LdpResponse::conflict("container is not empty");
            }
            self.containers.remove(iri);
            self.deleted_iris.insert(iri.to_string());
            self.remove_member_from_containers(iri);
            return LdpResponse::no_content();
        }
        if self.deleted_iris.contains(iri) {
            return LdpResponse::gone();
        }
        LdpResponse::not_found()
    }

    fn handle_patch(&mut self, iri: &str, _patch_ct: &str, body: &[u8]) -> LdpResponse {
        if !self.resources.contains_key(iri) && !self.containers.contains_key(iri) {
            return LdpResponse::not_found();
        }
        if let Some(resource) = self.resources.get_mut(iri) {
            resource.body.extend_from_slice(body);
            resource.etag = compute_etag_str(&format!("{iri}{}", resource.body.len()));
        }
        LdpResponse::no_content()
    }

    fn handle_options(&self, iri: &str) -> LdpResponse {
        let is_container = self.containers.contains_key(iri);
        let is_resource = self.resources.contains_key(iri);

        if is_container {
            LdpResponse::no_content()
                .with_allow(&["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
                .with_accept_post(ACCEPTED_RDF_TYPES)
                .with_ldp_type(&LdpResourceType::BasicContainer)
        } else if is_resource {
            LdpResponse::no_content()
                .with_allow(&["GET", "HEAD", "PUT", "DELETE", "PATCH", "OPTIONS"])
                .with_ldp_type(&LdpResourceType::RdfSource)
        } else {
            LdpResponse::no_content().with_allow(&["GET", "HEAD", "PUT", "OPTIONS"])
        }
    }

    // --- Helpers ---

    fn remove_member_from_containers(&mut self, iri: &str) {
        for container in self.containers.values_mut() {
            container.members.retain(|m| m.iri != iri);
        }
    }

    /// Register a pre-built container under the service.
    pub fn register_container(&mut self, container: LdpContainer) {
        self.containers.insert(container.iri.clone(), container);
    }
}

// --- Utility ---

fn compute_etag(iri: &str, time: &SystemTime) -> String {
    let secs = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    compute_etag_str(&format!("{iri}{secs}"))
}

fn compute_etag_str(input: &str) -> String {
    const FNV_PRIME: u64 = 1_099_511_628_211;
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    let mut hash: u64 = FNV_OFFSET;
    for byte in input.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

// --- Tests (extracted to ldp_tests.rs for file size compliance) ---

include!("ldp_tests.rs");
