//! # Store - fetch_rdf_from_url_group Methods
//!
//! This module contains the HTTP fetch used by SPARQL `LOAD`.
//!
//! ## SSRF hardening
//!
//! The URL fed to [`Store::fetch_rdf_from_url`] flows directly from a
//! user-supplied `LOAD <sourceIRI>` clause, so it is treated as fully
//! untrusted. Before any request is made the target host is resolved and every
//! resolved address is required to be a *global* (public) IP: loopback,
//! private, link-local, unique-local, shared (CGN), documentation and other
//! non-global ranges are rejected. This blocks the classic SSRF vectors —
//! `127.0.0.1`, the `169.254.169.254` cloud-metadata endpoint, and the
//! `10.x`/`172.16.x`/`192.168.x` private ranges. The validated address is then
//! *pinned* into the HTTP client so a subsequent DNS answer cannot rebind the
//! connection to an internal host, redirects are capped and each redirect hop
//! is re-validated, and the response body is streamed under a hard byte cap so
//! a hostile/oversized source cannot exhaust memory.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use url::Url;

/// Maximum number of bytes accepted from a single `LOAD` response body.
const MAX_LOAD_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum number of HTTP redirects followed during a `LOAD`.
const MAX_LOAD_REDIRECTS: usize = 5;

impl Store {
    /// Fetch RDF data from URL using HTTP, with SSRF and size protections.
    pub(super) async fn fetch_rdf_from_url(
        &self,
        url: &str,
    ) -> FusekiResult<(String, Option<String>)> {
        let parsed = Url::parse(url).map_err(|e| {
            FusekiError::update_execution(format!("Invalid LOAD source URL '{url}': {e}"))
        })?;

        // Only http(s) is fetched over the network. Other schemes (file, ftp,
        // gopher, …) are refused outright — file:// in particular would let a
        // LOAD read arbitrary local files.
        match parsed.scheme() {
            "http" | "https" => {}
            other => {
                return Err(FusekiError::update_execution(format!(
                    "LOAD only supports http(s) sources; refusing scheme '{other}' in '{url}'"
                )));
            }
        }

        let host = parsed.host_str().ok_or_else(|| {
            FusekiError::update_execution(format!("LOAD source URL '{url}' has no host"))
        })?;
        let port = parsed
            .port_or_known_default()
            .unwrap_or(if parsed.scheme() == "https" { 443 } else { 80 });

        // Resolve + validate the target host before connecting.
        let safe_addr = resolve_and_validate_host(host, port)?;

        // Build a client that (a) pins the resolved+validated address so DNS
        // rebinding cannot redirect the socket to an internal host, and (b)
        // re-validates every redirect hop against the same global-IP policy.
        let client = reqwest::Client::builder()
            .resolve(host, safe_addr)
            .redirect(guarded_redirect_policy())
            .build()
            .map_err(|e| {
                FusekiError::update_execution(format!("Failed to build LOAD HTTP client: {e}"))
            })?;

        let response = client.get(parsed.clone()).send().await.map_err(|e| {
            FusekiError::update_execution(format!("Failed to fetch '{}': {e}", url))
        })?;
        if !response.status().is_success() {
            return Err(FusekiError::update_execution(format!(
                "HTTP error fetching '{}': {}",
                url,
                response.status()
            )));
        }

        // Reject early if the server advertises an over-cap body.
        if let Some(len) = response.content_length() {
            if len > MAX_LOAD_BYTES {
                return Err(FusekiError::update_execution(format!(
                    "LOAD response from '{}' is too large ({len} bytes > {MAX_LOAD_BYTES} byte limit)",
                    url
                )));
            }
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        // Stream the body under a hard cap so a server that lies about (or
        // omits) Content-Length cannot exhaust memory.
        let mut response = response;
        let mut body_bytes: Vec<u8> = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|e| {
            FusekiError::update_execution(format!("Failed to read response body: {e}"))
        })? {
            if body_bytes.len() as u64 + chunk.len() as u64 > MAX_LOAD_BYTES {
                return Err(FusekiError::update_execution(format!(
                    "LOAD response from '{}' exceeded the {MAX_LOAD_BYTES} byte limit",
                    url
                )));
            }
            body_bytes.extend_from_slice(&chunk);
        }

        let body = String::from_utf8(body_bytes).map_err(|e| {
            FusekiError::parse(format!("LOAD response body is not valid UTF-8: {e}"))
        })?;
        Ok((body, content_type))
    }
}

/// Redirect policy: cap the number of hops and reject any redirect whose target
/// host does not resolve entirely to global addresses.
fn guarded_redirect_policy() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(|attempt| {
        if attempt.previous().len() >= MAX_LOAD_REDIRECTS {
            return attempt.error(format!(
                "LOAD exceeded the maximum of {MAX_LOAD_REDIRECTS} redirects"
            ));
        }
        // Extract everything needed from the borrowed URL first, so the
        // borrow ends before `attempt` is consumed by follow()/error().
        let url = attempt.url();
        let scheme = url.scheme().to_string();
        let host = url.host_str().map(|h| h.to_string());
        let port = url
            .port_or_known_default()
            .unwrap_or(if scheme == "https" { 443 } else { 80 });

        if scheme != "http" && scheme != "https" {
            return attempt.error(format!("LOAD redirect to unsupported scheme '{scheme}'"));
        }
        let host = match host {
            Some(h) => h,
            None => return attempt.error("LOAD redirect target has no host".to_string()),
        };
        match resolve_and_validate_host(&host, port) {
            Ok(_) => attempt.follow(),
            Err(e) => attempt.error(e.to_string()),
        }
    })
}

/// Resolve `host:port` and require that every resolved address is a global
/// (public) IP. Returns the first validated socket address (used to pin the
/// connection). Rejecting when *any* resolution is non-global defends against
/// split-horizon DNS returning a mix of public and internal addresses.
fn resolve_and_validate_host(host: &str, port: u16) -> FusekiResult<SocketAddr> {
    let addrs: Vec<SocketAddr> = (host, port)
        .to_socket_addrs()
        .map_err(|e| {
            FusekiError::update_execution(format!("Failed to resolve LOAD host '{host}': {e}"))
        })?
        .collect();

    let first = addrs.first().copied().ok_or_else(|| {
        FusekiError::update_execution(format!("LOAD host '{host}' did not resolve to any address"))
    })?;

    for addr in &addrs {
        if !ip_is_global(addr.ip()) {
            return Err(FusekiError::update_execution(format!(
                "LOAD refused: host '{host}' resolves to non-global address {} \
                 (loopback/private/link-local/metadata ranges are blocked to prevent SSRF)",
                addr.ip()
            )));
        }
    }
    Ok(first)
}

/// Whether `ip` is a global (publicly routable) address suitable for a `LOAD`
/// fetch. Implemented locally because `IpAddr::is_global` is still unstable.
fn ip_is_global(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => ipv4_is_global(v4),
        IpAddr::V6(v6) => ipv6_is_global(v6),
    }
}

/// Global-address test for IPv4, rejecting all IANA special-purpose ranges that
/// are relevant to SSRF (loopback, private, link-local, CGN, benchmarking,
/// documentation, protocol-assignment, multicast, broadcast, reserved).
fn ipv4_is_global(ip: Ipv4Addr) -> bool {
    if ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_documentation()
        || ip.is_multicast()
    {
        return false;
    }
    let o = ip.octets();
    // Shared Address Space (CGN) 100.64.0.0/10
    if o[0] == 100 && (o[1] & 0xc0) == 64 {
        return false;
    }
    // IETF Protocol Assignments 192.0.0.0/24
    if o[0] == 192 && o[1] == 0 && o[2] == 0 {
        return false;
    }
    // Benchmarking 198.18.0.0/15
    if o[0] == 198 && (o[1] & 0xfe) == 18 {
        return false;
    }
    // Reserved (incl. 240.0.0.0/4)
    if o[0] >= 240 {
        return false;
    }
    true
}

/// Global-address test for IPv6, rejecting unspecified/loopback/multicast,
/// unique-local (`fc00::/7`), link-local (`fe80::/10`) and documentation
/// (`2001:db8::/32`), and validating any embedded IPv4 for mapped addresses.
fn ipv6_is_global(ip: Ipv6Addr) -> bool {
    if ip.is_unspecified() || ip.is_loopback() || ip.is_multicast() {
        return false;
    }
    // IPv4-mapped (::ffff:a.b.c.d) — validate the embedded IPv4 range.
    if let Some(v4) = ip.to_ipv4_mapped() {
        return ipv4_is_global(v4);
    }
    let seg = ip.segments();
    // Unique local fc00::/7
    if (seg[0] & 0xfe00) == 0xfc00 {
        return false;
    }
    // Link-local unicast fe80::/10
    if (seg[0] & 0xffc0) == 0xfe80 {
        return false;
    }
    // Documentation 2001:db8::/32
    if seg[0] == 0x2001 && seg[1] == 0x0db8 {
        return false;
    }
    true
}

#[cfg(test)]
mod ssrf_tests {
    use super::*;

    #[test]
    fn regression_blocks_loopback_and_private_and_metadata() {
        // Loopback
        assert!(!ip_is_global("127.0.0.1".parse().expect("ip")));
        assert!(!ip_is_global("::1".parse().expect("ip")));
        // Cloud metadata endpoint (link-local)
        assert!(!ip_is_global("169.254.169.254".parse().expect("ip")));
        // Private ranges
        assert!(!ip_is_global("10.0.0.5".parse().expect("ip")));
        assert!(!ip_is_global("172.16.3.4".parse().expect("ip")));
        assert!(!ip_is_global("192.168.1.1".parse().expect("ip")));
        // Unspecified
        assert!(!ip_is_global("0.0.0.0".parse().expect("ip")));
        // CGN shared space
        assert!(!ip_is_global("100.64.0.1".parse().expect("ip")));
        // Unique-local IPv6 and link-local IPv6
        assert!(!ip_is_global("fc00::1".parse().expect("ip")));
        assert!(!ip_is_global("fe80::1".parse().expect("ip")));
        // IPv4-mapped loopback
        assert!(!ip_is_global("::ffff:127.0.0.1".parse().expect("ip")));
    }

    #[test]
    fn regression_allows_global_addresses() {
        assert!(ip_is_global("8.8.8.8".parse().expect("ip")));
        assert!(ip_is_global("1.1.1.1".parse().expect("ip")));
        assert!(ip_is_global("2606:4700:4700::1111".parse().expect("ip")));
    }

    #[test]
    fn regression_resolve_and_validate_rejects_localhost() {
        // localhost must resolve to loopback and therefore be rejected.
        let result = resolve_and_validate_host("localhost", 80);
        assert!(
            result.is_err(),
            "localhost must be rejected as a LOAD target, got: {result:?}"
        );
    }
}
