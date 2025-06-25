//! IRI Validator and Processor
//!
//! Validates, normalizes, and processes Internationalized Resource Identifiers (IRIs).

use super::{utils, ToolResult};

/// Run IRI validation and processing
pub async fn run(
    iri: String,
    resolve: Option<String>,
    validate: bool,
    normalize: bool,
) -> ToolResult {
    println!("IRI Processor");
    println!("Input IRI: {}", iri);

    if validate {
        match utils::validate_iri(&iri) {
            Ok(valid_iri) => {
                println!("✓ IRI is valid");
                if valid_iri != iri {
                    println!("  Normalized: {}", valid_iri);
                }
            }
            Err(e) => {
                println!("✗ IRI is invalid: {}", e);
                return Err(format!("Invalid IRI: {}", e).into());
            }
        }
    }

    if normalize {
        let normalized = normalize_iri(&iri)?;
        println!("Normalized IRI: {}", normalized);
    }

    if let Some(base) = resolve {
        let resolved = resolve_iri(&iri, &base)?;
        println!("Resolved IRI: {}", resolved);
    }

    // Additional IRI analysis
    analyze_iri(&iri)?;

    Ok(())
}

/// Normalize an IRI
fn normalize_iri(iri: &str) -> ToolResult<String> {
    let mut normalized = iri.trim().to_string();

    // Basic normalization steps
    // 1. Remove unnecessary percent-encoding
    normalized = decode_unnecessary_percent_encoding(&normalized);

    // 2. Normalize case of scheme and host
    normalized = normalize_scheme_and_host(&normalized)?;

    // 3. Remove default port numbers
    normalized = remove_default_port(&normalized);

    // 4. Normalize path
    normalized = normalize_path(&normalized);

    Ok(normalized)
}

/// Decode unnecessary percent-encoding
fn decode_unnecessary_percent_encoding(iri: &str) -> String {
    // This is a simplified implementation
    // In practice, you'd need to carefully decode only unreserved characters
    let mut result = String::new();
    let mut chars = iri.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '%' {
            // Check if next two characters form a valid hex sequence
            let hex1 = chars.peek().copied();
            if let Some(h1) = hex1 {
                chars.next();
                let hex2 = chars.peek().copied();
                if let Some(h2) = hex2 {
                    chars.next();
                    let hex_str = format!("{}{}", h1, h2);
                    if let Ok(byte) = u8::from_str_radix(&hex_str, 16) {
                        let decoded_char = byte as char;
                        // Only decode unreserved characters
                        if decoded_char.is_ascii_alphanumeric()
                            || matches!(decoded_char, '-' | '.' | '_' | '~')
                        {
                            result.push(decoded_char);
                        } else {
                            result.push('%');
                            result.push(h1);
                            result.push(h2);
                        }
                    } else {
                        result.push('%');
                        result.push(h1);
                        result.push(h2);
                    }
                } else {
                    result.push('%');
                    result.push(h1);
                }
            } else {
                result.push('%');
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Normalize scheme and host case
fn normalize_scheme_and_host(iri: &str) -> ToolResult<String> {
    if let Some(scheme_end) = iri.find(':') {
        let scheme = &iri[..scheme_end].to_lowercase();
        let rest = &iri[scheme_end..];

        if rest.starts_with("://") {
            // Has authority component
            if let Some(authority_end) = rest[3..].find('/') {
                let authority = &rest[3..authority_end + 3].to_lowercase();
                let path_and_rest = &rest[authority_end + 3..];
                Ok(format!(
                    "{}:{}//{}{}",
                    scheme,
                    &rest[..3],
                    authority,
                    path_and_rest
                ))
            } else if let Some(query_start) = rest[3..].find('?') {
                let authority = &rest[3..query_start + 3].to_lowercase();
                let query_and_rest = &rest[query_start + 3..];
                Ok(format!(
                    "{}:{}//{}?{}",
                    scheme,
                    &rest[..3],
                    authority,
                    query_and_rest
                ))
            } else if let Some(fragment_start) = rest[3..].find('#') {
                let authority = &rest[3..fragment_start + 3].to_lowercase();
                let fragment = &rest[fragment_start + 3..];
                Ok(format!(
                    "{}:{}//{}#{}",
                    scheme,
                    &rest[..3],
                    authority,
                    fragment
                ))
            } else {
                let authority = &rest[3..].to_lowercase();
                Ok(format!("{}:{}//{}", scheme, &rest[..3], authority))
            }
        } else {
            Ok(format!("{}{}", scheme, rest))
        }
    } else {
        Ok(iri.to_string())
    }
}

/// Remove default port numbers
fn remove_default_port(iri: &str) -> String {
    // Common default ports
    let default_ports = [("http://", ":80"), ("https://", ":443"), ("ftp://", ":21")];

    for (scheme, default_port) in &default_ports {
        if iri.starts_with(scheme) {
            let after_scheme = &iri[scheme.len()..];
            if let Some(port_pos) = after_scheme.find(default_port) {
                // Check if this is actually the port (not part of path)
                let before_port = &after_scheme[..port_pos];
                if !before_port.contains('/')
                    && !before_port.contains('?')
                    && !before_port.contains('#')
                {
                    let after_port = &after_scheme[port_pos + default_port.len()..];
                    if after_port.is_empty()
                        || after_port.starts_with('/')
                        || after_port.starts_with('?')
                        || after_port.starts_with('#')
                    {
                        return format!("{}{}{}", scheme, before_port, after_port);
                    }
                }
            }
        }
    }

    iri.to_string()
}

/// Normalize path component
fn normalize_path(iri: &str) -> String {
    // Find the path component
    if let Some(scheme_end) = iri.find(':') {
        let scheme_and_colon = &iri[..scheme_end + 1];
        let rest = &iri[scheme_end + 1..];

        if rest.starts_with("//") {
            // Has authority
            if let Some(path_start) = rest[2..].find('/') {
                let authority = &rest[..path_start + 2];
                let path_and_rest = &rest[path_start + 2..];

                // Find end of path
                let path_end = path_and_rest
                    .find('?')
                    .or_else(|| path_and_rest.find('#'))
                    .unwrap_or(path_and_rest.len());

                let path = &path_and_rest[..path_end];
                let query_and_fragment = &path_and_rest[path_end..];

                let normalized_path = normalize_path_segments(path);
                return format!(
                    "{}{}{}{}",
                    scheme_and_colon, authority, normalized_path, query_and_fragment
                );
            }
        } else {
            // No authority, rest is path
            let path_end = rest
                .find('?')
                .or_else(|| rest.find('#'))
                .unwrap_or(rest.len());

            let path = &rest[..path_end];
            let query_and_fragment = &rest[path_end..];

            let normalized_path = normalize_path_segments(path);
            return format!(
                "{}{}{}",
                scheme_and_colon, normalized_path, query_and_fragment
            );
        }
    }

    iri.to_string()
}

/// Normalize path segments by resolving . and .. components
fn normalize_path_segments(path: &str) -> String {
    if path.is_empty() {
        return path.to_string();
    }

    let segments: Vec<&str> = path.split('/').collect();
    let mut normalized_segments = Vec::new();

    for segment in segments {
        match segment {
            "." => {
                // Skip current directory references
                continue;
            }
            ".." => {
                // Go up one directory
                normalized_segments.pop();
            }
            _ => {
                normalized_segments.push(segment);
            }
        }
    }

    // Reconstruct path
    let result = normalized_segments.join("/");

    // Preserve leading/trailing slashes
    let mut final_result = String::new();
    if path.starts_with('/') && !result.starts_with('/') {
        final_result.push('/');
    }
    final_result.push_str(&result);
    if path.ends_with('/') && !result.ends_with('/') && !result.is_empty() {
        final_result.push('/');
    }

    final_result
}

/// Resolve a relative IRI against a base IRI
fn resolve_iri(iri: &str, base: &str) -> ToolResult<String> {
    // Basic IRI resolution (simplified implementation of RFC 3986)

    // If IRI is already absolute, return as-is
    if iri.contains(':') {
        return Ok(iri.to_string());
    }

    // Parse base IRI components
    let base_parts = parse_iri_components(base)?;

    if iri.is_empty() {
        return Ok(base.to_string());
    }

    if iri.starts_with("//") {
        // Authority-only relative reference
        return Ok(format!("{}:{}", base_parts.scheme, iri));
    }

    if iri.starts_with('/') {
        // Absolute path reference
        return Ok(format!(
            "{}://{}{}",
            base_parts.scheme, base_parts.authority, iri
        ));
    }

    if iri.starts_with('?') {
        // Query-only reference
        let base_without_query = if let Some(query_pos) = base.find('?') {
            &base[..query_pos]
        } else {
            base
        };
        return Ok(format!("{}{}", base_without_query, iri));
    }

    if iri.starts_with('#') {
        // Fragment-only reference
        let base_without_fragment = if let Some(fragment_pos) = base.find('#') {
            &base[..fragment_pos]
        } else {
            base
        };
        return Ok(format!("{}{}", base_without_fragment, iri));
    }

    // Relative path reference
    let base_path = base_parts.path.unwrap_or("/".to_string());
    let base_dir = if let Some(last_slash) = base_path.rfind('/') {
        &base_path[..last_slash + 1]
    } else {
        "/"
    };

    let combined_path = format!("{}{}", base_dir, iri);
    let normalized_path = normalize_path_segments(&combined_path);

    Ok(format!(
        "{}://{}{}",
        base_parts.scheme, base_parts.authority, normalized_path
    ))
}

/// IRI components
struct IriComponents {
    scheme: String,
    authority: String,
    path: Option<String>,
}

/// Parse IRI into components (simplified)
fn parse_iri_components(iri: &str) -> ToolResult<IriComponents> {
    let scheme_end = iri.find(':').ok_or("IRI must contain a scheme")?;

    let scheme = iri[..scheme_end].to_string();
    let rest = &iri[scheme_end + 1..];

    if rest.starts_with("//") {
        // Has authority
        let authority_start = 2;
        let authority_end = rest[authority_start..]
            .find('/')
            .map(|pos| pos + authority_start)
            .or_else(|| {
                rest[authority_start..]
                    .find('?')
                    .map(|pos| pos + authority_start)
            })
            .or_else(|| {
                rest[authority_start..]
                    .find('#')
                    .map(|pos| pos + authority_start)
            })
            .unwrap_or(rest.len());

        let authority = rest[authority_start..authority_end].to_string();
        let path = if authority_end < rest.len() && rest.chars().nth(authority_end) == Some('/') {
            Some(rest[authority_end..].to_string())
        } else {
            None
        };

        Ok(IriComponents {
            scheme,
            authority,
            path,
        })
    } else {
        Err("IRI authority parsing not implemented for this format".into())
    }
}

/// Analyze and display IRI components
fn analyze_iri(iri: &str) -> ToolResult<()> {
    println!("\nIRI Analysis:");
    println!("=============");

    // Basic component extraction
    if let Some(scheme_end) = iri.find(':') {
        let scheme = &iri[..scheme_end];
        println!("Scheme: {}", scheme);

        let rest = &iri[scheme_end + 1..];

        if rest.starts_with("//") {
            // Has authority component
            let authority_part = &rest[2..];

            // Find end of authority
            let authority_end = authority_part
                .find('/')
                .or_else(|| authority_part.find('?'))
                .or_else(|| authority_part.find('#'))
                .unwrap_or(authority_part.len());

            let authority = &authority_part[..authority_end];
            println!("Authority: {}", authority);

            // Parse authority components
            if let Some(at_pos) = authority.find('@') {
                let userinfo = &authority[..at_pos];
                let host_and_port = &authority[at_pos + 1..];
                println!("  Userinfo: {}", userinfo);

                if let Some(colon_pos) = host_and_port.rfind(':') {
                    let host = &host_and_port[..colon_pos];
                    let port = &host_and_port[colon_pos + 1..];
                    println!("  Host: {}", host);
                    println!("  Port: {}", port);
                } else {
                    println!("  Host: {}", host_and_port);
                }
            } else if let Some(colon_pos) = authority.rfind(':') {
                // Check if this is actually a port (not IPv6)
                let potential_port = &authority[colon_pos + 1..];
                if potential_port.chars().all(|c| c.is_ascii_digit()) {
                    let host = &authority[..colon_pos];
                    println!("  Host: {}", host);
                    println!("  Port: {}", potential_port);
                } else {
                    println!("  Host: {}", authority);
                }
            } else {
                println!("  Host: {}", authority);
            }

            // Path component
            if authority_end < authority_part.len() {
                let path_and_rest = &authority_part[authority_end..];

                let path_end = path_and_rest
                    .find('?')
                    .or_else(|| path_and_rest.find('#'))
                    .unwrap_or(path_and_rest.len());

                if path_end > 0 {
                    let path = &path_and_rest[..path_end];
                    println!("Path: {}", path);
                }

                // Query component
                if let Some(query_start) = path_and_rest.find('?') {
                    let query_end = path_and_rest[query_start + 1..]
                        .find('#')
                        .map(|pos| pos + query_start + 1)
                        .unwrap_or(path_and_rest.len());

                    let query = &path_and_rest[query_start + 1..query_end];
                    println!("Query: {}", query);
                }

                // Fragment component
                if let Some(fragment_start) = path_and_rest.find('#') {
                    let fragment = &path_and_rest[fragment_start + 1..];
                    println!("Fragment: {}", fragment);
                }
            }
        } else {
            // No authority, rest is path
            let path_end = rest
                .find('?')
                .or_else(|| rest.find('#'))
                .unwrap_or(rest.len());

            if path_end > 0 {
                let path = &rest[..path_end];
                println!("Path: {}", path);
            }

            // Query and fragment handling similar to above
            if let Some(query_start) = rest.find('?') {
                let query_end = rest[query_start + 1..]
                    .find('#')
                    .map(|pos| pos + query_start + 1)
                    .unwrap_or(rest.len());

                let query = &rest[query_start + 1..query_end];
                println!("Query: {}", query);
            }

            if let Some(fragment_start) = rest.find('#') {
                let fragment = &rest[fragment_start + 1..];
                println!("Fragment: {}", fragment);
            }
        }

        // Additional analysis
        println!("\nIRI Properties:");
        println!("Length: {} characters", iri.len());
        println!("Contains non-ASCII: {}", !iri.is_ascii());
        println!("Contains percent-encoding: {}", iri.contains('%'));

        // Scheme-specific validation
        match scheme.to_lowercase().as_str() {
            "http" | "https" => {
                println!("Type: HTTP(S) URL");
            }
            "ftp" => {
                println!("Type: FTP URL");
            }
            "mailto" => {
                println!("Type: Email address");
            }
            "urn" => {
                println!("Type: URN (Uniform Resource Name)");
            }
            _ => {
                println!("Type: Other/Unknown scheme");
            }
        }
    } else {
        return Err("Invalid IRI: no scheme found".into());
    }

    Ok(())
}
