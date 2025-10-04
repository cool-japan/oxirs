//! SPARQL query parsing utilities

use crate::Result;
use std::collections::HashMap;

/// Extract PREFIX declarations and expand prefixed names in SPARQL query
/// Handles both line-based and inline PREFIX declarations
pub fn extract_and_expand_prefixes(sparql: &str) -> Result<(HashMap<String, String>, String)> {
    let mut prefixes = HashMap::new();
    let mut remaining_query = sparql;

    // Extract all PREFIX declarations one by one
    // Pattern: PREFIX prefix: <uri>
    while let Some(prefix_pos) = remaining_query.to_uppercase().find("PREFIX") {
        // Find the PREFIX declaration boundaries
        // Look for: PREFIX name: <uri>
        let after_prefix = &remaining_query[prefix_pos + 6..]; // Skip "PREFIX"

        // Find colon
        if let Some(colon_pos) = after_prefix.find(':') {
            let prefix_name = after_prefix[..colon_pos].trim();

            // Find URI in angle brackets
            if let Some(uri_start) = after_prefix[colon_pos..].find('<') {
                if let Some(uri_end) = after_prefix[colon_pos + uri_start..].find('>') {
                    let uri_start_abs = colon_pos + uri_start;
                    let uri = after_prefix[uri_start_abs + 1..uri_start_abs + uri_end].trim();

                    prefixes.insert(prefix_name.to_string(), uri.to_string());

                    // Remove this PREFIX declaration and continue with the rest
                    let decl_end = prefix_pos + 6 + uri_start_abs + uri_end + 1;
                    remaining_query = &remaining_query[decl_end..];
                    continue;
                }
            }
        }

        // Couldn't parse this PREFIX - skip it and move on
        remaining_query = &remaining_query[prefix_pos + 6..];
    }

    // Clean up whitespace
    let query_without_prefixes = remaining_query
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    // If no prefixes found, return original
    if prefixes.is_empty() {
        return Ok((prefixes, sparql.to_string()));
    }

    // Expand all prefixed names (prefix:localName -> <uri#localName>)
    let mut expanded_query = query_without_prefixes.clone();

    for (prefix, uri) in &prefixes {
        let prefix_pattern = format!("{}:", prefix);

        // Find all occurrences of prefix:something
        let mut result = String::new();
        let mut last_end = 0;

        for (idx, _) in expanded_query.match_indices(&prefix_pattern) {
            // Check if this is actually a prefixed name (not inside a string or URI)
            let before = &expanded_query[..idx];
            let in_string = before.matches('"').count() % 2 == 1;
            let in_uri = before.matches('<').count() > before.matches('>').count();

            if !in_string && !in_uri {
                // Add the part before this match
                result.push_str(&expanded_query[last_end..idx]);

                // Find the local name (alphanumeric + underscore)
                let after_prefix = &expanded_query[idx + prefix_pattern.len()..];
                let local_name_len = after_prefix
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
                    .count();

                if local_name_len > 0 {
                    let local_name = &after_prefix[..local_name_len];

                    // Expand to full URI
                    let expanded = if uri.ends_with('#') || uri.ends_with('/') {
                        format!("<{}{}>", uri, local_name)
                    } else {
                        format!("<{}#{}>", uri, local_name)
                    };

                    result.push_str(&expanded);
                    last_end = idx + prefix_pattern.len() + local_name_len;
                } else {
                    result.push_str(&expanded_query[last_end..idx + prefix_pattern.len()]);
                    last_end = idx + prefix_pattern.len();
                }
            }
        }

        // Add the remaining part
        result.push_str(&expanded_query[last_end..]);
        expanded_query = result;
    }

    Ok((prefixes, expanded_query))
}

/// Extract variable names from SELECT clause
pub fn extract_select_variables(sparql: &str) -> Result<Vec<String>> {
    let mut variables = Vec::new();

    if let Some(select_start) = sparql.to_uppercase().find("SELECT") {
        if let Some(where_start) = sparql.to_uppercase().find("WHERE") {
            let select_clause = &sparql[select_start + 6..where_start];

            for token in select_clause.split_whitespace() {
                // Skip DISTINCT keyword
                if token.to_uppercase() == "DISTINCT" {
                    continue;
                }

                if token == "*" {
                    // Mark as SELECT * - will be expanded to pattern variables later
                    variables.push("*".to_string());
                } else if let Some(var_name) = token.strip_prefix('?') {
                    variables.push(var_name.to_string());
                }
            }
        }
    }

    Ok(variables)
}
