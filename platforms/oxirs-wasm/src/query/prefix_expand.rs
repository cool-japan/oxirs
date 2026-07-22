//! SPARQL prologue handling: `PREFIX` and `BASE`.
//!
//! The query parser in [`crate::query`] works on absolute IRIs written in
//! angle brackets. Real queries — everything YASGUI, a tutorial or a stored
//! query hands us — instead declare a prologue and then use prefixed names:
//!
//! ```sparql
//! PREFIX oxa: <https://cooljapan.tech/ns/oxiephemeris/astro#>
//! SELECT ?sign WHERE { ?sign oxa:element ?e }
//! ```
//!
//! [`expand_prologue`] rewrites such a query into the fully-qualified form the
//! parser understands: the prologue is consumed, and every prefixed name in
//! the body becomes `<...>`. Text that must not be touched — string literals,
//! IRIs that are already absolute, comments — is copied verbatim, so a colon
//! inside `"a: b"` or inside `<http://…>` never turns into an expansion.

use crate::error::{WasmError, WasmResult};
use std::collections::HashMap;
use url::Url;

/// Expand the `PREFIX`/`BASE` prologue of a SPARQL query.
///
/// Returns the query with the prologue removed and every prefixed name
/// replaced by an absolute `<iri>`. A query without a prologue is returned
/// with its prefixed names untouched (there is nothing to expand them with),
/// which keeps fully-qualified queries byte-identical.
///
/// # Errors
///
/// [`WasmError::QueryError`] if a declaration is malformed (`PREFIX` without a
/// name or without an IRI, `BASE` without an IRI) or if the body uses a prefix
/// that was never declared.
pub fn expand_prologue(sparql: &str) -> WasmResult<String> {
    let chars: Vec<char> = sparql.chars().collect();
    let (prefixes, base, body_start) = parse_prologue(&chars)?;
    expand_body(&chars[body_start..], &prefixes, base.as_deref())
}

/// Scan the prologue, returning the declared prefixes, the `BASE` IRI and the
/// offset at which the query form (`SELECT`, `ASK`, …) starts.
fn parse_prologue(chars: &[char]) -> WasmResult<(HashMap<String, String>, Option<String>, usize)> {
    let mut prefixes: HashMap<String, String> = HashMap::new();
    let mut base: Option<String> = None;
    let mut i = 0;

    loop {
        i = skip_ignorable(chars, i);
        match keyword_at(chars, i) {
            Some(Keyword::Prefix) => {
                let mut j = skip_ignorable(chars, i + 6);
                let name_start = j;
                while j < chars.len() && chars[j] != ':' && !chars[j].is_whitespace() {
                    j += 1;
                }
                if j >= chars.len() || chars[j] != ':' {
                    return Err(WasmError::QueryError(
                        "PREFIX declaration is missing its ':'".to_string(),
                    ));
                }
                let name: String = chars[name_start..j].iter().collect();
                j = skip_ignorable(chars, j + 1);
                let (iri, next) = read_iri(chars, j)?;
                prefixes.insert(name, iri);
                i = next;
            }
            Some(Keyword::Base) => {
                let j = skip_ignorable(chars, i + 4);
                let (iri, next) = read_iri(chars, j)?;
                base = Some(iri);
                i = next;
            }
            None => return Ok((prefixes, base, i)),
        }
    }
}

/// A prologue keyword.
enum Keyword {
    Prefix,
    Base,
}

/// Recognise `PREFIX` / `BASE` at `i`, case-insensitively, as a whole word.
fn keyword_at(chars: &[char], i: usize) -> Option<Keyword> {
    let word_ends_at = |len: usize| match chars.get(i + len) {
        Some(c) => c.is_whitespace() || *c == '<' || *c == ':',
        None => false,
    };
    if matches_ascii_ignore_case(chars, i, "PREFIX") && word_ends_at(6) {
        Some(Keyword::Prefix)
    } else if matches_ascii_ignore_case(chars, i, "BASE") && word_ends_at(4) {
        Some(Keyword::Base)
    } else {
        None
    }
}

/// Compare `word` against `chars[i..]`, ASCII case-insensitively.
fn matches_ascii_ignore_case(chars: &[char], i: usize, word: &str) -> bool {
    let word_chars: Vec<char> = word.chars().collect();
    if i + word_chars.len() > chars.len() {
        return false;
    }
    chars[i..i + word_chars.len()]
        .iter()
        .zip(word_chars.iter())
        .all(|(a, b)| a.eq_ignore_ascii_case(b))
}

/// Read an `<iri>` at `i`, returning the IRI body and the offset just past `>`.
fn read_iri(chars: &[char], i: usize) -> WasmResult<(String, usize)> {
    if chars.get(i) != Some(&'<') {
        return Err(WasmError::QueryError(
            "prologue declaration is missing its <iri>".to_string(),
        ));
    }
    let mut j = i + 1;
    let mut iri = String::new();
    while j < chars.len() && chars[j] != '>' {
        iri.push(chars[j]);
        j += 1;
    }
    if j >= chars.len() {
        return Err(WasmError::QueryError(
            "prologue declaration has an unterminated <iri>".to_string(),
        ));
    }
    Ok((iri, j + 1))
}

/// Try to read an `<iri>` at `i`. Returns `None` when the `<` cannot open an
/// IRIREF — i.e. when it is the less-than operator, as in `FILTER(?a < 28)` —
/// because an IRIREF may not contain whitespace or any of `<>"{}|^\``.
fn try_read_iri(chars: &[char], i: usize) -> Option<(String, usize)> {
    if chars.get(i) != Some(&'<') {
        return None;
    }
    let mut j = i + 1;
    let mut iri = String::new();
    while j < chars.len() {
        let c = chars[j];
        if c == '>' {
            return Some((iri, j + 1));
        }
        if c.is_whitespace() || matches!(c, '<' | '"' | '{' | '}' | '|' | '^' | '`' | '\\') {
            return None;
        }
        iri.push(c);
        j += 1;
    }
    None
}

/// Skip whitespace and `#` comments.
fn skip_ignorable(chars: &[char], mut i: usize) -> usize {
    loop {
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        if i < chars.len() && chars[i] == '#' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else {
            return i;
        }
    }
}

/// Rewrite the query body, expanding prefixed names into absolute IRIs.
fn expand_body(
    chars: &[char],
    prefixes: &HashMap<String, String>,
    base: Option<&str>,
) -> WasmResult<String> {
    let mut out = String::with_capacity(chars.len());
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        match c {
            // Comment: copy to end of line.
            '#' => {
                while i < chars.len() && chars[i] != '\n' {
                    out.push(chars[i]);
                    i += 1;
                }
            }
            // String literal: copy verbatim, honouring backslash escapes.
            '"' | '\'' => {
                out.push(c);
                i += 1;
                while i < chars.len() {
                    let s = chars[i];
                    out.push(s);
                    i += 1;
                    if s == '\\' && i < chars.len() {
                        out.push(chars[i]);
                        i += 1;
                    } else if s == c {
                        break;
                    }
                }
            }
            // `<` is an IRI here only if it closes as one — otherwise it is the
            // less-than operator of a FILTER expression.
            '<' => match try_read_iri(chars, i) {
                Some((iri, next)) => {
                    let resolved = resolve_against_base(&iri, base)?;
                    out.push_str(&format!("<{}>", resolved));
                    i = next;
                }
                None => {
                    out.push(c);
                    i += 1;
                }
            },
            // A prefixed name starts with a name character or the default `:`.
            _ if is_pname_start(c) || c == ':' => {
                let (token, next) = read_pname_candidate(chars, i);
                i = next;
                match split_pname(&token) {
                    Some((prefix, local)) => match prefixes.get(prefix) {
                        Some(namespace) => out.push_str(&format!("<{}{}>", namespace, local)),
                        None => {
                            return Err(WasmError::QueryError(format!(
                                "undefined prefix '{}:' — declare it with PREFIX",
                                prefix
                            )))
                        }
                    },
                    // A bare word: a keyword, a variable name, `a`, a number.
                    None => out.push_str(&token),
                }
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }

    Ok(out)
}

/// Resolve a possibly relative IRI reference against `BASE`, following RFC
/// 3986 §5.3 ("Component Recomposition") rather than naive string
/// concatenation. This correctly handles path-absolute references (`/x`
/// replaces the entire path), network-path references (`//host/x` replaces
/// the authority), and dot-segment removal (`../y`, `./y`) — all of which a
/// plain `format!("{base}{iri}")` gets wrong.
///
/// # Errors
///
/// Returns [`WasmError::QueryError`] if `base` is not itself a valid absolute
/// IRI (so it cannot serve as a resolution base at all), or if the relative
/// reference cannot be resolved against it.
fn resolve_against_base(iri: &str, base: Option<&str>) -> WasmResult<String> {
    match base {
        Some(base) if !is_absolute_iri(iri) => {
            let base_url = Url::parse(base)
                .map_err(|e| WasmError::QueryError(format!("invalid BASE <{base}>: {e}")))?;
            let resolved = base_url.join(iri).map_err(|e| {
                WasmError::QueryError(format!(
                    "cannot resolve relative IRI <{iri}> against BASE <{base}>: {e}"
                ))
            })?;
            Ok(resolved.to_string())
        }
        _ => Ok(iri.to_string()),
    }
}

/// An IRI is absolute if it carries a scheme (`scheme:` before any `/`).
fn is_absolute_iri(iri: &str) -> bool {
    match iri.find(':') {
        Some(colon) => !iri[..colon].contains('/'),
        None => false,
    }
}

/// Characters that may start a prefixed name (or a bare keyword).
fn is_pname_start(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Characters allowed inside the prefix or local part of a prefixed name.
fn is_pname_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | '%' | ':')
}

/// Read a maximal name-like token starting at `i` (which may or may not turn
/// out to be a prefixed name).
fn read_pname_candidate(chars: &[char], i: usize) -> (String, usize) {
    let mut j = i;
    while j < chars.len() && is_pname_char(chars[j]) {
        j += 1;
    }
    // A trailing '.' is statement punctuation, not part of the name.
    while j > i && chars[j - 1] == '.' {
        j -= 1;
    }
    (chars[i..j].iter().collect(), j)
}

/// Split `pfx:local` into its parts. Returns `None` for a bare word (no colon).
fn split_pname(token: &str) -> Option<(&str, &str)> {
    let colon = token.find(':')?;
    Some((&token[..colon], &token[colon + 1..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const OXA: &str = "https://cooljapan.tech/ns/oxiephemeris/astro#";

    #[test]
    fn expands_a_prefixed_name() {
        let q = format!("PREFIX oxa: <{OXA}>\nSELECT ?s WHERE {{ ?s oxa:element ?e }}");
        let out = expand_prologue(&q).expect("expand");
        assert_eq!(
            out.trim(),
            format!("SELECT ?s WHERE {{ ?s <{OXA}element> ?e }}")
        );
    }

    #[test]
    fn expands_the_default_prefix() {
        let q = format!("PREFIX : <{OXA}>\nASK {{ ?s :element ?e }}");
        let out = expand_prologue(&q).expect("expand");
        assert!(out.contains(&format!("<{OXA}element>")), "{out}");
    }

    #[test]
    fn leaves_absolute_iris_and_variables_alone() {
        let q = "SELECT ?s WHERE { ?s <http://ex/p> ?o }";
        assert_eq!(expand_prologue(q).expect("expand"), q);
    }

    #[test]
    fn never_expands_inside_a_literal() {
        let q =
            format!("PREFIX oxa: <{OXA}>\nSELECT ?s WHERE {{ ?s oxa:label \"oxa:not-a-name\" }}");
        let out = expand_prologue(&q).expect("expand");
        assert!(out.contains("\"oxa:not-a-name\""), "{out}");
        assert!(out.contains(&format!("<{OXA}label>")), "{out}");
    }

    #[test]
    fn expands_a_datatype_suffix() {
        let q = concat!(
            "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n",
            "ASK { ?s ?p \"2\"^^xsd:integer }"
        );
        let out = expand_prologue(q).expect("expand");
        assert!(
            out.contains("\"2\"^^<http://www.w3.org/2001/XMLSchema#integer>"),
            "{out}"
        );
    }

    #[test]
    fn keeps_the_type_shorthand_and_the_trailing_dot() {
        let q = format!("PREFIX oxa: <{OXA}>\nSELECT ?s WHERE {{ ?s a oxa:ZodiacSign . }}");
        let out = expand_prologue(&q).expect("expand");
        assert!(out.contains(" a "), "{out}");
        assert!(out.contains(&format!("<{OXA}ZodiacSign> .")), "{out}");
    }

    #[test]
    fn resolves_a_relative_iri_against_base() {
        let q = "BASE <http://ex/>\nSELECT ?o WHERE { <s> <p> ?o }";
        let out = expand_prologue(q).expect("expand");
        assert!(out.contains("<http://ex/s> <http://ex/p>"), "{out}");
    }

    #[test]
    fn regression_resolves_path_absolute_reference_by_replacing_whole_path() {
        // RFC 3986 §5.3: a reference starting with '/' replaces the base's
        // entire path, it does not get appended after it.
        let q = "BASE <http://ex/a/b/>\nSELECT ?o WHERE { </x> <p> ?o }";
        let out = expand_prologue(q).expect("expand");
        assert!(out.contains("<http://ex/x>"), "{out}");
        assert!(!out.contains("http://ex/a/b//x"), "{out}");
    }

    #[test]
    fn regression_resolves_dot_dot_segment() {
        // RFC 3986 §5.3: dot-segments must be removed, not left embedded.
        let q = "BASE <http://ex/a/b/>\nSELECT ?o WHERE { <../y> <p> ?o }";
        let out = expand_prologue(q).expect("expand");
        assert!(out.contains("<http://ex/a/y>"), "{out}");
        assert!(!out.contains(".."), "{out}");
    }

    #[test]
    fn regression_resolves_network_path_reference_by_replacing_authority() {
        // RFC 3986 §5.3: a reference starting with '//' replaces the base's
        // authority (and scheme carries over), not a literal concatenation.
        let q = "BASE <http://ex.org/a/>\nSELECT ?o WHERE { <//other.org/x> <p> ?o }";
        let out = expand_prologue(q).expect("expand");
        assert!(out.contains("<http://other.org/x>"), "{out}");
    }

    #[test]
    fn regression_invalid_base_iri_fails_loud_instead_of_silently_concatenating() {
        // A BASE that cannot serve as an RFC 3986 resolution base (no scheme)
        // must be a reported error, not silently glued onto the reference.
        let q = "BASE <not-a-valid-base>\nSELECT ?o WHERE { <s> <p> ?o }";
        let err = expand_prologue(q).expect_err("malformed BASE must fail");
        assert!(err.to_string().contains("BASE"), "{err}");
    }

    #[test]
    fn keeps_a_less_than_operator_out_of_iri_parsing() {
        let q = "SELECT ?s WHERE { ?s <http://ex/age> ?a . FILTER(?a > 18 && ?a < 28) }";
        assert_eq!(expand_prologue(q).expect("expand"), q);
    }

    #[test]
    fn rejects_an_undeclared_prefix() {
        let err = expand_prologue("SELECT ?s WHERE { ?s oxa:element ?e }")
            .expect_err("undeclared prefix must fail");
        assert!(err.to_string().contains("oxa"), "{err}");
    }

    #[test]
    fn skips_comments_in_the_prologue() {
        let q = format!("# a note\nPREFIX oxa: <{OXA}>\n# another\nASK {{ ?s oxa:element ?o }}");
        let out = expand_prologue(&q).expect("expand");
        assert!(out.contains(&format!("<{OXA}element>")), "{out}");
    }
}
