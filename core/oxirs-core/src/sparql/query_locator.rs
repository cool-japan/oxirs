//! Locating the SPARQL `GroupGraphPattern` (the WHERE clause) inside a query string.
//!
//! Per the SPARQL 1.1 grammar, `WhereClause ::= 'WHERE'? GroupGraphPattern`, the
//! `WHERE` keyword is **optional**. Legal queries frequently omit it, e.g.
//! `ASK { ?s ?p ?o }`, `SELECT * { ?s ?p ?o }`, `SELECT ?x { ?x a <T> }`, or
//! `CONSTRUCT { ... } { ... }`.
//!
//! The simplified executor historically located the pattern block by a naive
//! `sparql.to_uppercase().find("WHERE")` substring search. That approach both
//! failed on the omitted-`WHERE` forms (extracting zero patterns) and could
//! match the substring `WHERE` inside an IRI (e.g. `.../somewhere#...`). This
//! module provides scanner-based helpers that:
//!
//! 1. locate the `WHERE` keyword only when it appears as a stand-alone,
//!    top-level word (never inside an IRI `<...>`, a string literal, or a `#`
//!    line comment), and
//! 2. fall back to the first top-level `{ ... }` group when `WHERE` is omitted,
//!    honoring the fact that for a `CONSTRUCT` query the first group is the
//!    template and the graph pattern is the group that follows it.
//!
//! All offsets returned are **byte** indices into the original `sparql` string
//! and always fall on `char` boundaries (the markers scanned — `{`, `}`, `<`,
//! `>`, `"`, `'`, `\`, `#`, the `\n` that ends a comment, and the ASCII keyword
//! bytes — are all single-byte ASCII, and UTF-8 continuation bytes never collide
//! with them).

/// The top-level result form of a SPARQL query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryForm {
    /// `SELECT ...`
    Select,
    /// `ASK ...`
    Ask,
    /// `CONSTRUCT ...`
    Construct,
    /// `DESCRIBE ...`
    Describe,
    /// Query form could not be determined.
    Unknown,
}

/// Return `true` if `b` is an ASCII identifier byte (used for word-boundary
/// checks around keywords).
#[inline]
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Find the byte offset of `keyword` occurring as a stand-alone, top-level word.
///
/// The match is ASCII case-insensitive. Occurrences inside an IRI (`<...>`), a
/// string literal (`"..."` / `'...'`), or a `#` line comment are ignored, as are
/// occurrences that are adjacent to other identifier characters (so `WHERE`
/// never matches inside `somewhere`). Returns the byte index of the first
/// qualifying match.
pub fn find_keyword(sparql: &str, keyword: &str) -> Option<usize> {
    let bytes = sparql.as_bytes();
    let kw = keyword.as_bytes();
    let klen = kw.len();
    if klen == 0 {
        return None;
    }
    let n = bytes.len();

    let mut in_string: Option<u8> = None;
    let mut in_iri = false;
    let mut in_comment = false;
    let mut escaped = false;
    let mut i = 0usize;

    while i < n {
        let b = bytes[i];

        if let Some(quote) = in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == quote {
                in_string = None;
            }
            i += 1;
            continue;
        }

        if in_iri {
            if b == b'>' {
                in_iri = false;
            }
            i += 1;
            continue;
        }

        if in_comment {
            if b == b'\n' {
                in_comment = false;
            }
            i += 1;
            continue;
        }

        match b {
            b'"' | b'\'' => {
                in_string = Some(b);
                i += 1;
                continue;
            }
            b'<' => {
                in_iri = true;
                i += 1;
                continue;
            }
            b'#' => {
                in_comment = true;
                i += 1;
                continue;
            }
            _ => {}
        }

        if i + klen <= n && bytes[i..i + klen].eq_ignore_ascii_case(kw) {
            let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after_idx = i + klen;
            let after_ok = after_idx >= n || !is_ident_byte(bytes[after_idx]);
            if before_ok && after_ok {
                return Some(i);
            }
        }

        i += 1;
    }

    None
}

/// Find the byte index of the next top-level `{` at or after `from`, skipping
/// any `{` that appears inside an IRI (`<...>`), a string literal, or a `#` line
/// comment.
pub fn next_group_open_brace(sparql: &str, from: usize) -> Option<usize> {
    let bytes = sparql.as_bytes();
    let n = bytes.len();
    let mut i = from.min(n);

    let mut in_string: Option<u8> = None;
    let mut in_iri = false;
    let mut in_comment = false;
    let mut escaped = false;

    while i < n {
        let b = bytes[i];

        if let Some(quote) = in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == quote {
                in_string = None;
            }
            i += 1;
            continue;
        }

        if in_iri {
            if b == b'>' {
                in_iri = false;
            }
            i += 1;
            continue;
        }

        if in_comment {
            if b == b'\n' {
                in_comment = false;
            }
            i += 1;
            continue;
        }

        match b {
            b'"' | b'\'' => in_string = Some(b),
            b'<' => in_iri = true,
            b'#' => in_comment = true,
            b'{' => return Some(i),
            _ => {}
        }

        i += 1;
    }

    None
}

/// Find the byte index of the `}` that matches the `{` located at byte index
/// `open`, tracking nested braces and skipping braces inside IRIs, string
/// literals, and `#` line comments. Returns `None` if `open` does not point at a
/// `{` or the group is unbalanced.
pub fn matching_close_brace(sparql: &str, open: usize) -> Option<usize> {
    let bytes = sparql.as_bytes();
    let n = bytes.len();
    if open >= n || bytes[open] != b'{' {
        return None;
    }

    let mut depth: i32 = 0;
    let mut in_string: Option<u8> = None;
    let mut in_iri = false;
    let mut in_comment = false;
    let mut escaped = false;
    let mut i = open;

    while i < n {
        let b = bytes[i];

        if let Some(quote) = in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == quote {
                in_string = None;
            }
            i += 1;
            continue;
        }

        if in_iri {
            if b == b'>' {
                in_iri = false;
            }
            i += 1;
            continue;
        }

        if in_comment {
            if b == b'\n' {
                in_comment = false;
            }
            i += 1;
            continue;
        }

        match b {
            b'"' | b'\'' => in_string = Some(b),
            b'<' => in_iri = true,
            b'#' => in_comment = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }

        i += 1;
    }

    None
}

/// Detect the top-level query form by picking the earliest top-level form
/// keyword (`SELECT` / `ASK` / `CONSTRUCT` / `DESCRIBE`). This naturally ignores
/// `PREFIX`/`BASE` prologue and keywords that appear inside IRIs, literals, or
/// nested sub-queries (which necessarily start later than the outer form).
pub fn detect_query_form(sparql: &str) -> QueryForm {
    let candidates = [
        (QueryForm::Select, "SELECT"),
        (QueryForm::Ask, "ASK"),
        (QueryForm::Construct, "CONSTRUCT"),
        (QueryForm::Describe, "DESCRIBE"),
    ];

    let mut best: Option<(usize, QueryForm)> = None;
    for (form, kw) in candidates {
        if let Some(pos) = find_keyword(sparql, kw) {
            match best {
                Some((best_pos, _)) if best_pos <= pos => {}
                _ => best = Some((pos, form)),
            }
        }
    }

    best.map(|(_, form)| form).unwrap_or(QueryForm::Unknown)
}

/// Locate the byte index of the `{` that opens the WHERE `GroupGraphPattern`,
/// honoring the optional `WHERE` keyword.
///
/// * When the `WHERE` keyword is present (as a top-level word), the opening
///   brace of the group that follows it is returned. For
///   `CONSTRUCT { template } WHERE { pattern }` this correctly yields the
///   pattern group.
/// * When `WHERE` is omitted, the group that would have followed it is used:
///   the first top-level `{ ... }` group for `SELECT`/`ASK`/`DESCRIBE`, and the
///   group *after* the template for `CONSTRUCT { template } { pattern }`.
pub fn locate_where_brace(sparql: &str) -> Option<usize> {
    if let Some(where_pos) = find_keyword(sparql, "WHERE") {
        return next_group_open_brace(sparql, where_pos + "WHERE".len());
    }

    match detect_query_form(sparql) {
        QueryForm::Construct => {
            // The first top-level group is the CONSTRUCT template; the graph
            // pattern is the group that follows it.
            let template_open = next_group_open_brace(sparql, 0)?;
            let template_close = matching_close_brace(sparql, template_open)?;
            next_group_open_brace(sparql, template_close + 1)
        }
        _ => next_group_open_brace(sparql, 0),
    }
}

/// Convenience helper returning the `(open, close)` byte indices (inclusive) of
/// the WHERE `GroupGraphPattern` braces, or `None` when no group is present.
pub fn locate_where_group(sparql: &str) -> Option<(usize, usize)> {
    let open = locate_where_brace(sparql)?;
    let close = matching_close_brace(sparql, open)?;
    Some((open, close))
}

/// Return the byte offset at which the `SELECT` projection clause ends — i.e.
/// the position of the top-level `WHERE` keyword when present, otherwise the
/// opening brace of the `GroupGraphPattern`. `select_start` is the byte offset
/// of the `SELECT` keyword. Returns `None` when no group pattern can be found.
pub fn select_projection_end(sparql: &str, select_start: usize) -> Option<usize> {
    if let Some(where_pos) = find_keyword(sparql, "WHERE") {
        if where_pos > select_start {
            return Some(where_pos);
        }
    }
    next_group_open_brace(sparql, select_start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_keyword_ignores_iri_substring() {
        // "SOMEWHERE" contains "WHERE" but only inside an IRI -> must be skipped.
        let q = "SELECT * { ?s <http://example.org/somewhere> ?o }";
        assert_eq!(find_keyword(q, "WHERE"), None);
    }

    #[test]
    fn find_keyword_case_insensitive_word() {
        let q = "SELECT * where { ?s ?p ?o }";
        let pos = find_keyword(q, "WHERE").expect("keyword present");
        assert_eq!(&q[pos..pos + 5], "where");
    }

    #[test]
    fn locate_where_brace_with_keyword() {
        let q = "SELECT * WHERE { ?s ?p ?o }";
        let open = locate_where_brace(q).expect("group present");
        assert_eq!(q.as_bytes()[open], b'{');
        // Group content matches.
        let (o, c) = locate_where_group(q).expect("group");
        assert_eq!(&q[o + 1..c], " ?s ?p ?o ");
    }

    #[test]
    fn locate_where_brace_without_keyword_select() {
        let q = "SELECT * { ?s ?p ?o }";
        let (o, c) = locate_where_group(q).expect("group");
        assert_eq!(&q[o + 1..c], " ?s ?p ?o ");
    }

    #[test]
    fn locate_where_brace_without_keyword_ask() {
        let q = "ASK { ?s ?p ?o }";
        let (o, c) = locate_where_group(q).expect("group");
        assert_eq!(&q[o + 1..c], " ?s ?p ?o ");
    }

    #[test]
    fn locate_where_brace_construct_with_where() {
        let q = "CONSTRUCT { ?a ?b ?c } WHERE { ?s ?p ?o }";
        let (o, c) = locate_where_group(q).expect("group");
        assert_eq!(&q[o + 1..c], " ?s ?p ?o ");
    }

    #[test]
    fn locate_where_brace_construct_without_where() {
        let q = "CONSTRUCT { ?a ?b ?c } { ?s ?p ?o }";
        let (o, c) = locate_where_group(q).expect("group");
        assert_eq!(&q[o + 1..c], " ?s ?p ?o ");
    }

    #[test]
    fn detect_query_form_basic() {
        assert_eq!(detect_query_form("ASK { ?s ?p ?o }"), QueryForm::Ask);
        assert_eq!(
            detect_query_form("SELECT * { ?s ?p ?o }"),
            QueryForm::Select
        );
        assert_eq!(
            detect_query_form("CONSTRUCT { ?s ?p ?o } { ?s ?p ?o }"),
            QueryForm::Construct
        );
        assert_eq!(
            detect_query_form("DESCRIBE <http://example.org/x>"),
            QueryForm::Describe
        );
    }

    #[test]
    fn matching_close_brace_nested() {
        let q = "{ ?s ?p ?o OPTIONAL { ?a ?b ?c } }";
        let close = matching_close_brace(q, 0).expect("balanced");
        assert_eq!(close, q.len() - 1);
    }

    #[test]
    fn select_projection_end_without_where() {
        let q = "SELECT ?x ?y { ?x ?p ?y }";
        let select_start = find_keyword(q, "SELECT").expect("select");
        let end = select_projection_end(q, select_start).expect("end");
        assert_eq!(q.as_bytes()[end], b'{');
        assert_eq!(&q[select_start + 6..end], " ?x ?y ");
    }

    #[test]
    fn find_keyword_ignores_keyword_in_comment() {
        // The first "WHERE" sits inside a # line comment and must be skipped; the
        // real keyword on the next line is the one located.
        let q = "SELECT * # WHERE inside a comment\nWHERE { ?s ?p ?o }";
        let pos = find_keyword(q, "WHERE").expect("real WHERE present");
        assert_eq!(&q[pos..pos + 5], "WHERE");
        let newline = q.find('\n').expect("newline present");
        assert!(pos > newline, "matched the WHERE after the comment line");
    }

    #[test]
    fn find_keyword_hash_in_string_is_not_comment() {
        // A '#' inside a string literal must not start a comment, so the "WHERE"
        // that follows the string on the same line is still located.
        let q = "SELECT * \"# not a comment WHERE\" WHERE { ?s ?p ?o }";
        let pos = find_keyword(q, "WHERE").expect("WHERE present");
        let close_quote = q.rfind('"').expect("closing quote");
        assert!(pos > close_quote, "matched the real WHERE after the string");
    }

    #[test]
    fn find_keyword_hash_in_iri_is_not_comment() {
        // '#' is a legal IRI fragment delimiter; it must not start a comment, so
        // the "WHERE" after the IRI on the same line is still located.
        let q = "SELECT * { ?s <http://example.org/x#frag> ?o } WHERE { ?a ?b ?c }";
        let pos = find_keyword(q, "WHERE").expect("WHERE present");
        let frag = q.find("frag").expect("frag present");
        assert!(pos > frag, "matched the real WHERE after the IRI fragment");
    }

    #[test]
    fn next_group_open_brace_skips_brace_in_comment() {
        // A '{' inside a comment must not be taken as the group open.
        let q = "SELECT * # { not this one\nWHERE { ?s ?p ?o }";
        let open = next_group_open_brace(q, 0).expect("group present");
        let newline = q.find('\n').expect("newline present");
        assert!(open > newline, "skipped the fake brace inside the comment");
        assert_eq!(q.as_bytes()[open], b'{');
    }

    #[test]
    fn matching_close_brace_skips_brace_in_comment() {
        // A '}' inside a comment must not close the group prematurely.
        let q = "{ ?s ?p ?o # } fake close\n ?a ?b ?c }";
        let close = matching_close_brace(q, 0).expect("balanced");
        assert_eq!(close, q.len() - 1);
    }
}
