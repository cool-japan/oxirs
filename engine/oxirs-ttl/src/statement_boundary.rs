//! Shared complete-statement boundary detection for Turtle-family formats.
//!
//! Turtle/TriG/N3 statements are terminated by a `.` (or, for N3 formulas, a
//! `}`) that appears outside of any string literal. Several parts of this
//! crate need to know where such boundaries fall without doing a full parse:
//!
//! - [`crate::incremental::IncrementalParser`], to know how much of a
//!   partially-received chunk can be handed to the parser right now.
//! - [`crate::parallel::ParallelParser`], to split a document into chunks
//!   for parallel parsing without ever splitting a single statement across
//!   two chunks (which previously corrupted or silently dropped multi-line
//!   predicate/object lists that happened to land on a raw line-based chunk
//!   boundary).
//!
//! This module centralizes that boundary-scanning logic so both call sites
//! share one (tested) implementation of the string / long-string tracking
//! instead of maintaining separate copies that can drift out of sync.

/// Find the offset of the end of the *last* complete top-level statement in
/// `content`, if any.
///
/// Returns `(parseable, remaining)` where `parseable` is `content` up to (and
/// including any trailing whitespace after) that boundary, and `remaining` is
/// whatever comes after it — typically a partial trailing statement still
/// waiting on more data. Returns `("", content)` if no complete statement was
/// found anywhere in `content`.
pub(crate) fn find_last_statement_boundary(content: &str) -> (&str, &str) {
    match statement_boundaries(content).last() {
        Some(&end) => (&content[..end], &content[end..]),
        None => ("", content),
    }
}

/// Return the byte offset immediately after every top-level statement
/// terminator (a `.` or `}` outside of any string literal) in `content`,
/// including any whitespace immediately following it.
///
/// A "top-level" `.`/`}` is one that is not inside a `"..."`, `'...'`,
/// `"""..."""`, or `'''...'''` string literal (a backslash-escaped quote
/// inside a short string is also correctly skipped, so it never closes the
/// string early), inside a `<...>` IRI reference (so a `.` in a hostname like
/// `http://example.org` is never mistaken for a terminator), or immediately
/// followed by an ASCII digit (so the `.` in a decimal/double literal like
/// `3.14` — which the Turtle grammar guarantees is always followed directly
/// by a digit — is not mistaken for a terminator either; a statement-ending
/// `.` is always followed by whitespace, a comment, or end of input).
///
/// A lone `<` is only treated as opening an IRI reference if not immediately
/// followed by a second `<` (which instead starts the RDF-star `<<` quoted
/// triple token and is left alone here, since it never itself introduces
/// `.`/`}` ambiguity).
pub(crate) fn statement_boundaries(content: &str) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut in_string = false;
    let mut in_long_string = false;
    let mut in_iri = false;
    let mut string_quote = '\0';
    let mut chars = content.char_indices().peekable();

    while let Some((i, ch)) = chars.next() {
        if !in_string && !in_long_string && !in_iri && (ch == '"' || ch == '\'') {
            // Check for a long (triple-quoted) string opener.
            let mut count = 1;
            while let Some(&(_, next_ch)) = chars.peek() {
                if next_ch == ch && count < 3 {
                    chars.next();
                    count += 1;
                } else {
                    break;
                }
            }

            if count == 3 {
                in_long_string = true;
            } else {
                in_string = count == 1;
            }
            string_quote = ch;
        } else if in_long_string && ch == string_quote {
            // Check for the matching long-string closer.
            let mut count = 1;
            while let Some(&(_, next_ch)) = chars.peek() {
                if next_ch == string_quote && count < 3 {
                    chars.next();
                    count += 1;
                } else {
                    break;
                }
            }
            if count >= 3 {
                in_long_string = false;
            }
        } else if in_string && ch == string_quote {
            in_string = false;
        } else if in_string && ch == '\\' {
            // Skip the escaped character so an escaped quote can't close the
            // string early.
            chars.next();
        } else if !in_string && !in_long_string && !in_iri && ch == '<' {
            // Distinguish an IRIREF opener from the RDF-star `<<` token: a
            // real IRIREF can never start with a second `<` (it is an
            // excluded character), so `<<` is left untouched here.
            let starts_quoted_triple = matches!(chars.peek(), Some(&(_, '<')));
            if starts_quoted_triple {
                chars.next();
            } else {
                in_iri = true;
            }
        } else if in_iri && ch == '\\' {
            // Skip the escaped character (IRIREF UCHAR escapes: \uXXXX / \UXXXXXXXX).
            chars.next();
        } else if in_iri && ch == '>' {
            in_iri = false;
        } else if !in_string && !in_long_string && !in_iri && (ch == '.' || ch == '}') {
            if ch == '.' {
                // A `.` immediately followed by a digit is the decimal point
                // of a DECIMAL/DOUBLE literal (Turtle's grammar requires at
                // least one digit right after it), never a statement
                // terminator, which is always followed by whitespace/EOF.
                if matches!(chars.peek(), Some(&(_, next_ch)) if next_ch.is_ascii_digit()) {
                    continue;
                }
            }

            // Include trailing whitespace after the statement terminator.
            let mut end_pos = i + ch.len_utf8();
            while let Some(&(next_i, next_ch)) = chars.peek() {
                if next_ch == ' ' || next_ch == '\t' || next_ch == '\n' || next_ch == '\r' {
                    chars.next();
                    end_pos = next_i + next_ch.len_utf8();
                } else {
                    break;
                }
            }
            boundaries.push(end_pos);
        }
    }

    boundaries
}

/// Split `content` into chunks that each end on a complete-statement
/// boundary, grouping roughly `target_statements_per_chunk` statements per
/// chunk.
///
/// Unlike naive line-based chunking, a statement that spans multiple lines
/// (e.g. a pretty-printed predicate/object list, or a triple-quoted string
/// containing embedded newlines) is always kept whole in a single chunk. Any
/// trailing bytes after the final statement boundary (trailing whitespace, or
/// a malformed/truncated final statement) are appended to the last chunk so
/// that no input bytes are ever silently dropped.
///
/// Only used by [`crate::parallel::ParallelParser`], which is only compiled
/// when the `parallel` feature is enabled.
#[cfg(feature = "parallel")]
pub(crate) fn split_into_statement_chunks(
    content: &str,
    target_statements_per_chunk: usize,
) -> Vec<String> {
    if content.trim().is_empty() {
        return Vec::new();
    }

    let boundaries = statement_boundaries(content);
    if boundaries.is_empty() {
        // No complete statement found anywhere; hand the whole content to the
        // caller as a single chunk so the underlying parser can still surface
        // a proper syntax error instead of the data being silently lost.
        return vec![content.to_string()];
    }

    let target = target_statements_per_chunk.max(1);
    let mut chunks = Vec::new();
    let mut chunk_start = 0usize;
    let mut count_in_chunk = 0usize;
    let mut last_boundary = 0usize;

    for &boundary in &boundaries {
        count_in_chunk += 1;
        last_boundary = boundary;
        if count_in_chunk >= target {
            chunks.push(content[chunk_start..boundary].to_string());
            chunk_start = boundary;
            count_in_chunk = 0;
        }
    }

    if chunk_start < last_boundary {
        chunks.push(content[chunk_start..last_boundary].to_string());
        chunk_start = last_boundary;
    }

    if chunk_start < content.len() {
        match chunks.last_mut() {
            Some(last) => last.push_str(&content[chunk_start..]),
            None => chunks.push(content[chunk_start..].to_string()),
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statement_boundaries_simple() {
        let content = "a . b .";
        let boundaries = statement_boundaries(content);
        // The trailing space after each '.' is folded into the boundary.
        assert_eq!(boundaries, vec![4, 7]);
    }

    #[test]
    fn test_statement_boundaries_ignores_dot_in_string() {
        let content = r#"ex:s ex:p "a.b.c" ."#;
        let boundaries = statement_boundaries(content);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0], content.len());
    }

    #[test]
    fn test_statement_boundaries_ignores_dot_in_long_string() {
        let content = "ex:s ex:p \"\"\"a.\nb.\nc\"\"\" .";
        let boundaries = statement_boundaries(content);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0], content.len());
    }

    #[test]
    fn test_statement_boundaries_ignores_dot_in_iri() {
        // Regression test: a '.' inside an IRIREF (e.g. a hostname like
        // `example.org`) must never be treated as a statement terminator.
        let content = "@prefix ex: <http://example.org/> .\nex:s ex:p <http://a.b.c/x.y> .\n";
        let boundaries = statement_boundaries(content);
        assert_eq!(boundaries.len(), 2);
        assert_eq!(
            &content[..boundaries[0]],
            "@prefix ex: <http://example.org/> .\n"
        );
        assert_eq!(
            &content[boundaries[0]..boundaries[1]],
            "ex:s ex:p <http://a.b.c/x.y> .\n"
        );
    }

    #[test]
    fn test_statement_boundaries_ignores_dot_in_decimal_literal() {
        // Regression test: the decimal point of a DECIMAL/DOUBLE literal
        // (always followed directly by a digit per the Turtle grammar) must
        // never be treated as a statement terminator.
        let content = "ex:s ex:p 3.14 .\nex:s2 ex:p2 2.5e10 .\n";
        let boundaries = statement_boundaries(content);
        assert_eq!(boundaries.len(), 2);
        assert_eq!(&content[..boundaries[0]], "ex:s ex:p 3.14 .\n");
        assert_eq!(
            &content[boundaries[0]..boundaries[1]],
            "ex:s2 ex:p2 2.5e10 .\n"
        );
    }

    #[test]
    fn test_statement_boundaries_rdf_star_quoted_triple() {
        // The `<<`/`>>` RDF-star quoted-triple tokens must not be mistaken
        // for IRIREF delimiters, and the '.' terminating the whole statement
        // must still be found.
        let content = "<< ex:s ex:p ex:o >> ex:certainty 0.9 .\n";
        let boundaries = statement_boundaries(content);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0], content.len());
    }

    #[test]
    fn test_statement_boundaries_no_boundary() {
        assert!(statement_boundaries("ex:s ex:p ex:o").is_empty());
    }

    #[test]
    fn test_find_last_statement_boundary() {
        let content = "a . b . c";
        let (parseable, remaining) = find_last_statement_boundary(content);
        assert_eq!(parseable, "a . b . ");
        assert_eq!(remaining, "c");
    }

    #[test]
    fn test_find_last_statement_boundary_none() {
        let content = "no terminator here";
        let (parseable, remaining) = find_last_statement_boundary(content);
        assert_eq!(parseable, "");
        assert_eq!(remaining, content);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_split_into_statement_chunks_groups_by_count() {
        let content = "a . b . c . d . e .";
        let chunks = split_into_statement_chunks(content, 2);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "a . b . ");
        assert_eq!(chunks[1], "c . d . ");
        assert_eq!(chunks[2], "e .");
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_split_into_statement_chunks_never_splits_multiline_statement() {
        // A single statement pretty-printed across many lines must always
        // stay in one chunk even with a tiny target size.
        let content = "ex:s\n  ex:p1 ex:o1 ;\n  ex:p2 ex:o2 ;\n  ex:p3 ex:o3 .\nex:s2 ex:p ex:o .";
        let chunks = split_into_statement_chunks(content, 1);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("ex:p1"));
        assert!(chunks[0].contains("ex:p3"));
        assert!(chunks[1].contains("ex:s2"));
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_split_into_statement_chunks_keeps_trailing_partial_data() {
        let content = "a . b . incomplete";
        let chunks = split_into_statement_chunks(content, 1);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "a . ");
        assert_eq!(chunks[1], "b . incomplete");
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_split_into_statement_chunks_empty_input() {
        assert!(split_into_statement_chunks("", 10).is_empty());
        assert!(split_into_statement_chunks("   \n  ", 10).is_empty());
    }
}
