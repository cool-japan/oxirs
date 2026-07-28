//! Regression tests for the N-Triples / N-Quads literal parser char-boundary
//! panic (P0).
//!
//! `Parser::parse_literal` (core/oxirs-core/src/parser/mod.rs) used to locate
//! the closing quote of a literal by collecting the token into a
//! `Vec<char>` and recording the **char** index of the closing `"`, then
//! reused that index to slice the original `&str` token, which is indexed in
//! **bytes**. Any literal containing at least one multi-byte UTF-8 character
//! (Japanese text, emoji, accented Latin, ...) before the closing quote made
//! the char index diverge from the byte offset, and slicing a `&str` at a
//! non-char-boundary byte offset panics.
//!
//! These tests drive the fix through the public `Parser` API (which is what
//! `oxirs-fuseki`'s upload / SPARQL LOAD / INSERT-DATA-DATA-block code paths
//! use under the hood for N-Triples and N-Quads), asserting that:
//! - parsing no longer panics on multi-byte literals,
//! - the lexical value, language tag, and datatype are all preserved exactly,
//! - plain-ASCII behavior is unchanged.

use oxirs_core::model::{GraphName, Object};
use oxirs_core::parser::{Parser, RdfFormat};

const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";

fn parse_single_ntriples_object(line: &str) -> Object {
    let parser = Parser::new(RdfFormat::NTriples);
    let quads = parser
        .parse_str_to_quads(line)
        .unwrap_or_else(|e| panic!("N-Triples parse failed for {line:?}: {e}"));
    assert_eq!(quads.len(), 1, "expected exactly one quad from {line:?}");
    quads[0].object().clone()
}

fn parse_single_nquads_quad(line: &str) -> oxirs_core::model::Quad {
    let parser = Parser::new(RdfFormat::NQuads);
    let quads = parser
        .parse_str_to_quads(line)
        .unwrap_or_else(|e| panic!("N-Quads parse failed for {line:?}: {e}"));
    assert_eq!(quads.len(), 1, "expected exactly one quad from {line:?}");
    quads[0].clone()
}

fn assert_plain_literal(object: &Object, expected_value: &str) {
    match object {
        Object::Literal(lit) => {
            assert_eq!(lit.value(), expected_value);
            assert_eq!(lit.language(), None);
            assert_eq!(lit.datatype().as_str(), XSD_STRING);
        }
        other => panic!("expected a plain literal, got {other:?}"),
    }
}

fn assert_lang_literal(object: &Object, expected_value: &str, expected_lang: &str) {
    match object {
        Object::Literal(lit) => {
            assert_eq!(lit.value(), expected_value);
            assert_eq!(lit.language(), Some(expected_lang));
        }
        other => panic!("expected a language-tagged literal, got {other:?}"),
    }
}

fn assert_typed_literal(object: &Object, expected_value: &str, expected_datatype: &str) {
    match object {
        Object::Literal(lit) => {
            assert_eq!(lit.value(), expected_value);
            assert_eq!(lit.datatype().as_str(), expected_datatype);
        }
        other => panic!("expected a typed literal, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// N-Triples: multi-byte literals must not panic, and must round-trip
// exactly (value / language / datatype).
// ---------------------------------------------------------------------

#[test]
fn ntriples_japanese_plain_literal_does_not_panic_and_round_trips() {
    let line = r#"<http://example.org/s> <http://example.org/p> "日本語" ."#;
    let object = parse_single_ntriples_object(line);
    assert_plain_literal(&object, "日本語");
}

#[test]
fn ntriples_japanese_language_tagged_literal() {
    let line = r#"<http://example.org/s> <http://example.org/p> "日本語"@ja ."#;
    let object = parse_single_ntriples_object(line);
    assert_lang_literal(&object, "日本語", "ja");
}

#[test]
fn ntriples_japanese_explicit_xsd_string_typed_literal() {
    let line =
        format!(r#"<http://example.org/s> <http://example.org/p> "日本語"^^<{XSD_STRING}> ."#);
    let object = parse_single_ntriples_object(&line);
    // xsd:string is normalized to a plain literal by `Literal::new_typed`.
    assert_plain_literal(&object, "日本語");
}

#[test]
fn ntriples_japanese_non_string_typed_literal() {
    // A datatype other than xsd:string must be preserved (not normalized away).
    let line = r#"<http://example.org/s> <http://example.org/p> "日本語"^^<http://example.org/customType> ."#;
    let object = parse_single_ntriples_object(line);
    assert_typed_literal(&object, "日本語", "http://example.org/customType");
}

#[test]
fn ntriples_emoji_literal() {
    let line = r#"<http://example.org/s> <http://example.org/p> "🎉🎊絵文字" ."#;
    let object = parse_single_ntriples_object(line);
    assert_plain_literal(&object, "🎉🎊絵文字");
}

#[test]
fn ntriples_emoji_literal_with_language_tag() {
    let line = r#"<http://example.org/s> <http://example.org/p> "🎉party"@en ."#;
    let object = parse_single_ntriples_object(line);
    assert_lang_literal(&object, "🎉party", "en");
}

#[test]
fn ntriples_accented_latin_literal() {
    let line = r#"<http://example.org/s> <http://example.org/p> "café résumé naïve"@fr ."#;
    let object = parse_single_ntriples_object(line);
    assert_lang_literal(&object, "café résumé naïve", "fr");
}

#[test]
fn ntriples_multibyte_with_escaped_quote() {
    // Escaped quote (\") embedded between multi-byte characters. The
    // tokenizer preserves the `\"` escape sequence verbatim, and
    // `unescape_literal_value` turns it into a literal `"` character.
    let line = r#"<http://example.org/s> <http://example.org/p> "日\"本" ."#;
    let object = parse_single_ntriples_object(line);
    assert_plain_literal(&object, "日\"本");
}

#[test]
fn ntriples_multibyte_with_escaped_backslash_and_newline() {
    let line = r#"<http://example.org/s> <http://example.org/p> "日本\\語\n改行" ."#;
    let object = parse_single_ntriples_object(line);
    assert_plain_literal(&object, "日本\\語\n改行");
}

#[test]
fn ntriples_multibyte_iri_subject_and_predicate() {
    // IRI slicing (`token[1..len-1]`) only depends on the ASCII `<`/`>`
    // delimiters, so multi-byte IRI content was already safe -- pinned here
    // as a regression guard.
    let line = r#"<http://example.org/日本語/主語> <http://example.org/述語> "value" ."#;
    let object = parse_single_ntriples_object(line);
    assert_plain_literal(&object, "value");
}

#[test]
fn ntriples_multiline_document_with_mixed_ascii_and_multibyte() {
    let data = concat!(
        "<http://example.org/s1> <http://example.org/p> \"ascii value\" .\n",
        "<http://example.org/s2> <http://example.org/p> \"日本語の値\"@ja .\n",
        "<http://example.org/s3> <http://example.org/p> \"emoji 🚀\" .\n",
    );
    let parser = Parser::new(RdfFormat::NTriples);
    let quads = parser
        .parse_str_to_quads(data)
        .expect("multi-line N-Triples document with multi-byte literals must parse");
    assert_eq!(quads.len(), 3);
    assert_plain_literal(quads[0].object(), "ascii value");
    assert_lang_literal(quads[1].object(), "日本語の値", "ja");
    assert_plain_literal(quads[2].object(), "emoji 🚀");
}

// ---------------------------------------------------------------------
// N-Quads: same literal-parsing code path (tokenize_ntriples_line /
// parse_literal), plus a graph name component.
// ---------------------------------------------------------------------

#[test]
fn nquads_japanese_language_tagged_literal_with_graph() {
    let line =
        r#"<http://example.org/s> <http://example.org/p> "日本語"@ja <http://example.org/g> ."#;
    let quad = parse_single_nquads_quad(line);
    assert_lang_literal(quad.object(), "日本語", "ja");
    match quad.graph_name() {
        GraphName::NamedNode(n) => assert_eq!(n.as_str(), "http://example.org/g"),
        other => panic!("expected named graph, got {other:?}"),
    }
}

#[test]
fn nquads_japanese_plain_literal_with_graph() {
    let line = r#"<http://example.org/s> <http://example.org/p> "日本語" <http://example.org/g> ."#;
    let quad = parse_single_nquads_quad(line);
    assert_plain_literal(quad.object(), "日本語");
}

#[test]
fn nquads_emoji_typed_literal_with_graph() {
    let line = r#"<http://example.org/s> <http://example.org/p> "🎉🎊"^^<http://example.org/emoji> <http://example.org/g> ."#;
    let quad = parse_single_nquads_quad(line);
    assert_typed_literal(quad.object(), "🎉🎊", "http://example.org/emoji");
}

#[test]
fn nquads_multibyte_with_escaped_quote_and_graph() {
    let line =
        r#"<http://example.org/s> <http://example.org/p> "日\"本"@ja <http://example.org/g> ."#;
    let quad = parse_single_nquads_quad(line);
    assert_lang_literal(quad.object(), "日\"本", "ja");
}

#[test]
fn nquads_multibyte_iri_subject_predicate_and_graph() {
    let line = r#"<http://example.org/日本語/主語> <http://example.org/述語> "value"@en <http://example.org/グラフ> ."#;
    let quad = parse_single_nquads_quad(line);
    assert_lang_literal(quad.object(), "value", "en");
    match quad.graph_name() {
        GraphName::NamedNode(n) => assert_eq!(n.as_str(), "http://example.org/グラフ"),
        other => panic!("expected named graph, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// Plain-ASCII regression guard: existing behavior must be fully unchanged.
// ---------------------------------------------------------------------

#[test]
fn ntriples_ascii_plain_literal_unchanged() {
    let line = r#"<http://example.org/s> <http://example.org/p> "hello world" ."#;
    let object = parse_single_ntriples_object(line);
    assert_plain_literal(&object, "hello world");
}

#[test]
fn ntriples_ascii_language_tagged_literal_unchanged() {
    let line = r#"<http://example.org/s> <http://example.org/p> "hello"@en ."#;
    let object = parse_single_ntriples_object(line);
    assert_lang_literal(&object, "hello", "en");
}

#[test]
fn ntriples_ascii_typed_literal_unchanged() {
    let line = format!(r#"<http://example.org/s> <http://example.org/p> "42"^^<{XSD_INTEGER}> ."#);
    let object = parse_single_ntriples_object(&line);
    assert_typed_literal(&object, "42", XSD_INTEGER);
}

#[test]
fn ntriples_ascii_escaped_quote_unchanged() {
    let line = r#"<http://example.org/s> <http://example.org/p> "say \"hi\"" ."#;
    let object = parse_single_ntriples_object(line);
    assert_plain_literal(&object, "say \"hi\"");
}

#[test]
fn nquads_ascii_literal_with_graph_unchanged() {
    let line =
        r#"<http://example.org/s> <http://example.org/p> "hello"@en <http://example.org/g> ."#;
    let quad = parse_single_nquads_quad(line);
    assert_lang_literal(quad.object(), "hello", "en");
}
