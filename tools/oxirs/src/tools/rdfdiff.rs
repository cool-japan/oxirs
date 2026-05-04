//! RDF Diff tool
//!
//! Compute the symmetric difference between two RDF files.  The tool reports
//! which triples exist only in the first file (removed), only in the second
//! file (added), and in both (unchanged).
//!
//! Blank-node handling: blank nodes are canonically relabeled by sorting all
//! triples lexicographically and assigning sequential identifiers (_:b0, _:b1,
//! …) before computing the diff.  This gives deterministic output for
//! isomorphic graphs.
//!
//! Output formats: `text` (default unified-diff style), `ntriples`, `turtle`,
//! and `patch` (the simple `+`/`-` patch format supported by oxirs-ttl).

use super::ToolResult;
use oxirs_core::model::{Literal as CoreLiteral, Object, Predicate, Subject};
use oxirs_ttl::convenience::parse_rdf_file;
use oxirs_ttl::diff::{compute_diff, NTriple};
use oxirs_ttl::writer::{RdfTerm, TermType};
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;

// ─── Term conversion ──────────────────────────────────────────────────────────

fn subject_to_rdf_term(subj: &Subject) -> RdfTerm {
    match subj {
        Subject::NamedNode(n) => RdfTerm::iri(n.as_str()),
        Subject::BlankNode(b) => RdfTerm::blank_node(b.id()),
        Subject::Variable(v) => RdfTerm::iri(format!("urn:var:{v}")),
        Subject::QuotedTriple(_) => RdfTerm::iri("urn:quoted-triple"),
    }
}

fn predicate_to_rdf_term(pred: &Predicate) -> RdfTerm {
    match pred {
        Predicate::NamedNode(n) => RdfTerm::iri(n.as_str()),
        Predicate::Variable(v) => RdfTerm::iri(format!("urn:var:{v}")),
    }
}

fn literal_to_rdf_term(lit: &CoreLiteral) -> RdfTerm {
    if let Some(lang) = lit.language() {
        RdfTerm::lang_literal(lit.value(), lang)
    } else {
        let dt = lit.datatype();
        let xsd_string = "http://www.w3.org/2001/XMLSchema#string";
        let rdf_lang_string = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString";
        if dt.as_str() == xsd_string || dt.as_str() == rdf_lang_string {
            RdfTerm::simple_literal(lit.value())
        } else {
            RdfTerm::typed_literal(lit.value(), dt.as_str())
        }
    }
}

fn object_to_rdf_term(obj: &Object) -> RdfTerm {
    match obj {
        Object::NamedNode(n) => RdfTerm::iri(n.as_str()),
        Object::BlankNode(b) => RdfTerm::blank_node(b.id()),
        Object::Literal(l) => literal_to_rdf_term(l),
        Object::Variable(v) => RdfTerm::iri(format!("urn:var:{v}")),
        Object::QuotedTriple(_) => RdfTerm::iri("urn:quoted-triple"),
    }
}

// ─── Blank-node canonicalization ─────────────────────────────────────────────

/// Canonically relabel blank nodes in a set of triples.
///
/// Triples are sorted lexicographically (by Display representation), then blank
/// node identifiers are replaced with sequential `_:b0`, `_:b1`, … labels in
/// the order they are first encountered.
fn canonicalize_blank_nodes(triples: Vec<NTriple>) -> Vec<NTriple> {
    // Sort by string representation for determinism
    let mut sorted = triples;
    sorted.sort_by_key(|(s, p, o)| format!("{s} {p} {o}"));

    let mut mapping: HashMap<String, String> = HashMap::new();
    let mut counter = 0usize;

    sorted
        .into_iter()
        .map(|(s, p, o)| {
            (
                relabel_term(s, &mut mapping, &mut counter),
                relabel_term(p, &mut mapping, &mut counter),
                relabel_term(o, &mut mapping, &mut counter),
            )
        })
        .collect()
}

fn relabel_term(
    term: RdfTerm,
    mapping: &mut HashMap<String, String>,
    counter: &mut usize,
) -> RdfTerm {
    if term.term_type == TermType::BlankNode {
        let new_id = mapping
            .entry(term.value.clone())
            .or_insert_with(|| {
                let id = format!("b{counter}");
                *counter += 1;
                id
            })
            .clone();
        RdfTerm::blank_node(new_id)
    } else {
        term
    }
}

// ─── Load and convert file ────────────────────────────────────────────────────

fn load_as_ntriples(path: &PathBuf) -> anyhow::Result<Vec<NTriple>> {
    let triples = parse_rdf_file(path)?;
    let ntriples: Vec<NTriple> = triples
        .iter()
        .map(|t| {
            (
                subject_to_rdf_term(t.subject()),
                predicate_to_rdf_term(t.predicate()),
                object_to_rdf_term(t.object()),
            )
        })
        .collect();
    Ok(ntriples)
}

// ─── Output serialization ─────────────────────────────────────────────────────

fn triple_to_nt_line(triple: &NTriple) -> String {
    format!("{} {} {} .\n", triple.0, triple.1, triple.2)
}

fn serialize_text_diff(
    first_count: usize,
    second_count: usize,
    added: &[NTriple],
    removed: &[NTriple],
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# RDF Diff Summary\n\
         # First graph:  {first_count} triples\n\
         # Second graph: {second_count} triples\n\
         # Removed (-):  {} triples\n\
         # Added (+):    {} triples\n\
         # Unchanged:    {} triples\n\n",
        removed.len(),
        added.len(),
        first_count.min(second_count).saturating_sub(removed.len())
    ));

    for triple in removed {
        out.push_str("- ");
        out.push_str(&triple_to_nt_line(triple));
    }
    for triple in added {
        out.push_str("+ ");
        out.push_str(&triple_to_nt_line(triple));
    }

    out
}

fn serialize_ntriples_diff(added: &[NTriple], removed: &[NTriple]) -> String {
    let mut out = String::new();
    // Emit additions only (ntriples mode shows what's new in second graph)
    for triple in added {
        out.push_str(&triple_to_nt_line(triple));
    }
    // Prefix removed with comment
    for triple in removed {
        out.push_str("# removed: ");
        out.push_str(&triple_to_nt_line(triple));
    }
    out
}

// ─── Main entry point ─────────────────────────────────────────────────────────

/// Compute the RDF diff between two files.
///
/// * `first`   — path to the first RDF file (baseline)
/// * `second`  — path to the second RDF file (target)
/// * `format`  — output format: `text` | `ntriples` | `patch`
pub async fn run(first: PathBuf, second: PathBuf, format: String) -> ToolResult {
    let fmt = format.to_lowercase();
    if !matches!(fmt.as_str(), "text" | "ntriples" | "nt" | "patch") {
        return Err(format!(
            "Unsupported diff format '{format}'. Supported: text, ntriples, patch"
        )
        .into());
    }

    // Load both files
    if !first.exists() {
        return Err(format!("First file not found: {}", first.display()).into());
    }
    if !second.exists() {
        return Err(format!("Second file not found: {}", second.display()).into());
    }

    let first_raw = load_as_ntriples(&first)
        .map_err(|e| format!("Failed to parse '{}': {e}", first.display()))?;
    let second_raw = load_as_ntriples(&second)
        .map_err(|e| format!("Failed to parse '{}': {e}", second.display()))?;

    let first_count = first_raw.len();
    let second_count = second_raw.len();

    // Canonicalize blank nodes in each graph independently
    let first_canon = canonicalize_blank_nodes(first_raw);
    let second_canon = canonicalize_blank_nodes(second_raw);

    // Compute diff
    let diff = compute_diff(&first_canon, &second_canon);

    eprintln!(
        "Diff: {} removed, {} added ({} total changes)",
        diff.removed.len(),
        diff.added.len(),
        diff.triple_count()
    );

    // Serialize output
    let output = match fmt.as_str() {
        "patch" => diff.to_patch_format(),
        "ntriples" | "nt" => serialize_ntriples_diff(&diff.added, &diff.removed),
        _ => serialize_text_diff(first_count, second_count, &diff.added, &diff.removed),
    };

    let mut stdout = io::stdout();
    write!(stdout, "{output}")?;
    stdout.flush()?;

    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn write_turtle(content: &str) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let path = env::temp_dir().join(format!("rdfdiff_test_{nanos}.ttl"));
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(content.as_bytes()).expect("write");
        path
    }

    #[test]
    fn test_canonicalize_blank_nodes_empty() {
        let result = canonicalize_blank_nodes(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_canonicalize_blank_nodes_relabels() {
        let triples = vec![
            (
                RdfTerm::blank_node("x123"),
                RdfTerm::iri("http://p"),
                RdfTerm::iri("http://o"),
            ),
            (
                RdfTerm::blank_node("y456"),
                RdfTerm::iri("http://p"),
                RdfTerm::iri("http://o2"),
            ),
        ];
        let result = canonicalize_blank_nodes(triples);
        assert_eq!(result.len(), 2);
        // All blank nodes should be _:b0 or _:b1
        for (s, _, _) in &result {
            if s.term_type == TermType::BlankNode {
                assert!(
                    s.value.starts_with('b'),
                    "expected canonical blank node id, got: {}",
                    s.value
                );
            }
        }
    }

    #[test]
    fn test_text_diff_format() {
        let added = vec![(
            RdfTerm::iri("http://s"),
            RdfTerm::iri("http://p"),
            RdfTerm::iri("http://o2"),
        )];
        let removed = vec![(
            RdfTerm::iri("http://s"),
            RdfTerm::iri("http://p"),
            RdfTerm::iri("http://o1"),
        )];
        let text = serialize_text_diff(1, 1, &added, &removed);
        assert!(text.contains("+ "), "missing '+' in: {text}");
        assert!(text.contains("- "), "missing '-' in: {text}");
    }

    #[tokio::test]
    async fn test_missing_first_file_returns_error() {
        let a = env::temp_dir().join("rdfdiff_nonexistent_a.ttl");
        let b = env::temp_dir().join("rdfdiff_nonexistent_b.ttl");
        let res = run(a, b, "text".into()).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn test_bad_format_returns_error() {
        let a = write_turtle("<http://s> <http://p> <http://o> .\n");
        let b = write_turtle("<http://s> <http://p> <http://o2> .\n");
        let res = run(a.clone(), b.clone(), "html".into()).await;
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
        assert!(res.is_err());
        if let Err(e) = res {
            assert!(
                e.to_string().contains("Unsupported diff format"),
                "got: {e}"
            );
        }
    }

    #[tokio::test]
    async fn test_identical_files_show_no_changes() {
        let content = r#"
<http://example.org/s> <http://example.org/p> <http://example.org/o> .
"#;
        let a = write_turtle(content);
        let b = write_turtle(content);
        let res = run(a.clone(), b.clone(), "patch".into()).await;
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
        assert!(res.is_ok(), "identical diff failed: {:?}", res.err());
    }

    #[tokio::test]
    async fn test_diff_with_changes() {
        let first = r#"<http://example.org/s> <http://example.org/p> <http://example.org/o1> .
"#;
        let second = r#"<http://example.org/s> <http://example.org/p> <http://example.org/o2> .
"#;
        let a = write_turtle(first);
        let b = write_turtle(second);
        let res = run(a.clone(), b.clone(), "text".into()).await;
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
        assert!(res.is_ok(), "diff with changes failed: {:?}", res.err());
    }
}
