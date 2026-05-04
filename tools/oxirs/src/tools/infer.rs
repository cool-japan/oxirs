//! Inference tool
//!
//! Perform RDFS or OWL-RL inference over an RDF file, optionally loading
//! an ontology from a separate file.  The inferred triples (plus the
//! original data) are written to an output file or stdout.

use super::ToolResult;
use oxirs_core::model::Triple as CoreTriple;
use oxirs_rule::owl_rl::Owl2RlReasoner;
use oxirs_rule::rdfs::{RdfsProfile, RdfsReasoner};
use oxirs_rule::{RuleAtom, Term as RuleTerm};
use oxirs_ttl::convenience::parse_rdf_file;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;

// ─── Profile enum ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InferProfile {
    Rdfs,
    RdfsMinimal,
    OwlRl,
}

impl InferProfile {
    fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "RDFS" | "RDFS-FULL" => Some(Self::Rdfs),
            "RDFS-MINIMAL" | "RDFS-MIN" => Some(Self::RdfsMinimal),
            "OWL-RL" | "OWL_RL" | "OWLRL" => Some(Self::OwlRl),
            _ => None,
        }
    }
}

// ─── Conversion helpers ───────────────────────────────────────────────────────

fn core_triple_to_rule_atom(triple: &CoreTriple) -> RuleAtom {
    RuleAtom::Triple {
        subject: RuleTerm::Constant(triple.subject().to_string()),
        predicate: RuleTerm::Constant(triple.predicate().to_string()),
        object: RuleTerm::Constant(triple.object().to_string()),
    }
}

fn rule_atom_to_ntriples_line(atom: &RuleAtom) -> Option<String> {
    if let RuleAtom::Triple {
        subject,
        predicate,
        object,
    } = atom
    {
        let s = term_to_nt(subject);
        let p = term_to_nt(predicate);
        let o = term_to_nt(object);
        Some(format!("{s} {p} {o} .\n"))
    } else {
        None
    }
}

fn term_to_nt(term: &RuleTerm) -> String {
    match term {
        RuleTerm::Constant(v) => {
            // Already bracketed as IRI, quoted as literal, or prefixed as blank node
            if v.starts_with('<') && v.ends_with('>') || v.starts_with('"') || v.starts_with("_:") {
                v.clone()
            } else {
                format!("<{v}>")
            }
        }
        RuleTerm::Literal(v) => format!("\"{v}\""),
        RuleTerm::Variable(v) => format!("?{v}"),
        RuleTerm::Function { name, .. } => format!("<urn:function:{name}>"),
    }
}

// ─── RDFS inference ───────────────────────────────────────────────────────────

fn run_rdfs_inference(
    data_atoms: &[RuleAtom],
    ontology_atoms: &[RuleAtom],
    profile: InferProfile,
) -> anyhow::Result<Vec<RuleAtom>> {
    let rdfs_profile = match profile {
        InferProfile::RdfsMinimal => RdfsProfile::Minimal,
        _ => RdfsProfile::Full,
    };
    let mut reasoner = RdfsReasoner::with_profile(rdfs_profile);

    // Combine data and ontology facts
    let mut all_facts: Vec<RuleAtom> = ontology_atoms.to_vec();
    all_facts.extend_from_slice(data_atoms);

    let inferred = reasoner.infer(&all_facts)?;
    Ok(inferred)
}

// ─── OWL-RL inference ────────────────────────────────────────────────────────

fn run_owl_rl_inference(
    data_atoms: &[RuleAtom],
    ontology_atoms: &[RuleAtom],
) -> anyhow::Result<Vec<RuleAtom>> {
    let mut reasoner = Owl2RlReasoner::new();

    // Load ontology axioms
    for atom in ontology_atoms {
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = atom
        {
            if let (RuleTerm::Constant(s), RuleTerm::Constant(p), RuleTerm::Constant(o)) =
                (subject, predicate, object)
            {
                reasoner.add_axiom(s, p, o);
            }
        }
    }

    // Load data axioms
    for atom in data_atoms {
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = atom
        {
            if let (RuleTerm::Constant(s), RuleTerm::Constant(p), RuleTerm::Constant(o)) =
                (subject, predicate, object)
            {
                reasoner.add_axiom(s, p, o);
            }
        }
    }

    reasoner
        .materialize()
        .map_err(|e| anyhow::anyhow!("OWL-RL materialization failed: {e}"))?;

    // Collect all triples (original + inferred) as RuleAtoms
    let all_triples = reasoner.all_triples();
    let atoms: Vec<RuleAtom> = all_triples
        .into_iter()
        .map(|(s, p, o)| RuleAtom::Triple {
            subject: RuleTerm::Constant(s),
            predicate: RuleTerm::Constant(p),
            object: RuleTerm::Constant(o),
        })
        .collect();

    Ok(atoms)
}

// ─── Main entry point ─────────────────────────────────────────────────────────

/// Perform inference over RDF data.
///
/// * `data`      — path to the input RDF file (Turtle / N-Triples / auto-detect)
/// * `ontology`  — optional separate ontology/schema file to load
/// * `profile`   — reasoning profile: `RDFS` | `RDFS-MINIMAL` | `OWL-RL`
/// * `output`    — optional output file path (stdout if not provided)
/// * `format`    — output format: `ntriples` (default) | `turtle`
pub async fn run(
    data: PathBuf,
    ontology: Option<PathBuf>,
    profile: String,
    output: Option<PathBuf>,
    format: String,
) -> ToolResult {
    // Validate and parse profile
    let infer_profile = InferProfile::parse(&profile).ok_or_else(|| {
        format!("Unsupported inference profile '{profile}'. Supported: RDFS, RDFS-MINIMAL, OWL-RL")
    })?;

    // Validate format
    let fmt = format.to_lowercase();
    if !matches!(fmt.as_str(), "ntriples" | "nt" | "turtle" | "ttl") {
        return Err(
            format!("Unsupported output format '{format}'. Supported: ntriples, turtle").into(),
        );
    }

    // Load data file
    if !data.exists() {
        return Err(format!("Data file not found: {}", data.display()).into());
    }
    let data_triples = parse_rdf_file(&data)
        .map_err(|e| format!("Failed to parse data file '{}': {e}", data.display()))?;
    let data_atoms: Vec<RuleAtom> = data_triples.iter().map(core_triple_to_rule_atom).collect();

    // Load ontology file (optional)
    let ontology_atoms: Vec<RuleAtom> = if let Some(ref onto_path) = ontology {
        if !onto_path.exists() {
            return Err(format!("Ontology file not found: {}", onto_path.display()).into());
        }
        let onto_triples = parse_rdf_file(onto_path)
            .map_err(|e| format!("Failed to parse ontology '{}': {e}", onto_path.display()))?;
        onto_triples.iter().map(core_triple_to_rule_atom).collect()
    } else {
        Vec::new()
    };

    println!(
        "Loaded {} data triples, {} ontology triples",
        data_atoms.len(),
        ontology_atoms.len()
    );
    println!("Running {profile} inference...");

    // Run inference
    let result_atoms = match infer_profile {
        InferProfile::Rdfs | InferProfile::RdfsMinimal => {
            run_rdfs_inference(&data_atoms, &ontology_atoms, infer_profile)
        }
        InferProfile::OwlRl => run_owl_rl_inference(&data_atoms, &ontology_atoms),
    }
    .map_err(|e| format!("Inference failed: {e}"))?;

    let inferred_count = result_atoms.len().saturating_sub(data_atoms.len());
    println!(
        "Inference complete: {} total triples ({} new inferred)",
        result_atoms.len(),
        inferred_count
    );

    // Serialize output
    let mut writer: Box<dyn Write> = if let Some(ref out_path) = output {
        Box::new(
            File::create(out_path)
                .map_err(|e| format!("Cannot create output file '{}': {e}", out_path.display()))?,
        )
    } else {
        Box::new(io::stdout())
    };

    for atom in &result_atoms {
        if let Some(line) = rule_atom_to_ntriples_line(atom) {
            write!(writer, "{line}")?;
        }
    }
    writer.flush()?;

    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::io::Write;

    fn write_temp_turtle(content: &str) -> PathBuf {
        let path = env::temp_dir().join(format!("infer_test_{}.ttl", uuid_part()));
        let mut f = File::create(&path).expect("create temp file");
        f.write_all(content.as_bytes()).expect("write temp file");
        path
    }

    fn uuid_part() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64
    }

    #[test]
    fn test_infer_profile_parse() {
        assert_eq!(InferProfile::parse("rdfs"), Some(InferProfile::Rdfs));
        assert_eq!(InferProfile::parse("RDFS"), Some(InferProfile::Rdfs));
        assert_eq!(InferProfile::parse("OWL-RL"), Some(InferProfile::OwlRl));
        assert_eq!(InferProfile::parse("UNKNOWN"), None);
    }

    #[test]
    fn test_term_to_nt_iri() {
        let t = RuleTerm::Constant("http://example.org/test".to_string());
        let nt = term_to_nt(&t);
        assert_eq!(nt, "<http://example.org/test>");
    }

    #[test]
    fn test_term_to_nt_literal() {
        let t = RuleTerm::Literal("hello".to_string());
        let nt = term_to_nt(&t);
        assert_eq!(nt, "\"hello\"");
    }

    #[test]
    fn test_rule_atom_to_ntriples_line() {
        let atom = RuleAtom::Triple {
            subject: RuleTerm::Constant("http://s".to_string()),
            predicate: RuleTerm::Constant("http://p".to_string()),
            object: RuleTerm::Constant("http://o".to_string()),
        };
        let line = rule_atom_to_ntriples_line(&atom);
        assert!(line.is_some());
        if let Some(text) = line {
            assert!(text.contains("<http://s>"), "got: {text}");
            assert!(text.ends_with(".\n"), "got: {text}");
        }
    }

    #[tokio::test]
    async fn test_missing_data_file_returns_error() {
        let nonexistent = env::temp_dir().join("infer_no_such_file_99999.ttl");
        let res = run(nonexistent, None, "RDFS".into(), None, "ntriples".into()).await;
        assert!(res.is_err(), "should fail for missing data file");
    }

    #[tokio::test]
    async fn test_bad_profile_returns_error() {
        let tmp = env::temp_dir().join("infer_bad_profile.ttl");
        let res = run(tmp, None, "SHACL".into(), None, "ntriples".into()).await;
        assert!(res.is_err());
        if let Err(e) = res {
            assert!(
                e.to_string().contains("Unsupported inference profile"),
                "got: {e}"
            );
        }
    }

    #[tokio::test]
    async fn test_rdfs_inference_with_data() {
        let turtle = r#"
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .

ex:Animal a rdfs:Class .
ex:Dog rdfs:subClassOf ex:Animal .
ex:Rex a ex:Dog .
"#;
        let path = write_temp_turtle(turtle);
        let out_path = env::temp_dir().join(format!("infer_rdfs_out_{}.nt", uuid_part()));
        let res = run(
            path.clone(),
            None,
            "RDFS".into(),
            Some(out_path.clone()),
            "ntriples".into(),
        )
        .await;
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&out_path);
        assert!(res.is_ok(), "RDFS inference failed: {:?}", res.err());
    }
}
