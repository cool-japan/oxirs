//! # OWL 2 Profile Conformance Suite Runner
//!
//! Walks the JSON corpus under `tests/fixtures/owl2-tests/` and replays
//! each case through every applicable [`OwlProfile`] backend.  Pass-rate
//! thresholds (per the W1-S2 plan):
//!
//! - `RL`   → 1.00
//! - `EL`   → ≥ 0.95
//! - `QL`   → ≥ 0.95
//! - `RLEL` → ≥ 0.95
//! - `DL`   → ≥ 0.95
//!
//! Each test case is a JSON file with the schema:
//!
//! ```json
//! {
//!   "id": "el-intersection-001",
//!   "title": "EL intersection on LHS",
//!   "profiles": ["EL", "RLEL", "DL"],
//!   "axioms": [
//!     { "kind": "SubClassOf", "sub": "Cat", "sup": "Mammal" },
//!     { "kind": "IntersectionSubClassOf", "parts": ["Doctor", "Surgeon"], "sup": "DoctorSurgeon" }
//!   ],
//!   "data": [
//!     { "subject": "alice", "predicate": "rdf:type", "object": "Doctor" }
//!   ],
//!   "expected": [
//!     { "kind": "Type", "individual": "alice", "class": "Doctor" }
//!   ],
//!   "forbidden": [
//!     { "kind": "Type", "individual": "alice", "class": "Surgeon" }
//!   ]
//! }
//! ```
//!
//! Adding a new fixture: drop a `*.json` file under `tests/fixtures/owl2-tests/`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use oxirs_rule::owl2::{
    reason_in_profile, Owl2Axiom, Owl2Ontology, OwlProfile, ProfileEntailment, ReasoningOutcome,
};

#[derive(Debug, Deserialize)]
struct ConformanceCase {
    id: String,
    #[allow(dead_code)]
    title: Option<String>,
    profiles: Vec<String>,
    #[serde(default)]
    axioms: Vec<AxiomDescriptor>,
    #[serde(default)]
    data: Vec<TripleDescriptor>,
    #[serde(default)]
    expected: Vec<EntailmentDescriptor>,
    #[serde(default)]
    forbidden: Vec<EntailmentDescriptor>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind")]
enum AxiomDescriptor {
    SubClassOf {
        sub: String,
        sup: String,
    },
    EquivalentClasses {
        a: String,
        b: String,
    },
    IntersectionSubClassOf {
        parts: Vec<String>,
        sup: String,
    },
    SubClassOfSomeValuesFrom {
        sub: String,
        role: String,
        filler: String,
    },
    SomeValuesFromSubClassOf {
        role: String,
        filler: String,
        sup: String,
    },
    SubObjectPropertyOf {
        sub: String,
        sup: String,
    },
    EquivalentObjectProperties {
        a: String,
        b: String,
    },
    InverseObjectProperties {
        a: String,
        b: String,
    },
    PropertyChain {
        chain: Vec<String>,
        result_role: String,
    },
    TransitiveProperty {
        property: String,
    },
    ReflexiveProperty {
        property: String,
    },
    ObjectPropertyDomain {
        property: String,
        class: String,
    },
    ObjectPropertyRange {
        property: String,
        class: String,
    },
}

impl AxiomDescriptor {
    fn into_axiom(self) -> Owl2Axiom {
        match self {
            Self::SubClassOf { sub, sup } => Owl2Axiom::SubClassOf { sub, sup },
            Self::EquivalentClasses { a, b } => Owl2Axiom::EquivalentClasses(a, b),
            Self::IntersectionSubClassOf { parts, sup } => {
                Owl2Axiom::IntersectionSubClassOf { parts, sup }
            }
            Self::SubClassOfSomeValuesFrom { sub, role, filler } => {
                Owl2Axiom::SubClassOfSomeValuesFrom { sub, role, filler }
            }
            Self::SomeValuesFromSubClassOf { role, filler, sup } => {
                Owl2Axiom::SomeValuesFromSubClassOf { role, filler, sup }
            }
            Self::SubObjectPropertyOf { sub, sup } => Owl2Axiom::SubObjectPropertyOf { sub, sup },
            Self::EquivalentObjectProperties { a, b } => {
                Owl2Axiom::EquivalentObjectProperties(a, b)
            }
            Self::InverseObjectProperties { a, b } => Owl2Axiom::InverseObjectProperties(a, b),
            Self::PropertyChain { chain, result_role } => {
                Owl2Axiom::PropertyChain { chain, result_role }
            }
            Self::TransitiveProperty { property } => Owl2Axiom::TransitiveProperty(property),
            Self::ReflexiveProperty { property } => Owl2Axiom::ReflexiveProperty(property),
            Self::ObjectPropertyDomain { property, class } => {
                Owl2Axiom::ObjectPropertyDomain { property, class }
            }
            Self::ObjectPropertyRange { property, class } => {
                Owl2Axiom::ObjectPropertyRange { property, class }
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct TripleDescriptor {
    subject: String,
    predicate: String,
    object: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
enum EntailmentDescriptor {
    SubClassOf {
        sub: String,
        sup: String,
    },
    Type {
        individual: String,
        class: String,
    },
    Property {
        subject: String,
        property: String,
        object: String,
    },
}

impl EntailmentDescriptor {
    fn matches(&self, ent: &ProfileEntailment) -> bool {
        match (self, ent) {
            (Self::SubClassOf { sub, sup }, ProfileEntailment::SubClassOf { sub: s, sup: t }) => {
                sub == s && sup == t
            }
            (
                Self::Type { individual, class },
                ProfileEntailment::Type {
                    individual: i,
                    class: c,
                },
            ) => individual == i && class == c,
            (
                Self::Property {
                    subject,
                    property,
                    object,
                },
                ProfileEntailment::Property {
                    subject: s,
                    property: p,
                    object: o,
                },
            ) => subject == s && property == p && object == o,
            _ => false,
        }
    }

    fn label(&self) -> String {
        match self {
            Self::SubClassOf { sub, sup } => format!("SubClassOf({sub} ⊑ {sup})"),
            Self::Type { individual, class } => format!("Type({individual} : {class})"),
            Self::Property {
                subject,
                property,
                object,
            } => format!("Property({subject} {property} {object})"),
        }
    }
}

fn fixtures_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("tests");
    dir.push("fixtures");
    dir.push("owl2-tests");
    dir
}

fn load_cases(dir: &Path) -> Vec<(PathBuf, ConformanceCase)> {
    let mut cases = Vec::new();
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("Failed to read fixtures dir {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read fixture {}: {e}", path.display()));
        let case: ConformanceCase = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("Failed to parse fixture {} as JSON: {e}", path.display()));
        cases.push((path, case));
    }
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    cases
}

fn build_ontology(case: &ConformanceCase) -> Owl2Ontology {
    let mut ontology = Owl2Ontology::new();
    for axiom in &case.axioms {
        ontology.axioms.push(axiom.clone().into_axiom());
    }
    for triple in &case.data {
        ontology.data_triples.push((
            triple.subject.clone(),
            triple.predicate.clone(),
            triple.object.clone(),
        ));
    }
    ontology
}

fn check_case(
    case: &ConformanceCase,
    profile: OwlProfile,
    outcome: &ReasoningOutcome,
) -> CaseResult {
    let mut missing = Vec::new();
    let mut violated = Vec::new();

    for expected in &case.expected {
        let satisfied = outcome.entailments.iter().any(|ent| expected.matches(ent));
        if !satisfied {
            missing.push(expected.label());
        }
    }
    for forbidden in &case.forbidden {
        let violation = outcome.entailments.iter().any(|ent| forbidden.matches(ent));
        if violation {
            violated.push(forbidden.label());
        }
    }

    let pass = missing.is_empty() && violated.is_empty();
    CaseResult {
        case_id: case.id.clone(),
        profile,
        pass,
        missing,
        violated,
    }
}

#[derive(Debug)]
struct CaseResult {
    case_id: String,
    profile: OwlProfile,
    pass: bool,
    missing: Vec<String>,
    violated: Vec<String>,
}

fn run_corpus() -> HashMap<OwlProfile, Vec<CaseResult>> {
    let mut results: HashMap<OwlProfile, Vec<CaseResult>> = HashMap::new();
    for &profile in OwlProfile::all() {
        results.insert(profile, Vec::new());
    }

    let cases = load_cases(&fixtures_dir());
    assert!(!cases.is_empty(), "No conformance fixtures found");

    for (path, case) in &cases {
        let ontology = build_ontology(case);
        for profile_name in &case.profiles {
            let profile = OwlProfile::parse(profile_name).unwrap_or_else(|| {
                panic!(
                    "Fixture {} declared unknown profile {profile_name}",
                    path.display()
                )
            });

            match reason_in_profile(profile, &ontology) {
                Ok(outcome) => {
                    let r = check_case(case, profile, &outcome);
                    results.entry(profile).or_default().push(r);
                }
                Err(e) => {
                    results.entry(profile).or_default().push(CaseResult {
                        case_id: case.id.clone(),
                        profile,
                        pass: false,
                        missing: vec![format!("reasoner error: {e}")],
                        violated: vec![],
                    });
                }
            }
        }
    }
    results
}

fn pass_rate(results: &[CaseResult]) -> f64 {
    if results.is_empty() {
        return 1.0;
    }
    let pass_count = results.iter().filter(|r| r.pass).count();
    pass_count as f64 / results.len() as f64
}

fn report_failures(profile: OwlProfile, results: &[CaseResult]) -> String {
    let mut lines = Vec::new();
    for r in results.iter().filter(|r| !r.pass) {
        lines.push(format!(
            "  FAIL [{profile}] {} — missing: {:?} | violated: {:?}",
            r.case_id, r.missing, r.violated
        ));
    }
    lines.join("\n")
}

#[test]
fn corpus_runs_at_least_one_case_per_profile() {
    let results = run_corpus();
    for &profile in OwlProfile::all() {
        let count = results.get(&profile).map(|v| v.len()).unwrap_or(0);
        assert!(
            count > 0,
            "Profile {profile} must have at least one fixture in the corpus"
        );
    }
}

#[test]
fn rl_profile_pass_rate_is_one() {
    let results = run_corpus();
    let rl = results.get(&OwlProfile::Rl).cloned().unwrap_or_default();
    let rate = pass_rate(&rl);
    assert!(
        rate >= 1.0,
        "OWL 2 RL pass rate must be 1.00; got {rate:.2}\n{}",
        report_failures(OwlProfile::Rl, &rl)
    );
}

#[test]
fn el_profile_pass_rate_meets_threshold() {
    let results = run_corpus();
    let el = results.get(&OwlProfile::El).cloned().unwrap_or_default();
    let rate = pass_rate(&el);
    assert!(
        rate >= 0.95,
        "OWL 2 EL pass rate must be ≥ 0.95; got {rate:.2}\n{}",
        report_failures(OwlProfile::El, &el)
    );
}

#[test]
fn ql_profile_pass_rate_meets_threshold() {
    let results = run_corpus();
    let ql = results.get(&OwlProfile::Ql).cloned().unwrap_or_default();
    let rate = pass_rate(&ql);
    assert!(
        rate >= 0.95,
        "OWL 2 QL pass rate must be ≥ 0.95; got {rate:.2}\n{}",
        report_failures(OwlProfile::Ql, &ql)
    );
}

#[test]
fn rlel_profile_pass_rate_meets_threshold() {
    let results = run_corpus();
    let rlel = results.get(&OwlProfile::RlEl).cloned().unwrap_or_default();
    let rate = pass_rate(&rlel);
    assert!(
        rate >= 0.95,
        "OWL 2 RL+EL pass rate must be ≥ 0.95; got {rate:.2}\n{}",
        report_failures(OwlProfile::RlEl, &rlel)
    );
}

#[test]
fn dl_profile_pass_rate_meets_threshold() {
    let results = run_corpus();
    let dl = results.get(&OwlProfile::Dl).cloned().unwrap_or_default();
    let rate = pass_rate(&dl);
    assert!(
        rate >= 0.95,
        "OWL 2 DL pass rate must be ≥ 0.95; got {rate:.2}\n{}",
        report_failures(OwlProfile::Dl, &dl)
    );
}

impl Clone for CaseResult {
    fn clone(&self) -> Self {
        Self {
            case_id: self.case_id.clone(),
            profile: self.profile,
            pass: self.pass,
            missing: self.missing.clone(),
            violated: self.violated.clone(),
        }
    }
}
