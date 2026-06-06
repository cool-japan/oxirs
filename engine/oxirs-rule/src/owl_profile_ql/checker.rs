//! OWL 2 QL profile compliance checker implementation.
//!
//! Given a set of [`OntologyAxiom`]s, the [`Owl2QlProfileChecker`] reports
//! which (if any) axioms fall outside the OWL 2 QL profile and why.

use super::profile::{ClassExpr, OntologyAxiom};

/// Profile check result for a single axiom.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileAxiomReport {
    pub axiom_index: usize,
    pub is_compliant: bool,
    pub violation_reason: Option<String>,
}

/// Overall profile check report.
#[derive(Debug, Clone)]
pub struct ProfileReport {
    pub axioms_checked: usize,
    pub violations: Vec<ProfileAxiomReport>,
}

impl ProfileReport {
    /// True if every axiom in the input was OWL 2 QL compliant.
    pub fn is_ql_compliant(&self) -> bool {
        self.violations.is_empty()
    }

    /// Number of axioms that violated the OWL 2 QL profile.
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }

    /// Human-readable summary of the check result.
    pub fn summary(&self) -> String {
        if self.is_ql_compliant() {
            format!("OWL 2 QL compliant: {} axioms passed", self.axioms_checked)
        } else {
            format!(
                "OWL 2 QL non-compliant: {} of {} axioms violate the profile",
                self.violation_count(),
                self.axioms_checked
            )
        }
    }
}

/// OWL 2 QL profile compliance checker.
#[derive(Default)]
pub struct Owl2QlProfileChecker;

impl Owl2QlProfileChecker {
    /// Construct a new checker.
    pub fn new() -> Self {
        Self
    }

    /// Check a slice of axioms and return a [`ProfileReport`].
    pub fn check(&self, axioms: &[OntologyAxiom]) -> ProfileReport {
        let mut violations = Vec::new();
        for (idx, axiom) in axioms.iter().enumerate() {
            let report = self.check_axiom(idx, axiom);
            if !report.is_compliant {
                violations.push(report);
            }
        }
        ProfileReport {
            axioms_checked: axioms.len(),
            violations,
        }
    }

    fn check_axiom(&self, idx: usize, axiom: &OntologyAxiom) -> ProfileAxiomReport {
        let (is_compliant, reason) = match axiom {
            OntologyAxiom::SubClassOf { sub, sup } => {
                if !sub.is_ql_sub_class_expression() {
                    (
                        false,
                        Some("sub class expression not in QL form".to_string()),
                    )
                } else if !sup.is_ql_super_class_expression() {
                    (
                        false,
                        Some("super class expression not in QL form".to_string()),
                    )
                } else {
                    (true, None)
                }
            }
            OntologyAxiom::SubObjectPropertyChain { .. } => {
                (false, Some("property chains not allowed in QL".into()))
            }
            OntologyAxiom::TransitiveObjectProperty(_) => (
                false,
                Some("TransitiveObjectProperty not allowed in QL".into()),
            ),
            OntologyAxiom::FunctionalObjectProperty(_) => (
                false,
                Some("FunctionalObjectProperty not allowed in QL".into()),
            ),
            OntologyAxiom::InverseFunctionalObjectProperty(_) => (
                false,
                Some("InverseFunctionalObjectProperty not allowed in QL".into()),
            ),
            OntologyAxiom::FunctionalDataProperty(_) => (
                false,
                Some("FunctionalDataProperty not allowed in QL".into()),
            ),
            OntologyAxiom::DisjointUnion { .. } => {
                (false, Some("DisjointUnion not allowed in QL".into()))
            }
            OntologyAxiom::EquivalentClasses(parts) => {
                let all_ok = parts.iter().all(|c: &ClassExpr| {
                    c.is_ql_sub_class_expression() && c.is_ql_super_class_expression()
                });
                if all_ok {
                    (true, None)
                } else {
                    (
                        false,
                        Some("EquivalentClasses must be QL-compliant on both sides".into()),
                    )
                }
            }
            OntologyAxiom::ObjectPropertyDomain { domain, .. }
            | OntologyAxiom::ObjectPropertyRange {
                range: domain, ..
            }
            | OntologyAxiom::DataPropertyDomain { domain, .. } => {
                if domain.is_ql_super_class_expression() {
                    (true, None)
                } else {
                    (
                        false,
                        Some("property domain/range must be QL superClassExpression".into()),
                    )
                }
            }
            OntologyAxiom::ClassAssertion { class, .. } => {
                if class.is_atomic() {
                    (true, None)
                } else {
                    (false, Some("ClassAssertion must use atomic class".into()))
                }
            }
            // All others (SubObjectPropertyOf, InverseObjectProperties, symmetry/asymmetry/
            // reflexivity/irreflexivity, DisjointObjectProperties, DataPropertyRange,
            // ObjectPropertyAssertion, DataPropertyAssertion, NegativeObjectPropertyAssertion)
            // are permitted by the OWL 2 QL profile without extra structural restrictions.
            _ => (true, None),
        };

        ProfileAxiomReport {
            axiom_index: idx,
            is_compliant,
            violation_reason: reason,
        }
    }
}
