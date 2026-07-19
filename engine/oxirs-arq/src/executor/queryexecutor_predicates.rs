//! # QueryExecutor - predicates Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Check if two terms are equal according to SPARQL semantics
    pub(super) fn terms_equal(
        &self,
        left: &crate::algebra::Term,
        right: &crate::algebra::Term,
    ) -> bool {
        use crate::algebra::Term;
        match (left, right) {
            (Term::Literal(l1), Term::Literal(l2)) => {
                if l1.language.is_some() || l2.language.is_some() {
                    // A language-tagged literal only equals another with the
                    // identical lexical form and tag.
                    l1 == l2
                } else {
                    let numeric1 = l1
                        .datatype
                        .as_ref()
                        .is_some_and(|dt| self.is_numeric_datatype(dt));
                    let numeric2 = l2
                        .datatype
                        .as_ref()
                        .is_some_and(|dt| self.is_numeric_datatype(dt));
                    if numeric1 && numeric2 {
                        self.compare_numeric_literals(l1, l2) == Some(std::cmp::Ordering::Equal)
                    } else {
                        // In RDF 1.1 a simple literal (no datatype) is an
                        // xsd:string, so a missing datatype and an explicit
                        // xsd:string share one value space. This lets a typed
                        // xsd:string function result such as LANG(?l) / STR(?x)
                        // compare equal to a query string literal, which the
                        // parser leaves untyped.
                        const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
                        let dt1 = l1.datatype.as_ref().map_or(XSD_STRING, |d| d.as_str());
                        let dt2 = l2.datatype.as_ref().map_or(XSD_STRING, |d| d.as_str());
                        l1.value == l2.value && dt1 == dt2
                    }
                }
            }
            _ => left == right,
        }
    }
    /// Check if a datatype IRI represents a numeric type
    pub(super) fn is_numeric_datatype(&self, datatype: &oxirs_core::model::NamedNode) -> bool {
        matches!(
            datatype.as_str(),
            "http://www.w3.org/2001/XMLSchema#integer"
                | "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#double"
                | "http://www.w3.org/2001/XMLSchema#float"
                | "http://www.w3.org/2001/XMLSchema#int"
                | "http://www.w3.org/2001/XMLSchema#long"
                | "http://www.w3.org/2001/XMLSchema#short"
                | "http://www.w3.org/2001/XMLSchema#byte"
                | "http://www.w3.org/2001/XMLSchema#unsignedInt"
                | "http://www.w3.org/2001/XMLSchema#unsignedLong"
                | "http://www.w3.org/2001/XMLSchema#unsignedShort"
                | "http://www.w3.org/2001/XMLSchema#unsignedByte"
                | "http://www.w3.org/2001/XMLSchema#positiveInteger"
                | "http://www.w3.org/2001/XMLSchema#nonPositiveInteger"
                | "http://www.w3.org/2001/XMLSchema#negativeInteger"
                | "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
        )
    }
    /// Compare numeric literals
    pub(super) fn compare_numeric_literals(
        &self,
        left: &crate::algebra::Literal,
        right: &crate::algebra::Literal,
    ) -> Option<std::cmp::Ordering> {
        let left_val = left.value.parse::<f64>().ok()?;
        let right_val = right.value.parse::<f64>().ok()?;
        left_val.partial_cmp(&right_val)
    }
}
