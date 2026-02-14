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
                    l1 == l2
                } else if let (Some(dt1), Some(dt2)) = (&l1.datatype, &l2.datatype) {
                    if self.is_numeric_datatype(dt1) && self.is_numeric_datatype(dt2) {
                        self.compare_numeric_literals(l1, l2) == Some(std::cmp::Ordering::Equal)
                    } else {
                        l1 == l2
                    }
                } else {
                    l1 == l2
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
