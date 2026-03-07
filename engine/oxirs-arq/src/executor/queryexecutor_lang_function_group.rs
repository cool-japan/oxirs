//! # QueryExecutor - lang_function_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

pub use super::dataset::{
    convert_property_path, ConcreteStoreDataset, Dataset, DatasetPathAdapter, InMemoryDataset,
};
use crate::algebra::Solution;
use anyhow::Result;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Built-in LANG function
    pub(super) fn lang_function(&self, arg: &crate::algebra::Term) -> Result<crate::algebra::Term> {
        match arg {
            crate::algebra::Term::Literal(lit) => {
                let lang = lit.language.as_ref().unwrap_or(&String::new()).clone();
                Ok(crate::algebra::Term::Literal(crate::algebra::Literal {
                    value: lang,
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#string",
                    )),
                }))
            }
            _ => Err(anyhow::anyhow!("LANG function only applicable to literals")),
        }
    }
    /// Apply distinct to solution
    pub(super) fn apply_distinct(&self, solution: Solution) -> Solution {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        let mut result = Solution::new();
        for binding in solution {
            let key: Vec<_> = binding
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            if seen.insert(key) {
                result.push(binding);
            }
        }
        result
    }
    /// Execute pattern using dataset to find actual matches
    pub(super) fn execute_pattern_with_dataset(
        &self,
        pattern: &crate::algebra::TriplePattern,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        let mut solution = Solution::new();
        let triples = dataset.find_triples(pattern)?;
        for (s, p, o) in triples {
            let mut binding = crate::algebra::Binding::new();
            if let crate::algebra::Term::Variable(var) = &pattern.subject {
                binding.insert(var.clone(), s);
            }
            if let crate::algebra::Term::Variable(var) = &pattern.predicate {
                binding.insert(var.clone(), p);
            }
            if let crate::algebra::Term::Variable(var) = &pattern.object {
                binding.insert(var.clone(), o);
            }
            if !binding.is_empty() {
                solution.push(binding);
            } else {
                solution.push(crate::algebra::Binding::new());
            }
        }
        Ok(solution)
    }
    /// Create sample binding for pattern
    #[allow(dead_code)]
    pub(super) fn create_sample_binding(
        &self,
        pattern: &crate::algebra::TriplePattern,
    ) -> Result<Solution> {
        let mut solution = Solution::new();
        let mut binding = crate::algebra::Binding::new();
        if let crate::algebra::Term::Variable(var) = &pattern.subject {
            binding.insert(
                var.clone(),
                crate::algebra::Term::Iri(
                    oxirs_core::model::NamedNode::new("http://example.org/subject")
                        .expect("hardcoded IRI should be valid"),
                ),
            );
        }
        if let crate::algebra::Term::Variable(var) = &pattern.object {
            binding.insert(
                var.clone(),
                crate::algebra::Term::Iri(
                    oxirs_core::model::NamedNode::new("http://example.org/object")
                        .expect("hardcoded IRI should be valid"),
                ),
            );
        }
        if !binding.is_empty() {
            solution.push(binding);
        }
        Ok(solution)
    }
}
