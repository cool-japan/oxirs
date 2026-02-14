//! SPARQL Query representation
//!
//! Extracted and adapted from OxiGraph spargebra with OxiRS enhancements.
//! Based on W3C SPARQL 1.1 Query specification:
//! <https://www.w3.org/TR/sparql11-query/>

use super::sparql_algebra::{GraphPattern, TriplePattern};
use crate::model::{NamedNode, Variable};
use std::fmt;

/// A parsed [SPARQL query](https://www.w3.org/TR/sparql11-query/).
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Query {
    /// [SELECT](https://www.w3.org/TR/sparql11-query/#select).
    Select {
        /// The [query dataset specification](https://www.w3.org/TR/sparql11-query/#specifyingDataset).
        dataset: Option<QueryDataset>,
        /// The query selection graph pattern.
        pattern: GraphPattern,
        /// The query base IRI.
        base_iri: Option<String>,
    },
    /// [CONSTRUCT](https://www.w3.org/TR/sparql11-query/#construct).
    Construct {
        /// The query construction template.
        template: Vec<TriplePattern>,
        /// The [query dataset specification](https://www.w3.org/TR/sparql11-query/#specifyingDataset).
        dataset: Option<QueryDataset>,
        /// The query selection graph pattern.
        pattern: GraphPattern,
        /// The query base IRI.
        base_iri: Option<String>,
    },
    /// [DESCRIBE](https://www.w3.org/TR/sparql11-query/#describe).
    Describe {
        /// The [query dataset specification](https://www.w3.org/TR/sparql11-query/#specifyingDataset).
        dataset: Option<QueryDataset>,
        /// The query selection graph pattern.
        pattern: GraphPattern,
        /// The query base IRI.
        base_iri: Option<String>,
    },
    /// [ASK](https://www.w3.org/TR/sparql11-query/#ask).
    Ask {
        /// The [query dataset specification](https://www.w3.org/TR/sparql11-query/#specifyingDataset).
        dataset: Option<QueryDataset>,
        /// The query selection graph pattern.
        pattern: GraphPattern,
        /// The query base IRI.
        base_iri: Option<String>,
    },
}

impl Query {
    /// Creates a new SELECT query
    pub fn select(pattern: GraphPattern) -> Self {
        Self::Select {
            dataset: None,
            pattern,
            base_iri: None,
        }
    }

    /// Creates a new CONSTRUCT query
    pub fn construct(template: Vec<TriplePattern>, pattern: GraphPattern) -> Self {
        Self::Construct {
            template,
            dataset: None,
            pattern,
            base_iri: None,
        }
    }

    /// Creates a new ASK query
    pub fn ask(pattern: GraphPattern) -> Self {
        Self::Ask {
            dataset: None,
            pattern,
            base_iri: None,
        }
    }

    /// Creates a new DESCRIBE query
    pub fn describe(pattern: GraphPattern) -> Self {
        Self::Describe {
            dataset: None,
            pattern,
            base_iri: None,
        }
    }

    /// Returns the dataset specification for this query
    pub fn dataset(&self) -> Option<&QueryDataset> {
        match self {
            Query::Select { dataset, .. }
            | Query::Construct { dataset, .. }
            | Query::Describe { dataset, .. }
            | Query::Ask { dataset, .. } => dataset.as_ref(),
        }
    }

    /// Returns a mutable reference to the dataset specification for this query
    pub fn dataset_mut(&mut self) -> Option<&mut QueryDataset> {
        match self {
            Query::Select { dataset, .. }
            | Query::Construct { dataset, .. }
            | Query::Describe { dataset, .. }
            | Query::Ask { dataset, .. } => dataset.as_mut(),
        }
    }

    /// Returns the base IRI for this query
    pub fn base_iri(&self) -> Option<&str> {
        match self {
            Query::Select { base_iri, .. }
            | Query::Construct { base_iri, .. }
            | Query::Describe { base_iri, .. }
            | Query::Ask { base_iri, .. } => base_iri.as_deref(),
        }
    }

    /// Returns the graph pattern for this query
    pub fn pattern(&self) -> &GraphPattern {
        match self {
            Query::Select { pattern, .. }
            | Query::Construct { pattern, .. }
            | Query::Describe { pattern, .. }
            | Query::Ask { pattern, .. } => pattern,
        }
    }

    /// Sets the base IRI for this query
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Self {
        let base_iri = Some(base_iri.into());
        match &mut self {
            Query::Select { base_iri: iri, .. }
            | Query::Construct { base_iri: iri, .. }
            | Query::Describe { base_iri: iri, .. }
            | Query::Ask { base_iri: iri, .. } => *iri = base_iri,
        }
        self
    }

    /// Sets the dataset for this query
    pub fn with_dataset(mut self, dataset: QueryDataset) -> Self {
        let dataset = Some(dataset);
        match &mut self {
            Query::Select { dataset: ds, .. }
            | Query::Construct { dataset: ds, .. }
            | Query::Describe { dataset: ds, .. }
            | Query::Ask { dataset: ds, .. } => *ds = dataset,
        }
        self
    }

    /// Formats using the SPARQL S-Expression syntax
    pub fn to_sse(&self) -> String {
        let mut buffer = String::new();
        self.fmt_sse(&mut buffer)
            .expect("writing to String should not fail");
        buffer
    }

    /// Formats using the SPARQL S-Expression syntax
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::Select {
                dataset,
                pattern,
                base_iri,
            } => {
                if let Some(base_iri) = base_iri {
                    write!(f, "(base <{base_iri}> ")?;
                }
                if let Some(dataset) = dataset {
                    f.write_str("(dataset ")?;
                    dataset.fmt_sse(f)?;
                    f.write_str(" ")?;
                }
                pattern.fmt_sse(f)?;
                if dataset.is_some() {
                    f.write_str(")")?;
                }
                if base_iri.is_some() {
                    f.write_str(")")?;
                }
                Ok(())
            }
            Self::Construct {
                template,
                dataset,
                pattern,
                base_iri,
            } => {
                if let Some(base_iri) = base_iri {
                    write!(f, "(base <{base_iri}> ")?;
                }
                f.write_str("(construct (")?;
                for (i, triple) in template.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    triple.fmt_sse(f)?;
                }
                f.write_str(") ")?;
                if let Some(dataset) = dataset {
                    f.write_str("(dataset ")?;
                    dataset.fmt_sse(f)?;
                    f.write_str(" ")?;
                }
                pattern.fmt_sse(f)?;
                if dataset.is_some() {
                    f.write_str(")")?;
                }
                f.write_str(")")?;
                if base_iri.is_some() {
                    f.write_str(")")?;
                }
                Ok(())
            }
            Self::Describe {
                dataset,
                pattern,
                base_iri,
            } => {
                if let Some(base_iri) = base_iri {
                    write!(f, "(base <{base_iri}> ")?;
                }
                f.write_str("(describe ")?;
                if let Some(dataset) = dataset {
                    f.write_str("(dataset ")?;
                    dataset.fmt_sse(f)?;
                    f.write_str(" ")?;
                }
                pattern.fmt_sse(f)?;
                if dataset.is_some() {
                    f.write_str(")")?;
                }
                f.write_str(")")?;
                if base_iri.is_some() {
                    f.write_str(")")?;
                }
                Ok(())
            }
            Self::Ask {
                dataset,
                pattern,
                base_iri,
            } => {
                if let Some(base_iri) = base_iri {
                    write!(f, "(base <{base_iri}> ")?;
                }
                f.write_str("(ask ")?;
                if let Some(dataset) = dataset {
                    f.write_str("(dataset ")?;
                    dataset.fmt_sse(f)?;
                    f.write_str(" ")?;
                }
                pattern.fmt_sse(f)?;
                if dataset.is_some() {
                    f.write_str(")")?;
                }
                f.write_str(")")?;
                if base_iri.is_some() {
                    f.write_str(")")?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Select {
                dataset,
                pattern,
                base_iri,
            } => {
                if let Some(base_iri) = base_iri {
                    writeln!(f, "BASE <{base_iri}>")?;
                }
                write!(
                    f,
                    "{}",
                    SparqlGraphRootPattern::new(pattern, dataset.as_ref())
                )
            }
            Self::Construct {
                template,
                dataset,
                pattern,
                base_iri,
            } => {
                if let Some(base_iri) = base_iri {
                    writeln!(f, "BASE <{base_iri}>")?;
                }
                f.write_str("CONSTRUCT { ")?;
                for triple in template {
                    write!(f, "{triple} . ")?;
                }
                f.write_str("}")?;
                if let Some(dataset) = dataset {
                    dataset.fmt(f)?;
                }
                write!(
                    f,
                    " WHERE {{ {} }}",
                    SparqlGraphRootPattern::new(pattern, None)
                )
            }
            Self::Describe {
                dataset,
                pattern,
                base_iri,
            } => {
                if let Some(base_iri) = base_iri {
                    writeln!(f, "BASE <{base_iri}>")?;
                }
                f.write_str("DESCRIBE *")?;
                if let Some(dataset) = dataset {
                    dataset.fmt(f)?;
                }
                write!(
                    f,
                    " WHERE {{ {} }}",
                    SparqlGraphRootPattern::new(pattern, None)
                )
            }
            Self::Ask {
                dataset,
                pattern,
                base_iri,
            } => {
                if let Some(base_iri) = base_iri {
                    writeln!(f, "BASE <{base_iri}>")?;
                }
                f.write_str("ASK")?;
                if let Some(dataset) = dataset {
                    dataset.fmt(f)?;
                }
                write!(
                    f,
                    " WHERE {{ {} }}",
                    SparqlGraphRootPattern::new(pattern, None)
                )
            }
        }
    }
}

/// A SPARQL query [dataset specification](https://www.w3.org/TR/sparql11-query/#specifyingDataset).
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QueryDataset {
    /// Default graphs (FROM clauses)
    pub default: Vec<NamedNode>,
    /// Named graphs (FROM NAMED clauses)
    pub named: Option<Vec<NamedNode>>,
}

impl QueryDataset {
    /// Creates a new empty dataset
    pub fn new() -> Self {
        Self {
            default: Vec::new(),
            named: None,
        }
    }

    /// Creates a dataset with default graphs
    pub fn with_default_graphs(graphs: Vec<NamedNode>) -> Self {
        Self {
            default: graphs,
            named: None,
        }
    }

    /// Creates a dataset with named graphs
    pub fn with_named_graphs(graphs: Vec<NamedNode>) -> Self {
        Self {
            default: Vec::new(),
            named: Some(graphs),
        }
    }

    /// Adds a default graph
    pub fn add_default_graph(&mut self, graph: NamedNode) {
        self.default.push(graph);
    }

    /// Adds a named graph
    pub fn add_named_graph(&mut self, graph: NamedNode) {
        if let Some(ref mut named) = self.named {
            named.push(graph);
        } else {
            self.named = Some(vec![graph]);
        }
    }

    /// Returns true if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.default.is_empty() && self.named.as_ref().map_or(true, |v| v.is_empty())
    }

    /// Formats using the SPARQL S-Expression syntax
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        f.write_str("(")?;
        for (i, graph_name) in self.default.iter().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{graph_name}")?;
        }
        if let Some(named) = &self.named {
            for (i, graph_name) in named.iter().enumerate() {
                if !self.default.is_empty() || i > 0 {
                    f.write_str(" ")?;
                }
                write!(f, "(named {graph_name})")?;
            }
        }
        f.write_str(")")
    }
}

impl Default for QueryDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for QueryDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for g in &self.default {
            write!(f, " FROM {g}")?;
        }
        if let Some(named) = &self.named {
            for g in named {
                write!(f, " FROM NAMED {g}")?;
            }
        }
        Ok(())
    }
}

/// Helper struct for formatting graph patterns in SPARQL syntax
pub struct SparqlGraphRootPattern<'a> {
    pattern: &'a GraphPattern,
    dataset: Option<&'a QueryDataset>,
}

impl<'a> SparqlGraphRootPattern<'a> {
    pub fn new(pattern: &'a GraphPattern, dataset: Option<&'a QueryDataset>) -> Self {
        Self { pattern, dataset }
    }
}

impl fmt::Display for SparqlGraphRootPattern<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut distinct = false;
        let mut reduced = false;
        let mut order = None;
        let mut start = 0;
        let mut length = None;
        let mut project: &[Variable] = &[];

        let mut child = self.pattern;
        loop {
            match child {
                GraphPattern::OrderBy { inner, expression } => {
                    order = Some(expression);
                    child = inner;
                }
                GraphPattern::Project { inner, variables } => {
                    project = variables;
                    child = inner;
                }
                GraphPattern::Distinct { inner } => {
                    distinct = true;
                    child = inner;
                }
                GraphPattern::Reduced { inner } => {
                    reduced = true;
                    child = inner;
                }
                GraphPattern::Slice {
                    inner,
                    start: s,
                    length: l,
                } => {
                    start = *s;
                    length = *l;
                    child = inner;
                }
                _ => break,
            }
        }

        if project.is_empty() {
            f.write_str("SELECT ")?;
        } else {
            f.write_str("SELECT ")?;
            if distinct {
                f.write_str("DISTINCT ")?;
            } else if reduced {
                f.write_str("REDUCED ")?;
            }
            for (i, var) in project.iter().enumerate() {
                if i > 0 {
                    f.write_str(" ")?;
                }
                write!(f, "{var}")?;
            }
        }

        if let Some(dataset) = self.dataset {
            dataset.fmt(f)?;
        }

        write!(f, " WHERE {{ {} }}", SparqlInnerGraphPattern::new(child))?;

        if let Some(order) = order {
            f.write_str(" ORDER BY")?;
            for expr in order {
                write!(f, " {expr}")?;
            }
        }

        if start > 0 {
            write!(f, " OFFSET {start}")?;
        }

        if let Some(length) = length {
            write!(f, " LIMIT {length}")?;
        }

        Ok(())
    }
}

/// Helper struct for formatting inner graph patterns
struct SparqlInnerGraphPattern<'a> {
    pattern: &'a GraphPattern,
}

impl<'a> SparqlInnerGraphPattern<'a> {
    fn new(pattern: &'a GraphPattern) -> Self {
        Self { pattern }
    }
}

impl fmt::Display for SparqlInnerGraphPattern<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.pattern {
            GraphPattern::Bgp { patterns } => {
                for (i, pattern) in patterns.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" . ")?;
                    }
                    write!(f, "{pattern}")?;
                }
                Ok(())
            }
            GraphPattern::Path {
                subject,
                path,
                object,
            } => {
                write!(f, "{subject} {path} {object}")
            }
            GraphPattern::Join { left, right } => {
                write!(
                    f,
                    "{} . {}",
                    SparqlInnerGraphPattern::new(left),
                    SparqlInnerGraphPattern::new(right)
                )
            }
            GraphPattern::LeftJoin {
                left,
                right,
                expression,
            } => {
                write!(
                    f,
                    "{} OPTIONAL {{ {}",
                    SparqlInnerGraphPattern::new(left),
                    SparqlInnerGraphPattern::new(right)
                )?;
                if let Some(expr) = expression {
                    write!(f, " FILTER ({expr})")?;
                }
                f.write_str(" }")
            }
            GraphPattern::Filter { expr, inner } => {
                write!(
                    f,
                    "{} FILTER ({})",
                    SparqlInnerGraphPattern::new(inner),
                    expr
                )
            }
            GraphPattern::Union { left, right } => {
                write!(
                    f,
                    "{{ {} }} UNION {{ {} }}",
                    SparqlInnerGraphPattern::new(left),
                    SparqlInnerGraphPattern::new(right)
                )
            }
            GraphPattern::Graph { name, inner } => {
                write!(
                    f,
                    "GRAPH {} {{ {} }}",
                    name,
                    SparqlInnerGraphPattern::new(inner)
                )
            }
            GraphPattern::Extend {
                inner,
                variable,
                expression,
            } => {
                write!(
                    f,
                    "{} BIND ({} AS {})",
                    SparqlInnerGraphPattern::new(inner),
                    expression,
                    variable
                )
            }
            GraphPattern::Minus { left, right } => {
                write!(
                    f,
                    "{} MINUS {{ {} }}",
                    SparqlInnerGraphPattern::new(left),
                    SparqlInnerGraphPattern::new(right)
                )
            }
            GraphPattern::Values {
                variables,
                bindings,
            } => {
                f.write_str("VALUES ")?;
                if variables.len() == 1 {
                    write!(f, "{}", variables[0])?;
                } else {
                    f.write_str("(")?;
                    for (i, var) in variables.iter().enumerate() {
                        if i > 0 {
                            f.write_str(" ")?;
                        }
                        write!(f, "{var}")?;
                    }
                    f.write_str(")")?;
                }
                f.write_str(" { ")?;
                for (i, binding) in bindings.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    if variables.len() == 1 {
                        if let Some(term) = &binding[0] {
                            write!(f, "{term}")?;
                        } else {
                            f.write_str("UNDEF")?;
                        }
                    } else {
                        f.write_str("(")?;
                        for (j, value) in binding.iter().enumerate() {
                            if j > 0 {
                                f.write_str(" ")?;
                            }
                            if let Some(term) = value {
                                write!(f, "{term}")?;
                            } else {
                                f.write_str("UNDEF")?;
                            }
                        }
                        f.write_str(")")?;
                    }
                }
                f.write_str(" }")
            }
            GraphPattern::Service {
                name,
                inner,
                silent,
            } => {
                if *silent {
                    write!(
                        f,
                        "SERVICE SILENT {} {{ {} }}",
                        name,
                        SparqlInnerGraphPattern::new(inner)
                    )
                } else {
                    write!(
                        f,
                        "SERVICE {} {{ {} }}",
                        name,
                        SparqlInnerGraphPattern::new(inner)
                    )
                }
            }
            GraphPattern::Group {
                inner,
                variables: _,
                aggregates: _,
            } => {
                // For display purposes, just show the inner pattern
                // The GROUP BY clause is handled at a higher level
                write!(f, "{}", SparqlInnerGraphPattern::new(inner))
            }
            // These should be handled at the root level
            GraphPattern::Project { inner, .. }
            | GraphPattern::Distinct { inner }
            | GraphPattern::Reduced { inner }
            | GraphPattern::Slice { inner, .. }
            | GraphPattern::OrderBy { inner, .. } => {
                write!(f, "{}", SparqlInnerGraphPattern::new(inner))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NamedNode, Variable};
    use crate::query::sparql_algebra::TermPattern;

    #[test]
    fn test_query_creation() {
        let var = Variable::new("s").unwrap();
        let pattern = GraphPattern::Bgp {
            patterns: vec![TriplePattern::new(
                TermPattern::Variable(var.clone()),
                TermPattern::Variable(Variable::new("p").unwrap()),
                TermPattern::Variable(Variable::new("o").unwrap()),
            )],
        };

        let query = Query::select(pattern);
        assert!(matches!(query, Query::Select { .. }));

        let sse = query.to_sse();
        assert!(sse.contains("bgp"));
    }

    #[test]
    fn test_dataset() {
        let mut dataset = QueryDataset::new();
        assert!(dataset.is_empty());

        let graph = NamedNode::new("http://example.org/graph").unwrap();
        dataset.add_default_graph(graph.clone());
        dataset.add_named_graph(graph);

        assert!(!dataset.is_empty());
        assert_eq!(dataset.default.len(), 1);
        assert_eq!(dataset.named.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_query_display() {
        let pattern = GraphPattern::Bgp {
            patterns: vec![TriplePattern::new(
                TermPattern::Variable(Variable::new("s").unwrap()),
                TermPattern::Variable(Variable::new("p").unwrap()),
                TermPattern::Variable(Variable::new("o").unwrap()),
            )],
        };

        let query = Query::select(pattern);
        let query_str = query.to_string();

        assert!(query_str.contains("SELECT"));
        assert!(query_str.contains("WHERE"));
        assert!(query_str.contains("?s"));
    }
}
