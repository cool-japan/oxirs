//! Federation executor for SERVICE clause execution

use super::client::FederationClient;
use super::results::SparqlResultsParser;
use crate::model::Term;
use crate::query::{
    NamedNodePattern, SparqlGraphPattern as GraphPattern, SparqlTermPattern as TermPattern,
};
use crate::OxirsError;
use std::collections::HashMap;
use tracing::{debug, info};

/// Federation executor for executing SERVICE clauses
pub struct FederationExecutor {
    client: FederationClient,
}

impl FederationExecutor {
    /// Create a new federation executor
    pub fn new() -> Result<Self, OxirsError> {
        let client = FederationClient::new()?;
        Ok(Self { client })
    }

    /// Execute a SERVICE clause
    ///
    /// # Arguments
    /// * `endpoint` - The SERVICE endpoint (IRI or variable)
    /// * `pattern` - The graph pattern to execute at the endpoint
    /// * `silent` - If true, suppress errors and return empty results
    /// * `bindings` - Current variable bindings from local query
    ///
    /// # Returns
    /// Vector of solution bindings from the remote endpoint
    pub async fn execute_service(
        &self,
        endpoint: &NamedNodePattern,
        pattern: &GraphPattern,
        silent: bool,
        bindings: &[HashMap<String, Term>],
    ) -> Result<Vec<HashMap<String, Term>>, OxirsError> {
        // Extract endpoint URL
        let endpoint_url = match endpoint {
            NamedNodePattern::NamedNode(node) => node.as_str().to_string(),
            NamedNodePattern::Variable(_) => {
                return Err(OxirsError::Federation(
                    "Variable endpoints are not yet supported".to_string(),
                ))
            }
        };

        info!("Executing SERVICE clause at endpoint: {}", endpoint_url);
        debug!("Pattern: {:?}", pattern);
        debug!("Current bindings: {} solutions", bindings.len());

        // Convert graph pattern to SPARQL query string
        let sparql_query = self.pattern_to_sparql(pattern)?;
        debug!("Generated SPARQL query: {}", sparql_query);

        // Execute query at remote endpoint
        let json_response = self
            .client
            .execute_query(&endpoint_url, &sparql_query, silent)
            .await?;

        // Parse results
        let remote_bindings = SparqlResultsParser::parse(&json_response)?;

        info!(
            "Received {} solutions from remote endpoint",
            remote_bindings.len()
        );

        Ok(remote_bindings)
    }

    /// Convert a graph pattern to a SPARQL SELECT query
    fn pattern_to_sparql(&self, pattern: &GraphPattern) -> Result<String, OxirsError> {
        // Extract variables from the pattern
        let variables = self.extract_variables(pattern);

        // Build SELECT clause
        let select_clause = if variables.is_empty() {
            "SELECT *".to_string()
        } else {
            format!("SELECT {}", variables.join(" "))
        };

        // Convert pattern to WHERE clause
        let where_clause = self.pattern_to_where_clause(pattern)?;

        Ok(format!("{} WHERE {{ {} }}", select_clause, where_clause))
    }

    /// Extract variables from a graph pattern
    fn extract_variables(&self, pattern: &GraphPattern) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_variables(pattern, &mut vars);
        vars.sort();
        vars.dedup();
        vars.into_iter().map(|v| format!("?{}", v)).collect()
    }

    /// Recursively collect variables from pattern
    fn collect_variables(&self, pattern: &GraphPattern, vars: &mut Vec<String>) {
        match pattern {
            GraphPattern::Bgp { patterns } => {
                for tp in patterns {
                    // Extract variables from triple pattern
                    if let TermPattern::Variable(v) = &tp.subject {
                        vars.push(v.name().to_string());
                    }
                    if let TermPattern::Variable(v) = &tp.predicate {
                        vars.push(v.name().to_string());
                    }
                    if let TermPattern::Variable(v) = &tp.object {
                        vars.push(v.name().to_string());
                    }
                }
            }
            GraphPattern::Join { left, right } | GraphPattern::Union { left, right } => {
                self.collect_variables(left, vars);
                self.collect_variables(right, vars);
            }
            GraphPattern::Filter { inner, .. }
            | GraphPattern::Distinct { inner }
            | GraphPattern::Reduced { inner }
            | GraphPattern::Extend { inner, .. }
            | GraphPattern::Group { inner, .. }
            | GraphPattern::Project { inner, .. } => {
                self.collect_variables(inner, vars);
            }
            GraphPattern::LeftJoin { left, right, .. } => {
                self.collect_variables(left, vars);
                self.collect_variables(right, vars);
            }
            GraphPattern::Service { inner, .. } => {
                self.collect_variables(inner, vars);
            }
            _ => {}
        }
    }

    /// Convert graph pattern to WHERE clause string
    fn pattern_to_where_clause(&self, pattern: &GraphPattern) -> Result<String, OxirsError> {
        match pattern {
            GraphPattern::Bgp { patterns } => {
                let mut clauses = Vec::new();
                for tp in patterns {
                    let s = self.term_pattern_to_string(&tp.subject);
                    let p = self.term_pattern_to_string(&tp.predicate);
                    let o = self.term_pattern_to_string(&tp.object);
                    clauses.push(format!("{} {} {}", s, p, o));
                }
                Ok(clauses.join(" . "))
            }
            GraphPattern::Join { left, right } => {
                let left_str = self.pattern_to_where_clause(left)?;
                let right_str = self.pattern_to_where_clause(right)?;
                Ok(format!("{} . {}", left_str, right_str))
            }
            GraphPattern::Union { left, right } => {
                let left_str = self.pattern_to_where_clause(left)?;
                let right_str = self.pattern_to_where_clause(right)?;
                Ok(format!("{{ {} }} UNION {{ {} }}", left_str, right_str))
            }
            GraphPattern::Filter { expr: _, inner } => {
                let inner_str = self.pattern_to_where_clause(inner)?;
                // Simplified filter expression (full implementation would handle all expression types)
                Ok(format!("{} FILTER(?var)", inner_str))
            }
            _ => {
                // For other patterns, use a simplified representation
                Ok("?s ?p ?o".to_string())
            }
        }
    }

    /// Convert a term pattern to SPARQL string
    fn term_pattern_to_string(&self, term: &TermPattern) -> String {
        match term {
            TermPattern::Variable(v) => format!("?{}", v.name()),
            TermPattern::NamedNode(n) => format!("<{}>", n.as_str()),
            TermPattern::BlankNode(b) => format!("_:{}", b.as_str()),
            TermPattern::Literal(l) => {
                if let Some(lang) = l.language() {
                    format!("\"{}\"@{}", l.value(), lang)
                } else if l.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    format!("\"{}\"^^<{}>", l.value(), l.datatype().as_str())
                } else {
                    format!("\"{}\"", l.value())
                }
            }
            #[cfg(feature = "sparql-12")]
            TermPattern::Triple(triple) => {
                format!("<< {} {} {} >>",
                    self.term_pattern_to_string(&triple.subject),
                    self.term_pattern_to_string(&triple.predicate),
                    self.term_pattern_to_string(&triple.object))
            }
        }
    }

    /// Merge local and remote bindings
    pub fn merge_bindings(
        &self,
        local: Vec<HashMap<String, Term>>,
        remote: Vec<HashMap<String, Term>>,
    ) -> Vec<HashMap<String, Term>> {
        if local.is_empty() {
            return remote;
        }
        if remote.is_empty() {
            return local;
        }

        // Find common variables
        let local_vars: std::collections::HashSet<_> = local[0].keys().cloned().collect();
        let remote_vars: std::collections::HashSet<_> = remote[0].keys().cloned().collect();
        let common_vars: Vec<_> = local_vars.intersection(&remote_vars).cloned().collect();

        debug!(
            "Merging bindings with {} common variables",
            common_vars.len()
        );

        if common_vars.is_empty() {
            // Cartesian product if no common variables
            let mut result = Vec::new();
            for l in &local {
                for r in &remote {
                    let mut merged = l.clone();
                    merged.extend(r.clone());
                    result.push(merged);
                }
            }
            result
        } else {
            // Hash join on common variables
            let mut result = Vec::new();
            for l in &local {
                for r in &remote {
                    // Check if bindings are compatible
                    let mut compatible = true;
                    for var in &common_vars {
                        if let (Some(l_val), Some(r_val)) = (l.get(var), r.get(var)) {
                            if l_val != r_val {
                                compatible = false;
                                break;
                            }
                        }
                    }

                    if compatible {
                        let mut merged = l.clone();
                        merged.extend(r.clone());
                        result.push(merged);
                    }
                }
            }
            result
        }
    }
}

impl Default for FederationExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create default federation executor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::NamedNode;

    #[tokio::test]
    async fn test_executor_creation() {
        let executor = FederationExecutor::new();
        assert!(executor.is_ok());
    }

    #[test]
    fn test_merge_bindings_no_common_vars() {
        let executor = FederationExecutor::new().unwrap();

        let local = vec![{
            let mut m = HashMap::new();
            m.insert(
                "x".to_string(),
                Term::NamedNode(NamedNode::new("http://example.org/a").unwrap()),
            );
            m
        }];

        let remote = vec![{
            let mut m = HashMap::new();
            m.insert(
                "y".to_string(),
                Term::NamedNode(NamedNode::new("http://example.org/b").unwrap()),
            );
            m
        }];

        let result = executor.merge_bindings(local, remote);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
    }

    #[test]
    fn test_merge_bindings_with_common_vars() {
        let executor = FederationExecutor::new().unwrap();

        let node = Term::NamedNode(NamedNode::new("http://example.org/same").unwrap());

        let local = vec![{
            let mut m = HashMap::new();
            m.insert("x".to_string(), node.clone());
            m.insert(
                "y".to_string(),
                Term::NamedNode(NamedNode::new("http://example.org/a").unwrap()),
            );
            m
        }];

        let remote = vec![{
            let mut m = HashMap::new();
            m.insert("x".to_string(), node.clone());
            m.insert(
                "z".to_string(),
                Term::NamedNode(NamedNode::new("http://example.org/b").unwrap()),
            );
            m
        }];

        let result = executor.merge_bindings(local, remote);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3); // x, y, z
    }
}
