//! SPARQL-Generate executor.
//!
//! Evaluates a `GenerateQuery` against one or more rows of SPARQL variable
//! bindings, expanding template expressions to produce text output.

use std::collections::HashMap;

use super::ast::{GenerateLiteral, GenerateQuery, TemplateClause};
use super::GenerateError;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A set of SPARQL variable bindings for one solution row.
///
/// Keys are variable names **without** a leading `?`; values are the RDF term
/// strings as they would appear in a SPARQL result (e.g. `"Alice"`,
/// `<http://example.org/Alice>`, or `"42"^^xsd:integer`).
pub type Bindings = HashMap<String, String>;

/// The result of evaluating a GENERATE template for a single solution row.
#[derive(Debug)]
pub struct GenerateResult {
    /// The generated text produced by substituting bindings into the template.
    pub text: String,
    /// The number of variable bindings that were actually used during evaluation
    /// of this row (i.e., variable references that resolved to a value).
    pub binding_count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// GenerateExecutor
// ─────────────────────────────────────────────────────────────────────────────

/// Executes a parsed `GenerateQuery` over a set of SPARQL variable bindings.
///
/// # Example
///
/// ```rust
/// use std::collections::HashMap;
/// use oxirs_arq::generate::{GenerateExecutor, Bindings, GenerateQuery, TemplateClause, GenerateLiteral};
///
/// let clause = TemplateClause {
///     prefix: Some("name=".to_string()),
///     expr: GenerateLiteral::Var("name".to_string()),
///     suffix: None,
/// };
/// let query = GenerateQuery::new(vec![clause], "?s foaf:name ?name .");
/// let exec  = GenerateExecutor::new(query);
///
/// let mut row = HashMap::new();
/// row.insert("name".to_string(), "Alice".to_string());
///
/// let result = exec.evaluate_one(&row).unwrap();
/// assert_eq!(result.text, "name=Alice");
/// ```
pub struct GenerateExecutor {
    /// The parsed GENERATE query to execute.
    pub query: GenerateQuery,
}

impl GenerateExecutor {
    /// Create a new executor for the given `GenerateQuery`.
    pub fn new(query: GenerateQuery) -> Self {
        Self { query }
    }

    // ── Single-row evaluation ────────────────────────────────────────────────

    /// Evaluate the GENERATE template against a single solution row.
    ///
    /// Returns `GenerateResult` containing the concatenated text output and the
    /// count of distinct bindings used during evaluation.
    ///
    /// # Errors
    ///
    /// Returns `GenerateError::UnboundVariable` if a `Var` reference in the
    /// template is not present in `bindings`.
    pub fn evaluate_one(&self, bindings: &Bindings) -> Result<GenerateResult, GenerateError> {
        let mut parts = Vec::new();
        let mut used_vars: std::collections::HashSet<&str> = std::collections::HashSet::new();

        for clause in &self.query.template {
            let text = self.eval_clause(clause, bindings, &mut used_vars)?;
            parts.push(text);
        }

        Ok(GenerateResult {
            text: parts.concat(),
            binding_count: used_vars.len(),
        })
    }

    // ── Multi-row evaluation ─────────────────────────────────────────────────

    /// Evaluate the GENERATE template over multiple solution rows, collecting
    /// one `GenerateResult` per row.
    ///
    /// # Errors
    ///
    /// Propagates any error returned by `evaluate_one`.
    pub fn evaluate_all(&self, rows: &[Bindings]) -> Result<Vec<GenerateResult>, GenerateError> {
        rows.iter().map(|row| self.evaluate_one(row)).collect()
    }

    /// Concatenate all generated texts (one per row) into a single `String`,
    /// with `separator` between each adjacent pair.
    ///
    /// # Errors
    ///
    /// Propagates any error returned by `evaluate_all`.
    pub fn generate_text(
        &self,
        rows: &[Bindings],
        separator: &str,
    ) -> Result<String, GenerateError> {
        let results = self.evaluate_all(rows)?;
        let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
        Ok(texts.join(separator))
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Evaluate a single `TemplateClause` given `bindings`, appending variable
    /// names that are resolved into `used_vars`.
    fn eval_clause<'b>(
        &self,
        clause: &TemplateClause,
        bindings: &'b Bindings,
        used_vars: &mut std::collections::HashSet<&'b str>,
    ) -> Result<String, GenerateError> {
        let mut buf = String::new();

        if let Some(prefix) = &clause.prefix {
            buf.push_str(prefix);
        }

        buf.push_str(&self.eval_literal_tracked(&clause.expr, bindings, used_vars)?);

        if let Some(suffix) = &clause.suffix {
            buf.push_str(suffix);
        }

        Ok(buf)
    }

    /// Evaluate a `GenerateLiteral`, tracking which variables are used.
    fn eval_literal_tracked<'b>(
        &self,
        lit: &GenerateLiteral,
        bindings: &'b Bindings,
        used_vars: &mut std::collections::HashSet<&'b str>,
    ) -> Result<String, GenerateError> {
        match lit {
            GenerateLiteral::Text(s) => Ok(s.clone()),

            GenerateLiteral::Var(name) => {
                let value = bindings
                    .get(name.as_str())
                    .ok_or_else(|| GenerateError::UnboundVariable(name.clone()))?;
                // Track that this variable was resolved.
                // Safety: the key in `bindings` lives at least as long as `bindings`.
                if let Some(key) = bindings.keys().find(|k| k.as_str() == name.as_str()) {
                    used_vars.insert(key.as_str());
                }
                Ok(value.clone())
            }

            GenerateLiteral::Concat(parts) => {
                let mut result = String::new();
                for part in parts {
                    result.push_str(&self.eval_literal_tracked(part, bindings, used_vars)?);
                }
                Ok(result)
            }
        }
    }

    /// Evaluate a single `GenerateLiteral` given bindings.
    ///
    /// This is the public-facing variant without variable tracking; it is
    /// useful for unit-testing individual literal expressions.
    pub fn eval_literal(
        &self,
        lit: &GenerateLiteral,
        bindings: &Bindings,
    ) -> Result<String, GenerateError> {
        let mut used = std::collections::HashSet::new();
        self.eval_literal_tracked(lit, bindings, &mut used)
    }
}
