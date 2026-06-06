//! SPARQL-Generate AST types.
//!
//! Defines the abstract syntax tree for a `GENERATE { ... } WHERE { ... }` query,
//! following the W3C community specification at
//! <https://ci.mines-stetienne.fr/sparql-generate/>.

// ─────────────────────────────────────────────────────────────────────────────
// Literal
// ─────────────────────────────────────────────────────────────────────────────

/// A literal value in a GENERATE template — either a plain string fragment,
/// a SPARQL variable reference, or a string concatenation of inner literals.
#[derive(Debug, Clone, PartialEq)]
pub enum GenerateLiteral {
    /// A static string fragment that is emitted verbatim.
    Text(String),
    /// A variable reference whose value is substituted from SPARQL bindings.
    /// The `String` stores the variable name without the leading `?`.
    Var(String),
    /// A string concatenation of one or more inner `GenerateLiteral` values.
    /// Evaluated left-to-right; the sub-expressions may themselves be `Var`
    /// or `Text` nodes.
    Concat(Vec<GenerateLiteral>),
}

// ─────────────────────────────────────────────────────────────────────────────
// TemplateClause
// ─────────────────────────────────────────────────────────────────────────────

/// A single clause in the GENERATE template.
///
/// Each clause consists of:
/// - an optional *prefix* string emitted before the expression,
/// - a mandatory *expression* (`GenerateLiteral`), and
/// - an optional *suffix* string emitted after the expression.
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateClause {
    /// Static text emitted before the expression (may be `None`).
    pub prefix: Option<String>,
    /// The template expression — a literal, variable reference, or concat.
    pub expr: GenerateLiteral,
    /// Static text emitted after the expression (may be `None`).
    pub suffix: Option<String>,
}

impl TemplateClause {
    /// Convenience constructor for a clause with only an expression (no pre-/suffix).
    pub fn expr_only(expr: GenerateLiteral) -> Self {
        Self {
            prefix: None,
            expr,
            suffix: None,
        }
    }

    /// Convenience constructor for a clause with a leading prefix text and a variable.
    pub fn with_prefix(prefix: impl Into<String>, var: impl Into<String>) -> Self {
        Self {
            prefix: Some(prefix.into()),
            expr: GenerateLiteral::Var(var.into()),
            suffix: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GenerateQuery
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed SPARQL-Generate query (simplified subset of the full spec).
///
/// ```text
/// generate_query ::= prefix_decl* 'GENERATE' '{' template_clause* '}'
///                    ('ITERATOR' string_literal)?
///                    'WHERE' '{' where_body '}'
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateQuery {
    /// Namespace prefix declarations declared before the GENERATE keyword,
    /// e.g. `PREFIX ex: <http://example.org/>`.
    /// Each entry is `(prefix_label, iri)`, e.g. `("ex", "http://example.org/")`.
    pub prefix_decls: Vec<(String, String)>,

    /// Ordered list of template clauses inside the GENERATE `{ }` block.
    pub template: Vec<TemplateClause>,

    /// The raw WHERE clause body (everything between the `WHERE { }` braces),
    /// stored as a verbatim string. We do not re-implement full SPARQL parsing
    /// here — the WHERE body can be passed on to the SPARQL engine as-is.
    pub where_body: String,

    /// Optional ITERATOR expression (a SPARQL-Generate extension that drives
    /// iteration over an external source such as a JSON or CSV file).
    /// Stored as a raw string if present.
    pub iterator: Option<String>,
}

impl GenerateQuery {
    /// Create a new `GenerateQuery` with no prefix declarations and no iterator.
    pub fn new(template: Vec<TemplateClause>, where_body: impl Into<String>) -> Self {
        Self {
            prefix_decls: Vec::new(),
            template,
            where_body: where_body.into(),
            iterator: None,
        }
    }

    /// Return the number of template clauses.
    pub fn clause_count(&self) -> usize {
        self.template.len()
    }

    /// Return `true` if the GENERATE block contains no clauses.
    pub fn is_empty(&self) -> bool {
        self.template.is_empty()
    }
}
