//! Domain types for the standalone SPARQL 1.1 Update protocol.
//!
//! This sibling module hosts the data structures used by the
//! [`update_protocol`](crate::update_protocol) facade: concrete triples,
//! pattern terms, triple patterns, the top-level [`SparqlUpdate`] enum,
//! and the result / error types returned by the parser and executor.

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// A concrete RDF triple (no variables).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub s: String,
    pub p: String,
    pub o: String,
}

impl Triple {
    /// Convenience constructor.
    pub fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            s: s.into(),
            p: p.into(),
            o: o.into(),
        }
    }
}

/// A position in a triple pattern – can be an IRI, a plain literal, a blank
/// node, or a variable (placeholder for pattern matching).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PatternTerm {
    Iri(String),
    Literal(String),
    Variable(String),
    BlankNode(String),
}

impl PatternTerm {
    /// Returns `true` when this term is a variable (used during template instantiation).
    pub fn is_variable(&self) -> bool {
        matches!(self, PatternTerm::Variable(_))
    }

    /// Returns the variable name if this is a `Variable` variant.
    pub fn variable_name(&self) -> Option<&str> {
        if let PatternTerm::Variable(name) = self {
            Some(name.as_str())
        } else {
            None
        }
    }
}

/// A triple pattern where any position may be a variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriplePattern {
    pub s: PatternTerm,
    pub p: PatternTerm,
    pub o: PatternTerm,
}

impl TriplePattern {
    /// Construct a new triple pattern.
    pub fn new(s: PatternTerm, p: PatternTerm, o: PatternTerm) -> Self {
        Self { s, o, p }
    }
}

// ---------------------------------------------------------------------------
// DROP / CLEAR target type
// ---------------------------------------------------------------------------

/// Scope qualifier for `DROP` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DropType {
    /// A specific named graph identified by IRI.
    Graph,
    /// The default graph.
    Default,
    /// All named graphs.
    Named,
    /// Every graph in the dataset (default + all named).
    All,
}

/// Scope qualifier for `CLEAR` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClearType {
    /// A specific named graph identified by IRI.
    Graph,
    /// The default graph.
    Default,
    /// All named graphs.
    Named,
    /// Every graph in the dataset.
    All,
}

// ---------------------------------------------------------------------------
// Top-level update enum
// ---------------------------------------------------------------------------

/// A single SPARQL 1.1 update operation.
#[derive(Debug, Clone, PartialEq)]
pub enum SparqlUpdate {
    /// `INSERT DATA { … }` — adds concrete triples to the default graph.
    InsertData(Vec<Triple>),

    /// `DELETE DATA { … }` — removes concrete triples from the default graph.
    DeleteData(Vec<Triple>),

    /// `INSERT { template } WHERE { where_clause }` — pattern-based insert.
    InsertWhere {
        template: Vec<TriplePattern>,
        where_clause: Vec<TriplePattern>,
    },

    /// `DELETE { template } WHERE { where_clause }` — pattern-based delete.
    DeleteWhere {
        template: Vec<TriplePattern>,
        where_clause: Vec<TriplePattern>,
    },

    /// `DELETE { delete } INSERT { insert } WHERE { where_clause }` — combined modify.
    Modify {
        delete: Vec<TriplePattern>,
        insert: Vec<TriplePattern>,
        where_clause: Vec<TriplePattern>,
    },

    /// `CREATE [SILENT] GRAPH <iri>`.
    CreateGraph { iri: String, silent: bool },

    /// `DROP [SILENT] (GRAPH <iri> | DEFAULT | NAMED | ALL)`.
    DropGraph {
        iri: Option<String>,
        silent: bool,
        drop_type: DropType,
    },

    /// `CLEAR [SILENT] (GRAPH <iri> | DEFAULT | NAMED | ALL)`.
    ClearGraph {
        iri: Option<String>,
        silent: bool,
        clear_type: ClearType,
    },

    /// `COPY [SILENT] <source> TO <target>`.
    CopyGraph {
        source: String,
        target: String,
        silent: bool,
    },

    /// `MOVE [SILENT] <source> TO <target>`.
    MoveGraph {
        source: String,
        target: String,
        silent: bool,
    },

    /// `ADD [SILENT] <source> TO <target>`.
    AddGraph {
        source: String,
        target: String,
        silent: bool,
    },

    /// `LOAD [SILENT] <iri> [INTO GRAPH <into>]`.
    Load {
        iri: String,
        into: Option<String>,
        silent: bool,
    },
}

// ---------------------------------------------------------------------------
// Parse error
// ---------------------------------------------------------------------------

/// Error returned by `SparqlUpdateParser`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    /// Byte offset inside the input string where the error was detected.
    pub position: usize,
}

impl ParseError {
    /// Construct a `ParseError` at the given byte position.
    pub(crate) fn at(position: usize, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            position,
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "parse error at position {}: {}",
            self.position, self.message
        )
    }
}

impl std::error::Error for ParseError {}

// ---------------------------------------------------------------------------
// UpdateResult
// ---------------------------------------------------------------------------

/// Summary of the changes made by a single `UpdateExecutor::execute` call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpdateResult {
    /// Number of triples inserted into the default graph or named graphs.
    pub triples_inserted: usize,
    /// Number of triples deleted from the default graph or named graphs.
    pub triples_deleted: usize,
    /// Number of distinct graphs affected (created, cleared, populated, etc.).
    pub graphs_affected: usize,
}

// ---------------------------------------------------------------------------
// ArqError (thin wrapper)
// ---------------------------------------------------------------------------

/// Error type for the update executor.
#[derive(Debug, Clone, PartialEq)]
pub struct ArqError(pub String);

impl std::fmt::Display for ArqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ARQ error: {}", self.0)
    }
}

impl std::error::Error for ArqError {}
