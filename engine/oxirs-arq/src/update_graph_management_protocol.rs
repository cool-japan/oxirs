//! SPARQL 1.1 UPDATE Graph Management — Protocol and Request Parsing
//!
//! HTTP/SPARQL protocol layer: parses raw SPARQL Update graph management
//! request strings into [`GraphManagementOp`] values, and serialises
//! [`GraphManagementResult`] into HTTP-compatible response structures.

use thiserror::Error;

use crate::update_graph_management_types::{
    GraphManagementOp, GraphManagementResult, GraphManagementTarget,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the graph-management protocol layer.
#[derive(Debug, Error)]
pub enum GraphManagementProtocolError {
    /// The SPARQL Update string could not be parsed as a graph management op.
    #[error("parse error: {0}")]
    Parse(String),

    /// A required IRI argument was missing or malformed.
    #[error("invalid IRI: {0}")]
    InvalidIri(String),
}

// ---------------------------------------------------------------------------
// HTTP response representation
// ---------------------------------------------------------------------------

/// HTTP-level response from a graph management update request.
#[derive(Debug, Clone)]
pub struct GraphManagementHttpResponse {
    /// HTTP status code (200 OK or 4xx/5xx on failure).
    pub status_code: u16,
    /// Human-readable message body.
    pub body: String,
    /// Structured result (if the operation succeeded).
    pub result: Option<GraphManagementResult>,
}

impl GraphManagementHttpResponse {
    /// Build a success response.
    pub fn ok(result: GraphManagementResult) -> Self {
        let body = format!(
            "OK: {} triples affected, {} graphs affected",
            result.triples_affected,
            result.graphs_affected.len()
        );
        Self {
            status_code: 200,
            body,
            result: Some(result),
        }
    }

    /// Build a failure response.
    pub fn error(status_code: u16, message: impl Into<String>) -> Self {
        Self {
            status_code,
            body: message.into(),
            result: None,
        }
    }

    /// Returns `true` if the HTTP status indicates success (2xx).
    pub fn is_success(&self) -> bool {
        self.status_code >= 200 && self.status_code < 300
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Minimal parser for SPARQL 1.1 graph management update strings.
///
/// Supports the following subset of the SPARQL 1.1 Update grammar:
///
/// ```text
/// GraphManagement  ::=  Load | Clear | Drop | Create | Copy | Move | Add
/// Load             ::=  'LOAD' 'SILENT'? IRIref ( 'INTO' 'GRAPH' IRIref )?
/// Clear            ::=  'CLEAR' 'SILENT'? GraphRefAll
/// Drop             ::=  'DROP' 'SILENT'? GraphRefAll
/// Create           ::=  'CREATE' 'SILENT'? 'GRAPH' IRIref
/// Copy             ::=  'COPY' 'SILENT'? GraphOrDefault 'TO' GraphOrDefault
/// Move             ::=  'MOVE' 'SILENT'? GraphOrDefault 'TO' GraphOrDefault
/// Add              ::=  'ADD' 'SILENT'? GraphOrDefault 'TO' GraphOrDefault
/// GraphRefAll      ::=  'DEFAULT' | 'NAMED' | 'ALL' | 'GRAPH' IRIref
/// GraphOrDefault   ::=  'DEFAULT' | 'GRAPH'? IRIref
/// ```
pub struct GraphManagementParser;

impl GraphManagementParser {
    /// Parse a SPARQL 1.1 UPDATE graph management statement.
    ///
    /// Input tokens are compared case-insensitively.  IRIs must be enclosed in
    /// angle brackets (`<iri>`).
    ///
    /// # Errors
    ///
    /// Returns [`GraphManagementProtocolError::Parse`] when the input does not
    /// match any known graph management operation.
    pub fn parse(input: &str) -> Result<GraphManagementOp, GraphManagementProtocolError> {
        let tokens: Vec<&str> = input.split_whitespace().collect();
        if tokens.is_empty() {
            return Err(GraphManagementProtocolError::Parse(
                "empty input".to_string(),
            ));
        }

        let keyword = tokens[0].to_uppercase();
        match keyword.as_str() {
            "LOAD" => Self::parse_load(&tokens[1..]),
            "CLEAR" => Self::parse_clear(&tokens[1..]),
            "DROP" => Self::parse_drop(&tokens[1..]),
            "CREATE" => Self::parse_create(&tokens[1..]),
            "COPY" => Self::parse_copy(&tokens[1..]),
            "MOVE" => Self::parse_move(&tokens[1..]),
            "ADD" => Self::parse_add(&tokens[1..]),
            other => Err(GraphManagementProtocolError::Parse(format!(
                "unknown graph management keyword: {other}"
            ))),
        }
    }

    // -----------------------------------------------------------------------
    // Operation-specific parsers
    // -----------------------------------------------------------------------

    fn parse_load(tokens: &[&str]) -> Result<GraphManagementOp, GraphManagementProtocolError> {
        let mut pos = 0;
        let silent = Self::consume_silent(tokens, &mut pos);

        let iri = Self::consume_iri(tokens, &mut pos)?;

        let into_graph = if pos < tokens.len()
            && tokens[pos].to_uppercase() == "INTO"
            && pos + 1 < tokens.len()
            && tokens[pos + 1].to_uppercase() == "GRAPH"
        {
            pos += 2; // consume INTO GRAPH
            Some(Self::consume_iri(tokens, &mut pos)?)
        } else {
            None
        };

        Ok(GraphManagementOp::Load {
            iri,
            into_graph,
            silent,
        })
    }

    fn parse_clear(tokens: &[&str]) -> Result<GraphManagementOp, GraphManagementProtocolError> {
        let mut pos = 0;
        let silent = Self::consume_silent(tokens, &mut pos);
        let target = Self::consume_graph_ref_all(tokens, &mut pos)?;
        Ok(GraphManagementOp::Clear { target, silent })
    }

    fn parse_drop(tokens: &[&str]) -> Result<GraphManagementOp, GraphManagementProtocolError> {
        let mut pos = 0;
        let silent = Self::consume_silent(tokens, &mut pos);
        let target = Self::consume_graph_ref_all(tokens, &mut pos)?;
        Ok(GraphManagementOp::Drop { target, silent })
    }

    fn parse_create(tokens: &[&str]) -> Result<GraphManagementOp, GraphManagementProtocolError> {
        let mut pos = 0;
        let silent = Self::consume_silent(tokens, &mut pos);

        if pos < tokens.len() && tokens[pos].to_uppercase() == "GRAPH" {
            pos += 1;
        }

        let graph = Self::consume_iri(tokens, &mut pos)?;
        Ok(GraphManagementOp::Create { graph, silent })
    }

    fn parse_copy(tokens: &[&str]) -> Result<GraphManagementOp, GraphManagementProtocolError> {
        let mut pos = 0;
        let silent = Self::consume_silent(tokens, &mut pos);
        let source = Self::consume_graph_or_default(tokens, &mut pos)?;

        if pos >= tokens.len() || tokens[pos].to_uppercase() != "TO" {
            return Err(GraphManagementProtocolError::Parse(
                "expected TO after source graph in COPY".to_string(),
            ));
        }
        pos += 1;

        let destination = Self::consume_graph_or_default(tokens, &mut pos)?;
        Ok(GraphManagementOp::Copy {
            source,
            destination,
            silent,
        })
    }

    fn parse_move(tokens: &[&str]) -> Result<GraphManagementOp, GraphManagementProtocolError> {
        let mut pos = 0;
        let silent = Self::consume_silent(tokens, &mut pos);
        let source = Self::consume_graph_or_default(tokens, &mut pos)?;

        if pos >= tokens.len() || tokens[pos].to_uppercase() != "TO" {
            return Err(GraphManagementProtocolError::Parse(
                "expected TO after source graph in MOVE".to_string(),
            ));
        }
        pos += 1;

        let destination = Self::consume_graph_or_default(tokens, &mut pos)?;
        Ok(GraphManagementOp::Move {
            source,
            destination,
            silent,
        })
    }

    fn parse_add(tokens: &[&str]) -> Result<GraphManagementOp, GraphManagementProtocolError> {
        let mut pos = 0;
        let silent = Self::consume_silent(tokens, &mut pos);
        let source = Self::consume_graph_or_default(tokens, &mut pos)?;

        if pos >= tokens.len() || tokens[pos].to_uppercase() != "TO" {
            return Err(GraphManagementProtocolError::Parse(
                "expected TO after source graph in ADD".to_string(),
            ));
        }
        pos += 1;

        let destination = Self::consume_graph_or_default(tokens, &mut pos)?;
        Ok(GraphManagementOp::Add {
            source,
            destination,
            silent,
        })
    }

    // -----------------------------------------------------------------------
    // Token helpers
    // -----------------------------------------------------------------------

    /// Consume the optional `SILENT` keyword and advance `pos`.
    fn consume_silent(tokens: &[&str], pos: &mut usize) -> bool {
        if *pos < tokens.len() && tokens[*pos].to_uppercase() == "SILENT" {
            *pos += 1;
            true
        } else {
            false
        }
    }

    /// Consume an `<iri>` token (must start with `<` and end with `>`).
    fn consume_iri(
        tokens: &[&str],
        pos: &mut usize,
    ) -> Result<String, GraphManagementProtocolError> {
        if *pos >= tokens.len() {
            return Err(GraphManagementProtocolError::Parse(
                "expected IRI but found end of input".to_string(),
            ));
        }

        let token = tokens[*pos];
        if token.starts_with('<') && token.ends_with('>') && token.len() >= 2 {
            *pos += 1;
            Ok(token[1..token.len() - 1].to_owned())
        } else {
            Err(GraphManagementProtocolError::InvalidIri(format!(
                "expected <iri>, got: {token}"
            )))
        }
    }

    /// Parse a `GraphRefAll` production: `DEFAULT | NAMED | ALL | GRAPH <iri>`.
    fn consume_graph_ref_all(
        tokens: &[&str],
        pos: &mut usize,
    ) -> Result<GraphManagementTarget, GraphManagementProtocolError> {
        if *pos >= tokens.len() {
            return Err(GraphManagementProtocolError::Parse(
                "expected graph reference but found end of input".to_string(),
            ));
        }

        match tokens[*pos].to_uppercase().as_str() {
            "DEFAULT" => {
                *pos += 1;
                Ok(GraphManagementTarget::Default)
            }
            "NAMED" => {
                *pos += 1;
                Ok(GraphManagementTarget::AllNamed)
            }
            "ALL" => {
                *pos += 1;
                Ok(GraphManagementTarget::All)
            }
            "GRAPH" => {
                *pos += 1;
                let iri = Self::consume_iri(tokens, pos)?;
                Ok(GraphManagementTarget::Named(iri))
            }
            other => Err(GraphManagementProtocolError::Parse(format!(
                "expected DEFAULT | NAMED | ALL | GRAPH <iri>, got: {other}"
            ))),
        }
    }

    /// Parse a `GraphOrDefault` production: `DEFAULT | GRAPH? <iri>`.
    fn consume_graph_or_default(
        tokens: &[&str],
        pos: &mut usize,
    ) -> Result<GraphManagementTarget, GraphManagementProtocolError> {
        if *pos >= tokens.len() {
            return Err(GraphManagementProtocolError::Parse(
                "expected graph or DEFAULT but found end of input".to_string(),
            ));
        }

        match tokens[*pos].to_uppercase().as_str() {
            "DEFAULT" => {
                *pos += 1;
                Ok(GraphManagementTarget::Default)
            }
            "GRAPH" => {
                *pos += 1;
                let iri = Self::consume_iri(tokens, pos)?;
                Ok(GraphManagementTarget::Named(iri))
            }
            _ => {
                // Bare IRI (the GRAPH keyword is optional in GraphOrDefault)
                let iri = Self::consume_iri(tokens, pos)?;
                Ok(GraphManagementTarget::Named(iri))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Combines [`GraphManagementParser`] with
/// [`GraphManagementExecutor`](crate::update_graph_management_ops::GraphManagementExecutor)
/// into a single HTTP-level request handler.
pub struct GraphManagementRequestHandler;

impl GraphManagementRequestHandler {
    /// Parse and execute a SPARQL Update graph management statement.
    ///
    /// On success returns a 200 response; on parse error returns 400; on
    /// runtime error returns 500.
    pub fn handle(
        input: &str,
        dataset: &mut crate::update_graph_management_types::GraphManagementDataset,
    ) -> GraphManagementHttpResponse {
        let op = match GraphManagementParser::parse(input) {
            Ok(op) => op,
            Err(e) => {
                return GraphManagementHttpResponse::error(400, format!("Bad Request: {e}"));
            }
        };

        match crate::update_graph_management_ops::GraphManagementExecutor::execute(&op, dataset) {
            Ok(result) => GraphManagementHttpResponse::ok(result),
            Err(e) => {
                GraphManagementHttpResponse::error(500, format!("Internal Server Error: {e}"))
            }
        }
    }
}
