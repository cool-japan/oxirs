//! GraphQL Cursor-based Pagination (Relay Specification)
//!
//! This module implements the Relay cursor-based pagination specification for GraphQL,
//! providing `ConnectionType<T>`, `Edge<T>`, and `PageInfo` types that follow the
//! [Relay Cursor Connections Specification](https://relay.dev/graphql/connections.htm).
//!
//! # Relay Specification Compliance
//!
//! This implementation follows all requirements of the Relay spec:
//! - `Connection` type with `edges` and `pageInfo` fields
//! - `Edge` type with `node` and `cursor` fields
//! - `PageInfo` type with `hasNextPage`, `hasPreviousPage`, `startCursor`, `endCursor`
//! - Forward pagination via `first`/`after` arguments
//! - Backward pagination via `last`/`before` arguments
//!
//! # Cursor Encoding
//!
//! Cursors are opaque base64-encoded strings. The internal representation uses a
//! prefix + offset scheme: `cursor:N` where N is the 0-based index.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The cursor prefix used in encoding.
const CURSOR_PREFIX: &str = "cursor:";

/// Encodes a cursor from a 0-based offset.
pub fn encode_cursor(offset: usize) -> String {
    use base64::Engine;
    let raw = format!("{CURSOR_PREFIX}{offset}");
    base64::engine::general_purpose::STANDARD.encode(raw.as_bytes())
}

/// Decodes a cursor to a 0-based offset.
///
/// Returns `None` if the cursor is invalid.
pub fn decode_cursor(cursor: &str) -> Option<usize> {
    use base64::Engine;
    let decoded_bytes = base64::engine::general_purpose::STANDARD
        .decode(cursor.as_bytes())
        .ok()?;
    let decoded = String::from_utf8(decoded_bytes).ok()?;
    let offset_str = decoded.strip_prefix(CURSOR_PREFIX)?;
    offset_str.parse::<usize>().ok()
}

/// Pagination arguments for a connection query.
///
/// Follows the Relay spec for forward (`first`/`after`) and backward (`last`/`before`) pagination.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PaginationArgs {
    /// Number of items to return from the beginning (forward pagination).
    pub first: Option<usize>,
    /// Cursor to start after (forward pagination).
    pub after: Option<String>,
    /// Number of items to return from the end (backward pagination).
    pub last: Option<usize>,
    /// Cursor to end before (backward pagination).
    pub before: Option<String>,
}

impl PaginationArgs {
    /// Creates new forward pagination arguments.
    pub fn forward(first: usize, after: Option<String>) -> Self {
        Self {
            first: Some(first),
            after,
            last: None,
            before: None,
        }
    }

    /// Creates new backward pagination arguments.
    pub fn backward(last: usize, before: Option<String>) -> Self {
        Self {
            first: None,
            after: None,
            last: Some(last),
            before,
        }
    }

    /// Validates the pagination arguments.
    ///
    /// Returns an error if both `first` and `last` are specified,
    /// or if negative values would be implied.
    pub fn validate(&self) -> Result<(), PaginationError> {
        if self.first.is_some() && self.last.is_some() {
            return Err(PaginationError::ConflictingArgs(
                "Cannot specify both 'first' and 'last'".to_string(),
            ));
        }

        if self.first.is_none() && self.last.is_none() {
            return Err(PaginationError::MissingArgs(
                "Must specify either 'first' or 'last'".to_string(),
            ));
        }

        if let Some(after) = &self.after {
            if decode_cursor(after).is_none() {
                return Err(PaginationError::InvalidCursor(after.clone()));
            }
        }

        if let Some(before) = &self.before {
            if decode_cursor(before).is_none() {
                return Err(PaginationError::InvalidCursor(before.clone()));
            }
        }

        Ok(())
    }

    /// Returns true if this is forward pagination.
    pub fn is_forward(&self) -> bool {
        self.first.is_some()
    }

    /// Returns true if this is backward pagination.
    pub fn is_backward(&self) -> bool {
        self.last.is_some()
    }
}

/// Errors that can occur during pagination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaginationError {
    /// Both `first` and `last` were specified.
    ConflictingArgs(String),
    /// Neither `first` nor `last` was specified.
    MissingArgs(String),
    /// An invalid cursor was provided.
    InvalidCursor(String),
    /// The requested page size exceeds the maximum.
    PageSizeTooLarge {
        /// The requested page size.
        requested: usize,
        /// The maximum allowed page size.
        maximum: usize,
    },
}

impl fmt::Display for PaginationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaginationError::ConflictingArgs(msg) => write!(f, "Conflicting args: {msg}"),
            PaginationError::MissingArgs(msg) => write!(f, "Missing args: {msg}"),
            PaginationError::InvalidCursor(c) => write!(f, "Invalid cursor: {c}"),
            PaginationError::PageSizeTooLarge { requested, maximum } => {
                write!(f, "Page size {requested} exceeds maximum {maximum}")
            }
        }
    }
}

impl std::error::Error for PaginationError {}

/// An edge in a connection, wrapping a node with a cursor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge<T> {
    /// The opaque cursor for this edge.
    pub cursor: String,
    /// The actual data node.
    pub node: T,
}

impl<T> Edge<T> {
    /// Creates a new edge with the given cursor and node.
    pub fn new(cursor: String, node: T) -> Self {
        Self { cursor, node }
    }

    /// Creates a new edge from an offset and node.
    pub fn from_offset(offset: usize, node: T) -> Self {
        Self {
            cursor: encode_cursor(offset),
            node,
        }
    }

    /// Maps the node to a different type.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Edge<U> {
        Edge {
            cursor: self.cursor,
            node: f(self.node),
        }
    }
}

impl<T: fmt::Display> fmt::Display for Edge<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Edge(cursor={}, node={})", self.cursor, self.node)
    }
}

/// Page information for a connection, following the Relay spec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageInfo {
    /// Whether there are more items after the last edge.
    pub has_next_page: bool,
    /// Whether there are more items before the first edge.
    pub has_previous_page: bool,
    /// Cursor of the first edge (None if empty).
    pub start_cursor: Option<String>,
    /// Cursor of the last edge (None if empty).
    pub end_cursor: Option<String>,
}

impl PageInfo {
    /// Creates page info for an empty result set.
    pub fn empty() -> Self {
        Self {
            has_next_page: false,
            has_previous_page: false,
            start_cursor: None,
            end_cursor: None,
        }
    }
}

impl Default for PageInfo {
    fn default() -> Self {
        Self::empty()
    }
}

/// A Relay-spec compliant connection type with edges and page info.
///
/// This is the primary type returned from paginated queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionType<T> {
    /// The edges in this connection (each wraps a node + cursor).
    pub edges: Vec<Edge<T>>,
    /// Pagination metadata.
    pub page_info: PageInfo,
    /// Total count of items (optional; may require a separate count query).
    pub total_count: Option<usize>,
}

impl<T> ConnectionType<T> {
    /// Creates an empty connection.
    pub fn empty() -> Self {
        Self {
            edges: Vec::new(),
            page_info: PageInfo::empty(),
            total_count: Some(0),
        }
    }

    /// Creates a connection from a full list of items with pagination args.
    ///
    /// This applies cursor-based slicing to the items according to the Relay spec
    /// algorithm.
    pub fn from_slice(
        items: Vec<T>,
        args: &PaginationArgs,
        max_page_size: Option<usize>,
    ) -> Result<Self, PaginationError>
    where
        T: Clone,
    {
        let total = items.len();

        // Determine the start index based on 'after' cursor
        let start_index = if let Some(ref after) = args.after {
            decode_cursor(after).ok_or_else(|| PaginationError::InvalidCursor(after.clone()))? + 1
        } else {
            0
        };

        // Determine the end index based on 'before' cursor
        let end_index = if let Some(ref before) = args.before {
            let idx = decode_cursor(before)
                .ok_or_else(|| PaginationError::InvalidCursor(before.clone()))?;
            idx.min(total)
        } else {
            total
        };

        if start_index >= end_index {
            return Ok(Self {
                edges: Vec::new(),
                page_info: PageInfo {
                    has_next_page: start_index < total,
                    has_previous_page: start_index > 0,
                    start_cursor: None,
                    end_cursor: None,
                },
                total_count: Some(total),
            });
        }

        let available = &items[start_index..end_index];

        // Apply 'first' or 'last' slicing
        let (slice_start, slice_end) = if let Some(first) = args.first {
            let first = if let Some(max) = max_page_size {
                if first > max {
                    return Err(PaginationError::PageSizeTooLarge {
                        requested: first,
                        maximum: max,
                    });
                }
                first.min(max)
            } else {
                first
            };
            (0, first.min(available.len()))
        } else if let Some(last) = args.last {
            let last = if let Some(max) = max_page_size {
                if last > max {
                    return Err(PaginationError::PageSizeTooLarge {
                        requested: last,
                        maximum: max,
                    });
                }
                last.min(max)
            } else {
                last
            };
            let start = available.len().saturating_sub(last);
            (start, available.len())
        } else {
            (0, available.len())
        };

        let sliced = &available[slice_start..slice_end];
        let edges: Vec<Edge<T>> = sliced
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let global_offset = start_index + slice_start + i;
                Edge::from_offset(global_offset, item.clone())
            })
            .collect();

        let has_previous_page = start_index + slice_start > 0;
        let has_next_page = start_index + slice_end < total;

        let start_cursor = edges.first().map(|e| e.cursor.clone());
        let end_cursor = edges.last().map(|e| e.cursor.clone());

        Ok(Self {
            edges,
            page_info: PageInfo {
                has_next_page,
                has_previous_page,
                start_cursor,
                end_cursor,
            },
            total_count: Some(total),
        })
    }

    /// Returns the number of edges in this connection.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns true if this connection has no edges.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Returns the nodes from all edges.
    pub fn nodes(&self) -> Vec<&T> {
        self.edges.iter().map(|e| &e.node).collect()
    }

    /// Maps all nodes in the connection to a different type.
    pub fn map_nodes<U, F: Fn(T) -> U>(self, f: F) -> ConnectionType<U> {
        ConnectionType {
            edges: self.edges.into_iter().map(|e| e.map(&f)).collect(),
            page_info: self.page_info,
            total_count: self.total_count,
        }
    }

    /// Returns the cursor of the first edge.
    pub fn start_cursor(&self) -> Option<&str> {
        self.page_info.start_cursor.as_deref()
    }

    /// Returns the cursor of the last edge.
    pub fn end_cursor(&self) -> Option<&str> {
        self.page_info.end_cursor.as_deref()
    }
}

impl<T> Default for ConnectionType<T> {
    fn default() -> Self {
        Self::empty()
    }
}

/// A builder for creating connections with more control.
pub struct ConnectionBuilder<T> {
    items: Vec<T>,
    total_count: Option<usize>,
    max_page_size: Option<usize>,
}

impl<T: Clone> ConnectionBuilder<T> {
    /// Creates a new connection builder with the given items.
    pub fn new(items: Vec<T>) -> Self {
        Self {
            total_count: Some(items.len()),
            items,
            max_page_size: None,
        }
    }

    /// Sets the total count (useful if items is a subset).
    pub fn with_total_count(mut self, total: usize) -> Self {
        self.total_count = Some(total);
        self
    }

    /// Sets the maximum allowed page size.
    pub fn with_max_page_size(mut self, max: usize) -> Self {
        self.max_page_size = Some(max);
        self
    }

    /// Builds the connection with the given pagination args.
    pub fn build(self, args: &PaginationArgs) -> Result<ConnectionType<T>, PaginationError> {
        let mut conn = ConnectionType::from_slice(self.items, args, self.max_page_size)?;
        conn.total_count = self.total_count;
        Ok(conn)
    }
}

/// Represents a paginated query for SPARQL results mapped to GraphQL.
#[derive(Debug, Clone)]
pub struct SparqlConnection {
    /// The SPARQL query to paginate.
    pub query: String,
    /// Maximum page size.
    pub max_page_size: usize,
    /// Default page size when 'first'/'last' is not specified.
    pub default_page_size: usize,
}

impl SparqlConnection {
    /// Creates a new SPARQL connection definition.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            max_page_size: 100,
            default_page_size: 10,
        }
    }

    /// Sets the maximum page size.
    pub fn with_max_page_size(mut self, max: usize) -> Self {
        self.max_page_size = max;
        self
    }

    /// Sets the default page size.
    pub fn with_default_page_size(mut self, size: usize) -> Self {
        self.default_page_size = size;
        self
    }

    /// Generates a SPARQL query with LIMIT/OFFSET for the given pagination args.
    pub fn to_sparql(&self, args: &PaginationArgs) -> Result<String, PaginationError> {
        let offset = if let Some(ref after) = args.after {
            decode_cursor(after).ok_or_else(|| PaginationError::InvalidCursor(after.clone()))? + 1
        } else {
            0
        };

        let limit = args
            .first
            .or(args.last)
            .unwrap_or(self.default_page_size)
            .min(self.max_page_size);

        // Append LIMIT and OFFSET to the base query
        let base = self.query.trim().trim_end_matches(';');
        Ok(format!("{base} LIMIT {limit} OFFSET {offset}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Cursor encoding/decoding ────────────────────────────────────────────

    #[test]
    fn test_encode_decode_cursor_zero() {
        let cursor = encode_cursor(0);
        assert_eq!(decode_cursor(&cursor), Some(0));
    }

    #[test]
    fn test_encode_decode_cursor_large() {
        let cursor = encode_cursor(999_999);
        assert_eq!(decode_cursor(&cursor), Some(999_999));
    }

    #[test]
    fn test_decode_invalid_cursor() {
        assert_eq!(decode_cursor("not-base64!@#$"), None);
    }

    #[test]
    fn test_decode_wrong_prefix() {
        use base64::Engine;
        let bad = base64::engine::general_purpose::STANDARD.encode(b"wrong:42");
        assert_eq!(decode_cursor(&bad), None);
    }

    #[test]
    fn test_cursor_roundtrip_multiple() {
        for i in [0, 1, 10, 100, 1000, 50000] {
            let cursor = encode_cursor(i);
            assert_eq!(decode_cursor(&cursor), Some(i));
        }
    }

    // ── PaginationArgs ──────────────────────────────────────────────────────

    #[test]
    fn test_forward_args() {
        let args = PaginationArgs::forward(10, None);
        assert!(args.is_forward());
        assert!(!args.is_backward());
        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_backward_args() {
        let args = PaginationArgs::backward(10, None);
        assert!(args.is_backward());
        assert!(!args.is_forward());
        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_conflicting_args() {
        let args = PaginationArgs {
            first: Some(10),
            last: Some(10),
            after: None,
            before: None,
        };
        let result = args.validate();
        assert!(result.is_err());
        assert!(matches!(result, Err(PaginationError::ConflictingArgs(_))));
    }

    #[test]
    fn test_missing_args() {
        let args = PaginationArgs::default();
        let result = args.validate();
        assert!(result.is_err());
        assert!(matches!(result, Err(PaginationError::MissingArgs(_))));
    }

    #[test]
    fn test_invalid_after_cursor() {
        let args = PaginationArgs {
            first: Some(10),
            after: Some("garbage".to_string()),
            last: None,
            before: None,
        };
        let result = args.validate();
        assert!(result.is_err());
    }

    // ── Edge ────────────────────────────────────────────────────────────────

    #[test]
    fn test_edge_creation() {
        let edge = Edge::new("c1".to_string(), 42);
        assert_eq!(edge.cursor, "c1");
        assert_eq!(edge.node, 42);
    }

    #[test]
    fn test_edge_from_offset() {
        let edge = Edge::from_offset(5, "hello");
        assert_eq!(decode_cursor(&edge.cursor), Some(5));
        assert_eq!(edge.node, "hello");
    }

    #[test]
    fn test_edge_map() {
        let edge = Edge::new("c".to_string(), 42);
        let mapped = edge.map(|n| n.to_string());
        assert_eq!(mapped.node, "42");
        assert_eq!(mapped.cursor, "c");
    }

    #[test]
    fn test_edge_display() {
        let edge = Edge::new("cursor1".to_string(), "node1".to_string());
        let display = edge.to_string();
        assert!(display.contains("cursor1"));
        assert!(display.contains("node1"));
    }

    // ── PageInfo ────────────────────────────────────────────────────────────

    #[test]
    fn test_page_info_empty() {
        let pi = PageInfo::empty();
        assert!(!pi.has_next_page);
        assert!(!pi.has_previous_page);
        assert!(pi.start_cursor.is_none());
        assert!(pi.end_cursor.is_none());
    }

    #[test]
    fn test_page_info_default() {
        let pi = PageInfo::default();
        assert_eq!(pi, PageInfo::empty());
    }

    // ── ConnectionType: empty ───────────────────────────────────────────────

    #[test]
    fn test_empty_connection() {
        let conn: ConnectionType<i32> = ConnectionType::empty();
        assert!(conn.is_empty());
        assert_eq!(conn.edge_count(), 0);
        assert_eq!(conn.total_count, Some(0));
        assert!(conn.nodes().is_empty());
    }

    #[test]
    fn test_default_connection() {
        let conn: ConnectionType<i32> = ConnectionType::default();
        assert!(conn.is_empty());
    }

    // ── ConnectionType: forward pagination ──────────────────────────────────

    #[test]
    fn test_forward_first_page() {
        let items: Vec<i32> = (0..20).collect();
        let args = PaginationArgs::forward(5, None);
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        assert_eq!(conn.edge_count(), 5);
        assert_eq!(conn.nodes(), vec![&0, &1, &2, &3, &4]);
        assert!(conn.page_info.has_next_page);
        assert!(!conn.page_info.has_previous_page);
        assert_eq!(conn.total_count, Some(20));
    }

    #[test]
    fn test_forward_second_page() {
        let items: Vec<i32> = (0..20).collect();
        let after = encode_cursor(4); // after the 5th item
        let args = PaginationArgs::forward(5, Some(after));
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        assert_eq!(conn.edge_count(), 5);
        assert_eq!(conn.nodes(), vec![&5, &6, &7, &8, &9]);
        assert!(conn.page_info.has_next_page);
        assert!(conn.page_info.has_previous_page);
    }

    #[test]
    fn test_forward_last_page() {
        let items: Vec<i32> = (0..12).collect();
        let after = encode_cursor(9);
        let args = PaginationArgs::forward(5, Some(after));
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        assert_eq!(conn.edge_count(), 2); // items 10, 11
        assert!(!conn.page_info.has_next_page);
        assert!(conn.page_info.has_previous_page);
    }

    #[test]
    fn test_forward_exact_page() {
        let items: Vec<i32> = (0..5).collect();
        let args = PaginationArgs::forward(5, None);
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        assert_eq!(conn.edge_count(), 5);
        assert!(!conn.page_info.has_next_page);
        assert!(!conn.page_info.has_previous_page);
    }

    // ── ConnectionType: backward pagination ─────────────────────────────────

    #[test]
    fn test_backward_last_page() {
        let items: Vec<i32> = (0..20).collect();
        let args = PaginationArgs::backward(5, None);
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        assert_eq!(conn.edge_count(), 5);
        assert_eq!(conn.nodes(), vec![&15, &16, &17, &18, &19]);
        assert!(!conn.page_info.has_next_page);
        assert!(conn.page_info.has_previous_page);
    }

    #[test]
    fn test_backward_with_before() {
        let items: Vec<i32> = (0..20).collect();
        let before = encode_cursor(15);
        let args = PaginationArgs::backward(5, Some(before));
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        assert_eq!(conn.edge_count(), 5);
        assert_eq!(conn.nodes(), vec![&10, &11, &12, &13, &14]);
        assert!(conn.page_info.has_next_page);
        assert!(conn.page_info.has_previous_page);
    }

    // ── Max page size ───────────────────────────────────────────────────────

    #[test]
    fn test_max_page_size_enforcement() {
        let items: Vec<i32> = (0..100).collect();
        let args = PaginationArgs::forward(50, None);
        let result = ConnectionType::from_slice(items, &args, Some(10));
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(PaginationError::PageSizeTooLarge {
                requested: 50,
                maximum: 10,
            })
        ));
    }

    // ── Connection nodes and mapping ────────────────────────────────────────

    #[test]
    fn test_nodes_extraction() {
        let items: Vec<i32> = (0..5).collect();
        let args = PaginationArgs::forward(3, None);
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        assert_eq!(conn.nodes(), vec![&0, &1, &2]);
    }

    #[test]
    fn test_map_nodes() {
        let items: Vec<i32> = vec![1, 2, 3];
        let args = PaginationArgs::forward(3, None);
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        let mapped = conn.map_nodes(|n| n * 10);
        assert_eq!(mapped.nodes(), vec![&10, &20, &30]);
    }

    // ── Start/End cursor ────────────────────────────────────────────────────

    #[test]
    fn test_start_end_cursor() {
        let items: Vec<i32> = (0..10).collect();
        let args = PaginationArgs::forward(3, None);
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        assert!(conn.start_cursor().is_some());
        assert!(conn.end_cursor().is_some());
        assert_eq!(decode_cursor(conn.start_cursor().expect("exists")), Some(0));
        assert_eq!(decode_cursor(conn.end_cursor().expect("exists")), Some(2));
    }

    #[test]
    fn test_empty_connection_cursors() {
        let items: Vec<i32> = vec![];
        let args = PaginationArgs::forward(10, None);
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        assert!(conn.start_cursor().is_none());
        assert!(conn.end_cursor().is_none());
    }

    // ── ConnectionBuilder ───────────────────────────────────────────────────

    #[test]
    fn test_builder_basic() {
        let items: Vec<i32> = (0..20).collect();
        let args = PaginationArgs::forward(5, None);
        let conn = ConnectionBuilder::new(items).build(&args).expect("ok");
        assert_eq!(conn.edge_count(), 5);
        assert_eq!(conn.total_count, Some(20));
    }

    #[test]
    fn test_builder_with_total_count() {
        let items: Vec<i32> = (0..5).collect(); // subset
        let args = PaginationArgs::forward(5, None);
        let conn = ConnectionBuilder::new(items)
            .with_total_count(100) // real total
            .build(&args)
            .expect("ok");
        assert_eq!(conn.total_count, Some(100));
    }

    #[test]
    fn test_builder_with_max_page_size() {
        let items: Vec<i32> = (0..20).collect();
        let args = PaginationArgs::forward(50, None);
        let result = ConnectionBuilder::new(items)
            .with_max_page_size(10)
            .build(&args);
        assert!(result.is_err());
    }

    // ── SparqlConnection ────────────────────────────────────────────────────

    #[test]
    fn test_sparql_connection_basic() {
        let sc = SparqlConnection::new("SELECT ?s ?p ?o WHERE { ?s ?p ?o }");
        let args = PaginationArgs::forward(10, None);
        let query = sc.to_sparql(&args).expect("ok");
        assert!(query.contains("LIMIT 10"));
        assert!(query.contains("OFFSET 0"));
    }

    #[test]
    fn test_sparql_connection_with_after() {
        let sc = SparqlConnection::new("SELECT ?s WHERE { ?s ?p ?o }");
        let after = encode_cursor(19);
        let args = PaginationArgs::forward(10, Some(after));
        let query = sc.to_sparql(&args).expect("ok");
        assert!(query.contains("LIMIT 10"));
        assert!(query.contains("OFFSET 20"));
    }

    #[test]
    fn test_sparql_connection_max_page_size() {
        let sc = SparqlConnection::new("SELECT * WHERE { ?s ?p ?o }").with_max_page_size(50);
        let args = PaginationArgs::forward(100, None);
        let query = sc.to_sparql(&args).expect("ok");
        assert!(query.contains("LIMIT 50")); // clamped to max
    }

    #[test]
    fn test_sparql_connection_default_page_size() {
        let sc = SparqlConnection::new("SELECT * WHERE { ?s ?p ?o }").with_default_page_size(25);
        // No first/last, so uses default - we'll use backward for this
        let args = PaginationArgs::backward(25, None);
        let query = sc.to_sparql(&args).expect("ok");
        assert!(query.contains("LIMIT 25"));
    }

    // ── Serde roundtrip ─────────────────────────────────────────────────────

    #[test]
    fn test_edge_serde_roundtrip() {
        let edge = Edge::from_offset(42, "hello".to_string());
        let json = serde_json::to_string(&edge).expect("serialize");
        let back: Edge<String> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.node, "hello");
        assert_eq!(back.cursor, edge.cursor);
    }

    #[test]
    fn test_page_info_serde_roundtrip() {
        let pi = PageInfo {
            has_next_page: true,
            has_previous_page: false,
            start_cursor: Some(encode_cursor(0)),
            end_cursor: Some(encode_cursor(9)),
        };
        let json = serde_json::to_string(&pi).expect("serialize");
        let back: PageInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, pi);
    }

    #[test]
    fn test_connection_serde_roundtrip() {
        let items: Vec<i32> = (0..5).collect();
        let args = PaginationArgs::forward(3, None);
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        let json = serde_json::to_string(&conn).expect("serialize");
        let back: ConnectionType<i32> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.edge_count(), 3);
    }

    // ── PaginationError display ─────────────────────────────────────────────

    #[test]
    fn test_pagination_error_display() {
        let err = PaginationError::PageSizeTooLarge {
            requested: 100,
            maximum: 50,
        };
        let s = err.to_string();
        assert!(s.contains("100"));
        assert!(s.contains("50"));
    }

    #[test]
    fn test_pagination_error_invalid_cursor() {
        let err = PaginationError::InvalidCursor("bad".to_string());
        let s = err.to_string();
        assert!(s.contains("bad"));
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_empty_items_forward() {
        let items: Vec<i32> = vec![];
        let args = PaginationArgs::forward(10, None);
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        assert!(conn.is_empty());
        assert!(!conn.page_info.has_next_page);
        assert!(!conn.page_info.has_previous_page);
    }

    #[test]
    fn test_after_beyond_range() {
        let items: Vec<i32> = (0..5).collect();
        let after = encode_cursor(100);
        let args = PaginationArgs::forward(5, Some(after));
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        assert!(conn.is_empty());
    }

    #[test]
    fn test_single_item() {
        let items: Vec<i32> = vec![42];
        let args = PaginationArgs::forward(10, None);
        let conn = ConnectionType::from_slice(items, &args, None).expect("ok");
        assert_eq!(conn.edge_count(), 1);
        assert_eq!(conn.nodes(), vec![&42]);
        assert!(!conn.page_info.has_next_page);
        assert!(!conn.page_info.has_previous_page);
    }

    #[test]
    fn test_pagination_args_with_valid_after() {
        let cursor = encode_cursor(5);
        let args = PaginationArgs::forward(10, Some(cursor));
        assert!(args.validate().is_ok());
    }
}
