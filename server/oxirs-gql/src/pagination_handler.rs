//! GraphQL Pagination Handler (Relay Cursor Specification)
//!
//! Provides a complete Relay-spec cursor-based pagination implementation for
//! GraphQL endpoints serving RDF/knowledge-graph data.
//!
//! # Features
//!
//! - Opaque cursor encoding/decoding (base64 over offset string)
//! - [`PageInfo`] compliant with the Relay Connection Specification
//! - Forward pagination (`first` / `after`) and backward pagination (`last` / `before`)
//! - Offset-based pagination (`page` + `size`) with conversion to cursor arithmetic
//! - Total count support via optional `TotalCount` field
//! - Edge / Node / Connection pattern matching the Relay spec
//! - Cursor validation with structured [`PaginationHandlerError`] values
//! - Empty page handling (no panics, correct `PageInfo` flags)

use std::fmt;

// ── Cursor codec ──────────────────────────────────────────────────────────────

/// Prefix embedded into every cursor before base64-encoding.
const CURSOR_PREFIX: &str = "gql_cursor:";

/// Encode a 0-based item offset into an opaque cursor string.
///
/// The cursor is base64-encoded so that clients treat it as an opaque token.
pub fn encode_cursor(offset: usize) -> String {
    use base64::Engine;
    let raw = format!("{CURSOR_PREFIX}{offset}");
    base64::engine::general_purpose::STANDARD.encode(raw.as_bytes())
}

/// Decode an opaque cursor string back to a 0-based item offset.
///
/// Returns `None` when the cursor is invalid, truncated, or has the wrong prefix.
pub fn decode_cursor(cursor: &str) -> Option<usize> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(cursor.as_bytes())
        .ok()?;
    let s = String::from_utf8(bytes).ok()?;
    s.strip_prefix(CURSOR_PREFIX)?.parse::<usize>().ok()
}

/// Returns `true` if `cursor` is a syntactically valid pagination cursor.
pub fn is_valid_cursor(cursor: &str) -> bool {
    decode_cursor(cursor).is_some()
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur during pagination handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaginationHandlerError {
    /// The supplied cursor string is not a valid opaque cursor.
    InvalidCursor(String),
    /// Both `first` and `last` were provided simultaneously (Relay spec violation).
    ConflictingArgs,
    /// The requested page size exceeds the maximum.
    PageSizeTooLarge {
        /// The requested size.
        requested: usize,
        /// The maximum allowed size.
        maximum: usize,
    },
    /// Page number is zero (1-based page numbers are required).
    InvalidPageNumber,
    /// Page size is zero.
    InvalidPageSize,
}

impl fmt::Display for PaginationHandlerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaginationHandlerError::InvalidCursor(c) => {
                write!(f, "Invalid pagination cursor: '{c}'")
            }
            PaginationHandlerError::ConflictingArgs => {
                write!(
                    f,
                    "Cannot specify both 'first' and 'last' in the same query"
                )
            }
            PaginationHandlerError::PageSizeTooLarge { requested, maximum } => {
                write!(f, "Page size {requested} exceeds maximum of {maximum}")
            }
            PaginationHandlerError::InvalidPageNumber => {
                write!(f, "Page number must be >= 1")
            }
            PaginationHandlerError::InvalidPageSize => {
                write!(f, "Page size must be >= 1")
            }
        }
    }
}

impl std::error::Error for PaginationHandlerError {}

// ── PageInfo ──────────────────────────────────────────────────────────────────

/// Relay-spec `PageInfo` object returned with every paginated connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageInfo {
    /// `true` when there are items after the last edge in this page.
    pub has_next_page: bool,
    /// `true` when there are items before the first edge in this page.
    pub has_previous_page: bool,
    /// Opaque cursor for the first edge, or `None` if the page is empty.
    pub start_cursor: Option<String>,
    /// Opaque cursor for the last edge, or `None` if the page is empty.
    pub end_cursor: Option<String>,
}

impl PageInfo {
    /// Construct `PageInfo` for an empty result page.
    pub fn empty() -> Self {
        Self {
            has_next_page: false,
            has_previous_page: false,
            start_cursor: None,
            end_cursor: None,
        }
    }

    /// Returns `true` if the page has no edges.
    pub fn is_empty_page(&self) -> bool {
        self.start_cursor.is_none()
    }
}

impl Default for PageInfo {
    fn default() -> Self {
        Self::empty()
    }
}

// ── Edge / Node ───────────────────────────────────────────────────────────────

/// A single edge in a Relay connection wrapping a data node with its cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge<T> {
    /// Opaque cursor identifying this edge's position.
    pub cursor: String,
    /// The data node for this edge.
    pub node: T,
}

impl<T> Edge<T> {
    /// Create an edge directly from a cursor string and node.
    pub fn new(cursor: String, node: T) -> Self {
        Self { cursor, node }
    }

    /// Create an edge from a 0-based offset and node.
    pub fn from_offset(offset: usize, node: T) -> Self {
        Self {
            cursor: encode_cursor(offset),
            node,
        }
    }

    /// Apply a mapping function to the node, preserving the cursor.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Edge<U> {
        Edge {
            cursor: self.cursor,
            node: f(self.node),
        }
    }
}

impl<T: fmt::Display> fmt::Display for Edge<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Edge {{ cursor: {}, node: {} }}", self.cursor, self.node)
    }
}

// ── Connection ────────────────────────────────────────────────────────────────

/// A Relay-spec connection wrapping a paginated list of [`Edge<T>`] values.
#[derive(Debug, Clone)]
pub struct Connection<T> {
    /// The edges for this page, each carrying a node and a cursor.
    pub edges: Vec<Edge<T>>,
    /// Pagination metadata.
    pub page_info: PageInfo,
    /// Total count of all items (may require a separate count query).
    pub total_count: Option<usize>,
}

impl<T> Connection<T> {
    /// Construct an empty connection.
    pub fn empty() -> Self {
        Self {
            edges: Vec::new(),
            page_info: PageInfo::empty(),
            total_count: Some(0),
        }
    }

    /// Returns the number of edges in this connection.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Returns `true` if there are no edges.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Collect references to the nodes in all edges.
    pub fn nodes(&self) -> Vec<&T> {
        self.edges.iter().map(|e| &e.node).collect()
    }

    /// Transform all nodes using a mapping function.
    pub fn map_nodes<U, F: Fn(T) -> U>(self, f: F) -> Connection<U> {
        Connection {
            edges: self.edges.into_iter().map(|e| e.map(&f)).collect(),
            page_info: self.page_info,
            total_count: self.total_count,
        }
    }

    /// Return the cursor of the first edge (same as `page_info.start_cursor`).
    pub fn start_cursor(&self) -> Option<&str> {
        self.page_info.start_cursor.as_deref()
    }

    /// Return the cursor of the last edge (same as `page_info.end_cursor`).
    pub fn end_cursor(&self) -> Option<&str> {
        self.page_info.end_cursor.as_deref()
    }
}

impl<T> Default for Connection<T> {
    fn default() -> Self {
        Self::empty()
    }
}

// ── Pagination arguments ──────────────────────────────────────────────────────

/// Relay-spec cursor-based pagination arguments.
///
/// Only one of `first`/`last` may be set at a time.
#[derive(Debug, Clone, Default)]
pub struct CursorArgs {
    /// Forward pagination: take the first N edges after the `after` cursor.
    pub first: Option<usize>,
    /// Forward pagination: start after this cursor.
    pub after: Option<String>,
    /// Backward pagination: take the last N edges before the `before` cursor.
    pub last: Option<usize>,
    /// Backward pagination: end before this cursor.
    pub before: Option<String>,
}

impl CursorArgs {
    /// Construct forward pagination args.
    pub fn forward(first: usize, after: Option<String>) -> Self {
        Self {
            first: Some(first),
            after,
            ..Default::default()
        }
    }

    /// Construct backward pagination args.
    pub fn backward(last: usize, before: Option<String>) -> Self {
        Self {
            last: Some(last),
            before,
            ..Default::default()
        }
    }

    /// Validate the args, returning the first detected error.
    pub fn validate(&self) -> Result<(), PaginationHandlerError> {
        if self.first.is_some() && self.last.is_some() {
            return Err(PaginationHandlerError::ConflictingArgs);
        }
        if let Some(after) = &self.after {
            if !is_valid_cursor(after) {
                return Err(PaginationHandlerError::InvalidCursor(after.clone()));
            }
        }
        if let Some(before) = &self.before {
            if !is_valid_cursor(before) {
                return Err(PaginationHandlerError::InvalidCursor(before.clone()));
            }
        }
        Ok(())
    }
}

/// Offset-based (page/size) pagination arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct OffsetArgs {
    /// 1-based page number.
    pub page: usize,
    /// Number of items per page.
    pub page_size: usize,
}

impl OffsetArgs {
    /// Construct offset args with validation.
    pub fn new(page: usize, page_size: usize) -> Result<Self, PaginationHandlerError> {
        if page == 0 {
            return Err(PaginationHandlerError::InvalidPageNumber);
        }
        if page_size == 0 {
            return Err(PaginationHandlerError::InvalidPageSize);
        }
        Ok(Self { page, page_size })
    }

    /// Convert these offset args to a `CursorArgs` equivalent.
    ///
    /// The `after` cursor points to the last item on the previous page.
    pub fn to_cursor_args(&self) -> CursorArgs {
        let start_offset = (self.page - 1) * self.page_size;
        let after = if start_offset == 0 {
            None
        } else {
            Some(encode_cursor(start_offset - 1))
        };
        CursorArgs::forward(self.page_size, after)
    }

    /// 0-based index of the first item on this page.
    pub fn start_offset(&self) -> usize {
        (self.page - 1) * self.page_size
    }
}

// ── Pagination handler ────────────────────────────────────────────────────────

/// Configuration for the pagination handler.
#[derive(Debug, Clone)]
pub struct PaginationConfig {
    /// Maximum page size allowed.
    pub max_page_size: usize,
    /// Default page size when neither `first` nor `last` is specified.
    pub default_page_size: usize,
}

impl Default for PaginationConfig {
    fn default() -> Self {
        Self {
            max_page_size: 100,
            default_page_size: 10,
        }
    }
}

/// GraphQL pagination handler.
///
/// Applies Relay cursor-based or offset-based pagination to an in-memory item
/// list and returns a fully populated [`Connection<T>`].
pub struct PaginationHandler {
    config: PaginationConfig,
}

impl PaginationHandler {
    /// Create a new handler with the given configuration.
    pub fn new(config: PaginationConfig) -> Self {
        Self { config }
    }

    /// Create a handler with default configuration.
    pub fn default_config() -> Self {
        Self::new(PaginationConfig::default())
    }

    /// Apply cursor-based pagination to `items`.
    ///
    /// Follows the Relay Connection Specification algorithm:
    /// 1. Apply `before` constraint to determine the upper window boundary.
    /// 2. Apply `after` constraint to determine the lower window boundary.
    /// 3. Apply `first` or `last` to slice within the window.
    pub fn paginate<T: Clone>(
        &self,
        items: Vec<T>,
        args: &CursorArgs,
    ) -> Result<Connection<T>, PaginationHandlerError> {
        args.validate()?;

        let total = items.len();

        if total == 0 {
            return Ok(Connection {
                edges: Vec::new(),
                page_info: PageInfo {
                    has_next_page: false,
                    has_previous_page: false,
                    start_cursor: None,
                    end_cursor: None,
                },
                total_count: Some(0),
            });
        }

        // Determine the window start (inclusive) from the `after` cursor.
        let window_start = if let Some(after) = &args.after {
            let idx = decode_cursor(after)
                .ok_or_else(|| PaginationHandlerError::InvalidCursor(after.clone()))?;
            idx + 1
        } else {
            0
        };

        // Determine the window end (exclusive) from the `before` cursor.
        let window_end = if let Some(before) = &args.before {
            let idx = decode_cursor(before)
                .ok_or_else(|| PaginationHandlerError::InvalidCursor(before.clone()))?;
            idx.min(total)
        } else {
            total
        };

        if window_start >= window_end {
            return Ok(Connection {
                edges: Vec::new(),
                page_info: PageInfo {
                    has_next_page: window_start < total,
                    has_previous_page: window_start > 0,
                    start_cursor: None,
                    end_cursor: None,
                },
                total_count: Some(total),
            });
        }

        let window = &items[window_start..window_end];

        // Apply `first` or `last` slicing within the window.
        let (slice_start, slice_end) = self.compute_slice(window.len(), args)?;

        let sliced = &window[slice_start..slice_end];
        let edges: Vec<Edge<T>> = sliced
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let global = window_start + slice_start + i;
                Edge::from_offset(global, item.clone())
            })
            .collect();

        let has_previous = window_start + slice_start > 0;
        let has_next = window_start + slice_end < total;

        let start_cursor = edges.first().map(|e| e.cursor.clone());
        let end_cursor = edges.last().map(|e| e.cursor.clone());

        Ok(Connection {
            edges,
            page_info: PageInfo {
                has_next_page: has_next,
                has_previous_page: has_previous,
                start_cursor,
                end_cursor,
            },
            total_count: Some(total),
        })
    }

    /// Apply offset-based pagination to `items`.
    pub fn paginate_offset<T: Clone>(
        &self,
        items: Vec<T>,
        args: &OffsetArgs,
    ) -> Result<Connection<T>, PaginationHandlerError> {
        let cursor_args = args.to_cursor_args();
        let mut conn = self.paginate(items, &cursor_args)?;
        // Preserve original total (may differ from slice size).
        // total_count is already set correctly by paginate().
        // Apply max_page_size cap.
        if conn.len() > self.config.max_page_size {
            let capped: Vec<Edge<T>> = conn
                .edges
                .into_iter()
                .take(self.config.max_page_size)
                .collect();
            let end_cursor = capped.last().map(|e| e.cursor.clone());
            let start_cursor = capped.first().map(|e| e.cursor.clone());
            conn.edges = capped;
            conn.page_info.end_cursor = end_cursor;
            conn.page_info.start_cursor = start_cursor;
        }
        Ok(conn)
    }

    fn compute_slice(
        &self,
        available: usize,
        args: &CursorArgs,
    ) -> Result<(usize, usize), PaginationHandlerError> {
        if let Some(first) = args.first {
            if first > self.config.max_page_size {
                return Err(PaginationHandlerError::PageSizeTooLarge {
                    requested: first,
                    maximum: self.config.max_page_size,
                });
            }
            Ok((0, first.min(available)))
        } else if let Some(last) = args.last {
            if last > self.config.max_page_size {
                return Err(PaginationHandlerError::PageSizeTooLarge {
                    requested: last,
                    maximum: self.config.max_page_size,
                });
            }
            let start = available.saturating_sub(last);
            Ok((start, available))
        } else {
            // Neither `first` nor `last`: use the default page size.
            let end = self.config.default_page_size.min(available);
            Ok((0, end))
        }
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &PaginationConfig {
        &self.config
    }
}

impl Default for PaginationHandler {
    fn default() -> Self {
        Self::default_config()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn handler() -> PaginationHandler {
        PaginationHandler::default_config()
    }

    fn handler_with_max(max: usize) -> PaginationHandler {
        PaginationHandler::new(PaginationConfig {
            max_page_size: max,
            default_page_size: 5,
        })
    }

    fn items(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    // ── Cursor encode/decode ─────────────────────────────────────────────────

    #[test]
    fn test_encode_decode_zero() {
        let c = encode_cursor(0);
        assert_eq!(decode_cursor(&c), Some(0));
    }

    #[test]
    fn test_encode_decode_large() {
        let c = encode_cursor(1_000_000);
        assert_eq!(decode_cursor(&c), Some(1_000_000));
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        for i in [1, 10, 99, 500, 9999] {
            assert_eq!(decode_cursor(&encode_cursor(i)), Some(i));
        }
    }

    #[test]
    fn test_decode_invalid_base64() {
        assert!(decode_cursor("!!!not-base64!!!").is_none());
    }

    #[test]
    fn test_decode_wrong_prefix() {
        use base64::Engine;
        let bad = base64::engine::general_purpose::STANDARD.encode(b"wrong:42");
        assert!(decode_cursor(&bad).is_none());
    }

    #[test]
    fn test_is_valid_cursor_true() {
        assert!(is_valid_cursor(&encode_cursor(5)));
    }

    #[test]
    fn test_is_valid_cursor_false() {
        assert!(!is_valid_cursor("garbage"));
    }

    // ── PaginationHandlerError display ───────────────────────────────────────

    #[test]
    fn test_error_invalid_cursor_display() {
        let e = PaginationHandlerError::InvalidCursor("bad".to_string());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn test_error_conflicting_args_display() {
        let s = PaginationHandlerError::ConflictingArgs.to_string();
        assert!(s.contains("first") || s.contains("last"));
    }

    #[test]
    fn test_error_page_size_too_large_display() {
        let e = PaginationHandlerError::PageSizeTooLarge {
            requested: 200,
            maximum: 100,
        };
        let s = e.to_string();
        assert!(s.contains("200") && s.contains("100"));
    }

    #[test]
    fn test_error_invalid_page_number_display() {
        let s = PaginationHandlerError::InvalidPageNumber.to_string();
        assert!(!s.is_empty());
    }

    #[test]
    fn test_error_invalid_page_size_display() {
        let s = PaginationHandlerError::InvalidPageSize.to_string();
        assert!(!s.is_empty());
    }

    // ── PageInfo ─────────────────────────────────────────────────────────────

    #[test]
    fn test_page_info_empty() {
        let pi = PageInfo::empty();
        assert!(!pi.has_next_page);
        assert!(!pi.has_previous_page);
        assert!(pi.start_cursor.is_none());
        assert!(pi.end_cursor.is_none());
        assert!(pi.is_empty_page());
    }

    #[test]
    fn test_page_info_default_equals_empty() {
        assert_eq!(PageInfo::default(), PageInfo::empty());
    }

    // ── Edge ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_edge_new() {
        let e = Edge::new("c".to_string(), 42i32);
        assert_eq!(e.cursor, "c");
        assert_eq!(e.node, 42);
    }

    #[test]
    fn test_edge_from_offset() {
        let e = Edge::from_offset(7, "node");
        assert_eq!(decode_cursor(&e.cursor), Some(7));
        assert_eq!(e.node, "node");
    }

    #[test]
    fn test_edge_map() {
        let e = Edge::from_offset(3, 10i32);
        let mapped = e.map(|n| n * 2);
        assert_eq!(mapped.node, 20);
        assert_eq!(decode_cursor(&mapped.cursor), Some(3));
    }

    #[test]
    fn test_edge_display() {
        let e = Edge::new("cur".to_string(), "node_val".to_string());
        let s = e.to_string();
        assert!(s.contains("cur"));
        assert!(s.contains("node_val"));
    }

    // ── Connection ────────────────────────────────────────────────────────────

    #[test]
    fn test_connection_empty() {
        let c: Connection<i32> = Connection::empty();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.total_count, Some(0));
    }

    #[test]
    fn test_connection_nodes() {
        let c = handler()
            .paginate(items(5), &CursorArgs::forward(3, None))
            .expect("ok");
        assert_eq!(c.nodes(), vec![&0, &1, &2]);
    }

    #[test]
    fn test_connection_map_nodes() {
        let c = handler()
            .paginate(items(3), &CursorArgs::forward(3, None))
            .expect("ok");
        let mapped = c.map_nodes(|n| n * 10);
        assert_eq!(mapped.nodes(), vec![&0, &10, &20]);
    }

    #[test]
    fn test_connection_start_end_cursor() {
        let c = handler()
            .paginate(items(10), &CursorArgs::forward(3, None))
            .expect("ok");
        assert_eq!(decode_cursor(c.start_cursor().expect("start")), Some(0));
        assert_eq!(decode_cursor(c.end_cursor().expect("end")), Some(2));
    }

    // ── CursorArgs validation ────────────────────────────────────────────────

    #[test]
    fn test_cursor_args_forward_valid() {
        let a = CursorArgs::forward(10, None);
        assert!(a.validate().is_ok());
    }

    #[test]
    fn test_cursor_args_backward_valid() {
        let a = CursorArgs::backward(5, None);
        assert!(a.validate().is_ok());
    }

    #[test]
    fn test_cursor_args_conflict_error() {
        let a = CursorArgs {
            first: Some(5),
            last: Some(5),
            ..Default::default()
        };
        assert_eq!(a.validate(), Err(PaginationHandlerError::ConflictingArgs));
    }

    #[test]
    fn test_cursor_args_invalid_after() {
        let a = CursorArgs {
            first: Some(5),
            after: Some("bad_cursor".to_string()),
            ..Default::default()
        };
        assert!(matches!(
            a.validate(),
            Err(PaginationHandlerError::InvalidCursor(_))
        ));
    }

    #[test]
    fn test_cursor_args_invalid_before() {
        let a = CursorArgs {
            last: Some(5),
            before: Some("bad_cursor".to_string()),
            ..Default::default()
        };
        assert!(matches!(
            a.validate(),
            Err(PaginationHandlerError::InvalidCursor(_))
        ));
    }

    #[test]
    fn test_cursor_args_valid_after() {
        let cursor = encode_cursor(4);
        let a = CursorArgs::forward(5, Some(cursor));
        assert!(a.validate().is_ok());
    }

    // ── OffsetArgs ────────────────────────────────────────────────────────────

    #[test]
    fn test_offset_args_new_ok() {
        let a = OffsetArgs::new(1, 10).expect("ok");
        assert_eq!(a.page, 1);
        assert_eq!(a.page_size, 10);
    }

    #[test]
    fn test_offset_args_page_zero_error() {
        assert_eq!(
            OffsetArgs::new(0, 10),
            Err(PaginationHandlerError::InvalidPageNumber)
        );
    }

    #[test]
    fn test_offset_args_size_zero_error() {
        assert_eq!(
            OffsetArgs::new(1, 0),
            Err(PaginationHandlerError::InvalidPageSize)
        );
    }

    #[test]
    fn test_offset_args_start_offset_first_page() {
        let a = OffsetArgs::new(1, 10).expect("ok");
        assert_eq!(a.start_offset(), 0);
    }

    #[test]
    fn test_offset_args_start_offset_third_page() {
        let a = OffsetArgs::new(3, 10).expect("ok");
        assert_eq!(a.start_offset(), 20);
    }

    #[test]
    fn test_offset_args_to_cursor_args_first_page_no_after() {
        let a = OffsetArgs::new(1, 5).expect("ok");
        let ca = a.to_cursor_args();
        assert!(ca.after.is_none());
        assert_eq!(ca.first, Some(5));
    }

    #[test]
    fn test_offset_args_to_cursor_args_second_page_has_after() {
        let a = OffsetArgs::new(2, 5).expect("ok");
        let ca = a.to_cursor_args();
        assert!(ca.after.is_some());
        // After cursor should point to index 4 (last item of page 1)
        assert_eq!(decode_cursor(ca.after.as_ref().expect("after")), Some(4));
    }

    // ── Forward pagination ────────────────────────────────────────────────────

    #[test]
    fn test_forward_first_page() {
        let c = handler()
            .paginate(items(20), &CursorArgs::forward(5, None))
            .expect("ok");
        assert_eq!(c.len(), 5);
        assert_eq!(c.nodes(), vec![&0, &1, &2, &3, &4]);
        assert!(c.page_info.has_next_page);
        assert!(!c.page_info.has_previous_page);
        assert_eq!(c.total_count, Some(20));
    }

    #[test]
    fn test_forward_second_page() {
        let after = encode_cursor(4);
        let c = handler()
            .paginate(items(20), &CursorArgs::forward(5, Some(after)))
            .expect("ok");
        assert_eq!(c.len(), 5);
        assert_eq!(c.nodes(), vec![&5, &6, &7, &8, &9]);
        assert!(c.page_info.has_next_page);
        assert!(c.page_info.has_previous_page);
    }

    #[test]
    fn test_forward_last_page_partial() {
        let after = encode_cursor(17);
        let c = handler()
            .paginate(items(20), &CursorArgs::forward(5, Some(after)))
            .expect("ok");
        assert_eq!(c.len(), 2); // items 18, 19
        assert!(!c.page_info.has_next_page);
        assert!(c.page_info.has_previous_page);
    }

    #[test]
    fn test_forward_exact_fit() {
        let c = handler()
            .paginate(items(5), &CursorArgs::forward(5, None))
            .expect("ok");
        assert_eq!(c.len(), 5);
        assert!(!c.page_info.has_next_page);
        assert!(!c.page_info.has_previous_page);
    }

    #[test]
    fn test_forward_after_beyond_end() {
        let after = encode_cursor(100);
        let c = handler()
            .paginate(items(5), &CursorArgs::forward(5, Some(after)))
            .expect("ok");
        assert!(c.is_empty());
    }

    // ── Backward pagination ───────────────────────────────────────────────────

    #[test]
    fn test_backward_last_page() {
        let c = handler()
            .paginate(items(20), &CursorArgs::backward(5, None))
            .expect("ok");
        assert_eq!(c.len(), 5);
        assert_eq!(c.nodes(), vec![&15, &16, &17, &18, &19]);
        assert!(!c.page_info.has_next_page);
        assert!(c.page_info.has_previous_page);
    }

    #[test]
    fn test_backward_with_before() {
        let before = encode_cursor(15);
        let c = handler()
            .paginate(items(20), &CursorArgs::backward(5, Some(before)))
            .expect("ok");
        assert_eq!(c.len(), 5);
        assert_eq!(c.nodes(), vec![&10, &11, &12, &13, &14]);
        assert!(c.page_info.has_next_page);
        assert!(c.page_info.has_previous_page);
    }

    #[test]
    fn test_backward_all_items() {
        let c = handler()
            .paginate(items(3), &CursorArgs::backward(10, None))
            .expect("ok");
        assert_eq!(c.len(), 3);
        assert!(!c.page_info.has_next_page);
        assert!(!c.page_info.has_previous_page);
    }

    // ── Max page size ─────────────────────────────────────────────────────────

    #[test]
    fn test_max_page_size_enforced_forward() {
        let h = handler_with_max(5);
        let result = h.paginate(items(100), &CursorArgs::forward(10, None));
        assert!(matches!(
            result,
            Err(PaginationHandlerError::PageSizeTooLarge {
                requested: 10,
                maximum: 5
            })
        ));
    }

    #[test]
    fn test_max_page_size_enforced_backward() {
        let h = handler_with_max(5);
        let result = h.paginate(items(100), &CursorArgs::backward(10, None));
        assert!(matches!(
            result,
            Err(PaginationHandlerError::PageSizeTooLarge {
                requested: 10,
                maximum: 5
            })
        ));
    }

    #[test]
    fn test_max_page_size_exact_allowed() {
        let h = handler_with_max(5);
        let c = h
            .paginate(items(10), &CursorArgs::forward(5, None))
            .expect("ok");
        assert_eq!(c.len(), 5);
    }

    // ── Default page size ─────────────────────────────────────────────────────

    #[test]
    fn test_default_page_size_used_when_no_first_last() {
        let h = PaginationHandler::new(PaginationConfig {
            max_page_size: 100,
            default_page_size: 3,
        });
        let c = h.paginate(items(10), &CursorArgs::default()).expect("ok");
        assert_eq!(c.len(), 3);
    }

    // ── Empty items ───────────────────────────────────────────────────────────

    #[test]
    fn test_empty_items_forward() {
        let c = handler()
            .paginate(Vec::<i32>::new(), &CursorArgs::forward(5, None))
            .expect("ok");
        assert!(c.is_empty());
        assert!(!c.page_info.has_next_page);
        assert!(!c.page_info.has_previous_page);
    }

    #[test]
    fn test_empty_items_backward() {
        let c = handler()
            .paginate(Vec::<i32>::new(), &CursorArgs::backward(5, None))
            .expect("ok");
        assert!(c.is_empty());
    }

    // ── Offset pagination ─────────────────────────────────────────────────────

    #[test]
    fn test_offset_first_page() {
        let args = OffsetArgs::new(1, 5).expect("ok");
        let c = handler().paginate_offset(items(20), &args).expect("ok");
        assert_eq!(c.len(), 5);
        assert_eq!(c.nodes(), vec![&0, &1, &2, &3, &4]);
        assert!(c.page_info.has_next_page);
        assert!(!c.page_info.has_previous_page);
    }

    #[test]
    fn test_offset_second_page() {
        let args = OffsetArgs::new(2, 5).expect("ok");
        let c = handler().paginate_offset(items(20), &args).expect("ok");
        assert_eq!(c.len(), 5);
        assert_eq!(c.nodes(), vec![&5, &6, &7, &8, &9]);
        assert!(c.page_info.has_next_page);
        assert!(c.page_info.has_previous_page);
    }

    #[test]
    fn test_offset_last_page_partial() {
        let args = OffsetArgs::new(4, 7).expect("ok");
        let c = handler().paginate_offset(items(24), &args).expect("ok");
        // Page 4 * 7 = items 21, 22, 23 → 3 items
        assert_eq!(c.len(), 3);
        assert!(!c.page_info.has_next_page);
        assert!(c.page_info.has_previous_page);
    }

    #[test]
    fn test_offset_beyond_total() {
        let args = OffsetArgs::new(10, 5).expect("ok");
        let c = handler().paginate_offset(items(5), &args).expect("ok");
        assert!(c.is_empty());
    }

    // ── Total count ───────────────────────────────────────────────────────────

    #[test]
    fn test_total_count_present() {
        let c = handler()
            .paginate(items(20), &CursorArgs::forward(5, None))
            .expect("ok");
        assert_eq!(c.total_count, Some(20));
    }

    #[test]
    fn test_total_count_zero_items() {
        let c = handler()
            .paginate(Vec::<i32>::new(), &CursorArgs::forward(5, None))
            .expect("ok");
        assert_eq!(c.total_count, Some(0));
    }

    // ── Config accessor ───────────────────────────────────────────────────────

    #[test]
    fn test_config_accessor() {
        let h = handler();
        assert_eq!(h.config().max_page_size, 100);
        assert_eq!(h.config().default_page_size, 10);
    }

    // ── Edge cursor correctness ───────────────────────────────────────────────

    #[test]
    fn test_edge_cursors_match_global_offsets() {
        let after = encode_cursor(4); // start from item 5
        let c = handler()
            .paginate(items(20), &CursorArgs::forward(3, Some(after)))
            .expect("ok");
        // Edges should be at global offsets 5, 6, 7
        for (i, edge) in c.edges.iter().enumerate() {
            assert_eq!(decode_cursor(&edge.cursor), Some(5 + i));
        }
    }

    #[test]
    fn test_start_and_end_cursor_match_edges() {
        let c = handler()
            .paginate(items(10), &CursorArgs::forward(4, None))
            .expect("ok");
        assert_eq!(c.start_cursor(), c.edges.first().map(|e| e.cursor.as_str()));
        assert_eq!(c.end_cursor(), c.edges.last().map(|e| e.cursor.as_str()));
    }

    // ── Single item ───────────────────────────────────────────────────────────

    #[test]
    fn test_single_item_forward() {
        let c = handler()
            .paginate(vec![42i32], &CursorArgs::forward(10, None))
            .expect("ok");
        assert_eq!(c.len(), 1);
        assert_eq!(c.nodes(), vec![&42]);
        assert!(!c.page_info.has_next_page);
        assert!(!c.page_info.has_previous_page);
    }

    // ── Cursor validation in paginate ─────────────────────────────────────────

    #[test]
    fn test_paginate_invalid_after_error() {
        let args = CursorArgs {
            first: Some(5),
            after: Some("totally_invalid".to_string()),
            ..Default::default()
        };
        let result = handler().paginate(items(10), &args);
        assert!(matches!(
            result,
            Err(PaginationHandlerError::InvalidCursor(_))
        ));
    }
}
