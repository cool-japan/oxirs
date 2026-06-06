//! Pagination support for Triple Pattern Fragments.
//!
//! TPF responses describe fragments — finite subsets of the full result set.
//! Clients navigate through fragments using page-based pagination. This
//! module provides defaults aligned with widely deployed TPF servers.

use super::TpfQueryParams;

/// Default fragment size used when the client omits `page_size`.
pub const DEFAULT_PAGE_SIZE: usize = 100;

/// Hard upper bound on the page size to avoid pathological requests.
pub const MAX_PAGE_SIZE: usize = 10_000;

/// Effective pagination parameters resolved from request input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaginationParams {
    /// Page number (1-indexed).
    pub page: usize,
    /// Number of triples per page.
    pub page_size: usize,
}

impl PaginationParams {
    /// Build [`PaginationParams`] from raw query parameters, applying clamping
    /// and defaults.
    ///
    /// - `page` defaults to `1` and is clamped to a minimum of `1`.
    /// - `page_size` defaults to [`DEFAULT_PAGE_SIZE`] and is clamped to the
    ///   inclusive range `[1, MAX_PAGE_SIZE]`.
    pub fn from_params(params: &TpfQueryParams) -> Self {
        Self {
            page: params.page.unwrap_or(1).max(1),
            page_size: params
                .page_size
                .unwrap_or(DEFAULT_PAGE_SIZE)
                .clamp(1, MAX_PAGE_SIZE),
        }
    }

    /// Returns the zero-indexed offset of the first triple on this page.
    pub fn offset(&self) -> usize {
        (self.page - 1) * self.page_size
    }
}

impl Default for PaginationParams {
    fn default() -> Self {
        Self {
            page: 1,
            page_size: DEFAULT_PAGE_SIZE,
        }
    }
}

/// Computed pagination metadata for a TPF response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaginationMetadata {
    pub current_page: usize,
    pub page_size: usize,
    pub total_count: usize,
    pub total_pages: usize,
    pub has_next: bool,
    pub has_previous: bool,
}

impl PaginationMetadata {
    /// Build metadata for a request against a result set of `total_count` triples.
    pub fn new(params: &PaginationParams, total_count: usize) -> Self {
        let effective_size = params.page_size.max(1);
        let total_pages = total_count.div_ceil(effective_size);
        Self {
            current_page: params.page,
            page_size: params.page_size,
            total_count,
            total_pages,
            has_next: params.page < total_pages,
            has_previous: params.page > 1,
        }
    }
}
