//! Core GraphQL functionality
//!
//! This module provides the core GraphQL functionality including AST definitions,
//! type system, execution engine, and schema management.

// Re-export core GraphQL functionality
// Note: Clippy warning about ambiguous glob re-exports is acceptable here
// as this is intentional for providing a unified API surface
#[allow(ambiguous_glob_reexports)]
pub use crate::ast::*;
pub use crate::execution::*;
pub use crate::parser::*;
pub use crate::schema::*;
#[allow(ambiguous_glob_reexports)]
pub use crate::types::*;

/// Core GraphQL components
pub mod components {
    /// Individual core components
    /// Abstract Syntax Tree definitions
    pub mod ast {
        pub use crate::ast::*;
    }

    /// GraphQL type system
    pub mod types {
        pub use crate::types::*;
    }

    /// Query execution engine
    pub mod execution {
        pub use crate::execution::*;
    }

    /// Schema definition and management
    pub mod schema {
        pub use crate::schema::*;
    }

    /// GraphQL query parser
    pub mod parser {
        pub use crate::parser::*;
    }
}
