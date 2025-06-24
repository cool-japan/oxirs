//! Network and server functionality
//!
//! This module provides HTTP server, WebSocket subscriptions, and resolver functionality.

// Re-export server functionality
pub use crate::server::*;
pub use crate::resolvers::*;
pub use crate::subscriptions::*;

/// Server components
pub mod components {
    //! Individual server components
    
    /// HTTP GraphQL server
    pub mod http {
        pub use crate::server::*;
    }
    
    /// Field resolvers
    pub mod resolvers {
        pub use crate::resolvers::*;
    }
    
    /// WebSocket subscriptions
    pub mod subscriptions {
        pub use crate::subscriptions::*;
    }
}