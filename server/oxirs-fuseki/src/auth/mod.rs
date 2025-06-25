//! Authentication and authorization system for OxiRS Fuseki
//!
//! This module provides comprehensive authentication and authorization capabilities including:
//! - Basic username/password authentication
//! - JWT token authentication
//! - OAuth2/OIDC integration
//! - LDAP authentication (when available)
//! - Role-based access control (RBAC)
//! - Permission management

pub mod oauth;
pub mod ldap;

// Re-export main auth types and functions
pub use crate::auth::{
    AuthService, AuthResult, AuthUser, AuthError,
    User, Permission, LoginRequest, LoginResponse,
    UserSession, TokenValidation, RequirePermission,
    passwords, PasswordStrength,
};

// Re-export OAuth2 types
pub use oauth::{
    OAuth2Service, OAuth2Token, OAuth2State, OIDCUserInfo,
    OAuth2AuthRequest, OAuth2TokenRequest, OAuth2Flow,
};

// Re-export LDAP types
pub use ldap::{
    LdapService, LdapUserAttributes, LdapGroup, LdapAuthRequest,
    LdapSearchParams, LdapScope, CachedUser,
};