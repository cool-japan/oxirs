//! Authentication and user management handlers

use crate::{
    auth::{AuthService, AuthUser, AuthResult, LoginRequest, LoginResponse, User, Permission},
    config::UserConfig,
    error::{FusekiError, FusekiResult},
    server::AppState,
};
use axum::{
    extract::{Query, State},
    http::{StatusCode, HeaderMap, header::{SET_COOKIE, AUTHORIZATION}},
    response::{Json, IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{info, warn, error, debug, instrument};

/// User registration request
#[derive(Debug, Deserialize)]
pub struct RegisterUserRequest {
    pub username: String,
    pub password: String,
    pub email: Option<String>,
    pub full_name: Option<String>,
    pub roles: Vec<String>,
}

/// Change password request
#[derive(Debug, Deserialize)]
pub struct ChangePasswordRequest {
    pub current_password: String,
    pub new_password: String,
}

/// User information response
#[derive(Debug, Serialize)]
pub struct UserInfoResponse {
    pub username: String,
    pub email: Option<String>,
    pub full_name: Option<String>,
    pub roles: Vec<String>,
    pub permissions: Vec<Permission>,
    pub last_login: Option<String>,
    pub account_status: String,
}

/// Users list response
#[derive(Debug, Serialize)]
pub struct UsersListResponse {
    pub users: Vec<UserSummary>,
    pub total_count: usize,
}

/// User summary for admin listing
#[derive(Debug, Serialize)]
pub struct UserSummary {
    pub username: String,
    pub email: Option<String>,
    pub roles: Vec<String>,
    pub enabled: bool,
    pub last_login: Option<String>,
    pub failed_login_attempts: u32,
    pub locked_until: Option<String>,
}

/// Login handler
#[instrument(skip(state, request))]
pub async fn login_handler(
    State(state): State<AppState>,
    Json(request): Json<LoginRequest>,
) -> Result<Response, FusekiError> {
    let start_time = Instant::now();

    // Check if authentication is enabled
    let auth_service = state.auth_service.as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    // Record authentication attempt
    if let Some(metrics_service) = &state.metrics_service {
        metrics_service.record_authentication(false, "form").await;
    }

    // Authenticate user
    let auth_result = auth_service.authenticate_user(&request.username, &request.password).await
        .map_err(|e| {
            warn!("Authentication error for user '{}': {}", request.username, e);
            FusekiError::authentication("Authentication failed")
        })?;

    let execution_time = start_time.elapsed();

    match auth_result {
        AuthResult::Authenticated(user) => {
            // Record successful authentication
            if let Some(metrics_service) = &state.metrics_service {
                metrics_service.record_authentication(true, "form").await;
            }

            // Generate authentication token/session
            let (token, expires_at) = if state.config.security.jwt.is_some() {
                #[cfg(feature = "auth")]
                {
                    let token = auth_service.generate_jwt_token(&user)?;
                    let expires_at = chrono::Utc::now() + chrono::Duration::seconds(
                        state.config.security.jwt.as_ref().unwrap().expiration_secs as i64
                    );
                    (Some(token), Some(expires_at))
                }
                #[cfg(not(feature = "auth"))]
                {
                    (None, None)
                }
            } else {
                // Create session
                let session_id = auth_service.create_session(user.clone()).await?;
                let expires_at = chrono::Utc::now() + chrono::Duration::seconds(
                    state.config.security.session.timeout_secs as i64
                );
                (Some(session_id), Some(expires_at))
            };

            let response = LoginResponse {
                success: true,
                token,
                user: Some(user.clone()),
                expires_at,
                message: "Login successful".to_string(),
            };

            let mut resp = Json(response).into_response();

            // Set session cookie if using session-based auth
            if state.config.security.jwt.is_none() {
                if let Some(session_id) = response.token {
                    let cookie_value = format!(
                        "session_id={}; HttpOnly; Secure; SameSite=Strict; Max-Age={}",
                        session_id,
                        state.config.security.session.timeout_secs
                    );
                    resp.headers_mut().insert(SET_COOKIE, cookie_value.parse().unwrap());
                }
            }

            info!(
                "User '{}' logged in successfully in {}ms",
                user.username,
                execution_time.as_millis()
            );

            Ok(resp)
        }
        AuthResult::Unauthenticated => {
            warn!("Failed login attempt for user '{}'", request.username);
            
            let response = LoginResponse {
                success: false,
                token: None,
                user: None,
                expires_at: None,
                message: "Invalid username or password".to_string(),
            };

            Ok((StatusCode::UNAUTHORIZED, Json(response)).into_response())
        }
        AuthResult::Locked => {
            warn!("Login attempt for locked user '{}'", request.username);
            
            let response = LoginResponse {
                success: false,
                token: None,
                user: None,
                expires_at: None,
                message: "Account is temporarily locked due to failed login attempts".to_string(),
            };

            Ok((StatusCode::FORBIDDEN, Json(response)).into_response())
        }
        AuthResult::Forbidden => {
            warn!("Login attempt for disabled user '{}'", request.username);
            
            let response = LoginResponse {
                success: false,
                token: None,
                user: None,
                expires_at: None,
                message: "Account is disabled".to_string(),
            };

            Ok((StatusCode::FORBIDDEN, Json(response)).into_response())
        }
        _ => {
            error!("Unexpected authentication result for user '{}'", request.username);
            
            let response = LoginResponse {
                success: false,
                token: None,
                user: None,
                expires_at: None,
                message: "Authentication error".to_string(),
            };

            Ok((StatusCode::INTERNAL_SERVER_ERROR, Json(response)).into_response())
        }
    }
}

/// Logout handler
#[instrument(skip(state, headers))]
pub async fn logout_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, FusekiError> {
    let auth_service = state.auth_service.as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    // Extract session ID from cookie or token
    let session_id = extract_session_from_headers(&headers)?;

    if let Some(session_id) = session_id {
        // Invalidate session
        auth_service.logout(&session_id).await?;
        
        // Clear session cookie
        let mut resp = Json(serde_json::json!({
            "success": true,
            "message": "Logged out successfully"
        })).into_response();

        resp.headers_mut().insert(
            SET_COOKIE,
            "session_id=; HttpOnly; Secure; SameSite=Strict; Max-Age=0".parse().unwrap()
        );

        info!("User logged out successfully");
        Ok(resp)
    } else {
        Ok(Json(serde_json::json!({
            "success": true,
            "message": "No active session found"
        })).into_response())
    }
}

/// Get current user information
#[instrument(skip(state, auth_user))]
pub async fn user_info_handler(
    State(state): State<AppState>,
    auth_user: AuthUser,
) -> Result<Json<UserInfoResponse>, FusekiError> {
    let user = auth_user.0;
    
    let response = UserInfoResponse {
        username: user.username.clone(),
        email: user.email.clone(),
        full_name: user.full_name.clone(),
        roles: user.roles.clone(),
        permissions: user.permissions.clone(),
        last_login: user.last_login.map(|dt| dt.to_rfc3339()),
        account_status: "active".to_string(), // Would be determined from user config
    };

    Ok(Json(response))
}

/// List all users (admin only)
#[instrument(skip(state, auth_user))]
pub async fn list_users_handler(
    State(state): State<AppState>,
    auth_user: AuthUser,
) -> Result<Json<UsersListResponse>, FusekiError> {
    let user = auth_user.0;

    // Check admin permissions
    if !user.permissions.contains(&Permission::UserManagement) {
        return Err(FusekiError::forbidden("Insufficient permissions to list users"));
    }

    let auth_service = state.auth_service.as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    // Get all users
    let users_map = auth_service.list_users().await;
    
    let mut users = Vec::new();
    for (username, user_config) in users_map {
        users.push(UserSummary {
            username,
            email: user_config.email,
            roles: user_config.roles,
            enabled: user_config.enabled,
            last_login: user_config.last_login.map(|dt| dt.to_rfc3339()),
            failed_login_attempts: user_config.failed_login_attempts,
            locked_until: user_config.locked_until.map(|dt| dt.to_rfc3339()),
        });
    }

    let total_count = users.len();

    info!("Admin user '{}' listed {} users", user.username, total_count);

    Ok(Json(UsersListResponse {
        users,
        total_count,
    }))
}

/// Register new user (admin only)
#[instrument(skip(state, auth_user, request))]
pub async fn register_user_handler(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Json(request): Json<RegisterUserRequest>,
) -> Result<Json<UserInfoResponse>, FusekiError> {
    let admin_user = auth_user.0;

    // Check admin permissions
    if !admin_user.permissions.contains(&Permission::UserManagement) {
        return Err(FusekiError::forbidden("Insufficient permissions to create users"));
    }

    let auth_service = state.auth_service.as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    // Validate request
    validate_user_registration(&request)?;

    // Check if user already exists
    if auth_service.get_user(&request.username).await.is_some() {
        return Err(FusekiError::conflict(format!("User '{}' already exists", request.username)));
    }

    // Hash password
    let password_hash = auth_service.hash_password(&request.password)?;

    // Create user config
    let user_config = UserConfig {
        password_hash,
        roles: request.roles.clone(),
        enabled: true,
        email: request.email.clone(),
        full_name: request.full_name.clone(),
        last_login: None,
        failed_login_attempts: 0,
        locked_until: None,
    };

    // Add user
    auth_service.upsert_user(request.username.clone(), user_config).await?;

    // Compute permissions for response - simplified for now
    let user_config = auth_service.get_user(&request.username).await.unwrap();
    let permissions = vec![crate::auth::Permission::GlobalRead]; // Simplified

    let response = UserInfoResponse {
        username: request.username.clone(),
        email: request.email,
        full_name: request.full_name,
        roles: request.roles,
        permissions,
        last_login: None,
        account_status: "active".to_string(),
    };

    info!("Admin user '{}' created new user '{}'", admin_user.username, request.username);

    Ok(Json(response))
}

/// Change user password
#[instrument(skip(state, auth_user, request))]
pub async fn change_password_handler(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Json(request): Json<ChangePasswordRequest>,
) -> Result<Json<serde_json::Value>, FusekiError> {
    let user = auth_user.0;

    let auth_service = state.auth_service.as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    // Get current user config
    let mut user_config = auth_service.get_user(&user.username).await
        .ok_or_else(|| FusekiError::not_found("User not found"))?;

    // Verify current password
    if !auth_service.verify_password(&request.current_password, &user_config.password_hash)? {
        return Err(FusekiError::bad_request("Current password is incorrect"));
    }

    // Validate new password strength
    validate_password_strength(&request.new_password)?;

    // Hash new password
    let new_password_hash = auth_service.hash_password(&request.new_password)?;
    user_config.password_hash = new_password_hash;

    // Update user
    auth_service.upsert_user(user.username.clone(), user_config).await?;

    info!("User '{}' changed their password", user.username);

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Password changed successfully"
    })))
}

/// Delete user (admin only)
#[instrument(skip(state, auth_user, username))]
pub async fn delete_user_handler(
    State(state): State<AppState>,
    auth_user: AuthUser,
    axum::extract::Path(username): axum::extract::Path<String>,
) -> Result<StatusCode, FusekiError> {
    let admin_user = auth_user.0;

    // Check admin permissions
    if !admin_user.permissions.contains(&Permission::UserManagement) {
        return Err(FusekiError::forbidden("Insufficient permissions to delete users"));
    }

    // Prevent self-deletion
    if admin_user.username == username {
        return Err(FusekiError::bad_request("Cannot delete your own account"));
    }

    let auth_service = state.auth_service.as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    // Delete user
    let deleted = auth_service.remove_user(&username).await?;

    if deleted {
        info!("Admin user '{}' deleted user '{}'", admin_user.username, username);
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(FusekiError::not_found(format!("User '{}' not found", username)))
    }
}

// Helper functions

/// Extract session ID from headers (cookie or authorization header)
fn extract_session_from_headers(headers: &HeaderMap) -> FusekiResult<Option<String>> {
    // Try to extract from Authorization header (Bearer token)
    if let Some(auth_header) = headers.get(AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            if auth_str.starts_with("Bearer ") {
                return Ok(Some(auth_str[7..].to_string()));
            }
        }
    }

    // Try to extract from cookie
    if let Some(cookie_header) = headers.get("cookie") {
        if let Ok(cookie_str) = cookie_header.to_str() {
            for cookie in cookie_str.split(';') {
                let cookie = cookie.trim();
                if let Some(session_id) = cookie.strip_prefix("session_id=") {
                    return Ok(Some(session_id.to_string()));
                }
            }
        }
    }

    Ok(None)
}

/// Validate user registration request
fn validate_user_registration(request: &RegisterUserRequest) -> FusekiResult<()> {
    // Validate username
    if request.username.is_empty() {
        return Err(FusekiError::bad_request("Username cannot be empty"));
    }

    if request.username.len() > 64 {
        return Err(FusekiError::bad_request("Username too long (max 64 characters)"));
    }

    if !request.username.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.') {
        return Err(FusekiError::bad_request("Username contains invalid characters"));
    }

    // Validate password
    validate_password_strength(&request.password)?;

    // Validate email format if provided
    if let Some(ref email) = request.email {
        if !email.contains('@') || email.len() > 255 {
            return Err(FusekiError::bad_request("Invalid email format"));
        }
    }

    // Validate roles
    if request.roles.is_empty() {
        return Err(FusekiError::bad_request("User must have at least one role"));
    }

    for role in &request.roles {
        if !is_valid_role(role) {
            return Err(FusekiError::bad_request(format!("Invalid role: {}", role)));
        }
    }

    Ok(())
}

/// Validate password strength
fn validate_password_strength(password: &str) -> FusekiResult<()> {
    if password.len() < 8 {
        return Err(FusekiError::bad_request("Password must be at least 8 characters long"));
    }

    if password.len() > 128 {
        return Err(FusekiError::bad_request("Password too long (max 128 characters)"));
    }

    let has_uppercase = password.chars().any(|c| c.is_uppercase());
    let has_lowercase = password.chars().any(|c| c.is_lowercase());
    let has_digit = password.chars().any(|c| c.is_ascii_digit());
    let has_special = password.chars().any(|c| !c.is_alphanumeric());

    let requirements_met = [has_uppercase, has_lowercase, has_digit, has_special]
        .iter()
        .filter(|&&x| x)
        .count();

    if requirements_met < 3 {
        return Err(FusekiError::bad_request(
            "Password must contain at least 3 of: uppercase, lowercase, digit, special character"
        ));
    }

    Ok(())
}

/// Check if role is valid
fn is_valid_role(role: &str) -> bool {
    matches!(role, "admin" | "user" | "reader" | "writer") || 
    role.starts_with("dataset:")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_registration_validation() {
        // Valid registration
        let valid_request = RegisterUserRequest {
            username: "testuser".to_string(),
            password: "SecurePass123!".to_string(),
            email: Some("test@example.com".to_string()),
            full_name: Some("Test User".to_string()),
            roles: vec!["user".to_string()],
        };
        assert!(validate_user_registration(&valid_request).is_ok());

        // Invalid username (empty)
        let invalid_request = RegisterUserRequest {
            username: "".to_string(),
            password: "SecurePass123!".to_string(),
            email: None,
            full_name: None,
            roles: vec!["user".to_string()],
        };
        assert!(validate_user_registration(&invalid_request).is_err());

        // Invalid password (too weak)
        let weak_password_request = RegisterUserRequest {
            username: "testuser".to_string(),
            password: "weak".to_string(),
            email: None,
            full_name: None,
            roles: vec!["user".to_string()],
        };
        assert!(validate_user_registration(&weak_password_request).is_err());

        // Invalid roles (empty)
        let no_roles_request = RegisterUserRequest {
            username: "testuser".to_string(),
            password: "SecurePass123!".to_string(),
            email: None,
            full_name: None,
            roles: vec![],
        };
        assert!(validate_user_registration(&no_roles_request).is_err());
    }

    #[test]
    fn test_password_strength_validation() {
        // Strong passwords
        assert!(validate_password_strength("SecurePass123!").is_ok());
        assert!(validate_password_strength("MyP@ssw0rd").is_ok());
        assert!(validate_password_strength("Complex1Password!").is_ok());

        // Weak passwords
        assert!(validate_password_strength("weak").is_err()); // Too short
        assert!(validate_password_strength("onlylowercase").is_err()); // Missing requirements
        assert!(validate_password_strength("ONLYUPPERCASE").is_err()); // Missing requirements
        assert!(validate_password_strength("12345678").is_err()); // Missing requirements
        assert!(validate_password_strength("NoDigitsOrSpecial").is_err()); // Missing requirements
    }

    #[test]
    fn test_role_validation() {
        // Valid roles
        assert!(is_valid_role("admin"));
        assert!(is_valid_role("user"));
        assert!(is_valid_role("reader"));
        assert!(is_valid_role("writer"));
        assert!(is_valid_role("dataset:mydata:read"));
        assert!(is_valid_role("dataset:mydata:write"));

        // Invalid roles
        assert!(!is_valid_role("invalid"));
        assert!(!is_valid_role(""));
        assert!(!is_valid_role("custom_role"));
    }
}