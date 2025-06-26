//! LDAP authentication handlers

use crate::{
    auth::{AuthResult, AuthService, User, ldap::LdapAuthRequest},
    error::FusekiResult,
    server::AppState,
};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// LDAP login request
#[derive(Debug, Deserialize)]
pub struct LdapLoginRequest {
    pub username: String,
    pub password: String,
    pub domain: Option<String>,
}

/// LDAP login response
#[derive(Debug, Serialize)]
pub struct LdapLoginResponse {
    pub success: bool,
    pub access_token: Option<String>,
    pub token_type: Option<String>,
    pub expires_in: Option<u64>,
    pub user: Option<User>,
    pub message: String,
}

/// LDAP test connection parameters
#[derive(Debug, Deserialize)]
pub struct LdapTestParams {
    pub server: Option<String>,
    pub bind_dn: Option<String>,
    pub bind_password: Option<String>,
}

/// LDAP test connection response
#[derive(Debug, Serialize)]
pub struct LdapTestResponse {
    pub success: bool,
    pub message: String,
    pub details: Option<String>,
}

/// LDAP group query parameters
#[derive(Debug, Deserialize)]
pub struct LdapGroupParams {
    pub username: String,
}

/// LDAP group response
#[derive(Debug, Serialize)]
pub struct LdapGroupResponse {
    pub success: bool,
    pub groups: Vec<LdapGroupInfo>,
    pub message: String,
}

/// LDAP group information
#[derive(Debug, Serialize)]
pub struct LdapGroupInfo {
    pub dn: String,
    pub name: String,
    pub description: Option<String>,
}

/// Handle LDAP login
pub async fn ldap_login(
    State(state): State<Arc<AppState>>,
    Json(request): Json<LdapLoginRequest>,
) -> Result<Response, StatusCode> {
    debug!("LDAP login attempt for user: {}", request.username);

    // Check if LDAP is configured
    let auth_service = match &state.auth_service {
        Some(service) => service,
        None => {
            let response = LdapLoginResponse {
                success: false,
                access_token: None,
                token_type: None,
                expires_in: None,
                user: None,
                message: "Authentication service not configured".to_string(),
            };
            return Ok((StatusCode::SERVICE_UNAVAILABLE, Json(response)).into_response());
        }
    };

    if !auth_service.is_ldap_enabled() {
        let response = LdapLoginResponse {
            success: false,
            access_token: None,
            token_type: None,
            expires_in: None,
            user: None,
            message: "LDAP authentication not configured".to_string(),
        };
        return Ok((StatusCode::NOT_IMPLEMENTED, Json(response)).into_response());
    }

    // Authenticate with LDAP
    match auth_service
        .authenticate_ldap(&request.username, &request.password)
        .await
    {
        Ok(AuthResult::Authenticated(user)) => {
            info!("LDAP authentication successful for user: {}", user.username);

            // Create session
            let session_id = auth_service
                .create_session(user.clone())
                .await
                .map_err(|e| {
                    error!("Failed to create session: {}", e);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;

            // Generate JWT token if enabled
            let (access_token, token_type, expires_in) = if auth_service.config.jwt.is_some() {
                match auth_service.generate_jwt_token(&user) {
                    Ok(token) => {
                        let expires_in = auth_service
                            .config
                            .jwt
                            .as_ref()
                            .map(|jwt| jwt.expiration_secs)
                            .unwrap_or(3600);
                        (Some(token), Some("Bearer".to_string()), Some(expires_in))
                    }
                    Err(e) => {
                        warn!("Failed to generate JWT token: {}", e);
                        (Some(session_id), Some("Session".to_string()), Some(3600))
                    }
                }
            } else {
                (Some(session_id), Some("Session".to_string()), Some(3600))
            };

            let response = LdapLoginResponse {
                success: true,
                access_token,
                token_type,
                expires_in,
                user: Some(user),
                message: "LDAP authentication successful".to_string(),
            };

            Ok((StatusCode::OK, Json(response)).into_response())
        }
        Ok(AuthResult::Unauthenticated) => {
            warn!("LDAP authentication failed for user: {}", request.username);
            let response = LdapLoginResponse {
                success: false,
                access_token: None,
                token_type: None,
                expires_in: None,
                user: None,
                message: "Invalid username or password".to_string(),
            };
            Ok((StatusCode::UNAUTHORIZED, Json(response)).into_response())
        }
        Ok(AuthResult::Locked) => {
            warn!("LDAP authentication failed - account locked: {}", request.username);
            let response = LdapLoginResponse {
                success: false,
                access_token: None,
                token_type: None,
                expires_in: None,
                user: None,
                message: "Account is locked".to_string(),
            };
            Ok((StatusCode::FORBIDDEN, Json(response)).into_response())
        }
        Ok(_) => {
            let response = LdapLoginResponse {
                success: false,
                access_token: None,
                token_type: None,
                expires_in: None,
                user: None,
                message: "Authentication failed".to_string(),
            };
            Ok((StatusCode::UNAUTHORIZED, Json(response)).into_response())
        }
        Err(e) => {
            error!("LDAP authentication error: {}", e);
            let response = LdapLoginResponse {
                success: false,
                access_token: None,
                token_type: None,
                expires_in: None,
                user: None,
                message: "Authentication service error".to_string(),
            };
            Ok((StatusCode::INTERNAL_SERVER_ERROR, Json(response)).into_response())
        }
    }
}

/// Test LDAP connection
pub async fn test_ldap_connection(
    State(state): State<Arc<AppState>>,
    Query(params): Query<LdapTestParams>,
) -> Result<Response, StatusCode> {
    debug!("Testing LDAP connection");

    // Check if LDAP is configured
    let auth_service = match &state.auth_service {
        Some(service) => service,
        None => {
            let response = LdapTestResponse {
                success: false,
                message: "Authentication service not configured".to_string(),
                details: None,
            };
            return Ok((StatusCode::SERVICE_UNAVAILABLE, Json(response)).into_response());
        }
    };

    if !auth_service.is_ldap_enabled() {
        let response = LdapTestResponse {
            success: false,
            message: "LDAP authentication not configured".to_string(),
            details: None,
        };
        return Ok((StatusCode::NOT_IMPLEMENTED, Json(response)).into_response());
    }

    // Test connection
    match auth_service.test_ldap_connection().await {
        Ok(true) => {
            info!("LDAP connection test successful");
            let response = LdapTestResponse {
                success: true,
                message: "LDAP connection successful".to_string(),
                details: Some(format!(
                    "Connected to LDAP server at {}",
                    auth_service.config.ldap.as_ref().unwrap().server
                )),
            };
            Ok((StatusCode::OK, Json(response)).into_response())
        }
        Ok(false) => {
            warn!("LDAP connection test failed");
            let response = LdapTestResponse {
                success: false,
                message: "LDAP connection failed".to_string(),
                details: Some("Unable to bind to LDAP server with configured credentials".to_string()),
            };
            Ok((StatusCode::SERVICE_UNAVAILABLE, Json(response)).into_response())
        }
        Err(e) => {
            error!("LDAP connection test error: {}", e);
            let response = LdapTestResponse {
                success: false,
                message: "LDAP connection test error".to_string(),
                details: Some(e.to_string()),
            };
            Ok((StatusCode::INTERNAL_SERVER_ERROR, Json(response)).into_response())
        }
    }
}

/// Get user groups from LDAP
pub async fn get_ldap_groups(
    State(state): State<Arc<AppState>>,
    Query(params): Query<LdapGroupParams>,
) -> Result<Response, StatusCode> {
    debug!("Getting LDAP groups for user: {}", params.username);

    // Check if LDAP is configured
    let auth_service = match &state.auth_service {
        Some(service) => service,
        None => {
            let response = LdapGroupResponse {
                success: false,
                groups: vec![],
                message: "Authentication service not configured".to_string(),
            };
            return Ok((StatusCode::SERVICE_UNAVAILABLE, Json(response)).into_response());
        }
    };

    if !auth_service.is_ldap_enabled() {
        let response = LdapGroupResponse {
            success: false,
            groups: vec![],
            message: "LDAP authentication not configured".to_string(),
        };
        return Ok((StatusCode::NOT_IMPLEMENTED, Json(response)).into_response());
    }

    // Get user groups
    match auth_service.get_ldap_user_groups(&params.username).await {
        Ok(groups) => {
            let group_info: Vec<LdapGroupInfo> = groups
                .into_iter()
                .map(|g| LdapGroupInfo {
                    dn: g.dn,
                    name: g.cn,
                    description: g.description,
                })
                .collect();

            let response = LdapGroupResponse {
                success: true,
                groups: group_info,
                message: format!("Found {} groups for user", response.groups.len()),
            };
            Ok((StatusCode::OK, Json(response)).into_response())
        }
        Err(e) => {
            error!("Failed to get LDAP groups: {}", e);
            let response = LdapGroupResponse {
                success: false,
                groups: vec![],
                message: format!("Failed to retrieve groups: {}", e),
            };
            Ok((StatusCode::INTERNAL_SERVER_ERROR, Json(response)).into_response())
        }
    }
}

/// Get current LDAP configuration (without sensitive data)
pub async fn get_ldap_config(
    State(state): State<Arc<AppState>>,
) -> Result<Response, StatusCode> {
    debug!("Getting LDAP configuration");

    // Check if LDAP is configured
    let auth_service = match &state.auth_service {
        Some(service) => service,
        None => {
            return Ok((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "success": false,
                    "message": "Authentication service not configured"
                }))
            ).into_response());
        }
    };

    if let Some(ldap_config) = &auth_service.config.ldap {
        let config_info = serde_json::json!({
            "success": true,
            "configured": true,
            "server": ldap_config.server,
            "use_tls": ldap_config.use_tls,
            "user_base_dn": ldap_config.user_base_dn,
            "group_base_dn": ldap_config.group_base_dn,
            "user_filter": ldap_config.user_filter,
            "group_filter": ldap_config.group_filter,
        });
        Ok((StatusCode::OK, Json(config_info)).into_response())
    } else {
        Ok((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "success": false,
                "configured": false,
                "message": "LDAP not configured"
            }))
        ).into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LdapConfig, SecurityConfig};
    use crate::auth::AuthService;

    #[tokio::test]
    async fn test_ldap_login_response_serialization() {
        let response = LdapLoginResponse {
            success: true,
            access_token: Some("test_token".to_string()),
            token_type: Some("Bearer".to_string()),
            expires_in: Some(3600),
            user: Some(User {
                username: "testuser".to_string(),
                roles: vec!["user".to_string()],
                email: Some("test@example.com".to_string()),
                full_name: Some("Test User".to_string()),
                last_login: None,
                permissions: vec![],
            }),
            message: "Success".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"access_token\":\"test_token\""));
    }
}