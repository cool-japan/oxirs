//! OAuth2/OIDC authentication handlers for OxiRS Fuseki
//!
//! This module provides HTTP endpoints for OAuth2 and OpenID Connect authentication flows.

use crate::{
    auth::{AuthService, AuthResult},
    error::{FusekiError, FusekiResult},
    server::AppState,
};
use axum::{
    extract::{Query, State},
    http::{StatusCode, HeaderMap, header::{LOCATION, SET_COOKIE}},
    response::{Json, IntoResponse, Response, Redirect},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn, error, debug, instrument};

/// OAuth2 authorization request parameters
#[derive(Debug, Deserialize)]
pub struct OAuth2AuthParams {
    pub redirect_uri: Option<String>,
    pub scope: Option<String>,
    pub state: Option<String>,
    pub use_pkce: Option<bool>,
}

/// OAuth2 callback parameters
#[derive(Debug, Deserialize)]
pub struct OAuth2CallbackParams {
    pub code: Option<String>,
    pub state: Option<String>,
    pub error: Option<String>,
    pub error_description: Option<String>,
}

/// OAuth2 token refresh request
#[derive(Debug, Deserialize)]
pub struct OAuth2RefreshRequest {
    pub refresh_token: String,
}

/// OAuth2 authentication response
#[derive(Debug, Serialize)]
pub struct OAuth2AuthResponse {
    pub success: bool,
    pub authorization_url: Option<String>,
    pub state: Option<String>,
    pub message: String,
}

/// OAuth2 token response
#[derive(Debug, Serialize)]
pub struct OAuth2TokenResponse {
    pub success: bool,
    pub access_token: Option<String>,
    pub token_type: Option<String>,
    pub expires_in: Option<u64>,
    pub refresh_token: Option<String>,
    pub user: Option<crate::auth::User>,
    pub message: String,
}

/// OAuth2 user info response
#[derive(Debug, Serialize)]
pub struct OAuth2UserInfoResponse {
    pub success: bool,
    pub user_info: Option<crate::auth::oauth::OIDCUserInfo>,
    pub message: String,
}

/// Initiate OAuth2 authorization flow
#[instrument(skip(state))]
pub async fn initiate_oauth2_flow(
    State(state): State<AppState>,
    Query(params): Query<OAuth2AuthParams>,
) -> Result<Json<OAuth2AuthResponse>, FusekiError> {
    let auth_service = state.auth_service.as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    if !auth_service.is_oauth2_enabled() {
        return Err(FusekiError::service_unavailable("OAuth2 authentication not configured"));
    }

    // Determine redirect URI
    let redirect_uri = params.redirect_uri
        .unwrap_or_else(|| format!("{}://{}/auth/oauth2/callback", 
            if state.config.server.tls.is_some() { "https" } else { "http" },
            format!("{}:{}", state.config.server.host, state.config.server.port)
        ));

    // Parse scopes
    let scopes = params.scope
        .map(|s| s.split(' ').map(|scope| scope.to_string()).collect::<Vec<_>>())
        .unwrap_or_else(|| vec!["openid".to_string(), "profile".to_string(), "email".to_string()]);

    let use_pkce = params.use_pkce.unwrap_or(true);

    match auth_service.generate_oauth2_auth_url(&redirect_uri, &scopes, use_pkce).await {
        Ok((authorization_url, state_param)) => {
            info!("Generated OAuth2 authorization URL with state: {}", state_param);
            
            Ok(Json(OAuth2AuthResponse {
                success: true,
                authorization_url: Some(authorization_url),
                state: Some(state_param),
                message: "OAuth2 authorization URL generated successfully".to_string(),
            }))
        }
        Err(e) => {
            error!("Failed to generate OAuth2 authorization URL: {}", e);
            Err(e)
        }
    }
}

/// Handle OAuth2 authorization callback
#[instrument(skip(state))]
pub async fn handle_oauth2_callback(
    State(state): State<AppState>,
    Query(params): Query<OAuth2CallbackParams>,
) -> Result<Response, FusekiError> {
    let auth_service = state.auth_service.as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    // Check for OAuth2 errors
    if let Some(error) = params.error {
        warn!("OAuth2 authorization error: {} - {}", error, params.error_description.unwrap_or_default());
        
        let error_response = OAuth2TokenResponse {
            success: false,
            access_token: None,
            token_type: None,
            expires_in: None,
            refresh_token: None,
            user: None,
            message: format!("OAuth2 authorization failed: {}", error),
        };
        
        return Ok((StatusCode::BAD_REQUEST, Json(error_response)).into_response());
    }

    let code = params.code
        .ok_or_else(|| FusekiError::bad_request("Missing authorization code"))?;
    
    let state_param = params.state
        .ok_or_else(|| FusekiError::bad_request("Missing state parameter"))?;

    // Determine redirect URI (should match the one used in authorization)
    let redirect_uri = format!("{}://{}/auth/oauth2/callback", 
        if state.config.server.tls.is_some() { "https" } else { "http" },
        format!("{}:{}", state.config.server.host, state.config.server.port)
    );

    match auth_service.complete_oauth2_flow(&code, &state_param, &redirect_uri).await {
        Ok(AuthResult::Authenticated(user)) => {
            info!("OAuth2 authentication successful for user: {}", user.username);

            // Create session for the user
            let session_id = auth_service.create_session(user.clone()).await?;

            // Set session cookie
            let cookie_value = format!(
                "session_id={}; HttpOnly; Secure; SameSite=Strict; Max-Age={}",
                session_id,
                state.config.security.session.timeout_secs
            );

            let response = OAuth2TokenResponse {
                success: true,
                access_token: Some(session_id),
                token_type: Some("Bearer".to_string()),
                expires_in: Some(state.config.security.session.timeout_secs),
                refresh_token: None,
                user: Some(user),
                message: "OAuth2 authentication successful".to_string(),
            };

            let mut resp = (StatusCode::OK, Json(response)).into_response();
            resp.headers_mut().insert(SET_COOKIE, cookie_value.parse().unwrap());

            Ok(resp)
        }
        Ok(_) => {
            warn!("OAuth2 authentication failed");
            
            let response = OAuth2TokenResponse {
                success: false,
                access_token: None,
                token_type: None,
                expires_in: None,
                refresh_token: None,
                user: None,
                message: "OAuth2 authentication failed".to_string(),
            };

            Ok((StatusCode::UNAUTHORIZED, Json(response)).into_response())
        }
        Err(e) => {
            error!("OAuth2 callback processing failed: {}", e);
            Err(e)
        }
    }
}

/// Refresh OAuth2 access token
#[instrument(skip(state, request))]
pub async fn refresh_oauth2_token(
    State(state): State<AppState>,
    Json(request): Json<OAuth2RefreshRequest>,
) -> Result<Json<OAuth2TokenResponse>, FusekiError> {
    let auth_service = state.auth_service.as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    if !auth_service.is_oauth2_enabled() {
        return Err(FusekiError::service_unavailable("OAuth2 authentication not configured"));
    }

    match auth_service.refresh_oauth2_token(&request.refresh_token).await {
        Ok(token) => {
            info!("OAuth2 token refreshed successfully");

            Ok(Json(OAuth2TokenResponse {
                success: true,
                access_token: Some(token.access_token),
                token_type: Some(token.token_type),
                expires_in: Some(token.expires_in),
                refresh_token: token.refresh_token,
                user: None, // Don't include user info in refresh response
                message: "Token refreshed successfully".to_string(),
            }))
        }
        Err(e) => {
            error!("Failed to refresh OAuth2 token: {}", e);
            Err(e)
        }
    }
}

/// Get OAuth2 user information
#[instrument(skip(state, headers))]
pub async fn get_oauth2_user_info(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<OAuth2UserInfoResponse>, FusekiError> {
    let auth_service = state.auth_service.as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    if !auth_service.is_oauth2_enabled() {
        return Err(FusekiError::service_unavailable("OAuth2 authentication not configured"));
    }

    // Extract access token from Authorization header
    let access_token = extract_bearer_token(&headers)
        .ok_or_else(|| FusekiError::authentication("Missing or invalid authorization header"))?;

    match auth_service.get_oidc_user_info(&access_token).await {
        Ok(user_info) => {
            info!("Retrieved OAuth2 user info for subject: {}", user_info.sub);

            Ok(Json(OAuth2UserInfoResponse {
                success: true,
                user_info: Some(user_info),
                message: "User information retrieved successfully".to_string(),
            }))
        }
        Err(e) => {
            error!("Failed to retrieve OAuth2 user info: {}", e);
            Err(e)
        }
    }
}

/// Validate OAuth2 access token
#[instrument(skip(state, headers))]
pub async fn validate_oauth2_token(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, FusekiError> {
    let auth_service = state.auth_service.as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    if !auth_service.is_oauth2_enabled() {
        return Err(FusekiError::service_unavailable("OAuth2 authentication not configured"));
    }

    // Extract access token from Authorization header
    let access_token = extract_bearer_token(&headers)
        .ok_or_else(|| FusekiError::authentication("Missing or invalid authorization header"))?;

    match auth_service.validate_oauth2_token(&access_token).await {
        Ok(is_valid) => {
            Ok(Json(serde_json::json!({
                "valid": is_valid,
                "message": if is_valid { "Token is valid" } else { "Token is invalid or expired" },
                "timestamp": chrono::Utc::now()
            })))
        }
        Err(e) => {
            error!("Failed to validate OAuth2 token: {}", e);
            Err(e)
        }
    }
}

/// Get OAuth2 configuration and capabilities
#[instrument(skip(state))]
pub async fn get_oauth2_config(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, FusekiError> {
    let auth_service = state.auth_service.as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    if !auth_service.is_oauth2_enabled() {
        return Err(FusekiError::service_unavailable("OAuth2 authentication not configured"));
    }

    // Get OAuth2 configuration from the config (without sensitive data)
    let oauth_config = &state.config.security.oauth;
    
    if let Some(oauth_config) = oauth_config {
        let config_info = serde_json::json!({
            "enabled": true,
            "provider": oauth_config.provider,
            "authorization_endpoint": oauth_config.auth_url,
            "supported_scopes": oauth_config.scopes,
            "client_id": oauth_config.client_id,
            // Don't expose client_secret, token_url, or user_info_url for security
        });

        Ok(Json(config_info))
    } else {
        Ok(Json(serde_json::json!({
            "enabled": false,
            "message": "OAuth2 not configured"
        })))
    }
}

/// OAuth2 discovery endpoint (OpenID Connect Discovery)
#[instrument(skip(state))]
pub async fn oauth2_discovery(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, FusekiError> {
    let auth_service = state.auth_service.as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    if !auth_service.is_oauth2_enabled() {
        return Err(FusekiError::service_unavailable("OAuth2 authentication not configured"));
    }

    let oauth2_service = auth_service.oauth2_service()
        .ok_or_else(|| FusekiError::service_unavailable("OAuth2 service not available"))?;

    match oauth2_service.discover_oidc_config().await {
        Ok(discovery) => {
            Ok(Json(serde_json::to_value(discovery).unwrap_or_default()))
        }
        Err(e) => {
            error!("Failed to discover OAuth2 configuration: {}", e);
            Err(e)
        }
    }
}

/// Extract Bearer token from Authorization header
fn extract_bearer_token(headers: &HeaderMap) -> Option<String> {
    let auth_header = headers.get("authorization")?;
    let auth_str = auth_header.to_str().ok()?;
    
    if auth_str.starts_with("Bearer ") {
        Some(auth_str[7..].to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn test_bearer_token_extraction() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer test_token_123"));
        
        let token = extract_bearer_token(&headers);
        assert_eq!(token, Some("test_token_123".to_string()));
        
        // Test invalid format
        headers.insert("authorization", HeaderValue::from_static("Basic dGVzdA=="));
        let token = extract_bearer_token(&headers);
        assert_eq!(token, None);
        
        // Test missing header
        headers.remove("authorization");
        let token = extract_bearer_token(&headers);
        assert_eq!(token, None);
    }

    #[test]
    fn test_oauth2_response_serialization() {
        let response = OAuth2AuthResponse {
            success: true,
            authorization_url: Some("https://provider.example.com/auth".to_string()),
            state: Some("state123".to_string()),
            message: "Success".to_string(),
        };
        
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("authorization_url"));
        assert!(json.contains("state123"));
    }
}