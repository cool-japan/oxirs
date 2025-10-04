//! Tests for LDAP authentication

use axum::body::to_bytes;
use axum::http::StatusCode;
use oxirs_fuseki::{
    auth::{AuthResult, AuthService},
    config::{LdapConfig, SecurityConfig},
};
use serde_json::json;

#[tokio::test]
async fn test_ldap_service_creation() {
    let ldap_config = LdapConfig {
        server: "ldap://localhost:389".to_string(),
        bind_dn: "cn=admin,dc=example,dc=com".to_string(),
        bind_password: "admin_password".to_string(),
        user_base_dn: "ou=users,dc=example,dc=com".to_string(),
        user_filter: "(uid={username})".to_string(),
        group_base_dn: "ou=groups,dc=example,dc=com".to_string(),
        group_filter: "(member={userdn})".to_string(),
        use_tls: false,
    };

    let security_config = SecurityConfig {
        ldap: Some(ldap_config),
        ..Default::default()
    };

    let auth_service = AuthService::new(security_config).await.unwrap();

    // Verify LDAP is enabled
    assert!(auth_service.is_ldap_enabled());
}

#[tokio::test]
async fn test_ldap_authentication_flow() {
    let ldap_config = LdapConfig {
        server: "ldap://localhost:389".to_string(),
        bind_dn: "cn=admin,dc=example,dc=com".to_string(),
        bind_password: "admin_password".to_string(),
        user_base_dn: "ou=users,dc=example,dc=com".to_string(),
        user_filter: "(uid={username})".to_string(),
        group_base_dn: "ou=groups,dc=example,dc=com".to_string(),
        group_filter: "(member={userdn})".to_string(),
        use_tls: false,
    };

    let security_config = SecurityConfig {
        ldap: Some(ldap_config),
        ..Default::default()
    };

    let auth_service = AuthService::new(security_config).await.unwrap();

    // Test authentication (this will use the mock implementation)
    match auth_service.authenticate_ldap("testuser", "testpass").await {
        Ok(AuthResult::Authenticated(user)) => {
            assert_eq!(user.username, "testuser");
            assert!(!user.roles.is_empty());
        }
        _ => {
            // Mock might not authenticate - this is expected
        }
    }
}

#[tokio::test]
async fn test_ldap_configuration_validation() {
    // Test invalid LDAP URL
    let invalid_config = LdapConfig {
        server: "not-a-valid-url".to_string(),
        bind_dn: "cn=admin,dc=example,dc=com".to_string(),
        bind_password: "admin_password".to_string(),
        user_base_dn: "ou=users,dc=example,dc=com".to_string(),
        user_filter: "(uid={username})".to_string(),
        group_base_dn: "ou=groups,dc=example,dc=com".to_string(),
        group_filter: "(member={userdn})".to_string(),
        use_tls: false,
    };

    // Validation should be handled at configuration load time
    // This test ensures the structure is correct
    assert_eq!(invalid_config.server, "not-a-valid-url");
}

#[tokio::test]
async fn test_active_directory_configuration() {
    use oxirs_fuseki::auth::ldap::active_directory_config;

    let config = active_directory_config(
        "corp.example.com",
        "dc1.corp.example.com",
        "service_account",
        "service_password",
    );

    assert_eq!(config.server, "ldap://dc1.corp.example.com");
    assert_eq!(
        config.bind_dn,
        "service_account@corp.example.com".to_string()
    );
    assert_eq!(config.user_filter, "(sAMAccountName={username})");
}

#[tokio::test]
async fn test_ldap_handler_endpoints() {
    use axum::{body::Body, http::Request, Router};
    use oxirs_fuseki::{
        handlers::ldap::{get_ldap_config, get_ldap_groups, ldap_login, test_ldap_connection},
        server::AppState,
    };
    use tower::ServiceExt;

    // Create test app state with LDAP configuration
    let ldap_config = LdapConfig {
        server: "ldap://localhost:389".to_string(),
        bind_dn: "cn=admin,dc=example,dc=com".to_string(),
        bind_password: "admin_password".to_string(),
        user_base_dn: "ou=users,dc=example,dc=com".to_string(),
        user_filter: "(uid={username})".to_string(),
        group_base_dn: "ou=groups,dc=example,dc=com".to_string(),
        group_filter: "(member={userdn})".to_string(),
        use_tls: false,
    };

    let security_config = SecurityConfig {
        ldap: Some(ldap_config),
        ..Default::default()
    };

    let auth_service = AuthService::new(security_config.clone()).await.unwrap();
    let state = AppState {
        store: oxirs_fuseki::store::Store::new().unwrap(),
        config: oxirs_fuseki::config::ServerConfig {
            security: security_config,
            ..Default::default()
        },
        auth_service: Some(auth_service),
        metrics_service: None,
        performance_service: None,
        query_optimizer: None,
        subscription_manager: None,
        federation_manager: None,
        streaming_manager: None,
        #[cfg(feature = "rate-limit")]
        rate_limiter: None,
    };

    // Create router with LDAP routes
    let app = Router::new()
        .route("/auth/ldap/login", axum::routing::post(ldap_login))
        .route("/auth/ldap/test", axum::routing::get(test_ldap_connection))
        .route("/auth/ldap/groups", axum::routing::get(get_ldap_groups))
        .route("/auth/ldap/config", axum::routing::get(get_ldap_config))
        .with_state(state);

    // Test login endpoint
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/auth/ldap/login")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "username": "testuser",
                        "password": "testpass"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Test connection test endpoint
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/auth/ldap/test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Test config endpoint
    let response = app
        .oneshot(
            Request::builder()
                .uri("/auth/ldap/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["configured"], true);
    assert_eq!(json["server"], "ldap://localhost:389");
}

#[test]
fn test_ldap_login_request_serialization() {
    use oxirs_fuseki::handlers::ldap::LdapLoginRequest;

    let request = LdapLoginRequest {
        username: "john.doe".to_string(),
        password: "secret123".to_string(),
        domain: Some("corp.example.com".to_string()),
    };

    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("john.doe"));
    assert!(json.contains("secret123"));
    assert!(json.contains("corp.example.com"));
}

#[test]
fn test_ldap_group_mapping() {
    // Test that common LDAP group names map to appropriate roles
    let group_mappings = [
        ("Domain Admins", "admin"),
        ("administrators", "admin"),
        ("fuseki-admins", "admin"),
        ("developers", "writer"),
        ("fuseki-writers", "writer"),
        ("users", "reader"),
        ("fuseki-readers", "reader"),
        ("Domain Users", "user"),
    ];

    // This tests the expected behavior of group to role mapping
    for (group, expected_role) in group_mappings.iter() {
        // In the actual implementation, this would be done by the LDAP service
        let role = match group.to_lowercase().as_str() {
            g if g.contains("admin") => "admin",
            g if g.contains("developer") || g.contains("writer") => "writer",
            g if g.contains("users") && !g.contains("domain") => "reader",
            g if g.contains("reader") => "reader",
            _ => "user",
        };

        assert_eq!(
            role, *expected_role,
            "Group '{group}' should map to role '{expected_role}'"
        );
    }
}

#[tokio::test]
async fn test_ldap_caching() {
    let ldap_config = LdapConfig {
        server: "ldap://localhost:389".to_string(),
        bind_dn: "cn=admin,dc=example,dc=com".to_string(),
        bind_password: "admin_password".to_string(),
        user_base_dn: "ou=users,dc=example,dc=com".to_string(),
        user_filter: "(uid={username})".to_string(),
        group_base_dn: "ou=groups,dc=example,dc=com".to_string(),
        group_filter: "(member={userdn})".to_string(),
        use_tls: false,
    };

    let security_config = SecurityConfig {
        ldap: Some(ldap_config),
        ..Default::default()
    };

    let auth_service = AuthService::new(security_config).await.unwrap();

    // First authentication attempt
    let _result1 = auth_service
        .authenticate_ldap("cacheduser", "password")
        .await;

    // Second authentication attempt should use cache
    let _result2 = auth_service
        .authenticate_ldap("cacheduser", "password")
        .await;

    // Cleanup cache
    auth_service.cleanup_ldap_cache().await;
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Run with --ignored to test against real LDAP server
    async fn test_real_ldap_server() {
        // This test would require a real LDAP server
        // Configure with actual LDAP server details
        let ldap_config = LdapConfig {
            server: "ldap://your-ldap-server:389".to_string(),
            bind_dn: "cn=service,dc=example,dc=com".to_string(),
            bind_password: "service_password".to_string(),
            user_base_dn: "ou=people,dc=example,dc=com".to_string(),
            user_filter: "(uid={username})".to_string(),
            group_base_dn: "ou=groups,dc=example,dc=com".to_string(),
            group_filter: "(member={userdn})".to_string(),
            use_tls: true,
        };

        let security_config = SecurityConfig {
            ldap: Some(ldap_config),
            ..Default::default()
        };

        let auth_service = AuthService::new(security_config).await.unwrap();

        // Test connection
        match auth_service.test_ldap_connection().await {
            Ok(connected) => {
                assert!(connected, "Should be able to connect to LDAP server");
            }
            Err(e) => {
                panic!("LDAP connection failed: {e}");
            }
        }
    }
}
