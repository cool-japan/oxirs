//! JWT token handling and validation

#[cfg(feature = "auth")]
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};

use crate::auth::types::{AuthError, Claims, User, TokenValidation};
use crate::config::JwtConfig;
use crate::error::{FusekiError, FusekiResult};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// JWT token manager
pub struct JwtManager {
    #[cfg(feature = "auth")]
    encoding_key: EncodingKey,
    #[cfg(feature = "auth")]
    decoding_key: DecodingKey,
    #[cfg(feature = "auth")]
    algorithm: Algorithm,
    issuer: String,
    audience: String,
    expiration_hours: i64,
}

impl JwtManager {
    /// Create a new JWT manager
    pub fn new(config: &JwtConfig) -> FusekiResult<Self> {
        #[cfg(feature = "auth")]
        {
            let encoding_key = EncodingKey::from_secret(config.secret.as_bytes());
            let decoding_key = DecodingKey::from_secret(config.secret.as_bytes());
            let algorithm = match config.algorithm.as_str() {
                "HS256" => Algorithm::HS256,
                "HS384" => Algorithm::HS384,
                "HS512" => Algorithm::HS512,
                _ => return Err(FusekiError::configuration(format!("Unsupported JWT algorithm: {}", config.algorithm))),
            };

            Ok(Self {
                encoding_key,
                decoding_key,
                algorithm,
                issuer: config.issuer.clone(),
                audience: config.audience.clone(),
                expiration_hours: config.expiration_hours,
            })
        }
        #[cfg(not(feature = "auth"))]
        {
            Ok(Self {
                issuer: config.issuer.clone(),
                audience: config.audience.clone(),
                expiration_hours: config.expiration_hours,
            })
        }
    }

    /// Generate a JWT token for a user
    #[cfg(feature = "auth")]
    pub fn generate_token(&self, user: &User) -> FusekiResult<String> {
        let now = Utc::now();
        let expiration = now + Duration::hours(self.expiration_hours);

        let claims = Claims {
            sub: user.username.clone(),
            exp: expiration.timestamp() as usize,
            iat: now.timestamp() as usize,
            iss: self.issuer.clone(),
            aud: self.audience.clone(),
            roles: user.roles.clone(),
            permissions: user.permissions.clone(),
        };

        encode(&Header::new(self.algorithm), &claims, &self.encoding_key)
            .map_err(|e| FusekiError::authentication(format!("Failed to generate JWT token: {}", e)))
    }

    /// Validate a JWT token and return user information
    #[cfg(feature = "auth")]
    pub fn validate_token(&self, token: &str) -> FusekiResult<TokenValidation> {
        let mut validation = Validation::new(self.algorithm);
        validation.set_issuer(&[self.issuer.clone()]);
        validation.set_audience(&[self.audience.clone()]);

        let token_data = decode::<Claims>(token, &self.decoding_key, &validation)
            .map_err(|e| FusekiError::authentication(format!("Invalid JWT token: {}", e)))?;

        let claims = token_data.claims;
        
        // Check if token is expired
        let exp_time = DateTime::from_timestamp(claims.exp as i64, 0)
            .ok_or_else(|| FusekiError::authentication("Invalid expiration time in token".to_string()))?;
        
        if Utc::now() > exp_time {
            return Err(FusekiError::authentication("Token has expired".to_string()));
        }

        let user = User {
            username: claims.sub,
            roles: claims.roles,
            email: None, // JWT doesn't store email
            full_name: None, // JWT doesn't store full name
            last_login: None,
            permissions: claims.permissions,
        };

        Ok(TokenValidation {
            user,
            expires_at: exp_time,
        })
    }

    /// Extract token from authorization header
    pub fn extract_token_from_header(auth_header: &str) -> Option<&str> {
        if auth_header.starts_with("Bearer ") {
            Some(&auth_header[7..])
        } else {
            None
        }
    }

    /// Generate a refresh token
    #[cfg(feature = "auth")]
    pub fn generate_refresh_token(&self, user: &User) -> FusekiResult<String> {
        let now = Utc::now();
        let expiration = now + Duration::days(30); // Refresh tokens last 30 days

        let claims = Claims {
            sub: user.username.clone(),
            exp: expiration.timestamp() as usize,
            iat: now.timestamp() as usize,
            iss: self.issuer.clone(),
            aud: format!("{}-refresh", self.audience),
            roles: user.roles.clone(),
            permissions: user.permissions.clone(),
        };

        encode(&Header::new(self.algorithm), &claims, &self.encoding_key)
            .map_err(|e| FusekiError::authentication(format!("Failed to generate refresh token: {}", e)))
    }

    /// Validate a refresh token
    #[cfg(feature = "auth")]
    pub fn validate_refresh_token(&self, token: &str) -> FusekiResult<TokenValidation> {
        let mut validation = Validation::new(self.algorithm);
        validation.set_issuer(&[self.issuer.clone()]);
        validation.set_audience(&[format!("{}-refresh", self.audience)]);

        let token_data = decode::<Claims>(token, &self.decoding_key, &validation)
            .map_err(|e| FusekiError::authentication(format!("Invalid refresh token: {}", e)))?;

        let claims = token_data.claims;
        
        let exp_time = DateTime::from_timestamp(claims.exp as i64, 0)
            .ok_or_else(|| FusekiError::authentication("Invalid expiration time in token".to_string()))?;
        
        if Utc::now() > exp_time {
            return Err(FusekiError::authentication("Refresh token has expired".to_string()));
        }

        let user = User {
            username: claims.sub,
            roles: claims.roles,
            email: None,
            full_name: None,
            last_login: None,
            permissions: claims.permissions,
        };

        Ok(TokenValidation {
            user,
            expires_at: exp_time,
        })
    }

    /// Get token expiration time
    #[cfg(feature = "auth")]
    pub fn get_token_expiration(&self, token: &str) -> FusekiResult<DateTime<Utc>> {
        let validation = Validation::new(self.algorithm);
        let token_data = decode::<Claims>(token, &self.decoding_key, &validation)
            .map_err(|e| FusekiError::authentication(format!("Invalid token: {}", e)))?;

        DateTime::from_timestamp(token_data.claims.exp as i64, 0)
            .ok_or_else(|| FusekiError::authentication("Invalid expiration time in token".to_string()))
    }

    /// Check if token is close to expiration (within 1 hour)
    #[cfg(feature = "auth")]
    pub fn is_token_close_to_expiration(&self, token: &str) -> FusekiResult<bool> {
        let expiration = self.get_token_expiration(token)?;
        let one_hour_from_now = Utc::now() + Duration::hours(1);
        Ok(expiration <= one_hour_from_now)
    }

    /// Stub implementations when auth feature is disabled
    #[cfg(not(feature = "auth"))]
    pub fn generate_token(&self, _user: &User) -> FusekiResult<String> {
        Err(FusekiError::configuration("JWT authentication is disabled. Enable the 'auth' feature to use JWT tokens.".to_string()))
    }

    #[cfg(not(feature = "auth"))]
    pub fn validate_token(&self, _token: &str) -> FusekiResult<TokenValidation> {
        Err(FusekiError::configuration("JWT authentication is disabled. Enable the 'auth' feature to use JWT tokens.".to_string()))
    }

    #[cfg(not(feature = "auth"))]
    pub fn generate_refresh_token(&self, _user: &User) -> FusekiResult<String> {
        Err(FusekiError::configuration("JWT authentication is disabled. Enable the 'auth' feature to use JWT tokens.".to_string()))
    }

    #[cfg(not(feature = "auth"))]
    pub fn validate_refresh_token(&self, _token: &str) -> FusekiResult<TokenValidation> {
        Err(FusekiError::configuration("JWT authentication is disabled. Enable the 'auth' feature to use JWT tokens.".to_string()))
    }
}