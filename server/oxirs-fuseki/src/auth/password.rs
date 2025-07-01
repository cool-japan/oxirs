//! Password hashing and verification utilities

use crate::auth::types::{AuthError, PasswordStrength};
use crate::error::{FusekiError, FusekiResult};
use argon2::password_hash::{rand_core::OsRng, SaltString};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use regex::Regex;
use std::collections::HashSet;

/// Password utility functions
pub struct PasswordUtils;

impl PasswordUtils {
    /// Hash a password using Argon2
    pub fn hash_password(password: &str) -> FusekiResult<String> {
        let salt = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();

        argon2
            .hash_password(password.as_bytes(), &salt)
            .map(|hash| hash.to_string())
            .map_err(|e| FusekiError::authentication(format!("Failed to hash password: {}", e)))
    }

    /// Verify a password against its hash
    pub fn verify_password(password: &str, hash: &str) -> FusekiResult<bool> {
        let parsed_hash = PasswordHash::new(hash)
            .map_err(|e| FusekiError::authentication(format!("Invalid password hash: {}", e)))?;

        let argon2 = Argon2::default();
        Ok(argon2
            .verify_password(password.as_bytes(), &parsed_hash)
            .is_ok())
    }

    /// Check password strength
    pub fn check_password_strength(password: &str) -> PasswordStrength {
        let mut score = 0;
        
        // Length scoring
        if password.len() >= 8 { score += 1; }
        if password.len() >= 12 { score += 1; }
        if password.len() >= 16 { score += 1; }
        
        // Character variety scoring
        if password.chars().any(|c| c.is_lowercase()) { score += 1; }
        if password.chars().any(|c| c.is_uppercase()) { score += 1; }
        if password.chars().any(|c| c.is_numeric()) { score += 1; }
        if password.chars().any(|c| c.is_ascii_punctuation()) { score += 1; }
        
        // Additional complexity
        if password.len() >= 20 { score += 1; }
        if Self::has_mixed_case_and_numbers_and_symbols(password) { score += 1; }
        
        match score {
            0..=2 => PasswordStrength::VeryWeak,
            3..=4 => PasswordStrength::Weak,
            5..=6 => PasswordStrength::Medium,
            7..=8 => PasswordStrength::Strong,
            _ => PasswordStrength::VeryStrong,
        }
    }

    /// Validate password meets minimum requirements
    pub fn validate_password(password: &str) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if password.len() < 8 {
            errors.push("Password must be at least 8 characters long".to_string());
        }

        if password.len() > 128 {
            errors.push("Password must be no more than 128 characters long".to_string());
        }

        if !password.chars().any(|c| c.is_lowercase()) {
            errors.push("Password must contain at least one lowercase letter".to_string());
        }

        if !password.chars().any(|c| c.is_uppercase()) {
            errors.push("Password must contain at least one uppercase letter".to_string());
        }

        if !password.chars().any(|c| c.is_numeric()) {
            errors.push("Password must contain at least one number".to_string());
        }

        if !password.chars().any(|c| c.is_ascii_punctuation()) {
            errors.push("Password must contain at least one special character".to_string());
        }

        // Check for common patterns
        if Self::contains_common_patterns(password) {
            errors.push("Password contains common patterns and is not secure".to_string());
        }

        // Check for dictionary words
        if Self::contains_dictionary_words(password) {
            errors.push("Password contains common dictionary words".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Generate a secure random password
    pub fn generate_password(length: usize) -> String {
        use rand::Rng;
        
        const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz\
                                 ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                 0123456789\
                                 !@#$%^&*()_+-=[]{}|;:,.<>?";
        
        let mut rng = rand::thread_rng();
        let password: String = (0..length)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect();
        
        // Ensure the generated password meets requirements
        if Self::validate_password(&password).is_ok() {
            password
        } else {
            // Recursively generate until we get a valid one
            Self::generate_password(length)
        }
    }

    /// Check if password has good character variety
    fn has_mixed_case_and_numbers_and_symbols(password: &str) -> bool {
        let has_lower = password.chars().any(|c| c.is_lowercase());
        let has_upper = password.chars().any(|c| c.is_uppercase());
        let has_digit = password.chars().any(|c| c.is_numeric());
        let has_symbol = password.chars().any(|c| c.is_ascii_punctuation());
        
        has_lower && has_upper && has_digit && has_symbol
    }

    /// Check for common password patterns
    fn contains_common_patterns(password: &str) -> bool {
        let common_patterns = vec![
            r"123456",
            r"password",
            r"qwerty",
            r"abc123",
            r"admin",
            r"letmein",
            r"welcome",
            r"monkey",
            r"dragon",
            r"pass",
        ];

        let lower_password = password.to_lowercase();
        
        for pattern in common_patterns {
            if lower_password.contains(pattern) {
                return true;
            }
        }
        
        // Check for sequential patterns
        let sequential_patterns = vec![
            r"(?i)abcde",
            r"(?i)12345",
            r"(?i)qwert",
            r"(?i)asdfg",
            r"(?i)zxcvb",
        ];
        
        for pattern in sequential_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                if regex.is_match(password) {
                    return true;
                }
            }
        }
        
        false
    }

    /// Check for common dictionary words
    fn contains_dictionary_words(password: &str) -> bool {
        let common_words = vec![
            "password", "admin", "user", "login", "system", "computer",
            "server", "database", "access", "secret", "private", "public",
            "test", "demo", "guest", "root", "master", "default",
        ];

        let lower_password = password.to_lowercase();
        
        for word in common_words {
            if lower_password.contains(word) {
                return true;
            }
        }
        
        false
    }

    /// Generate password reset token
    pub fn generate_reset_token() -> String {
        use rand::Rng;
        
        const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz\
                                 ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                 0123456789";
        
        let mut rng = rand::thread_rng();
        (0..32)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    }

    /// Validate reset token format
    pub fn is_valid_reset_token(token: &str) -> bool {
        token.len() == 32 && token.chars().all(|c| c.is_alphanumeric())
    }
}