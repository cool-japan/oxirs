//! SHACL string-based constraints for validating literal values
//!
//! This module implements string-specific SHACL constraints that validate
//! properties of literal values including length, patterns, and language tags:
//!
//! - [`MinLengthConstraint`] - Validates minimum string length (`sh:minLength`)
//! - [`MaxLengthConstraint`] - Validates maximum string length (`sh:maxLength`)
//! - [`PatternConstraint`] - Validates against regular expressions (`sh:pattern`)
//! - [`LanguageInConstraint`] - Validates allowed language tags (`sh:languageIn`)
//! - [`UniqueLangConstraint`] - Ensures unique language tags (`sh:uniqueLang`)
//!
//! # Usage
//!
//! ```rust
//! use oxirs_shacl::constraints::string_constraints::*;
//!
//! // Create a constraint requiring at least 3 characters
//! let min_length = MinLengthConstraint {
//!     min_length: 3,
//! };
//!
//! // Create a pattern constraint for email validation
//! let email_pattern = PatternConstraint {
//!     pattern: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$".to_string(),
//!     flags: Some("i".to_string()), // case-insensitive
//!     message: Some("Must be a valid email address".to_string()),
//! };
//!
//! // Create a constraint allowing only English and French
//! let language_constraint = LanguageInConstraint {
//!     languages: vec!["en".to_string(), "fr".to_string()],
//! };
//! ```
//!
//! # SHACL Specification
//!
//! These constraints implement the string constraint components from the
//! [SHACL specification](https://www.w3.org/TR/shacl/#core-components-string):
//!
//! - `sh:minLength` - Specifies the minimum string length of each value node
//! - `sh:maxLength` - Specifies the maximum string length of each value node
//! - `sh:pattern` - Specifies a regular expression that each value node must match
//! - `sh:languageIn` - Specifies a list of language tags that each value node must have
//! - `sh:uniqueLang` - Specifies whether each value node must have a unique language tag

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use oxirs_core::{model::Term, Store};

use super::{
    ConstraintContext, ConstraintEvaluationResult, ConstraintEvaluator, ConstraintValidator,
};
use crate::{Result, ShaclError};

/// SHACL `sh:minLength` constraint that validates the minimum string length.
///
/// This constraint ensures that literal values have at least the specified number
/// of characters. The length is measured in Unicode characters, not bytes.
///
/// # SHACL Specification
///
/// From [SHACL Core Components - MinLength Constraint Component](https://www.w3.org/TR/shacl/#MinLengthConstraintComponent):
/// "Specifies the minimum string length of each value node that satisfies the condition."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::string_constraints::MinLengthConstraint;
///
/// // Require at least 8 characters (e.g., for password validation)
/// let password_length = MinLengthConstraint {
///     min_length: 8,
/// };
///
/// // Require at least 1 character (non-empty strings)
/// let non_empty = MinLengthConstraint {
///     min_length: 1,
/// };
/// ```
///
/// # Validation Behavior
///
/// - **Passes**: When literal values have length >= `min_length`
/// - **Fails**: When literal values have length < `min_length`
/// - **Fails**: When values are not literals (IRIs or blank nodes)
/// - **Unicode**: Length is counted in Unicode characters, handling emoji and multi-byte characters correctly
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MinLengthConstraint {
    /// The minimum number of characters required
    pub min_length: u32,
}

impl ConstraintValidator for MinLengthConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MinLengthConstraint {
    fn evaluate(
        &self,
        _store: &dyn Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            match value {
                Term::Literal(literal) => {
                    let string_value = literal.value();
                    if (string_value.chars().count() as u32) < self.min_length {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!(
                                "String length {} is less than minimum length {}",
                                string_value.chars().count(),
                                self.min_length
                            )),
                        ));
                    }
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some("Value must be a literal for length validation".to_string()),
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// SHACL `sh:maxLength` constraint that validates the maximum string length.
///
/// This constraint ensures that literal values have at most the specified number
/// of characters. The length is measured in Unicode characters, not bytes.
///
/// # SHACL Specification
///
/// From [SHACL Core Components - MaxLength Constraint Component](https://www.w3.org/TR/shacl/#MaxLengthConstraintComponent):
/// "Specifies the maximum string length of each value node that satisfies the condition."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::string_constraints::MaxLengthConstraint;
///
/// // Limit to 100 characters (e.g., for tweet-like content)
/// let tweet_length = MaxLengthConstraint {
///     max_length: 100,
/// };
///
/// // Limit to 255 characters (e.g., for database varchar fields)
/// let db_field_length = MaxLengthConstraint {
///     max_length: 255,
/// };
/// ```
///
/// # Validation Behavior
///
/// - **Passes**: When literal values have length <= `max_length`
/// - **Fails**: When literal values have length > `max_length`
/// - **Fails**: When values are not literals (IRIs or blank nodes)
/// - **Unicode**: Length is counted in Unicode characters, handling emoji and multi-byte characters correctly
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaxLengthConstraint {
    /// The maximum number of characters allowed
    pub max_length: u32,
}

impl ConstraintValidator for MaxLengthConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MaxLengthConstraint {
    fn evaluate(
        &self,
        _store: &dyn Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            match value {
                Term::Literal(literal) => {
                    let string_value = literal.value();
                    if (string_value.chars().count() as u32) > self.max_length {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!(
                                "String length {} is greater than maximum length {}",
                                string_value.chars().count(),
                                self.max_length
                            )),
                        ));
                    }
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some("Value must be a literal for length validation".to_string()),
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// SHACL `sh:pattern` constraint that validates literal values against a regular expression.
///
/// This constraint checks that each literal value matches the specified regular expression pattern.
/// It supports regex flags for case-insensitive matching, multiline mode, and other options.
///
/// # SHACL Specification
///
/// From [SHACL Core Components - Pattern Constraint Component](https://www.w3.org/TR/shacl/#PatternConstraintComponent):
/// "Specifies a regular expression that each value node must match to satisfy the condition."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::string_constraints::PatternConstraint;
///
/// // Email validation pattern
/// let email_constraint = PatternConstraint {
///     pattern: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$".to_string(),
///     flags: Some("i".to_string()), // case-insensitive
///     message: Some("Must be a valid email address".to_string()),
/// };
///
/// // Phone number pattern (US format)
/// let phone_constraint = PatternConstraint {
///     pattern: r"^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$".to_string(),
///     flags: None,
///     message: Some("Must be a valid US phone number".to_string()),
/// };
/// ```
///
/// # Regex Flags
///
/// The `flags` field supports standard regex flags:
/// - `i` - Case-insensitive matching
/// - `m` - Multiline mode (^ and $ match line boundaries)
/// - `s` - Dot matches newline characters
///
/// # Validation Behavior
///
/// - **Passes**: When literal values match the regular expression
/// - **Fails**: When literal values don't match the regular expression
/// - **Fails**: When values are not literals (IRIs or blank nodes)
/// - **Error**: When the pattern is an invalid regular expression
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatternConstraint {
    /// The regular expression pattern that values must match
    pub pattern: String,
    /// Optional regex flags (i=case-insensitive, m=multiline, s=dot-matches-newline)
    pub flags: Option<String>,
    /// Optional custom error message for pattern violations
    pub message: Option<String>,
}

impl ConstraintValidator for PatternConstraint {
    fn validate(&self) -> Result<()> {
        // Validate that the pattern is a valid regex
        let mut regex_builder = regex::RegexBuilder::new(&self.pattern);

        if let Some(flags) = &self.flags {
            // Parse regex flags
            let case_insensitive = flags.contains('i');
            let multi_line = flags.contains('m');
            let dot_matches_new_line = flags.contains('s');

            let _regex = regex_builder
                .case_insensitive(case_insensitive)
                .multi_line(multi_line)
                .dot_matches_new_line(dot_matches_new_line)
                .build()
                .map_err(|e| {
                    ShaclError::ConstraintValidation(format!(
                        "Invalid regex pattern '{}': {}",
                        self.pattern, e
                    ))
                })?;
        } else {
            let _regex = Regex::new(&self.pattern).map_err(|e| {
                ShaclError::ConstraintValidation(format!(
                    "Invalid regex pattern '{}': {}",
                    self.pattern, e
                ))
            })?;
        }

        Ok(())
    }
}

impl ConstraintEvaluator for PatternConstraint {
    fn evaluate(
        &self,
        _store: &dyn Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // Build the regex with flags
        let regex = if let Some(flags) = &self.flags {
            let case_insensitive = flags.contains('i');
            let multi_line = flags.contains('m');
            let dot_matches_new_line = flags.contains('s');

            regex::RegexBuilder::new(&self.pattern)
                .case_insensitive(case_insensitive)
                .multi_line(multi_line)
                .dot_matches_new_line(dot_matches_new_line)
                .build()
                .map_err(|e| {
                    ShaclError::ConstraintValidation(format!(
                        "Invalid regex pattern '{}': {}",
                        self.pattern, e
                    ))
                })?
        } else {
            Regex::new(&self.pattern).map_err(|e| {
                ShaclError::ConstraintValidation(format!(
                    "Invalid regex pattern '{}': {}",
                    self.pattern, e
                ))
            })?
        };

        // Check each value against the pattern
        for value in &context.values {
            match value {
                Term::Literal(literal) => {
                    let string_value = literal.value();
                    if !regex.is_match(string_value) {
                        let message = self.message.clone().unwrap_or_else(|| {
                            format!(
                                "Value '{}' does not match pattern '{}'",
                                string_value, self.pattern
                            )
                        });
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(message),
                        ));
                    }
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value {value} is not a literal, cannot check pattern"
                        )),
                    ));
                }
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// SHACL `sh:languageIn` constraint that validates language tags of literal values.
///
/// This constraint ensures that language-tagged literals have language tags from
/// the specified list. It enforces internationalization policies and language consistency.
///
/// # SHACL Specification
///
/// From [SHACL Core Components - LanguageIn Constraint Component](https://www.w3.org/TR/shacl/#LanguageInConstraintComponent):
/// "Specifies a list of language tags that all value nodes must have."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::string_constraints::LanguageInConstraint;
///
/// // Allow only English and French content
/// let multilingual_constraint = LanguageInConstraint {
///     languages: vec!["en".to_string(), "fr".to_string()],
/// };
///
/// // Allow various English dialects
/// let english_dialects = LanguageInConstraint {
///     languages: vec![
///         "en".to_string(),
///         "en-US".to_string(),
///         "en-GB".to_string(),
///         "en-CA".to_string(),
///     ],
/// };
/// ```
///
/// # Language Tags
///
/// Language tags should follow [BCP 47](https://tools.ietf.org/html/bcp47) format:
/// - `en` - English
/// - `fr` - French
/// - `de` - German
/// - `en-US` - English (United States)
/// - `fr-CA` - French (Canada)
///
/// # Validation Behavior
///
/// - **Passes**: When literal values have language tags in the allowed list
/// - **Fails**: When literal values have language tags not in the allowed list
/// - **Fails**: When literal values have no language tag but constraint requires one
/// - **Fails**: When values are not literals (IRIs or blank nodes)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageInConstraint {
    /// List of allowed language tags (BCP 47 format)
    pub languages: Vec<String>,
}

impl ConstraintValidator for LanguageInConstraint {
    fn validate(&self) -> Result<()> {
        // Validate that all language tags are valid BCP 47 language tags
        for lang in &self.languages {
            if lang.is_empty() {
                return Err(ShaclError::ConstraintValidation(
                    "Empty language tag in sh:languageIn".to_string(),
                ));
            }
            // TODO: More thorough BCP 47 validation
        }
        Ok(())
    }
}

impl ConstraintEvaluator for LanguageInConstraint {
    fn evaluate(
        &self,
        _store: &dyn Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            match value {
                Term::Literal(literal) => {
                    if let Some(lang) = literal.language() {
                        if !self.languages.contains(&lang.to_string()) {
                            return Ok(ConstraintEvaluationResult::violated(
                                Some(value.clone()),
                                Some(format!(
                                    "Language tag '{}' is not in allowed languages: {:?}",
                                    lang, self.languages
                                )),
                            ));
                        }
                    } else if !self.languages.is_empty() {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(
                                "Literal has no language tag but constraint requires one"
                                    .to_string(),
                            ),
                        ));
                    }
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some("Value must be a literal for language validation".to_string()),
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// SHACL `sh:uniqueLang` constraint that ensures unique language tags across values.
///
/// This constraint validates that no two literal values have the same language tag.
/// It's useful for ensuring that multilingual content doesn't have duplicate translations.
///
/// # SHACL Specification
///
/// From [SHACL Core Components - UniqueLang Constraint Component](https://www.w3.org/TR/shacl/#UniqueLangConstraintComponent):
/// "Specifies whether each value node must have a unique language tag."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::string_constraints::UniqueLangConstraint;
///
/// // Ensure unique language tags (no duplicate translations)
/// let unique_translations = UniqueLangConstraint {
///     unique_lang: true,
/// };
///
/// // Allow duplicate language tags
/// let allow_duplicates = UniqueLangConstraint {
///     unique_lang: false,
/// };
/// ```
///
/// # Use Cases
///
/// - **Multilingual Content**: Ensure each language appears only once in translations
/// - **Resource Localization**: Validate that localized resources don't have duplicates
/// - **Data Quality**: Prevent accidental duplicate entries in language-specific data
///
/// # Validation Behavior
///
/// When `unique_lang` is `true`:
/// - **Passes**: When all literal values have different language tags (or no language tag)
/// - **Fails**: When two or more literal values share the same language tag
/// - **Fails**: When values are not literals (IRIs or blank nodes)
///
/// When `unique_lang` is `false`:
/// - **Always Passes**: No uniqueness constraint is applied
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniqueLangConstraint {
    /// Whether language tags must be unique across all values
    pub unique_lang: bool,
}

impl ConstraintValidator for UniqueLangConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for UniqueLangConstraint {
    fn evaluate(
        &self,
        _store: &dyn Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        if !self.unique_lang {
            // If uniqueLang is false, no constraint
            return Ok(ConstraintEvaluationResult::satisfied());
        }

        let mut seen_languages = HashSet::new();
        for value in &context.values {
            match value {
                Term::Literal(literal) => {
                    if let Some(lang) = literal.language() {
                        if seen_languages.contains(lang) {
                            return Ok(ConstraintEvaluationResult::violated(
                                Some(value.clone()),
                                Some(format!(
                                    "Duplicate language tag '{lang}' found, but unique languages required"
                                )),
                            ));
                        }
                        seen_languages.insert(lang);
                    }
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(
                            "Value must be a literal for language uniqueness validation"
                                .to_string(),
                        ),
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}
