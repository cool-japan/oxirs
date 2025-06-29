//! String constraints (MinLength, MaxLength, Pattern, LanguageIn, UniqueLang)

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use oxirs_core::{model::Term, Store};

use crate::{Result, ShaclError};
use super::{ConstraintValidator, ConstraintEvaluator, ConstraintContext, ConstraintEvaluationResult};

/// sh:minLength constraint - validates minimum string length
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MinLengthConstraint {
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
        _store: &Store,
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

/// sh:maxLength constraint - validates maximum string length
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaxLengthConstraint {
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
        _store: &Store,
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

/// sh:pattern constraint - validates against regular expression
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatternConstraint {
    pub pattern: String,
    pub flags: Option<String>,
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
        store: &Store,
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
                            "Value {} is not a literal, cannot check pattern",
                            value
                        )),
                    ));
                }
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:languageIn constraint - validates language tags
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageInConstraint {
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
        _store: &Store,
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

/// sh:uniqueLang constraint - validates unique language tags
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniqueLangConstraint {
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
        _store: &Store,
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
                                    "Duplicate language tag '{}' found, but unique languages required",
                                    lang
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