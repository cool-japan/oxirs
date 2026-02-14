//! Text preprocessing for transformer models

use super::types::{DomainPreprocessingRules, TransformerConfig, TransformerType};
use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;

/// Text preprocessor for transformer models
#[derive(Debug, Clone)]
pub struct TransformerPreprocessor {
    config: TransformerConfig,
    domain_rules: Option<DomainPreprocessingRules>,
}

impl TransformerPreprocessor {
    pub fn new(config: TransformerConfig) -> Self {
        let domain_rules = match config.transformer_type {
            TransformerType::SciBERT => Some(DomainPreprocessingRules::scientific()),
            TransformerType::BioBERT => Some(DomainPreprocessingRules::biomedical()),
            TransformerType::LegalBERT => Some(DomainPreprocessingRules::legal()),
            TransformerType::NewsBERT => Some(DomainPreprocessingRules::news()),
            TransformerType::SocialMediaBERT => Some(DomainPreprocessingRules::social_media()),
            _ => None,
        };

        Self {
            config,
            domain_rules,
        }
    }

    /// Main preprocessing function that routes to appropriate domain-specific methods
    pub fn preprocess_text(&self, text: &str) -> String {
        let mut processed = text.to_string();

        // Common preprocessing steps
        processed = self.clean_uri(&processed);
        processed = self.normalize_whitespace(&processed);

        // Apply domain-specific preprocessing
        processed = match self.config.transformer_type {
            TransformerType::SciBERT => self.preprocess_scientific_text(&processed),
            TransformerType::BioBERT => self.preprocess_biomedical_text(&processed),
            TransformerType::CodeBERT => self.preprocess_code_text(&processed),
            TransformerType::LegalBERT => self.preprocess_legal_text(&processed),
            TransformerType::NewsBERT => self.preprocess_news_text(&processed),
            TransformerType::SocialMediaBERT => self.preprocess_social_media_text(&processed),
            _ => processed,
        };

        // Apply general domain rules if available
        if let Some(ref rules) = self.domain_rules {
            processed = self.apply_domain_rules(&processed, rules);
        }

        processed
    }

    /// Clean URI components for better semantic representation
    fn clean_uri(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Remove protocol prefixes
        result = result.replace("http://", "");
        result = result.replace("https://", "");
        result = result.replace("ftp://", "");

        // Replace common URI separators with spaces
        result = result.replace('/', " ");
        result = result.replace('#', " ");
        result = result.replace('?', " ");
        result = result.replace('&', " and ");
        result = result.replace('=', " equals ");

        // Handle underscores in URIs (common in ontologies)
        result = result.replace('_', " ");

        result
    }

    /// Normalize whitespace
    fn normalize_whitespace(&self, text: &str) -> String {
        // Replace multiple whitespace with single space
        let re = Regex::new(r"\s+").expect("regex pattern should be valid");
        re.replace_all(text, " ").trim().to_string()
    }

    /// Apply domain-specific preprocessing rules
    fn apply_domain_rules(&self, text: &str, rules: &DomainPreprocessingRules) -> String {
        let mut result = text.to_string();

        // Apply abbreviation expansions
        for (abbrev, expansion) in &rules.abbreviation_expansions {
            result = result.replace(abbrev, expansion);
        }

        // Apply pattern replacements
        for (pattern, replacement) in &rules.domain_specific_patterns {
            if let Ok(re) = Regex::new(pattern) {
                result = re.replace_all(&result, replacement).to_string();
            }
        }

        result
    }

    /// Preprocessing for scientific text (SciBERT)
    pub fn preprocess_scientific_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Scientific abbreviations
        let scientific_abbrevs = HashMap::from([
            ("DNA", "deoxyribonucleic acid"),
            ("RNA", "ribonucleic acid"),
            ("ATP", "adenosine triphosphate"),
            ("GDP", "guanosine diphosphate"),
            ("GTP", "guanosine triphosphate"),
            ("Co2", "carbon dioxide"),
            ("H2O", "water"),
            ("NaCl", "sodium chloride"),
        ]);

        for (abbrev, expansion) in scientific_abbrevs {
            result = result.replace(abbrev, expansion);
        }

        // Handle scientific notation and units
        result = result.replace("°C", " degrees celsius");
        result = result.replace("mg/ml", " milligrams per milliliter");
        result = result.replace("μg/ml", " micrograms per milliliter");
        result = result.replace("mM", " millimolar");
        result = result.replace("μM", " micromolar");

        // Handle chemical formulas and reactions
        result = result.replace("->", " produces ");
        result = result.replace("<->", " is in equilibrium with ");

        result
    }

    /// Preprocessing for biomedical text (BioBERT)
    pub fn preprocess_biomedical_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Biomedical gene and protein abbreviations
        let biomedical_abbrevs = HashMap::from([
            ("p53", "tumor protein p53"),
            ("BRCA1", "breast cancer gene 1"),
            ("BRCA2", "breast cancer gene 2"),
            ("TNF-α", "tumor necrosis factor alpha"),
            ("IL-1", "interleukin 1"),
            ("IL-6", "interleukin 6"),
            ("mRNA", "messenger ribonucleic acid"),
            ("tRNA", "transfer ribonucleic acid"),
            ("rRNA", "ribosomal ribonucleic acid"),
            ("CNS", "central nervous system"),
            ("PNS", "peripheral nervous system"),
        ]);

        for (abbrev, expansion) in biomedical_abbrevs {
            result = result.replace(abbrev, expansion);
        }

        // Handle medical terminology
        result = result.replace("bp", " base pairs");
        result = result.replace("kDa", " kilodaltons");
        result = result.replace("mg/kg", " milligrams per kilogram");

        result
    }

    /// Preprocessing for code text (CodeBERT)
    pub fn preprocess_code_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Programming language keywords and common terms
        result = result.replace("impl", "implementation");
        result = result.replace("func", "function");
        result = result.replace("var", "variable");
        result = result.replace("const", "constant");
        result = result.replace("struct", "structure");
        result = result.replace("enum", "enumeration");

        // Common type names
        result = result.replace("Vec<i32>", "vector of integers");
        result = result.replace("HashMap", "hash map");
        result = result.replace("String", "string");
        result = result.replace("bool", "boolean");

        // Expand camelCase and PascalCase
        result = self.expand_camel_case(&result);

        result
    }

    /// Preprocessing for legal text (LegalBERT)
    pub fn preprocess_legal_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Legal abbreviations
        let legal_abbrevs = HashMap::from([
            ("USC", "United States Code"),
            ("CFR", "Code of Federal Regulations"),
            ("plaintiff", "party bringing lawsuit"),
            ("defendant", "party being sued"),
            ("tort", "civil wrong"),
            ("v.", "versus"),
            ("vs.", "versus"),
            ("et al.", "and others"),
            ("cf.", "compare"),
            ("ibid.", "in the same place"),
            ("supra", "above mentioned"),
        ]);

        for (abbrev, expansion) in legal_abbrevs {
            result = result.replace(abbrev, expansion);
        }

        // Handle legal citations
        let section_re = Regex::new(r"§(\d+)").expect("regex pattern should be valid");
        result = section_re.replace_all(&result, "section $1").to_string();

        result
    }

    /// Preprocessing for news text (NewsBERT)
    pub fn preprocess_news_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Business and economics abbreviations
        let news_abbrevs = HashMap::from([
            ("CEO", "chief executive officer"),
            ("CFO", "chief financial officer"),
            ("CTO", "chief technology officer"),
            ("IPO", "initial public offering"),
            ("SEC", "Securities and Exchange Commission"),
            ("GDP", "gross domestic product"),
            ("CPI", "consumer price index"),
            ("NYSE", "New York Stock Exchange"),
            (
                "NASDAQ",
                "National Association of Securities Dealers Automated Quotations",
            ),
        ]);

        for (abbrev, expansion) in news_abbrevs {
            result = result.replace(abbrev, expansion);
        }

        // Handle financial terms
        result = result.replace("Q1", "first quarter");
        result = result.replace("Q2", "second quarter");
        result = result.replace("Q3", "third quarter");
        result = result.replace("Q4", "fourth quarter");

        // Handle percentages
        let percent_re = Regex::new(r"(\d+\.?\d*)%").expect("regex pattern should be valid");
        result = percent_re.replace_all(&result, "$1 percent").to_string();

        result
    }

    /// Preprocessing for social media text (SocialMediaBERT)
    pub fn preprocess_social_media_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Handle social media abbreviations
        result = result.replace("lol", "laugh out loud");
        result = result.replace("omg", "oh my god");
        result = result.replace("btw", "by the way");
        result = result.replace("fyi", "for your information");
        result = result.replace("imo", "in my opinion");
        result = result.replace("tbh", "to be honest");
        result = result.replace("smh", "shaking my head");
        result = result.replace("rn", "right now");
        result = result.replace("irl", "in real life");

        // Handle hashtags and mentions
        result = result.replace('#', "hashtag ");
        result = result.replace('@', "mention ");

        // Handle emoticons (basic)
        result = result.replace(":)", "happy");
        result = result.replace(":(", "sad");
        result = result.replace(":D", "very happy");
        result = result.replace(";)", "winking");
        result = result.replace(":P", "playful");
        result = result.replace(":/", "confused");

        // Handle emphasis
        result = result.replace("!!", "exclamation");
        result = result.replace("???", "question");

        result
    }

    /// Expand camelCase to separate words
    pub fn expand_camel_case(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        let mut result = String::new();
        let chars: Vec<char> = text.chars().collect();

        for (i, &ch) in chars.iter().enumerate() {
            // Add space before every uppercase letter (except the first character)
            if i > 0 && ch.is_uppercase() {
                result.push(' ');
            }

            result.push(ch.to_lowercase().next().unwrap_or(ch));
        }

        result
    }

    /// Simple tokenization for demonstration
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // In real implementation, this would use a proper tokenizer
        // For now, convert each character to a token
        let tokens: Vec<u32> = text
            .chars()
            .take(self.config.max_sequence_length)
            .map(|c| c as u32 % 30522) // Map to vocab range
            .collect();

        Ok(tokens)
    }

    /// Get maximum sequence length
    pub fn max_sequence_length(&self) -> usize {
        self.config.max_sequence_length
    }

    /// Check if text should be truncated
    pub fn needs_truncation(&self, text: &str) -> bool {
        text.len() > self.config.max_sequence_length
    }

    /// Truncate text to maximum sequence length
    pub fn truncate_text(&self, text: &str) -> String {
        if text.len() <= self.config.max_sequence_length {
            text.to_string()
        } else {
            text.chars().take(self.config.max_sequence_length).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::transformer::types::TransformerType;

    #[test]
    fn test_scientific_preprocessing() {
        let config = TransformerConfig {
            transformer_type: TransformerType::SciBERT,
            ..Default::default()
        };
        let preprocessor = TransformerPreprocessor::new(config);

        let text = "DNA synthesis with ATP and Co2 at 25°C using 5mg/ml concentration";
        let processed = preprocessor.preprocess_scientific_text(text);
        assert!(processed.contains("deoxyribonucleic acid"));
        assert!(processed.contains("adenosine triphosphate"));
        assert!(processed.contains("carbon dioxide"));
        assert!(processed.contains("degrees celsius"));
        assert!(processed.contains("milligrams per milliliter"));
    }

    #[test]
    fn test_biomedical_preprocessing() {
        let config = TransformerConfig {
            transformer_type: TransformerType::BioBERT,
            ..Default::default()
        };
        let preprocessor = TransformerPreprocessor::new(config);

        let text = "p53 and BRCA1 mutations affect TNF-α via mRNA expression in CNS";
        let processed = preprocessor.preprocess_biomedical_text(text);
        assert!(processed.contains("tumor protein p53"));
        assert!(processed.contains("breast cancer gene 1"));
        assert!(processed.contains("tumor necrosis factor"));
        assert!(processed.contains("messenger ribonucleic acid"));
        assert!(processed.contains("central nervous system"));
    }

    #[test]
    fn test_code_preprocessing() {
        let config = TransformerConfig {
            transformer_type: TransformerType::CodeBERT,
            ..Default::default()
        };
        let preprocessor = TransformerPreprocessor::new(config);

        let text = "MyClass impl func calculateValue() returns Vec<i32>";
        let processed = preprocessor.preprocess_code_text(text);
        assert!(processed.contains("my class"));
        assert!(processed.contains("implementation"));
        assert!(processed.contains("function"));
        assert!(processed.contains("calculate value"));
    }

    #[test]
    fn test_camel_case_expansion() {
        let config = TransformerConfig::default();
        let preprocessor = TransformerPreprocessor::new(config);

        assert_eq!(preprocessor.expand_camel_case("MyClass"), "my class");
        assert_eq!(
            preprocessor.expand_camel_case("calculateValue"),
            "calculate value"
        );
        assert_eq!(
            preprocessor.expand_camel_case("getUserNameFromAPI"),
            "get user name from a p i"
        );
        assert_eq!(preprocessor.expand_camel_case(""), "");
    }

    #[test]
    fn test_uri_cleaning() {
        let config = TransformerConfig::default();
        let preprocessor = TransformerPreprocessor::new(config);

        let uri = "http://example.org/DNA_molecule#structure";
        let cleaned = preprocessor.clean_uri(uri);
        assert!(cleaned.contains("example"));
        assert!(cleaned.contains("DNA"));
        assert!(cleaned.contains("molecule"));
        assert!(cleaned.contains("structure"));
        assert!(!cleaned.contains("http://"));
        assert!(!cleaned.contains("#"));
    }
}
