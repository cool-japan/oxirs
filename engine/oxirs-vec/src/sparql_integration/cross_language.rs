//! Cross-language search capabilities for SPARQL vector integration

use std::collections::HashMap;

/// Cross-language search processor
pub struct CrossLanguageProcessor {
    language_weights: HashMap<String, f32>,
    supported_languages: Vec<String>,
}

impl CrossLanguageProcessor {
    pub fn new() -> Self {
        let mut supported_languages = vec![
            "en".to_string(),
            "es".to_string(),
            "fr".to_string(),
            "de".to_string(),
            "it".to_string(),
            "pt".to_string(),
            "ru".to_string(),
            "zh".to_string(),
            "ja".to_string(),
            "ar".to_string(),
        ];

        let mut language_weights = HashMap::new();
        for lang in &supported_languages {
            language_weights.insert(lang.clone(), 1.0);
        }

        Self {
            language_weights,
            supported_languages,
        }
    }

    /// Process a query with cross-language capabilities
    pub fn process_cross_language_query(
        &self,
        query: &str,
        target_languages: &[String],
    ) -> Vec<(String, f32)> {
        let mut processed_queries = Vec::new();
        
        // Original query gets highest weight
        processed_queries.push((query.to_string(), 1.0));

        // Detect source language
        let detected_lang = self.detect_language(query);
        
        // Generate variations for each target language
        for target_lang in target_languages {
            if target_lang == &detected_lang {
                continue; // Skip same language
            }

            let weight = self.language_weights.get(target_lang).copied().unwrap_or(0.8);

            // Generate translations
            let translations = self.generate_translations(query, target_lang);
            for translation in translations {
                processed_queries.push((translation, weight * 0.9));
            }

            // Generate transliterations
            let transliterations = self.generate_transliterations(query, target_lang);
            for transliteration in transliterations {
                processed_queries.push((transliteration, weight * 0.8));
            }

            // Generate stemmed variants
            let stemmed_variants = self.generate_stemmed_variants(query, target_lang);
            for variant in stemmed_variants {
                processed_queries.push((variant, weight * 0.7));
            }
        }

        processed_queries
    }

    /// Detect language using simple heuristics
    pub fn detect_language(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();

        // Simple language detection based on character patterns and common words
        if text_lower.contains("machine learning")
            || text_lower.contains("artificial intelligence")
            || text_lower.contains("deep learning")
        {
            return "en".to_string();
        }

        if text_lower.contains("aprendizaje")
            || text_lower.contains("inteligencia")
            || text_lower.contains("máquina")
        {
            return "es".to_string();
        }

        if text_lower.contains("apprentissage")
            || text_lower.contains("intelligence")
            || text_lower.contains("automatique")
        {
            return "fr".to_string();
        }

        if text_lower.contains("lernen") || text_lower.contains("künstlich") {
            return "de".to_string();
        }

        // Check for Cyrillic characters
        if text.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c)) {
            return "ru".to_string();
        }

        // Check for Chinese characters
        if text.chars().any(|c| {
            ('\u{4E00}'..='\u{9FFF}').contains(&c) || ('\u{3400}'..='\u{4DBF}').contains(&c)
        }) {
            return "zh".to_string();
        }

        // Check for Arabic characters
        if text.chars().any(|c| ('\u{0600}'..='\u{06FF}').contains(&c)) {
            return "ar".to_string();
        }

        // Default to English
        "en".to_string()
    }

    /// Generate basic translations using simple dictionaries
    fn generate_translations(&self, query: &str, target_lang: &str) -> Vec<String> {
        let mut translations = Vec::new();

        let basic_dict = match target_lang {
            "es" => vec![
                ("artificial intelligence", "inteligencia artificial"),
                ("machine learning", "aprendizaje automático"),
                ("data science", "ciencia de datos"),
                ("neural network", "red neuronal"),
                ("deep learning", "aprendizaje profundo"),
            ],
            "fr" => vec![
                ("artificial intelligence", "intelligence artificielle"),
                ("machine learning", "apprentissage automatique"),
                ("data science", "science des données"),
                ("neural network", "réseau de neurones"),
                ("deep learning", "apprentissage profond"),
            ],
            "de" => vec![
                ("artificial intelligence", "künstliche Intelligenz"),
                ("machine learning", "maschinelles Lernen"),
                ("data science", "Datenwissenschaft"),
                ("neural network", "neuronales Netzwerk"),
                ("deep learning", "tiefes Lernen"),
            ],
            _ => vec![],
        };

        let query_lower = query.to_lowercase();
        for (en_term, target_term) in basic_dict {
            if query_lower.contains(en_term) {
                let translated = query_lower.replace(en_term, target_term);
                translations.push(translated);
            }
        }

        translations
    }

    /// Generate transliteration variations for different scripts
    fn generate_transliterations(&self, query: &str, target_lang: &str) -> Vec<String> {
        let mut transliterations = Vec::new();

        // For languages with different scripts, generate transliterations
        match target_lang {
            "ru" => {
                // Cyrillic transliteration (simplified)
                let latin_to_cyrillic = vec![
                    ("ai", "ай"),
                    ("machine", "машин"),
                    ("data", "дата"),
                    ("network", "сеть"),
                    ("learning", "обучение"),
                ];

                let mut transliterated = query.to_lowercase();
                for (latin, cyrillic) in latin_to_cyrillic {
                    transliterated = transliterated.replace(latin, cyrillic);
                }
                if transliterated != query.to_lowercase() {
                    transliterations.push(transliterated);
                }
            }
            "ar" => {
                // Arabic transliteration (simplified)
                let latin_to_arabic =
                    vec![("data", "بيانات"), ("machine", "آلة"), ("network", "شبكة")];

                let mut transliterated = query.to_lowercase();
                for (latin, arabic) in latin_to_arabic {
                    transliterated = transliterated.replace(latin, arabic);
                }
                if transliterated != query.to_lowercase() {
                    transliterations.push(transliterated);
                }
            }
            _ => {
                // For Latin-script languages, no transliteration needed
            }
        }

        transliterations
    }

    /// Generate stemmed variants for better cross-language matching
    fn generate_stemmed_variants(&self, query: &str, target_lang: &str) -> Vec<String> {
        let mut variants = Vec::new();

        // Simple stemming rules by language
        let words: Vec<&str> = query.split_whitespace().collect();

        for word in words {
            let stemmed = match target_lang {
                "es" => {
                    // Spanish stemming rules (simplified)
                    let word_lower = word.to_lowercase();
                    if word_lower.ends_with("ción") {
                        word_lower.replace("ción", "")
                    } else if word_lower.ends_with("mente") {
                        word_lower.replace("mente", "")
                    } else {
                        word_lower
                    }
                }
                "fr" => {
                    // French stemming rules (simplified)
                    let word_lower = word.to_lowercase();
                    if word_lower.ends_with("ment") {
                        word_lower.replace("ment", "")
                    } else if word_lower.ends_with("ique") {
                        word_lower.replace("ique", "")
                    } else {
                        word_lower
                    }
                }
                "de" => {
                    // German stemming rules (simplified)
                    let word_lower = word.to_lowercase();
                    if word_lower.ends_with("ung") {
                        word_lower.replace("ung", "")
                    } else if word_lower.ends_with("lich") {
                        word_lower.replace("lich", "")
                    } else {
                        word_lower
                    }
                }
                "en" => {
                    // English stemming rules (simplified)
                    let word_lower = word.to_lowercase();
                    if word_lower.ends_with("ing") {
                        word_lower.replace("ing", "")
                    } else if word_lower.ends_with("ed") {
                        word_lower.replace("ed", "")
                    } else if word_lower.ends_with("ly") {
                        word_lower.replace("ly", "")
                    } else {
                        word_lower
                    }
                }
                _ => word.to_lowercase(),
            };

            if stemmed != word.to_lowercase() && !stemmed.is_empty() {
                variants.push(stemmed);
            }
        }

        // Create variant queries by combining stemmed words
        if !variants.is_empty() {
            let original_words: Vec<&str> = query.split_whitespace().collect();
            let mut variant_query = String::new();
            
            for (i, word) in original_words.iter().enumerate() {
                if i < variants.len() && !variants[i].is_empty() {
                    variant_query.push_str(&variants[i]);
                } else {
                    variant_query.push_str(word);
                }
                if i < original_words.len() - 1 {
                    variant_query.push(' ');
                }
            }
            
            if variant_query != query.to_lowercase() {
                vec![variant_query]
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    }

    /// Set weight for a specific language
    pub fn set_language_weight(&mut self, language: &str, weight: f32) {
        self.language_weights.insert(language.to_string(), weight);
    }

    /// Get supported languages
    pub fn supported_languages(&self) -> &[String] {
        &self.supported_languages
    }

    /// Check if a language is supported
    pub fn is_language_supported(&self, language: &str) -> bool {
        self.supported_languages.contains(&language.to_string())
    }
}

impl Default for CrossLanguageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detection() {
        let processor = CrossLanguageProcessor::new();

        assert_eq!(
            processor.detect_language("machine learning algorithm"),
            "en"
        );
        assert_eq!(
            processor.detect_language("aprendizaje automático"),
            "es"
        );
        assert_eq!(
            processor.detect_language("apprentissage automatique"),
            "fr"
        );
        assert_eq!(
            processor.detect_language("maschinelles Lernen"),
            "de"
        );
    }

    #[test]
    fn test_translation_generation() {
        let processor = CrossLanguageProcessor::new();

        let translations = processor.generate_translations("machine learning", "es");
        assert!(translations.contains(&"aprendizaje automático".to_string()));

        let translations = processor.generate_translations("artificial intelligence", "fr");
        assert!(translations.contains(&"intelligence artificielle".to_string()));
    }

    #[test]
    fn test_cross_language_processing() {
        let processor = CrossLanguageProcessor::new();

        let processed = processor.process_cross_language_query(
            "machine learning",
            &["es".to_string(), "fr".to_string()],
        );

        // Should include original query plus variations
        assert!(processed.len() > 1);
        assert_eq!(processed[0].0, "machine learning");
        assert_eq!(processed[0].1, 1.0); // Original gets highest weight
    }

    #[test]
    fn test_stemming() {
        let processor = CrossLanguageProcessor::new();

        let variants = processor.generate_stemmed_variants("learning", "en");
        assert!(variants.iter().any(|v| v.contains("learn")));

        let variants = processor.generate_stemmed_variants("automático", "es");
        // Should generate stemmed variant if rules apply
        assert!(variants.len() <= 1); // Simplified stemming
    }

    #[test]
    fn test_language_support() {
        let processor = CrossLanguageProcessor::new();

        assert!(processor.is_language_supported("en"));
        assert!(processor.is_language_supported("es"));
        assert!(processor.is_language_supported("fr"));
        assert!(!processor.is_language_supported("xyz"));
    }
}