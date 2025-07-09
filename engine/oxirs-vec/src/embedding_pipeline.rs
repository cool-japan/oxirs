use crate::{embeddings::EmbeddableContent, Vector, VectorData};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Text preprocessing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingPipeline {
    /// Tokenization settings
    pub tokenizer: TokenizerConfig,
    /// Normalization settings
    pub normalization: NormalizationConfig,
    /// Stop words to remove
    pub stop_words: HashSet<String>,
    /// Entity recognition settings
    pub entity_recognition: Option<EntityRecognitionConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub lowercase: bool,
    pub remove_punctuation: bool,
    pub min_token_length: usize,
    pub max_token_length: usize,
    pub split_camel_case: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: true,
            min_token_length: 2,
            max_token_length: 50,
            split_camel_case: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    pub unicode_normalization: bool,
    pub accent_removal: bool,
    pub stemming: bool,
    pub lemmatization: bool,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            unicode_normalization: true,
            accent_removal: true,
            stemming: false,
            lemmatization: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRecognitionConfig {
    pub recognize_uris: bool,
    pub recognize_dates: bool,
    pub recognize_numbers: bool,
    pub entity_linking: bool,
}

impl Default for PreprocessingPipeline {
    fn default() -> Self {
        // Common English stop words
        let mut stop_words = HashSet::new();
        for word in &[
            "the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "with", "to",
            "for", "of", "as", "by", "that", "this", "it", "from", "be", "are", "was", "were",
            "been",
        ] {
            stop_words.insert(word.to_string());
        }

        Self {
            tokenizer: TokenizerConfig::default(),
            normalization: NormalizationConfig::default(),
            stop_words,
            entity_recognition: None,
        }
    }
}

impl PreprocessingPipeline {
    /// Process text through the preprocessing pipeline
    pub fn process(&self, text: &str) -> Vec<String> {
        let mut tokens = self.tokenize(text);

        if self.normalization.unicode_normalization {
            tokens = self.normalize_unicode(tokens);
        }

        if self.normalization.accent_removal {
            tokens = self.remove_accents(tokens);
        }

        tokens = self.filter_tokens(tokens);

        if self.normalization.stemming {
            tokens = self.stem_tokens(tokens);
        }

        tokens
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut processed = text.to_string();

        if self.tokenizer.remove_punctuation {
            processed = processed
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c.is_whitespace() {
                        c
                    } else {
                        ' '
                    }
                })
                .collect();
        }

        // Split on whitespace and filter
        let mut tokens: Vec<String> = processed
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        // Split camelCase if enabled (must happen before lowercasing)
        if self.tokenizer.split_camel_case {
            tokens = tokens
                .into_iter()
                .flat_map(|token| self.split_camel_case(&token))
                .collect();
        }

        // Lowercase after camel case splitting
        if self.tokenizer.lowercase {
            tokens = tokens.into_iter().map(|s| s.to_lowercase()).collect();
        }

        tokens
    }

    fn split_camel_case(&self, word: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();

        for (i, ch) in word.chars().enumerate() {
            if i > 0 && ch.is_uppercase() && !current.is_empty() {
                result.push(current.clone());
                current.clear();
            }
            current.push(ch);
        }

        if !current.is_empty() {
            result.push(current);
        }

        if result.is_empty() {
            vec![word.to_string()]
        } else {
            result
        }
    }

    fn normalize_unicode(&self, tokens: Vec<String>) -> Vec<String> {
        // Simple unicode normalization - in production, use unicode-normalization crate
        tokens
    }

    fn remove_accents(&self, tokens: Vec<String>) -> Vec<String> {
        tokens
            .into_iter()
            .map(|token| {
                token
                    .chars()
                    .map(|c| match c {
                        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => 'a',
                        'è' | 'é' | 'ê' | 'ë' => 'e',
                        'ì' | 'í' | 'î' | 'ï' => 'i',
                        'ò' | 'ó' | 'ô' | 'õ' | 'ö' => 'o',
                        'ù' | 'ú' | 'û' | 'ü' => 'u',
                        'ñ' => 'n',
                        'ç' => 'c',
                        _ => c,
                    })
                    .collect()
            })
            .collect()
    }

    fn filter_tokens(&self, tokens: Vec<String>) -> Vec<String> {
        tokens
            .into_iter()
            .filter(|token| {
                token.len() >= self.tokenizer.min_token_length
                    && token.len() <= self.tokenizer.max_token_length
                    && !self.stop_words.contains(token)
            })
            .collect()
    }

    fn stem_tokens(&self, tokens: Vec<String>) -> Vec<String> {
        // Production-ready Porter stemmer implementation
        tokens
            .into_iter()
            .map(|token| self.porter_stem(&token))
            .collect()
    }

    /// Porter stemmer algorithm implementation
    fn porter_stem(&self, word: &str) -> String {
        let word = word.to_lowercase();
        if word.len() <= 2 {
            return word;
        }

        let mut stem = word.clone();

        // Step 1a: plurals and past participles
        stem = self.stem_step_1a(stem);

        // Step 1b: past tense and gerunds
        stem = self.stem_step_1b(stem);

        // Step 2: derivational suffixes
        stem = self.stem_step_2(stem);

        // Step 3: more derivational suffixes
        stem = self.stem_step_3(stem);

        // Step 4: remove derivational suffixes
        stem = self.stem_step_4(stem);

        // Step 5: remove final e and double l
        stem = self.stem_step_5(stem);

        stem
    }

    fn stem_step_1a(&self, mut word: String) -> String {
        if word.ends_with("sses") {
            word.truncate(word.len() - 2); // sses -> ss
        } else if word.ends_with("ies") {
            word.truncate(word.len() - 2); // ies -> i
        } else if word.ends_with("ss") {
            // ss -> ss (no change)
        } else if word.ends_with("s") && word.len() > 1 {
            word.truncate(word.len() - 1); // s -> (empty)
        }
        word
    }

    fn stem_step_1b(&self, mut word: String) -> String {
        if word.ends_with("eed") {
            if self.measure(&word[..word.len() - 3]) > 0 {
                word.truncate(word.len() - 1); // eed -> ee
            }
        } else if word.ends_with("ed") && self.contains_vowel(&word[..word.len() - 2]) {
            word.truncate(word.len() - 2);
            word = self.post_process_1b(word);
        } else if word.ends_with("ing") && self.contains_vowel(&word[..word.len() - 3]) {
            word.truncate(word.len() - 3);
            word = self.post_process_1b(word);
        }
        word
    }

    fn stem_step_2(&self, mut word: String) -> String {
        let suffixes = [
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        ];

        for (suffix, replacement) in &suffixes {
            if word.ends_with(suffix) {
                let stem = &word[..word.len() - suffix.len()];
                if self.measure(stem) > 0 {
                    word = format!("{stem}{replacement}");
                }
                break;
            }
        }
        word
    }

    fn stem_step_3(&self, mut word: String) -> String {
        let suffixes = [
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ];

        for (suffix, replacement) in &suffixes {
            if word.ends_with(suffix) {
                let stem = &word[..word.len() - suffix.len()];
                if self.measure(stem) > 0 {
                    word = format!("{stem}{replacement}");
                }
                break;
            }
        }
        word
    }

    fn stem_step_4(&self, mut word: String) -> String {
        let suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion",
            "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        ];

        for suffix in &suffixes {
            if word.ends_with(suffix) {
                let stem = &word[..word.len() - suffix.len()];
                if self.measure(stem) > 1
                    && (*suffix != "ion" || (stem.ends_with("s") || stem.ends_with("t"))) {
                        word = stem.to_string();
                    }
                break;
            }
        }
        word
    }

    fn stem_step_5(&self, mut word: String) -> String {
        if word.ends_with("e") {
            let stem = &word[..word.len() - 1];
            let m = self.measure(stem);
            if m > 1 || (m == 1 && !self.cvc(stem)) {
                word.truncate(word.len() - 1);
            }
        }

        if word.ends_with("ll") && self.measure(&word) > 1 {
            word.truncate(word.len() - 1);
        }

        word
    }

    fn post_process_1b(&self, mut word: String) -> String {
        if word.ends_with("at") || word.ends_with("bl") || word.ends_with("iz") {
            word.push('e');
        } else if self.double_consonant(&word)
            && !word.ends_with("l")
            && !word.ends_with("s")
            && !word.ends_with("z")
        {
            word.truncate(word.len() - 1);
        } else if self.measure(&word) == 1 && self.cvc(&word) {
            word.push('e');
        }
        word
    }

    fn measure(&self, word: &str) -> usize {
        let chars: Vec<char> = word.chars().collect();
        let mut m = 0;
        let mut prev_was_vowel = false;

        for (i, &ch) in chars.iter().enumerate() {
            let is_vowel = self.is_vowel(ch, i, &chars);
            if !is_vowel && prev_was_vowel {
                m += 1;
            }
            prev_was_vowel = is_vowel;
        }
        m
    }

    fn contains_vowel(&self, word: &str) -> bool {
        let chars: Vec<char> = word.chars().collect();
        chars
            .iter()
            .enumerate()
            .any(|(i, &ch)| self.is_vowel(ch, i, &chars))
    }

    #[allow(clippy::only_used_in_recursion)]
    fn is_vowel(&self, ch: char, pos: usize, chars: &[char]) -> bool {
        match ch {
            'a' | 'e' | 'i' | 'o' | 'u' => true,
            'y' => pos > 0 && !self.is_vowel(chars[pos - 1], pos - 1, chars),
            _ => false,
        }
    }

    fn cvc(&self, word: &str) -> bool {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() < 3 {
            return false;
        }

        let len = chars.len();
        !self.is_vowel(chars[len - 3], len - 3, &chars)
            && self.is_vowel(chars[len - 2], len - 2, &chars)
            && !self.is_vowel(chars[len - 1], len - 1, &chars)
            && chars[len - 1] != 'w'
            && chars[len - 1] != 'x'
            && chars[len - 1] != 'y'
    }

    fn double_consonant(&self, word: &str) -> bool {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() < 2 {
            return false;
        }

        let len = chars.len();
        chars[len - 1] == chars[len - 2] && !self.is_vowel(chars[len - 1], len - 1, &chars)
    }
}

/// Vector postprocessing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessingPipeline {
    pub dimensionality_reduction: Option<DimensionalityReduction>,
    pub normalization: VectorNormalization,
    pub outlier_detection: Option<OutlierDetection>,
    pub quality_scoring: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DimensionalityReduction {
    PCA { target_dims: usize },
    RandomProjection { target_dims: usize },
    AutoEncoder { target_dims: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorNormalization {
    None,
    L2,
    L1,
    MinMax,
    ZScore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierDetection {
    pub method: OutlierMethod,
    pub threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierMethod {
    ZScore,
    IsolationForest,
    LocalOutlierFactor,
}

impl Default for PostprocessingPipeline {
    fn default() -> Self {
        Self {
            dimensionality_reduction: None,
            normalization: VectorNormalization::L2,
            outlier_detection: None,
            quality_scoring: true,
        }
    }
}

impl PostprocessingPipeline {
    /// Process vector through the postprocessing pipeline
    pub fn process(&self, vector: &mut Vector) -> Result<f32> {
        // Apply dimensionality reduction if configured
        if let Some(ref dr) = self.dimensionality_reduction {
            self.apply_dimensionality_reduction(vector, dr)?;
        }

        // Apply normalization
        self.apply_normalization(vector)?;

        // Calculate quality score
        let quality_score = if self.quality_scoring {
            self.calculate_quality_score(vector)
        } else {
            1.0
        };

        // Check for outliers
        if let Some(ref od) = self.outlier_detection {
            if self.is_outlier(vector, od) {
                return Ok(quality_score * 0.5); // Reduce quality score for outliers
            }
        }

        Ok(quality_score)
    }

    fn apply_dimensionality_reduction(
        &self,
        vector: &mut Vector,
        method: &DimensionalityReduction,
    ) -> Result<()> {
        match method {
            DimensionalityReduction::PCA { target_dims } => {
                // Simplified PCA - in production, use proper implementation
                let values = vector.as_f32();
                if values.len() <= *target_dims {
                    return Ok(());
                }

                // Take first target_dims dimensions (simplified)
                let reduced: Vec<f32> = values.into_iter().take(*target_dims).collect();
                vector.values = VectorData::F32(reduced);
                vector.dimensions = *target_dims;
            }
            DimensionalityReduction::RandomProjection { target_dims } => {
                // Random projection implementation
                let values = vector.as_f32();
                if values.len() <= *target_dims {
                    return Ok(());
                }

                // Generate random projection matrix (simplified)
                use rand::{Rng, SeedableRng};
                let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                let mut projected = vec![0.0; *target_dims];

                for projected_val in projected.iter_mut().take(*target_dims) {
                    for &val in values.iter() {
                        let random_weight: f32 = rng.gen_range(-1.0..1.0);
                        *projected_val += val * random_weight;
                    }
                    *projected_val /= (values.len() as f32).sqrt();
                }

                vector.values = VectorData::F32(projected);
                vector.dimensions = *target_dims;
            }
            DimensionalityReduction::AutoEncoder { .. } => {
                // AutoEncoder would require neural network - placeholder
            }
        }
        Ok(())
    }

    fn apply_normalization(&self, vector: &mut Vector) -> Result<()> {
        match self.normalization {
            VectorNormalization::None => Ok(()),
            VectorNormalization::L2 => {
                vector.normalize();
                Ok(())
            }
            VectorNormalization::L1 => {
                let values = vector.as_f32();
                let l1_norm: f32 = values.iter().map(|x| x.abs()).sum();
                if l1_norm > 0.0 {
                    let normalized: Vec<f32> = values.into_iter().map(|x| x / l1_norm).collect();
                    vector.values = VectorData::F32(normalized);
                }
                Ok(())
            }
            VectorNormalization::MinMax => {
                let values = vector.as_f32();
                let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let range = max - min;

                if range > 0.0 {
                    let normalized: Vec<f32> =
                        values.into_iter().map(|x| (x - min) / range).collect();
                    vector.values = VectorData::F32(normalized);
                }
                Ok(())
            }
            VectorNormalization::ZScore => {
                let values = vector.as_f32();
                let n = values.len() as f32;
                let mean: f32 = values.iter().sum::<f32>() / n;
                let variance: f32 = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
                let std_dev = variance.sqrt();

                if std_dev > 0.0 {
                    let normalized: Vec<f32> =
                        values.into_iter().map(|x| (x - mean) / std_dev).collect();
                    vector.values = VectorData::F32(normalized);
                }
                Ok(())
            }
        }
    }

    fn calculate_quality_score(&self, vector: &Vector) -> f32 {
        let values = vector.as_f32();

        // Quality based on several factors
        let mut score = 1.0;

        // Check for NaN or infinite values
        if values.iter().any(|x| !x.is_finite()) {
            return 0.0;
        }

        // Check sparsity (too many zeros might indicate poor quality)
        let zero_count = values.iter().filter(|&&x| x.abs() < f32::EPSILON).count();
        let sparsity = zero_count as f32 / values.len() as f32;
        if sparsity > 0.9 {
            score *= 0.5;
        }

        // Check variance (too low variance might indicate poor quality)
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;

        if variance < 0.01 {
            score *= 0.7;
        }

        // Check magnitude (vectors that are too small might be problematic)
        let magnitude = vector.magnitude();
        if magnitude < 0.1 {
            score *= 0.8;
        }

        score
    }

    fn is_outlier(&self, vector: &Vector, config: &OutlierDetection) -> bool {
        match config.method {
            OutlierMethod::ZScore => {
                let values = vector.as_f32();
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                let variance =
                    values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
                let std_dev = variance.sqrt();

                // Check if any dimension is beyond threshold standard deviations
                values
                    .iter()
                    .any(|&x| ((x - mean) / std_dev).abs() > config.threshold)
            }
            _ => false, // Other methods would require more complex implementations
        }
    }
}

/// Complete embedding pipeline combining preprocessing and postprocessing
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct EmbeddingPipeline {
    pub preprocessing: PreprocessingPipeline,
    pub postprocessing: PostprocessingPipeline,
}


impl EmbeddingPipeline {
    /// Process content through the complete pipeline
    pub fn process_content(&self, content: &EmbeddableContent) -> Result<(Vec<String>, f32)> {
        // Extract text from content
        let text = content.to_text();

        // Apply preprocessing
        let tokens = self.preprocessing.process(&text);

        // Return tokens and a placeholder quality score
        // In a real implementation, this would generate embeddings and apply postprocessing
        Ok((tokens, 1.0))
    }

    /// Process a vector through postprocessing
    pub fn process_vector(&self, vector: &mut Vector) -> Result<f32> {
        self.postprocessing.process(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessing_pipeline() {
        let pipeline = PreprocessingPipeline::default();

        let text = "The quick brown fox jumps over the lazy dog!";
        let tokens = pipeline.process(text);

        // Should remove stop words and punctuation
        assert!(!tokens.contains(&"the".to_string()));
        assert!(tokens.contains(&"quick".to_string()));
        assert!(tokens.contains(&"brown".to_string()));
    }

    #[test]
    fn test_camel_case_splitting() {
        let mut pipeline = PreprocessingPipeline::default();
        pipeline.tokenizer.split_camel_case = true;

        let text = "CamelCaseWord HTTPSConnection";
        let tokens = pipeline.process(text);

        assert!(tokens.contains(&"camel".to_string()));
        assert!(tokens.contains(&"case".to_string()));
        assert!(tokens.contains(&"word".to_string()));
    }

    #[test]
    fn test_postprocessing_normalization() {
        let pipeline = PostprocessingPipeline {
            normalization: VectorNormalization::L2,
            ..Default::default()
        };

        let mut vector = Vector::new(vec![3.0, 4.0, 0.0]);
        let quality = pipeline.process(&mut vector).unwrap();

        // Check L2 normalization
        let magnitude = vector.magnitude();
        assert!((magnitude - 1.0).abs() < 1e-6);
        assert!(quality > 0.0);
    }

    #[test]
    fn test_quality_scoring() {
        let pipeline = PostprocessingPipeline::default();

        // Good quality vector
        let mut good_vector = Vector::new(vec![0.5, 0.3, -0.2, 0.8]);
        let good_quality = pipeline.process(&mut good_vector).unwrap();
        assert!(good_quality > 0.9);

        // Poor quality vector (all zeros)
        let poor_vector = Vector::new(vec![0.0, 0.0, 0.0, 0.0]);
        let poor_quality = pipeline.calculate_quality_score(&poor_vector);
        assert!(poor_quality < 0.5);
    }
}
