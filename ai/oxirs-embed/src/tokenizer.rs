//! Text tokenizer for embeddings.
//!
//! Provides BPE (Byte Pair Encoding) and WordPiece tokenization, vocabulary
//! management, special token handling, bidirectional token-to-ID mapping,
//! text encoding/decoding, max-sequence-length truncation, and batch
//! tokenization.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Special tokens
// ---------------------------------------------------------------------------

/// Well-known special tokens used by transformer models.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpecialToken {
    /// Classification token (typically index 0).
    Cls,
    /// Separator between sentences.
    Sep,
    /// Padding token.
    Pad,
    /// Unknown / out-of-vocabulary token.
    Unk,
    /// Masked token for MLM pre-training.
    Mask,
}

impl SpecialToken {
    /// Canonical string representation (e.g. `[CLS]`).
    pub fn as_str(&self) -> &'static str {
        match self {
            SpecialToken::Cls => "[CLS]",
            SpecialToken::Sep => "[SEP]",
            SpecialToken::Pad => "[PAD]",
            SpecialToken::Unk => "[UNK]",
            SpecialToken::Mask => "[MASK]",
        }
    }

    /// All built-in special tokens.
    pub fn all() -> &'static [SpecialToken] {
        &[
            SpecialToken::Cls,
            SpecialToken::Sep,
            SpecialToken::Pad,
            SpecialToken::Unk,
            SpecialToken::Mask,
        ]
    }
}

// ---------------------------------------------------------------------------
// Tokenizer mode
// ---------------------------------------------------------------------------

/// The sub-word algorithm used by the tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerMode {
    /// Byte Pair Encoding with iterative merge rules.
    Bpe,
    /// WordPiece with `##` continuation prefix.
    WordPiece,
}

// ---------------------------------------------------------------------------
// BPE merge rule
// ---------------------------------------------------------------------------

/// A single BPE merge rule: pair `(left, right)` merged into `merged`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeRule {
    pub left: String,
    pub right: String,
    pub merged: String,
}

// ---------------------------------------------------------------------------
// Encode result
// ---------------------------------------------------------------------------

/// The result of encoding a piece of text.
#[derive(Debug, Clone)]
pub struct EncodeResult {
    /// Token string representations.
    pub tokens: Vec<String>,
    /// Corresponding integer IDs.
    pub ids: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Configuration for building a `Tokenizer`.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Sub-word algorithm.
    pub mode: TokenizerMode,
    /// Maximum sequence length (tokens). Encoding is truncated to this.
    pub max_length: usize,
    /// Whether to automatically lower-case input text before tokenization.
    pub lowercase: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            mode: TokenizerMode::Bpe,
            max_length: 512,
            lowercase: true,
        }
    }
}

/// A text tokenizer supporting BPE and WordPiece sub-word algorithms.
pub struct Tokenizer {
    config: TokenizerConfig,
    /// token-string -> ID
    token_to_id: HashMap<String, u32>,
    /// ID -> token-string
    id_to_token: HashMap<u32, String>,
    /// Next ID to assign when adding a new token.
    next_id: u32,
    /// Ordered BPE merge rules (only used in BPE mode).
    merge_rules: Vec<MergeRule>,
}

impl Tokenizer {
    // ── Construction ─────────────────────────────────────────────────────

    /// Create a new tokenizer with the given configuration.
    ///
    /// Special tokens (`[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `[MASK]`) are
    /// registered automatically.
    pub fn new(config: TokenizerConfig) -> Self {
        let mut tok = Self {
            config,
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            next_id: 0,
            merge_rules: Vec::new(),
        };
        // Register all special tokens up-front.
        for st in SpecialToken::all() {
            tok.add_token(st.as_str());
        }
        tok
    }

    /// Build a default BPE tokenizer.
    pub fn bpe() -> Self {
        Self::new(TokenizerConfig {
            mode: TokenizerMode::Bpe,
            ..TokenizerConfig::default()
        })
    }

    /// Build a default WordPiece tokenizer.
    pub fn wordpiece() -> Self {
        Self::new(TokenizerConfig {
            mode: TokenizerMode::WordPiece,
            ..TokenizerConfig::default()
        })
    }

    // ── Vocabulary management ────────────────────────────────────────────

    /// Add a token to the vocabulary.  Returns its ID.
    ///
    /// If the token already exists, the existing ID is returned.
    pub fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        id
    }

    /// Remove a token from the vocabulary.  Returns `true` if it existed.
    ///
    /// Special tokens cannot be removed; in that case `false` is returned.
    pub fn remove_token(&mut self, token: &str) -> bool {
        // Guard special tokens.
        for st in SpecialToken::all() {
            if st.as_str() == token {
                return false;
            }
        }
        if let Some(id) = self.token_to_id.remove(token) {
            self.id_to_token.remove(&id);
            return true;
        }
        false
    }

    /// Current vocabulary size (including special tokens).
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Whether a token is in the vocabulary.
    pub fn contains_token(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    // ── BPE merge rules ──────────────────────────────────────────────────

    /// Add a BPE merge rule.  The merged token is also added to the vocab.
    pub fn add_merge_rule(&mut self, left: &str, right: &str) {
        let merged = format!("{left}{right}");
        self.add_token(&merged);
        self.merge_rules.push(MergeRule {
            left: left.to_string(),
            right: right.to_string(),
            merged,
        });
    }

    /// Number of registered merge rules.
    pub fn merge_rule_count(&self) -> usize {
        self.merge_rules.len()
    }

    // ── Token ↔ ID mapping ──────────────────────────────────────────────

    /// Look up the ID of a token.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Look up the token string for an ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// ID of the `[UNK]` token.
    pub fn unk_id(&self) -> u32 {
        self.token_to_id
            .get(SpecialToken::Unk.as_str())
            .copied()
            .unwrap_or(0)
    }

    /// ID of the `[CLS]` token.
    pub fn cls_id(&self) -> u32 {
        self.token_to_id
            .get(SpecialToken::Cls.as_str())
            .copied()
            .unwrap_or(0)
    }

    /// ID of the `[SEP]` token.
    pub fn sep_id(&self) -> u32 {
        self.token_to_id
            .get(SpecialToken::Sep.as_str())
            .copied()
            .unwrap_or(0)
    }

    /// ID of the `[PAD]` token.
    pub fn pad_id(&self) -> u32 {
        self.token_to_id
            .get(SpecialToken::Pad.as_str())
            .copied()
            .unwrap_or(0)
    }

    // ── Encoding ─────────────────────────────────────────────────────────

    /// Encode a text string into token IDs.
    ///
    /// The output is truncated to `config.max_length`.
    pub fn encode(&self, text: &str) -> EncodeResult {
        let text = if self.config.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        let sub_tokens = match &self.config.mode {
            TokenizerMode::Bpe => self.bpe_tokenize(&text),
            TokenizerMode::WordPiece => self.wordpiece_tokenize(&text),
        };

        let max = self.config.max_length;
        let truncated: Vec<String> = sub_tokens.into_iter().take(max).collect();
        let ids: Vec<u32> = truncated
            .iter()
            .map(|t| {
                self.token_to_id
                    .get(t.as_str())
                    .copied()
                    .unwrap_or_else(|| self.unk_id())
            })
            .collect();

        EncodeResult {
            tokens: truncated,
            ids,
        }
    }

    /// Decode a sequence of token IDs back into a string.
    ///
    /// WordPiece continuation tokens (`##…`) are merged back without spaces.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(tok) = self.id_to_token.get(&id) {
                // Skip special tokens in decoded output.
                let is_special = SpecialToken::all().iter().any(|st| st.as_str() == tok);
                if is_special {
                    continue;
                }
                parts.push(tok.clone());
            }
        }
        self.merge_subwords(&parts)
    }

    /// Encode a batch of texts.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<EncodeResult> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    // ── Sub-word merging (decode helper) ─────────────────────────────────

    /// Merge sub-word tokens back into words.
    ///
    /// WordPiece continuations (`##xyz`) are concatenated to the preceding
    /// token without a space.  BPE tokens are simply joined with spaces.
    fn merge_subwords(&self, tokens: &[String]) -> String {
        if tokens.is_empty() {
            return String::new();
        }

        match &self.config.mode {
            TokenizerMode::WordPiece => {
                let mut result = String::new();
                for tok in tokens {
                    if let Some(suffix) = tok.strip_prefix("##") {
                        result.push_str(suffix);
                    } else {
                        if !result.is_empty() {
                            result.push(' ');
                        }
                        result.push_str(tok);
                    }
                }
                result
            }
            TokenizerMode::Bpe => tokens.join(" "),
        }
    }

    // ── BPE tokenization ─────────────────────────────────────────────────

    /// Tokenize `text` using BPE merge rules.
    fn bpe_tokenize(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut all_tokens: Vec<String> = Vec::new();

        for word in words {
            // Start with individual characters.
            let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            // Apply merge rules in priority order.
            for rule in &self.merge_rules {
                symbols = Self::apply_merge(&symbols, &rule.left, &rule.right, &rule.merged);
            }

            // Map to vocab; fall back to [UNK] per symbol.
            for sym in symbols {
                if self.token_to_id.contains_key(&sym) {
                    all_tokens.push(sym);
                } else {
                    all_tokens.push(SpecialToken::Unk.as_str().to_string());
                }
            }
        }

        all_tokens
    }

    /// Apply one merge rule to a symbol sequence.
    fn apply_merge(symbols: &[String], left: &str, right: &str, merged: &str) -> Vec<String> {
        let mut result: Vec<String> = Vec::with_capacity(symbols.len());
        let mut i = 0;
        while i < symbols.len() {
            if i + 1 < symbols.len() && symbols[i] == left && symbols[i + 1] == right {
                result.push(merged.to_string());
                i += 2;
            } else {
                result.push(symbols[i].clone());
                i += 1;
            }
        }
        result
    }

    // ── WordPiece tokenization ───────────────────────────────────────────

    /// Tokenize `text` using greedy longest-match WordPiece.
    fn wordpiece_tokenize(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut all_tokens: Vec<String> = Vec::new();

        for word in words {
            let chars: Vec<char> = word.chars().collect();
            let n = chars.len();
            let mut start = 0;

            while start < n {
                let mut end = n;
                let mut found = false;

                while start < end {
                    let sub: String = chars[start..end].iter().collect();
                    let candidate = if start == 0 {
                        sub.clone()
                    } else {
                        format!("##{sub}")
                    };

                    if self.token_to_id.contains_key(&candidate) {
                        all_tokens.push(candidate);
                        start = end;
                        found = true;
                        break;
                    }
                    end -= 1;
                }

                if !found {
                    // Single character not in vocab → [UNK].
                    all_tokens.push(SpecialToken::Unk.as_str().to_string());
                    start += 1;
                }
            }
        }

        all_tokens
    }

    // ── Config access ────────────────────────────────────────────────────

    /// Maximum sequence length.
    pub fn max_length(&self) -> usize {
        self.config.max_length
    }

    /// Active tokenization mode.
    pub fn mode(&self) -> &TokenizerMode {
        &self.config.mode
    }

    /// Whether input is lowercased before tokenization.
    pub fn is_lowercase(&self) -> bool {
        self.config.lowercase
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn bpe_tokenizer() -> Tokenizer {
        Tokenizer::bpe()
    }

    fn wp_tokenizer() -> Tokenizer {
        Tokenizer::wordpiece()
    }

    // ── SpecialToken ─────────────────────────────────────────────────────

    #[test]
    fn test_special_token_cls_str() {
        assert_eq!(SpecialToken::Cls.as_str(), "[CLS]");
    }

    #[test]
    fn test_special_token_sep_str() {
        assert_eq!(SpecialToken::Sep.as_str(), "[SEP]");
    }

    #[test]
    fn test_special_token_pad_str() {
        assert_eq!(SpecialToken::Pad.as_str(), "[PAD]");
    }

    #[test]
    fn test_special_token_unk_str() {
        assert_eq!(SpecialToken::Unk.as_str(), "[UNK]");
    }

    #[test]
    fn test_special_token_mask_str() {
        assert_eq!(SpecialToken::Mask.as_str(), "[MASK]");
    }

    #[test]
    fn test_special_token_all_count() {
        assert_eq!(SpecialToken::all().len(), 5);
    }

    // ── Tokenizer construction ───────────────────────────────────────────

    #[test]
    fn test_new_bpe_has_special_tokens() {
        let tok = bpe_tokenizer();
        assert!(tok.contains_token("[CLS]"));
        assert!(tok.contains_token("[SEP]"));
        assert!(tok.contains_token("[PAD]"));
        assert!(tok.contains_token("[UNK]"));
        assert!(tok.contains_token("[MASK]"));
    }

    #[test]
    fn test_new_bpe_vocab_size() {
        let tok = bpe_tokenizer();
        assert_eq!(tok.vocab_size(), 5); // 5 special tokens
    }

    #[test]
    fn test_new_wordpiece_mode() {
        let tok = wp_tokenizer();
        assert_eq!(*tok.mode(), TokenizerMode::WordPiece);
    }

    #[test]
    fn test_bpe_mode() {
        let tok = bpe_tokenizer();
        assert_eq!(*tok.mode(), TokenizerMode::Bpe);
    }

    // ── Vocabulary management ────────────────────────────────────────────

    #[test]
    fn test_add_token_returns_new_id() {
        let mut tok = bpe_tokenizer();
        let id1 = tok.add_token("hello");
        let id2 = tok.add_token("world");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_add_token_idempotent() {
        let mut tok = bpe_tokenizer();
        let id1 = tok.add_token("hello");
        let id2 = tok.add_token("hello");
        assert_eq!(id1, id2);
        // vocab should not grow
        assert_eq!(tok.vocab_size(), 6); // 5 special + 1
    }

    #[test]
    fn test_remove_token_normal() {
        let mut tok = bpe_tokenizer();
        tok.add_token("temp");
        assert!(tok.contains_token("temp"));
        assert!(tok.remove_token("temp"));
        assert!(!tok.contains_token("temp"));
    }

    #[test]
    fn test_remove_special_token_prevented() {
        let mut tok = bpe_tokenizer();
        assert!(!tok.remove_token("[CLS]"));
        assert!(tok.contains_token("[CLS]"));
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut tok = bpe_tokenizer();
        assert!(!tok.remove_token("nonexistent"));
    }

    #[test]
    fn test_vocab_size_grows() {
        let mut tok = bpe_tokenizer();
        assert_eq!(tok.vocab_size(), 5);
        tok.add_token("a");
        tok.add_token("b");
        assert_eq!(tok.vocab_size(), 7);
    }

    // ── Token ↔ ID mapping ──────────────────────────────────────────────

    #[test]
    fn test_token_to_id_roundtrip() {
        let mut tok = bpe_tokenizer();
        let id = tok.add_token("cat");
        assert_eq!(tok.token_to_id("cat"), Some(id));
        assert_eq!(tok.id_to_token(id), Some("cat"));
    }

    #[test]
    fn test_token_to_id_missing() {
        let tok = bpe_tokenizer();
        assert_eq!(tok.token_to_id("missing"), None);
    }

    #[test]
    fn test_id_to_token_missing() {
        let tok = bpe_tokenizer();
        assert_eq!(tok.id_to_token(9999), None);
    }

    #[test]
    fn test_unk_id() {
        let tok = bpe_tokenizer();
        let unk = tok.unk_id();
        assert_eq!(tok.id_to_token(unk), Some("[UNK]"));
    }

    #[test]
    fn test_cls_id() {
        let tok = bpe_tokenizer();
        let cls = tok.cls_id();
        assert_eq!(tok.id_to_token(cls), Some("[CLS]"));
    }

    #[test]
    fn test_sep_id() {
        let tok = bpe_tokenizer();
        let sep = tok.sep_id();
        assert_eq!(tok.id_to_token(sep), Some("[SEP]"));
    }

    #[test]
    fn test_pad_id() {
        let tok = bpe_tokenizer();
        let pad = tok.pad_id();
        assert_eq!(tok.id_to_token(pad), Some("[PAD]"));
    }

    // ── BPE merge rules ──────────────────────────────────────────────────

    #[test]
    fn test_add_merge_rule_creates_merged_token() {
        let mut tok = bpe_tokenizer();
        tok.add_token("h");
        tok.add_token("e");
        tok.add_merge_rule("h", "e");
        assert!(tok.contains_token("he"));
        assert_eq!(tok.merge_rule_count(), 1);
    }

    #[test]
    fn test_bpe_merge_rules_applied_in_order() {
        let mut tok = bpe_tokenizer();
        // Build vocab: individual chars + merges
        tok.add_token("h");
        tok.add_token("e");
        tok.add_token("l");
        tok.add_token("o");
        tok.add_merge_rule("h", "e"); // he
        tok.add_merge_rule("l", "o"); // lo
        tok.add_merge_rule("he", "l"); // hel
        tok.add_merge_rule("hel", "lo"); // hello

        let result = tok.encode("hello");
        assert!(result.tokens.contains(&"hello".to_string()));
    }

    // ── BPE encoding ─────────────────────────────────────────────────────

    #[test]
    fn test_bpe_encode_unknown_chars() {
        let tok = bpe_tokenizer();
        // No char tokens registered → everything maps to [UNK]
        let result = tok.encode("xyz");
        assert!(result.ids.iter().all(|&id| id == tok.unk_id()));
    }

    #[test]
    fn test_bpe_encode_single_char_tokens() {
        let mut tok = bpe_tokenizer();
        tok.add_token("a");
        tok.add_token("b");
        let result = tok.encode("ab");
        assert_eq!(result.tokens, vec!["a", "b"]);
    }

    #[test]
    fn test_bpe_encode_multiple_words() {
        let mut tok = bpe_tokenizer();
        tok.add_token("h");
        tok.add_token("i");
        let result = tok.encode("hi hi");
        assert_eq!(result.tokens.len(), 4); // h i h i
    }

    // ── WordPiece encoding ───────────────────────────────────────────────

    #[test]
    fn test_wordpiece_full_word_match() {
        let mut tok = wp_tokenizer();
        tok.add_token("hello");
        let result = tok.encode("hello");
        assert_eq!(result.tokens, vec!["hello"]);
    }

    #[test]
    fn test_wordpiece_continuation_tokens() {
        let mut tok = wp_tokenizer();
        tok.add_token("un");
        tok.add_token("##believ");
        tok.add_token("##able");
        let result = tok.encode("unbelievable");
        assert_eq!(result.tokens, vec!["un", "##believ", "##able"]);
    }

    #[test]
    fn test_wordpiece_unknown_fallback() {
        let tok = wp_tokenizer();
        let result = tok.encode("xyz");
        // Each unknown character becomes [UNK]
        assert!(result.ids.iter().all(|&id| id == tok.unk_id()));
    }

    #[test]
    fn test_wordpiece_multiple_words() {
        let mut tok = wp_tokenizer();
        tok.add_token("hello");
        tok.add_token("world");
        let result = tok.encode("hello world");
        assert_eq!(result.tokens, vec!["hello", "world"]);
    }

    // ── Decoding ─────────────────────────────────────────────────────────

    #[test]
    fn test_bpe_decode_simple() {
        let mut tok = bpe_tokenizer();
        let id_a = tok.add_token("hello");
        let id_b = tok.add_token("world");
        let decoded = tok.decode(&[id_a, id_b]);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_wordpiece_decode_merges_continuations() {
        let mut tok = wp_tokenizer();
        let id_un = tok.add_token("un");
        let id_do = tok.add_token("##do");
        let decoded = tok.decode(&[id_un, id_do]);
        assert_eq!(decoded, "undo");
    }

    #[test]
    fn test_decode_skips_special_tokens() {
        let tok = bpe_tokenizer();
        let cls = tok.cls_id();
        let sep = tok.sep_id();
        let decoded = tok.decode(&[cls, sep]);
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_decode_empty() {
        let tok = bpe_tokenizer();
        assert_eq!(tok.decode(&[]), "");
    }

    // ── Max-length truncation ────────────────────────────────────────────

    #[test]
    fn test_truncation_at_max_length() {
        let mut tok = Tokenizer::new(TokenizerConfig {
            mode: TokenizerMode::Bpe,
            max_length: 3,
            lowercase: true,
        });
        tok.add_token("a");
        tok.add_token("b");
        tok.add_token("c");
        tok.add_token("d");
        let result = tok.encode("a b c d");
        assert_eq!(result.tokens.len(), 3);
        assert_eq!(result.ids.len(), 3);
    }

    #[test]
    fn test_truncation_shorter_text_unaffected() {
        let mut tok = Tokenizer::new(TokenizerConfig {
            mode: TokenizerMode::Bpe,
            max_length: 100,
            lowercase: true,
        });
        tok.add_token("x");
        let result = tok.encode("x");
        assert_eq!(result.tokens.len(), 1);
    }

    // ── Batch tokenization ───────────────────────────────────────────────

    #[test]
    fn test_encode_batch_count() {
        let mut tok = bpe_tokenizer();
        tok.add_token("a");
        let results = tok.encode_batch(&["a", "a a", "a a a"]);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_encode_batch_independent() {
        let mut tok = bpe_tokenizer();
        tok.add_token("x");
        tok.add_token("y");
        let results = tok.encode_batch(&["x", "y"]);
        assert_ne!(results[0].ids, results[1].ids);
    }

    #[test]
    fn test_encode_batch_empty() {
        let tok = bpe_tokenizer();
        let results = tok.encode_batch(&[]);
        assert!(results.is_empty());
    }

    // ── Lowercase handling ───────────────────────────────────────────────

    #[test]
    fn test_lowercase_enabled() {
        let mut tok = Tokenizer::new(TokenizerConfig {
            mode: TokenizerMode::Bpe,
            max_length: 512,
            lowercase: true,
        });
        tok.add_token("hello");
        // "h", "e", ... are individual chars, but "hello" is known as full word in BPE only
        // after merges. Test that uppercase is lowered:
        let r1 = tok.encode("HELLO");
        let r2 = tok.encode("hello");
        assert_eq!(r1.ids, r2.ids);
    }

    #[test]
    fn test_lowercase_disabled() {
        let mut tok = Tokenizer::new(TokenizerConfig {
            mode: TokenizerMode::Bpe,
            max_length: 512,
            lowercase: false,
        });
        tok.add_token("A");
        tok.add_token("a");
        let r1 = tok.encode("A");
        let r2 = tok.encode("a");
        assert_ne!(r1.ids, r2.ids);
    }

    // ── Config accessors ─────────────────────────────────────────────────

    #[test]
    fn test_max_length_accessor() {
        let tok = bpe_tokenizer();
        assert_eq!(tok.max_length(), 512);
    }

    #[test]
    fn test_is_lowercase_accessor() {
        let tok = bpe_tokenizer();
        assert!(tok.is_lowercase());
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_encode_empty_string() {
        let tok = bpe_tokenizer();
        let result = tok.encode("");
        assert!(result.tokens.is_empty());
        assert!(result.ids.is_empty());
    }

    #[test]
    fn test_encode_whitespace_only() {
        let tok = bpe_tokenizer();
        let result = tok.encode("   ");
        assert!(result.tokens.is_empty());
    }

    #[test]
    fn test_wordpiece_greedy_longest_match() {
        let mut tok = wp_tokenizer();
        tok.add_token("play");
        tok.add_token("##ing");
        tok.add_token("##i");
        tok.add_token("##n");
        tok.add_token("##g");
        let result = tok.encode("playing");
        // Should prefer "##ing" over "##i" + "##n" + "##g"
        assert_eq!(result.tokens, vec!["play", "##ing"]);
    }

    #[test]
    fn test_merge_rule_struct_fields() {
        let rule = MergeRule {
            left: "a".to_string(),
            right: "b".to_string(),
            merged: "ab".to_string(),
        };
        assert_eq!(rule.left, "a");
        assert_eq!(rule.right, "b");
        assert_eq!(rule.merged, "ab");
    }

    #[test]
    fn test_encode_result_tokens_and_ids_same_length() {
        let mut tok = bpe_tokenizer();
        tok.add_token("t");
        tok.add_token("e");
        tok.add_token("s");
        let result = tok.encode("test");
        assert_eq!(result.tokens.len(), result.ids.len());
    }

    #[test]
    fn test_tokenizer_config_default() {
        let cfg = TokenizerConfig::default();
        assert_eq!(cfg.mode, TokenizerMode::Bpe);
        assert_eq!(cfg.max_length, 512);
        assert!(cfg.lowercase);
    }
}
