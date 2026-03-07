//! SAMM characteristic constraint validators.
//!
//! Validates values against SAMM characteristic constraints including
//! RangeConstraint, LengthConstraint, RegularExpressionConstraint, and
//! EncodingConstraint as defined in the SAMM 2.3.0 specification.

use std::collections::HashSet;

// ── Encoding type ─────────────────────────────────────────────────────────────

/// Supported string encoding types for the `EncodingConstraint`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodingType {
    /// UTF-8 encoding (all valid UTF-8 strings are accepted).
    Utf8,
    /// UTF-16 encoding (all code points in BMP and supplementary planes).
    Utf16,
    /// US-ASCII (only bytes 0x00–0x7F, i.e. code points ≤ 127).
    Ascii,
    /// Base64-encoded binary data (RFC 4648 alphabet, optional padding).
    Base64,
    /// Hexadecimal-encoded binary data (characters `0-9`, `A-F`, `a-f`).
    Hex,
}

// ── SammConstraint ────────────────────────────────────────────────────────────

/// Supported SAMM constraint types that can be applied to characteristic values.
#[derive(Debug, Clone, PartialEq)]
pub enum SammConstraint {
    /// Validates that a numeric value is within `[min, max]` (or open bounds).
    Range {
        /// Lower bound of the range.
        min: f64,
        /// Upper bound of the range.
        max: f64,
        /// Whether `min` is included (`true`) or excluded (`false`).
        min_inclusive: bool,
        /// Whether `max` is included (`true`) or excluded (`false`).
        max_inclusive: bool,
    },
    /// Validates string length is within `[min_length, max_length]` (Unicode scalar values).
    Length {
        /// Minimum allowed length (inclusive).
        min_length: usize,
        /// Maximum allowed length (inclusive).
        max_length: usize,
    },
    /// Validates that a string matches the given regular-expression pattern.
    RegularExpression {
        /// The regex pattern (XSD-compatible subset).
        pattern: String,
    },
    /// Validates that a string value conforms to the specified character encoding.
    Encoding {
        /// The required encoding type.
        encoding: EncodingType,
    },
}

// ── ConstraintResult ──────────────────────────────────────────────────────────

/// The outcome of validating a single value against a single constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintResult {
    /// The value satisfies the constraint.
    Valid,
    /// The value violates the constraint; `reason` explains why.
    Invalid {
        /// Human-readable explanation of the violation.
        reason: String,
    },
}

impl ConstraintResult {
    /// Returns `true` iff the result is [`ConstraintResult::Valid`].
    pub fn is_valid(&self) -> bool {
        matches!(self, ConstraintResult::Valid)
    }

    /// Returns the violation reason, or `None` if the result is `Valid`.
    pub fn reason(&self) -> Option<&str> {
        match self {
            ConstraintResult::Valid => None,
            ConstraintResult::Invalid { reason } => Some(reason.as_str()),
        }
    }
}

// ── Regex helper (minimalist, no external dep) ────────────────────────────────

/// A very lightweight regex matcher supporting the subset of XSD/SAMM patterns
/// that appear in most SAMM models:
///
/// * Anchored start/end via `^` / `$`
/// * Character classes `[…]`, `[^…]`
/// * Quantifiers `*`, `+`, `?`, `{n}`, `{n,}`, `{n,m}`
/// * Wildcard `.`
/// * Literal character matching
///
/// Full PCRE features are intentionally out of scope to stay dependency-free.
struct SimpleRegex {
    pattern: String,
}

impl SimpleRegex {
    fn compile(pattern: &str) -> Self {
        SimpleRegex {
            pattern: pattern.to_owned(),
        }
    }

    fn is_match(&self, input: &str) -> bool {
        let pat = self.pattern.as_str();
        // Handle anchored patterns for correctness
        let (full_match, core) = if let Some(stripped) = pat.strip_prefix('^') {
            if let Some(core_pat) = stripped.strip_suffix('$') {
                (true, core_pat)
            } else {
                // anchored at start: try to match from position 0 only
                return Self::match_prefix(input, stripped);
            }
        } else if let Some(core_pat) = pat.strip_suffix('$') {
            // anchored at end: try all start positions
            let chars: Vec<char> = input.chars().collect();
            for start in 0..=chars.len() {
                let slice: String = chars[start..].iter().collect();
                if Self::match_prefix_full(&slice, core_pat) {
                    return true;
                }
            }
            return false;
        } else {
            (false, pat)
        };

        if full_match {
            Self::match_prefix_full(input, core)
        } else {
            // Unanchored: try every start position
            let chars: Vec<char> = input.chars().collect();
            for start in 0..=chars.len() {
                let slice: String = chars[start..].iter().collect();
                if Self::match_prefix(&slice, core) {
                    return true;
                }
            }
            false
        }
    }

    /// Attempt to match `pattern` at the start of `input` (possibly consuming less).
    fn match_prefix(input: &str, pattern: &str) -> bool {
        Self::dp_match(
            &input.chars().collect::<Vec<_>>(),
            &pattern.chars().collect::<Vec<_>>(),
            0,
            0,
        )
        .is_some()
    }

    /// Attempt to match `pattern` consuming exactly all of `input`.
    fn match_prefix_full(input: &str, pattern: &str) -> bool {
        let ic: Vec<char> = input.chars().collect();
        let pc: Vec<char> = pattern.chars().collect();
        matches!(Self::dp_match(&ic, &pc, 0, 0), Some(pos) if pos == ic.len())
    }

    /// Simple recursive descent matcher.
    /// Returns `Some(consumed_chars)` on the first successful match, or `None`.
    fn dp_match(input: &[char], pattern: &[char], ip: usize, pp: usize) -> Option<usize> {
        if pp >= pattern.len() {
            return Some(ip);
        }

        // Peek at next pattern token and optional quantifier
        let (token_len, matches_char): (usize, Box<dyn Fn(char) -> bool>) =
            Self::next_token(pattern, pp);

        let after_token = pp + token_len;

        // Check for quantifier
        let quantifier = if after_token < pattern.len() {
            pattern[after_token]
        } else {
            '\0'
        };

        match quantifier {
            '*' => {
                // Zero or more: try greedy then fall back
                let mut cur = ip;
                let mut positions = vec![cur];
                while cur < input.len() && matches_char(input[cur]) {
                    cur += 1;
                    positions.push(cur);
                }
                for &pos in positions.iter().rev() {
                    if let Some(r) = Self::dp_match(input, pattern, pos, after_token + 1) {
                        return Some(r);
                    }
                }
                None
            }
            '+' => {
                // One or more
                if ip >= input.len() || !matches_char(input[ip]) {
                    return None;
                }
                let mut cur = ip + 1;
                let mut positions = vec![cur];
                while cur < input.len() && matches_char(input[cur]) {
                    cur += 1;
                    positions.push(cur);
                }
                for &pos in positions.iter().rev() {
                    if let Some(r) = Self::dp_match(input, pattern, pos, after_token + 1) {
                        return Some(r);
                    }
                }
                None
            }
            '?' => {
                // Zero or one
                if ip < input.len() && matches_char(input[ip]) {
                    if let Some(r) = Self::dp_match(input, pattern, ip + 1, after_token + 1) {
                        return Some(r);
                    }
                }
                Self::dp_match(input, pattern, ip, after_token + 1)
            }
            '{' => {
                // {n}, {n,}, {n,m}
                if let Some((min, max_opt, quant_end)) = Self::parse_braces(pattern, after_token) {
                    let max = max_opt.unwrap_or(usize::MAX);
                    // Greedily consume up to `max` matching chars
                    let mut cur = ip;
                    let mut positions = Vec::new();
                    let mut count = 0usize;
                    while count <= max && cur < input.len() && matches_char(input[cur]) {
                        cur += 1;
                        count += 1;
                        if count >= min {
                            positions.push(cur);
                        }
                    }
                    if count >= min && count <= max {
                        positions.push(cur); // edge: already tracked
                    }
                    // Remove duplicates while preserving order (greediest first)
                    let unique: Vec<usize> = {
                        let mut seen = HashSet::new();
                        let mut v: Vec<usize> = positions
                            .iter()
                            .rev()
                            .copied()
                            .filter(|&x| seen.insert(x))
                            .collect();
                        v.reverse();
                        v.into_iter().rev().collect()
                    };
                    for &pos in &unique {
                        let consumed = pos - ip;
                        if consumed >= min && consumed <= max {
                            if let Some(r) = Self::dp_match(input, pattern, pos, quant_end) {
                                return Some(r);
                            }
                        }
                    }
                    None
                } else {
                    // Not a valid brace quantifier, treat as literal
                    Self::match_single(input, pattern, ip, pp, token_len, &*matches_char)
                }
            }
            _ => Self::match_single(input, pattern, ip, pp, token_len, &*matches_char),
        }
    }

    fn match_single(
        input: &[char],
        pattern: &[char],
        ip: usize,
        pp: usize,
        token_len: usize,
        matches_char: &dyn Fn(char) -> bool,
    ) -> Option<usize> {
        if ip < input.len() && matches_char(input[ip]) {
            Self::dp_match(input, pattern, ip + 1, pp + token_len)
        } else {
            None
        }
    }

    /// Parse `{n}`, `{n,}`, `{n,m}` starting at `pos` in `pattern`.
    /// Returns `(min, max_opt, end_pos_in_pattern)`.
    fn parse_braces(pattern: &[char], pos: usize) -> Option<(usize, Option<usize>, usize)> {
        if pos >= pattern.len() || pattern[pos] != '{' {
            return None;
        }
        let mut i = pos + 1;
        let mut min_s = String::new();
        while i < pattern.len() && pattern[i].is_ascii_digit() {
            min_s.push(pattern[i]);
            i += 1;
        }
        let min: usize = min_s.parse().ok()?;
        if i >= pattern.len() {
            return None;
        }
        if pattern[i] == '}' {
            return Some((min, Some(min), i + 1));
        }
        if pattern[i] != ',' {
            return None;
        }
        i += 1; // skip ','
        let mut max_s = String::new();
        while i < pattern.len() && pattern[i].is_ascii_digit() {
            max_s.push(pattern[i]);
            i += 1;
        }
        if i >= pattern.len() || pattern[i] != '}' {
            return None;
        }
        let max_opt = if max_s.is_empty() {
            None
        } else {
            Some(max_s.parse().ok()?)
        };
        Some((min, max_opt, i + 1))
    }

    /// Parse the next regex token at `pp` in `pattern`.
    /// Returns `(token_char_length, matcher_closure)`.
    fn next_token(pattern: &[char], pp: usize) -> (usize, Box<dyn Fn(char) -> bool>) {
        if pp >= pattern.len() {
            return (0, Box::new(|_| false));
        }
        match pattern[pp] {
            '.' => (1, Box::new(|c: char| c != '\n')),
            '[' => {
                // Find the closing ']'
                let mut i = pp + 1;
                let negated = i < pattern.len() && pattern[i] == '^';
                if negated {
                    i += 1;
                }
                // Allow ']' as first char inside class
                if i < pattern.len() && pattern[i] == ']' {
                    i += 1;
                }
                while i < pattern.len() && pattern[i] != ']' {
                    i += 1;
                }
                let class_end = i; // index of ']'
                let token_len = class_end - pp + 1; // includes '[' and ']'
                let members: Vec<char> = pattern[pp + 1 + usize::from(negated)..class_end].to_vec();
                // Expand ranges like a-z
                let expanded: HashSet<char> = {
                    let mut set = HashSet::new();
                    let mut idx = 0usize;
                    while idx < members.len() {
                        if idx + 2 < members.len() && members[idx + 1] == '-' {
                            let start = members[idx] as u32;
                            let end = members[idx + 2] as u32;
                            for cp in start..=end {
                                if let Some(c) = char::from_u32(cp) {
                                    set.insert(c);
                                }
                            }
                            idx += 3;
                        } else {
                            set.insert(members[idx]);
                            idx += 1;
                        }
                    }
                    set
                };
                if negated {
                    (token_len, Box::new(move |c: char| !expanded.contains(&c)))
                } else {
                    (token_len, Box::new(move |c: char| expanded.contains(&c)))
                }
            }
            '\\' if pp + 1 < pattern.len() => {
                let escaped = pattern[pp + 1];
                let matcher: Box<dyn Fn(char) -> bool> = match escaped {
                    'd' => Box::new(|c: char| c.is_ascii_digit()),
                    'D' => Box::new(|c: char| !c.is_ascii_digit()),
                    'w' => Box::new(|c: char| c.is_alphanumeric() || c == '_'),
                    'W' => Box::new(|c: char| !(c.is_alphanumeric() || c == '_')),
                    's' => Box::new(|c: char| c.is_whitespace()),
                    'S' => Box::new(|c: char| !c.is_whitespace()),
                    other => {
                        let o = other;
                        Box::new(move |c: char| c == o)
                    }
                };
                (2, matcher)
            }
            lit => {
                let l = lit;
                (1, Box::new(move |c: char| c == l))
            }
        }
    }
}

// ── Encoding helpers ──────────────────────────────────────────────────────────

fn is_valid_ascii(s: &str) -> bool {
    s.bytes().all(|b| b <= 0x7F)
}

fn is_valid_base64(s: &str) -> bool {
    let trimmed = s.trim_end_matches('=');
    trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '-' || c == '_')
}

fn is_valid_hex(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c: char| c.is_ascii_hexdigit())
}

// ── ConstraintValidator ───────────────────────────────────────────────────────

/// Stateless validator for SAMM characteristic constraints.
///
/// All methods are either inherent methods (no `&self` data required) or take
/// `&self` for a consistent API surface. The struct carries no runtime state.
pub struct ConstraintValidator;

impl Default for ConstraintValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintValidator {
    /// Create a new `ConstraintValidator`.
    pub fn new() -> Self {
        ConstraintValidator
    }

    // ── Single-value methods ────────────────────────────────────────────────

    /// Validate a numeric `value` against a single `constraint`.
    ///
    /// Only `Range` constraints are meaningful for numeric values; other
    /// constraint types return `ConstraintResult::Invalid` with an explanation.
    pub fn validate_number(&self, value: f64, constraint: &SammConstraint) -> ConstraintResult {
        match constraint {
            SammConstraint::Range {
                min,
                max,
                min_inclusive,
                max_inclusive,
            } => {
                let lower_ok = if *min_inclusive {
                    value >= *min
                } else {
                    value > *min
                };
                let upper_ok = if *max_inclusive {
                    value <= *max
                } else {
                    value < *max
                };
                if lower_ok && upper_ok {
                    ConstraintResult::Valid
                } else {
                    let lo_sym = if *min_inclusive { "[" } else { "(" };
                    let hi_sym = if *max_inclusive { "]" } else { ")" };
                    ConstraintResult::Invalid {
                        reason: format!(
                            "Value {value} is outside range {lo_sym}{min}, {max}{hi_sym}"
                        ),
                    }
                }
            }
            SammConstraint::Length { .. } => ConstraintResult::Invalid {
                reason: "LengthConstraint is not applicable to numeric values".to_owned(),
            },
            SammConstraint::RegularExpression { .. } => ConstraintResult::Invalid {
                reason: "RegularExpressionConstraint is not applicable to numeric values"
                    .to_owned(),
            },
            SammConstraint::Encoding { .. } => ConstraintResult::Invalid {
                reason: "EncodingConstraint is not applicable to numeric values".to_owned(),
            },
        }
    }

    /// Validate a string `value` against a single `constraint`.
    pub fn validate_string(&self, value: &str, constraint: &SammConstraint) -> ConstraintResult {
        match constraint {
            SammConstraint::Range {
                min,
                max,
                min_inclusive,
                max_inclusive,
            } => {
                // Attempt numeric parse; fall back to lexicographic ordering
                match value.trim().parse::<f64>() {
                    Ok(n) => self.validate_number(n, constraint),
                    Err(_) => {
                        // Lexicographic comparison using stringified bounds
                        let min_s = format!("{min}");
                        let max_s = format!("{max}");
                        let lower_ok = if *min_inclusive {
                            value >= min_s.as_str()
                        } else {
                            value > min_s.as_str()
                        };
                        let upper_ok = if *max_inclusive {
                            value <= max_s.as_str()
                        } else {
                            value < max_s.as_str()
                        };
                        if lower_ok && upper_ok {
                            ConstraintResult::Valid
                        } else {
                            ConstraintResult::Invalid {
                                reason: format!("String value '{value}' is outside range bounds"),
                            }
                        }
                    }
                }
            }
            SammConstraint::Length {
                min_length,
                max_length,
            } => {
                let len = value.chars().count();
                if len < *min_length {
                    ConstraintResult::Invalid {
                        reason: format!("String length {len} is less than minimum {min_length}"),
                    }
                } else if len > *max_length {
                    ConstraintResult::Invalid {
                        reason: format!("String length {len} exceeds maximum {max_length}"),
                    }
                } else {
                    ConstraintResult::Valid
                }
            }
            SammConstraint::RegularExpression { pattern } => {
                let re = SimpleRegex::compile(pattern);
                if re.is_match(value) {
                    ConstraintResult::Valid
                } else {
                    ConstraintResult::Invalid {
                        reason: format!("Value '{value}' does not match pattern '{pattern}'"),
                    }
                }
            }
            SammConstraint::Encoding { encoding } => {
                let ok = match encoding {
                    EncodingType::Utf8 => {
                        // All Rust &str values are valid UTF-8 by construction
                        true
                    }
                    EncodingType::Utf16 => {
                        // All Unicode scalar values are representable in UTF-16
                        value.chars().all(|_| true)
                    }
                    EncodingType::Ascii => is_valid_ascii(value),
                    EncodingType::Base64 => is_valid_base64(value),
                    EncodingType::Hex => is_valid_hex(value),
                };
                if ok {
                    ConstraintResult::Valid
                } else {
                    ConstraintResult::Invalid {
                        reason: format!("Value '{value}' is not valid {:?} encoding", encoding),
                    }
                }
            }
        }
    }

    // ── Multi-constraint methods ────────────────────────────────────────────

    /// Validate a string `value` against every constraint in `constraints`,
    /// returning one [`ConstraintResult`] per constraint in the same order.
    pub fn validate_all(
        &self,
        value: &str,
        constraints: &[SammConstraint],
    ) -> Vec<ConstraintResult> {
        constraints
            .iter()
            .map(|c| self.validate_string(value, c))
            .collect()
    }

    /// Returns `true` iff `value` satisfies **all** `constraints`.
    pub fn all_valid(&self, value: &str, constraints: &[SammConstraint]) -> bool {
        constraints
            .iter()
            .all(|c| self.validate_string(value, c).is_valid())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn validator() -> ConstraintValidator {
        ConstraintValidator::new()
    }

    // ── Range (numeric) ────────────────────────────────────────────────────

    #[test]
    fn range_inclusive_inside() {
        let c = SammConstraint::Range {
            min: 0.0,
            max: 100.0,
            min_inclusive: true,
            max_inclusive: true,
        };
        assert_eq!(
            validator().validate_number(50.0, &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn range_inclusive_at_min() {
        let c = SammConstraint::Range {
            min: 0.0,
            max: 100.0,
            min_inclusive: true,
            max_inclusive: true,
        };
        assert_eq!(
            validator().validate_number(0.0, &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn range_inclusive_at_max() {
        let c = SammConstraint::Range {
            min: 0.0,
            max: 100.0,
            min_inclusive: true,
            max_inclusive: true,
        };
        assert_eq!(
            validator().validate_number(100.0, &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn range_exclusive_at_min_fails() {
        let c = SammConstraint::Range {
            min: 0.0,
            max: 100.0,
            min_inclusive: false,
            max_inclusive: true,
        };
        assert!(!validator().validate_number(0.0, &c).is_valid());
    }

    #[test]
    fn range_exclusive_at_max_fails() {
        let c = SammConstraint::Range {
            min: 0.0,
            max: 100.0,
            min_inclusive: true,
            max_inclusive: false,
        };
        assert!(!validator().validate_number(100.0, &c).is_valid());
    }

    #[test]
    fn range_exclusive_inside() {
        let c = SammConstraint::Range {
            min: 0.0,
            max: 100.0,
            min_inclusive: false,
            max_inclusive: false,
        };
        assert_eq!(
            validator().validate_number(1.0, &c),
            ConstraintResult::Valid
        );
        assert_eq!(
            validator().validate_number(99.0, &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn range_below_min_fails() {
        let c = SammConstraint::Range {
            min: 10.0,
            max: 20.0,
            min_inclusive: true,
            max_inclusive: true,
        };
        assert!(!validator().validate_number(5.0, &c).is_valid());
    }

    #[test]
    fn range_above_max_fails() {
        let c = SammConstraint::Range {
            min: 10.0,
            max: 20.0,
            min_inclusive: true,
            max_inclusive: true,
        };
        assert!(!validator().validate_number(25.0, &c).is_valid());
    }

    #[test]
    fn range_negative_values() {
        let c = SammConstraint::Range {
            min: -50.0,
            max: -10.0,
            min_inclusive: true,
            max_inclusive: true,
        };
        assert_eq!(
            validator().validate_number(-30.0, &c),
            ConstraintResult::Valid
        );
        assert!(!validator().validate_number(-5.0, &c).is_valid());
    }

    #[test]
    fn range_float_precision() {
        let c = SammConstraint::Range {
            min: 0.1,
            max: 0.9,
            min_inclusive: true,
            max_inclusive: true,
        };
        assert_eq!(
            validator().validate_number(0.5, &c),
            ConstraintResult::Valid
        );
        assert!(!validator().validate_number(1.0, &c).is_valid());
    }

    // ── Range (string via validate_string) ────────────────────────────────

    #[test]
    fn range_string_numeric_valid() {
        let c = SammConstraint::Range {
            min: 1.0,
            max: 10.0,
            min_inclusive: true,
            max_inclusive: true,
        };
        assert_eq!(
            validator().validate_string("5", &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn range_string_numeric_invalid() {
        let c = SammConstraint::Range {
            min: 1.0,
            max: 10.0,
            min_inclusive: true,
            max_inclusive: true,
        };
        assert!(!validator().validate_string("0", &c).is_valid());
    }

    // ── Length ─────────────────────────────────────────────────────────────

    #[test]
    fn length_within_bounds() {
        let c = SammConstraint::Length {
            min_length: 3,
            max_length: 10,
        };
        assert_eq!(
            validator().validate_string("hello", &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn length_at_min_boundary() {
        let c = SammConstraint::Length {
            min_length: 3,
            max_length: 10,
        };
        assert_eq!(
            validator().validate_string("abc", &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn length_at_max_boundary() {
        let c = SammConstraint::Length {
            min_length: 3,
            max_length: 5,
        };
        assert_eq!(
            validator().validate_string("abcde", &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn length_below_min_fails() {
        let c = SammConstraint::Length {
            min_length: 5,
            max_length: 10,
        };
        assert!(!validator().validate_string("hi", &c).is_valid());
    }

    #[test]
    fn length_above_max_fails() {
        let c = SammConstraint::Length {
            min_length: 1,
            max_length: 3,
        };
        assert!(!validator().validate_string("toolong", &c).is_valid());
    }

    #[test]
    fn length_empty_string_at_zero_min() {
        let c = SammConstraint::Length {
            min_length: 0,
            max_length: 5,
        };
        assert_eq!(validator().validate_string("", &c), ConstraintResult::Valid);
    }

    #[test]
    fn length_empty_string_nonzero_min_fails() {
        let c = SammConstraint::Length {
            min_length: 1,
            max_length: 5,
        };
        assert!(!validator().validate_string("", &c).is_valid());
    }

    #[test]
    fn length_unicode_chars_counted_as_scalars() {
        // "café" has 4 Unicode scalar values
        let c = SammConstraint::Length {
            min_length: 4,
            max_length: 4,
        };
        assert_eq!(
            validator().validate_string("café", &c),
            ConstraintResult::Valid
        );
    }

    // ── RegularExpression ─────────────────────────────────────────────────

    #[test]
    fn regex_simple_match() {
        let c = SammConstraint::RegularExpression {
            pattern: "^[a-z]+$".to_owned(),
        };
        assert_eq!(
            validator().validate_string("hello", &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn regex_simple_no_match() {
        let c = SammConstraint::RegularExpression {
            pattern: "^[a-z]+$".to_owned(),
        };
        assert!(!validator().validate_string("Hello123", &c).is_valid());
    }

    #[test]
    fn regex_digit_pattern_match() {
        let c = SammConstraint::RegularExpression {
            pattern: "^\\d+$".to_owned(),
        };
        assert_eq!(
            validator().validate_string("12345", &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn regex_digit_pattern_no_match() {
        let c = SammConstraint::RegularExpression {
            pattern: "^\\d+$".to_owned(),
        };
        assert!(!validator().validate_string("123abc", &c).is_valid());
    }

    #[test]
    fn regex_email_like_pattern() {
        let c = SammConstraint::RegularExpression {
            pattern: "^[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}$".to_owned(),
        };
        assert_eq!(
            validator().validate_string("user@example.com", &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn regex_empty_string_against_star() {
        let c = SammConstraint::RegularExpression {
            pattern: "^.*$".to_owned(),
        };
        assert_eq!(validator().validate_string("", &c), ConstraintResult::Valid);
    }

    #[test]
    fn regex_fixed_literal_match() {
        let c = SammConstraint::RegularExpression {
            pattern: "^ACTIVE$".to_owned(),
        };
        assert_eq!(
            validator().validate_string("ACTIVE", &c),
            ConstraintResult::Valid
        );
        assert!(!validator().validate_string("active", &c).is_valid());
    }

    #[test]
    fn regex_optional_suffix() {
        let c = SammConstraint::RegularExpression {
            pattern: "^colou?r$".to_owned(),
        };
        assert_eq!(
            validator().validate_string("color", &c),
            ConstraintResult::Valid
        );
        assert_eq!(
            validator().validate_string("colour", &c),
            ConstraintResult::Valid
        );
    }

    // ── Encoding ──────────────────────────────────────────────────────────

    #[test]
    fn encoding_utf8_any_string_valid() {
        let c = SammConstraint::Encoding {
            encoding: EncodingType::Utf8,
        };
        assert_eq!(
            validator().validate_string("こんにちは", &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn encoding_ascii_pure_ascii_valid() {
        let c = SammConstraint::Encoding {
            encoding: EncodingType::Ascii,
        };
        assert_eq!(
            validator().validate_string("Hello, World!", &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn encoding_ascii_non_ascii_invalid() {
        let c = SammConstraint::Encoding {
            encoding: EncodingType::Ascii,
        };
        assert!(!validator().validate_string("café", &c).is_valid());
    }

    #[test]
    fn encoding_base64_valid() {
        let c = SammConstraint::Encoding {
            encoding: EncodingType::Base64,
        };
        // "Hello" in Base64
        assert_eq!(
            validator().validate_string("SGVsbG8=", &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn encoding_base64_invalid_chars() {
        let c = SammConstraint::Encoding {
            encoding: EncodingType::Base64,
        };
        assert!(!validator().validate_string("SGVsbG8!!", &c).is_valid());
    }

    #[test]
    fn encoding_base64_empty_valid() {
        let c = SammConstraint::Encoding {
            encoding: EncodingType::Base64,
        };
        // Empty string is valid (empty base64)
        assert_eq!(validator().validate_string("", &c), ConstraintResult::Valid);
    }

    #[test]
    fn encoding_hex_valid_upper() {
        let c = SammConstraint::Encoding {
            encoding: EncodingType::Hex,
        };
        assert_eq!(
            validator().validate_string("DEADBEEF", &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn encoding_hex_valid_lower() {
        let c = SammConstraint::Encoding {
            encoding: EncodingType::Hex,
        };
        assert_eq!(
            validator().validate_string("deadbeef", &c),
            ConstraintResult::Valid
        );
    }

    #[test]
    fn encoding_hex_invalid() {
        let c = SammConstraint::Encoding {
            encoding: EncodingType::Hex,
        };
        assert!(!validator().validate_string("ZZZZ", &c).is_valid());
    }

    #[test]
    fn encoding_hex_empty_invalid() {
        let c = SammConstraint::Encoding {
            encoding: EncodingType::Hex,
        };
        assert!(!validator().validate_string("", &c).is_valid());
    }

    // ── validate_all ──────────────────────────────────────────────────────

    #[test]
    fn validate_all_empty_constraints() {
        let results = validator().validate_all("any", &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn validate_all_all_pass() {
        let constraints = vec![
            SammConstraint::Length {
                min_length: 3,
                max_length: 10,
            },
            SammConstraint::RegularExpression {
                pattern: "^[a-z]+$".to_owned(),
            },
            SammConstraint::Encoding {
                encoding: EncodingType::Ascii,
            },
        ];
        let results = validator().validate_all("hello", &constraints);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_valid()));
    }

    #[test]
    fn validate_all_some_fail() {
        let constraints = vec![
            SammConstraint::Length {
                min_length: 3,
                max_length: 10,
            },
            SammConstraint::RegularExpression {
                pattern: "^\\d+$".to_owned(),
            },
        ];
        let results = validator().validate_all("hello", &constraints);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_valid()); // length ok
        assert!(!results[1].is_valid()); // not digits
    }

    #[test]
    fn validate_all_mixed_types() {
        let constraints = vec![
            SammConstraint::Length {
                min_length: 2,
                max_length: 6,
            },
            SammConstraint::Encoding {
                encoding: EncodingType::Hex,
            },
        ];
        let results = validator().validate_all("FF00", &constraints);
        assert!(results.iter().all(|r| r.is_valid()));
    }

    // ── all_valid ─────────────────────────────────────────────────────────

    #[test]
    fn all_valid_empty_constraints() {
        assert!(validator().all_valid("any", &[]));
    }

    #[test]
    fn all_valid_returns_true_when_all_pass() {
        let constraints = vec![
            SammConstraint::Length {
                min_length: 1,
                max_length: 20,
            },
            SammConstraint::Encoding {
                encoding: EncodingType::Ascii,
            },
        ];
        assert!(validator().all_valid("test123", &constraints));
    }

    #[test]
    fn all_valid_returns_false_when_any_fail() {
        let constraints = vec![
            SammConstraint::Length {
                min_length: 1,
                max_length: 5,
            },
            SammConstraint::Encoding {
                encoding: EncodingType::Ascii,
            },
        ];
        assert!(!validator().all_valid("this-is-too-long", &constraints));
    }

    #[test]
    fn all_valid_range_and_length() {
        let constraints = vec![
            SammConstraint::Range {
                min: 0.0,
                max: 100.0,
                min_inclusive: true,
                max_inclusive: true,
            },
            SammConstraint::Length {
                min_length: 1,
                max_length: 3,
            },
        ];
        assert!(validator().all_valid("42", &constraints));
    }

    // ── Inapplicable constraint type errors ───────────────────────────────

    #[test]
    fn length_constraint_on_number_is_invalid() {
        let c = SammConstraint::Length {
            min_length: 0,
            max_length: 100,
        };
        let result = validator().validate_number(5.0, &c);
        assert!(!result.is_valid());
        assert!(result.reason().is_some());
    }

    #[test]
    fn regex_constraint_on_number_is_invalid() {
        let c = SammConstraint::RegularExpression {
            pattern: ".*".to_owned(),
        };
        let result = validator().validate_number(5.0, &c);
        assert!(!result.is_valid());
    }

    #[test]
    fn encoding_constraint_on_number_is_invalid() {
        let c = SammConstraint::Encoding {
            encoding: EncodingType::Utf8,
        };
        let result = validator().validate_number(5.0, &c);
        assert!(!result.is_valid());
    }

    // ── ConstraintResult helpers ──────────────────────────────────────────

    #[test]
    fn constraint_result_is_valid_flag() {
        assert!(ConstraintResult::Valid.is_valid());
        assert!(!ConstraintResult::Invalid {
            reason: "bad".to_owned()
        }
        .is_valid());
    }

    #[test]
    fn constraint_result_reason() {
        assert!(ConstraintResult::Valid.reason().is_none());
        let r = ConstraintResult::Invalid {
            reason: "something wrong".to_owned(),
        };
        assert_eq!(r.reason(), Some("something wrong"));
    }

    // ── Default impl ──────────────────────────────────────────────────────

    #[test]
    fn constraint_validator_default() {
        let _v = ConstraintValidator;
    }
}
