//! SPARQL FILTER expression types and evaluation
//!
//! Supports:
//! - Comparison operators: =, !=, <, <=, >, >=
//! - Logical operators: &&, ||, !
//! - Built-in functions: LANG(), STR(), DATATYPE(), BOUND(), regex(), isIRI(), isLiteral(), isBlank()
//! - FILTER EXISTS / FILTER NOT EXISTS (evaluated via callback)
//! - String functions: STRSTARTS(), STRENDS(), CONTAINS(), STRLEN(), UCASE(), LCASE(), SUBSTR()

use std::collections::HashMap;

/// A SPARQL FILTER expression tree
#[derive(Debug, Clone)]
pub(crate) enum FilterExpr {
    // -----------------------------------------------------------------------
    // Comparison operators
    // -----------------------------------------------------------------------
    /// ?var = value (string equality or numeric)
    Equals {
        lhs: Box<FilterTerm>,
        rhs: Box<FilterTerm>,
    },
    /// ?var != value
    NotEquals {
        lhs: Box<FilterTerm>,
        rhs: Box<FilterTerm>,
    },
    /// ?var > value
    GreaterThan {
        lhs: Box<FilterTerm>,
        rhs: Box<FilterTerm>,
    },
    /// ?var >= value
    GreaterEq {
        lhs: Box<FilterTerm>,
        rhs: Box<FilterTerm>,
    },
    /// ?var < value
    LessThan {
        lhs: Box<FilterTerm>,
        rhs: Box<FilterTerm>,
    },
    /// ?var <= value
    LessEq {
        lhs: Box<FilterTerm>,
        rhs: Box<FilterTerm>,
    },

    // -----------------------------------------------------------------------
    // Logical operators
    // -----------------------------------------------------------------------
    /// expr && expr
    And(Box<FilterExpr>, Box<FilterExpr>),
    /// expr || expr
    Or(Box<FilterExpr>, Box<FilterExpr>),
    /// !expr
    Not(Box<FilterExpr>),

    // -----------------------------------------------------------------------
    // Built-in function calls
    // -----------------------------------------------------------------------
    /// LANG(?var) = "tag"
    LangEquals { var: String, lang: String },
    /// LANG(?var)  — as a term (used inside comparisons)
    LangCall { var: String },
    /// regex(?var, "pattern")
    Regex {
        var: String,
        pattern: String,
        flags: Option<String>,
    },
    /// STR(?var) — strips datatype/lang, returns plain string value
    Str { var: String },
    /// DATATYPE(?var) — returns the datatype IRI
    Datatype { var: String },
    /// BOUND(?var)
    Bound { var: String },
    /// isIRI(?var)
    IsIri { var: String },
    /// isLiteral(?var)
    IsLiteral { var: String },
    /// isBlank(?var)
    IsBlank { var: String },
    /// STRSTARTS(?var, "prefix")
    StrStarts { var: String, prefix: String },
    /// STRENDS(?var, "suffix")
    StrEnds { var: String, suffix: String },
    /// CONTAINS(?var, "substring")
    Contains { var: String, substring: String },
    /// STRLEN(?var)
    StrLen { var: String },
    /// UCASE(?var)
    Ucase { var: String },
    /// LCASE(?var)
    Lcase { var: String },
    /// SUBSTR(?var, start[, length])
    Substr {
        var: String,
        start: usize,
        length: Option<usize>,
    },

    // -----------------------------------------------------------------------
    // EXISTS / NOT EXISTS  (inner pattern evaluated by caller via closure)
    // -----------------------------------------------------------------------
    /// FILTER EXISTS { patterns } — true if inner bindings non-empty
    Exists { inner_sparql: String },
    /// FILTER NOT EXISTS { patterns } — true if inner bindings empty
    NotExists { inner_sparql: String },
}

/// A term inside a filter comparison — either a variable or a literal value
#[derive(Debug, Clone)]
pub(crate) enum FilterTerm {
    Variable(String),
    Literal(String),
    Number(f64),
    /// A function call result (e.g. LANG(?x))
    FuncCall(Box<FilterExpr>),
}

impl FilterExpr {
    /// Evaluate this filter against a single binding row.
    ///
    /// Returns `true` if the binding passes the filter.
    pub(crate) fn evaluate(&self, binding: &HashMap<String, String>) -> bool {
        match self {
            // ---------------------------------------------------------------
            // Comparisons
            // ---------------------------------------------------------------
            FilterExpr::Equals { lhs, rhs } => {
                match (resolve_term(lhs, binding), resolve_term(rhs, binding)) {
                    (Some(l), Some(r)) => terms_equal(&l, &r),
                    _ => false,
                }
            }
            FilterExpr::NotEquals { lhs, rhs } => {
                match (resolve_term(lhs, binding), resolve_term(rhs, binding)) {
                    (Some(l), Some(r)) => !terms_equal(&l, &r),
                    _ => false,
                }
            }
            FilterExpr::GreaterThan { lhs, rhs } => {
                compare_numeric(lhs, rhs, binding, |a, b| a > b)
            }
            FilterExpr::GreaterEq { lhs, rhs } => compare_numeric(lhs, rhs, binding, |a, b| a >= b),
            FilterExpr::LessThan { lhs, rhs } => compare_numeric(lhs, rhs, binding, |a, b| a < b),
            FilterExpr::LessEq { lhs, rhs } => compare_numeric(lhs, rhs, binding, |a, b| a <= b),

            // ---------------------------------------------------------------
            // Logical
            // ---------------------------------------------------------------
            FilterExpr::And(a, b) => a.evaluate(binding) && b.evaluate(binding),
            FilterExpr::Or(a, b) => a.evaluate(binding) || b.evaluate(binding),
            FilterExpr::Not(inner) => !inner.evaluate(binding),

            // ---------------------------------------------------------------
            // Built-ins
            // ---------------------------------------------------------------
            FilterExpr::LangEquals { var, lang } => {
                let value = match binding.get(var) {
                    Some(v) => v,
                    None => return false,
                };
                extract_lang(value)
                    .map(|l| l.eq_ignore_ascii_case(lang))
                    .unwrap_or(false)
            }

            FilterExpr::LangCall { .. } => {
                // When used standalone (not in comparison), always true
                true
            }

            FilterExpr::Regex {
                var,
                pattern,
                flags,
            } => {
                let value = match binding.get(var) {
                    Some(v) => v,
                    None => return false,
                };
                let literal_value = extract_literal_value(value);
                let case_insensitive = flags.as_deref().map(|f| f.contains('i')).unwrap_or(false);
                if case_insensitive {
                    simple_regex_match(pattern, &literal_value.to_lowercase())
                } else {
                    simple_regex_match(pattern, &literal_value)
                }
            }

            FilterExpr::Bound { var } => binding.contains_key(var.as_str()),

            FilterExpr::IsIri { var } => binding
                .get(var.as_str())
                .map(|v| is_iri_term(v))
                .unwrap_or(false),
            FilterExpr::IsLiteral { var } => binding
                .get(var.as_str())
                .map(|v| is_literal_term(v))
                .unwrap_or(false),
            FilterExpr::IsBlank { var } => binding
                .get(var.as_str())
                .map(|v| v.starts_with("_:"))
                .unwrap_or(false),

            FilterExpr::StrStarts { var, prefix } => binding
                .get(var.as_str())
                .map(|v| extract_literal_value(v).starts_with(prefix.as_str()))
                .unwrap_or(false),
            FilterExpr::StrEnds { var, suffix } => binding
                .get(var.as_str())
                .map(|v| extract_literal_value(v).ends_with(suffix.as_str()))
                .unwrap_or(false),
            FilterExpr::Contains { var, substring } => binding
                .get(var.as_str())
                .map(|v| extract_literal_value(v).contains(substring.as_str()))
                .unwrap_or(false),
            FilterExpr::StrLen { var } => {
                // Returns true if bound (length itself needs numeric comparison)
                binding.contains_key(var.as_str())
            }
            FilterExpr::Ucase { .. } | FilterExpr::Lcase { .. } => true,
            FilterExpr::Substr { var, .. } => binding.contains_key(var.as_str()),

            FilterExpr::Str { var } => binding.contains_key(var.as_str()),
            FilterExpr::Datatype { var } => binding
                .get(var.as_str())
                .map(|v| extract_datatype(v).is_some())
                .unwrap_or(false),

            // EXISTS and NOT EXISTS need the outer evaluator — default to false here
            // (the outer evaluator should handle these before calling evaluate)
            FilterExpr::Exists { .. } => false,
            FilterExpr::NotExists { .. } => true,
        }
    }

    /// Evaluate EXISTS / NOT EXISTS using a provided closure that evaluates inner patterns.
    /// The closure receives inner SPARQL text and returns whether any bindings were found.
    pub(crate) fn evaluate_with_exists<F>(
        &self,
        binding: &HashMap<String, String>,
        exists_fn: &F,
    ) -> bool
    where
        F: Fn(&str, &HashMap<String, String>) -> bool,
    {
        match self {
            FilterExpr::Exists { inner_sparql } => exists_fn(inner_sparql, binding),
            FilterExpr::NotExists { inner_sparql } => !exists_fn(inner_sparql, binding),
            FilterExpr::And(a, b) => {
                a.evaluate_with_exists(binding, exists_fn)
                    && b.evaluate_with_exists(binding, exists_fn)
            }
            FilterExpr::Or(a, b) => {
                a.evaluate_with_exists(binding, exists_fn)
                    || b.evaluate_with_exists(binding, exists_fn)
            }
            FilterExpr::Not(inner) => !inner.evaluate_with_exists(binding, exists_fn),
            _ => self.evaluate(binding),
        }
    }
}

// -----------------------------------------------------------------------
// Resolution helpers
// -----------------------------------------------------------------------

fn resolve_term(term: &FilterTerm, binding: &HashMap<String, String>) -> Option<String> {
    match term {
        FilterTerm::Variable(var) => binding.get(var.as_str()).cloned(),
        FilterTerm::Literal(s) => Some(s.clone()),
        FilterTerm::Number(n) => Some(n.to_string()),
        FilterTerm::FuncCall(expr) => {
            // Only LANG, STR, DATATYPE produce string results here
            match expr.as_ref() {
                FilterExpr::LangCall { var } => binding
                    .get(var.as_str())
                    .and_then(|v| extract_lang(v).map(|s| s.to_string())),
                FilterExpr::Str { var } => {
                    binding.get(var.as_str()).map(|v| extract_literal_value(v))
                }
                FilterExpr::Datatype { var } => {
                    binding.get(var.as_str()).and_then(|v| extract_datatype(v))
                }
                FilterExpr::StrLen { var } => binding
                    .get(var.as_str())
                    .map(|v| extract_literal_value(v).chars().count().to_string()),
                FilterExpr::Ucase { var } => binding
                    .get(var.as_str())
                    .map(|v| extract_literal_value(v).to_uppercase()),
                FilterExpr::Lcase { var } => binding
                    .get(var.as_str())
                    .map(|v| extract_literal_value(v).to_lowercase()),
                FilterExpr::Substr { var, start, length } => {
                    binding.get(var.as_str()).map(|v| {
                        let s = extract_literal_value(v);
                        let chars: Vec<char> = s.chars().collect();
                        // SPARQL SUBSTR is 1-based
                        let begin = start.saturating_sub(1);
                        let slice = if let Some(len) = length {
                            &chars[begin.min(chars.len())..(begin + len).min(chars.len())]
                        } else {
                            &chars[begin.min(chars.len())..]
                        };
                        slice.iter().collect()
                    })
                }
                _ => None,
            }
        }
    }
}

fn terms_equal(a: &str, b: &str) -> bool {
    // Normalize: strip surrounding quotes for comparison
    let a_val = extract_literal_value(a);
    let b_val = extract_literal_value(b);

    // Try numeric comparison first
    if let (Ok(na), Ok(nb)) = (a_val.parse::<f64>(), b_val.parse::<f64>()) {
        return (na - nb).abs() < f64::EPSILON;
    }

    // Fall back to string equality (case-sensitive)
    a_val == b_val
}

fn compare_numeric<F: Fn(f64, f64) -> bool>(
    lhs: &FilterTerm,
    rhs: &FilterTerm,
    binding: &HashMap<String, String>,
    cmp: F,
) -> bool {
    let lhs_str = match resolve_term(lhs, binding) {
        Some(s) => s,
        None => return false,
    };
    let rhs_str = match resolve_term(rhs, binding) {
        Some(s) => s,
        None => return false,
    };
    let lhs_val = extract_literal_value(&lhs_str);
    let rhs_val = extract_literal_value(&rhs_str);
    if let (Ok(a), Ok(b)) = (lhs_val.parse::<f64>(), rhs_val.parse::<f64>()) {
        cmp(a, b)
    } else {
        // Lexicographic comparison as fallback
        let (a, b) = (lhs_val.as_str(), rhs_val.as_str());
        let ord = a.cmp(b);
        match ord {
            std::cmp::Ordering::Less => cmp(0.0, 1.0),
            std::cmp::Ordering::Equal => cmp(0.0, 0.0),
            std::cmp::Ordering::Greater => cmp(1.0, 0.0),
        }
    }
}

// -----------------------------------------------------------------------
// RDF term helpers
// -----------------------------------------------------------------------

/// Extract the language tag from a lang-tagged literal like `"hello"@en`
pub(crate) fn extract_lang(value: &str) -> Option<&str> {
    if value.ends_with('"') {
        return None;
    }
    // Find last @ that follows a closing quote
    if let Some(at_pos) = value.rfind('@') {
        let after_at = &value[at_pos + 1..];
        if !after_at.is_empty() && after_at.chars().all(|c| c.is_alphabetic() || c == '-') {
            return Some(after_at);
        }
    }
    None
}

/// Extract the lexical value from an RDF literal term
pub(crate) fn extract_literal_value(value: &str) -> String {
    if value.starts_with('"') {
        let chars: Vec<char> = value.chars().collect();
        let mut pos = 1usize;
        while pos < chars.len() && chars[pos] != '"' {
            if chars[pos] == '\\' {
                pos += 1; // skip escape
            }
            pos += 1;
        }
        chars[1..pos].iter().collect()
    } else {
        value.to_string()
    }
}

/// Extract the datatype IRI from a typed literal like `"42"^^<xsd:integer>`
pub(crate) fn extract_datatype(value: &str) -> Option<String> {
    if let Some(dt_pos) = value.find("^^") {
        let dt = &value[dt_pos + 2..];
        if dt.starts_with('<') && dt.ends_with('>') {
            Some(dt[1..dt.len() - 1].to_string())
        } else {
            Some(dt.to_string())
        }
    } else {
        None
    }
}

/// Return true if the RDF term is an IRI (not a literal, not a blank node)
pub(crate) fn is_iri_term(value: &str) -> bool {
    !value.starts_with('"') && !value.starts_with('_')
}

/// Return true if the RDF term is a literal (quoted string)
pub(crate) fn is_literal_term(value: &str) -> bool {
    value.starts_with('"')
}

// -----------------------------------------------------------------------
// Simple regex engine (no external crate needed for WASM)
// -----------------------------------------------------------------------

/// A minimal regex engine supporting: `.`, `*`, `+`, `?`, `^`, `$`, `[...]`, `\d`, `\w`, `\s`
pub(crate) fn simple_regex_match(pattern: &str, text: &str) -> bool {
    if let Some(p) = pattern.strip_prefix('^') {
        if let Some(anchor_end) = p.strip_suffix('$') {
            regex_full_match(anchor_end, text)
        } else {
            regex_starts_with(p, text)
        }
    } else if let Some(p) = pattern.strip_suffix('$') {
        regex_ends_with(p, text)
    } else {
        // Search anywhere in text
        for start in 0..=text.len() {
            if regex_starts_with(pattern, &text[start..]) {
                return true;
            }
        }
        false
    }
}

fn regex_full_match(pattern: &str, text: &str) -> bool {
    let pat_chars: Vec<char> = pattern.chars().collect();
    let txt_chars: Vec<char> = text.chars().collect();
    regex_match_dp(&pat_chars, &txt_chars)
}

fn regex_starts_with(pattern: &str, text: &str) -> bool {
    let pat_chars: Vec<char> = pattern.chars().collect();
    let txt_chars: Vec<char> = text.chars().collect();
    // Try matching from the start of text
    if let Some(_end) = regex_find_match(&pat_chars, &txt_chars, 0) {
        return true;
    }
    false
}

fn regex_ends_with(pattern: &str, text: &str) -> bool {
    let txt_chars: Vec<char> = text.chars().collect();
    let pat_chars: Vec<char> = pattern.chars().collect();
    for start in 0..=txt_chars.len() {
        if let Some(end) = regex_find_match(&pat_chars, &txt_chars, start) {
            if end == txt_chars.len() {
                return true;
            }
        }
    }
    false
}

/// Dynamic programming-based regex matcher.  Returns true if the pattern matches the full text.
fn regex_match_dp(pat: &[char], txt: &[char]) -> bool {
    let m = pat.len();
    let n = txt.len();
    // dp[i][j] = pattern[..i] matches text[..j]
    let mut dp = vec![vec![false; n + 1]; m + 1];
    dp[0][0] = true;
    // Handle leading x* patterns
    let mut i = 1;
    while i <= m {
        if i < m && pat[i] == '*' {
            dp[i + 1][0] = dp[i - 1][0];
            i += 2;
        } else {
            i += 1;
        }
    }
    for i in 1..=m {
        for j in 1..=n {
            if pat[i - 1] == '*' {
                dp[i][j] = if i >= 2 {
                    dp[i][j - 1]
                        && (pat[i - 2] == '.'
                            || pat[i - 2] == txt[j - 1]
                            || is_class_match(pat[i - 2], txt[j - 1]))
                } else {
                    false
                };
                if i >= 2 {
                    dp[i][j] = dp[i][j] || dp[i - 2][j];
                }
            } else if pat[i - 1] == '.' {
                dp[i][j] = dp[i - 1][j - 1];
            } else if pat[i - 1] == '\\' && i < m {
                // Escape: handled in next iteration; skip for now
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = dp[i - 1][j - 1]
                    && (pat[i - 1] == txt[j - 1] || is_class_match(pat[i - 1], txt[j - 1]));
            }
        }
    }
    dp[m][n]
}

/// Try to match pattern starting at txt[start], return end position if matched
fn regex_find_match(pat: &[char], txt: &[char], start: usize) -> Option<usize> {
    if pat.is_empty() {
        return Some(start);
    }
    let mut pi = 0usize;
    let mut ti = start;
    while pi < pat.len() {
        let next_star = pi + 1 < pat.len() && pat[pi + 1] == '*';
        let next_plus = pi + 1 < pat.len() && pat[pi + 1] == '+';
        let next_quest = pi + 1 < pat.len() && pat[pi + 1] == '?';

        if next_star {
            // Match 0 or more
            // Try 0
            if let Some(end) = regex_find_match(&pat[pi + 2..], txt, ti) {
                return Some(end);
            }
            // Try 1 or more
            while ti < txt.len() && char_matches(pat[pi], txt[ti], pat, pi) {
                ti += 1;
                if let Some(end) = regex_find_match(&pat[pi + 2..], txt, ti) {
                    return Some(end);
                }
            }
            return None;
        } else if next_plus {
            // Match 1 or more
            if ti >= txt.len() || !char_matches(pat[pi], txt[ti], pat, pi) {
                return None;
            }
            ti += 1;
            // Try more
            while ti < txt.len() && char_matches(pat[pi], txt[ti], pat, pi) {
                if let Some(end) = regex_find_match(&pat[pi + 2..], txt, ti) {
                    return Some(end);
                }
                ti += 1;
            }
            pi += 2;
        } else if next_quest {
            // Match 0 or 1
            if ti < txt.len() && char_matches(pat[pi], txt[ti], pat, pi) {
                if let Some(end) = regex_find_match(&pat[pi + 2..], txt, ti + 1) {
                    return Some(end);
                }
            }
            pi += 2;
        } else if pat[pi] == '\\' && pi + 1 < pat.len() {
            // Escape sequence — check for quantifier at pat[pi+2]
            let esc_char = pat[pi + 1];
            let quant_star = pi + 2 < pat.len() && pat[pi + 2] == '*';
            let quant_plus = pi + 2 < pat.len() && pat[pi + 2] == '+';
            let quant_quest = pi + 2 < pat.len() && pat[pi + 2] == '?';
            if quant_star {
                // \x* — zero or more of escaped class
                if let Some(end) = regex_find_match(&pat[pi + 3..], txt, ti) {
                    return Some(end);
                }
                while ti < txt.len() && char_class_match(esc_char, txt[ti]) {
                    ti += 1;
                    if let Some(end) = regex_find_match(&pat[pi + 3..], txt, ti) {
                        return Some(end);
                    }
                }
                return None;
            } else if quant_plus {
                // \x+ — one or more of escaped class
                if ti >= txt.len() || !char_class_match(esc_char, txt[ti]) {
                    return None;
                }
                ti += 1;
                while ti < txt.len() && char_class_match(esc_char, txt[ti]) {
                    if let Some(end) = regex_find_match(&pat[pi + 3..], txt, ti) {
                        return Some(end);
                    }
                    ti += 1;
                }
                pi += 3;
            } else if quant_quest {
                // \x? — zero or one of escaped class
                if ti < txt.len() && char_class_match(esc_char, txt[ti]) {
                    if let Some(end) = regex_find_match(&pat[pi + 3..], txt, ti + 1) {
                        return Some(end);
                    }
                }
                pi += 3;
            } else {
                // No quantifier — match exactly one
                if ti >= txt.len() {
                    return None;
                }
                if !char_class_match(esc_char, txt[ti]) {
                    return None;
                }
                ti += 1;
                pi += 2;
            }
        } else if pat[pi] == '[' {
            // Character class [...]
            let (class_end, negated, class_chars) = parse_char_class(pat, pi);
            if ti >= txt.len() {
                return None;
            }
            let matched = class_chars.contains(&txt[ti]) != negated;
            if !matched {
                return None;
            }
            ti += 1;
            pi = class_end + 1;
        } else if pat[pi] == '.' {
            if ti >= txt.len() {
                return None;
            }
            ti += 1;
            pi += 1;
        } else {
            if ti >= txt.len() || pat[pi] != txt[ti] {
                return None;
            }
            ti += 1;
            pi += 1;
        }
    }
    Some(ti)
}

fn char_matches(p: char, t: char, pat: &[char], pi: usize) -> bool {
    let _ = (pat, pi);
    p == '.' || p == t || is_class_match(p, t)
}

fn is_class_match(p: char, t: char) -> bool {
    match p {
        'd' => t.is_ascii_digit(),
        'w' => t.is_alphanumeric() || t == '_',
        's' => t.is_whitespace(),
        _ => false,
    }
}

fn char_class_match(escape: char, t: char) -> bool {
    match escape {
        'd' => t.is_ascii_digit(),
        'D' => !t.is_ascii_digit(),
        'w' => t.is_alphanumeric() || t == '_',
        'W' => !(t.is_alphanumeric() || t == '_'),
        's' => t.is_whitespace(),
        'S' => !t.is_whitespace(),
        c => c == t,
    }
}

/// Parse a character class `[...]`, returning (end_index, negated, chars)
fn parse_char_class(pat: &[char], start: usize) -> (usize, bool, Vec<char>) {
    let mut i = start + 1;
    let negated = i < pat.len() && pat[i] == '^';
    if negated {
        i += 1;
    }
    let mut chars = Vec::new();
    while i < pat.len() && pat[i] != ']' {
        if pat[i] == '\\' && i + 1 < pat.len() {
            chars.push(pat[i + 1]);
            i += 2;
        } else if i + 2 < pat.len() && pat[i + 1] == '-' {
            // Range a-z
            let from = pat[i];
            let to = pat[i + 2];
            for c in from..=to {
                chars.push(c);
            }
            i += 3;
        } else {
            chars.push(pat[i]);
            i += 1;
        }
    }
    (i, negated, chars)
}

// -----------------------------------------------------------------------
// Filter parser
// -----------------------------------------------------------------------

/// Parse a FILTER expression string (everything inside `FILTER(...)`) into a [`FilterExpr`]
pub(crate) fn parse_filter_expr(filter_str: &str) -> Option<FilterExpr> {
    let upper = filter_str.to_uppercase();

    // FILTER NOT EXISTS { ... }
    if upper.starts_with("NOT EXISTS") || upper.starts_with("FILTER NOT EXISTS") {
        let brace_start = filter_str.find('{')?;
        let inner = extract_braces(filter_str, brace_start)?;
        return Some(FilterExpr::NotExists {
            inner_sparql: inner,
        });
    }

    // FILTER EXISTS { ... }
    if upper.starts_with("EXISTS") || upper.starts_with("FILTER EXISTS") {
        let brace_start = filter_str.find('{')?;
        let inner = extract_braces(filter_str, brace_start)?;
        return Some(FilterExpr::Exists {
            inner_sparql: inner,
        });
    }

    // Find the FILTER(...) parentheses
    let paren_start = filter_str.find('(')?;
    let chars: Vec<char> = filter_str.chars().collect();
    let mut depth = 0usize;
    let mut in_str = false;
    let mut paren_end = paren_start;
    for (idx, &c) in chars[paren_start..].iter().enumerate() {
        if in_str {
            if c == '"' {
                in_str = false;
            }
        } else {
            match c {
                '"' => in_str = true,
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        paren_end = paren_start + idx;
                        break;
                    }
                }
                _ => {}
            }
        }
    }

    let inner = filter_str[paren_start + 1..paren_end].trim();
    parse_filter_inner(inner)
}

/// Parse the inner content of a FILTER(...) expression
pub(crate) fn parse_filter_inner(inner: &str) -> Option<FilterExpr> {
    let inner = inner.trim();
    let upper = inner.to_uppercase();

    // Logical OR — find top-level ||
    if let Some(pos) = find_top_level_op(inner, "||") {
        let lhs = parse_filter_inner(inner[..pos].trim())?;
        let rhs = parse_filter_inner(inner[pos + 2..].trim())?;
        return Some(FilterExpr::Or(Box::new(lhs), Box::new(rhs)));
    }

    // Logical AND — find top-level &&
    if let Some(pos) = find_top_level_op(inner, "&&") {
        let lhs = parse_filter_inner(inner[..pos].trim())?;
        let rhs = parse_filter_inner(inner[pos + 2..].trim())?;
        return Some(FilterExpr::And(Box::new(lhs), Box::new(rhs)));
    }

    // Logical NOT
    if inner.starts_with('!') && !inner.starts_with("!=") {
        let sub = parse_filter_inner(inner[1..].trim())?;
        return Some(FilterExpr::Not(Box::new(sub)));
    }

    // Parenthesized sub-expression
    if inner.starts_with('(') && inner.ends_with(')') {
        return parse_filter_inner(&inner[1..inner.len() - 1]);
    }

    // BOUND(?var)
    if upper.starts_with("BOUND(") {
        let var_part = &inner[6..inner.len().saturating_sub(1)];
        let var = var_part.trim().trim_start_matches(['?', '$']).to_string();
        return Some(FilterExpr::Bound { var });
    }

    // isIRI(?var)
    if upper.starts_with("ISIRI(") || upper.starts_with("ISURI(") {
        let var_part = &inner[6..inner.len().saturating_sub(1)];
        let var = var_part.trim().trim_start_matches(['?', '$']).to_string();
        return Some(FilterExpr::IsIri { var });
    }

    // isLiteral(?var)
    if upper.starts_with("ISLITERAL(") {
        let var_part = &inner[10..inner.len().saturating_sub(1)];
        let var = var_part.trim().trim_start_matches(['?', '$']).to_string();
        return Some(FilterExpr::IsLiteral { var });
    }

    // isBlank(?var)
    if upper.starts_with("ISBLANK(") {
        let var_part = &inner[8..inner.len().saturating_sub(1)];
        let var = var_part.trim().trim_start_matches(['?', '$']).to_string();
        return Some(FilterExpr::IsBlank { var });
    }

    // LANG(?var) = "tag"  or  LANG(?var) as standalone
    if upper.starts_with("LANG(") {
        let lang_close = inner.find(')')?;
        let var_part = inner[5..lang_close].trim();
        let var = var_part.trim_start_matches(['?', '$']).to_string();
        let rest = inner[lang_close + 1..].trim();
        if let Some(rest_after_eq) = rest.strip_prefix('=') {
            let tag_raw = rest_after_eq.trim();
            let tag = tag_raw.trim_matches('"').to_string();
            return Some(FilterExpr::LangEquals { var, lang: tag });
        }
        return Some(FilterExpr::LangCall { var });
    }

    // regex(?var, "pattern") or regex(?var, "pattern", "flags")
    if upper.starts_with("REGEX(") {
        let args_end = inner.rfind(')')?;
        let args_str = &inner[6..args_end];
        let comma1 = find_top_level_comma(args_str)?;
        let var_part = args_str[..comma1].trim();
        let var = var_part.trim_start_matches(['?', '$']).to_string();
        let rest = &args_str[comma1 + 1..];
        let flags_comma = find_top_level_comma(rest);
        let (pattern_raw, flags) = if let Some(fc) = flags_comma {
            let pat = rest[..fc].trim().trim_matches('"').to_string();
            let fl = rest[fc + 1..].trim().trim_matches('"').to_string();
            (pat, Some(fl))
        } else {
            (rest.trim().trim_matches('"').to_string(), None)
        };
        return Some(FilterExpr::Regex {
            var,
            pattern: pattern_raw,
            flags,
        });
    }

    // STR(?var)
    if upper.starts_with("STR(")
        && !upper.starts_with("STRSTARTS")
        && !upper.starts_with("STRENDS")
        && !upper.starts_with("STRLEN")
    {
        let var_part = &inner[4..inner.len().saturating_sub(1)];
        let var = var_part.trim().trim_start_matches(['?', '$']).to_string();
        return Some(FilterExpr::Str { var });
    }

    // DATATYPE(?var)
    if upper.starts_with("DATATYPE(") {
        let var_part = &inner[9..inner.len().saturating_sub(1)];
        let var = var_part.trim().trim_start_matches(['?', '$']).to_string();
        return Some(FilterExpr::Datatype { var });
    }

    // STRSTARTS(?var, "prefix")
    if upper.starts_with("STRSTARTS(") {
        let args = &inner[10..inner.len().saturating_sub(1)];
        let comma = find_top_level_comma(args)?;
        let var = args[..comma]
            .trim()
            .trim_start_matches(['?', '$'])
            .to_string();
        let prefix = args[comma + 1..].trim().trim_matches('"').to_string();
        return Some(FilterExpr::StrStarts { var, prefix });
    }

    // STRENDS(?var, "suffix")
    if upper.starts_with("STRENDS(") {
        let args = &inner[8..inner.len().saturating_sub(1)];
        let comma = find_top_level_comma(args)?;
        let var = args[..comma]
            .trim()
            .trim_start_matches(['?', '$'])
            .to_string();
        let suffix = args[comma + 1..].trim().trim_matches('"').to_string();
        return Some(FilterExpr::StrEnds { var, suffix });
    }

    // CONTAINS(?var, "substring")
    if upper.starts_with("CONTAINS(") {
        let args = &inner[9..inner.len().saturating_sub(1)];
        let comma = find_top_level_comma(args)?;
        let var = args[..comma]
            .trim()
            .trim_start_matches(['?', '$'])
            .to_string();
        let substring = args[comma + 1..].trim().trim_matches('"').to_string();
        return Some(FilterExpr::Contains { var, substring });
    }

    // STRLEN(?var)
    if upper.starts_with("STRLEN(") {
        let var_part = &inner[7..inner.len().saturating_sub(1)];
        let var = var_part.trim().trim_start_matches(['?', '$']).to_string();
        return Some(FilterExpr::StrLen { var });
    }

    // UCASE(?var)
    if upper.starts_with("UCASE(") {
        let var_part = &inner[6..inner.len().saturating_sub(1)];
        let var = var_part.trim().trim_start_matches(['?', '$']).to_string();
        return Some(FilterExpr::Ucase { var });
    }

    // LCASE(?var)
    if upper.starts_with("LCASE(") {
        let var_part = &inner[6..inner.len().saturating_sub(1)];
        let var = var_part.trim().trim_start_matches(['?', '$']).to_string();
        return Some(FilterExpr::Lcase { var });
    }

    // SUBSTR(?var, start) or SUBSTR(?var, start, length)
    if upper.starts_with("SUBSTR(") {
        let args = &inner[7..inner.len().saturating_sub(1)];
        let comma1 = find_top_level_comma(args)?;
        let var = args[..comma1]
            .trim()
            .trim_start_matches(['?', '$'])
            .to_string();
        let rest = &args[comma1 + 1..];
        let comma2 = find_top_level_comma(rest);
        let (start, length) = if let Some(c2) = comma2 {
            let s: usize = rest[..c2].trim().parse().unwrap_or(1);
            let l: usize = rest[c2 + 1..].trim().parse().unwrap_or(0);
            (s, Some(l))
        } else {
            let s: usize = rest.trim().parse().unwrap_or(1);
            (s, None)
        };
        return Some(FilterExpr::Substr { var, start, length });
    }

    // Comparison expressions: LHS OP RHS
    parse_comparison_filter(inner)
}

/// Parse `lhs OP rhs` filter expressions
pub(crate) fn parse_comparison_filter(expr: &str) -> Option<FilterExpr> {
    const OPS: &[&str] = &[">=", "<=", "!=", ">", "<", "="];

    for op in OPS {
        // Find top-level occurrence of the operator
        if let Some(op_pos) = find_top_level_str(expr, op) {
            let lhs_str = expr[..op_pos].trim();
            let rhs_str = expr[op_pos + op.len()..].trim();

            let lhs = parse_filter_term(lhs_str);
            let rhs = parse_filter_term(rhs_str);

            return Some(match *op {
                "=" => FilterExpr::Equals {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                "!=" => FilterExpr::NotEquals {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                ">" => FilterExpr::GreaterThan {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                ">=" => FilterExpr::GreaterEq {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                "<" => FilterExpr::LessThan {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                "<=" => FilterExpr::LessEq {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                _ => continue,
            });
        }
    }
    None
}

fn parse_filter_term(s: &str) -> FilterTerm {
    let s = s.trim();
    if s.starts_with('?') || s.starts_with('$') {
        FilterTerm::Variable(s.trim_start_matches(['?', '$']).to_string())
    } else if let Ok(n) = s.trim_matches('"').parse::<f64>() {
        FilterTerm::Number(n)
    } else if s.to_uppercase().starts_with("LANG(") {
        // LANG(?var) as a term
        let var_part = s[5..s.len().saturating_sub(1)]
            .trim()
            .trim_start_matches(['?', '$'])
            .to_string();
        FilterTerm::FuncCall(Box::new(FilterExpr::LangCall { var: var_part }))
    } else if s.to_uppercase().starts_with("STR(") {
        let var_part = s[4..s.len().saturating_sub(1)]
            .trim()
            .trim_start_matches(['?', '$'])
            .to_string();
        FilterTerm::FuncCall(Box::new(FilterExpr::Str { var: var_part }))
    } else if s.to_uppercase().starts_with("DATATYPE(") {
        let var_part = s[9..s.len().saturating_sub(1)]
            .trim()
            .trim_start_matches(['?', '$'])
            .to_string();
        FilterTerm::FuncCall(Box::new(FilterExpr::Datatype { var: var_part }))
    } else if s.to_uppercase().starts_with("STRLEN(") {
        let var_part = s[7..s.len().saturating_sub(1)]
            .trim()
            .trim_start_matches(['?', '$'])
            .to_string();
        FilterTerm::FuncCall(Box::new(FilterExpr::StrLen { var: var_part }))
    } else {
        FilterTerm::Literal(s.trim_matches('"').to_string())
    }
}

/// Find top-level position of a two-character operator (not inside parens/strings)
fn find_top_level_str(s: &str, needle: &str) -> Option<usize> {
    let chars: Vec<char> = s.chars().collect();
    let needle_chars: Vec<char> = needle.chars().collect();
    let mut depth = 0usize;
    let mut in_str = false;
    let mut i = 0;
    while i < chars.len() {
        if in_str {
            if chars[i] == '"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        match chars[i] {
            '"' => {
                in_str = true;
                i += 1;
            }
            '(' => {
                depth += 1;
                i += 1;
            }
            ')' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            _ => {
                if depth == 0 && chars[i..].starts_with(needle_chars.as_slice()) {
                    return Some(i);
                }
                i += 1;
            }
        }
    }
    None
}

/// Find top-level position of a logical operator `&&` or `||` (not inside parens/strings)
fn find_top_level_op(s: &str, op: &str) -> Option<usize> {
    find_top_level_str(s, op)
}

/// Find the position of the first top-level comma in `s`
pub(crate) fn find_top_level_comma(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_str = false;
    for (i, c) in s.chars().enumerate() {
        if in_str {
            if c == '"' {
                in_str = false;
            }
        } else {
            match c {
                '"' => in_str = true,
                '(' => depth += 1,
                ')' => {
                    depth = depth.saturating_sub(1);
                }
                ',' if depth == 0 => return Some(i),
                _ => {}
            }
        }
    }
    None
}

/// Extract content from `{...}` at the given offset
fn extract_braces(s: &str, open: usize) -> Option<String> {
    let chars: Vec<char> = s[open + 1..].chars().collect();
    let mut depth = 1usize;
    let mut pos = 0;
    while pos < chars.len() && depth > 0 {
        match chars[pos] {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
        if depth > 0 {
            pos += 1;
        }
    }
    if depth != 0 {
        return None;
    }
    Some(chars[..pos].iter().collect())
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_filter_lang_equals() {
        let expr = FilterExpr::LangEquals {
            var: "x".into(),
            lang: "en".into(),
        };
        assert!(expr.evaluate(&binding(&[("x", "\"hello\"@en")])));
        assert!(!expr.evaluate(&binding(&[("x", "\"hello\"@fr")])));
    }

    #[test]
    fn test_filter_bound_true() {
        let expr = FilterExpr::Bound { var: "x".into() };
        assert!(expr.evaluate(&binding(&[("x", "hello")])));
        assert!(!expr.evaluate(&binding(&[])));
    }

    #[test]
    fn test_filter_is_iri() {
        let expr = FilterExpr::IsIri { var: "x".into() };
        assert!(expr.evaluate(&binding(&[("x", "http://example.org/")])));
        assert!(!expr.evaluate(&binding(&[("x", "\"literal\"")])));
    }

    #[test]
    fn test_filter_is_literal() {
        let expr = FilterExpr::IsLiteral { var: "x".into() };
        assert!(expr.evaluate(&binding(&[("x", "\"literal\"")])));
        assert!(!expr.evaluate(&binding(&[("x", "http://example.org/")])));
    }

    #[test]
    fn test_filter_is_blank() {
        let expr = FilterExpr::IsBlank { var: "x".into() };
        assert!(expr.evaluate(&binding(&[("x", "_:b0")])));
        assert!(!expr.evaluate(&binding(&[("x", "http://example.org/")])));
    }

    #[test]
    fn test_filter_and() {
        let a = FilterExpr::Bound { var: "x".into() };
        let b = FilterExpr::Bound { var: "y".into() };
        let expr = FilterExpr::And(Box::new(a), Box::new(b));
        assert!(expr.evaluate(&binding(&[("x", "a"), ("y", "b")])));
        assert!(!expr.evaluate(&binding(&[("x", "a")])));
    }

    #[test]
    fn test_filter_or() {
        let a = FilterExpr::Bound { var: "x".into() };
        let b = FilterExpr::Bound { var: "y".into() };
        let expr = FilterExpr::Or(Box::new(a), Box::new(b));
        assert!(expr.evaluate(&binding(&[("x", "a")])));
        assert!(expr.evaluate(&binding(&[("y", "b")])));
        assert!(!expr.evaluate(&binding(&[])));
    }

    #[test]
    fn test_filter_not() {
        let inner = FilterExpr::Bound { var: "x".into() };
        let expr = FilterExpr::Not(Box::new(inner));
        assert!(expr.evaluate(&binding(&[])));
        assert!(!expr.evaluate(&binding(&[("x", "a")])));
    }

    #[test]
    fn test_filter_strstarts() {
        let expr = FilterExpr::StrStarts {
            var: "x".into(),
            prefix: "Hello".into(),
        };
        assert!(expr.evaluate(&binding(&[("x", "\"Hello World\"")])));
        assert!(!expr.evaluate(&binding(&[("x", "\"World Hello\"")])));
    }

    #[test]
    fn test_filter_strends() {
        let expr = FilterExpr::StrEnds {
            var: "x".into(),
            suffix: "World".into(),
        };
        assert!(expr.evaluate(&binding(&[("x", "\"Hello World\"")])));
        assert!(!expr.evaluate(&binding(&[("x", "\"World Hello\"")])));
    }

    #[test]
    fn test_filter_contains() {
        let expr = FilterExpr::Contains {
            var: "x".into(),
            substring: "llo".into(),
        };
        assert!(expr.evaluate(&binding(&[("x", "\"Hello\"")])));
        assert!(!expr.evaluate(&binding(&[("x", "\"World\"")])));
    }

    #[test]
    fn test_regex_match_simple() {
        assert!(simple_regex_match("hello", "hello world"));
        assert!(!simple_regex_match("^hello$", "hello world"));
        assert!(simple_regex_match("^hello", "hello world"));
        assert!(simple_regex_match("world$", "hello world"));
    }

    #[test]
    fn test_regex_match_dot_star() {
        assert!(simple_regex_match("h.*o", "hello"));
        assert!(simple_regex_match("^A.*e$", "Alice"));
    }

    #[test]
    fn test_regex_match_digit() {
        assert!(simple_regex_match("\\d+", "abc123def"));
        assert!(!simple_regex_match("^\\d+$", "abc"));
    }

    #[test]
    fn test_parse_filter_bound() {
        let result = parse_filter_expr("FILTER(BOUND(?x))");
        assert!(matches!(result, Some(FilterExpr::Bound { .. })));
    }

    #[test]
    fn test_parse_filter_logical_and() {
        let result = parse_filter_expr("FILTER(BOUND(?x) && BOUND(?y))");
        assert!(matches!(result, Some(FilterExpr::And(_, _))));
    }

    #[test]
    fn test_parse_filter_logical_or() {
        let result = parse_filter_expr("FILTER(BOUND(?x) || BOUND(?y))");
        assert!(matches!(result, Some(FilterExpr::Or(_, _))));
    }

    #[test]
    fn test_parse_filter_not() {
        let result = parse_filter_expr("FILTER(!BOUND(?x))");
        assert!(matches!(result, Some(FilterExpr::Not(_))));
    }

    #[test]
    fn test_parse_filter_strstarts() {
        let result = parse_filter_expr("FILTER(STRSTARTS(?name, \"Alice\"))");
        assert!(matches!(result, Some(FilterExpr::StrStarts { .. })));
    }

    #[test]
    fn test_parse_filter_contains() {
        let result = parse_filter_expr("FILTER(CONTAINS(?s, \"foo\"))");
        assert!(matches!(result, Some(FilterExpr::Contains { .. })));
    }

    #[test]
    fn test_parse_filter_regex_flags() {
        let result = parse_filter_expr("FILTER(regex(?name, \"^alice\", \"i\"))");
        assert!(matches!(
            result,
            Some(FilterExpr::Regex { flags: Some(_), .. })
        ));
    }

    #[test]
    fn test_exists_not_exists_eval_with_closure() {
        let expr = FilterExpr::Exists {
            inner_sparql: "?s ?p ?o".into(),
        };
        let b = binding(&[]);
        assert!(expr.evaluate_with_exists(&b, &|_, _| true));
        assert!(!expr.evaluate_with_exists(&b, &|_, _| false));

        let not_expr = FilterExpr::NotExists {
            inner_sparql: "?s ?p ?o".into(),
        };
        assert!(not_expr.evaluate_with_exists(&b, &|_, _| false));
        assert!(!not_expr.evaluate_with_exists(&b, &|_, _| true));
    }

    #[test]
    fn test_extract_datatype() {
        let dt = extract_datatype("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>");
        assert_eq!(
            dt.as_deref(),
            Some("http://www.w3.org/2001/XMLSchema#integer")
        );
        assert!(extract_datatype("\"hello\"@en").is_none());
    }

    #[test]
    fn test_filter_datatype() {
        let expr = FilterExpr::Datatype { var: "x".into() };
        assert!(expr.evaluate(&binding(&[(
            "x",
            "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"
        )])));
        assert!(!expr.evaluate(&binding(&[("x", "\"hello\"@en")])));
    }
}
