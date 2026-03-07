//! # SPARQL Query Formatter
//!
//! Pretty-prints, minifies, and analyses SPARQL queries.
//!
//! # Example
//!
//! ```rust
//! use oxirs_wasm::sparql_formatter::{SparqlFormatter, FormatOptions};
//!
//! let query = "prefix ex: <http://example.org/> select ?s where { ?s ex:p ?o . }";
//! let opts = FormatOptions { uppercase_keywords: true, ..FormatOptions::default() };
//! let formatted = SparqlFormatter::format(query, &opts);
//! assert!(formatted.contains("PREFIX") || formatted.contains("SELECT"));
//! ```

/// Formatting options for SPARQL pretty-printer
#[derive(Debug, Clone)]
pub struct FormatOptions {
    /// Number of spaces per indentation level (default: 2)
    pub indent_size: usize,
    /// Convert SPARQL keywords to UPPERCASE (default: true)
    pub uppercase_keywords: bool,
    /// Emit a newline before the WHERE keyword (default: true)
    pub newline_before_where: bool,
    /// Emit a newline after SELECT / ASK / CONSTRUCT / DESCRIBE (default: true)
    pub newline_after_select: bool,
    /// Align PREFIX declarations at the IRI column (default: true)
    pub align_prefixes: bool,
    /// Print all triple patterns on a single line (default: false)
    pub compact_triples: bool,
}

impl Default for FormatOptions {
    fn default() -> Self {
        Self {
            indent_size: 2,
            uppercase_keywords: true,
            newline_before_where: true,
            newline_after_select: true,
            align_prefixes: true,
            compact_triples: false,
        }
    }
}

/// A lexical token produced by the SPARQL tokeniser
#[derive(Debug, Clone, PartialEq)]
pub enum SparqlToken {
    /// A SPARQL keyword (SELECT, WHERE, PREFIX, …)
    Keyword(String),
    /// A variable reference (?name or $name)
    Variable(String),
    /// A full IRI enclosed in angle brackets (<…>)
    Iri(String),
    /// A prefixed name declaration (prefix part, e.g. `rdf:`)
    Prefix(String),
    /// A string literal ("…" or '''…''')
    Literal(String),
    /// A single punctuation character ({}, ;, ,, .)
    Punctuation(char),
    /// A whitespace run (collapsed to a single token)
    Whitespace,
    /// A newline
    Newline,
    /// A comment (# …)
    Comment(String),
}

/// Known SPARQL keywords (case-insensitive)
const KEYWORDS: &[&str] = &[
    "SELECT",
    "WHERE",
    "PREFIX",
    "BASE",
    "FILTER",
    "OPTIONAL",
    "UNION",
    "GRAPH",
    "BIND",
    "VALUES",
    "LIMIT",
    "OFFSET",
    "ORDER",
    "BY",
    "GROUP",
    "HAVING",
    "DISTINCT",
    "REDUCED",
    "FROM",
    "NAMED",
    "AS",
    "ASK",
    "CONSTRUCT",
    "DESCRIBE",
    "NOT",
    "EXISTS",
    "MINUS",
    "SERVICE",
    "SILENT",
    "INSERT",
    "DELETE",
    "LOAD",
    "CLEAR",
    "DROP",
    "CREATE",
    "ADD",
    "MOVE",
    "COPY",
    "WITH",
    "USING",
    "INTO",
    "DATA",
    "ALL",
    "DEFAULT",
    "IN",
];

/// Stateless SPARQL formatting utility
pub struct SparqlFormatter;

impl SparqlFormatter {
    // -----------------------------------------------------------------------
    // Tokeniser
    // -----------------------------------------------------------------------

    /// Tokenise a SPARQL query string into a `Vec<SparqlToken>`.
    ///
    /// Tokens:
    /// - `#…` comments
    /// - `?var` / `$var` variables
    /// - `<…>` IRIs
    /// - `"…"` / `'…'` literals (including triple-quote forms)
    /// - Known keywords (matched greedily against word boundaries)
    /// - Prefixed names (`ns:local`)
    /// - Punctuation ({};,.)
    /// - Whitespace / newlines (collapsed)
    pub fn tokenize(query: &str) -> Vec<SparqlToken> {
        let chars: Vec<char> = query.chars().collect();
        let n = chars.len();
        let mut tokens = Vec::new();
        let mut i = 0;

        while i < n {
            let c = chars[i];

            // --- Comment ---
            if c == '#' {
                let start = i + 1;
                while i < n && chars[i] != '\n' {
                    i += 1;
                }
                let comment: String = chars[start..i].iter().collect();
                tokens.push(SparqlToken::Comment(comment.trim().to_string()));
                continue;
            }

            // --- Newline ---
            if c == '\n' {
                // Collapse consecutive newlines
                while i < n && chars[i] == '\n' {
                    i += 1;
                }
                tokens.push(SparqlToken::Newline);
                continue;
            }

            // --- Other whitespace ---
            if c.is_whitespace() {
                while i < n && chars[i].is_whitespace() && chars[i] != '\n' {
                    i += 1;
                }
                tokens.push(SparqlToken::Whitespace);
                continue;
            }

            // --- Variable (?x or $x) ---
            if c == '?' || c == '$' {
                i += 1;
                let start = i;
                while i < n && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let var: String = chars[start..i].iter().collect();
                tokens.push(SparqlToken::Variable(var));
                continue;
            }

            // --- IRI (<…>) ---
            if c == '<' {
                i += 1;
                let start = i;
                while i < n && chars[i] != '>' {
                    i += 1;
                }
                let iri: String = chars[start..i].iter().collect();
                if i < n {
                    i += 1; // consume '>'
                }
                tokens.push(SparqlToken::Iri(iri));
                continue;
            }

            // --- String literals ---
            if c == '"' || c == '\'' {
                // Check for triple-quote
                let quote_char = c;
                let triple = i + 2 < n && chars[i + 1] == quote_char && chars[i + 2] == quote_char;
                if triple {
                    i += 3;
                    let start = i;
                    while i + 2 < n
                        && !(chars[i] == quote_char
                            && chars[i + 1] == quote_char
                            && chars[i + 2] == quote_char)
                    {
                        i += 1;
                    }
                    let lit: String = chars[start..i].iter().collect();
                    i += 3; // consume closing triple-quote
                    tokens.push(SparqlToken::Literal(lit));
                } else {
                    i += 1;
                    let start = i;
                    while i < n && chars[i] != quote_char {
                        if chars[i] == '\\' {
                            i += 1; // skip escaped char
                        }
                        i += 1;
                    }
                    let lit: String = chars[start..i].iter().collect();
                    if i < n {
                        i += 1; // consume closing quote
                    }
                    tokens.push(SparqlToken::Literal(lit));
                }
                continue;
            }

            // --- Punctuation ---
            if matches!(
                c,
                '{' | '}' | ';' | ',' | '.' | '(' | ')' | '[' | ']' | '*' | '+' | '|'
            ) {
                tokens.push(SparqlToken::Punctuation(c));
                i += 1;
                continue;
            }

            // --- Word tokens: keywords, prefixed names, bare words ---
            if c.is_alphabetic() || c == '_' || c == '@' {
                let start = i;
                // Consume leading word chars
                while i < n && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '-') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();

                // Check for prefix colon: `rdf:` or `rdf:local`
                if i < n && chars[i] == ':' {
                    i += 1; // consume ':'
                    let prefix_start = i;
                    while i < n
                        && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '-')
                    {
                        i += 1;
                    }
                    let local: String = chars[prefix_start..i].iter().collect();
                    let full = if local.is_empty() {
                        format!("{word}:")
                    } else {
                        format!("{word}:{local}")
                    };
                    tokens.push(SparqlToken::Prefix(full));
                    continue;
                }

                // Check keyword
                let upper = word.to_uppercase();
                if KEYWORDS.contains(&upper.as_str()) {
                    tokens.push(SparqlToken::Keyword(upper));
                } else {
                    // Bare word (could be language tag, type suffix, etc.)
                    tokens.push(SparqlToken::Prefix(word));
                }
                continue;
            }

            // --- Numeric literals and operators (pass through as prefix tokens) ---
            if c.is_ascii_digit()
                || c == '-'
                || c == '+'
                || c == '='
                || c == '!'
                || c == '<'
                || c == '>'
                || c == '&'
                || c == '^'
                || c == '@'
            {
                let start = i;
                i += 1;
                while i < n
                    && !chars[i].is_whitespace()
                    && !matches!(chars[i], '{' | '}' | ';' | ',' | '.' | '(' | ')')
                {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                tokens.push(SparqlToken::Prefix(word));
                continue;
            }

            // Fallback: skip unknown character
            i += 1;
        }

        tokens
    }

    // -----------------------------------------------------------------------
    // Formatter
    // -----------------------------------------------------------------------

    /// Format a SPARQL query according to `options`.
    pub fn format(query: &str, options: &FormatOptions) -> String {
        // Work on a normalised, minified version first to simplify parsing
        let normalised = Self::minify(query);
        let indent = " ".repeat(options.indent_size);

        let mut out = String::new();
        let mut depth: i32 = 0;
        // Track state for blank-line insertion
        let mut in_where = false;

        // We do a token-based reformatter
        let tokens = Self::tokenize(&normalised);
        let n = tokens.len();
        let mut i = 0;

        // Collect PREFIX lines first for alignment
        let prefixes = Self::extract_prefixes(query);
        let max_prefix_len = if options.align_prefixes && !prefixes.is_empty() {
            prefixes.iter().map(|(p, _)| p.len()).max().unwrap_or(0)
        } else {
            0
        };

        // Output PREFIX declarations
        if !prefixes.is_empty() {
            for (prefix, iri) in &prefixes {
                let kw = if options.uppercase_keywords {
                    "PREFIX"
                } else {
                    "prefix"
                };
                if options.align_prefixes && max_prefix_len > 0 {
                    let padding = " ".repeat(max_prefix_len - prefix.len());
                    out.push_str(&format!("{kw} {prefix}: {padding}<{iri}>\n"));
                } else {
                    out.push_str(&format!("{kw} {prefix}:<{iri}>\n"));
                }
            }
            out.push('\n');
        }

        // Now emit non-PREFIX tokens
        while i < n {
            let tok = &tokens[i];
            match tok {
                SparqlToken::Keyword(kw) => {
                    let emit = if options.uppercase_keywords {
                        kw.clone()
                    } else {
                        kw.to_lowercase()
                    };

                    match kw.as_str() {
                        "PREFIX" | "BASE" => {
                            // Already emitted above — skip until end of declaration
                            i += 1;
                            // Skip prefix token, whitespace, IRI
                            while i < n {
                                match &tokens[i] {
                                    SparqlToken::Iri(_) => {
                                        i += 1;
                                        break;
                                    }
                                    _ => i += 1,
                                }
                            }
                            continue;
                        }
                        "SELECT" | "ASK" | "CONSTRUCT" | "DESCRIBE" => {
                            out.push_str(&emit);
                            if options.newline_after_select {
                                out.push(' ');
                            }
                        }
                        "WHERE" => {
                            in_where = true;
                            if options.newline_before_where {
                                out.push('\n');
                            } else {
                                out.push(' ');
                            }
                            out.push_str(&emit);
                            out.push(' ');
                        }
                        "FILTER" | "OPTIONAL" | "BIND" | "VALUES" | "UNION" | "MINUS"
                        | "SERVICE" => {
                            if in_where && depth > 0 {
                                out.push('\n');
                                for _ in 0..depth {
                                    out.push_str(&indent);
                                }
                            }
                            out.push_str(&emit);
                            out.push(' ');
                        }
                        "LIMIT" | "OFFSET" | "ORDER" | "GROUP" | "HAVING" => {
                            out.push('\n');
                            out.push_str(&emit);
                            out.push(' ');
                        }
                        _ => {
                            out.push_str(&emit);
                            out.push(' ');
                        }
                    }
                }
                SparqlToken::Punctuation(c) => {
                    match c {
                        '{' => {
                            out.push('{');
                            depth += 1;
                            out.push('\n');
                            for _ in 0..depth {
                                out.push_str(&indent);
                            }
                        }
                        '}' => {
                            depth -= 1;
                            if depth < 0 {
                                depth = 0;
                            }
                            // Trim trailing whitespace on current line
                            let trimmed = out.trim_end_matches([' ', '\t']).to_string();
                            out = trimmed;
                            out.push('\n');
                            for _ in 0..depth {
                                out.push_str(&indent);
                            }
                            out.push('}');
                            if depth == 0 {
                                out.push('\n');
                            }
                        }
                        '.' => {
                            if options.compact_triples {
                                out.push_str(". ");
                            } else {
                                out.push_str(" .");
                                // Check if next non-whitespace token closes brace
                                let mut peek = i + 1;
                                while peek < n
                                    && matches!(
                                        tokens[peek],
                                        SparqlToken::Whitespace | SparqlToken::Newline
                                    )
                                {
                                    peek += 1;
                                }
                                let next_is_close = peek < n
                                    && matches!(&tokens[peek], SparqlToken::Punctuation('}'));
                                if !next_is_close {
                                    out.push('\n');
                                    for _ in 0..depth {
                                        out.push_str(&indent);
                                    }
                                }
                            }
                        }
                        ';' => {
                            out.push_str(" ;");
                            if !options.compact_triples {
                                out.push('\n');
                                for _ in 0..depth {
                                    out.push_str(&indent);
                                }
                            } else {
                                out.push(' ');
                            }
                        }
                        ',' => {
                            out.push_str(", ");
                        }
                        _ => {
                            out.push(*c);
                        }
                    }
                }
                SparqlToken::Variable(v) => {
                    out.push('?');
                    out.push_str(v);
                    out.push(' ');
                }
                SparqlToken::Iri(iri) => {
                    out.push('<');
                    out.push_str(iri);
                    out.push('>');
                    out.push(' ');
                }
                SparqlToken::Prefix(p) => {
                    out.push_str(p);
                    out.push(' ');
                }
                SparqlToken::Literal(lit) => {
                    out.push('"');
                    out.push_str(lit);
                    out.push('"');
                    out.push(' ');
                }
                SparqlToken::Comment(c) => {
                    out.push_str("# ");
                    out.push_str(c);
                    out.push('\n');
                    for _ in 0..depth {
                        out.push_str(&indent);
                    }
                }
                SparqlToken::Whitespace | SparqlToken::Newline => {
                    // Consumed by context — skip
                }
            }
            i += 1;
        }

        // Clean up trailing whitespace on each line
        out.lines()
            .map(|line| line.trim_end())
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string()
    }

    // -----------------------------------------------------------------------
    // Minifier
    // -----------------------------------------------------------------------

    /// Collapse whitespace and remove comments, producing a compact single-line query.
    pub fn minify(query: &str) -> String {
        let tokens = Self::tokenize(query);
        let mut out = String::new();
        let mut last_was_sep = true; // avoid leading space

        for tok in &tokens {
            match tok {
                SparqlToken::Comment(_) | SparqlToken::Newline | SparqlToken::Whitespace => {
                    if !last_was_sep && !out.is_empty() {
                        // Only insert space if needed
                        let last_char = out.chars().last().unwrap_or(' ');
                        if !matches!(last_char, '{' | '(' | '[') {
                            out.push(' ');
                        }
                        last_was_sep = true;
                    }
                }
                SparqlToken::Keyword(kw) => {
                    if last_was_sep {
                        // trim any trailing space we added
                        if out.ends_with(' ') {
                            out.pop();
                        }
                    }
                    out.push_str(kw);
                    out.push(' ');
                    last_was_sep = true;
                }
                SparqlToken::Variable(v) => {
                    if out.ends_with(' ') {
                        out.pop();
                    }
                    out.push('?');
                    out.push_str(v);
                    out.push(' ');
                    last_was_sep = true;
                }
                SparqlToken::Iri(iri) => {
                    if out.ends_with(' ') {
                        out.pop();
                    }
                    out.push('<');
                    out.push_str(iri);
                    out.push('>');
                    out.push(' ');
                    last_was_sep = true;
                }
                SparqlToken::Prefix(p) => {
                    if out.ends_with(' ') && last_was_sep {
                        out.pop();
                    }
                    out.push_str(p);
                    out.push(' ');
                    last_was_sep = true;
                }
                SparqlToken::Literal(lit) => {
                    out.push('"');
                    out.push_str(lit);
                    out.push('"');
                    out.push(' ');
                    last_was_sep = true;
                }
                SparqlToken::Punctuation(c) => {
                    // Remove spaces before/after certain punctuation
                    if matches!(c, '{' | '}' | '(' | ')' | '[' | ']') && out.ends_with(' ') {
                        out.pop();
                    }
                    out.push(*c);
                    // Space after opening delimiters and separators
                    if matches!(c, '{' | '(' | '.' | ';' | ',') {
                        out.push(' ');
                        last_was_sep = true;
                    } else {
                        last_was_sep = false;
                    }
                }
            }
        }

        // Trim and collapse multiple spaces
        let mut result = String::new();
        let mut prev_space = false;
        for ch in out.chars() {
            if ch == ' ' {
                if !prev_space {
                    result.push(ch);
                }
                prev_space = true;
            } else {
                prev_space = false;
                result.push(ch);
            }
        }
        result.trim().to_string()
    }

    // -----------------------------------------------------------------------
    // Variable extraction
    // -----------------------------------------------------------------------

    /// Return a sorted, deduplicated list of variable names (without `?`/`$` prefix).
    pub fn extract_variables(query: &str) -> Vec<String> {
        let tokens = Self::tokenize(query);
        let mut vars: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for tok in &tokens {
            if let SparqlToken::Variable(v) = tok {
                vars.insert(v.clone());
            }
        }
        vars.into_iter().collect()
    }

    // -----------------------------------------------------------------------
    // Prefix extraction
    // -----------------------------------------------------------------------

    /// Return ordered (prefix_name, iri) pairs from PREFIX declarations.
    ///
    /// Only returns entries of the form `PREFIX ns: <iri>`.
    pub fn extract_prefixes(query: &str) -> Vec<(String, String)> {
        let tokens = Self::tokenize(query);
        let mut prefixes: Vec<(String, String)> = Vec::new();
        let n = tokens.len();
        let mut i = 0;

        while i < n {
            if let SparqlToken::Keyword(kw) = &tokens[i] {
                if kw == "PREFIX" || kw == "BASE" {
                    i += 1;
                    // Skip whitespace
                    while i < n
                        && matches!(tokens[i], SparqlToken::Whitespace | SparqlToken::Newline)
                    {
                        i += 1;
                    }
                    // Next should be the prefix token (e.g. `rdf:`)
                    if i < n {
                        if let SparqlToken::Prefix(p) = &tokens[i] {
                            // Strip trailing colon if present
                            let prefix_name = if p.ends_with(':') {
                                p[..p.len() - 1].to_string()
                            } else {
                                p.clone()
                            };
                            i += 1;
                            // Skip whitespace
                            while i < n
                                && matches!(
                                    tokens[i],
                                    SparqlToken::Whitespace | SparqlToken::Newline
                                )
                            {
                                i += 1;
                            }
                            // Next should be the IRI
                            if i < n {
                                if let SparqlToken::Iri(iri) = &tokens[i] {
                                    prefixes.push((prefix_name, iri.clone()));
                                }
                            }
                        }
                    }
                }
            }
            i += 1;
        }

        prefixes
    }

    // -----------------------------------------------------------------------
    // Keyword detection
    // -----------------------------------------------------------------------

    /// Return `true` if the query contains `keyword` (case-insensitive, word boundary).
    pub fn has_keyword(query: &str, keyword: &str) -> bool {
        let upper_kw = keyword.to_uppercase();
        let tokens = Self::tokenize(query);
        for tok in &tokens {
            if let SparqlToken::Keyword(kw) = tok {
                if kw.to_uppercase() == upper_kw {
                    return true;
                }
            }
        }
        false
    }

    // -----------------------------------------------------------------------
    // Triple counter
    // -----------------------------------------------------------------------

    /// Estimate the number of triple patterns by counting `.` inside WHERE blocks.
    ///
    /// This is intentionally an estimate — it counts `.` punctuation tokens that
    /// appear while inside at least one `{…}` block.
    pub fn count_triples(query: &str) -> usize {
        let tokens = Self::tokenize(query);
        let mut depth: i32 = 0;
        let mut in_where = false;
        let mut count = 0usize;

        for tok in &tokens {
            match tok {
                SparqlToken::Keyword(kw) if kw == "WHERE" => {
                    in_where = true;
                }
                SparqlToken::Punctuation('{') => {
                    depth += 1;
                }
                SparqlToken::Punctuation('}') => {
                    depth -= 1;
                    if depth <= 0 {
                        in_where = false;
                        depth = 0;
                    }
                }
                SparqlToken::Punctuation('.') => {
                    if in_where && depth > 0 {
                        count += 1;
                    }
                }
                _ => {}
            }
        }

        count
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Tokeniser ---

    #[test]
    fn test_tokenize_keyword_select() {
        let tokens = SparqlFormatter::tokenize("SELECT ?s");
        assert!(tokens
            .iter()
            .any(|t| matches!(t, SparqlToken::Keyword(k) if k == "SELECT")));
    }

    #[test]
    fn test_tokenize_variable() {
        let tokens = SparqlFormatter::tokenize("?subject");
        assert!(tokens
            .iter()
            .any(|t| matches!(t, SparqlToken::Variable(v) if v == "subject")));
    }

    #[test]
    fn test_tokenize_iri() {
        let tokens = SparqlFormatter::tokenize("<http://example.org/>");
        assert!(tokens
            .iter()
            .any(|t| matches!(t, SparqlToken::Iri(iri) if iri == "http://example.org/")));
    }

    #[test]
    fn test_tokenize_prefix() {
        let tokens = SparqlFormatter::tokenize("ex:name");
        assert!(tokens
            .iter()
            .any(|t| matches!(t, SparqlToken::Prefix(p) if p == "ex:name")));
    }

    #[test]
    fn test_tokenize_literal() {
        let tokens = SparqlFormatter::tokenize("\"Alice\"");
        assert!(tokens
            .iter()
            .any(|t| matches!(t, SparqlToken::Literal(l) if l == "Alice")));
    }

    #[test]
    fn test_tokenize_comment() {
        let tokens = SparqlFormatter::tokenize("# a comment\nSELECT");
        assert!(tokens
            .iter()
            .any(|t| matches!(t, SparqlToken::Comment(c) if c == "a comment")));
    }

    #[test]
    fn test_tokenize_punctuation() {
        let tokens = SparqlFormatter::tokenize("{ }");
        assert!(tokens
            .iter()
            .any(|t| matches!(t, SparqlToken::Punctuation('{'))));
        assert!(tokens
            .iter()
            .any(|t| matches!(t, SparqlToken::Punctuation('}'))));
    }

    #[test]
    fn test_tokenize_dollar_variable() {
        let tokens = SparqlFormatter::tokenize("$var");
        assert!(tokens
            .iter()
            .any(|t| matches!(t, SparqlToken::Variable(v) if v == "var")));
    }

    // --- Minifier ---

    #[test]
    fn test_minify_collapses_whitespace() {
        let q = "SELECT   ?s   WHERE  {  ?s  ?p  ?o  .  }";
        let m = SparqlFormatter::minify(q);
        assert!(!m.contains("  "));
    }

    #[test]
    fn test_minify_removes_comments() {
        let q = "# comment\nSELECT ?s WHERE { ?s ?p ?o . }";
        let m = SparqlFormatter::minify(q);
        assert!(!m.contains('#'));
        assert!(!m.contains("comment"));
    }

    #[test]
    fn test_minify_preserves_keywords() {
        let q = "SELECT ?s WHERE { ?s ?p ?o . }";
        let m = SparqlFormatter::minify(q);
        assert!(m.contains("SELECT") || m.contains("select"));
        assert!(m.contains("WHERE") || m.contains("where"));
    }

    #[test]
    fn test_minify_empty_string() {
        let m = SparqlFormatter::minify("");
        assert!(m.is_empty());
    }

    // --- Variable extraction ---

    #[test]
    fn test_extract_variables_basic() {
        let q = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }";
        let vars = SparqlFormatter::extract_variables(q);
        assert!(vars.contains(&"s".to_string()));
        assert!(vars.contains(&"p".to_string()));
        assert!(vars.contains(&"o".to_string()));
    }

    #[test]
    fn test_extract_variables_sorted() {
        let q = "SELECT ?z ?a ?m WHERE { ?z ?a ?m . }";
        let vars = SparqlFormatter::extract_variables(q);
        assert_eq!(vars, vec!["a", "m", "z"]);
    }

    #[test]
    fn test_extract_variables_deduplicated() {
        let q = "SELECT ?s WHERE { ?s ?p ?o . ?s ?q ?r . }";
        let vars = SparqlFormatter::extract_variables(q);
        assert_eq!(vars.iter().filter(|v| v.as_str() == "s").count(), 1);
    }

    #[test]
    fn test_extract_variables_empty() {
        let q = "ASK { <http://example.org/a> <http://example.org/b> <http://example.org/c> . }";
        let vars = SparqlFormatter::extract_variables(q);
        assert!(vars.is_empty());
    }

    // --- Prefix extraction ---

    #[test]
    fn test_extract_prefixes_single() {
        let q = "PREFIX ex: <http://example.org/> SELECT ?s WHERE { ?s ex:p ?o . }";
        let prefixes = SparqlFormatter::extract_prefixes(q);
        assert_eq!(prefixes.len(), 1);
        assert_eq!(prefixes[0].0, "ex");
        assert_eq!(prefixes[0].1, "http://example.org/");
    }

    #[test]
    fn test_extract_prefixes_multiple() {
        let q = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nPREFIX owl: <http://www.w3.org/2002/07/owl#>\nSELECT ?s WHERE { }";
        let prefixes = SparqlFormatter::extract_prefixes(q);
        assert_eq!(prefixes.len(), 2);
        let names: Vec<&str> = prefixes.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"rdf"));
        assert!(names.contains(&"owl"));
    }

    #[test]
    fn test_extract_prefixes_empty() {
        let q = "SELECT ?s WHERE { ?s ?p ?o . }";
        let prefixes = SparqlFormatter::extract_prefixes(q);
        assert!(prefixes.is_empty());
    }

    #[test]
    fn test_extract_prefixes_order() {
        let q = "PREFIX a: <http://a.org/>\nPREFIX b: <http://b.org/>\nSELECT ?s WHERE { }";
        let prefixes = SparqlFormatter::extract_prefixes(q);
        assert_eq!(prefixes[0].0, "a");
        assert_eq!(prefixes[1].0, "b");
    }

    // --- has_keyword ---

    #[test]
    fn test_has_keyword_select() {
        assert!(SparqlFormatter::has_keyword(
            "SELECT ?s WHERE { }",
            "select"
        ));
        assert!(SparqlFormatter::has_keyword(
            "SELECT ?s WHERE { }",
            "SELECT"
        ));
    }

    #[test]
    fn test_has_keyword_optional() {
        let q = "SELECT ?s WHERE { ?s ?p ?o . OPTIONAL { ?s ?q ?r . } }";
        assert!(SparqlFormatter::has_keyword(q, "OPTIONAL"));
    }

    #[test]
    fn test_has_keyword_missing() {
        let q = "SELECT ?s WHERE { ?s ?p ?o . }";
        assert!(!SparqlFormatter::has_keyword(q, "CONSTRUCT"));
    }

    #[test]
    fn test_has_keyword_ask() {
        let q = "ASK { <http://a.org/> <http://b.org/> <http://c.org/> . }";
        assert!(SparqlFormatter::has_keyword(q, "ASK"));
    }

    // --- count_triples ---

    #[test]
    fn test_count_triples_single() {
        let q = "SELECT ?s WHERE { ?s ?p ?o . }";
        assert_eq!(SparqlFormatter::count_triples(q), 1);
    }

    #[test]
    fn test_count_triples_multiple() {
        let q = "SELECT ?s WHERE { ?s ?p ?o . ?s ?q ?r . }";
        assert_eq!(SparqlFormatter::count_triples(q), 2);
    }

    #[test]
    fn test_count_triples_none() {
        // No WHERE block
        let q = "ASK { }";
        assert_eq!(SparqlFormatter::count_triples(q), 0);
    }

    // --- format ---

    #[test]
    fn test_format_uppercase_keywords() {
        let q = "select ?s where { ?s ?p ?o . }";
        let opts = FormatOptions {
            uppercase_keywords: true,
            ..Default::default()
        };
        let formatted = SparqlFormatter::format(q, &opts);
        assert!(formatted.contains("SELECT") || formatted.contains("WHERE"));
    }

    #[test]
    fn test_format_lowercase_keywords() {
        let q = "SELECT ?s WHERE { ?s ?p ?o . }";
        let opts = FormatOptions {
            uppercase_keywords: false,
            ..Default::default()
        };
        let formatted = SparqlFormatter::format(q, &opts);
        assert!(formatted.contains("select") || formatted.contains("where"));
    }

    #[test]
    fn test_format_prefix_emitted() {
        let q = "PREFIX ex: <http://example.org/> SELECT ?s WHERE { ?s ex:p ?o . }";
        let opts = FormatOptions::default();
        let formatted = SparqlFormatter::format(q, &opts);
        assert!(formatted.contains("http://example.org/"));
    }

    #[test]
    fn test_format_contains_variable() {
        let q = "SELECT ?subject WHERE { ?subject ?predicate ?object . }";
        let opts = FormatOptions::default();
        let formatted = SparqlFormatter::format(q, &opts);
        assert!(formatted.contains("?subject") || formatted.contains("subject"));
    }

    #[test]
    fn test_format_indentation_increases_in_braces() {
        let q = "SELECT ?s WHERE { ?s ?p ?o . }";
        let opts = FormatOptions {
            indent_size: 4,
            ..Default::default()
        };
        let formatted = SparqlFormatter::format(q, &opts);
        // The body of the WHERE block should be indented
        let lines: Vec<&str> = formatted.lines().collect();
        let indented = lines.iter().any(|l| l.starts_with("    "));
        assert!(indented, "formatted=\n{formatted}");
    }

    #[test]
    fn test_format_no_double_whitespace() {
        let q = "SELECT   ?s   WHERE  {  ?s  ?p  ?o  .  }";
        let opts = FormatOptions::default();
        let formatted = SparqlFormatter::format(q, &opts);
        assert!(!formatted.contains("  ?s  "));
    }

    #[test]
    fn test_format_default_options() {
        let opts = FormatOptions::default();
        assert_eq!(opts.indent_size, 2);
        assert!(opts.uppercase_keywords);
        assert!(opts.newline_before_where);
        assert!(opts.newline_after_select);
        assert!(opts.align_prefixes);
        assert!(!opts.compact_triples);
    }

    // --- SparqlToken clone/debug ---

    #[test]
    fn test_token_clone() {
        let t = SparqlToken::Keyword("SELECT".to_string());
        let t2 = t.clone();
        assert_eq!(t, t2);
    }

    #[test]
    fn test_token_debug() {
        let t = SparqlToken::Variable("x".to_string());
        let s = format!("{t:?}");
        assert!(s.contains("Variable"));
    }

    #[test]
    fn test_minify_triple_quote_literal() {
        let q = r#"SELECT ?s WHERE { ?s ?p '''hello world''' . }"#;
        let m = SparqlFormatter::minify(q);
        assert!(m.contains("hello world"));
    }

    #[test]
    fn test_has_keyword_filter() {
        let q = "SELECT ?s WHERE { ?s ?p ?o . FILTER(?o > 5) }";
        assert!(SparqlFormatter::has_keyword(q, "FILTER"));
    }
}
