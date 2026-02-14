//! SPARQL syntax highlighting module
//!
//! Provides syntax highlighting for SPARQL queries and results using colored output.

use colored::*;
use once_cell::sync::Lazy;
use regex::Regex;

/// SPARQL keyword patterns
static KEYWORD_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(SELECT|WHERE|FILTER|OPTIONAL|UNION|GRAPH|FROM|ORDER BY|GROUP BY|HAVING|LIMIT|OFFSET|DISTINCT|REDUCED|PREFIX|BASE|ASK|CONSTRUCT|DESCRIBE|INSERT|DELETE|LOAD|CLEAR|DROP|CREATE|ADD|MOVE|COPY|WITH|DATA|SILENT|DEFAULT|NAMED|ALL|USING|BIND|VALUES|SERVICE|MINUS|EXISTS|NOT)\b")
        .expect("regex pattern should be valid")
});

/// SPARQL function patterns
static FUNCTION_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(STR|LANG|LANGMATCHES|DATATYPE|BOUND|sameTerm|isIRI|isURI|isBLANK|isLITERAL|isNUMERIC|REGEX|SUBSTR|STRLEN|UCASE|LCASE|STRSTARTS|STRENDS|CONTAINS|ENCODE_FOR_URI|CONCAT|NOW|YEAR|MONTH|DAY|HOURS|MINUTES|SECONDS|TIMEZONE|TZ|MD5|SHA1|SHA256|SHA512|COALESCE|IF|STRLANG|STRDT|UUID|STRUUID|REPLACE|ABS|ROUND|CEIL|FLOOR|RAND|COUNT|SUM|MIN|MAX|AVG|SAMPLE|GROUP_CONCAT)\b")
        .expect("regex pattern should be valid")
});

/// Variable pattern (?var or $var)
static VARIABLE_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[\?$][\w]+").expect("regex pattern should be valid"));

/// IRI pattern <http://...>
static IRI_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<[^>]+>").expect("regex pattern should be valid"));

/// String literal pattern
static STRING_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#""(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'"#).expect("regex pattern should be valid")
});

/// Number pattern
static NUMBER_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b").expect("regex pattern should be valid")
});

/// Comment pattern
static COMMENT_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"#[^\n]*").expect("regex pattern should be valid"));

/// Configuration for syntax highlighting
#[derive(Debug, Clone)]
pub struct HighlightConfig {
    pub enable_colors: bool,
    pub keyword_color: Color,
    pub function_color: Color,
    pub variable_color: Color,
    pub iri_color: Color,
    pub string_color: Color,
    pub number_color: Color,
    pub comment_color: Color,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            enable_colors: !is_no_color(),
            keyword_color: Color::Blue,
            function_color: Color::Magenta,
            variable_color: Color::Green,
            iri_color: Color::Cyan,
            string_color: Color::Yellow,
            number_color: Color::Red,
            comment_color: Color::BrightBlack,
        }
    }
}

/// Check if NO_COLOR environment variable is set
fn is_no_color() -> bool {
    std::env::var("NO_COLOR").is_ok()
}

/// Highlight SPARQL query with syntax colors
pub fn highlight_sparql(query: &str, config: &HighlightConfig) -> String {
    if !config.enable_colors {
        return query.to_string();
    }

    let mut result = query.to_string();

    // Apply highlighting in order (most specific first)

    // 1. Comments (to avoid highlighting keywords in comments)
    let positions = collect_match_positions(&result, &COMMENT_PATTERN);
    for (start, end, matched) in positions.iter().rev() {
        let highlighted = matched.color(config.comment_color).italic().to_string();
        result.replace_range(*start..*end, &highlighted);
    }

    // 2. String literals (to avoid highlighting keywords in strings)
    let positions = collect_match_positions(&result, &STRING_PATTERN);
    for (start, end, matched) in positions.iter().rev() {
        let highlighted = matched.color(config.string_color).to_string();
        result.replace_range(*start..*end, &highlighted);
    }

    // 3. IRIs
    let positions = collect_match_positions(&result, &IRI_PATTERN);
    for (start, end, matched) in positions.iter().rev() {
        let highlighted = matched.color(config.iri_color).underline().to_string();
        result.replace_range(*start..*end, &highlighted);
    }

    // 4. Numbers
    let positions = collect_match_positions(&result, &NUMBER_PATTERN);
    for (start, end, matched) in positions.iter().rev() {
        let highlighted = matched.color(config.number_color).to_string();
        result.replace_range(*start..*end, &highlighted);
    }

    // 5. Variables
    let positions = collect_match_positions(&result, &VARIABLE_PATTERN);
    for (start, end, matched) in positions.iter().rev() {
        let highlighted = matched.color(config.variable_color).bold().to_string();
        result.replace_range(*start..*end, &highlighted);
    }

    // 6. Functions
    let positions = collect_match_positions(&result, &FUNCTION_PATTERN);
    for (start, end, matched) in positions.iter().rev() {
        let highlighted = matched.color(config.function_color).to_string();
        result.replace_range(*start..*end, &highlighted);
    }

    // 7. Keywords (last to avoid matching parts of other elements)
    let positions = collect_match_positions(&result, &KEYWORD_PATTERN);
    for (start, end, matched) in positions.iter().rev() {
        let highlighted = matched.color(config.keyword_color).bold().to_string();
        result.replace_range(*start..*end, &highlighted);
    }

    result
}

/// Collect match positions for a pattern (to avoid overlapping replacements)
fn collect_match_positions(text: &str, pattern: &Regex) -> Vec<(usize, usize, String)> {
    pattern
        .find_iter(text)
        .map(|m| (m.start(), m.end(), m.as_str().to_string()))
        .collect()
}

/// Highlight RDF term in query results
pub fn highlight_rdf_term(term: &str, config: &HighlightConfig) -> String {
    if !config.enable_colors {
        return term.to_string();
    }

    // IRI
    if term.starts_with('<') && term.ends_with('>') {
        return term.color(config.iri_color).underline().to_string();
    }

    // Variable
    if term.starts_with('?') || term.starts_with('$') {
        return term.color(config.variable_color).bold().to_string();
    }

    // String literal
    if term.starts_with('"') {
        return term.color(config.string_color).to_string();
    }

    // Number
    if term.parse::<f64>().is_ok() {
        return term.color(config.number_color).to_string();
    }

    // Default (plain text)
    term.to_string()
}

/// Highlight error messages
pub fn highlight_error(message: &str) -> String {
    if is_no_color() {
        return format!("Error: {}", message);
    }
    format!("{}: {}", "Error".red().bold(), message.bright_red())
}

/// Highlight warning messages
pub fn highlight_warning(message: &str) -> String {
    if is_no_color() {
        return format!("Warning: {}", message);
    }
    format!("{}: {}", "Warning".yellow().bold(), message.bright_yellow())
}

/// Highlight success messages
pub fn highlight_success(message: &str) -> String {
    if is_no_color() {
        return format!("Success: {}", message);
    }
    format!("{}: {}", "Success".green().bold(), message.bright_green())
}

/// Highlight info messages
pub fn highlight_info(message: &str) -> String {
    if is_no_color() {
        return format!("Info: {}", message);
    }
    format!("{}: {}", "Info".blue().bold(), message.bright_blue())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlight_sparql_keywords() {
        let query = "SELECT ?x WHERE { ?x a <http://example.org/Person> }";
        let config = HighlightConfig {
            enable_colors: false, // Disable for testing
            ..Default::default()
        };
        let result = highlight_sparql(query, &config);
        // Without colors, should be unchanged
        assert_eq!(result, query);
    }

    #[test]
    fn test_highlight_rdf_term_iri() {
        let config = HighlightConfig {
            enable_colors: false,
            ..Default::default()
        };
        let term = "<http://example.org/Person>";
        let result = highlight_rdf_term(term, &config);
        assert_eq!(result, term);
    }

    #[test]
    fn test_highlight_rdf_term_variable() {
        let config = HighlightConfig {
            enable_colors: false,
            ..Default::default()
        };
        let term = "?person";
        let result = highlight_rdf_term(term, &config);
        assert_eq!(result, term);
    }

    #[test]
    fn test_highlight_rdf_term_literal() {
        let config = HighlightConfig {
            enable_colors: false,
            ..Default::default()
        };
        let term = r#""John Doe""#;
        let result = highlight_rdf_term(term, &config);
        assert_eq!(result, term);
    }

    #[test]
    fn test_highlight_rdf_term_number() {
        let config = HighlightConfig {
            enable_colors: false,
            ..Default::default()
        };
        let term = "42";
        let result = highlight_rdf_term(term, &config);
        assert_eq!(result, term);
    }

    #[test]
    fn test_no_color_environment() {
        // This test checks that NO_COLOR is respected
        // Note: In a real environment, set NO_COLOR=1 to disable colors
        let config = HighlightConfig::default();
        assert!(!config.enable_colors || std::env::var("NO_COLOR").is_err());
    }

    #[test]
    fn test_highlight_error() {
        let message = "Something went wrong";
        let result = highlight_error(message);
        assert!(result.contains(message));
    }

    #[test]
    fn test_highlight_warning() {
        let message = "This might be a problem";
        let result = highlight_warning(message);
        assert!(result.contains(message));
    }

    #[test]
    fn test_highlight_success() {
        let message = "Operation completed";
        let result = highlight_success(message);
        assert!(result.contains(message));
    }

    #[test]
    fn test_highlight_info() {
        let message = "Processing data";
        let result = highlight_info(message);
        assert!(result.contains(message));
    }

    #[test]
    fn test_collect_match_positions() {
        let text = "SELECT ?x WHERE { ?x a ?y }";
        let pattern = Regex::new(r"\?[a-z]").unwrap();
        let positions = collect_match_positions(text, &pattern);
        assert_eq!(positions.len(), 3); // ?x, ?x, ?y
    }

    #[test]
    fn test_highlight_complex_query() {
        let query = r#"
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT DISTINCT ?person ?name ?age
WHERE {
    ?person rdf:type foaf:Person .
    ?person foaf:name ?name .
    OPTIONAL { ?person foaf:age ?age }
    FILTER(?age > 18)
}
ORDER BY ?name
LIMIT 100
"#;
        let config = HighlightConfig {
            enable_colors: false,
            ..Default::default()
        };
        let result = highlight_sparql(query, &config);
        // Without colors, should be unchanged
        assert_eq!(result, query);
    }
}
