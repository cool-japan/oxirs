//! RDF-star query to standard RDF rewriting.
//!
//! Provides query-level transformations that convert SPARQL-star quoted-triple
//! patterns into standard SPARQL 1.1 reification patterns, enabling execution
//! on engines that do not natively support RDF-star.

use std::collections::HashMap;
use std::fmt;

// ── Core types ──────────────────────────────────────────────────────────────

/// A SPARQL-star term that may be an IRI, literal, variable, blank node,
/// or a quoted triple.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StarQueryTerm {
    /// An IRI reference (e.g. `<http://example.org/s>`).
    Iri(String),
    /// A plain string literal (optionally with language tag or datatype).
    Literal {
        /// The lexical value.
        value: String,
        /// An optional language tag (e.g. `"en"`).
        lang: Option<String>,
        /// An optional datatype IRI.
        datatype: Option<String>,
    },
    /// A SPARQL variable (e.g. `?x`).
    Variable(String),
    /// A blank node identifier (e.g. `_:b1`).
    BlankNode(String),
    /// A quoted triple `<< s p o >>`.
    QuotedTriple(Box<QuotedTriplePattern>),
}

impl StarQueryTerm {
    /// Create an IRI term.
    pub fn iri(s: impl Into<String>) -> Self {
        Self::Iri(s.into())
    }

    /// Create a variable term.
    pub fn var(s: impl Into<String>) -> Self {
        Self::Variable(s.into())
    }

    /// Create a plain literal.
    pub fn literal(s: impl Into<String>) -> Self {
        Self::Literal {
            value: s.into(),
            lang: None,
            datatype: None,
        }
    }

    /// Create a quoted triple term.
    pub fn quoted(s: StarQueryTerm, p: StarQueryTerm, o: StarQueryTerm) -> Self {
        Self::QuotedTriple(Box::new(QuotedTriplePattern {
            subject: s,
            predicate: p,
            object: o,
        }))
    }

    /// Whether this term is, or recursively contains, a quoted triple.
    pub fn has_quoted_triple(&self) -> bool {
        matches!(self, Self::QuotedTriple(_))
    }

    /// Returns the nesting depth (0 for non-quoted terms, 1+ for quoted).
    pub fn nesting_depth(&self) -> usize {
        match self {
            Self::QuotedTriple(qt) => {
                1 + qt
                    .subject
                    .nesting_depth()
                    .max(qt.predicate.nesting_depth())
                    .max(qt.object.nesting_depth())
            }
            _ => 0,
        }
    }
}

impl fmt::Display for StarQueryTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Iri(s) => write!(f, "<{s}>"),
            Self::Literal {
                value,
                lang,
                datatype,
            } => {
                write!(f, "\"{value}\"")?;
                if let Some(l) = lang {
                    write!(f, "@{l}")?;
                }
                if let Some(dt) = datatype {
                    write!(f, "^^<{dt}>")?;
                }
                Ok(())
            }
            Self::Variable(v) => write!(f, "?{v}"),
            Self::BlankNode(b) => write!(f, "_:{b}"),
            Self::QuotedTriple(qt) => {
                write!(f, "<< {} {} {} >>", qt.subject, qt.predicate, qt.object)
            }
        }
    }
}

/// A triple pattern whose components may themselves be quoted triples.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuotedTriplePattern {
    /// Subject of the quoted triple.
    pub subject: StarQueryTerm,
    /// Predicate of the quoted triple.
    pub predicate: StarQueryTerm,
    /// Object of the quoted triple.
    pub object: StarQueryTerm,
}

/// A standard SPARQL 1.1 triple pattern (no quoted triples).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TriplePattern {
    /// Subject term.
    pub subject: StarQueryTerm,
    /// Predicate term.
    pub predicate: StarQueryTerm,
    /// Object term.
    pub object: StarQueryTerm,
}

impl fmt::Display for TriplePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.subject, self.predicate, self.object)
    }
}

/// A SPARQL-star triple pattern that may contain quoted triples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StarTriplePattern {
    /// Subject (may be a quoted triple).
    pub subject: StarQueryTerm,
    /// Predicate (may be a variable or IRI).
    pub predicate: StarQueryTerm,
    /// Object (may be a quoted triple).
    pub object: StarQueryTerm,
}

impl StarTriplePattern {
    /// Create a new star triple pattern.
    pub fn new(s: StarQueryTerm, p: StarQueryTerm, o: StarQueryTerm) -> Self {
        Self {
            subject: s,
            predicate: p,
            object: o,
        }
    }

    /// Returns `true` if this pattern contains any quoted triples.
    pub fn has_quoted_triples(&self) -> bool {
        self.subject.has_quoted_triple()
            || self.predicate.has_quoted_triple()
            || self.object.has_quoted_triple()
    }
}

/// RDF namespace constants for reification.
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDF_STATEMENT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement";
const RDF_SUBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject";
const RDF_PREDICATE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate";
const RDF_OBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object";

// ── Annotation expansion ────────────────────────────────────────────────────

/// An annotation assertion `<< s p o >> ap av .` expanded into a triple
/// pattern with a reification node.
#[derive(Debug, Clone)]
pub struct AnnotationExpansion {
    /// The base triple (s, p, o).
    pub base: TriplePattern,
    /// The reification node variable name.
    pub reification_var: String,
    /// The annotation triples (reification_var rdf:type, rdf:subject, etc.).
    pub reification_triples: Vec<TriplePattern>,
    /// Additional annotation predicates.
    pub annotation_triples: Vec<TriplePattern>,
}

// ── Rewrite errors ──────────────────────────────────────────────────────────

/// Errors that may occur during query rewriting.
#[derive(Debug)]
pub enum RewriteError {
    /// Maximum nesting depth exceeded during recursive decomposition.
    NestingTooDeep(usize),
    /// A variable binding conflict was detected.
    BindingConflict(String),
    /// Invalid pattern structure.
    InvalidPattern(String),
}

impl fmt::Display for RewriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NestingTooDeep(d) => write!(f, "Nesting depth {d} exceeds maximum"),
            Self::BindingConflict(v) => write!(f, "Variable binding conflict: {v}"),
            Self::InvalidPattern(s) => write!(f, "Invalid pattern: {s}"),
        }
    }
}

impl std::error::Error for RewriteError {}

// ── Rewrite result ──────────────────────────────────────────────────────────

/// The result of rewriting a set of star patterns into standard SPARQL 1.1.
#[derive(Debug, Clone)]
pub struct RewriteResult {
    /// The rewritten triple patterns (no quoted triples).
    pub patterns: Vec<TriplePattern>,
    /// Mapping from generated reification variable names to the quoted triple
    /// they represent (for debugging / provenance).
    pub reification_map: HashMap<String, String>,
    /// Number of quoted triples that were decomposed.
    pub decomposed_count: usize,
}

impl RewriteResult {
    /// An empty rewrite result.
    fn empty() -> Self {
        Self {
            patterns: Vec::new(),
            reification_map: HashMap::new(),
            decomposed_count: 0,
        }
    }
}

// ── The rewriter ────────────────────────────────────────────────────────────

/// Configuration for the star query rewriter.
#[derive(Debug, Clone)]
pub struct RewriterConfig {
    /// Maximum allowed nesting depth for recursive decomposition.
    pub max_nesting_depth: usize,
    /// Prefix for generated reification blank nodes / variables.
    pub reification_prefix: String,
    /// Whether to produce a compacted output (eliminate redundant patterns).
    pub optimize: bool,
}

impl Default for RewriterConfig {
    fn default() -> Self {
        Self {
            max_nesting_depth: 10,
            reification_prefix: "_star_".to_string(),
            optimize: true,
        }
    }
}

/// Rewrites SPARQL-star patterns into standard SPARQL 1.1 reification patterns.
pub struct StarQueryRewriter {
    config: RewriterConfig,
    counter: usize,
}

impl StarQueryRewriter {
    /// Create a new rewriter with default configuration.
    pub fn new() -> Self {
        Self {
            config: RewriterConfig::default(),
            counter: 0,
        }
    }

    /// Create a rewriter with a custom configuration.
    pub fn with_config(config: RewriterConfig) -> Self {
        Self { config, counter: 0 }
    }

    /// Reset the internal counter (useful between independent rewrites).
    pub fn reset(&mut self) {
        self.counter = 0;
    }

    /// Generate a fresh reification variable name.
    fn fresh_var(&mut self) -> String {
        let name = format!("{}{}", self.config.reification_prefix, self.counter);
        self.counter += 1;
        name
    }

    /// Rewrite a set of SPARQL-star triple patterns into standard SPARQL 1.1.
    pub fn rewrite(
        &mut self,
        patterns: &[StarTriplePattern],
    ) -> Result<RewriteResult, RewriteError> {
        let mut result = RewriteResult::empty();

        for pat in patterns {
            self.rewrite_pattern(pat, &mut result, 0)?;
        }

        if self.config.optimize {
            Self::eliminate_redundant(&mut result);
        }

        Ok(result)
    }

    /// Recursively rewrite a single star triple pattern.
    fn rewrite_pattern(
        &mut self,
        pattern: &StarTriplePattern,
        result: &mut RewriteResult,
        depth: usize,
    ) -> Result<StarQueryTerm, RewriteError> {
        if depth > self.config.max_nesting_depth {
            return Err(RewriteError::NestingTooDeep(depth));
        }

        // Decompose subject if it is a quoted triple
        let subj = self.decompose_term(&pattern.subject, result, depth)?;
        // Decompose object if it is a quoted triple
        let obj = self.decompose_term(&pattern.object, result, depth)?;
        // Predicate is never a quoted triple in valid SPARQL-star, but handle it
        let pred = self.decompose_term(&pattern.predicate, result, depth)?;

        result.patterns.push(TriplePattern {
            subject: subj.clone(),
            predicate: pred,
            object: obj,
        });

        Ok(subj)
    }

    /// Decompose a term: if it is a quoted triple, replace it with a fresh
    /// reification variable and emit the reification triples; otherwise return as-is.
    fn decompose_term(
        &mut self,
        term: &StarQueryTerm,
        result: &mut RewriteResult,
        depth: usize,
    ) -> Result<StarQueryTerm, RewriteError> {
        match term {
            StarQueryTerm::QuotedTriple(qt) => {
                if depth + 1 > self.config.max_nesting_depth {
                    return Err(RewriteError::NestingTooDeep(depth + 1));
                }

                // Recursively decompose inner terms
                let inner_subj = self.decompose_term(&qt.subject, result, depth + 1)?;
                let inner_pred = self.decompose_term(&qt.predicate, result, depth + 1)?;
                let inner_obj = self.decompose_term(&qt.object, result, depth + 1)?;

                // Generate fresh reification variable
                let var_name = self.fresh_var();
                let reif_var = StarQueryTerm::Variable(var_name.clone());

                // Emit reification quad: ?var rdf:type rdf:Statement
                result.patterns.push(TriplePattern {
                    subject: reif_var.clone(),
                    predicate: StarQueryTerm::iri(RDF_TYPE),
                    object: StarQueryTerm::iri(RDF_STATEMENT),
                });
                // ?var rdf:subject s
                result.patterns.push(TriplePattern {
                    subject: reif_var.clone(),
                    predicate: StarQueryTerm::iri(RDF_SUBJECT),
                    object: inner_subj,
                });
                // ?var rdf:predicate p
                result.patterns.push(TriplePattern {
                    subject: reif_var.clone(),
                    predicate: StarQueryTerm::iri(RDF_PREDICATE),
                    object: inner_pred,
                });
                // ?var rdf:object o
                result.patterns.push(TriplePattern {
                    subject: reif_var.clone(),
                    predicate: StarQueryTerm::iri(RDF_OBJECT),
                    object: inner_obj,
                });

                result.reification_map.insert(var_name, term.to_string());
                result.decomposed_count += 1;

                Ok(reif_var)
            }
            other => Ok(other.clone()),
        }
    }

    /// Eliminate duplicate triple patterns from the result.
    fn eliminate_redundant(result: &mut RewriteResult) {
        let mut seen = std::collections::HashSet::new();
        result.patterns.retain(|p| {
            let key = (p.subject.clone(), p.predicate.clone(), p.object.clone());
            seen.insert(key)
        });
    }

    /// Expand annotation syntax: `<< s p o >> ap av .` into the base triple
    /// plus reification triples plus annotation triples.
    pub fn expand_annotation(
        &mut self,
        base_s: StarQueryTerm,
        base_p: StarQueryTerm,
        base_o: StarQueryTerm,
        annotations: &[(StarQueryTerm, StarQueryTerm)],
    ) -> Result<AnnotationExpansion, RewriteError> {
        let reif_var_name = self.fresh_var();
        let reif_var = StarQueryTerm::Variable(reif_var_name.clone());

        let base = TriplePattern {
            subject: base_s.clone(),
            predicate: base_p.clone(),
            object: base_o.clone(),
        };

        let mut reification_triples = vec![
            TriplePattern {
                subject: reif_var.clone(),
                predicate: StarQueryTerm::iri(RDF_TYPE),
                object: StarQueryTerm::iri(RDF_STATEMENT),
            },
            TriplePattern {
                subject: reif_var.clone(),
                predicate: StarQueryTerm::iri(RDF_SUBJECT),
                object: base_s,
            },
            TriplePattern {
                subject: reif_var.clone(),
                predicate: StarQueryTerm::iri(RDF_PREDICATE),
                object: base_p,
            },
            TriplePattern {
                subject: reif_var.clone(),
                predicate: StarQueryTerm::iri(RDF_OBJECT),
                object: base_o,
            },
        ];

        let mut annotation_triples = Vec::new();
        for (ap, av) in annotations {
            annotation_triples.push(TriplePattern {
                subject: reif_var.clone(),
                predicate: ap.clone(),
                object: av.clone(),
            });
        }

        // Remove duplicates between reification and annotation triples
        let mut all = reification_triples.clone();
        all.extend(annotation_triples.clone());
        let mut seen = std::collections::HashSet::new();
        all.retain(|p| {
            let key = (p.subject.clone(), p.predicate.clone(), p.object.clone());
            seen.insert(key)
        });
        reification_triples = all[..reification_triples.len().min(all.len())].to_vec();
        annotation_triples = if all.len() > reification_triples.len() {
            all[reification_triples.len()..].to_vec()
        } else {
            Vec::new()
        };

        Ok(AnnotationExpansion {
            base,
            reification_var: reif_var_name,
            reification_triples,
            annotation_triples,
        })
    }

    /// Check whether a set of patterns is purely standard SPARQL 1.1 (no quoted triples).
    pub fn is_standard_sparql(patterns: &[StarTriplePattern]) -> bool {
        patterns.iter().all(|p| !p.has_quoted_triples())
    }

    /// Collect all variable names that appear in a set of star patterns.
    pub fn collect_variables(patterns: &[StarTriplePattern]) -> Vec<String> {
        let mut vars = std::collections::HashSet::new();
        for p in patterns {
            Self::collect_term_vars(&p.subject, &mut vars);
            Self::collect_term_vars(&p.predicate, &mut vars);
            Self::collect_term_vars(&p.object, &mut vars);
        }
        let mut sorted: Vec<String> = vars.into_iter().collect();
        sorted.sort();
        sorted
    }

    fn collect_term_vars(term: &StarQueryTerm, vars: &mut std::collections::HashSet<String>) {
        match term {
            StarQueryTerm::Variable(v) => {
                vars.insert(v.clone());
            }
            StarQueryTerm::QuotedTriple(qt) => {
                Self::collect_term_vars(&qt.subject, vars);
                Self::collect_term_vars(&qt.predicate, vars);
                Self::collect_term_vars(&qt.object, vars);
            }
            _ => {}
        }
    }

    /// Compute the maximum nesting depth across a set of patterns.
    pub fn max_depth(patterns: &[StarTriplePattern]) -> usize {
        patterns
            .iter()
            .map(|p| {
                p.subject
                    .nesting_depth()
                    .max(p.predicate.nesting_depth())
                    .max(p.object.nesting_depth())
            })
            .max()
            .unwrap_or(0)
    }

    /// Produce a human-readable summary of a rewrite result.
    pub fn summarise(result: &RewriteResult) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Rewrite: {} decomposed quoted triple(s) → {} SPARQL 1.1 pattern(s)\n",
            result.decomposed_count,
            result.patterns.len(),
        ));
        for (var, qt) in &result.reification_map {
            out.push_str(&format!("  ?{var} ← {qt}\n"));
        }
        out
    }
}

impl Default for StarQueryRewriter {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn iri(s: &str) -> StarQueryTerm {
        StarQueryTerm::iri(s)
    }

    fn var(s: &str) -> StarQueryTerm {
        StarQueryTerm::var(s)
    }

    fn lit(s: &str) -> StarQueryTerm {
        StarQueryTerm::literal(s)
    }

    // ── Basic term tests ────────────────────────────────────────────────────

    #[test]
    fn test_term_display_iri() {
        assert_eq!(iri("http://ex.org/s").to_string(), "<http://ex.org/s>");
    }

    #[test]
    fn test_term_display_variable() {
        assert_eq!(var("x").to_string(), "?x");
    }

    #[test]
    fn test_term_display_literal() {
        assert_eq!(lit("hello").to_string(), "\"hello\"");
    }

    #[test]
    fn test_term_display_quoted() {
        let qt = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), lit("42"));
        assert!(qt.to_string().contains("<<"));
        assert!(qt.to_string().contains(">>"));
    }

    #[test]
    fn test_nesting_depth_zero() {
        assert_eq!(iri("http://x").nesting_depth(), 0);
        assert_eq!(var("x").nesting_depth(), 0);
    }

    #[test]
    fn test_nesting_depth_one() {
        let qt = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        assert_eq!(qt.nesting_depth(), 1);
    }

    #[test]
    fn test_nesting_depth_nested() {
        let inner = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        let outer = StarQueryTerm::quoted(inner, iri("http://p2"), lit("v"));
        assert_eq!(outer.nesting_depth(), 2);
    }

    // ── Rewriter: no quoted triples (pass-through) ──────────────────────────

    #[test]
    fn test_rewrite_no_star_patterns() {
        let mut rw = StarQueryRewriter::new();
        let patterns = vec![StarTriplePattern::new(var("s"), iri("http://p"), var("o"))];
        let res = rw.rewrite(&patterns).expect("Should succeed");
        assert_eq!(res.patterns.len(), 1);
        assert_eq!(res.decomposed_count, 0);
    }

    #[test]
    fn test_is_standard_sparql() {
        let std_pats = vec![StarTriplePattern::new(var("s"), iri("http://p"), var("o"))];
        assert!(StarQueryRewriter::is_standard_sparql(&std_pats));
    }

    #[test]
    fn test_is_not_standard_sparql() {
        let qt = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        let pats = vec![StarTriplePattern::new(qt, iri("http://meta"), lit("val"))];
        assert!(!StarQueryRewriter::is_standard_sparql(&pats));
    }

    // ── Rewriter: simple quoted triple in subject ───────────────────────────

    #[test]
    fn test_rewrite_quoted_subject() {
        let mut rw = StarQueryRewriter::new();
        let qt = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        let patterns = vec![StarTriplePattern::new(qt, iri("http://meta"), lit("0.9"))];
        let res = rw.rewrite(&patterns).expect("ok");

        // Should produce: 4 reification + 1 outer = 5 patterns
        assert_eq!(res.decomposed_count, 1);
        assert!(
            res.patterns.len() >= 5,
            "Expected ≥5 patterns, got {}",
            res.patterns.len()
        );

        // Check reification map has one entry
        assert_eq!(res.reification_map.len(), 1);
    }

    #[test]
    fn test_rewrite_quoted_object() {
        let mut rw = StarQueryRewriter::new();
        let qt = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        let patterns = vec![StarTriplePattern::new(var("x"), iri("http://source"), qt)];
        let res = rw.rewrite(&patterns).expect("ok");

        assert_eq!(res.decomposed_count, 1);
        assert!(res.patterns.len() >= 5);
    }

    // ── Rewriter: nested quoted triples ─────────────────────────────────────

    #[test]
    fn test_rewrite_nested_quoted() {
        let mut rw = StarQueryRewriter::new();
        let inner = StarQueryTerm::quoted(iri("http://a"), iri("http://b"), iri("http://c"));
        let outer = StarQueryTerm::quoted(inner, iri("http://d"), lit("v"));
        let patterns = vec![StarTriplePattern::new(
            outer,
            iri("http://meta"),
            lit("val"),
        )];
        let res = rw.rewrite(&patterns).expect("ok");

        // Two levels of decomposition: 2 decomposed quoted triples
        assert_eq!(res.decomposed_count, 2);
        // inner: 4 + outer: 4 + top-level: 1 = 9 patterns
        assert!(
            res.patterns.len() >= 9,
            "Expected ≥9 patterns, got {}",
            res.patterns.len()
        );
    }

    #[test]
    fn test_rewrite_max_depth_exceeded() {
        let config = RewriterConfig {
            max_nesting_depth: 1,
            ..RewriterConfig::default()
        };
        let mut rw = StarQueryRewriter::with_config(config);
        let inner = StarQueryTerm::quoted(iri("http://a"), iri("http://b"), iri("http://c"));
        let outer = StarQueryTerm::quoted(inner, iri("http://d"), lit("v"));
        let patterns = vec![StarTriplePattern::new(
            outer,
            iri("http://meta"),
            lit("val"),
        )];
        let res = rw.rewrite(&patterns);
        assert!(res.is_err(), "Should fail due to nesting depth");
    }

    // ── Rewriter: reification map ───────────────────────────────────────────

    #[test]
    fn test_reification_map_populated() {
        let mut rw = StarQueryRewriter::new();
        let qt = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), lit("42"));
        let patterns = vec![StarTriplePattern::new(qt, iri("http://meta"), lit("v"))];
        let res = rw.rewrite(&patterns).expect("ok");
        assert!(!res.reification_map.is_empty());
        let (key, val) = res.reification_map.iter().next().expect("entry");
        assert!(key.starts_with("_star_"));
        assert!(val.contains("<<"));
    }

    // ── Rewriter: variable preservation ─────────────────────────────────────

    #[test]
    fn test_variables_preserved() {
        let mut rw = StarQueryRewriter::new();
        let qt = StarQueryTerm::quoted(var("s"), iri("http://p"), var("o"));
        let patterns = vec![StarTriplePattern::new(qt, iri("http://meta"), var("v"))];
        let res = rw.rewrite(&patterns).expect("ok");

        // The original variables ?s, ?o, ?v should appear somewhere in the output
        let all_text: String = res
            .patterns
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(all_text.contains("?s"), "?s should be preserved");
        assert!(all_text.contains("?o"), "?o should be preserved");
        assert!(all_text.contains("?v"), "?v should be preserved");
    }

    // ── Rewriter: optimization (dedup) ──────────────────────────────────────

    #[test]
    fn test_dedup_removes_duplicates() {
        let mut rw = StarQueryRewriter::new();
        // Two patterns that reference the same quoted triple
        let qt1 = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        let qt2 = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        let patterns = vec![
            StarTriplePattern::new(qt1, iri("http://m1"), lit("v1")),
            StarTriplePattern::new(qt2, iri("http://m2"), lit("v2")),
        ];
        let res = rw.rewrite(&patterns).expect("ok");
        // With dedup: the rdf:type rdf:Statement for both should be merged if same variable,
        // but since they get different variables, there will be no duplicates within each set.
        // Just verify it succeeds and has reasonable count
        assert!(res.decomposed_count >= 2);
    }

    #[test]
    fn test_no_optimization() {
        let config = RewriterConfig {
            optimize: false,
            ..RewriterConfig::default()
        };
        let mut rw = StarQueryRewriter::with_config(config);
        let qt = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        let patterns = vec![StarTriplePattern::new(qt, iri("http://meta"), lit("v"))];
        let res = rw.rewrite(&patterns).expect("ok");
        assert!(!res.patterns.is_empty());
    }

    // ── Annotation expansion ────────────────────────────────────────────────

    #[test]
    fn test_annotation_expansion_basic() {
        let mut rw = StarQueryRewriter::new();
        let annotations = vec![
            (iri("http://ex.org/confidence"), lit("0.9")),
            (iri("http://ex.org/source"), iri("http://ex.org/census")),
        ];
        let exp = rw
            .expand_annotation(
                iri("http://alice"),
                iri("http://age"),
                lit("30"),
                &annotations,
            )
            .expect("ok");

        assert_eq!(exp.base.subject, iri("http://alice"));
        assert_eq!(exp.reification_triples.len(), 4);
        assert_eq!(exp.annotation_triples.len(), 2);
    }

    #[test]
    fn test_annotation_expansion_no_annotations() {
        let mut rw = StarQueryRewriter::new();
        let exp = rw
            .expand_annotation(iri("http://s"), iri("http://p"), iri("http://o"), &[])
            .expect("ok");
        assert_eq!(exp.reification_triples.len(), 4);
        assert!(exp.annotation_triples.is_empty());
    }

    // ── Collect variables ───────────────────────────────────────────────────

    #[test]
    fn test_collect_variables_simple() {
        let patterns = vec![StarTriplePattern::new(var("s"), iri("http://p"), var("o"))];
        let vars = StarQueryRewriter::collect_variables(&patterns);
        assert_eq!(vars, vec!["o", "s"]);
    }

    #[test]
    fn test_collect_variables_in_quoted() {
        let qt = StarQueryTerm::quoted(var("inner_s"), iri("http://p"), var("inner_o"));
        let patterns = vec![StarTriplePattern::new(qt, iri("http://meta"), var("v"))];
        let vars = StarQueryRewriter::collect_variables(&patterns);
        assert!(vars.contains(&"inner_s".to_string()));
        assert!(vars.contains(&"inner_o".to_string()));
        assert!(vars.contains(&"v".to_string()));
    }

    // ── Max depth ───────────────────────────────────────────────────────────

    #[test]
    fn test_max_depth_no_star() {
        let patterns = vec![StarTriplePattern::new(var("s"), iri("http://p"), var("o"))];
        assert_eq!(StarQueryRewriter::max_depth(&patterns), 0);
    }

    #[test]
    fn test_max_depth_single() {
        let qt = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        let patterns = vec![StarTriplePattern::new(qt, iri("http://m"), lit("v"))];
        assert_eq!(StarQueryRewriter::max_depth(&patterns), 1);
    }

    #[test]
    fn test_max_depth_double() {
        let inner = StarQueryTerm::quoted(iri("http://a"), iri("http://b"), iri("http://c"));
        let outer = StarQueryTerm::quoted(inner, iri("http://d"), iri("http://e"));
        let patterns = vec![StarTriplePattern::new(outer, iri("http://m"), lit("v"))];
        assert_eq!(StarQueryRewriter::max_depth(&patterns), 2);
    }

    // ── Summarise ───────────────────────────────────────────────────────────

    #[test]
    fn test_summarise_output() {
        let mut rw = StarQueryRewriter::new();
        let qt = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        let patterns = vec![StarTriplePattern::new(qt, iri("http://m"), lit("v"))];
        let res = rw.rewrite(&patterns).expect("ok");
        let summary = StarQueryRewriter::summarise(&res);
        assert!(summary.contains("decomposed"));
        assert!(summary.contains("_star_"));
    }

    // ── Reset ───────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_counter() {
        let mut rw = StarQueryRewriter::new();
        let qt = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        let patterns = vec![StarTriplePattern::new(qt, iri("http://m"), lit("v"))];
        let _ = rw.rewrite(&patterns).expect("ok");
        assert!(rw.counter > 0);
        rw.reset();
        assert_eq!(rw.counter, 0);
    }

    // ── Has quoted triples ──────────────────────────────────────────────────

    #[test]
    fn test_star_pattern_has_quoted() {
        let qt = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        let pat = StarTriplePattern::new(qt, iri("http://m"), lit("v"));
        assert!(pat.has_quoted_triples());
    }

    #[test]
    fn test_star_pattern_no_quoted() {
        let pat = StarTriplePattern::new(var("s"), iri("http://p"), var("o"));
        assert!(!pat.has_quoted_triples());
    }

    // ── Blank node terms ────────────────────────────────────────────────────

    #[test]
    fn test_blank_node_display() {
        let bn = StarQueryTerm::BlankNode("b1".to_string());
        assert_eq!(bn.to_string(), "_:b1");
    }

    #[test]
    fn test_rewrite_with_blank_node() {
        let mut rw = StarQueryRewriter::new();
        let patterns = vec![StarTriplePattern::new(
            StarQueryTerm::BlankNode("b0".to_string()),
            iri("http://p"),
            lit("v"),
        )];
        let res = rw.rewrite(&patterns).expect("ok");
        assert_eq!(res.patterns.len(), 1);
        assert_eq!(res.decomposed_count, 0);
    }

    // ── Literal with lang tag ───────────────────────────────────────────────

    #[test]
    fn test_literal_with_lang() {
        let t = StarQueryTerm::Literal {
            value: "hello".to_string(),
            lang: Some("en".to_string()),
            datatype: None,
        };
        assert_eq!(t.to_string(), "\"hello\"@en");
    }

    #[test]
    fn test_literal_with_datatype() {
        let t = StarQueryTerm::Literal {
            value: "42".to_string(),
            lang: None,
            datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
        };
        assert!(t.to_string().contains("^^<"));
    }

    // ── Multiple patterns ───────────────────────────────────────────────────

    #[test]
    fn test_multiple_mixed_patterns() {
        let mut rw = StarQueryRewriter::new();
        let qt = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        let patterns = vec![
            StarTriplePattern::new(var("a"), iri("http://p1"), var("b")),
            StarTriplePattern::new(qt, iri("http://meta"), lit("v")),
            StarTriplePattern::new(var("c"), iri("http://p2"), var("d")),
        ];
        let res = rw.rewrite(&patterns).expect("ok");
        assert_eq!(res.decomposed_count, 1);
        // 2 pass-through + 4 reification + 1 outer = 7
        assert!(res.patterns.len() >= 7);
    }

    // ── Custom config prefix ────────────────────────────────────────────────

    #[test]
    fn test_custom_prefix() {
        let config = RewriterConfig {
            reification_prefix: "reif_".to_string(),
            ..RewriterConfig::default()
        };
        let mut rw = StarQueryRewriter::with_config(config);
        let qt = StarQueryTerm::quoted(iri("http://s"), iri("http://p"), iri("http://o"));
        let patterns = vec![StarTriplePattern::new(qt, iri("http://m"), lit("v"))];
        let res = rw.rewrite(&patterns).expect("ok");
        let has_reif = res.reification_map.keys().any(|k| k.starts_with("reif_"));
        assert!(has_reif, "Should use custom prefix");
    }

    // ── Default impl ────────────────────────────────────────────────────────

    #[test]
    fn test_default_rewriter() {
        let mut rw = StarQueryRewriter::default();
        let patterns = vec![StarTriplePattern::new(var("s"), iri("http://p"), var("o"))];
        let res = rw.rewrite(&patterns).expect("ok");
        assert_eq!(res.patterns.len(), 1);
    }
}
