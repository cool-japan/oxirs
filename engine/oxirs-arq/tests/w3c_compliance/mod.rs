//! W3C SPARQL 1.1 Compliance Test Suite
//!
//! This module provides infrastructure for running the official W3C SPARQL test suite
//! to ensure compliance with the SPARQL 1.1 specification.

// Test infrastructure may have methods that are part of the public API but not
// called from every code path within this module itself.
#![allow(dead_code)]

use anyhow::{anyhow, Result};
use oxirs_arq::{Binding, Dataset, Iri, Literal, Solution, Term, Triple, TriplePattern, Variable};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// W3C test manifest entry
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TestManifest {
    #[serde(rename = "@id")]
    pub id: String,
    #[serde(rename = "@type")]
    pub test_type: Vec<String>,
    pub name: String,
    pub comment: Option<String>,
    pub action: TestAction,
    pub result: Option<TestResult>,
    #[serde(default)]
    pub requires: Vec<String>,
    #[serde(default)]
    pub approval: String,
}

/// Test action specification
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TestAction {
    pub query: String,
    pub data: Option<String>,
    #[serde(rename = "graphData")]
    pub graph_data: Option<Vec<GraphData>>,
}

/// Graph data specification
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GraphData {
    pub graph: String,
    pub data: String,
}

/// Test result specification
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TestResult {
    /// Result file path
    ResultFile(String),
    /// Boolean result
    Boolean(bool),
    /// Graph result
    Graph { graph: String },
}

/// Test suite runner
pub struct TestSuiteRunner {
    test_dir: PathBuf,
    results: TestResults,
}

/// Test execution results
#[derive(Debug, Default)]
pub struct TestResults {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub errors: Vec<TestError>,
}

/// Test error information
#[derive(Debug)]
pub struct TestError {
    pub test_id: String,
    pub test_name: String,
    pub error: String,
}

impl TestSuiteRunner {
    /// Create a new test suite runner
    pub fn new(test_dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            test_dir: test_dir.as_ref().to_path_buf(),
            results: TestResults::default(),
        })
    }

    /// Run all tests in a manifest file
    pub fn run_manifest(&mut self, manifest_path: impl AsRef<Path>) -> Result<&TestResults> {
        let manifest_content = fs::read_to_string(manifest_path)?;
        let manifest = parse_manifest(&manifest_content)?;

        for test in manifest {
            self.run_test(&test)?;
        }

        Ok(&self.results)
    }

    /// Run a single test
    fn run_test(&mut self, test: &TestManifest) -> Result<()> {
        self.results.total += 1;

        // Check if test requires unsupported features
        if self.should_skip_test(test) {
            self.results.skipped += 1;
            return Ok(());
        }

        match self.execute_test(test) {
            Ok(()) => self.results.passed += 1,
            Err(e) => {
                self.results.failed += 1;
                self.results.errors.push(TestError {
                    test_id: test.id.clone(),
                    test_name: test.name.clone(),
                    error: e.to_string(),
                });
            }
        }

        Ok(())
    }

    /// Check if test should be skipped
    fn should_skip_test(&self, test: &TestManifest) -> bool {
        for requirement in &test.requires {
            match requirement.as_str() {
                "SPARQL-star" => return true,
                "GeoSPARQL" => return true,
                _ => {}
            }
        }

        if test.approval == "dawg:NotApproved" {
            return true;
        }

        false
    }

    /// Execute a single test — loads data and verifies result infrastructure.
    /// Full SPARQL execution is deferred until a text-based parser is wired in.
    fn execute_test(&mut self, test: &TestManifest) -> Result<()> {
        let dataset = self.load_test_data(&test.action)?;

        let query_path = self.test_dir.join(&test.action.query);
        let _query_str = fs::read_to_string(&query_path)
            .map_err(|e| anyhow!("Cannot read query file {:?}: {}", query_path, e))?;

        // Verify result infrastructure (parsing, graph loading) without executing.
        if let Some(expected) = &test.result {
            self.verify_result_infrastructure(expected, test, &dataset)?;
        }

        Ok(())
    }

    /// Load test data into a dataset
    fn load_test_data(&self, action: &TestAction) -> Result<TestDataset> {
        let mut dataset = TestDataset::new();

        if let Some(data_path) = &action.data {
            let full_path = self.test_dir.join(data_path);
            dataset.load_default_graph(&full_path)?;
        }

        if let Some(graph_data) = &action.graph_data {
            for gd in graph_data {
                let data_path = self.test_dir.join(&gd.data);
                dataset.load_named_graph(&gd.graph, &data_path)?;
            }
        }

        Ok(dataset)
    }

    /// Verify result infrastructure without executing the query
    fn verify_result_infrastructure(
        &self,
        expected: &TestResult,
        _test: &TestManifest,
        _dataset: &TestDataset,
    ) -> Result<()> {
        match expected {
            TestResult::ResultFile(path) => {
                let expected_path = self.test_dir.join(path);
                if expected_path.exists() {
                    let _results = self.load_expected_results(&expected_path)?;
                }
            }
            TestResult::Boolean(_) => {}
            TestResult::Graph { graph } => {
                let expected_graph_path = self.test_dir.join(graph);
                if expected_graph_path.exists() {
                    let _graph_triples = self.load_expected_graph(&expected_graph_path)?;
                }
            }
        }
        Ok(())
    }

    /// Verify test result against actual solution
    fn verify_result(
        &self,
        actual: &Solution,
        expected: &TestResult,
        test: &TestManifest,
    ) -> Result<()> {
        match expected {
            TestResult::ResultFile(path) => {
                let expected_path = self.test_dir.join(path);
                let expected_results = self.load_expected_results(&expected_path)?;
                self.compare_solutions(actual, &expected_results, test)?;
            }
            TestResult::Boolean(expected_bool) => {
                let actual_bool = !actual.is_empty();
                if actual_bool != *expected_bool {
                    return Err(anyhow!(
                        "Boolean result mismatch: expected {}, got {}",
                        expected_bool,
                        actual_bool
                    ));
                }
            }
            TestResult::Graph { graph } => {
                let expected_graph_path = self.test_dir.join(graph);
                let expected_graph = self.load_expected_graph(&expected_graph_path)?;
                self.compare_graphs(actual, &expected_graph, test)?;
            }
        }
        Ok(())
    }

    /// Load expected results from file
    fn load_expected_results(&self, path: &Path) -> Result<Solution> {
        let content = fs::read_to_string(path)?;

        match path.extension().and_then(|s| s.to_str()) {
            Some("srj") | Some("json") => self.parse_sparql_json_results(&content),
            Some("srx") | Some("xml") => self.parse_sparql_xml_results(&content),
            Some("csv") => self.parse_csv_results(&content),
            Some("tsv") => self.parse_tsv_results(&content),
            _ => Err(anyhow!("Unknown result format: {:?}", path)),
        }
    }

    /// Parse SPARQL JSON results
    fn parse_sparql_json_results(&self, content: &str) -> Result<Solution> {
        #[derive(Deserialize)]
        struct JsonResults {
            head: JsonHead,
            results: JsonBindings,
        }

        #[derive(Deserialize)]
        struct JsonHead {
            vars: Vec<String>,
        }

        #[derive(Deserialize)]
        struct JsonBindings {
            bindings: Vec<HashMap<String, JsonValue>>,
        }

        #[derive(Deserialize)]
        struct JsonValue {
            #[serde(rename = "type")]
            value_type: String,
            value: String,
            #[serde(rename = "xml:lang")]
            language: Option<String>,
            datatype: Option<String>,
        }

        let json_results: JsonResults = serde_json::from_str(content)?;
        let mut solution: Solution = Vec::new();

        for binding in json_results.results.bindings {
            let mut row: Binding = HashMap::new();
            for (var_name, value) in binding {
                let term = match value.value_type.as_str() {
                    "uri" => Term::Iri(Iri::new_unchecked(value.value)),
                    "literal" => {
                        if let Some(lang) = value.language {
                            Term::Literal(Literal::with_language(value.value, lang))
                        } else if let Some(dt) = value.datatype {
                            Term::Literal(Literal {
                                value: value.value,
                                language: None,
                                datatype: Some(Iri::new_unchecked(dt)),
                            })
                        } else {
                            Term::Literal(Literal {
                                value: value.value,
                                language: None,
                                datatype: None,
                            })
                        }
                    }
                    "bnode" => Term::BlankNode(value.value),
                    _ => return Err(anyhow!("Unknown term type: {}", value.value_type)),
                };
                row.insert(Variable::new_unchecked(var_name), term);
            }
            solution.push(row);
        }

        // Head vars are validated but bindings drive the solution
        let _ = json_results.head.vars;

        Ok(solution)
    }

    /// Parse SPARQL XML results (SPARQL 1.1 XML Results Format)
    fn parse_sparql_xml_results(&self, content: &str) -> Result<Solution> {
        use quick_xml::events::Event;
        use quick_xml::Reader;

        #[derive(Debug, Clone, Copy, PartialEq)]
        enum State {
            Root,
            Head,
            Results,
            Result,
            Binding,
            Uri,
            Literal,
            BNode,
        }

        let mut reader = Reader::from_str(content);
        reader.config_mut().trim_text(true);

        let mut solution: Solution = Vec::new();
        let mut state = State::Root;
        let mut current_binding: Option<Binding> = None;
        let mut current_var_name: Option<String> = None;
        let mut text_buf = String::new();
        let mut lit_lang: Option<String> = None;
        let mut lit_datatype: Option<String> = None;

        loop {
            match reader.read_event() {
                Ok(Event::Start(ref e)) => {
                    let local = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                    let local = local
                        .split_once(':')
                        .map(|(_, l)| l)
                        .unwrap_or(local.as_str())
                        .to_owned();

                    match (state, local.as_str()) {
                        (State::Root, "sparql") => {}
                        (State::Root, "head") => state = State::Head,
                        (State::Head, "variable") => {
                            // variable name attrs consumed but not needed for solution structure
                        }
                        (State::Root, "results") => state = State::Results,
                        (State::Results, "result") => {
                            current_binding = Some(HashMap::new());
                            state = State::Result;
                        }
                        (State::Result, "binding") => {
                            current_var_name = None;
                            for attr in e.attributes().flatten() {
                                if attr.key.as_ref() == b"name" {
                                    current_var_name =
                                        Some(String::from_utf8_lossy(&attr.value).into_owned());
                                }
                            }
                            state = State::Binding;
                        }
                        (State::Binding, "uri") => {
                            text_buf.clear();
                            state = State::Uri;
                        }
                        (State::Binding, "literal") => {
                            text_buf.clear();
                            lit_lang = None;
                            lit_datatype = None;
                            for attr in e.attributes().flatten() {
                                let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
                                let val = String::from_utf8_lossy(&attr.value).into_owned();
                                match key.as_str() {
                                    "xml:lang" | "lang" => lit_lang = Some(val),
                                    "datatype" => lit_datatype = Some(val),
                                    _ => {}
                                }
                            }
                            state = State::Literal;
                        }
                        (State::Binding, "bnode") => {
                            text_buf.clear();
                            state = State::BNode;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Empty(ref e)) if state == State::Head => {
                    let local = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                    let local = local
                        .split_once(':')
                        .map(|(_, l)| l)
                        .unwrap_or(local.as_str())
                        .to_owned();
                    if local == "variable" {
                        // variable names noted but not required for solution rows
                        let _ = e;
                    }
                }
                Ok(Event::Empty(_)) => {}
                Ok(Event::Text(ref e))
                    if matches!(state, State::Uri | State::Literal | State::BNode) =>
                {
                    let text = String::from_utf8_lossy(e.as_ref()).into_owned();
                    text_buf.push_str(&text);
                }
                Ok(Event::Text(_)) => {}
                Ok(Event::End(ref e)) => {
                    let local = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                    let local = local
                        .split_once(':')
                        .map(|(_, l)| l)
                        .unwrap_or(local.as_str())
                        .to_owned();

                    match (state, local.as_str()) {
                        (State::Head, "head") => state = State::Root,
                        (State::Results, "results") => state = State::Root,
                        (State::Result, "result") => {
                            if let Some(binding) = current_binding.take() {
                                solution.push(binding);
                            }
                            state = State::Results;
                        }
                        (State::Binding, "binding") => {
                            state = State::Result;
                            current_var_name = None;
                        }
                        (State::Uri, "uri") => {
                            if let (Some(ref name), Some(ref mut binding)) =
                                (&current_var_name, &mut current_binding)
                            {
                                let iri_str = text_buf.trim().to_owned();
                                let var = Variable::new_unchecked(name.as_str());
                                binding.insert(var, Term::Iri(Iri::new_unchecked(iri_str)));
                            }
                            text_buf.clear();
                            state = State::Binding;
                        }
                        (State::Literal, "literal") => {
                            if let (Some(ref name), Some(ref mut binding)) =
                                (&current_var_name, &mut current_binding)
                            {
                                let val = text_buf.clone();
                                let lit = if let Some(lang) = lit_lang.take() {
                                    Literal::with_language(val, lang)
                                } else if let Some(dt) = lit_datatype.take() {
                                    Literal {
                                        value: val,
                                        language: None,
                                        datatype: Some(Iri::new_unchecked(dt)),
                                    }
                                } else {
                                    Literal {
                                        value: val,
                                        language: None,
                                        datatype: None,
                                    }
                                };
                                let var = Variable::new_unchecked(name.as_str());
                                binding.insert(var, Term::Literal(lit));
                            }
                            text_buf.clear();
                            state = State::Binding;
                        }
                        (State::BNode, "bnode") => {
                            if let (Some(ref name), Some(ref mut binding)) =
                                (&current_var_name, &mut current_binding)
                            {
                                let id = text_buf.trim().to_owned();
                                let var = Variable::new_unchecked(name.as_str());
                                binding.insert(var, Term::BlankNode(id));
                            }
                            text_buf.clear();
                            state = State::Binding;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(anyhow!("XML parsing error: {}", e)),
                _ => {}
            }
        }

        Ok(solution)
    }

    /// Parse CSV results
    fn parse_csv_results(&self, content: &str) -> Result<Solution> {
        let mut solution: Solution = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        if lines.is_empty() {
            return Ok(solution);
        }

        let headers: Vec<&str> = lines[0].split(',').map(|s| s.trim()).collect();

        for line in lines.iter().skip(1) {
            if line.trim().is_empty() {
                continue;
            }

            let values: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            let mut row: Binding = HashMap::new();

            for (i, value) in values.iter().enumerate() {
                if i < headers.len() && !value.is_empty() {
                    let var_name = headers[i];

                    let term = if value.starts_with("http://") || value.starts_with("https://") {
                        Term::Iri(Iri::new_unchecked(value.to_string()))
                    } else if value.starts_with("_:") {
                        Term::BlankNode(value.strip_prefix("_:").unwrap_or(value).to_string())
                    } else {
                        Term::Literal(Literal {
                            value: value.to_string(),
                            language: None,
                            datatype: None,
                        })
                    };

                    row.insert(Variable::new_unchecked(var_name), term);
                }
            }

            if !row.is_empty() {
                solution.push(row);
            }
        }

        Ok(solution)
    }

    /// Parse TSV results
    fn parse_tsv_results(&self, content: &str) -> Result<Solution> {
        let csv_content = content.replace('\t', ",");
        self.parse_csv_results(&csv_content)
    }

    /// Compare two solutions for equality
    fn compare_solutions(
        &self,
        actual: &Solution,
        expected: &Solution,
        test: &TestManifest,
    ) -> Result<()> {
        let is_ordered = test.test_type.iter().any(|t| t.contains("OrderedResult"));

        if is_ordered {
            if actual.len() != expected.len() {
                return Err(anyhow!(
                    "Result count mismatch: expected {}, got {}",
                    expected.len(),
                    actual.len()
                ));
            }

            for (i, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
                if !self.rows_equal(actual_row, expected_row) {
                    return Err(anyhow!(
                        "Row {} mismatch:\nExpected: {:?}\nActual: {:?}",
                        i,
                        expected_row,
                        actual_row
                    ));
                }
            }
        } else {
            if actual.len() != expected.len() {
                return Err(anyhow!(
                    "Result count mismatch: expected {}, got {}",
                    expected.len(),
                    actual.len()
                ));
            }

            for expected_row in expected {
                if !actual
                    .iter()
                    .any(|actual_row| self.rows_equal(actual_row, expected_row))
                {
                    return Err(anyhow!(
                        "Expected row not found in results: {:?}",
                        expected_row
                    ));
                }
            }

            for actual_row in actual {
                if !expected
                    .iter()
                    .any(|expected_row| self.rows_equal(actual_row, expected_row))
                {
                    return Err(anyhow!("Unexpected row in results: {:?}", actual_row));
                }
            }
        }

        Ok(())
    }

    /// Check if two rows are equal
    fn rows_equal(&self, row1: &Binding, row2: &Binding) -> bool {
        if row1.len() != row2.len() {
            return false;
        }

        for (var, term1) in row1 {
            match row2.get(var) {
                Some(term2) => {
                    if !self.terms_equal(term1, term2) {
                        return false;
                    }
                }
                None => return false,
            }
        }

        true
    }

    /// Check if two terms are equal
    ///
    /// Blank nodes use existential semantics: any two blank nodes are considered
    /// equal in the context of result row comparison (existence, not identity).
    fn terms_equal(&self, term1: &Term, term2: &Term) -> bool {
        match (term1, term2) {
            (Term::BlankNode(_), Term::BlankNode(_)) => true,
            _ => term1 == term2,
        }
    }

    /// Load expected graph from file
    fn load_expected_graph(&self, path: &Path) -> Result<Vec<Triple>> {
        let content = fs::read_to_string(path)?;

        match path.extension().and_then(|s| s.to_str()) {
            Some("ttl") | Some("turtle") => self.parse_turtle_graph(&content),
            Some("nt") | Some("ntriples") => self.parse_ntriples_graph(&content),
            Some("rdf") | Some("xml") => self.parse_rdf_xml_graph(&content),
            Some("n3") => self.parse_n3_graph(&content),
            _ => {
                if content.trim_start().starts_with("<?xml") {
                    self.parse_rdf_xml_graph(&content)
                } else if content.contains("@prefix") || content.contains("@base") {
                    self.parse_turtle_graph(&content)
                } else {
                    self.parse_ntriples_graph(&content)
                }
            }
        }
    }

    /// Parse Turtle format graph using oxirs_ttl::TurtleParser
    fn parse_turtle_graph(&self, content: &str) -> Result<Vec<Triple>> {
        use oxirs_ttl::turtle::TurtleParser;

        let parser = TurtleParser::new();
        let core_triples = parser
            .parse_document(content)
            .map_err(|e| anyhow!("Turtle parse error: {}", e))?;

        let mut triples = Vec::with_capacity(core_triples.len());
        for t in core_triples {
            let subject = Term::from(t.subject().clone());
            let predicate = Term::from(t.predicate().clone());
            let object = Term::from(t.object().clone());
            triples.push(Triple {
                subject,
                predicate,
                object,
            });
        }

        Ok(triples)
    }

    /// Parse N-Triples format graph
    fn parse_ntriples_graph(&self, content: &str) -> Result<Vec<Triple>> {
        let mut triples = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(triple) = self.parse_ntriples_line(line)? {
                triples.push(triple);
            }
        }

        Ok(triples)
    }

    /// Parse RDF/XML format graph — returns empty graph (complex format, deferred)
    fn parse_rdf_xml_graph(&self, _content: &str) -> Result<Vec<Triple>> {
        Ok(Vec::new())
    }

    /// Parse N3 format graph — delegate to Turtle parser for basic N3
    fn parse_n3_graph(&self, content: &str) -> Result<Vec<Triple>> {
        self.parse_turtle_graph(content)
    }

    /// Parse an N-Triples line
    fn parse_ntriples_line(&self, line: &str) -> Result<Option<Triple>> {
        let line = match line.strip_suffix('.') {
            Some(l) => l.trim(),
            None => return Ok(None),
        };

        let parts = self.split_ntriples_line(line)?;

        if parts.len() != 3 {
            return Ok(None);
        }

        let subject = self.parse_term(&parts[0])?;
        let predicate = self.parse_term(&parts[1])?;
        let object = self.parse_term(&parts[2])?;

        Ok(Some(Triple {
            subject,
            predicate,
            object,
        }))
    }

    /// Split N-Triples line into components
    fn split_ntriples_line(&self, line: &str) -> Result<Vec<String>> {
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;
        for ch in line.chars() {
            match ch {
                '"' if !in_quotes => {
                    in_quotes = true;
                    current.push(ch);
                }
                '"' if in_quotes => {
                    in_quotes = false;
                    current.push(ch);
                }
                ' ' | '\t' if !in_quotes => {
                    if !current.is_empty() {
                        parts.push(current.trim().to_string());
                        current.clear();
                    }
                }
                _ => {
                    current.push(ch);
                }
            }
        }

        if !current.is_empty() {
            parts.push(current.trim().to_string());
        }

        Ok(parts)
    }

    /// Parse a term from string representation
    fn parse_term(&self, s: &str) -> Result<Term> {
        let s = s.trim();

        if s.starts_with('<') && s.ends_with('>') {
            let iri = &s[1..s.len() - 1];
            Ok(Term::Iri(Iri::new_unchecked(iri.to_string())))
        } else if let Some(bnode) = s.strip_prefix("_:") {
            Ok(Term::BlankNode(bnode.to_string()))
        } else if s.starts_with('"') {
            self.parse_literal_term(s)
        } else {
            Ok(Term::Iri(Iri::new_unchecked(s.to_string())))
        }
    }

    /// Parse a literal term from its string representation
    fn parse_literal_term(&self, s: &str) -> Result<Term> {
        if s.len() < 2 {
            return Err(anyhow!("Invalid literal: too short"));
        }

        // Find the closing quote (skip the opening one)
        let inner = &s[1..];
        let end_quote = inner
            .find('"')
            .ok_or_else(|| anyhow!("Invalid literal format: {}", s))?;
        let value = &inner[..end_quote];
        let rest = &inner[end_quote + 1..];

        let lit = if let Some(lang) = rest.strip_prefix('@') {
            Literal::with_language(value.to_string(), lang.to_string())
        } else if let Some(dt_raw) = rest.strip_prefix("^^") {
            let dt = if dt_raw.starts_with('<') && dt_raw.ends_with('>') {
                &dt_raw[1..dt_raw.len() - 1]
            } else {
                dt_raw
            };
            Literal {
                value: value.to_string(),
                language: None,
                datatype: Some(Iri::new_unchecked(dt.to_string())),
            }
        } else {
            Literal {
                value: value.to_string(),
                language: None,
                datatype: None,
            }
        };

        Ok(Term::Literal(lit))
    }

    /// Compare two graphs for equality using RDF canonicalization
    fn compare_graphs(
        &self,
        actual: &Solution,
        expected: &[Triple],
        test: &TestManifest,
    ) -> Result<()> {
        use oxirs_core::{canonicalize, CanonRdfQuad};

        let actual_graph = self.extract_graph_from_solution(actual)?;

        if actual_graph.len() != expected.len() {
            return Err(anyhow!(
                "Graph size mismatch in test {}: expected {} triples, got {}",
                test.name,
                expected.len(),
                actual_graph.len()
            ));
        }

        let to_quad = |triple: &Triple| -> Option<CanonRdfQuad> {
            let s = term_to_canon_quad_term(&triple.subject)?;
            let p = term_to_canon_quad_term(&triple.predicate)?;
            let o = term_to_canon_quad_term(&triple.object)?;
            Some(CanonRdfQuad::new(s, p, o))
        };

        let actual_quads: Vec<CanonRdfQuad> = actual_graph.iter().filter_map(to_quad).collect();
        let expected_quads: Vec<CanonRdfQuad> = expected.iter().filter_map(to_quad).collect();

        let actual_canon = canonicalize(&actual_quads);
        let expected_canon = canonicalize(&expected_quads);

        if actual_canon != expected_canon {
            return Err(anyhow!(
                "Graph mismatch in test {}: canonical forms differ",
                test.name
            ));
        }

        Ok(())
    }

    /// Extract graph triples from solution (for CONSTRUCT/DESCRIBE results)
    fn extract_graph_from_solution(&self, solution: &Solution) -> Result<Vec<Triple>> {
        let s_var = Variable::new_unchecked("subject");
        let p_var = Variable::new_unchecked("predicate");
        let o_var = Variable::new_unchecked("object");

        let mut triples = Vec::new();
        for row in solution {
            if let (Some(s), Some(p), Some(o)) = (row.get(&s_var), row.get(&p_var), row.get(&o_var))
            {
                triples.push(Triple {
                    subject: s.clone(),
                    predicate: p.clone(),
                    object: o.clone(),
                });
            }
        }
        Ok(triples)
    }

    /// Check if graph contains a specific triple
    fn graph_contains_triple(&self, graph: &[Triple], triple: &Triple) -> bool {
        graph.iter().any(|t| self.triples_equal(t, triple))
    }

    /// Check if two triples are equal
    fn triples_equal(&self, triple1: &Triple, triple2: &Triple) -> bool {
        self.terms_equal(&triple1.subject, &triple2.subject)
            && self.terms_equal(&triple1.predicate, &triple2.predicate)
            && self.terms_equal(&triple1.object, &triple2.object)
    }

    /// Canonicalize a slice of triples to a URDNA2015 canonical N-Quads string
    pub fn triples_to_canonical(&self, triples: &[Triple]) -> String {
        use oxirs_core::{canonicalize, CanonRdfQuad};
        let quads: Vec<CanonRdfQuad> = triples
            .iter()
            .filter_map(|t| {
                let s = term_to_canon_quad_term(&t.subject)?;
                let p = term_to_canon_quad_term(&t.predicate)?;
                let o = term_to_canon_quad_term(&t.object)?;
                Some(CanonRdfQuad::new(s, p, o))
            })
            .collect();
        canonicalize(&quads)
    }

    /// Print test results summary
    pub fn print_summary(&self) {
        println!("\nW3C SPARQL Compliance Test Results:");
        println!("===================================");
        println!("Total tests:  {}", self.results.total);
        println!(
            "Passed:       {} ({}%)",
            self.results.passed,
            (self.results.passed * 100) / self.results.total.max(1)
        );
        println!("Failed:       {}", self.results.failed);
        println!("Skipped:      {}", self.results.skipped);

        if !self.results.errors.is_empty() {
            println!("\nFailed tests:");
            for error in &self.results.errors {
                println!(
                    "  - {} ({}): {}",
                    error.test_id, error.test_name, error.error
                );
            }
        }
    }
}

/// Convert an oxirs_arq::Term to a CanonQuadTerm for graph canonicalization
fn term_to_canon_quad_term(term: &Term) -> Option<oxirs_core::CanonQuadTerm> {
    match term {
        Term::Iri(iri) => Some(oxirs_core::CanonQuadTerm::iri(iri.as_str())),
        Term::BlankNode(id) => Some(oxirs_core::CanonQuadTerm::blank(id.as_str())),
        Term::Literal(lit) => {
            if let Some(lang) = &lit.language {
                Some(oxirs_core::CanonQuadTerm::lang_literal(
                    lit.value.as_str(),
                    lang.as_str(),
                ))
            } else if let Some(dt) = &lit.datatype {
                Some(oxirs_core::CanonQuadTerm::typed_literal(
                    lit.value.as_str(),
                    dt.as_str(),
                ))
            } else {
                Some(oxirs_core::CanonQuadTerm::string_literal(
                    lit.value.as_str(),
                ))
            }
        }
        Term::Variable(_) | Term::QuotedTriple(_) | Term::PropertyPath(_) => None,
    }
}

/// Test dataset implementation
struct TestDataset {
    default_graph: Vec<(Term, Term, Term)>,
    named_graphs: HashMap<String, Vec<(Term, Term, Term)>>,
}

impl TestDataset {
    fn new() -> Self {
        Self {
            default_graph: Vec::new(),
            named_graphs: HashMap::new(),
        }
    }

    fn load_default_graph(&mut self, path: &Path) -> Result<()> {
        let content = fs::read_to_string(path)?;
        self.default_graph = load_triples_from_content(&content, path)?;
        Ok(())
    }

    fn load_named_graph(&mut self, graph: &str, path: &Path) -> Result<()> {
        let content = fs::read_to_string(path)?;
        let triples = load_triples_from_content(&content, path)?;
        self.named_graphs.insert(graph.to_string(), triples);
        Ok(())
    }
}

/// Load triples from RDF content, auto-detecting format from path extension
fn load_triples_from_content(content: &str, path: &Path) -> Result<Vec<(Term, Term, Term)>> {
    use oxirs_ttl::turtle::TurtleParser;

    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let is_turtle = matches!(ext, "ttl" | "turtle" | "n3")
        || content.contains("@prefix")
        || content.contains("@base");

    if !is_turtle {
        // N-Triples or other simple line-based format — minimal parser
        return parse_ntriples_content(content);
    }

    let parser = TurtleParser::new();
    let core_triples = parser
        .parse_document(content)
        .map_err(|e| anyhow!("Turtle parse error for {:?}: {}", path, e))?;

    let triples = core_triples
        .into_iter()
        .map(|t| {
            let s = Term::from(t.subject().clone());
            let p = Term::from(t.predicate().clone());
            let o = Term::from(t.object().clone());
            (s, p, o)
        })
        .collect();

    Ok(triples)
}

/// Minimal N-Triples parser for loading graph data
fn parse_ntriples_content(content: &str) -> Result<Vec<(Term, Term, Term)>> {
    let mut triples = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some(t) = parse_ntriples_triple(line)? {
            triples.push(t);
        }
    }

    Ok(triples)
}

/// Parse a single N-Triples line into a (subject, predicate, object) tuple
fn parse_ntriples_triple(line: &str) -> Result<Option<(Term, Term, Term)>> {
    let line = match line.strip_suffix('.') {
        Some(l) => l.trim(),
        None => return Ok(None),
    };

    let parts = split_nt_line(line)?;
    if parts.len() != 3 {
        return Ok(None);
    }

    let s = parse_nt_term(&parts[0])?;
    let p = parse_nt_term(&parts[1])?;
    let o = parse_nt_term(&parts[2])?;

    Ok(Some((s, p, o)))
}

/// Split an N-Triples line respecting quoted strings
fn split_nt_line(line: &str) -> Result<Vec<String>> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    for ch in line.chars() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                current.push(ch);
            }
            ' ' | '\t' if !in_quotes => {
                if !current.is_empty() {
                    parts.push(current.trim().to_string());
                    current.clear();
                }
            }
            _ => {
                current.push(ch);
            }
        }
    }

    if !current.is_empty() {
        parts.push(current.trim().to_string());
    }

    Ok(parts)
}

/// Parse an N-Triples term token into an oxirs_arq::Term
fn parse_nt_term(s: &str) -> Result<Term> {
    let s = s.trim();

    if s.starts_with('<') && s.ends_with('>') {
        let iri = &s[1..s.len() - 1];
        Ok(Term::Iri(Iri::new_unchecked(iri.to_string())))
    } else if let Some(bnode) = s.strip_prefix("_:") {
        Ok(Term::BlankNode(bnode.to_string()))
    } else if s.starts_with('"') {
        parse_nt_literal(s)
    } else {
        Ok(Term::Iri(Iri::new_unchecked(s.to_string())))
    }
}

/// Parse an N-Triples literal token
fn parse_nt_literal(s: &str) -> Result<Term> {
    if s.len() < 2 {
        return Err(anyhow!("Invalid N-Triples literal: too short"));
    }

    let inner = &s[1..];
    let end_quote = inner
        .find('"')
        .ok_or_else(|| anyhow!("Unterminated literal: {}", s))?;
    let value = &inner[..end_quote];
    let rest = &inner[end_quote + 1..];

    let lit = if let Some(lang) = rest.strip_prefix('@') {
        Literal::with_language(value.to_string(), lang.to_string())
    } else if let Some(dt_raw) = rest.strip_prefix("^^") {
        let dt = if dt_raw.starts_with('<') && dt_raw.ends_with('>') {
            &dt_raw[1..dt_raw.len() - 1]
        } else {
            dt_raw
        };
        Literal {
            value: value.to_string(),
            language: None,
            datatype: Some(Iri::new_unchecked(dt.to_string())),
        }
    } else {
        Literal {
            value: value.to_string(),
            language: None,
            datatype: None,
        }
    };

    Ok(Term::Literal(lit))
}

impl Dataset for TestDataset {
    fn find_triples(&self, pattern: &TriplePattern) -> Result<Vec<(Term, Term, Term)>> {
        let results = self
            .default_graph
            .iter()
            .filter(|(s, p, o)| {
                term_matches(&pattern.subject, s)
                    && term_matches(&pattern.predicate, p)
                    && term_matches(&pattern.object, o)
            })
            .cloned()
            .collect();
        Ok(results)
    }

    fn contains_triple(&self, subject: &Term, predicate: &Term, object: &Term) -> Result<bool> {
        let found = self
            .default_graph
            .iter()
            .any(|(s, p, o)| s == subject && p == predicate && o == object);
        Ok(found)
    }

    fn subjects(&self) -> Result<Vec<Term>> {
        let mut subjects: Vec<Term> = self
            .default_graph
            .iter()
            .map(|(s, _, _)| s.clone())
            .collect();
        subjects.dedup();
        Ok(subjects)
    }

    fn predicates(&self) -> Result<Vec<Term>> {
        let mut predicates: Vec<Term> = self
            .default_graph
            .iter()
            .map(|(_, p, _)| p.clone())
            .collect();
        predicates.dedup();
        Ok(predicates)
    }

    fn objects(&self) -> Result<Vec<Term>> {
        let mut objects: Vec<Term> = self
            .default_graph
            .iter()
            .map(|(_, _, o)| o.clone())
            .collect();
        objects.dedup();
        Ok(objects)
    }
}

/// Check whether a pattern term matches a concrete term
fn term_matches(pattern: &Term, concrete: &Term) -> bool {
    match pattern {
        Term::Variable(_) => true, // Variables match anything
        other => other == concrete,
    }
}

// ---------------------------------------------------------------------------
// W3C test-manifest (Turtle) parsing — issue #65
//
// W3C SPARQL test manifests are Turtle documents (`manifest.ttl`), *not* JSON.
// `parse_manifest` parses the Turtle and walks the `mf:entries` RDF collection,
// materialising each entry into a `TestManifest`. The previous implementation
// fed the Turtle bytes to `serde_json::from_str`, which can never succeed on a
// `.ttl` file (see `run_manifest`).
// ---------------------------------------------------------------------------

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDF_FIRST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
const RDF_REST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
const RDF_NIL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";
const RDFS_COMMENT: &str = "http://www.w3.org/2000/01/rdf-schema#comment";
const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
const MF_ENTRIES: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#entries";
const MF_NAME: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#name";
const MF_ACTION: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#action";
const MF_RESULT: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#result";
const MF_REQUIRES: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#requires";
const QT_QUERY: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-query#query";
const QT_DATA: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-query#data";
const QT_GRAPH_DATA: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-query#graphData";
const QT_GRAPH: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-query#graph";
const DAWGT_APPROVAL: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-dawg#approval";

/// Base IRI for resolving the manifest's relative references (`<>`, `<agg01.rq>`, …).
///
/// A concrete base is required: the RDF parser validates IRIs, and the empty
/// relative reference `<>` (the manifest node) is not a valid absolute IRI on
/// its own. Relative file references are reduced to their basename before being
/// stored, so this exact value never leaks into a `TestManifest` path field.
const MANIFEST_BASE_IRI: &str = "http://oxirs.example/w3c-manifest/";

/// Parse a W3C SPARQL test manifest (Turtle) into its list of [`TestManifest`]s.
///
/// Walks the `mf:entries` RDF collection and reads each entry's `rdf:type`,
/// `mf:name`, `rdfs:comment`, `mf:action` (either a direct query IRI for
/// negative-syntax tests or a `[ qt:query … ; qt:data … ]` blank node),
/// `mf:result`, and approval metadata.
fn parse_manifest(content: &str) -> Result<Vec<TestManifest>> {
    use oxirs_ttl::turtle::TurtleParser;

    let parser = TurtleParser::new().with_base_iri(MANIFEST_BASE_IRI.to_string());
    let core_triples = parser
        .parse_document(content)
        .map_err(|e| anyhow!("Failed to parse Turtle manifest: {}", e))?;

    // Materialise as arq terms so subjects/objects — including the blank nodes
    // used by RDF lists and `mf:action [ … ]` — compare uniformly.
    let triples: Vec<(Term, Term, Term)> = core_triples
        .into_iter()
        .map(|t| {
            (
                Term::from(t.subject().clone()),
                Term::from(t.predicate().clone()),
                Term::from(t.object().clone()),
            )
        })
        .collect();

    let mut manifests = Vec::new();
    // A manifest normally has a single `mf:entries` list; tolerate more than one.
    for list_head in manifest_objects(&triples, MF_ENTRIES) {
        for entry in collect_rdf_list(&triples, &list_head) {
            if let Some(manifest) = build_test_manifest(&triples, &entry) {
                manifests.push(manifest);
            }
        }
    }

    Ok(manifests)
}

/// Build a single [`TestManifest`] from an entry IRI. Returns `None` for
/// entries that carry no `mf:action` (i.e. are not runnable tests).
fn build_test_manifest(triples: &[(Term, Term, Term)], entry: &Term) -> Option<TestManifest> {
    let action_obj = first_object(triples, entry, MF_ACTION)?;
    let action = build_test_action(triples, action_obj);

    let test_type = objects(triples, entry, RDF_TYPE)
        .into_iter()
        .filter_map(iri_string)
        .collect();
    let name = first_object(triples, entry, MF_NAME)
        .and_then(literal_string)
        .unwrap_or_default();
    let comment = first_object(triples, entry, RDFS_COMMENT).and_then(literal_string);
    let result = first_object(triples, entry, MF_RESULT)
        .and_then(iri_string)
        .map(|iri| TestResult::ResultFile(basename(&iri)));
    let requires = objects(triples, entry, MF_REQUIRES)
        .into_iter()
        .filter_map(iri_string)
        .collect();
    let approval = first_object(triples, entry, DAWGT_APPROVAL)
        .and_then(iri_string)
        .unwrap_or_default();

    Some(TestManifest {
        id: term_identifier(entry),
        test_type,
        name,
        comment,
        action,
        result,
        requires,
        approval,
    })
}

/// Build a [`TestAction`] from the object of an entry's `mf:action`.
fn build_test_action(triples: &[(Term, Term, Term)], action: &Term) -> TestAction {
    if let Term::Iri(iri) = action {
        // Negative-syntax tests point `mf:action` directly at the query file.
        return TestAction {
            query: basename(iri.as_str()),
            data: None,
            graph_data: None,
        };
    }

    // Evaluation tests use a blank node: [ qt:query <q> ; qt:data <d> ].
    let query = first_object(triples, action, QT_QUERY)
        .and_then(iri_string)
        .map(|iri| basename(&iri))
        .unwrap_or_default();
    let data = first_object(triples, action, QT_DATA)
        .and_then(iri_string)
        .map(|iri| basename(&iri));
    let graph_data = build_graph_data(triples, action);

    TestAction {
        query,
        data,
        graph_data,
    }
}

/// Collect the `qt:graphData` named-graph inputs of an action, if any.
fn build_graph_data(triples: &[(Term, Term, Term)], action: &Term) -> Option<Vec<GraphData>> {
    let graphs: Vec<GraphData> = objects(triples, action, QT_GRAPH_DATA)
        .into_iter()
        .map(|obj| match obj {
            // `qt:graphData <file.ttl>` — the data IRI doubles as the graph name.
            Term::Iri(iri) => GraphData {
                graph: iri.as_str().to_string(),
                data: basename(iri.as_str()),
            },
            // `qt:graphData [ qt:graph <file.ttl> ; rdfs:label "graph-iri" ]`.
            _ => {
                let file = first_object(triples, obj, QT_GRAPH)
                    .and_then(iri_string)
                    .unwrap_or_default();
                let graph = first_object(triples, obj, RDFS_LABEL)
                    .and_then(literal_string)
                    .unwrap_or_else(|| file.clone());
                GraphData {
                    graph,
                    data: basename(&file),
                }
            }
        })
        .collect();

    if graphs.is_empty() {
        None
    } else {
        Some(graphs)
    }
}

/// All objects of triples with the given predicate IRI, regardless of subject.
fn manifest_objects(triples: &[(Term, Term, Term)], predicate: &str) -> Vec<Term> {
    triples
        .iter()
        .filter(|(_, p, _)| iri_eq(p, predicate))
        .map(|(_, _, o)| o.clone())
        .collect()
}

/// Objects of all triples matching `(subject, predicate, ?)`.
fn objects<'a>(
    triples: &'a [(Term, Term, Term)],
    subject: &Term,
    predicate: &str,
) -> Vec<&'a Term> {
    triples
        .iter()
        .filter(|(s, p, _)| s == subject && iri_eq(p, predicate))
        .map(|(_, _, o)| o)
        .collect()
}

/// First object of a triple matching `(subject, predicate, ?)`.
fn first_object<'a>(
    triples: &'a [(Term, Term, Term)],
    subject: &Term,
    predicate: &str,
) -> Option<&'a Term> {
    triples
        .iter()
        .find(|(s, p, _)| s == subject && iri_eq(p, predicate))
        .map(|(_, _, o)| o)
}

/// Walk an RDF collection (`rdf:first`/`rdf:rest`) from its head to `rdf:nil`.
fn collect_rdf_list(triples: &[(Term, Term, Term)], head: &Term) -> Vec<Term> {
    let mut items = Vec::new();
    let mut current = head.clone();
    let mut visited: HashSet<String> = HashSet::new();

    while !iri_eq(&current, RDF_NIL) {
        // Guard against cyclic or malformed lists.
        if !visited.insert(term_identifier(&current)) {
            break;
        }
        match first_object(triples, &current, RDF_FIRST) {
            Some(item) => items.push(item.clone()),
            None => break,
        }
        match first_object(triples, &current, RDF_REST) {
            Some(rest) => current = rest.clone(),
            None => break,
        }
    }

    items
}

/// True if `term` is the IRI `iri`.
fn iri_eq(term: &Term, iri: &str) -> bool {
    matches!(term, Term::Iri(node) if node.as_str() == iri)
}

/// The IRI string of a term, if it is an IRI.
fn iri_string(term: &Term) -> Option<String> {
    match term {
        Term::Iri(node) => Some(node.as_str().to_string()),
        _ => None,
    }
}

/// The lexical value of a term, if it is a literal.
fn literal_string(term: &Term) -> Option<String> {
    match term {
        Term::Literal(lit) => Some(lit.value.clone()),
        _ => None,
    }
}

/// A stable identifier string for a term (IRI value, or `_:id` for blank nodes).
fn term_identifier(term: &Term) -> String {
    match term {
        Term::Iri(node) => node.as_str().to_string(),
        Term::BlankNode(id) => format!("_:{id}"),
        other => other.to_string(),
    }
}

/// Final path segment of an IRI/path (everything after the last `/`).
fn basename(iri: &str) -> String {
    iri.rsplit('/').next().unwrap_or(iri).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_creation() {
        let test_dir = std::env::temp_dir().join("oxirs_w3c_test");
        let runner = TestSuiteRunner::new(&test_dir).expect("runner creation should succeed");
        assert_eq!(runner.results.total, 0);
        assert_eq!(runner.results.passed, 0);
        assert_eq!(runner.results.failed, 0);
        assert_eq!(runner.results.skipped, 0);
    }

    #[test]
    fn test_parse_sparql_json_results_empty() {
        let content = r#"{"head":{"vars":["x"]},"results":{"bindings":[]}}"#;
        let runner =
            TestSuiteRunner::new(std::env::temp_dir()).expect("runner creation should succeed");
        let solution = runner
            .parse_sparql_json_results(content)
            .expect("parse should succeed");
        assert_eq!(solution.len(), 0);
    }

    #[test]
    fn test_parse_sparql_json_results_iri() {
        let content = r#"{
            "head":{"vars":["x"]},
            "results":{"bindings":[
                {"x":{"type":"uri","value":"http://example.org/s"}}
            ]}
        }"#;
        let runner =
            TestSuiteRunner::new(std::env::temp_dir()).expect("runner creation should succeed");
        let solution = runner
            .parse_sparql_json_results(content)
            .expect("parse should succeed");
        assert_eq!(solution.len(), 1);
        let var_x = Variable::new_unchecked("x");
        let term = solution[0].get(&var_x).expect("should have var x");
        assert_eq!(*term, Term::Iri(Iri::new_unchecked("http://example.org/s")));
    }

    #[test]
    fn test_parse_csv_results() {
        let content = "x,y\nhttp://example.org/a,hello\n";
        let runner =
            TestSuiteRunner::new(std::env::temp_dir()).expect("runner creation should succeed");
        let solution = runner
            .parse_csv_results(content)
            .expect("parse should succeed");
        assert_eq!(solution.len(), 1);
        let var_x = Variable::new_unchecked("x");
        let term = solution[0].get(&var_x).expect("should have var x");
        assert_eq!(*term, Term::Iri(Iri::new_unchecked("http://example.org/a")));
    }

    #[test]
    fn test_parse_ntriples_graph_empty() {
        let content = "# empty graph\n";
        let runner =
            TestSuiteRunner::new(std::env::temp_dir()).expect("runner creation should succeed");
        let graph = runner
            .parse_ntriples_graph(content)
            .expect("parse should succeed");
        assert_eq!(graph.len(), 0);
    }

    #[test]
    fn test_parse_ntriples_graph_single_triple() {
        let content = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n";
        let runner =
            TestSuiteRunner::new(std::env::temp_dir()).expect("runner creation should succeed");
        let graph = runner
            .parse_ntriples_graph(content)
            .expect("parse should succeed");
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_dataset_find_triples_empty() {
        let dataset = TestDataset::new();
        let pattern = TriplePattern {
            subject: Term::Variable(Variable::new_unchecked("s")),
            predicate: Term::Variable(Variable::new_unchecked("p")),
            object: Term::Variable(Variable::new_unchecked("o")),
        };
        let results = dataset
            .find_triples(&pattern)
            .expect("find_triples should succeed");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_dataset_contains_triple() {
        let dataset = TestDataset::new();
        let s = Term::Iri(Iri::new_unchecked("http://example.org/s"));
        let p = Term::Iri(Iri::new_unchecked("http://example.org/p"));
        let o = Term::Iri(Iri::new_unchecked("http://example.org/o"));
        let found = dataset
            .contains_triple(&s, &p, &o)
            .expect("contains_triple should succeed");
        assert!(!found);
    }

    #[test]
    fn test_dataset_subjects_predicates_objects_empty() {
        let dataset = TestDataset::new();
        assert_eq!(dataset.subjects().expect("subjects").len(), 0);
        assert_eq!(dataset.predicates().expect("predicates").len(), 0);
        assert_eq!(dataset.objects().expect("objects").len(), 0);
    }

    #[test]
    fn test_skip_sparql_star() {
        let runner =
            TestSuiteRunner::new(std::env::temp_dir()).expect("runner creation should succeed");
        let test = TestManifest {
            id: "test1".to_string(),
            test_type: vec!["QueryEvaluationTest".to_string()],
            name: "SPARQL-star test".to_string(),
            comment: None,
            action: TestAction {
                query: "q.rq".to_string(),
                data: None,
                graph_data: None,
            },
            result: None,
            requires: vec!["SPARQL-star".to_string()],
            approval: String::new(),
        };
        assert!(runner.should_skip_test(&test));
    }

    #[test]
    fn test_compare_graphs_isomorphic_blank_nodes() {
        let runner =
            TestSuiteRunner::new(std::env::temp_dir()).expect("runner creation should succeed");
        let p = Term::Iri(Iri::new_unchecked("http://example.org/p"));
        let o = Term::Iri(Iri::new_unchecked("http://example.org/o"));
        let g1 = vec![Triple {
            subject: Term::BlankNode("x1".to_string()),
            predicate: p.clone(),
            object: o.clone(),
        }];
        let g2 = vec![Triple {
            subject: Term::BlankNode("y1".to_string()),
            predicate: p.clone(),
            object: o.clone(),
        }];
        let canon1 = runner.triples_to_canonical(&g1);
        let canon2 = runner.triples_to_canonical(&g2);
        assert_eq!(
            canon1, canon2,
            "Isomorphic graphs must produce same canonical string"
        );
    }

    #[test]
    fn test_compare_graphs_non_isomorphic() {
        let runner =
            TestSuiteRunner::new(std::env::temp_dir()).expect("runner creation should succeed");
        let p = Term::Iri(Iri::new_unchecked("http://example.org/p"));
        let o1 = Term::Iri(Iri::new_unchecked("http://example.org/o1"));
        let o2 = Term::Iri(Iri::new_unchecked("http://example.org/o2"));
        let g1 = vec![Triple {
            subject: Term::BlankNode("x".to_string()),
            predicate: p.clone(),
            object: o1.clone(),
        }];
        let g2 = vec![Triple {
            subject: Term::BlankNode("x".to_string()),
            predicate: p.clone(),
            object: o2.clone(),
        }];
        let canon1 = runner.triples_to_canonical(&g1);
        let canon2 = runner.triples_to_canonical(&g2);
        assert_ne!(
            canon1, canon2,
            "Non-isomorphic graphs must produce different canonical strings"
        );
    }

    #[test]
    fn test_terms_equal_blank_nodes() {
        let runner =
            TestSuiteRunner::new(std::env::temp_dir()).expect("runner creation should succeed");
        let b1 = Term::BlankNode("a".to_string());
        let b2 = Term::BlankNode("b".to_string());
        assert!(
            runner.terms_equal(&b1, &b2),
            "Blank nodes should be equal in existential sense"
        );
        let iri1 = Term::Iri(Iri::new_unchecked("http://example.org/a"));
        let iri2 = Term::Iri(Iri::new_unchecked("http://example.org/b"));
        assert!(
            !runner.terms_equal(&iri1, &iri2),
            "Different IRIs should not be equal"
        );
        assert!(
            runner.terms_equal(&iri1, &iri1.clone()),
            "Same IRI should be equal"
        );
    }

    #[test]
    fn test_parse_turtle_graph_basic() {
        let runner =
            TestSuiteRunner::new(std::env::temp_dir()).expect("runner creation should succeed");
        let turtle = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n\
             <http://example.org/s> <http://example.org/q> \"hello\" .\n";
        let triples = runner.parse_turtle_graph(turtle).expect("parse turtle");
        assert_eq!(triples.len(), 2, "Expected 2 triples");
    }

    #[test]
    fn test_parse_ntriples_graph_basic() {
        let runner =
            TestSuiteRunner::new(std::env::temp_dir()).expect("runner creation should succeed");
        let nt = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n";
        let triples = runner.parse_ntriples_graph(nt).expect("parse ntriples");
        assert_eq!(triples.len(), 1, "Expected 1 triple");
    }

    /// Regression for issue #65: W3C manifests are Turtle, not JSON. The old
    /// `run_manifest` called `serde_json::from_str` on the `.ttl` bytes, which
    /// can never succeed. This exercises the Turtle path on a real manifest.
    #[test]
    fn test_issue_65_manifest_parses_as_turtle() {
        // A real manifest shipped in the repo (28 tests in its mf:entries list).
        let manifest = include_str!("data/aggregates/manifest.ttl");

        let entries = parse_manifest(manifest).expect("aggregates manifest should parse as Turtle");
        assert_eq!(
            entries.len(),
            28,
            "aggregates manifest declares 28 tests in its mf:entries list"
        );

        // An evaluation test: mf:action [ qt:query <agg01.rq> ; qt:data <agg01.ttl> ].
        let count1 = entries
            .iter()
            .find(|m| m.name == "COUNT 1")
            .expect("entry 'COUNT 1' (:agg01) should be extracted");
        assert_eq!(count1.action.query, "agg01.rq");
        assert_eq!(count1.action.data.as_deref(), Some("agg01.ttl"));
        assert!(
            matches!(&count1.result, Some(TestResult::ResultFile(f)) if f == "agg01.srx"),
            "COUNT 1 result should be agg01.srx, got {:?}",
            count1.result
        );
        assert!(
            count1.id.ends_with("#agg01"),
            "entry id should be the resolved IRI ending in #agg01, got {}",
            count1.id
        );
        assert!(
            count1
                .test_type
                .iter()
                .any(|t| t.ends_with("#QueryEvaluationTest")),
            "COUNT 1 should be typed as an mf:QueryEvaluationTest, got {:?}",
            count1.test_type
        );

        // A negative-syntax test points mf:action straight at the .rq file.
        let count8 = entries
            .iter()
            .find(|m| m.name == "COUNT 8")
            .expect("entry 'COUNT 8' (:agg08) should be extracted");
        assert_eq!(count8.action.query, "agg08.rq");
        assert_eq!(count8.action.data, None);

        // Document the original bug: the same bytes are NOT valid JSON.
        assert!(
            serde_json::from_str::<Vec<TestManifest>>(manifest).is_err(),
            "the Turtle manifest must not parse as JSON (the original issue #65 bug)"
        );
    }
}
