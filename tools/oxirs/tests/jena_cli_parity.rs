//! # Jena CLI Behavioral Parity Matrix
//!
//! This integration test exercises the OxiRS `oxirs` CLI binary against a
//! curated matrix of fixture-based commands and verifies that the resulting
//! `(stdout, stderr, exit_code)` triple satisfies the contract recorded in
//! the fixture.  The fixtures collectively model the behavioural surface of
//! the Jena `riot` / `arq`/`sparql` / `tdb2.tdbquery` / `shacl` CLI tools.
//!
//! ## Fixture layout
//!
//! Each subdirectory under `tests/fixtures/jena-cli-ref/` is a fixture and
//! must contain a single `manifest.toml` describing the test case:
//!
//! ```toml
//! # Required: human description of the case
//! description = "..."
//!
//! # Required: classification — `spec_required` mismatches FAIL the test;
//! #          `impl_detail` mismatches print a warning but PASS.
//! # One of: "spec-required" | "impl-detail"
//! classification = "spec-required"
//!
//! # Required: arguments to pass *after* the binary path.  Each entry is
//! # one argv element (no shell quoting).
//! argv = ["query", "--help"]
//!
//! # Optional: text written to stdin
//! stdin = ""
//!
//! # Required: the expected exit status (i32)
//! exit_code = 0
//!
//! # stdout/stderr expectations (each may be absent if no constraint).
//! # Modes:
//! #   "exact"     — full string equality (after newline normalisation)
//! #   "contains"  — every line in `expected` must appear *as a substring*
//! #                 of stdout (in any order)
//! #   "regex"     — `expected` is one or more regular expressions, each
//! #                 must find a match
//! #   "absent"    — the stream MUST be empty (after trimming)
//! #
//! # `expected` is multiline; for `contains` and `regex` each non-blank
//! # line is one expectation.  For `exact` the entire block is matched.
//! [stdout]
//! mode = "contains"
//! expected = """
//! Execute SPARQL query
//! """
//!
//! [stderr]
//! mode = "absent"
//! ```
//!
//! ## Substitutions
//!
//! Inside `argv`, the token `${TMPDIR}` is substituted with a per-fixture
//! scratch directory (created fresh for each run) and `${FIXTURE}` is
//! substituted with the absolute path of the fixture directory itself.
//! This lets fixtures reference vendored input files without hard-coded
//! absolute paths.
//!
//! ## Driver semantics
//!
//! `run_jena_parity_matrix` walks every fixture, executes the binary
//! (located via `env!("CARGO_BIN_EXE_oxirs")`), and aggregates the
//! results.  Spec-required mismatches are collected into a single failure
//! report; implementation-detail mismatches are reported via `eprintln!`
//! but do not fail the build.  This mirrors the gating policy used on the
//! HTTP parity side.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const FIXTURE_ROOT: &str = "tests/fixtures/jena-cli-ref";
const OXIRS_BIN: &str = env!("CARGO_BIN_EXE_oxirs");

// ─────────────────────────────────────────────────────────────────────────────
// Fixture model
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Classification {
    SpecRequired,
    ImplDetail,
}

impl Classification {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "spec-required" | "spec_required" => Ok(Classification::SpecRequired),
            "impl-detail" | "impl_detail" => Ok(Classification::ImplDetail),
            other => Err(format!(
                "unknown classification '{other}'; expected 'spec-required' or 'impl-detail'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamMode {
    Exact,
    Contains,
    Regex,
    Absent,
}

impl StreamMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "exact" => Ok(StreamMode::Exact),
            "contains" => Ok(StreamMode::Contains),
            "regex" => Ok(StreamMode::Regex),
            "absent" => Ok(StreamMode::Absent),
            other => Err(format!("unknown stream mode '{other}'")),
        }
    }
}

#[derive(Debug, Clone)]
struct StreamSpec {
    mode: StreamMode,
    expected: String,
}

#[derive(Debug, Clone)]
struct Fixture {
    /// Fixture directory name (used for diagnostics and ordering).
    name: String,
    /// Absolute path of the fixture directory.
    dir: PathBuf,
    description: String,
    classification: Classification,
    argv: Vec<String>,
    stdin: String,
    exit_code: i32,
    stdout: Option<StreamSpec>,
    stderr: Option<StreamSpec>,
    /// Optional preparatory commands run *before* the main `argv` invocation.
    /// Each entry is a complete argv (without the binary path) executed in
    /// sequence; non-zero exit aborts the fixture with a setup error.  Useful
    /// for fixtures that need an initialised TDB store.
    setup: Vec<Vec<String>>,
}

impl Fixture {
    fn load(dir: PathBuf) -> Result<Self, String> {
        let name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string)
            .ok_or_else(|| format!("fixture directory has no name: {}", dir.display()))?;

        let manifest = dir.join("manifest.toml");
        let raw = fs::read_to_string(&manifest)
            .map_err(|e| format!("[{name}] cannot read {}: {e}", manifest.display()))?;

        let value: toml::Value =
            toml::from_str(&raw).map_err(|e| format!("[{name}] manifest parse error: {e}"))?;

        let table = value
            .as_table()
            .ok_or_else(|| format!("[{name}] manifest must be a TOML table"))?;

        let description = table
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("[{name}] missing 'description'"))?
            .to_string();

        let classification = table
            .get("classification")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("[{name}] missing 'classification'"))?;
        let classification =
            Classification::parse(classification).map_err(|e| format!("[{name}] {e}"))?;

        let argv = table
            .get("argv")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("[{name}] missing 'argv' array"))?;
        let argv: Vec<String> = argv
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("[{name}] argv entries must be strings"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        if argv.is_empty() {
            return Err(format!("[{name}] argv must not be empty"));
        }

        let stdin = table
            .get("stdin")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_default();

        let exit_code = table
            .get("exit_code")
            .and_then(|v| v.as_integer())
            .ok_or_else(|| format!("[{name}] missing 'exit_code'"))? as i32;

        let stdout = parse_stream_spec(table.get("stdout"), &name, "stdout")?;
        let stderr = parse_stream_spec(table.get("stderr"), &name, "stderr")?;

        let setup = match table.get("setup") {
            None => Vec::new(),
            Some(value) => {
                let outer = value
                    .as_array()
                    .ok_or_else(|| format!("[{name}] 'setup' must be an array of argv arrays"))?;
                let mut sets = Vec::with_capacity(outer.len());
                for (idx, entry) in outer.iter().enumerate() {
                    let inner = entry.as_array().ok_or_else(|| {
                        format!("[{name}] setup[{idx}] must be an array of argv strings")
                    })?;
                    let mut argv_inner = Vec::with_capacity(inner.len());
                    for arg in inner {
                        let arg = arg.as_str().ok_or_else(|| {
                            format!("[{name}] setup[{idx}] entries must be strings")
                        })?;
                        argv_inner.push(arg.to_string());
                    }
                    if argv_inner.is_empty() {
                        return Err(format!("[{name}] setup[{idx}] must not be empty"));
                    }
                    sets.push(argv_inner);
                }
                sets
            }
        };

        Ok(Self {
            name,
            dir,
            description,
            classification,
            argv,
            stdin,
            exit_code,
            stdout,
            stderr,
            setup,
        })
    }
}

fn parse_stream_spec(
    raw: Option<&toml::Value>,
    fixture_name: &str,
    field: &str,
) -> Result<Option<StreamSpec>, String> {
    let Some(value) = raw else {
        return Ok(None);
    };
    let table = value
        .as_table()
        .ok_or_else(|| format!("[{fixture_name}] '{field}' must be a TOML table"))?;
    let mode = table
        .get("mode")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("[{fixture_name}] '{field}.mode' is required"))?;
    let mode = StreamMode::parse(mode).map_err(|e| format!("[{fixture_name}] {field}: {e}"))?;
    let expected = table
        .get("expected")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_default();
    if mode == StreamMode::Absent && !expected.trim().is_empty() {
        return Err(format!(
            "[{fixture_name}] {field}: 'absent' mode must not have a non-empty 'expected' value"
        ));
    }
    if matches!(
        mode,
        StreamMode::Exact | StreamMode::Contains | StreamMode::Regex
    ) && expected.trim().is_empty()
    {
        return Err(format!(
            "[{fixture_name}] {field}: mode '{mode:?}' requires a non-empty 'expected' value"
        ));
    }
    Ok(Some(StreamSpec { mode, expected }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Discovery
// ─────────────────────────────────────────────────────────────────────────────

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_ROOT)
}

fn discover_fixtures() -> Result<Vec<Fixture>, String> {
    let root = fixture_root();
    if !root.exists() {
        return Err(format!(
            "fixture root '{}' does not exist; expected vendored Jena CLI references",
            root.display()
        ));
    }
    let mut entries: Vec<_> = fs::read_dir(&root)
        .map_err(|e| format!("cannot read fixture root '{}': {e}", root.display()))?
        .filter_map(|res| res.ok())
        .filter(|entry| entry.path().is_dir())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    let mut fixtures = Vec::with_capacity(entries.len());
    for entry in entries {
        // Skip directories without a manifest (e.g. shared input data).
        if !entry.path().join("manifest.toml").exists() {
            continue;
        }
        fixtures.push(Fixture::load(entry.path())?);
    }
    Ok(fixtures)
}

// ─────────────────────────────────────────────────────────────────────────────
// Substitution
// ─────────────────────────────────────────────────────────────────────────────

fn substitute(value: &str, vars: &BTreeMap<&str, String>) -> String {
    let mut out = value.to_string();
    for (key, replacement) in vars {
        let token = format!("${{{key}}}");
        out = out.replace(&token, replacement);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Comparators
// ─────────────────────────────────────────────────────────────────────────────

fn normalise(text: &str) -> String {
    text.replace("\r\n", "\n")
}

fn check_stream(stream: &str, actual: &str, spec: &Option<StreamSpec>) -> Result<(), String> {
    let Some(spec) = spec else {
        return Ok(());
    };
    let actual_norm = normalise(actual);
    match spec.mode {
        StreamMode::Exact => {
            let expected_norm = normalise(&spec.expected);
            if actual_norm.trim_end() == expected_norm.trim_end() {
                Ok(())
            } else {
                Err(format!(
                    "{stream}: exact mismatch\n--- expected ---\n{expected}\n--- actual ---\n{actual}",
                    expected = expected_norm,
                    actual = actual_norm,
                ))
            }
        }
        StreamMode::Contains => {
            let mut missing = Vec::new();
            for needle in spec.expected.lines() {
                let needle = needle.trim();
                if needle.is_empty() {
                    continue;
                }
                if !actual_norm.contains(needle) {
                    missing.push(needle.to_string());
                }
            }
            if missing.is_empty() {
                Ok(())
            } else {
                Err(format!(
                    "{stream}: missing expected substring(s):\n  - {}\n--- actual ---\n{}",
                    missing.join("\n  - "),
                    actual_norm
                ))
            }
        }
        StreamMode::Regex => {
            let mut failures = Vec::new();
            for raw in spec.expected.lines() {
                let pattern = raw.trim();
                if pattern.is_empty() {
                    continue;
                }
                match regex::Regex::new(pattern) {
                    Ok(re) => {
                        if !re.is_match(&actual_norm) {
                            failures.push(format!("pattern not matched: {pattern}"));
                        }
                    }
                    Err(e) => {
                        failures.push(format!("invalid regex '{pattern}': {e}"));
                    }
                }
            }
            if failures.is_empty() {
                Ok(())
            } else {
                Err(format!(
                    "{stream}: regex check failed:\n  - {}\n--- actual ---\n{}",
                    failures.join("\n  - "),
                    actual_norm
                ))
            }
        }
        StreamMode::Absent => {
            if actual_norm.trim().is_empty() {
                Ok(())
            } else {
                Err(format!(
                    "{stream}: expected to be empty but had content:\n{actual_norm}"
                ))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Execution
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct FixtureOutcome {
    name: String,
    classification: Classification,
    description: String,
    failures: Vec<String>,
}

impl FixtureOutcome {
    fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

fn run_fixture(fixture: &Fixture) -> FixtureOutcome {
    let mut failures = Vec::new();

    // Prepare per-fixture substitutions.
    let scratch = match tempfile::tempdir() {
        Ok(dir) => dir,
        Err(e) => {
            return FixtureOutcome {
                name: fixture.name.clone(),
                classification: fixture.classification,
                description: fixture.description.clone(),
                failures: vec![format!("cannot create scratch dir: {e}")],
            }
        }
    };

    let mut vars: BTreeMap<&str, String> = BTreeMap::new();
    vars.insert("TMPDIR", scratch.path().to_string_lossy().to_string());
    vars.insert("FIXTURE", fixture.dir.to_string_lossy().to_string());

    // Run setup commands sequentially.  Any non-zero exit aborts the fixture
    // with a 'setup failed' diagnostic so the user can distinguish prep from
    // actual parity failures.
    for (idx, setup_argv) in fixture.setup.iter().enumerate() {
        let resolved: Vec<String> = setup_argv.iter().map(|s| substitute(s, &vars)).collect();
        let output = Command::new(OXIRS_BIN)
            .args(&resolved)
            .env("NO_COLOR", "1")
            .env("OXIRS_DISABLE_CACHE", "1")
            .env("RUST_LOG", "off")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();
        match output {
            Ok(o) if o.status.success() => continue,
            Ok(o) => {
                failures.push(format!(
                    "setup[{idx}] failed (exit={}): argv={resolved:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                    o.status.code().unwrap_or(-1),
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                ));
                return FixtureOutcome {
                    name: fixture.name.clone(),
                    classification: fixture.classification,
                    description: fixture.description.clone(),
                    failures,
                };
            }
            Err(e) => {
                failures.push(format!("setup[{idx}] could not be spawned: {e}"));
                return FixtureOutcome {
                    name: fixture.name.clone(),
                    classification: fixture.classification,
                    description: fixture.description.clone(),
                    failures,
                };
            }
        }
    }

    let argv: Vec<String> = fixture.argv.iter().map(|s| substitute(s, &vars)).collect();

    // Honour `quiet` to avoid color codes in stdout for deterministic checks.
    let mut command = Command::new(OXIRS_BIN);
    command
        .args(&argv)
        .env("NO_COLOR", "1")
        .env("OXIRS_DISABLE_CACHE", "1")
        .env("RUST_LOG", "off")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => {
            failures.push(format!(
                "failed to spawn '{}' with argv={argv:?}: {e}",
                OXIRS_BIN
            ));
            return FixtureOutcome {
                name: fixture.name.clone(),
                classification: fixture.classification,
                description: fixture.description.clone(),
                failures,
            };
        }
    };

    if !fixture.stdin.is_empty() {
        if let Some(mut stdin) = child.stdin.take() {
            if let Err(e) = stdin.write_all(fixture.stdin.as_bytes()) {
                failures.push(format!("stdin write failed: {e}"));
            }
        }
    }
    // Drop stdin to signal EOF whether or not we wrote anything.
    drop(child.stdin.take());

    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => {
            failures.push(format!("waiting for process failed: {e}"));
            return FixtureOutcome {
                name: fixture.name.clone(),
                classification: fixture.classification,
                description: fixture.description.clone(),
                failures,
            };
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let actual_exit = output
        .status
        .code()
        .unwrap_or(if output.status.success() { 0 } else { -1 });

    if actual_exit != fixture.exit_code {
        failures.push(format!(
            "exit code mismatch: expected {expected}, got {actual} (argv={argv:?})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            expected = fixture.exit_code,
            actual = actual_exit,
        ));
    }

    if let Err(e) = check_stream("stdout", &stdout, &fixture.stdout) {
        failures.push(e);
    }
    if let Err(e) = check_stream("stderr", &stderr, &fixture.stderr) {
        failures.push(e);
    }

    FixtureOutcome {
        name: fixture.name.clone(),
        classification: fixture.classification,
        description: fixture.description.clone(),
        failures,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level test
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn run_jena_parity_matrix() {
    let fixtures = match discover_fixtures() {
        Ok(f) => f,
        Err(e) => panic!("fixture discovery failed: {e}"),
    };

    assert!(
        !fixtures.is_empty(),
        "expected at least one Jena CLI parity fixture under {}",
        fixture_root().display()
    );

    let mut spec_failures: Vec<FixtureOutcome> = Vec::new();
    let mut impl_warnings: Vec<FixtureOutcome> = Vec::new();
    let mut spec_passed = 0usize;
    let mut impl_passed = 0usize;

    for fixture in &fixtures {
        let outcome = run_fixture(fixture);
        if outcome.passed() {
            match outcome.classification {
                Classification::SpecRequired => spec_passed += 1,
                Classification::ImplDetail => impl_passed += 1,
            }
            continue;
        }
        match outcome.classification {
            Classification::SpecRequired => spec_failures.push(outcome),
            Classification::ImplDetail => impl_warnings.push(outcome),
        }
    }

    let total = fixtures.len();
    let spec_total = spec_passed + spec_failures.len();
    let impl_total = impl_passed + impl_warnings.len();

    eprintln!(
        "[jena_cli_parity] fixtures={total} spec_required={spec_total} (passed={spec_passed}) impl_detail={impl_total} (passed={impl_passed})"
    );

    for warning in &impl_warnings {
        eprintln!(
            "[jena_cli_parity][warn] impl-detail mismatch in '{}': {}",
            warning.name, warning.description
        );
        for failure in &warning.failures {
            eprintln!("  {failure}");
        }
    }

    if !spec_failures.is_empty() {
        let mut report = String::new();
        report.push_str(&format!(
            "{} spec-required fixture(s) failed (out of {} spec-required total):\n",
            spec_failures.len(),
            spec_total
        ));
        for failure in &spec_failures {
            report.push_str(&format!(
                "\n=== FIXTURE: {} ===\n  description: {}\n",
                failure.name, failure.description
            ));
            for issue in &failure.failures {
                report.push_str(&format!("  {issue}\n"));
            }
        }
        panic!("{report}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Self-checks for the harness itself
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fixture_root_exists() {
    let root = fixture_root();
    assert!(
        root.exists(),
        "expected fixture root at {} (did the vendored fixtures get checked in?)",
        root.display()
    );
}

#[test]
fn at_least_one_fixture_per_command_group() {
    let fixtures = discover_fixtures().expect("fixture discovery");
    let mut groups: BTreeMap<&str, usize> = BTreeMap::new();
    for fixture in &fixtures {
        if let Some(prefix) = fixture.name.split('-').next() {
            *groups.entry(prefix).or_insert(0) += 1;
        }
    }
    let required = ["query", "convert", "validate", "riot", "tdb"];
    for r in required {
        assert!(
            groups.contains_key(r),
            "no fixture group named '{r}-…'; have {:?}",
            groups.keys().collect::<Vec<_>>()
        );
    }
}

#[test]
fn classification_parse_round_trip() {
    assert_eq!(
        Classification::parse("spec-required").expect("spec"),
        Classification::SpecRequired
    );
    assert_eq!(
        Classification::parse("spec_required").expect("spec_underscore"),
        Classification::SpecRequired
    );
    assert_eq!(
        Classification::parse("impl-detail").expect("impl"),
        Classification::ImplDetail
    );
    assert!(Classification::parse("nonsense").is_err());
}

#[test]
fn stream_mode_parse() {
    assert_eq!(
        StreamMode::parse("exact").expect("exact"),
        StreamMode::Exact
    );
    assert_eq!(
        StreamMode::parse("contains").expect("c"),
        StreamMode::Contains
    );
    assert_eq!(StreamMode::parse("regex").expect("r"), StreamMode::Regex);
    assert_eq!(StreamMode::parse("absent").expect("a"), StreamMode::Absent);
    assert!(StreamMode::parse("foo").is_err());
}

#[test]
fn substitute_replaces_known_tokens() {
    let tmpdir = std::env::temp_dir()
        .join("oxirs_parity_x")
        .to_string_lossy()
        .into_owned();
    let mut vars = BTreeMap::new();
    vars.insert("TMPDIR", tmpdir.clone());
    vars.insert("FIXTURE", "/fix/abc".to_string());
    let out = substitute("--input ${FIXTURE}/data.ttl --out ${TMPDIR}/out.nt", &vars);
    assert_eq!(
        out,
        format!("--input /fix/abc/data.ttl --out {tmpdir}/out.nt")
    );
}

#[test]
fn substitute_leaves_unknown_tokens_intact() {
    let vars = BTreeMap::new();
    let out = substitute("hello ${UNKNOWN}", &vars);
    assert_eq!(out, "hello ${UNKNOWN}");
}

#[test]
fn check_stream_exact_passes_when_equal() {
    let spec = Some(StreamSpec {
        mode: StreamMode::Exact,
        expected: "hello\n".to_string(),
    });
    assert!(check_stream("stdout", "hello\n", &spec).is_ok());
}

#[test]
fn check_stream_exact_fails_on_mismatch() {
    let spec = Some(StreamSpec {
        mode: StreamMode::Exact,
        expected: "hello".to_string(),
    });
    assert!(check_stream("stdout", "world", &spec).is_err());
}

#[test]
fn check_stream_contains_collects_all_misses() {
    let spec = Some(StreamSpec {
        mode: StreamMode::Contains,
        expected: "alpha\nbeta\ngamma".to_string(),
    });
    let err = check_stream("stdout", "alpha and gamma", &spec).expect_err("expected miss");
    assert!(
        err.contains("beta"),
        "report should mention missing 'beta': {err}"
    );
}

#[test]
fn check_stream_regex_validates_each_line() {
    // Multiline-mode (?m) so that `$` matches before the trailing newline as
    // well as at end-of-input — matches the convention adopted by fixture
    // patterns in this matrix.
    let spec = Some(StreamSpec {
        mode: StreamMode::Regex,
        expected: r"(?m)^line: \d+$".to_string(),
    });
    assert!(check_stream("stdout", "line: 42\n", &spec).is_ok());
    assert!(check_stream("stdout", "no match here", &spec).is_err());
}

#[test]
fn check_stream_absent_passes_when_blank() {
    let spec = Some(StreamSpec {
        mode: StreamMode::Absent,
        expected: String::new(),
    });
    assert!(check_stream("stderr", "", &spec).is_ok());
    assert!(check_stream("stderr", "  \n  \n", &spec).is_ok());
}

#[test]
fn check_stream_absent_fails_when_content_present() {
    let spec = Some(StreamSpec {
        mode: StreamMode::Absent,
        expected: String::new(),
    });
    assert!(check_stream("stderr", "boom", &spec).is_err());
}

#[test]
fn check_stream_none_always_ok() {
    assert!(check_stream("stdout", "anything", &None).is_ok());
}

#[test]
fn fixture_load_validates_required_fields() {
    let dir = tempfile::tempdir().expect("tempdir");
    let manifest = dir.path().join("manifest.toml");
    fs::write(
        &manifest,
        // Missing exit_code
        r#"
description = "broken"
classification = "spec-required"
argv = ["query"]
"#,
    )
    .expect("write");
    let err = Fixture::load(dir.path().to_path_buf()).expect_err("should fail");
    assert!(err.contains("exit_code"), "error: {err}");
}

#[test]
fn parse_stream_spec_rejects_absent_with_expected() {
    let value: toml::Value = toml::from_str(
        r#"
mode = "absent"
expected = "should not be here"
"#,
    )
    .expect("toml parse");
    let err = parse_stream_spec(Some(&value), "test", "stdout").expect_err("should err");
    assert!(err.contains("absent"), "error: {err}");
}

#[test]
fn parse_stream_spec_rejects_exact_without_expected() {
    let value: toml::Value = toml::from_str(
        r#"
mode = "exact"
"#,
    )
    .expect("toml parse");
    let err = parse_stream_spec(Some(&value), "test", "stdout").expect_err("should err");
    assert!(err.contains("requires"), "error: {err}");
}
