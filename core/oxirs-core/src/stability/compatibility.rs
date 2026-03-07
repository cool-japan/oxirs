//! # OxiRS Version Compatibility Matrix
//!
//! This module tracks the version history of OxiRS, LTS policy, breaking changes,
//! and compatibility guarantees for dependent projects.

/// A single entry in the OxiRS version history.
#[derive(Debug, Clone)]
pub struct VersionEntry {
    /// The release version string (e.g. "1.0.0").
    pub version: &'static str,
    /// ISO 8601 release date (e.g. "2026-03-01").
    pub release_date: &'static str,
    /// Whether this is a Long-Term Support release.
    pub lts: bool,
    /// End-of-life date for security patches (None = not yet determined).
    pub supported_until: Option<&'static str>,
    /// List of breaking API changes introduced in this version.
    pub breaking_changes: Vec<&'static str>,
    /// APIs deprecated in this version.
    pub deprecated_in: Vec<&'static str>,
}

impl VersionEntry {
    /// Returns true if this version is currently within its support window.
    ///
    /// Uses a conservative heuristic: returns true unless `supported_until` is set
    /// and is known to be in the past (hardcoded reference date: 2026-02-24).
    pub fn is_supported(&self) -> bool {
        match self.supported_until {
            None => true,
            Some(eol) => eol >= "2026-02-24",
        }
    }

    /// Returns the number of breaking changes in this release.
    pub fn breaking_change_count(&self) -> usize {
        self.breaking_changes.len()
    }

    /// Returns the number of deprecations in this release.
    pub fn deprecation_count(&self) -> usize {
        self.deprecated_in.len()
    }
}

/// OxiRS LTS policy constants.
pub struct LtsPolicy;

impl LtsPolicy {
    /// LTS releases receive security patches for this many months after EOL of next LTS.
    pub const SECURITY_PATCH_MONTHS: u32 = 24;

    /// LTS releases receive bug-fix patches for this many months.
    pub const BUG_FIX_PATCH_MONTHS: u32 = 12;

    /// Breaking changes are only introduced in major versions.
    pub const BREAKING_CHANGES_IN_MAJOR_ONLY: bool = true;

    /// Security patches are backported to the current LTS series.
    pub const SECURITY_BACKPORT_TO_LTS: bool = true;

    /// API deprecations must survive at least one full minor version before removal.
    pub const MIN_DEPRECATION_NOTICE_VERSIONS: u32 = 1;
}

/// The version compatibility matrix for OxiRS.
#[derive(Debug)]
pub struct CompatibilityMatrix {
    /// All known OxiRS releases, ordered from oldest to newest.
    pub versions: Vec<VersionEntry>,
}

impl CompatibilityMatrix {
    /// Creates an empty compatibility matrix.
    pub fn new() -> Self {
        Self {
            versions: Vec::new(),
        }
    }

    /// Returns the canonical OxiRS release history.
    pub fn oxirs_history() -> Self {
        let mut matrix = Self::new();

        matrix.versions.push(VersionEntry {
            version: "0.1.0",
            release_date: "2025-12-01",
            lts: false,
            supported_until: Some("2026-03-01"),
            breaking_changes: vec![],
            deprecated_in: vec![],
        });

        matrix.versions.push(VersionEntry {
            version: "0.1.1",
            release_date: "2025-12-15",
            lts: false,
            supported_until: Some("2026-03-01"),
            breaking_changes: vec![],
            deprecated_in: vec![],
        });

        matrix.versions.push(VersionEntry {
            version: "0.1.2",
            release_date: "2026-01-05",
            lts: false,
            supported_until: Some("2026-03-01"),
            breaking_changes: vec![],
            deprecated_in: vec![],
        });

        matrix.versions.push(VersionEntry {
            version: "0.2.0",
            release_date: "2026-02-24",
            lts: false,
            supported_until: Some("2026-06-01"),
            breaking_changes: vec![
                "oxirs-star: Renamed Iri to NamedNode for consistency with RDF 1.2 spec",
                "oxirs-vec: Removed persistence.rs in favour of modular persistence/ directory",
                "oxirs-wasm: Removed monolithic parser.rs, store.rs in favour of sub-modules",
                "oxirs-stream: Removed state.rs; state management moved to state/ sub-module",
            ],
            deprecated_in: vec!["RdfStore::legacy_query() - use RdfStore::query() instead"],
        });

        matrix.versions.push(VersionEntry {
            version: "0.3.0",
            release_date: "2026-04-01",
            lts: false,
            supported_until: Some("2026-09-01"),
            breaking_changes: vec![],
            deprecated_in: vec![],
        });

        matrix.versions.push(VersionEntry {
            version: "1.0.0",
            release_date: "2026-06-01",
            lts: true,
            supported_until: Some("2028-06-01"),
            breaking_changes: vec![
                "Minimum Rust edition raised to 2024",
                "All previously Deprecated APIs removed from public surface",
                "WASM API now requires explicit initialisation via oxirs_wasm::init()",
                "StabilityLevel enum is now #[non_exhaustive] to allow future variants",
            ],
            deprecated_in: vec![],
        });

        matrix
    }

    /// Returns only LTS releases from the history.
    pub fn lts_releases(&self) -> Vec<&VersionEntry> {
        self.versions.iter().filter(|v| v.lts).collect()
    }

    /// Returns only currently supported releases.
    pub fn supported_releases(&self) -> Vec<&VersionEntry> {
        self.versions.iter().filter(|v| v.is_supported()).collect()
    }

    /// Returns the latest release entry, if any.
    pub fn latest(&self) -> Option<&VersionEntry> {
        self.versions.last()
    }

    /// Finds a release by exact version string.
    pub fn find_version(&self, version: &str) -> Option<&VersionEntry> {
        self.versions.iter().find(|v| v.version == version)
    }

    /// Returns all releases that introduced at least one breaking change.
    pub fn releases_with_breaking_changes(&self) -> Vec<&VersionEntry> {
        self.versions
            .iter()
            .filter(|v| !v.breaking_changes.is_empty())
            .collect()
    }

    /// Returns all releases that deprecated at least one API.
    pub fn releases_with_deprecations(&self) -> Vec<&VersionEntry> {
        self.versions
            .iter()
            .filter(|v| !v.deprecated_in.is_empty())
            .collect()
    }

    /// Generates a Markdown compatibility report.
    pub fn generate_report(&self) -> String {
        let mut report = String::with_capacity(4096);
        report.push_str("# OxiRS Version Compatibility Matrix\n\n");
        report.push_str("## LTS Policy\n\n");
        report.push_str(&format!(
            "- Security patches backported to LTS: {}\n",
            LtsPolicy::SECURITY_BACKPORT_TO_LTS
        ));
        report.push_str(&format!(
            "- Security patch support window: {} months\n",
            LtsPolicy::SECURITY_PATCH_MONTHS
        ));
        report.push_str(&format!(
            "- Bug-fix support window: {} months\n",
            LtsPolicy::BUG_FIX_PATCH_MONTHS
        ));
        report.push_str(&format!(
            "- Breaking changes in major versions only: {}\n",
            LtsPolicy::BREAKING_CHANGES_IN_MAJOR_ONLY
        ));
        report.push_str(&format!(
            "- Minimum deprecation notice: {} minor version(s)\n\n",
            LtsPolicy::MIN_DEPRECATION_NOTICE_VERSIONS
        ));

        report.push_str("## Release History\n\n");
        report.push_str("| Version | Release Date | LTS | Supported Until | Breaking Changes | Deprecations |\n");
        report.push_str(
            "|---------|-------------|-----|----------------|-----------------|-------------|\n",
        );
        for v in &self.versions {
            let lts_flag = if v.lts { "YES" } else { "No" };
            let eol = v.supported_until.unwrap_or("TBD");
            report.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                v.version,
                v.release_date,
                lts_flag,
                eol,
                v.breaking_changes.len(),
                v.deprecated_in.len()
            ));
        }
        report.push('\n');

        // Breaking changes detail
        for v in &self.versions {
            if !v.breaking_changes.is_empty() {
                report.push_str(&format!("### Breaking changes in {}\n\n", v.version));
                for change in &v.breaking_changes {
                    report.push_str(&format!("- {change}\n"));
                }
                report.push('\n');
            }
        }

        report
    }
}

impl Default for CompatibilityMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix() -> CompatibilityMatrix {
        CompatibilityMatrix::oxirs_history()
    }

    #[test]
    fn test_matrix_is_non_empty() {
        assert!(!matrix().versions.is_empty());
    }

    #[test]
    fn test_matrix_has_v1_lts() {
        let m = matrix();
        let lts = m.lts_releases();
        assert!(!lts.is_empty(), "Should have at least one LTS release");
        assert!(lts.iter().any(|v| v.version == "1.0.0"));
    }

    #[test]
    fn test_v1_is_lts() {
        let m = matrix();
        let v1 = m.find_version("1.0.0");
        assert!(v1.is_some());
        assert!(v1.unwrap().lts);
    }

    #[test]
    fn test_v1_supported_until_set() {
        let m = matrix();
        let v1 = m.find_version("1.0.0").unwrap();
        assert!(v1.supported_until.is_some());
    }

    #[test]
    fn test_v1_supported_until_two_years() {
        let m = matrix();
        let v1 = m.find_version("1.0.0").unwrap();
        let eol = v1.supported_until.unwrap();
        assert!(
            eol >= "2028-01-01",
            "v1.0.0 LTS should be supported at least until 2028"
        );
    }

    #[test]
    fn test_v010_exists() {
        assert!(matrix().find_version("0.1.0").is_some());
    }

    #[test]
    fn test_v020_exists() {
        assert!(matrix().find_version("0.2.0").is_some());
    }

    #[test]
    fn test_v010_is_not_lts() {
        let m = matrix();
        let v = m.find_version("0.1.0").unwrap();
        assert!(!v.lts);
    }

    #[test]
    fn test_v020_has_breaking_changes() {
        let m = matrix();
        let v = m.find_version("0.2.0").unwrap();
        assert!(!v.breaking_changes.is_empty());
    }

    #[test]
    fn test_v010_has_no_breaking_changes() {
        let m = matrix();
        let v = m.find_version("0.1.0").unwrap();
        assert!(v.breaking_changes.is_empty());
    }

    #[test]
    fn test_lts_policy_constants() {
        assert_eq!(LtsPolicy::SECURITY_PATCH_MONTHS, 24);
        assert_eq!(LtsPolicy::BUG_FIX_PATCH_MONTHS, 12);
        const _: () = assert!(LtsPolicy::BREAKING_CHANGES_IN_MAJOR_ONLY);
        const _: () = assert!(LtsPolicy::SECURITY_BACKPORT_TO_LTS);
    }

    #[test]
    fn test_latest_is_v100() {
        let m = matrix();
        let latest = m.latest().unwrap();
        assert_eq!(latest.version, "1.0.0");
    }

    #[test]
    fn test_releases_with_breaking_changes_non_empty() {
        let m = matrix();
        assert!(!m.releases_with_breaking_changes().is_empty());
    }

    #[test]
    fn test_releases_with_deprecations_non_empty() {
        let m = matrix();
        assert!(!m.releases_with_deprecations().is_empty());
    }

    #[test]
    fn test_supported_releases_non_empty() {
        let m = matrix();
        assert!(!m.supported_releases().is_empty());
    }

    #[test]
    fn test_find_nonexistent_version_returns_none() {
        let m = matrix();
        assert!(m.find_version("99.99.99").is_none());
    }

    #[test]
    fn test_version_entry_breaking_change_count() {
        let m = matrix();
        let v = m.find_version("0.2.0").unwrap();
        assert_eq!(v.breaking_change_count(), v.breaking_changes.len());
    }

    #[test]
    fn test_version_entry_deprecation_count() {
        let m = matrix();
        let v = m.find_version("0.2.0").unwrap();
        assert_eq!(v.deprecation_count(), v.deprecated_in.len());
    }

    #[test]
    fn test_generate_report_non_empty() {
        assert!(!matrix().generate_report().is_empty());
    }

    #[test]
    fn test_generate_report_contains_lts_policy() {
        let report = matrix().generate_report();
        assert!(report.contains("LTS Policy"));
    }

    #[test]
    fn test_generate_report_contains_release_history() {
        let report = matrix().generate_report();
        assert!(report.contains("Release History"));
    }

    #[test]
    fn test_generate_report_contains_v1() {
        let report = matrix().generate_report();
        assert!(report.contains("1.0.0"));
    }

    #[test]
    fn test_default_matrix_is_empty() {
        let m = CompatibilityMatrix::default();
        assert!(m.versions.is_empty());
    }
}
