//! Tests for quality assessment core.

#[cfg(test)]
mod tests {
    use crate::quality::core_metrics::QualityAssessor;
    use crate::quality::core_types::{QualityIssueSeverity, QualityReport, QualityThresholds};

    #[test]
    fn test_quality_assessor_creation() {
        let assessor = QualityAssessor::new();
        assert!(assessor.config.enable_assessment);
        assert_eq!(assessor.config.quality_thresholds.min_completeness, 0.8);
        assert_eq!(assessor.config.max_issues_per_category, 50);
    }

    #[test]
    fn test_quality_report_creation() {
        let mut report = QualityReport::new();
        report.set_completeness_score(0.9);
        report.set_consistency_score(0.8);

        assert_eq!(report.completeness_score, 0.9);
        assert_eq!(report.consistency_score, 0.8);
    }

    #[test]
    fn test_quality_thresholds() {
        let thresholds = QualityThresholds::default();
        assert_eq!(thresholds.min_completeness, 0.8);
        assert_eq!(thresholds.min_consistency, 0.9);
        assert_eq!(thresholds.max_duplicate_ratio, 0.05);
    }

    #[test]
    fn test_quality_issue_severity_conversion() {
        use oxirs_shacl::Severity;

        assert_eq!(
            QualityIssueSeverity::from_shacl_severity(&Severity::Violation),
            QualityIssueSeverity::High
        );
        assert_eq!(
            QualityIssueSeverity::from_shacl_severity(&Severity::Warning),
            QualityIssueSeverity::Medium
        );
        assert_eq!(
            QualityIssueSeverity::from_shacl_severity(&Severity::Info),
            QualityIssueSeverity::Info
        );
    }
}
