//! # ModelAnalytics - generate_html_report_group Methods
//!
//! This module contains method implementations for `ModelAnalytics`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::modelanalytics_type::ModelAnalytics;
use crate::analytics::Severity;
use std::collections::{HashMap, HashSet};

impl ModelAnalytics {
    /// Generate HTML report (for visualization)
    pub fn generate_html_report(&self) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Model Analytics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .score {{ font-size: 48px; font-weight: bold; color: {}; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .recommendation {{ padding: 10px; margin: 5px 0; background: #fff3cd; border-left: 4px solid #ffc107; }}
        .error {{ border-left-color: #dc3545; background: #f8d7da; }}
        .warning {{ border-left-color: #ffc107; background: #fff3cd; }}
    </style>
</head>
<body>
    <h1>Model Analytics Report</h1>
    <div class="section">
        <h2>Quality Score</h2>
        <div class="score">{:.1}/100</div>
    </div>
    <div class="section">
        <h2>Complexity Assessment</h2>
        <div class="metric">Overall Level: <strong>{:?}</strong></div>
        <div class="metric">Structural: {:.1}</div>
        <div class="metric">Cognitive: {:.1}</div>
        <div class="metric">Coupling: {:.1}</div>
    </div>
    <div class="section">
        <h2>Best Practices</h2>
        <div class="metric">Compliance: {:.1}% ({}/{})</div>
    </div>
    <div class="section">
        <h2>Recommendations</h2>
        {}
    </div>
</body>
</html>"#,
            if self.quality_score >= 80.0 {
                "#28a745"
            } else if self.quality_score >= 60.0 {
                "#ffc107"
            } else {
                "#dc3545"
            },
            self.quality_score,
            self.complexity_assessment.overall_level,
            self.complexity_assessment.structural,
            self.complexity_assessment.cognitive,
            self.complexity_assessment.coupling,
            self.best_practices.compliance_percentage,
            self.best_practices.passed_checks,
            self.best_practices.total_checks,
            self.recommendations
                .iter()
                .map(|r| format!(
                    r#"<div class="recommendation {}">{}: {} - {}</div>"#,
                    match r.severity {
                        Severity::Error | Severity::Critical => "error",
                        _ => "warning",
                    },
                    r.severity,
                    r.message,
                    r.suggested_action
                ))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}
