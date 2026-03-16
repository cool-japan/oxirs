//! Predictive Maintenance Module
//!
//! Provides health indicators, Remaining Useful Life (RUL) prediction,
//! maintenance scheduling, anomaly classification, and prognostic reports
//! for physical assets monitored by digital twins.

use crate::error::{PhysicsError, PhysicsResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// HealthIndicator
// ─────────────────────────────────────────────────────────────────────────────

/// A scalar health indicator computed from one or more sensor time-series.
///
/// `score` is in [0, 1]: 1.0 = perfect health, 0.0 = complete failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIndicator {
    /// Name of the indicator (e.g. "vibration_rms").
    pub name: String,
    /// Computed health score ∈ [0, 1].
    pub score: f64,
    /// Supporting evidence (sensor name → raw value).
    pub evidence: HashMap<String, f64>,
    /// Narrative description.
    pub description: String,
}

impl HealthIndicator {
    /// Compute a vibration RMS health indicator.
    ///
    /// `samples` is a slice of acceleration samples (m/s²).
    /// `nominal_rms` is the expected RMS in healthy conditions.
    /// `failure_rms` is the RMS at which the component is considered failed.
    pub fn from_vibration(
        samples: &[f64],
        nominal_rms: f64,
        failure_rms: f64,
    ) -> PhysicsResult<Self> {
        if samples.is_empty() {
            return Err(PhysicsError::ConstraintViolation(
                "vibration samples must not be empty".to_string(),
            ));
        }
        if failure_rms <= nominal_rms {
            return Err(PhysicsError::ConstraintViolation(
                "failure_rms must be greater than nominal_rms".to_string(),
            ));
        }

        let rms = compute_rms(samples);
        let score = compute_linear_score(rms, nominal_rms, failure_rms);

        let mut evidence = HashMap::new();
        evidence.insert("rms".to_string(), rms);
        evidence.insert(
            "peak".to_string(),
            samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );

        Ok(Self {
            name: "vibration_rms".to_string(),
            score,
            evidence,
            description: format!("Vibration RMS = {rms:.4} m/s² (score = {score:.3})"),
        })
    }

    /// Compute a temperature-based health indicator.
    ///
    /// `temperature` in Kelvin.  Healthy below `nominal_temp`; failed at `max_temp`.
    pub fn from_temperature(
        temperature: f64,
        nominal_temp: f64,
        max_temp: f64,
    ) -> PhysicsResult<Self> {
        if max_temp <= nominal_temp {
            return Err(PhysicsError::ConstraintViolation(
                "max_temp must be greater than nominal_temp".to_string(),
            ));
        }
        let score = compute_linear_score(temperature, nominal_temp, max_temp);
        let mut evidence = HashMap::new();
        evidence.insert("temperature_K".to_string(), temperature);
        Ok(Self {
            name: "thermal_health".to_string(),
            score,
            evidence,
            description: format!("Temperature = {temperature:.2} K (score = {score:.3})"),
        })
    }

    /// Compute a pressure-based health indicator.
    pub fn from_pressure(pressure: f64, nominal_pa: f64, max_pa: f64) -> PhysicsResult<Self> {
        if max_pa <= nominal_pa {
            return Err(PhysicsError::ConstraintViolation(
                "max_pa must be greater than nominal_pa".to_string(),
            ));
        }
        let score = compute_linear_score(pressure, nominal_pa, max_pa);
        let mut evidence = HashMap::new();
        evidence.insert("pressure_Pa".to_string(), pressure);
        Ok(Self {
            name: "pressure_health".to_string(),
            score,
            evidence,
            description: format!("Pressure = {pressure:.2} Pa (score = {score:.3})"),
        })
    }

    /// Aggregate multiple indicators into one composite score (arithmetic mean).
    pub fn aggregate(indicators: &[HealthIndicator]) -> PhysicsResult<Self> {
        if indicators.is_empty() {
            return Err(PhysicsError::ConstraintViolation(
                "no indicators to aggregate".to_string(),
            ));
        }
        let score = indicators.iter().map(|h| h.score).sum::<f64>() / indicators.len() as f64;
        let mut evidence = HashMap::new();
        for h in indicators {
            for (k, v) in &h.evidence {
                evidence.insert(format!("{}/{}", h.name, k), *v);
            }
        }
        Ok(Self {
            name: "composite_health".to_string(),
            score,
            evidence,
            description: format!(
                "Composite health from {} indicators (score = {score:.3})",
                indicators.len()
            ),
        })
    }
}

fn compute_rms(samples: &[f64]) -> f64 {
    let sum_sq: f64 = samples.iter().map(|x| x * x).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Returns a score in [0, 1]; 1.0 when `value ≤ nominal`, 0.0 when `value ≥ failure`.
fn compute_linear_score(value: f64, nominal: f64, failure: f64) -> f64 {
    if value <= nominal {
        1.0
    } else if value >= failure {
        0.0
    } else {
        1.0 - (value - nominal) / (failure - nominal)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Degradation Models & RUL Predictor
// ─────────────────────────────────────────────────────────────────────────────

/// Remaining Useful Life estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulEstimate {
    /// Predicted remaining time (e.g. hours or cycles).
    pub remaining_time: f64,
    /// Confidence interval half-width (±).
    pub confidence_interval: f64,
    /// Model name used.
    pub model_name: String,
}

/// Trait for degradation models used in RUL prediction.
pub trait DegradationModel: Send + Sync {
    /// Fit the model to the given time-series of health scores.
    ///
    /// `times` and `health_scores` must have the same length.
    fn fit(&mut self, times: &[f64], health_scores: &[f64]) -> PhysicsResult<()>;

    /// Predict the time at which health will reach `failure_threshold`.
    fn predict_rul(&self, current_time: f64, failure_threshold: f64) -> PhysicsResult<RulEstimate>;
}

/// Linear degradation: health(t) = a + b·t.
///
/// Fits using ordinary least squares.
#[derive(Debug, Default)]
pub struct LinearDegradationModel {
    intercept: f64,
    slope: f64,
    residual_std: f64,
    fitted: bool,
}

impl LinearDegradationModel {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DegradationModel for LinearDegradationModel {
    fn fit(&mut self, times: &[f64], health_scores: &[f64]) -> PhysicsResult<()> {
        if times.len() != health_scores.len() {
            return Err(PhysicsError::ConstraintViolation(
                "times and health_scores must have equal length".to_string(),
            ));
        }
        if times.len() < 2 {
            return Err(PhysicsError::ConstraintViolation(
                "need at least 2 data points for linear fit".to_string(),
            ));
        }

        let n = times.len() as f64;
        let sum_t: f64 = times.iter().sum();
        let sum_h: f64 = health_scores.iter().sum();
        let sum_tt: f64 = times.iter().map(|t| t * t).sum();
        let sum_th: f64 = times
            .iter()
            .zip(health_scores.iter())
            .map(|(t, h)| t * h)
            .sum();

        let denom = n * sum_tt - sum_t * sum_t;
        if denom.abs() < 1e-14 {
            return Err(PhysicsError::ConstraintViolation(
                "degenerate time series (all times equal)".to_string(),
            ));
        }

        self.slope = (n * sum_th - sum_t * sum_h) / denom;
        self.intercept = (sum_h - self.slope * sum_t) / n;

        // Compute residual std.
        let ss_res: f64 = times
            .iter()
            .zip(health_scores.iter())
            .map(|(t, h)| (h - (self.intercept + self.slope * t)).powi(2))
            .sum();
        self.residual_std = (ss_res / (n - 2.0).max(1.0)).sqrt();
        self.fitted = true;
        Ok(())
    }

    fn predict_rul(&self, current_time: f64, failure_threshold: f64) -> PhysicsResult<RulEstimate> {
        if !self.fitted {
            return Err(PhysicsError::ConstraintViolation(
                "model has not been fitted yet".to_string(),
            ));
        }
        if self.slope.abs() < 1e-14 {
            // Flat degradation — never reaches failure.
            return Ok(RulEstimate {
                remaining_time: f64::INFINITY,
                confidence_interval: 0.0,
                model_name: "LinearDegradation".to_string(),
            });
        }

        // t_failure = (threshold - intercept) / slope
        let t_failure = (failure_threshold - self.intercept) / self.slope;
        let rul = (t_failure - current_time).max(0.0);
        // 95% CI: ±1.96 * residual_std / |slope|
        let ci = 1.96 * self.residual_std / self.slope.abs();

        Ok(RulEstimate {
            remaining_time: rul,
            confidence_interval: ci.abs(),
            model_name: "LinearDegradation".to_string(),
        })
    }
}

/// Exponential degradation: health(t) = A · exp(−λ·t).
///
/// Fitted by log-linearizing: ln(health) = ln(A) − λ·t.
#[derive(Debug, Default)]
pub struct ExponentialDegradationModel {
    amplitude: f64,
    decay_rate: f64,
    residual_std_log: f64,
    fitted: bool,
}

impl ExponentialDegradationModel {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DegradationModel for ExponentialDegradationModel {
    fn fit(&mut self, times: &[f64], health_scores: &[f64]) -> PhysicsResult<()> {
        if times.len() != health_scores.len() || times.len() < 2 {
            return Err(PhysicsError::ConstraintViolation(
                "need matching arrays with ≥ 2 points".to_string(),
            ));
        }

        // Filter out non-positive health scores.
        let log_health: Vec<f64> = health_scores
            .iter()
            .map(|&h| if h > 0.0 { h.ln() } else { f64::NEG_INFINITY })
            .collect();

        let valid: Vec<(f64, f64)> = times
            .iter()
            .zip(log_health.iter())
            .filter(|(_, &lh)| lh.is_finite())
            .map(|(&t, &lh)| (t, lh))
            .collect();

        if valid.len() < 2 {
            return Err(PhysicsError::ConstraintViolation(
                "not enough positive health scores for exponential fit".to_string(),
            ));
        }

        let n = valid.len() as f64;
        let sum_t: f64 = valid.iter().map(|(t, _)| t).sum();
        let sum_lh: f64 = valid.iter().map(|(_, lh)| lh).sum();
        let sum_tt: f64 = valid.iter().map(|(t, _)| t * t).sum();
        let sum_tlh: f64 = valid.iter().map(|(t, lh)| t * lh).sum();

        let denom = n * sum_tt - sum_t * sum_t;
        if denom.abs() < 1e-14 {
            return Err(PhysicsError::ConstraintViolation(
                "degenerate time series".to_string(),
            ));
        }

        let slope = (n * sum_tlh - sum_t * sum_lh) / denom;
        let ln_a = (sum_lh - slope * sum_t) / n;

        self.amplitude = ln_a.exp();
        self.decay_rate = -slope; // positive decay rate

        let ss_res: f64 = valid
            .iter()
            .map(|(t, lh)| (lh - (ln_a + slope * t)).powi(2))
            .sum();
        self.residual_std_log = (ss_res / (n - 2.0).max(1.0)).sqrt();
        self.fitted = true;
        Ok(())
    }

    fn predict_rul(&self, current_time: f64, failure_threshold: f64) -> PhysicsResult<RulEstimate> {
        if !self.fitted {
            return Err(PhysicsError::ConstraintViolation(
                "model not fitted".to_string(),
            ));
        }
        if self.amplitude <= 0.0 || failure_threshold <= 0.0 {
            return Err(PhysicsError::ConstraintViolation(
                "amplitude and threshold must be positive".to_string(),
            ));
        }
        if self.decay_rate.abs() < 1e-14 {
            return Ok(RulEstimate {
                remaining_time: f64::INFINITY,
                confidence_interval: 0.0,
                model_name: "ExponentialDegradation".to_string(),
            });
        }

        // A·exp(−λ·t_fail) = threshold  ⟹  t_fail = ln(A/threshold) / λ
        let t_fail = (self.amplitude / failure_threshold).ln() / self.decay_rate;
        let rul = (t_fail - current_time).max(0.0);
        let ci = 1.96 * self.residual_std_log / self.decay_rate;

        Ok(RulEstimate {
            remaining_time: rul,
            confidence_interval: ci.abs(),
            model_name: "ExponentialDegradation".to_string(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Maintenance Schedule
// ─────────────────────────────────────────────────────────────────────────────

/// Priority of a maintenance task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaintenancePriority {
    Low,
    Medium,
    High,
    Critical,
}

/// A single maintenance task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceTask {
    pub description: String,
    pub priority: MaintenancePriority,
    /// Estimated hours until the task must be performed.
    pub due_in_hours: f64,
    /// Estimated duration of the task (hours).
    pub estimated_duration_hours: f64,
}

/// A time-ordered, priority-sorted maintenance schedule.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MaintenanceSchedule {
    pub tasks: Vec<MaintenanceTask>,
}

impl MaintenanceSchedule {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a task; tasks are kept sorted by (priority desc, due_in_hours asc).
    pub fn add_task(&mut self, task: MaintenanceTask) {
        self.tasks.push(task);
        self.tasks.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then(
                a.due_in_hours
                    .partial_cmp(&b.due_in_hours)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
    }

    /// Next task by priority (highest first).
    pub fn next_task(&self) -> Option<&MaintenanceTask> {
        self.tasks.first()
    }

    /// Tasks due within `horizon_hours`.
    pub fn tasks_due_within(&self, horizon_hours: f64) -> Vec<&MaintenanceTask> {
        self.tasks
            .iter()
            .filter(|t| t.due_in_hours <= horizon_hours)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyClassifier
// ─────────────────────────────────────────────────────────────────────────────

/// Category of detected anomaly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyCategory {
    Thermal,
    Mechanical,
    Electrical,
    Pressure,
    Unknown,
}

/// A classified anomaly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    pub category: AnomalyCategory,
    pub description: String,
    /// Severity in [0, 1].
    pub severity: f64,
    /// The sensor / quantity that triggered the classification.
    pub triggered_by: String,
}

/// Rule-based anomaly classifier operating on named sensor readings.
pub struct AnomalyClassifier {
    /// Thermal limits: quantity name → (nominal, alarm) thresholds in Kelvin.
    thermal_limits: HashMap<String, (f64, f64)>,
    /// Mechanical limits: quantity name → (nominal_rms, alarm_rms).
    mechanical_limits: HashMap<String, (f64, f64)>,
    /// Electrical limits: quantity name → (nominal, alarm).
    electrical_limits: HashMap<String, (f64, f64)>,
    /// Pressure limits: quantity name → (nominal, alarm).
    pressure_limits: HashMap<String, (f64, f64)>,
}

impl AnomalyClassifier {
    pub fn new() -> Self {
        Self {
            thermal_limits: HashMap::new(),
            mechanical_limits: HashMap::new(),
            electrical_limits: HashMap::new(),
            pressure_limits: HashMap::new(),
        }
    }

    pub fn add_thermal_limit(&mut self, quantity: impl Into<String>, nominal: f64, alarm: f64) {
        self.thermal_limits
            .insert(quantity.into(), (nominal, alarm));
    }

    pub fn add_mechanical_limit(&mut self, quantity: impl Into<String>, nominal: f64, alarm: f64) {
        self.mechanical_limits
            .insert(quantity.into(), (nominal, alarm));
    }

    pub fn add_electrical_limit(&mut self, quantity: impl Into<String>, nominal: f64, alarm: f64) {
        self.electrical_limits
            .insert(quantity.into(), (nominal, alarm));
    }

    pub fn add_pressure_limit(&mut self, quantity: impl Into<String>, nominal: f64, alarm: f64) {
        self.pressure_limits
            .insert(quantity.into(), (nominal, alarm));
    }

    /// Classify anomalies in `readings` (quantity name → value).
    pub fn classify(&self, readings: &HashMap<String, f64>) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        Self::check_limits(
            readings,
            &self.thermal_limits,
            AnomalyCategory::Thermal,
            "thermal",
            &mut anomalies,
        );
        Self::check_limits(
            readings,
            &self.mechanical_limits,
            AnomalyCategory::Mechanical,
            "mechanical",
            &mut anomalies,
        );
        Self::check_limits(
            readings,
            &self.electrical_limits,
            AnomalyCategory::Electrical,
            "electrical",
            &mut anomalies,
        );
        Self::check_limits(
            readings,
            &self.pressure_limits,
            AnomalyCategory::Pressure,
            "pressure",
            &mut anomalies,
        );

        anomalies
    }

    fn check_limits(
        readings: &HashMap<String, f64>,
        limits: &HashMap<String, (f64, f64)>,
        category: AnomalyCategory,
        label: &str,
        out: &mut Vec<Anomaly>,
    ) {
        for (qty, &(nominal, alarm)) in limits {
            if let Some(&value) = readings.get(qty) {
                if value > nominal {
                    let severity = compute_linear_score(value, nominal, alarm);
                    // Invert: score → 0 means severe, severity 1 = fully alarmed.
                    let severity = 1.0 - severity;
                    let description = format!(
                        "{label} anomaly on `{qty}`: value={value:.4} > nominal={nominal:.4} (alarm={alarm:.4})"
                    );
                    out.push(Anomaly {
                        category: category.clone(),
                        description,
                        severity: severity.clamp(0.0, 1.0),
                        triggered_by: qty.clone(),
                    });
                }
            }
        }
    }
}

impl Default for AnomalyClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PrognosticReport
// ─────────────────────────────────────────────────────────────────────────────

/// Failure mode identified in the prognostic report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureMode {
    pub name: String,
    pub probability: f64,
    pub description: String,
}

/// Comprehensive prognostic report combining health, RUL, anomalies, and tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrognosticReport {
    /// Aggregate health score ∈ [0, 1].
    pub health_score: f64,
    /// Remaining useful life estimate.
    pub rul_estimate: Option<RulEstimate>,
    /// Identified failure modes.
    pub failure_modes: Vec<FailureMode>,
    /// Recommended maintenance tasks.
    pub maintenance_tasks: MaintenanceSchedule,
    /// Detected anomalies.
    pub anomalies: Vec<Anomaly>,
}

impl PrognosticReport {
    /// Build a report from constituent parts.
    pub fn new(
        health_indicators: &[HealthIndicator],
        rul_estimate: Option<RulEstimate>,
        anomalies: Vec<Anomaly>,
    ) -> PhysicsResult<Self> {
        let health_score = if health_indicators.is_empty() {
            1.0
        } else {
            health_indicators.iter().map(|h| h.score).sum::<f64>() / health_indicators.len() as f64
        };

        // Derive failure modes from anomalies and health score.
        let mut failure_modes = Vec::new();
        if health_score < 0.3 {
            failure_modes.push(FailureMode {
                name: "ImmidentFailure".to_string(),
                probability: 1.0 - health_score,
                description: "Health score critically low; failure imminent".to_string(),
            });
        }
        for anomaly in &anomalies {
            if anomaly.severity > 0.7 {
                failure_modes.push(FailureMode {
                    name: format!("{:?}Failure", anomaly.category),
                    probability: anomaly.severity,
                    description: anomaly.description.clone(),
                });
            }
        }

        // Build maintenance schedule from RUL and anomalies.
        let mut schedule = MaintenanceSchedule::new();
        if let Some(ref rul) = rul_estimate {
            let priority = if rul.remaining_time < 24.0 {
                MaintenancePriority::Critical
            } else if rul.remaining_time < 168.0 {
                MaintenancePriority::High
            } else {
                MaintenancePriority::Medium
            };
            schedule.add_task(MaintenanceTask {
                description: format!(
                    "Inspect before estimated failure (RUL: {:.1} h ± {:.1})",
                    rul.remaining_time, rul.confidence_interval
                ),
                priority,
                due_in_hours: rul.remaining_time * 0.8,
                estimated_duration_hours: 2.0,
            });
        }

        Ok(Self {
            health_score,
            rul_estimate,
            failure_modes,
            maintenance_tasks: schedule,
            anomalies,
        })
    }

    /// Returns `true` when no immediate action is required.
    pub fn is_healthy(&self) -> bool {
        self.health_score > 0.6
            && self
                .maintenance_tasks
                .next_task()
                .map(|t| t.priority < MaintenancePriority::High)
                .unwrap_or(true)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── HealthIndicator ───────────────────────────────────────────────────────

    #[test]
    fn health_indicator_vibration_nominal() {
        let samples = vec![0.1_f64; 100];
        let hi = HealthIndicator::from_vibration(&samples, 0.5, 2.0).expect("should succeed");
        // RMS of 0.1 < nominal 0.5 → perfect health.
        assert!((hi.score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn health_indicator_vibration_degraded() {
        let samples = vec![1.25_f64; 100];
        // nominal=0.5, failure=2.0 → score = 1 - (1.25-0.5)/(2.0-0.5) = 0.5
        let hi = HealthIndicator::from_vibration(&samples, 0.5, 2.0).expect("should succeed");
        assert!((hi.score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn health_indicator_temperature() {
        let hi = HealthIndicator::from_temperature(500.0, 400.0, 600.0).expect("should succeed");
        // score = 1 - (500-400)/(600-400) = 0.5
        assert!((hi.score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn health_indicator_aggregate() {
        let h1 = HealthIndicator::from_temperature(300.0, 400.0, 600.0).expect("should succeed"); // score 1.0
        let h2 = HealthIndicator::from_temperature(500.0, 400.0, 600.0).expect("should succeed"); // score 0.5
        let agg = HealthIndicator::aggregate(&[h1, h2]).expect("should succeed");
        assert!((agg.score - 0.75).abs() < 1e-6);
    }

    // ── LinearDegradationModel ────────────────────────────────────────────────

    #[test]
    fn linear_degradation_fit_and_rul() {
        // Perfect linear degradation: h(t) = 1.0 - 0.01·t
        let times: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        let scores: Vec<f64> = times.iter().map(|t| 1.0 - 0.01 * t).collect();

        let mut model = LinearDegradationModel::new();
        model.fit(&times, &scores).expect("should succeed");

        // At t=50 the health is 0.5; failure threshold 0.0 → failure at t=100.
        let rul = model.predict_rul(50.0, 0.0).expect("should succeed");
        assert!(
            (rul.remaining_time - 50.0).abs() < 0.1,
            "RUL: {}",
            rul.remaining_time
        );
    }

    // ── ExponentialDegradationModel ───────────────────────────────────────────

    #[test]
    fn exponential_degradation_fit_and_rul() {
        // Exponential: h(t) = 1.0 · exp(−0.05·t)
        let times: Vec<f64> = (0..=80).map(|i| i as f64 * 0.5).collect();
        let scores: Vec<f64> = times.iter().map(|t| (-0.05 * t).exp()).collect();

        let mut model = ExponentialDegradationModel::new();
        model.fit(&times, &scores).expect("should succeed");

        // At t=0; failure when h=0.05 → t_fail = ln(1/0.05)/0.05 ≈ 59.9.
        let rul = model.predict_rul(0.0, 0.05).expect("should succeed");
        let expected = (1.0_f64 / 0.05).ln() / 0.05;
        assert!(
            (rul.remaining_time - expected).abs() < 1.0,
            "RUL: {}",
            rul.remaining_time
        );
    }

    // ── AnomalyClassifier ─────────────────────────────────────────────────────

    #[test]
    fn anomaly_classifier_thermal_detection() {
        let mut clf = AnomalyClassifier::new();
        clf.add_thermal_limit("temperature", 350.0, 500.0);

        let mut readings = HashMap::new();
        readings.insert("temperature".to_string(), 450.0);

        let anomalies = clf.classify(&readings);
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].category, AnomalyCategory::Thermal);
        // severity = 1 - (1 - (450-350)/(500-350)) = (450-350)/(500-350) ≈ 0.666
        assert!(anomalies[0].severity > 0.0);
    }

    #[test]
    fn anomaly_classifier_no_anomaly_within_nominal() {
        let mut clf = AnomalyClassifier::new();
        clf.add_mechanical_limit("vibration_rms", 0.5, 2.0);

        let mut readings = HashMap::new();
        readings.insert("vibration_rms".to_string(), 0.3);

        let anomalies = clf.classify(&readings);
        assert!(anomalies.is_empty());
    }

    // ── PrognosticReport ──────────────────────────────────────────────────────

    #[test]
    fn prognostic_report_healthy() {
        let hi = HealthIndicator::from_temperature(310.0, 400.0, 600.0).expect("should succeed");
        let report = PrognosticReport::new(&[hi], None, Vec::new()).expect("should succeed");
        assert!(report.health_score > 0.9);
        assert!(report.is_healthy());
    }
}
