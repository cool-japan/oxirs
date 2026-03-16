//! Digital Twin Management
//!
//! Full implementation of Digital Twin synchronization, sensor/actuator management,
//! and DTDL (Digital Twin Definition Language) v2 support.
//!
//! ## State model
//!
//! Each `DigitalTwin` maintains two complementary state maps:
//!
//! - **`model_state`**: the digital model's computed / predicted values.
//!   These are set by actuator commands, simulation outputs, or explicit
//!   initialisation, but are *not* overwritten by raw sensor data.
//! - **`state`**: the merged view used externally.  After a sync it
//!   reflects the sensor-reported physical reality.
//!
//! Deviation analysis in [`DigitalTwinManager::synchronize`] always compares
//! the *physical* sensor reading against the *digital* `model_state`, so a
//! deviation is reported even when both values have been observed.

use crate::error::{PhysicsError, PhysicsResult};
use crate::simulation::SimulationParameters;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────────────────────
// Core data types
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque identifier for a registered digital twin.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TwinId(String);

impl TwinId {
    /// Create a new random TwinId.
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Return the inner string representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for TwinId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TwinId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sensor / Actuator primitives
// ─────────────────────────────────────────────────────────────────────────────

/// A single sensor measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    /// Unique sensor identifier.
    pub id: String,
    /// Physical quantity measured (e.g. "temperature", "pressure").
    pub quantity: String,
    /// Numeric value.
    pub value: f64,
    /// SI unit string (e.g. "K", "Pa").
    pub unit: String,
    /// Unix-epoch timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Measurement uncertainty (1-sigma), if known.
    pub uncertainty: Option<f64>,
}

impl SensorReading {
    /// Create a new sensor reading with the current time.
    pub fn new(
        id: impl Into<String>,
        quantity: impl Into<String>,
        value: f64,
        unit: impl Into<String>,
    ) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            id: id.into(),
            quantity: quantity.into(),
            value,
            unit: unit.into(),
            timestamp_ms,
            uncertainty: None,
        }
    }

    /// Builder: set uncertainty.
    pub fn with_uncertainty(mut self, sigma: f64) -> Self {
        self.uncertainty = Some(sigma);
        self
    }
}

/// A command sent to an actuator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActuatorCommand {
    /// Unique actuator identifier.
    pub id: String,
    /// Target physical quantity (e.g. "set_temperature").
    pub target_quantity: String,
    /// Desired numeric value.
    pub target_value: f64,
    /// Whether the command has been applied.
    pub applied: bool,
}

impl ActuatorCommand {
    /// Create a new unapplied command.
    pub fn new(
        id: impl Into<String>,
        target_quantity: impl Into<String>,
        target_value: f64,
    ) -> Self {
        Self {
            id: id.into(),
            target_quantity: target_quantity.into(),
            target_value,
            applied: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Synchronisation report
// ─────────────────────────────────────────────────────────────────────────────

/// Deviation of a single quantity after sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantityDeviation {
    /// Quantity name.
    pub quantity: String,
    /// Physical-world value (from latest sensor).
    pub physical_value: f64,
    /// Digital model value *before* this sync.
    pub digital_value: f64,
    /// Absolute deviation.
    pub deviation: f64,
}

/// Anomaly detected during synchronisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncAnomaly {
    /// Quantity that triggered the anomaly.
    pub quantity: String,
    /// Human-readable description.
    pub description: String,
    /// Severity: 0.0 (info) … 1.0 (critical).
    pub severity: f64,
}

/// Result of a bidirectional sync operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncReport {
    /// Names of quantities that were successfully synchronised.
    pub synced_quantities: Vec<String>,
    /// Per-quantity deviations.
    pub deviations: Vec<QuantityDeviation>,
    /// Detected anomalies.
    pub anomalies: Vec<SyncAnomaly>,
    /// Unix-epoch time of the sync in milliseconds.
    pub sync_time_ms: u64,
}

impl SyncReport {
    fn new_now() -> Self {
        let sync_time_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            synced_quantities: Vec::new(),
            deviations: Vec::new(),
            anomalies: Vec::new(),
            sync_time_ms,
        }
    }

    /// Returns `true` when no anomalies with severity ≥ `threshold` were found.
    pub fn is_healthy(&self, threshold: f64) -> bool {
        self.anomalies.iter().all(|a| a.severity < threshold)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DigitalTwin
// ─────────────────────────────────────────────────────────────────────────────

/// Anomaly detection threshold: alert when deviation exceeds this fraction of
/// the model value (or this absolute amount when the model value is ~0).
const ANOMALY_DEVIATION_THRESHOLD: f64 = 0.15;

/// Core digital twin linking a physical asset to its simulation state.
///
/// Two state maps are maintained:
/// - `model_state` — the digital model's predicted / set values.
/// - `state` — the externally-visible merged view; updated by sync.
pub struct DigitalTwin {
    /// IRI of the physical entity this twin represents.
    pub entity_iri: String,
    /// Human-readable type label.
    pub twin_type: String,
    /// Current model state: quantity-name → digital-model value.
    /// Updated by actuator commands and explicit `set_model_value` calls.
    pub model_state: HashMap<String, f64>,
    /// Merged state: updated to reflect sensor values after each sync.
    pub state: HashMap<String, f64>,
    /// Buffered sensor readings (most-recent per sensor id).
    pub sensors: Vec<SensorReading>,
    /// Pending actuator commands.
    pub actuators: Vec<ActuatorCommand>,
    /// Wall-clock time of the last successful sync.
    pub last_sync: SystemTime,
}

impl DigitalTwin {
    /// Construct a new, empty digital twin.
    pub fn new(entity_iri: impl Into<String>, twin_type: impl Into<String>) -> Self {
        Self {
            entity_iri: entity_iri.into(),
            twin_type: twin_type.into(),
            model_state: HashMap::new(),
            state: HashMap::new(),
            sensors: Vec::new(),
            actuators: Vec::new(),
            last_sync: UNIX_EPOCH,
        }
    }

    /// Convenience accessor.
    pub fn entity_iri(&self) -> &str {
        &self.entity_iri
    }

    /// Explicitly set a value in the digital model (without a sensor reading).
    pub fn set_model_value(&mut self, quantity: impl Into<String>, value: f64) {
        let qty = quantity.into();
        self.model_state.insert(qty.clone(), value);
        self.state.insert(qty, value);
    }

    /// Register a sensor reading.
    ///
    /// The reading is stored (replacing any previous reading from the same
    /// sensor id).  The `state` map is **not** updated here; call
    /// [`DigitalTwin::apply_sensor_to_state`] or run
    /// [`DigitalTwinManager::synchronize`] to merge sensor data into `state`.
    pub fn buffer_sensor(&mut self, reading: SensorReading) {
        self.sensors.retain(|s| s.id != reading.id);
        self.sensors.push(reading);
    }

    /// Apply all buffered sensor readings into `state` (merges physical reality).
    /// This does **not** update `model_state`.
    pub fn apply_sensor_to_state(&mut self) {
        for sensor in &self.sensors {
            self.state.insert(sensor.quantity.clone(), sensor.value);
        }
    }

    /// Queue an actuator command (replaces existing command for the same id).
    /// Also updates `model_state` immediately so sync can detect deviations.
    pub fn queue_actuator(&mut self, cmd: ActuatorCommand) {
        self.model_state
            .insert(cmd.target_quantity.clone(), cmd.target_value);
        self.actuators.retain(|a| a.id != cmd.id);
        self.actuators.push(cmd);
    }

    /// Flush all pending actuator commands: apply their target values to both
    /// `model_state` and `state`.
    pub fn apply_pending_actuators(&mut self) {
        for cmd in &mut self.actuators {
            if !cmd.applied {
                self.model_state
                    .insert(cmd.target_quantity.clone(), cmd.target_value);
                self.state
                    .insert(cmd.target_quantity.clone(), cmd.target_value);
                cmd.applied = true;
            }
        }
    }

    /// Build `SimulationParameters` from current model state.
    pub async fn extract_simulation_params(&self) -> PhysicsResult<SimulationParameters> {
        let initial_conditions = self
            .model_state
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    crate::simulation::parameter_extraction::PhysicalQuantity {
                        value: *v,
                        unit: String::from("SI"),
                        uncertainty: None,
                    },
                )
            })
            .collect();

        Ok(SimulationParameters {
            entity_iri: self.entity_iri.clone(),
            simulation_type: self.twin_type.clone(),
            initial_conditions,
            boundary_conditions: Vec::new(),
            time_span: (0.0, 100.0),
            time_steps: 100,
            material_properties: HashMap::new(),
            constraints: Vec::new(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DigitalTwinManager
// ─────────────────────────────────────────────────────────────────────────────

/// Manages a collection of digital twins.
pub struct DigitalTwinManager {
    twins: HashMap<TwinId, DigitalTwin>,
    /// Anomaly threshold (fraction) used when building sync reports.
    anomaly_threshold: f64,
}

impl DigitalTwinManager {
    /// Create a new manager.
    pub fn new() -> Self {
        Self {
            twins: HashMap::new(),
            anomaly_threshold: ANOMALY_DEVIATION_THRESHOLD,
        }
    }

    /// Set the fraction above which deviations are flagged as anomalies.
    pub fn with_anomaly_threshold(mut self, threshold: f64) -> Self {
        self.anomaly_threshold = threshold;
        self
    }

    /// Register a new digital twin.
    ///
    /// `name` is a human-readable label; `model_uri` is the IRI of the physical
    /// entity. Returns the new twin's [`TwinId`].
    pub fn register(
        &mut self,
        name: impl Into<String>,
        model_uri: impl Into<String>,
    ) -> PhysicsResult<TwinId> {
        let id = TwinId::new();
        let twin = DigitalTwin::new(model_uri, name);
        self.twins.insert(id.clone(), twin);
        Ok(id)
    }

    /// Buffer a sensor reading on a twin.
    ///
    /// The reading is stored but **not** immediately applied to the twin's
    /// `state`.  Call `synchronize` to perform the full bidirectional sync.
    pub fn update_from_sensor(
        &mut self,
        twin_id: &TwinId,
        reading: SensorReading,
    ) -> PhysicsResult<()> {
        let twin = self
            .twins
            .get_mut(twin_id)
            .ok_or_else(|| PhysicsError::Internal(format!("twin not found: {twin_id}")))?;
        twin.buffer_sensor(reading);
        Ok(())
    }

    /// Return a snapshot of the current (merged) twin state.
    pub fn get_state(&self, twin_id: &TwinId) -> PhysicsResult<HashMap<String, f64>> {
        let twin = self
            .twins
            .get(twin_id)
            .ok_or_else(|| PhysicsError::Internal(format!("twin not found: {twin_id}")))?;
        Ok(twin.state.clone())
    }

    /// Perform a bidirectional synchronisation of the named twin.
    ///
    /// Steps:
    /// 1. Apply pending actuator commands to `model_state` and `state`.
    /// 2. Compute deviations: sensor value vs. `model_state`.
    /// 3. Flag deviations that exceed `anomaly_threshold` as anomalies.
    /// 4. Merge sensor readings into `state` (so `state` reflects reality).
    /// 5. Update `last_sync`.
    pub fn synchronize(&mut self, twin_id: &TwinId) -> PhysicsResult<SyncReport> {
        let threshold = self.anomaly_threshold;
        let twin = self
            .twins
            .get_mut(twin_id)
            .ok_or_else(|| PhysicsError::Internal(format!("twin not found: {twin_id}")))?;

        // Step 1: apply actuators.
        twin.apply_pending_actuators();

        // Step 2 & 3: compute deviations, flag anomalies.
        let mut report = SyncReport::new_now();

        for sensor in &twin.sensors {
            // Compare sensor (physical) against the *digital model* value.
            let digital_value = *twin.model_state.get(&sensor.quantity).unwrap_or(&0.0);
            let deviation = (sensor.value - digital_value).abs();
            let relative = if digital_value.abs() > 1e-12 {
                deviation / digital_value.abs()
            } else {
                // When model has no prediction (0.0), use absolute deviation.
                deviation
            };

            report.synced_quantities.push(sensor.quantity.clone());
            report.deviations.push(QuantityDeviation {
                quantity: sensor.quantity.clone(),
                physical_value: sensor.value,
                digital_value,
                deviation,
            });

            if relative > threshold {
                let severity = (relative / (threshold + f64::EPSILON)).clamp(0.0, 1.0);
                report.anomalies.push(SyncAnomaly {
                    quantity: sensor.quantity.clone(),
                    description: format!(
                        "Deviation {deviation:.4} ({:.1}%) exceeds threshold {:.1}%",
                        relative * 100.0,
                        threshold * 100.0
                    ),
                    severity,
                });
            }
        }

        // Step 4: merge sensor readings into state.
        twin.apply_sensor_to_state();

        // Step 5: update last_sync.
        twin.last_sync = SystemTime::now();

        Ok(report)
    }

    /// Immutable access to a twin (for inspection / testing).
    pub fn get_twin(&self, twin_id: &TwinId) -> Option<&DigitalTwin> {
        self.twins.get(twin_id)
    }

    /// Mutable access to a twin.
    pub fn get_twin_mut(&mut self, twin_id: &TwinId) -> Option<&mut DigitalTwin> {
        self.twins.get_mut(twin_id)
    }

    /// Number of registered twins.
    pub fn len(&self) -> usize {
        self.twins.len()
    }

    /// Returns `true` when no twins are registered.
    pub fn is_empty(&self) -> bool {
        self.twins.is_empty()
    }
}

impl Default for DigitalTwinManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DTDL – Digital Twin Definition Language v2 support
// ─────────────────────────────────────────────────────────────────────────────

/// A DTDL v2 telemetry entry (time-series data reported by the twin).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtdlTelemetry {
    /// DTDL `@id` or display name.
    pub name: String,
    /// Schema / data type hint (e.g. "double", "float", "integer").
    pub schema: String,
    /// Optional unit annotation.
    pub unit: Option<String>,
}

/// A DTDL v2 property (writable configuration / state).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtdlProperty {
    pub name: String,
    pub schema: String,
    pub writable: bool,
}

/// A DTDL v2 component reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtdlComponent {
    pub name: String,
    pub schema: String,
}

/// A DTDL v2 relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtdlRelationship {
    pub name: String,
    pub target: Option<String>,
    pub max_multiplicity: Option<u32>,
}

/// Simplified representation of a DTDL v2 Interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtdlModel {
    /// `@id` of the interface (e.g. `dtmi:com:example:Thermostat;1`).
    pub id: String,
    /// Display name.
    pub display_name: String,
    /// Telemetry channels.
    pub telemetry: Vec<DtdlTelemetry>,
    /// Properties (writable and read-only).
    pub properties: Vec<DtdlProperty>,
    /// Components.
    pub components: Vec<DtdlComponent>,
    /// Relationships.
    pub relationships: Vec<DtdlRelationship>,
}

impl DtdlModel {
    /// Create a minimal empty model.
    pub fn new(id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            telemetry: Vec::new(),
            properties: Vec::new(),
            components: Vec::new(),
            relationships: Vec::new(),
        }
    }
}

/// Parse a DTDL v2 JSON payload into a [`DtdlModel`].
///
/// Supports the minimal structure produced by Azure Digital Twins tooling:
/// ```json
/// {
///   "@context": "dtmi:dtdl:context;2",
///   "@type": "Interface",
///   "@id": "dtmi:com:example:Thermostat;1",
///   "displayName": "Thermostat",
///   "contents": [
///     { "@type": "Telemetry", "name": "temperature", "schema": "double", "unit": "degreeCelsius" },
///     { "@type": "Property",  "name": "targetTemperature", "schema": "double", "writable": true }
///   ]
/// }
/// ```
pub fn parse_dtdl_json(json: &str) -> PhysicsResult<DtdlModel> {
    let root: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| PhysicsError::Internal(format!("DTDL JSON parse error: {e}")))?;

    let id = root
        .get("@id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let display_name = root
        .get("displayName")
        .and_then(|v| v.as_str())
        .unwrap_or(&id)
        .to_string();

    let mut model = DtdlModel::new(id, display_name);

    if let Some(contents) = root.get("contents").and_then(|v| v.as_array()) {
        for item in contents {
            let type_tag = item
                .get("@type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_lowercase();
            let name = item
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let schema = item
                .get("schema")
                .and_then(|v| v.as_str())
                .unwrap_or("double")
                .to_string();

            match type_tag.as_str() {
                "telemetry" => {
                    let unit = item.get("unit").and_then(|v| v.as_str()).map(String::from);
                    model.telemetry.push(DtdlTelemetry { name, schema, unit });
                }
                "property" => {
                    let writable = item
                        .get("writable")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    model.properties.push(DtdlProperty {
                        name,
                        schema,
                        writable,
                    });
                }
                "component" => {
                    model.components.push(DtdlComponent { name, schema });
                }
                "relationship" => {
                    let target = item
                        .get("target")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let max_multiplicity = item
                        .get("maxMultiplicity")
                        .and_then(|v| v.as_u64())
                        .map(|n| n as u32);
                    model.relationships.push(DtdlRelationship {
                        name,
                        target,
                        max_multiplicity,
                    });
                }
                _ => {
                    // unknown content type — skip gracefully
                }
            }
        }
    }

    Ok(model)
}

/// Convert a parsed [`DtdlModel`] into a [`DigitalTwin`].
///
/// Telemetry and properties are registered with a default model value of `0.0`;
/// the caller is expected to populate them via sensor updates and sync.
pub fn model_to_digital_twin(model: &DtdlModel) -> DigitalTwin {
    let mut twin = DigitalTwin::new(model.id.clone(), model.display_name.clone());

    for tel in &model.telemetry {
        twin.set_model_value(tel.name.clone(), 0.0);
    }
    for prop in &model.properties {
        twin.set_model_value(prop.name.clone(), 0.0);
    }

    twin
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reading(id: &str, qty: &str, value: f64) -> SensorReading {
        SensorReading::new(id, qty, value, "SI")
    }

    // ── TwinId ────────────────────────────────────────────────────────────────

    #[test]
    fn twin_id_uniqueness() {
        let a = TwinId::new();
        let b = TwinId::new();
        assert_ne!(a, b, "each TwinId must be unique");
    }

    #[test]
    fn twin_id_display() {
        let id = TwinId::new();
        assert!(!id.to_string().is_empty());
    }

    // ── DigitalTwin ───────────────────────────────────────────────────────────

    #[test]
    fn digital_twin_creation() {
        let twin = DigitalTwin::new("urn:example:motor:1", "ElectricMotor");
        assert_eq!(twin.entity_iri(), "urn:example:motor:1");
        assert_eq!(twin.twin_type, "ElectricMotor");
        assert!(twin.state.is_empty());
    }

    #[test]
    fn digital_twin_sensor_buffered() {
        let mut twin = DigitalTwin::new("urn:example:motor:1", "ElectricMotor");
        twin.buffer_sensor(make_reading("s1", "temperature", 350.0));

        // Buffered only — state not yet updated.
        assert!(!twin.state.contains_key("temperature"));
        assert_eq!(twin.sensors.len(), 1);
    }

    #[test]
    fn digital_twin_sensor_apply_to_state() {
        let mut twin = DigitalTwin::new("urn:example:motor:1", "ElectricMotor");
        twin.buffer_sensor(make_reading("s1", "temperature", 350.0));
        twin.apply_sensor_to_state();

        assert_eq!(twin.state["temperature"], 350.0);
    }

    #[test]
    fn digital_twin_sensor_deduplication() {
        let mut twin = DigitalTwin::new("urn:example:motor:1", "ElectricMotor");
        twin.buffer_sensor(make_reading("s1", "temperature", 350.0));
        twin.buffer_sensor(make_reading("s1", "temperature", 370.0));

        // Only the latest reading for sensor s1 should be kept.
        assert_eq!(twin.sensors.len(), 1);
        assert_eq!(twin.sensors[0].value, 370.0);
    }

    #[test]
    fn digital_twin_actuator_command() {
        let mut twin = DigitalTwin::new("urn:example:motor:1", "ElectricMotor");
        twin.set_model_value("set_temperature", 300.0);
        twin.queue_actuator(ActuatorCommand::new("a1", "set_temperature", 400.0));
        twin.apply_pending_actuators();

        assert_eq!(twin.model_state["set_temperature"], 400.0);
        assert_eq!(twin.state["set_temperature"], 400.0);
        assert!(twin.actuators[0].applied);
    }

    // ── DigitalTwinManager ────────────────────────────────────────────────────

    #[test]
    fn manager_register_and_get_state() {
        let mut mgr = DigitalTwinManager::new();
        let id = mgr
            .register("Thermostat", "urn:example:thermostat:1")
            .expect("register failed");

        assert_eq!(mgr.len(), 1);

        // Set model prediction first.
        mgr.get_twin_mut(&id)
            .expect("should succeed")
            .set_model_value("temperature", 298.15);

        // Add sensor matching prediction.
        mgr.update_from_sensor(&id, SensorReading::new("t1", "temperature", 298.15, "K"))
            .expect("sensor update failed");

        // Sync to merge sensor into state.
        mgr.synchronize(&id).expect("sync failed");

        let state = mgr.get_state(&id).expect("get_state failed");
        assert!((state["temperature"] - 298.15).abs() < 1e-9);
    }

    #[test]
    fn manager_synchronise_no_anomaly() {
        let mut mgr = DigitalTwinManager::new();
        let id = mgr
            .register("Motor", "urn:example:motor:1")
            .expect("should succeed");

        // Set model prediction.
        mgr.get_twin_mut(&id)
            .expect("should succeed")
            .set_model_value("speed", 1500.0);

        // Sensor matches model prediction exactly.
        mgr.update_from_sensor(&id, SensorReading::new("s1", "speed", 1500.0, "rpm"))
            .expect("should succeed");

        let report = mgr.synchronize(&id).expect("sync failed");
        assert_eq!(report.synced_quantities.len(), 1);
        assert!(
            report.anomalies.is_empty(),
            "no anomaly expected when sensor matches model"
        );
    }

    #[test]
    fn manager_synchronise_with_anomaly() {
        let mut mgr = DigitalTwinManager::new().with_anomaly_threshold(0.05);
        let id = mgr
            .register("Sensor", "urn:example:s:1")
            .expect("should succeed");

        // Set digital model state to 300 K.
        mgr.get_twin_mut(&id)
            .expect("should succeed")
            .set_model_value("temperature", 300.0);

        // Push sensor with large deviation from model (>5%).
        mgr.update_from_sensor(&id, SensorReading::new("s1", "temperature", 400.0, "K"))
            .expect("should succeed");

        let report = mgr.synchronize(&id).expect("should succeed");
        assert!(
            !report.anomalies.is_empty(),
            "expected anomaly to be detected (model=300, sensor=400)"
        );
        assert_eq!(report.deviations[0].digital_value, 300.0);
        assert_eq!(report.deviations[0].physical_value, 400.0);
    }

    // ── DTDL ──────────────────────────────────────────────────────────────────

    #[test]
    fn dtdl_parse_minimal() {
        let json = r#"{
            "@context": "dtmi:dtdl:context;2",
            "@type": "Interface",
            "@id": "dtmi:com:example:Thermostat;1",
            "displayName": "Thermostat",
            "contents": [
                { "@type": "Telemetry", "name": "temperature", "schema": "double", "unit": "degreeCelsius" },
                { "@type": "Property", "name": "targetTemperature", "schema": "double", "writable": true }
            ]
        }"#;

        let model = parse_dtdl_json(json).expect("parse failed");
        assert_eq!(model.id, "dtmi:com:example:Thermostat;1");
        assert_eq!(model.display_name, "Thermostat");
        assert_eq!(model.telemetry.len(), 1);
        assert_eq!(model.telemetry[0].name, "temperature");
        assert_eq!(model.telemetry[0].unit.as_deref(), Some("degreeCelsius"));
        assert_eq!(model.properties.len(), 1);
        assert!(model.properties[0].writable);
    }

    #[test]
    fn dtdl_parse_with_relationship() {
        let json = r#"{
            "@id": "dtmi:com:example:Room;1",
            "displayName": "Room",
            "contents": [
                { "@type": "Relationship", "name": "contains", "target": "dtmi:com:example:Device;1", "maxMultiplicity": 10 }
            ]
        }"#;

        let model = parse_dtdl_json(json).expect("parse failed");
        assert_eq!(model.relationships.len(), 1);
        assert_eq!(model.relationships[0].name, "contains");
        assert_eq!(model.relationships[0].max_multiplicity, Some(10));
    }

    #[test]
    fn dtdl_model_to_twin_state_initialised() {
        let json = r#"{
            "@id": "dtmi:com:example:Device;1",
            "displayName": "Device",
            "contents": [
                { "@type": "Telemetry", "name": "voltage", "schema": "double" },
                { "@type": "Property",  "name": "setPoint", "schema": "double", "writable": false }
            ]
        }"#;

        let model = parse_dtdl_json(json).expect("should succeed");
        let twin = model_to_digital_twin(&model);

        assert!(twin.model_state.contains_key("voltage"));
        assert!(twin.model_state.contains_key("setPoint"));
        assert!(twin.state.contains_key("voltage"));
        assert_eq!(twin.entity_iri(), "dtmi:com:example:Device;1");
    }

    #[test]
    fn dtdl_parse_invalid_json_returns_error() {
        let result = parse_dtdl_json("{ not valid json }");
        assert!(result.is_err());
    }
}
