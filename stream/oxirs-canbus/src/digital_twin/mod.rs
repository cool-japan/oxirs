//! Automotive Digital Twin
//!
//! Provides a real-time, aggregated vehicle state model that integrates data
//! from both OBD-II diagnostic responses and raw CAN frames. Generates SAREF/SSN
//! Turtle triples for semantic web integration.
//!
//! # Design
//!
//! The digital twin maintains a per-VIN vehicle state that is updated as new
//! CAN frames and OBD-II responses arrive. This enables:
//!
//! - Real-time dashboard displays
//! - Anomaly detection using historical state comparison
//! - RDF/knowledge graph integration via SAREF/SSN/SOSA ontologies
//! - Fleet management across multiple vehicles
//!
//! # Example
//!
//! ```
//! use oxirs_canbus::digital_twin::{VehicleState, DigitalTwinManager};
//! use oxirs_canbus::obd2::ObdResponse;
//!
//! let mut manager = DigitalTwinManager::new();
//! let vin = "1HGCM82633A123456";
//!
//! // Update state from OBD-II response (engine RPM)
//! let rpm_resp = ObdResponse::new(0x0C, vec![0x1A, 0xF0]);
//! manager.update_state(vin, &rpm_resp);
//!
//! if let Some(state) = manager.get_state(vin) {
//!     println!("Engine RPM: {:?}", state.engine_rpm);
//! }
//! ```

use std::collections::HashMap;

use crate::obd2::{ObdPid, ObdResponse, ObdValue};
use crate::protocol::CanFrame;

/// OBD-II PID for vehicle speed in J1939-compatible CAN frames
/// Standard OBD-II response CAN ID range: 0x7E8 - 0x7EF
pub const OBD_RESPONSE_ID_MIN: u32 = 0x7E8;
/// Maximum OBD-II response CAN ID
pub const OBD_RESPONSE_ID_MAX: u32 = 0x7EF;

/// Known J1939 PGN for vehicle speed: CCVS (Cruise Control/Vehicle Speed) = 65265 (0xFEF1)
pub const J1939_PGN_CCVS: u32 = 65265;

/// Known J1939 PGN for engine control: EEC1 = 61444 (0xF004)
pub const J1939_PGN_EEC1: u32 = 61444;

/// Real-time aggregated vehicle state
///
/// Updated from OBD-II responses and raw CAN frames as data arrives.
/// All fields are `Option<f64>` because sensor data may not be available
/// for all vehicles or may not have been received yet.
#[derive(Debug, Clone, Default)]
pub struct VehicleState {
    /// Engine speed in RPM (from PID 0x0C)
    pub engine_rpm: Option<f64>,

    /// Vehicle speed in km/h (from PID 0x0D)
    pub speed_kmh: Option<f64>,

    /// Engine coolant temperature in °C (from PID 0x05)
    pub coolant_temp_c: Option<f64>,

    /// Throttle position in percent (from PID 0x11)
    pub throttle_pct: Option<f64>,

    /// Fuel tank level in percent (from PID 0x2F)
    pub fuel_level_pct: Option<f64>,

    /// Malfunction Indicator Lamp (MIL / Check Engine) state
    pub mil_on: bool,

    /// Active Diagnostic Trouble Codes
    pub active_dtcs: Vec<String>,

    /// Intake air temperature in °C (from PID 0x0F)
    pub intake_air_temp_c: Option<f64>,

    /// Mass air flow rate in g/s (from PID 0x10)
    pub maf_grams_per_sec: Option<f64>,

    /// Engine load in percent (from PID 0x04)
    pub engine_load_pct: Option<f64>,

    /// Ambient air temperature in °C (from PID 0x46)
    pub ambient_temp_c: Option<f64>,

    /// Accelerator pedal position D in percent (from PID 0x49)
    pub accel_pedal_pct: Option<f64>,

    /// Total update count (for change detection)
    pub update_count: u64,
}

impl VehicleState {
    /// Create a new empty VehicleState
    pub fn new() -> Self {
        Self::default()
    }

    /// Update this state from an OBD-II response
    ///
    /// Decodes the response PID and updates the appropriate field.
    pub fn update_from_obd(&mut self, response: &ObdResponse) {
        let value = response.decode();
        self.update_count += 1;

        match response.pid {
            0x04 => {
                if let ObdValue::Percent(p) = value {
                    self.engine_load_pct = Some(p);
                }
            }
            0x05 => {
                if let ObdValue::Temperature(t) = value {
                    self.coolant_temp_c = Some(t);
                }
            }
            0x0C => {
                if let ObdValue::Rpm(rpm) = value {
                    self.engine_rpm = Some(rpm);
                }
            }
            0x0D => {
                if let ObdValue::Speed(s) = value {
                    self.speed_kmh = Some(s);
                }
            }
            0x0F => {
                if let ObdValue::Temperature(t) = value {
                    self.intake_air_temp_c = Some(t);
                }
            }
            0x10 => {
                if let ObdValue::Grams(g) = value {
                    self.maf_grams_per_sec = Some(g);
                }
            }
            0x11 => {
                if let ObdValue::Percent(p) = value {
                    self.throttle_pct = Some(p);
                }
            }
            0x2F => {
                if let ObdValue::Percent(p) = value {
                    self.fuel_level_pct = Some(p);
                }
            }
            0x46 => {
                if let ObdValue::Temperature(t) = value {
                    self.ambient_temp_c = Some(t);
                }
            }
            0x49 => {
                if let ObdValue::Percent(p) = value {
                    self.accel_pedal_pct = Some(p);
                }
            }
            // PID 0x01: Monitor status since DTCs cleared (bit 7 = MIL)
            0x01 => {
                if let ObdValue::Raw(ref data) = value {
                    if let Some(&first_byte) = data.first() {
                        self.mil_on = (first_byte & 0x80) != 0;
                    }
                }
            }
            // Other PIDs: not tracked in state
            _ => {}
        }
    }

    /// Update this state from a raw CAN frame
    ///
    /// Handles J1939 common PGNs (EEC1 for RPM, CCVS for speed) and
    /// OBD-II multi-frame responses.
    pub fn update_from_can(&mut self, frame: &CanFrame) {
        self.update_count += 1;

        // Extract J1939 PGN from extended CAN ID
        if let Some(pgn) = frame.id.extract_j1939_pgn() {
            match pgn {
                // EEC1 (Electronic Engine Controller 1, PGN 61444)
                // Byte 3-4: Engine Speed = (256*B3 + B4) / 8 RPM
                J1939_PGN_EEC1 => {
                    if frame.data.len() >= 5 {
                        let b3 = frame.data[3] as f64;
                        let b4 = frame.data[4] as f64;
                        let rpm = (256.0 * b3 + b4) / 8.0;
                        self.engine_rpm = Some(rpm);
                    }
                }
                // CCVS (Cruise Control/Vehicle Speed, PGN 65265)
                // Bytes 1-2: Wheel-Based Vehicle Speed = (256*B1 + B0) / 256 km/h
                J1939_PGN_CCVS => {
                    if frame.data.len() >= 3 {
                        let b0 = frame.data[1] as f64;
                        let b1 = frame.data[2] as f64;
                        let speed = (256.0 * b1 + b0) / 256.0;
                        self.speed_kmh = Some(speed);
                    }
                }
                _ => {}
            }
        }
    }

    /// Check if the engine is likely running (RPM > idle threshold)
    pub fn is_engine_running(&self) -> bool {
        self.engine_rpm.map(|rpm| rpm > 200.0).unwrap_or(false)
    }

    /// Check if vehicle is in motion
    pub fn is_moving(&self) -> bool {
        self.speed_kmh.map(|s| s > 0.5).unwrap_or(false)
    }

    /// Generate basic SAREF/SSN Turtle triples for this vehicle state
    ///
    /// Uses:
    /// - SAREF (Smart Appliance REFerence ontology) for device types
    /// - SSN (Semantic Sensor Network) for observations
    /// - SOSA (Sensor, Observation, Sample, Actuator) for results
    ///
    /// # Arguments
    ///
    /// * `vin` - Vehicle Identification Number (used as the subject IRI)
    pub fn to_rdf_triples(&self, vin: &str) -> Vec<String> {
        let mut triples = Vec::new();
        let vehicle_iri = format!("<https://automotive.example.com/vehicle/{}>", vin);

        // Vehicle type assertion
        triples.push(format!(
            "{} a <https://saref.etsi.org/core/Device> .",
            vehicle_iri
        ));
        triples.push(format!(
            "{} a <https://www.w3.org/ns/ssn/System> .",
            vehicle_iri
        ));
        triples.push(format!(
            "{} <http://schema.org/vehicleIdentificationNumber> \"{}\" .",
            vehicle_iri, vin
        ));

        // Engine RPM observation
        if let Some(rpm) = self.engine_rpm {
            let obs_iri = format!(
                "<https://automotive.example.com/observation/{}/engine_rpm>",
                vin
            );
            triples.push(format!(
                "{} a <http://www.w3.org/ns/sosa/Observation> .",
                obs_iri
            ));
            triples.push(format!(
                "{} <http://www.w3.org/ns/sosa/observedProperty> <https://automotive.example.com/property/EngineSpeed> .",
                obs_iri
            ));
            triples.push(format!(
                "{} <http://www.w3.org/ns/sosa/hasSimpleResult> \"{}\"^^<http://www.w3.org/2001/XMLSchema#double> .",
                obs_iri, rpm
            ));
            triples.push(format!(
                "{} <https://qudt.org/1.1/schema/qudt#unit> <https://qudt.org/vocab/unit/REV-PER-MIN> .",
                obs_iri
            ));
            triples.push(format!(
                "{} <https://www.w3.org/ns/ssn/hasProperty> {} .",
                vehicle_iri, obs_iri
            ));
        }

        // Vehicle speed observation
        if let Some(speed) = self.speed_kmh {
            let obs_iri = format!("<https://automotive.example.com/observation/{}/speed>", vin);
            triples.push(format!(
                "{} a <http://www.w3.org/ns/sosa/Observation> .",
                obs_iri
            ));
            triples.push(format!(
                "{} <http://www.w3.org/ns/sosa/observedProperty> <https://automotive.example.com/property/VehicleSpeed> .",
                obs_iri
            ));
            triples.push(format!(
                "{} <http://www.w3.org/ns/sosa/hasSimpleResult> \"{}\"^^<http://www.w3.org/2001/XMLSchema#double> .",
                obs_iri, speed
            ));
            triples.push(format!(
                "{} <https://qudt.org/vocab/unit#KilometerPerHour> <https://qudt.org/vocab/unit/KiloM-PER-HR> .",
                obs_iri
            ));
            triples.push(format!(
                "{} <https://www.w3.org/ns/ssn/hasProperty> {} .",
                vehicle_iri, obs_iri
            ));
        }

        // Coolant temperature observation
        if let Some(temp) = self.coolant_temp_c {
            let obs_iri = format!(
                "<https://automotive.example.com/observation/{}/coolant_temp>",
                vin
            );
            triples.push(format!(
                "{} a <http://www.w3.org/ns/sosa/Observation> .",
                obs_iri
            ));
            triples.push(format!(
                "{} <http://www.w3.org/ns/sosa/observedProperty> <https://automotive.example.com/property/CoolantTemperature> .",
                obs_iri
            ));
            triples.push(format!(
                "{} <http://www.w3.org/ns/sosa/hasSimpleResult> \"{}\"^^<http://www.w3.org/2001/XMLSchema#double> .",
                obs_iri, temp
            ));
        }

        // Throttle position
        if let Some(throttle) = self.throttle_pct {
            let obs_iri = format!(
                "<https://automotive.example.com/observation/{}/throttle>",
                vin
            );
            triples.push(format!(
                "{} a <http://www.w3.org/ns/sosa/Observation> .",
                obs_iri
            ));
            triples.push(format!(
                "{} <http://www.w3.org/ns/sosa/observedProperty> <https://automotive.example.com/property/ThrottlePosition> .",
                obs_iri
            ));
            triples.push(format!(
                "{} <http://www.w3.org/ns/sosa/hasSimpleResult> \"{}\"^^<http://www.w3.org/2001/XMLSchema#double> .",
                obs_iri, throttle
            ));
        }

        // Fuel level
        if let Some(fuel) = self.fuel_level_pct {
            let obs_iri = format!(
                "<https://automotive.example.com/observation/{}/fuel_level>",
                vin
            );
            triples.push(format!(
                "{} a <http://www.w3.org/ns/sosa/Observation> .",
                obs_iri
            ));
            triples.push(format!(
                "{} <http://www.w3.org/ns/sosa/observedProperty> <https://automotive.example.com/property/FuelLevel> .",
                obs_iri
            ));
            triples.push(format!(
                "{} <http://www.w3.org/ns/sosa/hasSimpleResult> \"{}\"^^<http://www.w3.org/2001/XMLSchema#double> .",
                obs_iri, fuel
            ));
        }

        // MIL status
        triples.push(format!(
            "{} <https://automotive.example.com/property/malfunctionIndicatorLamp> \"{}\"^^<http://www.w3.org/2001/XMLSchema#boolean> .",
            vehicle_iri, self.mil_on
        ));

        // Active DTCs
        for dtc in &self.active_dtcs {
            triples.push(format!(
                "{} <https://automotive.example.com/property/activeDTC> \"{}\" .",
                vehicle_iri, dtc
            ));
        }

        triples
    }
}

/// Manager for multiple vehicle digital twins
///
/// Maintains a registry of vehicle states keyed by VIN (Vehicle Identification Number).
/// Thread-safe state is left to the caller — this struct is `!Sync` by default.
#[derive(Debug, Default)]
pub struct DigitalTwinManager {
    /// Vehicle states keyed by VIN
    vehicles: HashMap<String, VehicleState>,
}

impl DigitalTwinManager {
    /// Create a new empty digital twin manager
    pub fn new() -> Self {
        Self {
            vehicles: HashMap::new(),
        }
    }

    /// Get an immutable reference to the state for a given VIN
    pub fn get_state(&self, vin: &str) -> Option<&VehicleState> {
        self.vehicles.get(vin)
    }

    /// Get a mutable reference to the state for a given VIN
    pub fn get_state_mut(&mut self, vin: &str) -> Option<&mut VehicleState> {
        self.vehicles.get_mut(vin)
    }

    /// Update the state for a given VIN from an OBD-II response
    ///
    /// Creates a new entry if the VIN is not yet registered.
    pub fn update_state(&mut self, vin: &str, response: &ObdResponse) {
        let state = self.vehicles.entry(vin.to_string()).or_default();
        state.update_from_obd(response);
    }

    /// Update the state for a given VIN from a raw CAN frame
    pub fn update_state_from_can(&mut self, vin: &str, frame: &CanFrame) {
        let state = self.vehicles.entry(vin.to_string()).or_default();
        state.update_from_can(frame);
    }

    /// Register a VIN with an empty initial state (explicit registration)
    pub fn register_vehicle(&mut self, vin: &str) {
        self.vehicles.entry(vin.to_string()).or_default();
    }

    /// Remove a vehicle from the registry
    pub fn remove_vehicle(&mut self, vin: &str) -> Option<VehicleState> {
        self.vehicles.remove(vin)
    }

    /// List all registered VINs
    pub fn list_vins(&self) -> Vec<&str> {
        let mut vins: Vec<&str> = self.vehicles.keys().map(String::as_str).collect();
        vins.sort(); // deterministic order for tests
        vins
    }

    /// Get the total number of registered vehicles
    pub fn vehicle_count(&self) -> usize {
        self.vehicles.len()
    }

    /// Generate RDF triples for all registered vehicles
    pub fn all_rdf_triples(&self) -> Vec<String> {
        let mut all_triples = Vec::new();
        let mut vins: Vec<&str> = self.vehicles.keys().map(String::as_str).collect();
        vins.sort(); // deterministic order
        for vin in vins {
            if let Some(state) = self.vehicles.get(vin) {
                all_triples.extend(state.to_rdf_triples(vin));
            }
        }
        all_triples
    }

    /// Returns true if the given VIN is already registered
    pub fn contains_vin(&self, vin: &str) -> bool {
        self.vehicles.contains_key(vin)
    }
}

/// Convenience function to get the description for a known OBD PID
pub fn obd_pid_description(pid: u8) -> &'static str {
    ObdPid::from_byte(pid)
        .map(|p| p.name())
        .unwrap_or("Unknown PID")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{CanFrame, CanId};

    // ---- VehicleState tests ----

    #[test]
    fn test_vehicle_state_default() {
        let state = VehicleState::new();
        assert!(state.engine_rpm.is_none());
        assert!(state.speed_kmh.is_none());
        assert!(state.coolant_temp_c.is_none());
        assert!(!state.mil_on);
        assert!(state.active_dtcs.is_empty());
        assert_eq!(state.update_count, 0);
    }

    #[test]
    fn test_update_from_obd_engine_rpm() {
        let mut state = VehicleState::new();
        // PID 0x0C: (256*0x1A + 0xF0) / 4 = 1724 RPM
        let resp = ObdResponse::new(0x0C, vec![0x1A, 0xF0]);
        state.update_from_obd(&resp);

        assert!(state.engine_rpm.is_some());
        let rpm = state.engine_rpm.expect("rpm should be Some");
        assert!((rpm - 1724.0).abs() < 0.01);
        assert_eq!(state.update_count, 1);
    }

    #[test]
    fn test_update_from_obd_vehicle_speed() {
        let mut state = VehicleState::new();
        // PID 0x0D: A = 0x64 = 100 km/h
        let resp = ObdResponse::new(0x0D, vec![0x64]);
        state.update_from_obd(&resp);

        let speed = state.speed_kmh.expect("speed should be Some");
        assert!((speed - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_update_from_obd_coolant_temp() {
        let mut state = VehicleState::new();
        // PID 0x05: A=0x69 (105): 105 - 40 = 65°C
        let resp = ObdResponse::new(0x05, vec![0x69]);
        state.update_from_obd(&resp);

        let temp = state.coolant_temp_c.expect("coolant temp should be Some");
        assert!((temp - 65.0).abs() < 0.01);
    }

    #[test]
    fn test_update_from_obd_throttle() {
        let mut state = VehicleState::new();
        // PID 0x11: A/2.55 %
        let resp = ObdResponse::new(0x11, vec![0x80]);
        state.update_from_obd(&resp);

        let throttle = state.throttle_pct.expect("throttle should be Some");
        assert!((throttle - 128.0 / 2.55).abs() < 0.01);
    }

    #[test]
    fn test_update_from_obd_fuel_level() {
        let mut state = VehicleState::new();
        // PID 0x2F: A/2.55 % = 0x7F/2.55 ≈ 49.8%
        let resp = ObdResponse::new(0x2F, vec![0x7F]);
        state.update_from_obd(&resp);

        let fuel = state.fuel_level_pct.expect("fuel level should be Some");
        assert!((fuel - 127.0 / 2.55).abs() < 0.01);
    }

    #[test]
    fn test_update_from_obd_engine_load() {
        let mut state = VehicleState::new();
        let resp = ObdResponse::new(0x04, vec![0xFF]);
        state.update_from_obd(&resp);
        let load = state.engine_load_pct.expect("engine load should be Some");
        assert!((load - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_update_from_obd_intake_temp() {
        let mut state = VehicleState::new();
        // PID 0x0F: A=0x3C (60): 60 - 40 = 20°C
        let resp = ObdResponse::new(0x0F, vec![0x3C]);
        state.update_from_obd(&resp);
        let temp = state.intake_air_temp_c.expect("intake temp should be Some");
        assert!((temp - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_update_from_obd_maf() {
        let mut state = VehicleState::new();
        // PID 0x10: (256*0x01 + 0x90) / 100 = 4.0 g/s
        let resp = ObdResponse::new(0x10, vec![0x01, 0x90]);
        state.update_from_obd(&resp);
        let maf = state.maf_grams_per_sec.expect("MAF should be Some");
        assert!((maf - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_update_from_obd_ambient_temp() {
        let mut state = VehicleState::new();
        // PID 0x46: A=0x1E (30): 30 - 40 = -10°C
        let resp = ObdResponse::new(0x46, vec![0x1E]);
        state.update_from_obd(&resp);
        let temp = state.ambient_temp_c.expect("ambient temp should be Some");
        assert!((temp - (-10.0)).abs() < 0.01);
    }

    #[test]
    fn test_update_from_obd_accel_pedal() {
        let mut state = VehicleState::new();
        let resp = ObdResponse::new(0x49, vec![0x80]);
        state.update_from_obd(&resp);
        let pos = state.accel_pedal_pct.expect("accel pedal should be Some");
        assert!((pos - 128.0 / 2.55).abs() < 0.01);
    }

    #[test]
    fn test_update_from_obd_mil_status() {
        let mut state = VehicleState::new();
        // PID 0x01: byte 0 bit 7 = MIL on
        let resp = ObdResponse::new(0x01, vec![0x80, 0x00, 0x00, 0x00]);
        state.update_from_obd(&resp);
        assert!(state.mil_on);

        // MIL off
        let resp2 = ObdResponse::new(0x01, vec![0x00, 0x00, 0x00, 0x00]);
        state.update_from_obd(&resp2);
        assert!(!state.mil_on);
    }

    #[test]
    fn test_is_engine_running() {
        let mut state = VehicleState::new();
        assert!(!state.is_engine_running()); // no data yet

        // Set idle RPM (below threshold)
        let resp = ObdResponse::new(0x0C, vec![0x00, 0x00]);
        state.update_from_obd(&resp);
        assert!(!state.is_engine_running()); // 0 RPM = not running

        // Set running RPM
        let resp2 = ObdResponse::new(0x0C, vec![0x1A, 0xF0]);
        state.update_from_obd(&resp2);
        assert!(state.is_engine_running()); // 1724 RPM = running
    }

    #[test]
    fn test_is_moving() {
        let mut state = VehicleState::new();
        assert!(!state.is_moving()); // no data

        let resp = ObdResponse::new(0x0D, vec![0x00]);
        state.update_from_obd(&resp);
        assert!(!state.is_moving()); // 0 km/h

        let resp2 = ObdResponse::new(0x0D, vec![0x64]);
        state.update_from_obd(&resp2);
        assert!(state.is_moving()); // 100 km/h
    }

    #[test]
    fn test_update_from_can_eec1_rpm() {
        let mut state = VehicleState::new();
        // EEC1 PGN 61444 = 0xF004
        // CAN ID for EEC1: Priority 3, PF=0xF0, PS=0x04, SA=0x00 => 0x0CF00400
        let can_id = CanId::extended(0x0CF00400).expect("valid extended CAN ID");
        // Bytes 3-4 = engine speed: (256*26 + 240) / 8 = 6896/8 = 862 RPM
        let data = vec![0x00, 0x00, 0x00, 0x1A, 0xF0, 0x00, 0x00, 0x00];
        let frame = CanFrame::new(can_id, data).expect("valid CAN frame");
        state.update_from_can(&frame);

        let rpm = state.engine_rpm.expect("RPM should be set from EEC1");
        assert!((rpm - 862.0).abs() < 0.1);
    }

    #[test]
    fn test_update_from_can_unknown_pgn() {
        let mut state = VehicleState::new();
        // Use a standard CAN ID (not J1939 extended)
        let can_id = CanId::standard(0x123).expect("valid standard CAN ID");
        let data = vec![0x01, 0x02, 0x03, 0x04];
        let frame = CanFrame::new(can_id, data).expect("valid CAN frame");
        state.update_from_can(&frame);

        // State should not change for unknown frames
        assert!(state.engine_rpm.is_none());
        assert_eq!(state.update_count, 1); // update_count still increments
    }

    #[test]
    fn test_to_rdf_triples_basic() {
        let mut state = VehicleState::new();
        state.engine_rpm = Some(1500.0);
        state.speed_kmh = Some(60.0);
        state.coolant_temp_c = Some(90.0);

        let vin = "1HGCM82633A123456";
        let triples = state.to_rdf_triples(vin);

        // Should contain vehicle type triple
        assert!(triples.iter().any(|t| t.contains("saref.etsi.org")));
        // Should contain RPM observation
        assert!(triples
            .iter()
            .any(|t| t.contains("engine_rpm") && t.contains("1500")));
        // Should contain speed observation
        assert!(triples
            .iter()
            .any(|t| t.contains("speed") && t.contains("60")));
        // VIN in triple
        assert!(triples.iter().any(|t| t.contains("1HGCM82633A123456")));
    }

    #[test]
    fn test_to_rdf_triples_with_mil_and_dtcs() {
        let mut state = VehicleState::new();
        state.mil_on = true;
        state.active_dtcs = vec!["P0301".to_string(), "P0420".to_string()];

        let triples = state.to_rdf_triples("WVWZZZ1JZ3W386752");

        // MIL should be in triples
        assert!(triples
            .iter()
            .any(|t| t.contains("malfunctionIndicatorLamp") && t.contains("true")));
        // DTCs should be in triples
        assert!(triples.iter().any(|t| t.contains("P0301")));
        assert!(triples.iter().any(|t| t.contains("P0420")));
    }

    #[test]
    fn test_to_rdf_triples_empty_state() {
        let state = VehicleState::new();
        let triples = state.to_rdf_triples("TESTVIN123");

        // Should still have base vehicle type triples
        assert!(!triples.is_empty());
        assert!(triples.iter().any(|t| t.contains("TESTVIN123")));
    }

    // ---- DigitalTwinManager tests ----

    #[test]
    fn test_digital_twin_manager_new() {
        let manager = DigitalTwinManager::new();
        assert_eq!(manager.vehicle_count(), 0);
        assert!(manager.list_vins().is_empty());
    }

    #[test]
    fn test_digital_twin_manager_update_state_creates_entry() {
        let mut manager = DigitalTwinManager::new();
        let vin = "1HGCM82633A123456";
        let resp = ObdResponse::new(0x0C, vec![0x1A, 0xF0]);

        assert!(!manager.contains_vin(vin));
        manager.update_state(vin, &resp);
        assert!(manager.contains_vin(vin));
        assert_eq!(manager.vehicle_count(), 1);
    }

    #[test]
    fn test_digital_twin_manager_get_state() {
        let mut manager = DigitalTwinManager::new();
        let vin = "1HGCM82633A123456";

        // Non-existent VIN returns None
        assert!(manager.get_state(vin).is_none());

        let resp = ObdResponse::new(0x0C, vec![0x1A, 0xF0]);
        manager.update_state(vin, &resp);

        let state = manager.get_state(vin).expect("state should exist");
        assert!(state.engine_rpm.is_some());
    }

    #[test]
    fn test_digital_twin_manager_list_vins() {
        let mut manager = DigitalTwinManager::new();
        let resp = ObdResponse::new(0x0D, vec![0x64]);

        manager.update_state("VIN_A", &resp);
        manager.update_state("VIN_B", &resp);
        manager.update_state("VIN_C", &resp);

        let vins = manager.list_vins();
        assert_eq!(vins.len(), 3);
        // list_vins() returns sorted order
        assert_eq!(vins, vec!["VIN_A", "VIN_B", "VIN_C"]);
    }

    #[test]
    fn test_digital_twin_manager_multiple_updates() {
        let mut manager = DigitalTwinManager::new();
        let vin = "TEST_VIN";

        manager.update_state(vin, &ObdResponse::new(0x0C, vec![0x1A, 0xF0]));
        manager.update_state(vin, &ObdResponse::new(0x0D, vec![0x64]));
        manager.update_state(vin, &ObdResponse::new(0x05, vec![0x69]));

        let state = manager.get_state(vin).expect("state should exist");
        assert!(state.engine_rpm.is_some());
        assert!(state.speed_kmh.is_some());
        assert!(state.coolant_temp_c.is_some());
        assert_eq!(state.update_count, 3);
    }

    #[test]
    fn test_digital_twin_manager_remove_vehicle() {
        let mut manager = DigitalTwinManager::new();
        manager.register_vehicle("VIN_X");
        assert_eq!(manager.vehicle_count(), 1);

        let removed = manager.remove_vehicle("VIN_X");
        assert!(removed.is_some());
        assert_eq!(manager.vehicle_count(), 0);

        // Removing non-existent VIN returns None
        let not_found = manager.remove_vehicle("VIN_X");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_digital_twin_manager_all_rdf_triples() {
        let mut manager = DigitalTwinManager::new();

        manager.update_state("VIN_A", &ObdResponse::new(0x0C, vec![0x1A, 0xF0]));
        manager.update_state("VIN_B", &ObdResponse::new(0x0D, vec![0x64]));

        let triples = manager.all_rdf_triples();
        assert!(!triples.is_empty());
        assert!(triples.iter().any(|t| t.contains("VIN_A")));
        assert!(triples.iter().any(|t| t.contains("VIN_B")));
    }

    #[test]
    fn test_digital_twin_manager_update_from_can() {
        let mut manager = DigitalTwinManager::new();
        let vin = "J1939_TRUCK_001";

        // EEC1 frame for RPM
        let can_id = CanId::extended(0x0CF00400).expect("valid extended CAN ID");
        let data = vec![0x00, 0x00, 0x00, 0x1A, 0xF0, 0x00, 0x00, 0x00];
        let frame = CanFrame::new(can_id, data).expect("valid CAN frame");

        manager.update_state_from_can(vin, &frame);
        let state = manager.get_state(vin).expect("state should exist");
        assert!(state.engine_rpm.is_some());
    }

    #[test]
    fn test_obd_pid_description() {
        assert!(
            obd_pid_description(0x0C).contains("RPM")
                || obd_pid_description(0x0C).contains("Engine")
        );
        assert_eq!(obd_pid_description(0xFF), "Unknown PID");
    }
}
