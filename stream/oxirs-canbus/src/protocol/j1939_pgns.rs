//! Common J1939 Parameter Group Numbers (PGNs)
//!
//! This module provides decoders for commonly used J1939 PGNs in heavy vehicles,
//! agricultural equipment, marine vessels, and construction equipment.
//!
//! # PGN Categories
//!
//! - Engine data (61444, 65262, 65263)
//! - Vehicle speed/cruise control (65265)
//! - Transmission (61442, 61445)
//! - Brakes (61441)
//! - Fuel consumption (65266)

use crate::protocol::j1939::J1939Message;
use std::collections::HashMap;

/// PGN value type alias
pub type PgnValue = u32;

// ============================================================================
// Common PGN Constants
// ============================================================================

/// Electronic Engine Controller 1 (EEC1) - Engine speed, torque
pub const PGN_EEC1: PgnValue = 61444;
/// Electronic Engine Controller 2 (EEC2) - Accelerator position
pub const PGN_EEC2: PgnValue = 61443;
/// Electronic Transmission Controller 1 (ETC1)
pub const PGN_ETC1: PgnValue = 61442;
/// Electronic Transmission Controller 2 (ETC2)
pub const PGN_ETC2: PgnValue = 61445;
/// Electronic Brake Controller 1 (EBC1)
pub const PGN_EBC1: PgnValue = 61441;
/// Engine Temperature 1 (ET1)
pub const PGN_ET1: PgnValue = 65262;
/// Engine Fluid Level/Pressure 1 (EFL/P1)
pub const PGN_EFLP1: PgnValue = 65263;
/// Cruise Control/Vehicle Speed (CCVS)
pub const PGN_CCVS: PgnValue = 65265;
/// Fuel Economy (LFE)
pub const PGN_LFE: PgnValue = 65266;
/// Dash Display (DD)
pub const PGN_DD: PgnValue = 65276;
/// Ambient Conditions (AMB)
pub const PGN_AMB: PgnValue = 65269;
/// Vehicle Weight (VW)
pub const PGN_VW: PgnValue = 65258;
/// High Resolution Wheel Speed (HRWS)
pub const PGN_HRWS: PgnValue = 65134;
/// Fuel Consumption (LFC)
pub const PGN_LFC: PgnValue = 65257;
/// Vehicle Electrical Power 1 (VEP1)
pub const PGN_VEP1: PgnValue = 65271;
/// Component Identification (CI)
pub const PGN_CI: PgnValue = 65259;
/// Software Identification (SOFT)
pub const PGN_SOFT: PgnValue = 65242;

// ============================================================================
// Signal Definitions
// ============================================================================

/// A decoded signal value with metadata
#[derive(Debug, Clone)]
pub struct DecodedSignal {
    /// Signal name
    pub name: &'static str,
    /// Physical value after scaling
    pub value: f64,
    /// Unit of measurement
    pub unit: &'static str,
    /// Raw value before scaling
    pub raw_value: u64,
    /// Whether the value is valid (not error/not available)
    pub valid: bool,
}

impl DecodedSignal {
    /// Create a new decoded signal
    pub fn new(name: &'static str, value: f64, unit: &'static str, raw: u64) -> Self {
        Self {
            name,
            value,
            unit,
            raw_value: raw,
            valid: true,
        }
    }

    /// Create an invalid/not available signal
    pub fn not_available(name: &'static str, unit: &'static str) -> Self {
        Self {
            name,
            value: f64::NAN,
            unit,
            raw_value: 0,
            valid: false,
        }
    }
}

/// Decoded PGN message with all signals
#[derive(Debug)]
pub struct DecodedPgn {
    /// PGN value
    pub pgn: PgnValue,
    /// PGN name
    pub name: &'static str,
    /// Description
    pub description: &'static str,
    /// Decoded signals
    pub signals: Vec<DecodedSignal>,
}

// ============================================================================
// PGN Decoders
// ============================================================================

/// Trait for PGN decoders
pub trait PgnDecoder: Send + Sync {
    /// Get PGN value
    fn pgn(&self) -> PgnValue;

    /// Get PGN name
    fn name(&self) -> &'static str;

    /// Get PGN description
    fn description(&self) -> &'static str;

    /// Decode message data
    fn decode(&self, message: &J1939Message) -> Option<DecodedPgn>;
}

/// Electronic Engine Controller 1 (EEC1) decoder
#[derive(Debug, Default)]
pub struct Eec1Decoder;

impl PgnDecoder for Eec1Decoder {
    fn pgn(&self) -> PgnValue {
        PGN_EEC1
    }

    fn name(&self) -> &'static str {
        "EEC1"
    }

    fn description(&self) -> &'static str {
        "Electronic Engine Controller 1"
    }

    fn decode(&self, message: &J1939Message) -> Option<DecodedPgn> {
        if message.data.len() < 8 {
            return None;
        }

        let mut signals = Vec::new();

        // Engine Torque Mode (byte 0, bits 0-3)
        let torque_mode = message.data[0] & 0x0F;
        signals.push(DecodedSignal::new(
            "EngineTorqueMode",
            torque_mode as f64,
            "",
            torque_mode as u64,
        ));

        // Driver's Demand Engine - Percent Torque (byte 1)
        let driver_demand = message.data[1];
        if driver_demand != 0xFF {
            signals.push(DecodedSignal::new(
                "DriversDemandPercentTorque",
                driver_demand as f64 - 125.0,
                "%",
                driver_demand as u64,
            ));
        }

        // Actual Engine - Percent Torque (byte 2)
        let actual_torque = message.data[2];
        if actual_torque != 0xFF {
            signals.push(DecodedSignal::new(
                "ActualEnginePercentTorque",
                actual_torque as f64 - 125.0,
                "%",
                actual_torque as u64,
            ));
        }

        // Engine Speed (bytes 3-4, little-endian)
        let engine_speed_raw = u16::from_le_bytes([message.data[3], message.data[4]]);
        if engine_speed_raw != 0xFFFF {
            let engine_speed = engine_speed_raw as f64 * 0.125;
            signals.push(DecodedSignal::new(
                "EngineSpeed",
                engine_speed,
                "rpm",
                engine_speed_raw as u64,
            ));
        } else {
            signals.push(DecodedSignal::not_available("EngineSpeed", "rpm"));
        }

        // Source Address of Controlling Device (byte 5)
        let source_addr = message.data[5];
        if source_addr != 0xFF {
            signals.push(DecodedSignal::new(
                "SourceAddressOfControllingDevice",
                source_addr as f64,
                "",
                source_addr as u64,
            ));
        }

        // Engine Starter Mode (byte 6, bits 0-3)
        let starter_mode = message.data[6] & 0x0F;
        signals.push(DecodedSignal::new(
            "EngineStarterMode",
            starter_mode as f64,
            "",
            starter_mode as u64,
        ));

        // Engine Demand - Percent Torque (byte 7)
        let engine_demand = message.data[7];
        if engine_demand != 0xFF {
            signals.push(DecodedSignal::new(
                "EngineDemandPercentTorque",
                engine_demand as f64 - 125.0,
                "%",
                engine_demand as u64,
            ));
        }

        Some(DecodedPgn {
            pgn: PGN_EEC1,
            name: "EEC1",
            description: "Electronic Engine Controller 1",
            signals,
        })
    }
}

/// Electronic Engine Controller 2 (EEC2) decoder
#[derive(Debug, Default)]
pub struct Eec2Decoder;

impl PgnDecoder for Eec2Decoder {
    fn pgn(&self) -> PgnValue {
        PGN_EEC2
    }

    fn name(&self) -> &'static str {
        "EEC2"
    }

    fn description(&self) -> &'static str {
        "Electronic Engine Controller 2"
    }

    fn decode(&self, message: &J1939Message) -> Option<DecodedPgn> {
        if message.data.len() < 8 {
            return None;
        }

        let mut signals = Vec::new();

        // Accelerator Pedal 1 Low Idle Switch (byte 0, bit 0)
        let low_idle = message.data[0] & 0x03;
        signals.push(DecodedSignal::new(
            "AcceleratorPedal1LowIdleSwitch",
            low_idle as f64,
            "",
            low_idle as u64,
        ));

        // Accelerator Pedal Kickdown Switch (byte 0, bit 2)
        let kickdown = (message.data[0] >> 2) & 0x03;
        signals.push(DecodedSignal::new(
            "AcceleratorPedalKickdownSwitch",
            kickdown as f64,
            "",
            kickdown as u64,
        ));

        // Accelerator Pedal Position 1 (byte 1)
        let pedal_pos = message.data[1];
        if pedal_pos != 0xFF {
            let position = pedal_pos as f64 * 0.4;
            signals.push(DecodedSignal::new(
                "AcceleratorPedalPosition1",
                position,
                "%",
                pedal_pos as u64,
            ));
        } else {
            signals.push(DecodedSignal::not_available(
                "AcceleratorPedalPosition1",
                "%",
            ));
        }

        // Engine Percent Load At Current Speed (byte 2)
        let load = message.data[2];
        if load != 0xFF {
            signals.push(DecodedSignal::new(
                "EnginePercentLoadAtCurrentSpeed",
                load as f64,
                "%",
                load as u64,
            ));
        }

        // Remote Accelerator Pedal Position (byte 3)
        let remote_pedal = message.data[3];
        if remote_pedal != 0xFF {
            let position = remote_pedal as f64 * 0.4;
            signals.push(DecodedSignal::new(
                "RemoteAcceleratorPedalPosition",
                position,
                "%",
                remote_pedal as u64,
            ));
        }

        Some(DecodedPgn {
            pgn: PGN_EEC2,
            name: "EEC2",
            description: "Electronic Engine Controller 2",
            signals,
        })
    }
}

/// Engine Temperature 1 (ET1) decoder
#[derive(Debug, Default)]
pub struct Et1Decoder;

impl PgnDecoder for Et1Decoder {
    fn pgn(&self) -> PgnValue {
        PGN_ET1
    }

    fn name(&self) -> &'static str {
        "ET1"
    }

    fn description(&self) -> &'static str {
        "Engine Temperature 1"
    }

    fn decode(&self, message: &J1939Message) -> Option<DecodedPgn> {
        if message.data.len() < 8 {
            return None;
        }

        let mut signals = Vec::new();

        // Engine Coolant Temperature (byte 0)
        let coolant_temp = message.data[0];
        if coolant_temp != 0xFF {
            let temp = coolant_temp as f64 - 40.0;
            signals.push(DecodedSignal::new(
                "EngineCoolantTemperature",
                temp,
                "°C",
                coolant_temp as u64,
            ));
        } else {
            signals.push(DecodedSignal::not_available(
                "EngineCoolantTemperature",
                "°C",
            ));
        }

        // Engine Fuel Temperature 1 (byte 1)
        let fuel_temp = message.data[1];
        if fuel_temp != 0xFF {
            let temp = fuel_temp as f64 - 40.0;
            signals.push(DecodedSignal::new(
                "EngineFuelTemperature1",
                temp,
                "°C",
                fuel_temp as u64,
            ));
        }

        // Engine Oil Temperature 1 (bytes 2-3)
        let oil_temp_raw = u16::from_le_bytes([message.data[2], message.data[3]]);
        if oil_temp_raw != 0xFFFF {
            let temp = oil_temp_raw as f64 * 0.03125 - 273.0;
            signals.push(DecodedSignal::new(
                "EngineOilTemperature1",
                temp,
                "°C",
                oil_temp_raw as u64,
            ));
        }

        // Engine Turbocharger Oil Temperature (bytes 4-5)
        let turbo_temp_raw = u16::from_le_bytes([message.data[4], message.data[5]]);
        if turbo_temp_raw != 0xFFFF {
            let temp = turbo_temp_raw as f64 * 0.03125 - 273.0;
            signals.push(DecodedSignal::new(
                "EngineTurbochargerOilTemperature",
                temp,
                "°C",
                turbo_temp_raw as u64,
            ));
        }

        // Engine Intercooler Temperature (byte 6)
        let intercooler_temp = message.data[6];
        if intercooler_temp != 0xFF {
            let temp = intercooler_temp as f64 - 40.0;
            signals.push(DecodedSignal::new(
                "EngineIntercoolerTemperature",
                temp,
                "°C",
                intercooler_temp as u64,
            ));
        }

        // Engine Intercooler Thermostat Opening (byte 7)
        let thermostat = message.data[7];
        if thermostat != 0xFF {
            let opening = thermostat as f64 * 0.4;
            signals.push(DecodedSignal::new(
                "EngineIntercoolerThermostatOpening",
                opening,
                "%",
                thermostat as u64,
            ));
        }

        Some(DecodedPgn {
            pgn: PGN_ET1,
            name: "ET1",
            description: "Engine Temperature 1",
            signals,
        })
    }
}

/// Engine Fluid Level/Pressure 1 (EFL/P1) decoder
#[derive(Debug, Default)]
pub struct Eflp1Decoder;

impl PgnDecoder for Eflp1Decoder {
    fn pgn(&self) -> PgnValue {
        PGN_EFLP1
    }

    fn name(&self) -> &'static str {
        "EFL/P1"
    }

    fn description(&self) -> &'static str {
        "Engine Fluid Level/Pressure 1"
    }

    fn decode(&self, message: &J1939Message) -> Option<DecodedPgn> {
        if message.data.len() < 8 {
            return None;
        }

        let mut signals = Vec::new();

        // Engine Fuel Delivery Pressure (byte 0)
        let fuel_pressure = message.data[0];
        if fuel_pressure != 0xFF {
            let pressure = fuel_pressure as f64 * 4.0;
            signals.push(DecodedSignal::new(
                "EngineFuelDeliveryPressure",
                pressure,
                "kPa",
                fuel_pressure as u64,
            ));
        }

        // Engine Extended Crankcase Blow-by Pressure (byte 1)
        let blowby = message.data[1];
        if blowby != 0xFF {
            let pressure = blowby as f64 * 0.05 - 6.25;
            signals.push(DecodedSignal::new(
                "EngineExtendedCrankcaseBlowbyPressure",
                pressure,
                "kPa",
                blowby as u64,
            ));
        }

        // Engine Oil Level (byte 2)
        let oil_level = message.data[2];
        if oil_level != 0xFF {
            let level = oil_level as f64 * 0.4;
            signals.push(DecodedSignal::new(
                "EngineOilLevel",
                level,
                "%",
                oil_level as u64,
            ));
        }

        // Engine Oil Pressure (byte 3)
        let oil_pressure = message.data[3];
        if oil_pressure != 0xFF {
            let pressure = oil_pressure as f64 * 4.0;
            signals.push(DecodedSignal::new(
                "EngineOilPressure",
                pressure,
                "kPa",
                oil_pressure as u64,
            ));
        }

        // Engine Crankcase Pressure (bytes 4-5)
        let crankcase_raw = u16::from_le_bytes([message.data[4], message.data[5]]);
        if crankcase_raw != 0xFFFF {
            let pressure = crankcase_raw as f64 * 0.0078125 - 250.0;
            signals.push(DecodedSignal::new(
                "EngineCrankcasePressure",
                pressure,
                "kPa",
                crankcase_raw as u64,
            ));
        }

        // Engine Coolant Pressure (byte 6)
        let coolant_pressure = message.data[6];
        if coolant_pressure != 0xFF {
            let pressure = coolant_pressure as f64 * 2.0;
            signals.push(DecodedSignal::new(
                "EngineCoolantPressure",
                pressure,
                "kPa",
                coolant_pressure as u64,
            ));
        }

        // Engine Coolant Level (byte 7)
        let coolant_level = message.data[7];
        if coolant_level != 0xFF {
            let level = coolant_level as f64 * 0.4;
            signals.push(DecodedSignal::new(
                "EngineCoolantLevel",
                level,
                "%",
                coolant_level as u64,
            ));
        }

        Some(DecodedPgn {
            pgn: PGN_EFLP1,
            name: "EFL/P1",
            description: "Engine Fluid Level/Pressure 1",
            signals,
        })
    }
}

/// Cruise Control/Vehicle Speed (CCVS) decoder
#[derive(Debug, Default)]
pub struct CcvsDecoder;

impl PgnDecoder for CcvsDecoder {
    fn pgn(&self) -> PgnValue {
        PGN_CCVS
    }

    fn name(&self) -> &'static str {
        "CCVS"
    }

    fn description(&self) -> &'static str {
        "Cruise Control/Vehicle Speed"
    }

    fn decode(&self, message: &J1939Message) -> Option<DecodedPgn> {
        if message.data.len() < 8 {
            return None;
        }

        let mut signals = Vec::new();

        // Two Speed Axle Switch (byte 0, bits 0-1)
        let two_speed = message.data[0] & 0x03;
        signals.push(DecodedSignal::new(
            "TwoSpeedAxleSwitch",
            two_speed as f64,
            "",
            two_speed as u64,
        ));

        // Parking Brake Switch (byte 0, bits 2-3)
        let parking = (message.data[0] >> 2) & 0x03;
        signals.push(DecodedSignal::new(
            "ParkingBrakeSwitch",
            parking as f64,
            "",
            parking as u64,
        ));

        // Cruise Control Pause Switch (byte 0, bits 4-5)
        let cc_pause = (message.data[0] >> 4) & 0x03;
        signals.push(DecodedSignal::new(
            "CruiseControlPauseSwitch",
            cc_pause as f64,
            "",
            cc_pause as u64,
        ));

        // Wheel-Based Vehicle Speed (bytes 1-2)
        let speed_raw = u16::from_le_bytes([message.data[1], message.data[2]]);
        if speed_raw != 0xFFFF {
            let speed = speed_raw as f64 / 256.0; // 1/256 km/h
            signals.push(DecodedSignal::new(
                "WheelBasedVehicleSpeed",
                speed,
                "km/h",
                speed_raw as u64,
            ));
        } else {
            signals.push(DecodedSignal::not_available(
                "WheelBasedVehicleSpeed",
                "km/h",
            ));
        }

        // Cruise Control Active (byte 3, bits 0-1)
        let cc_active = message.data[3] & 0x03;
        signals.push(DecodedSignal::new(
            "CruiseControlActive",
            cc_active as f64,
            "",
            cc_active as u64,
        ));

        // Cruise Control Enable Switch (byte 3, bits 2-3)
        let cc_enable = (message.data[3] >> 2) & 0x03;
        signals.push(DecodedSignal::new(
            "CruiseControlEnableSwitch",
            cc_enable as f64,
            "",
            cc_enable as u64,
        ));

        // Brake Switch (byte 3, bits 4-5)
        let brake = (message.data[3] >> 4) & 0x03;
        signals.push(DecodedSignal::new(
            "BrakeSwitch",
            brake as f64,
            "",
            brake as u64,
        ));

        // Clutch Switch (byte 3, bits 6-7)
        let clutch = (message.data[3] >> 6) & 0x03;
        signals.push(DecodedSignal::new(
            "ClutchSwitch",
            clutch as f64,
            "",
            clutch as u64,
        ));

        // Cruise Control Set Speed (byte 4)
        let set_speed = message.data[4];
        if set_speed != 0xFF {
            signals.push(DecodedSignal::new(
                "CruiseControlSetSpeed",
                set_speed as f64,
                "km/h",
                set_speed as u64,
            ));
        }

        // PTO Governor State (byte 5, bits 0-4)
        let pto_state = message.data[5] & 0x1F;
        signals.push(DecodedSignal::new(
            "PTOGovernorState",
            pto_state as f64,
            "",
            pto_state as u64,
        ));

        // Cruise Control States (byte 6)
        let cc_state = message.data[6] & 0x07;
        signals.push(DecodedSignal::new(
            "CruiseControlStates",
            cc_state as f64,
            "",
            cc_state as u64,
        ));

        // Engine Idle Increment Switch (byte 6, bits 6-7)
        let idle_inc = (message.data[6] >> 6) & 0x03;
        signals.push(DecodedSignal::new(
            "EngineIdleIncrementSwitch",
            idle_inc as f64,
            "",
            idle_inc as u64,
        ));

        Some(DecodedPgn {
            pgn: PGN_CCVS,
            name: "CCVS",
            description: "Cruise Control/Vehicle Speed",
            signals,
        })
    }
}

/// Fuel Economy (LFE) decoder
#[derive(Debug, Default)]
pub struct LfeDecoder;

impl PgnDecoder for LfeDecoder {
    fn pgn(&self) -> PgnValue {
        PGN_LFE
    }

    fn name(&self) -> &'static str {
        "LFE"
    }

    fn description(&self) -> &'static str {
        "Fuel Economy"
    }

    fn decode(&self, message: &J1939Message) -> Option<DecodedPgn> {
        if message.data.len() < 8 {
            return None;
        }

        let mut signals = Vec::new();

        // Engine Fuel Rate (bytes 0-1)
        let fuel_rate_raw = u16::from_le_bytes([message.data[0], message.data[1]]);
        if fuel_rate_raw != 0xFFFF {
            let rate = fuel_rate_raw as f64 * 0.05; // 0.05 L/h
            signals.push(DecodedSignal::new(
                "EngineFuelRate",
                rate,
                "L/h",
                fuel_rate_raw as u64,
            ));
        } else {
            signals.push(DecodedSignal::not_available("EngineFuelRate", "L/h"));
        }

        // Engine Instantaneous Fuel Economy (bytes 2-3)
        let economy_raw = u16::from_le_bytes([message.data[2], message.data[3]]);
        if economy_raw != 0xFFFF {
            let economy = economy_raw as f64 / 512.0; // 1/512 km/L
            signals.push(DecodedSignal::new(
                "EngineInstantaneousFuelEconomy",
                economy,
                "km/L",
                economy_raw as u64,
            ));
        }

        // Engine Average Fuel Economy (bytes 4-5)
        let avg_economy_raw = u16::from_le_bytes([message.data[4], message.data[5]]);
        if avg_economy_raw != 0xFFFF {
            let economy = avg_economy_raw as f64 / 512.0;
            signals.push(DecodedSignal::new(
                "EngineAverageFuelEconomy",
                economy,
                "km/L",
                avg_economy_raw as u64,
            ));
        }

        // Engine Throttle Valve 1 Position (byte 6)
        let throttle = message.data[6];
        if throttle != 0xFF {
            let position = throttle as f64 * 0.4;
            signals.push(DecodedSignal::new(
                "EngineThrottleValve1Position",
                position,
                "%",
                throttle as u64,
            ));
        }

        // Engine Throttle Valve 2 Position (byte 7)
        let throttle2 = message.data[7];
        if throttle2 != 0xFF {
            let position = throttle2 as f64 * 0.4;
            signals.push(DecodedSignal::new(
                "EngineThrottleValve2Position",
                position,
                "%",
                throttle2 as u64,
            ));
        }

        Some(DecodedPgn {
            pgn: PGN_LFE,
            name: "LFE",
            description: "Fuel Economy",
            signals,
        })
    }
}

/// Ambient Conditions (AMB) decoder
#[derive(Debug, Default)]
pub struct AmbDecoder;

impl PgnDecoder for AmbDecoder {
    fn pgn(&self) -> PgnValue {
        PGN_AMB
    }

    fn name(&self) -> &'static str {
        "AMB"
    }

    fn description(&self) -> &'static str {
        "Ambient Conditions"
    }

    fn decode(&self, message: &J1939Message) -> Option<DecodedPgn> {
        if message.data.len() < 8 {
            return None;
        }

        let mut signals = Vec::new();

        // Barometric Pressure (byte 0)
        let baro = message.data[0];
        if baro != 0xFF {
            let pressure = baro as f64 * 0.5; // 0.5 kPa
            signals.push(DecodedSignal::new(
                "BarometricPressure",
                pressure,
                "kPa",
                baro as u64,
            ));
        }

        // Cab Interior Temperature (bytes 1-2)
        let cab_temp_raw = u16::from_le_bytes([message.data[1], message.data[2]]);
        if cab_temp_raw != 0xFFFF {
            let temp = cab_temp_raw as f64 * 0.03125 - 273.0;
            signals.push(DecodedSignal::new(
                "CabInteriorTemperature",
                temp,
                "°C",
                cab_temp_raw as u64,
            ));
        }

        // Ambient Air Temperature (bytes 3-4)
        let ambient_raw = u16::from_le_bytes([message.data[3], message.data[4]]);
        if ambient_raw != 0xFFFF {
            let temp = ambient_raw as f64 * 0.03125 - 273.0;
            signals.push(DecodedSignal::new(
                "AmbientAirTemperature",
                temp,
                "°C",
                ambient_raw as u64,
            ));
        } else {
            signals.push(DecodedSignal::not_available("AmbientAirTemperature", "°C"));
        }

        // Engine Intake Manifold 1 Air Temperature (byte 5)
        let intake_temp = message.data[5];
        if intake_temp != 0xFF {
            let temp = intake_temp as f64 - 40.0;
            signals.push(DecodedSignal::new(
                "EngineIntakeManifold1AirTemperature",
                temp,
                "°C",
                intake_temp as u64,
            ));
        }

        // Road Surface Temperature (bytes 6-7)
        let road_temp_raw = u16::from_le_bytes([message.data[6], message.data[7]]);
        if road_temp_raw != 0xFFFF {
            let temp = road_temp_raw as f64 * 0.03125 - 273.0;
            signals.push(DecodedSignal::new(
                "RoadSurfaceTemperature",
                temp,
                "°C",
                road_temp_raw as u64,
            ));
        }

        Some(DecodedPgn {
            pgn: PGN_AMB,
            name: "AMB",
            description: "Ambient Conditions",
            signals,
        })
    }
}

/// Vehicle Electrical Power 1 (VEP1) decoder
#[derive(Debug, Default)]
pub struct Vep1Decoder;

impl PgnDecoder for Vep1Decoder {
    fn pgn(&self) -> PgnValue {
        PGN_VEP1
    }

    fn name(&self) -> &'static str {
        "VEP1"
    }

    fn description(&self) -> &'static str {
        "Vehicle Electrical Power 1"
    }

    fn decode(&self, message: &J1939Message) -> Option<DecodedPgn> {
        if message.data.len() < 8 {
            return None;
        }

        let mut signals = Vec::new();

        // Net Battery Current (bytes 0-1)
        let current_raw = u16::from_le_bytes([message.data[0], message.data[1]]);
        if current_raw != 0xFFFF {
            let current = current_raw as f64 - 32128.0; // Offset binary
            signals.push(DecodedSignal::new(
                "NetBatteryCurrent",
                current,
                "A",
                current_raw as u64,
            ));
        }

        // Alternator Current (bytes 2-3)
        let alt_current_raw = u16::from_le_bytes([message.data[2], message.data[3]]);
        if alt_current_raw != 0xFFFF {
            let current = alt_current_raw as f64; // 1 A/bit
            signals.push(DecodedSignal::new(
                "AlternatorCurrent",
                current,
                "A",
                alt_current_raw as u64,
            ));
        }

        // Alternator Potential (Voltage) (bytes 4-5)
        let voltage_raw = u16::from_le_bytes([message.data[4], message.data[5]]);
        if voltage_raw != 0xFFFF {
            let voltage = voltage_raw as f64 * 0.05; // 0.05 V/bit
            signals.push(DecodedSignal::new(
                "AlternatorPotential",
                voltage,
                "V",
                voltage_raw as u64,
            ));
        }

        // Electrical Potential (Battery) (bytes 6-7)
        let battery_raw = u16::from_le_bytes([message.data[6], message.data[7]]);
        if battery_raw != 0xFFFF {
            let voltage = battery_raw as f64 * 0.05;
            signals.push(DecodedSignal::new(
                "ElectricalPotential",
                voltage,
                "V",
                battery_raw as u64,
            ));
        } else {
            signals.push(DecodedSignal::not_available("ElectricalPotential", "V"));
        }

        Some(DecodedPgn {
            pgn: PGN_VEP1,
            name: "VEP1",
            description: "Vehicle Electrical Power 1",
            signals,
        })
    }
}

// ============================================================================
// PGN Registry
// ============================================================================

/// Registry of PGN decoders
pub struct PgnRegistry {
    decoders: HashMap<PgnValue, Box<dyn PgnDecoder>>,
}

impl PgnRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            decoders: HashMap::new(),
        }
    }

    /// Create registry with all standard decoders
    pub fn with_standard_decoders() -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(Eec1Decoder));
        registry.register(Box::new(Eec2Decoder));
        registry.register(Box::new(Et1Decoder));
        registry.register(Box::new(Eflp1Decoder));
        registry.register(Box::new(CcvsDecoder));
        registry.register(Box::new(LfeDecoder));
        registry.register(Box::new(AmbDecoder));
        registry.register(Box::new(Vep1Decoder));
        registry
    }

    /// Register a decoder
    pub fn register(&mut self, decoder: Box<dyn PgnDecoder>) {
        self.decoders.insert(decoder.pgn(), decoder);
    }

    /// Get decoder for PGN
    pub fn get_decoder(&self, pgn: PgnValue) -> Option<&dyn PgnDecoder> {
        self.decoders.get(&pgn).map(|d| d.as_ref())
    }

    /// Decode a J1939 message
    pub fn decode(&self, message: &J1939Message) -> Option<DecodedPgn> {
        let pgn = message.header.pgn.value();
        self.get_decoder(pgn)?.decode(message)
    }

    /// Get all registered PGN values
    pub fn registered_pgns(&self) -> Vec<PgnValue> {
        self.decoders.keys().copied().collect()
    }

    /// Get number of registered decoders
    pub fn decoder_count(&self) -> usize {
        self.decoders.len()
    }
}

impl Default for PgnRegistry {
    fn default() -> Self {
        Self::with_standard_decoders()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_message(pgn: u32, data: Vec<u8>) -> J1939Message {
        let pf = ((pgn >> 8) & 0xFF) as u8;
        let ps = (pgn & 0xFF) as u8;
        let dp = ((pgn >> 16) & 0x01) != 0;

        use crate::protocol::j1939::Pgn;

        J1939Message {
            header: crate::protocol::j1939::J1939Header {
                priority: 3,
                pgn: Pgn::from_components(dp, pf, ps),
                source_address: 0,
                destination_address: None,
            },
            data,
            timestamp: chrono::Utc::now(),
            is_multipacket: false,
        }
    }

    #[test]
    fn test_eec1_decoder() {
        let decoder = Eec1Decoder;

        // Test data: Engine speed = 2000 rpm (16000 raw * 0.125)
        // Bytes 3-4: 0x3E80 (16000) little-endian = [0x80, 0x3E]
        let data = vec![0x00, 0x7D, 0x7D, 0x80, 0x3E, 0x00, 0x00, 0x7D];
        let message = create_test_message(PGN_EEC1, data);

        let decoded = decoder.decode(&message).expect("decoding should succeed");
        assert_eq!(decoded.pgn, PGN_EEC1);
        assert_eq!(decoded.name, "EEC1");

        // Find engine speed signal
        let engine_speed = decoded
            .signals
            .iter()
            .find(|s| s.name == "EngineSpeed")
            .expect("operation should succeed");
        assert!((engine_speed.value - 2000.0).abs() < 0.1);
        assert_eq!(engine_speed.unit, "rpm");
    }

    #[test]
    fn test_ccvs_decoder() {
        let decoder = CcvsDecoder;

        // Test data: Vehicle speed = 100 km/h (25600 raw / 256)
        // Bytes 1-2: 0x6400 (25600) little-endian = [0x00, 0x64]
        let data = vec![0x00, 0x00, 0x64, 0x00, 0x64, 0x00, 0x00, 0x00];
        let message = create_test_message(PGN_CCVS, data);

        let decoded = decoder.decode(&message).expect("decoding should succeed");
        assert_eq!(decoded.pgn, PGN_CCVS);

        // Find vehicle speed signal
        let speed = decoded
            .signals
            .iter()
            .find(|s| s.name == "WheelBasedVehicleSpeed")
            .expect("operation should succeed");
        assert!((speed.value - 100.0).abs() < 0.1);
        assert_eq!(speed.unit, "km/h");
    }

    #[test]
    fn test_et1_decoder() {
        let decoder = Et1Decoder;

        // Test data: Coolant temp = 85°C (125 raw - 40 offset)
        let data = vec![125, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let message = create_test_message(PGN_ET1, data);

        let decoded = decoder.decode(&message).expect("decoding should succeed");
        assert_eq!(decoded.pgn, PGN_ET1);

        let coolant = decoded
            .signals
            .iter()
            .find(|s| s.name == "EngineCoolantTemperature")
            .expect("operation should succeed");
        assert!((coolant.value - 85.0).abs() < 0.1);
    }

    #[test]
    fn test_lfe_decoder() {
        let decoder = LfeDecoder;

        // Test data: Fuel rate = 50 L/h (1000 raw * 0.05)
        // Bytes 0-1: 1000 = [0xE8, 0x03]
        let data = vec![0xE8, 0x03, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let message = create_test_message(PGN_LFE, data);

        let decoded = decoder.decode(&message).expect("decoding should succeed");
        assert_eq!(decoded.pgn, PGN_LFE);

        let fuel_rate = decoded
            .signals
            .iter()
            .find(|s| s.name == "EngineFuelRate")
            .expect("operation should succeed");
        assert!((fuel_rate.value - 50.0).abs() < 0.1);
        assert_eq!(fuel_rate.unit, "L/h");
    }

    #[test]
    fn test_pgn_registry() {
        let registry = PgnRegistry::with_standard_decoders();

        assert!(registry.get_decoder(PGN_EEC1).is_some());
        assert!(registry.get_decoder(PGN_CCVS).is_some());
        assert!(registry.get_decoder(PGN_ET1).is_some());
        assert!(registry.get_decoder(PGN_LFE).is_some());
        assert!(registry.get_decoder(PGN_AMB).is_some());
        assert!(registry.get_decoder(PGN_VEP1).is_some());

        // Unknown PGN should return None
        assert!(registry.get_decoder(99999).is_none());
    }

    #[test]
    fn test_registry_decode() {
        let registry = PgnRegistry::default();

        let data = vec![0x00, 0x7D, 0x7D, 0x80, 0x3E, 0x00, 0x00, 0x7D];
        let message = create_test_message(PGN_EEC1, data);

        let decoded = registry.decode(&message).expect("decoding should succeed");
        assert_eq!(decoded.name, "EEC1");
    }

    #[test]
    fn test_decoded_signal_not_available() {
        let signal = DecodedSignal::not_available("TestSignal", "unit");
        assert!(!signal.valid);
        assert!(signal.value.is_nan());
    }

    #[test]
    fn test_vep1_decoder() {
        let decoder = Vep1Decoder;

        // Battery voltage = 14.0V (280 raw * 0.05)
        // Bytes 6-7: 280 = [0x18, 0x01]
        let data = vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x18, 0x01];
        let message = create_test_message(PGN_VEP1, data);

        let decoded = decoder.decode(&message).expect("decoding should succeed");

        let voltage = decoded
            .signals
            .iter()
            .find(|s| s.name == "ElectricalPotential")
            .expect("operation should succeed");
        assert!((voltage.value - 14.0).abs() < 0.1);
        assert_eq!(voltage.unit, "V");
    }

    #[test]
    fn test_amb_decoder() {
        let decoder = AmbDecoder;

        // Barometric pressure = 101.5 kPa (203 raw * 0.5)
        let data = vec![203, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let message = create_test_message(PGN_AMB, data);

        let decoded = decoder.decode(&message).expect("decoding should succeed");

        let baro = decoded
            .signals
            .iter()
            .find(|s| s.name == "BarometricPressure")
            .expect("operation should succeed");
        assert!((baro.value - 101.5).abs() < 0.1);
        assert_eq!(baro.unit, "kPa");
    }

    #[test]
    fn test_eec2_decoder() {
        let decoder = Eec2Decoder;

        // Accelerator position = 50% (125 raw * 0.4)
        let data = vec![0x00, 125, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let message = create_test_message(PGN_EEC2, data);

        let decoded = decoder.decode(&message).expect("decoding should succeed");

        let pedal = decoded
            .signals
            .iter()
            .find(|s| s.name == "AcceleratorPedalPosition1")
            .expect("operation should succeed");
        assert!((pedal.value - 50.0).abs() < 0.1);
        assert_eq!(pedal.unit, "%");
    }

    #[test]
    fn test_eflp1_decoder() {
        let decoder = Eflp1Decoder;

        // Oil pressure = 400 kPa (100 raw * 4.0)
        let data = vec![0xFF, 0xFF, 0xFF, 100, 0xFF, 0xFF, 0xFF, 0xFF];
        let message = create_test_message(PGN_EFLP1, data);

        let decoded = decoder.decode(&message).expect("decoding should succeed");

        let pressure = decoded
            .signals
            .iter()
            .find(|s| s.name == "EngineOilPressure")
            .expect("operation should succeed");
        assert!((pressure.value - 400.0).abs() < 0.1);
        assert_eq!(pressure.unit, "kPa");
    }
}
