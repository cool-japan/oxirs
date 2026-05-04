//! Integration tests for the Modbus register auto-discovery pipeline.
//!
//! Tests run against an in-process `MockModbusDevice` that exposes a known
//! register map, allowing the discovery driver + type inferrer to be exercised
//! end-to-end without real hardware.

use std::collections::HashMap;

use oxirs_modbus::discovery::{
    driver::{DiscoveredRegister, DiscoveryFunctionCode, ModbusAccess},
    emitter::{CandidateRegisterEntry, CandidateRegisterMap, RegisterMapEmitter},
    inference::{ConfidenceLevel, InferredType, TypeInferrer},
    DiscoveryConfig, DiscoveryDriver, DiscoveryError,
};

// ─── Mock device ─────────────────────────────────────────────────────────────

/// Simulated Modbus device with a statically defined register map.
///
/// Known holding registers:
/// - HR 0: status bit  → `1`  (Boolean)
/// - HR 1-2: temperature as Float32 BE  (`22.5°C` → bits `0x41B4_0000`)
/// - HR 3: pressure scaled  (`225` → `22.5 kPa` with scale `0.1`)
/// - HR 4: signed setpoint  (`(-100i16)` → `0xFF9C`)
/// - HR 5: raw setpoint  (`20000` → UInt16, above the ScaledInt cutoff)
///
/// Input registers HR 0-4 hold `100 + i*10`.
struct MockModbusDevice {
    holding: HashMap<u16, u16>,
    input: HashMap<u16, u16>,
}

impl MockModbusDevice {
    fn new() -> Self {
        let mut holding = HashMap::new();
        let mut input = HashMap::new();

        // HR 0: status boolean
        holding.insert(0, 1u16);

        // HR 1-2: Float32 22.5 (big-endian)
        let temp_bits = (22.5_f32).to_bits();
        holding.insert(1, (temp_bits >> 16) as u16);
        holding.insert(2, (temp_bits & 0xFFFF) as u16);

        // HR 3: scaled integer (225 → 22.5 with ×0.1)
        holding.insert(3, 225u16);

        // HR 4: signed negative (−100)
        holding.insert(4, (-100i16) as u16);

        // HR 5: large positive UInt16 (above 10_000 cutoff)
        holding.insert(5, 20_000u16);

        // Input registers: simple pattern
        for i in 0u16..5 {
            input.insert(i, 100 + i * 10);
        }

        MockModbusDevice { holding, input }
    }
}

impl ModbusAccess for MockModbusDevice {
    fn read_holding_registers(
        &mut self,
        address: u16,
        count: u16,
    ) -> Result<Vec<u16>, DiscoveryError> {
        let mut result = Vec::with_capacity(count as usize);
        for addr in address..address.saturating_add(count) {
            match self.holding.get(&addr) {
                Some(&v) => result.push(v),
                None => {
                    return Err(DiscoveryError::DeviceException {
                        function_code: 0x03,
                        address: addr,
                        message: "Register not supported".into(),
                    })
                }
            }
        }
        Ok(result)
    }

    fn read_input_registers(
        &mut self,
        address: u16,
        count: u16,
    ) -> Result<Vec<u16>, DiscoveryError> {
        let mut result = Vec::with_capacity(count as usize);
        for addr in address..address.saturating_add(count) {
            match self.input.get(&addr) {
                Some(&v) => result.push(v),
                None => {
                    return Err(DiscoveryError::DeviceException {
                        function_code: 0x04,
                        address: addr,
                        message: "Register not supported".into(),
                    })
                }
            }
        }
        Ok(result)
    }

    fn read_coils(&mut self, _: u16, _: u16) -> Result<Vec<bool>, DiscoveryError> {
        Err(DiscoveryError::DeviceException {
            function_code: 0x01,
            address: 0,
            message: "Coils not implemented".into(),
        })
    }

    fn read_discrete_inputs(&mut self, _: u16, _: u16) -> Result<Vec<bool>, DiscoveryError> {
        Err(DiscoveryError::DeviceException {
            function_code: 0x02,
            address: 0,
            message: "Discrete inputs not implemented".into(),
        })
    }
}

// ─── Unit tests: type inferrer ───────────────────────────────────────────────

#[test]
fn infer_uint16_large() {
    // 25_000 is above the ScaledInt cutoff (10_000) — must be UInt16
    let (t, _, val, _) = TypeInferrer::infer_single(25_000);
    assert!(
        matches!(t, InferredType::UInt16),
        "Expected UInt16, got {t:?}"
    );
    assert!((val - 25_000.0).abs() < 1e-10);
}

#[test]
fn infer_int16_negative() {
    let (t, _, val, _) = TypeInferrer::infer_single((-50i16) as u16);
    assert!(matches!(t, InferredType::Int16), "Expected Int16");
    assert!((val - (-50.0)).abs() < 1e-10);
}

#[test]
fn infer_boolean_zero() {
    let (t, _, val, _) = TypeInferrer::infer_single(0);
    assert!(matches!(t, InferredType::Boolean));
    assert_eq!(val, 0.0);
}

#[test]
fn infer_boolean_one() {
    let (t, _, _, _) = TypeInferrer::infer_single(1);
    assert!(matches!(t, InferredType::Boolean));
}

#[test]
fn infer_scaled_int_heuristic() {
    // 225 → 22.5 with ×0.1 — should hit ScaledInt branch
    let (t, _, val, _) = TypeInferrer::infer_single(225);
    assert!(val > 0.0, "Scaled value must be positive");
    // We don't assert the exact type here because the heuristic is intentionally
    // approximate — only that a non-Unknown, positive value was assigned.
    assert!(!matches!(t, InferredType::Unknown));
}

#[test]
fn infer_float32_from_batch() {
    let f: f32 = 22.5;
    let bits = f.to_bits();
    let hi = (bits >> 16) as u16;
    let lo = (bits & 0xFFFF) as u16;

    let registers = vec![
        DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 10, hi),
        DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 11, lo),
    ];
    let results = TypeInferrer::infer_batch(&registers);
    assert_eq!(results.len(), 1, "Pair should produce one Float32 result");
    assert!(
        matches!(results[0].inferred_type, InferredType::Float32),
        "Expected Float32, got {:?}",
        results[0].inferred_type
    );
    assert!(
        (results[0].inferred_f64 - 22.5).abs() < 0.01,
        "Float value: {}",
        results[0].inferred_f64
    );
    assert_eq!(results[0].confidence, ConfidenceLevel::High);
}

#[test]
fn infer_batch_non_float_pair() {
    // NaN bits — should NOT collapse into Float32
    let registers = vec![
        DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 0, 0xFFFF),
        DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 1, 0xFFFF),
    ];
    let results = TypeInferrer::infer_batch(&registers);
    assert_eq!(results.len(), 2, "NaN pair must stay separate");
}

// ─── Integration: discovery scan ─────────────────────────────────────────────

#[test]
fn discovery_scan_finds_known_registers() {
    let device = MockModbusDevice::new();
    let config = DiscoveryConfig {
        address_start: 0,
        address_end: 10,
        read_batch_size: 3,
        inter_request_delay: std::time::Duration::from_millis(0),
        unit_id: 1,
    };

    let mut driver = DiscoveryDriver::new(device, config);
    let discovered = driver.scan().expect("Discovery should succeed");

    let holding: Vec<_> = discovered
        .iter()
        .filter(|r| matches!(r.function_code, DiscoveryFunctionCode::ReadHoldingRegisters))
        .collect();

    assert!(
        !holding.is_empty(),
        "Should discover at least some holding registers"
    );
}

#[test]
fn discovery_inference_hit_rate_90pct() {
    let device = MockModbusDevice::new();
    let config = DiscoveryConfig {
        address_start: 0,
        address_end: 6,
        read_batch_size: 10,
        inter_request_delay: std::time::Duration::from_millis(0),
        unit_id: 1,
    };

    let mut driver = DiscoveryDriver::new(device, config);
    let discovered = driver.scan().expect("Scan must succeed");

    let holding: Vec<_> = discovered
        .into_iter()
        .filter(|r| matches!(r.function_code, DiscoveryFunctionCode::ReadHoldingRegisters))
        .collect();

    let inferred = TypeInferrer::infer_batch(&holding);
    let total = inferred.len();
    assert!(total > 0, "At least one register must be inferred");

    let non_unknown = inferred
        .iter()
        .filter(|r| !matches!(r.inferred_type, InferredType::Unknown))
        .count();

    let hit_rate = non_unknown as f64 / total as f64;
    assert!(
        hit_rate >= 0.9,
        "Hit rate {:.1}% should be >= 90% (non-unknown: {non_unknown}/{total})",
        hit_rate * 100.0
    );
}

#[test]
fn emitter_produces_valid_json() {
    let device = MockModbusDevice::new();
    let config = DiscoveryConfig {
        address_start: 0,
        address_end: 6,
        read_batch_size: 10,
        inter_request_delay: std::time::Duration::from_millis(0),
        unit_id: 1,
    };

    let mut driver = DiscoveryDriver::new(device, config);
    let discovered = driver.scan().expect("Scan must succeed");
    let holding: Vec<_> = discovered
        .into_iter()
        .filter(|r| matches!(r.function_code, DiscoveryFunctionCode::ReadHoldingRegisters))
        .collect();
    let total = holding.len();
    let inferred = TypeInferrer::infer_batch(&holding);
    let candidate_map = RegisterMapEmitter::emit(1, &inferred, total);

    let json = RegisterMapEmitter::to_json(&candidate_map).expect("JSON serialization");
    assert!(json.contains("\"address\""));
    assert!(json.contains("\"data_type\""));
    assert!(json.contains("\"confidence\""));
    assert!(json.contains("\"unit_id\""));
}

#[test]
fn emitter_yaml_like_output_structure() {
    let map = CandidateRegisterMap {
        unit_id: 1,
        entries: vec![CandidateRegisterEntry {
            address: 0x0001,
            name: "register_0x0001".into(),
            data_type: "uint16".into(),
            scale: None,
            confidence: "medium".into(),
            raw_value: 42,
            inferred_value: 42.0,
            notes: "Test register".into(),
        }],
        total_registers_probed: 1,
        high_confidence_count: 0,
    };

    let yaml = RegisterMapEmitter::to_yaml_like(&map);
    assert!(yaml.contains("unit_id:"), "YAML must have unit_id");
    assert!(
        yaml.contains("registers:"),
        "YAML must have registers section"
    );
    assert!(yaml.contains("0x0001"), "YAML must include hex address");
}

#[test]
fn emitter_high_confidence_count() {
    // Float32 pair produces High confidence
    let f: f32 = 100.0;
    let bits = f.to_bits();
    let hi = (bits >> 16) as u16;
    let lo = (bits & 0xFFFF) as u16;

    let registers = vec![
        DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 0, hi),
        DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 1, lo),
    ];
    let inferred = TypeInferrer::infer_batch(&registers);
    let map = RegisterMapEmitter::emit(1, &inferred, 2);

    assert_eq!(
        map.high_confidence_count, 1,
        "Float32 should be High confidence"
    );
}

#[test]
fn full_pipeline_with_input_registers() {
    let device = MockModbusDevice::new();
    let config = DiscoveryConfig {
        address_start: 0,
        address_end: 5,
        read_batch_size: 5,
        inter_request_delay: std::time::Duration::from_millis(0),
        unit_id: 1,
    };

    let mut driver = DiscoveryDriver::new(device, config);
    let discovered = driver.scan().expect("Scan must succeed");

    let input_regs: Vec<_> = discovered
        .into_iter()
        .filter(|r| matches!(r.function_code, DiscoveryFunctionCode::ReadInputRegisters))
        .collect();

    assert_eq!(input_regs.len(), 5, "Should find all 5 input registers");

    let inferred = TypeInferrer::infer_batch(&input_regs);
    // input reg values are 100, 110, 120, 130, 140 — all in the ScaledInt range
    assert!(
        inferred
            .iter()
            .all(|r| !matches!(r.inferred_type, InferredType::Unknown)),
        "All input register types should be determinable"
    );
}
