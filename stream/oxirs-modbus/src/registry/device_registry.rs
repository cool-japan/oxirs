//! Modbus device registry - runtime catalog of known devices
//!
//! Enables multi-device management with named register maps, polling
//! configurations, and connection metadata. Each device is uniquely
//! identified by a string ID and optionally by its Modbus unit ID
//! on a shared bus.

use crate::error::{ModbusError, ModbusResult};
use crate::mapping::{ByteOrder, ModbusDataType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Classification of a Modbus device by its application domain.
///
/// Used for documentation, display filtering, and semantic tagging.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceType {
    /// Electricity meter (kWh, power, current, voltage)
    EnergyMeter,
    /// Temperature sensor or transmitter
    TemperatureSensor,
    /// Pressure sensor or transmitter
    PressureSensor,
    /// Motor drive or frequency converter
    MotorController,
    /// Programmable Logic Controller (PLC)
    Plc,
    /// Flow meter (liquid or gas)
    FlowMeter,
    /// Level sensor or transmitter
    LevelSensor,
    /// Humidity / moisture sensor
    HumiditySensor,
    /// Variable frequency drive (VFD/inverter)
    FrequencyDrive,
    /// Generic custom device with free-form label
    Custom(String),
}

impl DeviceType {
    /// Human-readable display label for this device type.
    pub fn label(&self) -> &str {
        match self {
            DeviceType::EnergyMeter => "Energy Meter",
            DeviceType::TemperatureSensor => "Temperature Sensor",
            DeviceType::PressureSensor => "Pressure Sensor",
            DeviceType::MotorController => "Motor Controller",
            DeviceType::Plc => "PLC",
            DeviceType::FlowMeter => "Flow Meter",
            DeviceType::LevelSensor => "Level Sensor",
            DeviceType::HumiditySensor => "Humidity Sensor",
            DeviceType::FrequencyDrive => "Frequency Drive",
            DeviceType::Custom(s) => s.as_str(),
        }
    }
}

/// Register type within a device's register map.
///
/// Mirrors [`crate::mapping::RegisterType`] but is used specifically within the
/// device-registry context to keep the registry self-contained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterType {
    /// Read/Write holding registers (FC 0x03 / 0x06 / 0x10)
    HoldingRegister,
    /// Read-only input registers (FC 0x04)
    InputRegister,
    /// Read/Write coil (single-bit output) (FC 0x01 / 0x05 / 0x0F)
    Coil,
    /// Read-only discrete input (single-bit) (FC 0x02)
    DiscreteInput,
}

impl From<RegisterType> for crate::mapping::RegisterType {
    fn from(rt: RegisterType) -> Self {
        match rt {
            RegisterType::HoldingRegister => crate::mapping::RegisterType::Holding,
            RegisterType::InputRegister => crate::mapping::RegisterType::Input,
            RegisterType::Coil => crate::mapping::RegisterType::Coil,
            RegisterType::DiscreteInput => crate::mapping::RegisterType::DiscreteInput,
        }
    }
}

impl From<crate::mapping::RegisterType> for RegisterType {
    fn from(rt: crate::mapping::RegisterType) -> Self {
        match rt {
            crate::mapping::RegisterType::Holding => RegisterType::HoldingRegister,
            crate::mapping::RegisterType::Input => RegisterType::InputRegister,
            crate::mapping::RegisterType::Coil => RegisterType::Coil,
            crate::mapping::RegisterType::DiscreteInput => RegisterType::DiscreteInput,
        }
    }
}

/// Definition of a single named signal within a device's register space.
///
/// Combines the register address with semantic metadata so that raw Modbus
/// readings can be automatically decoded, scaled, and annotated with RDF
/// predicates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterDefinition {
    /// Unique signal name within the device (e.g. `"active_power"`)
    pub name: String,
    /// Short human-readable description
    pub description: String,
    /// Modbus register address (0-based, Modbus standard addressing)
    pub address: u16,
    /// Register space (holding, input, coil, discrete_input)
    pub register_type: RegisterType,
    /// Encoding / data type for this signal
    pub data_type: ModbusDataType,
    /// Byte order when reassembling multi-register words
    #[serde(default)]
    pub byte_order: ByteOrder,
    /// Multiplicative scale factor applied after decoding
    /// (`physical_value = raw_value * scale_factor + offset`)
    #[serde(default = "default_scale")]
    pub scale_factor: f64,
    /// Additive offset applied after scaling
    #[serde(default)]
    pub offset: f64,
    /// Physical unit symbol (SI preferred, e.g. `"°C"`, `"kW"`, `"bar"`)
    pub unit: Option<String>,
    /// If `true`, the register must not be written
    #[serde(default = "default_read_only")]
    pub read_only: bool,
    /// Optional RDF predicate IRI this signal maps to
    /// (e.g. `"http://qudt.org/vocab/unit/W"`)
    pub rdf_predicate: Option<String>,
    /// Lower bound of the expected physical value range (for validation)
    pub min_value: Option<f64>,
    /// Upper bound of the expected physical value range (for validation)
    pub max_value: Option<f64>,
    /// Deadband: only emit a new reading when the change exceeds this threshold
    pub deadband: Option<f64>,
}

fn default_scale() -> f64 {
    1.0
}

fn default_read_only() -> bool {
    false
}

impl RegisterDefinition {
    /// Create a new register definition with required fields.
    pub fn new(
        name: impl Into<String>,
        address: u16,
        register_type: RegisterType,
        data_type: ModbusDataType,
    ) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            address,
            register_type,
            data_type,
            byte_order: ByteOrder::BigEndian,
            scale_factor: 1.0,
            offset: 0.0,
            unit: None,
            read_only: false,
            rdf_predicate: None,
            min_value: None,
            max_value: None,
            deadband: None,
        }
    }

    /// Set human-readable description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set scale factor and offset for physical conversion.
    pub fn with_scaling(mut self, scale_factor: f64, offset: f64) -> Self {
        self.scale_factor = scale_factor;
        self.offset = offset;
        self
    }

    /// Set measurement unit symbol.
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Mark as read-only.
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Associate an RDF predicate IRI.
    pub fn with_rdf_predicate(mut self, predicate: impl Into<String>) -> Self {
        self.rdf_predicate = Some(predicate.into());
        self
    }

    /// Set value range for validation.
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    /// Set deadband threshold.
    pub fn with_deadband(mut self, deadband: f64) -> Self {
        self.deadband = Some(deadband);
        self
    }

    /// Apply scale factor and offset to a raw numeric value.
    pub fn apply_scaling(&self, raw: f64) -> f64 {
        raw * self.scale_factor + self.offset
    }

    /// Number of Modbus registers consumed by this signal.
    pub fn register_count(&self) -> usize {
        self.data_type.register_count()
    }

    /// Validate that a physical value is within the declared range.
    ///
    /// Returns `None` when no range is configured (unconstrained).
    pub fn is_in_range(&self, physical_value: f64) -> Option<bool> {
        match (self.min_value, self.max_value) {
            (Some(min), Some(max)) => Some(physical_value >= min && physical_value <= max),
            (Some(min), None) => Some(physical_value >= min),
            (None, Some(max)) => Some(physical_value <= max),
            (None, None) => None,
        }
    }
}

/// Named register map belonging to a device.
///
/// Groups all [`RegisterDefinition`]s for one logical device and provides
/// convenient lookup methods.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegisterMap {
    /// All register definitions in declaration order
    pub registers: Vec<RegisterDefinition>,
}

impl RegisterMap {
    /// Create an empty register map.
    pub fn new() -> Self {
        Self {
            registers: Vec::new(),
        }
    }

    /// Add a register definition, returning `self` for chaining.
    pub fn with_register(mut self, reg: RegisterDefinition) -> Self {
        self.registers.push(reg);
        self
    }

    /// Append a register definition in-place.
    pub fn add(&mut self, reg: RegisterDefinition) {
        self.registers.push(reg);
    }

    /// Look up a register by name (case-sensitive).
    pub fn find_by_name(&self, name: &str) -> Option<&RegisterDefinition> {
        self.registers.iter().find(|r| r.name == name)
    }

    /// Find all registers at a given address (there may be multiple with
    /// different bit interpretations).
    pub fn find_by_address(&self, address: u16) -> Vec<&RegisterDefinition> {
        self.registers
            .iter()
            .filter(|r| r.address == address)
            .collect()
    }

    /// Iterate over registers in a specific register space.
    pub fn by_type(
        &self,
        register_type: RegisterType,
    ) -> impl Iterator<Item = &RegisterDefinition> {
        self.registers
            .iter()
            .filter(move |r| r.register_type == register_type)
    }

    /// Total count of defined signals.
    pub fn len(&self) -> usize {
        self.registers.len()
    }

    /// True when no signals are defined.
    pub fn is_empty(&self) -> bool {
        self.registers.is_empty()
    }
}

/// Complete descriptor of a Modbus device known to the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModbusDevice {
    /// Unique logical identifier (e.g. `"meter_01"`, `"pump_a"`)
    pub device_id: String,
    /// Modbus unit / slave ID (1-247; 255 = broadcast)
    pub unit_id: u8,
    /// Optional IP address for Modbus TCP
    pub ip_address: Option<String>,
    /// TCP port (default 502)
    pub port: Option<u16>,
    /// Device type / application domain
    pub device_type: DeviceType,
    /// Register map with all signal definitions
    pub register_map: RegisterMap,
    /// Polling interval in milliseconds
    #[serde(default = "default_poll_ms")]
    pub poll_interval_ms: u64,
    /// Per-request timeout in milliseconds
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    /// Optional free-form tags for categorisation
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

fn default_poll_ms() -> u64 {
    1000
}

fn default_timeout_ms() -> u64 {
    5000
}

impl ModbusDevice {
    /// Create a new device descriptor with required fields.
    pub fn new(device_id: impl Into<String>, unit_id: u8, device_type: DeviceType) -> Self {
        Self {
            device_id: device_id.into(),
            unit_id,
            ip_address: None,
            port: None,
            device_type,
            register_map: RegisterMap::new(),
            poll_interval_ms: 1000,
            timeout_ms: 5000,
            tags: HashMap::new(),
        }
    }

    /// Set TCP connection details.
    pub fn with_tcp(mut self, ip: impl Into<String>, port: u16) -> Self {
        self.ip_address = Some(ip.into());
        self.port = Some(port);
        self
    }

    /// Set register map.
    pub fn with_register_map(mut self, map: RegisterMap) -> Self {
        self.register_map = map;
        self
    }

    /// Set polling interval.
    pub fn with_poll_interval_ms(mut self, ms: u64) -> Self {
        self.poll_interval_ms = ms;
        self
    }

    /// Set per-request timeout.
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Add a tag key-value pair.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Effective TCP port (defaults to 502 when unset).
    pub fn effective_port(&self) -> u16 {
        self.port.unwrap_or(502)
    }

    /// Validate device unit ID is within Modbus specification (1-247).
    pub fn validate(&self) -> ModbusResult<()> {
        if self.unit_id == 0 || (self.unit_id > 247 && self.unit_id != 255) {
            return Err(ModbusError::Config(format!(
                "Device '{}': unit_id {} is out of range [1-247] or broadcast (255)",
                self.device_id, self.unit_id
            )));
        }
        if self.poll_interval_ms == 0 {
            return Err(ModbusError::Config(format!(
                "Device '{}': poll_interval_ms must be > 0",
                self.device_id
            )));
        }
        Ok(())
    }
}

/// Thread-safe registry of known Modbus devices.
///
/// Devices are keyed by their `device_id`. Multiple devices may share the
/// same `unit_id` if they are on different TCP endpoints; the registry does
/// not enforce uniqueness of `unit_id`.
///
/// # Example
///
/// ```rust
/// use oxirs_modbus::registry::{
///     DeviceRegistry, ModbusDevice, DeviceType, RegisterMap, RegisterDefinition,
///     RegisterType as DeviceRegisterType,
/// };
/// use oxirs_modbus::mapping::ModbusDataType;
///
/// let mut registry = DeviceRegistry::new();
///
/// let map = RegisterMap::new()
///     .with_register(
///         RegisterDefinition::new("temperature", 30001, DeviceRegisterType::InputRegister, ModbusDataType::Int16)
///             .with_scaling(0.1, 0.0)
///             .with_unit("°C"),
///     );
///
/// let device = ModbusDevice::new("sensor_01", 1, DeviceType::TemperatureSensor)
///     .with_tcp("192.168.1.10", 502)
///     .with_register_map(map);
///
/// registry.register(device).unwrap();
///
/// assert_eq!(registry.device_count(), 1);
/// ```
pub struct DeviceRegistry {
    /// Map from device_id → device descriptor
    devices: HashMap<String, ModbusDevice>,
}

impl DeviceRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            devices: HashMap::new(),
        }
    }

    /// Register a device. Returns an error if the device_id is already taken.
    pub fn register(&mut self, device: ModbusDevice) -> ModbusResult<()> {
        device.validate()?;

        if self.devices.contains_key(&device.device_id) {
            return Err(ModbusError::Config(format!(
                "Device '{}' is already registered; call update() to replace it",
                device.device_id
            )));
        }

        self.devices.insert(device.device_id.clone(), device);
        Ok(())
    }

    /// Register or replace a device unconditionally.
    pub fn upsert(&mut self, device: ModbusDevice) -> ModbusResult<()> {
        device.validate()?;
        self.devices.insert(device.device_id.clone(), device);
        Ok(())
    }

    /// Remove a device from the registry. Returns the removed device if
    /// it existed.
    pub fn remove(&mut self, device_id: &str) -> Option<ModbusDevice> {
        self.devices.remove(device_id)
    }

    /// Look up a device by ID.
    pub fn get(&self, device_id: &str) -> Option<&ModbusDevice> {
        self.devices.get(device_id)
    }

    /// Mutably access a device by ID.
    pub fn get_mut(&mut self, device_id: &str) -> Option<&mut ModbusDevice> {
        self.devices.get_mut(device_id)
    }

    /// List all registered devices in arbitrary order.
    pub fn list(&self) -> Vec<&ModbusDevice> {
        self.devices.values().collect()
    }

    /// List devices sorted by `device_id` for deterministic output.
    pub fn list_sorted(&self) -> Vec<&ModbusDevice> {
        let mut v: Vec<&ModbusDevice> = self.devices.values().collect();
        v.sort_by(|a, b| a.device_id.cmp(&b.device_id));
        v
    }

    /// Find all devices sharing a given Modbus unit ID.
    pub fn find_by_unit_id(&self, unit_id: u8) -> Vec<&ModbusDevice> {
        self.devices
            .values()
            .filter(|d| d.unit_id == unit_id)
            .collect()
    }

    /// Find devices matching a tag key-value pair.
    pub fn find_by_tag(&self, key: &str, value: &str) -> Vec<&ModbusDevice> {
        self.devices
            .values()
            .filter(|d| d.tags.get(key).map(|v| v.as_str()) == Some(value))
            .collect()
    }

    /// Find devices of a specific type.
    pub fn find_by_type(&self, device_type: &DeviceType) -> Vec<&ModbusDevice> {
        self.devices
            .values()
            .filter(|d| &d.device_type == device_type)
            .collect()
    }

    /// Number of registered devices.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// True when the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    /// Serialize the entire registry to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let map: HashMap<&str, &ModbusDevice> =
            self.devices.iter().map(|(k, v)| (k.as_str(), v)).collect();
        serde_json::to_string_pretty(&map)
    }

    /// Deserialize a registry from JSON (created by `to_json`).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let devices: HashMap<String, ModbusDevice> = serde_json::from_str(json)?;
        Ok(Self { devices })
    }
}

impl Default for DeviceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::ModbusDataType;

    fn make_temperature_device(id: &str, unit_id: u8) -> ModbusDevice {
        let map = RegisterMap::new().with_register(
            RegisterDefinition::new(
                "temperature",
                30001,
                RegisterType::InputRegister,
                ModbusDataType::Int16,
            )
            .with_scaling(0.1, 0.0)
            .with_unit("°C")
            .with_range(-40.0, 85.0)
            .read_only(),
        );

        ModbusDevice::new(id, unit_id, DeviceType::TemperatureSensor)
            .with_tcp("192.168.1.10", 502)
            .with_register_map(map)
            .with_poll_interval_ms(500)
    }

    #[test]
    fn test_register_and_lookup() {
        let mut registry = DeviceRegistry::new();
        let device = make_temperature_device("sensor_01", 1);
        registry.register(device).unwrap();

        assert_eq!(registry.device_count(), 1);

        let found = registry.get("sensor_01").unwrap();
        assert_eq!(found.unit_id, 1);
        assert_eq!(found.poll_interval_ms, 500);
    }

    #[test]
    fn test_duplicate_registration_fails() {
        let mut registry = DeviceRegistry::new();
        registry
            .register(make_temperature_device("dev_a", 1))
            .unwrap();
        let result = registry.register(make_temperature_device("dev_a", 2));
        assert!(result.is_err());
    }

    #[test]
    fn test_upsert_replaces() {
        let mut registry = DeviceRegistry::new();
        registry
            .upsert(make_temperature_device("dev_a", 1))
            .unwrap();
        registry
            .upsert(make_temperature_device("dev_a", 2))
            .unwrap();
        assert_eq!(registry.get("dev_a").unwrap().unit_id, 2);
    }

    #[test]
    fn test_remove() {
        let mut registry = DeviceRegistry::new();
        registry
            .register(make_temperature_device("dev_x", 5))
            .unwrap();
        assert_eq!(registry.device_count(), 1);

        let removed = registry.remove("dev_x");
        assert!(removed.is_some());
        assert_eq!(registry.device_count(), 0);
    }

    #[test]
    fn test_find_by_unit_id() {
        let mut registry = DeviceRegistry::new();
        registry
            .register(make_temperature_device("dev_1", 3))
            .unwrap();
        registry
            .register(make_temperature_device("dev_2", 3))
            .unwrap();
        registry
            .register(make_temperature_device("dev_3", 7))
            .unwrap();

        let found = registry.find_by_unit_id(3);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_find_by_type() {
        let mut registry = DeviceRegistry::new();
        registry.register(make_temperature_device("t1", 1)).unwrap();
        registry.register(make_temperature_device("t2", 2)).unwrap();

        let mut plc = ModbusDevice::new("plc_1", 10, DeviceType::Plc)
            .with_tcp("10.0.0.1", 502)
            .with_register_map(RegisterMap::new());
        plc.poll_interval_ms = 2000;
        registry.upsert(plc).unwrap();

        let temp_devices = registry.find_by_type(&DeviceType::TemperatureSensor);
        assert_eq!(temp_devices.len(), 2);

        let plcs = registry.find_by_type(&DeviceType::Plc);
        assert_eq!(plcs.len(), 1);
    }

    #[test]
    fn test_find_by_tag() {
        let mut registry = DeviceRegistry::new();
        let device = make_temperature_device("tagged_dev", 4)
            .with_tag("location", "building_a")
            .with_tag("floor", "2");
        registry.register(device).unwrap();

        let found = registry.find_by_tag("location", "building_a");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].device_id, "tagged_dev");

        let not_found = registry.find_by_tag("location", "building_z");
        assert!(not_found.is_empty());
    }

    #[test]
    fn test_register_definition_scaling() {
        let reg = RegisterDefinition::new(
            "pressure",
            40010,
            RegisterType::HoldingRegister,
            ModbusDataType::Int16,
        )
        .with_scaling(0.01, 0.0)
        .with_range(0.0, 16.0);

        // Raw value 1000 → 10.0 bar
        let physical = reg.apply_scaling(1000.0);
        assert!((physical - 10.0).abs() < 1e-9);

        assert_eq!(reg.is_in_range(10.0), Some(true));
        assert_eq!(reg.is_in_range(20.0), Some(false));
    }

    #[test]
    fn test_register_map_lookups() {
        let map = RegisterMap::new()
            .with_register(
                RegisterDefinition::new(
                    "voltage",
                    30001,
                    RegisterType::InputRegister,
                    ModbusDataType::Uint16,
                )
                .with_unit("V"),
            )
            .with_register(
                RegisterDefinition::new(
                    "current",
                    30002,
                    RegisterType::InputRegister,
                    ModbusDataType::Uint16,
                )
                .with_unit("A"),
            );

        assert_eq!(map.len(), 2);
        assert!(map.find_by_name("voltage").is_some());
        assert!(map.find_by_name("current").is_some());
        assert!(map.find_by_name("power").is_none());

        let input_regs: Vec<_> = map.by_type(RegisterType::InputRegister).collect();
        assert_eq!(input_regs.len(), 2);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut registry = DeviceRegistry::new();
        registry.register(make_temperature_device("s1", 1)).unwrap();
        registry.register(make_temperature_device("s2", 2)).unwrap();

        let json = registry.to_json().unwrap();
        let restored = DeviceRegistry::from_json(&json).unwrap();

        assert_eq!(restored.device_count(), 2);
        assert!(restored.get("s1").is_some());
        assert!(restored.get("s2").is_some());
    }

    #[test]
    fn test_invalid_unit_id_rejected() {
        let mut registry = DeviceRegistry::new();
        let mut bad_device = make_temperature_device("bad", 0);
        bad_device.unit_id = 0; // invalid

        let result = registry.register(bad_device);
        assert!(result.is_err());
    }

    #[test]
    fn test_device_type_labels() {
        assert_eq!(DeviceType::EnergyMeter.label(), "Energy Meter");
        assert_eq!(DeviceType::Plc.label(), "PLC");
        assert_eq!(DeviceType::Custom("Inverter".into()).label(), "Inverter");
    }

    #[test]
    fn test_list_sorted() {
        let mut registry = DeviceRegistry::new();
        registry
            .register(make_temperature_device("zzz", 3))
            .unwrap();
        registry
            .register(make_temperature_device("aaa", 1))
            .unwrap();
        registry
            .register(make_temperature_device("mmm", 2))
            .unwrap();

        let sorted = registry.list_sorted();
        assert_eq!(sorted[0].device_id, "aaa");
        assert_eq!(sorted[1].device_id, "mmm");
        assert_eq!(sorted[2].device_id, "zzz");
    }
}
