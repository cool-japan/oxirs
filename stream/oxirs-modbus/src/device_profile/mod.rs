//! Modbus device profile — structured register map with metadata
//!
//! A `DeviceProfile` describes a Modbus device's complete register map,
//! including:
//! - Logical register descriptors (address, data type, scaling, units)
//! - Read/write access flags
//! - JSON and TOML serialisation
//!
//! Multiple profiles are managed through a `DeviceProfileRegistry`.

use crate::datatype::{Endianness, ModbusDataTypeKind};
use crate::error::{ModbusError, ModbusResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── AccessMode ────────────────────────────────────────────────────────────────

/// Read/write access mode for a register entry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessMode {
    /// Register is read-only (input registers, most sensor readings)
    #[default]
    ReadOnly,
    /// Register is write-only (command outputs, rarely used)
    WriteOnly,
    /// Register supports both reads and writes (holding registers)
    ReadWrite,
}

impl std::fmt::Display for AccessMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccessMode::ReadOnly => write!(f, "read_only"),
            AccessMode::WriteOnly => write!(f, "write_only"),
            AccessMode::ReadWrite => write!(f, "read_write"),
        }
    }
}

// ── RegisterEntry ────────────────────────────────────────────────────────────

/// A single register entry in a device profile's register map.
///
/// Combines the physical Modbus address with full semantic metadata for
/// automatic decoding, scaling, and RDF annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterEntry {
    /// Unique signal name within the device (e.g. `"active_power_l1"`)
    pub name: String,
    /// Human-readable description
    #[serde(default)]
    pub description: String,
    /// Modbus register address (0-based)
    pub address: u16,
    /// Data type and encoding for this register
    pub data_type: ModbusDataTypeKind,
    /// Byte / word ordering for multi-register types
    #[serde(default)]
    pub endianness: Endianness,
    /// Linear scale factor: `physical = raw * scale_factor + offset`
    #[serde(default = "default_scale_factor")]
    pub scale_factor: f64,
    /// Additive offset applied after scaling
    #[serde(default)]
    pub offset: f64,
    /// SI unit symbol (e.g. `"kW"`, `"°C"`, `"bar"`)
    pub unit: Option<String>,
    /// Read/write access
    #[serde(default)]
    pub access: AccessMode,
    /// Optional RDF predicate IRI this signal maps to
    pub rdf_predicate: Option<String>,
    /// Minimum expected physical value (for out-of-range detection)
    pub min_value: Option<f64>,
    /// Maximum expected physical value
    pub max_value: Option<f64>,
}

fn default_scale_factor() -> f64 {
    1.0
}

impl RegisterEntry {
    /// Create a new entry with required fields and sensible defaults.
    pub fn new(name: impl Into<String>, address: u16, data_type: ModbusDataTypeKind) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            address,
            data_type,
            endianness: Endianness::BigEndian,
            scale_factor: 1.0,
            offset: 0.0,
            unit: None,
            access: AccessMode::ReadOnly,
            rdf_predicate: None,
            min_value: None,
            max_value: None,
        }
    }

    /// Set the human-readable description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Configure byte / word ordering.
    pub fn with_endianness(mut self, e: Endianness) -> Self {
        self.endianness = e;
        self
    }

    /// Apply a linear scale factor and offset.
    ///
    /// `physical = raw * scale_factor + offset`
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

    /// Set access mode.
    pub fn with_access(mut self, access: AccessMode) -> Self {
        self.access = access;
        self
    }

    /// Set RDF predicate IRI.
    pub fn with_rdf_predicate(mut self, pred: impl Into<String>) -> Self {
        self.rdf_predicate = Some(pred.into());
        self
    }

    /// Set expected physical value range for validation.
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    /// Apply scale + offset to a raw numeric reading.
    pub fn apply_scaling(&self, raw: f64) -> f64 {
        raw * self.scale_factor + self.offset
    }

    /// Number of Modbus registers consumed by this entry.
    pub fn register_count(&self) -> usize {
        self.data_type.register_count()
    }

    /// Validate that a physical value is within the declared range.
    ///
    /// Returns `None` when no range bounds are configured.
    pub fn in_range(&self, physical: f64) -> Option<bool> {
        match (self.min_value, self.max_value) {
            (Some(mn), Some(mx)) => Some(physical >= mn && physical <= mx),
            (Some(mn), None) => Some(physical >= mn),
            (None, Some(mx)) => Some(physical <= mx),
            (None, None) => None,
        }
    }
}

// ── DeviceProfile ─────────────────────────────────────────────────────────────

/// Complete register-map profile for one Modbus device type.
///
/// A profile describes *how to interpret* registers from a given device
/// model (e.g. "Siemens PAC3200 energy meter"). It is model-specific, not
/// instance-specific — multiple physical devices of the same model share
/// one profile.
///
/// # Serialisation
///
/// Profiles serialise to / deserialise from both JSON and TOML.
///
/// ## JSON
/// ```json
/// {
///   "model": "ExampleMeter",
///   "vendor": "ACME",
///   "description": "3-phase energy meter",
///   "version": "1.0",
///   "registers": [...]
/// }
/// ```
///
/// ## TOML
/// ```toml
/// model = "ExampleMeter"
/// vendor = "ACME"
/// description = "3-phase energy meter"
/// version = "1.0"
///
/// [[registers]]
/// name = "voltage_l1"
/// address = 30001
/// data_type = "float32"
/// ...
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceProfile {
    /// Device model identifier (e.g. `"PAC3200"`)
    pub model: String,
    /// Manufacturer / vendor name
    #[serde(default)]
    pub vendor: String,
    /// Free-form description
    #[serde(default)]
    pub description: String,
    /// Profile schema version string (e.g. `"1.0"`)
    #[serde(default = "default_version")]
    pub version: String,
    /// Ordered list of register entries
    #[serde(default)]
    pub registers: Vec<RegisterEntry>,
    /// Optional free-form metadata tags
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

fn default_version() -> String {
    "1.0".to_string()
}

impl DeviceProfile {
    /// Create a new, empty profile with a model name.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            vendor: String::new(),
            description: String::new(),
            version: "1.0".to_string(),
            registers: Vec::new(),
            tags: HashMap::new(),
        }
    }

    /// Set vendor name.
    pub fn with_vendor(mut self, vendor: impl Into<String>) -> Self {
        self.vendor = vendor.into();
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set profile version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Append a register entry (builder style).
    pub fn with_register(mut self, entry: RegisterEntry) -> Self {
        self.registers.push(entry);
        self
    }

    /// Append a register entry in-place.
    pub fn add_register(&mut self, entry: RegisterEntry) {
        self.registers.push(entry);
    }

    /// Add a metadata tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Lookup a register entry by name (case-sensitive, O(n)).
    pub fn find_by_name(&self, name: &str) -> Option<&RegisterEntry> {
        self.registers.iter().find(|e| e.name == name)
    }

    /// Collect all entries at the given Modbus address.
    ///
    /// Multiple entries at the same address are valid when different bits
    /// or data types overlay the same register.
    pub fn find_by_address(&self, address: u16) -> Vec<&RegisterEntry> {
        self.registers
            .iter()
            .filter(|e| e.address == address)
            .collect()
    }

    /// Return entries filtered by access mode.
    pub fn entries_with_access(&self, access: AccessMode) -> Vec<&RegisterEntry> {
        self.registers
            .iter()
            .filter(|e| e.access == access)
            .collect()
    }

    /// Total number of register entries in the profile.
    pub fn len(&self) -> usize {
        self.registers.len()
    }

    /// True when the profile has no register entries.
    pub fn is_empty(&self) -> bool {
        self.registers.is_empty()
    }

    /// Serialise the profile to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialise a profile from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialise the profile to a TOML string.
    ///
    /// # Errors
    ///
    /// Returns a [`ModbusError`] when the TOML serialiser fails.
    pub fn to_toml(&self) -> ModbusResult<String> {
        toml::to_string_pretty(self)
            .map_err(|e| ModbusError::Config(format!("TOML serialisation failed: {}", e)))
    }

    /// Deserialise a profile from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns a [`ModbusError`] when the TOML deserialiser fails.
    pub fn from_toml(toml_str: &str) -> ModbusResult<Self> {
        toml::from_str(toml_str)
            .map_err(|e| ModbusError::Config(format!("TOML deserialisation failed: {}", e)))
    }

    /// Validate the profile for internal consistency.
    ///
    /// Checks:
    /// - Model name is non-empty
    /// - All register entries have unique names
    pub fn validate(&self) -> ModbusResult<()> {
        if self.model.trim().is_empty() {
            return Err(ModbusError::Config(
                "DeviceProfile: model name must not be empty".into(),
            ));
        }
        let mut seen_names: HashMap<&str, usize> = HashMap::new();
        for (idx, entry) in self.registers.iter().enumerate() {
            if let Some(prev) = seen_names.insert(entry.name.as_str(), idx) {
                return Err(ModbusError::Config(format!(
                    "DeviceProfile '{}': duplicate register name '{}' at indices {} and {}",
                    self.model, entry.name, prev, idx
                )));
            }
        }
        Ok(())
    }
}

// ── DeviceProfileRegistry ─────────────────────────────────────────────────────

/// Registry for managing multiple [`DeviceProfile`]s by model name.
///
/// Provides bulk import / export via JSON and TOML.
///
/// # Example
///
/// ```rust
/// use oxirs_modbus::device_profile::{DeviceProfile, DeviceProfileRegistry, RegisterEntry};
/// use oxirs_modbus::datatype::ModbusDataTypeKind;
///
/// let mut registry = DeviceProfileRegistry::new();
///
/// let profile = DeviceProfile::new("ExampleMeter")
///     .with_vendor("ACME")
///     .with_register(
///         RegisterEntry::new("voltage", 30001, ModbusDataTypeKind::Float32)
///             .with_unit("V"),
///     );
///
/// registry.register(profile).unwrap();
/// assert_eq!(registry.len(), 1);
/// ```
#[derive(Debug, Default)]
pub struct DeviceProfileRegistry {
    profiles: HashMap<String, DeviceProfile>,
}

impl DeviceProfileRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }

    /// Add a profile.  Returns an error if the model name is already present.
    pub fn register(&mut self, profile: DeviceProfile) -> ModbusResult<()> {
        profile.validate()?;
        if self.profiles.contains_key(&profile.model) {
            return Err(ModbusError::Config(format!(
                "Profile for model '{}' already registered; use upsert() to replace",
                profile.model
            )));
        }
        self.profiles.insert(profile.model.clone(), profile);
        Ok(())
    }

    /// Add or replace a profile unconditionally (after validation).
    pub fn upsert(&mut self, profile: DeviceProfile) -> ModbusResult<()> {
        profile.validate()?;
        self.profiles.insert(profile.model.clone(), profile);
        Ok(())
    }

    /// Remove a profile by model name.  Returns the removed profile if found.
    pub fn remove(&mut self, model: &str) -> Option<DeviceProfile> {
        self.profiles.remove(model)
    }

    /// Look up a profile by model name.
    pub fn get(&self, model: &str) -> Option<&DeviceProfile> {
        self.profiles.get(model)
    }

    /// Mutably access a profile by model name.
    pub fn get_mut(&mut self, model: &str) -> Option<&mut DeviceProfile> {
        self.profiles.get_mut(model)
    }

    /// List all profiles, sorted by model name for deterministic output.
    pub fn list_sorted(&self) -> Vec<&DeviceProfile> {
        let mut v: Vec<&DeviceProfile> = self.profiles.values().collect();
        v.sort_by(|a, b| a.model.cmp(&b.model));
        v
    }

    /// Number of profiles in the registry.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// True when no profiles are registered.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Find profiles that have a specific vendor.
    pub fn find_by_vendor(&self, vendor: &str) -> Vec<&DeviceProfile> {
        self.profiles
            .values()
            .filter(|p| p.vendor.eq_ignore_ascii_case(vendor))
            .collect()
    }

    /// Serialise the entire registry to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.profiles)
    }

    /// Deserialise a registry from JSON (produced by [`DeviceProfileRegistry::to_json`]).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let profiles: HashMap<String, DeviceProfile> = serde_json::from_str(json)?;
        Ok(Self { profiles })
    }

    /// Serialise the registry to TOML.
    pub fn to_toml(&self) -> ModbusResult<String> {
        toml::to_string_pretty(&self.profiles)
            .map_err(|e| ModbusError::Config(format!("Registry TOML serialisation failed: {}", e)))
    }

    /// Deserialise a registry from TOML.
    pub fn from_toml(toml_str: &str) -> ModbusResult<Self> {
        let profiles: HashMap<String, DeviceProfile> = toml::from_str(toml_str).map_err(|e| {
            ModbusError::Config(format!("Registry TOML deserialisation failed: {}", e))
        })?;
        Ok(Self { profiles })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_energy_meter_profile() -> DeviceProfile {
        DeviceProfile::new("PAC3200")
            .with_vendor("Siemens")
            .with_description("3-phase energy meter")
            .with_version("2.1")
            .with_register(
                RegisterEntry::new("voltage_l1", 30001, ModbusDataTypeKind::Float32)
                    .with_description("Line 1 voltage")
                    .with_unit("V")
                    .with_range(0.0, 600.0)
                    .with_access(AccessMode::ReadOnly),
            )
            .with_register(
                RegisterEntry::new("active_power", 30015, ModbusDataTypeKind::Float32)
                    .with_description("Total active power")
                    .with_scaling(1.0, 0.0)
                    .with_unit("kW")
                    .with_access(AccessMode::ReadOnly),
            )
            .with_register(
                RegisterEntry::new("setpoint", 40001, ModbusDataTypeKind::Uint16)
                    .with_description("Power setpoint")
                    .with_unit("%")
                    .with_access(AccessMode::ReadWrite),
            )
    }

    // ── RegisterEntry ─────────────────────────────────────────────────────

    #[test]
    fn test_register_entry_new() {
        let entry = RegisterEntry::new("temp", 30001, ModbusDataTypeKind::Int16);
        assert_eq!(entry.name, "temp");
        assert_eq!(entry.address, 30001);
        assert_eq!(entry.scale_factor, 1.0);
        assert_eq!(entry.offset, 0.0);
        assert_eq!(entry.access, AccessMode::ReadOnly);
    }

    #[test]
    fn test_register_entry_scaling() {
        let entry =
            RegisterEntry::new("temp", 30001, ModbusDataTypeKind::Int16).with_scaling(0.1, -40.0);
        // raw 625 → 62.5 - 40 = 22.5 °C
        let physical = entry.apply_scaling(625.0);
        assert!((physical - 22.5).abs() < 1e-9);
    }

    #[test]
    fn test_register_entry_in_range() {
        let entry = RegisterEntry::new("v", 1, ModbusDataTypeKind::Float32).with_range(0.0, 500.0);
        assert_eq!(entry.in_range(250.0), Some(true));
        assert_eq!(entry.in_range(600.0), Some(false));
    }

    #[test]
    fn test_register_entry_no_range() {
        let entry = RegisterEntry::new("x", 0, ModbusDataTypeKind::Uint16);
        assert_eq!(entry.in_range(999.0), None);
    }

    #[test]
    fn test_register_entry_register_count() {
        let e32 = RegisterEntry::new("f", 0, ModbusDataTypeKind::Float32);
        assert_eq!(e32.register_count(), 2);
        let e64 = RegisterEntry::new("d", 0, ModbusDataTypeKind::Float64);
        assert_eq!(e64.register_count(), 4);
    }

    // ── DeviceProfile ─────────────────────────────────────────────────────

    #[test]
    fn test_profile_find_by_name() {
        let profile = make_energy_meter_profile();
        assert!(profile.find_by_name("voltage_l1").is_some());
        assert!(profile.find_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_profile_find_by_address() {
        let profile = make_energy_meter_profile();
        let entries = profile.find_by_address(30001);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "voltage_l1");
    }

    #[test]
    fn test_profile_entries_with_access() {
        let profile = make_energy_meter_profile();
        let ro = profile.entries_with_access(AccessMode::ReadOnly);
        assert_eq!(ro.len(), 2);
        let rw = profile.entries_with_access(AccessMode::ReadWrite);
        assert_eq!(rw.len(), 1);
    }

    #[test]
    fn test_profile_validate_ok() {
        let profile = make_energy_meter_profile();
        assert!(profile.validate().is_ok());
    }

    #[test]
    fn test_profile_validate_empty_model() {
        let profile = DeviceProfile::new("  ");
        assert!(profile.validate().is_err());
    }

    #[test]
    fn test_profile_validate_duplicate_name() {
        let profile = DeviceProfile::new("Test")
            .with_register(RegisterEntry::new("v", 0, ModbusDataTypeKind::Uint16))
            .with_register(RegisterEntry::new("v", 1, ModbusDataTypeKind::Uint16));
        assert!(profile.validate().is_err());
    }

    // ── JSON serialisation ────────────────────────────────────────────────

    #[test]
    fn test_profile_json_roundtrip() {
        let profile = make_energy_meter_profile();
        let json = profile.to_json().unwrap();
        let restored = DeviceProfile::from_json(&json).unwrap();

        assert_eq!(restored.model, "PAC3200");
        assert_eq!(restored.vendor, "Siemens");
        assert_eq!(restored.registers.len(), 3);
    }

    #[test]
    fn test_profile_json_contains_fields() {
        let profile = make_energy_meter_profile();
        let json = profile.to_json().unwrap();
        assert!(json.contains("PAC3200"));
        assert!(json.contains("voltage_l1"));
        assert!(json.contains("kW"));
    }

    // ── TOML serialisation ────────────────────────────────────────────────

    #[test]
    fn test_profile_toml_roundtrip() {
        let profile = make_energy_meter_profile();
        let toml_str = profile.to_toml().unwrap();
        let restored = DeviceProfile::from_toml(&toml_str).unwrap();

        assert_eq!(restored.model, "PAC3200");
        assert_eq!(restored.registers.len(), 3);
    }

    #[test]
    fn test_profile_toml_invalid() {
        assert!(DeviceProfile::from_toml("this is not valid toml ][").is_err());
    }

    // ── DeviceProfileRegistry ─────────────────────────────────────────────

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = DeviceProfileRegistry::new();
        reg.register(make_energy_meter_profile()).unwrap();
        assert_eq!(reg.len(), 1);
        assert!(reg.get("PAC3200").is_some());
    }

    #[test]
    fn test_registry_duplicate_rejected() {
        let mut reg = DeviceProfileRegistry::new();
        reg.register(make_energy_meter_profile()).unwrap();
        assert!(reg.register(make_energy_meter_profile()).is_err());
    }

    #[test]
    fn test_registry_upsert() {
        let mut reg = DeviceProfileRegistry::new();
        reg.upsert(make_energy_meter_profile()).unwrap();
        reg.upsert(make_energy_meter_profile()).unwrap();
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = DeviceProfileRegistry::new();
        reg.register(make_energy_meter_profile()).unwrap();
        let removed = reg.remove("PAC3200");
        assert!(removed.is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_find_by_vendor() {
        let mut reg = DeviceProfileRegistry::new();
        reg.register(make_energy_meter_profile()).unwrap();
        let profile2 = DeviceProfile::new("Other").with_vendor("Siemens");
        reg.register(profile2).unwrap();
        let profile3 = DeviceProfile::new("ThirdParty").with_vendor("ACME");
        reg.register(profile3).unwrap();

        let siemens = reg.find_by_vendor("siemens");
        assert_eq!(siemens.len(), 2);

        let acme = reg.find_by_vendor("ACME");
        assert_eq!(acme.len(), 1);
    }

    #[test]
    fn test_registry_list_sorted() {
        let mut reg = DeviceProfileRegistry::new();
        reg.register(DeviceProfile::new("ZZZ")).unwrap();
        reg.register(DeviceProfile::new("AAA")).unwrap();
        reg.register(DeviceProfile::new("MMM")).unwrap();

        let sorted = reg.list_sorted();
        assert_eq!(sorted[0].model, "AAA");
        assert_eq!(sorted[1].model, "MMM");
        assert_eq!(sorted[2].model, "ZZZ");
    }

    #[test]
    fn test_registry_json_roundtrip() {
        let mut reg = DeviceProfileRegistry::new();
        reg.register(make_energy_meter_profile()).unwrap();
        reg.register(DeviceProfile::new("Simple")).unwrap();

        let json = reg.to_json().unwrap();
        let restored = DeviceProfileRegistry::from_json(&json).unwrap();

        assert_eq!(restored.len(), 2);
        assert!(restored.get("PAC3200").is_some());
        assert!(restored.get("Simple").is_some());
    }

    // ── AccessMode Display ────────────────────────────────────────────────

    #[test]
    fn test_access_mode_display() {
        assert_eq!(AccessMode::ReadOnly.to_string(), "read_only");
        assert_eq!(AccessMode::ReadWrite.to_string(), "read_write");
        assert_eq!(AccessMode::WriteOnly.to_string(), "write_only");
    }

    // ── temp_dir file persistence ─────────────────────────────────────────

    #[test]
    fn test_profile_json_file_roundtrip() {
        let profile = make_energy_meter_profile();
        let json = profile.to_json().unwrap();

        let tmp = std::env::temp_dir().join("oxirs_profile_test.json");
        std::fs::write(&tmp, &json).unwrap();
        let read_back = std::fs::read_to_string(&tmp).unwrap();
        let restored = DeviceProfile::from_json(&read_back).unwrap();

        std::fs::remove_file(&tmp).ok();

        assert_eq!(restored.model, "PAC3200");
        assert_eq!(restored.registers.len(), 3);
    }

    #[test]
    fn test_profile_toml_file_roundtrip() {
        let profile = make_energy_meter_profile();
        let toml_str = profile.to_toml().unwrap();

        let tmp = std::env::temp_dir().join("oxirs_profile_test.toml");
        std::fs::write(&tmp, &toml_str).unwrap();
        let read_back = std::fs::read_to_string(&tmp).unwrap();
        let restored = DeviceProfile::from_toml(&read_back).unwrap();

        std::fs::remove_file(&tmp).ok();

        assert_eq!(restored.model, "PAC3200");
        assert_eq!(restored.vendor, "Siemens");
    }
}
