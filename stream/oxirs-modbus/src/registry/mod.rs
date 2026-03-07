//! Modbus device registry
//!
//! Provides a runtime catalog of known Modbus devices with their register
//! maps, polling configurations, and connection metadata. This enables
//! multi-device management and dynamic discovery.

pub mod device_registry;

pub use device_registry::{
    DeviceRegistry, DeviceType, ModbusDevice, RegisterDefinition, RegisterMap,
    RegisterMap as DeviceRegisterMap, RegisterType, RegisterType as DeviceRegisterType,
};
