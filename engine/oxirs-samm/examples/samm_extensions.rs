//! Example: SAMM Extension System
//!
//! This example demonstrates how to create and use extensions to the SAMM metamodel.
//! Extensions allow you to add domain-specific concepts while maintaining compatibility
//! with the core SAMM specification.
//!
//! Run this example with:
//! ```bash
//! cargo run --example samm_extensions
//! ```

use oxirs_samm::metamodel::extension::{
    Extension, ExtensionElement, ExtensionRegistry, PropertyDefinition, ValidationRule,
    ValidationSeverity,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SAMM Extension System Example ===\n");

    // Example 1: Automotive Industry Extension
    println!("--- Automotive Industry Extension ---\n");

    let mut automotive_ext = Extension::new(
        "urn:extension:automotive:1.0.0",
        "Automotive Industry Extension",
    );

    automotive_ext.set_description("Extension for automotive industry-specific SAMM elements");
    automotive_ext.set_version("1.0.0");
    automotive_ext.add_author("Automotive Standards Consortium");
    automotive_ext.set_samm_version("2.1.0");

    // Add custom characteristic for VIN (Vehicle Identification Number)
    let vin_characteristic = ExtensionElement::new(
        "VinCharacteristic",
        "Characteristic for Vehicle Identification Numbers",
    )
    .with_type("Characteristic")
    .extends("samm:Characteristic")
    .require("vinPattern")
    .require("checkDigitValidation")
    .optional("manufacturerRegistry")
    .attribute("domain", "automotive")
    .attribute("standard", "ISO 3779");

    automotive_ext.add_element(vin_characteristic);

    // Add custom property for vehicle type
    let vehicle_type_prop = PropertyDefinition::new("vehicleType", "string")
        .with_description("Type of vehicle (passenger, commercial, etc.)")
        .required()
        .allow("PASSENGER")
        .allow("COMMERCIAL")
        .allow("MOTORCYCLE")
        .allow("TRUCK")
        .allow("BUS");

    automotive_ext.add_custom_property("vehicleType".to_string(), vehicle_type_prop);

    // Add validation rule
    let vin_validation = ValidationRule {
        name: "VIN Length Validation".to_string(),
        description: "VIN must be exactly 17 characters".to_string(),
        applies_to: "VinCharacteristic".to_string(),
        severity: ValidationSeverity::Error,
        expression: "length(value) == 17".to_string(),
    };

    automotive_ext.add_validation_rule(vin_validation);

    println!("Created automotive extension:");
    println!("  Name: {}", automotive_ext.name);
    println!("  Namespace: {}", automotive_ext.namespace);
    println!("  Version: {:?}", automotive_ext.version);
    println!("  Elements: {}", automotive_ext.elements.len());
    println!(
        "  Custom Properties: {}",
        automotive_ext.custom_properties.len()
    );
    println!(
        "  Validation Rules: {}\n",
        automotive_ext.validation_rules.len()
    );

    // Example 2: IoT (Internet of Things) Extension
    println!("--- IoT Extension ---\n");

    let mut iot_ext = Extension::new("urn:extension:iot:1.0.0", "IoT Extension");

    iot_ext.set_description("Extension for IoT sensor and device modeling");
    iot_ext.add_author("IoT Consortium");

    // Add sensor characteristic
    let sensor_characteristic =
        ExtensionElement::new("SensorCharacteristic", "Characteristic for sensor readings")
            .with_type("Characteristic")
            .extends("samm:Measurement")
            .require("unit")
            .require("accuracy")
            .require("samplingRate")
            .optional("calibrationDate")
            .attribute("domain", "iot")
            .attribute("category", "sensor");

    iot_ext.add_element(sensor_characteristic);

    // Add telemetry property
    let telemetry_prop = PropertyDefinition::new("telemetryType", "string")
        .with_description("Type of telemetry data")
        .required()
        .allow("TEMPERATURE")
        .allow("HUMIDITY")
        .allow("PRESSURE")
        .allow("ACCELERATION")
        .allow("LOCATION");

    iot_ext.add_custom_property("telemetryType".to_string(), telemetry_prop);

    println!("Created IoT extension:");
    println!("  Name: {}", iot_ext.name);
    println!("  Elements: {}", iot_ext.elements.len());
    println!("  Custom Properties: {}\n", iot_ext.custom_properties.len());

    // Example 3: Financial Services Extension
    println!("--- Financial Services Extension ---\n");

    let mut financial_ext = Extension::new(
        "urn:extension:finance:1.0.0",
        "Financial Services Extension",
    );

    financial_ext.set_description("Extension for financial data modeling");

    // Add currency characteristic
    let currency_characteristic = ExtensionElement::new(
        "CurrencyCharacteristic",
        "Characteristic for currency amounts",
    )
    .with_type("Characteristic")
    .extends("samm:Measurement")
    .require("currencyCode")
    .require("precision")
    .optional("conversionRate")
    .attribute("domain", "finance")
    .attribute("standard", "ISO 4217");

    financial_ext.add_element(currency_characteristic);

    // Add account type property
    let account_type_prop = PropertyDefinition::new("accountType", "string")
        .with_description("Type of financial account")
        .required()
        .allow("CHECKING")
        .allow("SAVINGS")
        .allow("INVESTMENT")
        .allow("CREDIT");

    financial_ext.add_custom_property("accountType".to_string(), account_type_prop);

    println!("Created financial services extension:");
    println!("  Name: {}", financial_ext.name);
    println!("  Elements: {}\n", financial_ext.elements.len());

    // Example 4: Using the Extension Registry
    println!("--- Extension Registry ---\n");

    let registry = ExtensionRegistry::new();

    // Register all extensions
    registry.register(automotive_ext.clone());
    registry.register(iot_ext.clone());
    registry.register(financial_ext.clone());

    println!("Registry contains {} extensions", registry.count());
    println!("\nRegistered extensions:");
    for namespace in registry.list() {
        if let Some(ext) = registry.get(&namespace) {
            println!("  - {} ({})", ext.name, namespace);
        }
    }

    // Query extensions
    println!("\nQuerying extensions:");

    if let Some(ext) = registry.get("urn:extension:automotive:1.0.0") {
        println!("  Found automotive extension: {}", ext.name);
        if ext.has_element("VinCharacteristic") {
            println!("    - Has VinCharacteristic element");
        }
    }

    // Find extensions by SAMM version
    let v21_extensions = registry.find_by_samm_version("2.1.0");
    println!("\n  Extensions for SAMM 2.1.0: {}", v21_extensions.len());

    // Example 5: Extension Element Details
    println!("\n--- Extension Element Details ---\n");

    if let Some(ext) = registry.get("urn:extension:automotive:1.0.0") {
        if let Some(element) = ext.get_element("VinCharacteristic") {
            println!("VIN Characteristic:");
            println!("  Name: {}", element.name);
            println!("  Type: {}", element.element_type);
            println!("  Extends: {:?}", element.extends);
            println!("  Required Properties: {:?}", element.required_properties);
            println!("  Optional Properties: {:?}", element.optional_properties);
            println!("  Attributes: {:?}", element.attributes);
        }
    }

    // Example 6: Custom Properties
    println!("\n--- Custom Property Definitions ---\n");

    if let Some(ext) = registry.get("urn:extension:automotive:1.0.0") {
        if let Some(prop) = ext.custom_properties.get("vehicleType") {
            println!("Vehicle Type Property:");
            println!("  Name: {}", prop.name);
            println!("  Data Type: {}", prop.data_type);
            println!("  Required: {}", prop.required);
            println!("  Allowed Values: {:?}", prop.allowed_values);
        }
    }

    // Example 7: Validation Rules
    println!("\n--- Validation Rules ---\n");

    if let Some(ext) = registry.get("urn:extension:automotive:1.0.0") {
        for rule in &ext.validation_rules {
            println!("Rule: {}", rule.name);
            println!("  Description: {}", rule.description);
            println!("  Applies To: {}", rule.applies_to);
            println!("  Severity: {:?}", rule.severity);
            println!("  Expression: {}\n", rule.expression);
        }
    }

    // Example 8: Serialization (save extension to JSON)
    println!("--- Extension Serialization ---\n");

    if let Some(ext) = registry.get("urn:extension:iot:1.0.0") {
        let json = serde_json::to_string_pretty(&ext)?;
        println!("IoT Extension as JSON:");
        println!("{}\n", json);
    }

    println!("=== Extension System Demonstration Complete ===");

    Ok(())
}
