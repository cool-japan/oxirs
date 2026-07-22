//! Complete Digital Twin Example: Smart Factory Battery Production
//!
//! This example demonstrates a digital twin implementation built directly on
//! oxirs-fuseki's NGSI-LD (ETSI GS CIM 009 V1.6.1) entity types:
//! - Simulated sensor data ingestion (temperature/voltage/current per cell)
//! - NGSI-LD entity construction (`NgsiEntity`/`NgsiProperty`/`GeoProperty`)
//! - A simple physics-based thermal projection (no external physics engine)
//! - Threshold-based anomaly detection
//!
//! ## Architecture
//!
//! ```text
//! [Factory Sensors (simulated)] ──▶ [NGSI-LD Entity Construction] ──▶ [Digital Twin State]
//!                                                                            │
//!                                                                            ▼
//!                                                              [Thermal Projection + Anomaly Detection]
//! ```
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example digital_twin_factory
//! ```
//!
//! This example uses only the crate's default feature set: the `ngsi_ld`
//! handler module (types used here) is compiled unconditionally, not behind
//! a Cargo feature.

use oxirs_fuseki::handlers::ngsi_ld::types::{
    GeoProperty, NgsiAttribute, NgsiEntity, NgsiProperty,
};
use tokio::time::Duration;

/// Battery cell sensor data
#[derive(Debug, Clone)]
struct BatteryCellData {
    cell_id: String,
    temperature: f64, // Celsius
    voltage: f64,     // Volts
    current: f64,     // Amperes
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Factory digital twin state
#[derive(Default)]
struct FactoryDigitalTwin {
    cells: Vec<BatteryCellData>,
}

impl FactoryDigitalTwin {
    /// Create new digital twin
    fn new() -> Self {
        Self::default()
    }

    /// Simulate sensor data ingestion
    async fn ingest_sensor_data(
        &mut self,
        data: BatteryCellData,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!(
            "📥 Ingesting sensor data: Cell {}, T={}°C, V={}V",
            data.cell_id, data.temperature, data.voltage
        );

        // Create NGSI-LD entity
        let entity = self.create_battery_entity(&data)?;

        // Store in RDF (via NGSI-LD API)
        self.create_ngsi_entity(entity).await?;

        // Update local state
        self.cells.push(data);

        Ok(())
    }

    /// Create NGSI-LD entity from sensor data
    fn create_battery_entity(
        &self,
        data: &BatteryCellData,
    ) -> Result<NgsiEntity, Box<dyn std::error::Error>> {
        let mut entity = NgsiEntity::new(
            format!("urn:ngsi-ld:BatteryCell:{}", data.cell_id),
            "BatteryCell",
        );

        entity = entity
            .with_property(
                "temperature",
                NgsiProperty::with_observed_at(data.temperature, data.timestamp).with_unit("CEL"),
            )
            .with_property(
                "voltage",
                NgsiProperty::with_observed_at(data.voltage, data.timestamp).with_unit("VLT"),
            )
            .with_property(
                "current",
                NgsiProperty::with_observed_at(data.current, data.timestamp).with_unit("AMP"),
            );

        // Location (factory floor position, Tokyo coordinates as a stand-in)
        entity.properties.insert(
            "location".to_string(),
            NgsiAttribute::GeoProperty(GeoProperty::point(139.6917, 35.6895)),
        );

        Ok(entity)
    }

    /// Create NGSI-LD entity via API
    async fn create_ngsi_entity(
        &self,
        entity: NgsiEntity,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // In production, this would POST the entity to the running server's
        // NGSI-LD API (`/ngsi-ld/v1/entities`). This example runs standalone
        // (no live server), so it only demonstrates entity construction.
        println!("  ✓ Created NGSI-LD entity: {}", entity.id);
        Ok(())
    }

    /// Run thermal simulation for overheating detection
    async fn run_thermal_simulation(
        &self,
        cell_id: &str,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // Find cell data
        let cell = self
            .cells
            .iter()
            .find(|c| c.cell_id == cell_id)
            .ok_or("Cell not found")?;

        // Simple thermal model: T_future = T_current + (I²R * dt) / (m * c)
        // Where:
        //   I = current (A)
        //   R = internal resistance (0.1 Ω)
        //   dt = time step (60s)
        //   m = cell mass (0.05 kg)
        //   c = specific heat (900 J/kg·K)

        let internal_resistance = 0.1; // Ω
        let mass = 0.05; // kg
        let specific_heat = 900.0; // J/kg·K
        let time_step = 60.0; // seconds

        let heat_generated = cell.current.powi(2) * internal_resistance * time_step;
        let temp_increase = heat_generated / (mass * specific_heat);
        let predicted_temp = cell.temperature + temp_increase;

        println!(
            "🔥 Thermal simulation for Cell {}: T_now={}°C → T_future={:.2}°C",
            cell_id, cell.temperature, predicted_temp
        );

        Ok(predicted_temp)
    }

    /// Run threshold-based anomaly detection over the current cell readings
    async fn detect_anomalies(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        println!("🔍 Running anomaly detection...");

        let mut anomalies = Vec::new();

        // Check for overheating (>40°C)
        for cell in &self.cells {
            if cell.temperature > 40.0 {
                let msg = format!(
                    "⚠️  ALERT: Cell {} overheating: {:.1}°C",
                    cell.cell_id, cell.temperature
                );
                println!("{}", msg);
                anomalies.push(msg);
            }

            // Check for overvoltage (>4.2V for Li-ion)
            if cell.voltage > 4.2 {
                let msg = format!(
                    "⚠️  ALERT: Cell {} overvoltage: {:.2}V",
                    cell.cell_id, cell.voltage
                );
                println!("{}", msg);
                anomalies.push(msg);
            }

            // Check for overcurrent (>3A)
            if cell.current > 3.0 {
                let msg = format!(
                    "⚠️  ALERT: Cell {} overcurrent: {:.2}A",
                    cell.cell_id, cell.current
                );
                println!("{}", msg);
                anomalies.push(msg);
            }
        }

        if anomalies.is_empty() {
            println!("  ✓ No anomalies detected");
        }

        Ok(anomalies)
    }

    /// Generate factory report
    fn generate_report(&self) {
        println!("\n📊 Factory Digital Twin Report");
        println!("═══════════════════════════════════════");
        println!("Total cells monitored: {}", self.cells.len());

        if !self.cells.is_empty() {
            let avg_temp: f64 =
                self.cells.iter().map(|c| c.temperature).sum::<f64>() / self.cells.len() as f64;
            let max_temp = self
                .cells
                .iter()
                .map(|c| c.temperature)
                .fold(f64::NEG_INFINITY, f64::max);
            let min_temp = self
                .cells
                .iter()
                .map(|c| c.temperature)
                .fold(f64::INFINITY, f64::min);

            println!("Temperature:");
            println!("  Average: {:.2}°C", avg_temp);
            println!("  Range: {:.2}°C - {:.2}°C", min_temp, max_temp);

            let avg_voltage: f64 =
                self.cells.iter().map(|c| c.voltage).sum::<f64>() / self.cells.len() as f64;
            println!("Voltage:");
            println!("  Average: {:.3}V", avg_voltage);
        }
        println!("═══════════════════════════════════════\n");
    }
}

/// Simulate factory sensor data
fn generate_sensor_data(cell_id: &str, iteration: u64) -> BatteryCellData {
    // Simulate realistic battery behavior
    let base_temp = 25.0;
    let temp_variation = 5.0 * ((iteration as f64 * 0.1).sin());
    let temperature = base_temp + temp_variation;

    let base_voltage = 3.7;
    let voltage_variation = 0.3 * ((iteration as f64 * 0.15).sin());
    let voltage = base_voltage + voltage_variation;

    let base_current = 1.5;
    let current_variation = 0.5 * ((iteration as f64 * 0.2).cos());
    let current = base_current + current_variation;

    BatteryCellData {
        cell_id: cell_id.to_string(),
        temperature,
        voltage,
        current,
        timestamp: chrono::Utc::now(),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏭 OxiRS Digital Twin Factory Example");
    println!("=====================================\n");

    println!("🔧 Initializing OxiRS Digital Twin...");
    let mut twin = FactoryDigitalTwin::new();
    println!("  ✓ Digital twin initialized\n");

    // Simulate production line with 3 battery cells
    let cell_ids = ["CELL-001", "CELL-002", "CELL-003"];

    println!("🚀 Starting production line simulation...\n");

    // Run for 10 iterations (simulating 10 minutes of production)
    for iteration in 0..10 {
        println!(
            "⏱️  Iteration {} (t={}s)",
            iteration + 1,
            (iteration + 1) * 60
        );
        println!("─────────────────────────────────────");

        // Ingest data from all cells
        for cell_id in &cell_ids {
            let data = generate_sensor_data(cell_id, iteration);
            twin.ingest_sensor_data(data).await?;
        }

        // Run anomaly detection
        let _anomalies = twin.detect_anomalies().await?;

        // Run thermal simulation for first cell
        if iteration % 3 == 0 {
            let predicted_temp = twin.run_thermal_simulation("CELL-001").await?;
            if predicted_temp > 45.0 {
                println!("🚨 CRITICAL: Predicted temperature exceeds safe limits!");
            }
        }

        println!();

        // Wait between iterations (simulating real-time)
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Generate final report
    twin.generate_report();

    println!("✅ Simulation complete!");
    println!("\n💡 Next steps:");
    println!("  1. Query RDF data: curl -X POST http://localhost:3030/sparql");
    println!("  2. View NGSI-LD entities: curl http://localhost:3030/ngsi-ld/v1/entities");
    println!("  3. Deploy to production with TLS/mTLS");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_data_generation() {
        let data = generate_sensor_data("TEST-001", 0);
        assert_eq!(data.cell_id, "TEST-001");
        assert!(data.temperature > 0.0 && data.temperature < 100.0);
        assert!(data.voltage > 0.0 && data.voltage < 5.0);
        assert!(data.current >= 0.0);
    }

    #[tokio::test]
    async fn test_digital_twin_creation() {
        let twin = FactoryDigitalTwin::new();
        assert_eq!(twin.cells.len(), 0);
    }

    #[test]
    fn test_battery_entity_creation() {
        let twin = FactoryDigitalTwin::new();

        let data = BatteryCellData {
            cell_id: "TEST-001".to_string(),
            temperature: 25.0,
            voltage: 3.7,
            current: 1.5,
            timestamp: chrono::Utc::now(),
        };

        let entity = twin.create_battery_entity(&data).unwrap();
        assert_eq!(entity.id, "urn:ngsi-ld:BatteryCell:TEST-001");
        assert!(entity.properties.contains_key("temperature"));
        assert!(entity.properties.contains_key("voltage"));
        assert!(entity.properties.contains_key("current"));
        assert!(entity.properties.contains_key("location"));
    }
}
