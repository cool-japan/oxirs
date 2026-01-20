//! Hybrid RDF + Time-Series Storage Example
//!
//! Demonstrates the concept of combining RDF graph storage with
//! time-series optimization for IoT data:
//!
//! - Semantic metadata in RDF (sensor descriptions, locations)
//! - High-frequency measurements in time-series store
//! - Unified query interface concept
//!
//! # Usage
//!
//! ```bash
//! cargo run --example hybrid_storage
//! ```

use chrono::{Duration, Utc};
use oxirs_tsdb::{
    Aggregation, DataPoint, QueryEngine, SeriesDescriptor, SeriesMetadata, TimeChunk,
};
use std::collections::HashMap;

/// Simulated RDF triple for sensor metadata
#[derive(Debug, Clone)]
struct RdfTriple {
    subject: String,
    predicate: String,
    object: String,
}

/// Hybrid store combining RDF metadata with time-series data
struct HybridStore {
    /// RDF triples for sensor metadata
    rdf_triples: Vec<RdfTriple>,
    /// Series descriptors mapping IRI to series ID
    series_registry: HashMap<String, SeriesDescriptor>,
    /// Series metadata
    series_metadata: HashMap<u64, SeriesMetadata>,
    /// Time-series query engine
    tsdb: QueryEngine,
}

impl HybridStore {
    fn new() -> Self {
        Self {
            rdf_triples: Vec::new(),
            series_registry: HashMap::new(),
            series_metadata: HashMap::new(),
            tsdb: QueryEngine::new(),
        }
    }

    /// Register a sensor with RDF metadata
    fn register_sensor(
        &mut self,
        sensor_iri: &str,
        predicate: &str,
        label: &str,
        location: &str,
        unit: &str,
        sampling_rate: f64,
    ) -> u64 {
        let series_id = self.series_registry.len() as u64 + 1;

        // Create series descriptor
        let descriptor =
            SeriesDescriptor::new(series_id, sensor_iri.to_string(), predicate.to_string());
        self.series_registry
            .insert(sensor_iri.to_string(), descriptor);

        // Create series metadata
        let metadata = SeriesMetadata {
            series_id,
            unit: Some(unit.to_string()),
            sampling_rate: Some(sampling_rate),
            data_type: Some("xsd:double".to_string()),
            description: Some(label.to_string()),
        };
        self.series_metadata.insert(series_id, metadata);

        // Store RDF metadata
        self.rdf_triples.push(RdfTriple {
            subject: sensor_iri.to_string(),
            predicate: "rdf:type".to_string(),
            object: "sosa:Sensor".to_string(),
        });
        self.rdf_triples.push(RdfTriple {
            subject: sensor_iri.to_string(),
            predicate: "rdfs:label".to_string(),
            object: format!("\"{}\"", label),
        });
        self.rdf_triples.push(RdfTriple {
            subject: sensor_iri.to_string(),
            predicate: "sosa:isHostedBy".to_string(),
            object: location.to_string(),
        });
        self.rdf_triples.push(RdfTriple {
            subject: sensor_iri.to_string(),
            predicate: "sosa:observes".to_string(),
            object: predicate.to_string(),
        });
        self.rdf_triples.push(RdfTriple {
            subject: sensor_iri.to_string(),
            predicate: "qudt:unit".to_string(),
            object: format!("unit:{}", unit),
        });

        series_id
    }

    /// Load time-series data for a sensor
    fn load_data(&mut self, series_id: u64, points: Vec<DataPoint>) {
        if points.is_empty() {
            return;
        }
        let start_time = points[0].timestamp;
        if let Ok(chunk) = TimeChunk::new(series_id, start_time, Duration::hours(2), points) {
            self.tsdb.add_chunk(chunk);
        }
    }

    /// Query sensor by IRI pattern (SPARQL-like)
    fn find_sensors_by_location(&self, location: &str) -> Vec<&str> {
        self.rdf_triples
            .iter()
            .filter(|t| t.predicate == "sosa:isHostedBy" && t.object == location)
            .map(|t| t.subject.as_str())
            .collect()
    }

    /// Get series ID for sensor IRI
    fn get_series_id(&self, sensor_iri: &str) -> Option<u64> {
        self.series_registry.get(sensor_iri).map(|d| d.series_id)
    }

    /// Get sensor label
    fn get_sensor_label(&self, sensor_iri: &str) -> Option<&str> {
        self.rdf_triples
            .iter()
            .find(|t| t.subject == sensor_iri && t.predicate == "rdfs:label")
            .map(|t| t.object.trim_matches('"'))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS Hybrid RDF + Time-Series Storage Demo ===\n");

    // Create hybrid store
    let store = create_demo_store()?;

    // Demo 1: Query sensors by location using RDF
    demo_rdf_query(&store);

    // Demo 2: Query time-series data for sensors
    demo_timeseries_query(&store)?;

    // Demo 3: Unified query concept
    demo_unified_query(&store)?;

    // Demo 4: Storage efficiency analysis
    demo_storage_efficiency(&store);

    println!("\n=== Hybrid Storage Demo Complete ===");
    Ok(())
}

fn create_demo_store() -> Result<HybridStore, Box<dyn std::error::Error>> {
    let mut store = HybridStore::new();
    let base_time = Utc::now() - Duration::hours(1);

    println!("Setting up hybrid store...\n");

    // Register sensors with RDF metadata
    // Building A sensors
    let temp1_id = store.register_sensor(
        "http://example.org/sensor/temp-1",
        "http://example.org/property/temperature",
        "Temperature Sensor 1 (Server Room)",
        "http://example.org/location/building-a/server-room",
        "DegreeCelsius",
        1.0,
    );

    let humidity1_id = store.register_sensor(
        "http://example.org/sensor/humidity-1",
        "http://example.org/property/humidity",
        "Humidity Sensor 1 (Server Room)",
        "http://example.org/location/building-a/server-room",
        "Percent",
        1.0,
    );

    let power1_id = store.register_sensor(
        "http://example.org/sensor/power-1",
        "http://example.org/property/power",
        "Power Meter 1 (Server Room)",
        "http://example.org/location/building-a/server-room",
        "Watt",
        10.0,
    );

    // Building B sensors
    let temp2_id = store.register_sensor(
        "http://example.org/sensor/temp-2",
        "http://example.org/property/temperature",
        "Temperature Sensor 2 (Office)",
        "http://example.org/location/building-b/office",
        "DegreeCelsius",
        0.1,
    );

    println!(
        "Registered {} sensors with RDF metadata",
        store.series_registry.len()
    );

    // Generate time-series data
    // Server room temperature (higher, more stable)
    let temp1_data: Vec<DataPoint> = (0..3600)
        .map(|i| DataPoint {
            timestamp: base_time + Duration::seconds(i),
            value: 22.0 + (i % 30) as f64 * 0.05,
        })
        .collect();
    store.load_data(temp1_id, temp1_data);

    // Server room humidity
    let humidity1_data: Vec<DataPoint> = (0..3600)
        .map(|i| DataPoint {
            timestamp: base_time + Duration::seconds(i),
            value: 45.0 + (i % 20) as f64 * 0.3,
        })
        .collect();
    store.load_data(humidity1_id, humidity1_data);

    // Server room power (variable with spikes)
    let power1_data: Vec<DataPoint> = (0..36000) // 10Hz for 1 hour
        .map(|i| {
            let base = 5000.0;
            let spike = if (i / 100) % 60 < 5 { 2000.0 } else { 0.0 };
            let noise = (i % 17) as f64 * 10.0;
            DataPoint {
                timestamp: base_time + Duration::milliseconds(i as i64 * 100),
                value: base + spike + noise,
            }
        })
        .collect();
    store.load_data(power1_id, power1_data);

    // Office temperature (more variable)
    let temp2_data: Vec<DataPoint> = (0..360) // 0.1Hz for 1 hour
        .map(|i| DataPoint {
            timestamp: base_time + Duration::seconds(i * 10),
            value: 21.0 + (i % 50) as f64 * 0.1,
        })
        .collect();
    store.load_data(temp2_id, temp2_data);

    println!("Loaded time-series data for all sensors\n");

    Ok(store)
}

fn demo_rdf_query(store: &HybridStore) {
    println!("Demo 1: RDF Metadata Query");
    println!("--------------------------");
    println!("Query: Find all sensors in server room");
    println!();

    let location = "http://example.org/location/building-a/server-room";
    let sensors = store.find_sensors_by_location(location);

    println!(
        "Location: {}",
        location.split('/').next_back().unwrap_or(location)
    );
    println!("Found {} sensors:", sensors.len());

    for sensor_iri in &sensors {
        let label = store.get_sensor_label(sensor_iri).unwrap_or("Unknown");
        let series_id = store.get_series_id(sensor_iri).unwrap_or(0);
        println!("  - {} (Series ID: {})", label, series_id);
    }

    println!();
    println!("RDF Triples for first sensor:");
    let first_sensor = sensors.first().unwrap_or(&"");
    for triple in &store.rdf_triples {
        if &triple.subject == first_sensor {
            println!(
                "  <{}> <{}> {}",
                triple
                    .subject
                    .split('/')
                    .next_back()
                    .unwrap_or(&triple.subject),
                triple.predicate,
                triple.object
            );
        }
    }
    println!();
}

fn demo_timeseries_query(store: &HybridStore) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 2: Time-Series Data Query");
    println!("------------------------------");

    // Query each sensor
    for (sensor_iri, descriptor) in &store.series_registry {
        let label = store.get_sensor_label(sensor_iri).unwrap_or("Unknown");
        let metadata = store.series_metadata.get(&descriptor.series_id);

        let result = store
            .tsdb
            .query()
            .series(descriptor.series_id)
            .last(Duration::minutes(30))
            .execute()?;

        let avg = store
            .tsdb
            .query()
            .series(descriptor.series_id)
            .aggregate(Aggregation::Avg)
            .execute()?;

        let min = store
            .tsdb
            .query()
            .series(descriptor.series_id)
            .aggregate(Aggregation::Min)
            .execute()?;

        let max = store
            .tsdb
            .query()
            .series(descriptor.series_id)
            .aggregate(Aggregation::Max)
            .execute()?;

        let unit = metadata.and_then(|m| m.unit.as_deref()).unwrap_or("unit");

        println!("{}:", label);
        println!("  Data points (last 30 min): {}", result.points.len());
        println!(
            "  Average: {:.2} {}",
            avg.aggregated_value.unwrap_or(0.0),
            unit
        );
        println!(
            "  Range: {:.2} - {:.2} {}",
            min.aggregated_value.unwrap_or(0.0),
            max.aggregated_value.unwrap_or(0.0),
            unit
        );
        println!();
    }

    Ok(())
}

fn demo_unified_query(store: &HybridStore) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 3: Unified Query Concept");
    println!("-----------------------------");
    println!("Concept: SPARQL query with time-series extensions");
    println!();
    println!("Hypothetical Query:");
    println!("  PREFIX ts: <http://oxirs.org/ts#>");
    println!("  PREFIX sosa: <http://www.w3.org/ns/sosa/>");
    println!();
    println!("  SELECT ?sensor ?label (ts:avg(?temp) AS ?avg_temp)");
    println!("  WHERE {{");
    println!("    ?sensor a sosa:Sensor ;");
    println!("            rdfs:label ?label ;");
    println!("            sosa:observes :temperature .");
    println!("    FILTER(ts:last(?sensor, \"1h\"))");
    println!("  }}");
    println!();

    // Simulate the unified query
    println!("Simulated Execution:");
    println!("  1. RDF query: Find temperature sensors");
    println!("  2. For each sensor: Query time-series for last 1 hour");
    println!("  3. Apply ts:avg() aggregation");
    println!();

    // Find temperature sensors
    let temp_sensors: Vec<_> = store
        .rdf_triples
        .iter()
        .filter(|t| t.predicate == "sosa:observes" && t.object.contains("temperature"))
        .map(|t| t.subject.as_str())
        .collect();

    println!("Results:");
    println!("  {:>40} | {:>15}", "Sensor", "Avg Temperature");
    println!("  {:-<40}-+-{:-<15}", "", "");

    for sensor_iri in temp_sensors {
        if let Some(series_id) = store.get_series_id(sensor_iri) {
            let result = store
                .tsdb
                .query()
                .series(series_id)
                .last(Duration::hours(1))
                .aggregate(Aggregation::Avg)
                .execute()?;

            let label = store.get_sensor_label(sensor_iri).unwrap_or("Unknown");
            println!(
                "  {:>40} | {:>12.2} degC",
                label,
                result.aggregated_value.unwrap_or(0.0)
            );
        }
    }
    println!();

    Ok(())
}

fn demo_storage_efficiency(store: &HybridStore) {
    println!("Demo 4: Storage Efficiency Analysis");
    println!("------------------------------------");

    println!("RDF Metadata Storage:");
    println!("  Triples: {}", store.rdf_triples.len());
    let rdf_size: usize = store
        .rdf_triples
        .iter()
        .map(|t| t.subject.len() + t.predicate.len() + t.object.len())
        .sum();
    println!("  Estimated size: {} bytes", rdf_size);
    println!();

    println!("Time-Series Data Storage:");
    let total_points: usize = store
        .tsdb
        .get_chunks_for_series(1)
        .iter()
        .chain(store.tsdb.get_chunks_for_series(2).iter())
        .chain(store.tsdb.get_chunks_for_series(3).iter())
        .chain(store.tsdb.get_chunks_for_series(4).iter())
        .map(|c| c.metadata.count)
        .sum();

    let compressed_size: usize = store
        .tsdb
        .get_chunks_for_series(1)
        .iter()
        .chain(store.tsdb.get_chunks_for_series(2).iter())
        .chain(store.tsdb.get_chunks_for_series(3).iter())
        .chain(store.tsdb.get_chunks_for_series(4).iter())
        .map(|c| c.metadata.compressed_size)
        .sum();

    let uncompressed_size = total_points * 16; // 8 bytes timestamp + 8 bytes value

    println!("  Total data points: {}", total_points);
    println!("  Uncompressed size: {} bytes", uncompressed_size);
    println!("  Compressed size: {} bytes", compressed_size);
    if compressed_size > 0 {
        println!(
            "  Compression ratio: {:.1}:1",
            uncompressed_size as f64 / compressed_size as f64
        );
    }
    println!();

    println!("Comparison with Pure RDF Storage:");
    let rdf_triple_size = 150; // Estimated average RDF triple size
    let pure_rdf_size = total_points * rdf_triple_size;
    println!("  Pure RDF (1 triple per reading): {} bytes", pure_rdf_size);
    println!(
        "  Hybrid (metadata + compressed TS): {} bytes",
        rdf_size + compressed_size
    );
    if (rdf_size + compressed_size) > 0 {
        println!(
            "  Space savings: {:.1}x",
            pure_rdf_size as f64 / (rdf_size + compressed_size) as f64
        );
    }
    println!();

    println!("Benefits of Hybrid Approach:");
    println!("  + Semantic queries on metadata (SPARQL)");
    println!("  + Efficient time-series compression");
    println!("  + Fast range queries and aggregations");
    println!("  + Unified query interface");
    println!("  + Scales to IoT workloads (1M+ points/sec)");
    println!();
}
