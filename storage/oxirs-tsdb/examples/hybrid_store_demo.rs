//! Hybrid Store Demonstration
//!
//! Shows how the HybridStore automatically routes data between
//! RDF storage and time-series storage based on predicates.

use chrono::Utc;
use oxirs_core::model::{Literal, NamedNode, Triple};
use oxirs_core::rdf_store::Store;
use oxirs_tsdb::HybridStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS Hybrid Store Demo ===\n");

    // Create hybrid store
    let store = HybridStore::new()?;
    println!("✓ Created hybrid store (RDF + time-series backends)\n");

    // 1. Insert metadata (goes to RDF store)
    println!("1. Inserting sensor metadata...");
    let sensor = NamedNode::new("http://example.org/sensors/temp1")?;
    let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;
    let sensor_class = NamedNode::new("http://www.w3.org/ns/sosa/Sensor")?;

    let metadata_triple = Triple::new(sensor.clone(), rdf_type, sensor_class);
    store.insert_triple(metadata_triple)?;
    println!("   ✓ Stored: <sensor> rdf:type sosa:Sensor → RDF store");

    // 2. Insert numeric value (goes to time-series store)
    println!("\n2. Inserting sensor reading...");
    let numeric_value = NamedNode::new("http://qudt.org/schema/qudt/numericValue")?;
    let value = Literal::new("22.5");

    let ts_triple = Triple::new(sensor.clone(), numeric_value, value);
    store.insert_triple(ts_triple)?;
    println!("   ✓ Stored: <sensor> qudt:numericValue 22.5 → Time-series store");

    // 3. Direct time-series insertion (bypasses RDF layer)
    println!("\n3. Direct time-series insertion...");
    let series_id = 1;
    let timestamp = Utc::now();
    let reading = 23.1;

    store.insert_ts(series_id, timestamp, reading)?;
    println!("   ✓ Direct insert: series={series_id}, value={reading}");

    // 4. Query time-series data
    println!("\n4. Querying time-series data...");
    let start = timestamp - chrono::Duration::minutes(5);
    let end = timestamp + chrono::Duration::minutes(5);
    let points = store.query_ts_range(series_id, start, end)?;
    println!("   ✓ Retrieved {} data points", points.len());

    // 5. Query RDF metadata
    println!("\n5. Querying RDF metadata...");
    let sparql = r#"
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX sosa: <http://www.w3.org/ns/sosa/>
        SELECT ?sensor WHERE {
            ?sensor rdf:type sosa:Sensor .
        }
    "#;
    let results = store.query(sparql)?;
    println!("   ✓ SPARQL query returned {} results", results.len());

    println!("\n=== Summary ===");
    println!("RDF store contains: {} quads (metadata)", store.len()?);
    println!("Time-series store contains: {} data points", points.len());
    println!("\n✓ Hybrid storage working correctly!");

    Ok(())
}
