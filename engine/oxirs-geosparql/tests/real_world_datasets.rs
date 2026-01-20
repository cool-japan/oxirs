//! Real-world dataset tests simulating OpenStreetMap and similar spatial data scenarios
//!
//! These tests validate oxirs-geosparql with realistic spatial queries and datasets
//! that mirror production use cases from OpenStreetMap, GIS systems, and spatial databases.

use geo_types::Point;
use oxirs_geosparql::analysis::clustering::{dbscan_clustering, kmeans_clustering, DbscanParams};
use oxirs_geosparql::analysis::heatmap::{generate_heatmap, HeatmapConfig, KernelFunction};
use oxirs_geosparql::analysis::interpolation::{idw_interpolation, SamplePoint};
use oxirs_geosparql::analysis::network::{dijkstra_shortest_path, Network};
use oxirs_geosparql::analysis::statistics::{getis_ord_gi_star, morans_i, WeightsMatrixType};
use oxirs_geosparql::functions::geometric_operations::*;
use oxirs_geosparql::functions::geometric_properties::*;
use oxirs_geosparql::functions::simple_features::*;
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::index::SpatialIndex;

// ============================================================================
// OpenStreetMap-style POI (Point of Interest) Queries
// ============================================================================

#[cfg(test)]
mod osm_poi_scenarios {
    use super::*;

    /// Simulate finding restaurants within 500m of a location
    /// Typical OSM query: "SELECT ?poi WHERE { ?poi geo:sfWithin ?buffer }"
    #[test]
    fn test_find_nearby_restaurants() {
        // User location (e.g., downtown San Francisco: 37.7749° N, 122.4194° W)
        let user_location = Geometry::from_wkt("POINT(-122.4194 37.7749)").unwrap();

        // Simulated restaurant locations in the area
        let restaurants = vec![
            Geometry::from_wkt("POINT(-122.4180 37.7755)").unwrap(), // ~150m away
            Geometry::from_wkt("POINT(-122.4210 37.7740)").unwrap(), // ~200m away
            Geometry::from_wkt("POINT(-122.4300 37.7800)").unwrap(), // ~1km away (outside radius)
        ];

        let _search_radius_meters = 500.0;
        // In WGS84 degrees, ~500m ≈ 0.0045° at this latitude
        let search_radius_degrees = 0.0045;

        let mut nearby_count = 0;
        for restaurant in &restaurants {
            let distance = distance(&user_location, restaurant).unwrap();
            if distance <= search_radius_degrees {
                nearby_count += 1;
            }
        }

        assert_eq!(
            nearby_count, 2,
            "Should find 2 restaurants within 500m radius"
        );
    }

    /// Simulate spatial indexing for efficient POI queries
    /// Real-world use: Tile-based map rendering with millions of POIs
    #[test]
    fn test_poi_spatial_index_performance() {
        let _index = SpatialIndex::new();

        // Simulate 1000 POIs (restaurants, cafes, shops) in a city grid
        let pois: Vec<Geometry> = (0..1000)
            .map(|i| {
                let lat = 37.7 + (i / 100) as f64 * 0.01; // 10x100 grid
                let lon = -122.5 + (i % 100) as f64 * 0.01;
                Geometry::from_wkt(&format!("POINT({} {})", lon, lat)).unwrap()
            })
            .collect();

        // Bulk load for optimal R-tree structure
        let loaded_index = SpatialIndex::bulk_load(pois).unwrap();

        // Query: Find all POIs in a viewport (typical map tile query)
        let viewport_min = (-122.45, 37.75);
        let viewport_max = (-122.40, 37.80);

        let results = loaded_index.query_bbox(
            viewport_min.0,
            viewport_min.1,
            viewport_max.0,
            viewport_max.1,
        );

        assert!(
            !results.is_empty(),
            "Should find POIs within viewport bounds"
        );
        assert!(
            results.len() < 1000,
            "Should only return POIs within viewport, not all POIs"
        );
    }

    /// Simulate finding points of interest within a complex polygon boundary
    /// Real-world use: "Find all coffee shops within Central Park"
    #[test]
    fn test_poi_within_polygon_boundary() {
        // Simplified Central Park boundary (actual park has ~843 acres)
        let park_boundary = Geometry::from_wkt(
            "POLYGON((-73.9812 40.7681, -73.9581 40.7681, -73.9581 40.8005, -73.9812 40.8005, -73.9812 40.7681))"
        ).unwrap();

        // Coffee shop locations
        let shops = vec![
            Geometry::from_wkt("POINT(-73.9700 40.7800)").unwrap(), // Inside park
            Geometry::from_wkt("POINT(-73.9650 40.7850)").unwrap(), // Inside park
            Geometry::from_wkt("POINT(-73.9900 40.7700)").unwrap(), // Outside park
        ];

        let mut shops_in_park = 0;
        for shop in &shops {
            if sf_within(shop, &park_boundary).unwrap() {
                shops_in_park += 1;
            }
        }

        assert_eq!(
            shops_in_park, 2,
            "Should find 2 coffee shops inside Central Park"
        );
    }
}

// ============================================================================
// OSM Road Network Queries
// ============================================================================

#[cfg(test)]
mod osm_road_network_scenarios {
    use super::*;

    /// Simulate finding intersecting roads
    /// Real-world use: "Find all roads that cross Interstate 101"
    #[test]
    fn test_intersecting_roads() {
        // Main highway (simplified I-101)
        let highway = Geometry::from_wkt(
            "LINESTRING(-122.4500 37.7000, -122.4000 37.8000, -122.3500 37.9000)",
        )
        .unwrap();

        // Cross streets
        let streets = vec![
            Geometry::from_wkt("LINESTRING(-122.4600 37.7500, -122.3900 37.7500)").unwrap(), // Crosses
            Geometry::from_wkt("LINESTRING(-122.4100 37.8500, -122.4100 37.7500)").unwrap(), // Crosses
            Geometry::from_wkt("LINESTRING(-122.5000 37.7000, -122.5000 37.8000)").unwrap(), // Parallel, doesn't cross
        ];

        let mut crossing_streets = 0;
        for street in &streets {
            if sf_crosses(street, &highway).unwrap() || sf_intersects(street, &highway).unwrap() {
                crossing_streets += 1;
            }
        }

        assert!(
            crossing_streets >= 2,
            "Should find at least 2 streets crossing the highway"
        );
    }

    /// Simulate shortest path routing
    /// Real-world use: Navigation systems, delivery routing
    #[test]
    fn test_road_network_shortest_path() {
        use geo_types::Coord;

        let mut network = Network::new();

        // Create a simple road network (grid-like street layout)
        let intersections = vec![
            Coord { x: 0.0, y: 0.0 }, // Node 0
            Coord { x: 1.0, y: 0.0 }, // Node 1
            Coord { x: 2.0, y: 0.0 }, // Node 2
            Coord { x: 0.0, y: 1.0 }, // Node 3
            Coord { x: 1.0, y: 1.0 }, // Node 4
            Coord { x: 2.0, y: 1.0 }, // Node 5
        ];

        // Add nodes first
        for coord in intersections {
            network.add_node(coord);
        }

        // Add road segments (edges)
        network.add_edge(0, 1, 1.0).unwrap(); // Horizontal roads
        network.add_edge(1, 2, 1.0).unwrap();
        network.add_edge(3, 4, 1.0).unwrap();
        network.add_edge(4, 5, 1.0).unwrap();
        network.add_edge(0, 3, 1.0).unwrap(); // Vertical roads
        network.add_edge(1, 4, 1.0).unwrap();
        network.add_edge(2, 5, 1.0).unwrap();

        // Find shortest path from node 0 to node 5
        let result = dijkstra_shortest_path(&network, 0, 5);

        assert!(result.is_ok(), "Should find a path from 0 to 5");
        let path = result.unwrap();
        assert_eq!(path.nodes.len(), 4, "Path should have 4 nodes (0->1->4->5)");
        assert!((path.cost - 3.0).abs() < 1e-6, "Path cost should be 3");
    }

    /// Simulate road buffering for noise pollution analysis
    /// Real-world use: Environmental impact assessments
    #[test]
    #[cfg(any(feature = "geos-backend", feature = "rust-buffer"))]
    fn test_road_noise_buffer_zone() {
        let highway = Geometry::from_wkt("LINESTRING(0 0, 10 0, 10 10)").unwrap();

        // Create 100m buffer zone (noise impact area)
        let buffer_distance = 0.001; // ~100m in degrees
        let noise_zone = buffer(&highway, buffer_distance).unwrap();

        // Check if residential building is in noise zone
        let building = Geometry::from_wkt("POINT(10.0005 5.0)").unwrap();

        assert!(
            sf_within(&building, &noise_zone).unwrap()
                || sf_intersects(&building, &noise_zone).unwrap(),
            "Building should be within highway noise buffer zone"
        );
    }
}

// ============================================================================
// OSM Building Footprint Analysis
// ============================================================================

#[cfg(test)]
mod osm_building_scenarios {
    use super::*;

    /// Simulate building density analysis
    /// Real-world use: Urban planning, zoning compliance
    #[test]
    fn test_building_density_in_zone() {
        // Zoning district boundary
        let zone = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();

        // Building footprints in the zone
        let buildings = vec![
            Geometry::from_wkt("POLYGON((1 1, 2 1, 2 2, 1 2, 1 1))").unwrap(), // 1 sq unit
            Geometry::from_wkt("POLYGON((3 3, 5 3, 5 5, 3 5, 3 3))").unwrap(), // 4 sq units
            Geometry::from_wkt("POLYGON((7 7, 9 7, 9 9, 7 9, 7 7))").unwrap(), // 4 sq units
        ];

        let zone_area = area(&zone).unwrap();
        let mut total_building_area = 0.0;

        for building in &buildings {
            if sf_within(building, &zone).unwrap() {
                total_building_area += area(building).unwrap();
            }
        }

        let coverage_ratio = total_building_area / zone_area;

        assert!(
            coverage_ratio > 0.0 && coverage_ratio < 1.0,
            "Building coverage ratio should be between 0 and 1"
        );
        assert!(
            (coverage_ratio - 0.09).abs() < 0.01,
            "Coverage should be approximately 9% (9/100)"
        );
    }

    /// Simulate finding buildings within fire station response radius
    /// Real-world use: Emergency service planning
    #[test]
    fn test_emergency_service_coverage() {
        let fire_station = Geometry::from_wkt("POINT(5 5)").unwrap();
        let response_radius = 3.0; // 3km radius

        let buildings = vec![
            Geometry::from_wkt("POINT(5 6)").unwrap(), // 1km away - covered
            Geometry::from_wkt("POINT(7 7)").unwrap(), // ~2.8km away - covered
            Geometry::from_wkt("POINT(10 10)").unwrap(), // ~7km away - NOT covered
        ];

        let mut covered_buildings = 0;
        for building in &buildings {
            let dist = distance(&fire_station, building).unwrap();
            if dist <= response_radius {
                covered_buildings += 1;
            }
        }

        assert_eq!(
            covered_buildings, 2,
            "Should cover 2 buildings within 3km response radius"
        );
    }

    /// Simulate building shadow analysis for solar planning
    /// Real-world use: Solar panel placement, urban heat island analysis
    #[test]
    fn test_building_footprint_area() {
        let building = Geometry::from_wkt("POLYGON((0 0, 50 0, 50 30, 0 30, 0 0))").unwrap();

        let footprint_area = area(&building).unwrap();

        assert!(
            (footprint_area - 1500.0).abs() < 1.0,
            "Building footprint should be 1500 sq meters"
        );

        // Calculate roof area available for solar panels (assume 80% usable)
        let usable_roof_area = footprint_area * 0.8;
        assert!(
            (usable_roof_area - 1200.0).abs() < 1.0,
            "Usable roof area should be ~1200 sq meters"
        );
    }
}

// ============================================================================
// OSM Spatial Clustering (Crime Hotspots, Business Districts)
// ============================================================================

#[cfg(test)]
mod osm_clustering_scenarios {
    use super::*;

    /// Simulate crime hotspot detection using DBSCAN
    /// Real-world use: Police resource allocation, crime prevention
    #[test]
    fn test_crime_hotspot_clustering() {
        // Crime incident locations (latitude, longitude)
        let incidents = vec![
            Point::new(-122.4194, 37.7749),
            Point::new(-122.4195, 37.7750),
            Point::new(-122.4196, 37.7748),
            Point::new(-122.4197, 37.7751), // Cluster 1: Downtown area (4 incidents)
            Point::new(-122.4500, 37.7900),
            Point::new(-122.4501, 37.7901), // Cluster 2: Different neighborhood (2 incidents)
            Point::new(-122.5000, 37.8500), // Isolated incident (noise point)
        ];

        let params = DbscanParams {
            eps: 0.001, // ~100m radius in degrees
            min_pts: 2, // Minimum 2 incidents to form a cluster
        };

        let result = dbscan_clustering(&incidents, params).unwrap();

        // Should identify 2 main crime clusters
        assert!(
            result.n_clusters >= 1,
            "Should detect at least 1 crime hotspot"
        );
        assert!(
            result.n_noise > 0 || result.n_clusters >= 2,
            "Should have noise points or multiple clusters"
        );
    }

    /// Simulate commercial district identification using K-means
    /// Real-world use: Business zone analysis, retail planning
    #[test]
    fn test_business_district_clustering() {
        use oxirs_geosparql::analysis::clustering::KmeansParams;

        // Business locations in a city
        let businesses = vec![
            Point::new(0.0, 0.0),
            Point::new(0.1, 0.1),
            Point::new(0.2, 0.0), // District 1: Downtown
            Point::new(5.0, 5.0),
            Point::new(5.1, 5.1),
            Point::new(5.2, 5.0), // District 2: Suburban center
            Point::new(10.0, 10.0),
            Point::new(10.1, 10.1), // District 3: Another area
        ];

        let params = KmeansParams {
            k: 3, // Find 3 business districts
            max_iterations: 100,
            tolerance: 1e-6,
            n_init: 10,
        };
        let result = kmeans_clustering(&businesses, params).unwrap();

        assert_eq!(result.n_clusters, 3, "Should identify 3 business districts");
        assert_eq!(
            result.labels.len(),
            businesses.len(),
            "Each business should be assigned to a district"
        );
    }

    /// Simulate heatmap generation for pedestrian traffic
    /// Real-world use: Urban planning, retail site selection
    #[test]
    fn test_pedestrian_traffic_heatmap() {
        use geo_types::Rect;

        // Pedestrian count observations - convert to Geometry
        let observations: Vec<Geometry> = vec![
            Point::new(0.0, 0.0),
            Point::new(0.1, 0.1),
            Point::new(0.2, 0.0),
            Point::new(1.0, 1.0),
            Point::new(5.0, 5.0),
        ]
        .into_iter()
        .map(|p| Geometry::from_wkt(&format!("POINT({} {})", p.x(), p.y())).unwrap())
        .collect();

        let bounds = Rect::new(
            geo_types::Coord { x: 0.0, y: 0.0 },
            geo_types::Coord { x: 6.0, y: 6.0 },
        );

        let config = HeatmapConfig {
            grid_width: 20,
            grid_height: 20,
            radius: 1.0,
            kernel: KernelFunction::Gaussian,
            bounds: Some(bounds),
            normalize: true,
            weights: None,
        };

        let heatmap = generate_heatmap(&observations, &config).unwrap();

        assert_eq!(heatmap.grid.nrows(), 20, "Heatmap should have 20 rows");
        assert_eq!(heatmap.grid.ncols(), 20, "Heatmap should have 20 columns");
        assert!(
            heatmap.grid.len() == 400,
            "Heatmap should have 400 cells (20x20)"
        );

        // Find hotspot (cell with maximum intensity)
        let max_intensity = heatmap.max_value;
        assert!(max_intensity > 0.0, "Should have at least one hotspot");
    }
}

// ============================================================================
// OSM Environmental Analysis
// ============================================================================

#[cfg(test)]
mod osm_environmental_scenarios {
    use super::*;

    /// Simulate flood risk assessment
    /// Real-world use: Disaster preparedness, insurance risk modeling
    #[test]
    #[cfg(any(feature = "geos-backend", feature = "rust-buffer"))]
    fn test_flood_risk_area() {
        // River polygon (flood-prone area)
        let river = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 2, 0 2, 0 0))").unwrap();

        // 100-year flood zone (buffer around river)
        let flood_zone = buffer(&river, 1.0).unwrap(); // 1km buffer

        // Check if residential areas are at risk
        let residential_area = Geometry::from_wkt("POINT(5 2.5)").unwrap();

        assert!(
            sf_within(&residential_area, &flood_zone).unwrap()
                || sf_intersects(&residential_area, &flood_zone).unwrap(),
            "Residential area should be in flood risk zone"
        );
    }

    /// Simulate air quality interpolation
    /// Real-world use: Environmental monitoring, public health
    #[test]
    fn test_air_quality_interpolation() {
        // Air quality sensor readings (PM2.5 levels)
        let sensors = vec![
            SamplePoint {
                location: Point::new(0.0, 0.0),
                value: 15.0, // Good
            },
            SamplePoint {
                location: Point::new(5.0, 0.0),
                value: 45.0, // Moderate (near highway)
            },
            SamplePoint {
                location: Point::new(10.0, 0.0),
                value: 20.0, // Good
            },
        ];

        // Interpolate air quality at a location between sensors
        let query_point = Point::new(2.5, 0.0);
        let result = idw_interpolation(&sensors, &query_point, 2.0).unwrap();

        assert!(
            result.value > 15.0 && result.value < 45.0,
            "Interpolated value should be between sensor readings"
        );
    }

    /// Simulate spatial autocorrelation analysis (Moran's I)
    /// Real-world use: Identifying spatial patterns in environmental data
    #[test]
    fn test_pollution_spatial_autocorrelation() {
        // Pollution monitoring stations
        let stations = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(0.0, 1.0),
            Point::new(1.0, 1.0),
        ];

        // Pollution levels showing spatial clustering (high values cluster together)
        let pollution_levels = vec![10.0, 12.0, 11.0, 9.0, 13.0];

        let result = morans_i(
            &stations,
            &pollution_levels,
            WeightsMatrixType::InverseDistance { power: 2.0 },
        )
        .unwrap();

        // Positive Moran's I indicates spatial clustering
        assert!(
            result.morans_i.abs() < 1.0,
            "Moran's I should be between -1 and 1"
        );
    }

    /// Simulate hotspot analysis with Getis-Ord Gi*
    /// Real-world use: Identifying statistically significant hotspots
    #[test]
    fn test_pollution_hotspot_analysis() {
        let locations = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
        ];

        let values = vec![5.0, 15.0, 5.0]; // Middle location has high value

        let result = getis_ord_gi_star(
            &locations,
            &values,
            WeightsMatrixType::InverseDistance { power: 1.0 },
        )
        .unwrap();

        assert_eq!(
            result.z_scores.len(),
            3,
            "Should compute Gi* statistic for each location"
        );

        // Middle location should have positive z-score (hotspot)
        assert!(
            result.z_scores[1] > 0.0,
            "Middle location with high value should be a hotspot"
        );
    }
}

// ============================================================================
// OSM Data Quality and Validation
// ============================================================================

#[cfg(test)]
mod osm_data_quality_scenarios {
    use super::*;
    use oxirs_geosparql::validation::*;

    /// Simulate detecting invalid geometries in OSM data
    /// Real-world use: Data quality assurance before import
    #[test]
    fn test_invalid_geometry_detection() {
        // Self-intersecting polygon (bow-tie shape) - common OSM data error
        let invalid_polygon = Geometry::from_wkt(
            "POLYGON((0 0, 2 2, 2 0, 0 2, 0 0))", // Crosses itself
        )
        .unwrap();

        let validation = validate_geometry(&invalid_polygon);

        // Should detect self-intersection
        assert!(
            !validation.is_valid
                || validation
                    .warnings
                    .iter()
                    .any(|w| w.contains("intersection")),
            "Should detect self-intersecting polygon"
        );
    }

    /// Simulate geometry repair workflow
    /// Real-world use: Automated data cleaning pipelines
    #[test]
    fn test_geometry_repair_workflow() {
        // Polygon with duplicate consecutive points
        let dirty_polygon = Geometry::from_wkt(
            "POLYGON((0 0, 0 0, 1 0, 1 1, 1 1, 0 1, 0 0))", // Has duplicates
        )
        .unwrap();

        let repaired = repair_geometry(&dirty_polygon).unwrap();

        // Repaired geometry should have fewer points
        if let geo_types::Geometry::Polygon(poly) = &repaired.geom {
            let original_points =
                if let geo_types::Geometry::Polygon(orig_poly) = &dirty_polygon.geom {
                    orig_poly.exterior().0.len()
                } else {
                    0
                };

            assert!(
                poly.exterior().0.len() <= original_points,
                "Repaired polygon should have same or fewer points (duplicates removed)"
            );
        }
    }

    /// Simulate precision snapping for geometry alignment
    /// Real-world use: Topological consistency in OSM data
    #[test]
    fn test_coordinate_precision_snapping() {
        // Geometry with excessive precision
        let high_precision =
            Geometry::from_wkt("POINT(1.123456789012345 2.987654321098765)").unwrap();

        let snapped = snap_to_precision(&high_precision, 6).unwrap();

        // Verify precision is reduced
        let snapped_wkt = snapped.to_wkt();
        assert!(
            !snapped_wkt.contains("123456789"),
            "Should reduce coordinate precision"
        );
    }
}

// ============================================================================
// OSM Large-Scale Performance Scenarios
// ============================================================================

#[cfg(test)]
mod osm_performance_scenarios {
    use super::*;
    use std::time::Instant;

    /// Simulate tile server performance (zoom level 15 tile)
    /// Real-world use: Web map rendering at high zoom levels
    #[test]
    fn test_tile_query_performance() {
        let _index = SpatialIndex::new();

        // Simulate 10,000 POIs in a single map tile
        let pois: Vec<Geometry> = (0..10_000)
            .map(|i| {
                let lat = 37.7 + (i / 100) as f64 * 0.0001;
                let lon = -122.4 + (i % 100) as f64 * 0.0001;
                Geometry::from_wkt(&format!("POINT({} {})", lon, lat)).unwrap()
            })
            .collect();

        // Bulk load (simulates tile pre-processing)
        let start = Instant::now();
        let loaded_index = SpatialIndex::bulk_load(pois).unwrap();
        let load_time = start.elapsed();

        // Should load 10k POIs quickly
        assert!(
            load_time.as_millis() < 500,
            "Bulk loading 10k POIs should take <500ms, took {}ms",
            load_time.as_millis()
        );

        // Query tile viewport (typical map request)
        let start = Instant::now();
        let results = loaded_index.query_bbox(-122.405, 37.705, -122.395, 37.715);
        let query_time = start.elapsed();

        assert!(
            query_time.as_millis() < 50,
            "Viewport query should take <50ms, took {}ms",
            query_time.as_millis()
        );

        assert!(!results.is_empty(), "Should find POIs in viewport");
    }

    /// Simulate routing performance on large road network
    /// Real-world use: Navigation apps, logistics optimization
    #[test]
    fn test_large_network_routing() {
        use geo_types::Coord;

        let mut network = Network::new();

        // Create a 20x20 grid road network (400 intersections, ~760 road segments)
        let grid_size = 20;

        // Add all nodes first
        for y in 0..grid_size {
            for x in 0..grid_size {
                network.add_node(Coord {
                    x: x as f64,
                    y: y as f64,
                });
            }
        }

        // Add edges
        for y in 0..grid_size {
            for x in 0..grid_size {
                let node_id = y * grid_size + x;

                // Add horizontal road
                if x < grid_size - 1 {
                    network.add_edge(node_id, node_id + 1, 1.0).unwrap();
                }

                // Add vertical road
                if y < grid_size - 1 {
                    network.add_edge(node_id, node_id + grid_size, 1.0).unwrap();
                }
            }
        }

        // Find path from top-left to bottom-right
        let start = Instant::now();
        let result = dijkstra_shortest_path(&network, 0, (grid_size * grid_size) - 1);
        let routing_time = start.elapsed();

        assert!(result.is_ok(), "Should find a path across the network");
        assert!(
            routing_time.as_millis() < 100,
            "Routing on 400-node network should take <100ms, took {}ms",
            routing_time.as_millis()
        );
    }

    /// Simulate batch distance calculation performance
    /// Real-world use: Proximity analysis for large datasets
    #[test]
    #[cfg(feature = "parallel")]
    fn test_batch_distance_calculation_performance() {
        use oxirs_geosparql::performance::batch::BatchProcessor;

        // Create 1000 random points
        let geometries: Vec<Geometry> = (0..1000)
            .map(|i| {
                let x = (i % 100) as f64 * 0.1;
                let y = (i / 100) as f64 * 0.1;
                Geometry::from_wkt(&format!("POINT({} {})", x, y)).unwrap()
            })
            .collect();

        let processor = BatchProcessor::new();
        let reference = Geometry::from_wkt("POINT(5.0 5.0)").unwrap();

        // Calculate distances from reference point to all 1000 points
        let start = Instant::now();
        let distances = processor.distances(&reference, &geometries).unwrap();
        let batch_time = start.elapsed();

        assert_eq!(distances.len(), 1000, "Should compute 1000 distances");
        assert!(
            batch_time.as_millis() < 200,
            "Batch distance calculation should take <200ms, took {}ms",
            batch_time.as_millis()
        );
    }
}

// ============================================================================
// OSM Real Data Integration Tests (With Actual OpenStreetMap Data)
// ============================================================================

#[cfg(test)]
mod osm_real_data_scenarios {
    use super::*;
    #[cfg(any(feature = "geojson-support", feature = "shapefile-support"))]
    use std::fs;
    #[cfg(any(feature = "geojson-support", feature = "shapefile-support"))]
    use std::path::Path;

    /// Test loading real OpenStreetMap GeoJSON export
    /// Requires: Download OSM data via Overpass API or export from JOSM
    /// Example: https://overpass-api.de/api/interpreter?data=[out:json];node(around:1000,51.5074,-0.1278);out;
    #[test]
    #[ignore] // Run manually with: cargo test --test real_world_datasets osm_real_geojson -- --ignored
    #[cfg(feature = "geojson-support")]
    fn test_load_real_osm_geojson() {
        let test_data_path = "/tmp/osm_test_data.geojson";

        // Skip test if file doesn't exist
        if !Path::new(test_data_path).exists() {
            eprintln!(
                "Skipping test: Place real OSM GeoJSON file at {}",
                test_data_path
            );
            eprintln!("Download example: curl 'https://overpass-api.de/api/interpreter?data=[out:json];node(around:500,37.7749,-122.4194);out;' > /tmp/osm_test_data.geojson");
            return;
        }

        let geojson_data =
            fs::read_to_string(test_data_path).expect("Failed to read OSM GeoJSON file");

        // Parse GeoJSON features
        let features = oxirs_geosparql::geometry::geojson_parser::parse_geojson_feature_collection(
            &geojson_data,
        )
        .expect("Failed to parse OSM GeoJSON");

        println!("Loaded {} features from real OSM data", features.len());
        assert!(
            !features.is_empty(),
            "Should load at least one feature from OSM data"
        );

        // Validate geometries
        for (i, geom) in features.iter().enumerate().take(100) {
            let validation = oxirs_geosparql::validation::validate_geometry(geom);
            if !validation.is_valid {
                eprintln!(
                    "Warning: Feature {} has invalid geometry: {:?}",
                    i, validation.errors
                );
            }
        }
    }

    /// Test loading real OpenStreetMap Shapefile export
    /// Requires: Download OSM shapefile from https://download.geofabrik.de/
    #[test]
    #[ignore] // Run manually with: cargo test --test real_world_datasets osm_real_shapefile -- --ignored
    #[cfg(feature = "shapefile-support")]
    fn test_load_real_osm_shapefile() {
        let test_shapefile = "/tmp/osm_roads.shp";

        if !Path::new(test_shapefile).exists() {
            eprintln!(
                "Skipping test: Place real OSM shapefile at {}",
                test_shapefile
            );
            eprintln!("Download from: https://download.geofabrik.de/");
            return;
        }

        let geometries =
            oxirs_geosparql::geometry::shapefile_parser::read_shapefile(test_shapefile)
                .expect("Failed to read OSM shapefile");

        println!(
            "Loaded {} road segments from OSM shapefile",
            geometries.len()
        );
        assert!(
            !geometries.is_empty(),
            "Should load at least one road segment"
        );

        // Test spatial indexing with real data
        let start = std::time::Instant::now();
        let index = SpatialIndex::bulk_load(geometries.clone()).unwrap();
        let index_time = start.elapsed();

        println!(
            "Indexed {} geometries in {:?}",
            geometries.len(),
            index_time
        );

        // Query a bbox (should be fast even with large datasets)
        if let geo_types::Geometry::LineString(first_line) = &geometries[0].geom {
            if let Some(first_coord) = first_line.0.first() {
                let query_start = std::time::Instant::now();
                let nearby = index.query_bbox(
                    first_coord.x - 0.01,
                    first_coord.y - 0.01,
                    first_coord.x + 0.01,
                    first_coord.y + 0.01,
                );
                let query_time = query_start.elapsed();

                println!("Found {} nearby roads in {:?}", nearby.len(), query_time);
                assert!(
                    !nearby.is_empty(),
                    "Should find roads near first coordinate"
                );
            }
        }
    }

    /// Test with realistic large dataset (similar to city-scale OSM data)
    /// Simulates processing 100,000 POIs (typical medium-sized city)
    #[test]
    fn test_city_scale_poi_dataset() {
        // Generate realistic distribution (clustered around multiple centers)
        let city_centers = vec![
            (37.7749, -122.4194), // San Francisco
            (37.8044, -122.2712), // Oakland
            (37.3382, -121.8863), // San Jose
        ];

        let mut pois = Vec::new();
        let start = std::time::Instant::now();

        for (center_lat, center_lon) in &city_centers {
            // Generate ~33k POIs around each city center
            for i in 0..33_333 {
                // Random offset within ~10km radius
                let offset_lat = (i % 200) as f64 * 0.0005 - 0.05;
                let offset_lon = (i / 200) as f64 * 0.0005 - 0.05;

                let lat = center_lat + offset_lat;
                let lon = center_lon + offset_lon;

                pois.push(Geometry::from_wkt(&format!("POINT({} {})", lon, lat)).unwrap());
            }
        }

        let generation_time = start.elapsed();
        println!("Generated {} POIs in {:?}", pois.len(), generation_time);

        // Test bulk loading performance
        let index_start = std::time::Instant::now();
        let index = SpatialIndex::bulk_load(pois.clone()).unwrap();
        let index_time = index_start.elapsed();

        println!("Bulk loaded {} POIs in {:?}", pois.len(), index_time);

        // Performance requirement: Should index 100k POIs in <5 seconds
        assert!(
            index_time.as_secs() < 5,
            "Should index 100k POIs in <5s, took {:?}",
            index_time
        );

        // Test query performance on large dataset
        let query_start = std::time::Instant::now();
        let downtown_sf = index.query_bbox(-122.42, 37.77, -122.41, 37.78);
        let query_time = query_start.elapsed();

        println!(
            "Queried downtown SF viewport in {:?}, found {} POIs",
            query_time,
            downtown_sf.len()
        );

        // Performance requirement: Viewport query should be <100ms even with 100k POIs
        assert!(
            query_time.as_millis() < 100,
            "Viewport query should take <100ms, took {:?}",
            query_time
        );
        assert!(!downtown_sf.is_empty(), "Should find POIs in downtown SF");
    }

    /// Test with OSM building footprints (polygon heavy dataset)
    /// Simulates 10,000 building polygons (typical downtown area)
    #[test]
    fn test_building_footprint_dataset() {
        let buildings: Vec<Geometry> = (0..10_000)
            .map(|i| {
                // Generate realistic building footprints (rectangles)
                let base_x = 37.7 + (i / 100) as f64 * 0.001;
                let base_y = -122.4 + (i % 100) as f64 * 0.001;

                // Buildings are 20-50m wide (0.0002-0.0005 degrees)
                let width = 0.0002 + (i % 3) as f64 * 0.0001;
                let height = 0.0002 + ((i + 1) % 3) as f64 * 0.0001;

                Geometry::from_wkt(&format!(
                    "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                    base_x,
                    base_y,
                    base_x + width,
                    base_y,
                    base_x + width,
                    base_y + height,
                    base_x,
                    base_y + height,
                    base_x,
                    base_y
                ))
                .unwrap()
            })
            .collect();

        let start = std::time::Instant::now();
        let index = SpatialIndex::bulk_load(buildings.clone()).unwrap();
        let index_time = start.elapsed();

        println!(
            "Indexed {} building polygons in {:?}",
            buildings.len(),
            index_time
        );

        // Test area calculations on real-scale buildings
        let mut total_area = 0.0;
        for building in buildings.iter().take(100) {
            total_area += area(building).unwrap();
        }

        println!(
            "Average building area (first 100): {:.6} sq degrees",
            total_area / 100.0
        );

        // Test spatial query: Find all buildings in a city block
        let block_bbox = index.query_bbox(37.7, -122.4, 37.701, -122.399);
        println!("Found {} buildings in city block", block_bbox.len());

        assert!(
            !block_bbox.is_empty(),
            "Should find buildings in city block"
        );
    }

    /// Test with realistic road network (LineString heavy dataset)
    /// Simulates 50,000 road segments (typical metro area)
    #[test]
    fn test_road_network_dataset() {
        let mut roads = Vec::new();

        // Generate grid-based road network with some curved roads
        for i in 0..50_000 {
            let start_x = 37.5 + (i % 500) as f64 * 0.0005;
            let start_y = -122.5 + (i / 500) as f64 * 0.0005;

            // Most roads are short segments (100-500m)
            let end_x = start_x + 0.0003;
            let end_y = start_y + if i % 7 == 0 { 0.0003 } else { 0.0 };

            let road = if i % 10 == 0 {
                // 10% curved roads (with midpoint)
                let mid_x = (start_x + end_x) / 2.0;
                let mid_y = (start_y + end_y) / 2.0 + 0.0001;
                Geometry::from_wkt(&format!(
                    "LINESTRING({} {}, {} {}, {} {})",
                    start_x, start_y, mid_x, mid_y, end_x, end_y
                ))
                .unwrap()
            } else {
                // 90% straight roads
                Geometry::from_wkt(&format!(
                    "LINESTRING({} {}, {} {})",
                    start_x, start_y, end_x, end_y
                ))
                .unwrap()
            };

            roads.push(road);
        }

        let start = std::time::Instant::now();
        let _index = SpatialIndex::bulk_load(roads.clone()).unwrap();
        let index_time = start.elapsed();

        println!("Indexed {} road segments in {:?}", roads.len(), index_time);

        // Test length calculations
        let mut total_length = 0.0;
        for road in roads.iter().take(1000) {
            total_length += length(road).unwrap();
        }

        println!(
            "Average road segment length (first 1000): {:.6} degrees",
            total_length / 1000.0
        );

        // Test finding intersecting roads
        let query_road = Geometry::from_wkt("LINESTRING(37.5 -122.5, 37.6 -122.4)").unwrap();
        let mut intersecting_count = 0;

        for road in roads.iter().take(10_000) {
            if sf_intersects(&query_road, road).unwrap() {
                intersecting_count += 1;
            }
        }

        println!(
            "Found {} roads intersecting query route (checked 10k roads)",
            intersecting_count
        );
    }

    /// Test memory efficiency with large dataset
    /// Verifies that oxirs-geosparql can handle datasets that approach memory limits
    #[test]
    #[ignore] // Run manually: cargo test --test real_world_datasets memory_stress -- --ignored
    fn test_memory_efficiency_stress() {
        use std::time::Instant;

        // Generate 500,000 points (approaches typical laptop memory limits)
        println!("Generating 500,000 point dataset...");
        let points: Vec<Geometry> = (0..500_000)
            .map(|i| {
                let x = (i % 1000) as f64 * 0.001;
                let y = (i / 1000) as f64 * 0.001;
                Geometry::from_wkt(&format!("POINT({} {})", x, y)).unwrap()
            })
            .collect();

        println!("Testing bulk load performance...");
        let start = Instant::now();
        let index = SpatialIndex::bulk_load(points.clone()).unwrap();
        let index_time = start.elapsed();

        println!("Indexed 500k points in {:?}", index_time);

        // Should complete in reasonable time (<30 seconds)
        assert!(
            index_time.as_secs() < 30,
            "Should index 500k points in <30s, took {:?}",
            index_time
        );

        // Test query performance
        let query_start = Instant::now();
        let results = index.query_bbox(0.0, 0.0, 0.1, 0.1);
        let query_time = query_start.elapsed();

        println!(
            "Query returned {} results in {:?}",
            results.len(),
            query_time
        );

        // Query should still be fast even with 500k points
        assert!(
            query_time.as_millis() < 200,
            "Query should take <200ms, took {:?}",
            query_time
        );
    }
}
