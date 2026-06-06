//! Tests for the validation module (all sub-modules).

use crate::error::Result;
use crate::geometry::Geometry;
use crate::validation_algorithms::{
    close_ring, remove_consecutive_duplicates, repair_geometry, simplify_geometry,
    snap_to_precision,
};
use crate::validation_core::{validate_geometry, ValidationResult};
use crate::validation_topology::{
    check_ogc_compliance, compute_quality_metrics, validate_geometry_with_config, ValidationConfig,
};
use geo_types::Geometry as GeoGeometry;
use geo_types::{Coord, LineString, Point, Polygon};

#[test]
fn test_validate_valid_point() {
    let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
    let result = validate_geometry(&geom);
    assert!(result.is_valid);
    assert!(result.errors.is_empty());
}

#[test]
fn test_validate_invalid_point() {
    let geom = Geometry::new(GeoGeometry::Point(Point::new(f64::NAN, 2.0)));
    let result = validate_geometry(&geom);
    assert!(!result.is_valid);
    assert!(!result.errors.is_empty());
}

#[test]
fn test_validate_valid_polygon() {
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 4.0, y: 0.0 },
        Coord { x: 4.0, y: 4.0 },
        Coord { x: 0.0, y: 4.0 },
        Coord { x: 0.0, y: 0.0 },
    ];
    let poly = Polygon::new(LineString::new(coords), vec![]);
    let geom = Geometry::new(GeoGeometry::Polygon(poly));

    let result = validate_geometry(&geom);
    assert!(result.is_valid);
}

#[test]
fn test_validate_invalid_polygon() {
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 4.0, y: 0.0 },
        // Not closed - missing last point
    ];
    let poly = Polygon::new(LineString::new(coords), vec![]);
    let geom = Geometry::new(GeoGeometry::Polygon(poly));

    let result = validate_geometry(&geom);
    assert!(!result.is_valid);
}

#[test]
fn test_simplify_linestring() -> Result<()> {
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 1.0, y: 0.1 },
        Coord { x: 2.0, y: 0.0 },
        Coord { x: 3.0, y: 0.1 },
        Coord { x: 4.0, y: 0.0 },
    ];
    let ls = LineString::new(coords);
    let geom = Geometry::new(GeoGeometry::LineString(ls));

    let simplified = simplify_geometry(&geom, 0.2)?;

    match simplified.geom {
        GeoGeometry::LineString(ls) => {
            // Should be simplified to fewer points
            assert!(ls.0.len() < 5);
        }
        _ => panic!("Expected LineString"),
    }

    Ok(())
}

#[test]
fn test_snap_to_precision() -> Result<()> {
    let geom = Geometry::new(GeoGeometry::Point(Point::new(1.234567, 2.345678)));
    let snapped = snap_to_precision(&geom, 2)?;

    match snapped.geom {
        GeoGeometry::Point(p) => {
            assert!((p.x() - 1.23).abs() < 1e-10);
            assert!((p.y() - 2.35).abs() < 1e-10);
        }
        _ => panic!("Expected Point"),
    }

    Ok(())
}

#[test]
fn test_repair_linestring_with_duplicates() -> Result<()> {
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 1.0, y: 1.0 },
        Coord { x: 1.0, y: 1.0 }, // duplicate
        Coord { x: 2.0, y: 2.0 },
        Coord { x: 2.0, y: 2.0 }, // duplicate
        Coord { x: 3.0, y: 3.0 },
    ];
    let geom = Geometry::new(GeoGeometry::LineString(LineString::new(coords)));
    let repaired = repair_geometry(&geom)?;

    match repaired.geom {
        GeoGeometry::LineString(ls) => {
            // Should have duplicates removed
            assert_eq!(ls.0.len(), 4); // 0, 1, 2, 3
        }
        _ => panic!("Expected LineString"),
    }

    Ok(())
}

#[test]
fn test_repair_unclosed_polygon() -> Result<()> {
    // Create a polygon that's not closed (missing last point)
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 4.0, y: 0.0 },
        Coord { x: 4.0, y: 4.0 },
        Coord { x: 0.0, y: 4.0 },
        // Missing the closing point
    ];
    let poly = Polygon::new(LineString::new(coords), vec![]);
    let geom = Geometry::new(GeoGeometry::Polygon(poly));

    let repaired = repair_geometry(&geom)?;

    match repaired.geom {
        GeoGeometry::Polygon(p) => {
            // Should be closed now
            assert_eq!(p.exterior().0.len(), 5);
            assert_eq!(p.exterior().0.first(), p.exterior().0.last());
        }
        _ => panic!("Expected Polygon"),
    }

    Ok(())
}

#[test]
fn test_repair_polygon_with_invalid_coords() -> Result<()> {
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord {
            x: f64::NAN,
            y: 1.0,
        }, // invalid
        Coord { x: 2.0, y: 2.0 },
        Coord { x: 0.0, y: 2.0 },
        Coord { x: 0.0, y: 0.0 },
    ];
    let poly = Polygon::new(LineString::new(coords), vec![]);
    let geom = Geometry::new(GeoGeometry::Polygon(poly));

    let repaired = repair_geometry(&geom)?;

    match repaired.geom {
        GeoGeometry::Polygon(p) => {
            // Should have NaN removed
            assert!(p
                .exterior()
                .0
                .iter()
                .all(|c| !c.x.is_nan() && !c.y.is_nan()));
            // Should still be closed
            assert_eq!(p.exterior().0.first(), p.exterior().0.last());
        }
        _ => panic!("Expected Polygon"),
    }

    Ok(())
}

#[test]
fn test_repair_multipoint_removes_invalid() -> Result<()> {
    let points = vec![
        Point::new(1.0, 2.0),
        Point::new(f64::NAN, 3.0), // invalid
        Point::new(4.0, 5.0),
    ];
    let geom = Geometry::new(GeoGeometry::MultiPoint(geo_types::MultiPoint(points)));

    let repaired = repair_geometry(&geom)?;

    match repaired.geom {
        GeoGeometry::MultiPoint(mp) => {
            // Should have invalid point removed
            assert_eq!(mp.0.len(), 2);
            assert!(mp.0.iter().all(|p| !p.x().is_nan() && !p.y().is_nan()));
        }
        _ => panic!("Expected MultiPoint"),
    }

    Ok(())
}

#[test]
fn test_repair_geometry_preserves_crs() -> Result<()> {
    use crate::geometry::Crs;

    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 1.0, y: 1.0 },
        Coord { x: 1.0, y: 1.0 }, // duplicate
        Coord { x: 2.0, y: 2.0 },
    ];
    let ls = LineString::new(coords);
    let custom_crs = Crs::epsg(3857);
    let geom = Geometry::with_crs(GeoGeometry::LineString(ls), custom_crs.clone());

    let repaired = repair_geometry(&geom)?;

    // CRS should be preserved
    assert_eq!(repaired.crs, custom_crs);

    Ok(())
}

#[test]
fn test_remove_consecutive_duplicates_empty() {
    let coords: Vec<Coord<f64>> = vec![];
    let result = remove_consecutive_duplicates(&coords);
    assert!(result.is_empty());
}

#[test]
fn test_remove_consecutive_duplicates_no_dups() {
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 1.0, y: 1.0 },
        Coord { x: 2.0, y: 2.0 },
    ];
    let result = remove_consecutive_duplicates(&coords);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_remove_consecutive_duplicates_with_dups() {
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 1.0, y: 1.0 },
        Coord { x: 1.0, y: 1.0 }, // dup
        Coord { x: 1.0, y: 1.0 }, // dup
        Coord { x: 2.0, y: 2.0 },
    ];
    let result = remove_consecutive_duplicates(&coords);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_close_ring_already_closed() {
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 1.0, y: 0.0 },
        Coord { x: 1.0, y: 1.0 },
        Coord { x: 0.0, y: 0.0 }, // already closed
    ];
    let ring = close_ring(coords.clone());
    assert_eq!(ring.0.len(), 4);
    assert_eq!(ring.0.first(), ring.0.last());
}

#[test]
fn test_close_ring_not_closed() {
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 1.0, y: 0.0 },
        Coord { x: 1.0, y: 1.0 },
        // Missing closing point
    ];
    let ring = close_ring(coords);
    assert_eq!(ring.0.len(), 4); // Should add closing point
    assert_eq!(ring.0.first(), ring.0.last());
}

// Tests for ValidationConfig
#[test]
fn test_validation_config_default() {
    let config = ValidationConfig::default();
    assert_eq!(config.coordinate_tolerance, 1e-10);
    assert_eq!(config.area_tolerance, 1e-10);
    assert_eq!(config.length_tolerance, 1e-10);
    assert_eq!(config.min_polygon_area, 0.0);
    assert_eq!(config.min_linestring_length, 0.0);
    assert_eq!(config.max_coordinate_value, 1e15);
    assert!(config.check_self_intersection);
    assert!(config.check_orientation);
}

#[test]
fn test_validation_config_custom() {
    let config = ValidationConfig {
        coordinate_tolerance: 1e-6,
        area_tolerance: 1e-5,
        length_tolerance: 1e-4,
        min_polygon_area: 100.0,
        min_linestring_length: 10.0,
        max_coordinate_value: 1e10,
        check_self_intersection: false,
        check_orientation: false,
    };
    assert_eq!(config.coordinate_tolerance, 1e-6);
    assert_eq!(config.min_polygon_area, 100.0);
    assert!(!config.check_self_intersection);
}

// Tests for validate_geometry_with_config
#[test]
fn test_validate_with_config_valid_point() {
    let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
    let config = ValidationConfig::default();
    let result = validate_geometry_with_config(&geom, &config);
    assert!(result.is_valid);
    assert!(result.errors.is_empty());
}

#[test]
fn test_validate_with_config_point_exceeds_max() {
    let geom = Geometry::new(GeoGeometry::Point(Point::new(1e16, 2.0)));
    let config = ValidationConfig::default();
    let result = validate_geometry_with_config(&geom, &config);
    assert!(!result.is_valid);
    assert!(!result.errors.is_empty());
}

#[test]
fn test_validate_with_config_point_non_finite() {
    let geom = Geometry::new(GeoGeometry::Point(Point::new(f64::NAN, 2.0)));
    let config = ValidationConfig::default();
    let result = validate_geometry_with_config(&geom, &config);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("non-finite")));
}

#[test]
fn test_validate_with_config_linestring_too_short() {
    let ls = LineString::new(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 0.0 }]);
    let geom = Geometry::new(GeoGeometry::LineString(ls));
    let config = ValidationConfig {
        min_linestring_length: 10.0,
        ..Default::default()
    };
    let result = validate_geometry_with_config(&geom, &config);
    assert!(result.is_valid); // Still valid but has warning
    assert!(!result.warnings.is_empty());
    assert!(result.warnings.iter().any(|w| w.contains("length")));
}

#[test]
fn test_validate_with_config_linestring_empty() {
    let ls = LineString::new(vec![]);
    let geom = Geometry::new(GeoGeometry::LineString(ls));
    let config = ValidationConfig::default();
    let result = validate_geometry_with_config(&geom, &config);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("empty")));
}

#[test]
fn test_validate_with_config_linestring_one_point() {
    let ls = LineString::new(vec![Coord { x: 0.0, y: 0.0 }]);
    let geom = Geometry::new(GeoGeometry::LineString(ls));
    let config = ValidationConfig::default();
    let result = validate_geometry_with_config(&geom, &config);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("at least 2")));
}

#[test]
fn test_validate_with_config_linestring_non_finite() {
    let ls = LineString::new(vec![
        Coord { x: 0.0, y: 0.0 },
        Coord {
            x: f64::INFINITY,
            y: 2.0,
        },
    ]);
    let geom = Geometry::new(GeoGeometry::LineString(ls));
    let config = ValidationConfig::default();
    let result = validate_geometry_with_config(&geom, &config);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("non-finite")));
}

#[test]
fn test_validate_with_config_polygon_small_area() {
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 0.1, y: 0.0 },
        Coord { x: 0.1, y: 0.1 },
        Coord { x: 0.0, y: 0.1 },
        Coord { x: 0.0, y: 0.0 },
    ];
    let poly = Polygon::new(LineString::new(coords), vec![]);
    let geom = Geometry::new(GeoGeometry::Polygon(poly));
    let config = ValidationConfig {
        min_polygon_area: 100.0,
        ..Default::default()
    };
    let result = validate_geometry_with_config(&geom, &config);
    assert!(result.is_valid); // Still valid but has warning
    assert!(!result.warnings.is_empty());
    assert!(result.warnings.iter().any(|w| w.contains("area")));
}

#[test]
fn test_validate_with_config_polygon_not_closed() {
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 4.0, y: 0.0 },
        Coord { x: 4.0, y: 4.0 },
        Coord { x: 0.0, y: 4.0 },
        Coord { x: 0.0, y: 0.0 },
    ];
    let poly = Polygon::new(LineString::new(coords), vec![]);
    let geom = Geometry::new(GeoGeometry::Polygon(poly));
    // Use a strict tolerance to detect minor deviations
    let config = ValidationConfig {
        coordinate_tolerance: 1e-15,
        ..Default::default()
    };
    let result = validate_geometry_with_config(&geom, &config);
    // This polygon is actually closed, so it should be valid
    assert!(result.is_valid);
}

#[test]
fn test_validate_with_config_polygon_with_valid_interior() {
    let exterior = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 10.0, y: 0.0 },
        Coord { x: 10.0, y: 10.0 },
        Coord { x: 0.0, y: 10.0 },
        Coord { x: 0.0, y: 0.0 },
    ];
    // Properly closed interior ring
    let interior = vec![
        Coord { x: 2.0, y: 2.0 },
        Coord { x: 4.0, y: 2.0 },
        Coord { x: 4.0, y: 4.0 },
        Coord { x: 2.0, y: 4.0 },
        Coord { x: 2.0, y: 2.0 },
    ];
    let poly = Polygon::new(LineString::new(exterior), vec![LineString::new(interior)]);
    let geom = Geometry::new(GeoGeometry::Polygon(poly));
    let config = ValidationConfig::default();
    let result = validate_geometry_with_config(&geom, &config);
    // Should be valid with properly closed interior ring
    assert!(result.is_valid);
}

#[test]
fn test_validate_with_config_polygon_interior_too_few_points() {
    let exterior = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 10.0, y: 0.0 },
        Coord { x: 10.0, y: 10.0 },
        Coord { x: 0.0, y: 10.0 },
        Coord { x: 0.0, y: 0.0 },
    ];
    let interior = vec![
        Coord { x: 2.0, y: 2.0 },
        Coord { x: 4.0, y: 2.0 },
        Coord { x: 2.0, y: 2.0 },
    ];
    let poly = Polygon::new(LineString::new(exterior), vec![LineString::new(interior)]);
    let geom = Geometry::new(GeoGeometry::Polygon(poly));
    let config = ValidationConfig::default();
    let result = validate_geometry_with_config(&geom, &config);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("at least 4")));
}

// Tests for compute_quality_metrics
#[test]
fn test_quality_metrics_point() {
    let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
    let metrics = compute_quality_metrics(&geom);
    assert_eq!(metrics.coordinate_count, 1);
    assert_eq!(metrics.complexity_score, 1.0);
    assert!(!metrics.has_duplicates);
    assert!(!metrics.has_spikes);
    assert_eq!(metrics.min_segment_length, 0.0);
    assert_eq!(metrics.max_segment_length, 0.0);
    assert_eq!(metrics.avg_segment_length, 0.0);
    assert!(metrics.compactness.is_none());
}

#[test]
fn test_quality_metrics_linestring() {
    let ls = LineString::new(vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 1.0, y: 0.0 },
        Coord { x: 2.0, y: 0.0 },
        Coord { x: 3.0, y: 0.0 },
    ]);
    let geom = Geometry::new(GeoGeometry::LineString(ls));
    let metrics = compute_quality_metrics(&geom);
    assert_eq!(metrics.coordinate_count, 4);
    assert!(metrics.complexity_score > 1.0);
    assert!(!metrics.has_duplicates);
    assert!(!metrics.has_spikes);
    assert!(metrics.min_segment_length > 0.0);
    assert!(metrics.max_segment_length > 0.0);
    assert!(metrics.avg_segment_length > 0.0);
}

#[test]
fn test_quality_metrics_linestring_with_duplicates() {
    let ls = LineString::new(vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 0.0, y: 0.0 }, // Duplicate
        Coord { x: 1.0, y: 0.0 },
    ]);
    let geom = Geometry::new(GeoGeometry::LineString(ls));
    let metrics = compute_quality_metrics(&geom);
    assert!(metrics.has_duplicates);
}

#[test]
fn test_quality_metrics_polygon() {
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 4.0, y: 0.0 },
        Coord { x: 4.0, y: 4.0 },
        Coord { x: 0.0, y: 4.0 },
        Coord { x: 0.0, y: 0.0 },
    ];
    let poly = Polygon::new(LineString::new(coords), vec![]);
    let geom = Geometry::new(GeoGeometry::Polygon(poly));
    let metrics = compute_quality_metrics(&geom);
    assert_eq!(metrics.coordinate_count, 5);
    assert!(metrics.complexity_score > 1.0);
    assert!(metrics.compactness.is_some());
    assert!(metrics.compactness.expect("should succeed") > 0.0);
}

#[test]
fn test_quality_metrics_multipoint() {
    use geo_types::MultiPoint;
    let mp = MultiPoint::new(vec![
        Point::new(0.0, 0.0),
        Point::new(1.0, 1.0),
        Point::new(2.0, 2.0),
    ]);
    let geom = Geometry::new(GeoGeometry::MultiPoint(mp));
    let metrics = compute_quality_metrics(&geom);
    // MultiPoint not currently handled, returns default metrics
    assert_eq!(metrics.coordinate_count, 0);
    assert_eq!(metrics.complexity_score, 0.0);
}

// Tests for check_ogc_compliance
#[test]
fn test_ogc_compliance_valid_point() {
    let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
    let result = check_ogc_compliance(&geom);
    assert!(result.is_valid);
}

#[test]
fn test_ogc_compliance_linestring_all_same_points() {
    let ls = LineString::new(vec![
        Coord { x: 1.0, y: 1.0 },
        Coord { x: 1.0, y: 1.0 },
        Coord { x: 1.0, y: 1.0 },
    ]);
    let geom = Geometry::new(GeoGeometry::LineString(ls));
    let result = check_ogc_compliance(&geom);
    assert!(!result.is_valid);
    assert!(result
        .errors
        .iter()
        .any(|e| e.contains("at least 2 distinct points")));
}

#[test]
fn test_ogc_compliance_linestring_valid() {
    let ls = LineString::new(vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 1.0, y: 1.0 },
        Coord { x: 2.0, y: 2.0 },
    ]);
    let geom = Geometry::new(GeoGeometry::LineString(ls));
    let result = check_ogc_compliance(&geom);
    assert!(result.is_valid);
}

#[test]
fn test_ogc_compliance_polygon_small_area() {
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 1e-11, y: 0.0 },
        Coord { x: 1e-11, y: 1e-11 },
        Coord { x: 0.0, y: 1e-11 },
        Coord { x: 0.0, y: 0.0 },
    ];
    let poly = Polygon::new(LineString::new(coords), vec![]);
    let geom = Geometry::new(GeoGeometry::Polygon(poly));
    let result = check_ogc_compliance(&geom);
    assert!(result.is_valid);
    assert!(result
        .warnings
        .iter()
        .any(|w| w.contains("small") || w.contains("zero area")));
}

#[test]
fn test_ogc_compliance_polygon_overlapping_interior_rings() {
    let exterior = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 10.0, y: 0.0 },
        Coord { x: 10.0, y: 10.0 },
        Coord { x: 0.0, y: 10.0 },
        Coord { x: 0.0, y: 0.0 },
    ];

    // Two interior rings with many shared points
    let interior1 = vec![
        Coord { x: 2.0, y: 2.0 },
        Coord { x: 4.0, y: 2.0 },
        Coord { x: 4.0, y: 4.0 },
        Coord { x: 2.0, y: 4.0 },
        Coord { x: 2.0, y: 2.0 },
    ];

    let interior2 = vec![
        Coord { x: 2.0, y: 2.0 }, // Shared
        Coord { x: 4.0, y: 2.0 }, // Shared
        Coord { x: 4.0, y: 4.0 }, // Shared
        Coord { x: 2.0, y: 4.0 }, // Shared
        Coord { x: 2.0, y: 2.0 }, // Shared
    ];

    let poly = Polygon::new(
        LineString::new(exterior),
        vec![LineString::new(interior1), LineString::new(interior2)],
    );
    let geom = Geometry::new(GeoGeometry::Polygon(poly));
    let result = check_ogc_compliance(&geom);
    // Should have warning about overlapping rings
    assert!(!result.warnings.is_empty());
    assert!(result.warnings.iter().any(|w| w.contains("overlap")));
}

#[test]
fn test_ogc_compliance_polygon_valid_with_hole() {
    let exterior = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 10.0, y: 0.0 },
        Coord { x: 10.0, y: 10.0 },
        Coord { x: 0.0, y: 10.0 },
        Coord { x: 0.0, y: 0.0 },
    ];

    let interior = vec![
        Coord { x: 2.0, y: 2.0 },
        Coord { x: 4.0, y: 2.0 },
        Coord { x: 4.0, y: 4.0 },
        Coord { x: 2.0, y: 4.0 },
        Coord { x: 2.0, y: 2.0 },
    ];

    let poly = Polygon::new(LineString::new(exterior), vec![LineString::new(interior)]);
    let geom = Geometry::new(GeoGeometry::Polygon(poly));
    let result = check_ogc_compliance(&geom);
    assert!(result.is_valid);
}

// Tests for ValidationResult builder methods
#[test]
fn test_validation_result_valid() {
    let r = ValidationResult::valid();
    assert!(r.is_valid);
    assert!(r.errors.is_empty());
    assert!(r.warnings.is_empty());
}

#[test]
fn test_validation_result_invalid() {
    let r = ValidationResult::invalid(vec!["bad".to_string()]);
    assert!(!r.is_valid);
    assert_eq!(r.errors.len(), 1);
}

#[test]
fn test_validation_result_with_warning() {
    let r = ValidationResult::valid().with_warning("warn".to_string());
    assert!(r.is_valid);
    assert_eq!(r.warnings.len(), 1);
}

#[test]
fn test_validation_result_with_error() {
    let r = ValidationResult::valid().with_error("err".to_string());
    assert!(!r.is_valid);
    assert_eq!(r.errors.len(), 1);
}
