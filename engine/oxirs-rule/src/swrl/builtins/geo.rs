//! SWRL Geographic Built-in Functions
//!
//! This module implements geographic/spatial operations for SWRL rules including:
//! - Distance calculations: distance, within
//! - Spatial relations: geo_contains, geo_intersects
//! - Area calculations: geo_area

use anyhow::Result;

use super::super::types::SwrlArgument;
use super::utils::*;

pub(crate) fn builtin_distance(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 5 {
        return Err(anyhow::anyhow!(
            "distance requires exactly 5 arguments: lat1, lon1, lat2, lon2, result"
        ));
    }

    let lat1 = extract_numeric_value(&args[0])?.to_radians();
    let lon1 = extract_numeric_value(&args[1])?.to_radians();
    let lat2 = extract_numeric_value(&args[2])?.to_radians();
    let lon2 = extract_numeric_value(&args[3])?.to_radians();
    let expected_distance = extract_numeric_value(&args[4])?;

    // Haversine formula for great circle distance
    let earth_radius = 6371.0; // Earth radius in kilometers
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    let distance = earth_radius * c;

    Ok((distance - expected_distance).abs() < 0.001) // 1 meter tolerance
}

pub(crate) fn builtin_within(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 5 {
        return Err(anyhow::anyhow!(
            "within requires exactly 5 arguments: lat1, lon1, lat2, lon2, max_distance"
        ));
    }

    let lat1 = extract_numeric_value(&args[0])?.to_radians();
    let lon1 = extract_numeric_value(&args[1])?.to_radians();
    let lat2 = extract_numeric_value(&args[2])?.to_radians();
    let lon2 = extract_numeric_value(&args[3])?.to_radians();
    let max_distance = extract_numeric_value(&args[4])?;

    // Haversine formula for great circle distance
    let earth_radius = 6371.0; // Earth radius in kilometers
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    let distance = earth_radius * c;

    Ok(distance <= max_distance)
}

pub(crate) fn builtin_geo_contains(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 9 {
        return Err(anyhow::anyhow!("contains requires exactly 9 arguments: poly_min_lat, poly_min_lon, poly_max_lat, poly_max_lon, point_lat, point_lon, result"));
    }

    // Simple bounding box containment check
    let min_lat = extract_numeric_value(&args[0])?;
    let min_lon = extract_numeric_value(&args[1])?;
    let max_lat = extract_numeric_value(&args[2])?;
    let max_lon = extract_numeric_value(&args[3])?;
    let point_lat = extract_numeric_value(&args[4])?;
    let point_lon = extract_numeric_value(&args[5])?;
    let expected_result = extract_string_value(&args[6])? == "true";

    let contains = point_lat >= min_lat
        && point_lat <= max_lat
        && point_lon >= min_lon
        && point_lon <= max_lon;

    Ok(contains == expected_result)
}

pub(crate) fn builtin_geo_intersects(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 8 {
        return Err(anyhow::anyhow!("intersects requires exactly 8 arguments: box1_min_lat, box1_min_lon, box1_max_lat, box1_max_lon, box2_min_lat, box2_min_lon, box2_max_lat, box2_max_lon"));
    }

    // Check if two bounding boxes intersect
    let box1_min_lat = extract_numeric_value(&args[0])?;
    let box1_min_lon = extract_numeric_value(&args[1])?;
    let box1_max_lat = extract_numeric_value(&args[2])?;
    let box1_max_lon = extract_numeric_value(&args[3])?;
    let box2_min_lat = extract_numeric_value(&args[4])?;
    let box2_min_lon = extract_numeric_value(&args[5])?;
    let box2_max_lat = extract_numeric_value(&args[6])?;
    let box2_max_lon = extract_numeric_value(&args[7])?;

    // Two boxes intersect if they overlap in both dimensions
    let intersects = !(box1_max_lat < box2_min_lat
        || box2_max_lat < box1_min_lat
        || box1_max_lon < box2_min_lon
        || box2_max_lon < box1_min_lon);

    Ok(intersects)
}

pub(crate) fn builtin_geo_area(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 5 {
        return Err(anyhow::anyhow!(
            "area requires exactly 5 arguments: min_lat, min_lon, max_lat, max_lon, result"
        ));
    }

    // Calculate approximate area of a bounding box in square kilometers
    let min_lat = extract_numeric_value(&args[0])?;
    let min_lon = extract_numeric_value(&args[1])?;
    let max_lat = extract_numeric_value(&args[2])?;
    let max_lon = extract_numeric_value(&args[3])?;
    let expected_area = extract_numeric_value(&args[4])?;

    // Simple approximation: treat Earth as sphere
    const EARTH_RADIUS_KM: f64 = 6371.0;

    // Convert to radians
    let lat1_rad = min_lat.to_radians();
    let lat2_rad = max_lat.to_radians();
    let lon_diff_rad = (max_lon - min_lon).to_radians();

    // Approximate area calculation
    let area = EARTH_RADIUS_KM * EARTH_RADIUS_KM * lon_diff_rad * (lat2_rad.sin() - lat1_rad.sin());

    // Allow some tolerance for floating point comparison
    Ok((area.abs() - expected_area).abs() < 0.1)
}
