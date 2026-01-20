//! SPARQL Function Integration
//!
//! This module provides integration with oxirs-arq for registering GeoSPARQL functions
//! as SPARQL filter and property functions.
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_geosparql::sparql_integration::register_geosparql_functions;
//!
//! // Register all GeoSPARQL functions with oxirs-arq
//! register_geosparql_functions(&mut function_registry);
//! ```

use crate::vocabulary;

/// GeoSPARQL function metadata
#[derive(Debug, Clone)]
pub struct GeoSparqlFunction {
    /// Function URI (e.g., `http://www.opengis.net/def/function/geosparql/sfContains`)
    pub uri: String,
    /// Function name (e.g., "sfContains")
    pub name: String,
    /// Function description
    pub description: String,
    /// Number of arguments
    pub arity: usize,
    /// Function category (filter, property, distance)
    pub category: FunctionCategory,
}

/// Function category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionCategory {
    /// Filter function (returns boolean)
    Filter,
    /// Property function (returns geometry or value)
    Property,
    /// Distance function
    Distance,
}

/// Get all registered GeoSPARQL Simple Features functions
pub fn get_simple_features_functions() -> Vec<GeoSparqlFunction> {
    vec![
        GeoSparqlFunction {
            uri: vocabulary::GEO_SF_EQUALS.to_string(),
            name: "sfEquals".to_string(),
            description: "Tests if two geometries are spatially equal".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_SF_DISJOINT.to_string(),
            name: "sfDisjoint".to_string(),
            description: "Tests if two geometries are spatially disjoint".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_SF_INTERSECTS.to_string(),
            name: "sfIntersects".to_string(),
            description: "Tests if two geometries spatially intersect".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_SF_TOUCHES.to_string(),
            name: "sfTouches".to_string(),
            description: "Tests if two geometries spatially touch".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_SF_CROSSES.to_string(),
            name: "sfCrosses".to_string(),
            description: "Tests if two geometries spatially cross".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_SF_WITHIN.to_string(),
            name: "sfWithin".to_string(),
            description: "Tests if geometry A is within geometry B".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_SF_CONTAINS.to_string(),
            name: "sfContains".to_string(),
            description: "Tests if geometry A contains geometry B".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_SF_OVERLAPS.to_string(),
            name: "sfOverlaps".to_string(),
            description: "Tests if two geometries spatially overlap".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
    ]
}

/// Get all registered GeoSPARQL Egenhofer functions
#[cfg(feature = "geos-backend")]
pub fn get_egenhofer_functions() -> Vec<GeoSparqlFunction> {
    vec![
        GeoSparqlFunction {
            uri: vocabulary::GEO_EH_EQUALS.to_string(),
            name: "ehEquals".to_string(),
            description: "Egenhofer relation: geometries are equal".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_EH_DISJOINT.to_string(),
            name: "ehDisjoint".to_string(),
            description: "Egenhofer relation: geometries are disjoint".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_EH_MEET.to_string(),
            name: "ehMeet".to_string(),
            description: "Egenhofer relation: geometries meet at boundary".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_EH_OVERLAP.to_string(),
            name: "ehOverlap".to_string(),
            description: "Egenhofer relation: geometries overlap".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_EH_COVERS.to_string(),
            name: "ehCovers".to_string(),
            description: "Egenhofer relation: geometry A covers geometry B".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_EH_COVERED_BY.to_string(),
            name: "ehCoveredBy".to_string(),
            description: "Egenhofer relation: geometry A is covered by geometry B".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_EH_INSIDE.to_string(),
            name: "ehInside".to_string(),
            description: "Egenhofer relation: geometry A is inside geometry B".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_EH_CONTAINS.to_string(),
            name: "ehContains".to_string(),
            description: "Egenhofer relation: geometry A contains geometry B".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
    ]
}

/// Get all registered GeoSPARQL RCC8 functions
#[cfg(feature = "geos-backend")]
pub fn get_rcc8_functions() -> Vec<GeoSparqlFunction> {
    vec![
        GeoSparqlFunction {
            uri: vocabulary::GEO_RCC8_EQ.to_string(),
            name: "rcc8eq".to_string(),
            description: "RCC8 relation: regions are equal".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_RCC8_DC.to_string(),
            name: "rcc8dc".to_string(),
            description: "RCC8 relation: regions are disconnected".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_RCC8_EC.to_string(),
            name: "rcc8ec".to_string(),
            description: "RCC8 relation: regions are externally connected".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_RCC8_PO.to_string(),
            name: "rcc8po".to_string(),
            description: "RCC8 relation: regions partially overlap".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_RCC8_TPPI.to_string(),
            name: "rcc8tppi".to_string(),
            description: "RCC8 relation: tangential proper part inverse".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_RCC8_TPP.to_string(),
            name: "rcc8tpp".to_string(),
            description: "RCC8 relation: tangential proper part".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_RCC8_NTPP.to_string(),
            name: "rcc8ntpp".to_string(),
            description: "RCC8 relation: non-tangential proper part".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
        GeoSparqlFunction {
            uri: vocabulary::GEO_RCC8_NTPPI.to_string(),
            name: "rcc8ntppi".to_string(),
            description: "RCC8 relation: non-tangential proper part inverse".to_string(),
            arity: 2,
            category: FunctionCategory::Filter,
        },
    ]
}

/// Get all property functions
pub fn get_property_functions() -> Vec<GeoSparqlFunction> {
    vec![
        GeoSparqlFunction {
            uri: format!("{}dimension", vocabulary::GEOSPARQL_NS),
            name: "dimension".to_string(),
            description: "Returns the dimension of a geometry".to_string(),
            arity: 1,
            category: FunctionCategory::Property,
        },
        GeoSparqlFunction {
            uri: format!("{}coordinateDimension", vocabulary::GEOSPARQL_NS),
            name: "coordinateDimension".to_string(),
            description: "Returns the coordinate dimension (2D, 3D, etc.)".to_string(),
            arity: 1,
            category: FunctionCategory::Property,
        },
        GeoSparqlFunction {
            uri: format!("{}spatialDimension", vocabulary::GEOSPARQL_NS),
            name: "spatialDimension".to_string(),
            description: "Returns the spatial dimension".to_string(),
            arity: 1,
            category: FunctionCategory::Property,
        },
        GeoSparqlFunction {
            uri: format!("{}isEmpty", vocabulary::GEOSPARQL_NS),
            name: "isEmpty".to_string(),
            description: "Tests if a geometry is empty".to_string(),
            arity: 1,
            category: FunctionCategory::Property,
        },
        GeoSparqlFunction {
            uri: format!("{}isSimple", vocabulary::GEOSPARQL_NS),
            name: "isSimple".to_string(),
            description: "Tests if a geometry is simple (no self-intersections)".to_string(),
            arity: 1,
            category: FunctionCategory::Property,
        },
        GeoSparqlFunction {
            uri: format!("{}is3D", vocabulary::GEOSPARQL_NS),
            name: "is3D".to_string(),
            description: "Tests if a geometry has Z coordinates".to_string(),
            arity: 1,
            category: FunctionCategory::Property,
        },
        GeoSparqlFunction {
            uri: format!("{}isMeasured", vocabulary::GEOSPARQL_NS),
            name: "isMeasured".to_string(),
            description: "Tests if a geometry has M coordinates".to_string(),
            arity: 1,
            category: FunctionCategory::Property,
        },
    ]
}

/// Get all distance functions
pub fn get_distance_functions() -> Vec<GeoSparqlFunction> {
    vec![GeoSparqlFunction {
        uri: vocabulary::GEO_DISTANCE.to_string(),
        name: "distance".to_string(),
        description: "Returns the minimum distance between two geometries".to_string(),
        arity: 2,
        category: FunctionCategory::Distance,
    }]
}

/// Get all registered GeoSPARQL functions
pub fn get_all_geosparql_functions() -> Vec<GeoSparqlFunction> {
    let mut functions = Vec::new();

    functions.extend(get_simple_features_functions());

    #[cfg(feature = "geos-backend")]
    {
        functions.extend(get_egenhofer_functions());
        functions.extend(get_rcc8_functions());
    }

    functions.extend(get_property_functions());
    functions.extend(get_distance_functions());

    functions
}

/// Print SPARQL function registration code for oxirs-arq
///
/// This generates example code showing how to register GeoSPARQL functions
/// with the oxirs-arq query engine.
pub fn print_registration_example() {
    println!("// Example: Register GeoSPARQL functions with oxirs-arq");
    println!("// Add this to your oxirs-arq integration code:\n");
    println!("use oxirs_geosparql::sparql_integration::{{get_all_geosparql_functions, FunctionCategory}};");
    println!("use oxirs_geosparql::geometry::Geometry;");
    println!("use oxirs_geosparql::functions::{{simple_features, egenhofer, rcc8}};\n");
    println!("let functions = get_all_geosparql_functions();");
    println!("for func in functions {{");
    println!("    match func.category {{");
    println!("        FunctionCategory::Filter => {{");
    println!("            // Register as SPARQL FILTER function");
    println!("            // registry.register_filter(&func.uri, func.arity, |args| {{ ... }});");
    println!("        }}");
    println!("        FunctionCategory::Property => {{");
    println!("            // Register as SPARQL property function");
    println!("            // registry.register_property(&func.uri, func.arity, |args| {{ ... }});");
    println!("        }}");
    println!("        FunctionCategory::Distance => {{");
    println!("            // Register as distance function");
    println!("            // registry.register_distance(&func.uri, func.arity, |args| {{ ... }});");
    println!("        }}");
    println!("    }}");
    println!("}}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_simple_features_functions() {
        let functions = get_simple_features_functions();
        assert_eq!(functions.len(), 8);

        // Check first function
        assert_eq!(functions[0].name, "sfEquals");
        assert_eq!(functions[0].arity, 2);
        assert_eq!(functions[0].category, FunctionCategory::Filter);
    }

    #[test]
    #[cfg(feature = "geos-backend")]
    fn test_get_egenhofer_functions() {
        let functions = get_egenhofer_functions();
        assert_eq!(functions.len(), 8);

        // Check first function
        assert_eq!(functions[0].name, "ehEquals");
        assert_eq!(functions[0].arity, 2);
        assert_eq!(functions[0].category, FunctionCategory::Filter);
    }

    #[test]
    #[cfg(feature = "geos-backend")]
    fn test_get_rcc8_functions() {
        let functions = get_rcc8_functions();
        assert_eq!(functions.len(), 8);

        // Check first function
        assert_eq!(functions[0].name, "rcc8eq");
        assert_eq!(functions[0].arity, 2);
        assert_eq!(functions[0].category, FunctionCategory::Filter);
    }

    #[test]
    fn test_get_property_functions() {
        let functions = get_property_functions();
        assert_eq!(functions.len(), 7);

        // Check dimension function
        let dim_func = functions.iter().find(|f| f.name == "dimension").unwrap();
        assert_eq!(dim_func.arity, 1);
        assert_eq!(dim_func.category, FunctionCategory::Property);
    }

    #[test]
    fn test_get_distance_functions() {
        let functions = get_distance_functions();
        assert_eq!(functions.len(), 1);

        // Check distance function
        assert_eq!(functions[0].name, "distance");
        assert_eq!(functions[0].arity, 2);
        assert_eq!(functions[0].category, FunctionCategory::Distance);
    }

    #[test]
    fn test_get_all_geosparql_functions() {
        let functions = get_all_geosparql_functions();

        // Should have at least Simple Features + Property + Distance functions
        #[cfg(not(feature = "geos-backend"))]
        assert_eq!(functions.len(), 8 + 7 + 1); // SF + Property + Distance

        #[cfg(feature = "geos-backend")]
        assert_eq!(functions.len(), 8 + 8 + 8 + 7 + 1); // SF + Egenhofer + RCC8 + Property + Distance
    }

    #[test]
    fn test_function_categories() {
        let functions = get_all_geosparql_functions();

        let filter_count = functions
            .iter()
            .filter(|f| f.category == FunctionCategory::Filter)
            .count();
        let property_count = functions
            .iter()
            .filter(|f| f.category == FunctionCategory::Property)
            .count();
        let distance_count = functions
            .iter()
            .filter(|f| f.category == FunctionCategory::Distance)
            .count();

        assert!(filter_count > 0);
        assert_eq!(property_count, 7);
        assert_eq!(distance_count, 1);
    }
}
