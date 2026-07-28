//! # QueryExecutor - GeoSPARQL filter functions
//!
//! Minimal GeoSPARQL / spatial support for the serial query executor:
//!
//! * `geof:distance(geomA, geomB, unitsIRI)` — OGC GeoSPARQL 1.0/1.1 distance
//!   over `geo:wktLiteral` `POINT` geometries, returned in the requested unit of
//!   measure (metre / kilometre / statute mile).
//! * `<http://oxirs.io/fn/geo#distanceLatLon>(lat1, lon1, lat2, lon2)` — an OxiRS
//!   extension convenience function that takes four `xsd:decimal`/`xsd:double`
//!   coordinates directly and returns the great-circle distance in metres. It
//!   exists because real-world W3C WGS84 (`geo:lat`/`geo:long`) datasets store
//!   bare decimal coordinates rather than WKT literals.
//!
//! Distances use the Haversine great-circle formula on a spherical Earth of mean
//! radius 6,371,000 m — numerically identical to `oxirs-geosparql`'s `haversine`.
//! The formula and WKT `POINT` parsing are re-implemented inline (≈40 lines) so
//! `oxirs-arq` need not take a dependency on `oxirs-geosparql` (which pulls in
//! `geo`/`wkt`/`rstar`/`scirs2-core` and would add a build-serialisation edge);
//! see the accompanying report for the rationale.
//!
//! Error handling follows the existing built-in convention: a recognised
//! function that fails on its arguments (wrong arity, non-numeric input, invalid
//! WKT, unknown units) returns an ordinary `anyhow` error — never an
//! `UnknownFunctionError` — so that inside a `FILTER` the offending row is
//! excluded (SPARQL 1.1 §17.3) rather than the whole query aborting.

use anyhow::{anyhow, Result};
use oxirs_core::model::NamedNode;

use super::queryexecutor_type::QueryExecutor;
use crate::algebra::{Binding, Expression, Literal, Term};

/// OGC GeoSPARQL `geof:distance` (full IRI).
const GEOF_DISTANCE: &str = "http://www.opengis.net/def/function/geosparql/distance";
/// OxiRS extension: four-argument lat/long great-circle distance in metres.
const OXIRS_DISTANCE_LAT_LON: &str = "http://oxirs.io/fn/geo#distanceLatLon";

/// OGC unit-of-measure IRI for metres.
const UOM_METRE: &str = "http://www.opengis.net/def/uom/OGC/1.0/metre";
/// EPSG code 9001 (metre) — a common alias for the OGC metre unit.
const UOM_METRE_EPSG: &str = "http://www.opengis.net/def/uom/EPSG/0/9001";
/// OGC unit-of-measure IRI for kilometres.
const UOM_KILOMETRE: &str = "http://www.opengis.net/def/uom/OGC/1.0/kilometre";
/// OGC unit-of-measure IRI for statute miles.
const UOM_MILE: &str = "http://www.opengis.net/def/uom/OGC/1.0/mile";

/// `xsd:double` datatype IRI.
const XSD_DOUBLE: &str = "http://www.w3.org/2001/XMLSchema#double";

/// OGC CRS84 — WGS84 longitude/latitude, axis order `(longitude, latitude)`.
/// This is the GeoSPARQL default coordinate reference system assumed when a
/// `wktLiteral` carries no leading CRS URI.
const CRS84: &str = "http://www.opengis.net/def/crs/OGC/1.3/CRS84";
/// EPSG:4326 — WGS84 with the EPSG-registered axis order `(latitude, longitude)`,
/// i.e. the *opposite* coordinate order from CRS84.
const EPSG_4326: &str = "http://www.opengis.net/def/crs/EPSG/0/4326";

/// Mean Earth radius in metres (matches `oxirs-geosparql::distance_calculator`).
const EARTH_RADIUS_M: f64 = 6_371_000.0;
/// Length of one statute mile in metres.
const METRES_PER_MILE: f64 = 1_609.344;

/// A geographic coordinate in degrees, ordered `(latitude, longitude)`.
type LatLon = (f64, f64);

impl QueryExecutor {
    /// Attempt to evaluate `name(args…)` as a supported GeoSPARQL / OxiRS geo
    /// function.
    ///
    /// Returns `None` when `name` is not a geo function (the caller then raises
    /// `UnknownFunctionError`). Returns `Some(Ok(term))` on success and
    /// `Some(Err(..))` when a *recognised* function fails on its arguments — an
    /// ordinary evaluation error, so a `FILTER` excludes that row rather than
    /// failing the query.
    pub(super) fn try_geosparql_function(
        &self,
        name: &str,
        args: &[Expression],
        binding: &Binding,
    ) -> Option<Result<Term>> {
        match name {
            GEOF_DISTANCE => Some(self.geof_distance(args, binding)),
            OXIRS_DISTANCE_LAT_LON => Some(self.geo_distance_lat_lon(args, binding)),
            _ => None,
        }
    }

    /// `geof:distance(geomA, geomB, unitsIRI)` over WKT `POINT` geometries.
    fn geof_distance(&self, args: &[Expression], binding: &Binding) -> Result<Term> {
        if args.len() != 3 {
            return Err(anyhow!(
                "geof:distance requires exactly 3 arguments (geomA, geomB, unitsIRI)"
            ));
        }
        let geom_a = self.evaluate_expression(&args[0], binding)?;
        let geom_b = self.evaluate_expression(&args[1], binding)?;
        let units = self.evaluate_expression(&args[2], binding)?;

        let point_a = geometry_point(&geom_a)?;
        let point_b = geometry_point(&geom_b)?;
        let metres = haversine_metres(point_a, point_b);
        let units_iri = units_iri_str(&units)?;
        let value = convert_from_metres(metres, &units_iri)?;
        Ok(double_literal(value))
    }

    /// `oxgeo:distanceLatLon(lat1, lon1, lat2, lon2)` → metres.
    fn geo_distance_lat_lon(&self, args: &[Expression], binding: &Binding) -> Result<Term> {
        if args.len() != 4 {
            return Err(anyhow!(
                "distanceLatLon requires exactly 4 arguments (lat1, lon1, lat2, lon2)"
            ));
        }
        let lat1 = self.numeric_arg(&args[0], binding)?;
        let lon1 = self.numeric_arg(&args[1], binding)?;
        let lat2 = self.numeric_arg(&args[2], binding)?;
        let lon2 = self.numeric_arg(&args[3], binding)?;
        let metres = haversine_metres((lat1, lon1), (lat2, lon2));
        Ok(double_literal(metres))
    }

    /// Evaluate an argument expression and coerce the result to `f64`, reusing
    /// the datatype-agnostic numeric extraction shared with comparison ops.
    fn numeric_arg(&self, expr: &Expression, binding: &Binding) -> Result<f64> {
        let term = self.evaluate_expression(expr, binding)?;
        self.extract_numeric_value(&term)
    }
}

/// Great-circle distance in metres between two `(lat, lon)` points (Haversine,
/// spherical Earth). Identical formula and radius to `oxirs-geosparql`.
fn haversine_metres(a: LatLon, b: LatLon) -> f64 {
    let (lat_a, lon_a) = a;
    let (lat_b, lon_b) = b;

    let phi_a = lat_a.to_radians();
    let phi_b = lat_b.to_radians();
    let d_phi = (lat_b - lat_a).to_radians();
    let d_lambda = (lon_b - lon_a).to_radians();

    let sin_half_dphi = (d_phi / 2.0).sin();
    let sin_half_dlambda = (d_lambda / 2.0).sin();
    let h = sin_half_dphi * sin_half_dphi
        + phi_a.cos() * phi_b.cos() * sin_half_dlambda * sin_half_dlambda;

    2.0 * EARTH_RADIUS_M * h.sqrt().asin()
}

/// Extract a `(lat, lon)` point from a geometry term. Only `POINT` WKT literals
/// are supported in this minimal implementation.
fn geometry_point(term: &Term) -> Result<LatLon> {
    match term {
        Term::Literal(lit) => parse_wkt_point(&lit.value),
        other => Err(anyhow!(
            "geof:distance geometry argument must be a wktLiteral, got {other}"
        )),
    }
}

/// Parse the minimal WKT the distance path needs — a `POINT` with two
/// coordinates and an optional leading CRS URI (`<http://…/CRS84> POINT(…)`) —
/// **honouring the CRS's axis order** and returning the point as `(latitude,
/// longitude)` regardless of how it was written.
///
/// GeoSPARQL embeds the CRS in the `wktLiteral` value. The two CRSs in practical
/// use for WGS84 disagree on axis order, and getting it wrong silently swaps a
/// point's latitude and longitude — a large, plausible-looking distance error
/// (Tokyo's `35.68 139.77` read as `lon lat` lands in the Arabian Sea). So:
///
/// * **No CRS URI** → GeoSPARQL default **CRS84**, axis order `(lon, lat)`.
/// * **CRS84** (`…/OGC/1.3/CRS84`) → `(lon, lat)`.
/// * **EPSG:4326** (`…/EPSG/0/4326`) → `(lat, lon)` — the coordinates are
///   swapped relative to CRS84.
/// * **Any other CRS URI** → an error. Per the module convention this is an
///   ordinary evaluation error, so inside a `FILTER` the offending row is
///   excluded rather than silently mis-projected.
fn parse_wkt_point(raw: &str) -> Result<LatLon> {
    let mut text = raw.trim();

    // Optional leading CRS URI: `<...>` followed by the geometry. Absence means
    // the GeoSPARQL default CRS (CRS84).
    let mut lat_first = false;
    if let Some(rest) = text.strip_prefix('<') {
        let close = rest
            .find('>')
            .ok_or_else(|| anyhow!("invalid WKT CRS prefix (unterminated '<'): {raw}"))?;
        let crs = rest[..close].trim();
        lat_first = match crs {
            CRS84 => false,
            EPSG_4326 => true,
            other => {
                return Err(anyhow!(
                    "unsupported WKT coordinate reference system '{other}' in {raw} \
                     (only CRS84 and EPSG:4326 are supported)"
                ));
            }
        };
        text = rest[close + 1..].trim_start();
    }

    let open = text
        .find('(')
        .ok_or_else(|| anyhow!("invalid WKT (missing '('): {raw}"))?;
    let keyword = text[..open].trim();
    if !keyword.eq_ignore_ascii_case("point") {
        return Err(anyhow!(
            "unsupported WKT geometry (only POINT is supported): {raw}"
        ));
    }
    let close = text
        .rfind(')')
        .ok_or_else(|| anyhow!("invalid WKT (missing ')'): {raw}"))?;
    if close < open {
        return Err(anyhow!("invalid WKT parentheses: {raw}"));
    }

    let inner = text[open + 1..close].trim();
    let mut coords = inner.split_whitespace();
    let first = coords
        .next()
        .ok_or_else(|| anyhow!("WKT POINT missing first coordinate: {raw}"))?;
    let second = coords
        .next()
        .ok_or_else(|| anyhow!("WKT POINT missing second coordinate: {raw}"))?;
    if coords.next().is_some() {
        return Err(anyhow!("WKT POINT expects exactly two coordinates: {raw}"));
    }

    let first: f64 = first
        .parse()
        .map_err(|_| anyhow!("invalid WKT coordinate '{first}' in {raw}"))?;
    let second: f64 = second
        .parse()
        .map_err(|_| anyhow!("invalid WKT coordinate '{second}' in {raw}"))?;

    // Map the two written coordinates onto (latitude, longitude) using the CRS's
    // axis order. CRS84 (and the no-CRS default) is (lon, lat); EPSG:4326 is
    // (lat, lon).
    if lat_first {
        Ok((first, second))
    } else {
        Ok((second, first))
    }
}

/// Extract the unit-of-measure IRI string from a units argument, accepting both
/// an IRI term and a plain string literal carrying the IRI.
fn units_iri_str(term: &Term) -> Result<String> {
    match term {
        Term::Iri(iri) => Ok(iri.as_str().to_string()),
        Term::Literal(lit) => Ok(lit.value.clone()),
        other => Err(anyhow!("units argument must be an IRI, got {other}")),
    }
}

/// Convert a distance in metres into the requested unit of measure.
fn convert_from_metres(metres: f64, units_iri: &str) -> Result<f64> {
    match units_iri {
        UOM_METRE | UOM_METRE_EPSG => Ok(metres),
        UOM_KILOMETRE => Ok(metres / 1000.0),
        UOM_MILE => Ok(metres / METRES_PER_MILE),
        other => Err(anyhow!("unsupported unit of measure: {other}")),
    }
}

/// Build an `xsd:double` literal term from a numeric value.
fn double_literal(value: f64) -> Term {
    Term::Literal(Literal {
        value: value.to_string(),
        language: None,
        datatype: Some(NamedNode::new_unchecked(XSD_DOUBLE)),
    })
}
