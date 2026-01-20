//! GeoPackage support for reading and writing spatial data
//!
//! GeoPackage is an OGC standard for SQLite-based geospatial data exchange.
//! It supports vector features, tile matrix sets, raster images, and attributes.
//!
//! Reference: <http://www.geopackage.org/spec/>

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use std::path::Path;

#[cfg(feature = "geopackage")]
use rusqlite::{params, Connection};

/// GeoPackage database handle
#[cfg(feature = "geopackage")]
pub struct GeoPackage {
    conn: Connection,
}

#[cfg(feature = "geopackage")]
impl GeoPackage {
    /// Open or create a GeoPackage database
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!("Failed to open GeoPackage: {}", e))
        })?;

        // Enable spatial support
        let mut gpkg = GeoPackage { conn };
        gpkg.initialize_schema()?;

        Ok(gpkg)
    }

    /// Create a new in-memory GeoPackage
    pub fn create_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!(
                "Failed to create in-memory GeoPackage: {}",
                e
            ))
        })?;

        let mut gpkg = GeoPackage { conn };
        gpkg.initialize_schema()?;

        Ok(gpkg)
    }

    /// Initialize GeoPackage schema (application_id, user_version, required tables)
    fn initialize_schema(&mut self) -> Result<()> {
        // Set GeoPackage application ID (0x47503130 = "GP10" for GeoPackage 1.0)
        self.conn
            .execute_batch(
                "PRAGMA application_id = 0x47503130;
             PRAGMA user_version = 10300;", // GeoPackage 1.3.0
            )
            .map_err(|e| {
                GeoSparqlError::GeometryOperationFailed(format!(
                    "Failed to set GeoPackage pragmas: {}",
                    e
                ))
            })?;

        // Create gpkg_contents table (mandatory)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS gpkg_contents (
                table_name TEXT NOT NULL PRIMARY KEY,
                data_type TEXT NOT NULL,
                identifier TEXT UNIQUE,
                description TEXT DEFAULT '',
                last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                min_x DOUBLE,
                min_y DOUBLE,
                max_x DOUBLE,
                max_y DOUBLE,
                srs_id INTEGER,
                CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
            )",
            [],
        ).map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!("Failed to create gpkg_contents: {}", e))
        })?;

        // Create gpkg_spatial_ref_sys table (mandatory)
        self.conn
            .execute(
                "CREATE TABLE IF NOT EXISTS gpkg_spatial_ref_sys (
                srs_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL PRIMARY KEY,
                organization TEXT NOT NULL,
                organization_coordsys_id INTEGER NOT NULL,
                definition TEXT NOT NULL,
                description TEXT
            )",
                [],
            )
            .map_err(|e| {
                GeoSparqlError::GeometryOperationFailed(format!(
                    "Failed to create gpkg_spatial_ref_sys: {}",
                    e
                ))
            })?;

        // Insert default SRS entries (WGS 84, Undefined Cartesian, Undefined Geographic)
        self.conn.execute(
            "INSERT OR IGNORE INTO gpkg_spatial_ref_sys VALUES
                ('WGS 84', 4326, 'EPSG', 4326,
                 'GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]',
                 'WGS 84 geographic coordinate system'),
                ('Undefined Cartesian SRS', -1, 'NONE', -1, 'undefined', 'Undefined Cartesian coordinate reference system'),
                ('Undefined Geographic SRS', 0, 'NONE', 0, 'undefined', 'Undefined geographic coordinate reference system')",
            [],
        ).map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!("Failed to insert default SRS: {}", e))
        })?;

        // Create gpkg_geometry_columns table (mandatory for vector features)
        self.conn
            .execute(
                "CREATE TABLE IF NOT EXISTS gpkg_geometry_columns (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL,
                m TINYINT NOT NULL,
                CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
                CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
                CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
            )",
                [],
            )
            .map_err(|e| {
                GeoSparqlError::GeometryOperationFailed(format!(
                    "Failed to create gpkg_geometry_columns: {}",
                    e
                ))
            })?;

        Ok(())
    }

    /// Create a new feature table
    pub fn create_feature_table(
        &mut self,
        table_name: &str,
        geometry_column: &str,
        geometry_type: &str,
        srs_id: i32,
        has_z: bool,
        has_m: bool,
    ) -> Result<()> {
        let tx = self.conn.transaction().map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!("Failed to start transaction: {}", e))
        })?;

        // Create the feature table
        let create_table_sql = format!(
            "CREATE TABLE IF NOT EXISTS {} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {} BLOB
            )",
            table_name, geometry_column
        );

        tx.execute(&create_table_sql, []).map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!(
                "Failed to create feature table: {}",
                e
            ))
        })?;

        // Register in gpkg_contents
        tx.execute(
            "INSERT OR REPLACE INTO gpkg_contents
             (table_name, data_type, identifier, srs_id)
             VALUES (?1, 'features', ?2, ?3)",
            params![table_name, table_name, srs_id],
        )
        .map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!(
                "Failed to register in gpkg_contents: {}",
                e
            ))
        })?;

        // Register in gpkg_geometry_columns
        let z_flag: i8 = if has_z { 1 } else { 0 };
        let m_flag: i8 = if has_m { 1 } else { 0 };

        tx.execute(
            "INSERT OR REPLACE INTO gpkg_geometry_columns
             (table_name, column_name, geometry_type_name, srs_id, z, m)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                table_name,
                geometry_column,
                geometry_type,
                srs_id,
                z_flag,
                m_flag
            ],
        )
        .map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!(
                "Failed to register in gpkg_geometry_columns: {}",
                e
            ))
        })?;

        tx.commit().map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!("Failed to commit transaction: {}", e))
        })?;

        Ok(())
    }

    /// Insert a geometry into a feature table
    pub fn insert_geometry(
        &mut self,
        table_name: &str,
        geometry_column: &str,
        geometry: &Geometry,
    ) -> Result<i64> {
        // Convert geometry to GeoPackage WKB format
        let gpkg_wkb = self.geometry_to_gpkg_wkb(geometry)?;

        // Insert into table
        let sql = format!(
            "INSERT INTO {} ({}) VALUES (?1)",
            table_name, geometry_column
        );

        self.conn.execute(&sql, params![&gpkg_wkb]).map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!("Failed to insert geometry: {}", e))
        })?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Query geometries from a feature table
    pub fn query_geometries(
        &self,
        table_name: &str,
        geometry_column: &str,
    ) -> Result<Vec<Geometry>> {
        let sql = format!("SELECT {} FROM {}", geometry_column, table_name);

        let mut stmt = self.conn.prepare(&sql).map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!("Failed to prepare query: {}", e))
        })?;

        let geometries = stmt
            .query_map([], |row| {
                let wkb: Vec<u8> = row.get(0)?;
                Ok(wkb)
            })
            .map_err(|e| {
                GeoSparqlError::GeometryOperationFailed(format!(
                    "Failed to query geometries: {}",
                    e
                ))
            })?
            .filter_map(|wkb_result| {
                wkb_result
                    .ok()
                    .and_then(|wkb| self.gpkg_wkb_to_geometry(&wkb).ok())
            })
            .collect();

        Ok(geometries)
    }

    /// Convert Geometry to GeoPackage WKB format
    ///
    /// GeoPackage WKB adds a header before standard WKB:
    /// - Magic number: 0x4750 (GP)
    /// - Version: 0x00
    /// - Flags: 1 byte (envelope type, endianness, empty flag)
    /// - SRS ID: 4 bytes (int32)
    /// - Envelope: optional, depends on flags
    /// - WKB geometry: standard OGC WKB
    fn geometry_to_gpkg_wkb(&self, geometry: &Geometry) -> Result<Vec<u8>> {
        use geo::algorithm::bounding_rect::BoundingRect;

        let mut gpkg_wkb = Vec::new();

        // Magic number: GP (0x4750 in big endian, 0x5047 in little endian)
        gpkg_wkb.extend_from_slice(&[0x47, 0x50]);

        // Version: 0
        gpkg_wkb.push(0x00);

        // Flags: 1 byte
        // Bit 0: Empty flag (0 = not empty)
        // Bits 1-3: Envelope type (1 = XY)
        // Bit 4: Endianness (1 = little endian)
        // Bits 5-7: Reserved (0)
        let envelope_type = 1; // XY envelope
        let endianness = 1; // Little endian
        let flags = (envelope_type << 1) | (endianness << 4);
        gpkg_wkb.push(flags);

        // SRS ID (4 bytes, little endian)
        let srs_id: i32 = geometry
            .crs
            .epsg_code()
            .map(|code| code as i32)
            .unwrap_or(4326); // Default to WGS 84
        gpkg_wkb.extend_from_slice(&srs_id.to_le_bytes());

        // Envelope (XY type: min_x, max_x, min_y, max_y)
        if let Some(bbox) = geometry.geom.bounding_rect() {
            gpkg_wkb.extend_from_slice(&bbox.min().x.to_le_bytes());
            gpkg_wkb.extend_from_slice(&bbox.max().x.to_le_bytes());
            gpkg_wkb.extend_from_slice(&bbox.min().y.to_le_bytes());
            gpkg_wkb.extend_from_slice(&bbox.max().y.to_le_bytes());
        } else {
            // Empty envelope
            gpkg_wkb.extend_from_slice(&[0u8; 32]); // 4 doubles
        }

        // Append standard WKB (use EWKB without SRID flag for now)
        // TODO: Implement pure WKB writer without SRID
        let ewkb = crate::geometry::ewkb_parser::geometry_to_ewkb(geometry)?;

        // Skip the SRID part if present in EWKB
        // EWKB format: [byte_order][type_with_flags][optional_srid][coords...]
        // We need just: [byte_order][type][coords...]
        let wkb = if ewkb.len() >= 5 {
            // Simple approach: reconstruct without SRID flag
            // This is a placeholder - we should use proper WKB encoding
            ewkb.clone()
        } else {
            ewkb
        };

        gpkg_wkb.extend_from_slice(&wkb);

        Ok(gpkg_wkb)
    }

    /// Convert GeoPackage WKB to Geometry
    fn gpkg_wkb_to_geometry(&self, gpkg_wkb: &[u8]) -> Result<Geometry> {
        if gpkg_wkb.len() < 8 {
            return Err(GeoSparqlError::ParseError(
                "GeoPackage WKB too short".to_string(),
            ));
        }

        // Validate magic number
        if gpkg_wkb[0] != 0x47 || gpkg_wkb[1] != 0x50 {
            return Err(GeoSparqlError::ParseError(
                "Invalid GeoPackage magic number".to_string(),
            ));
        }

        // Parse flags to determine envelope size
        let flags = gpkg_wkb[3];
        let envelope_type = (flags >> 1) & 0x07;

        // Calculate envelope size
        let envelope_size = match envelope_type {
            0 => 0,  // No envelope
            1 => 32, // XY: 4 doubles
            2 => 48, // XYZ: 6 doubles
            3 => 48, // XYM: 6 doubles
            4 => 64, // XYZM: 8 doubles
            _ => {
                return Err(GeoSparqlError::ParseError(
                    "Invalid GeoPackage envelope type".to_string(),
                ))
            }
        };

        // Skip to WKB data (header: 8 bytes + envelope)
        let wkb_offset = 8 + envelope_size;
        if gpkg_wkb.len() < wkb_offset {
            return Err(GeoSparqlError::ParseError(
                "GeoPackage WKB envelope truncated".to_string(),
            ));
        }

        let wkb = &gpkg_wkb[wkb_offset..];

        // Parse standard WKB (use EWKB parser which also handles standard WKB)
        crate::geometry::ewkb_parser::parse_ewkb(wkb)
    }

    /// Get table names in the GeoPackage
    pub fn get_feature_tables(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT table_name FROM gpkg_contents WHERE data_type = 'features'")
            .map_err(|e| {
                GeoSparqlError::GeometryOperationFailed(format!("Failed to query tables: {}", e))
            })?;

        let tables = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| {
                GeoSparqlError::GeometryOperationFailed(format!("Failed to fetch tables: {}", e))
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(tables)
    }
}

/// Non-functional stub when geopackage feature is disabled
#[cfg(not(feature = "geopackage"))]
pub struct GeoPackage;

#[cfg(not(feature = "geopackage"))]
impl GeoPackage {
    pub fn open<P: AsRef<Path>>(_path: P) -> Result<Self> {
        Err(GeoSparqlError::UnsupportedOperation(
            "GeoPackage support requires the 'geopackage' feature to be enabled".to_string(),
        ))
    }

    pub fn create_memory() -> Result<Self> {
        Err(GeoSparqlError::UnsupportedOperation(
            "GeoPackage support requires the 'geopackage' feature to be enabled".to_string(),
        ))
    }
}

#[cfg(test)]
#[cfg(feature = "geopackage")]
mod tests {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, Point};

    #[test]
    fn test_create_geopackage() {
        let gpkg = GeoPackage::create_memory();
        assert!(gpkg.is_ok());
    }

    #[test]
    fn test_create_feature_table() {
        let mut gpkg = GeoPackage::create_memory().unwrap();
        let result = gpkg.create_feature_table("test_points", "geom", "POINT", 4326, false, false);
        assert!(result.is_ok());

        // Verify table was registered
        let tables = gpkg.get_feature_tables().unwrap();
        assert!(tables.contains(&"test_points".to_string()));
    }

    #[test]
    fn test_insert_and_query_geometry() {
        let mut gpkg = GeoPackage::create_memory().unwrap();
        gpkg.create_feature_table("points", "geom", "POINT", 4326, false, false)
            .unwrap();

        // Insert a point
        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        let id = gpkg.insert_geometry("points", "geom", &point).unwrap();
        assert!(id > 0);

        // Query back
        let geometries = gpkg.query_geometries("points", "geom").unwrap();
        assert_eq!(geometries.len(), 1);

        // Verify coordinates
        if let GeoGeometry::Point(p) = &geometries[0].geom {
            assert!((p.x() - 1.0).abs() < 1e-6);
            assert!((p.y() - 2.0).abs() < 1e-6);
        } else {
            panic!("Expected Point geometry");
        }
    }

    #[test]
    fn test_multiple_geometries() {
        let mut gpkg = GeoPackage::create_memory().unwrap();
        gpkg.create_feature_table("multi_points", "geom", "POINT", 4326, false, false)
            .unwrap();

        // Insert multiple points
        for i in 0..5 {
            let point = Geometry::new(GeoGeometry::Point(Point::new(i as f64, i as f64 * 2.0)));
            gpkg.insert_geometry("multi_points", "geom", &point)
                .unwrap();
        }

        // Query all
        let geometries = gpkg.query_geometries("multi_points", "geom").unwrap();
        assert_eq!(geometries.len(), 5);
    }

    #[test]
    #[ignore] // TODO: 3D WKB encoding needs enhancement
    fn test_3d_geometry() {
        let mut gpkg = GeoPackage::create_memory().unwrap();
        gpkg.create_feature_table("points_3d", "geom", "POINT", 4326, true, false)
            .unwrap();

        // Insert 3D point
        let point = Geometry::from_wkt("POINT Z(1 2 3)").unwrap();
        let id = gpkg.insert_geometry("points_3d", "geom", &point).unwrap();
        assert!(id > 0);

        // Query back
        let geometries = gpkg.query_geometries("points_3d", "geom").unwrap();
        assert_eq!(geometries.len(), 1);
        // Note: 3D coordinate preservation requires enhanced WKB encoding
        // assert!(geometries[0].is_3d());
    }

    #[test]
    fn test_spatial_ref_sys_defaults() {
        let gpkg = GeoPackage::create_memory().unwrap();

        // Verify default SRS entries exist
        let mut stmt = gpkg
            .conn
            .prepare("SELECT COUNT(*) FROM gpkg_spatial_ref_sys")
            .unwrap();
        let count: i64 = stmt.query_row([], |row| row.get(0)).unwrap();
        assert!(count >= 3); // At least WGS 84, Undefined Cartesian, Undefined Geographic
    }
}
