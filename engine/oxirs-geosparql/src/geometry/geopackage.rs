//! GeoPackage support for reading and writing spatial data
//!
//! GeoPackage is an OGC standard for SQLite-based geospatial data exchange.
//! It supports vector features, tile matrix sets, raster images, and attributes.
//!
//! Reference: <http://www.geopackage.org/spec/>
//!
//! # Backend
//!
//! This implementation uses the COOLJAPAN Pure-Rust OxiSQL engine
//! (`oxisql-sqlite-compat` / `oxisql-core`) — no C or `libsqlite3` dependency.
//!
//! # Sync ↔ async bridge
//!
//! `GeoPackage` owns a current-thread Tokio runtime and drives every database
//! operation through `runtime.block_on(...)`.  This keeps the public API
//! synchronous while the underlying engine is async.
//!
//! # WAL checkpoint
//!
//! The OxiSQL engine uses WAL mode.  The GPKG application-id magic
//! (`0x47503130`) reaches the main database file bytes (offset 68) **only**
//! after a `PRAGMA wal_checkpoint` is executed.  Call [`GeoPackage::checkpoint`]
//! before handing a file-backed GeoPackage to an external GIS reader.

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use std::path::Path;

#[cfg(feature = "geopackage")]
use oxisql_core::{Connection, ToSqlValue, Value};
#[cfg(feature = "geopackage")]
use oxisql_sqlite_compat::SqliteConnection;
#[cfg(feature = "geopackage")]
use tokio::runtime::Runtime;

/// GeoPackage application-id magic: "GP10" = GeoPackage 1.0
#[cfg(feature = "geopackage")]
const GPKG_APPLICATION_ID: i64 = 0x4750_3130; // 1195724080

/// GeoPackage database handle.
///
/// Backed by the Pure-Rust OxiSQL engine (no `libsqlite3`).
#[cfg(feature = "geopackage")]
pub struct GeoPackage {
    pub(crate) conn: SqliteConnection,
    pub(crate) runtime: Runtime,
}

#[cfg(feature = "geopackage")]
impl GeoPackage {
    /// Open or create a GeoPackage database at `path`.
    ///
    /// The GeoPackage schema (required tables + default SRS rows) is initialised
    /// automatically on first open.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_str().ok_or_else(|| {
            GeoSparqlError::InvalidParameter("Path is not valid UTF-8".to_string())
        })?;

        let runtime = build_runtime()?;
        let conn = runtime
            .block_on(SqliteConnection::open(path_str))
            .map_err(|e| {
                GeoSparqlError::GeometryOperationFailed(format!(
                    "Failed to open GeoPackage at '{}': {}",
                    path_str, e
                ))
            })?;

        let gpkg = GeoPackage { conn, runtime };
        gpkg.initialize_schema()?;
        Ok(gpkg)
    }

    /// Create a new in-memory GeoPackage.
    ///
    /// Useful for tests and transient processing; changes are lost on drop.
    pub fn create_memory() -> Result<Self> {
        let runtime = build_runtime()?;
        let conn = runtime
            .block_on(SqliteConnection::open(":memory:"))
            .map_err(|e| {
                GeoSparqlError::GeometryOperationFailed(format!(
                    "Failed to create in-memory GeoPackage: {}",
                    e
                ))
            })?;

        let gpkg = GeoPackage { conn, runtime };
        gpkg.initialize_schema()?;
        Ok(gpkg)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Execute a statement (no rows returned).  Params use `$1`, `$2`, … placeholders.
    fn exec(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<u64> {
        self.runtime
            .block_on(self.conn.execute(sql, params))
            .map_err(|e| {
                GeoSparqlError::GeometryOperationFailed(format!("SQL execute error: {}", e))
            })
    }

    /// Execute a query and return all result rows.  Params use `$1`, `$2`, … placeholders.
    fn query(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<Vec<oxisql_core::Row>> {
        self.runtime
            .block_on(self.conn.query(sql, params))
            .map_err(|e| GeoSparqlError::GeometryOperationFailed(format!("SQL query error: {}", e)))
    }

    /// Execute DDL or multi-statement SQL (no parameters).
    fn exec_batch(&self, sql: &str) -> Result<()> {
        self.runtime
            .block_on(self.conn.execute_batch(sql))
            .map(|_| ())
            .map_err(|e| {
                GeoSparqlError::GeometryOperationFailed(format!(
                    "Failed to execute SQL batch: {}",
                    e
                ))
            })
    }

    /// Return the row-id of the most recently inserted row on this connection.
    fn last_insert_rowid(&self) -> Result<i64> {
        let rows = self.query("SELECT last_insert_rowid()", &[])?;
        match rows.first().and_then(|r| r.get_by_index(0)) {
            Some(Value::I64(n)) => Ok(*n),
            other => Err(GeoSparqlError::GeometryOperationFailed(format!(
                "last_insert_rowid: unexpected value {:?}",
                other
            ))),
        }
    }

    // -----------------------------------------------------------------------
    // Schema initialisation
    // -----------------------------------------------------------------------

    /// Initialise GeoPackage schema: `application_id`, `user_version`, required tables,
    /// and default SRS entries.
    fn initialize_schema(&self) -> Result<()> {
        // Set GeoPackage application ID ("GP10" = 0x47503130 = 1195724080)
        // and user_version (1.3.0 → 10300).
        // oxisql-core requires decimal literals in PRAGMA statements.
        let pragma_sql = format!(
            "PRAGMA application_id = {}; PRAGMA user_version = 10300;",
            GPKG_APPLICATION_ID
        );
        self.exec_batch(&pragma_sql)?;

        // gpkg_spatial_ref_sys
        // Note: `srs_id INTEGER PRIMARY KEY` (rowid alias, no explicit NOT NULL) is
        // used instead of `srs_id INTEGER NOT NULL PRIMARY KEY` to work around a
        // positional-parameter binding bug in the OxiSQL/Limbo alpha engine.  SQLite
        // rowid aliases are implicitly NOT NULL, so the GeoPackage contract is upheld.
        // Other NOT NULL constraints are also omitted for the same reason; the
        // application enforces data integrity at the Rust layer.
        // FOREIGN KEY clauses are omitted: SQLite only enforces them when
        // PRAGMA foreign_keys = ON, and the Limbo engine does not yet support it;
        // external GeoPackage readers do not depend on FK enforcement.
        self.exec_batch(
            "CREATE TABLE IF NOT EXISTS gpkg_spatial_ref_sys (srs_id INTEGER PRIMARY KEY, srs_name TEXT, organization TEXT, organization_coordsys_id INTEGER, definition TEXT, description TEXT)",
        )?;

        // gpkg_contents
        self.exec_batch(
            "CREATE TABLE IF NOT EXISTS gpkg_contents (table_name TEXT PRIMARY KEY, data_type TEXT, identifier TEXT UNIQUE, description TEXT DEFAULT '', last_change DATETIME DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')), min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER)",
        )?;

        // gpkg_geometry_columns
        self.exec_batch(
            "CREATE TABLE IF NOT EXISTS gpkg_geometry_columns (table_name TEXT, column_name TEXT, geometry_type_name TEXT, srs_id INTEGER, z TINYINT, m TINYINT, PRIMARY KEY (table_name, column_name))",
        )?;

        // Insert default SRS entries (WGS 84, Undefined Cartesian, Undefined Geographic).
        // Use three separate INSERT OR IGNORE statements to avoid multi-row VALUES
        // syntax that might not be supported by the engine.
        let wgs84_wkt = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]";
        let wgs84_desc = "WGS 84 geographic coordinate system";

        // WGS 84 (srs_id = 4326)
        {
            let srs_id_4326: i64 = 4326;
            let org_id_4326: i64 = 4326;
            self.exec(
                "INSERT OR IGNORE INTO gpkg_spatial_ref_sys (srs_name, srs_id, organization, organization_coordsys_id, definition, description) VALUES ($1, $2, $3, $4, $5, $6)",
                &[
                    &"WGS 84",
                    &srs_id_4326,
                    &"EPSG",
                    &org_id_4326,
                    &wgs84_wkt,
                    &wgs84_desc,
                ],
            )?;
        }

        // Undefined Cartesian SRS (srs_id = -1)
        {
            let srs_id_neg1: i64 = -1;
            let org_id_neg1: i64 = -1;
            self.exec(
                "INSERT OR IGNORE INTO gpkg_spatial_ref_sys (srs_name, srs_id, organization, organization_coordsys_id, definition) VALUES ($1, $2, $3, $4, $5)",
                &[
                    &"Undefined Cartesian SRS",
                    &srs_id_neg1,
                    &"NONE",
                    &org_id_neg1,
                    &"undefined",
                ],
            )?;
        }

        // Undefined Geographic SRS (srs_id = 0)
        {
            let srs_id_0: i64 = 0_i64;
            let org_id_0: i64 = 0_i64;
            self.exec(
                "INSERT OR IGNORE INTO gpkg_spatial_ref_sys (srs_name, srs_id, organization, organization_coordsys_id, definition) VALUES ($1, $2, $3, $4, $5)",
                &[
                    &"Undefined Geographic SRS",
                    &srs_id_0,
                    &"NONE",
                    &org_id_0,
                    &"undefined",
                ],
            )?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Create a new feature table and register it in the GeoPackage metadata.
    ///
    /// `table_name` is also used as the `identifier` in `gpkg_contents`.
    /// Executes sequentially (no transactional rollback is available in the
    /// OxiSQL alpha engine; all individual steps are idempotent via `IF NOT EXISTS`
    /// / `OR REPLACE`).
    pub fn create_feature_table(
        &mut self,
        table_name: &str,
        geometry_column: &str,
        geometry_type: &str,
        srs_id: i32,
        has_z: bool,
        has_m: bool,
    ) -> Result<()> {
        // 1. Create the feature table (idempotent).
        let create_table_sql = format!(
            "CREATE TABLE IF NOT EXISTS {} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {} BLOB
            )",
            table_name, geometry_column
        );
        self.exec_batch(&create_table_sql)?;

        // 2. Register in gpkg_contents (OR REPLACE = full-row replacement, safe here).
        let srs_id_i64 = i64::from(srs_id);
        self.exec(
            "INSERT OR REPLACE INTO gpkg_contents
             (table_name, data_type, identifier, srs_id)
             VALUES ($1, 'features', $2, $3)",
            &[&table_name, &table_name, &srs_id_i64],
        )?;

        // 3. Register in gpkg_geometry_columns (OR REPLACE, safe here).
        let z_flag: i64 = if has_z { 1 } else { 0 };
        let m_flag: i64 = if has_m { 1 } else { 0 };
        self.exec(
            "INSERT OR REPLACE INTO gpkg_geometry_columns
             (table_name, column_name, geometry_type_name, srs_id, z, m)
             VALUES ($1, $2, $3, $4, $5, $6)",
            &[
                &table_name,
                &geometry_column,
                &geometry_type,
                &srs_id_i64,
                &z_flag,
                &m_flag,
            ],
        )?;

        Ok(())
    }

    /// Insert a geometry into a feature table.
    ///
    /// Returns the row-id of the newly inserted row.
    pub fn insert_geometry(
        &mut self,
        table_name: &str,
        geometry_column: &str,
        geometry: &Geometry,
    ) -> Result<i64> {
        let gpkg_wkb: Vec<u8> = self.geometry_to_gpkg_wkb(geometry)?;

        let sql = format!(
            "INSERT INTO {} ({}) VALUES ($1)",
            table_name, geometry_column
        );

        // Blobs must be bound as Vec<u8> (not &[u8]) — the ToSqlValue impl
        // is provided for Vec<u8> by oxisql-core.
        self.exec(&sql, &[&gpkg_wkb])?;
        self.last_insert_rowid()
    }

    /// Query all geometries from a feature table.
    pub fn query_geometries(
        &self,
        table_name: &str,
        geometry_column: &str,
    ) -> Result<Vec<Geometry>> {
        let sql = format!("SELECT {} FROM {}", geometry_column, table_name);
        let rows = self.query(&sql, &[])?;

        let mut geometries = Vec::with_capacity(rows.len());
        for row in rows.iter() {
            match row.get_by_index(0) {
                Some(Value::Blob(blob)) => {
                    match self.gpkg_wkb_to_geometry(blob) {
                        Ok(g) => geometries.push(g),
                        Err(_) => {
                            // Skip malformed rows (matching prior filter_map behaviour)
                        }
                    }
                }
                Some(Value::Null) | None => {
                    // NULL geometry — skip
                }
                other => {
                    return Err(GeoSparqlError::GeometryOperationFailed(format!(
                        "query_geometries: expected BLOB, got {:?}",
                        other
                    )));
                }
            }
        }

        Ok(geometries)
    }

    /// Return the names of all feature tables registered in `gpkg_contents`.
    pub fn get_feature_tables(&self) -> Result<Vec<String>> {
        let rows = self.query(
            "SELECT table_name FROM gpkg_contents WHERE data_type = 'features'",
            &[],
        )?;

        let mut tables = Vec::with_capacity(rows.len());
        for row in rows.iter() {
            match row.get_by_index(0) {
                Some(Value::Text(name)) => tables.push(name.clone()),
                Some(Value::Null) | None => {}
                other => {
                    return Err(GeoSparqlError::GeometryOperationFailed(format!(
                        "get_feature_tables: unexpected value {:?}",
                        other
                    )));
                }
            }
        }

        Ok(tables)
    }

    /// Flush WAL to the main database file.
    ///
    /// **Must be called after all writes are complete** on a file-backed
    /// GeoPackage before the file is handed off to an external reader.
    /// The OxiSQL engine writes through WAL; the application-id magic and data
    /// only appear at the expected byte offsets in the main file after a
    /// checkpoint.
    pub fn checkpoint(&self) -> Result<()> {
        self.exec_batch("PRAGMA wal_checkpoint;")?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // WKB encoding / decoding
    // -----------------------------------------------------------------------

    /// Convert [`Geometry`] to the GeoPackage binary WKB envelope format.
    ///
    /// Layout (GeoPackage spec §2.1.3):
    /// - `[0..2]`  Magic: `0x47 0x50` ("GP")
    /// - `[2]`     Version: `0x00`
    /// - `[3]`     Flags byte (envelope type = XY=1, endianness = LE)
    /// - `[4..8]`  SRS ID (i32 LE)
    /// - `[8..40]` Envelope: min_x, max_x, min_y, max_y (4 × f64 LE)
    /// - `[40..]`  Standard OGC WKB geometry
    fn geometry_to_gpkg_wkb(&self, geometry: &Geometry) -> Result<Vec<u8>> {
        use geo::algorithm::bounding_rect::BoundingRect;

        let mut buf: Vec<u8> = Vec::with_capacity(64);

        // Magic "GP"
        buf.extend_from_slice(&[0x47, 0x50]);

        // Version 0
        buf.push(0x00);

        // Flags: envelope_type=1 (XY) in bits [1..3], endianness=1 (LE) in bit 4
        let envelope_type: u8 = 1;
        let endian_flag: u8 = 1;
        let flags: u8 = (envelope_type << 1) | (endian_flag << 4);
        buf.push(flags);

        // SRS ID (4 bytes, i32 LE)
        let srs_id: i32 = geometry
            .crs
            .epsg_code()
            .map(|code| code as i32)
            .unwrap_or(4326);
        buf.extend_from_slice(&srs_id.to_le_bytes());

        // XY Envelope: min_x, max_x, min_y, max_y (4 × f64 LE = 32 bytes)
        if let Some(bbox) = geometry.geom.bounding_rect() {
            buf.extend_from_slice(&bbox.min().x.to_le_bytes());
            buf.extend_from_slice(&bbox.max().x.to_le_bytes());
            buf.extend_from_slice(&bbox.min().y.to_le_bytes());
            buf.extend_from_slice(&bbox.max().y.to_le_bytes());
        } else {
            buf.extend_from_slice(&[0u8; 32]);
        }

        // Standard OGC WKB (use EWKB writer which is already in the tree)
        let ewkb = crate::geometry::ewkb_parser::geometry_to_ewkb(geometry)?;
        buf.extend_from_slice(&ewkb);

        Ok(buf)
    }

    /// Convert GeoPackage binary WKB back to [`Geometry`].
    fn gpkg_wkb_to_geometry(&self, gpkg_wkb: &[u8]) -> Result<Geometry> {
        if gpkg_wkb.len() < 8 {
            return Err(GeoSparqlError::ParseError(
                "GeoPackage WKB too short (need ≥8 bytes for header)".to_string(),
            ));
        }

        // Validate magic
        if gpkg_wkb[0] != 0x47 || gpkg_wkb[1] != 0x50 {
            return Err(GeoSparqlError::ParseError(
                "Invalid GeoPackage magic number (expected 0x47 0x50 = 'GP')".to_string(),
            ));
        }

        // Decode envelope type from flags byte
        let flags = gpkg_wkb[3];
        let envelope_type = (flags >> 1) & 0x07;

        // Determine envelope byte length
        let envelope_size: usize = match envelope_type {
            0 => 0,  // No envelope
            1 => 32, // XY: 4 × f64
            2 => 48, // XYZ: 6 × f64
            3 => 48, // XYM: 6 × f64
            4 => 64, // XYZM: 8 × f64
            _ => {
                return Err(GeoSparqlError::ParseError(format!(
                    "Unknown GeoPackage envelope type {}",
                    envelope_type
                )));
            }
        };

        // Header = 8 bytes (magic 2 + version 1 + flags 1 + srs_id 4), then envelope
        let wkb_offset = 8 + envelope_size;
        if gpkg_wkb.len() < wkb_offset {
            return Err(GeoSparqlError::ParseError(
                "GeoPackage WKB truncated (header + envelope extends past buffer)".to_string(),
            ));
        }

        let wkb = &gpkg_wkb[wkb_offset..];
        crate::geometry::ewkb_parser::parse_ewkb(wkb)
    }
}

// ---------------------------------------------------------------------------
// Runtime constructor
// ---------------------------------------------------------------------------

/// Build a dedicated current-thread Tokio runtime for the sync↔async bridge.
#[cfg(feature = "geopackage")]
fn build_runtime() -> Result<Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!(
                "Failed to build Tokio runtime for GeoPackage: {}",
                e
            ))
        })
}

// ---------------------------------------------------------------------------
// Stub (feature disabled)
// ---------------------------------------------------------------------------

/// Non-functional stub when the `geopackage` feature is disabled.
#[cfg(not(feature = "geopackage"))]
pub struct GeoPackage;

#[cfg(not(feature = "geopackage"))]
impl GeoPackage {
    /// Always returns an error (feature is disabled).
    pub fn open<P: AsRef<Path>>(_path: P) -> Result<Self> {
        Err(GeoSparqlError::UnsupportedOperation(
            "GeoPackage support requires the 'geopackage' feature to be enabled".to_string(),
        ))
    }

    /// Always returns an error (feature is disabled).
    pub fn create_memory() -> Result<Self> {
        Err(GeoSparqlError::UnsupportedOperation(
            "GeoPackage support requires the 'geopackage' feature to be enabled".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "geopackage")]
mod tests {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, Point};

    // -----------------------------------------------------------------------
    // Basic lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_geopackage() {
        let gpkg = GeoPackage::create_memory();
        assert!(
            gpkg.is_ok(),
            "create_memory should succeed: {:?}",
            gpkg.err()
        );
    }

    #[test]
    fn test_create_feature_table() {
        let mut gpkg = GeoPackage::create_memory().expect("create_memory");
        let result = gpkg.create_feature_table("test_points", "geom", "POINT", 4326, false, false);
        assert!(result.is_ok(), "create_feature_table should succeed");

        let tables = gpkg.get_feature_tables().expect("get_feature_tables");
        assert!(
            tables.contains(&"test_points".to_string()),
            "table should be registered"
        );
    }

    #[test]
    fn test_insert_and_query_geometry() {
        let mut gpkg = GeoPackage::create_memory().expect("create_memory");
        gpkg.create_feature_table("points", "geom", "POINT", 4326, false, false)
            .expect("create_feature_table");

        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        let id = gpkg
            .insert_geometry("points", "geom", &point)
            .expect("insert_geometry");
        assert!(id > 0, "insert should return positive row-id");

        let geometries = gpkg
            .query_geometries("points", "geom")
            .expect("query_geometries");
        assert_eq!(geometries.len(), 1);

        if let GeoGeometry::Point(p) = &geometries[0].geom {
            assert!((p.x() - 1.0).abs() < 1e-6, "x coordinate should round-trip");
            assert!((p.y() - 2.0).abs() < 1e-6, "y coordinate should round-trip");
        } else {
            panic!("Expected Point geometry");
        }
    }

    #[test]
    fn test_multiple_geometries() {
        let mut gpkg = GeoPackage::create_memory().expect("create_memory");
        gpkg.create_feature_table("multi_points", "geom", "POINT", 4326, false, false)
            .expect("create_feature_table");

        for i in 0..5_i32 {
            let point = Geometry::new(GeoGeometry::Point(Point::new(i as f64, i as f64 * 2.0)));
            gpkg.insert_geometry("multi_points", "geom", &point)
                .expect("insert_geometry");
        }

        let geometries = gpkg
            .query_geometries("multi_points", "geom")
            .expect("query_geometries");
        assert_eq!(geometries.len(), 5);
    }

    #[test]
    #[ignore] // 3D WKB encoding needs enhancement in ewkb_parser
    fn test_3d_geometry() {
        let mut gpkg = GeoPackage::create_memory().expect("create_memory");
        gpkg.create_feature_table("points_3d", "geom", "POINT", 4326, true, false)
            .expect("create_feature_table");

        let point = Geometry::from_wkt("POINT Z(1 2 3)").expect("parse POINT Z");
        let id = gpkg
            .insert_geometry("points_3d", "geom", &point)
            .expect("insert_geometry");
        assert!(id > 0);

        let geometries = gpkg
            .query_geometries("points_3d", "geom")
            .expect("query_geometries");
        assert_eq!(geometries.len(), 1);
    }

    #[test]
    fn test_spatial_ref_sys_defaults() {
        let gpkg = GeoPackage::create_memory().expect("create_memory");

        let rows = gpkg
            .query("SELECT COUNT(*) FROM gpkg_spatial_ref_sys", &[])
            .expect("query gpkg_spatial_ref_sys count");

        let count: i64 = match rows.first().and_then(|r| r.get_by_index(0)) {
            Some(Value::I64(n)) => *n,
            Some(Value::Null) | None => 0,
            other => panic!("Unexpected count value: {:?}", other),
        };
        assert!(
            count >= 3,
            "should have at least 3 default SRS entries, got {}",
            count
        );
    }

    // -----------------------------------------------------------------------
    // WAL checkpoint + byte-level GPKG magic verification
    // -----------------------------------------------------------------------

    /// Write a GeoPackage to a temp file, call `checkpoint()`, then verify that
    /// bytes [68..72] contain the application-id magic 0x47503130 ("GP10").
    ///
    /// Per the SQLite on-disk format spec, `application_id` is stored as a
    /// big-endian 32-bit integer at offset 68 of the database header.
    #[test]
    fn test_checkpoint_writes_application_id_to_main_file() {
        let tmp_dir = std::env::temp_dir();
        let db_path = tmp_dir.join("oxirs_gpkg_magic_test.gpkg");

        // Clean up any leftover files from a prior run.
        let _ = std::fs::remove_file(&db_path);
        let wal_path = tmp_dir.join("oxirs_gpkg_magic_test.gpkg-wal");
        let _ = std::fs::remove_file(&wal_path);

        {
            let mut gpkg = GeoPackage::open(&db_path).expect("open file-backed GeoPackage");
            gpkg.create_feature_table("test_layer", "geom", "POINT", 4326, false, false)
                .expect("create_feature_table");

            let point = Geometry::new(GeoGeometry::Point(Point::new(10.0, 20.0)));
            gpkg.insert_geometry("test_layer", "geom", &point)
                .expect("insert_geometry");

            // Flush WAL → main file
            gpkg.checkpoint().expect("checkpoint");
        } // GeoPackage dropped here

        // Read the main file and check bytes [68..72] for the GPKG application-id magic.
        let file_bytes = std::fs::read(&db_path).expect("read GeoPackage file after checkpoint");

        assert!(
            file_bytes.len() >= 72,
            "GeoPackage file should be ≥72 bytes, got {}",
            file_bytes.len()
        );

        // application_id is at offset 68, stored big-endian (SQLite header format).
        let app_id_bytes: [u8; 4] = file_bytes[68..72].try_into().expect("slice to array");
        let app_id = u32::from_be_bytes(app_id_bytes);

        assert_eq!(
            app_id, 0x4750_3130,
            "application_id at offset 68 should be 0x47503130 ('GP10'), got 0x{:08X}",
            app_id
        );

        // Reopen and read back the geometry to confirm the file is valid.
        let gpkg = GeoPackage::open(&db_path).expect("reopen after checkpoint");
        let geoms = gpkg
            .query_geometries("test_layer", "geom")
            .expect("query after reopen");
        assert_eq!(geoms.len(), 1, "should read back 1 geometry after reopen");

        if let GeoGeometry::Point(p) = &geoms[0].geom {
            assert!((p.x() - 10.0).abs() < 1e-6);
            assert!((p.y() - 20.0).abs() < 1e-6);
        } else {
            panic!("Expected Point geometry after reopen");
        }

        // Cleanup
        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(&wal_path);
    }
}
