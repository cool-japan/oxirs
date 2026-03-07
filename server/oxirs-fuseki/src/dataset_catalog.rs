//! Dataset catalog/registry for managing SPARQL dataset metadata.
//!
//! Provides a central registry for dataset metadata, supporting
//! registration, search, and JSON serialisation.

use std::collections::HashMap;

/// Access control mode for a dataset.
#[derive(Debug, Clone, PartialEq)]
pub enum AccessMode {
    ReadOnly,
    ReadWrite,
    ReadWriteUpdate,
}

impl AccessMode {
    fn as_str(&self) -> &'static str {
        match self {
            AccessMode::ReadOnly => "ReadOnly",
            AccessMode::ReadWrite => "ReadWrite",
            AccessMode::ReadWriteUpdate => "ReadWriteUpdate",
        }
    }

    fn parse_from_str(s: &str) -> Option<Self> {
        match s {
            "ReadOnly" => Some(AccessMode::ReadOnly),
            "ReadWrite" => Some(AccessMode::ReadWrite),
            "ReadWriteUpdate" => Some(AccessMode::ReadWriteUpdate),
            _ => None,
        }
    }
}

/// Metadata associated with a registered dataset.
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    pub name: String,
    pub description: Option<String>,
    pub created_at: u64,
    pub triples_count: u64,
    pub graphs: Vec<String>,
    pub tags: Vec<String>,
    pub access_mode: AccessMode,
}

impl DatasetMetadata {
    /// Create a new dataset metadata entry.
    pub fn new(name: impl Into<String>) -> Self {
        DatasetMetadata {
            name: name.into(),
            description: None,
            created_at: 0,
            triples_count: 0,
            graphs: Vec::new(),
            tags: Vec::new(),
            access_mode: AccessMode::ReadWrite,
        }
    }
}

/// Errors that can occur during catalog operations.
#[derive(Debug)]
pub enum CatalogError {
    AlreadyExists(String),
    NotFound(String),
    InvalidJson(String),
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogError::AlreadyExists(name) => write!(f, "Dataset already exists: {name}"),
            CatalogError::NotFound(name) => write!(f, "Dataset not found: {name}"),
            CatalogError::InvalidJson(msg) => write!(f, "Invalid JSON: {msg}"),
        }
    }
}

impl std::error::Error for CatalogError {}

/// A registry of dataset metadata entries.
#[derive(Debug)]
pub struct DatasetCatalog {
    entries: HashMap<String, DatasetMetadata>,
}

impl DatasetCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        DatasetCatalog {
            entries: HashMap::new(),
        }
    }

    /// Register a new dataset. Returns `AlreadyExists` if the name is taken.
    pub fn register(&mut self, meta: DatasetMetadata) -> Result<(), CatalogError> {
        if self.entries.contains_key(&meta.name) {
            return Err(CatalogError::AlreadyExists(meta.name.clone()));
        }
        self.entries.insert(meta.name.clone(), meta);
        Ok(())
    }

    /// Remove a dataset by name. Returns the metadata or `NotFound`.
    pub fn unregister(&mut self, name: &str) -> Result<DatasetMetadata, CatalogError> {
        self.entries
            .remove(name)
            .ok_or_else(|| CatalogError::NotFound(name.to_string()))
    }

    /// Retrieve a dataset by name.
    pub fn get(&self, name: &str) -> Option<&DatasetMetadata> {
        self.entries.get(name)
    }

    /// List all datasets sorted by name.
    pub fn list(&self) -> Vec<&DatasetMetadata> {
        let mut v: Vec<&DatasetMetadata> = self.entries.values().collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    /// Search datasets by name, description, or tags (case-insensitive substring).
    pub fn search(&self, query: &str) -> Vec<&DatasetMetadata> {
        let q = query.to_lowercase();
        let mut results: Vec<&DatasetMetadata> = self
            .entries
            .values()
            .filter(|m| {
                m.name.to_lowercase().contains(&q)
                    || m.description
                        .as_deref()
                        .unwrap_or("")
                        .to_lowercase()
                        .contains(&q)
                    || m.tags.iter().any(|t| t.to_lowercase().contains(&q))
            })
            .collect();
        results.sort_by(|a, b| a.name.cmp(&b.name));
        results
    }

    /// Update the triple count for a dataset.
    pub fn update_count(&mut self, name: &str, count: u64) -> Result<(), CatalogError> {
        self.entries
            .get_mut(name)
            .ok_or_else(|| CatalogError::NotFound(name.to_string()))
            .map(|m| m.triples_count = count)
    }

    /// Serialize the catalog to a JSON string.
    pub fn to_json(&self) -> String {
        let mut items: Vec<String> = self
            .entries
            .values()
            .map(|m| {
                let desc = match &m.description {
                    Some(d) => format!("\"{}\"", escape_json(d)),
                    None => "null".to_string(),
                };
                let graphs = m
                    .graphs
                    .iter()
                    .map(|g| format!("\"{}\"", escape_json(g)))
                    .collect::<Vec<_>>()
                    .join(",");
                let tags = m
                    .tags
                    .iter()
                    .map(|t| format!("\"{}\"", escape_json(t)))
                    .collect::<Vec<_>>()
                    .join(",");
                format!(
                    "{{\"name\":\"{}\",\"description\":{},\"created_at\":{},\"triples_count\":{},\"graphs\":[{}],\"tags\":[{}],\"access_mode\":\"{}\"}}",
                    escape_json(&m.name),
                    desc,
                    m.created_at,
                    m.triples_count,
                    graphs,
                    tags,
                    m.access_mode.as_str()
                )
            })
            .collect();
        items.sort();
        format!("{{\"datasets\":[{}]}}", items.join(","))
    }

    /// Deserialize a catalog from a JSON string produced by `to_json`.
    pub fn from_json(json: &str) -> Result<Self, CatalogError> {
        let mut catalog = DatasetCatalog::new();
        // Simple hand-rolled JSON parser for the structure we produce
        let inner = json
            .trim()
            .strip_prefix('{')
            .and_then(|s| s.strip_suffix('}'))
            .ok_or_else(|| CatalogError::InvalidJson("Expected outer object".to_string()))?;

        // Find the datasets array
        let arr_start = inner
            .find('[')
            .ok_or_else(|| CatalogError::InvalidJson("Missing datasets array".to_string()))?;
        let arr_end = inner
            .rfind(']')
            .ok_or_else(|| CatalogError::InvalidJson("Missing closing bracket".to_string()))?;
        let array_content = &inner[arr_start + 1..arr_end];

        if array_content.trim().is_empty() {
            return Ok(catalog);
        }

        // Split individual dataset objects
        for obj_str in split_json_objects(array_content) {
            let meta = parse_dataset_object(&obj_str)?;
            catalog.entries.insert(meta.name.clone(), meta);
        }

        Ok(catalog)
    }
}

impl Default for DatasetCatalog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal JSON helpers
// ---------------------------------------------------------------------------

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Split a comma-separated sequence of JSON objects `{...},{...}` at the
/// top level (not inside nested braces/strings).
fn split_json_objects(s: &str) -> Vec<String> {
    let mut objects = Vec::new();
    let mut depth: i32 = 0;
    let mut start = 0;
    let chars: Vec<char> = s.chars().collect();
    let mut in_string = false;
    let mut escape_next = false;

    for (i, &ch) in chars.iter().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch == '{' {
            if depth == 0 {
                start = i;
            }
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
            if depth == 0 {
                objects.push(chars[start..=i].iter().collect());
            }
        }
    }
    objects
}

fn parse_dataset_object(s: &str) -> Result<DatasetMetadata, CatalogError> {
    let name = extract_string_field(s, "name")
        .ok_or_else(|| CatalogError::InvalidJson("Missing 'name'".to_string()))?;
    let description = extract_optional_string_field(s, "description");
    let created_at = extract_u64_field(s, "created_at").unwrap_or(0);
    let triples_count = extract_u64_field(s, "triples_count").unwrap_or(0);
    let graphs = extract_string_array(s, "graphs");
    let tags = extract_string_array(s, "tags");
    let access_mode_str = extract_string_field(s, "access_mode").unwrap_or_default();
    let access_mode = AccessMode::parse_from_str(&access_mode_str).ok_or_else(|| {
        CatalogError::InvalidJson(format!("Unknown access_mode: {access_mode_str}"))
    })?;

    Ok(DatasetMetadata {
        name,
        description,
        created_at,
        triples_count,
        graphs,
        tags,
        access_mode,
    })
}

fn extract_string_field(s: &str, field: &str) -> Option<String> {
    let key = format!("\"{}\":", field);
    let pos = s.find(&key)?;
    let rest = &s[pos + key.len()..].trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        parse_json_string(stripped)
    } else {
        None
    }
}

fn extract_optional_string_field(s: &str, field: &str) -> Option<String> {
    let key = format!("\"{}\":", field);
    let pos = s.find(&key)?;
    let rest = s[pos + key.len()..].trim_start();
    if rest.starts_with("null") {
        None
    } else if let Some(stripped) = rest.strip_prefix('"') {
        parse_json_string(stripped)
    } else {
        None
    }
}

fn parse_json_string(s: &str) -> Option<String> {
    let mut result = String::new();
    let mut chars = s.chars();
    let mut escape_next = false;
    for ch in chars.by_ref() {
        if escape_next {
            let unescaped = match ch {
                '"' => '"',
                '\\' => '\\',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other => other,
            };
            result.push(unescaped);
            escape_next = false;
        } else if ch == '\\' {
            escape_next = true;
        } else if ch == '"' {
            return Some(result);
        } else {
            result.push(ch);
        }
    }
    None
}

fn extract_u64_field(s: &str, field: &str) -> Option<u64> {
    let key = format!("\"{}\":", field);
    let pos = s.find(&key)?;
    let rest = s[pos + key.len()..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn extract_string_array(s: &str, field: &str) -> Vec<String> {
    let key = format!("\"{}\":[", field);
    let pos = match s.find(&key) {
        Some(p) => p + key.len(),
        None => return Vec::new(),
    };
    let rest = &s[pos..];
    let end = rest.find(']').unwrap_or(rest.len());
    let inner = &rest[..end];
    if inner.trim().is_empty() {
        return Vec::new();
    }
    // Split by quoted strings
    let mut results = Vec::new();
    let mut remaining = inner;
    while let Some(start) = remaining.find('"') {
        remaining = &remaining[start + 1..];
        if let Some(value) = parse_json_string(remaining) {
            let skip = value.len() + 1; // +1 for closing quote
            results.push(value);
            if skip < remaining.len() {
                remaining = &remaining[skip..];
            } else {
                break;
            }
        } else {
            break;
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_meta(name: &str) -> DatasetMetadata {
        DatasetMetadata {
            name: name.to_string(),
            description: Some(format!("Description of {name}")),
            created_at: 1_700_000_000,
            triples_count: 1000,
            graphs: vec!["http://example.org/graph1".to_string()],
            tags: vec!["test".to_string(), "rdf".to_string()],
            access_mode: AccessMode::ReadWrite,
        }
    }

    #[test]
    fn test_catalog_new_empty() {
        let cat = DatasetCatalog::new();
        assert!(cat.list().is_empty());
    }

    #[test]
    fn test_register_and_get() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("ds1")).unwrap();
        let m = cat.get("ds1").unwrap();
        assert_eq!(m.name, "ds1");
    }

    #[test]
    fn test_register_duplicate_error() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("ds1")).unwrap();
        let err = cat.register(make_meta("ds1")).unwrap_err();
        assert!(matches!(err, CatalogError::AlreadyExists(_)));
    }

    #[test]
    fn test_unregister_existing() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("ds1")).unwrap();
        let m = cat.unregister("ds1").unwrap();
        assert_eq!(m.name, "ds1");
        assert!(cat.get("ds1").is_none());
    }

    #[test]
    fn test_unregister_not_found() {
        let mut cat = DatasetCatalog::new();
        let err = cat.unregister("missing").unwrap_err();
        assert!(matches!(err, CatalogError::NotFound(_)));
    }

    #[test]
    fn test_list_sorted() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("zebra")).unwrap();
        cat.register(make_meta("alpha")).unwrap();
        cat.register(make_meta("mango")).unwrap();
        let list = cat.list();
        assert_eq!(list[0].name, "alpha");
        assert_eq!(list[1].name, "mango");
        assert_eq!(list[2].name, "zebra");
    }

    #[test]
    fn test_search_by_name() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("sensor-data")).unwrap();
        cat.register(make_meta("graph-store")).unwrap();
        let results = cat.search("sensor");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "sensor-data");
    }

    #[test]
    fn test_search_by_description() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("ds1")).unwrap();
        let results = cat.search("Description of ds1");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_by_tag() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("ds1")).unwrap();
        let results = cat.search("rdf");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_case_insensitive() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("MyDataset")).unwrap();
        let results = cat.search("mydataset");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_no_match() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("ds1")).unwrap();
        let results = cat.search("nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_multiple_matches_sorted() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("z-rdf-store")).unwrap();
        cat.register(make_meta("a-rdf-data")).unwrap();
        let results = cat.search("rdf");
        assert_eq!(results[0].name, "a-rdf-data");
        assert_eq!(results[1].name, "z-rdf-store");
    }

    #[test]
    fn test_update_count() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("ds1")).unwrap();
        cat.update_count("ds1", 9999).unwrap();
        assert_eq!(cat.get("ds1").unwrap().triples_count, 9999);
    }

    #[test]
    fn test_update_count_not_found() {
        let mut cat = DatasetCatalog::new();
        let err = cat.update_count("missing", 100).unwrap_err();
        assert!(matches!(err, CatalogError::NotFound(_)));
    }

    #[test]
    fn test_to_json_empty() {
        let cat = DatasetCatalog::new();
        let json = cat.to_json();
        assert!(json.contains("\"datasets\":[]"));
    }

    #[test]
    fn test_json_roundtrip_empty() {
        let cat = DatasetCatalog::new();
        let json = cat.to_json();
        let cat2 = DatasetCatalog::from_json(&json).unwrap();
        assert!(cat2.list().is_empty());
    }

    #[test]
    fn test_json_roundtrip_single() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("ds1")).unwrap();
        let json = cat.to_json();
        let cat2 = DatasetCatalog::from_json(&json).unwrap();
        let m = cat2.get("ds1").unwrap();
        assert_eq!(m.name, "ds1");
        assert_eq!(m.triples_count, 1000);
    }

    #[test]
    fn test_json_roundtrip_multiple() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("alpha")).unwrap();
        cat.register(make_meta("beta")).unwrap();
        cat.register(make_meta("gamma")).unwrap();
        let json = cat.to_json();
        let cat2 = DatasetCatalog::from_json(&json).unwrap();
        assert_eq!(cat2.list().len(), 3);
    }

    #[test]
    fn test_json_roundtrip_description_none() {
        let mut cat = DatasetCatalog::new();
        let mut m = make_meta("ds1");
        m.description = None;
        cat.register(m).unwrap();
        let json = cat.to_json();
        let cat2 = DatasetCatalog::from_json(&json).unwrap();
        assert!(cat2.get("ds1").unwrap().description.is_none());
    }

    #[test]
    fn test_json_roundtrip_access_modes() {
        for mode in [
            AccessMode::ReadOnly,
            AccessMode::ReadWrite,
            AccessMode::ReadWriteUpdate,
        ] {
            let mut cat = DatasetCatalog::new();
            let mut m = make_meta("ds1");
            m.access_mode = mode.clone();
            cat.register(m).unwrap();
            let json = cat.to_json();
            let cat2 = DatasetCatalog::from_json(&json).unwrap();
            assert_eq!(cat2.get("ds1").unwrap().access_mode, mode);
        }
    }

    #[test]
    fn test_from_json_invalid() {
        let err = DatasetCatalog::from_json("not json").unwrap_err();
        assert!(matches!(err, CatalogError::InvalidJson(_)));
    }

    #[test]
    fn test_get_missing_returns_none() {
        let cat = DatasetCatalog::new();
        assert!(cat.get("nonexistent").is_none());
    }

    #[test]
    fn test_catalog_error_display() {
        let e1 = CatalogError::AlreadyExists("ds".to_string());
        let e2 = CatalogError::NotFound("ds".to_string());
        let e3 = CatalogError::InvalidJson("bad".to_string());
        assert!(e1.to_string().contains("already exists"));
        assert!(e2.to_string().contains("not found"));
        assert!(e3.to_string().contains("Invalid JSON"));
    }

    #[test]
    fn test_metadata_new() {
        let m = DatasetMetadata::new("test");
        assert_eq!(m.name, "test");
        assert!(m.description.is_none());
        assert_eq!(m.access_mode, AccessMode::ReadWrite);
    }

    #[test]
    fn test_json_roundtrip_tags_and_graphs() {
        let mut cat = DatasetCatalog::new();
        let mut m = make_meta("ds1");
        m.tags = vec!["iot".to_string(), "sensor".to_string()];
        m.graphs = vec!["http://g1".to_string(), "http://g2".to_string()];
        cat.register(m).unwrap();
        let json = cat.to_json();
        let cat2 = DatasetCatalog::from_json(&json).unwrap();
        let recovered = cat2.get("ds1").unwrap();
        assert_eq!(recovered.tags.len(), 2);
        assert_eq!(recovered.graphs.len(), 2);
    }

    #[test]
    fn test_default_trait() {
        let cat = DatasetCatalog::default();
        assert!(cat.list().is_empty());
    }

    #[test]
    fn test_unregister_then_reregister() {
        let mut cat = DatasetCatalog::new();
        cat.register(make_meta("ds1")).unwrap();
        cat.unregister("ds1").unwrap();
        cat.register(make_meta("ds1")).unwrap();
        assert!(cat.get("ds1").is_some());
    }

    #[test]
    fn test_list_count() {
        let mut cat = DatasetCatalog::new();
        for i in 0..5 {
            cat.register(make_meta(&format!("ds{i}"))).unwrap();
        }
        assert_eq!(cat.list().len(), 5);
    }
}
