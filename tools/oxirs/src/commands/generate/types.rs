//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[derive(Debug, Clone)]
pub struct PropertyConstraint {
    pub path: String,
    pub min_count: Option<u32>,
    pub max_count: Option<u32>,
    pub datatype: Option<String>,
    pub pattern: Option<String>,
    pub min_length: Option<u32>,
    pub max_length: Option<u32>,
    pub min_inclusive: Option<String>,
    pub max_inclusive: Option<String>,
    pub node_kind: Option<String>,
    pub class: Option<String>,
}
/// Dataset type
#[derive(Debug, Clone, Copy)]
pub enum DatasetType {
    Rdf,
    Graph,
    Semantic,
    Bibliographic,
    Geographic,
    Organizational,
}
impl DatasetType {
    pub fn from_string(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "rdf" => Ok(DatasetType::Rdf),
            "graph" => Ok(DatasetType::Graph),
            "semantic" => Ok(DatasetType::Semantic),
            "bibliographic" | "bib" | "biblio" => Ok(DatasetType::Bibliographic),
            "geographic" | "geo" => Ok(DatasetType::Geographic),
            "organizational" | "org" => Ok(DatasetType::Organizational),
            _ => {
                Err(
                    format!(
                        "Invalid dataset type: {}. Use rdf/graph/semantic/bibliographic/geographic/organizational",
                        s
                    ),
                )
            }
        }
    }
}
/// OWL ontology representation for data generation
#[derive(Debug, Clone)]
pub struct OwlOntology {
    pub classes: Vec<OwlClass>,
    pub properties: Vec<OwlProperty>,
}
#[derive(Debug, Clone)]
pub struct OwlProperty {
    pub uri: String,
    pub _label: Option<String>,
    pub _comment: Option<String>,
    pub property_type: OwlPropertyType,
    pub domain: Vec<String>,
    pub range: Vec<String>,
    pub _super_properties: Vec<String>,
    pub is_functional: bool,
    pub is_inverse_functional: bool,
    pub _is_transitive: bool,
    pub is_symmetric: bool,
}
#[derive(Debug, Clone, PartialEq)]
pub enum OwlPropertyType {
    Object,
    Datatype,
}
#[derive(Debug, Clone)]
pub struct OwlRestriction {
    pub on_property: String,
    pub restriction_type: OwlRestrictionType,
}
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum OwlRestrictionType {
    SomeValuesFrom(String),
    AllValuesFrom(String),
    MinCardinality(u32),
    MaxCardinality(u32),
    ExactCardinality(u32),
    HasValue(String),
}
/// RDFS schema representation for data generation
#[derive(Debug, Clone)]
pub struct RdfsSchema {
    pub classes: Vec<RdfsClass>,
    pub properties: Vec<RdfsProperty>,
}
#[derive(Debug, Clone)]
pub struct RdfsProperty {
    pub uri: String,
    pub _label: Option<String>,
    pub _comment: Option<String>,
    pub domain: Vec<String>,
    pub range: Vec<String>,
    pub _super_properties: Vec<String>,
}
#[derive(Debug, Clone)]
pub struct OwlClass {
    pub uri: String,
    pub _label: Option<String>,
    pub _comment: Option<String>,
    pub _super_classes: Vec<String>,
    pub _equivalent_classes: Vec<String>,
    pub _disjoint_with: Vec<String>,
    pub restrictions: Vec<OwlRestriction>,
}
/// Simplified SHACL shape representation for data generation
#[derive(Debug, Clone)]
pub struct ShaclShape {
    pub target_class: Option<String>,
    pub properties: Vec<PropertyConstraint>,
}
#[derive(Debug, Clone)]
pub struct RdfsClass {
    pub uri: String,
    pub _label: Option<String>,
    pub _comment: Option<String>,
    pub _super_classes: Vec<String>,
}
/// Dataset size presets
#[derive(Debug, Clone, Copy)]
pub enum DatasetSize {
    Tiny,
    Small,
    Medium,
    Large,
    XLarge,
    Custom(usize),
}
impl DatasetSize {
    pub fn triple_count(&self) -> usize {
        match self {
            DatasetSize::Tiny => 100,
            DatasetSize::Small => 1_000,
            DatasetSize::Medium => 10_000,
            DatasetSize::Large => 100_000,
            DatasetSize::XLarge => 1_000_000,
            DatasetSize::Custom(n) => *n,
        }
    }
    pub fn from_string(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "tiny" => Ok(DatasetSize::Tiny),
            "small" => Ok(DatasetSize::Small),
            "medium" => Ok(DatasetSize::Medium),
            "large" => Ok(DatasetSize::Large),
            "xlarge" => Ok(DatasetSize::XLarge),
            _ => {
                if let Ok(n) = s.parse::<usize>() {
                    Ok(DatasetSize::Custom(n))
                } else {
                    Err(format!(
                        "Invalid dataset size: {}. Use tiny/small/medium/large/xlarge or a number",
                        s
                    ))
                }
            }
        }
    }
}
