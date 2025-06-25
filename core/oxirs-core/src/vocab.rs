//! Common RDF vocabularies and namespaces

use crate::model::NamedNode;
use std::sync::LazyLock;

/// RDF vocabulary namespace
pub mod rdf {
    use super::*;

    /// The RDF namespace IRI
    pub const NAMESPACE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

    /// rdf:type predicate
    pub static TYPE: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}type", NAMESPACE)));

    /// rdf:Property class
    pub static PROPERTY: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}Property", NAMESPACE)));

    /// rdf:Resource class
    pub static RESOURCE: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}Resource", NAMESPACE)));

    /// rdf:Statement class
    pub static STATEMENT: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}Statement", NAMESPACE)));

    /// rdf:subject predicate
    pub static SUBJECT: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}subject", NAMESPACE)));

    /// rdf:predicate predicate
    pub static PREDICATE: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}predicate", NAMESPACE)));

    /// rdf:object predicate
    pub static OBJECT: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}object", NAMESPACE)));

    /// rdf:List class
    pub static LIST: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}List", NAMESPACE)));

    /// rdf:first predicate
    pub static FIRST: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}first", NAMESPACE)));

    /// rdf:rest predicate
    pub static REST: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}rest", NAMESPACE)));

    /// rdf:nil resource
    pub static NIL: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}nil", NAMESPACE)));

    /// rdf:langString datatype
    pub static LANG_STRING: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}langString", NAMESPACE)));

    /// rdf:dirLangString datatype (RDF 1.2)
    #[cfg(feature = "rdf-12")]
    pub static DIR_LANG_STRING: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}dirLangString", NAMESPACE)));
}

/// XML Schema datatypes vocabulary namespace
pub mod xsd {
    use super::*;

    /// The XSD namespace IRI
    pub const NAMESPACE: &str = "http://www.w3.org/2001/XMLSchema#";

    /// xsd:string datatype
    pub static STRING: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}string", NAMESPACE)));

    /// xsd:boolean datatype
    pub static BOOLEAN: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}boolean", NAMESPACE)));

    /// xsd:integer datatype
    pub static INTEGER: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}integer", NAMESPACE)));

    /// xsd:decimal datatype
    pub static DECIMAL: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}decimal", NAMESPACE)));

    /// xsd:double datatype
    pub static DOUBLE: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}double", NAMESPACE)));

    /// xsd:float datatype
    pub static FLOAT: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}float", NAMESPACE)));

    /// xsd:date datatype
    pub static DATE: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}date", NAMESPACE)));

    /// xsd:time datatype
    pub static TIME: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}time", NAMESPACE)));

    /// xsd:dateTime datatype
    pub static DATE_TIME: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}dateTime", NAMESPACE)));

    /// xsd:duration datatype
    pub static DURATION: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}duration", NAMESPACE)));

    /// xsd:base64Binary datatype
    pub static BASE64_BINARY: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}base64Binary", NAMESPACE)));

    /// xsd:hexBinary datatype
    pub static HEX_BINARY: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}hexBinary", NAMESPACE)));

    /// xsd:anyURI datatype
    pub static ANY_URI: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}anyURI", NAMESPACE)));

    /// xsd:language datatype
    pub static LANGUAGE: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}language", NAMESPACE)));

    /// xsd:langString datatype (actually in RDF namespace)
    pub static LANG_STRING: LazyLock<NamedNode> = LazyLock::new(|| {
        NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#langString")
    });

    /// xsd:long datatype
    pub static LONG: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}long", NAMESPACE)));

    /// xsd:int datatype
    pub static INT: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}int", NAMESPACE)));

    /// xsd:short datatype
    pub static SHORT: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}short", NAMESPACE)));

    /// xsd:byte datatype
    pub static BYTE: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}byte", NAMESPACE)));

    /// xsd:unsignedLong datatype
    pub static UNSIGNED_LONG: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}unsignedLong", NAMESPACE)));

    /// xsd:unsignedInt datatype
    pub static UNSIGNED_INT: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}unsignedInt", NAMESPACE)));

    /// xsd:unsignedShort datatype
    pub static UNSIGNED_SHORT: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}unsignedShort", NAMESPACE)));

    /// xsd:unsignedByte datatype
    pub static UNSIGNED_BYTE: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}unsignedByte", NAMESPACE)));

    /// xsd:positiveInteger datatype
    pub static POSITIVE_INTEGER: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}positiveInteger", NAMESPACE)));

    /// xsd:nonNegativeInteger datatype
    pub static NON_NEGATIVE_INTEGER: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}nonNegativeInteger", NAMESPACE)));

    /// xsd:negativeInteger datatype
    pub static NEGATIVE_INTEGER: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}negativeInteger", NAMESPACE)));

    /// xsd:nonPositiveInteger datatype
    pub static NON_POSITIVE_INTEGER: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}nonPositiveInteger", NAMESPACE)));

    /// xsd:normalizedString datatype
    pub static NORMALIZED_STRING: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}normalizedString", NAMESPACE)));

    /// xsd:token datatype
    pub static TOKEN: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}token", NAMESPACE)));

    /// xsd:yearMonthDuration datatype
    pub static YEAR_MONTH_DURATION: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}yearMonthDuration", NAMESPACE)));

    /// xsd:dayTimeDuration datatype
    pub static DAY_TIME_DURATION: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}dayTimeDuration", NAMESPACE)));

    /// xsd:gYear datatype
    pub static G_YEAR: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}gYear", NAMESPACE)));

    /// xsd:gYearMonth datatype
    pub static G_YEAR_MONTH: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}gYearMonth", NAMESPACE)));

    /// xsd:gMonth datatype
    pub static G_MONTH: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}gMonth", NAMESPACE)));

    /// xsd:gMonthDay datatype
    pub static G_MONTH_DAY: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}gMonthDay", NAMESPACE)));

    /// xsd:gDay datatype
    pub static G_DAY: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}gDay", NAMESPACE)));
}

/// RDFS vocabulary namespace
pub mod rdfs {
    use super::*;

    /// The RDFS namespace IRI
    pub const NAMESPACE: &str = "http://www.w3.org/2000/01/rdf-schema#";

    /// rdfs:Class class
    pub static CLASS: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}Class", NAMESPACE)));

    /// rdfs:subClassOf predicate
    pub static SUB_CLASS_OF: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}subClassOf", NAMESPACE)));

    /// rdfs:subPropertyOf predicate
    pub static SUB_PROPERTY_OF: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}subPropertyOf", NAMESPACE)));

    /// rdfs:domain predicate
    pub static DOMAIN: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}domain", NAMESPACE)));

    /// rdfs:range predicate
    pub static RANGE: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}range", NAMESPACE)));

    /// rdfs:label predicate
    pub static LABEL: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}label", NAMESPACE)));

    /// rdfs:comment predicate
    pub static COMMENT: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}comment", NAMESPACE)));
}

/// OWL vocabulary namespace
pub mod owl {
    use super::*;

    /// The OWL namespace IRI
    pub const NAMESPACE: &str = "http://www.w3.org/2002/07/owl#";

    /// owl:Class class
    pub static CLASS: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}Class", NAMESPACE)));

    /// owl:ObjectProperty class
    pub static OBJECT_PROPERTY: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}ObjectProperty", NAMESPACE)));

    /// owl:DatatypeProperty class
    pub static DATATYPE_PROPERTY: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}DatatypeProperty", NAMESPACE)));

    /// owl:FunctionalProperty class
    pub static FUNCTIONAL_PROPERTY: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}FunctionalProperty", NAMESPACE)));

    /// owl:InverseFunctionalProperty class
    pub static INVERSE_FUNCTIONAL_PROPERTY: LazyLock<NamedNode> = LazyLock::new(|| {
        NamedNode::new_unchecked(format!("{}InverseFunctionalProperty", NAMESPACE))
    });

    /// owl:sameAs predicate
    pub static SAME_AS: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}sameAs", NAMESPACE)));

    /// owl:differentFrom predicate
    pub static DIFFERENT_FROM: LazyLock<NamedNode> =
        LazyLock::new(|| NamedNode::new_unchecked(format!("{}differentFrom", NAMESPACE)));
}
