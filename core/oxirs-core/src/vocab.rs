//! Common RDF vocabularies and namespaces

use crate::model::NamedNode;
use once_cell::sync::Lazy;

/// RDF vocabulary namespace
pub mod rdf {
    use super::*;

    /// The RDF namespace IRI
    pub const NAMESPACE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

    /// rdf:type predicate
    pub static TYPE: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}type")));

    /// rdf:Property class
    pub static PROPERTY: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}Property")));

    /// rdf:Resource class
    pub static RESOURCE: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}Resource")));

    /// rdf:Statement class
    pub static STATEMENT: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}Statement")));

    /// rdf:subject predicate
    pub static SUBJECT: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}subject")));

    /// rdf:predicate predicate
    pub static PREDICATE: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}predicate")));

    /// rdf:object predicate
    pub static OBJECT: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}object")));

    /// rdf:List class
    pub static LIST: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}List")));

    /// rdf:first predicate
    pub static FIRST: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}first")));

    /// rdf:rest predicate
    pub static REST: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}rest")));

    /// rdf:nil resource
    pub static NIL: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}nil")));

    /// rdf:langString datatype
    pub static LANG_STRING: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}langString")));

    /// rdf:dirLangString datatype (RDF 1.2)
    #[cfg(feature = "rdf-12")]
    pub static DIR_LANG_STRING: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}dirLangString")));
}

/// XML Schema datatypes vocabulary namespace
pub mod xsd {
    use super::*;

    /// The XSD namespace IRI
    pub const NAMESPACE: &str = "http://www.w3.org/2001/XMLSchema#";

    /// xsd:string datatype
    pub static STRING: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}string")));

    /// xsd:boolean datatype
    pub static BOOLEAN: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}boolean")));

    /// xsd:integer datatype
    pub static INTEGER: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}integer")));

    /// xsd:decimal datatype
    pub static DECIMAL: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}decimal")));

    /// xsd:double datatype
    pub static DOUBLE: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}double")));

    /// xsd:float datatype
    pub static FLOAT: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}float")));

    /// xsd:date datatype
    pub static DATE: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}date")));

    /// xsd:time datatype
    pub static TIME: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}time")));

    /// xsd:dateTime datatype
    pub static DATE_TIME: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}dateTime")));

    /// xsd:duration datatype
    pub static DURATION: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}duration")));

    /// xsd:base64Binary datatype
    pub static BASE64_BINARY: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}base64Binary")));

    /// xsd:hexBinary datatype
    pub static HEX_BINARY: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}hexBinary")));

    /// xsd:anyURI datatype
    pub static ANY_URI: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}anyURI")));

    /// xsd:language datatype
    pub static LANGUAGE: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}language")));

    /// xsd:langString datatype (actually in RDF namespace)
    pub static LANG_STRING: Lazy<NamedNode> = Lazy::new(|| {
        NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#langString")
    });

    /// xsd:long datatype
    pub static LONG: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}long")));

    /// xsd:int datatype
    pub static INT: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}int")));

    /// xsd:short datatype
    pub static SHORT: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}short")));

    /// xsd:byte datatype
    pub static BYTE: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}byte")));

    /// xsd:unsignedLong datatype
    pub static UNSIGNED_LONG: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}unsignedLong")));

    /// xsd:unsignedInt datatype
    pub static UNSIGNED_INT: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}unsignedInt")));

    /// xsd:unsignedShort datatype
    pub static UNSIGNED_SHORT: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}unsignedShort")));

    /// xsd:unsignedByte datatype
    pub static UNSIGNED_BYTE: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}unsignedByte")));

    /// xsd:positiveInteger datatype
    pub static POSITIVE_INTEGER: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}positiveInteger")));

    /// xsd:nonNegativeInteger datatype
    pub static NON_NEGATIVE_INTEGER: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}nonNegativeInteger")));

    /// xsd:negativeInteger datatype
    pub static NEGATIVE_INTEGER: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}negativeInteger")));

    /// xsd:nonPositiveInteger datatype
    pub static NON_POSITIVE_INTEGER: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}nonPositiveInteger")));

    /// xsd:normalizedString datatype
    pub static NORMALIZED_STRING: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}normalizedString")));

    /// xsd:token datatype
    pub static TOKEN: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}token")));

    /// xsd:yearMonthDuration datatype
    pub static YEAR_MONTH_DURATION: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}yearMonthDuration")));

    /// xsd:dayTimeDuration datatype
    pub static DAY_TIME_DURATION: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}dayTimeDuration")));

    /// xsd:gYear datatype
    pub static G_YEAR: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}gYear")));

    /// xsd:gYearMonth datatype
    pub static G_YEAR_MONTH: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}gYearMonth")));

    /// xsd:gMonth datatype
    pub static G_MONTH: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}gMonth")));

    /// xsd:gMonthDay datatype
    pub static G_MONTH_DAY: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}gMonthDay")));

    /// xsd:gDay datatype
    pub static G_DAY: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}gDay")));
}

/// RDFS vocabulary namespace
pub mod rdfs {
    use super::*;

    /// The RDFS namespace IRI
    pub const NAMESPACE: &str = "http://www.w3.org/2000/01/rdf-schema#";

    /// rdfs:Class class
    pub static CLASS: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}Class")));

    /// rdfs:subClassOf predicate
    pub static SUB_CLASS_OF: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}subClassOf")));

    /// rdfs:subPropertyOf predicate
    pub static SUB_PROPERTY_OF: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}subPropertyOf")));

    /// rdfs:domain predicate
    pub static DOMAIN: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}domain")));

    /// rdfs:range predicate
    pub static RANGE: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}range")));

    /// rdfs:label predicate
    pub static LABEL: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}label")));

    /// rdfs:comment predicate
    pub static COMMENT: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}comment")));
}

/// OWL vocabulary namespace
pub mod owl {
    use super::*;

    /// The OWL namespace IRI
    pub const NAMESPACE: &str = "http://www.w3.org/2002/07/owl#";

    /// owl:Class class
    pub static CLASS: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}Class")));

    /// owl:ObjectProperty class
    pub static OBJECT_PROPERTY: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}ObjectProperty")));

    /// owl:DatatypeProperty class
    pub static DATATYPE_PROPERTY: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}DatatypeProperty")));

    /// owl:FunctionalProperty class
    pub static FUNCTIONAL_PROPERTY: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}FunctionalProperty")));

    /// owl:InverseFunctionalProperty class
    pub static INVERSE_FUNCTIONAL_PROPERTY: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}InverseFunctionalProperty")));

    /// owl:sameAs predicate
    pub static SAME_AS: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}sameAs")));

    /// owl:differentFrom predicate
    pub static DIFFERENT_FROM: Lazy<NamedNode> =
        Lazy::new(|| NamedNode::new_unchecked(format!("{NAMESPACE}differentFrom")));
}
