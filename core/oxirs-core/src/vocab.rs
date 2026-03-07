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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdf_namespace() {
        assert_eq!(
            rdf::NAMESPACE,
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        );
    }

    #[test]
    fn test_rdf_vocabulary() {
        assert_eq!(
            rdf::TYPE.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        );
        assert_eq!(
            rdf::PROPERTY.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
        );
        assert_eq!(
            rdf::RESOURCE.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#Resource"
        );
        assert_eq!(
            rdf::STATEMENT.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement"
        );
        assert_eq!(
            rdf::SUBJECT.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"
        );
        assert_eq!(
            rdf::PREDICATE.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"
        );
        assert_eq!(
            rdf::OBJECT.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#object"
        );
        assert_eq!(
            rdf::LIST.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#List"
        );
        assert_eq!(
            rdf::FIRST.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#first"
        );
        assert_eq!(
            rdf::REST.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest"
        );
        assert_eq!(
            rdf::NIL.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"
        );
        assert_eq!(
            rdf::LANG_STRING.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"
        );
    }

    #[test]
    #[cfg(feature = "rdf-12")]
    fn test_rdf_12_vocabulary() {
        assert_eq!(
            rdf::DIR_LANG_STRING.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#dirLangString"
        );
    }

    #[test]
    fn test_xsd_namespace() {
        assert_eq!(xsd::NAMESPACE, "http://www.w3.org/2001/XMLSchema#");
    }

    #[test]
    fn test_xsd_basic_datatypes() {
        assert_eq!(
            xsd::STRING.as_str(),
            "http://www.w3.org/2001/XMLSchema#string"
        );
        assert_eq!(
            xsd::BOOLEAN.as_str(),
            "http://www.w3.org/2001/XMLSchema#boolean"
        );
        assert_eq!(
            xsd::INTEGER.as_str(),
            "http://www.w3.org/2001/XMLSchema#integer"
        );
        assert_eq!(
            xsd::DECIMAL.as_str(),
            "http://www.w3.org/2001/XMLSchema#decimal"
        );
        assert_eq!(
            xsd::DOUBLE.as_str(),
            "http://www.w3.org/2001/XMLSchema#double"
        );
        assert_eq!(
            xsd::FLOAT.as_str(),
            "http://www.w3.org/2001/XMLSchema#float"
        );
    }

    #[test]
    fn test_xsd_date_time_datatypes() {
        assert_eq!(xsd::DATE.as_str(), "http://www.w3.org/2001/XMLSchema#date");
        assert_eq!(xsd::TIME.as_str(), "http://www.w3.org/2001/XMLSchema#time");
        assert_eq!(
            xsd::DATE_TIME.as_str(),
            "http://www.w3.org/2001/XMLSchema#dateTime"
        );
        assert_eq!(
            xsd::DURATION.as_str(),
            "http://www.w3.org/2001/XMLSchema#duration"
        );
        assert_eq!(
            xsd::YEAR_MONTH_DURATION.as_str(),
            "http://www.w3.org/2001/XMLSchema#yearMonthDuration"
        );
        assert_eq!(
            xsd::DAY_TIME_DURATION.as_str(),
            "http://www.w3.org/2001/XMLSchema#dayTimeDuration"
        );
    }

    #[test]
    fn test_xsd_gregorian_datatypes() {
        assert_eq!(
            xsd::G_YEAR.as_str(),
            "http://www.w3.org/2001/XMLSchema#gYear"
        );
        assert_eq!(
            xsd::G_YEAR_MONTH.as_str(),
            "http://www.w3.org/2001/XMLSchema#gYearMonth"
        );
        assert_eq!(
            xsd::G_MONTH.as_str(),
            "http://www.w3.org/2001/XMLSchema#gMonth"
        );
        assert_eq!(
            xsd::G_MONTH_DAY.as_str(),
            "http://www.w3.org/2001/XMLSchema#gMonthDay"
        );
        assert_eq!(xsd::G_DAY.as_str(), "http://www.w3.org/2001/XMLSchema#gDay");
    }

    #[test]
    fn test_xsd_binary_datatypes() {
        assert_eq!(
            xsd::BASE64_BINARY.as_str(),
            "http://www.w3.org/2001/XMLSchema#base64Binary"
        );
        assert_eq!(
            xsd::HEX_BINARY.as_str(),
            "http://www.w3.org/2001/XMLSchema#hexBinary"
        );
    }

    #[test]
    fn test_xsd_integer_datatypes() {
        assert_eq!(xsd::LONG.as_str(), "http://www.w3.org/2001/XMLSchema#long");
        assert_eq!(xsd::INT.as_str(), "http://www.w3.org/2001/XMLSchema#int");
        assert_eq!(
            xsd::SHORT.as_str(),
            "http://www.w3.org/2001/XMLSchema#short"
        );
        assert_eq!(xsd::BYTE.as_str(), "http://www.w3.org/2001/XMLSchema#byte");
        assert_eq!(
            xsd::UNSIGNED_LONG.as_str(),
            "http://www.w3.org/2001/XMLSchema#unsignedLong"
        );
        assert_eq!(
            xsd::UNSIGNED_INT.as_str(),
            "http://www.w3.org/2001/XMLSchema#unsignedInt"
        );
        assert_eq!(
            xsd::UNSIGNED_SHORT.as_str(),
            "http://www.w3.org/2001/XMLSchema#unsignedShort"
        );
        assert_eq!(
            xsd::UNSIGNED_BYTE.as_str(),
            "http://www.w3.org/2001/XMLSchema#unsignedByte"
        );
    }

    #[test]
    fn test_xsd_special_integer_datatypes() {
        assert_eq!(
            xsd::POSITIVE_INTEGER.as_str(),
            "http://www.w3.org/2001/XMLSchema#positiveInteger"
        );
        assert_eq!(
            xsd::NON_NEGATIVE_INTEGER.as_str(),
            "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
        );
        assert_eq!(
            xsd::NEGATIVE_INTEGER.as_str(),
            "http://www.w3.org/2001/XMLSchema#negativeInteger"
        );
        assert_eq!(
            xsd::NON_POSITIVE_INTEGER.as_str(),
            "http://www.w3.org/2001/XMLSchema#nonPositiveInteger"
        );
    }

    #[test]
    fn test_xsd_string_datatypes() {
        assert_eq!(
            xsd::NORMALIZED_STRING.as_str(),
            "http://www.w3.org/2001/XMLSchema#normalizedString"
        );
        assert_eq!(
            xsd::TOKEN.as_str(),
            "http://www.w3.org/2001/XMLSchema#token"
        );
        assert_eq!(
            xsd::LANGUAGE.as_str(),
            "http://www.w3.org/2001/XMLSchema#language"
        );
    }

    #[test]
    fn test_xsd_misc_datatypes() {
        assert_eq!(
            xsd::ANY_URI.as_str(),
            "http://www.w3.org/2001/XMLSchema#anyURI"
        );
        // Note: xsd::LANG_STRING should actually point to RDF namespace
        assert_eq!(
            xsd::LANG_STRING.as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"
        );
    }

    #[test]
    fn test_rdfs_namespace() {
        assert_eq!(rdfs::NAMESPACE, "http://www.w3.org/2000/01/rdf-schema#");
    }

    #[test]
    fn test_rdfs_vocabulary() {
        assert_eq!(
            rdfs::CLASS.as_str(),
            "http://www.w3.org/2000/01/rdf-schema#Class"
        );
        assert_eq!(
            rdfs::SUB_CLASS_OF.as_str(),
            "http://www.w3.org/2000/01/rdf-schema#subClassOf"
        );
        assert_eq!(
            rdfs::SUB_PROPERTY_OF.as_str(),
            "http://www.w3.org/2000/01/rdf-schema#subPropertyOf"
        );
        assert_eq!(
            rdfs::DOMAIN.as_str(),
            "http://www.w3.org/2000/01/rdf-schema#domain"
        );
        assert_eq!(
            rdfs::RANGE.as_str(),
            "http://www.w3.org/2000/01/rdf-schema#range"
        );
        assert_eq!(
            rdfs::LABEL.as_str(),
            "http://www.w3.org/2000/01/rdf-schema#label"
        );
        assert_eq!(
            rdfs::COMMENT.as_str(),
            "http://www.w3.org/2000/01/rdf-schema#comment"
        );
    }

    #[test]
    fn test_owl_namespace() {
        assert_eq!(owl::NAMESPACE, "http://www.w3.org/2002/07/owl#");
    }

    #[test]
    fn test_owl_vocabulary() {
        assert_eq!(owl::CLASS.as_str(), "http://www.w3.org/2002/07/owl#Class");
        assert_eq!(
            owl::OBJECT_PROPERTY.as_str(),
            "http://www.w3.org/2002/07/owl#ObjectProperty"
        );
        assert_eq!(
            owl::DATATYPE_PROPERTY.as_str(),
            "http://www.w3.org/2002/07/owl#DatatypeProperty"
        );
        assert_eq!(
            owl::FUNCTIONAL_PROPERTY.as_str(),
            "http://www.w3.org/2002/07/owl#FunctionalProperty"
        );
        assert_eq!(
            owl::INVERSE_FUNCTIONAL_PROPERTY.as_str(),
            "http://www.w3.org/2002/07/owl#InverseFunctionalProperty"
        );
        assert_eq!(
            owl::SAME_AS.as_str(),
            "http://www.w3.org/2002/07/owl#sameAs"
        );
        assert_eq!(
            owl::DIFFERENT_FROM.as_str(),
            "http://www.w3.org/2002/07/owl#differentFrom"
        );
    }

    #[test]
    fn test_lazy_static_initialization() {
        // Test that lazy statics are properly initialized and reused
        let type1 = &*rdf::TYPE;
        let type2 = &*rdf::TYPE;
        assert!(std::ptr::eq(type1, type2)); // Should be same instance

        let string1 = &*xsd::STRING;
        let string2 = &*xsd::STRING;
        assert!(std::ptr::eq(string1, string2)); // Should be same instance
    }

    #[test]
    fn test_vocabularies_contain_no_trailing_spaces() {
        // Ensure none of the vocabulary IRIs have trailing spaces
        assert!(!rdf::TYPE.as_str().contains(' '));
        assert!(!xsd::STRING.as_str().contains(' '));
        assert!(!rdfs::CLASS.as_str().contains(' '));
        assert!(!owl::CLASS.as_str().contains(' '));
    }

    #[test]
    fn test_namespace_consistency() {
        // Test that all RDF vocabulary items start with RDF namespace
        assert!(rdf::TYPE.as_str().starts_with(rdf::NAMESPACE));
        assert!(rdf::PROPERTY.as_str().starts_with(rdf::NAMESPACE));
        assert!(rdf::LIST.as_str().starts_with(rdf::NAMESPACE));

        // Test that all XSD vocabulary items start with XSD namespace
        assert!(xsd::STRING.as_str().starts_with(xsd::NAMESPACE));
        assert!(xsd::INTEGER.as_str().starts_with(xsd::NAMESPACE));
        // Exception: xsd::LANG_STRING should point to RDF namespace
        assert!(xsd::LANG_STRING.as_str().starts_with(rdf::NAMESPACE));

        // Test that all RDFS vocabulary items start with RDFS namespace
        assert!(rdfs::CLASS.as_str().starts_with(rdfs::NAMESPACE));
        assert!(rdfs::LABEL.as_str().starts_with(rdfs::NAMESPACE));

        // Test that all OWL vocabulary items start with OWL namespace
        assert!(owl::CLASS.as_str().starts_with(owl::NAMESPACE));
        assert!(owl::SAME_AS.as_str().starts_with(owl::NAMESPACE));
    }
}

/// SKOS Simple Knowledge Organization System vocabulary namespace
/// W3C spec: <https://www.w3.org/TR/skos-reference/>
pub mod skos {
    /// The SKOS namespace IRI
    pub const NAMESPACE: &str = "http://www.w3.org/2004/02/skos/core#";

    // --- SKOS Classes ---

    /// skos:Concept class — the unit of thought in a thesaurus or taxonomy
    pub const CONCEPT: &str = "http://www.w3.org/2004/02/skos/core#Concept";

    /// skos:ConceptScheme class — a set of concepts
    pub const CONCEPT_SCHEME: &str = "http://www.w3.org/2004/02/skos/core#ConceptScheme";

    /// skos:Collection class — a meaningful collection of concepts
    pub const COLLECTION: &str = "http://www.w3.org/2004/02/skos/core#Collection";

    /// skos:OrderedCollection class — an ordered collection of concepts
    pub const ORDERED_COLLECTION: &str = "http://www.w3.org/2004/02/skos/core#OrderedCollection";

    // --- SKOS Lexical Labels ---

    /// skos:prefLabel — preferred lexical label for a concept
    pub const PREF_LABEL: &str = "http://www.w3.org/2004/02/skos/core#prefLabel";

    /// skos:altLabel — alternative lexical label for a concept
    pub const ALT_LABEL: &str = "http://www.w3.org/2004/02/skos/core#altLabel";

    /// skos:hiddenLabel — a label not intended for display
    pub const HIDDEN_LABEL: &str = "http://www.w3.org/2004/02/skos/core#hiddenLabel";

    // --- SKOS Hierarchical Relations ---

    /// skos:broader — a concept that is more general
    pub const BROADER: &str = "http://www.w3.org/2004/02/skos/core#broader";

    /// skos:narrower — a concept that is more specific
    pub const NARROWER: &str = "http://www.w3.org/2004/02/skos/core#narrower";

    /// skos:broaderTransitive — reflexive transitive closure of skos:broader
    pub const BROADER_TRANSITIVE: &str = "http://www.w3.org/2004/02/skos/core#broaderTransitive";

    /// skos:narrowerTransitive — reflexive transitive closure of skos:narrower
    pub const NARROWER_TRANSITIVE: &str = "http://www.w3.org/2004/02/skos/core#narrowerTransitive";

    /// skos:related — an associative relationship between concepts
    pub const RELATED: &str = "http://www.w3.org/2004/02/skos/core#related";

    // --- SKOS Mapping Relations ---

    /// skos:exactMatch — two concepts are interchangeable across vocabularies
    pub const EXACT_MATCH: &str = "http://www.w3.org/2004/02/skos/core#exactMatch";

    /// skos:closeMatch — concepts are sufficiently similar to be useful
    pub const CLOSE_MATCH: &str = "http://www.w3.org/2004/02/skos/core#closeMatch";

    /// skos:broadMatch — the object concept is broader than the subject
    pub const BROAD_MATCH: &str = "http://www.w3.org/2004/02/skos/core#broadMatch";

    /// skos:narrowMatch — the object concept is narrower than the subject
    pub const NARROW_MATCH: &str = "http://www.w3.org/2004/02/skos/core#narrowMatch";

    /// skos:relatedMatch — an associative mapping relation
    pub const RELATED_MATCH: &str = "http://www.w3.org/2004/02/skos/core#relatedMatch";

    // --- SKOS Scheme Relations ---

    /// skos:inScheme — the concept scheme that a concept is a member of
    pub const IN_SCHEME: &str = "http://www.w3.org/2004/02/skos/core#inScheme";

    /// skos:hasTopConcept — top-level concept in the concept scheme
    pub const HAS_TOP_CONCEPT: &str = "http://www.w3.org/2004/02/skos/core#hasTopConcept";

    /// skos:topConceptOf — the concept scheme that this is a top concept of
    pub const TOP_CONCEPT_OF: &str = "http://www.w3.org/2004/02/skos/core#topConceptOf";

    // --- SKOS Documentation Properties ---

    /// skos:definition — a formal statement of the meaning of a concept
    pub const DEFINITION: &str = "http://www.w3.org/2004/02/skos/core#definition";

    /// skos:example — an example of how a concept is used
    pub const EXAMPLE: &str = "http://www.w3.org/2004/02/skos/core#example";

    /// skos:note — a general note for any purpose
    pub const NOTE: &str = "http://www.w3.org/2004/02/skos/core#note";

    /// skos:scopeNote — note that helps clarify the meaning of a concept
    pub const SCOPE_NOTE: &str = "http://www.w3.org/2004/02/skos/core#scopeNote";

    /// skos:changeNote — note about a modification to a concept
    pub const CHANGE_NOTE: &str = "http://www.w3.org/2004/02/skos/core#changeNote";

    /// skos:historyNote — note about the past state/meaning of a concept
    pub const HISTORY_NOTE: &str = "http://www.w3.org/2004/02/skos/core#historyNote";

    /// skos:editorialNote — administrative information for editors
    pub const EDITORIAL_NOTE: &str = "http://www.w3.org/2004/02/skos/core#editorialNote";

    // --- SKOS Notation ---

    /// skos:notation — a notation, unique within the scope of a scheme
    pub const NOTATION: &str = "http://www.w3.org/2004/02/skos/core#notation";

    // --- SKOS Collections ---

    /// skos:member — a member of a collection
    pub const MEMBER: &str = "http://www.w3.org/2004/02/skos/core#member";

    /// skos:memberList — the ordered list of members in an ordered collection
    pub const MEMBER_LIST: &str = "http://www.w3.org/2004/02/skos/core#memberList";

    // --- Helper functions ---

    /// Returns true if `iri` belongs to the SKOS namespace
    pub fn is_skos_iri(iri: &str) -> bool {
        iri.starts_with(NAMESPACE)
    }

    /// Constructs a full SKOS IRI from a local name
    pub fn skos(local: &str) -> String {
        format!("{NAMESPACE}{local}")
    }
}

#[cfg(test)]
mod skos_vocab_tests {
    use super::skos;

    #[test]
    fn test_skos_namespace() {
        assert_eq!(skos::NAMESPACE, "http://www.w3.org/2004/02/skos/core#");
    }

    #[test]
    fn test_skos_classes() {
        assert_eq!(skos::CONCEPT, "http://www.w3.org/2004/02/skos/core#Concept");
        assert_eq!(
            skos::CONCEPT_SCHEME,
            "http://www.w3.org/2004/02/skos/core#ConceptScheme"
        );
        assert_eq!(
            skos::COLLECTION,
            "http://www.w3.org/2004/02/skos/core#Collection"
        );
        assert_eq!(
            skos::ORDERED_COLLECTION,
            "http://www.w3.org/2004/02/skos/core#OrderedCollection"
        );
    }

    #[test]
    fn test_skos_lexical_labels() {
        assert_eq!(
            skos::PREF_LABEL,
            "http://www.w3.org/2004/02/skos/core#prefLabel"
        );
        assert_eq!(
            skos::ALT_LABEL,
            "http://www.w3.org/2004/02/skos/core#altLabel"
        );
        assert_eq!(
            skos::HIDDEN_LABEL,
            "http://www.w3.org/2004/02/skos/core#hiddenLabel"
        );
    }

    #[test]
    fn test_skos_hierarchical_relations() {
        assert_eq!(skos::BROADER, "http://www.w3.org/2004/02/skos/core#broader");
        assert_eq!(
            skos::NARROWER,
            "http://www.w3.org/2004/02/skos/core#narrower"
        );
        assert_eq!(
            skos::BROADER_TRANSITIVE,
            "http://www.w3.org/2004/02/skos/core#broaderTransitive"
        );
        assert_eq!(
            skos::NARROWER_TRANSITIVE,
            "http://www.w3.org/2004/02/skos/core#narrowerTransitive"
        );
        assert_eq!(skos::RELATED, "http://www.w3.org/2004/02/skos/core#related");
    }

    #[test]
    fn test_skos_mapping_relations() {
        assert_eq!(
            skos::EXACT_MATCH,
            "http://www.w3.org/2004/02/skos/core#exactMatch"
        );
        assert_eq!(
            skos::CLOSE_MATCH,
            "http://www.w3.org/2004/02/skos/core#closeMatch"
        );
        assert_eq!(
            skos::BROAD_MATCH,
            "http://www.w3.org/2004/02/skos/core#broadMatch"
        );
        assert_eq!(
            skos::NARROW_MATCH,
            "http://www.w3.org/2004/02/skos/core#narrowMatch"
        );
        assert_eq!(
            skos::RELATED_MATCH,
            "http://www.w3.org/2004/02/skos/core#relatedMatch"
        );
    }

    #[test]
    fn test_skos_scheme_relations() {
        assert_eq!(
            skos::IN_SCHEME,
            "http://www.w3.org/2004/02/skos/core#inScheme"
        );
        assert_eq!(
            skos::HAS_TOP_CONCEPT,
            "http://www.w3.org/2004/02/skos/core#hasTopConcept"
        );
        assert_eq!(
            skos::TOP_CONCEPT_OF,
            "http://www.w3.org/2004/02/skos/core#topConceptOf"
        );
    }

    #[test]
    fn test_skos_documentation_properties() {
        assert_eq!(
            skos::DEFINITION,
            "http://www.w3.org/2004/02/skos/core#definition"
        );
        assert_eq!(skos::EXAMPLE, "http://www.w3.org/2004/02/skos/core#example");
        assert_eq!(skos::NOTE, "http://www.w3.org/2004/02/skos/core#note");
        assert_eq!(
            skos::SCOPE_NOTE,
            "http://www.w3.org/2004/02/skos/core#scopeNote"
        );
        assert_eq!(
            skos::CHANGE_NOTE,
            "http://www.w3.org/2004/02/skos/core#changeNote"
        );
        assert_eq!(
            skos::HISTORY_NOTE,
            "http://www.w3.org/2004/02/skos/core#historyNote"
        );
        assert_eq!(
            skos::EDITORIAL_NOTE,
            "http://www.w3.org/2004/02/skos/core#editorialNote"
        );
    }

    #[test]
    fn test_skos_notation_and_collections() {
        assert_eq!(
            skos::NOTATION,
            "http://www.w3.org/2004/02/skos/core#notation"
        );
        assert_eq!(skos::MEMBER, "http://www.w3.org/2004/02/skos/core#member");
        assert_eq!(
            skos::MEMBER_LIST,
            "http://www.w3.org/2004/02/skos/core#memberList"
        );
    }

    #[test]
    fn test_skos_is_skos_iri() {
        assert!(skos::is_skos_iri(skos::CONCEPT));
        assert!(skos::is_skos_iri(skos::BROADER));
        assert!(skos::is_skos_iri(skos::EXACT_MATCH));
        assert!(!skos::is_skos_iri("http://www.w3.org/2002/07/owl#Class"));
        assert!(!skos::is_skos_iri(
            "http://www.w3.org/2000/01/rdf-schema#label"
        ));
        assert!(!skos::is_skos_iri(""));
        assert!(!skos::is_skos_iri("http://example.org/skos/concept"));
    }

    #[test]
    fn test_skos_helper_fn() {
        assert_eq!(
            skos::skos("Concept"),
            "http://www.w3.org/2004/02/skos/core#Concept"
        );
        assert_eq!(
            skos::skos("broader"),
            "http://www.w3.org/2004/02/skos/core#broader"
        );
        assert_eq!(skos::skos(""), "http://www.w3.org/2004/02/skos/core#");
    }

    #[test]
    fn test_all_skos_iris_start_with_namespace() {
        let all_iris = [
            skos::CONCEPT,
            skos::CONCEPT_SCHEME,
            skos::COLLECTION,
            skos::ORDERED_COLLECTION,
            skos::PREF_LABEL,
            skos::ALT_LABEL,
            skos::HIDDEN_LABEL,
            skos::BROADER,
            skos::NARROWER,
            skos::BROADER_TRANSITIVE,
            skos::NARROWER_TRANSITIVE,
            skos::RELATED,
            skos::EXACT_MATCH,
            skos::CLOSE_MATCH,
            skos::BROAD_MATCH,
            skos::NARROW_MATCH,
            skos::RELATED_MATCH,
            skos::IN_SCHEME,
            skos::HAS_TOP_CONCEPT,
            skos::TOP_CONCEPT_OF,
            skos::DEFINITION,
            skos::EXAMPLE,
            skos::NOTE,
            skos::SCOPE_NOTE,
            skos::CHANGE_NOTE,
            skos::HISTORY_NOTE,
            skos::EDITORIAL_NOTE,
            skos::NOTATION,
            skos::MEMBER,
            skos::MEMBER_LIST,
        ];
        for iri in &all_iris {
            assert!(
                iri.starts_with(skos::NAMESPACE),
                "IRI {iri} does not start with SKOS namespace"
            );
        }
    }

    #[test]
    fn test_skos_iris_are_unique() {
        use std::collections::HashSet;
        let all_iris = vec![
            skos::CONCEPT,
            skos::CONCEPT_SCHEME,
            skos::COLLECTION,
            skos::ORDERED_COLLECTION,
            skos::PREF_LABEL,
            skos::ALT_LABEL,
            skos::HIDDEN_LABEL,
            skos::BROADER,
            skos::NARROWER,
            skos::BROADER_TRANSITIVE,
            skos::NARROWER_TRANSITIVE,
            skos::RELATED,
            skos::EXACT_MATCH,
            skos::CLOSE_MATCH,
            skos::BROAD_MATCH,
            skos::NARROW_MATCH,
            skos::RELATED_MATCH,
            skos::IN_SCHEME,
            skos::HAS_TOP_CONCEPT,
            skos::TOP_CONCEPT_OF,
            skos::DEFINITION,
            skos::EXAMPLE,
            skos::NOTE,
            skos::SCOPE_NOTE,
            skos::CHANGE_NOTE,
            skos::HISTORY_NOTE,
            skos::EDITORIAL_NOTE,
            skos::NOTATION,
            skos::MEMBER,
            skos::MEMBER_LIST,
        ];
        let unique: HashSet<_> = all_iris.iter().collect();
        assert_eq!(unique.len(), all_iris.len(), "All SKOS IRIs must be unique");
    }

    #[test]
    fn test_skos_iris_contain_no_spaces() {
        let all_iris = [
            skos::CONCEPT,
            skos::CONCEPT_SCHEME,
            skos::COLLECTION,
            skos::ORDERED_COLLECTION,
            skos::PREF_LABEL,
            skos::ALT_LABEL,
            skos::HIDDEN_LABEL,
            skos::BROADER,
            skos::NARROWER,
            skos::BROADER_TRANSITIVE,
            skos::NARROWER_TRANSITIVE,
            skos::RELATED,
            skos::EXACT_MATCH,
            skos::CLOSE_MATCH,
            skos::BROAD_MATCH,
            skos::NARROW_MATCH,
            skos::RELATED_MATCH,
            skos::IN_SCHEME,
            skos::HAS_TOP_CONCEPT,
            skos::TOP_CONCEPT_OF,
            skos::DEFINITION,
            skos::EXAMPLE,
            skos::NOTE,
            skos::SCOPE_NOTE,
            skos::CHANGE_NOTE,
            skos::HISTORY_NOTE,
            skos::EDITORIAL_NOTE,
            skos::NOTATION,
            skos::MEMBER,
            skos::MEMBER_LIST,
        ];
        for iri in &all_iris {
            assert!(!iri.contains(' '), "IRI {iri} must not contain spaces");
        }
    }
}
