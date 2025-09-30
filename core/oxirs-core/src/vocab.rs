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
