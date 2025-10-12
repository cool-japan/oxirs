//! Type mapping between XSD and AAS data types

/// AAS DataTypeDefXsd enum (AAS V3.0 XML Schema types)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataTypeDefXsd {
    AnyUri,
    Base64Binary,
    Boolean,
    Byte,
    Date,
    DateTime,
    Decimal,
    Double,
    Duration,
    Float,
    GDay,
    GMonth,
    GMonthDay,
    GYear,
    GYearMonth,
    HexBinary,
    Int,
    Integer,
    Long,
    NegativeInteger,
    NonNegativeInteger,
    NonPositiveInteger,
    PositiveInteger,
    Short,
    String,
    Time,
    UnsignedByte,
    UnsignedInt,
    UnsignedLong,
    UnsignedShort,
}

impl DataTypeDefXsd {
    /// Convert to AAS XML schema string
    pub fn to_xsd_string(&self) -> &'static str {
        match self {
            Self::AnyUri => "xs:anyURI",
            Self::Base64Binary => "xs:base64Binary",
            Self::Boolean => "xs:boolean",
            Self::Byte => "xs:byte",
            Self::Date => "xs:date",
            Self::DateTime => "xs:dateTime",
            Self::Decimal => "xs:decimal",
            Self::Double => "xs:double",
            Self::Duration => "xs:duration",
            Self::Float => "xs:float",
            Self::GDay => "xs:gDay",
            Self::GMonth => "xs:gMonth",
            Self::GMonthDay => "xs:gMonthDay",
            Self::GYear => "xs:gYear",
            Self::GYearMonth => "xs:gYearMonth",
            Self::HexBinary => "xs:hexBinary",
            Self::Int => "xs:int",
            Self::Integer => "xs:integer",
            Self::Long => "xs:long",
            Self::NegativeInteger => "xs:negativeInteger",
            Self::NonNegativeInteger => "xs:nonNegativeInteger",
            Self::NonPositiveInteger => "xs:nonPositiveInteger",
            Self::PositiveInteger => "xs:positiveInteger",
            Self::Short => "xs:short",
            Self::String => "xs:string",
            Self::Time => "xs:time",
            Self::UnsignedByte => "xs:unsignedByte",
            Self::UnsignedInt => "xs:unsignedInt",
            Self::UnsignedLong => "xs:unsignedLong",
            Self::UnsignedShort => "xs:unsignedShort",
        }
    }
}

/// AAS DataTypeIec61360 enum (IEC 61360 data types)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataTypeIec61360 {
    Blob,
    Boolean,
    Date,
    File,
    Html,
    IntegerCount,
    IntegerCurrency,
    IntegerMeasure,
    Irdi,
    Iri,
    Rational,
    RationalMeasure,
    RealCount,
    RealCurrency,
    RealMeasure,
    String,
    StringTranslatable,
    Time,
    Timestamp,
}

impl DataTypeIec61360 {
    /// Convert to IEC 61360 string
    pub fn to_string(&self) -> &'static str {
        match self {
            Self::Blob => "BLOB",
            Self::Boolean => "BOOLEAN",
            Self::Date => "DATE",
            Self::File => "FILE",
            Self::Html => "HTML",
            Self::IntegerCount => "INTEGER_COUNT",
            Self::IntegerCurrency => "INTEGER_CURRENCY",
            Self::IntegerMeasure => "INTEGER_MEASURE",
            Self::Irdi => "IRDI",
            Self::Iri => "IRI",
            Self::Rational => "RATIONAL",
            Self::RationalMeasure => "RATIONAL_MEASURE",
            Self::RealCount => "REAL_COUNT",
            Self::RealCurrency => "REAL_CURRENCY",
            Self::RealMeasure => "REAL_MEASURE",
            Self::String => "STRING",
            Self::StringTranslatable => "STRING_TRANSLATABLE",
            Self::Time => "TIME",
            Self::Timestamp => "TIMESTAMP",
        }
    }
}

/// Map XSD data type URI to AAS DataTypeDefXsd
///
/// Reference: Eclipse ESMF AasDataTypeMapper.java
pub fn map_xsd_to_aas_data_type_def_xsd(xsd_type: &str) -> DataTypeDefXsd {
    match xsd_type {
        t if t.ends_with("anyURI") => DataTypeDefXsd::AnyUri,
        t if t.ends_with("yearMonthDuration")
            | t.ends_with("dayTimeDuration")
            | t.ends_with("duration") =>
        {
            DataTypeDefXsd::Duration
        }
        t if t.ends_with("boolean") => DataTypeDefXsd::Boolean,
        t if t.ends_with("byte") => DataTypeDefXsd::Byte,
        t if t.ends_with("date") => DataTypeDefXsd::Date,
        t if t.ends_with("dateTime") | t.ends_with("dateTimeStamp") => DataTypeDefXsd::DateTime,
        t if t.ends_with("decimal") => DataTypeDefXsd::Decimal,
        t if t.ends_with("double") => DataTypeDefXsd::Double,
        t if t.ends_with("float") => DataTypeDefXsd::Float,
        t if t.ends_with("gMonth") => DataTypeDefXsd::GMonth,
        t if t.ends_with("gMonthDay") => DataTypeDefXsd::GMonthDay,
        t if t.ends_with("gYear") => DataTypeDefXsd::GYear,
        t if t.ends_with("gYearMonth") => DataTypeDefXsd::GYearMonth,
        t if t.ends_with("hexBinary") => DataTypeDefXsd::HexBinary,
        t if t.ends_with("#int") => DataTypeDefXsd::Int,
        t if t.ends_with("integer") => DataTypeDefXsd::Integer,
        t if t.ends_with("long") => DataTypeDefXsd::Long,
        t if t.ends_with("negativeInteger") => DataTypeDefXsd::NegativeInteger,
        t if t.ends_with("nonNegativeInteger") => DataTypeDefXsd::NonNegativeInteger,
        t if t.ends_with("positiveInteger") => DataTypeDefXsd::PositiveInteger,
        t if t.ends_with("short") => DataTypeDefXsd::Short,
        t if t.ends_with("string") => DataTypeDefXsd::String,
        t if t.ends_with("time") => DataTypeDefXsd::Time,
        t if t.ends_with("unsignedByte") => DataTypeDefXsd::UnsignedByte,
        t if t.ends_with("unsignedInt") => DataTypeDefXsd::UnsignedInt,
        t if t.ends_with("unsignedLong") => DataTypeDefXsd::UnsignedLong,
        t if t.ends_with("unsignedShort") => DataTypeDefXsd::UnsignedShort,
        _ => DataTypeDefXsd::String, // Default fallback
    }
}

/// Map XSD data type URI to AAS DataTypeIec61360
///
/// Reference: Eclipse ESMF AspectModelAasVisitor.java TYPE_MAP
pub fn map_xsd_to_iec61360_data_type(xsd_type: &str) -> DataTypeIec61360 {
    match xsd_type {
        t if t.ends_with("boolean") => DataTypeIec61360::Boolean,
        t if t.ends_with("decimal") | t.ends_with("integer") => DataTypeIec61360::IntegerMeasure,
        t if t.ends_with("float") | t.ends_with("double") => DataTypeIec61360::RealMeasure,
        t if t.ends_with("byte")
            | t.ends_with("short")
            | t.ends_with("#int")
            | t.ends_with("long")
            | t.ends_with("unsignedByte")
            | t.ends_with("unsignedShort")
            | t.ends_with("unsignedInt")
            | t.ends_with("unsignedLong")
            | t.ends_with("positiveInteger")
            | t.ends_with("nonPositiveInteger")
            | t.ends_with("negativeInteger")
            | t.ends_with("nonNegativeInteger") =>
        {
            DataTypeIec61360::IntegerCount
        }
        t if t.ends_with("langString") => DataTypeIec61360::String,
        _ => DataTypeIec61360::String, // Default fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xsd_to_aas_data_type() {
        assert_eq!(
            map_xsd_to_aas_data_type_def_xsd("http://www.w3.org/2001/XMLSchema#string"),
            DataTypeDefXsd::String
        );
        assert_eq!(
            map_xsd_to_aas_data_type_def_xsd("http://www.w3.org/2001/XMLSchema#int"),
            DataTypeDefXsd::Int
        );
        assert_eq!(
            map_xsd_to_aas_data_type_def_xsd("http://www.w3.org/2001/XMLSchema#boolean"),
            DataTypeDefXsd::Boolean
        );
        assert_eq!(
            map_xsd_to_aas_data_type_def_xsd("http://www.w3.org/2001/XMLSchema#dateTime"),
            DataTypeDefXsd::DateTime
        );
    }

    #[test]
    fn test_xsd_to_iec61360_data_type() {
        assert_eq!(
            map_xsd_to_iec61360_data_type("http://www.w3.org/2001/XMLSchema#string"),
            DataTypeIec61360::String
        );
        assert_eq!(
            map_xsd_to_iec61360_data_type("http://www.w3.org/2001/XMLSchema#integer"),
            DataTypeIec61360::IntegerMeasure
        );
        assert_eq!(
            map_xsd_to_iec61360_data_type("http://www.w3.org/2001/XMLSchema#float"),
            DataTypeIec61360::RealMeasure
        );
    }

    #[test]
    fn test_data_type_def_xsd_to_string() {
        assert_eq!(DataTypeDefXsd::String.to_xsd_string(), "xs:string");
        assert_eq!(DataTypeDefXsd::Int.to_xsd_string(), "xs:int");
        assert_eq!(DataTypeDefXsd::Boolean.to_xsd_string(), "xs:boolean");
    }

    #[test]
    fn test_data_type_iec61360_to_string() {
        assert_eq!(DataTypeIec61360::String.to_string(), "STRING");
        assert_eq!(
            DataTypeIec61360::IntegerMeasure.to_string(),
            "INTEGER_MEASURE"
        );
        assert_eq!(DataTypeIec61360::RealMeasure.to_string(), "REAL_MEASURE");
    }
}
