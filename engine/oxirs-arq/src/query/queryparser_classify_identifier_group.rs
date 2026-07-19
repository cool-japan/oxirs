//! # QueryParser - classify_identifier_group Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::Token;

use super::queryparser_type::QueryParser;

/// Map a *bare* (colon-free) identifier to the canonical lower-case spelling of
/// its SPARQL 1.1 built-in call, case-insensitively. Returns `None` when the
/// identifier is not a built-in.
///
/// The SPARQL set aggregate functions (`COUNT`, `SUM`, `MIN`, `MAX`, `AVG`,
/// `SAMPLE`, `GROUP_CONCAT`) are deliberately excluded: they are recognised on
/// the projection / `HAVING` aggregate path as [`Token::PrefixedName`] tokens
/// (see `queryparser_projection::try_parse_aggregate`), and reclassifying them
/// here would break that path.
///
/// `EXISTS` / `NOT` are already keyword tokens and never reach this function.
pub(super) fn builtin_call_name(identifier: &str) -> Option<&'static str> {
    let canonical = match identifier.to_ascii_uppercase().as_str() {
        "STR" => "str",
        "LANG" => "lang",
        "LANGMATCHES" => "langmatches",
        "DATATYPE" => "datatype",
        "BOUND" => "bound",
        "IRI" => "iri",
        "URI" => "uri",
        "BNODE" => "bnode",
        "RAND" => "rand",
        "ABS" => "abs",
        "CEIL" => "ceil",
        "FLOOR" => "floor",
        "ROUND" => "round",
        "CONCAT" => "concat",
        "STRLEN" => "strlen",
        "UCASE" => "ucase",
        "LCASE" => "lcase",
        "ENCODE_FOR_URI" => "encode_for_uri",
        "CONTAINS" => "contains",
        "STRSTARTS" => "strstarts",
        "STRENDS" => "strends",
        "STRBEFORE" => "strbefore",
        "STRAFTER" => "strafter",
        "YEAR" => "year",
        "MONTH" => "month",
        "DAY" => "day",
        "HOURS" => "hours",
        "MINUTES" => "minutes",
        "SECONDS" => "seconds",
        "TIMEZONE" => "timezone",
        "TZ" => "tz",
        "NOW" => "now",
        "UUID" => "uuid",
        "STRUUID" => "struuid",
        "MD5" => "md5",
        "SHA1" => "sha1",
        "SHA256" => "sha256",
        "SHA384" => "sha384",
        "SHA512" => "sha512",
        "COALESCE" => "coalesce",
        "IF" => "if",
        "STRLANG" => "strlang",
        "STRDT" => "strdt",
        "SAMETERM" => "sameterm",
        "ISIRI" => "isiri",
        "ISURI" => "isuri",
        "ISBLANK" => "isblank",
        "ISLITERAL" => "isliteral",
        "ISNUMERIC" => "isnumeric",
        "REGEX" => "regex",
        "SUBSTR" => "substr",
        "REPLACE" => "replace",
        _ => return None,
    };
    Some(canonical)
}

impl QueryParser {
    pub(super) fn classify_identifier(&self, identifier: &str) -> Token {
        match identifier.to_uppercase().as_str() {
            "SELECT" => Token::Select,
            "CONSTRUCT" => Token::Construct,
            "ASK" => Token::Ask,
            "DESCRIBE" => Token::Describe,
            "WHERE" => Token::Where,
            "OPTIONAL" => Token::Optional,
            "UNION" => Token::Union,
            "MINUS" => Token::Minus,
            "FILTER" => Token::Filter,
            "BIND" => Token::Bind,
            "SERVICE" => Token::Service,
            "GRAPH" => Token::Graph,
            "FROM" => Token::From,
            "NAMED" => Token::Named,
            "PREFIX" => Token::Prefix,
            "BASE" => Token::Base,
            "DISTINCT" => Token::Distinct,
            "REDUCED" => Token::Reduced,
            "ORDER" => Token::OrderBy,
            "BY" => Token::OrderBy,
            "GROUP" => Token::GroupBy,
            "HAVING" => Token::Having,
            "LIMIT" => Token::Limit,
            "OFFSET" => Token::Offset,
            "ASC" => Token::Asc,
            "DESC" => Token::Desc,
            "AS" => Token::As,
            "VALUES" => Token::Values,
            "EXISTS" => Token::Exists,
            "NOT" => Token::Not,
            "IN" => Token::In,
            "AND" => Token::And,
            "OR" => Token::Or,
            "TRUE" => Token::BooleanLiteral(true),
            "FALSE" => Token::BooleanLiteral(false),
            "INSERT" => Token::Insert,
            "DELETE" => Token::Delete,
            "UPDATE" => Token::Update,
            "CREATE" => Token::Create,
            "DROP" => Token::Drop,
            "CLEAR" => Token::Clear,
            "LOAD" => Token::Load,
            "COPY" => Token::Copy,
            "MOVE" => Token::Move,
            "ADD" => Token::Add,
            "DATA" => Token::Data,
            "WITH" => Token::With,
            "USING" => Token::Using,
            "SILENT" => Token::Silent,
            "ALL" => Token::All,
            "DEFAULT" => Token::Default,
            "TO" => Token::To,
            _ => {
                if let Some(colon_pos) = identifier.find(':') {
                    let prefix = identifier[..colon_pos].to_string();
                    let local = identifier[colon_pos + 1..].to_string();
                    Token::PrefixedName(prefix, local)
                } else if let Some(stripped) = identifier.strip_prefix(':') {
                    let local = stripped.to_string();
                    Token::PrefixedName("".to_string(), local)
                } else if identifier == "a" {
                    // The bare, lowercase `a` is the SPARQL `rdf:type` predicate
                    // shorthand. Matched case-sensitively (`A` is not rdf:type)
                    // and only in this colon-free branch, so `:a` / `a:` / `?a` /
                    // `"a"` are all unaffected.
                    Token::A
                } else if let Some(canonical) = builtin_call_name(identifier) {
                    // Bare (colon-free) SPARQL built-in call, e.g. `LANG`, `isIRI`,
                    // `REGEX`. A leading-colon name such as `:lang` takes the
                    // `find(':')` branch above and is never reclassified here, so a
                    // user-defined default-prefix function stays a PrefixedName.
                    Token::BuiltIn(canonical.to_string())
                } else {
                    Token::PrefixedName("".to_string(), identifier.to_string())
                }
            }
        }
    }
}
