//! # QueryParser - classify_identifier_group Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::Token;

use super::queryparser_type::QueryParser;

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
                } else {
                    Token::PrefixedName("".to_string(), identifier.to_string())
                }
            }
        }
    }
}
