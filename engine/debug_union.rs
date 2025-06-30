use oxirs_arq::query::{QueryParser, parse_query};

fn main() {
    let query_str = r#"
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?name WHERE {
            { ?person foaf:name ?name }
            UNION
            { ?person rdfs:label ?name }
        }
    "#;

    println!("Testing tokenization...");
    let mut parser = QueryParser::new();
    
    match parser.tokenize(query_str) {
        Ok(_) => {
            println!("Tokenization successful!");
            println!("Tokens: {:?}", parser.tokens);
        }
        Err(e) => {
            println!("Tokenization failed: {}", e);
            return;
        }
    }

    println!("\nTesting parsing...");
    match parse_query(query_str) {
        Ok(query) => {
            println!("Parsing successful!");
            println!("Query type: {:?}", query.query_type);
            println!("Variables: {:?}", query.select_variables);
            println!("Where clause: {:?}", query.where_clause);
        }
        Err(e) => {
            println!("Parsing failed: {}", e);
        }
    }
}