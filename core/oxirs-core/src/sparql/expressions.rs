//! SPARQL expression evaluation: BIND clause and functions

use crate::error::OxirsError;
use crate::model::{Literal, Term};
use crate::rdf_store::VariableBinding;
use crate::sparql::aggregates::find_matching_paren;
use crate::Result;

/// Expression types for BIND clause
#[derive(Debug, Clone)]
pub enum Expression {
    Variable(String),
    Literal(String),
    Integer(i64),
    Float(f64),
    ArithmeticOp {
        left: Box<Expression>,
        op: ArithmeticOperator,
        right: Box<Expression>,
    },
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
}

/// Arithmetic operators for expressions
#[derive(Debug, Clone)]
pub enum ArithmeticOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
}

/// BIND clause for variable assignment
#[derive(Debug, Clone)]
pub struct BindExpression {
    pub expression: Expression,
    pub variable: String,
}

/// Extract BIND clauses from WHERE clause
pub fn extract_bind_expressions(sparql: &str) -> Result<Vec<BindExpression>> {
    let mut binds = Vec::new();

    // Find all BIND clauses
    let sparql_upper = sparql.to_uppercase();
    let mut search_pos = 0;

    while let Some(bind_pos) = sparql_upper[search_pos..].find("BIND") {
        let abs_pos = search_pos + bind_pos;
        let after_bind = &sparql[abs_pos + 4..];

        // Find the opening parenthesis
        if let Some(paren_start) = after_bind.find('(') {
            // Find matching closing parenthesis
            if let Some(paren_end) = find_matching_paren(&after_bind[paren_start..]) {
                let bind_content = &after_bind[paren_start + 1..paren_start + paren_end];

                // Parse BIND content: expression AS ?variable
                if let Some(as_pos) = bind_content.to_uppercase().find(" AS ") {
                    let expr_text = bind_content[..as_pos].trim();
                    let var_text = bind_content[as_pos + 4..].trim();

                    // Extract variable name
                    if let Some(var_name) = var_text.strip_prefix('?') {
                        // Parse expression
                        if let Ok(expression) = parse_expression(expr_text) {
                            binds.push(BindExpression {
                                expression,
                                variable: var_name.to_string(),
                            });
                        }
                    }
                }

                search_pos = abs_pos + 4 + paren_start + paren_end;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    Ok(binds)
}

/// Split function arguments respecting quotes and parentheses
pub fn split_function_args(args_text: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current_arg = String::new();
    let mut in_string = false;
    let mut string_delimiter = ' ';
    let mut paren_depth = 0;

    for ch in args_text.chars() {
        match ch {
            '"' | '\'' if !in_string => {
                in_string = true;
                string_delimiter = ch;
                current_arg.push(ch);
            }
            '"' | '\'' if in_string && ch == string_delimiter => {
                in_string = false;
                current_arg.push(ch);
            }
            '(' if !in_string => {
                paren_depth += 1;
                current_arg.push(ch);
            }
            ')' if !in_string => {
                paren_depth -= 1;
                current_arg.push(ch);
            }
            ',' if !in_string && paren_depth == 0 => {
                if !current_arg.trim().is_empty() {
                    args.push(current_arg.clone());
                }
                current_arg.clear();
            }
            _ => {
                current_arg.push(ch);
            }
        }
    }

    // Don't forget the last argument
    if !current_arg.trim().is_empty() {
        args.push(current_arg);
    }

    args
}

/// Parse an expression for BIND
pub fn parse_expression(expr: &str) -> Result<Expression> {
    let expr = expr.trim();

    // Check for arithmetic operators (simple parsing - left to right)
    if let Some(op_pos) = expr.rfind(" + ") {
        let left = parse_expression(&expr[..op_pos])?;
        let right = parse_expression(&expr[op_pos + 3..])?;
        return Ok(Expression::ArithmeticOp {
            left: Box::new(left),
            op: ArithmeticOperator::Add,
            right: Box::new(right),
        });
    }

    if let Some(op_pos) = expr.rfind(" - ") {
        let left = parse_expression(&expr[..op_pos])?;
        let right = parse_expression(&expr[op_pos + 3..])?;
        return Ok(Expression::ArithmeticOp {
            left: Box::new(left),
            op: ArithmeticOperator::Subtract,
            right: Box::new(right),
        });
    }

    if let Some(op_pos) = expr.rfind(" * ") {
        let left = parse_expression(&expr[..op_pos])?;
        let right = parse_expression(&expr[op_pos + 3..])?;
        return Ok(Expression::ArithmeticOp {
            left: Box::new(left),
            op: ArithmeticOperator::Multiply,
            right: Box::new(right),
        });
    }

    if let Some(op_pos) = expr.rfind(" / ") {
        let left = parse_expression(&expr[..op_pos])?;
        let right = parse_expression(&expr[op_pos + 3..])?;
        return Ok(Expression::ArithmeticOp {
            left: Box::new(left),
            op: ArithmeticOperator::Divide,
            right: Box::new(right),
        });
    }

    // Check for function calls
    if let Some(paren_pos) = expr.find('(') {
        if let Some(paren_end) = find_matching_paren(&expr[paren_pos..]) {
            let func_name = expr[..paren_pos].trim().to_uppercase();
            let args_text = &expr[paren_pos + 1..paren_pos + paren_end];

            // Parse function arguments
            let arg_strs = split_function_args(args_text);
            let mut args = Vec::new();
            for arg_str in arg_strs {
                args.push(parse_expression(arg_str.trim())?);
            }

            return Ok(Expression::FunctionCall {
                name: func_name,
                args,
            });
        }
    }

    // Check for variables
    if expr.starts_with('?') {
        return Ok(Expression::Variable(expr.to_string()));
    }

    // Check for string literals
    if (expr.starts_with('"') && expr.ends_with('"'))
        || (expr.starts_with('\'') && expr.ends_with('\''))
    {
        return Ok(Expression::Literal(expr[1..expr.len() - 1].to_string()));
    }

    // Check for numeric literals
    if let Ok(int_val) = expr.parse::<i64>() {
        return Ok(Expression::Integer(int_val));
    }

    if let Ok(float_val) = expr.parse::<f64>() {
        return Ok(Expression::Float(float_val));
    }

    // Default to literal
    Ok(Expression::Literal(expr.to_string()))
}

/// Evaluate an expression against a binding
pub fn evaluate_expression(expr: &Expression, binding: &VariableBinding) -> Result<Term> {
    match expr {
        Expression::Variable(var_name) => {
            let var = var_name.strip_prefix('?').unwrap_or(var_name);
            binding
                .get(var)
                .cloned()
                .ok_or_else(|| OxirsError::Query(format!("Unbound variable: {}", var_name)))
        }
        Expression::Literal(val) => Ok(Term::from(Literal::new(val.clone()))),
        Expression::Integer(val) => Ok(Term::from(Literal::new(val.to_string()))),
        Expression::Float(val) => Ok(Term::from(Literal::new(val.to_string()))),
        Expression::ArithmeticOp { left, op, right } => {
            let left_val = evaluate_expression(left, binding)?;
            let right_val = evaluate_expression(right, binding)?;

            let left_num = term_to_number(&left_val)?;
            let right_num = term_to_number(&right_val)?;

            let result = match op {
                ArithmeticOperator::Add => left_num + right_num,
                ArithmeticOperator::Subtract => left_num - right_num,
                ArithmeticOperator::Multiply => left_num * right_num,
                ArithmeticOperator::Divide => {
                    if right_num.abs() < f64::EPSILON {
                        return Err(OxirsError::Query("Division by zero".to_string()));
                    }
                    left_num / right_num
                }
            };

            Ok(Term::from(Literal::new(result.to_string())))
        }
        Expression::FunctionCall { name, args } => {
            match name.as_str() {
                "STR" => {
                    if args.len() != 1 {
                        return Err(OxirsError::Query(
                            "STR requires exactly one argument".to_string(),
                        ));
                    }
                    let val = evaluate_expression(&args[0], binding)?;
                    Ok(Term::from(Literal::new(term_to_string(&val))))
                }
                "CONCAT" => {
                    let mut result = String::new();
                    for arg in args {
                        let val = evaluate_expression(arg, binding)?;
                        result.push_str(&term_to_string(&val));
                    }
                    Ok(Term::from(Literal::new(result)))
                }
                "STRLEN" => {
                    if args.len() != 1 {
                        return Err(OxirsError::Query(
                            "STRLEN requires exactly one argument".to_string(),
                        ));
                    }
                    let val = evaluate_expression(&args[0], binding)?;
                    let s = term_to_string(&val);
                    Ok(Term::from(Literal::new(s.len().to_string())))
                }
                "UCASE" => {
                    if args.len() != 1 {
                        return Err(OxirsError::Query(
                            "UCASE requires exactly one argument".to_string(),
                        ));
                    }
                    let val = evaluate_expression(&args[0], binding)?;
                    let s = term_to_string(&val);
                    Ok(Term::from(Literal::new(s.to_uppercase())))
                }
                "LCASE" => {
                    if args.len() != 1 {
                        return Err(OxirsError::Query(
                            "LCASE requires exactly one argument".to_string(),
                        ));
                    }
                    let val = evaluate_expression(&args[0], binding)?;
                    let s = term_to_string(&val);
                    Ok(Term::from(Literal::new(s.to_lowercase())))
                }
                "CONTAINS" => {
                    if args.len() != 2 {
                        return Err(OxirsError::Query(
                            "CONTAINS requires exactly two arguments".to_string(),
                        ));
                    }
                    let haystack = evaluate_expression(&args[0], binding)?;
                    let needle = evaluate_expression(&args[1], binding)?;
                    let haystack_str = term_to_string(&haystack);
                    let needle_str = term_to_string(&needle);
                    let result = if haystack_str.contains(&needle_str) {
                        "true"
                    } else {
                        "false"
                    };
                    Ok(Term::from(Literal::new(result.to_string())))
                }
                "SUBSTR" | "SUBSTRING" => {
                    if args.len() < 2 || args.len() > 3 {
                        return Err(OxirsError::Query(
                            "SUBSTR requires 2 or 3 arguments".to_string(),
                        ));
                    }
                    let s = term_to_string(&evaluate_expression(&args[0], binding)?);
                    let start = term_to_number(&evaluate_expression(&args[1], binding)?)? as usize;

                    // SPARQL uses 1-based indexing
                    let start_idx = if start > 0 { start - 1 } else { 0 };

                    let result: String = if args.len() == 3 {
                        let length =
                            term_to_number(&evaluate_expression(&args[2], binding)?)? as usize;
                        s.chars().skip(start_idx).take(length).collect()
                    } else {
                        s.chars().skip(start_idx).collect()
                    };

                    Ok(Term::from(Literal::new(result)))
                }
                "REPLACE" => {
                    if args.len() < 3 || args.len() > 4 {
                        return Err(OxirsError::Query(
                            "REPLACE requires 3 or 4 arguments".to_string(),
                        ));
                    }
                    let s = term_to_string(&evaluate_expression(&args[0], binding)?);
                    let pattern = term_to_string(&evaluate_expression(&args[1], binding)?);
                    let replacement = term_to_string(&evaluate_expression(&args[2], binding)?);

                    // Simple string replacement (not regex for now)
                    let result = s.replace(&pattern, &replacement);
                    Ok(Term::from(Literal::new(result)))
                }
                "STRSTARTS" => {
                    if args.len() != 2 {
                        return Err(OxirsError::Query(
                            "STRSTARTS requires exactly two arguments".to_string(),
                        ));
                    }
                    let s = term_to_string(&evaluate_expression(&args[0], binding)?);
                    let prefix = term_to_string(&evaluate_expression(&args[1], binding)?);
                    let result = if s.starts_with(&prefix) {
                        "true"
                    } else {
                        "false"
                    };
                    Ok(Term::from(Literal::new(result.to_string())))
                }
                "STRENDS" => {
                    if args.len() != 2 {
                        return Err(OxirsError::Query(
                            "STRENDS requires exactly two arguments".to_string(),
                        ));
                    }
                    let s = term_to_string(&evaluate_expression(&args[0], binding)?);
                    let suffix = term_to_string(&evaluate_expression(&args[1], binding)?);
                    let result = if s.ends_with(&suffix) {
                        "true"
                    } else {
                        "false"
                    };
                    Ok(Term::from(Literal::new(result.to_string())))
                }
                // ========================================
                // SPARQL 1.2 RDF-star Built-in Functions
                // ========================================
                //
                // These functions provide full support for RDF-star quoted triples,
                // enabling meta-statements about statements.
                //
                // Specification: https://w3c.github.io/rdf-star/cg-spec/
                // Performance: All operations are O(1)
                //
                // Functions:
                // - TRIPLE(s, p, o) → Creates a quoted triple
                // - SUBJECT(qt) → Extracts subject from quoted triple
                // - PREDICATE(qt) → Extracts predicate from quoted triple
                // - OBJECT(qt) → Extracts object from quoted triple
                // - isTRIPLE(term) → Tests if term is a quoted triple
                // ========================================

                // TRIPLE(subject, predicate, object) → quoted triple
                //
                // Creates a quoted triple from three terms. Supports nested
                // quoted triples for meta-meta-statements.
                //
                // Example SPARQL:
                //   BIND(TRIPLE(?s, ?p, ?o) AS ?qt)
                //   BIND(TRIPLE(TRIPLE(?s1, ?p1, ?o1), ?p2, ?o2) AS ?nested)
                "TRIPLE" => {
                    if args.len() != 3 {
                        return Err(OxirsError::Query(
                            "TRIPLE requires exactly three arguments (subject, predicate, object)"
                                .to_string(),
                        ));
                    }
                    let subject_term = evaluate_expression(&args[0], binding)?;
                    let predicate_term = evaluate_expression(&args[1], binding)?;
                    let object_term = evaluate_expression(&args[2], binding)?;

                    // Convert Term to Subject/Predicate/Object
                    use crate::model::{Object, Predicate, QuotedTriple, Subject, Triple};
                    let subject: Subject = match subject_term {
                        Term::NamedNode(n) => Subject::NamedNode(n),
                        Term::BlankNode(b) => Subject::BlankNode(b),
                        Term::Variable(v) => Subject::Variable(v),
                        Term::QuotedTriple(qt) => Subject::QuotedTriple(qt),
                        _ => {
                            return Err(OxirsError::Query(
                                "TRIPLE subject must be NamedNode, BlankNode, Variable, or QuotedTriple".to_string(),
                            ))
                        }
                    };

                    let predicate: Predicate = match predicate_term {
                        Term::NamedNode(n) => Predicate::NamedNode(n),
                        Term::Variable(v) => Predicate::Variable(v),
                        _ => {
                            return Err(OxirsError::Query(
                                "TRIPLE predicate must be NamedNode or Variable".to_string(),
                            ))
                        }
                    };

                    let object: Object = match object_term {
                        Term::NamedNode(n) => Object::NamedNode(n),
                        Term::BlankNode(b) => Object::BlankNode(b),
                        Term::Literal(l) => Object::Literal(l),
                        Term::Variable(v) => Object::Variable(v),
                        Term::QuotedTriple(qt) => Object::QuotedTriple(qt),
                    };

                    // Create a Triple and wrap in QuotedTriple
                    let triple = Triple::new(subject, predicate, object);
                    let quoted = QuotedTriple::new(triple);
                    Ok(Term::QuotedTriple(Box::new(quoted)))
                }
                // SUBJECT(quotedTriple) → term
                //
                // Extracts the subject from a quoted triple.
                //
                // Example SPARQL:
                //   SELECT * WHERE {
                //     ?qt a ex:Statement .
                //     BIND(SUBJECT(?qt) AS ?subj)
                //   }
                "SUBJECT" => {
                    if args.len() != 1 {
                        return Err(OxirsError::Query(
                            "SUBJECT requires exactly one argument (a quoted triple)".to_string(),
                        ));
                    }
                    let term = evaluate_expression(&args[0], binding)?;
                    match term {
                        Term::QuotedTriple(triple) => {
                            // Convert Subject to Term
                            use crate::model::Subject;
                            let subject = triple.subject();
                            let term = match subject {
                                Subject::NamedNode(n) => Term::NamedNode(n.clone()),
                                Subject::BlankNode(b) => Term::BlankNode(b.clone()),
                                Subject::Variable(v) => Term::Variable(v.clone()),
                                Subject::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
                            };
                            Ok(term)
                        }
                        _ => Err(OxirsError::Query(
                            "SUBJECT function requires a quoted triple argument".to_string(),
                        )),
                    }
                }
                // PREDICATE(quotedTriple) → term
                //
                // Extracts the predicate from a quoted triple.
                //
                // Example SPARQL:
                //   SELECT ?qt WHERE {
                //     ?qt ?p ?o .
                //     FILTER(PREDICATE(?qt) = ex:hasAge)
                //   }
                "PREDICATE" => {
                    if args.len() != 1 {
                        return Err(OxirsError::Query(
                            "PREDICATE requires exactly one argument (a quoted triple)".to_string(),
                        ));
                    }
                    let term = evaluate_expression(&args[0], binding)?;
                    match term {
                        Term::QuotedTriple(triple) => {
                            // Convert Predicate to Term
                            use crate::model::Predicate;
                            let predicate = triple.predicate();
                            let term = match predicate {
                                Predicate::NamedNode(n) => Term::NamedNode(n.clone()),
                                Predicate::Variable(v) => Term::Variable(v.clone()),
                            };
                            Ok(term)
                        }
                        _ => Err(OxirsError::Query(
                            "PREDICATE function requires a quoted triple argument".to_string(),
                        )),
                    }
                }
                // OBJECT(quotedTriple) → term
                //
                // Extracts the object from a quoted triple.
                //
                // Example SPARQL:
                //   SELECT ?qt WHERE {
                //     ?qt ?p ?confidence .
                //     FILTER(OBJECT(?qt) = "high"^^xsd:string)
                //   }
                "OBJECT" => {
                    if args.len() != 1 {
                        return Err(OxirsError::Query(
                            "OBJECT requires exactly one argument (a quoted triple)".to_string(),
                        ));
                    }
                    let term = evaluate_expression(&args[0], binding)?;
                    match term {
                        Term::QuotedTriple(triple) => {
                            // Convert Object to Term
                            use crate::model::Object;
                            let object = triple.object();
                            let term = match object {
                                Object::NamedNode(n) => Term::NamedNode(n.clone()),
                                Object::BlankNode(b) => Term::BlankNode(b.clone()),
                                Object::Literal(l) => Term::Literal(l.clone()),
                                Object::Variable(v) => Term::Variable(v.clone()),
                                Object::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
                            };
                            Ok(term)
                        }
                        _ => Err(OxirsError::Query(
                            "OBJECT function requires a quoted triple argument".to_string(),
                        )),
                    }
                }
                // isTRIPLE(term) → boolean
                //
                // Tests whether a term is a quoted triple.
                //
                // Example SPARQL:
                //   SELECT * WHERE {
                //     ?x ?p ?o .
                //     FILTER(isTRIPLE(?x))
                //   }
                "ISTRIPLE" => {
                    if args.len() != 1 {
                        return Err(OxirsError::Query(
                            "isTRIPLE requires exactly one argument".to_string(),
                        ));
                    }
                    let term = evaluate_expression(&args[0], binding)?;
                    let result = if matches!(term, Term::QuotedTriple(_)) {
                        "true"
                    } else {
                        "false"
                    };
                    Ok(Term::from(Literal::new(result.to_string())))
                }
                _ => Err(OxirsError::Query(format!("Unsupported function: {}", name))),
            }
        }
    }
}

/// Convert a term to a number for arithmetic
pub fn term_to_number(term: &Term) -> Result<f64> {
    if let Term::Literal(lit) = term {
        lit.value()
            .parse::<f64>()
            .map_err(|_| OxirsError::Query(format!("Cannot convert to number: {}", lit.value())))
    } else {
        Err(OxirsError::Query("Expected numeric literal".to_string()))
    }
}

/// Convert a term to a string
pub fn term_to_string(term: &Term) -> String {
    match term {
        Term::NamedNode(node) => node.to_string(),
        Term::BlankNode(node) => node.to_string(),
        Term::Literal(lit) => lit.value().to_string(),
        Term::Variable(var) => var.to_string(),
        Term::QuotedTriple(triple) => format!("<< {} >>", triple),
    }
}

/// Apply BIND expressions to results
pub fn apply_bind_expressions(
    results: Vec<VariableBinding>,
    binds: &[BindExpression],
) -> Result<Vec<VariableBinding>> {
    if binds.is_empty() {
        return Ok(results);
    }

    let mut new_results = Vec::new();

    for binding in results {
        let mut new_binding = binding.clone();

        // Apply each BIND expression
        for bind_expr in binds {
            match evaluate_expression(&bind_expr.expression, &binding) {
                Ok(value) => {
                    new_binding.bind(bind_expr.variable.clone(), value);
                }
                Err(_) => {
                    // If evaluation fails, skip this binding
                    continue;
                }
            }
        }

        new_results.push(new_binding);
    }

    Ok(new_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode, Object, Predicate, Subject, Triple};

    #[test]
    fn test_sparql_12_triple_function() {
        // Test TRIPLE() function - creates a quoted triple
        let mut binding = VariableBinding::new();

        // Set up test data
        let alice = Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
        let knows = Term::NamedNode(NamedNode::new("http://example.org/knows").unwrap());
        let bob = Term::NamedNode(NamedNode::new("http://example.org/bob").unwrap());

        binding.bind("s".to_string(), alice.clone());
        binding.bind("p".to_string(), knows.clone());
        binding.bind("o".to_string(), bob.clone());

        // Create TRIPLE(?s, ?p, ?o) expression
        let expr = Expression::FunctionCall {
            name: "TRIPLE".to_string(),
            args: vec![
                Expression::Variable("?s".to_string()),
                Expression::Variable("?p".to_string()),
                Expression::Variable("?o".to_string()),
            ],
        };

        let result = evaluate_expression(&expr, &binding);
        assert!(result.is_ok());

        if let Ok(Term::QuotedTriple(qt)) = result {
            assert_eq!(
                qt.subject(),
                &Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap())
            );
            assert_eq!(
                qt.predicate(),
                &Predicate::NamedNode(NamedNode::new("http://example.org/knows").unwrap())
            );
        } else {
            panic!("Expected QuotedTriple");
        }
    }

    #[test]
    fn test_sparql_12_subject_function() {
        // Test SUBJECT() function - extracts subject from quoted triple
        let mut binding = VariableBinding::new();

        // Create a quoted triple
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
        let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/age").unwrap());
        let object = Object::Literal(Literal::new("30"));

        let triple = Triple::new(subject, predicate, object);
        let quoted = crate::model::QuotedTriple::new(triple);

        binding.bind("qt".to_string(), Term::QuotedTriple(Box::new(quoted)));

        // Create SUBJECT(?qt) expression
        let expr = Expression::FunctionCall {
            name: "SUBJECT".to_string(),
            args: vec![Expression::Variable("?qt".to_string())],
        };

        let result = evaluate_expression(&expr, &binding);
        assert!(result.is_ok());

        if let Ok(Term::NamedNode(n)) = result {
            assert_eq!(n.as_str(), "http://example.org/alice");
        } else {
            panic!("Expected NamedNode");
        }
    }

    #[test]
    fn test_sparql_12_predicate_function() {
        // Test PREDICATE() function - extracts predicate from quoted triple
        let mut binding = VariableBinding::new();

        // Create a quoted triple
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
        let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/age").unwrap());
        let object = Object::Literal(Literal::new("30"));

        let triple = Triple::new(subject, predicate, object);
        let quoted = crate::model::QuotedTriple::new(triple);

        binding.bind("qt".to_string(), Term::QuotedTriple(Box::new(quoted)));

        // Create PREDICATE(?qt) expression
        let expr = Expression::FunctionCall {
            name: "PREDICATE".to_string(),
            args: vec![Expression::Variable("?qt".to_string())],
        };

        let result = evaluate_expression(&expr, &binding);
        assert!(result.is_ok());

        if let Ok(Term::NamedNode(n)) = result {
            assert_eq!(n.as_str(), "http://example.org/age");
        } else {
            panic!("Expected NamedNode");
        }
    }

    #[test]
    fn test_sparql_12_object_function() {
        // Test OBJECT() function - extracts object from quoted triple
        let mut binding = VariableBinding::new();

        // Create a quoted triple
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
        let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/age").unwrap());
        let object = Object::Literal(Literal::new("30"));

        let triple = Triple::new(subject, predicate, object);
        let quoted = crate::model::QuotedTriple::new(triple);

        binding.bind("qt".to_string(), Term::QuotedTriple(Box::new(quoted)));

        // Create OBJECT(?qt) expression
        let expr = Expression::FunctionCall {
            name: "OBJECT".to_string(),
            args: vec![Expression::Variable("?qt".to_string())],
        };

        let result = evaluate_expression(&expr, &binding);
        assert!(result.is_ok());

        if let Ok(Term::Literal(lit)) = result {
            assert_eq!(lit.value(), "30");
        } else {
            panic!("Expected Literal");
        }
    }

    #[test]
    fn test_sparql_12_istriple_function_true() {
        // Test isTRIPLE() function - returns true for quoted triple
        let mut binding = VariableBinding::new();

        // Create a quoted triple
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
        let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/age").unwrap());
        let object = Object::Literal(Literal::new("30"));

        let triple = Triple::new(subject, predicate, object);
        let quoted = crate::model::QuotedTriple::new(triple);

        binding.bind("qt".to_string(), Term::QuotedTriple(Box::new(quoted)));

        // Create isTRIPLE(?qt) expression
        let expr = Expression::FunctionCall {
            name: "ISTRIPLE".to_string(),
            args: vec![Expression::Variable("?qt".to_string())],
        };

        let result = evaluate_expression(&expr, &binding);
        assert!(result.is_ok());

        if let Ok(Term::Literal(lit)) = result {
            assert_eq!(lit.value(), "true");
        } else {
            panic!("Expected true Literal");
        }
    }

    #[test]
    fn test_sparql_12_istriple_function_false() {
        // Test isTRIPLE() function - returns false for non-quoted triple
        let mut binding = VariableBinding::new();

        let alice = Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
        binding.bind("n".to_string(), alice);

        // Create isTRIPLE(?n) expression
        let expr = Expression::FunctionCall {
            name: "ISTRIPLE".to_string(),
            args: vec![Expression::Variable("?n".to_string())],
        };

        let result = evaluate_expression(&expr, &binding);
        assert!(result.is_ok());

        if let Ok(Term::Literal(lit)) = result {
            assert_eq!(lit.value(), "false");
        } else {
            panic!("Expected false Literal");
        }
    }

    #[test]
    fn test_sparql_12_nested_quoted_triples() {
        // Test nested quoted triples: TRIPLE(TRIPLE(?s, ?p, ?o), ?p2, ?o2)
        let mut binding = VariableBinding::new();

        let alice = Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
        let age = Term::NamedNode(NamedNode::new("http://example.org/age").unwrap());
        let thirty = Term::Literal(Literal::new("30"));
        let confidence = Term::NamedNode(NamedNode::new("http://example.org/confidence").unwrap());
        let high = Term::Literal(Literal::new("high"));

        binding.bind("s".to_string(), alice);
        binding.bind("p".to_string(), age);
        binding.bind("o".to_string(), thirty);
        binding.bind("p2".to_string(), confidence);
        binding.bind("o2".to_string(), high);

        // Create inner TRIPLE(?s, ?p, ?o)
        let inner_expr = Expression::FunctionCall {
            name: "TRIPLE".to_string(),
            args: vec![
                Expression::Variable("?s".to_string()),
                Expression::Variable("?p".to_string()),
                Expression::Variable("?o".to_string()),
            ],
        };

        // Create outer TRIPLE(inner_triple, ?p2, ?o2)
        let outer_expr = Expression::FunctionCall {
            name: "TRIPLE".to_string(),
            args: vec![
                inner_expr,
                Expression::Variable("?p2".to_string()),
                Expression::Variable("?o2".to_string()),
            ],
        };

        let result = evaluate_expression(&outer_expr, &binding);
        assert!(result.is_ok());

        // The result should be a quoted triple with a quoted triple as subject
        if let Ok(Term::QuotedTriple(outer_qt)) = result {
            assert!(matches!(outer_qt.subject(), Subject::QuotedTriple(_)));
        } else {
            panic!("Expected nested QuotedTriple");
        }
    }

    #[test]
    fn test_sparql_12_rdf_star_composition() {
        // Test: Create a triple, then extract its parts
        let mut binding = VariableBinding::new();

        let alice = Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
        let knows = Term::NamedNode(NamedNode::new("http://example.org/knows").unwrap());
        let bob = Term::NamedNode(NamedNode::new("http://example.org/bob").unwrap());

        binding.bind("s".to_string(), alice.clone());
        binding.bind("p".to_string(), knows.clone());
        binding.bind("o".to_string(), bob.clone());

        // Create TRIPLE(?s, ?p, ?o)
        let triple_expr = Expression::FunctionCall {
            name: "TRIPLE".to_string(),
            args: vec![
                Expression::Variable("?s".to_string()),
                Expression::Variable("?p".to_string()),
                Expression::Variable("?o".to_string()),
            ],
        };

        let quoted_triple = evaluate_expression(&triple_expr, &binding).unwrap();
        binding.bind("qt".to_string(), quoted_triple);

        // Extract subject
        let subject_expr = Expression::FunctionCall {
            name: "SUBJECT".to_string(),
            args: vec![Expression::Variable("?qt".to_string())],
        };
        let extracted_subject = evaluate_expression(&subject_expr, &binding).unwrap();

        // Verify subject matches original
        assert_eq!(extracted_subject, alice);

        // Extract predicate
        let predicate_expr = Expression::FunctionCall {
            name: "PREDICATE".to_string(),
            args: vec![Expression::Variable("?qt".to_string())],
        };
        let extracted_predicate = evaluate_expression(&predicate_expr, &binding).unwrap();

        // Verify predicate matches original
        assert_eq!(extracted_predicate, knows);

        // Extract object
        let object_expr = Expression::FunctionCall {
            name: "OBJECT".to_string(),
            args: vec![Expression::Variable("?qt".to_string())],
        };
        let extracted_object = evaluate_expression(&object_expr, &binding).unwrap();

        // Verify object matches original
        assert_eq!(extracted_object, bob);
    }
}
