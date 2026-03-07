use super::*;

#[test]
fn test_literal_creation_simple() {
    let lit = Literal {
        value: "test".to_string(),
        language: None,
        datatype: None,
    };

    assert_eq!(lit.value, "test");
    assert!(lit.language.is_none());
    assert!(lit.datatype.is_none());
    assert_eq!(lit.to_string(), "\"test\"");
}

#[test]
fn test_literal_with_language() {
    let lit = Literal::with_language("bonjour".to_string(), "fr".to_string());

    assert_eq!(lit.value, "bonjour");
    assert_eq!(lit.language.as_deref(), Some("fr"));
    assert!(lit.datatype.is_none());
    assert_eq!(lit.to_string(), "\"bonjour\"@fr");
}

#[test]
fn test_literal_with_datatype() {
    let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap();
    let lit = Literal {
        value: "42".to_string(),
        language: None,
        datatype: Some(datatype.clone()),
    };

    assert_eq!(lit.value, "42");
    assert!(lit.language.is_none());
    assert_eq!(lit.datatype, Some(datatype));
}

#[test]
fn test_literal_conversion_to_core() {
    let lit = Literal::with_language("hello".to_string(), "en".to_string());
    let core_lit: CoreLiteral = lit.into();

    assert_eq!(core_lit.value(), "hello");
    assert_eq!(core_lit.language(), Some("en"));
}

#[test]
fn test_literal_conversion_from_core() {
    let core_lit = CoreLiteral::new_language_tagged_literal("hola", "es".to_string()).unwrap();
    let lit: Literal = core_lit.into();

    assert_eq!(lit.value, "hola");
    assert_eq!(lit.language.as_deref(), Some("es"));
}

#[test]
fn test_term_variable() {
    let var = Variable::new("x").unwrap();
    let term = Term::Variable(var.clone());

    // Variable Display includes the "?" prefix
    assert!(term.to_string().contains("x"));
    if let Term::Variable(v) = term {
        assert_eq!(v.name(), "x");
    } else {
        panic!("Expected Variable term");
    }
}

#[test]
fn test_term_iri() {
    let iri = NamedNode::new("http://example.org/resource").unwrap();
    let term = Term::Iri(iri.clone());

    assert!(term.to_string().contains("example.org"));
    if let Term::Iri(i) = term {
        assert_eq!(i, iri);
    } else {
        panic!("Expected IRI term");
    }
}

#[test]
fn test_term_literal() {
    let lit = Literal {
        value: "test value".to_string(),
        language: None,
        datatype: None,
    };
    let term = Term::Literal(lit.clone());

    assert_eq!(term.to_string(), "\"test value\"");
    if let Term::Literal(l) = term {
        assert_eq!(l.value, "test value");
    } else {
        panic!("Expected Literal term");
    }
}

#[test]
fn test_term_blank_node() {
    let term = Term::BlankNode("b0".to_string());

    assert_eq!(term.to_string(), "_:b0");
    if let Term::BlankNode(id) = term {
        assert_eq!(id, "b0");
    } else {
        panic!("Expected BlankNode term");
    }
}

#[test]
fn test_term_ordering() {
    let var = Term::Variable(Variable::new("x").unwrap());
    let iri = Term::Iri(NamedNode::new("http://example.org/a").unwrap());
    let lit = Term::Literal(Literal {
        value: "a".to_string(),
        language: None,
        datatype: None,
    });
    let blank = Term::BlankNode("b0".to_string());

    // Test that terms can be compared (used in sorting, hash maps, etc.)
    assert!(var != iri); // Terms are comparable
    assert!(lit != blank);
}

#[test]
fn test_triple_pattern_creation() {
    let subject = Term::Variable(Variable::new("s").unwrap());
    let predicate = Term::Iri(NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap());
    let object = Term::Literal(Literal {
        value: "Alice".to_string(),
        language: None,
        datatype: None,
    });

    let triple = TriplePattern {
        subject,
        predicate,
        object,
    };

    assert!(matches!(triple.subject, Term::Variable(_)));
    assert!(matches!(triple.predicate, Term::Iri(_)));
    assert!(matches!(triple.object, Term::Literal(_)));
}

#[test]
fn test_binding_creation() {
    let var = Variable::new("x").unwrap();
    let value = Term::Literal(Literal {
        value: "42".to_string(),
        language: None,
        datatype: None,
    });

    let mut binding: Binding = HashMap::new();
    binding.insert(var.clone(), value.clone());

    assert_eq!(binding.len(), 1);
    assert_eq!(binding.get(&var), Some(&value));
}

#[test]
fn test_solution_creation() {
    let var1 = Variable::new("x").unwrap();
    let var2 = Variable::new("y").unwrap();

    let mut binding1 = HashMap::new();
    binding1.insert(
        var1.clone(),
        Term::Literal(Literal {
            value: "1".to_string(),
            language: None,
            datatype: None,
        }),
    );

    let mut binding2 = HashMap::new();
    binding2.insert(
        var2.clone(),
        Term::Literal(Literal {
            value: "2".to_string(),
            language: None,
            datatype: None,
        }),
    );

    let solution: Solution = vec![binding1, binding2];

    assert_eq!(solution.len(), 2);
    assert!(solution[0].contains_key(&var1));
    assert!(solution[1].contains_key(&var2));
}

#[test]
fn test_algebra_bgp() {
    let patterns = vec![TriplePattern {
        subject: Term::Variable(Variable::new("s").unwrap()),
        predicate: Term::Iri(
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
        ),
        object: Term::Iri(NamedNode::new("http://xmlns.com/foaf/0.1/Person").unwrap()),
    }];

    let bgp = Algebra::Bgp(patterns.clone());

    if let Algebra::Bgp(p) = bgp {
        assert_eq!(p.len(), 1);
        assert!(matches!(p[0].subject, Term::Variable(_)));
    } else {
        panic!("Expected BGP algebra");
    }
}

#[test]
fn test_algebra_join() {
    let left = Box::new(Algebra::Bgp(vec![]));
    let right = Box::new(Algebra::Bgp(vec![]));

    let join = Algebra::Join { left, right };

    if let Algebra::Join { .. } = join {
        // Successfully created join
    } else {
        panic!("Expected Join algebra");
    }
}

#[test]
fn test_algebra_filter() {
    let pattern = Box::new(Algebra::Bgp(vec![]));

    let var = Variable::new("x").unwrap();
    let filter_expr = Expression::Variable(var);

    let filter = Algebra::Filter {
        pattern,
        condition: filter_expr,
    };

    if let Algebra::Filter { condition, .. } = filter {
        assert!(matches!(condition, Expression::Variable(_)));
    } else {
        panic!("Expected Filter algebra");
    }
}

#[test]
fn test_algebra_union() {
    let left = Box::new(Algebra::Bgp(vec![]));
    let right = Box::new(Algebra::Bgp(vec![]));

    let union = Algebra::Union { left, right };

    if let Algebra::Union { .. } = union {
        // Successfully created union
    } else {
        panic!("Expected Union algebra");
    }
}

#[test]
fn test_algebra_left_join() {
    let left = Box::new(Algebra::Bgp(vec![]));
    let right = Box::new(Algebra::Bgp(vec![]));

    let left_join = Algebra::LeftJoin {
        left,
        right,
        filter: None,
    };

    if let Algebra::LeftJoin { filter, .. } = left_join {
        assert!(filter.is_none());
    } else {
        panic!("Expected LeftJoin algebra");
    }
}

#[test]
fn test_property_path_sequence() {
    let path1 = PropertyPath::Iri(NamedNode::new("http://example.org/p1").unwrap());
    let path2 = PropertyPath::Iri(NamedNode::new("http://example.org/p2").unwrap());

    let seq = PropertyPath::Sequence(Box::new(path1), Box::new(path2));

    if let PropertyPath::Sequence(left, right) = seq {
        assert!(matches!(*left, PropertyPath::Iri(_)));
        assert!(matches!(*right, PropertyPath::Iri(_)));
    } else {
        panic!("Expected Sequence property path");
    }
}

#[test]
fn test_property_path_alternative() {
    let path1 = PropertyPath::Iri(NamedNode::new("http://example.org/p1").unwrap());
    let path2 = PropertyPath::Iri(NamedNode::new("http://example.org/p2").unwrap());

    let alt = PropertyPath::Alternative(Box::new(path1), Box::new(path2));

    if let PropertyPath::Alternative(left, right) = alt {
        assert!(matches!(*left, PropertyPath::Iri(_)));
        assert!(matches!(*right, PropertyPath::Iri(_)));
    } else {
        panic!("Expected Alternative property path");
    }
}

#[test]
fn test_property_path_zero_or_more() {
    let path = PropertyPath::Iri(NamedNode::new("http://example.org/p").unwrap());
    let star = PropertyPath::ZeroOrMore(Box::new(path));

    if let PropertyPath::ZeroOrMore(inner) = star {
        assert!(matches!(*inner, PropertyPath::Iri(_)));
    } else {
        panic!("Expected ZeroOrMore property path");
    }
}

#[test]
fn test_filter_selectivity_estimation() {
    let var = Variable::new("x").unwrap();

    // Equal operator - very selective
    let equal_expr = Expression::Binary {
        op: BinaryOperator::Equal,
        left: Box::new(Expression::Variable(var.clone())),
        right: Box::new(Expression::Literal(Literal {
            value: "42".to_string(),
            language: None,
            datatype: None,
        })),
    };
    let selectivity = estimate_filter_selectivity(&equal_expr);
    assert_eq!(selectivity, 0.01);

    // Not equal - not selective
    let not_equal_expr = Expression::Binary {
        op: BinaryOperator::NotEqual,
        left: Box::new(Expression::Variable(var.clone())),
        right: Box::new(Expression::Literal(Literal {
            value: "42".to_string(),
            language: None,
            datatype: None,
        })),
    };
    let selectivity = estimate_filter_selectivity(&not_equal_expr);
    assert_eq!(selectivity, 0.99);

    // Range comparison
    let less_expr = Expression::Binary {
        op: BinaryOperator::Less,
        left: Box::new(Expression::Variable(var)),
        right: Box::new(Expression::Literal(Literal {
            value: "100".to_string(),
            language: None,
            datatype: None,
        })),
    };
    let selectivity = estimate_filter_selectivity(&less_expr);
    assert_eq!(selectivity, 0.33);
}

#[test]
fn test_evaluation_context_default() {
    let ctx = EvaluationContext::default();

    assert!(ctx.bindings.is_empty());
    assert!(ctx.dataset.is_none());
    assert!(ctx.options.is_empty());
}

#[test]
fn test_evaluation_context_with_bindings() {
    let mut ctx = EvaluationContext::default();
    let var = Variable::new("x").unwrap();
    let value = Term::Literal(Literal {
        value: "test".to_string(),
        language: None,
        datatype: None,
    });

    ctx.bindings.insert(var.clone(), value.clone());

    assert_eq!(ctx.bindings.len(), 1);
    assert_eq!(ctx.bindings.get(&var), Some(&value));
}

#[test]
fn test_order_condition() {
    let var = Variable::new("x").unwrap();
    let order = OrderCondition {
        expr: Expression::Variable(var),
        ascending: true,
    };

    assert!(order.ascending);
    assert!(matches!(order.expr, Expression::Variable(_)));
}

#[test]
fn test_group_condition() {
    let var = Variable::new("x").unwrap();
    let alias = Variable::new("y").unwrap();
    let group = GroupCondition {
        expr: Expression::Variable(var),
        alias: Some(alias.clone()),
    };

    assert!(group.alias.is_some());
    assert_eq!(group.alias.unwrap().name(), "y");
    assert!(matches!(group.expr, Expression::Variable(_)));
}

// ---- SPARQL Algebra SELECT/CONSTRUCT/DESCRIBE/ASK structure tests ----

#[test]
fn test_algebra_project_select() {
    let var_s = Variable::new("s").expect("valid variable");
    let var_p = Variable::new("p").expect("valid variable");
    let var_o = Variable::new("o").expect("valid variable");

    let bgp = Algebra::Bgp(vec![TriplePattern {
        subject: Term::Variable(var_s.clone()),
        predicate: Term::Variable(var_p.clone()),
        object: Term::Variable(var_o.clone()),
    }]);

    let projection = Algebra::Project {
        pattern: Box::new(bgp),
        variables: vec![var_s.clone(), var_p.clone()],
    };

    if let Algebra::Project { variables, .. } = &projection {
        assert_eq!(variables.len(), 2);
        assert!(variables.contains(&var_s));
        assert!(variables.contains(&var_p));
        assert!(!variables.contains(&var_o));
    } else {
        panic!("Expected Project algebra");
    }
}

#[test]
fn test_algebra_distinct() {
    let bgp = Algebra::Bgp(vec![]);
    let distinct = Algebra::Distinct {
        pattern: Box::new(bgp),
    };

    assert!(matches!(distinct, Algebra::Distinct { .. }));
}

#[test]
fn test_algebra_reduced() {
    let bgp = Algebra::Bgp(vec![]);
    let reduced = Algebra::Reduced {
        pattern: Box::new(bgp),
    };

    assert!(matches!(reduced, Algebra::Reduced { .. }));
}

#[test]
fn test_algebra_slice_limit_offset() {
    let bgp = Algebra::Bgp(vec![]);
    let sliced = Algebra::Slice {
        pattern: Box::new(bgp),
        offset: Some(10),
        limit: Some(25),
    };

    if let Algebra::Slice { offset, limit, .. } = sliced {
        assert_eq!(offset, Some(10));
        assert_eq!(limit, Some(25));
    } else {
        panic!("Expected Slice algebra");
    }
}

#[test]
fn test_algebra_slice_limit_only() {
    let bgp = Algebra::Bgp(vec![]);
    let sliced = Algebra::Slice {
        pattern: Box::new(bgp),
        offset: None,
        limit: Some(100),
    };

    if let Algebra::Slice { offset, limit, .. } = sliced {
        assert!(offset.is_none());
        assert_eq!(limit, Some(100));
    } else {
        panic!("Expected Slice algebra");
    }
}

#[test]
fn test_algebra_order_by_ascending() {
    let var_age = Variable::new("age").expect("valid variable");
    let bgp = Algebra::Bgp(vec![]);
    let order_condition = OrderCondition {
        expr: Expression::Variable(var_age.clone()),
        ascending: true,
    };
    let ordered = Algebra::OrderBy {
        pattern: Box::new(bgp),
        conditions: vec![order_condition],
    };

    if let Algebra::OrderBy { conditions, .. } = ordered {
        assert_eq!(conditions.len(), 1);
        assert!(conditions[0].ascending);
        assert!(matches!(&conditions[0].expr, Expression::Variable(v) if v == &var_age));
    } else {
        panic!("Expected OrderBy algebra");
    }
}

#[test]
fn test_algebra_order_by_descending() {
    let var_name = Variable::new("name").expect("valid variable");
    let bgp = Algebra::Bgp(vec![]);
    let order_condition = OrderCondition {
        expr: Expression::Variable(var_name.clone()),
        ascending: false,
    };
    let ordered = Algebra::OrderBy {
        pattern: Box::new(bgp),
        conditions: vec![order_condition],
    };

    if let Algebra::OrderBy { conditions, .. } = ordered {
        assert!(!conditions[0].ascending);
    } else {
        panic!("Expected OrderBy algebra");
    }
}

#[test]
fn test_algebra_order_by_multiple_conditions() {
    let var_a = Variable::new("a").expect("valid variable");
    let var_b = Variable::new("b").expect("valid variable");
    let bgp = Algebra::Bgp(vec![]);
    let ordered = Algebra::OrderBy {
        pattern: Box::new(bgp),
        conditions: vec![
            OrderCondition {
                expr: Expression::Variable(var_a),
                ascending: true,
            },
            OrderCondition {
                expr: Expression::Variable(var_b),
                ascending: false,
            },
        ],
    };

    if let Algebra::OrderBy { conditions, .. } = ordered {
        assert_eq!(conditions.len(), 2);
        assert!(conditions[0].ascending);
        assert!(!conditions[1].ascending);
    } else {
        panic!("Expected OrderBy algebra");
    }
}

#[test]
fn test_algebra_group_by_with_count_aggregate() {
    let var_type = Variable::new("type").expect("valid variable");
    let var_count = Variable::new("count").expect("valid variable");
    let bgp = Algebra::Bgp(vec![]);

    let group_condition = GroupCondition {
        expr: Expression::Variable(var_type.clone()),
        alias: None,
    };

    let aggregate = Aggregate::Count {
        distinct: false,
        expr: None,
    };

    let grouped = Algebra::Group {
        pattern: Box::new(bgp),
        variables: vec![group_condition],
        aggregates: vec![(var_count.clone(), aggregate)],
    };

    if let Algebra::Group {
        variables,
        aggregates,
        ..
    } = grouped
    {
        assert_eq!(variables.len(), 1);
        assert_eq!(aggregates.len(), 1);
        assert_eq!(aggregates[0].0, var_count);
        assert!(matches!(
            aggregates[0].1,
            Aggregate::Count {
                distinct: false,
                expr: None
            }
        ));
    } else {
        panic!("Expected Group algebra");
    }
}

#[test]
fn test_algebra_group_by_with_sum_aggregate() {
    let var_category = Variable::new("category").expect("valid variable");
    let var_total = Variable::new("total").expect("valid variable");
    let var_price = Variable::new("price").expect("valid variable");
    let bgp = Algebra::Bgp(vec![]);

    let group_condition = GroupCondition {
        expr: Expression::Variable(var_category),
        alias: None,
    };

    let sum_aggregate = Aggregate::Sum {
        distinct: false,
        expr: Expression::Variable(var_price),
    };

    let grouped = Algebra::Group {
        pattern: Box::new(bgp),
        variables: vec![group_condition],
        aggregates: vec![(var_total, sum_aggregate)],
    };

    if let Algebra::Group { aggregates, .. } = grouped {
        assert!(matches!(aggregates[0].1, Aggregate::Sum { .. }));
    } else {
        panic!("Expected Group algebra");
    }
}

#[test]
fn test_algebra_group_by_with_avg_aggregate() {
    let var_result = Variable::new("result").expect("valid variable");
    let var_val = Variable::new("val").expect("valid variable");
    let bgp = Algebra::Bgp(vec![]);

    let avg_aggregate = Aggregate::Avg {
        distinct: false,
        expr: Expression::Variable(var_val),
    };

    let grouped = Algebra::Group {
        pattern: Box::new(bgp),
        variables: vec![],
        aggregates: vec![(var_result, avg_aggregate)],
    };

    if let Algebra::Group { aggregates, .. } = grouped {
        assert!(matches!(aggregates[0].1, Aggregate::Avg { .. }));
    } else {
        panic!("Expected Group algebra");
    }
}

#[test]
fn test_algebra_group_by_with_min_max_aggregates() {
    let var_min = Variable::new("minval").expect("valid variable");
    let var_max = Variable::new("maxval").expect("valid variable");
    let var_v = Variable::new("v").expect("valid variable");
    let bgp = Algebra::Bgp(vec![]);

    let min_agg = Aggregate::Min {
        distinct: false,
        expr: Expression::Variable(var_v.clone()),
    };
    let max_agg = Aggregate::Max {
        distinct: false,
        expr: Expression::Variable(var_v),
    };

    let grouped = Algebra::Group {
        pattern: Box::new(bgp),
        variables: vec![],
        aggregates: vec![(var_min, min_agg), (var_max, max_agg)],
    };

    if let Algebra::Group { aggregates, .. } = grouped {
        assert_eq!(aggregates.len(), 2);
        assert!(matches!(aggregates[0].1, Aggregate::Min { .. }));
        assert!(matches!(aggregates[1].1, Aggregate::Max { .. }));
    } else {
        panic!("Expected Group algebra");
    }
}

#[test]
fn test_algebra_group_concat_aggregate() {
    let var_result = Variable::new("names").expect("valid variable");
    let var_name = Variable::new("name").expect("valid variable");
    let bgp = Algebra::Bgp(vec![]);

    let group_concat = Aggregate::GroupConcat {
        distinct: false,
        expr: Expression::Variable(var_name),
        separator: Some(", ".to_string()),
    };

    let grouped = Algebra::Group {
        pattern: Box::new(bgp),
        variables: vec![],
        aggregates: vec![(var_result, group_concat)],
    };

    if let Algebra::Group { aggregates, .. } = grouped {
        if let Aggregate::GroupConcat { separator, .. } = &aggregates[0].1 {
            assert_eq!(separator.as_deref(), Some(", "));
        } else {
            panic!("Expected GroupConcat aggregate");
        }
    } else {
        panic!("Expected Group algebra");
    }
}

#[test]
fn test_algebra_having_filter() {
    let var_count = Variable::new("count").expect("valid variable");
    let bgp = Algebra::Bgp(vec![]);
    let group = Algebra::Group {
        pattern: Box::new(bgp),
        variables: vec![],
        aggregates: vec![(
            var_count.clone(),
            Aggregate::Count {
                distinct: false,
                expr: None,
            },
        )],
    };

    // HAVING count > 5
    let having_condition = Expression::Binary {
        op: BinaryOperator::Greater,
        left: Box::new(Expression::Variable(var_count)),
        right: Box::new(Expression::Literal(Literal::integer(5))),
    };

    let having = Algebra::Having {
        pattern: Box::new(group),
        condition: having_condition,
    };

    if let Algebra::Having { condition, .. } = having {
        assert!(matches!(
            condition,
            Expression::Binary {
                op: BinaryOperator::Greater,
                ..
            }
        ));
    } else {
        panic!("Expected Having algebra");
    }
}

#[test]
fn test_algebra_values_clause() {
    let var_s = Variable::new("s").expect("valid variable");
    let var_label = Variable::new("label").expect("valid variable");

    let mut binding1 = HashMap::new();
    binding1.insert(
        var_s.clone(),
        Term::Iri(NamedNode::new("http://example.org/a").expect("valid IRI")),
    );
    binding1.insert(var_label.clone(), Term::Literal(Literal::string("Alice")));

    let mut binding2 = HashMap::new();
    binding2.insert(
        var_s.clone(),
        Term::Iri(NamedNode::new("http://example.org/b").expect("valid IRI")),
    );
    binding2.insert(var_label.clone(), Term::Literal(Literal::string("Bob")));

    let values = Algebra::Values {
        variables: vec![var_s.clone(), var_label.clone()],
        bindings: vec![binding1, binding2],
    };

    if let Algebra::Values {
        variables,
        bindings,
    } = values
    {
        assert_eq!(variables.len(), 2);
        assert_eq!(bindings.len(), 2);
        assert!(variables.contains(&var_s));
        assert!(variables.contains(&var_label));
    } else {
        panic!("Expected Values algebra");
    }
}

// ---- SPARQL OPTIONAL / LEFT JOIN with filter ----

#[test]
fn test_algebra_left_join_with_filter() {
    let var_s = Variable::new("s").expect("valid variable");
    let var_email = Variable::new("email").expect("valid variable");

    let left = Algebra::Bgp(vec![TriplePattern {
        subject: Term::Variable(var_s.clone()),
        predicate: Term::Iri(
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("valid IRI"),
        ),
        object: Term::Iri(NamedNode::new("http://xmlns.com/foaf/0.1/Person").expect("valid IRI")),
    }]);

    let right = Algebra::Bgp(vec![TriplePattern {
        subject: Term::Variable(var_s),
        predicate: Term::Iri(NamedNode::new("http://xmlns.com/foaf/0.1/mbox").expect("valid IRI")),
        object: Term::Variable(var_email.clone()),
    }]);

    let filter = Expression::Binary {
        op: BinaryOperator::NotEqual,
        left: Box::new(Expression::Variable(var_email)),
        right: Box::new(Expression::Literal(Literal::string(""))),
    };

    let optional = Algebra::LeftJoin {
        left: Box::new(left),
        right: Box::new(right),
        filter: Some(filter),
    };

    if let Algebra::LeftJoin {
        filter: Some(f), ..
    } = optional
    {
        assert!(matches!(
            f,
            Expression::Binary {
                op: BinaryOperator::NotEqual,
                ..
            }
        ));
    } else {
        panic!("Expected LeftJoin with filter");
    }
}

// ---- SPARQL expression / FILTER evaluation ----

#[test]
fn test_expression_bound_check() {
    let var = Variable::new("x").expect("valid variable");
    let bound_expr = Expression::Bound(var.clone());

    if let Expression::Bound(v) = bound_expr {
        assert_eq!(v.name(), "x");
    } else {
        panic!("Expected Bound expression");
    }
}

#[test]
fn test_expression_exists() {
    let bgp = Algebra::Bgp(vec![]);
    let exists = Expression::Exists(Box::new(bgp));

    assert!(matches!(exists, Expression::Exists(_)));
}

#[test]
fn test_expression_not_exists() {
    let bgp = Algebra::Bgp(vec![]);
    let not_exists = Expression::NotExists(Box::new(bgp));

    assert!(matches!(not_exists, Expression::NotExists(_)));
}

#[test]
fn test_expression_conditional_if_then_else() {
    let var_x = Variable::new("x").expect("valid variable");
    let cond = Expression::Bound(var_x.clone());
    let then_expr = Expression::Variable(var_x);
    let else_expr = Expression::Literal(Literal::string("default"));

    let conditional = Expression::Conditional {
        condition: Box::new(cond),
        then_expr: Box::new(then_expr),
        else_expr: Box::new(else_expr),
    };

    if let Expression::Conditional {
        condition,
        then_expr,
        else_expr,
    } = conditional
    {
        assert!(matches!(*condition, Expression::Bound(_)));
        assert!(matches!(*then_expr, Expression::Variable(_)));
        assert!(matches!(*else_expr, Expression::Literal(_)));
    } else {
        panic!("Expected Conditional expression");
    }
}

#[test]
fn test_expression_function_call() {
    let var = Variable::new("s").expect("valid variable");
    let func = Expression::Function {
        name: "str".to_string(),
        args: vec![Expression::Variable(var)],
    };

    if let Expression::Function { name, args } = func {
        assert_eq!(name, "str");
        assert_eq!(args.len(), 1);
    } else {
        panic!("Expected Function expression");
    }
}

#[test]
fn test_expression_unary_not() {
    let inner = Expression::Literal(Literal::boolean(true));
    let negated = Expression::Unary {
        op: UnaryOperator::Not,
        operand: Box::new(inner),
    };

    if let Expression::Unary { op, .. } = negated {
        assert!(matches!(op, UnaryOperator::Not));
    } else {
        panic!("Expected Unary Not expression");
    }
}

#[test]
fn test_expression_unary_is_iri() {
    let var = Variable::new("x").expect("valid variable");
    let is_iri = Expression::Unary {
        op: UnaryOperator::IsIri,
        operand: Box::new(Expression::Variable(var)),
    };

    if let Expression::Unary { op, .. } = is_iri {
        assert!(matches!(op, UnaryOperator::IsIri));
    } else {
        panic!("Expected Unary IsIri expression");
    }
}

#[test]
fn test_expression_unary_is_blank() {
    let var = Variable::new("x").expect("valid variable");
    let is_blank = Expression::Unary {
        op: UnaryOperator::IsBlank,
        operand: Box::new(Expression::Variable(var)),
    };

    if let Expression::Unary { op, .. } = is_blank {
        assert!(matches!(op, UnaryOperator::IsBlank));
    } else {
        panic!("Expected Unary IsBlank expression");
    }
}

#[test]
fn test_expression_binary_and() {
    let var_a = Variable::new("a").expect("valid variable");
    let var_b = Variable::new("b").expect("valid variable");

    let and_expr = Expression::Binary {
        op: BinaryOperator::And,
        left: Box::new(Expression::Variable(var_a)),
        right: Box::new(Expression::Variable(var_b)),
    };

    if let Expression::Binary { op, .. } = and_expr {
        assert!(matches!(op, BinaryOperator::And));
    } else {
        panic!("Expected Binary And expression");
    }
}

#[test]
fn test_expression_binary_or() {
    let var_a = Variable::new("a").expect("valid variable");
    let var_b = Variable::new("b").expect("valid variable");

    let or_expr = Expression::Binary {
        op: BinaryOperator::Or,
        left: Box::new(Expression::Variable(var_a)),
        right: Box::new(Expression::Variable(var_b)),
    };

    if let Expression::Binary { op, .. } = or_expr {
        assert!(matches!(op, BinaryOperator::Or));
    } else {
        panic!("Expected Binary Or expression");
    }
}

#[test]
fn test_expression_binary_comparison_operators() {
    let var = Variable::new("x").expect("valid variable");
    let lit = Expression::Literal(Literal::integer(10));

    let operators = [
        BinaryOperator::Less,
        BinaryOperator::LessEqual,
        BinaryOperator::Greater,
        BinaryOperator::GreaterEqual,
        BinaryOperator::Equal,
        BinaryOperator::NotEqual,
    ];

    for op in operators {
        let expr = Expression::Binary {
            op: op.clone(),
            left: Box::new(Expression::Variable(var.clone())),
            right: Box::new(lit.clone()),
        };
        assert!(matches!(expr, Expression::Binary { .. }));
    }
}

// ---- Property path patterns ----

#[test]
fn test_property_path_inverse() {
    let path = PropertyPath::Iri(NamedNode::new("http://example.org/parent").expect("valid IRI"));
    let inverse = PropertyPath::inverse(path);

    assert!(matches!(inverse, PropertyPath::Inverse(_)));
    let display = format!("{}", inverse);
    assert!(display.contains("^"));
}

#[test]
fn test_property_path_one_or_more() {
    let path = PropertyPath::Iri(NamedNode::new("http://example.org/ancestor").expect("valid IRI"));
    let one_or_more = PropertyPath::one_or_more(path);

    assert!(matches!(one_or_more, PropertyPath::OneOrMore(_)));
}

#[test]
fn test_property_path_zero_or_one() {
    let path = PropertyPath::Iri(NamedNode::new("http://example.org/optProp").expect("valid IRI"));
    let zero_or_one = PropertyPath::zero_or_one(path);

    assert!(matches!(zero_or_one, PropertyPath::ZeroOrOne(_)));
}

#[test]
fn test_property_path_negated_set() {
    let path1 = PropertyPath::Iri(NamedNode::new("http://example.org/p1").expect("valid IRI"));
    let path2 = PropertyPath::Iri(NamedNode::new("http://example.org/p2").expect("valid IRI"));
    let negated = PropertyPath::NegatedPropertySet(vec![path1, path2]);

    if let PropertyPath::NegatedPropertySet(paths) = negated {
        assert_eq!(paths.len(), 2);
    } else {
        panic!("Expected NegatedPropertySet");
    }
}

#[test]
fn test_property_path_complexity_ordering() {
    let direct = PropertyPath::Iri(NamedNode::new("http://example.org/p").expect("valid IRI"));
    let inverse = PropertyPath::inverse(PropertyPath::Iri(
        NamedNode::new("http://example.org/p").expect("valid IRI"),
    ));
    let star = PropertyPath::zero_or_more(PropertyPath::Iri(
        NamedNode::new("http://example.org/p").expect("valid IRI"),
    ));

    assert!(direct.complexity() < inverse.complexity());
    assert!(inverse.complexity() < star.complexity());
}

#[test]
fn test_property_path_display_sequence() {
    let path1 = PropertyPath::Iri(NamedNode::new("http://example.org/p1").expect("valid IRI"));
    let path2 = PropertyPath::Iri(NamedNode::new("http://example.org/p2").expect("valid IRI"));
    let seq = PropertyPath::sequence(path1, path2);
    let display = format!("{}", seq);
    assert!(display.contains('/'));
}

#[test]
fn test_property_path_display_alternative() {
    let path1 = PropertyPath::Iri(NamedNode::new("http://example.org/p1").expect("valid IRI"));
    let path2 = PropertyPath::Iri(NamedNode::new("http://example.org/p2").expect("valid IRI"));
    let alt = PropertyPath::alternative(path1, path2);
    let display = format!("{}", alt);
    assert!(display.contains('|'));
}

#[test]
fn test_property_path_is_simple() {
    let direct = PropertyPath::Iri(NamedNode::new("http://example.org/p").expect("valid IRI"));
    let star = PropertyPath::zero_or_more(PropertyPath::Iri(
        NamedNode::new("http://example.org/p").expect("valid IRI"),
    ));

    assert!(direct.is_simple());
    assert!(!star.is_simple());
}

// ---- Algebra helper constructors ----

#[test]
fn test_algebra_bgp_constructor() {
    let triple = TriplePattern {
        subject: Term::Variable(Variable::new("s").expect("valid variable")),
        predicate: Term::Iri(NamedNode::new("http://example.org/p").expect("valid IRI")),
        object: Term::Variable(Variable::new("o").expect("valid variable")),
    };
    let bgp = Algebra::bgp(vec![triple]);

    assert!(matches!(bgp, Algebra::Bgp(_)));
    if let Algebra::Bgp(patterns) = bgp {
        assert_eq!(patterns.len(), 1);
    }
}

#[test]
fn test_algebra_join_constructor() {
    let left = Algebra::bgp(vec![]);
    let right = Algebra::bgp(vec![]);
    let join = Algebra::join(left, right);

    assert!(matches!(join, Algebra::Join { .. }));
}

#[test]
fn test_algebra_union_constructor() {
    let left = Algebra::bgp(vec![]);
    let right = Algebra::bgp(vec![]);
    let union = Algebra::union(left, right);

    assert!(matches!(union, Algebra::Union { .. }));
}

#[test]
fn test_algebra_filter_constructor() {
    let bgp = Algebra::bgp(vec![]);
    let var = Variable::new("x").expect("valid variable");
    let condition = Expression::Bound(var);
    let filtered = Algebra::filter(bgp, condition);

    assert!(matches!(filtered, Algebra::Filter { .. }));
}

#[test]
fn test_algebra_left_join_constructor_no_filter() {
    let left = Algebra::bgp(vec![]);
    let right = Algebra::bgp(vec![]);
    let optional = Algebra::left_join(left, right, None);

    if let Algebra::LeftJoin { filter, .. } = optional {
        assert!(filter.is_none());
    } else {
        panic!("Expected LeftJoin");
    }
}

#[test]
fn test_algebra_extend_bind() {
    let bgp = Algebra::bgp(vec![]);
    let var = Variable::new("upper").expect("valid variable");
    let expr = Expression::Function {
        name: "ucase".to_string(),
        args: vec![Expression::Literal(Literal::string("hello"))],
    };
    let extended = Algebra::extend(bgp, var.clone(), expr);

    if let Algebra::Extend { variable, .. } = extended {
        assert_eq!(variable.name(), "upper");
    } else {
        panic!("Expected Extend algebra");
    }
}

// ---- Statistics and OptimizationHints ----

#[test]
fn test_statistics_with_cardinality() {
    let stats = Statistics::with_cardinality(1000);

    assert_eq!(stats.cardinality, 1000);
    assert_eq!(stats.selectivity, 1.0);
    assert!(stats.cost > 0.0);
}

#[test]
fn test_statistics_with_selectivity() {
    let stats = Statistics::with_cardinality(10000).with_selectivity(0.1);

    assert_eq!(stats.selectivity, 0.1);
    assert!(stats.cardinality <= 10000);
}

#[test]
fn test_statistics_selectivity_clamp() {
    let stats_over = Statistics::with_cardinality(100).with_selectivity(1.5);
    assert!(stats_over.selectivity <= 1.0);

    let stats_under = Statistics::with_cardinality(100).with_selectivity(-0.5);
    assert!(stats_under.selectivity >= 0.0);
}

#[test]
fn test_statistics_combine() {
    let stats_a = Statistics::with_cardinality(100);
    let stats_b = Statistics::with_cardinality(50);
    let combined = stats_a.combine(&stats_b);

    assert_eq!(combined.cardinality, 5000); // 100 * 50
}

#[test]
fn test_optimization_hints_for_bgp() {
    let pattern = TriplePattern {
        subject: Term::Variable(Variable::new("s").expect("valid variable")),
        predicate: Term::Iri(NamedNode::new("http://example.org/p").expect("valid IRI")),
        object: Term::Variable(Variable::new("o").expect("valid variable")),
    };

    let hints = OptimizationHints::for_bgp(&[pattern]);
    assert!(hints.statistics.is_some());
}

#[test]
fn test_optimization_hints_for_join() {
    let left_hints = OptimizationHints {
        statistics: Some(Statistics::with_cardinality(100)),
        ..Default::default()
    };
    let right_hints = OptimizationHints {
        statistics: Some(Statistics::with_cardinality(200)),
        ..Default::default()
    };

    let join_hints = OptimizationHints::for_join(&left_hints, &right_hints);
    assert!(join_hints.statistics.is_some());
    assert!(join_hints.join_algorithm.is_some());
}

// ---- Aggregate distinct flag ----

#[test]
fn test_aggregate_count_distinct() {
    let var_x = Variable::new("x").expect("valid variable");
    let agg = Aggregate::Count {
        distinct: true,
        expr: Some(Expression::Variable(var_x)),
    };

    if let Aggregate::Count { distinct, expr } = agg {
        assert!(distinct);
        assert!(expr.is_some());
    } else {
        panic!("Expected Count aggregate");
    }
}

#[test]
fn test_aggregate_count_star() {
    let agg = Aggregate::Count {
        distinct: false,
        expr: None,
    };

    if let Aggregate::Count { distinct, expr } = agg {
        assert!(!distinct);
        assert!(expr.is_none());
    } else {
        panic!("Expected Count aggregate");
    }
}

#[test]
fn test_aggregate_sample() {
    let var_x = Variable::new("x").expect("valid variable");
    let agg = Aggregate::Sample {
        distinct: false,
        expr: Expression::Variable(var_x),
    };

    assert!(matches!(agg, Aggregate::Sample { .. }));
}

// ---- Algebra variable collection ----

#[test]
fn test_algebra_variable_collection_bgp() {
    let var_s = Variable::new("s").expect("valid variable");
    let var_o = Variable::new("o").expect("valid variable");

    let bgp = Algebra::Bgp(vec![TriplePattern {
        subject: Term::Variable(var_s.clone()),
        predicate: Term::Iri(NamedNode::new("http://example.org/p").expect("valid IRI")),
        object: Term::Variable(var_o.clone()),
    }]);

    let vars = bgp.variables();
    assert!(vars.contains(&var_s));
    assert!(vars.contains(&var_o));
}

#[test]
fn test_algebra_variable_collection_project() {
    let var_s = Variable::new("s").expect("valid variable");
    let projection = Algebra::Project {
        pattern: Box::new(Algebra::Bgp(vec![])),
        variables: vec![var_s.clone()],
    };

    let vars = projection.variables();
    assert!(vars.contains(&var_s));
}

#[test]
fn test_algebra_variable_collection_values() {
    let var_a = Variable::new("a").expect("valid variable");
    let var_b = Variable::new("b").expect("valid variable");

    let values = Algebra::Values {
        variables: vec![var_a.clone(), var_b.clone()],
        bindings: vec![],
    };

    let vars = values.variables();
    assert!(vars.contains(&var_a));
    assert!(vars.contains(&var_b));
}

// ---- Annotated algebra ----

#[test]
fn test_annotated_algebra_new() {
    let bgp = Algebra::bgp(vec![]);
    let annotated = AnnotatedAlgebra::new(bgp);

    assert!(annotated.context.is_none());
}

#[test]
fn test_annotated_algebra_with_context() {
    let bgp = Algebra::bgp(vec![]);
    let annotated = AnnotatedAlgebra::new(bgp).with_context("test_query".to_string());

    assert_eq!(annotated.context.as_deref(), Some("test_query"));
}

#[test]
fn test_annotated_algebra_with_hints() {
    let bgp = Algebra::bgp(vec![]);
    let hints = OptimizationHints {
        statistics: Some(Statistics::with_cardinality(500)),
        ..Default::default()
    };

    let annotated = AnnotatedAlgebra::with_hints(bgp, hints);
    assert_eq!(annotated.estimated_cardinality(), 500);
}

// ---- Literal type checks ----

#[test]
fn test_literal_integer_type_check() {
    let lit = Literal::integer(42);
    assert!(lit.is_numeric());
    assert!(!lit.is_string());
    assert!(!lit.is_lang_string());
    assert!(!lit.is_boolean());
}

#[test]
fn test_literal_decimal_type_check() {
    let lit = Literal::decimal(std::f64::consts::PI);
    assert!(lit.is_numeric());
    assert!(!lit.is_string());
}

#[test]
fn test_literal_boolean_type_check() {
    let lit = Literal::boolean(true);
    assert!(lit.is_boolean());
    assert!(!lit.is_numeric());
    assert!(!lit.is_string());
}

#[test]
fn test_literal_string_type_check() {
    let lit = Literal::string("hello");
    assert!(lit.is_string());
    assert!(!lit.is_numeric());
    assert!(!lit.is_boolean());
}

#[test]
fn test_literal_lang_string_type_check() {
    let lit = Literal::lang_string("hola", "es");
    assert!(lit.is_lang_string());
    assert!(!lit.is_string());
    assert!(!lit.is_numeric());
}

#[test]
fn test_literal_date_type_check() {
    let lit = Literal::date("2024-01-15");
    assert!(lit.is_datetime());
    assert!(!lit.is_numeric());
}

#[test]
fn test_literal_datetime_type_check() {
    let lit = Literal::datetime("2024-01-15T10:30:00Z");
    assert!(lit.is_datetime());
}

#[test]
fn test_literal_effective_datatype_plain_string() {
    let lit = Literal::string("test");
    let dt = lit.effective_datatype();
    assert_eq!(dt.as_str(), "http://www.w3.org/2001/XMLSchema#string");
}

#[test]
fn test_literal_effective_datatype_lang_string() {
    let lit = Literal::lang_string("hello", "en");
    let dt = lit.effective_datatype();
    assert_eq!(
        dt.as_str(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"
    );
}

// ---- Filter selectivity for additional operators ----

#[test]
fn test_filter_selectivity_and_operator() {
    let var = Variable::new("x").expect("valid variable");
    let and_expr = Expression::Binary {
        op: BinaryOperator::And,
        left: Box::new(Expression::Variable(var.clone())),
        right: Box::new(Expression::Variable(var)),
    };

    let selectivity = estimate_filter_selectivity(&and_expr);
    assert_eq!(selectivity, 0.25);
}

#[test]
fn test_filter_selectivity_or_operator() {
    let var = Variable::new("x").expect("valid variable");
    let or_expr = Expression::Binary {
        op: BinaryOperator::Or,
        left: Box::new(Expression::Variable(var.clone())),
        right: Box::new(Expression::Variable(var)),
    };

    let selectivity = estimate_filter_selectivity(&or_expr);
    assert_eq!(selectivity, 0.75);
}

#[test]
fn test_filter_selectivity_function_regex() {
    let var = Variable::new("name").expect("valid variable");
    let regex_expr = Expression::Function {
        name: "regex".to_string(),
        args: vec![
            Expression::Variable(var),
            Expression::Literal(Literal::string("Alice")),
        ],
    };

    let selectivity = estimate_filter_selectivity(&regex_expr);
    assert_eq!(selectivity, 0.2);
}

#[test]
fn test_filter_selectivity_unary_not() {
    let inner = Expression::Literal(Literal::boolean(true));
    let not_expr = Expression::Unary {
        op: UnaryOperator::Not,
        operand: Box::new(inner),
    };

    let selectivity = estimate_filter_selectivity(&not_expr);
    // NOT inverts selectivity
    assert!(selectivity > 0.0 && selectivity <= 1.0);
}
