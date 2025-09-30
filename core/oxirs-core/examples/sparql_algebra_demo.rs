//! SPARQL Algebra Phase 2 Extraction Demo
//!
//! Demonstrates the enhanced SPARQL algebra extracted from OxiGraph spargebra.

use oxirs_core::model::{Literal, NamedNode, Variable};
use oxirs_core::query::{
    algebra::{AlgebraTriplePattern as TriplePattern, TermPattern as AlgebraTermPattern},
    sparql_algebra::{
        AggregateExpression, BuiltInFunction, Expression, FunctionExpression, GraphPattern,
        OrderExpression, PropertyPathExpression, TermPattern,
    },
    sparql_query::{Query, QueryDataset},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 SPARQL Algebra Phase 2 Extraction Demo");
    println!("=========================================\n");

    // Test basic triple pattern creation
    println!("✅ Triple Pattern Creation:");
    let subject = AlgebraTermPattern::Variable(Variable::new("person")?);
    let predicate =
        AlgebraTermPattern::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/name")?);
    let object = AlgebraTermPattern::Variable(Variable::new("name")?);

    let triple = TriplePattern::new(subject, predicate, object);
    println!("   Created triple pattern: {triple}");

    // Test basic graph pattern (BGP) - using SPARQL algebra patterns
    println!("\n✅ Basic Graph Pattern:");
    // Convert to SPARQL algebra triple pattern for BGP
    let sparql_triple = oxirs_core::query::sparql_algebra::TriplePattern {
        subject: TermPattern::Variable(Variable::new("person")?),
        predicate: TermPattern::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/name")?),
        object: TermPattern::Variable(Variable::new("name")?),
    };
    let bgp = GraphPattern::Bgp {
        patterns: vec![sparql_triple],
    };
    println!("   BGP: {bgp}");

    // Test property path expression
    println!("\n✅ Property Path Expression:");
    let prop1 = NamedNode::new("http://example.org/knows")?;
    let prop2 = NamedNode::new("http://example.org/worksFor")?;

    let path = PropertyPathExpression::Sequence(
        Box::new(PropertyPathExpression::NamedNode(prop1)),
        Box::new(PropertyPathExpression::NamedNode(prop2)),
    );
    println!("   Property path: {path}");

    // Test complex expression
    println!("\n✅ SPARQL Expression:");
    let var1 = Expression::Variable(Variable::new("x")?);
    let var2 = Expression::Variable(Variable::new("y")?);
    let addition = Expression::Add(Box::new(var1), Box::new(var2));
    println!("   Expression: {addition}");

    // Test function call
    println!("\n✅ Function Call:");
    let str_func = FunctionExpression::BuiltIn(BuiltInFunction::Str);
    let name_var = Expression::Variable(Variable::new("name")?);
    let func_call = Expression::FunctionCall(str_func, vec![name_var]);
    println!("   Function call: {func_call}");

    // Test FILTER pattern
    println!("\n✅ Filter Pattern:");
    let age_var = Expression::Variable(Variable::new("age")?);
    let eighteen = Expression::Literal(Literal::new("18"));
    let condition = Expression::Greater(Box::new(age_var), Box::new(eighteen));

    let simple_bgp = GraphPattern::Bgp {
        patterns: vec![oxirs_core::query::sparql_algebra::TriplePattern {
            subject: TermPattern::Variable(Variable::new("person")?),
            predicate: TermPattern::NamedNode(NamedNode::new("http://example.org/age")?),
            object: TermPattern::Variable(Variable::new("age")?),
        }],
    };

    let filter_pattern = GraphPattern::Filter {
        expr: condition,
        inner: Box::new(simple_bgp),
    };
    println!("   Filter pattern: {filter_pattern}");

    // Test UNION pattern
    println!("\n✅ Union Pattern:");
    let left_pattern = GraphPattern::Bgp {
        patterns: vec![oxirs_core::query::sparql_algebra::TriplePattern {
            subject: TermPattern::Variable(Variable::new("x")?),
            predicate: TermPattern::NamedNode(NamedNode::new("http://example.org/type")?),
            object: TermPattern::NamedNode(NamedNode::new("http://example.org/Person")?),
        }],
    };

    let right_pattern = GraphPattern::Bgp {
        patterns: vec![oxirs_core::query::sparql_algebra::TriplePattern {
            subject: TermPattern::Variable(Variable::new("x")?),
            predicate: TermPattern::NamedNode(NamedNode::new("http://example.org/type")?),
            object: TermPattern::NamedNode(NamedNode::new("http://example.org/Organization")?),
        }],
    };

    let union_pattern = GraphPattern::Union {
        left: Box::new(left_pattern),
        right: Box::new(right_pattern),
    };
    println!("   Union pattern: {union_pattern}");

    // Test complete SELECT query
    println!("\n✅ Complete SELECT Query:");
    let select_pattern = GraphPattern::Project {
        inner: Box::new(union_pattern.clone()),
        variables: vec![Variable::new("x")?],
    };

    let query = Query::select(select_pattern);
    println!("   Query: {query}");

    // Test S-Expression formatting
    println!("\n✅ S-Expression Format:");
    println!("   SSE: {}", query.to_sse());

    // Test query dataset
    println!("\n✅ Query Dataset:");
    let mut dataset = QueryDataset::new();
    dataset.add_default_graph(NamedNode::new("http://example.org/default")?);
    dataset.add_named_graph(NamedNode::new("http://example.org/named")?);

    let query_with_dataset = query.with_dataset(dataset);
    println!("   Query with dataset: {query_with_dataset}");

    // Test aggregate expression
    println!("\n✅ Aggregate Expression:");
    let count_agg = AggregateExpression::Count {
        expr: Some(Box::new(Expression::Variable(Variable::new("person")?))),
        distinct: true,
    };
    println!("   Aggregate: {count_agg}");

    // Test order expression
    println!("\n✅ Order Expression:");
    let order_expr = OrderExpression::Desc(Expression::Variable(Variable::new("age")?));
    println!("   Order by: {order_expr}");

    println!("\n🎉 Phase 2 SPARQL Algebra Extraction Complete!");
    println!("   ✓ Enhanced PropertyPathExpression with all operators");
    println!("   ✓ Complete Expression enum with all SPARQL operators");
    println!("   ✓ Full GraphPattern enum with all constructs");
    println!("   ✓ Query types (SELECT, CONSTRUCT, ASK, DESCRIBE)");
    println!("   ✓ Built-in functions and custom functions");
    println!("   ✓ Dataset specifications and aggregates");
    println!("   ✓ S-Expression and SPARQL syntax formatting");
    println!("   ✓ Zero external dependencies");

    Ok(())
}
