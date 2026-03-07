//! Date/Time Function Conformance Tests
//!
//! Tests SPARQL 1.1 built-in date/time function expressions:
//! NOW(), YEAR(), MONTH(), DAY(), HOURS(), MINUTES(), SECONDS(),
//! TIMEZONE(), TZ(), xsd:dateTime arithmetic.

use super::framework::*;
use super::helpers::*;
use crate::algebra::Literal;
use crate::executor::InMemoryDataset;

fn runner() -> ConformanceTestRunner {
    ConformanceTestRunner::new()
}

// ===== HELPER: Build a datetime-typed literal term =====

fn datetime_lit(value: &str) -> crate::algebra::Term {
    crate::algebra::Term::Literal(Literal {
        value: value.to_string(),
        language: None,
        datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
            "http://www.w3.org/2001/XMLSchema#dateTime",
        )),
    })
}

fn datetime_dataset() -> InMemoryDataset {
    let mut ds = InMemoryDataset::new();
    // Events with dateTime values
    ds.add_triple(
        ex("event1"),
        ex("timestamp"),
        datetime_lit("2023-01-15T10:30:00Z"),
    );
    ds.add_triple(
        ex("event2"),
        ex("timestamp"),
        datetime_lit("2023-06-21T14:00:00Z"),
    );
    ds.add_triple(
        ex("event3"),
        ex("timestamp"),
        datetime_lit("2024-03-10T08:45:30Z"),
    );
    ds.add_triple(
        ex("event4"),
        ex("timestamp"),
        datetime_lit("2024-12-25T00:00:00Z"),
    );
    ds.add_triple(
        ex("event5"),
        ex("timestamp"),
        datetime_lit("2022-11-01T23:59:59Z"),
    );
    ds.add_triple(ex("event1"), ex("label"), str_lit("Event One"));
    ds.add_triple(ex("event2"), ex("label"), str_lit("Event Two"));
    ds.add_triple(ex("event3"), ex("label"), str_lit("Event Three"));
    ds.add_triple(ex("event4"), ex("label"), str_lit("Event Four"));
    ds.add_triple(ex("event5"), ex("label"), str_lit("Event Five"));
    ds
}

fn duration_dataset() -> InMemoryDataset {
    let mut ds = InMemoryDataset::new();
    ds.add_triple(ex("s1"), ex("val"), int_lit(10));
    ds.add_triple(ex("s2"), ex("val"), int_lit(20));
    ds.add_triple(ex("s3"), ex("val"), int_lit(30));
    ds
}

// ===== NOW() FUNCTION TESTS =====

#[test]
fn test_dt_now_01_returns_result() {
    // NOW() should return a result (non-empty binding) in a query
    let ds = duration_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("val"), var("v"))]),
            variable("ts"),
            expr_fn("now", vec![]),
        ),
        vec![variable("s"), variable("ts")],
    );
    let test = ConformanceTest::new(
        "dt-now-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("dt-now-01 failed");
}

#[test]
fn test_dt_now_02_now_no_args() {
    // NOW() with empty BGP still executes
    let ds = InMemoryDataset::new();
    let algebra = project(
        extend(bgp(vec![]), variable("current"), expr_fn("now", vec![])),
        vec![variable("current")],
    );
    let test = ConformanceTest::new(
        "dt-now-02",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        // Empty BGP = no results
        ConformanceResult::ResultCount(0),
    );
    runner().run_test(&test).expect("dt-now-02 failed");
}

// ===== YEAR() FUNCTION TESTS =====

#[test]
fn test_dt_year_01_extract_from_literal() {
    // YEAR on a dateTime literal via BIND
    let ds = duration_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("val"), var("v"))]),
            variable("yr"),
            expr_fn(
                "year",
                vec![Expression::Literal(Literal {
                    value: "2023-06-15T12:00:00Z".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#dateTime",
                    )),
                })],
            ),
        ),
        vec![variable("s"), variable("yr")],
    );
    let test = ConformanceTest::new(
        "dt-year-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("dt-year-01 failed");
}

#[test]
fn test_dt_year_02_filter_by_year() {
    // Filter events from year 2023
    let ds = datetime_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            expr_eq(
                expr_fn("year", vec![expr_var("ts")]),
                Expression::Literal(Literal {
                    value: "2023".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }),
            ),
        ),
        vec![variable("e")],
    );
    let test = ConformanceTest::new(
        "dt-year-02",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        // event1 (2023-01) and event2 (2023-06)
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("dt-year-02 failed");
}

#[test]
fn test_dt_year_03_multiple_years() {
    // All events from year 2024
    let ds = datetime_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            expr_eq(
                expr_fn("year", vec![expr_var("ts")]),
                Expression::Literal(Literal {
                    value: "2024".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }),
            ),
        ),
        vec![variable("e")],
    );
    let test = ConformanceTest::new(
        "dt-year-03",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        // event3 (2024-03) and event4 (2024-12)
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("dt-year-03 failed");
}

// ===== MONTH() FUNCTION TESTS =====

#[test]
fn test_dt_month_01_extract_month() {
    // MONTH returns result from dateTime
    let ds = duration_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("val"), var("v"))]),
            variable("mo"),
            expr_fn(
                "month",
                vec![Expression::Literal(Literal {
                    value: "2023-06-15T12:00:00Z".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#dateTime",
                    )),
                })],
            ),
        ),
        vec![variable("s"), variable("mo")],
    );
    let test = ConformanceTest::new(
        "dt-month-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("dt-month-01 failed");
}

#[test]
fn test_dt_month_02_filter_december() {
    // Filter events from December
    let ds = datetime_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            expr_eq(
                expr_fn("month", vec![expr_var("ts")]),
                Expression::Literal(Literal {
                    value: "12".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }),
            ),
        ),
        vec![variable("e")],
    );
    let test = ConformanceTest::new(
        "dt-month-02",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        // event4 (2024-12-25)
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("dt-month-02 failed");
}

// ===== DAY() FUNCTION TESTS =====

#[test]
fn test_dt_day_01_extract_day() {
    // DAY function returns result
    let ds = duration_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("val"), var("v"))]),
            variable("d"),
            expr_fn(
                "day",
                vec![Expression::Literal(Literal {
                    value: "2023-06-15T12:00:00Z".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#dateTime",
                    )),
                })],
            ),
        ),
        vec![variable("s"), variable("d")],
    );
    let test = ConformanceTest::new(
        "dt-day-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("dt-day-01 failed");
}

#[test]
fn test_dt_day_02_filter_day_25() {
    // Filter events on day 25
    let ds = datetime_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            expr_eq(
                expr_fn("day", vec![expr_var("ts")]),
                Expression::Literal(Literal {
                    value: "25".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }),
            ),
        ),
        vec![variable("e")],
    );
    let test = ConformanceTest::new(
        "dt-day-02",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        // event4 (2024-12-25)
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("dt-day-02 failed");
}

// ===== HOURS() FUNCTION TESTS =====

#[test]
fn test_dt_hours_01_extract_hours() {
    // HOURS extracts hours from dateTime
    let ds = duration_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("val"), var("v"))]),
            variable("h"),
            expr_fn(
                "hours",
                vec![Expression::Literal(Literal {
                    value: "2023-06-15T10:30:00Z".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#dateTime",
                    )),
                })],
            ),
        ),
        vec![variable("s"), variable("h")],
    );
    let test = ConformanceTest::new(
        "dt-hours-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("dt-hours-01 failed");
}

#[test]
fn test_dt_hours_02_filter_midnight() {
    // Filter events at midnight (hour 0)
    let ds = datetime_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            expr_eq(
                expr_fn("hours", vec![expr_var("ts")]),
                Expression::Literal(Literal {
                    value: "0".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }),
            ),
        ),
        vec![variable("e")],
    );
    let test = ConformanceTest::new(
        "dt-hours-02",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        // event4 (2024-12-25T00:00:00Z)
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("dt-hours-02 failed");
}

// ===== MINUTES() FUNCTION TESTS =====

#[test]
fn test_dt_minutes_01_extract_minutes() {
    // MINUTES function
    let ds = duration_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("val"), var("v"))]),
            variable("m"),
            expr_fn(
                "minutes",
                vec![Expression::Literal(Literal {
                    value: "2023-06-15T10:30:00Z".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#dateTime",
                    )),
                })],
            ),
        ),
        vec![variable("s"), variable("m")],
    );
    let test = ConformanceTest::new(
        "dt-minutes-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("dt-minutes-01 failed");
}

#[test]
fn test_dt_minutes_02_filter_zero_minutes() {
    // Filter events at :00 minutes
    let ds = datetime_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            expr_eq(
                expr_fn("minutes", vec![expr_var("ts")]),
                Expression::Literal(Literal {
                    value: "0".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }),
            ),
        ),
        vec![variable("e")],
    );
    let test = ConformanceTest::new(
        "dt-minutes-02",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        // event2 (14:00:00) and event4 (00:00:00)
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("dt-minutes-02 failed");
}

// ===== SECONDS() FUNCTION TESTS =====

#[test]
fn test_dt_seconds_01_extract_seconds() {
    // SECONDS function
    let ds = duration_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("val"), var("v"))]),
            variable("sec"),
            expr_fn(
                "seconds",
                vec![Expression::Literal(Literal {
                    value: "2023-06-15T10:30:45Z".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#dateTime",
                    )),
                })],
            ),
        ),
        vec![variable("s"), variable("sec")],
    );
    let test = ConformanceTest::new(
        "dt-seconds-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("dt-seconds-01 failed");
}

#[test]
fn test_dt_seconds_02_non_zero_seconds() {
    // Filter events with non-zero seconds
    let ds = datetime_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            expr_gt(
                expr_fn("seconds", vec![expr_var("ts")]),
                Expression::Literal(Literal {
                    value: "0".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#decimal",
                    )),
                }),
            ),
        ),
        vec![variable("e")],
    );
    let test = ConformanceTest::new(
        "dt-seconds-02",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        // event3 (08:45:30) and event5 (23:59:59)
        ConformanceResult::ResultCount(2),
    );
    runner().run_test(&test).expect("dt-seconds-02 failed");
}

// ===== TIMEZONE() FUNCTION TESTS =====

#[test]
fn test_dt_timezone_01_utc_timezone() {
    // TIMEZONE of UTC dateTime
    let ds = duration_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("val"), var("v"))]),
            variable("tz"),
            expr_fn(
                "timezone",
                vec![Expression::Literal(Literal {
                    value: "2023-06-15T10:30:00Z".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#dateTime",
                    )),
                })],
            ),
        ),
        vec![variable("s"), variable("tz")],
    );
    let test = ConformanceTest::new(
        "dt-timezone-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("dt-timezone-01 failed");
}

// ===== TZ() FUNCTION TESTS =====

#[test]
fn test_dt_tz_01_tz_string() {
    // TZ returns timezone as string
    let ds = duration_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("val"), var("v"))]),
            variable("tz"),
            expr_fn(
                "tz",
                vec![Expression::Literal(Literal {
                    value: "2023-06-15T10:30:00Z".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#dateTime",
                    )),
                })],
            ),
        ),
        vec![variable("s"), variable("tz")],
    );
    let test = ConformanceTest::new(
        "dt-tz-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("dt-tz-01 failed");
}

// ===== DATETIME BIND AND EXTEND =====

#[test]
fn test_dt_bind_01_bind_year() {
    // BIND(YEAR(?ts) AS ?yr)
    let ds = datetime_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            variable("yr"),
            expr_fn("year", vec![expr_var("ts")]),
        ),
        vec![variable("e"), variable("yr")],
    );
    let test = ConformanceTest::new(
        "dt-bind-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("dt-bind-01 failed");
}

#[test]
fn test_dt_bind_02_bind_month() {
    // BIND(MONTH(?ts) AS ?mo)
    let ds = datetime_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            variable("mo"),
            expr_fn("month", vec![expr_var("ts")]),
        ),
        vec![variable("e"), variable("mo")],
    );
    let test = ConformanceTest::new(
        "dt-bind-02",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("dt-bind-02 failed");
}

#[test]
fn test_dt_bind_03_bind_day() {
    // BIND(DAY(?ts) AS ?d)
    let ds = datetime_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            variable("d"),
            expr_fn("day", vec![expr_var("ts")]),
        ),
        vec![variable("e"), variable("d")],
    );
    let test = ConformanceTest::new(
        "dt-bind-03",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("dt-bind-03 failed");
}

#[test]
fn test_dt_bind_04_bind_hours() {
    // BIND(HOURS(?ts) AS ?h)
    let ds = datetime_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            variable("h"),
            expr_fn("hours", vec![expr_var("ts")]),
        ),
        vec![variable("e"), variable("h")],
    );
    let test = ConformanceTest::new(
        "dt-bind-04",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("dt-bind-04 failed");
}

#[test]
fn test_dt_bind_05_bind_minutes() {
    // BIND(MINUTES(?ts) AS ?m)
    let ds = datetime_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            variable("m"),
            expr_fn("minutes", vec![expr_var("ts")]),
        ),
        vec![variable("e"), variable("m")],
    );
    let test = ConformanceTest::new(
        "dt-bind-05",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("dt-bind-05 failed");
}

#[test]
fn test_dt_bind_06_bind_seconds() {
    // BIND(SECONDS(?ts) AS ?sec)
    let ds = datetime_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            variable("sec"),
            expr_fn("seconds", vec![expr_var("ts")]),
        ),
        vec![variable("e"), variable("sec")],
    );
    let test = ConformanceTest::new(
        "dt-bind-06",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("dt-bind-06 failed");
}

// ===== COMBINED DATETIME TESTS =====

#[test]
fn test_dt_combined_01_year_and_month() {
    // Filter by both year AND month
    let ds = datetime_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            expr_and(
                expr_eq(
                    expr_fn("year", vec![expr_var("ts")]),
                    Expression::Literal(Literal {
                        value: "2023".to_string(),
                        language: None,
                        datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                            "http://www.w3.org/2001/XMLSchema#integer",
                        )),
                    }),
                ),
                expr_eq(
                    expr_fn("month", vec![expr_var("ts")]),
                    Expression::Literal(Literal {
                        value: "1".to_string(),
                        language: None,
                        datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                            "http://www.w3.org/2001/XMLSchema#integer",
                        )),
                    }),
                ),
            ),
        ),
        vec![variable("e")],
    );
    let test = ConformanceTest::new(
        "dt-combined-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        // event1 (2023-01-15)
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("dt-combined-01 failed");
}

#[test]
fn test_dt_combined_02_year_range() {
    // Filter events from 2023 or later (year >= 2023)
    let ds = datetime_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: Box::new(expr_fn("year", vec![expr_var("ts")])),
                right: Box::new(Expression::Literal(Literal {
                    value: "2023".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                })),
            },
        ),
        vec![variable("e")],
    );
    let test = ConformanceTest::new(
        "dt-combined-02",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        // event1 (2023), event2 (2023), event3 (2024), event4 (2024) — not event5 (2022)
        ConformanceResult::ResultCount(4),
    );
    runner().run_test(&test).expect("dt-combined-02 failed");
}

#[test]
fn test_dt_combined_03_bind_and_filter() {
    // BIND year, then filter on the bound variable
    let ds = datetime_dataset();
    let algebra = project(
        filter(
            extend(
                bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
                variable("yr"),
                expr_fn("year", vec![expr_var("ts")]),
            ),
            expr_eq(
                expr_var("yr"),
                Expression::Literal(Literal {
                    value: "2022".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }),
            ),
        ),
        vec![variable("e"), variable("yr")],
    );
    let test = ConformanceTest::new(
        "dt-combined-03",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        // event5 (2022-11-01)
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("dt-combined-03 failed");
}

#[test]
fn test_dt_combined_04_multiple_functions() {
    // Bind both year and month
    let ds = datetime_dataset();
    let algebra = project(
        extend(
            extend(
                bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
                variable("yr"),
                expr_fn("year", vec![expr_var("ts")]),
            ),
            variable("mo"),
            expr_fn("month", vec![expr_var("ts")]),
        ),
        vec![variable("e"), variable("yr"), variable("mo")],
    );
    let test = ConformanceTest::new(
        "dt-combined-04",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("dt-combined-04 failed");
}

// ===== ORDER BY DATETIME =====

#[test]
fn test_dt_order_01_order_by_year() {
    // ORDER BY YEAR(?ts)
    let ds = datetime_dataset();
    let algebra = project(
        order_by(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            vec![asc_cond(expr_fn("year", vec![expr_var("ts")]))],
        ),
        vec![variable("e")],
    );
    let test = ConformanceTest::new(
        "dt-order-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("dt-order-01 failed");
}

#[test]
fn test_dt_order_02_desc_order_by_timestamp() {
    // ORDER BY DESC(?ts) — descending by timestamp
    let ds = datetime_dataset();
    let algebra = project(
        order_by(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            vec![desc_cond(expr_var("ts"))],
        ),
        vec![variable("e"), variable("ts")],
    );
    let test = ConformanceTest::new(
        "dt-order-02",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("dt-order-02 failed");
}

// ===== DATATYPE FUNCTIONS ON DATETIME =====

#[test]
fn test_dt_datatype_01_datetime_datatype() {
    // FILTER(datatype(?ts) = xsd:dateTime)
    let ds = datetime_dataset();
    let algebra = project(
        filter(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            expr_eq(
                expr_fn("datatype", vec![expr_var("ts")]),
                expr_iri("http://www.w3.org/2001/XMLSchema#dateTime"),
            ),
        ),
        vec![variable("e")],
    );
    let test = ConformanceTest::new(
        "dt-datatype-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("dt-datatype-01 failed");
}

#[test]
fn test_dt_datatype_02_string_not_datetime() {
    // Mixed dataset: string values should NOT match xsd:dateTime filter
    let mut ds = datetime_dataset();
    ds.add_triple(ex("event6"), ex("timestamp"), str_lit("not-a-date"));
    let algebra = project(
        filter(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            expr_eq(
                expr_fn("datatype", vec![expr_var("ts")]),
                expr_iri("http://www.w3.org/2001/XMLSchema#dateTime"),
            ),
        ),
        vec![variable("e")],
    );
    let test = ConformanceTest::new(
        "dt-datatype-02",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        // Only the 5 proper dateTime values
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("dt-datatype-02 failed");
}

// ===== SPARQL-STYLE IRI DATETIME FUNCTIONS =====

#[test]
fn test_dt_sparql_iri_01_now_via_iri() {
    // NOW() via its XPath IRI name
    let ds = duration_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("val"), var("v"))]),
            variable("ts"),
            expr_fn(
                "http://www.w3.org/2005/xpath-functions#current-dateTime",
                vec![],
            ),
        ),
        vec![variable("s"), variable("ts")],
    );
    let test = ConformanceTest::new(
        "dt-sparql-iri-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("dt-sparql-iri-01 failed");
}

#[test]
fn test_dt_sparql_iri_02_year_via_iri() {
    // YEAR() via its XPath IRI name
    let ds = duration_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("s"), ex("val"), var("v"))]),
            variable("yr"),
            expr_fn(
                "http://www.w3.org/2005/xpath-functions#year-from-dateTime",
                vec![Expression::Literal(Literal {
                    value: "2024-03-15T10:00:00Z".to_string(),
                    language: None,
                    datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#dateTime",
                    )),
                })],
            ),
        ),
        vec![variable("s"), variable("yr")],
    );
    let test = ConformanceTest::new(
        "dt-sparql-iri-02",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(3),
    );
    runner().run_test(&test).expect("dt-sparql-iri-02 failed");
}

// ===== LIMIT + DATETIME =====

#[test]
fn test_dt_limit_01_limit_with_datetime_filter() {
    // Filter by year then LIMIT
    let ds = datetime_dataset();
    let algebra = project(
        slice(
            filter(
                bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
                expr_eq(
                    expr_fn("year", vec![expr_var("ts")]),
                    Expression::Literal(Literal {
                        value: "2024".to_string(),
                        language: None,
                        datatype: Some(oxirs_core::model::NamedNode::new_unchecked(
                            "http://www.w3.org/2001/XMLSchema#integer",
                        )),
                    }),
                ),
            ),
            None,
            Some(1),
        ),
        vec![variable("e")],
    );
    let test = ConformanceTest::new(
        "dt-limit-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(1),
    );
    runner().run_test(&test).expect("dt-limit-01 failed");
}

// ===== STR() ON DATETIME =====

#[test]
fn test_dt_str_01_str_on_datetime() {
    // STR(?ts) on a dateTime value
    let ds = datetime_dataset();
    let algebra = project(
        extend(
            bgp(vec![triple(var("e"), ex("timestamp"), var("ts"))]),
            variable("s"),
            expr_fn("str", vec![expr_var("ts")]),
        ),
        vec![variable("e"), variable("s")],
    );
    let test = ConformanceTest::new(
        "dt-str-01",
        ConformanceGroup::DateFunctions,
        algebra,
        ds,
        ConformanceResult::ResultCount(5),
    );
    runner().run_test(&test).expect("dt-str-01 failed");
}
