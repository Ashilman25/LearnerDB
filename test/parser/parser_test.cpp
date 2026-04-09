#include <gtest/gtest.h>
#include "parser/parser.hpp"
#include "parser/parse_exception.hpp"
#include "types/value.hpp"

namespace shilmandb {

// Helper: downcast Expression* to a specific type
template <typename T>
const T* As(const Expression* expr) {
    return dynamic_cast<const T*>(expr);
}

// -----------------------------------------------------------------------
// Test 1: SimpleSelectStar
// -----------------------------------------------------------------------
TEST(ParserTest, SimpleSelectStar) {
    Parser parser("SELECT * FROM lineitem");
    auto stmt = parser.Parse();

    ASSERT_EQ(stmt->select_list.size(), 1u);
    ASSERT_NE(As<StarExpr>(stmt->select_list[0].expr.get()), nullptr);

    ASSERT_EQ(stmt->from_clause.size(), 1u);
    EXPECT_EQ(stmt->from_clause[0].table_name, "lineitem");
    EXPECT_FALSE(stmt->from_clause[0].alias.has_value());

    EXPECT_TRUE(stmt->joins.empty());
    EXPECT_EQ(stmt->where_clause, nullptr);
    EXPECT_TRUE(stmt->group_by.empty());
    EXPECT_EQ(stmt->having, nullptr);
    EXPECT_TRUE(stmt->order_by.empty());
    EXPECT_FALSE(stmt->limit.has_value());
}

// -----------------------------------------------------------------------
// Test 2: WhereClause
// -----------------------------------------------------------------------
TEST(ParserTest, WhereClause) {
    Parser parser("SELECT * FROM lineitem WHERE l_quantity > 25");
    auto stmt = parser.Parse();

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* bin = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(bin, nullptr);
    EXPECT_EQ(bin->op, BinaryOp::Op::GT);

    auto* left = As<ColumnRef>(bin->left.get());
    ASSERT_NE(left, nullptr);
    EXPECT_EQ(left->column_name, "l_quantity");
    EXPECT_FALSE(left->table_name.has_value());

    auto* right = As<Literal>(bin->right.get());
    ASSERT_NE(right, nullptr);
    EXPECT_EQ(right->value.type_, TypeId::INTEGER);
    EXPECT_EQ(right->value.integer_, 25);
}

// -----------------------------------------------------------------------
// Test 3: JoinQuery
// -----------------------------------------------------------------------
TEST(ParserTest, JoinQuery) {
    Parser parser("SELECT * FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey");
    auto stmt = parser.Parse();

    ASSERT_EQ(stmt->from_clause.size(), 1u);
    EXPECT_EQ(stmt->from_clause[0].table_name, "orders");
    ASSERT_TRUE(stmt->from_clause[0].alias.has_value());
    EXPECT_EQ(stmt->from_clause[0].alias.value(), "o");

    ASSERT_EQ(stmt->joins.size(), 1u);
    EXPECT_EQ(stmt->joins[0].join_type, JoinType::INNER);
    EXPECT_EQ(stmt->joins[0].right_table.table_name, "customer");
    ASSERT_TRUE(stmt->joins[0].right_table.alias.has_value());
    EXPECT_EQ(stmt->joins[0].right_table.alias.value(), "c");

    auto* cond = As<BinaryOp>(stmt->joins[0].on_condition.get());
    ASSERT_NE(cond, nullptr);
    EXPECT_EQ(cond->op, BinaryOp::Op::EQ);

    auto* lhs = As<ColumnRef>(cond->left.get());
    ASSERT_NE(lhs, nullptr);
    ASSERT_TRUE(lhs->table_name.has_value());
    EXPECT_EQ(lhs->table_name.value(), "o");
    EXPECT_EQ(lhs->column_name, "o_custkey");

    auto* rhs = As<ColumnRef>(cond->right.get());
    ASSERT_NE(rhs, nullptr);
    ASSERT_TRUE(rhs->table_name.has_value());
    EXPECT_EQ(rhs->table_name.value(), "c");
    EXPECT_EQ(rhs->column_name, "c_custkey");
}

// -----------------------------------------------------------------------
// Test 4: AggregatesAndGroupBy
// -----------------------------------------------------------------------
TEST(ParserTest, AggregatesAndGroupBy) {
    Parser parser(
        "SELECT COUNT(*), SUM(l_extendedprice) FROM lineitem "
        "GROUP BY l_returnflag ORDER BY l_returnflag");
    auto stmt = parser.Parse();

    ASSERT_EQ(stmt->select_list.size(), 2u);

    auto* count_agg = As<Aggregate>(stmt->select_list[0].expr.get());
    ASSERT_NE(count_agg, nullptr);
    EXPECT_EQ(count_agg->func, Aggregate::Func::COUNT);
    EXPECT_EQ(count_agg->arg, nullptr);

    auto* sum_agg = As<Aggregate>(stmt->select_list[1].expr.get());
    ASSERT_NE(sum_agg, nullptr);
    EXPECT_EQ(sum_agg->func, Aggregate::Func::SUM);
    auto* sum_arg = As<ColumnRef>(sum_agg->arg.get());
    ASSERT_NE(sum_arg, nullptr);
    EXPECT_EQ(sum_arg->column_name, "l_extendedprice");

    ASSERT_EQ(stmt->group_by.size(), 1u);
    auto* gb = As<ColumnRef>(stmt->group_by[0].get());
    ASSERT_NE(gb, nullptr);
    EXPECT_EQ(gb->column_name, "l_returnflag");

    ASSERT_EQ(stmt->order_by.size(), 1u);
    auto* ob = As<ColumnRef>(stmt->order_by[0].expr.get());
    ASSERT_NE(ob, nullptr);
    EXPECT_EQ(ob->column_name, "l_returnflag");
    EXPECT_TRUE(stmt->order_by[0].ascending);
}

// -----------------------------------------------------------------------
// Test 5: TPCHQuery1
// -----------------------------------------------------------------------
TEST(ParserTest, TPCHQuery1) {
    const std::string q1 = R"(
        SELECT
            l_returnflag,
            l_linestatus,
            SUM(l_quantity) AS sum_qty,
            SUM(l_extendedprice) AS sum_base_price,
            SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
            SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
            AVG(l_quantity) AS avg_qty,
            AVG(l_extendedprice) AS avg_price,
            AVG(l_discount) AS avg_disc,
            COUNT(*) AS count_order
        FROM
            lineitem
        WHERE
            l_shipdate <= DATE '1998-12-01'
        GROUP BY
            l_returnflag,
            l_linestatus
        ORDER BY
            l_returnflag,
            l_linestatus
    )";
    Parser parser(q1);
    auto stmt = parser.Parse();

    EXPECT_EQ(stmt->select_list.size(), 10u);

    ASSERT_NE(As<ColumnRef>(stmt->select_list[0].expr.get()), nullptr);
    ASSERT_NE(As<ColumnRef>(stmt->select_list[1].expr.get()), nullptr);

    auto* last = As<Aggregate>(stmt->select_list[9].expr.get());
    ASSERT_NE(last, nullptr);
    EXPECT_EQ(last->func, Aggregate::Func::COUNT);
    EXPECT_EQ(last->arg, nullptr);
    ASSERT_TRUE(stmt->select_list[9].alias.has_value());
    EXPECT_EQ(stmt->select_list[9].alias.value(), "count_order");

    ASSERT_EQ(stmt->from_clause.size(), 1u);
    EXPECT_EQ(stmt->from_clause[0].table_name, "lineitem");

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* where = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(where, nullptr);
    EXPECT_EQ(where->op, BinaryOp::Op::LTE);
    auto* date_lit = As<Literal>(where->right.get());
    ASSERT_NE(date_lit, nullptr);
    EXPECT_EQ(date_lit->value.type_, TypeId::DATE);

    EXPECT_EQ(stmt->group_by.size(), 2u);
    EXPECT_EQ(stmt->order_by.size(), 2u);
}

// -----------------------------------------------------------------------
// Test 6: TPCHQuery3
// -----------------------------------------------------------------------
TEST(ParserTest, TPCHQuery3) {
    const std::string q3 = R"(
        SELECT
            l_orderkey,
            SUM(l_extendedprice * (1 - l_discount)) AS revenue,
            o_orderdate,
            o_shippriority
        FROM
            customer c
            JOIN orders o ON c.c_custkey = o.o_custkey
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
        WHERE
            c.c_mktsegment = 'BUILDING'
            AND o.o_orderdate < DATE '1995-03-15'
            AND l.l_shipdate > DATE '1995-03-15'
        GROUP BY
            l_orderkey,
            o_orderdate,
            o_shippriority
        ORDER BY
            revenue DESC,
            o_orderdate
        LIMIT 10
    )";
    Parser parser(q3);
    auto stmt = parser.Parse();

    EXPECT_EQ(stmt->from_clause.size(), 1u);
    EXPECT_EQ(stmt->joins.size(), 2u);

    EXPECT_EQ(stmt->group_by.size(), 3u);

    ASSERT_EQ(stmt->order_by.size(), 2u);
    EXPECT_FALSE(stmt->order_by[0].ascending);
    EXPECT_TRUE(stmt->order_by[1].ascending);

    ASSERT_TRUE(stmt->limit.has_value());
    EXPECT_EQ(stmt->limit.value(), 10);
}

// -----------------------------------------------------------------------
// Test 7: TPCHQuery5
// -----------------------------------------------------------------------
TEST(ParserTest, TPCHQuery5) {
    const std::string q5 = R"(
        SELECT
            n_name,
            SUM(l_extendedprice * (1 - l_discount)) AS revenue
        FROM
            customer c
            JOIN orders o ON c.c_custkey = o.o_custkey
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
            JOIN supplier s ON l.l_suppkey = s.s_suppkey
            JOIN nation n ON c.c_nationkey = n.n_nationkey
        WHERE
            n.n_name = 'FRANCE'
            AND o.o_orderdate >= DATE '1994-01-01'
            AND o.o_orderdate < DATE '1995-01-01'
        GROUP BY
            n_name
        ORDER BY
            revenue DESC
    )";
    Parser parser(q5);
    auto stmt = parser.Parse();

    EXPECT_EQ(stmt->from_clause.size(), 1u);
    EXPECT_EQ(stmt->joins.size(), 4u);

    EXPECT_EQ(stmt->joins[0].right_table.table_name, "orders");
    EXPECT_EQ(stmt->joins[1].right_table.table_name, "lineitem");
    EXPECT_EQ(stmt->joins[2].right_table.table_name, "supplier");
    EXPECT_EQ(stmt->joins[3].right_table.table_name, "nation");
}

// -----------------------------------------------------------------------
// Test 8: TPCHQuery6
// -----------------------------------------------------------------------
TEST(ParserTest, TPCHQuery6) {
    const std::string q6 = R"(
        SELECT
            SUM(l_extendedprice * l_discount) AS revenue
        FROM
            lineitem
        WHERE
            l_shipdate >= DATE '1994-01-01'
            AND l_shipdate < DATE '1995-01-01'
            AND l_discount >= 0.05
            AND l_discount <= 0.07
            AND l_quantity < 24
    )";
    Parser parser(q6);
    auto stmt = parser.Parse();

    ASSERT_EQ(stmt->select_list.size(), 1u);
    auto* agg = As<Aggregate>(stmt->select_list[0].expr.get());
    ASSERT_NE(agg, nullptr);
    EXPECT_EQ(agg->func, Aggregate::Func::SUM);

    EXPECT_EQ(stmt->from_clause.size(), 1u);
    EXPECT_EQ(stmt->from_clause[0].table_name, "lineitem");
    EXPECT_TRUE(stmt->joins.empty());

    ASSERT_NE(stmt->where_clause, nullptr);
    auto* top = As<BinaryOp>(stmt->where_clause.get());
    ASSERT_NE(top, nullptr);
    EXPECT_EQ(top->op, BinaryOp::Op::AND);

    // Descend to verify DECIMAL literal in the chain
    // top->left is the inner AND chain, its right is l_discount <= 0.07
    auto* inner = As<BinaryOp>(top->left.get());
    ASSERT_NE(inner, nullptr);
    EXPECT_EQ(inner->op, BinaryOp::Op::AND);
    auto* discount_check = As<BinaryOp>(inner->right.get());
    ASSERT_NE(discount_check, nullptr);
    EXPECT_EQ(discount_check->op, BinaryOp::Op::LTE);
    auto* dec_lit = As<Literal>(discount_check->right.get());
    ASSERT_NE(dec_lit, nullptr);
    EXPECT_EQ(dec_lit->value.type_, TypeId::DECIMAL);
    EXPECT_DOUBLE_EQ(dec_lit->value.decimal_, 0.07);
}

// -----------------------------------------------------------------------
// Test 9: Aliases
// -----------------------------------------------------------------------
TEST(ParserTest, Aliases) {
    Parser parser("SELECT a AS b FROM orders o");
    auto stmt = parser.Parse();

    ASSERT_EQ(stmt->select_list.size(), 1u);
    ASSERT_TRUE(stmt->select_list[0].alias.has_value());
    EXPECT_EQ(stmt->select_list[0].alias.value(), "b");

    auto* col = As<ColumnRef>(stmt->select_list[0].expr.get());
    ASSERT_NE(col, nullptr);
    EXPECT_EQ(col->column_name, "a");

    ASSERT_EQ(stmt->from_clause.size(), 1u);
    EXPECT_EQ(stmt->from_clause[0].table_name, "orders");
    ASSERT_TRUE(stmt->from_clause[0].alias.has_value());
    EXPECT_EQ(stmt->from_clause[0].alias.value(), "o");
}

// -----------------------------------------------------------------------
// Test 10: MalformedSQLThrows
// -----------------------------------------------------------------------
TEST(ParserTest, MalformedSQLThrows) {
    Parser parser("SELECT FROM");
    EXPECT_THROW((void)parser.Parse(), ParseException);
}

// -----------------------------------------------------------------------
// Test 11: LeftJoin
// -----------------------------------------------------------------------
TEST(ParserTest, LeftJoin) {
    Parser parser("SELECT * FROM orders o LEFT JOIN customer c ON o.o_custkey = c.c_custkey");
    auto stmt = parser.Parse();

    ASSERT_EQ(stmt->joins.size(), 1u);
    EXPECT_EQ(stmt->joins[0].join_type, JoinType::LEFT);
    EXPECT_EQ(stmt->joins[0].right_table.table_name, "customer");
}

// -----------------------------------------------------------------------
// Test 12: JoinWithoutOnThrows
// -----------------------------------------------------------------------
TEST(ParserTest, JoinWithoutOnThrows) {
    Parser parser("SELECT * FROM a JOIN b");
    EXPECT_THROW((void)parser.Parse(), ParseException);
}

// -----------------------------------------------------------------------
// Test 13: DateWithoutStringThrows
// -----------------------------------------------------------------------
TEST(ParserTest, DateWithoutStringThrows) {
    Parser parser("SELECT * FROM t WHERE d = DATE 42");
    EXPECT_THROW((void)parser.Parse(), ParseException);
}

}  // namespace shilmandb
