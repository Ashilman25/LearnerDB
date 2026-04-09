#pragma once
#include "types/value.hpp"
#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace shilmandb {

enum class ExprType {
    COLUMN_REF, LITERAL, BINARY_OP, UNARY_OP, AGGREGATE, STAR
};

struct Expression {
    ExprType type;
    virtual ~Expression() = default;

protected:
    explicit Expression(ExprType t) : type(t) {}
};

struct ColumnRef : Expression {
    std::optional<std::string> table_name;
    std::string column_name;
    ColumnRef() : Expression(ExprType::COLUMN_REF) {}
};

struct Literal : Expression {
    Value value;
    Literal() : Expression(ExprType::LITERAL) {}
};

struct BinaryOp : Expression {
    enum class Op {EQ, NEQ, LT, GT, LTE, GTE, AND, OR, ADD, SUB, MUL, DIV};
    Op op;

    std::unique_ptr<Expression> left;
    std::unique_ptr<Expression> right;
    BinaryOp() : Expression(ExprType::BINARY_OP) {}
};

struct UnaryOp : Expression {
    enum class Op {NOT, NEGATE};
    Op op;
    
    std::unique_ptr<Expression> operand;
    UnaryOp() : Expression(ExprType::UNARY_OP) {}
};

struct Aggregate : Expression {
    enum class Func {COUNT, SUM, AVG, MIN, MAX};
    Func func;

    std::unique_ptr<Expression> arg; //nullptr for COUNT(*)
    Aggregate() : Expression(ExprType::AGGREGATE) {}
};


struct StarExpr : Expression {
    StarExpr() : Expression(ExprType::STAR) {}
};


//statement components

struct SelectItem {
    std::unique_ptr<Expression> expr;
    std::optional<std::string> alias;
};

struct TableRef {
    std::string table_name;
    std::optional<std::string> alias;
};

enum class JoinType { INNER, LEFT };

struct JoinClause {
    JoinType join_type{JoinType::INNER};
    TableRef right_table;
    std::unique_ptr<Expression> on_condition;
};

struct OrderByItem {
    std::unique_ptr<Expression> expr;
    bool ascending{true}; //true = asc, false = desc
};


struct SelectStatement {
    std::vector<SelectItem> select_list;
    std::vector<TableRef> from_clause;
    std::vector<JoinClause> joins;
    std::unique_ptr<Expression> where_clause;
    std::vector<std::unique_ptr<Expression>> group_by;
    std::unique_ptr<Expression> having;
    std::vector<OrderByItem> order_by;
    std::optional<int64_t> limit;
};


} //namespace shilmandb