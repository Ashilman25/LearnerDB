#pragma once
#include <stdexcept>
#include <string>

namespace shilmandb {


class ParseException : public std::runtime_error {

public:

    ParseException(const std::string& message, size_t line, size_t column) : std::runtime_error("Parse error at line " + std::to_string(line) + ", column " + std::to_string(column) + ": " + message), line_(line), column_(column) {}

    size_t GetLine() const {return line_;}
    size_t GetColumn() const {return column_;}

private:
    size_t line_;
    size_t column_;

};



}