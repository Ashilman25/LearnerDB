#include "executor/data_chunk.hpp"

#include <algorithm>
#include <cassert>

namespace shilmandb {

DataChunk::DataChunk(const Schema& schema, size_t capacity) : schema_(schema), capacity_(capacity) {
    const auto n_cols = schema_.GetColumnCount();
    columns_.resize(n_cols);
    for (auto& col : columns_) col.reserve(capacity_);
    selection_vector_.reserve(capacity_);
}

void DataChunk::AppendTuple(const Tuple& tuple) {
    assert(!has_selection_ && "AppendTuple requires no active selection vector");
    assert(size_ < capacity_ && "AppendTuple on full chunk");

    const auto n_cols = schema_.GetColumnCount();
    for (uint32_t c = 0; c < n_cols; ++c) {
        columns_[c].push_back(tuple.GetValue(schema_, c));
    }
    ++size_;
}

Tuple DataChunk::MaterializeTuple(size_t logical_row_idx) const {
    assert(logical_row_idx < size() && "MaterializeTuple index out of range");
    const size_t physical = has_selection_ ? static_cast<size_t>(selection_vector_[logical_row_idx]) : logical_row_idx;

    const auto n_cols = columns_.size();
    std::vector<Value> values;
    values.reserve(n_cols);
    for (size_t c = 0; c < n_cols; ++c) {
        values.push_back(columns_[c][physical]);
    }
    return Tuple(std::move(values), schema_);
}

void DataChunk::SetSelectionVector(std::vector<uint32_t> sel) {
    assert((sel.empty() || *std::max_element(sel.begin(), sel.end()) < size_) && "SetSelectionVector: index out of range");
    selection_vector_ = std::move(sel);
    has_selection_ = true;
}

void DataChunk::Flatten() {
    if (!has_selection_) return;

    const size_t new_size = selection_vector_.size();
    for (size_t i = 0; i < new_size; ++i) {
        assert(selection_vector_[i] >= i && "Flatten requires selection_vector_[i] >= i for in-place rewrite");
    }
    for (auto& col : columns_) {
        for (size_t i = 0; i < new_size; ++i) {
            col[i] = col[selection_vector_[i]];
        }
        col.resize(new_size);
    }
    size_ = new_size;
    selection_vector_.clear();
    has_selection_ = false;
}

void DataChunk::Reset() {
    for (auto& col : columns_) col.clear();   // retains capacity
    selection_vector_.clear();                // retains capacity
    size_ = 0;
    has_selection_ = false;
}

}  // namespace shilmandb
