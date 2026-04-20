#include "executor/batch_to_tuple_adapter.hpp"

#include <cassert>

namespace shilmandb {

BatchToTupleAdapter::BatchToTupleAdapter(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child)
    : Executor(plan, ctx), child_(std::move(child)), current_chunk_(child_->GetOutputSchema()) {}

void BatchToTupleAdapter::Init() {
    child_->Init();
    current_chunk_.Reset();
    cursor_ = 0;
    initialized_ = true;
}

bool BatchToTupleAdapter::Next(Tuple* tuple) {
    assert(initialized_ && "Next() called before Init()");
    while (cursor_ >= current_chunk_.size()) {
        current_chunk_.Reset();
        cursor_ = 0;
        if (!child_->NextBatch(&current_chunk_)) return false;
    }
    *tuple = current_chunk_.MaterializeTuple(cursor_++);
    return true;
}

void BatchToTupleAdapter::Close() {
    child_->Close();
    current_chunk_.Reset();
    cursor_ = 0;
    initialized_ = false;
}

}  // namespace shilmandb
