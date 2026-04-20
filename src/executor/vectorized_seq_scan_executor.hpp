#pragma once
#include "executor/executor.hpp"
#include "storage/table_heap.hpp"
#include <optional>

namespace shilmandb {

class VectorizedSeqScanExecutor : public Executor {
public:
    VectorizedSeqScanExecutor(const PlanNode* plan, ExecutorContext* ctx);

    void Init() override;
    bool Next(Tuple* tuple) override;
    bool NextBatch(DataChunk* chunk) override;
    void Close() override;

private:
    std::optional<TableHeap::Iterator> iter_;
    std::optional<TableHeap::Iterator> end_;
    bool initialized_{false};

    // Used only when a tuple-at-a-time parent drains via Next().
    DataChunk buffer_;
    size_t buffer_cursor_{0};
};

}  // namespace shilmandb
