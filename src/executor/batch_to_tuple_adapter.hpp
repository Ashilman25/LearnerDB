#pragma once
#include "executor/executor.hpp"

#include <memory>

namespace shilmandb {

class BatchToTupleAdapter : public Executor {
public:
    BatchToTupleAdapter(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child);

    void Init() override;
    bool Next(Tuple* tuple) override;
    void Close() override;

private:
    std::unique_ptr<Executor> child_;
    DataChunk current_chunk_;
    size_t cursor_{0};
    bool initialized_{false};
};

}  // namespace shilmandb
