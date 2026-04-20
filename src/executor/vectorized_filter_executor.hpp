#pragma once
#include "executor/executor.hpp"
#include "parser/ast.hpp"

#include <memory>

namespace shilmandb {

class VectorizedFilterExecutor : public Executor {
public:
    VectorizedFilterExecutor(const PlanNode* plan, ExecutorContext* ctx, std::unique_ptr<Executor> child);

    void Init() override;
    bool Next(Tuple* tuple) override;
    bool NextBatch(DataChunk* chunk) override;
    void Close() override;

private:
    const Expression* predicate_;   // owned by FilterPlanNode
    std::unique_ptr<Executor> child_;
    bool initialized_{false};

    DataChunk buffer_;
    size_t buffer_cursor_{0};
};

}  // namespace shilmandb
