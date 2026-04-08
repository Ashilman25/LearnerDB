#pragma once
#include "buffer/eviction_policy.hpp"
#include <list>
#include <unordered_map>
#include <unordered_set>

namespace shilmandb {

class LRUEvictionPolicy : public EvictionPolicy {
public:
    explicit LRUEvictionPolicy(size_t num_frames);

    void RecordAccess(frame_id_t frame_id) override;
    void SetEvictable(frame_id_t frame_id, bool evictable) override;
    
    [[nodiscard]] std::optional<frame_id_t> Evict() override;
    void Remove(frame_id_t frame_id) override;

    [[nodiscard]] size_t Size() const override;

private:
    size_t num_frames_;
    std::list<frame_id_t> lru_list_;  // front = LRU (evict first), back = MRU
    std::unordered_map<frame_id_t, std::list<frame_id_t>::iterator> map_;
    std::unordered_set<frame_id_t> non_evictable_;
    size_t evictable_count_{0};
};

}  // namespace shilmandb
