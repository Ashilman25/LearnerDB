#include "buffer/lru_eviction_policy.hpp"

namespace shilmandb {

LRUEvictionPolicy::LRUEvictionPolicy(size_t num_frames) : num_frames_(num_frames) {}

void LRUEvictionPolicy::RecordAccess(frame_id_t frame_id) {
    // if already tracked move to MRU (back)
    if (auto it = map_.find(frame_id); it != map_.end()) {
        lru_list_.erase(it->second);
        lru_list_.push_back(frame_id);
        it->second = std::prev(lru_list_.end());
        return;
    }

    // new frame, add to MRU 
    lru_list_.push_back(frame_id);
    map_[frame_id] = std::prev(lru_list_.end());
    ++evictable_count_;
}

void LRUEvictionPolicy::SetEvictable(frame_id_t frame_id, bool evictable) {
    //ignore frames not tracked
    if (map_.find(frame_id) == map_.end()) { return; }

    bool was_non_evictable = non_evictable_.count(frame_id) > 0;

    if (evictable && was_non_evictable) {
        non_evictable_.erase(frame_id);
        ++evictable_count_;
    } else if (!evictable && !was_non_evictable) {
        non_evictable_.insert(frame_id);
        --evictable_count_;
    }
}

std::optional<frame_id_t> LRUEvictionPolicy::Evict() {
    for (auto it = lru_list_.begin(); it != lru_list_.end(); ++it) {
        if (non_evictable_.count(*it) == 0) {
            frame_id_t victim = *it;
            map_.erase(victim);
            lru_list_.erase(it);
            --evictable_count_;
            return victim;
        }
    }
    return std::nullopt;
}

void LRUEvictionPolicy::Remove(frame_id_t frame_id) {
    if (auto it = map_.find(frame_id); it != map_.end()) {
        if (non_evictable_.count(frame_id) == 0) {
            --evictable_count_;
        }
        lru_list_.erase(it->second);
        map_.erase(it);
    }
    non_evictable_.erase(frame_id);
}

size_t LRUEvictionPolicy::Size() const {
    return evictable_count_;
}

}  // namespace shilmandb
