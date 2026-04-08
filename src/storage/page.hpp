#pragma once
#include "common/config.hpp"
#include <cstring>

namespace shilmandb {


class Page {

public:
    char data_[PAGE_SIZE]{};
    page_id_t page_id_{INVALID_PAGE_ID};
    int pin_count_{0};
    bool is_dirty_{false};

    void ResetMemory() {
        std::memset(data_, 0, PAGE_SIZE);
        page_id_ = INVALID_PAGE_ID;
        pin_count_ = 0;
        is_dirty_ = false;
    }

    char* GetData() { return data_; }
    const char* GetData() const { return data_; }
    page_id_t GetPageId() const { return page_id_; }

};

  
}