#include "storage/slotted_page.hpp"

namespace shilmandb {


PageHeader* SlottedPage::GetHeader(char* page_data) {
    return reinterpret_cast<PageHeader*>(page_data);
}

const PageHeader* SlottedPage::GetHeader(const char* page_data) {
    return reinterpret_cast<const PageHeader*>(page_data);
}

SlotEntry* SlottedPage::GetSlotEntry(char* page_data, uint16_t slot_id) {
    return reinterpret_cast<SlotEntry*>(page_data + PAGE_HEADER_SIZE + slot_id * SLOT_ENTRY_SIZE);
}

const SlotEntry* SlottedPage::GetSlotEntry(const char* page_data, uint16_t slot_id) {
    return reinterpret_cast<const SlotEntry*>(page_data + PAGE_HEADER_SIZE + slot_id * SLOT_ENTRY_SIZE);
}



void SlottedPage::Init(char* page_data, page_id_t page_id) {
    std::memset(page_data, 0, PAGE_SIZE);
    auto* header = GetHeader(page_data);
    header->page_id = page_id;
    header->num_slots = 0;
    header->free_space_offset = static_cast<uint16_t>(PAGE_HEADER_SIZE);
    header->next_page_id = INVALID_PAGE_ID;
    header->flags = 0;
}



uint16_t SlottedPage::FindTupleDataStart(const char* page_data, uint16_t num_slots) {
    uint16_t lowest = static_cast<uint16_t>(PAGE_SIZE);
    for (uint16_t i = 0; i < num_slots; ++i) {
        auto* slot = GetSlotEntry(page_data, i);
        if (slot->length > 0 && slot->offset < lowest) {
            lowest = slot->offset;
        }
    }
    return lowest;
}

int SlottedPage::InsertTuple(char* page_data, const char* tuple_data, uint16_t tuple_length) {
    if (tuple_length == 0) { return -1; }

    auto* header = GetHeader(page_data);
    uint16_t tuple_data_start = FindTupleDataStart(page_data, header->num_slots);
    uint16_t slot_dir_end = header->free_space_offset + static_cast<uint16_t>(SLOT_ENTRY_SIZE);

    if (slot_dir_end + tuple_length > tuple_data_start) {
        return -1;  // not enough space
    }

    // Place tuple at end of free region - tuple data grows leftward
    uint16_t tuple_offset = tuple_data_start - tuple_length;
    std::memcpy(page_data + tuple_offset, tuple_data, tuple_length);

    // Write new slot entry
    auto slot_id = header->num_slots;
    auto* slot = GetSlotEntry(page_data, slot_id);
    slot->offset = tuple_offset;
    slot->length = tuple_length;

    header->num_slots++;
    header->free_space_offset += static_cast<uint16_t>(SLOT_ENTRY_SIZE);

    return static_cast<int>(slot_id);
}


bool SlottedPage::GetTuple(const char* page_data, uint16_t slot_id, char* out_data, uint16_t* out_length) {
    auto* header = GetHeader(page_data);
    if (slot_id >= header->num_slots) { return false; }

    auto* slot = GetSlotEntry(page_data, slot_id);
    if (slot->length == 0) { return false; }

    std::memcpy(out_data, page_data + slot->offset, slot->length);
    *out_length = slot->length;
    return true;
}


bool SlottedPage::DeleteTuple(char* page_data, uint16_t slot_id) {
    auto* header = GetHeader(page_data);
    if (slot_id >= header->num_slots) { return false; }

    auto* slot = GetSlotEntry(page_data, slot_id);
    if (slot->length == 0) { return false; }  // already deleted

    slot->length = 0;
    return true;
}


void SlottedPage::Compact(char* page_data) {
    auto* header = GetHeader(page_data);


    char temp[PAGE_SIZE];
    std::memcpy(temp, page_data, PAGE_SIZE);

    uint16_t write_pos = static_cast<uint16_t>(PAGE_SIZE);
    for (uint16_t i = 0; i < header->num_slots; ++i) {
        auto* src_slot = GetSlotEntry(temp, i);
        if (src_slot->length == 0) { continue; }

        write_pos -= src_slot->length;
        std::memcpy(page_data + write_pos, temp + src_slot->offset, src_slot->length);

        auto* dst_slot = GetSlotEntry(page_data, i);
        dst_slot->offset = write_pos;
    }

    //reclaimed gap between slot dir and tuple data
    std::memset(page_data + header->free_space_offset, 0, write_pos - header->free_space_offset);
}


uint16_t SlottedPage::GetFreeSpace(const char* page_data) {
    auto* header = GetHeader(page_data);
    uint16_t tuple_data_start = FindTupleDataStart(page_data, header->num_slots);
    uint16_t slot_dir_end = header->free_space_offset + static_cast<uint16_t>(SLOT_ENTRY_SIZE);

    if (slot_dir_end >= tuple_data_start) { return 0; }
    return tuple_data_start - slot_dir_end;
}

}  // namespace shilmandb
