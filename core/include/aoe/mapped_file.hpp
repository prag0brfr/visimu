// Author: Paulo R. A. Gilz
//
// Portable read-only memory-mapped file.
//
// One RAII type over mmap(2) on POSIX and CreateFileMapping/MapViewOfFile on
// Windows, so nothing above this file has to know which platform it is on.
// Mapping (rather than reading) matters here: AOE result
// files are routinely hundreds of megabytes and we only ever touch the columns
// the user actually plots, so we want the OS page cache to do the demand paging
// for us instead of copying the whole file into the heap.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace aoe {

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile() { close(); }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept { move_from(other); }
    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            close();
            move_from(other);
        }
        return *this;
    }

    // Maps `path` read-only. Returns false on any failure (missing file, empty
    // file, mapping refused). An empty file is deliberately treated as failure:
    // there is nothing to parse and every caller would have to special-case it.
    bool open(const std::string& path);
    void close();

    bool is_open() const noexcept { return data_ != nullptr; }
    const char* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    const char* begin() const noexcept { return data_; }
    const char* end() const noexcept { return data_ + size_; }

    // Advises the kernel that the mapping will be read front-to-back. On Linux
    // this issues madvise(MADV_SEQUENTIAL|MADV_WILLNEED) which measurably cuts
    // the cost of the first full scan; a no-op elsewhere.
    void advise_sequential() const noexcept;

    // Last-write time as an opaque monotonically-increasing integer, used only
    // to detect that a file changed on disk. Units differ per platform (100ns
    // ticks on Windows, nanoseconds on POSIX) and must not be interpreted.
    static std::uint64_t mod_time(const std::string& path) noexcept;
    static bool exists(const std::string& path) noexcept;

private:
    void move_from(MappedFile& other) noexcept;

    const char* data_ = nullptr;
    std::size_t size_ = 0;

#if defined(_WIN32)
    void* file_handle_ = nullptr;  // HANDLE
    void* map_handle_ = nullptr;   // HANDLE
#else
    int fd_ = -1;
#endif
};

}  // namespace aoe
