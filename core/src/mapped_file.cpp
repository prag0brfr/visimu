// Author: Paulo R. A. Gilz
#include "aoe/mapped_file.hpp"

#if defined(_WIN32)
#include "aoe/win32.hpp"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace aoe {

void MappedFile::move_from(MappedFile& other) noexcept {
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
#if defined(_WIN32)
    file_handle_ = other.file_handle_;
    map_handle_ = other.map_handle_;
    other.file_handle_ = nullptr;
    other.map_handle_ = nullptr;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif
}

#if defined(_WIN32)

bool MappedFile::open(const std::string& path) {
    close();

    // Widen to UTF-16 so paths with non-ASCII characters work; the ANSI
    // used the ANSI entry points and silently failed on such paths.
    int wlen = ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath;
    if (wlen > 0) {
        wpath.resize(static_cast<std::size_t>(wlen - 1));
        ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    }

    HANDLE file = ::CreateFileW(wpath.c_str(), GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER file_size{};
    if (!::GetFileSizeEx(file, &file_size) || file_size.QuadPart <= 0) {
        ::CloseHandle(file);
        return false;
    }

    HANDLE map = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (map == nullptr) {
        ::CloseHandle(file);
        return false;
    }

    const void* view = ::MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        ::CloseHandle(map);
        ::CloseHandle(file);
        return false;
    }

    file_handle_ = file;
    map_handle_ = map;
    data_ = static_cast<const char*>(view);
    size_ = static_cast<std::size_t>(file_size.QuadPart);
    return true;
}

void MappedFile::close() {
    if (data_) ::UnmapViewOfFile(data_);
    if (map_handle_) ::CloseHandle(static_cast<HANDLE>(map_handle_));
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE)
        ::CloseHandle(static_cast<HANDLE>(file_handle_));
    data_ = nullptr;
    size_ = 0;
    map_handle_ = nullptr;
    file_handle_ = nullptr;
}

void MappedFile::advise_sequential() const noexcept {
    // No direct equivalent; FILE_FLAG_SEQUENTIAL_SCAN would have to be set at
    // CreateFile time and measurably hurts the random column access we do later.
}

std::uint64_t MappedFile::mod_time(const std::string& path) noexcept {
    int wlen = ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (wlen <= 0) return 0;
    std::wstring wpath(static_cast<std::size_t>(wlen - 1), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);

    WIN32_FILE_ATTRIBUTE_DATA fad{};
    if (!::GetFileAttributesExW(wpath.c_str(), GetFileExInfoStandard, &fad)) return 0;
    ULARGE_INTEGER t{};
    t.LowPart = fad.ftLastWriteTime.dwLowDateTime;
    t.HighPart = fad.ftLastWriteTime.dwHighDateTime;
    return t.QuadPart;
}

bool MappedFile::exists(const std::string& path) noexcept {
    return mod_time(path) != 0;
}

#else  // POSIX

bool MappedFile::open(const std::string& path) {
    close();

    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) return false;

    struct stat st {};
    if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    void* view = ::mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ,
                        MAP_PRIVATE, fd_, 0);
    if (view == MAP_FAILED) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    data_ = static_cast<const char*>(view);
    size_ = static_cast<std::size_t>(st.st_size);
    return true;
}

void MappedFile::close() {
    if (data_) ::munmap(const_cast<char*>(data_), size_);
    if (fd_ >= 0) ::close(fd_);
    data_ = nullptr;
    size_ = 0;
    fd_ = -1;
}

void MappedFile::advise_sequential() const noexcept {
#if defined(MADV_SEQUENTIAL)
    if (data_)
        ::madvise(const_cast<char*>(data_), size_, MADV_SEQUENTIAL | MADV_WILLNEED);
#endif
}

std::uint64_t MappedFile::mod_time(const std::string& path) noexcept {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(st.st_mtimespec.tv_sec) * 1000000000ull +
           static_cast<std::uint64_t>(st.st_mtimespec.tv_nsec);
#else
    return static_cast<std::uint64_t>(st.st_mtim.tv_sec) * 1000000000ull +
           static_cast<std::uint64_t>(st.st_mtim.tv_nsec);
#endif
}

bool MappedFile::exists(const std::string& path) noexcept {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

#endif

}  // namespace aoe
