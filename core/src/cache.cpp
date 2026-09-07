// Author: Paulo R. A. Gilz
#include "aoe/cache.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(_WIN32)
#include <wchar.h>  // _wfopen
#include "aoe/win32.hpp"
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace aoe {

const char kCacheMagic[8] = {'A', 'O', 'E', 'C', 'A', 'C', 'H', 'E'};

namespace {

// The sidecar stores IEEE 754 binary32 verbatim. Every platform this targets
// uses it, but a compiler where it did not hold would produce a cache that
// reads back as garbage rather than failing to build, so say so out loud.
static_assert(sizeof(float) == 4, "the cache stores float32");
static_assert(sizeof(double) == 8, "the cache stores 64-bit integers");

constexpr std::size_t kHeaderBytes = 128;
constexpr std::size_t kColumnRecordBytes = 16;
constexpr std::size_t kSimRecordBytes = 24;
constexpr std::size_t kDataAlignment = 64;  // one cache line

// Hashing the whole of a 500 MB file would cost more than the parse the cache
// exists to avoid, so above this size the hash samples instead. Below it, the
// file is read in full and the hash is exact.
constexpr std::uint64_t kFullHashLimit = 4ull << 20;   // 4 MiB
constexpr std::size_t kHashWindowBytes = 64u << 10;    // 64 KiB
constexpr std::size_t kHashWindowCount = 64;           // 4 MiB of reading, at most

// A cache is only ever a saving, so refuse to produce an absurd one rather than
// filling a disk. Two hundred million (parameter, run) pairs is far past any
// real file and keeps the index arithmetic clear of overflow.
constexpr std::uint64_t kMaxBlocks = 200ull * 1000ull * 1000ull;

// ------------------------------------------------------------ byte order

inline void put_u32(unsigned char* p, std::uint32_t v) noexcept {
    p[0] = static_cast<unsigned char>(v);
    p[1] = static_cast<unsigned char>(v >> 8);
    p[2] = static_cast<unsigned char>(v >> 16);
    p[3] = static_cast<unsigned char>(v >> 24);
}

inline void put_u64(unsigned char* p, std::uint64_t v) noexcept {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<unsigned char>(v >> (8 * i));
}

inline std::uint32_t get_u32(const unsigned char* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint64_t get_u64(const unsigned char* p) noexcept {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return v;
}

// ------------------------------------------------------------------ hash

// FNV-1a, 64-bit. Chosen for being short enough to read and verify at a glance;
// this guards against a file changing underneath us, not against an adversary
// crafting a collision.
constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

inline std::uint64_t fnv1a(const void* data, std::size_t bytes,
                           std::uint64_t seed) noexcept {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    std::uint64_t h = seed;
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= kFnvPrime;
    }
    return h;
}

inline std::uint64_t fnv1a_u64(std::uint64_t value, std::uint64_t seed) noexcept {
    unsigned char buf[8];
    put_u64(buf, value);
    return fnv1a(buf, sizeof(buf), seed);
}

// ------------------------------------------------------- platform files

#if defined(_WIN32)

std::wstring widen(const std::string& utf8) {
    if (utf8.empty()) return std::wstring();
    const int len = ::MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    if (len <= 0) return std::wstring();
    std::wstring wide(static_cast<std::size_t>(len - 1), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wide[0], len);
    return wide;
}

std::string narrow(const std::wstring& wide) {
    if (wide.empty()) return std::string();
    const int len = ::WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0,
                                          nullptr, nullptr);
    if (len <= 0) return std::string();
    std::string utf8(static_cast<std::size_t>(len - 1), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, &utf8[0], len, nullptr, nullptr);
    return utf8;
}

// The narrow CRT entry points interpret a path in the active code page, so a
// results directory with an accented name -- routine on the machines this runs
// on -- silently fails to open. Everything goes through the wide ones.
std::FILE* open_file(const std::string& path, const wchar_t* mode) noexcept {
    const std::wstring wide = widen(path);
    if (wide.empty()) return nullptr;
    return ::_wfopen(wide.c_str(), mode);
}

bool seek_to(std::FILE* f, std::uint64_t offset) noexcept {
    return ::_fseeki64(f, static_cast<__int64>(offset), SEEK_SET) == 0;
}

#else

std::FILE* open_file(const std::string& path, const char* mode) noexcept {
    return std::fopen(path.c_str(), mode);
}

bool seek_to(std::FILE* f, std::uint64_t offset) noexcept {
#if defined(__APPLE__) || defined(_LARGEFILE_SOURCE) || defined(__linux__)
    return ::fseeko(f, static_cast<off_t>(offset), SEEK_SET) == 0;
#else
    return std::fseek(f, static_cast<long>(offset), SEEK_SET) == 0;
#endif
}

#endif

// One mode string per platform, so the call sites read the same on both.
#if defined(_WIN32)
#define AOE_MODE_WRITE L"w+b"
#define AOE_MODE_READ L"rb"
#else
#define AOE_MODE_WRITE "w+b"
#define AOE_MODE_READ "rb"
#endif

std::string environment(const char* name) {
#if defined(_WIN32)
    // getenv is not thread-safe against a concurrent _putenv and MSVC warns on
    // it; the wide API also gets a value that is not representable in the
    // active code page, which %LOCALAPPDATA% can be.
    const std::wstring wname = widen(name);
    if (wname.empty()) return std::string();
    const DWORD needed = ::GetEnvironmentVariableW(wname.c_str(), nullptr, 0);
    if (needed == 0) return std::string();
    std::wstring value(needed, L'\0');
    const DWORD written = ::GetEnvironmentVariableW(wname.c_str(), &value[0], needed);
    if (written == 0 || written >= needed) return std::string();
    value.resize(written);
    return narrow(value);
#else
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
#endif
}

bool directory_exists(const std::string& path) noexcept {
    if (path.empty()) return false;
#if defined(_WIN32)
    const std::wstring wide = widen(path);
    if (wide.empty()) return false;
    const DWORD attributes = ::GetFileAttributesW(wide.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES &&
           (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

bool make_one_directory(const std::string& path) noexcept {
#if defined(_WIN32)
    const std::wstring wide = widen(path);
    if (wide.empty()) return false;
    if (::CreateDirectoryW(wide.c_str(), nullptr)) return true;
    return ::GetLastError() == ERROR_ALREADY_EXISTS;
#else
    if (::mkdir(path.c_str(), 0755) == 0) return true;
    return errno == EEXIST;
#endif
}

inline bool is_separator(char c) noexcept { return c == '/' || c == '\\'; }

// Everything before the last separator, or empty when the path has none.
std::string parent_of(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return std::string();
    if (slash == 0) return path.substr(0, 1);  // "/x" -> "/"
    return path.substr(0, slash);
}

std::string basename_of(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// Renames over an existing destination.
//
// POSIX rename() replaces atomically. Windows MoveFileExW does the same given
// MOVEFILE_REPLACE_EXISTING; plain rename() there fails outright when the
// destination exists, which is the usual case -- we are replacing a cache that
// went stale.
bool replace_file(const std::string& from, const std::string& to) noexcept {
#if defined(_WIN32)
    const std::wstring wfrom = widen(from);
    const std::wstring wto = widen(to);
    if (wfrom.empty() || wto.empty()) return false;
    return ::MoveFileExW(wfrom.c_str(), wto.c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED) != 0;
#else
    return std::rename(from.c_str(), to.c_str()) == 0;
#endif
}

}  // namespace

bool remove_file(const std::string& path) noexcept {
    if (path.empty()) return true;
#if defined(_WIN32)
    const std::wstring wide = widen(path);
    if (wide.empty()) return false;
    if (::DeleteFileW(wide.c_str())) return true;
    // Captured once: GetLastError does not clear itself, but reading it twice
    // invites an edit that slips another call in between and silently reads
    // that one's code instead.
    const DWORD failure = ::GetLastError();
    return failure == ERROR_FILE_NOT_FOUND || failure == ERROR_PATH_NOT_FOUND;
#else
    if (std::remove(path.c_str()) == 0) return true;
    return errno == ENOENT;
#endif
}

bool make_directories(const std::string& dir) noexcept {
    if (dir.empty()) return false;
    if (directory_exists(dir)) return true;

    // Walk forward creating each component. Starting from the root rather than
    // recursing from the leaf keeps this to one pass and needs no stack.
    std::string partial;
    partial.reserve(dir.size());

    std::size_t i = 0;
    // Keep a leading separator, or a "C:" drive prefix, attached to the first
    // component: creating "" or "C:" is neither meaningful nor legal.
    if (i < dir.size() && is_separator(dir[i])) {
        partial.push_back(dir[i]);
        ++i;
    } else if (dir.size() >= 2 && dir[1] == ':') {
        partial.append(dir, 0, 2);
        i = 2;
        if (i < dir.size() && is_separator(dir[i])) {
            partial.push_back(dir[i]);
            ++i;
        }
    }

    while (i < dir.size()) {
        const std::size_t start = i;
        while (i < dir.size() && !is_separator(dir[i])) ++i;
        if (i > start) {
            partial.append(dir, start, i - start);
            if (!directory_exists(partial) && !make_one_directory(partial)) return false;
        }
        while (i < dir.size() && is_separator(dir[i])) {
            partial.push_back(dir[i]);
            ++i;
        }
    }
    return directory_exists(dir);
}

// ------------------------------------------------------------ fingerprint

SourceFingerprint fingerprint(const std::string& path) noexcept {
    SourceFingerprint print;

    std::FILE* f = open_file(path, AOE_MODE_READ);
    if (!f) return print;

    // Length from the handle rather than a separate stat, so the size and the
    // bytes hashed below are of the same file even if it is replaced meanwhile.
    if (!seek_to(f, 0) || std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        return print;
    }
#if defined(_WIN32)
    const long long length = ::_ftelli64(f);
#else
    const long long length = static_cast<long long>(::ftello(f));
#endif
    if (length <= 0) {
        std::fclose(f);
        return print;
    }

    const std::uint64_t size = static_cast<std::uint64_t>(length);
    print.size = size;
    print.mod_time = MappedFile::mod_time(path);

    // Seeding with the length means a file that is a strict prefix of another
    // cannot hash the same however the sampled windows happen to fall.
    std::uint64_t hash = fnv1a_u64(size, kFnvOffset);

    std::vector<char> buffer(kHashWindowBytes);

    if (size <= kFullHashLimit) {
        if (!seek_to(f, 0)) {
            std::fclose(f);
            print.size = 0;
            return print;
        }
        std::uint64_t remaining = size;
        while (remaining > 0) {
            const std::size_t want = static_cast<std::size_t>(
                std::min<std::uint64_t>(remaining, buffer.size()));
            const std::size_t got = std::fread(buffer.data(), 1, want, f);
            if (got == 0) break;
            hash = fnv1a(buffer.data(), got, hash);
            remaining -= got;
        }
    } else {
        // Evenly spaced windows, the first at the very start and the last
        // ending at the very end. The two extremes matter most: a header
        // rewrite and an append are the changes that actually happen, and both
        // land in one of them.
        const std::uint64_t span = size - kHashWindowBytes;
        for (std::size_t w = 0; w < kHashWindowCount; ++w) {
            const std::uint64_t offset =
                span * static_cast<std::uint64_t>(w) / (kHashWindowCount - 1);
            if (!seek_to(f, offset)) break;
            const std::size_t got = std::fread(buffer.data(), 1, kHashWindowBytes, f);
            if (got == 0) break;
            // Fold the offset in as well, so the same bytes appearing at a
            // different place in a rearranged file do not cancel out.
            hash = fnv1a_u64(offset, hash);
            hash = fnv1a(buffer.data(), got, hash);
        }
    }

    print.content_hash = hash;
    std::fclose(f);
    return print;
}

// ------------------------------------------------------------- CacheFile

namespace {

const std::string& empty_string() {
    static const std::string kEmpty;
    return kEmpty;
}

}  // namespace

const std::string& CacheFile::column_name(std::size_t i) const noexcept {
    return i < columns_.size() ? columns_[i].name : empty_string();
}

const std::string& CacheFile::column_unit(std::size_t i) const noexcept {
    return i < columns_.size() ? columns_[i].unit : empty_string();
}

const std::string& CacheFile::simulation_id(std::size_t i) const noexcept {
    return i < sims_.size() ? sims_[i].id : empty_string();
}

std::uint64_t CacheFile::data_begin(std::size_t sim) const noexcept {
    return sim < sims_.size() ? sims_[sim].data_begin : 0;
}

std::uint64_t CacheFile::data_end(std::size_t sim) const noexcept {
    return sim < sims_.size() ? sims_[sim].data_end : 0;
}

const CacheBlock* CacheFile::block(std::size_t column, std::size_t sim) const noexcept {
    if (!blocks_ || column >= columns_.size() || sim >= sims_.size()) return nullptr;
    // Parameter-major: every run of one parameter is contiguous here, and so
    // are their samples on disk.
    return blocks_ + column * sims_.size() + sim;
}

const float* CacheFile::values(std::size_t column, std::size_t sim,
                               std::size_t* count) const noexcept {
    if (count) *count = 0;
    const CacheBlock* entry = block(column, sim);
    if (!entry || entry->count == 0) return nullptr;
    if (count) *count = static_cast<std::size_t>(entry->count);
    // open() has already proved this range lies inside the mapping and that the
    // offset is a multiple of four, so the cast is aligned and in bounds.
    return reinterpret_cast<const float*>(map_.data() + entry->offset);
}

void CacheFile::close() {
    map_.close();
    path_.clear();
    columns_.clear();
    sims_.clear();
    blocks_ = nullptr;
    values_per_line_ = 0;
}

bool CacheFile::open(const std::string& path, const SourceFingerprint& expected) {
    close();

    if (!map_.open(path)) return false;

    const std::size_t file_size = map_.size();
    if (file_size < kHeaderBytes) {
        close();
        return false;
    }

    const unsigned char* const base =
        reinterpret_cast<const unsigned char*>(map_.data());

    if (std::memcmp(base, kCacheMagic, sizeof(kCacheMagic)) != 0) {
        close();
        return false;
    }
    // The byte-order mark is stored in *native* order and compared in native
    // order, which is the whole point of it: read with the same explicit
    // little-endian codec as everything else it would match on either
    // architecture and prove nothing. It guards the one part of the file that
    // is genuinely native -- the block index, which is written straight out of
    // memory -- so a cache produced on a big-endian machine is refused here
    // rather than read as scrambled offsets.
    std::uint32_t byte_order = 0;
    std::memcpy(&byte_order, base + 12, sizeof(byte_order));
    if (get_u32(base + 8) != kCacheFormatVersion || byte_order != kCacheByteOrderMark) {
        close();
        return false;
    }

    // The header carries a hash of itself, so a torn or partially overwritten
    // header is caught before any of its offsets are believed.
    if (get_u64(base + 112) != fnv1a(base, 112, kFnvOffset)) {
        close();
        return false;
    }

    SourceFingerprint stored;
    stored.size = get_u64(base + 16);
    stored.mod_time = get_u64(base + 24);
    stored.content_hash = get_u64(base + 32);
    if (!expected.valid() || stored != expected) {
        close();
        return false;
    }

    const std::uint64_t column_count = get_u32(base + 40);
    const std::uint64_t sim_count = get_u32(base + 44);
    const std::uint64_t columns_offset = get_u64(base + 48);
    const std::uint64_t sims_offset = get_u64(base + 56);
    const std::uint64_t blocks_offset = get_u64(base + 64);
    const std::uint64_t strings_offset = get_u64(base + 72);
    const std::uint64_t strings_bytes = get_u64(base + 80);
    const std::uint64_t data_offset = get_u64(base + 88);
    const std::uint64_t total_bytes = get_u64(base + 96);
    const std::uint64_t values_per_line = get_u64(base + 104);

    // The recorded length must be the actual length. This is what catches a
    // cache truncated by a full disk, and it is checked before any offset is
    // used so the bounds tests below are against a length we trust.
    if (total_bytes != file_size) {
        close();
        return false;
    }
    if (column_count == 0 || sim_count == 0) {
        close();
        return false;
    }
    if (column_count * sim_count > kMaxBlocks) {
        close();
        return false;
    }

    // Every section wholly inside the file. Written as "how much room is left
    // from here" rather than "offset + size <= total" so an overflowing sum
    // cannot wrap round into a range that looks valid.
    const auto fits = [file_size](std::uint64_t offset, std::uint64_t bytes) {
        return offset <= file_size && bytes <= file_size - offset;
    };

    if (!fits(columns_offset, column_count * kColumnRecordBytes) ||
        !fits(sims_offset, sim_count * kSimRecordBytes) ||
        !fits(blocks_offset, column_count * sim_count * sizeof(CacheBlock)) ||
        !fits(strings_offset, strings_bytes) || !fits(data_offset, 0)) {
        close();
        return false;
    }
    // Reading a CacheBlock straight out of the mapping needs it aligned; the
    // writer aligns it, so a file that is not is not one of ours.
    if (blocks_offset % 8 != 0) {
        close();
        return false;
    }

    const char* const strings = map_.data() + strings_offset;
    const auto string_at = [&](std::uint64_t offset, std::uint64_t length,
                               std::string& out) {
        if (offset > strings_bytes || length > strings_bytes - offset) return false;
        out.assign(strings + offset, static_cast<std::size_t>(length));
        return true;
    };

    columns_.resize(static_cast<std::size_t>(column_count));
    for (std::uint64_t c = 0; c < column_count; ++c) {
        const unsigned char* rec = base + columns_offset + c * kColumnRecordBytes;
        if (!string_at(get_u32(rec), get_u32(rec + 4),
                       columns_[static_cast<std::size_t>(c)].name) ||
            !string_at(get_u32(rec + 8), get_u32(rec + 12),
                       columns_[static_cast<std::size_t>(c)].unit)) {
            close();
            return false;
        }
    }

    sims_.resize(static_cast<std::size_t>(sim_count));
    for (std::uint64_t s = 0; s < sim_count; ++s) {
        const unsigned char* rec = base + sims_offset + s * kSimRecordBytes;
        SimMeta& meta = sims_[static_cast<std::size_t>(s)];
        if (!string_at(get_u32(rec), get_u32(rec + 4), meta.id)) {
            close();
            return false;
        }
        meta.data_begin = get_u64(rec + 8);
        meta.data_end = get_u64(rec + 16);
    }

    blocks_ = reinterpret_cast<const CacheBlock*>(map_.data() + blocks_offset);

    // Every block's samples inside the file, and aligned. The loop is over the
    // whole index -- tens of thousands of entries on a large file, microseconds
    // -- and it is what lets values() hand out a pointer with no checking at
    // all on the path that runs per curve, per redraw.
    const std::uint64_t total_blocks = column_count * sim_count;
    for (std::uint64_t i = 0; i < total_blocks; ++i) {
        const CacheBlock& entry = blocks_[i];
        if (entry.count == 0) {
            continue;  // a channel absent from this run; offset is not used
        }
        if (entry.count > file_size / sizeof(float)) {
            close();
            return false;
        }
        const std::uint64_t bytes = entry.count * sizeof(float);
        if (!fits(entry.offset, bytes) || entry.offset < data_offset ||
            entry.offset % sizeof(float) != 0) {
            close();
            return false;
        }
    }

    values_per_line_ = static_cast<std::size_t>(values_per_line);
    path_ = path;
    return true;
}

// ----------------------------------------------------------- CacheWriter

CacheWriter::~CacheWriter() { abort(); }

void CacheWriter::abort() {
    if (file_) {
        std::fclose(static_cast<std::FILE*>(file_));
        file_ = nullptr;
    }
    if (!temp_path_.empty()) {
        remove_file(temp_path_);
        temp_path_.clear();
    }
    final_path_.clear();
    blocks_.clear();
    column_written_.clear();
}

bool CacheWriter::write_at(std::uint64_t offset, const void* data, std::size_t bytes) {
    if (!file_) return false;
    if (bytes == 0) return true;
    std::FILE* f = static_cast<std::FILE*>(file_);
    if (!seek_to(f, offset)) return false;
    return std::fwrite(data, 1, bytes, f) == bytes;
}

bool CacheWriter::begin(const std::string& path, const SourceFingerprint& source,
                        const std::vector<std::string>& names,
                        const std::vector<std::string>& units,
                        const std::vector<std::string>& sim_ids,
                        const std::vector<std::uint64_t>& data_begin,
                        const std::vector<std::uint64_t>& data_end,
                        std::size_t values_per_line) {
    abort();

    if (path.empty() || names.empty() || sim_ids.empty() || !source.valid())
        return false;
    if (units.size() != names.size()) return false;
    if (data_begin.size() != sim_ids.size() || data_end.size() != sim_ids.size())
        return false;
    if (static_cast<std::uint64_t>(names.size()) *
            static_cast<std::uint64_t>(sim_ids.size()) >
        kMaxBlocks)
        return false;
    // The on-disk records hold these as 32-bit counts.
    if (names.size() > 0xFFFFFFFFull || sim_ids.size() > 0xFFFFFFFFull) return false;

    const std::string parent = parent_of(path);
    if (!parent.empty() && !make_directories(parent)) return false;

    // A distinct temporary per process, so two viewers caching the same file at
    // once cannot write over each other's partial output and then both rename
    // a mixture into place.
#if defined(_WIN32)
    const unsigned long pid = static_cast<unsigned long>(::GetCurrentProcessId());
#else
    const unsigned long pid = static_cast<unsigned long>(::getpid());
#endif
    char suffix[48];
    std::snprintf(suffix, sizeof(suffix), ".%lu.tmp", pid);
    temp_path_ = path + suffix;

    std::FILE* f = open_file(temp_path_, AOE_MODE_WRITE);
    if (!f) {
        temp_path_.clear();
        return false;
    }
    file_ = f;
    final_path_ = path;

    source_ = source;
    names_ = names;
    units_ = units;
    sim_ids_ = sim_ids;
    data_begin_ = data_begin;
    data_end_ = data_end;
    values_per_line_ = values_per_line;

    blocks_.assign(names.size() * sim_ids.size(), CacheBlock());
    column_written_.assign(names.size(), false);

    // Where the samples start. The index sections come first and their sizes
    // are known now, so the samples can be written straight away and the header
    // filled in afterwards, in finish().
    std::uint64_t offset = kHeaderBytes;
    offset += static_cast<std::uint64_t>(names.size()) * kColumnRecordBytes;
    offset += static_cast<std::uint64_t>(sim_ids.size()) * kSimRecordBytes;
    offset = (offset + 7) & ~static_cast<std::uint64_t>(7);  // CacheBlock wants 8
    const std::uint64_t blocks_bytes =
        static_cast<std::uint64_t>(blocks_.size()) * sizeof(CacheBlock);
    offset += blocks_bytes;

    std::uint64_t strings_bytes = 0;
    for (const auto& s : names) strings_bytes += s.size();
    for (const auto& s : units) strings_bytes += s.size();
    for (const auto& s : sim_ids) strings_bytes += s.size();
    offset += strings_bytes;

    data_offset_ = (offset + (kDataAlignment - 1)) & ~static_cast<std::uint64_t>(
                                                         kDataAlignment - 1);
    cursor_ = data_offset_;
    return true;
}

bool CacheWriter::write_column(std::size_t column, const std::vector<CacheSource>& runs) {
    if (!file_ || column >= names_.size()) return false;
    if (runs.size() != sim_ids_.size()) return false;
    if (column_written_[column]) return false;

    for (std::size_t s = 0; s < runs.size(); ++s) {
        CacheBlock& entry = blocks_[column * sim_ids_.size() + s];
        const CacheSource& src = runs[s];

        if (!src.values || src.count == 0) {
            // A channel this run does not have. Recorded as an empty block,
            // which reads back as "loaded, no samples" -- the same thing the
            // parser produces, and pointedly not a fabricated zero.
            entry = CacheBlock();
            continue;
        }

        entry.offset = cursor_;
        entry.count = src.count;

        // Minimum, maximum and the shape flags, computed here so that loading
        // the column later needs no pass over its samples at all. One test
        // covers NaN and both infinities: NaN fails every comparison and the
        // infinities fail the range test.
        float lo = 0.0f, hi = 0.0f, previous = 0.0f;
        bool seen = false, gaps = false, monotonic = true;
        for (std::size_t i = 0; i < src.count; ++i) {
            const float v = src.values[i];
            if (!(v > -std::numeric_limits<float>::infinity() &&
                  v < std::numeric_limits<float>::infinity())) {
                gaps = true;
                continue;
            }
            if (!seen) {
                lo = hi = previous = v;
                seen = true;
                continue;
            }
            if (v < lo) lo = v;
            if (v > hi) hi = v;
            if (v < previous) monotonic = false;
            previous = v;
        }

        entry.minimum = seen ? lo : 0.0f;
        entry.maximum = seen ? hi : 0.0f;
        entry.flags = 0;
        if (seen) entry.flags |= kCacheHasFinite;
        if (gaps) entry.flags |= kCacheHasGaps;
        if (seen && monotonic) entry.flags |= kCacheMonotonic;

        const std::size_t bytes = src.count * sizeof(float);
        if (!write_at(cursor_, src.values, bytes)) return false;
        cursor_ += bytes;
    }

    column_written_[column] = true;
    return true;
}

bool CacheWriter::finish() {
    if (!file_) return false;

    // A cache with a hole in it would read back as a channel that silently has
    // no data. Refuse instead, and leave whatever was there before in place.
    for (std::size_t c = 0; c < column_written_.size(); ++c) {
        if (!column_written_[c]) {
            abort();
            return false;
        }
    }

    const std::uint64_t column_count = names_.size();
    const std::uint64_t sim_count = sim_ids_.size();

    std::uint64_t offset = kHeaderBytes;
    const std::uint64_t columns_offset = offset;
    offset += column_count * kColumnRecordBytes;
    const std::uint64_t sims_offset = offset;
    offset += sim_count * kSimRecordBytes;
    offset = (offset + 7) & ~static_cast<std::uint64_t>(7);
    const std::uint64_t blocks_offset = offset;
    offset += static_cast<std::uint64_t>(blocks_.size()) * sizeof(CacheBlock);
    const std::uint64_t strings_offset = offset;

    // The string blob, and the (offset, length) pairs into it. Built together
    // so the two can never disagree.
    std::string strings;
    std::vector<unsigned char> column_records(
        static_cast<std::size_t>(column_count) * kColumnRecordBytes, 0);
    std::vector<unsigned char> sim_records(
        static_cast<std::size_t>(sim_count) * kSimRecordBytes, 0);

    const auto intern = [&strings](const std::string& s, unsigned char* out) {
        put_u32(out, static_cast<std::uint32_t>(strings.size()));
        put_u32(out + 4, static_cast<std::uint32_t>(s.size()));
        strings += s;
    };

    for (std::size_t c = 0; c < names_.size(); ++c) {
        unsigned char* rec = column_records.data() + c * kColumnRecordBytes;
        intern(names_[c], rec);
        intern(units_[c], rec + 8);
    }
    for (std::size_t s = 0; s < sim_ids_.size(); ++s) {
        unsigned char* rec = sim_records.data() + s * kSimRecordBytes;
        intern(sim_ids_[s], rec);
        put_u64(rec + 8, data_begin_[s]);
        put_u64(rec + 16, data_end_[s]);
    }

    const std::uint64_t strings_bytes = strings.size();
    // begin() reserved room from the same string sizes, so this is the space
    // that was left; if the two ever disagreed the samples would sit on top of
    // the blob, so check rather than assume.
    if (strings_offset + strings_bytes > data_offset_) {
        abort();
        return false;
    }

    unsigned char header[kHeaderBytes];
    std::memset(header, 0, sizeof(header));
    std::memcpy(header, kCacheMagic, sizeof(kCacheMagic));
    put_u32(header + 8, kCacheFormatVersion);
    // Native order deliberately -- see the note where this is read back.
    std::memcpy(header + 12, &kCacheByteOrderMark, sizeof(kCacheByteOrderMark));
    put_u64(header + 16, source_.size);
    put_u64(header + 24, source_.mod_time);
    put_u64(header + 32, source_.content_hash);
    put_u32(header + 40, static_cast<std::uint32_t>(column_count));
    put_u32(header + 44, static_cast<std::uint32_t>(sim_count));
    put_u64(header + 48, columns_offset);
    put_u64(header + 56, sims_offset);
    put_u64(header + 64, blocks_offset);
    put_u64(header + 72, strings_offset);
    put_u64(header + 80, strings_bytes);
    put_u64(header + 88, data_offset_);
    put_u64(header + 96, cursor_);  // total length: samples ended here
    put_u64(header + 104, static_cast<std::uint64_t>(values_per_line_));
    put_u64(header + 112, fnv1a(header, 112, kFnvOffset));

    // The gap between the end of the strings and the aligned start of the
    // samples is never read, but it is written so the file holds no
    // uninitialised bytes -- a sparse hole here would make two caches of
    // identical data compare unequal, which makes them awkward to diff.
    const std::uint64_t padding = data_offset_ - (strings_offset + strings_bytes);
    std::vector<unsigned char> zeros(static_cast<std::size_t>(padding), 0);

    if (!write_at(0, header, sizeof(header)) ||
        !write_at(columns_offset, column_records.data(), column_records.size()) ||
        !write_at(sims_offset, sim_records.data(), sim_records.size()) ||
        !write_at(blocks_offset, blocks_.data(), blocks_.size() * sizeof(CacheBlock)) ||
        !write_at(strings_offset, strings.data(), strings.size()) ||
        (padding > 0 && !write_at(strings_offset + strings_bytes, zeros.data(),
                                  zeros.size()))) {
        abort();
        return false;
    }

    std::FILE* f = static_cast<std::FILE*>(file_);
    const bool flushed = std::fflush(f) == 0;
    const bool closed = std::fclose(f) == 0;
    file_ = nullptr;
    if (!flushed || !closed) {
        abort();
        return false;
    }

    const std::string temp = temp_path_;
    const std::string dest = final_path_;
    temp_path_.clear();  // the rename owns it now

    if (!replace_file(temp, dest)) {
        remove_file(temp);
        abort();
        return false;
    }

    final_path_.clear();
    blocks_.clear();
    column_written_.clear();
    return true;
}

// ------------------------------------------------------------------ paths

bool caching_enabled() noexcept {
    // Read once: this is consulted on every file open, and an environment
    // lookup on Windows is a syscall.
    static const bool enabled = [] {
        std::string value = environment("AOE_CACHE");
        if (value.empty()) return true;
        for (char& c : value)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return !(value == "0" || value == "false" || value == "no" || value == "off");
    }();
    return enabled;
}

std::string user_cache_directory() {
    const std::string override_dir = environment("AOE_CACHE_DIR");
    if (!override_dir.empty()) return override_dir;

#if defined(_WIN32)
    std::string base = environment("LOCALAPPDATA");
    if (base.empty()) base = environment("APPDATA");
    if (base.empty()) base = environment("TEMP");
    if (base.empty()) return std::string();
    return base + "\\aoeViewer\\cache";
#elif defined(__APPLE__)
    const std::string home = environment("HOME");
    if (home.empty()) return std::string();
    return home + "/Library/Caches/aoeviewer";
#else
    const std::string xdg = environment("XDG_CACHE_HOME");
    if (!xdg.empty()) return xdg + "/aoeviewer";
    const std::string home = environment("HOME");
    if (home.empty()) return std::string();
    return home + "/.cache/aoeviewer";
#endif
}

std::vector<std::string> cache_candidates(const std::string& source_path) {
    std::vector<std::string> out;
    if (source_path.empty() || !caching_enabled()) return out;

    // Beside the source. The full source name is kept, extension and all, so
    // "run.aoe" caches to "run.aoe.aoec" and can never be mistaken for, or
    // collide with, a differently-suffixed file of the same stem.
    out.push_back(source_path + ".aoec");

    const std::string dir = user_cache_directory();
    if (!dir.empty()) {
        // Hash the whole source path into the name: two files called
        // "results.aoe" in different directories must not share a cache, and
        // the hash keeps the name short and free of separators.
        const std::uint64_t key = fnv1a(source_path.data(), source_path.size(), kFnvOffset);
        char stamp[24];
        std::snprintf(stamp, sizeof(stamp), "%016llx",
                      static_cast<unsigned long long>(key));

        std::string leaf = basename_of(source_path);
        // Keep a readable prefix so the directory can be browsed, but bound it
        // and drop anything a filesystem might object to.
        if (leaf.size() > 48) leaf.resize(48);
        for (char& c : leaf) {
            const unsigned char u = static_cast<unsigned char>(c);
            if (!(std::isalnum(u) || c == '.' || c == '-' || c == '_')) c = '_';
        }

#if defined(_WIN32)
        out.push_back(dir + "\\" + leaf + "." + stamp + ".aoec");
#else
        out.push_back(dir + "/" + leaf + "." + stamp + ".aoec");
#endif
    }
    return out;
}

}  // namespace aoe
