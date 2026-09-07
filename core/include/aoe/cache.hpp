// Author: Paulo R. A. Gilz
//
// The .aoec sidecar: a binary, parameter-major rewrite of an AOE file that the
// viewer can map and read without parsing anything.
//
// Why this exists
// ---------------
// An AOE file stores one *record* per line: every parameter of one time step,
// side by side. The viewer's access pattern is the exact transpose -- "one
// parameter, every run" -- so plotting a single channel forces a scan of the
// whole file, and the scan has to run a float parser over every value it throws
// away. On a 1200-run file that is most of a second before the first pixel.
//
// The sidecar stores the same numbers the other way round: all runs of TIME,
// then all runs of CH01, and so on. Reading a channel becomes one sequential
// read of a contiguous block, and the values are already float32, so there is
// nothing to parse. The header carries the run offsets (so opening does not
// scan) and each block's minimum and maximum (so autoscaling does not rescan).
//
// The layout on disk
// ------------------
//
//     +-----------------------------------------------------------+
//     | Header            128 bytes, fixed                        |
//     +-----------------------------------------------------------+
//     | Column records    column_count x 16 bytes                  |
//     |   name and unit, as offsets into the string blob           |
//     +-----------------------------------------------------------+
//     | Run records       simulation_count x 24 bytes              |
//     |   id, and the byte range it occupied in the source file    |
//     +-----------------------------------------------------------+
//     | Block records     column_count * simulation_count x 32     |
//     |   offset, count, minimum, maximum, flags                   |
//     |   indexed [parameter * simulation_count + simulation]      |
//     +-----------------------------------------------------------+
//     | String blob       names, units and run ids, back to back   |
//     +-----------------------------------------------------------+
//     | Samples           float32, 64-byte aligned                 |
//     |   parameter 0: run 0, run 1, ... run N-1                   |
//     |   parameter 1: run 0, run 1, ... run N-1                   |
//     |   ...                                                      |
//     +-----------------------------------------------------------+
//
// The block records are ordered parameter-major deliberately: the entries for
// "CH07 across every run" are adjacent in the index *and* their samples are
// adjacent on disk, so both the lookup and the read are sequential.
//
// Staleness
// ---------
// A cache that disagrees with its source is worse than no cache at all -- it
// shows the user numbers that are not in the file any more. Three independent
// properties of the source are recorded and all three must match before a byte
// of cached data is trusted: the length, the modification time, and a hash of
// the contents. See SourceFingerprint for what each one catches and why none of
// them is sufficient alone.
//
// Portability
// -----------
// Written little-endian with a byte-order mark. A big-endian reader recognises
// the mismatch and declines the cache rather than byte-swapping: the fallback
// costs a parse, and a swapping reader is code that would never once run on the
// platforms this ships to and so would never be exercised.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "aoe/mapped_file.hpp"

namespace aoe {

// Bumped whenever the layout changes. A reader that meets a version it does not
// know declines the file, and the sidecar is rebuilt.
constexpr std::uint32_t kCacheFormatVersion = 1;

// Written into the header as-is. Read back as a different value, the file was
// produced on a machine of the opposite byte order.
constexpr std::uint32_t kCacheByteOrderMark = 0x01020304u;

extern const char kCacheMagic[8];  // "AOECACHE", not NUL-terminated

// What a source file has to still look like for its cache to be usable.
//
// Each field catches something the others miss:
//
//   size          Truncation, extension, an append. Free to obtain and the
//                 single most common way a solver's output changes.
//   mod_time      An in-place rewrite that happens to preserve the length.
//                 Free to obtain. Defeated by anything that restores
//                 timestamps -- rsync --times, a restore from backup, some
//                 build systems -- which is why it is not trusted alone.
//   content_hash  An in-place rewrite that preserves both the length and the
//                 timestamp. Costs a bounded amount of reading (see
//                 fingerprint()) and is what makes the pair above safe.
//
// All three must match. The cost of a false negative is one rebuild; the cost
// of a false positive is silently wrong data on screen, so the comparison is
// deliberately strict in the direction that only ever wastes time.
struct SourceFingerprint {
    std::uint64_t size = 0;
    std::uint64_t mod_time = 0;
    std::uint64_t content_hash = 0;

    bool operator==(const SourceFingerprint& other) const noexcept {
        return size == other.size && mod_time == other.mod_time &&
               content_hash == other.content_hash;
    }
    bool operator!=(const SourceFingerprint& other) const noexcept {
        return !(*this == other);
    }
    bool valid() const noexcept { return size != 0; }
};

// Reads `path` and returns its fingerprint. `size` is zero if the file cannot
// be read at all.
//
// The hash covers the whole file when it is small. Above the threshold it
// covers a fixed budget of evenly spaced windows, always including the first
// and last, so the work does not grow with the file: hashing half a gigabyte in
// full would cost more than the parse the cache exists to avoid. The size is
// folded into the seed, so a file that is a prefix of another never collides
// with it however the windows land.
SourceFingerprint fingerprint(const std::string& path) noexcept;

// Flags in a block record.
enum CacheBlockFlags : std::uint32_t {
    kCacheHasFinite = 1u << 0,  // at least one sample that is neither NaN nor inf
    kCacheHasGaps = 1u << 1,    // at least one NaN or inf
    kCacheMonotonic = 1u << 2,  // non-decreasing across the finite samples
};

// One (parameter, run) pair: where its samples are and what they contain.
// Mirrors SampleData's summary so a loaded column needs no summarising pass.
#pragma pack(push, 1)
struct CacheBlock {
    std::uint64_t offset = 0;  // byte offset into the cache file
    std::uint64_t count = 0;   // number of float32 values
    float minimum = 0.0f;
    float maximum = 0.0f;
    std::uint32_t flags = 0;
    std::uint32_t reserved = 0;
};
#pragma pack(pop)
static_assert(sizeof(CacheBlock) == 32, "CacheBlock must be exactly 32 bytes on disk");

// A mapped, validated .aoec file.
//
// Every accessor is const and touches only the mapping, so any number of
// threads may read one of these at once without locking. Validation happens
// entirely in open(): once that returns true, no accessor can read out of
// bounds, which is what lets the hot paths index without checking.
class CacheFile {
public:
    CacheFile() = default;
    CacheFile(const CacheFile&) = delete;
    CacheFile& operator=(const CacheFile&) = delete;

    // Maps and validates `path`. Returns false -- leaving the object closed --
    // if the file is missing, is not a cache, was written by a different format
    // version or byte order, is internally inconsistent, or does not match
    // `expected`.
    //
    // "Internally inconsistent" is checked exhaustively: every section and
    // every one of the column_count * simulation_count blocks must lie wholly
    // inside the file. A cache half-written by a process that died, or
    // truncated by a full disk, is thereby rejected here rather than read past
    // its end later.
    bool open(const std::string& path, const SourceFingerprint& expected);
    void close();

    bool is_open() const noexcept { return map_.is_open(); }
    const std::string& path() const noexcept { return path_; }

    std::size_t column_count() const noexcept { return columns_.size(); }
    std::size_t simulation_count() const noexcept { return sims_.size(); }
    std::size_t values_per_line() const noexcept { return values_per_line_; }

    const std::string& column_name(std::size_t i) const noexcept;
    const std::string& column_unit(std::size_t i) const noexcept;
    const std::string& simulation_id(std::size_t i) const noexcept;

    // The byte range this run occupied in the *source* file, carried through so
    // a fall back to parsing needs no fresh index scan.
    std::uint64_t data_begin(std::size_t sim) const noexcept;
    std::uint64_t data_end(std::size_t sim) const noexcept;

    // The record for one (parameter, run) pair, or nullptr when out of range.
    const CacheBlock* block(std::size_t column, std::size_t sim) const noexcept;

    // Samples for one (parameter, run) pair. Sets *count and returns a pointer
    // into the mapping, valid for the lifetime of this object. Null when the
    // pair is out of range or holds no samples.
    const float* values(std::size_t column, std::size_t sim,
                        std::size_t* count) const noexcept;

private:
    MappedFile map_;
    std::string path_;

    struct ColumnMeta {
        std::string name;
        std::string unit;
    };
    struct SimMeta {
        std::string id;
        std::uint64_t data_begin = 0;
        std::uint64_t data_end = 0;
    };

    std::vector<ColumnMeta> columns_;
    std::vector<SimMeta> sims_;
    const CacheBlock* blocks_ = nullptr;  // into the mapping
    std::size_t values_per_line_ = 0;
};

// What a writer needs to know about one (parameter, run) pair. The caller owns
// the samples; the writer copies them out and never holds the pointer.
struct CacheSource {
    const float* values = nullptr;
    std::size_t count = 0;
};

// Builds a .aoec file.
//
// Used in two passes, because the block records have to precede the samples on
// disk but their offsets are only known once the samples are written:
//
//     CacheWriter writer;
//     writer.begin(out_path, fingerprint, names, units, ids, begins, ends, vpl);
//     for (each column c)
//         writer.write_column(c, sources_for_every_run);   // in column order
//     writer.finish();
//
// Columns must arrive in order, 0 first, and every one must be written exactly
// once -- finish() refuses otherwise rather than leaving a file with holes in
// it. Writing a column at a time is what bounds the writer's memory to one
// parameter across all runs, rather than the whole file.
//
// The output goes to a temporary beside the destination and is renamed over it
// in finish(). A reader therefore only ever sees a complete file: a writer that
// dies leaves the stale cache in place, or none, but never a half-written one
// that would have to be detected and rejected.
class CacheWriter {
public:
    CacheWriter() = default;
    ~CacheWriter();
    CacheWriter(const CacheWriter&) = delete;
    CacheWriter& operator=(const CacheWriter&) = delete;

    bool begin(const std::string& path, const SourceFingerprint& source,
               const std::vector<std::string>& names,
               const std::vector<std::string>& units,
               const std::vector<std::string>& sim_ids,
               const std::vector<std::uint64_t>& data_begin,
               const std::vector<std::uint64_t>& data_end,
               std::size_t values_per_line);

    // Writes every run's samples for one parameter. `runs` must hold exactly
    // simulation_count entries; an entry with a null pointer or zero count is
    // recorded as an empty block, which is how a channel missing from one run
    // is represented.
    bool write_column(std::size_t column, const std::vector<CacheSource>& runs);

    // Writes the header and index, closes, and renames into place.
    bool finish();

    // Closes and removes the temporary. Safe to call at any point, and called
    // by the destructor, so an abandoned writer leaves nothing behind.
    void abort();

private:
    bool write_at(std::uint64_t offset, const void* data, std::size_t bytes);

    void* file_ = nullptr;  // std::FILE*
    std::string final_path_;
    std::string temp_path_;

    SourceFingerprint source_;
    std::vector<std::string> names_;
    std::vector<std::string> units_;
    std::vector<std::string> sim_ids_;
    std::vector<std::uint64_t> data_begin_;
    std::vector<std::uint64_t> data_end_;
    std::vector<CacheBlock> blocks_;
    std::vector<bool> column_written_;

    std::size_t values_per_line_ = 0;
    std::uint64_t data_offset_ = 0;
    std::uint64_t cursor_ = 0;
};

// ------------------------------------------------------------------ paths

// Where the sidecar for `source_path` should live.
//
// Beside the source is the first choice: it keeps the cache with the data, so
// copying a results directory copies its cache too, and deleting the results
// deletes it. But solver output very often sits on a read-only share, so a
// per-user cache directory is tried second, with the source's full path hashed
// into the name to keep two files of the same basename apart.
//
// Returns both candidates, nearest first. Empty when caching is switched off.
std::vector<std::string> cache_candidates(const std::string& source_path);

// The directory the second candidate lives in, honouring AOE_CACHE_DIR and
// otherwise following the platform convention (%LOCALAPPDATA% on Windows,
// $XDG_CACHE_HOME or ~/.cache elsewhere, ~/Library/Caches on macOS). Empty when
// no such location can be determined.
std::string user_cache_directory();

// False when AOE_CACHE is set to 0, false, no or off. Checked once per process.
bool caching_enabled() noexcept;

// Creates `dir` and any missing parent. False if it does not exist afterwards.
bool make_directories(const std::string& dir) noexcept;

// Removes a file. True if it is gone afterwards, including when it was never
// there.
bool remove_file(const std::string& path) noexcept;

}  // namespace aoe
