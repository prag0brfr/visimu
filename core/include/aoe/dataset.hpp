// Author: Paulo R. A. Gilz
//
// AOE result-file parsing and lazy column storage.
//
// File shape:
//
//     *PARAMETRES
//      'TIME'    'ALPHA'   'MACH'      <- header names, single-quoted
//     *TABLEAU : SIM_001                <- one simulation block per marker
//      0.000  1.2340  0.7500
//      0.020  1.2350  0.7501
//     *TABLEAU : SIM_002
//      ...
//
// A logical record holds one value per header. Fortran writers wrap a record
// across several physical lines when it exceeds the format width, so the column
// counter must carry across newlines -- it resets only after num_columns values.
//
// Design notes:
//
//  * The index scan records each simulation's *end* offset as it goes, so a
//    parse task needs no search through the simulation list for the next
//    block's offset and no per-line test for the '*TABLEAU' terminator.
//  * Parsing parallelises over byte-range chunks, not just simulations. One
//    task per simulation would leave a file holding a single large table --
//    extremely common -- parsing entirely on one core.
//  * Columns are loaded once and cached. A failed load marks the column loaded
//    but empty rather than pushing a fake 0.0 sample, so a missing channel
//    reads as absent instead of plotting as a spurious zero line.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "aoe/cache.hpp"
#include "aoe/mapped_file.hpp"
#include "aoe/platform_thread.hpp"

namespace aoe {

// One column's samples, plus the summary computed in the same pass that parsed
// them. The summary is not a nicety: autoscaling a tile asks for the extent of
// every visible curve on every replot, and answering that by rescanning a
// million samples per curve was the single largest avoidable cost in the draw
// path. Computed once, at load, in parallel, it is then free forever.
struct SampleData {
    std::vector<float> values;

    float minimum = 0.0f;
    float maximum = 0.0f;
    bool has_finite = false;  // at least one sample that is neither NaN nor inf
    bool has_gaps = false;    // at least one NaN or inf
    bool monotonic = false;   // non-decreasing across the finite samples

    std::size_t size() const noexcept { return values.size(); }
    const float* data() const noexcept { return values.empty() ? nullptr : values.data(); }
};

// Samples are handed out as shared blocks rather than raw pointers into a
// vector the cache owns.
//
// The front end holds zero-copy NumPy views over these buffers for as long as a
// curve is on screen. With the samples living directly in the column, evicting a
// least-recently-used column freed the very memory the visible tiles were
// pointing at, and the next mouse move read it back. Sharing the block means
// eviction drops the *cache's* reference and nothing else: a block still held by
// a view stays alive until the view goes, and the memory cap becomes advisory
// for exactly the data that is genuinely in use.
using SampleBlock = std::shared_ptr<SampleData>;

struct Column {
    SampleBlock block;
    bool loaded = false;
    // Monotonic stamp of the last read, for least-recently-used eviction.
    mutable std::uint64_t last_used = 0;

    bool empty() const noexcept { return !block || block->values.empty(); }
    std::size_t size() const noexcept { return block ? block->values.size() : 0; }
    const float* data() const noexcept { return block ? block->data() : nullptr; }
    std::size_t bytes() const noexcept {
        return block ? block->values.capacity() * sizeof(float) : 0;
    }
};

// Fills in the summary fields of a block from its samples. Cheap: one pass,
// over data that is still in cache when it is called right after parsing.
void summarise(SampleData& block) noexcept;

class Simulation {
public:
    Simulation() = default;
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    std::string id;            // as written after "*TABLEAU :"
    std::string display_name;  // what the UI shows; defaults to id
    // "<file>::<id>::<ordinal>". Deliberately NOT derived from display_name:
    // styles are keyed on it, so renaming a run must not recolour it.
    std::string unique_key;

    std::size_t data_begin = 0;  // first byte of the block's data
    std::size_t data_end = 0;    // one past the last byte of the block's data

    std::vector<Column> columns;                          // by column index
    std::unordered_map<std::string, SampleBlock> computed;  // formula results

    mutable Mutex lock;
};

class Dataset {
public:
    Dataset() = default;
    ~Dataset() = default;
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;

    // Scans the file for headers and simulation blocks. No sample data is read;
    // that happens lazily in load_columns(). Returns false if the file cannot be
    // mapped or contains no recognisable header block.
    bool open(const std::string& path);

    // Drops every cached sample but keeps the index, so the next plot re-reads
    // from disk. Used when the file changes underneath us.
    void invalidate_data();

    // Re-runs open() on the same path, preserving nothing. Returns false if the
    // file has become unreadable.
    //
    // A dataset built through the array route below has no parser here to
    // re-run, so this refuses and the caller -- which is the front end that
    // supplied the arrays -- rebuilds it the way it built it the first time.
    bool reload();

    // ------------------------------------------- construction from arrays
    //
    // Some formats are containers rather than layouts: HDF5 carries no
    // convention for where a run ends and a channel begins, and every exporter
    // invents its own. Writing a native reader for one of them would mean a new
    // compiled dependency on a machine where adding one is expensive, to parse
    // a convention that the next file may not follow.
    //
    // So they are read where a reader is a page of code instead: in Python,
    // which hands the samples over through these four calls. Everything above
    // this class -- decimation, ISO scanning, formulas, the plot path -- then
    // works on the result exactly as it does on a parsed file, because what it
    // sees is the same headers, runs and columns.
    //
    // The trade-off is deliberate and worth knowing: arrays arrive complete, so
    // a pushed file has no lazy per-column loading and its samples are resident
    // from the moment it opens.
    //
    // Order: begin_arrays, then add_arrays_sim per run, then set_arrays_column
    // per (run, column), then end_arrays.
    bool begin_arrays(const std::string& path, const std::vector<std::string>& headers,
                      const std::vector<std::string>& units);
    int add_arrays_sim(const std::string& id);
    bool set_arrays_column(std::size_t sim_index, int column, const float* data,
                           std::size_t count);
    void end_arrays();

    // True when this dataset was pushed rather than parsed.
    bool arrays_backed() const noexcept { return arrays_backed_; }

    bool is_open() const noexcept { return !headers_.empty(); }
    const std::string& path() const noexcept { return path_; }
    const std::string& name() const noexcept { return name_; }

    // Renaming changes the label and nothing else. key_name_ is fixed by the
    // first name this dataset is ever given -- the one Session derives from
    // the path, before open() runs -- and every run's unique_key is built from
    // that rather than from name_.
    //
    // They used to be the same string, which was invisible until a file could
    // be reloaded on purpose: reload() re-opens and rebuilds every key, so a
    // file the user had renamed came back with different keys for the same
    // runs, and every colour, dash and marker the user had set for them was
    // suddenly attached to nothing.
    void set_name(const std::string& n) {
        name_ = n;
        if (key_name_.empty()) key_name_ = n;
    }
    const std::string& key_name() const noexcept {
        return key_name_.empty() ? name_ : key_name_;
    }

    std::uint64_t mod_time() const noexcept { return mod_time_; }
    bool changed_on_disk() const noexcept {
        return MappedFile::mod_time(path_) != mod_time_;
    }

    const std::vector<std::string>& headers() const noexcept { return headers_; }

    // Units as declared in *PARAMETRES, parallel to headers(). Empty where the
    // file gives none, which is the whole of the packed dialect.
    const std::vector<std::string>& units() const noexcept { return units_; }

    const std::string& unit(int column) const noexcept {
        static const std::string kNone;
        if (column < 0 || static_cast<std::size_t>(column) >= units_.size()) return kNone;
        return units_[static_cast<std::size_t>(column)];
    }
    std::size_t column_count() const noexcept { return headers_.size(); }
    std::size_t simulation_count() const noexcept { return sims_.size(); }

    Simulation& simulation(std::size_t i) { return *sims_[i]; }
    const Simulation& simulation(std::size_t i) const { return *sims_[i]; }

    // -1 when the name is not a physical column of this file.
    int column_index(const std::string& name) const;

    // Renames a run for display only. The identity key is untouched, so
    // colours, dash styles and saved workspaces survive the rename.
    void set_simulation_name(std::size_t index, const std::string& name);

    // Loads the given columns for the given simulations, skipping any that are
    // already cached. Thread-safe; returns once all requested data is resident.
    // Passing an empty sim_indices means "all simulations".
    void load_columns(const std::vector<int>& columns,
                      const std::vector<int>& sim_indices);

    // Convenience: pointer/size of one physical column, loading it on demand.
    // Returns nullptr when the column does not exist or holds no samples.
    //
    // `out_block`, when given, receives a reference to the sample block that
    // keeps the returned pointer valid regardless of later eviction. Callers
    // that hold on to the pointer -- which is every plotted curve -- must take
    // it; callers that read and discard within the call need not.
    const float* column_data(std::size_t sim_index, int column, std::size_t* out_count,
                             SampleBlock* out_block = nullptr);

    // The whole block for a column, loading it on demand. Null when absent.
    SampleBlock column_block(std::size_t sim_index, int column);

    // Number of samples in the longest loaded column of a simulation. Used to
    // size formula evaluation when a formula references only constants.
    std::size_t loaded_row_count(std::size_t sim_index) const;

    // First sample of a column without loading the column.
    //
    // An ISO parameter is constant within a run, so its value is the first
    // sample -- but reading it through the normal path parses the whole column,
    // which on a 500 MB file means scanning half a gigabyte to obtain one float
    // per run. This parses only the first record of the block instead, turning
    // an ISO scan from seconds into microseconds. Returns NaN when unavailable.
    float first_value(std::size_t sim_index, int column) const;

    // The same for every simulation at once, mapping the file a single time.
    // Called per run, first_value() maps a half-gigabyte file once per run; an
    // ISO scan over 75 runs therefore paid 75 map/unmap round trips to read 75
    // floats.
    std::vector<float> first_values(int column) const;

    // Bytes of sample data currently held for this file.
    std::size_t resident_bytes() const;

    // Frees cached columns until at most `keep_bytes` remain, dropping the
    // least recently used first. Columns are re-read on demand afterwards.
    //
    // Blocks the front end still holds views of survive the drop -- the cache
    // releases its reference, the view keeps the memory -- so this can never
    // pull the ground out from under a visible curve. Their bytes stop counting
    // towards the budget once the last view goes.
    std::size_t evict_until(std::size_t keep_bytes);

    // Drops cached formula results, keeping blocks that are still referenced
    // elsewhere alive until their last holder lets go.
    void clear_computed();

    // Bumps a column's last-use stamp. Reading through column_data() does this
    // automatically; call it directly only when bypassing that path.
    void touch(std::size_t sim_index, int column) const;

    // ------------------------------------------------------------ cache
    //
    // A .aoec sidecar holds the same numbers laid out the way this program
    // reads them: parameter-major, float32, with the run offsets and each
    // block's extent in the header. When one is present and matches the source,
    // open() takes its index from there instead of scanning the file, and
    // loading a column becomes a copy out of a mapping rather than a parse.
    //
    // Nothing above this class has to care: every path falls back to parsing
    // the text when there is no cache, when it does not match, or when it turns
    // out to be damaged. The cache is a shortcut, never the only route.

    // True when a valid sidecar is attached and serving reads.
    bool cache_active() const noexcept { return cache_ && cache_->is_open(); }

    // Path of the attached sidecar, or empty.
    std::string cache_file() const {
        return cache_ ? cache_->path() : std::string();
    }

    // Parses the whole file once and writes a sidecar, then attaches it.
    //
    // Deliberately not called on open(): it costs a full parse, so it is the
    // caller's decision when to spend that -- typically once, in the
    // background, after a file has been opened the slow way. Returns false if
    // the file cannot be parsed or no candidate location is writable.
    //
    // Memory is bounded to one parameter across every run: each column is
    // loaded, written and released before the next is touched.
    bool write_cache(std::string* out_path = nullptr);

    // Detaches and deletes the sidecar. Returns true if none remains.
    bool discard_cache();

    // Looks for a sidecar matching the file as it is now and attaches it.
    // Called by open(); public so a caller that has just written one, or that
    // suspects the file changed, can retry without reopening.
    bool attach_cache();

private:
    void build_index(const MappedFile& file);

    // Shared tail of the two open() routes: fills in the lookup table, squares
    // up the units vector, and gives every run its column slots.
    void finish_open();

    // Fills `pending` straight from the sidecar. Returns false when there is no
    // usable cache, in which case the caller parses instead.
    bool load_from_cache(const std::vector<int>& columns,
                         const std::vector<int>& sim_indices);

    // Parses the byte range [begin, end) of the mapping, appending the requested
    // columns into `out` (one vector per entry of `cols`, in the same order).
    void parse_range(const char* base, std::size_t begin, std::size_t end,
                     const std::vector<int>& cols,
                     std::vector<std::vector<float>>& out) const;

    std::string path_;
    std::string name_;
    // What unique_key is built from. Set once, by the first set_name().
    std::string key_name_;
    std::uint64_t mod_time_ = 0;
    // File length when the index was built. Block offsets are only meaningful
    // against a file of this length; a file that shrank underneath us -- the
    // solver this tool watches truncates and rewrites its output -- must have
    // its ranges clamped before anything parses them.
    std::size_t indexed_size_ = 0;

    std::vector<std::string> headers_;
    std::vector<std::string> units_;
    std::unordered_map<std::string, int> header_index_;
    std::vector<std::unique_ptr<Simulation>> sims_;

    // The sidecar, when one matched. Read-only and internally immutable once
    // open, so the parallel loads below share it without locking.
    std::unique_ptr<CacheFile> cache_;
    // What the source looked like when the sidecar was accepted -- and what a
    // sidecar written from here records, so the next open can check it.
    SourceFingerprint fingerprint_;

    // Values found on the first data line. When this equals column_count() the
    // file is one-record-per-line and byte ranges can be split at any newline.
    std::size_t values_per_line_ = 0;

    // Set when the samples were handed to us rather than parsed from path_.
    // Guards the routes that assume a parser: reloading and writing a sidecar.
    bool arrays_backed_ = false;
};

}  // namespace aoe
