// Author: Paulo R. A. Gilz
//
// Owns the loaded files and everything derived from them: parameter aliases,
// computed parameters, and the cache of evaluated formula results.
//
// Kept as a plain object the front end drives, rather than as a scattering of
// member maps on the window class. Session state mixed in among UI fields
// cannot be exercised without a live window, which puts aliases, computed
// parameters and the result cache -- the parts most worth testing -- out of
// reach of every test.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "aoe/dataset.hpp"
#include "aoe/formula.hpp"

namespace aoe {

class Session {
public:
    Session() = default;
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    // ------------------------------------------------------------- files

    // Loads a file and returns its index, or -1 on failure. Duplicate display
    // names are suffixed (" (2)", " (3)"...) so simulations stay distinguishable
    // in the tree and in legends.
    int add_file(const std::string& path);
    void remove_file(int index);
    void clear();

    std::size_t file_count() const noexcept { return files_.size(); }
    Dataset* file(int index);
    const Dataset* file(int index) const;

    // Files whose modification time changed since they were read.
    std::vector<int> stale_files() const;

    // Opens an empty dataset for a file this core has no parser for, to be
    // filled through Dataset's array route. `replace` names an existing file to
    // rebuild in place -- which is what a reload of such a file is -- or -1 for
    // a new one. Returns the file index, or -1.
    int begin_arrays_file(const std::string& path, const std::vector<std::string>& headers,
                          const std::vector<std::string>& units, int replace);

    // Closes the array route on `index` and refreshes the parameter union.
    void end_arrays_file(int index);

    // Re-reads a file in place, keeping its index and display name. Cached
    // formula results for that file are dropped.
    bool reload_file(int index);

    // ------------------------------------------------------- parameters

    // Union of the physical headers of every open file, plus computed
    // parameters, plus alias names -- what the UI lists as selectable channels.
    const std::vector<std::string>& parameters() const noexcept { return parameters_; }

    // Aliases rename a parameter for display. Chains are resolved and cycles
    // are broken, so A->B->C reports C and A->B->A reports A.
    void set_alias(const std::string& alias, const std::string& target);
    void remove_alias(const std::string& alias);
    std::string resolve(const std::string& name) const;

    // The unit declared for a parameter, searched across every open file after
    // resolving aliases. Empty when unknown, when the file's dialect declares
    // none, or when the parameter is computed -- deriving a unit through a
    // formula would mean implementing dimensional analysis, and a wrong unit
    // on an axis is worse than none.
    const std::string& parameter_unit(const std::string& name) const;
    const std::unordered_map<std::string, std::string>& aliases() const noexcept {
        return aliases_;
    }

    // Registers `name` as a computed channel. Returns false with `error` set if
    // the expression does not parse, references an unknown parameter, or would
    // create a dependency cycle.
    bool set_computed(const std::string& name, const std::string& expression,
                      std::string* error);

    // Removes a computed parameter and, transitively, every computed parameter
    // whose expression referenced it.
    std::vector<std::string> remove_computed(const std::string& name);

    const std::unordered_map<std::string, std::string>& computed() const noexcept {
        return computed_sources_;
    }
    bool is_computed(const std::string& name) const;

    // True when `name` is a physical column of some file or a computed channel.
    bool is_known_parameter(const std::string& name) const;

    // ----------------------------------------------------------- access

    // Samples of `name` for one simulation, evaluating and caching computed
    // channels on demand. Returns nullptr when unavailable.
    //
    // `out_block` receives the block that keeps the returned pointer alive.
    // Anything that holds the pointer past the call must take it: without it
    // the pointer is only valid until the next eviction.
    const float* channel(int file_index, int sim_index, const std::string& name,
                         std::size_t* out_count, SampleBlock* out_block = nullptr);

    // Loads every physical column the named parameters depend on, for all the
    // given simulations, in one parallel pass. Calling this before plotting a
    // tile turns N lazy per-column loads into a single batched one.
    void preload(int file_index, const std::vector<int>& sim_indices,
                 const std::vector<std::string>& names);

    // Value of an ISO parameter per simulation. An ISO is constant within a
    // run, so the first sample is the value; NaN marks a run lacking it.
    std::vector<float> iso_values(int file_index, const std::string& name);

    // Distinct ISO values across every open file, ascending.
    std::vector<float> iso_distinct_values(const std::string& name);

    // ----------------------------------------------------------- memory

    // Bytes of sample data currently cached across every open file.
    std::size_t resident_bytes() const;

    // Frees least-recently-used cached columns until at most `keep_bytes`
    // remain. Data is re-read on demand, so this is always safe -- it trades
    // time for memory. Returns the resident total afterwards.
    std::size_t trim_memory(std::size_t keep_bytes);

private:
    struct Entry {
        std::unique_ptr<Dataset> data;
    };

    void rebuild_parameters();
    void rebuild_resolved_aliases();

    // Physical columns needed to evaluate `name`, following computed
    // definitions. Guards against cycles via `depth`.
    void collect_dependencies(const std::string& name, std::vector<std::string>& out,
                              int depth) const;

    SampleBlock evaluate_computed(Dataset& dataset, std::size_t sim_index,
                                  const std::string& real_name, int depth);

    // A display name for `path` that no open file already uses: the basename,
    // or the basename with the lowest free " (n)" suffix.
    std::string unique_display_name(const std::string& path) const;

    std::vector<Entry> files_;

    std::vector<std::string> parameters_;
    std::unordered_map<std::string, std::string> aliases_;
    std::unordered_map<std::string, std::string> resolved_aliases_;

    std::unordered_map<std::string, std::string> computed_sources_;
    std::unordered_map<std::string, Formula> computed_programs_;
};

}  // namespace aoe
