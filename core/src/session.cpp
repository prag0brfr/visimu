// Author: Paulo R. A. Gilz
#include "aoe/session.hpp"

#include <algorithm>
#include <functional>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace aoe {
namespace {

constexpr int kMaxDepth = 32;
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

}  // namespace

// ------------------------------------------------------------- files

std::string Session::unique_display_name(const std::string& path) const {
    const std::size_t slash = path.find_last_of("/\\");
    const std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);

    // Two files with the same basename from different directories are common
    // (results/run1/out.dat, results/run2/out.dat); suffix so legends stay
    // readable and unique_key stays unique.
    //
    // The suffix is chosen against the names actually in use, not against a
    // running count of everything ever added. A counter never came back down on
    // removal, so closing a file and reopening it renamed it "flight.aoe (2)"
    // -- and since the display name is part of every run's identity key, that
    // silently discarded every colour and line style the user had set for it,
    // saved workspaces included.
    const auto taken = [this](const std::string& candidate) {
        for (const auto& entry : files_)
            if (entry.data->name() == candidate) return true;
        return false;
    };

    if (!taken(base)) return base;
    for (int n = 2; n < 10000; ++n) {
        std::string candidate = base + " (" + std::to_string(n) + ")";
        if (!taken(candidate)) return candidate;
    }
    return base;
}

int Session::begin_arrays_file(const std::string& path,
                               const std::vector<std::string>& headers,
                               const std::vector<std::string>& units, int replace) {
    if (replace >= 0) {
        if (static_cast<std::size_t>(replace) >= files_.size()) return -1;
        Dataset* existing = files_[static_cast<std::size_t>(replace)].data.get();
        if (!existing || !existing->begin_arrays(path, headers, units)) return -1;
        return replace;
    }

    auto dataset = std::unique_ptr<Dataset>(new Dataset());
    dataset->set_name(unique_display_name(path));
    if (!dataset->begin_arrays(path, headers, units)) return -1;

    files_.push_back(Entry{std::move(dataset)});
    return static_cast<int>(files_.size()) - 1;
}

void Session::end_arrays_file(int index) {
    if (index < 0 || static_cast<std::size_t>(index) >= files_.size()) return;
    Dataset* dataset = files_[static_cast<std::size_t>(index)].data.get();
    if (!dataset) return;
    dataset->end_arrays();
    rebuild_parameters();
}

int Session::add_file(const std::string& path) {
    auto dataset = std::unique_ptr<Dataset>(new Dataset());
    dataset->set_name(unique_display_name(path));

    if (!dataset->open(path)) return -1;

    files_.push_back(Entry{std::move(dataset)});
    rebuild_parameters();
    return static_cast<int>(files_.size()) - 1;
}

void Session::remove_file(int index) {
    if (index < 0 || index >= static_cast<int>(files_.size())) return;
    files_.erase(files_.begin() + index);
    rebuild_parameters();
}

void Session::clear() {
    files_.clear();
    parameters_.clear();
    // Aliases and computed definitions are user work, not file state; they
    // survive a "clear files" so reopening a study does not lose them.
    rebuild_parameters();
}

Dataset* Session::file(int index) {
    if (index < 0 || index >= static_cast<int>(files_.size())) return nullptr;
    return files_[static_cast<std::size_t>(index)].data.get();
}

const Dataset* Session::file(int index) const {
    if (index < 0 || index >= static_cast<int>(files_.size())) return nullptr;
    return files_[static_cast<std::size_t>(index)].data.get();
}

std::vector<int> Session::stale_files() const {
    std::vector<int> stale;
    for (std::size_t i = 0; i < files_.size(); ++i)
        if (files_[i].data->changed_on_disk()) stale.push_back(static_cast<int>(i));
    return stale;
}

bool Session::reload_file(int index) {
    Dataset* dataset = file(index);
    if (!dataset) return false;
    const bool ok = dataset->reload();
    rebuild_parameters();
    return ok;
}

// -------------------------------------------------------- parameters

void Session::rebuild_parameters() {
    parameters_.clear();
    std::unordered_set<std::string> seen;

    for (const auto& entry : files_)
        for (const auto& header : entry.data->headers())
            if (seen.insert(header).second) parameters_.push_back(header);

    for (const auto& item : computed_sources_)
        if (seen.insert(item.first).second) parameters_.push_back(item.first);

    for (const auto& item : aliases_)
        if (seen.insert(item.first).second) parameters_.push_back(item.first);

    std::sort(parameters_.begin(), parameters_.end());
    rebuild_resolved_aliases();
}

void Session::rebuild_resolved_aliases() {
    resolved_aliases_.clear();
    for (const auto& item : aliases_) {
        std::string current = item.first;
        std::unordered_set<std::string> visited;
        while (true) {
            auto it = aliases_.find(current);
            if (it == aliases_.end()) break;
            if (!visited.insert(current).second) break;  // cycle: stop where we are
            current = it->second;
        }
        resolved_aliases_[item.first] = current;
    }
}

void Session::set_alias(const std::string& alias, const std::string& target) {
    if (alias.empty() || target.empty() || alias == target) return;
    aliases_[alias] = target;
    rebuild_parameters();
}

void Session::remove_alias(const std::string& alias) {
    aliases_.erase(alias);
    rebuild_parameters();
}

std::string Session::resolve(const std::string& name) const {
    auto it = resolved_aliases_.find(name);
    return it == resolved_aliases_.end() ? name : it->second;
}

const std::string& Session::parameter_unit(const std::string& name) const {
    static const std::string kNone;
    const std::string real = resolve(name);
    for (const auto& entry : files_) {
        const int column = entry.data->column_index(real);
        if (column >= 0) {
            const std::string& unit = entry.data->unit(column);
            // Keep looking if this file declares no unit: a parameter present
            // in several files needs only one of them to name it.
            if (!unit.empty()) return unit;
        }
    }
    return kNone;
}

bool Session::is_computed(const std::string& name) const {
    return computed_sources_.count(resolve(name)) != 0;
}

bool Session::is_known_parameter(const std::string& name) const {
    const std::string real = resolve(name);
    if (computed_sources_.count(real)) return true;
    for (const auto& entry : files_)
        if (entry.data->column_index(real) >= 0) return true;
    return false;
}

bool Session::set_computed(const std::string& name, const std::string& expression,
                           std::string* error) {
    if (name.empty()) {
        if (error) *error = "the parameter needs a name";
        return false;
    }

    std::string compile_error;
    Formula formula = Formula::compile(expression, &compile_error);
    if (!formula.valid()) {
        if (error) *error = compile_error;
        return false;
    }

    for (const auto& variable : formula.variables()) {
        if (!is_known_parameter(variable)) {
            if (error) *error = "unknown parameter '" + variable + "'";
            return false;
        }
    }

    // Reject cycles before storing, so a bad definition can never make
    // evaluation recurse. Walk the dependency graph as it would look with this
    // definition in place.
    {
        auto saved_source = computed_sources_.find(name);
        const bool had = saved_source != computed_sources_.end();
        const std::string previous = had ? saved_source->second : std::string();
        Formula previous_program;
        if (had) previous_program = computed_programs_[name];

        computed_sources_[name] = expression;
        computed_programs_[name] = formula;

        std::vector<std::string> stack{name};
        std::unordered_set<std::string> visiting;
        bool cyclic = false;

        std::function<void(const std::string&, int)> walk = [&](const std::string& current,
                                                                int depth) {
            if (cyclic || depth > kMaxDepth) {
                if (depth > kMaxDepth) cyclic = true;
                return;
            }
            auto it = computed_programs_.find(resolve(current));
            if (it == computed_programs_.end()) return;
            if (!visiting.insert(resolve(current)).second) {
                cyclic = true;
                return;
            }
            for (const auto& variable : it->second.variables()) walk(variable, depth + 1);
            visiting.erase(resolve(current));
        };
        walk(name, 0);

        if (cyclic) {
            if (had) {
                computed_sources_[name] = previous;
                computed_programs_[name] = previous_program;
            } else {
                computed_sources_.erase(name);
                computed_programs_.erase(name);
            }
            if (error) *error = "'" + name + "' would depend on itself";
            return false;
        }
    }

    // Any cached results that could depend on this definition are now stale.
    for (auto& entry : files_) entry.data->clear_computed();

    rebuild_parameters();
    if (error) error->clear();
    return true;
}

std::vector<std::string> Session::remove_computed(const std::string& name) {
    std::vector<std::string> removed;
    std::vector<std::string> queue{name};

    while (!queue.empty()) {
        const std::string current = queue.back();
        queue.pop_back();
        if (!computed_sources_.count(current)) continue;

        computed_sources_.erase(current);
        computed_programs_.erase(current);
        removed.push_back(current);

        // Anything defined in terms of it can no longer be evaluated.
        for (const auto& item : computed_programs_) {
            const auto& variables = item.second.variables();
            if (std::find(variables.begin(), variables.end(), current) != variables.end())
                queue.push_back(item.first);
        }
    }

    for (auto& entry : files_) entry.data->clear_computed();

    rebuild_parameters();
    return removed;
}

// ------------------------------------------------------------ access

void Session::collect_dependencies(const std::string& name,
                                   std::vector<std::string>& out, int depth) const {
    if (depth > kMaxDepth) return;
    const std::string real = resolve(name);

    auto it = computed_programs_.find(real);
    if (it == computed_programs_.end()) {
        if (std::find(out.begin(), out.end(), real) == out.end()) out.push_back(real);
        return;
    }
    for (const auto& variable : it->second.variables())
        collect_dependencies(variable, out, depth + 1);
}

void Session::preload(int file_index, const std::vector<int>& sim_indices,
                      const std::vector<std::string>& names) {
    Dataset* dataset = file(file_index);
    if (!dataset) return;

    std::vector<std::string> physical;
    for (const auto& name : names) collect_dependencies(name, physical, 0);

    std::vector<int> columns;
    columns.reserve(physical.size());
    for (const auto& name : physical) {
        const int column = dataset->column_index(name);
        if (column >= 0) columns.push_back(column);
    }
    if (columns.empty()) return;

    std::sort(columns.begin(), columns.end());
    columns.erase(std::unique(columns.begin(), columns.end()), columns.end());

    dataset->load_columns(columns, sim_indices);
}

SampleBlock Session::evaluate_computed(Dataset& dataset, std::size_t sim_index,
                                       const std::string& real_name, int depth) {
    Simulation& sim = dataset.simulation(sim_index);

    {
        LockGuard guard(sim.lock);
        auto cached = sim.computed.find(real_name);
        if (cached != sim.computed.end()) return cached->second;
    }

    if (depth > kMaxDepth) return SampleBlock();

    auto program = computed_programs_.find(real_name);
    if (program == computed_programs_.end()) return SampleBlock();

    // Load everything the expression needs in one batch before evaluating.
    {
        std::vector<std::string> physical;
        collect_dependencies(real_name, physical, depth);
        std::vector<int> columns;
        for (const auto& name : physical) {
            const int column = dataset.column_index(name);
            if (column >= 0) columns.push_back(column);
        }
        if (!columns.empty()) {
            std::sort(columns.begin(), columns.end());
            columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
            dataset.load_columns(columns, {static_cast<int>(sim_index)});
        }
    }

    // Bound to this dataset and simulation, so nested computed channels resolve
    // recursively while physical ones come straight from the column cache.
    //
    // Every block the resolver hands out is kept in `held` for the duration of
    // the evaluation: an inner load can trip the memory cap and evict a block an
    // outer expression is still reading from.
    std::vector<SampleBlock> held;
    ChannelResolver resolver = [&](const std::string& variable,
                                   std::size_t& count) -> const float* {
        const std::string real = resolve(variable);
        SampleBlock block;
        if (computed_programs_.count(real)) {
            block = evaluate_computed(dataset, sim_index, real, depth + 1);
        } else {
            const int column = dataset.column_index(real);
            if (column < 0) return nullptr;
            block = dataset.column_block(sim_index, column);
        }
        if (!block || block->values.empty()) return nullptr;
        count = block->values.size();
        const float* data = block->values.data();
        held.push_back(std::move(block));
        return data;
    };

    std::vector<float> result;
    std::string error;

    std::size_t rows = dataset.loaded_row_count(sim_index);
    if (program->second.is_constant() && rows == 0) {
        // A formula with no channel references ("2 + 3 * 4", "pi") has nothing
        // to take its length from. Load one real column so the constant spans
        // the same samples as everything else in the run; without this the
        // result is empty and the parameter looks broken.
        if (dataset.column_count() > 0) {
            dataset.load_columns({0}, {static_cast<int>(sim_index)});
            rows = dataset.loaded_row_count(sim_index);
        }
        if (rows == 0) rows = 1;
    }
    SampleBlock block = std::make_shared<SampleData>();
    if (program->second.evaluate(resolver, program->second.is_constant() ? rows : 0,
                                 result, &error)) {
        block->values = std::move(result);
        summarise(*block);
    }
    // On failure the empty block is cached all the same. Without it the UI
    // retries the same doomed evaluation on every repaint, and the tempting
    // patch for that -- injecting a fake 0.0 sample -- turns a channel that
    // could not be computed into a flat line that looks like data.

    LockGuard guard(sim.lock);
    auto inserted = sim.computed.emplace(real_name, std::move(block));
    return inserted.first->second;
}

const float* Session::channel(int file_index, int sim_index, const std::string& name,
                              std::size_t* out_count, SampleBlock* out_block) {
    if (out_count) *out_count = 0;
    if (out_block) out_block->reset();

    Dataset* dataset = file(file_index);
    if (!dataset) return nullptr;
    if (sim_index < 0 || sim_index >= static_cast<int>(dataset->simulation_count()))
        return nullptr;

    const std::string real = resolve(name);

    SampleBlock block;
    if (computed_programs_.count(real)) {
        block = evaluate_computed(*dataset, static_cast<std::size_t>(sim_index), real, 0);
    } else {
        const int column = dataset->column_index(real);
        if (column < 0) return nullptr;
        block = dataset->column_block(static_cast<std::size_t>(sim_index), column);
    }

    if (!block || block->values.empty()) return nullptr;
    if (out_count) *out_count = block->values.size();
    const float* data = block->values.data();
    if (out_block) *out_block = std::move(block);
    return data;
}

std::vector<float> Session::iso_values(int file_index, const std::string& name) {
    std::vector<float> values;
    Dataset* dataset = file(file_index);
    if (!dataset) return values;

    const std::size_t count = dataset->simulation_count();
    values.reserve(count);

    // Fast path: a plain column read through first_value() parses only the
    // first record of each block. The obvious implementation -- preload the
    // column and take sample zero -- costs a full scan of the file to obtain
    // one float per run, which on a 500 MB file measured at 1.4 s per ISO
    // parameter. This is the same answer for the price of a few hundred bytes.
    const std::string real = resolve(name);
    if (!computed_programs_.count(real)) {
        const int column = dataset->column_index(real);
        if (column >= 0) return dataset->first_values(column);
    }

    // Computed channels have to be evaluated, which needs their inputs.
    preload(file_index, {}, {name});
    for (std::size_t s = 0; s < count; ++s) {
        std::size_t n = 0;
        const float* data = channel(file_index, static_cast<int>(s), name, &n);
        values.push_back((data && n > 0) ? data[0] : kNaN);
    }
    return values;
}

std::size_t Session::resident_bytes() const {
    std::size_t total = 0;
    for (const auto& entry : files_) total += entry.data->resident_bytes();
    return total;
}

std::size_t Session::trim_memory(std::size_t keep_bytes) {
    std::size_t total = resident_bytes();
    if (total <= keep_bytes) return total;

    // Spread the budget over the open files by their current share, so trimming
    // does not empty the first file while leaving the last untouched.
    for (auto& entry : files_) {
        const std::size_t share = entry.data->resident_bytes();
        if (total == 0) break;
        const std::size_t allowance =
            static_cast<std::size_t>(static_cast<double>(keep_bytes) *
                                     (static_cast<double>(share) / static_cast<double>(total)));
        entry.data->evict_until(allowance);
    }
    return resident_bytes();
}

std::vector<float> Session::iso_distinct_values(const std::string& name) {
    std::vector<float> all;
    for (std::size_t i = 0; i < files_.size(); ++i) {
        std::vector<float> values = iso_values(static_cast<int>(i), name);
        all.insert(all.end(), values.begin(), values.end());
    }

    all.erase(std::remove_if(all.begin(), all.end(),
                             [](float v) { return std::isnan(v); }),
              all.end());
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    return all;
}

}  // namespace aoe
