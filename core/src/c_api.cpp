// Author: Paulo R. A. Gilz
#include "aoe/aoe_c_api.h"

#include <cstring>
#include <string>
#include <vector>

#include "aoe/decimate.hpp"
#include "aoe/iso.hpp"
#include "aoe/session.hpp"
#include "aoe/thread_pool.hpp"

#define AOE_VERSION_STRING "2.0.0"

// The opaque handle. Carries the Session plus the small amount of scratch the
// C ABI needs to hand strings back safely.
struct aoe_session {
    aoe::Session session;
    // Guards the session's structure -- which files exist, which parameters and
    // aliases are defined. Readers share it; anything that adds, removes,
    // reloads or redefines takes it exclusively.
    aoe::RwLock lock;
    // The error and scratch strings are written from readers too, so they need
    // their own small lock rather than relying on the one above.
    aoe::Mutex text_lock;
    std::string error;
    std::string scratch;
    std::vector<std::string> removed;

    void set_error(const std::string& message) {
        aoe::LockGuard guard(text_lock);
        error = message;
    }
};

// A caller's reference to a channel's samples. It holds the block itself, not a
// path back to the session, so releasing one after the session has been
// destroyed is well defined -- which matters because on the Python side these
// are released by the garbage collector, in its own order, at its own time.
struct aoe_pin {
    aoe::SampleBlock block;
};

namespace {

// Every entry point funnels through one of these so a throw can never cross the
// ABI boundary -- unwinding out of a C function is undefined behaviour, and with
// ctypes on the other side it would take the interpreter down.
//
// They also take the session lock, which is what makes the core safe to use
// from the GUI thread and a worker at the same time. `guarded` is the shared,
// read-only form; `guarded_write` is exclusive and is used by everything that
// changes which files or parameters exist.
template <typename Fn>
auto run_guarded(aoe_session* s, Fn&& fn, decltype(fn()) fallback) -> decltype(fn()) {
    try {
        return fn();
    } catch (const std::exception& e) {
        s->set_error(e.what());
        return fallback;
    } catch (...) {
        s->set_error("unknown error");
        return fallback;
    }
}

template <typename Fn>
auto guarded(aoe_session* s, Fn&& fn, decltype(fn()) fallback) -> decltype(fn()) {
    if (!s) return fallback;
    aoe::ReadGuard guard(s->lock);
    return run_guarded(s, fn, fallback);
}

template <typename Fn>
auto guarded_write(aoe_session* s, Fn&& fn, decltype(fn()) fallback) -> decltype(fn()) {
    if (!s) return fallback;
    aoe::WriteGuard guard(s->lock);
    s->error.clear();
    return run_guarded(s, fn, fallback);
}

const char* empty_string() { return ""; }

}  // namespace

extern "C" {

// ------------------------------------------------------------ lifetime

const char* aoe_version(void) { return AOE_VERSION_STRING; }

int aoe_worker_count(void) {
    return static_cast<int>(aoe::ThreadPool::global().size()) + 1;
}

aoe_session* aoe_session_create(void) {
    try {
        return new aoe_session();
    } catch (...) {
        return nullptr;
    }
}

void aoe_session_destroy(aoe_session* session) { delete session; }

const char* aoe_last_error(aoe_session* session) {
    return session ? session->error.c_str() : empty_string();
}

// --------------------------------------------------------------- files

int aoe_begin_arrays(aoe_session* s, const char* path, const char* const* headers,
                     const char* const* units, int n_headers, int replace) {
    return guarded_write(s, [&] {
        if (!path || !headers || n_headers <= 0) {
            s->error = "no headers given";
            return -1;
        }
        std::vector<std::string> names;
        std::vector<std::string> unit_text;
        names.reserve(static_cast<std::size_t>(n_headers));
        unit_text.reserve(static_cast<std::size_t>(n_headers));
        for (int i = 0; i < n_headers; ++i) {
            names.push_back(headers[i] ? headers[i] : "");
            unit_text.push_back(units && units[i] ? units[i] : "");
        }
        const int index = s->session.begin_arrays_file(path, names, unit_text, replace);
        if (index < 0) s->error = std::string("could not open '") + path + "'";
        return index;
    }, -1);
}

int aoe_add_arrays_sim(aoe_session* s, int file, const char* id) {
    return guarded_write(s, [&] {
        aoe::Dataset* dataset = s->session.file(file);
        if (!dataset) {
            s->error = "no such file";
            return -1;
        }
        return dataset->add_arrays_sim(id ? id : "");
    }, -1);
}

int aoe_set_arrays_column(aoe_session* s, int file, int sim, int column,
                          const float* data, size_t count) {
    return guarded_write(s, [&] {
        aoe::Dataset* dataset = s->session.file(file);
        if (!dataset) {
            s->error = "no such file";
            return 0;
        }
        return dataset->set_arrays_column(static_cast<std::size_t>(sim), column, data,
                                          count)
                   ? 1
                   : 0;
    }, 0);
}

void aoe_end_arrays(aoe_session* s, int file) {
    guarded_write(s, [&] {
        s->session.end_arrays_file(file);
        return 0;
    }, 0);
}

int aoe_add_file(aoe_session* s, const char* path) {
    return guarded_write(s, [&] {
        if (!path) {
            s->error = "no path given";
            return -1;
        }
        const int index = s->session.add_file(path);
        if (index < 0) s->error = std::string("could not read '") + path + "'";
        return index;
    }, -1);
}

void aoe_remove_file(aoe_session* s, int file) {
    guarded_write(s, [&] { s->session.remove_file(file); return 0; }, 0);
}

void aoe_clear_files(aoe_session* s) {
    guarded_write(s, [&] { s->session.clear(); return 0; }, 0);
}

int aoe_file_count(aoe_session* s) {
    return guarded(s, [&] { return static_cast<int>(s->session.file_count()); }, 0);
}

const char* aoe_file_name(aoe_session* s, int file) {
    return guarded(s, [&]() -> const char* {
        const aoe::Dataset* d = s->session.file(file);
        return d ? d->name().c_str() : empty_string();
    }, empty_string());
}

const char* aoe_file_path(aoe_session* s, int file) {
    return guarded(s, [&]() -> const char* {
        const aoe::Dataset* d = s->session.file(file);
        return d ? d->path().c_str() : empty_string();
    }, empty_string());
}

int aoe_stale_files(aoe_session* s, int* out, int max) {
    return guarded(s, [&] {
        const std::vector<int> stale = s->session.stale_files();
        const int n = static_cast<int>(stale.size());
        if (out)
            for (int i = 0; i < n && i < max; ++i) out[i] = stale[static_cast<std::size_t>(i)];
        return n;
    }, 0);
}

int aoe_reload_file(aoe_session* s, int file) {
    return guarded_write(s, [&] { return s->session.reload_file(file) ? 1 : 0; }, 0);
}

void aoe_set_file_name(aoe_session* s, int file, const char* name) {
    guarded_write(s, [&] {
        aoe::Dataset* d = s->session.file(file);
        if (d && name && *name) d->set_name(name);
        return 0;
    }, 0);
}

void aoe_set_sim_name(aoe_session* s, int file, int sim, const char* name) {
    guarded_write(s, [&] {
        aoe::Dataset* d = s->session.file(file);
        if (d && name && *name && sim >= 0 &&
            sim < static_cast<int>(d->simulation_count()))
            d->set_simulation_name(static_cast<std::size_t>(sim), name);
        return 0;
    }, 0);
}

const char* aoe_sim_name(aoe_session* s, int file, int sim) {
    return guarded(s, [&]() -> const char* {
        const aoe::Dataset* d = s->session.file(file);
        if (!d || sim < 0 || sim >= static_cast<int>(d->simulation_count()))
            return empty_string();
        const aoe::Simulation& simulation = d->simulation(static_cast<std::size_t>(sim));
        return simulation.display_name.empty() ? simulation.id.c_str()
                                               : simulation.display_name.c_str();
    }, empty_string());
}

int aoe_sim_count(aoe_session* s, int file) {
    return guarded(s, [&] {
        const aoe::Dataset* d = s->session.file(file);
        return d ? static_cast<int>(d->simulation_count()) : 0;
    }, 0);
}

const char* aoe_sim_id(aoe_session* s, int file, int sim) {
    return guarded(s, [&]() -> const char* {
        const aoe::Dataset* d = s->session.file(file);
        if (!d || sim < 0 || sim >= static_cast<int>(d->simulation_count()))
            return empty_string();
        return d->simulation(static_cast<std::size_t>(sim)).id.c_str();
    }, empty_string());
}

const char* aoe_sim_key(aoe_session* s, int file, int sim) {
    return guarded(s, [&]() -> const char* {
        const aoe::Dataset* d = s->session.file(file);
        if (!d || sim < 0 || sim >= static_cast<int>(d->simulation_count()))
            return empty_string();
        return d->simulation(static_cast<std::size_t>(sim)).unique_key.c_str();
    }, empty_string());
}

int aoe_header_count(aoe_session* s, int file) {
    return guarded(s, [&] {
        const aoe::Dataset* d = s->session.file(file);
        return d ? static_cast<int>(d->headers().size()) : 0;
    }, 0);
}

const char* aoe_header(aoe_session* s, int file, int index) {
    return guarded(s, [&]() -> const char* {
        const aoe::Dataset* d = s->session.file(file);
        if (!d || index < 0 || index >= static_cast<int>(d->headers().size()))
            return empty_string();
        return d->headers()[static_cast<std::size_t>(index)].c_str();
    }, empty_string());
}

const char* aoe_header_unit(aoe_session* s, int file, int index) {
    return guarded(s, [&]() -> const char* {
        const aoe::Dataset* d = s->session.file(file);
        if (!d) return empty_string();
        return d->unit(index).c_str();
    }, empty_string());
}

const char* aoe_param_unit(aoe_session* s, const char* name) {
    return guarded(s, [&]() -> const char* {
        if (!name) return empty_string();
        return s->session.parameter_unit(name).c_str();
    }, empty_string());
}

// ---------------------------------------------------------- parameters

int aoe_param_count(aoe_session* s) {
    return guarded(s, [&] { return static_cast<int>(s->session.parameters().size()); }, 0);
}

const char* aoe_param(aoe_session* s, int index) {
    return guarded(s, [&]() -> const char* {
        const auto& params = s->session.parameters();
        if (index < 0 || index >= static_cast<int>(params.size())) return empty_string();
        return params[static_cast<std::size_t>(index)].c_str();
    }, empty_string());
}

void aoe_set_alias(aoe_session* s, const char* alias, const char* target) {
    guarded_write(s, [&] {
        if (alias && target) s->session.set_alias(alias, target);
        return 0;
    }, 0);
}

void aoe_remove_alias(aoe_session* s, const char* alias) {
    guarded_write(s, [&] {
        if (alias) s->session.remove_alias(alias);
        return 0;
    }, 0);
}

const char* aoe_resolve(aoe_session* s, const char* name) {
    return guarded_write(s, [&]() -> const char* {
        if (!name) return empty_string();
        s->scratch = s->session.resolve(name);
        return s->scratch.c_str();
    }, empty_string());
}

int aoe_set_computed(aoe_session* s, const char* name, const char* expression) {
    return guarded_write(s, [&] {
        if (!name || !expression) {
            s->error = "name and expression are required";
            return 0;
        }
        std::string error;
        const bool ok = s->session.set_computed(name, expression, &error);
        if (!ok) s->error = error;
        return ok ? 1 : 0;
    }, 0);
}

int aoe_remove_computed(aoe_session* s, const char* name) {
    return guarded_write(s, [&] {
        if (!name) return 0;
        s->removed = s->session.remove_computed(name);
        return static_cast<int>(s->removed.size());
    }, 0);
}

const char* aoe_removed_name(aoe_session* s, int index) {
    return guarded(s, [&]() -> const char* {
        if (index < 0 || index >= static_cast<int>(s->removed.size())) return empty_string();
        return s->removed[static_cast<std::size_t>(index)].c_str();
    }, empty_string());
}

int aoe_is_computed(aoe_session* s, const char* name) {
    return guarded(s, [&] { return (name && s->session.is_computed(name)) ? 1 : 0; }, 0);
}

const char* aoe_computed_source(aoe_session* s, const char* name) {
    return guarded(s, [&]() -> const char* {
        if (!name) return empty_string();
        const auto& sources = s->session.computed();
        auto it = sources.find(s->session.resolve(name));
        if (it == sources.end()) return empty_string();
        return it->second.c_str();
    }, empty_string());
}

int aoe_validate_formula(aoe_session* s, const char* expression) {
    return guarded(s, [&] {
        if (!expression) return 0;
        std::string error;
        const bool ok = aoe::Formula::validate(
            expression,
            [&](const std::string& name) { return s->session.is_known_parameter(name); },
            &error);
        if (!ok) s->error = error;
        return ok ? 1 : 0;
    }, 0);
}

// ---------------------------------------------------------------- data

const float* aoe_channel(aoe_session* s, int file, int sim, const char* name,
                         size_t* out_count) {
    if (out_count) *out_count = 0;
    return guarded(s, [&]() -> const float* {
        if (!name) return nullptr;
        std::size_t count = 0;
        const float* data = s->session.channel(file, sim, name, &count);
        if (out_count) *out_count = count;
        return data;
    }, static_cast<const float*>(nullptr));
}

const float* aoe_channel_pinned(aoe_session* s, int file, int sim, const char* name,
                                size_t* out_count, aoe_pin** out_pin) {
    if (out_count) *out_count = 0;
    if (out_pin) *out_pin = nullptr;
    return guarded(s, [&]() -> const float* {
        if (!name) return nullptr;
        std::size_t count = 0;
        aoe::SampleBlock block;
        const float* data = s->session.channel(file, sim, name, &count, &block);
        if (!data || !block) return nullptr;
        if (out_pin) {
            aoe_pin* pin = new aoe_pin();
            pin->block = std::move(block);
            *out_pin = pin;
        }
        if (out_count) *out_count = count;
        return data;
    }, static_cast<const float*>(nullptr));
}

void aoe_pin_release(aoe_pin* pin) { delete pin; }

int aoe_pin_sorted(aoe_pin* pin) {
    return (pin && pin->block && pin->block->monotonic) ? 1 : 0;
}

int aoe_pin_bounds(aoe_pin* pin, float* out_min, float* out_max) {
    if (!pin || !pin->block || !pin->block->has_finite) return 0;
    if (out_min) *out_min = pin->block->minimum;
    if (out_max) *out_max = pin->block->maximum;
    return 1;
}

size_t aoe_pin_count(aoe_pin* pin) {
    return (pin && pin->block) ? pin->block->values.size() : 0;
}

void aoe_preload(aoe_session* s, int file, const int* sims, int n_sims,
                 const char* const* names, int n_names) {
    guarded(s, [&] {
        std::vector<int> sim_list;
        if (sims && n_sims > 0) {
            sim_list.reserve(static_cast<std::size_t>(n_sims));
            for (int i = 0; i < n_sims; ++i) sim_list.push_back(sims[i]);
        }
        std::vector<std::string> name_list;
        if (names && n_names > 0) {
            name_list.reserve(static_cast<std::size_t>(n_names));
            for (int i = 0; i < n_names; ++i)
                if (names[i]) name_list.emplace_back(names[i]);
        }
        s->session.preload(file, sim_list, name_list);
        return 0;
    }, 0);
}

// ------------------------------------------------------------ plotting

size_t aoe_decimate_ex(const float* x, const float* y, size_t count, float x_min,
                       float x_max, size_t max_points, int x_sorted, float* out_x,
                       float* out_y) {
    if (!x || !y || !out_x || !out_y || count == 0) return 0;
    if (x_min < x_max)
        return aoe::decimate_minmax_range(x, y, count, x_min, x_max, max_points, out_x,
                                          out_y, x_sorted);
    return aoe::decimate_minmax(x, y, count, max_points, out_x, out_y);
}

size_t aoe_decimate(const float* x, const float* y, size_t count, float x_min,
                    float x_max, size_t max_points, float* out_x, float* out_y) {
    // -1: work the ordering out rather than assume it. Callers that already know
    // should say so through aoe_decimate_ex and save the scan.
    return aoe_decimate_ex(x, y, count, x_min, x_max, max_points, -1, out_x, out_y);
}

int aoe_series_sorted(const float* values, size_t count) {
    return aoe::is_non_decreasing(values, count) ? 1 : 0;
}

int aoe_series_bounds(const float* values, size_t count, float* out_min, float* out_max) {
    return aoe::series_bounds(values, count, out_min, out_max) ? 1 : 0;
}

size_t aoe_nearest_index(const float* x, size_t count, float target) {
    return aoe::nearest_index(x, count, target);
}

int aoe_series_stats_ex(const float* x, const float* y, size_t count, float x_min,
                        float x_max, int x_sorted, double* out) {
    if (!out) return 0;
    aoe::SeriesStats stats;
    if (!aoe::series_stats(x, y, count, x_min, x_max, &stats, x_sorted)) return 0;
    out[0] = stats.minimum;
    out[1] = stats.maximum;
    out[2] = stats.mean;
    out[3] = stats.stddev;
    out[4] = stats.first;
    out[5] = stats.last;
    out[6] = static_cast<double>(stats.count);
    return 1;
}

int aoe_series_stats(const float* x, const float* y, size_t count, float x_min,
                     float x_max, double* out) {
    return aoe_series_stats_ex(x, y, count, x_min, x_max, -1, out);
}

size_t aoe_resident_bytes(aoe_session* s) {
    return guarded(s, [&] { return s->session.resident_bytes(); }, std::size_t{0});
}

size_t aoe_trim_memory(aoe_session* s, size_t keep_bytes) {
    return guarded(s, [&] { return s->session.trim_memory(keep_bytes); }, std::size_t{0});
}

// --------------------------------------------------------------- cache

int aoe_cache_enabled(void) { return aoe::caching_enabled() ? 1 : 0; }

int aoe_cache_active(aoe_session* s, int file) {
    return guarded(s, [&] {
        const aoe::Dataset* d = s->session.file(file);
        return (d && d->cache_active()) ? 1 : 0;
    }, 0);
}

const char* aoe_cache_path(aoe_session* s, int file) {
    return guarded(s, [&]() -> const char* {
        const aoe::Dataset* d = s->session.file(file);
        if (!d) return empty_string();
        aoe::LockGuard guard(s->text_lock);
        s->scratch = d->cache_file();
        return s->scratch.c_str();
    }, empty_string());
}

int aoe_build_cache(aoe_session* s, int file) {
    // Exclusive: this parses the whole file, replaces the attached sidecar and
    // drops the columns it pulled in to do so. A reader running alongside would
    // see the dataset mid-rebuild.
    return guarded_write(s, [&] {
        aoe::Dataset* d = s->session.file(file);
        if (!d) {
            s->error = "no such file";
            return 0;
        }
        if (!aoe::caching_enabled()) {
            s->error = "caching is switched off (AOE_CACHE)";
            return 0;
        }
        std::string written;
        if (!d->write_cache(&written)) {
            s->error = "could not write a cache for '" + d->path() + "'";
            return 0;
        }
        return 1;
    }, 0);
}

int aoe_discard_cache(aoe_session* s, int file) {
    return guarded_write(s, [&] {
        aoe::Dataset* d = s->session.file(file);
        return (d && d->discard_cache()) ? 1 : 0;
    }, 0);
}

// ----------------------------------------------------------------- iso

int aoe_iso_values(aoe_session* s, int file, const char* name, float* out, int max) {
    return guarded(s, [&] {
        if (!name) return 0;
        const std::vector<float> values = s->session.iso_values(file, name);
        const int n = static_cast<int>(values.size());
        if (out)
            for (int i = 0; i < n && i < max; ++i) out[i] = values[static_cast<std::size_t>(i)];
        return n;
    }, 0);
}

int aoe_iso_distinct(aoe_session* s, const char* name, float* out, int max) {
    return guarded(s, [&] {
        if (!name) return 0;
        const std::vector<float> values = s->session.iso_distinct_values(name);
        const int n = static_cast<int>(values.size());
        if (out)
            for (int i = 0; i < n && i < max; ++i) out[i] = values[static_cast<std::size_t>(i)];
        return n;
    }, 0);
}

int aoe_iso_auto_group(const float* values, int count, int group_count, float* out_bounds,
                       int max_groups) {
    if (!values || count <= 0 || group_count <= 0) return 0;
    std::vector<float> input(values, values + count);
    const std::vector<aoe::IsoGroup> groups = aoe::iso_auto_group(input, group_count);
    const int n = static_cast<int>(groups.size());
    if (out_bounds)
        for (int i = 0; i < n && i < max_groups; ++i) {
            out_bounds[i * 2] = groups[static_cast<std::size_t>(i)].min_bound;
            out_bounds[i * 2 + 1] = groups[static_cast<std::size_t>(i)].max_bound;
        }
    return n;
}

int aoe_iso_group_index(float value, const float* bounds, int group_count) {
    if (!bounds || group_count <= 0) return -1;
    std::vector<aoe::IsoGroup> groups(static_cast<std::size_t>(group_count));
    for (int i = 0; i < group_count; ++i) {
        groups[static_cast<std::size_t>(i)].min_bound = bounds[i * 2];
        groups[static_cast<std::size_t>(i)].max_bound = bounds[i * 2 + 1];
    }
    return aoe::iso_group_index(value, groups);
}

const char* aoe_iso_format(aoe_session* s, float value) {
    return guarded_write(s, [&]() -> const char* {
        s->scratch = aoe::iso_format_value(value);
        return s->scratch.c_str();
    }, empty_string());
}

}  // extern "C"
