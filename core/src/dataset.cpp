// Author: Paulo R. A. Gilz
#include "aoe/dataset.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>

#include "aoe/fast_parse.hpp"
#include "aoe/thread_pool.hpp"

namespace aoe {
namespace {

constexpr char kParamMarker[] = "*PARAMETRES";
constexpr char kTableMarker[] = "*TABLEAU";

// Smallest byte range worth handing to its own thread. Below this the
// synchronisation costs more than the parsing saves.
constexpr std::size_t kMinChunkBytes = 1u << 21;  // 2 MiB

// Monotonic counter behind least-recently-used eviction. A counter rather than a
// clock: it needs only to order accesses, and reading a clock per column access
// would show up in profiles.
std::atomic<std::uint64_t> g_use_stamp{1};

inline std::uint64_t next_use_stamp() {
    return g_use_stamp.fetch_add(1, std::memory_order_relaxed);
}

struct Line {
    const char* begin;
    std::size_t length;   // trailing CR and leading blanks removed
    const char* next;     // start of the following line
};

// Splits one line out of [p, end), trimming leading blanks and a trailing CR.
inline Line next_line(const char* p, const char* end) noexcept {
    const char* eol = static_cast<const char*>(::memchr(p, '\n', static_cast<std::size_t>(end - p)));
    const char* stop = eol ? eol : end;
    const char* next = eol ? eol + 1 : end;

    std::size_t len = static_cast<std::size_t>(stop - p);
    while (len > 0 && (p[len - 1] == '\r' || p[len - 1] == ' ' || p[len - 1] == '\t')) --len;

    const char* begin = p;
    while (len > 0 && is_space(*begin)) {
        ++begin;
        --len;
    }
    return Line{begin, len, next};
}

inline bool starts_data(const char* p, std::size_t len) noexcept {
    if (len == 0) return false;
    const char c = *p;
    return is_digit(c) || c == '-' || c == '+' || c == '.';
}

// Extracts every single-quoted token on a header line, empty ones included.
// The empty ones matter: in the indexed dialect below, a parameter with no unit
// is written '' and dropping it would shift the unit of the next line onto it.
void extract_quoted(const char* p, std::size_t len, std::vector<std::string>& out) {
    const char* end = p + len;
    while (p < end) {
        const char* open = static_cast<const char*>(::memchr(p, '\'', static_cast<std::size_t>(end - p)));
        if (!open) return;
        ++open;
        if (open >= end) return;
        const char* close = static_cast<const char*>(::memchr(open, '\'', static_cast<std::size_t>(end - open)));
        if (!close) return;

        const char* s = open;
        const char* e = close;
        while (s < e && is_space(*s)) ++s;
        while (e > s && is_space(*(e - 1))) --e;
        out.emplace_back(s, static_cast<std::size_t>(e > s ? e - s : 0));

        p = close + 1;
    }
}

// Is this header line of the indexed form?
//
//     <tab>12<tab>'PARAMETER011'<tab> <tab>'deg'
//
// Two dialects of *PARAMETRES exist in the wild. One packs several names onto a
// line and gives no units; the other writes one parameter per line, prefixed by
// its ordinal and followed by its unit. Told apart by what precedes the first
// quote: an ordinal, or nothing.
//
// Getting this wrong is not a cosmetic problem. Treating every quoted token as
// a name turns a 585-parameter file into 1156 parameters -- 'deg' and 'ft' and
// 's' appearing in the parameter list -- and, far worse, shifts every column
// index, so TIME silently returns somebody else's data. It looks like it
// loaded.
//
// The test is deliberately strict: digits, then only blanks, then a quote.
// A packed line begins with the quote itself, so it can never match.
bool is_indexed_header_line(const char* p, std::size_t len) noexcept {
    const char* const end = p + len;
    if (p >= end || !is_digit(*p)) return false;
    while (p < end && is_digit(*p)) ++p;
    while (p < end && is_space(*p)) ++p;
    return p < end && *p == '\'';
}

// Parses one line of a *PARAMETRES block into names and their units, keeping
// the two vectors the same length. Returns the number of names added.
std::size_t parse_header_line(const char* p, std::size_t len,
                              std::vector<std::string>& names,
                              std::vector<std::string>& units) {
    std::vector<std::string> tokens;
    extract_quoted(p, len, tokens);
    if (tokens.empty()) return 0;

    if (is_indexed_header_line(p, len)) {
        // First token is the name, second (if present) is its unit. Anything
        // further on the line is a comment as far as this is concerned.
        if (tokens[0].empty()) return 0;   // '' as a name is not a parameter
        names.push_back(tokens[0]);
        units.push_back(tokens.size() > 1 ? tokens[1] : std::string());
        return 1;
    }

    std::size_t added = 0;
    for (const std::string& token : tokens) {
        if (token.empty()) continue;
        names.push_back(token);
        units.emplace_back();
        ++added;
    }
    return added;
}

}  // namespace

void summarise(SampleData& block) noexcept {
    block.minimum = 0.0f;
    block.maximum = 0.0f;
    block.has_finite = false;
    block.has_gaps = false;
    block.monotonic = true;

    const std::size_t n = block.values.size();
    const float* v = block.values.data();

    float lo = 0.0f, hi = 0.0f, previous = 0.0f;
    bool seen = false;

    for (std::size_t i = 0; i < n; ++i) {
        const float s = v[i];
        // One test covers NaN and both infinities: NaN fails every comparison,
        // and the infinities fail the range test.
        if (!(s > -std::numeric_limits<float>::infinity() &&
              s < std::numeric_limits<float>::infinity())) {
            block.has_gaps = true;
            continue;
        }
        if (!seen) {
            lo = hi = previous = s;
            seen = true;
            continue;
        }
        if (s < lo) lo = s;
        if (s > hi) hi = s;
        if (s < previous) block.monotonic = false;
        previous = s;
    }

    block.has_finite = seen;
    if (seen) {
        block.minimum = lo;
        block.maximum = hi;
    } else {
        block.monotonic = false;
    }
}

// ---------------------------------------------------------------- open

void Dataset::finish_open() {
    // Belt and braces: every later lookup indexes both with the same column
    // number, so a dialect that surprises the parser must not leave them
    // ragged.
    units_.resize(headers_.size());

    header_index_.reserve(headers_.size() * 2);
    for (std::size_t i = 0; i < headers_.size(); ++i)
        header_index_.emplace(headers_[i], static_cast<int>(i));

    for (auto& sim : sims_) sim->columns.resize(headers_.size());
}

bool Dataset::open(const std::string& path) {
    path_ = path;
    mod_time_ = MappedFile::mod_time(path);

    if (name_.empty()) {
        std::size_t slash = path.find_last_of("/\\");
        name_ = (slash == std::string::npos) ? path : path.substr(slash + 1);
    }

    headers_.clear();
    units_.clear();
    header_index_.clear();
    sims_.clear();
    values_per_line_ = 0;
    cache_.reset();

    // A matching sidecar already holds everything build_index() would go and
    // work out -- the headers, the units, every run's id and byte range -- so
    // when there is one the file is not scanned at all. This is the larger half
    // of what the cache buys: on a file with a thousand runs the index scan is
    // most of the time between choosing a file and seeing it.
    if (attach_cache()) {
        const std::size_t columns = cache_->column_count();
        const std::size_t runs = cache_->simulation_count();

        headers_.reserve(columns);
        units_.reserve(columns);
        for (std::size_t c = 0; c < columns; ++c) {
            headers_.push_back(cache_->column_name(c));
            units_.push_back(cache_->column_unit(c));
        }

        sims_.reserve(runs);
        for (std::size_t s = 0; s < runs; ++s) {
            auto sim = std::unique_ptr<Simulation>(new Simulation());
            sim->id = cache_->simulation_id(s);
            sim->display_name = sim->id;
            sim->unique_key = name_ + "::" + sim->id + "::" + std::to_string(s);
            sim->data_begin = static_cast<std::size_t>(cache_->data_begin(s));
            sim->data_end = static_cast<std::size_t>(cache_->data_end(s));
            sims_.push_back(std::move(sim));
        }

        values_per_line_ = cache_->values_per_line();
        indexed_size_ = static_cast<std::size_t>(fingerprint_.size);

        if (!headers_.empty()) {
            finish_open();
            return true;
        }
        // A cache with no columns should not have validated; if one somehow
        // did, fall through and read the file properly rather than presenting
        // an empty dataset.
        cache_.reset();
        headers_.clear();
        units_.clear();
        sims_.clear();
    }

    MappedFile file;
    if (!file.open(path)) return false;
    file.advise_sequential();
    indexed_size_ = file.size();

    build_index(file);

    if (headers_.empty()) return false;
    finish_open();
    return true;
}

bool Dataset::attach_cache() {
    cache_.reset();
    if (path_.empty() || !caching_enabled()) return false;

    fingerprint_ = fingerprint(path_);
    if (!fingerprint_.valid()) return false;

    // Nearest first: beside the source, then the per-user directory. The first
    // that opens *and* matches wins; one that fails to match is simply skipped,
    // because the other location may hold a current one.
    for (const std::string& candidate : cache_candidates(path_)) {
        auto file = std::unique_ptr<CacheFile>(new CacheFile());
        if (file->open(candidate, fingerprint_)) {
            cache_ = std::move(file);
            return true;
        }
    }
    return false;
}

bool Dataset::discard_cache() {
    cache_.reset();
    bool gone = true;
    for (const std::string& candidate : cache_candidates(path_))
        if (!remove_file(candidate)) gone = false;
    return gone;
}

void Dataset::build_index(const MappedFile& file) {
    const char* const base = file.begin();
    const char* p = base;
    const char* const end = file.end();

    bool reading_headers = false;
    Simulation* current = nullptr;

    const std::size_t param_len = sizeof(kParamMarker) - 1;
    const std::size_t table_len = sizeof(kTableMarker) - 1;

    while (p < end) {
        Line line = next_line(p, end);

        if (line.length > 0) {
            const bool is_marker = (line.begin[0] == '*');

            if (is_marker && line_contains(line.begin, line.length, kTableMarker, table_len)) {
                // Close the previous block at this marker.
                if (current) current->data_end = static_cast<std::size_t>(line.begin - base);

                auto sim = std::unique_ptr<Simulation>(new Simulation());

                const char* colon = static_cast<const char*>(
                    ::memchr(line.begin, ':', line.length));
                if (colon) {
                    const char* s = colon + 1;
                    const char* e = line.begin + line.length;
                    while (s < e && is_space(*s)) ++s;
                    while (e > s && is_space(*(e - 1))) --e;
                    if (e > s) sim->id.assign(s, static_cast<std::size_t>(e - s));
                }
                if (sim->id.empty())
                    sim->id = "SIM_" + std::to_string(sims_.size() + 1);

                sim->display_name = sim->id;
                sim->unique_key = name_ + "::" + sim->id + "::" + std::to_string(sims_.size());
                sim->data_begin = static_cast<std::size_t>(line.next - base);
                sim->data_end = file.size();

                current = sim.get();
                sims_.push_back(std::move(sim));
                reading_headers = false;

            } else if (is_marker &&
                       line_contains(line.begin, line.length, kParamMarker, param_len)) {
                if (headers_.empty()) reading_headers = true;
                if (current) {
                    current->data_end = static_cast<std::size_t>(line.begin - base);
                    current = nullptr;
                }

            } else if (is_marker) {
                // Any other '*' directive terminates whatever block is open.
                // Recording it here keeps the terminator test out of the hot
                // parse loop entirely.
                if (current) {
                    current->data_end = static_cast<std::size_t>(line.begin - base);
                    current = nullptr;
                }
                reading_headers = false;

            } else if (reading_headers) {
                const std::size_t added =
                    parse_header_line(line.begin, line.length, headers_, units_);
                // A line carrying no name after names have started ends the
                // section -- that is how a block with no explicit terminator
                // (which is most of them) is closed.
                if (added == 0 && !headers_.empty()) reading_headers = false;

            } else if (values_per_line_ == 0 && current && starts_data(line.begin, line.length)) {
                std::size_t count = 0;
                const char* q = line.begin;
                const char* stop = line.begin + line.length;
                while (q < stop) {
                    const char* next = nullptr;
                    parse_float(q, stop, &next);
                    if (next == q) break;
                    q = next;
                    ++count;
                    while (q < stop && is_space(*q)) ++q;
                }
                values_per_line_ = count;
            }
        } else if (reading_headers && !headers_.empty()) {
            reading_headers = false;
        }

        p = line.next;
    }
}

bool Dataset::reload() {
    std::string path = path_;
    headers_.clear();
    units_.clear();
    header_index_.clear();
    sims_.clear();
    return open(path);
}

void Dataset::invalidate_data() {
    for (auto& sim : sims_) {
        LockGuard guard(sim->lock);
        for (auto& col : sim->columns) {
            col.block.reset();
            col.loaded = false;
        }
        sim->computed.clear();
    }
}

void Dataset::clear_computed() {
    for (auto& sim : sims_) {
        LockGuard guard(sim->lock);
        sim->computed.clear();
    }
}

void Dataset::set_simulation_name(std::size_t index, const std::string& name) {
    if (index >= sims_.size() || name.empty()) return;
    sims_[index]->display_name = name;
}

int Dataset::column_index(const std::string& name) const {
    auto it = header_index_.find(name);
    return it == header_index_.end() ? -1 : it->second;
}

// ------------------------------------------------------------ parsing

void Dataset::parse_range(const char* base, std::size_t begin, std::size_t end_offset,
                          const std::vector<int>& cols,
                          std::vector<std::vector<float>>& out) const {
    const int num_cols = static_cast<int>(headers_.size());
    if (num_cols <= 0) return;

    // want[c] == slot index in `out`, or -1. A flat vector beats a hash lookup
    // per token by a wide margin; this is the innermost loop of the program.
    std::vector<int> want(static_cast<std::size_t>(num_cols), -1);
    for (std::size_t s = 0; s < cols.size(); ++s) {
        const int c = cols[s];
        if (c >= 0 && c < num_cols) want[static_cast<std::size_t>(c)] = static_cast<int>(s);
    }

    const char* p = base + begin;
    const char* const end = base + end_offset;
    if (p >= end) return;

    // Reserve from the observed bytes-per-line so the vectors do not grow
    // repeatedly through the parse.
    {
        const char* eol = static_cast<const char*>(
            ::memchr(p, '\n', static_cast<std::size_t>(end - p)));
        if (eol) {
            const std::size_t line_bytes = static_cast<std::size_t>(eol - p) + 1;
            if (line_bytes > 4) {
                const std::size_t span = static_cast<std::size_t>(end - p);
                std::size_t rows = span / line_bytes;
                if (values_per_line_ > 0 && values_per_line_ < static_cast<std::size_t>(num_cols))
                    rows = rows * values_per_line_ / static_cast<std::size_t>(num_cols);
                for (auto& v : out) v.reserve(rows + 16);
            }
        }
    }

    int col = 0;
    while (p < end) {
        Line line = next_line(p, end);
        p = line.next;

        if (line.length == 0) continue;
        if (!starts_data(line.begin, line.length)) continue;

        const char* q = line.begin;
        const char* const stop = line.begin + line.length;
        while (q < stop) {
            const int slot = want[static_cast<std::size_t>(col)];
            if (slot >= 0) {
                const char* next = nullptr;
                const float v = parse_float(q, stop, &next);
                if (next == q) break;  // malformed; abandon the line
                out[static_cast<std::size_t>(slot)].push_back(v);
                q = next;
            } else {
                // Skipping is much cheaper than parsing a value we discard, and
                // skip_token does it eight bytes at a time.
                q = skip_token(q, stop);
            }
            while (q < stop && is_space(*q)) ++q;
            if (++col >= num_cols) col = 0;
        }
    }
}

bool Dataset::load_from_cache(const std::vector<int>& columns,
                              const std::vector<int>& sim_indices) {
    // Checked before anything else is computed, so the parsing path below pays
    // nothing at all for this call when there is no cache.
    if (!cache_active()) return false;

    const int num_cols = static_cast<int>(headers_.size());
    // The sidecar was written from this file's own index, so these agree by
    // construction. Verified anyway: if they ever did not, indexing the cache
    // with this file's column numbers would read the wrong channel, and showing
    // the wrong channel is worse than being slow.
    if (cache_->column_count() != headers_.size() ||
        cache_->simulation_count() != sims_.size())
        return false;

    std::vector<int> sim_list;
    if (sim_indices.empty()) {
        sim_list.resize(sims_.size());
        for (std::size_t i = 0; i < sims_.size(); ++i) sim_list[i] = static_cast<int>(i);
    } else {
        sim_list.reserve(sim_indices.size());
        for (int i : sim_indices)
            if (i >= 0 && i < static_cast<int>(sims_.size())) sim_list.push_back(i);
    }
    if (sim_list.empty()) return true;  // nothing asked for is not a failure

    struct Fill {
        int sim_index;
        int column;
        SampleBlock block;
    };
    std::vector<Fill> fills;
    fills.reserve(sim_list.size() * columns.size());

    for (int si : sim_list) {
        Simulation& sim = *sims_[static_cast<std::size_t>(si)];
        LockGuard guard(sim.lock);
        if (sim.columns.size() < static_cast<std::size_t>(num_cols))
            sim.columns.resize(static_cast<std::size_t>(num_cols));
        for (int c : columns) {
            if (c < 0 || c >= num_cols) continue;
            if (!sim.columns[static_cast<std::size_t>(c)].loaded)
                fills.push_back(Fill{si, c, SampleBlock()});
        }
    }
    if (fills.empty()) return true;

    // A copy out of the mapping, spread over the pool. The samples for one
    // parameter across every run are contiguous on disk, so these reads are
    // sequential and the page faults overlap rather than queueing.
    ThreadPool::global().parallel_for(fills.size(), [&](std::size_t i) {
        Fill& fill = fills[i];
        const std::size_t column = static_cast<std::size_t>(fill.column);
        const std::size_t sim = static_cast<std::size_t>(fill.sim_index);

        auto block = std::make_shared<SampleData>();
        std::size_t count = 0;
        const float* values = cache_->values(column, sim, &count);
        if (values && count) block->values.assign(values, values + count);

        // The summary was computed when the sidecar was written, so the pass
        // that summarise() would make here does not happen at all. On a tile
        // showing a thousand curves that pass was the whole of the autoscale
        // cost.
        if (const CacheBlock* meta = cache_->block(column, sim)) {
            block->minimum = meta->minimum;
            block->maximum = meta->maximum;
            block->has_finite = (meta->flags & kCacheHasFinite) != 0;
            block->has_gaps = (meta->flags & kCacheHasGaps) != 0;
            block->monotonic = (meta->flags & kCacheMonotonic) != 0;
        }
        fill.block = std::move(block);
    });

    for (Fill& fill : fills) {
        Simulation& sim = *sims_[static_cast<std::size_t>(fill.sim_index)];
        LockGuard guard(sim.lock);
        Column& dest = sim.columns[static_cast<std::size_t>(fill.column)];
        dest.block = std::move(fill.block);
        dest.loaded = true;
        dest.last_used = next_use_stamp();
    }
    return true;
}

void Dataset::load_columns(const std::vector<int>& columns,
                           const std::vector<int>& sim_indices) {
    const int num_cols = static_cast<int>(headers_.size());
    if (num_cols <= 0 || columns.empty() || sims_.empty()) return;

    if (load_from_cache(columns, sim_indices)) return;

    std::vector<int> sim_list;
    if (sim_indices.empty()) {
        sim_list.resize(sims_.size());
        for (std::size_t i = 0; i < sims_.size(); ++i) sim_list[i] = static_cast<int>(i);
    } else {
        sim_list.reserve(sim_indices.size());
        for (int i : sim_indices)
            if (i >= 0 && i < static_cast<int>(sims_.size())) sim_list.push_back(i);
    }
    if (sim_list.empty()) return;

    // Work out what is actually missing before touching the disk.
    struct Pending {
        int sim_index;
        std::vector<int> cols;
    };
    std::vector<Pending> pending;
    pending.reserve(sim_list.size());

    for (int si : sim_list) {
        Simulation& sim = *sims_[static_cast<std::size_t>(si)];
        LockGuard guard(sim.lock);
        if (sim.columns.size() < static_cast<std::size_t>(num_cols))
            sim.columns.resize(static_cast<std::size_t>(num_cols));

        std::vector<int> missing;
        for (int c : columns) {
            if (c < 0 || c >= num_cols) continue;
            if (!sim.columns[static_cast<std::size_t>(c)].loaded) missing.push_back(c);
        }
        if (!missing.empty()) pending.push_back(Pending{si, std::move(missing)});
    }
    if (pending.empty()) return;

    MappedFile file;
    if (!file.open(path_)) {
        // Mark them loaded-but-empty so a vanished file cannot make the UI retry
        // this load on every frame. The tempting alternative -- pushing a fake
        // 0.0 sample -- hides the failure instead of recording it.
        for (const auto& item : pending) {
            Simulation& sim = *sims_[static_cast<std::size_t>(item.sim_index)];
            LockGuard guard(sim.lock);
            for (int c : item.cols) sim.columns[static_cast<std::size_t>(c)].loaded = true;
        }
        return;
    }
    const char* const base = file.begin();

    // The index was built against a file of `indexed_size_` bytes; this mapping
    // is of the file as it is now. A solver that truncated and rewrote its
    // output between the two would otherwise have us parsing to an offset past
    // the end of the mapping.
    const std::size_t limit = file.size();
    const auto clamp = [limit](std::size_t offset) {
        return offset > limit ? limit : offset;
    };

    ThreadPool& pool = ThreadPool::global();

    // Split each block into chunks when there are fewer simulations than
    // threads, so a file holding one big table still uses the whole machine.
    // Safe only when every record occupies exactly one line; otherwise the
    // column counter would not be zero at a chunk boundary.
    const bool one_record_per_line =
        (values_per_line_ == static_cast<std::size_t>(num_cols));

    struct Unit {
        std::size_t pending_index;
        std::size_t chunk_index;
        std::size_t begin;
        std::size_t end;
    };
    std::vector<Unit> units;
    // results[pending][chunk][slot]
    std::vector<std::vector<std::vector<std::vector<float>>>> results(pending.size());

    for (std::size_t pi = 0; pi < pending.size(); ++pi) {
        const Simulation& sim = *sims_[static_cast<std::size_t>(pending[pi].sim_index)];
        const std::size_t block_begin = clamp(sim.data_begin);
        const std::size_t block_end = clamp(sim.data_end);
        const std::size_t span = block_end > block_begin ? block_end - block_begin : 0;

        std::size_t chunks = 1;
        if (one_record_per_line && pending.size() < pool.size() + 1 && span > kMinChunkBytes) {
            chunks = span / kMinChunkBytes;
            const std::size_t cap = pool.size() + 1;
            if (chunks > cap) chunks = cap;
            if (chunks == 0) chunks = 1;
        }

        results[pi].resize(chunks);
        for (auto& chunk : results[pi])
            chunk.resize(pending[pi].cols.size());

        std::size_t cursor = block_begin;
        for (std::size_t c = 0; c < chunks; ++c) {
            std::size_t stop = (c + 1 == chunks) ? block_end
                                                 : block_begin + span * (c + 1) / chunks;
            if (stop < cursor) stop = cursor;
            if (c + 1 != chunks && stop < block_end) {
                // Advance to the next line boundary so no record is split.
                const char* nl = static_cast<const char*>(
                    ::memchr(base + stop, '\n',
                             static_cast<std::size_t>(block_end - stop)));
                stop = nl ? static_cast<std::size_t>(nl - base) + 1 : block_end;
            }
            units.push_back(Unit{pi, c, cursor, stop});
            cursor = stop;
            if (cursor >= block_end) {
                // Remaining chunks would be empty; keep them so indices line up.
                for (std::size_t rest = c + 1; rest < chunks; ++rest)
                    units.push_back(Unit{pi, rest, block_end, block_end});
                break;
            }
        }
    }

    pool.parallel_for(units.size(), [&](std::size_t ui) {
        const Unit& unit = units[ui];
        const Pending& item = pending[unit.pending_index];
        parse_range(base, unit.begin, unit.end, item.cols,
                    results[unit.pending_index][unit.chunk_index]);
    });

    // Stitch chunks together into their final blocks. Building them outside the
    // simulation locks lets the summary pass below run in parallel over the lot.
    struct Built {
        std::size_t pending_index;
        std::size_t slot;
        SampleBlock block;
    };
    std::vector<Built> built;
    for (std::size_t pi = 0; pi < pending.size(); ++pi) {
        built.reserve(built.size() + pending[pi].cols.size());
        auto& chunks = results[pi];

        for (std::size_t slot = 0; slot < pending[pi].cols.size(); ++slot) {
            SampleBlock block = std::make_shared<SampleData>();

            if (chunks.size() == 1) {
                block->values = std::move(chunks[0][slot]);
            } else {
                std::size_t total = 0;
                for (auto& chunk : chunks) total += chunk[slot].size();
                block->values.reserve(total);
                for (auto& chunk : chunks) {
                    block->values.insert(block->values.end(), chunk[slot].begin(),
                                         chunk[slot].end());
                    chunk[slot].clear();
                    chunk[slot].shrink_to_fit();
                }
            }

            // Give the slack back only when it is worth a reallocation. The
            // reserve above is an estimate from the observed line length, so it
            // usually overshoots by a few percent -- and an unconditional
            // shrink_to_fit copied every column a second time to reclaim it.
            const std::size_t capacity = block->values.capacity();
            const std::size_t size = block->values.size();
            if (capacity > size + (size >> 3) + 64) block->values.shrink_to_fit();

            built.push_back(Built{pi, slot, std::move(block)});
        }
    }

    // Min, max, gaps and monotonicity, computed while the samples are still hot
    // in cache and spread over the pool. Every one of these answers a question
    // the draw path would otherwise ask by rescanning the column.
    pool.parallel_for(built.size(), [&](std::size_t bi) { summarise(*built[bi].block); });

    for (auto& item : built) {
        const Pending& p = pending[item.pending_index];
        Simulation& sim = *sims_[static_cast<std::size_t>(p.sim_index)];
        LockGuard guard(sim.lock);
        Column& dest = sim.columns[static_cast<std::size_t>(p.cols[item.slot])];
        dest.block = std::move(item.block);
        dest.loaded = true;
        dest.last_used = next_use_stamp();
    }
}

const float* Dataset::column_data(std::size_t sim_index, int column,
                                  std::size_t* out_count, SampleBlock* out_block) {
    if (out_count) *out_count = 0;
    if (out_block) out_block->reset();

    SampleBlock block = column_block(sim_index, column);
    if (!block || block->values.empty()) return nullptr;
    if (out_count) *out_count = block->values.size();
    const float* data = block->values.data();
    if (out_block) *out_block = std::move(block);
    return data;
}

SampleBlock Dataset::column_block(std::size_t sim_index, int column) {
    if (sim_index >= sims_.size() || column < 0 ||
        column >= static_cast<int>(headers_.size()))
        return SampleBlock();

    Simulation& sim = *sims_[sim_index];
    {
        LockGuard guard(sim.lock);
        if (sim.columns.size() > static_cast<std::size_t>(column) &&
            sim.columns[static_cast<std::size_t>(column)].loaded) {
            Column& col = sim.columns[static_cast<std::size_t>(column)];
            col.last_used = next_use_stamp();
            return col.block;
        }
    }

    load_columns({column}, {static_cast<int>(sim_index)});

    LockGuard guard(sim.lock);
    return sim.columns[static_cast<std::size_t>(column)].block;
}

namespace {

// The first record's value for one column, read straight out of an open
// mapping. Walks forward only until that column of the first record -- which
// for a wrapped record may be a line or two in -- and reads nothing beyond.
float first_value_in(const char* base, std::size_t size, const Simulation& sim,
                     int column, int num_cols) noexcept {
    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    const std::size_t begin = std::min(sim.data_begin, size);
    const std::size_t stop_offset = std::min(sim.data_end, size);
    if (begin >= stop_offset) return nan_value;

    const char* p = base + begin;
    const char* const end = base + stop_offset;

    int col_index = 0;
    while (p < end) {
        Line line = next_line(p, end);
        p = line.next;
        if (line.length == 0 || !starts_data(line.begin, line.length)) continue;

        const char* q = line.begin;
        const char* const stop = line.begin + line.length;
        while (q < stop) {
            const char* next = nullptr;
            const float value = parse_float(q, stop, &next);
            if (next == q) break;
            if (col_index == column) return value;
            q = next;
            while (q < stop && is_space(*q)) ++q;
            if (++col_index >= num_cols) return nan_value;  // past it: not present
        }
    }
    return nan_value;
}

}  // namespace

float Dataset::first_value(std::size_t sim_index, int column) const {
    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    if (sim_index >= sims_.size() || column < 0 ||
        column >= static_cast<int>(headers_.size()))
        return nan_value;

    const Simulation& sim = *sims_[sim_index];

    // Already resident: no need to touch the file at all.
    {
        LockGuard guard(sim.lock);
        if (sim.columns.size() > static_cast<std::size_t>(column)) {
            const Column& col = sim.columns[static_cast<std::size_t>(column)];
            if (col.loaded) return col.empty() ? nan_value : col.data()[0];
        }
    }

    // Straight out of the sidecar: one float, no mapping of the source and no
    // parse. An ISO scan is exactly this call per run.
    if (cache_active() && cache_->column_count() == headers_.size() &&
        cache_->simulation_count() == sims_.size()) {
        std::size_t count = 0;
        const float* values = cache_->values(static_cast<std::size_t>(column),
                                             sim_index, &count);
        return (values && count) ? values[0] : nan_value;
    }

    MappedFile file;
    if (!file.open(path_)) return nan_value;
    return first_value_in(file.begin(), file.size(), sim, column,
                          static_cast<int>(headers_.size()));
}

std::vector<float> Dataset::first_values(int column) const {
    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> out(sims_.size(), nan_value);
    if (column < 0 || column >= static_cast<int>(headers_.size()) || sims_.empty())
        return out;

    // Which runs still need the file read for them, and which are resident.
    std::vector<std::size_t> wanted;
    wanted.reserve(sims_.size());
    for (std::size_t s = 0; s < sims_.size(); ++s) {
        const Simulation& sim = *sims_[s];
        LockGuard guard(sim.lock);
        if (sim.columns.size() > static_cast<std::size_t>(column) &&
            sim.columns[static_cast<std::size_t>(column)].loaded) {
            const Column& col = sim.columns[static_cast<std::size_t>(column)];
            out[s] = col.empty() ? nan_value : col.data()[0];
        } else {
            wanted.push_back(s);
        }
    }
    if (wanted.empty()) return out;

    // With a sidecar every one of these is the first float of a known block, so
    // the whole scan is a walk down one contiguous region of the mapping. No
    // pool, because the work per run is a single load.
    if (cache_active() && cache_->column_count() == headers_.size() &&
        cache_->simulation_count() == sims_.size()) {
        for (std::size_t s : wanted) {
            std::size_t count = 0;
            const float* values =
                cache_->values(static_cast<std::size_t>(column), s, &count);
            if (values && count) out[s] = values[0];
        }
        return out;
    }

    // One mapping for the whole scan, and the per-run reads spread over the
    // pool: each touches a few hundred bytes at a different offset, so they do
    // not contend and the page faults overlap.
    MappedFile file;
    if (!file.open(path_)) return out;

    const char* const base = file.begin();
    const std::size_t size = file.size();
    const int num_cols = static_cast<int>(headers_.size());

    ThreadPool::global().parallel_for(wanted.size(), [&](std::size_t i) {
        const std::size_t s = wanted[i];
        out[s] = first_value_in(base, size, *sims_[s], column, num_cols);
    });
    return out;
}

void Dataset::touch(std::size_t sim_index, int column) const {
    if (sim_index >= sims_.size() || column < 0) return;
    const Simulation& sim = *sims_[sim_index];
    LockGuard guard(sim.lock);
    if (sim.columns.size() > static_cast<std::size_t>(column))
        sim.columns[static_cast<std::size_t>(column)].last_used = next_use_stamp();
}

std::size_t Dataset::resident_bytes() const {
    std::size_t total = 0;
    for (const auto& sim : sims_) {
        LockGuard guard(sim->lock);
        for (const auto& col : sim->columns) total += col.bytes();
        for (const auto& item : sim->computed)
            if (item.second) total += item.second->values.capacity() * sizeof(float);
    }
    return total;
}

std::size_t Dataset::evict_until(std::size_t keep_bytes) {
    // Gather every resident column with its stamp, oldest first, and drop until
    // the budget is met. Computed results are dropped first: they are cheap to
    // recreate from columns that are themselves already resident.
    struct Victim {
        std::uint64_t stamp;
        std::size_t sim;
        std::size_t column;
        std::size_t bytes;
    };

    std::size_t total = resident_bytes();
    if (total <= keep_bytes) return total;

    for (auto& sim : sims_) {
        LockGuard guard(sim->lock);
        if (!sim->computed.empty()) {
            for (const auto& item : sim->computed)
                if (item.second)
                    total -= std::min(total, item.second->values.capacity() * sizeof(float));
            sim->computed.clear();
        }
    }
    if (total <= keep_bytes) return total;

    std::vector<Victim> victims;
    for (std::size_t si = 0; si < sims_.size(); ++si) {
        Simulation& sim = *sims_[si];
        LockGuard guard(sim.lock);
        for (std::size_t ci = 0; ci < sim.columns.size(); ++ci) {
            const Column& col = sim.columns[ci];
            if (!col.empty()) victims.push_back(Victim{col.last_used, si, ci, col.bytes()});
        }
    }

    std::sort(victims.begin(), victims.end(),
              [](const Victim& a, const Victim& b) { return a.stamp < b.stamp; });

    for (const Victim& victim : victims) {
        if (total <= keep_bytes) break;
        Simulation& sim = *sims_[victim.sim];
        LockGuard guard(sim.lock);
        Column& col = sim.columns[victim.column];
        if (col.empty()) continue;
        // Dropping the cache's reference. If a view of this block is on screen
        // the memory stays put until the view goes -- correctness first, budget
        // second -- but it stops counting here either way, and the column is
        // re-read on demand.
        total -= std::min(total, col.bytes());
        col.block.reset();
        col.loaded = false;
    }
    return total;
}

bool Dataset::write_cache(std::string* out_path) {
    if (out_path) out_path->clear();
    if (headers_.empty() || sims_.empty() || path_.empty()) return false;
    if (!caching_enabled()) return false;

    // Fingerprint before reading anything, and check it again at the end. A
    // file rewritten by the solver while we were parsing it would otherwise be
    // cached as a mixture of before and after -- the one way this could produce
    // a sidecar that matches its source and still holds numbers that were never
    // in it together.
    const SourceFingerprint before = fingerprint(path_);
    if (!before.valid()) return false;

    // Detach before anything else, for two reasons. Reading through the sidecar
    // we are about to replace would make this a copy of itself rather than a
    // fresh parse of the file. And on Windows a mapped file cannot be renamed
    // over or deleted -- MoveFileEx fails with a sharing violation -- so the
    // mapping has to be gone before finish() puts the new one in its place.
    // Nothing is lost by dropping it: on any failure below, attach_cache()
    // picks up whatever sidecar is still on disk.
    cache_.reset();

    const std::size_t column_count = headers_.size();
    const std::size_t sim_count = sims_.size();

    std::vector<std::string> ids;
    std::vector<std::uint64_t> begins, ends;
    ids.reserve(sim_count);
    begins.reserve(sim_count);
    ends.reserve(sim_count);
    for (const auto& sim : sims_) {
        ids.push_back(sim->id);
        begins.push_back(static_cast<std::uint64_t>(sim->data_begin));
        ends.push_back(static_cast<std::uint64_t>(sim->data_end));
    }

    std::vector<std::string> candidates = cache_candidates(path_);
    if (candidates.empty()) {
        attach_cache();
        return false;
    }

    // Every run of one column, then the next column. Peak memory is therefore
    // one parameter across every run -- a few megabytes on the files this is
    // for -- rather than the whole file, which is the point of writing in this
    // order rather than run by run.
    std::vector<int> all_sims(sim_count);
    for (std::size_t s = 0; s < sim_count; ++s) all_sims[s] = static_cast<int>(s);

    std::string written_to;

    for (const std::string& candidate : candidates) {
        CacheWriter writer;
        if (!writer.begin(candidate, before, headers_, units_, ids, begins, ends,
                          values_per_line_))
            continue;  // unwritable location -- a read-only share, typically

        bool ok = true;
        for (std::size_t c = 0; c < column_count && ok; ++c) {
            // Which runs already had this column, so the ones we pull in only
            // to write them out again can be released afterwards and the user's
            // warm cache is left as we found it.
            std::vector<char> was_resident(sim_count, 0);
            for (std::size_t s = 0; s < sim_count; ++s) {
                LockGuard guard(sims_[s]->lock);
                if (sims_[s]->columns.size() > c && sims_[s]->columns[c].loaded)
                    was_resident[s] = 1;
            }

            load_columns({static_cast<int>(c)}, all_sims);

            std::vector<CacheSource> runs(sim_count);
            std::vector<SampleBlock> hold(sim_count);
            for (std::size_t s = 0; s < sim_count; ++s) {
                LockGuard guard(sims_[s]->lock);
                if (sims_[s]->columns.size() <= c) continue;
                // Keep a reference while writing so an eviction on another
                // thread cannot free the samples out from under fwrite.
                hold[s] = sims_[s]->columns[c].block;
                if (hold[s] && !hold[s]->values.empty()) {
                    runs[s].values = hold[s]->values.data();
                    runs[s].count = hold[s]->values.size();
                }
            }

            ok = writer.write_column(c, runs);

            for (std::size_t s = 0; s < sim_count; ++s) {
                if (was_resident[s]) continue;
                LockGuard guard(sims_[s]->lock);
                if (sims_[s]->columns.size() <= c) continue;
                sims_[s]->columns[c].block.reset();
                sims_[s]->columns[c].loaded = false;
            }
        }

        if (!ok) {
            writer.abort();
            continue;
        }
        if (!writer.finish()) continue;

        written_to = candidate;
        break;
    }

    if (written_to.empty()) {
        attach_cache();
        return false;
    }

    // The source has to be the same file it was when the parse began. If it
    // changed, the sidecar describes a state that never existed; delete it
    // rather than leave it to be validated against a fingerprint it happens to
    // carry.
    const SourceFingerprint after = fingerprint(path_);
    if (after != before) {
        remove_file(written_to);
        attach_cache();
        return false;
    }

    if (!attach_cache()) return false;
    if (out_path) *out_path = cache_->path();
    return true;
}

std::size_t Dataset::loaded_row_count(std::size_t sim_index) const {
    if (sim_index >= sims_.size()) return 0;
    const Simulation& sim = *sims_[sim_index];
    LockGuard guard(sim.lock);
    std::size_t best = 0;
    for (const auto& col : sim.columns)
        if (col.size() > best) best = col.size();
    return best;
}

}  // namespace aoe
