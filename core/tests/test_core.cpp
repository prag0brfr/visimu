// Author: Paulo R. A. Gilz
// Self-contained C++ tests for the core. No framework needed, so this builds
// with nothing but the compiler that is already on the target machine.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "aoe/cache.hpp"
#include "aoe/decimate.hpp"
#include "aoe/thread_pool.hpp"
#include "aoe/fast_parse.hpp"
#include "aoe/formula.hpp"
#include "aoe/iso.hpp"
#include "aoe/session.hpp"
#include "aoe/aoe_c_api.h"
#include "sample_file.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const char* expression, const char* file, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("  FAIL %s:%d  %s\n", file, line, expression);
    }
}

void check_near(double a, double b, double tolerance, const char* expression,
                const char* file, int line) {
    ++g_checks;
    if (!(std::fabs(a - b) <= tolerance)) {
        ++g_failures;
        std::printf("  FAIL %s:%d  %s  (%.9g vs %.9g)\n", file, line, expression, a, b);
    }
}

#define CHECK(expr) check((expr), #expr, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) check_near((a), (b), (tol), #a " ~= " #b, __FILE__, __LINE__)

void section(const char* name) { std::printf("%s\n", name); }

// Float comparison for the places where a CHECK_NEAR would read worse than the
// expression it is testing.
inline bool close(float a, float b, float tolerance = 1e-6f) {
    return std::fabs(a - b) <= tolerance;
}

float parse_one(const char* text) {
    const char* next = nullptr;
    return aoe::parse_float(text, text + std::strlen(text), &next);
}

// ------------------------------------------------------------ parser

void test_float_parser() {
    section("float parser");

    CHECK_NEAR(parse_one("0"), 0.0, 0.0);
    CHECK_NEAR(parse_one("1"), 1.0, 0.0);
    CHECK_NEAR(parse_one("-42"), -42.0, 0.0);
    CHECK_NEAR(parse_one("3.25"), 3.25, 0.0);
    CHECK_NEAR(parse_one("-0.125"), -0.125, 0.0);
    CHECK_NEAR(parse_one("  7.5  "), 7.5, 0.0);

    // Exponent forms, including the two Fortran spellings a naive parser
    // cannot read.
    CHECK_NEAR(parse_one("1.5e3"), 1500.0, 0.0);
    CHECK_NEAR(parse_one("1.5E3"), 1500.0, 0.0);
    CHECK_NEAR(parse_one("1.5e-3"), 0.0015, 1e-9);
    CHECK_NEAR(parse_one("1.5D+03"), 1500.0, 0.0);
    CHECK_NEAR(parse_one("1.5+03"), 1500.0, 0.0);
    CHECK_NEAR(parse_one("1.5-03"), 0.0015, 1e-9);

    // Precision: accumulating fractional digits one at a time drifts. This
    // value must round-trip through float exactly.
    CHECK_NEAR(parse_one("0.123456789"), static_cast<double>(0.123456789f), 1e-7);
    CHECK_NEAR(parse_one("123456.789"), static_cast<double>(123456.789f), 0.01);

    // A trailing sign is not an exponent.
    {
        const char* text = "2.5+";
        const char* next = nullptr;
        const float v = aoe::parse_float(text, text + 4, &next);
        CHECK_NEAR(v, 2.5, 0.0);
    }

    // Junk becomes NaN, not a silent zero, and always consumes the token so a
    // caller cannot spin on it.
    CHECK(std::isnan(parse_one("NaN")));
    CHECK(std::isnan(parse_one("*******")));
    {
        const char* text = "abc 1.0";
        const char* next = nullptr;
        aoe::parse_float(text, text + 7, &next);
        CHECK(next > text);
    }

    // Consecutive values advance correctly.
    {
        const char* text = "1.0 2.0   3.0";
        const char* end = text + std::strlen(text);
        const char* p = text;
        float values[3];
        for (int i = 0; i < 3; ++i) {
            const char* next = nullptr;
            values[i] = aoe::parse_float(p, end, &next);
            p = next;
        }
        CHECK_NEAR(values[0], 1.0, 0.0);
        CHECK_NEAR(values[1], 2.0, 0.0);
        CHECK_NEAR(values[2], 3.0, 0.0);
    }
}

void test_skip_token() {
    section("token skipping");

    // The word-at-a-time path and the byte-at-a-time tail must agree, including
    // at every alignment and right at the buffer end.
    const char* samples[] = {
        "1.0 2.0",
        "-123.456789\t9",
        "1.0",
        "0.000001 0.2",
        "12345678901234567890 3",
        "  leading",
    };
    for (const char* text : samples) {
        const std::size_t len = std::strlen(text);
        for (std::size_t offset = 0; offset < len; ++offset) {
            const char* begin = text + offset;
            const char* end = text + len;

            const char* fast = aoe::skip_token(begin, end);
            const char* slow = begin;
            while (slow < end && !aoe::is_space(*slow)) ++slow;

            CHECK(fast == slow);
        }
    }

    // A token running to the very end must not read past it.
    {
        const char text[] = "123456789012345";
        const char* end = text + 15;
        CHECK(aoe::skip_token(text, end) == end);
    }
    // Empty range.
    {
        const char text[] = "x";
        CHECK(aoe::skip_token(text, text) == text);
    }
}

void test_stats() {
    section("statistics");

    std::vector<float> x(1000), y(1000);
    for (std::size_t i = 0; i < 1000; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i) * 2.0f;
    }

    aoe::SeriesStats stats;
    CHECK(aoe::series_stats(x.data(), y.data(), 1000, 0.0f, 0.0f, &stats));
    CHECK_NEAR(stats.minimum, 0.0, 1e-6);
    CHECK_NEAR(stats.maximum, 1998.0, 1e-6);
    CHECK_NEAR(stats.mean, 999.0, 1e-6);
    CHECK(stats.count == 1000);
    CHECK_NEAR(stats.first, 0.0, 1e-6);
    CHECK_NEAR(stats.last, 1998.0, 1e-6);

    // Restricted to a window.
    CHECK(aoe::series_stats(x.data(), y.data(), 1000, 100.0f, 200.0f, &stats));
    CHECK(stats.count > 90 && stats.count < 110);
    CHECK(stats.minimum >= 200.0);
    CHECK(stats.maximum <= 402.0);

    // NaNs are skipped, not propagated into the mean.
    std::vector<float> holed = y;
    holed[10] = std::numeric_limits<float>::quiet_NaN();
    CHECK(aoe::series_stats(x.data(), holed.data(), 1000, 0.0f, 0.0f, &stats));
    CHECK(stats.count == 999);
    CHECK(!std::isnan(stats.mean));

    // Standard deviation of a constant series is exactly zero.
    std::vector<float> flat(500, 7.0f);
    CHECK(aoe::series_stats(nullptr, flat.data(), 500, 0.0f, 0.0f, &stats));
    CHECK_NEAR(stats.stddev, 0.0, 1e-9);
    CHECK_NEAR(stats.mean, 7.0, 1e-9);

    // All-NaN reports failure rather than zeros.
    std::vector<float> all_nan(10, std::numeric_limits<float>::quiet_NaN());
    CHECK(!aoe::series_stats(nullptr, all_nan.data(), 10, 0.0f, 0.0f, &stats));
}

// ----------------------------------------------------------- dataset

void test_dataset(const std::string& path) {
    section("dataset");

    aoe::Session session;
    const int file = session.add_file(path);
    CHECK(file == 0);
    if (file < 0) return;

    const aoe::Dataset* dataset = session.file(file);
    CHECK(dataset != nullptr);
    CHECK(dataset->headers().size() == 4);
    CHECK(dataset->headers()[0] == "TIME");
    CHECK(dataset->headers()[3] == "NZ");
    CHECK(dataset->simulation_count() == 3);
    CHECK(dataset->simulation(0).id == "SIM_001");
    CHECK(dataset->simulation(2).id == "SIM_003");

    // The trailing *FIN must not be swallowed into the last block.
    std::size_t count = 0;
    const float* time = session.channel(file, 2, "TIME", &count);
    CHECK(time != nullptr);
    CHECK(count == 500);

    const float* mach = session.channel(file, 1, "MACH", &count);
    CHECK(mach != nullptr);
    CHECK_NEAR(mach[0], 0.65, 1e-6);

    CHECK(session.channel(file, 0, "DOES_NOT_EXIST", &count) == nullptr);
    CHECK(session.channel(file, 99, "TIME", &count) == nullptr);
}

void test_wrapped_records(const std::string& path) {
    section("wrapped records");

    aoe::Session session;
    const int file = session.add_file(path);
    CHECK(file >= 0);
    if (file < 0) return;

    // Two physical lines per logical record: the column counter has to carry
    // across the newline or every channel comes out interleaved.
    std::size_t count = 0;
    const float* nz = session.channel(file, 0, "NZ", &count);
    CHECK(nz != nullptr);
    CHECK(count == 200);
    if (nz && count > 0) CHECK_NEAR(nz[0], 1.25, 1e-5);

    const float* time = session.channel(file, 0, "TIME", &count);
    CHECK(time != nullptr);
    if (time && count > 1) CHECK_NEAR(time[1], 0.02, 1e-6);
}

// ----------------------------------------------------------- formula

void test_formula(const std::string& path) {
    section("formula");

    aoe::Session session;
    const int file = session.add_file(path);
    CHECK(file >= 0);
    if (file < 0) return;

    std::string error;

    CHECK(session.set_computed("DOUBLE_NZ", "NZ * 2", &error));
    CHECK(error.empty());

    std::size_t count = 0;
    const float* nz = session.channel(file, 0, "NZ", &count);
    const std::size_t nz_count = count;
    std::vector<float> nz_copy(nz, nz + nz_count);

    const float* doubled = session.channel(file, 0, "DOUBLE_NZ", &count);
    CHECK(doubled != nullptr);
    CHECK(count == nz_count);
    if (doubled) CHECK_NEAR(doubled[10], nz_copy[10] * 2.0, 1e-5);

    // Operator precedence and associativity.
    CHECK(session.set_computed("PREC", "2 + 3 * 4", &error));
    const float* prec = session.channel(file, 0, "PREC", &count);
    CHECK(prec && count > 0);
    if (prec) CHECK_NEAR(prec[0], 14.0, 1e-6);

    CHECK(session.set_computed("POW", "2 ^ 3 ^ 2", &error));  // right-associative
    const float* pw = session.channel(file, 0, "POW", &count);
    if (pw) CHECK_NEAR(pw[0], 512.0, 1e-3);

    CHECK(session.set_computed("UNARY", "-2 ^ 2", &error));  // -(2^2)
    const float* un = session.channel(file, 0, "UNARY", &count);
    if (un) CHECK_NEAR(un[0], -4.0, 1e-6);

    // Functions.
    CHECK(session.set_computed("FN", "max(abs(-3), sqrt(4))", &error));
    const float* fn = session.channel(file, 0, "FN", &count);
    if (fn) CHECK_NEAR(fn[0], 3.0, 1e-6);

    CHECK(session.set_computed("COND", "if(1 > 0, 10, 20)", &error));
    const float* cond = session.channel(file, 0, "COND", &count);
    if (cond) CHECK_NEAR(cond[0], 10.0, 1e-6);

    // Constant lookup: MACH is flat, so MACH[TIME=1.0] equals MACH.
    CHECK(session.set_computed("MREF", "MACH[TIME=1.0]", &error));
    const float* mref = session.channel(file, 0, "MREF", &count);
    CHECK(mref != nullptr);
    if (mref) CHECK_NEAR(mref[0], 0.60, 1e-5);

    // Division by zero is NaN, not infinity.
    CHECK(session.set_computed("DIVZERO", "1 / 0", &error));
    const float* dz = session.channel(file, 0, "DIVZERO", &count);
    if (dz) CHECK(std::isnan(dz[0]));

    // Errors.
    CHECK(!session.set_computed("BAD1", "NZ * ", &error));
    CHECK(!error.empty());
    CHECK(!session.set_computed("BAD2", "UNKNOWN_CHANNEL + 1", &error));
    CHECK(!session.set_computed("BAD3", "nosuchfn(1)", &error));
    CHECK(!session.set_computed("BAD4", "(1 + 2", &error));

    // Self-reference and mutual recursion must be refused, not recursed into.
    CHECK(!session.set_computed("CYCLE", "CYCLE + 1", &error));
    CHECK(session.set_computed("A_CH", "NZ + 1", &error));
    CHECK(session.set_computed("B_CH", "A_CH * 2", &error));
    CHECK(!session.set_computed("A_CH", "B_CH + 1", &error));

    // Removing a computed parameter removes its dependants too.
    const std::vector<std::string> removed = session.remove_computed("A_CH");
    CHECK(removed.size() == 2);
    CHECK(!session.is_computed("B_CH"));
}

// --------------------------------------------------------------- iso

void test_iso() {
    section("iso grouping");

    std::vector<float> values{0.60f, 0.65f, 0.70f, 0.75f, 0.80f};
    std::vector<aoe::IsoGroup> groups = aoe::iso_auto_group(values, 2);
    CHECK(groups.size() == 2);
    CHECK_NEAR(groups[0].min_bound, 0.60, 1e-6);
    CHECK_NEAR(groups[1].max_bound, 0.80, 1e-6);

    // Both extremes must land in a group; the maximum is the one that a
    // half-open final interval would silently drop.
    CHECK(aoe::iso_group_index(0.60f, groups) == 0);
    CHECK(aoe::iso_group_index(0.80f, groups) == 1);
    CHECK(aoe::iso_group_index(0.70f, groups) >= 0);

    groups[0].active = false;
    CHECK(!aoe::iso_value_active(0.60f, groups));
    CHECK(aoe::iso_value_active(0.80f, groups));

    // A degenerate sweep gives one group, not N empty ones.
    std::vector<float> flat{1.0f, 1.0f, 1.0f};
    CHECK(aoe::iso_auto_group(flat, 4).size() == 1);

    CHECK(aoe::iso_format_value(0.75f) == "0.75");
    CHECK(aoe::iso_format_value(3.0f) == "3");
    CHECK(aoe::iso_format_value(-0.0f) == "0");
}

// --------------------------------------------------------- decimation

void test_decimation() {
    section("decimation");

    const std::size_t n = 100000;
    std::vector<float> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i) * 0.001f;
        y[i] = std::sin(x[i]);
    }
    // A single narrow spike: the whole point of min/max over striding.
    y[50000] = 999.0f;

    std::vector<float> ox(4096), oy(4096);
    const std::size_t written =
        aoe::decimate_minmax(x.data(), y.data(), n, 600, ox.data(), oy.data());

    CHECK(written > 0);
    CHECK(written <= 1200);

    bool spike_kept = false;
    for (std::size_t i = 0; i < written; ++i)
        if (oy[i] > 998.0f) spike_kept = true;
    CHECK(spike_kept);

    // x must come out non-decreasing or the polyline doubles back on itself.
    bool sorted = true;
    for (std::size_t i = 1; i < written; ++i)
        if (ox[i] < ox[i - 1]) sorted = false;
    CHECK(sorted);

    // Short series pass through untouched.
    const std::size_t small_written =
        aoe::decimate_minmax(x.data(), y.data(), 50, 600, ox.data(), oy.data());
    CHECK(small_written == 50);

    // Range selection.
    const std::size_t ranged = aoe::decimate_minmax_range(
        x.data(), y.data(), n, 10.0f, 20.0f, 300, ox.data(), oy.data());
    CHECK(ranged > 0);
    for (std::size_t i = 0; i < ranged; ++i) CHECK(ox[i] >= 9.0f && ox[i] <= 21.0f);

    // Bounds and nearest-sample search.
    float lo = 0.0f, hi = 0.0f;
    CHECK(aoe::series_bounds(y.data(), n, &lo, &hi));
    CHECK_NEAR(hi, 999.0, 0.001);

    CHECK(aoe::nearest_index(x.data(), n, 50.0f) == 50000);
    CHECK(aoe::nearest_index(x.data(), n, -5.0f) == 0);
    CHECK(aoe::nearest_index(x.data(), n, 1e9f) == n - 1);
}

// --------------------------------------------------------------------------
// Thread pool
// --------------------------------------------------------------------------

void test_thread_pool() {
    section("thread pool");

    aoe::ThreadPool& pool = aoe::ThreadPool::global();

    // Hammer it with the shape that used to leave a dangling job pointer: a
    // count small enough that the calling thread drains every index before a
    // worker wakes up, so parallel_for returns while queue entries nobody
    // claimed still point at the (now destroyed) job. Repetition is the point
    // -- a single run almost always looks fine.
    bool all_ran = true;
    for (int round = 0; round < 4000 && all_ran; ++round) {
        const std::size_t count = 2 + static_cast<std::size_t>(round % 3);
        std::vector<std::atomic<int>> hits(count);
        for (auto& hit : hits) hit.store(0, std::memory_order_relaxed);

        pool.parallel_for(count, [&](std::size_t i) {
            hits[i].fetch_add(1, std::memory_order_relaxed);
        });

        for (std::size_t i = 0; i < count; ++i)
            if (hits[i].load(std::memory_order_relaxed) != 1) all_ran = false;
    }
    CHECK(all_ran);

    // A larger job, to be sure the work is actually distributed and complete.
    const std::size_t big = 10000;
    std::vector<int> marks(big, 0);
    pool.parallel_for(big, [&](std::size_t i) { marks[i] = 1; });
    bool complete = true;
    for (std::size_t i = 0; i < big; ++i) if (!marks[i]) complete = false;
    CHECK(complete);

    // Degenerate counts must not deadlock or touch anything.
    pool.parallel_for(0, [&](std::size_t) { CHECK(false); });
    std::size_t once = 0;
    pool.parallel_for(1, [&](std::size_t) { ++once; });
    CHECK(once == 1);
}

// --------------------------------------------------------------------------
// Header dialects
// --------------------------------------------------------------------------

void test_indexed_headers(const std::string& indexed, const std::string& plain) {
    section("indexed *PARAMETRES with units");

    aoe::Session session;
    const int file = session.add_file(indexed);
    CHECK(file == 0);
    if (file < 0) return;

    const aoe::Dataset* dataset = session.file(file);
    CHECK(dataset != nullptr);
    if (!dataset) return;

    // The whole point: 'deg' and 's' are units, not channels. Counting them as
    // channels would also shift every column index, so TIME would silently
    // return ALPHA's data -- the file would load and the plots would be wrong.
    CHECK(dataset->headers().size() == 4);
    CHECK(dataset->headers()[0] == "TIME");
    CHECK(dataset->headers()[1] == "ALPHA");
    CHECK(dataset->headers()[2] == "MACH");
    CHECK(dataset->headers()[3] == "NZ");

    CHECK(dataset->units().size() == dataset->headers().size());
    CHECK(dataset->unit(0) == "s");
    CHECK(dataset->unit(1) == "deg");
    // '' is a declared absence of unit, and must not pick up the next line's.
    CHECK(dataset->unit(2).empty());
    CHECK(dataset->unit(3).empty());
    // Out of range is empty rather than undefined behaviour.
    CHECK(dataset->unit(-1).empty());
    CHECK(dataset->unit(99).empty());

    CHECK(session.parameter_unit("TIME") == "s");
    CHECK(session.parameter_unit("MACH").empty());
    CHECK(session.parameter_unit("NOT_A_CHANNEL").empty());

    // *TABLEAU NUM SIMU: 1.2.0 -- a dotted id, not an integer.
    CHECK(dataset->simulation_count() == 3);
    CHECK(dataset->simulation(0).id == "1.2.0");
    CHECK(dataset->simulation(2).id == "3.2.0");

    // The *ENTITY block of free text must be skipped, not parsed as data.
    std::size_t count = 0;
    const float* time = session.channel(file, 0, "TIME", &count);
    CHECK(time != nullptr);
    CHECK(count == 300);
    if (time && count > 1) {
        CHECK(close(time[0], 0.0f));
        CHECK(close(time[1], 0.02f));
    }

    // Same numbers as the packed dialect, value for value.
    const int other = session.add_file(plain);
    CHECK(other == 1);
    if (other < 0) return;

    const char* names[] = {"TIME", "ALPHA", "MACH", "NZ"};
    for (const char* name : names) {
        std::size_t a = 0;
        std::size_t b = 0;
        const float* left = session.channel(file, 0, name, &a);
        const float* right = session.channel(other, 0, name, &b);
        CHECK(left != nullptr && right != nullptr);
        CHECK(a == b);
        if (!left || !right || a != b) continue;
        bool same = true;
        for (std::size_t i = 0; i < a; ++i)
            if (!close(left[i], right[i], 1e-5f)) { same = false; break; }
        CHECK(same);
    }

    // Aliases resolve to the underlying channel's unit.
    session.set_alias("Elapsed", "TIME");
    CHECK(session.parameter_unit("Elapsed") == "s");
}

}  // namespace


// --------------------------------------------------------------------------
// Regression tests for the defects listed in the post-flight review.
// --------------------------------------------------------------------------

void test_block_summary() {
    section("block summaries");

    // The summary a column carries is what the draw path asks instead of
    // rescanning: get it wrong and every autoscale is wrong.
    aoe::SampleData block;
    block.values = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
    aoe::summarise(block);
    CHECK(block.has_finite);
    CHECK(!block.has_gaps);
    CHECK(!block.monotonic);
    CHECK_NEAR(block.minimum, 1.0, 1e-6);
    CHECK_NEAR(block.maximum, 5.0, 1e-6);

    aoe::SampleData rising;
    rising.values = {0.0f, 0.0f, 1.0f, 2.5f, 2.5f, 9.0f};
    aoe::summarise(rising);
    CHECK(rising.monotonic);   // non-decreasing, repeats included
    CHECK_NEAR(rising.minimum, 0.0, 1e-6);
    CHECK_NEAR(rising.maximum, 9.0, 1e-6);

    // NaNs are skipped for the extent and ignored for the ordering, exactly as
    // series_bounds does, so the two can never disagree.
    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    aoe::SampleData gappy;
    gappy.values = {1.0f, nan_value, 2.0f, nan_value, 3.0f};
    aoe::summarise(gappy);
    CHECK(gappy.has_gaps);
    CHECK(gappy.has_finite);
    CHECK(gappy.monotonic);
    CHECK_NEAR(gappy.minimum, 1.0, 1e-6);
    CHECK_NEAR(gappy.maximum, 3.0, 1e-6);

    aoe::SampleData empty;
    empty.values = {nan_value, nan_value};
    aoe::summarise(empty);
    CHECK(!empty.has_finite);
    CHECK(!empty.monotonic);

    // And it must agree with the scanning implementations on real data.
    std::vector<float> values(5000);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = std::sin(static_cast<float>(i) * 0.01f) * 100.0f;
    aoe::SampleData real;
    real.values = values;
    aoe::summarise(real);
    float lo = 0.0f, hi = 0.0f;
    CHECK(aoe::series_bounds(values.data(), values.size(), &lo, &hi));
    CHECK_NEAR(real.minimum, lo, 1e-6);
    CHECK_NEAR(real.maximum, hi, 1e-6);
    CHECK(real.monotonic == aoe::is_non_decreasing(values.data(), values.size()));
}

void test_non_monotonic_range() {
    section("phase-plane decimation");

    // A climb and descent: x rises to a peak and comes back down, which is what
    // every phase-plane tile looks like. The binary search sees only the first
    // crossing and silently drops the descending branch.
    const std::size_t n = 2001;
    std::vector<float> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i);
        x[i] = static_cast<float>(1000.0 - std::fabs(t - 1000.0));  // 0 -> 1000 -> 0
        y[i] = static_cast<float>(t);                               // strictly rising
    }

    CHECK(!aoe::is_non_decreasing(x.data(), n));
    CHECK(aoe::is_non_decreasing(y.data(), n));

    std::vector<float> ox(8192), oy(8192);

    // Told the truth about the ordering, both branches survive: y reaches into
    // the second half of the run.
    const std::size_t honest = aoe::decimate_minmax_range(
        x.data(), y.data(), n, 400.0f, 600.0f, 600, ox.data(), oy.data(), 0);
    CHECK(honest > 0);
    float highest = 0.0f;
    for (std::size_t i = 0; i < honest; ++i)
        if (oy[i] > highest) highest = oy[i];
    CHECK(highest > 1000.0f);   // a sample from after the apex

    // Auto-detection must reach the same answer.
    const std::size_t detected = aoe::decimate_minmax_range(
        x.data(), y.data(), n, 400.0f, 600.0f, 600, ox.data(), oy.data(), -1);
    CHECK(detected == honest);

    // Statistics of the same window. Worked out here by brute force, which is
    // the definition the binary search was meant to be an optimisation of.
    double expected_min = 1e30, expected_max = -1e30;
    std::size_t expected_count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (x[i] < 400.0f || x[i] > 600.0f) continue;
        expected_min = std::fmin(expected_min, static_cast<double>(y[i]));
        expected_max = std::fmax(expected_max, static_cast<double>(y[i]));
        ++expected_count;
    }

    aoe::SeriesStats stats;
    CHECK(aoe::series_stats(x.data(), y.data(), n, 400.0f, 600.0f, &stats, 0));
    CHECK(stats.count == expected_count);
    CHECK_NEAR(stats.minimum, expected_min, 1e-6);
    CHECK_NEAR(stats.maximum, expected_max, 1e-6);

    // On genuinely sorted data the fast and slow paths must not diverge.
    aoe::SeriesStats fast, slow;
    CHECK(aoe::series_stats(y.data(), x.data(), n, 300.0f, 900.0f, &fast, 1));
    CHECK(aoe::series_stats(y.data(), x.data(), n, 300.0f, 900.0f, &slow, 0));
    CHECK(fast.count == slow.count);
    CHECK_NEAR(fast.mean, slow.mean, 1e-6);

    const std::size_t sorted_fast = aoe::decimate_minmax_range(
        y.data(), x.data(), n, 300.0f, 900.0f, 600, ox.data(), oy.data(), 1);
    std::vector<float> ox2(8192), oy2(8192);
    const std::size_t sorted_auto = aoe::decimate_minmax_range(
        y.data(), x.data(), n, 300.0f, 900.0f, 600, ox2.data(), oy2.data(), -1);
    CHECK(sorted_fast == sorted_auto);
    bool identical = true;
    for (std::size_t i = 0; i < sorted_fast; ++i)
        if (ox[i] != ox2[i] || oy[i] != oy2[i]) identical = false;
    CHECK(identical);
}

void test_pinned_blocks(const std::string& path) {
    section("pinned sample blocks");

    aoe::Session session;
    CHECK(session.add_file(path) == 0);

    std::size_t count = 0;
    aoe::SampleBlock pin;
    const float* data = session.channel(0, 0, "TIME", &count, &pin);
    CHECK(data != nullptr);
    CHECK(count > 0);
    CHECK(static_cast<bool>(pin));

    const float first = data[0];
    const float last = data[count - 1];
    CHECK(pin->monotonic);          // TIME rises
    CHECK(pin->has_finite);
    CHECK_NEAR(pin->minimum, first, 1e-6);
    CHECK_NEAR(pin->maximum, last, 1e-6);

    CHECK(session.resident_bytes() > 0);

    // Evicting everything must not disturb what the pin points at. Before the
    // blocks were shared this freed the buffer a visible curve was reading.
    session.trim_memory(0);
    CHECK(data[0] == first);
    CHECK(data[count - 1] == last);
    CHECK(pin->values.size() == count);

    // The cache dropped its reference, so the block is ours alone.
    CHECK(pin.use_count() == 1);

    // And the column comes back on demand, identical.
    std::size_t again = 0;
    const float* reread = session.channel(0, 0, "TIME", &again, nullptr);
    CHECK(reread != nullptr);
    CHECK(again == count);
    CHECK(reread[0] == first);
}

void test_file_name_reuse(const std::string& path) {
    section("file name reuse");

    aoe::Session session;
    CHECK(session.add_file(path) == 0);
    const std::string original = session.file(0)->name();

    // Removing and re-adding must give the same name back. The display name is
    // part of every run's identity key, so a suffix here quietly threw away
    // every colour and dash style the user had chosen, saved workspaces too.
    session.remove_file(0);
    CHECK(session.file_count() == 0);
    CHECK(session.add_file(path) == 0);
    CHECK(session.file(0)->name() == original);

    // A real duplicate is still distinguished.
    CHECK(session.add_file(path) == 1);
    CHECK(session.file(1)->name() != original);

    // ...and the gap a removal leaves is reused rather than skipped.
    CHECK(session.add_file(path) == 2);
    const std::string third = session.file(2)->name();
    session.remove_file(1);
    CHECK(session.add_file(path) == 2);
    bool unique = true;
    for (std::size_t i = 0; i < session.file_count(); ++i)
        for (std::size_t j = i + 1; j < session.file_count(); ++j)
            if (session.file(static_cast<int>(i))->name() ==
                session.file(static_cast<int>(j))->name())
                unique = false;
    CHECK(unique);
    CHECK(!third.empty());
}

void test_batched_first_values(const std::string& path) {
    section("batched ISO reads");

    aoe::Dataset dataset;
    CHECK(dataset.open(path));
    const int column = dataset.column_index("TIME");
    CHECK(column >= 0);

    // One mapping for every run must agree with the one-at-a-time version it
    // replaces -- that version reopened a half-gigabyte file per run.
    const std::vector<float> batched = dataset.first_values(column);
    CHECK(batched.size() == dataset.simulation_count());
    for (std::size_t s = 0; s < dataset.simulation_count(); ++s)
        CHECK_NEAR(batched[s], dataset.first_value(s, column), 1e-6);

    // A column already resident is answered from memory, same answer.
    dataset.load_columns({column}, {});
    const std::vector<float> cached = dataset.first_values(column);
    CHECK(cached.size() == batched.size());
    bool same = true;
    for (std::size_t s = 0; s < cached.size(); ++s)
        if (cached[s] != batched[s]) same = false;
    CHECK(same);

    CHECK(dataset.first_values(-1).size() == dataset.simulation_count());
    CHECK(dataset.first_values(9999).size() == dataset.simulation_count());
}

void test_truncated_file(const std::string& path) {
    section("file truncated after indexing");

    // The application is built to watch a running solver's output. Index the
    // file, let the solver truncate and rewrite it, then read a channel before
    // the three-second poll notices: the recorded offsets now point past the
    // end of the mapping.
    const std::string copy = path + ".trunc";
    {
        std::FILE* in = std::fopen(path.c_str(), "rb");
        std::FILE* out = std::fopen(copy.c_str(), "wb");
        if (!in || !out) {
            if (in) std::fclose(in);
            if (out) std::fclose(out);
            CHECK(false);
            return;
        }
        char buffer[8192];
        std::size_t read = 0;
        while ((read = std::fread(buffer, 1, sizeof(buffer), in)) > 0)
            std::fwrite(buffer, 1, read, out);
        std::fclose(in);
        std::fclose(out);
    }

    aoe::Dataset dataset;
    CHECK(dataset.open(copy));
    const int column = dataset.column_index("TIME");
    CHECK(column >= 0);

    // Truncate to a third of its length behind the dataset's back.
    std::size_t full = 0;
    {
        std::FILE* handle = std::fopen(copy.c_str(), "rb");
        std::fseek(handle, 0, SEEK_END);
        full = static_cast<std::size_t>(std::ftell(handle));
        std::fclose(handle);
    }
    CHECK(full > 0);
    {
        std::vector<char> head(full / 3);
        std::FILE* handle = std::fopen(copy.c_str(), "rb");
        const std::size_t got = std::fread(head.data(), 1, head.size(), handle);
        CHECK(got == head.size());
        std::fclose(handle);
        handle = std::fopen(copy.c_str(), "wb");
        std::fwrite(head.data(), 1, head.size(), handle);
        std::fclose(handle);
    }

    // The parse must stay inside the file rather than run off the mapping.
    // Fewer samples than before is the correct outcome; a crash is not.
    std::size_t count = 0;
    dataset.column_data(0, column, &count);
    CHECK(count <= 100000);

    std::remove(copy.c_str());
}


void test_concurrent_access(const std::string& path) {
    section("concurrent session access");

    // What ticking runs on and off actually does: a worker reopens files while
    // the GUI thread reads headers, names and cached channels. add_file grows a
    // vector of datasets and clear() destroys them, so a reader walking that
    // list mid-reallocation was reading freed memory. Without the reader/writer
    // lock at the C boundary this loop crashes or trips a sanitiser within a
    // few hundred rounds.
    aoe_session* session = aoe_session_create();
    CHECK(session != nullptr);
    if (!session) return;

    CHECK(aoe_add_file(session, path.c_str()) == 0);

    struct Shared {
        aoe_session* session;
        std::string path;
        std::atomic<bool> stop;
        // Both readers bump this. Plain int here is a data race in its own
        // right, and one a sanitiser reports on every run -- which is exactly
        // how a real race in the code under test gets lost in the noise.
        std::atomic<int> reads;
    } shared{session, path, {false}, {0}};

    auto reader = [](void* arg) {
        Shared* state = static_cast<Shared*>(arg);
        while (!state->stop) {
            const int files = aoe_file_count(state->session);
            for (int f = 0; f < files; ++f) {
                aoe_file_name(state->session, f);
                const int headers = aoe_header_count(state->session, f);
                for (int h = 0; h < headers && h < 4; ++h)
                    aoe_header(state->session, f, h);
                const int sims = aoe_sim_count(state->session, f);
                for (int i = 0; i < sims && i < 4; ++i) aoe_sim_key(state->session, f, i);
                std::size_t count = 0;
                aoe_channel(state->session, f, 0, "TIME", &count);
            }
            aoe_param_count(state->session);
            aoe_resident_bytes(state->session);
            state->reads.fetch_add(1, std::memory_order_relaxed);
        }
    };

    aoe::Thread a, b;
    CHECK(a.start(reader, &shared));
    CHECK(b.start(reader, &shared));

    for (int round = 0; round < 200; ++round) {
        aoe_add_file(session, path.c_str());
        aoe_trim_memory(session, 0);
        aoe_clear_files(session);
        aoe_add_file(session, path.c_str());
    }

    shared.stop = true;
    a.join();
    b.join();

    CHECK(shared.reads.load() > 0);
    CHECK(aoe_file_count(session) >= 1);
    aoe_session_destroy(session);
}


void test_formula_extensions(const std::string& path) {
    section("reductions and end-of-run lookup");

    aoe::Session session;
    CHECK(session.add_file(path) == 0);

    std::size_t n = 0;
    const float* nz = session.channel(0, 0, "NZ", &n);
    CHECK(nz != nullptr && n > 2);
    if (!nz || n < 2) return;

    double expected_min = nz[0], expected_max = nz[0], sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        expected_min = std::fmin(expected_min, static_cast<double>(nz[i]));
        expected_max = std::fmax(expected_max, static_cast<double>(nz[i]));
        sum += static_cast<double>(nz[i]);
    }
    const double expected_mean = sum / static_cast<double>(n);
    const float first_sample = nz[0];
    const float last_sample = nz[n - 1];

    std::string error;
    CHECK(session.set_computed("PEAK", "max(NZ)", &error));
    CHECK(session.set_computed("TROUGH", "min(NZ)", &error));
    CHECK(session.set_computed("AVERAGE", "mean(NZ)", &error));
    CHECK(session.set_computed("AT_START", "NZ[TIME=first]", &error));
    CHECK(session.set_computed("AT_END", "NZ[TIME=last]", &error));
    CHECK(session.set_computed("NORMALISED", "NZ / max(NZ)", &error));
    // The two-argument form must still be pointwise.
    CHECK(session.set_computed("PAIRWISE", "max(NZ, TIME)", &error));

    const auto whole = [&](const char* name, std::size_t* out) {
        return session.channel(0, 0, name, out);
    };

    std::size_t count = 0;
    const float* peak = whole("PEAK", &count);
    CHECK(peak != nullptr && count == n);
    if (peak) {
        // A reduction is one value held flat across the run.
        CHECK_NEAR(peak[0], expected_max, 1e-4);
        CHECK(peak[0] == peak[count - 1]);
    }

    const float* trough = whole("TROUGH", &count);
    if (trough) CHECK_NEAR(trough[0], expected_min, 1e-4);
    const float* average = whole("AVERAGE", &count);
    if (average) CHECK_NEAR(average[0], expected_mean, 1e-3);

    const float* at_start = whole("AT_START", &count);
    if (at_start) CHECK_NEAR(at_start[0], first_sample, 1e-5);
    const float* at_end = whole("AT_END", &count);
    if (at_end) CHECK_NEAR(at_end[0], last_sample, 1e-5);

    const float* normalised = whole("NORMALISED", &count);
    CHECK(normalised != nullptr && count == n);
    if (normalised && expected_max != 0.0) {
        bool ok = true;
        for (std::size_t i = 0; i < count; ++i)
            if (!close(normalised[i],
                       static_cast<float>(nz[i] / expected_max), 1e-4f))
                ok = false;
        CHECK(ok);
    }

    std::size_t time_count = 0;
    const float* time = session.channel(0, 0, "TIME", &time_count);
    const float* pairwise = whole("PAIRWISE", &count);
    CHECK(pairwise != nullptr);
    if (pairwise && time) {
        bool ok = true;
        const std::size_t limit = std::min(count, std::min(n, time_count));
        for (std::size_t i = 0; i < limit; ++i)
            if (!close(pairwise[i], std::fmax(nz[i], time[i]), 1e-5f)) ok = false;
        CHECK(ok);
    }

    // The parser must say which counts it would have accepted rather than
    // insisting on the first definition it happens to find.
    const auto rejects = [&](const char* expression) {
        std::string message;
        const bool ok = aoe::Formula::validate(
            expression,
            [&](const std::string& name) { return session.is_known_parameter(name); },
            &message);
        return !ok && !message.empty();
    };
    CHECK(rejects("max()"));
    CHECK(rejects("max(1, 2, 3)"));
    CHECK(rejects("NZ[TIME=somewhere]"));
    CHECK(rejects("mean(NZ, TIME)"));

    {
        std::string message;
        aoe::Formula::validate(
            "max(1, 2, 3)",
            [&](const std::string& name) { return session.is_known_parameter(name); },
            &message);
        CHECK(message.find("1 or 2") != std::string::npos);
    }
}

// ------------------------------------------------------------- cache
//
// The sidecar is the one part of the core that can be *wrong* rather than
// merely slow: if it is served when it should not be, the user sees numbers
// that are not in their file any more. So these tests spend most of their
// effort on the staleness check, and treat the speed as the easy part.

// Everything one Dataset knows about its samples, flattened, so the parsed and
// the cached routes can be compared in one shot rather than field by field.
struct Snapshot {
    std::vector<std::string> headers;
    std::vector<std::string> units;
    std::vector<std::string> ids;
    // [sim][column] -> values, and the summary that came with them
    std::vector<std::vector<std::vector<float>>> values;
    std::vector<std::vector<float>> minima;
    std::vector<std::vector<float>> maxima;
    std::vector<std::vector<int>> flags;  // finite | gaps<<1 | monotonic<<2
};

Snapshot snapshot_of(aoe::Dataset& d) {
    Snapshot snap;
    snap.headers = d.headers();
    snap.units = d.units();

    const std::size_t sims = d.simulation_count();
    const std::size_t cols = d.column_count();
    snap.ids.resize(sims);
    snap.values.resize(sims);
    snap.minima.resize(sims);
    snap.maxima.resize(sims);
    snap.flags.resize(sims);

    for (std::size_t s = 0; s < sims; ++s) {
        snap.ids[s] = d.simulation(s).id;
        snap.values[s].resize(cols);
        snap.minima[s].assign(cols, 0.0f);
        snap.maxima[s].assign(cols, 0.0f);
        snap.flags[s].assign(cols, 0);
        for (std::size_t c = 0; c < cols; ++c) {
            aoe::SampleBlock block = d.column_block(s, static_cast<int>(c));
            if (!block) continue;
            snap.values[s][c] = block->values;
            snap.minima[s][c] = block->minimum;
            snap.maxima[s][c] = block->maximum;
            snap.flags[s][c] = (block->has_finite ? 1 : 0) | (block->has_gaps ? 2 : 0) |
                               (block->monotonic ? 4 : 0);
        }
    }
    return snap;
}

bool snapshots_match(const Snapshot& a, const Snapshot& b, std::string* why) {
    const auto fail = [why](const std::string& message) {
        if (why) *why = message;
        return false;
    };
    if (a.headers != b.headers) return fail("headers differ");
    if (a.units != b.units) return fail("units differ");
    if (a.ids != b.ids) return fail("simulation ids differ");
    if (a.values.size() != b.values.size()) return fail("simulation count differs");

    for (std::size_t s = 0; s < a.values.size(); ++s) {
        if (a.values[s].size() != b.values[s].size())
            return fail("column count differs for run " + std::to_string(s));
        for (std::size_t c = 0; c < a.values[s].size(); ++c) {
            if (a.values[s][c].size() != b.values[s][c].size())
                return fail("sample count differs at run " + std::to_string(s) +
                            " column " + std::to_string(c));
            // Bit-for-bit. The cache stores the very floats the parser produced,
            // so anything less exact would hide a real defect.
            for (std::size_t i = 0; i < a.values[s][c].size(); ++i) {
                if (std::memcmp(&a.values[s][c][i], &b.values[s][c][i], sizeof(float)) != 0)
                    return fail("sample differs at run " + std::to_string(s) +
                                " column " + std::to_string(c) + " index " +
                                std::to_string(i));
            }
            if (a.minima[s][c] != b.minima[s][c] || a.maxima[s][c] != b.maxima[s][c])
                return fail("summary range differs at run " + std::to_string(s) +
                            " column " + std::to_string(c));
            if (a.flags[s][c] != b.flags[s][c])
                return fail("summary flags differ at run " + std::to_string(s) +
                            " column " + std::to_string(c));
        }
    }
    return true;
}

// Reads a whole file into memory. Used to put one back exactly as it was.
std::string slurp(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return std::string();
    std::string out;
    char buffer[65536];
    std::size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof(buffer), f)) > 0) out.append(buffer, got);
    std::fclose(f);
    return out;
}

bool spit(const std::string& path, const std::string& content) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(content.data(), 1, content.size(), f) == content.size();
    std::fclose(f);
    return ok;
}

std::string cache_of(const std::string& source) { return source + ".aoec"; }

void test_cache_round_trip(const std::string& path) {
    section("cache: a cached read is the parsed read");

    std::remove(cache_of(path).c_str());

    Snapshot parsed;
    {
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(!d.cache_active());  // nothing written yet
        parsed = snapshot_of(d);

        std::string written;
        CHECK(d.write_cache(&written));
        CHECK(!written.empty());
        // Writing one attaches it, so the same object is now serving from disk.
        CHECK(d.cache_active());
    }

    // A fresh object, which is what the next run of the program will be.
    aoe::Dataset cached;
    CHECK(cached.open(path));
    CHECK(cached.cache_active());

    std::string why;
    const Snapshot from_cache = snapshot_of(cached);
    const bool same = snapshots_match(parsed, from_cache, &why);
    CHECK(same);
    if (!same) std::printf("  ...%s\n", why.c_str());

    // The cheap reads have their own paths through the cache, so check them
    // separately rather than trusting that column_block() covering them.
    aoe::Dataset plain;
    std::remove(cache_of(path).c_str());
    CHECK(plain.open(path));
    CHECK(!plain.cache_active());

    for (int c = 0; c < static_cast<int>(plain.column_count()); ++c) {
        const std::vector<float> a = plain.first_values(c);
        const std::vector<float> b = cached.first_values(c);
        CHECK(a.size() == b.size());
        for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
            CHECK(std::memcmp(&a[i], &b[i], sizeof(float)) == 0);
    }
    // first_value() reads a single run and takes a different branch again.
    {
        aoe::Dataset one;
        CHECK(one.open(path));
        CHECK(one.cache_active() == false);
    }
    aoe::Dataset fresh_cached;
    // Rebuild so the single-value path has something to read.
    {
        aoe::Dataset builder;
        CHECK(builder.open(path));
        CHECK(builder.write_cache(nullptr));
    }
    CHECK(fresh_cached.open(path));
    CHECK(fresh_cached.cache_active());
    for (std::size_t s = 0; s < plain.simulation_count(); ++s) {
        for (int c = 0; c < static_cast<int>(plain.column_count()); ++c) {
            const float a = plain.first_value(s, c);
            const float b = fresh_cached.first_value(s, c);
            CHECK(std::memcmp(&a, &b, sizeof(float)) == 0);
        }
    }

    std::remove(cache_of(path).c_str());
}

void test_cache_detects_edits(const std::string& source) {
    section("cache: a file that changed is not served from cache");

    const std::string path = source + ".edit.aoe";
    const std::string original = slurp(source);
    CHECK(!original.empty());
    CHECK(spit(path, original));

    {
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(d.write_cache(nullptr));
        CHECK(d.cache_active());
    }
    CHECK(aoe::MappedFile::exists(cache_of(path)));

    // 1. Content changed, length unchanged. The modification time moves too, so
    //    this is caught twice over; the same-length case is here because it is
    //    the one a size check alone would miss.
    {
        std::string edited = original;
        const std::size_t at = edited.size() / 2;
        // Find a digit to bump, so the file stays parseable and the difference
        // is a real change of value rather than a corrupt line.
        std::size_t i = at;
        while (i < edited.size() && !(edited[i] >= '0' && edited[i] <= '8')) ++i;
        CHECK(i < edited.size());
        if (i < edited.size()) edited[i] = static_cast<char>(edited[i] + 1);
        CHECK(edited.size() == original.size());
        CHECK(spit(path, edited));

        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(!d.cache_active());  // the stale sidecar must be refused
    }

    // 2. Length changed by an append, which is how a running solver grows its
    //    output.
    CHECK(spit(path, original));
    {
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(d.write_cache(nullptr));
    }
    {
        std::string longer = original + "*TABLEAU : SIM_EXTRA\n 0.0 0.0 0.0 0.0\n*FIN\n";
        CHECK(spit(path, longer));
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(!d.cache_active());
    }

    // 3. Identical bytes, different modification time. Restoring a file from
    //    backup does this. The cache is still refused: the check is deliberately
    //    strict in the direction that only ever costs a rebuild.
    CHECK(spit(path, original));
    {
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(d.write_cache(nullptr));
        CHECK(d.cache_active());
    }
    {
        const std::uint64_t before = aoe::MappedFile::mod_time(path);
        CHECK(spit(path, original));  // same bytes, written again
        const std::uint64_t after = aoe::MappedFile::mod_time(path);

        aoe::Dataset d;
        CHECK(d.open(path));
        if (after != before) {
            CHECK(!d.cache_active());
        } else {
            // A filesystem too coarse to notice. The content hash agrees and
            // the length agrees, so serving the cache here is correct.
            CHECK(d.cache_active());
        }
    }

    // 4. The source disappearing takes the cache with it, rather than leaving a
    //    dataset that opens from a sidecar for a file that is not there.
    {
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(d.write_cache(nullptr));
    }
    std::remove(path.c_str());
    {
        aoe::Dataset d;
        CHECK(!d.open(path));
    }

    std::remove(cache_of(path).c_str());
    std::remove(path.c_str());
}

void test_cache_rejects_damage(const std::string& source) {
    section("cache: a damaged sidecar is refused, not read");

    const std::string path = source + ".damage.aoe";
    CHECK(spit(path, slurp(source)));

    const std::string cache = cache_of(path);
    const auto rebuild = [&] {
        std::remove(cache.c_str());
        aoe::Dataset d;
        return d.open(path) && d.write_cache(nullptr);
    };

    CHECK(rebuild());
    const std::string good = slurp(cache);
    CHECK(good.size() > 256);

    // A cache that opens is one whose every offset has been proved to lie
    // inside it, so each of these has to be rejected at open() rather than
    // read past the end later.
    struct Case {
        const char* what;
        std::size_t at;
        unsigned char to;
    };
    const Case cases[] = {
        {"magic", 0, 'X'},
        {"format version", 8, 0xFE},
        {"byte order mark", 12, 0xFE},
        {"recorded source length", 16, 0x7F},
        {"recorded content hash", 32, 0x7F},
        {"column count", 40, 0xFF},
        {"a section offset", 48, 0xFF},
        {"the recorded total length", 96, 0xFF},
        {"the header's own hash", 112, 0xFF},
    };

    for (const Case& c : cases) {
        std::string damaged = good;
        CHECK(c.at < damaged.size());
        damaged[c.at] = static_cast<char>(damaged[c.at] ^ c.to);
        CHECK(spit(cache, damaged));

        aoe::Dataset d;
        const bool opened = d.open(path);
        CHECK(opened);  // the source is still fine, so this must still work
        if (d.cache_active())
            std::printf("  FAIL damaged %s was accepted\n", c.what);
        CHECK(!d.cache_active());
    }

    // Truncation, which is what a full disk leaves behind. Half a header and
    // most of a file are both covered.
    for (std::size_t keep : {std::size_t{0}, std::size_t{64}, std::size_t{200},
                             good.size() / 2, good.size() - 1}) {
        CHECK(spit(cache, good.substr(0, keep)));
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(!d.cache_active());
    }

    // And the intact one still works, so the tests above are rejecting the
    // damage rather than rejecting everything.
    CHECK(spit(cache, good));
    {
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(d.cache_active());
    }

    std::remove(cache.c_str());
    std::remove(path.c_str());
}

void test_cache_large_file_fingerprint() {
    section("cache: a file too large to hash whole is still checked");

    // Over the 4 MiB threshold, so the fingerprint samples windows rather than
    // reading everything. The windows are what this exercises.
    const std::string path = "./aoe_test_big.aoe";
    CHECK(aoe_test::write_sample(path, 8, 22000, false));

    const std::string original = slurp(path);
    CHECK(original.size() > (4u << 20));

    {
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(d.write_cache(nullptr));
        CHECK(d.cache_active());
    }
    {
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(d.cache_active());
    }

    // A change in the first window: a rewritten header is the common case and
    // the first window always starts at byte zero.
    {
        std::string edited = original;
        edited[edited.find("ALPHA")] = 'B';
        CHECK(edited.size() == original.size());
        CHECK(spit(path, edited));
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(!d.cache_active());
    }

    // A change in the last window: an appended or rewritten tail. The last
    // window is anchored to the end of the file for exactly this.
    CHECK(spit(path, original));
    {
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(d.write_cache(nullptr));
    }
    {
        std::string edited = original;
        const std::size_t at = edited.size() - 2048;
        std::size_t i = at;
        while (i < edited.size() && !(edited[i] >= '0' && edited[i] <= '8')) ++i;
        CHECK(i < edited.size());
        if (i < edited.size()) edited[i] = static_cast<char>(edited[i] + 1);
        CHECK(spit(path, edited));
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(!d.cache_active());
    }

    std::remove(cache_of(path).c_str());
    std::remove(path.c_str());
}

void test_cache_through_the_session(const std::string& source) {
    section("cache: formulas and ISO agree across both routes");

    const std::string path = source + ".session.aoe";
    CHECK(spit(path, slurp(source)));
    std::remove(cache_of(path).c_str());

    std::vector<float> parsed_iso, cached_iso;
    std::vector<float> parsed_formula, cached_formula;

    for (int pass = 0; pass < 2; ++pass) {
        aoe::Session session;
        const int file = session.add_file(path);
        CHECK(file >= 0);

        aoe::Dataset* d = session.file(file);
        CHECK(d != nullptr);
        if (!d) break;
        CHECK(d->cache_active() == (pass == 1));

        std::string error;
        CHECK(session.set_computed("DRAG", "ALPHA * 2 + NZ", &error));

        std::vector<float>& iso = (pass == 0) ? parsed_iso : cached_iso;
        iso = session.iso_values(file, "MACH");

        std::vector<float>& formula = (pass == 0) ? parsed_formula : cached_formula;
        std::size_t count = 0;
        const float* values = session.channel(file, 0, "DRAG", &count);
        CHECK(values != nullptr);
        if (values) formula.assign(values, values + count);

        if (pass == 0) CHECK(d->write_cache(nullptr));
    }

    CHECK(parsed_iso == cached_iso);
    CHECK(parsed_formula.size() == cached_formula.size());
    CHECK(!parsed_formula.empty());
    for (std::size_t i = 0; i < parsed_formula.size() && i < cached_formula.size(); ++i)
        CHECK(std::memcmp(&parsed_formula[i], &cached_formula[i], sizeof(float)) == 0);

    std::remove(cache_of(path).c_str());
    std::remove(path.c_str());
}

void test_cache_survives_reload_and_threads(const std::string& source) {
    section("cache: reload re-checks it, and many threads may read it at once");

    const std::string path = source + ".reload.aoe";
    CHECK(spit(path, slurp(source)));
    std::remove(cache_of(path).c_str());

    aoe::Dataset d;
    CHECK(d.open(path));
    CHECK(d.write_cache(nullptr));
    CHECK(d.cache_active());

    // Reload with the file unchanged: the sidecar still matches, so it stays.
    CHECK(d.reload());
    CHECK(d.cache_active());

    // Every column of every run, pulled from the sidecar by the whole pool at
    // once. The mapping is shared and read-only, so this is only unsafe if the
    // load path is doing something to it that it should not be.
    const std::size_t sims = d.simulation_count();
    const std::size_t cols = d.column_count();
    std::vector<double> sums(sims * cols, 0.0);
    aoe::ThreadPool::global().parallel_for(sims * cols, [&](std::size_t i) {
        const std::size_t s = i / cols;
        const int c = static_cast<int>(i % cols);
        std::size_t n = 0;
        aoe::SampleBlock block;
        const float* v = d.column_data(s, c, &n, &block);
        double total = 0.0;
        for (std::size_t k = 0; k < n; ++k) total += v[k];
        sums[i] = total;
    });

    // The same figures read one at a time, with no cache at all.
    CHECK(d.discard_cache());
    aoe::Dataset plain;
    CHECK(plain.open(path));
    CHECK(!plain.cache_active());
    for (std::size_t s = 0; s < sims; ++s) {
        for (std::size_t c = 0; c < cols; ++c) {
            std::size_t n = 0;
            const float* v = plain.column_data(s, static_cast<int>(c), &n);
            double total = 0.0;
            for (std::size_t k = 0; k < n; ++k) total += v[k];
            CHECK(total == sums[s * cols + c]);
        }
    }

    // Now the file changes underneath a dataset that is holding a cache, and
    // reload() has to notice. This is the sequence the file watcher drives.
    {
        aoe::Dataset live;
        CHECK(live.open(path));
        CHECK(live.write_cache(nullptr));
        CHECK(live.cache_active());

        std::string longer = slurp(path);
        longer += "*TABLEAU : SIM_LATE\n 9.0 9.0 9.0 9.0\n*FIN\n";
        CHECK(spit(path, longer));

        CHECK(live.changed_on_disk());
        CHECK(live.reload());
        CHECK(!live.cache_active());  // the sidecar describes the old file
    }

    std::remove(cache_of(path).c_str());
    std::remove(path.c_str());
}

// Points the fallback cache location somewhere the test owns, rather than at
// the developer's real cache directory.
//
// SetEnvironmentVariable rather than _putenv_s on Windows, for two reasons.
// cache.cpp reads the variable with GetEnvironmentVariableW, so this writes to
// exactly the store that will be read -- the CRT keeps its own copy, and the
// two are only kept in step by the CRT's own setters. And _putenv_s needs
// __MSVCRT_VERSION__ >= 0x0800, which is not something to depend on in a test
// that has to build with whatever MinGW-w64 the target machine happens to
// carry. SetEnvironmentVariableA has been in kernel32 since NT 3.1.
bool set_environment(const char* name, const char* value) {
#if defined(_WIN32)
    return ::SetEnvironmentVariableA(name, value) != 0;
#else
    return ::setenv(name, value, 1) == 0;
#endif
}

void test_cache_falls_back_to_the_user_directory(const std::string& source) {
    section("cache: an unwritable results directory falls back to the user's");

    // Solver output routinely lives on a read-only share, which is the whole
    // reason for a second candidate location. Reproduced here without needing
    // a read-only filesystem: a *directory* sitting where the sidecar would go
    // means the rename onto it cannot succeed, so the writer moves on exactly
    // as it would if the share refused the write.
    const std::string path = source + ".fallback.aoe";
    CHECK(spit(path, slurp(source)));

    const std::string blocked = cache_of(path);
    std::remove(blocked.c_str());
#if defined(_WIN32)
    const bool made = ::_mkdir(blocked.c_str()) == 0;
#else
    const bool made = ::mkdir(blocked.c_str(), 0755) == 0;
#endif
    CHECK(made);

    const std::string fallback_dir = "./aoe_test_cachedir";
    CHECK(set_environment("AOE_CACHE_DIR", fallback_dir.c_str()));
    CHECK(aoe::user_cache_directory() == fallback_dir);

    // The second candidate is named from a hash of the source's full path, so
    // ask for it rather than guessing.
    const std::vector<std::string> candidates = aoe::cache_candidates(path);
    CHECK(candidates.size() == 2);
    if (candidates.size() == 2) {
        CHECK(candidates[0] == blocked);
        // The directory does not exist yet: begin() has to create it.
        CHECK(!aoe::MappedFile::exists(candidates[1]));

        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(d.write_cache(nullptr));
        CHECK(d.cache_active());
        CHECK(d.cache_file() == candidates[1]);
        CHECK(aoe::MappedFile::exists(candidates[1]));

        // And a fresh open finds it there, having tried and failed at the
        // first location.
        aoe::Dataset again;
        CHECK(again.open(path));
        CHECK(again.cache_active());
        CHECK(again.cache_file() == candidates[1]);

        // discard_cache() clears both locations, not just the attached one.
        CHECK(again.discard_cache());
        CHECK(!aoe::MappedFile::exists(candidates[1]));
    }

    CHECK(set_environment("AOE_CACHE_DIR", ""));
#if defined(_WIN32)
    ::_rmdir(blocked.c_str());
    ::_rmdir(fallback_dir.c_str());
#else
    ::rmdir(blocked.c_str());
    ::rmdir(fallback_dir.c_str());
#endif
    std::remove(path.c_str());
}

void test_cache_is_faster() {
    section("cache: opening and reading is materially faster");

    const std::string path = "./aoe_test_speed.aoe";
    CHECK(aoe_test::write_sample(path, 120, 2000, false));
    std::remove(cache_of(path).c_str());

    const auto read_everything = [&](aoe::Dataset& d) {
        double sum = 0.0;
        for (std::size_t s = 0; s < d.simulation_count(); ++s)
            for (int c = 0; c < static_cast<int>(d.column_count()); ++c) {
                std::size_t n = 0;
                const float* v = d.column_data(s, c, &n);
                if (v && n) sum += v[n / 2];
            }
        return sum;
    };

    double parsed_sum = 0.0, cached_sum = 0.0;
    double parsed_seconds = 0.0, cached_seconds = 0.0;

    {
        const auto t0 = std::chrono::steady_clock::now();
        aoe::Dataset d;
        CHECK(d.open(path));
        parsed_sum = read_everything(d);
        parsed_seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count();
        CHECK(d.write_cache(nullptr));
    }
    {
        const auto t0 = std::chrono::steady_clock::now();
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(d.cache_active());
        cached_sum = read_everything(d);
        cached_seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count();
    }

    CHECK(parsed_sum == cached_sum);
    std::printf("  parsed %.1f ms, cached %.1f ms  (%.1fx)\n", parsed_seconds * 1e3,
                cached_seconds * 1e3,
                cached_seconds > 0 ? parsed_seconds / cached_seconds : 0.0);

    // Deliberately a loose bound. The point of the check is to catch the cache
    // silently not being used at all -- a tight one would fail on a loaded
    // build machine and teach everyone to ignore it.
    CHECK(cached_seconds < parsed_seconds);

    std::remove(cache_of(path).c_str());
    std::remove(path.c_str());
}

void test_cache_units_and_wrapped_records(const std::string& indexed,
                                          const std::string& wrapped) {
    section("cache: units and wrapped records survive the round trip");

    // The indexed dialect is the only one that declares units, and units live
    // in the sidecar's string table rather than being re-derived.
    for (const std::string& source : {indexed, wrapped}) {
        const std::string path = source + ".rt.aoe";
        CHECK(spit(path, slurp(source)));
        std::remove(cache_of(path).c_str());

        Snapshot parsed;
        {
            aoe::Dataset d;
            CHECK(d.open(path));
            parsed = snapshot_of(d);
            CHECK(d.write_cache(nullptr));
        }
        aoe::Dataset d;
        CHECK(d.open(path));
        CHECK(d.cache_active());

        std::string why;
        const bool same = snapshots_match(parsed, snapshot_of(d), &why);
        CHECK(same);
        if (!same) std::printf("  ...%s in %s\n", why.c_str(), source.c_str());

        std::remove(cache_of(path).c_str());
        std::remove(path.c_str());
    }
}

void test_cache_discard(const std::string& source) {
    section("cache: discarding removes the sidecar and falls back to parsing");

    const std::string path = source + ".discard.aoe";
    CHECK(spit(path, slurp(source)));

    aoe::Dataset d;
    CHECK(d.open(path));
    CHECK(d.write_cache(nullptr));
    CHECK(d.cache_active());
    CHECK(aoe::MappedFile::exists(cache_of(path)));

    CHECK(d.discard_cache());
    CHECK(!d.cache_active());
    CHECK(!aoe::MappedFile::exists(cache_of(path)));

    // Still fully usable with no cache: this is the fallback every path has.
    std::size_t count = 0;
    const float* values = d.column_data(0, 0, &count);
    CHECK(values != nullptr);
    CHECK(count > 0);

    std::remove(path.c_str());
}

int main() {
    const std::string dir = ".";
    const std::string plain = dir + "/aoe_test_plain.aoe";
    const std::string wrapped = dir + "/aoe_test_wrapped.aoe";

    const std::string indexed = dir + "/aoe_test_indexed.aoe";
    const std::string indexed_plain = dir + "/aoe_test_indexed_ref.aoe";

    if (!aoe_test::write_sample(plain, 3, 500, false) ||
        !aoe_test::write_sample(wrapped, 2, 200, true) ||
        !aoe_test::write_sample_indexed(indexed, 3, 300, true) ||
        !aoe_test::write_sample(indexed_plain, 3, 300, false)) {
        std::printf("could not write test fixtures\n");
        return 2;
    }

    test_float_parser();
    test_skip_token();
    test_stats();
    test_dataset(plain);
    test_wrapped_records(wrapped);
    test_formula(plain);
    test_iso();
    test_decimation();
    test_thread_pool();
    test_indexed_headers(indexed, indexed_plain);
    test_block_summary();
    test_non_monotonic_range();
    test_pinned_blocks(plain);
    test_file_name_reuse(plain);
    test_batched_first_values(plain);
    test_truncated_file(plain);
    test_concurrent_access(plain);
    test_formula_extensions(plain);
    test_cache_round_trip(plain);
    test_cache_units_and_wrapped_records(indexed, wrapped);
    test_cache_detects_edits(plain);
    test_cache_rejects_damage(plain);
    test_cache_large_file_fingerprint();
    test_cache_through_the_session(plain);
    test_cache_discard(plain);
    test_cache_survives_reload_and_threads(plain);
    test_cache_falls_back_to_the_user_directory(plain);
    test_cache_is_faster();

    std::remove(plain.c_str());
    std::remove(wrapped.c_str());
    std::remove(indexed.c_str());
    std::remove(indexed_plain.c_str());

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
