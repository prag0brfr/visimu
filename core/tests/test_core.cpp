// Author: Paulo R. A. Gilz
// Self-contained C++ tests for the core. No framework needed, so this builds
// with nothing but the compiler that is already on the target machine.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

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
        volatile bool stop;
        int reads;
    } shared{session, path, false, 0};

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
            ++state->reads;
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

    CHECK(shared.reads > 0);
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

    std::remove(plain.c_str());
    std::remove(wrapped.c_str());
    std::remove(indexed.c_str());
    std::remove(indexed_plain.c_str());

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
