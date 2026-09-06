// Author: Paulo R. A. Gilz
#include "aoe/decimate.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace aoe {
namespace {

// First index with x[i] >= value, over non-decreasing x.
std::size_t lower_bound_x(const float* x, std::size_t count, float value) {
    std::size_t lo = 0, hi = count;
    while (lo < hi) {
        std::size_t mid = lo + (hi - lo) / 2;
        if (x[mid] < value)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

std::size_t decimate_span(const float* x, const float* y, std::size_t first,
                          std::size_t last, std::size_t max_points, float* out_x,
                          float* out_y) {
    const std::size_t count = last - first;
    if (count == 0) return 0;

    if (max_points < 2) max_points = 2;

    // Below the threshold there is nothing to gain; copy straight through.
    if (count <= max_points * 2) {
        std::memcpy(out_x, x + first, count * sizeof(float));
        std::memcpy(out_y, y + first, count * sizeof(float));
        return count;
    }

    std::size_t written = 0;
    for (std::size_t b = 0; b < max_points; ++b) {
        const std::size_t start = first + count * b / max_points;
        std::size_t stop = first + count * (b + 1) / max_points;
        if (stop <= start) continue;
        if (stop > last) stop = last;

        std::size_t min_i = start;
        std::size_t max_i = start;
        float min_v = y[start];
        float max_v = y[start];
        bool seen = !std::isnan(min_v);
        if (!seen) {
            min_v = 0.0f;
            max_v = 0.0f;
        }

        for (std::size_t i = start + 1; i < stop; ++i) {
            const float v = y[i];
            if (std::isnan(v)) continue;
            if (!seen) {
                min_v = max_v = v;
                min_i = max_i = i;
                seen = true;
                continue;
            }
            if (v < min_v) {
                min_v = v;
                min_i = i;
            } else if (v > max_v) {
                max_v = v;
                max_i = i;
            }
        }

        if (!seen) {
            // Whole bucket is NaN. Emit one NaN so the line breaks visibly
            // instead of interpolating across missing data.
            out_x[written] = x[start];
            out_y[written] = std::numeric_limits<float>::quiet_NaN();
            ++written;
            continue;
        }

        // Emit in x order so the polyline never doubles back.
        if (min_i <= max_i) {
            out_x[written] = x[min_i];
            out_y[written] = min_v;
            ++written;
            if (max_i != min_i) {
                out_x[written] = x[max_i];
                out_y[written] = max_v;
                ++written;
            }
        } else {
            out_x[written] = x[max_i];
            out_y[written] = max_v;
            ++written;
            out_x[written] = x[min_i];
            out_y[written] = min_v;
            ++written;
        }
    }
    return written;
}

}  // namespace

std::size_t decimate_minmax(const float* x, const float* y, std::size_t count,
                            std::size_t max_points, float* out_x, float* out_y) {
    if (!x || !y || !out_x || !out_y || count == 0) return 0;
    return decimate_span(x, y, 0, count, max_points, out_x, out_y);
}

bool is_non_decreasing(const float* v, std::size_t count) {
    if (!v || count < 2) return true;
    float previous = 0.0f;
    bool seen = false;
    for (std::size_t i = 0; i < count; ++i) {
        const float s = v[i];
        if (std::isnan(s) || std::isinf(s)) continue;
        if (seen && s < previous) return false;
        previous = s;
        seen = true;
    }
    return true;
}

std::size_t decimate_minmax_range(const float* x, const float* y, std::size_t count,
                                  float x_min, float x_max, std::size_t max_points,
                                  float* out_x, float* out_y, int x_sorted) {
    if (!x || !y || !out_x || !out_y || count == 0) return 0;
    if (!(x_min <= x_max)) return 0;

    if (x_sorted < 0) x_sorted = is_non_decreasing(x, count) ? 1 : 0;

    std::size_t first = 0;
    std::size_t last = 0;

    if (x_sorted) {
        first = lower_bound_x(x, count, x_min);
        last = lower_bound_x(x, count, x_max);
    } else {
        // No ordering to search, so find the outermost samples that fall inside
        // the window and keep everything between them. On a trajectory that
        // doubles back that is most of the series -- which is the right answer:
        // every part of it really is on screen at some point along the path.
        bool seen = false;
        for (std::size_t i = 0; i < count; ++i) {
            const float value = x[i];
            if (std::isnan(value) || value < x_min || value > x_max) continue;
            if (!seen) {
                first = i;
                seen = true;
            }
            last = i + 1;
        }
        if (!seen) return 0;
    }

    // Keep one sample either side so the curve is drawn to the plot border
    // rather than stopping short of it.
    if (first > 0) --first;
    if (last < count) ++last;
    if (first >= last) return 0;

    return decimate_span(x, y, first, last, max_points, out_x, out_y);
}

bool series_bounds(const float* v, std::size_t count, float* out_min, float* out_max) {
    if (!v || count == 0) return false;
    float lo = 0.0f, hi = 0.0f;
    bool seen = false;
    for (std::size_t i = 0; i < count; ++i) {
        const float s = v[i];
        if (std::isnan(s) || std::isinf(s)) continue;
        if (!seen) {
            lo = hi = s;
            seen = true;
        } else if (s < lo) {
            lo = s;
        } else if (s > hi) {
            hi = s;
        }
    }
    if (!seen) return false;
    if (out_min) *out_min = lo;
    if (out_max) *out_max = hi;
    return true;
}

std::size_t nearest_index(const float* x, std::size_t count, float target) {
    if (!x || count == 0) return 0;

    // Fast path for the sorted case, which is every time-based channel.
    if (x[0] <= x[count - 1]) {
        std::size_t i = lower_bound_x(x, count, target);
        if (i == 0) return 0;
        if (i >= count) return count - 1;
        return (target - x[i - 1] <= x[i] - target) ? i - 1 : i;
    }

    std::size_t best = 0;
    float best_diff = std::fabs(x[0] - target);
    for (std::size_t i = 1; i < count; ++i) {
        const float d = std::fabs(x[i] - target);
        if (d < best_diff) {
            best_diff = d;
            best = i;
        }
    }
    return best;
}

}  // namespace aoe

namespace aoe {

bool series_stats(const float* x, const float* y, std::size_t count, float x_min,
                  float x_max, SeriesStats* out, int x_sorted) {
    if (!y || count == 0 || !out) return false;

    const bool ranged = (x != nullptr) && (x_min < x_max);
    if (ranged && x_sorted < 0) x_sorted = is_non_decreasing(x, count) ? 1 : 0;

    // On unsorted x every sample has to be tested individually; there is no
    // contiguous span that means "in view". Statistics of a phase plane were
    // otherwise taken over whatever the binary search happened to bracket.
    const bool per_sample = ranged && !x_sorted;

    std::size_t first = 0;
    std::size_t last = count;
    if (ranged && x_sorted) {
        first = lower_bound_x(x, count, x_min);
        last = lower_bound_x(x, count, x_max);
        if (last < count) ++last;
    }
    if (first >= last) return false;

    // Welford's method: one pass, and it does not lose precision the way
    // accumulating a sum of squares does on channels with a large offset --
    // altitude in feet squared overflows float precision quickly.
    double mean = 0.0;
    double m2 = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double first_value = 0.0;
    double last_value = 0.0;
    std::size_t n = 0;

    for (std::size_t i = first; i < last; ++i) {
        if (per_sample) {
            const float xs = x[i];
            if (std::isnan(xs) || xs < x_min || xs > x_max) continue;
        }
        const float sample = y[i];
        if (std::isnan(sample) || std::isinf(sample)) continue;
        const double value = static_cast<double>(sample);

        if (n == 0) {
            minimum = maximum = first_value = value;
        } else {
            if (value < minimum) minimum = value;
            if (value > maximum) maximum = value;
        }
        last_value = value;

        ++n;
        const double delta = value - mean;
        mean += delta / static_cast<double>(n);
        m2 += delta * (value - mean);
    }

    if (n == 0) return false;

    out->minimum = minimum;
    out->maximum = maximum;
    out->mean = mean;
    out->stddev = (n > 1) ? std::sqrt(m2 / static_cast<double>(n - 1)) : 0.0;
    out->first = first_value;
    out->last = last_value;
    out->count = n;
    return true;
}

}  // namespace aoe
