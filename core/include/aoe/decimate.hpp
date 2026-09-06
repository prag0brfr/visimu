// Author: Paulo R. A. Gilz
//
// Min/max bucket decimation.
//
// This is what makes a Python front end viable. A tile 600 px wide cannot show
// more than ~600 distinct x positions, so handing the plot widget a million
// samples wastes the transfer, the allocation and the draw. We reduce each
// series to at most 2*width points before it ever crosses into Python.
//
// Min/max rather than stride: keeping every Nth sample silently deletes every
// spike between the samples it keeps, and on flight-test data that is exactly
// the part an engineer is looking for. Min/max emits both
// extremes of each bucket, so the drawn envelope is pixel-accurate -- peaks
// survive at any zoom level.
//
// Points are emitted in x order (the min and max of a bucket are swapped when
// needed) so the polyline does not zig-zag backwards.

#pragma once

#include <cstddef>

namespace aoe {

// Decimates the series (x[i], y[i]) for i in [0, count) into out_x/out_y, which
// must have room for at least 2*max_points entries (plus 2 for the endpoints).
// Returns the number of points written.
//
// When count <= max_points*2 the data is copied through unchanged.
std::size_t decimate_minmax(const float* x, const float* y, std::size_t count,
                            std::size_t max_points, float* out_x, float* out_y);

// True when the samples never decrease, skipping non-finite ones.
bool is_non_decreasing(const float* v, std::size_t count);

// As above but restricted to the samples whose x falls in [x_min, x_max].
//
// `x_sorted` says whether x is non-decreasing: 1 yes, 0 no, -1 work it out.
// When it is, the range is located by binary search, so a zoomed-in view of a
// time channel costs O(log n) rather than a scan.
//
// When it is not -- a phase plane, MACH against ALTITUDE on a climb and
// descend -- the binary search brackets only the first crossing and the whole
// descending branch of the trajectory vanishes without a trace, differently
// depending on where you have panned to. The unsorted path scans for the span
// of samples that touch the window and buckets that span *by index*, so both
// branches survive and the loop is drawn as flown.
std::size_t decimate_minmax_range(const float* x, const float* y, std::size_t count,
                                  float x_min, float x_max, std::size_t max_points,
                                  float* out_x, float* out_y, int x_sorted = -1);

// Exact bounds of a series, skipping NaNs. Returns false if every sample is NaN.
bool series_bounds(const float* v, std::size_t count, float* out_min, float* out_max);

// Index of the sample whose x is closest to `target`, for the crosshair/identify
// tools. Binary search when the data is sorted, which it is for time channels.
std::size_t nearest_index(const float* x, std::size_t count, float target);

// Statistics of y over the samples whose x falls in [x_min, x_max], or over the
// whole series when x is null or the range is empty. NaNs are skipped.
//
// Doing this in C++ rather than NumPy matters because the useful version is
// "statistics of what I am currently looking at", so it is recomputed on every
// pan and zoom of every visible tile.
struct SeriesStats {
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double first = 0.0;
    double last = 0.0;
    std::size_t count = 0;
};

bool series_stats(const float* x, const float* y, std::size_t count, float x_min,
                  float x_max, SeriesStats* out, int x_sorted = -1);

}  // namespace aoe
